"""Torch-free parsing of the nnU-Net v2 plan fields used at runtime."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from nnunet_mlx.contracts import (
    parse_boolean as _boolean,
)
from nnunet_mlx.contracts import (
    parse_mapping as _mapping,
)
from nnunet_mlx.contracts import parse_number
from nnunet_mlx.contracts import (
    parse_sequence as _sequence,
)
from nnunet_mlx.contracts import (
    parse_string as _string,
)
from nnunet_mlx.errors import ContractError

type SegmentationTieBreak = Literal["nearest", "lowest", "highest"]
type InterpolationMode = Literal["linear", "nearest", "nearest-exact"]


@dataclass(frozen=True, slots=True)
class ResamplingContract:
    """One supported upstream ``resampling_fn_*`` plan entry.

    ``order`` is the spline order for in-plane (or non-separate-axis) resizing and
    ``order_z`` the order along the separately resampled axis of anisotropic cases.
    ``force_separate_z`` is ``None`` to let the spacing-anisotropy rule decide.
    An explicit ``seg_tiebreak`` selects upstream's argmax label resampling in
    every pass. ``None`` represents an omitted plan field and retains the pinned
    threshold rule for higher-order separate-axis interpolation.

    ``resample_torch_fornnunet`` instead uses ``mode``/``aniso_axis_mode``;
    its spline orders are unused. Its omitted tie-break is ``nearest`` and its
    label scores use upstream's scale-1000 float16 quantization. Thread count and
    memory scheduling kwargs do not change the host resampler's numerical contract.
    """

    function_name: str
    order: int
    order_z: int
    force_separate_z: bool | None
    seg_tiebreak: SegmentationTieBreak | None = None
    mode: InterpolationMode = "linear"
    aniso_axis_mode: InterpolationMode = "nearest-exact"
    separate_z_anisotropy_threshold: float = 3.0

    @classmethod
    def from_configuration(
        cls,
        configuration: Mapping[str, object],
        *,
        field: str,
        expected_is_seg: bool,
    ) -> ResamplingContract:
        """Parse ``configuration[field]`` and its ``<field>_kwargs``.

        Args:
            configuration: Resolved plan configuration mapping.
            field: Entry name, e.g. ``"resampling_fn_data"``.
            expected_is_seg: Whether this entry must be declared as a segmentation
                resampler (``is_seg`` true) or an image/probability resampler.

        Raises:
            ContractError: If the function or its options are unsupported, or
                ``is_seg`` disagrees with the entry's runtime role.
        """
        function_name = _string(configuration.get(field), name=field)
        if function_name not in ("resample_data_or_seg_to_shape", "resample_torch_fornnunet"):
            raise ContractError(f"unsupported nnU-Net resampling function {function_name!r}")
        kwargs = _mapping(configuration.get(f"{field}_kwargs"), name=f"{field}_kwargs")
        torch_resampling = function_name == "resample_torch_fornnunet"
        is_seg = _boolean(
            kwargs.get("is_seg", False if torch_resampling else None), name=f"{field}.is_seg"
        )
        if is_seg is not expected_is_seg:
            raise ContractError(f"{field}.is_seg does not match its runtime role")
        force = kwargs.get("force_separate_z")
        if force is not None and not isinstance(force, bool):
            raise ContractError(f"{field}.force_separate_z must be boolean or null")
        seg_tiebreak: SegmentationTieBreak | None = None
        if "seg_tiebreak" in kwargs:
            value = kwargs["seg_tiebreak"]
            if value not in ("nearest", "lowest", "highest"):
                raise ContractError(
                    f"{field}.seg_tiebreak must be 'nearest', 'lowest', or 'highest'"
                )
            seg_tiebreak = value
        if torch_resampling:
            supported = {
                "is_seg", "force_separate_z", "seg_tiebreak", "mode", "aniso_axis_mode",
                "separate_z_anisotropy_threshold", "num_threads", "device",
                "memefficient_seg_resampling",
            }
            unknown = kwargs.keys() - supported
            if unknown:
                raise ContractError(f"{field}: unsupported resample_torch_fornnunet kwargs: "
                                    f"{', '.join(sorted(unknown))}")
            _positive_integer(kwargs.get("num_threads", 4), name=f"{field}.num_threads")
            _boolean(kwargs.get("memefficient_seg_resampling", True),
                     name=f"{field}.memefficient_seg_resampling")
            if kwargs.get("device", "cpu") != "cpu":
                raise ContractError(f"{field}.device supports only 'cpu' host resampling")
            threshold = parse_number(kwargs.get("separate_z_anisotropy_threshold", 3.0),
                                     name=f"{field}.separate_z_anisotropy_threshold")
            if threshold <= 0:
                raise ContractError(f"{field}.separate_z_anisotropy_threshold must be positive")
            return cls(
                function_name=function_name, order=1, order_z=0, force_separate_z=force,
                seg_tiebreak=seg_tiebreak or "nearest",
                mode=_interpolation_mode(kwargs.get("mode", "linear"), name=f"{field}.mode"),
                aniso_axis_mode=_interpolation_mode(
                    kwargs.get("aniso_axis_mode", "nearest-exact"), name=f"{field}.aniso_axis_mode"
                ),
                separate_z_anisotropy_threshold=threshold,
            )
        return cls(
            function_name=function_name,
            order=_nonnegative_integer(kwargs.get("order"), name=f"{field}.order"),
            order_z=_nonnegative_integer(kwargs.get("order_z"), name=f"{field}.order_z"),
            force_separate_z=force,
            seg_tiebreak=seg_tiebreak,
        )


def _interpolation_mode(value: object, *, name: str) -> InterpolationMode:
    """Require one of the supported linear or nearest interpolation modes."""
    if value not in ("linear", "nearest", "nearest-exact"):
        raise ContractError(f"{name} must be 'linear', 'nearest', or 'nearest-exact'")
    return value


@dataclass(frozen=True, slots=True)
class PlanConfiguration:
    """Validated subset of one resolved nnU-Net plans configuration.

    ``patch_size`` and ``spacing`` are in the plans' axis order, i.e. the spatial
    axes after ``PlansContract.transpose_forward`` has been applied. ``spacing`` is
    the target voxel spacing in mm; ``None`` keeps that axis's input spacing for
    each case. It has two entries for 2-D configurations (the leading axis keeps
    the input spacing). ``normalization_schemes`` and
    ``use_mask_for_norm`` have one entry per input channel. ``raw`` keeps the full
    inheritance-resolved mapping, including fields this class does not parse
    (e.g. ``architecture``).
    """

    name: str
    raw: dict[str, object]
    data_identifier: str
    batch_size: int
    batch_dice: bool
    patch_size: tuple[int, ...]
    spacing: tuple[float | None, ...]
    normalization_schemes: tuple[str, ...]
    use_mask_for_norm: tuple[bool, ...]
    data_resampling: ResamplingContract
    segmentation_resampling: ResamplingContract
    probability_resampling: ResamplingContract
    previous_stage_name: str | None

    @classmethod
    def from_mapping(
        cls, name: str, configuration: Mapping[str, object]
    ) -> PlanConfiguration:
        """Validate an inheritance-resolved configuration mapping.

        Raises:
            ContractError: If the preprocessor is not ``DefaultPreprocessor``, ranks
                or per-channel lists disagree, or a required field is malformed.
        """
        raw = dict(configuration)
        if raw.get("preprocessor_name", "DefaultPreprocessor") != "DefaultPreprocessor":
            raise ContractError("unsupported nnU-Net preprocessor")
        patch_size = _positive_integer_tuple(raw.get("patch_size"), name="patch_size")
        spacing = _plan_spacing_tuple(raw.get("spacing"), name="spacing")
        if len(patch_size) not in (2, 3) or len(spacing) != len(patch_size):
            raise ContractError("plan patch_size and spacing must have matching 2-D/3-D rank")
        schemes = _string_tuple(
            raw.get("normalization_schemes"), name="normalization_schemes"
        )
        masks = _boolean_tuple(raw.get("use_mask_for_norm"), name="use_mask_for_norm")
        if not schemes or len(schemes) != len(masks):
            raise ContractError("normalization schemes and mask flags must match")
        previous = raw.get("previous_stage")
        if previous is not None and not isinstance(previous, str):
            raise ContractError("previous_stage must be a string or null")
        return cls(
            name=name,
            raw=raw,
            data_identifier=_string(
                raw.get("data_identifier"), name="data_identifier"
            ),
            batch_size=_positive_integer(raw.get("batch_size"), name="batch_size"),
            batch_dice=_boolean(raw.get("batch_dice"), name="batch_dice"),
            patch_size=patch_size,
            spacing=spacing,
            normalization_schemes=schemes,
            use_mask_for_norm=masks,
            data_resampling=ResamplingContract.from_configuration(
                raw,
                field="resampling_fn_data",
                expected_is_seg=False,
            ),
            segmentation_resampling=ResamplingContract.from_configuration(
                raw,
                field="resampling_fn_seg",
                expected_is_seg=True,
            ),
            probability_resampling=ResamplingContract.from_configuration(
                raw,
                field="resampling_fn_probabilities",
                expected_is_seg=False,
            ),
            previous_stage_name=previous,
        )


@dataclass(frozen=True, slots=True)
class PlansContract:
    """Parsed ``plans.json`` for one selected configuration.

    ``transpose_forward`` permutes the three spatial axes of a raw (C, X, Y, Z)
    case into the plans' axis order; ``transpose_backward`` is its inverse and is
    checked as such. ``intensity_properties`` maps a channel index (as a string) to
    the foreground intensity statistics recorded by nnU-Net's planner; only the CT
    normalization scheme reads them.
    """

    raw: dict[str, object]
    configuration: PlanConfiguration
    transpose_forward: tuple[int, int, int]
    transpose_backward: tuple[int, int, int]
    image_reader_writer: str
    intensity_properties: dict[str, dict[str, object]]

    def __post_init__(self) -> None:
        # Fail at plan load, not mid-inference, if CT statistics are unusable.
        """Validate CT normalization statistics before inference."""
        for channel, scheme in enumerate(self.configuration.normalization_schemes):
            if scheme != "CTNormalization":
                continue
            statistics = self.intensity_properties.get(str(channel), {})
            values = {
                key: parse_number(statistics.get(key), name=f"CT {key}")
                for key in ("mean", "std", "percentile_00_5", "percentile_99_5")
            }
            if values["std"] <= 0:
                raise ContractError("CT std must be positive")
            if values["percentile_00_5"] > values["percentile_99_5"]:
                raise ContractError("CT intensity percentiles are reversed")

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, object], *, configuration: str
    ) -> PlansContract:
        """Parse a plans payload and select ``configuration``.

        Raises:
            ContractError: If the configuration is missing or circularly inherited,
                the transposes are not mutually inverse permutations of ``[0, 1, 2]``,
                the intensity-statistics key is unsupported, or any field is malformed.
        """
        raw = dict(payload)
        resolved = resolve_configuration(raw, configuration)
        forward = _axis_permutation(raw.get("transpose_forward"), name="transpose_forward")
        backward = _axis_permutation(
            raw.get("transpose_backward"), name="transpose_backward"
        )
        expected_backward = tuple(forward.index(axis) for axis in range(3))
        if backward != expected_backward:
            raise ContractError("transpose_backward is not the inverse of transpose_forward")
        intensity = _intensity_properties(raw)
        return cls(
            raw=raw,
            configuration=PlanConfiguration.from_mapping(configuration, resolved),
            transpose_forward=forward,
            transpose_backward=backward,
            image_reader_writer=_string(
                raw.get("image_reader_writer"), name="image_reader_writer"
            ),
            intensity_properties=intensity,
        )

    def topology_payload(self) -> dict[str, object]:
        """Return a deep copy of the plans with the resolved configuration substituted.

        Network-topology parsers read ``configurations[name]`` directly and do not
        follow ``inherits_from``, so the inheritance-merged mapping replaces it.
        """
        payload = copy.deepcopy(self.raw)
        configurations = _mapping(payload.get("configurations"), name="configurations")
        configurations[self.configuration.name] = copy.deepcopy(self.configuration.raw)
        payload["configurations"] = configurations
        return payload


def read_json_object(path: str | Path) -> dict[str, object]:
    """Read a UTF-8 JSON file whose top level must be an object with string keys.

    Raises:
        ContractError: If the file is missing, is not valid JSON, or is not an object.
    """
    source = Path(path)
    if not source.is_file():
        raise ContractError(f"JSON file does not exist: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"invalid JSON file: {source}") from error
    return _mapping(payload, name=source.name)


def resolve_configuration(
    plans: Mapping[str, object], configuration: str
) -> dict[str, object]:
    """Return a configuration with ``inherits_from`` parents merged in.

    Keys of a child override its parent's, as in upstream's ``ConfigurationManager``
    plan loading; the result keeps the ``inherits_from`` key of the child.

    Raises:
        ContractError: If the configuration or a parent is missing, or inheritance
            is circular.
    """
    configurations = _mapping(plans.get("configurations"), name="configurations")
    return _resolve_configuration(configurations, configuration, visited=())


def _resolve_configuration(
    configurations: Mapping[str, object],
    name: str,
    *,
    visited: tuple[str, ...],
) -> dict[str, object]:
    """Resolve ``inherits_from`` chains; child keys override the parent's. Cycles raise."""
    if name in visited:
        raise ContractError(f"circular configuration inheritance: {(*visited, name)}")
    raw = configurations.get(name)
    if raw is None:
        raise ContractError(f"nnU-Net plans lack configuration {name!r}")
    current = _mapping(raw, name=f"configuration {name}")
    parent = current.get("inherits_from")
    if parent is None:
        return current
    parent_name = _string(parent, name=f"configuration {name}.inherits_from")
    base = _resolve_configuration(
        configurations, parent_name, visited=(*visited, name)
    )
    base.update(current)
    return base


def _intensity_properties(raw: Mapping[str, object]) -> dict[str, dict[str, object]]:
    """Return per-channel foreground intensity statistics keyed by channel index string.

    The legacy ``..._by_modality`` key is rejected; absent statistics give ``{}``.
    """
    if "foreground_intensity_properties_per_channel" not in raw:
        if "foreground_intensity_properties_by_modality" in raw:
            raise ContractError(
                "plans must report intensity statistics through "
                "foreground_intensity_properties_per_channel; "
                "foreground_intensity_properties_by_modality is not supported"
            )
        return {}
    channels = _mapping(
        raw["foreground_intensity_properties_per_channel"],
        name="foreground_intensity_properties_per_channel",
    )
    return {
        key: _mapping(value, name=f"intensity properties {key}")
        for key, value in channels.items()
    }


def _string_tuple(value: object, *, name: str) -> tuple[str, ...]:
    """Parse a sequence of string values as a tuple."""
    return tuple(_string(item, name=name) for item in _sequence(value, name=name))


def _boolean_tuple(value: object, *, name: str) -> tuple[bool, ...]:
    """Parse a sequence of boolean values as a tuple."""
    return tuple(_boolean(item, name=name) for item in _sequence(value, name=name))


def _nonnegative_integer(value: object, *, name: str) -> int:
    """Require a nonnegative integer, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{name} must be a non-negative integer")
    return value


def _positive_integer(value: object, *, name: str) -> int:
    """Require a positive integer."""
    result = _nonnegative_integer(value, name=name)
    if result < 1:
        raise ContractError(f"{name} must be positive")
    return result


def _positive_integer_tuple(value: object, *, name: str) -> tuple[int, ...]:
    """Parse a sequence of positive integers."""
    result = tuple(
        _nonnegative_integer(item, name=name) for item in _sequence(value, name=name)
    )
    if any(item < 1 for item in result):
        raise ContractError(f"{name} must contain positive integers")
    return result


def _plan_spacing_tuple(value: object, *, name: str) -> tuple[float | None, ...]:
    """Keep native-axis markers unresolved until an individual image is available."""
    result: list[float | None] = []
    for item in _sequence(value, name=name):
        if item is None:
            result.append(None)
            continue
        number = parse_number(item, name=name)
        if number <= 0:
            raise ContractError(f"{name} must contain positive numbers")
        result.append(number)
    return tuple(result)


def _axis_permutation(value: object, *, name: str) -> tuple[int, int, int]:
    """Validate an axis-order permutation for the plans contract."""
    axes = tuple(
        _nonnegative_integer(item, name=name) for item in _sequence(value, name=name)
    )
    if sorted(axes) != [0, 1, 2]:
        raise ContractError(f"{name} must be a permutation of [0, 1, 2]")
    return cast(tuple[int, int, int], axes)
