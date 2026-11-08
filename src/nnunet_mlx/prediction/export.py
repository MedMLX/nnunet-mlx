"""nnU-Net probability restoration and geometry-aware prediction export."""

from __future__ import annotations

import json
import pickle
import platform
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt

from nnunet_mlx import __version__
from nnunet_mlx.contracts import parse_number
from nnunet_mlx.data.labels import LabelSemantics
from nnunet_mlx.data.resampling import resample_probabilities_or_data
from nnunet_mlx.errors import ContractError
from nnunet_mlx.kernels.metal_restore import (
    resample_class_logits_argmax_optimized,
    resample_device_logits_argmax,
)
from nnunet_mlx.plans import PlansContract
from nnunet_mlx.prediction.inference import logits_to_probabilities, probabilities_to_segmentation
from nnunet_mlx.storage.imageio import (
    image_reader_output_sidecar,
    read_segmentation,
    write_segmentation,
)

if TYPE_CHECKING:
    from mlx.core import array

type FloatArray = npt.NDArray[np.float32]

# Bump only when prediction, export, or continuation equivalence changes.
PREDICTION_CONTRACT_VERSION = 4
COMPLETION_RECEIPT_SCHEMA_VERSION = 3


@dataclass(frozen=True, slots=True)
class RestoredPrediction:
    """Prediction in the selected reader's array grid, before file writing.

    ``segmentation`` is ``(D, H, W)`` uint16 labels (a singleton leading axis for
    2-D images); ``probabilities`` is ``(C, D, H, W)`` float32 or ``None``. For
    ``NibabelIOWithReorient`` this is the reoriented reader grid; the segmentation
    writer restores the source orientation, while saved probabilities retain the
    reader grid and its geometry in the properties sidecar.
    """

    segmentation: npt.NDArray[np.uint16]
    probabilities: FloatArray | None


@dataclass(frozen=True, slots=True)
class ExportedFiles:
    """Paths of one prediction's outputs; a path is ``None`` when that file is not written.

    ``image_properties`` is the geometry sidecar some writers emit (TIFF);
    ``probabilities`` (``.npz``) and ``properties`` (``.pkl``) exist only when
    probabilities are saved.
    """

    segmentation: Path
    image_properties: Path | None
    probabilities: Path | None
    properties: Path | None

    @property
    def paths(self) -> tuple[Path, ...]:
        """Return all nonempty paths belonging to the prediction output."""
        return tuple(
            path
            for path in (
                self.segmentation,
                self.image_properties,
                self.probabilities,
                self.properties,
            )
            if path is not None
        )

    def to_payload(self) -> dict[str, object]:
        """Serialize output paths, retaining None for omitted artifacts."""
        return {
            "segmentation": str(self.segmentation),
            "image_properties": (
                None if self.image_properties is None else str(self.image_properties)
            ),
            "probabilities": (
                None if self.probabilities is None else str(self.probabilities)
            ),
            "properties": None if self.properties is None else str(self.properties),
        }


def prediction_output_files(
    output_prefix: str | Path,
    *,
    file_ending: str,
    image_reader_writer: str,
    save_probabilities: bool,
) -> ExportedFiles:
    """Return the complete output-file contract for one raw prediction."""

    prefix = Path(output_prefix)
    if not file_ending.startswith("."):
        raise ContractError("dataset file ending must start with a dot")
    segmentation = Path(f"{prefix}{file_ending}")
    return ExportedFiles(
        segmentation=segmentation,
        image_properties=image_reader_output_sidecar(
            image_reader_writer, segmentation
        ),
        probabilities=Path(f"{prefix}.npz") if save_probabilities else None,
        properties=Path(f"{prefix}.pkl") if save_probabilities else None,
    )


def restore_prediction(
    logits: npt.NDArray[np.generic],
    properties: Mapping[str, object],
    plans: PlansContract,
    semantics: LabelSemantics,
    *,
    return_probabilities: bool,
) -> RestoredPrediction:
    """Reverse nnU-Net preprocessing: resample, uncrop, then transpose.

    Args:
        logits: Finite ``(C, S0, S1, S2)`` float logits at the plans' spacing and axis
            order, with ``C == semantics.output_channels``.
        properties: Preprocessing properties; needs ``spacing`` (mm, raw axis order),
            ``shape_before_cropping``, ``bbox_used_for_cropping`` and
            ``shape_after_cropping_and_before_resampling``.
        plans: Parsed plans supplying the transposes, target spacing and the
            probability-resampling settings.
        semantics: Dataset label semantics (class or region based).
        return_probabilities: Also return softmax (class) or sigmoid (region)
            probabilities on the reader's grid. For class models the uncropped
            border is background (channel 0 = 1); for region models it is 0.
            For class models, this option also selects argmax after float32
            softmax rather than argmax of logits; rounding at near ties can
            therefore change the segmentation, matching the upstream paths.

    Raises:
        ContractError: If the logits or properties violate the contract.
    """

    values = np.asarray(logits)
    if values.ndim != 4 or int(values.shape[0]) != semantics.output_channels:
        raise ContractError("prediction logits do not match the nnU-Net output contract")
    if not np.issubdtype(values.dtype, np.floating):
        raise ContractError("prediction logits must be finite floating-point values")
    shape_before = _shape(
        properties.get("shape_before_cropping"), name="shape_before_cropping"
    )
    crop_shape = _shape(
        properties.get("shape_after_cropping_and_before_resampling"),
        name="shape_after_cropping_and_before_resampling",
    )
    bounding_box = _bounding_box(properties.get("bbox_used_for_cropping"), shape_before)

    if not return_probabilities and not semantics.has_regions:
        probabilities = None
        cropped_segmentation = _resample_class_logits_to_segmentation(
            values,
            properties,
            plans,
            crop_shape,
        )
    else:
        if not np.isfinite(values).all():
            raise ContractError("prediction logits must be finite floating-point values")
        restored_values = _resample_logits_to_crop(values, properties, plans, crop_shape)
        probabilities = logits_to_probabilities(
            restored_values, has_regions=semantics.has_regions
        )
        cropped_segmentation = probabilities_to_segmentation(
            probabilities,
            regions_class_order=semantics.regions_class_order,
        )

    segmentation = _uncrop_segmentation(cropped_segmentation, shape_before, bounding_box, plans)
    crop_slices = tuple(slice(start, stop) for start, stop in bounding_box)
    restored_probabilities = None
    if return_probabilities:
        if probabilities is None:
            raise AssertionError("probability restoration unexpectedly lacks probabilities")
        uncropped = np.zeros(
            (semantics.output_channels, *shape_before), dtype=np.float32
        )
        if not semantics.has_regions:
            uncropped[0] = 1
        uncropped[(slice(None), *crop_slices)] = probabilities
        probability_transpose = (
            0,
            *(axis + 1 for axis in plans.transpose_backward),
        )
        restored_probabilities = np.ascontiguousarray(
            uncropped.transpose(probability_transpose), dtype=np.float32
        )
    return RestoredPrediction(
        segmentation=segmentation,
        probabilities=restored_probabilities,
    )


def restore_device_segmentation(
    logits: array,
    properties: Mapping[str, object],
    plans: PlansContract,
    semantics: LabelSemantics,
) -> RestoredPrediction | None:
    """Restore class labels from Metal-resident channels-last ``(*spatial,C)`` logits.

    Matches ``restore_prediction(..., return_probabilities=False)`` for class
    models: logits are resampled and reduced on Metal and only uint16 labels
    cross to the host before uncropping and inverse transposition. Returns
    ``None`` when the fused Metal restoration does not support the case.
    """
    if (
        semantics.has_regions
        or logits.ndim != 4
        or int(logits.shape[-1]) != semantics.output_channels
    ):
        raise ContractError("device restoration requires class logits matching the output contract")
    shape_before = _shape(properties.get("shape_before_cropping"), name="shape_before_cropping")
    crop_shape = _shape(
        properties.get("shape_after_cropping_and_before_resampling"),
        name="shape_after_cropping_and_before_resampling",
    )
    bounding_box = _bounding_box(properties.get("bbox_used_for_cropping"), shape_before)
    current_spacing, spacing_transposed = _prediction_resampling_spacings(
        properties, plans, crop_shape
    )
    cropped = resample_device_logits_argmax(
        logits, crop_shape, current_spacing, spacing_transposed,
        plans.configuration.probability_resampling,
    )
    if cropped is None:
        return None
    return RestoredPrediction(
        segmentation=_uncrop_segmentation(cropped, shape_before, bounding_box, plans),
        probabilities=None,
    )


def _uncrop_segmentation(
    cropped: npt.NDArray[np.uint16],
    shape_before: tuple[int, int, int],
    bounding_box: Sequence[tuple[int, int]],
    plans: PlansContract,
) -> npt.NDArray[np.uint16]:
    """Restore a cropped label map to the original shape and axis order."""
    _validate_crop_shape(cropped.shape, bounding_box)
    segmentation = np.zeros(shape_before, dtype=np.uint16)
    segmentation[tuple(slice(start, stop) for start, stop in bounding_box)] = cropped
    return np.ascontiguousarray(segmentation.transpose(plans.transpose_backward), dtype=np.uint16)


def _resample_logits_to_crop(
    values: npt.NDArray[np.generic],
    properties: Mapping[str, object],
    plans: PlansContract,
    crop_shape: tuple[int, int, int],
) -> FloatArray:
    """Resample probabilities/logits from the plans' spacing to the cropped image grid."""
    current_spacing, spacing_transposed = _prediction_resampling_spacings(
        properties,
        plans,
        crop_shape,
    )
    return resample_probabilities_or_data(
        values,
        crop_shape,
        current_spacing,
        spacing_transposed,
        plans.configuration.probability_resampling,
    )


def _resample_class_logits_to_segmentation(
    values: npt.NDArray[np.generic],
    properties: Mapping[str, object],
    plans: PlansContract,
    crop_shape: tuple[int, int, int],
) -> npt.NDArray[np.uint16]:
    """Resample logits to the cropped grid and return the argmax labels directly."""
    current_spacing, spacing_transposed = _prediction_resampling_spacings(
        properties,
        plans,
        crop_shape,
    )
    return resample_class_logits_argmax_optimized(
        values,
        crop_shape,
        current_spacing,
        spacing_transposed,
        plans.configuration.probability_resampling,
    )


def _prediction_resampling_spacings(
    properties: Mapping[str, object],
    plans: PlansContract,
    crop_shape: tuple[int, int, int],
) -> tuple[tuple[float, ...], tuple[float, float, float]]:
    # Logits live at the plans' spacing and go back to the image's own spacing, both
    # in the plans' axis order. Null entries and the omitted leading axis in 2-D
    # plans take this image's spacing, matching preprocessing's per-case resolution.
    """Resolve plans and original-grid spacings for prediction restoration."""
    spacing = _positive_float_tuple(properties.get("spacing"), name="spacing")
    if len(spacing) != 3:
        raise ContractError("prediction properties spacing must contain three values")
    spacing_transposed = cast(
        tuple[float, float, float],
        tuple(spacing[axis] for axis in plans.transpose_forward),
    )
    configuration_spacing = plans.configuration.spacing
    configured_spacing = (
        configuration_spacing
        if len(configuration_spacing) == len(crop_shape)
        else (spacing_transposed[0], *configuration_spacing)
    )
    current_spacing = tuple(
        original if configured is None else configured
        for configured, original in zip(configured_spacing, spacing_transposed, strict=True)
    )
    return current_spacing, spacing_transposed


def export_prediction(
    prediction: RestoredPrediction,
    properties: Mapping[str, object],
    output_prefix: str | Path,
    *,
    file_ending: str,
    image_reader_writer: str,
    overwrite: bool,
    completion_identity: Mapping[str, object] | None = None,
) -> ExportedFiles:
    """Write a prediction and its completion receipt using staged outputs.

    Outputs are first written and read back in a temporary directory beside the
    target; they replace the destination files only after that check passes. The
    previous receipt is removed before any existing output is replaced and the new
    one is written last, so ``prediction_is_complete`` is true only for a whole,
    matching generation. Replacement is per file, not an atomic group; an interruption
    can leave mixed files without a valid receipt. Outputs of an earlier run that
    this call does not write (for example stale probability files) are deleted.

    Args:
        prediction: Restored labels and optional probabilities.
        properties: Image geometry properties from reading the input (written to
            the ``.pkl`` sidecar when probabilities are saved).
        output_prefix: Output path without the file ending.
        file_ending: Dataset file ending including the dot, e.g. ``".nii.gz"``.
        image_reader_writer: Reader/writer class name from the plans.
        overwrite: Replace existing outputs instead of raising.
        completion_identity: JSON-native identity recorded in the receipt;
            ``inputs``, ``labels`` and ``regions_class_order`` are also copied into
            the ``.pkl`` sidecar.

    Raises:
        ContractError: If outputs exist and ``overwrite`` is false, the identity is
            not JSON-native, or a staged file does not read back as written.
    """
    files = prediction_output_files(
        output_prefix, file_ending=file_ending, image_reader_writer=image_reader_writer,
        save_probabilities=prediction.probabilities is not None,
    )
    require_prediction_destination(files, overwrite=overwrite)
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    obsolete = _obsolete_prediction_outputs(
        prefix, files, file_ending=file_ending, image_reader_writer=image_reader_writer
    )
    with tempfile.TemporaryDirectory(prefix=f".{prefix.name}.", dir=prefix.parent) as staging:
        staged = prediction_output_files(
            Path(staging) / prefix.name, file_ending=file_ending,
            image_reader_writer=image_reader_writer,
            save_probabilities=prediction.probabilities is not None,
        )
        write_segmentation(
            image_reader_writer, prediction.segmentation, staged.segmentation,
            properties, overwrite=False,
        )
        if (staged.probabilities is not None and staged.properties is not None
                and prediction.probabilities is not None):
            np.savez_compressed(staged.probabilities, probabilities=prediction.probabilities)
            sidecar_properties = dict(properties)
            for key in ("inputs", "labels", "regions_class_order"):
                if completion_identity is not None and key in completion_identity:
                    sidecar_properties[key] = completion_identity[key]
            with staged.properties.open("wb") as handle:
                pickle.dump(sidecar_properties, handle)
        _validate_staged_prediction(staged, prediction, image_reader_writer, properties)
        receipt = {
            "schema_version": COMPLETION_RECEIPT_SCHEMA_VERSION,
            "prediction_contract_version": PREDICTION_CONTRACT_VERSION,
            "execution_provenance": _execution_provenance(),
            "identity": canonical_prediction_identity(completion_identity or {}),
            "outputs": {path.name: file_signature(path) for path in staged.paths},
        }
        staged_receipt = Path(staging) / "completion.json"
        staged_receipt.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
        # Invalidate the previous receipt before mutating any committed output, so an
        # interrupted overwrite can never leave a mixed generation marked complete.
        completion_receipt_path(prefix).unlink(missing_ok=True)
        for source, destination in zip(staged.paths, files.paths, strict=True):
            source.replace(destination)
        for path in obsolete:
            path.unlink(missing_ok=True)
        staged_receipt.replace(completion_receipt_path(prefix))
    return files


def _obsolete_prediction_outputs(
    output_prefix: Path,
    files: ExportedFiles,
    *,
    file_ending: str,
    image_reader_writer: str,
) -> tuple[Path, ...]:
    # The full owned generation includes the optional probability artifacts, so
    # switching to label-only output cannot leave stale siblings behind.
    """Identify stale artifacts belonging to an earlier prediction generation."""
    generation = prediction_output_files(
        output_prefix, file_ending=file_ending, image_reader_writer=image_reader_writer,
        save_probabilities=True,
    )
    requested = set(files.paths)
    return tuple(path for path in generation.paths if path not in requested)


def require_prediction_destination(files: ExportedFiles, *, overwrite: bool) -> None:
    """Reject existing output paths unless overwrite is enabled."""
    existing = [path for path in files.paths if path.exists()]
    if existing and not overwrite:
        raise ContractError(f"prediction outputs already exist: {existing}")


def canonical_prediction_identity(identity: Mapping[str, object]) -> dict[str, object]:
    """Return the JSON-native identity stored in completion receipts."""

    try:
        return cast(
            dict[str, object],
            json.loads(json.dumps(dict(identity), allow_nan=False)),
        )
    except (TypeError, ValueError) as error:
        raise ContractError(f"prediction identity must be JSON-native: {error}") from error


def completion_receipt_path(output_prefix: str | Path) -> Path:
    """Return ``<output_prefix>.complete.json``, the marker written after all outputs."""
    return Path(f"{output_prefix}.complete.json")


def prediction_is_complete(
    output_prefix: str | Path, files: ExportedFiles, identity: Mapping[str, object],
) -> bool:
    """Return whether a valid receipt matches ``identity`` and the current output files.

    Files are compared by size and modification time, not content. Any unreadable or
    mismatched receipt yields ``False`` rather than an error.
    """
    receipt_path = completion_receipt_path(output_prefix)
    if not receipt_path.is_file() or not all(path.is_file() for path in files.paths):
        return False
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(receipt, dict):
        return False
    receipt = cast(dict[str, object], receipt)
    return (
        type(receipt.get("schema_version")) is int
        and receipt.get("schema_version") == COMPLETION_RECEIPT_SCHEMA_VERSION
        and type(receipt.get("prediction_contract_version")) is int
        and receipt.get("prediction_contract_version") == PREDICTION_CONTRACT_VERSION
        and isinstance(receipt.get("execution_provenance"), dict)
        and receipt.get("identity") == canonical_prediction_identity(identity)
        and receipt.get("outputs") == {path.name: file_signature(path) for path in files.paths}
    )


def file_signature(path: Path) -> list[int]:
    """Return ``[size_bytes, mtime_ns]``; a change detector, not a content digest."""
    metadata = path.stat()
    return [metadata.st_size, metadata.st_mtime_ns]


def _execution_provenance() -> dict[str, str]:
    # Audit metadata does not invalidate semantically equivalent predictions.
    """Record package, MLX and Python versions for the prediction receipt."""
    return {
        "package_version": __version__,
        "mlx_distribution_version": version("mlx"),
        "python_version": platform.python_version(),
    }


def _validate_staged_prediction(
    files: ExportedFiles, prediction: RestoredPrediction, reader: str,
    properties: Mapping[str, object],
) -> None:
    """Read staged outputs back and require an exact match with ``prediction``."""
    if files.image_properties is not None and not files.image_properties.is_file():
        raise ContractError("staged segmentation geometry sidecar is missing")
    data = read_segmentation(reader, files.segmentation, properties)
    restored = np.asarray(data[0])
    expected = prediction.segmentation
    if not np.array_equal(restored, expected):
        raise ContractError("staged segmentation does not match prediction")
    if files.probabilities is not None and prediction.probabilities is not None:
        with np.load(files.probabilities) as payload:
            if not np.array_equal(payload["probabilities"], prediction.probabilities):
                raise ContractError("staged probabilities do not match prediction")


def _shape(value: object, *, name: str) -> tuple[int, int, int]:
    """Parse a positive three-dimensional output shape."""
    if not isinstance(value, list | tuple):
        raise ContractError(f"{name} must be an array")
    result: list[int] = []
    for item in cast(Sequence[object], value):
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise ContractError(f"{name} must contain positive integers")
        result.append(item)
    if len(result) != 3:
        raise ContractError(f"{name} must contain three dimensions")
    return cast(tuple[int, int, int], tuple(result))


def _bounding_box(
    value: object, shape: tuple[int, int, int]
) -> tuple[tuple[int, int], ...]:
    """Validate ``bbox_used_for_cropping`` as three half-open index pairs inside ``shape``."""
    if not isinstance(value, list | tuple):
        raise ContractError("bbox_used_for_cropping must contain three pairs")
    pairs = cast(Sequence[object], value)
    if len(pairs) != 3:
        raise ContractError("bbox_used_for_cropping must contain three pairs")
    result: list[tuple[int, int]] = []
    for axis, raw_pair in enumerate(pairs):
        if not isinstance(raw_pair, list | tuple):
            raise ContractError("bbox_used_for_cropping must contain three pairs")
        pair = cast(Sequence[object], raw_pair)
        if len(pair) != 2:
            raise ContractError("bbox_used_for_cropping must contain three pairs")
        start, stop = pair
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(stop, bool)
            or not isinstance(stop, int)
            or not 0 <= start < stop <= shape[axis]
        ):
            raise ContractError("bbox_used_for_cropping is outside the original shape")
        result.append((start, stop))
    return tuple(result)


def _validate_crop_shape(
    crop_shape: Sequence[int], bounding_box: Sequence[tuple[int, int]]
) -> None:
    """Require a restored crop to match its bounding-box extents."""
    expected = tuple(stop - start for start, stop in bounding_box)
    if tuple(crop_shape) != expected:
        raise ContractError(
            f"restored crop shape {tuple(crop_shape)} does not match bbox {expected}"
        )


def _positive_float_tuple(value: object, *, name: str) -> tuple[float, ...]:
    """Parse a sequence of positive finite numeric values."""
    if not isinstance(value, list | tuple):
        raise ContractError(f"{name} must be an array")
    result: list[float] = []
    for item in cast(Sequence[object], value):
        if isinstance(item, bool) or not isinstance(item, int | float) or item <= 0:
            raise ContractError(f"{name} must contain positive numbers")
        result.append(parse_number(item, name=name))
    return tuple(result)
