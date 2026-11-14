"""Convert trained upstream nnU-Net v2 model folders into native MLX model folders.

An upstream trained model folder is what ``nnUNetv2_train`` leaves in
``nnUNet_results/<dataset>/<trainer>__<plans>__<configuration>``: ``plans.json``,
``dataset.json`` and ``fold_<n>/checkpoint_final.pth``. Conversion reads each
PyTorch checkpoint on the host with ``torch.load(weights_only=True)``, renames and
re-lays-out its network weights for the MLX network built from the same plans,
loads them strictly, and writes the ``nnunet-mlx`` model folder that
``load_model_bundle`` and the CLI run. Torch is needed only here, never for
inference.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import shutil
from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path
from typing import Protocol, cast

import numpy as np

from nnunet_mlx.contracts import parse_integer, parse_mapping, parse_sequence, parse_string
from nnunet_mlx.data.labels import parse_dataset_label_semantics
from nnunet_mlx.errors import ContractError
from nnunet_mlx.model import metadata as model_metadata
from nnunet_mlx.model.factory import build_model_from_plans
from nnunet_mlx.model.source_plans import upgrade_legacy_plans
from nnunet_mlx.plans import PlansContract, read_json_object
from nnunet_mlx.prediction.source import (
    DEFAULT_INFERENCE_SOURCE_PROFILE,
    PLANS_BUILT_TRAINERS,
)
from nnunet_mlx.storage.checkpoint import (
    CheckpointIdentity,
    InferenceCheckpointState,
    save_inference_checkpoint,
)
from nnunet_mlx.storage.conversion import (
    CONVERSION_FILENAME,
    CONVERSION_SCHEMA,
    read_conversion_manifest,
)
from nnunet_mlx.storage.filesystem import write_new_directory
from nnunet_mlx.storage.weights import convert_state_dict, load_converted_state

UPSTREAM_CHECKPOINT = "checkpoint_final.pth"
MLX_CHECKPOINT = "checkpoint_final"
CONVERSION_CONTRACT_VERSION = 1
# nnU-Net's own environment variable for trained models; its spelling is upstream's.
_RESULTS_VARIABLE = "nnUNet_results"
_INSTALL_HINT = "install the converter with: pip install 'nnunet-mlx[convert]'"


class _Serialization(Protocol):
    def safe_globals(self, safe_globals: list[object]) -> AbstractContextManager[object]:
        """Temporarily allow the given globals during restricted Torch loading."""
        ...


@dataclass(frozen=True, slots=True)
class UpstreamCheckpoint:
    """The fields of an upstream ``checkpoint_final.pth`` that inference needs.

    ``mirroring_axes`` is the trainer's ``inference_allowed_mirroring_axes``, with
    ``None`` (mirroring disabled, e.g. ``nnUNetTrainerNoMirroring``) as ``()``.
    """

    network_weights: dict[str, object]
    trainer_name: str
    configuration: str
    mirroring_axes: tuple[int, ...]
    current_epoch: int | None


@dataclass(frozen=True, slots=True)
class ConversionResult:
    """What ``convert_trained_model`` wrote."""

    destination: Path
    configuration: str
    trainer_name: str
    folds: tuple[int | str, ...]


def load_upstream_checkpoint(path: str | Path) -> UpstreamCheckpoint:
    """Read an upstream nnU-Net checkpoint with Torch's restricted unpickler.

    ``weights_only=True`` admits tensors, containers and primitives; nnU-Net's
    training log also stores NumPy scalars, so exactly those NumPy reconstructors
    are allowlisted. Anything else in the pickle is rejected by Torch.

    Raises:
        ContractError: If Torch is not installed, the file cannot be loaded, or it
            lacks the network weights, trainer name, configuration or mirroring axes.
    """
    source = Path(path)
    if not source.is_file():
        raise ContractError(f"upstream checkpoint does not exist: {source}")
    if find_spec("torch") is None:
        raise ContractError(f"reading {source.name} requires torch; {_INSTALL_HINT}")
    torch = import_module("torch")
    serialization = cast(_Serialization, import_module("torch.serialization"))
    scalar = import_module("numpy._core.multiarray").scalar
    allowed: list[object] = [
        scalar,
        (scalar, "numpy.core.multiarray.scalar"),
        np.dtype,
        *(type(np.dtype(dtype)) for dtype in (
            np.bool_, np.int8, np.int16, np.int32, np.int64,
            np.uint8, np.uint16, np.uint32, np.uint64,
            np.float16, np.float32, np.float64, np.complex64, np.complex128,
        )),
    ]
    with serialization.safe_globals(allowed):
        try:
            raw: object = torch.load(source, map_location="cpu", weights_only=True)
        # Restricted unpickling rejects with UnpicklingError; a damaged archive
        # raises RuntimeError from Torch's zip reader.
        except (pickle.UnpicklingError, RuntimeError, EOFError, OSError) as error:
            raise ContractError(f"cannot read upstream checkpoint {source}: {error}") from error
    root = parse_mapping(raw, name=source.name)
    init_args = parse_mapping(root.get("init_args"), name="checkpoint init_args")
    axes = root.get("inference_allowed_mirroring_axes")
    epoch = root.get("current_epoch")
    return UpstreamCheckpoint(
        network_weights=parse_mapping(root.get("network_weights"), name="network_weights"),
        trainer_name=parse_string(root.get("trainer_name"), name="trainer_name"),
        configuration=parse_string(init_args.get("configuration"), name="configuration"),
        mirroring_axes=(
            () if axes is None else tuple(
                parse_integer(axis, name="mirroring axis")
                for axis in parse_sequence(axes, name="inference_allowed_mirroring_axes")
            )
        ),
        current_epoch=None if epoch is None else parse_integer(epoch, name="current_epoch"),
    )


def detect_upstream_folds(
    model_folder: str | Path, *, checkpoint_name: str = UPSTREAM_CHECKPOINT
) -> tuple[int | str, ...]:
    """Return the numeric folds, then ``"all"``, whose folder holds ``checkpoint_name``."""
    folder = Path(model_folder)
    numeric: list[int] = []
    has_all = False
    for candidate in folder.glob("fold_*"):
        if not (candidate / checkpoint_name).is_file():
            continue
        suffix = candidate.name.removeprefix("fold_")
        if suffix.isdigit() and str(int(suffix)) == suffix:
            numeric.append(int(suffix))
        elif suffix == "all":
            has_all = True
    folds: tuple[int | str, ...] = (*sorted(numeric), *(("all",) if has_all else ()))
    if not folds:
        raise ContractError(f"no fold_* folder in {folder} contains {checkpoint_name}")
    return folds


def convert_trained_model(
    source: str | Path,
    destination: str | Path,
    *,
    folds: Sequence[int | str] | None = None,
    checkpoint_name: str = UPSTREAM_CHECKPOINT,
) -> ConversionResult:
    """Convert a trained upstream model folder into a new MLX model folder.

    Every fold must come from the same trainer and configuration. The trainer must
    be one that builds its network from the plans (``PLANS_BUILT_TRAINERS``);
    otherwise the plans would not describe the trained network. Each fold's weights
    must map onto that network exactly. The finished folder is loaded with
    ``load_model_bundle`` before it is moved into place, so a folder that exists
    is one that loads.

    Args:
        source: Upstream trained model folder with ``plans.json``, ``dataset.json``
            and ``fold_<n>/<checkpoint_name>``.
        destination: New folder to create; it must not exist.
        folds: Folds to convert; ``None`` converts every fold that has the checkpoint.
        checkpoint_name: Upstream checkpoint file name in each fold folder.

    Raises:
        ContractError: If the source is incomplete, the trainer or architecture is
            unsupported, folds disagree, weights do not fit the planned network, or
            the destination exists.
    """
    folder = Path(source)
    if not folder.is_dir():
        raise ContractError(f"trained model folder does not exist: {folder}")
    if Path(checkpoint_name).name != checkpoint_name:
        raise ContractError("checkpoint name must be a file name")
    source_plans = read_json_object(folder / "plans.json")
    # Pre-2.4 plans get the architecture block nnU-Net itself reconstructs on load.
    plans_json = upgrade_legacy_plans(source_plans)
    dataset_json = read_json_object(folder / "dataset.json")
    selected = (
        detect_upstream_folds(folder, checkpoint_name=checkpoint_name)
        if folds is None
        else model_metadata.validate_folds(folds)
    )
    paths = [folder / f"fold_{fold}" / checkpoint_name for fold in selected]
    first = load_upstream_checkpoint(paths[0])
    if first.trainer_name not in PLANS_BUILT_TRAINERS:
        raise ContractError(
            f"trainer {first.trainer_name!r} is not supported: only nnU-Net v2.8.1 trainers "
            "that build the network from plans.json can be converted"
        )
    configuration = first.configuration
    plans = PlansContract.from_payload(plans_json, configuration=configuration)
    semantics = parse_dataset_label_semantics(dataset_json)
    topology = plans.topology_payload()
    model = build_model_from_plans(
        topology,
        configuration=configuration,
        input_channels=model_metadata.model_input_channel_count(dataset_json, plans, semantics),
        output_channels=semantics.output_channels,
    )
    configurations = parse_mapping(topology["configurations"], name="configurations")
    architecture = parse_mapping(
        parse_mapping(configurations[configuration], name=configuration)["architecture"],
        name="architecture",
    )
    spatial_dims = len(plans.configuration.patch_size)

    def write(staging: Path) -> None:
        if plans_json == source_plans:
            shutil.copyfile(folder / "plans.json", staging / "plans.json")
        else:
            (staging / "plans.json").write_text(
                json.dumps(plans_json, indent=2) + "\n", encoding="utf-8"
            )
        shutil.copyfile(folder / "dataset.json", staging / "dataset.json")
        entries: list[dict[str, object]] = []
        for fold, path in zip(selected, paths, strict=True):
            checkpoint = first if path == paths[0] else load_upstream_checkpoint(path)
            if (checkpoint.trainer_name, checkpoint.configuration) != (
                first.trainer_name, configuration
            ):
                raise ContractError(f"fold {fold} was trained with a different trainer or "
                                    "configuration than the other folds")
            converted = convert_state_dict(checkpoint.network_weights)
            try:
                load_converted_state(model, converted)
            except ValueError as error:
                raise ContractError(
                    f"fold {fold} weights do not match the network planned in plans.json: {error}"
                ) from error
            save_inference_checkpoint(
                staging / f"fold_{fold}" / MLX_CHECKPOINT,
                model=model,
                state=InferenceCheckpointState(
                    identity=CheckpointIdentity(checkpoint.trainer_name, configuration, fold),
                    current_epoch=checkpoint.current_epoch,
                    spatial_dims=spatial_dims,
                    inference_allowed_mirroring_axes=checkpoint.mirroring_axes,
                    dropped_source_keys=converted.dropped_keys,
                ),
            )
            stat = path.stat()
            # Size and modification time let a cache notice a retrained checkpoint.
            entries.append({
                "fold": fold, "checkpoint": MLX_CHECKPOINT,
                "source_checkpoint": checkpoint_name,
                "source_size_bytes": stat.st_size, "source_mtime_ns": stat.st_mtime_ns,
            })
        manifest = {
            "schema_version": CONVERSION_SCHEMA,
            "conversion_contract_version": CONVERSION_CONTRACT_VERSION,
            "source_plans": source_plans,
            "source_dataset": dataset_json,
            "engine_version": version("nnunet-mlx"),
            "configuration": configuration,
            "source_profile": DEFAULT_INFERENCE_SOURCE_PROFILE,
            "architecture": architecture["network_class_name"],
            "spatial_dims": spatial_dims,
            "folds": entries,
        }
        (staging / CONVERSION_FILENAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        read_conversion_manifest(staging)
        from nnunet_mlx.model.folder import load_model_bundle

        load_model_bundle(staging, configuration=configuration, folds=selected)

    write_new_directory(Path(destination), write)
    return ConversionResult(Path(destination), configuration, first.trainer_name, selected)


def resolve_results_folder(
    dataset: str, *, configuration: str, trainer: str = "nnUNetTrainer",
    plans: str = "nnUNetPlans", results: str | Path | None = None,
) -> Path:
    """Find ``<results>/<dataset>/<trainer>__<plans>__<configuration>`` as nnU-Net does.

    ``dataset`` is a dataset ID (``3``) or full name (``Dataset003_Liver``); an ID
    selects the one ``Dataset<ID:03d>_*`` folder. ``results`` defaults to the
    ``nnUNet_results`` environment variable.
    """
    root_value = results if results is not None else os.environ.get(_RESULTS_VARIABLE)
    if not root_value:
        raise ContractError("set nnUNet_results, or pass a trained model folder with -m")
    root = Path(root_value).expanduser()
    if dataset.isdigit():
        pattern = re.compile(rf"Dataset{int(dataset):03d}_.+")
        matches = sorted(path.name for path in root.glob("Dataset*")
                         if path.is_dir() and pattern.fullmatch(path.name))
        if len(matches) != 1:
            raise ContractError(f"expected one Dataset{int(dataset):03d}_* folder in {root}, "
                                f"found {len(matches)}")
        dataset = matches[0]
    folder = root / dataset / f"{trainer}__{plans}__{configuration}"
    if not folder.is_dir():
        raise ContractError(f"trained model folder does not exist: {folder}")
    return folder


def cached_conversion(
    trained_folder: str | Path, *, folds: Sequence[int | str] | None,
    checkpoint_name: str = UPSTREAM_CHECKPOINT, cache: str | Path | None = None,
) -> Path:
    """Return an MLX conversion of ``trained_folder``, converting when needed.

    Conversions live under ``NNUNET_MLX_CACHE`` (default ``~/.cache/nnunet-mlx``) in
    ``converted/<dataset>/<trainer__plans__configuration>``. A conversion is reused
    while it came from the same folder, holds the requested folds, and each source
    checkpoint keeps its recorded size and modification time, source JSON metadata
    is unchanged, and all payload files exist. A conversion-contract revision also
    forces rebuilding. Every available fold is converted for later fold selections.
    """
    # Normalize dot components in the cache name without resolving symlink aliases.
    # A trailing /.. must never become a cache entry passed to rmtree.
    folder = Path(trained_folder).absolute()
    available = detect_upstream_folds(folder, checkpoint_name=checkpoint_name)
    wanted = available if folds is None else model_metadata.validate_folds(folds)
    missing = [fold for fold in wanted if fold not in available]
    if missing:
        raise ContractError(f"folds {missing} have no {checkpoint_name} in {folder}")
    root = Path(cache or os.environ.get("NNUNET_MLX_CACHE", "~/.cache/nnunet-mlx")).expanduser()
    cache_folder = Path(os.path.normpath(str(folder)))
    if not cache_folder.name or not cache_folder.parent.name:
        raise ContractError("trained folder needs dataset and model directory names")
    dataset_cache = root / "converted" / cache_folder.parent.name
    expected_parent = root.resolve() / "converted" / cache_folder.parent.name
    if dataset_cache.resolve() != expected_parent:
        raise ContractError(f"conversion cache has a symlinked parent: {dataset_cache}")
    target = dataset_cache / cache_folder.name
    if target.is_symlink():
        raise ContractError(f"conversion cache entry must not be a symlink: {target}")
    if _conversion_is_current(target, folder, available, checkpoint_name):
        return target
    if target.exists():
        shutil.rmtree(target)
    convert_trained_model(folder, target, folds=available, checkpoint_name=checkpoint_name)
    marker = target / ".source"
    marker.write_text(str(folder) + "\n", encoding="utf-8")
    return target


def _conversion_is_current(
    target: Path, folder: Path, folds: tuple[int | str, ...], checkpoint_name: str
) -> bool:
    """Check source metadata, conversion semantics and expected payload availability."""
    marker = target / ".source"
    if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != str(folder):
        return False
    try:
        read_conversion_manifest(target)
        manifest = read_json_object(target / CONVERSION_FILENAME)
        source_plans = read_json_object(folder / "plans.json")
        source_dataset = read_json_object(folder / "dataset.json")
        if (
            manifest.get("conversion_contract_version") != CONVERSION_CONTRACT_VERSION
            or manifest.get("source_plans") != source_plans
            or manifest.get("source_dataset") != source_dataset
            or read_json_object(target / "plans.json") != upgrade_legacy_plans(source_plans)
            or read_json_object(target / "dataset.json") != source_dataset
        ):
            return False
    except ContractError:
        return False
    recorded: dict[object, tuple[object, object, object]] = {}
    for value in parse_sequence(manifest.get("folds"), name="conversion folds"):
        entry = parse_mapping(value, name="conversion fold")
        recorded[entry.get("fold")] = (
            entry.get("source_checkpoint"), entry.get("source_size_bytes"),
            entry.get("source_mtime_ns"),
        )
    if set(recorded) != set(folds):
        return False
    for fold in folds:
        stat = (folder / f"fold_{fold}" / checkpoint_name).stat()
        if recorded[fold] != (checkpoint_name, stat.st_size, stat.st_mtime_ns):
            return False
    return True
