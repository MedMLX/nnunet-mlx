"""MLX-native nnU-Net inference checkpoints with verified file sets."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from nnunet_mlx.contracts import (
    parse_integer as _integer,
)
from nnunet_mlx.contracts import (
    parse_mapping as _mapping,
)
from nnunet_mlx.contracts import (
    parse_string as _string,
)
from nnunet_mlx.errors import ContractError
from nnunet_mlx.runtime import ArrayTree, import_mlx, tree_flatten
from nnunet_mlx.storage.filesystem import write_new_directory

if TYPE_CHECKING:
    from mlx.nn import Module

SCHEMA_VERSION = "nnunet-mlx.checkpoint.v2"
INFERENCE_SCHEMA_VERSION = "nnunet-mlx.inference-checkpoint.v2"
SOURCE_NNUNET_VERSION = "v2.8.1"
_PAYLOAD_FILES = ("weights.safetensors", "optimizer.safetensors", "rng.safetensors")
_INFERENCE_PAYLOAD_FILES = ("weights.safetensors",)


@dataclass(frozen=True, slots=True)
class CheckpointIdentity:
    """Which upstream trainer, plans configuration and fold produced a checkpoint."""

    trainer_name: str
    configuration: str
    fold: int | str

    def __post_init__(self) -> None:
        """Require a trainer, configuration and valid nonnegative or all fold."""
        if not self.trainer_name or not self.configuration:
            raise ContractError("checkpoint trainer and configuration must be non-empty")
        if not (
            (isinstance(self.fold, int) and not isinstance(self.fold, bool) and self.fold >= 0)
            or self.fold == "all"
        ):
            raise ContractError("checkpoint fold must be a non-negative integer or 'all'")


@dataclass(frozen=True, slots=True)
class InferenceCheckpointState:
    """Contents of a checkpoint's ``state.json`` that inference consumes.

    ``inference_allowed_mirroring_axes`` are indices into the spatial axes of the
    network input (0 to ``spatial_dims - 1``, in the plans' axis order) along which
    test-time mirroring is allowed. ``dropped_source_keys`` lists upstream
    state-dict keys that were intentionally not converted.
    """

    identity: CheckpointIdentity
    current_epoch: int | None
    spatial_dims: int
    inference_allowed_mirroring_axes: tuple[int, ...]
    dropped_source_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate epoch, spatial mirroring axes and unique dropped keys."""
        if self.current_epoch is not None and self.current_epoch < 0:
            raise ContractError("checkpoint epoch must be non-negative")
        _validate_spatial_axes(self.spatial_dims, self.inference_allowed_mirroring_axes)
        if len(set(self.dropped_source_keys)) != len(self.dropped_source_keys):
            raise ContractError("dropped source checkpoint keys must be unique")

    def to_payload(self) -> dict[str, object]:
        """Serialize inference identity, source version and inference metadata."""
        return {
            "schema_version": INFERENCE_SCHEMA_VERSION,
            "source_nnunet_version": SOURCE_NNUNET_VERSION,
            "identity": asdict(self.identity),
            "current_epoch": self.current_epoch,
            "spatial_dims": self.spatial_dims,
            "inference_allowed_mirroring_axes": list(self.inference_allowed_mirroring_axes),
            "dropped_source_keys": list(self.dropped_source_keys),
        }

    @classmethod
    def from_payload(cls, payload: object) -> InferenceCheckpointState:
        """Parse a JSON payload, rejecting other schema versions or source nnU-Net versions."""
        root = _mapping(payload, name="inference checkpoint state")
        if root.get("schema_version") != INFERENCE_SCHEMA_VERSION:
            raise ContractError(
                f"unsupported inference checkpoint schema {root.get('schema_version')!r}"
            )
        if root.get("source_nnunet_version") != SOURCE_NNUNET_VERSION:
            raise ContractError("inference checkpoint source nnU-Net version does not match")
        return cls(
            identity=_checkpoint_identity(root.get("identity")),
            current_epoch=(
                _integer(root["current_epoch"], name="current epoch")
                if root.get("current_epoch") is not None
                else None
            ),
            spatial_dims=_integer(root.get("spatial_dims"), name="spatial dimensions"),
            inference_allowed_mirroring_axes=_integer_tuple(
                root.get("inference_allowed_mirroring_axes"), name="mirroring axes"
            ),
            dropped_source_keys=_string_tuple(
                root.get("dropped_source_keys"), name="dropped source keys"
            ),
        )


def save_inference_checkpoint(
    directory: str | Path,
    *,
    model: Module,
    state: InferenceCheckpointState,
) -> None:
    """Atomically create a converted, inference-only MLX checkpoint."""

    write_new_directory(
        Path(directory),
        lambda temporary: _write_inference_checkpoint(temporary, model, state),
    )


def load_inference_checkpoint(
    directory: str | Path,
    *,
    model: Module,
) -> InferenceCheckpointState:
    """Validate checkpoint structure and metadata, then load its weights strictly.

    The file set and schema are checked, and the loaded parameters must be finite.
    This does not authenticate files or compare their contents to a digest.
    """

    source = Path(directory)
    payload = _read_checkpoint_payload(source)
    schema = payload.get("schema_version")
    if schema == SCHEMA_VERSION:
        _verify_file_set(source, _PAYLOAD_FILES)
        # Existing native checkpoints retain their original file-set contract.
        # Only pretrained identity, epoch and mirroring metadata are consumed.
        state = InferenceCheckpointState(
            identity=_checkpoint_identity(payload.get("identity")),
            current_epoch=_integer(payload.get("current_epoch"), name="current epoch"),
            spatial_dims=_integer(payload.get("spatial_dims"), name="spatial dimensions"),
            inference_allowed_mirroring_axes=_integer_tuple(
                payload.get("inference_allowed_mirroring_axes"), name="mirroring axes"
            ),
            dropped_source_keys=("best_ema", "optimizer_config", "schedule_config"),
        )
        if payload.get("source_nnunet_version") != SOURCE_NNUNET_VERSION:
            raise ContractError("checkpoint source nnU-Net version does not match")
    elif schema == INFERENCE_SCHEMA_VERSION:
        _verify_file_set(source, _INFERENCE_PAYLOAD_FILES)
        state = InferenceCheckpointState.from_payload(payload)
    else:
        raise ContractError(f"unsupported inference checkpoint schema {schema!r}")
    _load_model_weights(source, model)
    return state


def checkpoint_weights_signature(directory: str | Path) -> list[int]:
    """Return ``[size_bytes, mtime_ns]`` of the weights file, for prediction provenance.

    This is a change detector for the file on disk, not a content digest.
    """

    weights = Path(directory) / "weights.safetensors"
    try:
        metadata = weights.stat()
    except OSError as error:
        raise ContractError(f"checkpoint weights are unavailable: {weights}") from error
    return [metadata.st_size, metadata.st_mtime_ns]


def _write_inference_checkpoint(
    directory: Path,
    model: Module,
    state: InferenceCheckpointState,
) -> None:
    """Write finite native weights and metadata into a new checkpoint directory."""
    mx, _ = import_mlx()
    assert_finite(model.parameters(), name="model")
    mx.eval(model.parameters())
    model.save_weights(str(directory / "weights.safetensors"))
    _write_state(directory, state.to_payload())


def _write_state(
    directory: Path,
    state_payload: Mapping[str, object],
) -> None:
    """Write checkpoint metadata as formatted JSON."""
    (directory / "state.json").write_text(
        json.dumps(state_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _verify_file_set(directory: Path, payload_files: tuple[str, ...]) -> None:
    """Require the exact checkpoint payload and metadata file names."""
    expected_files = {*payload_files, "state.json"}
    if not directory.is_dir():
        raise ContractError(f"checkpoint directory does not exist: {directory}")
    actual_entries = {path.name for path in directory.iterdir()}
    if actual_entries != expected_files:
        raise ContractError(
            f"checkpoint file set mismatch: expected {sorted(expected_files)}, "
            f"found {sorted(actual_entries)}"
        )


def _read_checkpoint_payload(directory: Path) -> dict[str, object]:
    """Read a checkpoint state mapping from JSON with explicit errors."""
    if not directory.is_dir():
        raise ContractError(f"checkpoint directory does not exist: {directory}")
    if not (directory / "state.json").is_file():
        raise ContractError("checkpoint state.json does not exist")
    try:
        payload = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError("checkpoint state is invalid JSON") from error
    return _mapping(payload, name="checkpoint state")


def _load_model_weights(directory: Path, model: Module) -> None:
    """Load safetensors strictly and reject nonfinite model parameters."""
    mx, _ = import_mlx()
    model.load_weights(str(directory / "weights.safetensors"), strict=True)
    assert_finite(model.parameters(), name="model")
    mx.eval(model.parameters())


def assert_finite(tree: Mapping[str, ArrayTree], *, name: str) -> None:
    """Raise ``ContractError`` naming the first array in ``tree`` with a NaN or infinity."""
    mx, _ = import_mlx()
    checks = [(key, mx.all(mx.isfinite(value))) for key, value in tree_flatten(dict(tree))]
    if not checks:
        return
    all_finite = mx.all(mx.stack([finite for _, finite in checks]))
    mx.eval(all_finite)
    if bool(all_finite.item()):
        return
    for key, finite in checks:
        if not bool(finite.item()):
            raise ContractError(f"{name} tensor {key!r} is non-finite")


def _validate_spatial_axes(spatial_dims: int, axes: tuple[int, ...]) -> None:
    """Require unique mirroring axes for a two- or three-dimensional model."""
    if spatial_dims not in (2, 3):
        raise ContractError("checkpoint spatial dimensions must be 2 or 3")
    if len(set(axes)) != len(axes) or any(axis not in range(spatial_dims) for axis in axes):
        raise ContractError(f"invalid checkpoint mirroring axes {axes}")


def _checkpoint_identity(value: object) -> CheckpointIdentity:
    """Parse trainer, configuration and fold from checkpoint metadata."""
    identity = _mapping(value, name="checkpoint identity")
    return CheckpointIdentity(
        trainer_name=_string(identity.get("trainer_name"), name="trainer name"),
        configuration=_string(identity.get("configuration"), name="configuration"),
        fold=_fold(identity.get("fold")),
    )


def _integer_tuple(value: object, *, name: str) -> tuple[int, ...]:
    """Parse a JSON array of integers as a tuple."""
    if not isinstance(value, list):
        raise ContractError(f"{name} must be an integer array")
    return tuple(_integer(item, name=name) for item in cast(list[object], value))


def _string_tuple(value: object, *, name: str) -> tuple[str, ...]:
    """Parse a JSON array of strings as a tuple."""
    if not isinstance(value, list):
        raise ContractError(f"{name} must be a string array")
    return tuple(_string(item, name=name) for item in cast(list[object], value))


def _fold(value: object) -> int | str:
    """Parse an integer fold or the all-fold identifier."""
    if value == "all":
        return "all"
    return _integer(value, name="fold")
