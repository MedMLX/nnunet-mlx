"""Atomically persist the last completed epoch and export native inference weights."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, cast

from mlx.optimizers import AdamW
from mlx.utils import tree_flatten, tree_unflatten

from nnunet_mlx.contracts import parse_mapping, parse_sequence
from nnunet_mlx.errors import ContractError
from nnunet_mlx.model.network import Network
from nnunet_mlx.prediction.source import NATIVE_TRAINER_NAME
from nnunet_mlx.runtime import import_mlx
from nnunet_mlx.storage.checkpoint import (
    CheckpointIdentity,
    InferenceCheckpointState,
    assert_finite,
    save_inference_checkpoint,
)

if TYPE_CHECKING:
    from mlx.nn import Parameters

type History = list[dict[str, object]]
_SCHEMA = "nnunet-mlx.training-checkpoint.v1"
_mx, _ = import_mlx()


def save_training_state(path: Path, model: Network, optimizer: AdamW, history: History) -> None:
    """Replace the resume file only after all weights, moments and metrics are written."""
    weights = model.parameters()
    state = optimizer.state
    assert_finite(weights, name="model")
    assert_finite(state, name="optimizer")
    arrays = dict(tree_flatten({"model": weights, "optimizer": state}))
    with TemporaryDirectory(dir=path.parent, prefix=".training-") as temporary:
        staged = Path(temporary) / path.name
        _mx.save_safetensors(
            staged,
            arrays,
            metadata={
                "schema": _SCHEMA,
                "history": json.dumps(history, allow_nan=False),
            },
        )
        staged.replace(path)


def load_training_state(path: Path, model: Network, optimizer: AdamW) -> History:
    """Restore both trees and return the history of committed epochs."""
    if not path.is_file():
        raise ContractError(f"training checkpoint does not exist: {path}")
    arrays, metadata = _mx.load(path, format="safetensors", return_metadata=True)
    if metadata.get("schema") != _SCHEMA or "history" not in metadata:
        raise ContractError("unsupported training checkpoint schema")
    expected = dict(tree_flatten({"model": model.parameters(), "optimizer": optimizer.state}))
    if arrays.keys() != expected.keys() or any(
        actual.shape != expected[key].shape or actual.dtype != expected[key].dtype
        for key, actual in arrays.items()
    ):
        raise ContractError("training checkpoint tensors do not match the model and optimizer")
    trees = parse_mapping(tree_unflatten(arrays), name="training checkpoint tensors")
    weights = cast("Parameters", trees["model"])
    state = cast("Parameters", trees["optimizer"])
    assert_finite(weights, name="model")
    assert_finite(state, name="optimizer")
    model.update(weights)
    optimizer.state = state
    optimizer.init(model.trainable_parameters())
    try:
        records: object = json.loads(metadata["history"])
    except json.JSONDecodeError as error:
        raise ContractError("training checkpoint history is invalid") from error
    history = [
        parse_mapping(record, name="epoch metrics")
        for record in parse_sequence(records, name="history")
    ]
    if any(record.get("epoch") != index for index, record in enumerate(history, start=1)):
        raise ContractError("training checkpoint epochs are out of sequence")
    return history


def export_final(
    folder: Path,
    model: Network,
    *,
    configuration: str,
    fold: int | str,
    epoch: int,
    augment: bool,
) -> Path:
    """Publish one immutable inference checkpoint and point checkpoint_final at it."""
    snapshot = folder / f"checkpoint_epoch_{epoch}"
    rank = model.spec.spatial_dims
    if not snapshot.is_dir():
        save_inference_checkpoint(
            snapshot,
            model=model,
            state=InferenceCheckpointState(
                identity=CheckpointIdentity(NATIVE_TRAINER_NAME, configuration, fold),
                current_epoch=epoch,
                spatial_dims=rank,
                inference_allowed_mirroring_axes=tuple(range(rank)) if augment else (),
                dropped_source_keys=(),
            ),
        )
    final = folder / "checkpoint_final"
    if final.exists() and not final.is_symlink():
        raise ContractError(f"cannot replace a checkpoint directory: {final}")
    with TemporaryDirectory(dir=folder, prefix=".inference-") as temporary:
        staged = Path(temporary) / final.name
        staged.symlink_to(snapshot.name, target_is_directory=True)
        staged.replace(final)
    return final


def write_training_json(path: Path, payload: object) -> None:
    """Replace run metadata and metrics atomically within the run filesystem."""
    with TemporaryDirectory(dir=path.parent, prefix=".metadata-") as temporary:
        staged = Path(temporary) / path.name
        staged.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        staged.replace(path)
