"""Checkpoint-backed fold residency, independent of prediction orchestration.

All wrappers in a bundle share one network and one activation marker. Calls must
be serialized by a single owner, including materialization of lazy MLX outputs
before another fold activates. This module does not provide thread safety.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from nnunet_mlx.errors import ContractError
from nnunet_mlx.runtime import import_mlx


class ResidentModel[I, O, P, S](Protocol):
    """Structural type of the shared MLX network whose weights are swapped per fold."""

    @property
    def spec(self) -> S:
        """Return the resident network topology specification."""
        ...

    @property
    def training(self) -> bool:
        """Report whether the resident network is in training mode."""
        ...

    def __call__(self, values: I, /) -> O:
        """Run the resident network on its input."""
        ...

    def load_weights(self, file_or_weights: str, /, *, strict: bool) -> object:
        """Load named weights into the resident network."""
        ...

    def parameters(self) -> P:
        """Return the resident network parameter tree."""
        ...


@dataclass(slots=True)
class FoldActivation:
    """Marker for which fold's weights are currently loaded; ``None`` means unknown."""

    fold: int | str | None = None


class CheckpointBackedFoldModel[I, O, P, S]:
    """Activate one verified fold into a shared resident MLX network on demand.

    On a switch, the weights file's device, inode, size and modification time
    must still match its construction-time signature. This detects ordinary file
    changes, not content identity, and is not repeated while the fold stays active.
    Calls and lazy output materialization require the serialization described above.
    """

    def __init__(
        self,
        model: ResidentModel[I, O, P, S],
        *,
        fold: int | str,
        checkpoint: Path,
        activation: FoldActivation,
    ) -> None:
        """Bind a fold and its file metadata to the shared resident network."""
        self._model = model
        self.fold = fold
        self._weights_file = checkpoint / "weights.safetensors"
        self._weights_signature = _file_signature(self._weights_file)
        self._activation = activation

    @property
    def spec(self) -> S:
        """Return the shared network topology specification."""
        return self._model.spec

    @property
    def training(self) -> bool:
        """Report the shared network training state."""
        return bool(self._model.training)

    def __call__(self, values: I) -> O:
        """Activate this fold and run the shared network."""
        self._activate()
        return self._model(values)

    def _activate(self) -> None:
        """Verify file metadata and load this fold when it is not already active."""
        if self._activation.fold == self.fold:
            return
        if _file_signature(self._weights_file) != self._weights_signature:
            raise ContractError(
                f"fold {self.fold} checkpoint weights changed after bundle validation"
            )
        # Loading or evaluating weights can fail after mutating the network.
        # Never let the previous fold take the fast path after a failed switch.
        self._activation.fold = None
        load_fold_weights(self._model, self._weights_file)
        self._activation.fold = self.fold


def load_fold_weights[I, O, P, S](model: ResidentModel[I, O, P, S], weights_file: Path) -> None:
    """Load ``weights_file`` into ``model`` (``strict=True``) and materialize the parameters."""
    mx, _ = import_mlx()
    model.load_weights(str(weights_file), strict=True)
    mx.eval(model.parameters())


def _file_signature(path: Path) -> tuple[int, int, int, int]:
    """Read device, inode, size and modification time for fold change detection."""
    try:
        metadata = path.stat()
    except OSError as error:
        raise ContractError(f"checkpoint weights are unavailable: {path}") from error
    return metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns
