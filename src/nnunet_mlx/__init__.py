"""MLX-native nnU-Net inference.

The names in ``__all__`` are the supported Python API. Other modules are
implementation details and may change between releases.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import version
from typing import TYPE_CHECKING

from nnunet_mlx.errors import ContractError
from nnunet_mlx.model.topology import PlainConvUNetSpec, ResidualEncoderUNetSpec

if TYPE_CHECKING:
    from nnunet_mlx.model.folder import ModelBundle, load_model_bundle
    from nnunet_mlx.prediction.nifti import NiftiPrediction
    from nnunet_mlx.prediction.pipeline import ModelFolderPredictor
    from nnunet_mlx.storage.conversion import read_conversion_manifest
    from nnunet_mlx.storage.upstream import convert_trained_model

    __version__: str

__all__ = [
    "ContractError",
    "ModelBundle",
    "ModelFolderPredictor",
    "NiftiPrediction",
    "PlainConvUNetSpec",
    "ResidualEncoderUNetSpec",
    "convert_trained_model",
    "load_model_bundle",
    "read_conversion_manifest",
]

# Execution entry points load on first access so importing the package, the
# CLI, or its help text never initializes MLX. The version comes from the
# installed distribution metadata, so pyproject.toml is its only source.
_DEFERRED_EXPORTS = {
    "ModelBundle": "nnunet_mlx.model.folder",
    "ModelFolderPredictor": "nnunet_mlx.prediction.pipeline",
    "NiftiPrediction": "nnunet_mlx.prediction.nifti",
    "load_model_bundle": "nnunet_mlx.model.folder",
    "read_conversion_manifest": "nnunet_mlx.storage.conversion",
    "convert_trained_model": "nnunet_mlx.storage.upstream",
}


def __getattr__(name: str) -> object:
    """Resolve the version or supported execution exports lazily on first access."""
    if name == "__version__":
        return version("nnunet-mlx")
    owner = _DEFERRED_EXPORTS.get(name)
    if owner is None:
        raise AttributeError(f"module 'nnunet_mlx' has no attribute {name!r}")
    return getattr(import_module(owner), name)


def __dir__() -> list[str]:
    """Include supported lazy exports in module introspection."""
    return sorted({*globals(), *__all__})
