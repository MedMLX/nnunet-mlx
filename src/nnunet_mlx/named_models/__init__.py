"""Released nnU-Net models, one module each, with fixed configurations.

Each module declares one MedMLX model as ``MODEL`` and its ``run``, which
returns the label map as ``outputs["mask"]`` on the input grid. Importing a
module does not initialize MLX.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from medmlx_core.files import PathLike

from nnunet_mlx import medmlx
from nnunet_mlx.medmlx_types import RunResult

READINESS = "nnunet_mlx.medmlx:readiness"


def segment_folder(
    folder: Path, images: Sequence[Path], output_dir: PathLike, configuration: str,
    folds: list[int] | None, *, overwrite: bool, grid_of: str,
    source_profile: str = "nnunet-v2.8.1-default", precision: str = "fast-fp32",
    use_mirroring: bool = True, step_fraction: float = 0.5,
) -> RunResult:
    """Segment one case of a released model; the generic path admits its tensors."""
    return medmlx.segment_case(
        grid_of=grid_of, weights_path=folder, image_paths=images, output_dir=output_dir,
        configuration=configuration, folds=folds, source_profile=source_profile,
        precision=precision, use_mirroring=use_mirroring, step_fraction=step_fraction,
        overwrite=overwrite,
    )
