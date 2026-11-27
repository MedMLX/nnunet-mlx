"""MRSegmentator 1.2 MR segmentation, MedMLX model ``mrsegmentator``."""

from __future__ import annotations

from typing import cast

from medmlx_core.errors import InvalidInputError
from medmlx_core.files import PathLike
from medmlx_core.weights import model_folder

from nnunet_mlx import medmlx
from nnunet_mlx.medmlx_types import RunResult
from nnunet_mlx.named_models import READINESS

MODEL = {
    "id": "mrsegmentator", "label": "MRSegmentator", "task": "segmentation",
    "readiness": READINESS,
    "run": "nnunet_mlx.named_models.mrsegmentator:run",
    "summary": "MRSegmentator 1.2 base weights with native Metal execution.",
}


def run(
    *, image_path: PathLike, output_dir: PathLike, weights_path: PathLike | None = None,
    overwrite: bool = False, fast: bool = False, fold: int | None = None,
) -> RunResult:
    """Segment MR with the original MRSegmentator 1.2 base profile and precision.

    Default inference averages all five folds. ``fast=True`` selects fold 0;
    an explicit fold takes precedence. Mirroring stays disabled, and precision
    remains ``source-fp32``. Invalid folds fail before downloading.

    Args:
        image_path: MR NIfTI volume.
        output_dir: Folder for ``mask.nii.gz``; created if missing.
        weights_path: Local model folder. Automatic weights are not configured.
        overwrite: Replace an existing mask; input/output aliases are always refused.
        fast: Use fold 0 when no explicit fold is supplied.
        fold: Optional single fold, 0 to 4.
    """
    if fold is not None and fold not in range(5):
        raise InvalidInputError("MRSegmentator fold must be between 0 and 4")
    if not isinstance(cast(object, fast), bool):
        raise InvalidInputError("fast must be boolean; pass fast=true or fast=false")
    images, _ = medmlx.check_request(
        [image_path], output_dir=output_dir, overwrite=overwrite,
        folds=None if fold is None else [fold],
    )
    folder = model_folder("mrsegmentator", weights_path)
    folds = [fold] if fold is not None else [0] if fast else list(range(5))
    return medmlx.segment_case(
        weights_path=folder, image_paths=images, output_dir=output_dir,
        configuration="3d_fullres_bs8", folds=folds, overwrite=overwrite,
        grid_of="image_path", source_profile="mrsegmentator-1.2-base",
        precision="source-fp32", use_mirroring=False,
    )
