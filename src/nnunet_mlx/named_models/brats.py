"""nnU-Net BraTS brain-tumor MRI segmentation, MedMLX model ``nnunet-brats2021``."""

from __future__ import annotations

from collections.abc import Mapping

from medmlx_core.errors import InvalidInputError
from medmlx_core.files import PathLike, input_channels
from medmlx_core.weights import model_folder

from nnunet_mlx import medmlx
from nnunet_mlx.medmlx_types import RunResult
from nnunet_mlx.named_models import READINESS, segment_folder

MODEL = {
    "id": "nnunet-brats2021", "label": "nnU-Net BraTS 2021", "task": "segmentation",
    "readiness": READINESS,
    "run": "nnunet_mlx.named_models.brats:run",
    # Measured MLX peak for a 240x240x155 four-sequence MRI; larger inputs need more.
    "peak_memory_gb": 7.5,
    "summary": "Zenodo 11582627 Dataset002_BRATS19 with its original BAMF labels.",
    "labels": {"1": "edema", "2": "nonenhancing", "3": "empty", "4": "enhancing"},
}

# Network channel order, as named in the checkpoint's dataset.json.
CHANNELS = ("t1", "t1ce", "t2", "flair")


def run(
    *, image_paths: Mapping[str, PathLike], output_dir: PathLike,
    weights_path: PathLike | None = None, overwrite: bool = False, fold: int | None = None,
    use_mirroring: bool = True,
) -> RunResult:
    """Segment four co-registered MRI sequences with the original BAMF label map.

    Labels remain background 0, edema 1, nonenhancing 2, empty 3 and enhancing 4.
    Invalid folds fail before downloading.

    Args:
        image_paths: ``{"t1": ..., "t1ce": ..., "t2": ..., "flair": ...}``:
            co-registered NIfTI volumes on one grid.
        output_dir: Folder for ``mask.nii.gz``; created if missing.
        weights_path: Model folder; omitted weights download MedMLX/nnunet-brats-mlx.
        overwrite: Replace an existing mask; input/output aliases are always refused.
        fold: Optional single trained fold, 0 to 4; by default all five are ensembled.
        use_mirroring: Average test-time flips as in the original runner.
    """
    if fold is not None and fold not in range(5):
        raise InvalidInputError("BraTS fold must be between 0 and 4")
    channels = input_channels(image_paths, CHANNELS)
    images, _ = medmlx.check_request(
        list(channels.values()), output_dir=output_dir, overwrite=overwrite,
        folds=None if fold is None else [fold], use_mirroring=use_mirroring,
    )
    folder = model_folder("nnunet-brats2021", weights_path)
    folds = list(range(5)) if fold is None else [fold]
    return segment_folder(
        folder, images, output_dir, "3d_fullres", folds, overwrite=overwrite,
        grid_of="image_paths", use_mirroring=use_mirroring,
    )
