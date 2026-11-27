"""nnU-Net AutoPET II PET/CT lesion segmentation, MedMLX model ``nnunet-autopet``."""

from __future__ import annotations

from collections.abc import Mapping

from medmlx_core.files import PathLike, input_channels
from medmlx_core.weights import model_folder

from nnunet_mlx import medmlx
from nnunet_mlx.medmlx_types import RunResult
from nnunet_mlx.named_models import READINESS

MODEL = {
    "id": "nnunet-autopet", "label": "nnU-Net AutoPET II", "task": "segmentation",
    "readiness": READINESS,
    "run": "nnunet_mlx.named_models.autopet:run",
    # Measured MLX peak for a 400x400x284 whole-body PET/CT; larger inputs need more.
    "peak_memory_gb": 4.8,
    "summary": "Tumor lesion segmentation of co-registered CT and SUV PET volumes.",
    "labels": {"1": "tumor"},
}

# Network channel order; dataset.json names both channels "CT".
CHANNELS = ("ct", "pet")


def run(
    *, image_paths: Mapping[str, PathLike], output_dir: PathLike,
    weights_path: PathLike | None = None, overwrite: bool = False,
    use_mirroring: bool = True, step_fraction: float = 0.5,
) -> RunResult:
    """Segment tumor lesions (1) from co-registered CT and SUV PET, on their grid.

    Every fold present in the model folder is ensembled with the ``3d_fullres_resenc_bs80``
    plans.

    Args:
        image_paths: ``{"ct": ..., "pet": ...}``: CT in Hounsfield units and FDG PET
            in SUV, co-registered NIfTI volumes on one grid.
        output_dir: Folder for ``mask.nii.gz``; created if missing.
        weights_path: Model folder; omitted weights download MedMLX/nnunet-autopet-mlx.
        overwrite: Replace an existing mask; input/output aliases are always refused.
        use_mirroring: Average the checkpoint's test-time flips.
        step_fraction: Sliding-window step as a fraction of the patch, in (0, 1].
    """
    channels = input_channels(image_paths, CHANNELS)
    images, _ = medmlx.check_request(
        list(channels.values()), output_dir=output_dir, overwrite=overwrite,
        use_mirroring=use_mirroring, step_fraction=step_fraction,
    )
    folder = model_folder("nnunet-autopet", weights_path)
    return medmlx.segment_case(
        weights_path=folder, image_paths=images, output_dir=output_dir,
        configuration="3d_fullres_resenc_bs80", overwrite=overwrite,
        grid_of="image_paths", use_mirroring=use_mirroring, step_fraction=step_fraction,
    )
