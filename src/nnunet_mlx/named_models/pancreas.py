"""PancCTMultiTalentV2 pancreas and cancer CT segmentation, MedMLX model ``nnunet-pancreas``."""

from __future__ import annotations

from medmlx_core.files import PathLike
from medmlx_core.weights import model_folder

from nnunet_mlx import medmlx
from nnunet_mlx.medmlx_types import RunResult
from nnunet_mlx.named_models import READINESS

MODEL = {
    "id": "nnunet-pancreas", "label": "Pancreas segmentation", "task": "segmentation",
    "readiness": READINESS,
    "run": "nnunet_mlx.named_models.pancreas:run",
    # Measured MLX peak for a 768x768x101 abdominal CT with mirroring; larger inputs need more.
    "peak_memory_gb": 14.5,
    "summary": "PancCTMultiTalentV2 pancreas and cancer segmentation on Metal.",
    "labels": {"1": "pancreas", "2": "cancer"},
}


def run(
    *, image_path: PathLike, output_dir: PathLike, weights_path: PathLike | None = None,
    overwrite: bool = False, fold: int = 0, use_mirroring: bool = True,
) -> RunResult:
    """Segment pancreas (1) and cancer (2) in one CT, on the CT's grid.

    Uses the inference-staged PancCTMultiTalentV2 checkpoint with its original
    ``3d_fullres_bs4`` configuration.

    Args:
        image_path: CT NIfTI volume in Hounsfield units.
        output_dir: Folder for ``mask.nii.gz``; created if missing.
        weights_path: Model folder; omitted weights download MedMLX/nnunet-pancreas-mlx.
        overwrite: Replace an existing mask; input/output aliases are always refused.
        fold: Converted checkpoint slot. The released fold_all is staged in slot 0.
        use_mirroring: Average test-time flips as in the original runner.
    """
    images, _ = medmlx.check_request(
        [image_path], output_dir=output_dir, overwrite=overwrite, folds=[fold],
        use_mirroring=use_mirroring,
    )
    folder = model_folder("nnunet-pancreas", weights_path)
    return medmlx.segment_case(
        weights_path=folder, image_paths=images, output_dir=output_dir,
        configuration="3d_fullres_bs4", folds=[fold], overwrite=overwrite,
        grid_of="image_path", use_mirroring=use_mirroring,
    )
