"""Named nnU-Net models with fixed modality and configuration choices.

Every runner returns the label map as ``outputs["mask"]``, like the generic runner.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

from medmlx_core.errors import InvalidInputError
from medmlx_core.weights import weights_dir

from nnunet_mlx import medmlx
from nnunet_mlx.medmlx_types import RunResult

LIVER = {
    "id": "nnunet-liver", "label": "nnU-Net liver CT", "task": "segmentation",
    "readiness": "nnunet_mlx.medmlx:readiness",
    "run": "nnunet_mlx.named_models:liver",
    "summary": "Liver and liver cancer segmentation of one CT volume.",
    "labels": {"1": "liver", "2": "cancer"},
}
AUTOPET = {
    "id": "nnunet-autopet", "label": "nnU-Net AutoPET II", "task": "segmentation",
    "readiness": "nnunet_mlx.medmlx:readiness",
    "run": "nnunet_mlx.named_models:autopet",
    "summary": "Tumor lesion segmentation of co-registered CT and SUV PET volumes.",
    "labels": {"1": "tumor"},
}
PANCREAS = {
    "id": "nnunet-pancreas", "label": "Pancreas segmentation", "task": "segmentation",
    "readiness": "nnunet_mlx.medmlx:readiness",
    "run": "nnunet_mlx.named_models:pancreas",
    "summary": "PancCTMultiTalentV2 pancreas and cancer segmentation on Metal.",
    "labels": {"1": "pancreas", "2": "cancer"},
}
BRATS = {
    "id": "nnunet-brats2021", "label": "nnU-Net BraTS 2021", "task": "segmentation",
    "readiness": "nnunet_mlx.medmlx:readiness",
    "run": "nnunet_mlx.named_models:brats",
    "summary": "Zenodo 11582627 Dataset002_BRATS19 with its original BAMF labels.",
    "labels": {"1": "edema", "2": "nonenhancing", "3": "empty", "4": "enhancing"},
}
MRSEGMENTATOR = {
    "id": "mrsegmentator", "label": "MRSegmentator", "task": "segmentation",
    "readiness": "nnunet_mlx.medmlx:readiness",
    "run": "nnunet_mlx.named_models:mrsegmentator",
    "summary": "MRSegmentator 1.2 base weights with native Metal execution.",
}


def _run(
    folder: Path, images: Sequence[str], output_dir: str, configuration: str,
    folds: list[int] | None, *, overwrite: bool,
    source_profile: str = "nnunet-v2.8.1-default", arithmetic: str = "native",
    use_mirroring: bool = True,
) -> RunResult:
    """Delegate inference and tensor/architecture admission to the generic runner."""
    return medmlx.run(
        weights_path=str(folder), image_paths=images, output_dir=output_dir,
        configuration=configuration, folds=folds, source_profile=source_profile,
        arithmetic=arithmetic, use_mirroring=use_mirroring, overwrite=overwrite,
    )


def liver(
    *, image_path: str, output_dir: str, weights_path: str | None = None,
    overwrite: bool = False,
) -> RunResult:
    """Segment liver CT; automatically obtain the named model when weights are omitted."""
    medmlx.check_request([image_path], output_dir=output_dir, overwrite=overwrite)
    folder = (weights_dir("nnunet-liver") if weights_path is None
              else Path(weights_path).expanduser())
    return _run(folder, [image_path], output_dir, "3d_fullres", None, overwrite=overwrite)


def autopet(
    *, image_paths: Sequence[str], output_dir: str, weights_path: str | None = None,
    overwrite: bool = False,
) -> RunResult:
    """Segment co-registered CT and SUV PET, supplied in that order.

    Every fold present in the model folder is ensembled.
    """
    medmlx.check_request(image_paths, output_dir=output_dir, overwrite=overwrite)
    if len(image_paths) != 2:
        raise InvalidInputError("AutoPET requires image_paths in CT, then SUV PET order")
    folder = (weights_dir("nnunet-autopet") if weights_path is None
              else Path(weights_path).expanduser())
    return _run(folder, image_paths, output_dir, "3d_fullres_resenc_bs80", None,
                overwrite=overwrite)


def pancreas(
    *, image_path: str, output_dir: str, weights_path: str | None = None,
    overwrite: bool = False, fold: int = 0, use_mirroring: bool = True,
) -> RunResult:
    """Segment pancreas and cancer on the input CT grid.

    Uses the inference-staged PancCTMultiTalentV2 checkpoint, with background 0,
    pancreas 1 and cancer 2, and its original ``3d_fullres_bs4`` configuration.

    Args:
        image_path: Input CT NIfTI volume.
        output_dir: Destination for the predicted mask.
        weights_path: Optional local model folder; omitted weights download automatically.
        overwrite: Allow replacing existing outputs.
        fold: Converted checkpoint slot. The released fold_all is staged in slot 0.
        use_mirroring: Average test-time flips as in the original runner.
    """
    medmlx.check_request([image_path], output_dir=output_dir, overwrite=overwrite,
                         folds=[fold], use_mirroring=use_mirroring)
    folder = (weights_dir("nnunet-pancreas") if weights_path is None
              else Path(weights_path).expanduser())
    return _run(folder, [image_path], output_dir, "3d_fullres_bs4", [fold],
                overwrite=overwrite, use_mirroring=use_mirroring)


def brats(
    *, flair_path: str, t1_path: str, t1ce_path: str, t2_path: str, output_dir: str,
    weights_path: str | None = None, overwrite: bool = False, fold: int | None = None,
    use_mirroring: bool = True,
) -> RunResult:
    """Segment four co-registered MRI volumes using the original BAMF label map.

    Channel order remains T1, T1ce, T2, FLAIR. Labels remain background 0, edema 1,
    nonenhancing 2, empty 3 and enhancing 4. Invalid folds fail before downloading.

    Args:
        flair_path, t1_path, t1ce_path, t2_path: Input co-registered NIfTI volumes.
        output_dir: Destination for the predicted mask.
        weights_path: Optional local model folder; omitted weights download automatically.
        overwrite: Allow replacing existing outputs.
        fold: Optional single trained fold, 0 to 4; by default all five are ensembled.
        use_mirroring: Average test-time flips as in the original runner.
    """
    if fold is not None and fold not in range(5):
        raise InvalidInputError("BraTS fold must be between 0 and 4")
    medmlx.check_request([t1_path, t1ce_path, t2_path, flair_path], output_dir=output_dir,
                         overwrite=overwrite, folds=None if fold is None else [fold],
                         use_mirroring=use_mirroring)
    folder = (weights_dir("nnunet-brats2021") if weights_path is None
              else Path(weights_path).expanduser())
    folds = list(range(5)) if fold is None else [fold]
    return _run(folder, [t1_path, t1ce_path, t2_path, flair_path], output_dir, "3d_fullres",
                folds, overwrite=overwrite, use_mirroring=use_mirroring)


def mrsegmentator(
    *, image_path: str, output_dir: str, weights_path: str | None = None,
    overwrite: bool = False, fast: bool = False, fold: int | None = None,
) -> RunResult:
    """Segment MR with the original MRSegmentator 1.2 base profile and arithmetic.

    Default inference averages all five folds. ``fast=True`` selects fold 0;
    an explicit fold takes precedence. Mirroring stays disabled, and arithmetic
    remains ``torch-order-fp32``. Invalid folds fail before downloading.

    Args:
        image_path: Input MR NIfTI volume.
        output_dir: Destination for the predicted mask.
        weights_path: Local model folder. Automatic weights are not configured.
        overwrite: Allow replacing existing outputs.
        fast: Use fold 0 when no explicit fold is supplied.
        fold: Optional single fold, 0 to 4.
    """
    if fold is not None and fold not in range(5):
        raise InvalidInputError("MRSegmentator fold must be between 0 and 4")
    if not isinstance(cast(object, fast), bool):
        raise InvalidInputError("fast must be boolean; pass fast=true or fast=false")
    medmlx.check_request([image_path], output_dir=output_dir, overwrite=overwrite,
                         folds=None if fold is None else [fold])
    folder = (weights_dir("mrsegmentator") if weights_path is None
              else Path(weights_path).expanduser())
    folds = [fold] if fold is not None else [0] if fast else list(range(5))
    return _run(folder, [image_path], output_dir, "3d_fullres_bs8", folds,
                overwrite=overwrite, source_profile="mrsegmentator-1.2-base",
                arithmetic="torch-order-fp32", use_mirroring=False)
