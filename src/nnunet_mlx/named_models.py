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
from nnunet_mlx.prediction.source import SIAM_SOURCE_PROFILE

LIVER = {
    "id": "nnunet-liver", "label": "nnU-Net liver CT", "task": "segmentation",
    "readiness": "nnunet_mlx.medmlx:readiness",
    "run": "nnunet_mlx.named_models:liver",
    "cohort": "nnunet_mlx.named_models:prepare_liver",
    # Measured MLX peak for a 768x768x101 abdominal CT; larger inputs need more.
    "peak_memory_gb": 9.0,
    "summary": "Liver and liver cancer segmentation of one CT volume.",
    "labels": {"1": "liver", "2": "cancer"},
}
AUTOPET = {
    "id": "nnunet-autopet", "label": "nnU-Net AutoPET II", "task": "segmentation",
    "readiness": "nnunet_mlx.medmlx:readiness",
    "run": "nnunet_mlx.named_models:autopet",
    # Measured MLX peak for a 400x400x284 whole-body PET/CT; larger inputs need more.
    "peak_memory_gb": 4.8,
    "summary": "Tumor lesion segmentation of co-registered CT and SUV PET volumes.",
    "labels": {"1": "tumor"},
}
PANCREAS = {
    "id": "nnunet-pancreas", "label": "Pancreas segmentation", "task": "segmentation",
    "readiness": "nnunet_mlx.medmlx:readiness",
    "run": "nnunet_mlx.named_models:pancreas",
    # Measured MLX peak for a 768x768x101 abdominal CT with mirroring; larger inputs need more.
    "peak_memory_gb": 14.5,
    "summary": "PancCTMultiTalentV2 pancreas and cancer segmentation on Metal.",
    "labels": {"1": "pancreas", "2": "cancer"},
}
BRATS = {
    "id": "nnunet-brats2021", "label": "nnU-Net BraTS 2021", "task": "segmentation",
    "readiness": "nnunet_mlx.medmlx:readiness",
    "run": "nnunet_mlx.named_models:brats",
    # Measured MLX peak for a 240x240x155 four-sequence MRI; larger inputs need more.
    "peak_memory_gb": 7.5,
    "summary": "Zenodo 11582627 Dataset002_BRATS19 with its original BAMF labels.",
    "labels": {"1": "edema", "2": "nonenhancing", "3": "empty", "4": "enhancing"},
}
MRSEGMENTATOR = {
    "id": "mrsegmentator", "label": "MRSegmentator", "task": "segmentation",
    "readiness": "nnunet_mlx.medmlx:readiness",
    "run": "nnunet_mlx.named_models:mrsegmentator",
    "summary": "MRSegmentator 1.2 base weights with native Metal execution.",
}
SIAM = {
    "id": "siam", "label": "SIAM head tissue segmentation", "task": "segmentation",
    "readiness": "nnunet_mlx.medmlx:readiness",
    "run": "nnunet_mlx.named_models:siam",
    "summary": "Contrast-agnostic full-head tissue segmentation of one MRI or CT volume "
               "(SIAM v0.3, with an anomaly label).",
    # Measured MLX peak for a full-head T1 (five folds, 20 tiles, M5 Max); grows with FOV.
    "peak_memory_gb": 16.0,
    "labels": {
        "1": "GM", "2": "WM", "3": "CSF", "4": "CSFv", "5": "cerGM", "6": "Thal",
        "7": "Pal", "8": "Put", "9": "Caud", "10": "Accu", "11": "Amyg", "12": "Hippo",
        "13": "Dura", "14": "vascular", "15": "Skull", "16": "Head", "17": "Anomalies",
    },
}


def _run(
    folder: Path, images: Sequence[str], output_dir: str, configuration: str,
    folds: list[int] | None, *, overwrite: bool,
    source_profile: str = "nnunet-v2.8.1-default", arithmetic: str = "native",
    use_mirroring: bool = True, step_fraction: float = 0.5,
) -> RunResult:
    """Delegate inference and tensor/architecture admission to the generic runner."""
    return medmlx.run(
        weights_path=str(folder), image_paths=images, output_dir=output_dir,
        configuration=configuration, folds=folds, source_profile=source_profile,
        arithmetic=arithmetic, use_mirroring=use_mirroring, step_fraction=step_fraction,
        overwrite=overwrite,
    )


def liver(
    *, image_path: str, output_dir: str, weights_path: str | None = None,
    overwrite: bool = False, use_mirroring: bool = True, step_fraction: float = 0.5,
) -> RunResult:
    """Segment liver CT; automatically obtain the named model when weights are omitted.

    The defaults run the released configuration: three-axis mirroring (eight
    network passes per tile) and a 0.5 step. ``use_mirroring=False`` runs one pass
    per tile and a larger ``step_fraction`` runs fewer tiles; both change the
    prediction.
    """
    medmlx.check_request([image_path], output_dir=output_dir, overwrite=overwrite,
                         use_mirroring=use_mirroring, step_fraction=step_fraction)
    folder = (weights_dir("nnunet-liver") if weights_path is None
              else Path(weights_path).expanduser())
    return _run(folder, [image_path], output_dir, "3d_fullres", None, overwrite=overwrite,
                use_mirroring=use_mirroring, step_fraction=step_fraction)


def autopet(
    *, image_paths: Sequence[str], output_dir: str, weights_path: str | None = None,
    overwrite: bool = False, use_mirroring: bool = True, step_fraction: float = 0.5,
) -> RunResult:
    """Segment co-registered CT and SUV PET, supplied in that order.

    Every fold present in the model folder is ensembled. ``use_mirroring`` and
    ``step_fraction`` are as for ``liver``.
    """
    medmlx.check_request(image_paths, output_dir=output_dir, overwrite=overwrite,
                         use_mirroring=use_mirroring, step_fraction=step_fraction)
    if len(image_paths) != 2:
        raise InvalidInputError("AutoPET requires image_paths in CT, then SUV PET order")
    folder = (weights_dir("nnunet-autopet") if weights_path is None
              else Path(weights_path).expanduser())
    return _run(folder, image_paths, output_dir, "3d_fullres_resenc_bs80", None,
                overwrite=overwrite, use_mirroring=use_mirroring, step_fraction=step_fraction)


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


def siam(
    *, image_path: str, output_dir: str, weights_path: str | None = None,
    overwrite: bool = False, voxel_size: float | None = None, fold: int | None = None,
) -> RunResult:
    """Segment head tissues of one MRI or CT volume as SIAM v0.3 does.

    Reproduces ``siam-pred -m 0`` (github.com/romainVala/SIAM): the network sees the
    volume in closest-canonical RAS voxel order, every fold is ensembled with
    Gaussian 0.5-step sliding windows and no mirroring, and the label map is returned
    on the input grid. SIAM writes the reoriented label map back with nearest-neighbour
    resampling onto the input; for a pure axis permutation and flip that is the exact
    inverse reorientation used here.

    Args:
        image_path: One 3-D NIfTI head volume of any contrast.
        output_dir: Destination for ``mask.nii.gz``.
        weights_path: Local converted model folder; omitted weights download automatically.
        overwrite: Allow replacing an existing mask; input/output aliases are always refused.
        voxel_size: SIAM's ``-voxelsize``: before prediction, overwrite the canonical
            affine's diagonal with this isotropic spacing in mm, without resampling.
            Upstream uses it to scale small heads, such as newborns (about 1.4 times the
            true spacing), towards adult size. The mask keeps the input geometry.
        fold: Optional single fold; by default every fold in the model folder is used.
    """
    import os
    from importlib import import_module
    from tempfile import TemporaryDirectory

    import numpy as np

    if voxel_size is not None and not (
        isinstance(cast(object, voxel_size), int | float)
        and not isinstance(cast(object, voxel_size), bool) and 0 < voxel_size < 10
    ):
        raise InvalidInputError("voxel_size must be a spacing in mm between 0 and 10")
    medmlx.check_request([image_path], output_dir=output_dir, overwrite=overwrite,
                         folds=None if fold is None else [fold], use_mirroring=False)
    nibabel = import_module("nibabel")
    orientations = import_module("nibabel.orientations")
    source_path = Path(image_path).expanduser()
    output = Path(output_dir).expanduser()
    target = output / "mask.nii.gz"
    # The generic runner sees only the staged canonical scan on the reoriented path.
    if target.resolve() == source_path.resolve() or (
        target.exists() and target.samefile(source_path)
    ):
        raise InvalidInputError("SIAM output must differ from the input image")
    if (target.exists() or target.is_symlink()) and not overwrite:
        raise InvalidInputError(f"SIAM output already exists: {target}; pass overwrite=True")
    source = nibabel.load(str(source_path))
    if int(source.ndim) != 3:
        raise InvalidInputError(f"SIAM takes one 3-D volume, got shape {source.shape}")
    folder = (weights_dir("siam") if weights_path is None
              else Path(weights_path).expanduser())
    folds = None if fold is None else [fold]
    source_affine = np.asarray(source.affine, dtype=np.float64)
    ornt = orientations.io_orientation(source_affine)
    ras = orientations.axcodes2ornt("RAS")
    if voxel_size is None and np.array_equal(ornt, ras):
        # Upstream predicts RAS inputs directly, without rewriting them.
        return _run(folder, [str(source_path)], output_dir, "3d_fullres", folds,
                    overwrite=overwrite, source_profile=SIAM_SOURCE_PROFILE,
                    use_mirroring=False)
    # torchio.ToCanonical, used upstream, is this nibabel reorientation: voxels are
    # permuted and flipped, never interpolated, and saved as float32.
    canonical = source.as_reoriented(ornt)
    affine = np.array(canonical.affine, dtype=np.float64)
    if voxel_size is not None:
        # Upstream overwrites only the diagonal of the canonical affine.
        affine[0, 0] = affine[1, 1] = affine[2, 2] = float(voxel_size)
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".siam-", dir=output) as staging:
        staged_input = Path(staging) / "canonical.nii.gz"
        nibabel.save(nibabel.Nifti1Image(
            np.asarray(canonical.get_fdata(), dtype=np.float32), affine,
        ), str(staged_input))
        result = _run(folder, [str(staged_input)], str(Path(staging) / "out"), "3d_fullres",
                      folds, overwrite=False, source_profile=SIAM_SOURCE_PROFILE,
                      use_mirroring=False)
        staged = nibabel.load(result["outputs"]["mask"])
        labels = np.asarray(staged.dataobj)
        # The mask is on the canonical voxel grid; invert the reorientation exactly.
        restored = orientations.apply_orientation(
            labels, orientations.ornt_transform(ras, ornt)
        )
        # NiBabel follows links when saving; publish a completed private file instead.
        staged_mask = Path(staging) / "mask.nii.gz"
        nibabel.save(nibabel.Nifti1Image(np.ascontiguousarray(restored), source_affine),
                     str(staged_mask))
        if overwrite:
            os.replace(staged_mask, target)
        else:
            # A destination created during prediction must not be overwritten either.
            try:
                os.link(staged_mask, target)
            except FileExistsError as error:
                raise InvalidInputError(
                    f"SIAM output already exists: {target}; pass overwrite=True"
                ) from error
    result["outputs"]["mask"] = target.resolve().as_posix()
    return result


class LiverCohort:
    """Prepared liver model with the named runner's case argument names."""

    def __init__(self, prepared: medmlx.PreparedModel) -> None:
        self._prepared = prepared

    def run(self, *, image_path: str, output_dir: str, overwrite: bool = False) -> RunResult:
        """Segment a CT using the model owned by this cohort."""
        return self._prepared.run(image_paths=[image_path], output_dir=output_dir,
                                  overwrite=overwrite)

    def close(self) -> None:
        """Release the prepared model."""
        self._prepared.close()


def prepare_liver(
    *, weights_path: str | None = None, use_mirroring: bool = True, step_fraction: float = 0.5,
) -> LiverCohort:
    """Load the named liver model once for MedMLX's explicit cohort protocol.

    ``use_mirroring`` and ``step_fraction`` are as for ``liver``.
    """
    folder = (weights_dir("nnunet-liver") if weights_path is None
              else Path(weights_path).expanduser())
    return LiverCohort(medmlx.prepare(
        weights_path=str(folder), configuration="3d_fullres",
        use_mirroring=use_mirroring, step_fraction=step_fraction,
    ))
