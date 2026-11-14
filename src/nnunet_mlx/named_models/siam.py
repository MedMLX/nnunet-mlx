"""SIAM v0.3 head-tissue segmentation of one MRI or CT, MedMLX model ``siam``."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from medmlx_core.errors import InvalidInputError
from medmlx_core.files import PathLike
from medmlx_core.weights import model_folder

from nnunet_mlx import medmlx
from nnunet_mlx.errors import ContractError
from nnunet_mlx.medmlx_types import RunResult
from nnunet_mlx.named_models import READINESS, segment_folder
from nnunet_mlx.prediction.source import SIAM_SOURCE_PROFILE

MODEL = {
    "id": "siam", "label": "SIAM head tissue segmentation", "task": "segmentation",
    "readiness": READINESS,
    "run": "nnunet_mlx.named_models.siam:segment",
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


def segment(
    *, image_path: PathLike, output_dir: PathLike, weights_path: PathLike | None = None,
    overwrite: bool = False, voxel_size: float | None = None, fold: int | None = None,
) -> RunResult:
    """Segment head tissues of one MRI or CT volume as SIAM v0.3 does.

    Reproduces ``siam-pred -m 0`` (github.com/romainVala/SIAM): the network sees the
    volume in closest-canonical RAS voxel order, every fold is ensembled with
    Gaussian 0.5-step sliding windows and no mirroring, and the label map is returned
    on the input grid. SIAM writes the reoriented label map back with nearest-neighbor
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
    from importlib import import_module
    from tempfile import TemporaryDirectory

    import numpy as np

    from nnunet_mlx.prediction.export import (
        ExportedFiles,
        RestoredPrediction,
        completion_receipt_path,
        export_prediction,
        publish_prediction,
        require_prediction_destination,
    )

    if voxel_size is not None and not (
        isinstance(cast(object, voxel_size), int | float)
        and not isinstance(cast(object, voxel_size), bool) and 0 < voxel_size < 10
    ):
        raise InvalidInputError("voxel_size must be a spacing in mm between 0 and 10")
    (source_path,), _ = medmlx.check_request(
        [image_path], output_dir=output_dir, overwrite=overwrite,
        folds=None if fold is None else [fold], use_mirroring=False,
    )
    nibabel = import_module("nibabel")
    orientations = import_module("nibabel.orientations")
    output = Path(output_dir).expanduser()
    target = output / "mask.nii.gz"
    generation = ExportedFiles(target, probabilities=output / medmlx.PROBABILITIES,
                              properties=output / medmlx.PROBABILITY_PROPERTIES)
    try:
        require_prediction_destination(generation, overwrite=overwrite,
                                       input_files=(source_path,),
                                       extra_paths=(completion_receipt_path(output / "mask"),))
    except ContractError as error:
        raise InvalidInputError(str(error)) from error
    source = nibabel.load(str(source_path))
    if int(source.ndim) != 3:
        raise InvalidInputError(f"SIAM takes one 3-D volume, got shape {source.shape}")
    folder = model_folder("siam", weights_path)
    folds = None if fold is None else [fold]
    source_affine = np.asarray(source.affine, dtype=np.float64)
    ornt = orientations.io_orientation(source_affine)
    ras = orientations.axcodes2ornt("RAS")
    if voxel_size is None and np.array_equal(ornt, ras):
        # Upstream predicts RAS inputs directly, without rewriting them.
        return segment_folder(
            folder, [source_path], output_dir, "3d_fullres", folds, overwrite=overwrite,
            grid_of="image_path", source_profile=SIAM_SOURCE_PROFILE, use_mirroring=False,
        )
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
        result = segment_folder(
            folder, [staged_input], Path(staging) / "out", "3d_fullres", folds,
            overwrite=False, grid_of="image_path", source_profile=SIAM_SOURCE_PROFILE,
            use_mirroring=False,
        )
        staged = nibabel.load(result["outputs"]["mask"])
        labels = np.asarray(staged.dataobj)
        # The mask is on the canonical voxel grid; invert the reorientation exactly.
        restored = orientations.apply_orientation(
            labels, orientations.ornt_transform(ras, ornt)
        )
        # Validate the restored source grid before the shared final publication.
        identity = {"inputs": [str(source_path)], "voxel_size": voxel_size,
                    "details": result["details"]}
        staged_files = export_prediction(
            RestoredPrediction(np.ascontiguousarray(restored.transpose(2, 1, 0),
                                                     dtype=np.uint16), None),
            {"nibabel_stuff": {"original_affine": source_affine},
             "spacing": tuple(float(v) for v in source.header.get_zooms()[::-1])},
            Path(staging) / "restored",
            file_ending=".nii.gz", image_reader_writer="NibabelIO", overwrite=False,
            completion_identity=identity,
        )
        try:
            publish_prediction(staged_files, ExportedFiles(target), generation=generation,
                               output_prefix=output / "mask", overwrite=overwrite,
                               input_files=(source_path,), completion_identity=identity)
        except ContractError as error:
            raise InvalidInputError(str(error)) from error
    result["outputs"]["mask"] = target.resolve().as_posix()
    return result
