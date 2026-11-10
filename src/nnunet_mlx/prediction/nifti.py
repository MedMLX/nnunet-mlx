"""In-memory NIfTI preparation, labels and probabilities for one 3-D class model."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, cast

import nibabel as nib
import numpy as np
import numpy.typing as npt

from nnunet_mlx.data.preprocessing import preprocess_raw_case
from nnunet_mlx.errors import ContractError
from nnunet_mlx.prediction.export import (
    predict_same_grid_segmentation,
    restore_device_segmentation,
    restore_prediction,
)
from nnunet_mlx.prediction.inference import device_logits_to_host
from nnunet_mlx.storage.imageio import label_dtype, resolve_image_reader

if TYPE_CHECKING:
    from nnunet_mlx.prediction.pipeline import ModelFolderPredictor


@dataclass(frozen=True)
class NiftiPrediction:
    """Source-grid labels, reader-grid probabilities and their verified geometry.

    ``probabilities`` is float32 ``(C, Z, Y, X)`` in the selected reader's grid;
    NibabelIOWithReorient retains canonical reader axes here. ``properties``
    carries reader geometry, crop, label metadata and an empty ``inputs`` list
    because no file was read. Labels retain the supplied image's shape/header.
    Returned arrays and metadata do not borrow mutable model state.
    """

    segmentation: nib.Nifti1Image
    probabilities: npt.NDArray[np.float32]
    properties: dict[str, object]


def predict_nifti(
    predictor: ModelFolderPredictor, image: nib.Nifti1Image,
) -> nib.Nifti1Image:
    """Return labels without probability storage; see ModelFolderPredictor.predict_nifti."""
    return _predict_nifti(predictor, image, return_probabilities=False)[0]


def predict_nifti_probabilities(
    predictor: ModelFolderPredictor, image: nib.Nifti1Image,
) -> NiftiPrediction:
    """Return probability-path labels and reader-grid probabilities without file I/O."""
    segmentation, probabilities, properties = _predict_nifti(
        predictor, image, return_probabilities=True,
    )
    assert probabilities is not None
    return NiftiPrediction(segmentation, probabilities, properties)


def _predict_nifti(
    predictor: ModelFolderPredictor, image: nib.Nifti1Image, *, return_probabilities: bool,
) -> tuple[nib.Nifti1Image, npt.NDArray[np.float32] | None, dict[str, object]]:
    """Prepare finite real voxels and restore on the existing reader/crop boundaries."""
    bundle = predictor.bundle
    runner = predictor.predictor
    reader = resolve_image_reader(bundle.image_reader_writer)
    if reader not in ("SimpleITKIO", "NibabelIO", "NibabelIOWithReorient"):
        raise ContractError("in-memory NIfTI inference requires a medical-image reader")
    if bundle.image_channels != 1 or bundle.input_channels != 1:
        raise ContractError("in-memory NIfTI inference requires one image channel")
    if bundle.plans.configuration.previous_stage_name is not None:
        raise ContractError("in-memory NIfTI inference does not support cascade models")
    if bundle.semantics.has_regions:
        raise ContractError("in-memory NIfTI inference requires a class model")
    if len(runner.patch_size) != 3:
        raise ContractError("in-memory NIfTI inference requires a three-dimensional model")
    if bundle.semantics.output_channels > np.iinfo(np.uint16).max:
        raise ContractError("in-memory NIfTI inference exceeds the uint16 class range")
    if image.ndim != 3 or not image.shape or min(image.shape) < 1:
        raise ContractError("in-memory NIfTI inference requires a nonempty three-dimensional image")
    affine = image.affine
    if affine is None or not np.isfinite(affine).all():
        raise ContractError("in-memory NIfTI inference requires a finite spatial affine")

    # NibabelIOWithReorient canonicalizes voxel axes first. Both readers then feed
    # preprocessing a (C, Z, Y, X) array with spacing reversed to match, as imageio does.
    reorient = reader == "NibabelIOWithReorient"
    orientations = vars(import_module("nibabel.orientations"))
    orientation = cast(
        "Callable[[object], npt.NDArray[np.float64]]", orientations["io_orientation"],
    )
    active = (image.as_reoriented(cast("Sequence[Sequence[int]]", orientation(affine)))
              if reorient else image)
    values = np.asanyarray(active.dataobj)
    if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
        raise ContractError("in-memory NIfTI voxels must be finite real values in float32 range")
    with np.errstate(over="ignore"):
        data = np.ascontiguousarray(values.transpose(2, 1, 0)[None], dtype=np.float32)
    if not np.isfinite(data).all():
        raise ContractError("in-memory NIfTI voxels must be finite real values in float32 range")
    spacing = tuple(float(value) for value in active.header.get_zooms()[:3][::-1])
    if reader == "SimpleITKIO":
        # ITK expresses NIfTI spacing in mm; NiBabel readers retain declared units.
        unit = active.header.get_xyzt_units()[0]
        scale = 1000.0 if unit == "meter" else 0.001 if unit == "micron" else 1.0
        spacing = tuple(value * scale for value in spacing)
    properties: dict[str, object] = {"spacing": spacing}
    if return_probabilities:
        properties.update(_reader_geometry(image, active, reader))
    prepared = preprocess_raw_case(data, properties, bundle.plans)
    probabilities = None
    if return_probabilities:
        logits = runner.predict_logits_from_preprocessed_data(prepared.data)
        restored = restore_prediction(
            logits, prepared.properties, bundle.plans, bundle.semantics, return_probabilities=True,
        )
        segmentation, probabilities = restored.segmentation, restored.probabilities
    else:
        streamed = predict_same_grid_segmentation(
            runner, prepared.data, prepared.properties, bundle.plans, bundle.semantics,
        )
        if streamed is not None:
            segmentation = streamed.segmentation
        else:
            segmentation = _restored_ensemble_labels(predictor, prepared.data, prepared.properties)
    labels = segmentation.transpose(2, 1, 0)
    result = nib.Nifti1Image(labels.astype(label_dtype(labels), copy=False), active.affine)
    if reorient:
        ras = cast("Callable[[str], npt.NDArray[np.float64]]", orientations["axcodes2ornt"])
        transform = cast(
            "Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]]",
            orientations["ornt_transform"],
        )(ras("RAS"), orientation(affine))
        result = result.as_reoriented(cast("Sequence[Sequence[int]]", transform))
    result_affine = result.affine
    if (result.shape != image.shape or result_affine is None
            or not np.allclose(result_affine, affine, rtol=0, atol=1e-5)):
        raise ContractError("in-memory prediction geometry differs from its NIfTI input")
    restored_labels = np.asanyarray(result.dataobj)
    result = nib.Nifti1Image(
        restored_labels, affine, image.header.copy(), dtype=label_dtype(restored_labels),
    )
    if return_probabilities:
        prepared.properties.update(inputs=[], labels=deepcopy(bundle.dataset_json["labels"]),
                                   regions_class_order=None)
    return result, probabilities, prepared.properties


def _restored_ensemble_labels(
    predictor: ModelFolderPredictor,
    data: npt.NDArray[np.float32],
    properties: dict[str, object],
) -> npt.NDArray[np.uint16]:
    """Average folds and restore labels on Metal, or on the host when the kernel declines."""
    bundle = predictor.bundle
    logits = predictor.predictor.predict_device_logits_from_preprocessed_data(data)
    restored = restore_device_segmentation(logits, properties, bundle.plans, bundle.semantics)
    if restored is None:
        host = device_logits_to_host(logits)
        del logits
        restored = restore_prediction(
            host, properties, bundle.plans, bundle.semantics, return_probabilities=False,
        )
    return restored.segmentation


def _reader_geometry(
    image: nib.Nifti1Image, active: nib.Nifti1Image, reader: str,
) -> dict[str, object]:
    """Preserve NIfTI header geometry; convert RAS to ITK's LPS millimeter convention.

    The header carries the same float32 sform/qform geometry a NIfTI-1 file
    would serialize. Conflicting coded forms or sheared ITK axes are rejected
    rather than choosing a different geometry from the file reader.
    """
    affine = np.asarray(image.header.get_best_affine(), dtype=np.float64)
    if image.affine is None or not np.allclose(affine, image.affine, rtol=0, atol=1e-5):
        raise ContractError("in-memory probability geometry differs from its NIfTI header")
    if reader != "SimpleITKIO":
        geometry: dict[str, object] = {"original_affine": affine}
        if reader == "NibabelIOWithReorient":
            geometry["reoriented_affine"] = np.asarray(active.header.get_best_affine(),
                                                       dtype=np.float64)
        return {"nibabel_stuff": geometry}
    qform, qcode = image.header.get_qform(coded=True)
    if qcode and (qform is None or not np.allclose(qform, affine, rtol=0, atol=1e-5)):
        raise ContractError("in-memory SimpleITK probabilities require consistent NIfTI forms")
    unit = image.header.get_xyzt_units()[0]
    scale = 1000.0 if unit == "meter" else 0.001 if unit == "micron" else 1.0
    spacing = np.asarray(image.header.get_zooms()[:3], dtype=np.float64)
    if np.any(spacing <= 0) or not np.isfinite(spacing).all():
        raise ContractError("in-memory probabilities require positive finite spacing")
    lps = np.diag([-1.0, -1.0, 1.0])
    direction = lps @ affine[:3, :3] / spacing
    if not np.allclose(direction.T @ direction, np.eye(3), rtol=0, atol=1e-5):
        raise ContractError("in-memory SimpleITK probabilities require orthonormal NIfTI axes")
    return {"sitk_stuff": {
        "spacing": tuple(float(value) for value in spacing * scale),
        "origin": tuple(float(value) for value in lps @ affine[:3, 3] * scale),
        "direction": tuple(float(value) for value in direction.ravel()),
    }}
