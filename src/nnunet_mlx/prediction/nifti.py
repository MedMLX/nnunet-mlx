"""In-memory NIfTI preparation and label restoration for one 3-D class model."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from importlib import import_module
from typing import TYPE_CHECKING, cast

import nibabel as nib
import numpy as np
import numpy.typing as npt

from nnunet_mlx.data.preprocessing import preprocess_raw_case
from nnunet_mlx.errors import ContractError
from nnunet_mlx.prediction.export import restore_device_segmentation, restore_prediction
from nnunet_mlx.prediction.inference import device_logits_to_host
from nnunet_mlx.prediction.sliding_window import predict_labels
from nnunet_mlx.storage.imageio import label_dtype, resolve_image_reader

if TYPE_CHECKING:
    from nnunet_mlx.prediction.pipeline import ModelFolderPredictor


def predict_nifti(
    predictor: ModelFolderPredictor, image: nib.Nifti1Image,
) -> nib.Nifti1Image:
    """Predict labels for one in-memory NIfTI image; see ``ModelFolderPredictor.predict_nifti``.

    Voxel data must be real and finite before and after conversion to float32. The returned
    labels retain the input image's geometry and header fields.
    """
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
    prepared = preprocess_raw_case(data, {"spacing": spacing}, bundle.plans)
    crop_shape = cast(
        "tuple[int, int, int]", prepared.properties["shape_after_cropping_and_before_resampling"],
    )
    if len(runner.models) == 1 and prepared.data.shape[1:] == crop_shape:
        # One fold without logit resampling reduces each finished slab on Metal.
        cropped = predict_labels(
            runner.models[0], prepared.data, patch_size=runner.patch_size,
            num_classes=runner.num_outputs, mirroring_axes=runner.mirroring_axes,
            step_fraction=runner.step_fraction, accumulation=runner.accumulation,
        )
        shape_before = cast("tuple[int, int, int]", prepared.properties["shape_before_cropping"])
        bounding_box = cast("list[list[int]]", prepared.properties["bbox_used_for_cropping"])
        uncropped = np.zeros(shape_before, dtype=np.uint16)
        uncropped[tuple(slice(lower, upper) for lower, upper in bounding_box)] = cropped
        segmentation = uncropped.transpose(bundle.plans.transpose_backward)
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
    return nib.Nifti1Image(
        restored_labels, affine, image.header.copy(), dtype=label_dtype(restored_labels),
    )


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
