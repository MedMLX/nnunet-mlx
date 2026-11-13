"""Paired real CT and liver-mask crops for image preprocessing tests."""

from importlib import import_module
from pathlib import Path

import numpy as np
import numpy.typing as npt


def liver_crop(
    shape: tuple[int, int, int],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int16]]:
    """Load aligned HCC-TACE contrast CT and liver annotations on one source grid."""
    root = Path.home() / ".cache/medmlx/test-data"
    nib = import_module("nibabel")
    ct, liver = (nib.load(root / f"hcc_063_{name}.nii.gz") for name in ("ct", "liver"))
    if ct.shape != liver.shape or not np.allclose(ct.affine, liver.affine):
        raise ValueError("HCC-TACE CT and liver grids differ")
    region = tuple(
        slice(start, start + size) for start, size in zip((120, 220, 60), shape, strict=True)
    )
    image = np.asarray(ct.dataobj, dtype=np.float32)[region] / np.float32(1024)
    labels = np.asarray(liver.dataobj, dtype=np.int16)[region]
    if image.shape != shape or not np.any(labels):
        raise ValueError("HCC-TACE crop must contain source liver")
    return image, labels


def tumor_crop(shape: tuple[int, int, int]) -> npt.NDArray[np.int16]:
    """Load the HCC-TACE tumor annotation on the same crop as the liver."""
    root = Path.home() / ".cache/medmlx/test-data"
    nib = import_module("nibabel")
    ct, tumor = (nib.load(root / f"hcc_063_{name}.nii.gz") for name in ("ct", "tumor"))
    if ct.shape != tumor.shape or not np.allclose(ct.affine, tumor.affine):
        raise ValueError("HCC-TACE CT and tumor grids differ")
    region = tuple(
        slice(start, start + size) for start, size in zip((120, 220, 60), shape, strict=True)
    )
    labels = np.asarray(tumor.dataobj, dtype=np.int16)[region]
    if labels.shape != shape or not np.any(labels):
        raise ValueError("HCC-TACE crop must contain source tumor")
    return labels


def anatomical_labels() -> npt.NDArray[np.int16]:
    """Map HCC-TACE liver and tumor masks and other CT tissue to distinct IDs."""
    root = Path.home() / ".cache/medmlx/test-data"
    nib = import_module("nibabel")
    ct, liver, tumor = (
        nib.load(root / f"hcc_063_{name}.nii.gz") for name in ("ct", "liver", "tumor")
    )
    if any(
        image.shape != ct.shape or not np.allclose(image.affine, ct.affine)
        for image in (liver, tumor)
    ):
        raise ValueError("HCC-TACE CT and annotation grids differ")
    region = (slice(100, 125), slice(205, 235), slice(55, 65))
    tissue = np.asarray(ct.dataobj, dtype=np.float32)[region] > -500
    organ = np.asarray(liver.dataobj, dtype=np.int16)[region] > 0
    lesion = np.asarray(tumor.dataobj, dtype=np.int16)[region] > 0
    return np.where(lesion, 7, np.where(organ, 2, np.where(tissue, 11, 0))).astype(np.int16)
