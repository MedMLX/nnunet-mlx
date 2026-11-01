"""Image readers and segmentation writers compared with the pinned upstream I/O."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from nnunet_reference.arrays import NumericArray

from nnunet_mlx.errors import ContractError
from nnunet_mlx.storage.imageio import (
    image_reader_output_sidecar,
    read_images,
    read_sitk_images,
    write_segmentation,
    write_sitk_segmentation,
)


def _write_image(
    path: Path,
    array: np.ndarray[tuple[int, ...], np.dtype[np.float32]],
    *,
    origin: tuple[float, float, float] = (7.0, 8.0, 9.0),
) -> None:
    """Write a synthetic image with known spatial geometry."""
    sitk = import_module("SimpleITK")
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.7, 0.8, 2.5))
    image.SetOrigin(origin)
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0))
    sitk.WriteImage(image, str(path))


def _write_compound_image(path: Path, array: NumericArray) -> None:
    """Write a synthetic multi-component image with spatial geometry."""
    sitk = import_module("SimpleITK")
    image = sitk.GetImageFromArray(array, isVector=False)
    image.SetSpacing((0.7, 0.8, 2.5, 4.0))
    image.SetOrigin((7.0, 8.0, 9.0, 0.0))
    image.SetDirection(tuple(np.eye(4, dtype=np.float64).ravel()))
    sitk.WriteImage(image, str(path))


@pytest.mark.oracle
def test_simpleitk_reader_matches_pinned_oracle(tmp_path: Path) -> None:
    """Compare SimpleITK image loading against the pinned source reader."""
    source = tmp_path / "case_0000.nii.gz"
    raw = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    _write_image(source, raw)
    source_reader = pytest.importorskip(
        "nnunetv2.imageio.simpleitk_reader_writer",
        reason="requires the optional 'oracle' extra",
        exc_type=ModuleNotFoundError,
    ).SimpleITKIO()

    expected, expected_properties = source_reader.read_images((str(source),))
    actual, properties = read_images("SimpleITKIO", (source,))

    assert np.array_equal(actual, expected)
    assert np.allclose(
        np.asarray(properties["spacing"]),
        np.asarray(expected_properties["spacing"]),
    )
    actual_geometry = cast(dict[str, object], properties["sitk_stuff"])
    expected_geometry = cast(dict[str, object], expected_properties["sitk_stuff"])
    assert actual_geometry.keys() == expected_geometry.keys()
    for key in actual_geometry:
        assert np.allclose(np.asarray(actual_geometry[key]), np.asarray(expected_geometry[key]))


@pytest.mark.oracle
def test_simpleitk_compound_reader_matches_oracle_and_exports_spatial_geometry(
    tmp_path: Path,
) -> None:
    """Match compound image reading and preserve spatial output geometry."""
    source = tmp_path / "compound.nii.gz"
    raw = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    _write_compound_image(source, raw)
    source_reader = pytest.importorskip(
        "nnunetv2.imageio.simpleitk_reader_writer",
        reason="requires the optional 'oracle' extra",
        exc_type=ModuleNotFoundError,
    ).SimpleITKIO()

    expected, expected_properties = source_reader.read_images((str(source),))
    actual, properties = read_sitk_images((source,))
    output = tmp_path / "prediction.nii.gz"
    write_sitk_segmentation(
        np.asarray(raw[0] > 30, dtype=np.uint16),
        output,
        properties,
        overwrite=False,
    )

    assert actual.shape == (2, 3, 4, 5)
    assert np.array_equal(actual, expected)
    assert np.allclose(
        np.asarray(properties["spacing"]),
        np.asarray(expected_properties["spacing"]),
    )
    sitk = import_module("SimpleITK")
    result = sitk.ReadImage(str(output))
    assert result.GetDimension() == 3
    assert np.allclose(result.GetSpacing(), (0.7, 0.8, 2.5))
    assert np.allclose(result.GetOrigin(), (7.0, 8.0, 9.0))
    assert np.allclose(result.GetDirection(), tuple(np.eye(3).ravel()))


@pytest.mark.oracle
def test_natural_rgb_reader_and_writer_match_upstream_layout(tmp_path: Path) -> None:
    """Match upstream RGB channel layout through natural image I/O."""
    image_io = import_module("skimage.io")
    source = tmp_path / "case_0000.png"
    rgb = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    image_io.imsave(str(source), rgb, check_contrast=False)

    data, properties = read_images("NaturalImage2DIO", (source,))
    oracle_reader = pytest.importorskip(
        "nnunetv2.imageio.natural_image_reader_writer",
        reason="requires the optional 'oracle' extra",
        exc_type=ModuleNotFoundError,
    ).NaturalImage2DIO()
    expected, expected_properties = oracle_reader.read_images((str(source),))
    labels = np.asarray(data[0] > 20, dtype=np.uint8)
    output = tmp_path / "prediction.png"
    sidecar = write_segmentation("NaturalImage2DIO", labels, output, properties, overwrite=False)

    assert data.shape == (3, 1, 5, 7)
    assert np.array_equal(data, expected)
    assert np.array_equal(data[:, 0], rgb.transpose(2, 0, 1))
    assert properties == expected_properties == {"spacing": (999.0, 1.0, 1.0)}
    assert sidecar is None
    assert np.array_equal(image_io.imread(str(output)), labels[0])
    with pytest.raises(ContractError, match="already exists"):
        write_segmentation("NaturalImage2DIO", labels, output, properties, overwrite=False)


@pytest.mark.oracle
def test_tiff_reader_writer_preserves_spacing_sidecar(tmp_path: Path) -> None:
    """Preserve volumetric TIFF spacing through the JSON sidecar."""
    tifffile = import_module("tifffile")
    source = tmp_path / "case_0000.tif"
    raw = np.arange(3 * 4 * 5, dtype=np.uint16).reshape(3, 4, 5)
    tifffile.imwrite(str(source), raw, photometric="minisblack")
    (tmp_path / "case.json").write_text('{"spacing": [2.5, 0.8, 0.7]}\n', encoding="utf-8")

    data, properties = read_images("Tiff3DIO", (source,))
    oracle_reader = pytest.importorskip(
        "nnunetv2.imageio.tif_reader_writer",
        reason="requires the optional 'oracle' extra",
        exc_type=ModuleNotFoundError,
    ).Tiff3DIO()
    expected, expected_properties = oracle_reader.read_images((str(source),))
    output = tmp_path / "prediction.tif"
    labels = np.asarray(data[0] > 20, dtype=np.uint16)
    sidecar = write_segmentation("Tiff3DIO", labels, output, properties, overwrite=False)

    assert data.shape == (1, 3, 4, 5)
    assert np.array_equal(data, expected)
    assert np.array_equal(data[0], raw)
    assert properties == {"spacing": (2.5, 0.8, 0.7)}
    assert np.allclose(
        np.asarray(properties["spacing"]),
        np.asarray(expected_properties["spacing"]),
    )
    assert sidecar == tmp_path / "prediction.json"
    assert image_reader_output_sidecar("Tiff3DIO", output) == sidecar
    assert np.array_equal(tifffile.imread(str(output)), labels)
    with tifffile.TiffFile(output) as image:
        assert image.pages[0].compression == tifffile.COMPRESSION.ADOBE_DEFLATE
    assert sidecar is not None
    assert sidecar.read_text(encoding="utf-8") == (
        '{\n  "spacing": [\n    2.5,\n    0.8,\n    0.7\n  ]\n}\n'
    )


@pytest.mark.oracle
@pytest.mark.parametrize("reorient", [False, True])
def test_nibabel_reader_writer_matches_pinned_oracle(tmp_path: Path, reorient: bool) -> None:
    """Compare Nibabel input and output behavior with the pinned source."""
    nibabel = import_module("nibabel")
    source_module = pytest.importorskip(
        "nnunetv2.imageio.nibabel_reader_writer",
        reason="requires the optional 'oracle' extra",
        exc_type=ModuleNotFoundError,
    )
    source = tmp_path / "case_0000.nii.gz"
    raw = np.arange(4 * 5 * 6, dtype=np.int16).reshape(4, 5, 6)
    affine = np.asarray(
        [[-0.7, 0, 0, 11], [0, 0.8, 0, -3], [0, 0, 2.5, 7], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    nibabel.save(nibabel.Nifti1Image(raw, affine), str(source))
    reader = "NibabelIOWithReorient" if reorient else "NibabelIO"
    oracle_reader = getattr(source_module, reader)()

    expected, expected_properties = oracle_reader.read_images((str(source),))
    actual, properties = read_images(reader, (source,))
    labels = np.asarray(actual[0] % 3, dtype=np.int16)
    output = tmp_path / "prediction.nii.gz"
    write_segmentation(reader, labels, output, properties, overwrite=False)

    assert np.array_equal(actual, expected)
    assert np.allclose(
        np.asarray(properties["spacing"]),
        np.asarray(expected_properties["spacing"]),
    )
    expected_geometry = cast(dict[str, object], expected_properties["nibabel_stuff"])
    actual_geometry = cast(dict[str, object], properties["nibabel_stuff"])
    assert actual_geometry.keys() == expected_geometry.keys()
    for key in actual_geometry:
        assert np.allclose(np.asarray(actual_geometry[key]), np.asarray(expected_geometry[key]))
    assert np.array_equal(np.asarray(nibabel.load(str(output)).dataobj), raw % 3)
