"""Torch-free implementations of the pinned nnU-Net image I/O registry."""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

from nnunet_mlx.contracts import parse_mapping as _mapping
from nnunet_mlx.contracts import parse_number
from nnunet_mlx.errors import ContractError

type HostArray = npt.NDArray[np.generic]

_SUPPORTED_ENDINGS = {
    "NaturalImage2DIO": (".png", ".bmp", ".tif", ".tiff"),
    "SimpleITKIO": (".nii.gz", ".nrrd", ".mha", ".gipl"),
    "Tiff3DIO": (".tif", ".tiff"),
    "NibabelIO": (".nii", ".nii.gz"),
    "NibabelIOWithReorient": (".nii", ".nii.gz"),
}


def resolve_image_reader(name: str, *, file_ending: str | None = None) -> str:
    """Resolve one of nnU-Net v2.8.1's built-in reader/writer classes."""

    class_name = name.rsplit(".", 1)[-1]
    supported = _SUPPORTED_ENDINGS.get(class_name)
    if supported is None:
        raise ContractError(f"unsupported nnU-Net image reader {name!r}")
    if file_ending is not None and file_ending.casefold() not in supported:
        raise ContractError(
            f"nnU-Net image reader {class_name!r} does not support {file_ending!r}"
        )
    return class_name


def image_reader_output_sidecar(
    reader_name: str, segmentation_file: str | Path
) -> Path | None:
    """Return the additional geometry file emitted by a built-in writer."""

    reader = resolve_image_reader(reader_name)
    target = Path(segmentation_file)
    return target.with_suffix(".json") if reader == "Tiff3DIO" else None


def image_reader_input_sidecars(
    reader_name: str, image_files: Sequence[str | Path]
) -> tuple[Path, ...]:
    """Return reader-consumed sidecar files that change how the images are read."""

    reader = resolve_image_reader(reader_name)
    if reader != "Tiff3DIO":
        return ()
    paths = _input_paths(image_files)
    return () if not paths else (_tiff_image_spacing_path(paths[0]),)


def segmentation_reader_input_sidecar(
    reader_name: str, segmentation_file: str | Path
) -> Path | None:
    """Return the reader-consumed sidecar that changes how a label map is read."""

    reader = resolve_image_reader(reader_name)
    return Path(segmentation_file).with_suffix(".json") if reader == "Tiff3DIO" else None


def read_images(
    reader_name: str,
    image_files: Sequence[str | Path],
) -> tuple[npt.NDArray[np.float32], dict[str, object]]:
    """Read modalities through the reader named in an nnU-Net plans file.

    Returns finite float32 data in ``(C, D, H, W)`` array order and a properties
    mapping containing spacing in that spatial order and reader-specific geometry.
    Complex input images are rejected before any real-valued conversion.
    Two-dimensional images use a singleton depth. For TIFF modalities, spacing
    comes from the first modality's case JSON sidecar, defaulting to unit spacing
    with a warning if absent.
    """

    reader = resolve_image_reader(reader_name)
    if reader == "SimpleITKIO":
        return read_sitk_images(image_files)
    if reader == "NaturalImage2DIO":
        data, properties = _read_natural_images(image_files)
        return cast(npt.NDArray[np.float32], data), properties
    if reader == "Tiff3DIO":
        return _read_tiff_images(image_files)
    data, properties = _read_nibabel_images(
        image_files,
        reorient=reader == "NibabelIOWithReorient",
    )
    return cast(npt.NDArray[np.float32], data), properties


def read_segmentation(
    reader_name: str,
    segmentation_file: str | Path,
    expected_properties: Mapping[str, object],
) -> HostArray:
    """Read one non-negative integer label channel and check its case geometry.

    Returns contiguous int32 data in ``(1, D, H, W)`` array order. The properties
    must match ``expected_properties`` according to the selected reader's geometry
    checks; the segmentation's properties are not returned. Labels are read
    without float32 rounding and must fit the nonnegative int32 range.
    """

    reader = resolve_image_reader(reader_name)
    data, properties = _read_label_image(reader, segmentation_file)
    labels = _validated_labels(data, name="previous-stage segmentation")
    require_matching_geometry(reader, properties, expected_properties)
    return labels


def read_exported_segmentation(
    reader_name: str, segmentation_file: str | Path
) -> tuple[HostArray, dict[str, object]]:
    """Read a segmentation this package itself exported, with its own geometry.

    Unlike :func:`read_segmentation` (which validates a previous-stage
    segmentation against another case's already-known ``expected_properties``
    and discards its own), this returns the file's own geometry for a
    write-back -- the ``nnUNetv2_apply_postprocessing`` read/modify/write
    round trip, where there is no second geometry source to validate against.
    Labels must be finite, integral and within the nonnegative int32 range.
    """

    reader = resolve_image_reader(reader_name)
    data, properties = _read_label_image(reader, segmentation_file)
    return _validated_labels(data, name="segmentation"), properties


def _read_label_image(reader: str, path: str | Path) -> tuple[HostArray, dict[str, object]]:
    """Read one label image without the intensity readers' float32 conversion."""
    if reader == "SimpleITKIO":
        return _read_sitk_images((path,), label_data=True)
    if reader == "Tiff3DIO":
        return _read_tiff_segmentation(path)
    if reader == "NaturalImage2DIO":
        return _read_natural_images((path,), label_data=True)
    return _read_nibabel_images(
        (path,), reorient=reader == "NibabelIOWithReorient", label_data=True,
    )


def _validated_labels(data: HostArray, *, name: str) -> HostArray:
    """Require a single channel of finite, nonnegative int32-range integer labels."""
    if (data.shape[0] != 1 or data.dtype.kind not in "biuf"
            or not np.isfinite(data).all() or not np.equal(data, np.rint(data)).all()):
        raise ContractError(f"{name} must contain integer labels")
    values = cast(npt.NDArray[np.bool_ | np.integer | np.floating], data)
    if np.any(values < 0):
        raise ContractError(f"{name} labels must be non-negative")
    if np.any(values > np.iinfo(np.int32).max):
        raise ContractError(f"{name} labels must fit in int32")
    return np.ascontiguousarray(data, dtype=np.int32)


def write_segmentation(
    reader_name: str,
    segmentation: HostArray,
    output_file: str | Path,
    properties: Mapping[str, object],
    *,
    overwrite: bool,
) -> Path | None:
    """Write a label map through the model folder's configured writer.

    ``segmentation`` is a three-dimensional integer array in reader spatial
    order, with a singleton first axis for 2-D images. Labels must be between
    0 and 65535; output uses uint8 when the maximum is below 255, otherwise
    uint16. Reader-specific properties supply the output geometry.
    Returns the path of the TIFF geometry JSON sidecar, or ``None`` for other
    writers. If ``overwrite`` is false, an existing image or sidecar is rejected.
    Parent directories are created by the selected writer.
    """

    reader = resolve_image_reader(reader_name)
    target = Path(output_file)
    sidecar = image_reader_output_sidecar(reader, target)
    outputs = (target,) if sidecar is None else (target, sidecar)
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise ContractError(f"segmentation output already exists: {existing}")
    if any(path.exists() and not path.is_file() for path in outputs):
        raise ContractError("segmentation output path is not a file")
    if reader == "SimpleITKIO":
        write_sitk_segmentation(segmentation, target, properties, overwrite=overwrite)
    elif reader == "NaturalImage2DIO":
        _write_natural_segmentation(segmentation, target)
    elif reader == "Tiff3DIO":
        if sidecar is None:
            raise AssertionError("TIFF writer unexpectedly lacks a sidecar")
        _write_tiff_segmentation(segmentation, target, sidecar, properties)
    else:
        _write_nibabel_segmentation(
            segmentation,
            target,
            properties,
            reorient=reader == "NibabelIOWithReorient",
        )
    return sidecar


def read_sitk_images(
    image_files: Sequence[str | Path],
) -> tuple[npt.NDArray[np.float32], dict[str, object]]:
    """Read 2-D, 3-D or 4-D SimpleITK inputs in upstream channel-first layout.

    For 4-D inputs the leading array dimension supplies channels; spatial
    spacing excludes the fourth image dimension.
    """

    data, properties = _read_sitk_images(image_files, label_data=False)
    return cast(npt.NDArray[np.float32], data), properties


def _read_sitk_images(
    image_files: Sequence[str | Path], *, label_data: bool,
) -> tuple[HostArray, dict[str, object]]:
    """Read co-registered SimpleITK files into ``(C, Z, Y, X)`` host arrays.

    Each 2-D file becomes one channel with a singleton Z axis and the upstream
    placeholder spacing ``(max_spacing * 999, sy, sx)``; 4-D files already carry
    channels. Spacing is reversed from SimpleITK's (x, y, z) to nnU-Net's (z, y, x).
    All files must share shape and geometry. The raw SimpleITK spacing, origin and
    direction are kept under ``sitk_stuff`` for export.
    """

    paths = tuple(Path(path) for path in image_files)
    if not paths:
        raise ContractError("raw prediction requires at least one image file")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise ContractError(f"input image files do not exist: {missing}")
    sitk = import_module("SimpleITK")
    arrays: list[HostArray] = []
    spacings: list[tuple[float, ...]] = []
    origins: list[tuple[float, ...]] = []
    directions: list[tuple[float, ...]] = []
    nnunet_spacings: list[tuple[float, float, float]] = []
    for path in paths:
        image = sitk.ReadImage(str(path))
        spacing = tuple(float(value) for value in image.GetSpacing())
        origin = tuple(float(value) for value in image.GetOrigin())
        direction = tuple(float(value) for value in image.GetDirection())
        array = np.asarray(sitk.GetArrayFromImage(image))
        if array.ndim == 2:
            array = array[None, None]
            maximum = max(spacing)
            nnunet_spacing = (maximum * 999, *reversed(spacing))
        elif array.ndim == 3:
            array = array[None]
            nnunet_spacing = tuple(reversed(spacing))
        elif array.ndim == 4:
            nnunet_spacing = tuple(reversed(spacing))[1:]
        else:
            raise ContractError(
                f"SimpleITK input {path.name} must be 2-D, 3-D, or 4-D, "
                f"got {array.ndim}-D"
            )
        arrays.append(array)
        spacings.append(spacing)
        origins.append(origin)
        directions.append(direction)
        nnunet_spacings.append(cast(tuple[float, float, float], nnunet_spacing))
    _require_same(arrays, spacings, origins, directions, nnunet_spacings)
    data = np.vstack(arrays) if label_data else _stack_finite(arrays)
    properties: dict[str, object] = {
        "sitk_stuff": {
            "spacing": spacings[0],
            "origin": origins[0],
            "direction": directions[0],
        },
        "spacing": nnunet_spacings[0],
    }
    return data, properties


def read_sitk_segmentation(
    segmentation_file: str | Path,
    expected_properties: Mapping[str, object],
) -> HostArray:
    """Read one integer label map and compare its spatial geometry to the case.

    Spacing, origin and direction are compared with NumPy's ``allclose``
    tolerances. A 4-D case must not couple its spatial and channel axes.
    """

    data, properties = _read_sitk_images((segmentation_file,), label_data=True)
    labels = _validated_labels(data, name="previous-stage segmentation")
    require_matching_geometry("SimpleITKIO", properties, expected_properties)
    return labels


def write_sitk_segmentation(
    segmentation: HostArray,
    output_file: str | Path,
    properties: Mapping[str, object],
    *,
    overwrite: bool,
) -> None:
    """Write a label volume with the geometry stored in ``properties["sitk_stuff"]``.

    ``segmentation`` is ``(Z, Y, X)``; 2-D geometry requires Z == 1. Existing
    targets raise unless ``overwrite`` is set. Output is compressed.
    """
    target = Path(output_file)
    if target.exists() and not overwrite:
        raise ContractError(f"segmentation output already exists: {target}")
    if target.exists() and not target.is_file():
        raise ContractError(f"segmentation output is not a file: {target}")
    labels = _output_labels(segmentation)
    spacing, origin, direction = _sitk_spatial_geometry(properties)
    dimensions = len(spacing)
    if dimensions == 2:
        if labels.shape[0] != 1:
            raise ContractError("2-D segmentation export requires singleton depth")
        labels = labels[0]
    sitk = import_module("SimpleITK")
    image = sitk.GetImageFromArray(labels.astype(label_dtype(labels), copy=False))
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image.SetDirection(direction)
    target.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(target), True)


def _read_natural_images(
    image_files: Sequence[str | Path], *, label_data: bool = False,
) -> tuple[HostArray, dict[str, object]]:
    """Read PNG/BMP/TIFF slices as ``(C, 1, H, W)``; RGB(A) maps to channels.

    No physical spacing exists, so the upstream placeholder ``(999, 1, 1)`` is used.
    """
    paths = _input_paths(image_files)
    image_io = import_module("skimage.io")
    arrays: list[HostArray] = []
    for path in paths:
        array = np.asarray(image_io.imread(str(path)))
        if array.ndim == 3:
            if int(array.shape[-1]) not in (3, 4):
                raise ContractError(
                    f"natural image {path.name} must have RGB or RGBA channels"
                )
            arrays.append(array.transpose(2, 0, 1)[:, None])
        elif array.ndim == 2:
            arrays.append(array[None, None])
        else:
            raise ContractError(
                f"natural image {path.name} must be 2-D grayscale, RGB, or RGBA"
            )
    _require_same_shapes(arrays)
    return (np.vstack(arrays) if label_data else _stack_finite(arrays)), {
        "spacing": (999.0, 1.0, 1.0)
    }


def _write_natural_segmentation(segmentation: HostArray, target: Path) -> None:
    """Write a singleton-depth label map as a 2D image."""
    labels = _output_labels(segmentation)
    if labels.shape[0] != 1:
        raise ContractError("2-D segmentation export requires singleton depth")
    target.parent.mkdir(parents=True, exist_ok=True)
    output = labels[0].astype(label_dtype(labels), copy=False)
    if target.suffix.casefold() in (".tif", ".tiff"):
        tifffile = import_module("tifffile")
        tifffile.imwrite(str(target), output, compression="packbits")
    else:
        image_io = import_module("skimage.io")
        image_io.imsave(str(target), output, check_contrast=False)


def _read_tiff_images(
    image_files: Sequence[str | Path],
) -> tuple[npt.NDArray[np.float32], dict[str, object]]:
    """Read 3D TIFF modalities and their shared spacing sidecar."""
    paths = _input_paths(image_files)
    arrays = [_read_tiff_array(path)[None] for path in paths]
    _require_same_shapes(arrays)
    spacing = _read_tiff_spacing(_tiff_image_spacing_path(paths[0]))
    return _stack_finite(arrays), {"spacing": spacing}


def _read_tiff_segmentation(
    segmentation_file: str | Path,
) -> tuple[HostArray, dict[str, object]]:
    """Read one 3D TIFF segmentation and its spacing sidecar."""
    path = _input_paths((segmentation_file,))[0]
    array = _read_tiff_array(path)[None]
    spacing = _read_tiff_spacing(path.with_suffix(".json"))
    return array, {"spacing": spacing}


def _read_tiff_array(path: Path) -> HostArray:
    """Read a TIFF volume, rejecting arrays that are not three-dimensional."""
    tifffile = import_module("tifffile")
    array = np.asarray(tifffile.imread(str(path)))
    if array.ndim != 3:
        raise ContractError(f"TIFF input {path.name} must be 3-D")
    return array


def _tiff_image_spacing_path(path: Path) -> Path:
    """Derive the shared spacing JSON path from a four-digit modality filename."""
    stem = path.stem
    if len(stem) < 5 or stem[-5] != "_" or not stem[-4:].isdigit():
        raise ContractError(
            f"TIFF modality {path.name} must end in a four-digit channel suffix"
        )
    return path.with_name(f"{stem[:-5]}.json")


def _read_tiff_spacing(path: Path) -> tuple[float, float, float]:
    """Read ``{"spacing": [z, y, x]}`` from the JSON sidecar.

    A missing sidecar warns and falls back to upstream's ``(1, 1, 1)``; a
    malformed one raises ``ContractError``.
    """
    if not path.is_file():
        warnings.warn(
            f"TIFF spacing sidecar {path.name} is missing; using upstream default (1, 1, 1)",
            RuntimeWarning,
            stacklevel=2,
        )
        return (1.0, 1.0, 1.0)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"TIFF spacing sidecar is invalid: {path}") from error
    root = _mapping(payload, name="TIFF spacing sidecar")
    return _three_positive_floats(root.get("spacing"), name="TIFF spacing")


def _write_tiff_segmentation(
    segmentation: HostArray,
    target: Path,
    sidecar: Path,
    properties: Mapping[str, object],
) -> None:
    """Write a zlib-compressed TIFF and the JSON spacing sidecar read back on load."""
    labels = _output_labels(segmentation)
    spacing = _three_positive_floats(
        properties.get("spacing"), name="TIFF spacing"
    )
    tifffile = import_module("tifffile")
    target.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        str(target),
        data=labels.astype(label_dtype(labels), copy=False),
        compression="zlib",
        photometric="minisblack",
    )
    sidecar.write_text(
        json.dumps({"spacing": list(spacing)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_nibabel_images(
    image_files: Sequence[str | Path],
    *,
    reorient: bool,
    label_data: bool = False,
) -> tuple[HostArray, dict[str, object]]:
    """Read 3-D NIfTI files as ``(C, Z, Y, X)`` (nibabel's x, y, z transposed).

    With ``reorient`` volumes are first brought to the closest canonical (RAS+)
    axis order; both original and reoriented affines are stored under
    ``nibabel_stuff`` so export can restore the source orientation. Spacing is the
    header zooms reversed to (z, y, x). Shape and spacing must match across files.
    """
    paths = _input_paths(image_files)
    nibabel = import_module("nibabel")
    orientations = import_module("nibabel.orientations")
    arrays: list[HostArray] = []
    original_affines: list[npt.NDArray[np.float64]] = []
    active_affines: list[npt.NDArray[np.float64]] = []
    spacings: list[tuple[float, float, float]] = []
    for path in paths:
        image = nibabel.load(str(path))
        if int(image.ndim) != 3:
            raise ContractError(f"Nibabel input {path.name} must be 3-D")
        if np.dtype(image.get_data_dtype()).kind == "c":
            raise ContractError("input images must contain real values")
        original_affine = _affine(image.affine, name=f"{path.name} affine")
        active_image = (
            image.as_reoriented(orientations.io_orientation(original_affine))
            if reorient
            else image
        )
        active_affine = _affine(
            active_image.affine, name=f"{path.name} reoriented affine"
        )
        zooms = tuple(float(value) for value in active_image.header.get_zooms())
        spacing = _three_positive_floats(
            zooms[::-1], name=f"{path.name} spacing"
        )
        arrays.append(np.asarray(active_image.get_fdata()).transpose(2, 1, 0)[None])
        original_affines.append(original_affine)
        active_affines.append(active_affine)
        spacings.append(spacing)
    _require_same_shapes(arrays)
    _require_same_values(spacings, name="spacing")
    _warn_different_affines(active_affines, paths)
    geometry: dict[str, object] = {"original_affine": original_affines[0]}
    if reorient:
        geometry["reoriented_affine"] = active_affines[0]
    return (np.vstack(arrays) if label_data else _stack_finite(arrays)), {
        "nibabel_stuff": geometry,
        "spacing": spacings[0],
    }


def _write_nibabel_segmentation(
    segmentation: HostArray,
    target: Path,
    properties: Mapping[str, object],
    *,
    reorient: bool,
) -> None:
    """Write ``(Z, Y, X)`` labels as NIfTI, undoing the reading reorientation.

    With ``reorient`` the volume carries the reoriented affine and is reoriented
    back to the source axis order; a warning is issued if the resulting affine
    differs from the stored original.
    """
    labels = _output_labels(segmentation).transpose(2, 1, 0)
    geometry = _mapping(properties.get("nibabel_stuff"), name="nibabel_stuff")
    original_affine = _affine(
        geometry.get("original_affine"), name="Nibabel original affine"
    )
    nibabel = import_module("nibabel")
    if reorient:
        reoriented_affine = _affine(
            geometry.get("reoriented_affine"), name="Nibabel reoriented affine"
        )
        image = nibabel.Nifti1Image(
            labels.astype(label_dtype(labels), copy=False),
            affine=reoriented_affine,
        )
        orientations = import_module("nibabel.orientations")
        image_orientation = orientations.io_orientation(original_affine)
        ras_orientation = orientations.axcodes2ornt("RAS")
        from_canonical = orientations.ornt_transform(
            ras_orientation, image_orientation
        )
        image = image.as_reoriented(from_canonical)
        if not np.allclose(original_affine, image.affine):
            warnings.warn(
                f"restored Nibabel affine does not match the source for {target.name}",
                RuntimeWarning,
                stacklevel=2,
            )
    else:
        image = nibabel.Nifti1Image(
            labels.astype(label_dtype(labels), copy=False),
            affine=original_affine,
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    nibabel.save(image, str(target))


def _input_paths(image_files: Sequence[str | Path]) -> tuple[Path, ...]:
    """Require at least one existing input image file."""
    paths = tuple(Path(path) for path in image_files)
    if not paths:
        raise ContractError("raw prediction requires at least one image file")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise ContractError(f"input image files do not exist: {missing}")
    return paths


def _stack_finite(arrays: Sequence[HostArray]) -> npt.NDArray[np.float32]:
    """Stack real input modalities as float32 and reject nonfinite values."""
    if any(array.dtype.kind == "c" for array in arrays):
        raise ContractError("input images must contain real values")
    data = np.vstack(arrays, dtype=np.float32, casting="unsafe")
    if not np.isfinite(data).all():
        raise ContractError("input images contain non-finite values")
    return data


def _require_same_shapes(arrays: Sequence[HostArray]) -> None:
    """Reject input modalities with different array shapes."""
    shapes = [array.shape for array in arrays]
    if any(shape != shapes[0] for shape in shapes[1:]):
        raise ContractError(f"input modalities have different shapes: {shapes}")


def _require_same_values(
    values: Sequence[Sequence[float]], *, name: str
) -> None:
    """Reject modality metadata values that do not agree numerically."""
    if any(not np.allclose(values[0], value) for value in values[1:]):
        raise ContractError(f"input modalities have different {name}")


def _warn_different_affines(
    affines: Sequence[npt.NDArray[np.float64]], paths: Sequence[Path]
) -> None:
    """Warn when Nibabel modality affines differ."""
    if any(not np.allclose(affines[0], affine) for affine in affines[1:]):
        warnings.warn(
            f"input modalities have different Nibabel affines: {[path.name for path in paths]}",
            RuntimeWarning,
            stacklevel=2,
        )


def require_matching_geometry(
    reader: str,
    actual: Mapping[str, object],
    expected: Mapping[str, object],
) -> None:
    """Require physical geometry in the reader's canonical coordinate convention."""

    reader = resolve_image_reader(reader)
    if reader == "SimpleITKIO":
        _require_matching_properties(actual, expected)
        return
    actual_spacing = _positive_float_tuple(actual.get("spacing"), name="spacing")
    expected_spacing = _positive_float_tuple(expected.get("spacing"), name="spacing")
    if not np.allclose(actual_spacing, expected_spacing):
        raise ContractError("segmentation has different spacing")
    if not reader.startswith("NibabelIO"):
        return
    actual_geometry = _mapping(actual.get("nibabel_stuff"), name="nibabel_stuff")
    expected_geometry = _mapping(expected.get("nibabel_stuff"), name="nibabel_stuff")
    keys = (
        ("original_affine", "reoriented_affine")
        if reader == "NibabelIOWithReorient"
        else ("original_affine",)
    )
    for key in keys:
        if not np.allclose(
            _affine(actual_geometry.get(key), name=key),
            _affine(expected_geometry.get(key), name=key),
        ):
            raise ContractError(
                f"segmentation has different {key.replace('_', ' ')}"
            )


def _affine(value: object, *, name: str) -> npt.NDArray[np.float64]:
    """Parse a finite 4 by 4 affine matrix."""
    affine = np.asarray(value, dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ContractError(f"{name} must be a finite 4x4 matrix")
    return affine


def _output_labels(segmentation: HostArray) -> npt.NDArray[np.int64]:
    """Require a 3D integer label map with values in the uint16 range."""
    raw_labels = np.asarray(segmentation)
    if raw_labels.ndim != 3 or not np.issubdtype(raw_labels.dtype, np.integer):
        raise ContractError("export segmentation must be an integer XYZ array")
    integer_labels = cast(npt.NDArray[np.integer], raw_labels)
    if np.any(integer_labels > np.iinfo(np.uint16).max):
        raise ContractError("export segmentation labels must fit in uint16")
    labels = np.asarray(raw_labels, dtype=np.int64)
    if np.any(labels < 0):
        raise ContractError("export segmentation labels must be non-negative")
    return labels


def label_dtype(
    labels: npt.NDArray[np.int64] | npt.NDArray[np.uint16],
) -> type[np.uint8] | type[np.uint16]:
    """Select uint8 when the maximum label is below 255, otherwise uint16."""
    return np.uint8 if int(labels.max(initial=0)) < 255 else np.uint16


def _positive_float_tuple(value: object, *, name: str) -> tuple[float, ...]:
    """Parse a numeric tuple and require every value to be positive."""
    values = _float_tuple(value, name=name)
    if any(item <= 0 for item in values):
        raise ContractError(f"{name} must contain positive numbers")
    return values


def _three_positive_floats(
    value: object, *, name: str
) -> tuple[float, float, float]:
    """Require exactly three positive numeric values."""
    values = _positive_float_tuple(value, name=name)
    if len(values) != 3:
        raise ContractError(f"{name} must contain three values")
    return values[0], values[1], values[2]


def _require_same(
    arrays: Sequence[HostArray],
    spacings: Sequence[tuple[float, ...]],
    origins: Sequence[tuple[float, ...]],
    directions: Sequence[tuple[float, ...]],
    nnunet_spacings: Sequence[tuple[float, float, float]],
) -> None:
    """Require matching modality shapes, spacing, origins and directions."""
    shapes = [array.shape for array in arrays]
    if any(shape != shapes[0] for shape in shapes[1:]):
        raise ContractError(f"input modalities have different shapes: {shapes}")
    for name, values in (
        ("spacing", spacings),
        ("origin", origins),
        ("direction", directions),
        ("nnU-Net spacing", nnunet_spacings),
    ):
        if any(not np.allclose(values[0], value) for value in values[1:]):
            raise ContractError(f"input modalities have different {name}")


def _require_matching_properties(
    actual: Mapping[str, object], expected: Mapping[str, object]
) -> None:
    """Require matching SimpleITK spacing, origin and direction."""
    actual_values = _sitk_spatial_geometry(actual)
    expected_values = _sitk_spatial_geometry(expected)
    for name, actual_value, expected_value in zip(
        ("spacing", "origin", "direction"),
        actual_values,
        expected_values,
        strict=True,
    ):
        if not np.allclose(actual_value, expected_value):
            raise ContractError(f"segmentation has different {name}")


def _sitk_spatial_geometry(
    properties: Mapping[str, object],
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Validate and return ``(spacing, origin, direction)`` from ``sitk_stuff``.

    4-D geometry (modality as the fourth axis) is reduced to its spatial 3-D part
    and must not couple modalities to spatial axes.
    """
    geometry = _mapping(properties.get("sitk_stuff"), name="sitk_stuff")
    spacing = _float_tuple(geometry.get("spacing"), name="SimpleITK spacing")
    origin = _float_tuple(geometry.get("origin"), name="SimpleITK origin")
    direction = _float_tuple(geometry.get("direction"), name="SimpleITK direction")
    dimensions = len(spacing)
    if dimensions not in (2, 3, 4) or len(origin) != dimensions:
        raise ContractError("SimpleITK geometry must be 2-D, 3-D, or 4-D")
    if len(direction) != dimensions * dimensions:
        raise ContractError("SimpleITK direction rank does not match its geometry")
    if dimensions < 4:
        return spacing, origin, direction
    matrix = np.asarray(direction, dtype=np.float64).reshape(4, 4)
    if not np.allclose(matrix[:3, 3], 0) or not np.allclose(matrix[3, :3], 0):
        raise ContractError(
            "4-D SimpleITK geometry couples modalities to spatial axes"
        )
    return spacing[:3], origin[:3], tuple(matrix[:3, :3].ravel())


def _float_tuple(value: object, *, name: str) -> tuple[float, ...]:
    """Parse a sequence of finite numeric metadata values."""
    if not isinstance(value, list | tuple):
        raise ContractError(f"{name} must be an array")
    result: list[float] = []
    for item in cast(Sequence[object], value):
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise ContractError(f"{name} must contain numbers")
        result.append(parse_number(item, name=name))
    return tuple(result)
