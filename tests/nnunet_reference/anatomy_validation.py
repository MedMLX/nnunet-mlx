"""Semantic geometry checks for known-normal reference segmentations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Literal, cast

import numpy as np
import numpy.typing as npt

from nnunet_mlx.errors import ContractError

type AxisName = Literal["x", "y", "z"]
type HostArray = npt.NDArray[np.generic]
type Float64Array = npt.NDArray[np.float64]

_AXIS_INDEX: dict[AxisName, int] = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True, slots=True)
class LabelAxisOrderResult:
    lower_label: int
    upper_label: int
    axis: AxisName
    lower_coordinate_mm: float
    upper_coordinate_mm: float
    separation_mm: float
    minimum_separation_mm: float

    def to_payload(self) -> dict[str, object]:
        """Serialize the world-coordinate label-order result."""
        return asdict(self)


def label_centroid_world(
    segmentation: HostArray,
    properties: Mapping[str, object],
    *,
    label: int,
) -> tuple[float, float, float]:
    """Return a label centroid as ``(x, y, z)`` in mm in the SimpleITK physical frame.

    ``segmentation`` is a 3-D integer-valued label map in array order (Z, Y, X), as read by
    ``SimpleITKIO``; ``properties["sitk_stuff"]`` supplies spacing, origin and
    direction. The centroid is the mean voxel index of ``label`` mapped through
    ``origin + direction @ (index * spacing)`` (SimpleITK's convention is LPS: world X
    increases toward the patient's left). This establishes the centroid of the
    supplied label map only; it does not establish anatomical correctness.

    Raises:
        ContractError: If the label map or geometry is invalid or the label is absent.
    """

    values = _label_map(segmentation)
    if label < 0:
        raise ContractError("anatomic geometry labels must be non-negative")
    coordinates = np.nonzero(values == label)
    if not coordinates[0].size:
        raise ContractError(f"anatomic geometry label {label} is absent")

    index_xyz = np.asarray(
        [
            np.mean(coordinates[2], dtype=np.float64),
            np.mean(coordinates[1], dtype=np.float64),
            np.mean(coordinates[0], dtype=np.float64),
        ],
        dtype=np.float64,
    )
    spacing, origin, direction = _sitk_geometry(properties)
    physical = origin + direction @ (index_xyz * spacing)
    return cast(
        tuple[float, float, float],
        tuple(float(value) for value in physical),
    )


def require_label_axis_order(
    segmentation: HostArray,
    properties: Mapping[str, object],
    *,
    lower_label: int,
    upper_label: int,
    axis: AxisName,
    minimum_separation_mm: float = 0.0,
) -> LabelAxisOrderResult:
    """Fail unless the ``upper_label`` centroid lies beyond the ``lower_label`` centroid.

    Compares world coordinates (mm) along ``axis`` ("x", "y" or "z" of the SimpleITK
    physical frame, see ``label_centroid_world``). The check passes only when
    ``upper - lower`` is strictly greater than ``minimum_separation_mm``.

    Raises:
        ContractError: If the arguments or geometry are invalid, a label is absent,
            or the ordering is violated.
    """

    if lower_label == upper_label:
        raise ContractError("anatomic geometry ordering requires two distinct labels")
    if not np.isfinite(minimum_separation_mm) or minimum_separation_mm < 0:
        raise ContractError("minimum anatomic separation must be finite and non-negative")
    axis_index = _AXIS_INDEX.get(axis)
    if axis_index is None:
        raise ContractError(f"unsupported anatomic geometry axis {axis!r}")

    lower = label_centroid_world(segmentation, properties, label=lower_label)
    upper = label_centroid_world(segmentation, properties, label=upper_label)
    separation = upper[axis_index] - lower[axis_index]
    if separation <= minimum_separation_mm:
        raise ContractError(
            "anatomic axis-order check failed: "
            f"label {lower_label} at {lower[axis_index]:.3f} mm must be more than "
            f"{minimum_separation_mm:.3f} mm below label {upper_label} at "
            f"{upper[axis_index]:.3f} mm on world-{axis}"
        )
    return LabelAxisOrderResult(
        lower_label=lower_label,
        upper_label=upper_label,
        axis=axis,
        lower_coordinate_mm=lower[axis_index],
        upper_coordinate_mm=upper[axis_index],
        separation_mm=separation,
        minimum_separation_mm=minimum_separation_mm,
    )


def _label_map(segmentation: HostArray) -> HostArray:
    """Require an integer-valued 3D label map for anatomical checks."""
    values = np.asarray(segmentation)
    if values.ndim != 3:
        raise ContractError("anatomic geometry validation requires a 3-D ZYX label map")
    if not np.isfinite(values).all():
        raise ContractError("anatomic geometry segmentation contains non-finite values")
    integer_dtype = np.issubdtype(values.dtype, np.integer)
    integer_valued = np.equal(values, np.rint(values)).all()
    if not integer_dtype and not integer_valued:
        raise ContractError("anatomic geometry segmentation must contain integer labels")
    return values


def _sitk_geometry(
    properties: Mapping[str, object],
) -> tuple[Float64Array, Float64Array, Float64Array]:
    """Read and validate SimpleITK spacing, origin and direction."""
    raw = properties.get("sitk_stuff")
    if not isinstance(raw, Mapping):
        raise ContractError(
            "anatomic geometry validation requires SimpleITK spatial metadata"
        )
    metadata = cast(Mapping[object, object], raw)
    spacing = np.asarray(
        _finite_floats(metadata.get("spacing"), count=3, name="SimpleITK spacing"),
        dtype=np.float64,
    )
    if np.any(spacing <= 0):
        raise ContractError("SimpleITK spacing must be positive")
    origin = np.asarray(
        _finite_floats(metadata.get("origin"), count=3, name="SimpleITK origin"),
        dtype=np.float64,
    )
    direction = np.asarray(
        _finite_floats(
            metadata.get("direction"),
            count=9,
            name="SimpleITK direction",
        ),
        dtype=np.float64,
    ).reshape(3, 3)
    if abs(float(np.linalg.det(direction))) < 1e-8:
        raise ContractError("SimpleITK direction matrix must be invertible")
    return spacing, origin, direction


def _finite_floats(value: object, *, count: int, name: str) -> tuple[float, ...]:
    """Parse finite numeric geometry values with the expected length."""
    if not isinstance(value, list | tuple):
        raise ContractError(f"{name} must contain {count} values")
    items = cast(Sequence[object], value)
    if len(items) != count:
        raise ContractError(f"{name} must contain {count} values")
    result: list[float] = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise ContractError(f"{name} must contain numeric values")
        converted = float(item)
        if not np.isfinite(converted):
            raise ContractError(f"{name} must contain finite values")
        result.append(converted)
    return tuple(result)
