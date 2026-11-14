"""Append a full-volume NIfTI comparison to a versioned qualification matrix.

Compare predictions on identical source grids. Dice is prediction agreement when
the reference is upstream or previous MLX output. Surface distances use Euclidean
distances between six-neighbor boundary voxel centers under the NIfTI affine;
components also use six-neighbor connectivity. Distances follow nnU-Net's mm
input-spacing contract. Empty-class Dice and one-sided empty surfaces are null.
"""

from __future__ import annotations

import argparse
import json
from importlib import import_module
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
from scipy.ndimage import binary_erosion, generate_binary_structure, label
from scipy.spatial import cKDTree

from nnunet_mlx.data.labels import parse_dataset_label_semantics
from nnunet_mlx.errors import ContractError
from nnunet_mlx.plans import read_json_object
from nnunet_mlx.prediction.ensembling import validate_ensemble_properties
from nnunet_mlx.storage.properties import load_properties_pickle

SCHEMA = "nnunet-mlx.qualification.v1"
type Labels = npt.NDArray[np.uint16]
type Affine = npt.NDArray[np.float64]


def _read_labels(path: Path) -> tuple[Labels, Affine, dict[str, object]]:
    """Read integer 3D labels and the exact file geometry used for comparison."""
    nib = import_module("nibabel")
    image = nib.load(path)
    values = np.asarray(image.dataobj)
    if (
        values.ndim != 3
        or values.dtype.kind not in "iuf"
        or not np.isfinite(values).all()
        or np.any(values < 0)
        or np.any(values > 65535)
        or not np.array_equal(values, np.rint(values))
    ):
        raise ContractError(f"qualification requires 3D uint16-compatible labels: {path}")
    affine = np.asarray(image.affine, dtype=np.float64)
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) == 0:
        raise ContractError("qualification requires a finite, nonsingular affine")
    geometry: dict[str, object] = {
        "shape": list(values.shape),
        "affine": affine.tolist(),
        "qform_code": int(image.header["qform_code"]),
        "sform_code": int(image.header["sform_code"]),
        "header_units": list(image.header.get_xyzt_units()),
    }
    return np.ascontiguousarray(values, dtype=np.uint16), affine, geometry


def _class_metrics(
    reference: Labels, candidate: Labels, value: int, affine: Affine
) -> dict[str, object]:
    """Measure foreground agreement without pooling classes or assuming isotropy."""
    expected, actual = reference == value, candidate == value
    expected_count, actual_count = int(expected.sum()), int(actual.sum())
    overlap = int(np.count_nonzero(expected & actual))
    structure = generate_binary_structure(3, 1)
    components, reference_components = label(expected, structure=structure)
    actual_components, candidate_components = label(actual, structure=structure)
    sizes = np.bincount(components.ravel())[1:]
    # Components match when at least one of their voxels overlaps the other mask.
    overlaps = np.bincount(components[actual], minlength=len(sizes) + 1)[1:]
    actual_sizes = np.bincount(actual_components.ravel())[1:]
    actual_overlaps = np.bincount(
        actual_components[expected], minlength=len(actual_sizes) + 1,
    )[1:]
    voxel_mm3 = abs(float(np.linalg.det(affine[:3, :3])))
    small = sizes * voxel_mm3 <= 100
    actual_small = actual_sizes * voxel_mm3 <= 100
    surface: dict[str, object] = {"mean_mm": None, "p95_mm": None, "maximum_mm": None}
    if expected_count and actual_count:
        points: list[Affine] = []
        for mask in (expected, actual):
            boundary = mask & ~binary_erosion(mask, structure=structure, border_value=0)
            xyz = np.argwhere(boundary).astype(np.float64)
            points.append(xyz @ affine[:3, :3].T + affine[:3, 3])
        distances = np.concatenate(
            (
                cKDTree(points[0]).query(points[1], workers=2)[0],
                cKDTree(points[1]).query(points[0], workers=2)[0],
            )
        )
        surface = {
            "mean_mm": float(distances.mean()),
            "p95_mm": float(np.percentile(distances, 95)),
            "maximum_mm": float(distances.max()),
        }
    voxel_ml = voxel_mm3 / 1000
    return {
        "label": value,
        "reference_voxels": expected_count,
        "candidate_voxels": actual_count,
        "disagreement_voxels": expected_count + actual_count - 2 * overlap,
        "dice": (
            2 * overlap / (expected_count + actual_count) if expected_count + actual_count else None
        ),
        "volume_difference_ml": (actual_count - expected_count) * voxel_ml,
        "surface_voxel_centers": surface,
        "reference_components": reference_components,
        "candidate_components": candidate_components,
        "missed_reference_components": int(np.count_nonzero(overlaps == 0)),
        "small_reference_components_le_100_mm3": int(small.sum()),
        "missed_small_reference_components": int(np.count_nonzero(small & (overlaps == 0))),
        "unmatched_candidate_components": int(np.count_nonzero(actual_overlaps == 0)),
        "unmatched_small_candidate_components": int(
            np.count_nonzero(actual_small & (actual_overlaps == 0)),
        ),
    }


def main() -> None:
    """Append one named comparison, replacing its previous row on rerun."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--mode", choices=("label-only", "probabilities"), required=True)
    parser.add_argument(
        "--comparison", choices=("upstream", "previous-mlx", "annotation"), required=True
    )
    parser.add_argument("--candidate-mask", type=Path, required=True)
    parser.add_argument("--reference-mask", type=Path, required=True)
    parser.add_argument("--candidate-info", type=Path, required=True)
    parser.add_argument("--reference-info", type=Path, required=True)
    parser.add_argument("--dataset-json", type=Path)
    parser.add_argument("--candidate-probabilities", type=Path)
    parser.add_argument("--reference-probabilities", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    reference, affine, geometry = _read_labels(options.reference_mask)
    candidate, other_affine, _ = _read_labels(options.candidate_mask)
    if reference.shape != candidate.shape or not np.allclose(
        affine, other_affine, rtol=0, atol=1e-6
    ):
        raise ContractError("qualification masks must have the same shape and affine")
    values = sorted(set(np.unique(reference).tolist()) | set(np.unique(candidate).tolist()))
    if options.dataset_json is not None:
        semantics = parse_dataset_label_semantics(read_json_object(options.dataset_json))
        if not set(values) <= set(semantics.all_labels):
            raise ContractError("qualification masks contain labels outside dataset.json")
        values = list(semantics.all_labels)
    record: dict[str, object] = {
        "case": options.case,
        "mode": options.mode,
        "comparison": options.comparison,
        "geometry": geometry,
        "voxels": reference.size,
        "disagreement_voxels": int(np.count_nonzero(reference != candidate)),
        "candidate": read_json_object(options.candidate_info),
        "reference": read_json_object(options.reference_info),
        "classes": [
            _class_metrics(reference, candidate, int(value), affine)
            for value in values
            if value != 0
        ],
    }
    probability_paths = (options.reference_probabilities, options.candidate_probabilities)
    if any(path is not None for path in probability_paths):
        if any(path is None for path in probability_paths):
            raise ContractError("both reference and candidate probabilities are required")
        paths = cast(tuple[Path, Path], probability_paths)
        validate_ensemble_properties(
            [load_properties_pickle(path.with_suffix(".pkl")) for path in paths]
        )
        with (
            np.load(paths[0], allow_pickle=False) as left,
            np.load(paths[1], allow_pickle=False) as right,
        ):
            expected = np.asarray(left["probabilities"], dtype=np.float32)
            actual = np.asarray(right["probabilities"], dtype=np.float32)
        if (
            expected.shape != actual.shape
            or not np.isfinite(expected).all()
            or not np.isfinite(actual).all()
        ):
            raise ContractError("qualification probability shapes and finite values must agree")
        # Channel-wise reductions bound scratch space to one probability channel.
        maxima: list[float] = []
        totals: list[float] = []
        for left, right in zip(expected, actual, strict=True):
            difference = np.abs(left - right)
            maxima.append(float(difference.max()))
            totals.append(float(difference.sum(dtype=np.float64)))
        record["probabilities"] = {
            "maximum_absolute_difference": max(maxima),
            "mean_absolute_difference": sum(totals) / expected.size,
            "per_channel_maximum": maxima,
        }
    output = Path(options.output)
    matrix: dict[str, object] = (
        read_json_object(output)
        if output.is_file()
        else {"schema_version": SCHEMA, "comparisons": []}
    )
    if matrix.get("schema_version") != SCHEMA:
        raise ContractError("unsupported qualification matrix schema")
    records = cast(list[dict[str, object]], matrix["comparisons"])
    records[:] = [
        item
        for item in records
        if (item["case"], item["mode"], item["comparison"])
        != (options.case, options.mode, options.comparison)
    ]
    records.append(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".staging.json")
    temporary.write_text(json.dumps(matrix, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)


if __name__ == "__main__":
    main()
