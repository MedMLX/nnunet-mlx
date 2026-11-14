"""Qualification metrics must reveal small-component and physical-boundary changes."""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.host


@pytest.mark.parametrize("same_grid", [True, False])
def test_qualification_report_measures_components_on_the_physical_grid(
    tmp_path: Path, same_grid: bool,
) -> None:
    """A two-mm shift changes one component even when whole-class volume is unchanged."""
    nib = import_module("nibabel")
    reference = np.zeros((7, 7, 7), dtype=np.uint16)
    reference[1, 1, 1] = reference[4, 4, 4] = 1
    candidate = reference.copy()
    candidate[4, 4, 4], candidate[5, 4, 4] = 0, 1
    affine = np.diag([2.0, 1.0, 1.0, 1.0])
    nib.save(nib.Nifti1Image(reference, affine), tmp_path / "reference.nii.gz")
    other_affine = affine.copy()
    if not same_grid:
        other_affine[0, 3] = 2
    nib.save(nib.Nifti1Image(candidate, other_affine), tmp_path / "candidate.nii.gz")
    for name, labels in (("reference", reference), ("candidate", candidate)):
        probabilities = np.stack([labels == value for value in range(3)]).astype(np.float32)
        np.savez_compressed(tmp_path / f"{name}.npz",
                            probabilities=probabilities.transpose(0, 3, 2, 1))
        with (tmp_path / f"{name}.pkl").open("wb") as handle:
            pickle.dump({"nibabel_stuff": {"original_affine": affine},
                         "spacing": [1, 1, 2]}, handle)
    (tmp_path / "info.json").write_text(json.dumps({"versions": {"reference": "controlled"}}))
    (tmp_path / "dataset.json").write_text(json.dumps({
        "labels": {"background": 0, "lesion": 1, "absent": 2},
    }))
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "matrix.json"
    completed = subprocess.run([
        sys.executable, str(root / "scripts/qualification_report.py"), "--case", "controlled",
        "--mode", "probabilities", "--comparison", "upstream",
        "--candidate-mask", str(tmp_path / "candidate.nii.gz"),
        "--reference-mask", str(tmp_path / "reference.nii.gz"),
        "--candidate-info", str(tmp_path / "info.json"),
        "--reference-info", str(tmp_path / "info.json"),
        "--candidate-probabilities", str(tmp_path / "candidate.npz"),
        "--reference-probabilities", str(tmp_path / "reference.npz"),
        "--dataset-json", str(tmp_path / "dataset.json"), "--output", str(output),
    ], env={**os.environ, "PYTHONPATH": str(root / "src")}, capture_output=True, text=True)
    if not same_grid:
        assert completed.returncode != 0
        assert not output.exists()
        return
    assert completed.returncode == 0, completed.stderr
    row = json.loads(output.read_text())["comparisons"][0]
    assert row["disagreement_voxels"] == 2
    foreground, absent = row["classes"]
    assert foreground["dice"] == 0.5
    assert foreground["volume_difference_ml"] == 0
    assert foreground["reference_components"] == foreground["candidate_components"] == 2
    assert foreground["missed_reference_components"] == 1
    assert foreground["missed_small_reference_components"] == 1
    assert foreground["unmatched_candidate_components"] == 1
    assert foreground["unmatched_small_candidate_components"] == 1
    assert foreground["surface_voxel_centers"] == {"mean_mm": 1.0, "p95_mm": 2.0,
                                                "maximum_mm": 2.0}
    assert absent["dice"] is None
    assert row["probabilities"]["maximum_absolute_difference"] == 1.0
    assert row["probabilities"]["per_channel_maximum"] == [1.0, 1.0, 0.0]
    assert row["probabilities"]["mean_absolute_difference"] == 4 / (3 * 7**3)
