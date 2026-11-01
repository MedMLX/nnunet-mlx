"""Probability-ensemble CLI parity with the pinned upstream folder workflow."""

from __future__ import annotations

import json
import pickle
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

from nnunet_mlx.cli import main


def _write_inputs(root: Path, *, regions: bool = False) -> None:
    """Write synthetic class or region probability inputs for CLI ensembling."""
    labels: dict[str, object] = (
        {"background": 0, "whole": [1, 2], "core": 2}
        if regions else {"background": 0, "lesion": 1}
    )
    dataset: dict[str, object] = {"labels": labels, "file_ending": ".nii.gz"}
    if regions:
        dataset["regions_class_order"] = [1, 2]
    (root / "dataset.json").write_text(json.dumps(dataset))
    (root / "plans.json").write_text(json.dumps({"image_reader_writer": "SimpleITKIO"}))
    for index, folder in enumerate((root / "run-a", root / "run-b")):
        folder.mkdir()
        for case in ("case_z", "case_a"):
            foreground = np.full((2, 3, 4), 0.75, dtype=np.float32)
            if case == "case_z":
                foreground.fill(0.25 if index == 0 else 0.75)
                foreground[0, 0, 0] = 0.5 if index == 0 else 1.0
            np.savez_compressed(
                folder / f"{case}.npz", probabilities=np.stack((1 - foreground, foreground)),
            )
            properties: dict[str, object] = {
                "case_identifier": case, "spacing": (3.0, 2.0, 1.0), "labels": labels,
                "sitk_stuff": {
                    "spacing": (1.0, 2.0, 3.0), "origin": (4.0, 5.0, 6.0),
                    "direction": (-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
                },
            }
            if regions:
                properties["regions_class_order"] = [1, 2]
            with (folder / f"{case}.pkl").open("wb") as handle:
                pickle.dump(properties, handle)


def _arguments(root: Path, output: Path | None = None) -> list[str]:
    """Build CLI arguments for the synthetic ensemble folders."""
    return [
        "model", "ensemble", "--input-folder", str(root / "run-a"),
        "--input-folder", str(root / "run-b"),
        "--output-folder", str(root / "merged" if output is None else output),
        "--plans-json", str(root / "plans.json"),
        "--dataset-json", str(root / "dataset.json"), "--json",
    ]


@pytest.mark.oracle
@pytest.mark.parametrize("regions", [False, True])
def test_ensemble_cli_matches_pinned_upstream_folder_workflow(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], regions: bool,
) -> None:
    """Compare the ensemble CLI folder workflow with pinned upstream behavior."""
    _write_inputs(tmp_path, regions=regions)
    upstream = import_module("nnunetv2.ensembling.ensemble")
    reference = tmp_path / "reference"
    upstream.ensemble_folders(
        [str(tmp_path / "run-a"), str(tmp_path / "run-b")], str(reference),
        save_merged_probabilities=True, num_processes=1,
        dataset_json_file_or_dict=str(tmp_path / "dataset.json"),
        plans_json_file_or_dict=str(tmp_path / "plans.json"),
    )
    capsys.readouterr()
    assert main([*_arguments(tmp_path), "--save-probabilities"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    sitk = import_module("SimpleITK")
    for item in json.loads(captured.out)["files"]:
        path = Path(item["segmentation"])
        expected = sitk.ReadImage(str(reference / path.name))
        actual = sitk.ReadImage(str(path))
        np.testing.assert_array_equal(
            sitk.GetArrayFromImage(actual), sitk.GetArrayFromImage(expected),
        )
        assert (actual.GetSpacing(), actual.GetOrigin(), actual.GetDirection()) == (
            expected.GetSpacing(), expected.GetOrigin(), expected.GetDirection(),
        )
        probability_path = Path(item["probabilities"])
        with (np.load(probability_path) as candidate,
              np.load(reference / probability_path.name) as oracle):
            np.testing.assert_array_equal(candidate["probabilities"], oracle["probabilities"])
