"""A stopped CPU qualification run must resume without changing upstream results."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from nnunet_reference.fixture_types import UpstreamRecording
from nnunet_reference.runner_network import build_network

pytestmark = [pytest.mark.oracle, pytest.mark.qualification]


def test_reference_resume_matches_uncached_upstream(
    tmp_path: Path, controlled_runner_recording: UpstreamRecording,
) -> None:
    """Resume tiles and cropped, strided folds with exact source-order half arithmetic."""
    torch = pytest.importorskip("torch")
    nib = import_module("nibabel")
    metadata, _ = controlled_runner_recording
    runner = metadata["runner"]
    plans = copy.deepcopy(runner["plans"])
    configurations = cast(dict[str, dict[str, object]], plans["configurations"])
    base = configurations["base"]
    base["preprocessor_name"] = "DefaultPreprocessor"
    architecture = cast(dict[str, object], base["architecture"])
    architecture["_kw_requires_import"] = ["conv_op", "norm_op", "nonlin"]
    cast(dict[str, object], architecture["arch_kwargs"])["n_stages"] = 2
    model = tmp_path / "model"
    model.mkdir()
    (model / "plans.json").write_text(json.dumps(plans))
    (model / "dataset.json").write_text(json.dumps(runner["dataset_json"]))
    for fold in (0, 1):
        directory = model / f"fold_{fold}"
        directory.mkdir()
        torch.save({
            "network_weights": build_network(runner, runner["weight_seeds"][fold]).state_dict(),
            "init_args": {"configuration": "3d_fullres"},
            "trainer_name": "nnUNetTrainer",
            "inference_allowed_mirroring_axes": (0, 1, 2),
        }, directory / "checkpoint_final.pth")
    image = tmp_path / "input.nii.gz"
    nib.save(nib.Nifti1Image(
        np.random.default_rng(574).normal(size=(48, 48, 7)).astype(np.float32), np.eye(4),
    ), image)
    command = [
        sys.executable, "scripts/record_model_reference.py", "--model", str(model),
        "--input", str(image), "--threads", "1", "--timeout-seconds", "120",
    ]
    resumed, fresh = tmp_path / "resumed", tmp_path / "fresh"
    initial_output: list[str] = []
    with subprocess.Popen(
        [*command, "--work", str(resumed)], stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True,
    ) as stopped:
        assert stopped.stdout is not None
        try:
            for line in stopped.stdout:
                initial_output.append(line)
                if '"event": "tile_complete"' in line:
                    stopped.terminate()
                    break
            stopped.wait(timeout=10)
        finally:
            if stopped.poll() is None:
                stopped.kill()
                stopped.wait()
    assert list(resumed.glob("fold_0-tile_*.npy")), "".join(initial_output)
    assert not (resumed / "fold_0.npy").exists()
    recovered = subprocess.run(
        [*command, "--work", str(resumed)], capture_output=True, text=True, timeout=120,
    )
    assert recovered.returncode == 0, recovered.stdout + recovered.stderr
    assert '"cache_hit": true' in recovered.stdout
    uncached = subprocess.run(
        [*command, "--work", str(fresh), "--cache-folds-only"],
        capture_output=True, text=True, timeout=120,
    )
    assert uncached.returncode == 0, uncached.stdout + uncached.stderr
    for name in ("logits.npy", "fold_0.npy", "fold_1.npy"):
        np.testing.assert_array_equal(np.load(resumed / name), np.load(fresh / name))
    for name in ("label-mask.nii.gz", "probability-mask.nii.gz"):
        np.testing.assert_array_equal(
            nib.load(resumed / name).get_fdata(), nib.load(fresh / name).get_fdata(),
        )
    with np.load(resumed / "probabilities.npz") as actual, np.load(
        fresh / "probabilities.npz"
    ) as expected:
        np.testing.assert_array_equal(actual["probabilities"], expected["probabilities"])
    # Complete folds are reusable too, including the in-place upstream fold average.
    folded = subprocess.run(
        [*command, "--work", str(resumed)], capture_output=True, text=True, timeout=120,
    )
    assert folded.returncode == 0, folded.stdout + folded.stderr
    assert folded.stdout.count('"event": "fold_cache_hit"') == 2
    np.testing.assert_array_equal(
        np.load(resumed / "logits.npy"), np.load(fresh / "logits.npy"),
    )
    changed = subprocess.run(
        [*command, "--work", str(resumed), "--disable-mirroring"],
        capture_output=True, text=True, timeout=120,
    )
    assert changed.returncode != 0 and "reference settings changed" in changed.stderr
