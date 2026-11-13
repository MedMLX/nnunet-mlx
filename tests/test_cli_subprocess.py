"""CLI smoke test and upstream postprocessing parity, run in subprocesses."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import TypedDict

import numpy as np
import pytest
from nnunet_reference.oracle.fixtures import raw_inference_fixture
from nnunet_reference.oracle.network import real_image_input


class InferenceFolder(TypedDict):
    root: Path
    model_folder: Path


@pytest.fixture(scope="module")
def inference_model_folder(tmp_path_factory: pytest.TempPathFactory) -> InferenceFolder:
    """Create a small native model folder for CLI subprocess tests."""
    from nnunet_mlx.model.network import PlainConvUNet
    from nnunet_mlx.model.topology import PlainConvUNetSpec
    from nnunet_mlx.plans import PlansContract
    from nnunet_mlx.storage.checkpoint import (
        CheckpointIdentity,
        InferenceCheckpointState,
        save_inference_checkpoint,
    )

    plans, dataset = raw_inference_fixture()
    spec = PlainConvUNetSpec.from_plans(
        PlansContract.from_payload(plans, configuration="3d_fullres").topology_payload(),
        configuration="3d_fullres",
        input_channels=1,
        num_classes=2,
    )
    root = tmp_path_factory.mktemp("cli-subprocess")
    folder = root / "model"
    folder.mkdir()
    (folder / "plans.json").write_text(json.dumps(plans), encoding="utf-8")
    (folder / "dataset.json").write_text(json.dumps(dataset), encoding="utf-8")
    save_inference_checkpoint(
        folder / "fold_0" / "checkpoint_final",
        model=PlainConvUNet(spec),
        state=InferenceCheckpointState(
            identity=CheckpointIdentity("nnUNetTrainer", "3d_fullres", 0),
            current_epoch=1000,
            spatial_dims=3,
            inference_allowed_mirroring_axes=(0, 1, 2),
            dropped_source_keys=(),
        ),
    )
    return {"root": root, "model_folder": folder}


def _cli_command() -> list[str]:
    """Select the installed CLI executable or module entry point."""
    script = Path(sys.executable).with_name("nnunet-mlx")
    if script.is_file():
        return [str(script)]
    return [sys.executable, "-m", "nnunet_mlx.cli"]


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run a CLI subprocess and capture its result."""
    return subprocess.run(
        [*_cli_command(), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=900,
    )


def _write_raw_image(path: Path) -> None:
    """Write an HCC-TACE CT crop with controlled physical geometry."""
    sitk = pytest.importorskip("SimpleITK")
    data = np.zeros((8, 8, 8), dtype=np.float32)
    data[2:6, 2:6, 2:6] = real_image_input((4, 4, 4), 1)[0] + np.float32(1)
    image = sitk.GetImageFromArray(data)
    image.SetSpacing((1.0, 1.0, 1.0))
    image.SetOrigin((2.0, 3.0, 4.0))
    sitk.WriteImage(image, str(path))


def test_model_predict_subprocess_preserves_geometry_and_rejects_overwrite(
    inference_model_folder: InferenceFolder,
) -> None:
    """Preserve input geometry and enforce output overwrite behavior through the CLI."""
    source = inference_model_folder["root"] / "case_0000.nii.gz"
    _write_raw_image(source)
    prefix = inference_model_folder["root"] / "predictions" / "case"
    arguments = [
        "model",
        "predict",
        "--model-folder",
        str(inference_model_folder["model_folder"]),
        "--input",
        str(source),
        "--output-prefix",
        str(prefix),
        "--configuration",
        "3d_fullres",
        "--fold",
        "0",
        "--save-probabilities",
        "--json",
    ]
    result = _run_cli(*arguments)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)

    assert payload["folds"] == [0]
    assert payload["preprocessed_shape"] == [1, 4, 4, 4]
    sitk = pytest.importorskip("SimpleITK")
    output_image = sitk.ReadImage(payload["files"]["segmentation"])
    source_image = sitk.ReadImage(str(source))
    assert output_image.GetSize() == source_image.GetSize()
    assert output_image.GetSpacing() == source_image.GetSpacing()
    assert output_image.GetOrigin() == source_image.GetOrigin()
    assert output_image.GetDirection() == source_image.GetDirection()
    assert Path(payload["files"]["probabilities"]).is_file()
    assert Path(payload["files"]["properties"]).is_file()

    rejected = _run_cli(*arguments)
    assert rejected.returncode == 1
    assert rejected.stdout == ""
    assert json.loads(rejected.stderr)["error"]["message"]


@pytest.mark.oracle
def test_model_apply_postprocessing_subprocess_matches_pinned_nnunet_oracle(
    inference_model_folder: InferenceFolder,
) -> None:
    """The inference-time counterpart of upstream's ``nnUNetv2_apply_postprocessing``."""

    import pickle

    pytest.importorskip(
        "nnunetv2", reason="requires the optional 'oracle' extra", exc_type=ModuleNotFoundError
    )
    from importlib import import_module

    upstream_pp_fn = import_module(
        "nnunetv2.postprocessing.remove_connected_components"
    ).remove_all_but_largest_component_from_segmentation

    sitk = pytest.importorskip("SimpleITK")
    root = inference_model_folder["root"]
    input_folder = root / "postprocessing-input"
    input_folder.mkdir()
    nib = pytest.importorskip("nibabel")
    tumor_path = Path.home() / ".cache/medmlx/test-data/hcc_063_tumor.nii.gz"
    labels = np.asarray(nib.load(tumor_path).dataobj, dtype=np.uint8)[90:155, 205:305, 50:80]
    image = sitk.GetImageFromArray(labels)
    sitk.WriteImage(image, str(input_folder / "case_0.nii.gz"))

    postprocessing_pkl = root / "postprocessing.pkl"
    with postprocessing_pkl.open("wb") as handle:
        pickle.dump(([upstream_pp_fn], [{"labels_or_regions": [1]}]), handle)
    output_folder = root / "postprocessing-output"

    result = _run_cli(
        "model",
        "apply-postprocessing",
        "--input-folder",
        str(input_folder),
        "--output-folder",
        str(output_folder),
        "--postprocessing-pkl",
        str(postprocessing_pkl),
        "--plans-json",
        str(inference_model_folder["model_folder"] / "plans.json"),
        "--dataset-json",
        str(inference_model_folder["model_folder"] / "dataset.json"),
        "--json",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["files"] == [str(output_folder / "case_0.nii.gz")]

    actual = sitk.GetArrayFromImage(sitk.ReadImage(str(output_folder / "case_0.nii.gz")))
    expected = upstream_pp_fn(labels.copy(), [1])
    assert 0 < np.count_nonzero(expected) < np.count_nonzero(labels)
    np.testing.assert_array_equal(actual, expected)
