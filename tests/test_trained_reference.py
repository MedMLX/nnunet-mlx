"""Check that the upstream reference path can run without test-time mirroring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

if TYPE_CHECKING:
    from torch.nn import Module as TorchModule

pytestmark = pytest.mark.oracle

def _write_upstream_model_folder(
    folder: Path, plans: dict[str, object], dataset_json: dict[str, object]
) -> TorchModule:
    """Create a small reference network and its upstream checkpoint."""
    torch = pytest.importorskip("torch")
    networks = pytest.importorskip("dynamic_network_architectures.architectures.unet")
    folder.mkdir()
    (folder / "plans.json").write_text(json.dumps(plans), encoding="utf-8")
    (folder / "dataset.json").write_text(json.dumps(dataset_json), encoding="utf-8")
    torch.manual_seed(401)
    model = networks.PlainConvUNet(
        input_channels=1, n_stages=2, features_per_stage=(4, 8),
        conv_op=torch.nn.Conv3d, kernel_sizes=((3, 3, 3),) * 2,
        strides=((1, 1, 1), (2, 2, 2)), n_conv_per_stage=(2, 2),
        num_classes=2, n_conv_per_stage_decoder=(2,), conv_bias=True,
        norm_op=torch.nn.InstanceNorm3d, norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=torch.nn.LeakyReLU, nonlin_kwargs={"negative_slope": 0.01},
    ).eval()
    checkpoint = {
        "network_weights": model.state_dict(),
        "current_epoch": 1000,
        "init_args": {"configuration": "3d_fullres"},
        "trainer_name": "nnUNetTrainer",
        "inference_allowed_mirroring_axes": (0, 1, 2),
    }
    (folder / "fold_0").mkdir()
    torch.save(checkpoint, folder / "fold_0" / "checkpoint_final.pth")
    return model


def test_trained_reference_can_disable_mirroring(
    tmp_path: Path,
    tiny_model_plans: dict[str, object],
    tiny_model_dataset: dict[str, object],
) -> None:
    """Allow mirroring to be disabled for the trained reference workflow."""
    from nnunet_reference.model_folder import source_logits

    torch = pytest.importorskip("torch")
    nib = pytest.importorskip("nibabel")
    tiny_model_plans["transpose_forward"] = [0, 1, 2]
    tiny_model_plans["transpose_backward"] = [0, 1, 2]
    configurations = cast(dict[str, object], tiny_model_plans["configurations"])
    base = cast(dict[str, object], configurations["base"])
    base["preprocessor_name"] = "DefaultPreprocessor"
    architecture = cast(dict[str, object], base["architecture"])
    cast(dict[str, object], architecture["arch_kwargs"])["n_stages"] = 2
    architecture["_kw_requires_import"] = [
        "conv_op", "norm_op", "nonlin",
    ]
    source = tmp_path / "upstream"
    model = _write_upstream_model_folder(source, tiny_model_plans, tiny_model_dataset)
    image = tmp_path / "case_0000.nii.gz"
    values = np.random.default_rng(403).normal(size=(8, 8, 8)).astype(np.float32)
    nib.save(nib.Nifti1Image(values, np.eye(4)), image)
    actual, *_ = source_logits(
        source, [image], (0,), "checkpoint_final.pth", use_mirroring=False,
    )
    # SimpleITK reads XYZ NIfTI voxels as ZYX; one tile needs no padding or resampling.
    with torch.inference_mode():
        expected = model(torch.from_numpy(values.transpose(2, 1, 0).copy()[None, None]))
    np.testing.assert_allclose(actual, expected.numpy()[0], rtol=1e-6, atol=1e-6)
