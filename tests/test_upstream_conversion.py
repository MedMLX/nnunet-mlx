"""Converted trained nnU-Net folders must compute what the upstream network computes."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from nnunet_reference.arrays import host_array

if TYPE_CHECKING:
    from torch.nn import Module as TorchModule

pytestmark = pytest.mark.oracle


def _write_trained_folder(
    folder: Path, plans: dict[str, object], dataset_json: dict[str, object], *,
    trainer: str, mirroring_axes: tuple[int, ...] | None,
) -> TorchModule:
    """Write a two-fold upstream trained model folder with a small PlainConvUNet."""
    torch = pytest.importorskip("torch")
    networks = pytest.importorskip("dynamic_network_architectures.architectures.unet")
    folder.mkdir()
    (folder / "plans.json").write_text(json.dumps(plans), encoding="utf-8")
    (folder / "dataset.json").write_text(json.dumps(dataset_json), encoding="utf-8")
    torch.manual_seed(409)
    model = networks.PlainConvUNet(
        input_channels=1, n_stages=2, features_per_stage=(4, 8),
        conv_op=torch.nn.Conv3d, kernel_sizes=((3, 3, 3),) * 2,
        strides=((1, 1, 1), (2, 2, 2)), n_conv_per_stage=(2, 2),
        num_classes=2, n_conv_per_stage_decoder=(2,), conv_bias=True,
        norm_op=torch.nn.InstanceNorm3d, norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=torch.nn.LeakyReLU, nonlin_kwargs={"negative_slope": 0.01},
    ).eval()
    for fold in (0, 1):
        (folder / f"fold_{fold}").mkdir()
        torch.save({
            "network_weights": model.state_dict(),
            "current_epoch": 250,
            "init_args": {"configuration": "3d_fullres"},
            "trainer_name": trainer,
            "inference_allowed_mirroring_axes": mirroring_axes,
            # Upstream logs hold NumPy scalars, which the restricted loader must admit.
            "logging": {"ema_fg_dice": [np.float64(0.5)]},
        }, folder / f"fold_{fold}" / "checkpoint_final.pth")
    return model


def _tiny_plans(plans: dict[str, object]) -> dict[str, object]:
    """Point the fixture plans at the two-stage network the helper trains."""
    configurations = cast(dict[str, object], plans["configurations"])
    base = cast(dict[str, object], configurations["base"])
    architecture = cast(dict[str, object], base["architecture"])
    cast(dict[str, object], architecture["arch_kwargs"])["n_stages"] = 2
    return plans


@pytest.mark.metal
def test_converted_folder_matches_upstream_network(
    tmp_path: Path,
    tiny_model_plans: dict[str, object],
    tiny_model_dataset: dict[str, object],
) -> None:
    """Weights, trainer variant and disabled mirroring survive conversion."""
    torch = pytest.importorskip("torch")

    from nnunet_mlx.model.folder import load_model_bundle
    from nnunet_mlx.runtime import import_mlx
    from nnunet_mlx.storage.upstream import convert_trained_model

    source = tmp_path / "nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres"
    model = _write_trained_folder(
        source, _tiny_plans(tiny_model_plans), tiny_model_dataset,
        trainer="nnUNetTrainer_250epochs", mirroring_axes=None,
    )
    result = convert_trained_model(source, tmp_path / "mlx")
    assert result.folds == (0, 1)
    bundle = load_model_bundle(tmp_path / "mlx", configuration="3d_fullres", folds=[1])
    assert bundle.mirroring_axes == ()
    values = np.random.default_rng(419).normal(size=(1, 1, 8, 8, 8)).astype(np.float32)
    with torch.inference_mode():
        expected = model(torch.from_numpy(values)).numpy()
    mx, _ = import_mlx()
    actual = bundle.models[0](mx.array(np.moveaxis(values, 1, -1)))
    mx.eval(actual)
    np.testing.assert_allclose(
        np.moveaxis(host_array(actual), -1, 1), expected, rtol=1e-5, atol=1e-5
    )


def test_trainer_with_its_own_network_is_rejected(
    tmp_path: Path,
    tiny_model_plans: dict[str, object],
    tiny_model_dataset: dict[str, object],
) -> None:
    """A trainer that replaces the planned network cannot be converted from the plans."""
    from nnunet_mlx.errors import ContractError
    from nnunet_mlx.storage.upstream import convert_trained_model

    source = tmp_path / "trained"
    _write_trained_folder(
        source, _tiny_plans(tiny_model_plans), tiny_model_dataset,
        trainer="nnUNetTrainerBN", mirroring_axes=(0, 1, 2),
    )
    with pytest.raises(ContractError, match="nnUNetTrainerBN"):
        convert_trained_model(source, tmp_path / "mlx")
    assert not (tmp_path / "mlx").exists()


@pytest.mark.parametrize(
    ("class_name", "patch_size"),
    [("PlainConvUNet", [64, 64]), ("ResidualEncoderUNet", [32, 64, 64])],
)
def test_legacy_plans_match_upstream_reconstruction(
    class_name: str, patch_size: list[int]
) -> None:
    """Pre-2.4 plans get exactly the architecture nnU-Net rebuilds when it loads them."""
    handler = pytest.importorskip("nnunetv2.utilities.plans_handling.plans_handler")

    from nnunet_mlx.model.source_plans import upgrade_legacy_plans

    dims = len(patch_size)
    legacy: dict[str, object] = {
        "patch_size": patch_size,
        "UNet_class_name": class_name,
        "UNet_base_num_features": 32,
        "unet_max_num_features": 320,
        "n_conv_per_stage_encoder": [2, 2, 2, 2],
        "n_conv_per_stage_decoder": [2, 2, 2],
        "num_pool_per_axis": [3] * dims,
        "conv_kernel_sizes": [[3] * dims] * 4,
        "pool_op_kernel_sizes": [[1] * dims, *[[2] * dims] * 3],
    }
    expected = handler.ConfigurationManager(copy.deepcopy(legacy)).configuration
    upgraded = upgrade_legacy_plans({"configurations": {"legacy": legacy}})
    configurations = cast(dict[str, object], upgraded["configurations"])
    assert configurations["legacy"] == expected
