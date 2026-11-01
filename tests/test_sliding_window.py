"""Sliding-window accumulation compared with pinned upstream, and compiled inference."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from mlx.core import array

import numpy as np
import pytest

from nnunet_mlx.model.network import PlainConvUNet, ResidualEncoderUNet
from nnunet_mlx.model.topology import PlainConvUNetSpec, ResidualEncoderUNetSpec
from nnunet_mlx.prediction.sliding_window import nnunet_fp16_gaussian, predict_logits
from nnunet_mlx.runtime import compile_inference_model, import_mlx


class PointwiseModel:
    def __call__(self, values: array) -> array:
        """Return deterministic pointwise logits for tile tests."""
        mx, _ = import_mlx()
        return mx.concatenate((values, -values), axis=-1)


@pytest.mark.oracle
def test_fp16_gaussian_matches_pinned_upstream() -> None:
    """Compare the FP16 Gaussian importance map with the pinned source."""
    torch = pytest.importorskip("torch")
    sliding_window = pytest.importorskip("nnunetv2.inference.sliding_window_prediction")

    actual = nnunet_fp16_gaussian((128, 128, 128))
    expected = sliding_window.compute_gaussian(
        (128, 128, 128), value_scaling_factor=10, device=torch.device("cpu")
    ).numpy()
    assert actual.dtype == np.float16
    assert np.array_equal(actual, expected)
    assert actual.min() > 0


@pytest.mark.oracle
def test_fp16_multitile_accumulation_matches_pinned_upstream() -> None:
    """Compare overlapping FP16 tile accumulation with upstream."""
    torch = pytest.importorskip("torch")
    prediction = pytest.importorskip("nnunetv2.inference.predict_from_raw_data")
    labels = pytest.importorskip("nnunetv2.utilities.label_handling.label_handling")
    plans = pytest.importorskip("nnunetv2.utilities.plans_handling.plans_handler")

    class TorchPointwise(torch.nn.Module):
        def forward(self, values: array) -> array:
            """Return source pointwise logits for the FP16 tile comparison."""
            return torch.cat((values, -values), dim=1)

    values = np.random.default_rng(72).normal(size=(1, 7, 9, 11)).astype(np.float32)
    predictor = prediction.nnUNetPredictor(
        device=torch.device("cpu"), perform_everything_on_device=False, use_mirroring=False,
        allow_tqdm=False,
    )
    predictor.network = TorchPointwise()
    predictor.configuration_manager = plans.ConfigurationManager(
        {"patch_size": [4, 5, 6], "architecture": {}}
    )
    predictor.label_manager = labels.LabelManager({"background": 0, "foreground": 1}, None)
    returned = predictor.predict_sliding_window_return_logits(torch.tensor(values))
    assert isinstance(returned, torch.Tensor)
    expected = returned.numpy()
    actual = predict_logits(
        PointwiseModel(), values, patch_size=(4, 5, 6), num_classes=2, accumulation="nnunet-fp16"
    )
    assert actual.dtype == np.float32
    assert np.array_equal(actual, expected.astype(np.float32))


@pytest.mark.parametrize(
    ("spec", "shape"), [("tiny_spec", (9, 11, 13)), ("tiny_residual_spec", (8, 8, 8))],
)
def test_compiled_model_preserves_multitile_logits(
    request: pytest.FixtureRequest, spec: str, shape: tuple[int, int, int],
) -> None:
    """Preserve logits exactly under compilation for qualified plain and residual models."""
    mx, _ = import_mlx()
    mx.random.seed(43)
    topology = cast(PlainConvUNetSpec | ResidualEncoderUNetSpec, request.getfixturevalue(spec))
    model = (
        PlainConvUNet(topology)
        if isinstance(topology, PlainConvUNetSpec)
        else ResidualEncoderUNet(topology)
    )
    values = np.random.default_rng(43).normal(size=(1, *shape)).astype(np.float32)
    reference, compiled = (
        predict_logits(
            candidate, values, patch_size=(8, 8, 8), num_classes=3, mirroring_axes=(0, 1, 2),
        )
        for candidate in (model, compile_inference_model(model))
    )
    assert np.array_equal(compiled, reference)
