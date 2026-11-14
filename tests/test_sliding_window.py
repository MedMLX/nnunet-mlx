"""Sliding-window accumulation compared with pinned upstream, and compiled inference."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from mlx.core import array

import numpy as np
import pytest
from nnunet_reference.arrays import host_array
from nnunet_reference.oracle.network import real_image_input

from nnunet_mlx.model.topology import PlainConvUNetSpec, ResidualEncoderUNetSpec
from nnunet_mlx.prediction.sliding_window import nnunet_fp16_gaussian, predict_logits
from nnunet_mlx.runtime import compile_inference_model, import_mlx


class PointwiseModel:
    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale

    def __call__(self, values: array) -> array:
        """Return deterministic pointwise logits for tile tests."""
        mx, _ = import_mlx()
        return mx.concatenate((values * self.scale, -values * self.scale), axis=-1)


@pytest.mark.metal
@pytest.mark.parametrize("classes", [3, 30])
def test_mirrored_running_sum_preserves_order_bitwise(classes: int) -> None:
    """Materializing each running sum retains the original ordered FP32 arithmetic."""
    from itertools import combinations

    from nnunet_mlx.prediction.sliding_window import mirrored_prediction
    from nnunet_mlx.runtime import InferenceRuntime

    mx, nn = import_mlx()
    mx.random.seed(715)
    model = nn.Conv3d(1, classes, 3, padding=1)
    shape = (1, 8, 7, 6, 1) if classes == 3 else (1, 64, 64, 72, 1)
    tile = mx.array(np.random.default_rng(715).normal(size=shape).astype(np.float32))
    outputs = [host_array(model(tile)).astype(np.float32, copy=False)]
    for count in range(1, 4):
        for selected in combinations((1, 2, 3), count):
            output = model(mx.contiguous(mx.flip(tile, axis=selected)))
            outputs.append(
                host_array(mx.flip(output, axis=selected)).astype(np.float32, copy=False)
            )
    expected = outputs[0].copy()
    for output in outputs[1:]:
        expected += output
    expected /= len(outputs)
    actual = mirrored_prediction(
        model, tile, (0, 1, 2), cast(InferenceRuntime, mx), num_classes=classes,
    )
    mx.eval(actual)
    np.testing.assert_array_equal(host_array(actual), expected)


@pytest.mark.metal
@pytest.mark.parametrize(
    ("accumulation", "shape", "axes", "folds"),
    [("float32", (3, 3, 7), (), 1), ("nnunet-fp16", (4, 7, 9), (0, 1), 2)],
)
def test_2d_volume_logits_match_ordered_slice_stack(
    accumulation: str, shape: tuple[int, int, int], axes: tuple[int, ...], folds: int,
) -> None:
    """Keep slice placement and fold/half rounding equal to separate slice outputs."""
    from nnunet_mlx.prediction.inference import average_fold_logits

    data = real_image_input(shape, 1)
    models = tuple(PointwiseModel(scale) for scale in (1.0, 0.7)[:folds])
    expected = None
    for model in models:
        current = np.stack([
            predict_logits(
                model, data[:, index], patch_size=(4, 5), num_classes=2,
                mirroring_axes=axes, accumulation=accumulation,
            ) for index in range(shape[0])
        ], axis=1)
        if accumulation == "nnunet-fp16":
            current = current.astype(np.float16)
        if expected is None:
            expected = current
        else:
            expected += current
    assert expected is not None
    expected /= np.float32(folds)
    actual = average_fold_logits(
        models, data, patch_size=(4, 5), num_outputs=2,
        mirroring_axes=axes, accumulation=accumulation,
    )
    np.testing.assert_array_equal(actual, expected)


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

    values = real_image_input((7, 9, 11), 1)
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
    ("spec", "shape"), [("tiny_spec", (9, 11, 13)), ("tiny_residual_spec", (9, 11, 13))],
)
def test_compiled_model_preserves_multitile_logits(
    request: pytest.FixtureRequest, spec: str, shape: tuple[int, int, int],
) -> None:
    """Preserve logits exactly under compilation for qualified plain and residual models."""
    mx, _ = import_mlx()
    mx.random.seed(43)
    from nnunet_mlx.model.network import PlainConvUNet, ResidualEncoderUNet

    topology = cast(PlainConvUNetSpec | ResidualEncoderUNetSpec, request.getfixturevalue(spec))
    model = (
        PlainConvUNet(topology)
        if isinstance(topology, PlainConvUNetSpec)
        else ResidualEncoderUNet(topology)
    )
    # Match loaded inference models, including residual stage evaluation boundaries.
    model.eval()
    values = real_image_input(shape, 1)
    reference, compiled = (
        predict_logits(
            candidate, values, patch_size=(8, 8, 8), num_classes=3, mirroring_axes=(0, 1, 2),
        )
        for candidate in (model, compile_inference_model(model))
    )
    assert np.array_equal(compiled, reference)
