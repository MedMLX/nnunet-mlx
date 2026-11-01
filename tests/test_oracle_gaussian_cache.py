"""The upstream Gaussian cache must not make oracle behavior order-dependent."""

import numpy as np
import pytest
from nnunet_reference.oracle.inference import pinned_sliding_window_logits


@pytest.mark.oracle
def test_sliding_window_after_inference_mode_cache() -> None:
    """Reuse the source Gaussian cache after leaving Torch inference mode."""
    torch = pytest.importorskip("torch")
    sliding = pytest.importorskip("nnunetv2.inference.sliding_window_prediction")
    model = torch.nn.Conv2d(1, 1, kernel_size=1, bias=False).eval()
    with torch.no_grad():
        model.weight.fill_(1.0)
    values = np.ones((1, 8, 12), dtype=np.float32)
    sliding.compute_gaussian.cache_clear()
    try:
        with torch.inference_mode():
            first = pinned_sliding_window_logits(
                model, values, patch=(8, 8), num_classes=1, mirror_axes=(1,),
            )
        second = pinned_sliding_window_logits(
            model, values, patch=(8, 8), num_classes=1, mirror_axes=(1,),
        )
        np.testing.assert_allclose(first, values, rtol=0, atol=1e-6)
        np.testing.assert_array_equal(second, first)
    finally:
        sliding.compute_gaussian.cache_clear()
