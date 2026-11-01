"""Fuse sliding-window normalization and class reduction on Apple Metal."""

from __future__ import annotations

from functools import cache
from math import prod
from typing import TYPE_CHECKING, cast

from nnunet_mlx.errors import ContractError

if TYPE_CHECKING:
    from mlx.core import array

    from nnunet_mlx.kernels.types import MetalKernel

INVALID_LABEL = 65535
_ROW_VOXELS = 32768

_SOURCE = r"""
uint voxel = thread_position_in_grid.x / 32 + thread_position_in_grid.y * ROW_VOXELS;
uint lane = thread_index_in_simdgroup;
if (voxel >= N) return;
float denominator = float(normalizer[voxel]);
uint invalid = !isfinite(denominator) || denominator <= 0.0f;
float best = -INFINITY;
uint winner = 0xffffffffu;
for (uint c = lane; c < C; c += 32) {
    float value = float(logits[ulong(voxel) * C + c]) / denominator;
    if constexpr (ROUND_HALF) value = float(half(value));
    invalid |= !isfinite(value);
    if (value > best || (value == best && c < winner)) {
        best = value;
        winner = c;
    }
}
float maximum = simd_max(best);
winner = simd_min(best == maximum ? winner : 0xffffffffu);
invalid = simd_sum(invalid);
if (lane == 0) labels[voxel] = invalid ? ushort(65535) : ushort(winner);
"""


@cache
def _kernel() -> MetalKernel:
    """Build the fused normalization and class-label reduction kernel."""
    import mlx.core as mx

    return cast("MetalKernel", mx.fast.metal_kernel(
        name="nnunet_mlx_normalized_labels",
        input_names=["logits", "normalizer"],
        output_names=["labels"],
        source=_SOURCE,
        compile_options={"math_mode": "safe"},
    ))


def normalized_labels(
    logits: array, normalizer: array, *, round_to_float16: bool = False,
) -> tuple[array, array]:
    """Return lazy uint16 spatial labels and a scalar validity flag on Metal.

    Inputs are channels-last `(D,H,W,C)` accumulated logits and `(D,H,W,1)`
    positive Gaussian weights, both float16 or both float32. Each quotient uses
    float32 division; ``round_to_float16`` rounds it through float16 before
    comparing classes, preserving ties created by nnU-Net's final half cast.
    Equal scores choose the earliest channel. The same three spatial axes and
    their padding remain intact in the output.

    A nonpositive/nonfinite denominator or nonfinite normalized score writes
    ``INVALID_LABEL`` and makes the scalar flag false. Callers must evaluate
    that flag and reject invalid results before exporting or casting labels.
    """
    import mlx.core as mx

    if logits.ndim != 4 or normalizer.shape != (*logits.shape[:3], 1) or min(logits.shape) < 1:
        raise ContractError("normalized labels require nonempty DHWC logits and DHW1 weights")
    if logits.dtype not in (mx.float16, mx.float32) or normalizer.dtype != logits.dtype:
        raise ContractError("normalized labels require matching float16 or float32 inputs")
    classes = logits.shape[-1]
    if classes > INVALID_LABEL:
        raise ContractError("normalized labels exceed the uint16 class range")
    voxels = prod(logits.shape[:3])
    if voxels > 2**32 - 1:
        raise ContractError("normalized labels exceed uint32 spatial indexing")
    row_voxels = min(voxels, _ROW_VOXELS)
    row_threads = ((row_voxels + 3) // 4) * 128
    rows = (voxels + _ROW_VOXELS - 1) // _ROW_VOXELS
    labels = _kernel()(
        inputs=[logits, normalizer],
        template=[("N", voxels), ("C", classes), ("ROUND_HALF", round_to_float16),
                  ("ROW_VOXELS", _ROW_VOXELS)],
        grid=(row_threads, rows, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[logits.shape[:3]],
        output_dtypes=[mx.uint16],
        stream=mx.gpu,
    )[0]
    return labels, mx.all(labels != INVALID_LABEL)
