"""Final-score label reduction on Metal, migrated from MLX fork 63bbcccf.

Derived Metal sources retain MIT terms in THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from math import prod
from numbers import Integral
from typing import TYPE_CHECKING

from nnunet_mlx.errors import ContractError
from nnunet_mlx.kernels.types import MetalKernel
from nnunet_mlx.runtime import import_mlx

if TYPE_CHECKING:
    from mlx.core import array

INVALID_LABEL = 2**32 - 1

_NCDHW = r"""
uint voxel = thread_position_in_grid.x;
if (voxel >= N * D * H * W) return;
uint x = voxel % W, y = (voxel / W) % H;
uint z = (voxel / (W * H)) % D, n = voxel / (W * H * D);
long base = long(n)*scores_strides[0] + long(z)*scores_strides[2]
          + long(y)*scores_strides[3] + long(x)*scores_strides[4];
float best = -INFINITY;
uint winner = 0;
uint invalid = 0;
// Adjacent threads walk adjacent voxels; channels stay in their existing layout.
for (uint c = 0; c < C; ++c) {
    float value = scores[base + long(c)*scores_strides[1]];
    invalid |= !isfinite(value);
    if (value > best) {
        best = value;
        winner = c;
    }
}
if (invalid != 0) out[voxel] = 0xffffffffu;
else out[voxel] = lut[long(winner)*lut_strides[0]];
"""

_NDHWC = r"""
uint voxel = thread_position_in_grid.x / 32;
uint lane = thread_index_in_simdgroup;
if (voxel >= N * D * H * W) return;
uint x = voxel % W, y = (voxel / W) % H;
uint z = (voxel / (W * H)) % D, n = voxel / (W * H * D);
long base = long(n)*scores_strides[0] + long(z)*scores_strides[1]
          + long(y)*scores_strides[2] + long(x)*scores_strides[3];
float best = -INFINITY;
uint winner = 0xffffffffu;
uint invalid = 0;
for (uint c = lane; c < C; c += 32) {
    float value = scores[base + long(c)*scores_strides[4]];
    invalid |= !isfinite(value);
    if (value > best || (value == best && c < winner)) {
        best = value;
        winner = c;
    }
}
float maximum = simd_max(best);
winner = simd_min(best == maximum ? winner : 0xffffffffu);
invalid = simd_sum(invalid);
if (lane == 0) {
    if (invalid != 0) out[voxel] = 0xffffffffu;
    else out[voxel] = lut[long(winner)*lut_strides[0]];
}
"""


@cache
def _kernel(layout: str) -> MetalKernel:
    """Build a class-label reduction kernel for the selected tensor layout."""
    mx, _ = import_mlx()
    return mx.fast.metal_kernel(
        name=f"nnunet_labels_{layout.lower()}",
        input_names=["scores", "lut"],
        output_names=["out"],
        source=_NDHWC if layout == "NDHWC" else _NCDHW,
        ensure_row_contiguous=False,
        compile_options={"math_mode": "safe"},
    )


def labels_from_scores(scores: array, labels: Sequence[int], *, layout: str = "NDHWC") -> array:
    """Map final float32 class scores to uint32 [N,D,H,W], choosing first ties.

    ``labels`` supplies one integer in ``[0, 2**32 - 2]`` per channel; entries
    need not be unique. Nonfinite scores produce ``INVALID_LABEL``; call
    ``require_valid_labels`` before export. This does not resample, blend windows,
    or threshold independent regions.
    """
    mx, _ = import_mlx()
    if layout not in ("NDHWC", "NCDHW"):
        raise ContractError("layout must be NDHWC or NCDHW")
    if scores.ndim != 5 or scores.dtype != mx.float32 or min(scores.shape) < 1:
        raise ContractError("final scores must be a nonempty float32 rank-5 array")
    if prod(scores.shape) > 2**31 - 1:
        raise ContractError("final scores exceed int32 indexing")
    n, d, h, w, c = (
        scores.shape if layout == "NDHWC" else (scores.shape[0], *scores.shape[2:], scores.shape[1])
    )
    table = tuple(labels)
    if len(table) != c or any(
        isinstance(v, bool) or not isinstance(v, Integral) or not 0 <= v < INVALID_LABEL
        for v in table
    ):
        raise ContractError("labels must match channels and be integers in [0, 2^32-2]")
    voxels = n * d * h * w
    group = 128 if layout == "NDHWC" else 256
    threads = ((voxels + 3) // 4) * 128 if layout == "NDHWC" else ((voxels + 255) // 256) * 256
    if threads > 2**31 - 1:
        raise ContractError("label dispatch exceeds int32 indexing; use smaller slabs")
    return _kernel(layout)(
        inputs=[scores, mx.array([int(v) for v in table], dtype=mx.uint32)],
        template=list(dict(N=n, D=d, H=h, W=w, C=c).items()),
        grid=(threads, 1, 1),
        threadgroup=(group, 1, 1),
        output_shapes=[(n, d, h, w)],
        output_dtypes=[mx.uint32],
        stream=mx.gpu,
    )[0]


def require_valid_labels(labels: array) -> array:
    """Synchronize and reject the nonfinite-score sentinel before mask export."""
    mx, _ = import_mlx()
    if labels.dtype != mx.uint32:
        raise ContractError("expected uint32 labels")
    if bool(mx.any(labels == INVALID_LABEL).item()):
        raise ContractError("nonfinite class scores cannot produce a valid segmentation")
    return labels
