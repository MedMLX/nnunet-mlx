"""Differentiable MLX segmentation loss and synchronized optimizer updates."""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Literal, cast

from medmlx_core.typing import IndexedUpdates
from mlx import core as mx
from mlx.nn import value_and_grad
from mlx.optimizers import AdamW, clip_grad_norm
from mlx.utils import tree_map

from nnunet_mlx.data.labels import LabelSemantics
from nnunet_mlx.errors import ContractError
from nnunet_mlx.model.network import Network
from nnunet_mlx.runtime import import_mlx
from nnunet_mlx.training.data import Batch

if TYPE_CHECKING:
    from mlx.nn import Parameters

type Precision = Literal["float32", "bfloat16"]
type Loss = Callable[[mx.array, mx.array], mx.array]
_mx, _ = import_mlx()


def segmentation_loss(
    semantics: LabelSemantics, *, balanced_ce: bool = False, batch_dice: bool = True,
) -> Loss:
    """Create Dice plus cross-entropy, with -1 masking padded and ignored targets."""
    if not semantics.has_regions:
        return partial(_class_loss, balanced_ce=balanced_ce, batch_dice=batch_dice)
    if balanced_ce:
        raise ContractError("balanced cross-entropy requires mutually exclusive class targets")
    membership = mx.array(
        [[label in region for region in semantics.regions] for label in semantics.all_labels],
        dtype=mx.bool_,
    )
    _mx.eval(membership)

    def loss(logits: mx.array, labels: mx.array) -> mx.array:
        return _region_loss(logits, labels, membership, batch_dice=batch_dice)

    return loss


def _class_loss(
    logits: mx.array, labels: mx.array, *, balanced_ce: bool, batch_dice: bool,
) -> mx.array:
    """Gather class targets and reduce small per-tile histograms without one-hot labels."""
    classes = logits.shape[-1]
    groups = 1 if batch_dice else logits.shape[0]
    scores = mx.reshape(logits, (groups, -1, classes)).astype(mx.float32)
    targets = mx.reshape(labels, (groups, -1))
    valid = targets >= 0
    indices = mx.maximum(targets, 0)
    log_probs = scores - mx.logsumexp(scores, axis=-1, keepdims=True)
    true_log_probs = mx.take_along_axis(log_probs, indices[..., None], axis=-1)[..., 0]
    negative_log_likelihood = -mx.where(valid, true_log_probs, 0.0)
    predicted = mx.sum(mx.where(valid[..., None], mx.exp(log_probs), 0.0), axis=1)

    # Tiles bound scatter-add contention without pooling across Dice groups.
    tile = 64
    voxels = targets.shape[1]
    tiles = (voxels + tile - 1) // tile
    offsets = mx.arange(groups, dtype=mx.int32)[:, None] * tiles
    bins = (offsets + mx.arange(voxels, dtype=mx.int32) // tile) * classes + indices
    empty = mx.zeros((groups * tiles * classes,), mx.float32)
    true_prob = mx.where(valid, mx.exp(true_log_probs), 0.0)
    scatter = cast(IndexedUpdates, empty.at)[bins]
    inter = mx.sum(mx.reshape(scatter.add(true_prob), (groups, tiles, classes)), axis=1)
    actual = mx.sum(
        mx.reshape(scatter.add(valid.astype(mx.float32)), (groups, tiles, classes)), axis=1,
    )
    if balanced_ce:
        # Inverse-square-root weighting tempers the contribution of tiny classes.
        counts = mx.sum(actual, axis=0)
        weights = mx.rsqrt(mx.maximum(counts, 1))
        ce = mx.sum(negative_log_likelihood * weights[indices]) / mx.maximum(
            mx.sum(weights * counts),
            1.0,
        )
    else:
        ce = mx.sum(negative_log_likelihood) / mx.maximum(mx.sum(valid), 1)
    return ce + _dice_loss(inter[..., 1:], predicted[..., 1:], actual[..., 1:])


def _region_loss(
    logits: mx.array, labels: mx.array, membership: mx.array, *, batch_dice: bool,
) -> mx.array:
    """Use independent sigmoid outputs for overlapping regions."""
    groups = 1 if batch_dice else logits.shape[0]
    scores = mx.reshape(logits, (groups, -1, logits.shape[-1])).astype(mx.float32)
    targets = mx.reshape(labels, (groups, -1))
    valid = (targets >= 0)[..., None]
    positive = membership[mx.maximum(targets, 0)]
    cost = mx.logaddexp(mx.where(positive, -scores, scores), 0.0)
    ce = mx.sum(mx.where(valid, cost, 0.0)) / mx.maximum(mx.sum(valid) * scores.shape[-1], 1)
    probabilities = mx.where(valid, mx.sigmoid(scores), 0.0)
    truth = positive & valid
    inter = mx.sum(mx.where(truth, probabilities, 0.0), axis=1)
    return ce + _dice_loss(inter, mx.sum(probabilities, axis=1), mx.sum(truth, axis=1))


def _dice_loss(intersection: mx.array, predicted: mx.array, actual: mx.array) -> mx.array:
    """Average soft Dice loss over foreground labels present in each reduction group."""
    present = actual > 0
    dice = 2.0 * intersection / mx.maximum(predicted + actual, 1e-5)
    return mx.sum(mx.where(present, 1.0 - dice, 0.0)) / mx.maximum(mx.sum(present), 1)


class TrainingStep:
    """Own the compiled graph for a fixed batch shape, model and optimizer."""

    def __init__(
        self,
        model: Network,
        optimizer: AdamW,
        semantics: LabelSemantics,
        *,
        precision: Precision,
        compile_step: bool,
        balanced_ce: bool = False,
        batch_dice: bool = True,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        loss = segmentation_loss(semantics, balanced_ce=balanced_ce, batch_dice=batch_dice)

        def objective(images: mx.array, labels: mx.array) -> mx.array:
            if precision == "float32":
                return loss(model(images), labels)
            # Cast inside autodiff so the persistent parameters and AdamW moments stay FP32.
            master = model.trainable_parameters()
            compute = cast("Parameters", tree_map(lambda value: value.astype(mx.bfloat16), master))
            model.update(compute)
            try:
                return loss(model(images.astype(mx.bfloat16)), labels)
            finally:
                model.update(master)

        loss_and_grad = value_and_grad(model, objective)

        def update(images: mx.array, labels: mx.array) -> tuple[mx.array, mx.array]:
            value, gradients = loss_and_grad(images, labels)
            clipped, norm = clip_grad_norm(gradients, 12.0)
            optimizer.update(model, clipped)
            return value, norm

        state = [model.state, optimizer.state]
        self._update = mx.compile(update, inputs=state, outputs=state) if compile_step else update

    def __call__(self, batch: Batch) -> float:
        """Update the model and return a finite loss after one evaluation boundary."""
        images, labels = batch
        loss, norm = self._update(mx.array(images), mx.array(labels))
        _mx.eval(loss, norm, self.model.parameters(), self.optimizer.state)
        value, gradient_norm = cast(float, loss.item()), cast(float, norm.item())
        if not math.isfinite(value) or not math.isfinite(gradient_norm):
            raise ContractError("training produced a non-finite loss or gradient norm")
        return value
