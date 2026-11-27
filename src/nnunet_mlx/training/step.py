"""Differentiable MLX segmentation loss and synchronized optimizer updates."""

from __future__ import annotations

import math
from collections.abc import Callable
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


def segmentation_loss(semantics: LabelSemantics) -> Loss:
    """Create Dice plus cross-entropy, with -1 masking padded and ignored targets."""
    if not semantics.has_regions:
        return _class_loss
    membership = mx.array(
        [[label in region for region in semantics.regions] for label in semantics.all_labels],
        dtype=mx.bool_,
    )
    _mx.eval(membership)

    def loss(logits: mx.array, labels: mx.array) -> mx.array:
        return _region_loss(logits, labels, membership)

    return loss


def _class_loss(logits: mx.array, labels: mx.array) -> mx.array:
    """Gather class targets and reduce small per-tile histograms without one-hot labels."""
    classes = logits.shape[-1]
    scores = mx.reshape(logits, (-1, classes)).astype(mx.float32)
    targets = mx.reshape(labels, (-1,))
    valid = targets >= 0
    indices = mx.maximum(targets, 0)
    log_probs = scores - mx.logsumexp(scores, axis=-1, keepdims=True)
    true_log_probs = mx.reshape(mx.take_along_axis(log_probs, indices[:, None], axis=-1), (-1,))
    ce = -mx.sum(mx.where(valid, true_log_probs, 0.0)) / mx.maximum(mx.sum(valid), 1)
    predicted = mx.sum(mx.where(valid[:, None], mx.exp(log_probs), 0.0), axis=0)

    # Independent tiles bound the contention of repeated labels in scatter-add.
    tile = 64
    tiles = (targets.size + tile - 1) // tile
    bins = (mx.arange(targets.size, dtype=mx.int32) // tile) * classes + indices
    empty = mx.zeros((tiles * classes,), mx.float32)
    true_prob = mx.where(valid, mx.exp(true_log_probs), 0.0)
    scatter = cast(IndexedUpdates, empty.at)[bins]
    inter = mx.sum(mx.reshape(scatter.add(true_prob), (tiles, classes)), axis=0)
    actual = mx.sum(mx.reshape(scatter.add(valid.astype(mx.float32)), (tiles, classes)), axis=0)
    return ce + _dice_loss(inter[1:], predicted[1:], actual[1:])


def _region_loss(logits: mx.array, labels: mx.array, membership: mx.array) -> mx.array:
    """Use independent sigmoid outputs for overlapping regions."""
    scores = mx.reshape(logits, (-1, logits.shape[-1])).astype(mx.float32)
    targets = mx.reshape(labels, (-1,))
    valid = (targets >= 0)[:, None]
    positive = membership[mx.maximum(targets, 0)]
    cost = mx.logaddexp(mx.where(positive, -scores, scores), 0.0)
    ce = mx.sum(mx.where(valid, cost, 0.0)) / mx.maximum(mx.sum(valid) * scores.shape[-1], 1)
    probabilities = mx.where(valid, mx.sigmoid(scores), 0.0)
    truth = positive & valid
    inter = mx.sum(mx.where(truth, probabilities, 0.0), axis=0)
    return ce + _dice_loss(inter, mx.sum(probabilities, axis=0), mx.sum(truth, axis=0))


def _dice_loss(intersection: mx.array, predicted: mx.array, actual: mx.array) -> mx.array:
    """Average soft Dice loss over foreground labels represented in the batch."""
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
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        loss = segmentation_loss(semantics)

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
