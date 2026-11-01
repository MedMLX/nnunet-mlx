"""Shared model channel and fold contracts, independent of execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from nnunet_mlx.contracts import parse_mapping as _mapping
from nnunet_mlx.errors import ContractError


class PreviousStageMetadata(Protocol):
    """Expose the previous cascade stage name from parsed plans metadata."""

    @property
    def previous_stage_name(self) -> str | None:
        """Return the prior cascade stage name, if one is configured."""
        ...


class ChannelPlanMetadata(Protocol):
    """Expose the plans configuration used to size network input channels."""

    @property
    def configuration(self) -> PreviousStageMetadata:
        """Return the selected plans configuration metadata."""
        ...


class ChannelLabelMetadata(Protocol):
    """Expose the dataset label values that become model output channels."""

    @property
    def all_labels(self) -> tuple[int, ...]:
        """Return the dataset label values used for model channels."""
        ...


def input_channel_count(dataset_json: Mapping[str, object]) -> int:
    """Return the number of image modalities declared by ``dataset_json["channel_names"]``.

    Raises:
        ContractError: If the keys are not the canonical integer strings ``"0"`` to
            ``"N-1"``.
    """
    channels = _mapping(dataset_json.get("channel_names"), name="dataset channel_names")
    if not channels:
        raise ContractError("dataset channel_names must not be empty")
    indices: list[int] = []
    for key in channels:
        try:
            index = int(key)
        except ValueError as error:
            raise ContractError("dataset channel_names keys must be integer strings") from error
        if index < 0 or str(index) != key:
            raise ContractError("dataset channel_names keys must be canonical integers")
        indices.append(index)
    if sorted(indices) != list(range(len(indices))):
        raise ContractError("dataset channel_names must be contiguous from zero")
    return len(indices)


def model_input_channel_count(
    dataset_json: Mapping[str, object],
    plans: ChannelPlanMetadata,
    semantics: ChannelLabelMetadata,
) -> int:
    """Return the network's input channel count.

    For a cascade model (the plans name a previous stage) this is the image
    modalities plus one one-hot channel per foreground label, matching
    ``labelmap_to_one_hot``; otherwise it equals ``input_channel_count``.
    """
    image_channels = input_channel_count(dataset_json)
    if plans.configuration.previous_stage_name is None:
        return image_channels
    foreground_channels = sum(label != 0 for label in semantics.all_labels)
    if foreground_channels < 1:
        raise ContractError("cascade model requires at least one foreground label")
    return image_channels + foreground_channels


def validate_folds(values: Sequence[int | str]) -> tuple[int | str, ...]:
    """Return ``values`` as a tuple after checking they are unique non-negative ints or "all"."""
    folds = tuple(values)
    if not folds or len(folds) != len(set(folds)):
        raise ContractError("model folds must be non-empty and unique")
    for fold in folds:
        if not (
            (isinstance(fold, int) and not isinstance(fold, bool) and fold >= 0) or fold == "all"
        ):
            raise ContractError("model folds must be non-negative integers or 'all'")
    return folds
