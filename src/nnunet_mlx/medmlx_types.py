"""Declared runner input and JSON result contracts."""

from collections.abc import Sequence
from typing import NotRequired, TypedDict


class RunArguments(TypedDict):
    """Declare the keyword arguments accepted by the MedMLX segmentation runner.

    ``weights_path``, ``image_paths`` and ``output_dir`` are required. Optional
    keys select the plans configuration, folds, checkpoint, inference arithmetic
    and output options, with the defaults of ``nnunet_mlx.medmlx.run``.
    """

    weights_path: str
    image_paths: Sequence[str]
    output_dir: str
    overwrite: NotRequired[bool]
    configuration: NotRequired[str | None]
    folds: NotRequired[Sequence[int | str] | None]
    checkpoint_name: NotRequired[str]
    source_profile: NotRequired[str | None]
    arithmetic: NotRequired[str]
    step_fraction: NotRequired[float]
    use_mirroring: NotRequired[bool]
    save_probabilities: NotRequired[bool]


class RunDetails(TypedDict):
    """Record the metadata returned alongside a completed MedMLX run."""

    weights_revision: str | None
    labels: list[dict[str, object]]
    output_kind: str
    configuration: str
    source_profile: str
    arithmetic: str
    folds: list[int | str]
    checkpoints: list[dict[str, object]]
    mirroring_axes: list[int]
    step_fraction: float
    accumulation: str
    backend: str
    preprocessed_shape: list[int]
    metrics: dict[str, object]


class RunResult(TypedDict):
    """Represent one MedMLX run: absolute output paths by role and its run details."""

    outputs: dict[str, str]
    details: RunDetails
