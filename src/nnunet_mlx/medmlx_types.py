"""The JSON result contract of ``run``."""

from typing import TypedDict

from medmlx_core.outputs import RunResult as _RunResult


class RunDetails(TypedDict):
    """Record the metadata returned alongside a completed MedMLX run."""

    weights_revision: str | None
    labels: list[dict[str, object]]
    configuration: str
    source_profile: str
    precision: str
    folds: list[int | str]
    checkpoints: list[dict[str, object]]
    mirroring_axes: list[int]
    step_fraction: float
    accumulation: str
    decoding: str
    save_probabilities: bool
    backend: str
    preprocessed_shape: list[int]
    metrics: dict[str, object]


# Return value of ``nnunet_mlx.run``: ``mask`` (a label_map, or a binary_mask for
# a {0, 1} label table) and, with ``save_probabilities``, ``probabilities``.
type RunResult = _RunResult[RunDetails]
