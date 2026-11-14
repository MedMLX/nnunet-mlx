"""The JSON result contract of ``segment``."""

from typing import TypedDict

from medmlx_core.outputs import Artifact


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


class RunResult(TypedDict):
    """One segmentation: absolute output paths and artifacts by role, and run details."""

    outputs: dict[str, str]
    artifacts: dict[str, Artifact]
    details: RunDetails
