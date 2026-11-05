"""Declared MedMLX runner JSON result contracts."""

from typing import TypedDict


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
