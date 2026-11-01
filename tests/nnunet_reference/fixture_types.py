"""Schemas of pinned upstream-runner qualification inputs and observations."""

from typing import NotRequired, TypedDict

from nnunet_reference.arrays import NumericArray


class ReferenceVersion(TypedDict):
    version: str
    commit: NotRequired[str]


class RunnerOption(TypedDict):
    name: str
    folds: list[int]
    mirroring_axes: list[int]
    step_fraction: float


class RunnerFixture(TypedDict):
    plans: dict[str, object]
    dataset_json: dict[str, object]
    image_seed: int
    weight_seeds: list[int]
    image_shape: list[int]
    image_affine: list[list[float]]
    options: list[RunnerOption]


class LabelCase(TypedDict):
    name: str
    dataset_json: dict[str, object]


class FixtureInputs(TypedDict):
    schema_version: str
    reference: dict[str, ReferenceVersion]
    label_cases: list[LabelCase]
    runner: RunnerFixture


class RunnerRecording(TypedDict):
    schema_version: str
    runner: RunnerFixture
    reference: dict[str, ReferenceVersion]
    framework: str
    platform: dict[str, str]
    dependencies: dict[str, str]
    image_properties: dict[str, object]
    preprocessed_properties: dict[str, object]


type RecordedArrays = dict[str, NumericArray]
type UpstreamRecording = tuple[RunnerRecording, RecordedArrays]
