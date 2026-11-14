"""Shared fixtures: recorded runner outputs, tiny plans and tiny network specs."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from nnunet_reference.fixture_types import (
    RecordedArrays,
    RunnerRecording,
    UpstreamRecording,
)

from nnunet_mlx.model.topology import PlainConvUNetSpec, ResidualEncoderUNetSpec


@pytest.fixture
def controlled_runner_recording() -> UpstreamRecording:
    """Load tracked head-conversion and rounding operands without sourced images."""
    return _load_recording("tests/fixtures/darwin-arm64/upstream_runner.npz")


@pytest.fixture
def upstream_runner_recording() -> UpstreamRecording:
    """Load the provisioned image-dependent upstream reference."""
    return _load_recording("work/real-reference/upstream_runner.npz")


def _load_recording(relative_path: str) -> UpstreamRecording:
    """Read a restricted NumPy recording with an actionable missing-asset error."""
    import numpy as np

    path = Path(__file__).resolve().parents[1] / relative_path
    if not path.is_file():
        pytest.fail(
            f"missing upstream recording: {relative_path}; "
            "run python scripts/preflight.py for preparation instructions"
        )
    with np.load(path, allow_pickle=False) as saved:
        metadata = cast(RunnerRecording, json.loads(str(saved["metadata"].item())))
        arrays = {name: saved[name] for name in saved.files if name != "metadata"}
    return metadata, cast(RecordedArrays, arrays)


@pytest.fixture
def tiny_model_plans() -> dict[str, object]:
    """Provide compact plans for synthetic model-folder tests."""
    from nnunet_reference.oracle.fixtures import raw_inference_fixture

    plans, _ = raw_inference_fixture()
    return copy.deepcopy(plans)


@pytest.fixture
def tiny_model_dataset() -> dict[str, object]:
    """Provide a small labeled dataset definition for model tests."""
    from nnunet_reference.oracle.fixtures import raw_inference_fixture

    _, dataset_json = raw_inference_fixture()
    return copy.deepcopy(dataset_json)


@pytest.fixture
def tiny_spec() -> PlainConvUNetSpec:
    """Build a small 3D plain U-Net topology for tests."""
    return PlainConvUNetSpec(
        input_channels=1,
        num_classes=3,
        features=(4, 8),
        kernel_sizes=((3, 3, 3), (3, 3, 3)),
        strides=((1, 1, 1), (2, 2, 2)),
        encoder_convs=(2, 2),
        decoder_convs=(2,),
        conv_bias=True,
    )


@pytest.fixture
def tiny_two_class_spec(tiny_spec: PlainConvUNetSpec) -> PlainConvUNetSpec:
    """Build a small two-class 3D plain U-Net topology."""
    return replace(tiny_spec, num_classes=2)


@pytest.fixture
def tiny_spec_2d() -> PlainConvUNetSpec:
    """Build a small 2D plain U-Net topology for tests."""
    return PlainConvUNetSpec(
        input_channels=1,
        num_classes=3,
        features=(4, 8),
        kernel_sizes=((3, 3), (3, 3)),
        strides=((1, 1), (2, 2)),
        encoder_convs=(2, 2),
        decoder_convs=(2,),
        conv_bias=True,
    )


@pytest.fixture
def tiny_residual_spec() -> ResidualEncoderUNetSpec:
    """Build a small 3D basic residual U-Net topology."""
    return ResidualEncoderUNetSpec(
        input_channels=1,
        num_classes=3,
        features=(4, 8, 16),
        kernel_sizes=((3, 3, 3), (3, 3, 3), (3, 3, 3)),
        strides=((1, 1, 1), (2, 2, 2), (2, 2, 2)),
        encoder_blocks=(1, 3, 4),
        decoder_convs=(1, 1),
        conv_bias=True,
    )


@pytest.fixture
def tiny_residual_spec_2d() -> ResidualEncoderUNetSpec:
    """Build a small 2D basic residual U-Net topology."""
    return ResidualEncoderUNetSpec(
        input_channels=1,
        num_classes=3,
        features=(4, 8, 16),
        kernel_sizes=((3, 3), (3, 3), (3, 3)),
        strides=((1, 1), (2, 2), (2, 2)),
        encoder_blocks=(1, 3, 4),
        decoder_convs=(1, 1),
        conv_bias=True,
    )


@pytest.fixture
def tiny_bottleneck_spec() -> ResidualEncoderUNetSpec:
    """Build a small 3D bottleneck residual U-Net topology."""
    return ResidualEncoderUNetSpec(
        input_channels=1,
        num_classes=3,
        features=(4, 8, 16),
        kernel_sizes=((3, 3, 3), (3, 3, 3), (3, 3, 3)),
        strides=((1, 1, 1), (2, 2, 2), (2, 2, 2)),
        encoder_blocks=(1, 2, 2),
        decoder_convs=(1, 1),
        conv_bias=True,
        bottleneck_channels=(2, 4, 8),
    )


@pytest.fixture
def tiny_bottleneck_spec_2d() -> ResidualEncoderUNetSpec:
    """Build a small 2D bottleneck residual U-Net topology."""
    return ResidualEncoderUNetSpec(
        input_channels=1,
        num_classes=3,
        features=(4, 8, 16),
        kernel_sizes=((3, 3), (3, 3), (3, 3)),
        strides=((1, 1), (2, 2), (2, 2)),
        encoder_blocks=(1, 2, 2),
        decoder_convs=(1, 1),
        conv_bias=True,
        bottleneck_channels=(2, 4, 8),
    )
