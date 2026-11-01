"""Pinned nnU-Net oracle: preprocessing."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import cast

import numpy as np

from nnunet_reference.oracle.metrics import (
    ContractParityResult,
    ExactParityResult,
    array_metrics,
    exact_array_parity,
    nested_tuple,
)


def run_plan_configuration_parity(
    plans: Mapping[str, object], configurations: Sequence[str]
) -> ContractParityResult:
    """Compare resolved plan configuration payloads with pinned nnU-Net."""

    from nnunet_mlx.plans import PlansContract

    plans_module = import_module("nnunetv2.utilities.plans_handling.plans_handler")
    source = plans_module.PlansManager(dict(plans))
    exact: list[ExactParityResult] = []
    for name in configurations:
        expected = source.get_configuration(name).configuration
        actual = PlansContract.from_payload(
            plans, configuration=name
        ).configuration.raw
        exact.append(
            ExactParityResult(
                name=f"plan_configuration_{name}",
                expected=expected,
                actual=actual,
            )
        )
    return ContractParityResult(arrays=(), exact=tuple(exact))


def run_raw_preprocessing_parity(
    plans_payload: Mapping[str, object],
    dataset_json: Mapping[str, object],
    *,
    normalization_scheme: str = "NoNormalization",
    use_mask_for_norm: bool = False,
    include_previous_stage: bool = False,
) -> ContractParityResult:
    """Compare one synthetic raw-case preprocessing result with pinned nnU-Net.

    Uses a fixed single-channel 3-D fixture and optional previous-stage labels.
    The supplied plans are copied before the normalization options are changed;
    selected crop/shape properties and any previous-stage labels are compared
    exactly, while the preprocessed image uses numeric tolerances.
    """

    from nnunet_mlx.data.preprocessing import preprocess_raw_case
    from nnunet_mlx.plans import PlansContract

    plans_module = import_module("nnunetv2.utilities.plans_handling.plans_handler")
    preprocessor_module = import_module(
        "nnunetv2.preprocessing.preprocessors.default_preprocessor"
    )
    fixture_plans = copy.deepcopy(dict(plans_payload))
    configurations = cast(dict[str, object], fixture_plans["configurations"])
    base = cast(dict[str, object], configurations["base"])
    base["normalization_schemes"] = [normalization_scheme]
    base["use_mask_for_norm"] = [use_mask_for_norm]
    intensity = (
        {
            "mean": 5.0,
            "std": 2.0,
            "percentile_00_5": 2.0,
            "percentile_99_5": 8.0,
        }
        if normalization_scheme == "CTNormalization"
        else {}
    )
    fixture_plans["foreground_intensity_properties_per_channel"] = {"0": intensity}
    source_plans = plans_module.PlansManager(fixture_plans)
    source_configuration = source_plans.get_configuration("3d_fullres")
    data = np.zeros((1, 5, 6, 7), dtype=np.float32)
    data[0, 1:5, 1:5, 2:7] = np.random.default_rng(101).uniform(
        1, 10, size=(4, 4, 5)
    )
    properties: dict[str, object] = {"spacing": (2.5, 0.8, 0.7)}
    previous_stage = None
    if include_previous_stage:
        previous_stage = np.zeros(data.shape[1:], dtype=np.int16)[None]
        previous_stage[0, 2:5, 2:5, 3:7] = 1
        previous_stage[0, 2, 3, 4] = 2
    expected_data, expected_segmentation, expected_properties = (
        preprocessor_module.DefaultPreprocessor(verbose=False).run_case_npy(
            data,
            previous_stage,
            dict(properties),
            source_plans,
            source_configuration,
            dict(dataset_json),
        )
    )
    candidate_plans = PlansContract.from_payload(
        fixture_plans, configuration="3d_fullres"
    )
    actual = preprocess_raw_case(
        data,
        properties,
        candidate_plans,
        segmentation_previous_stage=previous_stage,
    )
    property_names = (
        "shape_before_cropping",
        "bbox_used_for_cropping",
        "shape_after_cropping_and_before_resampling",
    )
    exact = [
        ExactParityResult(
            name=f"preprocessing_{name}",
            expected=nested_tuple(expected_properties[name]),
            actual=nested_tuple(actual.properties[name]),
        )
        for name in property_names
    ]
    if include_previous_stage:
        if expected_segmentation is None or actual.previous_stage is None:
            raise AssertionError("cascade preprocessing unexpectedly dropped segmentation")
        exact.append(
            exact_array_parity(
                name="raw_cascade_previous_stage",
                expected=np.asarray(expected_segmentation, dtype=np.int64),
                actual=np.asarray(actual.previous_stage, dtype=np.int64),
            )
        )
    return ContractParityResult(
        arrays=(
            array_metrics(
                np.asarray(expected_data, dtype=np.float32),
                actual.data,
                name=f"raw_preprocessing_{normalization_scheme}_{use_mask_for_norm}",
                rtol=1e-5,
                atol=1e-5,
            ),
        ),
        exact=tuple(exact),
    )
