"""Pinned nnU-Net oracle: inference."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module as TorchModule

import itertools
from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path

import numpy as np
import numpy.typing as npt

from nnunet_mlx.data.labels import parse_dataset_label_semantics
from nnunet_mlx.model.topology import PlainConvUNetSpec
from nnunet_mlx.storage.weights import convert_state_dict, load_converted_state
from nnunet_reference.oracle.fixtures import (
    raw_inference_fixture,
    restoration_properties,
)
from nnunet_reference.oracle.metrics import (
    ContractParityResult,
    ExactParityResult,
    FloatArray,
    ForwardParityResult,
    array_metrics,
    exact_array_parity,
    measure_forward_parity,
    nested_tuple,
)
from nnunet_reference.oracle.network import (
    build_torch_plain_conv_unet,
    converted_candidate,
    real_image_input,
)


def run_prediction_restoration_parity(
    plans_payload: Mapping[str, object],
    dataset_json: Mapping[str, object],
) -> ContractParityResult:
    """Compare restoration on one fixed synthetic 3-D two-class logit array.

    Probabilities use a 1e-6 allclose gate; labels and dtype are checked exactly.
    Plans and dataset metadata are supplied by the caller.
    """

    from nnunet_mlx.plans import PlansContract
    from nnunet_mlx.prediction.export import restore_prediction

    export_module = import_module("nnunetv2.inference.export_prediction")
    plans_module = import_module("nnunetv2.utilities.plans_handling.plans_handler")
    source_plans = plans_module.PlansManager(dict(plans_payload))
    source_configuration = source_plans.get_configuration("3d_fullres")
    source_labels = source_plans.get_label_manager(dict(dataset_json))
    properties = restoration_properties()
    logits = np.random.default_rng(103).normal(size=(2, 3, 4, 5)).astype(np.float32)
    expected_segmentation, expected_probabilities = (
        export_module.convert_predicted_logits_to_segmentation_with_correct_shape(
            logits,
            source_plans,
            source_configuration,
            source_labels,
            dict(properties),
            return_probabilities=True,
        )
    )
    candidate_plans = PlansContract.from_payload(plans_payload, configuration="3d_fullres")
    semantics = parse_dataset_label_semantics(dataset_json)
    actual = restore_prediction(
        logits,
        properties,
        candidate_plans,
        semantics,
        return_probabilities=True,
    )
    if actual.probabilities is None:
        raise AssertionError("candidate probability restoration returned no probabilities")
    return ContractParityResult(
        arrays=(
            array_metrics(
                np.asarray(expected_probabilities, dtype=np.float32),
                actual.probabilities,
                name="restored_probabilities",
                rtol=1e-6,
                atol=1e-6,
            ),
        ),
        exact=(
            exact_array_parity(
                name="restored_segmentation",
                expected=np.asarray(expected_segmentation, dtype=np.uint16),
                actual=actual.segmentation,
            ),
        ),
    )


def run_head_conversion_parity(*, seed: int = 25) -> ContractParityResult:
    """Compare head activation and label conversion with the pinned LabelManager."""

    torch = import_module("torch")
    label_handling = import_module("nnunetv2.utilities.label_handling.label_handling")
    from nnunet_mlx.prediction.inference import (
        logits_to_probabilities,
        probabilities_to_segmentation,
    )

    rng = np.random.default_rng(seed)
    class_logits = rng.normal(size=(3, 5, 6, 7)).astype(np.float32)
    region_logits = rng.normal(size=(2, 5, 6, 7)).astype(np.float32)
    class_manager = label_handling.LabelManager({"background": 0, "class_1": 1, "class_2": 2}, None)
    region_manager = label_handling.LabelManager(
        {"background": 0, "region_1": (0, 1), "region_2": (0, 2)}, [1, 2]
    )
    expected_class_probabilities = np.asarray(
        class_manager.apply_inference_nonlin(torch.from_numpy(class_logits))
    )
    expected_region_probabilities = np.asarray(
        region_manager.apply_inference_nonlin(torch.from_numpy(region_logits))
    )
    expected_class_segmentation = np.asarray(
        class_manager.convert_logits_to_segmentation(torch.from_numpy(class_logits))
    )
    expected_region_segmentation = np.asarray(
        region_manager.convert_logits_to_segmentation(torch.from_numpy(region_logits))
    )
    actual_class_probabilities = logits_to_probabilities(class_logits, has_regions=False)
    actual_region_probabilities = logits_to_probabilities(region_logits, has_regions=True)
    actual_class_segmentation = probabilities_to_segmentation(
        actual_class_probabilities, regions_class_order=None
    )
    actual_region_segmentation = probabilities_to_segmentation(
        actual_region_probabilities, regions_class_order=(1, 2)
    )
    return ContractParityResult(
        arrays=(
            array_metrics(
                expected_class_probabilities.astype(np.float32),
                actual_class_probabilities,
                name="class_head_probabilities",
                rtol=1e-6,
                atol=1e-6,
            ),
            array_metrics(
                expected_region_probabilities.astype(np.float32),
                actual_region_probabilities,
                name="region_head_probabilities",
                rtol=1e-6,
                atol=1e-6,
            ),
        ),
        exact=(
            ExactParityResult(
                name="class_head_segmentation",
                expected=nested_tuple(expected_class_segmentation),
                actual=nested_tuple(actual_class_segmentation),
            ),
            ExactParityResult(
                name="region_head_segmentation",
                expected=nested_tuple(expected_region_segmentation),
                actual=nested_tuple(actual_region_segmentation),
            ),
        ),
    )


def run_sliding_window_parity(
    spec: PlainConvUNetSpec,
    *,
    spatial: tuple[int, ...],
    patch: tuple[int, ...],
    mirror_axes: tuple[int, ...] = (),
    seed: int = 23,
    step_fraction: float = 0.5,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> ForwardParityResult:
    """Compare Torch CPU and MLX sliding-window logits on a real image crop.

    The Torch helper uses explicit float32 accumulation rather than the source
    predictor's default float16 buffer. The pass gate is raw-logit allclose.
    """

    from nnunet_mlx.prediction.sliding_window import predict_logits

    source = build_torch_plain_conv_unet(spec, seed=seed)
    from nnunet_mlx.model.network import PlainConvUNet

    candidate = PlainConvUNet(spec)
    load_converted_state(candidate, convert_state_dict(source.state_dict()))
    values = real_image_input(spatial, spec.input_channels)
    expected = pinned_sliding_window_logits(
        source,
        values,
        patch=patch,
        num_classes=spec.num_classes,
        mirror_axes=mirror_axes,
        step_fraction=step_fraction,
    )
    actual = predict_logits(
        candidate,
        values,
        patch_size=patch,
        num_classes=spec.num_classes,
        mirroring_axes=mirror_axes,
        step_fraction=step_fraction,
    )
    return ForwardParityResult(
        expected=expected,
        actual=actual,
        metrics=measure_forward_parity(
            expected[None].astype(np.float32),
            actual[None],
            seed=seed,
            rtol=rtol,
            atol=atol,
        ),
    )


def run_fold_ensemble_parity(
    spec: PlainConvUNetSpec,
    *,
    spatial: tuple[int, ...],
    patch: tuple[int, ...],
    mirror_axes: tuple[int, ...] = (),
    seed: int = 30,
    step_fraction: float = 0.5,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> ContractParityResult:
    """Compare two independently seeded folds and their averaged logits on a real image.

    Torch uses the float32 sliding-window helper; the gate is allclose on the
    mean logits, not segmentation accuracy or a pretrained-fold qualification.
    """

    from nnunet_mlx.prediction.inference import average_fold_logits

    sources = tuple(
        build_torch_plain_conv_unet(spec, seed=network_seed) for network_seed in (seed, seed + 1)
    )
    candidates = [converted_candidate(spec, source) for source in sources]
    values = real_image_input(spatial, spec.input_channels)
    total: FloatArray | None = None
    for source in sources:
        logits = pinned_sliding_window_logits(
            source,
            values,
            patch=patch,
            num_classes=spec.num_classes,
            mirror_axes=mirror_axes,
            step_fraction=step_fraction,
        )
        total = logits.copy() if total is None else total + logits
    if total is None:
        raise AssertionError("fold ensemble oracle unexpectedly produced no logits")
    total /= np.float32(len(sources))
    actual = np.asarray(average_fold_logits(
        tuple(candidates),
        values,
        patch_size=patch,
        num_outputs=spec.num_classes,
        mirroring_axes=mirror_axes,
        step_fraction=step_fraction,
    ), dtype=np.float32)
    return ContractParityResult(
        arrays=(
            array_metrics(
                total,
                actual,
                name="fold_averaged_logits",
                rtol=rtol,
                atol=atol,
            ),
        ),
    )


def run_configuration_ensemble_parity(
    spec: PlainConvUNetSpec,
    *,
    work_dir: Path,
    spatial: tuple[int, ...],
    mirror_axes: tuple[int, ...] = (0, 1, 2),
    seed: int = 34,
    step_fraction: float = 0.5,
) -> ContractParityResult:
    """Compare two probability-file ensembles on a CT crop in a fresh ``work_dir``.

    Creates configuration, expected and actual subdirectories and files. The
    gate requires allclose probabilities plus exact labels and image geometry.
    Existing subdirectories cause an error rather than being reused.
    """

    from nnunet_mlx.prediction.ensembling import merge_probability_files

    torch = import_module("torch")
    ensembling = import_module("nnunetv2.ensembling.ensemble")
    export_module = import_module("nnunetv2.inference.export_prediction")
    plans_module = import_module("nnunetv2.utilities.plans_handling.plans_handler")
    plans_payload, dataset_json = raw_inference_fixture()
    source_plans = plans_module.PlansManager(dict(plans_payload))
    configuration = source_plans.get_configuration("3d_fullres")
    labels = source_plans.get_label_manager(dict(dataset_json))
    values = real_image_input(spatial, spec.input_channels)
    folders: list[Path] = []
    for index, network_seed in enumerate((seed, seed + 1)):
        source = build_torch_plain_conv_unet(spec, seed=network_seed)
        logits = pinned_sliding_window_logits(
            source,
            values,
            patch=tuple(int(size) for size in configuration.patch_size),
            num_classes=spec.num_classes,
            mirror_axes=mirror_axes,
            step_fraction=step_fraction,
        )
        folder = work_dir / f"configuration_{index}"
        folder.mkdir(parents=True)
        export_module.export_prediction_from_logits(
            torch.from_numpy(logits),
            dict(restoration_properties()),
            configuration,
            source_plans,
            dict(dataset_json),
            str(folder / "case_001"),
            save_probabilities=True,
        )
        folders.append(folder)
    expected_prefix = work_dir / "expected" / "case_001"
    expected_prefix.parent.mkdir(parents=True)
    ensembling.merge_files(
        [str(folder / "case_001.npz") for folder in folders],
        str(expected_prefix),
        ".nii.gz",
        source_plans.image_reader_writer_class(),
        labels,
        True,
    )
    semantics = parse_dataset_label_semantics(dataset_json)
    merge_probability_files(
        [folder / "case_001.npz" for folder in folders],
        work_dir / "actual" / "case_001",
        file_ending=".nii.gz",
        image_reader_writer="SimpleITKIO",
        semantics=semantics,
        overwrite=False,
        save_probabilities=True,
    )
    expected_labels, expected_geometry = _sitk_label_payload(f"{expected_prefix}.nii.gz")
    actual_labels, actual_geometry = _sitk_label_payload(
        str(work_dir / "actual" / "case_001.nii.gz")
    )
    with np.load(f"{expected_prefix}.npz") as payload:
        expected_merged = np.asarray(payload["probabilities"], dtype=np.float32)
    with np.load(work_dir / "actual" / "case_001.npz") as payload:
        actual_merged = np.asarray(payload["probabilities"], dtype=np.float32)
    return ContractParityResult(
        arrays=(
            array_metrics(
                expected_merged,
                actual_merged,
                name="merged_probabilities",
                rtol=1e-6,
                atol=1e-6,
            ),
        ),
        exact=(
            ExactParityResult(
                name="merged_segmentation",
                expected=nested_tuple(expected_labels),
                actual=nested_tuple(actual_labels),
            ),
            ExactParityResult(
                name="merged_geometry",
                expected=expected_geometry,
                actual=actual_geometry,
            ),
        ),
    )


def _sitk_label_payload(path: str) -> tuple[FloatArray, tuple[object, ...]]:
    """Read labels and physical geometry from a SimpleITK image."""
    sitk = import_module("SimpleITK")
    image = sitk.ReadImage(path)
    labels = np.asarray(sitk.GetArrayFromImage(image), dtype=np.float32).transpose(2, 1, 0)
    return (
        np.ascontiguousarray(labels, dtype=np.float32),
        (
            tuple(float(value) for value in image.GetSpacing()),
            tuple(float(value) for value in image.GetOrigin()),
            tuple(float(value) for value in image.GetDirection()),
        ),
    )


def pinned_sliding_window_logits(
    model: TorchModule,
    values: npt.NDArray[np.float32],
    *,
    patch: Sequence[int],
    num_classes: int,
    mirror_axes: Sequence[int],
    step_fraction: float = 0.5,
) -> FloatArray:
    """Run Torch CPU tiles with nnU-Net placement, mirrors and Gaussian weights.

    This helper explicitly accumulates and normalizes in float32; it is not a
    direct invocation of the upstream predictor's float16 accumulation path.
    It returns channel-first, unpadded float32 logits.
    """

    torch = import_module("torch")
    sliding_window = import_module("nnunetv2.inference.sliding_window_prediction")
    data, revert = sliding_window.pad_nd_image(
        torch.from_numpy(np.asarray(values, dtype=np.float32)),
        tuple(patch),
        "constant",
        {"value": 0},
        True,
        None,
    )
    steps = sliding_window.compute_steps_for_sliding_window(
        tuple(int(size) for size in data.shape[1:]), tuple(patch), step_fraction
    )
    gaussian = sliding_window.compute_gaussian(
        tuple(patch),
        sigma_scale=1.0 / 8.0,
        value_scaling_factor=10,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    predicted = torch.zeros((num_classes, *data.shape[1:]), dtype=torch.float32)
    normalizer = torch.zeros(data.shape[1:], dtype=torch.float32)
    # The upstream LRU cache can hold a Gaussian created under inference_mode;
    # autograd cannot save that tensor, and this oracle never computes gradients.
    with torch.no_grad():
        for starts in itertools.product(*steps):
            slices = (
                slice(None),
                *(slice(start, start + size) for start, size in zip(starts, patch, strict=True)),
            )
            prediction = _torch_mirrored_prediction(model, data[slices][None], tuple(mirror_axes))
            predicted[slices] += prediction[0] * gaussian
            normalizer[slices[1:]] += gaussian
    logits = predicted / normalizer
    return np.asarray(logits[(slice(None), *revert[1:])].detach().numpy(), dtype=np.float32)


def _torch_mirrored_prediction(
    model: TorchModule, tile: Tensor, mirror_axes: tuple[int, ...]
) -> Tensor:
    """Average source predictions over the requested mirror-axis combinations."""
    torch = import_module("torch")
    prediction = model(tile)
    axes = [int(axis) + 2 for axis in mirror_axes]
    combinations = [
        combination
        for count in range(1, len(axes) + 1)
        for combination in itertools.combinations(axes, count)
    ]
    for flipped_axes in combinations:
        flipped = torch.flip(tile, list(flipped_axes))
        prediction = prediction + torch.flip(model(flipped), list(flipped_axes))
    return prediction / (len(combinations) + 1)
