"""Reference helpers for direct upstream model comparisons in tests."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from importlib import import_module
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Final, Literal, Protocol, TypedDict, cast

if TYPE_CHECKING:
    from torch import Tensor
    from torch import device as Device
    from torch import dtype as TorchDtype
    from torch.nn import Module as TorchModule

import numpy as np
import numpy.typing as npt

from nnunet_mlx.plans import read_json_object
from nnunet_reference.arrays import NumericArray
from nnunet_reference.oracle.checkpoint import load_reference_checkpoint

type FloatArray = npt.NDArray[np.float32]
type Properties = dict[str, object]
type Slicer = tuple[slice, ...]

_SOURCE_SLICERS: Final = "_internal_get_sliding_window_slicers"
_SOURCE_PREDICT_TILE: Final = "_internal_maybe_mirror_and_predict"


class Architecture(TypedDict):
    network_class_name: str
    arch_kwargs: dict[str, object]
    _kw_requires_import: list[str]


class ConfigurationPayload(TypedDict):
    architecture: Architecture


class SourceCheckpointInit(TypedDict):
    configuration: str


class SourceCheckpoint(TypedDict):
    init_args: SourceCheckpointInit
    network_weights: dict[str, Tensor]
    inference_allowed_mirroring_axes: tuple[int, ...] | list[int] | None
    trainer_name: str


class SourceInfo(TypedDict):
    properties: Properties
    preprocessed_shape: list[int]
    inference_s: float


class SourceReader(Protocol):
    def read_images(self, image_fnames: list[str]) -> tuple[FloatArray, Properties]:
        """Read source image channels and their spatial properties."""
        ...
    def write_seg(self, seg: NumericArray, output_fname: str, properties: Properties) -> None:
        """Write a source segmentation with its spatial properties."""
        ...


class SourcePreprocessor(Protocol):
    def run_case_npy(
        self,
        data: FloatArray,
        seg: None,
        properties: Properties,
        plans_manager: SourcePlans,
        configuration_manager: SourceConfiguration,
        dataset_json: Properties,
    ) -> tuple[FloatArray, NumericArray | None, Properties]:
        """Preprocess source image arrays with plans and dataset metadata."""
        ...


class PreprocessorFactory(Protocol):
    def __call__(self, *, verbose: bool) -> SourcePreprocessor:
        """Construct the source preprocessor."""
        ...


class SourceConfiguration(Protocol):
    configuration: ConfigurationPayload
    patch_size: list[int]
    preprocessor_class: PreprocessorFactory


class SourceLabels(Protocol):
    num_segmentation_heads: int


class SourcePlans(Protocol):
    def get_configuration(self, configuration_name: str) -> SourceConfiguration:
        """Resolve a named source plans configuration."""
        ...
    def get_label_manager(self, dataset_json: Properties) -> SourceLabels:
        """Create source label semantics from dataset metadata."""
        ...
    def image_reader_writer_class(self) -> SourceReader:
        """Construct the plans-selected image reader and writer."""
        ...


class PlansModule(Protocol):
    def PlansManager(self, plans_file_or_dict: Properties) -> SourcePlans:
        """Construct an upstream plans manager."""
        ...


class NetworkModule(Protocol):
    def get_network_from_plans(
        self,
        arch_class_name: str,
        arch_kwargs: Properties,
        arch_kwargs_req_import: list[str],
        input_channels: int,
        output_channels: int,
        *,
        allow_init: bool,
        deep_supervision: bool,
    ) -> TorchModule:
        """Build a reference network from its plans architecture."""
        ...


class SourcePredictor(Protocol):
    def manual_initialization(
        self,
        network: TorchModule,
        plans_manager: SourcePlans,
        configuration_manager: SourceConfiguration,
        parameters: None,
        dataset_json: Properties,
        trainer_name: str,
        inference_allowed_mirroring_axes: tuple[int, ...] | list[int] | None,
    ) -> None:
        """Initialize a source predictor with a network and fold parameters."""
        ...


class PredictionModule(Protocol):
    def nnUNetPredictor(
        self,
        *,
        tile_step_size: float,
        use_gaussian: bool,
        use_mirroring: bool,
        perform_everything_on_device: bool,
        device: Device,
        verbose: bool,
        verbose_preprocessing: bool,
        allow_tqdm: bool,
    ) -> SourcePredictor:
        """Construct the upstream sliding-window predictor."""
        ...


class PaddingModule(Protocol):
    def pad_nd_image(
        self,
        image: Tensor,
        new_shape: Sequence[int],
        mode: str,
        kwargs: dict[str, int],
        return_slicer: Literal[True],
        shape_must_be_divisible_by: None,
    ) -> tuple[Tensor, Slicer]:
        """Pad an input array using the source utility."""
        ...


class SlidingModule(Protocol):
    def compute_gaussian(
        self,
        tile_size: tuple[int, ...],
        *,
        sigma_scale: float,
        value_scaling_factor: int,
        dtype: TorchDtype,
        device: Device,
    ) -> Tensor:
        """Construct the source sliding-window Gaussian importance map."""
        ...


class TorchRuntime(Protocol):
    float32: TorchDtype

    def set_num_threads(self, num_threads: int) -> None:
        """Set the Torch CPU thread count."""
        ...
    def from_numpy(self, ndarray: FloatArray) -> Tensor:
        """Create a Torch tensor from NumPy values."""
        ...
    def zeros(self, size: Sequence[int], *, dtype: TorchDtype) -> Tensor:
        """Create a zero-filled Torch tensor."""
        ...
    def device(self, type: str) -> Device:
        """Construct a Torch device descriptor."""
        ...
    def inference_mode(self) -> AbstractContextManager[None]:
        """Enter Torch inference mode for the reference run."""
        ...



def source_logits(
    model_dir: Path,
    inputs: list[Path],
    folds: tuple[int | str, ...],
    checkpoint_name: str,
    network_class: str | None = None,
    *,
    use_mirroring: bool = True,
) -> tuple[FloatArray, SourceReader, SourcePlans, SourceConfiguration, SourceLabels, SourceInfo]:
    """Predict CPU tiles with one Torch thread and source-order FP32 accumulation.

    Sets Torch's process-wide thread count to one without restoring it. Inputs and
    checkpoints are caller-supplied; source reader/preprocessor and model execution
    require a compatible installed nnU-Net environment.
    """
    torch = cast(TorchRuntime, import_module("torch"))
    torch.set_num_threads(1)
    plans_module = cast(
        PlansModule, import_module("nnunetv2.utilities.plans_handling.plans_handler")
    )
    network_module = cast(NetworkModule, import_module("nnunetv2.utilities.get_network_from_plans"))
    prediction_module = cast(
        PredictionModule, import_module("nnunetv2.inference.predict_from_raw_data")
    )
    padding_module = cast(PaddingModule, import_module("acvl_utils.cropping_and_padding.padding"))
    sliding_module = cast(
        SlidingModule, import_module("nnunetv2.inference.sliding_window_prediction")
    )
    plans = plans_module.PlansManager(read_json_object(model_dir / "plans.json"))
    dataset = read_json_object(model_dir / "dataset.json")
    checkpoints: list[SourceCheckpoint] = [
        cast(
            SourceCheckpoint,
            load_reference_checkpoint(model_dir / f"fold_{fold}" / checkpoint_name),
        )
        for fold in folds
    ]
    configuration_name = checkpoints[0]["init_args"]["configuration"]
    configuration = plans.get_configuration(configuration_name)
    labels = plans.get_label_manager(dataset)
    reader = plans.image_reader_writer_class()
    raw, properties = reader.read_images([str(path) for path in inputs])
    data, _, properties = configuration.preprocessor_class(verbose=False).run_case_npy(
        raw,
        None,
        properties,
        plans,
        configuration,
        dataset,
    )
    padded, revert = padding_module.pad_nd_image(
        torch.from_numpy(np.asarray(data, dtype=np.float32)),
        configuration.patch_size,
        "constant",
        {"value": 0},
        True,
        None,
    )
    total = None
    started = perf_counter()
    for fold, checkpoint in zip(folds, checkpoints, strict=True):
        if checkpoint["init_args"]["configuration"] != configuration_name:
            raise ValueError(f"Fold {fold} uses a different configuration")
        architecture = configuration.configuration["architecture"]
        network = network_module.get_network_from_plans(
            network_class or architecture["network_class_name"],
            architecture["arch_kwargs"],
            architecture["_kw_requires_import"],
            int(data.shape[0]),
            int(labels.num_segmentation_heads),
            allow_init=False,
            deep_supervision=False,
        ).eval()
        network.load_state_dict(checkpoint["network_weights"])
        predictor = prediction_module.nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=use_mirroring,
            perform_everything_on_device=False,
            device=torch.device("cpu"),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        axes = checkpoint["inference_allowed_mirroring_axes"]
        predictor.manual_initialization(
            network,
            plans,
            configuration,
            None,
            dataset,
            checkpoint["trainer_name"],
            axes or None,
        )
        gaussian = sliding_module.compute_gaussian(
            tuple(configuration.patch_size),
            sigma_scale=1 / 8,
            value_scaling_factor=10,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        sums = torch.zeros(
            (int(labels.num_segmentation_heads), *padded.shape[1:]), dtype=torch.float32
        )
        counts = torch.zeros(padded.shape[1:], dtype=torch.float32)
        # Pinned nnU-Net exposes these execution steps only through private methods.
        get_slicers = cast(
            Callable[[Sequence[int]], list[Slicer]],
            getattr(predictor, _SOURCE_SLICERS),
        )
        predict_tile = cast(
            "Callable[[Tensor], Tensor]",
            getattr(predictor, _SOURCE_PREDICT_TILE),
        )
        slicers = get_slicers(padded.shape[1:])
        for index, slicer in enumerate(slicers):
            with torch.inference_mode():
                prediction = predict_tile(padded[slicer][None])[0]
            sums[slicer] += prediction * gaussian
            counts[slicer[1:]] += gaussian
            print(f"source fold={fold} tile={index + 1}/{len(slicers)}", flush=True)
        logits = cast(FloatArray, (sums / counts)[(slice(None), *revert[1:])].numpy())
        total = logits.copy() if total is None else total + logits
    assert total is not None
    total /= np.float32(len(folds))
    return (
        total,
        reader,
        plans,
        configuration,
        labels,
        {
            "properties": properties,
            "preprocessed_shape": list(data.shape),
            "inference_s": perf_counter() - started,
        },
    )
