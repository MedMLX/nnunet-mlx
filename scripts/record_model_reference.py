"""Record resumable CPU references with the pinned upstream nnU-Net predictor.

The upstream predictor owns mirroring, Gaussian weighting, half accumulation and
fold averaging. Wrappers cache its completed tile/fold results without changing
their arithmetic. Model/input metadata and runtime settings must match on resume.
Only trusted upstream NIfTI model folders without a cascade are supported here.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import signal
import sys
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Literal, Protocol, cast, overload

import numpy as np

from nnunet_mlx.commands.parsing import parse_selected_folds
from nnunet_mlx.errors import ContractError
from nnunet_mlx.plans import read_json_object
from nnunet_mlx.storage.upstream import detect_upstream_folds

ROOT = Path(__file__).resolve().parents[1]
_SOURCE_TILE = "_internal_maybe_mirror_and_predict"
_SOURCE_SLICERS = "_internal_get_sliding_window_slicers"
sys.path.insert(0, str(ROOT / "tests"))
from nnunet_reference.arrays import NumericArray  # noqa: E402
from nnunet_reference.model_folder import (  # noqa: E402
    SourceConfiguration,
    SourceLabels,
    SourcePlans,
    SourcePredictor,
)
from nnunet_reference.runner_network import TorchApi  # noqa: E402

if TYPE_CHECKING:
    from torch import Tensor
    from torch import device as Device


class ReferenceTorch(TorchApi, Protocol):
    def set_num_interop_threads(self, count: int) -> None:
        """Bound inter-operation CPU scheduling."""
        ...


class ReferencePredictor(SourcePredictor, Protocol):
    plans_manager: SourcePlans
    configuration_manager: SourceConfiguration
    label_manager: SourceLabels
    dataset_json: dict[str, object]
    predict_sliding_window_return_logits: Callable[[Tensor], Tensor]

    def initialize_from_trained_model_folder(
        self,
        folder: str,
        use_folds: tuple[int | str, ...],
        checkpoint_name: str,
    ) -> None:
        """Load the requested trained folds using the upstream loader."""
        ...

    def predict_logits_from_preprocessed_data(self, data: Tensor) -> Tensor:
        """Return upstream fold-averaged logits."""
        ...


class ReferenceFactory(Protocol):
    def nnUNetPredictor(
        self,
        *,
        tile_step_size: float,
        use_gaussian: bool,
        use_mirroring: bool,
        perform_everything_on_device: bool,
        device: Device,
        verbose: bool,
        allow_tqdm: bool,
    ) -> ReferencePredictor:
        """Construct the pinned CPU inference predictor."""
        ...


class ReferenceExport(Protocol):
    @overload
    def convert_predicted_logits_to_segmentation_with_correct_shape(
        self,
        logits: Tensor,
        plans: SourcePlans,
        configuration: SourceConfiguration,
        labels: SourceLabels,
        properties: dict[str, object],
        *,
        return_probabilities: Literal[False],
        num_threads_torch: int,
    ) -> NumericArray: ...

    @overload
    def convert_predicted_logits_to_segmentation_with_correct_shape(
        self,
        logits: Tensor,
        plans: SourcePlans,
        configuration: SourceConfiguration,
        labels: SourceLabels,
        properties: dict[str, object],
        *,
        return_probabilities: Literal[True],
        num_threads_torch: int,
    ) -> tuple[NumericArray, NumericArray]: ...


def _signature(path: Path) -> dict[str, object]:
    info = path.stat()
    return {"path": str(path.resolve()), "size_bytes": info.st_size, "mtime_ns": info.st_mtime_ns}


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Publish complete progress/identity metadata beside its destination."""
    temporary = path.with_suffix(".staging.json")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _save_array(path: Path, values: NumericArray) -> None:
    """Publish a completed NumPy array so interrupted writes never become cache hits."""
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as handle:
            temporary = Path(handle.name)
            np.save(handle, values, allow_pickle=False)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _load_array(path: Path, shape: tuple[int, ...], *, half: bool) -> NumericArray:
    """Require the cached upstream dtype, shape and finite values before reuse."""
    loaded: object = np.load(path, allow_pickle=False)
    if not isinstance(loaded, np.ndarray):
        raise ContractError(f"invalid reference cache array: {path}")
    values = cast(NumericArray, loaded)
    if (
        values.shape != shape
        or values.dtype != (np.float16 if half else np.float32)
        or not np.isfinite(values).all()
    ):
        raise ContractError(f"invalid reference cache array: {path}")
    return values


class CachedReference:
    """Cache upstream tile outputs and complete folds, preserving their original order."""

    def __init__(
        self,
        predictor: ReferencePredictor,
        torch: ReferenceTorch,
        work: Path,
        folds: tuple[int | str, ...],
        *,
        cache_tiles: bool,
    ) -> None:
        self.predictor, self.torch, self.work, self.folds = predictor, torch, work, folds
        self.cache_tiles = cache_tiles
        self.fold_index, self.tile_index, self.tile_count = -1, 0, 0
        self.started = monotonic()
        self.fold_call = predictor.predict_sliding_window_return_logits
        self.tile_call = cast(
            "Callable[[Tensor], Tensor]", getattr(predictor, _SOURCE_TILE)
        )
        self.slicers = cast(
            Callable[[tuple[int, ...]], list[tuple[slice, ...]]],
            getattr(predictor, _SOURCE_SLICERS),
        )
        predictor.predict_sliding_window_return_logits = self.fold
        setattr(predictor, _SOURCE_TILE, self.tile)

    def progress(self, event: str, **fields: object) -> None:
        payload: dict[str, object] = {
            "event": event,
            "updated_at": datetime.now(UTC).isoformat(),
            "elapsed_s": monotonic() - self.started,
            "fold_index": self.fold_index,
            "folds": list(self.folds),
            "tile": self.tile_index,
            "tiles_per_fold": self.tile_count,
            **fields,
        }
        _write_json(self.work / "progress.json", payload)
        print(json.dumps(payload), flush=True)

    def fold(self, data: Tensor) -> Tensor:
        self.fold_index += 1
        fold = self.folds[self.fold_index]
        path = self.work / f"fold_{fold}.npy"
        shape = (self.predictor.label_manager.num_segmentation_heads, *tuple(data.shape[1:]))
        if path.is_file():
            values = _load_array(path, shape, half=True)
            self.progress("fold_cache_hit", fold=fold)
            return self.torch.from_numpy(values)
        self.tile_index = 0
        patch = self.predictor.configuration_manager.patch_size
        padded = list(data.shape[1:])
        for index in range(1, len(patch) + 1):
            padded[-index] = max(padded[-index], patch[-index])
        self.tile_count = len(self.slicers(tuple(padded)))
        self.progress("fold_started", fold=fold)
        output = self.fold_call(data)
        _save_array(path, cast(NumericArray, output.numpy()))
        # Completed folds replace their much larger per-tile FP32 cache.
        for tile in self.work.glob(f"fold_{fold}-tile_*.npy"):
            tile.unlink()
        self.progress("fold_complete", fold=fold)
        return output

    def tile(self, data: Tensor) -> Tensor:
        fold = self.folds[self.fold_index]
        index = self.tile_index
        path = self.work / f"fold_{fold}-tile_{index:05d}.npy"
        shape = (
            int(data.shape[0]),
            self.predictor.label_manager.num_segmentation_heads,
            *tuple(data.shape[2:]),
        )
        if self.cache_tiles and path.is_file():
            output = self.torch.from_numpy(_load_array(path, shape, half=False))
            cached = True
        else:
            output = self.tile_call(data)
            if self.cache_tiles:
                _save_array(path, cast(NumericArray, output.numpy()))
            cached = False
        self.tile_index += 1
        self.progress("tile_complete", fold=fold, cache_hit=cached)
        return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--fold", action="append")
    parser.add_argument("--checkpoint", default="checkpoint_final.pth")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=int, default=10800)
    parser.add_argument("--disable-mirroring", action="store_true")
    parser.add_argument("--cache-folds-only", action="store_true")
    options = parser.parse_args()
    if options.threads < 1 or options.timeout_seconds < 1:
        raise ContractError("thread count and timeout must be positive")
    signal.alarm(options.timeout_seconds)
    # The upstream predictor otherwise caps inference at its default of eight.
    os.environ.update(nnUNet_compile="false", nnUNet_def_n_proc=str(options.threads))
    pinned = {"nnunetv2": "2.8.1", "dynamic-network-architectures": "0.4.4", "torch": "2.13.0"}
    for package, expected in pinned.items():
        if version(package) != expected:
            raise ContractError(f"{package} {expected} is required")
    model, work = Path(options.model).resolve(), Path(options.work).resolve()
    inputs = tuple(Path(path).resolve() for path in options.input)
    folds = parse_selected_folds(options.fold) or detect_upstream_folds(
        model,
        checkpoint_name=options.checkpoint,
    )
    dataset = read_json_object(model / "dataset.json")
    if dataset.get("file_ending") not in (".nii", ".nii.gz"):
        raise ContractError("reference recording supports NIfTI models")
    identity: dict[str, object] = {
        "schema_version": "nnunet-mlx.cpu-reference.v1",
        "model": str(model),
        "plans": read_json_object(model / "plans.json"),
        "dataset": dataset,
        "inputs": [_signature(path) for path in inputs],
        "folds": list(folds),
        "checkpoints": [_signature(model / f"fold_{fold}" / options.checkpoint) for fold in folds],
        "versions": {
            package: version(package) for package in (*pinned, "numpy", "scipy", "scikit-image")
        },
        "threads": options.threads,
        "step_fraction": 0.5,
        "mirroring": not options.disable_mirroring,
        "compilation": False,
        "accumulation": "nnunet-fp16",
        "network_dtype": "float32",
        "device": "cpu",
    }
    work.mkdir(mode=0o700, parents=True, exist_ok=True)
    marker = work / "identity.json"
    if marker.exists() and read_json_object(marker) != identity:
        raise ContractError("reference settings changed; use a new work directory")
    if not marker.exists() and any(work.iterdir()):
        raise ContractError("reference work directory must be empty or have a matching identity")
    _write_json(marker, identity)
    # An interrupted rerun must not leave a completion marker over partial exports.
    (work / "reference.json").unlink(missing_ok=True)
    torch = cast(ReferenceTorch, import_module("torch"))
    torch.set_num_threads(options.threads)
    torch.set_num_interop_threads(1)
    factory = cast(ReferenceFactory, import_module("nnunetv2.inference.predict_from_raw_data"))
    exporter = cast(ReferenceExport, import_module("nnunetv2.inference.export_prediction"))
    predictor = factory.nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=not options.disable_mirroring,
        perform_everything_on_device=False,
        device=torch.device("cpu"),
        verbose=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(str(model), folds, options.checkpoint)
    configuration = predictor.configuration_manager
    if configuration.configuration.get("previous_stage") is not None:
        raise ContractError("use the upstream cascade workflow for previous-stage inputs")
    reader = predictor.plans_manager.image_reader_writer_class()
    raw, properties = reader.read_images([str(path) for path in inputs])
    processed, _, properties = configuration.preprocessor_class(verbose=False).run_case_npy(
        raw,
        None,
        properties,
        predictor.plans_manager,
        configuration,
        predictor.dataset_json,
    )
    cached = CachedReference(
        predictor, torch, work, folds, cache_tiles=not options.cache_folds_only
    )
    cached.progress("preprocessing_complete", preprocessed_shape=list(processed.shape))
    logits = predictor.predict_logits_from_preprocessed_data(torch.from_numpy(processed))
    _save_array(work / "logits.npy", cast(NumericArray, logits.numpy()))
    ending = str(dataset["file_ending"])
    segmentation = exporter.convert_predicted_logits_to_segmentation_with_correct_shape(
        logits,
        predictor.plans_manager,
        configuration,
        predictor.label_manager,
        properties,
        return_probabilities=False,
        num_threads_torch=options.threads,
    )
    reader.write_seg(segmentation, str(work / f"label-mask{ending}"), properties)
    segmentation, probabilities = (
        exporter.convert_predicted_logits_to_segmentation_with_correct_shape(
            logits,
            predictor.plans_manager,
            configuration,
            predictor.label_manager,
            properties,
            return_probabilities=True,
            num_threads_torch=options.threads,
        )
    )
    reader.write_seg(segmentation, str(work / f"probability-mask{ending}"), properties)
    np.savez_compressed(work / "probabilities.npz", probabilities=probabilities)
    properties.update(
        labels=dataset["labels"], regions_class_order=dataset.get("regions_class_order")
    )
    with (work / "probabilities.pkl").open("wb") as handle:
        pickle.dump(properties, handle)
    cached.progress("complete")
    _write_json(
        work / "reference.json",
        {
            **identity,
            "preprocessed_shape": list(processed.shape),
            "elapsed_s": monotonic() - cached.started,
        },
    )


if __name__ == "__main__":
    main()
