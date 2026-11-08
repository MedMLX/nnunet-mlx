"""MLX-native nnU-Net raw-case prediction orchestration."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import asdict, dataclass
from math import prod
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

from nnunet_mlx.contracts import parse_integer
from nnunet_mlx.errors import ContractError
from nnunet_mlx.model.folder import ModelBundle
from nnunet_mlx.prediction.export import (
    ExportedFiles,
    RestoredPrediction,
    completion_receipt_path,
    export_prediction,
    prediction_is_complete,
    prediction_output_files,
    require_prediction_destination,
    restore_device_segmentation,
    restore_prediction,
)
from nnunet_mlx.prediction.inference import (
    ArrayPredictor,
    LogitArray,
    device_logits_to_host,
)
from nnunet_mlx.prediction.io import (
    PreparedPredictionInput,
    prediction_input_signature,
    prepare_prediction_input,
    restore_and_export_prediction,
)
from nnunet_mlx.prediction.sliding_window import compute_steps
from nnunet_mlx.runtime import (
    compile_inference_model,
    default_inference_compilation,
)

if TYPE_CHECKING:
    from nibabel.nifti1 import Nifti1Image


@dataclass(frozen=True, slots=True)
class PredictionMetrics:
    """Wall-clock timings (seconds) and workload size for one ``predict_case`` call.

    ``tile_count`` is the number of sliding-window tiles run, summed over folds (and
    over slices for 2-D models), not counting mirrored forward passes. ``padded_shape``
    is the spatial shape after minimum-size padding and ``patch_size`` the patch shape,
    both in the plans' axis order.
    """

    preprocessing_s: float
    tile_inference_s: float
    restoration_export_s: float
    total_wall_s: float
    tile_count: int
    padded_shape: tuple[int, ...]
    patch_size: tuple[int, ...]

    def to_payload(self) -> dict[str, object]:
        """Serialize the recorded prediction metrics."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RawPredictionResult:
    """Outputs of one predicted case.

    ``preprocessed_shape`` is the ``(C, *spatial)`` shape of the network input after
    preprocessing. ``metrics`` is only filled by ``predict_case``.
    """

    files: ExportedFiles
    folds: tuple[int | str, ...]
    preprocessed_shape: tuple[int, ...]
    metrics: PredictionMetrics | None = None

    def to_payload(self) -> dict[str, object]:
        """Serialize output files, fold selection and preprocessing details."""
        return {
            "files": self.files.to_payload(),
            "folds": list(self.folds),
            "preprocessed_shape": list(self.preprocessed_shape),
            "metrics": None if self.metrics is None else self.metrics.to_payload(),
        }


@dataclass(frozen=True, slots=True)
class RawInputCase:
    """One case of a raw input folder: its identifier and image files in channel order."""

    identifier: str
    image_files: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class CasePredictionResult:
    """Result for one case of a folder prediction."""

    identifier: str
    prediction: RawPredictionResult

    def to_payload(self) -> dict[str, object]:
        """Serialize one case identifier and its prediction result."""
        return {
            "identifier": self.identifier,
            "prediction": self.prediction.to_payload(),
        }


@dataclass(frozen=True, slots=True)
class FolderPredictionResult:
    """Result of ``predict_folder`` for one partition.

    ``skipped`` lists identifiers whose outputs were already complete for the same
    inputs and options (``continue_existing``).
    """

    output_folder: Path
    predictions: tuple[CasePredictionResult, ...]
    skipped: tuple[str, ...]
    part_id: int
    num_parts: int

    def to_payload(self) -> dict[str, object]:
        """Serialize partition results, output folder and skipped cases."""
        return {
            "output_folder": str(self.output_folder),
            "predictions": [result.to_payload() for result in self.predictions],
            "skipped": list(self.skipped),
            "part_id": self.part_id,
            "num_parts": self.num_parts,
        }


@dataclass(frozen=True, slots=True)
class _PredictionCasePlan:
    case: RawInputCase
    output_prefix: Path
    previous_stage_file: Path | None
    identity: dict[str, object]


class ModelFolderPredictor:
    """Predict raw cases from an MLX-native model folder.

    Args:
        bundle: Loaded model folder (see ``load_model_bundle``).
        step_fraction: Sliding-window stride as a fraction of the patch size, in (0, 1].
        compile_models: ``None`` compiles only models inside the supported compilation
            envelope (native arithmetic only); ``True``/``False`` force the choice.
            Ordered ``torch-order-fp32`` arithmetic disables automatic compilation
            and rejects ``True``.
        accumulation: ``"float32"`` or ``"nnunet-fp16"`` window accumulation; see
            ``predict_logits``.

    Raises:
        ContractError: If compilation is requested with ordered arithmetic, or a model
            is outside the supported inference envelope.
    """

    def __init__(
        self,
        bundle: ModelBundle,
        *,
        step_fraction: float = 0.5,
        compile_models: bool | None = None,
        accumulation: str = "float32",
    ) -> None:
        """Configure the model bundle, sliding-window settings and compilation policy."""
        self.bundle = bundle
        compilation_enabled = (
            bundle.arithmetic == "native"
            and all(default_inference_compilation(model) for model in bundle.models)
            if compile_models is None
            else compile_models
        )
        if bundle.arithmetic != "native" and compilation_enabled:
            raise ContractError("ordered inference arithmetic requires compile_models=False")
        self.compilation_enabled = compilation_enabled
        models = (
            tuple(compile_inference_model(model) for model in bundle.models)
            if compilation_enabled
            else bundle.models
        )
        self.predictor = ArrayPredictor(
            models,
            patch_size=bundle.plans.configuration.patch_size,
            num_outputs=bundle.semantics.output_channels,
            mirroring_axes=bundle.mirroring_axes,
            regions_class_order=bundle.semantics.regions_class_order,
            step_fraction=step_fraction,
            accumulation=accumulation,
        )

    def predict_nifti(self, image: Nifti1Image) -> Nifti1Image:
        """Return local labels on one NIfTI-1 image's XYZ voxel grid without file I/O.

        The single-channel 3-D image follows the bundle's medical-image reader
        orientation, plans transpose, nonzero crop and normalization. Unsupported
        readers, cascades and region models raise ``ContractError``;
        file/probability prediction remains available through ``predict_case``
        and ``predict_folder``.

        Model execution and ordered Gaussian accumulation stay on Metal. One
        fold without logit resampling reduces labels slab by slab; ensembles
        and resampled logits are averaged and restored on Metal, using host
        restoration only where the fused kernel does not apply. The label-only
        path corresponds to ``predict_case(..., save_probabilities=False)``;
        requesting probabilities can change a class label at float32 near ties.
        Uncrop and inverse transpose
        restore the source shape, affine and orientation. Reader-specific spacing
        units apply during preprocessing; output retains the input header units
        and coded forms. Output uses the same
        uint8/uint16 label dtype selection as the existing NIfTI writer. The
        input image and its data are not modified, and no output is saved.
        """
        from nnunet_mlx.prediction.nifti import predict_nifti

        return predict_nifti(self, image)

    def predict_case(
        self,
        image_files: Sequence[str | Path],
        output_prefix: str | Path,
        *,
        previous_stage_file: str | Path | None = None,
        save_probabilities: bool,
        overwrite: bool,
        logits_file: str | Path | None = None,
    ) -> RawPredictionResult:
        """Predict one raw case and export labels (and optionally probabilities).

        Args:
            image_files: One image per modality, in channel order, read with the
                bundle's image reader.
            output_prefix: Output path without the dataset file ending.
            previous_stage_file: Previous-stage label map; required by cascade models.
            save_probabilities: Also write ``<prefix>.npz`` (``probabilities``,
                ``(C, D, H, W)`` float32 in the reader's array grid) and
                ``<prefix>.pkl`` with geometry properties. A reorienting reader's
                probability grid need not have the source NIfTI's axis order.
            overwrite: Replace existing outputs instead of raising.
            logits_file: If given, also save the fold-averaged network logits,
                ``(C, *spatial)`` float32 in preprocessed space (before resampling
                back to the input grid), as a ``.npy`` stream.

        Raises:
            ContractError: If outputs or ``logits_file`` already exist without
                ``overwrite``, ``logits_file`` overlaps an input or output, or any
                input violates the model contract.
        """
        started = perf_counter()
        expected = prediction_output_files(
            output_prefix,
            file_ending=self.bundle.file_ending,
            image_reader_writer=self.bundle.image_reader_writer,
            save_probabilities=save_probabilities,
        )
        require_prediction_destination(expected, overwrite=overwrite)
        logits_path = None if logits_file is None else Path(logits_file)
        if logits_path is not None:
            forbidden = {Path(path).resolve() for path in image_files}
            for destination in (
                expected.segmentation,
                expected.probabilities,
                expected.properties,
                expected.image_properties,
                completion_receipt_path(output_prefix),
            ):
                if destination is not None:
                    forbidden.add(destination.resolve())
            if previous_stage_file is not None:
                forbidden.add(Path(previous_stage_file).resolve())
            if logits_path.resolve() in forbidden:
                raise ContractError(
                    "logits destination overlaps a prediction input or segmentation"
                )
            if logits_path.exists() and not overwrite:
                raise ContractError(f"logits destination already exists: {logits_path}")
        identity = self._prediction_identity(image_files, previous_stage_file, save_probabilities)
        preprocessing_started = perf_counter()
        prepared = self._prepare_input(image_files, previous_stage_file)
        preprocessing_s = perf_counter() - preprocessing_started
        inference_started = perf_counter()
        logits, restored = self._infer(
            prepared, save_probabilities=save_probabilities, host_logits=logits_path is not None
        )
        tile_inference_s = perf_counter() - inference_started
        export_started = perf_counter()
        files = self._export(
            logits,
            restored,
            prepared,
            output_prefix,
            save_probabilities=save_probabilities,
            overwrite=overwrite,
            completion_identity=identity,
        )
        restoration_export_s = perf_counter() - export_started
        if logits_path is not None:
            if logits is None:
                raise AssertionError("requested logits export lacks host logits")
            logits_path.parent.mkdir(parents=True, exist_ok=True)
            with logits_path.open("wb") as stream:
                np.save(stream, logits)
        patch = tuple(self.bundle.plans.configuration.patch_size)
        padded = tuple(
            max(size, tile)
            for size, tile in zip(
                prepared.data.shape[-len(patch) :],
                patch,
                strict=True,
            )
        )
        slices = prepared.data.shape[1] if len(patch) == 2 and prepared.data.ndim == 4 else 1
        metrics = PredictionMetrics(
            preprocessing_s,
            tile_inference_s,
            restoration_export_s,
            perf_counter() - started,
            prod(
                len(axis)
                for axis in compute_steps(
                    padded,
                    patch,
                    step_fraction=self.predictor.step_fraction,
                )
            )
            * slices
            * len(self.bundle.folds),
            padded,
            patch,
        )
        return RawPredictionResult(
            files=files,
            folds=self.bundle.folds,
            preprocessed_shape=tuple(int(value) for value in prepared.data.shape),
            metrics=metrics,
        )

    def predict_folder(
        self,
        input_folder: str | Path,
        output_folder: str | Path,
        *,
        previous_stage_folder: str | Path | None = None,
        save_probabilities: bool,
        overwrite: bool,
        continue_existing: bool = False,
        num_parts: int = 1,
        part_id: int = 0,
        preprocessing_processes: int = 0,
        export_processes: int = 0,
        max_inflight_bytes: int = 2 * 1024**3,
    ) -> FolderPredictionResult:
        """Discover and predict one deterministic partition of a raw-data folder.

        Cases are the sorted ``<case>_<NNNN><file_ending>`` groups of ``input_folder``;
        partition ``part_id`` of ``num_parts`` takes every ``num_parts``-th case.
        ``continue_existing`` skips cases whose receipt matches their inputs and
        options, and replaces stale or incomplete owned outputs; it is mutually
        exclusive with ``overwrite``. File matching uses size and modification time,
        not contents. ``preprocessing_processes`` and
        ``export_processes`` set spawned CPU workers (0 runs inline), and
        ``max_inflight_bytes`` bounds the prepared and queued payloads shared by them.
        """

        if overwrite and continue_existing:
            raise ContractError("overwrite and continue-existing are mutually exclusive")
        _validate_prediction_partition(part_id=part_id, num_parts=num_parts)
        _validate_process_count(preprocessing_processes, name="preprocessing processes")
        _validate_process_count(export_processes, name="export processes")
        if parse_integer(max_inflight_bytes, name="max-inflight-bytes") < 1:
            raise ContractError("max-inflight-bytes must be positive")
        cases = discover_input_cases(input_folder, self.bundle.file_ending)
        selected = cases[part_id::num_parts]
        destination = _prepare_prediction_folder(output_folder)
        previous = _previous_stage_folder(previous_stage_folder)
        plans, skipped = self._prediction_case_plans(
            selected,
            destination,
            previous,
            save_probabilities=save_probabilities,
            continue_existing=continue_existing,
            overwrite=overwrite or continue_existing,
        )
        predictions = self._predict_planned_cases(
            plans,
            save_probabilities=save_probabilities,
            overwrite=overwrite or continue_existing,
            preprocessing_processes=preprocessing_processes,
            export_processes=export_processes,
            max_inflight_bytes=max_inflight_bytes,
        )
        return FolderPredictionResult(
            output_folder=destination,
            predictions=tuple(predictions),
            skipped=tuple(skipped),
            part_id=part_id,
            num_parts=num_parts,
        )

    def _prediction_case_plans(
        self,
        cases: Sequence[RawInputCase],
        destination: Path,
        previous_stage_folder: Path | None,
        *,
        save_probabilities: bool,
        continue_existing: bool,
        overwrite: bool,
    ) -> tuple[tuple[_PredictionCasePlan, ...], tuple[str, ...]]:
        """Plan outputs for each case, returning ``(plans, skipped_identifiers)``.

        With ``continue_existing`` a case whose receipt matches the current prediction
        identity is skipped; otherwise existing outputs raise unless ``overwrite``.
        """
        plans: list[_PredictionCasePlan] = []
        skipped: list[str] = []
        for case in cases:
            prefix = destination / case.identifier
            expected = prediction_output_files(
                prefix,
                file_ending=self.bundle.file_ending,
                image_reader_writer=self.bundle.image_reader_writer,
                save_probabilities=save_probabilities,
            )
            previous_file = (
                None
                if previous_stage_folder is None
                else previous_stage_folder / f"{case.identifier}{self.bundle.file_ending}"
            )
            identity = self._prediction_identity(
                case.image_files, previous_file, save_probabilities
            )
            if continue_existing and prediction_is_complete(prefix, expected, identity):
                skipped.append(case.identifier)
                continue
            require_prediction_destination(expected, overwrite=overwrite)
            plans.append(_PredictionCasePlan(case, prefix, previous_file, identity))
        return tuple(plans), tuple(skipped)

    def _predict_planned_cases(
        self,
        plans: Sequence[_PredictionCasePlan],
        *,
        save_probabilities: bool,
        overwrite: bool,
        preprocessing_processes: int,
        export_processes: int,
        max_inflight_bytes: int,
    ) -> list[CasePredictionResult]:
        """Run the pipeline with fresh preparation/export pools, always shutting them down."""
        preprocess_executor = _process_pool(preprocessing_processes)
        export_executor = _process_pool(export_processes)
        try:
            return self._run_prediction_pipeline(
                plans,
                preprocess_executor=preprocess_executor,
                export_executor=export_executor,
                save_probabilities=save_probabilities,
                overwrite=overwrite,
                preprocessing_processes=preprocessing_processes,
                export_processes=export_processes,
                max_inflight_bytes=max_inflight_bytes,
            )
        finally:
            _shutdown_pool(preprocess_executor)
            _shutdown_pool(export_executor)

    def _run_prediction_pipeline(
        self,
        plans: Sequence[_PredictionCasePlan],
        *,
        preprocess_executor: ProcessPoolExecutor | None,
        export_executor: ProcessPoolExecutor | None,
        save_probabilities: bool,
        overwrite: bool,
        preprocessing_processes: int,
        export_processes: int,
        max_inflight_bytes: int,
    ) -> list[CasePredictionResult]:
        """Overlap preprocessing, inference and export across cases.

        Inference runs sequentially in this process (one accelerator). Preprocessing
        runs ahead in worker processes and exports lag behind, bounded to
        ``export_processes`` pending futures. ``max_inflight_bytes`` is split evenly
        across preparation and export slots to give a per-case byte allowance.
        """
        # Reserve payload space for queued preparation, the active input, and exports.
        preparation_slots = preprocessing_processes + 1
        export_slots = export_processes + 1 if export_executor is not None else 0
        allowance = max_inflight_bytes // (preparation_slots + export_slots)
        if allowance < 1:
            raise ContractError("max-inflight-bytes is too small for the requested workers")
        predictions: list[CasePredictionResult] = []
        pending_exports: deque[
            tuple[_PredictionCasePlan, tuple[int, ...], Future[ExportedFiles]]
        ] = deque()
        for plan, prepared in self._prepared_cases(
            plans, preprocess_executor, preprocessing_processes, allowance
        ):
            logits, restored = self._infer(
                prepared, save_probabilities=save_probabilities, host_logits=False
            )
            shape = tuple(int(value) for value in prepared.data.shape)
            if export_executor is None:
                files = self._export(
                    logits,
                    restored,
                    prepared,
                    plan.output_prefix,
                    save_probabilities=save_probabilities,
                    overwrite=overwrite,
                    completion_identity=plan.identity,
                )
                predictions.append(_case_result(plan, shape, files, self.bundle.folds))
                del prepared
                continue
            pending_exports.append(
                (
                    plan,
                    shape,
                    self._submit_export(
                        export_executor,
                        logits,
                        restored,
                        prepared,
                        plan.output_prefix,
                        save_probabilities=save_probabilities,
                        overwrite=overwrite,
                        completion_identity=plan.identity,
                        max_output_bytes=allowance,
                    ),
                )
            )
            del prepared
            if len(pending_exports) > export_processes:
                predictions.append(_completed_export(pending_exports.popleft(), self.bundle.folds))
        while pending_exports:
            predictions.append(_completed_export(pending_exports.popleft(), self.bundle.folds))
        return predictions

    def _prepared_cases(
        self,
        plans: Sequence[_PredictionCasePlan],
        executor: ProcessPoolExecutor | None,
        processes: int,
        allowance: int,
    ) -> Iterator[tuple[_PredictionCasePlan, PreparedPredictionInput]]:
        """Yield ``(plan, prepared)`` in order, keeping ``processes`` preparations queued ahead."""
        if executor is None:
            for plan in plans:
                yield (
                    plan,
                    self._prepare_input(
                        plan.case.image_files,
                        plan.previous_stage_file,
                        max_output_bytes=allowance,
                    ),
                )
            return
        pending: deque[tuple[_PredictionCasePlan, Future[PreparedPredictionInput]]] = deque()
        iterator = iter(plans)
        for _ in range(processes):
            plan = next(iterator, None)
            if plan is None:
                break
            pending.append((plan, self._submit_preparation(executor, plan, allowance)))
        while pending:
            plan, future = pending.popleft()
            prepared = future.result()
            next_plan = next(iterator, None)
            if next_plan is not None:
                pending.append(
                    (next_plan, self._submit_preparation(executor, next_plan, allowance))
                )
            yield plan, prepared
            del prepared, future

    def _prepare_input(
        self,
        image_files: Sequence[str | Path],
        previous_stage_file: str | Path | None,
        *,
        max_output_bytes: int | None = None,
    ) -> PreparedPredictionInput:
        """Load and preprocess the requested raw input and previous-stage labels."""
        return prepare_prediction_input(
            image_files,
            image_reader_writer=self.bundle.image_reader_writer,
            image_channels=self.bundle.image_channels,
            plans=self.bundle.plans,
            semantics=self.bundle.semantics,
            previous_stage_file=previous_stage_file,
            max_output_bytes=max_output_bytes,
        )

    def _submit_preparation(
        self,
        executor: ProcessPoolExecutor,
        plan: _PredictionCasePlan,
        max_output_bytes: int,
    ) -> Future[PreparedPredictionInput]:
        """Submit one case preprocessing task to the process pool."""
        return executor.submit(
            prepare_prediction_input,
            plan.case.image_files,
            image_reader_writer=self.bundle.image_reader_writer,
            image_channels=self.bundle.image_channels,
            plans=self.bundle.plans,
            semantics=self.bundle.semantics,
            previous_stage_file=plan.previous_stage_file,
            max_output_bytes=max_output_bytes,
        )

    def _infer(
        self,
        prepared: PreparedPredictionInput,
        *,
        save_probabilities: bool,
        host_logits: bool,
    ) -> tuple[LogitArray | None, RestoredPrediction | None]:
        """Return host logits, or labels restored without leaving Metal.

        Label-only 3-D class predictions keep fold logits on Metal through
        resampling and argmax, transferring only uint16 labels. Cases the fused
        restoration rejects receive the same logits on the host instead.
        """
        if (
            host_logits
            or save_probabilities
            or self.bundle.semantics.has_regions
            or len(self.predictor.patch_size) != 3
        ):
            return self.predictor.predict_logits_from_preprocessed_data(prepared.data), None
        device_logits = self.predictor.predict_device_logits_from_preprocessed_data(prepared.data)
        restored = restore_device_segmentation(
            device_logits, prepared.properties, self.bundle.plans, self.bundle.semantics
        )
        if restored is not None:
            return None, restored
        host = device_logits_to_host(device_logits)
        del device_logits
        return host, None

    def _export(
        self,
        logits: LogitArray | None,
        restored: RestoredPrediction | None,
        prepared: PreparedPredictionInput,
        output_prefix: str | Path,
        *,
        save_probabilities: bool,
        overwrite: bool,
        completion_identity: Mapping[str, object],
    ) -> ExportedFiles:
        """Restore and export a prediction with its completion identity."""
        if restored is not None:
            return export_prediction(
                restored,
                prepared.properties,
                output_prefix,
                file_ending=self.bundle.file_ending,
                image_reader_writer=self.bundle.image_reader_writer,
                overwrite=overwrite,
                completion_identity=completion_identity,
            )
        if logits is None:
            raise AssertionError("prediction produced neither logits nor restored labels")
        return restore_and_export_prediction(
            logits,
            prepared.properties,
            output_prefix,
            plans=self.bundle.plans,
            semantics=self.bundle.semantics,
            file_ending=self.bundle.file_ending,
            image_reader_writer=self.bundle.image_reader_writer,
            save_probabilities=save_probabilities,
            overwrite=overwrite,
            completion_identity=completion_identity,
        )

    def _submit_export(
        self,
        executor: ProcessPoolExecutor,
        logits: LogitArray | None,
        restored: RestoredPrediction | None,
        prepared: PreparedPredictionInput,
        output_prefix: Path,
        *,
        save_probabilities: bool,
        overwrite: bool,
        completion_identity: Mapping[str, object],
        max_output_bytes: int,
    ) -> Future[ExportedFiles]:
        """Restore labels in this process, then queue only the host export in a worker.

        The restored case must fit ``max_output_bytes`` so queued results stay bounded.
        """
        if restored is None:
            if logits is None:
                raise AssertionError("prediction produced neither logits nor restored labels")
            restored = restore_prediction(
                logits,
                prepared.properties,
                self.bundle.plans,
                self.bundle.semantics,
                return_probabilities=save_probabilities,
            )
        output_bytes = restored.segmentation.nbytes
        if restored.probabilities is not None:
            output_bytes += restored.probabilities.nbytes
        if output_bytes > max_output_bytes:
            raise ContractError(
                "restored case exceeds worker byte allowance; increase max-inflight-bytes"
            )
        return executor.submit(
            export_prediction,
            restored,
            prepared.properties,
            output_prefix,
            file_ending=self.bundle.file_ending,
            image_reader_writer=self.bundle.image_reader_writer,
            overwrite=overwrite,
            completion_identity=completion_identity,
        )

    def _prediction_identity(
        self,
        image_files: Sequence[str | Path],
        previous_stage_file: str | Path | None,
        save_probabilities: bool,
    ) -> dict[str, object]:
        """JSON-native description of everything that determines a case's output.

        Stored as the completion receipt so ``continue_existing`` reruns only when
        inputs, model, settings or arithmetic changed.
        """
        # Receipts round-trip through JSON, so every value must be JSON-native.
        regions_class_order = self.bundle.semantics.regions_class_order
        return {
            "inputs": [str(path) for path in image_files],
            "input_signatures": prediction_input_signature(
                image_files,
                previous_stage_file,
                image_reader_writer=self.bundle.image_reader_writer,
            ),
            "previous_stage": (None if previous_stage_file is None else str(previous_stage_file)),
            "configuration": self.bundle.plans.configuration.name,
            "resolved_configuration": self.bundle.plans.configuration.raw,
            # Global plan fields preprocessing and restoration actually consume.
            "preprocessing": {
                "transpose_forward": list(self.bundle.plans.transpose_forward),
                "transpose_backward": list(self.bundle.plans.transpose_backward),
                "image_reader_writer": self.bundle.image_reader_writer,
                "intensity_properties": self.bundle.plans.intensity_properties,
            },
            "file_ending": self.bundle.file_ending,
            "labels": self.bundle.dataset_json["labels"],
            "regions_class_order": (
                None if regions_class_order is None else list(regions_class_order)
            ),
            "checkpoints": [provenance.to_payload() for provenance in self.bundle.checkpoints],
            "folds": list(self.bundle.folds),
            "arithmetic": self.bundle.arithmetic,
            "compiled": self.compilation_enabled,
            "save_probabilities": save_probabilities,
            "step_fraction": self.predictor.step_fraction,
            "mirroring_axes": list(self.predictor.mirroring_axes),
            "accumulation": self.predictor.accumulation,
        }


def _process_pool(processes: int) -> ProcessPoolExecutor | None:
    """Create a spawn-based worker pool, or None when worker count is zero."""
    return (
        None
        if processes == 0
        else ProcessPoolExecutor(
            max_workers=processes,
            mp_context=get_context("spawn"),
        )
    )


def _shutdown_pool(executor: ProcessPoolExecutor | None) -> None:
    """Wait for the process pool and cancel queued work when one exists."""
    if executor is not None:
        executor.shutdown(wait=True, cancel_futures=True)


def _case_result(
    plan: _PredictionCasePlan,
    preprocessed_shape: tuple[int, ...],
    files: ExportedFiles,
    folds: tuple[int | str, ...],
) -> CasePredictionResult:
    """Build a case result from its prediction plan, shape, files and folds."""
    return CasePredictionResult(
        plan.case.identifier,
        RawPredictionResult(
            files=files,
            folds=folds,
            preprocessed_shape=preprocessed_shape,
        ),
    )


def _completed_export(
    pending: tuple[_PredictionCasePlan, tuple[int, ...], Future[ExportedFiles]],
    folds: tuple[int | str, ...],
) -> CasePredictionResult:
    """Collect one completed export future into a case result."""
    plan, shape, future = pending
    return _case_result(plan, shape, future.result(), folds)


def discover_input_cases(input_folder: str | Path, file_ending: str) -> tuple[RawInputCase, ...]:
    """Group split-dataset files by case and numeric modality suffix.

    Files are named ``<case>_<NNNN><file_ending>`` with a four-digit modality index;
    each case's indices must be contiguous from ``0000``.

    Raises:
        ContractError: If the folder is missing or empty, a name lacks the suffix, a
            channel is duplicated, or a case has a gap in its channels.
    """

    folder = Path(input_folder)
    if not folder.is_dir():
        raise ContractError(f"prediction input folder does not exist: {folder}")
    grouped: dict[str, dict[int, Path]] = {}
    for path in sorted(folder.iterdir()):
        if not path.is_file() or not path.name.endswith(file_ending):
            continue
        stem = path.name[: -len(file_ending)]
        if len(stem) < 6 or stem[-5] != "_" or not stem[-4:].isdigit():
            raise ContractError(f"prediction input {path.name!r} lacks a four-digit channel suffix")
        identifier = stem[:-5]
        channel = int(stem[-4:])
        if not identifier or channel in grouped.setdefault(identifier, {}):
            raise ContractError(f"prediction input channel is ambiguous: {path.name}")
        grouped[identifier][channel] = path
    if not grouped:
        raise ContractError(f"prediction input folder contains no {file_ending!r} channel files")
    return tuple(
        _raw_input_case(identifier, channels) for identifier, channels in sorted(grouped.items())
    )


def _raw_input_case(identifier: str, channels: Mapping[int, Path]) -> RawInputCase:
    """Require contiguous channel indices and build one ordered input case."""
    indices = sorted(channels)
    if indices != list(range(len(indices))):
        raise ContractError(f"prediction case {identifier!r} channels must be contiguous from 0000")
    return RawInputCase(identifier, tuple(channels[index] for index in indices))


def _validate_prediction_partition(*, part_id: object, num_parts: object) -> None:
    """Require integer partition indices satisfying 0 <= part_id < num_parts."""
    if (
        isinstance(part_id, bool)
        or isinstance(num_parts, bool)
        or not isinstance(part_id, int)
        or not isinstance(num_parts, int)
        or num_parts < 1
        or part_id not in range(num_parts)
    ):
        raise ContractError("prediction partition requires 0 <= part-id < num-parts")


def _validate_process_count(value: object, *, name: str) -> None:
    """Require a nonnegative integer worker count."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{name} must be a non-negative integer")


def _prepare_prediction_folder(output_folder: str | Path) -> Path:
    """Create or validate the prediction output folder."""
    folder = Path(output_folder)
    if folder.exists() and not folder.is_dir():
        raise ContractError(f"prediction output folder is not a directory: {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _previous_stage_folder(folder: str | Path | None) -> Path | None:
    """Validate an optional previous-stage prediction folder."""
    if folder is None:
        return None
    result = Path(folder)
    if not result.is_dir():
        raise ContractError(f"previous-stage prediction folder does not exist: {result}")
    return result
