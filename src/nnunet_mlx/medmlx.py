"""One-shot and reusable prediction for native MLX model folders.

The released models in ``nnunet_mlx.named_models`` run through the same path.
Prediction uses :class:`ModelFolderPredictor` with nnU-Net v2.8.1's
sliding-window accumulation and checkpoint mirroring. Label metadata uses the
package's shared dataset contract. Importing this module does not initialize MLX.

The operation reports failures as ``medmlx_core`` errors, whose type names reach
MCP clients. Each step re-raises the package's :class:`ContractError` as the
error for its kind of failure, with the same message and the ``ContractError``
as ``__cause__``.
"""

from __future__ import annotations

import os
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Literal, cast

from medmlx_core.files import PathLike
from medmlx_core.outputs import Artifact, artifact
from medmlx_core.weights import weights_revision

from nnunet_mlx.contracts import parse_boolean, parse_number, parse_sequence
from nnunet_mlx.errors import ContractError
from nnunet_mlx.medmlx_types import RunResult

if TYPE_CHECKING:
    from nnunet_mlx.prediction.pipeline import ModelFolderPredictor

Failure = Literal["input", "asset", "dependency", "execution"]

# MedMLX runs and the prediction CLI accumulate windows in float16, as nnU-Net v2.8.1 does.
ACCUMULATION = "nnunet-fp16"

SEGMENTATION_STEM = "mask"
_PROBABILITY_STEM = "probabilities"
PROBABILITIES = f"{_PROBABILITY_STEM}.npz"
PROBABILITY_PROPERTIES = f"{_PROBABILITY_STEM}.pkl"
NIFTI_ENDINGS = (".nii.gz", ".nii")
SOURCE_FLOAT32 = "source-fp32"
_WEIGHTS_HINT = (
    "download a compatible native safetensors model folder from MedMLX "
    "and pass it as weights_path"
)


def dataset_labels(dataset_json: Mapping[str, object]) -> list[dict[str, object]]:
    """Return the checkpoint's label table, including region-based label order."""

    from nnunet_mlx.data.labels import parse_dataset_label_semantics

    semantics = parse_dataset_label_semantics(dataset_json)
    return [label.to_payload() for label in semantics.label_table]


def output_kind(labels: Sequence[Mapping[str, object]]) -> str:
    """The label vocabulary defines the output kind even when a case is empty."""

    values = {cast(int, item["value"]) for item in labels}
    return "binary_mask" if labels and values == {0, 1} else "label_map"


@contextmanager
def _reported_as(failure: Failure) -> Generator[None]:
    """Re-raise the package's ContractError as the medmlx_core error for ``failure``.

    Error-class imports are deferred until a failure is reported, keeping request
    validation free of compute imports.
    """

    try:
        yield
    except ContractError as error:
        from medmlx_core.errors import (
            AssetNotReadyError,
            InvalidInputError,
            MissingDependencyError,
            ModelExecutionError,
        )

        kinds: dict[Failure, type[Exception]] = {
            "input": InvalidInputError,
            "asset": AssetNotReadyError,
            "dependency": MissingDependencyError,
            "execution": ModelExecutionError,
        }
        raise kinds[failure](str(error)) from error


def readiness() -> dict[str, object]:
    """Report whether the installed MLX is the exact release every nnU-Net run requires.

    MedMLX checks Metal and run arguments itself and accepts any MLX from 0.32.4 below
    0.33; nnU-Net runs need exactly the pinned release. This reads package metadata
    only and imports no MLX.
    """
    from nnunet_mlx.runtime import MLX_VERSION

    try:
        installed = version("mlx")
    except PackageNotFoundError:
        installed = None
    if installed != MLX_VERSION:
        return {
            "ready": False,
            "reason": f"nnunet-mlx runs require MLX {MLX_VERSION}, found {installed or 'none'}",
            "setup_hint": f"Install mlx=={MLX_VERSION} in this environment, for example by "
            "reinstalling medmlx[segment].",
        }
    return {"ready": True, "reason": None, "setup_hint": None}


def check_request(
    image_paths: Sequence[PathLike],
    *,
    output_dir: PathLike,
    overwrite: object = False,
    folds: Sequence[int | str] | None = None,
    source_profile: str | None = None,
    precision: str = "fast-fp32",
    step_fraction: object = 0.5,
    use_mirroring: object = True,
    save_probabilities: object = False,
) -> tuple[list[Path], tuple[int | str, ...] | None]:
    """Check the arguments that need no model folder, before weights are resolved.

    Named models call this before downloading their model folder, so a typo fails
    at once. Returns the image paths and the validated folds.

    Raises:
        InvalidInputError: Missing or non-NIfTI images; non-boolean flags; invalid
            folds, ``step_fraction``, ``precision`` or ``source_profile``; or an
            ``output_dir`` that is not a directory.
    """
    from nnunet_mlx.model.metadata import validate_folds
    from nnunet_mlx.prediction.source import resolve_inference_source_profile

    with _reported_as("input"):
        paths = cast(object, image_paths)
        if not isinstance(paths, list | tuple) or not paths or not all(
            isinstance(path, str | os.PathLike) for path in cast(Sequence[object], paths)
        ):
            raise ContractError(
                "image_paths must be a non-empty list of NIfTI images, one per modality "
                "in channel order"
            )
        images = [Path(path).expanduser() for path in image_paths]
        for image in images:
            if not image.is_file():
                raise ContractError(f"Missing input image: {image}")
            if not image.name.endswith(NIFTI_ENDINGS):
                raise ContractError(
                    f"nnU-Net input must be a NIfTI image (.nii or .nii.gz): {image}"
                )
        for name, flag in (
            ("overwrite", overwrite),
            ("use_mirroring", use_mirroring),
            ("save_probabilities", save_probabilities),
        ):
            parse_boolean(flag, name=name)
        selected_folds = (
            None
            if folds is None
            else validate_folds(cast(Sequence[int | str], parse_sequence(folds, name="folds")))
        )
        step = parse_number(step_fraction, name="step_fraction")
        if not 0 < step <= 1:
            raise ContractError(f"step_fraction must be in (0, 1], got {step_fraction}")
        if precision not in {"fast-fp32", SOURCE_FLOAT32}:
            raise ContractError(f"unsupported precision {precision!r}")
        if source_profile is not None:
            resolve_inference_source_profile(source_profile)
        output = Path(output_dir).expanduser()
        if output.exists() and not output.is_dir():
            raise ContractError(f"output_dir must be a directory: {output}")
    return images, selected_folds


def run(
    *,
    weights_path: PathLike,
    image_paths: Sequence[PathLike],
    output_dir: PathLike,
    overwrite: bool = False,
    configuration: str | None = None,
    folds: Sequence[int | str] | None = None,
    checkpoint_name: str = "checkpoint_final",
    source_profile: str | None = None,
    precision: str = "fast-fp32",
    step_fraction: float = 0.5,
    use_mirroring: bool = True,
    save_probabilities: bool = False,
) -> RunResult:
    """Segment one case and restore the label map to the input image's grid.

    Args:
        weights_path: Local native MLX model folder. The released models in
            ``nnunet_mlx.named_models`` download theirs automatically. A folder
            such as the published ``MedMLX/nnunet-liver-mlx`` bundle contains
            ``plans.json``, ``dataset.json`` and
            ``fold_<n>/checkpoint_final/{state.json,weights.safetensors}`` for the
            selected folds. Pass the folder root as ``weights_path``.
            Conversion receipts are optional export metadata, not inference inputs.
        image_paths: NIfTI images (``.nii`` or ``.nii.gz``), one per channel in
            ``dataset.json`` channel-index order; released channel names are not
            unique (AutoPET's are CT and CT), so order identifies the channel.
            Filenames are unrestricted apart from the NIfTI ending.
        output_dir: Directory for the outputs; created if missing.
        overwrite: Replace existing outputs; input/output aliases are always refused.
        configuration: Plans configuration; defaults to ``3d_fullres``.
        folds: Folds to ensemble; ``None`` discovers every numeric fold with the
            checkpoint. An explicit ``["all"]`` selects ``fold_all``.
        checkpoint_name: MLX checkpoint directory inside each fold; defaults to
            ``checkpoint_final``.
        source_profile: Compatibility profile; defaults to the nnU-Net v2.8.1 profile.
        precision: ``fast-fp32`` (default), or ``source-fp32`` to run MLX kernels
            on Metal in the pinned PyTorch CPU float32 operation order (only for
            the model profiles that admit it).
        step_fraction: Sliding-window step as a fraction of the patch, in (0, 1].
        use_mirroring: Apply the checkpoint's mirroring test-time augmentation.
        save_probabilities: Also write class or region probabilities and their properties.

    Returns:
        ``outputs`` maps ``mask`` to ``mask.nii.gz`` (or ``mask.nii``,
        following ``dataset.json``), a label map with the input's shape and
        geometry, and, with ``save_probabilities``, ``probabilities`` and
        ``probability_properties`` to ``probabilities.npz`` and
        ``probabilities.pkl``. ``artifacts`` describes the mask as a
        ``label_map`` (``binary_mask`` for a 0/1 label table) with the
        checkpoint's labels on the input grid, and the probabilities as a
        ``scalar_field`` on nnU-Net's preprocessed grid. ``details`` holds the
        label table, selected options, checkpoint signatures and timings.

    Raises:
        InvalidInputError: Missing or non-NIfTI images; non-boolean flags; invalid
            folds, ``step_fraction``, ``precision`` or ``source_profile``; an
            ``output_dir`` that is a file; existing outputs without ``overwrite``.
        AssetNotReadyError: A missing model folder, or one whose files,
            configuration, folds or checkpoints are missing, malformed,
            unsupported, or inconsistent with the selected configuration and profile.
        MissingDependencyError: MLX is missing, is not version 0.32.5, or has
            no usable accelerator.
        ModelExecutionError: Reading, predicting or publishing the case failed,
            including images that do not give the model's channel count.
    """

    return segment_case(
        weights_path=weights_path, image_paths=image_paths, output_dir=output_dir,
        overwrite=overwrite, configuration=configuration, folds=folds,
        checkpoint_name=checkpoint_name, source_profile=source_profile, precision=precision,
        step_fraction=step_fraction, use_mirroring=use_mirroring,
        save_probabilities=save_probabilities,
    )


def segment_case(
    *,
    prepared: ModelFolderPredictor | None = None,
    grid_of: str = "image_paths",
    weights_path: PathLike,
    image_paths: Sequence[PathLike],
    output_dir: PathLike,
    overwrite: bool = False,
    configuration: str | None = None,
    folds: Sequence[int | str] | None = None,
    checkpoint_name: str = "checkpoint_final",
    source_profile: str | None = None,
    precision: str = "fast-fp32",
    step_fraction: float = 0.5,
    use_mirroring: bool = True,
    save_probabilities: bool = False,
) -> RunResult:
    """Segment one case through the shared publication path, optionally with loaded state.

    ``grid_of`` names the caller's input argument whose grid the mask is on.
    """
    from nnunet_mlx.plans import read_json_object
    from nnunet_mlx.prediction.source import DEFAULT_INFERENCE_SOURCE_PROFILE

    images, selected_folds = check_request(
        image_paths,
        output_dir=output_dir,
        overwrite=overwrite,
        folds=folds,
        source_profile=source_profile,
        precision=precision,
        step_fraction=step_fraction,
        use_mirroring=use_mirroring,
        save_probabilities=save_probabilities,
    )
    folder = Path(weights_path).expanduser()
    with _reported_as("asset"):
        if prepared is None:
            if not folder.is_dir():
                raise ContractError(
                    f"nnU-Net model folder does not exist: {folder}; {_WEIGHTS_HINT}"
                )
            if not (folder / "dataset.json").is_file():
                raise ContractError(f"model folder lacks dataset.json: {folder}; {_WEIGHTS_HINT}")
            dataset_json = read_json_object(folder / "dataset.json")
        else:
            dataset_json = prepared.bundle.dataset_json
        labels = dataset_labels(dataset_json)
        file_ending = dataset_json.get("file_ending")
        if file_ending not in NIFTI_ENDINGS:
            raise ContractError(
                f"MedMLX nnU-Net runs write NIfTI; dataset.json file_ending is {file_ending!r}"
            )
    if configuration is None:
        configuration = "3d_fullres"
    if source_profile is None:
        source_profile = DEFAULT_INFERENCE_SOURCE_PROFILE
    output = Path(output_dir).expanduser()
    from nnunet_mlx.prediction.export import (
        ExportedFiles,
        completion_receipt_path,
        require_prediction_destination,
    )

    targets = {"mask": output / f"{SEGMENTATION_STEM}{file_ending}"}
    if save_probabilities:
        targets["probabilities"] = output / PROBABILITIES
        targets["probability_properties"] = output / PROBABILITY_PROPERTIES
    generation = ExportedFiles(targets["mask"], probabilities=output / PROBABILITIES,
                              properties=output / PROBABILITY_PROPERTIES)
    prefix = output / SEGMENTATION_STEM
    with _reported_as("input"):
        require_prediction_destination(generation, overwrite=overwrite, input_files=images,
                                       extra_paths=(completion_receipt_path(prefix),))

    from nnunet_mlx.runtime import import_mlx, runtime_info

    with _reported_as("dependency"):
        backend = runtime_info().backend
        mx, _ = import_mlx()
    started = perf_counter()
    if prepared is None:
        from nnunet_mlx.model.folder import load_model_bundle
        from nnunet_mlx.prediction.pipeline import ModelFolderPredictor

        mx.reset_peak_memory()
        with _reported_as("asset"):
            bundle = load_model_bundle(
                folder,
                configuration=configuration,
                folds=selected_folds,
                checkpoint_name=checkpoint_name,
                source_profile=source_profile,
                precision=precision,
            )
        if not use_mirroring:
            bundle = replace(bundle, mirroring_axes=())
        model_load_s = perf_counter() - started
        with _reported_as("execution"):
            case_predictor = ModelFolderPredictor(
                bundle, step_fraction=step_fraction, accumulation=ACCUMULATION
            )
    else:
        bundle, case_predictor = prepared.bundle, prepared
        model_load_s = 0.0
    with _reported_as("execution"):
        result = case_predictor.predict_case(
            images,
            prefix,
            save_probabilities=save_probabilities,
            overwrite=overwrite,
            probability_stem=_PROBABILITY_STEM,
        )
    metrics: dict[str, object] = {}
    if result.metrics is not None:
        metrics = {
            key: list(cast(tuple[int, ...], value)) if isinstance(value, tuple) else value
            for key, value in result.metrics.to_payload().items()
        }
    metrics.update(
        model_load_s=model_load_s,
        total_wall_s=perf_counter() - started,
        peak_memory_bytes=int(mx.get_peak_memory()),
    )
    artifacts: dict[str, Artifact] = {
        "mask": artifact(
            kind=output_kind(labels), coordinate_space="image_physical", grid_of=grid_of,
            labels=[
                {"value": cast(int, item["value"]), "name": cast(str, item["name"])}
                for item in labels
            ],
        ),
    }
    if save_probabilities:
        # nnU-Net's preprocessed (cropped, resampled) grid, which is not written.
        artifacts["probabilities"] = artifact(kind="scalar_field", coordinate_space="unresolved")
    payload: RunResult = {
        "outputs": {role: path.resolve().as_posix() for role, path in targets.items()},
        "artifacts": artifacts,
        "details": {
            "weights_revision": weights_revision(folder),
            "labels": labels,
            "configuration": configuration,
            "source_profile": source_profile,
            "precision": precision,
            "folds": list(bundle.folds),
            "checkpoints": [item.to_payload() for item in bundle.checkpoints],
            "mirroring_axes": list(bundle.mirroring_axes),
            "step_fraction": step_fraction,
            "accumulation": result.accumulation,
            "decoding": result.decoding,
            "save_probabilities": save_probabilities,
            "backend": backend,
            "preprocessed_shape": list(result.preprocessed_shape),
            "metrics": metrics,
        },
    }
    return payload



def _source_signature(
    folder: Path, checkpoint_name: str,
) -> tuple[tuple[str, int, int, int, int, int], ...]:
    """Detect ordinary model/configuration edits without reading or hashing weights."""
    paths = [folder / "plans.json", folder / "dataset.json"]
    for checkpoint in sorted(folder.glob(f"fold_*/{checkpoint_name}")):
        paths.extend((checkpoint / "state.json", checkpoint / "weights.safetensors"))
    result: list[tuple[str, int, int, int, int, int]] = []
    for path in paths:
        try:
            info = path.stat()
        except OSError as error:
            raise ContractError(f"Model source is unavailable: {path}") from error
        result.append((str(path.relative_to(folder)), info.st_dev, info.st_ino,
                       info.st_size, info.st_mtime_ns, info.st_ctime_ns))
    return tuple(result)


@dataclass(frozen=True)
class _Preparation:
    """Immutable admitted model and inference choices for a prepared lifetime."""

    folder: Path
    configuration: str
    folds: tuple[int | str, ...] | None
    checkpoint_name: str
    source_profile: str
    precision: str
    step_fraction: float
    use_mirroring: bool


class NnUNetModel:
    """One loaded model and configuration reused for serial cases until closed.

    :meth:`run` has the :func:`run` contract for the per-case arguments.
    Model files must remain unchanged while loaded. Metadata signatures detect
    ordinary replacements/edits before each case; load again after changing
    weights or configuration. Inputs, predictions and output staging belong to
    each call, and no per-case tensor is stored on this object. Per-case
    ``model_load_s`` is zero; ``peak_memory_bytes`` covers loading through that
    case rather than resetting the MLX peak between cases.
    """

    def __init__(
        self, *, weights_path: PathLike, configuration: str = "3d_fullres",
        folds: Sequence[int | str] | None = None, checkpoint_name: str = "checkpoint_final",
        source_profile: str | None = None, precision: str = "fast-fp32",
        step_fraction: float = 0.5, use_mirroring: bool = True,
    ) -> None:
        """Load and bind the selected model once; cases are supplied to :meth:`run`."""
        from nnunet_mlx.model.folder import load_model_bundle
        from nnunet_mlx.prediction.pipeline import ModelFolderPredictor
        from nnunet_mlx.prediction.source import DEFAULT_INFERENCE_SOURCE_PROFILE
        from nnunet_mlx.runtime import import_mlx, runtime_info

        self._options = _Preparation(
            Path(weights_path).expanduser().absolute(), configuration,
            None if folds is None else tuple(folds), checkpoint_name,
            DEFAULT_INFERENCE_SOURCE_PROFILE if source_profile is None else source_profile,
            precision, step_fraction, use_mirroring,
        )
        options = self._options
        with _reported_as("input"):
            if Path(checkpoint_name).name != checkpoint_name or checkpoint_name in {".", ".."}:
                raise ContractError("checkpoint_name must name a checkpoint directory")
            parse_boolean(use_mirroring, name="use_mirroring")
            step = parse_number(step_fraction, name="step_fraction")
            if not 0 < step <= 1:
                raise ContractError("step_fraction must be in (0, 1]")
        with _reported_as("dependency"):
            runtime_info()
            mx, _ = import_mlx()
        mx.reset_peak_memory()
        with _reported_as("asset"):
            self._signature = _source_signature(options.folder, checkpoint_name)
            bundle = load_model_bundle(
                options.folder, configuration=configuration, folds=options.folds,
                checkpoint_name=checkpoint_name, source_profile=options.source_profile,
                precision=precision,
            )
            if _source_signature(options.folder, checkpoint_name) != self._signature:
                raise ContractError("Model files changed while loading; load the model again")
        if not use_mirroring:
            bundle = replace(bundle, mirroring_axes=())
        self._state: ModelFolderPredictor | None = ModelFolderPredictor(
            bundle, step_fraction=step_fraction, accumulation=ACCUMULATION
        )

    def run(
        self, *, image_paths: Sequence[PathLike], output_dir: PathLike,
        overwrite: bool = False, save_probabilities: bool = False,
    ) -> RunResult:
        """Segment one case with the one-shot geometry and publication checks."""
        return self.segment_case(
            image_paths=image_paths, output_dir=output_dir, overwrite=overwrite,
            save_probabilities=save_probabilities,
        )

    def segment_case(
        self, *, image_paths: Sequence[PathLike], output_dir: PathLike, overwrite: bool,
        save_probabilities: bool = False, grid_of: str = "image_paths",
    ) -> RunResult:
        """Segment one case; ``grid_of`` names the caller's input argument for the mask grid.

        Raises:
            RuntimeError: The model is closed.
        """
        options = self._options
        state = self._state
        if state is None:
            raise RuntimeError("nnU-Net model is closed")
        with _reported_as("asset"):
            if _source_signature(options.folder, options.checkpoint_name) != self._signature:
                raise ContractError("Model files changed after loading; load the model again")
        return segment_case(
            prepared=state, grid_of=grid_of, weights_path=options.folder,
            image_paths=image_paths,
            output_dir=output_dir, overwrite=overwrite, save_probabilities=save_probabilities,
            configuration=options.configuration, folds=options.folds,
            checkpoint_name=options.checkpoint_name,
            source_profile=options.source_profile, precision=options.precision,
            step_fraction=options.step_fraction, use_mirroring=options.use_mirroring,
        )

    def close(self) -> None:
        """Release the model; later :meth:`run` calls raise ``RuntimeError``."""
        self._state = None

    def __enter__(self) -> NnUNetModel:
        """Return this model for a ``with`` block that closes it on exit."""
        return self

    def __exit__(self, *args: object) -> None:
        """Close the model."""
        self.close()


def load_model(
    *, weights_path: PathLike, configuration: str = "3d_fullres",
    folds: Sequence[int | str] | None = None, checkpoint_name: str = "checkpoint_final",
    source_profile: str | None = None, precision: str = "fast-fp32", step_fraction: float = 0.5,
    use_mirroring: bool = True,
) -> NnUNetModel:
    """Load a converted model folder once to segment several cases.

    The arguments mean what they mean in :func:`run`; the returned model's
    ``run`` takes the per-case ones. Close it, or use it as a context manager.
    """
    return NnUNetModel(
        weights_path=weights_path, configuration=configuration, folds=folds,
        checkpoint_name=checkpoint_name, source_profile=source_profile, precision=precision,
        step_fraction=step_fraction, use_mirroring=use_mirroring,
    )
