"""MedMLX liver and AutoPET declarations and the runner for converted MLX model folders.

Prediction uses :class:`ModelFolderPredictor` with nnU-Net v2.8.1's
sliding-window accumulation and checkpoint mirroring; the prediction CLI builds
the same predictor through :func:`predictor`. Label metadata uses the
package's shared dataset contract. Importing this module does not initialize MLX.

The runner reports failures as ``medmlx_core`` errors, whose type names reach
MCP clients. Each step re-raises the package's :class:`ContractError` as the
error for its kind of failure, with the same message and the ``ContractError``
as ``__cause__``.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Literal, cast

from medmlx_core.weights import weights_dir, weights_revision

from nnunet_mlx.contracts import parse_boolean, parse_number, parse_sequence
from nnunet_mlx.errors import ContractError
from nnunet_mlx.medmlx_types import RunResult

if TYPE_CHECKING:
    from nnunet_mlx.model.folder import ModelBundle
    from nnunet_mlx.prediction.pipeline import ModelFolderPredictor

Failure = Literal["input", "asset", "dependency", "execution"]

LIVER = {
    "id": "nnunet-liver",
    "label": "nnU-Net liver CT",
    "task": "segmentation",
    "readiness": "nnunet_mlx.medmlx:readiness",
    "run": "nnunet_mlx.medmlx:liver",
    "summary": "Liver and liver cancer segmentation of one CT volume.",
    "labels": {"1": "liver", "2": "cancer"},
}
AUTOPET = {
    "id": "nnunet-autopet",
    "label": "nnU-Net AutoPET II",
    "task": "segmentation",
    "readiness": "nnunet_mlx.medmlx:readiness",
    "run": "nnunet_mlx.medmlx:autopet",
    "summary": "Tumor lesion segmentation of co-registered CT and SUV PET volumes.",
    "labels": {"1": "tumor"},
}

# MedMLX runs and the prediction CLI accumulate windows in float16, as nnU-Net v2.8.1 does.
ACCUMULATION = "nnunet-fp16"

SEGMENTATION_STEM = "mask"
PROBABILITIES = "probabilities.npz"
PROBABILITY_PROPERTIES = "probabilities.pkl"
NIFTI_ENDINGS = (".nii.gz", ".nii")
TORCH_ORDER_FLOAT32 = "torch-order-fp32"
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


def predictor(bundle: ModelBundle, *, step_fraction: float) -> ModelFolderPredictor:
    """Return the predictor MedMLX runs and the prediction CLI use for ``bundle``.

    Windows and folds accumulate in float16 as in nnU-Net v2.8.1 (``nnunet-fp16``),
    and models run uncompiled. ``ModelFolderPredictor``'s own defaults (float32
    accumulation, automatic compilation) are for direct Python use.
    """
    from nnunet_mlx.prediction.pipeline import ModelFolderPredictor

    return ModelFolderPredictor(
        bundle,
        compile_models=False,
        accumulation=ACCUMULATION,
        step_fraction=step_fraction,
    )


def check_request(
    image_paths: Sequence[str],
    *,
    output_dir: str,
    overwrite: object = False,
    folds: Sequence[int | str] | None = None,
    source_profile: str | None = None,
    arithmetic: str = "native",
    step_fraction: object = 0.5,
    use_mirroring: object = True,
    save_probabilities: object = False,
) -> tuple[list[Path], tuple[int | str, ...] | None]:
    """Check the arguments that need no model folder, before weights are resolved.

    Named runners call this before downloading their model folder, so a typo fails
    at once. Returns the image paths and the validated folds.

    Raises:
        InvalidInputError: Missing or non-NIfTI images; non-boolean flags; invalid
            folds, ``step_fraction``, ``arithmetic`` or ``source_profile``; or an
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
        if arithmetic not in {"native", TORCH_ORDER_FLOAT32}:
            raise ContractError(f"unsupported inference arithmetic {arithmetic!r}")
        if source_profile is not None:
            resolve_inference_source_profile(source_profile)
        output = Path(output_dir).expanduser()
        if output.exists() and not output.is_dir():
            raise ContractError(f"output_dir must be a directory: {output}")
    return images, selected_folds


def run(
    *,
    weights_path: str,
    image_paths: Sequence[str],
    output_dir: str,
    overwrite: bool = False,
    configuration: str | None = None,
    folds: Sequence[int | str] | None = None,
    checkpoint_name: str = "checkpoint_final",
    source_profile: str | None = None,
    arithmetic: str = "native",
    step_fraction: float = 0.5,
    use_mirroring: bool = True,
    save_probabilities: bool = False,
) -> RunResult:
    """Segment one case and restore the label map to the input image's grid.

    Args:
        weights_path: Local native MLX model folder. Use a named runner for
            automatic downloads where configured. A folder such as the published
            ``MedMLX/nnunet-liver-mlx`` bundle contains ``plans.json``,
            ``dataset.json`` and
            ``fold_<n>/checkpoint_final/{state.json,weights.safetensors}`` for the
            selected folds. Pass the folder root as ``weights_path``.
            Conversion receipts are optional export metadata, not inference inputs.
        image_paths: NIfTI images (``.nii`` or ``.nii.gz``), one per modality in
            ``dataset.json`` channel order. Single-case filenames are unrestricted
            apart from the NIfTI ending.
        output_dir: Directory for the outputs; created if missing.
        overwrite: Replace existing outputs; input/output aliases are always refused.
        configuration: Plans configuration; defaults to ``3d_fullres``.
        folds: Folds to ensemble; ``None`` discovers every numeric fold with the
            checkpoint. An explicit ``["all"]`` selects ``fold_all``.
        checkpoint_name: MLX checkpoint directory inside each fold; defaults to
            ``checkpoint_final``.
        source_profile: Compatibility profile; defaults to the nnU-Net v2.8.1 profile.
        arithmetic: ``native`` (default), or ``torch-order-fp32`` to run MLX kernels
            on Metal in the pinned PyTorch CPU float32 operation order (only for
            the model profiles that admit it).
        step_fraction: Sliding-window step as a fraction of the patch, in (0, 1].
        use_mirroring: Apply the checkpoint's mirroring test-time augmentation.
        save_probabilities: Also write class or region probabilities and their properties.

    Returns:
        ``outputs`` maps ``segmentation`` to ``mask.nii.gz`` (or ``mask.nii``,
        following ``dataset.json``), a label map with the input's shape and
        geometry, and, with ``save_probabilities``, ``probabilities`` and
        ``probability_properties`` to ``probabilities.npz`` and
        ``probabilities.pkl``. ``details`` holds the label table, output kind,
        selected options, checkpoint signatures and timings.

    Raises:
        InvalidInputError: Missing or non-NIfTI images; non-boolean flags; invalid
            folds, ``step_fraction``, ``arithmetic`` or ``source_profile``; an
            ``output_dir`` that is a file; existing outputs without ``overwrite``.
        AssetNotReadyError: A missing model folder, or one whose files,
            configuration, folds or checkpoints are missing, malformed,
            unsupported, or inconsistent with the selected configuration and profile.
        MissingDependencyError: MLX is missing, is not version 0.32.4, or has
            no usable accelerator.
        ModelExecutionError: Reading, predicting or publishing the case failed,
            including images that do not give the model's channel count.
    """

    from nnunet_mlx.plans import read_json_object
    from nnunet_mlx.prediction.source import DEFAULT_INFERENCE_SOURCE_PROFILE

    images, selected_folds = check_request(
        image_paths,
        output_dir=output_dir,
        overwrite=overwrite,
        folds=folds,
        source_profile=source_profile,
        arithmetic=arithmetic,
        step_fraction=step_fraction,
        use_mirroring=use_mirroring,
        save_probabilities=save_probabilities,
    )
    folder = Path(weights_path).expanduser()
    with _reported_as("asset"):
        if not folder.is_dir():
            raise ContractError(f"nnU-Net model folder does not exist: {folder}; {_WEIGHTS_HINT}")
    with _reported_as("asset"):
        if not (folder / "dataset.json").is_file():
            raise ContractError(f"model folder lacks dataset.json: {folder}; {_WEIGHTS_HINT}")
        dataset_json = read_json_object(folder / "dataset.json")
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
    targets = {"segmentation": output / f"{SEGMENTATION_STEM}{file_ending}"}
    if save_probabilities:
        targets["probabilities"] = output / PROBABILITIES
        targets["probability_properties"] = output / PROBABILITY_PROPERTIES
    existing = [path.as_posix() for path in targets.values() if path.exists()]
    with _reported_as("input"):
        if any(
            target.resolve() == image.resolve()
            or (target.exists() and target.samefile(image))
            for target in targets.values()
            for image in images
        ):
            raise ContractError("nnU-Net outputs must differ from every input image")
        if existing and not overwrite:
            raise ContractError(
                f"nnU-Net outputs already exist: {', '.join(existing)}; "
                "pass overwrite=True to replace them"
            )

    from nnunet_mlx.runtime import import_mlx, runtime_info

    with _reported_as("dependency"):
        backend = runtime_info().backend
    from nnunet_mlx.model.folder import load_model_bundle

    with _reported_as("dependency"):
        mx, _ = import_mlx()
    mx.reset_peak_memory()
    started = perf_counter()
    with _reported_as("asset"):
        bundle = load_model_bundle(
            folder,
            configuration=configuration,
            folds=selected_folds,
            checkpoint_name=checkpoint_name,
            source_profile=source_profile,
            arithmetic=arithmetic,
        )
    if not use_mirroring:
        bundle = replace(bundle, mirroring_axes=())
    model_load_s = perf_counter() - started
    with _reported_as("execution"):
        case_predictor = predictor(bundle, step_fraction=float(step_fraction))
        output.mkdir(parents=True, exist_ok=True)
        # Predict into a private directory, then publish under the runner's output names.
        with tempfile.TemporaryDirectory(prefix=".nnunet-mlx-", dir=output) as staging:
            result = case_predictor.predict_case(
                images,
                Path(staging) / "case",
                save_probabilities=save_probabilities,
                overwrite=True,
            )
            produced = {
                "segmentation": result.files.segmentation,
                "probabilities": result.files.probabilities,
                "probability_properties": result.files.properties,
            }
            for role, target in targets.items():
                source = produced[role]
                if source is None:
                    raise ContractError(f"nnU-Net prediction did not write its {role} output")
                os.replace(source, target)
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
    payload: RunResult = {
        "outputs": {role: path.resolve().as_posix() for role, path in targets.items()},
        "details": {
            "weights_revision": weights_revision(folder),
            "labels": labels,
            "output_kind": output_kind(labels),
            "configuration": configuration,
            "source_profile": source_profile,
            "arithmetic": arithmetic,
            "folds": list(bundle.folds),
            "checkpoints": [item.to_payload() for item in bundle.checkpoints],
            "mirroring_axes": list(bundle.mirroring_axes),
            "step_fraction": step_fraction,
            "accumulation": ACCUMULATION,
            "backend": backend,
            "preprocessed_shape": list(result.preprocessed_shape),
            "metrics": metrics,
        },
    }
    return payload


def liver(
    *, image_path: str, output_dir: str, weights_path: str | None = None,
    overwrite: bool = False,
) -> RunResult:
    """Segment liver CT; automatically obtain the named model when weights are omitted."""
    check_request([image_path], output_dir=output_dir, overwrite=overwrite)
    folder = (weights_dir("nnunet-liver") if weights_path is None
              else Path(weights_path).expanduser())
    return run(weights_path=str(folder), image_paths=[image_path], output_dir=output_dir,
               overwrite=overwrite)


def autopet(
    *, image_paths: Sequence[str], output_dir: str, weights_path: str | None = None,
    overwrite: bool = False,
) -> RunResult:
    """Segment co-registered CT and SUV PET, supplied in that order."""
    from medmlx_core.errors import InvalidInputError

    check_request(image_paths, output_dir=output_dir, overwrite=overwrite)
    if len(image_paths) != 2:
        raise InvalidInputError("AutoPET requires image_paths in CT, then SUV PET order")
    folder = (weights_dir("nnunet-autopet") if weights_path is None
              else Path(weights_path).expanduser())
    return run(weights_path=str(folder), image_paths=image_paths, output_dir=output_dir,
               configuration="3d_fullres_resenc_bs80", overwrite=overwrite)
