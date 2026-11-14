"""MLX-native nnU-Net probability-file ensembling."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

from nnunet_mlx.contracts import parse_mapping
from nnunet_mlx.data.labels import LabelSemantics, parse_dataset_label_semantics
from nnunet_mlx.errors import ContractError
from nnunet_mlx.prediction.export import (
    COMPLETION_RECEIPT_SCHEMA_VERSION,
    PREDICTION_CONTRACT_VERSION,
    ExportedFiles,
    RestoredPrediction,
    completion_receipt_path,
    export_prediction,
    file_signature,
    prediction_output_files,
    require_prediction_destination,
)
from nnunet_mlx.prediction.inference import probabilities_to_segmentation
from nnunet_mlx.storage.properties import load_properties_pickle

type FloatArray = npt.NDArray[np.float32]


def average_probability_files(
    files: Sequence[str | Path], *, semantics: LabelSemantics | None = None
) -> FloatArray:
    """Average saved post-nonlinearity probabilities in file order.

    Each ``.npz`` must hold a ``probabilities`` array of shape ``(C, *spatial)``
    convertible to float32; its converted values must be finite and in [0, 1].
    Exported arrays use the selected reader's array grid; a reorienting reader can
    differ from the source file's axis order. This function checks shapes, not geometry.
    With ``semantics``, the channel count must equal ``semantics.output_channels`` and
    class-based files must sum to 1 over channels (absolute tolerance 1e-6). Returns the
    float32 mean.

    Raises:
        ContractError: If ``files`` is empty or any file is missing, malformed,
            inconsistent in shape, or violates the checks above.
    """

    paths = tuple(Path(file) for file in files)
    if not paths:
        raise ContractError("probability ensembling requires at least one file")
    total: FloatArray | None = None
    for path in paths:
        probabilities = _load_probabilities(path)
        if semantics is not None:
            if probabilities.shape[0] != semantics.output_channels:
                raise ContractError("ensemble probabilities do not match dataset output channels")
            if not semantics.has_regions and not np.allclose(
                probabilities.sum(axis=0), 1, rtol=0, atol=1e-6
            ):
                raise ContractError("class probabilities must sum to one")
        if total is None:
            total = np.array(probabilities, dtype=np.float32)
            continue
        if total.shape != probabilities.shape:
            raise ContractError(
                f"probability files disagree on shape: {total.shape} vs "
                f"{probabilities.shape} in {path}"
            )
        total += probabilities
    if total is None:
        raise AssertionError("probability averaging unexpectedly produced no total")
    total /= np.float32(len(paths))
    if not np.isfinite(total).all():
        raise ContractError("probability averaging produced non-finite probabilities")
    return np.ascontiguousarray(total, dtype=np.float32)


def merge_probability_files(
    files: Sequence[str | Path],
    output_prefix: str | Path,
    *,
    file_ending: str,
    image_reader_writer: str,
    semantics: LabelSemantics,
    overwrite: bool,
    save_probabilities: bool,
) -> ExportedFiles:
    """Average probability files and export the merged prediction contract.

    Every input needs its ``.pkl`` properties sidecar, restricted to plain Python
    and non-object NumPy metadata without executable globals. Inputs must share the
    same reader-space grid and agree on metadata fields present in multiple sidecars.
    The loader does not establish sidecar provenance. The geometry check compares
    original-grid and active reader-grid metadata. Region inputs must record matching
    labels and region order; class
    inputs can omit label metadata, but recorded labels must match the requested
    output channel count. Where a completion receipt exists, files must still match it.
    The label map is decoded from the averaged probabilities and written with
    ``image_reader_writer``.
    """

    paths = tuple(Path(file) for file in files)
    if not paths:
        raise ContractError("probability merging requires at least one file")
    properties = tuple(_probability_properties(path) for path in paths)
    for path, item in zip(paths, properties, strict=True):
        _validate_probability_receipt(path, item)
    validate_ensemble_properties(properties)
    _validate_requested_semantics(properties, semantics)
    probabilities = average_probability_files(paths, semantics=semantics)
    if probabilities.ndim not in (3, 4) or probabilities.shape[0] != semantics.output_channels:
        raise ContractError("ensemble probabilities do not match dataset output channels")
    for item in properties:
        if "probability_grid" in item:
            grid = parse_mapping(item["probability_grid"], name="probability_grid")
            if (grid.get("shape") != list(probabilities.shape[1:])
                    or grid.get("axis_order") != "ZYX"):
                raise ContractError("ensemble probability grid does not match the saved array")
    segmentation = probabilities_to_segmentation(
        probabilities, regions_class_order=semantics.regions_class_order
    )
    return export_prediction(
        RestoredPrediction(
            segmentation=segmentation,
            probabilities=probabilities if save_probabilities else None,
        ),
        properties[0],
        output_prefix,
        file_ending=file_ending,
        image_reader_writer=image_reader_writer,
        overwrite=overwrite,
        input_files=(*paths, *(path.with_suffix(".pkl") for path in paths)),
    )


def ensemble_folders(
    input_folders: Sequence[str | Path],
    output_folder: str | Path,
    *,
    file_ending: str,
    image_reader_writer: str,
    semantics: LabelSemantics,
    overwrite: bool = False,
    save_probabilities: bool = False,
) -> tuple[ExportedFiles, ...]:
    """Ensemble matching saved cases using nnU-Net's sequential probability mean.

    Input folders must contain the same nonempty set of top-level ``.npz`` files,
    each with a ``.pkl`` properties sidecar. Cases are processed in filename order;
    probability accumulation retains the caller's folder order. The destination
    must differ from every input folder, including symlink aliases.

    Case sets, sidecar presence and output conflicts are checked before writing.
    Each case then uses :func:`merge_probability_files` for probability, geometry,
    label and receipt validation. Inputs must share its reader-space grid contract.
    A later case failure leaves earlier completed outputs in place; retrying those
    outputs requires ``overwrite=True``. No models are loaded or executed.
    """
    sources = tuple(Path(folder) for folder in input_folders)
    if not sources:
        raise ContractError("folder ensembling requires at least one input folder")
    destination = Path(output_folder)
    if destination.exists() and not destination.is_dir():
        raise ContractError(f"output folder is not a directory: {destination}")
    files_per_folder: list[set[str]] = []
    for source in sources:
        if not source.is_dir():
            raise ContractError(f"input folder does not exist: {source}")
        if source.resolve() == destination.resolve():
            raise ContractError("ensemble output folder must differ from input folders")
        files_per_folder.append(
            {path.name for path in source.iterdir() if path.is_file() and path.suffix == ".npz"}
        )
    if not files_per_folder[0]:
        raise ContractError(f"input folder contains no probability files: {sources[0]}")
    for source, names in zip(sources[1:], files_per_folder[1:], strict=True):
        if names != files_per_folder[0]:
            raise ContractError(
                f"ensemble input folders must contain the same case files: {source}"
            )
    case_files = sorted(files_per_folder[0])
    for name in case_files:
        for source in sources:
            properties_file = (source / name).with_suffix(".pkl")
            if not properties_file.is_file():
                raise ContractError(
                    f"probability properties file does not exist: {properties_file}"
                )
        # Export also removes stale probability sidecars, so protect those even
        # when this invocation requests segmentation-only output.
        outputs = prediction_output_files(
            destination / name[:-4], file_ending=file_ending,
            image_reader_writer=image_reader_writer, save_probabilities=True,
        )
        require_prediction_destination(outputs, overwrite=overwrite)
    return tuple(
        merge_probability_files(
            tuple(source / name for source in sources), destination / name[:-4],
            file_ending=file_ending, image_reader_writer=image_reader_writer,
            semantics=semantics, overwrite=overwrite, save_probabilities=save_probabilities,
        )
        for name in case_files
    )


def _load_probabilities(path: Path) -> FloatArray:
    """Load a channel-first float32 ``probabilities`` array, requiring finite values in [0, 1]."""
    if not path.is_file():
        raise ContractError(f"probability file does not exist: {path}")
    with np.load(path) as payload:
        if "probabilities" not in payload:
            raise ContractError(f"probability file lacks 'probabilities': {path}")
        values = np.ascontiguousarray(payload["probabilities"], dtype=np.float32)
    if values.ndim not in (3, 4) or values.shape[0] < 1:
        raise ContractError("probability files require channel-first 2-D/3-D arrays")
    if not np.isfinite(values).all():
        raise ContractError("probability file contains non-finite values")
    if np.any(values < 0) or np.any(values > 1):
        raise ContractError("probabilities must be in [0, 1]")
    return values


def _probability_properties(probabilities_file: Path) -> dict[str, object]:
    """Load the restricted properties sidecar for a probability file."""
    properties_file = probabilities_file.with_suffix(".pkl")
    if not properties_file.is_file():
        raise ContractError(f"probability properties file does not exist: {properties_file}")
    return load_properties_pickle(properties_file)


def validate_ensemble_properties(properties: Sequence[dict[str, object]]) -> None:
    """Require equal original and reader grids and compatible case/label metadata.

    Geometry (SimpleITK spacing/origin/direction, Nibabel original affine, or plain
    spacing) must agree to an absolute tolerance of 1e-6; the identifier, label and
    input-file fields must match exactly where present.

    Raises:
        ContractError: If any input differs.
    """
    grids = tuple(_original_grid(item) for item in properties)
    first = grids[0]
    for grid in grids[1:]:
        if grid.keys() != first.keys():
            raise ContractError("ensemble inputs have incompatible original grid metadata")
        for key, value in first.items():
            other = grid[key]
            if value.shape != other.shape or not np.allclose(value, other, rtol=0, atol=1e-6):
                raise ContractError(f"ensemble inputs have different original grid {key}")
    for key in (
        "case_identifier",
        "case_id",
        "case_name",
        "labels",
        "regions_class_order",
        "inputs",
    ):
        available = [_canonical_json(item[key]) for item in properties if key in item]
        if available and any(value != available[0] for value in available[1:]):
            raise ContractError(f"ensemble inputs have incompatible {key}")


def _validate_probability_receipt(
    probabilities_file: Path, properties: Mapping[str, object],
) -> None:
    """Reject probability files that their completion receipt no longer owns."""

    name = properties.get("completion_receipt")
    if name is None:
        receipt_path = completion_receipt_path(probabilities_file.with_suffix(""))
    else:
        if not isinstance(name, str) or Path(name).name != name:
            raise ContractError("probability completion receipt must be a file name")
        receipt_path = probabilities_file.parent / name
        if not receipt_path.is_file():
            raise ContractError(f"probability completion receipt is unavailable: {receipt_path}")
    if not receipt_path.is_file():
        # External or legacy probability files carry no completion receipt.
        return
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"probability receipt is unreadable: {receipt_path}") from error
    if not isinstance(receipt, dict):
        raise ContractError(f"probability receipt is not an object: {receipt_path}")
    verified = cast(dict[str, object], receipt)
    if (
        verified.get("schema_version") != COMPLETION_RECEIPT_SCHEMA_VERSION
        or verified.get("prediction_contract_version") != PREDICTION_CONTRACT_VERSION
    ):
        raise ContractError(
            f"probability receipt was written by a different contract version; "
            f"re-run prediction: {receipt_path}"
        )
    outputs = verified.get("outputs")
    if not isinstance(outputs, dict):
        raise ContractError(f"probability receipt lacks output signatures: {receipt_path}")
    signatures = cast(dict[str, object], outputs)
    properties_file = probabilities_file.with_suffix(".pkl")
    for member in (probabilities_file, properties_file):
        if not member.is_file() or signatures.get(member.name) != file_signature(member):
            raise ContractError(
                f"probability file is not covered by its completion receipt: {member}"
            )


def _validate_requested_semantics(
    properties: Sequence[dict[str, object]], semantics: LabelSemantics
) -> None:
    """Require every input's recorded label/region provenance to match the requested semantics."""
    # Channel meaning comes from the ordered label provenance, so inputs must
    # match the requested semantics, not just each other.
    if semantics.has_regions:
        requested = _canonical_json(list(semantics.regions_class_order or ()))
        for item in properties:
            recorded = item.get("regions_class_order")
            if recorded is None:
                raise ContractError(
                    "ensemble inputs do not record the requested regions_class_order"
                )
            if _canonical_json(recorded) != requested:
                raise ContractError(
                    "ensemble inputs disagree with the requested regions_class_order"
                )
        for item in properties:
            _require_matching_region_provenance(item, semantics)
        return
    for item in properties:
        if item.get("regions_class_order") is not None:
            raise ContractError(
                "ensemble inputs record a regions_class_order the request does not define"
            )
        if "labels" not in item:
            continue
        if _recorded_semantics(item).output_channels != semantics.output_channels:
            raise ContractError("ensemble inputs record different output labels than the request")


def _require_matching_region_provenance(item: dict[str, object], semantics: LabelSemantics) -> None:
    """Require recorded label regions and class order to match ensemble semantics."""
    labels = item.get("labels")
    if not isinstance(labels, Mapping) or not labels:
        raise ContractError(
            "ensemble inputs do not record the dataset labels required to verify "
            "region channel semantics"
        )
    parsed = _recorded_semantics(item)
    if (
        parsed.regions != semantics.regions
        or parsed.regions_class_order != semantics.regions_class_order
        or parsed.output_channels != semantics.output_channels
    ):
        raise ContractError("ensemble inputs do not match the requested region channel semantics")


def _recorded_semantics(item: dict[str, object]) -> LabelSemantics:
    """Parse the label and region order recorded with an ensemble input."""
    return parse_dataset_label_semantics(
        {"labels": item["labels"], "regions_class_order": item.get("regions_class_order")}
    )


def _canonical_json(value: object) -> str:
    """Serialize metadata deterministically while rejecting nonfinite JSON values."""
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _original_grid(properties: dict[str, object]) -> dict[str, npt.NDArray[np.float64]]:
    """Collect recorded source-grid metadata (SimpleITK, nibabel or bare spacing) as float64.

    Callers compare these across ensemble members to require a shared grid.
    """
    if "sitk_stuff" in properties:
        raw = parse_mapping(properties["sitk_stuff"], name="sitk_stuff")
        fields = {key: raw.get(key) for key in ("spacing", "origin", "direction")}
    elif "nibabel_stuff" in properties:
        raw = parse_mapping(properties["nibabel_stuff"], name="nibabel_stuff")
        fields = {"original_affine": raw.get("original_affine"),
                  "reader_affine": raw.get("reoriented_affine", raw.get("original_affine"))}
    else:
        fields = {"spacing": properties.get("spacing")}
    result = {key: np.asarray(value, dtype=np.float64) for key, value in fields.items()}
    if any(value.ndim == 0 or not np.isfinite(value).all() for value in result.values()):
        raise ContractError("ensemble original grid metadata must be finite arrays")
    if "nibabel_stuff" in properties and "probability_grid" in properties:
        grid = parse_mapping(properties["probability_grid"], name="probability_grid")
        affine = np.asarray(grid.get("affine"), dtype=np.float64)
        expected = result["reader_affine"][:, [2, 1, 0, 3]]
        if affine.shape != (4, 4) or not np.allclose(affine, expected, rtol=0, atol=1e-6):
            raise ContractError("ensemble probability affine differs from its reader grid")
    return result
