"""Host input preparation and prediction restoration/export tasks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from nnunet_mlx.data.labels import LabelSemantics
from nnunet_mlx.data.preprocessing import labelmap_to_one_hot, preprocess_raw_case
from nnunet_mlx.errors import ContractError
from nnunet_mlx.plans import PlansContract
from nnunet_mlx.prediction.export import ExportedFiles, export_prediction, restore_prediction
from nnunet_mlx.prediction.sliding_window import InputArray
from nnunet_mlx.storage.imageio import (
    image_reader_input_sidecars,
    read_images,
    read_segmentation,
    segmentation_reader_input_sidecar,
)


@dataclass(frozen=True, slots=True)
class PreparedPredictionInput:
    """Network input for one case: ``data`` is ``(C, *spatial)`` float32.

    Channels are the preprocessed image modalities followed, for cascade models, by
    one one-hot channel per foreground label of the previous stage. ``properties``
    carries the geometry and crop records needed to restore the prediction.
    Direct prediction may retain resampled MLX data; process workers return NumPy.
    """

    data: InputArray
    properties: dict[str, object]


def prediction_input_signature(
    image_files: Sequence[str | Path],
    previous_stage_file: str | Path | None,
    *,
    image_reader_writer: str,
) -> dict[str, object]:
    """Record input sizes and modification times for receipt change detection.

    These metadata can miss a content change that preserves both values.
    """

    paths = tuple(Path(file) for file in image_files)
    previous = None if previous_stage_file is None else Path(previous_stage_file)
    sidecars = list(image_reader_input_sidecars(image_reader_writer, paths))
    if previous is not None:
        previous_sidecar = segmentation_reader_input_sidecar(image_reader_writer, previous)
        if previous_sidecar is not None:
            sidecars.append(previous_sidecar)
    return {
        "images": [_file_signature(path, name="prediction input file") for path in paths],
        "reader_sidecars": [
            None if not path.is_file() else _file_signature(path, name="reader sidecar")
            for path in sidecars
        ],
        "previous_stage": (
            None if previous is None else _file_signature(previous, name="previous-stage file")
        ),
    }


def _file_signature(path: Path, *, name: str) -> dict[str, object]:
    """Record the size and modification time of an existing input file."""
    if not path.is_file():
        raise ContractError(f"{name} does not exist: {path}")
    metadata = path.stat()
    return {"size": metadata.st_size, "mtime_ns": metadata.st_mtime_ns}


def prepare_prediction_input(
    image_files: Sequence[str | Path],
    *,
    image_reader_writer: str,
    image_channels: int,
    plans: PlansContract,
    semantics: LabelSemantics,
    previous_stage_file: str | Path | None,
    max_output_bytes: int | None = None,
    prefer_mlx: bool = False,
) -> PreparedPredictionInput:
    """Read and preprocess one raw case, optionally retaining resampling on Metal.

    Process-pool callers leave ``prefer_mlx=False`` to return host payloads.

    ``max_output_bytes`` bounds the size of the returned array (for process-pool
    workers); exceeding it raises ``ContractError``. Cascade models need
    ``previous_stage_file`` and non-cascade models reject it.
    """

    data, properties = read_images(image_reader_writer, image_files)
    if int(data.shape[0]) != image_channels:
        raise ContractError(
            f"model expects {image_channels} image channels, reader produced {int(data.shape[0])}"
        )
    previous_stage = _read_previous_stage(
        previous_stage_file,
        properties,
        image_reader_writer=image_reader_writer,
        is_cascade=plans.configuration.previous_stage_name is not None,
    )
    preprocessed = preprocess_raw_case(
        data,
        properties,
        plans,
        segmentation_previous_stage=previous_stage,
        prefer_mlx=prefer_mlx,
    )
    network_input = preprocessed.data
    if preprocessed.previous_stage is not None:
        one_hot = labelmap_to_one_hot(preprocessed.previous_stage[0], semantics.all_labels)
        network_input = np.concatenate((np.asarray(network_input), one_hot))
    if max_output_bytes is not None and network_input.nbytes > max_output_bytes:
        raise ContractError(
            "preprocessed case exceeds worker byte allowance; increase max-inflight-bytes"
        )
    return PreparedPredictionInput(
        data=(np.ascontiguousarray(network_input, dtype=np.float32)
              if isinstance(network_input, np.ndarray) else network_input),
        properties=preprocessed.properties,
    )


def restore_and_export_prediction(
    logits: npt.NDArray[np.generic],
    properties: Mapping[str, object],
    output_prefix: str | Path,
    *,
    plans: PlansContract,
    semantics: LabelSemantics,
    file_ending: str,
    image_reader_writer: str,
    save_probabilities: bool,
    overwrite: bool,
    completion_identity: Mapping[str, object] | None = None,
) -> ExportedFiles:
    """Restore in the inference process, then write the completed prediction."""

    restored = restore_prediction(
        logits,
        properties,
        plans,
        semantics,
        return_probabilities=save_probabilities,
    )
    return export_prediction(
        restored,
        properties,
        output_prefix,
        file_ending=file_ending,
        image_reader_writer=image_reader_writer,
        overwrite=overwrite,
        completion_identity=completion_identity,
    )


def _read_previous_stage(
    previous_stage_file: str | Path | None,
    properties: Mapping[str, object],
    *,
    image_reader_writer: str,
    is_cascade: bool,
) -> npt.NDArray[np.generic] | None:
    """Read and validate the previous-stage segmentation required by a cascade."""
    if is_cascade and previous_stage_file is None:
        raise ContractError("cascade prediction requires --previous-stage")
    if not is_cascade and previous_stage_file is not None:
        raise ContractError("non-cascade prediction does not accept --previous-stage")
    return (
        None
        if previous_stage_file is None
        else read_segmentation(image_reader_writer, previous_stage_file, properties)
    )
