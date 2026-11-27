"""Prepare aligned cases and stream fixed-size channels-last training patches."""

from __future__ import annotations

import json
from collections.abc import Generator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from math import cos, radians, sin
from pathlib import Path
from typing import cast
from uuid import uuid4

import numpy as np
import numpy.typing as npt
from scipy.ndimage import affine_transform

from nnunet_mlx.contracts import parse_integer, parse_mapping, parse_sequence, parse_string
from nnunet_mlx.data.cases import RawInputCase, discover_input_cases
from nnunet_mlx.data.labels import LabelSemantics, parse_dataset_label_semantics
from nnunet_mlx.data.preprocessing import preprocess_raw_case
from nnunet_mlx.data.resampling import resample_segmentation
from nnunet_mlx.errors import ContractError
from nnunet_mlx.model.metadata import input_channel_count, validate_folds
from nnunet_mlx.plans import PlansContract, read_json_object
from nnunet_mlx.storage.filesystem import write_new_directory
from nnunet_mlx.storage.imageio import (
    read_images,
    read_segmentation,
    require_matching_geometry,
    resolve_image_reader,
)
from nnunet_mlx.training.planning import native_plans, select_reader

type FloatArray = npt.NDArray[np.float32]
type Labels = npt.NDArray[np.integer]
type Batch = tuple[FloatArray, npt.NDArray[np.int32]]

_DATA_SCHEMA = "nnunet-mlx.training-data.v1"
_SAMPLE_LIMIT = 1024


@dataclass(frozen=True, slots=True)
class Split:
    """Nonoverlapping case identifiers for one training and validation split."""

    train: tuple[str, ...]
    val: tuple[str, ...]

    @classmethod
    def from_payload(cls, payload: object, *, allow_empty_validation: bool = False) -> Split:
        values = parse_mapping(payload, name="training split")
        train = _case_ids(values.get("train"), name="train")
        val = _case_ids(values.get("val"), name="val", allow_empty=allow_empty_validation)
        if set(train).intersection(val):
            raise ContractError("training and validation cases must be disjoint")
        return cls(train, val)


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    """A prepared folder of cropped and resampled cases for one train/validation split.

    ``folder`` is the directory passed to training. ``split`` records case IDs,
    and ``arrays`` maps an individual case without loading it into memory.
    """

    folder: Path
    identifier: str
    fold: int | str
    split: Split
    plans: PlansContract
    dataset: dict[str, object]
    semantics: LabelSemantics
    channels: int

    @classmethod
    def load(cls, folder: str | Path) -> PreparedDataset:
        """Open an existing prepared folder and parse its plans and split."""
        folder = Path(folder)
        metadata = read_json_object(folder / "prepared.json")
        if metadata.get("schema") != _DATA_SCHEMA:
            raise ContractError("unsupported prepared training dataset schema")
        dataset = read_json_object(folder / "dataset.json")
        raw_fold = metadata.get("fold")
        fold = "all" if raw_fold == "all" else parse_integer(raw_fold, name="fold")
        validate_folds((fold,))
        split = Split.from_payload(metadata.get("split"), allow_empty_validation=fold == "all")
        return cls(
            folder,
            parse_string(metadata.get("id"), name="prepared dataset ID"),
            fold,
            split,
            PlansContract.from_payload(
                read_json_object(folder / "plans.json"),
                configuration=parse_string(metadata.get("configuration"), name="configuration"),
            ),
            dataset,
            parse_dataset_label_semantics(dataset),
            input_channel_count(dataset),
        )

    def arrays(self, case: str) -> tuple[FloatArray, Labels]:
        """Map an image in DHWC order and its spatial label map without loading a volume."""
        folder = self.folder / "cases" / case
        try:
            image = cast(
                FloatArray, np.load(folder / "image.npy", mmap_mode="r", allow_pickle=False)
            )
            labels = cast(Labels, np.load(folder / "labels.npy", mmap_mode="r", allow_pickle=False))
        except FileNotFoundError as error:
            raise ContractError(f"prepared files are missing for {case!r}") from error
        if image.dtype != np.float32 or image.shape != (*labels.shape, self.channels):
            raise ContractError(f"prepared image and label shapes disagree for {case!r}")
        if labels.ndim != 3 or labels.dtype.kind not in "iu":
            raise ContractError(f"prepared labels for {case!r} must be a 3-D integer array")
        return image, labels


def prepare_training_data(
    source: str | Path,
    destination: str | Path,
    *,
    plans_file: str | Path | None = None,
    configuration: str | None = None,
    patch_size: tuple[int, ...] | None = None,
    splits_file: str | Path | None = None,
    fold: int | str = 0,
) -> PreparedDataset:
    """Prepare a raw dataset with ``dataset.json``, ``imagesTr`` and ``labelsTr``.

    The destination is created after all selected cases have been prepared.
    Set ``fold`` to ``"all"`` to prepare every case for training.
    """
    source = Path(source)
    destination = Path(destination)
    dataset = read_json_object(source / "dataset.json")
    semantics = parse_dataset_label_semantics(dataset)
    if len(semantics.all_labels) < 2:
        raise ContractError("training needs at least one foreground label")
    if semantics.all_labels[-1] > np.iinfo(np.uint16).max:
        raise ContractError("output labels must fit in uint16")
    ending = parse_string(dataset.get("file_ending"), name="dataset file_ending")
    if not ending.startswith(".") or Path(ending).name != ending:
        raise ContractError("file_ending must be a file extension")
    channels = input_channel_count(dataset)
    cases = {case.identifier: case for case in discover_input_cases(source / "imagesTr", ending)}
    split = _select_split(_case_ids(tuple(cases), name="dataset"), fold, splits_file)
    if plans_file is None:
        plans = native_plans(
            [cases[case] for case in split.train],
            dataset_name=source.resolve().name,
            image_channels=channels,
            reader=select_reader(ending, dataset.get("overwrite_image_reader_writer")),
            configuration=configuration,
            patch_size=patch_size,
        )
    else:
        if patch_size is not None:
            raise ContractError("set patch_size in the supplied plans file")
        payload = read_json_object(plans_file)
        available = parse_mapping(payload.get("configurations"), name="configurations")
        if configuration is None:
            configuration = "3d_fullres" if "3d_fullres" in available else "2d"
        plans = PlansContract.from_payload(payload, configuration=configuration)
    if plans.configuration.previous_stage_name is not None:
        raise ContractError("training requires a single-stage configuration")
    reader = resolve_image_reader(plans.image_reader_writer, file_ending=ending)
    selected = (*split.train, *split.val)
    for case_id in selected:
        if not (source / "labelsTr" / f"{case_id}{ending}").is_file():
            raise ContractError(f"missing training label for {case_id!r}")

    def write(folder: Path) -> None:
        rng = np.random.default_rng(0)
        for case_id in selected:
            case = cases[case_id]
            label_file = source / "labelsTr" / f"{case_id}{ending}"
            try:
                _prepare_case(
                    folder / "cases" / case_id,
                    case,
                    label_file,
                    reader,
                    plans,
                    semantics,
                    channels,
                    rng,
                    sample_foreground=case_id in split.train,
                )
            except ContractError as error:
                raise ContractError(f"case {case_id!r}: {error}") from error
        _write_json(folder / "plans.json", plans.raw)
        _write_json(folder / "dataset.json", dataset)
        _write_json(
            folder / "prepared.json",
            {
                "schema": _DATA_SCHEMA,
                "id": uuid4().hex,
                "fold": fold,
                "split": {"train": split.train, "val": split.val},
                "configuration": plans.configuration.name,
            },
        )

    write_new_directory(destination, write)
    return PreparedDataset.load(destination)


def _prepare_case(
    folder: Path,
    case: RawInputCase,
    label_file: Path,
    reader: str,
    plans: PlansContract,
    semantics: LabelSemantics,
    channels: int,
    rng: np.random.Generator,
    *,
    sample_foreground: bool,
) -> None:
    if reader.startswith("NibabelIO") and len(case.image_files) > 1:
        modalities = [read_images(reader, (path,)) for path in case.image_files]
        properties = modalities[0][1]
        for modality, geometry in modalities[1:]:
            require_matching_geometry(reader, geometry, properties)
            if modality.shape != modalities[0][0].shape:
                raise ContractError("input modalities have different shapes")
        image = np.concatenate([values for values, _ in modalities], axis=0)
        del modalities
    else:
        image, properties = read_images(reader, case.image_files)
    target = read_segmentation(reader, label_file, properties)
    if target.shape != (1, *image.shape[1:]) or image.shape[0] != channels:
        raise ContractError("image channels or label shape do not match the dataset")
    maximum = semantics.ignore_label or semantics.all_labels[-1]
    if int(target.max()) > maximum:
        raise ContractError("segmentation contains labels absent from dataset.json")
    prepared = preprocess_raw_case(image, properties, plans)
    geometry = prepared.geometry
    transposed = target.transpose(0, *(axis + 1 for axis in plans.transpose_forward))
    slices = (slice(None), *(slice(start, stop) for start, stop in geometry.bounding_box))
    cropped = transposed[slices]
    foreground_max = semantics.all_labels[-1]
    if _foreground_count(cropped, foreground_max) != _foreground_count(target, foreground_max):
        raise ContractError("the image crop excludes a foreground label")
    labels = cast(
        Labels,
        resample_segmentation(
            cropped,
            prepared.data.shape[1:],
            geometry.source_spacing,
            geometry.target_spacing,
            plans.configuration.segmentation_resampling,
        )[0],
    ).astype(np.min_scalar_type(maximum), copy=False)
    folder.mkdir(parents=True)
    channels_last = np.moveaxis(prepared.data, 0, -1)
    cached = np.lib.format.open_memmap(
        folder / "image.npy",
        mode="w+",
        dtype=np.float32,
        shape=channels_last.shape,
    )
    np.copyto(cached, channels_last)
    cached.flush()
    np.save(folder / "labels.npy", np.ascontiguousarray(labels))
    if sample_foreground:
        np.savez(
            folder / "foreground.npz",
            *(_sample_points(labels, label, rng) for label in semantics.all_labels[1:]),
            allow_pickle=False,
        )


def _foreground_count(labels: Labels, maximum: int) -> int:
    return int(np.count_nonzero(labels) - np.count_nonzero(labels > maximum))


def _sample_points(
    labels: Labels, label: int, rng: np.random.Generator, *, exclude: bool = False
) -> npt.NDArray[np.int32]:
    """Sample coordinates matching or excluding a label with bounded temporary storage."""
    values = labels.reshape(-1)
    sampled = np.empty(0, np.intp)
    seen = 0
    block_size = 1 << 20
    for start in range(0, len(values), block_size):
        block = values[start : start + block_size]
        matches = np.flatnonzero(block != label if exclude else block == label)
        count = len(matches)
        if count:
            size = min(seen + count, _SAMPLE_LIMIT)
            # The hypergeometric draw combines two uniform subsamples exactly.
            take = rng.hypergeometric(count, seen, size) if seen else size
            old = rng.choice(sampled, size - take, replace=False)
            new = start + rng.choice(matches, take, replace=False)
            sampled = np.concatenate((old, new))
            seen += count
    return np.stack(np.unravel_index(sampled, labels.shape), axis=-1).astype(np.int32)


class PatchSampler:
    """Prefetch one fixed-size image/label batch while Metal evaluates the current step."""

    def __init__(
        self,
        data: PreparedDataset,
        *,
        batch_size: int,
        augment: bool,
        focus_label: int | None = None,
        spatial_augment: bool = False,
    ) -> None:
        self._data = data
        self._patch = data.plans.configuration.patch_size
        self._batch_size = batch_size
        self._augment = augment
        self._spatial_augment = spatial_augment
        self._focus_index = (
            data.semantics.all_labels[1:].index(focus_label) if focus_label is not None else None
        )
        self._focus_cases = (
            _cases_with_foreground(data, self._focus_index) if self._focus_index is not None else ()
        )
        ignore = data.semantics.ignore_label
        self._annotations = {} if ignore is None else _training_annotations(data, ignore)
        self._sample_cases = tuple(self._annotations) if self._annotations else data.split.train

        @lru_cache(maxsize=8)
        def load_case(identifier: str) -> tuple[FloatArray, Labels, tuple[Labels, ...]]:
            image, labels = data.arrays(identifier)
            path = data.folder / "cases" / identifier / "foreground.npz"
            with np.load(path, allow_pickle=False) as foreground:
                points = tuple(cast(Labels, foreground[key]) for key in foreground.files)
            return image, labels, points

        self._load_case = load_case

    def batches(self, *, seed: int, epoch: int, steps: int) -> Generator[Batch]:
        """Use an epoch-local generator so resuming reproduces the next batch stream."""
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(self._batch, rng)
            for step in range(steps):
                batch = pending.result()
                if step + 1 < steps:
                    pending = pool.submit(self._batch, rng)
                yield batch

    def _batch(self, rng: np.random.Generator) -> Batch:
        channels = self._data.channels
        images = np.zeros((self._batch_size, *self._patch, channels), dtype=np.float32)
        targets = np.full((self._batch_size, *self._patch), -1, dtype=np.int32)
        cases = self._sample_cases
        rank = len(self._patch)
        ignore = self._data.semantics.ignore_label
        for index in range(self._batch_size):
            case = cases[int(rng.integers(len(cases)))]
            image, labels, class_locations = self._load_case(case)
            foreground = tuple(points for points in class_locations if points.size)
            center: tuple[int, ...] | None = None
            if (foreground or self._focus_cases) and rng.random() < 1 / 3:
                if self._focus_cases and self._focus_index is not None and rng.random() < 2 / 3:
                    focused = self._focus_cases[int(rng.integers(len(self._focus_cases)))]
                    image, labels, class_locations = self._load_case(focused)
                    center = _sample_location(class_locations[self._focus_index], rng)
                elif foreground:
                    center = _sample_location(foreground[int(rng.integers(len(foreground)))], rng)
            if center is None and self._annotations:
                center = _sample_location(self._annotations[case], rng)
            if rank == 2:
                plane = center[0] if center is not None else int(rng.integers(len(labels)))
                image, labels = image[plane], labels[plane]
            source, dest = _patch_slices(labels.shape, self._patch, center, rng)
            values, target = image[source], labels[source]
            if self._augment and not self._spatial_augment:
                axes = tuple(axis for axis in range(rank) if rng.random() < 0.5)
                values, target = np.flip(values, axis=axes), np.flip(target, axis=axes)
            images[(index, *dest)] = values
            targets[(index, *dest)] = target
            # Intensity jitter is applied to real voxels only, leaving padding at zero.
            if self._augment:
                patch = images[(index, *dest)]
                patch *= rng.uniform(0.9, 1.1, channels).astype(np.float32)
                patch += rng.uniform(-0.1, 0.1, channels).astype(np.float32)
                if self._spatial_augment and rng.random() < 1 / 3:
                    _rotate_in_plane(
                        images[index], targets[index], radians(rng.uniform(-10, 10)), ignore
                    )
        if ignore is not None:
            targets[targets == ignore] = -1
        return images, targets


def _training_annotations(data: PreparedDataset, ignore: int) -> dict[str, Labels]:
    """Index bounded samples of annotated voxels from the prepared training split."""
    rng = np.random.default_rng(0)
    annotated: dict[str, Labels] = {}
    for case in data.split.train:
        _, labels = data.arrays(case)
        points = _sample_points(labels, ignore, rng, exclude=True)
        if points.size:
            annotated[case] = points
    if not annotated:
        raise ContractError("the prepared training split contains no annotated voxels")
    return annotated


def _cases_with_foreground(data: PreparedDataset, index: int) -> tuple[str, ...]:
    """Find training cases whose cached foreground coordinates contain the selected class."""
    cases: list[str] = []
    for identifier in data.split.train:
        path = data.folder / "cases" / identifier / "foreground.npz"
        with np.load(path, allow_pickle=False) as foreground:
            if cast(Labels, foreground[f"arr_{index}"]).size:
                cases.append(identifier)
    if not cases:
        raise ContractError("the focus label is absent from the prepared training cases")
    return tuple(cases)


def _sample_location(points: Labels, rng: np.random.Generator) -> tuple[int, ...]:
    return tuple(int(value) for value in points[int(rng.integers(len(points)))])


def _rotate_in_plane(
    image: FloatArray, labels: npt.NDArray[np.int32], angle: float, ignore: int | None
) -> None:
    """Rotate both tensors around the patch center while retaining any sparse annotation."""
    transform = np.eye(labels.ndim)
    transform[-2:, -2:] = ((cos(angle), -sin(angle)), (sin(angle), cos(angle)))
    center = (np.asarray(labels.shape, dtype=np.float64) - 1) * 0.5
    offset = center - transform @ center
    rotated = np.empty_like(labels)
    affine_transform(
        labels,
        transform,
        offset,
        output=rotated,
        order=0,
        cval=-1,
        prefilter=False,
    )
    if ignore is not None and not np.any((rotated >= 0) & (rotated != ignore)):
        return
    labels[...] = rotated
    for channel in range(image.shape[-1]):
        values = image[..., channel]
        affine_transform(values.copy(), transform, offset, output=values, order=1, prefilter=False)


def _patch_slices(
    shape: tuple[int, ...],
    patch: tuple[int, ...],
    center: tuple[int, ...] | None,
    rng: np.random.Generator,
) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    positions = center[-len(patch) :] if center is not None else (None,) * len(patch)
    source: list[slice] = []
    destination: list[slice] = []
    for size, width, position in zip(shape, patch, positions, strict=True):
        if size <= width:
            start = -((width - size) // 2)
        elif position is None:
            start = int(rng.integers(size - width + 1))
        else:
            start = min(max(position - width // 2, 0), size - width)
        lower, upper = max(0, start), min(size, start + width)
        source.append(slice(lower, upper))
        destination.append(slice(lower - start, upper - start))
    return tuple(source), tuple(destination)


def _select_split(cases: Sequence[str], fold: int | str, path: str | Path | None) -> Split:
    validate_folds((fold,))
    if isinstance(fold, str):
        return Split(tuple(cases), ())
    if path is None:
        count = min(5, len(cases))
        if count < 2 or fold >= count:
            raise ContractError("training needs at least two cases and a valid fold")
        shuffled = np.random.default_rng(0).permutation(len(cases))
        validation = {cases[int(index)] for index in np.array_split(shuffled, count)[fold]}
        return Split(
            tuple(case for case in cases if case not in validation),
            tuple(sorted(validation)),
        )
    try:
        payload: object = json.loads(Path(path).read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"invalid splits file: {path}") from error
    splits = parse_sequence(payload, name="splits")
    if fold >= len(splits):
        raise ContractError(f"splits file does not contain fold {fold}")
    split = Split.from_payload(splits[fold])
    if absent := set((*split.train, *split.val)).difference(cases):
        raise ContractError(f"split cases absent from imagesTr: {sorted(absent)}")
    return split


def _case_ids(values: object, *, name: str, allow_empty: bool = False) -> tuple[str, ...]:
    result = tuple(parse_string(value, name=name) for value in parse_sequence(values, name=name))
    if not (result or allow_empty) or len(set(result)) != len(result):
        raise ContractError(f"{name} requires unique case IDs and cannot be empty here")
    if any(value in {".", ".."} or Path(value).name != value for value in result):
        raise ContractError(f"{name} case IDs must be single path components")
    return result


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
