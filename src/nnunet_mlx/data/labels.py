"""Shared dataset class, region, and ignore-label contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from nnunet_mlx.errors import ContractError


@dataclass(frozen=True, slots=True)
class DatasetLabel:
    """A named output label: ``value`` is the integer written to exported segmentations."""

    name: str
    value: int

    def to_payload(self) -> dict[str, object]:
        """Serialize the label name and exported integer value."""
        return {"name": self.name, "value": self.value}


@dataclass(frozen=True, slots=True)
class LabelSemantics:
    """Class/region structure of a dataset's ``labels``, as nnU-Net's ``LabelManager`` sees it.

    ``output_channels`` is the number of network output channels: the number of
    labels (background included) for class-based datasets, or the number of
    foreground regions for region-based ones. ``all_labels`` are the sorted label
    values excluding the ignore label. ``regions`` holds, per foreground region, the
    label values it covers (empty for class-based datasets);
    ``regions_class_order`` gives the value written to the output for each region.
    ``ignore_label`` is ``None`` if the dataset declares none.
    """

    has_regions: bool
    ignore_label: int | None
    output_channels: int
    all_labels: tuple[int, ...]
    regions: tuple[tuple[int, ...], ...]
    regions_class_order: tuple[int, ...] | None
    label_table: tuple[DatasetLabel, ...]


def parse_dataset_label_semantics(
    dataset_json: Mapping[str, object],
) -> LabelSemantics:
    """Resolve pinned nnU-Net class/region and ignore-label semantics.

    Raises:
        ContractError: If ``labels`` is malformed, background is not 0, label values
            are not consecutive from 0, the ignore label is not the largest label
            plus one, or a region-based dataset lacks a matching
            ``regions_class_order``.
    """

    raw_labels = dataset_json.get("labels")
    if not isinstance(raw_labels, Mapping) or not raw_labels:
        raise ContractError("dataset_json.labels must be a non-empty object")
    labels: dict[str, object] = {}
    for key, value in cast(Mapping[object, object], raw_labels).items():
        if not isinstance(key, str):
            raise ContractError("dataset_json.labels keys must be strings")
        labels[key] = value
    raw_ignore = labels.pop("ignore", None)
    ignore_label = None
    if raw_ignore is not None:
        if isinstance(raw_ignore, bool) or not isinstance(raw_ignore, int):
            raise ContractError("ignore label must be an integer, as required by nnU-Net")
        ignore_label = _label_index(raw_ignore, name="ignore label")
    if _label_index(labels.get("background"), name="background label") != 0:
        raise ContractError("background label must be the scalar integer 0")
    label_values = {
        name: _label_values(value, name=f"label {name!r}") for name, value in labels.items()
    }
    all_labels = sorted({label for values in label_values.values() for label in values})
    if all_labels != list(range(all_labels[-1] + 1)):
        raise ContractError("dataset labels must be consecutive integers starting at 0")
    if ignore_label is not None and ignore_label != max(all_labels) + 1:
        raise ContractError("ignore label must equal the largest label plus one")
    has_regions = any(_is_region_value(value) for value in labels.values())
    if has_regions:
        raw_order = dataset_json.get("regions_class_order")
        if not isinstance(raw_order, list | tuple) or not raw_order:
            raise ContractError("region labels require regions_class_order")
        order = tuple(
            _label_index(value, name="regions_class_order")
            for value in cast(Sequence[object], raw_order)
        )
        region_labels = [name for name, values in label_values.items() if values != (0,)]
        if len(order) != len(region_labels):
            raise ContractError("regions_class_order must have one entry per foreground region")
        return LabelSemantics(
            has_regions=True,
            ignore_label=ignore_label,
            output_channels=len(order),
            all_labels=tuple(all_labels),
            regions=tuple(label_values[name] for name in region_labels),
            regions_class_order=order,
            label_table=(
                DatasetLabel("background", 0),
                *(
                    DatasetLabel(name, value)
                    for name, value in zip(region_labels, order, strict=True)
                ),
            ),
        )
    return LabelSemantics(
        has_regions=False,
        ignore_label=ignore_label,
        output_channels=len(all_labels),
        all_labels=tuple(all_labels),
        regions=(),
        regions_class_order=None,
        label_table=tuple(
            sorted(
                (DatasetLabel(name, values[0]) for name, values in label_values.items()),
                key=lambda label: label.value,
            )
        ),
    )


def _label_index(value: object, *, name: str) -> int:
    # Released datasets use decimal strings; upstream LabelManager applies int().
    """Parse a nonnegative integer label, including decimal-string labels."""
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{name} must be a non-negative integer")
    return value


def _label_values(value: object, *, name: str) -> tuple[int, ...]:
    """Parse a scalar label or a sequence of region labels."""
    raw_values: Sequence[object]
    if isinstance(value, list | tuple):
        if not value:
            raise ContractError(f"{name} must not be empty")
        raw_values = cast(Sequence[object], value)
    else:
        raw_values = (value,)
    return tuple(_label_index(item, name=name) for item in raw_values)


def _is_region_value(value: object) -> bool:
    """Identify a label value containing more than one region member."""
    return isinstance(value, list | tuple) and len(cast(Sequence[object], value)) > 1
