"""Upstream label semantics and request rejection for the released models."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

import medmlx_core.weights as core_weights
import pytest
from medmlx_core.errors import InvalidInputError

from nnunet_mlx import medmlx
from nnunet_mlx.errors import ContractError
from nnunet_mlx.named_models import autopet, brats, liver, mrsegmentator, pancreas

pytestmark = pytest.mark.host

ROOT = Path(__file__).resolve().parents[2]
LABEL_FIXTURE = json.loads(
    (ROOT / "tests" / "fixtures" / "upstream_dataset_labels.json").read_text(encoding="utf-8")
)
LABEL_CASES = cast(list[dict[str, object]], LABEL_FIXTURE["cases"])

RUNNERS: dict[str, Callable[..., object]] = {
    "liver": liver.segment,
    "autopet": autopet.segment,
    "pancreas": pancreas.segment,
    "brats": brats.segment,
    "mrsegmentator": mrsegmentator.segment,
}
# One case per rejection category; the categories a runner has no argument for are absent.
REJECTIONS: dict[str, dict[str, dict[str, object]]] = {
    "liver": {"out-of-range": {"step_fraction": 0}, "non-bool": {"use_mirroring": "false"}},
    "autopet": {"out-of-range": {"images": 1}, "non-bool": {"overwrite": "false"}},
    "pancreas": {"out-of-range": {"fold": -1}, "non-bool": {"use_mirroring": "false"}},
    "brats": {"out-of-range": {"fold": 5}, "non-bool": {"use_mirroring": "false"}},
    "mrsegmentator": {"out-of-range": {"fold": 5}, "non-bool": {"fast": "false"}},
}
CASES = [
    (runner, category, changes)
    for runner, specific in REJECTIONS.items()
    for category, changes in {
        "bad-suffix": {"image": "case_0000.png"},
        **specific,
        "existing-output": {"existing": True},
        "before-download": {"image": "absent.nii.gz"},
    }.items()
]


@pytest.mark.parametrize("case", LABEL_CASES, ids=[str(case["name"]) for case in LABEL_CASES])
def test_dataset_labels_follow_upstream(case: dict[str, object]) -> None:
    """Match dataset class and region labels to upstream semantics."""
    dataset_json = cast(Mapping[str, object], case["dataset_json"])
    unsupported = {
        "decimal_strings_and_ignore",
        "unsorted_with_rejected_values",
        "no_labels",
        "labels_not_mapping",
    }
    if case["name"] in unsupported:
        with pytest.raises(ContractError):
            medmlx.dataset_labels(dataset_json)
        return
    assert medmlx.dataset_labels(dataset_json) == case["labels"]


@pytest.mark.parametrize(
    ("runner", "category", "changes"), CASES, ids=[f"{r}-{c}" for r, c, _ in CASES]
)
def test_runner_rejects_bad_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner: str,
    category: str,
    changes: dict[str, object],
) -> None:
    """Reject each category of bad request; omitted weights are never downloaded first."""

    def download(model_id: str, **_options: object) -> Path:
        """Fail if a runner reaches its weights download before rejecting the request."""
        raise AssertionError(f"downloaded {model_id} before rejecting {category}")

    monkeypatch.setattr(core_weights, "weights_dir", download)
    for name in ("case_0000.nii.gz", "case_0000.png"):
        (tmp_path / name).write_bytes(b"not read")
    arguments = dict(changes)
    image = str(tmp_path / cast(str, arguments.pop("image", "case_0000.nii.gz")))
    count = cast(int, arguments.pop("images", 2))
    output = tmp_path / "out"
    arguments["output_dir"] = str(output)
    if runner == "autopet":
        arguments["image_paths"] = dict(zip(autopet.CHANNELS, [image] * count, strict=False))
    elif runner == "brats":
        arguments["image_paths"] = dict.fromkeys(brats.CHANNELS, image)
    else:
        arguments["image_path"] = image
    if arguments.pop("existing", False):
        folder = tmp_path / "model"
        folder.mkdir()
        dataset = {"labels": {"background": 0, "tumor": 1}, "file_ending": ".nii.gz"}
        (folder / "dataset.json").write_text(json.dumps(dataset), encoding="utf-8")
        output.mkdir()
        (output / "mask.nii.gz").write_bytes(b"keep")
        arguments["weights_path"] = str(folder)
    with pytest.raises(InvalidInputError):
        RUNNERS[runner](**arguments)
