"""Upstream label semantics and request rejection for the registered MedMLX runners."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

import pytest
from medmlx_core.errors import InvalidInputError

from nnunet_mlx import medmlx, named_models
from nnunet_mlx.errors import ContractError

pytestmark = pytest.mark.host

ROOT = Path(__file__).resolve().parents[2]
LABEL_FIXTURE = json.loads(
    (ROOT / "tests" / "fixtures" / "upstream_dataset_labels.json").read_text(encoding="utf-8")
)
LABEL_CASES = cast(list[dict[str, object]], LABEL_FIXTURE["cases"])

RUNNERS: dict[str, Callable[..., object]] = {
    "liver": medmlx.liver,
    "autopet": medmlx.autopet,
    "pancreas": named_models.pancreas,
    "brats": named_models.brats,
    "mrsegmentator": named_models.mrsegmentator,
}
# One case per rejection category; the categories a runner has no argument for are absent.
REJECTIONS: dict[str, dict[str, dict[str, object]]] = {
    "liver": {"non-bool": {"overwrite": "false"}},
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

    def download(model_id: str) -> Path:
        """Fail if a runner reaches its weights download before rejecting the request."""
        raise AssertionError(f"downloaded {model_id} before rejecting {category}")

    monkeypatch.setattr(medmlx, "weights_dir", download)
    monkeypatch.setattr(named_models, "weights_dir", download)
    for name in ("case_0000.nii.gz", "case_0000.png"):
        (tmp_path / name).write_bytes(b"not read")
    arguments = dict(changes)
    image = str(tmp_path / cast(str, arguments.pop("image", "case_0000.nii.gz")))
    count = cast(int, arguments.pop("images", 2))
    output = tmp_path / "out"
    arguments["output_dir"] = str(output)
    if runner == "autopet":
        arguments["image_paths"] = [image] * count
    elif runner == "brats":
        arguments.update({f"{role}_path": image for role in ("flair", "t1", "t1ce", "t2")})
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
