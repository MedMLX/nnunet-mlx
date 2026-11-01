"""Download CLI contracts at the real shared-cache and Hub boundaries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

import pytest

from nnunet_mlx.cli import main

pytestmark = pytest.mark.host

# The liver bundle revision in the released README; no weights are fetched here.
REVISION = "574a9c601b9b1a9e906ce985fbd46b106c5b7caa"


@pytest.mark.parametrize(("model_id", "repo"), [
    ("nnunet-liver", "MedMLX--nnunet-liver-mlx"),
    ("nnunet-autopet", "MedMLX--nnunet-autopet-mlx"),
    ("nnunet-pancreas", "MedMLX--nnunet-pancreas-mlx"),
    ("nnunet-brats2021", "MedMLX--nnunet-brats-mlx"),
])
def test_download_reuses_the_selected_model_cache_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    model_id: str, repo: str,
) -> None:
    """Both output formats identify the requested completed bundle without network IO."""
    monkeypatch.setenv("MEDMLX_ASSET_CACHE", str(tmp_path))
    folder = tmp_path / model_id / repo
    folder.mkdir(parents=True)
    (folder / ".medmlx-revision").write_text(REVISION + "\n", encoding="utf-8")

    def offline() -> None:
        raise AssertionError("a completed cache must not contact the Hub")

    monkeypatch.setattr(import_module("huggingface_hub"), "HfApi", offline)
    assert main(["download", model_id, "--json"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "model_id": model_id, "weights_path": str(folder), "weights_revision": REVISION,
    }
    assert main(["download", model_id]) == 0
    assert capsys.readouterr().out == str(folder) + "\n"


@dataclass(frozen=True)
class HubModelInfo:
    sha: str = REVISION


class HubApi:
    def model_info(self, repo_id: str) -> HubModelInfo:
        assert repo_id == "MedMLX/nnunet-liver-mlx"
        return HubModelInfo()


def test_download_reports_interruptions_and_records_only_a_completed_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """A failed Hub transfer is a CLI error; retry resumes and records its exact revision."""
    monkeypatch.setenv("MEDMLX_ASSET_CACHE", str(tmp_path))
    hub = import_module("huggingface_hub")
    monkeypatch.setattr(hub, "HfApi", HubApi)
    interrupted = True

    def snapshot_download(
        repo_id: str, *, revision: str, local_dir: Path, force_download: bool,
        tqdm_class: type[object],
    ) -> str:
        assert repo_id == "MedMLX/nnunet-liver-mlx"
        assert revision == REVISION
        assert force_download is False
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "plans.json").write_text('{"fixture": "download CLI"}', encoding="utf-8")
        if interrupted:
            raise OSError("simulated interrupted transfer")
        return str(local_dir)

    monkeypatch.setattr(hub, "snapshot_download", snapshot_download)
    folder = tmp_path / "nnunet-liver" / "MedMLX--nnunet-liver-mlx"
    assert main(["download", "nnunet-liver", "--json"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    failure = json.loads(captured.err.splitlines()[-1])
    assert failure["error"]["message"] == (
        "Could not download MedMLX/nnunet-liver-mlx: simulated interrupted transfer"
    )
    assert not (folder / ".medmlx-revision").exists()

    interrupted = False
    assert main(["download", "nnunet-liver", "--json"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "model_id": "nnunet-liver", "weights_path": str(folder), "weights_revision": REVISION,
    }
    assert (folder / ".medmlx-revision").read_text(encoding="utf-8") == REVISION + "\n"
    assert json.loads((folder / "plans.json").read_text(encoding="utf-8")) == {
        "fixture": "download CLI",
    }


@pytest.mark.parametrize("model_id", ["mrsegmentator", "swin-unetr"])
def test_download_rejects_manual_and_other_runtime_models_before_creating_a_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    model_id: str,
) -> None:
    """The runtime offers only its four automatically downloadable named models."""
    cache = tmp_path / "cache"
    monkeypatch.setenv("MEDMLX_ASSET_CACHE", str(cache))

    def unexpected_network() -> None:
        raise AssertionError("unsupported model IDs must be rejected before a download")

    monkeypatch.setattr(import_module("huggingface_hub"), "HfApi", unexpected_network)
    assert main(["download", model_id]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    message = json.loads(captured.err)["error"]["message"]
    assert message == (
        f"'{model_id}' is not one of 'nnunet-liver', 'nnunet-autopet', "
        "'nnunet-pancreas', 'nnunet-brats2021'."
    )
    assert not cache.exists()
