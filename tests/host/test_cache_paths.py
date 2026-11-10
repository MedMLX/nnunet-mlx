"""Upstream conversion cache paths stay inside the selected model entry."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from nnunet_mlx.storage import upstream

pytestmark = pytest.mark.host


def test_cached_conversion_normalizes_parent_components_in_cache_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A trailing child/.. source path cannot remove neighboring cache files."""
    trained = tmp_path / "Dataset001" / "Trainer"
    (trained / "child").mkdir(parents=True)
    source = trained / "child" / ".."
    cache = tmp_path / "cache"
    converted = cache / "converted"
    (converted / "child").mkdir(parents=True)
    neighbor = converted / "keep.txt"
    neighbor.write_text("keep", encoding="utf-8")

    def detect(_folder: Path, *, checkpoint_name: str) -> tuple[int, ...]:
        return (0,)

    def is_current(
        _target: Path, _folder: Path, _folds: tuple[int | str, ...], _checkpoint: str
    ) -> bool:
        return False

    def convert(
        _folder: Path, destination: Path, *, folds: Sequence[int | str], checkpoint_name: str
    ) -> None:
        destination.mkdir(parents=True)

    monkeypatch.setattr(upstream, "detect_upstream_folds", detect)
    monkeypatch.setattr(upstream, "_conversion_is_current", is_current)
    monkeypatch.setattr(upstream, "convert_trained_model", convert)

    result = upstream.cached_conversion(source, folds=None, cache=cache)

    assert result == converted / "Dataset001" / "Trainer"
    assert neighbor.read_text(encoding="utf-8") == "keep"
