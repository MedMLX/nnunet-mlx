"""The qualification preflight must report missing dependencies without importing them."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.host


def test_preflight_reports_missing_packages_without_site_packages(tmp_path: Path) -> None:
    """A fresh interpreter reports all missing prerequisites together as JSON."""
    script = tmp_path / "scripts" / "preflight.py"
    script.parent.mkdir()
    shutil.copyfile(Path(__file__).resolve().parents[2] / "scripts/preflight.py", script)
    completed = subprocess.run(
        [sys.executable, "-S", str(script)], capture_output=True, text=True, timeout=10,
    )
    assert completed.returncode == 1, completed.stderr
    assert completed.stdout, completed.stderr
    report = json.loads(completed.stdout)
    assert report["ready"] is False
    missing = {item["dependency"] for item in report["checks"] if not item["ready"]
               and "dependency" in item}
    assert {"mlx", "numpy", "nnunetv2", "Metal on Apple silicon"} <= missing
    assert any("asset" in item and not item["ready"] for item in report["checks"])
