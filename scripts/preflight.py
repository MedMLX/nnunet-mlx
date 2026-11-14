"""Report all missing dependencies and sourced assets for local qualification."""

from __future__ import annotations

import json
import os
import platform
import zipfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """Check provisioned inputs together, without downloading or changing environments."""
    checks: list[dict[str, object]] = []
    mlx_ready = False
    for package, pinned in (
        ("mlx", "0.32.4"), ("numpy", "2.5.4"), ("scipy", "1.18.1"),
        ("scikit-image", "0.26.0"), ("nnunetv2", "2.8.1"),
        ("dynamic-network-architectures", "0.4.4"), ("torch", "2.13.0"),
    ):
        try:
            installed = version(package)
        except PackageNotFoundError:
            installed = None
        checks.append({"dependency": package, "expected_version": pinned,
                       "installed_version": installed, "ready": installed == pinned})
        if package == "mlx":
            mlx_ready = installed == pinned
    metal = False
    if mlx_ready and platform.system() == "Darwin" and platform.machine() == "arm64":
        import mlx.core as mx

        metal = mx.metal.is_available() and "gpu" in str(mx.default_device()).lower()
    checks.append({"dependency": "Metal on Apple silicon", "ready": metal})
    data = Path.home() / ".cache/medmlx/test-data"
    for path in (
        Path(os.environ.get("MEDMLX_TEST_CT", str(data / "hcc_063_ct.nii.gz"))),
        data / "hcc_063_liver.nii.gz", data / "hcc_063_tumor.nii.gz",
        Path(os.environ.get("MEDMLX_TEST_XRAY", str(data / "chest_xray.jpg"))),
    ):
        checks.append({"asset": str(path), "ready": path.is_file(),
                       "prepare": "Provision the sourced HCC_063 CT, liver/tumor masks and X-ray; "
                                  "see tests/README.md."})
    for name, keys, preparation in (
        ("upstream_runner", ("raw_image", "preprocessed", "logits_default"),
         "python scripts/record_upstream_runner.py"),
        ("torch_resampling", ("raw",),
         "python scripts/record_torch_resampling.py --source PATH_TO_PINNED_UPSTREAM"),
    ):
        path = ROOT / "work/real-reference" / f"{name}.npz"
        ready = path.is_file()
        if ready:
            # NPZ member names are ZIP entries; checking them needs no NumPy import.
            try:
                with zipfile.ZipFile(path) as saved:
                    ready = all(f"{key}.npy" in saved.namelist() for key in keys)
            except zipfile.BadZipFile:
                ready = False
        checks.append({"asset": str(path.relative_to(ROOT)), "ready": ready,
                       "prepare": preparation})
    ready = all(item["ready"] for item in checks)
    print(json.dumps({"ready": ready, "checks": checks}, indent=2))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
