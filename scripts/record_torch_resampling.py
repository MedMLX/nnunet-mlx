"""Record HCC-TACE preprocessing with the selected upstream nnU-Net source."""

import argparse
import copy
import json
import subprocess
import sys
from importlib import import_module
from importlib.metadata import version
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REVISION = "17cb295b6258c660b7c6ea84f3bcb67f1c7a1e3f"


def main() -> None:
    """Replace image operands and run the original resampler and preprocessor.

    Segmentation, logits and precision-edge operands retain their existing
    controlled values. The raw-case crop retains its explicit zero border.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    options = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(options.source), "rev-parse", "HEAD"], text=True,
    ).strip()
    if revision != REVISION:
        raise ValueError(f"Expected upstream nnU-Net {REVISION}, got {revision}")
    if subprocess.check_output(
        ["git", "-C", str(options.source), "status", "--porcelain"], text=True,
    ).strip():
        raise ValueError("Upstream nnU-Net checkout must be clean")
    for package, pinned in (("torch", "2.13.0"), ("numpy", "2.5.4")):
        if version(package) != pinned:
            raise ValueError(f"Expected {package} {pinned}, got {version(package)}")
    sys.path[:0] = [str(options.source), str(ROOT / "tests")]
    real_image_input = import_module("nnunet_reference.oracle.network").real_image_input
    fixture_plans, _ = import_module("nnunet_reference.oracle.fixtures").raw_inference_fixture()
    resampler = import_module("nnunetv2.preprocessing.resampling.resample_torch")
    plans_module = import_module("nnunetv2.utilities.plans_handling.plans_handler")
    preprocessor = import_module("nnunetv2.preprocessing.preprocessors.default_preprocessor")
    exporter = import_module("nnunetv2.inference.export_prediction")
    fixture = ROOT / "tests/fixtures/darwin-arm64/torch_resampling.npz"
    with np.load(fixture, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    metadata = json.loads(str(arrays["metadata"].item()))
    for case in metadata["cases"]:
        if case["kwargs"]["is_seg"]:
            continue
        name = case["name"]
        shape = case["input_shape"]
        values = real_image_input(tuple(shape[1:]), shape[0])
        arrays[f"{name}-input"] = values
        arrays[f"{name}-expected"] = resampler.resample_torch_fornnunet(
            values, case["shape"], case["spacing"], case["new_spacing"],
            **case["kwargs"],
        )
    occupied = arrays["raw_nonzero"]
    raw = np.where(
        occupied, real_image_input(tuple(occupied.shape[1:]), occupied.shape[0]), 0,
    )
    arrays["raw"] = raw.astype(np.float32)
    for case in metadata["pipeline"]:
        name = case["name"]
        payload = copy.deepcopy(case["plans"])
        # The source manager requires a network description even for a host-only
        # preprocessing call. Reuse the maintained fixture architecture; no
        # network is constructed or executed by this recorder.
        payload["configurations"]["selected"]["architecture"] = (
            fixture_plans["configurations"]["base"]["architecture"]
        )
        plans = plans_module.PlansManager(payload)
        configuration = plans.get_configuration("selected")
        labels = plans.get_label_manager(case["dataset"])
        processed, segmentation, properties = preprocessor.DefaultPreprocessor(
            verbose=False,
        ).run_case_npy(
            raw.copy(), arrays["previous"].copy(), dict(case["properties"]),
            plans, configuration, case["dataset"],
        )
        arrays[f"{name}-processed"] = processed
        arrays[f"{name}-labels"] = segmentation
        case["geometry"] = {field: properties[field] for field in case["geometry"]}
        logits = arrays[f"{name}-logits"]
        result, probabilities = (
            exporter.convert_predicted_logits_to_segmentation_with_correct_shape(
                logits.copy(), plans, configuration, labels, properties, return_probabilities=True,
            )
        )
        arrays[f"{name}-segmentation"] = result
        arrays[f"{name}-probabilities"] = probabilities
        arrays[f"{name}-label-only"] = (
            exporter.convert_predicted_logits_to_segmentation_with_correct_shape(
                logits.copy(), plans, configuration, labels, properties, return_probabilities=False,
            )
        )
    metadata["image_source"] = "HCC-TACE-Seg HCC_063 contrast CT / 1024 HU"
    arrays["metadata"] = np.asarray(json.dumps(metadata))
    destination = ROOT / "work/real-reference/torch_resampling.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    print(f"Recorded {destination.name} from upstream {REVISION}")


if __name__ == "__main__":
    main()
