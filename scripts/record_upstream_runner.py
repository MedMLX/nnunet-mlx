"""Record pinned nnU-Net runner outputs for an HCC-TACE CT crop."""

from __future__ import annotations

import json
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from nnunetv2.inference.export_prediction import (
    convert_predicted_logits_to_segmentation_with_correct_shape,
)
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from nnunet_reference.oracle.network import real_image_input  # noqa: E402
from nnunet_reference.runner_network import build_network  # noqa: E402


def main() -> None:
    """Run source preprocessing, seeded folds and restoration, saving locally."""
    for package, pinned in (
        ("nnunetv2", "2.8.1"),
        ("dynamic-network-architectures", "0.4.4"),
        ("torch", "2.13.0"),
    ):
        if version(package) != pinned:
            raise ValueError(f"Expected {package} {pinned}, got {version(package)}")
    torch.set_num_threads(1)
    source = ROOT / "tests/fixtures/darwin-arm64/upstream_runner.npz"
    with np.load(source, allow_pickle=False) as fixture:
        arrays = {key: fixture[key] for key in fixture.files}
    metadata = json.loads(arrays["metadata"].item())
    runner = metadata["runner"]
    image = real_image_input(tuple(runner["image_shape"]), 1)[0]
    affine = np.asarray(runner["image_affine"], dtype=np.float64)
    plans = PlansManager(runner["plans"])
    config = plans.get_configuration("3d_fullres")
    dataset = runner["dataset_json"]
    labels = plans.get_label_manager(dataset)
    reader = plans.image_reader_writer_class()
    with tempfile.TemporaryDirectory(prefix="nnunet-real-") as temp:
        path = Path(temp) / "case_0000.nii.gz"
        nib.save(nib.Nifti1Image(image, affine), path)
        raw, image_properties = reader.read_images([str(path)])
        processed, _, properties = DefaultPreprocessor(verbose=False).run_case_npy(
            raw.copy(), None, dict(image_properties), plans, config, dataset
        )
    arrays.update(image=image, affine=affine, raw_image=raw, preprocessed=processed)
    metadata["image_properties"] = image_properties
    metadata["preprocessed_properties"] = properties
    metadata["runner"].pop("image_seed", None)
    metadata["runner"]["image_source"] = "HCC-TACE-Seg HCC_063 contrast CT"
    models = [build_network(runner, seed) for seed in runner["weight_seeds"]]
    padded = np.pad(processed, ((0, 0), *((0, max(0, 8 - size)) for size in processed.shape[1:])))
    patch = np.asarray(padded[:, :8, :8, :8], dtype=np.float32)
    with torch.inference_mode():
        for index, model in enumerate(models):
            arrays[f"network_input_{index}"] = patch
            arrays[f"network_logits_{index}"] = model(torch.from_numpy(patch[None])).numpy()[0]
        for options in runner["options"]:
            folds = []
            for index in options["folds"]:
                predictor = nnUNetPredictor(
                    tile_step_size=options["step_fraction"],
                    use_gaussian=True,
                    use_mirroring=bool(options["mirroring_axes"]),
                    perform_everything_on_device=False,
                    device=torch.device("cpu"),
                    verbose=False,
                    verbose_preprocessing=False,
                    allow_tqdm=False,
                )
                predictor.manual_initialization(
                    models[index],
                    plans,
                    config,
                    None,
                    dataset,
                    "nnUNetTrainer",
                    options["mirroring_axes"],
                )
                value = predictor.predict_sliding_window_return_logits(torch.from_numpy(processed))
                folds.append(value)
                if options["name"] == "default":
                    arrays[f"fold_logits_{index}"] = value.numpy().copy()
            logits = folds[0].clone()
            for value in folds[1:]:
                logits += value
            if len(folds) > 1:
                logits /= len(folds)
            name = options["name"]
            arrays[f"logits_{name}"] = logits.numpy()
            segmentation, probabilities = (
                convert_predicted_logits_to_segmentation_with_correct_shape(
                    logits, plans, config, labels, properties, return_probabilities=True
                )
            )
            arrays[f"segmentation_{name}"] = segmentation.transpose(2, 1, 0)
            arrays[f"probabilities_{name}"] = probabilities
            arrays[f"affine_{name}"] = affine
    arrays["metadata"] = np.asarray(json.dumps(metadata))
    destination = ROOT / "work/real-reference/upstream_runner.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)


if __name__ == "__main__":
    main()
