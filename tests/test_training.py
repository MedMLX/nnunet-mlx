"""Native training through raw preparation, checkpoint resume and prediction."""

from __future__ import annotations

import json
import math
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from nibabel.nifti1 import Nifti1Image
from nnunet_reference.arrays import host_array

from nnunet_mlx.cli import main
from nnunet_mlx.contracts import parse_mapping, parse_number, parse_sequence
from nnunet_mlx.errors import ContractError
from nnunet_mlx.plans import read_json_object
from nnunet_mlx.runtime import import_mlx
from nnunet_mlx.training.data import PreparedDataset

if TYPE_CHECKING:
    from mlx.core import array


@pytest.mark.metal
def test_scheduled_training_resumes_and_keeps_best_validation_checkpoint(tmp_path: Path) -> None:
    from nnunet_mlx.training.runner import TrainingOverrides, train_model

    raw = tmp_path / "dataset"
    _write_dataset(raw)
    prepared, resumed = (tmp_path / name for name in ("prepared", "resumed"))
    assert main(["prepare-training", str(raw), str(prepared), "--patch-size", "16x16x16"]) == 0
    settings: TrainingOverrides = {
        "steps_per_epoch": 8,
        "validate_every": 1,
        "cosine_lr": True,
        "focus_label": 1,
        "spatial_augment": True,
        "recompute_stages": 1,
    }

    def interrupt(_: object) -> None:
        with pytest.raises(ContractError, match="a training process is already using"):
            train_model(prepared, resumed, resume=True)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        train_model(prepared, resumed, epochs=3, options=settings, progress=interrupt)
    mx, _ = import_mlx()
    fold = resumed / "fold_0"
    # Recover selection if the best alias was lost after the epoch was saved.
    (fold / "checkpoint_best").unlink()
    for completed in (8, 24):
        if completed == 24:
            train_model(prepared, resumed, resume=True)
        state = mx.load(fold / "training.safetensors", format="safetensors")
        # One warmup update, followed by 23 cosine updates on the original horizon.
        phase = (completed - 2) / 23
        expected_rate = 5e-5 + 0.5 * (1e-3 - 5e-5) * (1 + math.cos(math.pi * phase))
        np.testing.assert_array_equal(host_array(state["optimizer.step"]), completed)
        np.testing.assert_allclose(
            host_array(state["optimizer.learning_rate"]), expected_rate, rtol=1e-5
        )

    score_payload: object = json.loads((resumed / "metrics.json").read_text(encoding="utf-8"))
    history = [
        parse_mapping(record, name="epoch metrics")
        for record in parse_sequence(score_payload, name="history")
    ]
    best = max(history, key=lambda record: parse_number(record["validation_dice"], name="Dice"))
    selected = read_json_object(fold / "checkpoint_best/state.json")
    assert selected["current_epoch"] == best["epoch"]
    assert selected["inference_allowed_mirroring_axes"] == []


@pytest.mark.metal
def test_native_training_resume_and_nifti_prediction(tmp_path: Path) -> None:
    raw = tmp_path / "dataset"
    _write_dataset(raw)
    prepared = tmp_path / "prepared"
    run = tmp_path / "resumed"
    direct = tmp_path / "direct"
    assert main(["prepare-training", str(raw), str(prepared), "--patch-size", "16x16x16"]) == 0
    args = ["--steps-per-epoch", "8", "--no-augment", "--precision", "bfloat16", "--recompute"]
    assert main(["train", str(prepared), str(run), "--epochs", "1", *args]) == 0
    saved = parse_mapping(read_json_object(run / "training.json")["settings"], name="settings")
    assert saved["balanced_ce"] is True
    assert main(["train", str(prepared), str(run), "--resume", "--epochs", "2"]) == 0
    assert main(["train", str(prepared), str(direct), "--epochs", "2", *args]) == 0

    mx, _ = import_mlx()
    relative = "fold_0/checkpoint_final/weights.safetensors"
    first = mx.load(run / "fold_0/checkpoint_epoch_1/weights.safetensors", format="safetensors")
    resumed = mx.load(run / relative, format="safetensors")
    uninterrupted = mx.load(direct / relative, format="safetensors")
    # Weights must change, and restoring the optimizer and
    # epoch-local sample stream must agree with an uninterrupted run.
    assert any(
        not np.array_equal(host_array(first[key]), host_array(v)) for key, v in resumed.items()
    )
    for key, weights in resumed.items():
        assert weights.dtype == mx.float32
        np.testing.assert_allclose(
            host_array(weights),
            host_array(uninterrupted[key]),
            rtol=3e-3,
            atol=5e-6,
        )

    records: object = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    final = parse_mapping(parse_sequence(records, name="metrics")[-1], name="final metrics")
    assert final["epoch"] == 2
    assert parse_number(final.get("validation_dice"), name="Dice") > 0.8

    (case,) = PreparedDataset.load(prepared).split.val
    prediction = tmp_path / "prediction"
    assert (
        main(
            [
                "model",
                "predict",
                "--model-folder",
                str(run),
                "--disable-mirroring",
                "--input",
                str(raw / "imagesTr" / f"{case}_0000.nii.gz"),
                "--output-prefix",
                str(prediction),
            ]
        )
        == 0
    )
    actual = Nifti1Image.from_filename(str(prediction.with_suffix(".nii.gz")))
    expected = Nifti1Image.from_filename(str(raw / "labelsTr" / f"{case}.nii.gz"))
    assert actual.affine is not None and expected.affine is not None
    np.testing.assert_allclose(actual.affine, expected.affine)
    mask = np.asarray(actual.dataobj, dtype=np.uint8) == 1
    truth = np.asarray(expected.dataobj, dtype=np.uint8) == 1
    assert mask.shape == truth.shape
    assert 2 * np.count_nonzero(mask & truth) / (mask.sum() + truth.sum()) > 0.8


@pytest.mark.metal
def test_warm_start_checks_source_labels_and_topology(tmp_path: Path) -> None:
    from nnunet_mlx import TrainingOverrides, prepare_training_data, train_model

    raw, original = tmp_path / "raw", tmp_path / "original"
    _write_dataset(raw)
    prepared = prepare_training_data(str(raw), str(tmp_path / "prepared"), patch_size=(16, 16, 16))
    options: TrainingOverrides = {"steps_per_epoch": 1, "cosine_lr": True}
    checkpoint = train_model(prepared, original, epochs=1, options=options)
    mx, _ = import_mlx()
    source_weights = checkpoint / "weights.safetensors"
    weights = mx.load(source_weights, format="safetensors")
    half = tmp_path / "half.safetensors"
    mx.save_safetensors(half, {key: array.astype(mx.float16) for key, array in weights.items()})
    half.replace(source_weights)

    warm = tmp_path / "warm"
    train_model(prepared, warm, epochs=1, options=options, initialize_from=checkpoint)
    train_model(str(prepared.folder), str(warm), epochs=2, resume=True)
    arrays = mx.load(warm / "fold_0/training.safetensors", format="safetensors")
    assert arrays["optimizer.step"].item() == 2
    assert arrays["optimizer.learning_rate"].item() == pytest.approx(5e-5)

    dataset_path = original / "dataset.json"
    dataset = read_json_object(dataset_path)
    changed = dataset | {"labels": {"background": 0, "other structure": 1}}
    dataset_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ContractError, match="labels do not match"):
        train_model(prepared, tmp_path / "bad-labels", epochs=1, initialize_from=checkpoint)
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    plans_path = original / "plans.json"
    plans = read_json_object(plans_path)
    configs = parse_mapping(plans["configurations"], name="configurations")
    config = parse_mapping(configs["3d_fullres"], name="configuration")
    arch = parse_mapping(config["architecture"], name="architecture")
    kwargs = parse_mapping(arch["arch_kwargs"], name="architecture keyword arguments")
    kwargs["nonlin_kwargs"] = {"negative_slope": 0.2}
    arch["arch_kwargs"] = kwargs
    config["architecture"] = arch
    configs["3d_fullres"] = config
    plans["configurations"] = configs
    plans_path.write_text(json.dumps(plans), encoding="utf-8")
    with pytest.raises(ContractError, match="topology does not match"):
        train_model(prepared, tmp_path / "bad-topology", epochs=1, initialize_from=checkpoint)


@pytest.mark.metal
def test_dice_respects_patch_grouping() -> None:
    from nnunet_mlx.data.labels import parse_dataset_label_semantics
    from nnunet_mlx.training.step import segmentation_loss

    mx, _ = import_mlx()
    semantics = parse_dataset_label_semantics({"labels": {"background": 0, "target": 1}})
    logits = mx.log(mx.array([[[[0.9, 0.1], [0.5, 0.5]]], [[[0.6, 0.4], [0.8, 0.2]]]]))
    labels = mx.array([[[0, 1]], [[1, 1]]])
    ce = -math.log(0.9 * 0.5 * 0.4 * 0.2) / 4
    pooled = 1 - 2.2 / 4.2
    per_patch = 1 - (1.0 / 1.6 + 1.2 / 2.6) / 2
    assert segmentation_loss(semantics, batch_dice=True)(logits, labels).item() == pytest.approx(
        ce + pooled
    )
    assert segmentation_loss(semantics, batch_dice=False)(logits, labels).item() == pytest.approx(
        ce + per_patch
    )


@pytest.mark.metal
@pytest.mark.parametrize("regions", [False, True])
def test_segmentation_loss_masks_ignored_voxels(regions: bool) -> None:
    from mlx.core import grad

    from nnunet_mlx.data.labels import parse_dataset_label_semantics
    from nnunet_mlx.training.step import segmentation_loss

    mx, _ = import_mlx()
    metadata: dict[str, object] = {
        "labels": {"background": 0, "outer": [1, 2] if regions else 1, "core": 2, "ignore": 3},
    }
    if regions:
        metadata["regions_class_order"] = [1, 2]
    semantics = parse_dataset_label_semantics(metadata)
    loss = segmentation_loss(semantics)
    targets = mx.array([0, 1, 1, 2, -1], dtype=mx.int32)
    logits = mx.zeros((5, semantics.output_channels), dtype=mx.float32)

    def objective(scores: array) -> array:
        return loss(scores, targets)

    value = objective(logits)
    gradients = grad(objective)(logits)
    mx.eval(value, gradients)
    if regions:
        expected = math.log(2) + 1 - (3 / 5 + 1 / 3) / 2
    else:
        expected = math.log(3) + 1 - ((4 / 3) / (4 / 3 + 2) + (2 / 3) / (4 / 3 + 1)) / 2
    np.testing.assert_allclose(host_array(value), expected, rtol=1e-6)
    derivative = host_array(gradients)
    assert np.isfinite(derivative).all()
    assert np.any(derivative[:-1])
    np.testing.assert_array_equal(derivative[-1], np.zeros(semantics.output_channels))


def _write_dataset(folder: Path) -> None:
    (folder / "imagesTr").mkdir(parents=True)
    (folder / "labelsTr").mkdir()
    (folder / "dataset.json").write_text(
        json.dumps(
            {
                "channel_names": {"0": "MRI"},
                "labels": {"background": 0, "ellipsoid": 1},
                "file_ending": ".nii.gz",
            }
        ),
        encoding="utf-8",
    )
    shape = (18, 20, 20)
    x, y, z = np.indices(shape)
    affine = np.diag([2.0, 1.0, 1.0, 1.0])
    rng = np.random.default_rng(8)
    nibabel = import_module("nibabel")
    for index in range(4):
        label = (
            ((x - 8 - index % 2) / 3.5) ** 2 + ((y - 9) / 5) ** 2 + ((z - 9) / 4) ** 2 < 1
        ).astype(np.uint8)
        signal = (0.2 + rng.normal(0.0, 0.025, shape) + 1.5 * label).astype(np.float32)
        nibabel.save(Nifti1Image(signal, affine), folder / "imagesTr" / f"case{index}_0000.nii.gz")
        nibabel.save(Nifti1Image(label, affine), folder / "labelsTr" / f"case{index}.nii.gz")
