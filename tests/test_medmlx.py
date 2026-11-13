"""MedMLX runs on converted model folders, end to end and against upstream recordings."""

from __future__ import annotations

import json
import pickle
from collections.abc import Sequence
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from medmlx_core.errors import InvalidInputError
from nnunet_reference.fixture_types import RunnerFixture, UpstreamRecording
from nnunet_reference.oracle.network import real_image_input
from nnunet_reference.runner_network import build_network

from nnunet_mlx import medmlx, named_models
from nnunet_mlx.model.factory import build_model_from_plans
from nnunet_mlx.model.folder import load_model_bundle
from nnunet_mlx.plans import PlansContract
from nnunet_mlx.prediction.inference import ArrayPredictor
from nnunet_mlx.prediction.pipeline import ModelFolderPredictor
from nnunet_mlx.prediction.source import DEFAULT_INFERENCE_SOURCE_PROFILE
from nnunet_mlx.runtime import import_mlx
from nnunet_mlx.storage.checkpoint import (
    CheckpointIdentity,
    InferenceCheckpointState,
    save_inference_checkpoint,
)
from nnunet_mlx.storage.conversion import CONVERSION_FILENAME, CONVERSION_SCHEMA
from nnunet_mlx.storage.weights import (
    ConvertedState,
    HostArray,
    load_converted_state,
    map_key_and_layout,
)

nib = import_module("nibabel")
PLAIN = "dynamic_network_architectures.architectures.unet.PlainConvUNet"
INPUT_AFFINE = np.array(
    [[-1.0, 0.0, 0.0, 12.0], [0.0, -1.0, 0.0, 8.0], [0.0, 0.0, 1.0, -3.0], [0, 0, 0, 1]]
)


def _converted_folder(
    root: Path,
    plans: dict[str, object],
    dataset_json: dict[str, object],
    *,
    recording: RunnerFixture | None = None,
    trainer_name: str = "nnUNetTrainer",
    source_profile: str = DEFAULT_INFERENCE_SOURCE_PROFILE,
) -> Path:
    """Write a model folder with seeded MLX or regenerated upstream weights."""

    folder = root / "converted"
    folder.mkdir()
    (folder / "plans.json").write_text(json.dumps(plans), encoding="utf-8")
    (folder / "dataset.json").write_text(json.dumps(dataset_json), encoding="utf-8")
    topology = PlansContract.from_payload(plans, configuration="3d_fullres").topology_payload()
    mx, _ = import_mlx()
    folds: list[dict[str, object]] = []
    for fold in (0, 1):
        mx.random.seed(910 + fold)
        model = build_model_from_plans(
            topology,
            configuration="3d_fullres",
            input_channels=1,
            output_channels=2,
        )
        if recording is not None:
            load_converted_state(model, _recorded_state(recording, fold))
        mx.eval(model.parameters())
        checkpoint = folder / f"fold_{fold}" / "checkpoint_final"
        save_inference_checkpoint(
            checkpoint,
            model=model,
            state=InferenceCheckpointState(
                identity=CheckpointIdentity(
                    trainer_name=trainer_name,
                    configuration="3d_fullres",
                    fold=fold,
                ),
                current_epoch=None,
                spatial_dims=3,
                inference_allowed_mirroring_axes=(0, 1, 2),
                dropped_source_keys=(),
            ),
        )
        folds.append(
            {
                "fold": fold,
                "source_checkpoint": "checkpoint_final.pth",
                "checkpoint": "checkpoint_final",
                "mirroring_axes": [0, 1, 2],
            }
        )
    manifest = {
        "schema_version": CONVERSION_SCHEMA,
        "engine_version": version("nnunet-mlx"),
        "configuration": "3d_fullres",
        "source_profile": source_profile,
        "architecture": PLAIN,
        "spatial_dims": 3,
        "folds": folds,
    }
    (folder / CONVERSION_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")
    return folder


def _recorded_state(runner: RunnerFixture, fold: int) -> ConvertedState:
    """Recreate the pinned upstream network and map its tensors to native layout."""
    pytest.importorskip("torch")
    pytest.importorskip("dynamic_network_architectures")
    weights: dict[str, HostArray] = {}
    dropped: list[str] = []
    network = build_network(runner, runner["weight_seeds"][fold])
    for name, parameter in network.state_dict().items():
        mapped = map_key_and_layout(name, cast(HostArray, parameter.numpy()))
        if mapped is None:
            assert name.startswith("decoder.encoder.") or "all_modules" in name.split(".")
            dropped.append(name)
            continue
        target, values = mapped
        assert target not in weights
        weights[target] = values
    assert weights
    return ConvertedState(weights, tuple(sorted(dropped)))


@pytest.fixture
def case(
    tmp_path: Path,
    tiny_model_plans: dict[str, object],
    tiny_model_dataset: dict[str, object],
) -> tuple[Path, Path]:
    """Create a seeded model folder and HCC-TACE input case for runner tests."""
    folder = _converted_folder(tmp_path, tiny_model_plans, tiny_model_dataset)
    data = real_image_input((10, 9, 7), 1)[0]
    source = tmp_path / "case_0000.nii.gz"
    nib.save(nib.Nifti1Image(data, INPUT_AFFINE), source)
    return folder, source


def test_liver_runner_writes_label_map_on_the_input_grid(
    tmp_path: Path,
    case: tuple[Path, Path],
) -> None:
    """Run the registered liver runner end to end and write labels on the input grid."""
    folder, source = case
    output = tmp_path / "out"
    result = named_models.liver(
        image_path=str(source), output_dir=str(output), weights_path=str(folder)
    )

    mask_path = (output / "mask.nii.gz").resolve()
    assert result["outputs"] == {"mask": mask_path.as_posix()}
    assert sorted(path.name for path in output.iterdir()) == ["mask.nii.gz"]
    mask = nib.load(mask_path)
    labels = np.asarray(mask.dataobj)
    assert labels.shape == (10, 9, 7)
    assert set(np.unique(labels).tolist()) <= {0, 1}
    assert np.allclose(mask.affine, INPUT_AFFINE, rtol=0, atol=1e-6)


@pytest.mark.metal
def test_prepared_liver_cohort_matches_isolated_cases_and_bounds_model_memory(
    case: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reuse loaded weights while keeping per-case geometry, errors and memory bounded."""
    import mlx.core as mx

    from nnunet_mlx.model import folder as model_folder

    folder, source = case
    isolated = named_models.liver(
        image_path=str(source), output_dir=str(tmp_path / "isolated"), weights_path=str(folder))
    expected = nib.load(isolated["outputs"]["mask"])
    loads = 0
    original_load = model_folder.load_model_bundle

    def counted_load(
        model_folder: str | Path, *, configuration: str, folds: Sequence[int | str] | None = None,
        checkpoint_name: str = "checkpoint_final",
        source_profile: str = DEFAULT_INFERENCE_SOURCE_PROFILE, arithmetic: str = "native",
    ) -> model_folder.ModelBundle:
        nonlocal loads
        loads += 1
        return original_load(model_folder, configuration=configuration, folds=folds,
                             checkpoint_name=checkpoint_name, source_profile=source_profile,
                             arithmetic=arithmetic)

    monkeypatch.setattr(model_folder, "load_model_bundle", counted_load)
    prepared = named_models.prepare_liver(weights_path=str(folder))
    active: list[int] = []
    try:
        for index in range(24):
            result = prepared.run(image_path=str(source), output_dir=str(tmp_path / f"case{index}"))
            actual = nib.load(result["outputs"]["mask"])
            np.testing.assert_array_equal(np.asarray(actual.dataobj), np.asarray(expected.dataobj))
            np.testing.assert_array_equal(actual.affine, expected.affine)
            active.append(mx.get_active_memory())
        with pytest.raises(InvalidInputError):
            prepared.run(image_path=str(source), output_dir=str(tmp_path / "case0"))
        with pytest.raises(InvalidInputError):
            prepared.run(image_path=str(tmp_path / "missing.nii.gz"),
                         output_dir=str(tmp_path / "missing"))
        alias = tmp_path / "alias" / "mask.nii.gz"
        alias.parent.mkdir()
        source_before = source.read_bytes()
        alias.write_bytes(source_before)
        with pytest.raises(InvalidInputError):
            prepared.run(image_path=str(alias), output_dir=str(alias.parent), overwrite=True)
        assert alias.read_bytes() == source_before
        assert loads == 1
        # After warm-up the live MLX allocation must plateau rather than retain
        # one input/prediction tree per case. Free allocator cache is separate.
        assert max(active[4:]) <= min(active[4:]) + 64 * 1024
    finally:
        prepared.close()
    with pytest.raises(RuntimeError, match="closed"):
        prepared.run(image_path=str(source), output_dir=str(tmp_path / "closed"))


@pytest.mark.metal
def test_prepared_configuration_and_folder_stay_fixed_after_caller_changes(
    case: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A relative preparation path and attached attributes cannot redirect admitted state."""
    folder, source = case
    monkeypatch.chdir(folder.parent)
    prepared = medmlx.prepare(weights_path=folder.name, folds=[0], use_mirroring=False)
    for name, value in (("configuration", "different"), ("step_fraction", 0.9)):
        setattr(prepared, name, value)
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    try:
        result = prepared.run(image_paths=[str(source)], output_dir=str(tmp_path / "fixed"))
        assert result["details"]["configuration"] == "3d_fullres"
        assert result["details"]["step_fraction"] == 0.5
        assert result["details"]["folds"] == [0]
        assert result["details"]["mirroring_axes"] == []
    finally:
        prepared.close()


@pytest.mark.metal
@pytest.mark.parametrize("changed", ["plans.json", "fold_0/checkpoint_final/weights.safetensors"])
def test_prepared_model_rejects_source_changes_and_new_preparation_reloads(
    case: tuple[Path, Path], tmp_path: Path, changed: str,
) -> None:
    """An ordinary metadata/weight edit cannot silently reuse a stale loaded model."""
    from medmlx_core.errors import AssetNotReadyError

    folder, source = case
    prepared = medmlx.prepare(weights_path=str(folder))
    try:
        target = folder / changed
        target.touch()
        with pytest.raises(AssetNotReadyError, match="changed"):
            prepared.run(image_paths=[str(source)], output_dir=str(tmp_path / "stale"))
    finally:
        prepared.close()
    replacement = medmlx.prepare(weights_path=str(folder))
    try:
        result = replacement.run(image_paths=[str(source)], output_dir=str(tmp_path / "fresh"))
        assert Path(result["outputs"]["mask"]).is_file()
    finally:
        replacement.close()


@pytest.mark.metal
@pytest.mark.parametrize("reader", ["SimpleITKIO", "NibabelIO", "NibabelIOWithReorient"])
def test_in_memory_probabilities_match_file_reader_grid_crop_and_geometry(
    case: tuple[Path, Path], tmp_path: Path, reader: str,
) -> None:
    """Preserve the existing probability export contract without a temporary input file."""
    from nnunet_mlx.prediction.pipeline import ModelFolderPredictor
    from nnunet_mlx.storage.imageio import require_matching_geometry

    folder, source = case
    plans_path = folder / "plans.json"
    plans = json.loads(plans_path.read_text())
    plans["image_reader_writer"] = reader
    plans_path.write_text(json.dumps(plans))
    data = np.zeros((10, 9, 7), dtype=np.float32)
    data[1:-1, 2:-1, 1:-1] = real_image_input((8, 6, 5), 1)[0]
    affine = np.array([[0.9, -0.8, 0, 12], [1.2, 0.6, 0, 8], [0, 0, 2, -3], [0, 0, 0, 1]])
    image = nib.Nifti1Image(data, affine)
    nib.save(image, source)
    # Use the file's own NIfTI-1 header representation for the exact same
    # input grid; labels still restore to that image's full source geometry.
    image = nib.load(source)
    original = np.asarray(image.dataobj).copy()
    bundle = load_model_bundle(folder, configuration="3d_fullres", folds=(0, 1))
    predictor = ModelFolderPredictor(bundle, compile_models=False, accumulation="nnunet-fp16")
    baseline = predictor.predict_case([source], tmp_path / "baseline", save_probabilities=True,
                                      overwrite=False)
    actual = predictor.predict_nifti_probabilities(image)
    expected = nib.load(baseline.files.segmentation)
    np.testing.assert_array_equal(actual.segmentation.dataobj, expected.dataobj)
    assert actual.segmentation.affine is not None
    np.testing.assert_allclose(actual.segmentation.affine, expected.affine, rtol=0, atol=1e-5)
    assert baseline.files.probabilities is not None and baseline.files.properties is not None
    with np.load(baseline.files.probabilities) as arrays:
        np.testing.assert_array_equal(actual.probabilities, arrays["probabilities"])
    with baseline.files.properties.open("rb") as handle:
        properties = cast(dict[str, object], pickle.load(handle))
    require_matching_geometry(reader, actual.properties, properties)
    for field in ("shape_before_cropping", "bbox_used_for_cropping",
                  "shape_after_cropping_and_before_resampling", "labels", "regions_class_order"):
        assert actual.properties[field] == properties[field]
    assert actual.properties["inputs"] == []
    np.testing.assert_array_equal(np.asarray(image.dataobj), original)
    # Caller-owned result metadata must not mutate the reused model bundle.
    labels = cast(dict[str, object], actual.properties["labels"])
    labels["background"] = 999
    assert bundle.dataset_json["labels"] == properties["labels"]


@pytest.mark.metal
def test_in_memory_probability_geometry_rejects_ambiguous_forms(
    case: tuple[Path, Path], tmp_path: Path,
) -> None:
    """Refuse to invent a reader geometry when coded forms disagree."""
    from nnunet_mlx.errors import ContractError
    from nnunet_mlx.prediction.pipeline import ModelFolderPredictor

    folder, source = case
    predictor = ModelFolderPredictor(load_model_bundle(folder, configuration="3d_fullres"),
                                     compile_models=False)
    image = nib.load(source)
    other = np.array(image.affine, copy=True)
    other[0, 3] += 20
    image.set_qform(other, code=1, update_affine=False)
    with pytest.raises(ContractError, match="consistent"):
        predictor.predict_nifti_probabilities(image)


@pytest.mark.metal
@pytest.mark.parametrize("accumulation", ["float32", "nnunet-fp16"])
def test_file_and_nifti_labels_stream_without_a_full_logit_volume(
    case: tuple[Path, Path], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, accumulation: str,
) -> None:
    """Keep cropped/transposed/padded labels equal to explicit host-logit export."""
    folder, source = case
    data = np.zeros((10, 9, 7), dtype=np.float32)
    data[1:-1, 1:-1, 1:-1] = real_image_input((8, 7, 5), 1)[0]
    image = nib.Nifti1Image(data, INPUT_AFFINE)
    nib.save(image, source)
    bundle = load_model_bundle(folder, configuration="3d_fullres", folds=(0,))
    predictor = ModelFolderPredictor(bundle, compile_models=False, accumulation=accumulation)
    baseline = predictor.predict_case(
        (source,), tmp_path / "baseline", save_probabilities=False, overwrite=False,
        logits_file=tmp_path / "logits.npy",
    )
    expected = np.asarray(nib.load(baseline.files.segmentation).dataobj)

    def forbid_full_logits(*args: object, **kwargs: object) -> object:
        raise AssertionError("eligible labels must not allocate full host/device logits")

    monkeypatch.setattr(
        ArrayPredictor, "predict_logits_from_preprocessed_data", forbid_full_logits,
    )
    monkeypatch.setattr(
        ArrayPredictor, "predict_device_logits_from_preprocessed_data", forbid_full_logits,
    )
    result = predictor.predict_case(
        (source,), tmp_path / "streamed", save_probabilities=False, overwrite=False,
    )
    written = nib.load(result.files.segmentation)
    in_memory = predictor.predict_nifti(image)
    np.testing.assert_array_equal(np.asarray(written.dataobj), expected)
    np.testing.assert_array_equal(np.asarray(in_memory.dataobj), expected)
    np.testing.assert_array_equal(np.asarray(image.dataobj), data)
    np.testing.assert_allclose(written.affine, INPUT_AFFINE, rtol=0, atol=1e-6)


@pytest.fixture
def siam_model(
    tmp_path: Path,
    tiny_model_plans: dict[str, object],
    tiny_model_dataset: dict[str, object],
) -> Path:
    """Create a tiny native model with SIAM's source profile."""
    from nnunet_mlx.prediction.source import SIAM_SOURCE_PROFILE

    tiny_model_plans["dataset_name"] = "Dataset1007_Partialization_MixLowDill"
    return _converted_folder(tmp_path, tiny_model_plans, tiny_model_dataset,
                             trainer_name="nnUNetTrainerNoDA",
                             source_profile=SIAM_SOURCE_PROFILE)


def test_siam_runner_predicts_in_canonical_order_and_restores_the_input_grid(
    tmp_path: Path,
    siam_model: Path,
) -> None:
    """Predict a non-RAS input as SIAM does: on its RAS reorientation, mapped back exactly."""
    from nnunet_mlx.prediction.source import SIAM_SOURCE_PROFILE

    # Permuted and flipped axes (PIL), so the inverse reorientation is not the forward one.
    affine = np.array([[0.0, 0.0, -1.0, 4.0], [-1.0, 0.0, 0.0, 6.0],
                       [0.0, -1.0, 0.0, 2.0], [0, 0, 0, 1]])
    data = real_image_input((10, 9, 7), 1)[0]
    source = tmp_path / "case_0000.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), source)
    canonical = nib.as_closest_canonical(nib.load(source))
    canonical_path = tmp_path / "canonical.nii.gz"
    nib.save(canonical, canonical_path)

    result = named_models.siam(image_path=str(source), output_dir=str(tmp_path / "siam"),
                               weights_path=str(siam_model), fold=1)
    expected = medmlx.run(weights_path=str(siam_model), image_paths=[str(canonical_path)],
                          output_dir=str(tmp_path / "canonical"), folds=[1],
                          source_profile=SIAM_SOURCE_PROFILE, use_mirroring=False)

    mask = nib.load(result["outputs"]["mask"])
    assert sorted(path.name for path in (tmp_path / "siam").iterdir()) == ["mask.nii.gz"]
    assert mask.shape == (10, 9, 7)
    assert np.allclose(mask.affine, affine, rtol=0, atol=1e-6)
    assert result["details"]["folds"] == [1] and result["details"]["mirroring_axes"] == []
    restored = nib.as_closest_canonical(mask)
    assert np.array_equal(restored.dataobj, nib.load(expected["outputs"]["mask"]).dataobj)


@pytest.mark.parametrize("voxel_size", [None, 1.4], ids=["non-ras", "voxel-size"])
@pytest.mark.parametrize("alias", ["same-path", "symlink", "hardlink", "linked-directory"])
def test_siam_rejects_output_aliases_without_changing_input(
    tmp_path: Path, siam_model: Path, voxel_size: float | None, alias: str,
) -> None:
    """Keep the original scan byte-for-byte even when overwrite is requested."""
    source = tmp_path / "mask.nii.gz"
    affine = INPUT_AFFINE if voxel_size is None else np.eye(4)
    data = real_image_input((10, 9, 7), 1)[0]
    nib.save(nib.Nifti1Image(data, affine), source)
    original = source.read_bytes()
    output = tmp_path if alias == "same-path" else tmp_path / "out"
    if alias == "linked-directory":
        output.symlink_to(tmp_path, target_is_directory=True)
    elif alias != "same-path":
        output.mkdir()
        target = output / "mask.nii.gz"
        if alias == "symlink":
            target.symlink_to(source)
        else:
            target.hardlink_to(source)

    try:
        with pytest.raises(InvalidInputError, match="must differ from"):
            named_models.siam(
                image_path=str(source), output_dir=str(output), weights_path=str(siam_model),
                voxel_size=voxel_size, fold=1, overwrite=True,
            )
    finally:
        assert source.read_bytes() == original


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("existing", ["file", "symlink", "hardlink", "dangling-symlink"])
def test_siam_replaces_only_the_output_directory_entry(
    tmp_path: Path, siam_model: Path, overwrite: bool, existing: str,
) -> None:
    """Honor overwrite without following output links or modifying their referents."""
    source = tmp_path / "scan.nii.gz"
    data = real_image_input((10, 9, 7), 1)[0]
    nib.save(nib.Nifti1Image(data, INPUT_AFFINE), source)
    output = tmp_path / "out"
    output.mkdir()
    target = output / "mask.nii.gz"
    other = tmp_path / "other.nii.gz"
    previous = b"previous output"
    if existing == "file":
        target.write_bytes(previous)
    elif existing == "dangling-symlink":
        target.symlink_to(other)
    else:
        other.write_bytes(previous)
        if existing == "symlink":
            target.symlink_to(other)
        else:
            target.hardlink_to(other)

    if overwrite:
        result = named_models.siam(
            image_path=str(source), output_dir=str(output), weights_path=str(siam_model),
            fold=1, overwrite=True,
        )
        assert result["outputs"]["mask"] == str(target)
        assert not target.is_symlink()
        assert nib.load(target).shape == data.shape
    else:
        with pytest.raises(InvalidInputError, match="already exists"):
            named_models.siam(
                image_path=str(source), output_dir=str(output), weights_path=str(siam_model),
                fold=1,
            )
        if existing != "dangling-symlink":
            assert target.read_bytes() == previous
        assert target.is_symlink() == (existing in {"symlink", "dangling-symlink"})
    if existing in {"symlink", "hardlink"}:
        assert other.read_bytes() == previous
    else:
        assert not other.exists()
    assert sorted(path.name for path in output.iterdir()) == ["mask.nii.gz"]


def test_cli_predict_computes_what_the_runner_computes(
    tmp_path: Path,
    case: tuple[Path, Path],
) -> None:
    """Give the same probabilities and labels from ``model predict`` and the runner."""
    from nnunet_mlx.cli import main

    folder, source = case
    result = medmlx.run(weights_path=str(folder), image_paths=[str(source)],
                        output_dir=str(tmp_path / "runner"), save_probabilities=True)
    prefix = tmp_path / "cli" / "case"
    arguments = ["model", "predict", "--model-folder", str(folder), "--input", str(source),
                 "--output-prefix", str(prefix), "--save-probabilities"]
    assert main(arguments) == 0
    with np.load(result["outputs"]["probabilities"]) as runner, np.load(f"{prefix}.npz") as cli:
        assert np.array_equal(cli["probabilities"], runner["probabilities"])
    labels = nib.load(f"{prefix}.nii.gz").dataobj
    assert np.array_equal(labels, nib.load(result["outputs"]["mask"]).dataobj)

@pytest.mark.oracle
@pytest.mark.parametrize("name", ["default", "no_mirroring", "fold_1"])
def test_run_matches_recorded_upstream_segmentation(
    tmp_path: Path,
    upstream_runner_recording: UpstreamRecording,
    name: str,
) -> None:
    """Compare runner segmentation with the recorded upstream fixture."""
    metadata, arrays = upstream_runner_recording
    runner = metadata["runner"]
    folder = _converted_folder(
        tmp_path,
        runner["plans"],
        runner["dataset_json"],
        recording=runner,
    )
    source = tmp_path / "case_0000.nii.gz"
    nib.save(nib.Nifti1Image(arrays["image"], arrays["affine"]), source)
    options = next(item for item in runner["options"] if item["name"] == name)
    result = medmlx.run(
        weights_path=str(folder),
        image_paths=[str(source)],
        output_dir=str(tmp_path / "out"),
        folds=options["folds"],
        use_mirroring=bool(options["mirroring_axes"]),
        step_fraction=options["step_fraction"],
        save_probabilities=True,
    )
    outputs = result["outputs"]
    actual = nib.load(outputs["mask"])
    # Discrete labels and the exactly representable input affine have no tolerance.
    np.testing.assert_array_equal(np.asarray(actual.dataobj), arrays[f"segmentation_{name}"])
    np.testing.assert_array_equal(actual.affine, arrays[f"affine_{name}"])
    details = cast(dict[str, object], result["details"])
    assert details["arithmetic"] == "native"
    assert details["accumulation"] == "nnunet-fp16"
    assert details["folds"] == options["folds"]
    assert details["mirroring_axes"] == options["mirroring_axes"]


@pytest.mark.oracle
@pytest.mark.parametrize("fold", [0, 1])
def test_network_matches_recorded_upstream_logits(
    upstream_runner_recording: UpstreamRecording,
    fold: int,
) -> None:
    """Compare native network logits with the recorded upstream fixture."""
    metadata, arrays = upstream_runner_recording
    runner = metadata["runner"]
    plans = PlansContract.from_payload(runner["plans"], configuration="3d_fullres")
    model = build_model_from_plans(
        plans.topology_payload(),
        configuration="3d_fullres",
        input_channels=1,
        output_channels=2,
    )
    load_converted_state(model, _recorded_state(runner, fold))
    model.eval()
    mx, _ = import_mlx()
    patch = arrays[f"network_input_{fold}"]
    actual = model(mx.array(np.moveaxis(patch, 0, -1)[None]))
    mx.eval(actual)
    # Retain the existing upstream forward-parity bound: GPU reductions differ from Torch CPU.
    # Metal execution of this host's recording must be requalified by the reviewer.
    np.testing.assert_allclose(
        np.moveaxis(np.asarray(actual)[0], -1, 0),
        arrays[f"network_logits_{fold}"],
        rtol=1e-4,
        atol=1e-4,
    )
