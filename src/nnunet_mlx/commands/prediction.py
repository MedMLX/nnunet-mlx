"""Raw prediction, saved-probability ensembling, and postprocessing commands."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from nnunet_mlx.commands.parsing import parse_selected_folds
from nnunet_mlx.prediction.source import DEFAULT_INFERENCE_SOURCE_PROFILE

if TYPE_CHECKING:
    from nnunet_mlx.model.folder import ModelBundle

app = typer.Typer(help="Run MLX-native nnU-Net model folders.", no_args_is_help=True)


@app.command("predict")
def model_predict_command(
    model_folder: Annotated[
        Path,
        typer.Option(
            "--model-folder",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            help="MLX-native nnU-Net model folder.",
        ),
    ],
    image_files: Annotated[
        list[Path],
        typer.Option(
            "--input",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="One input image per modality, in channel order.",
        ),
    ],
    output_prefix: Annotated[
        Path,
        typer.Option("--output-prefix", help="Output path without the file ending."),
    ],
    previous_stage_file: Annotated[
        Path | None,
        typer.Option(
            "--previous-stage",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Previous-stage segmentation for cascade prediction.",
        ),
    ] = None,
    configuration: Annotated[
        str, typer.Option("--configuration", help="Plans configuration name.")
    ] = "3d_fullres",
    folds: Annotated[
        list[str] | None,
        typer.Option(
            "--fold",
            help="Fold number or 'all' for fold_all; repeat for an ensemble. "
            "Omit to detect numeric folds.",
        ),
    ] = None,
    checkpoint_name: Annotated[
        str, typer.Option("--checkpoint", help="MLX checkpoint directory name.")
    ] = "checkpoint_final",
    source_profile: Annotated[
        str,
        typer.Option(
            "--source-profile",
            help="Explicit source-bound model identity profile.",
        ),
    ] = DEFAULT_INFERENCE_SOURCE_PROFILE,
    arithmetic: Annotated[
        str,
        typer.Option(
            "--arithmetic",
            help="native, or torch-order-fp32: MLX on Metal in the pinned Torch rounding order.",
        ),
    ] = "native",
    save_probabilities: Annotated[
        bool, typer.Option("--save-probabilities", help="Write .npz and .pkl sidecars.")
    ] = False,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Replace existing prediction outputs.")
    ] = False,
    step_fraction: Annotated[
        float,
        typer.Option(
            "--step-fraction", min=0, max=1,
            help="Sliding-window step fraction. Zero is rejected during prediction.",
        ),
    ] = 0.5,
    disable_mirroring: Annotated[
        bool,
        typer.Option(
            "--disable-mirroring", help="Skip the checkpoint's mirroring test-time augmentation."
        ),
    ] = False,
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit stable JSON on stdout.")
    ] = False,
) -> None:
    """Predict one raw case from a validated MLX model folder, as the MedMLX runner does."""

    bundle = _load_selected_model_bundle(
        model_folder,
        configuration=configuration,
        folds=folds,
        checkpoint_name=checkpoint_name,
        source_profile=source_profile,
        arithmetic=arithmetic,
        disable_mirroring=disable_mirroring,
    )
    from nnunet_mlx.medmlx import predictor

    result = predictor(bundle, step_fraction=step_fraction).predict_case(
        image_files,
        output_prefix,
        previous_stage_file=previous_stage_file,
        save_probabilities=save_probabilities,
        overwrite=overwrite,
    )
    payload = result.to_payload()
    if json_output:
        typer.echo(json.dumps(payload, sort_keys=True))
        return
    typer.echo(f"segmentation: {result.files.segmentation}")


@app.command("predict-folder")
def model_predict_folder_command(
    model_folder: Annotated[
        Path,
        typer.Option(
            "--model-folder",
            exists=True,
            file_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    input_folder: Annotated[
        Path,
        typer.Option(
            "--input-folder",
            exists=True,
            file_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    output_folder: Annotated[Path, typer.Option("--output-folder")],
    previous_stage_folder: Annotated[
        Path | None,
        typer.Option(
            "--previous-stage-folder",
            exists=True,
            file_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    configuration: Annotated[str, typer.Option("--configuration")] = "3d_fullres",
    folds: Annotated[
        list[str] | None,
        typer.Option(
            "--fold",
            help="Fold number or 'all' for fold_all; repeat for an ensemble. "
            "Omit to detect numeric folds.",
        ),
    ] = None,
    checkpoint_name: Annotated[
        str, typer.Option("--checkpoint", help="MLX checkpoint directory name.")
    ] = "checkpoint_final",
    source_profile: Annotated[
        str,
        typer.Option(
            "--source-profile",
            help="Explicit source-bound model identity profile.",
        ),
    ] = DEFAULT_INFERENCE_SOURCE_PROFILE,
    arithmetic: Annotated[
        str,
        typer.Option(
            "--arithmetic",
            help="native, or torch-order-fp32: MLX on Metal in the pinned Torch rounding order.",
        ),
    ] = "native",
    save_probabilities: Annotated[
        bool, typer.Option("--save-probabilities", help="Write .npz and .pkl sidecars.")
    ] = False,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    continue_existing: Annotated[
        bool,
        typer.Option(
            "--continue-existing",
            help="Skip matching receipts and replace stale cases; file checks use metadata.",
        ),
    ] = False,
    num_parts: Annotated[int, typer.Option("--num-parts", min=1)] = 1,
    part_id: Annotated[int, typer.Option("--part-id", min=0)] = 0,
    preprocessing_processes: Annotated[
        int,
        typer.Option(
            "--preprocessing-processes",
            min=0,
            help="Spawned CPU workers; zero runs preprocessing inline.",
        ),
    ] = 0,
    export_processes: Annotated[
        int,
        typer.Option(
            "--export-processes",
            min=0,
            help="Spawned CPU/I/O export workers; restoration stays in the inference process.",
        ),
    ] = 0,
    max_inflight_bytes: Annotated[
        int, typer.Option("--max-inflight-bytes", min=1,
                          help="Shared prepared-input and queued-export payload budget in bytes.")
    ] = 2 * 1024**3,
    step_fraction: Annotated[
        float,
        typer.Option(
            "--step-fraction", min=0, max=1,
            help="Sliding-window step fraction. Zero is rejected during prediction.",
        ),
    ] = 0.5,
    disable_mirroring: Annotated[
        bool,
        typer.Option(
            "--disable-mirroring", help="Skip the checkpoint's mirroring test-time augmentation."
        ),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Predict split-dataset cases from a raw input folder, as the MedMLX runner does."""

    bundle = _load_selected_model_bundle(
        model_folder,
        configuration=configuration,
        folds=folds,
        checkpoint_name=checkpoint_name,
        source_profile=source_profile,
        arithmetic=arithmetic,
        disable_mirroring=disable_mirroring,
    )
    from nnunet_mlx.medmlx import predictor

    result = predictor(bundle, step_fraction=step_fraction).predict_folder(
        input_folder,
        output_folder,
        previous_stage_folder=previous_stage_folder,
        save_probabilities=save_probabilities,
        overwrite=overwrite,
        continue_existing=continue_existing,
        num_parts=num_parts,
        part_id=part_id,
        preprocessing_processes=preprocessing_processes,
        export_processes=export_processes,
        max_inflight_bytes=max_inflight_bytes,
    )
    payload = result.to_payload()
    if json_output:
        typer.echo(json.dumps(payload, sort_keys=True))
        return
    typer.echo(f"predicted: {len(result.predictions)}")
    typer.echo(f"skipped: {len(result.skipped)}")


@app.command("ensemble")
def model_ensemble_command(
    input_folders: Annotated[
        list[Path],
        typer.Option(
            "--input-folder", exists=True, file_okay=False, readable=True, resolve_path=True,
            help="Saved probability folder; repeat in the desired averaging order.",
        ),
    ],
    output_folder: Annotated[Path, typer.Option("--output-folder", file_okay=False)],
    plans_json: Annotated[
        Path,
        typer.Option(
            "--plans-json", exists=True, dir_okay=False, readable=True, resolve_path=True,
            help="plans.json naming the probability files' reader/writer.",
        ),
    ],
    dataset_json: Annotated[
        Path,
        typer.Option(
            "--dataset-json", exists=True, dir_okay=False, readable=True, resolve_path=True,
            help="dataset.json defining channel labels, region order, and output extension.",
        ),
    ],
    save_probabilities: Annotated[
        bool, typer.Option("--save-probabilities", help="Write merged .npz and .pkl sidecars.")
    ] = False,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Replace existing prediction outputs.")
    ] = False,
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit output-file records as JSON on stdout.")
    ] = False,
) -> None:
    """Average matching saved probability cases and export their segmentations."""
    from nnunet_mlx.data.labels import parse_dataset_label_semantics
    from nnunet_mlx.plans import read_json_object
    from nnunet_mlx.prediction.ensembling import ensemble_folders
    from nnunet_mlx.prediction.postprocessing import resolve_exported_segmentation_reader

    dataset = read_json_object(dataset_json)
    reader_name, file_ending = resolve_exported_segmentation_reader(
        plans_json=read_json_object(plans_json), dataset_json=dataset,
    )
    written = ensemble_folders(
        input_folders, output_folder, file_ending=file_ending, image_reader_writer=reader_name,
        semantics=parse_dataset_label_semantics(dataset), overwrite=overwrite,
        save_probabilities=save_probabilities,
    )
    if json_output:
        typer.echo(json.dumps({"files": [item.to_payload() for item in written]}, sort_keys=True))
        return
    typer.echo(f"ensembled: {len(written)}")


@app.command("apply-postprocessing")
def model_apply_postprocessing_command(
    input_folder: Annotated[
        Path,
        typer.Option(
            "--input-folder",
            exists=True,
            file_okay=False,
            readable=True,
            resolve_path=True,
            help="Folder of already-exported segmentations (from 'model predict-folder').",
        ),
    ],
    output_folder: Annotated[Path, typer.Option("--output-folder")],
    postprocessing_pkl: Annotated[
        Path,
        typer.Option(
            "--postprocessing-pkl",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="A released model's postprocessing.pkl.",
        ),
    ],
    plans_json: Annotated[
        Path,
        typer.Option(
            "--plans-json",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="plans.json naming the exported files' reader/writer.",
        ),
    ],
    dataset_json: Annotated[
        Path,
        typer.Option(
            "--dataset-json",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="dataset.json naming the exported files' extension.",
        ),
    ],
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Apply a released model's postprocessing.pkl to already-exported segmentations.

    The inference-time counterpart of upstream's ``nnUNetv2_apply_postprocessing``:
    a separate, manual step after ``model predict``/``model predict-folder``,
    never applied automatically by either.
    """

    from nnunet_mlx.plans import read_json_object
    from nnunet_mlx.prediction.postprocessing import (
        apply_postprocessing_to_folder,
        load_postprocessing_pkl,
        resolve_exported_segmentation_reader,
    )

    reader_name, file_ending = resolve_exported_segmentation_reader(
        plans_json=read_json_object(plans_json),
        dataset_json=read_json_object(dataset_json),
    )
    steps = load_postprocessing_pkl(postprocessing_pkl)
    written = apply_postprocessing_to_folder(
        reader_name,
        input_folder,
        output_folder,
        steps,
        file_ending=file_ending,
        overwrite=overwrite,
    )
    if json_output:
        typer.echo(json.dumps({"files": [str(path) for path in written]}, sort_keys=True))
        return
    typer.echo(f"postprocessed: {len(written)}")


def _load_selected_model_bundle(
    model_folder: Path,
    *,
    configuration: str,
    folds: Sequence[str] | None,
    checkpoint_name: str,
    source_profile: str,
    arithmetic: str,
    disable_mirroring: bool = False,
) -> ModelBundle:
    """Load the requested folds and optionally disable test-time mirroring."""
    selected_folds = parse_selected_folds(folds)

    from nnunet_mlx.model.folder import load_model_bundle

    bundle = load_model_bundle(
        model_folder,
        configuration=configuration,
        folds=selected_folds,
        checkpoint_name=checkpoint_name,
        source_profile=source_profile,
        arithmetic=arithmetic,
    )
    return replace(bundle, mirroring_axes=()) if disable_mirroring else bundle
