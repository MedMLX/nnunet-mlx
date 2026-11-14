"""Commands for trained upstream nnU-Net folders: conversion and nnUNetv2_predict-style runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from nnunet_mlx.commands.parsing import parse_selected_folds
from nnunet_mlx.errors import ContractError
from nnunet_mlx.prediction.source import DEFAULT_INFERENCE_SOURCE_PROFILE


def convert_command(
    source: Annotated[
        Path,
        typer.Argument(
            help="Trained nnU-Net folder: nnUNet_results/<dataset>/<trainer>__<plans>__<config>.",
            exists=True, file_okay=False, readable=True, resolve_path=True,
        ),
    ],
    destination: Annotated[
        Path, typer.Argument(help="New MLX model folder to create; must not exist.")
    ],
    folds: Annotated[
        list[str] | None,
        typer.Option("--fold", help="Fold number or 'all'; repeat for several. "
                     "Omit to convert every fold."),
    ] = None,
    checkpoint_name: Annotated[
        str, typer.Option("--checkpoint", help="Upstream checkpoint file in each fold folder.")
    ] = "checkpoint_final.pth",
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Convert a trained nnU-Net model folder (PyTorch checkpoints) to an MLX model folder."""
    from nnunet_mlx.storage.upstream import convert_trained_model

    result = convert_trained_model(
        source, destination, folds=parse_selected_folds(folds), checkpoint_name=checkpoint_name
    )
    payload = {
        "destination": str(result.destination.resolve()),
        "configuration": result.configuration,
        "trainer": result.trainer_name,
        "folds": list(result.folds),
    }
    if json_output:
        typer.echo(json.dumps(payload, sort_keys=True))
        return
    typer.echo(f"converted {len(result.folds)} fold(s) of {result.trainer_name} "
               f"{result.configuration} to {payload['destination']}")


def predict_command(
    input_folder: Annotated[
        Path,
        typer.Option("-i", help="Input folder with nnU-Net-named images (case_0000.nii.gz).",
                     exists=True, file_okay=False, readable=True, resolve_path=True),
    ],
    output_folder: Annotated[Path, typer.Option("-o", help="Output folder.")],
    dataset: Annotated[
        str | None,
        typer.Option("-d", help="Dataset ID or name under $nnUNet_results."),
    ] = None,
    configuration: Annotated[
        str | None, typer.Option("-c", help="Configuration, e.g. 3d_fullres; required with -d.")
    ] = None,
    plans: Annotated[str, typer.Option("-p", help="Plans identifier.")] = "nnUNetPlans",
    trainer: Annotated[str, typer.Option("-tr", help="Trainer class name.")] = "nnUNetTrainer",
    model_folder: Annotated[
        Path | None,
        typer.Option("-m", help="Trained model folder, instead of -d/-c/-tr/-p.",
                     exists=True, file_okay=False, readable=True, resolve_path=True),
    ] = None,
    first_fold: Annotated[
        str | None,
        typer.Option("-f", help="Folds to ensemble, e.g. -f 0 1 2 or -f all. "
                     "Default: 0 1 2 3 4 with -d, every fold with -m."),
    ] = None,
    more_folds: Annotated[
        list[str] | None, typer.Argument(hidden=True, show_default=False)
    ] = None,
    checkpoint_name: Annotated[
        str, typer.Option("-chk", help="Upstream checkpoint file in each fold folder.")
    ] = "checkpoint_final.pth",
    step_size: Annotated[
        float, typer.Option("-step_size", min=0, max=1, help="Sliding-window step fraction.")
    ] = 0.5,
    disable_tta: Annotated[
        bool, typer.Option("--disable_tta", help="Skip mirroring test-time augmentation.")
    ] = False,
    save_probabilities: Annotated[
        bool, typer.Option("--save_probabilities", help="Also write .npz and .pkl sidecars.")
    ] = False,
    continue_prediction: Annotated[
        bool, typer.Option("--continue_prediction", help="Skip cases already predicted.")
    ] = False,
    preprocessing_processes: Annotated[
        int, typer.Option("-npp", min=0, help="Preprocessing workers; 0 runs inline.")
    ] = 0,
    export_processes: Annotated[
        int, typer.Option("-nps", min=0, help="Segmentation export workers; 0 runs inline.")
    ] = 0,
    previous_stage: Annotated[
        Path | None,
        typer.Option("-prev_stage_predictions", help="Previous cascade stage's predictions.",
                     exists=True, file_okay=False, readable=True, resolve_path=True),
    ] = None,
    num_parts: Annotated[int, typer.Option("-num_parts", min=1)] = 1,
    part_id: Annotated[int, typer.Option("-part_id", min=0)] = 0,
    max_inflight_bytes: Annotated[
        int, typer.Option("--max-inflight-bytes", min=1,
                          help="Memory budget for prepared and queued case payloads, in bytes.")
    ] = 2 * 1024**3,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Predict a folder from a trained nnU-Net model, taking nnUNetv2_predict's options.

    The trained PyTorch model is converted to MLX on first use and cached in
    NNUNET_MLX_CACHE (default ~/.cache/nnunet-mlx); later runs reuse the conversion
    until checkpoint or JSON metadata changes. Existing outputs are replaced unless
    --continue_prediction is given.
    """
    from nnunet_mlx.storage.upstream import cached_conversion, resolve_results_folder

    if more_folds and first_fold is None:
        raise ContractError(f"unexpected arguments {more_folds}; list folds after -f")
    fold_values = None if first_fold is None else [first_fold, *(more_folds or [])]
    if model_folder is not None:
        if dataset is not None or configuration is not None:
            raise ContractError("pass either -m or -d with -c, not both")
        trained = model_folder
    else:
        if dataset is None or configuration is None:
            raise ContractError("pass -d and -c, or a trained model folder with -m")
        trained = resolve_results_folder(
            dataset, configuration=configuration, trainer=trainer, plans=plans
        )
        if fold_values is None:
            fold_values = ["0", "1", "2", "3", "4"]
    folds = parse_selected_folds(fold_values)
    converted = cached_conversion(trained, folds=folds, checkpoint_name=checkpoint_name)

    from dataclasses import replace

    from nnunet_mlx.medmlx import predictor
    from nnunet_mlx.model.folder import load_model_bundle
    from nnunet_mlx.storage.conversion import read_conversion_manifest

    # The checkpoints record their configuration; -m runs need no -c.
    manifest = read_conversion_manifest(converted)
    bundle = load_model_bundle(
        converted, configuration=manifest["configuration"],
        folds=tuple(entry["fold"] for entry in manifest["folds"]) if folds is None else folds,
        source_profile=DEFAULT_INFERENCE_SOURCE_PROFILE,
    )
    if disable_tta:
        bundle = replace(bundle, mirroring_axes=())
    result = predictor(bundle, step_fraction=step_size).predict_folder(
        input_folder,
        output_folder,
        previous_stage_folder=previous_stage,
        save_probabilities=save_probabilities,
        overwrite=not continue_prediction,
        continue_existing=continue_prediction,
        num_parts=num_parts,
        part_id=part_id,
        preprocessing_processes=preprocessing_processes,
        export_processes=export_processes,
        max_inflight_bytes=max_inflight_bytes,
    )
    if json_output:
        typer.echo(json.dumps(result.to_payload(), sort_keys=True))
        return
    typer.echo(f"predicted: {len(result.predictions)}")
    typer.echo(f"skipped: {len(result.skipped)}")
