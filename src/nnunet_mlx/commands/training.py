"""CLI entry points for preparing cases and training on Metal."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Literal, cast

import typer

from nnunet_mlx.commands.parsing import parse_fold
from nnunet_mlx.errors import ContractError


def prepare_training_command(
    dataset: Annotated[
        Path, typer.Argument(exists=True, file_okay=False, help="Raw dataset directory.")
    ],
    output: Annotated[Path, typer.Argument(help="New prepared dataset directory.")],
    plans: Annotated[
        Path | None, typer.Option(exists=True, dir_okay=False, help="Use an existing plans file.")
    ] = None,
    configuration: Annotated[
        str | None, typer.Option(help="Plans configuration, such as 2d or 3d_fullres.")
    ] = None,
    patch_size: Annotated[
        str | None, typer.Option(help="Patch in plans axis order, e.g. 64x64x64.")
    ] = None,
    splits: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, help="Case splits in nnU-Net JSON form."),
    ] = None,
    fold: Annotated[
        str, typer.Option(help="Validation fold index, or 'all' to use every case.")
    ] = "0",
) -> None:
    """Prepare training cases from dataset.json, imagesTr and labelsTr."""
    from nnunet_mlx.training.data import prepare_training_data

    prepared = prepare_training_data(
        dataset,
        output,
        plans_file=plans,
        configuration=configuration,
        patch_size=None if patch_size is None else _parse_patch_size(patch_size),
        splits_file=splits,
        fold=parse_fold(fold),
    )
    typer.echo(
        f"Prepared {len(prepared.split.train)} training and {len(prepared.split.val)} "
        f"validation cases for {prepared.plans.configuration.name} in {output}."
    )


def train_command(
    prepared: Annotated[
        Path, typer.Argument(exists=True, file_okay=False, help="Prepared dataset directory.")
    ],
    output: Annotated[Path, typer.Argument(help="Output model directory.")],
    epochs: Annotated[int | None, typer.Option(min=1, help="Total epochs. Default is 100.")] = None,
    steps_per_epoch: Annotated[
        int | None, typer.Option(min=1, help="Batches per epoch. Default is 250.")
    ] = None,
    batch_size: Annotated[
        int | None, typer.Option(min=1, help="Patches per batch. Defaults to the plans value.")
    ] = None,
    learning_rate: Annotated[
        float | None, typer.Option(help="AdamW learning rate. Default is 0.001.")
    ] = None,
    validate_every: Annotated[
        int | None, typer.Option(min=1, help="Validate every N epochs and at the final epoch.")
    ] = None,
    seed: Annotated[int | None, typer.Option(min=0, help="Initialization and patch seed.")] = None,
    precision: Annotated[
        Literal["float32", "bfloat16"] | None,
        typer.Option(help="Forward compute precision. Parameters and optimizer use float32."),
    ] = None,
    recompute: Annotated[
        bool | None, typer.Option("--recompute/--no-recompute", help="Recompute stage activations.")
    ] = None,
    augment: Annotated[
        bool | None,
        typer.Option("--augment/--no-augment", help="Mirror and jitter training patches."),
    ] = None,
    compile_step: Annotated[
        bool | None, typer.Option("--compile/--eager", help="Compile the training step.")
    ] = None,
    resume: Annotated[bool, typer.Option(help="Resume from the last completed epoch.")] = False,
    initialize_from: Annotated[
        Path | None,
        typer.Option(
            exists=True, file_okay=False, help="Initialize from a compatible MLX checkpoint."
        ),
    ] = None,
) -> None:
    """Train a 2-D or 3-D segmentation network natively in MLX."""
    from nnunet_mlx.runtime import runtime_info

    runtime_info()

    from nnunet_mlx.training.runner import TrainingOverrides, train_model

    overrides = cast(
        TrainingOverrides,
        {
            key: value
            for key, value in {
                "batch_size": batch_size,
                "steps_per_epoch": steps_per_epoch,
                "validate_every": validate_every,
                "seed": seed,
                "learning_rate": learning_rate,
                "precision": precision,
                "recompute": recompute,
                "augment": augment,
                "compile_step": compile_step,
            }.items()
            if value is not None
        },
    )
    checkpoint = train_model(
        prepared,
        output,
        epochs=epochs,
        options=overrides,
        resume=resume,
        initialize_from=initialize_from,
        progress=_echo_epoch,
    )
    typer.echo(f"Saved prediction checkpoint at {checkpoint}.")


def _parse_patch_size(value: str) -> tuple[int, ...]:
    try:
        patch = tuple(int(size) for size in value.lower().split("x"))
    except ValueError as error:
        raise ContractError("patch size must be written as 64x64 or 64x64x64") from error
    if len(patch) not in (2, 3) or min(patch) < 2:
        raise ContractError("patch size needs two or three dimensions of at least two voxels")
    return patch


def _echo_epoch(metrics: Mapping[str, object]) -> None:
    loss = cast(float, metrics["loss"])
    seconds = cast(float, metrics["training_seconds"])
    dice = metrics.get("validation_dice")
    validation = f", validation Dice {dice:.4f}" if isinstance(dice, float) else ""
    typer.echo(f"Epoch {metrics['epoch']}  loss {loss:.4f}{validation}  ({seconds:.1f} s training)")
