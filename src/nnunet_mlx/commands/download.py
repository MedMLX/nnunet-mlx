"""Download named model bundles through the shared MedMLX cache."""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Annotated

import typer
from medmlx_core.errors import MissingDependencyError
from medmlx_core.weights import weights_dir, weights_revision

from nnunet_mlx.errors import ContractError


class DownloadModel(StrEnum):
    """This runtime's named models with automatic weight downloads."""

    LIVER = "nnunet-liver"
    AUTOPET = "nnunet-autopet"
    PANCREAS = "nnunet-pancreas"
    BRATS = "nnunet-brats2021"


def download_command(
    model_id: Annotated[DownloadModel, typer.Argument(help="Published MedMLX model ID.")],
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit stable JSON on stdout.")
    ] = False,
) -> None:
    """Download or reuse a named model bundle and print its model-folder path."""
    try:
        folder = weights_dir(
            model_id.value, progress=lambda message: typer.echo(message, err=True),
        )
    except (MissingDependencyError, OSError) as error:
        raise ContractError(str(error)) from error
    if json_output:
        typer.echo(json.dumps({
            "model_id": model_id.value,
            "weights_path": str(folder),
            "weights_revision": weights_revision(folder),
        }, sort_keys=True))
        return
    typer.echo(str(folder))
