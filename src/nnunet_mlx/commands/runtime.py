"""Runtime inspection command."""

from __future__ import annotations

import json
from typing import Annotated

import typer

app = typer.Typer(help="Inspect the active MLX runtime.", no_args_is_help=True)


@app.command("info")
def runtime_info_command(
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit stable JSON on stdout.")
    ] = False,
) -> None:
    """Report the selected MLX backend and device contract."""

    from nnunet_mlx.runtime import runtime_info

    payload = runtime_info().to_payload()
    if json_output:
        typer.echo(json.dumps(payload, sort_keys=True))
        return
    for key, value in payload.items():
        typer.echo(f"{key}: {value}")
