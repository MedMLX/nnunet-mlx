"""CLI composition and structured errors; command modules own their adapters."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from importlib.metadata import version
from typing import Annotated

import typer
from typer import _click as click

from nnunet_mlx.commands import download, prediction, runtime, upstream
from nnunet_mlx.errors import ContractError

app = typer.Typer(
    name="nnunet-mlx",
    help="MLX-native nnU-Net v2 inference.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
app.add_typer(runtime.app, name="runtime")
app.add_typer(prediction.app, name="model")
app.command("download")(download.download_command)
app.command("convert")(upstream.convert_command)
app.command("predict")(upstream.predict_command)


def _version_callback(value: bool) -> None:
    """Print the installed version and exit when requested."""
    if value:
        typer.echo(version("nnunet-mlx"))
        raise typer.Exit()


@app.callback()
def root(
    version_requested: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Print the package version and exit.",
        ),
    ] = False,
) -> None:
    """Select a command group."""


def _emit_error(error: BaseException) -> None:
    """Write a JSON error and usage hint to standard error."""
    payload = {"error": {"message": str(error), "hint": "Run with --help for usage."}}
    typer.echo(json.dumps(payload, sort_keys=True), err=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and serialize contract and Click errors to stderr.

    ``ContractError`` returns 1; Click errors retain their exit code. Other
    exceptions propagate to the caller, and Click exit requests retain their code.
    """

    try:
        app(
            args=None if argv is None else list(argv),
            prog_name="nnunet-mlx",
            standalone_mode=False,
        )
    except (ContractError, click.ClickException) as error:
        _emit_error(error)
        return 1 if isinstance(error, ContractError) else error.exit_code
    except click.exceptions.Exit as error:
        return int(error.exit_code)
    return 0


if __name__ == "__main__":
    sys.exit(main())
