"""Portable identities for converted model folders."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict, cast

from nnunet_mlx.contracts import parse_mapping, parse_string
from nnunet_mlx.errors import ContractError
from nnunet_mlx.model.source_plans import normalize_inference_plans
from nnunet_mlx.plans import PlansContract, read_json_object
from nnunet_mlx.prediction.source import TOTALSEGMENTATOR_CT_SOURCE_PROFILE

CONVERSION_SCHEMA = "nnunet-mlx.conversion.v1"
CONVERSION_FILENAME = "conversion.json"


class ConversionFold(TypedDict):
    """Record one converted fold and the checkpoint directory holding its files."""

    fold: int | str
    checkpoint: str


class ConversionManifest(TypedDict):
    """Record the schema, plans identity and fold payloads of a converted model folder."""

    schema_version: str
    engine_version: str
    configuration: str
    source_profile: object
    architecture: object
    spatial_dims: object
    folds: list[ConversionFold]


def read_conversion_manifest(model_folder: str | Path) -> ConversionManifest:
    """Verify a converted folder before loading its model or reading an image.

    Source files are unnecessary for standalone inference. The producer package
    version is descriptive; compatibility follows the schema and model contracts.
    This check requires each listed fold's state and weights files to exist, but
    does not inspect weight contents or authenticate them. Checkpoint loading
    subsequently validates the checkpoint file set, state and model weights.

    Returns:
        The parsed ``conversion.json`` contents.

    Raises:
        ContractError: If the manifest or plans are missing or malformed, the schema
            is unsupported, a listed fold lacks ``weights.safetensors`` or
            ``state.json``, or the recorded architecture or spatial rank disagrees
            with the plans.
    """
    folder = Path(model_folder)
    payload = read_json_object(folder / CONVERSION_FILENAME)
    fields = {
        "schema_version",
        "engine_version",
        "configuration",
        "source_profile",
        "architecture",
        "spatial_dims",
        "folds",
    }
    # Extra keys are tolerated: older converted folders also carry legacy digest
    # fields, which are ignored.
    if not fields <= set(payload) or payload["schema_version"] != CONVERSION_SCHEMA:
        raise ContractError("unsupported conversion manifest; reconvert the model folder")
    parse_string(payload["engine_version"], name="conversion engine version")
    configuration = parse_string(payload["configuration"], name="conversion configuration")
    plans = PlansContract.from_payload(
        read_json_object(folder / "plans.json"), configuration=configuration
    )
    entries = payload["folds"]
    if not isinstance(entries, list) or not entries:
        raise ContractError("conversion manifest requires checkpoint folds")
    admitted: set[int | str] = set()
    first_directory: Path | None = None
    for value in cast(list[object], entries):
        entry = parse_mapping(value, name="conversion fold")
        fold = entry.get("fold")
        if not (
            (isinstance(fold, int) and not isinstance(fold, bool) and fold >= 0) or fold == "all"
        ):
            raise ContractError("conversion manifest has an invalid fold")
        fold = cast(int | str, fold)
        if fold in admitted:
            raise ContractError("conversion manifest has duplicate folds")
        admitted.add(fold)
        checkpoint = parse_string(entry.get("checkpoint"), name="conversion checkpoint")
        if Path(checkpoint).name != checkpoint:
            raise ContractError("conversion checkpoint must be one path component")
        directory = folder / f"fold_{fold}" / checkpoint
        if first_directory is None:
            first_directory = directory
        for filename in ("weights.safetensors", "state.json"):
            if not (directory / filename).is_file():
                raise ContractError(f"conversion payload is unavailable: {directory / filename}")
    if (
        payload["source_profile"] == TOTALSEGMENTATOR_CT_SOURCE_PROFILE
        and "architecture" not in plans.configuration.raw
    ):
        assert first_directory is not None
        state = read_json_object(first_directory / "state.json")
        identity = parse_mapping(state.get("identity"), name="checkpoint identity")
        plans = normalize_inference_plans(
            plans, source_profile=TOTALSEGMENTATOR_CT_SOURCE_PROFILE,
            trainer_name=parse_string(identity.get("trainer_name"), name="trainer name"),
        )
    configurations = parse_mapping(
        plans.topology_payload()["configurations"], name="configurations"
    )
    selected = parse_mapping(configurations[configuration], name="configuration")
    architecture = parse_mapping(selected["architecture"], name="architecture")
    if payload["architecture"] != architecture["network_class_name"] or payload[
        "spatial_dims"
    ] != len(plans.configuration.patch_size):
        raise ContractError("conversion architecture does not match plans")
    # The fields typed above are proved by the identity and per-fold checks.
    return cast(ConversionManifest, dict(payload))
