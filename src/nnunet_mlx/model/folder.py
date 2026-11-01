"""MLX-native nnU-Net model-folder loading and checkpoint validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from nnunet_mlx.contracts import parse_mapping
from nnunet_mlx.contracts import parse_string as _string
from nnunet_mlx.data.labels import LabelSemantics, parse_dataset_label_semantics
from nnunet_mlx.errors import ContractError
from nnunet_mlx.model import factory as model_factory
from nnunet_mlx.model import metadata as model_metadata
from nnunet_mlx.model.residency import CheckpointBackedFoldModel, FoldActivation
from nnunet_mlx.model.source_plans import normalize_inference_plans
from nnunet_mlx.plans import PlansContract, read_json_object
from nnunet_mlx.prediction.source import (
    DEFAULT_INFERENCE_SOURCE_PROFILE,
    SMILE_SOURCE_PROFILE,
    TOTALSEGMENTATOR_CT_SOURCE_PROFILE,
    InferenceSourceProfile,
    resolve_inference_source_profile,
    validate_inference_model_identity,
)
from nnunet_mlx.storage.checkpoint import (
    InferenceCheckpointState,
    checkpoint_weights_signature,
    load_inference_checkpoint,
)
from nnunet_mlx.storage.imageio import resolve_image_reader

if TYPE_CHECKING:
    from nnunet_mlx.model.types import InferenceModel


@dataclass(frozen=True, slots=True)
class ModelBundle:
    """A loaded MLX model folder, ready for ``ModelFolderPredictor``.

    ``models`` holds one callable per entry of ``folds`` (all sharing one resident
    network when several folds are selected). ``image_channels`` is the number of
    input modalities in dataset.json; ``input_channels`` additionally counts the
    one-hot previous-stage channels of a cascade model. ``mirroring_axes`` are the
    spatial axes (plans axis order) used for test-time mirroring; the empty tuple
    disables it. ``arithmetic`` is ``"native"`` or ``"torch-order-fp32"``.
    """

    folder: Path
    plans: PlansContract
    dataset_json: dict[str, object]
    semantics: LabelSemantics
    models: tuple[InferenceModel, ...]
    folds: tuple[int | str, ...]
    checkpoints: tuple[CheckpointProvenance, ...]
    mirroring_axes: tuple[int, ...]
    file_ending: str
    image_channels: int
    input_channels: int
    image_reader_writer: str
    arithmetic: str = "native"


@dataclass(frozen=True, slots=True)
class CheckpointProvenance:
    """Recorded checkpoint identity behind one fold of a model bundle.

    ``weights_signature`` is ``[size_bytes, mtime_ns]`` of ``weights.safetensors``.
    It is a file-change indicator, not a content digest or a guarantee that the
    file contents are unchanged.
    """

    fold: int | str
    checkpoint: str
    weights_signature: list[int]

    def to_payload(self) -> dict[str, object]:
        """Serialize checkpoint fold, identity and file-change metadata."""
        return {
            "fold": self.fold,
            "checkpoint": self.checkpoint,
            "weights_signature": self.weights_signature,
        }


def load_model_bundle(
    model_folder: str | Path,
    *,
    configuration: str,
    folds: Sequence[int | str] | None = None,
    checkpoint_name: str = "checkpoint_final",
    source_profile: str = DEFAULT_INFERENCE_SOURCE_PROFILE,
    arithmetic: str = "native",
) -> ModelBundle:
    """Load and verify a converted MLX model folder.

    Args:
        model_folder: Folder with ``plans.json``, ``dataset.json`` and
            ``fold_<n>/<checkpoint_name>/`` MLX checkpoints containing ``state.json``
            and ``weights.safetensors``. Conversion receipts are not required.
        configuration: Plans configuration to run, e.g. ``"3d_fullres"``.
        folds: Folds to load (non-negative integers or ``"all"``); ``None`` detects
            every numeric fold that has ``checkpoint_name``.
        checkpoint_name: MLX checkpoint directory name inside each fold folder.
        source_profile: Name of the admitted model-identity profile the checkpoint
            identity is checked against.
        arithmetic: ``"native"``, or ``"torch-order-fp32"`` to follow the pinned
            PyTorch CPU float32 operation order on Metal.

    Raises:
        ContractError: If the folder, plans or dataset metadata are missing or
            inconsistent, a checkpoint does not match the requested
            fold, configuration or profile, an upstream ``.pth`` file is given
            instead of a converted checkpoint, or the folds disagree on mirroring.
    """
    if arithmetic not in {"native", "torch-order-fp32"}:
        raise ContractError(f"unsupported inference arithmetic {arithmetic!r}")
    folder = Path(model_folder)
    profile = resolve_inference_source_profile(source_profile)
    if not folder.is_dir():
        raise ContractError(f"model folder does not exist: {folder}")
    plans_json = read_json_object(folder / "plans.json")
    dataset_json = read_json_object(folder / "dataset.json")
    plans = PlansContract.from_payload(plans_json, configuration=configuration)
    semantics = parse_dataset_label_semantics(dataset_json)
    image_channels = model_metadata.input_channel_count(dataset_json)
    input_channels = model_metadata.model_input_channel_count(dataset_json, plans, semantics)
    selected_folds = (
        auto_detect_folds(folder, checkpoint_name=checkpoint_name)
        if folds is None
        else model_metadata.validate_folds(folds)
    )
    if profile.name == SMILE_SOURCE_PROFILE and selected_folds != ("all",):
        raise ContractError("SMILE inference is qualified only for the single fold 'all'")
    if (
        profile.name == TOTALSEGMENTATOR_CT_SOURCE_PROFILE
        and "architecture" not in plans.configuration.raw
    ):
        state = read_json_object(
            folder / f"fold_{selected_folds[0]}" / checkpoint_name / "state.json"
        )
        identity = parse_mapping(state.get("identity"), name="checkpoint identity")
        plans = normalize_inference_plans(
            plans, source_profile=profile.name,
            trainer_name=_string(identity.get("trainer_name"), name="trainer name"),
            arithmetic=arithmetic,
        )
    topology_plans = plans.topology_payload()
    model = model_factory.build_model_from_plans(
        topology_plans,
        configuration=configuration,
        input_channels=input_channels,
        output_channels=semantics.output_channels,
    )
    activation = FoldActivation()
    models: list[InferenceModel] = []
    states: list[InferenceCheckpointState] = []
    provenance: list[CheckpointProvenance] = []
    for fold in selected_folds:
        checkpoint = folder / f"fold_{fold}" / checkpoint_name
        if checkpoint.is_file() or checkpoint.suffix == ".pth":
            raise ContractError(
                "checkpoint must be a native nnunet-mlx safetensors directory"
            )
        state = load_inference_checkpoint(checkpoint, model=model)
        _validate_checkpoint_identity(
            state,
            configuration=configuration,
            fold=fold,
            plans_json=plans_json,
            source_profile=profile,
            arithmetic=arithmetic,
        )
        models.append(
            model
            if len(selected_folds) == 1
            else CheckpointBackedFoldModel(
                model,
                fold=fold,
                checkpoint=checkpoint,
                activation=activation,
            )
        )
        states.append(state)
        provenance.append(
            CheckpointProvenance(
                fold=fold,
                checkpoint=checkpoint_name,
                weights_signature=checkpoint_weights_signature(checkpoint),
            )
        )
        activation.fold = fold
    # All folds share one network: the loop above leaves the last fold's weights
    # resident, which ``activation`` records so the wrappers can skip a reload.
    model.eval()
    if arithmetic != "native":
        from nnunet_mlx.prediction.arithmetic import apply_inference_arithmetic

        apply_inference_arithmetic(
            model,
            arithmetic,
            source_profile=profile.name,
            patch_size=tuple(plans.configuration.patch_size),
        )
    mirroring_axes = states[0].inference_allowed_mirroring_axes
    if any(state.inference_allowed_mirroring_axes != mirroring_axes for state in states[1:]):
        raise ContractError("model folds disagree on inference mirroring axes")
    file_ending = _string(dataset_json.get("file_ending"), name="dataset file_ending")
    if not file_ending.startswith("."):
        raise ContractError("dataset file_ending must start with a dot")
    image_reader_writer = resolve_image_reader(plans.image_reader_writer, file_ending=file_ending)
    return ModelBundle(
        folder=folder,
        plans=plans,
        dataset_json=dataset_json,
        semantics=semantics,
        models=tuple(models),
        folds=selected_folds,
        checkpoints=tuple(provenance),
        mirroring_axes=mirroring_axes,
        file_ending=file_ending,
        image_channels=image_channels,
        input_channels=input_channels,
        image_reader_writer=image_reader_writer,
        arithmetic=arithmetic,
    )


def auto_detect_folds(model_folder: str | Path, *, checkpoint_name: str) -> tuple[int, ...]:
    """Return the sorted numeric folds whose folder holds the MLX ``checkpoint_name`` directory."""
    folder = Path(model_folder)
    detected: list[int] = []
    for candidate in folder.glob("fold_*"):
        suffix = candidate.name.removeprefix("fold_")
        if suffix.isdigit() and (candidate / checkpoint_name).is_dir():
            detected.append(int(suffix))
    if not detected:
        raise ContractError(f"no numeric folds contain MLX checkpoint {checkpoint_name!r}")
    return tuple(sorted(detected))


def _validate_checkpoint_identity(
    state: InferenceCheckpointState,
    *,
    configuration: str,
    fold: int | str,
    plans_json: Mapping[str, object],
    source_profile: InferenceSourceProfile,
    arithmetic: str,
) -> None:
    """Require a checkpoint's identity to match its folder, source profile and arithmetic."""
    identity = state.identity
    validate_inference_model_identity(
        source_profile,
        plans_json=plans_json,
        trainer_name=identity.trainer_name,
        configuration=identity.configuration,
        arithmetic=arithmetic,
    )
    if identity.configuration != configuration or identity.fold != fold:
        raise ContractError("checkpoint configuration or fold does not match its folder")
