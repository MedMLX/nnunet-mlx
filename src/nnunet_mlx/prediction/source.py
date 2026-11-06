"""Source-bound inference model identities admitted by nnunet-mlx."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from nnunet_mlx.contracts import parse_string
from nnunet_mlx.errors import ContractError

DEFAULT_INFERENCE_SOURCE_PROFILE = "nnunet-v2.8.1-default"
PROSTATE158_2D_SOURCE_PROFILE = "prostate158-2d-100epochs"
PROSTATE_MRL_SOURCE_PROFILE = "prostate-mrl-no-mirroring"
CADS_OAR258_SOURCE_PROFILE = "cads-oar258-open"
CHRONOROOT2_SOURCE_PROFILE = "chronoroot2-arabidopsis-2000epochs"
SMILE_SOURCE_PROFILE = "smile-abdomenatlasf-2d"
TOTALSEGMENTATOR_CT_SOURCE_PROFILE = "totalsegmentator-v2-ct"
LION_FDG_SOURCE_PROFILE = "lion-fdg-1.0.4"
MRSEGMENTATOR_SOURCE_PROFILE = "mrsegmentator-1.2-base"
LESIONTRACER_SOURCE_PROFILE = "lesiontracer-autopet3-2024"


# Every nnU-Net v2.8.1 trainer that keeps nnUNetTrainer.build_network_architecture.
# These variants change only training (epochs, loss, optimizer, augmentation,
# sampling), so the inference network follows from plans.json and the checkpoint,
# and mirroring follows from its stored inference_allowed_mirroring_axes.
# nnUNetTrainerBN and the Primus trainers build other networks and are excluded.
PLANS_BUILT_TRAINERS = (
    "nnUNetTrainer",
    "nnUNetTrainerAdam", "nnUNetTrainerAdam1en3", "nnUNetTrainerAdam3en4",
    "nnUNetTrainerAdan", "nnUNetTrainerAdan1en1", "nnUNetTrainerAdan1en3",
    "nnUNetTrainerAdan3en4", "nnUNetTrainerAdanCosAnneal",
    "nnUNetTrainerBenchmark_5epochs", "nnUNetTrainerBenchmark_5epochs_noDataLoading",
    "nnUNetTrainerCELoss", "nnUNetTrainerCELoss_5epochs", "nnUNetTrainerCosAnneal",
    "nnUNetTrainerDA5", "nnUNetTrainerDA5Segord0", "nnUNetTrainerDA5_10epochs",
    "nnUNetTrainerDA5ord0", "nnUNetTrainerDiceCELoss_noSmooth", "nnUNetTrainerDiceLoss",
    "nnUNetTrainerDiceTopK10Loss", "nnUNetTrainerNoDA", "nnUNetTrainerNoDeepSupervision",
    "nnUNetTrainerNoMirroring", "nnUNetTrainerTopk10Loss", "nnUNetTrainerTopk10LossLS01",
    "nnUNetTrainerVanillaAdam", "nnUNetTrainerVanillaAdam1en3",
    "nnUNetTrainerVanillaAdam3en4",
    "nnUNetTrainer_1epoch", "nnUNetTrainer_5epochs", "nnUNetTrainer_10epochs",
    "nnUNetTrainer_20epochs", "nnUNetTrainer_50epochs", "nnUNetTrainer_100epochs",
    "nnUNetTrainer_250epochs", "nnUNetTrainer_500epochs", "nnUNetTrainer_750epochs",
    "nnUNetTrainer_2000epochs", "nnUNetTrainer_4000epochs", "nnUNetTrainer_8000epochs",
    "nnUNetTrainer_250epochs_NoMirroring", "nnUNetTrainer_2000epochs_NoMirroring",
    "nnUNetTrainer_4000epochs_NoMirroring", "nnUNetTrainer_8000epochs_NoMirroring",
    "nnUNetTrainer_DASegOrd0", "nnUNetTrainer_DASegOrd0_NoMirroring",
    "nnUNetTrainer_noDummy2DDA", "nnUNetTrainer_onlyMirror01",
    "nnUNetTrainer_onlyMirror01_1500ep", "nnUNetTrainer_onlyMirror01_DA5",
    "nnUNetTrainer_onlyMirror01_DASegOrd0", "nnUNetTrainer_probabilisticOversampling",
    "nnUNetTrainer_probabilisticOversampling_010",
    "nnUNetTrainer_probabilisticOversampling_033", "nnUNetTrainer_warmup",
    "nnUNet_Trainer_BS8",
)


@dataclass(frozen=True, slots=True)
class InferenceModelIdentity:
    """One admitted combination of plans dataset, upstream trainer and configuration.

    ``None`` for ``dataset_name`` or ``configuration`` matches any value.
    ``required_arithmetic``, when set, is the only inference arithmetic admitted for
    this model.
    """

    dataset_name: str | None
    trainer_name: str
    configuration: str | None
    required_arithmetic: str | None = None


@dataclass(frozen=True, slots=True)
class InferenceSourceProfile:
    """A named set of model identities (``models``) admitted for inference."""

    name: str
    models: tuple[InferenceModelIdentity, ...]


_SOURCE_PROFILES = {
    LESIONTRACER_SOURCE_PROFILE: InferenceSourceProfile(
        name=LESIONTRACER_SOURCE_PROFILE,
        models=(InferenceModelIdentity(
            "Dataset226_AutoPETIII_only_organs", "autoPET3_Trainer", "3d_fullres_bs3",
        ),),
    ),
    TOTALSEGMENTATOR_CT_SOURCE_PROFILE: InferenceSourceProfile(
        name=TOTALSEGMENTATOR_CT_SOURCE_PROFILE,
        models=tuple(
            InferenceModelIdentity(dataset, trainer, "3d_fullres", "torch-order-fp32")
            for dataset, trainer in (
                ("Dataset291_TotalSegmentator_part1_organs_1559subj", "nnUNetTrainerNoMirroring"),
                ("Dataset292_TotalSegmentator_part2_vertebrae_1532subj",
                 "nnUNetTrainerNoMirroring"),
                ("Dataset293_TotalSegmentator_part3_cardiac_1559subj", "nnUNetTrainerNoMirroring"),
                ("Dataset294_TotalSegmentator_part4_muscles_1559subj", "nnUNetTrainerNoMirroring"),
                ("Dataset295_TotalSegmentator_part5_ribs_1559subj", "nnUNetTrainerNoMirroring"),
                ("Dataset297_TotalSegmentator_total_3mm_1559subj",
                 "nnUNetTrainer_4000epochs_NoMirroring"),
                ("Dataset298_TotalSegmentator_total_6mm_1559subj",
                 "nnUNetTrainer_4000epochs_NoMirroring"),
                # Body localization model; TotalSegmentator runs it without mirroring.
                ("Dataset300_body_6mm_1559subj", "nnUNetTrainer"),
            )
        ),
    ),
    MRSEGMENTATOR_SOURCE_PROFILE: InferenceSourceProfile(
        name=MRSEGMENTATOR_SOURCE_PROFILE,
        models=(InferenceModelIdentity(
            "Dataset050_ukbb_inhouse_nako", "nnUNetTrainerNoMirroring",
            "3d_fullres_bs8", "torch-order-fp32",
        ),),
    ),
    LION_FDG_SOURCE_PROFILE: InferenceSourceProfile(
        name=LION_FDG_SOURCE_PROFILE,
        models=(InferenceModelIdentity(
            "Dataset789_Tumors", "nnUNetTrainerDA5", "3d_fullres",
        ),),
    ),
    PROSTATE_MRL_SOURCE_PROFILE: InferenceSourceProfile(
        name=PROSTATE_MRL_SOURCE_PROFILE,
        models=(
            InferenceModelIdentity(
                "Dataset509_Prostate_MRL", "nnUNetTrainerNoMirroring", "3d_fullres",
            ),
        ),
    ),
    SMILE_SOURCE_PROFILE: InferenceSourceProfile(
        name=SMILE_SOURCE_PROFILE,
        models=(
            InferenceModelIdentity(
                "Dataset808_AbdomenAtlasF", "nnUNetTrainer", "2d", "torch-order-fp32",
            ),
        ),
    ),
    DEFAULT_INFERENCE_SOURCE_PROFILE: InferenceSourceProfile(
        name=DEFAULT_INFERENCE_SOURCE_PROFILE,
        models=tuple(
            InferenceModelIdentity(None, trainer, None) for trainer in PLANS_BUILT_TRAINERS
        ),
    ),
    PROSTATE158_2D_SOURCE_PROFILE: InferenceSourceProfile(
        name=PROSTATE158_2D_SOURCE_PROFILE,
        models=(
            InferenceModelIdentity(
                "Dataset158_prostate158_train", "nnUNetTrainer_100epochs", "2d",
            ),
        ),
    ),
    CADS_OAR258_SOURCE_PROFILE: InferenceSourceProfile(
        name=CADS_OAR258_SOURCE_PROFILE,
        models=(
            InferenceModelIdentity(
                "Dataset558_OAR258", "nnUNetTrainerNoMirroring", "3d_fullres",
            ),
        ),
    ),
    CHRONOROOT2_SOURCE_PROFILE: InferenceSourceProfile(
        name=CHRONOROOT2_SOURCE_PROFILE,
        models=(
            InferenceModelIdentity(
                "Dataset789_ChronoRoot2", "nnUNetTrainer_2000epochs", "2d",
            ),
        ),
    ),
}


def resolve_inference_source_profile(name: str) -> InferenceSourceProfile:
    """Resolve one explicitly named, immutable inference source profile."""

    try:
        return _SOURCE_PROFILES[name]
    except KeyError as error:
        supported = ", ".join(sorted(_SOURCE_PROFILES))
        raise ContractError(
            f"unsupported inference source profile {name!r}; expected one of {supported}"
        ) from error


def validate_inference_model_identity(
    profile: InferenceSourceProfile,
    *,
    plans_json: Mapping[str, object],
    trainer_name: str,
    configuration: str,
    arithmetic: str | None = None,
) -> None:
    """Check supplied model identity fields and, when supplied, execution arithmetic.

    Matching these metadata fields admits the profile; it does not verify the
    checkpoint's origin or prove numerical parity.
    """

    dataset_name = _dataset_name(plans_json, required=_requires_dataset(profile))
    for identity in profile.models:
        if not _matches_identity(
            identity,
            dataset_name=dataset_name,
            trainer_name=trainer_name,
            configuration=configuration,
        ):
            continue
        if (
            arithmetic is not None
            and identity.required_arithmetic is not None
            and arithmetic != identity.required_arithmetic
        ):
            raise ContractError(
                f"model {dataset_name!r} requires inference arithmetic "
                f"{identity.required_arithmetic!r}"
            )
        return
    raise ContractError(
        "unsupported inference model identity for source profile "
        f"{profile.name!r}: dataset={dataset_name!r}, trainer={trainer_name!r}, "
        f"configuration={configuration!r}"
    )


def _dataset_name(
    plans_json: Mapping[str, object], *, required: bool
) -> str | None:
    """Read the dataset name from plans when required or present."""
    value = plans_json.get("dataset_name")
    if value is None and not required:
        return None
    return parse_string(value, name="plans dataset_name")


def _requires_dataset(profile: InferenceSourceProfile) -> bool:
    """Report whether a source profile requires a dataset identity."""
    return any(identity.dataset_name is not None for identity in profile.models)


def _matches_identity(
    identity: InferenceModelIdentity,
    *,
    dataset_name: str | None,
    trainer_name: str,
    configuration: str,
) -> bool:
    """Compare trainer, optional dataset and configuration with a source identity."""
    return (
        identity.trainer_name == trainer_name
        and identity.dataset_name in (None, dataset_name)
        and identity.configuration in (None, configuration)
    )
