"""Run native training and resume from the last committed epoch."""

from __future__ import annotations

import fcntl
from collections.abc import Callable, Mapping
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Literal, TypedDict

from mlx import core as mx
from mlx.optimizers import AdamW, cosine_decay, join_schedules, linear_schedule

from nnunet_mlx.contracts import parse_boolean, parse_integer, parse_mapping, parse_number
from nnunet_mlx.errors import ContractError
from nnunet_mlx.model.factory import build_model_from_plans
from nnunet_mlx.model.network import Network, configure_training_forward
from nnunet_mlx.plans import read_json_object
from nnunet_mlx.runtime import runtime_info, validate_inference_execution
from nnunet_mlx.storage.checkpoint import load_inference_checkpoint
from nnunet_mlx.storage.filesystem import write_new_directory
from nnunet_mlx.training.checkpoint import (
    export_checkpoint,
    load_training_state,
    save_training_state,
    write_training_json,
)
from nnunet_mlx.training.data import PatchSampler, PreparedDataset
from nnunet_mlx.training.step import Precision, TrainingStep
from nnunet_mlx.training.validation import validate_cases

_RUN_SCHEMA = "nnunet-mlx.training-run.v1"
_CHECKPOINT = "training.safetensors"


class TrainingOverrides(TypedDict, total=False):
    """Options that remain fixed across restarts of a run."""

    batch_size: int
    steps_per_epoch: int
    validate_every: int
    seed: int
    learning_rate: float
    precision: Precision
    recompute: bool
    recompute_stages: int
    augment: bool
    spatial_augment: bool
    focus_label: int
    balanced_ce: bool
    cosine_lr: bool
    compile_step: bool


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    batch_size: int
    steps_per_epoch: int
    validate_every: int
    seed: int
    learning_rate: float
    precision: Precision
    recompute: bool
    augment: bool
    compile_step: bool
    recompute_stages: int
    spatial_augment: bool
    focus_label: int | None
    balanced_ce: bool
    cosine_lr: bool

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> TrainingSettings:
        batch = _positive_int(payload.get("batch_size"), name="batch_size")
        steps = _positive_int(payload.get("steps_per_epoch"), name="steps_per_epoch")
        validate_every = _positive_int(payload.get("validate_every"), name="validate_every")
        seed = parse_integer(payload.get("seed"), name="seed")
        if not 0 <= seed < 2**64:
            raise ContractError("seed must fit in uint64")
        rate = parse_number(payload.get("learning_rate"), name="learning_rate")
        if rate <= 0:
            raise ContractError("learning_rate must be positive")
        precision = payload.get("precision")
        if precision not in ("float32", "bfloat16"):
            raise ContractError("precision must be float32 or bfloat16")
        recompute = parse_boolean(payload.get("recompute"), name="recompute")
        stages = parse_integer(payload.get("recompute_stages", 0), name="recompute_stages")
        if stages < 0 or (recompute and stages):
            raise ContractError("choose full recomputation or a nonnegative recompute_stages count")
        augment = parse_boolean(payload.get("augment"), name="augment")
        spatial = parse_boolean(payload.get("spatial_augment", False), name="spatial_augment")
        if spatial and not augment:
            raise ContractError("spatial augmentation requires augmentation to be enabled")
        focus = payload.get("focus_label")
        return cls(
            batch,
            steps,
            validate_every,
            seed,
            rate,
            precision,
            recompute,
            augment,
            parse_boolean(payload.get("compile_step"), name="compile_step"),
            stages,
            spatial,
            None if focus is None else _positive_int(focus, name="focus_label"),
            parse_boolean(payload.get("balanced_ce", False), name="balanced_ce"),
            parse_boolean(payload.get("cosine_lr", False), name="cosine_lr"),
        )


def train_model(
    prepared_folder: Path,
    destination: Path,
    *,
    epochs: int | None = None,
    options: TrainingOverrides | None = None,
    resume: bool = False,
    initialize_from: Path | None = None,
    progress: Callable[[Mapping[str, object]], None] | None = None,
) -> Path:
    """Train on Metal and return a checkpoint in the standard prediction folder."""
    runtime = runtime_info()
    data = PreparedDataset.load(prepared_folder)
    overrides = dict(options or {})
    if resume:
        if initialize_from is not None:
            raise ContractError("initialize_from starts a new run")
        if not (destination / "training.json").is_file():
            raise ContractError(f"no training run to resume at {destination}")
    else:
        if destination.exists():
            raise ContractError(f"training destination exists: {destination}. Use --resume")
        settings = TrainingSettings.from_payload(
            {
                "batch_size": data.plans.configuration.batch_size,
                "steps_per_epoch": 250,
                "validate_every": 5,
                "seed": 0,
                "learning_rate": 1e-3,
                "precision": "float32",
                "recompute": False,
                "augment": True,
                "compile_step": True,
            }
            | overrides
        )
        target = 100 if epochs is None else _positive_int(epochs, name="epochs")
        schedule_steps = target * settings.steps_per_epoch if settings.cosine_lr else None
        model, optimizer = _build(
            data,
            settings,
            initialize_from=initialize_from,
            schedule_steps=schedule_steps,
        )

        def initialize(folder: Path) -> None:
            fold_folder = folder / f"fold_{data.fold}"
            fold_folder.mkdir()
            write_training_json(folder / "plans.json", data.plans.raw)
            write_training_json(folder / "dataset.json", data.dataset)
            write_training_json(
                folder / "training.json",
                {
                    "schema": _RUN_SCHEMA,
                    "prepared_dataset": data.identifier,
                    "settings": asdict(settings),
                    "target_epochs": target,
                    "schedule_steps": schedule_steps,
                    "runtime": runtime.to_payload() | {"dtype": settings.precision},
                },
            )
            save_training_state(fold_folder / _CHECKPOINT, model, optimizer, [])

        write_new_directory(destination, initialize)

    with (destination / ".training.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ContractError(f"a training process is already using {destination}") from error
        return _run_epochs(destination, data, epochs, overrides, progress)


def _build(
    data: PreparedDataset,
    settings: TrainingSettings,
    *,
    initialize_from: Path | None = None,
    schedule_steps: int | None = None,
) -> tuple[Network, AdamW]:
    config = data.plans.configuration
    if config.previous_stage_name is not None:
        raise ContractError("training requires a single-stage configuration")
    if (
        settings.focus_label is not None
        and settings.focus_label not in data.semantics.all_labels[1:]
    ):
        raise ContractError("focus_label must be a foreground label in dataset.json")
    if settings.balanced_ce and data.semantics.has_regions:
        raise ContractError("balanced cross-entropy requires mutually exclusive class targets")
    mx.random.seed(settings.seed)
    model = build_model_from_plans(
        data.plans.topology_payload(),
        configuration=config.name,
        input_channels=data.channels,
        output_channels=data.semantics.output_channels,
    )
    rank = len(config.patch_size)
    if model.spec.spatial_dims != rank or model.spec.strides[0] != (1,) * rank:
        raise ContractError("training requires a network that emits logits at the patch resolution")
    validate_inference_execution(model)
    if initialize_from is not None:
        load_inference_checkpoint(initialize_from, model=model)
    configure_training_forward(
        model,
        recompute=settings.recompute,
        recompute_stages=settings.recompute_stages,
    )
    model.train()
    rate: float | Callable[[mx.array], mx.array] = settings.learning_rate
    if schedule_steps is not None and schedule_steps > 1:
        warmup = max(1, schedule_steps // 20)
        rate = join_schedules(
            (
                linear_schedule(0.1 * settings.learning_rate, settings.learning_rate, warmup),
                cosine_decay(
                    settings.learning_rate,
                    schedule_steps - warmup,
                    0.05 * settings.learning_rate,
                ),
            ),
            (warmup,),
        )
    optimizer = AdamW(learning_rate=rate, weight_decay=1e-4, bias_correction=True)
    optimizer.init(model.trainable_parameters())
    return model, optimizer


def _run_epochs(
    destination: Path,
    data: PreparedDataset,
    epochs: int | None,
    overrides: Mapping[str, object],
    progress: Callable[[Mapping[str, object]], None] | None,
) -> Path:
    run_path = destination / "training.json"
    run = read_json_object(run_path)
    if run.get("schema") != _RUN_SCHEMA or run.get("prepared_dataset") != data.identifier:
        raise ContractError("resume requires the prepared dataset used to start this run")
    if (
        read_json_object(destination / "plans.json") != data.plans.raw
        or read_json_object(destination / "dataset.json") != data.dataset
    ):
        raise ContractError("prepared plans or dataset metadata changed since training started")
    saved = parse_mapping(run.get("settings"), name="training settings")
    settings = TrainingSettings.from_payload(saved)
    resolved = asdict(settings)
    if changed := [key for key, value in overrides.items() if resolved.get(key) != value]:
        raise ContractError(f"resume cannot change training settings: {', '.join(sorted(changed))}")
    target = _positive_int(run["target_epochs"] if epochs is None else epochs, name="epochs")
    schedule_steps = (
        _positive_int(run.get("schedule_steps"), name="schedule_steps")
        if settings.cosine_lr
        else None
    )
    model, optimizer = _build(data, settings, schedule_steps=schedule_steps)
    fold = destination / f"fold_{data.fold}"
    history = load_training_state(fold / _CHECKPOINT, model, optimizer)
    if target < len(history):
        raise ContractError(f"this run already completed {len(history)} epochs")
    if target != run["target_epochs"]:
        write_training_json(run_path, run | {"target_epochs": target})

    def publish(
        epoch: int,
        name: Literal["checkpoint_best", "checkpoint_final"],
    ) -> Path:
        return export_checkpoint(
            fold,
            model if epoch == len(history) else None,
            configuration=data.plans.configuration.name,
            fold=data.fold,
            epoch=epoch,
            augment=settings.augment and not settings.spatial_augment,
            name=name,
        )

    best = float("-inf")
    previous_best = next((record for record in reversed(history) if record.get("best")), None)
    if previous_best is not None:
        best = parse_number(previous_best.get("validation_dice"), name="best validation Dice")
        publish(parse_integer(previous_best.get("epoch"), name="best epoch"), "checkpoint_best")

    step = TrainingStep(
        model,
        optimizer,
        data.semantics,
        precision=settings.precision,
        compile_step=settings.compile_step,
        balanced_ce=settings.balanced_ce,
    )
    sampler = PatchSampler(
        data,
        batch_size=settings.batch_size,
        augment=settings.augment,
        focus_label=settings.focus_label,
        spatial_augment=settings.spatial_augment,
    )
    metrics_path = destination / "metrics.json"
    if history:
        write_training_json(metrics_path, history)
    for epoch in range(len(history) + 1, target + 1):
        metrics = _run_epoch(epoch, target, settings, sampler, step, model, data)
        score = metrics.get("validation_dice")
        if isinstance(score, float) and score > best:
            best = score
            metrics["best"] = True
        history.append(metrics)
        save_training_state(fold / _CHECKPOINT, model, optimizer, history)
        if metrics.get("best"):
            publish(epoch, "checkpoint_best")
        write_training_json(metrics_path, history)
        if progress is not None:
            progress(metrics)
    return publish(target, "checkpoint_final")


def _run_epoch(
    epoch: int,
    target: int,
    settings: TrainingSettings,
    sampler: PatchSampler,
    step: TrainingStep,
    model: Network,
    data: PreparedDataset,
) -> dict[str, object]:
    start = perf_counter()
    with closing(
        sampler.batches(
            seed=settings.seed,
            epoch=epoch,
            steps=settings.steps_per_epoch,
        )
    ) as batches:
        loss = fmean(step(batch) for batch in batches)
    record: dict[str, object] = {
        "epoch": epoch,
        "loss": loss,
        "training_seconds": perf_counter() - start,
    }
    if data.split.val and (epoch % settings.validate_every == 0 or epoch == target):
        validation = validate_cases(model, data)
        record.update(validation_dice=validation.dice, validation_per_output=validation.per_output)
    return record


def _positive_int(value: object, *, name: str) -> int:
    result = parse_integer(value, name=name)
    if result < 1:
        raise ContractError(f"{name} must be positive")
    return result
