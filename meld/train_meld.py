#!/usr/bin/env python3
"""Train TriMEF on MELD with dev-WF1 checkpoint selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Subset

from config import BATCH_SIZE, DEFAULT_EPOCHS, get_recipe
from dataset import MELDMMERDataset, collate_fn_mmer
from training import (
    atomic_json_save,
    atomic_torch_save,
    capture_rng_state,
    compute_metrics,
    evaluate,
    resolve_device,
    set_seed,
    train_one_epoch,
)
from model_factory import (
    CONTEXT_CONFIGURATION,
    DECISION_MOE_CONFIGURATION,
    UTEC_CONFIGURATION,
    build_model,
    model_audit,
)


PROJECT_ROOT = Path(__file__).resolve().parent
RESULT_SCHEMA_VERSION = 1
LEARNING_RATE = 5e-5
CTC_WEIGHT = 0.1
SCHEDULER_PATIENCE = 5
INITIAL_EPOCH_BUDGET = 50
MAX_EPOCH_BUDGET = 100
EXTENSION_WINDOW = 10
ALLOWED_UTEC_WEIGHTS = (0.1,)
CODE_FILES = (
    "train_meld.py",
    "dataset.py",
    "model_factory.py",
    "config.py",
    "interaction.py",
    "training.py",
    "model.py",
    "contrastive.py",
    "utec.py",
    "reliability_logit_fusion.py",
    "avel.py",
    "decision_moe.py",
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-feature-dir", type=Path, required=True)
    parser.add_argument("--wav2vec2-path", type=Path, required=True)
    parser.add_argument("--roberta-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=INITIAL_EPOCH_BUDGET)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCH_BUDGET)
    parser.add_argument(
        "--lambda-utec",
        type=float,
        required=True,
        choices=ALLOWED_UTEC_WEIGHTS,
    )
    parser.add_argument("--defer-test", action="store_true")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    args = parser.parse_args(argv)
    if args.epochs != INITIAL_EPOCH_BUDGET:
        parser.error(
            f"TriMEF fixes the initial budget to {INITIAL_EPOCH_BUDGET} epochs"
        )
    if args.max_epochs != MAX_EPOCH_BUDGET:
        parser.error(f"TriMEF fixes the maximum budget to {MAX_EPOCH_BUDGET} epochs")
    if not args.defer_test:
        parser.error(
            "training must use --defer-test; test is evaluated exactly once "
            "after dev-only checkpoint selection"
        )
    if args.batch_size != 16:
        parser.error("TriMEF on MELD requires batch size 16")
    if args.seed != 42:
        parser.error("TriMEF on MELD fixes seed=42")
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be positive")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", args.run_name):
        parser.error("--run-name contains unsupported characters")
    return args


def meld_utec_recipe(lambda_utec: float) -> dict:
    recipe = dict(get_recipe())
    if not any(
        math.isclose(lambda_utec, allowed, rel_tol=0.0, abs_tol=1e-12)
        for allowed in ALLOWED_UTEC_WEIGHTS
    ):
        raise ValueError(f"unsupported UTEC coefficient: {lambda_utec}")
    recipe["lambda_utec"] = float(lambda_utec)
    suffix = f"{lambda_utec:.2f}".replace(".", "")
    recipe["recipe"] = f"jointattn_ctc010_utec{suffix}_uniform_experts_meld"
    return recipe


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def hash_code_files() -> dict[str, str]:
    hashes = {}
    for name in CODE_FILES:
        path = PROJECT_ROOT / name
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[name] = sha256_file(path)
    return hashes


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(
    dataset,
    indices: list[int],
    batch_size: int,
    shuffle: bool,
    workers: int,
    seed: int | None = None,
):
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn_mmer,
        num_workers=workers,
        pin_memory=True,
        generator=generator,
        worker_init_fn=seed_worker,
    )
    return loader, generator


def environment_record(device: torch.device) -> dict:
    record = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda":
        record["gpu_name"] = torch.cuda.get_device_name(device)
    return record


@torch.no_grad()
def evaluate_detailed(model, loader, device, dataset) -> dict:

    model.eval()
    predictions = []
    labels = []
    file_ids = []
    for batch in loader:
        audio_inputs = batch["audio_inputs"].to(device)
        audio_lengths = batch["audio_lengths"].to(device)
        text_input_ids = batch["text_input_ids"].to(device)
        text_attention_mask = batch["text_attention_mask"].to(device)
        text_target_mask = batch["text_target_mask"].to(device)
        video_features = batch["video_features"].to(device, non_blocking=True)
        video_attention_mask = batch["video_attention_mask"].to(
            device,
            non_blocking=True,
        )
        video_quality = batch["video_quality"].to(device, non_blocking=True)
        batch_size, max_audio_len = audio_inputs.shape
        audio_attention_mask = torch.zeros(
            batch_size,
            max_audio_len,
            dtype=torch.long,
            device=device,
        )
        for index, length in enumerate(audio_lengths):
            audio_attention_mask[index, :length] = 1
        outputs = model(
            audio_input=audio_inputs,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            text_target_mask=text_target_mask,
            audio_attention_mask=audio_attention_mask,
            mode="eval",
            video_features=video_features,
            video_attention_mask=video_attention_mask,
            video_quality=video_quality,
        )
        logits = outputs["emotion_logits"]
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("non-finite formal test logits")
        predictions.extend(torch.argmax(logits, dim=1).cpu().tolist())
        labels.extend(batch["emotion_labels"].tolist())
        file_ids.extend(batch["file_ids"])

    predictions_array = np.asarray(predictions, dtype=np.int64)
    labels_array = np.asarray(labels, dtype=np.int64)
    num_classes = len(dataset.emotion_to_id)
    wa, ua, wf1 = compute_metrics(
        predictions_array,
        labels_array,
        num_classes,
    )
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, prediction in zip(labels_array, predictions_array):
        confusion[int(target), int(prediction)] += 1
    per_class = {}
    for class_id in range(num_classes):
        true_positive = int(confusion[class_id, class_id])
        support = int(confusion[class_id].sum())
        predicted = int(confusion[:, class_id].sum())
        recall = true_positive / support if support else 0.0
        precision = true_positive / predicted if predicted else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        per_class[dataset.id_to_emotion[class_id]] = {
            "class_id": class_id,
            "support": support,
            "precision": precision * 100,
            "recall": recall * 100,
            "f1": f1 * 100,
        }
    return {
        "wa": float(wa),
        "ua": float(ua),
        "wf1": float(wf1),
        "count": len(labels),
        "confusion_matrix": confusion.tolist(),
        "class_order": [dataset.id_to_emotion[index] for index in range(num_classes)],
        "per_class": per_class,
        "predictions": [
            {
                "file_id": file_id,
                "label": int(label),
                "prediction": int(prediction),
            }
            for file_id, label, prediction in zip(
                file_ids,
                labels,
                predictions,
            )
        ],
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    recipe = meld_utec_recipe(args.lambda_utec)
    output_dir = args.output_root / "results" / args.run_name
    checkpoint_dir = args.output_root / "checkpoints" / args.run_name
    log_dir = args.output_root / "logs" / args.run_name
    collisions = [
        path for path in (output_dir, checkpoint_dir, log_dir) if path.exists()
    ]
    if collisions:
        raise FileExistsError(f"formal run already exists: {collisions}")
    for path in (output_dir, checkpoint_dir, log_dir):
        path.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "train.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )

    paths = {
        "manifest": args.manifest.resolve(),
        "video_features": args.video_feature_dir.resolve(),
        "wav2vec2": args.wav2vec2_path.resolve(),
        "roberta": args.roberta_path.resolve(),
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"missing {name}: {path}")
    code_hashes = hash_code_files()
    started_at = datetime.now()
    run_config = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_name": args.run_name,
        "started_at": started_at.isoformat(),
        "arguments": {
            **vars(args),
            **{
                key: str(value)
                for key, value in vars(args).items()
                if isinstance(value, Path)
            },
        },
        "locked_recipe": recipe,
        "loss": {
            "ser": 1.0,
            "ctc": CTC_WEIGHT,
            "utec": recipe["lambda_utec"],
            "expert_ce": DECISION_MOE_CONFIGURATION["expert_loss_weight"],
        },
        "context_configuration": CONTEXT_CONFIGURATION,
        "decision_moe_configuration": DECISION_MOE_CONFIGURATION,
        "utec_configuration": UTEC_CONFIGURATION,
        "protocol": {
            "name": "trimef_meld_dev_selected_v1",
            "readout": "fixed_equal_available_mean",
            "train_split": "train",
            "selection_split": "dev",
            "selection_metric": "dev_wf1",
            "selection_direction": "max",
            "initial_epoch_budget": args.epochs,
            "maximum_epoch_budget": args.max_epochs,
            "extension_rule": (
                "extend_to_100_iff_best_dev_wf1_epoch_is_within_41_to_50"
            ),
            "no_metric_early_stopping_within_active_budget": True,
            "test_evaluations_during_training": 0,
            "formal_test_evaluations_after_dev_completion": 1,
            "formal_test_deferred_until_both_arms_complete": True,
            "test_labels_consumed_by_forward": False,
        },
        "source_hashes": code_hashes,
        "data_hashes": {"manifest": sha256_file(paths["manifest"])},
    }
    atomic_json_save(run_config, output_dir / "run_config.json")

    set_seed(args.seed)
    device = resolve_device(args.device)
    dataset = MELDMMERDataset(
        csv_path=str(paths["manifest"]),
        roberta_path=str(paths["roberta"]),
        max_text_length=CONTEXT_CONFIGURATION["max_text_length"],
        video_feature_dir=str(paths["video_features"]),
        use_dialogue_context=True,
    )
    frame = pd.read_csv(paths["manifest"])
    split_indices = {
        split: frame.index[frame["split"].astype(str) == split].tolist()
        for split in ("train", "dev", "test")
    }
    assigned = split_indices["train"] + split_indices["dev"] + split_indices["test"]
    if len(assigned) != len(frame) or len(set(assigned)) != len(frame):
        raise AssertionError("train/dev/test must partition the manifest")
    expected_counts = {"train": 9988, "dev": 1108, "test": 2610}
    actual_counts = {split: len(indices) for split, indices in split_indices.items()}
    if actual_counts != expected_counts:
        raise ValueError(f"locked common_v1 split counts changed: {actual_counts}")
    missing_features = [
        str(paths["video_features"] / f"{file_id}.pt")
        for file_id in frame["file"].astype(str)
        if not (paths["video_features"] / f"{file_id}.pt").is_file()
    ]
    if missing_features:
        raise FileNotFoundError(
            f"{len(missing_features)} video payloads missing; "
            f"first={missing_features[:5]}"
        )

    train_loader, train_generator = make_loader(
        dataset,
        split_indices["train"],
        args.batch_size,
        True,
        4,
        args.seed,
    )
    dev_loader, _ = make_loader(
        dataset,
        split_indices["dev"],
        args.batch_size,
        False,
        2,
    )
    model_args = argparse.Namespace(
        wav2vec2_path=str(paths["wav2vec2"]),
        roberta_path=str(paths["roberta"]),
    )
    model = build_model(model_args, dataset, recipe).to(device)
    architecture_audit = model_audit(model)
    architecture_audit.update(
        {
            "dataset_adapter": "MELD_7class_standard_split",
            "num_emotions": len(dataset.emotion_to_id),
            "emotion_mapping": dataset.emotion_to_id,
            "model_core_source": "TriMEF",
            "decision_moe_aggregation": "fixed_equal_available_mean",
        }
    )
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=SCHEDULER_PATIENCE,
        verbose=True,
    )
    criterion_ser = nn.CrossEntropyLoss()
    criterion_ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    best_path = checkpoint_dir / "dev_wf1_best.pt"
    last_path = checkpoint_dir / "last.pt"
    best_dev = {"wa": 0.0, "ua": 0.0, "wf1": -float("inf")}
    best_epoch = 0
    history = []
    active_epoch_budget = args.epochs
    extension_triggered = False
    extension_reason = None

    logging.info(
        "TriMEF MELD: train=%d dev=%d test=%d; "
        "dev WF1 selects the checkpoint; test remains untouched until selection",
        actual_counts["train"],
        actual_counts["dev"],
        actual_counts["test"],
    )
    logging.info("Locked recipe: %s", recipe)
    logging.info("Emotion mapping: %s", dataset.emotion_to_id)
    logging.info("Architecture audit: %s", architecture_audit)

    for epoch in range(1, args.max_epochs + 1):
        if epoch > active_epoch_budget:
            break
        learning_rate = float(optimizer.param_groups[0]["lr"])
        losses = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion_ser,
            criterion_ctc,
            CTC_WEIGHT,
            recipe["lambda_at"],
            recipe["lambda_vt"],
            device,
            epoch,
            lambda_video_aux=recipe["lambda_video_aux"],
            lambda_consistency=recipe["lambda_consistency"],
            lambda_rank=recipe["lambda_rank"],
            rank_margin=recipe["rank_margin"],
            lambda_avel=recipe["lambda_avel"],
            lambda_decision_moe=DECISION_MOE_CONFIGURATION["expert_loss_weight"],
            lambda_utec=recipe["lambda_utec"],
        )
        for name in (
            "utec",
            "utec_audio_text",
            "utec_audio_video",
            "utec_text_video",
        ):
            if name not in losses or not math.isfinite(float(losses[name])):
                raise RuntimeError(f"missing/non-finite training loss: {name}")
        dev_wa, dev_ua, dev_wf1 = evaluate(
            model,
            dev_loader,
            device,
            len(dataset.emotion_to_id),
        )
        scheduler.step(dev_wf1)
        improved = dev_wf1 > best_dev["wf1"]
        if improved:
            best_dev = {
                "wa": float(dev_wa),
                "ua": float(dev_ua),
                "wf1": float(dev_wf1),
            }
            best_epoch = epoch
            atomic_torch_save(model.state_dict(), best_path)
        history.append(
            {
                "epoch": epoch,
                "train": {name: float(value) for name, value in losses.items()},
                "dev": {
                    "wa": float(dev_wa),
                    "ua": float(dev_ua),
                    "wf1": float(dev_wf1),
                },
                "dev_evaluation_index": epoch,
                "test_evaluations": 0,
                "lr": learning_rate,
                "lr_after_scheduler": float(optimizer.param_groups[0]["lr"]),
                "improved": bool(improved),
            }
        )
        if epoch == args.epochs:
            extension_window_start = args.epochs - EXTENSION_WINDOW + 1
            if best_epoch >= extension_window_start:
                active_epoch_budget = args.max_epochs
                extension_triggered = True
                extension_reason = (
                    f"best_dev_wf1_epoch_{best_epoch}_within_"
                    f"{extension_window_start}_{args.epochs}"
                )
                logging.info(
                    "Budget extension triggered at epoch %d: %s; "
                    "continuing the same run to epoch %d without test access",
                    epoch,
                    extension_reason,
                    active_epoch_budget,
                )
            else:
                extension_reason = (
                    f"best_dev_wf1_epoch_{best_epoch}_before_"
                    f"{extension_window_start}"
                )
                logging.info(
                    "Budget extension not triggered at epoch %d: %s; "
                    "ending training without test access",
                    epoch,
                    extension_reason,
                )
        atomic_json_save(
            {
                "schema_version": RESULT_SCHEMA_VERSION,
                "status": "running",
                "run_name": args.run_name,
                "epoch": epoch,
                "initial_epoch_budget": args.epochs,
                "maximum_epoch_budget": args.max_epochs,
                "active_epoch_budget": active_epoch_budget,
                "extension_triggered": extension_triggered,
                "extension_reason": extension_reason,
                "dev_evaluations": epoch,
                "test_evaluations": 0,
                "best_epoch": best_epoch,
                "best_dev": best_dev,
                "history": history,
            },
            output_dir / "progress.json",
        )
        if epoch % args.checkpoint_every == 0 or epoch == active_epoch_budget:
            atomic_torch_save(
                {
                    "kind": "trimef_meld_training_state_v1",
                    "run_name": args.run_name,
                    "epoch": epoch,
                    "initial_epoch_budget": args.epochs,
                    "maximum_epoch_budget": args.max_epochs,
                    "active_epoch_budget": active_epoch_budget,
                    "extension_triggered": extension_triggered,
                    "extension_reason": extension_reason,
                    "seed": args.seed,
                    "recipe": recipe,
                    "best_epoch": best_epoch,
                    "best_dev": best_dev,
                    "best_checkpoint": str(best_path.resolve()),
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "rng_state": capture_rng_state(),
                    "train_generator_state": train_generator.get_state(),
                    "history": history,
                },
                last_path,
            )
        logging.info(
            "Epoch %d/%d - Dev WA %.2f UA %.2f WF1 %.2f - "
            "best Dev WF1 %.2f at epoch %d - UTEC %.4f "
            "(AT %.4f AV %.4f TV %.4f) - test_evaluations=0",
            epoch,
            active_epoch_budget,
            dev_wa,
            dev_ua,
            dev_wf1,
            best_dev["wf1"],
            best_epoch,
            losses["utec"],
            losses["utec_audio_text"],
            losses["utec_audio_video"],
            losses["utec_text_video"],
        )

    ending_hashes = hash_code_files()
    if ending_hashes != code_hashes:
        changed = sorted(
            name
            for name in CODE_FILES
            if ending_hashes.get(name) != code_hashes.get(name)
        )
        raise RuntimeError(f"source files changed during training: {changed}")
    if best_epoch < 1 or not best_path.is_file():
        raise RuntimeError("no best dev checkpoint was created")

    epochs_completed = len(history)
    if epochs_completed not in (args.epochs, args.max_epochs):
        raise RuntimeError(
            f"unexpected adaptive-budget completion epoch: {epochs_completed}"
        )
    if not last_path.is_file():
        raise FileNotFoundError(last_path)
    ended_at = datetime.now()
    dev_result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "dev_complete",
        "scientific_status": "MELD_dev_selected_test_deferred",
        "run_name": args.run_name,
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "protocol": {
            "name": "trimef_meld_dev_selected_v1",
            "epochs_completed": epochs_completed,
            "initial_epoch_budget": args.epochs,
            "maximum_epoch_budget": args.max_epochs,
            "extension_triggered": extension_triggered,
            "extension_reason": extension_reason,
            "selection_split": "dev",
            "selection_metric": "wf1",
            "dev_evaluations": epochs_completed,
            "test_evaluations": 0,
            "test_used_for_model_selection": False,
            "test_labels_consumed_by_forward": False,
        },
        "configuration": {
            **recipe,
            "seed": args.seed,
            "epochs": epochs_completed,
            "batch_size": args.batch_size,
            "learning_rate": LEARNING_RATE,
            "ctc_weight": CTC_WEIGHT,
            "lambda_utec": recipe["lambda_utec"],
            "lambda_expert_ce": DECISION_MOE_CONFIGURATION["expert_loss_weight"],
            "num_emotions": len(dataset.emotion_to_id),
            "emotion_mapping": dataset.emotion_to_id,
            "architecture_audit": architecture_audit,
        },
        "locked_recipe": recipe,
        "split": actual_counts,
        "best_epoch": best_epoch,
        "best_dev": best_dev,
        "checkpoint": str(best_path.resolve()),
        "checkpoint_sha256": sha256_file(best_path),
        "last_checkpoint": str(last_path.resolve()),
        "last_checkpoint_sha256": sha256_file(last_path),
        "history": history,
        "all_active_budget_epochs_completed": True,
        "stop_reason": (
            "planned_50_epoch_budget"
            if not extension_triggered
            else "planned_extension_to_100_epochs"
        ),
        "source_hashes": {
            **code_hashes,
            "manifest": sha256_file(paths["manifest"]),
        },
        "paths": {name: str(path) for name, path in paths.items()},
        "environment": environment_record(device),
    }
    atomic_json_save(dev_result, output_dir / "dev_results.json")
    atomic_json_save(
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "dev_complete",
            "run_name": args.run_name,
            "epoch": epochs_completed,
            "initial_epoch_budget": args.epochs,
            "maximum_epoch_budget": args.max_epochs,
            "extension_triggered": extension_triggered,
            "extension_reason": extension_reason,
            "dev_evaluations": epochs_completed,
            "test_evaluations": 0,
            "best_epoch": best_epoch,
            "best_dev": best_dev,
            "dev_result": str((output_dir / "dev_results.json").resolve()),
        },
        output_dir / "progress.json",
    )
    logging.info(
        "Dev training complete: epochs=%d best epoch=%d "
        "Dev WA=%.2f UA=%.2f WF1=%.2f; formal test remains untouched",
        epochs_completed,
        best_epoch,
        best_dev["wa"],
        best_dev["ua"],
        best_dev["wf1"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
