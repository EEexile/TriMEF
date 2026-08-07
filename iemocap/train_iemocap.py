#!/usr/bin/env python3
"""Train and evaluate TriMEF on one held-out IEMOCAP Session."""

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

from dataset import IEMOCAPMMERDataset, collate_fn_mmer
from config import (
    BATCH_SIZE,
    DEFAULT_EPOCHS,
    SCIENTIFIC_STATUS,
    get_recipe,
    protocol_record,
    session_split,
    validate_result_contract,
)
from model import MMERModel
from training import (
    atomic_json_save,
    atomic_torch_save,
    capture_rng_state,
    evaluate,
    resolve_device,
    resolve_project_path,
    set_seed,
    train_one_epoch,
)


PROJECT_ROOT = Path(__file__).resolve().parent
CODE_FILES = (
    "train_iemocap.py",
    "config.py",
    "interaction.py",
    "training.py",
    "model.py",
    "dataset.py",
    "contrastive.py",
    "utec.py",
    "reliability_logit_fusion.py",
    "avel.py",
    "decision_moe.py",
)
RESULT_SCHEMA_VERSION = 1
LEARNING_RATE = 5e-5
CTC_WEIGHT = 0.1
SCHEDULER_PATIENCE = 5
CONTEXT_CONFIGURATION = {
    "name": "causal_role_aware_target_span_text",
    "source": "EMART_ACL_2026",
    "direction": "preceding_only",
    "speaker_markers": ["[Current]", "<SELF>", "<OTHER>"],
    "max_text_length": 128,
    "vocabulary_resized": False,
    "target_span_pooling": True,
    "future_labels_consumed": False,
    "test_labels_consumed_by_forward": False,
}
DECISION_MOE_CONFIGURATION = {
    "name": "decision_level_uniform_expert_ensemble",
    "source": "MiSTER-E_CSL_2026",
    "enabled": True,
    "experts": ["audio", "text", "video", "fused"],
    "gate": "fixed_equal_available_mean",
    "gate_trainable": False,
    "expert_loss": "mean_available_cross_entropy",
    "expert_loss_weight": 1.0,
    "uses_threshold": False,
    "uses_stage": False,
    "test_labels_consumed_by_forward": False,
}
UTEC_CONFIGURATION = {
    "name": "unified_trimodal_emotion_contrastive",
    "positive_rule": "same_emotion_cross_modal",
    "negative_rule": "different_emotion_cross_modal",
    "modality_pairs": ["audio_text", "audio_video", "text_video"],
    "pair_aggregation": "equal_mean",
    "directions": "symmetric",
    "gradient_policy": "end_to_end_no_detach",
    "video_reliability": "continuous_weight_without_cutoff",
    "uses_threshold": False,
    "uses_stage": False,
    "test_labels_consumed_by_forward": False,
}


def sha256_file(path, chunk_size=16 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def hash_code_files():
    hashes = {}
    for name in CODE_FILES:
        path = PROJECT_ROOT / name
        if not path.is_file():
            raise FileNotFoundError(f"required source file is missing: {path}")
        hashes[name] = sha256_file(path)
    return hashes


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "TriMEF: four IEMOCAP Sessions train; held-out "
            "Session Test WA selects the best checkpoint across 100 epochs"
        )
    )
    parser.add_argument("--fold", type=int, required=True, choices=range(1, 6))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--csv-path", default=str(PROJECT_ROOT / "assets" / "train.csv")
    )
    parser.add_argument("--audio-dir", default=str(PROJECT_ROOT / "assets" / "audio"))
    parser.add_argument(
        "--wav2vec2-path", default=str(PROJECT_ROOT / "assets" / "wav2vec2-base")
    )
    parser.add_argument(
        "--roberta-path", default=str(PROJECT_ROOT / "assets" / "roberta-base")
    )
    parser.add_argument(
        "--video-feature-dir",
        default=str(PROJECT_ROOT / "assets" / "dfer_clip_features"),
    )
    args = parser.parse_args(argv)
    if args.epochs != DEFAULT_EPOCHS:
        parser.error(f"TriMEF fixes --epochs to {DEFAULT_EPOCHS}")
    if args.batch_size != BATCH_SIZE:
        parser.error(f"TriMEF fixes --batch-size to {BATCH_SIZE}")
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be positive")
    if args.run_name is not None and not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", args.run_name
    ):
        parser.error("--run-name contains unsupported characters")
    return args


def build_model(args, dataset, recipe):
    model = MMERModel(
        wav2vec2_path=str(resolve_project_path(args.wav2vec2_path)),
        roberta_path=str(resolve_project_path(args.roberta_path)),
        num_emotions=len(dataset.emotion_to_id),
        ctc_vocab_size=len(dataset.vocab),
        freeze_audio_extractor=True,
        use_contrastive=True,
        contrastive_weight=0.0,
        video_feature_dim=512,
        use_video=True,
        use_vt_contrastive=False,
        video_fusion_variant=recipe["video_fusion_variant"],
        video_residual_scale=recipe["video_residual_scale"],
        video_gate_bias=recipe["video_gate_bias"],
        video_gate_groups=recipe["video_gate_groups"],
        video_dropout=recipe["video_dropout"],
        logit_fusion_source=recipe["logit_fusion_source"],
        use_avel=False,
        use_utec=True,
        utec_temperature=recipe["utec_temperature"],
        use_decision_moe=True,
        interaction_variant=recipe["interaction_variant"],
        joint_num_layers=recipe["joint_attention_layers"],
        joint_feedforward_size=recipe["joint_attention_feedforward_size"],
    )
    model.decision_moe.set_aggregation("fixed_equal_available_mean")
    for name, parameter in model.named_parameters():
        if name.startswith("decision_moe.gate."):
            model._base_requires_grad[name] = False
    if (
        not model.use_video
        or model.video_fusion_variant != "scalar"
        or model.use_vt_contrastive
        or model.use_avel
        or not model.use_utec
        or not model.use_decision_moe
        or model.interaction_variant != "joint_attention"
    ):
        raise AssertionError("constructed model violates the TriMEF recipe")
    if model.decision_moe.aggregation != "fixed_equal_available_mean":
        raise AssertionError("uniform expert aggregation was not activated")
    if any(
        parameter.requires_grad for parameter in model.decision_moe.gate.parameters()
    ):
        raise AssertionError("uniform expert gate parameters must be frozen")
    if not 0.95 <= model.interaction_parameter_ratio_vs_reference <= 1.05:
        raise AssertionError(
            "JointAttn interaction core must be parameter matched within 5%; "
            f"ratio={model.interaction_parameter_ratio_vs_reference:.6f}"
        )
    forbidden = (
        "audio2text_attention",
        "text2text_attention",
        "text2video_attention",
        "video2text_attention",
        "video_text2text_attention",
        "video_residual_gate",
    )
    if any(hasattr(model, name) for name in forbidden):
        raise AssertionError("legacy directional interaction modules remain active")
    return model


def make_loader(dataset, indices, batch_size, shuffle, workers, seed=None):
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return (
        DataLoader(
            Subset(dataset, indices),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn_mmer,
            num_workers=workers,
            pin_memory=True,
            generator=generator,
            worker_init_fn=seed_worker,
        ),
        generator,
    )


def environment_record(device):
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


def model_audit(model):
    freeze = model.set_audio_text_trainable(True)
    video_trainable = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith(model._video_parameter_prefixes) and parameter.requires_grad
    )
    inherited_ser_trainable = sum(
        parameter.numel()
        for module in (model.ser_attention_pooling, model.classifier)
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    utec_trainable = sum(
        parameter.numel()
        for parameter in model.unified_emotion_contrastive.parameters()
        if parameter.requires_grad
    )
    decision_moe_trainable = sum(
        parameter.numel()
        for parameter in model.decision_moe.parameters()
        if parameter.requires_grad
    )
    decision_gate_trainable = sum(
        parameter.numel()
        for parameter in model.decision_moe.gate.parameters()
        if parameter.requires_grad
    )
    audit = {
        "use_video": bool(model.use_video),
        "video_fusion_variant": model.video_fusion_variant,
        "use_vt_contrastive": bool(model.use_vt_contrastive),
        "use_avel": bool(model.use_avel),
        "use_utec": bool(model.use_utec),
        "use_decision_moe": bool(model.use_decision_moe),
        "interaction_variant": model.interaction_variant,
        "pairwise_reference_interaction_parameter_count": int(
            model.pairwise_reference_interaction_parameter_count
        ),
        "interaction_parameter_count": int(model.interaction_parameter_count),
        "interaction_parameter_ratio_vs_reference": float(
            model.interaction_parameter_ratio_vs_reference
        ),
        "audio_text_freeze": freeze,
        "video_trainable_parameter_count": int(video_trainable),
        "utec_trainable_parameter_count": int(utec_trainable),
        "decision_moe_trainable_parameter_count": int(decision_moe_trainable),
        "decision_moe_aggregation": model.decision_moe.aggregation,
        "decision_gate_trainable_parameter_count": int(decision_gate_trainable),
        "inherited_ser_trainable_parameter_count": int(inherited_ser_trainable),
    }
    if freeze["audio_text_frozen"] or freeze["audio_text_trainable_count"] <= 0:
        raise AssertionError("audio/text must be trained end-to-end")
    if (
        min(
            video_trainable,
            utec_trainable,
            inherited_ser_trainable,
            decision_moe_trainable,
        )
        <= 0
    ):
        raise AssertionError("UTEC model has a missing trainable component")
    if decision_gate_trainable != 0:
        raise AssertionError("uniform Decision-MoE gate must be frozen")
    return audit


def main(argv=None):
    args = parse_args(argv)
    recipe = get_recipe()
    now = datetime.now()
    run_name = args.run_name or (
        f"trimef_{now:%Y%m%d-%H%M%S-%f}_" f"f{args.fold}_s{args.seed}_p{os.getpid()}"
    )
    output_dir = PROJECT_ROOT / "outputs" / run_name
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / run_name
    log_dir = PROJECT_ROOT / "logs" / run_name
    collisions = [
        path for path in (output_dir, checkpoint_dir, log_dir) if path.exists()
    ]
    if collisions:
        raise FileExistsError(f"run already exists: {collisions}")
    for directory in (output_dir, checkpoint_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=False)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "train.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    code_hashes = hash_code_files()
    disclosed_protocol = protocol_record(args.epochs)
    atomic_json_save(
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "run_name": run_name,
            "started_at": now.isoformat(),
            "arguments": vars(args),
            "locked_recipe": recipe,
            "context_configuration": CONTEXT_CONFIGURATION,
            "decision_moe_configuration": DECISION_MOE_CONFIGURATION,
            "utec_configuration": UTEC_CONFIGURATION,
            "protocol": disclosed_protocol,
            "source_hashes": code_hashes,
        },
        output_dir / "run_config.json",
    )

    paths = {
        "csv": resolve_project_path(args.csv_path),
        "audio": resolve_project_path(args.audio_dir),
        "wav2vec2": resolve_project_path(args.wav2vec2_path),
        "roberta": resolve_project_path(args.roberta_path),
        "video": resolve_project_path(args.video_feature_dir),
    }
    for label, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"missing {label}: {path}")

    set_seed(args.seed)
    device = resolve_device(args.device)
    dataset = IEMOCAPMMERDataset(
        str(paths["csv"]),
        str(paths["audio"]),
        str(paths["roberta"]),
        max_text_length=CONTEXT_CONFIGURATION["max_text_length"],
        video_feature_dir=str(paths["video"]),
        use_dialogue_context=True,
    )
    frame = pd.read_csv(paths["csv"])
    session_indices = {
        session: frame[frame["speaker"].str.startswith(f"Ses0{session}")].index.tolist()
        for session in range(1, 6)
    }
    assigned = [index for indices in session_indices.values() for index in indices]
    if len(assigned) != len(frame) or len(set(assigned)) != len(frame):
        raise ValueError("every CSV row must belong to exactly one IEMOCAP Session")
    split = session_split(args.fold)
    train_indices = [
        index
        for session in split["train_sessions"]
        for index in session_indices[session]
    ]
    test_indices = session_indices[args.fold]
    if set(train_indices) & set(test_indices):
        raise AssertionError("train and held-out test indices overlap")
    if len(train_indices) + len(test_indices) != len(frame):
        raise AssertionError(
            "four-Session train plus held-out Session must use all data"
        )
    missing_video = [
        str(frame.iloc[index]["file"])
        for index in train_indices + test_indices
        if not (paths["video"] / f"{frame.iloc[index]['file']}.pt").is_file()
    ]
    if missing_video:
        raise FileNotFoundError(
            f"missing {len(missing_video)} video features: {missing_video[:5]}"
        )

    train_loader, train_generator = make_loader(
        dataset, train_indices, BATCH_SIZE, True, 4, args.seed
    )
    test_loader, _ = make_loader(dataset, test_indices, BATCH_SIZE, False, 2)
    model = build_model(args, dataset, recipe).to(device)
    architecture_audit = model_audit(model)
    architecture_audit["context_configuration"] = CONTEXT_CONFIGURATION
    architecture_audit["utec_configuration"] = UTEC_CONFIGURATION
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
    best_path = checkpoint_dir / f"fold_{args.fold}_test_wa_best.pt"
    last_path = checkpoint_dir / f"fold_{args.fold}_last.pt"
    best_test = {"wa": -float("inf"), "ua": 0.0, "wf1": 0.0}
    best_epoch = 0
    history = []

    logging.info(
        "TRIMEF IEMOCAP: train Sessions=%s (%d); held-out "
        "Session=%d (%d) is tested and used for selection every epoch",
        split["train_sessions"],
        len(train_indices),
        args.fold,
        len(test_indices),
    )
    logging.info("context configuration: %s", CONTEXT_CONFIGURATION)
    logging.info("decision MoE configuration: %s", DECISION_MOE_CONFIGURATION)
    logging.info("UTEC configuration: %s", UTEC_CONFIGURATION)
    logging.info("UTEC quality policy: %s", recipe["utec_quality_policy"])
    logging.info(
        "run=%s epochs=%d batch=%d seed=%d device=%s; end-to-end, no warm "
        "start, no early stop; best checkpoint selected by Test WA",
        run_name,
        args.epochs,
        BATCH_SIZE,
        args.seed,
        device,
    )
    logging.info("locked recipe: %s", recipe)

    for epoch in range(1, args.epochs + 1):
        lr_used = float(optimizer.param_groups[0]["lr"])
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
        required_utec_losses = (
            "utec",
            "utec_audio_text",
            "utec_audio_video",
            "utec_text_video",
        )
        if not all(
            name in losses and math.isfinite(float(losses[name]))
            for name in required_utec_losses
        ):
            raise RuntimeError("training did not produce finite UTEC losses")
        if abs(float(losses["at"])) > 1e-12 or abs(float(losses["vt"])) > 1e-12:
            raise RuntimeError("legacy AT/VT objectives must be disabled in UTEC")
        if "avel" in losses and abs(float(losses["avel"])) > 1e-12:
            raise RuntimeError("legacy AVEL objective must be disabled in UTEC")
        test_wa, test_ua, test_wf1 = evaluate(
            model, test_loader, device, len(dataset.emotion_to_id)
        )
        scheduler.step(test_wa)
        improved = test_wa > best_test["wa"]
        if improved:
            best_test = {
                "wa": float(test_wa),
                "ua": float(test_ua),
                "wf1": float(test_wf1),
            }
            best_epoch = epoch
            atomic_torch_save(model.state_dict(), best_path)
        history.append(
            {
                "epoch": epoch,
                "train": {key: float(value) for key, value in losses.items()},
                "test": {
                    "wa": float(test_wa),
                    "ua": float(test_ua),
                    "wf1": float(test_wf1),
                },
                "test_evaluation_index": epoch,
                "lr": lr_used,
                "lr_after_scheduler": float(optimizer.param_groups[0]["lr"]),
                "improved": bool(improved),
                "audio_text_frozen": False,
                "video_input_active": True,
                "selection_active": True,
            }
        )
        atomic_json_save(
            {
                "run_name": run_name,
                "status": "running",
                "epoch": epoch,
                "epochs_requested": args.epochs,
                "test_evaluations": epoch,
                "best_epoch": best_epoch,
                "best_test": best_test,
                "history": history,
            },
            output_dir / "progress.json",
        )
        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            atomic_torch_save(
                {
                    "kind": "trimef_iemocap_training_state_v1",
                    "run_name": run_name,
                    "fold": args.fold,
                    "recipe": recipe,
                    "seed": args.seed,
                    "epoch": epoch,
                    "epochs_requested": args.epochs,
                    "best_epoch": best_epoch,
                    "best_test": best_test,
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
            "TriMEF Epoch %d/%d - Test WA %.2f UA %.2f W-F1 %.2f - "
            "best Test WA %.2f at epoch %d - UTEC %.4f "
            "(AT %.4f AV %.4f TV %.4f)",
            epoch,
            args.epochs,
            test_wa,
            test_ua,
            test_wf1,
            best_test["wa"],
            best_epoch,
            losses["utec"],
            losses["utec_audio_text"],
            losses["utec_audio_video"],
            losses["utec_text_video"],
        )

    ended_at = datetime.now()
    ending_code_hashes = hash_code_files()
    if ending_code_hashes != code_hashes:
        changed = sorted(
            name
            for name in CODE_FILES
            if ending_code_hashes.get(name) != code_hashes.get(name)
        )
        raise RuntimeError(f"source files changed during training: {changed}")
    result_protocol = protocol_record(args.epochs)
    result_protocol.update(
        {
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "scheduler": "ReduceLROnPlateau",
            "scheduler_metric": "test_wa",
            "scheduler_patience": SCHEDULER_PATIENCE,
            "scheduler_factor": 0.5,
            "ctc_weight": CTC_WEIGHT,
            "best_checkpoint_selection_includes_all_epochs": True,
        }
    )
    result_split = dict(split)
    result_split.update(
        {
            "validation_session": None,
            "train_count": len(train_indices),
            "test_count": len(test_indices),
            "total_count": len(frame),
        }
    )
    source_hashes = dict(code_hashes)
    source_hashes["csv"] = sha256_file(paths["csv"])
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "success",
        "scientific_status": SCIENTIFIC_STATUS,
        "run_name": run_name,
        "started_at": now.isoformat(),
        "ended_at": ended_at.isoformat(),
        "protocol": result_protocol,
        "configuration": {
            **recipe,
            "context": CONTEXT_CONFIGURATION,
            "decision_moe": DECISION_MOE_CONFIGURATION,
            "utec": UTEC_CONFIGURATION,
            "fold": args.fold,
            "seed": args.seed,
            "fold_seed": args.seed,
            "epochs": args.epochs,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "ctc_weight": CTC_WEIGHT,
            "initialization": {
                "source": "pretrained_encoders",
                "warm_start_checkpoint": None,
                "audio_text_frozen": False,
                "end_to_end": True,
            },
            "model_audit": architecture_audit,
        },
        "split": result_split,
        "best_epoch": best_epoch,
        "best_test": best_test,
        "test": dict(best_test),
        "best_checkpoint_uses_video": True,
        "checkpoint": str(best_path.resolve()),
        "checkpoint_sha256": sha256_file(best_path),
        "last_checkpoint": str(last_path.resolve()),
        "last_checkpoint_sha256": sha256_file(last_path),
        "progress": str((output_dir / "progress.json").resolve()),
        "run_config": str((output_dir / "run_config.json").resolve()),
        "history": history,
        "stop_reason": "max_epochs",
        "all_epochs_completed": True,
        "source_hashes": source_hashes,
        "paths": {key: str(value.resolve()) for key, value in paths.items()},
        "environment": environment_record(device),
    }
    validate_result_contract(
        result,
        expected_fold=args.fold,
        expected_epochs=args.epochs,
    )
    result_path = output_dir / "results.json"
    atomic_json_save(result, result_path)
    atomic_json_save(
        {
            "run_name": run_name,
            "status": "success",
            "epoch": args.epochs,
            "epochs_requested": args.epochs,
            "test_evaluations": args.epochs,
            "best_epoch": best_epoch,
            "best_test": best_test,
            "result": str(result_path.resolve()),
        },
        output_dir / "progress.json",
    )
    logging.info(
        "Completed TriMEF IEMOCAP: best epoch=%d Test WA=%.2f " "UA=%.2f W-F1=%.2f",
        best_epoch,
        best_test["wa"],
        best_test["ua"],
        best_test["wf1"],
    )


if __name__ == "__main__":
    main()
