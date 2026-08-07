#!/usr/bin/env python3
"""Locked MELD configuration for TriMEF."""

import math
import re


SESSION_IDS = frozenset({1, 2, 3, 4, 5})
PROTOCOL_NAME = "iemocap_loso_test_selected_jointattn_uniform_experts_v1"
SCIENTIFIC_STATUS = "test_selected_development_result"
SELECTION_METRIC = "test_wa"
BATCH_SIZE = 16
DEFAULT_EPOCHS = 100
RECIPE_NAME = "jointattn_utec005_uniform_experts"

UTEC_RECIPE = {
    "recipe": RECIPE_NAME,
    "interaction_variant": "joint_attention",
    "interaction_symmetry": "unrestricted_shared_attention",
    "joint_attention_layers": 6,
    "joint_attention_heads": 12,
    "joint_attention_feedforward_size": 3584,
    "joint_modality_embeddings": True,
    "joint_readout": "equal_modality_mean_with_continuous_video_reliability",
    "legacy_pairwise_interaction_active": False,
    "legacy_video_gate_active": False,
    "video_mode": "full",
    "video_fusion_variant": "scalar",
    "logit_fusion_source": "learned",
    "video_residual_scale": 1.0,
    "video_gate_bias": -2.0,
    "video_gate_groups": 12,
    "video_dropout": 0.35,
    "lambda_at": 0.0,
    "lambda_vt": 0.0,
    "lambda_avel": 0.0,
    "lambda_utec": 0.05,
    "decision_moe_active": True,
    "decision_moe_aggregation": "fixed_equal_available_mean",
    "lambda_expert_ce": 1.0,
    "utec_temperature": 0.07,
    "utec_positive_rule": "all_cross_modal_same_emotion",
    "utec_pair_balance": "equal_unordered_pairs",
    "utec_gradient_policy": "symmetric_end_to_end_no_detach",
    "utec_quality_policy": "continuous_video_reliability_without_cutoff",
    "lambda_video_aux": 0.0,
    "lambda_consistency": 0.0,
    "lambda_rank": 0.0,
    "rank_margin": 0.0,
    "freeze_audio_text": False,
    "no_warm_start": True,
    "initialization_source": "pretrained_encoders",
    "end_to_end": True,
}


def get_recipe():

    return dict(UTEC_RECIPE)


def locked_recipe():

    return get_recipe()


def session_split(fold):

    fold = int(fold)
    if fold not in SESSION_IDS:
        raise ValueError(f"fold must be one of {sorted(SESSION_IDS)}")
    return {
        "train_sessions": sorted(SESSION_IDS - {fold}),
        "model_selection_session": fold,
        "test_session": fold,
    }


def protocol_record(epochs=DEFAULT_EPOCHS):

    epochs = int(epochs)
    if epochs != DEFAULT_EPOCHS:
        raise ValueError(f"JointAttn uniform experts fixes epochs to {DEFAULT_EPOCHS}")
    return {
        "name": PROTOCOL_NAME,
        "held_out_role": "test_and_model_selection",
        "test_set_used_for_model_selection": True,
        "selection_metric": SELECTION_METRIC,
        "selection_direction": "max",
        "selection_scope": "all_epochs",
        "test_evaluations": epochs,
        "epochs_requested": epochs,
        "epochs_completed": epochs,
        "no_early_stopping": True,
        "scientific_status": SCIENTIFIC_STATUS,
    }


def _finite_number(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _same_number(actual, expected):
    return _finite_number(actual) and math.isclose(
        float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12
    )


def _sha256(value):
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def best_history_row(history):

    if not isinstance(history, list) or len(history) != DEFAULT_EPOCHS:
        raise ValueError(f"history must contain exactly {DEFAULT_EPOCHS} epochs")
    best = None
    running_best = -float("inf")
    for expected_epoch, row in enumerate(history, 1):
        if not isinstance(row, dict) or row.get("epoch") != expected_epoch:
            raise ValueError("history epochs must be contiguous and one-based")
        if row.get("test_evaluation_index") != expected_epoch:
            raise ValueError("each epoch must have exactly one ordered test evaluation")
        if row.get("selection_active") is not True:
            raise ValueError("Test WA selection must be active for every epoch")
        if row.get("audio_text_frozen") is not False:
            raise ValueError("UTEC must train audio/text end-to-end")
        if row.get("video_input_active") is not True:
            raise ValueError("UTEC requires video at every epoch")
        train = row.get("train")
        if not isinstance(train, dict) or not all(
            _finite_number(train.get(loss_name))
            for loss_name in (
                "total",
                "ser",
                "ctc",
                "at",
                "vt",
                "utec",
                "utec_audio_text",
                "utec_audio_video",
                "utec_text_video",
                "decision_moe_expert",
            )
        ):
            raise ValueError(
                f"history epoch {expected_epoch} lacks a required finite loss"
            )
        if not _same_number(train.get("at"), 0.0):
            raise ValueError("legacy AT loss must be disabled in UTEC")
        if not _same_number(train.get("vt"), 0.0):
            raise ValueError("legacy VT loss must be disabled in UTEC")
        expert_weights = [
            train.get(f"moe_weight_{name}")
            for name in ("audio", "text", "video", "fused")
        ]
        if not all(
            _finite_number(value) and float(value) >= 0.0 for value in expert_weights
        ):
            raise ValueError("uniform expert weights must be finite and non-negative")
        if not _same_number(sum(float(value) for value in expert_weights), 1.0):
            raise ValueError("uniform expert weights must sum to one")
        test = row.get("test")
        if not isinstance(test, dict) or not all(
            _finite_number(test.get(metric)) for metric in ("wa", "ua", "wf1")
        ):
            raise ValueError(f"history epoch {expected_epoch} has invalid metrics")
        improved = float(test["wa"]) > running_best
        if row.get("improved") is not improved:
            raise ValueError(
                f"history epoch {expected_epoch} has invalid improved flag"
            )
        if improved:
            best = row
            running_best = float(test["wa"])
    return best


def validate_result_contract(
    payload, expected_fold=None, expected_epochs=DEFAULT_EPOCHS
):

    if not isinstance(payload, dict):
        raise ValueError("result must be a dictionary")
    protocol = payload.get("protocol")
    configuration = payload.get("configuration")
    split = payload.get("split")
    history = payload.get("history")
    if not all(isinstance(value, dict) for value in (protocol, configuration, split)):
        raise ValueError("protocol, configuration, and split records are required")
    if not isinstance(history, list):
        raise ValueError("history must be a list")

    failed = []
    required_protocol = {
        "name": PROTOCOL_NAME,
        "held_out_role": "test_and_model_selection",
        "test_set_used_for_model_selection": True,
        "selection_metric": SELECTION_METRIC,
        "selection_direction": "max",
        "selection_scope": "all_epochs",
        "no_early_stopping": True,
        "scientific_status": SCIENTIFIC_STATUS,
    }
    for key, expected in required_protocol.items():
        if protocol.get(key) != expected:
            failed.append(f"protocol.{key}")
    if payload.get("status") != "success":
        failed.append("status")
    if payload.get("scientific_status") != SCIENTIFIC_STATUS:
        failed.append("scientific_status")

    try:
        epochs_requested = int(protocol.get("epochs_requested"))
        epochs_completed = int(protocol.get("epochs_completed"))
        test_evaluations = int(protocol.get("test_evaluations"))
    except (TypeError, ValueError):
        raise ValueError("epoch and test-evaluation counts must be integers")
    if not (epochs_requested == epochs_completed == test_evaluations == DEFAULT_EPOCHS):
        failed.append("epoch_evaluation_counts")
    if len(history) != DEFAULT_EPOCHS:
        failed.append("history_length")
    if expected_epochs is not None and int(expected_epochs) != DEFAULT_EPOCHS:
        failed.append("expected_epochs")

    for key, expected in UTEC_RECIPE.items():
        actual = configuration.get(key)
        if isinstance(expected, float):
            if not _same_number(actual, expected):
                failed.append(f"configuration.{key}")
        elif actual != expected:
            failed.append(f"configuration.{key}")
    if configuration.get("batch_size") != BATCH_SIZE:
        failed.append("configuration.batch_size")
    if configuration.get("epochs") != DEFAULT_EPOCHS:
        failed.append("configuration.epochs")
    utec_audit = configuration.get("utec")
    if not isinstance(utec_audit, dict):
        failed.append("configuration.utec")
    else:
        required_utec_audit = {
            "positive_rule": "same_emotion_cross_modal",
            "negative_rule": "different_emotion_cross_modal",
            "pair_aggregation": "equal_mean",
            "directions": "symmetric",
            "gradient_policy": "end_to_end_no_detach",
            "video_reliability": "continuous_weight_without_cutoff",
            "uses_threshold": False,
            "uses_stage": False,
            "test_labels_consumed_by_forward": False,
        }
        for key, expected in required_utec_audit.items():
            if utec_audit.get(key) != expected:
                failed.append(f"configuration.utec.{key}")
    moe_audit = configuration.get("decision_moe")
    if not isinstance(moe_audit, dict):
        failed.append("configuration.decision_moe")
    else:
        required_moe_audit = {
            "enabled": True,
            "experts": ["audio", "text", "video", "fused"],
            "gate": "fixed_equal_available_mean",
            "gate_trainable": False,
            "expert_loss": "mean_available_cross_entropy",
            "expert_loss_weight": 1.0,
            "test_labels_consumed_by_forward": False,
        }
        for key, expected in required_moe_audit.items():
            if moe_audit.get(key) != expected:
                failed.append(f"configuration.decision_moe.{key}")

    try:
        fold = int(configuration.get("fold"))
    except (TypeError, ValueError):
        raise ValueError("configuration.fold must be an integer")
    expected_split = session_split(fold)
    for key, expected in expected_split.items():
        if split.get(key) != expected:
            failed.append(f"split.{key}")
    if split.get("validation_session") is not None:
        failed.append("split.validation_session")
    if expected_fold is not None and fold != int(expected_fold):
        failed.append("expected_fold")
    try:
        train_count = int(split.get("train_count"))
        test_count = int(split.get("test_count"))
        total_count = int(split.get("total_count"))
    except (TypeError, ValueError):
        raise ValueError("split sample counts must be integers")
    if train_count <= 0 or test_count <= 0 or train_count + test_count != total_count:
        failed.append("split.sample_counts")

    best_row = best_history_row(history)
    if payload.get("best_epoch") != best_row["epoch"]:
        failed.append("best_epoch")
    best_test = payload.get("best_test")
    if not isinstance(best_test, dict):
        failed.append("best_test")
    else:
        for metric in ("wa", "ua", "wf1"):
            if not _same_number(best_test.get(metric), best_row["test"][metric]):
                failed.append(f"best_test.{metric}")
    if payload.get("test") != best_test:
        failed.append("test_alias")
    if payload.get("stop_reason") != "max_epochs":
        failed.append("stop_reason")
    if payload.get("all_epochs_completed") is not True:
        failed.append("all_epochs_completed")
    if payload.get("best_checkpoint_uses_video") is not True:
        failed.append("best_checkpoint_uses_video")

    for path_key, hash_key in (
        ("checkpoint", "checkpoint_sha256"),
        ("last_checkpoint", "last_checkpoint_sha256"),
    ):
        if not isinstance(payload.get(path_key), str) or not payload[path_key]:
            failed.append(path_key)
        if not _sha256(payload.get(hash_key)):
            failed.append(hash_key)
    source_hashes = payload.get("source_hashes")
    if not isinstance(source_hashes, dict) or not source_hashes:
        failed.append("source_hashes")
    elif not all(_sha256(value) for value in source_hashes.values()):
        failed.append("source_hashes.values")

    if failed:
        raise ValueError(
            f"JointAttn uniform-experts result contract failed: "
            f"{sorted(set(failed))}"
        )
    return {
        "fold": fold,
        "recipe": configuration["recipe"],
        "epochs": epochs_completed,
        "test_evaluations": test_evaluations,
        "best_epoch": best_row["epoch"],
        "best_test": dict(best_row["test"]),
    }
