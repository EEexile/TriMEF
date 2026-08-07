"""Construct the paper configuration of TriMEF."""

from model import MMERModel
from training import resolve_project_path


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
        or (not model.use_utec)
        or (not model.use_decision_moe)
        or (model.interaction_variant != "joint_attention")
    ):
        raise AssertionError("constructed model violates the TriMEF recipe")
    if model.decision_moe.aggregation != "fixed_equal_available_mean":
        raise AssertionError("uniform expert aggregation was not activated")
    if any(
        (parameter.requires_grad for parameter in model.decision_moe.gate.parameters())
    ):
        raise AssertionError("uniform expert gate parameters must be frozen")
    if not 0.95 <= model.interaction_parameter_ratio_vs_reference <= 1.05:
        raise AssertionError(
            f"JointAttn interaction core must be parameter matched within 5%; ratio={model.interaction_parameter_ratio_vs_reference:.6f}"
        )
    forbidden = (
        "audio2text_attention",
        "text2text_attention",
        "text2video_attention",
        "video2text_attention",
        "video_text2text_attention",
        "video_residual_gate",
    )
    if any((hasattr(model, name) for name in forbidden)):
        raise AssertionError("legacy directional interaction modules remain active")
    return model


def model_audit(model):
    freeze = model.set_audio_text_trainable(True)
    video_trainable = sum(
        (
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name.startswith(model._video_parameter_prefixes)
            and parameter.requires_grad
        )
    )
    inherited_ser_trainable = sum(
        (
            parameter.numel()
            for module in (model.ser_attention_pooling, model.classifier)
            for parameter in module.parameters()
            if parameter.requires_grad
        )
    )
    utec_trainable = sum(
        (
            parameter.numel()
            for parameter in model.unified_emotion_contrastive.parameters()
            if parameter.requires_grad
        )
    )
    decision_moe_trainable = sum(
        (
            parameter.numel()
            for parameter in model.decision_moe.parameters()
            if parameter.requires_grad
        )
    )
    decision_gate_trainable = sum(
        (
            parameter.numel()
            for parameter in model.decision_moe.gate.parameters()
            if parameter.requires_grad
        )
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
