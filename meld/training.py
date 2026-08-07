"""Training and evaluation utilities for TriMEF."""

import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch

from decision_moe import EXPERT_NAMES, decision_moe_expert_loss
from reliability_logit_fusion import reliability_auxiliary_losses

PROJECT_ROOT = Path(__file__).resolve().parent


def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logging.info(f"Random seed set to {seed} for reproducibility")


def compute_metrics(predictions, labels, num_classes):

    correct = (predictions == labels).sum()
    total = len(labels)
    wa = correct / total
    unweighted_correct = [0] * num_classes
    unweighted_total = [0] * num_classes
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes
    for i in range(len(labels)):
        true_label = labels[i]
        pred_label = predictions[i]
        unweighted_total[true_label] += 1
        if pred_label == true_label:
            unweighted_correct[true_label] += 1
            tp[true_label] += 1
        else:
            fp[pred_label] += 1
            fn[true_label] += 1
    ua_per_class = []
    for c in range(num_classes):
        if unweighted_total[c] > 0:
            ua_per_class.append(unweighted_correct[c] / unweighted_total[c])
    ua = np.mean(ua_per_class) if ua_per_class else 0.0
    f1_scores = []
    for c in range(num_classes):
        if tp[c] + fp[c] == 0:
            precision = 0
        else:
            precision = tp[c] / (tp[c] + fp[c])
        if tp[c] + fn[c] == 0:
            recall = 0
        else:
            recall = tp[c] / (tp[c] + fn[c])
        if precision + recall == 0:
            f1_scores.append(0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))
    wf1 = sum([f1_scores[c] * unweighted_total[c] for c in range(num_classes)]) / sum(
        unweighted_total
    )
    return (wa * 100, ua * 100, wf1 * 100)


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    criterion_ser,
    criterion_ctc,
    lambda_ctc,
    lambda_at,
    lambda_vt,
    device,
    epoch,
    lambda_video_aux=0.0,
    lambda_consistency=0.0,
    lambda_rank=0.0,
    rank_margin=0.0,
    lambda_avel=0.0,
    lambda_decision_moe=0.0,
    lambda_utec=0.0,
):

    model.train()
    if getattr(model, "audio_text_frozen", False):
        model.enforce_frozen_audio_text_eval()
    total_loss = 0
    total_ser_loss = 0
    total_ctc_loss = 0
    total_at_loss = 0
    total_vt_loss = 0
    total_utec_loss = 0
    total_utec_at_loss = 0
    total_utec_av_loss = 0
    total_utec_tv_loss = 0
    total_video_aux_loss = 0
    total_consistency_loss = 0
    total_rank_loss = 0
    total_avel_loss = 0
    total_decision_moe_loss = 0
    total_moe_weights = {name: 0.0 for name in EXPERT_NAMES}
    total_video_gate = 0
    total_conflict = 0
    total_video_evidence_scale = 0
    saw_logit_fusion = False
    saw_avel = bool(getattr(model, "use_avel", False))
    saw_utec = bool(getattr(model, "use_utec", False))
    saw_decision_moe = bool(getattr(model, "use_decision_moe", False))
    num_batches = len(train_loader)
    for batch_idx, batch in enumerate(train_loader):
        audio_inputs = batch["audio_inputs"].to(device)
        audio_lengths = batch["audio_lengths"].to(device)
        text_input_ids = batch["text_input_ids"].to(device)
        text_attention_mask = batch["text_attention_mask"].to(device)
        text_target_mask = batch.get(
            "text_target_mask", batch["text_attention_mask"]
        ).to(device)
        emotion_labels = batch["emotion_labels"].to(device)
        asr_labels = batch["asr_labels"].to(device)
        asr_lengths = batch["asr_lengths"].to(device)
        if model.use_video:
            video_features = batch["video_features"].to(device, non_blocking=True)
            video_attention_mask = batch["video_attention_mask"].to(
                device, non_blocking=True
            )
            video_quality = batch["video_quality"].to(device, non_blocking=True)
        else:
            video_features = None
            video_attention_mask = None
            video_quality = None
        batch_size = audio_inputs.size(0)
        max_audio_len = audio_inputs.size(1)
        audio_attention_mask = torch.zeros(
            batch_size, max_audio_len, dtype=torch.long, device=device
        )
        for i, length in enumerate(audio_lengths):
            audio_attention_mask[i, :length] = 1
        optimizer.zero_grad()
        outputs = model(
            audio_input=audio_inputs,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            text_target_mask=text_target_mask,
            audio_attention_mask=audio_attention_mask,
            mode="train",
            video_features=video_features,
            video_attention_mask=video_attention_mask,
            video_quality=video_quality,
            emotion_labels=emotion_labels,
            rank_margin=rank_margin,
        )
        loss_ser = criterion_ser(outputs["emotion_logits"], emotion_labels)
        ctc_logits = outputs["ctc_logits"].transpose(0, 1)
        ctc_input_lengths = model.wav2vec2._get_feat_extract_output_lengths(
            audio_lengths
        ).long()
        ctc_input_lengths = torch.clamp(ctc_input_lengths, max=ctc_logits.size(0))
        valid_samples = ctc_input_lengths >= asr_lengths
        if valid_samples.sum() > 0:
            loss_ctc = criterion_ctc(
                ctc_logits[:, valid_samples, :].log_softmax(2),
                asr_labels[valid_samples],
                ctc_input_lengths[valid_samples],
                asr_lengths[valid_samples],
            )
        else:
            loss_ctc = torch.tensor(0.0, device=device)
        loss_at = outputs["emotion_logits"].new_zeros(())
        loss_vt = outputs["emotion_logits"].new_zeros(())
        loss_utec = outputs["emotion_logits"].new_zeros(())
        loss_utec_at = outputs["emotion_logits"].new_zeros(())
        loss_utec_av = outputs["emotion_logits"].new_zeros(())
        loss_utec_tv = outputs["emotion_logits"].new_zeros(())
        if outputs["contrastive_outputs"] is not None:
            loss_at = outputs["contrastive_outputs"]["loss_audio_text"]
            loss_vt = outputs["contrastive_outputs"]["loss_video_text"]
            loss_utec = outputs["contrastive_outputs"].get("loss_utec", loss_utec)
            loss_utec_at = outputs["contrastive_outputs"].get(
                "loss_utec_audio_text", loss_utec_at
            )
            loss_utec_av = outputs["contrastive_outputs"].get(
                "loss_utec_audio_video", loss_utec_av
            )
            loss_utec_tv = outputs["contrastive_outputs"].get(
                "loss_utec_text_video", loss_utec_tv
            )
        loss_video_aux = outputs["emotion_logits"].new_zeros(())
        loss_consistency = outputs["emotion_logits"].new_zeros(())
        loss_rank = outputs["emotion_logits"].new_zeros(())
        if outputs.get("logit_fusion_outputs") is not None:
            saw_logit_fusion = True
            auxiliary = outputs.get("logit_auxiliary_losses")
            if auxiliary is None:
                auxiliary = reliability_auxiliary_losses(
                    outputs["logit_fusion_outputs"],
                    emotion_labels,
                    rank_margin=rank_margin,
                )
            loss_video_aux = auxiliary["video_aux"]
            loss_consistency = auxiliary["conflict_consistency"]
            loss_rank = auxiliary["no_harm_rank"]
            total_video_gate += float(
                outputs["logit_fusion_outputs"]["video_gate"].detach().mean()
            )
            total_conflict += float(
                outputs["logit_fusion_outputs"]["conflict"].detach().mean()
            )
            total_video_evidence_scale += float(
                outputs["logit_fusion_outputs"]["video_evidence_scale"].detach()
            )
        loss_avel = outputs.get("loss_audio_video_emotion")
        if loss_avel is None:
            loss_avel = outputs["emotion_logits"].new_zeros(())
        loss_decision_moe = outputs["emotion_logits"].new_zeros(())
        if outputs.get("decision_moe_outputs") is not None:
            loss_decision_moe = decision_moe_expert_loss(
                outputs["decision_moe_outputs"], emotion_labels
            )
            mean_weights = (
                outputs["decision_moe_outputs"]["weights"].detach().mean(dim=0)
            )
            for index, name in enumerate(EXPERT_NAMES):
                total_moe_weights[name] += float(mean_weights[index])
        loss = (
            loss_ser
            + lambda_ctc * loss_ctc
            + lambda_at * loss_at
            + lambda_vt * loss_vt
            + lambda_utec * loss_utec
            + lambda_video_aux * loss_video_aux
            + lambda_consistency * loss_consistency
            + lambda_rank * loss_rank
            + lambda_avel * loss_avel
            + lambda_decision_moe * loss_decision_moe
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                f"Non-finite total loss at epoch={epoch}, batch={batch_idx + 1}"
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0, error_if_nonfinite=True
        )
        optimizer.step()
        total_loss += loss.item()
        total_ser_loss += loss_ser.item()
        total_ctc_loss += loss_ctc.item() if isinstance(loss_ctc, torch.Tensor) else 0
        total_at_loss += loss_at.item()
        total_vt_loss += loss_vt.item()
        total_utec_loss += loss_utec.item()
        total_utec_at_loss += loss_utec_at.item()
        total_utec_av_loss += loss_utec_av.item()
        total_utec_tv_loss += loss_utec_tv.item()
        total_video_aux_loss += loss_video_aux.item()
        total_consistency_loss += loss_consistency.item()
        total_rank_loss += loss_rank.item()
        total_avel_loss += loss_avel.item()
        total_decision_moe_loss += loss_decision_moe.item()
        if (batch_idx + 1) % 50 == 0 or batch_idx + 1 == num_batches:
            avg_loss = total_loss / (batch_idx + 1)
            avg_ser = total_ser_loss / (batch_idx + 1)
            avg_ctc = total_ctc_loss / (batch_idx + 1)
            avg_at = total_at_loss / (batch_idx + 1)
            avg_vt = total_vt_loss / (batch_idx + 1)
            detail = f"Loss: {avg_loss:.4f} (SER: {avg_ser:.4f}, CTC: {avg_ctc:.4f}, AT: {avg_at:.4f}, VT: {avg_vt:.4f}"
            if saw_logit_fusion:
                denominator = batch_idx + 1
                detail += f", V-Aux: {total_video_aux_loss / denominator:.4f}, Cons: {total_consistency_loss / denominator:.4f}, Rank: {total_rank_loss / denominator:.4f}, Gate: {total_video_gate / denominator:.4f}, Conflict: {total_conflict / denominator:.4f}, EvidenceScale: {total_video_evidence_scale / denominator:.4f}"
            if saw_avel:
                detail += f", AVEL: {total_avel_loss / (batch_idx + 1):.4f}"
            if saw_utec:
                denominator = batch_idx + 1
                detail += f", UTEC: {total_utec_loss / denominator:.4f} (AT:{total_utec_at_loss / denominator:.4f}, AV:{total_utec_av_loss / denominator:.4f}, TV:{total_utec_tv_loss / denominator:.4f})"
            if saw_decision_moe:
                denominator = batch_idx + 1
                weight_text = "/".join(
                    (
                        f"{name}:{total_moe_weights[name] / denominator:.3f}"
                        for name in EXPERT_NAMES
                    )
                )
                detail += f", MoEExpert: {total_decision_moe_loss / denominator:.4f}, MoEWeights: {weight_text}"
            logging.info(f"  Epoch {epoch} [{batch_idx + 1}/{num_batches}] {detail})")
    metrics = {
        "total": total_loss / num_batches,
        "ser": total_ser_loss / num_batches,
        "ctc": total_ctc_loss / num_batches,
        "at": total_at_loss / num_batches,
        "vt": total_vt_loss / num_batches,
    }
    if saw_logit_fusion:
        metrics.update(
            {
                "video_aux": total_video_aux_loss / num_batches,
                "conflict_consistency": total_consistency_loss / num_batches,
                "no_harm_rank": total_rank_loss / num_batches,
                "video_gate_mean": total_video_gate / num_batches,
                "conflict_mean": total_conflict / num_batches,
                "video_evidence_scale": total_video_evidence_scale / num_batches,
            }
        )
    if saw_avel:
        metrics["avel"] = total_avel_loss / num_batches
    if saw_utec:
        metrics.update(
            {
                "utec": total_utec_loss / num_batches,
                "utec_audio_text": total_utec_at_loss / num_batches,
                "utec_audio_video": total_utec_av_loss / num_batches,
                "utec_text_video": total_utec_tv_loss / num_batches,
            }
        )
    if saw_decision_moe:
        metrics["decision_moe_expert"] = total_decision_moe_loss / num_batches
        for name in EXPERT_NAMES:
            metrics[f"moe_weight_{name}"] = total_moe_weights[name] / num_batches
    return metrics


@torch.no_grad()
def evaluate(model, val_loader, device, num_emotions):

    model.eval()
    all_preds = []
    all_labels = []
    for batch in val_loader:
        audio_inputs = batch["audio_inputs"].to(device)
        audio_lengths = batch["audio_lengths"].to(device)
        text_input_ids = batch["text_input_ids"].to(device)
        text_attention_mask = batch["text_attention_mask"].to(device)
        text_target_mask = batch.get(
            "text_target_mask", batch["text_attention_mask"]
        ).to(device)
        emotion_labels = batch["emotion_labels"].to(device)
        if model.use_video:
            video_features = batch["video_features"].to(device, non_blocking=True)
            video_attention_mask = batch["video_attention_mask"].to(
                device, non_blocking=True
            )
            video_quality = batch["video_quality"].to(device, non_blocking=True)
        else:
            video_features = None
            video_attention_mask = None
            video_quality = None
        batch_size = audio_inputs.size(0)
        max_audio_len = audio_inputs.size(1)
        audio_attention_mask = torch.zeros(
            batch_size, max_audio_len, dtype=torch.long, device=device
        )
        for i, length in enumerate(audio_lengths):
            audio_attention_mask[i, :length] = 1
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
        if not bool(torch.isfinite(outputs["emotion_logits"]).all()):
            raise FloatingPointError("Non-finite emotion logits during evaluation")
        preds = torch.argmax(outputs["emotion_logits"], dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(emotion_labels.cpu().numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    wa, ua, wf1 = compute_metrics(all_preds, all_labels, num_emotions)
    return (wa, ua, wf1)


def resolve_device(device_arg):

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and (not torch.cuda.is_available()):
        raise RuntimeError(
            f"CUDA device requested ({device_arg}), but CUDA is unavailable"
        )
    return device


def resolve_project_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def atomic_torch_save(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(str(temporary), str(path))


def atomic_json_save(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
