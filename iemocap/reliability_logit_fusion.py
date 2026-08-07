"""Observability-aware logit fusion used by TriMEF."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


LOGIT_FUSION_SOURCES = ("learned", "expert_add", "expert_mix")


def first_full_evidence_epoch(warmup_epochs, ramp_epochs):

    warmup_epochs = int(warmup_epochs)
    ramp_epochs = int(ramp_epochs)
    if warmup_epochs < 0 or ramp_epochs < 0:
        raise ValueError("warmup_epochs and ramp_epochs must be non-negative")
    if ramp_epochs > 0:
        return max(1, warmup_epochs + ramp_epochs)
    return max(1, warmup_epochs + 1)


def scheduled_evidence_scale(epoch, target_scale, warmup_epochs, ramp_epochs):

    epoch = int(epoch)
    warmup_epochs = int(warmup_epochs)
    ramp_epochs = int(ramp_epochs)
    target_scale = float(target_scale)
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    if warmup_epochs < 0 or ramp_epochs < 0:
        raise ValueError("warmup_epochs and ramp_epochs must be non-negative")
    if not math.isfinite(target_scale) or target_scale < 0.0:
        raise ValueError("target_scale must be finite and non-negative")
    if epoch == 0 or epoch <= warmup_epochs:
        return 0.0
    if ramp_epochs == 0:
        return target_scale
    ramp_step = min(epoch - warmup_epochs, ramp_epochs)
    return target_scale * float(ramp_step) / float(ramp_epochs)


class MaskedAttentionPool(nn.Module):

    def __init__(self, input_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens, mask):
        if tokens.dim() != 3 or mask.shape != tokens.shape[:2]:
            raise ValueError("tokens must be [B,T,D] and mask must be [B,T]")
        valid = mask.bool().any(dim=1)
        safe_mask = mask.bool().clone()
        safe_mask[~valid, 0] = True
        clean_tokens = tokens.masked_fill(~mask.bool().unsqueeze(-1), 0.0)
        scores = self.score(clean_tokens).squeeze(-1)
        scores = scores.masked_fill(~safe_mask, -1e4)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(clean_tokens * weights.unsqueeze(-1), dim=1)
        pooled = pooled * valid.to(tokens.dtype).unsqueeze(-1)
        return pooled, weights, valid


class ReliabilityAwareLogitResidual(nn.Module):

    def __init__(
        self,
        num_classes=4,
        at_dim=1536,
        vt_dim=1536,
        video_dim=768,
        embed_dim=256,
        hidden_dim=256,
        dropout=0.2,
        video_dropout=0.35,
        residual_scale=1.0,
        gate_bias=-2.0,
        fusion_source="learned",
        evidence_temperature=2.0,
        evidence_target_scale=0.25,
        evidence_warmup_epochs=5,
        evidence_ramp_epochs=5,
    ):
        super().__init__()
        if not 0.0 <= video_dropout < 1.0:
            raise ValueError("video_dropout must be in [0, 1)")
        if not math.isfinite(float(residual_scale)) or residual_scale < 0.0:
            raise ValueError("residual_scale must be finite and non-negative")
        if not math.isfinite(float(gate_bias)):
            raise ValueError("gate_bias must be finite")
        if fusion_source not in LOGIT_FUSION_SOURCES:
            raise ValueError(f"fusion_source must be one of {LOGIT_FUSION_SOURCES}")
        if (
            not math.isfinite(float(evidence_temperature))
            or float(evidence_temperature) <= 0.0
        ):
            raise ValueError("evidence_temperature must be finite and positive")
        if (
            not math.isfinite(float(evidence_target_scale))
            or float(evidence_target_scale) < 0.0
        ):
            raise ValueError("evidence_target_scale must be finite and non-negative")
        if fusion_source == "expert_mix" and float(evidence_target_scale) > 1.0:
            raise ValueError("expert_mix evidence_target_scale must be at most 1")
        if int(evidence_warmup_epochs) != evidence_warmup_epochs:
            raise ValueError("evidence_warmup_epochs must be an integer")
        if int(evidence_ramp_epochs) != evidence_ramp_epochs:
            raise ValueError("evidence_ramp_epochs must be an integer")
        if int(evidence_warmup_epochs) < 0 or int(evidence_ramp_epochs) < 0:
            raise ValueError("evidence warmup/ramp epochs must be non-negative")

        self.num_classes = int(num_classes)
        self.video_dropout = float(video_dropout)
        self.residual_scale = float(residual_scale)
        self.fusion_source = fusion_source
        self.evidence_temperature = float(evidence_temperature)
        self.evidence_target_scale = float(evidence_target_scale)
        self.evidence_warmup_epochs = int(evidence_warmup_epochs)
        self.evidence_ramp_epochs = int(evidence_ramp_epochs)
        self.register_buffer(
            "current_evidence_scale", torch.tensor(0.0), persistent=True
        )

        self.at_projection = nn.Sequential(
            nn.LayerNorm(at_dim), nn.Linear(at_dim, embed_dim), nn.GELU()
        )
        self.vt_projection = nn.Sequential(
            nn.LayerNorm(vt_dim), nn.Linear(vt_dim, embed_dim), nn.GELU()
        )
        self.video_pool = MaskedAttentionPool(video_dim, hidden_dim=128, dropout=0.1)
        self.video_projection = nn.Sequential(
            nn.LayerNorm(video_dim), nn.Linear(video_dim, embed_dim), nn.GELU()
        )
        self.video_classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(embed_dim, self.num_classes)
        )

        residual_input_dim = 5 * embed_dim
        self.residual_head = nn.Sequential(
            nn.LayerNorm(residual_input_dim),
            nn.Linear(residual_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_classes),
        )

        gate_input_dim = residual_input_dim + 2 * self.num_classes + 7
        self.gate_head = nn.Sequential(
            nn.LayerNorm(gate_input_dim),
            nn.Linear(gate_input_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, self.num_classes),
        )

        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, float(gate_bias))

    @property
    def evidence_selection_active_epoch(self):

        return first_full_evidence_epoch(
            self.evidence_warmup_epochs, self.evidence_ramp_epochs
        )

    def set_evidence_epoch(self, epoch):

        scale = scheduled_evidence_scale(
            epoch=epoch,
            target_scale=self.evidence_target_scale,
            warmup_epochs=self.evidence_warmup_epochs,
            ramp_epochs=self.evidence_ramp_epochs,
        )
        self.current_evidence_scale.fill_(scale)
        return scale

    def _video_log_evidence(self, video_logits):

        log_probabilities = F.log_softmax(
            video_logits.detach() / self.evidence_temperature, dim=-1
        )

        return (log_probabilities + math.log(self.num_classes)).clamp_min(-4.0)

    def _expert_fused_logits(self, at_logits, video_logits, keep):

        scale = float(self.current_evidence_scale.item())
        if scale == 0.0:

            return at_logits, self._video_log_evidence(video_logits)

        active = keep.bool().unsqueeze(-1)
        if self.fusion_source == "expert_add":
            evidence = self._video_log_evidence(video_logits)
            candidate = at_logits + scale * evidence
        elif self.fusion_source == "expert_mix":
            p_at = torch.softmax(at_logits.detach(), dim=-1)
            p_video = torch.softmax(
                video_logits.detach() / self.evidence_temperature, dim=-1
            )
            p_mix = (1.0 - scale) * p_at + scale * p_video
            eps = torch.finfo(p_mix.dtype).tiny
            correction = p_mix.clamp_min(eps).log() - p_at.clamp_min(eps).log()
            evidence = correction
            candidate = at_logits + correction
        else:
            raise AssertionError("expert fusion called for the learned source")

        return torch.where(active, candidate, at_logits), evidence

    @staticmethod
    def _entropy(probabilities):
        eps = torch.finfo(probabilities.dtype).eps
        return -(probabilities * probabilities.clamp_min(eps).log()).sum(
            dim=-1, keepdim=True
        )

    @staticmethod
    def _masked_temporal_variance(tokens, mask):
        weights = mask.to(tokens.dtype).unsqueeze(-1)
        clean_tokens = tokens.masked_fill(~mask.bool().unsqueeze(-1), 0.0)
        count = weights.sum(dim=1).clamp_min(1.0)
        mean = (clean_tokens * weights).sum(dim=1) / count
        variance = ((clean_tokens - mean.unsqueeze(1)).square() * weights).sum(
            dim=1
        ) / count
        return variance.mean(dim=-1, keepdim=True)

    def forward(
        self,
        at_pooled,
        at_logits,
        vt_pooled,
        video_tokens,
        video_mask,
        force_drop_video=False,
    ):
        if at_logits.size(-1) != self.num_classes:
            raise ValueError("at_logits has the wrong class dimension")

        video_pooled, video_attention, video_valid = self.video_pool(
            video_tokens, video_mask
        )

        at_embedding = self.at_projection(at_pooled.detach())
        vt_embedding = self.vt_projection(vt_pooled)
        video_embedding = self.video_projection(video_pooled)
        video_logits = self.video_classifier(video_embedding)

        residual_input = torch.cat(
            (
                at_embedding,
                vt_embedding,
                video_embedding,
                torch.abs(at_embedding - vt_embedding),
                at_embedding * vt_embedding,
            ),
            dim=-1,
        )
        delta_logits = self.residual_head(residual_input)

        p_at = torch.softmax(at_logits.detach(), dim=-1)
        p_video = torch.softmax(video_logits.detach(), dim=-1)
        mixture = 0.5 * (p_at + p_video)
        eps = torch.finfo(p_at.dtype).eps
        js_divergence = 0.5 * (
            (p_at * (p_at.clamp_min(eps).log() - mixture.clamp_min(eps).log())).sum(
                dim=-1, keepdim=True
            )
            + (
                p_video * (p_video.clamp_min(eps).log() - mixture.clamp_min(eps).log())
            ).sum(dim=-1, keepdim=True)
        )
        entropy_scale = math.log(self.num_classes)
        entropy_at = self._entropy(p_at) / entropy_scale
        entropy_video = self._entropy(p_video) / entropy_scale
        at_vt_cosine = F.cosine_similarity(
            at_embedding, vt_embedding, dim=-1
        ).unsqueeze(-1)
        at_video_cosine = F.cosine_similarity(
            at_embedding, video_embedding, dim=-1
        ).unsqueeze(-1)
        coverage = video_mask.to(video_tokens.dtype).mean(dim=1, keepdim=True)
        temporal_variance = torch.log1p(
            self._masked_temporal_variance(video_tokens, video_mask)
        )
        reliability_stats = torch.cat(
            (
                entropy_at,
                entropy_video,
                js_divergence,
                at_vt_cosine,
                at_video_cosine,
                coverage,
                temporal_variance,
            ),
            dim=-1,
        )
        gate_input = torch.cat(
            (residual_input, p_at, p_video, reliability_stats), dim=-1
        )
        raw_gate = torch.sigmoid(self.gate_head(gate_input))

        keep = video_valid.to(at_logits.dtype)
        if force_drop_video:
            keep = torch.zeros_like(keep)
        elif self.training and self.video_dropout > 0.0:
            keep = keep * (torch.rand_like(keep) >= self.video_dropout).to(keep.dtype)
        effective_gate = raw_gate * keep.unsqueeze(-1)
        if self.fusion_source == "learned":
            fused_logits = (
                at_logits + self.residual_scale * effective_gate * delta_logits
            )
            reported_delta = delta_logits
        else:
            fused_logits, reported_delta = self._expert_fused_logits(
                at_logits=at_logits,
                video_logits=video_logits,
                keep=keep,
            )

            effective_gate = keep.unsqueeze(-1).expand_as(at_logits)

        return {
            "fused_logits": fused_logits,
            "at_logits": at_logits,
            "video_logits": video_logits,
            "delta_logits": reported_delta,
            "video_gate": effective_gate,
            "raw_video_gate": raw_gate,
            "video_valid": video_valid,
            "fusion_video_valid": keep.bool(),
            "conflict": js_divergence.squeeze(-1),
            "video_attention": video_attention,
            "logit_fusion_source": self.fusion_source,
            "video_evidence_scale": self.current_evidence_scale.clone(),
        }


def reliability_auxiliary_losses(outputs, labels, rank_margin=0.0):

    zero = outputs["fused_logits"].new_zeros(())
    video_valid = outputs["video_valid"]
    fusion_valid = outputs["fusion_video_valid"]

    video_loss = zero
    if video_valid.any():
        video_loss = F.cross_entropy(
            outputs["video_logits"][video_valid], labels[video_valid]
        )

    consistency_loss = zero
    rank_loss = zero
    if fusion_valid.any():
        fused_logits = outputs["fused_logits"][fusion_valid]
        live_at_logits = outputs["at_logits"][fusion_valid]
        at_logits = live_at_logits.detach()

        visual_residual = fused_logits - live_at_logits
        protected_fused_logits = at_logits + visual_residual
        selected_labels = labels[fusion_valid]

        per_sample_kl = F.kl_div(
            F.log_softmax(protected_fused_logits, dim=-1),
            F.softmax(at_logits, dim=-1),
            reduction="none",
        ).sum(dim=-1)
        conflict = outputs["conflict"][fusion_valid].detach()
        consistency_loss = (conflict * per_sample_kl).sum() / conflict.sum().clamp_min(
            1e-6
        )

        true_at = at_logits.gather(1, selected_labels.unsqueeze(1)).squeeze(1)
        true_fused = protected_fused_logits.gather(
            1, selected_labels.unsqueeze(1)
        ).squeeze(1)
        class_mask = F.one_hot(
            selected_labels, num_classes=protected_fused_logits.size(-1)
        ).bool()
        other_at = at_logits.masked_fill(class_mask, -1e4).max(dim=-1).values
        other_fused = (
            protected_fused_logits.masked_fill(class_mask, -1e4).max(dim=-1).values
        )
        at_margin = true_at - other_at
        fused_margin = true_fused - other_fused
        rank_loss = F.relu(at_margin + float(rank_margin) - fused_margin).mean()

    return {
        "video_aux": video_loss,
        "conflict_consistency": consistency_loss,
        "no_harm_rank": rank_loss,
    }
