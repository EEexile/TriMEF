"""Decision-level expert aggregation used by TriMEF."""

import torch
import torch.nn as nn
import torch.nn.functional as F


EXPERT_NAMES = ("audio", "text", "video", "fused")


def _masked_mean(tokens, mask):
    if tokens.dim() != 3:
        raise ValueError("tokens must have shape (batch, time, hidden)")
    if mask.shape != tokens.shape[:2]:
        raise ValueError("mask must match the first two token dimensions")
    weights = mask.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(-1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return (tokens * weights).sum(dim=1) / denominator


class DecisionLevelEmotionMoE(nn.Module):

    def __init__(self, hidden_size, num_classes, dropout=0.2):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_classes = int(num_classes)
        if self.hidden_size < 1 or self.num_classes < 2:
            raise ValueError("hidden_size and num_classes are invalid")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        def make_head():
            return nn.Sequential(
                nn.LayerNorm(self.hidden_size),
                nn.Dropout(float(dropout)),
                nn.Linear(self.hidden_size, self.num_classes),
            )

        self.audio_head = make_head()
        self.text_head = make_head()
        self.video_head = make_head()
        self.gate = nn.Linear(
            len(EXPERT_NAMES) * self.num_classes,
            len(EXPERT_NAMES),
        )

        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.aggregation = "learned_softmax"

    def set_aggregation(self, aggregation):

        aggregation = str(aggregation)
        if aggregation not in {"learned_softmax", "fixed_equal_available_mean"}:
            raise ValueError("unsupported expert aggregation")
        self.aggregation = aggregation
        gate_trainable = aggregation == "learned_softmax"
        for parameter in self.gate.parameters():
            parameter.requires_grad = gate_trainable

    def forward(
        self,
        audio_tokens,
        audio_mask,
        text_tokens,
        text_mask,
        video_tokens,
        video_mask,
        fused_logits,
    ):
        batch_size = fused_logits.size(0)
        if fused_logits.shape != (batch_size, self.num_classes):
            raise ValueError("fused_logits has an invalid shape")
        if any(
            x.size(0) != batch_size for x in (audio_tokens, text_tokens, video_tokens)
        ):
            raise ValueError("all expert inputs must share the batch dimension")

        audio_logits = self.audio_head(_masked_mean(audio_tokens, audio_mask))
        text_logits = self.text_head(_masked_mean(text_tokens, text_mask))
        video_logits = self.video_head(_masked_mean(video_tokens, video_mask))
        expert_logits = torch.stack(
            (audio_logits, text_logits, video_logits, fused_logits),
            dim=1,
        )

        availability = torch.stack(
            (
                audio_mask.bool().any(dim=1),
                text_mask.bool().any(dim=1),
                video_mask.bool().any(dim=1),
                torch.ones(batch_size, dtype=torch.bool, device=fused_logits.device),
            ),
            dim=1,
        )
        if self.aggregation == "fixed_equal_available_mean":
            weights = availability.to(dtype=expert_logits.dtype)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        else:
            gate_scores = self.gate(expert_logits.reshape(batch_size, -1))
            gate_scores = gate_scores.masked_fill(~availability, -1e4)
            weights = F.softmax(gate_scores, dim=1)
        logits = (weights.unsqueeze(-1) * expert_logits).sum(dim=1)
        return {
            "logits": logits,
            "expert_logits": expert_logits,
            "weights": weights,
            "availability": availability,
        }


def decision_moe_expert_loss(outputs, labels):

    expert_logits = outputs["expert_logits"]
    availability = outputs["availability"]
    if expert_logits.dim() != 3:
        raise ValueError("expert_logits must have shape (batch, experts, classes)")
    if availability.shape != expert_logits.shape[:2]:
        raise ValueError("availability must match batch and expert dimensions")
    if labels.shape != (expert_logits.size(0),):
        raise ValueError("labels must have shape (batch,)")

    batch_size, num_experts, num_classes = expert_logits.shape
    per_item = F.cross_entropy(
        expert_logits.reshape(batch_size * num_experts, num_classes),
        labels[:, None].expand(-1, num_experts).reshape(-1),
        reduction="none",
    ).reshape(batch_size, num_experts)
    weights = availability.to(dtype=per_item.dtype)
    return (per_item * weights).sum() / weights.sum().clamp_min(1.0)
