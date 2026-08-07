"""Unified trimodal emotion contrastive learning."""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean_pool(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    if hidden.ndim != 3:
        raise ValueError(
            "hidden must have shape (batch, time, channels); "
            f"received {tuple(hidden.shape)}"
        )
    if attention_mask.ndim != 2:
        raise ValueError(
            "attention_mask must have shape (batch, time); "
            f"received {tuple(attention_mask.shape)}"
        )
    if tuple(attention_mask.shape) != tuple(hidden.shape[:2]):
        raise ValueError(
            "attention_mask must match the first two hidden dimensions; "
            f"received hidden={tuple(hidden.shape)}, "
            f"mask={tuple(attention_mask.shape)}"
        )

    mask = attention_mask.to(device=hidden.device).ne(0)
    valid = mask.any(dim=1)
    clean_hidden = hidden.masked_fill(~mask.unsqueeze(-1), 0.0)
    denominator = mask.sum(dim=1, keepdim=True).to(dtype=hidden.dtype).clamp_min(1.0)
    return clean_hidden.sum(dim=1) / denominator, valid


class UnifiedTrimodalEmotionContrastive(nn.Module):

    def __init__(
        self,
        audio_dim: int = 768,
        text_dim: int = 768,
        video_dim: int = 768,
        projection_dim: int = 256,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        dimensions = (audio_dim, text_dim, video_dim, projection_dim)
        if any(int(value) <= 0 for value in dimensions):
            raise ValueError("all feature dimensions must be positive")
        if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
            raise ValueError("temperature must be finite and positive")

        self.temperature = float(temperature)
        self.audio_projector = nn.Linear(audio_dim, projection_dim)
        self.text_projector = nn.Linear(text_dim, projection_dim)
        self.video_projector = nn.Linear(video_dim, projection_dim)

    def _differentiable_zero(self, *tensors: torch.Tensor) -> torch.Tensor:
        zero = self.audio_projector.weight.sum() * 0.0
        zero = zero + self.text_projector.weight.sum() * 0.0
        zero = zero + self.video_projector.weight.sum() * 0.0
        for projector in (
            self.audio_projector,
            self.text_projector,
            self.video_projector,
        ):
            if projector.bias is not None:
                zero = zero + projector.bias.sum() * 0.0
        for tensor in tensors:
            zero = (
                zero
                + torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0
            )
        return zero

    def _directed_loss(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        query_labels: torch.Tensor,
        key_labels: torch.Tensor,
        query_valid: torch.Tensor,
        key_valid: torch.Tensor,
        anchor_weights: Optional[torch.Tensor] = None,
        key_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        query_indices = query_valid.nonzero(as_tuple=False).flatten()
        key_indices = key_valid.nonzero(as_tuple=False).flatten()
        if query_indices.numel() == 0 or key_indices.numel() == 0:
            return self._differentiable_zero(query, key)

        query_selected = query[query_indices]
        key_selected = key[key_indices]
        q_labels = query_labels[query_indices]
        k_labels = key_labels[key_indices]
        logits = torch.matmul(query_selected, key_selected.transpose(0, 1))
        logits = logits / self.temperature

        positive_mask = q_labels[:, None].eq(k_labels[None, :])
        negative_mask = ~positive_mask
        selected_key_weights = None
        if key_weights is not None:
            if key_weights.ndim != 1 or key_weights.size(0) != key.size(0):
                raise ValueError("key_weights must have shape (batch,)")
            selected_key_weights = key_weights.to(
                device=logits.device, dtype=logits.dtype
            )[key_indices].clamp(0.0, 1.0)
            positive_mass = (
                positive_mask.to(logits.dtype) * selected_key_weights.unsqueeze(0)
            ).sum(dim=1)
            negative_mass = (
                negative_mask.to(logits.dtype) * selected_key_weights.unsqueeze(0)
            ).sum(dim=1)
            eligible = positive_mass.gt(0.0) & negative_mass.gt(0.0)
        else:
            eligible = positive_mask.any(dim=1) & negative_mask.any(dim=1)
        if not bool(eligible.any()):
            return self._differentiable_zero(query, key)

        logits = logits[eligible]
        positive_mask = positive_mask[eligible]
        if selected_key_weights is not None:
            key_log_weight = torch.where(
                selected_key_weights.gt(0.0),
                selected_key_weights.log(),
                selected_key_weights.new_full((), -torch.inf),
            )
            logits = logits + key_log_weight.unsqueeze(0)
            target = positive_mask.to(
                dtype=logits.dtype
            ) * selected_key_weights.unsqueeze(0)
        else:
            target = positive_mask.to(dtype=logits.dtype)
        target = target / target.sum(dim=1, keepdim=True)
        log_probabilities = F.log_softmax(logits, dim=1)
        weighted_log_probabilities = torch.where(
            target.gt(0.0),
            target * log_probabilities,
            torch.zeros_like(log_probabilities),
        )
        row_loss = -weighted_log_probabilities.sum(dim=1)

        if anchor_weights is None:
            return row_loss.mean()
        if anchor_weights.ndim != 1 or anchor_weights.size(0) != query.size(0):
            raise ValueError("anchor_weights must have shape (batch,)")
        weights = anchor_weights.to(device=row_loss.device, dtype=row_loss.dtype)[
            query_indices
        ][eligible]
        weights = weights.clamp(0.0, 1.0)
        if not bool(weights.gt(0).any()):
            return self._differentiable_zero(query, key)
        return (row_loss * weights).sum() / weights.sum().clamp_min(1e-6)

    def _symmetric_pair_loss(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
        labels: torch.Tensor,
        first_valid: torch.Tensor,
        second_valid: torch.Tensor,
        reliability: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        first_to_second = self._directed_loss(
            first,
            second,
            labels,
            labels,
            first_valid,
            second_valid,
            key_weights=reliability,
        )
        second_to_first = self._directed_loss(
            second,
            first,
            labels,
            labels,
            second_valid,
            first_valid,
            anchor_weights=reliability,
        )
        return 0.5 * (first_to_second + second_to_first)

    def forward(
        self,
        audio_hidden: torch.Tensor,
        text_hidden: torch.Tensor,
        video_hidden: torch.Tensor,
        audio_attention_mask: torch.Tensor,
        text_attention_mask: torch.Tensor,
        video_attention_mask: torch.Tensor,
        emotion_labels: torch.Tensor,
        video_quality: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        batch_size = audio_hidden.size(0)
        if text_hidden.size(0) != batch_size or video_hidden.size(0) != batch_size:
            raise ValueError("audio, text, and video batch sizes must match")
        if emotion_labels.ndim != 1 or emotion_labels.size(0) != batch_size:
            raise ValueError("emotion_labels must have shape (batch,)")
        if emotion_labels.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise ValueError("emotion_labels must use an integer dtype")

        audio_pooled, audio_valid = masked_mean_pool(audio_hidden, audio_attention_mask)
        text_pooled, text_valid = masked_mean_pool(text_hidden, text_attention_mask)
        video_pooled, video_valid = masked_mean_pool(video_hidden, video_attention_mask)
        pooled = torch.cat((audio_pooled, text_pooled, video_pooled), dim=-1)
        if not bool(torch.isfinite(pooled).all()):
            raise ValueError("UTEC received non-finite values at pooled tokens")

        labels = emotion_labels.to(device=audio_hidden.device)
        audio_embedding = F.normalize(
            self.audio_projector(audio_pooled).float(), p=2, dim=-1, eps=1e-8
        )
        text_embedding = F.normalize(
            self.text_projector(text_pooled).float(), p=2, dim=-1, eps=1e-8
        )
        video_embedding = F.normalize(
            self.video_projector(video_pooled).float(), p=2, dim=-1, eps=1e-8
        )

        if video_quality is None:
            reliability = audio_embedding.new_ones(batch_size)
        else:
            if video_quality.ndim != 1 or video_quality.size(0) != batch_size:
                raise ValueError("video_quality must have shape (batch,)")
            reliability = (
                video_quality.detach()
                .to(
                    device=audio_embedding.device,
                    dtype=audio_embedding.dtype,
                )
                .clamp(0.0, 1.0)
            )
        reliability = reliability * video_valid.to(reliability.dtype)

        loss_at = self._symmetric_pair_loss(
            audio_embedding,
            text_embedding,
            labels,
            audio_valid,
            text_valid,
        )
        loss_av = self._symmetric_pair_loss(
            audio_embedding,
            video_embedding,
            labels,
            audio_valid,
            video_valid,
            reliability,
        )
        loss_tv = self._symmetric_pair_loss(
            text_embedding,
            video_embedding,
            labels,
            text_valid,
            video_valid,
            reliability,
        )
        loss = (loss_at + loss_av + loss_tv) / 3.0
        return {
            "loss": loss,
            "loss_audio_text": loss_at,
            "loss_audio_video": loss_av,
            "loss_text_video": loss_tv,
        }


UTEC = UnifiedTrimodalEmotionContrastive
