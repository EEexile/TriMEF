"""Audio-video alignment used by TriMEF."""

import math
from typing import Tuple

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
    pooled = clean_hidden.sum(dim=1) / denominator
    return pooled, valid


class SupervisedAudioVideoEmotionAlignment(nn.Module):

    def __init__(
        self,
        audio_dim: int = 768,
        video_dim: int = 768,
        projection_dim: int = 256,
        temperature: float = 0.07,
        min_valid_pairs: int = 2,
    ) -> None:
        super().__init__()
        if audio_dim <= 0 or video_dim <= 0 or projection_dim <= 0:
            raise ValueError(
                "audio_dim, video_dim, and projection_dim must be positive"
            )
        if not math.isfinite(float(temperature)) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if min_valid_pairs < 2:
            raise ValueError("min_valid_pairs must be at least 2")

        self.temperature = float(temperature)
        self.min_valid_pairs = int(min_valid_pairs)
        self.audio_projector = nn.Linear(audio_dim, projection_dim)
        self.video_projector = nn.Linear(video_dim, projection_dim)

    def _differentiable_zero(
        self,
        audio_hidden: torch.Tensor,
        video_hidden: torch.Tensor,
    ) -> torch.Tensor:

        zero = (
            torch.nan_to_num(audio_hidden, nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0
        )
        zero = (
            zero
            + torch.nan_to_num(video_hidden, nan=0.0, posinf=0.0, neginf=0.0).sum()
            * 0.0
        )

        zero = zero + self.audio_projector.weight.sum() * 0.0
        zero = zero + self.video_projector.weight.sum() * 0.0
        if self.audio_projector.bias is not None:
            zero = zero + self.audio_projector.bias.sum() * 0.0
        if self.video_projector.bias is not None:
            zero = zero + self.video_projector.bias.sum() * 0.0
        return zero

    @staticmethod
    def _multi_positive_nce(
        logits: torch.Tensor,
        positive_mask: torch.Tensor,
    ) -> torch.Tensor:

        if logits.ndim != 2 or positive_mask.shape != logits.shape:
            raise ValueError("logits and positive_mask must be equal-shaped matrices")
        if not bool(positive_mask.any(dim=1).all()):
            raise ValueError("each contrastive anchor must have at least one positive")

        positive_count = positive_mask.sum(dim=1, keepdim=True)
        target = positive_mask.to(dtype=logits.dtype) / positive_count.to(
            dtype=logits.dtype
        )
        log_probabilities = F.log_softmax(logits, dim=1)
        return -(target * log_probabilities).sum(dim=1).mean()

    def forward(
        self,
        audio_hidden: torch.Tensor,
        video_hidden: torch.Tensor,
        audio_attention_mask: torch.Tensor,
        video_attention_mask: torch.Tensor,
        emotion_labels: torch.Tensor,
    ) -> torch.Tensor:

        if audio_hidden.size(0) != video_hidden.size(0):
            raise ValueError("audio_hidden and video_hidden batch sizes must match")
        batch_size = audio_hidden.size(0)
        if emotion_labels.ndim != 1 or emotion_labels.size(0) != batch_size:
            raise ValueError(
                "emotion_labels must have shape (batch,); "
                f"received {tuple(emotion_labels.shape)} for batch={batch_size}"
            )
        if emotion_labels.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise ValueError("emotion_labels must use an integer dtype")
        if audio_hidden.device != video_hidden.device:
            raise ValueError("audio_hidden and video_hidden must be on the same device")

        audio_pooled, audio_valid = masked_mean_pool(audio_hidden, audio_attention_mask)
        video_pooled, video_valid = masked_mean_pool(video_hidden, video_attention_mask)
        valid = audio_valid & video_valid
        if int(valid.sum().item()) < self.min_valid_pairs:
            return self._differentiable_zero(audio_hidden, video_hidden)

        pooled_valid = torch.cat((audio_pooled[valid], video_pooled[valid]), dim=-1)
        if not bool(torch.isfinite(pooled_valid).all()):
            raise ValueError(
                "AVEL received non-finite values at valid audio/video tokens"
            )

        labels = emotion_labels.to(device=audio_hidden.device)[valid]

        if int(labels.unique().numel()) < 2:
            return self._differentiable_zero(audio_hidden, video_hidden)

        audio_embedding = F.normalize(
            self.audio_projector(audio_pooled[valid]).float(),
            p=2,
            dim=-1,
            eps=1e-8,
        )
        video_embedding = F.normalize(
            self.video_projector(video_pooled[valid]).float(),
            p=2,
            dim=-1,
            eps=1e-8,
        )

        logits = torch.matmul(audio_embedding, video_embedding.transpose(0, 1))
        logits = logits / self.temperature
        positive_mask = labels[:, None].eq(labels[None, :])

        loss_audio_to_video = self._multi_positive_nce(logits, positive_mask)
        loss_video_to_audio = self._multi_positive_nce(
            logits.transpose(0, 1), positive_mask.transpose(0, 1)
        )
        return 0.5 * (loss_audio_to_video + loss_video_to_audio)


AVEL = SupervisedAudioVideoEmotionAlignment
