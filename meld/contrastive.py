"""Contrastive learning modules used by TriMEF."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleContrastiveLearning(nn.Module):

    def __init__(self, hidden_size=768, projection_dim=256, temperature=0.07):
        super(SimpleContrastiveLearning, self).__init__()

        self.temperature = temperature

        self.audio_projector = nn.Linear(hidden_size, projection_dim)
        self.text_projector = nn.Linear(hidden_size, projection_dim)

    def forward(
        self, audio_hidden, text_hidden, audio_attention_mask, text_attention_mask
    ):

        audio_mask = audio_attention_mask.unsqueeze(-1).float()
        text_mask = text_attention_mask.unsqueeze(-1).float()

        audio_utterance = (audio_hidden * audio_mask).sum(1) / audio_mask.sum(1).clamp(
            min=1e-9
        )
        text_utterance = (text_hidden * text_mask).sum(1) / text_mask.sum(1).clamp(
            min=1e-9
        )

        audio_proj = self.audio_projector(audio_utterance)
        text_proj = self.text_projector(text_utterance)

        audio_proj = F.normalize(audio_proj, p=2, dim=1)
        text_proj = F.normalize(text_proj, p=2, dim=1)

        logits = torch.matmul(audio_proj, text_proj.t()) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)

        loss = (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
        ) / 2.0

        return loss


class VideoTextContrastiveLearning(nn.Module):

    def __init__(self, hidden_size=768, projection_dim=256, temperature=0.07):
        super(VideoTextContrastiveLearning, self).__init__()
        self.temperature = temperature
        self.video_projector = nn.Linear(hidden_size, projection_dim)
        self.text_projector = nn.Linear(hidden_size, projection_dim)

    def forward(
        self,
        video_hidden,
        text_hidden,
        video_attention_mask,
        text_attention_mask,
        video_quality=None,
    ):
        valid = video_attention_mask.bool().any(dim=1)
        if int(valid.sum()) < 2:
            return video_hidden.sum() * 0.0

        video_hidden = video_hidden[valid]

        text_hidden = text_hidden[valid].detach()
        video_mask = video_attention_mask[valid].unsqueeze(-1).float()
        text_mask = text_attention_mask[valid].unsqueeze(-1).float()

        video_utterance = (video_hidden * video_mask).sum(1) / video_mask.sum(1).clamp(
            min=1.0
        )
        text_utterance = (text_hidden * text_mask).sum(1) / text_mask.sum(1).clamp(
            min=1.0
        )

        video_proj = F.normalize(self.video_projector(video_utterance), p=2, dim=1)
        text_proj = F.normalize(self.text_projector(text_utterance), p=2, dim=1)
        logits = torch.matmul(video_proj, text_proj.t()) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)

        loss_video = F.cross_entropy(logits, labels, reduction="none")
        loss_text = F.cross_entropy(logits.t(), labels, reduction="none")
        per_sample = (loss_video + loss_text) / 2.0
        if video_quality is None:
            return per_sample.mean()

        weights = video_quality[valid].detach().float().clamp(0.0, 1.0)
        return (per_sample * weights).sum() / weights.sum().clamp(min=1e-6)
