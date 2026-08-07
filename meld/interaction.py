"""Cross-modal interaction modules used by TriMEF."""

from typing import Dict

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

    if tokens.dim() != 3 or mask.shape != tokens.shape[:2]:
        raise ValueError("tokens and mask must have shapes (B,T,D) and (B,T)")
    weights = mask.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(-1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return (tokens * weights).sum(dim=1) / denominator


class JointTrimodalAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        num_layers: int = 6,
        feedforward_size: int = 3584,
        dropout: float = 0.1,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        if hidden_size <= 0 or hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be positive and divisible by num_heads")
        if num_layers < 1 or feedforward_size < hidden_size:
            raise ValueError("invalid joint Transformer depth or feed-forward size")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")

        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.feedforward_size = int(feedforward_size)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.modality_embeddings = nn.Parameter(torch.empty(3, hidden_size))
        nn.init.normal_(self.modality_embeddings, mean=0.0, std=0.02)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_heads,
                    dim_feedforward=feedforward_size,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_size)

    def _run_layer(
        self,
        layer: nn.Module,
        hidden: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.training and self.gradient_checkpointing and hidden.requires_grad:
            return checkpoint(
                lambda value: layer(
                    value,
                    src_key_padding_mask=key_padding_mask,
                ),
                hidden,
                use_reentrant=False,
            )
        return layer(hidden, src_key_padding_mask=key_padding_mask)

    def forward(
        self,
        text_tokens: torch.Tensor,
        audio_tokens: torch.Tensor,
        video_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        text_target_mask: torch.Tensor,
        audio_mask: torch.Tensor,
        video_mask: torch.Tensor,
        video_quality: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size = text_tokens.size(0)
        if not (
            audio_tokens.size(0) == video_tokens.size(0) == batch_size
            and text_tokens.size(-1)
            == audio_tokens.size(-1)
            == video_tokens.size(-1)
            == self.hidden_size
        ):
            raise ValueError("all modalities must be (B,T,hidden_size)")
        if text_mask.shape != text_tokens.shape[:2]:
            raise ValueError("text_mask shape mismatch")
        if text_target_mask.shape != text_tokens.shape[:2]:
            raise ValueError("text_target_mask shape mismatch")
        if audio_mask.shape != audio_tokens.shape[:2]:
            raise ValueError("audio_mask shape mismatch")
        if video_mask.shape != video_tokens.shape[:2]:
            raise ValueError("video_mask shape mismatch")
        if video_quality.numel() != batch_size:
            raise ValueError("video_quality must provide one value per sample")

        dtype = text_tokens.dtype
        device = text_tokens.device
        quality = video_quality.to(device=device, dtype=dtype).view(-1, 1, 1)
        quality = quality.clamp(0.0, 1.0)
        video_valid = video_mask.bool().any(dim=1, keepdim=True).unsqueeze(-1)
        quality = quality * video_valid.to(dtype)

        text_input = text_tokens + self.modality_embeddings[0].to(dtype=dtype)
        audio_input = audio_tokens + self.modality_embeddings[1].to(dtype=dtype)

        video_input = quality * (
            video_tokens + self.modality_embeddings[2].to(dtype=dtype)
        )
        hidden = torch.cat((text_input, audio_input, video_input), dim=1)
        valid_mask = torch.cat(
            (text_mask.bool(), audio_mask.bool(), video_mask.bool()), dim=1
        )
        key_padding_mask = ~valid_mask

        for layer in self.layers:
            hidden = self._run_layer(layer, hidden, key_padding_mask)
        hidden = self.output_norm(hidden)

        text_length = text_tokens.size(1)
        audio_length = audio_tokens.size(1)
        text_output = hidden[:, :text_length]
        audio_output = hidden[:, text_length : text_length + audio_length]
        video_output = hidden[:, text_length + audio_length :]

        text_pooled = masked_mean(text_output, text_target_mask)
        audio_pooled = masked_mean(audio_output, audio_mask)
        video_pooled = masked_mean(video_output, video_mask)
        reliability = quality.view(batch_size, 1)
        pooled = (text_pooled + audio_pooled + reliability * video_pooled) / (
            2.0 + reliability
        )
        return {
            "text_tokens": text_output,
            "audio_tokens": audio_output,
            "video_tokens": video_output,
            "pooled": pooled,
            "video_reliability": reliability,
        }


class TextAnchorResidual(nn.Module):

    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.audio_projection = nn.Linear(hidden_size, hidden_size)
        self.video_projection = nn.Linear(hidden_size, hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size)
        nn.init.zeros_(self.audio_projection.weight)
        nn.init.zeros_(self.audio_projection.bias)
        nn.init.zeros_(self.video_projection.weight)
        nn.init.zeros_(self.video_projection.bias)

    def forward(
        self,
        text_tokens: torch.Tensor,
        audio_context: torch.Tensor,
        video_context: torch.Tensor,
        text_target_mask: torch.Tensor,
        video_quality: torch.Tensor,
        video_valid: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if not (
            text_tokens.shape == audio_context.shape == video_context.shape
            and text_target_mask.shape == text_tokens.shape[:2]
        ):
            raise ValueError("text-anchor tensors have incompatible shapes")
        quality = (
            video_quality.to(device=text_tokens.device, dtype=text_tokens.dtype)
            .view(-1, 1, 1)
            .clamp(0.0, 1.0)
        )
        quality = quality * video_valid.to(text_tokens.dtype).view(-1, 1, 1)
        target = text_target_mask.to(text_tokens.dtype).unsqueeze(-1)
        audio_delta = self.audio_projection(audio_context)
        video_delta = self.video_projection(video_context)
        candidate = self.output_norm(
            text_tokens + target * (audio_delta + quality * video_delta)
        )

        fused = text_tokens + target * (candidate - text_tokens)
        return {
            "tokens": fused,
            "audio_delta": audio_delta,
            "video_delta": video_delta,
            "video_reliability": quality.squeeze(-1),
        }
