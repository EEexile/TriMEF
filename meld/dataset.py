#!/usr/bin/env python3
"""MELD data loading utilities for TriMEF."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import RobertaTokenizer


MELD_EMOTIONS = (
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
)


def build_causal_context_tokens(
    frame: pd.DataFrame,
    tokenizer: RobertaTokenizer,
    max_text_length: int,
) -> dict:

    required = {
        "file",
        "split",
        "dialogue_id",
        "utterance_id",
        "text",
        "speaker",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing context columns: {sorted(missing)}")
    if max_text_length < 8:
        raise ValueError("max_text_length must be at least 8")

    working = frame.copy()
    working["row_index"] = np.arange(len(working))
    working["dialogue_id"] = working["dialogue_id"].astype(int)
    working["utterance_id"] = working["utterance_id"].astype(int)
    bos_id = int(tokenizer.bos_token_id)
    eos_id = int(tokenizer.eos_token_id)
    pad_id = int(tokenizer.pad_token_id)

    def encode(text: str) -> list[int]:
        return list(tokenizer.encode(str(text), add_special_tokens=False))

    current_marker = encode("[Current] <SELF>")
    same_marker = encode("<SELF>")
    other_marker = encode("<OTHER>")
    if not current_marker or not same_marker or not other_marker:
        raise ValueError("context markers produced no RoBERTa tokens")

    input_ids = [None] * len(working)
    attention_masks = [None] * len(working)
    target_masks = [None] * len(working)
    context_counts = [0] * len(working)
    for _, group in working.groupby(["split", "dialogue_id"], sort=False):
        group = group.sort_values(["utterance_id", "file"], kind="stable")
        records = group.to_dict("records")
        for position, row in enumerate(records):
            current_text = encode(row["text"])
            available = max_text_length - 2
            current_segment = (current_marker + current_text)[:available]
            combined = list(current_segment)
            context_count = 0
            for previous in reversed(records[:position]):
                role_marker = (
                    same_marker
                    if str(previous["speaker"]) == str(row["speaker"])
                    else other_marker
                )
                previous_segment = (
                    role_marker + encode(previous["text"]) + [eos_id, eos_id]
                )
                if len(previous_segment) + len(combined) > available:
                    break
                combined = previous_segment + combined
                context_count += 1

            current_start = 1 + len(combined) - len(current_segment)
            ids = [bos_id] + combined + [eos_id]
            attention = [1] * len(ids)
            target = [0] * len(ids)
            target[current_start : current_start + len(current_segment)] = [1] * len(
                current_segment
            )
            padding = max_text_length - len(ids)
            ids.extend([pad_id] * padding)
            attention.extend([0] * padding)
            target.extend([0] * padding)
            index = int(row["row_index"])
            input_ids[index] = torch.tensor(ids, dtype=torch.long)
            attention_masks[index] = torch.tensor(attention, dtype=torch.long)
            target_masks[index] = torch.tensor(target, dtype=torch.long)
            context_counts[index] = context_count

    if any(value is None for value in input_ids):
        raise AssertionError("context construction did not cover every row")
    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "target_mask": target_masks,
        "context_count": context_counts,
    }


class MELDMMERDataset(Dataset):

    def __init__(
        self,
        csv_path: str,
        roberta_path: str,
        max_audio_duration: int = 10,
        sample_rate: int = 16000,
        max_text_length: int = 128,
        video_feature_dir: str | None = None,
        use_dialogue_context: bool = True,
        require_standard_splits: bool = True,
    ):
        self.df = pd.read_csv(csv_path)
        required = {
            "file",
            "split",
            "emotion",
            "text",
            "speaker",
            "dialogue_id",
            "utterance_id",
            "audio_path",
        }
        missing = required.difference(self.df.columns)
        if missing:
            raise ValueError(f"MELD manifest missing columns: {sorted(missing)}")
        if self.df["file"].astype(str).duplicated().any():
            raise ValueError("MELD manifest contains duplicate file IDs")
        if require_standard_splits and set(self.df["split"].astype(str)) != {
            "train",
            "dev",
            "test",
        }:
            raise ValueError("MELD manifest must contain train/dev/test")

        self.max_audio_len = int(max_audio_duration * sample_rate)
        self.sample_rate = int(sample_rate)
        self.max_text_length = int(max_text_length)
        self.video_feature_dir = (
            None if video_feature_dir is None else Path(video_feature_dir)
        )
        self.tokenizer = RobertaTokenizer.from_pretrained(roberta_path)
        observed = set(self.df["emotion"].astype(str))
        if not observed.issubset(set(MELD_EMOTIONS)):
            raise ValueError(f"unexpected MELD emotion labels: {sorted(observed)}")
        self.emotion_to_id = {
            emotion: index for index, emotion in enumerate(MELD_EMOTIONS)
        }
        self.id_to_emotion = {
            index: emotion for emotion, index in self.emotion_to_id.items()
        }
        self.vocab = ["<blank>"]
        self.vocab.extend(chr(value) for value in range(ord("a"), ord("z") + 1))
        self.vocab.extend([" ", ",", ".", "?", "!", "'"])
        self.char_to_id = {
            character: index for index, character in enumerate(self.vocab)
        }
        self.id_to_char = {
            index: character for character, index in self.char_to_id.items()
        }
        self.context_tokens = (
            build_causal_context_tokens(
                self.df,
                self.tokenizer,
                self.max_text_length,
            )
            if use_dialogue_context
            else None
        )

    def __len__(self) -> int:
        return len(self.df)

    def text_to_indices(self, text: str) -> list[int]:
        return [
            self.char_to_id[character]
            for character in str(text).lower().strip()
            if character in self.char_to_id
        ]

    def __getitem__(self, index: int) -> dict:
        row = self.df.iloc[index]
        file_id = str(row["file"])
        audio_path = Path(str(row["audio_path"]))
        try:
            waveform, sample_rate = torchaudio.load(str(audio_path))
        except Exception as error:
            raise RuntimeError(f"failed to load audio {audio_path}: {error}") from error
        if sample_rate != self.sample_rate:
            waveform = torchaudio.transforms.Resample(
                sample_rate,
                self.sample_rate,
            )(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)
        if waveform.numel() == 0 or not torch.isfinite(waveform).all():
            raise ValueError(f"invalid audio waveform: {audio_path}")
        audio_length = min(int(waveform.shape[0]), self.max_audio_len)
        if waveform.shape[0] > self.max_audio_len:
            waveform = waveform[: self.max_audio_len]
        else:
            waveform = torch.nn.functional.pad(
                waveform,
                (0, self.max_audio_len - waveform.shape[0]),
                value=0.0,
            )

        if self.context_tokens is None:
            encoded = self.tokenizer(
                str(row["text"]),
                max_length=self.max_text_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = encoded["input_ids"].squeeze(0)
            text_attention_mask = encoded["attention_mask"].squeeze(0)
            text_target_mask = text_attention_mask.clone()
            context_count = 0
        else:
            text_input_ids = self.context_tokens["input_ids"][index]
            text_attention_mask = self.context_tokens["attention_mask"][index]
            text_target_mask = self.context_tokens["target_mask"][index]
            context_count = self.context_tokens["context_count"][index]

        if self.video_feature_dir is None:
            video_features = torch.zeros(16, 512, dtype=torch.float32)
            video_attention_mask = torch.zeros(16, dtype=torch.long)
            video_quality = torch.tensor(0.0, dtype=torch.float32)
        else:
            feature_path = self.video_feature_dir / f"{file_id}.pt"
            if not feature_path.is_file():
                raise FileNotFoundError(feature_path)
            payload = torch.load(feature_path, map_location="cpu")
            if str(payload.get("file_id")) != file_id:
                raise ValueError(f"feature identity mismatch: {feature_path}")
            video_features = payload["tokens"].float()
            video_attention_mask = payload["attention_mask"].long()
            video_quality = torch.as_tensor(
                payload["quality"],
                dtype=torch.float32,
            ).reshape(())
            if tuple(video_features.shape) != (16, 512):
                raise ValueError(f"invalid feature shape: {feature_path}")
            if tuple(video_attention_mask.shape) != (16,):
                raise ValueError(f"invalid mask shape: {feature_path}")
            if not torch.isfinite(video_features).all():
                raise FloatingPointError(feature_path)
            if not torch.all((video_attention_mask == 0) | (video_attention_mask == 1)):
                raise ValueError(f"non-binary video mask: {feature_path}")
            if (
                not torch.isfinite(video_quality)
                or not 0.0 <= float(video_quality) <= 1.0
            ):
                raise ValueError(f"invalid video quality: {feature_path}")
            expected_available = int(row.get("video_available", 0)) == 1
            if bool(video_attention_mask.any()) != expected_available:
                raise ValueError(
                    f"manifest/feature availability mismatch: {feature_path}"
                )

        emotion_label = self.emotion_to_id[str(row["emotion"])]
        asr_indices = self.text_to_indices(str(row["text"]))
        return {
            "audio_input": waveform,
            "audio_length": audio_length,
            "text_input_ids": text_input_ids,
            "text_attention_mask": text_attention_mask,
            "text_target_mask": text_target_mask,
            "context_utterance_count": context_count,
            "video_features": video_features,
            "video_attention_mask": video_attention_mask,
            "video_quality": video_quality,
            "emotion_label": torch.tensor(emotion_label, dtype=torch.long),
            "asr_labels": torch.tensor(asr_indices, dtype=torch.long),
            "text": str(row["text"]),
            "file_id": file_id,
        }


def collate_fn_mmer(batch: list[dict]) -> dict:
    audio_inputs = torch.stack([item["audio_input"] for item in batch])
    audio_lengths = torch.tensor(
        [item["audio_length"] for item in batch],
        dtype=torch.long,
    )
    text_input_ids = torch.stack([item["text_input_ids"] for item in batch])
    text_attention_mask = torch.stack([item["text_attention_mask"] for item in batch])
    text_target_mask = torch.stack([item["text_target_mask"] for item in batch])
    context_counts = torch.tensor(
        [item["context_utterance_count"] for item in batch],
        dtype=torch.long,
    )
    video_features = torch.stack([item["video_features"] for item in batch])
    video_attention_mask = torch.stack([item["video_attention_mask"] for item in batch])
    video_quality = torch.stack([item["video_quality"] for item in batch])
    emotion_labels = torch.stack([item["emotion_label"] for item in batch])
    asr_labels = [item["asr_labels"] for item in batch]
    asr_lengths = torch.tensor(
        [len(label) for label in asr_labels],
        dtype=torch.long,
    )
    max_length = max(int(asr_lengths.max()), 1)
    padded = torch.zeros(len(batch), max_length, dtype=torch.long)
    for index, label in enumerate(asr_labels):
        if len(label):
            padded[index, : len(label)] = label
    return {
        "audio_inputs": audio_inputs,
        "audio_lengths": audio_lengths,
        "text_input_ids": text_input_ids,
        "text_attention_mask": text_attention_mask,
        "text_target_mask": text_target_mask,
        "context_utterance_counts": context_counts,
        "video_features": video_features,
        "video_attention_mask": video_attention_mask,
        "video_quality": video_quality,
        "emotion_labels": emotion_labels,
        "asr_labels": padded,
        "asr_lengths": asr_lengths,
        "texts": [item["text"] for item in batch],
        "file_ids": [item["file_id"] for item in batch],
    }
