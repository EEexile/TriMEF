"""IEMOCAP data loading utilities for TriMEF."""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchaudio
import os
import numpy as np
import re
from transformers import RobertaTokenizer


UTTERANCE_RE = re.compile(r"^(?P<dialogue>.+)_(?P<role>[FM])(?P<turn>\d+)$")


def build_causal_context_tokens(frame, tokenizer, max_text_length):

    if max_text_length < 8:
        raise ValueError("max_text_length must be at least 8")
    required = {"file", "text", "speaker"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing context columns: {sorted(missing)}")

    parsed = frame["file"].astype(str).map(UTTERANCE_RE.fullmatch)
    if parsed.isna().any():
        bad = frame.loc[parsed.isna(), "file"].astype(str).tolist()[:5]
        raise ValueError(f"unparseable IEMOCAP utterance ids: {bad}")

    working = frame.copy()
    working["row_index"] = np.arange(len(working))
    working["dialogue"] = parsed.map(lambda match: match.group("dialogue"))
    working["turn"] = parsed.map(lambda match: int(match.group("turn")))

    bos_id = int(tokenizer.bos_token_id)
    eos_id = int(tokenizer.eos_token_id)
    pad_id = int(tokenizer.pad_token_id)

    def encode(text):
        return list(tokenizer.encode(str(text), add_special_tokens=False))

    current_marker = encode("[Current] <SELF>")
    same_marker = encode("<SELF>")
    other_marker = encode("<OTHER>")
    if not current_marker or not same_marker or not other_marker:
        raise ValueError("context markers produced no RoBERTa tokens")

    all_input_ids = [None] * len(working)
    all_attention_masks = [None] * len(working)
    all_target_masks = [None] * len(working)
    all_context_counts = [0] * len(working)

    for _, rows in working.groupby("dialogue", sort=False):
        rows = rows.sort_values(["turn", "file"], kind="stable")
        records = rows.to_dict("records")
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
            input_ids = [bos_id] + combined + [eos_id]
            attention_mask = [1] * len(input_ids)
            target_mask = [0] * len(input_ids)
            target_end = current_start + len(current_segment)
            target_mask[current_start:target_end] = [1] * len(current_segment)

            padding = max_text_length - len(input_ids)
            input_ids.extend([pad_id] * padding)
            attention_mask.extend([0] * padding)
            target_mask.extend([0] * padding)

            row_index = int(row["row_index"])
            all_input_ids[row_index] = torch.tensor(input_ids, dtype=torch.long)
            all_attention_masks[row_index] = torch.tensor(
                attention_mask, dtype=torch.long
            )
            all_target_masks[row_index] = torch.tensor(target_mask, dtype=torch.long)
            all_context_counts[row_index] = context_count

    if any(value is None for value in all_input_ids):
        raise AssertionError("context construction did not cover every row")
    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "target_mask": all_target_masks,
        "context_count": all_context_counts,
    }


class IEMOCAPMMERDataset(Dataset):

    def __init__(
        self,
        csv_path,
        audio_root_dir,
        roberta_path,
        max_audio_duration=10,
        sample_rate=16000,
        max_text_length=128,
        video_feature_dir=None,
        use_dialogue_context=False,
    ):

        self.df = pd.read_csv(csv_path)
        self.audio_root = audio_root_dir
        self.max_audio_len = int(max_audio_duration * sample_rate)
        self.sample_rate = sample_rate
        self.max_text_length = max_text_length
        self.video_feature_dir = video_feature_dir
        self.use_dialogue_context = bool(use_dialogue_context)

        print(f"Loading RoBERTa tokenizer from: {roberta_path}")
        self.tokenizer = RobertaTokenizer.from_pretrained(roberta_path)

        unique_emotions = sorted(self.df["emotion"].unique())
        self.emotion_to_id = {emo: idx for idx, emo in enumerate(unique_emotions)}
        self.id_to_emotion = {idx: emo for emo, idx in self.emotion_to_id.items()}

        self.vocab = self._build_vocab()
        self.char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self.id_to_char = {idx: char for char, idx in self.char_to_id.items()}

        print(f"Dataset loaded: {len(self.df)} samples")
        print(f"Emotions ({len(self.emotion_to_id)}): {self.emotion_to_id}")
        print(f"CTC Vocab size: {len(self.vocab)}")
        if self.video_feature_dir is not None:
            print(f"DFER-CLIP feature directory: {self.video_feature_dir}")
        self.context_tokens = None
        if self.use_dialogue_context:
            self.context_tokens = build_causal_context_tokens(
                self.df, self.tokenizer, self.max_text_length
            )
            counts = np.asarray(self.context_tokens["context_count"])
            print(
                "Causal role-aware text context enabled: "
                f"max_length={self.max_text_length}, "
                f"mean_previous={counts.mean():.2f}, "
                f"coverage={(counts > 0).mean():.3f}"
            )

    def _build_vocab(self):

        vocab = ["<blank>"]
        vocab.extend([chr(i) for i in range(ord("a"), ord("z") + 1)])
        vocab.extend([" ", ",", ".", "?", "!", "'"])
        return vocab

    def text_to_indices(self, text):

        text = text.lower().strip()
        indices = []
        for char in text:
            if char in self.char_to_id:
                indices.append(self.char_to_id[char])
        return indices

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_id = row["file"]
        text = row["text"]
        emotion = row["emotion"]

        audio_path = os.path.join(self.audio_root, f"{file_id}.wav")

        try:
            waveform, sr = torchaudio.load(audio_path)

            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            waveform = waveform.squeeze(0)

        except Exception as e:
            raise RuntimeError(f"Failed to load audio {audio_path}: {e}") from e

        if waveform.numel() == 0 or not torch.isfinite(waveform).all():
            raise ValueError(f"Invalid audio waveform: {audio_path}")

        original_audio_length = waveform.shape[0]

        if waveform.shape[0] > self.max_audio_len:
            waveform = waveform[: self.max_audio_len]
            original_audio_length = self.max_audio_len
        else:
            pad_len = self.max_audio_len - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len), value=0.0)

        if self.context_tokens is None:
            text_encoding = self.tokenizer(
                text,
                max_length=self.max_text_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_encoding["input_ids"].squeeze(0)
            text_attention_mask = text_encoding["attention_mask"].squeeze(0)
            text_target_mask = text_attention_mask.clone()
            context_utterance_count = 0
        else:
            text_input_ids = self.context_tokens["input_ids"][idx]
            text_attention_mask = self.context_tokens["attention_mask"][idx]
            text_target_mask = self.context_tokens["target_mask"][idx]
            context_utterance_count = self.context_tokens["context_count"][idx]

        if self.video_feature_dir is None:
            video_features = torch.zeros(16, 512, dtype=torch.float32)
            video_attention_mask = torch.zeros(16, dtype=torch.long)
            video_quality = torch.tensor(0.0, dtype=torch.float32)
        else:
            video_path = os.path.join(self.video_feature_dir, f"{file_id}.pt")
            if not os.path.isfile(video_path):
                raise FileNotFoundError(f"Missing DFER-CLIP feature: {video_path}")
            video_data = torch.load(video_path, map_location="cpu")
            if str(video_data.get("file_id")) != str(file_id):
                raise ValueError(
                    f"{video_path} payload file_id={video_data.get('file_id')} "
                    f"does not match requested {file_id}"
                )
            video_features = video_data["tokens"].float()
            video_attention_mask = video_data["attention_mask"].long()
            video_quality = torch.as_tensor(
                video_data["quality"], dtype=torch.float32
            ).reshape(())
            if tuple(video_features.shape) != (16, 512):
                raise ValueError(
                    f"{video_path} has shape {tuple(video_features.shape)}, expected (16, 512)"
                )
            if tuple(video_attention_mask.shape) != (16,):
                raise ValueError(
                    f"{video_path} has mask shape {tuple(video_attention_mask.shape)}, expected (16,)"
                )
            if not torch.isfinite(video_features).all():
                raise FloatingPointError(f"Non-finite DFER-CLIP feature: {video_path}")
            if not torch.all((video_attention_mask == 0) | (video_attention_mask == 1)):
                raise ValueError(f"Non-binary DFER-CLIP mask: {video_path}")
            if (
                not torch.isfinite(video_quality)
                or not 0.0 <= float(video_quality) <= 1.0
            ):
                raise ValueError(f"Invalid DFER-CLIP quality: {video_path}")

        emotion_label = self.emotion_to_id[emotion]

        asr_indices = self.text_to_indices(text)

        return {
            "audio_input": waveform,
            "audio_length": original_audio_length,
            "text_input_ids": text_input_ids,
            "text_attention_mask": text_attention_mask,
            "text_target_mask": text_target_mask,
            "context_utterance_count": context_utterance_count,
            "video_features": video_features,
            "video_attention_mask": video_attention_mask,
            "video_quality": video_quality,
            "emotion_label": torch.tensor(emotion_label, dtype=torch.long),
            "asr_labels": torch.tensor(asr_indices, dtype=torch.long),
            "text": text,
            "file_id": file_id,
        }


def collate_fn_mmer(batch):

    audio_inputs = torch.stack([item["audio_input"] for item in batch])
    audio_lengths = torch.tensor(
        [item["audio_length"] for item in batch], dtype=torch.long
    )

    text_input_ids = torch.stack([item["text_input_ids"] for item in batch])
    text_attention_mask = torch.stack([item["text_attention_mask"] for item in batch])
    text_target_mask = torch.stack([item["text_target_mask"] for item in batch])
    context_utterance_counts = torch.tensor(
        [item["context_utterance_count"] for item in batch], dtype=torch.long
    )

    video_features = torch.stack([item["video_features"] for item in batch])
    video_attention_mask = torch.stack([item["video_attention_mask"] for item in batch])
    video_quality = torch.stack([item["video_quality"] for item in batch])

    emotion_labels = torch.stack([item["emotion_label"] for item in batch])

    asr_labels = [item["asr_labels"] for item in batch]
    asr_lengths = torch.tensor([len(label) for label in asr_labels], dtype=torch.long)

    if len(asr_labels) > 0 and max(asr_lengths) > 0:
        max_asr_len = max(asr_lengths)
        asr_labels_padded = torch.zeros(len(batch), max_asr_len, dtype=torch.long)
        for i, label in enumerate(asr_labels):
            if len(label) > 0:
                asr_labels_padded[i, : len(label)] = label
    else:
        asr_labels_padded = torch.zeros(len(batch), 1, dtype=torch.long)

    texts = [item["text"] for item in batch]
    file_ids = [item["file_id"] for item in batch]

    return {
        "audio_inputs": audio_inputs,
        "audio_lengths": audio_lengths,
        "text_input_ids": text_input_ids,
        "text_attention_mask": text_attention_mask,
        "text_target_mask": text_target_mask,
        "context_utterance_counts": context_utterance_counts,
        "video_features": video_features,
        "video_attention_mask": video_attention_mask,
        "video_quality": video_quality,
        "emotion_labels": emotion_labels,
        "asr_labels": asr_labels_padded,
        "asr_lengths": asr_lengths,
        "texts": texts,
        "file_ids": file_ids,
    }


def get_mmer_dataloaders(
    csv_path,
    audio_dir,
    roberta_path,
    batch_size=8,
    val_split=0.2,
    num_workers=0,
    video_feature_dir=None,
):

    dataset = IEMOCAPMMERDataset(
        csv_path,
        audio_dir,
        roberta_path,
        video_feature_dir=video_feature_dir,
    )

    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_mmer,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_mmer,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Train samples: {train_size}, Val samples: {val_size}")

    return train_loader, val_loader, dataset
