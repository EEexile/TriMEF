#!/usr/bin/env python3
"""Extract frozen DFER-CLIP temporal tokens from 16-frame face tracks."""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dfer-repo", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class FaceTrackDataset(Dataset):
    def __init__(self, metadata, output_root, overwrite=False, limit=0):
        data = pd.read_csv(metadata)
        data = data[data["status"] == "ok"].copy()
        if limit > 0:
            data = data.iloc[:limit]
        self.rows = []
        output_root = Path(output_root)
        for row in data.to_dict("records"):
            output_path = output_root / (str(row["file_id"]) + ".pt")
            if overwrite or not output_path.exists():
                row["output_path"] = str(output_path)
                self.rows.append(row)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        frame_dir = Path(row["frame_dir"])
        images = []
        for frame_index in range(16):
            path = frame_dir / ("%03d.jpg" % frame_index)
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(str(path))
            if image.shape[:2] != (224, 224):
                image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0)
            images.append(tensor)

        valid_mask = torch.tensor(
            [int(value) for value in str(row["valid_mask"]).split()],
            dtype=torch.long,
        )
        if valid_mask.numel() != 16:
            raise ValueError(
                "%s has %d mask entries instead of 16"
                % (row["file_id"], valid_mask.numel())
            )
        return {
            "file_id": str(row["file_id"]),
            "images": torch.stack(images),
            "attention_mask": valid_mask,
            "quality": torch.tensor(float(row["quality"]), dtype=torch.float32),
            "output_path": row["output_path"],
        }


class DFERClipVisualEncoder(torch.nn.Module):
    """Official ViT-B/32 + CLS temporal transformer with token export."""

    def __init__(self, checkpoint, dfer_repo):
        super().__init__()
        repo = str(Path(dfer_repo).resolve())
        repo_models = str(Path(repo) / "models")
        if repo_models not in sys.path:
            sys.path.insert(0, repo_models)
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from clip.model import VisionTransformer
        from models.Temporal_Model import Temporal_Transformer_Cls

        self.image_encoder = VisionTransformer(
            input_resolution=224,
            patch_size=32,
            width=768,
            layers=12,
            heads=12,
            output_dim=512,
        )
        self.temporal_net = Temporal_Transformer_Cls(
            num_patches=16,
            input_dim=512,
            depth=1,
            heads=8,
            mlp_dim=1024,
            dim_head=64,
        )

        state = torch.load(checkpoint, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        visual_prefix = "module.image_encoder."
        temporal_prefix = "module.temporal_net."
        visual_state = {
            key[len(visual_prefix) :]: value
            for key, value in state.items()
            if key.startswith(visual_prefix)
        }
        temporal_state = {
            key[len(temporal_prefix) :]: value
            for key, value in state.items()
            if key.startswith(temporal_prefix)
        }
        if not visual_state or not temporal_state:
            raise RuntimeError("Checkpoint does not contain official DFER-CLIP prefixes")
        self.image_encoder.load_state_dict(visual_state, strict=True)
        self.temporal_net.load_state_dict(temporal_state, strict=True)

        for parameter in self.parameters():
            parameter.requires_grad = False
        self.eval()

    def forward(self, images):
        batch, frames, channels, height, width = images.shape
        if frames != 16:
            raise ValueError("DFER-CLIP expects exactly 16 frames")
        frame_features = self.image_encoder(
            images.reshape(batch * frames, channels, height, width)
        ).reshape(batch, frames, 512)

        cls_tokens = self.temporal_net.cls_token.expand(batch, -1, -1)
        temporal = torch.cat((cls_tokens, frame_features), dim=1)
        temporal = temporal + self.temporal_net.pos_embedding[:, :17]
        temporal = self.temporal_net.temporal_transformer(temporal)
        return temporal[:, 1:, :], temporal[:, 0, :]


def save_feature(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(str(temporary), str(path))


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = DFERClipVisualEncoder(args.checkpoint, args.dfer_repo).to(device)
    dataset = FaceTrackDataset(
        args.metadata,
        args.output_root,
        overwrite=args.overwrite,
        limit=args.limit,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    print("pending utterances:", len(dataset), flush=True)

    completed = 0
    started = time.time()
    with torch.inference_mode():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            tokens, cls_features = model(images)
            tokens = tokens.float().cpu()
            cls_features = cls_features.float().cpu()
            for index, file_id in enumerate(batch["file_id"]):
                payload = {
                    "file_id": file_id,
                    "tokens": tokens[index].contiguous(),
                    "cls": cls_features[index].contiguous(),
                    "attention_mask": batch["attention_mask"][index].long(),
                    "quality": batch["quality"][index].float(),
                    "encoder": "DFER-CLIP FERV39K ViT-B/32",
                    "preprocessing": (
                        "16 official DFER-CLIP test-midpoint frames, square face crop, "
                        "25% margin, RGB / 255"
                    ),
                }
                if not torch.isfinite(payload["tokens"]).all():
                    raise FloatingPointError(file_id + " contains non-finite tokens")
                if not torch.isfinite(payload["cls"]).all():
                    raise FloatingPointError(file_id + " contains a non-finite CLS feature")
                save_feature(batch["output_path"][index], payload)
                completed += 1
            if completed == len(batch["file_id"]) or completed % 50 < len(batch["file_id"]):
                print(
                    "[%d/%d] elapsed=%.1fs"
                    % (completed, len(dataset), time.time() - started),
                    flush=True,
                )

    print("completed:", completed)
    if completed:
        example = torch.load(dataset.rows[0]["output_path"], map_location="cpu")
        print(
            "example shapes:",
            tuple(example["tokens"].shape),
            tuple(example["cls"].shape),
            tuple(example["attention_mask"].shape),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
