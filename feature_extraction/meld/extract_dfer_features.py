#!/usr/bin/env python3
"""Extract frozen DFER-CLIP tokens for MELD.

Every manifest sample receives exactly one feature payload.  Samples whose
video is unavailable receive deterministic zero tokens, an all-zero mask, and
quality zero; this makes TriMEF exclude the unavailable visual expert.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader, Dataset


SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dfer-repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.num_shards < 1:
        parser.error("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, num-shards)")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def save_feature(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(str(temporary), str(path))


def feature_valid(path: Path, file_id: str) -> bool:
    if not path.is_file():
        return False
    payload = torch.load(path, map_location="cpu")
    return (
        str(payload.get("file_id")) == str(file_id)
        and tuple(payload.get("tokens", torch.empty(0)).shape) == (16, 512)
        and tuple(payload.get("cls", torch.empty(0)).shape) == (512,)
        and tuple(payload.get("attention_mask", torch.empty(0)).shape) == (16,)
        and torch.isfinite(payload["tokens"]).all()
        and torch.isfinite(payload["cls"]).all()
        and torch.isfinite(torch.as_tensor(payload.get("quality"))).all()
    )


class AvailableTrackDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], output_root: Path):
        self.rows = []
        for row in rows:
            if int(row["video_available"]) != 1:
                continue
            output_path = output_root / f"{row['file']}.pt"
            if feature_valid(output_path, row["file"]):
                continue
            if output_path.exists():
                raise RuntimeError(
                    f"invalid pre-existing feature requires manual audit: {output_path}"
                )
            self.rows.append({**row, "output_path": str(output_path)})

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        with Path(row["media_metadata_path"]).open(encoding="utf-8") as handle:
            media = json.load(handle)
        frame_dir = Path(media["frame_dir"])
        images = []
        for frame_index in range(16):
            path = frame_dir / f"{frame_index:03d}.jpg"
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(path)
            if image.shape[:2] != (224, 224):
                raise ValueError(f"unexpected crop shape {image.shape}: {path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = (
                torch.from_numpy(image)
                .permute(2, 0, 1)
                .float()
                .div_(255.0)
            )
            images.append(tensor)
        mask = torch.tensor(
            [int(value) for value in row["video_valid_mask"].split()],
            dtype=torch.long,
        )
        if mask.numel() != 16 or not bool(mask.any()):
            raise ValueError(f"invalid available-video mask: {row['file']}")
        return {
            "file_id": row["file"],
            "images": torch.stack(images),
            "attention_mask": mask,
            "quality": torch.tensor(
                float(row["video_quality"]),
                dtype=torch.float32,
            ),
            "output_path": row["output_path"],
        }


class DFERClipVisualEncoder(torch.nn.Module):
    def __init__(self, checkpoint: Path, dfer_repo: Path):
        super().__init__()
        repo = str(dfer_repo.resolve())
        models = str((dfer_repo / "models").resolve())
        if models not in sys.path:
            sys.path.insert(0, models)
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
            raise RuntimeError("DFER checkpoint lacks official model prefixes")
        self.image_encoder.load_state_dict(visual_state, strict=True)
        self.temporal_net.load_state_dict(temporal_state, strict=True)
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.eval()

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


def zero_payload(file_id: str) -> dict:
    return {
        "file_id": file_id,
        "tokens": torch.zeros(16, 512, dtype=torch.float32),
        "cls": torch.zeros(512, dtype=torch.float32),
        "attention_mask": torch.zeros(16, dtype=torch.long),
        "quality": torch.tensor(0.0, dtype=torch.float32),
        "encoder": "DFER-CLIP FERV39K ViT-B/32",
        "preprocessing": (
            "video unavailable under frozen MELD rule; deterministic zeros"
        ),
        "video_available": False,
    }


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))
    rows = [
        row
        for index, row in enumerate(all_rows)
        if index % args.num_shards == args.shard_index
    ]
    if args.limit > 0:
        rows = rows[: args.limit]

    zero_written = 0
    skipped = 0
    for row in rows:
        output_path = args.output_root / f"{row['file']}.pt"
        if feature_valid(output_path, row["file"]):
            skipped += 1
            continue
        if output_path.exists():
            raise RuntimeError(
                f"invalid pre-existing feature requires manual audit: {output_path}"
            )
        if int(row["video_available"]) == 0:
            save_feature(output_path, zero_payload(row["file"]))
            zero_written += 1

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = DFERClipVisualEncoder(args.checkpoint, args.dfer_repo).to(device)
    dataset = AvailableTrackDataset(rows, args.output_root)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    encoded = 0
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
                        "MELD-FAIR realignment and public active-speaker bbox; "
                        "16 official DFER-CLIP test-midpoint frames; square "
                        "crop with 25% margin; RGB / 255; MTCNN validity mask"
                    ),
                    "video_available": True,
                }
                if not torch.isfinite(payload["tokens"]).all():
                    raise FloatingPointError(
                        f"{file_id} contains non-finite DFER tokens"
                    )
                if not torch.isfinite(payload["cls"]).all():
                    raise FloatingPointError(
                        f"{file_id} contains non-finite DFER CLS"
                    )
                save_feature(Path(batch["output_path"][index]), payload)
                encoded += 1
            if encoded == len(batch["file_id"]) or encoded % 100 < len(
                batch["file_id"]
            ):
                print(
                    json.dumps(
                        {
                            "shard": args.shard_index,
                            "encoded": encoded,
                            "pending_available": len(dataset),
                            "elapsed_seconds": round(time.time() - started, 1),
                        }
                    ),
                    flush=True,
                )

    for row in rows:
        path = args.output_root / f"{row['file']}.pt"
        if not feature_valid(path, row["file"]):
            raise RuntimeError(f"feature validation failed: {path}")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "success",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "assigned": len(rows),
        "encoded": encoded,
        "zero_written": zero_written,
        "skipped_valid": skipped,
        "elapsed_seconds": time.time() - started,
        "source_hashes": {
            "manifest": sha256_file(args.manifest),
            "checkpoint": sha256_file(args.checkpoint),
            "script": sha256_file(Path(__file__)),
        },
    }
    atomic_json_save(
        summary,
        args.output_root.parent
        / "shard_summaries"
        / f"dfer_shard_{args.shard_index:02d}.json",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
