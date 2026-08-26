#!/usr/bin/env python3
"""Verify one-to-one alignment and contents of all visual Stage-I artifacts."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--skip-images", action="store_true")
    return parser.parse_args()


def fail(message):
    raise AssertionError(message)


def main():
    args = parse_args()
    source = pd.read_csv(args.csv, dtype={"file": str})
    metadata = pd.read_csv(args.metadata, dtype={"file_id": str})
    expected_ids = source["file"].astype(str).tolist()
    expected_set = set(expected_ids)
    if len(expected_ids) != len(expected_set):
        fail("Duplicate file IDs in the source CSV")
    if metadata["file_id"].duplicated().any():
        fail("Duplicate file IDs in face metadata")

    metadata_set = set(metadata["file_id"].astype(str))
    if metadata_set != expected_set:
        fail(
            f"Metadata ID mismatch: missing={len(expected_set - metadata_set)}, "
            f"extra={len(metadata_set - expected_set)}"
        )
    bad_status = metadata[metadata["status"] != "ok"]
    if not bad_status.empty:
        fail(
            f"Face crop failures={len(bad_status)}; "
            f"examples={bad_status['file_id'].head(10).tolist()}"
        )

    feature_dir = Path(args.feature_dir)
    feature_paths = {path.stem: path for path in feature_dir.glob("*.pt")}
    feature_set = set(feature_paths)
    if feature_set != expected_set:
        fail(
            f"Feature ID mismatch: missing={len(expected_set - feature_set)}, "
            f"extra={len(feature_set - expected_set)}"
        )

    errors = []
    qualities = []
    valid_ratios = []
    zero_mask_count = 0
    for position, row in enumerate(metadata.itertuples(index=False), start=1):
        file_id = str(row.file_id)
        try:
            mask = np.asarray(
                [int(value) for value in str(row.valid_mask).split()],
                dtype=np.int64,
            )
            if mask.shape != (16,) or not set(mask.tolist()).issubset({0, 1}):
                fail("invalid metadata mask")
            if int(row.detected_frames) != int(mask.sum()):
                fail("detected_frames does not match mask")
            if abs(float(row.valid_ratio) - float(mask.mean())) > 1e-6:
                fail("valid_ratio does not match mask")
            if str(row.sampling_strategy) != "official_dfer_clip_test_midpoint_16":
                fail("unexpected temporal sampling strategy")
            if abs(float(row.margin) - 0.25) > 1e-8:
                fail("unexpected face margin")

            quality = float(row.quality)
            if not np.isfinite(quality) or not 0.0 <= quality <= 1.0:
                fail("invalid quality")
            if mask.sum() == 0:
                zero_mask_count += 1
                if quality != 0.0:
                    fail("all-invalid mask has nonzero quality")

            if not args.skip_images:
                frame_dir = Path(row.frame_dir)
                for frame_index in range(16):
                    frame_path = frame_dir / f"{frame_index:03d}.jpg"
                    image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                    if image is None or image.shape != (224, 224, 3):
                        fail(f"bad cropped frame: {frame_path}")

            payload = torch.load(feature_paths[file_id], map_location="cpu")
            if str(payload["file_id"]) != file_id:
                fail("payload file_id mismatch")
            tokens = payload["tokens"]
            cls_feature = payload["cls"]
            feature_mask = payload["attention_mask"].long()
            feature_quality = torch.as_tensor(payload["quality"])
            if tokens.shape != (16, 512) or tokens.dtype != torch.float32:
                fail("bad token shape or dtype")
            if cls_feature.shape != (512,) or cls_feature.dtype != torch.float32:
                fail("bad CLS shape or dtype")
            if not torch.isfinite(tokens).all() or not torch.isfinite(cls_feature).all():
                fail("non-finite DFER-CLIP feature")
            if feature_mask.shape != (16,):
                fail("bad feature mask shape")
            if not torch.equal(feature_mask, torch.from_numpy(mask)):
                fail("feature mask does not match metadata")
            if feature_quality.numel() != 1 or not torch.isfinite(feature_quality).all():
                fail("invalid feature quality")
            if abs(float(feature_quality) - quality) > 1e-6:
                fail("feature quality does not match metadata")
            if "official DFER-CLIP test-midpoint" not in payload["preprocessing"]:
                fail("feature preprocessing provenance is missing")

            qualities.append(quality)
            valid_ratios.append(float(row.valid_ratio))
        except Exception as error:
            errors.append((file_id, repr(error)))
        if position % 500 == 0 or position == len(metadata):
            print(
                f"checked {position}/{len(metadata)} errors={len(errors)}",
                flush=True,
            )

    if errors:
        fail(f"Artifact validation errors={len(errors)}; first={errors[:20]}")
    qualities = np.asarray(qualities)
    valid_ratios = np.asarray(valid_ratios)
    print("PASS", len(expected_ids), "utterances")
    print(
        "quality min/mean/median/max:",
        float(qualities.min()),
        float(qualities.mean()),
        float(np.median(qualities)),
        float(qualities.max()),
    )
    print(
        "valid-ratio min/mean/median/max:",
        float(valid_ratios.min()),
        float(valid_ratios.mean()),
        float(np.median(valid_ratios)),
        float(valid_ratios.max()),
    )
    print("all-invalid tracks:", zero_mask_count)


if __name__ == "__main__":
    main()
