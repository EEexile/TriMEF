#!/usr/bin/env python3
"""Read-only integrity checks for TriMEF visual feature payloads.

This handoff helper does not create or alter model features.  It validates the
common IEMOCAP/MELD ``.pt`` schema and, when present, checks MELD manifest fields
against each payload.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--id-column", default="file")
    parser.add_argument("--expected-count", type=int, default=0)
    return parser.parse_args()


def parse_mask(value: str) -> torch.Tensor:
    return torch.tensor([int(item) for item in value.split()], dtype=torch.long)


def main() -> int:
    args = parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("manifest is empty")
    if args.id_column not in rows[0]:
        raise ValueError(f"missing ID column: {args.id_column}")

    file_ids = [str(row[args.id_column]) for row in rows]
    if len(file_ids) != len(set(file_ids)):
        raise ValueError("manifest contains duplicate IDs")
    if args.expected_count and len(file_ids) != args.expected_count:
        raise ValueError(
            f"expected {args.expected_count} rows, found {len(file_ids)}"
        )

    feature_paths = {path.stem: path for path in args.feature_root.glob("*.pt")}
    expected = set(file_ids)
    actual = set(feature_paths)
    if expected != actual:
        raise ValueError(
            f"feature ID mismatch: missing={len(expected - actual)}, "
            f"extra={len(actual - expected)}"
        )

    available = 0
    unavailable = 0
    quality_values = []
    for position, row in enumerate(rows, start=1):
        file_id = str(row[args.id_column])
        payload = torch.load(feature_paths[file_id], map_location="cpu")
        if str(payload.get("file_id")) != file_id:
            raise ValueError(f"payload ID mismatch: {file_id}")

        tokens = torch.as_tensor(payload["tokens"])
        cls = torch.as_tensor(payload["cls"])
        mask = torch.as_tensor(payload["attention_mask"]).long()
        quality = float(torch.as_tensor(payload["quality"]))
        if tuple(tokens.shape) != (16, 512) or tokens.dtype != torch.float32:
            raise ValueError(f"invalid tokens for {file_id}: {tokens.shape}")
        if tuple(cls.shape) != (512,) or cls.dtype != torch.float32:
            raise ValueError(f"invalid cls for {file_id}: {cls.shape}")
        if tuple(mask.shape) != (16,) or not set(mask.tolist()).issubset({0, 1}):
            raise ValueError(f"invalid mask for {file_id}")
        if not torch.isfinite(tokens).all() or not torch.isfinite(cls).all():
            raise FloatingPointError(f"non-finite feature: {file_id}")
        if not 0.0 <= quality <= 1.0:
            raise ValueError(f"quality outside [0,1]: {file_id}")

        if "video_valid_mask" in row:
            manifest_mask = parse_mask(row["video_valid_mask"])
            if not torch.equal(mask, manifest_mask):
                raise ValueError(f"manifest/payload mask mismatch: {file_id}")
        if "video_quality" in row:
            if abs(quality - float(row["video_quality"])) > 1e-6:
                raise ValueError(f"manifest/payload quality mismatch: {file_id}")
        if "video_available" in row:
            is_available = int(row["video_available"]) == 1
            if is_available:
                available += 1
                if not bool(mask.any()):
                    raise ValueError(f"available item has empty mask: {file_id}")
            else:
                unavailable += 1
                if (
                    bool(mask.any())
                    or quality != 0.0
                    or bool(torch.count_nonzero(tokens))
                    or bool(torch.count_nonzero(cls))
                ):
                    raise ValueError(f"unavailable item is not zeroed: {file_id}")
        quality_values.append(quality)
        if position % 1000 == 0 or position == len(rows):
            print(f"checked {position}/{len(rows)}", flush=True)

    print(f"PASS: {len(rows)} unique feature payloads")
    print(
        "quality min/mean/max: "
        f"{min(quality_values):.6f}/"
        f"{sum(quality_values) / len(quality_values):.6f}/"
        f"{max(quality_values):.6f}"
    )
    if "video_available" in rows[0]:
        print(f"video available/unavailable: {available}/{unavailable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
