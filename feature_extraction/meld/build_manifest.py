#!/usr/bin/env python3
"""Freeze a complete MELD manifest from per-item preprocessing evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    return parser.parse_args()


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


def atomic_csv_save(rows: list[dict], fieldnames: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def main() -> int:
    args = parse_args()
    with args.input_manifest.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    if not source_rows:
        raise ValueError("input manifest is empty")

    rows = []
    status_counts = Counter()
    split_counts = Counter()
    missing = []
    for source in source_rows:
        file_id = source["file"]
        split = source["split"]
        item_path = (
            args.media_root
            / "item_metadata"
            / split
            / f"{file_id}.json"
        )
        if not item_path.is_file():
            missing.append(str(item_path))
            continue
        with item_path.open(encoding="utf-8") as handle:
            media = json.load(handle)
        if media.get("file_id") != file_id or media.get("split") != split:
            raise ValueError(f"item identity mismatch: {item_path}")
        if media.get("status") == "error":
            raise RuntimeError(f"failed media item present: {item_path}")
        audio_path = Path(str(media["audio_path"]))
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        mask = list(media.get("valid_mask", []))
        if len(mask) != 16 or any(value not in (0, 1) for value in mask):
            raise ValueError(f"invalid video mask in {item_path}")
        quality = float(media.get("quality", -1.0))
        if not 0.0 <= quality <= 1.0:
            raise ValueError(f"invalid video quality in {item_path}")
        available = bool(media.get("video_available", False))
        if available != bool(sum(mask)):
            raise ValueError(f"video availability/mask mismatch in {item_path}")
        if not available and quality != 0.0:
            raise ValueError(f"unavailable video must have q=0 in {item_path}")
        if available:
            frame_dir = Path(str(media["frame_dir"]))
            if not all(
                (frame_dir / f"{index:03d}.jpg").is_file()
                for index in range(16)
            ):
                raise FileNotFoundError(f"incomplete frame directory: {frame_dir}")

        row = dict(source)
        row.update(
            {
                "audio_path": str(audio_path.resolve()),
                "video_available": int(available),
                "video_status": str(media["status"]),
                "video_quality": f"{quality:.12g}",
                "video_valid_mask": " ".join(str(int(value)) for value in mask),
                "media_metadata_path": str(item_path.resolve()),
                "audio_source": str(media.get("audio_source", "")),
            }
        )
        rows.append(row)
        status_counts[str(media["status"])] += 1
        split_counts[split] += 1

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} preprocessing records are missing; first={missing[:5]}"
        )
    if len(rows) != len(source_rows):
        raise AssertionError("manifest aggregation did not preserve every sample")
    if [row["file"] for row in rows] != [row["file"] for row in source_rows]:
        raise AssertionError("manifest row order changed")

    fieldnames = list(source_rows[0].keys())
    for name in (
        "video_available",
        "video_status",
        "video_quality",
        "video_valid_mask",
        "media_metadata_path",
        "audio_source",
    ):
        if name not in fieldnames:
            fieldnames.append(name)
    atomic_csv_save(rows, fieldnames, args.output_manifest)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "success",
        "total": len(rows),
        "split_counts": dict(split_counts),
        "status_counts": dict(status_counts),
        "video_available": sum(int(row["video_available"]) for row in rows),
        "video_unavailable": sum(
            1 - int(row["video_available"]) for row in rows
        ),
        "input_manifest": str(args.input_manifest.resolve()),
        "output_manifest": str(args.output_manifest.resolve()),
        "source_hashes": {
            "input_manifest": sha256_file(args.input_manifest),
            "output_manifest": sha256_file(args.output_manifest),
            "script": sha256_file(Path(__file__)),
        },
    }
    atomic_json_save(summary, args.output_summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
