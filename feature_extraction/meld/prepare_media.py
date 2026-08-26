#!/usr/bin/env python3
"""Prepare MELD media for the TriMEF visual interface.

MELD-FAIR timestamps and active-speaker boxes are used only to correct the
utterance segment and localize the target face.  The model-facing visual
interface remains identical to the frozen IEMOCAP pipeline:

* 16 official DFER-CLIP test-midpoint frames;
* square 224x224 crops with a 25% face-box margin;
* an MTCNN-derived 16-frame validity mask;
* quality = valid_face_frame_ratio * mean_MTCNN_confidence.

There is no largest-face fallback and no sample-specific correction.  Missing
timestamps, missing public active-speaker boxes, explicitly non-unique speaker
labels, or zero MTCNN-valid frames make video unavailable.  Audio is extracted
from the public realigned clip when timestamps are available; otherwise the
already locked manifest audio is retained.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image


MAIN_FPS = 24_000 / 1_001
ALT_FPS = 25.0
ALT_DIALOGUES = {
    "train": {128, 203, 383, 517, 710, 967},
    "dev": {88},
    "test": {184},
}
SPLIT_FOLDERS = {
    "train": "train_splits",
    "dev": "dev_splits_complete",
    "test": "output_repeated_splits_test",
}
NUM_FRAMES = 16
IMAGE_SIZE = 224
MARGIN = 0.25
SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--timestamps-csv", type=Path, required=True)
    parser.add_argument("--bboxes-csv", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retain-video", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1:
        parser.error("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, num-shards)")
    invalid = sorted(set(args.splits).difference(SPLIT_FOLDERS))
    if invalid:
        parser.error("unsupported splits: " + ", ".join(invalid))
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


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "file",
        "split",
        "emotion",
        "text",
        "speaker",
        "dialogue_id",
        "utterance_id",
        "video_path",
        "audio_path",
    }
    missing = required.difference(rows[0] if rows else {})
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    file_ids = [row["file"] for row in rows]
    if len(file_ids) != len(set(file_ids)):
        raise ValueError("manifest contains duplicate file IDs")
    return rows


def read_grouped_csv(
    path: Path,
) -> dict[tuple[str, int, int], list[dict[str, str]]]:
    grouped: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (
                row["Split"].strip().lower(),
                int(row["Dialogue ID"]),
                int(row["Utterance ID"]),
            )
            grouped[key].append(row)
    return grouped


def nonunique_speaker(label: str) -> bool:
    normalized = " ".join(str(label).strip().lower().split())
    return (
        normalized in {"all", "both", "everyone", "group"}
        or " and " in normalized
    )


def source_video(
    raw_root: Path,
    split: str,
    dialogue: int,
    utterance: int,
) -> Path:
    return (
        raw_root
        / SPLIT_FOLDERS[split]
        / f"dia{dialogue}_utt{utterance}.mp4"
    )


def run_ffmpeg(command: list[str], command_log) -> None:
    command_log.write(json.dumps(command, ensure_ascii=False) + "\n")
    command_log.flush()
    completed = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        tail = "\n".join(completed.stderr.splitlines()[-20:])
        raise RuntimeError(
            f"ffmpeg failed with exit code {completed.returncode}\n{tail}"
        )


def dfer_test_indices(frame_count: int, num_frames: int = NUM_FRAMES) -> np.ndarray:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if frame_count <= num_frames:
        indices = list(range(frame_count))
        indices.extend([frame_count - 1] * (num_frames - frame_count))
        return np.asarray(indices, dtype=np.int64)
    tick = frame_count / float(num_frames)
    return np.asarray(
        [int(tick / 2.0 + tick * index) for index in range(num_frames)],
        dtype=np.int64,
    )


def decode_target_frames(
    video_path: Path,
) -> tuple[list[np.ndarray], np.ndarray, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if frame_count <= 0:
        decoded = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            decoded.append(frame)
        cap.release()
        if not decoded:
            raise RuntimeError(f"video contains no decodable frames: {video_path}")
        targets = dfer_test_indices(len(decoded))
        return [decoded[int(index)] for index in targets], targets, fps

    targets = dfer_test_indices(frame_count)
    frames: list[np.ndarray | None] = [None] * NUM_FRAMES
    slots: dict[int, list[int]] = defaultdict(list)
    for slot, target in enumerate(targets.tolist()):
        slots[target].append(slot)
    current = 0
    last_target = int(targets[-1])
    while current <= last_target:
        ok, frame = cap.read()
        if not ok:
            break
        for slot in slots.get(current, []):
            frames[slot] = frame.copy()
        current += 1
    cap.release()
    valid_slots = [index for index, frame in enumerate(frames) if frame is not None]
    if not valid_slots:
        raise RuntimeError(f"video contains no sampled frames: {video_path}")
    for slot, frame in enumerate(frames):
        if frame is None:
            nearest = min(valid_slots, key=lambda index: abs(index - slot))
            frames[slot] = frames[nearest].copy()
    return [frame for frame in frames if frame is not None], targets, fps


def interpolate_public_boxes(
    rows: list[dict[str, str]],
    targets: np.ndarray,
) -> np.ndarray:
    if not rows:
        raise ValueError("cannot interpolate an empty public face track")
    by_frame: dict[int, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        frame = int(row["Frame Number"])
        by_frame[frame].append(
            np.asarray(
                [
                    float(row["X Left"]),
                    float(row["Y Top"]),
                    float(row["X Right"]),
                    float(row["Y Bottom"]),
                ],
                dtype=np.float32,
            )
        )
    frame_numbers = np.asarray(sorted(by_frame), dtype=np.float32)
    values = np.stack(
        [
            np.mean(np.stack(by_frame[int(frame)], axis=0), axis=0)
            for frame in frame_numbers
        ],
        axis=0,
    )
    output = np.empty((len(targets), 4), dtype=np.float32)
    for coordinate in range(4):
        output[:, coordinate] = np.interp(
            targets.astype(np.float32),
            frame_numbers,
            values[:, coordinate],
        )
    return output


def square_margin_crop(
    frame: np.ndarray,
    box: np.ndarray,
    margin: float = MARGIN,
    image_size: int = IMAGE_SIZE,
) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in box]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1, x2 = np.clip([x1, x2], 0, width)
    y1, y2 = np.clip([y1, y2], 0, height)
    if x2 - x1 < 2 or y2 - y1 < 2:
        raise RuntimeError(f"degenerate public face box: {box.tolist()}")
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * (1.0 + 2.0 * margin)
    side = max(side, 16.0)
    left = int(np.floor(center_x - side / 2.0))
    top = int(np.floor(center_y - side / 2.0))
    right = int(np.ceil(center_x + side / 2.0))
    bottom = int(np.ceil(center_y + side / 2.0))
    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - width)
    pad_bottom = max(0, bottom - height)
    if any((pad_left, pad_top, pad_right, pad_bottom)):
        frame = cv2.copyMakeBorder(
            frame,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REPLICATE,
        )
        left += pad_left
        right += pad_left
        top += pad_top
        bottom += pad_top
    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        raise RuntimeError("computed an empty public active-speaker crop")
    return cv2.resize(
        crop,
        (image_size, image_size),
        interpolation=cv2.INTER_LINEAR,
    )


def mtcnn_validate(
    detector: MTCNN,
    crops: list[np.ndarray],
) -> tuple[np.ndarray, list[float]]:
    pil_crops = [
        Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        for crop in crops
    ]
    all_boxes, all_probs = detector.detect(pil_crops)
    mask = np.zeros(NUM_FRAMES, dtype=np.uint8)
    confidences = [0.0] * NUM_FRAMES
    for index, (boxes, probs) in enumerate(zip(all_boxes, all_probs)):
        if boxes is None or probs is None or len(boxes) == 0:
            continue
        finite = np.asarray(probs, dtype=np.float32)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            continue
        mask[index] = 1
        confidences[index] = float(finite.max())
    return mask, confidences


def completion_valid(record: dict, output_root: Path) -> bool:
    status = str(record.get("status", ""))
    audio_path = Path(str(record.get("audio_path", "")))
    if not audio_path.is_file():
        return False
    if status != "ok":
        return status.startswith("unavailable_")
    frame_dir = output_root / "frames" / record["split"] / record["file_id"]
    return all((frame_dir / f"{index:03d}.jpg").is_file() for index in range(16))


def assemble_realigned_clip(
    row: dict[str, str],
    timestamp_rows: list[dict[str, str]],
    raw_root: Path,
    temporary_root: Path,
    audio_path: Path,
    command_log,
) -> tuple[Path, list[dict[str, object]]]:
    split = row["split"]
    file_id = row["file"]
    temporary_root.mkdir(parents=True, exist_ok=False)
    fps = (
        ALT_FPS
        if int(row["dialogue_id"]) in ALT_DIALOGUES.get(split, set())
        else MAIN_FPS
    )
    segments = []
    segment_paths = []
    for index, timestamp in enumerate(timestamp_rows):
        dialogue = int(timestamp["Original Dialogue ID"])
        utterance = int(timestamp["Original Utterance ID"])
        start = float(timestamp["Start Time"])
        end = float(timestamp["End Time"])
        if not 0 <= start < end:
            raise ValueError(
                f"{file_id} has invalid timestamp interval [{start}, {end}]"
            )
        source = source_video(raw_root, split, dialogue, utterance)
        if not source.is_file():
            raise FileNotFoundError(source)
        segment_path = temporary_root / f"segment_{index:02d}.mp4"
        run_ffmpeg(
            [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-ss",
                f"{start:.9f}",
                "-to",
                f"{end:.9f}",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-ac",
                "2",
                "-c:a",
                "aac",
                "-b:a",
                "127k",
                "-r",
                f"{fps:.12f}",
                str(segment_path),
            ],
            command_log,
        )
        segment_paths.append(segment_path)
        segments.append(
            {
                "original_dialogue_id": dialogue,
                "original_utterance_id": utterance,
                "start": start,
                "end": end,
                "source_video": str(source),
            }
        )

    concat_list = temporary_root / "concat.txt"
    with concat_list.open("w", encoding="utf-8") as handle:
        for segment_path in segment_paths:
            handle.write(f"file '{segment_path.resolve().as_posix()}'\n")
    video_path = temporary_root / f"{file_id}.mp4"
    run_ffmpeg(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-ac",
            "2",
            "-c:a",
            "aac",
            "-b:a",
            "127k",
            "-r",
            f"{fps:.12f}",
            str(video_path),
        ],
        command_log,
    )
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_audio = audio_path.with_suffix(audio_path.suffix + ".tmp.wav")
    run_ffmpeg(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(temporary_audio),
        ],
        command_log,
    )
    os.replace(str(temporary_audio), str(audio_path))
    return video_path, segments


def process_one(
    row: dict[str, str],
    timestamp_rows: list[dict[str, str]],
    bbox_rows: list[dict[str, str]],
    detector: MTCNN,
    args: argparse.Namespace,
    command_log,
) -> dict:
    file_id = row["file"]
    split = row["split"]
    key = (split, int(row["dialogue_id"]), int(row["utterance_id"]))
    item_path = args.output_root / "item_metadata" / split / f"{file_id}.json"
    if item_path.is_file():
        with item_path.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        if completion_valid(existing, args.output_root):
            return existing
        raise RuntimeError(
            f"incomplete pre-existing output requires manual audit: {item_path}"
        )

    base_record = {
        "schema_version": SCHEMA_VERSION,
        "file_id": file_id,
        "split": split,
        "dialogue_id": key[1],
        "utterance_id": key[2],
        "speaker": row["speaker"],
        "original_audio_path": row["audio_path"],
        "timestamp_segments": len(timestamp_rows),
        "public_bbox_rows": len(bbox_rows),
        "sampling_strategy": "official_dfer_clip_test_midpoint_16",
        "crop_strategy": "public_active_speaker_bbox_square_margin_025",
        "validation_strategy": "IEMOCAP_MTCNN_on_16_sampled_crops",
        "quality_definition": (
            "valid_face_frame_ratio_times_mean_MTCNN_confidence"
        ),
        "no_largest_face_fallback": True,
    }

    if not timestamp_rows:
        original_audio = Path(row["audio_path"])
        if not original_audio.is_file():
            raise FileNotFoundError(original_audio)
        record = {
            **base_record,
            "status": "unavailable_no_public_timestamp",
            "video_available": False,
            "audio_path": str(original_audio),
            "audio_source": "locked_common_v1_original",
            "quality": 0.0,
            "valid_mask": [0] * NUM_FRAMES,
            "mtcnn_confidences": [0.0] * NUM_FRAMES,
            "source_segments": [],
        }
        atomic_json_save(record, item_path)
        return record

    audio_path = args.output_root / "audio" / split / f"{file_id}.wav"
    temporary_root = (
        args.output_root
        / "tmp"
        / f"shard_{args.shard_index:02d}"
        / file_id
    )
    video_path, source_segments = assemble_realigned_clip(
        row,
        timestamp_rows,
        args.raw_root,
        temporary_root,
        audio_path,
        command_log,
    )
    try:
        if nonunique_speaker(row["speaker"]):
            status = "unavailable_nonunique_speaker"
            record = {
                **base_record,
                "status": status,
                "video_available": False,
                "audio_path": str(audio_path),
                "audio_source": "MELD_FAIR_realigned",
                "quality": 0.0,
                "valid_mask": [0] * NUM_FRAMES,
                "mtcnn_confidences": [0.0] * NUM_FRAMES,
                "source_segments": source_segments,
            }
            atomic_json_save(record, item_path)
            return record

        if not bbox_rows:
            record = {
                **base_record,
                "status": "unavailable_no_public_active_speaker_bbox",
                "video_available": False,
                "audio_path": str(audio_path),
                "audio_source": "MELD_FAIR_realigned",
                "quality": 0.0,
                "valid_mask": [0] * NUM_FRAMES,
                "mtcnn_confidences": [0.0] * NUM_FRAMES,
                "source_segments": source_segments,
            }
            atomic_json_save(record, item_path)
            return record

        frames, targets, actual_fps = decode_target_frames(video_path)
        boxes = interpolate_public_boxes(bbox_rows, targets)
        crops = [
            square_margin_crop(frame, box)
            for frame, box in zip(frames, boxes)
        ]
        valid_mask, confidences = mtcnn_validate(detector, crops)
        valid_confidences = [
            confidence
            for confidence, valid in zip(confidences, valid_mask.tolist())
            if valid
        ]
        valid_ratio = float(valid_mask.mean())
        mean_confidence = (
            float(np.mean(valid_confidences)) if valid_confidences else 0.0
        )
        quality = float(np.clip(valid_ratio * mean_confidence, 0.0, 1.0))
        if valid_mask.sum() == 0:
            record = {
                **base_record,
                "status": "unavailable_mtcnn_no_valid_face",
                "video_available": False,
                "audio_path": str(audio_path),
                "audio_source": "MELD_FAIR_realigned",
                "quality": 0.0,
                "valid_mask": [0] * NUM_FRAMES,
                "mtcnn_confidences": confidences,
                "source_segments": source_segments,
                "source_frame_indices": targets.tolist(),
                "source_fps": actual_fps,
                "interpolated_public_boxes": boxes.tolist(),
            }
            atomic_json_save(record, item_path)
            return record

        frame_dir = args.output_root / "frames" / split / file_id
        frame_dir.mkdir(parents=True, exist_ok=False)
        for index, crop in enumerate(crops):
            frame_path = frame_dir / f"{index:03d}.jpg"
            if not cv2.imwrite(
                str(frame_path),
                crop,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            ):
                raise RuntimeError(f"failed to write {frame_path}")
        record = {
            **base_record,
            "status": "ok",
            "video_available": True,
            "audio_path": str(audio_path),
            "audio_source": "MELD_FAIR_realigned",
            "frame_dir": str(frame_dir),
            "quality": quality,
            "valid_ratio": valid_ratio,
            "mean_confidence": mean_confidence,
            "valid_mask": valid_mask.astype(int).tolist(),
            "mtcnn_confidences": confidences,
            "source_segments": source_segments,
            "source_frame_indices": targets.tolist(),
            "source_fps": actual_fps,
            "interpolated_public_boxes": boxes.tolist(),
        }
        atomic_json_save(record, item_path)
        if args.retain_video:
            retained = args.output_root / "realigned_video" / split / video_path.name
            retained.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(video_path, retained)
            record["retained_video"] = str(retained)
            atomic_json_save(record, item_path)
        return record
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = [
        row
        for row in read_manifest(args.manifest)
        if row["split"] in set(args.splits)
    ]
    rows = [
        row
        for global_index, row in enumerate(rows)
        if global_index % args.num_shards == args.shard_index
    ]
    if args.limit > 0:
        rows = rows[: args.limit]
    timestamps = read_grouped_csv(args.timestamps_csv)
    bboxes = read_grouped_csv(args.bboxes_csv)

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    detector = MTCNN(
        keep_all=True,
        select_largest=False,
        post_process=False,
        min_face_size=24,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        device=device,
    )
    log_path = (
        args.output_root
        / "logs"
        / f"commands_shard_{args.shard_index:02d}.jsonl"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    completed = 0
    status_counts: dict[str, int] = defaultdict(int)
    with log_path.open("a", encoding="utf-8") as command_log:
        for row in rows:
            key = (
                row["split"],
                int(row["dialogue_id"]),
                int(row["utterance_id"]),
            )
            try:
                record = process_one(
                    row,
                    timestamps.get(key, []),
                    bboxes.get(key, []),
                    detector,
                    args,
                    command_log,
                )
            except Exception as error:
                failure = {
                    "schema_version": SCHEMA_VERSION,
                    "file_id": row["file"],
                    "split": row["split"],
                    "status": "error",
                    "error": repr(error),
                    "shard_index": args.shard_index,
                }
                failure_path = (
                    args.output_root
                    / "failures"
                    / row["split"]
                    / f"{row['file']}.json"
                )
                atomic_json_save(failure, failure_path)
                print(json.dumps(failure, ensure_ascii=False), flush=True)
                raise
            completed += 1
            status_counts[record["status"]] += 1
            if completed == 1 or completed % 25 == 0 or completed == len(rows):
                print(
                    json.dumps(
                        {
                            "shard": args.shard_index,
                            "completed": completed,
                            "total": len(rows),
                            "status_counts": dict(status_counts),
                            "elapsed_seconds": round(time.time() - started, 1),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    flush=True,
                )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "success",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "completed": completed,
        "status_counts": dict(status_counts),
        "elapsed_seconds": time.time() - started,
        "source_hashes": {
            "manifest": sha256_file(args.manifest),
            "timestamps": sha256_file(args.timestamps_csv),
            "bboxes": sha256_file(args.bboxes_csv),
            "script": sha256_file(Path(__file__)),
        },
    }
    atomic_json_save(
        summary,
        args.output_root
        / "shard_summaries"
        / f"media_shard_{args.shard_index:02d}.json",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
