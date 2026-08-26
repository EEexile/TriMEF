#!/usr/bin/env python3
"""Create deterministic 16-frame face tracks for IEMOCAP utterance videos.

The original utterance videos are never modified.  Each output directory contains
16 square 224x224 JPEGs plus per-utterance quality metadata.  Detection is done
only on the 16 uniformly sampled frames because DFER-CLIP consumes exactly 16
frames.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from facenet_pytorch import MTCNN
from PIL import Image


VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--preview", default=None)
    parser.add_argument("--preview-count", type=int, default=16)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--margin",
        type=float,
        default=0.25,
        help="Fraction of face-box width/height added on every side.",
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_video_index(video_root):
    index = {}
    duplicates = []
    for path in Path(video_root).rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
            if path.stem in index:
                duplicates.append(path.stem)
            else:
                index[path.stem] = path
    if duplicates:
        raise RuntimeError(
            "Duplicate utterance video stems found: " + ", ".join(duplicates[:10])
        )
    return index


def dfer_test_indices(frame_count, num_frames):
    """Match the official DFER-CLIP test-time temporal sampler."""
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


def uniform_sample_frames(video_path, num_frames):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("OpenCV cannot open video")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = [None] * num_frames

    if frame_count > 0:
        targets = dfer_test_indices(frame_count, num_frames)
        target_to_slots = {}
        for slot, target in enumerate(targets.tolist()):
            target_to_slots.setdefault(target, []).append(slot)

        current = 0
        last_target = int(targets[-1])
        while current <= last_target:
            ok, frame = cap.read()
            if not ok:
                break
            for slot in target_to_slots.get(current, []):
                frames[slot] = frame.copy()
            current += 1
    else:
        decoded = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            decoded.append(frame)
        if decoded:
            targets = dfer_test_indices(len(decoded), num_frames)
            frames = [decoded[int(i)] for i in targets]
            frame_count = len(decoded)

    cap.release()

    valid_slots = [i for i, frame in enumerate(frames) if frame is not None]
    if not valid_slots:
        raise RuntimeError("Video contains no decodable frames")
    for slot, frame in enumerate(frames):
        if frame is None:
            nearest = min(valid_slots, key=lambda i: abs(i - slot))
            frames[slot] = frames[nearest].copy()
    return frames, frame_count


def box_iou(box_a, box_b):
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    return inter / max(area_a + area_b - inter, 1e-6)


def choose_face_track(all_boxes, all_probs, frame_shapes):
    selected_boxes = []
    selected_probs = []
    previous = None

    for boxes, probs, shape in zip(all_boxes, all_probs, frame_shapes):
        if boxes is None or len(boxes) == 0:
            selected_boxes.append(None)
            selected_probs.append(0.0)
            continue

        boxes = np.asarray(boxes, dtype=np.float32)
        probs = np.asarray(probs, dtype=np.float32)
        height, width = shape[:2]
        areas = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
            0, boxes[:, 3] - boxes[:, 1]
        )
        area_score = areas / max(float(height * width), 1.0)

        if previous is None:
            scores = 0.7 * probs + 0.3 * np.minimum(area_score / 0.25, 1.0)
        else:
            ious = np.asarray([box_iou(previous, box) for box in boxes])
            scores = (
                0.65 * ious
                + 0.25 * probs
                + 0.10 * np.minimum(area_score / 0.25, 1.0)
            )
        chosen = int(np.argmax(scores))
        previous = boxes[chosen].copy()
        selected_boxes.append(previous)
        selected_probs.append(float(probs[chosen]))
    return selected_boxes, selected_probs


def interpolate_boxes(boxes, frame_shape):
    valid = np.asarray([box is not None for box in boxes], dtype=np.uint8)
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) == 0:
        height, width = frame_shape[:2]
        side = min(height, width) * 0.9
        cx, cy = width / 2.0, height / 2.0
        fallback = np.asarray(
            [cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2],
            dtype=np.float32,
        )
        return [fallback.copy() for _ in boxes], valid

    values = np.stack([boxes[i] for i in valid_indices], axis=0)
    positions = np.arange(len(boxes), dtype=np.float32)
    interpolated = np.zeros((len(boxes), 4), dtype=np.float32)
    for coordinate in range(4):
        interpolated[:, coordinate] = np.interp(
            positions, valid_indices.astype(np.float32), values[:, coordinate]
        )
    return [interpolated[i] for i in range(len(boxes))], valid


def square_margin_crop(frame, box, margin, image_size):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * (1.0 + 2.0 * margin)
    side = max(side, 16.0)

    left = int(np.floor(cx - side / 2.0))
    top = int(np.floor(cy - side / 2.0))
    right = int(np.ceil(cx + side / 2.0))
    bottom = int(np.ceil(cy + side / 2.0))

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
        raise RuntimeError("Computed an empty face crop")
    crop = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return crop


def make_preview(records, output_path, max_count):
    usable = [record for record in records if record.get("status") == "ok"][:max_count]
    if not usable:
        return
    columns = 4
    tile_width, tile_height = 224, 254
    rows = int(np.ceil(len(usable) / columns))
    canvas = np.full((rows * tile_height, columns * tile_width, 3), 255, np.uint8)
    for index, record in enumerate(usable):
        frame_path = Path(record["frame_dir"]) / "007.jpg"
        image = cv2.imread(str(frame_path))
        if image is None:
            continue
        image = cv2.resize(image, (tile_width, 224))
        row, column = divmod(index, columns)
        y, x = row * tile_height, column * tile_width
        canvas[y : y + 224, x : x + tile_width] = image
        label = "%s q=%.2f" % (record["file_id"], record["quality"])
        cv2.putText(
            canvas,
            label,
            (x + 3, y + 244),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def process_one(file_id, video_path, output_root, detector, args):
    output_dir = Path(output_root) / ("Session" + str(int(file_id[3:5]))) / file_id
    metadata_path = output_dir / "metadata.json"
    expected = [output_dir / ("%03d.jpg" % i) for i in range(args.num_frames)]
    if (
        not args.overwrite
        and metadata_path.exists()
        and all(path.exists() for path in expected)
    ):
        with metadata_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    frames, source_frame_count = uniform_sample_frames(video_path, args.num_frames)
    pil_frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
    all_boxes, all_probs = detector.detect(pil_frames)
    selected_boxes, selected_probs = choose_face_track(
        all_boxes, all_probs, [frame.shape for frame in frames]
    )
    boxes, valid_mask = interpolate_boxes(selected_boxes, frames[0].shape)

    output_dir.mkdir(parents=True, exist_ok=True)
    blur_scores = []
    area_ratios = []
    for index, (frame, box) in enumerate(zip(frames, boxes)):
        crop = square_margin_crop(frame, box, args.margin, args.image_size)
        blur_scores.append(
            float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        )
        x1, y1, x2, y2 = box
        area_ratios.append(
            float(max(0, x2 - x1) * max(0, y2 - y1) / (frame.shape[0] * frame.shape[1]))
        )
        if not cv2.imwrite(str(expected[index]), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError("Failed to write " + str(expected[index]))

    detected_probs = [
        probability
        for probability, is_valid in zip(selected_probs, valid_mask.tolist())
        if is_valid
    ]
    valid_ratio = float(valid_mask.mean())
    mean_confidence = float(np.mean(detected_probs)) if detected_probs else 0.0
    quality = float(np.clip(valid_ratio * mean_confidence, 0.0, 1.0))
    record = {
        "file_id": file_id,
        "status": "ok",
        "video_path": str(video_path),
        "frame_dir": str(output_dir),
        "source_frame_count": int(source_frame_count),
        "detected_frames": int(valid_mask.sum()),
        "valid_ratio": valid_ratio,
        "mean_confidence": mean_confidence,
        "mean_face_area_ratio": float(np.mean(area_ratios)),
        "median_blur": float(np.median(blur_scores)),
        "quality": quality,
        "valid_mask": " ".join(str(int(value)) for value in valid_mask.tolist()),
        "margin": float(args.margin),
        "sampling_strategy": "official_dfer_clip_test_midpoint_16",
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
    return record


def main():
    args = parse_args()
    if args.num_frames != 16:
        raise ValueError("The official DFER-CLIP checkpoint requires exactly 16 frames")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = pd.read_csv(args.csv)
    file_ids = data["file"].astype(str).tolist()
    if args.sample_size > 0:
        generator = random.Random(args.seed)
        file_ids = generator.sample(file_ids, min(args.sample_size, len(file_ids)))

    video_index = build_video_index(args.video_root)
    missing = [file_id for file_id in file_ids if file_id not in video_index]
    if missing:
        raise FileNotFoundError(
            "Missing %d videos; first IDs: %s" % (len(missing), ", ".join(missing[:10]))
        )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    detector = MTCNN(
        keep_all=True,
        select_largest=False,
        post_process=False,
        min_face_size=24,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        device=device,
    )

    records = []
    failures = []
    started = time.time()
    for position, file_id in enumerate(file_ids, start=1):
        try:
            record = process_one(
                file_id, video_index[file_id], args.output_root, detector, args
            )
            records.append(record)
        except Exception as error:
            failures.append((file_id, repr(error)))
            records.append(
                {
                    "file_id": file_id,
                    "status": "error",
                    "video_path": str(video_index[file_id]),
                    "error": repr(error),
                }
            )
        if position == 1 or position % 10 == 0 or position == len(file_ids):
            elapsed = time.time() - started
            print(
                "[%d/%d] failures=%d elapsed=%.1fs"
                % (position, len(file_ids), len(failures), elapsed),
                flush=True,
            )

    metadata_path = Path(args.metadata)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    new_data = pd.DataFrame(records)
    if metadata_path.exists() and not args.overwrite:
        old_data = pd.read_csv(metadata_path)
        old_data = old_data[~old_data["file_id"].isin(new_data["file_id"])]
        new_data = pd.concat([old_data, new_data], ignore_index=True)
    new_data.sort_values("file_id").to_csv(metadata_path, index=False)

    if args.preview:
        make_preview(records, args.preview, args.preview_count)

    ok_records = [record for record in records if record.get("status") == "ok"]
    if ok_records:
        qualities = np.asarray([record["quality"] for record in ok_records])
        print(
            "quality min/mean/median/max: %.3f %.3f %.3f %.3f"
            % (
                qualities.min(),
                qualities.mean(),
                np.median(qualities),
                qualities.max(),
            )
        )
    print("metadata:", metadata_path)
    if failures:
        print("Failures:", failures[:10], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
