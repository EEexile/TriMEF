#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${MELD_RAW_ROOT:?set MELD_RAW_ROOT}"
: "${MELD_PROCESSED_ROOT:?set MELD_PROCESSED_ROOT}"
: "${MELD_TIMESTAMPS_CSV:?set MELD_TIMESTAMPS_CSV}"
: "${MELD_BBOXES_CSV:?set MELD_BBOXES_CSV}"
: "${DFER_REPO:?set DFER_REPO}"
: "${DFER_CHECKPOINT:?set DFER_CHECKPOINT}"
: "${WORK_ROOT:?set WORK_ROOT to a new output directory}"
VISIBLE_GPU="${VISIBLE_GPU:-0}"

if [[ -e "$WORK_ROOT" ]]; then
  echo "WORK_ROOT already exists: $WORK_ROOT" >&2
  exit 3
fi

mkdir -p "$WORK_ROOT"

python "$ROOT/meld/prepare_manifest.py" \
  --raw-root "$MELD_RAW_ROOT" \
  --processed-root "$MELD_PROCESSED_ROOT" \
  --output-root "$WORK_ROOT/common_manifest"

CUDA_VISIBLE_DEVICES="$VISIBLE_GPU" python "$ROOT/meld/prepare_media.py" \
  --manifest "$WORK_ROOT/common_manifest/dataset_all.csv" \
  --timestamps-csv "$MELD_TIMESTAMPS_CSV" \
  --bboxes-csv "$MELD_BBOXES_CSV" \
  --raw-root "$MELD_RAW_ROOT" \
  --output-root "$WORK_ROOT/media" \
  --device cuda:0 \
  --num-shards 1 \
  --shard-index 0 \
  --seed 42

python "$ROOT/meld/build_manifest.py" \
  --input-manifest "$WORK_ROOT/common_manifest/dataset_all.csv" \
  --media-root "$WORK_ROOT/media" \
  --output-manifest "$WORK_ROOT/dataset_all_trimef.csv" \
  --output-summary "$WORK_ROOT/media_summary.json"

CUDA_VISIBLE_DEVICES="$VISIBLE_GPU" python "$ROOT/meld/extract_dfer_features.py" \
  --manifest "$WORK_ROOT/dataset_all_trimef.csv" \
  --checkpoint "$DFER_CHECKPOINT" \
  --dfer-repo "$DFER_REPO" \
  --output-root "$WORK_ROOT/features" \
  --device cuda:0 \
  --batch-size 8 \
  --workers 4 \
  --num-shards 1 \
  --shard-index 0

python "$ROOT/verify_feature_payloads.py" \
  --manifest "$WORK_ROOT/dataset_all_trimef.csv" \
  --feature-root "$WORK_ROOT/features" \
  --expected-count 13706
