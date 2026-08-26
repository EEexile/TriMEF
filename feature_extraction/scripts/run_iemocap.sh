#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${IEMOCAP_CSV:?set IEMOCAP_CSV}"
: "${IEMOCAP_VIDEO_ROOT:?set IEMOCAP_VIDEO_ROOT}"
: "${DFER_REPO:?set DFER_REPO}"
: "${DFER_CHECKPOINT:?set DFER_CHECKPOINT}"
: "${WORK_ROOT:?set WORK_ROOT to a new output directory}"
VISIBLE_GPU="${VISIBLE_GPU:-0}"

if [[ -e "$WORK_ROOT" ]]; then
  echo "WORK_ROOT already exists: $WORK_ROOT" >&2
  exit 3
fi

mkdir -p "$WORK_ROOT/quality" "$WORK_ROOT/face_tracks" "$WORK_ROOT/features"

CUDA_VISIBLE_DEVICES="$VISIBLE_GPU" python "$ROOT/iemocap/crop_faces.py" \
  --csv "$IEMOCAP_CSV" \
  --video-root "$IEMOCAP_VIDEO_ROOT" \
  --output-root "$WORK_ROOT/face_tracks" \
  --metadata "$WORK_ROOT/quality/face_quality.csv" \
  --preview "$WORK_ROOT/quality/face_preview.jpg" \
  --device cuda:0 \
  --seed 42

CUDA_VISIBLE_DEVICES="$VISIBLE_GPU" python "$ROOT/iemocap/extract_dfer_features.py" \
  --metadata "$WORK_ROOT/quality/face_quality.csv" \
  --checkpoint "$DFER_CHECKPOINT" \
  --dfer-repo "$DFER_REPO" \
  --output-root "$WORK_ROOT/features" \
  --device cuda:0 \
  --batch-size 4 \
  --workers 4

python "$ROOT/iemocap/verify_features.py" \
  --csv "$IEMOCAP_CSV" \
  --metadata "$WORK_ROOT/quality/face_quality.csv" \
  --feature-dir "$WORK_ROOT/features"

python "$ROOT/verify_feature_payloads.py" \
  --manifest "$IEMOCAP_CSV" \
  --feature-root "$WORK_ROOT/features" \
  --expected-count 5531
