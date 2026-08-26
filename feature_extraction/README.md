# Visual preprocessing and feature extraction

This directory contains the visual pipeline used by TriMEF for IEMOCAP and MELD. It samples 16 target-face frames, records frame validity and target-face observability, and exports frozen DFER-CLIP temporal features.

Datasets, generated crops, extracted features, DFER-CLIP source code, and model checkpoints are not included.

## Pipeline

```text
IEMOCAP video -> target-face crops -> DFER-CLIP features
MELD video + MELD-FAIR metadata -> target-face crops -> DFER-CLIP features
```

IEMOCAP uses MTCNN on the aligned speaker view. MELD uses MELD-FAIR timestamps and active-speaker boxes before MTCNN validation. Both pipelines use 16 equal-bin midpoint frames, 224 x 224 square crops with a 25% margin, and the same frozen DFER-CLIP encoder.

The continuous target-face observability value is

```text
q = valid-frame ratio x mean MTCNN confidence over valid frames.
```

If all 16 frames are invalid, the payload contains zero visual features, an all-zero mask, and `q=0`.

## Files

```text
feature_extraction/
|-- iemocap/
|   |-- crop_faces.py
|   |-- extract_dfer_features.py
|   `-- verify_features.py
|-- meld/
|   |-- prepare_manifest.py
|   |-- prepare_media.py
|   |-- build_manifest.py
|   `-- extract_dfer_features.py
|-- scripts/
|   |-- run_iemocap.sh
|   `-- run_meld.sh
|-- verify_feature_payloads.py
`-- requirements.txt
```

## Requirements

Use Linux, an NVIDIA GPU, Python 3.10 or later, and system `ffmpeg`. Install PyTorch and torchvision for the local CUDA version, then install the remaining packages:

```bash
pip install -r feature_extraction/requirements.txt
```

Obtain the official [DFER-CLIP repository](https://github.com/zengqunzhao/DFER-CLIP) and its FERV39K checkpoint separately. The paper pipeline used DFER-CLIP commit `d10d5a4310a60c477778a086a292b591ee50e2ce`. The extraction scripts accept the repository and checkpoint through `--dfer-repo` and `--checkpoint`; no model files are downloaded automatically.

## IEMOCAP

The input CSV must contain a `file` column. Each utterance video stem must match its `file` value.

```bash
export IEMOCAP_CSV=/path/to/iemocap.csv
export IEMOCAP_VIDEO_ROOT=/path/to/utterance_videos
export DFER_REPO=/path/to/DFER-CLIP
export DFER_CHECKPOINT=/path/to/FER39K-model.pth
export WORK_ROOT=/path/to/new_iemocap_visual_work
export VISIBLE_GPU=0
bash feature_extraction/scripts/run_iemocap.sh
```

## MELD

The MELD pipeline expects the official split CSVs and videos, processed utterance WAV files, and the public MELD-FAIR files `MELD_video_realignment_timestamps.csv` and `MELD_active_speaker_face_bboxes.csv`.

```bash
export MELD_RAW_ROOT=/path/to/MELD.Raw
export MELD_PROCESSED_ROOT=/path/to/processed_MELD
export MELD_TIMESTAMPS_CSV=/path/to/MELD_video_realignment_timestamps.csv
export MELD_BBOXES_CSV=/path/to/MELD_active_speaker_face_bboxes.csv
export DFER_REPO=/path/to/DFER-CLIP
export DFER_CHECKPOINT=/path/to/FER39K-model.pth
export WORK_ROOT=/path/to/new_meld_visual_work
export VISIBLE_GPU=0
bash feature_extraction/scripts/run_meld.sh
```

The expected official intersection is 9,988 train, 1,108 dev, and 2,610 test utterances. Stop and inspect the input versions if these counts differ.

## Output

Each `<file>.pt` payload contains:

| Field | Shape | Description |
|---|---:|---|
| `tokens` | `(16, 512)` | DFER-CLIP temporal tokens |
| `cls` | `(512,)` | Temporal CLS feature |
| `attention_mask` | `(16,)` | Binary valid-frame mask |
| `quality` | scalar | Continuous target-face observability `q` |
| `file_id` | string | Unique utterance identifier |

The scripts write to a new work directory and refuse to mix with an existing output directory.
