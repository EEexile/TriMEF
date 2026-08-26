# TriMEF

Code for **TriMEF: Pre- and Post-Interaction Expert Fusion with Target-Face Observability for Multimodal Emotion Recognition**.

This release contains the paper configuration, training and evaluation code, and the visual preprocessing pipeline used for IEMOCAP and MELD. It does not include datasets, pretrained encoders, extracted features, checkpoints, ablations, or analysis code.

## Structure

```text
TriMEF/
|-- iemocap/
|   |-- train_iemocap.py
|   |-- dataset.py
|   |-- model.py
|   `-- supporting model and loss modules
|-- meld/
|   |-- train_meld.py
|   |-- evaluate_meld.py
|   |-- dataset.py
|   |-- model.py
|   `-- supporting model and loss modules
`-- feature_extraction/
    |-- iemocap/
    |-- meld/
    |-- scripts/
    `-- README.md
```

## Requirements

- Python 3.10+
- PyTorch and torchaudio
- transformers
- NumPy
- pandas

Install PyTorch and torchaudio for your CUDA version, then install the remaining packages:

```bash
pip install transformers numpy pandas
```

## Data

IEMOCAP requires a CSV containing at least `file`, `emotion`, `text`, and `speaker`. Audio is read from `<audio-dir>/<file>.wav`; visual features are read from `<video-feature-dir>/<file>.pt`.

MELD requires a manifest containing at least `file`, `split`, `emotion`, `text`, `speaker`, `dialogue_id`, `utterance_id`, and `audio_path`. Official split names are `train`, `dev`, and `test`; visual features are read from `<video-feature-dir>/<file>.pt`.

## Visual preprocessing

The visual pipeline samples 16 target-face frames, builds the validity mask and continuous observability value, and exports frozen DFER-CLIP temporal features. See [`feature_extraction/README.md`](feature_extraction/README.md) for inputs and commands.

DFER-CLIP source code and checkpoints are external dependencies and are not redistributed in this repository.

## IEMOCAP

Run each held-out Session separately:

```bash
cd iemocap
python train_iemocap.py \
  --fold 1 \
  --device cuda:0 \
  --csv-path /path/to/iemocap.csv \
  --audio-dir /path/to/wav \
  --wav2vec2-path /path/to/wav2vec2-base \
  --roberta-path /path/to/roberta-base \
  --video-feature-dir /path/to/video_features \
  --run-name trimef_iemocap_fold1
```

Repeat with `--fold 2` through `--fold 5`. The paper protocol uses seed 42, batch size 16, 100 epochs, no early stopping, and held-out Session WA for checkpoint selection.

## MELD

Train with dev-WF1 checkpoint selection while deferring test evaluation:

```bash
cd meld
python train_meld.py \
  --manifest /path/to/meld_manifest.csv \
  --video-feature-dir /path/to/video_features \
  --wav2vec2-path /path/to/wav2vec2-base \
  --roberta-path /path/to/roberta-base \
  --output-root ./runs \
  --run-name trimef_meld_seed42 \
  --lambda-utec 0.1 \
  --device cuda:0 \
  --defer-test
```

Evaluate the selected checkpoint on test once:

```bash
python evaluate_meld.py \
  --dev-result ./runs/results/trimef_meld_seed42/dev_results.json \
  --device cuda:0
```

The MELD protocol uses seed 42, batch size 16, an initial 50-epoch budget, and an extension to 100 epochs only when the best dev epoch is between 41 and 50.

## Paper configuration

- Available-expert mean over Audio, Text, Video, and Joint logits
- CTC weight: 0.1
- UTEC weight: 0.1
- ExpertCE weight: 1.0
- Temperature: 0.07
- Label smoothing: 0
- Learning rate: `5e-5`
- Seed: 42
- Batch size: 16
