#!/usr/bin/env python3
"""Build a validated MELD audio/video intersection from official split CSVs."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd


EXPECTED_COUNTS = {'train': 9988, 'dev': 1108, 'test': 2610}
VIDEO_DIRS = {
    'train': 'train_splits',
    'dev': 'dev_splits_complete',
    'test': 'output_repeated_splits_test',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-root', required=True)
    parser.add_argument('--processed-root', required=True)
    parser.add_argument('--output-root', required=True)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(frame, path):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.tmp')
    frame.to_csv(temporary, index=False)
    with temporary.open('rb') as handle:
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def atomic_json(payload, path):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.tmp')
    with temporary.open('x', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def main():
    args = parse_args()
    raw_root = Path(args.raw_root).resolve()
    processed_root = Path(args.processed_root).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=False)

    all_rows = []
    manifest = {
        'schema_version': 1,
        'kind': 'meld_official_audio_video_common_intersection',
        'raw_root': str(raw_root),
        'processed_root': str(processed_root),
        'splits': {},
    }
    for split in ('train', 'dev', 'test'):
        official_path = raw_root / f'{split}_sent_emo.csv'
        processed_csv = processed_root / split / f'meld_{split}.csv'
        audio_root = processed_root / split / 'wavs'
        video_root = raw_root / VIDEO_DIRS[split]
        for path in (official_path, processed_csv, audio_root, video_root):
            if not path.exists():
                raise FileNotFoundError(path)
        official = pd.read_csv(official_path)
        if official.duplicated(['Dialogue_ID', 'Utterance_ID']).any():
            raise RuntimeError(f'Duplicate official IDs in {split}')
        processed = pd.read_csv(processed_csv)
        processed_ids = set(processed['file'].astype(str))
        rows, missing_video, missing_audio, missing_processed = [], [], [], []
        for row in official.to_dict('records'):
            dialogue = int(row['Dialogue_ID'])
            utterance = int(row['Utterance_ID'])
            file_id = f'meld_{split}_dia{dialogue}_utt{utterance}'
            raw_stem = f'dia{dialogue}_utt{utterance}'
            video_path = video_root / (raw_stem + '.mp4')
            audio_path = audio_root / (file_id + '.wav')
            if not video_path.is_file():
                missing_video.append(file_id)
            if not audio_path.is_file():
                missing_audio.append(file_id)
            if file_id not in processed_ids:
                missing_processed.append(file_id)
            if video_path.is_file() and audio_path.is_file() and file_id in processed_ids:
                rows.append({
                    'file': file_id,
                    'split': split,
                    'emotion': str(row['Emotion']),
                    'text': str(row['Utterance']),
                    'speaker': str(row['Speaker']),
                    'dialogue_id': dialogue,
                    'utterance_id': utterance,
                    'video_path': str(video_path),
                    'audio_path': str(audio_path),
                })
        frame = pd.DataFrame(rows).sort_values(['dialogue_id', 'utterance_id'])
        if len(frame) != EXPECTED_COUNTS[split]:
            raise RuntimeError(
                f'{split} common count {len(frame)} != {EXPECTED_COUNTS[split]}'
            )
        if frame['file'].duplicated().any():
            raise RuntimeError(f'Duplicate normalized IDs in {split}')
        atomic_csv(frame, output_root / f'dataset_{split}.csv')
        atomic_csv(
            frame[['file', 'split', 'video_path']],
            output_root / f'visual_{split}.csv',
        )
        all_rows.append(frame)
        manifest['splits'][split] = {
            'official_count': len(official),
            'common_count': len(frame),
            'missing_video': sorted(set(missing_video)),
            'missing_audio': sorted(set(missing_audio)),
            'missing_processed_csv': sorted(set(missing_processed)),
            'official_csv': str(official_path),
            'official_csv_sha256': sha256_file(official_path),
            'processed_csv': str(processed_csv),
            'processed_csv_sha256': sha256_file(processed_csv),
        }
    combined = pd.concat(all_rows, ignore_index=True)
    if len(combined) != sum(EXPECTED_COUNTS.values()):
        raise RuntimeError('Combined MELD intersection count drift')
    atomic_csv(combined, output_root / 'dataset_all.csv')
    atomic_csv(
        combined[['file', 'split', 'video_path']],
        output_root / 'visual_all.csv',
    )
    manifest['combined_count'] = len(combined)
    manifest['emotion_counts'] = {
        split: combined[combined['split'] == split]['emotion'].value_counts().to_dict()
        for split in ('train', 'dev', 'test')
    }
    atomic_json(manifest, output_root / 'manifest.json')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
