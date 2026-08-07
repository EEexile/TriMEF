#!/usr/bin/env python3
"""Evaluate a dev-selected TriMEF checkpoint on MELD test once."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

import train_meld as trainer
from dataset import MELDMMERDataset
from training import atomic_json_save, set_seed
from model_factory import (
    CONTEXT_CONFIGURATION,
    DECISION_MOE_CONFIGURATION,
    build_model,
    model_audit,
)


EXPECTED_SPLITS = {"train": 9988, "dev": 1108, "test": 2610}
EXPECTED_WEIGHTS = {"ctc": 0.1, "utec": 0.1, "expert_ce": 1.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-result", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def close(value: object, expected: float) -> bool:
    return math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-12)


def main() -> int:
    args = parse_args()
    record = read_json(args.dev_result)
    if record.get("status") != "dev_complete":
        raise AssertionError("run is not dev_complete")
    if record.get("split") != EXPECTED_SPLITS:
        raise AssertionError(f"locked MELD split drift: {record.get('split')}")
    if record.get("protocol", {}).get("test_evaluations") != 0:
        raise AssertionError("run accessed test before finalization")
    configuration = record["configuration"]
    for name, expected in EXPECTED_WEIGHTS.items():
        key = {
            "ctc": "ctc_weight",
            "utec": "lambda_utec",
            "expert_ce": "lambda_expert_ce",
        }[name]
        if not close(configuration[key], expected):
            raise AssertionError(f"{name} coefficient drift: {configuration[key]}")

    output_dir = args.dev_result.parent
    result_path = output_dir / "results.json"
    prediction_path = output_dir / "test_predictions.json"
    if result_path.exists() or prediction_path.exists():
        raise FileExistsError("formal test output already exists")
    checkpoint = Path(record["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if trainer.sha256_file(checkpoint) != record["checkpoint_sha256"]:
        raise AssertionError("best checkpoint hash drift")

    paths = {name: Path(value) for name, value in record["paths"].items()}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"missing locked path {name}: {path}")
    if trainer.sha256_file(paths["manifest"]) != record["source_hashes"]["manifest"]:
        raise AssertionError("manifest hash drift")

    set_seed(int(configuration["seed"]))
    dataset = MELDMMERDataset(
        csv_path=str(paths["manifest"]),
        roberta_path=str(paths["roberta"]),
        max_text_length=CONTEXT_CONFIGURATION["max_text_length"],
        video_feature_dir=str(paths["video_features"]),
        use_dialogue_context=True,
    )
    frame = pd.read_csv(paths["manifest"])
    split_indices = {
        split: frame.index[frame["split"].astype(str) == split].tolist()
        for split in ("train", "dev", "test")
    }
    counts = {name: len(indices) for name, indices in split_indices.items()}
    if counts != EXPECTED_SPLITS:
        raise AssertionError(f"locked MELD split drift: {counts}")

    device = torch.device(args.device)
    model_args = argparse.Namespace(
        wav2vec2_path=str(paths["wav2vec2"]),
        roberta_path=str(paths["roberta"]),
    )
    model = build_model(model_args, dataset, dict(record["locked_recipe"])).to(device)
    selected_state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(selected_state, strict=True)
    audit = model_audit(model)
    if DECISION_MOE_CONFIGURATION.get("gate") != "fixed_equal_available_mean":
        raise AssertionError("fixed uniform aggregation is not configured")
    if getattr(model.decision_moe, "aggregation", None) != "fixed_equal_available_mean":
        raise AssertionError("fixed uniform aggregation is not active")

    test_loader, _ = trainer.make_loader(
        dataset,
        split_indices["test"],
        int(configuration["batch_size"]),
        False,
        2,
    )
    test_detail = trainer.evaluate_detailed(model, test_loader, device, dataset)
    predictions = test_detail.pop("predictions")
    atomic_json_save(
        {
            "schema_version": trainer.RESULT_SCHEMA_VERSION,
            "run_name": record["run_name"],
            "selection_rule": "own_max_dev_wf1_then_single_test",
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": record["checkpoint_sha256"],
            "class_order": test_detail["class_order"],
            "predictions": predictions,
        },
        prediction_path,
    )

    result = dict(record)
    result["status"] = "success"
    result["scientific_status"] = "MELD_own_dev_selected_test_once"
    result["ended_at"] = datetime.now().isoformat()
    result["protocol"] = {
        **record["protocol"],
        "test_evaluations": 1,
        "test_used_for_model_selection": False,
    }
    result["test"] = test_detail
    result["finalizer"] = {
        "source": str(Path(__file__).resolve()),
        "source_sha256": trainer.sha256_file(Path(__file__)),
        "model_audit": audit,
        "environment": trainer.environment_record(device),
    }
    atomic_json_save(result, result_path)
    atomic_json_save(
        {
            "schema_version": trainer.RESULT_SCHEMA_VERSION,
            "status": "success",
            "run_name": record["run_name"],
            "epoch": record["protocol"]["epochs_completed"],
            "dev_evaluations": record["protocol"]["dev_evaluations"],
            "test_evaluations": 1,
            "best_epoch": record["best_epoch"],
            "best_dev": record["best_dev"],
            "test": {key: test_detail[key] for key in ("wa", "ua", "wf1")},
            "result": str(result_path.resolve()),
        },
        output_dir / "progress.json",
    )
    print(
        json.dumps(
            {"run_name": record["run_name"], "test": test_detail}, ensure_ascii=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
