from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np

from v3_data import LOADERS, class_counts


def validate(args: argparse.Namespace) -> dict:
    directories = {
        "unsw": args.unsw_dir,
        "nslkdd": args.nslkdd_dir,
        "cicids2017": args.cicids2017_dir,
    }
    selected = list(LOADERS) if args.dataset == "all" else [args.dataset]
    reports = {}
    for dataset in selected:
        raw = LOADERS[dataset](directories[dataset], smoke=args.smoke)
        if raw.X_train.columns.tolist() != raw.X_test.columns.tolist():
            raise RuntimeError(f"Feature schema mismatch for {dataset}")
        if len(raw.X_train) != len(raw.y_train):
            raise RuntimeError(f"Training row/label mismatch for {dataset}")
        if len(raw.X_test) != len(raw.y_test):
            raise RuntimeError(f"Test row/label mismatch for {dataset}")
        n_classes = len(raw.class_names)
        if set(np.unique(raw.y_train)) != set(range(n_classes)):
            raise RuntimeError(f"Training class coverage failed for {dataset}")
        if set(np.unique(raw.y_test)) != set(range(n_classes)):
            raise RuntimeError(f"Test class coverage failed for {dataset}")
        reports[dataset] = {
            "train_rows": int(len(raw.y_train)),
            "test_rows": int(len(raw.y_test)),
            "features": int(raw.X_train.shape[1]),
            "categorical_features": len(raw.categorical_columns),
            "class_names": raw.class_names,
            "normal_class_name": raw.normal_class_name,
            "train_class_counts": class_counts(raw.y_train, n_classes),
            "test_class_counts": class_counts(raw.y_test, n_classes),
            "protocol": raw.protocol,
        }
        print(json.dumps({dataset: reports[dataset]}, ensure_ascii=False), flush=True)
        del raw
        gc.collect()
    return {
        "gate": "passed",
        "smoke": args.smoke,
        "datasets": reports,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=("all", *sorted(LOADERS)), default="all"
    )
    parser.add_argument("--unsw-dir", default="/opt/UNSW-NB15")
    parser.add_argument("--nslkdd-dir", default="/opt/NSL-KDD")
    parser.add_argument(
        "--cicids2017-dir",
        default="/opt/CIC-IDS-2017/MachineLearningCVE",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output",
        default="/opt/ids_revision/v3_strengthened/data_gate.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    report = validate(arguments)
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OK] {output}", flush=True)
