"""Aggregate the V4 deep-baseline runs using the V3 statistical protocol.

Means, sample standard deviations, and paired seed-level bootstrap intervals are
computed exactly as in ``aggregate_v3_results.py`` (20 000 resamples, seed
20260902) so the deep-model evidence is directly comparable with the reported
XGBoost and tree-ensemble results. Integrity verification runs here, on the
host that produced the artefacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Final

import numpy as np

ARCHITECTURES: Final = ("MLP",)
CONFIGURATIONS: Final = (
    "raw",
    "smotenc_tomek_weight",
    "smotenc_tomek_residual_weight",
)
HEADLINE_METRICS: Final = (
    "macro_f1",
    "macro_pr_auc_ovr",
    "attack_recall_at_validation_1pct_fpr",
    "threshold_false_alerts_per_10000_benign",
)
EXPECTED_SEEDS: Final = {
    "unsw": (42, 123, 456, 789, 1001, 2024, 31415, 65537, 77777, 99991),
    "nslkdd": (42, 123, 456, 789, 1001, 2024, 31415, 65537, 77777, 99991),
    "cicids2017": (42, 123, 456),
}
PERCENTAGE_POINT_METRICS: Final = (
    "macro_f1",
    "attack_recall_at_validation_1pct_fpr",
)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean_sd(values: list[float]) -> dict:
    """Return the mean and sample standard deviation of paired-seed values."""
    return {
        "values": values,
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def paired_seed_bootstrap(
    differences: list[float], *, repetitions: int = 20_000, seed: int = 20260902
) -> dict:
    """Bootstrap paired seed-level differences, matching the V3 protocol."""
    values = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        **mean_sd(values.tolist()),
        "bootstrap_repetitions": repetitions,
        "ci_level": 0.95,
        "ci_low": float(low),
        "ci_high": float(high),
        "probability_difference_above_zero": float(np.mean(means > 0.0)),
    }


def load_runs(results: Path) -> tuple[dict, list[dict]]:
    """Load every metrics.json and build the integrity manifest."""
    runs: dict[tuple[str, int], dict] = {}
    manifest: list[dict] = []
    for path in sorted(results.rglob("metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "v4_deep_baseline":
            raise ValueError(f"Unexpected schema in {path}")
        if payload.get("smoke"):
            raise ValueError(f"Smoke result must not be aggregated: {path}")
        key = (payload["dataset"], int(payload["seed"]))
        if key in runs:
            raise ValueError(f"Duplicate run: {key}")
        expected = {
            f"{architecture}::{configuration}"
            for architecture in ARCHITECTURES
            for configuration in CONFIGURATIONS
        }
        missing = expected - set(payload["runs"])
        if missing:
            raise ValueError(f"Missing configurations {sorted(missing)} in {path}")
        runs[key] = payload
        manifest.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "dataset": payload["dataset"],
                "seed": int(payload["seed"]),
                "device": payload["environment"]["device"],
            }
        )
    return runs, manifest


def verify_completeness(runs: dict) -> dict:
    """Confirm that every dataset has all of its expected seeds."""
    report: dict[str, dict] = {}
    for dataset, seeds in EXPECTED_SEEDS.items():
        present = sorted(seed for (name, seed) in runs if name == dataset)
        report[dataset] = {
            "expected": list(seeds),
            "present": present,
            "missing": [seed for seed in seeds if seed not in present],
            "complete": present == sorted(seeds),
        }
    return report


def metric_values(
    runs: dict, dataset: str, architecture: str, configuration: str, metric: str
) -> list[float]:
    """Collect one metric across the paired seeds of a dataset."""
    return [
        float(runs[(dataset, seed)]["runs"][f"{architecture}::{configuration}"]["metrics"][metric])
        for seed in EXPECTED_SEEDS[dataset]
        if (dataset, seed) in runs
    ]


def build_summary(runs: dict) -> dict:
    """Compute per-configuration summaries and paired differences."""
    summary: dict[str, dict] = {}
    differences: dict[str, dict] = {}
    for dataset in EXPECTED_SEEDS:
        if not any(name == dataset for (name, _seed) in runs):
            continue
        for architecture in ARCHITECTURES:
            for configuration in CONFIGURATIONS:
                key = f"{dataset}|{architecture}|{configuration}"
                summary[key] = {
                    metric: mean_sd(
                        metric_values(runs, dataset, architecture, configuration, metric)
                    )
                    for metric in HEADLINE_METRICS
                }
            for control in ("raw", "smotenc_tomek_weight"):
                paired: dict[str, dict] = {}
                for metric in HEADLINE_METRICS:
                    residual = metric_values(
                        runs, dataset, architecture, "smotenc_tomek_residual_weight", metric
                    )
                    baseline = metric_values(runs, dataset, architecture, control, metric)
                    scale = 100.0 if metric in PERCENTAGE_POINT_METRICS else 1.0
                    paired[metric] = paired_seed_bootstrap(
                        [(a - b) * scale for a, b in zip(residual, baseline)]
                    )
                differences[f"{dataset}|{architecture}|residual_minus_{control}"] = paired
    return {"per_configuration": summary, "paired_differences": differences}


def write_csv(summary: dict, path: Path) -> None:
    """Write a flat per-configuration table for manuscript use."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["dataset", "architecture", "configuration", "metric", "mean", "sample_sd", "seeds"]
        )
        for key, metrics in summary["per_configuration"].items():
            dataset, architecture, configuration = key.split("|")
            for metric, stats in metrics.items():
                writer.writerow(
                    [
                        dataset,
                        architecture,
                        configuration,
                        metric,
                        f"{stats['mean']:.10f}",
                        f"{stats['sample_sd']:.10f}",
                        len(stats["values"]),
                    ]
                )


def main() -> None:
    """Aggregate results and emit the publication-ready summary files."""
    parser = argparse.ArgumentParser(description="Aggregate V4 deep baseline results")
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    results = Path(args.results)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)

    runs, manifest = load_runs(results)
    completeness = verify_completeness(runs)
    summary = build_summary(runs)

    integrity = {
        "run_count": len(runs),
        "file_count": len(manifest),
        "total_bytes": sum(item["bytes"] for item in manifest),
        "completeness": completeness,
        "all_complete": all(item["complete"] for item in completeness.values()),
        "files": manifest,
    }
    (output / "v4_integrity.json").write_text(
        json.dumps(integrity, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "v4_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(summary, output / "v4_performance_summary.csv")
    print(
        json.dumps(
            {
                "runs": len(runs),
                "all_complete": integrity["all_complete"],
                "missing": {
                    name: report["missing"]
                    for name, report in completeness.items()
                    if report["missing"]
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
