"""Regression tests for aggregation from the evidence shipped publicly."""

from __future__ import annotations

import json
from pathlib import Path

from rwxai.aggregate_v3_results import aggregate
from rwxai.aggregate_v4_results import main as aggregate_v4_main

REPO = Path(__file__).resolve().parents[1]


def test_v3_metrics_only_recreates_published_tables(tmp_path: Path) -> None:
    """Given public metrics, When aggregated, Then four CSVs match exactly."""
    generated = tmp_path / "v3"

    aggregate(
        REPO / "evidence" / "v3" / "per_seed",
        generated,
        require_prediction_arrays=False,
    )

    published = REPO / "evidence" / "v3" / "aggregate"
    for name in (
        "performance_summary.csv",
        "baseline_summary.csv",
        "per_class_summary.csv",
        "explanation_summary.csv",
    ):
        assert (generated / name).read_bytes() == (published / name).read_bytes(), name


def test_v4_missing_runs_returns_failure(tmp_path: Path) -> None:
    """Given no runs, When aggregated, Then the command reports failure."""
    results = tmp_path / "results"
    output = tmp_path / "output"
    results.mkdir()

    exit_code = aggregate_v4_main(["--results", str(results), "--out", str(output)])

    assert exit_code == 1
    payload = json.loads((output / "v4_integrity.json").read_text(encoding="utf-8"))
    assert payload["all_complete"] is False
