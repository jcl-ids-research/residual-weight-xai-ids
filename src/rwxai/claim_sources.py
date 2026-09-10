"""Typed access to the published evidence behind manuscript result tables."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

PROPOSED: Final = "smotenc_tomek_residual_weight"
V3_SEEDS: Final = {
    "unsw": (42, 123, 456, 789, 1001, 2024, 31415, 65537, 77777, 99991),
    "nslkdd": (42, 123, 456, 789, 1001, 2024, 31415, 65537, 77777, 99991),
    "cicids2017": (42, 123, 456),
}


@dataclass(frozen=True, slots=True)
class Summary:
    """A mean and sample standard deviation."""

    mean: float
    sample_sd: float


@dataclass(frozen=True, slots=True)
class Interval:
    """A paired mean difference and its percentile interval."""

    mean: float
    low: float
    high: float


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


class ClaimSources:
    """Index every committed source used by Tables 3-8."""

    def __init__(self, root: Path) -> None:
        evidence = root / "evidence"
        v3 = evidence / "v3" / "aggregate"
        v4 = evidence / "v4" / "aggregate"
        self.evidence = evidence
        self.performance = self._summary_index(
            v3 / "performance_summary.csv", ("dataset", "variant", "metric")
        )
        self.baselines = self._summary_index(
            v3 / "baseline_summary.csv", ("dataset", "baseline", "metric")
        )
        self.per_class = self._summary_index(
            v3 / "per_class_summary.csv",
            ("dataset", "class", "variant", "metric"),
        )
        self.performance_values = {
            (row["dataset"], row["variant"], row["metric"]): tuple(
                float(value) for value in json.loads(row["seed_values_json"])
            )
            for row in _rows(v3 / "performance_summary.csv")
        }
        self.explanations = {
            (row["dataset"], row["variant"], row["comparison"]): row
            for row in _rows(v3 / "explanation_summary.csv")
        }
        self.ddos = json.loads(
            (v4 / "ddos2019_summary.json").read_text(encoding="utf-8")
        )
        self.v4 = json.loads((v4 / "v4_summary.json").read_text(encoding="utf-8"))

    @staticmethod
    def _summary_index(
        path: Path, keys: tuple[str, ...]
    ) -> dict[tuple[str, ...], Summary]:
        return {
            tuple(row[key] for key in keys): Summary(
                float(row["mean"]), float(row["sample_sd"])
            )
            for row in _rows(path)
        }

    def paired_v3(self, dataset: str, comparator: str, metric: str) -> Interval:
        """Recompute the manuscript's paired seed-level interval."""
        residual = np.asarray(
            self.performance_values[(dataset, PROPOSED, metric)], dtype=np.float64
        )
        baseline = np.asarray(
            self.performance_values[(dataset, comparator, metric)], dtype=np.float64
        )
        differences = residual - baseline
        rng = np.random.default_rng(20260902)
        indices = rng.integers(0, len(differences), size=(20_000, len(differences)))
        means = differences[indices].mean(axis=1)
        low, high = np.quantile(means, [0.025, 0.975])
        return Interval(float(differences.mean()), float(low), float(high))

    def ddos_interval(self, comparator: str, metric: str) -> Interval:
        entry = self.ddos["paired_differences"][f"residual_minus_{comparator}"][metric]
        return Interval(
            float(entry["mean"]), float(entry["ci_low"]), float(entry["ci_high"])
        )

    def v4_interval(self, dataset: str, comparator: str, metric: str) -> Interval:
        entry = self.v4["paired_differences"][
            f"{dataset}|MLP|residual_minus_{comparator}"
        ][metric]
        return Interval(
            float(entry["mean"]), float(entry["ci_low"]), float(entry["ci_high"])
        )

    def support(self, dataset: str, class_name: str) -> int:
        """Read fixed test support from one representative run."""
        if dataset == "cicddos2019":
            return int(self.ddos["per_class"][class_name]["support"])
        seed = V3_SEEDS[dataset][0]
        payload = json.loads(
            (
                self.evidence
                / "v3"
                / "per_seed"
                / dataset
                / f"seed{seed}"
                / "metrics.json"
            ).read_text(encoding="utf-8")
        )
        return int(
            payload["variants"]["raw"]["metrics"]["per_class"][class_name]["support"]
        )

    def max_additivity_error(self, dataset: str) -> float:
        """Return the largest proposed-path TreeSHAP additivity error."""
        if dataset == "cicddos2019":
            return float(self.ddos["explanation"]["max_additivity_error"])
        values = []
        for seed in V3_SEEDS[dataset]:
            payload = json.loads(
                (
                    self.evidence
                    / "v3"
                    / "per_seed"
                    / dataset
                    / f"seed{seed}"
                    / "metrics.json"
                ).read_text(encoding="utf-8")
            )
            values.append(
                float(
                    payload["variants"][PROPOSED]["tree_shap"][
                        "additivity_max_abs_error"
                    ]
                )
            )
        return max(values)


def mean_sd_cell(summary: Summary, decimals: int, scale: float = 1.0) -> str:
    """Format a manuscript mean-plus-minus cell."""
    return (
        f"{summary.mean * scale:.{decimals}f}±{summary.sample_sd * scale:.{decimals}f}"
    )


def interval_cell(interval: Interval, decimals: int, scale: float = 1.0) -> str:
    """Format a signed manuscript paired-difference cell."""
    return (
        f"{interval.mean * scale:+.{decimals}f} "
        f"[{interval.low * scale:+.{decimals}f}, {interval.high * scale:+.{decimals}f}]"
    )
