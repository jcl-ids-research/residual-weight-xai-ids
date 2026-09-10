"""Regenerate figures 2-5 with the fourth dataset included.

The original panels were produced before CIC-DDoS2019 was added, so they cover
only three datasets while the corresponding tables now cover four. This module
merges the V3 aggregates with the CIC-DDoS2019 summary and redraws the same
four panels using the style constants of ``make_v3_figures.py``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WIDTH_INCHES: Final = 16.4 / 2.54
COLORS: Final = ("#28666e", "#b85c38", "#4f772d", "#6c5b7b", "#3d405b")
DATASETS: Final = ("unsw", "nslkdd", "cicids2017", "cicddos2019")
DATASET_LABELS: Final = {
    "unsw": "UNSW-NB15",
    "nslkdd": "NSL-KDD",
    "cicids2017": "CIC-IDS-2017\n(cross-day)",
    "cicddos2019": "CIC-DDoS2019\n(cross-day)",
}
RC_PARAMS: Final = {
    "font.family": "serif",
    "font.size": 8.5,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}
PROPOSED: Final = "smotenc_tomek_residual_weight"


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


V3_SEEDS: Final = {
    "unsw": (42, 123, 456, 789, 1001, 2024, 31415, 65537, 77777, 99991),
    "nslkdd": (42, 123, 456, 789, 1001, 2024, 31415, 65537, 77777, 99991),
    "cicids2017": (42, 123, 456),
}


def rank_vector(values: np.ndarray) -> np.ndarray:
    """Return average ranks, assigning tied values their mean rank."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    ordered = values[order]
    start = 0
    for position in range(1, len(ordered) + 1):
        if position == len(ordered) or ordered[position] != ordered[start]:
            if position - start > 1:
                ranks[order[start:position]] = ranks[order[start:position]].mean()
            start = position
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Spearman rank correlation between two importance vectors."""
    return float(np.corrcoef(rank_vector(left), rank_vector(right))[0, 1])


def top_k_jaccard(left: np.ndarray, right: np.ndarray, k: int = 10) -> float:
    """Jaccard overlap of the top-k features of two importance vectors."""
    left_top = set(np.argsort(-left)[:k].tolist())
    right_top = set(np.argsort(-right)[:k].tolist())
    return len(left_top & right_top) / len(left_top | right_top)


def between_pair_spread(runs: Path, dataset: str) -> tuple[float, float] | None:
    """Recompute the between-seed stability spread from per-run TreeSHAP output."""
    vectors: list[np.ndarray] = []
    reference: list[str] | None = None
    for seed in V3_SEEDS.get(dataset, ()):  # noqa: B007 - explicit seed list
        path = runs / dataset / f"seed{seed}" / "metrics.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        mapping = payload["variants"][PROPOSED]["tree_shap"]["global_importance"]
        names = list(mapping)
        if reference is None:
            reference = names
        elif names != reference:
            return None
        vectors.append(np.asarray([mapping[name] for name in names], dtype=np.float64))
    if len(vectors) < 2:
        return None
    pairs = [
        (spearman(vectors[i], vectors[j]), top_k_jaccard(vectors[i], vectors[j]))
        for i in range(len(vectors))
        for j in range(i + 1, len(vectors))
    ]
    spearman_values = [item[0] for item in pairs]
    jaccard_values = [item[1] for item in pairs]
    return (
        float(np.std(spearman_values, ddof=1)),
        float(np.std(jaccard_values, ddof=1)),
    )


class Source:
    """Unified access to the V3 aggregates and the CIC-DDoS2019 summary."""

    def __init__(self, tables: Path, ddos: Path, runs: Path | None = None) -> None:
        self.performance = self._index(
            tables / "performance_summary.csv", ("dataset", "variant", "metric")
        )
        self.baselines = self._index(
            tables / "baseline_summary.csv", ("dataset", "baseline", "metric")
        )
        self.explanation = list(
            self._rows(tables / "explanation_summary.csv")
        )
        self.ddos = json.loads(ddos.read_text(encoding="utf-8"))
        self.stability_spread: dict[str, tuple[float, float]] = {}
        if runs is not None and runs.exists():
            for dataset in V3_SEEDS:
                spread = between_pair_spread(runs, dataset)
                if spread is not None:
                    self.stability_spread[dataset] = spread

    @staticmethod
    def _rows(path: Path) -> list[dict[str, str]]:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    def _index(self, path: Path, keys: tuple[str, ...]) -> dict:
        return {
            tuple(row[key] for key in keys): (
                float(row["mean"]),
                float(row["sample_sd"]),
            )
            for row in self._rows(path)
        }

    def variant(self, dataset: str, variant: str, metric: str) -> tuple[float, float]:
        """Return mean and standard deviation for one configuration."""
        if dataset == "cicddos2019":
            entry = self.ddos["per_variant"][variant][metric]
            return float(entry["mean"]), float(entry["sample_sd"])
        return self.performance[(dataset, variant, metric)]

    def baseline(self, dataset: str, name: str, metric: str) -> tuple[float, float]:
        """Return mean and standard deviation for one strong baseline."""
        if dataset == "cicddos2019":
            entry = self.ddos["baselines"][name][metric]
            return float(entry["mean"]), float(entry["sample_sd"])
        return self.baselines[(dataset, name, metric)]

    def stability_sd(self, dataset: str) -> tuple[float, float]:
        """Return the between-pair spread of the two stability measures.

        The V3 aggregate stores only the mean for the first three datasets, so
        their spread is recomputed from the per-run TreeSHAP vectors when those
        are available and reported as zero-width only when a single pair exists.
        """
        if dataset == "cicddos2019":
            entry = self.ddos["explanation"]
            return (
                float(entry["between_seed_spearman"].get("sample_sd", 0.0)),
                float(entry["between_seed_top10_jaccard"].get("sample_sd", 0.0)),
            )
        spread = self.stability_spread.get(dataset)
        if spread is None:
            return (0.0, 0.0)
        return spread

    def stability(self, dataset: str) -> tuple[float, float]:
        """Return between-seed Spearman and top-10 Jaccard for the proposal."""
        if dataset == "cicddos2019":
            entry = self.ddos["explanation"]
            return (
                float(entry["between_seed_spearman"]["mean"]),
                float(entry["between_seed_top10_jaccard"]["mean"]),
            )
        row = next(
            item
            for item in self.explanation
            if item["dataset"] == dataset
            and item["comparison"] == "between_seed_stability"
            and item["variant"] == PROPOSED
        )
        return float(row["mean_spearman"]), float(row["mean_top10_jaccard"])


def grouped_bars(
    ax: plt.Axes,
    series: list[tuple[str, list[float], list[float]]],
    ylabel: str,
) -> None:
    """Draw grouped bars using the shared submission style."""
    x = np.arange(len(DATASETS), dtype=np.float64)
    width = min(0.22, 0.72 / len(series))
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2) * width
    for index, (label, means, errors) in enumerate(series):
        ax.bar(
            x + offsets[index],
            means,
            width=width,
            label=label,
            color=COLORS[index % len(COLORS)],
            edgecolor="black",
            linewidth=0.45,
            yerr=errors,
            capsize=2.2,
            error_kw={"elinewidth": 0.8, "capthick": 0.8},
        )
    ax.set_xticks(x, [DATASET_LABELS[name] for name in DATASETS])
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#d7d7d7", linewidth=0.6)
    ax.set_axisbelow(True)


def save(figure: plt.Figure, output: Path, stem: str) -> list[dict]:
    """Write the figure in the formats used by the submission."""
    records = []
    for suffix in ("png", "pdf", "svg"):
        path = output / f"{stem}.{suffix}"
        options = {"dpi": 600} if suffix == "png" else {}
        figure.savefig(
            path, facecolor="white", bbox_inches="tight", pad_inches=0.03, **options
        )
        records.append(
            {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        )
    plt.close(figure)
    return records


def figure_baselines(source: Source, output: Path) -> list[dict]:
    """Figure 2: raw XGBoost, imbalance ensembles, and the proposal."""
    series: list[tuple[str, list[float], list[float]]] = []
    for label, getter in (
        ("Raw XGBoost", lambda d: source.variant(d, "raw", "macro_f1")),
        (
            "Balanced RF",
            lambda d: source.baseline(d, "BalancedRandomForest", "macro_f1"),
        ),
        ("EasyEnsemble", lambda d: source.baseline(d, "EasyEnsemble", "macro_f1")),
        ("Proposed", lambda d: source.variant(d, PROPOSED, "macro_f1")),
    ):
        values = [getter(dataset) for dataset in DATASETS]
        series.append(
            (label, [item[0] for item in values], [item[1] for item in values])
        )
    figure, ax = plt.subplots(figsize=(WIDTH_INCHES, 8.2 / 2.54))
    grouped_bars(ax, series, "Macro-F1")
    upper = max(
        mean + error for _, means, errors in series for mean, error in zip(means, errors)
    )
    ax.set_ylim(0.0, upper + 0.045)
    handles, labels = ax.get_legend_handles_labels()
    figure.legend(
        handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.99)
    )
    figure.subplots_adjust(left=0.10, right=0.99, bottom=0.21, top=0.81)
    return save(figure, output, "fig02_baseline_macro_f1")


def figure_ablation(source: Source, output: Path) -> list[dict]:
    """Figure 3: residual-weight ablation under identical resampling."""
    series: list[tuple[str, list[float], list[float]]] = []
    for label, variant in (
        ("SMOTE(-NC)+Tomek", "smotenc_tomek"),
        ("+ original weights", "smotenc_tomek_weight"),
        ("+ residual correction", PROPOSED),
    ):
        values = [source.variant(dataset, variant, "macro_f1") for dataset in DATASETS]
        series.append(
            (label, [item[0] for item in values], [item[1] for item in values])
        )
    figure, ax = plt.subplots(figsize=(WIDTH_INCHES, 8.2 / 2.54))
    grouped_bars(ax, series, "Macro-F1")
    upper = max(
        mean + error for _, means, errors in series for mean, error in zip(means, errors)
    )
    ax.set_ylim(0.0, upper + 0.045)
    handles, labels = ax.get_legend_handles_labels()
    figure.legend(
        handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.99)
    )
    figure.subplots_adjust(left=0.10, right=0.99, bottom=0.21, top=0.81)
    return save(figure, output, "fig03_residual_ablation")


def figure_operating_point(source: Source, output: Path) -> list[dict]:
    """Figure 4: budget recall and false alerts for raw versus the proposal.

    The alert panel uses a logarithmic axis because CIC-DDoS2019 carries a far
    larger alert scale and a seed-to-seed spread wider than its own mean; a
    linear axis would compress the other three datasets into invisibility.
    """
    # Four datasets cannot share one row: side-by-side panels collapse the
    # category labels into each other, so the panels are stacked instead.
    figure, axes = plt.subplots(2, 1, figsize=(WIDTH_INCHES, 13.6 / 2.54))
    recall_series: list[tuple[str, list[float], list[float]]] = []
    alert_series: list[tuple[str, list[float], list[float]]] = []
    for label, variant in (("Raw", "raw"), ("Proposed", PROPOSED)):
        recall = [
            source.variant(dataset, variant, "attack_recall_at_validation_1pct_fpr")
            for dataset in DATASETS
        ]
        alerts = [
            source.variant(
                dataset, variant, "threshold_false_alerts_per_10000_benign"
            )
            for dataset in DATASETS
        ]
        recall_series.append(
            (label, [item[0] for item in recall], [item[1] for item in recall])
        )
        alert_series.append(
            (label, [item[0] for item in alerts], [item[1] for item in alerts])
        )

    for ax, series, ylabel in (
        (axes[0], recall_series, "Attack recall at validation 1% FPR"),
        (axes[1], alert_series, "False alerts per 10,000 benign flows"),
    ):
        x = np.arange(len(DATASETS), dtype=np.float64)
        width = 0.32
        offsets = (np.arange(len(series)) - (len(series) - 1) / 2) * width
        logarithmic = ax is axes[1]
        for index, (label, means, errors) in enumerate(series):
            if logarithmic:
                # A log axis cannot render a negative lower bound, so the lower
                # whisker is clipped at the axis floor and the caption states it.
                floor = 1.0
                lower = [
                    max(mean - error, floor) * 0 + (mean - max(mean - error, floor))
                    for mean, error in zip(means, errors)
                ]
                yerr = [lower, list(errors)]
            else:
                yerr = list(errors)
            ax.bar(
                x + offsets[index],
                means,
                width=width,
                label=label,
                color=COLORS[index % len(COLORS)],
                edgecolor="black",
                linewidth=0.45,
                yerr=yerr,
                capsize=2.2,
                error_kw={"elinewidth": 0.8, "capthick": 0.8},
            )
        ax.set_xticks(x, [DATASET_LABELS[name] for name in DATASETS])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#d7d7d7", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylim(0.0, 1.08)
    axes[1].set_yscale("log")
    axes[1].set_ylim(1.0, 4000.0)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.995)
    )
    figure.subplots_adjust(
        left=0.11, right=0.99, bottom=0.13, top=0.92, hspace=0.55
    )
    return save(figure, output, "fig04_operating_point")


def figure_stability(source: Source, output: Path) -> list[dict]:
    """Figure 5: between-seed TreeSHAP stability of the proposal."""
    spearman = [source.stability(dataset)[0] for dataset in DATASETS]
    jaccard = [source.stability(dataset)[1] for dataset in DATASETS]
    spearman_sd = [source.stability_sd(dataset)[0] for dataset in DATASETS]
    jaccard_sd = [source.stability_sd(dataset)[1] for dataset in DATASETS]
    series = [
        ("Spearman rank correlation", spearman, spearman_sd),
        ("Top-10 Jaccard", jaccard, jaccard_sd),
    ]
    figure, ax = plt.subplots(figsize=(WIDTH_INCHES, 8.2 / 2.54))
    grouped_bars(ax, series, "Between-seed explanation stability")
    ax.set_ylim(0.0, 1.12)
    handles, labels = ax.get_legend_handles_labels()
    figure.legend(
        handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.99)
    )
    figure.subplots_adjust(left=0.10, right=0.99, bottom=0.21, top=0.83)
    return save(figure, output, "fig05_explanation_stability")


def main() -> None:
    """Regenerate the four panels that needed the fourth dataset."""
    parser = argparse.ArgumentParser(description="Rebuild figures 2-5")
    parser.add_argument("--tables", required=True)
    parser.add_argument("--ddos", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--runs", default=None)
    args = parser.parse_args()

    plt.rcParams.update(RC_PARAMS)
    source = Source(
        Path(args.tables),
        Path(args.ddos),
        Path(args.runs) if args.runs else None,
    )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "fig02": figure_baselines(source, output),
        "fig03": figure_ablation(source, output),
        "fig04": figure_operating_point(source, output),
        "fig05": figure_stability(source, output),
    }
    print(json.dumps({k: [f["file"] for f in v] for k, v in manifest.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
