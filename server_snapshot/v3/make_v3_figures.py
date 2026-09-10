from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


WIDTH_INCHES = 16.4 / 2.54
COLORS = ("#28666e", "#b85c38", "#4f772d", "#6c5b7b", "#3d405b")
DATASET_LABELS = {
    "unsw": "UNSW-NB15",
    "nslkdd": "NSL-KDD",
    "cicids2017": "CIC-IDS-2017\n(cross-day)",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric(summary: dict, dataset: str, variant: str, name: str) -> tuple[float, float]:
    values = summary["performance"][dataset][variant][name]
    return float(values["mean"]), float(values["sample_sd"])


def save_figure(figure: plt.Figure, output: Path, stem: str) -> list[dict]:
    records = []
    for suffix in ("png", "pdf"):
        path = output / f"{stem}.{suffix}"
        options = {"dpi": 600} if suffix == "png" else {}
        figure.savefig(path, facecolor="white", **options)
        records.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    plt.close(figure)
    return records


def grouped_bars(
    ax: plt.Axes,
    datasets: list[str],
    series: list[tuple[str, list[float], list[float]]],
    ylabel: str,
) -> None:
    x = np.arange(len(datasets), dtype=np.float64)
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
    ax.set_xticks(x, [DATASET_LABELS[name] for name in datasets])
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#d7d7d7", linewidth=0.6)
    ax.set_axisbelow(True)


def create_figures(summary_path: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    datasets = list(summary["datasets"])
    plt.rcParams.update(
        {
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
        }
    )
    manifest: list[dict] = []

    ablation_variants = (
        ("SMOTE(-NC)+Tomek", "smotenc_tomek"),
        ("+ original weights", "smotenc_tomek_weight"),
        ("+ residual correction", "smotenc_tomek_residual_weight"),
    )
    series = []
    for label, variant in ablation_variants:
        values = [metric(summary, dataset, variant, "macro_f1") for dataset in datasets]
        series.append(
            (label, [item[0] for item in values], [item[1] for item in values])
        )
    figure, ax = plt.subplots(
        figsize=(WIDTH_INCHES, 8.2 / 2.54), constrained_layout=True
    )
    grouped_bars(ax, datasets, series, "Macro-F1")
    ax.legend(ncol=3, loc="upper center")
    manifest.extend(save_figure(figure, output, "v3_fig01_residual_weight_ablation"))

    comparison = []
    for label, source, name in (
        ("Raw XGBoost", "variant", "raw"),
        ("Balanced RF", "baseline", "BalancedRandomForest"),
        ("EasyEnsemble", "baseline", "EasyEnsemble"),
        ("Proposed", "variant", "smotenc_tomek_residual_weight"),
    ):
        means = []
        errors = []
        for dataset in datasets:
            if source == "variant":
                mean, error = metric(summary, dataset, name, "macro_f1")
            else:
                values = summary["raw_training_baselines"][dataset][name]["macro_f1"]
                mean, error = float(values["mean"]), float(values["sample_sd"])
            means.append(mean)
            errors.append(error)
        comparison.append((label, means, errors))
    figure, ax = plt.subplots(
        figsize=(WIDTH_INCHES, 8.2 / 2.54), constrained_layout=True
    )
    grouped_bars(ax, datasets, comparison, "Macro-F1")
    ax.legend(ncol=4, loc="upper center")
    manifest.extend(save_figure(figure, output, "v3_fig02_strong_baselines"))

    operating_series = []
    alert_series = []
    for label, variant in (
        ("Raw", "raw"),
        ("Proposed", "smotenc_tomek_residual_weight"),
    ):
        recall_values = [
            metric(
                summary,
                dataset,
                variant,
                "attack_recall_at_validation_1pct_fpr",
            )
            for dataset in datasets
        ]
        alert_values = [
            metric(
                summary,
                dataset,
                variant,
                "threshold_false_alerts_per_10000_benign",
            )
            for dataset in datasets
        ]
        operating_series.append(
            (
                label,
                [item[0] for item in recall_values],
                [item[1] for item in recall_values],
            )
        )
        alert_series.append(
            (
                label,
                [item[0] for item in alert_values],
                [item[1] for item in alert_values],
            )
        )
    figure, axes = plt.subplots(
        1, 2, figsize=(WIDTH_INCHES, 8.3 / 2.54), constrained_layout=True
    )
    grouped_bars(
        axes[0],
        datasets,
        operating_series,
        "Attack recall at validation 1% FPR",
    )
    grouped_bars(
        axes[1],
        datasets,
        alert_series,
        "False alerts per 10,000 benign flows",
    )
    axes[0].legend(ncol=2, loc="upper center")
    manifest.extend(save_figure(figure, output, "v3_fig03_operating_point"))

    stability_spearman = []
    stability_jaccard = []
    for dataset in datasets:
        values = summary["explanation"][dataset][
            "smotenc_tomek_residual_weight"
        ]
        stability_spearman.append(float(values["mean_seed_spearman"]))
        stability_jaccard.append(float(values["mean_seed_top10_jaccard"]))
    figure, ax = plt.subplots(
        figsize=(WIDTH_INCHES, 8.2 / 2.54), constrained_layout=True
    )
    grouped_bars(
        ax,
        datasets,
        [
            ("Spearman rank correlation", stability_spearman, [0.0] * len(datasets)),
            ("Top-10 Jaccard", stability_jaccard, [0.0] * len(datasets)),
        ],
        "Between-seed explanation stability",
    )
    ax.set_ylim(0.0, 1.0)
    ax.legend(ncol=2, loc="upper center")
    manifest.extend(save_figure(figure, output, "v3_fig04_explanation_stability"))

    result = {
        "source_summary": str(summary_path),
        "source_summary_sha256": sha256(summary_path),
        "width_cm": 16.4,
        "png_dpi": 600,
        "files": manifest,
    }
    (output / "figure_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        default="/opt/ids_revision/v3_strengthened/aggregate/summary.json",
    )
    parser.add_argument(
        "--output",
        default="/opt/ids_revision/v3_strengthened/aggregate/figures",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    created = create_figures(Path(args.summary), Path(args.output))
    print(json.dumps({"figure_files": len(created["files"])}))
