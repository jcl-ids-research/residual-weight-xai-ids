"""Render the total-class-weight invariance figure.

The manuscript derives that residual weighting holds each class's total weight
at N'/K regardless of the resampling expansion factor, while reusing the
original weights lets that total drift. This figure plots the measured
maximum-to-minimum ratio of that total across all four datasets, so the
derivation and the audit can be checked against each other at a glance.

Style constants mirror ``make_v3_figures.py`` so the new panel matches the
existing figures in the submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WIDTH_INCHES: Final = 16.4 / 2.54
COLORS: Final = ("#b85c38", "#28666e")
DATASET_ORDER: Final = ("unsw", "nslkdd", "cicids2017", "cicddos2019")
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


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_ratios(mass: dict, ddos: dict) -> dict[str, tuple[float, float]]:
    """Return the original and residual total-weight ratios per dataset."""
    ratios: dict[str, tuple[float, float]] = {}
    for dataset in ("unsw", "nslkdd", "cicids2017"):
        entry = mass[dataset]
        ratios[dataset] = (
            float(entry["original_ratio"]["mean"]),
            float(entry["residual_ratio"]["mean"]),
        )
    ratios["cicddos2019"] = (
        float(ddos["weight_mass_ratio"]["smotenc_tomek_weight"]["mean"]),
        float(ddos["weight_mass_ratio"]["smotenc_tomek_residual_weight"]["mean"]),
    )
    return ratios


def render(ratios: dict[str, tuple[float, float]], output: Path, stem: str) -> list[dict]:
    """Draw and save the figure in the formats used by the submission."""
    plt.rcParams.update(RC_PARAMS)
    datasets = [name for name in DATASET_ORDER if name in ratios]
    original = [ratios[name][0] for name in datasets]
    residual = [ratios[name][1] for name in datasets]

    x = np.arange(len(datasets), dtype=np.float64)
    width = 0.32
    figure, ax = plt.subplots(figsize=(WIDTH_INCHES, 8.2 / 2.54))
    bars_original = ax.bar(
        x - width / 2,
        original,
        width=width,
        label="Original weights reused",
        color=COLORS[0],
        edgecolor="black",
        linewidth=0.45,
    )
    bars_residual = ax.bar(
        x + width / 2,
        residual,
        width=width,
        label="Post-cleaning residual weights",
        color=COLORS[1],
        edgecolor="black",
        linewidth=0.45,
    )
    ax.axhline(
        1.0,
        color="#3d405b",
        linewidth=0.9,
        linestyle="--",
        label="Equal total weight (ratio = 1)",
    )
    for bars in (bars_original, bars_residual):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}" if height >= 1.005 else f"{height:.6f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax.set_xticks(x, [DATASET_LABELS[name] for name in datasets])
    ax.set_ylabel("Max/min total class weight")
    ax.set_ylim(0.0, max(original) * 1.22)
    ax.grid(axis="y", color="#d7d7d7", linewidth=0.6)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    figure.legend(
        handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.99)
    )
    figure.subplots_adjust(left=0.10, right=0.99, bottom=0.20, top=0.84)

    output.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
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


def main() -> None:
    """Build the figure from the verified weight-mass audits."""
    parser = argparse.ArgumentParser(description="Render weight invariance figure")
    parser.add_argument("--mass", required=True)
    parser.add_argument("--ddos", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stem", default="fig06_weight_invariance")
    args = parser.parse_args()

    mass = json.loads(Path(args.mass).read_text(encoding="utf-8"))
    ddos = json.loads(Path(args.ddos).read_text(encoding="utf-8"))
    ratios = collect_ratios(mass, ddos)
    records = render(ratios, Path(args.out), args.stem)
    print(
        json.dumps(
            {
                "ratios": {k: {"original": v[0], "residual": v[1]} for k, v in ratios.items()},
                "files": records,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
