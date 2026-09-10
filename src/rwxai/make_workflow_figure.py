"""Redraw the workflow figure so it matches the study's current scope.

Three corrections over the earlier version: the fourth dataset is listed, the
multilayer-perceptron extension appears as its own model-fitting branch, and
encoding is split into the integer step that precedes resampling and the
one-hot expansion that follows it, which the single "Encoding" box previously
made ambiguous.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

WIDTH_INCHES: Final = 16.4 / 2.54
HEIGHT_INCHES: Final = 8.4 / 2.54
RC_PARAMS: Final = {
    "font.family": "serif",
    "font.size": 7.4,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}
FILLS: Final = ("#dcebe3", "#ece7d5", "#f0dcd6", "#e2dcec", "#d9e4ee")
STAGES: Final = (
    (
        "Data protocol",
        (
            "Official splits:",
            "UNSW-NB15,",
            "NSL-KDD",
            "Cross-day:",
            "CIC-IDS-2017,",
            "CIC-DDoS2019",
        ),
    ),
    (
        "Training-only",
        (
            "Imputation",
            "Integer encoding",
            "SMOTE(-NC)",
            "Tomek cleaning",
            "One-hot expansion",
        ),
    ),
    (
        "Weight ablation",
        ("No weights", "Original weights", "Post-cleaning", "residual weights"),
    ),
    (
        "Model fitting",
        (
            "XGBoost",
            "Five strong",
            "baselines",
            "MLP extension",
            "(three datasets)",
        ),
    ),
    (
        "Evaluation",
        ("Macro PR-AUC", "1% FPR budget", "Paired bootstrap", "TreeSHAP stability"),
    ),
)
CAPTION: Final = (
    "Validation selects stopping and the 1% FPR threshold.\n"
    "Locked test records produce every reported result."
)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def draw(output: Path, stem: str) -> list[dict]:
    """Render and save the workflow diagram."""
    plt.rcParams.update(RC_PARAMS)
    figure, ax = plt.subplots(figsize=(WIDTH_INCHES, HEIGHT_INCHES))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    count = len(STAGES)
    gap = 1.8
    box_width = (100 - gap * (count - 1)) / count
    box_bottom, box_height = 30.0, 62.0

    for index, (title, lines) in enumerate(STAGES):
        left = index * (box_width + gap)
        ax.add_patch(
            FancyBboxPatch(
                (left, box_bottom),
                box_width,
                box_height,
                boxstyle="round,pad=0,rounding_size=2.2",
                linewidth=0.9,
                edgecolor="#3a3a3a",
                facecolor=FILLS[index % len(FILLS)],
            )
        )
        centre = left + box_width / 2
        ax.text(
            centre,
            box_bottom + box_height - 7.0,
            title,
            ha="center",
            va="center",
            fontsize=8.2,
        )
        for offset, line in enumerate(lines):
            ax.text(
                centre,
                box_bottom + box_height - 18.0 - offset * 7.2,
                line,
                ha="center",
                va="center",
                fontsize=7.0,
            )
        if index < count - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (left + box_width + 0.15, box_bottom + box_height / 2),
                    (left + box_width + gap - 0.15, box_bottom + box_height / 2),
                    arrowstyle="-|>",
                    mutation_scale=8,
                    linewidth=0.9,
                    color="#3a3a3a",
                )
            )

    ax.add_patch(
        FancyBboxPatch(
            (6.0, 6.0),
            88.0,
            20.0,
            boxstyle="round,pad=0,rounding_size=2.0",
            linewidth=0.9,
            linestyle=(0, (4, 3)),
            edgecolor="#3a3a3a",
            facecolor="#f4f4f4",
        )
    )
    ax.text(50.0, 16.0, CAPTION, ha="center", va="center", fontsize=8.2)

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
    """Entry point for regenerating the workflow figure."""
    parser = argparse.ArgumentParser(description="Redraw the workflow figure")
    parser.add_argument("--out", required=True)
    parser.add_argument("--stem", default="fig01_workflow")
    args = parser.parse_args()
    for record in draw(Path(args.out), args.stem):
        print(record["file"], record["bytes"], record["sha256"][:16])


if __name__ == "__main__":
    main()
