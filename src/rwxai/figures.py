"""Redraw the published figures and validate what they contain.

Kept separate from `verify.py` because this is the only check that runs code
rather than reading data: it re-executes the figure scripts in a scratch
directory.

Two different guarantees are involved, and conflating them was a mistake worth
naming. The *published* PNG/PDF/SVG files are pinned by `figures/MANIFEST.sha256`,
so editing one is detected immediately. The *regenerated* PNGs are checked for
successful rendering, non-blank content and dimensions matching the published
copies - but not by pixel hash, because font rasterisation differs across
operating systems even when the plotted data and layout are identical. A hash
comparison therefore passes on the machine that drew the figures and fails
everywhere else, which is a property of the font stack rather than of the
evidence.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageStat

from rwxai.evidence import CheckResult

FIGURE_STEMS: tuple[str, ...] = (
    "fig01_workflow",
    "fig02_baseline_macro_f1",
    "fig03_residual_ablation",
    "fig04_operating_point",
    "fig05_explanation_stability",
    "fig06_weight_invariance",
)


def _commands(evidence: Path, source: Path, out: Path) -> list[list[str]]:
    """Build the figure-script invocations, all writing into one directory."""
    aggregate_v3 = evidence / "v3" / "aggregate"
    aggregate_v4 = evidence / "v4" / "aggregate"
    ddos_summary = aggregate_v4 / "ddos2019_summary.json"
    return [
        [
            sys.executable,
            str(source / "make_v4_figures.py"),
            "--tables",
            str(aggregate_v3),
            "--ddos",
            str(ddos_summary),
            "--runs",
            str(evidence / "v3" / "per_seed"),
            "--out",
            str(out),
        ],
        [
            sys.executable,
            str(source / "make_workflow_figure.py"),
            "--out",
            str(out),
        ],
        [
            sys.executable,
            str(source / "make_weight_mass_figure.py"),
            "--mass",
            str(aggregate_v4 / "weight_mass_summary.json"),
            "--ddos",
            str(ddos_summary),
            "--out",
            str(out),
        ],
    ]


def check_figures(evidence: Path, figures: Path, source: Path) -> CheckResult:
    """Regenerate all six figures and reject missing, blank or shifted output."""
    compared = 0
    problems: list[str] = []

    with tempfile.TemporaryDirectory(prefix="rwxai_figs_") as scratch:
        out = Path(scratch)
        for command in _commands(evidence, source, out):
            result = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            if result.returncode != 0:
                script = Path(command[1]).name
                problems.append(f"{script} failed: {result.stderr.strip()[:160]}")

        for stem in FIGURE_STEMS:
            regenerated = out / f"{stem}.png"
            if not regenerated.is_file():
                problems.append(f"{stem}.png was not regenerated")
                continue
            compared += 1
            published = figures / f"{stem}.png"
            with (
                Image.open(published) as expected_image,
                Image.open(regenerated) as actual_image,
            ):
                expected_size = expected_image.size
                actual_size = actual_image.size
                width_shift = abs(actual_size[0] - expected_size[0]) / expected_size[0]
                height_shift = abs(actual_size[1] - expected_size[1]) / expected_size[1]
                if width_shift > 0.02 or height_shift > 0.02:
                    problems.append(
                        f"{stem}.png dimensions shifted from "
                        f"{expected_size} to {actual_size}"
                    )
                grayscale = actual_image.convert("L")
                extrema = grayscale.getextrema()
                deviation = ImageStat.Stat(grayscale).stddev[0]
                if extrema is None or extrema[0] >= 250 or deviation < 1.0:
                    problems.append(f"{stem}.png is blank or nearly uniform")

    detail = (
        "; ".join(problems)
        if problems
        else f"{compared} figures regenerated with valid cross-platform structure"
    )
    return CheckResult(
        name="figures:regenerated",
        passed=not problems and compared == len(FIGURE_STEMS),
        detail=detail,
        counts={"compared": compared, "problems": len(problems)},
    )
