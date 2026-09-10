"""Redraw the published figures and compare them with the committed copies.

Kept separate from `verify.py` because this is the only check that runs code
rather than reading data: it re-executes the figure scripts in a scratch
directory and compares the result byte for byte.

Only PNG is compared. Matplotlib writes a creation timestamp into PDF and SVG,
so those formats differ on every run even when the drawing is identical, and
comparing them would produce a failure that means nothing.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from rwxai.evidence import CheckResult, sha256_of

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
    """Regenerate all six figures and compare their PNG bytes."""
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
            if sha256_of(figures / f"{stem}.png") != sha256_of(regenerated):
                problems.append(f"{stem}.png differs from the published figure")

    detail = "; ".join(problems) if problems else f"{compared} figures match exactly"
    return CheckResult(
        name="figures:regenerated",
        passed=not problems and compared == len(FIGURE_STEMS),
        detail=detail,
        counts={"compared": compared, "problems": len(problems)},
    )
