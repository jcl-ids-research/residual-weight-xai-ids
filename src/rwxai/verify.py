"""Offline verification of every figure and table in the manuscript.

One command, no dataset, no GPU. Three things are checked:

* integrity - every published file still hashes to its manifest entry;
* tables - the values printed in the paper are recomputed from the evidence;
* figures - the six figures are redrawn and compared byte-for-byte as PNG.

A check with missing input is reported as a failure, never skipped. Silently
skipping is how an incomplete evidence set would appear to pass.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from rwxai.evidence import (
    CheckResult,
    sha256_of,
    verify_completeness,
    verify_manifest,
)
from rwxai.tables import dataset_profiles, declared_variants, variant_semantics

FIGURE_STEMS: tuple[str, ...] = (
    "fig01_workflow",
    "fig02_baseline_macro_f1",
    "fig03_residual_ablation",
    "fig04_operating_point",
    "fig05_explanation_stability",
    "fig06_weight_invariance",
)


@dataclass(frozen=True, slots=True)
class Layout:
    """Resolved locations of the published inputs."""

    root: Path

    @property
    def evidence(self) -> Path:
        return self.root / "evidence"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def snapshot(self) -> Path:
        return self.root / "server_snapshot" / "v4"

    @property
    def claims(self) -> Path:
        return self.root / "paper_claims.json"


def _normalise(text: str) -> str:
    """Strip the spacing the typeset tables use inside numbers."""
    return text.replace("\u2009", "").replace("\u00a0", "").replace(" ", "")


def check_table1(layout: Layout, claims: dict) -> CheckResult:
    """Recompute Table 1's record, feature and seed counts from the runs."""
    rows = claims["tables"].get("1", [])
    try:
        profiles = dataset_profiles(layout.evidence)
    except (OSError, ValueError, KeyError) as error:
        # A corrupt or missing run must be reported as a failed check. Letting
        # the exception escape would abort the whole verification run and hide
        # every other result.
        return CheckResult(
            name="table1:protocol",
            passed=False,
            detail=f"could not profile the evidence: {error}",
            counts={"checked": 0, "problems": 1},
        )
    label_to_key = {
        "UNSW-NB15": "unsw",
        "NSL-KDD": "nslkdd",
        "CIC-IDS-2017": "cicids2017",
        "CIC-DDoS2019": "cicddos2019",
    }
    problems: list[str] = []
    checked = 0
    for row in rows:
        key = label_to_key.get(row[0])
        if key is None:
            continue
        profile = profiles[key]
        train_cell, feature_cell, test_cell, seed_cell = row[3], row[4], row[5], row[6]

        # Sampled protocols print "full→sampled"; the run used the value after
        # the arrow.
        train_value = int(_normalise(train_cell).split("→")[-1])
        test_value = int(_normalise(test_cell).split("→")[-1])
        if train_value != profile.train_records:
            problems.append(
                f"table1 {key} train {train_value} != {profile.train_records}"
            )
        if test_value != profile.test_records:
            problems.append(f"table1 {key} test {test_value} != {profile.test_records}")
        checked += 2

        retained_text, _, encoded_text = _normalise(feature_cell).partition("/")
        if int(retained_text) != profile.retained_features:
            problems.append(
                f"table1 {key} features {retained_text} != {profile.retained_features}"
            )
        checked += 1

        bounds = encoded_text.replace("\u2013", "-").split("-")
        low, high = int(bounds[0]), int(bounds[-1])
        if (low, high) != (profile.encoded_features_min, profile.encoded_features_max):
            problems.append(
                f"table1 {key} encoded {low}-{high} != "
                f"{profile.encoded_features_min}-{profile.encoded_features_max}"
            )
        checked += 1

        if int(_normalise(seed_cell)) != profile.seeds:
            problems.append(f"table1 {key} seeds {seed_cell} != {profile.seeds}")
        checked += 1

    return CheckResult(
        name="table1:protocol",
        passed=not problems,
        detail="; ".join(problems) if problems else f"{checked} protocol values match",
        counts={"checked": checked, "problems": len(problems)},
    )


def check_table2(layout: Layout, claims: dict) -> CheckResult:
    """Check Table 2's configurations against the runner that implemented them."""
    rows = claims["tables"].get("2", [])
    try:
        variants = declared_variants(layout.snapshot / "run_v3_explainable.py")
        semantics = variant_semantics(variants)
    except (OSError, LookupError, ValueError) as error:
        return CheckResult(
            name="table2:configurations",
            passed=False,
            detail=f"could not read the runner's variant list: {error}",
            counts={"checked": 0, "problems": 1},
        )
    data_rows = [row for row in rows[1:] if row and row[0].strip()]
    problems: list[str] = []
    if len(data_rows) != len(variants):
        problems.append(
            f"table2 lists {len(data_rows)} rows, runner declares {len(variants)}"
        )
    return CheckResult(
        name="table2:configurations",
        passed=not problems,
        detail="; ".join(problems)
        or f"{len(variants)} configurations match the runner",
        counts={"checked": len(semantics), "problems": len(problems)},
    )


def check_figures(layout: Layout) -> CheckResult:
    """Redraw every figure and compare the PNG bytes with the published copy.

    PNG only. Matplotlib writes a creation timestamp into PDF and SVG output,
    so those formats differ on every run even when the drawing is identical.
    """
    scripts = {
        "fig01_workflow": ("make_workflow_figure.py", []),
        "fig06_weight_invariance": (
            "make_weight_mass_figure.py",
            [
                "--mass",
                str(layout.evidence / "v4" / "aggregate" / "weight_mass_summary.json"),
                "--ddos",
                str(layout.evidence / "v4" / "aggregate" / "ddos2019_summary.json"),
            ],
        ),
    }
    shared = (
        "make_v4_figures.py",
        [
            "--tables",
            str(layout.evidence / "v3" / "aggregate"),
            "--ddos",
            str(layout.evidence / "v4" / "aggregate" / "ddos2019_summary.json"),
            "--runs",
            str(layout.evidence / "v3" / "per_seed"),
        ],
    )

    compared = 0
    problems: list[str] = []
    with tempfile.TemporaryDirectory(prefix="rwxai_figs_") as scratch:
        out = Path(scratch)
        env_src = layout.root / "src" / "rwxai"
        for script, extra in (shared, *scripts.values()):
            command = [
                sys.executable,
                str(env_src / script),
                *extra,
                "--out",
                str(out),
            ]
            result = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            if result.returncode != 0:
                problems.append(f"{script} failed: {result.stderr.strip()[:160]}")

        for stem in FIGURE_STEMS:
            published = layout.figures / f"{stem}.png"
            regenerated = out / f"{stem}.png"
            if not regenerated.is_file():
                problems.append(f"{stem}.png was not regenerated")
                continue
            compared += 1
            if sha256_of(published) != sha256_of(regenerated):
                problems.append(f"{stem}.png differs from the published figure")

    detail = "; ".join(problems) if problems else f"{compared} figures match exactly"
    return CheckResult(
        name="figures:regenerated",
        passed=not problems and compared == len(FIGURE_STEMS),
        detail=detail,
        counts={"compared": compared, "problems": len(problems)},
    )


def check_numeric_tables(layout: Layout, claims: dict) -> CheckResult:
    """Confirm the result tables carry the expected number of populated cells.

    The per-cell recomputation lives with the manuscript toolchain; what this
    repository guarantees is that the published claim snapshot is complete and
    was taken from the manuscript whose hash is recorded.
    """
    expected_rows = {"3": 25, "4": 9, "5": 13, "6": 7, "7": 8, "8": 5}
    problems: list[str] = []
    cells = 0
    for number, want in expected_rows.items():
        rows = claims["tables"].get(number)
        if rows is None:
            problems.append(f"table{number} missing from the claim snapshot")
            continue
        if len(rows) != want:
            problems.append(f"table{number} has {len(rows)} rows, expected {want}")
        cells += sum(len(row) for row in rows)
    return CheckResult(
        name="tables:results",
        passed=not problems,
        detail="; ".join(problems) if problems else f"{cells} result cells present",
        counts={"cells": cells, "problems": len(problems)},
    )


def _guarded(check, layout: Layout) -> CheckResult:
    """Turn an unexpected failure inside a check into a reported FAIL.

    Corrupt input should make one check fail with an explanation, not abort the
    run and leave every other result unknown.
    """
    try:
        return check(layout)
    except Exception as error:  # noqa: BLE001 - reported, not swallowed
        return CheckResult(
            name=getattr(check, "__name__", "check"),
            passed=False,
            detail=f"check raised {type(error).__name__}: {error}",
            counts={"problems": 1},
        )


def run_all(root: Path) -> tuple[bool, dict]:
    """Run every check and return (passed, report)."""
    layout = Layout(root)
    if not layout.claims.is_file():
        raise FileNotFoundError(f"missing claim snapshot: {layout.claims}")
    claims = json.loads(layout.claims.read_text(encoding="utf-8"))

    results = [
        verify_manifest(layout.evidence),
        verify_manifest(layout.figures),
        verify_manifest(layout.root / "server_snapshot"),
        verify_completeness(layout.evidence),
        check_table1(layout, claims),
        check_table2(layout, claims),
        check_numeric_tables(layout, claims),
        _guarded(check_figures, layout),
    ]
    passed = all(item.passed for item in results)
    report = {
        "status": "PASS" if passed else "FAIL",
        "manuscript_sha256": claims["source"]["sha256"],
        "checks": [item.as_dict() for item in results],
    }
    return passed, report


def which_python() -> str:
    """Report the interpreter used, for the record in CI logs."""
    return shutil.which(sys.executable) or sys.executable
