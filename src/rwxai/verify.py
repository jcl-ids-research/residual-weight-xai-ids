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
from dataclasses import dataclass
from pathlib import Path

from rwxai.aggregates import check_public_aggregates
from rwxai.claim_validation import check_result_claims, check_table2_claims
from rwxai.evidence import (
    CheckResult,
    sha256_of,
    verify_completeness,
    verify_manifest,
)
from rwxai.figures import check_figures as regenerate_and_compare
from rwxai.tables import dataset_profiles, declared_variants, variant_semantics


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
    """Check every Table 2 field against the implemented configurations."""
    try:
        variants = declared_variants(layout.snapshot / "run_v3_explainable.py")
        variant_semantics(variants)
    except (OSError, LookupError, ValueError) as error:
        return CheckResult(
            name="table2:configurations",
            passed=False,
            detail=f"could not read the runner's variant list: {error}",
            counts={"checked": 0, "problems": 1},
        )
    return check_table2_claims(claims)


def check_figures(layout: Layout) -> CheckResult:
    """Redraw every figure and compare it with the published copy."""
    return regenerate_and_compare(
        evidence=layout.evidence,
        figures=layout.figures,
        source=layout.root / "src" / "rwxai",
    )


def check_numeric_tables(layout: Layout, claims: dict) -> CheckResult:
    """Compare every result-table cell with committed evidence."""
    try:
        return check_result_claims(layout.root, claims)
    except (OSError, KeyError, ValueError) as error:
        return CheckResult(
            name="tables:results",
            passed=False,
            detail=f"could not recompute result tables: {error}",
            counts={"checked": 0, "problems": 1},
        )


def check_manuscript_hash(claims: dict, manuscript: Path) -> CheckResult:
    """Bind a reviewer-provided manuscript to the published table snapshot."""
    expected = str(claims["source"]["sha256"])
    actual = sha256_of(manuscript) if manuscript.is_file() else "missing"
    passed = actual == expected
    return CheckResult(
        name="manuscript:sha256",
        passed=passed,
        detail=(
            "manuscript matches the published claim snapshot"
            if passed
            else f"manuscript does not match: expected {expected}, got {actual}"
        ),
        counts={"checked": 1, "problems": 0 if passed else 1},
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


def run_all(root: Path, manuscript: Path | None = None) -> tuple[bool, dict]:
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
        check_public_aggregates(layout.root),
        check_numeric_tables(layout, claims),
        _guarded(check_figures, layout),
    ]
    if manuscript is not None:
        results.append(check_manuscript_hash(claims, manuscript))
    passed = all(item.passed for item in results)
    report = {
        "status": "PASS" if passed else "FAIL",
        "claim_snapshot_sha256": claims["source"]["sha256"],
        "manuscript": str(manuscript) if manuscript is not None else None,
        "checks": [item.as_dict() for item in results],
    }
    return passed, report
