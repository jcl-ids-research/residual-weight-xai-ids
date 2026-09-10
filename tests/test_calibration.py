"""Negative controls for the verification pipeline.

A verifier that has never failed is an unproven verifier. Each test corrupts
one published input in a throwaway copy and requires the corresponding check to
fail and to name what broke. Without these, a green run would only prove that
the checks executed, not that they can detect anything.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from rwxai.verify import Layout, check_table1, run_all

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def repo_copy(tmp_path: Path) -> Path:
    """A copy of the published inputs that a test may safely corrupt."""
    target = tmp_path / "repo"
    target.mkdir()
    for name in ("evidence", "figures", "server_snapshot", "src"):
        shutil.copytree(REPO / name, target / name)
    shutil.copy2(REPO / "paper_claims.json", target / "paper_claims.json")
    return target


def test_untouched_copy_passes(repo_copy: Path) -> None:
    """Given an untouched copy, When verified, Then it passes.

    The positive control. If this failed, later failures would prove nothing
    about tampering - only that the copy was broken.
    """
    passed, report = run_all(repo_copy)
    assert passed, report


def test_tampered_evidence_file_is_detected(repo_copy: Path) -> None:
    """Given a modified metrics file, When verified, Then integrity fails."""
    seed_dir = repo_copy / "evidence" / "v3" / "per_seed" / "unsw" / "seed42"
    victim = seed_dir / "metrics.json"
    payload = json.loads(victim.read_text(encoding="utf-8"))
    payload["sizes"]["official_test"] += 1
    victim.write_text(json.dumps(payload), encoding="utf-8")

    passed, report = run_all(repo_copy)
    assert not passed
    failed = {c["name"] for c in report["checks"] if c["status"] == "FAIL"}
    assert "manifest:evidence" in failed, report


def test_missing_seed_is_detected(repo_copy: Path) -> None:
    """Given a deleted run, When verified, Then completeness names the seed."""
    shutil.rmtree(repo_copy / "evidence" / "v3" / "per_seed" / "nslkdd" / "seed789")

    passed, report = run_all(repo_copy)
    assert not passed
    completeness = next(
        c for c in report["checks"] if c["name"] == "evidence:completeness"
    )
    assert completeness["status"] == "FAIL"
    assert "789" in completeness["detail"], completeness


def test_tampered_table1_value_is_detected(repo_copy: Path) -> None:
    """Given a wrong record count in the paper, When checked, Then it fails."""
    claims_path = repo_copy / "paper_claims.json"
    claims = json.loads(claims_path.read_text(encoding="utf-8"))
    for row in claims["tables"]["1"]:
        if row and row[0] == "UNSW-NB15":
            row[3] = "175 342"
    result = check_table1(Layout(repo_copy), claims)
    assert not result.passed
    assert "train" in result.detail, result.detail


def test_tampered_figure_is_detected(repo_copy: Path) -> None:
    """Given an altered figure, When verified, Then the figure check fails."""
    victim = repo_copy / "figures" / "fig02_baseline_macro_f1.png"
    data = bytearray(victim.read_bytes())
    data[-1] ^= 0xFF
    victim.write_bytes(bytes(data))

    passed, report = run_all(repo_copy)
    assert not passed
    failed = {c["name"] for c in report["checks"] if c["status"] == "FAIL"}
    assert "manifest:figures" in failed or "figures:regenerated" in failed, report


def test_tampered_snapshot_is_detected(repo_copy: Path) -> None:
    """Given an edited server snapshot, When verified, Then provenance fails.

    The snapshot exists to show the published numbers came from that exact
    code. Editing it must be visible.
    """
    victim = repo_copy / "server_snapshot" / "v4" / "v3_data.py"
    victim.write_text(
        victim.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8"
    )

    passed, report = run_all(repo_copy)
    assert not passed
    failed = {c["name"] for c in report["checks"] if c["status"] == "FAIL"}
    assert "manifest:server_snapshot" in failed, report
