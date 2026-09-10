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

from rwxai.evidence import sha256_of, verify_completeness, verify_manifest
from rwxai.verify import (
    Layout,
    check_manuscript_hash,
    check_numeric_tables,
    check_table1,
    check_table2,
    run_all,
)

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

    result = verify_manifest(repo_copy / "evidence")

    assert not result.passed
    assert result.name == "manifest:evidence"


def test_windows_manifest_paths_are_portable(tmp_path: Path) -> None:
    """Given backslash paths, When checked on any OS, Then files resolve."""
    nested = tmp_path / "nested"
    nested.mkdir()
    evidence = nested / "metrics.json"
    evidence.write_text("{}", encoding="utf-8")
    (tmp_path / "MANIFEST.sha256").write_text(
        f"{sha256_of(evidence)}  nested\\metrics.json\n", encoding="utf-8"
    )

    result = verify_manifest(tmp_path)

    assert result.passed, result.detail


def test_missing_seed_is_detected(repo_copy: Path) -> None:
    """Given a deleted run, When verified, Then completeness names the seed."""
    shutil.rmtree(repo_copy / "evidence" / "v3" / "per_seed" / "nslkdd" / "seed789")

    result = verify_completeness(repo_copy / "evidence")

    assert not result.passed
    assert "789" in result.detail


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


def test_tampered_table2_label_is_detected(repo_copy: Path) -> None:
    """Given a renamed configuration, When verified, Then Table 2 fails."""
    claims_path = repo_copy / "paper_claims.json"
    claims = json.loads(claims_path.read_text(encoding="utf-8"))
    claims["tables"]["2"][1][0] = "CORRUPTED_VARIANT"
    result = check_table2(Layout(repo_copy), claims)

    assert not result.passed
    assert result.name == "table2:configurations"


def test_tampered_result_cell_is_detected(repo_copy: Path) -> None:
    """Given a changed Table 3 value, When verified, Then results fail."""
    claims_path = repo_copy / "paper_claims.json"
    claims = json.loads(claims_path.read_text(encoding="utf-8"))
    claims["tables"]["3"][1][2] = "0.9999±0.0000"
    result = check_numeric_tables(Layout(repo_copy), claims)

    assert not result.passed
    assert result.name == "tables:results"


def test_manuscript_hash_rejects_wrong_file(tmp_path: Path) -> None:
    """Given a different manuscript, When bound, Then its hash fails."""
    manuscript = tmp_path / "manuscript.docx"
    manuscript.write_bytes(b"different manuscript")
    claims = {"source": {"sha256": "0" * 64}}

    result = check_manuscript_hash(claims, manuscript)

    assert not result.passed
    assert "does not match" in result.detail


def test_tampered_figure_is_detected(repo_copy: Path) -> None:
    """Given an altered figure, When verified, Then the figure check fails."""
    victim = repo_copy / "figures" / "fig02_baseline_macro_f1.png"
    data = bytearray(victim.read_bytes())
    data[-1] ^= 0xFF
    victim.write_bytes(bytes(data))

    result = verify_manifest(repo_copy / "figures")

    assert not result.passed
    assert result.name == "manifest:figures"


def test_tampered_snapshot_is_detected(repo_copy: Path) -> None:
    """Given an edited server snapshot, When verified, Then provenance fails.

    The snapshot exists to show the published numbers came from that exact
    code. Editing it must be visible.
    """
    victim = repo_copy / "server_snapshot" / "v4" / "v3_data.py"
    victim.write_text(
        victim.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8"
    )

    result = verify_manifest(repo_copy / "server_snapshot")

    assert not result.passed
    assert result.name == "manifest:server_snapshot"
