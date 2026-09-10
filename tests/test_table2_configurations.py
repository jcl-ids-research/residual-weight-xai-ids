"""Table 2 checks: the seven class-imbalance configurations.

Table 2 is a design table, so it cannot be recomputed from metrics. What it can
be checked against is the code that actually ran: the variant list in the
experiment runner. If the manuscript ever described a configuration the runner
does not implement, that mismatch is a factual error in the paper.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rwxai.tables import (
    RESIDUAL_VARIANT,
    declared_variants,
    variant_semantics,
)

SNAPSHOT = Path(__file__).resolve().parents[1] / "server_snapshot" / "v4"


def test_runner_declares_exactly_seven_variants() -> None:
    """Given the runner source, When parsed, Then seven variants are declared."""
    variants = declared_variants(SNAPSHOT / "run_v3_explainable.py")
    assert len(variants) == 7, variants


def test_variant_names_match_table_two() -> None:
    """Given Table 2's rows, When compared, Then names match the runner."""
    variants = declared_variants(SNAPSHOT / "run_v3_explainable.py")
    assert variants[0] == "raw", variants
    assert RESIDUAL_VARIANT in variants
    assert variants[-1] == RESIDUAL_VARIANT, "the paper's method is the final row"


def test_each_variant_has_resampling_and_weight_semantics() -> None:
    """Given each variant, When described, Then resampling and weight are set."""
    variants = declared_variants(SNAPSHOT / "run_v3_explainable.py")
    semantics = variant_semantics(variants)
    assert set(semantics) == set(variants)
    assert semantics["raw"] == ("none", "none")
    assert semantics["raw_weight"] == ("none", "original")
    assert semantics[RESIDUAL_VARIANT] == ("smotenc_tomek", "residual")


def test_missing_runner_is_reported(tmp_path: Path) -> None:
    """Given no runner source, When parsed, Then the failure is explicit."""
    with pytest.raises(FileNotFoundError):
        declared_variants(tmp_path / "absent.py")
