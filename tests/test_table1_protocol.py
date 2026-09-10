"""Table 1 checks: dataset sizes, feature counts and seed counts.

Table 1 states the record counts, retained/encoded feature counts and seed
counts for each dataset. Until now nothing recomputed those numbers from the
run evidence, so a typo in the manuscript would have gone unnoticed. Every
value here is read back out of the per-seed metrics rather than restated.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rwxai.evidence import DDOS_SEEDS, EXPECTED_SEEDS, load_metrics
from rwxai.tables import (
    collect_dataset_profile,
    dataset_profiles,
    seed_counts,
)

EVIDENCE = Path(__file__).resolve().parents[1] / "evidence"


def test_every_seed_agrees_on_dataset_sizes() -> None:
    """Given ten seeds, When sizes are read, Then all seeds report one shape.

    Seeds change the internal validation split, not the official partition. If
    two seeds disagreed on the official record counts, the partition itself
    moved and Table 1 could not describe the runs.
    """
    for dataset, seeds in EXPECTED_SEEDS.items():
        shapes = {
            (
                load_metrics(EVIDENCE, "v3", dataset, seed)["sizes"]["official_train"],
                load_metrics(EVIDENCE, "v3", dataset, seed)["sizes"]["official_test"],
            )
            for seed in seeds
        }
        assert len(shapes) == 1, f"{dataset} seeds disagree: {shapes}"


def test_dataset_profile_reports_records_and_features() -> None:
    """Given the evidence tree, When profiled, Then Table 1 fields are present."""
    profiles = dataset_profiles(EVIDENCE)
    assert set(profiles) == {"unsw", "nslkdd", "cicids2017", "cicddos2019"}
    for name, profile in profiles.items():
        assert profile.train_records > 0, name
        assert profile.test_records > 0, name
        assert profile.retained_features > 0, name
        assert profile.encoded_features_max >= profile.retained_features, name
        assert profile.encoded_features_min <= profile.encoded_features_max, name


def test_seed_counts_match_the_reported_protocol() -> None:
    """Given the runs, When seeds are counted, Then 10/10/3/3 is reproduced."""
    counts = seed_counts(EVIDENCE)
    assert counts["unsw"] == len(EXPECTED_SEEDS["unsw"]) == 10
    assert counts["nslkdd"] == len(EXPECTED_SEEDS["nslkdd"]) == 10
    assert counts["cicids2017"] == len(EXPECTED_SEEDS["cicids2017"]) == 3
    assert counts["cicddos2019"] == len(DDOS_SEEDS) == 3


def test_one_hot_width_may_vary_across_seeds() -> None:
    """Given UNSW seeds, When widths are read, Then variation is tolerated.

    The encoded width depends on which categorical values appear in the
    sub-training split, so it is not constant across seeds. This pins that as
    expected behaviour rather than a defect, and keeps the profile honest about
    reporting a range.
    """
    profile = dataset_profiles(EVIDENCE)["unsw"]
    assert profile.encoded_features_min <= profile.encoded_features_max
    assert profile.retained_features == 42
    assert profile.train_records == 175341
    assert profile.test_records == 82332


def test_corrupted_size_is_detected(tmp_path: Path) -> None:
    """Given a tampered size field, When profiled, Then the check fails.

    This is the negative control. A checker that cannot fail on corrupt input
    proves nothing when it passes.
    """
    dataset = "unsw"
    for seed in EXPECTED_SEEDS[dataset]:
        source = EVIDENCE / "v3" / "per_seed" / dataset / f"seed{seed}"
        target = tmp_path / "v3" / "per_seed" / dataset / f"seed{seed}"
        target.mkdir(parents=True)
        payload = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
        if seed == EXPECTED_SEEDS[dataset][0]:
            payload["sizes"]["official_train"] += 1
        (target / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="disagree"):
        collect_dataset_profile(tmp_path, "v3", dataset, EXPECTED_SEEDS[dataset])
