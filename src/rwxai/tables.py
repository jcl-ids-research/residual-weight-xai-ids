"""Recompute the values Table 1 and Table 2 assert.

Tables 3-8 are numeric results and are checked cell by cell against the
per-seed evidence elsewhere. Tables 1 and 2 describe the protocol instead, and
were previously unverified: nothing confirmed that the record counts, feature
counts and seed counts in Table 1 matched the runs, or that the seven
configurations in Table 2 were the ones the runner implements.

Table 1 is recomputed from per-seed metrics. Table 2 is checked against the
variant list parsed out of the immutable server snapshot, because a design
table has no numbers to recompute - only code to agree with.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from rwxai.evidence import DDOS_SEEDS, EXPECTED_SEEDS, load_metrics

RESIDUAL_VARIANT = "smotenc_tomek_residual_weight"

# Resampling and weighting semantics for each configuration, as implemented by
# the runner: (resampling, weight source).
VARIANT_SEMANTICS: dict[str, tuple[str, str]] = {
    "raw": ("none", "none"),
    "raw_weight": ("none", "original"),
    "random_over": ("random_oversample", "none"),
    "smotenc": ("smotenc", "none"),
    "smotenc_tomek": ("smotenc_tomek", "none"),
    "smotenc_tomek_weight": ("smotenc_tomek", "original"),
    RESIDUAL_VARIANT: ("smotenc_tomek", "residual"),
}


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    """The Table 1 row for one dataset."""

    dataset: str
    train_records: int
    test_records: int
    retained_features: int
    encoded_features_min: int
    encoded_features_max: int
    seeds: int

    @property
    def encoded_features(self) -> int:
        """The upper end of the encoded width, as reported in the manuscript."""
        return self.encoded_features_max


def collect_dataset_profile(
    evidence_root: Path,
    layer: str,
    dataset: str,
    seeds: tuple[int, ...],
) -> DatasetProfile:
    """Read one dataset's Table 1 row, requiring all seeds to agree.

    Seeds vary the internal validation split only, so the official record
    counts and the retained field count must be identical across seeds; a
    disagreement means the partition moved and no single Table 1 row can
    describe the runs.

    The one-hot width is deliberately not held constant. It depends on which
    categorical values survive the sub-training split, so it varies slightly
    between seeds and is reported as an observed range instead.
    """
    partitions: set[tuple[int, int, int]] = set()
    widths: set[int] = set()
    for seed in seeds:
        sizes = load_metrics(evidence_root, layer, dataset, seed)["sizes"]
        partitions.add(
            (
                int(sizes["official_train"]),
                int(sizes["official_test"]),
                int(sizes["original_features"]),
            )
        )
        widths.add(int(sizes["classifier_features_after_onehot"]))
    if len(partitions) != 1:
        raise ValueError(
            f"{dataset}: seeds disagree on protocol sizes: {sorted(partitions)}"
        )
    train, test, retained = partitions.pop()
    return DatasetProfile(
        dataset=dataset,
        train_records=train,
        test_records=test,
        retained_features=retained,
        encoded_features_min=min(widths),
        encoded_features_max=max(widths),
        seeds=len(seeds),
    )


def dataset_profiles(evidence_root: Path) -> dict[str, DatasetProfile]:
    """Build the Table 1 rows for all four datasets."""
    profiles = {
        dataset: collect_dataset_profile(evidence_root, "v3", dataset, seeds)
        for dataset, seeds in EXPECTED_SEEDS.items()
    }
    profiles["cicddos2019"] = collect_dataset_profile(
        evidence_root, "ddos", "cicddos2019", DDOS_SEEDS
    )
    return profiles


def seed_counts(evidence_root: Path) -> dict[str, int]:
    """Return the number of paired seeds behind each dataset."""
    profiles = dataset_profiles(evidence_root)
    return {name: profile.seeds for name, profile in profiles.items()}


def declared_variants(runner_source: Path) -> tuple[str, ...]:
    """Parse `VARIANT_ORDER` out of the runner without importing it.

    Importing the runner would pull in the full training stack; the variant
    list is a literal tuple, so reading it from the syntax tree is both cheaper
    and safe against side effects.
    """
    tree = ast.parse(runner_source.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "VARIANT_ORDER":
                return tuple(ast.literal_eval(node.value))
    raise LookupError(f"VARIANT_ORDER not found in {runner_source}")


def variant_semantics(variants: tuple[str, ...]) -> dict[str, tuple[str, str]]:
    """Map each declared variant to its resampling and weighting semantics."""
    unknown = [name for name in variants if name not in VARIANT_SEMANTICS]
    if unknown:
        raise LookupError(
            f"runner declares variants Table 2 does not describe: {unknown}"
        )
    return {name: VARIANT_SEMANTICS[name] for name in variants}
