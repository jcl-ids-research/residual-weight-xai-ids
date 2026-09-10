"""Validation for per-class and explanation manuscript tables."""

from __future__ import annotations

from typing import Final

from rwxai.claim_sources import PROPOSED, ClaimSources, Summary, mean_sd_cell

DATASETS: Final = {
    "UNSW-NB15": "unsw",
    "NSL-KDD": "nslkdd",
    "CIC-IDS-2017跨日": "cicids2017",
    "CIC-DDoS2019跨日": "cicddos2019",
}


def _record(problems: list[str], context: str, got: str, expected: str) -> int:
    if got != expected:
        problems.append(f"{context}: claim={got!r}, evidence={expected!r}")
    return 1


def check_table7(
    source: ClaimSources, rows: list[list[str]], problems: list[str]
) -> int:
    """Validate support and per-class metrics."""
    columns = (
        ("raw", "precision"),
        (PROPOSED, "precision"),
        ("raw", "recall"),
        (PROPOSED, "recall"),
        (PROPOSED, "f1"),
    )
    checked = 0
    for row in rows[1:]:
        dataset, class_name = DATASETS.get(row[0]), row[1]
        if dataset is None:
            problems.append(f"Table 7 unknown dataset: {row[0]}")
            continue
        checked += _record(
            problems,
            f"Table 7 {row[0]}/{class_name}/support",
            row[2],
            f"{source.support(dataset, class_name):,}".replace(",", " "),
        )
        for column, (variant, metric) in enumerate(columns, start=3):
            if dataset == "cicddos2019":
                prefix = "raw" if variant == "raw" else "residual"
                entry = source.ddos["per_class"][class_name][f"{prefix}_{metric}"]
                summary = Summary(float(entry["mean"]), float(entry["sample_sd"]))
            else:
                summary = source.per_class[(dataset, class_name, variant, metric)]
            checked += _record(
                problems,
                f"Table 7 {row[0]}/{class_name}/{variant}/{metric}",
                row[column],
                mean_sd_cell(summary, 2, 100.0),
            )
    return checked


def check_table8(
    source: ClaimSources, rows: list[list[str]], problems: list[str]
) -> int:
    """Validate explanation stability and additivity values."""
    checked = 0
    for row in rows[1:]:
        dataset = DATASETS.get(row[0])
        if dataset is None:
            problems.append(f"Table 8 unknown dataset: {row[0]}")
            continue
        seeds = 3 if dataset in {"cicids2017", "cicddos2019"} else 10
        checked += _record(problems, f"Table 8 {row[0]}/seeds", row[1], str(seeds))
        if dataset == "cicddos2019":
            explanation = source.ddos["explanation"]
            expected = (
                explanation["between_seed_spearman"]["mean"],
                explanation["between_seed_top10_jaccard"]["mean"],
                explanation["versus_raw_spearman"]["mean"],
                explanation["versus_raw_top10_jaccard"]["mean"],
            )
        else:
            between = source.explanations[(dataset, PROPOSED, "between_seed_stability")]
            versus = source.explanations[
                (dataset, PROPOSED, "proposed_vs_raw_explanation_drift")
            ]
            expected = (
                between["mean_spearman"],
                between["mean_top10_jaccard"],
                versus["mean_spearman"],
                versus["mean_top10_jaccard"],
            )
        for column, value in enumerate(expected, start=2):
            checked += _record(
                problems,
                f"Table 8 {row[0]}/column{column}",
                row[column],
                f"{float(value):.4f}",
            )
        checked += _record(
            problems,
            f"Table 8 {row[0]}/additivity",
            row[6],
            f"{source.max_additivity_error(dataset):.3e}",
        )
    return checked
