"""Aggregate the CIC-DDoS2019 cross-day runs under the V3 statistical protocol.

This dataset inverts the imbalance direction of the other three benchmarks:
benign traffic is the minority class. The script reports the same headline
metrics, the same paired seed-level bootstrap intervals, and the per-class
total weight mass that evidences duplicate compensation.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Final

import numpy as np

DATASET: Final = "cicddos2019"
SEEDS: Final = (42, 123, 456)
RAW: Final = "raw"
UNWEIGHTED: Final = "smotenc_tomek"
ORIGINAL: Final = "smotenc_tomek_weight"
RESIDUAL: Final = "smotenc_tomek_residual_weight"
CONTROLS: Final = (RAW, UNWEIGHTED, ORIGINAL)
HEADLINE: Final = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "macro_pr_auc_ovr",
    "attack_pr_auc",
    "attack_recall_at_validation_1pct_fpr",
    "threshold_false_alerts_per_10000_benign",
)
PERCENTAGE_POINT: Final = (
    "macro_f1",
    "attack_recall_at_validation_1pct_fpr",
)


def mean_sd(values: list[float]) -> dict[str, float]:
    """Return the mean and sample standard deviation."""
    return {
        "values": values,
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def paired_seed_bootstrap(
    differences: list[float], *, repetitions: int = 20_000, seed: int = 20260902
) -> dict:
    """Bootstrap paired seed-level differences, matching the V3 protocol."""
    values = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        **mean_sd(values.tolist()),
        "bootstrap_repetitions": repetitions,
        "ci_level": 0.95,
        "ci_low": float(low),
        "ci_high": float(high),
    }


def mass_statistics(mass: dict[str, float]) -> dict[str, float]:
    """Describe how unevenly total class weight is distributed."""
    values = [float(item) for item in mass.values()]
    lowest, highest = min(values), max(values)
    return {
        "minimum": lowest,
        "maximum": highest,
        "ratio": highest / lowest if lowest > 0 else float("inf"),
    }


def rank_vector(values: np.ndarray) -> np.ndarray:
    """Return average ranks, assigning tied values their mean rank.

    The earlier implementation broke ties by sort position, which is not the
    Spearman definition. Tied SHAP importances do occur, so ties now receive the
    average of the ranks they span.
    """
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    ordered = values[order]
    start = 0
    for position in range(1, len(ordered) + 1):
        if position == len(ordered) or ordered[position] != ordered[start]:
            if position - start > 1:
                ranks[order[start:position]] = ranks[order[start:position]].mean()
            start = position
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Spearman rank correlation between two importance vectors."""
    return float(np.corrcoef(rank_vector(left), rank_vector(right))[0, 1])


def top_k_jaccard(left: np.ndarray, right: np.ndarray, k: int = 10) -> float:
    """Jaccard overlap of the top-k features of two importance vectors."""
    left_top = set(np.argsort(-left)[:k].tolist())
    right_top = set(np.argsort(-right)[:k].tolist())
    return len(left_top & right_top) / len(left_top | right_top)


def shap_vector(payload: dict, variant: str) -> tuple[list[str], np.ndarray]:
    """Extract the global TreeSHAP importance vector for one variant."""
    mapping = payload["variants"][variant]["tree_shap"]["global_importance"]
    names = list(mapping)
    return names, np.asarray([mapping[name] for name in names], dtype=np.float64)


def explanation_stability(payloads: dict) -> dict:
    """Compute between-seed and versus-raw TreeSHAP stability."""
    vectors: dict[int, np.ndarray] = {}
    raw_vectors: dict[int, np.ndarray] = {}
    reference: list[str] | None = None
    errors: list[float] = []
    for seed, payload in payloads.items():
        names, values = shap_vector(payload, RESIDUAL)
        raw_names, raw_values = shap_vector(payload, RAW)
        if reference is None:
            reference = names
        if names != reference or raw_names != reference:
            raise ValueError("TreeSHAP feature ordering differs across variants")
        vectors[seed] = values
        raw_vectors[seed] = raw_values
        errors.append(
            float(
                payload["variants"][RESIDUAL]["tree_shap"][
                    "additivity_max_abs_error"
                ]
            )
        )
    seeds = sorted(vectors)
    between_spearman = [
        spearman(vectors[a], vectors[b])
        for index, a in enumerate(seeds)
        for b in seeds[index + 1 :]
    ]
    between_jaccard = [
        top_k_jaccard(vectors[a], vectors[b])
        for index, a in enumerate(seeds)
        for b in seeds[index + 1 :]
    ]
    versus_raw_spearman = [spearman(vectors[s], raw_vectors[s]) for s in seeds]
    versus_raw_jaccard = [top_k_jaccard(vectors[s], raw_vectors[s]) for s in seeds]
    return {
        "paired_seeds": len(seeds),
        "between_seed_spearman": mean_sd(between_spearman),
        "between_seed_top10_jaccard": mean_sd(between_jaccard),
        "versus_raw_spearman": mean_sd(versus_raw_spearman),
        "versus_raw_top10_jaccard": mean_sd(versus_raw_jaccard),
        "max_additivity_error": max(errors),
        "top_features": list(payloads[seeds[0]]["variants"][RESIDUAL]["tree_shap"]["top_10"])[:5],
    }


def baseline_summary(payloads: dict) -> dict:
    """Aggregate the five strong baselines across paired seeds."""
    seeds = sorted(payloads)
    names = list(payloads[seeds[0]]["raw_training_baselines"])
    metrics = (
        "macro_f1",
        "macro_pr_auc_ovr",
        "attack_recall_at_validation_1pct_fpr",
        "threshold_false_alerts_per_10000_benign",
    )
    return {
        name: {
            metric: mean_sd(
                [
                    float(
                        payloads[seed]["raw_training_baselines"][name]["metrics"][metric]
                    )
                    for seed in seeds
                ]
            )
            for metric in metrics
        }
        for name in names
    }


def per_class_summary(payloads: dict) -> dict:
    """Aggregate per-class precision, recall, and F1 for raw and residual paths."""
    seeds = sorted(payloads)
    classes = list(payloads[seeds[0]]["variants"][RAW]["metrics"]["per_class"])
    result: dict[str, dict] = {}
    for name in classes:
        entry: dict[str, dict] = {
            "support": int(
                payloads[seeds[0]]["variants"][RAW]["metrics"]["per_class"][name][
                    "support"
                ]
            )
        }
        for variant, tag in ((RAW, "raw"), (RESIDUAL, "residual")):
            for metric in ("precision", "recall", "f1"):
                entry[f"{tag}_{metric}"] = mean_sd(
                    [
                        float(
                            payloads[seed]["variants"][variant]["metrics"]["per_class"][
                                name
                            ][metric]
                        )
                        for seed in seeds
                    ]
                )
        result[name] = entry
    return result


def main() -> None:
    """Aggregate the runs and write the publication-ready summary."""
    parser = argparse.ArgumentParser(description="Analyse CIC-DDoS2019 runs")
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.results) / DATASET
    payloads = {}
    for seed in SEEDS:
        path = root / f"seed{seed}" / "metrics.json"
        payloads[seed] = json.loads(path.read_text(encoding="utf-8"))

    variants = sorted(payloads[SEEDS[0]]["variants"])
    summary = {
        variant: {
            metric: mean_sd(
                [
                    float(payloads[seed]["variants"][variant]["metrics"][metric])
                    for seed in SEEDS
                ]
            )
            for metric in HEADLINE
        }
        for variant in variants
    }

    differences = {}
    for control in CONTROLS:
        entry = {}
        for metric in HEADLINE:
            scale = 100.0 if metric in PERCENTAGE_POINT else 1.0
            entry[metric] = paired_seed_bootstrap(
                [
                    (
                        float(payloads[seed]["variants"][RESIDUAL]["metrics"][metric])
                        - float(payloads[seed]["variants"][control]["metrics"][metric])
                    )
                    * scale
                    for seed in SEEDS
                ]
            )
        differences[f"residual_minus_{control}"] = entry

    mass = {}
    for variant in (ORIGINAL, RESIDUAL):
        ratios = [
            mass_statistics(
                payloads[seed]["variants"][variant]["effective_weighted_class_mass"]
            )["ratio"]
            for seed in SEEDS
        ]
        mass[variant] = mean_sd(ratios)

    protocol = payloads[SEEDS[0]].get("protocol", {})
    report = {
        "dataset": DATASET,
        "seeds": list(SEEDS),
        "protocol": protocol,
        "per_variant": summary,
        "paired_differences": differences,
        "weight_mass_ratio": mass,
        "baselines": baseline_summary(payloads),
        "per_class": per_class_summary(payloads),
        "explanation": explanation_stability(payloads),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for variant in (RAW, ORIGINAL, RESIDUAL):
        stats = summary[variant]
        print(
            f"{variant:34s} F1={stats['macro_f1']['mean']:.4f}"
            f"+-{stats['macro_f1']['sample_sd']:.4f}"
            f"  rec={stats['attack_recall_at_validation_1pct_fpr']['mean']:.4f}"
            f"  FA/1e4={stats['threshold_false_alerts_per_10000_benign']['mean']:.1f}"
        )
    for name, entry in differences.items():
        f1 = entry["macro_f1"]
        print(
            f"{name:28s} dF1={f1['mean']:+.2f}pp "
            f"[{f1['ci_low']:+.2f}, {f1['ci_high']:+.2f}]"
        )
    for variant, stats in mass.items():
        print(f"mass ratio {variant:34s} {stats['mean']:.6f}")


if __name__ == "__main__":
    main()
