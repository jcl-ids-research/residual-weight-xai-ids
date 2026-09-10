from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import statistics
from pathlib import Path

import numpy as np


DATASET_SEEDS = {
    "unsw": (42, 123, 456, 789, 1001, 2024, 31415, 65537, 77777, 99991),
    "nslkdd": (42, 123, 456, 789, 1001, 2024, 31415, 65537, 77777, 99991),
    "cicids2017": (42, 123, 456),
}
DATASETS = tuple(DATASET_SEEDS)
VARIANTS = (
    "raw",
    "raw_weight",
    "random_over",
    "smotenc",
    "smotenc_tomek",
    "smotenc_tomek_weight",
    "smotenc_tomek_residual_weight",
)
PROPOSED = "smotenc_tomek_residual_weight"
BASELINES = (
    "DecisionTree",
    "RandomForest",
    "ExtraTrees",
    "BalancedRandomForest",
    "EasyEnsemble",
)
METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "mcc",
    "log_loss",
    "macro_roc_auc_ovr",
    "macro_pr_auc_ovr",
    "attack_pr_auc",
    "default_attack_recall",
    "false_alerts_per_10000_benign",
    "attack_recall_at_validation_1pct_fpr",
    "attack_precision_at_validation_1pct_fpr",
    "test_fpr_at_validation_1pct_fpr",
    "threshold_false_alerts_per_10000_benign",
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean_sd(values: list[float]) -> dict:
    return {
        "values": values,
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values),
    }


def paired_seed_bootstrap(
    differences: list[float],
    *,
    repetitions: int = 20_000,
    seed: int = 20260902,
) -> dict:
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
        "probability_difference_above_zero": float(np.mean(means > 0.0)),
    }


def rank_vector(values: np.ndarray) -> np.ndarray:
    """Return average ranks, assigning tied values their mean rank.

    Sort-position ranking silently breaks ties and is not the Spearman
    definition. This mirrors ``analyze_ddos2019.rank_vector`` so both pipelines
    compute the statistic identically.
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
    return float(np.corrcoef(rank_vector(left), rank_vector(right))[0, 1])


def top_k_jaccard(left: np.ndarray, right: np.ndarray, k: int = 10) -> float:
    left_top = set(np.argsort(-left)[:k].tolist())
    right_top = set(np.argsort(-right)[:k].tolist())
    return len(left_top & right_top) / len(left_top | right_top)


def sign_consistency(values: list[float], tolerance: float = 1e-12) -> str:
    signs = {
        1 if value > tolerance else -1 if value < -tolerance else 0
        for value in values
    }
    if signs == {1}:
        return "all_positive"
    if signs == {-1}:
        return "all_negative"
    if signs == {0}:
        return "all_zero"
    return "mixed"


def shap_vector(result: dict, variant: str) -> tuple[list[str], np.ndarray]:
    mapping = result["variants"][variant]["tree_shap"]["global_importance"]
    names = list(mapping)
    return names, np.asarray([mapping[name] for name in names], dtype=np.float64)


def verify_run(path: Path, result: dict) -> None:
    if result.get("smoke"):
        raise RuntimeError(f"Smoke result cannot enter the paper: {path}")
    protocol = result["protocol"]
    required_protocol = (
        "official_test_set_untouched",
        "split_before_preprocessing",
        "test_set_used_for_selection",
    )
    if not protocol[required_protocol[0]] or not protocol[required_protocol[1]]:
        raise RuntimeError(f"Protocol gate failed: {path}")
    if protocol[required_protocol[2]]:
        raise RuntimeError(f"Test-set selection gate failed: {path}")
    if protocol.get("operating_threshold_fit_scope") != "validation partition only":
        raise RuntimeError(f"Threshold-selection gate failed: {path}")
    if result.get("schema_version") != 2:
        raise RuntimeError(f"Unexpected result schema: {path}")
    if result.get("proposed_variant") != PROPOSED:
        raise RuntimeError(f"Proposed-method identity failed: {path}")
    if set(result["variants"]) != set(VARIANTS):
        raise RuntimeError(f"Variant coverage failed: {path}")
    if set(result["raw_training_baselines"]) != set(BASELINES):
        raise RuntimeError(f"Baseline coverage failed: {path}")
    if result["dataset"] == "cicids2017":
        partition = protocol["dataset_partition"]
        if (
            not partition.get("temporal_holdout")
            or partition.get("training_period") != "Monday through Thursday"
            or partition.get("test_period") != "Friday"
            or partition.get("test_cap") is not None
        ):
            raise RuntimeError(f"CIC temporal protocol gate failed: {path}")
    for variant in VARIANTS:
        error = result["variants"][variant]["tree_shap"][
            "additivity_max_abs_error"
        ]
        if error > 1e-3:
            raise RuntimeError(
                f"TreeSHAP additivity failed ({error}) for {path}, {variant}"
            )
        validation = result["variants"][variant]["metrics"][
            "validation_operating_point"
        ]
        if validation["actual_false_positive_rate"] > 0.0100000001:
            raise RuntimeError(
                f"Validation false-positive budget failed for {path}, {variant}"
            )


def verify_test_identity(root: Path, dataset: str) -> str:
    reference_y = None
    reference_names = None
    reference_indices = None
    validation_by_seed: dict[int, np.ndarray] = {}
    digest = None
    for seed in DATASET_SEEDS[dataset]:
        for variant in VARIANTS:
            prediction_path = (
                root / dataset / f"seed{seed}" / f"predictions_{variant}.npz"
            )
            shap_path = root / dataset / f"seed{seed}" / f"shap_{variant}.npz"
            validation_path = (
                root / dataset / f"seed{seed}"
                / f"validation_predictions_{variant}.npz"
            )
            saved = np.load(prediction_path, allow_pickle=True)
            y_true = saved["y_true"]
            names = saved["class_names"].tolist()
            shap_saved = np.load(shap_path, allow_pickle=True)
            indices = shap_saved["test_indices"]
            validation_saved = np.load(validation_path, allow_pickle=True)
            validation_y = validation_saved["y_true"]
            validation_threshold = float(
                validation_saved["selected_attack_threshold"][0]
            )
            metric_threshold = float(
                load_json(root / dataset / f"seed{seed}" / "metrics.json")
                ["variants"][variant]["metrics"]
                ["validation_attack_threshold_1pct_fpr"]
            )
            if not np.isclose(validation_threshold, metric_threshold):
                raise RuntimeError(
                    f"Saved threshold identity failed: {dataset}, {seed}, {variant}"
                )
            if seed not in validation_by_seed:
                validation_by_seed[seed] = validation_y.copy()
            elif not np.array_equal(validation_by_seed[seed], validation_y):
                raise RuntimeError(
                    f"Validation identity failed: {dataset}, {seed}, {variant}"
                )
            if reference_y is None:
                reference_y = y_true.copy()
                reference_names = names
                reference_indices = indices.copy()
                digest = hashlib.sha256(y_true.tobytes()).hexdigest()
            elif (
                not np.array_equal(reference_y, y_true)
                or reference_names != names
                or not np.array_equal(reference_indices, indices)
            ):
                raise RuntimeError(
                    f"Test/explanation identity failed: {dataset}, seed {seed}, {variant}"
                )
    assert digest is not None
    return digest


def aggregate(root: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    runs: dict[tuple[str, int], dict] = {}
    for dataset in DATASETS:
        for seed in DATASET_SEEDS[dataset]:
            path = root / dataset / f"seed{seed}" / "metrics.json"
            if not path.exists():
                raise FileNotFoundError(path)
            result = load_json(path)
            verify_run(path, result)
            if result["dataset"] != dataset or result["seed"] != seed:
                raise RuntimeError(f"Run identity failed: {path}")
            runs[(dataset, seed)] = result

    test_digests = {
        dataset: verify_test_identity(root, dataset) for dataset in DATASETS
    }

    performance_rows = []
    performance_summary: dict[str, dict] = {}
    per_class_rows = []
    explanation_rows = []
    explanation_summary: dict[str, dict] = {}
    baseline_rows = []
    baseline_summary: dict[str, dict] = {}

    for dataset in DATASETS:
        seeds = DATASET_SEEDS[dataset]
        performance_summary[dataset] = {}
        baseline_summary[dataset] = {}
        for variant in VARIANTS:
            variant_summary = {}
            for metric in METRICS:
                values = [
                    runs[(dataset, seed)]["variants"][variant]["metrics"][metric]
                    for seed in seeds
                ]
                if any(value is None for value in values):
                    continue
                stats = mean_sd([float(value) for value in values])
                variant_summary[metric] = stats
                performance_rows.append(
                    [
                        dataset,
                        variant,
                        metric,
                        f"{stats['mean']:.10f}",
                        f"{stats['sample_sd']:.10f}",
                        json.dumps(list(seeds)),
                        json.dumps(stats["values"]),
                    ]
                )
            performance_summary[dataset][variant] = variant_summary

        paired_comparisons = {}
        for metric in METRICS:
            raw_values = [
                runs[(dataset, seed)]["variants"]["raw"]["metrics"][metric]
                for seed in seeds
            ]
            proposed_values = [
                runs[(dataset, seed)]["variants"][PROPOSED]["metrics"][metric]
                for seed in seeds
            ]
            if any(value is None for value in raw_values + proposed_values):
                continue
            differences = [
                float(proposed - raw)
                for raw, proposed in zip(raw_values, proposed_values)
            ]
            paired_comparisons[metric] = {
                **paired_seed_bootstrap(
                    differences,
                    seed=20260902 + sum(ord(char) for char in dataset + metric),
                ),
                "sign_consistency": sign_consistency(differences),
                "difference_definition": "proposed minus raw",
                "preferred_direction": (
                    "negative"
                    if metric
                    in {
                        "log_loss",
                        "false_alerts_per_10000_benign",
                        "test_fpr_at_validation_1pct_fpr",
                        "threshold_false_alerts_per_10000_benign",
                    }
                    else "positive"
                ),
            }
        performance_summary[dataset][
            "paired_proposed_minus_raw"
        ] = paired_comparisons

        for baseline in BASELINES:
            baseline_summary[dataset][baseline] = {}
            for metric in METRICS:
                values = [
                    runs[(dataset, seed)]["raw_training_baselines"][baseline]
                    ["metrics"][metric]
                    for seed in seeds
                ]
                if any(value is None for value in values):
                    continue
                stats = mean_sd([float(value) for value in values])
                baseline_summary[dataset][baseline][metric] = stats
                baseline_rows.append(
                    [
                        dataset,
                        baseline,
                        metric,
                        f"{stats['mean']:.10f}",
                        f"{stats['sample_sd']:.10f}",
                        json.dumps(list(seeds)),
                        json.dumps(stats["values"]),
                    ]
                )

        class_names = runs[(dataset, seeds[0])]["class_names"]
        for class_name in class_names:
            for variant in ("raw", PROPOSED):
                for metric in ("precision", "recall", "f1"):
                    values = [
                        runs[(dataset, seed)]["variants"][variant]["metrics"]
                        ["per_class"][class_name][metric]
                        for seed in seeds
                    ]
                    stats = mean_sd([float(value) for value in values])
                    per_class_rows.append(
                        [
                            dataset,
                            class_name,
                            variant,
                            metric,
                            f"{stats['mean']:.10f}",
                            f"{stats['sample_sd']:.10f}",
                            json.dumps(list(seeds)),
                            json.dumps(stats["values"]),
                        ]
                    )

        explanation_summary[dataset] = {}
        for variant in VARIANTS:
            vectors = []
            names_reference = None
            for seed in seeds:
                names, vector = shap_vector(runs[(dataset, seed)], variant)
                if names_reference is None:
                    names_reference = names
                elif names_reference != names:
                    raise RuntimeError(
                        f"Feature order differs: {dataset}, {variant}, seed {seed}"
                    )
                vectors.append(vector)
            correlations = []
            overlaps = []
            for left, right in itertools.combinations(vectors, 2):
                correlations.append(spearman(left, right))
                overlaps.append(top_k_jaccard(left, right))
            explanation_summary[dataset][variant] = {
                "pairwise_seed_spearman": correlations,
                "mean_seed_spearman": statistics.mean(correlations),
                "pairwise_seed_top10_jaccard": overlaps,
                "mean_seed_top10_jaccard": statistics.mean(overlaps),
            }
            explanation_rows.append(
                [
                    dataset,
                    variant,
                    "between_seed_stability",
                    f"{statistics.mean(correlations):.10f}",
                    f"{statistics.mean(overlaps):.10f}",
                ]
            )

        drift_correlations = []
        drift_overlaps = []
        for seed in seeds:
            raw_names, raw_vector = shap_vector(runs[(dataset, seed)], "raw")
            proposed_names, proposed_vector = shap_vector(
                runs[(dataset, seed)], PROPOSED
            )
            if raw_names != proposed_names:
                raise RuntimeError(f"Raw/proposed feature mismatch: {dataset}, {seed}")
            drift_correlations.append(spearman(raw_vector, proposed_vector))
            drift_overlaps.append(top_k_jaccard(raw_vector, proposed_vector))
        explanation_summary[dataset]["proposed_vs_raw"] = {
            "same_seed_spearman": drift_correlations,
            "mean_same_seed_spearman": statistics.mean(drift_correlations),
            "same_seed_top10_jaccard": drift_overlaps,
            "mean_same_seed_top10_jaccard": statistics.mean(drift_overlaps),
        }
        explanation_rows.append(
            [
                dataset,
                PROPOSED,
                "proposed_vs_raw_explanation_drift",
                f"{statistics.mean(drift_correlations):.10f}",
                f"{statistics.mean(drift_overlaps):.10f}",
            ]
        )

    manifest = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in {".json", ".npz", ".ubj"}:
            manifest.append(
                {
                    "path": str(path.relative_to(root)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )

    summary = {
        "schema_version": 2,
        "datasets": list(DATASETS),
        "seeds_by_dataset": {
            dataset: list(seeds) for dataset, seeds in DATASET_SEEDS.items()
        },
        "required_runs": sum(len(seeds) for seeds in DATASET_SEEDS.values()),
        "verified_runs": len(runs),
        "proposed_variant": PROPOSED,
        "performance": performance_summary,
        "raw_training_baselines": baseline_summary,
        "explanation": explanation_summary,
        "test_label_sha256": test_digests,
        "evidence_files": len(manifest),
        "statistical_scope": (
            "UNSW-NB15 and NSL-KDD use ten paired seeds; CIC-IDS-2017 uses "
            "three paired seeds as a temporal external validation. The 95% "
            "bootstrap intervals resample paired seed-level differences, not "
            "individual traffic records, to avoid pseudoreplication."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "evidence_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (output / "performance_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "dataset", "variant", "metric", "mean", "sample_sd",
                "seed_ids_json", "seed_values_json",
            ]
        )
        writer.writerows(performance_rows)
    with (output / "baseline_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "dataset", "baseline", "metric", "mean", "sample_sd",
                "seed_ids_json", "seed_values_json",
            ]
        )
        writer.writerows(baseline_rows)
    with (output / "per_class_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "dataset", "class", "variant", "metric", "mean", "sample_sd",
                "seed_ids_json", "seed_values_json",
            ]
        )
        writer.writerows(per_class_rows)
    with (output / "explanation_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["dataset", "variant", "comparison", "mean_spearman", "mean_top10_jaccard"]
        )
        writer.writerows(explanation_rows)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="/opt/ids_revision/v3_strengthened/results"
    )
    parser.add_argument(
        "--output", default="/opt/ids_revision/v3_strengthened/aggregate"
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = aggregate(Path(arguments.root), Path(arguments.output))
    print(json.dumps({
        "verified_runs": result["verified_runs"],
        "evidence_files": result["evidence_files"],
    }))
