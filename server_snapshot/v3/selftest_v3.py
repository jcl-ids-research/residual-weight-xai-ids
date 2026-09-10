from __future__ import annotations

import json

import numpy as np
import pandas as pd

from aggregate_v3_results import paired_seed_bootstrap
from run_v3_explainable import (
    build_resampled_sets,
    evaluate,
    inverse_frequency_weights,
    sample_weights,
    select_attack_threshold,
    train_baselines,
)
from v3_data import TrainOnlyPreprocessor


def main() -> None:
    rng = np.random.default_rng(20260902)
    y = np.asarray([0] * 180 + [1] * 60, dtype=np.int64)
    frame = pd.DataFrame(
        {
            "f1": rng.normal(y * 0.8, 1.0),
            "f2": rng.normal(y * -0.6, 1.0),
            "f3": rng.normal(0.0, 1.0, size=len(y)),
            "constant": 1.0,
        }
    )
    preprocessor = TrainOnlyPreprocessor([]).fit(frame)
    encoded = preprocessor.encode(frame)
    matrix = preprocessor.classifier_matrix(encoded)
    if matrix.shape != (240, 3):
        raise AssertionError(matrix.shape)

    sets, audit = build_resampled_sets(
        encoded,
        y,
        [],
        {1: 120},
        seed=42,
        n_jobs=2,
    )
    if audit["smotenc"]["sampler"] != "SMOTE":
        raise AssertionError(audit)
    tomek_y = sets["smotenc_tomek"][1]
    residual = inverse_frequency_weights(tomek_y, 2)
    weights = sample_weights(tomek_y, residual)
    masses = [float(weights[tomek_y == label].sum()) for label in (0, 1)]
    if not np.isclose(masses[0], masses[1], rtol=1e-6):
        raise AssertionError(masses)

    indices = rng.permutation(len(y))
    train_indices = indices[:140]
    validation_indices = indices[140:190]
    test_indices = indices[190:]
    baselines = train_baselines(
        matrix[train_indices],
        y[train_indices],
        matrix[validation_indices],
        y[validation_indices],
        matrix[test_indices],
        y[test_indices],
        ["BENIGN", "ATTACK"],
        0,
        seed=42,
        n_jobs=2,
        smoke=True,
    )
    if set(baselines) != {
        "DecisionTree",
        "RandomForest",
        "ExtraTrees",
        "BalancedRandomForest",
        "EasyEnsemble",
    }:
        raise AssertionError(baselines)

    multi_y = np.asarray([0] * 210 + [1] * 60 + [2] * 30, dtype=np.int64)
    multi_X = rng.normal(size=(len(multi_y), 5)).astype(np.float32)
    multi_X[:, 0] += multi_y
    train_parts = []
    validation_parts = []
    test_parts = []
    for label in range(3):
        label_indices = np.flatnonzero(multi_y == label)
        train_parts.append(label_indices[: int(len(label_indices) * 0.6)])
        validation_parts.append(
            label_indices[int(len(label_indices) * 0.6): int(len(label_indices) * 0.8)]
        )
        test_parts.append(label_indices[int(len(label_indices) * 0.8):])
    multi_train = np.concatenate(train_parts)
    multi_validation = np.concatenate(validation_parts)
    multi_test = np.concatenate(test_parts)
    multi_baselines = train_baselines(
        multi_X[multi_train],
        multi_y[multi_train],
        multi_X[multi_validation],
        multi_y[multi_validation],
        multi_X[multi_test],
        multi_y[multi_test],
        ["Normal", "Attack-A", "Attack-B"],
        0,
        seed=42,
        n_jobs=2,
        smoke=True,
    )
    if set(multi_baselines) != set(baselines):
        raise AssertionError(multi_baselines)

    validation_y = np.asarray([0] * 100 + [1] * 100)
    validation_attack_probability = np.concatenate(
        [np.linspace(0.0, 0.5, 100), np.linspace(0.2, 1.0, 100)]
    )
    validation_probability = np.column_stack(
        [1.0 - validation_attack_probability, validation_attack_probability]
    )
    test_y = validation_y.copy()
    test_probability = validation_probability.copy()
    test_prediction = test_probability.argmax(axis=1)
    metrics = evaluate(
        validation_y,
        validation_probability,
        test_y,
        test_prediction,
        test_probability,
        ["BENIGN", "ATTACK"],
        0,
    )
    if (
        metrics["validation_operating_point"]["actual_false_positive_rate"]
        > 0.0100000001
    ):
        raise AssertionError(metrics)
    small_y = np.asarray([0] * 25 + [1] * 25)
    small_attack_probability = np.linspace(
        0.01, 0.99, 50, dtype=np.float32
    )
    small_probability = np.column_stack(
        [1.0 - small_attack_probability, small_attack_probability]
    ).astype(np.float32)
    _, small_audit = select_attack_threshold(
        small_y, small_probability, normal_index=0
    )
    if small_audit["allowed_false_positives"] != 0:
        raise AssertionError(small_audit)
    if small_audit["actual_false_positives"] != 0:
        raise AssertionError(small_audit)

    interval = paired_seed_bootstrap(
        [0.01, 0.02, 0.015, 0.012], repetitions=2000
    )
    if interval["ci_low"] <= 0.0:
        raise AssertionError(interval)
    print(
        json.dumps(
            {
                "status": "passed",
                "numeric_features_after_filter": matrix.shape[1],
                "post_tomek_rows": int(len(tomek_y)),
                "effective_class_mass": masses,
                "baselines": sorted(baselines),
                "multiclass_baselines": sorted(multi_baselines),
                "validation_fpr": metrics["validation_operating_point"][
                    "actual_false_positive_rate"
                ],
                "small_validation_false_positives": small_audit[
                    "actual_false_positives"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
