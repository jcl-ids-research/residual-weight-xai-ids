from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path

import imblearn
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC
from imblearn.under_sampling import TomekLinks
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier

from v3_data import (
    LOADERS,
    TrainOnlyPreprocessor,
    class_counts,
    moderate_targets,
    split_official_training,
)


VARIANT_ORDER = (
    "raw",
    "raw_weight",
    "random_over",
    "smotenc",
    "smotenc_tomek",
    "smotenc_tomek_weight",
    "smotenc_tomek_residual_weight",
)

WEIGHTED_VARIANTS = {"raw_weight", "smotenc_tomek_weight"}
RESIDUAL_WEIGHTED_VARIANTS = {"smotenc_tomek_residual_weight"}
ALL_SEEDS = (42, 123, 456, 789, 1001, 2024, 31415, 65537, 77777, 99991)


def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot encode {type(value)!r}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    normal_index: int,
) -> dict:
    labels = np.arange(len(class_names))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    per_p, per_r, per_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class = {
        class_names[index]: {
            "precision": float(per_p[index]),
            "recall": float(per_r[index]),
            "f1": float(per_f1[index]),
            "support": int(support[index]),
        }
        for index in labels
    }
    try:
        auc = float(
            roc_auc_score(
                y_true,
                y_prob,
                labels=labels,
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        auc = None
    class_average_precision = []
    for label in labels:
        try:
            value = average_precision_score(
                (y_true == label).astype(np.int8), y_prob[:, label]
            )
            class_average_precision.append(float(value))
        except ValueError:
            class_average_precision.append(float("nan"))
    finite_ap = [
        value for value in class_average_precision if np.isfinite(value)
    ]
    attack_true = y_true != normal_index
    attack_pred = y_pred != normal_index
    attack_probability = 1.0 - y_prob[:, normal_index]
    attack_pr_auc = float(
        average_precision_score(attack_true.astype(np.int8), attack_probability)
    )
    benign_count = int((~attack_true).sum())
    false_alerts = int((attack_pred & ~attack_true).sum())
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_prob, labels=labels)),
        "macro_roc_auc_ovr": auc,
        "macro_pr_auc_ovr": (
            float(np.mean(finite_ap)) if finite_ap else None
        ),
        "per_class_pr_auc": {
            class_names[index]: class_average_precision[index]
            for index in labels
        },
        "attack_pr_auc": attack_pr_auc,
        "default_attack_recall": float(
            (attack_pred & attack_true).sum() / max(1, int(attack_true.sum()))
        ),
        "false_alerts": false_alerts,
        "false_alerts_per_10000_benign": float(
            false_alerts * 10_000 / max(1, benign_count)
        ),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=labels
        ).tolist(),
    }


def select_attack_threshold(
    y_validation: np.ndarray,
    validation_probability: np.ndarray,
    normal_index: int,
    false_positive_budget: float = 0.01,
) -> tuple[float, dict]:
    validation_attack_scores = (
        1.0
        - validation_probability[:, normal_index].astype(np.float64, copy=False)
    )
    benign_scores = validation_attack_scores[y_validation == normal_index]
    if len(benign_scores) == 0:
        raise ValueError("Validation data contain no benign records")
    allowed_false_positives = int(
        np.floor(false_positive_budget * len(benign_scores))
    )
    threshold = None
    for candidate in np.unique(benign_scores):
        if int((benign_scores >= candidate).sum()) <= allowed_false_positives:
            threshold = float(candidate)
            break
    if threshold is None:
        threshold = float(
            np.nextafter(
                np.float64(benign_scores.max()), np.float64(np.inf)
            )
        )

    validation_attack = y_validation != normal_index
    validation_prediction = validation_attack_scores >= threshold
    validation_false_positives = int(
        (validation_prediction & ~validation_attack).sum()
    )
    return threshold, {
        "false_positive_budget": false_positive_budget,
        "benign_records": int((~validation_attack).sum()),
        "attack_records": int(validation_attack.sum()),
        "allowed_false_positives": allowed_false_positives,
        "actual_false_positives": validation_false_positives,
        "actual_false_positive_rate": float(
            validation_false_positives / max(1, int((~validation_attack).sum()))
        ),
        "attack_recall": float(
            (validation_prediction & validation_attack).sum()
            / max(1, int(validation_attack.sum()))
        ),
    }


def evaluate(
    y_validation: np.ndarray,
    validation_probability: np.ndarray,
    y_test: np.ndarray,
    prediction: np.ndarray,
    probability: np.ndarray,
    class_names: list[str],
    normal_index: int,
) -> dict:
    metrics = score(
        y_test, prediction, probability, class_names, normal_index
    )
    threshold, validation_audit = select_attack_threshold(
        y_validation,
        validation_probability,
        normal_index,
        false_positive_budget=0.01,
    )
    attack_true = y_test != normal_index
    threshold_prediction = (1.0 - probability[:, normal_index]) >= threshold
    true_positives = int((threshold_prediction & attack_true).sum())
    false_positives = int((threshold_prediction & ~attack_true).sum())
    predicted_attacks = int(threshold_prediction.sum())
    benign_count = int((~attack_true).sum())
    metrics.update(
        {
            "validation_attack_threshold_1pct_fpr": threshold,
            "validation_operating_point": validation_audit,
            "attack_recall_at_validation_1pct_fpr": float(
                true_positives / max(1, int(attack_true.sum()))
            ),
            "attack_precision_at_validation_1pct_fpr": float(
                true_positives / max(1, predicted_attacks)
            ),
            "test_fpr_at_validation_1pct_fpr": float(
                false_positives / max(1, benign_count)
            ),
            "threshold_false_alerts": false_positives,
            "threshold_false_alerts_per_10000_benign": float(
                false_positives * 10_000 / max(1, benign_count)
            ),
        }
    )
    return metrics


def inverse_frequency_weights(y: np.ndarray, n_classes: int) -> dict[int, float]:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    total = float(counts.sum())
    result = {}
    for label, count in enumerate(counts):
        if count <= 0:
            raise ValueError(f"Class {label} is absent from sub-training data")
        result[label] = total / (n_classes * count)
    return result


def sample_weights(y: np.ndarray, weights: dict[int, float]) -> np.ndarray:
    values = np.asarray([weights[int(label)] for label in y], dtype=np.float32)
    return values / float(values.mean())


def build_resampled_sets(
    X: np.ndarray,
    y: np.ndarray,
    categorical_indices: list[int],
    strategy: dict[int, int],
    seed: int,
    n_jobs: int,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    sets: dict[str, tuple[np.ndarray, np.ndarray]] = {"raw": (X, y)}
    audit: dict[str, dict] = {}
    if not strategy:
        for name in ("random_over", "smotenc", "smotenc_tomek"):
            sets[name] = (X.copy(), y.copy())
            audit[name] = {"seconds": 0.0, "note": "No class required sampling"}
        return sets, audit

    started = time.perf_counter()
    ros = RandomOverSampler(sampling_strategy=strategy, random_state=seed)
    X_ros, y_ros = ros.fit_resample(X, y)
    sets["random_over"] = (
        X_ros.astype(np.float32, copy=False), y_ros.astype(np.int64, copy=False)
    )
    audit["random_over"] = {"seconds": time.perf_counter() - started}

    targeted_counts = np.bincount(y, minlength=max(strategy) + 1)[list(strategy)]
    neighbors = int(max(1, min(5, targeted_counts.min() - 1)))
    started = time.perf_counter()
    if categorical_indices:
        smote = SMOTENC(
            categorical_features=categorical_indices,
            sampling_strategy=strategy,
            random_state=seed,
            k_neighbors=neighbors,
        )
        sampler_name = "SMOTE-NC"
    else:
        smote = SMOTE(
            sampling_strategy=strategy,
            random_state=seed,
            k_neighbors=neighbors,
        )
        sampler_name = "SMOTE"
    X_smote, y_smote = smote.fit_resample(X, y)
    X_smote = X_smote.astype(np.float32, copy=False)
    y_smote = y_smote.astype(np.int64, copy=False)
    sets["smotenc"] = (X_smote, y_smote)
    audit["smotenc"] = {
        "seconds": time.perf_counter() - started,
        "k_neighbors": neighbors,
        "sampler": sampler_name,
    }

    started = time.perf_counter()
    tomek = TomekLinks(sampling_strategy="all", n_jobs=n_jobs)
    X_tomek, y_tomek = tomek.fit_resample(X_smote, y_smote)
    sets["smotenc_tomek"] = (
        X_tomek.astype(np.float32, copy=False),
        y_tomek.astype(np.int64, copy=False),
    )
    audit["smotenc_tomek"] = {
        "seconds": time.perf_counter() - started,
        "removed_by_tomek": int(len(y_smote) - len(y_tomek)),
    }
    return sets, audit


def select_explanation_indices(
    y: np.ndarray,
    n_classes: int,
    maximum: int,
) -> np.ndarray:
    rng = np.random.default_rng(20260902)
    per_class = max(1, maximum // n_classes)
    pieces: list[np.ndarray] = []
    for label in range(n_classes):
        indices = np.flatnonzero(y == label)
        if len(indices) > per_class:
            indices = rng.choice(indices, size=per_class, replace=False)
        pieces.append(indices)
    return np.sort(np.concatenate(pieces)).astype(np.int64)


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    n_classes: int,
    n_jobs: int,
    device: str,
    smoke: bool,
    weights: np.ndarray | None,
) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        n_estimators=40 if smoke else 800,
        learning_rate=0.08 if smoke else 0.05,
        max_depth=6,
        min_child_weight=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        max_bin=256,
        tree_method="hist",
        device=device,
        random_state=seed,
        n_jobs=n_jobs,
        eval_metric="mlogloss",
        early_stopping_rounds=5 if smoke else 40,
    )
    model.fit(
        X_train,
        y_train,
        sample_weight=weights,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def timed_predictions(
    model: xgb.XGBClassifier,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    model.predict(X_test[: min(512, len(X_test))])
    durations = []
    prediction = None
    probability = None
    for _ in range(3):
        started = time.perf_counter()
        probability = model.predict_proba(X_test)
        prediction = probability.argmax(axis=1)
        durations.append(time.perf_counter() - started)
    assert prediction is not None and probability is not None
    return prediction, probability, {
        "test_seconds_median": float(statistics.median(durations)),
        "test_seconds_all": durations,
        "microseconds_per_record": float(
            statistics.median(durations) * 1_000_000 / len(X_test)
        ),
    }


def tree_shap_audit(
    model: xgb.XGBClassifier,
    X_explain: np.ndarray,
    encoded_explain: np.ndarray,
    explain_indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    preprocessor: TrainOnlyPreprocessor,
    out_path: Path,
) -> dict:
    if preprocessor.expanded_to_original is None:
        raise RuntimeError("Missing feature-group mapping")
    started = time.perf_counter()
    matrix = xgb.DMatrix(X_explain)
    booster = model.get_booster()
    contributions = booster.predict(
        matrix,
        pred_contribs=True,
        approx_contribs=False,
        strict_shape=True,
    )
    margins = booster.predict(matrix, output_margin=True, strict_shape=True)
    if contributions.ndim != 3:
        raise RuntimeError(f"Unexpected SHAP shape: {contributions.shape}")
    feature_contrib = contributions[:, :, :-1]
    bias = contributions[:, :, -1]
    additivity_error = float(
        np.max(np.abs(contributions.sum(axis=2) - margins))
    )

    n_original = len(preprocessor.original_feature_names)
    grouped = np.zeros(
        (len(X_explain), contributions.shape[1], n_original), dtype=np.float32
    )
    for group in range(n_original):
        columns = np.flatnonzero(preprocessor.expanded_to_original == group)
        grouped[:, :, group] = feature_contrib[:, :, columns].sum(axis=2)

    global_importance = np.abs(grouped).mean(axis=(0, 1))
    per_class_importance = np.abs(grouped).mean(axis=0)
    signed_mean = grouped.mean(axis=(0, 1))
    np.savez_compressed(
        out_path,
        test_indices=explain_indices,
        y_true=y_true.astype(np.int64),
        y_pred=y_pred.astype(np.int64),
        y_prob=y_prob.astype(np.float32),
        encoded_features=encoded_explain.astype(np.float32),
        grouped_shap=grouped.astype(np.float32),
        bias=bias.astype(np.float32),
        margins=margins.astype(np.float32),
        feature_names=np.asarray(preprocessor.original_feature_names, dtype=object),
    )
    order = np.argsort(-global_importance)
    return {
        "seconds": time.perf_counter() - started,
        "n_explained": int(len(X_explain)),
        "algorithm": "XGBoost exact TreeSHAP (pred_contribs, approx_contribs=False)",
        "output_space": "raw class margin",
        "additivity_max_abs_error": additivity_error,
        "global_importance": {
            name: float(global_importance[index])
            for index, name in enumerate(preprocessor.original_feature_names)
        },
        "global_signed_mean": {
            name: float(signed_mean[index])
            for index, name in enumerate(preprocessor.original_feature_names)
        },
        "per_class_importance": {
            str(class_index): {
                name: float(per_class_importance[class_index, feature_index])
                for feature_index, name in enumerate(
                    preprocessor.original_feature_names
                )
            }
            for class_index in range(per_class_importance.shape[0])
        },
        "top_10": [preprocessor.original_feature_names[index] for index in order[:10]],
    }


def train_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
    normal_index: int,
    seed: int,
    n_jobs: int,
    smoke: bool,
) -> dict:
    estimators = {
        "DecisionTree": DecisionTreeClassifier(
            random_state=seed, min_samples_leaf=2
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=50 if smoke else 300,
            random_state=seed,
            n_jobs=n_jobs,
            min_samples_leaf=1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=50 if smoke else 300,
            random_state=seed,
            n_jobs=n_jobs,
            min_samples_leaf=1,
        ),
        "BalancedRandomForest": BalancedRandomForestClassifier(
            n_estimators=50 if smoke else 300,
            random_state=seed,
            n_jobs=n_jobs,
            sampling_strategy="all",
            replacement=True,
            bootstrap=False,
        ),
        "EasyEnsemble": EasyEnsembleClassifier(
            n_estimators=3 if smoke else 10,
            random_state=seed,
            n_jobs=n_jobs,
            sampling_strategy="auto",
        ),
    }
    results = {}
    for name, estimator in estimators.items():
        started = time.perf_counter()
        estimator.fit(X_train, y_train)
        train_seconds = time.perf_counter() - started
        validation_probability_raw = estimator.predict_proba(X_validation)
        started = time.perf_counter()
        probability_raw = estimator.predict_proba(X_test)
        classes = np.asarray(estimator.classes_, dtype=np.int64)
        expected = np.arange(len(class_names), dtype=np.int64)
        if not np.array_equal(classes, expected):
            validation_probability = np.zeros(
                (len(X_validation), len(class_names)), dtype=np.float64
            )
            probability = np.zeros(
                (len(X_test), len(class_names)), dtype=np.float64
            )
            validation_probability[:, classes] = validation_probability_raw
            probability[:, classes] = probability_raw
        else:
            validation_probability = validation_probability_raw
            probability = probability_raw
        prediction = probability.argmax(axis=1)
        test_seconds = time.perf_counter() - started
        results[name] = {
            "metrics": evaluate(
                y_validation,
                validation_probability,
                y_test,
                prediction,
                probability,
                class_names,
                normal_index,
            ),
            "train_seconds": train_seconds,
            "test_seconds": test_seconds,
        }
    return results


def run(args: argparse.Namespace) -> Path:
    np.random.seed(args.seed)
    started_all = time.perf_counter()
    output = Path(args.out) / args.dataset / f"seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.json"
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite completed result: {metrics_path}"
        )

    data_dirs = {
        "unsw": args.unsw_dir,
        "nslkdd": args.nslkdd_dir,
        "cicids2017": args.cicids2017_dir,
        "cicddos2019": args.cicddos2019_dir,
    }
    data_dir = data_dirs[args.dataset]
    raw = LOADERS[args.dataset](data_dir, smoke=args.smoke)
    n_classes = len(raw.class_names)
    normal_index = raw.class_names.index(raw.normal_class_name)
    X_sub_frame, X_val_frame, y_sub, y_val = split_official_training(
        raw, args.seed
    )

    preprocessor = TrainOnlyPreprocessor(raw.categorical_columns).fit(X_sub_frame)
    X_sub_encoded = preprocessor.encode(X_sub_frame)
    X_val_encoded = preprocessor.encode(X_val_frame)
    X_test_encoded = preprocessor.encode(raw.X_test)
    X_val = preprocessor.classifier_matrix(X_val_encoded)
    X_test = preprocessor.classifier_matrix(X_test_encoded)
    X_raw_classifier = preprocessor.classifier_matrix(X_sub_encoded)

    strategy = moderate_targets(y_sub, n_classes, expansion_cap=15)
    resampled, sampling_audit = build_resampled_sets(
        X_sub_encoded,
        y_sub,
        preprocessor.categorical_indices,
        strategy,
        args.seed,
        args.n_jobs,
    )
    resampled["raw_weight"] = resampled["raw"]
    resampled["smotenc_tomek_weight"] = resampled["smotenc_tomek"]
    resampled["smotenc_tomek_residual_weight"] = resampled["smotenc_tomek"]
    original_weights = inverse_frequency_weights(y_sub, n_classes)
    residual_weights = inverse_frequency_weights(
        resampled["smotenc_tomek"][1], n_classes
    )
    explanation_indices = select_explanation_indices(
        raw.y_test, n_classes, args.shap_samples
    )

    variants: dict[str, dict] = {}
    for variant in VARIANT_ORDER:
        encoded_train, y_train = resampled[variant]
        X_train = preprocessor.classifier_matrix(encoded_train)
        if variant in WEIGHTED_VARIANTS:
            active_weight_values = original_weights
            weight_source = "original_sub_training_distribution"
        elif variant in RESIDUAL_WEIGHTED_VARIANTS:
            active_weight_values = residual_weights
            weight_source = "post_smote_tomek_distribution"
        else:
            active_weight_values = None
            weight_source = "none"
        weights = (
            sample_weights(y_train, active_weight_values)
            if active_weight_values is not None
            else None
        )
        train_started = time.perf_counter()
        model = train_xgboost(
            X_train,
            y_train,
            X_val,
            y_val,
            args.seed,
            n_classes,
            args.n_jobs,
            args.device,
            args.smoke,
            weights,
        )
        train_seconds = time.perf_counter() - train_started
        validation_probability = model.predict_proba(X_val)
        prediction, probability, inference = timed_predictions(model, X_test)
        variant_metrics = evaluate(
            y_val,
            validation_probability,
            raw.y_test,
            prediction,
            probability,
            raw.class_names,
            normal_index,
        )
        model_path = output / f"xgboost_{variant}.ubj"
        model.save_model(model_path)
        np.savez_compressed(
            output / f"predictions_{variant}.npz",
            y_true=raw.y_test.astype(np.int64),
            y_pred=prediction.astype(np.int64),
            y_prob=probability.astype(np.float32),
            class_names=np.asarray(raw.class_names, dtype=object),
        )
        np.savez_compressed(
            output / f"validation_predictions_{variant}.npz",
            y_true=y_val.astype(np.int64),
            y_prob=validation_probability.astype(np.float32),
            selected_attack_threshold=np.asarray(
                [variant_metrics["validation_attack_threshold_1pct_fpr"]],
                dtype=np.float64,
            ),
        )
        shap_audit = tree_shap_audit(
            model,
            X_test[explanation_indices],
            X_test_encoded[explanation_indices],
            explanation_indices,
            raw.y_test[explanation_indices],
            prediction[explanation_indices],
            probability[explanation_indices],
            preprocessor,
            output / f"shap_{variant}.npz",
        )
        if weights is None:
            effective_class_mass = {
                str(label): float((y_train == label).sum())
                for label in range(n_classes)
            }
        else:
            effective_class_mass = {
                str(label): float(weights[y_train == label].sum())
                for label in range(n_classes)
            }
        variants[variant] = {
            "metrics": variant_metrics,
            "train_seconds": train_seconds,
            "inference": inference,
            "best_iteration": int(model.best_iteration),
            "model_size_bytes": int(model_path.stat().st_size),
            "model_sha256": sha256(model_path),
            "training_rows": int(len(y_train)),
            "training_class_counts": class_counts(y_train, n_classes),
            "weight_source": weight_source,
            "class_weight_values": (
                None
                if active_weight_values is None
                else {
                    str(label): float(value)
                    for label, value in active_weight_values.items()
                }
            ),
            "effective_weighted_class_mass": effective_class_mass,
            "tree_shap": shap_audit,
        }
        print(
            f"[{args.dataset} seed={args.seed} {variant}] "
            f"macro-F1={variants[variant]['metrics']['macro_f1']:.4f} "
            f"macro-recall={variants[variant]['metrics']['macro_recall']:.4f} "
            f"train={train_seconds:.1f}s",
            flush=True,
        )
        del X_train, model

    baselines = train_baselines(
        X_raw_classifier,
        y_sub,
        X_val,
        y_val,
        X_test,
        raw.y_test,
        raw.class_names,
        normal_index,
        args.seed,
        args.n_jobs,
        args.smoke,
    )

    result = {
        "schema_version": 2,
        "dataset": args.dataset,
        "seed": args.seed,
        "smoke": args.smoke,
        "protocol": {
            # CIC-DDoS2019 evaluates on a fixed stratified sample of the later
            # capture day, so the untouched-test claim does not hold there.
            "official_test_set_untouched": args.dataset != "cicddos2019",
            "validation_fraction_of_official_train": 0.1,
            "split_before_preprocessing": True,
            "preprocessing_fit_scope": "sub-training only",
            "resampling_fit_scope": "sub-training only",
            "validation_content": "real records only",
            "test_content": "real records only",
            "common_xgboost_configuration_across_variants": True,
            "test_set_used_for_selection": False,
            "operating_threshold_fit_scope": "validation partition only",
            "operating_threshold_false_positive_budget": 0.01,
            "dataset_partition": raw.protocol,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "imbalanced_learn": imblearn.__version__,
            "xgboost": xgb.__version__,
            "device": args.device,
            "n_jobs": args.n_jobs,
        },
        "class_names": raw.class_names,
        "normal_class_name": raw.normal_class_name,
        "normal_class_index": normal_index,
        "sizes": {
            "official_train": int(len(raw.y_train)),
            "sub_train": int(len(y_sub)),
            "validation": int(len(y_val)),
            "official_test": int(len(raw.y_test)),
            "original_features": len(preprocessor.original_feature_names),
            "classifier_features_after_onehot": len(
                preprocessor.expanded_feature_names
            ),
        },
        "class_counts": {
            "official_train": class_counts(raw.y_train, n_classes),
            "sub_train": class_counts(y_sub, n_classes),
            "validation": class_counts(y_val, n_classes),
            "official_test": class_counts(raw.y_test, n_classes),
        },
        "preprocessing": preprocessor.audit(),
        "sampling": {
            "target_rule": "floor(mean sub-training class count), capped at 15x each original class",
            "targets": {str(k): v for k, v in strategy.items()},
            "audit": sampling_audit,
            "sampler_policy": (
                "SMOTE-NC when categorical features exist; SMOTE for purely "
                "numeric data"
            ),
            "categorical_interpolation": (
                "categorical values are never linearly interpolated"
                if raw.categorical_columns
                else "not applicable to the purely numeric dataset"
            ),
            "tomek_policy": (
                "TomekLinks(sampling_strategy='all') after SMOTE or SMOTE-NC"
            ),
        },
        "class_weights": {
            "formula": "N / (K * n_c)",
            "normalization": "per-variant sample weights normalized to mean 1",
            "original_distribution_values": {
                str(k): v for k, v in original_weights.items()
            },
            "post_smote_tomek_residual_values": {
                str(k): v for k, v in residual_weights.items()
            },
            "proposed_correction": (
                "recompute inverse-frequency weights from the post-SMOTE/"
                "Tomek class distribution to avoid duplicate compensation"
            ),
        },
        "proposed_variant": "smotenc_tomek_residual_weight",
        "variants": variants,
        "raw_training_baselines": baselines,
        "total_seconds": time.perf_counter() - started_all,
    }
    temporary = output / "metrics.json.tmp"
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    os.replace(temporary, metrics_path)
    print(f"[OK] {metrics_path}", flush=True)
    return metrics_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(LOADERS), required=True)
    parser.add_argument("--seed", type=int, choices=ALL_SEEDS, required=True)
    parser.add_argument(
        "--out", default="/opt/ids_revision/v3_strengthened/results"
    )
    parser.add_argument("--unsw-dir", default="/opt/UNSW-NB15")
    parser.add_argument("--nslkdd-dir", default="/opt/NSL-KDD")
    parser.add_argument(
        "--cicids2017-dir",
        default="/opt/CIC-IDS-2017/MachineLearningCVE",
    )
    parser.add_argument(
        "--cicddos2019-dir",
        dest="cicddos2019_dir",
        default="/opt/ids_revision/v4_deep_baseline/cache",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--n-jobs", type=int, default=12)
    parser.add_argument("--shap-samples", type=int, default=1000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
