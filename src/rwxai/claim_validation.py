"""Cell-by-cell validation of manuscript Tables 2-8."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from rwxai.claim_sources import (
    PROPOSED,
    ClaimSources,
    Summary,
    interval_cell,
    mean_sd_cell,
)
from rwxai.claim_validation_extended import check_table7, check_table8
from rwxai.evidence import CheckResult

DATASETS: Final = {
    "UNSW-NB15": "unsw",
    "NSL-KDD": "nslkdd",
    "CIC-IDS-2017跨日": "cicids2017",
    "CIC-DDoS2019跨日": "cicddos2019",
}
BASELINES: Final = {
    "Decision Tree": "DecisionTree",
    "Random Forest": "RandomForest",
    "Extra Trees": "ExtraTrees",
    "Balanced Random Forest": "BalancedRandomForest",
    "EasyEnsemble": "EasyEnsemble",
}
CONTROLS: Final = {
    "Raw": "raw",
    "SMOTE(-NC)+Tomek无权重": "smotenc_tomek",
    "SMOTE(-NC)+Tomek原始权重": "smotenc_tomek_weight",
}
TABLE2_ROWS: Final = (
    ("Raw", "无", "无", "无", "未处理XGBoost基线"),
    ("Weight", "无", "无", "原始分布", "隔离算法层加权"),
    ("ROS", "随机复制", "无", "无", "简单采样基线"),
    ("SMOTE(-NC)", "适度生成", "无", "无", "隔离类别安全生成"),
    ("SMOTE(-NC)+Tomek", "适度生成", "有", "无", "隔离边界清理"),
    ("SMOTE(-NC)+Tomek+原始权重", "适度生成", "有", "原始分布", "检验重复补偿"),
    ("SMOTE(-NC)+Tomek+残差权重", "适度生成", "有", "清理后分布", "本文校正方案"),
)


def _record(problems: list[str], context: str, got: str, expected: str) -> int:
    if got != expected:
        problems.append(f"{context}: claim={got!r}, evidence={expected!r}")
    return 1


def check_table2_claims(claims: dict[str, object]) -> CheckResult:
    """Require every Table 2 cell to match the implemented configurations."""
    rows = claims["tables"]["2"]
    problems: list[str] = []
    expected = [list(row) for row in TABLE2_ROWS]
    if rows[1:] != expected:
        problems.append("Table 2 rows do not match the runner semantics")
    return CheckResult(
        name="table2:configurations",
        passed=not problems,
        detail="7 configurations match every claimed field"
        if not problems
        else "; ".join(problems),
        counts={
            "checked": sum(len(row) for row in expected),
            "problems": len(problems),
        },
    )


def _table3(source: ClaimSources, rows: list[list[str]], problems: list[str]) -> int:
    metrics = (
        "macro_f1",
        "macro_pr_auc_ovr",
        "attack_recall_at_validation_1pct_fpr",
        "threshold_false_alerts_per_10000_benign",
    )
    checked = 0
    for row in rows[1:]:
        dataset = DATASETS.get(row[0])
        baseline = BASELINES.get(row[1])
        if dataset is None or (baseline is None and row[1] != "本文残差权重"):
            problems.append(f"Table 3 unknown row: {row[:2]}")
            continue
        for column, metric in enumerate(metrics, start=2):
            if dataset == "cicddos2019":
                entry = (
                    source.ddos["per_variant"][PROPOSED][metric]
                    if baseline is None
                    else source.ddos["baselines"][baseline][metric]
                )
                summary = Summary(float(entry["mean"]), float(entry["sample_sd"]))
            else:
                summary = (
                    source.performance[(dataset, PROPOSED, metric)]
                    if baseline is None
                    else source.baselines[(dataset, baseline, metric)]
                )
            checked += _record(
                problems,
                f"Table 3 {row[0]}/{row[1]}/{metric}",
                row[column],
                mean_sd_cell(summary, 1 if metric.startswith("threshold_false") else 4),
            )
    return checked


def _table4(source: ClaimSources, rows: list[list[str]], problems: list[str]) -> int:
    metrics = (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "macro_pr_auc_ovr",
        "attack_pr_auc",
        "attack_recall_at_validation_1pct_fpr",
        "threshold_false_alerts_per_10000_benign",
    )
    variants = {"Raw": "raw", "残差权重": PROPOSED}
    checked = 0
    for row in rows[1:]:
        dataset, variant = DATASETS.get(row[0]), variants.get(row[1])
        if dataset is None or variant is None:
            problems.append(f"Table 4 unknown row: {row[:2]}")
            continue
        for column, metric in enumerate(metrics, start=2):
            if dataset == "cicddos2019":
                entry = source.ddos["per_variant"][variant][metric]
                summary = Summary(float(entry["mean"]), float(entry["sample_sd"]))
            else:
                summary = source.performance[(dataset, variant, metric)]
            checked += _record(
                problems,
                f"Table 4 {row[0]}/{row[1]}/{metric}",
                row[column],
                mean_sd_cell(summary, 1 if metric.startswith("threshold_false") else 4),
            )
    return checked


def _table5(source: ClaimSources, rows: list[list[str]], problems: list[str]) -> int:
    metrics = (
        ("macro_f1", 100.0, 2),
        ("attack_recall_at_validation_1pct_fpr", 100.0, 2),
        ("threshold_false_alerts_per_10000_benign", 1.0, 1),
    )
    checked = 0
    for row in rows[1:]:
        dataset, control = DATASETS.get(row[0]), CONTROLS.get(row[1])
        if dataset is None or control is None:
            problems.append(f"Table 5 unknown row: {row[:2]}")
            continue
        for column, (metric, scale, decimals) in enumerate(metrics, start=2):
            interval = (
                source.ddos_interval(control, metric)
                if dataset == "cicddos2019"
                else source.paired_v3(dataset, control, metric)
            )
            actual_scale = 1.0 if dataset == "cicddos2019" else scale
            checked += _record(
                problems,
                f"Table 5 {row[0]}/{row[1]}/{metric}",
                row[column],
                interval_cell(interval, decimals, actual_scale),
            )
    return checked


def _table6(source: ClaimSources, rows: list[list[str]], problems: list[str]) -> int:
    metrics = (
        ("macro_f1", 2),
        ("attack_recall_at_validation_1pct_fpr", 2),
        ("threshold_false_alerts_per_10000_benign", 1),
    )
    checked = 0
    for row in rows[1:]:
        dataset, control = DATASETS.get(row[0]), CONTROLS.get(row[1])
        if dataset not in {"unsw", "nslkdd", "cicids2017"} or control not in {
            "raw",
            "smotenc_tomek_weight",
        }:
            problems.append(f"Table 6 unknown row: {row[:2]}")
            continue
        for column, (metric, decimals) in enumerate(metrics, start=2):
            interval = source.v4_interval(dataset, control, metric)
            checked += _record(
                problems,
                f"Table 6 {row[0]}/{row[1]}/{metric}",
                row[column],
                interval_cell(interval, decimals),
            )
    return checked


def check_result_claims(root: Path, claims: dict[str, object]) -> CheckResult:
    """Compare every result-table value with committed evidence."""
    source = ClaimSources(root)
    tables = claims["tables"]
    problems: list[str] = []
    checked = sum(
        checker(source, tables[str(number)], problems)
        for number, checker in (
            (3, _table3),
            (4, _table4),
            (5, _table5),
            (6, _table6),
            (7, check_table7),
            (8, check_table8),
        )
    )
    return CheckResult(
        name="tables:results",
        passed=not problems,
        detail=f"{checked} result cells match evidence"
        if not problems
        else "; ".join(problems[:3]),
        counts={"checked": checked, "problems": len(problems)},
    )
