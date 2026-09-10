"""Rebuild publication aggregates from the per-seed evidence shipped here."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

from rwxai.evidence import CheckResult

V3_TABLES = (
    "performance_summary.csv",
    "baseline_summary.csv",
    "per_class_summary.csv",
    "explanation_summary.csv",
)


def _run(command: list[str], label: str, problems: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        problems.append(f"{label} failed: {result.stderr.strip()[:200]}")


def _same_value(left: object, right: object) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _same_value(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _same_value(a, b) for a, b in zip(left, right, strict=True)
        )
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    return left == right


def _same_json(left: Path, right: Path) -> bool:
    return _same_value(
        json.loads(left.read_text(encoding="utf-8")),
        json.loads(right.read_text(encoding="utf-8")),
    )


def check_public_aggregates(root: Path) -> CheckResult:
    """Rebuild V3, V4 and DDoS summaries from all public per-seed metrics."""
    evidence = root / "evidence"
    source = root / "src" / "rwxai"
    problems: list[str] = []
    compared = 0
    with tempfile.TemporaryDirectory(prefix="rwxai_aggregates_") as scratch:
        output = Path(scratch)
        v3_output = output / "v3"
        v4_output = output / "v4"
        ddos_output = output / "ddos.json"
        commands = (
            (
                [
                    sys.executable,
                    str(source / "aggregate_v3_results.py"),
                    "--root",
                    str(evidence / "v3" / "per_seed"),
                    "--output",
                    str(v3_output),
                    "--metrics-only",
                ],
                "V3 public aggregation",
            ),
            (
                [
                    sys.executable,
                    str(source / "aggregate_v4_results.py"),
                    "--results",
                    str(evidence / "v4" / "per_seed"),
                    "--out",
                    str(v4_output),
                ],
                "V4 public aggregation",
            ),
            (
                [
                    sys.executable,
                    str(source / "analyze_ddos2019.py"),
                    "--results",
                    str(evidence / "ddos" / "per_seed"),
                    "--out",
                    str(ddos_output),
                ],
                "DDoS public aggregation",
            ),
        )
        for command, label in commands:
            _run(command, label, problems)

        published_v3 = evidence / "v3" / "aggregate"
        for name in V3_TABLES:
            generated = v3_output / name
            published = published_v3 / name
            if (
                not generated.is_file()
                or generated.read_bytes() != published.read_bytes()
            ):
                problems.append(f"V3 aggregate differs: {name}")
            compared += 1

        json_pairs = (
            (
                v4_output / "v4_summary.json",
                evidence / "v4" / "aggregate" / "v4_summary.json",
            ),
            (
                ddos_output,
                evidence / "v4" / "aggregate" / "ddos2019_summary.json",
            ),
        )
        for generated, published in json_pairs:
            if not generated.is_file() or not _same_json(generated, published):
                problems.append(f"aggregate differs: {published.name}")
            compared += 1

    return CheckResult(
        name="aggregates:recomputed",
        passed=not problems,
        detail=(
            f"{compared} aggregate files rebuilt from public per-seed metrics"
            if not problems
            else "; ".join(problems[:3])
        ),
        counts={"compared": compared, "problems": len(problems)},
    )
