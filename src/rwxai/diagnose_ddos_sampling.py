#!/usr/bin/env python3
"""Quantify the CIC-DDoS2019 sampling bias before deciding on a re-run.

The cache builder keeps rows with probability ``per_file_target / rows_seen``,
which decays as a file is consumed, and caps every file at the same absolute
count regardless of its size. Two consequences need measuring rather than
assuming:

  1. within-file position bias -- only harmful if rows are ordered, so the
     Timestamp monotonicity and the label mix of the first chunk versus the
     whole file are measured directly;
  2. between-file distortion -- a uniform cap over unequal files changes the
     attack-family mix, so the realised per-family share is compared against
     the true full-period share.

Nothing here is inferred from the existing cache; every number is recomputed
from the raw captures.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CHUNK_ROWS = 400_000


def file_profile(path: Path, first_chunk_rows: int) -> dict[str, Any]:
    """Measure ordering and label drift between a file's head and its whole."""
    head_labels: Counter[str] = Counter()
    all_labels: Counter[str] = Counter()
    timestamps_sorted = True
    previous: str | None = None
    rows = 0
    for chunk in pd.read_csv(
        path, chunksize=CHUNK_ROWS, low_memory=False,
        usecols=lambda c: c.strip().casefold() in {"label", "timestamp"},
    ):
        chunk.columns = [str(c).strip().casefold() for c in chunk.columns]
        labels = chunk["label"].astype(str).str.strip()
        all_labels.update(labels.tolist())
        if rows < first_chunk_rows:
            head_labels.update(labels.head(first_chunk_rows - rows).tolist())
        if "timestamp" in chunk.columns:
            stamps = chunk["timestamp"].astype(str)
            if previous is not None and stamps.iloc[0] < previous:
                timestamps_sorted = False
            if not stamps.is_monotonic_increasing:
                timestamps_sorted = False
            previous = stamps.iloc[-1]
        rows += len(chunk)

    head_total = max(1, sum(head_labels.values()))
    all_total = max(1, sum(all_labels.values()))
    drift = {
        name: round(head_labels.get(name, 0) / head_total
                    - count / all_total, 6)
        for name, count in all_labels.most_common()
    }
    return {
        "file": path.name,
        "rows": rows,
        "timestamps_monotonic": timestamps_sorted,
        "labels_full": dict(all_labels.most_common()),
        "head_share_minus_full_share": drift,
        "max_abs_drift": max((abs(v) for v in drift.values()), default=0.0),
    }


def simulate_current_sampler(
    profiles: list[dict[str, Any]], per_file_target: int
) -> dict[str, Any]:
    """Replay the current per-file cap to expose between-file distortion.

    Attack rows from each file are capped at the same absolute number, so a file
    holding ten million flows contributes no more than one holding a hundred
    thousand. The realised family share is compared with the true share.
    """
    true_attacks: Counter[str] = Counter()
    capped_attacks: Counter[str] = Counter()
    for profile in profiles:
        attacks = {
            name: count
            for name, count in profile["labels_full"].items()
            if name.strip().casefold() != "benign"
        }
        file_attack_rows = sum(attacks.values())
        true_attacks.update(attacks)
        if file_attack_rows == 0:
            continue
        # The cap applies to the file's pooled attack rows; families inside one
        # file keep their relative share.
        retained = min(file_attack_rows, per_file_target)
        scale = retained / file_attack_rows
        for name, count in attacks.items():
            capped_attacks[name] += count * scale

    true_total = max(1.0, sum(true_attacks.values()))
    capped_total = max(1.0, sum(capped_attacks.values()))
    comparison = {}
    for name in sorted(set(true_attacks) | set(capped_attacks)):
        true_share = true_attacks.get(name, 0) / true_total
        capped_share = capped_attacks.get(name, 0.0) / capped_total
        comparison[name] = {
            "true_share": round(true_share, 6),
            "sampler_share": round(capped_share, 6),
            "absolute_shift_pp": round((capped_share - true_share) * 100, 4),
            "relative_factor": round(capped_share / true_share, 3)
            if true_share > 0 else None,
        }
    return {
        "per_file_target": per_file_target,
        "families": comparison,
        "max_abs_shift_pp": max(
            (abs(v["absolute_shift_pp"]) for v in comparison.values()),
            default=0.0,
        ),
    }


def inclusion_curve(per_file_target: int, rows: int) -> list[dict[str, float]]:
    """Inclusion probability per chunk under the current decaying fraction."""
    curve = []
    seen = 0
    while seen < rows:
        seen += CHUNK_ROWS
        curve.append({
            "chunk": len(curve) + 1,
            "rows_seen": min(seen, rows),
            "inclusion_probability": round(
                min(1.0, per_file_target / max(1, seen)), 6
            ),
        })
        if len(curve) >= 12:
            break
    return curve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day", required=True, help="capture day directory")
    parser.add_argument("--target-rows", type=int, default=250_000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    day = Path(args.day)
    files = sorted(day.glob("*.csv"))
    per_file_target = max(args.target_rows // len(files) * 3, 20_000)

    profiles = [file_profile(path, CHUNK_ROWS) for path in files]
    distortion = simulate_current_sampler(profiles, per_file_target)
    biggest = max(profiles, key=lambda p: p["rows"])

    report = {
        "day": day.name,
        "files": len(files),
        "per_file_target": per_file_target,
        "total_rows": sum(p["rows"] for p in profiles),
        "row_span": {
            "smallest_file": min(p["rows"] for p in profiles),
            "largest_file": max(p["rows"] for p in profiles),
        },
        "inclusion_curve_largest_file": inclusion_curve(
            per_file_target, biggest["rows"]
        ),
        "profiles": profiles,
        "family_distortion": distortion,
        "any_file_unordered": any(
            not p["timestamps_monotonic"] for p in profiles
        ),
        "max_head_drift": max(p["max_abs_drift"] for p in profiles),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "day": report["day"],
        "files": report["files"],
        "per_file_target": per_file_target,
        "total_rows": report["total_rows"],
        "smallest_file_rows": report["row_span"]["smallest_file"],
        "largest_file_rows": report["row_span"]["largest_file"],
        "all_files_time_ordered": not report["any_file_unordered"],
        "max_head_vs_full_label_drift": round(report["max_head_drift"], 5),
        "max_family_share_shift_pp": distortion["max_abs_shift_pp"],
        "inclusion_first_chunk": report["inclusion_curve_largest_file"][0][
            "inclusion_probability"
        ],
        "inclusion_tenth_chunk": (
            report["inclusion_curve_largest_file"][9]["inclusion_probability"]
            if len(report["inclusion_curve_largest_file"]) > 9 else None
        ),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
