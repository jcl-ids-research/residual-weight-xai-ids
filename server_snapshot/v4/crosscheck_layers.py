#!/usr/bin/env python3
"""Cross-source consistency check for the V2 manuscript.

Three real defects in this manuscript survived every existing check because
each one lived *between* evidence layers rather than inside one of them:

  * CIC-DDoS2019 cache kept 80 columns while the run used 68 (12 silent drops)
  * CIC-IDS-2017 raw had 79 columns while the run used 70 (8 silent drops)
  * NSL-KDD raw had 41 features while the run used 40 (1 silent drop)

Every per-layer audit passed, because each layer was self-consistent. The
manuscript numbers were also "correct" in isolation. What was missing was a
check that every column disappearing between layers is *accounted for*.

This script walks raw source -> cache -> run -> manuscript and classifies each
vanished column as target, identifier, or constant. A constant column is only
accepted after its variance is measured on the real data; nothing is assumed.
Anything left over lands in ``unexplained``, which must stay empty.

Modes
-----
``--mode server``      collect raw/cache/run facts on the machine holding the
                       data and emit a JSON fact sheet.
``--mode manuscript``  compare a downloaded fact sheet against the DOCX.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import re
import socket
from datetime import datetime, timezone
from typing import Any, Iterable

CHUNK_ROWS = 200_000

# NSL-KDD ships without a header row; these are the canonical field names.
NSLKDD_FEATURES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

# Columns that are legitimately absent from the feature matrix: prediction
# targets and row/host identifiers. Anything else must prove it is constant.
# A proportional sample must reproduce the population mix. The earlier sampler
# shifted an attack family's share by up to 31 percentage points while every
# per-layer audit still passed, so fidelity is asserted numerically here.
FAMILY_SHIFT_TOLERANCE_PP = 0.01
# Under proportional allocation every file is drawn at the same rate; the old
# uniform per-file cap made small files 15x over-represented.
FILE_RATE_TOLERANCE = 0.02

TARGET_COLUMNS = {"label", "attack_cat", "class", "difficulty"}
IDENTIFIER_COLUMNS = {
    "id", "flow id", "source ip", "destination ip", "source port",
    "timestamp", "unnamed: 0", "simillarhttp",
}

DATASETS: dict[str, dict[str, Any]] = {
    "unsw": {
        "raw_globs": ["/opt/UNSW-NB15/UNSW_NB15_training-set.csv"],
        "header": True,
        "manuscript_name": "UNSW-NB15",
    },
    "nslkdd": {
        "raw_globs": ["/opt/NSL-KDD/KDDTrain+.txt"],
        "header": False,
        "column_names": NSLKDD_FEATURES + ["label", "difficulty"],
        "manuscript_name": "NSL-KDD",
    },
    "cicids2017": {
        "raw_globs": ["/opt/CIC-IDS-2017/MachineLearningCVE/*.csv"],
        "header": True,
        "manuscript_name": "CIC-IDS-2017",
    },
    "cicddos2019": {
        "raw_globs": ["/opt/CIC-DDoS2019/CSV-03-11/*.csv",
                      "/opt/CIC-DDoS2019/CSV-01-12/*.csv"],
        "header": True,
        "cache_audit": "/opt/ids_revision/v4_deep_baseline/cache/"
                       "cicddos2019.audit.json",
        "manuscript_name": "CIC-DDoS2019",
    },
}


def normalise(name: str) -> str:
    """Strip BOM/whitespace and case so column names compare reliably."""
    return name.replace("\ufeff", "").strip().lower()


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_files(globs: Iterable[str]) -> list[str]:
    found: list[str] = []
    for pattern in globs:
        found.extend(sorted(glob.glob(pattern)))
    return found


def raw_columns(path: str, spec: dict[str, Any]) -> list[str]:
    if not spec["header"]:
        return [normalise(c) for c in spec["column_names"]]
    with open(path, newline="", encoding="utf-8", errors="replace") as handle:
        return [normalise(c) for c in next(csv.reader(handle))]


def measure_candidates(
    files: list[str], spec: dict[str, Any], candidates: list[str]
) -> dict[str, dict[str, Any]]:
    """Measure distinct values of ``candidates`` across the full raw data.

    Reads in chunks and stops tracking a column as soon as it shows a second
    distinct value, so proving non-constancy costs little. A column is only
    reported constant after every row has been read.
    """
    import pandas as pd

    if not candidates:
        return {}
    seen: dict[str, set[Any]] = {c: set() for c in candidates}
    settled: set[str] = set()
    rows = 0
    for path in files:
        kwargs: dict[str, Any] = {"chunksize": CHUNK_ROWS, "low_memory": False}
        if spec["header"]:
            kwargs["usecols"] = lambda c: normalise(c) in seen  # noqa: B023
        else:
            kwargs["header"] = None
            kwargs["names"] = [normalise(c) for c in spec["column_names"]]
            kwargs["usecols"] = [c for c in candidates]
        for chunk in pd.read_csv(path, **kwargs):
            chunk.columns = [normalise(c) for c in chunk.columns]
            rows += len(chunk)
            for column in list(seen):
                if column in settled or column not in chunk.columns:
                    continue
                seen[column].update(chunk[column].unique().tolist())
                if len(seen[column]) > 1:
                    settled.add(column)
    return {
        column: {
            "distinct_values_observed": len(values),
            "is_constant": len(values) <= 1,
            "value": (sorted(values, key=repr)[:1] or [None])[0]
            if len(values) <= 1
            else None,
            "rows_scanned": rows,
        }
        for column, values in seen.items()
    }


def run_facts(dataset: str) -> dict[str, Any]:
    """Collect per-seed feature and size facts from the completed runs."""
    patterns = [
        f"/opt/ids_revision/v3_strengthened/results/{dataset}/seed*/metrics.json",
        f"/opt/ids_revision/v4_deep_baseline/ddos_results/{dataset}/seed*/metrics.json",
    ]
    files = resolve_files(patterns)
    seeds: list[dict[str, Any]] = []
    feature_names: list[str] = []
    for path in files:
        with open(path, encoding="utf-8") as handle:
            metrics = json.load(handle)
        sizes = metrics.get("sizes", {})
        names = [
            normalise(c)
            for c in metrics.get("preprocessing", {}).get(
                "original_feature_names", []
            )
        ]
        feature_names = feature_names or names
        seeds.append({
            "seed": metrics.get("seed"),
            "server_path": path,
            "sha256": sha256_of(path),
            "original_features": sizes.get("original_features"),
            "classifier_features_after_onehot": sizes.get(
                "classifier_features_after_onehot"
            ),
            "official_test_set_untouched": metrics.get("protocol", {}).get(
                "official_test_set_untouched"
            ),
        })
    dims = {s["original_features"] for s in seeds if s["original_features"]}
    encoded = {
        s["classifier_features_after_onehot"]
        for s in seeds
        if s["classifier_features_after_onehot"]
    }
    return {
        "seed_count": len(seeds),
        "seeds": seeds,
        "feature_names": feature_names,
        "original_features": sorted(dims),
        "classifier_features_after_onehot": sorted(encoded),
        "dimension_consistent": len(dims) <= 1,
    }


def audit_dataset(dataset: str, spec: dict[str, Any]) -> dict[str, Any]:
    files = resolve_files(spec["raw_globs"])
    if not files:
        return {"error": "no raw files matched", "globs": spec["raw_globs"]}
    columns = raw_columns(files[0], spec)
    runs = run_facts(dataset)
    used = set(runs["feature_names"])

    vanished = [c for c in columns if c not in used]
    targets = [c for c in vanished if c in TARGET_COLUMNS]
    identifiers = [c for c in vanished if c in IDENTIFIER_COLUMNS]
    remainder = [c for c in vanished if c not in targets and c not in identifiers]

    measured = measure_candidates(files, spec, remainder)
    constants = [c for c in remainder if measured.get(c, {}).get("is_constant")]
    unexplained = [c for c in remainder if c not in constants]

    cache: dict[str, Any] | None = None
    cache_path = spec.get("cache_audit")
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        cache = {
            "server_path": cache_path,
            "sha256": sha256_of(cache_path),
            "feature_count": payload.get("feature_count"),
            "dropped_identity_columns": payload.get("dropped_identity_columns"),
            "train_capture_date": payload.get("train", {}).get("capture_date"),
            "test_capture_date": payload.get("test", {}).get("capture_date"),
        }

    return {
        "raw_files": len(files),
        "raw_example": files[0],
        "raw_column_count": len(columns),
        "cache": cache,
        "run": runs,
        "accounting": {
            "raw_columns": len(columns),
            "target_columns": targets,
            "identifier_columns": identifiers,
            "constant_columns": constants,
            "constant_evidence": measured,
            "unexplained": unexplained,
            "features_used": len(used),
            "balances": len(columns)
            - len(targets)
            - len(identifiers)
            - len(constants)
            == len(used),
        },
    }


def audit_sampling(cache_audit_path: str) -> dict[str, Any]:
    """Check that the cached sample reproduces the capture period it claims.

    Column accounting alone cannot see a biased sampler: the earlier build kept
    all 68 columns yet drew rows so unevenly that one attack family's share moved
    by 31 percentage points. Rows, per-file rates, and family shares are
    therefore reconciled explicitly.
    """
    if not os.path.exists(cache_audit_path):
        return {"error": "cache audit not found", "path": cache_audit_path}
    with open(cache_audit_path, encoding="utf-8") as handle:
        audit = json.load(handle)

    periods: dict[str, Any] = {}
    problems: list[str] = []
    for tag in ("train", "test"):
        period = audit.get(tag)
        if not isinstance(period, dict):
            continue
        fidelity = period.get("family_fidelity") or {}
        worst = max(
            (abs(item["shift_pp"]) for item in fidelity.values()), default=None
        )
        rates = list((period.get("per_file_sampling_rate") or {}).values())
        rate_spread = (max(rates) - min(rates)) if rates else None
        sampled = period.get("sampled_rows")
        full = period.get("full_period_rows")
        prevalence_gap = None
        if period.get("sampled_attack_rate") is not None:
            prevalence_gap = abs(
                period["sampled_attack_rate"] - period["full_attack_rate"]
            ) * 100

        periods[tag] = {
            "day": period.get("day"),
            "capture_date": period.get("capture_date"),
            "full_period_rows": full,
            "sampled_rows": sampled,
            "sampling_method": period.get("sampling_method"),
            "max_family_shift_pp": worst,
            "per_file_rate_spread": rate_spread,
            "prevalence_gap_pp": prevalence_gap,
            "families_tracked": len(fidelity),
        }

        if not fidelity:
            problems.append(f"{tag}: no per-family fidelity recorded")
        elif worst is not None and worst > FAMILY_SHIFT_TOLERANCE_PP:
            problems.append(
                f"{tag}: attack family share shifted {worst:.4f} pp "
                f"(limit {FAMILY_SHIFT_TOLERANCE_PP} pp)"
            )
        if rate_spread is not None and rate_spread > FILE_RATE_TOLERANCE:
            problems.append(
                f"{tag}: per-file sampling rates span {rate_spread:.4f} "
                f"(limit {FILE_RATE_TOLERANCE})"
            )
        if prevalence_gap is not None and prevalence_gap > FAMILY_SHIFT_TOLERANCE_PP:
            problems.append(
                f"{tag}: binary prevalence differs by {prevalence_gap:.4f} pp"
            )
        if not period.get("sampling_method"):
            problems.append(f"{tag}: sampling method not documented")

    return {
        "server_path": cache_audit_path,
        "sha256": sha256_of(cache_audit_path),
        "periods": periods,
        "problems": problems,
        "status": "PASS" if not problems else "FAIL",
    }


def collect(output: str) -> dict[str, Any]:
    facts = {
        "host": socket.gethostname(),
        "collected_at": datetime.now(timezone.utc).astimezone().isoformat(
            timespec="seconds"
        ),
        "datasets": {
            name: audit_dataset(name, spec) for name, spec in DATASETS.items()
        },
        "sampling": {
            name: audit_sampling(spec["cache_audit"])
            for name, spec in DATASETS.items()
            if spec.get("cache_audit")
        },
    }
    problems: list[str] = []
    for name, data in facts["datasets"].items():
        if "error" in data:
            problems.append(f"{name}: {data['error']}")
            continue
        acc = data["accounting"]
        if acc["unexplained"]:
            problems.append(f"{name}: unexplained columns {acc['unexplained']}")
        if not acc["balances"]:
            problems.append(f"{name}: column accounting does not balance")
        if not data["run"]["dimension_consistent"]:
            problems.append(f"{name}: feature dimension differs across seeds")
    for name, data in facts["sampling"].items():
        if "error" in data:
            problems.append(f"{name} sampling: {data['error']}")
            continue
        problems.extend(f"{name} sampling: {item}" for item in data["problems"])
    facts["problems"] = problems
    facts["status"] = "PASS" if not problems else "FAIL"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(facts, handle, ensure_ascii=False, indent=2)
    return facts


def check_manuscript(facts_path: str, docx_path: str) -> dict[str, Any]:
    """Assert the manuscript matches the measured facts.

    Covers Table 1's feature column and the in-text constant-drop counts, which
    is where the three historical defects surfaced.
    """
    from docx import Document

    with open(facts_path, encoding="utf-8-sig") as handle:
        facts = json.load(handle)
    document = Document(docx_path)
    body = "\n".join(p.text for p in document.paragraphs)

    table_specs: dict[str, str] = {}
    for table in document.tables:
        header = [c.text.strip() for c in table.rows[0].cells]
        if not any("保留字段" in h for h in header):
            continue
        column = next(i for i, h in enumerate(header) if "保留字段" in h)
        for row in table.rows[1:]:
            cells = [c.text.strip() for c in row.cells]
            table_specs[cells[0]] = cells[column]

    findings: list[str] = []
    for name, data in facts["datasets"].items():
        if "error" in data:
            continue
        label = DATASETS[name]["manuscript_name"]
        spec = next(
            (v for k, v in table_specs.items() if label in k), None
        )
        run = data["run"]
        if spec is None:
            findings.append(f"{label}: no Table 1 row found")
            continue
        numbers = [int(n) for n in re.findall(r"\d+", spec)]
        if run["original_features"] and run["original_features"][0] not in numbers:
            findings.append(
                f"{label}: Table 1 shows '{spec}' but runs report "
                f"{run['original_features'][0]} retained fields"
            )
        dropped = len(data["accounting"]["constant_columns"])
        claim = f"{label}删除{dropped}项"
        none_claim = f"{label}无此类字段"
        if dropped and claim not in body:
            findings.append(f"{label}: text must state '{claim}'")
        if not dropped and none_claim not in body:
            findings.append(f"{label}: text must state '{none_claim}'")

    return {
        "facts_source": facts_path,
        "facts_status": facts.get("status"),
        "docx_sha256": sha256_of(docx_path),
        "table1_specs": table_specs,
        "findings": findings,
        "status": "PASS"
        if not findings and facts.get("status") == "PASS"
        else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("server", "manuscript"), required=True)
    parser.add_argument("--out", default="/opt/ids_revision/v4_deep_baseline/"
                                         "aggregate/crosscheck_layers.json")
    parser.add_argument("--facts")
    parser.add_argument("--docx")
    args = parser.parse_args()

    if args.mode == "server":
        result = collect(args.out)
        summary = {
            "status": result["status"],
            "problems": result["problems"],
            "datasets": {
                name: {
                    "raw_cols": d.get("raw_column_count"),
                    "features": d.get("run", {}).get("original_features"),
                    "targets": len(d.get("accounting", {}).get("target_columns", [])),
                    "identifiers": len(
                        d.get("accounting", {}).get("identifier_columns", [])
                    ),
                    "constants": len(
                        d.get("accounting", {}).get("constant_columns", [])
                    ),
                    "unexplained": d.get("accounting", {}).get("unexplained", []),
                    "balances": d.get("accounting", {}).get("balances"),
                }
                for name, d in result["datasets"].items()
            },
            "sampling": {
                name: {
                    "status": s.get("status", "ERROR"),
                    "max_family_shift_pp": {
                        tag: p.get("max_family_shift_pp")
                        for tag, p in s.get("periods", {}).items()
                    },
                    "per_file_rate_spread": {
                        tag: p.get("per_file_rate_spread")
                        for tag, p in s.get("periods", {}).items()
                    },
                    "problems": s.get("problems", []),
                }
                for name, s in result["sampling"].items()
            },
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        raise SystemExit(0 if result["status"] == "PASS" else 1)

    if not args.facts or not args.docx:
        parser.error("--mode manuscript requires --facts and --docx")
    result = check_manuscript(args.facts, args.docx)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
