"""Close the CIC-DDoS2019 evidence chain.

The capture-date decision and the reported feature dimension were taken from
console output that left no reviewable artefact. This script re-derives both
from the raw files, records the server paths and SHA-256 digests required by the
project rules, and writes one JSON that a reader can check independently.

Run on the host that owns the data:

    python3 collect_ddos2019_evidence.py \
        --source /opt/CIC-DDoS2019 \
        --results /opt/ids_revision/v4_deep_baseline/ddos_results \
        --cache /opt/ids_revision/v4_deep_baseline/cache/cicddos2019.npz \
        --out /opt/ids_revision/v4_deep_baseline/aggregate/ddos2019_evidence.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import socket
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Final

DAYS: Final = ("CSV-03-11", "CSV-01-12")
SEEDS: Final = (42, 123, 456)
SAMPLE_ROWS: Final = 20000
DATE_FORMATS: Final = (
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_raw_timestamps(path: Path, limit: int = SAMPLE_ROWS) -> dict:
    """Collect verbatim timestamp strings without any date parsing.

    The raw text is reported alongside any interpretation so that a reader can
    judge the day/month order instead of trusting an inferred value.
    """
    with path.open("r", encoding="utf-8", errors="replace", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration:
            return {"file": path.name, "error": "empty file"}
        index = next(
            (
                position
                for position, name in enumerate(header)
                if name.strip().casefold() == "timestamp"
            ),
            None,
        )
        if index is None:
            return {"file": path.name, "error": "no Timestamp column"}
        samples: list[str] = []
        prefixes: Counter[str] = Counter()
        for position, row in enumerate(reader):
            if position >= limit:
                break
            if index >= len(row):
                continue
            value = row[index].strip()
            if not value:
                continue
            prefixes[value[:10]] += 1
            if len(samples) < 5:
                samples.append(value)
    return {
        "file": path.name,
        "raw_samples": samples,
        "distinct_date_prefixes": prefixes.most_common(6),
        "rows_inspected": min(limit, sum(prefixes.values())),
    }


def interpret(sample: str) -> dict:
    """Report every format that parses a timestamp, without picking one."""
    matches = {}
    for pattern in DATE_FORMATS:
        try:
            matches[pattern] = datetime.strptime(sample, pattern).date().isoformat()
        except ValueError:
            continue
    return matches


def collect_day(source: Path, day: str) -> dict:
    """Gather raw timestamp evidence for one capture day."""
    directory = source / day
    files = sorted(directory.glob("*.csv"))
    evidence = [read_raw_timestamps(path) for path in files]
    first = next(
        (item["raw_samples"][0] for item in evidence if item.get("raw_samples")),
        None,
    )
    return {
        "day": day,
        "server_path": str(directory),
        "files": [item["file"] for item in evidence],
        "timestamp_evidence": evidence,
        "example_raw_timestamp": first,
        "parses_as": interpret(first) if first else {},
    }


def collect_runs(results: Path) -> list[dict]:
    """Record dimensions and digests for the three CIC-DDoS2019 runs."""
    records: list[dict] = []
    for seed in SEEDS:
        path = results / "cicddos2019" / f"seed{seed}" / "metrics.json"
        if not path.exists():
            records.append({"seed": seed, "error": f"missing {path}"})
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        sizes = payload.get("sizes", {})
        records.append(
            {
                "seed": seed,
                "server_path": str(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "original_features": sizes.get("original_features"),
                "classifier_features_after_onehot": sizes.get(
                    "classifier_features_after_onehot"
                ),
                "sub_train": sizes.get("sub_train"),
                "validation": sizes.get("validation"),
                "official_test": sizes.get("official_test"),
            }
        )
    return records


def main() -> None:
    """Write the CIC-DDoS2019 evidence record."""
    parser = argparse.ArgumentParser(description="Collect CIC-DDoS2019 evidence")
    parser.add_argument("--source", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    source = Path(args.source)
    cache = Path(args.cache)
    runs = collect_runs(Path(args.results))
    dimensions = {
        record.get("classifier_features_after_onehot")
        for record in runs
        if "error" not in record
    }
    report = {
        "collected_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "capture_days": [collect_day(source, day) for day in DAYS],
        "cache": {
            "server_path": str(cache),
            "sha256": sha256(cache) if cache.exists() else None,
            "bytes": cache.stat().st_size if cache.exists() else None,
        },
        "runs": runs,
        "feature_dimension_consistent": len(dimensions) == 1,
        "feature_dimension_values": sorted(x for x in dimensions if x is not None),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for day in report["capture_days"]:
        print(f"{day['day']}: raw={day['example_raw_timestamp']}")
        for pattern, value in day["parses_as"].items():
            print(f"    {pattern} -> {value}")
    print("feature dims:", report["feature_dimension_values"])


if __name__ == "__main__":
    main()
