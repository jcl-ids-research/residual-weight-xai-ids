"""Build a reproducible, leakage-free sample cache for CIC-DDoS2019.

The raw capture is roughly 70 million flows across 30 GB, so each experiment
seed cannot re-read it. A fixed sample is cached once instead.

Sampling design
---------------
An earlier version kept rows with probability ``per_file_target / rows_seen``
and capped every file at the same absolute count. Both choices were biased, and
the bias was measured rather than assumed:

  * the decaying fraction gave rows in the first chunk about ten times the
    inclusion probability of rows in the tenth chunk (0.267855 against
    0.026786), and a file head's label mix differed from the whole file by up
    to 41.6 percentage points;
  * a uniform cap over files spanning 191 694 to 5 775 786 rows shifted an
    attack family's share by up to 14.1 percentage points.

This version performs proportional stratified sampling over (file, label)
cells. A first pass counts every label in every file exactly. Quotas are then
allocated in proportion to those counts using the largest-remainder method, and
a second pass draws each cell's quota uniformly without replacement by choosing
positions over the cell's known population before streaming. Binary prevalence,
per-family composition, and each file's share are preserved by construction
rather than repaired afterwards.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

# Directory names do not match the recorded dates: the Timestamp column shows
# CSV-03-11 is 3 November 2018 and CSV-01-12 is 1 December 2018. Training uses
# the earlier day so the held-out day is genuinely later.
DAY_ONE: Final = "CSV-03-11"
DAY_TWO: Final = "CSV-01-12"
DAY_LABELS: Final = {"CSV-03-11": "2018-11-03", "CSV-01-12": "2018-12-01"}
CAP_SEED: Final = 20260902
CHUNK_ROWS: Final = 400_000

# Identity and bookkeeping fields. Retaining source/destination addresses or the
# flow identifier would let the model memorise attack provenance instead of
# learning traffic behaviour.
DROP_COLUMNS: Final = frozenset(
    {
        "unnamed: 0",
        "flow id",
        "source ip",
        "source port",
        "destination ip",
        "timestamp",
        "simillarhttp",
    }
)


def usable_columns(path: Path) -> list[str]:
    """Return retained column names exactly as spelled in the file.

    CIC capture files carry leading spaces in most headers, so the original
    spelling must be preserved for ``usecols`` while comparison uses the
    normalised form.
    """
    header = pd.read_csv(path, nrows=0, low_memory=False)
    return [
        str(column)
        for column in header.columns
        if str(column).strip()
        and str(column).strip().casefold() not in DROP_COLUMNS
    ]


def count_file(filename: str) -> tuple[str, dict[str, int], list[str]]:
    """First pass: exact per-label row counts for one capture file.

    Headers in these captures carry leading spaces, so the label column is
    selected by normalised comparison rather than by literal name.
    """
    path = Path(filename)
    keep = usable_columns(path)
    counts: Counter[str] = Counter()
    for chunk in pd.read_csv(
        path,
        usecols=lambda column: str(column).strip().casefold() == "label",
        chunksize=CHUNK_ROWS,
        low_memory=False,
    ):
        chunk.columns = [str(column).strip() for column in chunk.columns]
        counts.update(chunk["Label"].astype(str).str.strip().tolist())
    return path.name, dict(counts), [column.strip() for column in keep]


def allocate(
    counts: dict[tuple[str, str], int], target: int
) -> dict[tuple[str, str], int]:
    """Split ``target`` across cells in proportion to their true sizes.

    Largest-remainder allocation makes the quotas sum to ``target`` exactly.
    Proportional shares never exceed a cell's population, so no cell is asked
    for more rows than it holds.
    """
    cells = sorted(counts)
    sizes = np.array([counts[cell] for cell in cells], dtype=np.float64)
    total = float(sizes.sum())
    if total <= target:
        return {cell: counts[cell] for cell in cells}
    exact = sizes * (target / total)
    quota = np.floor(exact).astype(np.int64)
    deficit = int(target - int(quota.sum()))
    if deficit > 0:
        for position in np.argsort(-(exact - quota))[:deficit]:
            quota[position] += 1
    return {
        cell: int(min(int(quota[index]), counts[cell]))
        for index, cell in enumerate(cells)
    }


def draw_file(
    job: tuple[str, dict[str, tuple[int, int]], int]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Second pass: take each label's quota uniformly from one file.

    ``job`` carries, per label, the cell population and the quota. Positions are
    drawn over the population before reading, so every row of a label has the
    same inclusion probability regardless of where it sits in the file.
    """
    filename, plan, seed = job
    path = Path(filename)
    keep = usable_columns(path)
    rng = np.random.default_rng(seed)

    wanted = {
        label: np.sort(rng.choice(population, size=quota, replace=False))
        for label, (population, quota) in plan.items()
        if quota > 0
    }
    cursor: dict[str, int] = {label: 0 for label in wanted}
    seen: Counter[str] = Counter()
    taken: list[pd.DataFrame] = []

    for chunk in pd.read_csv(
        path, usecols=keep, chunksize=CHUNK_ROWS, low_memory=False
    ):
        chunk.columns = [str(column).strip() for column in chunk.columns]
        labels = chunk["Label"].astype(str).str.strip()
        mask = np.zeros(len(chunk), dtype=bool)
        for label, positions in wanted.items():
            if cursor[label] >= len(positions):
                continue
            rows = np.flatnonzero((labels == label).to_numpy())
            if not len(rows):
                continue
            offsets = seen[label] + np.arange(len(rows))
            hit = np.isin(offsets, positions[cursor[label]:], assume_unique=True)
            if hit.any():
                mask[rows[hit]] = True
                cursor[label] += int(hit.sum())
        for label, value in labels.value_counts().items():
            seen[str(label)] += int(value)
        if mask.any():
            taken.append(chunk.loc[mask])

    columns = [column.strip() for column in keep]
    frame = (
        pd.concat(taken, ignore_index=True, copy=False)
        if taken
        else pd.DataFrame(columns=columns)
    )
    realised = (
        {
            str(name): int(value)
            for name, value in frame["Label"]
            .astype(str)
            .str.strip()
            .value_counts()
            .items()
        }
        if len(frame)
        else {}
    )
    print(f"  drew {path.name}: kept={len(frame)}", flush=True)
    return frame, realised


def read_day(
    directory: Path, target_rows: int, seed: int, workers: int
) -> tuple[pd.DataFrame, pd.Series, dict[str, int], int, dict]:
    """Sample one capture day with proportional (file, label) quotas."""
    files = sorted(directory.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files under {directory}")

    with Pool(processes=min(workers, len(files))) as pool:
        counted = pool.map(count_file, [str(path) for path in files])

    reference: list[str] | None = None
    cells: dict[tuple[str, str], int] = {}
    label_counts: Counter[str] = Counter()
    for name, counts, columns in counted:
        if reference is None:
            reference = columns
        elif columns != reference:
            missing = sorted(set(reference) - set(columns))
            extra = sorted(set(columns) - set(reference))
            raise ValueError(
                f"Schema mismatch in {name}: missing={missing}, extra={extra}"
            )
        for label, count in counts.items():
            cells[(name, label)] = count
            label_counts[label] += count

    quotas = allocate(cells, target_rows)
    jobs: list[tuple[str, dict[str, tuple[int, int]], int]] = []
    for offset, path in enumerate(files):
        plan = {
            label: (cells[(name, label)], quotas[(name, label)])
            for (name, label) in cells
            if name == path.name and quotas[(name, label)] > 0
        }
        jobs.append((str(path), plan, seed + offset))

    with Pool(processes=min(workers, len(jobs))) as pool:
        drawn = pool.map(draw_file, jobs)

    realised: Counter[str] = Counter()
    per_file_rate: dict[str, float] = {}
    frames: list[pd.DataFrame] = []
    for (frame, counts), path in zip(drawn, files):
        realised.update(counts)
        file_rows = sum(
            value for (name, _), value in cells.items() if name == path.name
        )
        per_file_rate[path.name] = round(len(frame) / max(1, file_rows), 8)
        if len(frame):
            frames.append(frame)

    frame = pd.concat(frames, ignore_index=True, copy=False)
    labels = frame.pop("Label").astype(str).str.strip()
    total_rows = sum(cells.values())

    full_total = max(1, total_rows)
    drawn_total = max(1, sum(realised.values()))
    fidelity = {
        label: {
            "full_share": round(count / full_total, 8),
            "sampled_share": round(realised.get(label, 0) / drawn_total, 8),
            "shift_pp": round(
                (realised.get(label, 0) / drawn_total - count / full_total) * 100,
                6,
            ),
        }
        for label, count in label_counts.most_common()
    }
    extra = {
        "sampling_method": (
            "proportional stratified over (file, label) cells; "
            "largest-remainder quotas; uniform draw within each cell"
        ),
        "per_file_sampling_rate": per_file_rate,
        "family_fidelity": fidelity,
        "max_family_shift_pp": max(
            (abs(item["shift_pp"]) for item in fidelity.values()), default=0.0
        ),
        "sampled_label_counts": dict(realised.most_common()),
    }
    return frame, labels, dict(label_counts), total_rows, extra


def build(
    source: Path, output: Path, train_rows: int, test_rows: int, workers: int
) -> dict:
    """Create the cache for both capture days and return the audit record."""
    audit: dict[str, object] = {"cap_seed": CAP_SEED}
    arrays: dict[str, np.ndarray] = {}
    for tag, day, target in (
        ("train", DAY_ONE, train_rows),
        ("test", DAY_TWO, test_rows),
    ):
        print(f"[{tag}] scanning {day} with {workers} workers", flush=True)
        frame, labels, counts, total, extra = read_day(
            source / day, target, CAP_SEED, workers
        )
        binary = (~labels.str.casefold().eq("benign")).to_numpy().astype(np.int64)
        order = np.random.default_rng(CAP_SEED).permutation(len(binary))
        frame = frame.iloc[order].reset_index(drop=True)
        binary = binary[order]
        features = frame.apply(pd.to_numeric, errors="coerce").astype(np.float32)
        arrays[f"X_{tag}"] = features.to_numpy()
        arrays[f"y_{tag}"] = binary
        arrays[f"columns_{tag}"] = np.asarray(
            features.columns.tolist(), dtype=object
        )
        audit[tag] = {
            "day": day,
            "capture_date": DAY_LABELS.get(day, day),
            "full_period_rows": total,
            "full_label_counts": counts,
            "sampled_rows": int(len(binary)),
            "sampled_attack_rate": float(binary.mean()),
            "full_attack_rate": float(
                1.0 - counts.get("BENIGN", 0) / max(1, sum(counts.values()))
            ),
            "attack_labels": sorted(
                name for name in counts if name.strip().casefold() != "benign"
            ),
            **extra,
        }
        print(
            f"[{tag}] rows={len(binary)} attack_rate={binary.mean():.6f} "
            f"max_family_shift={extra['max_family_shift_pp']:.4f}pp "
            f"features={features.shape[1]}",
            flush=True,
        )

    train_columns = list(arrays["columns_train"])
    test_columns = list(arrays["columns_test"])
    if train_columns != test_columns:
        raise ValueError("Training and test feature schemas differ")
    audit["features"] = train_columns
    audit["feature_count"] = len(train_columns)
    train_attacks = set(audit["train"]["attack_labels"])  # type: ignore[index]
    test_attacks = set(audit["test"]["attack_labels"])  # type: ignore[index]
    audit["test_only_attack_labels"] = sorted(test_attacks - train_attacks)
    audit["dropped_identity_columns"] = sorted(DROP_COLUMNS)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        X_train=arrays["X_train"],
        y_train=arrays["y_train"],
        X_test=arrays["X_test"],
        y_test=arrays["y_test"],
        columns=np.asarray(train_columns, dtype=object),
    )
    output.with_suffix(".audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return audit


def main() -> None:
    """Entry point for cache construction."""
    parser = argparse.ArgumentParser(description="Prepare CIC-DDoS2019 cache")
    parser.add_argument("--source", default="/opt/CIC-DDoS2019")
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-rows", type=int, default=250_000)
    parser.add_argument("--test-rows", type=int, default=250_000)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    audit = build(
        Path(args.source),
        Path(args.out),
        args.train_rows,
        args.test_rows,
        args.workers,
    )
    print(
        json.dumps(
            {
                "feature_count": audit["feature_count"],
                "test_only_attack_labels": audit["test_only_attack_labels"],
                "train_max_family_shift_pp": audit["train"]["max_family_shift_pp"],
                "test_max_family_shift_pp": audit["test"]["max_family_shift_pp"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
