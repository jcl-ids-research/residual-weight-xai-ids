"""Quantify duplicate compensation from the verified V3 runs.

For every dataset and paired seed this reads the per-class effective weighted
class mass under the original-weight and residual-weight paths. The manuscript
derives that the residual path holds this mass constant at N'/K while the
original path scales it by the per-class expansion factor; this script measures
both directly so the claim rests on evidence rather than derivation alone.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Final

ORIGINAL: Final = "smotenc_tomek_weight"
RESIDUAL: Final = "smotenc_tomek_residual_weight"
SEEDS: Final = {
    "unsw": (42, 123, 456, 789, 1001, 2024, 31415, 65537, 77777, 99991),
    "nslkdd": (42, 123, 456, 789, 1001, 2024, 31415, 65537, 77777, 99991),
    "cicids2017": (42, 123, 456),
}


def summarise(values: list[float]) -> dict[str, float]:
    """Return the mean and sample standard deviation of a paired-seed series."""
    return {
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def mass_statistics(mass: dict[str, float]) -> dict[str, float]:
    """Describe how unevenly total class weight is distributed."""
    values = [float(item) for item in mass.values()]
    lowest = min(values)
    highest = max(values)
    mean = statistics.mean(values)
    deviation = statistics.pstdev(values)
    return {
        "minimum": lowest,
        "maximum": highest,
        "ratio": highest / lowest if lowest > 0 else float("inf"),
        "relative_spread": deviation / mean if mean > 0 else 0.0,
    }


def collect(results: Path) -> dict:
    """Read every run and aggregate the duplicate-compensation indicators."""
    report: dict[str, dict] = {}
    for dataset, seeds in SEEDS.items():
        per_seed: dict[str, list[float]] = {
            "original_ratio": [],
            "original_relative_spread": [],
            "residual_ratio": [],
            "residual_relative_spread": [],
            "original_minimum": [],
            "original_maximum": [],
            "residual_level": [],
        }
        classes: list[str] = []
        worst: list[str] = []
        for seed in seeds:
            path = results / dataset / f"seed{seed}" / "metrics.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            variants = payload["variants"]
            names = payload.get("class_names") or []
            classes = list(names)
            original = mass_statistics(
                variants[ORIGINAL]["effective_weighted_class_mass"]
            )
            residual = mass_statistics(
                variants[RESIDUAL]["effective_weighted_class_mass"]
            )
            per_seed["original_ratio"].append(original["ratio"])
            per_seed["original_relative_spread"].append(original["relative_spread"])
            per_seed["residual_ratio"].append(residual["ratio"])
            per_seed["residual_relative_spread"].append(residual["relative_spread"])
            per_seed["original_minimum"].append(original["minimum"])
            per_seed["original_maximum"].append(original["maximum"])
            per_seed["residual_level"].append(
                statistics.mean(
                    float(v)
                    for v in variants[RESIDUAL]["effective_weighted_class_mass"].values()
                )
            )
            mass = variants[ORIGINAL]["effective_weighted_class_mass"]
            index = max(mass, key=lambda key: float(mass[key]))
            if names and int(index) < len(names):
                worst.append(names[int(index)])
        report[dataset] = {
            "seeds": len(seeds),
            "class_names": classes,
            "most_over_weighted_class": (
                max(set(worst), key=worst.count) if worst else None
            ),
            **{name: summarise(values) for name, values in per_seed.items()},
        }
    return report


def main() -> None:
    """Write the duplicate-compensation summary for manuscript use."""
    parser = argparse.ArgumentParser(description="Extract weighted class mass")
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    report = collect(Path(args.results))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for dataset, entry in report.items():
        print(
            f"{dataset}: original ratio="
            f"{entry['original_ratio']['mean']:.2f}, "
            f"residual ratio={entry['residual_ratio']['mean']:.6f}, "
            f"over-weighted={entry['most_over_weighted_class']}"
        )


if __name__ == "__main__":
    main()
