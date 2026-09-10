"""Integrity and completeness checks for the published evidence tree.

The manuscript's numbers are only trustworthy if the files they came from are
the files shipped here. Two things are therefore checked separately: that every
file still hashes to its manifest entry, and that no per-seed run is missing.
A missing run is reported rather than skipped, because a silent skip is exactly
how an incomplete evidence set would pass unnoticed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

# Seeds the manuscript reports per dataset. UNSW-NB15 and NSL-KDD use ten
# paired seeds; both CIC protocols use three.
EXPECTED_SEEDS: dict[str, tuple[int, ...]] = {
    "unsw": (42, 123, 456, 789, 1001, 2024, 31415, 65537, 77777, 99991),
    "nslkdd": (42, 123, 456, 789, 1001, 2024, 31415, 65537, 77777, 99991),
    "cicids2017": (42, 123, 456),
}
DDOS_SEEDS: tuple[int, ...] = (42, 123, 456)

MANIFEST_NAME = "MANIFEST.sha256"


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Outcome of one verification step."""

    name: str
    passed: bool
    detail: str
    counts: dict[str, int]

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": "PASS" if self.passed else "FAIL",
            "detail": self.detail,
            "counts": dict(self.counts),
        }


def sha256_of(path: Path) -> str:
    """Return the SHA-256 of a file, read in chunks to bound memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(root: Path, relative_paths: list[str]) -> Path:
    """Write `MANIFEST.sha256` for the given files, sorted for stable diffs."""
    lines = [f"{sha256_of(root / rel)}  {rel}" for rel in sorted(relative_paths)]
    target = root / MANIFEST_NAME
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def read_manifest(manifest: Path) -> dict[str, str]:
    """Parse a `sha256sum`-style manifest into {relative path: digest}."""
    entries: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, relative = line.partition("  ")
        entries[relative.strip()] = digest.strip()
    return entries


def verify_manifest(root: Path) -> CheckResult:
    """Check every manifest entry against the file actually present."""
    manifest = root / MANIFEST_NAME
    if not manifest.is_file():
        return CheckResult(
            name=f"manifest:{root.name}",
            passed=False,
            detail=f"missing {manifest}",
            counts={"checked": 0, "mismatched": 0, "missing": 0},
        )

    entries = read_manifest(manifest)
    mismatched: list[str] = []
    missing: list[str] = []
    for relative, expected in sorted(entries.items()):
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        if sha256_of(path) != expected:
            mismatched.append(relative)

    problems = len(mismatched) + len(missing)
    detail = "all files match the manifest"
    if problems:
        shown = (mismatched + missing)[:5]
        detail = f"{problems} problem(s): {', '.join(shown)}"
    return CheckResult(
        name=f"manifest:{root.name}",
        passed=problems == 0,
        detail=detail,
        counts={
            "checked": len(entries),
            "mismatched": len(mismatched),
            "missing": len(missing),
        },
    )


def _seed_dirs(base: Path) -> dict[str, set[int]]:
    found: dict[str, set[int]] = {}
    if not base.is_dir():
        return found
    for dataset_dir in sorted(base.iterdir()):
        if not dataset_dir.is_dir():
            continue
        seeds = set()
        for seed_dir in dataset_dir.iterdir():
            if seed_dir.is_dir() and (seed_dir / "metrics.json").is_file():
                name = seed_dir.name.removeprefix("seed")
                if name.isdigit():
                    seeds.add(int(name))
        found[dataset_dir.name] = seeds
    return found


def verify_completeness(evidence_root: Path) -> CheckResult:
    """Check that every reported per-seed run is present."""
    problems: list[str] = []
    total = 0

    for layer, expected in (
        ("v3", EXPECTED_SEEDS),
        ("v4", EXPECTED_SEEDS),
    ):
        found = _seed_dirs(evidence_root / layer / "per_seed")
        for dataset, seeds in expected.items():
            present = found.get(dataset, set())
            total += len(present)
            gap = sorted(set(seeds) - present)
            if gap:
                problems.append(f"{layer}/{dataset} missing seeds {gap}")

    ddos = _seed_dirs(evidence_root / "ddos" / "per_seed").get("cicddos2019", set())
    total += len(ddos)
    ddos_gap = sorted(set(DDOS_SEEDS) - ddos)
    if ddos_gap:
        problems.append(f"ddos/cicddos2019 missing seeds {ddos_gap}")

    return CheckResult(
        name="evidence:completeness",
        passed=not problems,
        detail="; ".join(problems) if problems else f"{total} per-seed runs present",
        counts={"per_seed_runs": total, "problems": len(problems)},
    )


def load_metrics(evidence_root: Path, layer: str, dataset: str, seed: int) -> dict:
    """Load one per-seed metrics file."""
    path = evidence_root / layer / "per_seed" / dataset / f"seed{seed}" / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))
