"""Command-line entry point.

python -m rwxai verify        # everything: integrity, tables, figures
python -m rwxai verify-evidence
python -m rwxai coverage
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rwxai.evidence import verify_completeness, verify_manifest
from rwxai.tables import dataset_profiles
from rwxai.verify import run_all

REPO_ROOT = Path(__file__).resolve().parents[2]


def _verify(args: argparse.Namespace) -> int:
    passed, report = run_all(args.root, manuscript=args.manuscript)
    if args.report:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    for check in report["checks"]:
        print(f"{check['status']:4s}  {check['name']:28s} {check['detail']}")
    print(f"\nstatus: {report['status']}")
    return 0 if passed else 1


def _verify_evidence(args: argparse.Namespace) -> int:
    evidence = args.root / "evidence"
    results = [verify_manifest(evidence), verify_completeness(evidence)]
    for item in results:
        print(f"{'PASS' if item.passed else 'FAIL':4s}  {item.name:28s} {item.detail}")
    return 0 if all(item.passed for item in results) else 1


def _coverage(args: argparse.Namespace) -> int:
    for name, profile in dataset_profiles(args.root / "evidence").items():
        width = (
            str(profile.encoded_features_max)
            if profile.encoded_features_min == profile.encoded_features_max
            else f"{profile.encoded_features_min}-{profile.encoded_features_max}"
        )
        print(
            f"{name:12s} train={profile.train_records:>9,} "
            f"test={profile.test_records:>9,} "
            f"features={profile.retained_features}/{width} seeds={profile.seeds}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rwxai", description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)

    verify = sub.add_parser("verify", help="verify every figure and table")
    verify.add_argument("--report", type=Path, default=None)
    verify.add_argument(
        "--manuscript",
        type=Path,
        default=None,
        help="also require this DOCX to match the published claim snapshot",
    )
    verify.add_argument("--all", action="store_true", help="accepted for clarity")
    verify.set_defaults(handler=_verify)

    evidence = sub.add_parser("verify-evidence", help="check integrity only")
    evidence.set_defaults(handler=_verify_evidence)

    coverage = sub.add_parser("coverage", help="print the recomputed Table 1 rows")
    coverage.set_defaults(handler=_coverage)

    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
