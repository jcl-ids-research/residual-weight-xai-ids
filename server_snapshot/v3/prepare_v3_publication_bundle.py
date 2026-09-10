from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


EXPECTED_RUNS = 23
EXPECTED_EVIDENCE_FILES = 667
EXPECTED_FIGURE_FILES = 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def safe_evidence_path(results_root: Path, raw_path: str) -> Path:
    relative = PurePosixPath(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe evidence path: {raw_path!r}")
    return results_root.joinpath(*relative.parts)


def audit(root: Path) -> dict:
    results_root = root / "results"
    aggregate_root = root / "aggregate"
    manifest_path = aggregate_root / "evidence_manifest.json"
    summary_path = aggregate_root / "summary.json"
    errors: list[str] = []

    if not manifest_path.is_file() or not summary_path.is_file():
        missing = [
            str(path) for path in (manifest_path, summary_path) if not path.is_file()
        ]
        raise FileNotFoundError("Missing aggregate files: " + ", ".join(missing))

    manifest = load_json(manifest_path)
    summary = load_json(summary_path)
    if not isinstance(manifest, list):
        raise TypeError("evidence_manifest.json must contain a list")

    verified_files = 0
    verified_bytes = 0
    seen_paths: set[str] = set()
    metrics_paths: list[Path] = []
    for index, item in enumerate(manifest):
        if not isinstance(item, dict):
            errors.append(f"manifest[{index}] is not an object")
            continue
        if not {"path", "bytes", "sha256"}.issubset(item):
            errors.append(f"manifest[{index}] is missing required fields")
            continue
        raw_path = str(item["path"])
        if raw_path in seen_paths:
            errors.append(f"duplicate manifest path: {raw_path}")
            continue
        seen_paths.add(raw_path)
        try:
            path = safe_evidence_path(results_root, raw_path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if not path.is_file():
            errors.append(f"missing evidence file: {raw_path}")
            continue
        actual_bytes = path.stat().st_size
        expected_bytes = int(item["bytes"])
        if actual_bytes != expected_bytes:
            errors.append(
                f"size mismatch: {raw_path}: {actual_bytes} != {expected_bytes}"
            )
            continue
        actual_hash = sha256(path)
        expected_hash = str(item["sha256"]).lower()
        if actual_hash != expected_hash:
            errors.append(f"sha256 mismatch: {raw_path}")
            continue
        verified_files += 1
        verified_bytes += actual_bytes
        if path.name == "metrics.json":
            metrics_paths.append(path)

    required_runs = int(summary.get("required_runs", -1))
    verified_runs = int(summary.get("verified_runs", -1))
    evidence_files = int(summary.get("evidence_files", -1))
    if required_runs != EXPECTED_RUNS or verified_runs != EXPECTED_RUNS:
        errors.append(
            f"run count mismatch: required={required_runs}, verified={verified_runs}"
        )
    if len(metrics_paths) != EXPECTED_RUNS:
        errors.append(f"metrics.json count mismatch: {len(metrics_paths)}")
    if evidence_files != len(manifest):
        errors.append(
            f"summary/manifest count mismatch: {evidence_files} != {len(manifest)}"
        )
    if len(manifest) != EXPECTED_EVIDENCE_FILES:
        errors.append(f"unexpected evidence file count: {len(manifest)}")

    metrics_schema_verified = 0
    for path in metrics_paths:
        try:
            metrics = load_json(path)
            variants = metrics.get("variants", {})
            baselines = metrics.get("raw_training_baselines", {})
            residual = variants.get("smotenc_tomek_residual_weight", {})
            if metrics.get("schema_version") != 2:
                raise ValueError("schema_version is not 2")
            if len(variants) != 7:
                raise ValueError(f"variant count is {len(variants)}, expected 7")
            if len(baselines) != 5:
                raise ValueError(f"baseline count is {len(baselines)}, expected 5")
            if residual.get("weight_source") != "post_smote_tomek_distribution":
                raise ValueError("residual weight source is incorrect")
            metrics_schema_verified += 1
        except Exception as exc:  # Report every malformed run in one audit.
            errors.append(f"metrics schema failure: {path}: {exc}")

    figure_root = aggregate_root / "figures"
    figure_paths = sorted(
        path for path in figure_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".pdf"}
    )
    png_stems = {path.stem for path in figure_paths if path.suffix.lower() == ".png"}
    pdf_stems = {path.stem for path in figure_paths if path.suffix.lower() == ".pdf"}
    if len(figure_paths) != EXPECTED_FIGURE_FILES:
        errors.append(f"unexpected figure file count: {len(figure_paths)}")
    if png_stems != pdf_stems or len(png_stems) != EXPECTED_FIGURE_FILES // 2:
        errors.append("PNG/PDF figure pairs are incomplete")

    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "server_root": str(root),
        "status": "PASS" if not errors else "FAIL",
        "required_runs": required_runs,
        "verified_runs": verified_runs,
        "metrics_schema_verified": metrics_schema_verified,
        "manifest_files": len(manifest),
        "verified_evidence_files": verified_files,
        "verified_evidence_bytes": verified_bytes,
        "figure_files": len(figure_paths),
        "figure_pairs": len(png_stems & pdf_stems),
        "errors": errors,
    }
    report_path = aggregate_root / "server_integrity_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def publication_files(root: Path) -> list[tuple[Path, str]]:
    selected: dict[str, Path] = {}

    aggregate_root = root / "aggregate"
    for path in aggregate_root.rglob("*"):
        if path.is_file():
            arcname = str(PurePosixPath("aggregate", *path.relative_to(aggregate_root).parts))
            selected[arcname] = path

    results_root = root / "results"
    for path in results_root.rglob("metrics.json"):
        arcname = str(PurePosixPath("results", *path.relative_to(results_root).parts))
        selected[arcname] = path

    logs_root = root / "logs"
    if logs_root.is_dir():
        for path in logs_root.glob("*.log"):
            if path.is_file():
                selected[str(PurePosixPath("logs", path.name))] = path

    for name in ("queue.status", "queue.log", "data_gate.json"):
        path = root / name
        if path.is_file():
            selected[name] = path

    return [(path, arcname) for arcname, path in sorted(selected.items())]


def build_bundle(root: Path, audit_report: dict) -> dict:
    if audit_report.get("status") != "PASS":
        raise RuntimeError("Refusing to package results because the audit failed")

    files = publication_files(root)
    bundle_manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "server_root": str(root),
        "selection_policy": (
            "Aggregate outputs and figures, per-run metrics.json files, concise "
            "logs, queue status, and the data-gate report. Model weights, full "
            "prediction arrays, SHAP arrays, caches, and checkpoints are excluded."
        ),
        "files": [
            {
                "path": arcname,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path, arcname in files
        ],
    }
    manifest_path = root / "publication_bundle_manifest.json"
    manifest_path.write_text(
        json.dumps(bundle_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    files.append((manifest_path, "publication_bundle_manifest.json"))

    bundle_path = root / "publication_bundle_v3.tar.gz"
    temporary_path = root / "publication_bundle_v3.tar.gz.tmp"
    with tarfile.open(temporary_path, "w:gz", compresslevel=6) as archive:
        for path, arcname in files:
            archive.add(path, arcname=arcname, recursive=False)
    os.replace(temporary_path, bundle_path)

    return {
        "bundle_path": str(bundle_path),
        "bundle_bytes": bundle_path.stat().st_size,
        "bundle_sha256": sha256(bundle_path),
        "selected_files": len(files),
        "selected_source_bytes": sum(path.stat().st_size for path, _ in files),
        "excluded_large_artifacts": True,
        "audit_status": audit_report["status"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="/opt/ids_revision/v3_strengthened", type=Path
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    integrity = audit(arguments.root)
    print("SERVER_AUDIT=" + json.dumps(integrity, ensure_ascii=False), flush=True)
    if integrity["status"] != "PASS":
        raise SystemExit(2)
    bundle = build_bundle(arguments.root, integrity)
    print("PUBLICATION_BUNDLE_META=" + json.dumps(bundle), flush=True)
