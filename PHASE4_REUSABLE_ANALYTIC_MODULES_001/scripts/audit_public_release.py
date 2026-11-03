#!/usr/bin/env python3
"""Audit a Phase 4 public release against its deterministic integrity metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import stat
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
MANIFEST_BUILDER_PATH = SCRIPT_DIR / "build_release_manifest.py"
_spec = importlib.util.spec_from_file_location(
    "qstable_build_release_manifest", MANIFEST_BUILDER_PATH
)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Unable to load release manifest builder: {MANIFEST_BUILDER_PATH}")
builder = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = builder
_spec.loader.exec_module(builder)

EXPECTED_RESULTS_COUNT = 41
EXPECTED_RESULTS_TREE_SHA256 = (
    "9da2eb226a47c8633a5bf43afbdef48625026501e3ec78402627293091a40138"
)
EXPECTED_CONTRACT_SHA256 = (
    "b701177f25c7c758e50880bbf63901ba95d0d1e4bc7dc03d7af4038e4e8ff725"
)


def sha256_file(path: Path) -> str:
    return builder.sha256_file(path)


def results_tree_sha256(root: Path) -> str:
    files = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    rows = [
        f"{relative}\0{(root / relative).stat().st_size}\0{digest}\n"
        for relative, digest in files.items()
    ]
    return hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected_fields = [
        "relative_path",
        "file_type",
        "size_bytes",
        "sha256",
        "executable",
        "release_role",
    ]
    if rows and list(rows[0]) != expected_fields:
        raise builder.ReleaseManifestError(
            f"Unexpected release manifest columns: {list(rows[0])}"
        )
    return rows


def _read_checksums(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if "  " not in line:
            raise builder.ReleaseManifestError(
                f"Malformed SHA256SUMS line {line_number}: {line!r}"
            )
        digest, relative = line.split("  ", 1)
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise builder.ReleaseManifestError(
                f"Malformed SHA-256 on line {line_number}: {digest!r}"
            )
        if relative in values:
            raise builder.ReleaseManifestError(
                f"Duplicate SHA256SUMS path: {relative}"
            )
        values[relative] = digest
    return values


def _check(
    checks: list[dict[str, Any]],
    check_id: str,
    condition: bool,
    observed: Any,
    expected: Any,
) -> None:
    checks.append(
        {
            "check_id": check_id,
            "status": "PASS" if condition else "FAIL",
            "observed": observed,
            "expected": expected,
        }
    )


def audit_release(
    root: Path,
    *,
    require_verified_pass: bool = False,
    expected_results_count: int | None = EXPECTED_RESULTS_COUNT,
    expected_results_tree_sha256: str | None = EXPECTED_RESULTS_TREE_SHA256,
    expected_contract_sha256: str | None = EXPECTED_CONTRACT_SHA256,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise builder.ReleaseManifestError(f"Release root not found: {root}")

    manifest_path = root / builder.MANIFEST_RELATIVE_PATH
    summary_path = root / builder.SUMMARY_RELATIVE_PATH
    checksums_path = root / builder.CHECKSUMS_RELATIVE_PATH
    for path in (manifest_path, summary_path, checksums_path):
        if not path.is_file():
            raise builder.ReleaseManifestError(f"Required release metadata missing: {path}")

    manifest_rows = _read_manifest(manifest_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise builder.ReleaseManifestError("Release summary must contain a JSON object")
    checksum_rows = _read_checksums(checksums_path)
    current_rows, exclusions = builder.inventory_release(root)
    stable_exclusions = [
        item
        for item in exclusions
        if item.reason != "generated_release_metadata_self_exclusion"
    ]
    checks: list[dict[str, Any]] = []

    manifest_paths = [row["relative_path"] for row in manifest_rows]
    _check(
        checks,
        "REL-001-MANIFEST-SORTED-UNIQUE",
        manifest_paths == sorted(manifest_paths) and len(manifest_paths) == len(set(manifest_paths)),
        manifest_paths,
        "sorted unique paths",
    )

    current_by_path = {row.relative_path: row for row in current_rows}
    manifest_by_path = {row["relative_path"]: row for row in manifest_rows}
    _check(
        checks,
        "REL-002-ACTIVE-INVENTORY-EXACT",
        set(current_by_path) == set(manifest_by_path),
        {
            "missing_from_tree": sorted(set(manifest_by_path) - set(current_by_path)),
            "unmanifested_active": sorted(set(current_by_path) - set(manifest_by_path)),
        },
        {"missing_from_tree": [], "unmanifested_active": []},
    )

    mismatches: list[dict[str, Any]] = []
    for relative in sorted(set(current_by_path) & set(manifest_by_path)):
        current = current_by_path[relative]
        recorded = manifest_by_path[relative]
        observed = {
            "file_type": current.file_type,
            "size_bytes": current.size_bytes,
            "sha256": current.sha256,
            "executable": "true" if current.executable else "false",
            "release_role": current.release_role,
        }
        expected = {key: recorded[key] for key in observed}
        expected["size_bytes"] = int(expected["size_bytes"])
        if observed != expected:
            mismatches.append(
                {"relative_path": relative, "observed": observed, "expected": expected}
            )
    _check(
        checks,
        "REL-003-FILE-METADATA-HASHES",
        not mismatches,
        mismatches,
        [],
    )

    manifest_sha = sha256_file(manifest_path)
    _check(
        checks,
        "REL-004-SUMMARY-MANIFEST-IDENTITY",
        summary.get("manifest")
        == {
            "relative_path": builder.MANIFEST_RELATIVE_PATH,
            "sha256": manifest_sha,
            "rows": len(manifest_rows),
        },
        summary.get("manifest"),
        {
            "relative_path": builder.MANIFEST_RELATIVE_PATH,
            "sha256": manifest_sha,
            "rows": len(manifest_rows),
        },
    )

    role_counts = Counter(row.release_role for row in current_rows)
    expected_inventory = {
        "files": len(current_rows),
        "bytes": sum(row.size_bytes for row in current_rows),
        "executable_files": sum(row.executable for row in current_rows),
        "role_counts": dict(sorted(role_counts.items())),
    }
    _check(
        checks,
        "REL-005-SUMMARY-INVENTORY",
        summary.get("inventory") == expected_inventory,
        summary.get("inventory"),
        expected_inventory,
    )

    reason_counts = Counter(item.reason for item in stable_exclusions)
    expected_exclusions = {
        "count": len(stable_exclusions),
        "reason_counts": dict(sorted(reason_counts.items())),
        "entries": [
            {"relative_path": item.relative_path, "reason": item.reason}
            for item in stable_exclusions
        ],
        "generated_metadata_self_excluded": sorted(builder.GENERATED_RELEASE_PATHS),
    }
    _check(
        checks,
        "REL-006-SUMMARY-EXCLUSIONS",
        summary.get("exclusions") == expected_exclusions,
        summary.get("exclusions"),
        expected_exclusions,
    )

    expected_checksum_paths = set(manifest_by_path) | {
        builder.MANIFEST_RELATIVE_PATH,
        builder.SUMMARY_RELATIVE_PATH,
    }
    _check(
        checks,
        "REL-007-SHA256SUMS-COVERAGE",
        set(checksum_rows) == expected_checksum_paths,
        sorted(checksum_rows),
        sorted(expected_checksum_paths),
    )
    checksum_mismatches = []
    for relative, expected_digest in sorted(checksum_rows.items()):
        path = root / relative
        observed_digest = sha256_file(path) if path.is_file() else None
        if observed_digest != expected_digest:
            checksum_mismatches.append(
                {
                    "relative_path": relative,
                    "observed": observed_digest,
                    "expected": expected_digest,
                }
            )
    _check(
        checks,
        "REL-008-SHA256SUMS-VALUES",
        not checksum_mismatches,
        checksum_mismatches,
        [],
    )

    contract_path = root / builder.CONTRACT_RELATIVE_PATH
    contract_sha = sha256_file(contract_path) if contract_path.is_file() else None
    summary_contract = summary.get("source_contract")
    expected_contract = {
        "relative_path": builder.CONTRACT_RELATIVE_PATH,
        "sha256": contract_sha,
    }
    _check(
        checks,
        "REL-009-CONTRACT-SUMMARY",
        summary_contract == expected_contract,
        summary_contract,
        expected_contract,
    )
    if expected_contract_sha256 is not None:
        _check(
            checks,
            "REL-010-CONTRACT-FROZEN-HASH",
            contract_sha == expected_contract_sha256,
            contract_sha,
            expected_contract_sha256,
        )

    _check(
        checks,
        "REL-011-PACKAGE-IDENTITY",
        summary.get("package_name") == builder.EXPECTED_PACKAGE_NAME
        and summary.get("package_version") == builder.EXPECTED_VERSION,
        {
            "package_name": summary.get("package_name"),
            "package_version": summary.get("package_version"),
        },
        {
            "package_name": builder.EXPECTED_PACKAGE_NAME,
            "package_version": builder.EXPECTED_VERSION,
        },
    )

    boundaries = summary.get("scientific_boundaries", {})
    expected_boundaries = {
        "phase3_bundled": False,
        "historical_notebooks_are_runtime_inputs": False,
        "pumping_validation_claimed": False,
        "groundwater_attribution_claimed": False,
        "completed_class_i_assignments": 0,
    }
    _check(
        checks,
        "REL-012-SCIENTIFIC-BOUNDARIES",
        boundaries == expected_boundaries,
        boundaries,
        expected_boundaries,
    )

    results_root = root / "results"
    result_files = sorted(path for path in results_root.rglob("*") if path.is_file())
    if expected_results_count is not None:
        _check(
            checks,
            "REL-013-CANONICAL-RESULT-COUNT",
            len(result_files) == expected_results_count,
            len(result_files),
            expected_results_count,
        )
    if expected_results_tree_sha256 is not None:
        observed_tree = results_tree_sha256(results_root)
        _check(
            checks,
            "REL-014-CANONICAL-RESULT-TREE",
            observed_tree == expected_results_tree_sha256,
            observed_tree,
            expected_results_tree_sha256,
        )

    prohibited_inventory = [
        row.relative_path
        for row in current_rows
        if "OLD" in Path(row.relative_path).parts
        or builder.PHASE3_NAME in Path(row.relative_path).parts
        or row.relative_path.lower().startswith("phase3_cura")
    ]
    _check(
        checks,
        "REL-015-NO-OLD-OR-PHASE3",
        not prohibited_inventory,
        prohibited_inventory,
        [],
    )

    if require_verified_pass:
        verified = summary.get("verified_status")
        required_sections = ("tests", "runtime_validation", "pipeline", "clean_room")
        statuses = (
            {section: verified.get(section, {}).get("status") for section in required_sections}
            if isinstance(verified, dict)
            else {}
        )
        _check(
            checks,
            "REL-016-VERIFIED-STATUS",
            statuses == {section: "PASS" for section in required_sections},
            statuses,
            {section: "PASS" for section in required_sections},
        )

    failures = [check for check in checks if check["status"] == "FAIL"]
    return {
        "audit_version": "1.0.0",
        "release_root": str(root),
        "status": "PASS" if not failures else "FAIL",
        "checks": len(checks),
        "passed": len(checks) - len(failures),
        "failed": len(failures),
        "results_tree_sha256": results_tree_sha256(results_root),
        "check_results": checks,
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--require-verified-pass", action="store_true")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--expected-results-count", type=int, default=EXPECTED_RESULTS_COUNT)
    parser.add_argument(
        "--expected-results-tree-sha256",
        default=EXPECTED_RESULTS_TREE_SHA256,
    )
    parser.add_argument(
        "--expected-contract-sha256",
        default=EXPECTED_CONTRACT_SHA256,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = audit_release(
            args.root,
            require_verified_pass=args.require_verified_pass,
            expected_results_count=args.expected_results_count,
            expected_results_tree_sha256=args.expected_results_tree_sha256,
            expected_contract_sha256=args.expected_contract_sha256,
        )
        if args.output_json is not None:
            _atomic_write_json(args.output_json, report)
    except (builder.ReleaseManifestError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"QSTABLE_PUBLIC_RELEASE_AUDIT_ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
