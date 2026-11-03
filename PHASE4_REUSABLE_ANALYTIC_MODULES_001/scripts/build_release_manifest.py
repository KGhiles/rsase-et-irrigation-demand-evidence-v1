#!/usr/bin/env python3
"""Build deterministic integrity metadata for the Phase 4 public release.

The release inventory is intentionally scoped to the Phase 4 software/results
repository. Frozen Phase 3 inputs, historical ``OLD`` trees, caches, build
artifacts, validation scratch files, and pipeline staging trees are excluded.
The three generated release-metadata files are self-excluded from the CSV
manifest to avoid circular hashes; ``SHA256SUMS`` covers the manifest and
summary in addition to every inventoried public-release file.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence

MANIFEST_RELATIVE_PATH = "release/PHASE4_RELEASE_MANIFEST.csv"
SUMMARY_RELATIVE_PATH = "release/PHASE4_RELEASE_SUMMARY.json"
CHECKSUMS_RELATIVE_PATH = "release/SHA256SUMS"
GENERATED_RELEASE_PATHS = {
    MANIFEST_RELATIVE_PATH,
    SUMMARY_RELATIVE_PATH,
    CHECKSUMS_RELATIVE_PATH,
}
PHASE3_NAME = "PHASE3_CURATED_INPUT_LAYER_FINAL_002"
EXPECTED_PACKAGE_NAME = "PHASE4_REUSABLE_ANALYTIC_MODULES_001"
EXPECTED_VERSION = "1.0.0"
CONTRACT_RELATIVE_PATH = "config/phase4_source_contract.yml"

EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".pyre",
    ".pytype",
    ".ruff_cache",
    ".hypothesis",
    ".tox",
    ".nox",
    ".venv",
    "venv",
    "env",
    "ENV",
    ".conda",
    "__pycache__",
    ".ipynb_checkpoints",
    "build",
    "dist",
    ".eggs",
    "pip-wheel-metadata",
    "validation_reports",
    "results_candidate",
}
EXCLUDED_FILE_NAMES = {
    ".DS_Store",
    "Thumbs.db",
    ".coverage",
    "coverage.xml",
    "junit.xml",
}
EXCLUDED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".tmp",
    ".temp",
    ".bak",
    ".swp",
    ".swo",
}
ARCHIVE_SUFFIXES = {".zip", ".tar", ".tgz", ".gz", ".bz2", ".xz", ".7z"}


class ReleaseManifestError(RuntimeError):
    """Raised when a repository cannot be safely inventoried."""


@dataclass(frozen=True, slots=True)
class InventoryRow:
    relative_path: str
    file_type: str
    size_bytes: int
    sha256: str
    executable: bool
    release_role: str


@dataclass(frozen=True, slots=True)
class Exclusion:
    relative_path: str
    reason: str


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(path: Path, root: Path) -> str:
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ReleaseManifestError(f"Path escapes release root: {path}") from exc
    value = relative.as_posix()
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ReleaseManifestError(f"Unsafe release path: {value!r}")
    return value


def _is_phase3_path(parts: tuple[str, ...], name: str) -> bool:
    if PHASE3_NAME in parts:
        return True
    lowered = name.lower()
    return (
        lowered in {"phase3_cura.zip", f"{PHASE3_NAME.lower()}.zip"}
        or lowered.startswith(PHASE3_NAME.lower() + ".")
    )


def _directory_exclusion_reason(relative_parts: tuple[str, ...], name: str) -> str | None:
    if name == "OLD":
        return "historical_old_directory"
    if _is_phase3_path(relative_parts, name):
        return "phase3_distributed_separately"
    if name in EXCLUDED_DIRECTORY_NAMES or name.endswith(".egg-info"):
        return "cache_build_or_environment_directory"
    if name.startswith(".phase4-pipeline-staging-"):
        return "pipeline_staging_directory"
    if name.startswith(".phase4-pipeline-backup-"):
        return "pipeline_backup_directory"
    if name.startswith(".qstable-validation."):
        return "validation_scratch_directory"
    if name.startswith("results_candidate_"):
        return "candidate_results_directory"
    if name.startswith(".release-manifest-staging-"):
        return "release_manifest_staging_directory"
    return None


def _file_exclusion_reason(relative_path: str, name: str) -> str | None:
    if relative_path in GENERATED_RELEASE_PATHS:
        return "generated_release_metadata_self_exclusion"
    if _is_phase3_path(PurePosixPath(relative_path).parts, name):
        return "phase3_distributed_separately"
    if name in EXCLUDED_FILE_NAMES:
        return "cache_or_platform_file"
    if Path(name).suffix.lower() in EXCLUDED_SUFFIXES:
        return "temporary_backup_or_bytecode_file"
    if name.startswith(".coverage."):
        return "coverage_scratch_file"
    if name.startswith("slurm-") and Path(name).suffix.lower() in {".out", ".err"}:
        return "scheduler_output"
    if name.endswith((".sbatch.out", ".sbatch.err")):
        return "scheduler_output"
    return None


def _file_type(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".py": "python",
        ".sh": "shell",
        ".toml": "toml",
        ".yml": "yaml",
        ".yaml": "yaml",
        ".json": "json",
        ".csv": "csv",
        ".md": "markdown",
        ".txt": "text",
        ".tif": "geotiff",
        ".tiff": "geotiff",
    }.get(suffix, suffix.lstrip(".") or "binary")


def _release_role(relative_path: str) -> str:
    path = PurePosixPath(relative_path)
    if len(path.parts) == 1:
        return "root_metadata"
    if path.parts[0] == "config":
        return "configuration"
    if path.parts[:2] == ("src", "qstable_awm"):
        return "python_source"
    if path.parts[0] == "scripts":
        return "release_tool" if path.name in {
            "build_release_manifest.py",
            "audit_public_release.py",
        } else "operational_script"
    if path.parts[0] == "tests":
        return "test"
    if path.parts[0] == "results":
        return "canonical_result"
    if path.parts[0] == "release":
        return "release_metadata"
    return "other"


def inventory_release(root: Path) -> tuple[list[InventoryRow], list[Exclusion]]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ReleaseManifestError(f"Release root not found: {root}")
    if root.name != EXPECTED_PACKAGE_NAME:
        raise ReleaseManifestError(
            f"Release root must be named {EXPECTED_PACKAGE_NAME}: {root}"
        )

    rows: list[InventoryRow] = []
    exclusions: list[Exclusion] = []
    casefold_paths: dict[str, str] = {}

    for current_text, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        current = Path(current_text)
        current_relative = current.relative_to(root)

        kept_directories: list[str] = []
        for name in sorted(directory_names):
            candidate = current / name
            relative = (current_relative / name).as_posix()
            reason = _directory_exclusion_reason(tuple((current_relative / name).parts), name)
            if reason:
                exclusions.append(Exclusion(relative + "/", reason))
                continue
            if candidate.is_symlink():
                raise ReleaseManifestError(f"Symlink is not allowed in release tree: {relative}")
            kept_directories.append(name)
        directory_names[:] = kept_directories

        for name in sorted(file_names):
            path = current / name
            lexical_relative = (current_relative / name).as_posix()
            reason = _file_exclusion_reason(lexical_relative, name)
            if reason:
                exclusions.append(Exclusion(lexical_relative, reason))
                continue
            if path.is_symlink():
                raise ReleaseManifestError(
                    f"Symlink is not allowed in release tree: {lexical_relative}"
                )
            if not path.is_file():
                raise ReleaseManifestError(f"Unsupported filesystem entry: {lexical_relative}")
            relative = _safe_relative(path, root)
            folded = relative.casefold()
            previous = casefold_paths.get(folded)
            if previous is not None and previous != relative:
                raise ReleaseManifestError(
                    "Case-insensitive duplicate release paths: "
                    f"{previous!r} and {relative!r}"
                )
            casefold_paths[folded] = relative
            mode = path.stat().st_mode
            rows.append(
                InventoryRow(
                    relative_path=relative,
                    file_type=_file_type(path),
                    size_bytes=path.stat().st_size,
                    sha256=sha256_file(path),
                    executable=bool(mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)),
                    release_role=_release_role(relative),
                )
            )

    rows.sort(key=lambda row: row.relative_path)
    exclusions.sort(key=lambda item: (item.reason, item.relative_path))
    if len({row.relative_path for row in rows}) != len(rows):
        raise ReleaseManifestError("Duplicate paths detected in active release inventory")
    return rows, exclusions


def _read_package_version(root: Path) -> str:
    path = root / "src" / "qstable_awm" / "__init__.py"
    if not path.is_file():
        raise ReleaseManifestError(f"Package initializer missing: {path}")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "VERSION_INFO" and isinstance(node.value, ast.Tuple):
                values = [ast.literal_eval(element) for element in node.value.elts]
                version = ".".join(str(value) for value in values)
                if version != EXPECTED_VERSION:
                    raise ReleaseManifestError(
                        f"Unexpected package version {version}; expected {EXPECTED_VERSION}"
                    )
                return version
    raise ReleaseManifestError("VERSION_INFO was not found in package initializer")


def _verification_payload(args: argparse.Namespace) -> dict[str, object]:
    verification: dict[str, object] = {}
    test_values = (args.tests_collected, args.tests_passed, args.tests_failed)
    if any(value is not None for value in test_values):
        if not all(value is not None for value in test_values):
            raise ReleaseManifestError(
                "Test status requires --tests-collected, --tests-passed, and --tests-failed"
            )
        verification["tests"] = {
            "collected": args.tests_collected,
            "passed": args.tests_passed,
            "failed": args.tests_failed,
            "status": "PASS" if args.tests_failed == 0 and args.tests_passed == args.tests_collected else "FAIL",
        }

    runtime_values = (
        args.runtime_checks,
        args.runtime_passed,
        args.runtime_warnings,
        args.runtime_failed,
    )
    if any(value is not None for value in runtime_values):
        if not all(value is not None for value in runtime_values):
            raise ReleaseManifestError(
                "Runtime status requires checks, passed, warnings, and failed values"
            )
        verification["runtime_validation"] = {
            "checks": args.runtime_checks,
            "passed": args.runtime_passed,
            "warnings": args.runtime_warnings,
            "failed": args.runtime_failed,
            "status": "PASS" if args.runtime_failed == 0 and args.runtime_warnings == 0 and args.runtime_passed == args.runtime_checks else "FAIL",
        }

    if args.pipeline_status is not None:
        if args.pipeline_status not in {"PASS", "FAIL"}:
            raise ReleaseManifestError("--pipeline-status must be PASS or FAIL")
        verification["pipeline"] = {
            "status": args.pipeline_status,
            "canonical_result_files": args.canonical_result_files,
            "results_tree_sha256": args.results_tree_sha256,
        }
    if args.clean_room_status is not None:
        if args.clean_room_status not in {"PASS", "FAIL"}:
            raise ReleaseManifestError("--clean-room-status must be PASS or FAIL")
        verification["clean_room"] = {"status": args.clean_room_status}
    return verification


def _manifest_bytes(rows: Iterable[InventoryRow]) -> bytes:
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=(
            "relative_path",
            "file_type",
            "size_bytes",
            "sha256",
            "executable",
            "release_role",
        ),
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "relative_path": row.relative_path,
                "file_type": row.file_type,
                "size_bytes": row.size_bytes,
                "sha256": row.sha256,
                "executable": "true" if row.executable else "false",
                "release_role": row.release_role,
            }
        )
    return buffer.getvalue().encode("utf-8")


def _summary_bytes(
    *,
    root: Path,
    rows: list[InventoryRow],
    exclusions: list[Exclusion],
    manifest_sha256: str,
    verification: dict[str, object],
) -> bytes:
    contract_path = root / CONTRACT_RELATIVE_PATH
    if not contract_path.is_file():
        raise ReleaseManifestError(f"Source contract missing: {contract_path}")
    role_counts = Counter(row.release_role for row in rows)
    reason_counts = Counter(item.reason for item in exclusions)
    payload: dict[str, object] = {
        "release_summary_version": "1.0.0",
        "package_name": root.name,
        "package_version": _read_package_version(root),
        "source_contract": {
            "relative_path": CONTRACT_RELATIVE_PATH,
            "sha256": sha256_file(contract_path),
        },
        "manifest": {
            "relative_path": MANIFEST_RELATIVE_PATH,
            "sha256": manifest_sha256,
            "rows": len(rows),
        },
        "inventory": {
            "files": len(rows),
            "bytes": sum(row.size_bytes for row in rows),
            "executable_files": sum(row.executable for row in rows),
            "role_counts": dict(sorted(role_counts.items())),
        },
        "exclusions": {
            "count": len(exclusions),
            "reason_counts": dict(sorted(reason_counts.items())),
            "entries": [
                {"relative_path": item.relative_path, "reason": item.reason}
                for item in exclusions
            ],
            "generated_metadata_self_excluded": sorted(GENERATED_RELEASE_PATHS),
        },
        "scientific_boundaries": {
            "phase3_bundled": False,
            "historical_notebooks_are_runtime_inputs": False,
            "pumping_validation_claimed": False,
            "groundwater_attribution_claimed": False,
            "completed_class_i_assignments": 0,
        },
    }
    if verification:
        payload["verified_status"] = verification
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _checksums_bytes(
    root: Path,
    rows: list[InventoryRow],
    manifest_bytes: bytes,
    summary_bytes: bytes,
) -> bytes:
    entries = [(row.relative_path, row.sha256) for row in rows]
    entries.extend(
        [
            (MANIFEST_RELATIVE_PATH, hashlib.sha256(manifest_bytes).hexdigest()),
            (SUMMARY_RELATIVE_PATH, hashlib.sha256(summary_bytes).hexdigest()),
        ]
    )
    entries.sort()
    return "".join(f"{digest}  {relative}\n" for relative, digest in entries).encode("utf-8")


def _publish_transaction(
    root: Path,
    payloads: dict[str, bytes],
    *,
    overwrite: bool,
) -> None:
    release_dir = root / "release"
    release_dir.mkdir(parents=True, exist_ok=True)
    destinations = {relative: root / relative for relative in payloads}
    existing = [relative for relative, path in destinations.items() if path.exists()]
    if existing and not overwrite:
        raise ReleaseManifestError(
            f"Refusing to overwrite existing release metadata: {sorted(existing)}"
        )

    staging = Path(tempfile.mkdtemp(prefix=".release-manifest-staging-", dir=release_dir))
    backup = staging / "backup"
    published: list[str] = []
    try:
        for relative, data in payloads.items():
            staged = staging / Path(relative).name
            staged.write_bytes(data)
        if overwrite:
            backup.mkdir()
            for relative, destination in destinations.items():
                if destination.exists():
                    os.replace(destination, backup / Path(relative).name)
        for relative, destination in destinations.items():
            os.replace(staging / Path(relative).name, destination)
            published.append(relative)
    except Exception:
        for relative in published:
            destination = destinations[relative]
            if destination.exists():
                destination.unlink()
        if backup.exists():
            for relative, destination in destinations.items():
                saved = backup / Path(relative).name
                if saved.exists():
                    os.replace(saved, destination)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def build_release_metadata(root: Path, args: argparse.Namespace) -> dict[str, str]:
    root = root.expanduser().resolve()
    rows, exclusions = inventory_release(root)
    # Generated release metadata are always listed separately in the summary.
    # Excluding their presence-dependent inventory entries keeps repeated builds
    # byte deterministic while preserving the explicit self-exclusion policy.
    stable_exclusions = [
        item
        for item in exclusions
        if item.reason != "generated_release_metadata_self_exclusion"
    ]
    verification = _verification_payload(args)
    manifest_bytes = _manifest_bytes(rows)
    summary_bytes = _summary_bytes(
        root=root,
        rows=rows,
        exclusions=stable_exclusions,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        verification=verification,
    )
    checksums_bytes = _checksums_bytes(root, rows, manifest_bytes, summary_bytes)
    payloads = {
        MANIFEST_RELATIVE_PATH: manifest_bytes,
        SUMMARY_RELATIVE_PATH: summary_bytes,
        CHECKSUMS_RELATIVE_PATH: checksums_bytes,
    }
    _publish_transaction(root, payloads, overwrite=args.overwrite)
    return {
        relative: hashlib.sha256(data).hexdigest() for relative, data in payloads.items()
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Phase 4 repository root.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--tests-collected", type=int)
    parser.add_argument("--tests-passed", type=int)
    parser.add_argument("--tests-failed", type=int)
    parser.add_argument("--runtime-checks", type=int)
    parser.add_argument("--runtime-passed", type=int)
    parser.add_argument("--runtime-warnings", type=int)
    parser.add_argument("--runtime-failed", type=int)
    parser.add_argument("--pipeline-status")
    parser.add_argument("--canonical-result-files", type=int)
    parser.add_argument("--results-tree-sha256")
    parser.add_argument("--clean-room-status")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        hashes = build_release_metadata(args.root, args)
    except (ReleaseManifestError, OSError, SyntaxError, ValueError) as exc:
        print(f"QSTABLE_RELEASE_MANIFEST_ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {"status": "PASS", "outputs": hashes},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
