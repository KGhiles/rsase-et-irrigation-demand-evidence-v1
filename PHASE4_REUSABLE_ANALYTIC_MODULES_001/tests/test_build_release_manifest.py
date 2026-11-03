"""Tests for deterministic public-release manifest construction."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PHASE4_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PHASE4_ROOT / "scripts" / "build_release_manifest.py"

spec = importlib.util.spec_from_file_location("build_release_manifest", SCRIPT)
assert spec and spec.loader
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def _minimal_repo(tmp_path: Path) -> Path:
    root = tmp_path / m.EXPECTED_PACKAGE_NAME
    (root / "config").mkdir(parents=True)
    (root / "src" / "qstable_awm").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "tests").mkdir()
    (root / "results" / "pipeline").mkdir(parents=True)
    (root / "config" / "phase4_source_contract.yml").write_text(
        "contract_version: 1.4.0\n", encoding="utf-8"
    )
    (root / "src" / "qstable_awm" / "__init__.py").write_text(
        "from typing import Final\nVERSION_INFO: Final[tuple[int, int, int]] = (1, 0, 0)\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text("# Test\n", encoding="utf-8")
    runner = root / "scripts" / "run.sh"
    runner.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    runner.chmod(0o755)
    (root / "tests" / "test_sample.py").write_text("def test_ok(): assert True\n")
    (root / "results" / "pipeline" / "result.json").write_text("{}\n")
    return root


def _args(**overrides):
    values = dict(
        overwrite=False,
        tests_collected=None,
        tests_passed=None,
        tests_failed=None,
        runtime_checks=None,
        runtime_passed=None,
        runtime_warnings=None,
        runtime_failed=None,
        pipeline_status=None,
        canonical_result_files=None,
        results_tree_sha256=None,
        clean_room_status=None,
    )
    values.update(overrides)
    return type("Args", (), values)()


def _read_manifest(root: Path) -> list[dict[str, str]]:
    with (root / m.MANIFEST_RELATIVE_PATH).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_script_is_executable_and_syntax_valid() -> None:
    assert os.access(SCRIPT, os.X_OK)
    compile(SCRIPT.read_text(encoding="utf-8"), str(SCRIPT), "exec")


def test_inventory_classifies_canonical_files(tmp_path: Path) -> None:
    root = _minimal_repo(tmp_path)
    rows, exclusions = m.inventory_release(root)
    roles = {row.relative_path: row.release_role for row in rows}
    assert roles["README.md"] == "root_metadata"
    assert roles["config/phase4_source_contract.yml"] == "configuration"
    assert roles["src/qstable_awm/__init__.py"] == "python_source"
    assert roles["scripts/run.sh"] == "operational_script"
    assert roles["tests/test_sample.py"] == "test"
    assert roles["results/pipeline/result.json"] == "canonical_result"
    assert exclusions == []


def test_exact_old_directories_are_recursively_excluded(tmp_path: Path) -> None:
    root = _minimal_repo(tmp_path)
    (root / "OLD" / "nested").mkdir(parents=True)
    (root / "OLD" / "nested" / "stale.py").write_text("bad\n")
    (root / "hold").mkdir()
    (root / "hold" / "active.txt").write_text("active\n")
    rows, exclusions = m.inventory_release(root)
    paths = {row.relative_path for row in rows}
    assert "hold/active.txt" in paths
    assert not any(path.startswith("OLD/") for path in paths)
    assert any(item.reason == "historical_old_directory" for item in exclusions)


@pytest.mark.parametrize(
    "relative",
    [
        ".pytest_cache/cache",
        "src/qstable_awm/__pycache__/module.pyc",
        "build/artifact.txt",
        "dist/package.whl",
        "src/qstable_awm.egg-info/PKG-INFO",
        ".ipynb_checkpoints/a.txt",
        ".phase4-pipeline-staging-abc/file",
        ".phase4-pipeline-backup-abc/file",
        "validation_reports/report.json",
        "scratch.tmp",
    ],
)
def test_cache_build_and_scratch_entries_are_excluded(tmp_path: Path, relative: str) -> None:
    root = _minimal_repo(tmp_path)
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("excluded\n")
    rows, exclusions = m.inventory_release(root)
    assert relative not in {row.relative_path for row in rows}
    assert exclusions


def test_phase3_directory_and_archives_are_excluded(tmp_path: Path) -> None:
    root = _minimal_repo(tmp_path)
    phase3 = root / m.PHASE3_NAME
    phase3.mkdir()
    (phase3 / "VERSION").write_text("002\n")
    (root / "PHASE3_CURA.zip").write_bytes(b"archive")
    rows, exclusions = m.inventory_release(root)
    assert not any(m.PHASE3_NAME in row.relative_path for row in rows)
    assert "PHASE3_CURA.zip" not in {row.relative_path for row in rows}
    assert sum(item.reason == "phase3_distributed_separately" for item in exclusions) == 2


def test_symlink_is_rejected(tmp_path: Path) -> None:
    root = _minimal_repo(tmp_path)
    target = root / "README.md"
    (root / "linked.md").symlink_to(target)
    with pytest.raises(m.ReleaseManifestError, match="Symlink"):
        m.inventory_release(root)


def test_case_insensitive_duplicate_paths_are_rejected(tmp_path: Path) -> None:
    root = _minimal_repo(tmp_path)
    (root / "A.txt").write_text("A\n")
    (root / "a.txt").write_text("a\n")
    with pytest.raises(m.ReleaseManifestError, match="Case-insensitive duplicate"):
        m.inventory_release(root)


def test_wrong_root_name_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "wrong-name"
    root.mkdir()
    with pytest.raises(m.ReleaseManifestError, match="must be named"):
        m.inventory_release(root)


def test_build_writes_three_release_metadata_files(tmp_path: Path) -> None:
    root = _minimal_repo(tmp_path)
    hashes = m.build_release_metadata(root, _args())
    assert set(hashes) == m.GENERATED_RELEASE_PATHS
    assert all((root / relative).is_file() for relative in hashes)
    rows = _read_manifest(root)
    assert len(rows) == 6


def test_generated_metadata_are_self_excluded_and_documented(tmp_path: Path) -> None:
    root = _minimal_repo(tmp_path)
    m.build_release_metadata(root, _args())
    rows, exclusions = m.inventory_release(root)
    assert not (m.GENERATED_RELEASE_PATHS & {row.relative_path for row in rows})
    assert m.GENERATED_RELEASE_PATHS <= {item.relative_path for item in exclusions}


def test_refuses_overwrite_without_explicit_flag(tmp_path: Path) -> None:
    root = _minimal_repo(tmp_path)
    m.build_release_metadata(root, _args())
    with pytest.raises(m.ReleaseManifestError, match="Refusing to overwrite"):
        m.build_release_metadata(root, _args())


def test_overwrite_is_byte_deterministic(tmp_path: Path) -> None:
    root = _minimal_repo(tmp_path)
    m.build_release_metadata(root, _args())
    first = {
        relative: (root / relative).read_bytes()
        for relative in m.GENERATED_RELEASE_PATHS
    }
    m.build_release_metadata(root, _args(overwrite=True))
    second = {
        relative: (root / relative).read_bytes()
        for relative in m.GENERATED_RELEASE_PATHS
    }
    assert first == second
    assert not list((root / "release").glob(".release-manifest-staging-*"))


def test_manifest_records_size_hash_and_executable_permission(tmp_path: Path) -> None:
    root = _minimal_repo(tmp_path)
    m.build_release_metadata(root, _args())
    by_path = {row["relative_path"]: row for row in _read_manifest(root)}
    runner = root / "scripts" / "run.sh"
    row = by_path["scripts/run.sh"]
    assert int(row["size_bytes"]) == runner.stat().st_size
    assert row["sha256"] == hashlib.sha256(runner.read_bytes()).hexdigest()
    assert row["executable"] == "true"


def test_summary_records_authority_and_verified_status(tmp_path: Path) -> None:
    root = _minimal_repo(tmp_path)
    args = _args(
        tests_collected=10,
        tests_passed=10,
        tests_failed=0,
        runtime_checks=1708,
        runtime_passed=1708,
        runtime_warnings=0,
        runtime_failed=0,
        pipeline_status="PASS",
        canonical_result_files=41,
        results_tree_sha256="a" * 64,
        clean_room_status="PASS",
    )
    m.build_release_metadata(root, args)
    summary = json.loads((root / m.SUMMARY_RELATIVE_PATH).read_text())
    assert summary["package_version"] == "1.0.0"
    assert summary["verified_status"]["tests"]["status"] == "PASS"
    assert summary["verified_status"]["runtime_validation"]["checks"] == 1708
    assert summary["verified_status"]["pipeline"]["canonical_result_files"] == 41
    assert summary["scientific_boundaries"]["phase3_bundled"] is False


def test_partial_verification_status_is_rejected(tmp_path: Path) -> None:
    root = _minimal_repo(tmp_path)
    with pytest.raises(m.ReleaseManifestError, match="Test status requires"):
        m.build_release_metadata(root, _args(tests_collected=10))


def test_sha256sums_covers_inventory_manifest_and_summary(tmp_path: Path) -> None:
    root = _minimal_repo(tmp_path)
    m.build_release_metadata(root, _args())
    lines = (root / m.CHECKSUMS_RELATIVE_PATH).read_text().splitlines()
    parsed = {line.split("  ", 1)[1]: line.split("  ", 1)[0] for line in lines}
    manifest_paths = {row["relative_path"] for row in _read_manifest(root)}
    assert set(parsed) == manifest_paths | {
        m.MANIFEST_RELATIVE_PATH,
        m.SUMMARY_RELATIVE_PATH,
    }
    for relative, digest in parsed.items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == digest


def test_cli_reports_pass_and_error_exit_codes(tmp_path: Path) -> None:
    root = _minimal_repo(tmp_path)
    passed = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert passed.returncode == 0
    assert json.loads(passed.stdout)["status"] == "PASS"
    failed = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert failed.returncode == 2
    assert "QSTABLE_RELEASE_MANIFEST_ERROR" in failed.stderr
