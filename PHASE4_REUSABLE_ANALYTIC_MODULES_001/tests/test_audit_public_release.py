"""Tests for the final public-release integrity audit."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PHASE4_ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = PHASE4_ROOT / "scripts" / "audit_public_release.py"
BUILD_SCRIPT = PHASE4_ROOT / "scripts" / "build_release_manifest.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


b = _load("test_build_release_manifest_module", BUILD_SCRIPT)
a = _load("test_audit_public_release_module", AUDIT_SCRIPT)


def _repo(tmp_path: Path, *, verified: bool = False) -> Path:
    root = tmp_path / b.EXPECTED_PACKAGE_NAME
    for directory in (
        "config",
        "src/qstable_awm",
        "scripts",
        "tests",
        "results/module",
        "release",
    ):
        (root / directory).mkdir(parents=True, exist_ok=True)
    (root / "config/phase4_source_contract.yml").write_text(
        "contract_version: 1.4.0\n", encoding="utf-8"
    )
    (root / "src/qstable_awm/__init__.py").write_text(
        "from typing import Final\nVERSION_INFO: Final[tuple[int, int, int]] = (1, 0, 0)\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text("# Audit fixture\n", encoding="utf-8")
    runner = root / "scripts/run.sh"
    runner.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    runner.chmod(0o755)
    (root / "tests/test_sample.py").write_text("def test_ok(): assert True\n")
    (root / "results/module/result.csv").write_text("x\n1\n")
    values = dict(
        overwrite=False,
        tests_collected=1 if verified else None,
        tests_passed=1 if verified else None,
        tests_failed=0 if verified else None,
        runtime_checks=1 if verified else None,
        runtime_passed=1 if verified else None,
        runtime_warnings=0 if verified else None,
        runtime_failed=0 if verified else None,
        pipeline_status="PASS" if verified else None,
        canonical_result_files=1 if verified else None,
        results_tree_sha256=a.results_tree_sha256(root / "results") if verified else None,
        clean_room_status="PASS" if verified else None,
    )
    args = type("Args", (), values)()
    b.build_release_metadata(root, args)
    return root


def _audit(root: Path, **kwargs):
    return a.audit_release(
        root,
        expected_results_count=1,
        expected_results_tree_sha256=a.results_tree_sha256(root / "results"),
        expected_contract_sha256=b.sha256_file(root / b.CONTRACT_RELATIVE_PATH),
        **kwargs,
    )


def test_audit_script_is_executable_and_syntax_valid() -> None:
    assert os.access(AUDIT_SCRIPT, os.X_OK)
    compile(AUDIT_SCRIPT.read_text(encoding="utf-8"), str(AUDIT_SCRIPT), "exec")


def test_valid_release_passes_all_checks(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    report = _audit(root)
    assert report["status"] == "PASS"
    assert report["failed"] == 0
    assert report["passed"] == report["checks"]


def test_verified_pass_requirement_accepts_complete_status(tmp_path: Path) -> None:
    root = _repo(tmp_path, verified=True)
    report = _audit(root, require_verified_pass=True)
    assert report["status"] == "PASS"
    verified_check = report["check_results"][-1]
    assert verified_check["check_id"] == "REL-016-VERIFIED-STATUS"
    assert verified_check["status"] == "PASS"


def test_verified_pass_requirement_rejects_missing_status(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    report = _audit(root, require_verified_pass=True)
    assert report["status"] == "FAIL"
    assert any(
        check["check_id"] == "REL-016-VERIFIED-STATUS"
        and check["status"] == "FAIL"
        for check in report["check_results"]
    )


@pytest.mark.parametrize(
    "relative",
    [
        "README.md",
        "src/qstable_awm/__init__.py",
        "results/module/result.csv",
        "scripts/run.sh",
    ],
)
def test_file_mutation_is_detected(tmp_path: Path, relative: str) -> None:
    root = _repo(tmp_path)
    path = root / relative
    path.write_bytes(path.read_bytes() + b"mutation\n")
    report = _audit(root)
    assert report["status"] == "FAIL"
    assert any(
        check["check_id"] in {"REL-003-FILE-METADATA-HASHES", "REL-008-SHA256SUMS-VALUES"}
        and check["status"] == "FAIL"
        for check in report["check_results"]
    )


def test_unmanifested_active_file_is_detected(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    (root / "unexpected.txt").write_text("unexpected\n")
    report = _audit(root)
    assert report["status"] == "FAIL"
    check = next(c for c in report["check_results"] if c["check_id"] == "REL-002-ACTIVE-INVENTORY-EXACT")
    assert "unexpected.txt" in check["observed"]["unmanifested_active"]


def test_cache_and_old_entries_do_not_break_audit(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    (root / ".pytest_cache").mkdir()
    (root / ".pytest_cache/cache").write_text("cache\n")
    (root / "OLD/nested").mkdir(parents=True)
    (root / "OLD/nested/stale.py").write_text("stale\n")
    # Rebuild so the exclusion summary records the stable exclusions.
    b.build_release_metadata(root, type("Args", (), {
        "overwrite": True,
        "tests_collected": None,
        "tests_passed": None,
        "tests_failed": None,
        "runtime_checks": None,
        "runtime_passed": None,
        "runtime_warnings": None,
        "runtime_failed": None,
        "pipeline_status": None,
        "canonical_result_files": None,
        "results_tree_sha256": None,
        "clean_room_status": None,
    })())
    assert _audit(root)["status"] == "PASS"


def test_summary_tampering_is_detected(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    path = root / b.SUMMARY_RELATIVE_PATH
    payload = json.loads(path.read_text())
    payload["scientific_boundaries"]["pumping_validation_claimed"] = True
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    report = _audit(root)
    assert report["status"] == "FAIL"
    assert any(
        check["check_id"] == "REL-012-SCIENTIFIC-BOUNDARIES"
        and check["status"] == "FAIL"
        for check in report["check_results"]
    )


def test_malformed_checksum_file_is_rejected(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    (root / b.CHECKSUMS_RELATIVE_PATH).write_text("malformed\n")
    with pytest.raises(a.builder.ReleaseManifestError, match="Malformed"):
        _audit(root)


def test_missing_release_metadata_is_rejected(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    (root / b.MANIFEST_RELATIVE_PATH).unlink()
    with pytest.raises(a.builder.ReleaseManifestError, match="missing"):
        _audit(root)


def test_atomic_json_report_write(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    report = _audit(root)
    output = tmp_path / "audit.json"
    a._atomic_write_json(output, report)
    assert json.loads(output.read_text())["status"] == "PASS"
    assert not list(tmp_path.glob(".audit.json.*.tmp"))


def test_cli_pass_fail_and_error_exit_codes(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    expected_tree = a.results_tree_sha256(root / "results")
    expected_contract = b.sha256_file(root / b.CONTRACT_RELATIVE_PATH)
    base = [
        sys.executable,
        str(AUDIT_SCRIPT),
        "--root",
        str(root),
        "--expected-results-count",
        "1",
        "--expected-results-tree-sha256",
        expected_tree,
        "--expected-contract-sha256",
        expected_contract,
    ]
    passed = subprocess.run(base, capture_output=True, text=True, check=False)
    assert passed.returncode == 0
    (root / "README.md").write_text("changed\n")
    failed = subprocess.run(base, capture_output=True, text=True, check=False)
    assert failed.returncode == 1
    missing = subprocess.run(
        [sys.executable, str(AUDIT_SCRIPT), "--root", str(tmp_path / "missing")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing.returncode == 2
    assert "QSTABLE_PUBLIC_RELEASE_AUDIT_ERROR" in missing.stderr
