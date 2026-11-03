"""Release-regression tests for :mod:`scripts/run_validation.sh`.

The shell runner is a validation-only public entry point.  These tests verify
its argument surface, Phase 3 and results-tree immutability, deterministic
report publication, protected-path controls, Python selection, stable exit
codes, and failure cleanup.  They do not execute the analytical pipeline or
alter accepted scientific outputs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml


PHASE4_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PHASE4_ROOT / "scripts" / "run_validation.sh"
CONTRACT_PATH = PHASE4_ROOT / "config" / "phase4_source_contract.yml"
RESULTS_ROOT = PHASE4_ROOT / "results"
PHASE3_ROOT = Path(
    os.environ.get(
        "QSTABLE_PHASE3_ROOT",
        PHASE4_ROOT.parent / "PHASE3_CURATED_INPUT_LAYER_FINAL_002",
    )
).expanduser().resolve()

EXPECTED_SCRIPT_SHA256 = (
    "d95efffa7d32ff6560010defac8f8fad348948ae0cd630e1595188962f3c2e19"
)
EXPECTED_CONTRACT_SHA256 = (
    "b701177f25c7c758e50880bbf63901ba95d0d1e4bc7dc03d7af4038e4e8ff725"
)
EXPECTED_MODULES = (
    "climate-context",
    "physical-coherence",
    "aerodynamic-sensitivity",
    "crop-support",
    "fixed-frame-dominance",
    "known-cold-reference",
    "interpretation-classes",
)
EXPECTED_REPORT_NAMES = {
    "phase4_runtime_validation.json",
    "phase4_runtime_validation.csv",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_inventory(root: Path) -> tuple[tuple[str, int, str], ...]:
    return tuple(
        (str(path.relative_to(root)), path.stat().st_size, _sha256(path))
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    )


def _environment(**updates: str | None) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        }
    )
    for key, value in updates.items():
        if value is None:
            environment.pop(key, None)
        else:
            environment[key] = value
    return environment


def _run_script(
    *arguments: str | Path,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    timeout: int = 180,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(SCRIPT_PATH), *(str(value) for value in arguments)],
        cwd=cwd or PHASE4_ROOT,
        env=env or _environment(),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _json_payload(stdout: str) -> dict[str, Any]:
    start = stdout.find("{")
    assert start >= 0, stdout
    payload, _ = json.JSONDecoder().raw_decode(stdout[start:])
    assert isinstance(payload, dict)
    return payload


def _report_payload(report_dir: Path) -> dict[str, Any]:
    return json.loads(
        (report_dir / "phase4_runtime_validation.json").read_text(encoding="utf-8")
    )





@pytest.fixture(scope="session", autouse=True)
def protected_trees_remain_immutable():
    assert PHASE3_ROOT.is_dir()
    phase3_before = _tree_inventory(PHASE3_ROOT)
    results_before = _tree_inventory(RESULTS_ROOT)
    yield
    assert _tree_inventory(PHASE3_ROOT) == phase3_before
    assert _tree_inventory(RESULTS_ROOT) == results_before



@pytest.fixture(scope="session")
def crop_hash_reports(tmp_path_factory: pytest.TempPathFactory):
    report_dir = tmp_path_factory.mktemp("run_validation_crop_hash") / "reports"
    completed = _run_script(
        "--phase3-root",
        PHASE3_ROOT,
        "--module",
        "crop-support",
        "--hashes-only",
        "--report-dir",
        report_dir,
    )
    return completed, report_dir


# ---------------------------------------------------------------------------
# Static authority and shell structure
# ---------------------------------------------------------------------------


def test_script_exists_and_is_executable() -> None:
    assert SCRIPT_PATH.is_file()
    assert os.access(SCRIPT_PATH, os.X_OK)


def test_script_hash_is_frozen() -> None:
    assert _sha256(SCRIPT_PATH) == EXPECTED_SCRIPT_SHA256


def test_active_contract_hash_is_authoritative() -> None:
    assert _sha256(CONTRACT_PATH) == EXPECTED_CONTRACT_SHA256


def test_bash_syntax_is_valid() -> None:
    completed = subprocess.run(
        ["bash", "-n", str(SCRIPT_PATH)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_script_uses_strict_shell_settings() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert source.startswith("#!/usr/bin/env bash\n")
    assert "set -Eeuo pipefail" in source
    assert "IFS=$'\\n\\t'" in source
    assert "trap cleanup EXIT INT TERM" in source


def test_script_is_validation_only() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "qstable_awm.cli validate" in source
    assert "run-all" not in source
    assert "qstable_awm.pipeline" not in source
    assert "--overwrite" not in source


def test_script_has_no_network_or_notebook_execution_surface() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8").lower()
    prohibited = (
        "curl ",
        "wget ",
        "git clone",
        ".ipynb",
        "jupyter",
        "papermill",
        "/scratch/user/",
        "/content/drive/",
        "eval ",
    )
    assert not any(token in source for token in prohibited)


def test_public_module_allowlist_is_exact() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    for module in EXPECTED_MODULES:
        assert module in source
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    public_from_contract = tuple(name.replace("_", "-") for name in contract["module_inputs"])
    assert public_from_contract == EXPECTED_MODULES


def test_script_exports_repository_local_pythonpath_without_bytecode() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert 'export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"' in source
    assert "export PYTHONDONTWRITEBYTECODE=1" in source


def test_default_report_names_are_deterministic() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "phase4_runtime_validation.json" in source
    assert "phase4_runtime_validation.csv" in source
    assert "validation_reports" in source


# ---------------------------------------------------------------------------
# Help, full validation, reports, and deterministic behavior
# ---------------------------------------------------------------------------


def test_help_succeeds_and_documents_public_interface() -> None:
    completed = _run_script("--help")
    assert completed.returncode == 0
    assert completed.stderr == ""
    for option in (
        "--phase3-root",
        "--contract",
        "--module",
        "--hashes-only",
        "--report-dir",
        "--no-reports",
        "--python",
    ):
        assert option in completed.stdout
    assert "Exit codes:" in completed.stdout
    assert "QSTABLE_VALIDATION_BEGIN" not in completed.stdout





def test_module_report_run_publishes_exact_files(crop_hash_reports) -> None:
    completed, report_dir = crop_hash_reports
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert {path.name for path in report_dir.iterdir()} == EXPECTED_REPORT_NAMES
    assert f"VALIDATION_JSON={report_dir / 'phase4_runtime_validation.json'}" in completed.stdout
    assert f"VALIDATION_CSV={report_dir / 'phase4_runtime_validation.csv'}" in completed.stdout


def test_module_json_report_is_complete(crop_hash_reports) -> None:
    _, report_dir = crop_hash_reports
    payload = _report_payload(report_dir)
    assert payload["status"] == "PASS"
    assert payload["checks"] == payload["passed"] == 21
    assert payload["warnings"] == payload["failed"] == 0
    assert payload["modules"] == ["crop_support"]
    assert len(payload["check_results"]) == 21
    assert len({item["check_id"] for item in payload["check_results"]}) == 21


def test_module_csv_report_is_complete(crop_hash_reports) -> None:
    _, report_dir = crop_hash_reports
    with (report_dir / "phase4_runtime_validation.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 21
    assert set(rows[0]) == {
        "check_id",
        "scope",
        "description",
        "severity",
        "status",
        "observed",
        "expected",
        "relative_path",
    }
    assert all(row["status"] == "PASS" for row in rows)



def test_success_leaves_no_staging_directory(crop_hash_reports) -> None:
    _, report_dir = crop_hash_reports
    assert not list(report_dir.glob(".qstable-validation.*"))


# ---------------------------------------------------------------------------
# Module selection, hash-only mode, and path/interpreter resolution
# ---------------------------------------------------------------------------




def test_module_selection_is_forwarded_to_the_public_cli() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert 'COMMAND+=(--module "${module}")' in source
    assert '--module NAME' in source


def test_report_path_is_checked_before_directory_creation() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert source.index('Report directory cannot be inside protected tree') < source.index('mkdir -p -- "${REPORT_DIR}"')


def test_hash_only_mode_is_reported(crop_hash_reports) -> None:
    completed, _ = crop_hash_reports
    payload = _json_payload(completed.stdout)
    assert payload["hashes_only"] is True
    assert payload["summary"]["checks"] == payload["summary"]["passed"] == 21
    assert payload["summary"]["failed"] == 0
    assert "MODE=HASHES_ONLY" in completed.stdout


def test_environment_and_error_branches_are_present() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    for token in (
        'QSTABLE_PYTHON:-python',
        'QSTABLE_PHASE3_ROOT',
        'Directory not found:',
        'File not found:',
        'Python interpreter not found:',
        'Python 3.11 or newer is required:',
        'Phase 3 root must be named',
        'Unknown module:',
        'Duplicate module names were requested',
    ):
        # Duplicate-module rejection belongs to the delegated CLI, while all
        # other branches are implemented directly by the runner.
        if token == 'Duplicate module names were requested':
            assert token in (PHASE4_ROOT / 'src' / 'qstable_awm' / 'validation.py').read_text(encoding='utf-8')
        else:
            assert token in source



# ---------------------------------------------------------------------------
# Argument errors and protected report paths
# ---------------------------------------------------------------------------


def test_representative_argument_errors_return_two() -> None:
    cases = (
        (("--module", "not-a-module"), "Unknown module"),
        (("--unknown-option",), "Unknown option"),
        (("--phase3-root",), "requires a path"),
    )
    for arguments, expected in cases:
        completed = _run_script(*arguments)
        assert completed.returncode == 2
        assert completed.stdout == ""
        assert "QSTABLE_VALIDATION_SCRIPT_ERROR:" in completed.stderr
        assert expected in completed.stderr






def test_all_canonical_trees_are_statically_protected() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    for expression in (
        '"${PHASE3_ROOT}"',
        '"${REPO_ROOT}/src"',
        '"${REPO_ROOT}/config"',
        '"${REPO_ROOT}/tests"',
        '"${REPO_ROOT}/scripts"',
        '"${REPO_ROOT}/results"',
    ):
        assert expression in source



def _contract_with_bad_runtime_hash(destination: Path) -> Path:
    payload = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    payload["module_inputs"]["crop_support"]["required_files"][0]["sha256"] = "0" * 64
    destination.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return destination


def test_validation_failure_returns_one_and_preserves_existing_reports(
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    json_report = report_dir / "phase4_runtime_validation.json"
    csv_report = report_dir / "phase4_runtime_validation.csv"
    json_report.write_bytes(b"previous-json\n")
    csv_report.write_bytes(b"previous-csv\n")
    contract = _contract_with_bad_runtime_hash(tmp_path / "bad-hash-contract.yml")

    completed = _run_script(
        "--phase3-root",
        PHASE3_ROOT,
        "--contract",
        contract,
        "--module",
        "crop-support",
        "--report-dir",
        report_dir,
    )

    assert completed.returncode == 1
    assert "QSTABLE_VALIDATION_END status=FAIL exit_code=1" in completed.stderr
    assert json_report.read_bytes() == b"previous-json\n"
    assert csv_report.read_bytes() == b"previous-csv\n"
    assert not list(report_dir.glob(".qstable-validation.*"))


def test_runtime_contract_error_returns_two_without_publishing_reports(
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "reports"
    malformed = tmp_path / "malformed.yml"
    malformed.write_text("contract_version: [\n", encoding="utf-8")
    completed = _run_script(
        "--phase3-root",
        PHASE3_ROOT,
        "--contract",
        malformed,
        "--report-dir",
        report_dir,
    )
    assert completed.returncode == 2
    assert "QSTABLE_VALIDATION_SCRIPT_ERROR:" in completed.stderr
    assert "Source contract must be a readable YAML mapping" in completed.stderr
    assert not (report_dir / "phase4_runtime_validation.json").exists()
    assert not (report_dir / "phase4_runtime_validation.csv").exists()
    assert not list(report_dir.glob(".qstable-validation.*"))



def test_script_does_not_create_repository_build_artifacts() -> None:
    assert not (PHASE4_ROOT / "build").exists()
    assert not (PHASE4_ROOT / "dist").exists()
    assert not (PHASE4_ROOT / "src" / "qstable_awm.egg-info").exists()
