"""Release-regression tests for :mod:`scripts/run_pipeline.sh`.

The shell runner is the public operational wrapper around the validated Phase 4
pipeline.  These tests verify argument handling, dry-run and selected-module
execution, complete deterministic publication, overwrite protection, protected
path controls, malformed-contract handling, environment fallback, stable exit
codes, and immutability of the frozen Phase 3 and accepted result trees.

All analytical executions are directed to pytest temporary directories.  The
canonical repository result tree and the frozen Phase 3 package are never
modified by this test module.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml


PHASE4_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PHASE4_ROOT / "scripts" / "run_pipeline.sh"
CONTRACT_PATH = PHASE4_ROOT / "config" / "phase4_source_contract.yml"
REFERENCE_RESULTS_ROOT = PHASE4_ROOT / "results"
PHASE3_ROOT = Path(
    os.environ.get(
        "QSTABLE_PHASE3_ROOT",
        PHASE4_ROOT.parent / "PHASE3_CURATED_INPUT_LAYER_FINAL_002",
    )
).expanduser().resolve()

EXPECTED_SCRIPT_SHA256 = (
    "5f18b0ea11c649a0ab7413a2ad444cde61b0e62448dec3d11af4370ed88300b5"
)
EXPECTED_CONTRACT_SHA256 = (
    "b701177f25c7c758e50880bbf63901ba95d0d1e4bc7dc03d7af4038e4e8ff725"
)
EXPECTED_PHASE3_TREE_SHA256 = (
    "b74edffd081bba5962cda93f9696fa43d4ceab42028ac1db65c996556360421f"
)
EXPECTED_RESULTS_TREE_SHA256 = (
    "9da2eb226a47c8633a5bf43afbdef48625026501e3ec78402627293091a40138"
)
PUBLIC_MODULES = (
    "climate-context",
    "physical-coherence",
    "aerodynamic-sensitivity",
    "crop-support",
    "fixed-frame-dominance",
    "known-cold-reference",
    "interpretation-classes",
)
INTERNAL_MODULES = tuple(value.replace("-", "_") for value in PUBLIC_MODULES)
EXPECTED_OUTPUT_COUNTS = {
    "climate_context": 5,
    "physical_coherence": 6,
    "aerodynamic_sensitivity": 6,
    "crop_support": 5,
    "fixed_frame_dominance": 5,
    "known_cold_reference": 6,
    "interpretation_classes": 5,
}
PIPELINE_FILES = {
    "phase4_pipeline_output_manifest.csv",
    "phase4_pipeline_validation.json",
    "phase4_pipeline_execution_record.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_hashes(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _tree_digest(root: Path) -> str:
    rows = [
        f"{relative}\0{(root / relative).stat().st_size}\0{digest}\n"
        for relative, digest in _file_hashes(root).items()
    ]
    return hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()




def _portable_execution_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    def normalize(value):
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if isinstance(value, str) and value.endswith("/config/phase4_source_contract.yml"):
            return "<PHASE4_ROOT>/config/phase4_source_contract.yml"
        return value

    return normalize(payload)


def _portable_scientific_hashes(root: Path) -> dict[str, str]:
    return {
        relative: digest
        for relative, digest in _file_hashes(root).items()
        if not relative.endswith("_execution_record.json")
        and not relative.startswith("pipeline/")
    }

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


def _manifest_rows(results_root: Path) -> list[dict[str, str]]:
    path = results_root / "pipeline" / "phase4_pipeline_output_manifest.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


@pytest.fixture(scope="session", autouse=True)
def canonical_trees_remain_immutable():
    assert PHASE3_ROOT.is_dir()
    assert REFERENCE_RESULTS_ROOT.is_dir()
    phase3_before = _file_hashes(PHASE3_ROOT)
    results_before = _file_hashes(REFERENCE_RESULTS_ROOT)
    yield
    assert _file_hashes(PHASE3_ROOT) == phase3_before
    assert _file_hashes(REFERENCE_RESULTS_ROOT) == results_before


@pytest.fixture(scope="session")
def dry_run_result(tmp_path_factory: pytest.TempPathFactory):
    results_root = tmp_path_factory.mktemp("pipeline_script_dry") / "not-created"
    completed = _run_script(
        "--phase3-root",
        PHASE3_ROOT,
        "--results-root",
        results_root,
        "--dry-run",
        "--python",
        sys.executable,
    )
    return completed, results_root


@pytest.fixture(scope="session")
def subset_run(tmp_path_factory: pytest.TempPathFactory):
    results_root = tmp_path_factory.mktemp("pipeline_script_subset") / "results"
    completed = _run_script(
        "--phase3-root",
        PHASE3_ROOT,
        "--results-root",
        results_root,
        "--module",
        "fixed-frame-dominance",
        "--module",
        "crop-support",
    )
    return completed, results_root


@pytest.fixture(scope="session")
def full_run(tmp_path_factory: pytest.TempPathFactory):
    results_root = tmp_path_factory.mktemp("pipeline_script_full") / "results"
    completed = _run_script(
        "--phase3-root",
        PHASE3_ROOT,
        "--results-root",
        results_root,
    )
    return completed, results_root


# ---------------------------------------------------------------------------
# Static authority and shell structure
# ---------------------------------------------------------------------------


def test_script_exists_is_executable_and_hash_frozen() -> None:
    assert SCRIPT_PATH.is_file()
    assert os.access(SCRIPT_PATH, os.X_OK)
    assert _sha256(SCRIPT_PATH) == EXPECTED_SCRIPT_SHA256


def test_active_contract_is_authoritative() -> None:
    assert _sha256(CONTRACT_PATH) == EXPECTED_CONTRACT_SHA256
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert contract["contract_version"] == "1.4.0"


def test_bash_syntax_is_valid() -> None:
    completed = subprocess.run(
        ["bash", "-n", str(SCRIPT_PATH)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_script_uses_strict_shell_and_no_bytecode() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert source.startswith("#!/usr/bin/env bash\n")
    assert "set -Eeuo pipefail" in source
    assert "IFS=$'\\n\\t'" in source
    assert "export PYTHONDONTWRITEBYTECODE=1" in source


def test_script_invokes_pipeline_not_notebooks_or_network_tools() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    lowered = source.lower()
    assert "-m qstable_awm.pipeline" in source
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
    assert not any(token in lowered for token in prohibited)


def test_public_module_allowlist_and_internal_mapping_are_exact() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    for public, internal in zip(PUBLIC_MODULES, INTERNAL_MODULES, strict=True):
        assert public in source
        assert internal in source
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert tuple(contract["module_inputs"]) == INTERNAL_MODULES


def test_script_documents_stable_exit_codes_and_operational_markers() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "QSTABLE_PIPELINE_BEGIN" in source
    assert "QSTABLE_PIPELINE_END status=PASS exit_code=0" in source
    assert "QSTABLE_PIPELINE_END status=FAIL" in source
    assert "0  Dry run or pipeline execution passed." in source
    assert "2  Invalid arguments" in source


def test_script_exports_repository_local_pythonpath() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert 'export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"' in source


# ---------------------------------------------------------------------------
# Public help and dry-run behavior
# ---------------------------------------------------------------------------


def test_help_succeeds_and_documents_interface() -> None:
    completed = _run_script("--help")
    assert completed.returncode == 0
    assert completed.stderr == ""
    for option in (
        "--phase3-root",
        "--contract",
        "--results-root",
        "--module",
        "--dry-run",
        "--overwrite",
        "--python",
    ):
        assert option in completed.stdout
    assert "QSTABLE_PIPELINE_BEGIN" not in completed.stdout


def test_complete_dry_run_passes_full_preflight(dry_run_result) -> None:
    completed, _ = dry_run_result
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = _json_payload(completed.stdout)
    assert payload["dry_run"] is True
    assert payload["selected_modules"] == list(INTERNAL_MODULES)
    assert payload["preflight_validation"]["status"] == "PASS"
    assert payload["preflight_validation"]["checks"] == 1_708
    assert payload["preflight_validation"]["passed"] == 1_708
    assert payload["preflight_validation"]["failed"] == 0
    assert len(payload["plan"]) == 7


def test_dry_run_writes_nothing(dry_run_result) -> None:
    completed, results_root = dry_run_result
    assert completed.returncode == 0
    assert not results_root.exists()
    assert "MODE=DRY_RUN" in completed.stdout
    assert "MODULES=ALL" in completed.stdout
    assert "QSTABLE_PIPELINE_END status=PASS exit_code=0" in completed.stdout


def test_environment_phase3_fallback_works(tmp_path: Path) -> None:
    results_root = tmp_path / "fallback-results"
    completed = _run_script(
        "--results-root",
        results_root,
        "--dry-run",
        env=_environment(QSTABLE_PHASE3_ROOT=str(PHASE3_ROOT)),
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert f"PHASE3_ROOT={PHASE3_ROOT}" in completed.stdout
    assert not results_root.exists()


def test_custom_python_interpreter_is_reported(dry_run_result) -> None:
    completed, _ = dry_run_result
    assert f"PYTHON={Path(sys.executable).resolve()}" in completed.stdout


# ---------------------------------------------------------------------------
# Selected-module and complete execution
# ---------------------------------------------------------------------------


def test_subset_execution_reorders_modules_into_frozen_order(subset_run) -> None:
    completed, _ = subset_run
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = _json_payload(completed.stdout)
    assert payload["selected_modules"] == ["crop_support", "fixed_frame_dominance"]
    assert [item["module"] for item in payload["plan"]] == [
        "crop_support",
        "fixed_frame_dominance",
    ]


def test_subset_execution_writes_exact_13_file_tree(subset_run) -> None:
    _, results_root = subset_run
    files = _file_hashes(results_root)
    assert len(files) == 13
    assert {path.split("/", 1)[0] for path in files} == {
        "crop_support",
        "fixed_frame_dominance",
        "pipeline",
    }
    assert set((results_root / "pipeline").iterdir()) == {
        results_root / "pipeline" / name for name in PIPELINE_FILES
    }


def test_subset_manifest_has_ten_authorized_rows(subset_run) -> None:
    _, results_root = subset_run
    rows = _manifest_rows(results_root)
    assert len(rows) == 10
    assert {row["module"] for row in rows} == {
        "crop_support",
        "fixed_frame_dominance",
    }
    assert len({row["relative_path"] for row in rows}) == 10
    assert all(len(row["sha256"]) == 64 for row in rows)


def test_complete_execution_writes_41_files(full_run) -> None:
    completed, results_root = full_run
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert len(_file_hashes(results_root)) == 41
    payload = _json_payload(completed.stdout)
    assert payload["dry_run"] is False
    assert payload["selected_modules"] == list(INTERNAL_MODULES)
    assert set(payload["outputs"]) == {
        "phase4_pipeline_output_manifest",
        "phase4_pipeline_validation",
        "phase4_pipeline_execution_record",
    }
    assert all(Path(value).is_file() for value in payload["outputs"].values())


def test_complete_execution_matches_canonical_scientific_outputs(full_run) -> None:
    _, results_root = full_run
    assert _portable_scientific_hashes(results_root) == _portable_scientific_hashes(
        REFERENCE_RESULTS_ROOT
    )
    for module in INTERNAL_MODULES:
        name = f"{module}_execution_record.json"
        assert _portable_execution_payload(results_root / module / name) == (
            _portable_execution_payload(REFERENCE_RESULTS_ROOT / module / name)
        )


def test_complete_manifest_reconciles_contract_output_count(full_run) -> None:
    _, results_root = full_run
    rows = _manifest_rows(results_root)
    assert len(rows) == sum(EXPECTED_OUTPUT_COUNTS.values()) == 38
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["module"]] = counts.get(row["module"], 0) + 1
        output_path = results_root / row["relative_path"]
        assert output_path.is_file()
        assert int(row["size_bytes"]) == output_path.stat().st_size
        assert row["sha256"] == _sha256(output_path)
    assert counts == EXPECTED_OUTPUT_COUNTS


def test_complete_pipeline_validation_and_execution_record_pass(full_run) -> None:
    _, results_root = full_run
    validation = json.loads(
        (results_root / "pipeline" / "phase4_pipeline_validation.json").read_text(
            encoding="utf-8"
        )
    )
    record = json.loads(
        (results_root / "pipeline" / "phase4_pipeline_execution_record.json").read_text(
            encoding="utf-8"
        )
    )
    assert validation["status"] == "PASS"
    assert validation["checks"] == validation["passed"] == 10
    assert validation["failed"] == 0
    assert record["selected_modules"] == list(INTERNAL_MODULES)
    assert record["phase3_tree_sha256"] == EXPECTED_PHASE3_TREE_SHA256
    assert record["contract_sha256"] == EXPECTED_CONTRACT_SHA256


# ---------------------------------------------------------------------------
# Overwrite, invalid arguments, protected paths, and failures
# ---------------------------------------------------------------------------


def test_existing_targets_require_explicit_overwrite(subset_run) -> None:
    _, results_root = subset_run
    before = _file_hashes(results_root)
    completed = _run_script(
        "--phase3-root",
        PHASE3_ROOT,
        "--results-root",
        results_root,
        "--module",
        "crop-support",
        "--module",
        "fixed-frame-dominance",
    )
    assert completed.returncode == 2
    assert "QSTABLE_PIPELINE_END status=FAIL exit_code=2" in completed.stderr
    assert _file_hashes(results_root) == before


def test_explicit_overwrite_is_byte_deterministic(subset_run) -> None:
    _, results_root = subset_run
    before = _file_hashes(results_root)
    completed = _run_script(
        "--phase3-root",
        PHASE3_ROOT,
        "--results-root",
        results_root,
        "--module",
        "fixed-frame-dominance",
        "--module",
        "crop-support",
        "--overwrite",
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert _file_hashes(results_root) == before


@pytest.mark.parametrize(
    "arguments,message",
    [
        (("--module", "unknown-module"), "Unknown module"),
        (
            ("--module", "crop-support", "--module", "crop-support"),
            "Duplicate module",
        ),
        (("--unknown-option",), "Unknown option"),
    ],
)
def test_invalid_arguments_return_stable_exit_code_two(
    arguments: tuple[str, ...], message: str
) -> None:
    completed = _run_script("--phase3-root", PHASE3_ROOT, *arguments)
    assert completed.returncode == 2
    assert "QSTABLE_PIPELINE_SCRIPT_ERROR:" in completed.stderr
    assert message in completed.stderr


@pytest.mark.parametrize(
    "unsafe_root",
    [
        PHASE3_ROOT / "pipeline-results",
        PHASE4_ROOT,
        PHASE4_ROOT / "src" / "pipeline-results",
        PHASE4_ROOT / "config" / "pipeline-results",
        PHASE4_ROOT / "tests" / "pipeline-results",
        PHASE4_ROOT / "scripts" / "pipeline-results",
    ],
)
def test_unsafe_results_roots_are_rejected_before_creation(unsafe_root: Path) -> None:
    existed_before = unsafe_root.exists()
    completed = _run_script(
        "--phase3-root",
        PHASE3_ROOT,
        "--results-root",
        unsafe_root,
        "--dry-run",
    )
    assert completed.returncode == 2
    assert "Results root cannot" in completed.stderr
    if not existed_before:
        assert not unsafe_root.exists()


def test_malformed_contract_is_rejected_before_output_creation(tmp_path: Path) -> None:
    bad_contract = tmp_path / "bad.yml"
    bad_contract.write_text("module_inputs: [\n", encoding="utf-8")
    results_root = tmp_path / "results"
    completed = _run_script(
        "--phase3-root",
        PHASE3_ROOT,
        "--contract",
        bad_contract,
        "--results-root",
        results_root,
    )
    assert completed.returncode == 2
    assert "Source contract must be a readable YAML mapping" in completed.stderr
    assert not results_root.exists()


def test_structurally_invalid_contract_fails_without_publication(tmp_path: Path) -> None:
    bad_contract = tmp_path / "incomplete.yml"
    bad_contract.write_text("contract_version: '1.4.0'\n", encoding="utf-8")
    results_root = tmp_path / "results"
    completed = _run_script(
        "--phase3-root",
        PHASE3_ROOT,
        "--contract",
        bad_contract,
        "--results-root",
        results_root,
    )
    assert completed.returncode == 2
    assert "QSTABLE_PIPELINE_END status=FAIL exit_code=2" in completed.stderr
    assert not _file_hashes(results_root)


def test_missing_python_interpreter_is_rejected() -> None:
    completed = _run_script(
        "--phase3-root",
        PHASE3_ROOT,
        "--python",
        "definitely-not-a-python-interpreter",
        "--dry-run",
    )
    assert completed.returncode == 2
    assert "Python interpreter not found" in completed.stderr


def test_wrong_phase3_directory_name_is_rejected(tmp_path: Path) -> None:
    wrong = tmp_path / "not-the-phase3-package"
    wrong.mkdir()
    completed = _run_script("--phase3-root", wrong, "--dry-run")
    assert completed.returncode == 2
    assert "Phase 3 root must be named" in completed.stderr


# ---------------------------------------------------------------------------
# Canonical integrity and cleanup
# ---------------------------------------------------------------------------


def test_canonical_phase3_and_result_tree_hashes_are_frozen() -> None:
    assert len(_file_hashes(PHASE3_ROOT)) == 172
    assert _tree_digest(PHASE3_ROOT) == EXPECTED_PHASE3_TREE_SHA256
    assert len(_file_hashes(REFERENCE_RESULTS_ROOT)) == 41
    assert _tree_digest(REFERENCE_RESULTS_ROOT) == EXPECTED_RESULTS_TREE_SHA256


def test_repository_contains_no_build_or_runner_staging_artifacts() -> None:
    assert not (PHASE4_ROOT / "build").exists()
    assert not (PHASE4_ROOT / "dist").exists()
    assert not (PHASE4_ROOT / "src" / "qstable_awm.egg-info").exists()
    assert not list(PHASE4_ROOT.glob(".qstable-pipeline.*"))
    assert not list(PHASE4_ROOT.glob(".qstable-pipeline-backup.*"))
