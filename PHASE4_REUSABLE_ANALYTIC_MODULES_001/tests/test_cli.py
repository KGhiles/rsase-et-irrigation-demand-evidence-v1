"""Regression and release-engineering tests for the stable Phase 4 CLI.

The tests exercise :mod:`qstable_awm.cli` against the complete frozen
``PHASE3_CURATED_INPUT_LAYER_FINAL_002`` package and active source contract.
They verify every public command, argument resolution, environment fallback,
structured JSON output, validation-report export, safe output-root policy,
overwrite behavior, dry-run and complete execution, deterministic module
outputs, stable exit codes, and absence of notebook or prohibited runtime
sources.

All analytical executions write to pytest temporary directories. The frozen
Phase 3 package and accepted repository result tree are never modified.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from qstable_awm import cli
from qstable_awm.pipeline import MODULE_ORDER
from qstable_awm.validation import PHASE3_PACKAGE_NAME


PHASE4_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = PHASE4_ROOT / "src" / "qstable_awm" / "cli.py"
CONTRACT_PATH = PHASE4_ROOT / "config" / "phase4_source_contract.yml"
PHASE3_ROOT = Path(
    os.environ.get(
        "QSTABLE_PHASE3_ROOT",
        PHASE4_ROOT.parent / PHASE3_PACKAGE_NAME,
    )
).expanduser().resolve()

EXPECTED_CLI_HASH = (
    "c36cc75fc87dbbb00771544cbb2928fb1384ee871971969263b9b86b5a05cb6d"
)
EXPECTED_CONTRACT_HASH = (
    "b701177f25c7c758e50880bbf63901ba95d0d1e4bc7dc03d7af4038e4e8ff725"
)
EXPECTED_OUTPUT_COUNTS = {
    "climate-context": 5,
    "physical-coherence": 6,
    "aerodynamic-sensitivity": 6,
    "crop-support": 5,
    "fixed-frame-dominance": 5,
    "known-cold-reference": 6,
    "interpretation-classes": 5,
}
EXPECTED_RUNTIME_CHECKS = {
    "climate-context": 130,
    "physical-coherence": 176,
    "aerodynamic-sensitivity": 155,
    "crop-support": 214,
    "fixed-frame-dominance": 115,
    "known-cold-reference": 922,
    "interpretation-classes": 68,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_hashes(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _json_stdout(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    captured = capsys.readouterr()
    assert captured.err == ""
    return json.loads(captured.out)


@pytest.fixture(scope="session", autouse=True)
def _require_phase3() -> None:
    assert PHASE3_ROOT.is_dir(), (
        "Frozen Phase 3 root not found. Set QSTABLE_PHASE3_ROOT to "
        "PHASE3_CURATED_INPUT_LAYER_FINAL_002."
    )


@pytest.fixture(scope="session")
def module_command_runs(tmp_path_factory: pytest.TempPathFactory):
    records: dict[str, dict[str, object]] = {}
    for command, expected_count in EXPECTED_OUTPUT_COUNTS.items():
        results_root = tmp_path_factory.mktemp(f"cli_{command}") / "results"
        args = cli.build_parser().parse_args(
            [
                command,
                "--phase3-root",
                str(PHASE3_ROOT),
                "--results-root",
                str(results_root),
                "--contract",
                str(CONTRACT_PATH),
            ]
        )
        status, payload = cli.dispatch(args)
        module_name = cli.COMMAND_TO_MODULE[command]
        module_dir = results_root / module_name
        files = sorted(path for path in module_dir.iterdir() if path.is_file())
        assert status == 0
        assert len(files) == expected_count
        records[command] = {
            "results_root": results_root,
            "module_dir": module_dir,
            "payload": payload,
            "hashes": _tree_hashes(module_dir),
        }
    return records


@pytest.fixture(scope="session")
def full_cli_run(tmp_path_factory: pytest.TempPathFactory):
    results_root = tmp_path_factory.mktemp("cli_full_run") / "results"
    args = cli.build_parser().parse_args(
        [
            "run-all",
            "--phase3-root",
            str(PHASE3_ROOT),
            "--results-root",
            str(results_root),
            "--contract",
            str(CONTRACT_PATH),
        ]
    )
    status, payload = cli.dispatch(args)
    return results_root, status, payload


def test_cli_and_contract_hashes_are_authoritative() -> None:
    assert _sha256(CLI_PATH) == EXPECTED_CLI_HASH
    assert _sha256(CONTRACT_PATH) == EXPECTED_CONTRACT_HASH


def test_public_command_registry_is_complete_and_bijective() -> None:
    assert tuple(cli.COMMAND_TO_MODULE) == tuple(EXPECTED_OUTPUT_COUNTS)
    assert tuple(cli.COMMAND_TO_MODULE.values()) == MODULE_ORDER
    assert cli.MODULE_TO_COMMAND == {
        module: command for command, module in cli.COMMAND_TO_MODULE.items()
    }
    assert tuple(cli.MODULE_RUNNERS) == MODULE_ORDER


def test_cli_metadata_and_exports_are_stable() -> None:
    assert cli.CLI_VERSION == "1.0.0"
    assert cli.CLI_RELATIVE_PATH == "src/qstable_awm/cli.py"
    assert set(cli.__all__) == {
        "CLIError",
        "CLI_RELATIVE_PATH",
        "CLI_VERSION",
        "COMMAND_TO_MODULE",
        "MODULE_TO_COMMAND",
        "build_parser",
        "dispatch",
        "main",
    }


def test_top_level_help_lists_every_public_command() -> None:
    help_text = cli.build_parser().format_help()
    for command in ("validate", *EXPECTED_OUTPUT_COUNTS, "run-all"):
        assert command in help_text
    assert "Q-Stable/AWM Phase 4" in help_text


def test_version_action_reports_stable_version() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PHASE4_ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-m", "qstable_awm.cli", "--version"],
        cwd=PHASE4_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0
    assert completed.stdout.strip() == "qstable-awm 1.0.0"
    assert completed.stderr == ""


def test_environment_phase3_fallback_is_resolved_at_parser_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QSTABLE_PHASE3_ROOT", str(PHASE3_ROOT))
    args = cli.build_parser().parse_args(["validate"])
    assert args.phase3_root == PHASE3_ROOT


def test_sibling_phase3_fallback_is_used_when_environment_is_absent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("QSTABLE_PHASE3_ROOT", raising=False)
    portable_phase4_root = tmp_path / "PHASE4_REUSABLE_ANALYTIC_MODULES_001"
    portable_phase4_root.mkdir()
    portable_phase3_root = tmp_path / PHASE3_PACKAGE_NAME
    portable_phase3_root.mkdir()
    monkeypatch.setattr(cli, "PHASE4_ROOT", portable_phase4_root)
    assert cli._default_phase3_root().resolve() == portable_phase3_root.resolve()


def test_phase3_argument_becomes_required_when_no_default_exists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("QSTABLE_PHASE3_ROOT", raising=False)
    monkeypatch.setattr(cli, "PHASE4_ROOT", tmp_path / "phase4")
    parser = cli.build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["validate"])
    assert exc_info.value.code == 2


def test_validate_complete_command_reports_1708_passes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = cli.main(
        [
            "validate",
            "--phase3-root",
            str(PHASE3_ROOT),
            "--contract",
            str(CONTRACT_PATH),
        ]
    )
    assert status == 0
    payload = _json_stdout(capsys)
    assert payload["command"] == "validate"
    assert payload["hashes_only"] is False
    assert payload["summary"] == {
        "report_version": "1.0.0",
        "phase3_root": str(PHASE3_ROOT),
        "contract_path": str(CONTRACT_PATH.resolve()),
        "modules": list(MODULE_ORDER),
        "status": "PASS",
        "checks": 1708,
        "passed": 1708,
        "warnings": 0,
        "failed": 0,
    }


@pytest.mark.parametrize("command", tuple(EXPECTED_RUNTIME_CHECKS))
def test_validate_each_public_module_command_name(
    command: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = cli.main(
        [
            "validate",
            "--phase3-root",
            str(PHASE3_ROOT),
            "--contract",
            str(CONTRACT_PATH),
            "--module",
            command,
        ]
    )
    assert status == 0
    payload = _json_stdout(capsys)
    module_name = cli.COMMAND_TO_MODULE[command]
    assert payload["summary"]["modules"] == [module_name]
    assert payload["summary"]["checks"] == EXPECTED_RUNTIME_CHECKS[command]
    assert payload["summary"]["passed"] == EXPECTED_RUNTIME_CHECKS[command]


def test_validate_multiple_modules_preserves_requested_mapping(
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = cli.main(
        [
            "validate",
            "--phase3-root",
            str(PHASE3_ROOT),
            "--module",
            "crop-support",
            "--module",
            "climate-context",
        ]
    )
    assert status == 0
    payload = _json_stdout(capsys)
    assert payload["summary"]["modules"] == ["crop_support", "climate_context"]
    assert payload["summary"]["status"] == "PASS"


def test_validate_hashes_only_omits_structured_schema_checks(
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = cli.main(
        [
            "validate",
            "--phase3-root",
            str(PHASE3_ROOT),
            "--hashes-only",
        ]
    )
    assert status == 0
    payload = _json_stdout(capsys)
    assert payload["hashes_only"] is True
    assert payload["summary"]["status"] == "PASS"
    assert 0 < payload["summary"]["checks"] < 1708


def test_validate_writes_detailed_json_and_csv_reports(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    json_path = tmp_path / "validation.json"
    csv_path = tmp_path / "validation.csv"
    status = cli.main(
        [
            "validate",
            "--phase3-root",
            str(PHASE3_ROOT),
            "--module",
            "crop-support",
            "--output-json",
            str(json_path),
            "--output-csv",
            str(csv_path),
        ]
    )
    assert status == 0
    payload = _json_stdout(capsys)
    assert payload["output_json"] == str(json_path)
    assert payload["output_csv"] == str(csv_path)
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["checks"] == 214
    assert report["passed"] == 214
    assert len(report["check_results"]) == 214
    assert len(csv_path.read_text(encoding="utf-8").splitlines()) == 215


@pytest.mark.parametrize("command", tuple(EXPECTED_OUTPUT_COUNTS))
def test_each_module_command_executes_exact_authorized_output_set(
    command: str,
    module_command_runs,
) -> None:
    record = module_command_runs[command]
    module_name = cli.COMMAND_TO_MODULE[command]
    payload = record["payload"]
    assert payload["command"] == command
    assert payload["module"] == module_name
    assert payload["phase3_package"] == PHASE3_PACKAGE_NAME
    assert payload["results_directory"] == str(record["module_dir"])
    assert payload["contract"] == str(CONTRACT_PATH.resolve())
    assert payload["overwrite"] is False
    assert len(payload["outputs"]) == EXPECTED_OUTPUT_COUNTS[command]
    assert len(record["hashes"]) == EXPECTED_OUTPUT_COUNTS[command]


def test_module_command_refuses_implicit_overwrite(
    module_command_runs,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results_root = module_command_runs["crop-support"]["results_root"]
    before = _tree_hashes(results_root)
    status = cli.main(
        [
            "crop-support",
            "--phase3-root",
            str(PHASE3_ROOT),
            "--results-root",
            str(results_root),
        ]
    )
    assert status == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("QSTABLE_AWM_CLI_ERROR:")
    assert "Refusing to overwrite" in captured.err
    assert _tree_hashes(results_root) == before


def test_module_command_explicit_overwrite_is_byte_deterministic(
    module_command_runs,
    capsys: pytest.CaptureFixture[str],
) -> None:
    record = module_command_runs["crop-support"]
    before = _tree_hashes(record["module_dir"])
    status = cli.main(
        [
            "crop-support",
            "--phase3-root",
            str(PHASE3_ROOT),
            "--results-root",
            str(record["results_root"]),
            "--overwrite",
        ]
    )
    assert status == 0
    payload = _json_stdout(capsys)
    assert payload["overwrite"] is True
    assert _tree_hashes(record["module_dir"]) == before


@pytest.mark.parametrize("unsafe_relative", [".", "nested/results"])
def test_module_command_rejects_results_root_equal_to_or_inside_phase3(
    unsafe_relative: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    unsafe = PHASE3_ROOT if unsafe_relative == "." else PHASE3_ROOT / unsafe_relative
    status = cli.main(
        [
            "crop-support",
            "--phase3-root",
            str(PHASE3_ROOT),
            "--results-root",
            str(unsafe),
        ]
    )
    assert status == 2
    captured = capsys.readouterr()
    assert "must not equal or be inside the Phase 3 root" in captured.err


@pytest.mark.parametrize("protected", ["src", "config", "tests"])
def test_module_command_rejects_results_under_protected_repository_tree(
    protected: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = cli.main(
        [
            "crop-support",
            "--phase3-root",
            str(PHASE3_ROOT),
            "--results-root",
            str(PHASE4_ROOT / protected / "cli-output"),
        ]
    )
    assert status == 2
    captured = capsys.readouterr()
    assert "must not be inside src, config, or tests" in captured.err


def test_run_all_dry_run_validates_and_writes_nothing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results_root = tmp_path / "dry-results"
    status = cli.main(
        [
            "run-all",
            "--phase3-root",
            str(PHASE3_ROOT),
            "--results-root",
            str(results_root),
            "--dry-run",
        ]
    )
    assert status == 0
    payload = _json_stdout(capsys)
    assert payload["command"] == "run-all"
    assert payload["dry_run"] is True
    assert payload["selected_modules"] == list(MODULE_ORDER)
    assert payload["preflight_validation"]["checks"] == 1708
    assert payload["outputs"] is None
    assert not results_root.exists()


def test_run_all_complete_execution_has_41_files(full_cli_run) -> None:
    results_root, status, payload = full_cli_run
    assert status == 0
    assert payload["command"] == "run-all"
    assert payload["dry_run"] is False
    assert payload["selected_modules"] == list(MODULE_ORDER)
    assert payload["preflight_validation"]["checks"] == 1708
    assert payload["preflight_validation"]["passed"] == 1708
    assert payload["outputs"]["phase4_pipeline_output_manifest"] == str(
        results_root / "pipeline" / "phase4_pipeline_output_manifest.csv"
    )
    assert len(_tree_hashes(results_root)) == 41


def test_run_all_refuses_implicit_overwrite(
    full_cli_run,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results_root, _, _ = full_cli_run
    before = _tree_hashes(results_root)
    status = cli.main(
        [
            "run-all",
            "--phase3-root",
            str(PHASE3_ROOT),
            "--results-root",
            str(results_root),
        ]
    )
    assert status == 2
    captured = capsys.readouterr()
    assert "QSTABLE_AWM_CLI_ERROR" in captured.err
    assert "Refusing to overwrite existing pipeline targets" in captured.err
    assert _tree_hashes(results_root) == before


def test_missing_phase3_root_returns_validation_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = cli.main(
        [
            "validate",
            "--phase3-root",
            str(tmp_path / "missing-phase3"),
        ]
    )
    assert status == 1
    payload = _json_stdout(capsys)
    assert payload["summary"]["status"] == "FAIL"
    assert payload["summary"]["failed"] == 1


def test_missing_contract_returns_stable_runtime_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = cli.main(
        [
            "validate",
            "--phase3-root",
            str(PHASE3_ROOT),
            "--contract",
            str(tmp_path / "missing-contract.yml"),
        ]
    )
    assert status == 2
    captured = capsys.readouterr()
    assert "Source contract not found" in captured.err


def test_dispatch_rejects_unsupported_command() -> None:
    namespace = type("Namespace", (), {"command": "unsupported"})()
    with pytest.raises(cli.CLIError, match="Unsupported command"):
        cli.dispatch(namespace)


def test_argparse_rejects_missing_or_unknown_command() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit) as missing:
        parser.parse_args([])
    assert missing.value.code == 2
    with pytest.raises(SystemExit) as unknown:
        parser.parse_args(["unknown-command"])
    assert unknown.value.code == 2


def test_validate_rejects_internal_underscore_module_name() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(
            [
                "validate",
                "--phase3-root",
                str(PHASE3_ROOT),
                "--module",
                "crop_support",
            ]
        )
    assert exc_info.value.code == 2


def test_cli_source_contains_no_notebook_or_prohibited_runtime_dependency() -> None:
    source = CLI_PATH.read_text(encoding="utf-8")
    prohibited = (
        ".ipynb",
        "AWM_figures",
        "V46D",
        "V48",
        "V49",
        "/scratch/",
        "METRIC_daily_integrated_ET_over_season",
    )
    for token in prohibited:
        assert token not in source
    assert "run_pipeline" in source
    assert "validate_phase3_root" in source
    assert "MODULE_RUNNERS" in source


def test_cli_does_not_modify_phase3_during_actual_commands(module_command_runs, full_cli_run) -> None:
    del module_command_runs, full_cli_run
    # The exact Phase 3 tree hash is already locked by the pipeline suite. Here
    # the CLI layer independently confirms that no output directory was created
    # beneath the frozen package during all actual CLI-command executions.
    unexpected = [
        path
        for path in PHASE3_ROOT.rglob("results")
        if path.is_dir() and path.parent == PHASE3_ROOT
    ]
    assert unexpected == []
