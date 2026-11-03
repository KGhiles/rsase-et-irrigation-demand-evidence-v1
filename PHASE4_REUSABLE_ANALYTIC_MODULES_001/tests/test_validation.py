"""Regression and security-boundary tests for Phase 4 runtime validation.

The suite validates only the frozen Phase 3 authority layer and the Phase 4
source contract. Negative tests operate on temporary copies and never modify
frozen inputs or accepted manuscript outputs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml

from qstable_awm import validation as v
from qstable_awm.schemas import (
    ALL_RUNTIME_PATHS,
    JSON_SCHEMAS,
    PHASE3_PACKAGE_NAME,
    RASTER_SCHEMAS,
    TABLE_SCHEMAS,
    ScalarKind,
)

PHASE4_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PHASE4_ROOT / "src" / "qstable_awm" / "validation.py"
SCHEMAS_PATH = PHASE4_ROOT / "src" / "qstable_awm" / "schemas.py"
CONTRACT_PATH = PHASE4_ROOT / "config" / "phase4_source_contract.yml"
PHASE3_ROOT = Path(
    os.environ.get("QSTABLE_PHASE3_ROOT", PHASE4_ROOT.parent / PHASE3_PACKAGE_NAME)
).expanduser().resolve()

EXPECTED_VALIDATION_HASH = (
    "98eddd363a205dfc8c625190f2b552faf339e59927403d39044dc198c6efd390"
)
EXPECTED_SCHEMAS_HASH = (
    "27e7dbef16f8cc52eedc81c18f0662fe49c97eb51e0d2a42e9675a0f78125471"
)
EXPECTED_CONTRACT_HASH = (
    "b701177f25c7c758e50880bbf63901ba95d0d1e4bc7dc03d7af4038e4e8ff725"
)
EXPECTED_MODULE_COUNTS = {
    "climate_context": 130,
    "physical_coherence": 176,
    "aerodynamic_sensitivity": 155,
    "crop_support": 214,
    "fixed_frame_dominance": 115,
    "known_cold_reference": 922,
    "interpretation_classes": 68,
}
EXPECTED_MODULE_FILE_COUNTS = {
    "climate_context": 2,
    "physical_coherence": 4,
    "aerodynamic_sensitivity": 5,
    "crop_support": 3,
    "fixed_frame_dominance": 2,
    "known_cold_reference": 6,
    "interpretation_classes": 3,
}
EXPECTED_OUTPUT_COUNTS = {
    "climate_context": 5,
    "physical_coherence": 6,
    "aerodynamic_sensitivity": 6,
    "crop_support": 5,
    "fixed_frame_dominance": 5,
    "known_cold_reference": 6,
    "interpretation_classes": 5,
}
PACKAGE_ANCHORS = (
    "VERSION",
    "package_file_manifest.csv",
    "provenance/phase3_validation_summary.json",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_contract(tmp_path: Path, mutate) -> Path:
    payload = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    mutate(payload)
    path = tmp_path / "phase4_source_contract.yml"
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def _minimal_phase3_copy(tmp_path: Path, module: str) -> Path:
    root = tmp_path / PHASE3_PACKAGE_NAME
    contract = v.load_source_contract(CONTRACT_PATH)
    relative_paths = list(PACKAGE_ANCHORS) + [
        item.relative_path for item in contract.files_for_module(module)
    ]
    for relative_path in dict.fromkeys(relative_paths):
        source = PHASE3_ROOT / relative_path
        destination = root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return root


@pytest.fixture(scope="session")
def contract() -> v.SourceContract:
    assert PHASE3_ROOT.is_dir(), (
        "Frozen Phase 3 root not found. Set QSTABLE_PHASE3_ROOT to "
        "PHASE3_CURATED_INPUT_LAYER_FINAL_002."
    )
    return v.load_source_contract(CONTRACT_PATH)


@pytest.fixture(scope="session")
def full_report(contract: v.SourceContract) -> v.ValidationReport:
    return v.validate_phase3_root(PHASE3_ROOT, contract_path=contract.path)


def test_authoritative_validation_and_schema_hashes() -> None:
    assert _sha256(SOURCE_PATH) == EXPECTED_VALIDATION_HASH
    assert _sha256(SCHEMAS_PATH) == EXPECTED_SCHEMAS_HASH


def test_contract_hash_and_version_are_frozen(contract: v.SourceContract) -> None:
    assert contract.contract_sha256 == EXPECTED_CONTRACT_HASH
    assert contract.raw["contract_version"] == "1.4.0"


def test_contract_identity_and_module_order(contract: v.SourceContract) -> None:
    assert contract.phase4_package == "PHASE4_REUSABLE_ANALYTIC_MODULES_001"
    assert contract.runtime_input["package_name"] == PHASE3_PACKAGE_NAME
    assert contract.runtime_input["root_directory_name"] == PHASE3_PACKAGE_NAME
    assert contract.module_names == tuple(EXPECTED_MODULE_COUNTS)


def test_contract_runtime_policy_is_strict(contract: v.SourceContract) -> None:
    assert contract.runtime_policy == {
        "notebooks_are_runtime_inputs": False,
        "original_workflow_archives_are_runtime_inputs": False,
        "manuscript_text_is_a_numeric_data_source": False,
        "hard_coded_manuscript_results_allowed": False,
        "absolute_internal_paths_allowed": False,
        "inputs_outside_phase3_root_allowed": False,
        "scientific_values_must_be_derived_from_curated_inputs": True,
        "every_runtime_file_requires_sha256_match": True,
        "fail_on_unexpected_input_file": True,
        "fail_on_missing_required_input": True,
    }


def test_contract_files_schemas_and_outputs_are_complete(contract: v.SourceContract) -> None:
    assert len(contract.required_files()) == 25
    assert len(TABLE_SCHEMAS) == 21
    assert len(JSON_SCHEMAS) == 2
    assert len(RASTER_SCHEMAS) == 1
    assert len(ALL_RUNTIME_PATHS) == 24
    structured = {
        item.relative_path
        for item in contract.required_files()
        if Path(item.relative_path).suffix.lower() in {".csv", ".json", ".tif"}
    }
    assert structured == set(ALL_RUNTIME_PATHS)


@pytest.mark.parametrize("module,expected", EXPECTED_MODULE_FILE_COUNTS.items())
def test_module_input_counts(contract: v.SourceContract, module: str, expected: int) -> None:
    assert len(contract.files_for_module(module)) == expected


@pytest.mark.parametrize("module,expected", EXPECTED_OUTPUT_COUNTS.items())
def test_module_output_counts(contract: v.SourceContract, module: str, expected: int) -> None:
    outputs = contract.allowed_outputs(module)
    assert len(outputs) == expected
    assert len(outputs) == len(set(outputs))
    assert all(Path(name).name == name for name in outputs)


def test_full_phase3_validation_passes_1708_checks(full_report: v.ValidationReport) -> None:
    assert full_report.summary() == {
        "report_version": "1.0.0",
        "phase3_root": str(PHASE3_ROOT),
        "contract_path": str(CONTRACT_PATH.resolve()),
        "modules": list(EXPECTED_MODULE_COUNTS),
        "status": "PASS",
        "checks": 1708,
        "passed": 1708,
        "warnings": 0,
        "failed": 0,
    }
    assert full_report.is_valid
    assert len({check.check_id for check in full_report.checks}) == 1708


@pytest.mark.parametrize("module,expected", EXPECTED_MODULE_COUNTS.items())
def test_each_module_validation_count(module: str, expected: int) -> None:
    report = v.validate_phase3_root(
        PHASE3_ROOT, contract_path=CONTRACT_PATH, modules=(module,)
    )
    assert report.summary()["checks"] == expected
    assert report.summary()["passed"] == expected
    assert report.overall_status == "PASS"


@pytest.mark.parametrize(
    "module,expected", [("climate_context", 18), ("interpretation_classes", 33)]
)
def test_hashes_only_mode_skips_structured_schemas(module: str, expected: int) -> None:
    report = v.validate_phase3_root(
        PHASE3_ROOT,
        contract_path=CONTRACT_PATH,
        modules=(module,),
        validate_schemas=False,
    )
    assert report.summary()["checks"] == expected
    assert not any(
        check.check_id.startswith(("TAB-", "JSON-", "RASTER-"))
        for check in report.checks
    )


def test_validation_report_status_and_error_semantics() -> None:
    report = v.ValidationReport("phase3", "contract", ("module",))
    report.add(
        check_id="PASS-1", scope="test", description="pass",
        observed=True, expected=True, passed=True,
    )
    report.add(
        check_id="WARN-1", scope="test", description="warning",
        observed=False, expected=True, passed=False,
        severity=v.ValidationSeverity.WARNING,
    )
    assert report.overall_status == "PASS_WITH_WARNINGS"
    assert report.is_valid
    report.raise_for_errors()
    report.add(
        check_id="FAIL-1", scope="test", description="failure",
        observed=False, expected=True, passed=False,
    )
    assert report.overall_status == "FAIL"
    with pytest.raises(v.RuntimeInputError, match="FAIL-1"):
        report.raise_for_errors()


@pytest.mark.parametrize(
    "kind,value,expected",
    [
        (ScalarKind.STRING, "text", "text"),
        (ScalarKind.INTEGER, "17.0", 17),
        (ScalarKind.FLOAT, "1.25", 1.25),
        (ScalarKind.BOOLEAN, "yes", True),
        (ScalarKind.BOOLEAN, "No", False),
        (ScalarKind.DATE, "2023-09-30", "2023-09-30"),
    ],
)
def test_scalar_normalization(kind: ScalarKind, value, expected) -> None:
    assert v._normalize_value(value, kind) == expected


@pytest.mark.parametrize(
    "kind,value",
    [
        (ScalarKind.STRING, 5),
        (ScalarKind.INTEGER, "1.5"),
        (ScalarKind.FLOAT, "nan"),
        (ScalarKind.BOOLEAN, "maybe"),
        (ScalarKind.DATE, "2023-02-30"),
    ],
)
def test_invalid_scalar_values_are_rejected(kind: ScalarKind, value) -> None:
    with pytest.raises((TypeError, ValueError, OverflowError)):
        v._normalize_value(value, kind)


@pytest.mark.parametrize(
    "relative_path,message",
    [
        ("/absolute/input.csv", "relative"),
        ("../escape.csv", "Unsafe"),
        ("provenance/source.csv", "Provenance"),
        ("data/input.ipynb", "Notebook"),
    ],
)
def test_unsafe_runtime_paths_are_rejected(relative_path: str, message: str) -> None:
    with pytest.raises(v.RuntimeInputError, match=message):
        v.resolve_runtime_path(PHASE3_ROOT, relative_path)


def test_authorized_runtime_path_resolves_inside_phase3(contract: v.SourceContract) -> None:
    item = contract.files_for_module("climate_context")[0]
    path = v.resolve_runtime_path(PHASE3_ROOT, item.relative_path)
    assert path == (PHASE3_ROOT / item.relative_path).resolve()
    path.relative_to(PHASE3_ROOT)


def test_exact_module_input_set_is_accepted(contract: v.SourceContract) -> None:
    paths = [PHASE3_ROOT / i.relative_path for i in contract.files_for_module("climate_context")]
    resolved = v.assert_module_input_paths(
        contract, PHASE3_ROOT, "climate_context", reversed(paths)
    )
    assert resolved == tuple(sorted(path.resolve() for path in paths))


def test_duplicate_unexpected_and_outside_inputs_are_rejected(
    contract: v.SourceContract, tmp_path: Path
) -> None:
    expected = [PHASE3_ROOT / i.relative_path for i in contract.files_for_module("climate_context")]
    with pytest.raises(v.RuntimeInputError, match="Duplicate supplied"):
        v.assert_module_input_paths(
            contract, PHASE3_ROOT, "climate_context", [expected[0], expected[0]]
        )
    with pytest.raises(v.RuntimeInputError, match="Input-set mismatch"):
        v.assert_module_input_paths(contract, PHASE3_ROOT, "climate_context", expected[:1])
    outside = tmp_path / "outside.csv"
    outside.write_text("x\n1\n", encoding="utf-8")
    with pytest.raises(v.RuntimeInputError, match="outside Phase 3"):
        v.assert_module_input_paths(
            contract, PHASE3_ROOT, "climate_context", [*expected, outside]
        )


def test_output_name_authorization(contract: v.SourceContract) -> None:
    v.assert_allowed_output_name(contract, "climate_context", "figure2_climate_source.csv")
    with pytest.raises(v.RuntimeInputError, match="basename"):
        v.assert_allowed_output_name(
            contract, "climate_context", "results/figure2_climate_source.csv"
        )
    with pytest.raises(v.RuntimeInputError, match="not allowed"):
        v.assert_allowed_output_name(contract, "climate_context", "unauthorized.csv")


def test_contract_rejects_unsafe_runtime_policy(tmp_path: Path) -> None:
    path = _write_contract(
        tmp_path,
        lambda payload: payload["runtime_policy"].__setitem__(
            "notebooks_are_runtime_inputs", True
        ),
    )
    with pytest.raises(v.RuntimeInputError, match="Unsafe runtime policy"):
        v.load_source_contract(path)


def test_contract_rejects_unsafe_interpretation_guardrail(tmp_path: Path) -> None:
    def mutate(payload):
        payload["module_inputs"]["interpretation_classes"]["classification_guardrails"][
            "completed_class_i_assignment_present"
        ] = True

    with pytest.raises(v.RuntimeInputError, match="Unsafe interpretation guardrail"):
        v.load_source_contract(_write_contract(tmp_path, mutate))


def test_contract_rejects_missing_schema_bound_input(tmp_path: Path) -> None:
    def mutate(payload):
        payload["module_inputs"]["climate_context"]["required_files"].pop()

    with pytest.raises(v.RuntimeInputError, match="Schema paths absent"):
        v.load_source_contract(_write_contract(tmp_path, mutate))


def test_unknown_and_duplicate_requested_modules_are_rejected() -> None:
    with pytest.raises(v.RuntimeInputError, match="Unknown module names"):
        v.validate_phase3_root(
            PHASE3_ROOT, contract_path=CONTRACT_PATH, modules=("unknown",)
        )
    with pytest.raises(v.RuntimeInputError, match="Duplicate module names"):
        v.validate_phase3_root(
            PHASE3_ROOT,
            contract_path=CONTRACT_PATH,
            modules=("climate_context", "climate_context"),
        )


def test_missing_phase3_root_returns_machine_readable_failure(tmp_path: Path) -> None:
    report = v.validate_phase3_root(
        tmp_path / PHASE3_PACKAGE_NAME, contract_path=CONTRACT_PATH
    )
    assert report.summary()["status"] == "FAIL"
    assert [check.check_id for check in report.failed_checks] == ["PKG-000"]


def test_altered_runtime_hash_is_detected(tmp_path: Path) -> None:
    root = _minimal_phase3_copy(tmp_path, "climate_context")
    contract = v.load_source_contract(CONTRACT_PATH)
    item = contract.files_for_module("climate_context")[0]
    path = root / item.relative_path
    path.write_bytes(path.read_bytes() + b"\n")
    report = v.validate_module_inputs(
        root,
        "climate_context",
        contract_path=CONTRACT_PATH,
        supplied_paths=[root / i.relative_path for i in contract.files_for_module("climate_context")],
        raise_on_error=False,
    )
    failed_ids = {check.check_id for check in report.failed_checks}
    assert "HASH-01-SHA256" in failed_ids
    assert "HASH-01-MANIFEST" in failed_ids


def test_missing_required_runtime_file_is_detected(tmp_path: Path) -> None:
    root = _minimal_phase3_copy(tmp_path, "climate_context")
    item = v.load_source_contract(CONTRACT_PATH).files_for_module("climate_context")[0]
    (root / item.relative_path).unlink()
    report = v.validate_phase3_root(
        root, contract_path=CONTRACT_PATH, modules=("climate_context",)
    )
    assert "HASH-01-EXISTS" in {check.check_id for check in report.failed_checks}


def test_schema_column_mutation_is_detected(tmp_path: Path) -> None:
    root = _minimal_phase3_copy(tmp_path, "climate_context")
    item = v.load_source_contract(CONTRACT_PATH).files_for_module("climate_context")[0]
    path = root / item.relative_path
    frame = pd.read_csv(path)
    frame = frame.rename(columns={frame.columns[0]: "unauthorized_column"})
    frame.to_csv(path, index=False)
    report = v.validate_phase3_root(
        root, contract_path=CONTRACT_PATH, modules=("climate_context",)
    )
    assert any(
        check.description == "Exact column names and order"
        for check in report.failed_checks
    )


def test_completed_class_i_promotion_is_detected(tmp_path: Path) -> None:
    root = _minimal_phase3_copy(tmp_path, "interpretation_classes")
    path = root / "data/derived/interpretation_classes/interpretation_classification.csv"
    frame = pd.read_csv(path)
    row = frame.index[frame["final_class"].eq("Class I candidate")][0]
    frame.loc[row, "final_class"] = "Class I"
    frame.to_csv(path, index=False)
    report = v.validate_phase3_root(
        root, contract_path=CONTRACT_PATH, modules=("interpretation_classes",)
    )
    assert {"SCI-030", "SCI-031"} <= {
        check.check_id for check in report.failed_checks
    }


def test_report_serialization_is_deterministic(
    full_report: v.ValidationReport, tmp_path: Path
) -> None:
    first_json, second_json = tmp_path / "first.json", tmp_path / "second.json"
    first_csv, second_csv = tmp_path / "first.csv", tmp_path / "second.csv"
    v.write_validation_report(full_report, json_path=first_json, csv_path=first_csv)
    v.write_validation_report(full_report, json_path=second_json, csv_path=second_csv)
    assert first_json.read_bytes() == second_json.read_bytes()
    assert first_csv.read_bytes() == second_csv.read_bytes()
    payload = json.loads(first_json.read_text(encoding="utf-8"))
    assert payload["checks"] == 1708
    with first_csv.open(newline="", encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == 1708


def test_sha256_file_matches_known_digest(tmp_path: Path) -> None:
    path = tmp_path / "known.txt"
    path.write_bytes(b"qstable\n")
    assert v.sha256_file(path) == hashlib.sha256(b"qstable\n").hexdigest()
    with pytest.raises(FileNotFoundError):
        v.sha256_file(tmp_path / "missing.txt")


def test_prohibited_token_scanner_detects_and_honors_exclusions(tmp_path: Path) -> None:
    source_root = tmp_path / "src"
    source_root.mkdir()
    safe = source_root / "safe.py"
    bad = source_root / "bad.py"
    ignored = source_root / "ignored.txt"
    safe.write_text("VALUE = 'safe'\n", encoding="utf-8")
    bad.write_text("RUNTIME_SOURCE = 'V48'\n", encoding="utf-8")
    ignored.write_text("V49\n", encoding="utf-8")
    assert v.scan_phase4_sources_for_prohibited_tokens(
        source_root, contract_path=CONTRACT_PATH
    ) == {"bad.py": ["V48"]}
    assert v.scan_phase4_sources_for_prohibited_tokens(
        source_root, contract_path=CONTRACT_PATH, excluded_paths=(bad,)
    ) == {}


def test_cli_success_writes_detailed_reports(tmp_path: Path, capsys) -> None:
    json_path = tmp_path / "validation.json"
    csv_path = tmp_path / "validation.csv"
    status = v.main(
        [
            "--phase3-root", str(PHASE3_ROOT),
            "--contract", str(CONTRACT_PATH),
            "--module", "climate_context",
            "--hashes-only",
            "--output-json", str(json_path),
            "--output-csv", str(csv_path),
        ]
    )
    captured = capsys.readouterr()
    assert status == 0
    assert '"checks": 18' in captured.out
    assert captured.err == ""
    assert json.loads(json_path.read_text(encoding="utf-8"))["status"] == "PASS"


def test_cli_failure_and_contract_error_exit_codes(tmp_path: Path, capsys) -> None:
    missing_root = tmp_path / PHASE3_PACKAGE_NAME
    assert v.main(
        ["--phase3-root", str(missing_root), "--contract", str(CONTRACT_PATH)]
    ) == 1
    assert "PKG-000" in capsys.readouterr().err
    assert v.main(
        [
            "--phase3-root", str(PHASE3_ROOT),
            "--contract", str(tmp_path / "missing-contract.yml"),
        ]
    ) == 2
    assert "VALIDATION_ERROR" in capsys.readouterr().err
