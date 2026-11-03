"""Regression and release-engineering tests for the Phase 4 pipeline.

The tests exercise :mod:`qstable_awm.pipeline` against the complete frozen
``PHASE3_CURATED_INPUT_LAYER_FINAL_002`` package and the active Phase 4 source
contract. They verify complete preflight validation, fixed execution order,
dry-run behavior, module-specific and full execution, deterministic manifests,
Phase 3 immutability, atomic publication, overwrite protection, failure
propagation, runtime-source scanning, and command-line behavior.

All analytical executions write to pytest temporary directories. The frozen
Phase 3 package and accepted repository result tree are never modified.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from qstable_awm import pipeline as p
from qstable_awm.validation import PHASE3_PACKAGE_NAME, load_source_contract, sha256_file


PHASE4_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = PHASE4_ROOT / "src" / "qstable_awm" / "pipeline.py"
CONTRACT_PATH = PHASE4_ROOT / "config" / "phase4_source_contract.yml"
REFERENCE_RESULTS_ROOT = PHASE4_ROOT / "results"
PHASE3_ROOT = Path(
    os.environ.get(
        "QSTABLE_PHASE3_ROOT",
        PHASE4_ROOT.parent / PHASE3_PACKAGE_NAME,
    )
).expanduser().resolve()

EXPECTED_PIPELINE_HASH = (
    "4561d939e2257574c279677e1c21d0dd491c9b1147313efad0a872370e0421b3"
)
EXPECTED_CONTRACT_HASH = (
    "b701177f25c7c758e50880bbf63901ba95d0d1e4bc7dc03d7af4038e4e8ff725"
)
EXPECTED_PHASE3_TREE_HASH = (
    "b74edffd081bba5962cda93f9696fa43d4ceab42028ac1db65c996556360421f"
)
EXPECTED_PIPELINE_METADATA_HASHES = {
    p.PIPELINE_MANIFEST_NAME: (
        "014d5682e49cbbf0f70eb4e2f6c02dacfd92d64ebbea6b1a0fcaa21a6cc82d14"
    ),
    p.PIPELINE_VALIDATION_NAME: (
        "e4349aa1b189efb482cc8d8377728bf288c1f402044d3db545b4ac799e7e7f3b"
    ),
    p.PIPELINE_EXECUTION_RECORD_NAME: (
        "d4b75b0d0b7fac2e38d75150ff27cba41f7ff347ca1abf30cfa7644d3fc1a00d"
    ),
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
EXPECTED_DERIVED_CHECKS = {
    "climate_context": 7,
    "physical_coherence": 10,
    "aerodynamic_sensitivity": 16,
    "crop_support": 26,
    "fixed_frame_dominance": 30,
    "known_cold_reference": 52,
    "interpretation_classes": 40,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }




def _portable_execution_payload(path: Path) -> dict:
    """Normalize repository-location metadata without changing scientific content."""

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
    """Return path-independent scientific and validation output hashes."""

    return {
        relative: digest
        for relative, digest in _file_hashes(root).items()
        if not relative.endswith("_execution_record.json")
        and not relative.startswith("pipeline/")
    }

def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.fixture(scope="session")
def contract():
    assert PHASE3_ROOT.is_dir(), (
        "Frozen Phase 3 root not found. Set QSTABLE_PHASE3_ROOT to "
        "PHASE3_CURATED_INPUT_LAYER_FINAL_002."
    )
    return load_source_contract(CONTRACT_PATH)


@pytest.fixture(scope="session")
def phase3_inventory():
    return p._tree_inventory(PHASE3_ROOT)


@pytest.fixture(scope="session")
def full_runs(tmp_path_factory: pytest.TempPathFactory):
    first_root = tmp_path_factory.mktemp("pipeline_full_first") / "results"
    second_root = tmp_path_factory.mktemp("pipeline_full_second") / "results"
    first = p.run_pipeline(
        PHASE3_ROOT,
        first_root,
        contract_path=CONTRACT_PATH,
    )
    second = p.run_pipeline(
        PHASE3_ROOT,
        second_root,
        contract_path=CONTRACT_PATH,
    )
    return first_root, first, second_root, second


@pytest.fixture(scope="session")
def crop_only_run(tmp_path_factory: pytest.TempPathFactory):
    results_root = tmp_path_factory.mktemp("pipeline_crop_only") / "results"
    result = p.run_pipeline(
        PHASE3_ROOT,
        results_root,
        modules=("crop_support",),
        contract_path=CONTRACT_PATH,
    )
    return results_root, result


def test_pipeline_and_contract_hashes_are_authoritative() -> None:
    assert _sha256(PIPELINE_PATH) == EXPECTED_PIPELINE_HASH
    assert _sha256(CONTRACT_PATH) == EXPECTED_CONTRACT_HASH


def test_frozen_module_order_and_run_function_registry_are_complete() -> None:
    assert p.MODULE_ORDER == tuple(EXPECTED_OUTPUT_COUNTS)
    assert tuple(p.MODULE_RUN_FUNCTIONS) == p.MODULE_ORDER
    assert set(p.MODULE_RUN_FUNCTIONS.values()) == {
        "run_climate_context",
        "run_physical_coherence",
        "run_aerodynamic_sensitivity",
        "run_crop_support",
        "run_fixed_frame_dominance",
        "run_known_cold_reference",
        "run_interpretation_classes",
    }


def test_phase3_inventory_is_complete_and_frozen(phase3_inventory) -> None:
    rows, tree_hash = phase3_inventory
    assert len(rows) == 172
    assert tree_hash == EXPECTED_PHASE3_TREE_HASH
    assert len({row["path"] for row in rows}) == 172
    assert all(row["size_bytes"] > 0 for row in rows)
    assert all(len(row["sha256"]) == 64 for row in rows)


def test_module_selection_defaults_to_all_and_preserves_frozen_order() -> None:
    assert p._resolve_selected_modules(None) == p.MODULE_ORDER
    assert p._resolve_selected_modules(()) == p.MODULE_ORDER
    assert p._resolve_selected_modules(
        ("interpretation_classes", "climate_context", "crop_support")
    ) == ("climate_context", "crop_support", "interpretation_classes")


@pytest.mark.parametrize(
    "modules,message",
    [
        (("crop_support", "crop_support"), "Duplicate module names"),
        (("unauthorized",), "Unknown module names"),
    ],
)
def test_invalid_module_selection_is_rejected(modules, message: str) -> None:
    with pytest.raises(p.PipelineError, match=message):
        p._resolve_selected_modules(modules)


@pytest.mark.parametrize("protected_name", ["src", "config", "tests"])
def test_results_root_inside_protected_repository_directories_is_rejected(
    protected_name: str,
) -> None:
    with pytest.raises(p.PipelineError, match="must not be inside"):
        p._assert_safe_roots(PHASE3_ROOT, PHASE4_ROOT / protected_name / "outputs")


def test_results_root_equal_to_or_inside_phase3_is_rejected() -> None:
    with pytest.raises(p.PipelineError, match="must not equal or be inside"):
        p._assert_safe_roots(PHASE3_ROOT, PHASE3_ROOT)
    with pytest.raises(p.PipelineError, match="must not equal or be inside"):
        p._assert_safe_roots(PHASE3_ROOT, PHASE3_ROOT / "results")


def test_plan_uses_logical_contract_paths_not_physical_output_root(
    tmp_path: Path,
    contract,
) -> None:
    physical_root = tmp_path / "arbitrary-physical-results"
    plan = p._build_plan(p.MODULE_ORDER, results_root=physical_root, contract=contract)
    assert [item["order"] for item in plan] == list(range(1, 8))
    assert [item["module"] for item in plan] == list(p.MODULE_ORDER)
    assert all(item["output_directory"].startswith("results/") for item in plan)
    assert str(physical_root) not in json.dumps(plan)
    for item in plan:
        module = item["module"]
        assert item["authorized_inputs"] == [
            entry.relative_path for entry in contract.files_for_module(module)
        ]
        assert item["authorized_outputs"] == list(contract.allowed_outputs(module))


def test_dry_run_performs_full_preflight_and_writes_nothing(tmp_path: Path) -> None:
    results_root = tmp_path / "dry-results"
    result = p.run_pipeline(
        PHASE3_ROOT,
        results_root,
        contract_path=CONTRACT_PATH,
        dry_run=True,
    )
    assert result.dry_run is True
    assert result.selected_modules == p.MODULE_ORDER
    assert result.outputs is None
    assert result.preflight_validation["status"] == "PASS"
    assert result.preflight_validation["checks"] == 1708
    assert result.preflight_validation["passed"] == 1708
    assert len(result.plan) == 7
    assert not results_root.exists()
    assert not list(tmp_path.glob(".phase4-pipeline-staging-*"))


def test_module_specific_dry_run_still_validates_complete_phase3(tmp_path: Path) -> None:
    result = p.run_pipeline(
        PHASE3_ROOT,
        tmp_path / "dry-crop-results",
        modules=("crop_support",),
        contract_path=CONTRACT_PATH,
        dry_run=True,
    )
    assert result.selected_modules == ("crop_support",)
    assert result.preflight_validation["checks"] == 1708
    assert [item["module"] for item in result.plan] == ["crop_support"]


def test_complete_run_writes_exact_contract_output_tree(full_runs, contract) -> None:
    results_root, result, _, _ = full_runs
    assert result.dry_run is False
    assert result.selected_modules == p.MODULE_ORDER
    assert result.outputs is not None
    assert result.preflight_validation["checks"] == 1708

    for module, expected_count in EXPECTED_OUTPUT_COUNTS.items():
        directory = results_root / module
        assert directory.is_dir()
        observed = {path.name for path in directory.iterdir() if path.is_file()}
        assert observed == set(contract.allowed_outputs(module))
        assert len(observed) == expected_count

    pipeline_dir = results_root / p.PIPELINE_DIRECTORY_NAME
    assert {path.name for path in pipeline_dir.iterdir() if path.is_file()} == set(
        EXPECTED_PIPELINE_METADATA_HASHES
    )
    assert len(_file_hashes(results_root)) == 41


def test_authoritative_repository_pipeline_metadata_hashes_are_frozen(full_runs) -> None:
    first_root, _, second_root, _ = full_runs
    for name, expected_hash in EXPECTED_PIPELINE_METADATA_HASHES.items():
        assert _sha256(REFERENCE_RESULTS_ROOT / p.PIPELINE_DIRECTORY_NAME / name) == expected_hash
        assert _sha256(first_root / p.PIPELINE_DIRECTORY_NAME / name) == _sha256(
            second_root / p.PIPELINE_DIRECTORY_NAME / name
        )


def test_two_complete_runs_are_byte_deterministic_across_output_locations(full_runs) -> None:
    first_root, _, second_root, _ = full_runs
    first_hashes = _file_hashes(first_root)
    second_hashes = _file_hashes(second_root)
    assert len(first_hashes) == 41
    assert second_hashes == first_hashes


def test_full_run_matches_accepted_scientific_outputs_and_portable_records(full_runs) -> None:
    first_root, _, _, _ = full_runs
    assert _portable_scientific_hashes(first_root) == _portable_scientific_hashes(
        REFERENCE_RESULTS_ROOT
    )
    for module in p.MODULE_ORDER:
        name = f"{module}_execution_record.json"
        assert _portable_execution_payload(first_root / module / name) == (
            _portable_execution_payload(REFERENCE_RESULTS_ROOT / module / name)
        )


def test_pipeline_manifest_has_38_unique_contract_authorized_rows(full_runs, contract) -> None:
    first_root, _, _, _ = full_runs
    manifest_path = first_root / p.PIPELINE_DIRECTORY_NAME / p.PIPELINE_MANIFEST_NAME
    rows = _read_manifest(manifest_path)
    assert len(rows) == 38
    assert len({row["relative_path"] for row in rows}) == 38
    assert all(row["contract_sha256"] == EXPECTED_CONTRACT_HASH for row in rows)

    expected_paths = {
        f"{module}/{filename}"
        for module in p.MODULE_ORDER
        for filename in contract.allowed_outputs(module)
    }
    assert {row["relative_path"] for row in rows} == expected_paths
    for row in rows:
        output = first_root / row["relative_path"]
        assert row["filename"] == output.name
        assert int(row["size_bytes"]) == output.stat().st_size
        assert row["sha256"] == _sha256(output)
        assert row["file_type"] == output.suffix.lstrip(".")


def test_pipeline_validation_passes_all_ten_checks(full_runs) -> None:
    first_root, _, _, _ = full_runs
    validation = json.loads(
        (
            first_root
            / p.PIPELINE_DIRECTORY_NAME
            / p.PIPELINE_VALIDATION_NAME
        ).read_text(encoding="utf-8")
    )
    assert validation["status"] == "PASS"
    assert validation["checks"] == validation["passed"] == 10
    assert validation["warnings"] == validation["failed"] == 0
    assert {item["check_id"] for item in validation["check_results"]} == {
        f"PIPE-{number:03d}" for number in range(1, 11)
    }
    assert all(item["status"] == "PASS" for item in validation["check_results"])
    observed_checks = {
        item["module"]: item["derived_validation"]["checks"]
        for item in validation["module_validation"]
    }
    assert observed_checks == EXPECTED_DERIVED_CHECKS


def test_pipeline_execution_record_reconciles_sources_inputs_and_metadata(
    full_runs,
    contract,
) -> None:
    first_root, _, _, _ = full_runs
    pipeline_dir = first_root / p.PIPELINE_DIRECTORY_NAME
    record = json.loads(
        (pipeline_dir / p.PIPELINE_EXECUTION_RECORD_NAME).read_text(encoding="utf-8")
    )
    assert record["record_version"] == p.PIPELINE_VERSION
    assert record["pipeline_relative_path"] == p.PIPELINE_RELATIVE_PATH
    assert record["pipeline_sha256"] == EXPECTED_PIPELINE_HASH
    assert record["contract_sha256"] == EXPECTED_CONTRACT_HASH
    assert record["phase3_package"] == PHASE3_PACKAGE_NAME
    assert record["phase3_root_name"] == PHASE3_PACKAGE_NAME
    assert record["phase3_file_count"] == 172
    assert record["phase3_tree_sha256"] == EXPECTED_PHASE3_TREE_HASH
    assert record["selected_modules"] == list(p.MODULE_ORDER)
    assert record["module_order"] == list(p.MODULE_ORDER)
    assert len(record["module_sources"]) == 7
    assert len(record["authorized_inputs"]) == 25
    assert len(record["module_validation"]) == 7
    assert record["output_manifest"] == {
        "relative_path": f"pipeline/{p.PIPELINE_MANIFEST_NAME}",
        "rows": 38,
        "sha256": _sha256(pipeline_dir / p.PIPELINE_MANIFEST_NAME),
    }
    assert record["pipeline_validation"] == {
        "relative_path": f"pipeline/{p.PIPELINE_VALIDATION_NAME}",
        "sha256": _sha256(pipeline_dir / p.PIPELINE_VALIDATION_NAME),
    }
    assert record["dry_run"] is False
    assert record["atomic_commit"] is True
    assert record["timestamp_policy"] == "omitted_for_deterministic_outputs"
    assert record["notebook_runtime_dependency"] is False
    assert record["phase3_modified"] is False
    assert record["runtime_results_root"] == "results"

    declared_inputs = {
        item.relative_path: {"sha256": item.sha256}
        for item in contract.required_files()
    }
    assert record["authorized_inputs"] == dict(sorted(declared_inputs.items()))


def test_execution_record_uses_logical_output_paths(full_runs) -> None:
    first_root, _, _, _ = full_runs
    record_text = (
        first_root
        / p.PIPELINE_DIRECTORY_NAME
        / p.PIPELINE_EXECUTION_RECORD_NAME
    ).read_text(encoding="utf-8")
    assert str(first_root) not in record_text
    record = json.loads(record_text)
    assert all(
        item["output_directory"] == f"results/{item['module']}"
        for item in record["execution_plan"]
    )


def test_module_specific_run_writes_only_selected_module_and_pipeline_metadata(
    crop_only_run,
) -> None:
    results_root, result = crop_only_run
    assert result.selected_modules == ("crop_support",)
    assert {path.name for path in results_root.iterdir() if path.is_dir()} == {
        "crop_support",
        "pipeline",
    }
    assert len(_file_hashes(results_root)) == 8

    manifest = _read_manifest(
        results_root / "pipeline" / p.PIPELINE_MANIFEST_NAME
    )
    assert len(manifest) == 5
    assert {row["module"] for row in manifest} == {"crop_support"}

    validation = json.loads(
        (results_root / "pipeline" / p.PIPELINE_VALIDATION_NAME).read_text()
    )
    assert validation["status"] == "PASS"
    assert validation["selected_modules"] == ["crop_support"]
    assert validation["module_validation"][0]["derived_validation"]["checks"] == 26


def test_existing_targets_require_explicit_overwrite_and_remain_unchanged(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "overwrite-results"
    p.run_pipeline(
        PHASE3_ROOT,
        results_root,
        modules=("crop_support",),
        contract_path=CONTRACT_PATH,
    )
    before = _file_hashes(results_root)
    with pytest.raises(p.PipelineError, match="Refusing to overwrite"):
        p.run_pipeline(
            PHASE3_ROOT,
            results_root,
            modules=("crop_support",),
            contract_path=CONTRACT_PATH,
        )
    assert _file_hashes(results_root) == before

    p.run_pipeline(
        PHASE3_ROOT,
        results_root,
        modules=("crop_support",),
        contract_path=CONTRACT_PATH,
        overwrite=True,
    )
    assert _file_hashes(results_root) == before


def test_failed_module_run_publishes_nothing_and_cleans_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_root = tmp_path / "failed-results"

    def fail_run(*args, **kwargs):
        raise RuntimeError("injected module failure")

    monkeypatch.setattr(p, "_load_run_function", lambda module: fail_run)
    with pytest.raises(p.PipelineError, match="injected module failure"):
        p.run_pipeline(
            PHASE3_ROOT,
            results_root,
            modules=("crop_support",),
            contract_path=CONTRACT_PATH,
        )
    assert not results_root.exists()
    assert not list(tmp_path.glob(".phase4-pipeline-staging-*"))
    assert not list(tmp_path.glob(".phase4-pipeline-backup-*"))


def test_phase3_change_detection_aborts_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase3_inventory,
) -> None:
    results_root = tmp_path / "phase3-change-results"
    rows, tree_hash = phase3_inventory
    calls = 0

    def changing_inventory(root: Path):
        nonlocal calls
        calls += 1
        if calls == 1:
            return rows, tree_hash
        changed = [dict(item) for item in rows]
        changed[0]["sha256"] = "0" * 64
        return changed, "1" * 64

    monkeypatch.setattr(p, "_tree_inventory", changing_inventory)
    with pytest.raises(p.PipelineError, match="Phase 3 contents changed"):
        p.run_pipeline(
            PHASE3_ROOT,
            results_root,
            modules=("crop_support",),
            contract_path=CONTRACT_PATH,
        )
    assert not results_root.exists()
    assert not list(tmp_path.glob(".phase4-pipeline-staging-*"))


def test_atomic_commit_rolls_back_existing_targets_on_install_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = tmp_path / "staged"
    results = tmp_path / "results"
    for root, marker in ((staged, "new"), (results, "old")):
        for name in ("crop_support", "pipeline"):
            directory = root / name
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "marker.txt").write_text(f"{marker}-{name}", encoding="utf-8")

    real_replace = os.replace

    def failing_replace(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        if source_path == staged / "pipeline" and destination_path == results / "pipeline":
            raise OSError("injected atomic-install failure")
        return real_replace(source, destination)

    monkeypatch.setattr(p.os, "replace", failing_replace)
    with pytest.raises(OSError, match="injected atomic-install failure"):
        p._atomic_commit(
            staged_results_root=staged,
            results_root=results,
            selected_modules=("crop_support",),
            overwrite=True,
        )

    assert (results / "crop_support" / "marker.txt").read_text() == "old-crop_support"
    assert (results / "pipeline" / "marker.txt").read_text() == "old-pipeline"
    assert not list(tmp_path.glob(".phase4-pipeline-backup-*"))


def test_module_directory_validation_rejects_tampered_recorded_output(
    tmp_path: Path,
    crop_only_run,
    contract,
) -> None:
    source_root = crop_only_run[0] / "crop_support"
    copied = tmp_path / "crop_support"
    shutil.copytree(source_root, copied)
    with (copied / "figure5_domain_crop_support_source.csv").open(
        "a", encoding="utf-8"
    ) as stream:
        stream.write("tamper\n")
    with pytest.raises(p.PipelineError, match="Recorded output hash mismatch"):
        p._validate_module_directory("crop_support", copied, contract=contract)


def test_module_directory_validation_rejects_unexpected_nonfile(
    tmp_path: Path,
    crop_only_run,
    contract,
) -> None:
    copied = tmp_path / "crop_support"
    shutil.copytree(crop_only_run[0] / "crop_support", copied)
    (copied / "unexpected-directory").mkdir()
    with pytest.raises(p.PipelineError, match="Unexpected non-file entries"):
        p._validate_module_directory("crop_support", copied, contract=contract)


def test_runtime_source_scan_ignores_defensive_literals_but_detects_file_access(
    tmp_path: Path,
) -> None:
    defensive = tmp_path / "defensive.py"
    defensive.write_text(
        "PROHIBITED = ('V49', '/scratch/')\nraise RuntimeError('reject V49')\n",
        encoding="utf-8",
    )
    unsafe = tmp_path / "unsafe.py"
    unsafe.write_text(
        "from pathlib import Path\nPath('/scratch/example/V49/output.csv').read_text()\n",
        encoding="utf-8",
    )
    tokens = ("V49", "V48")
    assert p._scan_runtime_file_access((defensive,), tokens) == {}
    assert p._scan_runtime_file_access((unsafe,), tokens) == {
        "unsafe.py": ["/scratch/", "V49"]
    }


def test_manifest_writer_is_byte_deterministic(tmp_path: Path, full_runs, contract) -> None:
    first_root, _, _, _ = full_runs
    rows = p._manifest_rows(first_root, p.MODULE_ORDER, contract=contract)
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    p._write_manifest(first, rows)
    p._write_manifest(second, rows)
    assert first.read_bytes() == second.read_bytes()
    assert _sha256(first) == _sha256(
        first_root / p.PIPELINE_DIRECTORY_NAME / p.PIPELINE_MANIFEST_NAME
    )


def test_pipeline_validation_reports_failure_for_changed_phase3_hash(contract) -> None:
    report = p._pipeline_validation(
        selected_modules=("crop_support",),
        preflight_summary={"status": "PASS", "failed": 0},
        module_summaries=(
            {
                "module": "crop_support",
                "status": "PASS",
                "execution_record_sha256": "a" * 64,
            },
        ),
        manifest_rows=tuple(
            {
                "relative_path": f"crop_support/{name}",
                "sha256": "b" * 64,
            }
            for name in contract.allowed_outputs("crop_support")
        ),
        phase3_before="c" * 64,
        phase3_after="d" * 64,
        source_scan_findings={},
        contract=contract,
    )
    assert report["status"] == "FAIL"
    assert report["failed"] == 1
    failed = [item for item in report["check_results"] if item["status"] == "FAIL"]
    assert [item["check_id"] for item in failed] == ["PIPE-008"]


def test_cli_dry_run_success_and_overwrite_refusal(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dry_root = tmp_path / "cli-dry"
    dry_args = [
        "--phase3-root",
        str(PHASE3_ROOT),
        "--results-root",
        str(dry_root),
        "--contract",
        str(CONTRACT_PATH),
        "--module",
        "crop_support",
        "--dry-run",
    ]
    assert p.main(dry_args) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["dry_run"] is True
    assert payload["selected_modules"] == ["crop_support"]
    assert captured.err == ""
    assert not dry_root.exists()

    run_root = tmp_path / "cli-run"
    run_args = [
        "--phase3-root",
        str(PHASE3_ROOT),
        "--results-root",
        str(run_root),
        "--contract",
        str(CONTRACT_PATH),
        "--module",
        "crop_support",
    ]
    assert p.main(run_args) == 0
    capsys.readouterr()
    assert p.main(run_args) == 2
    captured = capsys.readouterr()
    assert "PIPELINE_ERROR" in captured.err
    assert "Refusing to overwrite" in captured.err


def test_source_contains_no_notebook_execution_or_absolute_aces_dependency() -> None:
    source = PIPELINE_PATH.read_text(encoding="utf-8")
    prohibited = (
        "AWM_figures.ipynb",
        "METRIC_daily_integrated_ET_over_season",
        "subprocess",
        "nbconvert",
        "papermill",
        "/scratch/user/",
    )
    for token in prohibited:
        assert token not in source
    assert "validate_phase3_root" in source
    assert "_tree_inventory" in source
    assert "_atomic_commit" in source
    assert "dry_run" in source
