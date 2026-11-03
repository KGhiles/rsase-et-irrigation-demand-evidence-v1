"""Regression and scientific-boundary tests for Figure 8 classification.

The tests exercise :mod:`qstable_awm.interpretation_classes` only against the
three frozen Phase 3 inputs authorized by the Phase 4 source contract. The
corrected Phase 2 Figure 8 package is the classification authority. Historical
notebooks, plotting code, and the manuscript are not production runtime inputs.

A manuscript-concordance test verifies that the exact Phase 4 assignments match
Section 3.7: 2011 is Class II, the fixed later-year frame is Class III,
unavailable concurrent known-cold support is Class IV, and the non-2011
workflow remains a Class I candidate rather than a completed Class I result.
This is release QA only; production values continue to derive from Phase 3.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from qstable_awm import interpretation_classes as ic
from qstable_awm.validation import RuntimeInputError, validate_phase3_root


PHASE4_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = PHASE4_ROOT / "config" / "phase4_source_contract.yml"
SOURCE_PATH = PHASE4_ROOT / "src" / "qstable_awm" / "interpretation_classes.py"
PHASE3_ROOT = Path(
    os.environ.get(
        "QSTABLE_PHASE3_ROOT",
        PHASE4_ROOT.parent / "PHASE3_CURATED_INPUT_LAYER_FINAL_002",
    )
).expanduser().resolve()

EXPECTED_MODULE_HASH = (
    "63031d26ac250803855c3237ff89139551a3d310f15e389d8930444bf65fe799"
)
EXPECTED_INPUT_HASHES = {
    ic.CLASSES_RELATIVE_PATH: (
        "4163714fbbc0ef20825c859f1a153c6b922e017cbab6e2be3ce3281e7adcb45e"
    ),
    ic.HLS_RELATIVE_PATH: (
        "35a0c6e5988c00aab7834e0b5eab9d3e585dfc6b23c93767c64232864bb3e514"
    ),
    ic.CLASSIFICATION_RELATIVE_PATH: (
        "b5dc590ca788a777af14ab22b9f4f3bfff986c8932e956e7001a65b22e433647"
    ),
}
EXPECTED_STABLE_OUTPUT_HASHES = {
    "figure8_interpretation_source.csv": (
        "e30f20b2e6dcdc4e75c578d0b66846048c87fce0dc4ab06abc035e2e15e33d1d"
    ),
    "figure8_evidence_matrix_source.csv": (
        "0f96118efcdd623cc88eccc7113b656efbcbc41153ce65f9efbae4ebc9af5e9b"
    ),
    "interpretation_classes_text_statistics.csv": (
        "285c9b86459f0c7859a92c6ecd0d2039f1e0c6de6500617509e99eec303b1d38"
    ),
    "interpretation_classes_validation.json": (
        "20b56f302c1f665076b484a8b626a3b4d2f07dc12711e75e5bdcb19c369f8898"
    ),
}
EXPECTED_OUTPUT_NAMES = set(EXPECTED_STABLE_OUTPUT_HASHES) | {
    "interpretation_classes_execution_record.json"
}
EXPECTED_CLASS_ASSIGNMENTS = {
    "2011 active-canopy large-ETrF scenes": "Class II",
    "Fixed 2011 frame in later years": "Class III",
    "Known-cold source unavailable": "Class IV",
    "Non-2011 year-specific reconstruction workflow": "Class I candidate",
}
EXPECTED_SUPPORTED_INTERPRETATIONS = {
    "2011 active-canopy large-ETrF scenes": "Stress-test diagnostic",
    "Fixed 2011 frame in later years": "Landscape/crop-support diagnostic",
    "Known-cold source unavailable": (
        "Excluded from strict reference-sensitivity analysis"
    ),
    "Non-2011 year-specific reconstruction workflow": "Proxy candidate after QA",
}
EXPECTED_STATUS_COUNTS = {
    "supported": 12,
    "material": 3,
    "conditional": 8,
    "invalid": 9,
}
EXPECTED_CANDIDATE_STATUS = {
    "E1": "supported",
    "E2": "supported",
    "E3": "supported",
    "E4": "supported",
    "E5": "supported",
    "E6": "conditional",
    "E7": "supported",
    "E8": "conditional",
}
EXPECTED_HLS_COUNTS = {
    "object_years": 4_196,
    "accepted_anchor_intervals": 26_085,
    "hls_temporal_bins": 17,
    "hls_object_bin_records": 71_332,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture(scope="session")
def loaded_inputs():
    assert PHASE3_ROOT.is_dir(), (
        "Frozen Phase 3 root not found. Set QSTABLE_PHASE3_ROOT to "
        "PHASE3_CURATED_INPUT_LAYER_FINAL_002."
    )
    return ic.load_interpretation_inputs(PHASE3_ROOT, contract_path=CONTRACT_PATH)


@pytest.fixture(scope="session")
def derived_tables(loaded_inputs):
    classes, hls, classification, runtime = loaded_inputs
    ic._assert_input_structure(classes, hls, classification)
    interpretation = ic.build_figure8_interpretation_source(classification)
    evidence = ic.build_figure8_evidence_matrix_source(classes)
    statistics = ic.build_text_statistics(interpretation, evidence, hls)
    return interpretation, evidence, statistics, runtime


@pytest.fixture(scope="session")
def generated_outputs(tmp_path_factory: pytest.TempPathFactory):
    output_dir = tmp_path_factory.mktemp("interpretation_classes_outputs")
    return ic.run_interpretation_classes(
        PHASE3_ROOT,
        output_dir,
        contract_path=CONTRACT_PATH,
    )


def test_module_and_authorized_input_hashes_are_frozen() -> None:
    assert _sha256(SOURCE_PATH) == EXPECTED_MODULE_HASH
    for relative_path, expected_hash in EXPECTED_INPUT_HASHES.items():
        path = PHASE3_ROOT / relative_path
        assert path.is_file(), relative_path
        assert _sha256(path) == expected_hash


def test_runtime_input_validation_passes_68_checks(loaded_inputs) -> None:
    *_, runtime = loaded_inputs
    assert runtime == {
        "report_version": "1.0.0",
        "modules": ["interpretation_classes"],
        "status": "PASS",
        "checks": 68,
        "passed": 68,
        "warnings": 0,
        "failed": 0,
    }


def test_corrected_phase2_authority_and_release_decisions(loaded_inputs) -> None:
    classes, hls, classification, _ = loaded_inputs
    assert classes["version"] == "PHASE2_FIGURE8_PUBLICATION_FINAL_003"
    assert classes["figure"] == "Figure 8"
    assert classes["release_decision"] == ic.EXPECTED_RELEASE_DECISIONS
    assert classes["authority"]["V50"] == (
        "Class I-candidate and HLS temporal-support evidence only"
    )
    assert "V49" not in classes["authority"]
    ic._assert_input_structure(classes, hls, classification)


def test_interpretation_source_preserves_exact_four_assignments(derived_tables) -> None:
    source = derived_tables[0]
    assert len(source) == 4
    assert source["condition_order"].tolist() == [1, 2, 3, 4]
    assert source["analysis_condition"].tolist() == list(ic.EXPECTED_CONDITIONS)
    assert source.duplicated(["analysis_condition"]).sum() == 0
    assert source.set_index("analysis_condition")["final_class"].to_dict() == (
        EXPECTED_CLASS_ASSIGNMENTS
    )
    assert source.set_index("analysis_condition")[
        "supported_interpretation"
    ].to_dict() == EXPECTED_SUPPORTED_INTERPRETATIONS
    assert source["figure_changed"].eq(False).all()
    assert source["interpretation_boundary"].eq(ic.INTERPRETATION_BOUNDARY).all()


def test_candidate_is_unique_and_never_completed_class_I(derived_tables) -> None:
    source = derived_tables[0]
    assert source["is_class_I_candidate"].sum() == 1
    assert source["is_completed_class_I"].sum() == 0
    candidate = source[source["is_class_I_candidate"]].iloc[0]
    assert candidate["analysis_condition"] == (
        "Non-2011 year-specific reconstruction workflow"
    )
    assert candidate["final_class"] == "Class I candidate"
    assert candidate["E6"] == "Pending QA"
    assert candidate["E8"] == "Candidate"


def test_manuscript_section_3_7_classification_concordance(derived_tables) -> None:
    """Release QA: exact outputs match the manuscript's stated class roles."""

    source = derived_tables[0].set_index("analysis_condition")
    assert source.loc[
        "2011 active-canopy large-ETrF scenes", "final_class"
    ] == "Class II"
    assert source.loc[
        "Fixed 2011 frame in later years", "final_class"
    ] == "Class III"
    assert source.loc[
        "Known-cold source unavailable", "final_class"
    ] == "Class IV"
    assert source.loc[
        "Non-2011 year-specific reconstruction workflow", "final_class"
    ] == "Class I candidate"
    assert not source["is_completed_class_I"].any()


def test_evidence_matrix_has_32_unique_ordered_cells(derived_tables) -> None:
    evidence = derived_tables[1]
    assert len(evidence) == 32
    assert evidence.duplicated(["analysis_condition", "evidence_code"]).sum() == 0
    assert evidence.groupby("analysis_condition")["evidence_code"].nunique().eq(8).all()
    assert evidence.groupby("condition_order")["evidence_order"].apply(list).to_dict() == {
        1: list(range(1, 9)),
        2: list(range(1, 9)),
        3: list(range(1, 9)),
        4: list(range(1, 9)),
    }
    assert dict(zip(evidence["evidence_code"], evidence["evidence_label"])) == (
        ic.EXPECTED_EVIDENCE_LABELS
    )


def test_evidence_status_counts_and_candidate_conditionals(derived_tables) -> None:
    evidence = derived_tables[1]
    assert evidence["status"].value_counts().to_dict() == EXPECTED_STATUS_COUNTS
    candidate = evidence[evidence["is_class_I_candidate"]]
    assert len(candidate) == 8
    assert dict(zip(candidate["evidence_code"], candidate["status"])) == (
        EXPECTED_CANDIDATE_STATUS
    )
    assert candidate.loc[candidate["evidence_code"].eq("E6"), "evidence_value"].iloc[0] == (
        "Pending QA"
    )
    assert candidate.loc[candidate["evidence_code"].eq("E8"), "evidence_value"].iloc[0] == (
        "Candidate"
    )


def test_known_cold_unavailable_is_scope_exclusion_not_baseline_invalidation(
    derived_tables,
) -> None:
    source = derived_tables[0].set_index("analysis_condition")
    row = source.loc["Known-cold source unavailable"]
    assert row["final_class"] == "Class IV"
    assert row["supported_interpretation"] == (
        "Excluded from strict reference-sensitivity analysis"
    )
    assert "does not invalidate the accepted baseline scene" in row["basis"]


def test_narrow_v50_hls_inventory_and_identity(loaded_inputs) -> None:
    hls = loaded_inputs[1]
    assert hls["study_years"] == [2015, 2017, 2023]
    for key, expected in EXPECTED_HLS_COUNTS.items():
        assert hls[key] == expected
    assert hls["object_years"] * hls["hls_temporal_bins"] == (
        hls["hls_object_bin_records"]
    )
    assert hls["identity_check"] == "4196 * 17 = 71332"
    assert hls["interpretation_status"] == "Class I candidate"
    assert hls["completed_class_I_assignment"] is False


def test_hls_role_and_prohibition_set_remain_support_only(loaded_inputs) -> None:
    hls = loaded_inputs[1]
    assert hls["hls_role"] == "canopy-trajectory and interpolation-plausibility QA"
    assert set(hls["hls_not_used_for"]) == ic.EXPECTED_HLS_NOT_USED_FOR
    assert hls["source_notebook_sha256"] == ic.EXPECTED_V50_NOTEBOOK_SHA256
    assert hls["source_evidence_member_sha256"] == ic.EXPECTED_V50_EVIDENCE_SHA256


def test_text_statistics_report_counts_without_fabricated_sd(derived_tables) -> None:
    statistics = derived_tables[2]
    required = {
        "statistic",
        "unit",
        "observational_unit",
        "n",
        "ddof",
        "count",
        "percentage",
        "mean",
        "standard_deviation",
        "median",
        "minimum",
        "maximum",
        "interpretation_boundary",
    }
    assert len(statistics) == 15
    assert required.issubset(statistics.columns)
    assert statistics["ddof"].eq("not_applicable").all()
    assert statistics["mean"].astype(str).str.len().eq(0).all()
    assert statistics["standard_deviation"].astype(str).str.len().eq(0).all()
    assert statistics["median"].astype(str).str.len().eq(0).all()
    assert statistics["minimum"].astype(str).str.len().eq(0).all()
    assert statistics["maximum"].astype(str).str.len().eq(0).all()
    assert statistics["standard_deviation_definition"].str.contains(
        "Not calculated", regex=False
    ).all()


def test_text_statistics_preserve_class_and_evidence_counts(derived_tables) -> None:
    statistics = derived_tables[2]
    class_counts = statistics[
        statistics["statistic"].eq("classification_assignment_count")
    ].set_index("category")["count"].astype(int).to_dict()
    assert class_counts == {
        "Class II": 1,
        "Class III": 1,
        "Class IV": 1,
        "Class I candidate": 1,
    }
    completed = statistics[
        statistics["statistic"].eq("completed_class_I_assignment_count")
    ]
    assert len(completed) == 1
    assert int(completed["count"].iloc[0]) == 0
    status_counts = statistics[
        statistics["statistic"].eq("evidence_status_count")
    ].set_index("category")["count"].astype(int).to_dict()
    assert status_counts == EXPECTED_STATUS_COUNTS


def test_text_statistics_preserve_v50_inventory_counts(derived_tables) -> None:
    statistics = derived_tables[2].set_index("statistic")
    assert int(statistics.loc["study_year_count", "count"]) == 3
    assert int(statistics.loc["object_year_count", "count"]) == 4_196
    assert int(statistics.loc["accepted_anchor_interval_count", "count"]) == 26_085
    assert int(statistics.loc["hls_temporal_bin_count", "count"]) == 17
    assert int(statistics.loc["hls_object_bin_record_count", "count"]) == 71_332


def test_derived_validation_passes_all_40_checks(loaded_inputs, derived_tables) -> None:
    classes, hls, classification, _ = loaded_inputs
    interpretation, evidence, statistics, runtime = derived_tables
    report = ic.validate_derived_outputs(
        interpretation,
        evidence,
        statistics,
        classes=classes,
        hls=hls,
        classification_input=classification,
        runtime_validation_summary=runtime,
        contract_path=CONTRACT_PATH,
    )
    assert report["status"] == "PASS"
    assert report["checks"] == report["passed"] == 40
    assert report["warnings"] == report["failed"] == 0
    assert {item["check_id"] for item in report["check_results"]} == {
        f"INT-{number:03d}" for number in range(1, 41)
    }
    assert all(item["status"] == "PASS" for item in report["check_results"])
    assert report["standard_deviation"]["status"] == "not_applicable"
    assert report["standard_deviation"]["ddof"] == "not_applicable"


def test_run_writes_stable_outputs_and_semantic_execution_record(
    generated_outputs,
) -> None:
    paths = {path.name: path for path in generated_outputs.paths()}
    assert set(paths) == EXPECTED_OUTPUT_NAMES
    for name, expected_hash in EXPECTED_STABLE_OUTPUT_HASHES.items():
        assert paths[name].is_file()
        assert paths[name].stat().st_size > 0
        assert _sha256(paths[name]) == expected_hash

    validation = json.loads(
        generated_outputs.interpretation_classes_validation.read_text(encoding="utf-8")
    )
    assert validation["status"] == "PASS"
    assert validation["checks"] == validation["passed"] == 40
    assert validation["warnings"] == validation["failed"] == 0


def test_execution_record_reconciles_current_contract_inputs_and_outputs(
    generated_outputs,
) -> None:
    record = json.loads(
        generated_outputs.interpretation_classes_execution_record.read_text(
            encoding="utf-8"
        )
    )
    assert record["module"] == "interpretation_classes"
    assert record["phase3_package"] == "PHASE3_CURATED_INPUT_LAYER_FINAL_002"
    assert record["phase2_figure8_authority"] == ic.PHASE2_VERSION
    assert record["module_relative_path"] == ic.MODULE_RELATIVE_PATH
    assert record["module_sha256"] == _sha256(SOURCE_PATH)
    assert record["contract_sha256"] == _sha256(CONTRACT_PATH)
    assert record["authorized_inputs"] == {
        path: {"sha256": digest} for path, digest in EXPECTED_INPUT_HASHES.items()
    }
    assert record["runtime_input_validation"]["passed"] == 68
    assert record["derived_output_validation"] == {
        "status": "PASS",
        "checks": 40,
        "passed": 40,
        "warnings": 0,
        "failed": 0,
    }
    assert record["classification_guardrails"] == {
        "class_I_candidate_count": 1,
        "completed_class_I_assignment_count": 0,
        "candidate_is_completed_class_I": False,
    }
    assert record["v50_narrow_authority"]["object_years"] == 4_196
    assert record["v50_narrow_authority"]["hls_object_bin_records"] == 71_332
    assert record["v50_narrow_authority"]["completed_class_I_assignment"] is False
    assert record["statistical_decisions"]["status"] == "not_applicable"
    assert record["figure_change"] is False
    assert record["scientific_boundary"] == ic.INTERPRETATION_BOUNDARY

    for name, item in record["outputs"].items():
        path = generated_outputs.interpretation_classes_execution_record.parent / name
        assert item["size_bytes"] == path.stat().st_size
        assert item["sha256"] == _sha256(path)


def test_second_execution_is_byte_deterministic(generated_outputs, tmp_path: Path) -> None:
    second = ic.run_interpretation_classes(
        PHASE3_ROOT,
        tmp_path / "second_run",
        contract_path=CONTRACT_PATH,
    )
    first_hashes = {path.name: _sha256(path) for path in generated_outputs.paths()}
    second_hashes = {path.name: _sha256(path) for path in second.paths()}
    assert second_hashes == first_hashes
    for name, expected_hash in EXPECTED_STABLE_OUTPUT_HASHES.items():
        assert first_hashes[name] == expected_hash


def test_existing_outputs_require_explicit_overwrite(generated_outputs) -> None:
    output_dir = generated_outputs.interpretation_classes_validation.parent
    with pytest.raises(ic.InterpretationClassesError, match="Refusing to overwrite"):
        ic.run_interpretation_classes(
            PHASE3_ROOT,
            output_dir,
            contract_path=CONTRACT_PATH,
        )


def test_explicit_overwrite_is_byte_deterministic(generated_outputs) -> None:
    output_dir = generated_outputs.interpretation_classes_validation.parent
    before = {path.name: _sha256(path) for path in generated_outputs.paths()}
    overwritten = ic.run_interpretation_classes(
        PHASE3_ROOT,
        output_dir,
        contract_path=CONTRACT_PATH,
        overwrite=True,
    )
    after = {path.name: _sha256(path) for path in overwritten.paths()}
    assert after == before


def test_cli_success_and_refusal_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "cli_outputs"
    args = [
        "--phase3-root",
        str(PHASE3_ROOT),
        "--output-dir",
        str(output_dir),
        "--contract",
        str(CONTRACT_PATH),
    ]
    assert ic.main(args) == 0
    success = capsys.readouterr()
    assert "Class I candidates: 1" in success.out
    assert "Completed Class I assignments: 0" in success.out
    assert "Standard deviation: not applicable" in success.out
    assert success.err == ""

    assert ic.main(args) == 2
    failure = capsys.readouterr()
    assert "INTERPRETATION_CLASSES_ERROR" in failure.err
    assert "Refusing to overwrite" in failure.err


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("phase2_version", "Unexpected Figure 8 authority"),
        ("v50_scope", "V50 authority is broader"),
        ("condition_order", "Figure 8 condition order changed"),
        ("class_assignment", "Unexpected class"),
        ("status_count", "Evidence-status counts changed"),
        ("csv_json_mismatch", "JSON/CSV evidence mismatch"),
        ("candidate_count", "Classification assignments changed"),
        ("hls_count", "Narrow V50/HLS value changed"),
        ("hls_identity", "Narrow V50/HLS value changed"),
        ("hls_role", "HLS role is not temporal-support QA"),
        ("hls_prohibition", "HLS prohibition list changed"),
        ("obsolete_frames", "Obsolete known-cold evidence entered Figure 8"),
    ],
)
def test_malformed_authority_is_rejected(
    loaded_inputs,
    mutation: str,
    message: str,
) -> None:
    classes, hls, classification, _ = loaded_inputs
    classes_copy = copy.deepcopy(classes)
    hls_copy = copy.deepcopy(hls)
    classification_copy = classification.copy(deep=True)

    if mutation == "phase2_version":
        classes_copy["version"] = "OBSOLETE"
    elif mutation == "v50_scope":
        classes_copy["authority"]["V50"] = "All manuscript figures"
    elif mutation == "condition_order":
        classes_copy["matrix_rows"][0], classes_copy["matrix_rows"][1] = (
            classes_copy["matrix_rows"][1],
            classes_copy["matrix_rows"][0],
        )
    elif mutation == "class_assignment":
        classes_copy["matrix_rows"][0]["final_class"] = "Class I"
    elif mutation == "status_count":
        classes_copy["matrix_rows"][0]["evidence"][0]["status"] = "conditional"
    elif mutation == "csv_json_mismatch":
        classification_copy.loc[0, "E1"] = "Changed"
    elif mutation == "candidate_count":
        classification_copy.loc[
            classification_copy["final_class"].eq("Class I candidate"), "final_class"
        ] = "Class II"
    elif mutation == "hls_count":
        hls_copy["accepted_anchor_intervals"] = 26_084
    elif mutation == "hls_identity":
        hls_copy["hls_object_bin_records"] = 71_331
    elif mutation == "hls_role":
        hls_copy["hls_role"] = "direct ET conversion"
    elif mutation == "hls_prohibition":
        hls_copy["hls_not_used_for"] = ["ET validation"]
    elif mutation == "obsolete_frames":
        hls_copy["audit_note"] = "102 logical frames"
    else:  # pragma: no cover
        raise AssertionError(mutation)

    with pytest.raises(ic.InterpretationClassesError, match=message):
        ic._assert_input_structure(classes_copy, hls_copy, classification_copy)


def test_completed_class_I_promotion_is_rejected(loaded_inputs, monkeypatch) -> None:
    classes, hls, classification, _ = loaded_inputs
    classes_copy = copy.deepcopy(classes)
    classification_copy = classification.copy(deep=True)
    condition = "2011 active-canopy large-ETrF scenes"
    classes_copy["matrix_rows"][0]["final_class"] = "Class I"
    classification_copy.loc[
        classification_copy["analysis_condition"].eq(condition), "final_class"
    ] = "Class I"
    monkeypatch.setitem(ic.EXPECTED_FINAL_CLASS, condition, "Class I")
    with pytest.raises(
        ic.InterpretationClassesError,
        match="Completed Class I assignments are prohibited",
    ):
        ic._assert_input_structure(classes_copy, hls, classification_copy)


def test_altered_input_hash_is_rejected(tmp_path: Path) -> None:
    altered_root = tmp_path / "PHASE3_CURATED_INPUT_LAYER_FINAL_002"
    target = altered_root / ic.CLASSIFICATION_RELATIVE_PATH
    target.parent.mkdir(parents=True)
    target.write_bytes((PHASE3_ROOT / ic.CLASSIFICATION_RELATIVE_PATH).read_bytes() + b"\n")
    for relative_path in (ic.CLASSES_RELATIVE_PATH, ic.HLS_RELATIVE_PATH):
        destination = altered_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((PHASE3_ROOT / relative_path).read_bytes())

    with pytest.raises(RuntimeInputError, match="SHA-256"):
        ic.load_interpretation_inputs(altered_root, contract_path=CONTRACT_PATH)


def test_missing_required_input_is_rejected(tmp_path: Path) -> None:
    incomplete_root = tmp_path / "PHASE3_CURATED_INPUT_LAYER_FINAL_002"
    for relative_path in (ic.CLASSES_RELATIVE_PATH, ic.CLASSIFICATION_RELATIVE_PATH):
        destination = incomplete_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((PHASE3_ROOT / relative_path).read_bytes())

    with pytest.raises(RuntimeInputError, match="missing|not found|Missing"):
        ic.load_interpretation_inputs(incomplete_root, contract_path=CONTRACT_PATH)


def test_repository_wide_phase3_validation_remains_1708_of_1708() -> None:
    report = validate_phase3_root(PHASE3_ROOT, contract_path=CONTRACT_PATH)
    summary = report.summary()
    assert summary["status"] == "PASS"
    assert summary["checks"] == summary["passed"] == 1_708
    assert summary["warnings"] == summary["failed"] == 0


def test_source_contains_no_executable_notebook_or_absolute_runtime_dependency() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    prohibited = (
        ".ipynb",
        "AWM_figures",
        "/scratch/",
        "subprocess",
        "nbconvert",
        "jupyter",
    )
    for token in prohibited:
        assert token not in source
    assert "validate_module_inputs" in source
    assert ic.CLASSES_RELATIVE_PATH in source
    assert ic.HLS_RELATIVE_PATH in source
    assert ic.CLASSIFICATION_RELATIVE_PATH in source
