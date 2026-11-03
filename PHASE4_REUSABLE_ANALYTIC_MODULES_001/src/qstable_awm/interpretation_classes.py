"""Figure 8 interpretation-classification sources and evidence counts.

This module reads only the three Phase 3 inputs authorized for the
``interpretation_classes`` domain. The corrected Phase 2 Figure 8 release is
the classification authority. V50 is used only for the narrow non-2011 HLS
temporal-support inventory and the single ``Class I candidate`` condition.
Historical notebooks, Phase 2 plotting code, and original workflow archives
are not runtime dependencies.

The module writes the four-condition Figure 8 classification table, a
32-record long evidence matrix, a text-ready evidence-count table, a validation
report, and an execution record. It does not render or revise Figure 8.

Statistical convention
----------------------
The accepted inputs contain categorical evidence assignments and deterministic
inventory totals, not replicate numerical observations. A scientifically
meaningful standard deviation is therefore not calculated. The text-statistics
file explicitly records counts and percentages and marks ``ddof`` and
``standard_deviation`` as not applicable rather than fabricating dispersion.

Scientific boundary
-------------------
Interpretation-support classification only. ``Class I candidate`` is not a
completed Class I assignment. HLS is canopy-trajectory and interpolation-
plausibility support only; it is not ET validation, irrigation application,
pumping, groundwater withdrawal, or water-source attribution.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

try:
    from .validation import (
        DEFAULT_CONTRACT_PATH,
        RuntimeInputError,
        load_source_contract,
        sha256_file,
        validate_module_inputs,
    )
except ImportError:  # Support direct execution before package installation.
    _SRC_ROOT = Path(__file__).resolve().parents[1]
    if str(_SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(_SRC_ROOT))
    from qstable_awm.validation import (  # type: ignore[no-redef]
        DEFAULT_CONTRACT_PATH,
        RuntimeInputError,
        load_source_contract,
        sha256_file,
        validate_module_inputs,
    )


MODULE_NAME = "interpretation_classes"
MODULE_RELATIVE_PATH = "src/qstable_awm/interpretation_classes.py"
CLASSES_RELATIVE_PATH = (
    "data/derived/interpretation_classes/interpretation_classes.json"
)
HLS_RELATIVE_PATH = (
    "data/derived/interpretation_classes/non2011_hls_temporal_support_evidence.json"
)
CLASSIFICATION_RELATIVE_PATH = (
    "data/derived/interpretation_classes/interpretation_classification.csv"
)

PHASE2_VERSION = "PHASE2_FIGURE8_PUBLICATION_FINAL_003"
EXPECTED_CONDITIONS = (
    "2011 active-canopy large-ETrF scenes",
    "Fixed 2011 frame in later years",
    "Known-cold source unavailable",
    "Non-2011 year-specific reconstruction workflow",
)
EXPECTED_FINAL_CLASS = {
    "2011 active-canopy large-ETrF scenes": "Class II",
    "Fixed 2011 frame in later years": "Class III",
    "Known-cold source unavailable": "Class IV",
    "Non-2011 year-specific reconstruction workflow": "Class I candidate",
}
EXPECTED_EVIDENCE_CODES = tuple(f"E{index}" for index in range(1, 9))
EXPECTED_EVIDENCE_LABELS = {
    "E1": "Crop",
    "E2": "Scene",
    "E3": "Physical",
    "E4": "P/Peff",
    "E5": "Object",
    "E6": "Reference",
    "E7": "Scale",
    "E8": "Status",
}
EXPECTED_STATUS_COUNTS = {
    "supported": 12,
    "material": 3,
    "conditional": 8,
    "invalid": 9,
}
EXPECTED_RELEASE_DECISIONS = {
    "v50_included_narrowly": True,
    "hls_included_as_support_only": True,
    "v49_known_cold_evidence_included": False,
    "class_I_candidate_present": True,
    "completed_class_I_assignment_present": False,
    "class_I_candidate_is_not_class_I": True,
}
EXPECTED_HLS_NOT_USED_FOR = {
    "direct ETrF conversion",
    "direct ETa conversion",
    "ET validation",
    "irrigation application",
    "pumping",
    "groundwater withdrawal",
    "water-source attribution",
}
EXPECTED_V50_NOTEBOOK_SHA256 = (
    "3456c9af48413f44e30b09d48bb6003727195642cd030adc84d1b2bf41d93f0d"
)
EXPECTED_V50_EVIDENCE_SHA256 = (
    "4208def3e69e0ff26de80618cc5a950a5cde35d137b220264efbdecc10b728b0"
)
EXPECTED_OUTPUT_NAMES = (
    "figure8_interpretation_source.csv",
    "figure8_evidence_matrix_source.csv",
    "interpretation_classes_text_statistics.csv",
    "interpretation_classes_validation.json",
    "interpretation_classes_execution_record.json",
)
CSV_ROUNDTRIP_RTOL = 1e-12
CSV_ROUNDTRIP_ATOL = 1e-15

INTERPRETATION_BOUNDARY = (
    "Interpretation-support classification only; Class I candidate is not completed "
    "Class I; HLS is temporal-support QA only; no corrected ET; no ET validation; "
    "no irrigation application; no pumping; no pumping validation; no groundwater-"
    "withdrawal validation; no groundwater attribution; no water-source attribution."
)


class InterpretationClassesError(RuntimeError):
    """Raised when Figure 8 inputs or outputs violate accepted authority."""


@dataclass(frozen=True, slots=True)
class InterpretationClassesOutputs:
    """Paths written by :func:`run_interpretation_classes`."""

    figure8_interpretation_source: Path
    figure8_evidence_matrix_source: Path
    interpretation_classes_text_statistics: Path
    interpretation_classes_validation: Path
    interpretation_classes_execution_record: Path

    def as_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}

    def paths(self) -> tuple[Path, ...]:
        return tuple(Path(value) for value in asdict(self).values())


# ---------------------------------------------------------------------------
# Input loading and authority checks
# ---------------------------------------------------------------------------


def _portable_runtime_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "report_version": summary.get("report_version"),
        "modules": list(summary.get("modules", [])),
        "status": summary.get("status"),
        "checks": int(summary.get("checks", 0)),
        "passed": int(summary.get("passed", 0)),
        "warnings": int(summary.get("warnings", 0)),
        "failed": int(summary.get("failed", 0)),
    }


def load_interpretation_inputs(
    phase3_root: Path,
    *,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Validate and load the exact three authorized Figure 8 inputs."""

    root = Path(phase3_root).expanduser().resolve()
    supplied = (
        root / CLASSES_RELATIVE_PATH,
        root / HLS_RELATIVE_PATH,
        root / CLASSIFICATION_RELATIVE_PATH,
    )
    report = validate_module_inputs(
        root,
        MODULE_NAME,
        contract_path=contract_path,
        supplied_paths=supplied,
        raise_on_error=True,
    )
    classes = json.loads(supplied[0].read_text(encoding="utf-8"))
    hls = json.loads(supplied[1].read_text(encoding="utf-8"))
    classification = pd.read_csv(supplied[2], keep_default_na=False)
    return classes, hls, classification, _portable_runtime_summary(report.summary())


def _matrix_long_from_json(classes: Mapping[str, Any]) -> pd.DataFrame:
    labels = {
        str(item["code"]): str(item["label"])
        for item in classes["evidence_dimensions"]
    }
    rows: list[dict[str, Any]] = []
    for condition_order, item in enumerate(classes["matrix_rows"], start=1):
        for evidence_order, evidence in enumerate(item["evidence"], start=1):
            code = str(evidence["code"])
            rows.append(
                {
                    "condition_order": condition_order,
                    "evidence_order": evidence_order,
                    "analysis_condition": str(item["condition"]),
                    "evidence_code": code,
                    "evidence_label": labels[code],
                    "evidence_value": str(evidence["value"]),
                    "status": str(evidence["status"]),
                    "final_class": str(item["final_class"]),
                    "supported_interpretation": str(
                        item["supported_interpretation"]
                    ),
                    "basis": str(item["basis"]),
                }
            )
    return pd.DataFrame(rows)


def _normalized_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return "" if text.lower() == "nan" else text


def _assert_input_structure(
    classes: Mapping[str, Any],
    hls: Mapping[str, Any],
    classification: pd.DataFrame,
) -> None:
    """Apply strict Figure 8 and narrow-V50 scientific checks."""

    if classes.get("version") != PHASE2_VERSION:
        raise InterpretationClassesError(
            f"Unexpected Figure 8 authority: {classes.get('version')!r}"
        )
    if classes.get("figure") != "Figure 8":
        raise InterpretationClassesError("Interpretation authority is not Figure 8")
    if classes.get("release_decision") != EXPECTED_RELEASE_DECISIONS:
        raise InterpretationClassesError(
            "Figure 8 release decisions do not match the corrected authority"
        )

    authority = classes.get("authority")
    if not isinstance(authority, Mapping):
        raise InterpretationClassesError("Figure 8 authority mapping is missing")
    if authority.get("V50") != "Class I-candidate and HLS temporal-support evidence only":
        raise InterpretationClassesError("V50 authority is broader than the narrow accepted scope")
    if "V49" in authority or any("V49" in str(value) for value in authority.values()):
        raise InterpretationClassesError("Obsolete V49 evidence entered Figure 8 authority")

    dimensions = classes.get("evidence_dimensions")
    if not isinstance(dimensions, list) or len(dimensions) != 8:
        raise InterpretationClassesError("Expected eight Figure 8 evidence dimensions")
    observed_labels = {
        str(item.get("code")): str(item.get("label")) for item in dimensions
    }
    if observed_labels != EXPECTED_EVIDENCE_LABELS:
        raise InterpretationClassesError(
            f"Evidence dimensions changed: {observed_labels}"
        )

    definitions = classes.get("class_definitions")
    if not isinstance(definitions, list) or len(definitions) != 4:
        raise InterpretationClassesError("Expected four interpretation-class definitions")
    if tuple(str(item.get("class_id")) for item in definitions) != ("I", "II", "III", "IV"):
        raise InterpretationClassesError("Interpretation-class definition order changed")

    matrix_rows = classes.get("matrix_rows")
    if not isinstance(matrix_rows, list) or len(matrix_rows) != 4:
        raise InterpretationClassesError("Expected four Figure 8 matrix rows")
    conditions = tuple(str(item.get("condition")) for item in matrix_rows)
    if conditions != EXPECTED_CONDITIONS:
        raise InterpretationClassesError(
            f"Figure 8 condition order changed: {conditions}"
        )
    for item in matrix_rows:
        condition = str(item["condition"])
        if str(item["final_class"]) != EXPECTED_FINAL_CLASS[condition]:
            raise InterpretationClassesError(
                f"Unexpected class for {condition!r}: {item['final_class']!r}"
            )
        evidence = item.get("evidence")
        if not isinstance(evidence, list) or len(evidence) != 8:
            raise InterpretationClassesError(
                f"Expected eight evidence cells for {condition!r}"
            )
        codes = tuple(str(cell.get("code")) for cell in evidence)
        if codes != EXPECTED_EVIDENCE_CODES:
            raise InterpretationClassesError(
                f"Evidence-code order changed for {condition!r}: {codes}"
            )

    if len(classification) != 4:
        raise InterpretationClassesError(
            f"Expected four classification rows; found {len(classification)}"
        )
    if classification.duplicated(["analysis_condition"]).any():
        raise InterpretationClassesError("Classification condition keys are not unique")
    if tuple(classification["analysis_condition"]) != EXPECTED_CONDITIONS:
        raise InterpretationClassesError("Classification row order changed")
    observed_classes = dict(
        zip(classification["analysis_condition"], classification["final_class"])
    )
    if observed_classes != EXPECTED_FINAL_CLASS:
        raise InterpretationClassesError(
            f"Classification assignments changed: {observed_classes}"
        )
    candidate_count = int(classification["final_class"].eq("Class I candidate").sum())
    completed_count = int(classification["final_class"].eq("Class I").sum())
    if candidate_count != 1:
        raise InterpretationClassesError(
            f"Expected exactly one Class I candidate; found {candidate_count}"
        )
    if completed_count != 0:
        raise InterpretationClassesError(
            f"Completed Class I assignments are prohibited; found {completed_count}"
        )

    long_matrix = _matrix_long_from_json(classes)
    status_counts = long_matrix["status"].value_counts().to_dict()
    if status_counts != EXPECTED_STATUS_COUNTS:
        raise InterpretationClassesError(
            f"Evidence-status counts changed: {status_counts}"
        )
    for item in classes["matrix_rows"]:
        condition = str(item["condition"])
        row = classification[
            classification["analysis_condition"].eq(condition)
        ].iloc[0]
        if _normalized_cell(row["final_class"]) != str(item["final_class"]):
            raise InterpretationClassesError(
                f"JSON/CSV class mismatch for {condition!r}"
            )
        if _normalized_cell(row["supported_interpretation"]) != str(
            item["supported_interpretation"]
        ):
            raise InterpretationClassesError(
                f"JSON/CSV supported-use mismatch for {condition!r}"
            )
        if _normalized_cell(row["basis"]) != str(item["basis"]):
            raise InterpretationClassesError(
                f"JSON/CSV basis mismatch for {condition!r}"
            )
        by_code = {str(cell["code"]): _normalized_cell(cell["value"]) for cell in item["evidence"]}
        for code in EXPECTED_EVIDENCE_CODES:
            if _normalized_cell(row[code]) != by_code[code]:
                raise InterpretationClassesError(
                    f"JSON/CSV evidence mismatch for {condition!r}, {code}"
                )

    expected_hls = {
        "study_years": [2015, 2017, 2023],
        "object_years": 4196,
        "accepted_anchor_intervals": 26085,
        "hls_temporal_bins": 17,
        "hls_object_bin_records": 71332,
        "interpretation_status": "Class I candidate",
        "completed_class_I_assignment": False,
    }
    for key, expected in expected_hls.items():
        if hls.get(key) != expected:
            raise InterpretationClassesError(
                f"Narrow V50/HLS value changed for {key!r}: {hls.get(key)!r}"
            )
    if int(hls["object_years"]) * int(hls["hls_temporal_bins"]) != int(
        hls["hls_object_bin_records"]
    ):
        raise InterpretationClassesError("HLS object-bin identity failed")
    if hls.get("identity_check") != "4196 * 17 = 71332":
        raise InterpretationClassesError("HLS identity text changed")
    role = str(hls.get("hls_role", "")).lower()
    if "canopy-trajectory" not in role or "plausibility" not in role:
        raise InterpretationClassesError("HLS role is not temporal-support QA")
    if set(hls.get("hls_not_used_for", [])) != EXPECTED_HLS_NOT_USED_FOR:
        raise InterpretationClassesError("HLS prohibition list changed")
    if hls.get("source_notebook_sha256") != EXPECTED_V50_NOTEBOOK_SHA256:
        raise InterpretationClassesError("V50 provenance notebook hash changed")
    if hls.get("source_evidence_member_sha256") != EXPECTED_V50_EVIDENCE_SHA256:
        raise InterpretationClassesError("V50 evidence-member hash changed")

    serialized = json.dumps([classes, hls], sort_keys=True)
    for obsolete in ("102 logical frames", "69 valid frames"):
        if obsolete in serialized:
            raise InterpretationClassesError(
                f"Obsolete known-cold evidence entered Figure 8: {obsolete}"
            )


# ---------------------------------------------------------------------------
# Derived tables
# ---------------------------------------------------------------------------


def build_figure8_interpretation_source(
    classification: pd.DataFrame,
) -> pd.DataFrame:
    result = classification.copy()
    result.insert(0, "condition_order", range(1, len(result) + 1))
    result["is_class_I_candidate"] = result["final_class"].eq("Class I candidate")
    result["is_completed_class_I"] = result["final_class"].eq("Class I")
    result["figure_changed"] = False
    result["interpretation_boundary"] = INTERPRETATION_BOUNDARY
    return result


def build_figure8_evidence_matrix_source(
    classes: Mapping[str, Any],
) -> pd.DataFrame:
    result = _matrix_long_from_json(classes)
    result["is_class_I_candidate"] = result["final_class"].eq("Class I candidate")
    result["is_completed_class_I"] = result["final_class"].eq("Class I")
    result["figure_changed"] = False
    result["interpretation_boundary"] = INTERPRETATION_BOUNDARY
    return result.sort_values(["condition_order", "evidence_order"]).reset_index(drop=True)


def _count_record(
    *,
    statistic: str,
    category: str,
    count: int,
    unit: str,
    observational_unit: str,
    population_n: int,
    percentage: float | str,
    reporting_role: str,
) -> dict[str, Any]:
    return {
        "statistic": statistic,
        "category": category,
        "unit": unit,
        "observational_unit": observational_unit,
        "n": population_n,
        "ddof": "not_applicable",
        "count": count,
        "percentage": "" if percentage == "" else format(float(percentage), ".15g"),
        "mean": "",
        "standard_deviation": "",
        "median": "",
        "minimum": "",
        "maximum": "",
        "standard_deviation_definition": (
            "Not calculated: the accepted source is a deterministic categorical "
            "classification or inventory total without a scientifically defensible "
            "replicate unit."
        ),
        "reporting_role": reporting_role,
        "interpretation_boundary": INTERPRETATION_BOUNDARY,
    }


def build_text_statistics(
    interpretation_source: pd.DataFrame,
    evidence_source: pd.DataFrame,
    hls: Mapping[str, Any],
) -> pd.DataFrame:
    """Build text-ready evidence counts without fabricating an SD."""

    rows: list[dict[str, Any]] = []
    rows.append(
        _count_record(
            statistic="analysis_condition_count",
            category="all conditions",
            count=len(interpretation_source),
            unit="conditions",
            observational_unit="analysis condition",
            population_n=len(interpretation_source),
            percentage=100.0,
            reporting_role="Figure 8 condition inventory",
        )
    )
    for final_class in ("Class II", "Class III", "Class IV", "Class I candidate"):
        count = int(interpretation_source["final_class"].eq(final_class).sum())
        rows.append(
            _count_record(
                statistic="classification_assignment_count",
                category=final_class,
                count=count,
                unit="conditions",
                observational_unit="analysis condition",
                population_n=len(interpretation_source),
                percentage=100.0 * count / len(interpretation_source),
                reporting_role="Figure 8 class assignment",
            )
        )
    completed = int(interpretation_source["is_completed_class_I"].sum())
    rows.append(
        _count_record(
            statistic="completed_class_I_assignment_count",
            category="Class I",
            count=completed,
            unit="conditions",
            observational_unit="analysis condition",
            population_n=len(interpretation_source),
            percentage=100.0 * completed / len(interpretation_source),
            reporting_role="critical classification guardrail",
        )
    )

    total_cells = len(evidence_source)
    for status in ("supported", "material", "conditional", "invalid"):
        count = int(evidence_source["status"].eq(status).sum())
        rows.append(
            _count_record(
                statistic="evidence_status_count",
                category=status,
                count=count,
                unit="evidence cells",
                observational_unit="condition-evidence cell",
                population_n=total_cells,
                percentage=100.0 * count / total_cells,
                reporting_role="Figure 8 evidence matrix count",
            )
        )

    inventory = (
        ("study_year_count", "non-2011 study years", len(hls["study_years"]), "years", "study year"),
        ("object_year_count", "V50 object-years", int(hls["object_years"]), "object-years", "object-year"),
        ("accepted_anchor_interval_count", "accepted-anchor intervals", int(hls["accepted_anchor_intervals"]), "intervals", "accepted-anchor interval"),
        ("hls_temporal_bin_count", "HLS temporal bins", int(hls["hls_temporal_bins"]), "bins", "temporal bin"),
        ("hls_object_bin_record_count", "HLS object-bin records", int(hls["hls_object_bin_records"]), "records", "object-bin record"),
    )
    for statistic, category, count, unit, observational_unit in inventory:
        rows.append(
            _count_record(
                statistic=statistic,
                category=category,
                count=count,
                unit=unit,
                observational_unit=observational_unit,
                population_n=count,
                percentage="",
                reporting_role="narrow V50 temporal-support inventory",
            )
        )

    result = pd.DataFrame(rows)
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Derived-output validation
# ---------------------------------------------------------------------------


def validate_derived_outputs(
    interpretation_source: pd.DataFrame,
    evidence_source: pd.DataFrame,
    text_statistics: pd.DataFrame,
    *,
    classes: Mapping[str, Any],
    hls: Mapping[str, Any],
    classification_input: pd.DataFrame,
    runtime_validation_summary: Mapping[str, Any],
    contract_path: Path,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(
        check_id: str,
        description: str,
        observed: Any,
        expected: Any,
        passed: bool,
    ) -> None:
        checks.append(
            {
                "check_id": check_id,
                "description": description,
                "observed": observed,
                "expected": expected,
                "status": "PASS" if passed else "FAIL",
            }
        )

    add("INT-001", "Runtime-input validator status", runtime_validation_summary.get("status"), "PASS", runtime_validation_summary.get("status") == "PASS")
    add("INT-002", "Runtime-input validator failures", runtime_validation_summary.get("failed"), 0, int(runtime_validation_summary.get("failed", -1)) == 0)
    add("INT-003", "Corrected Phase 2 Figure 8 authority", classes.get("version"), PHASE2_VERSION, classes.get("version") == PHASE2_VERSION)
    add("INT-004", "Frozen classification rows", len(classification_input), 4, len(classification_input) == 4)
    add("INT-005", "Figure 8 interpretation rows", len(interpretation_source), 4, len(interpretation_source) == 4)
    add("INT-006", "Figure 8 condition uniqueness", int(interpretation_source.duplicated(["analysis_condition"]).sum()), 0, not interpretation_source.duplicated(["analysis_condition"]).any())
    add("INT-007", "Figure 8 condition order", interpretation_source["analysis_condition"].tolist(), list(EXPECTED_CONDITIONS), tuple(interpretation_source["analysis_condition"]) == EXPECTED_CONDITIONS)
    observed_classes = dict(zip(interpretation_source["analysis_condition"], interpretation_source["final_class"]))
    add("INT-008", "Accepted condition-to-class assignments", observed_classes, EXPECTED_FINAL_CLASS, observed_classes == EXPECTED_FINAL_CLASS)
    add("INT-009", "Exactly one Class I candidate", int(interpretation_source["is_class_I_candidate"].sum()), 1, int(interpretation_source["is_class_I_candidate"].sum()) == 1)
    add("INT-010", "Zero completed Class I assignments", int(interpretation_source["is_completed_class_I"].sum()), 0, int(interpretation_source["is_completed_class_I"].sum()) == 0)
    add("INT-011", "Evidence matrix rows", len(evidence_source), 32, len(evidence_source) == 32)
    add("INT-012", "Evidence matrix key uniqueness", int(evidence_source.duplicated(["analysis_condition", "evidence_code"]).sum()), 0, not evidence_source.duplicated(["analysis_condition", "evidence_code"]).any())
    per_condition = evidence_source.groupby("analysis_condition")["evidence_code"].nunique().to_dict()
    add("INT-013", "Eight evidence dimensions per condition", per_condition, {condition: 8 for condition in EXPECTED_CONDITIONS}, per_condition == {condition: 8 for condition in EXPECTED_CONDITIONS})
    status_counts = evidence_source["status"].value_counts().to_dict()
    add("INT-014", "Evidence-status counts", status_counts, EXPECTED_STATUS_COUNTS, status_counts == EXPECTED_STATUS_COUNTS)
    labels = dict(zip(evidence_source["evidence_code"], evidence_source["evidence_label"]))
    add("INT-015", "Evidence-code labels", labels, EXPECTED_EVIDENCE_LABELS, labels == EXPECTED_EVIDENCE_LABELS)
    add("INT-016", "No completed Class I evidence rows", int(evidence_source["is_completed_class_I"].sum()), 0, int(evidence_source["is_completed_class_I"].sum()) == 0)
    candidate_cells = evidence_source[evidence_source["is_class_I_candidate"]]
    add("INT-017", "Candidate row contains eight evidence cells", len(candidate_cells), 8, len(candidate_cells) == 8)
    candidate_status = dict(zip(candidate_cells["evidence_code"], candidate_cells["status"]))
    expected_candidate_status = {"E1": "supported", "E2": "supported", "E3": "supported", "E4": "supported", "E5": "supported", "E6": "conditional", "E7": "supported", "E8": "conditional"}
    add("INT-018", "Class I-candidate evidence remains conditional", candidate_status, expected_candidate_status, candidate_status == expected_candidate_status)
    add("INT-019", "V50 object-years", int(hls["object_years"]), 4196, int(hls["object_years"]) == 4196)
    add("INT-020", "V50 accepted-anchor intervals", int(hls["accepted_anchor_intervals"]), 26085, int(hls["accepted_anchor_intervals"]) == 26085)
    add("INT-021", "HLS temporal bins", int(hls["hls_temporal_bins"]), 17, int(hls["hls_temporal_bins"]) == 17)
    add("INT-022", "HLS object-bin records", int(hls["hls_object_bin_records"]), 71332, int(hls["hls_object_bin_records"]) == 71332)
    identity = int(hls["object_years"]) * int(hls["hls_temporal_bins"])
    add("INT-023", "HLS object-bin identity", identity, int(hls["hls_object_bin_records"]), identity == int(hls["hls_object_bin_records"]) == 71332)
    add("INT-024", "Non-2011 HLS study years", hls["study_years"], [2015, 2017, 2023], hls["study_years"] == [2015, 2017, 2023])
    add("INT-025", "HLS role remains support-only", hls["hls_role"], "canopy-trajectory and interpolation-plausibility QA", hls["hls_role"] == "canopy-trajectory and interpolation-plausibility QA")
    add("INT-026", "HLS prohibition set", sorted(hls["hls_not_used_for"]), sorted(EXPECTED_HLS_NOT_USED_FOR), set(hls["hls_not_used_for"]) == EXPECTED_HLS_NOT_USED_FOR)
    add("INT-027", "HLS does not complete Class I", hls["completed_class_I_assignment"], False, hls["completed_class_I_assignment"] is False)
    add("INT-028", "V50 notebook remains provenance-only", hls["source_notebook_sha256"], EXPECTED_V50_NOTEBOOK_SHA256, hls["source_notebook_sha256"] == EXPECTED_V50_NOTEBOOK_SHA256)
    add("INT-029", "V50 evidence-member hash", hls["source_evidence_member_sha256"], EXPECTED_V50_EVIDENCE_SHA256, hls["source_evidence_member_sha256"] == EXPECTED_V50_EVIDENCE_SHA256)
    add("INT-030", "Text/evidence-statistics rows", len(text_statistics), 15, len(text_statistics) == 15)
    required_columns = {"statistic", "unit", "observational_unit", "n", "ddof", "mean", "standard_deviation", "median", "minimum", "maximum", "interpretation_boundary"}
    add("INT-031", "Text-statistics definition columns", sorted(required_columns & set(text_statistics.columns)), sorted(required_columns), required_columns.issubset(text_statistics.columns))
    add("INT-032", "No fabricated standard deviation", int(text_statistics["standard_deviation"].astype(str).str.len().gt(0).sum()), 0, text_statistics["standard_deviation"].astype(str).str.len().eq(0).all())
    add("INT-033", "SD explicitly marked not applicable", sorted(text_statistics["ddof"].unique().tolist()), ["not_applicable"], text_statistics["ddof"].eq("not_applicable").all())
    observed_class_counts = text_statistics[text_statistics["statistic"].eq("classification_assignment_count")].set_index("category")["count"].astype(int).to_dict()
    add("INT-034", "Text class-assignment counts", observed_class_counts, {key: 1 for key in ("Class II", "Class III", "Class IV", "Class I candidate")}, observed_class_counts == {key: 1 for key in ("Class II", "Class III", "Class IV", "Class I candidate")})
    completed_count = int(text_statistics[text_statistics["statistic"].eq("completed_class_I_assignment_count")]["count"].iloc[0])
    add("INT-035", "Text completed-Class-I count", completed_count, 0, completed_count == 0)
    observed_status_counts = text_statistics[text_statistics["statistic"].eq("evidence_status_count")].set_index("category")["count"].astype(int).to_dict()
    add("INT-036", "Text evidence-status counts", observed_status_counts, EXPECTED_STATUS_COUNTS, observed_status_counts == EXPECTED_STATUS_COUNTS)
    contract = load_source_contract(contract_path)
    allowed = set(contract.allowed_outputs(MODULE_NAME))
    add("INT-037", "Output names authorized by source contract", sorted(allowed), sorted(EXPECTED_OUTPUT_NAMES), allowed == set(EXPECTED_OUTPUT_NAMES))
    add("INT-038", "Figure 8 design unchanged", bool(interpretation_source["figure_changed"].any() or evidence_source["figure_changed"].any()), False, not interpretation_source["figure_changed"].any() and not evidence_source["figure_changed"].any())
    boundary_ok = interpretation_source["interpretation_boundary"].str.contains("Class I candidate is not completed Class I", regex=False).all() and evidence_source["interpretation_boundary"].str.contains("no ET validation", regex=False).all()
    add("INT-039", "Scientific inference boundary", bool(boundary_ok), True, bool(boundary_ok))
    serialized_outputs = " ".join(interpretation_source.astype(str).to_numpy().ravel()) + " " + " ".join(evidence_source.astype(str).to_numpy().ravel())
    prohibited_present = [token for token in ("V46D", "V48", "V49", "102 logical frames", "69 valid frames") if token in serialized_outputs]
    add("INT-040", "No prohibited or obsolete authority in outputs", prohibited_present, [], not prohibited_present)

    failures = [check for check in checks if check["status"] == "FAIL"]
    return {
        "module": MODULE_NAME,
        "status": "PASS" if not failures else "FAIL",
        "checks": len(checks),
        "passed": len(checks) - len(failures),
        "warnings": 0,
        "failed": len(failures),
        "standard_deviation": {
            "status": "not_applicable",
            "ddof": "not_applicable",
            "reason": (
                "Figure 8 inputs are deterministic categorical assignments and inventory "
                "counts; no scientifically defensible replicate numerical unit exists."
            ),
            "reported_instead": "counts and percentages by class and evidence status",
        },
        "figure_change": False,
        "scientific_boundary": INTERPRETATION_BOUNDARY,
        "check_results": checks,
    }


# ---------------------------------------------------------------------------
# Deterministic writing and execution metadata
# ---------------------------------------------------------------------------


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as stream:
        temp = Path(stream.name)
        frame.to_csv(stream, index=False, lineterminator="\n")
    os.replace(temp, path)


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as stream:
        temp = Path(stream.name)
        stream.write(text)
    os.replace(temp, path)


def _prepare_outputs(
    output_dir: Path,
    *,
    contract_path: Path,
    overwrite: bool,
) -> InterpretationClassesOutputs:
    contract = load_source_contract(contract_path)
    allowed = set(contract.allowed_outputs(MODULE_NAME))
    expected = set(EXPECTED_OUTPUT_NAMES)
    if allowed != expected:
        raise InterpretationClassesError(
            f"Source-contract output authorization mismatch: allowed={sorted(allowed)}, "
            f"expected={sorted(expected)}"
        )
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    outputs = InterpretationClassesOutputs(
        figure8_interpretation_source=out / EXPECTED_OUTPUT_NAMES[0],
        figure8_evidence_matrix_source=out / EXPECTED_OUTPUT_NAMES[1],
        interpretation_classes_text_statistics=out / EXPECTED_OUTPUT_NAMES[2],
        interpretation_classes_validation=out / EXPECTED_OUTPUT_NAMES[3],
        interpretation_classes_execution_record=out / EXPECTED_OUTPUT_NAMES[4],
    )
    existing = [path for path in outputs.paths() if path.exists()]
    if existing and not overwrite:
        raise InterpretationClassesError(
            "Refusing to overwrite existing outputs without --overwrite: "
            + ", ".join(path.name for path in existing)
        )
    return outputs


def _file_record(path: Path) -> dict[str, Any]:
    return {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _build_execution_record(
    *,
    contract_path: Path,
    runtime_validation_summary: Mapping[str, Any],
    validation: Mapping[str, Any],
    outputs: InterpretationClassesOutputs,
    hls: Mapping[str, Any],
) -> dict[str, Any]:
    contract = load_source_contract(contract_path)
    inputs = {
        item.relative_path: {"sha256": item.sha256}
        for item in contract.files_for_module(MODULE_NAME)
    }
    output_records = {
        path.name: _file_record(path)
        for path in outputs.paths()
        if path != outputs.interpretation_classes_execution_record
    }
    return {
        "record_version": "1.0.0",
        "module": MODULE_NAME,
        "module_relative_path": MODULE_RELATIVE_PATH,
        "module_sha256": sha256_file(Path(__file__).resolve()),
        "contract_relative_path": "config/phase4_source_contract.yml",
        "contract_sha256": sha256_file(contract_path),
        "phase3_package": "PHASE3_CURATED_INPUT_LAYER_FINAL_002",
        "phase2_figure8_authority": PHASE2_VERSION,
        "authorized_inputs": inputs,
        "runtime_input_validation": dict(runtime_validation_summary),
        "derived_output_validation": {
            "status": validation["status"],
            "checks": validation["checks"],
            "passed": validation["passed"],
            "warnings": validation["warnings"],
            "failed": validation["failed"],
        },
        "outputs": output_records,
        "command_template": [
            "python",
            "-m",
            "qstable_awm.interpretation_classes",
            "--phase3-root",
            "PHASE3_CURATED_INPUT_LAYER_FINAL_002",
            "--output-dir",
            "results/interpretation_classes",
            "--contract",
            "config/phase4_source_contract.yml",
        ],
        "classification_guardrails": {
            "class_I_candidate_count": 1,
            "completed_class_I_assignment_count": 0,
            "candidate_is_completed_class_I": False,
        },
        "v50_narrow_authority": {
            "study_years": list(hls["study_years"]),
            "object_years": int(hls["object_years"]),
            "accepted_anchor_intervals": int(hls["accepted_anchor_intervals"]),
            "hls_temporal_bins": int(hls["hls_temporal_bins"]),
            "hls_object_bin_records": int(hls["hls_object_bin_records"]),
            "hls_role": hls["hls_role"],
            "completed_class_I_assignment": False,
        },
        "statistical_decisions": validation["standard_deviation"],
        "figure_change": False,
        "scientific_boundary": INTERPRETATION_BOUNDARY,
        "determinism_note": (
            "Outputs derive only from three frozen Phase 3 inputs; rows and columns "
            "are explicitly ordered and no notebook is executed."
        ),
    }


def run_interpretation_classes(
    phase3_root: Path,
    output_dir: Path,
    *,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
    overwrite: bool = False,
) -> InterpretationClassesOutputs:
    """Validate inputs, derive Figure 8 sources, and write five outputs."""

    contract_path = Path(contract_path).expanduser().resolve()
    outputs = _prepare_outputs(
        output_dir, contract_path=contract_path, overwrite=overwrite
    )
    classes, hls, classification, runtime_summary = load_interpretation_inputs(
        phase3_root, contract_path=contract_path
    )
    _assert_input_structure(classes, hls, classification)

    interpretation_source = build_figure8_interpretation_source(classification)
    evidence_source = build_figure8_evidence_matrix_source(classes)
    text_statistics = build_text_statistics(
        interpretation_source, evidence_source, hls
    )
    validation = validate_derived_outputs(
        interpretation_source,
        evidence_source,
        text_statistics,
        classes=classes,
        hls=hls,
        classification_input=classification,
        runtime_validation_summary=runtime_summary,
        contract_path=contract_path,
    )
    if validation["status"] != "PASS":
        failures = [
            check for check in validation["check_results"] if check["status"] == "FAIL"
        ]
        raise InterpretationClassesError(
            f"Derived-output validation failed: {failures}"
        )

    _atomic_write_csv(
        interpretation_source, outputs.figure8_interpretation_source
    )
    _atomic_write_csv(evidence_source, outputs.figure8_evidence_matrix_source)
    _atomic_write_csv(
        text_statistics, outputs.interpretation_classes_text_statistics
    )
    _atomic_write_json(validation, outputs.interpretation_classes_validation)

    for path, expected in (
        (outputs.figure8_interpretation_source, interpretation_source),
        (outputs.figure8_evidence_matrix_source, evidence_source),
        (outputs.interpretation_classes_text_statistics, text_statistics),
    ):
        observed = pd.read_csv(path, keep_default_na=False)
        try:
            pd.testing.assert_frame_equal(
                observed,
                expected,
                check_dtype=False,
                check_exact=False,
                rtol=CSV_ROUNDTRIP_RTOL,
                atol=CSV_ROUNDTRIP_ATOL,
            )
        except AssertionError as exc:
            raise InterpretationClassesError(
                f"CSV round-trip verification failed for {path.name}: {exc}"
            ) from exc

    serialized_validation = json.loads(
        outputs.interpretation_classes_validation.read_text(encoding="utf-8")
    )
    canonical_validation = json.loads(
        json.dumps(validation, sort_keys=True, ensure_ascii=False, allow_nan=False)
    )
    if serialized_validation != canonical_validation:
        raise InterpretationClassesError(
            "Validation JSON round-trip verification failed"
        )

    execution_record = _build_execution_record(
        contract_path=contract_path,
        runtime_validation_summary=runtime_summary,
        validation=validation,
        outputs=outputs,
        hls=hls,
    )
    _atomic_write_json(
        execution_record, outputs.interpretation_classes_execution_record
    )
    for path in outputs.paths():
        if not path.is_file() or path.stat().st_size <= 0:
            raise InterpretationClassesError(
                f"Output was not written correctly: {path}"
            )
    return outputs


def _text_summary(
    interpretation_path: Path,
    statistics_path: Path,
) -> str:
    interpretation = pd.read_csv(interpretation_path)
    statistics = pd.read_csv(statistics_path, keep_default_na=False)
    candidate = interpretation[
        interpretation["is_class_I_candidate"].astype(str).str.lower().eq("true")
    ]
    status = statistics[
        statistics["statistic"].eq("evidence_status_count")
    ].set_index("category")["count"].astype(int).to_dict()
    return "\n".join(
        [
            "Figure 8 interpretation classification:",
            f"- Conditions: {len(interpretation)}.",
            f"- Class I candidates: {len(candidate)}.",
            "- Completed Class I assignments: 0.",
            f"- Evidence cells: {sum(status.values())}; status counts={status}.",
            "- Standard deviation: not applicable to deterministic categorical assignments.",
            "- HLS remains temporal-support QA only, not ET or pumping validation.",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Derive corrected Figure 8 classification and evidence sources."
    )
    parser.add_argument("--phase3-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing interpretation-class outputs in the selected directory.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        outputs = run_interpretation_classes(
            args.phase3_root,
            args.output_dir,
            contract_path=args.contract,
            overwrite=args.overwrite,
        )
    except (
        InterpretationClassesError,
        RuntimeInputError,
        FileNotFoundError,
        ValueError,
        KeyError,
    ) as exc:
        print(f"INTERPRETATION_CLASSES_ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(outputs.as_dict(), indent=2, sort_keys=True))
    print(
        _text_summary(
            outputs.figure8_interpretation_source,
            outputs.interpretation_classes_text_statistics,
        )
    )
    return 0


__all__ = [
    "InterpretationClassesError",
    "InterpretationClassesOutputs",
    "build_figure8_evidence_matrix_source",
    "build_figure8_interpretation_source",
    "build_text_statistics",
    "load_interpretation_inputs",
    "run_interpretation_classes",
    "validate_derived_outputs",
]


if __name__ == "__main__":
    raise SystemExit(main())
