"""Fixed-frame 2011-dominance analysis for manuscript Figure 6 and text.

This module reads only the two Phase 3 tables authorized for the
``fixed_frame_dominance`` domain. It recreates the accepted Figure 6 source
records from the V47 clean-object package and adds descriptive standard
deviations requested for manuscript text. Historical notebooks and workflow
archives are not read or executed.

The accepted Figure 6 comparison holds the 2011 retained-object frame fixed and
compares the full four-year summary (2011, 2015, 2017, 2023) with the reduced
non-2011 summary (2015, 2017, 2023). The plotting subset is fixed to the
``penalized_median`` track for ETa, NIR, and gross irrigation-demand proxy depth
at Ei = 0.88.

Standard-deviation conventions
------------------------------
* County dispersion: sample SD (``ddof=1``) across the four county summaries
  for each metric and reported variable.
* Object dispersion: secondary sample SD (``ddof=1``) across the 1,140
  classifiable fixed-frame objects for each metric-specific difference or
  percentage-change variable. The 44 ``insufficient_data`` objects are not
  silently imputed and are excluded only because the accepted source fields
  are null.

These SDs describe spread across counties or fixed-frame objects. They are not
uncertainty bounds, confidence intervals, ET validation, prediction errors, or
pumping uncertainty.

Scientific boundary
-------------------
Fixed-frame stress-year sensitivity only. Removing 2011 does not create a
corrected product and the resulting changes are not uncertainty bounds. No ET
validation, observed pumping, pumping validation, groundwater-withdrawal
validation, groundwater attribution, county-wide pumping estimate, or
persistent year-specific corn water-use inference is supported.
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

import numpy as np
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


MODULE_NAME = "fixed_frame_dominance"
MODULE_RELATIVE_PATH = "src/qstable_awm/fixed_frame_dominance.py"
SENSITIVITY_RELATIVE_PATH = (
    "data/derived/fixed_frame_dominance/four_year_vs_non2011_sensitivity.csv"
)
OBJECT_RELATIVE_PATH = (
    "data/derived/fixed_frame_dominance/object_2011_dominance.csv"
)

EXPECTED_COUNTIES = ("DALLAM", "HARTLEY", "MOORE", "OCHILTREE")
EXPECTED_METRICS = ("ETa", "NIR", "Qgross_depth")
EXPECTED_SENSITIVITY_ROWS = 80
EXPECTED_PLOT_ROWS = 12
EXPECTED_OBJECTS = 1_184
EXPECTED_CLASSIFIABLE_OBJECTS = 1_140
EXPECTED_STRONG_OBJECTS = 1_139
EXPECTED_LOW_OBJECTS = 1
EXPECTED_INSUFFICIENT_OBJECTS = 44
SAMPLE_SD_DDOF = 1
CSV_ROUNDTRIP_RTOL = 1e-12
CSV_ROUNDTRIP_ATOL = 1e-15

METRIC_LABELS: Mapping[str, str] = {
    "ETa": "ETa",
    "NIR": "NIR",
    "Qgross_depth": "Qgross proxy Ei = 0.88",
}
METRIC_UNITS: Mapping[str, str] = {
    "ETa": "mm",
    "NIR": "mm",
    "Qgross_depth": "mm",
}
OBJECT_VARIABLES: Mapping[str, Mapping[str, str]] = {
    "ETa": {
        "difference": "ETa_difference_due_to_2011",
        "percent": "ETa_percent_change_without_2011",
    },
    "NIR": {
        "difference": "NIR_difference_due_to_2011",
        "percent": "NIR_percent_change_without_2011",
    },
    "Qgross_depth": {
        "difference": "Qgross_difference_due_to_2011",
        "percent": "Qgross_percent_change_without_2011",
    },
}
COUNTY_STATISTIC_VARIABLES: Mapping[str, tuple[str, str, str]] = {
    "full_4yr_mean": (
        "county_full_four_year_mean_dispersion",
        "Full four-year summary",
        "mm",
    ),
    "reduced_3yr_mean": (
        "county_reduced_non2011_mean_dispersion",
        "Reduced non-2011 summary",
        "mm",
    ),
    "difference_4yr_minus_3yr": (
        "county_full_minus_reduced_difference_dispersion",
        "Full four-year minus reduced non-2011 summary",
        "mm",
    ),
    "percent_change_when_2011_removed": (
        "county_percent_change_when_2011_removed_dispersion",
        "Percent change when 2011 is removed",
        "percent",
    ),
}

INTERPRETATION_BOUNDARY = (
    "Fixed-frame stress-year sensitivity only; removal of 2011 is not a corrected "
    "product or uncertainty bound; no ET validation; no observed pumping; no pumping "
    "validation; no groundwater-withdrawal validation; no groundwater attribution; "
    "no county-wide pumping estimate; no persistent year-specific corn water-use inference."
)

EXPECTED_OUTPUT_NAMES = (
    "figure6_fixed_frame_sensitivity_source.csv",
    "figure6_dominance_class_source.csv",
    "fixed_frame_dominance_text_statistics.csv",
    "fixed_frame_dominance_validation.json",
    "fixed_frame_dominance_execution_record.json",
)


class FixedFrameDominanceError(RuntimeError):
    """Raised when fixed-frame inputs or outputs violate the accepted authority."""


@dataclass(frozen=True, slots=True)
class FixedFrameDominanceOutputs:
    """Paths written by :func:`run_fixed_frame_dominance`."""

    figure6_fixed_frame_sensitivity_source: Path
    figure6_dominance_class_source: Path
    fixed_frame_dominance_text_statistics: Path
    fixed_frame_dominance_validation: Path
    fixed_frame_dominance_execution_record: Path

    def as_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}

    def paths(self) -> tuple[Path, ...]:
        return tuple(Path(value) for value in asdict(self).values())


# ---------------------------------------------------------------------------
# Input loading and authority checks
# ---------------------------------------------------------------------------


def _portable_runtime_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Remove machine-specific absolute paths from a validator summary."""

    return {
        "report_version": summary.get("report_version"),
        "modules": list(summary.get("modules", [])),
        "status": summary.get("status"),
        "checks": int(summary.get("checks", 0)),
        "passed": int(summary.get("passed", 0)),
        "warnings": int(summary.get("warnings", 0)),
        "failed": int(summary.get("failed", 0)),
    }


def load_fixed_frame_inputs(
    phase3_root: Path,
    *,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Validate and load the exact two authorized Phase 3 tables."""

    root = Path(phase3_root).expanduser().resolve()
    supplied = (
        root / SENSITIVITY_RELATIVE_PATH,
        root / OBJECT_RELATIVE_PATH,
    )
    report = validate_module_inputs(
        root,
        MODULE_NAME,
        contract_path=contract_path,
        supplied_paths=supplied,
        raise_on_error=True,
    )
    sensitivity = pd.read_csv(supplied[0])
    objects = pd.read_csv(supplied[1])
    return sensitivity, objects, _portable_runtime_summary(report.summary())


def _normalized_sensitivity(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    result["county"] = result["county"].astype(str).str.upper()
    result["metric"] = result["metric"].astype(str)
    result["track"] = result["track"].fillna("NA").astype(str)
    result["efficiency_scenario"] = (
        result["efficiency_scenario"].fillna("NA").astype(str)
    )
    return result


def _assert_input_structure(sensitivity: pd.DataFrame, objects: pd.DataFrame) -> None:
    """Apply analysis-specific structural and scientific checks beyond schemas."""

    if len(sensitivity) != EXPECTED_SENSITIVITY_ROWS:
        raise FixedFrameDominanceError(
            f"Expected {EXPECTED_SENSITIVITY_ROWS} sensitivity rows; found {len(sensitivity)}"
        )
    if sensitivity.duplicated(["county", "source_column"]).any():
        raise FixedFrameDominanceError(
            "Sensitivity county + source_column keys are not unique"
        )
    counties = tuple(sorted(sensitivity["county"].astype(str).unique()))
    if counties != EXPECTED_COUNTIES:
        raise FixedFrameDominanceError(
            f"Unexpected sensitivity counties: {counties}; expected={EXPECTED_COUNTIES}"
        )

    difference = sensitivity["full_4yr_mean"] - sensitivity["reduced_3yr_mean"]
    if not np.allclose(
        sensitivity["difference_4yr_minus_3yr"].to_numpy(dtype=float),
        difference.to_numpy(dtype=float),
        rtol=0,
        atol=2e-8,
    ):
        raise FixedFrameDominanceError(
            "Sensitivity full-minus-reduced difference identity failed"
        )
    percent = 100.0 * (
        sensitivity["reduced_3yr_mean"] - sensitivity["full_4yr_mean"]
    ) / sensitivity["full_4yr_mean"]
    if not np.allclose(
        sensitivity["percent_change_when_2011_removed"].to_numpy(dtype=float),
        percent.to_numpy(dtype=float),
        rtol=0,
        atol=1e-8,
    ):
        raise FixedFrameDominanceError(
            "Sensitivity percentage-change identity failed"
        )
    if not sensitivity["percent_change_when_2011_removed"].between(
        -100.0, 100.0, inclusive="both"
    ).all():
        raise FixedFrameDominanceError(
            "Sensitivity percentage change falls outside the signed [-100, 100] range"
        )

    normalized = _normalized_sensitivity(sensitivity)
    selected = normalized[
        normalized["track"].eq("penalized_median")
        & (
            (
                normalized["metric"].isin(["ETa", "NIR"])
                & normalized["efficiency_scenario"].eq("NA")
            )
            | (
                normalized["metric"].eq("Qgross_depth")
                & normalized["efficiency_scenario"].eq("Ei088")
            )
        )
    ]
    if len(selected) != EXPECTED_PLOT_ROWS:
        raise FixedFrameDominanceError(
            f"Expected {EXPECTED_PLOT_ROWS} Figure 6 sensitivity rows; found {len(selected)}"
        )
    if set(selected["metric"]) != set(EXPECTED_METRICS):
        raise FixedFrameDominanceError("Figure 6 metric selection changed")
    if selected.groupby("county")["metric"].nunique().ne(3).any():
        raise FixedFrameDominanceError(
            "Each county must contribute ETa, NIR, and Qgross depth"
        )

    if len(objects) != EXPECTED_OBJECTS:
        raise FixedFrameDominanceError(
            f"Expected {EXPECTED_OBJECTS} fixed-frame objects; found {len(objects)}"
        )
    if objects["object_id"].nunique() != EXPECTED_OBJECTS:
        raise FixedFrameDominanceError("Object identifiers are not unique")
    if objects.duplicated(["object_id"]).any():
        raise FixedFrameDominanceError("Duplicate object identifiers found")
    object_counties = tuple(sorted(objects["county"].astype(str).unique()))
    if object_counties != EXPECTED_COUNTIES:
        raise FixedFrameDominanceError(
            f"Unexpected object counties: {object_counties}; expected={EXPECTED_COUNTIES}"
        )
    if not pd.to_numeric(objects["object_area_ha"], errors="raise").gt(0).all():
        raise FixedFrameDominanceError("Object areas must be positive")
    if not objects["valid_cross_year_fraction_2011"].eq(1.0).all():
        raise FixedFrameDominanceError(
            "The fixed 2011 frame must have complete 2011 validity"
        )
    if set(objects["support_quality_class"]) != {"good support"}:
        raise FixedFrameDominanceError("Unexpected support-quality category")

    counts = objects["dominance_class"].value_counts().to_dict()
    expected_counts = {
        "strong_2011_dominance": EXPECTED_STRONG_OBJECTS,
        "low_2011_dominance": EXPECTED_LOW_OBJECTS,
        "insufficient_data": EXPECTED_INSUFFICIENT_OBJECTS,
    }
    if counts != expected_counts:
        raise FixedFrameDominanceError(
            f"Unexpected object dominance counts: {counts}; expected={expected_counts}"
        )

    classifiable = objects[objects["dominance_class"].ne("insufficient_data")]
    if len(classifiable) != EXPECTED_CLASSIFIABLE_OBJECTS:
        raise FixedFrameDominanceError(
            f"Expected {EXPECTED_CLASSIFIABLE_OBJECTS} classifiable objects; "
            f"found {len(classifiable)}"
        )
    insufficient = objects[objects["dominance_class"].eq("insufficient_data")]
    metric_fields = [
        "reduced_3yr_ETa_penalized_mean",
        "ETa_difference_due_to_2011",
        "ETa_percent_change_without_2011",
        "reduced_3yr_NIR_penalized_mean",
        "NIR_difference_due_to_2011",
        "NIR_percent_change_without_2011",
        "reduced_3yr_Qgross_penalized_Ei088_mean",
        "Qgross_difference_due_to_2011",
        "Qgross_percent_change_without_2011",
    ]
    if not insufficient[metric_fields].isna().all().all():
        raise FixedFrameDominanceError(
            "Insufficient-data objects must retain null reduced-reference metrics"
        )
    if classifiable[metric_fields].isna().any().any():
        raise FixedFrameDominanceError(
            "Classifiable objects contain unexpected null dominance metrics"
        )

    object_specs = {
        "ETa": (
            "full_4yr_ETa_penalized_mean",
            "reduced_3yr_ETa_penalized_mean",
            "ETa_difference_due_to_2011",
            "ETa_percent_change_without_2011",
        ),
        "NIR": (
            "full_4yr_NIR_penalized_mean",
            "reduced_3yr_NIR_penalized_mean",
            "NIR_difference_due_to_2011",
            "NIR_percent_change_without_2011",
        ),
        "Qgross_depth": (
            "full_4yr_Qgross_penalized_Ei088_mean",
            "reduced_3yr_Qgross_penalized_Ei088_mean",
            "Qgross_difference_due_to_2011",
            "Qgross_percent_change_without_2011",
        ),
    }
    for metric, (full, reduced, delta, pct) in object_specs.items():
        expected_delta = classifiable[full] - classifiable[reduced]
        if not np.allclose(
            classifiable[delta].to_numpy(dtype=float),
            expected_delta.to_numpy(dtype=float),
            rtol=0,
            atol=1e-8,
        ):
            raise FixedFrameDominanceError(
                f"Object difference identity failed for {metric}"
            )
        expected_pct = 100.0 * (
            classifiable[reduced] - classifiable[full]
        ) / classifiable[full]
        if not np.allclose(
            classifiable[pct].to_numpy(dtype=float),
            expected_pct.to_numpy(dtype=float),
            rtol=0,
            atol=1e-7,
        ):
            raise FixedFrameDominanceError(
                f"Object percentage-change identity failed for {metric}"
            )
        if not classifiable[pct].between(-100.0, 100.0, inclusive="both").all():
            raise FixedFrameDominanceError(
                f"Object signed percentage range failed for {metric}"
            )

    if not np.allclose(
        classifiable["full_4yr_Qgross_penalized_Ei088_mean"].to_numpy(dtype=float),
        classifiable["full_4yr_NIR_penalized_mean"].to_numpy(dtype=float) / 0.88,
        rtol=0,
        atol=2e-5,
    ):
        raise FixedFrameDominanceError(
            "Object Qgross Ei=0.88 full-summary identity failed"
        )
    if not np.allclose(
        classifiable["reduced_3yr_Qgross_penalized_Ei088_mean"].to_numpy(dtype=float),
        classifiable["reduced_3yr_NIR_penalized_mean"].to_numpy(dtype=float) / 0.88,
        rtol=0,
        atol=2e-5,
    ):
        raise FixedFrameDominanceError(
            "Object Qgross Ei=0.88 reduced-summary identity failed"
        )

    area = float(objects["object_area_ha"].sum())
    if not math.isfinite(area) or area <= 0:
        raise FixedFrameDominanceError("Fixed-frame area must be finite and positive")


# ---------------------------------------------------------------------------
# Derived source tables and descriptive statistics
# ---------------------------------------------------------------------------


def _sample_sd(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="raise").dropna()
    if len(numeric) < 2:
        return float("nan")
    return float(numeric.std(ddof=SAMPLE_SD_DDOF))


def _summary_record(
    values: pd.Series,
    *,
    statistic: str,
    statistic_label: str,
    metric: str,
    variable: str,
    unit: str,
    observational_unit: str,
    role: str,
) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="raise").dropna()
    return {
        "statistic": statistic,
        "statistic_label": statistic_label,
        "metric": metric,
        "metric_label": METRIC_LABELS[metric],
        "variable": variable,
        "unit": unit,
        "observational_unit": observational_unit,
        "n": int(len(numeric)),
        "ddof": SAMPLE_SD_DDOF,
        "mean": float(numeric.mean()),
        "standard_deviation": _sample_sd(numeric),
        "median": float(numeric.median()),
        "minimum": float(numeric.min()),
        "maximum": float(numeric.max()),
        "standard_deviation_definition": (
            f"Sample standard deviation across {observational_unit} records; ddof=1."
        ),
        "reporting_role": role,
        "interpretation_boundary": INTERPRETATION_BOUNDARY,
    }


def build_figure6_sensitivity_source(sensitivity: pd.DataFrame) -> pd.DataFrame:
    """Return the accepted 12-row Figure 6 county-metric plotting source."""

    normalized = _normalized_sensitivity(sensitivity)
    selected = normalized[
        normalized["track"].eq("penalized_median")
        & (
            (
                normalized["metric"].isin(["ETa", "NIR"])
                & normalized["efficiency_scenario"].eq("NA")
            )
            | (
                normalized["metric"].eq("Qgross_depth")
                & normalized["efficiency_scenario"].eq("Ei088")
            )
        )
    ].copy()
    county_order = {county: index for index, county in enumerate(EXPECTED_COUNTIES)}
    metric_order = {metric: index for index, metric in enumerate(EXPECTED_METRICS)}
    selected["county_order"] = selected["county"].map(county_order).astype(int)
    selected["metric_order"] = selected["metric"].map(metric_order).astype(int)
    selected["county_label"] = selected["county"].str.title()
    selected["metric_label"] = selected["metric"].map(METRIC_LABELS)
    selected["unit"] = selected["metric"].map(METRIC_UNITS)
    selected["comparison"] = "full_4yr_vs_reduced_non2011"
    selected["full_year_set"] = "2011|2015|2017|2023"
    selected["reduced_year_set"] = "2015|2017|2023"
    selected["figure_changed"] = False
    selected["interpretation_boundary"] = INTERPRETATION_BOUNDARY
    columns = [
        "county_order",
        "metric_order",
        "county",
        "county_label",
        "metric",
        "metric_label",
        "unit",
        "track",
        "efficiency_scenario",
        "source_column",
        "full_4yr_mean",
        "reduced_3yr_mean",
        "difference_4yr_minus_3yr",
        "percent_change_when_2011_removed",
        "dominance_flag",
        "comparison",
        "full_year_set",
        "reduced_year_set",
        "figure_changed",
        "interpretation_boundary",
    ]
    return selected.sort_values(["county_order", "metric_order"])[columns].reset_index(
        drop=True
    )


def build_figure6_dominance_source(objects: pd.DataFrame) -> pd.DataFrame:
    """Return object-dominance counts and fixed-frame population context."""

    class_order = (
        "strong_2011_dominance",
        "low_2011_dominance",
        "insufficient_data",
    )
    labels = {
        "strong_2011_dominance": "Strong 2011 dominance",
        "low_2011_dominance": "Low dominance",
        "insufficient_data": "Insufficient data",
    }
    total_objects = int(len(objects))
    total_area = float(objects["object_area_ha"].sum())
    classifiable = int(objects["dominance_class"].ne("insufficient_data").sum())
    rows: list[dict[str, Any]] = []
    for order, class_name in enumerate(class_order):
        subset = objects[objects["dominance_class"].eq(class_name)]
        count = int(len(subset))
        area = float(subset["object_area_ha"].sum())
        is_classifiable = class_name != "insufficient_data"
        rows.append(
            {
                "display_order": order,
                "dominance_class": class_name,
                "dominance_label": labels[class_name],
                "n_objects": count,
                "percent_all_objects": 100.0 * count / total_objects,
                "percent_classifiable_objects": (
                    100.0 * count / classifiable if is_classifiable else float("nan")
                ),
                "object_area_ha": area,
                "percent_fixed_frame_area": 100.0 * area / total_area,
                "is_classifiable": is_classifiable,
                "total_fixed_frame_objects": total_objects,
                "classifiable_objects": classifiable,
                "fixed_frame_area_ha": total_area,
                "figure_changed": False,
                "interpretation_boundary": INTERPRETATION_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def build_text_statistics(
    sensitivity_source: pd.DataFrame,
    objects: pd.DataFrame,
) -> pd.DataFrame:
    """Return county- and object-unit descriptive statistics with explicit SDs."""

    rows: list[dict[str, Any]] = []
    for metric in EXPECTED_METRICS:
        subset = sensitivity_source[sensitivity_source["metric"].eq(metric)]
        for variable, (statistic, label, unit) in COUNTY_STATISTIC_VARIABLES.items():
            rows.append(
                _summary_record(
                    subset[variable],
                    statistic=statistic,
                    statistic_label=label,
                    metric=metric,
                    variable=variable,
                    unit=unit,
                    observational_unit="county",
                    role="primary text statistic",
                )
            )

    classifiable = objects[objects["dominance_class"].ne("insufficient_data")].copy()
    for metric in EXPECTED_METRICS:
        for kind, column in OBJECT_VARIABLES[metric].items():
            if kind == "difference":
                statistic = "object_full_minus_reduced_difference_dispersion"
                label = "Object full four-year minus reduced non-2011 difference"
                unit = "mm"
            else:
                statistic = "object_percent_change_when_2011_removed_dispersion"
                label = "Object percent change when 2011 is removed"
                unit = "percent"
            rows.append(
                _summary_record(
                    classifiable[column],
                    statistic=statistic,
                    statistic_label=label,
                    metric=metric,
                    variable=column,
                    unit=unit,
                    observational_unit="fixed-frame object",
                    role="secondary object-level diagnostic",
                )
            )

    result = pd.DataFrame(rows)
    metric_order = {metric: index for index, metric in enumerate(EXPECTED_METRICS)}
    role_order = {"primary text statistic": 0, "secondary object-level diagnostic": 1}
    result["_metric_order"] = result["metric"].map(metric_order)
    result["_role_order"] = result["reporting_role"].map(role_order)
    result = result.sort_values(
        ["_role_order", "_metric_order", "statistic", "variable"]
    ).drop(columns=["_metric_order", "_role_order"])
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Derived-output validation
# ---------------------------------------------------------------------------


def validate_derived_outputs(
    sensitivity_source: pd.DataFrame,
    dominance_source: pd.DataFrame,
    text_statistics: pd.DataFrame,
    *,
    sensitivity_input: pd.DataFrame,
    object_input: pd.DataFrame,
    runtime_validation_summary: Mapping[str, Any],
    contract_path: Path,
) -> dict[str, Any]:
    """Return deterministic module-specific validation checks."""

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

    add(
        "FFD-001",
        "Runtime-input validator status",
        runtime_validation_summary.get("status"),
        "PASS",
        runtime_validation_summary.get("status") == "PASS",
    )
    add(
        "FFD-002",
        "Runtime-input validator failures",
        runtime_validation_summary.get("failed"),
        0,
        int(runtime_validation_summary.get("failed", -1)) == 0,
    )
    add(
        "FFD-003",
        "Frozen sensitivity row count",
        len(sensitivity_input),
        EXPECTED_SENSITIVITY_ROWS,
        len(sensitivity_input) == EXPECTED_SENSITIVITY_ROWS,
    )
    add(
        "FFD-004",
        "Frozen object population",
        len(object_input),
        EXPECTED_OBJECTS,
        len(object_input) == EXPECTED_OBJECTS,
    )
    add(
        "FFD-005",
        "Figure 6 sensitivity row count",
        len(sensitivity_source),
        EXPECTED_PLOT_ROWS,
        len(sensitivity_source) == EXPECTED_PLOT_ROWS,
    )
    add(
        "FFD-006",
        "Figure 6 county-metric key uniqueness",
        int(sensitivity_source.duplicated(["county", "metric"]).sum()),
        0,
        not sensitivity_source.duplicated(["county", "metric"]).any(),
    )
    add(
        "FFD-007",
        "Figure 6 county set",
        sorted(sensitivity_source["county"].unique().tolist()),
        list(EXPECTED_COUNTIES),
        tuple(sorted(sensitivity_source["county"].unique())) == EXPECTED_COUNTIES,
    )
    add(
        "FFD-008",
        "Figure 6 metric set",
        sorted(sensitivity_source["metric"].unique().tolist()),
        sorted(EXPECTED_METRICS),
        set(sensitivity_source["metric"]) == set(EXPECTED_METRICS),
    )
    add(
        "FFD-009",
        "Figure 6 track remains penalized median",
        sorted(sensitivity_source["track"].unique().tolist()),
        ["penalized_median"],
        sensitivity_source["track"].eq("penalized_median").all(),
    )
    correct_efficiency = (
        sensitivity_source.loc[
            sensitivity_source["metric"].isin(["ETa", "NIR"]),
            "efficiency_scenario",
        ].eq("NA").all()
        and sensitivity_source.loc[
            sensitivity_source["metric"].eq("Qgross_depth"),
            "efficiency_scenario",
        ].eq("Ei088").all()
    )
    add(
        "FFD-010",
        "Figure 6 efficiency-scenario filter",
        bool(correct_efficiency),
        True,
        bool(correct_efficiency),
    )
    add(
        "FFD-011",
        "Figure 6 full-minus-reduced identity",
        bool(
            np.allclose(
                sensitivity_source["difference_4yr_minus_3yr"],
                sensitivity_source["full_4yr_mean"]
                - sensitivity_source["reduced_3yr_mean"],
                rtol=0,
                atol=2e-8,
            )
        ),
        True,
        bool(
            np.allclose(
                sensitivity_source["difference_4yr_minus_3yr"],
                sensitivity_source["full_4yr_mean"]
                - sensitivity_source["reduced_3yr_mean"],
                rtol=0,
                atol=2e-8,
            )
        ),
    )
    expected_pct = 100.0 * (
        sensitivity_source["reduced_3yr_mean"] - sensitivity_source["full_4yr_mean"]
    ) / sensitivity_source["full_4yr_mean"]
    pct_identity = np.allclose(
        sensitivity_source["percent_change_when_2011_removed"],
        expected_pct,
        rtol=0,
        atol=1e-8,
    )
    add(
        "FFD-012",
        "Figure 6 signed percentage-change identity",
        bool(pct_identity),
        True,
        bool(pct_identity),
    )
    add(
        "FFD-013",
        "All selected Figure 6 percentage changes are reductions",
        float(sensitivity_source["percent_change_when_2011_removed"].max()),
        "<0",
        bool(sensitivity_source["percent_change_when_2011_removed"].lt(0).all()),
    )
    add(
        "FFD-014",
        "Dominance source row count",
        len(dominance_source),
        3,
        len(dominance_source) == 3,
    )
    observed_counts = dict(
        zip(dominance_source["dominance_class"], dominance_source["n_objects"])
    )
    expected_counts = {
        "strong_2011_dominance": EXPECTED_STRONG_OBJECTS,
        "low_2011_dominance": EXPECTED_LOW_OBJECTS,
        "insufficient_data": EXPECTED_INSUFFICIENT_OBJECTS,
    }
    add(
        "FFD-015",
        "Accepted dominance-class counts",
        observed_counts,
        expected_counts,
        observed_counts == expected_counts,
    )
    add(
        "FFD-016",
        "Dominance counts reconcile retained population",
        int(dominance_source["n_objects"].sum()),
        EXPECTED_OBJECTS,
        int(dominance_source["n_objects"].sum()) == EXPECTED_OBJECTS,
    )
    add(
        "FFD-017",
        "Classifiable object population",
        int(dominance_source["classifiable_objects"].iloc[0]),
        EXPECTED_CLASSIFIABLE_OBJECTS,
        dominance_source["classifiable_objects"].eq(
            EXPECTED_CLASSIFIABLE_OBJECTS
        ).all(),
    )
    strong_row = dominance_source[
        dominance_source["dominance_class"].eq("strong_2011_dominance")
    ].iloc[0]
    add(
        "FFD-018",
        "Strong dominance fraction across all objects",
        float(strong_row["percent_all_objects"]),
        100.0 * EXPECTED_STRONG_OBJECTS / EXPECTED_OBJECTS,
        math.isclose(
            float(strong_row["percent_all_objects"]),
            100.0 * EXPECTED_STRONG_OBJECTS / EXPECTED_OBJECTS,
            rel_tol=0,
            abs_tol=1e-12,
        ),
    )
    add(
        "FFD-019",
        "Strong dominance fraction across classifiable objects",
        float(strong_row["percent_classifiable_objects"]),
        100.0 * EXPECTED_STRONG_OBJECTS / EXPECTED_CLASSIFIABLE_OBJECTS,
        math.isclose(
            float(strong_row["percent_classifiable_objects"]),
            100.0 * EXPECTED_STRONG_OBJECTS / EXPECTED_CLASSIFIABLE_OBJECTS,
            rel_tol=0,
            abs_tol=1e-12,
        ),
    )
    input_area = float(object_input["object_area_ha"].sum())
    add(
        "FFD-020",
        "Fixed-frame area is derived from the frozen object table",
        float(dominance_source["fixed_frame_area_ha"].iloc[0]),
        input_area,
        dominance_source["fixed_frame_area_ha"].apply(
            lambda value: math.isclose(
                float(value), input_area, rel_tol=0, abs_tol=1e-8
            )
        ).all(),
    )
    add(
        "FFD-021",
        "Dominance-class areas reconcile frozen fixed-frame area",
        float(dominance_source["object_area_ha"].sum()),
        input_area,
        math.isclose(
            float(dominance_source["object_area_ha"].sum()),
            input_area,
            rel_tol=0,
            abs_tol=1e-8,
        ),
    )
    add(
        "FFD-022",
        "Text-statistics row count",
        len(text_statistics),
        18,
        len(text_statistics) == 18,
    )
    add(
        "FFD-023",
        "Text-statistics key uniqueness",
        int(text_statistics.duplicated(["statistic", "metric", "variable"]).sum()),
        0,
        not text_statistics.duplicated(["statistic", "metric", "variable"]).any(),
    )
    county_stats = text_statistics[
        text_statistics["observational_unit"].eq("county")
    ]
    add(
        "FFD-024",
        "County SD uses n=4 and ddof=1",
        {
            "n": sorted(county_stats["n"].unique().tolist()),
            "ddof": sorted(county_stats["ddof"].unique().tolist()),
        },
        {"n": [4], "ddof": [1]},
        county_stats["n"].eq(4).all() and county_stats["ddof"].eq(1).all(),
    )
    object_stats = text_statistics[
        text_statistics["observational_unit"].eq("fixed-frame object")
    ]
    add(
        "FFD-025",
        "Object SD uses n=1,140 and ddof=1",
        {
            "n": sorted(object_stats["n"].unique().tolist()),
            "ddof": sorted(object_stats["ddof"].unique().tolist()),
        },
        {"n": [EXPECTED_CLASSIFIABLE_OBJECTS], "ddof": [1]},
        object_stats["n"].eq(EXPECTED_CLASSIFIABLE_OBJECTS).all()
        and object_stats["ddof"].eq(1).all(),
    )
    add(
        "FFD-026",
        "All descriptive SDs are finite and non-negative",
        bool(
            np.isfinite(text_statistics["standard_deviation"]).all()
            and text_statistics["standard_deviation"].ge(0).all()
        ),
        True,
        bool(
            np.isfinite(text_statistics["standard_deviation"]).all()
            and text_statistics["standard_deviation"].ge(0).all()
        ),
    )
    required_fields = [
        "statistic",
        "unit",
        "observational_unit",
        "n",
        "ddof",
        "mean",
        "standard_deviation",
        "median",
        "minimum",
        "maximum",
        "interpretation_boundary",
    ]
    add(
        "FFD-027",
        "Required statistical-definition fields are complete",
        int(text_statistics[required_fields].isna().sum().sum()),
        0,
        not text_statistics[required_fields].isna().any().any(),
    )
    contract = load_source_contract(contract_path)
    allowed_outputs = contract.allowed_outputs(MODULE_NAME)
    add(
        "FFD-028",
        "Output names are authorized by source contract",
        list(allowed_outputs),
        list(EXPECTED_OUTPUT_NAMES),
        allowed_outputs == EXPECTED_OUTPUT_NAMES,
    )
    add(
        "FFD-029",
        "Figure design remains unchanged",
        bool(
            sensitivity_source["figure_changed"].eq(False).all()
            and dominance_source["figure_changed"].eq(False).all()
        ),
        True,
        bool(
            sensitivity_source["figure_changed"].eq(False).all()
            and dominance_source["figure_changed"].eq(False).all()
        ),
    )
    add(
        "FFD-030",
        "Scientific boundary is attached to all public records",
        bool(
            sensitivity_source["interpretation_boundary"]
            .eq(INTERPRETATION_BOUNDARY)
            .all()
            and dominance_source["interpretation_boundary"]
            .eq(INTERPRETATION_BOUNDARY)
            .all()
            and text_statistics["interpretation_boundary"]
            .eq(INTERPRETATION_BOUNDARY)
            .all()
        ),
        True,
        bool(
            sensitivity_source["interpretation_boundary"]
            .eq(INTERPRETATION_BOUNDARY)
            .all()
            and dominance_source["interpretation_boundary"]
            .eq(INTERPRETATION_BOUNDARY)
            .all()
            and text_statistics["interpretation_boundary"]
            .eq(INTERPRETATION_BOUNDARY)
            .all()
        ),
    )

    failures = [check for check in checks if check["status"] == "FAIL"]
    return {
        "module": MODULE_NAME,
        "status": "PASS" if not failures else "FAIL",
        "checks": len(checks),
        "passed": len(checks) - len(failures),
        "warnings": 0,
        "failed": len(failures),
        "runtime_input_validation": dict(runtime_validation_summary),
        "accepted_population": {
            "retained_objects": EXPECTED_OBJECTS,
            "classifiable_objects": EXPECTED_CLASSIFIABLE_OBJECTS,
            "strong_2011_dominance_objects": EXPECTED_STRONG_OBJECTS,
            "low_2011_dominance_objects": EXPECTED_LOW_OBJECTS,
            "insufficient_data_objects": EXPECTED_INSUFFICIENT_OBJECTS,
            "fixed_frame_area_ha": float(object_input["object_area_ha"].sum()),
        },
        "standard_deviation": {
            "county": {
                "type": "sample standard deviation",
                "observational_unit": "county",
                "n_per_metric_variable": 4,
                "ddof": SAMPLE_SD_DDOF,
            },
            "object": {
                "type": "sample standard deviation",
                "observational_unit": "fixed-frame object",
                "n_per_metric_variable": EXPECTED_CLASSIFIABLE_OBJECTS,
                "ddof": SAMPLE_SD_DDOF,
                "role": "secondary object-level diagnostic",
                "excluded_null_objects": EXPECTED_INSUFFICIENT_OBJECTS,
            },
        },
        "figure_change": False,
        "scientific_boundary": INTERPRETATION_BOUNDARY,
        "check_results": checks,
    }


# ---------------------------------------------------------------------------
# Deterministic writing and execution record
# ---------------------------------------------------------------------------


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        frame.to_csv(stream, index=False)
    os.replace(temporary, path)


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(text)
    os.replace(temporary, path)


def _prepare_outputs(
    output_dir: Path, *, overwrite: bool
) -> FixedFrameDominanceOutputs:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    outputs = FixedFrameDominanceOutputs(
        figure6_fixed_frame_sensitivity_source=out
        / "figure6_fixed_frame_sensitivity_source.csv",
        figure6_dominance_class_source=out / "figure6_dominance_class_source.csv",
        fixed_frame_dominance_text_statistics=out
        / "fixed_frame_dominance_text_statistics.csv",
        fixed_frame_dominance_validation=out
        / "fixed_frame_dominance_validation.json",
        fixed_frame_dominance_execution_record=out
        / "fixed_frame_dominance_execution_record.json",
    )
    existing = [path for path in outputs.paths() if path.exists()]
    if existing and not overwrite:
        raise FixedFrameDominanceError(
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
    outputs: FixedFrameDominanceOutputs,
) -> dict[str, Any]:
    contract = load_source_contract(contract_path)
    input_files = {
        item.relative_path: {"sha256": item.sha256}
        for item in contract.files_for_module(MODULE_NAME)
    }
    output_records = {
        path.name: _file_record(path)
        for path in outputs.paths()
        if path != outputs.fixed_frame_dominance_execution_record
    }
    return {
        "record_version": "1.0.0",
        "module": MODULE_NAME,
        "module_relative_path": MODULE_RELATIVE_PATH,
        "module_sha256": sha256_file(Path(__file__).resolve()),
        "contract_relative_path": "config/phase4_source_contract.yml",
        "contract_sha256": sha256_file(contract_path),
        "phase3_package": "PHASE3_CURATED_INPUT_LAYER_FINAL_002",
        "authorized_inputs": input_files,
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
            "qstable_awm.fixed_frame_dominance",
            "--phase3-root",
            "PHASE3_CURATED_INPUT_LAYER_FINAL_002",
            "--output-dir",
            "results/fixed_frame_dominance",
            "--contract",
            "config/phase4_source_contract.yml",
        ],
        "statistical_decisions": validation["standard_deviation"],
        "figure_change": False,
        "scientific_boundary": INTERPRETATION_BOUNDARY,
        "determinism_note": (
            "Numerical outputs derive only from the two frozen Phase 3 tables; "
            "rows and columns are explicitly ordered."
        ),
    }


def run_fixed_frame_dominance(
    phase3_root: Path,
    output_dir: Path,
    *,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
    overwrite: bool = False,
) -> FixedFrameDominanceOutputs:
    """Validate inputs, derive Figure 6/text sources, and write five outputs."""

    outputs = _prepare_outputs(output_dir, overwrite=overwrite)
    sensitivity, objects, runtime_summary = load_fixed_frame_inputs(
        phase3_root, contract_path=contract_path
    )
    _assert_input_structure(sensitivity, objects)

    sensitivity_source = build_figure6_sensitivity_source(sensitivity)
    dominance_source = build_figure6_dominance_source(objects)
    text_statistics = build_text_statistics(sensitivity_source, objects)
    validation = validate_derived_outputs(
        sensitivity_source,
        dominance_source,
        text_statistics,
        sensitivity_input=sensitivity,
        object_input=objects,
        runtime_validation_summary=runtime_summary,
        contract_path=contract_path,
    )
    if validation["status"] != "PASS":
        failures = [
            check for check in validation["check_results"] if check["status"] == "FAIL"
        ]
        raise FixedFrameDominanceError(
            f"Derived-output validation failed: {failures}"
        )

    _atomic_write_csv(
        sensitivity_source, outputs.figure6_fixed_frame_sensitivity_source
    )
    _atomic_write_csv(dominance_source, outputs.figure6_dominance_class_source)
    _atomic_write_csv(
        text_statistics, outputs.fixed_frame_dominance_text_statistics
    )
    _atomic_write_json(validation, outputs.fixed_frame_dominance_validation)

    try:
        pd.testing.assert_frame_equal(
            pd.read_csv(
                outputs.figure6_fixed_frame_sensitivity_source, keep_default_na=False
            ),
            sensitivity_source,
            check_dtype=False,
            check_exact=False,
            rtol=CSV_ROUNDTRIP_RTOL,
            atol=CSV_ROUNDTRIP_ATOL,
        )
        pd.testing.assert_frame_equal(
            pd.read_csv(outputs.figure6_dominance_class_source),
            dominance_source,
            check_dtype=False,
            check_exact=False,
            rtol=CSV_ROUNDTRIP_RTOL,
            atol=CSV_ROUNDTRIP_ATOL,
        )
        pd.testing.assert_frame_equal(
            pd.read_csv(outputs.fixed_frame_dominance_text_statistics),
            text_statistics,
            check_dtype=False,
            check_exact=False,
            rtol=CSV_ROUNDTRIP_RTOL,
            atol=CSV_ROUNDTRIP_ATOL,
        )
    except AssertionError as exc:
        raise FixedFrameDominanceError(
            f"CSV round-trip verification failed: {exc}"
        ) from exc

    with outputs.fixed_frame_dominance_validation.open(
        "r", encoding="utf-8"
    ) as stream:
        serialized_validation = json.load(stream)
    canonical_validation = json.loads(
        json.dumps(validation, sort_keys=True, ensure_ascii=False, allow_nan=False)
    )
    if serialized_validation != canonical_validation:
        raise FixedFrameDominanceError(
            "Validation JSON round-trip verification failed"
        )
    for path in (
        outputs.figure6_fixed_frame_sensitivity_source,
        outputs.figure6_dominance_class_source,
        outputs.fixed_frame_dominance_text_statistics,
        outputs.fixed_frame_dominance_validation,
    ):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FixedFrameDominanceError(
                f"Output was not written correctly: {path}"
            )

    execution_record = _build_execution_record(
        contract_path=Path(contract_path).expanduser().resolve(),
        runtime_validation_summary=runtime_summary,
        validation=validation,
        outputs=outputs,
    )
    _atomic_write_json(
        execution_record, outputs.fixed_frame_dominance_execution_record
    )
    return outputs


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def _text_summary(statistics_path: Path, dominance_path: Path) -> str:
    statistics = pd.read_csv(statistics_path)
    dominance = pd.read_csv(dominance_path)
    pct = statistics[
        statistics["statistic"].eq(
            "county_percent_change_when_2011_removed_dispersion"
        )
    ].set_index("metric")
    strong = dominance[
        dominance["dominance_class"].eq("strong_2011_dominance")
    ].iloc[0]
    lines = [
        "Fixed-frame 2011-dominance descriptive statistics:",
        "County SD: sample SD across four counties (ddof=1).",
    ]
    for metric in EXPECTED_METRICS:
        row = pct.loc[metric]
        lines.append(
            f"- {METRIC_LABELS[metric]}: {row['mean']:.3f} ± "
            f"{row['standard_deviation']:.3f}% (n={int(row['n'])} counties)."
        )
    lines.append(
        f"- Strong dominance: {int(strong['n_objects']):,} of "
        f"{int(strong['total_fixed_frame_objects']):,} objects "
        f"({strong['percent_all_objects']:.1f}% of all; "
        f"{strong['percent_classifiable_objects']:.1f}% of classifiable)."
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Derive Figure 6 fixed-frame dominance and text-statistics tables."
    )
    parser.add_argument("--phase3-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing fixed-frame-dominance outputs in the selected directory.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        outputs = run_fixed_frame_dominance(
            args.phase3_root,
            args.output_dir,
            contract_path=args.contract,
            overwrite=args.overwrite,
        )
    except (
        FixedFrameDominanceError, RuntimeInputError, FileNotFoundError, ValueError
    ) as exc:
        print(f"FIXED_FRAME_DOMINANCE_ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(outputs.as_dict(), indent=2, sort_keys=True))
    print(
        _text_summary(
            outputs.fixed_frame_dominance_text_statistics,
            outputs.figure6_dominance_class_source,
        )
    )
    return 0


__all__ = [
    "FixedFrameDominanceError",
    "FixedFrameDominanceOutputs",
    "build_figure6_dominance_source",
    "build_figure6_sensitivity_source",
    "build_text_statistics",
    "load_fixed_frame_inputs",
    "run_fixed_frame_dominance",
    "validate_derived_outputs",
]


if __name__ == "__main__":
    raise SystemExit(main())
