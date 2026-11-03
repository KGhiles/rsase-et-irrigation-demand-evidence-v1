"""Fixed-frame crop-support analysis for manuscript Figure 5 and results text.

This module reads only the three Phase 3 crop-support tables authorized by
``phase4_source_contract.yml``. It reproduces the accepted fixed-frame Figure 5
source tables and adds descriptive crop-support dispersion statistics requested
for manuscript text. Historical notebooks and workflow archives are never read
or executed.

The accepted Figure 5 quantity is the CDL corn fraction inside the fixed 2011
object frame. It is a comparability diagnostic, not an ET correction factor.
The primary threshold remains 0.70.

Standard-deviation conventions
------------------------------
* County dispersion: sample SD (``ddof=1``) across the four county-level,
  area-weighted crop-support fractions within each year.
* Object dispersion, area weighted: population-style weighted dispersion,
  ``sqrt(sum(w * (x - mean_w)^2) / sum(w))``, with fixed 2011 object area as
  the weight. This is not labelled as a sample SD because the 1,184 fixed-frame
  objects constitute the complete analytical population for this diagnostic.
* Object dispersion, unweighted: secondary sample SD (``ddof=1``) across the
  1,184 object-level crop-support fractions within each year.

Scientific boundary
-------------------
Crop-support QA and fixed-frame comparability only. No ET validation, pumping
estimate, withdrawal validation, groundwater attribution, year-specific object
creation, or implication that crop fraction is a direct ET correction factor.
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
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from .validation import (
        DEFAULT_CONTRACT_PATH,
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
        load_source_contract,
        sha256_file,
        validate_module_inputs,
    )


MODULE_NAME = "crop_support"
MODULE_RELATIVE_PATH = "src/qstable_awm/crop_support.py"
OBJECT_RELATIVE_PATH = (
    "data/derived/crop_support/fixed_frame_object_year_crop_support.csv"
)
COUNTY_RELATIVE_PATH = "data/derived/crop_support/county_year_crop_support.csv"
DOMAIN_RELATIVE_PATH = "data/derived/crop_support/domain_year_crop_support.csv"

EXPECTED_YEARS = (2011, 2015, 2017, 2023)
EXPECTED_COUNTIES = ("DALLAM", "HARTLEY", "MOORE", "OCHILTREE")
EXPECTED_OBJECT_YEARS = 4_736
EXPECTED_OBJECTS = 1_184
EXPECTED_COUNTY_ROWS = 16
EXPECTED_DOMAIN_ROWS = 4
PRIMARY_CROP_SUPPORT_THRESHOLD = 0.70
SAMPLE_SD_DDOF = 1
CSV_ROUNDTRIP_RTOL = 1e-12
CSV_ROUNDTRIP_ATOL = 1e-15
WEIGHTED_POPULATION_DDOF = 0

# Acceptance invariants only. Public outputs are recalculated from Phase 3.
EXPECTED_DOMAIN_FRACTIONS: Mapping[int, float] = {
    2011: 1.000000,
    2015: 0.560147,
    2017: 0.515189,
    2023: 0.347085,
}

INTERPRETATION_BOUNDARY = (
    "Crop-support QA only; fixed-frame comparability only; no ET validation; "
    "no pumping estimate; no withdrawal validation; no groundwater attribution; "
    "no year-specific object creation; crop fraction is not a direct ET correction factor."
)

EXPECTED_OUTPUT_NAMES = (
    "figure5_domain_crop_support_source.csv",
    "figure5_county_crop_support_source.csv",
    "crop_support_text_statistics.csv",
    "crop_support_validation.json",
    "crop_support_execution_record.json",
)


class CropSupportError(RuntimeError):
    """Raised when crop-support inputs or derived outputs violate authority."""


@dataclass(frozen=True, slots=True)
class CropSupportOutputs:
    """Paths written by :func:`run_crop_support`."""

    figure5_domain_crop_support_source: Path
    figure5_county_crop_support_source: Path
    crop_support_text_statistics: Path
    crop_support_validation: Path
    crop_support_execution_record: Path

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


def load_crop_support_inputs(
    phase3_root: Path,
    *,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Validate and load the exact three authorized Phase 3 crop tables."""

    root = Path(phase3_root).expanduser().resolve()
    relative_paths = (
        OBJECT_RELATIVE_PATH,
        COUNTY_RELATIVE_PATH,
        DOMAIN_RELATIVE_PATH,
    )
    supplied = tuple(root / relative_path for relative_path in relative_paths)
    report = validate_module_inputs(
        root,
        MODULE_NAME,
        contract_path=contract_path,
        supplied_paths=supplied,
        raise_on_error=True,
    )
    object_year = pd.read_csv(supplied[0])
    county_year = pd.read_csv(supplied[1])
    domain_year = pd.read_csv(supplied[2])
    return object_year, county_year, domain_year, _portable_runtime_summary(
        report.summary()
    )


def _assert_input_structure(
    object_year: pd.DataFrame,
    county_year: pd.DataFrame,
    domain_year: pd.DataFrame,
) -> None:
    """Apply crop-specific structural and scientific checks beyond schemas."""

    if len(object_year) != EXPECTED_OBJECT_YEARS:
        raise CropSupportError(
            f"Expected {EXPECTED_OBJECT_YEARS} object-years; found {len(object_year)}"
        )
    if object_year["object_id"].nunique() != EXPECTED_OBJECTS:
        raise CropSupportError(
            f"Expected {EXPECTED_OBJECTS} fixed objects; found "
            f"{object_year['object_id'].nunique()}"
        )
    if object_year.duplicated(["object_id", "year"]).any():
        raise CropSupportError("Object-year keys are not unique")

    years = tuple(sorted(int(value) for value in object_year["year"].unique()))
    counties = tuple(sorted(object_year["county"].astype(str).unique()))
    if years != EXPECTED_YEARS:
        raise CropSupportError(f"Unexpected years: {years}; expected={EXPECTED_YEARS}")
    if counties != EXPECTED_COUNTIES:
        raise CropSupportError(
            f"Unexpected counties: {counties}; expected={EXPECTED_COUNTIES}"
        )

    per_object_year_count = object_year.groupby("object_id")["year"].nunique()
    if not per_object_year_count.eq(len(EXPECTED_YEARS)).all():
        raise CropSupportError("The fixed object frame is incomplete across years")
    if object_year.groupby("object_id")["county"].nunique().max() != 1:
        raise CropSupportError("An object changes county across years")
    if object_year.groupby("object_id")["fixed_2011_object_area_ha"].nunique().max() != 1:
        raise CropSupportError("Fixed 2011 object area changes across years")
    if object_year.groupby("object_id")["object_area_ha"].nunique().max() != 1:
        raise CropSupportError("Object area changes across years")
    if not np.allclose(
        object_year["object_area_ha"].to_numpy(dtype=float),
        object_year["fixed_2011_object_area_ha"].to_numpy(dtype=float),
        rtol=0,
        atol=1e-10,
    ):
        raise CropSupportError("Object area does not equal the fixed 2011 frame area")

    fraction_columns = (
        "corn_fraction_raw",
        "corn_fraction",
        "supported_corn_fraction_raw",
        "supported_corn_fraction",
    )
    for column in fraction_columns:
        values = pd.to_numeric(object_year[column], errors="raise")
        if not values.between(0.0, 1.0, inclusive="both").all():
            raise CropSupportError(f"Invalid fraction outside [0, 1]: {column}")

    expected_fraction = (
        object_year["corn_area_ha"] / object_year["fixed_2011_object_area_ha"]
    ).clip(0.0, 1.0)
    if not np.allclose(
        object_year["corn_fraction"].to_numpy(dtype=float),
        expected_fraction.to_numpy(dtype=float),
        rtol=0,
        atol=1e-12,
    ):
        raise CropSupportError("Object crop-support fraction identity failed")

    county_only = county_year[county_year["county"].isin(EXPECTED_COUNTIES)].copy()
    embedded_domain = county_year[county_year["county"].eq("DOMAIN")].copy()
    if len(county_only) != EXPECTED_COUNTY_ROWS:
        raise CropSupportError(
            f"Expected {EXPECTED_COUNTY_ROWS} county-year rows; found {len(county_only)}"
        )
    if county_only.duplicated(["county", "year"]).any():
        raise CropSupportError("County-year keys are not unique")
    if len(embedded_domain) != EXPECTED_DOMAIN_ROWS:
        raise CropSupportError("County table does not contain four embedded domain rows")
    if len(domain_year) != EXPECTED_DOMAIN_ROWS:
        raise CropSupportError(
            f"Expected {EXPECTED_DOMAIN_ROWS} domain-year rows; found {len(domain_year)}"
        )
    if domain_year.duplicated(["county", "year"]).any():
        raise CropSupportError("Domain-year keys are not unique")
    if not domain_year["county"].eq("DOMAIN").all():
        raise CropSupportError("Domain-year table contains non-DOMAIN rows")

    comparison_counties = county_only[county_only["year"].ne(2011)]
    if len(comparison_counties) != 12:
        raise CropSupportError("Expected twelve non-2011 county-years")
    if not (
        comparison_counties["corn_fraction_area_weighted"]
        < PRIMARY_CROP_SUPPORT_THRESHOLD
    ).all():
        raise CropSupportError(
            "At least one non-2011 county-year reaches the 0.70 crop-support threshold"
        )


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def _sample_sd(values: Sequence[float] | pd.Series | np.ndarray) -> float:
    numeric = np.asarray(values, dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    if numeric.size < 2:
        return float("nan")
    return float(np.std(numeric, ddof=SAMPLE_SD_DDOF))


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if values.shape != weights.shape:
        raise CropSupportError("Weighted values and weights have different shapes")
    if not np.isfinite(values).all() or not np.isfinite(weights).all():
        raise CropSupportError("Weighted statistics contain non-finite inputs")
    if (weights <= 0).any():
        raise CropSupportError("Area weights must be strictly positive")
    return float(np.average(values, weights=weights))


def _weighted_population_sd(values: np.ndarray, weights: np.ndarray) -> float:
    """Return population-style weighted dispersion with denominator sum(w)."""

    mean = _weighted_mean(values, weights)
    variance = np.average(np.square(values - mean), weights=weights)
    return float(math.sqrt(max(float(variance), 0.0)))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Return the lower weighted median using cumulative fixed-frame area."""

    if values.shape != weights.shape or values.size == 0:
        raise CropSupportError("Invalid weighted-median inputs")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cutoff = 0.5 * float(sorted_weights.sum())
    index = int(np.searchsorted(np.cumsum(sorted_weights), cutoff, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


# ---------------------------------------------------------------------------
# Derived source tables
# ---------------------------------------------------------------------------


def build_figure5_domain_source(domain_year: pd.DataFrame) -> pd.DataFrame:
    """Return the accepted four-row domain source for Figure 5."""

    source = domain_year.sort_values("year", kind="mergesort").copy()
    source = source.rename(
        columns={
            "object_count": "fixed_frame_object_count",
            "fixed_2011_object_area_ha": "fixed_frame_area_ha",
            "cdl_corn_area_inside_fixed_objects_ha": "corn_area_inside_fixed_frame_ha",
            "supported_corn_area_inside_fixed_objects_ha": "supported_corn_area_inside_fixed_frame_ha",
            "corn_fraction_area_weighted": "crop_support_fraction",
            "supported_corn_fraction_area_weighted": "supported_crop_support_fraction",
            "object_count_corn_ge_70": "object_count_crop_support_ge_70",
            "object_area_corn_ge_70_ha": "object_area_crop_support_ge_70_ha",
        }
    )
    source.insert(2, "crop_support_percent", 100.0 * source["crop_support_fraction"])
    source.insert(3, "primary_threshold", PRIMARY_CROP_SUPPORT_THRESHOLD)
    source.insert(
        4,
        "threshold_status",
        np.where(
            source["crop_support_fraction"] >= PRIMARY_CROP_SUPPORT_THRESHOLD,
            "meets_or_exceeds_0.70",
            "below_0.70",
        ),
    )
    source["figure_changed"] = False
    source["interpretation_boundary"] = INTERPRETATION_BOUNDARY

    columns = [
        "county",
        "year",
        "crop_support_percent",
        "primary_threshold",
        "threshold_status",
        "crop_support_fraction",
        "supported_crop_support_fraction",
        "fixed_frame_object_count",
        "fixed_frame_area_ha",
        "corn_area_inside_fixed_frame_ha",
        "supported_corn_area_inside_fixed_frame_ha",
        "object_count_crop_support_ge_70",
        "object_area_crop_support_ge_70_ha",
        "interpretation_class",
        "figure_changed",
        "interpretation_boundary",
    ]
    return source[columns].reset_index(drop=True)


def build_figure5_county_source(
    county_year: pd.DataFrame,
    domain_year: pd.DataFrame,
) -> pd.DataFrame:
    """Return the sixteen county-year records used by Figure 5."""

    county = county_year[county_year["county"].isin(EXPECTED_COUNTIES)].copy()
    domain_lookup = domain_year.set_index("year")["corn_fraction_area_weighted"]
    county = county.sort_values(["year", "county"], kind="mergesort")
    county = county.rename(
        columns={
            "object_count": "fixed_frame_object_count",
            "fixed_2011_object_area_ha": "fixed_frame_area_ha",
            "cdl_corn_area_inside_fixed_objects_ha": "corn_area_inside_fixed_frame_ha",
            "supported_corn_area_inside_fixed_objects_ha": "supported_corn_area_inside_fixed_frame_ha",
            "corn_fraction_area_weighted": "crop_support_fraction",
            "supported_corn_fraction_area_weighted": "supported_crop_support_fraction",
            "object_count_corn_ge_70": "object_count_crop_support_ge_70",
            "object_area_corn_ge_70_ha": "object_area_crop_support_ge_70_ha",
        }
    )
    county.insert(2, "crop_support_percent", 100.0 * county["crop_support_fraction"])
    county.insert(3, "primary_threshold", PRIMARY_CROP_SUPPORT_THRESHOLD)
    county.insert(
        4,
        "threshold_status",
        np.where(
            county["crop_support_fraction"] >= PRIMARY_CROP_SUPPORT_THRESHOLD,
            "meets_or_exceeds_0.70",
            "below_0.70",
        ),
    )
    county.insert(
        7,
        "domain_crop_support_fraction",
        county["year"].map(domain_lookup).astype(float),
    )
    county["figure_changed"] = False
    county["interpretation_boundary"] = INTERPRETATION_BOUNDARY

    columns = [
        "county",
        "year",
        "crop_support_percent",
        "primary_threshold",
        "threshold_status",
        "crop_support_fraction",
        "supported_crop_support_fraction",
        "domain_crop_support_fraction",
        "fixed_frame_object_count",
        "fixed_frame_area_ha",
        "corn_area_inside_fixed_frame_ha",
        "supported_corn_area_inside_fixed_frame_ha",
        "object_count_crop_support_ge_70",
        "object_area_crop_support_ge_70_ha",
        "object_percent_corn_ge_70",
        "object_area_percent_corn_ge_70",
        "interpretation_class",
        "figure_changed",
        "interpretation_boundary",
    ]
    return county[columns].reset_index(drop=True)


def build_text_statistics(
    object_year: pd.DataFrame,
    county_year: pd.DataFrame,
) -> pd.DataFrame:
    """Return explicit county and object dispersion definitions for each year."""

    county = county_year[county_year["county"].isin(EXPECTED_COUNTIES)].copy()
    rows: list[dict[str, Any]] = []

    for year in EXPECTED_YEARS:
        county_values = county.loc[
            county["year"].eq(year), "corn_fraction_area_weighted"
        ].to_numpy(dtype=float)
        objects = object_year[object_year["year"].eq(year)].copy()
        object_values = objects["corn_fraction"].to_numpy(dtype=float)
        weights = objects["fixed_2011_object_area_ha"].to_numpy(dtype=float)
        fixed_frame_area = float(weights.sum())

        rows.append(
            {
                "year": year,
                "statistic": "county_crop_support_sample_dispersion",
                "unit": "fraction",
                "observational_unit": "county",
                "n": int(county_values.size),
                "ddof": SAMPLE_SD_DDOF,
                "mean": float(np.mean(county_values)),
                "standard_deviation": _sample_sd(county_values),
                "median": float(np.median(county_values)),
                "minimum": float(np.min(county_values)),
                "maximum": float(np.max(county_values)),
                "weighting": "none; each county contributes one county-year value",
                "statistic_definition": (
                    "Distribution of the four county-level area-weighted crop-support fractions within the year."
                ),
                "standard_deviation_definition": (
                    "Sample standard deviation across four counties; denominator n-1."
                ),
                "object_count": int(objects["object_id"].nunique()),
                "county_count": int(county_values.size),
                "fixed_frame_area_ha": fixed_frame_area,
                "primary_for_text": True,
                "interpretation_boundary": INTERPRETATION_BOUNDARY,
            }
        )
        rows.append(
            {
                "year": year,
                "statistic": "object_crop_support_area_weighted_population_dispersion",
                "unit": "fraction",
                "observational_unit": "fixed-frame object",
                "n": int(object_values.size),
                "ddof": WEIGHTED_POPULATION_DDOF,
                "mean": _weighted_mean(object_values, weights),
                "standard_deviation": _weighted_population_sd(object_values, weights),
                "median": _weighted_median(object_values, weights),
                "minimum": float(np.min(object_values)),
                "maximum": float(np.max(object_values)),
                "weighting": "fixed_2011_object_area_ha",
                "statistic_definition": (
                    "Area-representative distribution of object-level crop-support fractions within the complete fixed frame."
                ),
                "standard_deviation_definition": (
                    "Population-style weighted dispersion sqrt(sum(w*(x-mean_w)^2)/sum(w)); not a sample SD."
                ),
                "object_count": int(objects["object_id"].nunique()),
                "county_count": len(EXPECTED_COUNTIES),
                "fixed_frame_area_ha": fixed_frame_area,
                "primary_for_text": True,
                "interpretation_boundary": INTERPRETATION_BOUNDARY,
            }
        )
        rows.append(
            {
                "year": year,
                "statistic": "object_crop_support_unweighted_sample_dispersion",
                "unit": "fraction",
                "observational_unit": "fixed-frame object",
                "n": int(object_values.size),
                "ddof": SAMPLE_SD_DDOF,
                "mean": float(np.mean(object_values)),
                "standard_deviation": _sample_sd(object_values),
                "median": float(np.median(object_values)),
                "minimum": float(np.min(object_values)),
                "maximum": float(np.max(object_values)),
                "weighting": "none; each fixed-frame object contributes equally",
                "statistic_definition": (
                    "Secondary diagnostic distribution of object-level crop-support fractions without area weighting."
                ),
                "standard_deviation_definition": (
                    "Sample standard deviation across fixed-frame objects; denominator n-1."
                ),
                "object_count": int(objects["object_id"].nunique()),
                "county_count": len(EXPECTED_COUNTIES),
                "fixed_frame_area_ha": fixed_frame_area,
                "primary_for_text": False,
                "interpretation_boundary": INTERPRETATION_BOUNDARY,
            }
        )

    columns = [
        "year",
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
        "weighting",
        "statistic_definition",
        "standard_deviation_definition",
        "object_count",
        "county_count",
        "fixed_frame_area_ha",
        "primary_for_text",
        "interpretation_boundary",
    ]
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["year", "statistic"], kind="mergesort")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Derived-output validation
# ---------------------------------------------------------------------------


def validate_derived_outputs(
    domain_source: pd.DataFrame,
    county_source: pd.DataFrame,
    text_statistics: pd.DataFrame,
    *,
    object_year: pd.DataFrame,
    county_year: pd.DataFrame,
    domain_year: pd.DataFrame,
    runtime_validation_summary: Mapping[str, Any],
    contract_path: Path,
) -> dict[str, Any]:
    """Return deterministic crop-support, statistical, and contract checks."""

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
                "status": "PASS" if passed else "FAIL",
                "observed": observed,
                "expected": expected,
            }
        )

    contract = load_source_contract(contract_path)
    allowed_outputs = tuple(contract.allowed_outputs(MODULE_NAME))
    county_only = county_year[county_year["county"].isin(EXPECTED_COUNTIES)]

    observed_domain = {
        int(row.year): float(row.crop_support_fraction)
        for row in domain_source.itertuples(index=False)
    }
    expected_domain_rounded = dict(EXPECTED_DOMAIN_FRACTIONS)
    domain_acceptance = all(
        math.isclose(observed_domain[year], expected, rel_tol=0, abs_tol=5e-7)
        for year, expected in expected_domain_rounded.items()
    )

    object_weighted = {}
    for year in EXPECTED_YEARS:
        frame = object_year[object_year["year"].eq(year)]
        object_weighted[year] = _weighted_mean(
            frame["corn_fraction"].to_numpy(dtype=float),
            frame["fixed_2011_object_area_ha"].to_numpy(dtype=float),
        )
    object_domain_match = all(
        math.isclose(
            object_weighted[year], observed_domain[year], rel_tol=0, abs_tol=1e-12
        )
        for year in EXPECTED_YEARS
    )

    county_area_totals = county_only.groupby("year", sort=True)[
        "fixed_2011_object_area_ha"
    ].sum()
    domain_area = domain_year.set_index("year")["fixed_2011_object_area_ha"]
    county_area_match = np.allclose(
        county_area_totals.loc[list(EXPECTED_YEARS)].to_numpy(dtype=float),
        domain_area.loc[list(EXPECTED_YEARS)].to_numpy(dtype=float),
        rtol=0,
        atol=1e-8,
    )

    embedded_domain = (
        county_year[county_year["county"].eq("DOMAIN")]
        .sort_values("year", kind="mergesort")
        .reset_index(drop=True)
    )
    standalone_domain = domain_year.sort_values("year", kind="mergesort").reset_index(
        drop=True
    )

    add(
        "CROP-001",
        "Phase 3 crop-input validation",
        runtime_validation_summary.get("status"),
        "PASS",
        runtime_validation_summary.get("status") == "PASS",
    )
    add(
        "CROP-002",
        "Object-year row count",
        len(object_year),
        EXPECTED_OBJECT_YEARS,
        len(object_year) == EXPECTED_OBJECT_YEARS,
    )
    add(
        "CROP-003",
        "Fixed object count",
        int(object_year["object_id"].nunique()),
        EXPECTED_OBJECTS,
        object_year["object_id"].nunique() == EXPECTED_OBJECTS,
    )
    add(
        "CROP-004",
        "Each object occurs in four accepted years",
        sorted(object_year.groupby("object_id")["year"].nunique().unique().tolist()),
        [4],
        object_year.groupby("object_id")["year"].nunique().eq(4).all(),
    )
    add(
        "CROP-005",
        "Object-year key uniqueness",
        int(object_year.duplicated(["object_id", "year"]).sum()),
        0,
        not object_year.duplicated(["object_id", "year"]).any(),
    )
    add(
        "CROP-006",
        "Accepted year set",
        sorted(object_year["year"].unique().tolist()),
        list(EXPECTED_YEARS),
        set(object_year["year"]) == set(EXPECTED_YEARS),
    )
    add(
        "CROP-007",
        "Accepted county set",
        sorted(object_year["county"].unique().tolist()),
        list(EXPECTED_COUNTIES),
        set(object_year["county"]) == set(EXPECTED_COUNTIES),
    )
    add(
        "CROP-008",
        "Object crop fractions are bounded",
        [
            float(object_year["corn_fraction"].min()),
            float(object_year["corn_fraction"].max()),
        ],
        "within [0, 1]",
        object_year["corn_fraction"].between(0.0, 1.0).all(),
    )
    add(
        "CROP-009",
        "Figure 5 domain rows",
        len(domain_source),
        EXPECTED_DOMAIN_ROWS,
        len(domain_source) == EXPECTED_DOMAIN_ROWS,
    )
    add(
        "CROP-010",
        "Figure 5 county rows",
        len(county_source),
        EXPECTED_COUNTY_ROWS,
        len(county_source) == EXPECTED_COUNTY_ROWS,
    )
    add(
        "CROP-011",
        "Domain crop-support acceptance values",
        observed_domain,
        expected_domain_rounded,
        domain_acceptance,
    )
    add(
        "CROP-012",
        "Object-area weighted means reproduce domain source",
        object_weighted,
        observed_domain,
        object_domain_match,
    )
    add(
        "CROP-013",
        "County fixed-frame areas sum to domain area",
        county_area_totals.to_dict(),
        domain_area.to_dict(),
        bool(county_area_match),
    )
    add(
        "CROP-014",
        "Embedded and standalone domain summaries agree",
        bool(embedded_domain.equals(standalone_domain)),
        True,
        bool(embedded_domain.equals(standalone_domain)),
    )
    comparison = county_source[county_source["year"].ne(2011)]
    add(
        "CROP-015",
        "Twelve comparison county-years remain below 0.70",
        {
            "n": len(comparison),
            "maximum": float(comparison["crop_support_fraction"].max()),
        },
        {"n": 12, "maximum": "<0.70"},
        len(comparison) == 12
        and bool(
            (comparison["crop_support_fraction"] < PRIMARY_CROP_SUPPORT_THRESHOLD).all()
        ),
    )
    add(
        "CROP-016",
        "2011 domain and county support equals one",
        {
            "domain": float(
                domain_source.loc[domain_source["year"].eq(2011), "crop_support_fraction"].iloc[0]
            ),
            "county_min": float(
                county_source.loc[county_source["year"].eq(2011), "crop_support_fraction"].min()
            ),
        },
        1.0,
        bool(
            domain_source.loc[
                domain_source["year"].eq(2011), "crop_support_fraction"
            ].eq(1.0).all()
            and county_source.loc[
                county_source["year"].eq(2011), "crop_support_fraction"
            ].eq(1.0).all()
        ),
    )
    add(
        "CROP-017",
        "Text-statistics row count",
        len(text_statistics),
        12,
        len(text_statistics) == 12,
    )
    add(
        "CROP-018",
        "Text-statistics key uniqueness",
        int(text_statistics.duplicated(["year", "statistic"]).sum()),
        0,
        not text_statistics.duplicated(["year", "statistic"]).any(),
    )
    county_stats = text_statistics[
        text_statistics["statistic"].eq("county_crop_support_sample_dispersion")
    ]
    add(
        "CROP-019",
        "County SD uses four counties and ddof=1",
        {
            "n": sorted(county_stats["n"].unique().tolist()),
            "ddof": sorted(county_stats["ddof"].unique().tolist()),
        },
        {"n": [4], "ddof": [1]},
        county_stats["n"].eq(4).all() and county_stats["ddof"].eq(1).all(),
    )
    weighted_stats = text_statistics[
        text_statistics["statistic"].eq(
            "object_crop_support_area_weighted_population_dispersion"
        )
    ]
    add(
        "CROP-020",
        "Weighted object SD is explicitly population-style",
        {
            "n": sorted(weighted_stats["n"].unique().tolist()),
            "ddof": sorted(weighted_stats["ddof"].unique().tolist()),
        },
        {"n": [EXPECTED_OBJECTS], "ddof": [0]},
        weighted_stats["n"].eq(EXPECTED_OBJECTS).all()
        and weighted_stats["ddof"].eq(0).all()
        and weighted_stats["standard_deviation_definition"]
        .str.contains("Population-style", case=False)
        .all(),
    )
    unweighted_stats = text_statistics[
        text_statistics["statistic"].eq(
            "object_crop_support_unweighted_sample_dispersion"
        )
    ]
    add(
        "CROP-021",
        "Unweighted object SD uses ddof=1",
        {
            "n": sorted(unweighted_stats["n"].unique().tolist()),
            "ddof": sorted(unweighted_stats["ddof"].unique().tolist()),
        },
        {"n": [EXPECTED_OBJECTS], "ddof": [1]},
        unweighted_stats["n"].eq(EXPECTED_OBJECTS).all()
        and unweighted_stats["ddof"].eq(1).all(),
    )
    add(
        "CROP-022",
        "All descriptive SDs are finite and non-negative",
        bool(np.isfinite(text_statistics["standard_deviation"]).all()),
        True,
        bool(
            np.isfinite(text_statistics["standard_deviation"]).all()
            and text_statistics["standard_deviation"].ge(0.0).all()
        ),
    )
    add(
        "CROP-023",
        "Required statistical-definition fields are complete",
        int(
            text_statistics[
                [
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
            ].isna().sum().sum()
        ),
        0,
        not text_statistics[
            [
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
        ].isna().any().any(),
    )
    add(
        "CROP-024",
        "Output names are authorized by source contract",
        list(allowed_outputs),
        list(EXPECTED_OUTPUT_NAMES),
        allowed_outputs == EXPECTED_OUTPUT_NAMES,
    )
    add(
        "CROP-025",
        "Figure design remains unchanged",
        bool(
            domain_source["figure_changed"].eq(False).all()
            and county_source["figure_changed"].eq(False).all()
        ),
        True,
        bool(
            domain_source["figure_changed"].eq(False).all()
            and county_source["figure_changed"].eq(False).all()
        ),
    )
    add(
        "CROP-026",
        "Scientific boundary is attached to all public records",
        bool(
            domain_source["interpretation_boundary"].eq(INTERPRETATION_BOUNDARY).all()
            and county_source["interpretation_boundary"].eq(INTERPRETATION_BOUNDARY).all()
            and text_statistics["interpretation_boundary"].eq(
                INTERPRETATION_BOUNDARY
            ).all()
        ),
        True,
        bool(
            domain_source["interpretation_boundary"].eq(INTERPRETATION_BOUNDARY).all()
            and county_source["interpretation_boundary"].eq(INTERPRETATION_BOUNDARY).all()
            and text_statistics["interpretation_boundary"].eq(
                INTERPRETATION_BOUNDARY
            ).all()
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
        "primary_crop_support_threshold": PRIMARY_CROP_SUPPORT_THRESHOLD,
        "standard_deviation": {
            "county": {
                "type": "sample standard deviation",
                "observational_unit": "county",
                "n_per_year": 4,
                "ddof": SAMPLE_SD_DDOF,
            },
            "object_area_weighted": {
                "type": "population-style weighted dispersion",
                "observational_unit": "fixed-frame object",
                "n_per_year": EXPECTED_OBJECTS,
                "weights": "fixed_2011_object_area_ha",
                "ddof": WEIGHTED_POPULATION_DDOF,
                "formula": "sqrt(sum(w*(x-mean_w)^2)/sum(w))",
            },
            "object_unweighted": {
                "type": "sample standard deviation",
                "observational_unit": "fixed-frame object",
                "n_per_year": EXPECTED_OBJECTS,
                "ddof": SAMPLE_SD_DDOF,
                "role": "secondary diagnostic",
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


def _prepare_outputs(output_dir: Path, *, overwrite: bool) -> CropSupportOutputs:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    outputs = CropSupportOutputs(
        figure5_domain_crop_support_source=out
        / "figure5_domain_crop_support_source.csv",
        figure5_county_crop_support_source=out
        / "figure5_county_crop_support_source.csv",
        crop_support_text_statistics=out / "crop_support_text_statistics.csv",
        crop_support_validation=out / "crop_support_validation.json",
        crop_support_execution_record=out / "crop_support_execution_record.json",
    )
    existing = [path for path in outputs.paths() if path.exists()]
    if existing and not overwrite:
        raise CropSupportError(
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
    outputs: CropSupportOutputs,
) -> dict[str, Any]:
    contract = load_source_contract(contract_path)
    input_files = {
        item.relative_path: {"sha256": item.sha256}
        for item in contract.files_for_module(MODULE_NAME)
    }
    output_records = {
        path.name: _file_record(path)
        for path in outputs.paths()
        if path != outputs.crop_support_execution_record
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
            "qstable_awm.crop_support",
            "--phase3-root",
            "PHASE3_CURATED_INPUT_LAYER_FINAL_002",
            "--output-dir",
            "results/crop_support",
            "--contract",
            "config/phase4_source_contract.yml",
        ],
        "statistical_decisions": validation["standard_deviation"],
        "figure_change": False,
        "scientific_boundary": INTERPRETATION_BOUNDARY,
        "determinism_note": (
            "Numerical outputs derive only from frozen Phase 3 tables; rows and columns are explicitly sorted."
        ),
    }


def run_crop_support(
    phase3_root: Path,
    output_dir: Path,
    *,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
    overwrite: bool = False,
) -> CropSupportOutputs:
    """Validate inputs, derive Figure 5/text sources, and write five outputs."""

    outputs = _prepare_outputs(output_dir, overwrite=overwrite)
    object_year, county_year, domain_year, runtime_summary = load_crop_support_inputs(
        phase3_root, contract_path=contract_path
    )
    _assert_input_structure(object_year, county_year, domain_year)

    domain_source = build_figure5_domain_source(domain_year)
    county_source = build_figure5_county_source(county_year, domain_year)
    text_statistics = build_text_statistics(object_year, county_year)
    validation = validate_derived_outputs(
        domain_source,
        county_source,
        text_statistics,
        object_year=object_year,
        county_year=county_year,
        domain_year=domain_year,
        runtime_validation_summary=runtime_summary,
        contract_path=contract_path,
    )
    if validation["status"] != "PASS":
        failures = [
            check for check in validation["check_results"] if check["status"] == "FAIL"
        ]
        raise CropSupportError(f"Derived-output validation failed: {failures}")

    _atomic_write_csv(domain_source, outputs.figure5_domain_crop_support_source)
    _atomic_write_csv(county_source, outputs.figure5_county_crop_support_source)
    _atomic_write_csv(text_statistics, outputs.crop_support_text_statistics)
    _atomic_write_json(validation, outputs.crop_support_validation)

    # Round-trip verification before the execution record is committed.
    try:
        pd.testing.assert_frame_equal(
            pd.read_csv(outputs.figure5_domain_crop_support_source),
            domain_source,
            check_dtype=False,
            check_exact=False,
            rtol=CSV_ROUNDTRIP_RTOL,
            atol=CSV_ROUNDTRIP_ATOL,
        )
        pd.testing.assert_frame_equal(
            pd.read_csv(outputs.figure5_county_crop_support_source),
            county_source,
            check_dtype=False,
            check_exact=False,
            rtol=CSV_ROUNDTRIP_RTOL,
            atol=CSV_ROUNDTRIP_ATOL,
        )
        pd.testing.assert_frame_equal(
            pd.read_csv(outputs.crop_support_text_statistics),
            text_statistics,
            check_dtype=False,
            check_exact=False,
            rtol=CSV_ROUNDTRIP_RTOL,
            atol=CSV_ROUNDTRIP_ATOL,
        )
    except AssertionError as exc:
        raise CropSupportError(f"CSV round-trip verification failed: {exc}") from exc
    with outputs.crop_support_validation.open("r", encoding="utf-8") as stream:
        serialized_validation = json.load(stream)
    canonical_validation = json.loads(
        json.dumps(validation, sort_keys=True, ensure_ascii=False, allow_nan=False)
    )
    if serialized_validation != canonical_validation:
        raise CropSupportError("Validation JSON round-trip verification failed")
    for path in (
        outputs.figure5_domain_crop_support_source,
        outputs.figure5_county_crop_support_source,
        outputs.crop_support_text_statistics,
        outputs.crop_support_validation,
    ):
        if not path.is_file() or path.stat().st_size <= 0:
            raise CropSupportError(f"Output was not written correctly: {path}")

    execution_record = _build_execution_record(
        contract_path=Path(contract_path).expanduser().resolve(),
        runtime_validation_summary=runtime_summary,
        validation=validation,
        outputs=outputs,
    )
    _atomic_write_json(execution_record, outputs.crop_support_execution_record)
    return outputs


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def _text_summary(statistics_path: Path) -> str:
    statistics = pd.read_csv(statistics_path)
    county = statistics[
        statistics["statistic"].eq("county_crop_support_sample_dispersion")
    ]
    weighted = statistics[
        statistics["statistic"].eq(
            "object_crop_support_area_weighted_population_dispersion"
        )
    ].set_index("year")
    lines = [
        "Crop-support descriptive statistics:",
        "County SD: sample SD across four counties (ddof=1).",
        "Object weighted SD: population-style area-weighted dispersion (ddof=0).",
    ]
    for row in county.itertuples(index=False):
        weighted_row = weighted.loc[int(row.year)]
        lines.append(
            f"- {int(row.year)}: county mean {row.mean:.6f} ± "
            f"{row.standard_deviation:.6f} (n={int(row.n)}); "
            f"area-weighted object mean {weighted_row['mean']:.6f} ± "
            f"{weighted_row['standard_deviation']:.6f} (n={int(weighted_row['n'])})."
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Derive Figure 5 crop-support and text-statistics tables."
    )
    parser.add_argument("--phase3-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing crop-support outputs in the selected directory.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        outputs = run_crop_support(
            args.phase3_root,
            args.output_dir,
            contract_path=args.contract,
            overwrite=args.overwrite,
        )
    except (CropSupportError, FileNotFoundError, ValueError) as exc:
        print(f"CROP_SUPPORT_ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(outputs.as_dict(), indent=2, sort_keys=True))
    print(_text_summary(outputs.crop_support_text_statistics))
    return 0


__all__ = [
    "CropSupportError",
    "CropSupportOutputs",
    "build_figure5_county_source",
    "build_figure5_domain_source",
    "build_text_statistics",
    "load_crop_support_inputs",
    "run_crop_support",
    "validate_derived_outputs",
]


if __name__ == "__main__":
    raise SystemExit(main())
