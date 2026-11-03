"""Strict known-cold reference sensitivity summaries for manuscript Figure 7.

This module reads only the six Phase 3 inputs authorized for the
``known_cold_reference`` domain. It does not open the external full-resolution
rasters and it does not rerun the METRIC solver. The accepted V51 strict raster
QA tables and finalized Supplementary Methods S1 are the runtime authority.

The Figure 7 response statistic is the median, across retained raster-radius
records, of each raster's median absolute daily ETa-equivalent response. Four
nested support domains are retained: valid output, active canopy, large ETrF,
and physical-coherence mechanism evidence.

Standard-deviation conventions
------------------------------
* Primary text dispersion: sample SD (``ddof=1``) across the four accepted
  county-radius group medians for each support domain. The observational unit is
  a county-radius group; the four records are Moore and Ochiltree at 1000 and
  3000 m. These are descriptive grouped summaries, not independent validation
  replicates.
* Secondary diagnostic dispersion: sample SD (``ddof=1``) across retained
  raster-radius record medians. The observational unit is a strict
  raster-radius record. The mechanism domain has 38 non-null records because
  one accepted raster contains zero mechanism pixels.
* Numerical-QA dispersion: sample SD across the four county-radius groups for
  thermal separation, source-pixel support, clamp fractions, nonconvergence,
  and aerodynamic-resistance cap hits.

Scientific boundary
-------------------
Known-cold outputs are diagnostic reference-sensitivity evidence only. They are
not corrected ET, observed ET, validation ET, a replacement production
product, pumping evidence, groundwater-withdrawal evidence, or water-source
attribution.
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
        RuntimeInputError,
        assert_allowed_output_name,
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
        assert_allowed_output_name,
        load_source_contract,
        sha256_file,
        validate_module_inputs,
    )


MODULE_NAME = "known_cold_reference"
MODULE_RELATIVE_PATH = "src/qstable_awm/known_cold_reference.py"

RASTER_SUMMARY_RELATIVE_PATH = (
    "data/derived/known_cold_reference/strict_mask_by_raster.csv"
)
COUNTY_COUNT_RELATIVE_PATH = (
    "data/derived/known_cold_reference/strict_county_raster_counts.csv"
)
COUNTY_RADIUS_QA_RELATIVE_PATH = (
    "data/derived/known_cold_reference/county_radius_response_summary.csv"
)
FUNNEL_RELATIVE_PATH = (
    "data/derived/known_cold_reference/strict_processing_funnel.csv"
)
SOLVER_PARAMETERS_RELATIVE_PATH = (
    "data/derived/supplementary/table_s4_known_cold_solver_parameters.csv"
)
METHODS_RELATIVE_PATH = (
    "data/derived/supplementary/Supplementary_Methods_S1.md"
)

EXPECTED_OUTPUT_NAMES = (
    "figure7_processing_funnel_source.csv",
    "figure7_county_radius_source.csv",
    "figure7_mask_response_source.csv",
    "known_cold_reference_text_statistics.csv",
    "known_cold_reference_validation.json",
    "known_cold_reference_execution_record.json",
)

EXPECTED_RASTER_COUNT = 39
EXPECTED_COUNTIES = ("MOORE", "OCHILTREE")
EXPECTED_RADII_M = (1000, 3000)
EXPECTED_COUNTY_COUNTS: Mapping[str, int] = {"MOORE": 30, "OCHILTREE": 9}
EXPECTED_GROUP_COUNTS: Mapping[tuple[str, int], int] = {
    ("MOORE", 1000): 12,
    ("MOORE", 3000): 18,
    ("OCHILTREE", 1000): 4,
    ("OCHILTREE", 3000): 5,
}
EXPECTED_FUNNEL_STAGES = (
    "cloned_geotiffs",
    "readable_geotiffs",
    "valid_output_positive",
    "strict_hierarchy_pass",
)
SAMPLE_SD_DDOF = 1
MATERIAL_RESPONSE_MM_D = 1.0
CSV_ROUNDTRIP_RTOL = 1e-12
CSV_ROUNDTRIP_ATOL = 1e-15

INTERPRETATION_BOUNDARY = (
    "Diagnostic known-cold reference sensitivity only; accepted METRIC remains "
    "the production record; no corrected ET; no observed or validation ET; no "
    "pumping or groundwater-withdrawal evidence; no water-source attribution."
)

EXPECTED_SOLVER_PARAMETERS: Mapping[str, str] = {
    "Source locator": "35.999399 N, 101.979024 W",
    "Source radii": "1000 and 3000 m",
    "Temporal matching": "Same date only",
    "Minimum source pixels": "30",
    "Primary source statistic": "Median",
    "Thermal-separation gate": "6.5 K",
    "Maximum stability iterations": "10",
    "H convergence tolerance": "2 W m-2",
    "r_ah convergence tolerance": "0.02 fractional change",
    "r_ah cap": "8000 s m-1",
    "ETrF clamp": "-0.1 to 1.2",
    "Material |Delta ETa|": "1.0 mm d-1",
}

METHODS_REQUIRED_FRAGMENTS = (
    "The main analysis did not use temporal fallback.",
    "NDVI >= 0.65",
    "LAI >= 2.50",
    "LE/Rn >= 0.85",
    "ETrF >= 0.85",
    "H <= 50 W m-2",
    "at least 30 pixels",
    "6.5 K thermal-separation gate",
    "The factor 1.05 was applied only",
    "without the 1.05 cold-reference multiplier",
    "LAI was bounded to 0-7",
    "0.123 times canopy height",
    "one tenth of momentum roughness",
    "at most 10 iterations",
    "2 W m-2",
    "0.02",
    "Rn > 20 W m-2",
    "-300 < LE < 1000 W m-2",
    "-500 < H < 800 W m-2",
    "-1 < raw ETrF < 2",
    "valid output >= active canopy >= large ETrF >= mechanism evidence",
    "final strict set contained 39 readable",
    "30 from Moore and 9 from Ochiltree",
    "not corrected ET",
    "not corrected ET, validation ET, observed ET, pumping estimates",
)

DOMAIN_SPECS: Mapping[str, Mapping[str, str]] = {
    "valid_output": {
        "label": "All valid output",
        "median_column": "abs_delta_ETa_KC_mm_d_valid_median",
        "n_column": "abs_delta_ETa_KC_mm_d_valid_n",
        "pixel_column": "valid_pixels",
        "area_column": "valid_output_area_ha",
    },
    "active_canopy": {
        "label": "Active canopy",
        "median_column": "abs_delta_ETa_KC_mm_d_active_strict_median",
        "n_column": "abs_delta_ETa_KC_mm_d_active_strict_n",
        "pixel_column": "active_strict_pixels",
        "area_column": "active_strict_area_ha",
    },
    "large_ETrF": {
        "label": "Large ETrF",
        "median_column": "abs_delta_ETa_KC_mm_d_high_strict_median",
        "n_column": "abs_delta_ETa_KC_mm_d_high_strict_n",
        "pixel_column": "high_strict_pixels",
        "area_column": "high_strict_area_ha",
    },
    "mechanism_evidence": {
        "label": "Mechanism evidence",
        "median_column": "abs_delta_ETa_KC_mm_d_mechanism_strict_median",
        "n_column": "abs_delta_ETa_KC_mm_d_mechanism_strict_n",
        "pixel_column": "mechanism_strict_pixels",
        "area_column": "mechanism_strict_area_ha",
    },
}

COUNTY_ORDER = {county: index for index, county in enumerate(EXPECTED_COUNTIES)}
RADIUS_ORDER = {radius: index for index, radius in enumerate(EXPECTED_RADII_M)}
DOMAIN_ORDER = {domain: index for index, domain in enumerate(DOMAIN_SPECS)}
STAGE_LABELS = {
    "cloned_geotiffs": "Candidate rasters",
    "readable_geotiffs": "Readable rasters",
    "valid_output_positive": "Valid-output rasters",
    "strict_hierarchy_pass": "Retained strict rasters",
}


class KnownColdReferenceError(RuntimeError):
    """Raised when known-cold inputs or derived outputs violate authority."""


@dataclass(frozen=True, slots=True)
class KnownColdReferenceOutputs:
    """Paths written by :func:`run_known_cold_reference`."""

    figure7_processing_funnel_source: Path
    figure7_county_radius_source: Path
    figure7_mask_response_source: Path
    known_cold_reference_text_statistics: Path
    known_cold_reference_validation: Path
    known_cold_reference_execution_record: Path

    def as_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}

    def paths(self) -> tuple[Path, ...]:
        return tuple(Path(value) for value in asdict(self).values())


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


def load_known_cold_inputs(
    phase3_root: Path,
    *,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    str,
    dict[str, Any],
]:
    """Validate and load the exact six authorized known-cold inputs."""

    root = Path(phase3_root).expanduser().resolve()
    relative_paths = (
        RASTER_SUMMARY_RELATIVE_PATH,
        COUNTY_COUNT_RELATIVE_PATH,
        COUNTY_RADIUS_QA_RELATIVE_PATH,
        FUNNEL_RELATIVE_PATH,
        SOLVER_PARAMETERS_RELATIVE_PATH,
        METHODS_RELATIVE_PATH,
    )
    supplied = tuple(root / relative_path for relative_path in relative_paths)
    report = validate_module_inputs(
        root,
        MODULE_NAME,
        contract_path=contract_path,
        supplied_paths=supplied,
        raise_on_error=True,
    )
    raster_summary = pd.read_csv(supplied[0])
    county_counts = pd.read_csv(supplied[1])
    county_radius_qa = pd.read_csv(supplied[2])
    funnel = pd.read_csv(supplied[3])
    solver_parameters = pd.read_csv(supplied[4], dtype=str, keep_default_na=False)
    methods_text = supplied[5].read_text(encoding="utf-8")
    return (
        raster_summary,
        county_counts,
        county_radius_qa,
        funnel,
        solver_parameters,
        methods_text,
        _portable_runtime_summary(report.summary()),
    )


def _close(left: float, right: float, *, atol: float = 1e-10) -> bool:
    return bool(math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=atol))


def _assert_input_structure(
    raster_summary: pd.DataFrame,
    county_counts: pd.DataFrame,
    county_radius_qa: pd.DataFrame,
    funnel: pd.DataFrame,
    solver_parameters: pd.DataFrame,
    methods_text: str,
) -> None:
    """Apply known-cold scientific checks beyond the generic schemas."""

    if len(raster_summary) != EXPECTED_RASTER_COUNT:
        raise KnownColdReferenceError(
            f"Expected {EXPECTED_RASTER_COUNT} strict rasters; found {len(raster_summary)}"
        )
    if raster_summary["raster_name"].duplicated().any():
        raise KnownColdReferenceError("Strict raster names are not unique")
    if raster_summary["scene_date"].astype(str).str[:4].ne("2011").any():
        raise KnownColdReferenceError("Known-cold strict records must remain restricted to 2011")
    if set(raster_summary["county"].astype(str)) != set(EXPECTED_COUNTIES):
        raise KnownColdReferenceError("Unexpected county set in strict raster table")
    if set(raster_summary["radius_m"].astype(int)) != set(EXPECTED_RADII_M):
        raise KnownColdReferenceError("Unexpected source-radius set")
    if not raster_summary["raster_path"].astype(str).str.startswith("source://").all():
        raise KnownColdReferenceError(
            "Phase 3 strict raster pointers must remain non-runtime source:// references"
        )
    if not raster_summary["read_status"].eq("PASS_READABLE").all():
        raise KnownColdReferenceError("Every retained strict raster must be readable")
    if not raster_summary["crs_projected"].astype(int).eq(1).all():
        raise KnownColdReferenceError("Every retained strict raster must use a projected CRS")
    if not raster_summary["hierarchy_pass"].astype(int).eq(1).all():
        raise KnownColdReferenceError("Every retained raster must pass the strict hierarchy")

    nested = (
        raster_summary["valid_pixels"].ge(raster_summary["active_strict_pixels"])
        & raster_summary["active_strict_pixels"].ge(
            raster_summary["high_strict_pixels"]
        )
        & raster_summary["high_strict_pixels"].ge(
            raster_summary["mechanism_strict_pixels"]
        )
    )
    if not nested.all():
        raise KnownColdReferenceError(
            "Strict mask hierarchy failed: valid >= active >= large ETrF >= mechanism"
        )

    for pixels, area in (
        ("active_strict_pixels", "active_strict_area_ha"),
        ("high_strict_pixels", "high_strict_area_ha"),
        ("mechanism_strict_pixels", "mechanism_strict_area_ha"),
    ):
        expected_area = (
            raster_summary[pixels].astype(float)
            * raster_summary["pixel_area_m2"].astype(float)
            / 10_000.0
        )
        if not np.allclose(
            raster_summary[area].astype(float),
            expected_area,
            rtol=0,
            atol=1e-8,
            equal_nan=True,
        ):
            raise KnownColdReferenceError(f"Pixel-to-area identity failed for {area}")

    observed_counties = raster_summary["county"].value_counts().to_dict()
    if observed_counties != dict(EXPECTED_COUNTY_COUNTS):
        raise KnownColdReferenceError(
            f"Unexpected county raster counts: {observed_counties}"
        )
    observed_groups = raster_summary.groupby(["county", "radius_m"]).size().to_dict()
    if observed_groups != dict(EXPECTED_GROUP_COUNTS):
        raise KnownColdReferenceError(
            f"Unexpected county-radius raster counts: {observed_groups}"
        )

    if county_counts.duplicated(["county"]).any():
        raise KnownColdReferenceError("County-count keys are not unique")
    count_map = county_counts.set_index("county")["raster_count"].astype(int).to_dict()
    if count_map != dict(EXPECTED_COUNTY_COUNTS):
        raise KnownColdReferenceError(
            f"Frozen county count table disagrees with strict rasters: {count_map}"
        )
    if not county_counts["hierarchy_pass_all"].astype(int).eq(1).all():
        raise KnownColdReferenceError("County summary must preserve hierarchy_pass_all=1")

    if county_radius_qa.duplicated(["county", "radius_m"]).any():
        raise KnownColdReferenceError("County-radius QA keys are not unique")
    qa_keys = set(
        zip(
            county_radius_qa["county"].astype(str),
            county_radius_qa["radius_m"].astype(int),
        )
    )
    if qa_keys != set(EXPECTED_GROUP_COUNTS):
        raise KnownColdReferenceError("County-radius QA groups changed")

    for key, expected_count in EXPECTED_GROUP_COUNTS.items():
        county, radius = key
        strict = raster_summary[
            raster_summary["county"].eq(county)
            & raster_summary["radius_m"].eq(radius)
        ]
        qa = county_radius_qa[
            county_radius_qa["county"].eq(county)
            & county_radius_qa["radius_m"].eq(radius)
        ].iloc[0]
        if int(qa["raster_count"]) != expected_count:
            raise KnownColdReferenceError(f"Raster-count mismatch for {key}")
        if int(qa["valid_output_pixels"]) != int(strict["valid_pixels"].sum()):
            raise KnownColdReferenceError(f"Valid-pixel mismatch for {key}")
        if int(qa["active_valid_pixels"]) != int(
            strict["active_strict_pixels"].sum()
        ):
            raise KnownColdReferenceError(f"Active-pixel mismatch for {key}")
        comparisons = {
            "abs_delta_ETa_KC_mm_d_valid_median": (
                "abs_delta_ETa_KC_mm_d_valid_median"
            ),
            "abs_delta_ETa_KC_mm_d_active_median": (
                "abs_delta_ETa_KC_mm_d_active_strict_median"
            ),
            "delta_ETa_KC_mm_d_valid_median": "delta_ETa_KC_mm_d_valid_median",
            "delta_ETf_KC_valid_median": "delta_ETf_KC_valid_median",
            "T_hot_minus_T_KC_r_valid_median": (
                "T_hot_minus_T_KC_r_valid_median"
            ),
            "v51_source_pixel_count_valid_median": (
                "v51_source_pixel_count_valid_median"
            ),
            "nonconverged_fraction_valid": "nonconverged_strict_frac_valid",
            "high_ETf_clamp_fraction_valid": "high_ETf_clamp_strict_frac_valid",
            "low_ETf_clamp_fraction_valid": "low_ETf_clamp_strict_frac_valid",
            "rah_cap_hit_fraction_valid": "rah_cap_hit_strict_frac_valid",
        }
        for qa_column, raster_column in comparisons.items():
            observed = float(strict[raster_column].median())
            expected = float(qa[qa_column])
            if not _close(observed, expected, atol=2e-10):
                raise KnownColdReferenceError(
                    f"County-radius median mismatch for {key}: {qa_column}"
                )

    if tuple(funnel["stage"].astype(str)) != EXPECTED_FUNNEL_STAGES:
        raise KnownColdReferenceError("Strict processing-funnel stage order changed")
    if not funnel["count"].astype(int).eq(EXPECTED_RASTER_COUNT).all():
        raise KnownColdReferenceError("Every strict processing-funnel stage must equal 39")

    parameter_map = solver_parameters.set_index("parameter")["value"].to_dict()
    if parameter_map != dict(EXPECTED_SOLVER_PARAMETERS):
        raise KnownColdReferenceError(
            "Known-cold solver parameter table changed from Supplementary Methods S1"
        )
    missing_fragments = [
        fragment for fragment in METHODS_REQUIRED_FRAGMENTS if fragment not in methods_text
    ]
    if missing_fragments:
        raise KnownColdReferenceError(
            "Supplementary Methods S1 is missing required authority statements: "
            + repr(missing_fragments)
        )

    if not raster_summary["nonconverged_strict_frac_valid"].eq(0.0).all():
        raise KnownColdReferenceError("Expected zero nonconvergence in all strict rasters")
    if not raster_summary["rah_cap_hit_strict_frac_valid"].eq(0.0).all():
        raise KnownColdReferenceError("Expected zero aerodynamic-resistance cap hits")


def _sample_sd(values: Sequence[float] | pd.Series | np.ndarray) -> float:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if len(numeric) < 2:
        return float("nan")
    return float(numeric.std(ddof=SAMPLE_SD_DDOF))


def build_processing_funnel_source(funnel: pd.DataFrame) -> pd.DataFrame:
    """Return the unchanged Figure 7A processing-funnel source."""

    result = funnel.copy()
    result["stage_order"] = result["stage"].map(
        {stage: index for index, stage in enumerate(EXPECTED_FUNNEL_STAGES)}
    ).astype(int)
    result["stage_label"] = result["stage"].map(STAGE_LABELS)
    base = float(result.loc[result["stage"].eq("cloned_geotiffs"), "count"].iloc[0])
    result["retention_fraction_of_candidates"] = result["count"].astype(float) / base
    result["figure_panel"] = "A"
    result["figure_changed"] = False
    result["interpretation_boundary"] = INTERPRETATION_BOUNDARY
    return result[
        [
            "stage_order",
            "stage",
            "stage_label",
            "count",
            "retention_fraction_of_candidates",
            "figure_panel",
            "figure_changed",
            "interpretation_boundary",
        ]
    ].sort_values("stage_order").reset_index(drop=True)


def _group_summary(strict: pd.DataFrame, county: str, radius: int) -> dict[str, Any]:
    valid_area = float(
        (
            strict["valid_pixels"].astype(float)
            * strict["pixel_area_m2"].astype(float)
            / 10_000.0
        ).sum()
    )
    return {
        "county": county,
        "radius_m": int(radius),
        "county_order": COUNTY_ORDER[county],
        "radius_order": RADIUS_ORDER[int(radius)],
        "raster_count": int(len(strict)),
        "unique_scene_dates": int(strict["scene_date"].nunique()),
        "valid_pixels": int(strict["valid_pixels"].sum()),
        "active_strict_pixels": int(strict["active_strict_pixels"].sum()),
        "high_strict_pixels": int(strict["high_strict_pixels"].sum()),
        "mechanism_strict_pixels": int(strict["mechanism_strict_pixels"].sum()),
        "valid_output_area_ha": valid_area,
        "active_strict_area_ha": float(strict["active_strict_area_ha"].sum()),
        "high_strict_area_ha": float(strict["high_strict_area_ha"].sum()),
        "mechanism_strict_area_ha": float(
            strict["mechanism_strict_area_ha"].sum()
        ),
        "hierarchy_pass_all": bool(strict["hierarchy_pass"].astype(bool).all()),
        "thermal_gap_median_K": float(
            strict["T_hot_minus_T_KC_r_valid_median"].median()
        ),
        "source_pixel_count_median": float(
            strict["v51_source_pixel_count_valid_median"].median()
        ),
        "source_pixel_count_minimum": float(
            strict["v51_source_pixel_count_valid_min"].min()
        ),
        "nonconverged_fraction_median": float(
            strict["nonconverged_strict_frac_valid"].median()
        ),
        "high_ETrF_clamp_fraction_median": float(
            strict["high_ETf_clamp_strict_frac_valid"].median()
        ),
        "low_ETrF_clamp_fraction_median": float(
            strict["low_ETf_clamp_strict_frac_valid"].median()
        ),
        "rah_cap_hit_fraction_median": float(
            strict["rah_cap_hit_strict_frac_valid"].median()
        ),
        "minimum_source_pixels_required": 30,
        "thermal_separation_gate_K": 6.5,
        "same_date_source_only": True,
        "figure_panels": "B,D",
        "figure_changed": False,
        "interpretation_boundary": INTERPRETATION_BOUNDARY,
    }


def build_county_radius_source(raster_summary: pd.DataFrame) -> pd.DataFrame:
    """Return Figure 7B/D county-radius composition and numerical QA."""

    rows: list[dict[str, Any]] = []
    for county in EXPECTED_COUNTIES:
        for radius in EXPECTED_RADII_M:
            strict = raster_summary[
                raster_summary["county"].eq(county)
                & raster_summary["radius_m"].eq(radius)
            ].copy()
            rows.append(_group_summary(strict, county, radius))
    result = pd.DataFrame(rows)
    return result.sort_values(["county_order", "radius_order"]).reset_index(drop=True)


def build_mask_response_source(raster_summary: pd.DataFrame) -> pd.DataFrame:
    """Return Figure 7C long-form nested-domain response summaries."""

    rows: list[dict[str, Any]] = []
    for county in EXPECTED_COUNTIES:
        for radius in EXPECTED_RADII_M:
            strict = raster_summary[
                raster_summary["county"].eq(county)
                & raster_summary["radius_m"].eq(radius)
            ].copy()
            for domain, spec in DOMAIN_SPECS.items():
                values = pd.to_numeric(strict[spec["median_column"]], errors="coerce").dropna()
                rows.append(
                    {
                        "county": county,
                        "radius_m": int(radius),
                        "county_order": COUNTY_ORDER[county],
                        "radius_order": RADIUS_ORDER[int(radius)],
                        "support_domain": domain,
                        "support_domain_label": spec["label"],
                        "support_domain_order": DOMAIN_ORDER[domain],
                        "raster_record_n": int(len(values)),
                        "group_median_abs_delta_ETa_mm_d": float(values.median()),
                        "mean_across_raster_medians_mm_d": float(values.mean()),
                        "sample_sd_across_raster_medians_mm_d": _sample_sd(values),
                        "minimum_raster_median_mm_d": float(values.min()),
                        "maximum_raster_median_mm_d": float(values.max()),
                        "material_response_threshold_mm_d": MATERIAL_RESPONSE_MM_D,
                        "group_median_is_material": bool(
                            float(values.median()) >= MATERIAL_RESPONSE_MM_D
                        ),
                        "aggregation_definition": (
                            "median across retained raster-radius record medians"
                        ),
                        "figure_panel": "C",
                        "figure_changed": False,
                        "interpretation_boundary": INTERPRETATION_BOUNDARY,
                    }
                )
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["county_order", "radius_order", "support_domain_order"]
    ).reset_index(drop=True)


def _statistics_row(
    *,
    statistic: str,
    statistic_label: str,
    unit: str,
    observational_unit: str,
    values: Sequence[float] | pd.Series | np.ndarray,
    scope: str,
    support_domain: str,
    primary_for_text: bool,
    definition: str,
) -> dict[str, Any]:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if numeric.empty:
        raise KnownColdReferenceError(f"No finite values for text statistic {statistic}")
    return {
        "statistic": statistic,
        "statistic_label": statistic_label,
        "unit": unit,
        "scope": scope,
        "support_domain": support_domain,
        "observational_unit": observational_unit,
        "n": int(len(numeric)),
        "ddof": SAMPLE_SD_DDOF,
        "mean": float(numeric.mean()),
        "standard_deviation": _sample_sd(numeric),
        "median": float(numeric.median()),
        "minimum": float(numeric.min()),
        "maximum": float(numeric.max()),
        "primary_for_text": bool(primary_for_text),
        "statistical_definition": definition,
        "interpretation_boundary": INTERPRETATION_BOUNDARY,
    }


def build_text_statistics(
    raster_summary: pd.DataFrame,
    county_radius_source: pd.DataFrame,
    mask_response_source: pd.DataFrame,
) -> pd.DataFrame:
    """Return explicit descriptive SD statistics for manuscript text."""

    rows: list[dict[str, Any]] = []
    for domain, spec in DOMAIN_SPECS.items():
        grouped = mask_response_source[
            mask_response_source["support_domain"].eq(domain)
        ]["group_median_abs_delta_ETa_mm_d"]
        rows.append(
            _statistics_row(
                statistic="county_radius_group_response_dispersion",
                statistic_label=(
                    f"{spec['label']} absolute ETa-equivalent response"
                ),
                unit="mm d-1",
                observational_unit="county-radius group",
                values=grouped,
                scope="four accepted county-radius groups",
                support_domain=domain,
                primary_for_text=True,
                definition=(
                    "Sample SD with ddof=1 across the four accepted group medians; "
                    "descriptive grouped spread, not validation uncertainty."
                ),
            )
        )

        raster_values = pd.to_numeric(
            raster_summary[spec["median_column"]], errors="coerce"
        ).dropna()
        rows.append(
            _statistics_row(
                statistic="strict_raster_record_response_dispersion",
                statistic_label=(
                    f"{spec['label']} raster-level median absolute response"
                ),
                unit="mm d-1",
                observational_unit="strict raster-radius record",
                values=raster_values,
                scope="all retained strict raster-radius records",
                support_domain=domain,
                primary_for_text=False,
                definition=(
                    "Sample SD with ddof=1 across raster-level medians; secondary "
                    "diagnostic because paired radii and repeated scene dates are not "
                    "treated as independent validation replicates."
                ),
            )
        )

    qa_specs = (
        (
            "thermal_gap_group_dispersion",
            "Thermal separation",
            "K",
            "thermal_gap_median_K",
        ),
        (
            "source_pixel_support_group_dispersion",
            "Concurrent source-pixel support",
            "pixels",
            "source_pixel_count_median",
        ),
        (
            "high_ETrF_clamp_group_dispersion",
            "High-ETrF clamp fraction",
            "percent",
            "high_ETrF_clamp_fraction_median",
        ),
        (
            "low_ETrF_clamp_group_dispersion",
            "Low-ETrF clamp fraction",
            "percent",
            "low_ETrF_clamp_fraction_median",
        ),
        (
            "nonconvergence_group_dispersion",
            "Nonconvergence fraction",
            "percent",
            "nonconverged_fraction_median",
        ),
        (
            "rah_cap_hit_group_dispersion",
            "Aerodynamic-resistance cap-hit fraction",
            "percent",
            "rah_cap_hit_fraction_median",
        ),
    )
    for statistic, label, unit, column in qa_specs:
        values = county_radius_source[column].astype(float)
        if unit == "percent":
            values = values * 100.0
        rows.append(
            _statistics_row(
                statistic=statistic,
                statistic_label=label,
                unit=unit,
                observational_unit="county-radius group",
                values=values,
                scope="four accepted county-radius groups",
                support_domain="numerical_QA",
                primary_for_text=True,
                definition=(
                    "Sample SD with ddof=1 across four county-radius group medians; "
                    "descriptive numerical-QA spread only."
                ),
            )
        )

    result = pd.DataFrame(rows)
    order = {domain: index for index, domain in enumerate(DOMAIN_SPECS)}
    result["domain_order"] = result["support_domain"].map(order).fillna(99).astype(int)
    result["statistic_order"] = result.groupby("support_domain", sort=False).cumcount()
    return result.sort_values(["domain_order", "statistic_order", "statistic"]).reset_index(
        drop=True
    )


def _check_record(
    checks: list[dict[str, Any]],
    *,
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


def validate_derived_outputs(
    funnel_source: pd.DataFrame,
    county_radius_source: pd.DataFrame,
    mask_response_source: pd.DataFrame,
    text_statistics: pd.DataFrame,
    *,
    raster_summary: pd.DataFrame,
    county_counts: pd.DataFrame,
    county_radius_qa: pd.DataFrame,
    solver_parameters: pd.DataFrame,
    runtime_validation_summary: Mapping[str, Any],
    contract_path: Path = DEFAULT_CONTRACT_PATH,
) -> dict[str, Any]:
    """Validate Figure 7 and text outputs against frozen Phase 3 authority."""

    checks: list[dict[str, Any]] = []
    index = 1

    def add(description: str, observed: Any, expected: Any, passed: bool) -> None:
        nonlocal index
        _check_record(
            checks,
            check_id=f"KC-{index:03d}",
            description=description,
            observed=observed,
            expected=expected,
            passed=passed,
        )
        index += 1

    add(
        "Runtime-input validation status",
        runtime_validation_summary.get("status"),
        "PASS",
        runtime_validation_summary.get("status") == "PASS",
    )
    add(
        "Runtime-input validation failures",
        runtime_validation_summary.get("failed"),
        0,
        int(runtime_validation_summary.get("failed", -1)) == 0,
    )
    add("Strict raster count", len(raster_summary), 39, len(raster_summary) == 39)
    add(
        "County raster counts",
        raster_summary["county"].value_counts().to_dict(),
        dict(EXPECTED_COUNTY_COUNTS),
        raster_summary["county"].value_counts().to_dict()
        == dict(EXPECTED_COUNTY_COUNTS),
    )
    observed_group_counts = {
        f"{county}_{int(radius)}m": int(count)
        for (county, radius), count in raster_summary.groupby(
            ["county", "radius_m"]
        ).size().to_dict().items()
    }
    expected_group_counts = {
        f"{county}_{radius}m": count
        for (county, radius), count in EXPECTED_GROUP_COUNTS.items()
    }
    add(
        "County-radius raster counts",
        observed_group_counts,
        expected_group_counts,
        observed_group_counts == expected_group_counts,
    )
    add(
        "Every raster is readable",
        int(raster_summary["read_status"].eq("PASS_READABLE").sum()),
        39,
        raster_summary["read_status"].eq("PASS_READABLE").all(),
    )
    add(
        "Every raster passes hierarchy",
        int(raster_summary["hierarchy_pass"].astype(int).sum()),
        39,
        raster_summary["hierarchy_pass"].astype(int).eq(1).all(),
    )
    hierarchy = (
        raster_summary["valid_pixels"].ge(raster_summary["active_strict_pixels"])
        & raster_summary["active_strict_pixels"].ge(
            raster_summary["high_strict_pixels"]
        )
        & raster_summary["high_strict_pixels"].ge(
            raster_summary["mechanism_strict_pixels"]
        )
    )
    add("Nested mask hierarchy", int(hierarchy.sum()), 39, hierarchy.all())
    add(
        "Processing-funnel rows",
        len(funnel_source),
        4,
        len(funnel_source) == 4,
    )
    add(
        "Processing-funnel counts",
        funnel_source["count"].astype(int).tolist(),
        [39, 39, 39, 39],
        funnel_source["count"].astype(int).eq(39).all(),
    )
    add(
        "County-radius source rows",
        len(county_radius_source),
        4,
        len(county_radius_source) == 4,
    )
    add(
        "Mask-response source rows",
        len(mask_response_source),
        16,
        len(mask_response_source) == 16,
    )
    add(
        "Mask-response keys are unique",
        int(mask_response_source.duplicated(["county", "radius_m", "support_domain"]).sum()),
        0,
        not mask_response_source.duplicated(
            ["county", "radius_m", "support_domain"]
        ).any(),
    )
    add(
        "Text-statistics rows",
        len(text_statistics),
        14,
        len(text_statistics) == 14,
    )
    required_text_columns = {
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
    }
    add(
        "Text-statistics definitions are complete",
        sorted(required_text_columns - set(text_statistics.columns)),
        [],
        required_text_columns.issubset(text_statistics.columns),
    )
    add(
        "All text statistics use ddof=1",
        sorted(text_statistics["ddof"].unique().tolist()),
        [1],
        text_statistics["ddof"].eq(1).all(),
    )
    grouped_stats = text_statistics[
        text_statistics["statistic"].eq("county_radius_group_response_dispersion")
    ]
    add(
        "Primary response SD uses four county-radius groups",
        grouped_stats["n"].astype(int).tolist(),
        [4, 4, 4, 4],
        len(grouped_stats) == 4 and grouped_stats["n"].astype(int).eq(4).all(),
    )
    raster_stats = text_statistics[
        text_statistics["statistic"].eq("strict_raster_record_response_dispersion")
    ].set_index("support_domain")
    expected_raster_n = {
        "valid_output": 39,
        "active_canopy": 39,
        "large_ETrF": 39,
        "mechanism_evidence": 38,
    }
    add(
        "Secondary raster SD uses non-null strict records",
        raster_stats["n"].astype(int).to_dict(),
        expected_raster_n,
        raster_stats["n"].astype(int).to_dict() == expected_raster_n,
    )

    group_ranges = (
        mask_response_source.groupby("support_domain")[
            "group_median_abs_delta_ETa_mm_d"
        ]
        .agg(["min", "max"])
        .to_dict("index")
    )
    add(
        "Valid-output response range rounds to manuscript source",
        [round(group_ranges["valid_output"]["min"], 2), round(group_ranges["valid_output"]["max"], 2)],
        [0.86, 1.04],
        round(group_ranges["valid_output"]["min"], 2) == 0.86
        and round(group_ranges["valid_output"]["max"], 2) == 1.04,
    )
    add(
        "Active-canopy response range rounds to manuscript source",
        [round(group_ranges["active_canopy"]["min"], 2), round(group_ranges["active_canopy"]["max"], 2)],
        [2.91, 3.92],
        round(group_ranges["active_canopy"]["min"], 2) == 2.91
        and round(group_ranges["active_canopy"]["max"], 2) == 3.92,
    )
    add(
        "Large-ETrF response range rounds to manuscript source",
        [round(group_ranges["large_ETrF"]["min"], 2), round(group_ranges["large_ETrF"]["max"], 2)],
        [1.99, 2.81],
        round(group_ranges["large_ETrF"]["min"], 2) == 1.99
        and round(group_ranges["large_ETrF"]["max"], 2) == 2.81,
    )
    add(
        "Mechanism response range rounds to manuscript source",
        [round(group_ranges["mechanism_evidence"]["min"], 2), round(group_ranges["mechanism_evidence"]["max"], 2)],
        [1.48, 2.21],
        round(group_ranges["mechanism_evidence"]["min"], 2) == 1.48
        and round(group_ranges["mechanism_evidence"]["max"], 2) == 2.21,
    )

    for domain, spec in DOMAIN_SPECS.items():
        for key in EXPECTED_GROUP_COUNTS:
            county, radius = key
            source_value = mask_response_source[
                mask_response_source["county"].eq(county)
                & mask_response_source["radius_m"].eq(radius)
                & mask_response_source["support_domain"].eq(domain)
            ]["group_median_abs_delta_ETa_mm_d"].iloc[0]
            strict = raster_summary[
                raster_summary["county"].eq(county)
                & raster_summary["radius_m"].eq(radius)
            ]
            expected_value = strict[spec["median_column"]].median()
            add(
                f"{county} {radius} m {domain} response derives from raster medians",
                float(source_value),
                float(expected_value),
                _close(source_value, expected_value),
            )

    add(
        "Nonconvergence is zero in all county-radius groups",
        county_radius_source["nonconverged_fraction_median"].tolist(),
        [0.0, 0.0, 0.0, 0.0],
        county_radius_source["nonconverged_fraction_median"].eq(0.0).all(),
    )
    add(
        "Aerodynamic-resistance cap hits are zero",
        county_radius_source["rah_cap_hit_fraction_median"].tolist(),
        [0.0, 0.0, 0.0, 0.0],
        county_radius_source["rah_cap_hit_fraction_median"].eq(0.0).all(),
    )
    high_pct = county_radius_source["high_ETrF_clamp_fraction_median"] * 100.0
    low_pct = county_radius_source["low_ETrF_clamp_fraction_median"] * 100.0
    add(
        "High-ETrF clamp range rounds to accepted QA",
        [round(float(high_pct.min()), 2), round(float(high_pct.max()), 2)],
        [0.21, 0.77],
        round(float(high_pct.min()), 2) == 0.21
        and round(float(high_pct.max()), 2) == 0.77,
    )
    add(
        "Low-ETrF clamp range rounds to accepted QA",
        [round(float(low_pct.min()), 2), round(float(low_pct.max()), 2)],
        [0.96, 1.36],
        round(float(low_pct.min()), 2) == 0.96
        and round(float(low_pct.max()), 2) == 1.36,
    )
    add(
        "Every group median thermal gap exceeds 6.5 K",
        float(county_radius_source["thermal_gap_median_K"].min()),
        ">= 6.5",
        county_radius_source["thermal_gap_median_K"].ge(6.5).all(),
    )
    add(
        "Every strict raster has at least 30 source pixels",
        float(county_radius_source["source_pixel_count_minimum"].min()),
        ">= 30",
        county_radius_source["source_pixel_count_minimum"].ge(30).all(),
    )
    for county in EXPECTED_COUNTIES:
        values = county_radius_source[
            county_radius_source["county"].eq(county)
        ].set_index("radius_m")["source_pixel_count_median"]
        add(
            f"{county} 3000 m source support exceeds 1000 m",
            {int(radius): float(value) for radius, value in values.items()},
            "3000 m > 1000 m",
            float(values.loc[3000]) > float(values.loc[1000]),
        )

    add(
        "Solver parameters remain exact",
        solver_parameters.set_index("parameter")["value"].to_dict(),
        dict(EXPECTED_SOLVER_PARAMETERS),
        solver_parameters.set_index("parameter")["value"].to_dict()
        == dict(EXPECTED_SOLVER_PARAMETERS),
    )
    add(
        "County count source reconciles",
        county_counts.set_index("county")["raster_count"].astype(int).to_dict(),
        dict(EXPECTED_COUNTY_COUNTS),
        county_counts.set_index("county")["raster_count"].astype(int).to_dict()
        == dict(EXPECTED_COUNTY_COUNTS),
    )
    add(
        "County-radius QA source reconciles",
        int(len(county_radius_qa)),
        4,
        len(county_radius_qa) == 4,
    )
    add(
        "Figure design remains unchanged",
        bool(
            funnel_source["figure_changed"].any()
            or county_radius_source["figure_changed"].any()
            or mask_response_source["figure_changed"].any()
        ),
        False,
        not funnel_source["figure_changed"].any()
        and not county_radius_source["figure_changed"].any()
        and not mask_response_source["figure_changed"].any(),
    )
    add(
        "Scientific boundary is attached to every public row",
        {
            "funnel": funnel_source["interpretation_boundary"].nunique(),
            "county_radius": county_radius_source[
                "interpretation_boundary"
            ].nunique(),
            "mask": mask_response_source["interpretation_boundary"].nunique(),
            "text": text_statistics["interpretation_boundary"].nunique(),
        },
        "one identical boundary per output",
        funnel_source["interpretation_boundary"].eq(INTERPRETATION_BOUNDARY).all()
        and county_radius_source["interpretation_boundary"].eq(
            INTERPRETATION_BOUNDARY
        ).all()
        and mask_response_source["interpretation_boundary"].eq(
            INTERPRETATION_BOUNDARY
        ).all()
        and text_statistics["interpretation_boundary"].eq(
            INTERPRETATION_BOUNDARY
        ).all(),
    )

    contract = load_source_contract(contract_path)
    allowed = set(contract.allowed_outputs(MODULE_NAME))
    add(
        "Source contract authorizes all outputs",
        sorted(allowed),
        sorted(EXPECTED_OUTPUT_NAMES),
        allowed == set(EXPECTED_OUTPUT_NAMES),
    )

    failed = sum(check["status"] == "FAIL" for check in checks)
    passed = len(checks) - failed
    return {
        "validation_version": "1.0.0",
        "module": MODULE_NAME,
        "status": "PASS" if failed == 0 else "FAIL",
        "checks": len(checks),
        "passed": passed,
        "warnings": 0,
        "failed": failed,
        "runtime_input_validation": dict(runtime_validation_summary),
        "accepted_population": {
            "strict_rasters": EXPECTED_RASTER_COUNT,
            "county_raster_counts": dict(EXPECTED_COUNTY_COUNTS),
            "county_radius_raster_counts": {
                f"{county}_{radius}m": count
                for (county, radius), count in EXPECTED_GROUP_COUNTS.items()
            },
            "source_radii_m": list(EXPECTED_RADII_M),
            "same_date_source_only": True,
        },
        "standard_deviation": {
            "primary": {
                "type": "sample standard deviation",
                "observational_unit": "county-radius group median",
                "n_per_support_domain": 4,
                "ddof": SAMPLE_SD_DDOF,
                "interpretation": "descriptive grouped spread only",
            },
            "secondary": {
                "type": "sample standard deviation",
                "observational_unit": "strict raster-radius record median",
                "n": {
                    "valid_output": 39,
                    "active_canopy": 39,
                    "large_ETrF": 39,
                    "mechanism_evidence": 38,
                },
                "ddof": SAMPLE_SD_DDOF,
                "interpretation": "secondary diagnostic spread only",
            },
        },
        "figure_change": False,
        "scientific_boundary": INTERPRETATION_BOUNDARY,
        "check_results": checks,
    }


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
    output_dir: Path,
    *,
    contract_path: Path,
    overwrite: bool,
) -> KnownColdReferenceOutputs:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    contract = load_source_contract(contract_path)
    for name in EXPECTED_OUTPUT_NAMES:
        assert_allowed_output_name(contract, MODULE_NAME, name)
    outputs = KnownColdReferenceOutputs(
        figure7_processing_funnel_source=out
        / "figure7_processing_funnel_source.csv",
        figure7_county_radius_source=out / "figure7_county_radius_source.csv",
        figure7_mask_response_source=out / "figure7_mask_response_source.csv",
        known_cold_reference_text_statistics=out
        / "known_cold_reference_text_statistics.csv",
        known_cold_reference_validation=out / "known_cold_reference_validation.json",
        known_cold_reference_execution_record=out
        / "known_cold_reference_execution_record.json",
    )
    existing = [path for path in outputs.paths() if path.exists()]
    if existing and not overwrite:
        raise KnownColdReferenceError(
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
    outputs: KnownColdReferenceOutputs,
) -> dict[str, Any]:
    contract = load_source_contract(contract_path)
    input_files = {
        item.relative_path: {"sha256": item.sha256}
        for item in contract.files_for_module(MODULE_NAME)
    }
    output_records = {
        path.name: _file_record(path)
        for path in outputs.paths()
        if path != outputs.known_cold_reference_execution_record
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
            "qstable_awm.known_cold_reference",
            "--phase3-root",
            "PHASE3_CURATED_INPUT_LAYER_FINAL_002",
            "--output-dir",
            "results/known_cold_reference",
            "--contract",
            "config/phase4_source_contract.yml",
        ],
        "statistical_decisions": validation["standard_deviation"],
        "solver_authority": {
            "supplementary_method": METHODS_RELATIVE_PATH,
            "parameter_table": SOLVER_PARAMETERS_RELATIVE_PATH,
            "main_thermal_separation_gate_K": 6.5,
            "same_date_source_only": True,
            "primary_source_statistic": "median",
            "minimum_source_pixels": 30,
            "material_response_threshold_mm_d": MATERIAL_RESPONSE_MM_D,
        },
        "figure_change": False,
        "scientific_boundary": INTERPRETATION_BOUNDARY,
        "determinism_note": (
            "Outputs derive only from six frozen Phase 3 inputs; full-resolution "
            "rasters remain external and are not opened. Rows and columns are explicitly ordered."
        ),
    }


def run_known_cold_reference(
    phase3_root: Path,
    output_dir: Path,
    *,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
    overwrite: bool = False,
) -> KnownColdReferenceOutputs:
    """Validate inputs, derive Figure 7/text sources, and write six outputs."""

    contract_path = Path(contract_path).expanduser().resolve()
    outputs = _prepare_outputs(
        output_dir, contract_path=contract_path, overwrite=overwrite
    )
    (
        raster_summary,
        county_counts,
        county_radius_qa,
        funnel,
        solver_parameters,
        methods_text,
        runtime_summary,
    ) = load_known_cold_inputs(phase3_root, contract_path=contract_path)
    _assert_input_structure(
        raster_summary,
        county_counts,
        county_radius_qa,
        funnel,
        solver_parameters,
        methods_text,
    )

    funnel_source = build_processing_funnel_source(funnel)
    county_radius_source = build_county_radius_source(raster_summary)
    mask_response_source = build_mask_response_source(raster_summary)
    text_statistics = build_text_statistics(
        raster_summary, county_radius_source, mask_response_source
    )
    validation = validate_derived_outputs(
        funnel_source,
        county_radius_source,
        mask_response_source,
        text_statistics,
        raster_summary=raster_summary,
        county_counts=county_counts,
        county_radius_qa=county_radius_qa,
        solver_parameters=solver_parameters,
        runtime_validation_summary=runtime_summary,
        contract_path=contract_path,
    )
    if validation["status"] != "PASS":
        failures = [
            check for check in validation["check_results"] if check["status"] == "FAIL"
        ]
        raise KnownColdReferenceError(
            f"Derived-output validation failed: {failures}"
        )

    _atomic_write_csv(funnel_source, outputs.figure7_processing_funnel_source)
    _atomic_write_csv(county_radius_source, outputs.figure7_county_radius_source)
    _atomic_write_csv(mask_response_source, outputs.figure7_mask_response_source)
    _atomic_write_csv(
        text_statistics, outputs.known_cold_reference_text_statistics
    )
    _atomic_write_json(validation, outputs.known_cold_reference_validation)

    csv_pairs = (
        (outputs.figure7_processing_funnel_source, funnel_source),
        (outputs.figure7_county_radius_source, county_radius_source),
        (outputs.figure7_mask_response_source, mask_response_source),
        (outputs.known_cold_reference_text_statistics, text_statistics),
    )
    for path, expected in csv_pairs:
        try:
            pd.testing.assert_frame_equal(
                pd.read_csv(path, keep_default_na=False),
                expected,
                check_dtype=False,
                check_exact=False,
                rtol=CSV_ROUNDTRIP_RTOL,
                atol=CSV_ROUNDTRIP_ATOL,
            )
        except AssertionError as exc:
            raise KnownColdReferenceError(
                f"CSV round-trip verification failed for {path.name}: {exc}"
            ) from exc

    with outputs.known_cold_reference_validation.open(
        "r", encoding="utf-8"
    ) as stream:
        serialized_validation = json.load(stream)
    canonical_validation = json.loads(
        json.dumps(validation, sort_keys=True, ensure_ascii=False, allow_nan=False)
    )
    if serialized_validation != canonical_validation:
        raise KnownColdReferenceError(
            "Validation JSON round-trip verification failed"
        )
    for path in (
        outputs.figure7_processing_funnel_source,
        outputs.figure7_county_radius_source,
        outputs.figure7_mask_response_source,
        outputs.known_cold_reference_text_statistics,
        outputs.known_cold_reference_validation,
    ):
        if not path.is_file() or path.stat().st_size <= 0:
            raise KnownColdReferenceError(f"Output was not written correctly: {path}")

    execution_record = _build_execution_record(
        contract_path=contract_path,
        runtime_validation_summary=runtime_summary,
        validation=validation,
        outputs=outputs,
    )
    _atomic_write_json(
        execution_record, outputs.known_cold_reference_execution_record
    )
    return outputs


def _text_summary(statistics_path: Path, mask_path: Path) -> str:
    statistics = pd.read_csv(statistics_path)
    mask = pd.read_csv(mask_path)
    primary = statistics[
        statistics["statistic"].eq("county_radius_group_response_dispersion")
    ].set_index("support_domain")
    lines = [
        "Known-cold reference-sensitivity descriptive statistics:",
        "Primary SD: sample SD across four county-radius group medians (ddof=1).",
    ]
    for domain, spec in DOMAIN_SPECS.items():
        row = primary.loc[domain]
        source = mask[mask["support_domain"].eq(domain)][
            "group_median_abs_delta_ETa_mm_d"
        ]
        lines.append(
            f"- {spec['label']}: {row['mean']:.3f} ± "
            f"{row['standard_deviation']:.3f} mm d-1 "
            f"(n={int(row['n'])} groups; range {source.min():.2f}-{source.max():.2f})."
        )
    lines.append(
        "These SDs describe grouped reference-sensitivity dispersion, not ET validation uncertainty."
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Derive Figure 7 known-cold reference-sensitivity sources."
    )
    parser.add_argument("--phase3-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing known-cold outputs in the selected directory.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        outputs = run_known_cold_reference(
            args.phase3_root,
            args.output_dir,
            contract_path=args.contract,
            overwrite=args.overwrite,
        )
    except (
        KnownColdReferenceError,
        RuntimeInputError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        print(f"KNOWN_COLD_REFERENCE_ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(outputs.as_dict(), indent=2, sort_keys=True))
    print(
        _text_summary(
            outputs.known_cold_reference_text_statistics,
            outputs.figure7_mask_response_source,
        )
    )
    return 0


__all__ = [
    "KnownColdReferenceError",
    "KnownColdReferenceOutputs",
    "build_county_radius_source",
    "build_mask_response_source",
    "build_processing_funnel_source",
    "build_text_statistics",
    "load_known_cold_inputs",
    "run_known_cold_reference",
    "validate_derived_outputs",
]


if __name__ == "__main__":
    raise SystemExit(main())
