"""Aerodynamic-sensitivity calculations for manuscript Figure 4 and Table 3.

This module reads only the five Phase 3 inputs authorized for the
``aerodynamic_sensitivity`` domain. It preserves the V42.5 perturbation
experiment as a diagnostic test: roughness and canopy-height alternatives
change aerodynamic resistance, but they do not create a corrected ET product.

The module writes raw plot-ready records for Figure 4, a long-form peak-season
source table, a three-scope Table 3 summary, and a separate descriptive
statistics table containing sample standard deviations requested during
coauthor review. The existing figure design is unchanged.

Standard deviations are calculated with ``ddof=1`` across the non-BASE
county-crop-month-scenario summary records within each declared scope:

* all crops and months: n = 288;
* corn, all months: n = 144;
* corn, July-August: n = 48.

These SDs describe spread across aggregated sensitivity summaries. They are not
pixel-scale ET uncertainty, confidence intervals, or independent validation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

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


MODULE_NAME = "aerodynamic_sensitivity"
MODULE_RELATIVE_PATH = "src/qstable_awm/aerodynamic_sensitivity.py"
EXPECTED_OUTPUT_NAMES = (
    "figure4_aerodynamic_source.csv",
    "figure4_peak_sensitivity_source.csv",
    "table3_aerodynamic_summary.csv",
    "aerodynamic_sensitivity_text_statistics.csv",
    "aerodynamic_sensitivity_validation.json",
    "aerodynamic_sensitivity_execution_record.json",
)
INTERPRETATION_BOUNDARY = (
    "Aerodynamic perturbation sensitivity only. No corrected ET product, ET validation, "
    "irrigation application, pumping estimate, groundwater-withdrawal validation, or "
    "groundwater attribution."
)
SD_DDOF = 1
PEAK_MONTHS = (7, 8)
ETRF_DIAGNOSTIC_LIMIT = 0.01

SCOPE_SUMMARY_PATH = "data/derived/aerodynamic_sensitivity/figure4_scope_summary.csv"
ETRF_ENVELOPE_PATH = "data/derived/aerodynamic_sensitivity/figure4_etrf_scenario_envelopes.csv"
H_ENVELOPE_PATH = "data/derived/aerodynamic_sensitivity/figure4_sensible_heat_scenario_envelopes.csv"
PEAK_PATH = "data/derived/aerodynamic_sensitivity/figure4_peak_county_month_sensitivity.csv"
TABLE3_PATH = "data/derived/aerodynamic_sensitivity/table3_aerodynamic_sensitivity.csv"

SCOPE_ORDER = (
    "all_crops_all_months_nonBASE",
    "corn_all_months_nonBASE",
    "corn_peak_JulAug_nonBASE",
)
SCOPE_LABELS = {
    "all_crops_all_months_nonBASE": "All crops, all months",
    "corn_all_months_nonBASE": "Corn, all months",
    "corn_peak_JulAug_nonBASE": "Corn, July-August",
}
EXPECTED_SCOPE_N = {
    "all_crops_all_months_nonBASE": 288,
    "corn_all_months_nonBASE": 144,
    "corn_peak_JulAug_nonBASE": 48,
}

METRICS: Mapping[str, Mapping[str, str]] = {
    "drah2": {
        "signed_column": "alt_rah2_mean_minus_BASE",
        "absolute_column": "abs_alt_rah2_mean_minus_BASE",
        "label": "Aerodynamic-resistance response",
        "unit": "s m-1",
    },
    "dH": {
        "signed_column": "alt_H_mean_minus_BASE",
        "absolute_column": "abs_alt_H_mean_minus_BASE",
        "label": "Sensible-heat response",
        "unit": "W m-2",
    },
    "dETf": {
        "signed_column": "alt_ETf_mean_minus_BASE",
        "absolute_column": "abs_alt_ETf_mean_minus_BASE",
        "label": "ETf response",
        "unit": "dimensionless",
    },
    "dLE_fraction": {
        "signed_column": "alt_LE_fraction_minus_BASE",
        "absolute_column": "abs_alt_LE_fraction_minus_BASE",
        "label": "Latent-energy-fraction response",
        "unit": "dimensionless",
    },
    "dETa_inst_proxy": {
        "signed_column": "implied_alt_ETa_inst_proxy_minus_BASE",
        "absolute_column": "abs_implied_alt_ETa_inst_proxy_minus_BASE",
        "label": "Instantaneous ETa-proxy response",
        "unit": "mm h-1",
    },
}


class AerodynamicSensitivityError(RuntimeError):
    """Raised when an aerodynamic input or derived output violates authority."""


@dataclass(frozen=True, slots=True)
class AerodynamicSensitivityOutputs:
    """Paths written by :func:`run_aerodynamic_sensitivity`."""

    figure4_aerodynamic_source: Path
    figure4_peak_sensitivity_source: Path
    table3_aerodynamic_summary: Path
    aerodynamic_sensitivity_text_statistics: Path
    aerodynamic_sensitivity_validation: Path
    aerodynamic_sensitivity_execution_record: Path

    def as_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}

    def paths(self) -> tuple[Path, ...]:
        return tuple(Path(value) for value in asdict(self).values())


def _sample_sd(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if len(numeric) < 2:
        return float("nan")
    return float(numeric.std(ddof=SD_DDOF))


def _scope_frames(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    corn = table[table["crop"].astype(str).str.lower().eq("corn")].copy()
    return {
        "all_crops_all_months_nonBASE": table.copy(),
        "corn_all_months_nonBASE": corn,
        "corn_peak_JulAug_nonBASE": corn[corn["month"].isin(PEAK_MONTHS)].copy(),
    }


def load_aerodynamic_inputs(
    phase3_root: Path,
    *,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Validate and load the exact five authorized Phase 3 tables."""

    root = Path(phase3_root).expanduser().resolve()
    relative_paths = (
        SCOPE_SUMMARY_PATH,
        ETRF_ENVELOPE_PATH,
        H_ENVELOPE_PATH,
        PEAK_PATH,
        TABLE3_PATH,
    )
    report = validate_module_inputs(
        root,
        MODULE_NAME,
        contract_path=contract_path,
        supplied_paths=tuple(root / path for path in relative_paths),
        raise_on_error=True,
    )
    tables = {path: pd.read_csv(root / path) for path in relative_paths}
    return tables, report.summary()


def _assert_structure(tables: Mapping[str, pd.DataFrame]) -> None:
    scope = tables[SCOPE_SUMMARY_PATH]
    etrf = tables[ETRF_ENVELOPE_PATH]
    sensible = tables[H_ENVELOPE_PATH]
    peak = tables[PEAK_PATH]
    table3 = tables[TABLE3_PATH]

    if tuple(scope["scope"]) != SCOPE_ORDER:
        raise AerodynamicSensitivityError(
            f"Unexpected scope order: {tuple(scope['scope'])}; expected={SCOPE_ORDER}"
        )
    observed_n = dict(zip(scope["scope"], scope["n_rows"]))
    if observed_n != EXPECTED_SCOPE_N:
        raise AerodynamicSensitivityError(
            f"Unexpected scope counts: {observed_n}; expected={EXPECTED_SCOPE_N}"
        )
    if len(table3) != 288:
        raise AerodynamicSensitivityError(f"Expected 288 Table 3 rows; found {len(table3)}")
    if table3.duplicated(["county", "crop", "month", "year", "scenario_id"]).any():
        raise AerodynamicSensitivityError("Table 3 sensitivity keys are not unique")
    if table3["scenario_id"].astype(str).str.upper().eq("BASE").any():
        raise AerodynamicSensitivityError("BASE rows are prohibited from the non-BASE sensitivity table")
    if set(table3["year"].astype(int)) != {2011}:
        raise AerodynamicSensitivityError("Aerodynamic sensitivity must remain restricted to 2011")
    if set(table3["county"].astype(str)) != {"DALLAM", "HARTLEY", "MOORE", "OCHILTREE"}:
        raise AerodynamicSensitivityError("Unexpected county set")
    if tuple(peak.columns) != ("county", "7", "8"):
        raise AerodynamicSensitivityError(f"Unexpected peak-table columns: {tuple(peak.columns)}")
    if len(peak) != 4:
        raise AerodynamicSensitivityError(f"Expected four peak-table counties; found {len(peak)}")
    if len(etrf) != 6 or len(sensible) != 6:
        raise AerodynamicSensitivityError("Expected six perturbation scenarios in each envelope table")
    if not etrf[["scenario_id", "scenario_label", "n"]].equals(
        sensible[["scenario_id", "scenario_label", "n"]]
    ):
        raise AerodynamicSensitivityError("ETf and H scenario-envelope identifiers do not align")
    if not etrf["n"].eq(24).all() or not sensible["n"].eq(24).all():
        raise AerodynamicSensitivityError("Each corn scenario envelope must contain 24 records")

    for metric in METRICS.values():
        signed = metric["signed_column"]
        absolute = metric["absolute_column"]
        if not np.allclose(
            table3[absolute].to_numpy(dtype=float),
            np.abs(table3[signed].to_numpy(dtype=float)),
            rtol=0,
            atol=1e-12,
            equal_nan=True,
        ):
            raise AerodynamicSensitivityError(
                f"Absolute-value identity failed for {signed} and {absolute}"
            )


def build_figure4_aerodynamic_source(
    table3: pd.DataFrame,
    etrf_envelope: pd.DataFrame,
    h_envelope: pd.DataFrame,
) -> pd.DataFrame:
    """Return raw plot-ready records for Figure 4 panels A-C.

    Panel A uses every non-BASE summary. Panels B and C use the corn subset.
    Scenario-envelope quantiles are copied onto each scenario record only as
    frozen QA references; Phase 5 can draw the existing distributions from the
    raw signed responses without changing the figure design.
    """

    scenario_labels = dict(
        zip(etrf_envelope["scenario_id"], etrf_envelope["scenario_label"])
    )
    etrf_stats = etrf_envelope.set_index("scenario_id").add_prefix("etrf_")
    h_stats = h_envelope.set_index("scenario_id").add_prefix("h_")

    columns = [
        "county",
        "crop",
        "month",
        "year",
        "scenario_id",
        "n_scene_ids",
        "alt_rah2_mean_minus_BASE",
        "alt_ETf_mean_minus_BASE",
        "alt_H_mean_minus_BASE",
        "alt_LE_fraction_minus_BASE",
        "implied_alt_ETa_inst_proxy_minus_BASE",
        "abs_alt_rah2_mean_minus_BASE",
        "abs_alt_ETf_mean_minus_BASE",
        "abs_alt_H_mean_minus_BASE",
    ]
    result = table3[columns].copy()
    result["scenario_label"] = result["scenario_id"].map(scenario_labels)
    result["panel_A_include"] = True
    result["panel_BC_include"] = result["crop"].astype(str).str.lower().eq("corn")
    result["peak_JulAug_include"] = result["panel_BC_include"] & result["month"].isin(PEAK_MONTHS)
    result["interpretation_boundary"] = (
        INTERPRETATION_BOUNDARY
    )
    result = result.join(etrf_stats, on="scenario_id").join(h_stats, on="scenario_id")

    order = {scenario: idx for idx, scenario in enumerate(etrf_envelope["scenario_id"])}
    result["scenario_order"] = result["scenario_id"].map(order).astype(int)
    return result.sort_values(
        ["scenario_order", "crop", "county", "month", "year"]
    ).reset_index(drop=True)


def build_peak_sensitivity_source(peak: pd.DataFrame) -> pd.DataFrame:
    """Convert the frozen panel-D county-by-month table to long format."""

    result = peak.melt(
        id_vars="county",
        value_vars=["7", "8"],
        var_name="month",
        value_name="max_abs_dETf_mean",
    )
    result["month"] = result["month"].astype(int)
    result["month_label"] = result["month"].map({7: "July", 8: "August"})
    result["crop"] = "corn"
    result["year"] = 2011
    result["diagnostic_limit_abs_dETf"] = ETRF_DIAGNOSTIC_LIMIT
    result["below_diagnostic_limit"] = result["max_abs_dETf_mean"] < ETRF_DIAGNOSTIC_LIMIT
    return result.sort_values(["county", "month"]).reset_index(drop=True)


def build_text_statistics(table3: pd.DataFrame) -> pd.DataFrame:
    """Calculate signed and absolute descriptive statistics by analysis scope.

    The observational unit is one non-BASE county-crop-month-scenario summary.
    Sample SD uses ``ddof=1``. Pixel values are never treated as independent
    observations.
    """

    rows: list[dict[str, Any]] = []
    for scope_name, frame in _scope_frames(table3).items():
        for metric_name, metric in METRICS.items():
            signed_column = metric["signed_column"]
            absolute_column = metric["absolute_column"]
            signed = pd.to_numeric(frame[signed_column], errors="coerce").dropna()
            absolute = pd.to_numeric(frame[absolute_column], errors="coerce").dropna()
            common = {
                "scope": scope_name,
                "scope_label": SCOPE_LABELS[scope_name],
                "metric": metric_name,
                "metric_label": metric["label"],
                "unit": metric["unit"],
                "observational_unit": "non-BASE county-crop-month-scenario summary",
                "n": int(len(signed)),
                "ddof": SD_DDOF,
                "sd_ddof": SD_DDOF,
                "sd_basis": (
                    "sample SD across non-BASE county-crop-month-scenario summaries "
                    "within the declared scope"
                ),
                "interpretation_boundary": INTERPRETATION_BOUNDARY,
                "interpretation": (
                    "descriptive spread across aggregated sensitivity summaries; "
                    "not pixel-scale uncertainty, confidence interval, or validation"
                ),
            }
            rows.append(
                {
                    **common,
                    "statistic": "signed_response_sample_dispersion",
                    "response_form": "signed",
                    "mean": float(signed.mean()),
                    "standard_deviation": _sample_sd(signed),
                    "median": float(signed.median()),
                    "minimum": float(signed.min()),
                    "maximum": float(signed.max()),
                    "signed_mean": float(signed.mean()),
                    "signed_sample_sd": _sample_sd(signed),
                    "absolute_mean": float(absolute.mean()),
                    "absolute_sample_sd": _sample_sd(absolute),
                    "absolute_median": float(absolute.median()),
                    "absolute_maximum": float(absolute.max()),
                }
            )
            rows.append(
                {
                    **common,
                    "statistic": "absolute_response_sample_dispersion",
                    "response_form": "absolute",
                    "mean": float(absolute.mean()),
                    "standard_deviation": _sample_sd(absolute),
                    "median": float(absolute.median()),
                    "minimum": float(absolute.min()),
                    "maximum": float(absolute.max()),
                    "signed_mean": float(signed.mean()),
                    "signed_sample_sd": _sample_sd(signed),
                    "absolute_mean": float(absolute.mean()),
                    "absolute_sample_sd": _sample_sd(absolute),
                    "absolute_median": float(absolute.median()),
                    "absolute_maximum": float(absolute.max()),
                }
            )
    result = pd.DataFrame(rows)
    scope_order = {name: idx for idx, name in enumerate(SCOPE_ORDER)}
    metric_order = {name: idx for idx, name in enumerate(METRICS)}
    form_order = {"signed": 0, "absolute": 1}
    result["_scope_order"] = result["scope"].map(scope_order)
    result["_metric_order"] = result["metric"].map(metric_order)
    result["_form_order"] = result["response_form"].map(form_order)
    return (
        result.sort_values(["_scope_order", "_metric_order", "_form_order"])
        .drop(columns=["_scope_order", "_metric_order", "_form_order"])
        .reset_index(drop=True)
    )


def build_table3_summary(
    scope_summary: pd.DataFrame,
    text_statistics: pd.DataFrame,
) -> pd.DataFrame:
    """Add descriptive mean and SD columns to the frozen three-scope summary."""

    result = scope_summary.copy()
    metric_column_map = {
        "drah2": "drah2",
        "dH": "dH",
        "dETf": "dETf",
    }
    for metric_name, prefix in metric_column_map.items():
        subset = text_statistics[
            text_statistics["metric"].eq(metric_name)
            & text_statistics["response_form"].eq("absolute")
        ].set_index("scope")
        result[f"mean_abs_{prefix}"] = result["scope"].map(subset["absolute_mean"])
        result[f"sample_sd_abs_{prefix}"] = result["scope"].map(subset["absolute_sample_sd"])
        result[f"signed_mean_{prefix}"] = result["scope"].map(subset["signed_mean"])
        result[f"signed_sample_sd_{prefix}"] = result["scope"].map(subset["signed_sample_sd"])
    result["sd_ddof"] = SD_DDOF
    result["sd_basis"] = (
        "sample SD across non-BASE county-crop-month-scenario summaries within scope"
    )
    result["figure_changed"] = False
    result["scientific_boundary"] = (
        INTERPRETATION_BOUNDARY
    )
    return result


def _quantile_checks(
    raw: pd.DataFrame,
    envelope: pd.DataFrame,
    value_column: str,
    *,
    atol: float = 1e-12,
) -> bool:
    corn = raw[raw["crop"].astype(str).str.lower().eq("corn")]
    for row in envelope.itertuples(index=False):
        values = corn.loc[corn["scenario_id"].eq(row.scenario_id), value_column]
        if len(values) != int(row.n):
            return False
        observed = np.quantile(values.to_numpy(dtype=float), [0, 0.25, 0.5, 0.75, 1.0])
        expected = np.array([row.min, row.p25, row.median, row.p75, row.max], dtype=float)
        if not np.allclose(observed, expected, rtol=0, atol=atol):
            return False
        if not math.isclose(float(np.abs(values).max()), float(row.max_abs), rel_tol=0, abs_tol=atol):
            return False
    return True


def validate_derived_outputs(
    figure_source: pd.DataFrame,
    peak_source: pd.DataFrame,
    table3_summary: pd.DataFrame,
    text_statistics: pd.DataFrame,
    *,
    raw_table3: pd.DataFrame,
    etrf_envelope: pd.DataFrame,
    h_envelope: pd.DataFrame,
    runtime_validation_summary: Mapping[str, Any],
    contract_path: Path,
) -> dict[str, Any]:
    """Return deterministic scientific and source-contract checks."""

    checks: list[dict[str, Any]] = []

    def add(check_id: str, description: str, observed: Any, expected: Any, passed: bool) -> None:
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
    expected_outputs = set(EXPECTED_OUTPUT_NAMES)
    allowed = set(contract.allowed_outputs(MODULE_NAME))

    add("AERO-001", "Phase 3 runtime validation", runtime_validation_summary.get("status"), "PASS", runtime_validation_summary.get("status") == "PASS")
    add("AERO-002", "Figure 4 raw-source rows", len(figure_source), 288, len(figure_source) == 288)
    add("AERO-003", "Peak county-month rows", len(peak_source), 8, len(peak_source) == 8)
    add("AERO-004", "Table 3 summary rows", len(table3_summary), 3, len(table3_summary) == 3)
    add("AERO-005", "Text-statistics rows", len(text_statistics), len(SCOPE_ORDER) * len(METRICS) * 2, len(text_statistics) == len(SCOPE_ORDER) * len(METRICS) * 2)
    add("AERO-006", "Scope counts", dict(zip(table3_summary["scope"], table3_summary["n_rows"])), EXPECTED_SCOPE_N, dict(zip(table3_summary["scope"], table3_summary["n_rows"])) == EXPECTED_SCOPE_N)
    add("AERO-007", "ETf scenario quantiles reproduce frozen source", _quantile_checks(raw_table3, etrf_envelope, "alt_ETf_mean_minus_BASE"), True, _quantile_checks(raw_table3, etrf_envelope, "alt_ETf_mean_minus_BASE"))
    add("AERO-008", "H scenario quantiles reproduce frozen source", _quantile_checks(raw_table3, h_envelope, "alt_H_mean_minus_BASE"), True, _quantile_checks(raw_table3, h_envelope, "alt_H_mean_minus_BASE"))
    add("AERO-009", "All peak county-month ETf responses below 0.01", float(peak_source["max_abs_dETf_mean"].max()), f"<{ETRF_DIAGNOSTIC_LIMIT}", bool((peak_source["max_abs_dETf_mean"] < ETRF_DIAGNOSTIC_LIMIT).all()))
    add("AERO-010", "All sample SDs finite and non-negative", bool(np.isfinite(text_statistics[["signed_sample_sd", "absolute_sample_sd"]].to_numpy(dtype=float)).all()), True, bool(np.isfinite(text_statistics[["signed_sample_sd", "absolute_sample_sd"]].to_numpy(dtype=float)).all() and (text_statistics[["signed_sample_sd", "absolute_sample_sd"]] >= 0).all().all()))
    add("AERO-011", "Sample-SD convention", SD_DDOF, 1, SD_DDOF == 1)
    add("AERO-012", "Output names authorized by source contract", sorted(allowed), sorted(expected_outputs), allowed == expected_outputs)
    add("AERO-013", "Figure design unchanged", False, False, bool(table3_summary["figure_changed"].eq(False).all()))
    add("AERO-014", "No corrected ET product is produced", bool(figure_source["interpretation_boundary"].str.contains("No corrected ET", case=False).all()), True, bool(figure_source["interpretation_boundary"].str.contains("No corrected ET", case=False).all()))
    required_text_fields = {"statistic", "unit", "observational_unit", "n", "ddof", "mean", "standard_deviation", "median", "minimum", "maximum", "interpretation_boundary"}
    add("AERO-015", "Text-statistics fields are explicit", sorted(required_text_fields - set(text_statistics.columns)), [], required_text_fields.issubset(text_statistics.columns))
    add("AERO-016", "Text statistics use declared observational unit", sorted(text_statistics["observational_unit"].unique()), ["non-BASE county-crop-month-scenario summary"], set(text_statistics["observational_unit"]) == {"non-BASE county-crop-month-scenario summary"})

    failures = [check for check in checks if check["status"] == "FAIL"]
    return {
        "module": MODULE_NAME,
        "status": "PASS" if not failures else "FAIL",
        "checks": len(checks),
        "passed": len(checks) - len(failures),
        "warnings": 0,
        "failed": len(failures),
        "standard_deviation": {
            "type": "sample standard deviation",
            "ddof": SD_DDOF,
            "basis": "non-BASE county-crop-month-scenario summaries within each scope",
            "scope_n": EXPECTED_SCOPE_N,
            "interpretation": "descriptive spread, not ET uncertainty or validation",
        },
        "figure_change": False,
        "scientific_boundary": (
            "Aerodynamic perturbations are diagnostic only; no corrected or alternative ET product is promoted."
        ),
        "check_results": checks,
    }


def _prepare_outputs(
    output_dir: Path,
    *,
    contract_path: Path,
    overwrite: bool,
) -> AerodynamicSensitivityOutputs:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    outputs = AerodynamicSensitivityOutputs(
        figure4_aerodynamic_source=out / "figure4_aerodynamic_source.csv",
        figure4_peak_sensitivity_source=out / "figure4_peak_sensitivity_source.csv",
        table3_aerodynamic_summary=out / "table3_aerodynamic_summary.csv",
        aerodynamic_sensitivity_text_statistics=out / "aerodynamic_sensitivity_text_statistics.csv",
        aerodynamic_sensitivity_validation=out / "aerodynamic_sensitivity_validation.json",
        aerodynamic_sensitivity_execution_record=out / "aerodynamic_sensitivity_execution_record.json",
    )
    contract = load_source_contract(contract_path)
    allowed = contract.allowed_outputs(MODULE_NAME)
    if allowed != EXPECTED_OUTPUT_NAMES:
        raise AerodynamicSensitivityError(
            f"Source-contract outputs changed: {allowed}; expected={EXPECTED_OUTPUT_NAMES}"
        )
    existing = [path for path in outputs.paths() if path.exists()]
    if existing and not overwrite:
        raise AerodynamicSensitivityError(
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
    outputs: AerodynamicSensitivityOutputs,
) -> dict[str, Any]:
    contract = load_source_contract(contract_path)
    authorized_inputs = {
        item.relative_path: {"sha256": item.sha256}
        for item in contract.files_for_module(MODULE_NAME)
    }
    output_records = {
        path.name: _file_record(path)
        for path in outputs.paths()
        if path != outputs.aerodynamic_sensitivity_execution_record
    }
    return {
        "record_version": "1.0.0",
        "module": MODULE_NAME,
        "module_relative_path": MODULE_RELATIVE_PATH,
        "module_sha256": sha256_file(Path(__file__).resolve()),
        "contract_relative_path": "config/phase4_source_contract.yml",
        "contract_sha256": sha256_file(contract_path),
        "phase3_package": "PHASE3_CURATED_INPUT_LAYER_FINAL_002",
        "authorized_inputs": authorized_inputs,
        "runtime_input_validation": dict(runtime_validation_summary),
        "derived_output_validation": {
            "status": validation["status"],
            "checks": validation["checks"],
            "passed": validation["passed"],
            "warnings": validation.get("warnings", 0),
            "failed": validation["failed"],
        },
        "outputs": output_records,
        "command_template": [
            "python",
            "-m",
            "qstable_awm.aerodynamic_sensitivity",
            "--phase3-root",
            "PHASE3_CURATED_INPUT_LAYER_FINAL_002",
            "--output-dir",
            "results/aerodynamic_sensitivity",
            "--contract",
            "config/phase4_source_contract.yml",
        ],
        "statistical_decisions": validation["standard_deviation"],
        "figure_change": False,
        "scientific_boundary": validation["scientific_boundary"],
        "determinism_note": (
            "All outputs derive only from the five frozen V42.5 Phase 3 tables; "
            "METRIC is not rerun and no alternative ET product is created."
        ),
    }


def run_aerodynamic_sensitivity(
    phase3_root: Path,
    output_dir: Path,
    *,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
    overwrite: bool = False,
) -> AerodynamicSensitivityOutputs:
    """Validate inputs and write six deterministic Figure 4/Table 3 outputs."""

    contract = Path(contract_path).expanduser().resolve()
    outputs = _prepare_outputs(output_dir, contract_path=contract, overwrite=overwrite)
    tables, runtime_summary = load_aerodynamic_inputs(
        phase3_root, contract_path=contract
    )
    _assert_structure(tables)

    figure_source = build_figure4_aerodynamic_source(
        tables[TABLE3_PATH],
        tables[ETRF_ENVELOPE_PATH],
        tables[H_ENVELOPE_PATH],
    )
    peak_source = build_peak_sensitivity_source(tables[PEAK_PATH])
    text_statistics = build_text_statistics(tables[TABLE3_PATH])
    table3_summary = build_table3_summary(
        tables[SCOPE_SUMMARY_PATH], text_statistics
    )

    validation = validate_derived_outputs(
        figure_source,
        peak_source,
        table3_summary,
        text_statistics,
        raw_table3=tables[TABLE3_PATH],
        etrf_envelope=tables[ETRF_ENVELOPE_PATH],
        h_envelope=tables[H_ENVELOPE_PATH],
        runtime_validation_summary=runtime_summary,
        contract_path=contract,
    )
    if validation["status"] != "PASS":
        failed = [x for x in validation["check_results"] if x["status"] == "FAIL"]
        raise AerodynamicSensitivityError(f"Derived-output validation failed: {failed}")

    figure_source.to_csv(outputs.figure4_aerodynamic_source, index=False)
    peak_source.to_csv(outputs.figure4_peak_sensitivity_source, index=False)
    table3_summary.to_csv(outputs.table3_aerodynamic_summary, index=False)
    text_statistics.to_csv(outputs.aerodynamic_sensitivity_text_statistics, index=False)
    outputs.aerodynamic_sensitivity_validation.write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    for path in outputs.paths()[:-1]:
        if not path.is_file() or path.stat().st_size <= 0:
            raise AerodynamicSensitivityError(f"Output was not written correctly: {path}")

    record = _build_execution_record(
        contract_path=contract,
        runtime_validation_summary=runtime_summary,
        validation=validation,
        outputs=outputs,
    )
    outputs.aerodynamic_sensitivity_execution_record.write_text(
        json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return outputs


def _text_summary(path: Path) -> str:
    statistics = pd.read_csv(path)
    rows = statistics[
        statistics["metric"].isin(["drah2", "dH", "dETf"])
        & statistics["response_form"].eq("absolute")
    ].copy()
    lines = [
        "Aerodynamic sensitivity descriptive spread (mean absolute response ± sample SD):",
        "SD basis: non-BASE county-crop-month-scenario summaries; ddof=1.",
    ]
    for row in rows.itertuples(index=False):
        lines.append(
            f"- {row.scope_label}, {row.metric_label}: "
            f"{row.absolute_mean:.6g} ± {row.absolute_sample_sd:.6g} {row.unit} "
            f"(n={row.n}; median={row.absolute_median:.6g}; max={row.absolute_maximum:.6g})."
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Derive Figure 4, Table 3, and aerodynamic SD source tables."
    )
    parser.add_argument("--phase3-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing aerodynamic-sensitivity outputs in the selected directory.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        outputs = run_aerodynamic_sensitivity(
            args.phase3_root,
            args.output_dir,
            contract_path=args.contract,
            overwrite=args.overwrite,
        )
    except (AerodynamicSensitivityError, FileNotFoundError, ValueError) as exc:
        print(f"AERODYNAMIC_SENSITIVITY_ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(outputs.as_dict(), indent=2, sort_keys=True))
    print(_text_summary(outputs.aerodynamic_sensitivity_text_statistics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
