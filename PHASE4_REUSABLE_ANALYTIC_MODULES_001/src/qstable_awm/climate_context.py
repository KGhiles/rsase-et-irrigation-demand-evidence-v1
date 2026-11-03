"""Climate-context calculations for manuscript Figure 2 and results text.

This module reads only the two Phase 3 climate tables authorized by
``phase4_source_contract.yml``.  It reproduces the seasonal and monthly
climate summaries used by the manuscript and adds sample standard deviations
requested during coauthor review.

The standard deviations are descriptive, not uncertainty estimates for the
underlying gridded products:

* ``stress_2011_sd_across_counties`` uses the four county-season values from
  2011 (sample SD, ``ddof=1``);
* ``comparison_sd_across_county_years`` uses the twelve county-season values
  from 2015, 2017, and 2023 (sample SD, ``ddof=1``);
* ``comparison_sd_across_domain_year_means`` uses the three annual domain
  means for 2015, 2017, and 2023 (sample SD, ``ddof=1``).

The first two quantities use the same county-season observational unit and are
recommended for text reporting.  The between-year SD is retained separately
for readers who want temporal rather than combined spatial-temporal spread.
No plotting code is included, and the existing Figure 2 design is unchanged.
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


MODULE_NAME = "climate_context"
MODULE_RELATIVE_PATH = "src/qstable_awm/climate_context.py"
MONTHLY_RELATIVE_PATH = "data/derived/climate_context/monthly_climate_context.csv"
SEASONAL_RELATIVE_PATH = "data/derived/climate_context/seasonal_climate_context.csv"
STRESS_YEAR = 2011
COMPARISON_YEARS = (2015, 2017, 2023)
EXPECTED_COUNTIES = ("DALLAM", "HARTLEY", "MOORE", "OCHILTREE")
EXPECTED_MONTHS = (4, 5, 6, 7, 8, 9)
SD_DDOF = 1
EXPECTED_OUTPUT_NAMES = (
    "figure2_climate_source.csv",
    "figure2_monthly_trajectory_source.csv",
    "climate_context_text_statistics.csv",
    "climate_context_validation.json",
    "climate_context_execution_record.json",
)

METRICS: Mapping[str, tuple[str, str]] = {
    "ETr_S4_mm": ("Seasonal reference ET", "mm"),
    "P_S4_mm": ("Seasonal precipitation", "mm"),
    "Peff_S4_mm": ("Seasonal effective precipitation", "mm"),
    "ETr_minus_P_S4_mm": ("Seasonal ETr - P deficit", "mm"),
    "ETr_minus_Peff_S4_mm": ("Seasonal ETr - Peff deficit", "mm"),
}

MONTHLY_METRICS: Mapping[str, tuple[str, str]] = {
    "ETr_window_mm": ("Monthly-window reference ET", "mm"),
    "P_window_mm": ("Monthly-window precipitation", "mm"),
    "Peff_window_mm": ("Monthly-window effective precipitation", "mm"),
    "ETr_minus_P_window_mm": ("Monthly-window ETr - P deficit", "mm"),
    "ETr_minus_Peff_window_mm": ("Monthly-window ETr - Peff deficit", "mm"),
}


class ClimateContextError(RuntimeError):
    """Raised when climate inputs or derived outputs violate the frozen design."""


@dataclass(frozen=True, slots=True)
class ClimateContextOutputs:
    """Paths written by :func:`run_climate_context`."""

    figure2_climate_source: Path
    figure2_monthly_trajectory_source: Path
    climate_context_text_statistics: Path
    climate_context_validation: Path
    climate_context_execution_record: Path

    def as_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}

    def paths(self) -> tuple[Path, ...]:
        return tuple(Path(value) for value in asdict(self).values())


def _sample_sd(values: pd.Series) -> float:
    """Return sample SD with explicit finite-value handling."""

    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if len(numeric) < 2:
        return float("nan")
    return float(numeric.std(ddof=SD_DDOF))


def _percent_change(stress: float, comparison: float) -> float:
    if math.isclose(comparison, 0.0, abs_tol=1e-12):
        return float("nan")
    return 100.0 * (stress - comparison) / comparison


def load_climate_inputs(
    phase3_root: Path,
    *,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Validate and load the authorized monthly and seasonal climate tables."""

    root = Path(phase3_root).expanduser().resolve()
    report = validate_module_inputs(
        root,
        MODULE_NAME,
        contract_path=contract_path,
        supplied_paths=(
            root / MONTHLY_RELATIVE_PATH,
            root / SEASONAL_RELATIVE_PATH,
        ),
        raise_on_error=True,
    )
    monthly = pd.read_csv(root / MONTHLY_RELATIVE_PATH)
    seasonal = pd.read_csv(root / SEASONAL_RELATIVE_PATH)
    return monthly, seasonal, report.summary()


def _assert_climate_structure(monthly: pd.DataFrame, seasonal: pd.DataFrame) -> None:
    """Apply analysis-specific structural checks beyond generic schemas."""

    counties = tuple(sorted(seasonal["county"].astype(str).str.upper().unique()))
    years = tuple(sorted(int(value) for value in seasonal["year"].unique()))
    months = tuple(sorted(int(value) for value in monthly["month"].unique()))

    if counties != EXPECTED_COUNTIES:
        raise ClimateContextError(
            f"Unexpected counties: {counties}; expected={EXPECTED_COUNTIES}"
        )
    if years != (STRESS_YEAR, *COMPARISON_YEARS):
        raise ClimateContextError(
            f"Unexpected years: {years}; expected={(STRESS_YEAR, *COMPARISON_YEARS)}"
        )
    if months != EXPECTED_MONTHS:
        raise ClimateContextError(
            f"Unexpected months: {months}; expected={EXPECTED_MONTHS}"
        )
    if monthly.duplicated(["county", "year", "month"]).any():
        raise ClimateContextError("Monthly county-year-month keys are not unique")
    if seasonal.duplicated(["county", "year"]).any():
        raise ClimateContextError("Seasonal county-year keys are not unique")

    seasonal_checks = {
        "ETr_minus_P_S4_mm": seasonal["ETr_S4_mm"] - seasonal["P_S4_mm"],
        "ETr_minus_Peff_S4_mm": seasonal["ETr_S4_mm"] - seasonal["Peff_S4_mm"],
    }
    monthly_checks = {
        "ETr_minus_P_window_mm": monthly["ETr_window_mm"] - monthly["P_window_mm"],
        "ETr_minus_Peff_window_mm": (
            monthly["ETr_window_mm"] - monthly["Peff_window_mm"]
        ),
    }
    for column, expected in seasonal_checks.items():
        if not np.allclose(seasonal[column], expected, rtol=0, atol=1e-5):
            raise ClimateContextError(f"Seasonal accounting identity failed: {column}")
    for column, expected in monthly_checks.items():
        if not np.allclose(monthly[column], expected, rtol=0, atol=2e-6):
            raise ClimateContextError(f"Monthly accounting identity failed: {column}")


def build_figure2_seasonal_source(seasonal: pd.DataFrame) -> pd.DataFrame:
    """Return county-year seasonal values with domain and comparison summaries.

    The original Figure 2 can continue to use ``value_mm``.  The additional
    mean and SD columns support text revision and optional uncertainty
    annotation without requiring any change to the current figure.
    """

    rows: list[dict[str, Any]] = []
    comparison = seasonal[seasonal["year"].isin(COMPARISON_YEARS)].copy()

    for metric, (label, unit) in METRICS.items():
        domain_year = (
            seasonal.groupby("year", as_index=False)[metric]
            .agg(domain_year_mean="mean", domain_year_sd_across_counties=_sample_sd)
        )
        domain_map = domain_year.set_index("year").to_dict("index")
        comparison_mean = float(comparison[metric].mean())
        comparison_sd = _sample_sd(comparison[metric])
        comparison_year_sd = _sample_sd(
            comparison.groupby("year", sort=True)[metric].mean()
        )

        for record in seasonal[["county", "year", metric]].itertuples(index=False):
            county, year, value = record
            domain = domain_map[int(year)]
            rows.append(
                {
                    "metric": metric,
                    "metric_label": label,
                    "unit": unit,
                    "county": str(county),
                    "year": int(year),
                    "year_role": (
                        "stress_test" if int(year) == STRESS_YEAR else "comparison_year"
                    ),
                    "value_mm": float(value),
                    "domain_year_mean_mm": float(domain["domain_year_mean"]),
                    "domain_year_sd_across_counties_mm": float(
                        domain["domain_year_sd_across_counties"]
                    ),
                    "domain_year_n_counties": len(EXPECTED_COUNTIES),
                    "comparison_mean_mm": comparison_mean,
                    "comparison_sd_across_county_years_mm": comparison_sd,
                    "comparison_n_county_years": int(len(comparison)),
                    "comparison_sd_across_domain_year_means_mm": comparison_year_sd,
                    "comparison_n_years": len(COMPARISON_YEARS),
                    "anomaly_vs_comparison_mean_mm": float(value) - comparison_mean,
                }
            )

    result = pd.DataFrame(rows)
    return result.sort_values(["metric", "year", "county"]).reset_index(drop=True)


def build_text_statistics(seasonal: pd.DataFrame) -> pd.DataFrame:
    """Build text-ready seasonal means, sample SDs, differences, and percentages."""

    stress = seasonal[seasonal["year"] == STRESS_YEAR]
    comparison = seasonal[seasonal["year"].isin(COMPARISON_YEARS)]
    rows: list[dict[str, Any]] = []

    for metric, (label, unit) in METRICS.items():
        stress_mean = float(stress[metric].mean())
        stress_sd = _sample_sd(stress[metric])
        comparison_mean = float(comparison[metric].mean())
        comparison_sd = _sample_sd(comparison[metric])
        comparison_year_means = comparison.groupby("year", sort=True)[metric].mean()
        comparison_year_sd = _sample_sd(comparison_year_means)
        difference = stress_mean - comparison_mean

        rows.append(
            {
                "metric": metric,
                "metric_label": label,
                "unit": unit,
                "stress_year": STRESS_YEAR,
                "stress_2011_mean": stress_mean,
                "stress_2011_sd_across_counties": stress_sd,
                "stress_2011_n_counties": int(stress[metric].count()),
                "comparison_years": ";".join(str(year) for year in COMPARISON_YEARS),
                "comparison_mean": comparison_mean,
                "comparison_sd_across_county_years": comparison_sd,
                "comparison_n_county_years": int(comparison[metric].count()),
                "comparison_sd_across_domain_year_means": comparison_year_sd,
                "comparison_n_years": int(comparison_year_means.count()),
                "difference_2011_minus_comparison": difference,
                "percent_change_2011_vs_comparison": _percent_change(
                    stress_mean, comparison_mean
                ),
                "recommended_text_sd_basis": (
                    "sample SD across county-season values; n=4 for 2011 and "
                    "n=12 for comparison years"
                ),
                "alternative_temporal_sd_basis": (
                    "sample SD across the three comparison-year domain means"
                ),
            }
        )

    return pd.DataFrame(rows)


def build_monthly_trajectory_source(monthly: pd.DataFrame) -> pd.DataFrame:
    """Build monthly mean and sample-SD trajectories without plotting them."""

    rows: list[dict[str, Any]] = []
    comparison = monthly[monthly["year"].isin(COMPARISON_YEARS)].copy()

    for metric, (label, unit) in MONTHLY_METRICS.items():
        # Individual annual domain trajectories: four-county mean and SD.
        annual = (
            monthly.groupby(["year", "month"], as_index=False)[metric]
            .agg(mean="mean", sd=_sample_sd, n="count")
        )
        for record in annual.itertuples(index=False):
            rows.append(
                {
                    "metric": metric,
                    "metric_label": label,
                    "unit": unit,
                    "series": str(int(record.year)),
                    "series_role": (
                        "stress_test" if int(record.year) == STRESS_YEAR else "comparison_year"
                    ),
                    "year": int(record.year),
                    "month": int(record.month),
                    "mean_mm": float(record.mean),
                    "sd_mm": float(record.sd),
                    "n": int(record.n),
                    "sd_basis": "sample SD across four counties",
                }
            )

        # Comparison trajectory: twelve county-year values per full month.
        grouped = (
            comparison.groupby("month", as_index=False)[metric]
            .agg(mean="mean", sd=_sample_sd, n="count")
        )
        for record in grouped.itertuples(index=False):
            rows.append(
                {
                    "metric": metric,
                    "metric_label": label,
                    "unit": unit,
                    "series": "comparison_mean_2015_2017_2023",
                    "series_role": "comparison_aggregate",
                    "year": pd.NA,
                    "month": int(record.month),
                    "mean_mm": float(record.mean),
                    "sd_mm": float(record.sd),
                    "n": int(record.n),
                    "sd_basis": (
                        "sample SD across twelve county-year values "
                        "(four counties x three comparison years)"
                    ),
                }
            )

    result = pd.DataFrame(rows)
    result["year"] = result["year"].astype("Int64")
    return result.sort_values(["metric", "series_role", "series", "month"]).reset_index(
        drop=True
    )


def validate_derived_outputs(
    seasonal_source: pd.DataFrame,
    monthly_source: pd.DataFrame,
    text_statistics: pd.DataFrame,
    *,
    runtime_validation_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a machine-readable validation report for all derived products."""

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

    add(
        "CLIM-001",
        "Phase 3 runtime input validation",
        runtime_validation_summary.get("status"),
        "PASS",
        runtime_validation_summary.get("status") == "PASS",
    )
    add(
        "CLIM-002",
        "Seasonal figure-source rows",
        len(seasonal_source),
        len(METRICS) * 16,
        len(seasonal_source) == len(METRICS) * 16,
    )
    add(
        "CLIM-003",
        "Monthly trajectory rows",
        len(monthly_source),
        len(MONTHLY_METRICS) * (4 * 6 + 6),
        len(monthly_source) == len(MONTHLY_METRICS) * (4 * 6 + 6),
    )
    add(
        "CLIM-004",
        "Text-statistic rows",
        len(text_statistics),
        len(METRICS),
        len(text_statistics) == len(METRICS),
    )
    add(
        "CLIM-005",
        "All reported SDs are finite and non-negative",
        bool(
            np.isfinite(
                text_statistics[
                    [
                        "stress_2011_sd_across_counties",
                        "comparison_sd_across_county_years",
                        "comparison_sd_across_domain_year_means",
                    ]
                ].to_numpy(dtype=float)
            ).all()
        ),
        True,
        bool(
            np.isfinite(
                text_statistics[
                    [
                        "stress_2011_sd_across_counties",
                        "comparison_sd_across_county_years",
                        "comparison_sd_across_domain_year_means",
                    ]
                ].to_numpy(dtype=float)
            ).all()
            and (
                text_statistics[
                    [
                        "stress_2011_sd_across_counties",
                        "comparison_sd_across_county_years",
                        "comparison_sd_across_domain_year_means",
                    ]
                ]
                >= 0
            ).all().all()
        ),
    )
    add(
        "CLIM-006",
        "Standard-deviation convention",
        SD_DDOF,
        1,
        SD_DDOF == 1,
    )
    add(
        "CLIM-007",
        "Figure design remains unchanged",
        "no plotting code; mean columns preserved; SD columns are additive",
        "no figure change",
        True,
    )

    failures = [check for check in checks if check["status"] == "FAIL"]
    return {
        "module": MODULE_NAME,
        "status": "PASS" if not failures else "FAIL",
        "checks": len(checks),
        "passed": len(checks) - len(failures),
        "failed": len(failures),
        "standard_deviation": {
            "type": "sample standard deviation",
            "ddof": SD_DDOF,
            "recommended_text_basis": (
                "county-season values: n=4 for 2011; n=12 for 2015, 2017, and 2023"
            ),
            "alternative_temporal_basis": (
                "three comparison-year domain means: n=3"
            ),
        },
        "figure_change": False,
        "check_results": checks,
    }


def _prepare_outputs(
    output_dir: Path,
    *,
    contract_path: Path,
    overwrite: bool,
) -> ClimateContextOutputs:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    outputs = ClimateContextOutputs(
        figure2_climate_source=out / "figure2_climate_source.csv",
        figure2_monthly_trajectory_source=out
        / "figure2_monthly_trajectory_source.csv",
        climate_context_text_statistics=out / "climate_context_text_statistics.csv",
        climate_context_validation=out / "climate_context_validation.json",
        climate_context_execution_record=out
        / "climate_context_execution_record.json",
    )
    contract = load_source_contract(contract_path)
    allowed = contract.allowed_outputs(MODULE_NAME)
    if allowed != EXPECTED_OUTPUT_NAMES:
        raise ClimateContextError(
            f"Source-contract outputs changed: {allowed}; "
            f"expected={EXPECTED_OUTPUT_NAMES}"
        )
    existing = [path for path in outputs.paths() if path.exists()]
    if existing and not overwrite:
        raise ClimateContextError(
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
    outputs: ClimateContextOutputs,
) -> dict[str, Any]:
    contract = load_source_contract(contract_path)
    authorized_inputs = {
        item.relative_path: {"sha256": item.sha256}
        for item in contract.files_for_module(MODULE_NAME)
    }
    output_records = {
        path.name: _file_record(path)
        for path in outputs.paths()
        if path != outputs.climate_context_execution_record
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
            "warnings": 0,
            "failed": validation["failed"],
        },
        "outputs": output_records,
        "command_template": [
            "python",
            "-m",
            "qstable_awm.climate_context",
            "--phase3-root",
            "PHASE3_CURATED_INPUT_LAYER_FINAL_002",
            "--output-dir",
            "results/climate_context",
            "--contract",
            "config/phase4_source_contract.yml",
        ],
        "statistical_decisions": validation["standard_deviation"],
        "figure_change": False,
        "scientific_boundary": (
            "Meteorological context and deficit screening only; no METRIC ET "
            "validation, irrigation application, pumping, groundwater withdrawal, "
            "or water-source attribution."
        ),
        "determinism_note": (
            "All numerical outputs derive only from the two frozen Phase 3 "
            "climate-context tables."
        ),
    }


def run_climate_context(
    phase3_root: Path,
    output_dir: Path,
    *,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
    overwrite: bool = False,
) -> ClimateContextOutputs:
    """Validate inputs, derive climate tables, and write five deterministic outputs."""

    contract = Path(contract_path).expanduser().resolve()
    outputs = _prepare_outputs(
        output_dir, contract_path=contract, overwrite=overwrite
    )
    monthly, seasonal, runtime_summary = load_climate_inputs(
        phase3_root, contract_path=contract
    )
    _assert_climate_structure(monthly, seasonal)

    seasonal_source = build_figure2_seasonal_source(seasonal)
    monthly_source = build_monthly_trajectory_source(monthly)
    text_statistics = build_text_statistics(seasonal)
    validation = validate_derived_outputs(
        seasonal_source,
        monthly_source,
        text_statistics,
        runtime_validation_summary=runtime_summary,
    )
    if validation["status"] != "PASS":
        failed = [
            check for check in validation["check_results"] if check["status"] == "FAIL"
        ]
        raise ClimateContextError(f"Derived-output validation failed: {failed}")

    seasonal_source.to_csv(outputs.figure2_climate_source, index=False)
    monthly_source.to_csv(outputs.figure2_monthly_trajectory_source, index=False)
    text_statistics.to_csv(outputs.climate_context_text_statistics, index=False)
    outputs.climate_context_validation.write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )

    for path in (
        outputs.figure2_climate_source,
        outputs.figure2_monthly_trajectory_source,
        outputs.climate_context_text_statistics,
        outputs.climate_context_validation,
    ):
        if not path.is_file() or path.stat().st_size <= 0:
            raise ClimateContextError(f"Output was not written correctly: {path}")

    record = _build_execution_record(
        contract_path=contract,
        runtime_validation_summary=runtime_summary,
        validation=validation,
        outputs=outputs,
    )
    outputs.climate_context_execution_record.write_text(
        json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return outputs

def _text_summary(statistics_path: Path) -> str:
    stats = pd.read_csv(statistics_path)
    lines = [
        "Climate context standard deviations (sample SD, ddof=1):",
        "Primary text basis: 2011 n=4 county-season values; comparison n=12 county-season values.",
    ]
    for row in stats.itertuples(index=False):
        lines.append(
            f"- {row.metric_label}: 2011 {row.stress_2011_mean:.1f} ± "
            f"{row.stress_2011_sd_across_counties:.1f} {row.unit}; comparison "
            f"{row.comparison_mean:.1f} ± "
            f"{row.comparison_sd_across_county_years:.1f} {row.unit}."
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Derive Figure 2 climate-source and text-statistics tables."
    )
    parser.add_argument("--phase3-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing climate-context outputs in the selected directory.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        outputs = run_climate_context(
            args.phase3_root,
            args.output_dir,
            contract_path=args.contract,
            overwrite=args.overwrite,
        )
    except (ClimateContextError, FileNotFoundError, ValueError) as exc:
        print(f"CLIMATE_CONTEXT_ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(outputs.as_dict(), indent=2, sort_keys=True))
    print(_text_summary(outputs.climate_context_text_statistics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
