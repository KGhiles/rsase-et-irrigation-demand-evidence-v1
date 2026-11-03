"""Physical-coherence summaries for manuscript Figure 3.

This module reads only the four Phase 3 inputs authorized for the
``physical_coherence`` domain. It recreates the source tables used by Figure 3
without re-running METRIC and without recalculating full-resolution area
statistics from the compact display raster.

The compact GeoTIFF is opened only to confirm its eight-band visualization
contract. All scene statistics, mechanism fractions, and connected-component
areas come from the frozen V46C tables.

Sample standard deviations requested during coauthor review are added to a
separate text-statistics table. They are descriptive spread across the eight
targeted active-canopy 2011 scenes (``ddof=1``), not uncertainty estimates and
not error bars added to Figure 3. The published figure design remains
unchanged.

Scientific boundary
-------------------
The outputs establish internal physical coherence only. They do not validate
ET magnitude, create corrected ET, estimate irrigation application, infer
pumping, or identify a groundwater source.
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
import rasterio

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


MODULE_NAME = "physical_coherence"
MODULE_RELATIVE_PATH = "src/qstable_awm/physical_coherence.py"
EXPECTED_OUTPUT_NAMES = (
    "figure3_scene_fingerprint_source.csv",
    "figure3_mechanism_source.csv",
    "figure3_spatial_coherence_source.csv",
    "physical_coherence_text_statistics.csv",
    "physical_coherence_validation.json",
    "physical_coherence_execution_record.json",
)
FINGERPRINT_RELATIVE_PATH = (
    "data/derived/physical_coherence/active_scene_fingerprint.csv"
)
SPATIAL_RELATIVE_PATH = (
    "data/derived/physical_coherence/spatial_coherence_figure_source.csv"
)
SCENE_SUMMARY_RELATIVE_PATH = (
    "data/derived/physical_coherence/physical_coherence_scene_summary.csv"
)
RASTER_RELATIVE_PATH = (
    "data/derived/physical_coherence/representative_scene_plot_ready.tif"
)

ACTIVE_CLASS = "active_canopy_energy_partitioning_mechanism"
AMBIGUOUS_CLASS = "crop_mask_or_low_canopy_artifact_candidate"
CONTROL_CLASS = "low_or_no_high_ETf_signal"
OCTOBER_CLASS = "october_diagnostic_only_not_correction_evidence"
EXPECTED_ACTIVE_SCENES = 8
SD_DDOF = 1
INTERPRETATION_BOUNDARY = (
    "Internal physical-coherence evidence only. No ET validation, corrected ET, "
    "irrigation application, pumping inference, groundwater withdrawal validation, "
    "or groundwater-source attribution."
)

METRIC_SPECS: Mapping[str, dict[str, Any]] = {
    "ETf p50": {
        "column": "ETf_p50",
        "criterion": "≥ 0.85",
        "threshold": 0.85,
        "operator": "ge",
        "label": "ETrF p50",
        "unit": "dimensionless",
    },
    "NDVI p50": {
        "column": "NDVI_p50",
        "criterion": "≥ 0.50",
        "threshold": 0.50,
        "operator": "ge",
        "label": "NDVI p50",
        "unit": "dimensionless",
    },
    "LAI p50": {
        "column": "LAI_p50",
        "criterion": "≥ 2.00",
        "threshold": 2.00,
        "operator": "ge",
        "label": "LAI p50",
        "unit": "m2 m-2",
    },
    "H p50": {
        "column": "H_p50",
        "criterion": "≤ 25 W m⁻²",
        "threshold": 25.0,
        "operator": "le",
        "label": "Sensible heat flux p50",
        "unit": "W m-2",
    },
    "LE/Rn p50": {
        "column": "LE_Rn_ratio_p50",
        "criterion": "≥ 0.90",
        "threshold": 0.90,
        "operator": "ge",
        "label": "LE/Rn p50",
        "unit": "dimensionless",
    },
    "Ts-position p50": {
        "column": "Ts_position_between_anchors_p50",
        "criterion": "≤ 0.15",
        "threshold": 0.15,
        "operator": "le",
        "label": "Thermal position p50",
        "unit": "dimensionless",
    },
}

CLASS_ORDER = {
    ACTIVE_CLASS: 0,
    AMBIGUOUS_CLASS: 1,
    CONTROL_CLASS: 2,
}
CLASS_LABELS = {
    ACTIVE_CLASS: "active-canopy mechanism",
    AMBIGUOUS_CLASS: "early weak-canopy / ambiguous",
    CONTROL_CLASS: "control: low/no large ETrF",
}


class PhysicalCoherenceError(RuntimeError):
    """Raised when physical-coherence inputs or outputs violate the freeze."""


@dataclass(frozen=True, slots=True)
class PhysicalCoherenceOutputs:
    """Paths written by :func:`run_physical_coherence`."""

    figure3_scene_fingerprint_source: Path
    figure3_mechanism_source: Path
    figure3_spatial_coherence_source: Path
    physical_coherence_text_statistics: Path
    physical_coherence_validation: Path
    physical_coherence_execution_record: Path

    def as_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}

    def paths(self) -> tuple[Path, ...]:
        return tuple(Path(value) for value in asdict(self).values())


def _sample_sd(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if len(numeric) < 2:
        return float("nan")
    return float(numeric.std(ddof=SD_DDOF))


def _passes(values: pd.Series, operator: str, threshold: float) -> pd.Series:
    if operator == "ge":
        return values >= threshold
    if operator == "le":
        return values <= threshold
    raise PhysicalCoherenceError(f"Unsupported threshold operator: {operator}")


def load_physical_coherence_inputs(
    phase3_root: Path,
    *,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path, dict[str, Any]]:
    """Validate and load the four authorized physical-coherence inputs."""

    root = Path(phase3_root).expanduser().resolve()
    paths = (
        root / FINGERPRINT_RELATIVE_PATH,
        root / SPATIAL_RELATIVE_PATH,
        root / SCENE_SUMMARY_RELATIVE_PATH,
        root / RASTER_RELATIVE_PATH,
    )
    report = validate_module_inputs(
        root,
        MODULE_NAME,
        contract_path=contract_path,
        supplied_paths=paths,
        raise_on_error=True,
    )
    fingerprint = pd.read_csv(paths[0])
    spatial = pd.read_csv(paths[1])
    scenes = pd.read_csv(paths[2])
    return fingerprint, spatial, scenes, paths[3], report.summary()


def _assert_input_structure(
    fingerprint: pd.DataFrame,
    spatial: pd.DataFrame,
    scenes: pd.DataFrame,
    raster_path: Path,
) -> None:
    """Apply analysis-specific checks beyond the generic Phase 4 schemas."""

    active = scenes[scenes["raster_mechanism_class"] == ACTIVE_CLASS]
    ambiguous = scenes[scenes["raster_mechanism_class"] == AMBIGUOUS_CLASS]
    controls = scenes[scenes["raster_mechanism_class"] == CONTROL_CLASS]
    october = scenes[scenes["raster_mechanism_class"] == OCTOBER_CLASS]

    counts = {
        "active": len(active),
        "ambiguous": len(ambiguous),
        "control": len(controls),
        "october": len(october),
    }
    expected = {"active": 8, "ambiguous": 2, "control": 2, "october": 1}
    if counts != expected:
        raise PhysicalCoherenceError(
            f"Unexpected scene-class counts: {counts}; expected={expected}"
        )
    if not (active["year"] == 2011).all():
        raise PhysicalCoherenceError("All active-canopy mechanism scenes must be 2011")
    if set(fingerprint["metric"]) != set(METRIC_SPECS):
        raise PhysicalCoherenceError(
            "Fingerprint metrics do not match the frozen six-metric definition"
        )
    if set(spatial["filename"]) != set(active["filename"]):
        raise PhysicalCoherenceError(
            "Spatial-coherence rows do not match the eight active-scene filenames"
        )
    if len(spatial) != EXPECTED_ACTIVE_SCENES:
        raise PhysicalCoherenceError("Expected eight spatial-coherence rows")

    with rasterio.open(raster_path) as ds:
        expected_bands = (
            "ETrF",
            "NDVI",
            "LAI",
            "H_W_m2",
            "LE_over_Rn",
            "Ts_position",
            "accepted_corn_mask",
            "mechanism_mask",
        )
        descriptions = tuple(ds.descriptions)
        if ds.count != 8 or descriptions != expected_bands:
            raise PhysicalCoherenceError(
                f"Compact raster contract changed: count={ds.count}, bands={descriptions}"
            )
        if not math.isclose(abs(ds.transform.a), 90.0, abs_tol=1e-9):
            raise PhysicalCoherenceError(
                f"Compact raster resolution changed: {abs(ds.transform.a)} m"
            )


def build_scene_fingerprint_source(
    frozen_fingerprint: pd.DataFrame,
    scenes: pd.DataFrame,
) -> pd.DataFrame:
    """Recompute panel-B source values and add descriptive scene spread.

    Existing Figure 3 columns are retained. ``mean`` and ``sample_sd`` are
    additive text-reporting fields and do not alter the figure.
    """

    active = scenes[scenes["raster_mechanism_class"] == ACTIVE_CLASS].copy()
    rows: list[dict[str, Any]] = []

    frozen_by_metric = frozen_fingerprint.set_index("metric")
    for metric, spec in METRIC_SPECS.items():
        values = pd.to_numeric(active[spec["column"]], errors="raise")
        passed = _passes(values, spec["operator"], float(spec["threshold"]))
        recomputed = {
            "metric": metric,
            "display_metric": spec["label"],
            "criterion": spec["criterion"],
            "pass_pct": 100.0 * float(passed.mean()),
            "n_pass": int(passed.sum()),
            "n_total": int(len(values)),
            "median": float(values.median()),
            "mean": float(values.mean()),
            "sample_sd": _sample_sd(values),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "unit": spec["unit"],
            "sd_basis": "sample SD across eight active-canopy 2011 scenes",
        }
        frozen = frozen_by_metric.loc[metric]
        comparisons = {
            "criterion": str(frozen["criterion"]) == recomputed["criterion"],
            "pass_pct": math.isclose(
                float(frozen["pass_pct"]), recomputed["pass_pct"], abs_tol=1e-12
            ),
            "n_pass": int(frozen["n_pass"]) == recomputed["n_pass"],
            "n_total": int(frozen["n_total"]) == recomputed["n_total"],
            "median": math.isclose(
                float(frozen["median"]), recomputed["median"], abs_tol=1e-12
            ),
        }
        if not all(comparisons.values()):
            raise PhysicalCoherenceError(
                f"Recomputed fingerprint differs from frozen source for {metric}: "
                f"{comparisons}"
            )
        rows.append(recomputed)

    return pd.DataFrame(rows)


def build_mechanism_source(scenes: pd.DataFrame) -> pd.DataFrame:
    """Return the 12-scene panel-C source table with explicit plot classes."""

    result = scenes[
        scenes["raster_mechanism_class"].isin(CLASS_ORDER)
    ].copy()
    result["plot_class_order"] = result["raster_mechanism_class"].map(CLASS_ORDER)
    result["plot_class_label"] = result["raster_mechanism_class"].map(CLASS_LABELS)
    result["is_active_2011_mechanism"] = (
        result["raster_mechanism_class"] == ACTIVE_CLASS
    )
    result["mechanism_fraction_pct"] = (
        100.0 * result["active_canopy_high_mechanism_frac"]
    )
    result["interpretation_boundary"] = (
        "physical-coherence diagnostic only; not ET validation or pumping inference"
    )
    columns = [
        "county",
        "year",
        "month",
        "date",
        "scene_id",
        "filename",
        "scene_label",
        "short_label",
        "comparison_group",
        "raster_mechanism_class",
        "plot_class_order",
        "plot_class_label",
        "is_active_2011_mechanism",
        "mechanism_priority_score",
        "high_ETf_frac",
        "active_canopy_high_mechanism_frac",
        "mechanism_fraction_pct",
        "active_canopy_high_mechanism_area_ha",
        "early_artifact_candidate_frac",
        "interpretation_boundary",
    ]
    return result[columns].sort_values(
        ["plot_class_order", "active_canopy_high_mechanism_frac"],
        ascending=[True, False],
    ).reset_index(drop=True)


def build_spatial_coherence_source(spatial: pd.DataFrame) -> pd.DataFrame:
    """Return the frozen panel-D component table with additive percentages."""

    result = spatial.copy()
    result["largest_component_share_pct"] = (
        100.0 * result["largest_component_frac_of_mask"]
    )
    result["interpretation_boundary"] = (
        "component areas are frozen V46C full-resolution results; compact raster not used"
    )
    return result.sort_values("total_area_ha", ascending=True).reset_index(drop=True)


def build_text_statistics(
    fingerprint_source: pd.DataFrame,
    scenes: pd.DataFrame,
    spatial_source: pd.DataFrame,
) -> pd.DataFrame:
    """Create text-ready means, medians, and sample SDs for active scenes."""

    rows: list[dict[str, Any]] = []
    for row in fingerprint_source.itertuples(index=False):
        rows.append(
            {
                "statistic_group": "active_scene_fingerprint",
                "statistic": f"{row.metric}_scene_dispersion",
                "metric": row.metric,
                "metric_label": row.display_metric,
                "unit": row.unit,
                "observational_unit": "targeted active-canopy scene",
                "n": row.n_total,
                "ddof": SD_DDOF,
                "mean": row.mean,
                "median": row.median,
                "sample_sd": row.sample_sd,
                "standard_deviation": row.sample_sd,
                "minimum": row.minimum,
                "maximum": row.maximum,
                "pass_count": row.n_pass,
                "pass_pct": row.pass_pct,
                "criterion": row.criterion,
                "sd_basis": row.sd_basis,
                "interpretation_boundary": INTERPRETATION_BOUNDARY,
            }
        )

    active = scenes[scenes["raster_mechanism_class"] == ACTIVE_CLASS]
    extra_metrics = {
        "active_canopy_high_mechanism_frac": (
            "Active-canopy mechanism fraction",
            "fraction",
        ),
        "active_canopy_high_mechanism_area_ha": (
            "Active-canopy mechanism area",
            "ha",
        ),
        "high_ETf_frac": ("Large-ETrF fraction", "fraction"),
    }
    for column, (label, unit) in extra_metrics.items():
        values = pd.to_numeric(active[column], errors="raise")
        rows.append(
            {
                "statistic_group": "active_scene_mechanism",
                "statistic": f"{column}_scene_dispersion",
                "metric": column,
                "metric_label": label,
                "unit": unit,
                "observational_unit": "targeted active-canopy scene",
                "n": len(values),
                "ddof": SD_DDOF,
                "mean": float(values.mean()),
                "median": float(values.median()),
                "sample_sd": _sample_sd(values),
                "standard_deviation": _sample_sd(values),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "pass_count": pd.NA,
                "pass_pct": pd.NA,
                "criterion": "descriptive",
                "sd_basis": "sample SD across eight active-canopy 2011 scenes",
                "interpretation_boundary": INTERPRETATION_BOUNDARY,
            }
        )

    spatial_metrics = {
        "total_area_ha": ("Total active-mechanism area", "ha"),
        "largest_component_area_ha": ("Largest connected-component area", "ha"),
        "largest_component_frac_of_mask": (
            "Largest component share of mechanism mask",
            "fraction",
        ),
        "component_count": ("Connected-component count", "count"),
    }
    for column, (label, unit) in spatial_metrics.items():
        values = pd.to_numeric(spatial_source[column], errors="raise")
        rows.append(
            {
                "statistic_group": "active_scene_spatial_coherence",
                "statistic": f"{column}_scene_dispersion",
                "metric": column,
                "metric_label": label,
                "unit": unit,
                "observational_unit": "targeted active-canopy scene",
                "n": len(values),
                "ddof": SD_DDOF,
                "mean": float(values.mean()),
                "median": float(values.median()),
                "sample_sd": _sample_sd(values),
                "standard_deviation": _sample_sd(values),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "pass_count": pd.NA,
                "pass_pct": pd.NA,
                "criterion": "descriptive",
                "sd_basis": "sample SD across eight active-canopy 2011 scenes",
                "interpretation_boundary": INTERPRETATION_BOUNDARY,
            }
        )

    return pd.DataFrame(rows)


def validate_derived_outputs(
    fingerprint_source: pd.DataFrame,
    mechanism_source: pd.DataFrame,
    spatial_source: pd.DataFrame,
    text_statistics: pd.DataFrame,
    *,
    runtime_validation_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a machine-readable validation report for derived products."""

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

    add(
        "PHYS-001",
        "Phase 3 runtime input validation",
        runtime_validation_summary.get("status"),
        "PASS",
        runtime_validation_summary.get("status") == "PASS",
    )
    add(
        "PHYS-002",
        "Scene-fingerprint rows",
        len(fingerprint_source),
        6,
        len(fingerprint_source) == 6,
    )
    add(
        "PHYS-003",
        "Mechanism-source rows",
        len(mechanism_source),
        12,
        len(mechanism_source) == 12,
    )
    add(
        "PHYS-004",
        "Active mechanism rows",
        int(mechanism_source["is_active_2011_mechanism"].sum()),
        8,
        int(mechanism_source["is_active_2011_mechanism"].sum()) == 8,
    )
    add(
        "PHYS-005",
        "Spatial-source rows",
        len(spatial_source),
        8,
        len(spatial_source) == 8,
    )
    add(
        "PHYS-006",
        "Text-statistic rows",
        len(text_statistics),
        13,
        len(text_statistics) == 13,
    )
    finite_sd = np.isfinite(text_statistics["sample_sd"].to_numpy(dtype=float)).all()
    nonnegative_sd = (text_statistics["sample_sd"] >= 0).all()
    add(
        "PHYS-007",
        "All sample SD values finite and non-negative",
        bool(finite_sd and nonnegative_sd),
        True,
        bool(finite_sd and nonnegative_sd),
    )
    add(
        "PHYS-008",
        "Standard-deviation convention",
        SD_DDOF,
        1,
        SD_DDOF == 1,
    )
    add(
        "PHYS-009",
        "Figure 3 design remains unchanged",
        "no plotting code; SD values stored separately",
        "no figure change",
        True,
    )
    boundaries_ok = mechanism_source["interpretation_boundary"].str.contains(
        "not ET validation", regex=False
    ).all()
    add(
        "PHYS-010",
        "Scientific interpretation boundary",
        bool(boundaries_ok),
        True,
        bool(boundaries_ok),
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
            "basis": "eight targeted active-canopy 2011 scenes",
            "interpretation": "descriptive scene-to-scene spread, not product uncertainty",
        },
        "figure_change": False,
        "scientific_boundary": (
            "internal physical-coherence evidence only; no ET validation, corrected ET, "
            "irrigation, pumping, or groundwater-source inference"
        ),
        "check_results": checks,
    }


def _prepare_outputs(
    output_dir: Path,
    *,
    contract_path: Path,
    overwrite: bool,
) -> PhysicalCoherenceOutputs:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    outputs = PhysicalCoherenceOutputs(
        figure3_scene_fingerprint_source=out
        / "figure3_scene_fingerprint_source.csv",
        figure3_mechanism_source=out / "figure3_mechanism_source.csv",
        figure3_spatial_coherence_source=out
        / "figure3_spatial_coherence_source.csv",
        physical_coherence_text_statistics=out
        / "physical_coherence_text_statistics.csv",
        physical_coherence_validation=out / "physical_coherence_validation.json",
        physical_coherence_execution_record=out
        / "physical_coherence_execution_record.json",
    )
    contract = load_source_contract(contract_path)
    allowed = contract.allowed_outputs(MODULE_NAME)
    if allowed != EXPECTED_OUTPUT_NAMES:
        raise PhysicalCoherenceError(
            f"Source-contract outputs changed: {allowed}; "
            f"expected={EXPECTED_OUTPUT_NAMES}"
        )
    existing = [path for path in outputs.paths() if path.exists()]
    if existing and not overwrite:
        raise PhysicalCoherenceError(
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
    outputs: PhysicalCoherenceOutputs,
) -> dict[str, Any]:
    contract = load_source_contract(contract_path)
    authorized_inputs = {
        item.relative_path: {"sha256": item.sha256}
        for item in contract.files_for_module(MODULE_NAME)
    }
    output_records = {
        path.name: _file_record(path)
        for path in outputs.paths()
        if path != outputs.physical_coherence_execution_record
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
            "qstable_awm.physical_coherence",
            "--phase3-root",
            "PHASE3_CURATED_INPUT_LAYER_FINAL_002",
            "--output-dir",
            "results/physical_coherence",
            "--contract",
            "config/phase4_source_contract.yml",
        ],
        "statistical_decisions": validation["standard_deviation"],
        "figure_change": False,
        "compact_raster_boundary": (
            "The compact eight-band raster is opened only for metadata and "
            "visualization-contract validation; no authoritative area or "
            "connected-component statistic is derived from it."
        ),
        "scientific_boundary": validation["scientific_boundary"],
        "determinism_note": (
            "All numerical outputs derive only from the frozen V46C Phase 3 "
            "tables; the compact raster contributes metadata validation only."
        ),
    }


def run_physical_coherence(
    phase3_root: Path,
    output_dir: Path,
    *,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
    overwrite: bool = False,
) -> PhysicalCoherenceOutputs:
    """Validate inputs and write six deterministic Figure 3 outputs."""

    contract = Path(contract_path).expanduser().resolve()
    outputs = _prepare_outputs(
        output_dir, contract_path=contract, overwrite=overwrite
    )
    frozen_fingerprint, spatial, scenes, raster_path, runtime_summary = (
        load_physical_coherence_inputs(phase3_root, contract_path=contract)
    )
    _assert_input_structure(frozen_fingerprint, spatial, scenes, raster_path)

    fingerprint_source = build_scene_fingerprint_source(
        frozen_fingerprint, scenes
    )
    mechanism_source = build_mechanism_source(scenes)
    spatial_source = build_spatial_coherence_source(spatial)
    text_statistics = build_text_statistics(
        fingerprint_source, scenes, spatial_source
    )
    validation = validate_derived_outputs(
        fingerprint_source,
        mechanism_source,
        spatial_source,
        text_statistics,
        runtime_validation_summary=runtime_summary,
    )
    if validation["status"] != "PASS":
        failed = [
            check for check in validation["check_results"]
            if check["status"] == "FAIL"
        ]
        raise PhysicalCoherenceError(f"Derived-output validation failed: {failed}")

    fingerprint_source.to_csv(outputs.figure3_scene_fingerprint_source, index=False)
    mechanism_source.to_csv(outputs.figure3_mechanism_source, index=False)
    spatial_source.to_csv(outputs.figure3_spatial_coherence_source, index=False)
    text_statistics.to_csv(outputs.physical_coherence_text_statistics, index=False)
    outputs.physical_coherence_validation.write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )

    for path in (
        outputs.figure3_scene_fingerprint_source,
        outputs.figure3_mechanism_source,
        outputs.figure3_spatial_coherence_source,
        outputs.physical_coherence_text_statistics,
        outputs.physical_coherence_validation,
    ):
        if not path.is_file() or path.stat().st_size <= 0:
            raise PhysicalCoherenceError(f"Output was not written correctly: {path}")

    record = _build_execution_record(
        contract_path=contract,
        runtime_validation_summary=runtime_summary,
        validation=validation,
        outputs=outputs,
    )
    outputs.physical_coherence_execution_record.write_text(
        json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return outputs

def _text_summary(statistics_path: Path) -> str:
    stats = pd.read_csv(statistics_path)
    fingerprint = stats[stats["statistic_group"] == "active_scene_fingerprint"]
    lines = [
        "Physical-coherence scene spread (sample SD, ddof=1; n=8 active scenes):"
    ]
    for row in fingerprint.itertuples(index=False):
        lines.append(
            f"- {row.metric_label}: median {row.median:.3f}; mean {row.mean:.3f} "
            f"± {row.sample_sd:.3f} {row.unit}; {int(row.pass_count)}/{int(row.n)} pass."
        )
    lines.append("Figure 3 design changed: no.")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Derive Figure 3 physical-coherence source tables."
    )
    parser.add_argument("--phase3-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing physical-coherence outputs in the selected directory.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        outputs = run_physical_coherence(
            args.phase3_root,
            args.output_dir,
            contract_path=args.contract,
            overwrite=args.overwrite,
        )
    except (PhysicalCoherenceError, FileNotFoundError, ValueError) as exc:
        print(f"PHYSICAL_COHERENCE_ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(outputs.as_dict(), indent=2, sort_keys=True))
    print(_text_summary(outputs.physical_coherence_text_statistics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
