"""Regression and scientific-boundary tests for climate-context analysis.

The frozen Phase 3 climate tables are the exclusive numerical runtime inputs.
Manuscript values are tested only as rounded release-concordance invariants;
the manuscript is never imported or used to calculate a Phase 4 result.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PHASE4_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PHASE4_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qstable_awm import climate_context as cc  # noqa: E402
from qstable_awm.validation import (  # noqa: E402
    RuntimeInputError,
    load_source_contract,
    validate_phase3_root,
)


PHASE3_ROOT = Path(
    os.environ.get(
        "QSTABLE_PHASE3_ROOT",
        PHASE4_ROOT.parent / "PHASE3_CURATED_INPUT_LAYER_FINAL_002",
    )
).expanduser().resolve()
CONTRACT_PATH = PHASE4_ROOT / "config" / "phase4_source_contract.yml"
MODULE_PATH = PHASE4_ROOT / "src" / "qstable_awm" / "climate_context.py"

EXPECTED_MODULE_SHA256 = (
    "7d5e2e9d9fff10a3dfbf857d8ca9d4e9641fa73ad06d720f69d757bc28e4776b"
)
EXPECTED_INPUT_HASHES = {
    cc.MONTHLY_RELATIVE_PATH: (
        "efa851d0322cfc304495e024033e94e4838364bcf85b0af6b1d89d5efc3bc3f6"
    ),
    cc.SEASONAL_RELATIVE_PATH: (
        "7a429a855fce8aee4e8f98dc58fb27ca0bda7a699d04a44174954a68e58a1f72"
    ),
}
EXPECTED_STABLE_OUTPUT_HASHES = {
    "figure2_climate_source.csv": (
        "be3476b9c1cb458076e35d91309316a8d714380a49c426a31903ee7e0a8c4136"
    ),
    "figure2_monthly_trajectory_source.csv": (
        "e1d9c3e9fbfb9e1630818006df322453bd16da75a3bc70904969d3e2654caa1d"
    ),
    "climate_context_text_statistics.csv": (
        "0a5157585ef4adc76209b201bd1de0affa078f202adedd13f19f45d38cf1ce91"
    ),
    "climate_context_validation.json": (
        "e9b8946a7bcfe76077ddfb083c01bb2b7285a029603becd0c5eb965c34edff07"
    ),
}
EXPECTED_TEXT_STATISTICS = {
    "ETr_S4_mm": {
        "stress_mean": 1879.983883,
        "stress_sd": 29.571942,
        "comparison_mean": 1385.224149,
        "comparison_sd": 31.288179,
    },
    "P_S4_mm": {
        "stress_mean": 145.223685,
        "stress_sd": 22.205981,
        "comparison_mean": 493.264611,
        "comparison_sd": 77.461292,
    },
    "Peff_S4_mm": {
        "stress_mean": 134.377828,
        "stress_sd": 18.985534,
        "comparison_mean": 401.425632,
        "comparison_sd": 50.040127,
    },
    "ETr_minus_P_S4_mm": {
        "stress_mean": 1734.760200,
        "stress_sd": 31.467903,
        "comparison_mean": 891.959538,
        "comparison_sd": 99.981712,
    },
    "ETr_minus_Peff_S4_mm": {
        "stress_mean": 1745.606057,
        "stress_sd": 30.344382,
        "comparison_mean": 983.798517,
        "comparison_sd": 73.945732,
    },
}
EXPECTED_COUNTY_PEFF_DEFICIT_ANOMALIES = {
    "DALLAM": 720.1707740220558,
    "HARTLEY": 722.6870213161991,
    "MOORE": 794.0653310284787,
    "OCHILTREE": 810.307033362035,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture(scope="session")
def loaded_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    assert PHASE3_ROOT.is_dir(), (
        "Frozen Phase 3 root not found. Set QSTABLE_PHASE3_ROOT to "
        "PHASE3_CURATED_INPUT_LAYER_FINAL_002."
    )
    return cc.load_climate_inputs(PHASE3_ROOT, contract_path=CONTRACT_PATH)


@pytest.fixture(scope="session")
def generated_outputs(
    tmp_path_factory: pytest.TempPathFactory,
) -> cc.ClimateContextOutputs:
    output_dir = tmp_path_factory.mktemp("climate_context_outputs")
    return cc.run_climate_context(
        PHASE3_ROOT,
        output_dir,
        contract_path=CONTRACT_PATH,
    )


def test_module_and_authorized_input_hashes_are_frozen() -> None:
    assert _sha256(MODULE_PATH) == EXPECTED_MODULE_SHA256
    for relative_path, expected_hash in EXPECTED_INPUT_HASHES.items():
        path = PHASE3_ROOT / relative_path
        assert path.is_file(), relative_path
        assert _sha256(path) == expected_hash


def test_source_contract_authorizes_exact_five_outputs() -> None:
    contract = load_source_contract(CONTRACT_PATH)
    assert contract.allowed_outputs(cc.MODULE_NAME) == cc.EXPECTED_OUTPUT_NAMES
    assert len(contract.files_for_module(cc.MODULE_NAME)) == 2


def test_input_validation_and_structure(loaded_inputs) -> None:
    monthly, seasonal, summary = loaded_inputs
    assert summary["status"] == "PASS"
    assert summary["checks"] == summary["passed"] == 130
    assert summary["warnings"] == summary["failed"] == 0
    assert summary["modules"] == ["climate_context"]

    assert len(monthly) == 96
    assert len(seasonal) == 16
    assert monthly.duplicated(["county", "year", "month"]).sum() == 0
    assert seasonal.duplicated(["county", "year"]).sum() == 0
    assert set(monthly["county"]) == set(cc.EXPECTED_COUNTIES)
    assert set(seasonal["county"]) == set(cc.EXPECTED_COUNTIES)
    assert set(monthly["year"]) == {2011, 2015, 2017, 2023}
    assert set(seasonal["year"]) == {2011, 2015, 2017, 2023}
    assert set(monthly["month"]) == set(cc.EXPECTED_MONTHS)
    cc._assert_climate_structure(monthly, seasonal)


def test_inputs_remain_meteorological_context_only(loaded_inputs) -> None:
    monthly, seasonal, _ = loaded_inputs
    for frame in (monthly, seasonal):
        assert frame["analysis_role"].eq("meteorological_context_only").all()
        assert frame["uses_METRIC_ETf"].eq("No").all()
        assert frame["uses_pumping_records"].eq("No").all()
        assert frame["uses_retained_objects"].eq("No").all()
        assert frame["ETr_source"].eq("IDAHO_EPSCOR/GRIDMET").all()
        assert frame["P_source"].eq("OREGONSTATE/PRISM/ANd").all()
        assert frame["Peff_method"].eq(
            "USDA_SCS_FAO_CROPWAT_simplified_monthly_bounded_0_to_P"
        ).all()


def test_seasonal_accounting_identities(loaded_inputs) -> None:
    _, seasonal, _ = loaded_inputs
    assert np.allclose(
        seasonal["ETr_minus_P_S4_mm"],
        seasonal["ETr_S4_mm"] - seasonal["P_S4_mm"],
        rtol=0,
        atol=1e-5,
    )
    assert np.allclose(
        seasonal["ETr_minus_Peff_S4_mm"],
        seasonal["ETr_S4_mm"] - seasonal["Peff_S4_mm"],
        rtol=0,
        atol=1e-5,
    )
    assert seasonal["Peff_S4_mm"].le(seasonal["P_S4_mm"]).all()


def test_monthly_accounting_identities(loaded_inputs) -> None:
    monthly, _, _ = loaded_inputs
    assert np.allclose(
        monthly["ETr_minus_P_window_mm"],
        monthly["ETr_window_mm"] - monthly["P_window_mm"],
        rtol=0,
        atol=2e-6,
    )
    assert np.allclose(
        monthly["ETr_minus_Peff_window_mm"],
        monthly["ETr_window_mm"] - monthly["Peff_window_mm"],
        rtol=0,
        atol=2e-6,
    )
    assert monthly["Peff_window_mm"].le(monthly["P_window_mm"]).all()


def test_figure2_seasonal_source_structure(loaded_inputs) -> None:
    _, seasonal, _ = loaded_inputs
    source = cc.build_figure2_seasonal_source(seasonal)
    assert len(source) == 80
    assert source.duplicated(["metric", "county", "year"]).sum() == 0
    assert set(source["metric"]) == set(cc.METRICS)
    assert source.groupby("metric").size().eq(16).all()
    assert source["domain_year_n_counties"].eq(4).all()
    assert source["comparison_n_county_years"].eq(12).all()
    assert source["comparison_n_years"].eq(3).all()


def test_monthly_trajectory_structure_and_sd_units(loaded_inputs) -> None:
    monthly, _, _ = loaded_inputs
    source = cc.build_monthly_trajectory_source(monthly)
    assert len(source) == 150
    assert source.duplicated(["metric", "series", "month"]).sum() == 0
    assert set(source["metric"]) == set(cc.MONTHLY_METRICS)
    comparison = source[source["series_role"].eq("comparison_aggregate")]
    annual = source[source["series_role"].ne("comparison_aggregate")]
    assert len(comparison) == 30
    assert comparison["n"].eq(12).all()
    assert comparison["sd_basis"].str.contains("twelve county-year").all()
    assert len(annual) == 120
    assert annual["n"].eq(4).all()
    assert annual["sd_basis"].eq("sample SD across four counties").all()
    assert source["sd_mm"].ge(0).all()


def test_text_statistics_reproduce_expected_values(loaded_inputs) -> None:
    _, seasonal, _ = loaded_inputs
    statistics = cc.build_text_statistics(seasonal).set_index("metric")
    assert len(statistics) == 5
    for metric, expected in EXPECTED_TEXT_STATISTICS.items():
        row = statistics.loc[metric]
        assert row["stress_2011_mean"] == pytest.approx(
            expected["stress_mean"], abs=5e-7
        )
        assert row["stress_2011_sd_across_counties"] == pytest.approx(
            expected["stress_sd"], abs=5e-7
        )
        assert row["comparison_mean"] == pytest.approx(
            expected["comparison_mean"], abs=5e-7
        )
        assert row["comparison_sd_across_county_years"] == pytest.approx(
            expected["comparison_sd"], abs=5e-7
        )
        assert row["stress_2011_n_counties"] == 4
        assert row["comparison_n_county_years"] == 12
        assert row["comparison_n_years"] == 3


def test_sample_sd_is_recalculated_with_ddof_one(loaded_inputs) -> None:
    _, seasonal, _ = loaded_inputs
    statistics = cc.build_text_statistics(seasonal).set_index("metric")
    stress = seasonal[seasonal["year"].eq(2011)]
    comparison = seasonal[seasonal["year"].isin(cc.COMPARISON_YEARS)]
    for metric in cc.METRICS:
        assert statistics.loc[metric, "stress_2011_sd_across_counties"] == pytest.approx(
            stress[metric].std(ddof=1), abs=1e-12
        )
        assert statistics.loc[
            metric, "comparison_sd_across_county_years"
        ] == pytest.approx(comparison[metric].std(ddof=1), abs=1e-12)
        year_means = comparison.groupby("year")[metric].mean()
        assert statistics.loc[
            metric, "comparison_sd_across_domain_year_means"
        ] == pytest.approx(year_means.std(ddof=1), abs=1e-12)


def test_manuscript_section_3_1_rounding_concordance(loaded_inputs) -> None:
    """Lock values reported in the attached manuscript, Section 3.1."""

    _, seasonal, _ = loaded_inputs
    statistics = cc.build_text_statistics(seasonal).set_index("metric")
    rounded = {
        metric: (
            round(float(row["stress_2011_mean"]), 1),
            round(float(row["comparison_mean"]), 1),
        )
        for metric, row in statistics.iterrows()
    }
    assert rounded == {
        "ETr_S4_mm": (1880.0, 1385.2),
        "P_S4_mm": (145.2, 493.3),
        "Peff_S4_mm": (134.4, 401.4),
        "ETr_minus_P_S4_mm": (1734.8, 892.0),
        "ETr_minus_Peff_S4_mm": (1745.6, 983.8),
    }
    peff = statistics.loc["ETr_minus_Peff_S4_mm"]
    assert round(float(peff["difference_2011_minus_comparison"]), 1) == 761.8
    assert round(float(peff["percent_change_2011_vs_comparison"]), 1) == 77.4


def test_manuscript_county_deficit_anomalies_rounding(loaded_inputs) -> None:
    _, seasonal, _ = loaded_inputs
    metric = "ETr_minus_Peff_S4_mm"
    stress = seasonal[seasonal["year"].eq(2011)].set_index("county")[metric]
    comparison = (
        seasonal[seasonal["year"].isin(cc.COMPARISON_YEARS)]
        .groupby("county")[metric]
        .mean()
    )
    observed = (stress - comparison).to_dict()
    assert observed == pytest.approx(EXPECTED_COUNTY_PEFF_DEFICIT_ANOMALIES, abs=1e-12)
    assert {county: round(value, 1) for county, value in observed.items()} == {
        "DALLAM": 720.2,
        "HARTLEY": 722.7,
        "MOORE": 794.1,
        "OCHILTREE": 810.3,
    }


def test_derived_validation_passes_all_seven_checks(loaded_inputs) -> None:
    monthly, seasonal, runtime_summary = loaded_inputs
    report = cc.validate_derived_outputs(
        cc.build_figure2_seasonal_source(seasonal),
        cc.build_monthly_trajectory_source(monthly),
        cc.build_text_statistics(seasonal),
        runtime_validation_summary=runtime_summary,
    )
    assert report["status"] == "PASS"
    assert report["checks"] == report["passed"] == 7
    assert report["failed"] == 0
    assert report["standard_deviation"]["ddof"] == 1
    assert report["figure_change"] is False
    assert {item["check_id"] for item in report["check_results"]} == {
        f"CLIM-{number:03d}" for number in range(1, 8)
    }


def test_run_writes_stable_outputs_and_execution_record(generated_outputs) -> None:
    paths = {path.name: path for path in generated_outputs.paths()}
    assert set(paths) == set(cc.EXPECTED_OUTPUT_NAMES)
    for name, expected_hash in EXPECTED_STABLE_OUTPUT_HASHES.items():
        assert paths[name].is_file()
        assert paths[name].stat().st_size > 0
        assert _sha256(paths[name]) == expected_hash
    assert paths["climate_context_execution_record.json"].stat().st_size > 0


def test_execution_record_reconciles_inputs_outputs_and_contract(generated_outputs) -> None:
    record = json.loads(generated_outputs.climate_context_execution_record.read_text())
    assert record["module"] == cc.MODULE_NAME
    assert record["module_relative_path"] == cc.MODULE_RELATIVE_PATH
    assert record["module_sha256"] == _sha256(MODULE_PATH)
    assert record["contract_sha256"] == _sha256(CONTRACT_PATH)
    assert record["phase3_package"] == "PHASE3_CURATED_INPUT_LAYER_FINAL_002"
    assert record["authorized_inputs"] == {
        path: {"sha256": digest} for path, digest in EXPECTED_INPUT_HASHES.items()
    }
    assert record["runtime_input_validation"]["checks"] == 130
    assert record["runtime_input_validation"]["passed"] == 130
    assert record["derived_output_validation"] == {
        "status": "PASS",
        "checks": 7,
        "passed": 7,
        "warnings": 0,
        "failed": 0,
    }
    assert record["figure_change"] is False
    assert "no METRIC ET validation" in record["scientific_boundary"]
    for name, item in record["outputs"].items():
        path = generated_outputs.climate_context_execution_record.parent / name
        assert item["size_bytes"] == path.stat().st_size
        assert item["sha256"] == _sha256(path)


def test_second_execution_is_byte_deterministic(generated_outputs, tmp_path: Path) -> None:
    second = cc.run_climate_context(
        PHASE3_ROOT,
        tmp_path / "second_run",
        contract_path=CONTRACT_PATH,
    )
    first_hashes = {path.name: _sha256(path) for path in generated_outputs.paths()}
    second_hashes = {path.name: _sha256(path) for path in second.paths()}
    assert second_hashes == first_hashes


def test_existing_outputs_require_explicit_overwrite(generated_outputs) -> None:
    output_dir = generated_outputs.climate_context_validation.parent
    with pytest.raises(cc.ClimateContextError, match="Refusing to overwrite"):
        cc.run_climate_context(
            PHASE3_ROOT,
            output_dir,
            contract_path=CONTRACT_PATH,
        )


def test_explicit_overwrite_is_byte_deterministic(generated_outputs) -> None:
    output_dir = generated_outputs.climate_context_validation.parent
    before = {path.name: _sha256(path) for path in generated_outputs.paths()}
    overwritten = cc.run_climate_context(
        PHASE3_ROOT,
        output_dir,
        contract_path=CONTRACT_PATH,
        overwrite=True,
    )
    after = {path.name: _sha256(path) for path in overwritten.paths()}
    assert after == before


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate_month", "Monthly county-year-month keys are not unique"),
        ("duplicate_season", "Seasonal county-year keys are not unique"),
        ("county", "Unexpected counties"),
        ("year", "Unexpected years"),
        ("month", "Unexpected months"),
        ("seasonal_identity", "Seasonal accounting identity failed"),
        ("monthly_identity", "Monthly accounting identity failed"),
    ],
)
def test_malformed_climate_tables_are_rejected(
    loaded_inputs,
    mutation: str,
    message: str,
) -> None:
    monthly, seasonal, _ = loaded_inputs
    monthly_copy = monthly.copy(deep=True)
    seasonal_copy = seasonal.copy(deep=True)

    if mutation == "duplicate_month":
        monthly_copy.loc[1, ["county", "year", "month"]] = monthly_copy.loc[
            0, ["county", "year", "month"]
        ].to_numpy()
    elif mutation == "duplicate_season":
        seasonal_copy.loc[1, ["county", "year"]] = seasonal_copy.loc[
            0, ["county", "year"]
        ].to_numpy()
    elif mutation == "county":
        seasonal_copy.loc[0, "county"] = "UNAUTHORIZED"
    elif mutation == "year":
        seasonal_copy.loc[seasonal_copy["year"].eq(2023), "year"] = 2024
    elif mutation == "month":
        monthly_copy.loc[0, "month"] = 10
    elif mutation == "seasonal_identity":
        seasonal_copy.loc[0, "ETr_minus_Peff_S4_mm"] += 1.0
    elif mutation == "monthly_identity":
        monthly_copy.loc[0, "ETr_minus_P_window_mm"] += 1.0
    else:  # pragma: no cover
        raise AssertionError(mutation)

    with pytest.raises(cc.ClimateContextError, match=message):
        cc._assert_climate_structure(monthly_copy, seasonal_copy)


def _build_minimal_phase3_copy(tmp_path: Path) -> Path:
    root = tmp_path / "PHASE3_CURATED_INPUT_LAYER_FINAL_002"
    for relative_path in (
        "VERSION",
        "package_file_manifest.csv",
        "provenance/phase3_validation_summary.json",
        cc.MONTHLY_RELATIVE_PATH,
        cc.SEASONAL_RELATIVE_PATH,
    ):
        source = PHASE3_ROOT / relative_path
        target = root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return root


def test_altered_input_hash_is_rejected(tmp_path: Path) -> None:
    root = _build_minimal_phase3_copy(tmp_path)
    path = root / cc.MONTHLY_RELATIVE_PATH
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(RuntimeInputError, match="runtime-input validation failed"):
        cc.load_climate_inputs(root, contract_path=CONTRACT_PATH)


def test_cli_success_and_overwrite_refusal(
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
    assert cc.main(args) == 0
    success = capsys.readouterr()
    assert '"climate_context_execution_record"' in success.out
    assert "2011 1880.0 ± 29.6 mm" in success.out
    assert success.err == ""

    assert cc.main(args) == 2
    failure = capsys.readouterr()
    assert "CLIMATE_CONTEXT_ERROR" in failure.err
    assert "Refusing to overwrite" in failure.err


def test_complete_phase3_runtime_validation_remains_clean() -> None:
    report = validate_phase3_root(PHASE3_ROOT, contract_path=CONTRACT_PATH)
    assert report.summary()["status"] == "PASS"
    assert report.summary()["checks"] == 1708
    assert report.summary()["passed"] == 1708
    assert report.summary()["warnings"] == 0
    assert report.summary()["failed"] == 0


def test_source_contains_no_notebook_or_prohibited_runtime_dependency() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    prohibited = (
        ".ipynb",
        "AWM_figures",
        "V46D",
        "V48",
        "V49",
        "/scratch/",
        "METRIC_daily_integrated_ET_over_season",
    )
    for token in prohibited:
        assert token not in source
    assert "validate_module_inputs" in source
    assert cc.MONTHLY_RELATIVE_PATH in source
    assert cc.SEASONAL_RELATIVE_PATH in source
