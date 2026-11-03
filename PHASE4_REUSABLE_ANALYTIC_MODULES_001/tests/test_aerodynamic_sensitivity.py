"""Regression and scientific-boundary tests for aerodynamic sensitivity.

The frozen V42.5-derived Phase 3 tables are the exclusive analytical inputs.
Historical notebooks remain provenance-only. Manuscript Table 3 values are
checked only as rounded release-concordance invariants and never enter the
production calculation.
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

from qstable_awm import aerodynamic_sensitivity as aero  # noqa: E402
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
MODULE_PATH = PHASE4_ROOT / "src" / "qstable_awm" / "aerodynamic_sensitivity.py"

EXPECTED_MODULE_SHA256 = (
    "c68a14e337a9f8ffb0de234b74458d9433097a0ff21faf17a127ecc0dd14860d"
)
EXPECTED_INPUT_HASHES = {
    aero.SCOPE_SUMMARY_PATH: (
        "bb28825af4cfd3bcf0bc4af7a075e2f9dd6d356fac0a6c6190c4d21731119f84"
    ),
    aero.ETRF_ENVELOPE_PATH: (
        "8580aafa4f93fb792ad9a44e138b6c59e83e5c671a3a6049900568313e432485"
    ),
    aero.H_ENVELOPE_PATH: (
        "be62e6b298f5f1ba7972004185beee585a3d14ebaa22b364b19f3a75ccb67d39"
    ),
    aero.PEAK_PATH: (
        "5622087ee8d292d45444a1ae59df4533afbdd10cf8d3672d1a7b1f8eeb040779"
    ),
    aero.TABLE3_PATH: (
        "13bf81ca894f46c96cc03f1eeaf6ba4392d3378dd945857fd1e74417a53dbb44"
    ),
}
EXPECTED_STABLE_OUTPUT_HASHES = {
    "figure4_aerodynamic_source.csv": (
        "6c65e1697decbd994c20c5bc5d022b56acd2f3ec4ce6ea3f89f268a7edcd221e"
    ),
    "figure4_peak_sensitivity_source.csv": (
        "2a10621e26d86c2d8e9c8078783b17d342061d8f4213386dfece6c6a7811f0c6"
    ),
    "table3_aerodynamic_summary.csv": (
        "2e9ee99daa801581404d61d8d004066bd5bb343fd731bd79131ed6136bf635d6"
    ),
    "aerodynamic_sensitivity_text_statistics.csv": (
        "c555ee0afe109f4ea38519a2f71721fbe42f5d4485dffcfc29928e95ca158386"
    ),
    "aerodynamic_sensitivity_validation.json": (
        "8073e5ed61dcc888c8d03637b1806c6450e23b44279c955fb818df3c645161df"
    ),
}
EXPECTED_OUTPUT_NAMES = set(EXPECTED_STABLE_OUTPUT_HASHES) | {
    "aerodynamic_sensitivity_execution_record.json"
}
EXPECTED_ABSOLUTE_STATS = {
    "all_crops_all_months_nonBASE": {
        "n": 288,
        "drah2": (120.25208974625207, 77.40773112791085),
        "dH": (0.5816630396680894, 0.5155644909809513),
        "dETf": (0.0012162247599671655, 0.0010451012539120076),
    },
    "corn_all_months_nonBASE": {
        "n": 144,
        "drah2": (132.02918406611334, 86.99002644002445),
        "dH": (0.5532903067833238, 0.4509869442160453),
        "dETf": (0.0011594107660396545, 0.0009321987227739699),
    },
    "corn_peak_JulAug_nonBASE": {
        "n": 48,
        "drah2": (153.65993755058275, 86.03870746353718),
        "dH": (0.6575753825853186, 0.5583938728022148),
        "dETf": (0.0012212133015090147, 0.0010536907207321594),
    },
}
EXPECTED_TABLE3_ROUNDED = {
    "all_crops_all_months_nonBASE": (288, 547.3, 93.1, 0.0078, 0.0009, 4.16, 0.42),
    "corn_all_months_nonBASE": (144, 547.3, 102.2, 0.0061, 0.0009, 3.20, 0.41),
    "corn_peak_JulAug_nonBASE": (48, 392.0, 119.7, 0.0061, 0.0009, 3.20, 0.46),
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
    return aero.load_aerodynamic_inputs(
        PHASE3_ROOT, contract_path=CONTRACT_PATH
    )


@pytest.fixture(scope="session")
def derived_tables(loaded_inputs):
    tables, runtime = loaded_inputs
    aero._assert_structure(tables)
    figure = aero.build_figure4_aerodynamic_source(
        tables[aero.TABLE3_PATH],
        tables[aero.ETRF_ENVELOPE_PATH],
        tables[aero.H_ENVELOPE_PATH],
    )
    peak = aero.build_peak_sensitivity_source(tables[aero.PEAK_PATH])
    statistics = aero.build_text_statistics(tables[aero.TABLE3_PATH])
    summary = aero.build_table3_summary(
        tables[aero.SCOPE_SUMMARY_PATH], statistics
    )
    return figure, peak, summary, statistics, runtime, tables


@pytest.fixture(scope="session")
def generated_outputs(tmp_path_factory: pytest.TempPathFactory):
    output_dir = tmp_path_factory.mktemp("aerodynamic_sensitivity_outputs")
    return aero.run_aerodynamic_sensitivity(
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


def test_source_contract_authorizes_exact_six_outputs() -> None:
    contract = load_source_contract(CONTRACT_PATH)
    assert contract.allowed_outputs(aero.MODULE_NAME) == aero.EXPECTED_OUTPUT_NAMES
    assert set(contract.allowed_outputs(aero.MODULE_NAME)) == EXPECTED_OUTPUT_NAMES
    assert len(contract.files_for_module(aero.MODULE_NAME)) == 5


def test_runtime_input_validation_passes_155_checks(loaded_inputs) -> None:
    _, runtime = loaded_inputs
    assert runtime["status"] == "PASS"
    assert runtime["checks"] == runtime["passed"] == 155
    assert runtime["warnings"] == runtime["failed"] == 0
    assert runtime["modules"] == ["aerodynamic_sensitivity"]


def test_frozen_input_structure_and_population(loaded_inputs) -> None:
    tables, _ = loaded_inputs
    aero._assert_structure(tables)
    raw = tables[aero.TABLE3_PATH]
    assert len(raw) == 288
    assert raw.duplicated(["county", "crop", "month", "year", "scenario_id"]).sum() == 0
    assert set(raw["county"]) == {"DALLAM", "HARTLEY", "MOORE", "OCHILTREE"}
    assert set(raw["crop"].str.lower()) == {"corn", "cotton"}
    assert set(raw["month"]) == {5, 6, 7, 8, 9, 10}
    assert set(raw["year"]) == {2011}
    assert raw["scenario_id"].nunique() == 6
    assert raw["scenario_id"].str.upper().ne("BASE").all()
    assert raw.groupby("scenario_id").size().eq(48).all()


def test_absolute_columns_equal_absolute_signed_columns(loaded_inputs) -> None:
    raw = loaded_inputs[0][aero.TABLE3_PATH]
    for spec in aero.METRICS.values():
        assert np.allclose(
            raw[spec["absolute_column"]],
            np.abs(raw[spec["signed_column"]]),
            rtol=0,
            atol=1e-12,
            equal_nan=True,
        )


def test_scenario_envelopes_reproduce_raw_corn_quantiles(loaded_inputs) -> None:
    tables, _ = loaded_inputs
    raw = tables[aero.TABLE3_PATH]
    assert aero._quantile_checks(
        raw,
        tables[aero.ETRF_ENVELOPE_PATH],
        "alt_ETf_mean_minus_BASE",
    )
    assert aero._quantile_checks(
        raw,
        tables[aero.H_ENVELOPE_PATH],
        "alt_H_mean_minus_BASE",
    )


def test_figure4_source_preserves_panel_populations_and_boundaries(derived_tables) -> None:
    figure = derived_tables[0]
    assert len(figure) == 288
    assert figure["panel_A_include"].all()
    assert int(figure["panel_BC_include"].sum()) == 144
    assert int(figure["peak_JulAug_include"].sum()) == 48
    assert figure["scenario_order"].between(0, 5).all()
    assert figure["interpretation_boundary"].eq(aero.INTERPRETATION_BOUNDARY).all()
    assert figure["interpretation_boundary"].str.contains(
        "No corrected ET product", regex=False
    ).all()


def test_peak_source_remains_below_diagnostic_band(derived_tables) -> None:
    peak = derived_tables[1]
    assert len(peak) == 8
    assert peak.duplicated(["county", "month"]).sum() == 0
    assert set(peak["month"]) == {7, 8}
    assert peak["diagnostic_limit_abs_dETf"].eq(0.01).all()
    assert peak["below_diagnostic_limit"].all()
    assert peak["max_abs_dETf_mean"].max() == pytest.approx(
        0.0060984693236338, abs=1e-15
    )
    maximum = peak.loc[peak["max_abs_dETf_mean"].idxmax()]
    assert maximum["county"] == "OCHILTREE"
    assert int(maximum["month"]) == 8


def test_text_statistics_have_explicit_required_definitions(derived_tables) -> None:
    statistics = derived_tables[3]
    required = {
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
    assert required.issubset(statistics.columns)
    assert len(statistics) == 30
    assert statistics.duplicated(["scope", "metric", "response_form"]).sum() == 0
    assert set(statistics["response_form"]) == {"signed", "absolute"}
    assert statistics["observational_unit"].eq(
        "non-BASE county-crop-month-scenario summary"
    ).all()
    assert statistics["ddof"].eq(1).all()
    assert statistics["standard_deviation"].ge(0).all()
    assert statistics["interpretation_boundary"].eq(
        aero.INTERPRETATION_BOUNDARY
    ).all()


def test_absolute_mean_and_sample_sd_values(derived_tables) -> None:
    statistics = derived_tables[3]
    absolute = statistics[statistics["response_form"].eq("absolute")].set_index(
        ["scope", "metric"]
    )
    for scope, expected in EXPECTED_ABSOLUTE_STATS.items():
        for metric in ("drah2", "dH", "dETf"):
            row = absolute.loc[(scope, metric)]
            mean, sd = expected[metric]
            assert int(row["n"]) == expected["n"]
            assert row["mean"] == pytest.approx(mean, abs=1e-15)
            assert row["standard_deviation"] == pytest.approx(sd, abs=1e-15)


def test_table3_summary_reproduces_manuscript_rounding(derived_tables) -> None:
    summary = derived_tables[2].set_index("scope")
    for scope, expected in EXPECTED_TABLE3_ROUNDED.items():
        row = summary.loc[scope]
        n, max_rah, med_rah, max_etf, med_etf, max_h, med_h = expected
        assert int(row["n_rows"]) == n
        if scope == "corn_peak_JulAug_nonBASE":
            assert round(float(row["max_abs_drah2_mean_s_m"])) == int(max_rah)
        else:
            assert round(float(row["max_abs_drah2_mean_s_m"]), 1) == max_rah
        assert round(float(row["median_abs_drah2_mean_s_m"]), 1) == med_rah
        assert round(float(row["max_abs_dETf_mean"]), 4) == max_etf
        assert round(float(row["median_abs_dETf_mean"]), 4) == med_etf
        assert round(float(row["max_abs_dH_mean_W_m2"]), 2) == max_h
        assert round(float(row["median_abs_dH_mean_W_m2"]), 2) == med_h
    assert summary["figure_changed"].eq(False).all()
    assert summary["scientific_boundary"].eq(aero.INTERPRETATION_BOUNDARY).all()


def test_manuscript_section_3_3_corn_and_peak_claims(derived_tables) -> None:
    summary = derived_tables[2].set_index("scope")
    corn = summary.loc["corn_all_months_nonBASE"]
    peak = summary.loc["corn_peak_JulAug_nonBASE"]
    assert round(float(corn["max_abs_dETf_mean"]), 4) == 0.0061
    assert round(float(corn["median_abs_dETf_mean"]), 4) == 0.0009
    assert round(float(peak["max_abs_dETf_mean"]), 4) == 0.0061
    assert round(float(peak["median_abs_dETf_mean"]), 4) == 0.0009
    assert float(peak["max_abs_dETf_mean"]) < aero.ETRF_DIAGNOSTIC_LIMIT


def test_derived_validation_passes_all_16_checks(derived_tables) -> None:
    figure, peak, summary, statistics, runtime, tables = derived_tables
    report = aero.validate_derived_outputs(
        figure,
        peak,
        summary,
        statistics,
        raw_table3=tables[aero.TABLE3_PATH],
        etrf_envelope=tables[aero.ETRF_ENVELOPE_PATH],
        h_envelope=tables[aero.H_ENVELOPE_PATH],
        runtime_validation_summary=runtime,
        contract_path=CONTRACT_PATH,
    )
    assert report["status"] == "PASS"
    assert report["checks"] == report["passed"] == 16
    assert report["warnings"] == report["failed"] == 0
    assert {item["check_id"] for item in report["check_results"]} == {
        f"AERO-{number:03d}" for number in range(1, 17)
    }


def test_generated_stable_outputs_match_frozen_hashes(generated_outputs) -> None:
    paths = {path.name: path for path in generated_outputs.paths()}
    assert set(paths) == EXPECTED_OUTPUT_NAMES
    for name, expected_hash in EXPECTED_STABLE_OUTPUT_HASHES.items():
        assert paths[name].is_file()
        assert paths[name].stat().st_size > 0
        assert _sha256(paths[name]) == expected_hash


def test_execution_record_reconciles_current_contract_and_outputs(generated_outputs) -> None:
    record = json.loads(
        generated_outputs.aerodynamic_sensitivity_execution_record.read_text()
    )
    assert record["module"] == aero.MODULE_NAME
    assert record["module_relative_path"] == aero.MODULE_RELATIVE_PATH
    assert record["module_sha256"] == _sha256(MODULE_PATH)
    assert record["contract_sha256"] == _sha256(CONTRACT_PATH)
    assert record["phase3_package"] == "PHASE3_CURATED_INPUT_LAYER_FINAL_002"
    assert record["authorized_inputs"] == {
        path: {"sha256": digest} for path, digest in EXPECTED_INPUT_HASHES.items()
    }
    assert record["runtime_input_validation"]["passed"] == 155
    assert record["derived_output_validation"] == {
        "status": "PASS",
        "checks": 16,
        "passed": 16,
        "warnings": 0,
        "failed": 0,
    }
    assert record["figure_change"] is False
    assert "METRIC is not rerun" in record["determinism_note"]
    for name, item in record["outputs"].items():
        path = generated_outputs.aerodynamic_sensitivity_execution_record.parent / name
        assert item["size_bytes"] == path.stat().st_size
        assert item["sha256"] == _sha256(path)


def test_second_execution_is_byte_deterministic(generated_outputs, tmp_path: Path) -> None:
    second = aero.run_aerodynamic_sensitivity(
        PHASE3_ROOT,
        tmp_path / "second_run",
        contract_path=CONTRACT_PATH,
    )
    first_hashes = {path.name: _sha256(path) for path in generated_outputs.paths()}
    second_hashes = {path.name: _sha256(path) for path in second.paths()}
    assert second_hashes == first_hashes


def test_existing_outputs_require_explicit_overwrite(generated_outputs) -> None:
    output_dir = generated_outputs.aerodynamic_sensitivity_validation.parent
    with pytest.raises(aero.AerodynamicSensitivityError, match="Refusing to overwrite"):
        aero.run_aerodynamic_sensitivity(
            PHASE3_ROOT,
            output_dir,
            contract_path=CONTRACT_PATH,
        )


def test_explicit_overwrite_preserves_all_output_hashes(generated_outputs) -> None:
    output_dir = generated_outputs.aerodynamic_sensitivity_validation.parent
    before = {path.name: _sha256(path) for path in generated_outputs.paths()}
    overwritten = aero.run_aerodynamic_sensitivity(
        PHASE3_ROOT,
        output_dir,
        contract_path=CONTRACT_PATH,
        overwrite=True,
    )
    after = {path.name: _sha256(path) for path in overwritten.paths()}
    assert after == before


def test_cli_success_and_refusal_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output_dir = tmp_path / "cli_outputs"
    args = [
        "--phase3-root",
        str(PHASE3_ROOT),
        "--output-dir",
        str(output_dir),
        "--contract",
        str(CONTRACT_PATH),
    ]
    assert aero.main(args) == 0
    success = capsys.readouterr()
    assert '"aerodynamic_sensitivity_execution_record"' in success.out
    assert "mean absolute response ± sample SD" in success.out
    assert success.err == ""

    assert aero.main(args) == 2
    failure = capsys.readouterr()
    assert "AERODYNAMIC_SENSITIVITY_ERROR" in failure.err
    assert "Refusing to overwrite" in failure.err


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate", "keys are not unique"),
        ("base", "BASE rows are prohibited"),
        ("year", "restricted to 2011"),
        ("county", "Unexpected county set"),
        ("absolute", "Absolute-value identity failed"),
        ("scope_n", "Unexpected scope counts"),
        ("peak_columns", "Unexpected peak-table columns"),
        ("envelope_n", "Each corn scenario envelope must contain 24 records"),
    ],
)
def test_malformed_aerodynamic_tables_are_rejected(
    loaded_inputs, mutation: str, message: str
) -> None:
    tables = {key: value.copy(deep=True) for key, value in loaded_inputs[0].items()}
    raw = tables[aero.TABLE3_PATH]
    if mutation == "duplicate":
        raw.loc[1, ["county", "crop", "month", "year", "scenario_id"]] = raw.loc[
            0, ["county", "crop", "month", "year", "scenario_id"]
        ].to_numpy()
    elif mutation == "base":
        raw.loc[0, "scenario_id"] = "BASE"
    elif mutation == "year":
        raw.loc[0, "year"] = 2015
    elif mutation == "county":
        raw.loc[0, "county"] = "UNAUTHORIZED"
    elif mutation == "absolute":
        raw.loc[0, "abs_alt_ETf_mean_minus_BASE"] += 0.1
    elif mutation == "scope_n":
        tables[aero.SCOPE_SUMMARY_PATH].loc[0, "n_rows"] = 287
    elif mutation == "peak_columns":
        tables[aero.PEAK_PATH] = tables[aero.PEAK_PATH].rename(columns={"8": "August"})
    elif mutation == "envelope_n":
        tables[aero.ETRF_ENVELOPE_PATH].loc[0, "n"] = 23
        tables[aero.H_ENVELOPE_PATH].loc[0, "n"] = 23
    else:  # pragma: no cover
        raise AssertionError(mutation)
    with pytest.raises(aero.AerodynamicSensitivityError, match=message):
        aero._assert_structure(tables)


def test_altered_runtime_input_hash_is_rejected(tmp_path: Path) -> None:
    copied_root = tmp_path / "PHASE3_CURATED_INPUT_LAYER_FINAL_002"
    copied_root.mkdir()
    contract = load_source_contract(CONTRACT_PATH)
    for item in contract.files_for_module(aero.MODULE_NAME):
        target = copied_root / item.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PHASE3_ROOT / item.relative_path, target)
    altered = copied_root / aero.PEAK_PATH
    altered.write_text(altered.read_text() + "\n", encoding="utf-8")
    with pytest.raises(RuntimeInputError, match="SHA-256"):
        aero.load_aerodynamic_inputs(copied_root, contract_path=CONTRACT_PATH)


def test_complete_phase3_validation_remains_1708_of_1708() -> None:
    report = validate_phase3_root(PHASE3_ROOT, contract_path=CONTRACT_PATH)
    summary = report.summary()
    assert summary["status"] == "PASS"
    assert summary["checks"] == summary["passed"] == 1708
    assert summary["warnings"] == summary["failed"] == 0


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
    assert "METRIC is not rerun" in source
    for relative_path in EXPECTED_INPUT_HASHES:
        assert relative_path in source
