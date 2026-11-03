"""Regression and scientific-boundary tests for known-cold sensitivity.

The tests exercise :mod:`qstable_awm.known_cold_reference` only against the six
frozen Phase 3 inputs authorized by the Phase 4 source contract. Historical
notebooks, full-resolution external rasters, and the manuscript are not runtime
inputs.

A manuscript-concordance test checks that exact Phase 4 results round to the
ranges reported in the abstract and Section 3.6. This is release QA only; the
production module continues to derive every value from the frozen Phase 3
known-cold tables and Supplementary Methods S1.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qstable_awm import known_cold_reference as kc
from qstable_awm.validation import RuntimeInputError, validate_phase3_root


PHASE4_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = PHASE4_ROOT / "config" / "phase4_source_contract.yml"
SOURCE_PATH = PHASE4_ROOT / "src" / "qstable_awm" / "known_cold_reference.py"
PHASE3_ROOT = Path(
    os.environ.get(
        "QSTABLE_PHASE3_ROOT",
        PHASE4_ROOT.parent / "PHASE3_CURATED_INPUT_LAYER_FINAL_002",
    )
).expanduser().resolve()

EXPECTED_MODULE_HASH = (
    "21b8019c04d8f618cc91c2e8bc41fc7ec060d01a57b41892b7d37397035f11cd"
)
EXPECTED_INPUT_HASHES = {
    kc.RASTER_SUMMARY_RELATIVE_PATH: (
        "0d818f59b2ab3911f6514ba96cd2ff99e10615cc1d26fa4b7261155769c38ba8"
    ),
    kc.COUNTY_COUNT_RELATIVE_PATH: (
        "2783f0cd0eab76813656cac717d6e9b8f64a26747bc19a0b1dea931592bf4287"
    ),
    kc.COUNTY_RADIUS_QA_RELATIVE_PATH: (
        "7c2b2c8dc2f40f9c145db457e7917b3cf380e006fd69f9b91ef8fc6daaa666d7"
    ),
    kc.FUNNEL_RELATIVE_PATH: (
        "61b571edfb2aceae6255c5f55553d717eed883a084b81d03cbb19a87d4b36a65"
    ),
    kc.SOLVER_PARAMETERS_RELATIVE_PATH: (
        "f5e48a51dcd343e565ce144c5d91bbe89d9a79258b1d761990245d019fdced84"
    ),
    kc.METHODS_RELATIVE_PATH: (
        "0133ead00178228b481d3fc291a433484cceae7007327eb2747e219fe266f186"
    ),
}
EXPECTED_STABLE_OUTPUT_HASHES = {
    "figure7_processing_funnel_source.csv": (
        "106ec36d533a32f1cf065a0d789c8170e295f33208849c8c07e2704c88c186d1"
    ),
    "figure7_county_radius_source.csv": (
        "f9a5bd32d5daca44631dfee5c16d9eb23a3cf06012ec935244be061fc9fafc77"
    ),
    "figure7_mask_response_source.csv": (
        "0ddc6eeedee16cec00c465a177451834dfd8dfd6adac78ee2219362deea000a1"
    ),
    "known_cold_reference_text_statistics.csv": (
        "ab9f7e48b854bcb4e739ee063b93fd0b3a7163265c702cd314027d2c69812188"
    ),
    "known_cold_reference_validation.json": (
        "729d5a659b1c2f372926147597056ef81509d39f79198d268fb22df1927aa27e"
    ),
}
EXPECTED_OUTPUT_NAMES = set(EXPECTED_STABLE_OUTPUT_HASHES) | {
    "known_cold_reference_execution_record.json"
}

EXPECTED_GROUP_RESPONSES = {
    ("MOORE", 1000, "valid_output"): 1.040242612361908,
    ("MOORE", 1000, "active_canopy"): 3.377700984477997,
    ("MOORE", 1000, "large_ETrF"): 2.812928080558777,
    ("MOORE", 1000, "mechanism_evidence"): 2.2084786891937256,
    ("MOORE", 3000, "valid_output"): 0.8618480265140533,
    ("MOORE", 3000, "active_canopy"): 3.9198991656303406,
    ("MOORE", 3000, "large_ETrF"): 2.2738720774650574,
    ("MOORE", 3000, "mechanism_evidence"): 1.4783296585083008,
    ("OCHILTREE", 1000, "valid_output"): 1.0419633388519287,
    ("OCHILTREE", 1000, "active_canopy"): 2.9132819175720215,
    ("OCHILTREE", 1000, "large_ETrF"): 2.4977356791496277,
    ("OCHILTREE", 1000, "mechanism_evidence"): 1.6471348404884338,
    ("OCHILTREE", 3000, "valid_output"): 1.0373782515525818,
    ("OCHILTREE", 3000, "active_canopy"): 2.9143993854522705,
    ("OCHILTREE", 3000, "large_ETrF"): 1.9855144023895264,
    ("OCHILTREE", 3000, "mechanism_evidence"): 1.8970860242843628,
}
EXPECTED_PRIMARY_SD = {
    "valid_output": 0.08902677611656903,
    "active_canopy": 0.4785935688531933,
    "large_ETrF": 0.3500210370462172,
    "mechanism_evidence": 0.3177417180761491,
}
EXPECTED_PRIMARY_MEAN = {
    "valid_output": 0.995358057320118,
    "active_canopy": 3.2813203632831573,
    "large_ETrF": 2.392512559890747,
    "mechanism_evidence": 1.8077573031187057,
}
MANUSCRIPT_RANGES_2DP = {
    "valid_output": (0.86, 1.04),
    "active_canopy": (2.91, 3.92),
    "large_ETrF": (1.99, 2.81),
    "mechanism_evidence": (1.48, 2.21),
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
    return kc.load_known_cold_inputs(PHASE3_ROOT, contract_path=CONTRACT_PATH)


@pytest.fixture(scope="session")
def derived_tables(loaded_inputs):
    rasters, counts, qa, funnel, parameters, methods, runtime = loaded_inputs
    kc._assert_input_structure(rasters, counts, qa, funnel, parameters, methods)
    funnel_source = kc.build_processing_funnel_source(funnel)
    county_source = kc.build_county_radius_source(rasters)
    mask_source = kc.build_mask_response_source(rasters)
    statistics = kc.build_text_statistics(rasters, county_source, mask_source)
    return funnel_source, county_source, mask_source, statistics, runtime


@pytest.fixture(scope="session")
def generated_outputs(tmp_path_factory: pytest.TempPathFactory):
    output_dir = tmp_path_factory.mktemp("known_cold_outputs")
    return kc.run_known_cold_reference(
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


def test_runtime_input_validation_passes_922_checks(loaded_inputs) -> None:
    *_, runtime = loaded_inputs
    assert runtime == {
        "report_version": "1.0.0",
        "modules": ["known_cold_reference"],
        "status": "PASS",
        "checks": 922,
        "passed": 922,
        "warnings": 0,
        "failed": 0,
    }


def test_strict_population_and_source_scope(loaded_inputs) -> None:
    rasters, counts, qa, funnel, parameters, methods, _ = loaded_inputs
    kc._assert_input_structure(rasters, counts, qa, funnel, parameters, methods)
    assert len(rasters) == 39
    assert rasters["raster_name"].nunique() == 39
    assert rasters["county"].value_counts().to_dict() == {"MOORE": 30, "OCHILTREE": 9}
    assert rasters.groupby(["county", "radius_m"]).size().to_dict() == {
        ("MOORE", 1000): 12,
        ("MOORE", 3000): 18,
        ("OCHILTREE", 1000): 4,
        ("OCHILTREE", 3000): 5,
    }
    assert rasters["raster_path"].str.startswith("source://").all()
    assert rasters["scene_date"].astype(str).str.startswith("2011-").all()


def test_strict_hierarchy_and_area_identities(loaded_inputs) -> None:
    rasters = loaded_inputs[0]
    assert (
        rasters["valid_pixels"].ge(rasters["active_strict_pixels"])
        & rasters["active_strict_pixels"].ge(rasters["high_strict_pixels"])
        & rasters["high_strict_pixels"].ge(rasters["mechanism_strict_pixels"])
    ).all()
    for pixel_column, area_column in (
        ("active_strict_pixels", "active_strict_area_ha"),
        ("high_strict_pixels", "high_strict_area_ha"),
        ("mechanism_strict_pixels", "mechanism_strict_area_ha"),
    ):
        expected = rasters[pixel_column] * rasters["pixel_area_m2"] / 10_000.0
        assert np.allclose(rasters[area_column], expected, rtol=0, atol=1e-8)


def test_solver_parameters_and_methods_preserve_finalized_authority(loaded_inputs) -> None:
    parameters = loaded_inputs[4]
    methods = loaded_inputs[5]
    assert parameters.set_index("parameter")["value"].to_dict() == dict(
        kc.EXPECTED_SOLVER_PARAMETERS
    )
    assert all(fragment in methods for fragment in kc.METHODS_REQUIRED_FRAGMENTS)
    assert "6.5 K thermal-separation gate" in methods
    assert "The main analysis did not use temporal fallback." in methods
    assert "without the 1.05 cold-reference multiplier" in methods


def test_processing_funnel_preserves_39_at_every_stage(derived_tables) -> None:
    funnel = derived_tables[0]
    assert funnel["stage"].tolist() == list(kc.EXPECTED_FUNNEL_STAGES)
    assert funnel["count"].tolist() == [39, 39, 39, 39]
    assert funnel["retention_fraction_of_candidates"].eq(1.0).all()
    assert funnel["figure_changed"].eq(False).all()


def test_county_radius_source_preserves_composition_and_numerical_qa(derived_tables) -> None:
    source = derived_tables[1]
    assert len(source) == 4
    assert source.duplicated(["county", "radius_m"]).sum() == 0
    assert source.set_index(["county", "radius_m"])["raster_count"].to_dict() == {
        ("MOORE", 1000): 12,
        ("MOORE", 3000): 18,
        ("OCHILTREE", 1000): 4,
        ("OCHILTREE", 3000): 5,
    }
    assert source["same_date_source_only"].eq(True).all()
    assert source["thermal_separation_gate_K"].eq(6.5).all()
    assert source["minimum_source_pixels_required"].eq(30).all()
    assert source["source_pixel_count_minimum"].ge(30).all()
    assert source["thermal_gap_median_K"].ge(6.5).all()
    assert source["nonconverged_fraction_median"].eq(0.0).all()
    assert source["rah_cap_hit_fraction_median"].eq(0.0).all()


def test_mask_response_values_derive_from_strict_raster_medians(derived_tables) -> None:
    source = derived_tables[2]
    assert len(source) == 16
    observed = source.set_index(["county", "radius_m", "support_domain"])[
        "group_median_abs_delta_ETa_mm_d"
    ].to_dict()
    assert observed == pytest.approx(EXPECTED_GROUP_RESPONSES, abs=1e-12, rel=0)
    assert source["aggregation_definition"].eq(
        "median across retained raster-radius record medians"
    ).all()
    assert source["material_response_threshold_mm_d"].eq(1.0).all()


def test_manuscript_ranges_and_counts_reconcile(derived_tables) -> None:
    source = derived_tables[2]
    for domain, expected in MANUSCRIPT_RANGES_2DP.items():
        values = source.loc[
            source["support_domain"].eq(domain),
            "group_median_abs_delta_ETa_mm_d",
        ]
        observed = (round(float(values.min()), 2), round(float(values.max()), 2))
        assert observed == expected
    county = derived_tables[1].groupby("county")["raster_count"].sum().to_dict()
    assert county == {"MOORE": 30, "OCHILTREE": 9}


def test_text_statistics_have_explicit_sd_definitions(derived_tables) -> None:
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
    assert len(statistics) == 14
    assert statistics["ddof"].eq(1).all()
    assert statistics["standard_deviation"].ge(0).all()
    primary = statistics[
        statistics["statistic"].eq("county_radius_group_response_dispersion")
    ].set_index("support_domain")
    assert primary["n"].eq(4).all()
    assert primary["observational_unit"].eq("county-radius group").all()
    assert primary["primary_for_text"].eq(True).all()
    assert primary["mean"].to_dict() == pytest.approx(EXPECTED_PRIMARY_MEAN, abs=1e-12)
    assert primary["standard_deviation"].to_dict() == pytest.approx(
        EXPECTED_PRIMARY_SD, abs=1e-12
    )


def test_secondary_raster_sd_preserves_non_null_counts(derived_tables) -> None:
    statistics = derived_tables[3]
    secondary = statistics[
        statistics["statistic"].eq("strict_raster_record_response_dispersion")
    ].set_index("support_domain")
    assert secondary["n"].to_dict() == {
        "valid_output": 39,
        "active_canopy": 39,
        "large_ETrF": 39,
        "mechanism_evidence": 38,
    }
    assert secondary["primary_for_text"].eq(False).all()
    assert secondary["statistical_definition"].str.contains(
        "not treated as independent validation replicates", regex=False
    ).all()


def test_sample_sd_function_uses_ddof_one() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert kc._sample_sd(values) == pytest.approx(values.std(ddof=1), abs=1e-15)
    assert np.isnan(kc._sample_sd(pd.Series([1.0])))


def test_derived_validation_passes_all_52_checks(loaded_inputs, derived_tables) -> None:
    rasters, counts, qa, _, parameters, _, _ = loaded_inputs
    funnel, county, masks, statistics, runtime = derived_tables
    report = kc.validate_derived_outputs(
        funnel,
        county,
        masks,
        statistics,
        raster_summary=rasters,
        county_counts=counts,
        county_radius_qa=qa,
        solver_parameters=parameters,
        runtime_validation_summary=runtime,
        contract_path=CONTRACT_PATH,
    )
    assert report["status"] == "PASS"
    assert report["checks"] == report["passed"] == 52
    assert report["warnings"] == report["failed"] == 0
    assert {item["check_id"] for item in report["check_results"]} == {
        f"KC-{number:03d}" for number in range(1, 53)
    }


def test_repository_wide_phase3_validation_remains_1708_of_1708() -> None:
    report = validate_phase3_root(PHASE3_ROOT, contract_path=CONTRACT_PATH)
    summary = report.summary()
    assert summary["status"] == "PASS"
    assert summary["checks"] == summary["passed"] == 1_708
    assert summary["warnings"] == summary["failed"] == 0


def test_run_writes_stable_scientific_outputs(generated_outputs) -> None:
    paths = {path.name: path for path in generated_outputs.paths()}
    assert set(paths) == EXPECTED_OUTPUT_NAMES
    for name, expected_hash in EXPECTED_STABLE_OUTPUT_HASHES.items():
        assert paths[name].is_file()
        assert paths[name].stat().st_size > 0
        assert _sha256(paths[name]) == expected_hash
    assert paths["known_cold_reference_execution_record.json"].stat().st_size > 0
    validation = json.loads(
        generated_outputs.known_cold_reference_validation.read_text(encoding="utf-8")
    )
    assert validation["status"] == "PASS"
    assert validation["checks"] == validation["passed"] == 52
    assert validation["warnings"] == validation["failed"] == 0


def test_execution_record_reconciles_current_contract_and_outputs(generated_outputs) -> None:
    record = json.loads(
        generated_outputs.known_cold_reference_execution_record.read_text(
            encoding="utf-8"
        )
    )
    assert record["module"] == "known_cold_reference"
    assert record["module_relative_path"] == kc.MODULE_RELATIVE_PATH
    assert record["module_sha256"] == EXPECTED_MODULE_HASH
    assert record["contract_sha256"] == _sha256(CONTRACT_PATH)
    assert record["phase3_package"] == "PHASE3_CURATED_INPUT_LAYER_FINAL_002"
    assert {
        path: item["sha256"] for path, item in record["authorized_inputs"].items()
    } == EXPECTED_INPUT_HASHES
    assert record["runtime_input_validation"]["passed"] == 922
    assert record["derived_output_validation"] == {
        "status": "PASS",
        "checks": 52,
        "passed": 52,
        "warnings": 0,
        "failed": 0,
    }
    assert set(record["outputs"]) == set(EXPECTED_STABLE_OUTPUT_HASHES)
    for name, metadata in record["outputs"].items():
        assert metadata["sha256"] == EXPECTED_STABLE_OUTPUT_HASHES[name]
        assert metadata["size_bytes"] > 0
    assert record["solver_authority"]["main_thermal_separation_gate_K"] == 6.5
    assert record["solver_authority"]["same_date_source_only"] is True
    assert record["figure_change"] is False
    assert record["scientific_boundary"] == kc.INTERPRETATION_BOUNDARY


def test_second_execution_is_byte_deterministic(generated_outputs, tmp_path: Path) -> None:
    second = kc.run_known_cold_reference(
        PHASE3_ROOT,
        tmp_path / "second",
        contract_path=CONTRACT_PATH,
    )
    first_hashes = {path.name: _sha256(path) for path in generated_outputs.paths()}
    second_hashes = {path.name: _sha256(path) for path in second.paths()}
    assert second_hashes == first_hashes
    assert {name: first_hashes[name] for name in EXPECTED_STABLE_OUTPUT_HASHES} == (
        EXPECTED_STABLE_OUTPUT_HASHES
    )


def test_existing_outputs_require_explicit_overwrite(generated_outputs) -> None:
    output_dir = generated_outputs.figure7_processing_funnel_source.parent
    with pytest.raises(kc.KnownColdReferenceError, match="Refusing to overwrite"):
        kc.run_known_cold_reference(
            PHASE3_ROOT,
            output_dir,
            contract_path=CONTRACT_PATH,
        )


def test_explicit_overwrite_is_byte_deterministic(generated_outputs) -> None:
    output_dir = generated_outputs.figure7_processing_funnel_source.parent
    before = {path.name: _sha256(path) for path in generated_outputs.paths()}
    overwritten = kc.run_known_cold_reference(
        PHASE3_ROOT,
        output_dir,
        contract_path=CONTRACT_PATH,
        overwrite=True,
    )
    after = {path.name: _sha256(path) for path in overwritten.paths()}
    assert after == before
    assert {name: after[name] for name in EXPECTED_STABLE_OUTPUT_HASHES} == (
        EXPECTED_STABLE_OUTPUT_HASHES
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate_raster", "raster names are not unique"),
        ("invalid_year", "restricted to 2011"),
        ("runtime_raster_path", "non-runtime source:// references"),
        ("broken_hierarchy", "Strict mask hierarchy failed"),
        ("funnel_count", "processing-funnel stage must equal 39"),
        ("thermal_gate", "solver parameter table changed"),
        ("methods_fragment", "missing required authority statements"),
        ("nonconvergence", "Expected zero nonconvergence"),
        ("rah_cap", "Expected zero aerodynamic-resistance cap hits"),
    ],
)
def test_malformed_known_cold_authority_is_rejected(
    loaded_inputs,
    mutation: str,
    message: str,
) -> None:
    rasters, counts, qa, funnel, parameters, methods, _ = loaded_inputs
    rasters = rasters.copy(deep=True)
    counts = counts.copy(deep=True)
    qa = qa.copy(deep=True)
    funnel = funnel.copy(deep=True)
    parameters = parameters.copy(deep=True)

    if mutation == "duplicate_raster":
        rasters.loc[1, "raster_name"] = rasters.loc[0, "raster_name"]
    elif mutation == "invalid_year":
        rasters.loc[0, "scene_date"] = "2012-07-01"
    elif mutation == "runtime_raster_path":
        rasters.loc[0, "raster_path"] = "/scratch/uncontrolled.tif"
    elif mutation == "broken_hierarchy":
        rasters.loc[0, "active_strict_pixels"] = rasters.loc[0, "valid_pixels"] + 1
    elif mutation == "funnel_count":
        funnel.loc[0, "count"] = 38
    elif mutation == "thermal_gate":
        parameters.loc[
            parameters["parameter"].eq("Thermal-separation gate"), "value"
        ] = "1.0 K"
    elif mutation == "methods_fragment":
        methods = methods.replace("6.5 K thermal-separation gate", "1.0 K gate")
    elif mutation == "nonconvergence":
        rasters.loc[0, "nonconverged_strict_frac_valid"] = 0.01
    elif mutation == "rah_cap":
        rasters.loc[0, "rah_cap_hit_strict_frac_valid"] = 0.01
    else:  # pragma: no cover
        raise AssertionError(mutation)

    with pytest.raises(kc.KnownColdReferenceError, match=message):
        kc._assert_input_structure(rasters, counts, qa, funnel, parameters, methods)


def test_unauthorized_or_altered_input_is_rejected(tmp_path: Path) -> None:
    fake_root = tmp_path / "PHASE3_CURATED_INPUT_LAYER_FINAL_002"
    for relative_path in EXPECTED_INPUT_HASHES:
        target = fake_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((PHASE3_ROOT / relative_path).read_bytes())
    altered = fake_root / kc.RASTER_SUMMARY_RELATIVE_PATH
    altered.write_bytes(altered.read_bytes() + b"\n")
    with pytest.raises(RuntimeInputError):
        kc.load_known_cold_inputs(fake_root, contract_path=CONTRACT_PATH)


def test_cli_success_and_refusal_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "cli"
    args = [
        "--phase3-root",
        str(PHASE3_ROOT),
        "--output-dir",
        str(output_dir),
        "--contract",
        str(CONTRACT_PATH),
    ]
    assert kc.main(args) == 0
    captured = capsys.readouterr()
    assert "Known-cold reference-sensitivity descriptive statistics" in captured.out
    assert "Primary SD: sample SD across four county-radius group medians" in captured.out
    assert captured.err == ""

    assert kc.main(args) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "KNOWN_COLD_REFERENCE_ERROR" in captured.err
    assert "Refusing to overwrite" in captured.err


def test_source_contains_no_notebook_full_raster_or_prohibited_runtime_dependency() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    prohibited = (
        ".ipynb",
        "/scratch/",
        "AWM_figures",
        "V46D",
        "V48",
        "V49",
        "papermill",
        "nbconvert",
        "rasterio.open",
    )
    assert {token for token in prohibited if token in source} == set()
    assert "validate_module_inputs" in source
    assert kc.METHODS_RELATIVE_PATH in source
    assert kc.SOLVER_PARAMETERS_RELATIVE_PATH in source
    assert "6.5 K" in source
