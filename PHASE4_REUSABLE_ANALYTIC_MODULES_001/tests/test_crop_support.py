"""Regression and scientific-boundary tests for :mod:`qstable_awm.crop_support`.

These tests lock the accepted Figure 5 crop-support authority without using
historical notebooks. They exercise only the frozen Phase 3 crop-support
inputs, the Phase 4 source contract, and deterministic Phase 4 outputs.

The numerical acceptance values below are test invariants, not alternate
runtime data sources. Production calculations must continue to derive them
from the three authorized Phase 3 tables.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PHASE4_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PHASE4_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qstable_awm import crop_support as cs  # noqa: E402


PHASE3_ROOT = Path(
    os.environ.get(
        "QSTABLE_PHASE3_ROOT",
        PHASE4_ROOT.parent / "PHASE3_CURATED_INPUT_LAYER_FINAL_002",
    )
).expanduser().resolve()
CONTRACT_PATH = PHASE4_ROOT / "config" / "phase4_source_contract.yml"
MODULE_PATH = PHASE4_ROOT / "src" / "qstable_awm" / "crop_support.py"

EXPECTED_INPUT_HASHES = {
    cs.OBJECT_RELATIVE_PATH: (
        "528854d42a191ffa0bd96ace61b3f3b00f8d8ef08996be960d14ec23e712f4de"
    ),
    cs.COUNTY_RELATIVE_PATH: (
        "88103fc1852d9e38efc3870d805908be2abcee417d83616ee38b975a6be8dc64"
    ),
    cs.DOMAIN_RELATIVE_PATH: (
        "fa136001045f44206e51f2912053096b7b8f11bd2aa78aeb591879c30780213f"
    ),
}
EXPECTED_SCIENTIFIC_OUTPUT_HASHES = {
    "figure5_domain_crop_support_source.csv": (
        "fbd2abdd86a6953647898ed11c35b07195e72f51826a81aef57ec551785010a3"
    ),
    "figure5_county_crop_support_source.csv": (
        "61b06815f7381aed81724ac6c21564def8efeb23d5f8114640dcbeea32109628"
    ),
    "crop_support_text_statistics.csv": (
        "504554f4b3e38f6866dbca9179c3eedccf9e0228bd58828c512d8fd6d8c8b7b2"
    ),
    "crop_support_validation.json": (
        "14794f8caaacfffe2f063b34b51240740f288a04111d254a6a28f03983d8a805"
    ),
}
EXPECTED_DOMAIN_FRACTIONS = {
    2011: 1.0000000000000000,
    2015: 0.5601469953626429,
    2017: 0.5151891407869190,
    2023: 0.3470846443491381,
}
EXPECTED_COUNTY_SD = {
    2011: 0.0,
    2015: 0.0575074000022263,
    2017: 0.0799864679707078,
    2023: 0.0436648176127665,
}
EXPECTED_WEIGHTED_OBJECT_SD = {
    2011: 0.0,
    2015: 0.3502655759023395,
    2017: 0.3600325177691951,
    2023: 0.3396444730619016,
}
EXPECTED_UNWEIGHTED_OBJECT_SD = {
    2011: 0.0,
    2015: 0.4131059208206352,
    2017: 0.4152158757643297,
    2023: 0.3700673172222830,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture(scope="session")
def loaded_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    assert PHASE3_ROOT.is_dir(), (
        "Frozen Phase 3 root not found. Set QSTABLE_PHASE3_ROOT to "
        "PHASE3_CURATED_INPUT_LAYER_FINAL_002."
    )
    return cs.load_crop_support_inputs(PHASE3_ROOT, contract_path=CONTRACT_PATH)


@pytest.fixture(scope="session")
def generated_outputs(tmp_path_factory: pytest.TempPathFactory) -> cs.CropSupportOutputs:
    output_dir = tmp_path_factory.mktemp("crop_support_outputs")
    return cs.run_crop_support(
        PHASE3_ROOT,
        output_dir,
        contract_path=CONTRACT_PATH,
    )


def test_authorized_input_hashes_are_frozen() -> None:
    for relative_path, expected_hash in EXPECTED_INPUT_HASHES.items():
        path = PHASE3_ROOT / relative_path
        assert path.is_file(), relative_path
        assert _sha256(path) == expected_hash


def test_input_validation_and_fixed_frame_structure(loaded_inputs) -> None:
    object_year, county_year, domain_year, summary = loaded_inputs

    assert summary == {
        "report_version": "1.0.0",
        "modules": ["crop_support"],
        "status": "PASS",
        "checks": 214,
        "passed": 214,
        "warnings": 0,
        "failed": 0,
    }
    assert len(object_year) == 4_736
    assert object_year["object_id"].nunique() == 1_184
    assert object_year.duplicated(["object_id", "year"]).sum() == 0
    assert set(object_year["year"]) == {2011, 2015, 2017, 2023}
    assert set(object_year["county"]) == {
        "DALLAM",
        "HARTLEY",
        "MOORE",
        "OCHILTREE",
    }
    assert object_year.groupby("object_id")["year"].nunique().eq(4).all()
    assert len(county_year) == 20  # 16 county-years plus four embedded domain rows.
    assert len(domain_year) == 4


def test_input_structure_accepts_frozen_tables(loaded_inputs) -> None:
    object_year, county_year, domain_year, _ = loaded_inputs
    cs._assert_input_structure(object_year, county_year, domain_year)


def test_domain_source_reproduces_accepted_figure5_values(loaded_inputs) -> None:
    _, _, domain_year, _ = loaded_inputs
    source = cs.build_figure5_domain_source(domain_year)

    assert source["year"].tolist() == [2011, 2015, 2017, 2023]
    assert source["county"].eq("DOMAIN").all()
    assert source["fixed_frame_object_count"].eq(1_184).all()
    assert source["figure_changed"].eq(False).all()
    assert source["interpretation_boundary"].eq(cs.INTERPRETATION_BOUNDARY).all()

    observed = source.set_index("year")["crop_support_fraction"].to_dict()
    assert observed == pytest.approx(EXPECTED_DOMAIN_FRACTIONS, abs=1e-15, rel=0)
    assert source.loc[source["year"].eq(2011), "threshold_status"].iloc[0] == (
        "meets_or_exceeds_0.70"
    )
    assert source.loc[source["year"].ne(2011), "threshold_status"].eq(
        "below_0.70"
    ).all()


def test_county_source_preserves_threshold_boundary(loaded_inputs) -> None:
    _, county_year, domain_year, _ = loaded_inputs
    source = cs.build_figure5_county_source(county_year, domain_year)

    assert len(source) == 16
    assert source.duplicated(["county", "year"]).sum() == 0
    assert set(source["county"]) == set(cs.EXPECTED_COUNTIES)
    assert source.loc[source["year"].eq(2011), "crop_support_fraction"].eq(1.0).all()

    comparison = source[source["year"].ne(2011)]
    assert len(comparison) == 12
    assert comparison["crop_support_fraction"].max() == pytest.approx(
        0.6240340312941867, abs=1e-15
    )
    assert comparison["crop_support_fraction"].lt(0.70).all()
    assert comparison["threshold_status"].eq("below_0.70").all()
    assert source["figure_changed"].eq(False).all()


def test_text_statistics_have_explicit_units_and_sd_definitions(loaded_inputs) -> None:
    object_year, county_year, _, _ = loaded_inputs
    statistics = cs.build_text_statistics(object_year, county_year)

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
    assert len(statistics) == 12
    assert statistics.duplicated(["year", "statistic"]).sum() == 0
    assert statistics[list(required)].isna().sum().sum() == 0
    assert statistics["standard_deviation"].ge(0.0).all()
    assert statistics["interpretation_boundary"].eq(cs.INTERPRETATION_BOUNDARY).all()

    county = statistics[
        statistics["statistic"].eq("county_crop_support_sample_dispersion")
    ].set_index("year")
    weighted = statistics[
        statistics["statistic"].eq(
            "object_crop_support_area_weighted_population_dispersion"
        )
    ].set_index("year")
    unweighted = statistics[
        statistics["statistic"].eq(
            "object_crop_support_unweighted_sample_dispersion"
        )
    ].set_index("year")

    assert county["n"].eq(4).all()
    assert county["ddof"].eq(1).all()
    assert county["observational_unit"].eq("county").all()
    assert county["primary_for_text"].eq(True).all()

    assert weighted["n"].eq(1_184).all()
    assert weighted["ddof"].eq(0).all()
    assert weighted["observational_unit"].eq("fixed-frame object").all()
    assert weighted["standard_deviation_definition"].str.contains(
        "Population-style", regex=False
    ).all()

    assert unweighted["n"].eq(1_184).all()
    assert unweighted["ddof"].eq(1).all()
    assert unweighted["primary_for_text"].eq(False).all()

    assert county["standard_deviation"].to_dict() == pytest.approx(
        EXPECTED_COUNTY_SD, abs=1e-15, rel=0
    )
    assert weighted["standard_deviation"].to_dict() == pytest.approx(
        EXPECTED_WEIGHTED_OBJECT_SD, abs=1e-15, rel=0
    )
    assert unweighted["standard_deviation"].to_dict() == pytest.approx(
        EXPECTED_UNWEIGHTED_OBJECT_SD, abs=1e-15, rel=0
    )
    assert weighted["mean"].to_dict() == pytest.approx(
        EXPECTED_DOMAIN_FRACTIONS, abs=1e-15, rel=0
    )


def test_weighted_population_dispersion_formula() -> None:
    values = np.array([0.0, 1.0], dtype=float)
    weights = np.array([1.0, 3.0], dtype=float)

    assert cs._weighted_mean(values, weights) == pytest.approx(0.75)
    assert cs._weighted_population_sd(values, weights) == pytest.approx(
        math.sqrt(0.1875)
    )
    assert cs._weighted_median(values, weights) == pytest.approx(1.0)


def test_derived_validation_passes_all_26_checks(loaded_inputs) -> None:
    object_year, county_year, domain_year, runtime_summary = loaded_inputs
    domain_source = cs.build_figure5_domain_source(domain_year)
    county_source = cs.build_figure5_county_source(county_year, domain_year)
    statistics = cs.build_text_statistics(object_year, county_year)

    report = cs.validate_derived_outputs(
        domain_source,
        county_source,
        statistics,
        object_year=object_year,
        county_year=county_year,
        domain_year=domain_year,
        runtime_validation_summary=runtime_summary,
        contract_path=CONTRACT_PATH,
    )

    assert report["status"] == "PASS"
    assert report["checks"] == 26
    assert report["passed"] == 26
    assert report["warnings"] == 0
    assert report["failed"] == 0
    assert {item["check_id"] for item in report["check_results"]} == {
        f"CROP-{number:03d}" for number in range(1, 27)
    }
    assert all(item["status"] == "PASS" for item in report["check_results"])


def test_run_writes_exact_deterministic_outputs(generated_outputs) -> None:
    paths = {path.name: path for path in generated_outputs.paths()}
    assert set(paths) == set(cs.EXPECTED_OUTPUT_NAMES)

    for name, expected_hash in EXPECTED_SCIENTIFIC_OUTPUT_HASHES.items():
        assert paths[name].is_file()
        assert paths[name].stat().st_size > 0
        assert _sha256(paths[name]) == expected_hash

    validation = json.loads(generated_outputs.crop_support_validation.read_text())
    assert validation["status"] == "PASS"
    assert validation["checks"] == validation["passed"] == 26
    assert validation["warnings"] == validation["failed"] == 0


def test_execution_record_reconciles_module_inputs_and_outputs(generated_outputs) -> None:
    record = json.loads(generated_outputs.crop_support_execution_record.read_text())

    assert record["module"] == "crop_support"
    assert record["phase3_package"] == "PHASE3_CURATED_INPUT_LAYER_FINAL_002"
    assert record["module_relative_path"] == "src/qstable_awm/crop_support.py"
    assert record["module_sha256"] == _sha256(MODULE_PATH)
    assert record["contract_sha256"] == _sha256(CONTRACT_PATH)
    assert record["authorized_inputs"] == {
        path: {"sha256": digest} for path, digest in EXPECTED_INPUT_HASHES.items()
    }
    assert record["runtime_input_validation"]["passed"] == 214
    assert record["derived_output_validation"] == {
        "status": "PASS",
        "checks": 26,
        "passed": 26,
        "warnings": 0,
        "failed": 0,
    }
    assert record["figure_change"] is False
    assert record["scientific_boundary"] == cs.INTERPRETATION_BOUNDARY

    for name, item in record["outputs"].items():
        path = generated_outputs.crop_support_execution_record.parent / name
        assert item["size_bytes"] == path.stat().st_size
        assert item["sha256"] == _sha256(path)


def test_second_execution_is_byte_deterministic(
    generated_outputs,
    tmp_path: Path,
) -> None:
    second = cs.run_crop_support(
        PHASE3_ROOT,
        tmp_path / "second_run",
        contract_path=CONTRACT_PATH,
    )
    first_hashes = {path.name: _sha256(path) for path in generated_outputs.paths()}
    second_hashes = {path.name: _sha256(path) for path in second.paths()}
    assert second_hashes == first_hashes
    assert {name: first_hashes[name] for name in EXPECTED_SCIENTIFIC_OUTPUT_HASHES} == EXPECTED_SCIENTIFIC_OUTPUT_HASHES


def test_existing_outputs_require_explicit_overwrite(generated_outputs) -> None:
    output_dir = generated_outputs.crop_support_validation.parent
    with pytest.raises(cs.CropSupportError, match="Refusing to overwrite"):
        cs.run_crop_support(
            PHASE3_ROOT,
            output_dir,
            contract_path=CONTRACT_PATH,
        )


def test_explicit_overwrite_preserves_output_hashes(generated_outputs) -> None:
    output_dir = generated_outputs.crop_support_validation.parent
    overwritten = cs.run_crop_support(
        PHASE3_ROOT,
        output_dir,
        contract_path=CONTRACT_PATH,
        overwrite=True,
    )
    assert {path.name: _sha256(path) for path in overwritten.paths()} == (
        {**EXPECTED_SCIENTIFIC_OUTPUT_HASHES, "crop_support_execution_record.json": _sha256(generated_outputs.crop_support_execution_record)}
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate", "Object-year keys are not unique"),
        ("fraction", "Invalid fraction outside"),
        ("year", "Unexpected years"),
        ("county", "Unexpected counties"),
        ("threshold", "reaches the 0.70 crop-support threshold"),
    ],
)
def test_malformed_crop_tables_are_rejected(loaded_inputs, mutation: str, message: str) -> None:
    object_year, county_year, domain_year, _ = loaded_inputs
    object_copy = object_year.copy(deep=True)
    county_copy = county_year.copy(deep=True)
    domain_copy = domain_year.copy(deep=True)

    if mutation == "duplicate":
        object_copy.loc[1, ["object_id", "year"]] = object_copy.loc[
            0, ["object_id", "year"]
        ].to_numpy()
    elif mutation == "fraction":
        object_copy.loc[0, "corn_fraction"] = 1.01
    elif mutation == "year":
        object_copy.loc[object_copy["year"].eq(2023), "year"] = 2024
    elif mutation == "county":
        object_copy.loc[0, "county"] = "UNAUTHORIZED"
    elif mutation == "threshold":
        row = county_copy.index[
            county_copy["county"].eq("DALLAM") & county_copy["year"].eq(2015)
        ][0]
        county_copy.loc[row, "corn_fraction_area_weighted"] = 0.70
    else:  # pragma: no cover - protected by parametrization.
        raise AssertionError(mutation)

    with pytest.raises(cs.CropSupportError, match=message):
        cs._assert_input_structure(object_copy, county_copy, domain_copy)


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

    assert cs.main(args) == 0
    success = capsys.readouterr()
    assert '"crop_support_validation"' in success.out
    assert "County SD: sample SD across four counties" in success.out
    assert success.err == ""

    assert cs.main(args) == 2
    failure = capsys.readouterr()
    assert "CROP_SUPPORT_ERROR" in failure.err
    assert "Refusing to overwrite" in failure.err


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
    assert cs.OBJECT_RELATIVE_PATH in source
    assert cs.COUNTY_RELATIVE_PATH in source
    assert cs.DOMAIN_RELATIVE_PATH in source
