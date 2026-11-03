"""Regression and scientific-boundary tests for fixed-frame dominance.

The tests exercise :mod:`qstable_awm.fixed_frame_dominance` only against the
frozen ``PHASE3_CURATED_INPUT_LAYER_FINAL_002`` tables authorized by the Phase
4 source contract. Historical notebooks and the manuscript are not runtime
inputs.

A dedicated concordance test verifies that the exact Phase 4 values round to
the values stated in manuscript Section 3.5 and Figure 6. This is a release QA
check, not a numerical source for the analytical module.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qstable_awm import fixed_frame_dominance as ffd
from qstable_awm.validation import RuntimeInputError, validate_phase3_root


PHASE4_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = PHASE4_ROOT / "config" / "phase4_source_contract.yml"
SOURCE_PATH = PHASE4_ROOT / "src" / "qstable_awm" / "fixed_frame_dominance.py"
SCHEMA_PATH = PHASE4_ROOT / "src" / "qstable_awm" / "schemas.py"
PHASE3_ROOT = Path(
    os.environ.get(
        "QSTABLE_PHASE3_ROOT",
        PHASE4_ROOT.parent / "PHASE3_CURATED_INPUT_LAYER_FINAL_002",
    )
).expanduser().resolve()

EXPECTED_MODULE_HASH = (
    "35f0209d2ec1ac861066c4975ec28cc317a5b11bf324a21de392f83f48355f94"
)
EXPECTED_SCHEMA_HASH = (
    "27e7dbef16f8cc52eedc81c18f0662fe49c97eb51e0d2a42e9675a0f78125471"
)
EXPECTED_INPUT_HASHES = {
    ffd.SENSITIVITY_RELATIVE_PATH: (
        "211a49a19a22e27dfe2038e1ddb2a2a2978d3bdfc71287d2e8262af7d00623d2"
    ),
    ffd.OBJECT_RELATIVE_PATH: (
        "1ff7441441798f52a2bad40909f3dd6058ba511610276bfde7242044b935654c"
    ),
}
EXPECTED_SCIENTIFIC_OUTPUT_HASHES = {
    "figure6_fixed_frame_sensitivity_source.csv": (
        "804134a4923370545af3fd899ee8464e3fac87fb5fd03399246e9c4659005232"
    ),
    "figure6_dominance_class_source.csv": (
        "136b852aec206343454e8e90e600f1b01c29a0573dc92112430893c0b6124392"
    ),
    "fixed_frame_dominance_text_statistics.csv": (
        "09281405e2840ab4552dd8665586cc7c6a9ae0e317a1deafcbfddda8c9af93df"
    ),
    "fixed_frame_dominance_validation.json": (
        "06b3247c985b3a34d72bb7472ecc083eb17e80a6502c0b10022e8882811ab347"
    ),
}

EXPECTED_PERCENT_CHANGE = {
    ("DALLAM", "ETa"): -31.93337820575056,
    ("HARTLEY", "ETa"): -28.921374912919607,
    ("MOORE", "ETa"): -17.440115983190676,
    ("OCHILTREE", "ETa"): -42.46097638132779,
    ("DALLAM", "NIR"): -79.77162282743792,
    ("HARTLEY", "NIR"): -65.38007254283406,
    ("MOORE", "NIR"): -44.18936884007064,
    ("OCHILTREE", "NIR"): -90.61262577628952,
    ("DALLAM", "Qgross_depth"): -79.77162282797512,
    ("HARTLEY", "Qgross_depth"): -65.38007254275622,
    ("MOORE", "Qgross_depth"): -44.18936884290331,
    ("OCHILTREE", "Qgross_depth"): -90.61262577528252,
}
EXPECTED_DIFFERENCE_MM = {
    ("DALLAM", "ETa"): 167.95989077071607,
    ("HARTLEY", "ETa"): 169.67998739880323,
    ("MOORE", "ETa"): 122.32864712206776,
    ("OCHILTREE", "ETa"): 215.9862785981527,
    ("DALLAM", "NIR"): 208.9212692208527,
    ("HARTLEY", "NIR"): 212.08610097934945,
    ("MOORE", "NIR"): 179.6143356829757,
    ("OCHILTREE", "NIR"): 247.2101736368159,
    ("DALLAM", "Qgross_depth"): 237.4105332089102,
    ("HARTLEY", "Qgross_depth"): 241.0069329343362,
    ("MOORE", "Qgross_depth"): 204.10719966652616,
    ("OCHILTREE", "Qgross_depth"): 280.92065181977216,
}
EXPECTED_COUNTY_SD = {
    ("ETa", "difference_4yr_minus_3yr"): 38.242512362373944,
    ("ETa", "percent_change_when_2011_removed"): 10.29215197222375,
    ("NIR", "difference_4yr_minus_3yr"): 27.67710552818503,
    ("NIR", "percent_change_when_2011_removed"): 20.065698995251562,
    ("Qgross_depth", "difference_4yr_minus_3yr"): 31.451256254390103,
    ("Qgross_depth", "percent_change_when_2011_removed"): 20.0656989937858,
}
EXPECTED_OBJECT_SD = {
    ("ETa", "ETa_difference_due_to_2011"): 80.37588889128047,
    ("ETa", "ETa_percent_change_without_2011"): 13.699990720434316,
    ("NIR", "NIR_difference_due_to_2011"): 83.74249322817269,
    ("NIR", "NIR_percent_change_without_2011"): 18.977060996649143,
    ("Qgross_depth", "Qgross_difference_due_to_2011"): 95.16192414811131,
    ("Qgross_depth", "Qgross_percent_change_without_2011"): 18.977061004736505,
}

# Values visible in manuscript Section 3.5 after its stated rounding.
MANUSCRIPT_PERCENT_CHANGE_1DP = {
    "ETa": [-31.9, -28.9, -17.4, -42.5],
    "NIR": [-79.8, -65.4, -44.2, -90.6],
    "Qgross_depth": [-79.8, -65.4, -44.2, -90.6],
}
MANUSCRIPT_DIFFERENCE_MM_0DP = {
    "ETa": [168, 170, 122, 216],
    "NIR": [209, 212, 180, 247],
    "Qgross_depth": [237, 241, 204, 281],
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
    return ffd.load_fixed_frame_inputs(PHASE3_ROOT, contract_path=CONTRACT_PATH)


@pytest.fixture(scope="session")
def derived_tables(loaded_inputs):
    sensitivity, objects, runtime_summary = loaded_inputs
    sensitivity_source = ffd.build_figure6_sensitivity_source(sensitivity)
    dominance_source = ffd.build_figure6_dominance_source(objects)
    statistics = ffd.build_text_statistics(sensitivity_source, objects)
    return sensitivity_source, dominance_source, statistics, runtime_summary


@pytest.fixture(scope="session")
def generated_outputs(
    tmp_path_factory: pytest.TempPathFactory,
) -> ffd.FixedFrameDominanceOutputs:
    output_dir = tmp_path_factory.mktemp("fixed_frame_dominance_outputs")
    return ffd.run_fixed_frame_dominance(
        PHASE3_ROOT,
        output_dir,
        contract_path=CONTRACT_PATH,
    )


def test_authoritative_code_and_schema_hashes() -> None:
    assert _sha256(SOURCE_PATH) == EXPECTED_MODULE_HASH
    assert _sha256(SCHEMA_PATH) == EXPECTED_SCHEMA_HASH


def test_authorized_input_hashes_are_frozen() -> None:
    for relative_path, expected_hash in EXPECTED_INPUT_HASHES.items():
        path = PHASE3_ROOT / relative_path
        assert path.is_file(), relative_path
        assert _sha256(path) == expected_hash


def test_input_validation_and_population_structure(loaded_inputs) -> None:
    sensitivity, objects, summary = loaded_inputs

    assert summary == {
        "report_version": "1.0.0",
        "modules": ["fixed_frame_dominance"],
        "status": "PASS",
        "checks": 115,
        "passed": 115,
        "warnings": 0,
        "failed": 0,
    }
    assert len(sensitivity) == 80
    assert sensitivity.duplicated(["county", "source_column"]).sum() == 0
    assert len(objects) == 1_184
    assert objects["object_id"].nunique() == 1_184
    assert set(objects["county"]) == set(ffd.EXPECTED_COUNTIES)
    assert objects["valid_cross_year_fraction_2011"].eq(1.0).all()
    assert objects["object_area_ha"].gt(0).all()


def test_input_structure_accepts_frozen_tables(loaded_inputs) -> None:
    sensitivity, objects, _ = loaded_inputs
    ffd._assert_input_structure(sensitivity, objects)


def test_figure6_selection_is_exact_and_preserves_signed_changes(derived_tables) -> None:
    source, _, _, _ = derived_tables

    assert len(source) == 12
    assert source.duplicated(["county", "metric"]).sum() == 0
    assert source["track"].eq("penalized_median").all()
    assert source.loc[source["metric"].isin(["ETa", "NIR"]), "efficiency_scenario"].eq(
        "NA"
    ).all()
    assert source.loc[source["metric"].eq("Qgross_depth"), "efficiency_scenario"].eq(
        "Ei088"
    ).all()
    assert source["percent_change_when_2011_removed"].lt(0).all()
    assert source["figure_changed"].eq(False).all()
    assert source["interpretation_boundary"].eq(ffd.INTERPRETATION_BOUNDARY).all()

    observed_pct = source.set_index(["county", "metric"])[
        "percent_change_when_2011_removed"
    ].to_dict()
    observed_delta = source.set_index(["county", "metric"])[
        "difference_4yr_minus_3yr"
    ].to_dict()
    assert observed_pct == pytest.approx(EXPECTED_PERCENT_CHANGE, abs=1e-12, rel=0)
    assert observed_delta == pytest.approx(EXPECTED_DIFFERENCE_MM, abs=1e-12, rel=0)


def test_figure6_accounting_identities_and_efficiency_proportionality(derived_tables) -> None:
    source, _, _, _ = derived_tables

    assert np.allclose(
        source["difference_4yr_minus_3yr"],
        source["full_4yr_mean"] - source["reduced_3yr_mean"],
        rtol=0,
        atol=2e-8,
    )
    expected_pct = 100.0 * (
        source["reduced_3yr_mean"] - source["full_4yr_mean"]
    ) / source["full_4yr_mean"]
    assert np.allclose(
        source["percent_change_when_2011_removed"],
        expected_pct,
        rtol=0,
        atol=1e-8,
    )

    nir = source[source["metric"].eq("NIR")].set_index("county")
    gross = source[source["metric"].eq("Qgross_depth")].set_index("county")
    for column in ("full_4yr_mean", "reduced_3yr_mean", "difference_4yr_minus_3yr"):
        assert np.allclose(gross[column], nir[column] / 0.88, rtol=0, atol=2e-6)
    assert np.allclose(
        gross["percent_change_when_2011_removed"],
        nir["percent_change_when_2011_removed"],
        rtol=0,
        atol=1e-8,
    )


def test_manuscript_section_3_5_rounding_concordance(derived_tables) -> None:
    """Confirm Phase 4 exact values reproduce manuscript Section 3.5 rounding."""

    source, dominance, _, _ = derived_tables
    county_order = list(ffd.EXPECTED_COUNTIES)
    for metric in ffd.EXPECTED_METRICS:
        frame = source[source["metric"].eq(metric)].set_index("county").loc[county_order]
        assert frame["percent_change_when_2011_removed"].round(1).tolist() == (
            MANUSCRIPT_PERCENT_CHANGE_1DP[metric]
        )
        assert frame["difference_4yr_minus_3yr"].round(0).astype(int).tolist() == (
            MANUSCRIPT_DIFFERENCE_MM_0DP[metric]
        )

    classes = dominance.set_index("dominance_class")
    strong = classes.loc["strong_2011_dominance"]
    low = classes.loc["low_2011_dominance"]
    insufficient = classes.loc["insufficient_data"]
    assert int(strong["n_objects"]) == 1_139
    assert int(low["n_objects"]) == 1
    assert int(insufficient["n_objects"]) == 44
    assert int(strong["classifiable_objects"]) == 1_140
    assert int(strong["total_fixed_frame_objects"]) == 1_184
    assert round(float(strong["percent_classifiable_objects"]), 1) == 99.9
    assert round(float(strong["percent_all_objects"]), 1) == 96.2


def test_dominance_source_reconciles_population_and_area(derived_tables, loaded_inputs) -> None:
    _, dominance, _, _ = derived_tables
    _, objects, _ = loaded_inputs

    assert dominance["dominance_class"].tolist() == [
        "strong_2011_dominance",
        "low_2011_dominance",
        "insufficient_data",
    ]
    counts = dominance.set_index("dominance_class")["n_objects"].to_dict()
    assert counts == {
        "strong_2011_dominance": 1_139,
        "low_2011_dominance": 1,
        "insufficient_data": 44,
    }
    assert int(dominance["n_objects"].sum()) == 1_184
    assert dominance["total_fixed_frame_objects"].eq(1_184).all()
    assert dominance["classifiable_objects"].eq(1_140).all()
    assert dominance["interpretation_boundary"].eq(ffd.INTERPRETATION_BOUNDARY).all()
    assert dominance["figure_changed"].eq(False).all()
    assert dominance["object_area_ha"].sum() == pytest.approx(
        objects["object_area_ha"].sum(), abs=1e-9
    )
    assert dominance["fixed_frame_area_ha"].eq(objects["object_area_ha"].sum()).all()


def test_text_statistics_have_explicit_units_and_sd_definitions(derived_tables) -> None:
    _, _, statistics, _ = derived_tables
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
    assert len(statistics) == 18
    assert statistics.duplicated(["statistic", "metric", "variable"]).sum() == 0
    assert statistics[list(required)].isna().sum().sum() == 0
    assert statistics["standard_deviation"].ge(0.0).all()
    assert statistics["interpretation_boundary"].eq(ffd.INTERPRETATION_BOUNDARY).all()

    county = statistics[statistics["observational_unit"].eq("county")]
    objects = statistics[statistics["observational_unit"].eq("fixed-frame object")]
    assert len(county) == 12
    assert county["n"].eq(4).all()
    assert county["ddof"].eq(1).all()
    assert county["reporting_role"].eq("primary text statistic").all()
    assert len(objects) == 6
    assert objects["n"].eq(1_140).all()
    assert objects["ddof"].eq(1).all()
    assert objects["reporting_role"].eq("secondary object-level diagnostic").all()

    county_sd = county.set_index(["metric", "variable"])["standard_deviation"].to_dict()
    object_sd = objects.set_index(["metric", "variable"])["standard_deviation"].to_dict()
    for key, expected in EXPECTED_COUNTY_SD.items():
        assert county_sd[key] == pytest.approx(expected, abs=1e-12)
    for key, expected in EXPECTED_OBJECT_SD.items():
        assert object_sd[key] == pytest.approx(expected, abs=1e-12)


def test_sample_sd_function_uses_ddof_one() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert ffd._sample_sd(values) == pytest.approx(values.std(ddof=1), abs=1e-15)
    assert ffd._sample_sd(pd.Series([1.0])) != ffd._sample_sd(pd.Series([1.0]))


def test_derived_validation_passes_all_30_checks(derived_tables, loaded_inputs) -> None:
    sensitivity_source, dominance_source, statistics, runtime_summary = derived_tables
    sensitivity, objects, _ = loaded_inputs
    report = ffd.validate_derived_outputs(
        sensitivity_source,
        dominance_source,
        statistics,
        sensitivity_input=sensitivity,
        object_input=objects,
        runtime_validation_summary=runtime_summary,
        contract_path=CONTRACT_PATH,
    )

    assert report["status"] == "PASS"
    assert report["checks"] == 30
    assert report["passed"] == 30
    assert report["warnings"] == 0
    assert report["failed"] == 0
    assert {item["check_id"] for item in report["check_results"]} == {
        f"FFD-{number:03d}" for number in range(1, 31)
    }
    assert all(item["status"] == "PASS" for item in report["check_results"])


def test_repository_wide_phase3_validation_remains_1708_of_1708() -> None:
    report = validate_phase3_root(PHASE3_ROOT, contract_path=CONTRACT_PATH)
    assert report.summary()["status"] == "PASS"
    assert report.summary()["checks"] == 1_708
    assert report.summary()["passed"] == 1_708
    assert report.summary()["warnings"] == 0
    assert report.summary()["failed"] == 0


def test_run_writes_exact_deterministic_outputs(generated_outputs) -> None:
    paths = {path.name: path for path in generated_outputs.paths()}
    assert set(paths) == set(ffd.EXPECTED_OUTPUT_NAMES)
    for name, expected_hash in EXPECTED_SCIENTIFIC_OUTPUT_HASHES.items():
        assert paths[name].is_file()
        assert paths[name].stat().st_size > 0
        assert _sha256(paths[name]) == expected_hash

    validation = json.loads(
        generated_outputs.fixed_frame_dominance_validation.read_text(encoding="utf-8")
    )
    assert validation["status"] == "PASS"
    assert validation["checks"] == validation["passed"] == 30
    assert validation["warnings"] == validation["failed"] == 0


def test_execution_record_reconciles_module_inputs_and_outputs(generated_outputs) -> None:
    record = json.loads(
        generated_outputs.fixed_frame_dominance_execution_record.read_text(
            encoding="utf-8"
        )
    )
    assert record["module"] == "fixed_frame_dominance"
    assert record["phase3_package"] == "PHASE3_CURATED_INPUT_LAYER_FINAL_002"
    assert record["module_relative_path"] == ffd.MODULE_RELATIVE_PATH
    assert record["module_sha256"] == EXPECTED_MODULE_HASH
    assert record["contract_sha256"] == _sha256(CONTRACT_PATH)
    assert record["runtime_input_validation"] == {
        "report_version": "1.0.0",
        "modules": ["fixed_frame_dominance"],
        "status": "PASS",
        "checks": 115,
        "passed": 115,
        "warnings": 0,
        "failed": 0,
    }
    assert record["derived_output_validation"] == {
        "status": "PASS",
        "checks": 30,
        "passed": 30,
        "warnings": 0,
        "failed": 0,
    }
    assert {
        path: item["sha256"] for path, item in record["authorized_inputs"].items()
    } == EXPECTED_INPUT_HASHES
    assert set(record["outputs"]) == set(EXPECTED_SCIENTIFIC_OUTPUT_HASHES)
    for name, metadata in record["outputs"].items():
        assert metadata["sha256"] == EXPECTED_SCIENTIFIC_OUTPUT_HASHES[name]
        assert metadata["size_bytes"] > 0
    assert record["figure_change"] is False
    assert record["scientific_boundary"] == ffd.INTERPRETATION_BOUNDARY


def test_second_execution_is_byte_deterministic(
    generated_outputs,
    tmp_path: Path,
) -> None:
    second = ffd.run_fixed_frame_dominance(
        PHASE3_ROOT,
        tmp_path / "second",
        contract_path=CONTRACT_PATH,
    )
    first_hashes = {path.name: _sha256(path) for path in generated_outputs.paths()}
    second_hashes = {path.name: _sha256(path) for path in second.paths()}
    assert second_hashes == first_hashes
    assert {name: first_hashes[name] for name in EXPECTED_SCIENTIFIC_OUTPUT_HASHES} == EXPECTED_SCIENTIFIC_OUTPUT_HASHES


def test_existing_outputs_require_explicit_overwrite(generated_outputs) -> None:
    output_dir = generated_outputs.figure6_fixed_frame_sensitivity_source.parent
    with pytest.raises(ffd.FixedFrameDominanceError, match="Refusing to overwrite"):
        ffd.run_fixed_frame_dominance(
            PHASE3_ROOT,
            output_dir,
            contract_path=CONTRACT_PATH,
        )


def test_explicit_overwrite_preserves_output_hashes(generated_outputs) -> None:
    output_dir = generated_outputs.figure6_fixed_frame_sensitivity_source.parent
    before = {path.name: _sha256(path) for path in generated_outputs.paths()}
    overwritten = ffd.run_fixed_frame_dominance(
        PHASE3_ROOT,
        output_dir,
        contract_path=CONTRACT_PATH,
        overwrite=True,
    )
    after = {path.name: _sha256(path) for path in overwritten.paths()}
    assert before == after
    assert {name: before[name] for name in EXPECTED_SCIENTIFIC_OUTPUT_HASHES} == EXPECTED_SCIENTIFIC_OUTPUT_HASHES


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate_sensitivity_key", "keys are not unique"),
        ("broken_sensitivity_difference", "difference identity failed"),
        ("broken_sensitivity_percent", "percentage-change identity failed"),
        ("missing_figure6_selection", "Expected 12 Figure 6 sensitivity rows"),
        ("duplicate_object_id", "Object identifiers are not unique"),
        ("nonpositive_object_area", "Object areas must be positive"),
        ("invalid_dominance_count", "Unexpected object dominance counts"),
        ("classifiable_null", "unexpected null dominance metrics"),
        ("broken_qgross_identity", "Qgross Ei=0.88 full-summary identity failed"),
    ],
)
def test_malformed_fixed_frame_tables_are_rejected(
    loaded_inputs,
    mutation: str,
    message: str,
) -> None:
    sensitivity, objects, _ = loaded_inputs
    sensitivity = sensitivity.copy(deep=True)
    objects = objects.copy(deep=True)

    if mutation == "duplicate_sensitivity_key":
        sensitivity.loc[1, ["county", "source_column"]] = sensitivity.loc[
            0, ["county", "source_column"]
        ].to_numpy()
    elif mutation == "broken_sensitivity_difference":
        sensitivity.loc[0, "difference_4yr_minus_3yr"] += 1.0
    elif mutation == "broken_sensitivity_percent":
        sensitivity.loc[0, "percent_change_when_2011_removed"] += 1.0
    elif mutation == "missing_figure6_selection":
        mask = (
            sensitivity["county"].eq("DALLAM")
            & sensitivity["metric"].eq("ETa")
            & sensitivity["track"].eq("penalized_median")
        )
        sensitivity.loc[mask, "track"] = "median"
    elif mutation == "duplicate_object_id":
        objects.loc[1, "object_id"] = objects.loc[0, "object_id"]
    elif mutation == "nonpositive_object_area":
        objects.loc[0, "object_area_ha"] = 0.0
    elif mutation == "invalid_dominance_count":
        index = objects.index[objects["dominance_class"].eq("strong_2011_dominance")][0]
        objects.loc[index, "dominance_class"] = "low_2011_dominance"
    elif mutation == "classifiable_null":
        index = objects.index[objects["dominance_class"].ne("insufficient_data")][0]
        objects.loc[index, "NIR_difference_due_to_2011"] = np.nan
    elif mutation == "broken_qgross_identity":
        index = objects.index[objects["dominance_class"].ne("insufficient_data")][0]
        full = "full_4yr_Qgross_penalized_Ei088_mean"
        reduced = "reduced_3yr_Qgross_penalized_Ei088_mean"
        delta = "Qgross_difference_due_to_2011"
        pct = "Qgross_percent_change_without_2011"
        objects.loc[index, full] += 1.0
        objects.loc[index, delta] = objects.loc[index, full] - objects.loc[index, reduced]
        objects.loc[index, pct] = 100.0 * (
            objects.loc[index, reduced] - objects.loc[index, full]
        ) / objects.loc[index, full]
    else:  # pragma: no cover - test parameter exhaustiveness guard.
        raise AssertionError(mutation)

    with pytest.raises(ffd.FixedFrameDominanceError, match=message):
        ffd._assert_input_structure(sensitivity, objects)


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
    assert ffd.main(args) == 0
    captured = capsys.readouterr()
    assert "Fixed-frame 2011-dominance descriptive statistics" in captured.out
    assert "Strong dominance: 1,139 of 1,184 objects" in captured.out
    assert captured.err == ""

    assert ffd.main(args) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "FIXED_FRAME_DOMINANCE_ERROR" in captured.err
    assert "Refusing to overwrite" in captured.err


def test_unauthorized_or_altered_input_is_rejected(tmp_path: Path) -> None:
    fake_root = tmp_path / "PHASE3_CURATED_INPUT_LAYER_FINAL_002"
    target = fake_root / ffd.SENSITIVITY_RELATIVE_PATH
    target.parent.mkdir(parents=True)
    target.write_bytes((PHASE3_ROOT / ffd.SENSITIVITY_RELATIVE_PATH).read_bytes() + b"\n")
    other = fake_root / ffd.OBJECT_RELATIVE_PATH
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_bytes((PHASE3_ROOT / ffd.OBJECT_RELATIVE_PATH).read_bytes())

    with pytest.raises(RuntimeInputError):
        ffd.load_fixed_frame_inputs(fake_root, contract_path=CONTRACT_PATH)


def test_source_contains_no_notebook_or_prohibited_runtime_dependency() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    prohibited = (
        ".ipynb",
        "/scratch/",
        "AWM_figures",
        "V46D",
        "V48",
        "V49",
        "exec(",
        "papermill",
        "nbconvert",
    )
    assert {token for token in prohibited if token in source} == set()
