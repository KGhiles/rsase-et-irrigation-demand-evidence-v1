"""Regression and scientific-boundary tests for physical coherence.

The frozen V46C-derived Phase 3 tables are the exclusive analytical inputs.
The compact Figure 3 raster is metadata/visualization-only. Manuscript values
are checked only as rounded release-concordance invariants and never enter the
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
import rasterio


PHASE4_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PHASE4_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qstable_awm import physical_coherence as pc  # noqa: E402
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
MODULE_PATH = PHASE4_ROOT / "src" / "qstable_awm" / "physical_coherence.py"

EXPECTED_MODULE_SHA256 = (
    "ec36da70eb15b29296c1501da5c531b5e84fb92f5b355b885ce85f54858d7e05"
)
EXPECTED_INPUT_HASHES = {
    pc.FINGERPRINT_RELATIVE_PATH: (
        "dfde37a117769aae69a79e89ebb8614a48236669cd6be36d1a2fdfe005e139ac"
    ),
    pc.SPATIAL_RELATIVE_PATH: (
        "8b64dd0d493f804764c2b889627023e3c904b16cde7a06d76a48dd3f9585c5eb"
    ),
    pc.SCENE_SUMMARY_RELATIVE_PATH: (
        "a604df5e443e0caacc210a91d4d7605fee5bbab0426904f37d8b0c882ef4f6f3"
    ),
    pc.RASTER_RELATIVE_PATH: (
        "a2bb65acf377f4d5bc3836563a654e71b4ec612a2ccb4ef46254f49191fdaea8"
    ),
}
EXPECTED_STABLE_OUTPUT_HASHES = {
    "figure3_scene_fingerprint_source.csv": (
        "eb032a4124a41adc8eca5988a49fb3f40b25cd12979b9952307fbd08b0c73c3d"
    ),
    "figure3_mechanism_source.csv": (
        "4fb159643008254bdf3826e6bab0458c2fa19c0b53eb71906077fb66de37910e"
    ),
    "figure3_spatial_coherence_source.csv": (
        "416b1ef601c2544ea92e64308f983b45e42f16b90a2a1428acfe6bb1edc67862"
    ),
    "physical_coherence_text_statistics.csv": (
        "0a0637fe9f0c23740369edba347805c933e51d3121684a5848d8ad901625583b"
    ),
    "physical_coherence_validation.json": (
        "727ed643ce8ba2a9d78c4d69ca38414e83369ff452c155e543168fc24a449d7f"
    ),
}
EXPECTED_OUTPUT_NAMES = set(EXPECTED_STABLE_OUTPUT_HASHES) | {
    "physical_coherence_execution_record.json"
}
EXPECTED_FINGERPRINT = {
    "ETf p50": {
        "passes": 7,
        "median": 0.915156215429306,
        "mean": 0.9015979059040546,
        "sd": 0.0380514305905953,
    },
    "NDVI p50": {
        "passes": 8,
        "median": 0.7792706787586212,
        "mean": 0.7523385584354401,
        "sd": 0.0542760961717421,
    },
    "LAI p50": {
        "passes": 8,
        "median": 4.075624227523804,
        "mean": 3.914031356573105,
        "sd": 0.325656599949323,
    },
    "H p50": {
        "passes": 7,
        "median": -14.287691771984104,
        "mean": -8.80597822368145,
        "sd": 37.75574603797784,
    },
    "LE/Rn p50": {
        "passes": 5,
        "median": 0.953746661543846,
        "mean": 0.9420452937483788,
        "sd": 0.0733287616090067,
    },
    "Ts-position p50": {
        "passes": 6,
        "median": 0.1185448244214057,
        "mean": 0.1330025200732051,
        "sd": 0.0461240026654041,
    },
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
    return pc.load_physical_coherence_inputs(
        PHASE3_ROOT, contract_path=CONTRACT_PATH
    )


@pytest.fixture(scope="session")
def derived_tables(loaded_inputs):
    fingerprint, spatial, scenes, raster_path, runtime = loaded_inputs
    pc._assert_input_structure(fingerprint, spatial, scenes, raster_path)
    fingerprint_source = pc.build_scene_fingerprint_source(fingerprint, scenes)
    mechanism_source = pc.build_mechanism_source(scenes)
    spatial_source = pc.build_spatial_coherence_source(spatial)
    statistics = pc.build_text_statistics(
        fingerprint_source, scenes, spatial_source
    )
    return fingerprint_source, mechanism_source, spatial_source, statistics, runtime


@pytest.fixture(scope="session")
def generated_outputs(tmp_path_factory: pytest.TempPathFactory):
    output_dir = tmp_path_factory.mktemp("physical_coherence_outputs")
    return pc.run_physical_coherence(
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
    assert contract.allowed_outputs(pc.MODULE_NAME) == pc.EXPECTED_OUTPUT_NAMES
    assert set(contract.allowed_outputs(pc.MODULE_NAME)) == EXPECTED_OUTPUT_NAMES
    assert len(contract.files_for_module(pc.MODULE_NAME)) == 4


def test_runtime_input_validation_passes_176_checks(loaded_inputs) -> None:
    *_, runtime = loaded_inputs
    assert runtime["status"] == "PASS"
    assert runtime["checks"] == runtime["passed"] == 176
    assert runtime["warnings"] == runtime["failed"] == 0
    assert runtime["modules"] == ["physical_coherence"]


def test_frozen_input_structure_and_scene_classes(loaded_inputs) -> None:
    fingerprint, spatial, scenes, raster_path, _ = loaded_inputs
    pc._assert_input_structure(fingerprint, spatial, scenes, raster_path)
    assert len(fingerprint) == 6
    assert len(spatial) == 8
    assert len(scenes) == 13
    assert scenes["raster_mechanism_class"].value_counts().to_dict() == {
        pc.ACTIVE_CLASS: 8,
        pc.AMBIGUOUS_CLASS: 2,
        pc.CONTROL_CLASS: 2,
        pc.OCTOBER_CLASS: 1,
    }
    active = scenes[scenes["raster_mechanism_class"].eq(pc.ACTIVE_CLASS)]
    assert active["year"].eq(2011).all()
    assert set(spatial["filename"]) == set(active["filename"])


def test_compact_raster_metadata_contract(loaded_inputs) -> None:
    raster_path = loaded_inputs[3]
    with rasterio.open(raster_path) as dataset:
        assert dataset.driver == "GTiff"
        assert dataset.count == 8
        assert dataset.crs.to_string() == "EPSG:32614"
        assert dataset.res == pytest.approx((90.0, 90.0))
        assert dataset.dtypes == ("float32",) * 8
        assert dataset.descriptions == (
            "ETrF",
            "NDVI",
            "LAI",
            "H_W_m2",
            "LE_over_Rn",
            "Ts_position",
            "accepted_corn_mask",
            "mechanism_mask",
        )


def test_frozen_fingerprint_matches_expected_counts_and_medians(loaded_inputs) -> None:
    fingerprint = loaded_inputs[0].set_index("metric")
    assert set(fingerprint.index) == set(EXPECTED_FINGERPRINT)
    for metric, expected in EXPECTED_FINGERPRINT.items():
        row = fingerprint.loc[metric]
        assert int(row["n_total"]) == 8
        assert int(row["n_pass"]) == expected["passes"]
        assert float(row["median"]) == pytest.approx(expected["median"], abs=1e-15)
        assert float(row["pass_pct"]) == pytest.approx(
            100.0 * expected["passes"] / 8.0, abs=1e-12
        )


def test_recomputed_fingerprint_and_scene_sd(derived_tables) -> None:
    fingerprint_source = derived_tables[0].set_index("metric")
    assert len(fingerprint_source) == 6
    for metric, expected in EXPECTED_FINGERPRINT.items():
        row = fingerprint_source.loc[metric]
        assert int(row["n_total"]) == 8
        assert int(row["n_pass"]) == expected["passes"]
        assert row["median"] == pytest.approx(expected["median"], abs=1e-15)
        assert row["mean"] == pytest.approx(expected["mean"], abs=1e-15)
        assert row["sample_sd"] == pytest.approx(expected["sd"], abs=1e-15)
        assert row["sd_basis"] == (
            "sample SD across eight active-canopy 2011 scenes"
        )


def test_manuscript_section_3_2_rounding_concordance(derived_tables) -> None:
    source = derived_tables[0].set_index("metric")
    assert round(source.loc["ETf p50", "median"], 3) == 0.915
    assert round(source.loc["NDVI p50", "median"], 3) == 0.779
    assert round(source.loc["LAI p50", "median"], 2) == 4.08
    assert round(source.loc["LE/Rn p50", "median"], 3) == 0.954
    assert round(source.loc["Ts-position p50", "median"], 3) == 0.119
    assert source["n_pass"].to_dict() == {
        metric: values["passes"] for metric, values in EXPECTED_FINGERPRINT.items()
    }


def test_active_mechanism_fraction_matches_manuscript_range(loaded_inputs) -> None:
    scenes = loaded_inputs[2]
    active = scenes[scenes["raster_mechanism_class"].eq(pc.ACTIVE_CLASS)]
    fraction = active["active_canopy_high_mechanism_frac"]
    assert round(100.0 * fraction.min(), 1) == 41.2
    assert round(100.0 * fraction.max(), 1) == 82.3
    minimum = active.loc[fraction.idxmin()]
    assert minimum["county"] == "OCHILTREE"
    assert int(minimum["month"]) == 6


def test_ambiguous_and_control_active_mechanism_fractions_remain_below_half_percent(
    loaded_inputs,
) -> None:
    scenes = loaded_inputs[2]
    subset = scenes[
        scenes["raster_mechanism_class"].isin(
            [pc.AMBIGUOUS_CLASS, pc.CONTROL_CLASS]
        )
    ]
    assert len(subset) == 4
    assert subset["active_canopy_high_mechanism_frac"].lt(0.005).all()


def test_mechanism_source_preserves_plot_classes_and_boundaries(derived_tables) -> None:
    source = derived_tables[1]
    assert len(source) == 12
    assert source.duplicated(["county", "scene_id"]).sum() == 0
    assert int(source["is_active_2011_mechanism"].sum()) == 8
    assert source["plot_class_order"].is_monotonic_increasing
    assert source["mechanism_fraction_pct"].equals(
        100.0 * source["active_canopy_high_mechanism_frac"]
    )
    assert source["interpretation_boundary"].str.contains(
        "not ET validation", regex=False
    ).all()


def test_spatial_source_preserves_full_resolution_authority(derived_tables) -> None:
    source = derived_tables[2]
    assert len(source) == 8
    assert source["total_area_ha"].is_monotonic_increasing
    assert np.allclose(
        source["largest_component_share_pct"],
        100.0 * source["largest_component_frac_of_mask"],
        rtol=0,
        atol=1e-12,
    )
    assert source["interpretation_boundary"].str.contains(
        "frozen V46C full-resolution results", regex=False
    ).all()
    assert source["interpretation_boundary"].str.contains(
        "compact raster not used", regex=False
    ).all()


def test_text_statistics_have_required_explicit_definitions(derived_tables) -> None:
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
    assert len(statistics) == 13
    assert statistics["statistic"].is_unique
    assert statistics["observational_unit"].eq(
        "targeted active-canopy scene"
    ).all()
    assert statistics["n"].eq(8).all()
    assert statistics["ddof"].eq(1).all()
    assert statistics["standard_deviation"].ge(0).all()
    assert np.allclose(
        statistics["standard_deviation"],
        statistics["sample_sd"],
        rtol=0,
        atol=0,
    )
    assert statistics["interpretation_boundary"].eq(
        pc.INTERPRETATION_BOUNDARY
    ).all()


def test_text_statistics_reproduce_coauthor_requested_sd_values(derived_tables) -> None:
    statistics = derived_tables[3]
    fingerprint = statistics[
        statistics["statistic_group"].eq("active_scene_fingerprint")
    ].set_index("metric")
    for metric, expected in EXPECTED_FINGERPRINT.items():
        row = fingerprint.loc[metric]
        assert row["mean"] == pytest.approx(expected["mean"], abs=1e-15)
        assert row["standard_deviation"] == pytest.approx(
            expected["sd"], abs=1e-15
        )


def test_area_and_spatial_text_statistics_use_scene_as_unit(derived_tables) -> None:
    statistics = derived_tables[3]
    area = statistics.set_index("metric")
    assert area.loc["active_canopy_high_mechanism_area_ha", "mean"] == pytest.approx(
        6981.80625
    )
    assert area.loc["active_canopy_high_mechanism_area_ha", "standard_deviation"] == pytest.approx(
        7103.859628429237
    )
    assert area.loc["component_count", "mean"] == pytest.approx(788.125)
    assert area.loc["component_count", "standard_deviation"] == pytest.approx(
        589.066680072323
    )


def test_derived_validation_passes_all_ten_checks(derived_tables) -> None:
    fingerprint, mechanism, spatial, statistics, runtime = derived_tables
    report = pc.validate_derived_outputs(
        fingerprint,
        mechanism,
        spatial,
        statistics,
        runtime_validation_summary=runtime,
    )
    assert report["status"] == "PASS"
    assert report["checks"] == report["passed"] == 10
    assert report["failed"] == 0
    assert {item["check_id"] for item in report["check_results"]} == {
        f"PHYS-{number:03d}" for number in range(1, 11)
    }
    assert report["standard_deviation"]["ddof"] == 1
    assert report["figure_change"] is False


def test_run_writes_stable_scientific_outputs(generated_outputs) -> None:
    paths = {path.name: path for path in generated_outputs.paths()}
    assert set(paths) == EXPECTED_OUTPUT_NAMES
    for name, expected_hash in EXPECTED_STABLE_OUTPUT_HASHES.items():
        assert paths[name].is_file()
        assert paths[name].stat().st_size > 0
        assert _sha256(paths[name]) == expected_hash
    validation = json.loads(
        generated_outputs.physical_coherence_validation.read_text(encoding="utf-8")
    )
    assert validation["status"] == "PASS"
    assert validation["checks"] == validation["passed"] == 10
    assert validation["failed"] == 0


def test_execution_record_reconciles_current_contract_and_outputs(
    generated_outputs,
) -> None:
    record = json.loads(
        generated_outputs.physical_coherence_execution_record.read_text(
            encoding="utf-8"
        )
    )
    assert record["module"] == pc.MODULE_NAME
    assert record["module_relative_path"] == pc.MODULE_RELATIVE_PATH
    assert record["module_sha256"] == _sha256(MODULE_PATH)
    assert record["contract_sha256"] == _sha256(CONTRACT_PATH)
    assert record["phase3_package"] == "PHASE3_CURATED_INPUT_LAYER_FINAL_002"
    assert record["authorized_inputs"] == {
        path: {"sha256": digest} for path, digest in EXPECTED_INPUT_HASHES.items()
    }
    assert record["runtime_input_validation"]["passed"] == 176
    assert record["derived_output_validation"] == {
        "status": "PASS",
        "checks": 10,
        "passed": 10,
        "warnings": 0,
        "failed": 0,
    }
    assert "metadata" in record["compact_raster_boundary"]
    assert "no authoritative area" in record["compact_raster_boundary"]
    for name, item in record["outputs"].items():
        path = generated_outputs.physical_coherence_execution_record.parent / name
        assert item["size_bytes"] == path.stat().st_size
        assert item["sha256"] == _sha256(path)


def test_second_execution_is_byte_deterministic(generated_outputs, tmp_path: Path) -> None:
    second = pc.run_physical_coherence(
        PHASE3_ROOT,
        tmp_path / "second",
        contract_path=CONTRACT_PATH,
    )
    first_hashes = {path.name: _sha256(path) for path in generated_outputs.paths()}
    second_hashes = {path.name: _sha256(path) for path in second.paths()}
    assert second_hashes == first_hashes


def test_existing_outputs_require_explicit_overwrite(generated_outputs) -> None:
    output_dir = generated_outputs.physical_coherence_validation.parent
    with pytest.raises(pc.PhysicalCoherenceError, match="Refusing to overwrite"):
        pc.run_physical_coherence(
            PHASE3_ROOT,
            output_dir,
            contract_path=CONTRACT_PATH,
        )


def test_explicit_overwrite_is_byte_deterministic(generated_outputs) -> None:
    output_dir = generated_outputs.physical_coherence_validation.parent
    before = {path.name: _sha256(path) for path in generated_outputs.paths()}
    overwritten = pc.run_physical_coherence(
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
        ("class_count", "Unexpected scene-class counts"),
        ("active_year", "must be 2011"),
        ("fingerprint_metric", "six-metric definition"),
        ("spatial_filename", "do not match"),
    ],
)
def test_malformed_physical_tables_are_rejected(
    loaded_inputs,
    mutation: str,
    message: str,
) -> None:
    fingerprint, spatial, scenes, raster_path, _ = loaded_inputs
    fingerprint = fingerprint.copy(deep=True)
    spatial = spatial.copy(deep=True)
    scenes = scenes.copy(deep=True)
    if mutation == "class_count":
        scenes.loc[
            scenes["raster_mechanism_class"].eq(pc.ACTIVE_CLASS).idxmax(),
            "raster_mechanism_class",
        ] = pc.CONTROL_CLASS
    elif mutation == "active_year":
        row = scenes["raster_mechanism_class"].eq(pc.ACTIVE_CLASS).idxmax()
        scenes.loc[row, "year"] = 2015
    elif mutation == "fingerprint_metric":
        fingerprint.loc[0, "metric"] = "UNAUTHORIZED"
    elif mutation == "spatial_filename":
        spatial.loc[0, "filename"] = "unexpected.tif"
    else:  # pragma: no cover
        raise AssertionError(mutation)
    with pytest.raises(pc.PhysicalCoherenceError, match=message):
        pc._assert_input_structure(fingerprint, spatial, scenes, raster_path)


def test_compact_raster_band_change_is_rejected(loaded_inputs, tmp_path: Path) -> None:
    fingerprint, spatial, scenes, raster_path, _ = loaded_inputs
    altered = tmp_path / "altered.tif"
    shutil.copy2(raster_path, altered)
    with rasterio.open(altered, "r+") as dataset:
        dataset.set_band_description(1, "ALTERED")
    with pytest.raises(pc.PhysicalCoherenceError, match="raster contract changed"):
        pc._assert_input_structure(fingerprint, spatial, scenes, altered)


def test_altered_authorized_input_is_rejected(tmp_path: Path) -> None:
    fake_root = tmp_path / "PHASE3_CURATED_INPUT_LAYER_FINAL_002"
    for relative_path in EXPECTED_INPUT_HASHES:
        target = fake_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((PHASE3_ROOT / relative_path).read_bytes())
    altered = fake_root / pc.FINGERPRINT_RELATIVE_PATH
    altered.write_bytes(altered.read_bytes() + b"\n")
    with pytest.raises(RuntimeInputError):
        pc.load_physical_coherence_inputs(fake_root, contract_path=CONTRACT_PATH)


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
    assert pc.main(args) == 0
    captured = capsys.readouterr()
    assert "Physical-coherence scene spread" in captured.out
    assert "Figure 3 design changed: no" in captured.out
    assert captured.err == ""

    assert pc.main(args) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "PHYSICAL_COHERENCE_ERROR" in captured.err
    assert "Refusing to overwrite" in captured.err


def test_source_has_no_notebook_or_compact_raster_analytical_read() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    prohibited = (
        ".ipynb",
        "/scratch/",
        "AWM_figures",
        "V46D",
        "V48",
        "V49",
        "papermill",
        "nbconvert",
        "ds.read(",
        "dataset.read(",
    )
    assert {token for token in prohibited if token in source} == set()
    assert "rasterio.open" in source
    assert "compact raster not used" in source
    assert "validate_module_inputs" in source


def test_complete_phase3_layer_still_passes_1708_checks() -> None:
    report = validate_phase3_root(
        PHASE3_ROOT,
        contract_path=CONTRACT_PATH,
    )
    report.raise_for_errors()
    summary = report.summary()
    assert summary["status"] == "PASS"
    assert summary["checks"] == summary["passed"] == 1708
    assert summary["warnings"] == summary["failed"] == 0
