"""Integrity and scientific-boundary tests for :mod:`qstable_awm.schemas`.

The schema registry is the immutable structural authority used by Phase 4
runtime validation. These tests do not read manuscript values or historical
notebooks. They lock registry identity, exact runtime paths, corrected signed
ranges, key definitions, JSON/raster contracts, lookup behavior, immutability,
and deterministic failure behavior for malformed registries.
"""

from __future__ import annotations

import ast
import hashlib
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import MappingProxyType

import pytest


PHASE4_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PHASE4_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qstable_awm import schemas as s  # noqa: E402


SCHEMAS_PATH = PHASE4_ROOT / "src" / "qstable_awm" / "schemas.py"
EXPECTED_SCHEMAS_SHA256 = (
    "27e7dbef16f8cc52eedc81c18f0662fe49c97eb51e0d2a42e9675a0f78125471"
)
EXPECTED_TABLE_NAMES = (
    "monthly_climate_context",
    "seasonal_climate_context",
    "active_scene_fingerprint",
    "spatial_coherence_figure_source",
    "physical_coherence_scene_summary",
    "aerodynamic_scope_summary",
    "aerodynamic_etrf_scenario_envelopes",
    "aerodynamic_sensible_heat_scenario_envelopes",
    "aerodynamic_peak_county_month_sensitivity",
    "aerodynamic_table3_summary",
    "fixed_frame_object_year_crop_support",
    "county_year_crop_support",
    "domain_year_crop_support",
    "four_year_vs_non2011_sensitivity",
    "object_2011_dominance",
    "strict_mask_by_raster",
    "strict_county_raster_counts",
    "county_radius_response_summary",
    "strict_processing_funnel",
    "known_cold_solver_parameters",
    "interpretation_classification",
)
EXPECTED_JSON_NAMES = (
    "interpretation_classes",
    "non2011_hls_temporal_support_evidence",
)
EXPECTED_RASTER_NAMES = ("representative_scene_plot_ready",)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_authoritative_schema_file_hash() -> None:
    assert SCHEMAS_PATH.is_file()
    assert _sha256(SCHEMAS_PATH) == EXPECTED_SCHEMAS_SHA256


def test_frozen_package_and_study_constants() -> None:
    assert s.PHASE3_PACKAGE_NAME == "PHASE3_CURATED_INPUT_LAYER_FINAL_002"
    assert s.STUDY_COUNTIES == ("DALLAM", "HARTLEY", "MOORE", "OCHILTREE")
    assert s.STUDY_YEARS == (2011, 2015, 2017, 2023)
    assert s.COMPARISON_YEARS == (2015, 2017, 2023)


def test_registry_counts_and_order_are_frozen() -> None:
    assert tuple(s.TABLE_SCHEMAS) == EXPECTED_TABLE_NAMES
    assert tuple(s.JSON_SCHEMAS) == EXPECTED_JSON_NAMES
    assert tuple(s.RASTER_SCHEMAS) == EXPECTED_RASTER_NAMES
    assert len(s.TABLE_SCHEMAS) == 21
    assert len(s.JSON_SCHEMAS) == 2
    assert len(s.RASTER_SCHEMAS) == 1
    assert len(s.ALL_RUNTIME_PATHS) == 24


def test_runtime_paths_are_exact_unique_relative_structured_inputs() -> None:
    expected = tuple(
        schema.relative_path
        for registry in (s.TABLE_SCHEMAS, s.JSON_SCHEMAS, s.RASTER_SCHEMAS)
        for schema in registry.values()
    )
    assert s.ALL_RUNTIME_PATHS == expected
    assert len(s.ALL_RUNTIME_PATHS) == len(set(s.ALL_RUNTIME_PATHS))
    for relative_path in s.ALL_RUNTIME_PATHS:
        path = Path(relative_path)
        assert not path.is_absolute()
        assert ".." not in path.parts
        assert path.parts[:2] == ("data", "derived")
        assert path.suffix.lower() in {".csv", ".json", ".tif"}
        assert ".ipynb" not in relative_path


def test_schema_registry_integrity_passes() -> None:
    s.assert_schema_registry_integrity()


@pytest.mark.parametrize("name", EXPECTED_TABLE_NAMES)
def test_table_lookup_round_trip(name: str) -> None:
    schema = s.table_schema(name)
    assert schema is s.TABLE_SCHEMAS[name]
    assert s.schema_for_relative_path(schema.relative_path) is schema


@pytest.mark.parametrize("name", EXPECTED_JSON_NAMES)
def test_json_lookup_round_trip(name: str) -> None:
    schema = s.json_schema(name)
    assert schema is s.JSON_SCHEMAS[name]
    assert s.schema_for_relative_path(schema.relative_path) is schema


@pytest.mark.parametrize("name", EXPECTED_RASTER_NAMES)
def test_raster_lookup_round_trip(name: str) -> None:
    schema = s.raster_schema(name)
    assert schema is s.RASTER_SCHEMAS[name]
    assert s.schema_for_relative_path(schema.relative_path) is schema


def test_unknown_lookup_errors_are_descriptive() -> None:
    with pytest.raises(KeyError, match="Unknown Phase 4 table schema"):
        s.table_schema("not_registered")
    with pytest.raises(KeyError, match="Unknown Phase 4 JSON schema"):
        s.json_schema("not_registered")
    with pytest.raises(KeyError, match="Unknown Phase 4 raster schema"):
        s.raster_schema("not_registered")
    with pytest.raises(KeyError, match="No Phase 4 schema registered"):
        s.schema_for_relative_path("data/derived/not_registered.csv")


def test_registry_mappings_are_immutable() -> None:
    with pytest.raises(TypeError):
        s.TABLE_SCHEMAS["new"] = next(iter(s.TABLE_SCHEMAS.values()))  # type: ignore[index]
    with pytest.raises(TypeError):
        s.JSON_SCHEMAS["new"] = next(iter(s.JSON_SCHEMAS.values()))  # type: ignore[index]
    with pytest.raises(TypeError):
        s.RASTER_SCHEMAS["new"] = next(iter(s.RASTER_SCHEMAS.values()))  # type: ignore[index]


def test_schema_dataclasses_are_frozen() -> None:
    table = s.TABLE_SCHEMAS["seasonal_climate_context"]
    json_schema = s.JSON_SCHEMAS["interpretation_classes"]
    raster = s.RASTER_SCHEMAS["representative_scene_plot_ready"]
    rule = table.column_rules["year"]
    with pytest.raises(FrozenInstanceError):
        table.expected_rows = 0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        json_schema.exact_top_level_keys = False  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        raster.band_count = 0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        rule.nullable = True  # type: ignore[misc]


def test_table_properties_and_rule_coverage() -> None:
    for name, schema in s.TABLE_SCHEMAS.items():
        assert schema.name == name
        assert schema.required_columns == schema.exact_columns
        assert schema.expected_rows > 0
        assert schema.exact_columns
        assert set(schema.primary_key).issubset(schema.exact_columns)
        assert set(schema.column_rules) == set(schema.exact_columns)
        assert schema.nullable_columns == tuple(
            column
            for column in schema.exact_columns
            if schema.column_rules[column].nullable
        )
        for rule in schema.column_rules.values():
            assert isinstance(rule.kind, s.ScalarKind)
            if rule.min_value is not None and rule.max_value is not None:
                assert rule.min_value <= rule.max_value


def test_corrected_aerodynamic_signed_latent_energy_range() -> None:
    schema = s.TABLE_SCHEMAS["aerodynamic_table3_summary"]
    signed = schema.column_rules["alt_LE_fraction_minus_BASE"]
    absolute = schema.column_rules["abs_alt_LE_fraction_minus_BASE"]
    assert signed.kind is s.ScalarKind.FLOAT
    assert (signed.min_value, signed.max_value) == (-1.0, 1.0)
    assert (absolute.min_value, absolute.max_value) == (0.0, 1.0)


def test_corrected_fixed_frame_unique_key_and_signed_percent_range() -> None:
    schema = s.TABLE_SCHEMAS["four_year_vs_non2011_sensitivity"]
    assert schema.primary_key == ("county", "source_column")
    rule = schema.column_rules["percent_change_when_2011_removed"]
    assert rule.kind is s.ScalarKind.FLOAT
    assert (rule.min_value, rule.max_value) == (-100.0, 100.0)


def test_object_dominance_signed_percent_ranges() -> None:
    schema = s.TABLE_SCHEMAS["object_2011_dominance"]
    for column in (
        "ETa_percent_change_without_2011",
        "NIR_percent_change_without_2011",
        "Qgross_percent_change_without_2011",
    ):
        rule = schema.column_rules[column]
        assert rule.kind is s.ScalarKind.FLOAT
        assert rule.nullable is True
        assert (rule.min_value, rule.max_value) == (-100.0, 100.0)


def test_crop_support_fraction_and_year_boundaries() -> None:
    object_schema = s.TABLE_SCHEMAS["fixed_frame_object_year_crop_support"]
    for column in (
        "corn_fraction_raw",
        "corn_fraction",
        "supported_corn_fraction_raw",
        "supported_corn_fraction",
        "valid_cross_year_fraction",
        "analysis_valid_fraction",
    ):
        rule = object_schema.column_rules[column]
        assert (rule.min_value, rule.max_value) == (0.0, 1.0)
    assert object_schema.column_rules["year"].allowed_values == s.STUDY_YEARS
    assert object_schema.column_rules["county"].allowed_values == s.STUDY_COUNTIES


def test_known_cold_radius_and_numerical_stability_ranges() -> None:
    summary = s.TABLE_SCHEMAS["county_radius_response_summary"]
    assert summary.column_rules["radius_m"].allowed_values == (1000, 3000)
    for column in (
        "nonconverged_fraction_valid",
        "high_ETf_clamp_fraction_valid",
        "low_ETf_clamp_fraction_valid",
        "rah_cap_hit_fraction_valid",
    ):
        rule = summary.column_rules[column]
        assert (rule.min_value, rule.max_value) == (0.0, 1.0)


def test_interpretation_classification_boundary_values() -> None:
    schema = s.TABLE_SCHEMAS["interpretation_classification"]
    assert schema.expected_rows == 4
    assert schema.primary_key == ("analysis_condition",)
    assert schema.column_rules["final_class"].allowed_values == (
        "Class II",
        "Class III",
        "Class IV",
        "Class I candidate",
    )
    assert "Class I" not in schema.column_rules["final_class"].allowed_values


def test_json_contracts_have_exact_required_top_level_keys() -> None:
    figure8 = s.JSON_SCHEMAS["interpretation_classes"]
    hls = s.JSON_SCHEMAS["non2011_hls_temporal_support_evidence"]
    assert figure8.exact_top_level_keys is True
    assert hls.exact_top_level_keys is True
    assert figure8.required_top_level_keys == (
        "version",
        "figure",
        "scientific_boundary",
        "release_decision",
        "authority",
        "evidence_dimensions",
        "class_definitions",
        "matrix_rows",
    )
    assert "completed_class_I_assignment" in hls.required_top_level_keys
    assert "hls_not_used_for" in hls.required_top_level_keys
    assert "source_notebook_sha256" in hls.required_top_level_keys


def test_compact_raster_contract_is_visualization_only() -> None:
    raster = s.RASTER_SCHEMAS["representative_scene_plot_ready"]
    assert raster.driver == "GTiff"
    assert raster.band_count == 8
    assert raster.band_descriptions == (
        "ETrF",
        "NDVI",
        "LAI",
        "H_W_m2",
        "LE_over_Rn",
        "Ts_position",
        "accepted_corn_mask",
        "mechanism_mask",
    )
    assert raster.crs == "EPSG:32614"
    assert raster.pixel_size_x == raster.pixel_size_y == 90.0
    assert raster.dtype == "float32"
    assert "Visualization input only" in raster.use_boundary
    assert "analytical" in raster.use_boundary.lower()


def test_schema_source_is_metadata_only_and_has_no_runtime_data_library() -> None:
    source = SCHEMAS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    assert "pandas" not in imported_roots
    assert "rasterio" not in imported_roots
    assert "/scratch/" not in source
    assert "read_csv(" not in source
    assert "assert_schema_registry_integrity()" in source


def test_duplicate_runtime_path_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    duplicate = s.ALL_RUNTIME_PATHS + (s.ALL_RUNTIME_PATHS[0],)
    monkeypatch.setattr(s, "ALL_RUNTIME_PATHS", duplicate)
    with pytest.raises(RuntimeError, match="Duplicate runtime paths"):
        s.assert_schema_registry_integrity()


def test_table_registry_key_name_mismatch_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = s.TABLE_SCHEMAS["seasonal_climate_context"]
    monkeypatch.setattr(s, "TABLE_SCHEMAS", {"wrong_name": original})
    monkeypatch.setattr(s, "ALL_RUNTIME_PATHS", (original.relative_path,))
    with pytest.raises(RuntimeError, match="Table registry key/name mismatch"):
        s.assert_schema_registry_integrity()


def test_table_without_columns_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    original = s.TABLE_SCHEMAS["seasonal_climate_context"]
    malformed = replace(
        original,
        exact_columns=(),
        primary_key=(),
        column_rules=MappingProxyType({}),
    )
    monkeypatch.setattr(s, "TABLE_SCHEMAS", {malformed.name: malformed})
    monkeypatch.setattr(s, "ALL_RUNTIME_PATHS", (malformed.relative_path,))
    with pytest.raises(RuntimeError, match="Table schema has no columns"):
        s.assert_schema_registry_integrity()


def test_missing_primary_key_column_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = s.TABLE_SCHEMAS["seasonal_climate_context"]
    malformed = replace(original, primary_key=("not_a_column",))
    monkeypatch.setattr(s, "TABLE_SCHEMAS", {malformed.name: malformed})
    monkeypatch.setattr(s, "ALL_RUNTIME_PATHS", (malformed.relative_path,))
    with pytest.raises(RuntimeError, match="Primary-key columns missing"):
        s.assert_schema_registry_integrity()


def test_missing_or_extra_column_rule_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = s.TABLE_SCHEMAS["seasonal_climate_context"]
    rules = dict(original.column_rules)
    rules.pop(original.exact_columns[0])
    rules["not_a_column"] = s.ColumnRule(kind=s.ScalarKind.STRING)
    malformed = replace(original, column_rules=MappingProxyType(rules))
    monkeypatch.setattr(s, "TABLE_SCHEMAS", {malformed.name: malformed})
    monkeypatch.setattr(s, "ALL_RUNTIME_PATHS", (malformed.relative_path,))
    with pytest.raises(RuntimeError, match="Column-rule mismatch"):
        s.assert_schema_registry_integrity()


def test_json_registry_key_name_mismatch_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = s.JSON_SCHEMAS["interpretation_classes"]
    monkeypatch.setattr(s, "TABLE_SCHEMAS", {})
    monkeypatch.setattr(s, "JSON_SCHEMAS", {"wrong_name": original})
    monkeypatch.setattr(s, "RASTER_SCHEMAS", {})
    monkeypatch.setattr(s, "ALL_RUNTIME_PATHS", (original.relative_path,))
    with pytest.raises(RuntimeError, match="JSON_SCHEMAS key/name mismatch"):
        s.assert_schema_registry_integrity()


def test_raster_registry_key_name_mismatch_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = s.RASTER_SCHEMAS["representative_scene_plot_ready"]
    monkeypatch.setattr(s, "TABLE_SCHEMAS", {})
    monkeypatch.setattr(s, "JSON_SCHEMAS", {})
    monkeypatch.setattr(s, "RASTER_SCHEMAS", {"wrong_name": original})
    monkeypatch.setattr(s, "ALL_RUNTIME_PATHS", (original.relative_path,))
    with pytest.raises(RuntimeError, match="RASTER_SCHEMAS key/name mismatch"):
        s.assert_schema_registry_integrity()


def test_duplicate_cross_registry_path_lookup_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = s.TABLE_SCHEMAS["seasonal_climate_context"]
    json_schema = replace(
        s.JSON_SCHEMAS["interpretation_classes"],
        relative_path=table.relative_path,
    )
    monkeypatch.setattr(s, "TABLE_SCHEMAS", {table.name: table})
    monkeypatch.setattr(s, "JSON_SCHEMAS", {json_schema.name: json_schema})
    monkeypatch.setattr(s, "RASTER_SCHEMAS", {})
    with pytest.raises(RuntimeError, match="Duplicate schema registrations"):
        s.schema_for_relative_path(table.relative_path)
