# Analytical software for the remote-sensing ET evidence study

Python analysis code supporting the manuscript:

> *When physical coherence is not enough: support screening for irrigation-demand proxies derived from satellite ET in the northern Texas Panhandle*

These modules regenerate analytical source tables for Figures 2–8 and Table 3 from the frozen curated inputs. The package includes input validation, a command-line interface, recorded outputs and tests.

## Scope

The package supports seven analytical domains:

1. climate context;
2. physical coherence;
3. aerodynamic sensitivity;
4. fixed-frame crop support;
5. fixed-frame 2011 dominance;
6. strict known-cold reference sensitivity; and
7. interpretation classes with narrow HLS temporal-support evidence.

The runtime authority is the extracted sibling package:

```text
PHASE3_CURATED_INPUT_LAYER_FINAL_002/
```

The Phase 4 modules do not execute historical notebooks and do not read uncurated branch outputs directly.

## Reproducibility level

This repository supports:

- **Level 1 — manuscript-output reproducibility:** deterministic regeneration of the accepted Phase 4 source tables from frozen curated inputs;
- **Level 2 — scientific provenance:** traceability through source contracts, checksums, validation reports, and retained provenance materials.

It does **not** claim complete upstream regeneration of Google Earth Engine processing, historical METRIC runs, ACES workflows, or the full V51 raster archive.

## Scientific interpretation boundary

The following boundaries apply to every public output:

- **ETa** is land-surface actual evapotranspiration or consumptive use. It is not measured irrigation application or pumping and does not identify the water source.
- **NIR** is a derived seasonal deficit under explicit effective-precipitation assumptions.
- **Qgross** is an efficiency-scaled gross-depth equivalent. It is not measured pumping or groundwater withdrawal.
- No pumping validation, groundwater-withdrawal validation, or water-source attribution is claimed.
- No corrected-METRIC-ET product is created.
- NDVI, LAI, thermal position, H, LE, Rn, and LE/Rn are physical-coherence diagnostics rather than independent ET validation.
- CDL objects are analytical units, not ownership parcels, irrigation-system units, or surveyed fields.
- The 2011 season was used for an extreme-drought stress test. The accepted METRIC output remains the baseline for the seasonal comparison.
- HLS evidence is restricted to canopy-trajectory and interpolation-plausibility support. `Class I candidate` is not a completed `Class I` assignment.

## Software layout

```text
PHASE4_REUSABLE_ANALYTIC_MODULES_001/
├── config/
│   └── phase4_source_contract.yml
├── results/
│   ├── aerodynamic_sensitivity/
│   ├── climate_context/
│   ├── crop_support/
│   ├── fixed_frame_dominance/
│   ├── interpretation_classes/
│   ├── known_cold_reference/
│   ├── physical_coherence/
│   └── pipeline/
├── src/qstable_awm/
│   ├── aerodynamic_sensitivity.py
│   ├── cli.py
│   ├── climate_context.py
│   ├── crop_support.py
│   ├── fixed_frame_dominance.py
│   ├── interpretation_classes.py
│   ├── known_cold_reference.py
│   ├── physical_coherence.py
│   ├── pipeline.py
│   ├── schemas.py
│   └── validation.py
├── tests/
├── pyproject.toml
└── README.md
```

The Phase 3 package should normally be placed beside the Phase 4 directory:

```text
parent_directory/
├── PHASE3_CURATED_INPUT_LAYER_FINAL_002/
└── PHASE4_REUSABLE_ANALYTIC_MODULES_001/
```

Alternatively, set `QSTABLE_PHASE3_ROOT` to the extracted Phase 3 directory.

## Requirements

- Python 3.11 or newer
- NumPy 1.24–2.x
- pandas 2.x
- PyYAML 6.x
- rasterio 1.3–1.x
- pytest 8–9 for testing

## Installation

From the Phase 4 repository root:

```bash
python -m pip install --editable ".[test]"
```

The installed console command is:

```bash
qstable-awm --version
```

Expected version:

```text
qstable-awm 1.0.0
```

An editable repository installation is the recommended scientific execution mode because the source contract and frozen Phase 3 package are external runtime authorities. The wheel contains the Python software but does not bundle the source contract, results, tests, or Phase 3 data.

## Validate the frozen inputs

Validate the complete Phase 3 layer:

```bash
qstable-awm validate \
  --phase3-root ../PHASE3_CURATED_INPUT_LAYER_FINAL_002
```

When Phase 3 is a sibling directory or `QSTABLE_PHASE3_ROOT` is set, the path may be omitted:

```bash
qstable-awm validate
```

Write detailed reports:

```bash
qstable-awm validate \
  --output-json results/phase3_runtime_validation.json \
  --output-csv results/phase3_runtime_validation.csv
```

Validate selected modules by repeating `--module`:

```bash
qstable-awm validate \
  --module climate-context \
  --module crop-support
```

The complete accepted validator performs 1,708 deterministic checks across package identity, file hashes, schemas, key uniqueness, scalar constraints, raster metadata, source authority, and scientific guardrails.

## Preview the complete workflow

Run a no-write preflight:

```bash
qstable-awm run-all \
  --phase3-root ../PHASE3_CURATED_INPUT_LAYER_FINAL_002 \
  --dry-run
```

The dry run validates all inputs and prints the frozen seven-module execution plan without creating outputs.

## Run the complete pipeline

```bash
qstable-awm run-all \
  --phase3-root ../PHASE3_CURATED_INPUT_LAYER_FINAL_002 \
  --results-root results
```

Existing outputs are protected. To replace them explicitly:

```bash
qstable-awm run-all \
  --phase3-root ../PHASE3_CURATED_INPUT_LAYER_FINAL_002 \
  --results-root results \
  --overwrite
```

The pipeline executes modules in this fixed order:

1. `climate_context`
2. `physical_coherence`
3. `aerodynamic_sensitivity`
4. `crop_support`
5. `fixed_frame_dominance`
6. `known_cold_reference`
7. `interpretation_classes`

Modules run in an isolated staging tree. Selected result directories are published atomically only after all selected modules pass. The pipeline inventories the complete Phase 3 tree before and after execution and aborts if any Phase 3 file changes.

A complete run generates:

- 38 analytical-module outputs;
- `results/pipeline/phase4_pipeline_output_manifest.csv`;
- `results/pipeline/phase4_pipeline_validation.json`; and
- `results/pipeline/phase4_pipeline_execution_record.json`.

## Run an individual module

```bash
qstable-awm crop-support \
  --phase3-root ../PHASE3_CURATED_INPUT_LAYER_FINAL_002 \
  --results-root results
```

Available public commands:

| Command | Manuscript role | Output files |
|---|---|---:|
| `climate-context` | Figure 2 and climate-context text statistics | 5 |
| `physical-coherence` | Figure 3 scene and spatial-coherence sources | 6 |
| `aerodynamic-sensitivity` | Figure 4, Table 3, and sensitivity statistics | 6 |
| `crop-support` | Figure 5 fixed-frame crop-support sources | 5 |
| `fixed-frame-dominance` | Figure 6 stress-year dominance sources | 5 |
| `known-cold-reference` | Figure 7 strict reference-sensitivity sources | 6 |
| `interpretation-classes` | Figure 8 classification and evidence matrix | 5 |

Each command writes only the filenames authorized for that module by `config/phase4_source_contract.yml`.

## Output contract

### Climate context

```text
results/climate_context/
├── figure2_climate_source.csv
├── figure2_monthly_trajectory_source.csv
├── climate_context_text_statistics.csv
├── climate_context_validation.json
└── climate_context_execution_record.json
```

### Physical coherence

```text
results/physical_coherence/
├── figure3_scene_fingerprint_source.csv
├── figure3_mechanism_source.csv
├── figure3_spatial_coherence_source.csv
├── physical_coherence_text_statistics.csv
├── physical_coherence_validation.json
└── physical_coherence_execution_record.json
```

The compact GeoTIFF is opened only to validate its visualization metadata. Full-resolution area statistics are not recalculated from it.

### Aerodynamic sensitivity

```text
results/aerodynamic_sensitivity/
├── figure4_aerodynamic_source.csv
├── figure4_peak_sensitivity_source.csv
├── table3_aerodynamic_summary.csv
├── aerodynamic_sensitivity_text_statistics.csv
├── aerodynamic_sensitivity_validation.json
└── aerodynamic_sensitivity_execution_record.json
```

### Crop support

```text
results/crop_support/
├── figure5_domain_crop_support_source.csv
├── figure5_county_crop_support_source.csv
├── crop_support_text_statistics.csv
├── crop_support_validation.json
└── crop_support_execution_record.json
```

### Fixed-frame dominance

```text
results/fixed_frame_dominance/
├── figure6_fixed_frame_sensitivity_source.csv
├── figure6_dominance_class_source.csv
├── fixed_frame_dominance_text_statistics.csv
├── fixed_frame_dominance_validation.json
└── fixed_frame_dominance_execution_record.json
```

### Known-cold reference

```text
results/known_cold_reference/
├── figure7_processing_funnel_source.csv
├── figure7_county_radius_source.csv
├── figure7_mask_response_source.csv
├── known_cold_reference_text_statistics.csv
├── known_cold_reference_validation.json
└── known_cold_reference_execution_record.json
```

### Interpretation classes

```text
results/interpretation_classes/
├── figure8_interpretation_source.csv
├── figure8_evidence_matrix_source.csv
├── interpretation_classes_text_statistics.csv
├── interpretation_classes_validation.json
└── interpretation_classes_execution_record.json
```

## Statistical conventions

The text-statistics tables add explicit observational units, `n`, `ddof`, units, means, standard deviations, medians, minima, maxima, and interpretation boundaries.

- Sample SD uses `ddof=1` when the reported values are treated as a sample of counties, county-years, scenes, scenarios, or retained grouped records.
- Crop-support area-weighted object dispersion uses population-style weighting with `ddof=0` because the 1,184 fixed-frame objects constitute the complete analytical population for that diagnostic.
- Interpretation classes contain deterministic categorical counts rather than replicate numerical observations; no SD is fabricated.
- All reported SDs are descriptive dispersion. They are not confidence intervals, prediction errors, pixel-scale uncertainty, ET validation uncertainty, or pumping uncertainty.

## Source authority and integrity

`config/phase4_source_contract.yml` is the executable authority for:

- the Phase 3 package identity;
- exact module input sets;
- expected SHA-256 digests;
- allowed output filenames;
- notebook and provenance restrictions;
- prohibited historical-source tokens;
- Figure 8 and V50/HLS guardrails; and
- overwrite and output-manifest requirements.

At runtime, each analytical module:

1. loads the source contract;
2. validates its complete authorized input set;
3. verifies frozen hashes and structured schemas;
4. calculates outputs from Phase 3 tables;
5. validates derived outputs;
6. writes deterministic public files; and
7. records module, contract, input, validation, and output hashes.

Execution-record hashes are deterministic under a fixed contract and repository location, but they are metadata rather than permanent scientific-value locks because each record embeds the active contract hash. Three earlier module records also retain the resolved source-contract path, so their bytes—and the pipeline metadata that inventories them—can change when the repository is extracted to a different absolute location. The regression suite therefore locks accepted repository metadata at its frozen hashes while comparing regenerated scientific CSV/validation outputs exactly and execution records semantically across locations. Scientific values are path-independent.

## Testing

Run the complete repository suite:

```bash
python -m pytest
```

The suite covers:

- manuscript-value and figure-source regression checks;
- standard-deviation definitions and observational units;
- complete Phase 3 runtime validation;
- immutable schema registries;
- source-contract and path protections;
- deterministic output generation;
- overwrite refusal;
- pipeline staging, atomic installation, and rollback;
- CLI behavior and exit codes;
- Phase 3 immutability;
- package discovery, editable installation, and wheel contents.

## Phase 3 validation status and documented gaps

The frozen Phase 3 package status is:

```text
PASS_WITH_DOCUMENTED_GAPS
```

Its two documented provenance gaps are:

1. the exact frozen Figure 1 study-region GeoJSON; and
2. individual hashes and archive location for the 39 full V51 rasters.

These gaps do not prevent the Phase 4 modules from reproducing their authorized Figures 2–8 and Table 3 source outputs. They do limit claims of complete upstream regeneration.

## License and citation metadata

The release-root `LICENSE` provides the MIT software license, and
`LICENSE-DATA.md` describes CC BY 4.0 coverage of original research outputs.
Third-party material retains its source-provider terms as documented in the
release-root `THIRD_PARTY_SOURCES.md`.

A permanent DOI and repository citation will be added once the release is
archived. Historical source filenames do not establish authorship or ownership.

## License

The original software is distributed under the MIT license in the release root (`../LICENSE`). Dataset rights and third-party attribution are documented separately in the release root.
