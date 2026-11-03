# Remote-sensing ET evidence: reproducibility package

Code, curated analytical inputs and supplementary material supporting the
manuscript *When physical coherence is not enough: support screening for
irrigation-demand proxies derived from satellite ET in the northern Texas
Panhandle*.

## Contents

- `PHASE4_REUSABLE_ANALYTIC_MODULES_001/` — Python code, tests, recorded results and source tables for Figures 2–8 and Table 3.
- `PHASE3_CURATED_INPUT_LAYER_FINAL_002/data/derived/` — frozen analytical inputs, including seasonal accounting, crop support, physical diagnostics and cold-reference sensitivity.
- `supplementary/` — Supplementary Methods S1 and Tables S1–S11b (Table S4 is in Supplementary Methods S1).
- `SOURCE_DATA_INDEX.csv` — source files for reported figures and tables.
- `THIRD_PARTY_SOURCES.md` — upstream dataset provenance and geographic-data disclosure.
- `RELEASE_QA.md` — executed checks and reproducibility limits.
- `SHA256SUMS.txt` and `verify_release.py` — complete archive checksums and verifier.

Historical package folder names, Python import names and source identifiers
are retained because renaming them would change the tested software interfaces.
They are not manuscript terminology.

## Install and reproduce tables

Use Python 3.11 or newer in a compatible environment. From the extracted
release directory:

```bash
cd PHASE4_REUSABLE_ANALYTIC_MODULES_001
python -m pip install --editable ".[test]"
qstable-awm validate --phase3-root ../PHASE3_CURATED_INPUT_LAYER_FINAL_002
qstable-awm run-all --phase3-root ../PHASE3_CURATED_INPUT_LAYER_FINAL_002 --dry-run
qstable-awm run-all --phase3-root ../PHASE3_CURATED_INPUT_LAYER_FINAL_002 --results-root ./reproduced_results
```

The archived source tables are in `PHASE4_REUSABLE_ANALYTIC_MODULES_001/results/`.
The command above writes new results in `reproduced_results`; it does not
replace the archived results. The detailed command reference is in the Phase 4
README. On Unix-like systems, some ZIP extractors discard executable file
permissions. Before running the optional legacy test suite, restore them
with `chmod +x PHASE4_REUSABLE_ANALYTIC_MODULES_001/scripts/*.py
PHASE4_REUSABLE_ANALYTIC_MODULES_001/scripts/*.sh`. The historical tests
also include a full-archive inventory assertion that is not applicable to
this curated subset. See `RELEASE_QA.md`.

To check distributed file integrity before installation:

```bash
python verify_release.py
```

## Scientific limits

`ETa` is modeled actual evapotranspiration. `NIR` is the nonnegative seasonal
ET-minus-effective-precipitation deficit. `Qgross` is NIR divided by an assumed
application efficiency of 0.88. None measures irrigation applied, groundwater
pumping or the source of water consumed. Crop objects are analytical units, not
surveyed farms or water-right boundaries.

This archive reproduces the distributed analytical source tables from frozen
curated inputs. It does not rerun the historical Landsat imagery collection,
Google Earth Engine processing, original METRIC image-scale solution, or all
full-resolution raster experiments. The compact representative raster is a
visualization source rather than a replacement for the original scene archive.
See `DATA_AND_REPOSITORY_LIMITS.md`.

## Licenses and citation

- Original Python software: **MIT** (`LICENSE`).
- Original derived tables and release prose where the project controls rights:
  **CC BY 4.0** (`LICENSE-DATA.md`).
- Upstream third-party material: source-provider terms; see
  `THIRD_PARTY_SOURCES.md`. Public-domain data are not relicensed.

The licenses cover only original material the releasing rights holder is authorized
to license. Third-party data retain their provider terms, as documented in
`THIRD_PARTY_SOURCES.md`.

A permanent repository/DOI has not yet been assigned. Cite the manuscript
and the versioned dataset when its permanent identifier is available.
