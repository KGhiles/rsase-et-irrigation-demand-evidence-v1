# Data provenance, source terms and location disclosure

The analysis uses public satellite, crop-classification, meteorological and
administrative boundary datasets. The repository contains derived analytical
outputs, not a replacement distribution of the complete upstream datasets.
The listed source permissions apply to the original providers' material;
CC BY 4.0 in this release applies only to original outputs that the project
is authorized to license.

| Source | Use in the study | Provider terms and credit |
| --- | --- | --- |
| USGS Landsat Collection 2 | Surface reflectance, thermal observations and METRIC diagnostics | USGS Landsat data are public domain. Credit USGS. https://www.usgs.gov/faqs/are-landsat-data-cloud-still-considered-be-within-public-domain |
| USDA NASS Cropland Data Layer | Annual corn classification, fixed-object crop support | USDA NASS considers CDL public domain and free to redistribute. Credit USDA NASS and the relevant product years. https://www.nass.usda.gov/Research_and_Science/Cropland/sarsfaqs2.php |
| NASA NLDAS | Image-date meteorological forcing | Cite the NASA NLDAS dataset used for forcing; no raw NLDAS product is included. https://ldas.gsfc.nasa.gov/nldas/nldas-get-data |
| gridMET | Daily alfalfa-reference ET forcing | gridMET is released with a public-domain dedication. Credit the gridMET project. https://www.climatologylab.org/gridmet.html |
| PRISM Group, Oregon State University | Daily precipitation (`ppt` band) from Earth Engine `OREGONSTATE/PRISM/ANd`; aggregated for monthly effective-precipitation accounting | PRISM permits redistribution and retains ownership. Credit the PRISM Group and record the source access date. Terms: https://prism.oregonstate.edu/terms/ ; Earth Engine collection: https://developers.google.com/earth-engine/datasets/catalog/OREGONSTATE_PRISM_ANd |
| U.S. Census Bureau TIGER/Line | County analysis boundaries | U.S. Census Bureau geographical boundaries are publicly distributed and not legal descriptions of ownership. https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html |

## PRISM provenance

- **Provider:** PRISM Group, Oregon State University.
- **Source used:** Google Earth Engine collection `OREGONSTATE/PRISM/ANd`, band `ppt` (daily precipitation, mm). The source identifier appears in the archived climate-context CSVs and the original processing notebook.
- **Study years:** 2011, 2015, 2017 and 2023; daily precipitation was aggregated for monthly effective-precipitation accounting.
- **Original PRISM data access date:** **April 17, 2026** (reported by the research team; not independently established by retained acquisition logs).
- **Source documentation checked:** November 3, 2026 (PRISM provider terms and the Google Earth Engine dataset catalog). This review date is distinct from the April 17, 2026 data-access date.
- **Required attribution:** PRISM Group, Oregon State University (https://prism.oregonstate.edu), daily precipitation from `OREGONSTATE/PRISM/ANd` (band `ppt`), accessed April 17, 2026. PRISM retains ownership of its data; the project's CC BY 4.0 statement does not relicense them.

## Object-level location data

The curated files `data/derived/crop_support/fixed_frame_object_year_crop_support.csv`
and `data/derived/fixed_frame_dominance/fixed_frame_object_year_water_accounting.csv`
under `PHASE3_CURATED_INPUT_LAYER_FINAL_002/` each retain `centroid_x`,
`centroid_y`, and `bounds` for the fixed analytical units (1,184 unique
objects, four study years). These locations describe modeled crop-analysis
objects, not verified parcel boundaries, farmer identities, irrigated-field
surveys or observations of applied irrigation or well pumping.

Precise geographic coordinates could permit linkage to individual properties.
Their inclusion should be recorded as an explicit project disclosure decision
under the relevant source permissions and any applicable cooperative research
obligations. No farmer or well ownership information is part of the declared
analytical workflow. This statement is about the included data, not proof that
external linkages are impossible.

The original raster-derived estimates and spatial objects are not independently
validated field water-use records. Refer to `SCIENTIFIC_SCOPE.md` for the
interpretation constraints.
