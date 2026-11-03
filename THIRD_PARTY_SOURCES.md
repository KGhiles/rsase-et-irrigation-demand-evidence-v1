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
| PRISM Group, Oregon State University | Daily precipitation used in monthly effective-precipitation accounting | PRISM permits reproduction and distribution but retains ownership and requires clear source credit, its URL and the date of original data access. https://prism.oregonstate.edu/terms/ |
| U.S. Census Bureau TIGER/Line | County analysis boundaries | U.S. Census Bureau geographical boundaries are publicly distributed and not legal descriptions of ownership. https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html |

**PRISM access-date record:** The curated release identifies the PRISM dataset
but does not contain a verified original download/access date. The research
team must supply the historical access date from its acquisition log before
public release. Do not substitute the date of this licensing audit for that
historical date.

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
