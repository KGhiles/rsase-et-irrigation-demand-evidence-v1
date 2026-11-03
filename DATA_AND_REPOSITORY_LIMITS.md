# Reproducibility limits

The distribution contains the frozen curated scientific inputs, analytical
source tables, Python modules and published supplementary methods and tables.
It excludes older working drafts, reviewer correspondence and full historical
notebook archives.

The following are not distributed:

- the original frozen spatial geometry used to prepare Figure 1;
- the complete high-resolution raster set underlying the 39 known-cold
  comparisons and its individual-file checksum inventory;
- all image-processing inputs needed to rerun historical Earth Engine and
  METRIC image-scale processing;
- a complete pinned environment specification for every upstream stage.

The historical Phase 3 source manifest describes a larger archive than this
repository. Its self-reference entry has a preexisting checksum discrepancy;
the scientifically required runtime inputs are validated separately. The
present archive has a new distribution-wide SHA-256 inventory. Historical
scientific CSV inputs were not changed to repair that older metadata entry.

Reproducing the distributed derived results is not the same as regenerating
all upstream observations or reproducing every original raster. Original
provider conditions are listed in `THIRD_PARTY_SOURCES.md`.
