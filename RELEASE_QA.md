# Release audit and verification

## Baseline scientific record

The previous curated release reported 1,708 scientific input checks passing,
a successful seven-module pipeline, and 24/24 regenerated analytical CSV files
identical to the frozen results. Those were executed on the earlier package.
This archive preserves the curated scientific input files and analytical code;
only distribution documentation, supplementary file metadata and license files
were changed. See the fresh-extraction checks added below for this version.

The earlier software package also records 637 passing regression tests, but
that entire historical run is not independently confirmed by this report.

## Scientific and licensing scope

- Source numbers and scientific analytical code are not edited by the license
  or public-documentation revision.
- The original Phase 3 self-manifest discrepancy is historical metadata; it is
  not treated as a corrected scientific input.
- Raster- and image-scale upstream reproducibility is limited as described in
  `DATA_AND_REPOSITORY_LIMITS.md`.
- `THIRD_PARTY_SOURCES.md` separates public-domain material, PRISM's retained
  rights, and original derived data that may be offered under CC BY 4.0.
- Object-level geographic positions are retained; the public-disclosure
  decision remains subject to co-author and funding-agreement review.
- The license files are included as the proposed MIT/CC BY 4.0 release terms.
  Final authorization and rights-holder identity must be confirmed before the
  archive is made public.

## Fresh archive checks

Verification results are recorded in `QA_RESULTS.json`. The archive-wide hashes
are stored in `SHA256SUMS.txt` and can be checked using `verify_release.py`.

## PRISM and copyright attribution review (2026-11-03)

The archived climate tables identify the Earth Engine PRISM collection as
`OREGONSTATE/PRISM/ANd` and its precipitation band as `ppt`. The research
team identified April 17, 2026 as the original data-access date. The source
terms and Earth Engine catalog were reviewed on November 3, 2026. The
access date is research-team reported, not verified from an acquisition log.

The `LICENSE` file retains the proposed MIT copyright notice for Texas A&M
AgriLife Research. The institutional name is correct, but the materials
reviewed do not establish whether AgriLife Research or The Texas A&M University
System holds legal title to this specific sponsored software. The notice was
not changed without evidence from the rights holder. The intellectual-property
office's favorable commercialization assessment is distinct from a formal
determination of copyright ownership.

## Verification scope

The curated release provides the analytical inputs, Python modules, and
source tables supporting the manuscript. The recorded validation included
1,708 scientific input checks and successful reproduction of 24 analytical
CSV files. These results were obtained with the validated source package
retained in the project archive. The public distribution contains the inputs
needed for the released analytical workflows; tests requiring the larger
historical development archive are outside its scope.
