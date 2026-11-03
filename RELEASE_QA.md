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

## Regression-test qualification

The current limited run passed 432 tests before stopping at a historical
package-inventory assertion that requires 172 files, compared with 81 in the
public curated Phase 3 package. The full 637-test suite was not certified
against this reduced distribution. Tests that assume the upstream full archive
should be run only with that archive; they are not evidence that the published
analytical CSV values differ. Scripts need executable permissions restored
after extraction if the ZIP tool discards Unix mode bits.
