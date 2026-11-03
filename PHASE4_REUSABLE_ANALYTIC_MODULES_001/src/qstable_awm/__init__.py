"""Public package metadata for the Q-Stable/AWM Phase 4 repository.

Importing :mod:`qstable_awm` is intentionally lightweight.  The package root
exposes only stable release metadata and immutable module/command names; it
does not eagerly import pandas, rasterio, the runtime validator, the pipeline,
or any scientific analytical module.

Scientific functions remain available from their explicit modules, for
example::

    from qstable_awm.crop_support import run_crop_support
    from qstable_awm.pipeline import run_pipeline

The command-line interface is available through::

    python -m qstable_awm.cli --help
"""

from __future__ import annotations

from typing import Final

PACKAGE_NAME: Final[str] = "qstable-awm"
PACKAGE_IMPORT_NAME: Final[str] = "qstable_awm"
PHASE3_PACKAGE_NAME: Final[str] = "PHASE3_CURATED_INPUT_LAYER_FINAL_002"
PHASE4_PACKAGE_NAME: Final[str] = "PHASE4_REUSABLE_ANALYTIC_MODULES_001"

VERSION_INFO: Final[tuple[int, int, int]] = (1, 0, 0)
__version__: Final[str] = ".".join(str(part) for part in VERSION_INFO)

ANALYTICAL_MODULES: Final[tuple[str, ...]] = (
    "climate_context",
    "physical_coherence",
    "aerodynamic_sensitivity",
    "crop_support",
    "fixed_frame_dominance",
    "known_cold_reference",
    "interpretation_classes",
)

PUBLIC_COMMANDS: Final[tuple[str, ...]] = (
    "validate",
    "climate-context",
    "physical-coherence",
    "aerodynamic-sensitivity",
    "crop-support",
    "fixed-frame-dominance",
    "known-cold-reference",
    "interpretation-classes",
    "run-all",
)

__all__ = (
    "ANALYTICAL_MODULES",
    "PACKAGE_IMPORT_NAME",
    "PACKAGE_NAME",
    "PHASE3_PACKAGE_NAME",
    "PHASE4_PACKAGE_NAME",
    "PUBLIC_COMMANDS",
    "VERSION_INFO",
    "__version__",
)
