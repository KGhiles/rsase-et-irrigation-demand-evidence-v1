"""Stable command-line interface for the Phase 4 analytical repository.

The interface exposes validation, each accepted analytical module, and the
complete deterministic pipeline through one entry point::

    python -m qstable_awm.cli <command> [options]

Scientific calculations remain inside the validated analytical modules.  This
file only resolves command arguments, enforces safe output locations, invokes
the requested operation, and emits machine-readable JSON summaries.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

try:
    from .aerodynamic_sensitivity import (
        AerodynamicSensitivityError,
        run_aerodynamic_sensitivity,
    )
    from .climate_context import ClimateContextError, run_climate_context
    from .crop_support import CropSupportError, run_crop_support
    from .fixed_frame_dominance import (
        FixedFrameDominanceError,
        run_fixed_frame_dominance,
    )
    from .interpretation_classes import (
        InterpretationClassesError,
        run_interpretation_classes,
    )
    from .known_cold_reference import (
        KnownColdReferenceError,
        run_known_cold_reference,
    )
    from .physical_coherence import (
        PhysicalCoherenceError,
        run_physical_coherence,
    )
    from .pipeline import (
        DEFAULT_RESULTS_ROOT,
        PipelineError,
        run_pipeline,
    )
    from .schemas import PHASE3_PACKAGE_NAME
    from .validation import (
        DEFAULT_CONTRACT_PATH,
        PHASE4_ROOT,
        RuntimeInputError,
        validate_phase3_root,
        write_validation_report,
    )
except ImportError:  # Support direct execution before package installation.
    _SRC_ROOT = Path(__file__).resolve().parents[1]
    if str(_SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(_SRC_ROOT))
    from qstable_awm.aerodynamic_sensitivity import (  # type: ignore[no-redef]
        AerodynamicSensitivityError,
        run_aerodynamic_sensitivity,
    )
    from qstable_awm.climate_context import (  # type: ignore[no-redef]
        ClimateContextError,
        run_climate_context,
    )
    from qstable_awm.crop_support import (  # type: ignore[no-redef]
        CropSupportError,
        run_crop_support,
    )
    from qstable_awm.fixed_frame_dominance import (  # type: ignore[no-redef]
        FixedFrameDominanceError,
        run_fixed_frame_dominance,
    )
    from qstable_awm.interpretation_classes import (  # type: ignore[no-redef]
        InterpretationClassesError,
        run_interpretation_classes,
    )
    from qstable_awm.known_cold_reference import (  # type: ignore[no-redef]
        KnownColdReferenceError,
        run_known_cold_reference,
    )
    from qstable_awm.physical_coherence import (  # type: ignore[no-redef]
        PhysicalCoherenceError,
        run_physical_coherence,
    )
    from qstable_awm.pipeline import (  # type: ignore[no-redef]
        DEFAULT_RESULTS_ROOT,
        PipelineError,
        run_pipeline,
    )
    from qstable_awm.schemas import PHASE3_PACKAGE_NAME  # type: ignore[no-redef]
    from qstable_awm.validation import (  # type: ignore[no-redef]
        DEFAULT_CONTRACT_PATH,
        PHASE4_ROOT,
        RuntimeInputError,
        validate_phase3_root,
        write_validation_report,
    )


CLI_VERSION = "1.0.0"
CLI_RELATIVE_PATH = "src/qstable_awm/cli.py"

COMMAND_TO_MODULE: Mapping[str, str] = {
    "climate-context": "climate_context",
    "physical-coherence": "physical_coherence",
    "aerodynamic-sensitivity": "aerodynamic_sensitivity",
    "crop-support": "crop_support",
    "fixed-frame-dominance": "fixed_frame_dominance",
    "known-cold-reference": "known_cold_reference",
    "interpretation-classes": "interpretation_classes",
}
MODULE_TO_COMMAND: Mapping[str, str] = {
    module: command for command, module in COMMAND_TO_MODULE.items()
}

Runner = Callable[..., Any]
MODULE_RUNNERS: Mapping[str, Runner] = {
    "climate_context": run_climate_context,
    "physical_coherence": run_physical_coherence,
    "aerodynamic_sensitivity": run_aerodynamic_sensitivity,
    "crop_support": run_crop_support,
    "fixed_frame_dominance": run_fixed_frame_dominance,
    "known_cold_reference": run_known_cold_reference,
    "interpretation_classes": run_interpretation_classes,
}

_EXPECTED_ERRORS = (
    AerodynamicSensitivityError,
    ClimateContextError,
    CropSupportError,
    FixedFrameDominanceError,
    InterpretationClassesError,
    KnownColdReferenceError,
    PhysicalCoherenceError,
    PipelineError,
    RuntimeInputError,
    FileNotFoundError,
    ValueError,
    KeyError,
)


class CLIError(RuntimeError):
    """Raised when command routing or output-root policy is invalid."""


def _default_phase3_root() -> Path | None:
    """Return an explicit environment or sibling Phase 3 root when available."""

    environment = os.environ.get("QSTABLE_PHASE3_ROOT")
    if environment:
        return Path(environment).expanduser()
    sibling = PHASE4_ROOT.parent / PHASE3_PACKAGE_NAME
    if sibling.is_dir():
        return sibling
    return None


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _assert_safe_results_root(phase3_root: Path, results_root: Path) -> None:
    """Prevent analytical outputs from entering frozen or source directories."""

    phase3 = Path(phase3_root).expanduser().resolve()
    results = Path(results_root).expanduser().resolve()
    if results == phase3 or _is_relative_to(results, phase3):
        raise CLIError("Results root must not equal or be inside the Phase 3 root")
    protected = tuple(
        (PHASE4_ROOT / name).resolve() for name in ("src", "config", "tests")
    )
    if any(results == item or _is_relative_to(results, item) for item in protected):
        raise CLIError("Results root must not be inside src, config, or tests")


def _add_phase3_argument(parser: argparse.ArgumentParser) -> None:
    default_root = _default_phase3_root()
    parser.add_argument(
        "--phase3-root",
        type=Path,
        default=default_root,
        required=default_root is None,
        help=(
            f"Extracted {PHASE3_PACKAGE_NAME} directory. Defaults to "
            "QSTABLE_PHASE3_ROOT or the sibling package when available."
        ),
    )


def _add_contract_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--contract",
        type=Path,
        default=DEFAULT_CONTRACT_PATH,
        help="Path to config/phase4_source_contract.yml.",
    )


def _add_results_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Repository results directory. Module output is written below it.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing outputs for the requested operation.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the stable top-level Phase 4 command parser."""

    parser = argparse.ArgumentParser(
        prog="qstable-awm",
        description=(
            "Validate and execute the Q-Stable/AWM Phase 4 analytical repository."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {CLI_VERSION}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate the complete frozen Phase 3 layer or selected modules.",
    )
    _add_phase3_argument(validate_parser)
    _add_contract_argument(validate_parser)
    validate_parser.add_argument(
        "--module",
        action="append",
        choices=tuple(COMMAND_TO_MODULE),
        help="Validate a module; repeat for several. Omit to validate all modules.",
    )
    validate_parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional detailed validation-report JSON path.",
    )
    validate_parser.add_argument(
        "--output-csv",
        type=Path,
        help="Optional detailed validation-report CSV path.",
    )
    validate_parser.add_argument(
        "--hashes-only",
        action="store_true",
        help="Validate package identity and hashes without structured schemas.",
    )

    for command in COMMAND_TO_MODULE:
        module_parser = subparsers.add_parser(
            command,
            help=f"Run the {command} analytical module.",
        )
        _add_phase3_argument(module_parser)
        _add_contract_argument(module_parser)
        _add_results_arguments(module_parser)

    run_all_parser = subparsers.add_parser(
        "run-all",
        help="Validate Phase 3 and atomically run all analytical modules.",
    )
    _add_phase3_argument(run_all_parser)
    _add_contract_argument(run_all_parser)
    _add_results_arguments(run_all_parser)
    run_all_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the complete plan without writing outputs.",
    )
    return parser


def _run_validate(args: argparse.Namespace) -> tuple[int, Mapping[str, Any]]:
    modules = (
        None
        if not args.module
        else tuple(COMMAND_TO_MODULE[command] for command in args.module)
    )
    report = validate_phase3_root(
        args.phase3_root,
        contract_path=args.contract,
        modules=modules,
        validate_schemas=not args.hashes_only,
    )
    write_validation_report(
        report,
        json_path=args.output_json,
        csv_path=args.output_csv,
    )
    payload: dict[str, Any] = {
        "command": "validate",
        "summary": report.summary(),
        "output_json": None if args.output_json is None else str(args.output_json),
        "output_csv": None if args.output_csv is None else str(args.output_csv),
        "hashes_only": bool(args.hashes_only),
    }
    return (1 if report.failed_checks else 0), payload


def _run_module(args: argparse.Namespace) -> Mapping[str, Any]:
    module_name = COMMAND_TO_MODULE[args.command]
    _assert_safe_results_root(args.phase3_root, args.results_root)
    output_dir = Path(args.results_root).expanduser().resolve() / module_name
    runner = MODULE_RUNNERS[module_name]
    outputs = runner(
        args.phase3_root,
        output_dir,
        contract_path=args.contract,
        overwrite=args.overwrite,
    )
    output_mapping = (
        outputs.as_dict() if hasattr(outputs, "as_dict") else {"value": str(outputs)}
    )
    return {
        "command": args.command,
        "module": module_name,
        "phase3_package": PHASE3_PACKAGE_NAME,
        "results_directory": str(output_dir),
        "contract": str(Path(args.contract).expanduser().resolve()),
        "overwrite": bool(args.overwrite),
        "outputs": output_mapping,
    }


def _run_all(args: argparse.Namespace) -> Mapping[str, Any]:
    result = run_pipeline(
        args.phase3_root,
        args.results_root,
        modules=None,
        contract_path=args.contract,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    return {
        "command": "run-all",
        **result.to_dict(),
    }


def dispatch(args: argparse.Namespace) -> tuple[int, Mapping[str, Any]]:
    """Execute one parsed command and return exit status plus JSON payload."""

    if args.command == "validate":
        return _run_validate(args)
    if args.command in COMMAND_TO_MODULE:
        return 0, _run_module(args)
    if args.command == "run-all":
        return 0, _run_all(args)
    raise CLIError(f"Unsupported command: {args.command!r}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the stable CLI and return a process-compatible status code."""

    args = build_parser().parse_args(argv)
    try:
        status, payload = dispatch(args)
    except (CLIError, *_EXPECTED_ERRORS) as exc:
        print(f"QSTABLE_AWM_CLI_ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    return status


__all__ = [
    "CLIError",
    "CLI_RELATIVE_PATH",
    "CLI_VERSION",
    "COMMAND_TO_MODULE",
    "MODULE_TO_COMMAND",
    "build_parser",
    "dispatch",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
