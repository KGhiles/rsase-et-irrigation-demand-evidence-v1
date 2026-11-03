"""Deterministic orchestration for the Phase 4 analytical modules.

The pipeline validates the complete frozen Phase 3 curated-input layer before
any scientific module runs. It then executes a selected module subset, or all
seven modules, in the fixed publication order. Module outputs are generated in
a staging tree, validated against the active source contract, and committed to
the results tree only after every selected module passes.

Runtime authority
-----------------
* Phase 3 inputs are read only from ``PHASE3_CURATED_INPUT_LAYER_FINAL_002``.
* Historical notebooks and workflow archives are never imported or executed.
* Each analytical module writes only the filenames authorized by
  ``config/phase4_source_contract.yml``.
* The Phase 3 tree is fingerprinted before and after execution; any change
  aborts the run.
* Dry-run mode validates and prints the execution plan without writing files.

Pipeline metadata are written under ``results/pipeline``:

* ``phase4_pipeline_output_manifest.csv``
* ``phase4_pipeline_validation.json``
* ``phase4_pipeline_execution_record.json``

The output manifest covers selected analytical-module outputs. It excludes the
pipeline metadata files themselves to avoid self-referential checksums.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

try:
    from .validation import (
        DEFAULT_CONTRACT_PATH,
        PHASE4_ROOT,
        RuntimeInputError,
        load_source_contract,
        sha256_file,
        validate_phase3_root,
    )
except ImportError:  # Support direct execution before package installation.
    _SRC_ROOT = Path(__file__).resolve().parents[1]
    if str(_SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(_SRC_ROOT))
    from qstable_awm.validation import (  # type: ignore[no-redef]
        DEFAULT_CONTRACT_PATH,
        PHASE4_ROOT,
        RuntimeInputError,
        load_source_contract,
        sha256_file,
        validate_phase3_root,
    )


PIPELINE_VERSION = "1.0.0"
PIPELINE_RELATIVE_PATH = "src/qstable_awm/pipeline.py"
DEFAULT_RESULTS_ROOT = PHASE4_ROOT / "results"
PIPELINE_DIRECTORY_NAME = "pipeline"
PIPELINE_MANIFEST_NAME = "phase4_pipeline_output_manifest.csv"
PIPELINE_VALIDATION_NAME = "phase4_pipeline_validation.json"
PIPELINE_EXECUTION_RECORD_NAME = "phase4_pipeline_execution_record.json"

MODULE_ORDER = (
    "climate_context",
    "physical_coherence",
    "aerodynamic_sensitivity",
    "crop_support",
    "fixed_frame_dominance",
    "known_cold_reference",
    "interpretation_classes",
)

MODULE_RUN_FUNCTIONS = {
    "climate_context": "run_climate_context",
    "physical_coherence": "run_physical_coherence",
    "aerodynamic_sensitivity": "run_aerodynamic_sensitivity",
    "crop_support": "run_crop_support",
    "fixed_frame_dominance": "run_fixed_frame_dominance",
    "known_cold_reference": "run_known_cold_reference",
    "interpretation_classes": "run_interpretation_classes",
}

# File-access calls inspected by the AST-based runtime-source scan. Defensive
# string literals used in rejection checks are ignored unless they are passed
# to an actual file-access call.
_FILE_ACCESS_CALLS = {
    "open",
    "Path",
    "read_csv",
    "read_json",
    "read_table",
    "read_parquet",
    "read_excel",
    "read_text",
    "read_bytes",
    "load",
}


class PipelineError(RuntimeError):
    """Raised when orchestration, validation, or atomic commit fails."""


@dataclass(frozen=True, slots=True)
class PipelineOutputs:
    """Pipeline-level metadata files written after a successful non-dry run."""

    phase4_pipeline_output_manifest: Path
    phase4_pipeline_validation: Path
    phase4_pipeline_execution_record: Path

    def as_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}

    def paths(self) -> tuple[Path, ...]:
        return tuple(Path(value) for value in asdict(self).values())


@dataclass(frozen=True, slots=True)
class PipelineRunResult:
    """Return value for dry-run and executed pipeline invocations."""

    dry_run: bool
    selected_modules: tuple[str, ...]
    preflight_validation: Mapping[str, Any]
    plan: tuple[Mapping[str, Any], ...]
    outputs: PipelineOutputs | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "selected_modules": list(self.selected_modules),
            "preflight_validation": dict(self.preflight_validation),
            "plan": [dict(item) for item in self.plan],
            "outputs": None if self.outputs is None else self.outputs.as_dict(),
        }


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json_bytes(value))


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _resolve_selected_modules(modules: Sequence[str] | None) -> tuple[str, ...]:
    if not modules:
        return MODULE_ORDER
    if len(modules) != len(set(modules)):
        raise PipelineError("Duplicate module names were requested")
    unknown = set(modules) - set(MODULE_ORDER)
    if unknown:
        raise PipelineError(
            f"Unknown module names: {sorted(unknown)}; allowed={MODULE_ORDER}"
        )
    requested = set(modules)
    return tuple(module for module in MODULE_ORDER if module in requested)


def _assert_safe_roots(phase3_root: Path, results_root: Path) -> None:
    phase3 = phase3_root.expanduser().resolve()
    results = results_root.expanduser().resolve()
    if phase3 == results or _is_relative_to(results, phase3):
        raise PipelineError("Results root must not equal or be inside the Phase 3 root")
    protected = (
        (PHASE4_ROOT / "src").resolve(),
        (PHASE4_ROOT / "config").resolve(),
        (PHASE4_ROOT / "tests").resolve(),
    )
    if any(results == item or _is_relative_to(results, item) for item in protected):
        raise PipelineError("Results root must not be inside src, config, or tests")


def _tree_inventory(root: Path) -> tuple[list[dict[str, Any]], str]:
    """Return a complete deterministic file inventory and aggregate fingerprint."""

    resolved = root.expanduser().resolve()
    if not resolved.is_dir():
        raise PipelineError(f"Directory does not exist: {resolved}")
    rows: list[dict[str, Any]] = []
    for path in sorted(resolved.rglob("*")):
        if path.is_symlink():
            raise PipelineError(f"Symlinks are not permitted in the Phase 3 tree: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(resolved).as_posix()
        rows.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            (
                f"{row['path']}\0{row['size_bytes']}\0{row['sha256']}\n"
            ).encode("utf-8")
        )
    return rows, digest.hexdigest()


def _call_name(node: ast.Call) -> str:
    function = node.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return ""


def _string_constants(node: ast.AST) -> tuple[str, ...]:
    return tuple(
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    )


def _scan_runtime_file_access(
    source_paths: Sequence[Path], prohibited_tokens: Sequence[str]
) -> dict[str, list[str]]:
    """Detect prohibited tokens only when used in executable file access.

    This avoids false positives from defensive literals that document or reject
    obsolete sources while still catching direct file-access dependencies.
    """

    findings: dict[str, list[str]] = {}
    for path in source_paths:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        hits: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imported = " ".join(
                    alias.name for alias in getattr(node, "names", ())
                )
                for token in prohibited_tokens:
                    if token in imported:
                        hits.add(token)
            if not isinstance(node, ast.Call) or _call_name(node) not in _FILE_ACCESS_CALLS:
                continue
            values: list[str] = []
            for argument in (*node.args, *(kw.value for kw in node.keywords)):
                values.extend(_string_constants(argument))
            for value in values:
                if value.startswith("/scratch/"):
                    hits.add("/scratch/")
                for token in prohibited_tokens:
                    if token in value:
                        hits.add(token)
        if hits:
            findings[path.name] = sorted(hits)
    return findings


def _load_run_function(module_name: str) -> Callable[..., Any]:
    module = importlib.import_module(f"qstable_awm.{module_name}")
    function_name = MODULE_RUN_FUNCTIONS[module_name]
    function = getattr(module, function_name, None)
    if not callable(function):
        raise PipelineError(
            f"Missing callable qstable_awm.{module_name}.{function_name}"
        )
    return function


def _module_source_path(module_name: str) -> Path:
    return PHASE4_ROOT / "src" / "qstable_awm" / f"{module_name}.py"


def _build_plan(
    selected_modules: Sequence[str],
    *,
    results_root: Path,
    contract: Any,
) -> tuple[dict[str, Any], ...]:
    plan: list[dict[str, Any]] = []
    for index, module_name in enumerate(selected_modules, start=1):
        plan.append(
            {
                "order": index,
                "module": module_name,
                "source": _module_source_path(module_name).relative_to(PHASE4_ROOT).as_posix(),
                "output_directory": f"{contract.raw['output_policy']['runtime_results_root']}/{module_name}",
                "authorized_inputs": [
                    item.relative_path for item in contract.files_for_module(module_name)
                ],
                "authorized_outputs": list(contract.allowed_outputs(module_name)),
            }
        )
    return tuple(plan)


def _precheck_targets(
    results_root: Path,
    selected_modules: Sequence[str],
    *,
    overwrite: bool,
) -> None:
    targets = [results_root / module for module in selected_modules]
    targets.append(results_root / PIPELINE_DIRECTORY_NAME)
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        rendered = ", ".join(str(path) for path in existing)
        raise PipelineError(
            "Refusing to overwrite existing pipeline targets without --overwrite: "
            + rendered
        )
    for path in existing:
        if not path.is_dir() or path.is_symlink():
            raise PipelineError(f"Pipeline target is not a normal directory: {path}")


def _validate_module_directory(
    module_name: str,
    directory: Path,
    *,
    contract: Any,
) -> dict[str, Any]:
    expected = tuple(contract.allowed_outputs(module_name))
    observed = tuple(sorted(path.name for path in directory.iterdir() if path.is_file()))
    nonfiles = tuple(sorted(path.name for path in directory.iterdir() if not path.is_file()))
    if nonfiles:
        raise PipelineError(
            f"Unexpected non-file entries for {module_name}: {nonfiles}"
        )
    if set(observed) != set(expected) or len(observed) != len(expected):
        raise PipelineError(
            f"Output authorization mismatch for {module_name}: "
            f"observed={observed}; expected={tuple(sorted(expected))}"
        )

    validation_name = f"{module_name}_validation.json"
    execution_name = f"{module_name}_execution_record.json"
    validation_path = directory / validation_name
    execution_path = directory / execution_name
    if not validation_path.is_file() or not execution_path.is_file():
        raise PipelineError(
            f"Missing validation or execution record for {module_name}"
        )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("status") != "PASS" or validation.get("failed") != 0:
        raise PipelineError(
            f"Derived validation did not pass for {module_name}: {validation}"
        )
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    if execution.get("module") != module_name:
        raise PipelineError(f"Execution record module mismatch for {module_name}")
    if execution.get("contract_sha256") != contract.contract_sha256:
        raise PipelineError(f"Execution record contract hash mismatch for {module_name}")

    declared_inputs = {
        item.relative_path: item.sha256 for item in contract.files_for_module(module_name)
    }
    recorded_inputs = {
        key: value.get("sha256")
        for key, value in execution.get("authorized_inputs", {}).items()
    }
    if recorded_inputs != declared_inputs:
        raise PipelineError(f"Execution record input mismatch for {module_name}")

    recorded_outputs = execution.get("outputs")
    if not isinstance(recorded_outputs, Mapping):
        raise PipelineError(f"Execution record outputs missing for {module_name}")
    stable_outputs = set(expected) - {execution_name}
    if set(recorded_outputs) != stable_outputs:
        raise PipelineError(
            f"Execution record output set mismatch for {module_name}: "
            f"observed={sorted(recorded_outputs)} expected={sorted(stable_outputs)}"
        )
    for filename, metadata in recorded_outputs.items():
        path = directory / filename
        if not path.is_file():
            raise PipelineError(f"Recorded output is missing: {path}")
        if metadata.get("sha256") != sha256_file(path):
            raise PipelineError(f"Recorded output hash mismatch: {path}")
        if metadata.get("size_bytes") != path.stat().st_size:
            raise PipelineError(f"Recorded output size mismatch: {path}")

    return {
        "module": module_name,
        "status": "PASS",
        "output_count": len(expected),
        "derived_validation": {
            "status": validation.get("status"),
            "checks": validation.get("checks"),
            "passed": validation.get("passed"),
            "warnings": validation.get("warnings", 0),
            "failed": validation.get("failed", 0),
        },
        "execution_record_sha256": sha256_file(execution_path),
    }


def _manifest_rows(
    staging_results_root: Path,
    selected_modules: Sequence[str],
    *,
    contract: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for module_name in selected_modules:
        for filename in sorted(contract.allowed_outputs(module_name)):
            path = staging_results_root / module_name / filename
            rows.append(
                {
                    "module": module_name,
                    "relative_path": f"{module_name}/{filename}",
                    "filename": filename,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "file_type": path.suffix.lower().lstrip("."),
                    "contract_sha256": contract.contract_sha256,
                }
            )
    return rows


def _write_manifest(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "module",
        "relative_path",
        "filename",
        "size_bytes",
        "sha256",
        "file_type",
        "contract_sha256",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def _pipeline_validation(
    *,
    selected_modules: Sequence[str],
    preflight_summary: Mapping[str, Any],
    module_summaries: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
    phase3_before: str,
    phase3_after: str,
    source_scan_findings: Mapping[str, Sequence[str]],
    contract: Any,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(check_id: str, description: str, observed: Any, expected: Any, passed: bool) -> None:
        checks.append(
            {
                "check_id": check_id,
                "description": description,
                "observed": observed,
                "expected": expected,
                "status": "PASS" if passed else "FAIL",
            }
        )

    add(
        "PIPE-001",
        "Complete Phase 3 validation passed before execution",
        preflight_summary.get("status"),
        "PASS",
        preflight_summary.get("status") == "PASS",
    )
    add(
        "PIPE-002",
        "Complete Phase 3 validation had zero failures",
        preflight_summary.get("failed"),
        0,
        preflight_summary.get("failed") == 0,
    )
    add(
        "PIPE-003",
        "Selected modules follow the frozen execution order",
        list(selected_modules),
        [module for module in MODULE_ORDER if module in selected_modules],
        tuple(selected_modules)
        == tuple(module for module in MODULE_ORDER if module in selected_modules),
    )
    add(
        "PIPE-004",
        "Every selected module passed output validation",
        [item.get("status") for item in module_summaries],
        ["PASS"] * len(selected_modules),
        len(module_summaries) == len(selected_modules)
        and all(item.get("status") == "PASS" for item in module_summaries),
    )
    expected_manifest_count = sum(
        len(contract.allowed_outputs(module)) for module in selected_modules
    )
    add(
        "PIPE-005",
        "Output manifest row count matches the source contract",
        len(manifest_rows),
        expected_manifest_count,
        len(manifest_rows) == expected_manifest_count,
    )
    add(
        "PIPE-006",
        "Output manifest paths are unique",
        len({row["relative_path"] for row in manifest_rows}),
        len(manifest_rows),
        len({row["relative_path"] for row in manifest_rows}) == len(manifest_rows),
    )
    add(
        "PIPE-007",
        "Every manifest hash is a lowercase SHA-256 digest",
        all(
            isinstance(row["sha256"], str)
            and len(row["sha256"]) == 64
            and row["sha256"] == row["sha256"].lower()
            for row in manifest_rows
        ),
        True,
        all(
            isinstance(row["sha256"], str)
            and len(row["sha256"]) == 64
            and row["sha256"] == row["sha256"].lower()
            for row in manifest_rows
        ),
    )
    add(
        "PIPE-008",
        "Phase 3 tree remained byte-identical",
        phase3_after,
        phase3_before,
        phase3_after == phase3_before,
    )
    add(
        "PIPE-009",
        "No prohibited runtime file-access dependencies were detected",
        dict(source_scan_findings),
        {},
        not source_scan_findings,
    )
    add(
        "PIPE-010",
        "Every selected module execution record uses the active contract",
        [item.get("execution_record_sha256") for item in module_summaries],
        f"{len(selected_modules)} validated execution records",
        len(module_summaries) == len(selected_modules),
    )

    failed = sum(item["status"] == "FAIL" for item in checks)
    passed = len(checks) - failed
    return {
        "pipeline_validation_version": PIPELINE_VERSION,
        "status": "PASS" if failed == 0 else "FAIL",
        "checks": len(checks),
        "passed": passed,
        "warnings": 0,
        "failed": failed,
        "selected_modules": list(selected_modules),
        "preflight_validation": dict(preflight_summary),
        "module_validation": [dict(item) for item in module_summaries],
        "check_results": checks,
    }


def _execution_record(
    *,
    phase3_root: Path,
    results_root: Path,
    selected_modules: Sequence[str],
    preflight_summary: Mapping[str, Any],
    plan: Sequence[Mapping[str, Any]],
    module_summaries: Sequence[Mapping[str, Any]],
    phase3_inventory: Sequence[Mapping[str, Any]],
    phase3_tree_sha256: str,
    manifest_path: Path,
    validation_path: Path,
    manifest_rows: Sequence[Mapping[str, Any]],
    contract: Any,
) -> dict[str, Any]:
    unique_inputs: dict[str, str] = {}
    for module_name in selected_modules:
        for item in contract.files_for_module(module_name):
            unique_inputs[item.relative_path] = item.sha256
    module_sources = {
        module_name: {
            "path": _module_source_path(module_name)
            .relative_to(PHASE4_ROOT)
            .as_posix(),
            "sha256": sha256_file(_module_source_path(module_name)),
        }
        for module_name in selected_modules
    }
    return {
        "record_version": PIPELINE_VERSION,
        "pipeline_relative_path": PIPELINE_RELATIVE_PATH,
        "pipeline_sha256": sha256_file(Path(__file__).resolve()),
        "contract_relative_path": contract.path.relative_to(PHASE4_ROOT).as_posix(),
        "contract_sha256": contract.contract_sha256,
        "phase3_package": contract.runtime_input["package_name"],
        "phase3_root_name": phase3_root.name,
        "phase3_file_count": len(phase3_inventory),
        "phase3_tree_sha256": phase3_tree_sha256,
        "runtime_results_root": contract.raw["output_policy"]["runtime_results_root"],
        "selected_modules": list(selected_modules),
        "module_order": list(MODULE_ORDER),
        "preflight_validation": dict(preflight_summary),
        "execution_plan": [dict(item) for item in plan],
        "module_sources": module_sources,
        "authorized_inputs": {
            path: {"sha256": unique_inputs[path]} for path in sorted(unique_inputs)
        },
        "module_validation": [dict(item) for item in module_summaries],
        "output_manifest": {
            "relative_path": f"{PIPELINE_DIRECTORY_NAME}/{PIPELINE_MANIFEST_NAME}",
            "rows": len(manifest_rows),
            "sha256": sha256_file(manifest_path),
        },
        "pipeline_validation": {
            "relative_path": f"{PIPELINE_DIRECTORY_NAME}/{PIPELINE_VALIDATION_NAME}",
            "sha256": sha256_file(validation_path),
        },
        "command_template": [
            "python",
            "-m",
            "qstable_awm.pipeline",
            "--phase3-root",
            contract.runtime_input["root_directory_name"],
            "--results-root",
            contract.raw["output_policy"]["runtime_results_root"],
            "--contract",
            "config/phase4_source_contract.yml",
            "--overwrite",
        ],
        "dry_run": False,
        "atomic_commit": True,
        "timestamp_policy": "omitted_for_deterministic_outputs",
        "notebook_runtime_dependency": False,
        "phase3_modified": False,
        "scientific_boundary": (
            "Orchestration and reproducibility metadata only. Scientific values "
            "remain those produced and validated by the seven analytical modules."
        ),
    }


def _atomic_commit(
    *,
    staged_results_root: Path,
    results_root: Path,
    selected_modules: Sequence[str],
    overwrite: bool,
) -> None:
    results_root.mkdir(parents=True, exist_ok=True)
    names = [*selected_modules, PIPELINE_DIRECTORY_NAME]
    backup_root = Path(
        tempfile.mkdtemp(prefix=".phase4-pipeline-backup-", dir=results_root.parent)
    )
    moved_old: list[str] = []
    installed_new: list[str] = []
    try:
        for name in names:
            source = staged_results_root / name
            target = results_root / name
            if not source.is_dir():
                raise PipelineError(f"Missing staged directory: {source}")
            if target.exists():
                if not overwrite:
                    raise PipelineError(f"Refusing to overwrite: {target}")
                os.replace(target, backup_root / name)
                moved_old.append(name)
            os.replace(source, target)
            installed_new.append(name)
    except Exception:
        for name in reversed(installed_new):
            target = results_root / name
            if target.exists():
                shutil.rmtree(target)
        for name in reversed(moved_old):
            backup = backup_root / name
            target = results_root / name
            if backup.exists():
                os.replace(backup, target)
        raise
    finally:
        shutil.rmtree(backup_root, ignore_errors=True)


def run_pipeline(
    phase3_root: Path,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    *,
    modules: Sequence[str] | None = None,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
    overwrite: bool = False,
    dry_run: bool = False,
) -> PipelineRunResult:
    """Validate and execute selected Phase 4 modules deterministically."""

    phase3 = Path(phase3_root).expanduser().resolve()
    results = Path(results_root).expanduser().resolve()
    contract_file = Path(contract_path).expanduser().resolve()
    selected = _resolve_selected_modules(modules)
    _assert_safe_roots(phase3, results)

    contract = load_source_contract(contract_file)
    if tuple(contract.module_names) != MODULE_ORDER:
        raise PipelineError(
            f"Contract module order changed: {contract.module_names}; expected={MODULE_ORDER}"
        )

    preflight = validate_phase3_root(
        phase3,
        contract_path=contract_file,
        modules=None,
        validate_schemas=True,
    )
    preflight.raise_for_errors()
    if preflight.overall_status != "PASS":
        raise PipelineError(
            f"Complete Phase 3 preflight must be PASS; observed={preflight.overall_status}"
        )

    phase3_inventory_before, phase3_tree_before = _tree_inventory(phase3)
    source_paths = [Path(__file__).resolve()] + [
        _module_source_path(module) for module in selected
    ]
    source_scan = _scan_runtime_file_access(
        source_paths, contract.prohibited_runtime_tokens
    )
    if source_scan:
        raise PipelineError(
            f"Prohibited runtime file-access dependencies detected: {source_scan}"
        )

    plan = _build_plan(selected, results_root=results, contract=contract)
    if dry_run:
        return PipelineRunResult(
            dry_run=True,
            selected_modules=selected,
            preflight_validation=preflight.summary(),
            plan=plan,
            outputs=None,
        )

    _precheck_targets(results, selected, overwrite=overwrite)
    staging_parent = results.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=".phase4-pipeline-staging-", dir=staging_parent)
    )
    staged_results = staging_root / "results"
    staged_results.mkdir(parents=True)

    module_summaries: list[dict[str, Any]] = []
    try:
        for module_name in selected:
            run_function = _load_run_function(module_name)
            output_dir = staged_results / module_name
            run_function(
                phase3,
                output_dir,
                contract_path=contract_file,
                overwrite=False,
            )
            module_summaries.append(
                _validate_module_directory(
                    module_name, output_dir, contract=contract
                )
            )

        phase3_inventory_after, phase3_tree_after = _tree_inventory(phase3)
        if phase3_inventory_after != phase3_inventory_before:
            raise PipelineError("Phase 3 contents changed during pipeline execution")

        manifest_rows = _manifest_rows(
            staged_results, selected, contract=contract
        )
        pipeline_dir = staged_results / PIPELINE_DIRECTORY_NAME
        pipeline_dir.mkdir()
        manifest_path = pipeline_dir / PIPELINE_MANIFEST_NAME
        validation_path = pipeline_dir / PIPELINE_VALIDATION_NAME
        record_path = pipeline_dir / PIPELINE_EXECUTION_RECORD_NAME

        _write_manifest(manifest_path, manifest_rows)
        pipeline_validation = _pipeline_validation(
            selected_modules=selected,
            preflight_summary=preflight.summary(),
            module_summaries=module_summaries,
            manifest_rows=manifest_rows,
            phase3_before=phase3_tree_before,
            phase3_after=phase3_tree_after,
            source_scan_findings=source_scan,
            contract=contract,
        )
        if pipeline_validation["status"] != "PASS":
            raise PipelineError(
                f"Pipeline validation failed: {pipeline_validation['check_results']}"
            )
        _write_json(validation_path, pipeline_validation)
        record = _execution_record(
            phase3_root=phase3,
            results_root=results,
            selected_modules=selected,
            preflight_summary=preflight.summary(),
            plan=plan,
            module_summaries=module_summaries,
            phase3_inventory=phase3_inventory_before,
            phase3_tree_sha256=phase3_tree_before,
            manifest_path=manifest_path,
            validation_path=validation_path,
            manifest_rows=manifest_rows,
            contract=contract,
        )
        _write_json(record_path, record)

        for path in (manifest_path, validation_path, record_path):
            if not path.is_file() or path.stat().st_size <= 0:
                raise PipelineError(f"Pipeline metadata was not written: {path}")

        _atomic_commit(
            staged_results_root=staged_results,
            results_root=results,
            selected_modules=selected,
            overwrite=overwrite,
        )
    except Exception as exc:
        if isinstance(exc, (PipelineError, RuntimeInputError)):
            raise
        raise PipelineError(f"Pipeline execution failed: {exc}") from exc
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)

    outputs = PipelineOutputs(
        phase4_pipeline_output_manifest=(
            results / PIPELINE_DIRECTORY_NAME / PIPELINE_MANIFEST_NAME
        ),
        phase4_pipeline_validation=(
            results / PIPELINE_DIRECTORY_NAME / PIPELINE_VALIDATION_NAME
        ),
        phase4_pipeline_execution_record=(
            results / PIPELINE_DIRECTORY_NAME / PIPELINE_EXECUTION_RECORD_NAME
        ),
    )
    return PipelineRunResult(
        dry_run=False,
        selected_modules=selected,
        preflight_validation=preflight.summary(),
        plan=plan,
        outputs=outputs,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and run the Phase 4 analytical modules."
    )
    parser.add_argument("--phase3-root", required=True, type=Path)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument(
        "--module",
        action="append",
        choices=MODULE_ORDER,
        help="Run one or more modules. Omit to run all modules.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the plan without writing outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace selected module outputs and pipeline metadata.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        result = run_pipeline(
            args.phase3_root,
            args.results_root,
            modules=args.module,
            contract_path=args.contract,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
    except (PipelineError, RuntimeInputError, FileNotFoundError, ValueError) as exc:
        print(f"PIPELINE_ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
