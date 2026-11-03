"""Strict runtime validation for the Phase 4 analytical modules.

The validator enforces the publication authority chain before any manuscript
statistic is calculated:

1. the runtime root must be the frozen Phase 3 curated-input package;
2. every module input must be declared in ``phase4_source_contract.yml``;
3. every declared runtime file must match its frozen SHA-256 digest;
4. CSV, JSON, and raster inputs must satisfy the immutable schemas in
   :mod:`qstable_awm.schemas`;
5. Figure 8 and HLS guardrails must preserve the narrow V50 authority;
6. notebooks and original workflow archives remain provenance-only.

This module does not calculate manuscript results or generate figures. It is
safe to run directly and prints a concise validation summary. Analytical
modules should call :func:`validate_module_inputs` before reading any data.

Examples
--------
Validate the complete Phase 3 runtime layer::

    python -m qstable_awm.validation \
        --phase3-root PHASE3_CURATED_INPUT_LAYER_FINAL_002

Validate only the climate-context inputs::

    python -m qstable_awm.validation \
        --phase3-root PHASE3_CURATED_INPUT_LAYER_FINAL_002 \
        --module climate_context
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import rasterio
import yaml

try:
    from .schemas import (
        ALL_RUNTIME_PATHS,
        JSON_SCHEMAS,
        PHASE3_PACKAGE_NAME,
        RASTER_SCHEMAS,
        TABLE_SCHEMAS,
        ColumnRule,
        JsonSchema,
        RasterSchema,
        ScalarKind,
        TableSchema,
        assert_schema_registry_integrity,
        schema_for_relative_path,
    )
except ImportError:  # Support direct execution of this file before installation.
    _SRC_ROOT = Path(__file__).resolve().parents[1]
    if str(_SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(_SRC_ROOT))
    from qstable_awm.schemas import (  # type: ignore[no-redef]
        ALL_RUNTIME_PATHS,
        JSON_SCHEMAS,
        PHASE3_PACKAGE_NAME,
        RASTER_SCHEMAS,
        TABLE_SCHEMAS,
        ColumnRule,
        JsonSchema,
        RasterSchema,
        ScalarKind,
        TableSchema,
        assert_schema_registry_integrity,
        schema_for_relative_path,
    )


PHASE4_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT_PATH = PHASE4_ROOT / "config" / "phase4_source_contract.yml"
VALIDATION_REPORT_VERSION = "1.0.0"
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class ValidationSeverity(str, Enum):
    """Severity assigned to one deterministic validation check."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class ValidationStatus(str, Enum):
    """Outcome assigned to one deterministic validation check."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True, slots=True)
class ValidationCheck:
    """One machine-readable validation result."""

    check_id: str
    scope: str
    description: str
    severity: ValidationSeverity
    status: ValidationStatus
    observed: str
    expected: str
    relative_path: str = ""


@dataclass(slots=True)
class ValidationReport:
    """Complete validation result for a Phase 3 runtime root."""

    phase3_root: str
    contract_path: str
    modules: tuple[str, ...]
    checks: list[ValidationCheck] = field(default_factory=list)
    report_version: str = VALIDATION_REPORT_VERSION

    def add(
        self,
        *,
        check_id: str,
        scope: str,
        description: str,
        observed: Any,
        expected: Any,
        passed: bool,
        severity: ValidationSeverity = ValidationSeverity.ERROR,
        relative_path: str = "",
    ) -> None:
        """Append a check using deterministic status semantics."""

        if passed:
            status = ValidationStatus.PASS
        elif severity is ValidationSeverity.WARNING:
            status = ValidationStatus.WARN
        else:
            status = ValidationStatus.FAIL
        self.checks.append(
            ValidationCheck(
                check_id=check_id,
                scope=scope,
                description=description,
                severity=severity,
                status=status,
                observed=_stable_text(observed),
                expected=_stable_text(expected),
                relative_path=relative_path,
            )
        )

    @property
    def failed_checks(self) -> tuple[ValidationCheck, ...]:
        return tuple(c for c in self.checks if c.status is ValidationStatus.FAIL)

    @property
    def warning_checks(self) -> tuple[ValidationCheck, ...]:
        return tuple(c for c in self.checks if c.status is ValidationStatus.WARN)

    @property
    def passed_checks(self) -> tuple[ValidationCheck, ...]:
        return tuple(c for c in self.checks if c.status is ValidationStatus.PASS)

    @property
    def is_valid(self) -> bool:
        return not self.failed_checks

    @property
    def overall_status(self) -> str:
        if self.failed_checks:
            return "FAIL"
        if self.warning_checks:
            return "PASS_WITH_WARNINGS"
        return "PASS"

    def summary(self) -> dict[str, Any]:
        """Return a compact deterministic summary."""

        return {
            "report_version": self.report_version,
            "phase3_root": self.phase3_root,
            "contract_path": self.contract_path,
            "modules": list(self.modules),
            "status": self.overall_status,
            "checks": len(self.checks),
            "passed": len(self.passed_checks),
            "warnings": len(self.warning_checks),
            "failed": len(self.failed_checks),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-serializable report."""

        return {
            **self.summary(),
            "check_results": [
                {
                    **asdict(check),
                    "severity": check.severity.value,
                    "status": check.status.value,
                }
                for check in self.checks
            ],
        }

    def raise_for_errors(self) -> None:
        """Raise :class:`RuntimeInputError` when any ERROR check failed."""

        if not self.failed_checks:
            return
        details = "\n".join(
            f"- {c.check_id}: {c.description}; observed={c.observed}; "
            f"expected={c.expected}"
            for c in self.failed_checks[:20]
        )
        suffix = "" if len(self.failed_checks) <= 20 else "\n- additional failures omitted"
        raise RuntimeInputError(
            "Phase 4 runtime-input validation failed:\n" + details + suffix
        )


class RuntimeInputError(RuntimeError):
    """Raised when a Phase 4 input violates the frozen source contract."""


@dataclass(frozen=True, slots=True)
class RuntimeFile:
    """One file declared as a runtime dependency in the source contract."""

    module: str
    relative_path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class SourceContract:
    """Parsed and internally validated source-contract representation."""

    path: Path
    raw: Mapping[str, Any]
    contract_sha256: str

    @property
    def phase4_package(self) -> str:
        return str(self.raw["phase4_package"])

    @property
    def runtime_input(self) -> Mapping[str, Any]:
        return _require_mapping(self.raw, "runtime_input")

    @property
    def runtime_policy(self) -> Mapping[str, Any]:
        return _require_mapping(self.raw, "runtime_policy")

    @property
    def module_inputs(self) -> Mapping[str, Any]:
        return _require_mapping(self.raw, "module_inputs")

    @property
    def module_names(self) -> tuple[str, ...]:
        return tuple(self.module_inputs.keys())

    @property
    def prohibited_runtime_tokens(self) -> tuple[str, ...]:
        values = self.raw.get("prohibited_runtime_tokens", ())
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            raise RuntimeInputError("prohibited_runtime_tokens must be a list of strings")
        return tuple(values)

    def files_for_module(self, module: str) -> tuple[RuntimeFile, ...]:
        """Return the exact input set allowed for one analytical module."""

        try:
            config = self.module_inputs[module]
        except KeyError as exc:
            raise RuntimeInputError(
                f"Unknown Phase 4 module {module!r}; allowed={self.module_names}"
            ) from exc
        if not isinstance(config, Mapping):
            raise RuntimeInputError(f"module_inputs.{module} must be a mapping")
        entries = config.get("required_files")
        if not isinstance(entries, list) or not entries:
            raise RuntimeInputError(
                f"module_inputs.{module}.required_files must be a non-empty list"
            )
        files: list[RuntimeFile] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise RuntimeInputError(
                    f"module_inputs.{module}.required_files[{index}] must be a mapping"
                )
            relative_path = entry.get("path")
            digest = entry.get("sha256")
            if not isinstance(relative_path, str) or not relative_path:
                raise RuntimeInputError(
                    f"Missing path for module {module!r}, required file {index}"
                )
            if not isinstance(digest, str) or not _HASH_PATTERN.fullmatch(digest):
                raise RuntimeInputError(
                    f"Invalid SHA-256 for module {module!r}: {relative_path!r}"
                )
            files.append(RuntimeFile(module, relative_path, digest))
        return tuple(files)

    def required_files(
        self, modules: Sequence[str] | None = None
    ) -> tuple[RuntimeFile, ...]:
        """Return de-duplicated files for the requested modules."""

        selected = tuple(modules) if modules else self.module_names
        by_path: dict[str, RuntimeFile] = {}
        for module in selected:
            for item in self.files_for_module(module):
                existing = by_path.get(item.relative_path)
                if existing and existing.sha256 != item.sha256:
                    raise RuntimeInputError(
                        "Conflicting SHA-256 declarations for "
                        f"{item.relative_path!r}: {existing.sha256} vs {item.sha256}"
                    )
                by_path[item.relative_path] = item
        return tuple(by_path[path] for path in sorted(by_path))

    def allowed_outputs(self, module: str) -> tuple[str, ...]:
        """Return the only public filenames a module may write."""

        config = self.module_inputs.get(module)
        if not isinstance(config, Mapping):
            raise RuntimeInputError(f"Unknown Phase 4 module: {module!r}")
        values = config.get("allowed_outputs")
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            raise RuntimeInputError(
                f"module_inputs.{module}.allowed_outputs must be a list of strings"
            )
        return tuple(values)


# ---------------------------------------------------------------------------
# Contract and path utilities
# ---------------------------------------------------------------------------


def _stable_text(value: Any) -> str:
    if isinstance(value, (dict, list, tuple, set)):
        try:
            if isinstance(value, set):
                value = sorted(value)
            return json.dumps(value, sort_keys=True, ensure_ascii=False)
        except TypeError:
            return repr(value)
    return str(value)


def _require_mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise RuntimeInputError(f"Contract key {key!r} must be a mapping")
    return value


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the lowercase SHA-256 digest of a local file."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source_contract(path: Path = DEFAULT_CONTRACT_PATH) -> SourceContract:
    """Load and internally validate ``phase4_source_contract.yml``."""

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise RuntimeInputError(f"Source contract not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise RuntimeInputError("The Phase 4 source contract must contain a mapping")

    required_top_level = {
        "contract_version",
        "phase4_package",
        "purpose",
        "runtime_input",
        "runtime_policy",
        "notebook_policy",
        "module_inputs",
        "prohibited_runtime_tokens",
        "output_policy",
    }
    missing = required_top_level - set(data)
    if missing:
        raise RuntimeInputError(
            f"Source contract is missing top-level keys: {sorted(missing)}"
        )

    contract = SourceContract(path=path, raw=data, contract_sha256=sha256_file(path))
    _assert_contract_integrity(contract)
    return contract


def _assert_contract_integrity(contract: SourceContract) -> None:
    """Fail on inconsistent or unsafe contract metadata."""

    runtime = contract.runtime_input
    if runtime.get("package_name") != PHASE3_PACKAGE_NAME:
        raise RuntimeInputError(
            "Source contract package_name does not match schemas.PHASE3_PACKAGE_NAME"
        )
    if runtime.get("root_directory_name") != PHASE3_PACKAGE_NAME:
        raise RuntimeInputError(
            "Source contract root_directory_name does not match the frozen Phase 3 name"
        )

    required_policy = {
        "notebooks_are_runtime_inputs": False,
        "original_workflow_archives_are_runtime_inputs": False,
        "manuscript_text_is_a_numeric_data_source": False,
        "hard_coded_manuscript_results_allowed": False,
        "absolute_internal_paths_allowed": False,
        "inputs_outside_phase3_root_allowed": False,
        "scientific_values_must_be_derived_from_curated_inputs": True,
        "every_runtime_file_requires_sha256_match": True,
        "fail_on_unexpected_input_file": True,
        "fail_on_missing_required_input": True,
    }
    for key, expected in required_policy.items():
        observed = contract.runtime_policy.get(key)
        if observed is not expected:
            raise RuntimeInputError(
                f"Unsafe runtime policy {key!r}: observed={observed!r}, "
                f"expected={expected!r}"
            )

    all_files = contract.required_files()
    paths = [item.relative_path for item in all_files]
    if len(paths) != len(set(paths)):
        raise RuntimeInputError("Duplicate runtime paths remain after contract resolution")
    for item in all_files:
        _validate_relative_runtime_path(item.relative_path)

    # Every structured schema must be tied to an explicit contract hash.
    missing_schema_paths = set(ALL_RUNTIME_PATHS) - set(paths)
    if missing_schema_paths:
        raise RuntimeInputError(
            "Schema paths absent from the source contract: "
            f"{sorted(missing_schema_paths)}"
        )

    guardrails = contract.module_inputs["interpretation_classes"].get(
        "classification_guardrails"
    )
    if not isinstance(guardrails, Mapping):
        raise RuntimeInputError("Missing interpretation-class guardrails")
    expected_guardrails = {
        "class_i_candidate_is_completed_class_i": False,
        "completed_class_i_assignment_present": False,
        "hls_is_et_validation": False,
        "hls_is_pumping_validation": False,
    }
    for key, expected in expected_guardrails.items():
        if guardrails.get(key) is not expected:
            raise RuntimeInputError(
                f"Unsafe interpretation guardrail {key!r}: {guardrails.get(key)!r}"
            )


def _validate_relative_runtime_path(relative_path: str) -> None:
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise RuntimeInputError(f"Runtime path must be relative: {relative_path!r}")
    if not candidate.parts or ".." in candidate.parts:
        raise RuntimeInputError(f"Unsafe runtime path: {relative_path!r}")
    if candidate.suffix.lower() == ".ipynb":
        raise RuntimeInputError(
            f"Notebook cannot be a Phase 4 runtime input: {relative_path!r}"
        )
    if candidate.parts[0] == "provenance":
        raise RuntimeInputError(
            f"Provenance file cannot be an analytical runtime input: {relative_path!r}"
        )


def _resolve_inside_root(phase3_root: Path, relative_path: str) -> Path:
    """Resolve any package-relative path while preventing root escape."""

    candidate = Path(relative_path)
    if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
        raise RuntimeInputError(f"Unsafe package-relative path: {relative_path!r}")
    root = Path(phase3_root).expanduser().resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeInputError(
            f"Package path escapes the Phase 3 root: {relative_path!r}"
        ) from exc
    return resolved


def resolve_runtime_path(phase3_root: Path, relative_path: str) -> Path:
    """Resolve an analytical input and prove it remains inside Phase 3."""

    _validate_relative_runtime_path(relative_path)
    return _resolve_inside_root(phase3_root, relative_path)


def assert_module_input_paths(
    contract: SourceContract,
    phase3_root: Path,
    module: str,
    supplied_paths: Iterable[Path | str],
) -> tuple[Path, ...]:
    """Reject missing, unexpected, duplicated, or out-of-root module inputs.

    Analytical modules should call this function on the paths they intend to
    read. The supplied set must equal the module's contract set exactly.
    """

    root = Path(phase3_root).expanduser().resolve()
    expected = {
        resolve_runtime_path(root, item.relative_path)
        for item in contract.files_for_module(module)
    }
    supplied_list = [Path(path).expanduser().resolve() for path in supplied_paths]
    supplied = set(supplied_list)
    if len(supplied) != len(supplied_list):
        raise RuntimeInputError(f"Duplicate supplied input paths for module {module!r}")
    for path in supplied:
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise RuntimeInputError(
                f"Module {module!r} attempted to read outside Phase 3: {path}"
            ) from exc
    missing = expected - supplied
    unexpected = supplied - expected
    if missing or unexpected:
        raise RuntimeInputError(
            f"Input-set mismatch for module {module!r}; "
            f"missing={[str(p) for p in sorted(missing)]}; "
            f"unexpected={[str(p) for p in sorted(unexpected)]}"
        )
    return tuple(sorted(supplied))


def assert_allowed_output_name(
    contract: SourceContract, module: str, output_name: str
) -> None:
    """Reject an output filename not assigned to the module."""

    if Path(output_name).name != output_name:
        raise RuntimeInputError("Output name must be a basename, not a path")
    allowed = contract.allowed_outputs(module)
    if output_name not in allowed:
        raise RuntimeInputError(
            f"Output {output_name!r} is not allowed for module {module!r}; "
            f"allowed={allowed}"
        )


# ---------------------------------------------------------------------------
# Scalar and structured-data validation
# ---------------------------------------------------------------------------


def _non_null_values(series: pd.Series) -> pd.Series:
    return series[~series.isna()]


def _parse_boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(value)


def _normalize_value(value: Any, kind: ScalarKind) -> Any:
    if kind is ScalarKind.STRING:
        if not isinstance(value, str):
            raise ValueError(value)
        return value
    if kind is ScalarKind.INTEGER:
        number = float(str(value).strip())
        if not math.isfinite(number) or not number.is_integer():
            raise ValueError(value)
        return int(number)
    if kind is ScalarKind.FLOAT:
        number = float(str(value).strip())
        if not math.isfinite(number):
            raise ValueError(value)
        return number
    if kind is ScalarKind.BOOLEAN:
        return _parse_boolean(value)
    if kind is ScalarKind.DATE:
        text = str(value).strip()
        if not _DATE_PATTERN.fullmatch(text):
            raise ValueError(value)
        datetime.strptime(text, "%Y-%m-%d")
        return text
    raise TypeError(f"Unsupported scalar kind: {kind}")


def _validate_column(
    report: ValidationReport,
    *,
    schema: TableSchema,
    column: str,
    rule: ColumnRule,
    series: pd.Series,
    check_prefix: str,
) -> None:
    scope = f"table:{schema.name}"
    relative_path = schema.relative_path
    null_count = int(series.isna().sum())
    report.add(
        check_id=f"{check_prefix}-NULL-{column}",
        scope=scope,
        description=f"Nullability for {column}",
        observed=null_count,
        expected="allowed" if rule.nullable else 0,
        passed=rule.nullable or null_count == 0,
        relative_path=relative_path,
    )

    values = _non_null_values(series)
    normalized: list[Any] = []
    invalid_examples: list[str] = []
    for value in values.tolist():
        try:
            normalized.append(_normalize_value(value, rule.kind))
        except (TypeError, ValueError, OverflowError):
            if len(invalid_examples) < 5:
                invalid_examples.append(repr(value))
    report.add(
        check_id=f"{check_prefix}-TYPE-{column}",
        scope=scope,
        description=f"Scalar kind for {column}",
        observed=(
            f"{len(invalid_examples)} invalid examples: {invalid_examples}"
            if invalid_examples
            else f"{len(normalized)} valid non-null values"
        ),
        expected=rule.kind.value,
        passed=not invalid_examples and len(normalized) == len(values),
        relative_path=relative_path,
    )
    if invalid_examples or not normalized:
        return

    if rule.allowed_values:
        allowed = {
            _normalize_value(value, rule.kind) for value in rule.allowed_values
        }
        unexpected = sorted(
            {value for value in normalized if value not in allowed}, key=repr
        )
        report.add(
            check_id=f"{check_prefix}-ALLOWED-{column}",
            scope=scope,
            description=f"Accepted values for {column}",
            observed=unexpected[:10],
            expected=sorted(allowed, key=repr),
            passed=not unexpected,
            relative_path=relative_path,
        )

    if rule.regex is not None:
        pattern = re.compile(rule.regex)
        nonmatching = [
            str(value) for value in normalized if pattern.fullmatch(str(value)) is None
        ]
        report.add(
            check_id=f"{check_prefix}-REGEX-{column}",
            scope=scope,
            description=f"Pattern for {column}",
            observed=nonmatching[:10],
            expected=rule.regex,
            passed=not nonmatching,
            relative_path=relative_path,
        )

    if rule.min_value is not None:
        below = [
            value for value in normalized if isinstance(value, (int, float))
            and value < rule.min_value
        ]
        report.add(
            check_id=f"{check_prefix}-MIN-{column}",
            scope=scope,
            description=f"Minimum for {column}",
            observed=below[:10],
            expected=f">= {rule.min_value}",
            passed=not below,
            relative_path=relative_path,
        )

    if rule.max_value is not None:
        above = [
            value for value in normalized if isinstance(value, (int, float))
            and value > rule.max_value
        ]
        report.add(
            check_id=f"{check_prefix}-MAX-{column}",
            scope=scope,
            description=f"Maximum for {column}",
            observed=above[:10],
            expected=f"<= {rule.max_value}",
            passed=not above,
            relative_path=relative_path,
        )


def _validate_table(
    report: ValidationReport,
    *,
    root: Path,
    schema: TableSchema,
    sequence: int,
) -> None:
    path = resolve_runtime_path(root, schema.relative_path)
    prefix = f"TAB-{sequence:02d}"
    if not path.is_file():
        report.add(
            check_id=f"{prefix}-EXISTS",
            scope=f"table:{schema.name}",
            description="Required CSV exists",
            observed=False,
            expected=True,
            passed=False,
            relative_path=schema.relative_path,
        )
        return

    try:
        frame = pd.read_csv(path, dtype=object, keep_default_na=True)
    except Exception as exc:  # pandas surfaces parser details in observed text.
        report.add(
            check_id=f"{prefix}-READ",
            scope=f"table:{schema.name}",
            description="CSV is readable",
            observed=f"{type(exc).__name__}: {exc}",
            expected="readable CSV",
            passed=False,
            relative_path=schema.relative_path,
        )
        return

    report.add(
        check_id=f"{prefix}-READ",
        scope=f"table:{schema.name}",
        description="CSV is readable",
        observed=True,
        expected=True,
        passed=True,
        relative_path=schema.relative_path,
    )
    report.add(
        check_id=f"{prefix}-ROWS",
        scope=f"table:{schema.name}",
        description="Exact row count",
        observed=len(frame),
        expected=schema.expected_rows,
        passed=len(frame) == schema.expected_rows,
        relative_path=schema.relative_path,
    )
    report.add(
        check_id=f"{prefix}-COLUMNS",
        scope=f"table:{schema.name}",
        description="Exact column names and order",
        observed=list(frame.columns),
        expected=list(schema.exact_columns),
        passed=tuple(frame.columns) == schema.exact_columns,
        relative_path=schema.relative_path,
    )

    if not set(schema.primary_key).issubset(frame.columns):
        report.add(
            check_id=f"{prefix}-KEY",
            scope=f"table:{schema.name}",
            description="Primary-key columns are present and unique",
            observed="missing key columns",
            expected=list(schema.primary_key),
            passed=False,
            relative_path=schema.relative_path,
        )
    else:
        key_nulls = int(frame[list(schema.primary_key)].isna().any(axis=1).sum())
        duplicate_count = int(
            frame.duplicated(subset=list(schema.primary_key), keep=False).sum()
        )
        report.add(
            check_id=f"{prefix}-KEY",
            scope=f"table:{schema.name}",
            description="Primary-key columns are non-null and unique",
            observed={"key_null_rows": key_nulls, "duplicate_rows": duplicate_count},
            expected={"key_null_rows": 0, "duplicate_rows": 0},
            passed=key_nulls == 0 and duplicate_count == 0,
            relative_path=schema.relative_path,
        )

    if tuple(frame.columns) != schema.exact_columns:
        return
    for column in schema.exact_columns:
        _validate_column(
            report,
            schema=schema,
            column=column,
            rule=schema.column_rules[column],
            series=frame[column],
            check_prefix=prefix,
        )


def _validate_json(
    report: ValidationReport,
    *,
    root: Path,
    schema: JsonSchema,
    sequence: int,
) -> None:
    path = resolve_runtime_path(root, schema.relative_path)
    prefix = f"JSON-{sequence:02d}"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        report.add(
            check_id=f"{prefix}-READ",
            scope=f"json:{schema.name}",
            description="JSON is readable",
            observed=f"{type(exc).__name__}: {exc}",
            expected="readable JSON object",
            passed=False,
            relative_path=schema.relative_path,
        )
        return

    report.add(
        check_id=f"{prefix}-READ",
        scope=f"json:{schema.name}",
        description="JSON is readable",
        observed=type(value).__name__,
        expected="dict",
        passed=isinstance(value, dict),
        relative_path=schema.relative_path,
    )
    if not isinstance(value, dict):
        return
    required = set(schema.required_top_level_keys)
    observed = set(value)
    passed = observed == required if schema.exact_top_level_keys else required <= observed
    report.add(
        check_id=f"{prefix}-KEYS",
        scope=f"json:{schema.name}",
        description="Top-level JSON keys",
        observed=sorted(observed),
        expected=sorted(required),
        passed=passed,
        relative_path=schema.relative_path,
    )


def _validate_raster(
    report: ValidationReport,
    *,
    root: Path,
    schema: RasterSchema,
    sequence: int,
) -> None:
    path = resolve_runtime_path(root, schema.relative_path)
    prefix = f"RAS-{sequence:02d}"
    try:
        dataset = rasterio.open(path)
    except Exception as exc:
        report.add(
            check_id=f"{prefix}-READ",
            scope=f"raster:{schema.name}",
            description="Raster is readable",
            observed=f"{type(exc).__name__}: {exc}",
            expected="readable GeoTIFF",
            passed=False,
            relative_path=schema.relative_path,
        )
        return

    with dataset:
        checks = [
            ("DRIVER", "Raster driver", dataset.driver, schema.driver),
            ("BANDS", "Raster band count", dataset.count, schema.band_count),
            (
                "DESCRIPTIONS",
                "Raster band descriptions",
                tuple(dataset.descriptions),
                schema.band_descriptions,
            ),
            ("CRS", "Raster CRS", dataset.crs.to_string() if dataset.crs else None, schema.crs),
            ("RES-X", "Raster x resolution", abs(float(dataset.res[0])), schema.pixel_size_x),
            ("RES-Y", "Raster y resolution", abs(float(dataset.res[1])), schema.pixel_size_y),
            ("DTYPE", "Raster data types", tuple(dataset.dtypes), (schema.dtype,) * schema.band_count),
        ]
        report.add(
            check_id=f"{prefix}-READ",
            scope=f"raster:{schema.name}",
            description="Raster is readable",
            observed=True,
            expected=True,
            passed=True,
            relative_path=schema.relative_path,
        )
        for suffix, description, observed, expected in checks:
            if isinstance(expected, float):
                passed = math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=1e-9)
            else:
                passed = observed == expected
            report.add(
                check_id=f"{prefix}-{suffix}",
                scope=f"raster:{schema.name}",
                description=description,
                observed=observed,
                expected=expected,
                passed=passed,
                relative_path=schema.relative_path,
            )
        tags = {str(k).lower(): str(v).lower() for k, v in dataset.tags().items()}
        use_text = " ".join(tags.values())
        report.add(
            check_id=f"{prefix}-BOUNDARY",
            scope=f"raster:{schema.name}",
            description="Compact raster retains visualization-only boundary",
            observed=tags,
            expected="visualization-only use tag",
            passed=("visualization" in use_text or "visualisation" in use_text),
            relative_path=schema.relative_path,
        )


# ---------------------------------------------------------------------------
# Package, hash, and scientific-boundary validation
# ---------------------------------------------------------------------------


def _manifest_lookup(root: Path, relative_path: str) -> tuple[str, int] | None:
    manifest_path = root / "package_file_manifest.csv"
    if not manifest_path.is_file():
        return None
    frame = pd.read_csv(manifest_path, dtype=object)
    expected_columns = {"relative_path", "size_bytes", "sha256"}
    if not expected_columns.issubset(frame.columns):
        return None
    matches = frame[frame["relative_path"] == relative_path]
    if len(matches) != 1:
        return None
    row = matches.iloc[0]
    return str(row["sha256"]), int(float(row["size_bytes"]))


def _validate_package_anchors(
    report: ValidationReport, *, root: Path, contract: SourceContract
) -> None:
    runtime = contract.runtime_input
    report.add(
        check_id="PKG-001",
        scope="package",
        description="Runtime root directory name",
        observed=root.name,
        expected=runtime["root_directory_name"],
        passed=root.name == runtime["root_directory_name"],
    )

    version = _require_mapping(runtime, "version_file")
    version_path = _resolve_inside_root(root, str(version["path"]))
    version_exists = version_path.is_file()
    report.add(
        check_id="PKG-002",
        scope="package",
        description="VERSION file exists",
        observed=version_exists,
        expected=True,
        passed=version_exists,
        relative_path=str(version["path"]),
    )
    if version_exists:
        report.add(
            check_id="PKG-003",
            scope="package",
            description="VERSION file SHA-256",
            observed=sha256_file(version_path),
            expected=version["sha256"],
            passed=sha256_file(version_path) == version["sha256"],
            relative_path=str(version["path"]),
        )
        observed_text = version_path.read_text(encoding="utf-8").strip()
        report.add(
            check_id="PKG-004",
            scope="package",
            description="VERSION file text",
            observed=observed_text,
            expected=version["expected_text"],
            passed=observed_text == version["expected_text"],
            relative_path=str(version["path"]),
        )

    manifest = _require_mapping(runtime, "package_manifest")
    manifest_path = _resolve_inside_root(root, str(manifest["path"]))
    manifest_exists = manifest_path.is_file()
    report.add(
        check_id="PKG-005",
        scope="package",
        description="Package manifest exists",
        observed=manifest_exists,
        expected=True,
        passed=manifest_exists,
        relative_path=str(manifest["path"]),
    )
    if manifest_exists:
        observed_hash = sha256_file(manifest_path)
        report.add(
            check_id="PKG-006",
            scope="package",
            description="Package manifest SHA-256",
            observed=observed_hash,
            expected=manifest["sha256"],
            passed=observed_hash == manifest["sha256"],
            relative_path=str(manifest["path"]),
        )

    validation = _require_mapping(runtime, "validation_summary")
    validation_path = _resolve_inside_root(root, str(validation["path"]))
    validation_exists = validation_path.is_file()
    report.add(
        check_id="PKG-007",
        scope="package",
        description="Phase 3 validation summary exists",
        observed=validation_exists,
        expected=True,
        passed=validation_exists,
        relative_path=str(validation["path"]),
    )
    if validation_exists:
        observed_hash = sha256_file(validation_path)
        report.add(
            check_id="PKG-008",
            scope="package",
            description="Phase 3 validation summary SHA-256",
            observed=observed_hash,
            expected=validation["sha256"],
            passed=observed_hash == validation["sha256"],
            relative_path=str(validation["path"]),
        )
        try:
            payload = json.loads(validation_path.read_text(encoding="utf-8"))
            status = payload.get("status") if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            status = None
        accepted = tuple(validation.get("accepted_statuses", ()))
        report.add(
            check_id="PKG-009",
            scope="package",
            description="Phase 3 validation status is accepted",
            observed=status,
            expected=accepted,
            passed=status in accepted,
            relative_path=str(validation["path"]),
        )


def _validate_runtime_hashes(
    report: ValidationReport,
    *,
    root: Path,
    contract: SourceContract,
    modules: Sequence[str],
) -> None:
    for index, item in enumerate(contract.required_files(modules), start=1):
        path = resolve_runtime_path(root, item.relative_path)
        prefix = f"HASH-{index:02d}"
        exists = path.is_file()
        report.add(
            check_id=f"{prefix}-EXISTS",
            scope=f"module:{item.module}",
            description="Required runtime file exists",
            observed=exists,
            expected=True,
            passed=exists,
            relative_path=item.relative_path,
        )
        if not exists:
            continue
        observed_hash = sha256_file(path)
        report.add(
            check_id=f"{prefix}-SHA256",
            scope=f"module:{item.module}",
            description="Runtime file SHA-256",
            observed=observed_hash,
            expected=item.sha256,
            passed=observed_hash == item.sha256,
            relative_path=item.relative_path,
        )
        manifest_entry = _manifest_lookup(root, item.relative_path)
        if manifest_entry is None:
            report.add(
                check_id=f"{prefix}-MANIFEST",
                scope=f"module:{item.module}",
                description="Runtime file has one package-manifest record",
                observed=None,
                expected={"sha256": item.sha256, "size_bytes": path.stat().st_size},
                passed=False,
                relative_path=item.relative_path,
            )
        else:
            manifest_hash, manifest_size = manifest_entry
            report.add(
                check_id=f"{prefix}-MANIFEST",
                scope=f"module:{item.module}",
                description="Runtime file agrees with package manifest",
                observed={"sha256": manifest_hash, "size_bytes": manifest_size},
                expected={"sha256": observed_hash, "size_bytes": path.stat().st_size},
                passed=(
                    manifest_hash == observed_hash
                    and manifest_size == path.stat().st_size
                ),
                relative_path=item.relative_path,
            )


def _validate_structured_schemas(
    report: ValidationReport,
    *,
    root: Path,
    contract: SourceContract,
    modules: Sequence[str],
) -> None:
    selected_paths = {
        item.relative_path for item in contract.required_files(modules)
    }
    table_schemas = sorted(
        (s for s in TABLE_SCHEMAS.values() if s.relative_path in selected_paths),
        key=lambda s: s.relative_path,
    )
    json_schemas = sorted(
        (s for s in JSON_SCHEMAS.values() if s.relative_path in selected_paths),
        key=lambda s: s.relative_path,
    )
    raster_schemas = sorted(
        (s for s in RASTER_SCHEMAS.values() if s.relative_path in selected_paths),
        key=lambda s: s.relative_path,
    )
    for index, schema in enumerate(table_schemas, start=1):
        _validate_table(report, root=root, schema=schema, sequence=index)
    for index, schema in enumerate(json_schemas, start=1):
        _validate_json(report, root=root, schema=schema, sequence=index)
    for index, schema in enumerate(raster_schemas, start=1):
        _validate_raster(report, root=root, schema=schema, sequence=index)


def _validate_interpretation_guardrails(
    report: ValidationReport, *, root: Path, selected_modules: Sequence[str]
) -> None:
    if "interpretation_classes" not in selected_modules:
        return
    config_path = root / "data/derived/interpretation_classes/interpretation_classes.json"
    hls_path = root / "data/derived/interpretation_classes/non2011_hls_temporal_support_evidence.json"
    table_path = root / "data/derived/interpretation_classes/interpretation_classification.csv"

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        decision = config["release_decision"]
    except Exception as exc:
        report.add(
            check_id="SCI-001",
            scope="scientific-boundary",
            description="Interpretation decision is readable",
            observed=f"{type(exc).__name__}: {exc}",
            expected="valid release_decision",
            passed=False,
            relative_path=config_path.relative_to(root).as_posix(),
        )
        return

    expected_decisions = {
        "v50_included_narrowly": True,
        "hls_included_as_support_only": True,
        "v49_known_cold_evidence_included": False,
        "class_I_candidate_present": True,
        "completed_class_I_assignment_present": False,
        "class_I_candidate_is_not_class_I": True,
    }
    for index, (key, expected) in enumerate(expected_decisions.items(), start=1):
        report.add(
            check_id=f"SCI-{index:03d}",
            scope="scientific-boundary",
            description=f"Figure 8 release decision: {key}",
            observed=decision.get(key),
            expected=expected,
            passed=decision.get(key) is expected,
            relative_path=config_path.relative_to(root).as_posix(),
        )

    try:
        hls = json.loads(hls_path.read_text(encoding="utf-8"))
        identity_pass = (
            int(hls["object_years"]) * int(hls["hls_temporal_bins"])
            == int(hls["hls_object_bin_records"])
            == 71332
        )
        narrow_counts_pass = (
            int(hls["object_years"]) == 4196
            and int(hls["accepted_anchor_intervals"]) == 26085
            and int(hls["hls_temporal_bins"]) == 17
        )
        support_text = str(hls.get("hls_role", "")).lower()
        non_validation = not bool(hls.get("completed_class_I_assignment"))
    except Exception as exc:
        report.add(
            check_id="SCI-020",
            scope="scientific-boundary",
            description="HLS evidence is readable",
            observed=f"{type(exc).__name__}: {exc}",
            expected="valid narrow V50 evidence",
            passed=False,
            relative_path=hls_path.relative_to(root).as_posix(),
        )
    else:
        report.add(
            check_id="SCI-020",
            scope="scientific-boundary",
            description="HLS object-bin identity",
            observed={
                "object_years": hls["object_years"],
                "bins": hls["hls_temporal_bins"],
                "records": hls["hls_object_bin_records"],
            },
            expected="4196 × 17 = 71332",
            passed=identity_pass,
            relative_path=hls_path.relative_to(root).as_posix(),
        )
        report.add(
            check_id="SCI-021",
            scope="scientific-boundary",
            description="Narrow V50 counts",
            observed={
                "object_years": hls["object_years"],
                "intervals": hls["accepted_anchor_intervals"],
                "bins": hls["hls_temporal_bins"],
            },
            expected={"object_years": 4196, "intervals": 26085, "bins": 17},
            passed=narrow_counts_pass,
            relative_path=hls_path.relative_to(root).as_posix(),
        )
        report.add(
            check_id="SCI-022",
            scope="scientific-boundary",
            description="HLS role is temporal-support QA",
            observed=support_text,
            expected="contains support or plausibility language",
            passed=("support" in support_text or "plausibility" in support_text),
            relative_path=hls_path.relative_to(root).as_posix(),
        )
        report.add(
            check_id="SCI-023",
            scope="scientific-boundary",
            description="HLS does not complete Class I assignment",
            observed=hls.get("completed_class_I_assignment"),
            expected=False,
            passed=non_validation,
            relative_path=hls_path.relative_to(root).as_posix(),
        )

    try:
        classes = pd.read_csv(table_path, dtype=object)
        final_classes = classes["final_class"].tolist()
        candidate_count = int((classes["final_class"] == "Class I candidate").sum())
        completed_count = int((classes["final_class"] == "Class I").sum())
    except Exception as exc:
        report.add(
            check_id="SCI-030",
            scope="scientific-boundary",
            description="Interpretation-class table is readable",
            observed=f"{type(exc).__name__}: {exc}",
            expected="four class rows",
            passed=False,
            relative_path=table_path.relative_to(root).as_posix(),
        )
    else:
        report.add(
            check_id="SCI-030",
            scope="scientific-boundary",
            description="Exactly one Class I-candidate row",
            observed={"candidate_count": candidate_count, "classes": final_classes},
            expected={"candidate_count": 1},
            passed=candidate_count == 1,
            relative_path=table_path.relative_to(root).as_posix(),
        )
        report.add(
            check_id="SCI-031",
            scope="scientific-boundary",
            description="No completed Class I row",
            observed=completed_count,
            expected=0,
            passed=completed_count == 0,
            relative_path=table_path.relative_to(root).as_posix(),
        )


def validate_phase3_root(
    phase3_root: Path,
    *,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
    modules: Sequence[str] | None = None,
    validate_schemas: bool = True,
) -> ValidationReport:
    """Validate the complete or module-scoped Phase 3 runtime layer.

    Parameters
    ----------
    phase3_root:
        Extracted ``PHASE3_CURATED_INPUT_LAYER_FINAL_002`` directory.
    contract_path:
        Phase 4 source-contract YAML.
    modules:
        Module names to validate. ``None`` validates all seven modules.
    validate_schemas:
        When true, validate every selected CSV, JSON, and raster schema in
        addition to package identity and file hashes.
    """

    assert_schema_registry_integrity()
    contract = load_source_contract(contract_path)
    selected = tuple(modules) if modules else contract.module_names
    unknown = set(selected) - set(contract.module_names)
    if unknown:
        raise RuntimeInputError(
            f"Unknown module names: {sorted(unknown)}; allowed={contract.module_names}"
        )
    if len(selected) != len(set(selected)):
        raise RuntimeInputError("Duplicate module names were requested")

    root = Path(phase3_root).expanduser().resolve()
    report = ValidationReport(
        phase3_root=str(root),
        contract_path=str(contract.path),
        modules=selected,
    )
    report.add(
        check_id="CONTRACT-001",
        scope="contract",
        description="Source contract loaded and internally consistent",
        observed=contract.contract_sha256,
        expected="valid SHA-256-bound contract",
        passed=True,
    )
    report.add(
        check_id="CONTRACT-002",
        scope="contract",
        description="Requested modules are declared",
        observed=selected,
        expected=contract.module_names if modules is None else selected,
        passed=True,
    )

    root_exists = root.is_dir()
    report.add(
        check_id="PKG-000",
        scope="package",
        description="Phase 3 runtime root exists and is a directory",
        observed=root_exists,
        expected=True,
        passed=root_exists,
    )
    if not root_exists:
        return report

    _validate_package_anchors(report, root=root, contract=contract)
    _validate_runtime_hashes(
        report, root=root, contract=contract, modules=selected
    )
    if validate_schemas:
        _validate_structured_schemas(
            report, root=root, contract=contract, modules=selected
        )
    _validate_interpretation_guardrails(
        report, root=root, selected_modules=selected
    )
    return report


def validate_module_inputs(
    phase3_root: Path,
    module: str,
    *,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
    supplied_paths: Iterable[Path | str] | None = None,
    raise_on_error: bool = True,
) -> ValidationReport:
    """Validate one module and optionally prove its caller-supplied input set."""

    contract = load_source_contract(contract_path)
    if supplied_paths is not None:
        assert_module_input_paths(contract, phase3_root, module, supplied_paths)
    report = validate_phase3_root(
        phase3_root,
        contract_path=contract_path,
        modules=(module,),
        validate_schemas=True,
    )
    if raise_on_error:
        report.raise_for_errors()
    return report


def scan_phase4_sources_for_prohibited_tokens(
    source_root: Path,
    *,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
    excluded_paths: Iterable[Path | str] = (),
) -> dict[str, list[str]]:
    """Return prohibited-token hits in executable Phase 4 source files.

    The source contract itself is excluded because it deliberately records the
    prohibited tokens. Historical provenance files should not be placed under
    the executable source tree.
    """

    contract = load_source_contract(contract_path)
    root = Path(source_root).expanduser().resolve()
    excluded = {Path(p).expanduser().resolve() for p in excluded_paths}
    excluded.add(contract.path.resolve())
    hits: dict[str, list[str]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.resolve() in excluded:
            continue
        if path.suffix.lower() not in {".py", ".toml", ".yml", ".yaml", ".json"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        matched = [token for token in contract.prohibited_runtime_tokens if token in text]
        if matched:
            hits[path.relative_to(root).as_posix()] = matched
    return hits


# ---------------------------------------------------------------------------
# Report serialization and command-line interface
# ---------------------------------------------------------------------------


def write_validation_report(
    report: ValidationReport,
    *,
    json_path: Path | None = None,
    csv_path: Path | None = None,
) -> None:
    """Write deterministic JSON and/or CSV validation records."""

    if json_path is not None:
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if csv_path is not None:
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "check_id",
            "scope",
            "description",
            "severity",
            "status",
            "observed",
            "expected",
            "relative_path",
        ]
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for check in report.checks:
                writer.writerow(
                    {
                        **asdict(check),
                        "severity": check.severity.value,
                        "status": check.status.value,
                    }
                )


def _default_phase3_root() -> Path | None:
    env_value = os.environ.get("QSTABLE_PHASE3_ROOT")
    if env_value:
        return Path(env_value)
    candidate = PHASE4_ROOT.parent / PHASE3_PACKAGE_NAME
    return candidate if candidate.is_dir() else None


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the exclusive Phase 3 runtime inputs for Phase 4."
    )
    default_root = _default_phase3_root()
    parser.add_argument(
        "--phase3-root",
        type=Path,
        default=default_root,
        required=default_root is None,
        help=(
            "Extracted PHASE3_CURATED_INPUT_LAYER_FINAL_002 directory. "
            "Defaults to QSTABLE_PHASE3_ROOT or a sibling directory when found."
        ),
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=DEFAULT_CONTRACT_PATH,
        help="Path to phase4_source_contract.yml.",
    )
    parser.add_argument(
        "--module",
        action="append",
        dest="modules",
        help="Validate one module; repeat to validate several. Default: all modules.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional detailed JSON report path.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Optional detailed CSV report path.",
    )
    parser.add_argument(
        "--hashes-only",
        action="store_true",
        help="Skip table/JSON/raster schema checks and validate identity plus hashes only.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        report = validate_phase3_root(
            args.phase3_root,
            contract_path=args.contract,
            modules=args.modules,
            validate_schemas=not args.hashes_only,
        )
        write_validation_report(
            report, json_path=args.output_json, csv_path=args.output_csv
        )
    except (RuntimeInputError, FileNotFoundError, yaml.YAMLError) as exc:
        print(f"VALIDATION_ERROR: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(report.summary(), indent=2, ensure_ascii=False))
    if report.failed_checks:
        print("\nFailed checks:", file=sys.stderr)
        for check in report.failed_checks[:20]:
            print(
                f"  {check.check_id}: {check.description} "
                f"(observed={check.observed}; expected={check.expected})",
                file=sys.stderr,
            )
        return 1
    return 0


__all__ = [
    "DEFAULT_CONTRACT_PATH",
    "RuntimeFile",
    "RuntimeInputError",
    "SourceContract",
    "ValidationCheck",
    "ValidationReport",
    "ValidationSeverity",
    "ValidationStatus",
    "assert_allowed_output_name",
    "assert_module_input_paths",
    "load_source_contract",
    "resolve_runtime_path",
    "scan_phase4_sources_for_prohibited_tokens",
    "sha256_file",
    "validate_module_inputs",
    "validate_phase3_root",
    "write_validation_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
