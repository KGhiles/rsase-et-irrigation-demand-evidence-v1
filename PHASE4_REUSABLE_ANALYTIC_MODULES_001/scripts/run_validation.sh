#!/usr/bin/env bash
# Validate the frozen Phase 3 runtime layer against the Phase 4 source contract.
#
# This script is intentionally validation-only. It does not execute analytical
# modules, modify Phase 3, or replace published Phase 4 result files.

set -Eeuo pipefail
IFS=$'\n\t'

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
readonly DEFAULT_PHASE3_NAME="PHASE3_CURATED_INPUT_LAYER_FINAL_002"
readonly DEFAULT_CONTRACT="${REPO_ROOT}/config/phase4_source_contract.yml"
readonly DEFAULT_REPORT_DIR="${REPO_ROOT}/validation_reports"

PHASE3_ROOT=""
CONTRACT_PATH="${DEFAULT_CONTRACT}"
REPORT_DIR="${DEFAULT_REPORT_DIR}"
WRITE_REPORTS=true
HASHES_ONLY=false
PYTHON_BIN="${QSTABLE_PYTHON:-python}"
MODULES=()

usage() {
    cat <<'USAGE'
Usage:
  scripts/run_validation.sh [options]

Validate the frozen Phase 3 curated-input package against the Phase 4 source
contract. By default, the script validates all seven analytical modules and
writes deterministic JSON and CSV reports to ./validation_reports.

Options:
  --phase3-root PATH   Extracted PHASE3_CURATED_INPUT_LAYER_FINAL_002 directory.
                       Defaults to QSTABLE_PHASE3_ROOT, then the sibling package.
  --contract PATH      Source contract path.
                       Default: config/phase4_source_contract.yml
  --module NAME        Validate one module; repeat for several modules.
                       Accepted public names:
                         climate-context
                         physical-coherence
                         aerodynamic-sensitivity
                         crop-support
                         fixed-frame-dominance
                         known-cold-reference
                         interpretation-classes
  --hashes-only        Validate package identity and SHA-256 values only.
  --report-dir PATH    Directory for validation JSON and CSV reports.
                       Default: validation_reports
  --no-reports         Print the validation summary without writing reports.
  --python PATH        Python interpreter to use. Default: QSTABLE_PYTHON or python.
  -h, --help           Show this help and exit.

Examples:
  scripts/run_validation.sh \
    --phase3-root ../PHASE3_CURATED_INPUT_LAYER_FINAL_002

  scripts/run_validation.sh \
    --module crop-support \
    --module fixed-frame-dominance

  QSTABLE_PYTHON="$CONDA_PREFIX/bin/python" \
    scripts/run_validation.sh --hashes-only --no-reports

Exit codes:
  0  Validation passed.
  1  Validation completed with one or more failed checks.
  2  Invalid arguments, missing files, unsafe paths, or runtime failure.
USAGE
}

fail() {
    printf 'QSTABLE_VALIDATION_SCRIPT_ERROR: %s\n' "$*" >&2
    exit 2
}

canonical_existing_dir() {
    local value=$1
    [[ -d "${value}" ]] || fail "Directory not found: ${value}"
    (cd -- "${value}" && pwd -P)
}

canonical_existing_file() {
    local value=$1
    [[ -f "${value}" ]] || fail "File not found: ${value}"
    local parent base
    parent=$(cd -- "$(dirname -- "${value}")" && pwd -P)
    base=$(basename -- "${value}")
    printf '%s/%s\n' "${parent}" "${base}"
}

is_allowed_module() {
    case "$1" in
        climate-context|physical-coherence|aerodynamic-sensitivity|crop-support|\
        fixed-frame-dominance|known-cold-reference|interpretation-classes)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

while (($#)); do
    case "$1" in
        --phase3-root)
            (($# >= 2)) || fail "--phase3-root requires a path"
            PHASE3_ROOT=$2
            shift 2
            ;;
        --contract)
            (($# >= 2)) || fail "--contract requires a path"
            CONTRACT_PATH=$2
            shift 2
            ;;
        --module)
            (($# >= 2)) || fail "--module requires a name"
            is_allowed_module "$2" || fail "Unknown module: $2"
            MODULES+=("$2")
            shift 2
            ;;
        --hashes-only)
            HASHES_ONLY=true
            shift
            ;;
        --report-dir)
            (($# >= 2)) || fail "--report-dir requires a path"
            REPORT_DIR=$2
            shift 2
            ;;
        --no-reports)
            WRITE_REPORTS=false
            shift
            ;;
        --python)
            (($# >= 2)) || fail "--python requires an interpreter path or command"
            PYTHON_BIN=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            (($# == 0)) || fail "Unexpected positional arguments: $*"
            ;;
        -*)
            fail "Unknown option: $1"
            ;;
        *)
            fail "Unexpected positional argument: $1"
            ;;
    esac
done

if [[ -z "${PHASE3_ROOT}" ]]; then
    if [[ -n "${QSTABLE_PHASE3_ROOT:-}" ]]; then
        PHASE3_ROOT=${QSTABLE_PHASE3_ROOT}
    elif [[ -d "${REPO_ROOT}/../${DEFAULT_PHASE3_NAME}" ]]; then
        PHASE3_ROOT="${REPO_ROOT}/../${DEFAULT_PHASE3_NAME}"
    else
        fail "Phase 3 root was not provided and no sibling ${DEFAULT_PHASE3_NAME} exists"
    fi
fi

PHASE3_ROOT=$(canonical_existing_dir "${PHASE3_ROOT}")
CONTRACT_PATH=$(canonical_existing_file "${CONTRACT_PATH}")

[[ "$(basename -- "${PHASE3_ROOT}")" == "${DEFAULT_PHASE3_NAME}" ]] || \
    fail "Phase 3 root must be named ${DEFAULT_PHASE3_NAME}: ${PHASE3_ROOT}"

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || fail "Python interpreter not found: ${PYTHON_BIN}"
"${PYTHON_BIN}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' || \
    fail "Python 3.11 or newer is required: ${PYTHON_BIN}"

if ! "${PYTHON_BIN}" - "${CONTRACT_PATH}" <<'PY'
from pathlib import Path
import sys
import yaml

try:
    payload = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if isinstance(payload, dict) else 1)
PY
then
    fail "Source contract must be a readable YAML mapping: ${CONTRACT_PATH}"
fi

# Prevent validation reports from modifying immutable inputs or canonical
# source/result trees. Resolve the requested path before creating it so a bad
# path cannot cause even an empty directory to be written in a protected tree.
if [[ "${WRITE_REPORTS}" == true ]]; then
    if [[ "${REPORT_DIR}" != /* ]]; then
        REPORT_DIR="${REPO_ROOT}/${REPORT_DIR}"
    fi
    REPORT_DIR=$("${PYTHON_BIN}" -c         'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())'         "${REPORT_DIR}")
    for protected_root in         "${PHASE3_ROOT}"         "${REPO_ROOT}/src"         "${REPO_ROOT}/config"         "${REPO_ROOT}/tests"         "${REPO_ROOT}/scripts"         "${REPO_ROOT}/results"
    do
        case "${REPORT_DIR}/" in
            "${protected_root}/"*)
                fail "Report directory cannot be inside protected tree ${protected_root}: ${REPORT_DIR}"
                ;;
        esac
    done
    mkdir -p -- "${REPORT_DIR}"
    REPORT_DIR=$(canonical_existing_dir "${REPORT_DIR}")
fi

# Make the source package importable without requiring a prior editable install.
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1

COMMAND=(
    "${PYTHON_BIN}" -m qstable_awm.cli validate
    --phase3-root "${PHASE3_ROOT}"
    --contract "${CONTRACT_PATH}"
)

for module in "${MODULES[@]}"; do
    COMMAND+=(--module "${module}")
done

if [[ "${HASHES_ONLY}" == true ]]; then
    COMMAND+=(--hashes-only)
fi

TEMP_DIR=""
cleanup() {
    if [[ -n "${TEMP_DIR}" && -d "${TEMP_DIR}" ]]; then
        rm -rf -- "${TEMP_DIR}"
    fi
}
trap cleanup EXIT INT TERM

if [[ "${WRITE_REPORTS}" == true ]]; then
    TEMP_DIR=$(mktemp -d "${REPORT_DIR}/.qstable-validation.XXXXXX")
    COMMAND+=(
        --output-json "${TEMP_DIR}/phase4_runtime_validation.json"
        --output-csv "${TEMP_DIR}/phase4_runtime_validation.csv"
    )
fi

printf 'QSTABLE_VALIDATION_BEGIN\n'
printf 'REPOSITORY_ROOT=%s\n' "${REPO_ROOT}"
printf 'PHASE3_ROOT=%s\n' "${PHASE3_ROOT}"
printf 'CONTRACT=%s\n' "${CONTRACT_PATH}"
printf 'PYTHON=%s\n' "$(command -v "${PYTHON_BIN}")"
printf 'MODE=%s\n' "$([[ "${HASHES_ONLY}" == true ]] && printf 'HASHES_ONLY' || printf 'FULL_SCHEMA')"
if ((${#MODULES[@]})); then
    printf 'MODULES=%s\n' "$(IFS=,; printf '%s' "${MODULES[*]}")"
else
    printf 'MODULES=ALL\n'
fi

set +e
"${COMMAND[@]}"
status=$?
set -e

if ((status != 0)); then
    printf 'QSTABLE_VALIDATION_END status=FAIL exit_code=%d\n' "${status}" >&2
    exit "${status}"
fi

if [[ "${WRITE_REPORTS}" == true ]]; then
    mv -f -- \
        "${TEMP_DIR}/phase4_runtime_validation.json" \
        "${REPORT_DIR}/phase4_runtime_validation.json"
    mv -f -- \
        "${TEMP_DIR}/phase4_runtime_validation.csv" \
        "${REPORT_DIR}/phase4_runtime_validation.csv"
    printf 'VALIDATION_JSON=%s\n' "${REPORT_DIR}/phase4_runtime_validation.json"
    printf 'VALIDATION_CSV=%s\n' "${REPORT_DIR}/phase4_runtime_validation.csv"
fi

printf 'QSTABLE_VALIDATION_END status=PASS exit_code=0\n'
