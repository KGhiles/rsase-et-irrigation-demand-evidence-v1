#!/usr/bin/env bash
# Validate and execute the Phase 4 analytical pipeline.
#
# The Python pipeline performs complete Phase 3 preflight validation, executes
# selected modules in an isolated staging tree, validates every staged output,
# and publishes results atomically. This wrapper provides stable local/ACES
# argument handling and prevents unsafe output roots before Python is invoked.

set -Eeuo pipefail
IFS=$'\n\t'

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
readonly DEFAULT_PHASE3_NAME="PHASE3_CURATED_INPUT_LAYER_FINAL_002"
readonly DEFAULT_CONTRACT="${REPO_ROOT}/config/phase4_source_contract.yml"
readonly DEFAULT_RESULTS_ROOT="${REPO_ROOT}/results"

PHASE3_ROOT=""
CONTRACT_PATH="${DEFAULT_CONTRACT}"
RESULTS_ROOT="${DEFAULT_RESULTS_ROOT}"
PYTHON_BIN="${QSTABLE_PYTHON:-python}"
DRY_RUN=false
OVERWRITE=false
PUBLIC_MODULES=()
INTERNAL_MODULES=()

usage() {
    cat <<'USAGE'
Usage:
  scripts/run_pipeline.sh [options]

Validate the frozen Phase 3 curated-input package and run the Phase 4
analytical pipeline. Modules execute in a staging tree and are published only
after all selected outputs pass validation.

Options:
  --phase3-root PATH   Extracted PHASE3_CURATED_INPUT_LAYER_FINAL_002 directory.
                       Defaults to QSTABLE_PHASE3_ROOT, then the sibling package.
  --contract PATH      Source contract path.
                       Default: config/phase4_source_contract.yml
  --results-root PATH  Destination root for module and pipeline outputs.
                       Default: results
  --module NAME        Run one module; repeat for several modules.
                       Accepted public names:
                         climate-context
                         physical-coherence
                         aerodynamic-sensitivity
                         crop-support
                         fixed-frame-dominance
                         known-cold-reference
                         interpretation-classes
                       Omit --module to run all seven modules.
  --dry-run            Validate all Phase 3 inputs and print the execution plan
                       without writing output files.
  --overwrite          Atomically replace selected existing module outputs and
                       pipeline metadata. Required when targets already exist.
  --python PATH        Python interpreter to use. Default: QSTABLE_PYTHON or python.
  -h, --help           Show this help and exit.

Examples:
  scripts/run_pipeline.sh \
    --phase3-root ../PHASE3_CURATED_INPUT_LAYER_FINAL_002 \
    --dry-run

  scripts/run_pipeline.sh \
    --module crop-support \
    --results-root results_candidate

  QSTABLE_PYTHON="$CONDA_PREFIX/bin/python" \
    scripts/run_pipeline.sh --overwrite

Exit codes:
  0  Dry run or pipeline execution passed.
  2  Invalid arguments, unsafe paths, overwrite refusal, validation failure,
     module failure, or another expected runtime error.
USAGE
}

fail() {
    printf 'QSTABLE_PIPELINE_SCRIPT_ERROR: %s\n' "$*" >&2
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

module_to_internal() {
    case "$1" in
        climate-context) printf 'climate_context\n' ;;
        physical-coherence) printf 'physical_coherence\n' ;;
        aerodynamic-sensitivity) printf 'aerodynamic_sensitivity\n' ;;
        crop-support) printf 'crop_support\n' ;;
        fixed-frame-dominance) printf 'fixed_frame_dominance\n' ;;
        known-cold-reference) printf 'known_cold_reference\n' ;;
        interpretation-classes) printf 'interpretation_classes\n' ;;
        *) return 1 ;;
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
        --results-root)
            (($# >= 2)) || fail "--results-root requires a path"
            RESULTS_ROOT=$2
            shift 2
            ;;
        --module)
            (($# >= 2)) || fail "--module requires a name"
            internal=$(module_to_internal "$2") || fail "Unknown module: $2"
            PUBLIC_MODULES+=("$2")
            INTERNAL_MODULES+=("${internal}")
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --overwrite)
            OVERWRITE=true
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

"${PYTHON_BIN}" -c \
    'from pathlib import Path; import sys, yaml; value=yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")); raise SystemExit(0 if isinstance(value, dict) else 1)' \
    "${CONTRACT_PATH}" >/dev/null 2>&1 || \
    fail "Source contract must be a readable YAML mapping: ${CONTRACT_PATH}"

# Reject duplicate module requests before invoking Python. The Python pipeline
# also enforces this, but an early shell error gives a clearer operational log.
if ((${#INTERNAL_MODULES[@]})); then
    declare -A seen_modules=()
    for module in "${INTERNAL_MODULES[@]}"; do
        [[ -z "${seen_modules[${module}]:-}" ]] || fail "Duplicate module: ${module//_/-}"
        seen_modules[${module}]=1
    done
fi

# Resolve the output path without creating it. Relative output paths are rooted
# at the repository so execution is independent of the caller's working dir.
if [[ "${RESULTS_ROOT}" != /* ]]; then
    RESULTS_ROOT="${REPO_ROOT}/${RESULTS_ROOT}"
fi
RESULTS_ROOT=$("${PYTHON_BIN}" -c \
    'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' \
    "${RESULTS_ROOT}")

# Protect immutable inputs and canonical source/control trees. Output roots may
# be elsewhere under the repository (for example results_candidate/) or outside
# it, but may not equal the repository root itself.
[[ "${RESULTS_ROOT}" != "${REPO_ROOT}" ]] || \
    fail "Results root cannot equal the repository root: ${RESULTS_ROOT}"
for protected_root in \
    "${PHASE3_ROOT}" \
    "${REPO_ROOT}/src" \
    "${REPO_ROOT}/config" \
    "${REPO_ROOT}/tests" \
    "${REPO_ROOT}/scripts"
do
    if [[ "${RESULTS_ROOT}" == "${protected_root}" ]]; then
        fail "Results root cannot equal protected tree ${protected_root}: ${RESULTS_ROOT}"
    fi
    case "${RESULTS_ROOT}/" in
        "${protected_root}/"*)
            fail "Results root cannot be inside protected tree ${protected_root}: ${RESULTS_ROOT}"
            ;;
    esac
done

# Make the package importable without requiring an editable install and avoid
# writing __pycache__ into the repository during release checks.
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1

COMMAND=(
    "${PYTHON_BIN}" -m qstable_awm.pipeline
    --phase3-root "${PHASE3_ROOT}"
    --contract "${CONTRACT_PATH}"
    --results-root "${RESULTS_ROOT}"
)

for module in "${INTERNAL_MODULES[@]}"; do
    COMMAND+=(--module "${module}")
done

if [[ "${DRY_RUN}" == true ]]; then
    COMMAND+=(--dry-run)
fi
if [[ "${OVERWRITE}" == true ]]; then
    COMMAND+=(--overwrite)
fi

printf 'QSTABLE_PIPELINE_BEGIN\n'
printf 'REPOSITORY_ROOT=%s\n' "${REPO_ROOT}"
printf 'PHASE3_ROOT=%s\n' "${PHASE3_ROOT}"
printf 'CONTRACT=%s\n' "${CONTRACT_PATH}"
printf 'RESULTS_ROOT=%s\n' "${RESULTS_ROOT}"
printf 'PYTHON=%s\n' "$(command -v "${PYTHON_BIN}")"
printf 'MODE=%s\n' "$([[ "${DRY_RUN}" == true ]] && printf 'DRY_RUN' || printf 'EXECUTE')"
printf 'OVERWRITE=%s\n' "${OVERWRITE}"
if ((${#PUBLIC_MODULES[@]})); then
    printf 'MODULES=%s\n' "$(IFS=,; printf '%s' "${PUBLIC_MODULES[*]}")"
else
    printf 'MODULES=ALL\n'
fi

set +e
"${COMMAND[@]}"
status=$?
set -e

if ((status != 0)); then
    printf 'QSTABLE_PIPELINE_END status=FAIL exit_code=%d\n' "${status}" >&2
    exit "${status}"
fi

printf 'QSTABLE_PIPELINE_END status=PASS exit_code=0\n'
