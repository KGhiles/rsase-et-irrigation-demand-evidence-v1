"""Regression tests for the public-release ``.gitignore`` policy.

The ignore policy must remove local caches, environments, build products,
validation reports, temporary pipeline trees, separately distributed Phase 3
inputs, and scheduler logs without hiding any authoritative source, test,
configuration, documentation, script, or canonical result file.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest


PHASE4_ROOT = Path(__file__).resolve().parents[1]
GITIGNORE_PATH = PHASE4_ROOT / ".gitignore"
EXPECTED_GITIGNORE_SHA256 = (
    "d92599e224e8e6e6de1d258ada54fabf73933a8e15ae9dce5868a5ceaf40b930"
)

REQUIRED_PATTERNS = {
    "__pycache__/",
    "*.py[cod]",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    ".coverage",
    "htmlcov/",
    "build/",
    "dist/",
    "*.egg-info/",
    ".venv/",
    ".conda/",
    ".ipynb_checkpoints/",
    ".env",
    ".env.*",
    "!.env.example",
    "validation_reports/",
    ".qstable-validation.*/",
    ".phase4-pipeline-staging-*/",
    ".phase4-pipeline-backup-*/",
    "results_candidate/",
    "results_candidate_*/",
    "PHASE3_CURATED_INPUT_LAYER_FINAL_002/",
    "PHASE3_CURATED_INPUT_LAYER_FINAL_002.zip",
    "PHASE3_CURA.zip",
    "slurm-*.out",
    "slurm-*.err",
    "*.tmp",
    "*.bak",
}

PROHIBITED_BROAD_PATTERNS = {
    "results/",
    "results/**",
    "*.csv",
    "*.json",
    "*.py",
    "*.sh",
    "*.yml",
    "*.yaml",
    "*.md",
    "src/",
    "tests/",
    "scripts/",
    "config/",
    "README.md",
    "pyproject.toml",
    "environment.yml",
}

IGNORED_CASES = (
    "src/qstable_awm/__pycache__/schemas.cpython-311.pyc",
    ".pytest_cache/v/cache/nodeids",
    ".mypy_cache/3.11/cache.json",
    ".ruff_cache/0.9/cache",
    ".coverage",
    "htmlcov/index.html",
    "build/lib/qstable_awm/__init__.py",
    "dist/qstable_awm-1.0.0-py3-none-any.whl",
    "src/qstable_awm.egg-info/PKG-INFO",
    ".venv/bin/python",
    ".conda/conda-meta/history",
    "notebooks/.ipynb_checkpoints/a-checkpoint.ipynb",
    ".env",
    ".env.local",
    "validation_reports/phase4_validation.json",
    ".qstable-validation.123/report.csv",
    ".phase4-pipeline-staging-abc/output.csv",
    ".phase4-pipeline-backup-abc/output.csv",
    "results_candidate/pipeline/phase4_pipeline_validation.json",
    "results_candidate_trial/output.csv",
    "PHASE3_CURATED_INPUT_LAYER_FINAL_002/VERSION",
    "PHASE3_CURATED_INPUT_LAYER_FINAL_002.zip",
    "PHASE3_CURA.zip",
    "slurm-12345.out",
    "slurm-12345.err",
    "analysis.tmp",
    "analysis.bak",
)

TRACKED_CASES = (
    ".gitignore",
    ".env.example",
    "README.md",
    "pyproject.toml",
    "environment.yml",
    "config/phase4_source_contract.yml",
    "src/qstable_awm/__init__.py",
    "src/qstable_awm/pipeline.py",
    "tests/test_pipeline.py",
    "scripts/run_validation.sh",
    "scripts/run_pipeline.sh",
    "results/climate_context/figure2_climate_source.csv",
    "results/physical_coherence/physical_coherence_validation.json",
    "results/pipeline/phase4_pipeline_output_manifest.csv",
    "notebooks/provenance_example.ipynb",
    "release_manifest.csv",
    "SHA256SUMS",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _active_patterns() -> list[str]:
    return [
        line.strip()
        for line in GITIGNORE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _materialize(root: Path, relative_path: str) -> None:
    path = root / relative_path
    if relative_path.endswith("/"):
        path.mkdir(parents=True, exist_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("test\n", encoding="utf-8")


def _git_check_ignore(root: Path, relative_path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", relative_path],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode not in {0, 1}:
        raise AssertionError(
            f"git check-ignore failed for {relative_path!r}: {result.stderr}"
        )
    return result.returncode == 0


@pytest.fixture()
def git_policy_repo(tmp_path: Path) -> Path:
    if shutil.which("git") is None:
        pytest.skip("git is required for .gitignore behavior tests")
    subprocess.run(
        ["git", "init", "--quiet"],
        cwd=tmp_path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    shutil.copy2(GITIGNORE_PATH, tmp_path / ".gitignore")
    for relative_path in set(IGNORED_CASES) | set(TRACKED_CASES):
        if relative_path != ".gitignore":
            _materialize(tmp_path, relative_path)
    return tmp_path


def test_gitignore_file_is_frozen_and_nonempty() -> None:
    assert GITIGNORE_PATH.is_file()
    assert GITIGNORE_PATH.stat().st_size > 0
    assert _sha256(GITIGNORE_PATH) == EXPECTED_GITIGNORE_SHA256


def test_gitignore_is_normalized_text() -> None:
    raw = GITIGNORE_PATH.read_bytes()
    text = raw.decode("utf-8")
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in raw
    assert text.endswith("\n")
    assert all(not line.endswith((" ", "\t")) for line in text.splitlines())


def test_active_patterns_are_unique_and_expected_in_number() -> None:
    patterns = _active_patterns()
    assert len(patterns) == 54
    assert len(patterns) == len(set(patterns))


def test_required_release_patterns_are_present() -> None:
    patterns = set(_active_patterns())
    assert REQUIRED_PATTERNS <= patterns


def test_no_broad_pattern_hides_authoritative_file_types() -> None:
    patterns = set(_active_patterns())
    assert patterns.isdisjoint(PROHIBITED_BROAD_PATTERNS)


def test_canonical_results_tracking_is_explicitly_documented() -> None:
    text = GITIGNORE_PATH.read_text(encoding="utf-8")
    assert "Canonical release outputs under results/ are intentionally tracked." in text


@pytest.mark.parametrize("relative_path", IGNORED_CASES)
def test_expected_local_artifacts_are_ignored(
    git_policy_repo: Path, relative_path: str
) -> None:
    assert _git_check_ignore(git_policy_repo, relative_path)


@pytest.mark.parametrize("relative_path", TRACKED_CASES)
def test_authoritative_and_release_files_remain_trackable(
    git_policy_repo: Path, relative_path: str
) -> None:
    assert not _git_check_ignore(git_policy_repo, relative_path)


def test_environment_exception_overrides_environment_wildcard(
    git_policy_repo: Path,
) -> None:
    assert _git_check_ignore(git_policy_repo, ".env")
    assert _git_check_ignore(git_policy_repo, ".env.production")
    assert not _git_check_ignore(git_policy_repo, ".env.example")


def test_notebooks_are_trackable_but_checkpoint_state_is_ignored(
    git_policy_repo: Path,
) -> None:
    assert not _git_check_ignore(git_policy_repo, "notebooks/provenance_example.ipynb")
    assert _git_check_ignore(
        git_policy_repo,
        "notebooks/.ipynb_checkpoints/a-checkpoint.ipynb",
    )


def test_candidate_results_are_ignored_but_canonical_results_are_tracked(
    git_policy_repo: Path,
) -> None:
    assert _git_check_ignore(
        git_policy_repo,
        "results_candidate/pipeline/phase4_pipeline_validation.json",
    )
    assert not _git_check_ignore(
        git_policy_repo,
        "results/pipeline/phase4_pipeline_validation.json",
    )


def test_all_existing_authoritative_files_remain_trackable(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git is required for .gitignore behavior tests")
    subprocess.run(
        ["git", "init", "--quiet"],
        cwd=tmp_path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    shutil.copy2(GITIGNORE_PATH, tmp_path / ".gitignore")

    authoritative_roots = (
        PHASE4_ROOT / "config",
        PHASE4_ROOT / "src",
        PHASE4_ROOT / "tests",
        PHASE4_ROOT / "scripts",
        PHASE4_ROOT / "results",
    )
    relative_paths: list[str] = [
        ".gitignore",
        "README.md",
        "pyproject.toml",
        "environment.yml",
    ]
    for authority_root in authoritative_roots:
        relative_paths.extend(
            str(path.relative_to(PHASE4_ROOT)).replace("\\", "/")
            for path in authority_root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        )

    for relative_path in sorted(set(relative_paths)):
        if relative_path != ".gitignore":
            _materialize(tmp_path, relative_path)
        assert not _git_check_ignore(tmp_path, relative_path), relative_path


def test_phase3_package_is_ignored_only_as_separately_distributed_data(
    git_policy_repo: Path,
) -> None:
    assert _git_check_ignore(
        git_policy_repo,
        "PHASE3_CURATED_INPUT_LAYER_FINAL_002/VERSION",
    )
    assert not _git_check_ignore(
        git_policy_repo,
        "docs/PHASE3_CURATED_INPUT_LAYER_FINAL_002_provenance.md",
    )


def test_release_manifests_and_checksums_remain_trackable(
    git_policy_repo: Path,
) -> None:
    assert not _git_check_ignore(git_policy_repo, "release_manifest.csv")
    assert not _git_check_ignore(git_policy_repo, "SHA256SUMS")
