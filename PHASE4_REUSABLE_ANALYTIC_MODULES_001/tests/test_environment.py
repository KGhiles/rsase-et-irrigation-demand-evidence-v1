"""Conda-environment and offline installation tests for Phase 4.

These tests validate release and execution metadata only.  They do not execute
historical notebooks, read Phase 3 scientific tables, or recalculate any
manuscript value.  The accepted scientific runtime remains the repository
checkout plus the frozen sibling ``PHASE3_CURATED_INPUT_LAYER_FINAL_002``
package and the Phase 4 source contract.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet


PHASE4_ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT_PATH = PHASE4_ROOT / "environment.yml"
PYPROJECT_PATH = PHASE4_ROOT / "pyproject.toml"
README_PATH = PHASE4_ROOT / "README.md"
PACKAGE_ROOT = PHASE4_ROOT / "src" / "qstable_awm"

EXPECTED_ENVIRONMENT_SHA256 = (
    "9f9beee84d2e48ea94a026e48c332227169d86c27758d113c2d89b7f578bfba9"
)
EXPECTED_PYPROJECT_SHA256 = (
    "1c6aa0d377729249a03448bb67d65869bdab6f4c4ba9d3bfa82768d63f17519f"
)
EXPECTED_ENVIRONMENT_NAME = "qstable-awm"
EXPECTED_CHANNELS = ["conda-forge", "nodefaults"]
EXPECTED_DEPENDENCIES = [
    "python=3.11",
    "pip>=24",
    "setuptools>=68",
    "numpy>=1.24,<3",
    "pandas>=2.0,<3",
    "pyyaml>=6.0,<7",
    "rasterio>=1.3,<2",
    "pytest>=8,<10",
]
EXPECTED_RUNTIME_SPECS = {
    "numpy": ">=1.24,<3",
    "pandas": ">=2.0,<3",
    "pyyaml": ">=6.0,<7",
    "rasterio": ">=1.3,<2",
}
EXPECTED_TEST_SPECS = {"pytest": ">=8,<10"}
EXPECTED_BUILD_SPECS = {"setuptools": ">=68"}
EXPECTED_ENVIRONMENT_ONLY_SPECS = {"pip": ">=24"}
EXPECTED_EDITABLE_COMMAND = (
    "python -m pip install --no-build-isolation --no-deps --editable ."
)
PROHIBITED_TEXT_TOKENS = (
    "/scratch/",
    ".ipynb",
    "V46D",
    "V48",
    "V49",
    "AWM_figures",
    "http://",
    "https://",
    "git+",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _split_conda_spec(spec: str) -> tuple[str, str]:
    """Return normalized package name and version constraint.

    The environment intentionally uses simple conda-compatible package specs,
    not channel-qualified or URL dependencies.  Python is pinned with one
    equals sign; all other packages use PEP 440-compatible comparator syntax.
    """

    match = re.fullmatch(r"([A-Za-z0-9_.-]+)(.*)", spec.strip())
    if match is None:
        raise AssertionError(f"Unparseable dependency: {spec!r}")
    name, constraint = match.groups()
    return _normalized_name(name), constraint


def _dependency_map(environment: dict[str, Any]) -> dict[str, str]:
    dependencies = environment["dependencies"]
    assert isinstance(dependencies, list)
    assert all(isinstance(item, str) for item in dependencies)
    parsed = [_split_conda_spec(item) for item in dependencies]
    names = [name for name, _ in parsed]
    assert len(names) == len(set(names)), f"Duplicate dependencies: {names}"
    return dict(parsed)


def _project_requirement_map(requirements: list[str]) -> dict[str, str]:
    parsed = [Requirement(item) for item in requirements]
    return {
        _normalized_name(requirement.name): str(requirement.specifier)
        for requirement in parsed
    }


def _copy_install_project(destination: Path) -> Path:
    project = destination / "project"
    (project / "src").mkdir(parents=True)
    shutil.copy2(PYPROJECT_PATH, project / "pyproject.toml")
    shutil.copy2(README_PATH, project / "README.md")
    shutil.copytree(
        PACKAGE_ROOT,
        project / "src" / "qstable_awm",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    return project


@pytest.fixture(scope="session")
def environment_text() -> str:
    return ENVIRONMENT_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def environment() -> dict[str, Any]:
    payload = yaml.safe_load(ENVIRONMENT_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


@pytest.fixture(scope="session")
def pyproject() -> dict[str, Any]:
    with PYPROJECT_PATH.open("rb") as stream:
        return tomllib.load(stream)


@pytest.fixture(scope="session")
def dependency_map(environment: dict[str, Any]) -> dict[str, str]:
    return _dependency_map(environment)


@pytest.fixture(scope="session")
def offline_editable_dry_run(
    tmp_path_factory: pytest.TempPathFactory,
) -> subprocess.CompletedProcess[str]:
    root = tmp_path_factory.mktemp("environment_offline_editable")
    project = _copy_install_project(root)
    environment = os.environ.copy()
    environment.update(
        {
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INDEX": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--dry-run",
            "--no-build-isolation",
            "--no-deps",
            "--editable",
            ".",
        ],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


# ---------------------------------------------------------------------------
# Frozen environment authority and YAML structure
# ---------------------------------------------------------------------------


def test_environment_hash_is_frozen() -> None:
    assert _sha256(ENVIRONMENT_PATH) == EXPECTED_ENVIRONMENT_SHA256


def test_pyproject_hash_used_for_parity_is_frozen() -> None:
    assert _sha256(PYPROJECT_PATH) == EXPECTED_PYPROJECT_SHA256


def test_environment_yaml_is_a_mapping(environment: dict[str, Any]) -> None:
    assert set(environment) == {"name", "channels", "dependencies"}


def test_environment_name_is_stable(environment: dict[str, Any]) -> None:
    assert environment["name"] == EXPECTED_ENVIRONMENT_NAME


def test_channel_order_is_explicit(environment: dict[str, Any]) -> None:
    assert environment["channels"] == EXPECTED_CHANNELS


def test_nodefaults_is_last_channel(environment: dict[str, Any]) -> None:
    assert environment["channels"][-1] == "nodefaults"


def test_defaults_channel_is_not_enabled(environment: dict[str, Any]) -> None:
    assert "defaults" not in environment["channels"]
    assert "anaconda" not in environment["channels"]


def test_dependency_list_is_exact(environment: dict[str, Any]) -> None:
    assert environment["dependencies"] == EXPECTED_DEPENDENCIES


def test_dependency_names_are_unique(dependency_map: dict[str, str]) -> None:
    assert tuple(dependency_map) == (
        "python",
        "pip",
        "setuptools",
        "numpy",
        "pandas",
        "pyyaml",
        "rasterio",
        "pytest",
    )


def test_dependencies_are_plain_strings(environment: dict[str, Any]) -> None:
    assert all(isinstance(item, str) for item in environment["dependencies"])


def test_no_nested_pip_install_section(environment: dict[str, Any]) -> None:
    assert not any(isinstance(item, dict) for item in environment["dependencies"])


# ---------------------------------------------------------------------------
# Dependency parity with pyproject.toml
# ---------------------------------------------------------------------------


def test_python_pin_matches_supported_minimum(
    dependency_map: dict[str, str], pyproject: dict[str, Any]
) -> None:
    assert dependency_map["python"] == "=3.11"
    assert pyproject["project"]["requires-python"] == ">=3.11"
    assert "Programming Language :: Python :: 3.11" in pyproject["project"][
        "classifiers"
    ]


@pytest.mark.parametrize("name", sorted(EXPECTED_RUNTIME_SPECS))
def test_runtime_dependency_spec_matches_pyproject(
    name: str,
    dependency_map: dict[str, str],
    pyproject: dict[str, Any],
) -> None:
    project_specs = _project_requirement_map(pyproject["project"]["dependencies"])
    assert dependency_map[name] == EXPECTED_RUNTIME_SPECS[name]
    assert SpecifierSet(dependency_map[name]) == SpecifierSet(project_specs[name])


@pytest.mark.parametrize("name", sorted(EXPECTED_TEST_SPECS))
def test_test_dependency_spec_matches_pyproject(
    name: str,
    dependency_map: dict[str, str],
    pyproject: dict[str, Any],
) -> None:
    test_specs = _project_requirement_map(
        pyproject["project"]["optional-dependencies"]["test"]
    )
    assert dependency_map[name] == EXPECTED_TEST_SPECS[name]
    assert SpecifierSet(dependency_map[name]) == SpecifierSet(test_specs[name])


@pytest.mark.parametrize("name", sorted(EXPECTED_BUILD_SPECS))
def test_build_dependency_spec_matches_pyproject(
    name: str,
    dependency_map: dict[str, str],
    pyproject: dict[str, Any],
) -> None:
    build_specs = _project_requirement_map(pyproject["build-system"]["requires"])
    assert dependency_map[name] == EXPECTED_BUILD_SPECS[name]
    assert SpecifierSet(dependency_map[name]) == SpecifierSet(build_specs[name])


@pytest.mark.parametrize("name", sorted(EXPECTED_ENVIRONMENT_ONLY_SPECS))
def test_environment_only_tool_is_explicit(
    name: str, dependency_map: dict[str, str]
) -> None:
    assert dependency_map[name] == EXPECTED_ENVIRONMENT_ONLY_SPECS[name]


def test_no_runtime_dependency_is_missing(
    dependency_map: dict[str, str], pyproject: dict[str, Any]
) -> None:
    project_names = set(
        _project_requirement_map(pyproject["project"]["dependencies"])
    )
    assert project_names <= set(dependency_map)


def test_no_unexplained_environment_package(
    dependency_map: dict[str, str], pyproject: dict[str, Any]
) -> None:
    runtime = set(_project_requirement_map(pyproject["project"]["dependencies"]))
    tests = set(
        _project_requirement_map(
            pyproject["project"]["optional-dependencies"]["test"]
        )
    )
    build = set(_project_requirement_map(pyproject["build-system"]["requires"]))
    explained = {"python", "pip"} | runtime | tests | build
    assert set(dependency_map) == explained


# ---------------------------------------------------------------------------
# Offline/HPC execution policy
# ---------------------------------------------------------------------------


def test_creation_commands_are_documented(environment_text: str) -> None:
    assert "conda env create --file environment.yml" in environment_text
    assert "conda activate qstable-awm" in environment_text


def test_offline_editable_install_command_is_documented(environment_text: str) -> None:
    assert EXPECTED_EDITABLE_COMMAND in environment_text


@pytest.mark.parametrize(
    "required_flag",
    ["--no-build-isolation", "--no-deps", "--editable"],
)
def test_editable_install_includes_offline_safety_flags(
    required_flag: str, environment_text: str
) -> None:
    command_line = next(
        line.strip().lstrip("#").strip()
        for line in environment_text.splitlines()
        if "python -m pip install" in line
    )
    assert required_flag in command_line.split()


def test_environment_does_not_embed_local_editable_path_as_dependency(
    environment: dict[str, Any]
) -> None:
    assert "." not in environment["dependencies"]
    assert "-e ." not in environment["dependencies"]
    assert "qstable-awm" not in environment["dependencies"]


@pytest.mark.parametrize("token", PROHIBITED_TEXT_TOKENS)
def test_environment_avoids_network_historical_and_internal_tokens(
    token: str, environment_text: str
) -> None:
    assert token not in environment_text


def test_no_channel_qualified_or_url_dependency(environment: dict[str, Any]) -> None:
    for item in environment["dependencies"]:
        assert "::" not in item
        assert "://" not in item
        assert "@" not in item


def test_offline_editable_install_dry_run_succeeds(
    offline_editable_dry_run: subprocess.CompletedProcess[str],
) -> None:
    assert offline_editable_dry_run.returncode == 0, (
        offline_editable_dry_run.stdout + "\n" + offline_editable_dry_run.stderr
    )


def test_offline_editable_install_dry_run_resolves_local_distribution(
    offline_editable_dry_run: subprocess.CompletedProcess[str],
) -> None:
    output = offline_editable_dry_run.stdout + offline_editable_dry_run.stderr
    assert "qstable-awm" in output.lower()
    assert "1.0.0" in output


def test_offline_editable_install_dry_run_does_not_resolve_dependencies(
    offline_editable_dry_run: subprocess.CompletedProcess[str],
) -> None:
    output = (offline_editable_dry_run.stdout + offline_editable_dry_run.stderr).lower()
    for dependency in EXPECTED_RUNTIME_SPECS:
        assert f"collecting {dependency}" not in output
        assert f"downloading {dependency}" not in output


# ---------------------------------------------------------------------------
# Repository cleanliness and scientific boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "relative_path",
    ["build", "dist", "src/qstable_awm.egg-info"],
)
def test_canonical_repository_has_no_build_artifact(relative_path: str) -> None:
    assert not (PHASE4_ROOT / relative_path).exists()


def test_environment_file_is_not_an_analytical_runtime_input() -> None:
    contract_text = (
        PHASE4_ROOT / "config" / "phase4_source_contract.yml"
    ).read_text(encoding="utf-8")
    assert "environment.yml" not in contract_text


def test_environment_file_contains_no_scientific_result_literals(
    environment_text: str,
) -> None:
    prohibited_result_terms = (
        "Qgross",
        "ETa",
        "Class I candidate",
        "crop_support_fraction",
        "standard_deviation",
        "2011_dominance",
    )
    assert all(term not in environment_text for term in prohibited_result_terms)


def test_environment_file_does_not_reference_phase3_data_paths(
    environment_text: str,
) -> None:
    assert "PHASE3_CURATED_INPUT_LAYER_FINAL_002/data/" not in environment_text
    assert "data/derived/" not in environment_text
