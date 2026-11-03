"""Packaging and installation tests for the Phase 4 analytical repository.

These tests validate only release metadata and installation behavior.  They do
not execute historical notebooks, recalculate manuscript statistics, or treat
an installed wheel as a self-contained scientific-data package.  The accepted
scientific runtime remains the repository checkout plus the frozen sibling
``PHASE3_CURATED_INPUT_LAYER_FINAL_002`` package and source contract.
"""

from __future__ import annotations

import ast
import base64
import configparser
import csv
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tomllib
import zipfile
from email.parser import Parser
from pathlib import Path
from typing import Any

import pytest

import qstable_awm
from qstable_awm import cli


PHASE4_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = PHASE4_ROOT / "pyproject.toml"
PACKAGE_ROOT = PHASE4_ROOT / "src" / "qstable_awm"

EXPECTED_PYPROJECT_SHA256 = (
    "1c6aa0d377729249a03448bb67d65869bdab6f4c4ba9d3bfa82768d63f17519f"
)
EXPECTED_INIT_SHA256 = (
    "c6d18f190a41c5356527ed81d035e206a145ca7cc48163e8543b857f32a4db42"
)
EXPECTED_CLI_SHA256 = (
    "c36cc75fc87dbbb00771544cbb2928fb1384ee871971969263b9b86b5a05cb6d"
)
EXPECTED_DESCRIPTION = (
    "Reproducible Phase 4 analytical modules for the Q-Stable Agricultural "
    "Water Management manuscript"
)
EXPECTED_RUNTIME_DEPENDENCIES = [
    "numpy>=1.24,<3",
    "pandas>=2.0,<3",
    "PyYAML>=6.0,<7",
    "rasterio>=1.3,<2",
]
EXPECTED_METADATA_DEPENDENCIES = [
    "numpy<3,>=1.24",
    "pandas<3,>=2.0",
    "PyYAML<7,>=6.0",
    "rasterio<2,>=1.3",
]
EXPECTED_CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering",
]
EXPECTED_PACKAGE_MODULES = (
    "__init__.py",
    "aerodynamic_sensitivity.py",
    "cli.py",
    "climate_context.py",
    "crop_support.py",
    "fixed_frame_dominance.py",
    "interpretation_classes.py",
    "known_cold_reference.py",
    "physical_coherence.py",
    "pipeline.py",
    "schemas.py",
    "validation.py",
)
EXPECTED_WHEEL_PACKAGE_MEMBERS = tuple(
    f"qstable_awm/{name}" for name in EXPECTED_PACKAGE_MODULES
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INDEX": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return environment


def _copy_packaging_project(destination: Path) -> Path:
    project = destination / "project"
    (project / "src").mkdir(parents=True)
    shutil.copy2(PYPROJECT_PATH, project / "pyproject.toml")
    shutil.copytree(
        PACKAGE_ROOT,
        project / "src" / "qstable_awm",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    return project


def _build_wheel(project: Path, wheel_dir: Path) -> Path:
    wheel_dir.mkdir(parents=True)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_dir),
            str(project),
        ],
        cwd=project,
        env=_subprocess_environment(),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    wheels = sorted(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1
    return wheels[0]


@pytest.fixture(scope="session")
def pyproject() -> dict[str, Any]:
    with PYPROJECT_PATH.open("rb") as stream:
        return tomllib.load(stream)


@pytest.fixture(scope="session")
def wheel_artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("packaging_wheel")
    project = _copy_packaging_project(root)
    return _build_wheel(project, root / "dist")


@pytest.fixture(scope="session")
def second_wheel_artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("packaging_wheel_second")
    project = _copy_packaging_project(root)
    return _build_wheel(project, root / "dist")


@pytest.fixture(scope="session")
def wheel_files(wheel_artifact: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(wheel_artifact) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


@pytest.fixture(scope="session")
def dist_info_prefix(wheel_files: dict[str, bytes]) -> str:
    prefixes = {
        name.split("/", 1)[0]
        for name in wheel_files
        if ".dist-info/" in name
    }
    assert len(prefixes) == 1
    return prefixes.pop()


@pytest.fixture(scope="session")
def editable_install(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("packaging_editable")
    project = _copy_packaging_project(root)
    prefix = root / "prefix"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "--editable",
            str(project),
            "--prefix",
            str(prefix),
        ],
        cwd=project,
        env=_subprocess_environment(),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr

    site_packages = prefix / sysconfig.get_path(
        "purelib",
        vars={
            "base": str(prefix),
            "platbase": str(prefix),
        },
    ).lstrip("/")
    # sysconfig may retain the interpreter prefix when custom variables are
    # insufficient on a platform.  Locate the installed metadata robustly.
    if not site_packages.is_dir():
        candidates = list(prefix.glob("lib/python*/site-packages"))
        assert len(candidates) == 1
        site_packages = candidates[0]

    script_name = "qstable-awm.exe" if os.name == "nt" else "qstable-awm"
    script = prefix / ("Scripts" if os.name == "nt" else "bin") / script_name
    assert site_packages.is_dir()
    assert script.is_file()
    return {
        "root": root,
        "project": project,
        "prefix": prefix,
        "site_packages": site_packages,
        "script": script,
    }


# ---------------------------------------------------------------------------
# pyproject.toml authority
# ---------------------------------------------------------------------------


def test_pyproject_hash_is_frozen() -> None:
    assert _sha256(PYPROJECT_PATH) == EXPECTED_PYPROJECT_SHA256


def test_authoritative_package_metadata_hashes_are_frozen() -> None:
    assert _sha256(PACKAGE_ROOT / "__init__.py") == EXPECTED_INIT_SHA256
    assert _sha256(PACKAGE_ROOT / "cli.py") == EXPECTED_CLI_SHA256


def test_pyproject_has_only_expected_top_level_sections(pyproject) -> None:
    assert list(pyproject) == [
        "build-system",
        "project",
        "tool",
    ]


def test_build_backend_is_setuptools_pep517(pyproject) -> None:
    assert pyproject["build-system"] == {
        "requires": ["setuptools>=68"],
        "build-backend": "setuptools.build_meta",
    }


def test_project_identity_and_dynamic_version(pyproject) -> None:
    project = pyproject["project"]
    assert project["name"] == qstable_awm.PACKAGE_NAME == "qstable-awm"
    assert project["dynamic"] == ["version"]
    assert "version" not in project
    assert pyproject["tool"]["setuptools"]["dynamic"]["version"] == {
        "attr": "qstable_awm.__version__"
    }


def test_project_description_and_python_floor(pyproject) -> None:
    project = pyproject["project"]
    assert project["description"] == EXPECTED_DESCRIPTION
    assert project["requires-python"] == ">=3.11"


def test_runtime_dependencies_are_exact(pyproject) -> None:
    assert pyproject["project"]["dependencies"] == EXPECTED_RUNTIME_DEPENDENCIES


def test_test_extra_is_exact(pyproject) -> None:
    assert pyproject["project"]["optional-dependencies"] == {
        "test": ["pytest>=8,<10"]
    }


def test_console_entry_point_is_exact(pyproject) -> None:
    assert pyproject["project"]["scripts"] == {
        "qstable-awm": "qstable_awm.cli:main"
    }


def test_package_discovery_is_restricted_to_src_qstable(pyproject) -> None:
    setuptools = pyproject["tool"]["setuptools"]
    assert setuptools["include-package-data"] is False
    assert setuptools["packages"]["find"] == {
        "where": ["src"],
        "include": ["qstable_awm*"],
        "namespaces": False,
    }


def test_pytest_configuration_is_strict_and_repository_scoped(pyproject) -> None:
    options = pyproject["tool"]["pytest"]["ini_options"]
    assert options == {
        "minversion": "8.0",
        "testpaths": ["tests"],
        "pythonpath": ["src"],
        "addopts": ["-ra", "--strict-config", "--strict-markers"],
    }


def test_unverified_release_metadata_is_not_invented(pyproject) -> None:
    project = pyproject["project"]
    prohibited = {
        "authors",
        "maintainers",
        "license",
        "license-files",
        "readme",
        "urls",
    }
    assert prohibited.isdisjoint(project)


def test_classifier_set_is_exact(pyproject) -> None:
    assert pyproject["project"]["classifiers"] == EXPECTED_CLASSIFIERS


def test_package_and_cli_versions_agree() -> None:
    assert qstable_awm.VERSION_INFO == (1, 0, 0)
    assert qstable_awm.__version__ == "1.0.0"
    assert cli.CLI_VERSION == qstable_awm.__version__


def test_public_module_and_command_metadata_agree_with_cli() -> None:
    assert qstable_awm.ANALYTICAL_MODULES == tuple(cli.COMMAND_TO_MODULE.values())
    assert qstable_awm.PUBLIC_COMMANDS == (
        "validate",
        *tuple(cli.COMMAND_TO_MODULE),
        "run-all",
    )


def test_package_root_remains_lightweight() -> None:
    source = (PACKAGE_ROOT / "__init__.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level
            imported_modules.add(prefix + (node.module or ""))
    prohibited_modules = {
        "numpy",
        "pandas",
        "rasterio",
        "yaml",
        ".validation",
        ".pipeline",
        ".crop_support",
    }
    assert imported_modules.isdisjoint(prohibited_modules)


# ---------------------------------------------------------------------------
# Standard wheel contents and metadata
# ---------------------------------------------------------------------------


def test_wheel_build_succeeds_with_expected_filename(wheel_artifact: Path) -> None:
    assert wheel_artifact.name == "qstable_awm-1.0.0-py3-none-any.whl"
    assert wheel_artifact.stat().st_size > 0


def test_wheel_contains_exact_package_module_set(wheel_files) -> None:
    package_members = tuple(
        sorted(name for name in wheel_files if name.startswith("qstable_awm/"))
    )
    assert package_members == tuple(sorted(EXPECTED_WHEEL_PACKAGE_MEMBERS))


def test_wheel_excludes_repository_only_material(wheel_files) -> None:
    prohibited_fragments = (
        "tests/",
        "results/",
        "config/",
        "PHASE3_CURATED_INPUT_LAYER_FINAL_002",
        "__pycache__",
        ".pyc",
        ".pytest_cache",
    )
    assert not any(
        fragment in name for name in wheel_files for fragment in prohibited_fragments
    )


def test_wheel_source_members_match_repository_bytes(wheel_files) -> None:
    for member in EXPECTED_WHEEL_PACKAGE_MEMBERS:
        repository_path = PHASE4_ROOT / "src" / member
        assert wheel_files[member] == repository_path.read_bytes()


def test_wheel_core_metadata_is_exact(wheel_files, dist_info_prefix) -> None:
    metadata = Parser().parsestr(
        wheel_files[f"{dist_info_prefix}/METADATA"].decode("utf-8")
    )
    assert metadata["Name"] == "qstable-awm"
    assert metadata["Version"] == "1.0.0"
    assert metadata["Summary"] == EXPECTED_DESCRIPTION
    assert metadata["Requires-Python"] == ">=3.11"


def test_wheel_dependency_metadata_is_exact(wheel_files, dist_info_prefix) -> None:
    metadata = Parser().parsestr(
        wheel_files[f"{dist_info_prefix}/METADATA"].decode("utf-8")
    )
    requirements = metadata.get_all("Requires-Dist")
    assert requirements[:4] == EXPECTED_METADATA_DEPENDENCIES
    assert requirements[4] == 'pytest<10,>=8; extra == "test"'
    assert metadata.get_all("Provides-Extra") == ["test"]


def test_wheel_classifier_metadata_is_exact(wheel_files, dist_info_prefix) -> None:
    metadata = Parser().parsestr(
        wheel_files[f"{dist_info_prefix}/METADATA"].decode("utf-8")
    )
    assert metadata.get_all("Classifier") == EXPECTED_CLASSIFIERS


def test_wheel_entry_point_is_exact(wheel_files, dist_info_prefix) -> None:
    parser = configparser.ConfigParser()
    parser.read_string(
        wheel_files[f"{dist_info_prefix}/entry_points.txt"].decode("utf-8")
    )
    assert parser.sections() == ["console_scripts"]
    assert dict(parser["console_scripts"]) == {
        "qstable-awm": "qstable_awm.cli:main"
    }


def test_wheel_is_universal_pure_python(wheel_files, dist_info_prefix) -> None:
    text = wheel_files[f"{dist_info_prefix}/WHEEL"].decode("utf-8")
    metadata = Parser().parsestr(text)
    assert metadata["Wheel-Version"] == "1.0"
    assert metadata["Root-Is-Purelib"] == "true"
    assert metadata["Tag"] == "py3-none-any"
    assert metadata["Generator"].startswith("setuptools (")


def test_wheel_top_level_name_is_exact(wheel_files, dist_info_prefix) -> None:
    assert wheel_files[f"{dist_info_prefix}/top_level.txt"].decode("utf-8") == (
        "qstable_awm\n"
    )


def test_wheel_record_hashes_and_sizes_reconcile(wheel_files, dist_info_prefix) -> None:
    record_name = f"{dist_info_prefix}/RECORD"
    rows = list(
        csv.reader(wheel_files[record_name].decode("utf-8").splitlines())
    )
    assert {row[0] for row in rows} == set(wheel_files)
    for relative_path, digest_field, size_field in rows:
        if relative_path == record_name:
            assert digest_field == ""
            assert size_field == ""
            continue
        payload = wheel_files[relative_path]
        algorithm, encoded = digest_field.split("=", 1)
        assert algorithm == "sha256"
        observed = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode(
            "ascii"
        ).rstrip("=")
        assert encoded == observed
        assert int(size_field) == len(payload)


def test_wheel_text_payloads_do_not_embed_absolute_repository_path(wheel_files) -> None:
    root_text = str(PHASE4_ROOT.resolve())
    for name, payload in wheel_files.items():
        if name.endswith((".py", ".txt", "METADATA", "WHEEL", "RECORD")):
            assert root_text not in payload.decode("utf-8", errors="ignore")


def test_two_wheel_builds_are_logically_deterministic(
    wheel_artifact: Path,
    second_wheel_artifact: Path,
) -> None:
    with zipfile.ZipFile(wheel_artifact) as first_archive:
        first = {name: first_archive.read(name) for name in first_archive.namelist()}
    with zipfile.ZipFile(second_wheel_artifact) as second_archive:
        second = {name: second_archive.read(name) for name in second_archive.namelist()}
    assert set(first) == set(second)
    assert first == second


def test_wheel_package_imports_with_expected_version(
    wheel_artifact: Path,
    tmp_path: Path,
) -> None:
    extraction = tmp_path / "wheel_extract"
    with zipfile.ZipFile(wheel_artifact) as archive:
        archive.extractall(extraction)
    code = (
        "import json, sys; "
        f"sys.path.insert(0, {str(extraction)!r}); "
        "import qstable_awm; "
        "print(json.dumps({'version': qstable_awm.__version__, "
        "'file': qstable_awm.__file__}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=_subprocess_environment(),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["version"] == "1.0.0"
    assert Path(payload["file"]).resolve().is_relative_to(extraction.resolve())


# ---------------------------------------------------------------------------
# Editable installation and console entry point
# ---------------------------------------------------------------------------


def test_editable_install_creates_only_distribution_metadata_and_link(
    editable_install,
) -> None:
    site_packages = editable_install["site_packages"]
    names = {path.name for path in site_packages.iterdir()}
    assert any(name.startswith("__editable__.qstable_awm-1.0.0") for name in names)
    assert "qstable_awm-1.0.0.dist-info" in names
    assert "qstable_awm" not in names


def test_editable_link_targets_copied_src_tree(editable_install) -> None:
    site_packages = editable_install["site_packages"]
    pth_files = list(site_packages.glob("__editable__.qstable_awm-1.0.0*.pth"))
    assert len(pth_files) == 1
    target = Path(pth_files[0].read_text(encoding="utf-8").strip()).resolve()
    assert target == (editable_install["project"] / "src").resolve()


def test_editable_distribution_metadata_has_expected_version(editable_install) -> None:
    site_packages = editable_install["site_packages"]
    code = (
        "import importlib.metadata, json, site, sys; "
        f"site.addsitedir({str(site_packages)!r}); "
        "import qstable_awm; "
        "eps=importlib.metadata.entry_points(group='console_scripts'); "
        "ep=[e.value for e in eps if e.name=='qstable-awm']; "
        "print(json.dumps({'version': importlib.metadata.version('qstable-awm'), "
        "'package_version': qstable_awm.__version__, 'file': qstable_awm.__file__, "
        "'entry_points': ep}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=editable_install["root"],
        env=_subprocess_environment(),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["version"] == payload["package_version"] == "1.0.0"
    assert payload["entry_points"] == ["qstable_awm.cli:main"]
    assert Path(payload["file"]).resolve() == (
        editable_install["project"] / "src" / "qstable_awm" / "__init__.py"
    ).resolve()


def test_editable_console_script_reports_version(editable_install) -> None:
    environment = _subprocess_environment()
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            str(editable_install["project"] / "src"),
            str(editable_install["site_packages"]),
        ]
    )
    completed = subprocess.run(
        [str(editable_install["script"]), "--version"],
        cwd=editable_install["root"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "qstable-awm 1.0.0"


def test_editable_direct_url_records_editable_local_install(editable_install) -> None:
    path = (
        editable_install["site_packages"]
        / "qstable_awm-1.0.0.dist-info"
        / "direct_url.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["dir_info"] == {"editable": True}
    assert payload["url"].startswith("file:")
    assert payload["url"].endswith(str(editable_install["project"]).replace(" ", "%20"))


def test_editable_install_does_not_vendor_runtime_dependencies(editable_install) -> None:
    site_packages = editable_install["site_packages"]
    prohibited = {"numpy", "pandas", "rasterio", "yaml", "pytest"}
    assert prohibited.isdisjoint({path.name for path in site_packages.iterdir()})


def test_packaging_tests_leave_canonical_repository_free_of_build_artifacts() -> None:
    assert not (PHASE4_ROOT / "build").exists()
    assert not (PHASE4_ROOT / "dist").exists()
    assert not (PHASE4_ROOT / "src" / "qstable_awm.egg-info").exists()
