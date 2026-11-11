"""Guard against the container-startup crash loop.

What happened on 2026-11-10: modal_app/stage.py did
``sys.path.insert(0, Path(__file__).parent)`` and then ``from images import
...``. That resolves fine locally, so everything looked correct. Inside the
container it fails, because Modal ships the ``binder_bench`` package via
``add_local_python_source`` and does NOT ship the modal_app directory, so
``images`` does not exist there.

The failure mode is what makes this worth a test. A missing import at module
scope is a container STARTUP error, and Modal responds by relaunching the
container, repeatedly: "Function probe.probe is crash-looping: containers are
repeatedly failing to start." ``retries=0`` does not prevent it, because
retries govern function failures, not startup failures. So a typo that would be
a one-line local traceback instead becomes a billing loop that runs until
someone notices and stops the app.

That one was CPU-only and cost pennies. The same mistake on a GPU function
would not.

These tests are deliberately static. They catch the mistake without launching
anything, which is the only way to catch it cheaply.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

MODAL_APP_DIR = Path(__file__).resolve().parents[1] / "modal_app"
SHIPPED_PACKAGE = "binder_bench"

# Modules the container is guaranteed to have: the stdlib, modal itself, and
# whatever the image installs. Anything first-party must come from the shipped
# package.
ALLOWED_TOP_LEVEL = {
    "__future__",
    "modal",
    "binder_bench",
    # stdlib and image-provided, used at module scope
    "annotations",
    "sys",
    "os",
    "json",
    "pathlib",
    "subprocess",
    "dataclasses",
    "typing",
    "numpy",
    "pandas",
}


def _app_files() -> list[Path]:
    files = sorted(p for p in MODAL_APP_DIR.glob("*.py") if p.name != "__init__.py")
    assert files, f"no Modal app files found under {MODAL_APP_DIR}"
    return files


@pytest.mark.parametrize("path", _app_files(), ids=lambda p: p.name)
def test_no_sibling_imports_in_modal_apps(path: Path):
    """Every first-party import must come from the shipped package."""
    tree = ast.parse(path.read_text())
    sibling_names = {p.stem for p in _app_files()}

    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level and node.level > 0:
                offenders.append(f"relative import (level {node.level})")
            elif root in sibling_names:
                offenders.append(f"from {node.module} import ...")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in sibling_names:
                    offenders.append(f"import {alias.name}")

    assert not offenders, (
        f"{path.name} imports a sibling module in modal_app/, which exists "
        f"locally but not in the container. This causes a startup crash loop "
        f"that retries=0 does not stop. Move the shared code into "
        f"src/{SHIPPED_PACKAGE}/ instead. Offenders: {offenders}"
    )


@pytest.mark.parametrize("path", _app_files(), ids=lambda p: p.name)
def test_no_sys_path_manipulation(path: Path):
    """sys.path hacks are how the sibling import got past local testing."""
    src = path.read_text()
    assert "sys.path.insert" not in src and "sys.path.append" not in src, (
        f"{path.name} manipulates sys.path. That makes a container-only import "
        f"failure invisible locally. Import from src/{SHIPPED_PACKAGE}/ instead."
    )


@pytest.mark.parametrize("path", _app_files(), ids=lambda p: p.name)
def test_module_scope_imports_are_container_safe(path: Path):
    """Imports at module scope must be ones the container definitely has.

    Heavy or image-specific imports belong inside the function body, where they
    run in the container rather than on the laptop while Modal parses the file
    looking for the entrypoint.
    """
    tree = ast.parse(path.read_text())
    bad = []
    for node in tree.body:  # module scope only
        if isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root and root not in ALLOWED_TOP_LEVEL:
                bad.append(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_TOP_LEVEL:
                    bad.append(alias.name)

    assert not bad, (
        f"{path.name} imports {bad} at module scope. If the container may not "
        f"have it, move the import inside the function body."
    )


@pytest.mark.parametrize("path", _app_files(), ids=lambda p: p.name)
def test_gpu_functions_set_an_explicit_timeout(path: Path):
    """modal's timeout defaults to 300s, which silently kills and bills folds."""
    tree = ast.parse(path.read_text())
    missing = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call):
                continue
            kwargs = {k.arg for k in dec.keywords if k.arg}
            if "image" in kwargs and "timeout" not in kwargs:
                missing.append(node.name)

    assert not missing, (
        f"{path.name}: {missing} have a Modal decorator without an explicit "
        f"timeout. The default is 300 seconds, which would kill a five-model "
        f"fold partway through and bill for the attempt."
    )


@pytest.mark.parametrize("path", _app_files(), ids=lambda p: p.name)
def test_no_warm_pool_or_buffer(path: Path):
    """A kept-warm GPU container is the most expensive way to wait."""
    tree = ast.parse(path.read_text())
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg == "min_containers":
                offenders.append("min_containers is set")
            if kw.arg == "buffer_containers" and not (
                isinstance(kw.value, ast.Constant) and kw.value.value == 0
            ):
                offenders.append("buffer_containers is not 0")

    assert not offenders, f"{path.name}: {offenders}"
