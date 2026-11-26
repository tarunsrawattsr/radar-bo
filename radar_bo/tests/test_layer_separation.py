"""Extended layer-separation and ground-truth-leakage guards (M3 instruction 12).

* The tracking layer (``radar_bo.environment``, ``radar_bo.radar``) and the
  config layer must not import optimization libraries (torch/botorch/gpytorch)
  or the optimizer package. The estimation core is NumPy-only (no scipy/
  matplotlib), keeping the "standalone NumPy baseline" claim verifiable.
* No function in the environment/radar layers may accept a parameter that
  looks like ground truth: the true simulated state is for simulation and
  evaluation only (README non-negotiables).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

BANNED_ROOTS = {"torch", "botorch", "gpytorch", "linear_operator"}
BANNED_MODULES = {"radar_bo.optimization"}
NUMPY_ONLY_EXTRA_BANS = {"scipy", "matplotlib", "pandas"}

GUARDED_PACKAGES = ("config", "environment", "radar")

TRUTH_PARAM_TOKENS = ("truth", "true_state", "true_position", "true_velocity", "ground")


def _iter_module_paths(package: str):
    return sorted((PACKAGE_ROOT / package).glob("*.py"))


def _imported_roots_and_modules(tree: ast.AST):
    roots, modules = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            roots.add((node.module or "").split(".")[0])
            modules.add(node.module or "")
    return roots, modules


@pytest.mark.parametrize(
    "py_file",
    sorted(str(p.relative_to(PACKAGE_ROOT)) for pkg in GUARDED_PACKAGES for p in (PACKAGE_ROOT / pkg).glob("*.py")),
)
def test_no_optimizer_or_torch_imports(py_file: str):
    tree = ast.parse((PACKAGE_ROOT / py_file).read_text(encoding="utf-8"))
    roots, modules = _imported_roots_and_modules(tree)
    bad_roots = roots & (BANNED_ROOTS | BANNED_MODULES)
    assert not bad_roots, f"{py_file} imports banned: {sorted(bad_roots)}"
    bad_modules = modules & BANNED_MODULES
    assert not bad_modules, f"{py_file} imports banned module: {sorted(bad_modules)}"


@pytest.mark.parametrize("package", ("environment", "radar"))
def test_estimation_core_is_numpy_only(package: str):
    """environment/ and radar/ stay NumPy-only (standalone baseline, M3)."""
    for path in _iter_module_paths(package):
        roots, _ = _imported_roots_and_modules(ast.parse(path.read_text(encoding="utf-8")))
        bad = roots & NUMPY_ONLY_EXTRA_BANS
        assert not bad, f"{path.name} imports {sorted(bad)} (estimation core must be NumPy-only)"


@pytest.mark.parametrize("package", ("environment", "radar"))
def test_no_ground_truth_parameters(package: str):
    """No estimator/environment function accepts ground-truth-like inputs."""
    offenders = []
    for path in _iter_module_paths(package):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for arg in node.args.args + node.args.kwonlyargs:
                    lowered = arg.arg.lower()
                    if any(token in lowered for token in TRUTH_PARAM_TOKENS):
                        offenders.append(f"{path.name}:{node.name}:{arg.arg}")
    assert not offenders, f"Ground-truth-like parameters found: {offenders}"


def test_config_layer_guard_still_in_place():
    """Retain the original M2 guard scope: config must not import optimizer."""
    for path in _iter_module_paths("config"):
        roots, modules = _imported_roots_and_modules(ast.parse(path.read_text(encoding="utf-8")))
        assert not (roots & BANNED_ROOTS), f"{path.name} imports banned root"
        assert not (modules & BANNED_MODULES), f"{path.name} imports banned module"
