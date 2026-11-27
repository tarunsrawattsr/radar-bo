"""Extended layer-separation and ground-truth-leakage guards (M3 instruction 12,
extended for M5 instruction 30).

Import-matrix rules enforced per file class:

* ``config`` / ``environment`` / ``radar`` / ``simulation``: no torch-stack
  imports and no ``radar_bo.optimization`` imports at all (README section 3:
  config and simulation must not import the optimizer).
* ``optimization`` top level (M4 utility layer: allocation, constraints,
  resource accounting, objective, simulation state): stays NumPy-only and must
  not depend on the BO engine (``radar_bo.optimization.bo``) — existing
  components stay independent of the BO engine.
* ``optimization/bo`` (M5 generic BO engine): MAY import the torch stack (that
  is its purpose) but MUST NOT import the simulator layers
  (``radar_bo.environment`` / ``radar_bo.radar`` / ``radar_bo.simulation``) nor
  the M4 optimizer modules: the problem interface is application-free.
* The estimation core (``environment`` / ``radar``) stays NumPy-only beyond
  torch (no scipy/matplotlib/pandas) — the "standalone NumPy baseline" claim.
* No function in the environment/radar layers may accept a parameter that looks
  like ground truth (README non-negotiables).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

TORCH_ROOTS = {"torch", "botorch", "gpytorch", "linear_operator"}
NUMPY_ONLY_EXTRA_BANS = {"scipy", "matplotlib", "pandas"}
SIMULATOR_MODULES = ("radar_bo.environment", "radar_bo.radar", "radar_bo.simulation")
M4_OPTIMIZER_MODULES = (
    "radar_bo.optimization.allocation",
    "radar_bo.optimization.constraints",
    "radar_bo.optimization.resource_accounting",
    "radar_bo.optimization.objective",
    "radar_bo.optimization.simulation_state",
)
OPTIMIZER_MODULES = ("radar_bo.optimization",)  # any submodule (prefix match)

GUARDED_PACKAGES = ("config", "environment", "radar", "optimization", "simulation")

TRUTH_PARAM_TOKENS = ("truth", "true_state", "true_position", "true_velocity", "ground")


def _iter_module_paths(package: str):
    return sorted((PACKAGE_ROOT / package).rglob("*.py"))


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


def _matches_banned_module(module_name: str, banned: tuple[str, ...]) -> bool:
    """Exact or submodule match (``radar_bo.optimization.bo`` counts as an
    import of the banned ``radar_bo.optimization`` layer)."""
    return any(module_name == b or module_name.startswith(b + ".") for b in banned)


def _banned_for_file(py_file: str) -> tuple[set[str], tuple[str, ...]]:
    """(banned import roots, banned module prefixes) for one guarded file."""
    f = py_file.replace("\\", "/")
    if f.startswith("optimization/bo/"):
        # M5 BO engine: torch allowed; simulator and M4-optimizer imports banned.
        return set(), SIMULATOR_MODULES + M4_OPTIMIZER_MODULES
    if f.startswith("optimization/"):
        # M4 optimizer layer: NumPy-only and independent of the BO engine.
        return TORCH_ROOTS, ("radar_bo.optimization.bo",)
    # config / environment / radar / simulation: no torch, no optimizer at all.
    return TORCH_ROOTS, OPTIMIZER_MODULES


@pytest.mark.parametrize(
    "py_file",
    sorted(
        str(p.relative_to(PACKAGE_ROOT))
        for pkg in GUARDED_PACKAGES
        for p in (PACKAGE_ROOT / pkg).rglob("*.py")
    ),
)
def test_no_optimizer_or_torch_imports(py_file: str):
    banned_roots, banned_modules = _banned_for_file(py_file)
    tree = ast.parse((PACKAGE_ROOT / py_file).read_text(encoding="utf-8"))
    roots, modules = _imported_roots_and_modules(tree)
    bad_roots = roots & banned_roots
    assert not bad_roots, f"{py_file} imports banned roots: {sorted(bad_roots)}"
    bad_modules = {m for m in modules if _matches_banned_module(m, banned_modules)}
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
        assert not (roots & TORCH_ROOTS), f"{path.name} imports banned root"
        assert not {m for m in modules if _matches_banned_module(m, OPTIMIZER_MODULES)}, (
            f"{path.name} imports banned module"
        )


def test_problem_interface_is_radar_free():
    """M5 instruction 30: the generic problem interface and the whole bo/
    subpackage contain no radar/simulator references (defense in depth on top
    of the parametrized per-file guard)."""
    bo_root = PACKAGE_ROOT / "optimization" / "bo"
    offenders = []
    for path in sorted(bo_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        _, modules = _imported_roots_and_modules(tree)
        for module in modules:
            if _matches_banned_module(module, SIMULATOR_MODULES + M4_OPTIMIZER_MODULES):
                offenders.append(f"{path.relative_to(PACKAGE_ROOT)}:{module}")
    assert not offenders, f"BO engine has radar-layer dependencies: {offenders}"


# ---- M7 extension (item 32): experiment-layer separation --------------------

EXPERIMENT_FRAMEWORK = "radar_bo.experiments.framework"
EXPERIMENT_ROOT = "radar_bo.experiments"
SIMULATOR_AND_OPTIMIZER = ("radar_bo.environment", "radar_bo.radar", "radar_bo.simulation",
                           "radar_bo.optimization")


def test_core_layers_do_not_import_experiment_framework():
    """config/environment/radar/numerics/simulation must not import the
    experiment framework or any experiment orchestration (item 32)."""
    import re

    for package in ("config", "environment", "radar", "simulation", "numerics.py"):
        paths = (PACKAGE_ROOT / package).rglob("*.py") if not package.endswith(".py") else [PACKAGE_ROOT / package]
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            roots, modules = _imported_roots_and_modules(tree)
            bad = {m for m in roots | modules if m == EXPERIMENT_ROOT or m.startswith(EXPERIMENT_ROOT)}
            assert not bad, f"{path.name} (layer {package}) imports experiment orchestration: {sorted(bad)}"


def test_experiment_framework_does_not_import_simulator_layers():
    """The M7 framework may depend on generic optimization interfaces and
    config only — never on radar/environment/simulation modules (item 29/32:
    interface validation uses mock adapters, no physical sensing models)."""
    framework_dir = PACKAGE_ROOT / "experiments" / "framework"
    for path in sorted(framework_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        roots, modules = _imported_roots_and_modules(tree)
        banned = {m for m in roots | modules
                  if any(m == s or m.startswith(s + ".") for s in SIMULATOR_AND_OPTIMIZER)
                  and not m.startswith("radar_bo.optimization.bo")}
        # 'radar_bo.optimization' prefix ban applies except the generic bo engine
        unexpected = {
            m for m in roots | modules
            if (m.startswith("radar_bo.environment") or m.startswith("radar_bo.radar")
                or m.startswith("radar_bo.simulation"))
        }
        assert not unexpected, f"{path.name} imports simulator layers: {sorted(unexpected)}"
        assert not banned or banned == set(), f"{path.name} imports banned optimizer modules: {sorted(banned)}"


def test_no_circular_imports_in_experiment_layer():
    """Importing the framework must not pull the package into an import cycle:
    import it fresh in a subprocess and verify core layers stay import-light."""
    import subprocess
    import sys

    code = (
        "import radar_bo.experiments.framework as fw;"
        "import radar_bo.config as cfg;"
        "import radar_bo.numerics as num;"
        "print('OK')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
