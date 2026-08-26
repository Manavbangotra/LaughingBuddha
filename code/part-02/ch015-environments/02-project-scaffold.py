# -*- coding: utf-8 -*-
# Extracted from: Chapter 15 — Environments, Packaging, and Project Structure
# Source: src/.../ch015-environments.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Scaffold a correctly-structured project and prove the layout works.

Creates the src/ layout of section 5.4, installs the package into the current
environment in editable mode, and imports it — demonstrating why the src/
layout catches packaging mistakes that a flat layout hides.
"""
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

root = Path(tempfile.mkdtemp(prefix="scaffold-"))
pkg = root / "src" / "demo_project"
pkg.mkdir(parents=True)
(root / "tests").mkdir()

# --- pyproject.toml: the single manifest ------------------------------------
(root / "pyproject.toml").write_text(textwrap.dedent("""
    [project]
    name = "demo-project"
    version = "0.1.0"
    requires-python = ">=3.10"
    dependencies = ["numpy>=1.24"]

    [project.optional-dependencies]
    dev = ["pytest>=8"]

    [build-system]
    requires = ["setuptools>=68"]
    build-backend = "setuptools.build_meta"

    [tool.setuptools.packages.find]
    where = ["src"]
""").lstrip())

(pkg / "__init__.py").write_text('__version__ = "0.1.0"\n')
(pkg / "features.py").write_text(textwrap.dedent('''
    """A pure transformation — trivially testable (Chapter 14)."""
    import numpy as np


    def standardise(x: np.ndarray) -> np.ndarray:
        """Centre and scale to unit variance."""
        x = np.asarray(x, dtype=float)
        std = x.std()
        return (x - x.mean()) / (std if std else 1.0)
''').lstrip())

(root / "tests" / "test_features.py").write_text(textwrap.dedent('''
    import numpy as np
    from demo_project.features import standardise


    def test_standardise_gives_zero_mean_unit_variance():
        out = standardise([1.0, 2.0, 3.0, 4.0])
        assert np.isclose(out.mean(), 0.0)
        assert np.isclose(out.std(), 1.0)


    def test_standardise_handles_constant_input():
        out = standardise([5.0, 5.0, 5.0])
        assert np.allclose(out, 0.0)
''').lstrip())

(root / ".gitignore").write_text(".venv/\n__pycache__/\n*.egg-info/\ndata/\n")

print("created project:")
for p in sorted(root.rglob("*")):
    if "__pycache__" not in str(p) and "egg-info" not in str(p):
        print(f"  {p.relative_to(root)}")

# --- the src/ layout means the package is NOT importable until installed ----
probe = [sys.executable, "-c", "import demo_project; print('imported')"]
before = subprocess.run(probe, cwd=root, capture_output=True, text=True)
print(f"\nimport from the project root BEFORE installing: "
      f"{'succeeded' if before.returncode == 0 else 'failed (as intended)'}")
print("  With a flat layout this would have succeeded by accident, hiding a")
print("  packaging error until someone else cloned the repository.")

# --- install it, editable ----------------------------------------------------
install = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps", "-q"],
    cwd=root, capture_output=True, text=True, timeout=600)
print(f"\neditable install: {'ok' if install.returncode == 0 else 'FAILED'}")
if install.returncode != 0:
    print(install.stderr[-500:])

after = subprocess.run(probe, cwd=root, capture_output=True, text=True)
print(f"import after installing: {after.stdout.strip() or after.stderr[-200:]}")

# --- and now the tests run from anywhere, not just the project root ---------
tests = subprocess.run([sys.executable, "-m", "pytest", str(root / "tests"),
                        "-q", "--no-header"],
                       cwd=tempfile.gettempdir(), capture_output=True,
                       text=True, timeout=600)
print(f"\npytest run from a DIFFERENT directory:")
print("  " + "\n  ".join(tests.stdout.strip().splitlines()[-3:]))

# --- editable means edits take effect without reinstalling ------------------
(pkg / "features.py").write_text(
    (pkg / "features.py").read_text().replace('"""Centre and scale to unit variance."""',
                                              '"""EDITED docstring."""'))
check = subprocess.run(
    [sys.executable, "-c",
     "from demo_project.features import standardise; print(standardise.__doc__)"],
    cwd=tempfile.gettempdir(), capture_output=True, text=True)
print(f"\nafter editing the source, with no reinstall: {check.stdout.strip()!r}")

# --- clean up ----------------------------------------------------------------
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q",
                "demo-project"], capture_output=True, timeout=300)
import shutil
shutil.rmtree(root, ignore_errors=True)
print("\nuninstalled and cleaned up.")
