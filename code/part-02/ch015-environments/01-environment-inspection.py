# Extracted from: Chapter 15 — Environments, Packaging, and Project Structure
# Source: src/.../ch015-environments.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Inspecting an environment, and demonstrating what reproducibility requires.

Everything here runs against the environment actually executing it — the same
introspection you should record with every experiment.
"""
import hashlib
import importlib.metadata as md
import json
import os
import platform
import random
import subprocess
import sys

import numpy as np

# --- where am I running? -----------------------------------------------------
print("=" * 66)
print("environment")
print("=" * 66)
print(f"python executable : {sys.executable}")
print(f"python version    : {sys.version.split()[0]}")
print(f"platform          : {platform.platform()}")
print(f"in a virtualenv   : {sys.prefix != sys.base_prefix}")
print(f"site-packages     : "
      f"{[p for p in sys.path if 'site-packages' in p][:1]}")

# --- what is installed? ------------------------------------------------------
interesting = ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib",
               "torch", "pytest"]
print(f"\n{'package':<16} {'version':>12}")
installed = {}
for name in interesting:
    try:
        v = md.version(name)
        installed[name] = v
    except md.PackageNotFoundError:
        v = "not installed"
    print(f"{name:<16} {v:>12}")

print(f"\ntotal distributions installed: "
      f"{len(list(md.distributions()))}")

# --- a dependency graph is bigger than you declared -------------------------
def requirements_of(pkg):
    try:
        return md.requires(pkg) or []
    except md.PackageNotFoundError:
        return []


print(f"\ndirect requirements of pandas:")
for req in requirements_of("pandas")[:6]:
    print(f"  {req}")
print("  ...each of which has its own, transitively — which is what a")
print("  resolver must satisfy simultaneously (eq. 15.1).")

# --- an environment fingerprint you can record with a run -------------------
def environment_fingerprint() -> dict:
    """Everything needed to explain a result, minus the data."""
    dists = sorted(
        (d.metadata["Name"], d.version)
        for d in md.distributions()
        if d.metadata["Name"]
    )
    blob = json.dumps(dists, sort_keys=True).encode()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "n_packages": len(dists),
        "env_hash": hashlib.sha256(blob).hexdigest()[:16],
    }


fp = environment_fingerprint()
print(f"\nfingerprint to store alongside every experiment:")
for k, v in fp.items():
    print(f"  {k:<12} {v}")
print("Two runs with different env_hash values are not comparable.")

# --- seeding: one generator is not all of them -------------------------------
print("\n" + "=" * 66)
print("reproducibility: seeding every generator, not just one")
print("=" * 66)


def partial_seed(seed=42):
    random.seed(seed)                 # only stdlib random


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)              # legacy global NumPy state
    return np.random.default_rng(seed)   # and a modern explicit generator


def draw():
    return (round(random.random(), 6),
            round(float(np.random.random()), 6))


partial_seed(); a = draw()
partial_seed(); b = draw()
print(f"seeding only random: {a} then {b}")
print(f"  stdlib matches: {a[0] == b[0]}, numpy matches: {a[1] == b[1]}"
      "   <- numpy drifted")

seed_everything(); c = draw()
seed_everything(); d = draw()
print(f"seeding both       : {c} then {d}")
print(f"  stdlib matches: {c[0] == d[0]}, numpy matches: {c[1] == d[1]}")
assert c == d

# --- prefer an explicit Generator to the global legacy state ----------------
print("\nglobal np.random.seed sets hidden process-wide state; a Generator")
print("is explicit and cannot be disturbed by a library you called:")
rng1 = np.random.default_rng(7)
rng2 = np.random.default_rng(7)
print(f"  rng1: {np.round(rng1.normal(size=3), 4)}")
np.random.seed(999)                   # a library does this behind your back
print(f"  rng2: {np.round(rng2.normal(size=3), 4)}   <- unaffected")

# --- floating-point non-associativity, the root of GPU nondeterminism -------
print("\n" + "=" * 66)
print("why fixed seeds are still not bit-identical on a GPU")
print("=" * 66)
rng = np.random.default_rng(0)
vals = rng.normal(size=100_000).astype(np.float32)
forward = np.float32(0.0)
for v in vals:
    forward += v
backward = np.float32(0.0)
for v in vals[::-1]:
    backward += v
print(f"summing forwards : {forward:.10f}")
print(f"summing backwards: {backward:.10f}")
print(f"identical        : {forward == backward}")
print("Floating-point addition is not associative. GPU kernels accumulate in")
print("a thread-scheduling-dependent order, so the sum differs run to run —")
print("with every seed fixed. Deterministic mode forces an order, and costs")
print("throughput.")

# --- what pip freeze gives you, and what it does not -------------------------
print("\n" + "=" * 66)
print("pinned list vs lock file")
print("=" * 66)
out = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                     capture_output=True, text=True, timeout=120)
lines = [l for l in out.stdout.splitlines() if l and not l.startswith("-e")]
print(f"pip freeze lists {len(lines)} pinned packages, e.g.:")
for line in lines[:4]:
    print(f"  {line}")
print("\nThis pins versions but records no hashes, no resolution metadata,")
print("and no marker for which packages you actually asked for. A real lock")
print("file records all three, which is what makes `sync` exact.")
