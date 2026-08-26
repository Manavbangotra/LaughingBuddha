# -*- coding: utf-8 -*-
# Extracted from: Chapter 18 — Visualization with Matplotlib
# Source: src/.../ch018-visualization.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Matplotlib's object model, and the failure modes of section 6.

Renders with the Agg backend so it runs headless, and writes files to a
temporary directory rather than displaying anything.
"""
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # must come before importing pyplot
import matplotlib.pyplot as plt
import numpy as np

out_dir = Path(tempfile.mkdtemp(prefix="figs-"))
rng = np.random.default_rng(0)
print(f"backend: {matplotlib.get_backend()}   (headless)")
print(f"writing to: {out_dir}\n")

# --- the object hierarchy ----------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 3))
x = np.linspace(0, 10, 200)
line, = ax.plot(x, np.sin(x), label="sin")     # returns a list; unpack it
ax.set_xlabel("x")
ax.set_ylabel("sin(x)")
ax.set_title("the object-oriented interface")
ax.legend()

print(f"figure : {type(fig).__name__}")
print(f"axes   : {type(ax).__name__}, {len(fig.axes)} on this figure")
print(f"line   : {type(line).__name__}, colour {line.get_color()!r}")
line.set_color("crimson")                       # artists are mutable
print(f"         recoloured to {line.get_color()!r}")
fig.savefig(out_dir / "01-basic.png", dpi=110, bbox_inches="tight")
plt.close(fig)

# --- eq. 18.1: Anscombe's quartet -------------------------------------------
print("\n" + "=" * 66)
print("Anscombe's quartet: identical statistics, four different realities")
print("=" * 66)

anscombe = {
    "I":   ([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
            [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]),
    "II":  ([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
            [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]),
    "III": ([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
            [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]),
    "IV":  ([8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8],
            [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]),
}

print(f"{'set':>5} {'mean x':>8} {'mean y':>8} {'var x':>8} {'var y':>8} "
      f"{'corr':>7} {'slope':>7} {'intercept':>10}")
for name, (xs, ys) in anscombe.items():
    xa, ya = np.array(xs, float), np.array(ys, float)
    slope, intercept = np.polyfit(xa, ya, 1)
    print(f"{name:>5} {xa.mean():>8.2f} {ya.mean():>8.2f} "
          f"{xa.var(ddof=1):>8.2f} {ya.var(ddof=1):>8.2f} "
          f"{np.corrcoef(xa, ya)[0,1]:>7.3f} {slope:>7.3f} {intercept:>10.3f}")

print("\nEvery statistic agrees. Now look at them:")
descriptions = {"I": "linear + noise", "II": "a parabola",
                "III": "a line + 1 outlier", "IV": "1 point sets the slope"}
fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True)
for axis, (name, (xs, ys)) in zip(axes.flat, anscombe.items()):
    xa, ya = np.array(xs, float), np.array(ys, float)
    axis.scatter(xa, ya, s=28, zorder=3)
    fit = np.polyfit(xa, ya, 1)
    grid = np.linspace(2, 20, 50)
    axis.plot(grid, np.polyval(fit, grid), lw=1.2, color="crimson")
    axis.set_title(f"{name}: {descriptions[name]}", fontsize=9)
    axis.grid(alpha=0.3)
fig.suptitle("Anscombe's quartet — identical summary statistics")
fig.tight_layout()
fig.savefig(out_dir / "02-anscombe.png", dpi=110, bbox_inches="tight")
plt.close(fig)
for name, desc in descriptions.items():
    print(f"  {name:>3}: {desc}")
print("Only set I justifies reporting a correlation (eq. 18.2).")

# --- eq. 18.4: a truncated bar axis --------------------------------------
print("\n" + "=" * 66)
print("truncated axes exaggerate differences")
print("=" * 66)
models, acc = ["A", "B", "C"], np.array([0.94, 0.96, 0.945])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
ax1.bar(models, acc, color="steelblue")
ax1.set_ylim(0, 1)
ax1.set_title("honest: axis from zero")
ax2.bar(models, acc, color="indianred")
ax2.set_ylim(0.93, 0.97)
ax2.set_title("misleading: truncated axis")
for a in (ax1, ax2):
    a.set_ylabel("accuracy")
fig.tight_layout()
fig.savefig(out_dir / "03-truncated.png", dpi=110, bbox_inches="tight")
plt.close(fig)

b_zero, b_trunc = 0.0, 0.93
ratio_zero = (acc[1] - b_zero) / (acc[0] - b_zero)
ratio_trunc = (acc[1] - b_trunc) / (acc[0] - b_trunc)
print(f"accuracies: A={acc[0]}, B={acc[1]}  (a {acc[1]-acc[0]:.3f} difference)")
print(f"  drawn length ratio, axis from 0.00 : {ratio_zero:.3f}  (eq. 18.3)")
print(f"  drawn length ratio, axis from 0.93 : {ratio_trunc:.3f}  (eq. 18.4)")
print(f"Truncation exaggerates the visual difference "
      f"{ratio_trunc/ratio_zero:.1f}-fold. Same data.")

# --- section 6.3: overplotting hides density --------------------------------
print("\n" + "=" * 66)
print("overplotting")
print("=" * 66)
n = 60_000
xs = rng.normal(size=n)
ys = xs * 0.6 + rng.normal(size=n) * 0.8

fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
axes[0].scatter(xs, ys, s=4)
axes[0].set_title(f"{n:,} points, alpha=1 — saturated")
axes[1].scatter(xs, ys, s=4, alpha=0.02)
axes[1].set_title("alpha=0.02 — density visible")
hb = axes[2].hexbin(xs, ys, gridsize=40, cmap="viridis")
axes[2].set_title("hexbin — density encoded")
fig.colorbar(hb, ax=axes[2], label="count")
fig.tight_layout()
fig.savefig(out_dir / "04-overplotting.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"plotted {n:,} points three ways; only the last two show density")

# --- log scale ----------------------------------------------------------------
epochs = np.arange(1, 101)
loss = 4.0 * np.exp(-epochs / 12) + 0.02 + rng.normal(0, 0.01, 100)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
ax1.plot(epochs, loss); ax1.set_title("linear y"); ax1.set_ylabel("loss")
ax2.plot(epochs, loss); ax2.set_yscale("log"); ax2.set_title("log y")
for a in (ax1, ax2):
    a.set_xlabel("epoch"); a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(out_dir / "05-logscale.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"\nloss fell {loss[0]:.2f} -> {loss[-1]:.3f}")
print("On a linear axis the last 50 epochs look flat; on a log axis the")
print("continuing improvement is visible. Both are the same numbers.")

# --- colormaps ----------------------------------------------------------------
print("\n" + "=" * 66)
print("perceptual uniformity")
print("=" * 66)
grad = np.linspace(0, 1, 256).reshape(1, -1)
fig, axes = plt.subplots(3, 1, figsize=(7, 2.2))
for a, cmap in zip(axes, ["viridis", "jet", "RdBu"]):
    a.imshow(grad, aspect="auto", cmap=cmap)
    a.set_yticks([]); a.set_xticks([])
    a.set_ylabel(cmap, rotation=0, ha="right", va="center", fontsize=9)
fig.suptitle("viridis (uniform) · jet (not) · RdBu (diverging)", fontsize=10)
fig.tight_layout()
fig.savefig(out_dir / "06-colormaps.png", dpi=110, bbox_inches="tight")
plt.close(fig)

# Measure non-uniformity: perceived lightness should change at a constant rate.
for cmap_name in ("viridis", "jet"):
    cmap = matplotlib.colormaps[cmap_name]
    rgb = cmap(np.linspace(0, 1, 256))[:, :3]
    lightness = rgb @ np.array([0.2126, 0.7152, 0.0722])   # luminance
    steps = np.abs(np.diff(lightness))
    # Coefficient of variation is the scale-free measure: the sd of the step
    # sizes relative to their mean. Perfect uniformity would give zero.
    print(f"{cmap_name:<9} lightness step cv = {steps.std()/steps.mean():.3f}")
print("jet's steps are about three times as variable, relative to their own")
print("size, as viridis's. That non-uniformity creates visible bands the data")
print("does not contain, and it is why viridis is the default.")

print(f"\nfigures written: {sorted(p.name for p in out_dir.glob('*.png'))}")
print(f"open figures still held by pyplot: {len(plt.get_fignums())} "
      f"(closed each one — see the production tip)")

import shutil
shutil.rmtree(out_dir, ignore_errors=True)
