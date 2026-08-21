---
id: py-visualization
number: 18
part: II
tier: focused
status: reviewed
requires: [py-pandas]
provides: [figure-axes]
citations: [hunter2007]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Distinguish the figure from the axes, and use the object-oriented interface
   rather than global `pyplot` state.
2. Explain why the state-machine interface breaks down in programmatic code.
3. Choose a chart type from the structure of the question rather than by habit.
4. Build multi-panel figures with shared scales.
5. Produce figures that render correctly in a headless environment and in a
   document.
6. Recognise the standard ways a chart misleads, including truncated axes and
   summary statistics that hide their distributions.
7. Plot the diagnostics that recur throughout this book: learning curves,
   residuals, distributions, and confusion matrices.

## 2. Why This Matters

Visualisation in this book is a diagnostic instrument, not a presentation
medium. Its job is to show you things about your data and your models that
numbers do not.

Anscombe's quartet is the standard demonstration and it earns its place: four
datasets with identical means, variances, correlations and regression lines,
which look nothing alike. Summary statistics are lossy, and a single plot
reveals in a second what a table of moments conceals entirely.
{{sec:6-mathematical-foundation}} constructs it.

The specific plots you will rely on later are diagnostic ones. A learning curve
distinguishes overfitting from underfitting ({{ch:ml-metrics}}). A residual plot
reveals that a linear model has missed a nonlinearity
({{ch:ml-linear-regression}}). A distribution plot exposes the skew that will
break your assumption of normality ({{ch:math-inference}}). An attention
heatmap shows which of two very different failure modes a retrieval system is in
({{ch:tf-scaled-dot-product}}). None of these are decoration; each one answers a
question that cannot be answered from a number.

Matplotlib is the substrate {{cite:hunter2007}}. It is not the friendliest
library, but almost every other Python plotting tool either builds on it or
defines itself against it, so learning its object model is the transferable
skill.

## 3. Prerequisites

{{ch:py-pandas}} for DataFrames, and {{ch:py-numpy}} for arrays — Matplotlib
consumes both directly.

## 4. Intuitive Explanation

### 4.1 Figure and axes

Two objects carry almost all of Matplotlib's structure.

A **Figure** is the whole canvas — the page. An **Axes** is one plot on it, with
its own coordinate system, ticks, labels and data. A figure holds one or many
axes.

```text
┌─ Figure ───────────────────────────────┐
│  ┌─ Axes ──────┐   ┌─ Axes ──────┐     │
│  │  a plot     │   │  another    │     │
│  │  x/y scales │   │  plot       │     │
│  └─────────────┘   └─────────────┘     │
│  suptitle, shared legend, layout       │
└────────────────────────────────────────┘
```

Confusingly, "axes" is singular here — one `Axes` object is one plot, not one
axis. The individual x and y axes are `ax.xaxis` and `ax.yaxis`.

Almost everything you want to do is a method on an `Axes`: `ax.plot`,
`ax.scatter`, `ax.set_xlabel`, `ax.legend`.

### 4.2 Two interfaces, and why one of them is a trap

Matplotlib has a **state-machine** interface inherited from MATLAB, and an
**object-oriented** one.

```python {tier=C name=two-interfaces}
# state machine: operates on "the current figure", whatever that is
plt.plot(x, y)
plt.xlabel("time")
plt.title("results")

# object-oriented: explicit about what you are modifying
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xlabel("time")
ax.set_title("results")
```

The first is shorter and fine for a throwaway line in a notebook. It fails as
soon as code becomes programmatic, because "the current figure" is global
mutable state. A function that plots into the current figure cannot be called
twice safely, cannot be composed, and interacts unpredictably with anything else
that plots.

> IMPORTANT: Use `fig, ax = plt.subplots()` and call methods on `ax`. This book
> does so throughout. A function that takes an `ax` parameter and draws into it
> is reusable and testable; one that calls `plt.plot` is neither.

### 4.3 Choosing a chart

The chart type follows from the question, and there are not many questions.

{#tbl:chart-choice caption="Chart selection by question. Most of what goes wrong is choosing by habit rather than by question."}

| Question | Chart |
|---|---|
| How does one variable change over time or order? | line |
| How are two continuous variables related? | scatter |
| What is the distribution of one variable? | histogram, KDE |
| How do distributions compare across groups? | box, violin, strip |
| How do quantities compare across categories? | bar |
| How do two categorical dimensions interact? | heatmap |
| How much of a total does each part contribute? | stacked bar — not a pie |

Two rules worth internalising. **Bar charts must start at zero**, because the
bar's length encodes the value and truncating it lies proportionally. **Line
charts need not**, because the line's *slope* encodes the change and zero is
often irrelevant.

## 5. Formal Explanation

### 5.1 The object hierarchy

Everything drawn is an **Artist**. The hierarchy:

```text
Figure
├── Axes
│   ├── XAxis / YAxis  → ticks, labels, gridlines
│   ├── Line2D, PathCollection, Rectangle, ...   (the data)
│   ├── Legend
│   └── Text (title, annotations)
└── Text (suptitle)
```

Plotting methods return the artists they create, which is how you modify them
later:

```python {tier=C name=artists}
line, = ax.plot(x, y)          # note the comma: plot returns a list
line.set_color("crimson")
line.set_linewidth(2)
```

### 5.2 Creating figures

```python {tier=C name=creating-figures}
fig, ax = plt.subplots(figsize=(6, 4), dpi=120)           # one panel
fig, axes = plt.subplots(2, 3, figsize=(12, 6),           # a grid
                         sharex=True, sharey=True)
fig, axes = plt.subplot_mosaic([["big", "top"],           # uneven layout
                                ["big", "bottom"]])
```

`sharex`/`sharey` matter more than they look: panels on different scales invite
false comparison, and sharing makes the comparison honest.

`fig.tight_layout()` or `layout="constrained"` prevents overlapping labels.

### 5.3 Backends and headless rendering

A **backend** is what Matplotlib renders through. Interactive backends open a
window; file backends write to disk.

```python {tier=C name=backend}
import matplotlib
matplotlib.use("Agg")      # must precede pyplot import
import matplotlib.pyplot as plt
```

`Agg` is the non-interactive raster backend. It is what you need on a server,
in CI, or in any script that will run without a display — which includes every
Tier A listing in this book.

> PRODUCTION TIP: In any long-running script, close figures explicitly with
> `plt.close(fig)`. Matplotlib keeps a reference to every figure created through
> `pyplot`, so a loop that creates a thousand figures leaks all thousand. In a
> training loop that logs a plot per epoch, this is a genuine memory leak and a
> common one.

Save with `fig.savefig(path, dpi=150, bbox_inches="tight")`. Use PNG for raster
output and SVG or PDF for anything that will be printed or zoomed.

### 5.4 Scales and axis limits

```python {tier=C name=scales}
ax.set_yscale("log")           # for data spanning orders of magnitude
ax.set_xlim(0, 100)
ax.set_ylim(bottom=0)          # bars must start at zero
```

Log scales are the right default whenever data spans orders of magnitude —
learning rates, loss curves, latency distributions, model sizes. On a linear
axis, a loss falling from 4.0 to 0.4 and then from 0.4 to 0.04 looks like one
big improvement and one negligible one, when both are tenfold. On a log axis
they are equal drops, which is the honest picture ({{ch:math-functions}}).

### 5.5 Direct labelling and small multiples

Two techniques do more for readability than any amount of styling.

**Direct labelling.** A legend forces the reader to look away from the data,
decode a colour, and look back — for every series, every time. Labelling each
line at its right-hand end removes that round trip entirely:

```python {tier=C name=direct-labelling}
for name, series in results.items():
    ax.plot(x, series)
    ax.annotate(name, (x[-1], series[-1]),
                xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=9)
```

This also survives greyscale printing and colour blindness, because the
identity of a series no longer depends on its colour
({{sec:5-formal-explanation}} below). Use a legend when series overlap so
heavily that labels would collide; otherwise label directly.

**Small multiples.** When comparing many groups, one panel per group on
*shared axes* beats one crowded panel with many overlapping series. The eye
compares positions across panels well and disentangles overlapping lines
badly.

```python {tier=C name=small-multiples}
fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True, sharey=True)
for ax, (name, group) in zip(axes.flat, df.groupby("cohort")):
    ax.plot(group["x"], group["y"])
    ax.set_title(name, fontsize=9)
```

The shared axes are what make it work: without them each panel has its own
scale and the visual comparison is meaningless, which is a common and
convincing way to mislead yourself. With eight series on one panel you are
reading a tangle; with eight small panels you are reading eight shapes.

> PRODUCTION TIP: Both techniques become more valuable as the figure gets more
> important. A quick exploratory plot can carry a legend and overlap; a figure
> that will be read by someone who was not present when it was made should be
> readable without a key.

### 5.6 Showing uncertainty

{{ch:math-inference}} established that every measured quantity carries
uncertainty. A plot that shows only point estimates hides it, and hiding it is
how two indistinguishable models come to look different.

```python {tier=C name=error-bands}
ax.plot(x, mean, label="mean")
ax.fill_between(x, mean - 1.96 * se, mean + 1.96 * se, alpha=0.2)
ax.errorbar(x, mean, yerr=1.96 * se, fmt="o", capsize=3)
```

`fill_between` suits a continuous curve — a learning curve averaged over seeds,
a metric over time. `errorbar` suits discrete categories.

Two rules. **Always say what the interval represents** — a standard deviation, a
standard error, and a 95% confidence interval are three different widths, and a
band without a stated meaning is decoration. **Show the raw points when there
are few of them**: with five runs, plotting all five conveys more than a mean
and a band computed from five numbers, and it does not imply a precision the
sample cannot support.

> IMPORTANT: When comparing two models, overlapping error bars do *not*
> automatically mean the difference is insignificant — the correct test is on
> the difference, and for a shared test set it should be paired
> ({{ch:math-inference}}). But *non*-overlapping bars are strong evidence the
> difference is real, so plotting them is a cheap and honest first filter.

### 5.7 Colour

Three kinds of colormap, and using the wrong one misleads:

- **Sequential** (`viridis`, `magma`) — ordered data from low to high.
- **Diverging** (`RdBu`, `coolwarm`) — data with a meaningful midpoint, such as
  a correlation or a residual. Centre the scale on it.
- **Qualitative** (`tab10`) — unordered categories.

> WARNING: Do not use `jet` or `rainbow`. They are not perceptually uniform:
> equal steps in value produce unequal perceived steps, creating false banding
> and hiding real structure. They also become unreadable in greyscale and for
> readers with the most common forms of colour blindness. `viridis` is the
> default precisely because it is uniform and colour-blind safe.

About 8% of men have some form of colour-vision deficiency. Never encode
information by colour alone — pair it with position, shape, or a direct label.

## 6. Mathematical Foundation

### 6.1 Anscombe's quartet

Four datasets, thirteen points each, constructed so that they share:

$$
\bar{x} = 9.0, \quad \bar{y} \approx 7.50, \quad
s_x^2 = 11.0, \quad s_y^2 \approx 4.13
$$
$$
r \approx 0.816, \quad \hat{y} = 3.00 + 0.500x
$$ (eq:anscombe-stats)

Every summary statistic from {{ch:math-covariance}} agrees to two decimal
places. The datasets are:

1. A genuine linear relationship with noise — the regression is appropriate.
2. A clean **parabola** — the relationship is deterministic and not linear, so
   the linear fit is systematically wrong everywhere.
3. A perfect line with **one outlier** dragging the slope away from it.
4. Eleven points at one x value plus **one distant point** that alone determines
   the slope. Remove it and the slope is undefined.

Only the first justifies reporting a correlation. The other three are cases where
the number is arithmetically correct and substantively meaningless.

This is the same point {{ch:math-covariance}} made with $Y = X^2$ having zero
correlation, seen from the other direction: there, a real relationship gave a
null statistic; here, four different realities give identical statistics.

$$
\text{summary statistics} \;\not\Rightarrow\; \text{the data}
$$ (eq:stats-not-data)

The mapping loses information, and there is no way to recover it except by
looking.

### 6.2 Why truncating a bar axis lies

A bar chart encodes value in **length**. With the axis starting at zero, the
ratio of drawn lengths equals the ratio of values:

$$
\frac{L_1}{L_2} = \frac{v_1}{v_2}
$$ (eq:bar-honest)

Start the axis at $b > 0$ and the drawn ratio becomes

$$
\frac{L_1}{L_2} = \frac{v_1 - b}{v_2 - b}
$$ (eq:bar-truncated)

which can be made arbitrarily large by pushing $b$ toward $\min(v_1, v_2)$.

Concretely: accuracies of 94% and 96% on a zero-based axis differ in length by
about 2%. With the axis starting at 93%, they differ by a factor of three. The
data is unchanged and the visual claim is completely different.

A line or scatter chart does not have this problem, because it encodes value in
**position** rather than length, and position is already relative to whatever
the axis shows.

### 6.3 Overplotting

A scatter plot of $n$ points in a region holding $m$ distinguishable positions
saturates when $n \gg m$. Beyond that, adding points changes nothing visible,
and a dense region and a saturated one look identical — so the plot no longer
shows density.

Three fixes, in increasing order of scale: reduce `alpha` so overlap accumulates
visibly; add jitter for discrete values; or switch to a 2-D histogram
(`hexbin`) which encodes density explicitly rather than relying on overlap.

The crossover is typically a few thousand points for a normal-sized panel.

## 7. Implementation

```python {tier=A name=matplotlib-core}
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
```

## 8. Practical Example

The diagnostic panel below is the figure you will actually build over and over:
several views of one model's behaviour, on shared scales, in one call.

```python {tier=A name=diagnostic-panel}
"""A model diagnostic panel — the plots this book relies on, in one figure.

Each function takes an `ax` and draws into it, which is what makes them
composable and testable. None of them touch global pyplot state.
"""
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(3)
out_dir = Path(tempfile.mkdtemp(prefix="diag-"))

# --- synthetic "training run" and predictions --------------------------------
epochs = np.arange(1, 61)
train_loss = 2.2 * np.exp(-epochs / 10) + 0.05 + rng.normal(0, 0.015, 60)
val_loss = 2.2 * np.exp(-epochs / 11) + 0.16 + rng.normal(0, 0.02, 60)
val_loss[35:] += np.linspace(0, 0.22, 25)          # overfitting sets in

# The classic case: the truth is quadratic, the model fitted is a straight
# line. Ordinary least squares GUARANTEES the residuals are uncorrelated with
# the fitted values, so every summary statistic is blind to the mis-specification
# by construction — but the residual plot shows it immediately.
n = 900
x = rng.uniform(-10, 10, n)
y_true = 30 + 1.5 * x + 0.35 * x ** 2 + rng.normal(0, 4, n)
slope, intercept = np.polyfit(x, y_true, 1)
y_pred = intercept + slope * x

labels = rng.integers(0, 3, n)
preds = np.where(rng.random(n) < 0.78, labels, rng.integers(0, 3, n))


# --- each plot is a function taking an axes ---------------------------------
def plot_learning_curves(ax, epochs, train, val):
    ax.plot(epochs, train, label="train", lw=1.5)
    ax.plot(epochs, val, label="validation", lw=1.5)
    best = int(np.argmin(val))
    ax.axvline(epochs[best], ls="--", c="grey", lw=1)
    ax.annotate(f"best epoch {epochs[best]}", (epochs[best], val[best]),
                textcoords="offset points", xytext=(8, 14), fontsize=8)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.set_title("learning curves")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    return best


def plot_residuals(ax, y_true, y_pred):
    resid = y_true - y_pred
    ax.scatter(y_pred, resid, s=6, alpha=0.25)
    ax.axhline(0, c="crimson", lw=1)
    # A smoothed trend makes systematic structure visible.
    order = np.argsort(y_pred)
    k = 60
    smooth = np.convolve(resid[order], np.ones(k) / k, mode="valid")
    ax.plot(y_pred[order][k//2: k//2 + len(smooth)], smooth,
            c="crimson", lw=1.6, label="local mean")
    ax.set_xlabel("predicted"); ax.set_ylabel("residual")
    ax.set_title("residuals vs fitted")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)


def plot_distribution(ax, values, label):
    ax.hist(values, bins=40, alpha=0.75, edgecolor="white", linewidth=0.4)
    ax.axvline(np.mean(values), c="crimson", lw=1.4, label=f"mean {np.mean(values):.1f}")
    ax.axvline(np.median(values), c="darkorange", lw=1.4, ls="--",
               label=f"median {np.median(values):.1f}")
    ax.set_xlabel(label); ax.set_ylabel("count")
    ax.set_title(f"distribution of {label}")
    ax.legend(fontsize=8)


def plot_confusion(ax, labels, preds, classes=("A", "B", "C")):
    k = len(classes)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    normed = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(normed, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(k), classes); ax.set_yticks(range(k), classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("confusion matrix (row-normalised)")
    for i in range(k):
        for j in range(k):
            # Contrast-aware text colour: never encode by colour alone.
            ax.text(j, i, f"{normed[i,j]:.2f}", ha="center", va="center",
                    fontsize=8,
                    color="white" if normed[i, j] > 0.5 else "black")
    return im, cm


# --- compose them ------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
best_epoch = plot_learning_curves(axes[0, 0], epochs, train_loss, val_loss)
plot_residuals(axes[0, 1], y_true, y_pred)
plot_distribution(axes[1, 0], y_true, "target")
im, cm = plot_confusion(axes[1, 1], labels, preds)
fig.colorbar(im, ax=axes[1, 1], fraction=0.046, label="fraction of true class")
fig.suptitle("model diagnostics", fontsize=13)
fig.tight_layout()
path = out_dir / "diagnostics.png"
fig.savefig(path, dpi=120, bbox_inches="tight")
plt.close(fig)

print(f"wrote {path.name} ({path.stat().st_size/1024:.0f} KB)")

# --- what each panel actually told us ----------------------------------------
print(f"\nlearning curves : validation loss bottoms at epoch "
      f"{epochs[best_epoch]} then rises")
print(f"                  train {train_loss[best_epoch]:.3f} vs "
      f"val {val_loss[best_epoch]:.3f} at that point; by epoch 60 the gap is "
      f"{val_loss[-1] - train_loss[-1]:.3f}")
print("                  -> classic overfitting; stop early (Chapter 58)")

resid = y_true - y_pred
edges = np.linspace(y_pred.min(), y_pred.max(), 7)
binned = [resid[(y_pred >= edges[i]) & (y_pred < edges[i + 1])].mean()
          for i in range(6)]
print(f"\nresiduals       : mean residual across six bins of fitted value:")
print(f"                  {np.round(binned, 1).tolist()}")
print("                  -> a clear U. High at both ends, negative in the")
print("                     middle: the model has missed a curvature.")
print(f"\n                  and yet corr(residual, fitted) = "
      f"{np.corrcoef(y_pred, resid)[0, 1]:+.4f}")
print("                  Least squares GUARANTEES that correlation is zero,")
print("                  so no summary statistic of the residuals could ever")
print("                  reveal this. Only the plot does (Chapter 32).")

print(f"\nconfusion       : per-class recall "
      f"{np.round(np.diag(cm) / cm.sum(axis=1), 3).tolist()}")
print(f"                  overall accuracy {np.trace(cm)/cm.sum():.3f}")

import shutil
shutil.rmtree(out_dir, ignore_errors=True)
print(f"\ncleaned up. open figures: {len(plt.get_fignums())}")
```

## 9. Common Mistakes

**Using the `pyplot` state machine in reusable code.** Pass an `ax`.

**Not closing figures in a loop.** `pyplot` holds a reference to each one; a
per-epoch plot leaks memory across a long run.

**Forgetting `matplotlib.use("Agg")` before importing pyplot.** Scripts fail on
headless machines, and the error is confusing.

**Truncating a bar chart's axis.** Exaggerates differences by
{{eq:bar-truncated}}.

**Using `jet` or `rainbow`.** Not perceptually uniform; invents structure.

**Encoding information by colour alone.** Fails for colour-blind readers and in
greyscale.

**Comparing panels on different scales.** Use `sharex`/`sharey`.

**Plotting only summary statistics.** Anscombe's quartet.

**Overplotting without alpha or binning.** Beyond a few thousand points, the
plot stops showing density.

**Using a linear axis for data spanning orders of magnitude.** Loss curves,
latencies and learning rates all want log.

**Pie charts with more than three slices.** Angles are hard to compare; a sorted
bar chart is strictly better.

## 10. Connection to Previous Chapters

{{ch:py-pandas}} and {{ch:py-numpy}} supply the data structures Matplotlib
consumes. {{ch:math-covariance}} supplies the summary statistics that Anscombe's
quartet defeats, and this chapter is the visual counterpart of that chapter's
$Y = X^{2}$ counterexample. {{ch:math-functions}} explains why log scales are
right for data spanning orders of magnitude.

Forward: {{ch:ds-eda}} makes visualisation the primary tool of exploratory
analysis; {{ch:ml-metrics}} uses learning curves and confusion matrices as
standard diagnostics; {{ch:ds-causation}} uses plots to expose the confounding
that correlations hide; and {{ch:ops-observability}} extends these ideas to
monitoring dashboards.

{{cite:hunter2007}} introduced Matplotlib.

## 11. Exercises

**Beginner**

1. Create a figure with one axes and plot $y = x^{2}$ for $x \in [-5, 5]$, with
   labelled axes and a title.
2. Make a 2×2 grid of subplots with shared axes.
3. Plot a histogram of 1,000 normal samples with 30 bins.
4. Save a figure as both PNG and SVG. When would you prefer each?
5. Set a y-axis to log scale and describe what changes.

**Intermediate**

6. Reproduce Anscombe's quartet and verify the statistics agree to two decimals.
7. Build the truncated-axis comparison and compute the exaggeration factor with
   {{eq:bar-truncated}}.
8. Plot 100,000 points three ways — plain, alpha, hexbin — and say at what
   count plain scatter stops being informative.
9. Write `plot_learning_curves(ax, ...)` that takes an axes, and call it twice
   into different panels of one figure.
10. Make a diverging heatmap of a correlation matrix, centred correctly at zero.
11. Add value labels to a bar chart, positioned readably regardless of bar
    height.

**Advanced**

12. Measure the perceptual non-uniformity of `jet` against `viridis` by
    computing lightness steps, and quantify the worst-case ratio.
13. Write a function detecting likely overplotting from the data and panel size,
    and recommending alpha or hexbin.
14. Build a figure with an inset zoom on a region of interest.
15. Simulate the three common forms of colour blindness on a figure and check
    that yours survives all of them.

**Implementation**

16. Build a reusable `diagnostics(fig, y_true, y_pred)` producing a four-panel
    report, with each panel a separate testable function.
17. Write a plotting function and a test asserting properties of the returned
    artists — number of lines, axis labels, limits — without rendering.
18. Create an animation of gradient descent on a 2-D surface
    ({{ch:math-optimization}}), saved as frames.
19. Take a chart you have made and produce a deliberately misleading version,
    then document every technique you used.

**Reasoning**

20. Summary statistics are compact and lossy; plots are rich and subjective.
    When is each the right thing to report?
21. Should truncated axes ever be used? Construct a case for the defence.

## 12. Chapter Summary

A Figure is the canvas; an Axes is one plot on it. Use the object-oriented
interface — `fig, ax = plt.subplots()`, then methods on `ax` — because the
`pyplot` state machine relies on a global "current figure" that makes plotting
functions non-composable.

Chart type follows from the question. Bar charts encode value in length and must
start at zero; truncating the axis changes the drawn ratio from $v_1/v_2$ to
$(v_1 - b)/(v_2 - b)$, which can exaggerate a small difference several-fold
without changing a single number. Line and scatter charts encode position and
are not subject to this.

Anscombe's quartet shows four datasets with identical means, variances,
correlations and regression lines that look nothing alike: a real linear
relationship, a parabola, a line with an outlier, and a case where one point
determines the slope. Summary statistics do not determine the data, and only
looking recovers the difference.

Beyond a few thousand points a scatter plot saturates and stops showing density.
Reduce alpha, jitter, or switch to hexbin.

Log scales are correct whenever data spans orders of magnitude, which covers
loss curves, latencies, learning rates and model sizes.

Colormaps come in sequential, diverging and qualitative kinds, and using the
wrong one misleads. Avoid `jet`: it is not perceptually uniform, so it creates
banding the data does not contain and fails in greyscale and for colour-blind
readers.

For headless execution set the `Agg` backend before importing `pyplot`, and
close figures explicitly in loops — `pyplot` retains every figure it creates,
which is a real memory leak in a training run that logs a plot per epoch.
