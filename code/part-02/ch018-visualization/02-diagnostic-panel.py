# -*- coding: utf-8 -*-
# Extracted from: Chapter 18 — Visualization with Matplotlib
# Source: src/.../ch018-visualization.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

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
