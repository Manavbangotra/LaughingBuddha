# -*- coding: utf-8 -*-
# Extracted from: Chapter 1 — Mathematical Notation and the Language of Machine Learning
# Source: src/.../ch001-notation.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Mathematical notation and its NumPy equivalents, with the identities of
section 5.2 checked numerically rather than asserted.
"""
import numpy as np

# --- summation: sum_{i=1}^{5} i^2 ------------------------------------------
print("sum i^2, i=1..5 :", sum(i**2 for i in range(1, 6)),
      "| numpy:", int((np.arange(1, 6) ** 2).sum()))

# --- product: prod_{i=1}^{5} i  (= 5!) -------------------------------------
print("prod i, i=1..5  :", int(np.prod(np.arange(1, 6))))

# --- the empty-sum and empty-product conventions ---------------------------
# A sum with no terms is 0; a product with no terms is 1. NumPy agrees, and it
# matters: it is why an unnormalised probability of "no evidence" is 1, not 0.
print("empty sum       :", np.sum(np.array([])),
      "| empty product:", np.prod(np.array([])))

# --- conditional sum: sum over i where y_i == 1 ----------------------------
x = np.array([10.0, 20.0, 30.0, 40.0])
y = np.array([1, 0, 1, 1])
print("sum x_i where y_i==1 :", x[y == 1].sum())

# --- indicator functions and eq. 1.10 --------------------------------------
y_true = np.array([1, 0, 1, 1, 0])
y_pred = np.array([1, 0, 0, 1, 0])
indicator = (y_pred == y_true).astype(float)     # 1[y_hat == y]
print("indicators      :", indicator, "| accuracy:", indicator.mean())

# --- nested sums, and eq. 1.7 (order does not matter for finite sums) -------
A = np.arange(12).reshape(3, 4)
print("sum over i then j:", A.sum(axis=0).sum(),
      "| j then i:", A.sum(axis=1).sum(),
      "| all at once:", A.sum())
assert A.sum(axis=0).sum() == A.sum(axis=1).sum() == A.sum()

# --- eq. 1.6: linearity of summation ---------------------------------------
f = np.array([1.0, 2.0, 3.0])
g = np.array([10.0, 20.0, 30.0])
c = 2.5
assert np.isclose((c * f).sum(), c * f.sum())
assert np.isclose((f + g).sum(), f.sum() + g.sum())
print("linearity of summation verified")

# --- max vs argmax ----------------------------------------------------------
scores = np.array([0.1, 0.7, 0.2])
print("max  :", scores.max(), "(the value)")
print("argmax:", scores.argmax(), "(the index) — a classifier returns this")

# --- eq. 1.1 on the worked example of section 6.2 ---------------------------
p_true = np.array([0.9, 0.6, 0.2])
loss = -np.mean(np.log(p_true))
print(f"\ncross-entropy loss: {loss:.4f}")
assert np.isclose(loss, 0.7419, atol=1e-4)
print("per-example contributions:", np.round(-np.log(p_true), 4))
print(f"the near-miss contributes {np.log(0.2) / np.log(0.9):.1f}x as much "
      f"as the confident-correct example")
