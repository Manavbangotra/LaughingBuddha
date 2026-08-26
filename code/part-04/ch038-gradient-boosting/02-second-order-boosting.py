# -*- coding: utf-8 -*-
# Extracted from: Chapter 38 — Gradient Boosting: Theory, XGBoost, LightGBM, CatBoost
# Source: src/.../ch038-gradient-boosting.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Log-loss boosting, and the second-order (XGBoost) formulation derived in
section 6.2 — including the exact split criterion.
"""
import numpy as np

rng = np.random.default_rng(2)


def sigmoid(z):
    out = np.empty_like(z, dtype=float)
    p = z >= 0
    out[p] = 1 / (1 + np.exp(-z[p]))
    e = np.exp(z[~p])
    out[~p] = e / (1 + e)
    return out


# --- a tree grown to maximise the XGBoost gain of eq. 38.9 ------------------
def xgb_grow(X, g, h, depth, max_depth, lam, gamma, min_child_weight=1.0):
    """Split scoring comes from eq. 38.9; leaf values from eq. 38.7. The tree
    is grown against the ACTUAL loss, not a Gini proxy."""
    G, H = g.sum(), h.sum()
    node = {"w": -G / (H + lam)}
    if depth >= max_depth or len(g) < 2:
        return node
    best = (0.0, None, None)
    base = G ** 2 / (H + lam)
    for j in range(X.shape[1]):
        o = np.argsort(X[:, j], kind="mergesort")
        xs, gs, hs = X[o, j], g[o], h[o]
        cg, ch = np.cumsum(gs), np.cumsum(hs)
        k = np.arange(1, len(gs))
        GL, HL = cg[:-1], ch[:-1]
        GR, HR = G - GL, H - HL
        ok = ((xs[1:] != xs[:-1]) & (HL >= min_child_weight)
              & (HR >= min_child_weight))
        gain = np.where(ok,
                        0.5 * (GL ** 2 / (HL + lam) + GR ** 2 / (HR + lam)
                               - base) - gamma,
                        -np.inf)
        i = int(gain.argmax())
        if np.isfinite(gain[i]) and gain[i] > best[0]:
            best = (float(gain[i]), j, 0.5 * (xs[i] + xs[i + 1]))
    _, j, thr = best
    if j is None:
        return node
    m = X[:, j] <= thr
    node["f"], node["t"] = j, thr
    node["l"] = xgb_grow(X[m], g[m], h[m], depth + 1, max_depth, lam, gamma,
                         min_child_weight)
    node["r"] = xgb_grow(X[~m], g[~m], h[~m], depth + 1, max_depth, lam,
                         gamma, min_child_weight)
    return node


def apply_xgb(node, X):
    out = np.empty(len(X))

    def walk(nd, idx):
        if "f" not in nd:
            out[idx] = nd["w"]
            return
        m = X[idx, nd["f"]] <= nd["t"]
        walk(nd["l"], idx[m])
        walk(nd["r"], idx[~m])

    walk(node, np.arange(len(X)))
    return out


class XGBClassifier:
    """Second-order boosting for log loss. For log loss:
         g = p - y            (the gradient of eq. 33.5, i.e. eq. 33.10)
         h = p(1 - p)         (its second derivative, the sigmoid slope)
    """

    def __init__(self, n_trees=300, eta=0.1, max_depth=3, lam=1.0, gamma=0.0,
                 min_child_weight=1.0):
        self.M, self.eta, self.depth = n_trees, eta, max_depth
        self.lam, self.gamma, self.mcw = lam, gamma, min_child_weight

    def fit(self, X, y, X_val=None, y_val=None):
        base = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        self.F0 = float(np.log(base / (1 - base)))     # log-odds of the prior
        F = np.full(len(y), self.F0)
        self.trees, self.val_curve, self.train_curve = [], [], []
        for _ in range(self.M):
            p = sigmoid(F)
            g, h = p - y, np.maximum(p * (1 - p), 1e-9)
            t = xgb_grow(X, g, h, 0, self.depth, self.lam, self.gamma, self.mcw)
            self.trees.append(t)
            F += self.eta * apply_xgb(t, X)
            self.train_curve.append(self._logloss(y, sigmoid(F)))
            if X_val is not None:
                self.val_curve.append(
                    self._logloss(y_val, self.predict_proba(X_val)))
        return self

    @staticmethod
    def _logloss(y, p):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def decision(self, X, n_trees=None):
        M = len(self.trees) if n_trees is None else n_trees
        F = np.full(len(X), self.F0)
        for t in self.trees[:M]:
            F += self.eta * apply_xgb(t, X)
        return F

    def predict_proba(self, X, n_trees=None):
        return sigmoid(self.decision(X, n_trees))


def make_clf(n, rs):
    X = rs.uniform(-3, 3, (n, 8))
    z = (1.4 * np.sin(1.2 * X[:, 0]) + 0.9 * X[:, 1]
         - 1.1 * X[:, 0] * X[:, 2] + 0.7 * np.abs(X[:, 3]) - 0.8)
    return X, (rs.random(n) < sigmoid(z)).astype(float)


rs = np.random.default_rng(3)
Xtr, ytr = make_clf(900, rs)
Xva, yva = make_clf(900, rs)
Xte, yte = make_clf(3000, rs)
print(f"positive rate: {ytr.mean():.4f}")


def roc_auc(y, s):
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    npos = int(y.sum())
    return float((r[y == 1].sum() - npos * (npos + 1) / 2)
                 / (npos * (len(y) - npos)))


# --- the pseudo-residual for log loss IS the logistic gradient --------------
print("\n" + "=" * 72)
print("table 38.1: the log-loss pseudo-residual is y - sigma(F)")
print("=" * 72)
m0 = XGBClassifier(n_trees=1, eta=0.1).fit(Xtr, ytr)
F0 = np.full(len(ytr), m0.F0)
print(f"initial F0 = log-odds of the base rate = {m0.F0:.4f}")
print(f"sigmoid(F0) = {sigmoid(np.array([m0.F0]))[0]:.4f}  "
      f"vs base rate {ytr.mean():.4f}")
print(f"mean pseudo-residual at round 1 = "
      f"{np.mean(ytr - sigmoid(F0)):.2e}  (zero by construction)")
print("This is exactly eq. 33.10 from Chapter 33. Boosting for")
print("classification is logistic regression's gradient, fitted by trees.")

# --- early stopping ---------------------------------------------------------
print("\n" + "=" * 72)
print("early stopping is how n_estimators is chosen")
print("=" * 72)
clf = XGBClassifier(n_trees=350, eta=0.1, max_depth=4,
                    lam=1.0).fit(Xtr, ytr, Xva, yva)
print(f"{'round':>7} {'train log loss':>16} {'val log loss':>14} "
      f"{'test AUC':>10}")
for m in (1, 10, 50, 100, 200, 350):
    print(f"{m:>7} {clf.train_curve[m - 1]:>16.4f} "
          f"{clf.val_curve[m - 1]:>14.4f} "
          f"{roc_auc(yte, clf.predict_proba(Xte, m)):>10.4f}")
best = int(np.argmin(clf.val_curve)) + 1
print(f"\nvalidation minimum at round {best}: "
      f"log loss {min(clf.val_curve):.4f}, "
      f"test AUC {roc_auc(yte, clf.predict_proba(Xte, best)):.4f}")
print(f"at 350 rounds:              log loss {clf.val_curve[-1]:.4f}, "
      f"test AUC {roc_auc(yte, clf.predict_proba(Xte)):.4f}")
print("\nNote which metric degrades. Log loss turns up clearly while AUC")
print("barely moves — the extra rounds are making the model OVERCONFIDENT")
print("rather than changing its ranking, which is Chapter 34's")
print("calibration-versus-discrimination split appearing again.")

# --- section 6.2: what lambda and gamma actually do -------------------------
print("\n" + "=" * 72)
print("the regularisers in eq. 38.9, one at a time")
print("=" * 72)
print(f"{'lambda':>8} {'gamma':>7} {'best round':>12} {'best val loss':>15} "
      f"{'test AUC':>10}")
for lam, gam in ((0.0, 0.0), (1.0, 0.0), (10.0, 0.0), (100.0, 0.0),
                 (1.0, 0.5), (1.0, 5.0)):
    c = XGBClassifier(n_trees=200, eta=0.1, max_depth=4, lam=lam,
                      gamma=gam).fit(Xtr, ytr, Xva, yva)
    b = int(np.argmin(c.val_curve)) + 1
    print(f"{lam:>8} {gam:>7} {b:>12} {min(c.val_curve):>15.4f} "
          f"{roc_auc(yte, c.predict_proba(Xte, b)):>10.4f}")
print("\nlambda shrinks every leaf value (eq. 38.7); gamma charges a fixed")
print("toll per split (eq. 38.9), so a split must earn its place. They are")
print("different levers: lambda softens the model, gamma makes it smaller.")

# --- first-order vs second-order --------------------------------------------
print("\n" + "=" * 72)
print("first-order vs second-order boosting (section 5.3)")
print("=" * 72)


class FirstOrderClassifier(XGBClassifier):
    """Same loop, but each leaf takes the MEAN NEGATIVE GRADIENT — the plain
    Friedman step, with no Hessian and no eq. 38.7 re-optimisation."""

    def fit(self, X, y, X_val=None, y_val=None):
        base = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        self.F0 = float(np.log(base / (1 - base)))
        F = np.full(len(y), self.F0)
        self.trees, self.val_curve, self.train_curve = [], [], []
        for _ in range(self.M):
            p = sigmoid(F)
            g = p - y
            t = xgb_grow(X, g, np.ones(len(g)), 0, self.depth, self.lam,
                         self.gamma, self.mcw)
            self.trees.append(t)
            F += self.eta * apply_xgb(t, X)
            self.train_curve.append(self._logloss(y, sigmoid(F)))
            if X_val is not None:
                self.val_curve.append(
                    self._logloss(y_val, self.predict_proba(X_val)))
        return self


print(f"{'method':<22} {'best round':>12} {'best val loss':>15} "
      f"{'test AUC':>10}")
for name, cls in (("first-order (Friedman)", FirstOrderClassifier),
                  ("second-order (XGBoost)", XGBClassifier)):
    c = cls(n_trees=250, eta=0.1, max_depth=4, lam=1.0).fit(Xtr, ytr, Xva, yva)
    b = int(np.argmin(c.val_curve)) + 1
    print(f"{name:<22} {b:>12} {min(c.val_curve):>15.4f} "
          f"{roc_auc(yte, c.predict_proba(Xte, b)):>10.4f}")
print("\nBoth use the same trees and the same learning rate. The difference")
print("is that the second-order version divides each leaf by its Hessian")
print("mass (eq. 38.7), so a leaf full of confidently-classified points —")
print("where h = p(1-p) is tiny — is not allowed to make a large correction.")
print("It is a better-scaled step, which is why it needs fewer rounds.")
