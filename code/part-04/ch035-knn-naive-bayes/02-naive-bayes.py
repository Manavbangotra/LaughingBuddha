# -*- coding: utf-8 -*-
# Extracted from: Chapter 35 — k-Nearest Neighbors and Naive Bayes
# Source: src/.../ch035-knn-naive-bayes.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Naive Bayes: derivation, smoothing, calibration failure, and the
generative/discriminative crossover.
"""
import numpy as np

rng = np.random.default_rng(3)


class MultinomialNB:
    """Multinomial naive Bayes with add-alpha smoothing (eq. 35.11)."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes = np.unique(y)
        V = X.shape[1]
        self.log_prior = np.array(
            [np.log((y == c).mean()) for c in self.classes])
        self.log_lik = np.empty((len(self.classes), V))
        for i, c in enumerate(self.classes):
            counts = X[y == c].sum(axis=0) + self.alpha
            self.log_lik[i] = np.log(counts / counts.sum())
        return self

    def log_joint(self, X):
        """log P(c) + sum_j x_j log P(w_j | c)  — eq. 35.10, in logs because
        a product of thousands of probabilities underflows."""
        return X @ self.log_lik.T + self.log_prior

    def predict(self, X):
        return self.classes[self.log_joint(X).argmax(1)]

    def predict_proba(self, X):
        lj = self.log_joint(X)
        lj = lj - lj.max(1, keepdims=True)          # the softmax shift again
        e = np.exp(lj)
        return e / e.sum(1, keepdims=True)


# --- a synthetic bag-of-words corpus ----------------------------------------
# Deliberately HARD: short documents and a weak vocabulary contrast, so the
# task lands around 80% and there is room for the failure modes to show.
V, n_docs, CONTRAST = 200, 4500, 1.8
w1 = np.ones(V); w1[:40] = CONTRAST      # class 1 favours words 0-39
w0 = np.ones(V); w0[40:80] = CONTRAST    # class 0 favours words 40-79
w1, w0 = w1 / w1.sum(), w0 / w0.sum()    # the other 120 words are shared

y = (rng.random(n_docs) < 0.5).astype(int)
lengths = rng.poisson(20, n_docs) + 3
X = np.array([rng.multinomial(L, w1 if lab else w0)
              for L, lab in zip(lengths, y)])

cut = 3000
Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]

nb = MultinomialNB(alpha=1.0).fit(Xtr, ytr)
print(f"naive Bayes test accuracy: {(nb.predict(Xte) == yte).mean():.4f}")
print(f"(chance is {max(yte.mean(), 1 - yte.mean()):.4f})")

# --- section 5.3: what happens without smoothing ----------------------------
print("\n" + "=" * 72)
print("Laplace smoothing is not optional (eq. 35.11)")
print("=" * 72)
print("Unseen words only occur when the training set is small — which is")
print("exactly when naive Bayes is the model you reached for.\n")
print(f"{'train docs':>11} {'unseen words':>14} " +
      " ".join(f"{'a=' + str(a):>8}"
               for a in (0.0, 1e-10, 0.01, 0.1, 1.0, 10.0, 100.0)))
for n_small in (40, 80, 200, 3000):
    A, b = X[:n_small], y[:n_small]
    unseen = max((A[b == c].sum(0) == 0).sum() for c in (0, 1))
    accs = []
    for alpha in (0.0, 1e-10, 0.01, 0.1, 1.0, 10.0, 100.0):
        with np.errstate(divide="ignore", invalid="ignore"):
            accs.append((MultinomialNB(alpha).fit(A, b).predict(Xte)
                         == yte).mean())
    print(f"{n_small:>11} {unseen:>14} " + " ".join(f"{a:>8.3f}" for a in accs))

print("\nWith 40 training documents about 29 of the 200 words are unseen in")
print("some class, and alpha=0 scores 0.52 — chance. Each unseen word")
print("contributes log(0) = -inf and vetoes that class outright, whatever")
print("the other 199 words say. Any positive alpha removes the veto and")
print("recovers 17 accuracy points. Too large an alpha (100) washes the")
print("evidence out towards the prior. Once the training set is large")
print("enough that nothing is unseen, alpha stops mattering — which is why")
print("this bug hides until the day you deploy on a rare class.")

# --- section 6.3: good classifier, terrible probabilities -------------------
print("\n" + "=" * 72)
print("correlated features destroy the probabilities, not the decisions")
print("=" * 72)


def ece(y_true, p, n_bins=10):
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    tot = 0.0
    for i in range(n_bins):
        m = (p >= edges[i]) & (p <= edges[i + 1])
        if m.sum():
            tot += m.sum() / len(p) * abs(y_true[m].mean() - p[m].mean())
    return tot


def roc_auc(y_true, s):
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), float)
    r[o] = np.arange(1, len(s) + 1)
    npos, nneg = int(y_true.sum()), int((1 - y_true).sum())
    return (r[y_true == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


print(f"{'duplicates of each word':>24} {'accuracy':>10} {'ROC-AUC':>9} "
      f"{'ECE':>8} {'mean max prob':>15}")
for m_dup in (1, 2, 4, 8):
    Xd_tr = np.tile(Xtr, (1, m_dup))
    Xd_te = np.tile(Xte, (1, m_dup))
    nbd = MultinomialNB(1.0).fit(Xd_tr, ytr)
    P = nbd.predict_proba(Xd_te)
    pred = nbd.predict(Xd_te)
    print(f"{m_dup:>24} {(pred == yte).mean():>10.4f} "
          f"{roc_auc(yte, P[:, 1]):>9.4f} {ece(yte, P[:, 1]):>8.4f} "
          f"{P.max(1).mean():>15.6f}")

print("\nDuplicating every feature adds exactly zero information. Accuracy")
print("moves by less than 0.3 points and ROC-AUC is IDENTICAL to four")
print("decimal places — the ranking cannot change, because the log-odds are")
print("merely multiplied by m, a monotone map (section 6.3). Calibration")
print("collapses: mean confidence climbs from 0.83 to 0.98 while accuracy")
print("stays at 0.81, and ECE goes up almost tenfold. This is the mechanism")
print("behind naive Bayes' reputation for returning 0.99999, and it is why")
print("you recalibrate before using the number for anything.")

# --- section 5.4: the generative/discriminative crossover -------------------
print("\n" + "=" * 72)
print("generative vs discriminative: who wins depends on how much data")
print("=" * 72)
print("The corpus is augmented with 30 near-duplicate word pairs, so naive")
print("Bayes' independence assumption is genuinely FALSE here — otherwise")
print("it would be the true model and could never be overtaken.\n")

Xc = np.hstack([X, X[:, :30] + rng.binomial(X[:, :30], 0.85)])
Xc_te, yc_te = Xc[cut:], y[cut:]


def fit_logistic(A, b, lam=0.01, n_iter=100):
    A1 = np.column_stack([np.ones(len(A)), A])
    w = np.zeros(A1.shape[1])
    for _ in range(n_iter):
        p = 1 / (1 + np.exp(-np.clip(A1 @ w, -30, 30)))
        g = A1.T @ (p - b) / len(b)
        g[1:] += 2 * lam * w[1:]
        S = np.maximum(p * (1 - p), 1e-7)
        H = (A1 * S[:, None]).T @ A1 / len(b) + (2 * lam + 1e-6) * np.eye(len(w))
        w -= np.linalg.solve(H, g)
    return w


def score_logistic(w, A, b):
    A1 = np.column_stack([np.ones(len(A)), A])
    return float((((A1 @ w) >= 0).astype(int) == b).mean())


print(f"{'train size':>11} {'naive Bayes':>13} {'logistic':>10} "
      f"{'winner':>13} {'margin':>9}")
for n_train in (6, 12, 25, 50, 100, 300, 1000):
    # average over many disjoint training samples of this size — a single
    # draw at n=6 says nothing at all
    starts = range(0, min(2400, cut - n_train) + 1, 150)
    nb_acc = np.mean([
        (MultinomialNB(1.0).fit(Xc[s:s + n_train], y[s:s + n_train])
         .predict(Xc_te) == yc_te).mean() for s in starts])
    lr_acc = np.mean([
        score_logistic(fit_logistic(np.log1p(Xc[s:s + n_train]),
                                    y[s:s + n_train].astype(float)),
                       np.log1p(Xc_te), yc_te) for s in starts])
    winner = "naive Bayes" if nb_acc > lr_acc else "logistic"
    print(f"{n_train:>11} {nb_acc:>13.4f} {lr_acc:>10.4f} "
          f"{winner:>13} {abs(nb_acc - lr_acc):>9.4f}")

print("\nThe crossover is real and lands between 25 and 50 documents here.")
print("Below it the generative model's assumptions substitute for data it")
print("does not have — with 6 documents and 230 features there is nothing")
print("to estimate a boundary from, and naive Bayes leads by 3.2 points.")
print("Above it logistic regression is ahead at every size, though only by")
print("a few tenths of a point: the independence violation planted here is")
print("mild, so naive Bayes' ceiling is only slightly below the truth. The")
print("honest summary is that the ORDERING flips reliably and the MARGIN")
print("above the crossover is small — which is why 'naive Bayes for small")
print("data' is sound advice and 'logistic regression is better' is not")
print("worth much without knowing how false the independence assumption is")
print("on your data (Table 35.2).")
