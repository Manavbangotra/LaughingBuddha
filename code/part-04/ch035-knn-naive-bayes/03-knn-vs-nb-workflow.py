# Extracted from: Chapter 35 — k-Nearest Neighbors and Naive Bayes
# Source: src/.../ch035-knn-naive-bayes.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Choosing between them on a document-classification problem, and fixing
naive Bayes' probabilities without touching its decisions.
"""
import numpy as np

rng = np.random.default_rng(9)

# --- a 3-class corpus, deliberately hard ------------------------------------
V, K, n = 300, 3, 7500
topic_w = np.ones((K, V))
for c in range(K):
    topic_w[c, c * 50:(c + 1) * 50] = 1.9      # a weak, realistic contrast
topic_w /= topic_w.sum(1, keepdims=True)

y = rng.integers(0, K, n)
X = np.array([rng.multinomial(L, topic_w[c])
              for L, c in zip(rng.poisson(22, n) + 4, y)])
# near-duplicate word pairs: real corpora are full of them ("cannot"/"can't",
# "NYC"/"New York"), and they are what breaks the independence assumption
X = np.hstack([X, X[:, :60] + rng.binomial(X[:, :60], 0.85)])

n_tr, n_va = 3000, 1500
Xtr, ytr = X[:n_tr], y[:n_tr]
Xva, yva = X[n_tr:n_tr + n_va], y[n_tr:n_tr + n_va]
Xte, yte = X[n_tr + n_va:], y[n_tr + n_va:]


class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.log_prior = np.array([np.log((y == c).mean())
                                   for c in self.classes])
        self.log_lik = np.empty((len(self.classes), X.shape[1]))
        for i, c in enumerate(self.classes):
            ct = X[y == c].sum(0) + self.alpha
            self.log_lik[i] = np.log(ct / ct.sum())
        return self

    def decision(self, X):
        return X @ self.log_lik.T + self.log_prior

    def predict_proba(self, X):
        d = self.decision(X)
        d = d - d.max(1, keepdims=True)
        e = np.exp(d)
        return e / e.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes[self.decision(X).argmax(1)]


def knn_cosine(Xtr, ytr, Xq, k):
    A = Xq / np.maximum(np.linalg.norm(Xq, axis=1, keepdims=True), 1e-12)
    B = Xtr / np.maximum(np.linalg.norm(Xtr, axis=1, keepdims=True), 1e-12)
    sim = A @ B.T
    idx = np.argpartition(-sim, k, axis=1)[:, :k]
    lab = ytr[idx]
    return np.array([np.bincount(r, minlength=3).argmax() for r in lab])


# --- tune both on the VALIDATION set, never the test set --------------------
print("=" * 72)
print("tuning on validation (Chapter 34: the test set is touched once)")
print("=" * 72)
print(f"{'k (cosine k-NN)':>17} {'val accuracy':>14}")
best_k, best = None, -1
for k in (1, 3, 5, 11, 25, 51, 101, 201):
    a = (knn_cosine(Xtr, ytr, Xva, k) == yva).mean()
    print(f"{k:>17} {a:>14.4f}")
    if a > best:
        best_k, best = k, a

print(f"\n{'alpha (naive Bayes)':>19} {'val accuracy':>14}")
best_a, best_nb = None, -1
for alpha in (0.01, 0.1, 0.5, 1.0, 5.0):
    a = (MultinomialNB(alpha).fit(Xtr, ytr).predict(Xva) == yva).mean()
    print(f"{alpha:>19} {a:>14.4f}")
    if a > best_nb:
        best_a, best_nb = alpha, a

print(f"\nchosen: k={best_k}, alpha={best_a}")

# --- one look at the test set -----------------------------------------------
nb = MultinomialNB(best_a).fit(np.vstack([Xtr, Xva]), np.r_[ytr, yva])
nb_pred = nb.predict(Xte)
knn_pred = knn_cosine(np.vstack([Xtr, Xva]), np.r_[ytr, yva], Xte, best_k)
D = X.shape[1]
n_fit = n_tr + n_va
print(f"\n{'model':<24} {'test accuracy':>14} {'ops per query':>16}")
print(f"{'naive Bayes':<24} {(nb_pred == yte).mean():>14.4f} "
      f"{K * D:>16,}")
print(f"{'cosine k-NN':<24} {(knn_pred == yte).mean():>14.4f} "
      f"{n_fit * D:>16,}")
print(f"{'chance':<24} {max(np.bincount(yte)) / len(yte):>14.4f} {'-':>16}")

print("\nNaive Bayes wins on both axes here, and the k-NN result is the more")
print("instructive one. These are 360-dimensional count vectors with about")
print("26 non-zero entries each: two documents on the same topic share")
print("almost no words by chance, so cosine distance is dominated by which")
print("particular words happened to be sampled. That is section 6.2's curse")
print("arriving in a realistic setting — and it is precisely why Part XI")
print("learns a DENSE low-dimensional representation before doing k-NN in")
print("it, rather than running k-NN on raw counts.")
print("\nThe cost asymmetry is the other half of the decision: naive Bayes")
print("compresses the training set into K x D numbers, so its prediction")
print("cost does not depend on N at all, while k-NN keeps every document")
print("and pays for it on every query.")

# --- naive Bayes' probabilities, and repairing them -------------------------
print("\n" + "=" * 72)
print("naive Bayes' probabilities are unusable, and cheap to repair")
print("=" * 72)
P_va = MultinomialNB(best_a).fit(Xtr, ytr).predict_proba(Xva)
P_te = MultinomialNB(best_a).fit(Xtr, ytr).predict_proba(Xte)
conf_te = P_te.max(1)
correct_te = (MultinomialNB(best_a).fit(Xtr, ytr).predict(Xte) == yte)
print(f"mean predicted confidence : {conf_te.mean():.6f}")
print(f"actual accuracy           : {correct_te.mean():.6f}")
print(f"fraction with p > 0.999   : {(conf_te > 0.999).mean():.4f}")
print(f"overconfidence            : {conf_te.mean() - correct_te.mean():+.4f}")
print("The model claims 84% confidence and is right 71% of the time — a")
print("13-point gap, with 9% of documents rated above 0.999. Anything that")
print("consumes these numbers as probabilities is being lied to.")


def temperature_fit(logits, y_true, grid=np.logspace(-2.5, 0.5, 80)):
    """Divide the log-joint by T and pick T by validation log loss.

    A single scalar: it cannot reorder anything, so accuracy and AUC are
    mathematically unchanged (section 6.3).
    """
    best_T, best_ll = 1.0, np.inf
    for T in grid:
        d = logits / T
        d = d - d.max(1, keepdims=True)
        p = np.exp(d)
        p /= p.sum(1, keepdims=True)
        ll = -np.mean(np.log(np.clip(p[np.arange(len(y_true)), y_true],
                                     1e-12, 1)))
        if ll < best_ll:
            best_T, best_ll = T, ll
    return best_T


m = MultinomialNB(best_a).fit(Xtr, ytr)
T = temperature_fit(m.decision(Xva), yva)
d = m.decision(Xte) / T
d -= d.max(1, keepdims=True)
P_cal = np.exp(d)
P_cal /= P_cal.sum(1, keepdims=True)


def multiclass_ece(y_true, P, n_bins=10):
    conf, pred = P.max(1), P.argmax(1)
    edges = np.quantile(conf, np.linspace(0, 1, n_bins + 1))
    tot = 0.0
    for i in range(n_bins):
        msk = (conf >= edges[i]) & (conf <= edges[i + 1])
        if msk.sum():
            tot += msk.sum() / len(conf) * abs(
                (pred[msk] == y_true[msk]).mean() - conf[msk].mean())
    return tot


def logloss(y_true, P):
    return float(-np.mean(np.log(np.clip(
        P[np.arange(len(y_true)), y_true], 1e-12, 1))))


print(f"\nfitted temperature T = {T:.4f}\n")
print(f"{'':<14} {'accuracy':>10} {'mean conf':>11} {'ECE':>9} {'log loss':>10}")
print(f"{'raw':<14} {(P_te.argmax(1) == yte).mean():>10.4f} "
      f"{P_te.max(1).mean():>11.4f} {multiclass_ece(yte, P_te):>9.4f} "
      f"{logloss(yte, P_te):>10.4f}")
print(f"{'temperature':<14} {(P_cal.argmax(1) == yte).mean():>10.4f} "
      f"{P_cal.max(1).mean():>11.4f} {multiclass_ece(yte, P_cal):>9.4f} "
      f"{logloss(yte, P_cal):>10.4f}")
print("\nAccuracy is IDENTICAL — dividing every logit by a positive constant")
print("cannot change an argmax — while calibration error and log loss")
print("improve substantially. One scalar, fitted on validation data, turns")
print("an unusable probability into a usable one at no cost to the decision.")
print("The same trick reappears as sampling temperature in Chapter 90.")
