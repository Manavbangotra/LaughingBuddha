# -*- coding: utf-8 -*-
# Extracted from: Chapter 39 — Support Vector Machines and Kernels
# Source: src/.../ch039-svm.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""When an SVM is the right choice, and what it costs — against the rest of
Part IV, using scikit-learn so the comparison is fair.
"""
import time

import numpy as np

rng = np.random.default_rng(19)

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC, LinearSVC
    HAVE_SK = True
except ImportError:
    HAVE_SK = False


def roc_auc(y, s):
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    npos = int((y == 1).sum())
    return float((r[y == 1].sum() - npos * (npos + 1) / 2)
                 / (npos * ((y != 1).sum())))


if not HAVE_SK:
    print("scikit-learn not installed — this listing needs it")
else:
    # --- 1. the regime where SVMs still win: D > N ---------------------------
    print("=" * 72)
    print("1. high dimension, few samples — the SVM's remaining home ground")
    print("=" * 72)
    print("The margin argument of section 6.1 does not mention the dimension,")
    print("which is why SVMs degrade gracefully when D exceeds N.\n")
    print(f"{'N':>6} {'D':>6} {'logistic':>10} {'linear SVM':>12} "
          f"{'RBF SVM':>9} {'boosting':>10}")
    for N, D in ((40, 500), (80, 500), (200, 500), (1000, 500)):
        w_true = np.zeros(D)
        w_true[rng.choice(D, 15, replace=False)] = rng.normal(0, 1.5, 15)
        Xall = rng.normal(size=(N + 3000, D))
        z = Xall @ w_true
        yall = np.where(z + rng.normal(0, 1.0, len(z)) > 0, 1, -1)
        Xtr, ytr, Xte, yte = Xall[:N], yall[:N], Xall[N:], yall[N:]
        sc = StandardScaler().fit(Xtr)
        A, B = sc.transform(Xtr), sc.transform(Xte)
        scores = []
        for m in (LogisticRegression(max_iter=4000),
                  LinearSVC(C=0.1, max_iter=20000),
                  SVC(kernel="rbf", C=1.0, gamma="scale"),
                  HistGradientBoostingClassifier(max_iter=200,
                                                 random_state=0)):
            m.fit(A, ytr)
            scores.append((m.predict(B) == yte).mean())
        print(f"{N:>6} {D:>6} {scores[0]:>10.4f} {scores[1]:>12.4f} "
              f"{scores[2]:>9.4f} {scores[3]:>10.4f}")

    print("\nAt N=40 with D=500 the linear SVM is clearly ahead: with twelve")
    print("times more features than rows, a maximum-margin solution is a")
    print("better-posed thing to ask for than a fitted probability. Boosting")
    print("needs rows to split on and has too few. The ordering reverses as")
    print("N grows — the same sample-size axis as Chapters 35 and 38.")

    # --- 2. the cost that ended their dominance ------------------------------
    print("\n" + "=" * 72)
    print("2. why they lost: training cost is superlinear (section 5.5)")
    print("=" * 72)
    print(f"{'N':>7} {'RBF SVM fit s':>15} {'boosting fit s':>16} "
          f"{'ratio':>8} {'kernel matrix':>15}")
    for N in (500, 1000, 2000, 4000, 8000):
        X = rng.normal(size=(N, 20))
        yv = np.where(np.sin(X[:, 0] * 2) + X[:, 1] - X[:, 2] ** 2
                      + rng.normal(0, 0.4, N) > 0, 1, -1)
        t0 = time.perf_counter()
        SVC(kernel="rbf", gamma="scale").fit(X, yv)
        t_svm = time.perf_counter() - t0
        t0 = time.perf_counter()
        HistGradientBoostingClassifier(max_iter=200,
                                       random_state=0).fit(X, yv)
        t_gb = time.perf_counter() - t0
        mem = N * N * 8 / 1e6
        print(f"{N:>7} {t_svm:>15.3f} {t_gb:>16.3f} {t_svm / t_gb:>8.2f} "
              f"{mem:>12.1f} MB")

    print("\nSVM training time grows faster than linearly while boosting's is")
    print("nearly flat over this range. Extrapolate the kernel-matrix column:")
    print("at N = 1,000,000 it is 8 terabytes. That is the whole story of why")
    print("kernel methods stopped being the default, and it is about scaling")
    print("rather than about the idea being wrong.")

    # --- 3. no probabilities, and what it costs to add them ------------------
    print("\n" + "=" * 72)
    print("3. an SVM has no probabilities (section 5.5)")
    print("=" * 72)
    N = 3000
    X = rng.normal(size=(N + 4000, 12))
    z = 1.3 * np.sin(X[:, 0]) + X[:, 1] - 0.8 * X[:, 2] ** 2 + 0.5 * X[:, 3]
    y01 = (rng.random(len(z)) < 1 / (1 + np.exp(-z))).astype(int)
    Xtr, Xte = X[:N], X[N:]
    ytr01, yte01 = y01[:N], y01[N:]
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)

    svm = SVC(kernel="rbf", C=1.0, gamma="scale").fit(A, ytr01)
    raw = svm.decision_function(B)
    print(f"decision_function range: [{raw.min():.3f}, {raw.max():.3f}]")
    print("These are signed distances to the boundary, not probabilities:")
    print("they are unbounded and have no frequency interpretation at all.\n")

    cal = CalibratedClassifierCV(
        SVC(kernel="rbf", C=1.0, gamma="scale"), method="sigmoid",
        cv=3).fit(A, ytr01)
    p_cal = cal.predict_proba(B)[:, 1]
    logit = LogisticRegression(max_iter=4000).fit(A, ytr01)
    p_log = logit.predict_proba(B)[:, 1]

    def ece(y, p, nb=10):
        e = np.quantile(p, np.linspace(0, 1, nb + 1))
        t = 0.0
        for i in range(nb):
            m = (p >= e[i]) & (p <= e[i + 1])
            if m.sum():
                t += m.sum() / len(p) * abs(y[m].mean() - p[m].mean())
        return t

    ys = np.where(yte01 == 1, 1, -1)
    print(f"{'model':<34} {'AUC':>8} {'ECE':>8} {'accuracy':>10}")
    print(f"{'SVM raw decision function':<34} {roc_auc(ys, raw):>8.4f} "
          f"{'n/a':>8} {((raw > 0) == (yte01 == 1)).mean():>10.4f}")
    print(f"{'SVM + Platt scaling (3-fold)':<34} {roc_auc(ys, p_cal):>8.4f} "
          f"{ece(yte01, p_cal):>8.4f} "
          f"{((p_cal >= 0.5) == (yte01 == 1)).mean():>10.4f}")
    print(f"{'logistic regression':<34} {roc_auc(ys, p_log):>8.4f} "
          f"{ece(yte01, p_log):>8.4f} "
          f"{((p_log >= 0.5) == (yte01 == 1)).mean():>10.4f}")

    print("\nPlatt scaling gives the SVM usable probabilities, and it is not")
    print("free: it needs an internal 3-fold cross-validation, so the model")
    print("is fitted four times. Logistic regression produced calibrated")
    print("probabilities as a property of its loss function, at no extra")
    print("cost — the difference traced in table 39.1 back to the exact zero")
    print("in the hinge gradient.")

    # --- 4. a decision rule --------------------------------------------------
    print("\n" + "=" * 72)
    print("4. when to reach for an SVM in 2026")
    print("=" * 72)
    rules = [
        ("N < ~10,000 and the boundary is curved", "RBF SVM is competitive"),
        ("D > N (genomics, spectra, small text)", "linear SVM, strong"),
        ("large sparse text, N up to millions", "LinearSVC, still excellent"),
        ("tabular, N > ~50,000", "gradient boosting (Chapter 38)"),
        ("you need calibrated probabilities", "logistic or boosting"),
        ("images, audio, language", "learned representations, Part VI+"),
    ]
    for cond, verdict in rules:
        print(f"  {cond:<42} -> {verdict}")
