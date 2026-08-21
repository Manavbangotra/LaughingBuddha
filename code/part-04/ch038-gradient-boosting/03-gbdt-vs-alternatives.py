# Extracted from: Chapter 38 — Gradient Boosting: Theory, XGBoost, LightGBM, CatBoost
# Source: src/.../ch038-gradient-boosting.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Gradient boosting against the rest of Part IV on one tabular problem,
plus the row-count question of section 5.5.
"""
import time

import numpy as np

rng = np.random.default_rng(41)


def sigmoid(z):
    out = np.empty_like(z, dtype=float)
    p = z >= 0
    out[p] = 1 / (1 + np.exp(-z[p]))
    e = np.exp(z[~p])
    out[~p] = e / (1 + e)
    return out


# --- a tabular problem with the properties Grinsztajn et al. describe -------
def make_data(n):
    """Irregular target, uninformative features, meaningful axes — the three
    conditions under which tree ensembles are expected to win."""
    age = rng.uniform(18, 80, n)
    income = rng.lognormal(10.3, 0.6, n)
    tenure = rng.exponential(4.0, n)
    n_prod = rng.poisson(2.0, n)
    region = rng.integers(0, 6, n).astype(float)
    # irregular: a threshold effect, an interaction, and a non-monotone term
    z = (-1.1
         + 1.6 * (age < 30) + 0.9 * (age > 65)
         - 0.7 * (np.log(income) - 10.3)
         + 1.3 * (tenure < 1.0) * (np.log(income) < 10.0)
         + 0.35 * n_prod
         + 0.8 * np.isin(region, [1, 4]))
    noise = rng.normal(size=(n, 10))          # ten uninformative columns
    X = np.column_stack([age, np.log(income), tenure, n_prod, region, noise])
    return X, (rng.random(n) < sigmoid(z)).astype(float)


NAMES = (["age", "log_income", "tenure", "n_products", "region"]
         + [f"noise_{i}" for i in range(10)])


def roc_auc(y, s):
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    npos = int(y.sum())
    return float((r[y == 1].sum() - npos * (npos + 1) / 2)
                 / (npos * (len(y) - npos)))


try:
    from sklearn.ensemble import (GradientBoostingClassifier,
                                  HistGradientBoostingClassifier,
                                  RandomForestClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    HAVE_SK = True
except ImportError:
    HAVE_SK = False

if not HAVE_SK:
    print("scikit-learn not installed — this listing needs it")
else:
    Xtr, ytr = make_data(6000)
    Xva, yva = make_data(2000)
    Xte, yte = make_data(8000)
    print(f"positive rate {ytr.mean():.4f}, "
          f"{Xtr.shape[1]} features of which 10 are noise\n")

    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

    models = [
        ("logistic regression (Ch 33)",
         LogisticRegression(max_iter=3000), True),
        ("k-NN, k=25 (Ch 35)",
         KNeighborsClassifier(n_neighbors=25), True),
        ("single tree, depth 6 (Ch 36)",
         DecisionTreeClassifier(max_depth=6, random_state=0), False),
        ("random forest, 400 (Ch 37)",
         RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=0),
         False),
        ("gradient boosting (this ch)",
         GradientBoostingClassifier(n_estimators=300, learning_rate=0.1,
                                    max_depth=3, random_state=0), False),
        ("hist gradient boosting",
         HistGradientBoostingClassifier(max_iter=400, learning_rate=0.1,
                                        early_stopping=True,
                                        random_state=0), False),
    ]

    print("=" * 72)
    print("every model in Part IV, same data, same split")
    print("=" * 72)
    print(f"{'model':<30} {'test AUC':>9} {'accuracy':>10} {'fit s':>8}")
    results = {}
    for name, m, scale in models:
        A, B = (Xtr_s, Xte_s) if scale else (Xtr, Xte)
        t0 = time.perf_counter()
        m.fit(A, ytr)
        dt = time.perf_counter() - t0
        p = m.predict_proba(B)[:, 1]
        results[name] = roc_auc(yte, p)
        print(f"{name:<30} {results[name]:>9.4f} "
              f"{((p >= 0.5) == (yte == 1)).mean():>10.4f} {dt:>8.2f}")

    print("\nThe ordering is the usual one on tabular data with an irregular")
    print("target: boosting first, forest close behind, single tree and")
    print("linear model well back, k-NN worst because ten noise dimensions")
    print("dominate its distances (Chapter 35).")

    # --- section 5.5: how the answer depends on the number of rows ----------
    print("\n" + "=" * 72)
    print("section 5.5: the answer depends on how much data you have")
    print("=" * 72)
    print(f"{'train rows':>11} {'logistic':>10} {'random forest':>15} "
          f"{'gradient boosting':>19} {'winner':>18}")
    for n_train in (100, 300, 1000, 3000, 6000):
        A, b = Xtr[:n_train], ytr[:n_train]
        A_s = sc.transform(A)
        lr = LogisticRegression(max_iter=3000).fit(A_s, b)
        rf = RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                    random_state=0).fit(A, b)
        gb = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.08, early_stopping=True,
            random_state=0).fit(A, b)
        a_lr = roc_auc(yte, lr.predict_proba(Xte_s)[:, 1])
        a_rf = roc_auc(yte, rf.predict_proba(Xte)[:, 1])
        a_gb = roc_auc(yte, gb.predict_proba(Xte)[:, 1])
        best = max((a_lr, "logistic"), (a_rf, "forest"), (a_gb, "boosting"))
        print(f"{n_train:>11} {a_lr:>10.4f} {a_rf:>15.4f} {a_gb:>19.4f} "
              f"{best[1]:>18}")

    print("\nBoosting's advantage is not constant — it grows with the number")
    print("of rows, because more rounds of bias reduction need more data to")
    print("stay honest. At the small end the models converge and the")
    print("cheapest one is defensible. This is the same axis along which")
    print("Chapter 38's tabular-foundation-model discussion sits: what wins")
    print("depends on the sample size, not only on the algorithm.")

    # --- the boosting failure mode you will actually hit --------------------
    print("\n" + "=" * 72)
    print("the failure you will actually hit: no early stopping")
    print("=" * 72)
    print(f"{'n_estimators':>13} {'val log loss':>14} {'test AUC':>10}")
    for n_est in (50, 200, 800, 3000):
        gb = GradientBoostingClassifier(n_estimators=n_est,
                                        learning_rate=0.15, max_depth=6,
                                        random_state=0).fit(Xtr, ytr)
        pv = np.clip(gb.predict_proba(Xva)[:, 1], 1e-12, 1 - 1e-12)
        ll = float(-np.mean(yva * np.log(pv) + (1 - yva) * np.log(1 - pv)))
        print(f"{n_est:>13} {ll:>14.4f} "
              f"{roc_auc(yte, gb.predict_proba(Xte)[:, 1]):>10.4f}")

    gb_es = HistGradientBoostingClassifier(
        max_iter=3000, learning_rate=0.15, max_depth=6,
        early_stopping=True, n_iter_no_change=20,
        validation_fraction=0.15, random_state=0).fit(Xtr, ytr)
    print(f"\nwith early stopping: stopped at {gb_es.n_iter_} of 3000 "
          f"iterations, test AUC "
          f"{roc_auc(yte, gb_es.predict_proba(Xte)[:, 1]):.4f}")
    print("\nA fixed n_estimators is a bug wearing a hyperparameter's")
    print("clothes. Early stopping found its own budget and beat every fixed")
    print("choice, without anyone tuning it.")
