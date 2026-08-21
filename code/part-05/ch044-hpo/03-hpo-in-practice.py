# Extracted from: Chapter 44 — Hyperparameter Optimization: Grid, Random, and Bayesian
# Source: src/.../ch044-hpo.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Tuning gradient boosting properly: log scales, pruning, a stopping rule,
and an honest final number.
"""
import time

import numpy as np

rng = np.random.default_rng(17)

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    HAVE_SK = True
except ImportError:
    HAVE_SK = False


def make_data(n):
    X = rng.normal(size=(n, 14))
    z = (1.3 * np.sin(1.4 * X[:, 0]) + 0.9 * X[:, 1]
         - 1.1 * X[:, 0] * X[:, 2] + 0.8 * np.abs(X[:, 3])
         + 0.5 * (X[:, 4] > 0.7) - 0.4)
    return X, (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)


if not HAVE_SK:
    print("scikit-learn not installed — this listing needs it")
else:
    Xtr, ytr = make_data(4000)
    Xte, yte = make_data(6000)

    # --- 1. the search space, and why the scales are what they are ----------
    print("=" * 72)
    print("1. search-space design is where most of the decisions are made")
    print("=" * 72)
    SPACE = {
        "learning_rate":     ("log",   0.01, 0.5),
        "max_leaf_nodes":    ("logint",  8,  256),
        "min_samples_leaf":  ("logint",  5,  200),
        "l2_regularization": ("log",   1e-4, 10.0),
        "max_features":      ("lin",    0.4, 1.0),
    }
    for k, (scale, lo, hi) in SPACE.items():
        why = {"log": "spans orders of magnitude — a uniform draw would put "
                      "90% of its mass in the top decade",
               "logint": "same, and integer-valued",
               "lin": "a genuine proportion; uniform is what we mean"}[scale]
        print(f"  {k:<19} {scale:>7}  [{lo}, {hi}]")
        print(f"  {'':<19} {why}")

    def sample(rs):
        out = {}
        for k, (scale, lo, hi) in SPACE.items():
            if scale == "lin":
                out[k] = float(rs.uniform(lo, hi))
            elif scale == "log":
                out[k] = float(10 ** rs.uniform(np.log10(lo), np.log10(hi)))
            else:
                out[k] = int(round(10 ** rs.uniform(np.log10(lo),
                                                    np.log10(hi))))
        return out

    # --- 2. why the log scale is not a detail -------------------------------
    print("\n" + "=" * 72)
    print("2. what a uniform draw would have done to the learning rate")
    print("=" * 72)
    rs = np.random.default_rng(0)
    lin_draws = rs.uniform(0.01, 0.5, 4000)
    log_draws = 10 ** rs.uniform(np.log10(0.01), np.log10(0.5), 4000)
    print(f"{'decade':>16} {'uniform draws':>15} {'log-uniform draws':>19}")
    for lo, hi in ((0.01, 0.05), (0.05, 0.1), (0.1, 0.5)):
        print(f"  [{lo:>5}, {hi:>5}] "
              f"{np.mean((lin_draws >= lo) & (lin_draws < hi)):>15.1%} "
              f"{np.mean((log_draws >= lo) & (log_draws < hi)):>19.1%}")
    print("\nA uniform draw spends over 80% of its trials in the top decade")
    print("and barely visits the small learning rates, which are where a")
    print("boosted model with enough rounds usually wants to be. The scale")
    print("is not a formatting choice; it decides where the budget goes.")

    # --- 3. random search with a median pruner over boosting rounds ---------
    print("\n" + "=" * 72)
    print("3. random search, with and without a pruner")
    print("=" * 72)

    folds = list(StratifiedKFold(4, shuffle=True, random_state=0)
                 .split(Xtr, ytr))

    def score_config(cfg, max_iter, folds_to_use):
        aucs = []
        for tr, va in folds_to_use:
            m = HistGradientBoostingClassifier(
                max_iter=max_iter, early_stopping=False, random_state=0,
                **cfg).fit(Xtr[tr], ytr[tr])
            p = m.predict_proba(Xtr[va])[:, 1]
            o = np.argsort(p, kind="mergesort")
            r = np.empty(len(p))
            r[o] = np.arange(1, len(p) + 1)
            npos = int(ytr[va].sum())
            aucs.append((r[ytr[va] == 1].sum() - npos * (npos + 1) / 2)
                        / (npos * (len(p) - npos)))
        return float(np.mean(aucs)), max_iter * len(folds_to_use)

    def run_search(n_trials, use_pruner, seed=0):
        rs = np.random.default_rng(seed)
        best, cost, seen = (None, -np.inf), 0, 0
        rung_scores = {30: [], 100: []}
        for _ in range(n_trials):
            cfg = sample(rs)
            seen += 1
            if use_pruner:
                s30, c = score_config(cfg, 30, folds[:2])
                cost += c
                rung_scores[30].append(s30)
                if (len(rung_scores[30]) >= 5
                        and s30 < np.median(rung_scores[30])):
                    continue                       # pruned at rung 1
            s, c = score_config(cfg, 300, folds)
            cost += c
            if s > best[1]:
                best = (cfg, s)
        return best, cost, seen

    for use_pruner in (False, True):
        t0 = time.perf_counter()
        (cfg, s), cost, seen = run_search(14, use_pruner, seed=1)
        dt = time.perf_counter() - t0
        label = "with median pruner" if use_pruner else "no pruner"
        print(f"{label:<22} best CV AUC {s:.4f}   "
              f"{seen} trials, {cost:,} tree-fold-rounds, {dt:.1f}s")

    print("\nThe pruner cut the compute by about a third and cost a few")
    print("tenths of an AUC point. That is the small-budget regime from")
    print("listing 2 showing up in a real fit: with only fourteen trials the")
    print("median rule has very little to rank against, and it killed a")
    print("configuration that would have done well. At fourteen trials a")
    print("pruner is not obviously worth it; at four hundred it is not")
    print("optional. Match the machinery to the budget.")

    # --- 4. the stopping rule -----------------------------------------------
    print("\n" + "=" * 72)
    print("4. when to stop: compare improvement against the noise floor")
    print("=" * 72)
    rs = np.random.default_rng(4)
    running_best, history = -np.inf, []
    fold_ses = []
    for t in range(1, 17):
        cfg = sample(rs)
        aucs = []
        for tr, va in folds:
            m = HistGradientBoostingClassifier(
                max_iter=200, early_stopping=False, random_state=0,
                **cfg).fit(Xtr[tr], ytr[tr])
            p = m.predict_proba(Xtr[va])[:, 1]
            o = np.argsort(p, kind="mergesort")
            r = np.empty(len(p))
            r[o] = np.arange(1, len(p) + 1)
            npos = int(ytr[va].sum())
            aucs.append((r[ytr[va] == 1].sum() - npos * (npos + 1) / 2)
                        / (npos * (len(p) - npos)))
        s = float(np.mean(aucs))
        fold_ses.append(float(np.std(aucs, ddof=1) / np.sqrt(len(aucs))))
        running_best = max(running_best, s)
        history.append(running_best)

    se = float(np.mean(fold_ses))
    print(f"typical fold-to-fold standard error: {se:.4f}\n")
    print(f"{'trial':>6} {'best so far':>13} {'gain over 5 trials ago':>24} "
          f"{'vs noise floor':>16}")
    for t in range(5, len(history), 3):
        gain = history[t] - history[t - 5]
        verdict = "still worth it" if gain > se else "now selecting noise"
        print(f"{t + 1:>6} {history[t]:>13.4f} {gain:>24.4f} "
              f"{verdict:>16}")

    print("\nThe stopping criterion is not convergence — it is the point at")
    print("which the improvement over the last several trials falls below")
    print("the standard error of the estimate itself. Past that, eq. 44.12")
    print("says the reported best is drifting further above the truth with")
    print("every trial you add.")
    print("\nNote that the verdict FLICKERS: the rule fires at trial 9 and")
    print("then un-fires at 12. That is expected — the improvement is itself")
    print("a noisy quantity, so a single-window rule will trip early. Use it")
    print("with patience, as an early-stopping rule is used in Chapter 38:")
    print("stop after k consecutive windows below the floor, not the first")
    print("one.")

    # --- 5. the honest final number -----------------------------------------
    print("\n" + "=" * 72)
    print("5. re-evaluate the winner: a search selects, it does not measure")
    print("=" * 72)
    (best_cfg, best_cv), _, _ = run_search(14, True, seed=1)
    m = HistGradientBoostingClassifier(max_iter=300, early_stopping=False,
                                       random_state=0,
                                       **best_cfg).fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    o = np.argsort(p, kind="mergesort")
    r = np.empty(len(p))
    r[o] = np.arange(1, len(p) + 1)
    npos = int(yte.sum())
    test_auc = float((r[yte == 1].sum() - npos * (npos + 1) / 2)
                     / (npos * (len(p) - npos)))
    print(f"best CV AUC reported by the search : {best_cv:.4f}")
    print(f"same configuration on held-out data: {test_auc:.4f}")
    print(f"optimism                           : {best_cv - test_auc:+.4f}")
    print("\nOn this single run the two agree to within half a point, which")
    print("is well inside the fold-to-fold standard error above — so this")
    print("particular number demonstrates nothing on its own. The systematic")
    print("effect is the averaged measurement in listing 2, where the")
    print("overstatement grew monotonically with the number of trials.")
    print("\nThe discipline is what matters, not this run's arithmetic: one")
    print("extra fit turns a number you SELECTED into a number you MEASURED,")
    print("and at fourteen trials the correction is small only because the")
    print("search was small.")
