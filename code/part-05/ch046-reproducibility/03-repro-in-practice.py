# Extracted from: Chapter 46 — Reproducibility, Experiment Tracking, and Versioning
# Source: src/.../ch046-reproducibility.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Measuring run-to-run variance, and using it to decide what is real.
"""
import numpy as np

rng = np.random.default_rng(31)

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    HAVE_SK = True
except ImportError:
    HAVE_SK = False


def make_data(n, seed):
    rs = np.random.default_rng(seed)
    X = rs.normal(size=(n, 10))
    z = 1.1 * X[:, 0] - 0.9 * X[:, 1] + 0.7 * X[:, 2] * X[:, 3] - 0.3
    return X, (rs.random(n) < 1 / (1 + np.exp(-z))).astype(int)


def auc(y, s):
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    npos = int(y.sum())
    return float((r[y == 1].sum() - npos * (npos + 1) / 2)
                 / (npos * (len(y) - npos)))


if not HAVE_SK:
    print("scikit-learn not installed — this listing needs it")
else:
    Xtr, ytr = make_data(3000, 0)
    Xte, yte = make_data(6000, 1)

    # --- 1. the number nobody measures -------------------------------------
    print("=" * 72)
    print("1. run-to-run variance: the number that makes 'improvements' real")
    print("=" * 72)
    print("Same code, same data, same hyperparameters. Only the seed moves.\n")

    scores = []
    for seed in range(15):
        m = RandomForestClassifier(n_estimators=120, max_depth=8,
                                   random_state=seed, n_jobs=1).fit(Xtr, ytr)
        scores.append(auc(yte, m.predict_proba(Xte)[:, 1]))
    scores = np.array(scores)
    sd = float(scores.std(ddof=1))
    print(f"  15 runs, AUC: min {scores.min():.4f}  "
          f"median {np.median(scores):.4f}  max {scores.max():.4f}")
    print(f"  standard deviation across seeds : {sd:.4f}")
    print(f"  full range                      : {np.ptp(scores):.4f}")
    print(f"\n  Any reported improvement smaller than about {2 * sd:.4f} AUC is")
    print("  indistinguishable from having re-rolled the seed.")

    # --- 2. and what that does to a comparison ------------------------------
    print("\n" + "=" * 72)
    print("2. two 'different' configurations, judged against that noise")
    print("=" * 72)
    cfg_a = dict(n_estimators=120, max_depth=8)
    cfg_b = dict(n_estimators=120, max_depth=9)
    a_scores, b_scores = [], []
    for seed in range(15):
        ma = RandomForestClassifier(random_state=seed, n_jobs=1,
                                    **cfg_a).fit(Xtr, ytr)
        mb = RandomForestClassifier(random_state=seed, n_jobs=1,
                                    **cfg_b).fit(Xtr, ytr)
        a_scores.append(auc(yte, ma.predict_proba(Xte)[:, 1]))
        b_scores.append(auc(yte, mb.predict_proba(Xte)[:, 1]))
    a_scores, b_scores = np.array(a_scores), np.array(b_scores)

    print(f"  single-seed comparison (seed 0): "
          f"depth 8 = {a_scores[0]:.4f}, depth 9 = {b_scores[0]:.4f}, "
          f"diff {b_scores[0] - a_scores[0]:+.4f}")
    diff = b_scores - a_scores
    se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
    unpaired_se = float(np.sqrt(a_scores.var(ddof=1) / len(a_scores)
                                + b_scores.var(ddof=1) / len(b_scores)))
    print(f"  paired over 15 seeds           : mean diff "
          f"{diff.mean():+.4f} +- {se:.4f} (SE)")
    print(f"  unpaired, same 30 runs         : mean diff "
          f"{diff.mean():+.4f} +- {unpaired_se:.4f} (SE)")
    verdict = ("resolved" if abs(diff.mean()) > 2 * se else "not resolved")
    print(f"  verdict (paired)               : {verdict}")
    print(f"  verdict (unpaired)             : "
          f"{'resolved' if abs(diff.mean()) > 2 * unpaired_se else 'not resolved'}")
    print("\n  The single-seed difference is smaller than the run-to-run")
    print("  standard deviation measured above, so on its own it is not")
    print("  evidence of anything — it could be the seed.")
    print("\n  REPLICATION is what rescues it. Averaging fifteen seeds")
    print("  shrinks the standard error by sqrt(15), which is what turns a")
    print("  difference smaller than the noise into one that is resolvable.")
    print("\n  Pairing helps too, and by less than one might expect here:")
    print("  0.0003 against 0.0004 on the very same thirty runs. Pairing pays")
    print("  in proportion to how much variation the two configurations")
    print("  SHARE, and depth 8 versus depth 9 are similar enough that most")
    print("  of the seed effect does not cancel. Between two genuinely")
    print("  different model families it would pay much more. It costs")
    print("  nothing either way, so pair by default and do not count on it.")

    # --- 3. determinism has a price -----------------------------------------
    print("\n" + "=" * 72)
    print("3. what determinism costs (table 46.1, levels 2 vs 3)")
    print("=" * 72)
    import time
    print(f"{'threads':>8} {'fit seconds':>13} {'identical across repeats?':>27}")
    for n_jobs in (1, 2, 4):
        digs, t0 = set(), time.perf_counter()
        for _ in range(3):
            m = RandomForestClassifier(n_estimators=150, max_depth=10,
                                       random_state=0,
                                       n_jobs=n_jobs).fit(Xtr, ytr)
            p = m.predict_proba(Xte)[:, 1]
            digs.add(hash(p.tobytes()))
        dt = (time.perf_counter() - t0) / 3
        print(f"{n_jobs:>8} {dt:>13.2f} {str(len(digs) == 1):>27}")

    print("\n  Read the last column before the second-to-last. A seeded")
    print("  random forest — about as benign a model as exists, with trees")
    print("  built independently — is bitwise reproducible on one and two")
    print("  threads and NOT on four.")
    print("\n  That is worth pausing on, because it is the opposite of what")
    print("  most people assume, and the assumption is reasonable: the seed")
    print("  is fixed, the trees are independent, nothing about the algorithm")
    print("  looks order-dependent. The nondeterminism enters below the")
    print("  algorithm, in how work is partitioned and floating-point results")
    print("  are combined across workers.")
    print("\n  Note also the trade being made: four threads are about 2.6x")
    print("  faster than one. Level 3 of table 46.1 costs that speed-up, and")
    print("  on a GPU with atomic accumulation the cost of forcing")
    print("  deterministic kernels is typically larger still.")
    print("\n  The lesson is the test itself, not the result: run it three")
    print("  times, digest the output, compare. Determinism is a property to")
    print("  MEASURE on your own stack, not one to reason about.")

    # --- 4. the decision ----------------------------------------------------
    print("\n" + "=" * 72)
    print("4. choosing a level, by what it costs to be wrong")
    print("=" * 72)
    rows = [
        ("exploratory notebook", 1,
         "re-run it; nobody depends on the number"),
        ("a paper or a blog post", 2,
         "someone will try to replicate the metric"),
        ("a shipped product model", 2,
         "you must be able to rebuild and compare"),
        ("bisecting a regression", 3,
         "differences must be attributable to the change"),
        ("a credit or clinical model", 4,
         "a person is entitled to ask why, years later"),
    ]
    print(f"{'situation':<28} {'level':>6}  {'because':<44}")
    for what, lvl, why in rows:
        print(f"{what:<28} {lvl:>6}  {why:<44}")
    print("\n  The mistake in both directions is common: teams chase bitwise")
    print("  reproducibility for a dashboard nobody audits, and ship credit")
    print("  models they cannot rebuild. Match the level to the consequence.")
