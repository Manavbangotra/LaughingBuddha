# -*- coding: utf-8 -*-
# Extracted from: Chapter 43 — Splits, Cross-Validation, and Honest Evaluation
# Source: src/.../ch043-splits.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A split that audits itself, and the decay it is there to catch.
"""
import numpy as np

rng = np.random.default_rng(11)


class SplitAudit(AssertionError):
    """Raised when a split violates an invariant it declared."""


def audited_split(train_idx, val_idx, *, n_total, groups=None, times=None,
                  min_embargo=0.0, min_val_frac=0.05, name="split"):
    """Check every invariant a split claims, and fail loudly if one breaks.

    The point is not that these checks are clever. It is that they run on
    every build, so a split that silently stops being honest becomes a red
    test rather than an improved validation score.
    """
    problems = []
    tr, va = np.asarray(train_idx), np.asarray(val_idx)

    if len(np.intersect1d(tr, va)):
        problems.append(f"{len(np.intersect1d(tr, va))} rows in BOTH sides")
    if len(va) < min_val_frac * n_total:
        problems.append(f"validation is {len(va) / n_total:.1%} of the data, "
                        f"below the declared floor of {min_val_frac:.0%}")
    if len(tr) == 0 or len(va) == 0:
        problems.append("one side is empty")

    if groups is not None:
        shared = np.intersect1d(np.unique(groups[tr]), np.unique(groups[va]))
        if len(shared):
            problems.append(
                f"{len(shared)} group(s) appear on both sides "
                f"(e.g. {shared[:3].tolist()}) — this is a grouped leak")
        # a group split also silently loses power; report it rather than fail
        n_tr_g, n_va_g = len(np.unique(groups[tr])), len(np.unique(groups[va]))
        if n_va_g < 5:
            problems.append(f"only {n_va_g} validation groups: the estimate "
                            f"has almost no resolution")

    if times is not None:
        if times[tr].max() > times[va].min() - min_embargo:
            overlap = times[tr].max() - (times[va].min() - min_embargo)
            problems.append(
                f"temporal overlap of {overlap:.1f} units: training data "
                f"reaches within the {min_embargo} embargo of validation")

    if problems:
        raise SplitAudit(f"[{name}] " + "; ".join(problems))
    return True


def group_split(groups, frac=0.75, seed=0):
    uniq = np.unique(groups)
    perm = np.random.default_rng(seed).permutation(uniq)
    keep = set(perm[:int(frac * len(uniq))].tolist())
    m = np.array([g in keep for g in groups])
    return np.flatnonzero(m), np.flatnonzero(~m)


def random_split(n, frac=0.75, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    return idx[:int(frac * n)], idx[int(frac * n):]


# --- the decay this is built to catch ---------------------------------------
print("=" * 72)
print("the failure mode: a split that WAS honest and quietly stopped being")
print("=" * 72)
print("Version 1 of the pipeline emits one row per customer, so a random")
print("split is correct and the audit passes.\n")

n_cust = 800
cust_v1 = np.arange(n_cust)                     # one row each
tr, va = random_split(len(cust_v1), seed=1)
try:
    audited_split(tr, va, n_total=len(cust_v1), groups=cust_v1,
                  name="v1 random split")
    print("  v1: audit PASSED — no customer on both sides")
except AssertionError as e:
    print(f"  v1: audit FAILED — {e}")

print("\nSix months later an upstream team starts emitting one row per")
print("SESSION. Nothing in the modelling code changed. Nothing errors.\n")
cust_v2 = np.repeat(np.arange(n_cust), 4)       # four rows each now
tr, va = random_split(len(cust_v2), seed=1)
try:
    audited_split(tr, va, n_total=len(cust_v2), groups=cust_v2,
                  name="v2 random split")
    print("  v2: audit PASSED")
except AssertionError as e:
    print(f"  v2: audit FAILED\n       {e}")

print("\nThe audit is what turns a silent optimism into a build failure. The")
print("fix is to switch to a grouped split, which the audit then accepts:\n")
tr, va = group_split(cust_v2, seed=1)
try:
    audited_split(tr, va, n_total=len(cust_v2), groups=cust_v2,
                  name="v2 grouped split")
    print("  v2 grouped: audit PASSED")
except AssertionError as e:
    print(f"  v2 grouped: audit FAILED — {e}")

# --- and what the difference was worth --------------------------------------
def make_data(groups, icc=0.35, seed=4):
    rs = np.random.default_rng(seed)
    n = len(groups)
    sig_u = np.sqrt(icc / (1 - icc))
    u = rs.normal(0, sig_u, groups.max() + 1)[groups]
    X = rs.normal(size=(n, 5))
    y = 1.1 * X[:, 0] - 0.8 * X[:, 1] + u + rs.normal(0, 1.0, n)
    return np.column_stack([X, groups.astype(float)]), y


def fit_knn(Xtr, ytr, Xte, k=5):
    d = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
    idx = np.argpartition(d, k, axis=1)[:, :k]
    return ytr[idx].mean(1)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


X2, y2 = make_data(cust_v2)
tr_r, va_r = random_split(len(y2), seed=1)
tr_g, va_g = group_split(cust_v2, seed=1)
print(f"\n{'split':<26} {'reported RMSE':>15} {'what it means':<32}")
print(f"{'v2 random (leaking)':<26} "
      f"{rmse(fit_knn(X2[tr_r], y2[tr_r], X2[va_r]), y2[va_r]):>15.4f} "
      f"{'sessions from known customers':<32}")
print(f"{'v2 grouped (honest)':<26} "
      f"{rmse(fit_knn(X2[tr_g], y2[tr_g], X2[va_g]), y2[va_g]):>15.4f} "
      f"{'sessions from NEW customers':<32}")
print("\nBoth numbers are correct answers to different questions. Only one of")
print("them is the question production asks.")

# --- the holdout ledger -----------------------------------------------------
print("\n" + "=" * 72)
print("the holdout ledger: making K_total knowable (eq. 43.7)")
print("=" * 72)


class Holdout:
    """Evaluation goes through a function, and the function keeps a ledger.

    A team that cannot say how many times its test set has been evaluated
    does not have a test set — it has a second validation set.
    """

    def __init__(self, X, y, val_se=0.01):
        self._X, self._y, self.val_se = X, y, val_se
        self.log = []

    def evaluate(self, model_fn, who, why):
        score = float(np.mean((model_fn(self._X) - self._y) ** 2) ** 0.5)
        self.log.append({"who": who, "why": why, "score": score})
        return score

    def optimism(self):
        k = max(len(self.log), 1)
        return float(self.val_se * np.sqrt(2 * np.log(max(k, 2))))

    def report(self):
        k = len(self.log)
        best = min(self.log, key=lambda r: r["score"]) if k else None
        print(f"  evaluations to date : {k}")
        print(f"  distinct people     : {len({r['who'] for r in self.log})}")
        if best:
            print(f"  best score reported : {best['score']:.4f} "
                  f"({best['who']}, {best['why']})")
        print(f"  expected optimism   : {self.optimism():.4f} "
              f"= SE x sqrt(2 log K)")
        if best:
            print(f"  corrected estimate  : "
                  f"{best['score'] + self.optimism():.4f}  (worse is honest)")


X_ho, y_ho = make_data(np.repeat(np.arange(200), 4), seed=9)
hold = Holdout(X_ho, y_ho)

# four people, over some months, each trying a few things — none of them
# aware of the others' usage
Xf, yf = make_data(cust_v2, seed=4)
tr_f, _ = group_split(cust_v2, seed=1)
for who, n_tries in (("ana", 6), ("ben", 11), ("cara", 4), ("dev", 9)):
    for i in range(n_tries):
        k = 3 + i
        hold.evaluate(lambda Z, k=k: fit_knn(Xf[tr_f], yf[tr_f], Z, k=k),
                      who, f"knn k={k}")

hold.report()
print("\nNobody looked more than eleven times. Collectively the test set has")
print("served thirty evaluations, and the best of thirty noisy scores is")
print("optimistic by roughly 2.6 standard errors — which is larger than most")
print("of the improvements anyone was chasing.")
print("\nThe ledger does not stop the problem. It makes K_total a number you")
print("can put in eq. 43.7 instead of a number nobody knows.")
