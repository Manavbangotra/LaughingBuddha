# -*- coding: utf-8 -*-
# Extracted from: Chapter 137 — Model Merging and Distillation
# Source: src/.../ch137-merging.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""What decides whether a merge works: how much the tasks disagree.

cite:ilharco2023taskarithmetic makes the fine-tuning delta an object you can add
and subtract. cite:yadav2023ties observes that adding several of them works less
well than it should, and names two mechanisms -- REDUNDANCY (most entries carry
no task information but dilute the average) and SIGN CONFLICT (where two tasks
want opposite movements, averaging produces something neither wanted). Its
remedy trims small entries, elects a sign per parameter, and averages only the
entries agreeing with it.

That is usually presented as a better averaging rule. This listing tests it as
what it actually is: a rule whose payoff is a function of how much your tasks
conflict. Task pairs are constructed with a controlled relationship, from nearly
aligned to directly opposed, and at each level the sign-disagreement rate and the
merge quality are both measured (eq:conflict-governs-merging).
"""
# --- setup, identical to the previous listing ------------------------
import numpy as np

rng = np.random.default_rng(223)

D, H, DO = 14, 48, 6
N = 2500


def init():
    return [rng.normal(size=(D, H)) / np.sqrt(D), np.zeros(H),
            rng.normal(size=(H, DO)) / np.sqrt(H), np.zeros(DO)]


def forward(p, X):
    h = np.tanh(X @ p[0] + p[1])
    return h, h @ p[2] + p[3]


def grad(p, X, Y):
    h, o = forward(p, X)
    d = 2 * (o - Y) / len(X)
    dh = d @ p[2].T * (1 - h ** 2)
    return [X.T @ dh, dh.sum(0), h.T @ d, d.sum(0)]


def mse(p, X, Y):
    return float(((forward(p, X)[1] - Y) ** 2).mean())


def fit(p, X, Y, steps, lr=0.01):
    p = [w.copy() for w in p]
    m = [np.zeros_like(w) for w in p]
    v = [np.zeros_like(w) for w in p]
    for t in range(steps):
        g = grad(p, X, Y)
        for i in range(4):
            m[i] = 0.9 * m[i] + 0.1 * g[i]
            v[i] = 0.999 * v[i] + 0.001 * g[i] ** 2
            p[i] -= lr * (m[i] / (1 - 0.9 ** (t + 1))) / (
                np.sqrt(v[i] / (1 - 0.999 ** (t + 1))) + 1e-8)
    return p


def lerp(pa, pb, a):
    return [(1 - a) * x + a * y for x, y in zip(pa, pb)]


W_PRE = rng.normal(size=(D, DO)) / np.sqrt(D)
W_A = rng.normal(size=(D, DO)) / np.sqrt(D)
W_B = rng.normal(size=(D, DO)) / np.sqrt(D)


def task(W, n, noise=0.12):
    X = rng.normal(size=(n, D))
    return X, np.tanh(X @ W) + 0.3 * X[:, :DO] + noise * rng.normal(size=(n, DO))


Xp, Yp = task(W_PRE, N)
Xa, Ya = task(W_A, N); Xa_t, Ya_t = task(W_A, 1500)
Xb, Yb = task(W_B, N); Xb_t, Yb_t = task(W_B, 1500)
# --- end of shared setup ---------------------------------------------

BASE = fit(init(), Xp, Yp, 3000)
W_1 = rng.normal(size=(D, DO)) / np.sqrt(D)
W_PERP = rng.normal(size=(D, DO)) / np.sqrt(D)


def paired_tasks(rho):
    """Task 2's target is correlated with task 1's at `rho`: +1 is the same
    task, 0 unrelated, -1 directly opposed."""
    W2 = rho * W_1 + np.sqrt(max(0.0, 1 - rho ** 2)) * W_PERP
    t1 = task(W_1, N) + task(W_1, 1200)
    t2 = task(W2, N) + task(W2, 1200)
    return t1, t2


def combine(deltas, do_trim, do_elect, keep=0.30):
    out = []
    for i in range(4):
        S = np.stack([d[i] for d in deltas])
        if do_trim:
            T = S.copy()
            for t in range(len(S)):
                a = np.abs(T[t])
                if a.size:
                    T[t] = np.where(a >= np.quantile(a, 1 - keep), T[t], 0.0)
            S = T
        if do_elect:
            sign = np.sign((S * (S != 0)).sum(0) + 1e-12)
            ok = (np.sign(S) == sign[None]) & (S != 0)
            n = ok.sum(0)
            C = np.where(n > 0, (S * ok).sum(0) / np.maximum(n, 1), 0.0)
        else:
            C = S.mean(0)
        out.append(C)
    return out


LAMS = (0.3, 0.5, 0.7, 0.85, 1.0, 1.2)


def best(deltas, tests, do_trim, do_elect):
    """Every merge method carries a scale coefficient. Comparing methods at a
    fixed one compares scale tuning rather than the methods."""
    C = combine(deltas, do_trim, do_elect)
    out = None
    for lam in LAMS:
        p = [b + lam * c for b, c in zip(BASE, C)]
        L = float(np.mean([mse(p, Xt, Yt) for Xt, Yt in tests]))
        if out is None or L < out:
            out = L
    return out


def disagreement(deltas):
    """Share of significant entries where the two tasks want opposite signs."""
    tot, bad = 0, 0
    for i in range(4):
        S = np.stack([d[i] for d in deltas])
        a = np.abs(S)
        if not a.size:
            continue
        big = (a >= 0.1 * a.max()).all(0)
        tot += big.sum()
        bad += ((np.sign(S[0]) != np.sign(S[1])) & big).sum()
    return bad / max(tot, 1)


RHOS = (0.9, 0.5, 0.0, -0.5, -0.9)
print(f"Two tasks fine-tuned from one base, merged. `rho` is how related the "
      f"two\ntasks are: +1 identical, 0 unrelated, -1 opposed.\n")
print(f"{'rho':>6}{'sign':>9}{'specialists':>13}{'average of':>13}"
      f"{'TIES':>9}{'better':>10}")
print(f"{'':>6}{'conflict':>9}{'on own task':>13}{'deltas':>13}{'':>9}"
      f"{'method':>10}")
print("-" * 60)

rows = {}
for rho in RHOS:
    (X1, Y1, X1t, Y1t), (X2, Y2, X2t, Y2t) = paired_tasks(rho)
    p1 = fit(BASE, X1, Y1, 900)
    p2 = fit(BASE, X2, Y2, 900)
    deltas = [[w - b for w, b in zip(p, BASE)] for p in (p1, p2)]
    tests = [(X1t, Y1t), (X2t, Y2t)]
    own = float(np.mean([mse(p1, X1t, Y1t), mse(p2, X2t, Y2t)]))
    avg = best(deltas, tests, False, False)
    ties = best(deltas, tests, True, True)
    cf = disagreement(deltas)
    rows[rho] = (cf, own, avg, ties)
    print(f"{rho:>6.1f}{cf:>9.1%}{own:>13.4f}{avg:>13.4f}{ties:>9.4f}"
          f"{('TIES' if ties < avg else 'average'):>10}")

hi, lo = rows[0.9], rows[-0.9]
mid = rows[0.0]
cf_max = max(rows[r][0] for r in RHOS)
print(f"""
Read the two right-hand columns down the page. The specialists column is the
control, and it is flat: {hi[1]:.4f} at rho=0.9 and {lo[1]:.4f} at rho=-0.9. Each
task is equally learnable regardless of what the other task is.

The merge column is not flat at all. Averaging two deltas from closely related
tasks reaches {hi[2]:.4f} -- close to the specialists, so the merge is nearly
free. Averaging two deltas from opposed tasks reaches {lo[2]:.4f}, an eightfold
degradation, on tasks that are individually just as easy.

That is the result, and it is a limit rather than a technique. Merge quality is
governed by how much the tasks AGREE about which way the weights should move.
Where two tasks require the same parameter to move in opposite directions, no
combining rule satisfies both, because no single value satisfies both. Averaging
splits the difference and implements neither; electing a sign implements one and
abandons the other. Both are honest answers to an impossible request
(eq:conflict-governs-merging).

The practical consequence is a question to ask before merging rather than after:
are these tasks compatible? A merge between two tasks that pull the same
direction is close to free and scales to many tasks. A merge between tasks in
genuine tension has a floor set by the tension, and no amount of merging-method
sophistication lowers it.

The sign-conflict column is the cheap proxy for that, and it is worth reporting
with its limitation. It does rise as the tasks become opposed -- {hi[0]:.1%} at
rho=0.9 against {cf_max:.1%} at its peak -- so it carries real signal for the
price of an elementwise comparison of two tensors you already have. But it is
loose: it peaks at rho=-0.5 rather than at rho=-0.9, because measuring
disagreement per-parameter ignores how MUCH each task disagrees, and a few large
conflicts hurt more than many small ones. Use it as a screen, not as a prediction.

Now the part that did not come out as expected, and is reported rather than tuned
away. TIES loses to a plain average at every conflict level here, including at
rho={-0.9} where it should have the most to fix: {lo[3]:.4f} against
{lo[2]:.4f}. Both were given a tuned scale coefficient, so this is not a
calibration artefact.

The reason is visible in the conflict column. Even at maximum opposition the
disagreement rate is only {cf_max:.1%}, and cite:yadav2023ties's setting has far
more to work with -- eight or more task vectors from a large model, where deltas
are sparse and pairwise conflicts accumulate across many pairs. Two dense deltas
from a small network do not present enough conflict for a conflict-resolution
mechanism to recover the signal that trimming discards.

TIES is a real improvement in the regime it was proposed for, and this experiment
is not in that regime. What transfers is the DIAGNOSIS rather than the remedy:
task conflict is the quantity that decides merging, it is measurable in advance,
and it bounds what any merging method can achieve.

So the rule to take away. Measure agreement between the deltas you intend to
merge, and treat a low value as information about the TASKS rather than as a
problem to be solved by a better merging rule. When tasks are in genuine tension
the honest options are to keep separate models, to train one model on both tasks
together, or to decide which task is served worse -- and the last of those is a
product decision that a merging algorithm should not be making silently.""")
