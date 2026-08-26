# -*- coding: utf-8 -*-
# Extracted from: Chapter 44 — Hyperparameter Optimization: Grid, Random, and Bayesian
# Source: src/.../ch044-hpo.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The measurement that matters: which contributes more, the sampler or the
pruner?
"""
import numpy as np

rng = np.random.default_rng(3)


# --- a realistic surrogate: an expensive, noisy, iterative fit --------------
class Objective:
    """Stands in for 'fit a gradient-boosting model at this configuration'.

    Cost is counted in resource units (boosting rounds). A partial fit gives
    an intermediate score, which is what makes pruning possible at all.
    """

    def __init__(self, n_dims=6, n_important=2, seed=0):
        rs = np.random.default_rng(seed)
        self.opt = rs.uniform(0.25, 0.75, n_important)
        self.w = rs.uniform(0.8, 1.2, n_important)
        self.n_dims, self.n_imp = n_dims, n_important
        self.spent = 0

    def final(self, phi):
        d = np.asarray(phi, float)[:self.n_imp] - self.opt
        return float(np.sum(self.w * d ** 2))       # lower is better

    def evaluate(self, phi, resource, rs):
        """Score after `resource` rounds: converges to `final` from above."""
        self.spent += resource
        f = self.final(phi)
        gap = 0.6 * np.exp(-resource / 12.0)        # not yet converged
        return f + gap + float(rs.normal(0, 0.02 + 0.10 / np.sqrt(resource)))


# --- samplers ---------------------------------------------------------------
def random_sampler(n_dims, history, rs):
    return rs.uniform(0, 1, n_dims)


def tpe_sampler(n_dims, history, rs, gamma=0.25, n_candidates=24, bw=0.12):
    """A compact TPE (eq. 44.11): split trials at the gamma quantile, fit a
    Parzen density to each side, and propose the candidate maximising
    l(phi)/g(phi) — which section 6.3 shows is expected improvement."""
    if len(history) < 8:
        return rs.uniform(0, 1, n_dims)
    phis = np.array([h[0] for h in history])
    ys = np.array([h[1] for h in history])
    cut = np.quantile(ys, gamma)
    good, bad = phis[ys <= cut], phis[ys > cut]
    if len(good) < 2 or len(bad) < 2:
        return rs.uniform(0, 1, n_dims)

    cands = rs.uniform(0, 1, (n_candidates, n_dims))

    def logdens(C, pts):
        # product of per-dimension Gaussian kernel density estimates
        d = (C[:, None, :] - pts[None, :, :]) / bw
        logk = -0.5 * d ** 2
        return np.sum(np.log(np.mean(np.exp(logk), axis=1) + 1e-12), axis=1)

    return cands[int(np.argmax(logdens(cands, good) - logdens(cands, bad)))]


# --- the search loop, with sampler and pruner as separate knobs -------------
def search(sampler, use_pruner, budget, n_dims=6, seed=0,
           r_min=1, r_max=27, eta=3):
    """Spend `budget` resource units; return the true final loss of the
    configuration the search would report."""
    rs = np.random.default_rng(seed)
    obj = Objective(n_dims=n_dims, seed=seed)
    history, best = [], (None, np.inf)
    rungs = [r_min * eta ** k for k in range(int(np.log(r_max / r_min)
                                                 / np.log(eta)) + 1)]
    rung_scores = {r: [] for r in rungs}

    while obj.spent < budget:
        phi = sampler(n_dims, history, rs)
        if not use_pruner:
            score = obj.evaluate(phi, r_max, rs)
        else:
            score, killed = None, False
            for r in rungs:
                score = obj.evaluate(phi, r, rs)
                rung_scores[r].append(score)
                if r < rungs[-1] and len(rung_scores[r]) >= 5:
                    # prune if worse than the median at this rung
                    if score > np.median(rung_scores[r]):
                        killed = True
                        break
            if killed:
                history.append((phi, score))
                continue
        history.append((phi, score))
        true = obj.final(phi)
        if score < best[1]:
            best = (phi, score)
    return obj.final(best[0]) if best[0] is not None else np.inf


print("=" * 72)
print("sampler vs pruner: which one is buying the improvement?")
print("=" * 72)
print("Both axes varied independently, at three budgets. Lower is better;")
print("each cell is the mean TRUE loss of the configuration reported, over")
print("40 independent runs.\n")
print(f"{'budget':>8} {'random, no prune':>18} {'random + prune':>16} "
      f"{'TPE, no prune':>15} {'TPE + prune':>13}")
results = {}
for budget in (270, 810, 2700):
    row = []
    for sampler, use_prune in ((random_sampler, False), (random_sampler, True),
                               (tpe_sampler, False), (tpe_sampler, True)):
        vals = [search(sampler, use_prune, budget, seed=s) for s in range(40)]
        row.append(float(np.mean(vals)))
    results[budget] = row
    print(f"{budget:>8} {row[0]:>18.5f} {row[1]:>16.5f} {row[2]:>15.5f} "
          f"{row[3]:>13.5f}")

print("\nRead the two effects separately at each budget:")
for budget, row in results.items():
    prune_gain = (row[0] - row[1]) / max(row[0], 1e-12)
    samp_gain = (row[0] - row[2]) / max(row[0], 1e-12)
    both = (row[0] - row[3]) / max(row[0], 1e-12)
    print(f"  budget {budget:>5}: pruning alone {prune_gain:>+7.1%}   "
          f"TPE alone {samp_gain:>+7.1%}   both {both:>+7.1%}")

print("\nThree things in those numbers, and the folk story gets one of them")
print("right.")
print("\nAt the SMALLEST budget neither component helps much. Both need")
print("history: the median pruner cannot rank a trial until several have")
print("reached the same rung, and TPE cannot fit its densities until it has")
print("a handful of scores. A search too short for either is just random")
print("search, and that is fine — random search is a strong baseline.")
print("\nAt the larger budgets the two effects are COMPARABLE in size. The")
print("common claim that model-based sampling is the modern answer, and the")
print("counter-claim that it is all early stopping, are both wrong here:")
print("neither dominates.")
print("\nAnd they COMPOSE SUPER-ADDITIVELY — the combination beats the sum of")
print("the two separate gains. That is not a coincidence, and the mechanism")
print("is worth knowing: pruning multiplies how many configurations the")
print("search can afford to look at, and every one of those cheap looks")
print("becomes an observation TPE can fit its densities to. The pruner does")
print("not merely save time; it feeds the sampler.")
print("\nThe practical reading: use both, and do not expect either alone to")
print("account for the improvement.")

# --- section 6.4: the search's own optimism ---------------------------------
print("\n" + "=" * 72)
print("the reported best is the luckiest, not the best (eq. 44.12)")
print("=" * 72)


def search_with_reported(budget, seed, resource=81):
    """Return (reported validation score, true loss) of the winner.

    Evaluated at full resource, where the not-yet-converged gap is
    negligible (0.6 * exp(-81/12) = 0.0007). That matters: at a smaller
    resource the gap would add a constant offset to every reported score and
    swamp the effect being measured, which is SELECTION noise alone.
    """
    rs = np.random.default_rng(seed)
    obj = Objective(seed=seed)
    best = (None, np.inf)
    n = 0
    while obj.spent < budget:
        phi = rs.uniform(0, 1, 6)
        s = obj.evaluate(phi, resource, rs)
        n += 1
        if s < best[1]:
            best = (phi, s)
    return best[1], obj.final(best[0]), n


print(f"{'trials':>8} {'reported score':>16} {'true loss':>11} "
      f"{'overstatement':>14}")
for budget in (405, 810, 2430, 8100, 24300):
    rep, true, n = [], [], []
    for s in range(60):
        r, t, k = search_with_reported(budget, s)
        rep.append(r)
        true.append(t)
        n.append(k)
    print(f"{np.mean(n):>8.0f} {np.mean(rep):>16.5f} {np.mean(true):>11.5f} "
          f"{np.mean(true) - np.mean(rep):>14.5f}")

print("\nThe reported score is systematically BELOW the winner's true loss,")
print("at every budget. It is the minimum of many noisy evaluations, so it")
print("captures the downward noise as well as the genuine quality — the")
print("configuration that got the luckiest draw is the one that gets")
print("reported.")
print("\nBoth columns improve with more trials, because the search really is")
print("finding better configurations. But the overstatement GROWS — it")
print("roughly sextuples from 5 trials to 300 — because it is the price of")
print("taking a minimum over an increasing number of noisy numbers, exactly")
print("as eq. 44.12 says.")
print("\nThe reported score is worth staring at: past thirty trials it goes")
print("NEGATIVE, and the objective is a sum of squares whose smallest")
print("possible value is zero. The search is reporting an impossible score.")
print("That is as clean a demonstration as exists that a search's best")
print("number is not a measurement of anything — and in a real project the")
print("floor is unknown, so nothing flags it.")
print("\nThe fix costs one extra fit and is almost never done: re-evaluate")
print("the chosen configuration on a fresh split and report THAT. A search")
print("selects a configuration; it does not measure one.")
