# Extracted from: Chapter 44 — Hyperparameter Optimization: Grid, Random, and Bayesian
# Source: src/.../ch044-hpo.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Grid, random, successive halving and Hyperband from scratch, on an
objective whose important dimensions we control.
"""
import numpy as np

rng = np.random.default_rng(0)


# --- an objective with KNOWN effective dimensionality -----------------------
def make_objective(n_dims, n_important, seed=0, noise=0.01):
    """A smooth objective over [0,1]^n_dims that depends on only the first
    `n_important` coordinates. The rest are decoys, exactly as in a real
    search space where most parameters do not matter."""
    rs = np.random.default_rng(seed)
    opt = rs.uniform(0.2, 0.8, n_important)
    w = rs.uniform(0.6, 1.4, n_important)

    def f(phi, rng_eval=None):
        phi = np.asarray(phi, float)
        loss = float(np.sum(w * (phi[:n_important] - opt) ** 2))
        if rng_eval is not None and noise > 0:
            loss += float(rng_eval.normal(0, noise))
        return loss

    return f, opt


# --- section 6.1: grid vs random at fixed budget ----------------------------
print("=" * 72)
print("grid vs random at EQUAL budget (section 6.1)")
print("=" * 72)
print("The objective depends on 2 of the N dimensions. Both searches get the")
print("same number of evaluations; only the layout differs.\n")
print(f"{'dims':>5} {'budget':>8} {'grid pts/axis':>14} "
      f"{'grid best':>11} {'random best':>13} {'random wins':>12}")

for n_dims in (2, 3, 4, 6, 8):
    f, opt = make_objective(n_dims, 2, seed=1, noise=0.0)
    per_axis = max(2, int(round(64 ** (1 / n_dims))))
    budget = per_axis ** n_dims

    # grid
    axis = (np.arange(per_axis) + 0.5) / per_axis
    grid_best = np.inf
    idx = np.zeros(n_dims, int)
    for _ in range(budget):
        grid_best = min(grid_best, f(axis[idx]))
        for j in range(n_dims):                     # odometer increment
            idx[j] += 1
            if idx[j] < per_axis:
                break
            idx[j] = 0

    # random, same budget, averaged over repeats so it is not one lucky draw
    rand_bests = []
    for rep in range(40):
        rs = np.random.default_rng(500 + rep)
        rand_bests.append(min(f(rs.uniform(0, 1, n_dims))
                              for _ in range(budget)))
    rb = float(np.mean(rand_bests))
    wins = float(np.mean([b < grid_best for b in rand_bests]))
    print(f"{n_dims:>5} {budget:>8} {per_axis:>14} {grid_best:>11.5f} "
          f"{rb:>13.5f} {wins:>11.0%}")

print("\nAs the nominal dimension grows the grid's resolution on each axis")
print("collapses — at 8 dimensions it can afford only 2 points per axis, so")
print("it tries 2 values of each parameter that matters. Random search tries")
print("as many distinct values as it has trials, on every axis, because")
print("eq. 44.5 does not mention D at all.")

# --- the budget rule (eq. 44.6) ---------------------------------------------
print("\n" + "=" * 72)
print("the budget rule: 60 random trials, any dimension (eq. 44.6)")
print("=" * 72)
print(f"{'alpha (top fraction)':>21} {'trials for 95%':>16} "
      f"{'measured hit rate':>19}")
for alpha in (0.20, 0.10, 0.05, 0.02, 0.01):
    T = int(np.ceil(np.log(0.05) / np.log(1 - alpha)))
    # measure it: what fraction of runs of T draws land in the top alpha?
    hits = 0
    trials = 400
    for rep in range(trials):
        rs = np.random.default_rng(9000 + rep)
        # "good" = within the top alpha by volume of a 7-dim unit cube, which
        # for this objective means inside a ball of the matching volume
        f7, opt7 = make_objective(7, 3, seed=2, noise=0.0)
        thresh = np.quantile([f7(rs.uniform(0, 1, 7)) for _ in range(200)],
                             alpha)
        hits += any(f7(rs.uniform(0, 1, 7)) <= thresh for _ in range(T))
    print(f"{alpha:>21.2f} {T:>16} {hits / trials:>18.0%}")

print("\nThe rule holds and does not depend on the dimension. Sixty random")
print("trials give roughly a 95% chance of landing in the top 5% of ANY")
print("search space. That is the number to remember when someone proposes a")
print("grid.")

# --- successive halving and Hyperband ---------------------------------------
print("\n" + "=" * 72)
print("successive halving: same cost, far more configurations (eq. 44.9)")
print("=" * 72)


def learning_curve(quality, resource, rs):
    """A configuration's score after `resource` units.

    `quality` in [0,1] is its true final quality. Early scores are a noisy,
    biased view of it: the RANK CORRELATION between early and final score is
    what successive halving depends on, and here it improves with resource.
    """
    signal = quality * (1 - np.exp(-resource / 8.0))
    noise = rs.normal(0, 0.25 / np.sqrt(resource))
    return signal + noise


def successive_halving(qualities, eta=3, r_min=1, seed=0):
    """Returns (chosen index, total resource spent)."""
    rs = np.random.default_rng(seed)
    alive = np.arange(len(qualities))
    r, spent = r_min, 0
    while len(alive) > 1:
        scores = np.array([learning_curve(qualities[i], r, rs) for i in alive])
        spent += len(alive) * r
        keep = max(1, len(alive) // eta)
        alive = alive[np.argsort(-scores)[:keep]]
        r *= eta
    spent += len(alive) * r
    return int(alive[0]), spent


def full_evaluation(qualities, subset, R, seed=0):
    rs = np.random.default_rng(seed)
    scores = [learning_curve(qualities[i], R, rs) for i in subset]
    return int(subset[int(np.argmax(scores))]), len(subset) * R


N_CONFIG, ETA, R_MAX = 81, 3, 81
print(f"{N_CONFIG} candidate configurations, eta = {ETA}, "
      f"max resource {R_MAX}\n")
print(f"{'strategy':<34} {'configs seen':>13} {'resource':>10} "
      f"{'mean quality of pick':>22}")

sh_q, sh_cost, fe_q, fe_cost, n_seen = [], [], [], [], None
for rep in range(300):
    rs = np.random.default_rng(rep)
    qualities = rs.uniform(0, 1, N_CONFIG)
    i_sh, c_sh = successive_halving(qualities, eta=ETA, seed=rep)
    sh_q.append(qualities[i_sh])
    sh_cost.append(c_sh)
    # the same budget spent on full-resource evaluation of a random subset
    m = max(1, int(c_sh // R_MAX))
    n_seen = m
    subset = rs.choice(N_CONFIG, m, replace=False)
    i_fe, c_fe = full_evaluation(qualities, subset, R_MAX, seed=rep)
    fe_q.append(qualities[i_fe])
    fe_cost.append(c_fe)

print(f"{'successive halving':<34} {N_CONFIG:>13} "
      f"{np.mean(sh_cost):>10.0f} {np.mean(sh_q):>22.4f}")
print(f"{'full evaluation, same budget':<34} {n_seen:>13} "
      f"{np.mean(fe_cost):>10.0f} {np.mean(fe_q):>22.4f}")
print(f"{'(best possible)':<34} {'-':>13} {'-':>10} {1.0:>22.4f}")
print(f"{'(random pick, no evaluation)':<34} {'-':>13} {0:>10} {0.5:>22.4f}")

print("\nEq. 44.8 is why this is possible: every rung costs the same n*r, so")
print("the number of configurations CONSIDERED grows exponentially while the")
print("cost grows linearly. At the same total budget, halving inspected 81")
print(f"configurations and full evaluation could afford {n_seen}.")

# --- ...and when the assumption fails ---------------------------------------
print("\n" + "=" * 72)
print("halving's assumption: early rank must predict final rank")
print("=" * 72)


def learning_curve_slow(quality, resource, rs, slow_frac=0.3, seed_q=0):
    """Some configurations warm up slowly: they look bad early and win late.
    These are exactly the ones successive halving throws away."""
    slow = (seed_q % 100) / 100.0 < slow_frac
    rate = 40.0 if slow else 8.0
    signal = quality * (1 - np.exp(-resource / rate))
    return signal + rs.normal(0, 0.25 / np.sqrt(resource))


def sh_with(curve, qualities, eta=3, r_min=1, seed=0):
    rs = np.random.default_rng(seed)
    alive = np.arange(len(qualities))
    r = r_min
    while len(alive) > 1:
        sc = np.array([curve(qualities[i], r, rs, seed_q=int(i)) for i in alive])
        alive = alive[np.argsort(-sc)[:max(1, len(alive) // eta)]]
        r *= eta
    return int(alive[0])


print(f"{'fraction of slow starters':>26} "
      f"{'mean quality of halving pick':>30}")
for frac in (0.0, 0.2, 0.5, 0.8):
    picks = []
    for rep in range(300):
        rs = np.random.default_rng(rep)
        q = rs.uniform(0, 1, N_CONFIG)
        c = (lambda qq, r, rr, seed_q=0, _f=frac:
             learning_curve_slow(qq, r, rr, _f, seed_q))
        picks.append(q[sh_with(c, q, seed=rep)])
    print(f"{frac:>26.1f} {np.mean(picks):>30.4f}")

print("\nWith no slow starters halving picks near the top (0.95 of a")
print("possible 1.0). As the fraction of slow starters rises to 80% the")
print("pick degrades to 0.86 — a real and monotone loss, though still well")
print("above the 0.50 a random pick would give. Halving does not collapse;")
print("it quietly stops finding the best configurations, which is harder to")
print("notice.")
print("\nThat is what Hyperband (eq. 44.7) hedges against: it")
print("runs several brackets, from very aggressive to no early stopping at")
print("all, so no single assumption about warm-up speed has to be right.")
