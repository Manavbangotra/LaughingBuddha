# -*- coding: utf-8 -*-
# Extracted from: Chapter 79 — What Foundation Models Are
# Source: src/.../ch079-what-they-are.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""How a defect in one base propagates to everything adapted from it."""
import numpy as np

rng = np.random.default_rng(0)

N_APPS, N_PROBES = 200, 2000
BASE_DEFECT_RATE = 0.04          # 4% of probes trigger a base-model defect
ADAPT_FIX_RATE = 0.25            # adaptation happens to fix a quarter of them
ADAPT_OWN_DEFECT = 0.01          # and introduces its own, independently


def simulate(n_bases):
    """N_APPS applications spread over n_bases distinct base models."""
    base_of = rng.integers(0, n_bases, N_APPS)
    base_defects = rng.random((n_bases, N_PROBES)) < BASE_DEFECT_RATE

    fails = np.zeros((N_APPS, N_PROBES), dtype=bool)
    for a in range(N_APPS):
        inherited = base_defects[base_of[a]] & (rng.random(N_PROBES) > ADAPT_FIX_RATE)
        own = rng.random(N_PROBES) < ADAPT_OWN_DEFECT
        fails[a] = inherited | own

    per_app = fails.mean(1).mean()
    # The quantity that matters for systemic risk: given one app fails on a
    # probe, how much of the ecosystem fails on that same probe?
    hit = fails.sum(0)
    correlated = float(hit[hit > 0].mean() / N_APPS)
    worst = float(hit.max() / N_APPS)
    return per_app, correlated, worst


print(f"{N_APPS} applications, {N_PROBES} probes, "
      f"base defect rate {BASE_DEFECT_RATE:.0%}\n")
print(f"{'distinct bases':>15} {'per-app failure':>17} "
      f"{'mean co-failure':>17} {'worst probe':>13}")
for n_bases in (1, 2, 5, 20, 200):
    per_app, corr, worst = simulate(n_bases)
    print(f"{n_bases:>15} {per_app:>17.3f} {corr:>17.1%} {worst:>13.1%}")

print("""
Read the first column against the last.

Per-app failure barely moves: about 4% however many bases exist, because each
application's reliability is dominated by its own adaptation. An audit that
samples one application cannot distinguish these worlds at all.

The worst-probe column is where homogenisation lives. On one shared base, the
single worst input takes down ~82% of the ecosystem simultaneously; spread over
two hundred bases, the same per-application reliability caps the worst input at
~13%. Mean co-failure moves much less, because it averages over the many probes
that trip only one or two applications — the tail is the risk, not the mean.

That is equation (eq:homogenisation) as a measurement: homogenisation does not
make any individual system worse, it makes the whole system fail TOGETHER, and
only a tail statistic can see it.""")
