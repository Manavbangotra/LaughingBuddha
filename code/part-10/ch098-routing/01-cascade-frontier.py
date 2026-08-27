# -*- coding: utf-8 -*-
# Extracted from: Chapter 98 — Model Routing and Model Selection
# Source: src/.../ch098-routing.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The cost-quality frontier of a cascade. Equations (eq:cascade-cost) onward."""
import math

import numpy as np

rng = np.random.default_rng(0)
N = 20_000

SMALL = dict(cost=1.0, accuracy=0.78)
LARGE = dict(cost=12.0, accuracy=0.94)


def _probit(p):
    lo, hi = -6.0, 6.0
    for _ in range(80):
        mid = (lo + hi) / 2
        cdf = 0.5 * (1 + math.erf(mid / math.sqrt(2)))
        lo, hi = (mid, hi) if cdf < p else (lo, mid)
    return (lo + hi) / 2


def make_population(auc):
    """Requests, whether the small model gets each right, and a confidence
    signal whose ranking quality is `auc`."""
    small_ok = rng.random(N) < SMALL["accuracy"]
    large_ok = rng.random(N) < LARGE["accuracy"]
    sep = math.sqrt(2) * _probit(auc)
    conf = rng.normal(np.where(small_ok, sep, 0.0), 1.0)
    return small_ok, large_ok, conf


def cascade(small_ok, large_ok, conf, escalation_rate):
    """Escalate the least-confident `escalation_rate` fraction."""
    k = int(escalation_rate * N)
    order = np.argsort(conf)          # least confident first
    escalated = np.zeros(N, dtype=bool)
    escalated[order[:k]] = True
    correct = np.where(escalated, large_ok, small_ok)
    cost = SMALL["cost"] + escalated.mean() * LARGE["cost"]  # eq:cascade-cost
    return float(correct.mean()), float(cost)


small_ok, large_ok, conf = make_population(0.82)
print(f"small: cost {SMALL['cost']:.0f}, accuracy {SMALL['accuracy']:.2f}")
print(f"large: cost {LARGE['cost']:.0f}, accuracy {LARGE['accuracy']:.2f}")
print(f"confidence signal AUC 0.82\n")

print(f"{'escalated':>10} {'accuracy':>10} {'cost':>8} {'vs always-large':>17} "
      f"{'quality kept':>14}")
for e in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
    acc, cost = cascade(small_ok, large_ok, conf, e)
    print(f"{e:>10.0%} {acc:>10.4f} {cost:>8.2f} "
          f"{cost / LARGE['cost']:>16.0%} {acc / LARGE['accuracy']:>14.1%}")

# Equation (eq:cascade-breakeven): where does the cascade stop paying?
breakeven = 1 - SMALL["cost"] / LARGE["cost"]
print(f"\nbreak-even escalation (eq:cascade-breakeven): {breakeven:.0%}")
print(f"above that, the cascade costs more than always using the large model")

# The signal's value: eq:signal-value, as the gap from the linear baseline.
print(f"\n{'AUC':>6} " + " ".join(f"{'e=' + f'{e:.0%}':>9}"
                                   for e in (0.1, 0.3, 0.5)))
for auc in (0.5, 0.7, 0.82, 0.9, 0.99):
    so, lo, cf = make_population(auc)
    row = " ".join(f"{cascade(so, lo, cf, e)[0]:>9.4f}"
                   for e in (0.1, 0.3, 0.5))
    print(f"{auc:>6.2f} {row}")

print("""
At AUC 0.5 the accuracy barely moves with escalation — a useless signal
escalates a random 30%, which is as likely to include requests the small model
would have got right as ones it would not. The rows above it bow upward, and
that bowing IS the signal's value (eq:signal-value).

Note what this says about where to invest: the models are identical in every
row. Only the routing signal changed.""")
