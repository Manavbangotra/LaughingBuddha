# -*- coding: utf-8 -*-
# Extracted from: Chapter 96 — Hallucination: Causes, Taxonomy, and Mitigation
# Source: src/.../ch096-hallucination.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Risk against coverage. Equation (eq:risk-coverage), measured."""
import math

import numpy as np

rng = np.random.default_rng(0)
N = 20_000
BASE_ACCURACY = 0.82


def make_system(auc):
    """Answers with a given accuracy, plus a confidence signal of a given
    ranking quality. AUC 0.5 is useless, 1.0 is perfect."""
    correct = rng.random(N) < BASE_ACCURACY
    # Separation chosen to hit the target AUC for two Gaussians.
    sep = math.sqrt(2) * _probit(auc)
    conf = rng.normal(np.where(correct, sep, 0.0), 1.0)
    return correct, conf


def _probit(p):
    """Inverse normal CDF by bisection — no scipy dependency."""
    lo, hi = -6.0, 6.0
    for _ in range(80):
        mid = (lo + hi) / 2
        cdf = 0.5 * (1 + math.erf(mid / math.sqrt(2)))
        lo, hi = (mid, hi) if cdf < p else (lo, mid)
    return (lo + hi) / 2


def risk_at_coverage(correct, conf, coverage):
    """Answer the top `coverage` fraction by confidence; report the error rate."""
    k = max(int(coverage * len(conf)), 1)
    idx = np.argsort(-conf)[:k]
    return float(1 - correct[idx].mean())


print(f"base accuracy {BASE_ACCURACY:.0%}, so risk at full coverage is "
      f"{1 - BASE_ACCURACY:.0%}\n")
print(f"{'coverage':>10} " + " ".join(f"{'AUC ' + str(a):>10}"
                                       for a in (0.5, 0.7, 0.8, 0.9, 0.99)))
systems = {a: make_system(a) for a in (0.5, 0.7, 0.8, 0.9, 0.99)}
for cov in (1.0, 0.8, 0.6, 0.4, 0.2, 0.1):
    row = " ".join(f"{risk_at_coverage(*systems[a], cov):>10.3f}"
                   for a in (0.5, 0.7, 0.8, 0.9, 0.99))
    print(f"{cov:>10.0%} {row}")

print("""
The AUC=0.5 column is flat: a useless confidence signal means abstaining buys
nothing, because the questions you decline are no worse than the ones you keep.
Every other column falls with coverage, and the rate it falls at IS the value of
the confidence signal.""")

# The product question: what coverage does a risk budget buy?
TARGET_RISK = 0.05
print(f"\nrisk budget {TARGET_RISK:.0%} — what coverage is achievable?\n")
print(f"{'AUC':>6} {'max coverage':>14} {'questions answered':>20}")
for a in (0.5, 0.7, 0.8, 0.9, 0.99):
    correct, conf = systems[a]
    best = 0.0
    for cov in np.linspace(0.02, 1.0, 99):
        if risk_at_coverage(correct, conf, cov) <= TARGET_RISK:
            best = cov
    print(f"{a:>6.2f} {best:>13.0%} {int(best * N):>20,}")

print("""
This is the table to take to a product discussion, and the AUC column is the
one to read. A useless signal answers NOTHING within a 5% budget — every
question it would keep is as likely to be wrong as one it would decline. At AUC
0.8 the system answers 44% of questions; at 0.9 it answers 72%.

Note what that implies about where to invest. The model's accuracy is 82% in
every row — only the confidence signal changed, and it moved usable coverage
from 0% to 85%. Improving the SIGNAL raises coverage at fixed risk, and it is
frequently cheaper than improving accuracy: a well-calibrated 82% model is worth
far more here than a poorly-calibrated 86% one.""")
