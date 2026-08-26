# -*- coding: utf-8 -*-
# Extracted from: Chapter 10 — Statistical Inference, Sampling, and Hypothesis Testing
# Source: src/.../ch010-inference.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Comparing two models on a shared test set — the right way.

Because both models see the same examples, their errors are correlated. A
paired test exploits that and is far more sensitive than treating the two
accuracy figures as independent.
"""
import numpy as np

rng = np.random.default_rng(1)

n = 1000
# Per-example difficulty, shared by both models — this is what correlates them.
difficulty = rng.random(n)
# Model B is genuinely better, by 1.5 percentage points.
correct_a = rng.random(n) < (0.94 - 0.10 * difficulty)
correct_b = rng.random(n) < (0.955 - 0.10 * difficulty)

acc_a, acc_b = correct_a.mean(), correct_b.mean()
print(f"model A accuracy: {acc_a:.4f}")
print(f"model B accuracy: {acc_b:.4f}")
print(f"observed difference: {acc_b - acc_a:+.4f}")

# --- the naive, unpaired analysis -------------------------------------------
se_a = np.sqrt(acc_a * (1 - acc_a) / n)
se_b = np.sqrt(acc_b * (1 - acc_b) / n)
z_unpaired = (acc_b - acc_a) / np.sqrt(se_a**2 + se_b**2)
print(f"\nunpaired z = {z_unpaired:.3f}  -> "
      f"{'significant' if abs(z_unpaired) > 1.96 else 'NOT significant'}")

# --- the paired analysis: McNemar's test -------------------------------------
# Only the disagreements carry information. Examples both got right, or both
# got wrong, tell you nothing about which model is better.
b_only = int(np.sum(~correct_a & correct_b))     # B right, A wrong
a_only = int(np.sum(correct_a & ~correct_b))     # A right, B wrong
both_right = int(np.sum(correct_a & correct_b))
both_wrong = int(np.sum(~correct_a & ~correct_b))

print(f"\ncontingency table:")
print(f"  both right : {both_right:>4}      (uninformative)")
print(f"  both wrong : {both_wrong:>4}      (uninformative)")
print(f"  B only     : {b_only:>4}      <- evidence for B")
print(f"  A only     : {a_only:>4}      <- evidence for A")

# Under H0 the disagreements split 50/50, so the count is Binomial(m, 0.5).
m = a_only + b_only
z_paired = (b_only - a_only) / np.sqrt(m) if m else 0.0
print(f"\npaired z = {z_paired:.3f}  -> "
      f"{'significant' if abs(z_paired) > 1.96 else 'NOT significant'}")
print(f"the paired test uses only the {m} disagreements, not all {n} examples,")
print("and is more sensitive precisely because it removes the shared difficulty.")

# --- how often does each test find a real effect? (power) -------------------
def trial(true_a=0.940, true_b=0.955, n=1000):
    diff = rng.random(n)
    ca = rng.random(n) < (true_a + 0.05 - 0.10 * diff)
    cb = rng.random(n) < (true_b + 0.05 - 0.10 * diff)
    pa, pb = ca.mean(), cb.mean()
    sa = np.sqrt(max(pa*(1-pa), 1e-12) / n)
    sb = np.sqrt(max(pb*(1-pb), 1e-12) / n)
    zu = (pb - pa) / np.sqrt(sa**2 + sb**2)
    ao = int(np.sum(ca & ~cb)); bo = int(np.sum(~ca & cb)); mm = ao + bo
    zp = (bo - ao) / np.sqrt(mm) if mm else 0.0
    return abs(zu) > 1.96, abs(zp) > 1.96


results = np.array([trial() for _ in range(3000)])
print(f"\npower over 3000 simulated experiments (the effect IS real):")
print(f"  unpaired test detects it: {results[:,0].mean():.1%}")
print(f"  paired test detects it  : {results[:,1].mean():.1%}")
print("\nSame data, same effect. The paired test finds it far more often.")
