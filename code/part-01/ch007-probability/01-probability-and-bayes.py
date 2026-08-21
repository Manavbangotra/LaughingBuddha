# Extracted from: Chapter 7 — Probability, Conditional Probability, and Bayes' Theorem
# Source: src/.../ch007-probability.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Probability axioms, conditioning, independence, and Bayes — all checked
numerically against simulation rather than asserted.
"""
import numpy as np

rng = np.random.default_rng(0)

# --- the two-children puzzle, by enumeration and by simulation --------------
space = ["GG", "GB", "BG", "BB"]
at_least_one_girl = [s for s in space if "G" in s]
both_girls = [s for s in at_least_one_girl if s == "GG"]
print(f"enumeration : P(both girls | at least one girl) = "
      f"{len(both_girls)}/{len(at_least_one_girl)} = "
      f"{len(both_girls)/len(at_least_one_girl):.4f}")

trials = rng.integers(0, 2, size=(400_000, 2))       # 0 = boy, 1 = girl
has_girl = trials.sum(axis=1) >= 1
both = trials.sum(axis=1) == 2
print(f"simulation  : {both.sum() / has_girl.sum():.4f}   <- 1/3, not 1/2")
assert abs(both.sum() / has_girl.sum() - 1/3) < 0.01

# --- independence vs mutual exclusivity -------------------------------------
# Two fair dice. A = "first is 6", B = "second is 6" -> independent.
# C = "sum is 2",  D = "sum is 12"  -> mutually exclusive, hence DEPENDENT.
d1 = rng.integers(1, 7, size=300_000)
d2 = rng.integers(1, 7, size=300_000)
A, B = d1 == 6, d2 == 6
C, D = (d1 + d2) == 2, (d1 + d2) == 12

print(f"\nP(A)P(B) = {A.mean() * B.mean():.5f}  vs  P(A and B) = "
      f"{(A & B).mean():.5f}   -> independent")
print(f"P(C)P(D) = {C.mean() * D.mean():.7f}  vs  P(C and D) = "
      f"{(C & D).mean():.7f}   -> mutually exclusive means DEPENDENT")

# --- eq. 7.13: Bayes, and the base-rate fallacy -----------------------------
def bayes(prior, sensitivity, false_positive_rate):
    """P(D | +) from the prior, P(+|D), and P(+|not D)."""
    evidence = sensitivity * prior + false_positive_rate * (1 - prior)
    return sensitivity * prior / evidence


prior, sens, fpr = 0.001, 0.99, 0.05
post = bayes(prior, sens, fpr)
print(f"\nP(disease) = {prior}, sensitivity = {sens}, "
      f"false-positive rate = {fpr}")
print(f"P(disease | positive) = {post:.4f}  ({post:.2%})")
assert abs(post - 0.0194) < 0.0005

# The contingency table of table 7.1, from a simulated population.
N = 1_000_000
sick = rng.random(N) < prior
positive = np.where(sick, rng.random(N) < sens, rng.random(N) < fpr)
print(f"\nsimulated population of {N:,}:")
print(f"  sick and positive     : {int((sick & positive).sum()):>7,}")
print(f"  healthy and positive  : {int((~sick & positive).sum()):>7,}")
print(f"  P(sick | positive)    : {(sick & positive).sum()/positive.sum():.4f}")

# --- eq. 7.15: sequential updating ------------------------------------------
print("\nrepeated independent positive tests:")
p = prior
for k in range(1, 6):
    p = bayes(p, sens, fpr)
    print(f"  after {k} positive test(s): {p:.4f}  ({p:.1%})")

# --- eq. 7.16: the odds form -------------------------------------------------
prior_odds = prior / (1 - prior)
lr = sens / fpr
posterior_odds = lr * prior_odds
print(f"\nprior odds      : {prior_odds:.6f}  (about 1 in {1/prior_odds:.0f})")
print(f"likelihood ratio: {lr:.2f}   <- how much one test multiplies your odds")
print(f"posterior odds  : {posterior_odds:.6f}")
print(f"back to probability: {posterior_odds/(1+posterior_odds):.4f}  <- matches")
assert np.isclose(posterior_odds / (1 + posterior_odds), post)

# --- how the answer depends on the base rate --------------------------------
print(f"\n{'base rate':>12} {'P(disease | positive)':>22}")
for br in (0.0001, 0.001, 0.01, 0.1, 0.5):
    print(f"{br:>12.4f} {bayes(br, sens, fpr):>21.1%}")
print("The test never changed. Only the prior did.")

# --- eq. 7.9: the chain rule, and why language models use logs --------------
seq_probs = rng.uniform(0.05, 0.6, size=1500)       # per-token probabilities
print(f"\nproduct of 1500 token probabilities: {np.prod(seq_probs)}  <- underflow")
total_logp = np.sum(np.log(seq_probs))
print(f"sum of their logs                  : {total_logp:.2f}")
print(f"average log-prob per token         : {total_logp/len(seq_probs):.4f}")
print(f"perplexity = exp(-avg log-prob)    : "
      f"{np.exp(-total_logp/len(seq_probs)):.2f}")
