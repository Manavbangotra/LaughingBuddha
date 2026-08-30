# -*- coding: utf-8 -*-
# Extracted from: Chapter 169 — Multi-Agent Failure Modes
# Source: src/.../ch169-failures.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The independence assumption, which is what makes multi-agent estimates wrong.

Reliability for a multi-agent system is nearly always estimated the same way: each
agent is r reliable, there are k of them, so the system is r^k. That calculation
assumes the agents fail independently.

They do not. Agents in a real system share a base model, share a prompt lineage,
share a context window, and are handed the same upstream artefacts. A shared cause
makes their errors correlated (eq:agent-errors-correlate).

The listing was written expecting correlation to make everything worse. It does not:
it makes a CHAIN better and REDUNDANCY worse, because those two structures use the
independence assumption in opposite directions (eq:correlation-cuts-both-ways). The
tail behaviour is the part that should worry you.
"""
import numpy as np

rng = np.random.default_rng(3833)

M = 60000
K = 5
R = 0.85


def correlated_failures(m, k, r, rho):
    """Errors with a shared component. Each agent fails if its latent score falls
    below a threshold; the scores share a common factor of weight rho, so rho=0 is
    independence and rho=1 makes every agent fail together."""
    common = rng.standard_normal((m, 1))
    idio = rng.standard_normal((m, k))
    z = np.sqrt(rho) * common + np.sqrt(1 - rho) * idio
    from math import erf, sqrt
    # threshold such that P(fail) = 1 - r
    lo, hi = -8.0, 8.0
    for _ in range(60):
        mid = (lo + hi) / 2
        p = 0.5 * (1 + erf(mid / sqrt(2)))
        if p < 1 - r:
            lo = mid
        else:
            hi = mid
    return z < (lo + hi) / 2


def pipeline(rho, m=M, k=K, r=R):
    """A chain: every agent must be right."""
    f = correlated_failures(m, k, r, rho)
    return float((~f.any(1)).mean())


def vote(rho, m=M, k=K, r=R):
    """Redundancy: k agents answer, majority wins."""
    f = correlated_failures(m, k, r, rho)
    return float(((~f).sum(1) > k / 2).mean())


print(f"{M:,} runs, {K} agents at {R:.0%} reliability each.")
print(f"The independence estimate for a chain is {R ** K:.1%}.")
print()
print(f"{'correlation':>13}{'chain':>9}{'vs r^k':>10}{'vote of 5':>12}"
      f"{'all fail':>11}")
print("-" * 55)
tab = {}
for rho in (0.0, 0.2, 0.4, 0.7, 0.95):
    c = pipeline(rho)
    v = vote(rho)
    f = correlated_failures(M, K, R, rho)
    allf = float(f.all(1).mean())
    tab[rho] = (c, v, allf)
    print(f"{rho:>13.2f}{c:>9.1%}{c - R ** K:>+10.1%}{v:>12.1%}{allf:>11.2%}")

print()
print()
print("The same, as agent count grows. Chain reliability under independence")
print("against chain reliability at a realistic shared-cause correlation.")
print()
print(f"{'agents':>8}{'independent':>14}{'rho=0.4':>10}{'rho=0.7':>10}"
      f"{'optimism':>11}")
print("-" * 53)
gr = {}
for k in (2, 3, 5, 8, 12):
    a = pipeline(0.0, k=k)
    b = pipeline(0.4, k=k)
    c = pipeline(0.7, k=k)
    gr[k] = (a, b, c)
    print(f"{k:>8}{a:>14.1%}{b:>10.1%}{c:>10.1%}{a - c:>+11.1%}")

print()
print()
print("Voting is the standard mitigation, and it is the one correlation removes.")
print("Marginal value of redundancy, over a single agent at the same reliability:")
print()
print(f"{'correlation':>13}{'1 agent':>10}{'vote of 3':>12}{'vote of 5':>12}"
      f"{'vote of 9':>12}")
print("-" * 59)
vt = {}
for rho in (0.0, 0.3, 0.6, 0.9):
    row = (R, vote(rho, k=3), vote(rho, k=5), vote(rho, k=9))
    vt[rho] = row
    print(f"{rho:>13.2f}{row[0]:>10.1%}{row[1]:>12.1%}{row[2]:>12.1%}"
          f"{row[3]:>12.1%}")

print()
print()
print("And what diversity buys: agents that share less. Same vote of 5, with the")
print("shared component reduced by using genuinely different models or prompts.")
print()
print(f"{'shared cause':>14}{'vote of 5':>12}{'gain over 1':>14}"
      f"{'gain over rho=0.9':>20}")
print("-" * 60)
dv = {}
for rho in (0.9, 0.7, 0.5, 0.3, 0.1):
    v = vote(rho)
    dv[rho] = v
    print(f"{rho:>14.2f}{v:>12.1%}{v - R:>+14.1%}{v - vt[0.9][2]:>+20.1%}")

print(f"""
The first table's second column is not the sign the listing expected.

At correlation {0.95:.2f} the chain reaches {tab[0.95][0]:.1%} against the
independence estimate's {R ** K:.1%} -- **the r^k calculation is PESSIMISTIC by
{tab[0.95][0] - R ** K:.1f} points**, not optimistic. Positive correlation
concentrates failure into fewer runs, so more runs come through entirely clean.

That is correct probability rather than a modelling artefact, and it is worth
absorbing before the rest: for a structure where everything must go right,
correlated agents are better than independent ones.

The fourth column is where the same correlation is bad news. The probability that
ALL {K} agents fail together goes from {tab[0.0][2]:.2%} at independence to
{tab[0.95][2]:.2%} -- a factor of about
{tab[0.95][2] / max(tab[0.0][2], 1e-9):.0f}. **Correlation moves probability mass
out of the middle and into both tails**, and one of those tails is every agent in
your system being wrong the same way at the same time.

The second table shows the chain effect growing with agent count: at {12} agents,
independence predicts {gr[12][0]:.1%} and correlation {0.7:.2f} gives
{gr[12][2]:.1%}. A team that sized its pipeline using r^k built something more
reliable than it thought, for a reason it would not have been able to state.

The third table is the one that should change a design decision. Redundancy is the
standard mitigation for unreliable agents -- run several, take the majority -- and
it is the mitigation correlation removes.

At independence, a vote of {9} turns {R:.0%} into {vt[0.0][3]:.1%}, worth
{vt[0.0][3] - R:+.1%}. At correlation {0.9:.2f} the same vote of {9} gives
{vt[0.9][3]:.1%}, worth {vt[0.9][3] - R:+.1%}.

**Nine agents that share a cause are worth about one agent**
(eq:redundancy-needs-independence). The votes are not independent evidence; they are
one opinion, restated nine times, at nine times the cost.

So the independence assumption is wrong in OPPOSITE DIRECTIONS for the two things it
is used for (eq:correlation-cuts-both-ways). It understates how well a chain
performs and it drastically overstates what redundancy buys. A team using r^k for
both -- which is the normal case -- is being too pessimistic about its pipeline and
far too optimistic about its voting.

The last table says what to do about the second problem, and it is not "add more
agents". Reducing the shared cause from {0.9:.2f} to {0.1:.2f} takes a vote of {K}
from {dv[0.9]:.1%} to {dv[0.1]:.1%}, worth {dv[0.1] - dv[0.9]:+.1%} -- more than
going from three voters to nine buys at any correlation in the table.

**Diversity is the input redundancy actually consumes**, and it comes from
genuinely different models, genuinely different prompts, and genuinely different
evidence rather than from more copies. This is the mechanism behind
ch:as-multi-agent's finding that one diverse agent beat a debate panel: the panel
members were correlated, so the panel was closer to one agent than to five.""")
