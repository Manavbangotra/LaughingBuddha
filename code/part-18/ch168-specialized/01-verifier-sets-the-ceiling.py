# -*- coding: utf-8 -*-
# Extracted from: Chapter 168 — Specialized Agents: Research, Coding, Data, Browser, Computer-Use
# Source: src/.../ch168-specialized.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""What actually separates a coding agent from a browser agent.

The usual account is difficulty: coding is 'easier' for agents than computer-use
because the tasks are more structured. This listing proposes a different variable
and measures which one predicts success.

Every domain differs in whether the agent can CHECK its own work, and how well:

  coding        a compiler and a test suite -- cheap, fast, near-perfect
  data          schema and type checks -- cheap, partial
  research      other sources agreeing -- expensive, weak
  browser       the page changed, somehow -- cheap, very weak
  computer-use  a screenshot -- cheap, very weak

ch:ag-recovery's retry needs a verifier to retry AGAINST. Without one, a retry is
a fresh sample rather than a correction (eq:retry-needs-a-verifier), so a domain's
ceiling may be set by its verifier rather than by its difficulty
(eq:verifier-sets-the-ceiling).
"""
import numpy as np

rng = np.random.default_rng(3719)

M = 40000
STEPS = 12
RETRIES = 4

# (name, per-step success, verifier detects a bad step, verifier false alarm)
DOMAINS = [
    ("coding",       0.82, 0.97, 0.02),
    ("data",         0.86, 0.80, 0.06),
    ("research",     0.90, 0.45, 0.15),
    ("browser",      0.78, 0.30, 0.10),
    ("computer-use", 0.72, 0.25, 0.12),
]


def run(p_step, p_detect, p_fa, m=M, steps=STEPS, retries=RETRIES):
    """Each step succeeds with p_step. A verifier flags bad steps with p_detect
    and good ones with p_fa. A flagged step is retried, up to `retries` times.
    An undetected bad step is carried forward and the task is wrong."""
    ok = np.ones(m, dtype=bool)
    cost = np.zeros(m, dtype=np.int64)
    for _ in range(steps):
        live = np.flatnonzero(ok)
        if not len(live):
            break
        good = rng.random(len(live)) < p_step
        cost[live] += 1
        # Retry loop: only fires on a flagged step.
        for _ in range(retries):
            bad = ~good
            flagged = np.where(bad, rng.random(len(live)) < p_detect,
                               rng.random(len(live)) < p_fa)
            redo = flagged
            if not redo.any():
                break
            cost[live[redo]] += 1
            fresh = rng.random(int(redo.sum())) < p_step
            good = good.copy()
            good[redo] = fresh
        ok[live[~good]] = False
    return float(ok.mean()), float(cost.mean())


print(f"{M:,} tasks of {STEPS} steps, up to {RETRIES} retries per step.")
print("Per-step success is the domain's raw difficulty; detection is how well")
print("the domain lets an agent tell a bad step from a good one.")
print()
print(f"{'domain':>14}{'step success':>14}{'detection':>11}{'task success':>14}"
      f"{'steps':>8}")
print("-" * 61)
tab = {}
for name, ps, pd, pf in DOMAINS:
    r = run(ps, pd, pf)
    tab[name] = (ps, pd, r[0], r[1])
    print(f"{name:>14}{ps:>14.0%}{pd:>11.0%}{r[0]:>14.1%}{r[1]:>8.1f}")

print()
print()
print("Which variable predicts the outcome? Correlations across the five domains:")
print()
xs = np.array([v[0] for v in tab.values()])
ds = np.array([v[1] for v in tab.values()])
ys = np.array([v[2] for v in tab.values()])
c_diff = float(np.corrcoef(xs, ys)[0, 1])
c_ver = float(np.corrcoef(ds, ys)[0, 1])
print(f"{'per-step success (difficulty) vs task success':>50}{c_diff:>10.2f}")
print(f"{'verifier detection vs task success':>50}{c_ver:>10.2f}")

print()
print()
print("The controlled version: hold difficulty fixed and sweep the verifier,")
print("then hold the verifier fixed and sweep difficulty.")
print()
print(f"{'detection':>11}{'task success':>14}      {'step success':>14}"
      f"{'task success':>14}")
print("-" * 69)
sweep_v, sweep_d = {}, {}
DET = (0.25, 0.45, 0.70, 0.90, 0.97)
DIF = (0.72, 0.78, 0.82, 0.86, 0.90)
for pd, ps in zip(DET, DIF):
    a = run(0.80, pd, 0.08)[0]
    b = run(ps, 0.60, 0.08)[0]
    sweep_v[pd] = a
    sweep_d[ps] = b
    print(f"{pd:>11.0%}{a:>14.1%}      {ps:>14.0%}{b:>14.1%}")

print()
print()
print("And what retries are worth with and without a verifier -- the mechanism")
print("behind the whole table.")
print()
print(f"{'retries':>9}{'detection 97%':>16}{'detection 60%':>16}"
      f"{'detection 25%':>16}")
print("-" * 57)
rt = {}
for k in (0, 1, 2, 4, 8):
    row = tuple(run(0.80, pd, 0.08, retries=k)[0] for pd in (0.97, 0.60, 0.25))
    rt[k] = row
    print(f"{k:>9}{row[0]:>16.1%}{row[1]:>16.1%}{row[2]:>16.1%}")

print(f"""
Compare the coding and research rows, which is the whole listing in two lines.

Research has the HIGHER per-step success -- {tab['research'][0]:.0%} against coding's
{tab['coding'][0]:.0%} -- and the lower task success:
{tab['research'][2]:.1%} against {tab['coding'][2]:.1%}. On the variable everyone
reaches for, research is the easier domain, and it loses by
{tab['coding'][2] - tab['research'][2]:.1f} points.

The difference is the other column. A coding agent has a compiler and a test suite:
it can tell a bad step from a good one {tab['coding'][1]:.0%} of the time. A research
agent has other sources agreeing, at {tab['research'][1]:.0%}.

The correlations make it quantitative. Across the five domains, per-step difficulty
correlates {c_diff:.2f} with task success and verifier detection correlates
{c_ver:.2f}. **A domain's ceiling is set more by whether the agent can check its own
work than by how hard the work is** (eq:verifier-sets-the-ceiling).

Five points is a small sample and the profiles are hand-set, so the controlled
sweep matters more. Holding difficulty at {0.80:.0%} and moving detection from
{0.25:.0%} to {0.97:.0%} moves task success from {sweep_v[0.25]:.1%} to
{sweep_v[0.97]:.1%}. Holding detection at {0.60:.0%} and moving difficulty across
its full observed range moves it from {sweep_d[0.72]:.1%} to {sweep_d[0.90]:.1%}.

Both matter. The verifier range is wider, and -- more usefully -- **the verifier is
the one you can build.** Per-step difficulty is a property of the domain and the
model. Detection is a property of the tooling you wrap around it, and a test suite
is something a team can write this week.

The last table shows the mechanism, and it is ch:ag-recovery's with a condition
attached. At {0.97:.0%} detection, going from {0} to {4} retries is worth
{rt[4][0] - rt[0][0]:+.1%}. At {0.25:.0%} detection the same retries are worth
{rt[4][2] - rt[0][2]:+.1%}.

**A retry without a verifier is a fresh sample rather than a correction**
(eq:retry-needs-a-verifier). It cannot preferentially re-run the steps that went
wrong, so it re-runs everything at the same rate and buys far less. Every
reliability mechanism in part:17 that depends on noticing a failure inherits this
condition, which is why the domains at the bottom of the first table are hard in a
way that more retries do not fix.

So the practical reading of 'specialising an agent for a domain' is narrower than it
sounds. It is mostly not prompt engineering and mostly not model choice. **It is
building the domain's verifier**, and where the domain does not offer one cheaply,
that is the work.""")
