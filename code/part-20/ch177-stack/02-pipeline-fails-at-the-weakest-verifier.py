# -*- coding: utf-8 -*-
# Extracted from: Chapter 177 — The AI-Assisted Data Science Stack
# Source: src/.../ch177-stack.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Errors in an analysis pipeline, and where a check is worth putting.

An analysis is a chain: access, clean, explore, feature, model, validate,
conclude. An error at any stage flows downstream, and the stages after it produce
confident, well-formed output built on it. A model trained on mis-joined data
converges; its validation score is real; the conclusion is wrong.

Two things vary by stage and decide everything:

  error rate       how often that stage gets it wrong
  verifier         whether anything can TELL, which ch:as-specialized found sets
                   a domain's ceiling and which differs sharply by stage

The end of the pipeline has the best verifier -- a held-out score is a real
number -- and the least ability to use it, because by then the error is a premise
that everything agrees with (eq:pipeline-fails-at-the-weakest-verifier).
"""
import numpy as np

rng = np.random.default_rng(4457)

M = 60000

# (stage, per-run error rate, detection rate of the check available there,
#  cost of a check in analyst-hours)
STAGES = [
    ("access",     0.10, 0.90, 0.4),   # row counts, schema, freshness: strong
    ("clean",      0.22, 0.45, 1.2),   # some checks; most errors are plausible
    ("explore",    0.14, 0.15, 0.8),   # almost nothing to check against
    ("feature",    0.16, 0.35, 1.0),   # leakage checks catch some
    ("model",      0.09, 0.80, 0.5),   # a held-out score is a real number
    ("validate",   0.07, 0.55, 0.6),   # checks the model, not the premises
    ("conclude",   0.12, 0.10, 0.3),   # no reference answer at all
]
N = len(STAGES)
DECAY = 0.62        # ch:as-failures: detection falls as an error ages downstream


def run(checks, m=M, decay=DECAY, fix=0.85):
    """`checks` is a set of stage indices where a check is performed. Returns
    (correct conclusions, checks run, analyst-hours spent)."""
    err_at = np.full(m, -1, dtype=np.int64)     # -1 = no outstanding error
    hours = np.zeros(m)
    n_checks = 0
    for i, (name, p_err, detect, cost) in enumerate(STAGES):
        # This stage may introduce an error, if one is not already loose.
        fresh = (err_at < 0) & (rng.random(m) < p_err)
        err_at[fresh] = i
        if i in checks:
            n_checks += 1
            hours += cost
            live = err_at >= 0
            lag = np.where(live, i - err_at, 0)
            # A check at stage i uses stage i's verifier, and its power decays
            # with how long the error has been propagating.
            p = detect * (decay ** np.clip(lag, 0, None))
            caught = live & (rng.random(m) < p) & (rng.random(m) < fix)
            err_at[caught] = -1
    return float((err_at < 0).mean()), n_checks, float(hours.mean())


print(f"{M:,} analyses through {N} stages. An error at any stage propagates,")
print("and every stage after it produces work that agrees with it.")
print()
print(f"{'stage':>12}{'error rate':>12}{'detection here':>16}{'hours':>8}")
print("-" * 48)
for name, p_err, detect, cost in STAGES:
    print(f"{name:>12}{p_err:>12.0%}{detect:>16.0%}{cost:>8.1f}")

print()
print()
print("No checks at all, then the check placements teams actually use.")
print()
PLANS = [
    ("none", set()),
    ("at the end (validate)", {5}),
    ("end pair (model+validate)", {4, 5}),
    ("early pair (access+clean)", {0, 1}),
    ("spread three", {0, 2, 4}),
    ("every stage", set(range(N))),
]
print(f"{'placement':>28}{'correct':>10}{'checks':>9}{'hours':>8}"
      f"{'correct/hour':>14}")
print("-" * 69)
tab = {}
for label, ck in PLANS:
    r = run(ck)
    tab[label] = r
    cell = "--" if r[2] <= 0 else f"{(r[0] - tab['none'][0]) / r[2]:.3f}"
    print(f"{label:>28}{r[0]:>10.1%}{r[1]:>9}{r[2]:>8.1f}{cell:>14}")

print()
print()
print("Two checks, placed every possible way. The best and worst pairs:")
print()
pairs = {}
for i in range(N):
    for j in range(i + 1, N):
        pairs[(i, j)] = run({i, j})[0]
order = sorted(pairs, key=lambda k: -pairs[k])
print(f"{'pair':>26}{'correct':>10}")
print("-" * 36)
for k in order[:3]:
    print(f"{f'{STAGES[k[0]][0]} + {STAGES[k[1]][0]}':>26}{pairs[k]:>10.1%}")
print(f"{'...':>26}")
for k in order[-2:]:
    print(f"{f'{STAGES[k[0]][0]} + {STAGES[k[1]][0]}':>26}{pairs[k]:>10.1%}")

print()
print()
print("What each single check is worth, placed alone -- which separates a")
print("stage's own detection power from how much damage is upstream of it.")
print()
base = run(set())[0]
print(f"{'check at':>12}{'correct':>10}{'gain':>9}{'gain/hour':>12}")
print("-" * 43)
single = {}
for i, (name, p_err, detect, cost) in enumerate(STAGES):
    r = run({i})
    single[name] = (r[0], r[0] - base, (r[0] - base) / cost)
    print(f"{name:>12}{r[0]:>10.1%}{r[0] - base:>+9.1%}"
          f"{(r[0] - base) / cost:>12.3f}")

print()
print()
print("And what a better verifier at the weakest stage would buy, against a")
print("better verifier at the stage that already has the best one.")
print()
print(f"{'intervention':>34}{'correct':>10}{'gain':>9}")
print("-" * 53)
allck = set(range(N))
base_all = run(allck)[0]
print(f"{'checks everywhere, as is':>34}{base_all:>10.1%}{'--':>9}")
imp = {}
for idx, label in ((2, "explore detection 15% -> 60%"),
                   (6, "conclude detection 10% -> 60%"),
                   (4, "model detection 80% -> 98%")):
    saved = STAGES[idx]
    STAGES[idx] = (saved[0], saved[1], 0.60 if idx != 4 else 0.98, saved[3])
    v = run(allck)[0]
    STAGES[idx] = saved
    imp[label] = (v, v - base_all)
    print(f"{label:>34}{v:>10.1%}{v - base_all:>+9.1%}")

print(f"""
The first placement table contains the result and the fourth explains it.

Checking only at the end -- "does the model validate?", which is the default
practice -- takes correctness from {tab['none'][0]:.1%} to
{tab['at the end (validate)'][0]:.1%}. Checking at every stage reaches
{tab['every stage'][0]:.1%} for {tab['every stage'][2]:.1f} analyst-hours.

The early pair, which the previous chapters' logic would recommend, gives
{tab['early pair (access+clean)'][0]:.1%} -- WORSE than the end pair's
{tab['end pair (model+validate)'][0]:.1%}, and at more than the cost.

That is not what ch:as-failures found about critics, and the difference is
instructive. There, every critic had the same detection rate and coverage decided
everything. Here **detection rates differ by a factor of nine across stages**, and
that dominates.

The single-check table shows it cleanly. A check at `model` alone is worth
{single['model'][1]:+.1%}; at `explore` alone, {single['explore'][1]:+.1%}; at
`conclude`, {single['conclude'][1]:+.1%}. Per analyst-hour the model check returns
{single['model'][2]:.3f} against {single['explore'][2]:.3f} for exploration.

The model stage has two things going for it at once: a genuine verifier -- a
held-out score is a real number -- and a position late enough that most upstream
errors have already been made and are available to catch. **Check where the
verifier is strong, not where the error is fresh**
(eq:pipeline-fails-at-the-weakest-verifier), which is the opposite of the
freshness intuition and follows directly from ch:as-specialized's ceiling result.

The pair table agrees: `model` appears in all three best pairs, and the two worst
both consist of weak-verifier stages.

But the last table reverses the advice as soon as the verifiers themselves are in
play, and this is the finding to take away.

With checks already at every stage, improving `conclude` detection from
{0.10:.0%} to {0.60:.0%} is worth {imp['conclude detection 10% -> 60%'][1]:+.1%}
and improving `explore` from {0.15:.0%} to {0.60:.0%} is worth
{imp['explore detection 15% -> 60%'][1]:+.1%}. Improving `model` from
{0.80:.0%} to {0.98:.0%} -- a larger relative gain in detector quality -- is worth
{imp['model detection 80% -> 98%'][1]:+.1%}.

So the two halves say different things and both are right:

**Given the verifiers you have, spend checking time where detection is strong.**
**Given the chance to build a verifier, build it where detection is weakest.**

The strong stages are near their ceiling and the weak ones are nowhere near it, so
the marginal return on a new verifier is inverted relative to the marginal return
on a new check. Teams routinely do the first and almost never do the second,
because building a check for "is this exploration sound" or "does this conclusion
follow" is hard and unglamorous, and adding one more model-validation metric is
neither.

Which is this part's argument in one table. The stages with no verifier are where
the time goes (the previous listing), where the errors survive (this one), and
where nobody is working.""")
