# -*- coding: utf-8 -*-
# Extracted from: Chapter 237 — Continual, Online, and Self-Improving Systems
# Source: src/.../ch237-continual.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A self-improvement loop improves the part it can check, and shrinks the part it can see.

Generate answers, keep the ones a verifier accepts, train on those, repeat (cite:zelikman2022star,
cite:wang2023selfinstruct). It works, and both of its failure modes are structural rather than
incidental.

The first: only the verifiable fraction of the task improves. Whatever the verifier cannot
adjudicate receives no signal at all and stays exactly where it started, so the loop's ceiling is
set by verifiability rather than by the number of rounds
(eq:self-training-improves-only-the-verifiable-fraction).

The second: each round trains on the previous round's outputs, so the training distribution
drifts toward the model's own. Diversity contracts geometrically, at a rate set by the fraction
of genuinely external data mixed back in
(eq:collapse-rate-is-set-by-the-real-data-fraction).
"""
V_FRACTION = 0.72        # share of the task a verifier can adjudicate
P0 = 0.550               # starting competence
SENS, SPEC, RHO = 0.95, 0.90, 0.35    # verifier sensitivity, specificity, shared-error rate
LAMBDA = 0.55            # how much of the accepted-set precision a round absorbs
KAPPA = 0.93             # diversity retained when training on own outputs
ROUNDS = 8


def precision(p):
    """Share of the accepted set that is actually correct."""
    accept_wrong = RHO + (1.0 - RHO) * (1.0 - SPEC)
    return SENS * p / (SENS * p + accept_wrong * (1.0 - p))


def run(real_fraction, rounds=ROUNDS, v=V_FRACTION):
    pv, cov = P0, 1.0
    out = []
    for r in range(rounds + 1):
        p = v * pv + (1.0 - v) * P0
        out.append((r, pv, p, cov, p * cov))
        pv = pv + LAMBDA * (precision(pv) - pv)
        cov = cov * (real_fraction + (1.0 - real_fraction) * KAPPA)
    return out


print("Eight rounds of a self-improvement loop.")
print()
REAL = 0.05
print(f"verifiable fraction {V_FRACTION:.2f}, real data mixed back in {REAL:.0%}")
print()
print(f"{'round':>8}{'verifiable competence':>24}{'overall competence':>21}"
      f"{'diversity retained':>21}{'net capability':>17}")
print("-" * 91)
base = run(REAL)
for r, pv, p, cov, net in base:
    print(f"{r:>8}{pv:>24.4f}{p:>21.4f}{cov:>21.4f}{net:>17.4f}")

PEAK = max(base, key=lambda x: x[4])
print()
print(f"peak net capability at round {PEAK[0]}: {PEAK[4]:.4f}")
print(f"round 0: {base[0][4]:.4f}; round {ROUNDS}: {base[-1][4]:.4f}")
print(f"the loop gains {PEAK[4] - base[0][4]:+.4f}, then gives back"
      f" {PEAK[4] - base[-1][4]:.4f}")

print()
print()
print("The ceiling is verifiability, and no number of rounds moves it.")
print()
print(f"{'verifiable fraction':>21}{'competence ceiling':>21}{'peak net':>12}"
      f"{'peak round':>13}{'gain over round 0':>21}")
print("-" * 88)
ceil = {}
for v in (0.20, 0.45, 0.72, 0.90, 1.00):
    series = run(REAL, rounds=40, v=v)
    top = max(series, key=lambda x: x[4])
    limit = v * 1.0 + (1.0 - v) * P0
    ceil[v] = (limit, top[4], top[0])
    print(f"{v:>21.2f}{limit:>21.4f}{top[4]:>12.4f}{top[0]:>13}"
          f"{top[4] - series[0][4]:>21.4f}")

print()
print(f"at verifiable fraction {0.20:.2f} the ceiling is {ceil[0.20][0]:.4f};"
      f" at {1.00:.2f} it is {ceil[1.00][0]:.4f}")
print("(eq:self-training-improves-only-the-verifiable-fraction)")

print()
print()
print("And diversity contracts at a rate set by the external data fraction.")
print()
print(f"{'real data mixed in':>21}{'diversity at round 8':>23}{'peak net':>12}"
      f"{'peak round':>13}{'net at round 20':>19}")
print("-" * 88)
mix = {}
for r_frac in (0.00, 0.05, 0.15, 0.35, 0.70, 1.00):
    series = run(r_frac, rounds=20)
    top = max(series, key=lambda x: x[4])
    mix[r_frac] = (series[8][3], top[4], top[0], series[20][4])
    print(f"{r_frac:>21.0%}{series[8][3]:>23.4f}{top[4]:>12.4f}{top[0]:>13}"
          f"{series[20][4]:>19.4f}")

print()
print(f"with no external data, diversity at round 8 is {mix[0.00][0]:.4f}"
      f" and net at round 20 is {mix[0.00][3]:.4f}")
print(f"with {0.35:.0%} external data: {mix[0.35][0]:.4f} and {mix[0.35][3]:.4f}")
print("(eq:collapse-rate-is-set-by-the-real-data-fraction)")

print()
print()
print("What that means task by task.")
print()
TASKS = [
    ("arithmetic with a checker",         0.99, "an exact answer"),
    ("code with a test suite",            0.86, "tests are partial"),
    ("factual QA with retrieval",         0.71, "sources may disagree"),
    ("summarisation",                     0.38, "no reference exists"),
    ("open-ended advice",                 0.19, "the criterion is contested"),
    ("creative writing",                  0.08, "quality is a preference"),
]
print(f"{'task':>32}{'verifiable fraction':>22}{'loop ceiling':>15}"
      f"{'gain available':>17}{'why':>30}")
print("-" * 116)
for name, v, why in TASKS:
    limit = v + (1.0 - v) * P0
    print(f"{name:>32}{v:>22.2f}{limit:>15.4f}{limit - P0:>17.4f}{why:>30}")

print()
print(f"the spread of available gain across these tasks is"
      f" {(0.99 + 0.01 * P0 - P0) / (0.08 + 0.92 * P0 - P0):.1f}x")

print()
print()
print("And what the loop costs to run at each round.")
print()
GEN_COST, TRAIN_COST = 0.00038, 61_000.0
ITEMS = 400_000
print(f"{'round':>8}{'items generated':>18}{'accepted':>12}{'generation $':>15}"
      f"{'training $':>14}{'net gain':>12}{'$ per 0.001 gain':>19}")
print("-" * 98)
prev_net = None
total = 0.0
for r, pv, p, cov, net in base[:6]:
    kept = ITEMS * (SENS * pv + (RHO + (1 - RHO) * (1 - SPEC)) * (1 - pv))
    gen = ITEMS * 6 * GEN_COST
    cost = gen + TRAIN_COST
    total += cost
    if prev_net is None:
        print(f"{r:>8}{'--':>18}{'--':>12}{'--':>15}{'--':>14}"
              f"{'--':>12}{'--':>19}")
    else:
        d = net - prev_net
        per = cost / (d / 0.001) if d > 0 else float("inf")
        ps = f"{per:>19,.0f}" if d > 0 else f"{'negative':>19}"
        print(f"{r:>8}{ITEMS:>18,}{kept:>12,.0f}{gen:>15,.0f}"
              f"{TRAIN_COST:>14,.0f}{d:>+12.4f}{ps}")
    prev_net = net

print()
print(f"total for {5} rounds: {total:,.0f}")

print(f"""
The first table is a self-improvement loop behaving exactly as advertised, and then not. Net
capability rises from {base[0][4]:.4f} to a peak of **{PEAK[4]:.4f} at round {PEAK[0]}**, then
falls to {base[-1][4]:.4f} by round {ROUNDS}.

Two separate things are happening in those columns and they are worth reading apart. Verifiable
competence rises monotonically -- the loop genuinely works on the part the verifier can judge.
Diversity falls monotonically, because each round trains on the last round's outputs. The net is
their product, and a rising bounded factor times a falling one has a peak.

**A self-improvement loop is not a process that converges. It is a process with an optimal number
of rounds**, and running it longer is not conservative.

The ceiling table says how high the rising factor can go, and the answer is not "1". Competence
converges to `v + (1-v)p0` -- the verifiable fraction improves, the rest stays exactly where it
started (eq:self-training-improves-only-the-verifiable-fraction). At a verifiable fraction of
{0.20:.2f} the ceiling is {ceil[0.20][0]:.4f}; at {0.90:.2f} it is {ceil[0.90][0]:.4f}.

That is the same structure ch:res-test-time found for sampling and it is worth stating together:
**both ways of using a verifier to improve a model are bounded by the verifier, in different
ways.** Sampling is bounded by how well the verifier ranks; self-training is bounded by how much
of the task it can adjudicate at all.

The mixture table is the second failure mode
(eq:collapse-rate-is-set-by-the-real-data-fraction). With no external data, diversity at round 8
is {mix[0.00][0]:.4f} and net capability at round 20 has fallen to {mix[0.00][3]:.4f}. With
{0.35:.0%} external data, {mix[0.35][0]:.4f} and {mix[0.35][3]:.4f}.

Read the mechanism rather than the rows. Diversity is multiplied each round by
`r + (1-r)*kappa`, which is below 1 for every `r` short of {1.00:.0%}. **The external-data
fraction sets the rate of contraction, not a floor** -- mixing in real data buys rounds, and only
data that is entirely external avoids collapse in the limit
(eq:collapse-rate-is-set-by-the-real-data-fraction).

The {1.00:.0%} row is the control, and it is not a self-improvement loop at all: it is ordinary
training on external data, and it is the only row that improves monotonically to round 20.

That reframes what the technique is for. It is a way to convert compute into capability *for a
bounded number of rounds* when external data is scarce -- not a way to escape needing it. The
right quantity to control is the mixture ratio and the right thing to monitor is diversity
itself, and neither appears on a standard training dashboard.

The task table converts this into a decision. `arithmetic with a checker` has a verifiable
fraction of {0.99:.2f} and a ceiling of {0.99 + 0.01 * P0:.4f}. `creative writing` has
{0.08:.2f} and {0.08 + 0.92 * P0:.4f}. The available gain differs by
**{(0.99 + 0.01 * P0 - P0) / (0.08 + 0.92 * P0 - P0):.1f}x** across the same technique.

**Self-improvement is a technique for verifiable tasks and a distribution-narrowing procedure
for everything else**, and the same code produces both outcomes.

The cost table is the last practical point. The gain per round falls sharply while the cost per
round is constant, so the dollars per unit of improvement rise every round -- and eventually the
gain turns negative while the bill does not.

**The loop has a stopping rule and it is not "when it stops improving"**, because by the time the
measured metric turns over, diversity has already contracted past the point where the earlier
rounds could be reproduced. The stopping rule has to be set in advance from the ceiling and the
mixture ratio, which are both computable before the first round runs.""")
