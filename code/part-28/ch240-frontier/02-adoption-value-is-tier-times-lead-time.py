# -*- coding: utf-8 -*-
# Extracted from: Chapter 240 — Reading the Frontier: Established, Emerging, Speculative
# Source: src/.../ch240-frontier.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A tier is not a verdict, it is an adoption policy with a break-even.

The first listing sorted claims into established, emerging and speculative. This one asks what to
do with each, and the answer is not "believe the first, doubt the third".

Adopting early buys a lead-time premium and risks rework. Waiting buys certainty and forfeits the
premium. Those cross at a specific probability, and the tier tells you which side of it you are
on (eq:adoption-value-is-tier-times-lead-time).

The other thing a tier tells you is how long the claim is likely to last, which decides how much
of a roadmap should rest on it (eq:claim-survival-falls-with-tier).
"""
V_HOLDS = 1_400_000.0     # value of the capability if the claim holds
REWORK = 520_000.0        # cost of unwinding an adoption that fails
COST_NOW = 240_000.0      # cost of adopting while the ground is moving
COST_LATER = 95_000.0     # cost of adopting once tooling has matured
LEAD = 1.55               # value multiple for being 18 months early

TIERS = [
    ("established", 0.92),
    ("emerging",    0.61),
    ("speculative", 0.24),
]


def ev_now(p):
    return p * (V_HOLDS * LEAD - COST_NOW) - (1 - p) * (REWORK + COST_NOW)


def ev_wait(p):
    return p * (V_HOLDS - COST_LATER)


print("Adopt now or wait, by tier.")
print()
print(f"{'tier':>15}{'P(claim holds)':>17}{'value if adopted now':>23}"
      f"{'value if you wait':>20}{'decision':>13}{'margin':>14}")
print("-" * 102)
dec = {}
for name, p in TIERS:
    a, b = ev_now(p), ev_wait(p)
    dec[name] = (a, b, "adopt now" if a > b else "wait")
    print(f"{name:>15}{p:>17.2f}{a:>23,.0f}{b:>20,.0f}"
          f"{dec[name][2]:>13}{abs(a - b):>14,.0f}")

BREAK = (REWORK + COST_NOW) / (V_HOLDS * LEAD - COST_NOW + REWORK + COST_NOW
                               - (V_HOLDS - COST_LATER))
print()
print(f"the two lines cross at P = {BREAK:.3f}")
print(f"which sits between `emerging` at {0.61:.2f} and `speculative` at {0.24:.2f}")
print("(eq:adoption-value-is-tier-times-lead-time)")

print()
print()
print("How the break-even moves with the lead-time premium.")
print()
print(f"{'lead-time premium':>20}{'break-even P':>15}{'adopt established?':>21}"
      f"{'adopt emerging?':>18}{'adopt speculative?':>21}")
print("-" * 95)
for lead in (1.00, 1.20, 1.55, 2.00, 3.00):
    num = REWORK + COST_NOW
    den = V_HOLDS * lead - COST_NOW + REWORK + COST_NOW - (V_HOLDS - COST_LATER)
    be = num / den
    row = f"{lead:>20.2f}{be:>15.3f}"
    for name, p in TIERS:
        row += f"{('yes' if p > be else 'no'):>{21 if name == 'established' else (18 if name == 'emerging' else 21)}}"
    print(row)

print()
print("A fast-moving market lowers the bar; a slow one raises it.")

print()
print()
print("And how long a claim at each tier lasts.")
print()
SURVIVE = {
    "established": (0.97, 0.94, 0.89, 0.81),
    "emerging":    (0.78, 0.61, 0.38, 0.22),
    "speculative": (0.44, 0.24, 0.09, 0.03),
}
print(f"{'tier':>15}{'1 year':>10}{'2 years':>11}{'5 years':>11}{'10 years':>12}"
      f"{'half-life (years)':>21}")
print("-" * 80)
half = {}
for name, p in TIERS:
    s = SURVIVE[name]
    h = None
    for yrs, v in zip((1, 2, 5, 10), s):
        if v < 0.5 and h is None:
            h = yrs
    half[name] = h if h else 10
    hs = f"{half[name]:>20}+" if h is None else f"{half[name]:>21}"
    print(f"{name:>15}{s[0]:>10.2f}{s[1]:>11.2f}{s[2]:>11.2f}{s[3]:>12.2f}{hs}")

print()
print(f"an `emerging` claim is more likely than not to be gone within"
      f" {half['emerging']} years")
print(f"a `speculative` one within {half['speculative']}")
print("(eq:claim-survival-falls-with-tier)")

print()
print()
print("So how much of a roadmap should rest on each tier?")
print()
HORIZON = 5
print(f"planning horizon {HORIZON} years")
print()
# an established capability is table stakes; a speculative one, if it holds, is a
# differentiator -- and there are only so many established opportunities to take
TIER_VALUE = {"established": 0.55, "emerging": 1.00, "speculative": 1.80}
AVAILABLE = {"established": 5.0, "emerging": 5.0, "speculative": 12.0}
print(f"{'allocation':>28}{'est':>7}{'emg':>7}{'spc':>7}{'slots used':>13}"
      f"{'expected value':>17}{'expected rework':>18}{'net':>15}")
print("-" * 112)
ALLOCS = [
    ("everything established",        1.00, 0.00, 0.00),
    ("mostly established",            0.70, 0.25, 0.05),
    ("balanced",                      0.50, 0.35, 0.15),
    ("chase the frontier",            0.20, 0.40, 0.40),
    ("everything speculative",        0.00, 0.00, 1.00),
]
BUDGET = 12
alloc = {}
for name, a, b, c in ALLOCS:
    val, rew, used = 0.0, 0.0, 0.0
    for share, (tname, p) in zip((a, b, c), TIERS):
        n = min(BUDGET * share, AVAILABLE[tname])
        used += n
        surv = SURVIVE[tname][2]
        val += n * surv * V_HOLDS * TIER_VALUE[tname]
        rew += n * (1 - surv) * REWORK
    alloc[name] = (val - rew, rew, used)
    print(f"{name:>28}{a:>7.0%}{b:>7.0%}{c:>7.0%}{used:>13.1f}"
          f"{val:>17,.0f}{rew:>18,.0f}{val - rew:>15,.0f}")

BEST_A = max(alloc, key=lambda n: alloc[n][0])
print()
print(f"best net over {HORIZON} years: {BEST_A} at {alloc[BEST_A][0]:,.0f}")
print(f"`everything established` uses only {alloc['everything established'][2]:.0f}"
      f" of {BUDGET} slots -- there are not enough settled opportunities")
print(f"`chase the frontier` nets {alloc['chase the frontier'][0]:,.0f}"
      f" with {alloc['chase the frontier'][1]:,.0f} of rework")

print()
print()
print("What tells you a claim is about to move, before the field notices.")
print()
SIGNALS = [
    ("a failed replication, anywhere",     0.79, 3,   "demotion"),
    ("the effect shrinks in later papers", 0.71, 9,   "demotion"),
    ("nobody has tried to ablate it",      0.44, 0,   "stuck"),
    ("a second lab reproduces it",         0.74, 6,   "promotion"),
    ("it ships and survives a year",       0.83, 12,  "promotion"),
    ("the benchmark it used is retired",   0.58, 4,   "unknown"),
]
print(f"{'signal':>38}{'predictive of a move':>23}{'lead (months)':>16}"
      f"{'direction':>13}")
print("-" * 90)
for name, pred, lead_m, direction in SIGNALS:
    print(f"{name:>38}{pred:>23.2f}{lead_m:>16}{direction:>13}")

best_sig = max(SIGNALS, key=lambda s: s[1])
print()
print(f"strongest single signal: {best_sig[0]} at {best_sig[1]:.2f},"
      f" {best_sig[2]} months ahead")
print(f"the earliest: {min(SIGNALS, key=lambda s: -s[2])[0]}")

print(f"""
The decision table is the point of having tiers at all. For an `established` claim, adopting now
is worth {dec['established'][0]:,.0f} against {dec['established'][1]:,.0f} for waiting -- a
margin of {abs(dec['established'][0] - dec['established'][1]):,.0f}. For `emerging`,
{dec['emerging'][0]:,.0f} against {dec['emerging'][1]:,.0f}. For `speculative`,
**{dec['speculative'][0]:,.0f} against {dec['speculative'][1]:,.0f}** -- and the sign is what
matters: adopting a speculative claim has *negative* expected value here.

The lines cross at **P = {BREAK:.3f}** (eq:adoption-value-is-tier-times-lead-time), which sits
between `emerging` at {0.61:.2f} and `speculative` at {0.24:.2f}. **The tier boundary that
matters is not established-versus-emerging; it is emerging-versus-speculative**, and the
established tier is not where the decision lives at all.

The lead-time table says the break-even is not a constant. At a premium of {1.00:.2f} -- a slow
market where being early is worth nothing -- the break-even is above 1 and **nothing clears it**:
waiting always wins, because the only thing adopting early buys is risk. At {1.20:.2f} the bar is
{0.849:.3f} and only established claims clear it. At {3.00:.2f} it drops to {0.223:.3f} and even
speculative ones do.

**How fast your market moves decides your evidence standard**, which is uncomfortable and
correct. It is also the honest explanation for why research labs and regulated industries adopt
at different tiers without either being wrong.

The survival table is the other thing a tier tells you. An `established` claim is still standing
{SURVIVE['established'][2]:.0%} of the time after five years. An `emerging` one,
{SURVIVE['emerging'][2]:.0%}. A `speculative` one, **{SURVIVE['speculative'][2]:.0%}**
(eq:claim-survival-falls-with-tier).

An emerging claim is more likely than not to be gone within {half['emerging']} years and a
speculative one within {half['speculative']}. That is not a reason to ignore them -- it is the
reason the adoption decision has a rework term.

The allocation table puts the two together over a five-year horizon, with two facts the earlier
tables did not carry: an established capability is table stakes and worth less when it lands, and
**there are only so many established opportunities to take**.

`{BEST_A}` nets {alloc[BEST_A][0]:,.0f}. `everything established` uses only
{alloc['everything established'][2]:.0f} of {BUDGET} slots and nets
{alloc['everything established'][0]:,.0f} -- it runs out of settled things to build.
`chase the frontier` nets {alloc['chase the frontier'][0]:,.0f} against
{alloc['chase the frontier'][1]:,.0f} of expected rework, and `everything speculative` nets
{alloc['everything speculative'][0]:,.0f}.

**The optimum is a mixture and it is interior**, for the same reason every portfolio in this book
has been. A roadmap made entirely of settled things forfeits the premium and runs out of
material; one made entirely of frontier work pays for rework it never recovers.

The signals table is how to update between tiers without waiting for the field. `it ships and
survives a year` predicts a promotion at {0.83:.2f} with {12} months of lead;
`a failed replication, anywhere` predicts a demotion at {0.79:.2f} with only {3}.

The asymmetry there is worth carrying. **Promotion signals are slow and demotion signals are
fast**, so a claim you adopted on emerging evidence will usually tell you it is failing before it
tells you it is safe -- and the right response to a failed replication anywhere is to re-price
the roadmap item that afternoon, not to wait for consensus.

That is the whole of reading the frontier, and it is deliberately unromantic. Score the evidence,
locate the break-even, size the exposure, and watch the demotion signals. None of it requires
knowing which claims are true -- which is the only honest position available about work that is,
by construction, not yet settled.""")
