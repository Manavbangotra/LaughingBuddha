# -*- coding: utf-8 -*-
# Extracted from: Chapter 208 — Observability: Logging, Metrics, and Tracing
# Source: src/.../ch208-observability.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Uniform sampling answers questions about the distribution, not about the failures.

Tracing is sampled because storing everything is expensive, and the sampling is usually
uniform because that gives an unbiased view of the distribution. For latency work that is
exactly right: you want to know the shape, and a random tenth of requests describes it.

For attribution you do not want the shape. You want the specific requests that went
wrong, and they are rare -- so a uniform sample of a rare event contains almost none of it
(eq:uniform-sampling-misses-rare-failures).

The fix is to sample non-uniformly, keeping what looks suspicious. That requires a
suspicion signal available AT REQUEST TIME, and this listing measures what happens when
the signal is imperfect -- which ch:sd-architecture says it must be.
"""
TRAFFIC_PER_DAY = 1.4e6
ERR_RATE = 0.04
RATES = [0.001, 0.005, 0.02, 0.10, 1.00]

print("A service at %.1f million requests a day with a %.0f%% semantic error rate."
      % (TRAFFIC_PER_DAY / 1e6, ERR_RATE * 100))
print("That is %.0f wrong answers a day." % (TRAFFIC_PER_DAY * ERR_RATE))
print()
print("Under uniform sampling, how many bad requests are captured.")
print()
print(f"{'sample rate':>13}{'traces kept':>14}{'bad ones kept':>16}"
      f"{'share of bad':>14}{'days for 200':>15}")
print("-" * 74)
uni = {}
for r in RATES:
    kept = TRAFFIC_PER_DAY * r
    bad = kept * ERR_RATE
    uni[r] = (kept, bad, r, 200.0 / bad if bad > 0 else float("inf"))
    print(f"{r:>13.1%}{kept:>14,.0f}{bad:>16,.0f}{r:>14.1%}"
          f"{200.0 / bad:>15.2f}")

print()
print("The 'share of bad' column is the sample rate. Uniform sampling captures")
print("the same fraction of failures as of everything, by construction.")

print()
print()
print("Now bias the sampling toward requests a signal flags as suspicious.")
print("The signal has a recall and a false-positive rate.")
print()
BUDGET = 0.005          # keep 0.5% of traces, spent however we like
print(f"trace budget: {BUDGET:.1%} of traffic = {TRAFFIC_PER_DAY * BUDGET:,.0f} a day")
print()
print(f"{'signal recall':>15}{'signal FPR':>12}{'flagged/day':>14}"
       f"{'bad in sample':>16}{'vs uniform':>13}")
print("-" * 72)
bias = {}
uniform_bad = TRAFFIC_PER_DAY * BUDGET * ERR_RATE
for rec, fpr in ((1.00, 0.000), (0.80, 0.004), (0.55, 0.012),
                 (0.30, 0.030), (0.10, 0.060)):
    n_bad = TRAFFIC_PER_DAY * ERR_RATE
    n_good = TRAFFIC_PER_DAY * (1 - ERR_RATE)
    flagged_bad = n_bad * rec
    flagged_good = n_good * fpr
    flagged = flagged_bad + flagged_good
    # Spend the budget on flagged requests first.
    keep = TRAFFIC_PER_DAY * BUDGET
    if flagged <= keep:
        captured_bad = flagged_bad
    else:
        captured_bad = flagged_bad * (keep / flagged)
    bias[rec] = (flagged, captured_bad, captured_bad / uniform_bad)
    print(f"{rec:>15.0%}{fpr:>12.1%}{flagged:>14,.0f}"
          f"{captured_bad:>16,.0f}{captured_bad / uniform_bad:>12.1f}x")

print()
print()
print("What that buys in investigation terms: days to accumulate 200 examples")
print("of a failure mode, which is roughly what a pattern needs.")
print()
print(f"{'strategy':>26}{'bad/day':>12}{'days for 200':>15}"
       f"{'days for 1000':>16}")
print("-" * 70)
print(f"{'uniform at 0.5%':>26}{uniform_bad:>12,.0f}"
      f"{200.0 / uniform_bad:>15.1f}{1000.0 / uniform_bad:>16.1f}")
for rec in (0.80, 0.55, 0.30):
    b = bias[rec][1]
    print(f"{('biased, %.0f%% recall' % (rec * 100)):>26}{b:>12,.0f}"
          f"{200.0 / b:>15.1f}{1000.0 / b:>16.1f}")

print()
print()
print("But the bias has a cost the uniform sample does not: it can only find")
print("failures the signal recognises. What it misses, it misses completely.")
print()
MODES = [
    ("schema violation",        0.21, 0.97),
    ("refusal when it should not", 0.14, 0.88),
    ("tool call malformed",     0.11, 0.94),
    ("confidently wrong fact",  0.27, 0.19),
    ("subtly wrong reasoning",  0.18, 0.08),
    ("right but unhelpful",     0.09, 0.04),
]
print(f"{'failure mode':>30}{'share of failures':>20}{'signal recall':>16}"
      f"{'in biased sample':>19}")
print("-" * 86)
covered = 0.0
for name, share, rec in MODES:
    covered += share * rec
    print(f"{name:>30}{share:>20.0%}{rec:>16.0%}"
          f"{share * rec:>19.1%}")
print("-" * 86)
print(f"{'TOTAL':>30}{1.0:>20.0%}{covered:>16.0%}{covered:>19.1%}")

print()
print()
print("How each strategy REPRESENTS the failure modes -- the composition of the")
print("sample, against the true composition of failures.")
print()
denom = sum(sh * rc for _, sh, rc in MODES)
print(f"{'failure mode':>30}{'true share':>13}{'in uniform':>13}"
      f"{'in biased':>12}{'distortion':>13}")
print("-" * 84)
comp = {}
for name, share, rec in MODES:
    in_bias = share * rec / denom
    comp[name] = (share, share, in_bias, in_bias / share)
    print(f"{name:>30}{share:>13.0%}{share:>13.0%}{in_bias:>12.0%}"
          f"{in_bias / share:>12.1f}x")

print()
print("Uniform sampling reproduces the true composition exactly. Biased sampling")
print("reproduces the signal's recall profile instead.")

print()
print()
print("What that does to a team reading counts off the sample.")
print()
print(f"{'failure mode':>30}{'true failures/day':>20}"
      f"{'implied by biased sample':>27}{'error':>11}")
print("-" * 90)
TOTAL_BAD = TRAFFIC_PER_DAY * ERR_RATE
implied = {}
for name, share, rec in MODES:
    true_n = TOTAL_BAD * share
    seen_share = share * rec / denom
    imp = TOTAL_BAD * seen_share
    implied[name] = (true_n, imp, imp / true_n)
    print(f"{name:>30}{true_n:>20,.0f}{imp:>27,.0f}"
          f"{imp / true_n:>10.1f}x")

print()
print()
print("And the strategy that covers both: split the budget.")
print()
print(f"{'split to biased':>17}{'schema/day':>13}{'subtle/day':>13}"
      f"{'subtle share of sample':>25}{'distortion':>13}")
print("-" * 84)
for split in (1.00, 0.80, 0.50, 0.20, 0.00):
    b_budget = TRAFFIC_PER_DAY * BUDGET * split
    u_budget = TRAFFIC_PER_DAY * BUDGET * (1 - split)
    flagged_all = sum(TOTAL_BAD * sh * rc for _, sh, rc in MODES) \
        + TRAFFIC_PER_DAY * (1 - ERR_RATE) * 0.004
    scale = min(1.0, b_budget / flagged_all) if flagged_all > 0 else 0.0
    schema = TOTAL_BAD * 0.21 * 0.97 * scale + u_budget * ERR_RATE * 0.21
    subtle = TOTAL_BAD * 0.18 * 0.08 * scale + u_budget * ERR_RATE * 0.18
    tot = sum(TOTAL_BAD * sh * rc * scale + u_budget * ERR_RATE * sh
              for _, sh, rc in MODES)
    sh_subtle = subtle / tot if tot > 0 else 0.0
    print(f"{split:>17.0%}{schema:>13,.0f}{subtle:>13,.0f}"
          f"{sh_subtle:>25.1%}{sh_subtle / 0.18:>12.2f}x")

print(f"""
The uniform table states the problem in one column. Sampling at {0.005:.1%} captures
{0.005:.1%} of the bad requests -- **by construction**, because a uniform sample is
uniform (eq:uniform-sampling-misses-rare-failures). That gives
{uni[0.005][1]:,.0f} bad traces a day, spread across every failure mode.

Biased sampling looks like an unambiguous win. At {0.80:.0%} recall it captures
{bias[0.8][1]:,.0f} bad traces a day against uniform's {uniform_bad:,.0f} --
**{bias[0.8][2]:.0f} times more** for the same storage budget, and two hundred examples
in {200.0 / bias[0.8][1]:.2f} days rather than {200.0 / uniform_bad:.1f}.

It is a win on volume. The composition table is where the cost appears.

A signal can only flag what it recognises. Schema violations are recognisable at
{0.97:.0%}; a confidently wrong fact at {0.19:.0%}; subtly wrong reasoning at
{0.08:.0%}. So the biased sample's composition is not the failure distribution -- **it is
the signal's recall profile**.

Schema violations are {0.21:.0%} of real failures and
{comp['schema violation'][2]:.0%} of the biased sample, over-represented
{comp['schema violation'][3]:.1f} times. Subtly wrong reasoning is {0.18:.0%} of real
failures and {comp['subtly wrong reasoning'][2]:.0%} of the sample, under-represented
{1 / comp['subtly wrong reasoning'][3]:.0f}-fold.

**Uniform sampling reproduces the true composition exactly; biased sampling reproduces
the detector's blind spots.**

The implied-counts table is why that matters operationally rather than
philosophically. A team reading failure counts off a biased sample sees
{implied['schema violation'][1]:,.0f} schema violations a day against a true
{implied['schema violation'][0]:,.0f}, and {implied['subtly wrong reasoning'][1]:,.0f}
subtle reasoning failures against a true {implied['subtly wrong reasoning'][0]:,.0f}.

They will conclude that schema validation is the pressing problem and reasoning quality
is a minor one. **That conclusion is an artefact of how they sampled**, and nothing in
the data will contradict it, because the evidence that would has not been collected.

This is ch:sd-architecture's missing instrument arriving one level up. There the problem
was that semantic failure has no detector. Here the problem is that **building a detector
and sampling by it makes the undetected failures statistically invisible** -- worse than
before, because now there is a confident-looking distribution to point at.

The split table is the practical answer and it is unsatisfying in an honest way. At a
{1.0:.0%} biased split, subtle failures are {0.029:.1%} of the sample against a true
{0.18:.0%}. At a {0.50:.0%} split they are closer, and at {0.0:.0%} the composition is
exact and the volume is {uniform_bad:,.0f} a day.

**Keep a uniform stratum, always.** It is the only view of the failures your detectors
cannot see, its volume is low, and its value is entirely in what it can discover rather
than in what it can investigate. A team that switched wholly to intelligent sampling has
optimised its ability to study problems it already knows about and given up its ability
to find new ones -- a trade that never appears in a metric, because the thing given up
does not show up until after it is gone.""")
