# -*- coding: utf-8 -*-
# Extracted from: Chapter 160 — Termination, Budgets, and Human-in-the-Loop
# Source: src/.../ch160-termination.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Confirming everything is close to confirming nothing.

A human-in-the-loop gate is usually specified as a policy about WHICH actions need
approval, and evaluated on the assumption that an approved action was actually
reviewed. That assumption is the whole problem.

A reviewer's attention is finite and degrades with load. A person asked to approve
four actions a day reads them; the same person asked to approve four hundred clicks
through. So the catch rate is not a constant -- it is a function of how much you ask
them to look at (eq:habituation), and a gate that routes more work to the same
people buys less per item than it did before.

This listing prices five gating policies on the two things that matter: the harm
that reached production, and the human hours consumed.
"""
import numpy as np

rng = np.random.default_rng(2423)

DAYS = 400
N = 3000                # actions proposed per day
P_BAD = 0.06            # share of proposed actions that are wrong
BASE_CATCH = 0.92       # catch rate for an attentive reviewer
REVIEWERS = 3
HAB_K = 25              # reviews PER REVIEWER PER DAY at which attention halves
MINUTES = 1.5


def catch_rate(n_reviews):
    """Attention decays with each reviewer's daily load, not with the total."""
    return BASE_CATCH / (1.0 + (n_reviews / REVIEWERS) / HAB_K)


def make_day():
    bad = rng.random(N) < P_BAD
    # Blast radius: most actions are trivial, a few are not.
    harm = np.where(rng.random(N) < 0.08, rng.gamma(3.0, 6.0, N),
                    rng.gamma(1.0, 0.4, N))
    # The agent's own confidence: informative, and correlated with its errors.
    conf = np.clip(rng.normal(0.72 - 0.30 * bad, 0.18), 0, 1)
    return bad, harm, conf


def mask_for(kind, frac, harm, conf):
    if kind == "none":
        return np.zeros(N, dtype=bool)
    if kind == "all":
        return np.ones(N, dtype=bool)
    if kind == "random":
        return rng.random(N) < frac
    if kind == "harm":
        return harm >= np.quantile(harm, 1 - frac)
    if kind == "lowconf":
        return conf <= np.quantile(conf, frac)
    raise ValueError(kind)


def run(kind, frac=0.02, days=DAYS, hab=None):
    hk = HAB_K if hab is None else hab
    shipped_n = shipped_h = revs = 0.0
    catch_acc = 0.0
    for _ in range(days):
        bad, harm, conf = make_day()
        m = mask_for(kind, frac, harm, conf)
        n_rev = int(m.sum())
        c = BASE_CATCH / (1.0 + (n_rev / REVIEWERS) / hk)
        caught = m & bad & (rng.random(N) < c)
        ship = bad & ~caught
        shipped_n += ship.sum()
        shipped_h += harm[ship].sum()
        revs += n_rev
        catch_acc += c
    d = days
    return (shipped_n / (d * N), shipped_h / d, revs / d,
            revs / d * MINUTES / 60.0, catch_acc / d)


print(f"{N:,} proposed actions a day, {P_BAD:.0%} of them wrong, over {DAYS} days.")
print(f"{REVIEWERS} reviewers; an attentive one catches {BASE_CATCH:.0%}, and")
print(f"attention halves at {HAB_K} reviews per person per day.")
print(f"Each review costs {MINUTES} human minutes.")
print()
print(f"{'gate rate':>11}{'reviews/day':>13}{'catch rate':>12}"
      f"{'harm/day':>11}{'hours/day':>12}")
print("-" * 59)
sweep = {}
for g in (0.0, 0.005, 0.02, 0.05, 0.20, 1.0):
    r = run("random" if g not in (0.0, 1.0) else ("none" if g == 0 else "all"),
            frac=g)
    sweep[g] = r
    print(f"{g:>11.1%}{r[2]:>13.0f}{r[4]:>12.1%}{r[1]:>11.1f}{r[3]:>12.1f}")

print()
print()
print("Five gating policies at a fixed 2% review budget, so the comparison is")
print("about WHAT you look at rather than how much.")
print()
BF = 0.02
print(f"{'policy':>26}{'reviews/day':>13}{'catch':>8}{'harm/day':>11}"
      f"{'vs no gate':>12}")
print("-" * 70)
pol = {}
base = run("none")[1]
for name, kind in [("no gate", "none"), ("gate everything", "all"),
                   ("random 2%", "random"),
                   ("lowest-confidence 2%", "lowconf"),
                   ("highest-blast-radius 2%", "harm")]:
    r = run(kind, frac=BF)
    pol[name] = r
    print(f"{name:>26}{r[2]:>13.0f}{r[4]:>8.1%}{r[1]:>11.1f}"
          f"{r[1] - base:>+12.1f}")

print()
print()
print("Harm avoided per human hour -- the only ratio that decides the policy.")
print()
print(f"{'policy':>26}{'hours/day':>12}{'harm avoided':>15}{'per hour':>12}")
print("-" * 65)
for name, r in pol.items():
    saved = base - r[1]
    per = saved / r[3] if r[3] > 0 else float("nan")
    print(f"{name:>26}{r[3]:>12.1f}{saved:>15.1f}"
          f"{('--' if r[3] == 0 else format(per, '.2f')):>12}")

print()
print()
print("How much does habituation matter? Sweep the load at which attention")
print("halves, for gate-everything against gating the top 2% by blast radius.")
print()
print(f"{'halving load':>14}{'gate all: catch':>18}{'gate all: harm':>17}"
      f"{'top 2%: harm':>15}")
print("-" * 64)
hab = {}
for k in (10, 25, 100, 500, 10 ** 7):
    a = run("all", frac=BF, days=120, hab=k)
    b = run("harm", frac=BF, days=120, hab=k)
    hab[k] = (a, b)
    label = "none" if k > 10 ** 6 else f"{k:,}"
    print(f"{label:>14}{a[4]:>18.1%}{a[1]:>17.1f}{b[1]:>15.1f}")

print()
print()
print("And how much review is worth buying, keyed to blast radius.")
print()
print(f"{'review budget':>15}{'hours/day':>12}{'harm/day':>11}"
      f"{'harm avoided':>15}{'per hour':>11}")
print("-" * 64)
bud = {}
for f in (0.002, 0.005, 0.01, 0.02, 0.05, 0.15):
    r = run("harm", frac=f, days=200)
    bud[f] = r
    saved = base - r[1]
    print(f"{f:>15.1%}{r[3]:>12.1f}{r[1]:>11.1f}{saved:>15.1f}"
          f"{saved / max(r[3], 1e-9):>11.2f}")

print(f"""
The first table is the case against the reflex, and the two end rows are the whole
argument.

Gating nothing ships {sweep[0.0][1]:.0f} units of harm a day and costs zero human
hours. Gating EVERYTHING ships {sweep[1.0][1]:.0f} and costs
{sweep[1.0][3]:.0f} human hours a day.

Seventy-five hours -- more than nine person-days -- to avoid about
{(sweep[0.0][1] - sweep[1.0][1]) / sweep[0.0][1]:.0%} of the harm. The catch-rate
column says why: it falls from {sweep[0.0][4]:.0%} for an attentive reviewer to
{sweep[1.0][4]:.1%} when three people are asked to approve three thousand actions a
day. **They are not reviewing. They are clicking** (eq:habituation).

That is the failure mode a policy of "require human approval for agent actions"
produces at scale, and it is worse than it looks on paper, because the
organisation now believes those actions were reviewed.

The second table holds the review budget fixed at {BF:.0%} and changes only WHAT
gets looked at, which is where the leverage is.

Random {BF:.0%} avoids {base - pol['random 2%'][1]:.1f} units of harm a day.
Gating the lowest-confidence {BF:.0%} avoids
{base - pol['lowest-confidence 2%'][1]:.1f}. Gating the highest-blast-radius
{BF:.0%} avoids {base - pol['highest-blast-radius 2%'][1]:.1f}.

Same number of reviews, same reviewers, same catch rate --
{pol['highest-blast-radius 2%'][1] / pol['random 2%'][1] - 1:+.0%} difference in
harm shipped, from the selection criterion alone.

The third table is the ratio that should decide the policy, and the spread is the
result of this listing. Harm avoided per human hour:
{base - pol['gate everything'][1] and (base - pol['gate everything'][1]) / pol['gate everything'][3]:.2f} for gating everything,
{(base - pol['random 2%'][1]) / pol['random 2%'][3]:.2f} for random sampling,
{(base - pol['highest-blast-radius 2%'][1]) / pol['highest-blast-radius 2%'][3]:.2f}
for gating by blast radius.

**A factor of about five hundred between the worst policy and the best**, at
identical human cost per review. Almost nobody computes this ratio, and it is the
only number that matters for the decision.

The fourth table proves that habituation is the mechanism rather than an
assumption I baked in. Remove it entirely -- an infinitely patient reviewer -- and
gating everything ships {hab[10 ** 7][0][1]:.1f} harm against blast-radius
gating's {hab[10 ** 7][1][1]:.1f}. **With no habituation, gating everything is by
far the best policy**, exactly as intuition says it should be.

Reintroduce it and the ordering inverts. At a halving load of {25} reviews per
person per day, gate-everything ships {hab[25][0][1]:.1f} and the {BF:.0%} policy
ships {hab[25][1][1]:.1f}.

So the argument for selective gating is not that review is unhelpful. It is that
**attention is the scarce resource, and spending it uniformly is spending it on
the actions that did not need it.** If your reviewers genuinely do not habituate,
gate more; the table tells you what that assumption is worth.

The last table sizes the budget, and it turns over. Harm avoided rises from
{base - bud[0.002][1]:.1f} at a {0.002:.1%} budget to {base - bud[0.05][1]:.1f} at
{0.05:.0%}, then FALLS to {base - bud[0.15][1]:.1f} at {0.15:.0%}.

More review made things worse. The marginal action added to the queue is less
consequential than the ones already there, and adding it dilutes the attention
being paid to those -- so past a point the gate is trading a careful look at the
important items for a cursory look at more of them. **A confirmation gate has an
interior optimum**, and it is smaller than most teams set it.

Harm avoided per hour falls monotonically throughout
({(base - bud[0.002][1]) / bud[0.002][3]:.0f} down to
{(base - bud[0.15][1]) / bud[0.15][3]:.1f}), which is the number to use when
deciding where to stop.

Four rules follow, and the first two contradict standard practice.

**Do not gate on confidence alone.** It is better than random
({base - pol['lowest-confidence 2%'][1]:.1f} against
{base - pol['random 2%'][1]:.1f}) and worse than blast radius, and it inherits
ch:rsn-self-consistency's correlation -- the agent is most confident where it is
most wrong.

**Gate on consequence, and specifically on reversibility.** An action you can undo
does not need a human before it; it needs a human after it, if at all.

**Size the gate from the harm-per-hour column**, not from a policy about
categories. The optimum is interior and it is small.

**Measure your reviewers' actual catch rate under load.** Every number in this
listing turns on a curve nobody measures, and it is measurable: seed known-bad
actions into the review queue at varying volumes and count the catches.""")
