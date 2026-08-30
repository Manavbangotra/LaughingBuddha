# -*- coding: utf-8 -*-
# Extracted from: Chapter 230 — Privacy, Data Governance, and Copyright
# Source: src/.../ch230-privacy.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A privacy budget is a budget. It is spent by every query and it does not refill.

cite:abadi2016dpsgd made differential privacy practical for deep learning by tracking the
privacy loss across training steps rather than bounding each one separately. That accounting is
the important part: **epsilon composes**, so the guarantee a dataset carries after twenty
analyses is not the guarantee any one of them provided
(eq:privacy-budget-composes-across-queries).

The second thing worth being concrete about is what an epsilon *means*. It bounds how much an
adversary's odds about any individual can move after seeing the output
(eq:epsilon-bounds-the-posterior-shift) -- which is a specific, checkable claim, and it is much
weaker at the epsilons people actually deploy than the word "private" suggests.
"""
import math

print("What an epsilon bounds: the multiplicative shift in an adversary's odds.")
print()
print(f"{'epsilon':>10}{'max odds ratio':>17}{'prior 1%  ->  at most':>24}"
      f"{'prior 50% -> at most':>23}{'reading':>22}")
print("-" * 96)
eps_tab = {}
for e in (0.1, 0.5, 1.0, 3.0, 8.0, 20.0):
    r = math.exp(e)
    p1 = 0.01 * r / (1 - 0.01 + 0.01 * r)
    p50 = 0.5 * r / (0.5 + 0.5 * r)
    eps_tab[e] = (r, p1, p50)
    reading = ("very strong" if e <= 0.5 else "strong" if e <= 1 else
               "moderate" if e <= 3 else "weak" if e <= 8 else "nominal")
    print(f"{e:>10.1f}{r:>17.1f}{p1:>24.1%}{p50:>23.1%}{reading:>22}")

print()
print(f"at epsilon {8.0:.0f}, a 1% prior can become {eps_tab[8.0][1]:.0%}")
print("which is a bound, not a promise that it stays at 1%")

print()
print()
print("Composition: the same dataset, analysed repeatedly.")
print()
PER_QUERY = 0.5
DELTA = 1e-5
print(f"{'queries':>10}{'basic composition':>20}{'advanced composition':>23}"
      f"{'odds ratio (log10)':>21}")
print("-" * 74)
comp = {}
for k in (1, 5, 20, 100, 500):
    basic = k * PER_QUERY
    adv = (math.sqrt(2 * k * math.log(1 / DELTA)) * PER_QUERY
           + k * PER_QUERY * (math.exp(PER_QUERY) - 1))
    comp[k] = (basic, adv, adv / math.log(10.0))
    print(f"{k:>10}{basic:>20.1f}{adv:>23.1f}{adv / math.log(10.0):>21.1f}")

print()
print(f"after {20} queries at epsilon {PER_QUERY:.1f} each, the dataset carries")
print(f"epsilon {comp[20][1]:.1f} -- an odds ratio of 10^{comp[20][2]:.1f}")

print()
print()
print("The utility cost of each epsilon, at a fixed model and dataset.")
print()
print(f"{'epsilon':>10}{'accuracy':>11}{'vs non-private':>17}"
      f"{'training cost':>16}{'usable?':>11}")
print("-" * 65)
NONPRIV = 0.912
util = {}
for e in (0.1, 0.5, 1.0, 3.0, 8.0, 20.0, 1e9):
    if e > 1e8:
        acc = NONPRIV
        label = "inf"
    else:
        acc = NONPRIV - 0.29 * math.exp(-e / 2.4)
        label = f"{e:.1f}"
    cost = 1.0 if e > 1e8 else 2.4
    util[e] = (acc, acc / NONPRIV)
    print(f"{label:>10}{acc:>11.3f}{acc / NONPRIV:>16.1%}{cost:>16.1f}x"
          f"{('yes' if acc / NONPRIV > 0.94 else 'marginal' if acc / NONPRIV > 0.88 else 'no'):>11}")

print()
print(f"at epsilon {1.0:.1f} accuracy is {util[1.0][1]:.0%} of non-private;")
print(f"at epsilon {8.0:.1f} it is {util[8.0][1]:.0%}")

print()
print()
print("Which produces the actual decision: how much guarantee for how much loss.")
print()
print(f"{'posture':>34}{'epsilon':>10}{'odds ratio':>13}{'accuracy kept':>16}"
      f"  {'what it defends against':<34}")
print("-" * 109)
POSTURES = [
    ("no formal guarantee",       None, None, 1.000, "nothing, formally"),
    ("epsilon 20, one training run", 20.0, math.exp(20.0), util[20.0][1],
     "a catastrophic single-record leak"),
    ("epsilon 8, one training run",  8.0, math.exp(8.0), util[8.0][1],
     "membership inference, partly"),
    ("epsilon 3, one training run",  3.0, math.exp(3.0), util[3.0][1],
     "membership inference"),
    ("epsilon 1, one training run",  1.0, math.exp(1.0), util[1.0][1],
     "membership and reconstruction"),
]
for name, e, r, acc, defends in POSTURES:
    es = "--" if e is None else f"{e:.0f}"
    rs = "--" if r is None else f"{r:,.0f}"
    print(f"{name:>34}{es:>10}{rs:>13}{acc:>16.1%}  {defends:<34}")

print()
print("The middle rows are where most published deployments sit, and their")
print("odds ratios are large numbers.")

print()
print()
print("And the budget as an operational object: who spends it.")
print()
SPENDERS = [
    ("the production training run",   1, 0.5,  "planned"),
    ("a hyperparameter sweep",       40, 0.5,  "usually not counted"),
    ("an ablation study",            12, 0.5,  "usually not counted"),
    ("a fairness audit by subgroup",  8, 0.5,  "usually not counted"),
    ("an analyst's exploratory query", 60, 0.1, "never counted"),
    ("a dashboard refreshed hourly", 8760, 0.01, "never counted"),
]
print(f"{'who spends it':>34}{'queries':>10}{'epsilon each':>15}"
      f"{'basic total':>14}{'counted?':>22}")
print("-" * 95)
tot_counted, tot_all = 0.0, 0.0
for name, n, e, counted in SPENDERS:
    t = n * e
    tot_all += t
    if counted == "planned":
        tot_counted += t
    print(f"{name:>34}{n:>10,}{e:>15.2f}{t:>14.1f}{counted:>22}")
print("-" * 95)
print(f"{'TOTAL':>34}{'':>10}{'':>15}{tot_all:>14.1f}")
print(f"{'COUNTED':>34}{'':>10}{'':>15}{tot_counted:>14.1f}")

print()
print(f"the accounted budget is {tot_counted:.1f} and the spent budget is "
      f"{tot_all:.1f}")
print(f"a factor of {tot_all / tot_counted:.0f}")

print(f"""
The epsilon table is the first thing to get concrete about, because "differentially private" is
frequently reported without one. An epsilon bounds the multiplicative change in an adversary's
odds about any individual: at {1.0:.0f} the bound is {eps_tab[1.0][0]:.1f}x, at {8.0:.0f} it is
{eps_tab[8.0][0]:.0f}x (eq:epsilon-bounds-the-posterior-shift).

Read the third column. At epsilon {8.0:.0f}, an adversary who thought there was a
{0.01:.0%} chance you were in the dataset is permitted to end at
{eps_tab[8.0][1]:.0%}. That is a bound rather than a prediction -- most adversaries will learn
far less -- and it is the guarantee the system is offering.

**Epsilon {8.0:.0f} is a real guarantee and it is not the guarantee the word "private"
suggests**, and stating the number is the difference between a claim and a slogan.

The composition table is cite:abadi2016dpsgd's central accounting concern. Twenty queries at
epsilon {PER_QUERY:.1f} each give a basic-composition total of {comp[20][0]:.1f} and an
advanced-composition total of {comp[20][1]:.1f}
(eq:privacy-budget-composes-across-queries) -- an odds ratio of
10^{comp[20][2]:.1f}.

Advanced composition is much better than adding, which is why it exists. It is still growing,
and there is no mechanism by which it shrinks. **A dataset's privacy guarantee is the guarantee
after everything anyone has ever run on it.**

The utility table prices the guarantee. At epsilon {1.0:.1f} the model keeps
{util[1.0][1]:.0%} of non-private accuracy; at {3.0:.1f}, {util[3.0][1]:.0%}; at
{8.0:.1f}, {util[8.0][1]:.0%}. Training costs about {2.4:.1f} times as much throughout,
because per-example gradient clipping is not free.

That is the real trade and it is steep in a specific region. Between epsilon {8.0:.0f} and
{3.0:.0f} you give up {util[8.0][1] - util[3.0][1]:.1%} of accuracy and gain a
{math.exp(8.0) / math.exp(3.0):.0f}-fold tightening of the bound. Between {3.0:.0f} and
{1.0:.0f} you give up {util[3.0][1] - util[1.0][1]:.1%} more for another
{math.exp(3.0) / math.exp(1.0):.0f}-fold.

**The cheap part of the curve is the part between weak and moderate guarantees**, which is a
useful thing to know before the argument about whether epsilon 1 is achievable.

The posture table is how to present this to whoever decides. The row that most deployments
actually occupy is `no formal guarantee` -- accuracy {1.000:.0%}, defending against
{POSTURES[0][4]}. Every row below it trades measurable accuracy for a stated bound, and the
last column says what the bound is *for*: membership inference at moderate epsilon,
reconstruction at low.

That last column matters because it connects to ch:sec-data-leakage's finding.
cite:shokri2017membership's attack asks whether a record was in the training set, and for a
hospital discharge list or a customer roster **membership is itself the sensitive fact**. No
content control reaches it. Differential privacy is the only mechanism that does, which is the
argument for paying the accuracy cost -- and it applies to a specific class of data rather than
to everything.

The spender table is the operational failure and it is the one that voids most deployed
guarantees. The production training run is planned and counted, at {0.5:.1f}. A hyperparameter
sweep runs {40} times against the same data and is usually not counted. An ablation study,
{12}. A subgroup fairness audit, {8}. An analyst's exploratory queries, {60}. A dashboard,
{8760} a year.

The accounted budget is {tot_counted:.1f}. The spent budget is {tot_all:.1f} --
**a factor of {tot_all / tot_counted:.0f}**.

Every one of those queries touched the same data and every one leaked. **A privacy budget that
only counts the training run is not an accounting, it is a label**, and the fix is the same as
every other budget in this book: an instrument that measures the whole spend, in the place
where the spending happens.""")
