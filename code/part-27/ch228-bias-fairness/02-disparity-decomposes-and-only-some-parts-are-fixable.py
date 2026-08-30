# -*- coding: utf-8 -*-
# Extracted from: Chapter 228 — Bias and Fairness
# Source: src/.../ch228-bias-fairness.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Measured disparity is five different things added together, and only some are fixable.

A single "the model performs worse for group B" number tells you nothing about what to do,
because it sums contributions with completely different remedies: a base-rate difference that
is a fact about the world, label bias that is a fact about the annotation, a representation
gap that is a fact about the sample, a feature-quality gap, and a threshold choice
(eq:disparity-decomposes-and-only-some-parts-are-fixable).

And one contribution arrives before any model runs. cite:petrov2023 measured that tokenizers
fragment some languages far more than others, so the same content costs more tokens -- which
is more money, less context, and worse latency, for identical requests
(eq:tokenisation-imposes-a-cost-disparity-before-any-model-runs).
"""
# (source, contribution to measured disparity, fixable by, cost, is it a model fix?)
SOURCES = [
    ("base-rate difference",       0.31, "nothing in the model",  0.0, "no"),
    ("label bias in the training set", 0.22, "relabel a sample",  6.0, "no"),
    ("representation: sample size", 0.18, "collect more data",    9.0, "partly"),
    ("feature quality for the group", 0.14, "better features",    5.0, "yes"),
    ("threshold chosen on the pooled set", 0.15, "set it per group", 0.4, "yes"),
]

print("Where a measured disparity comes from.")
print()
print(f"{'source':>36}{'share':>9}{'remedy':>24}{'effort':>9}"
      f"{'a model fix?':>15}")
print("-" * 93)
src = {}
for name, share, fix, eff, ismodel in SOURCES:
    src[name] = (share, eff, ismodel, share / eff if eff > 0 else 0.0)
    print(f"{name:>36}{share:>9.0%}{fix:>24}{eff:>9.1f}{ismodel:>15}")

removable = sum(s for n, s, f, e, m in SOURCES if e > 0)
print()
print(f"{removable:.0%} of the disparity has a remedy; "
      f"{1 - removable:.0%} does not")

print()
print()
print("Ranked by disparity removed per unit of effort.")
print()
order = sorted([s for s in SOURCES if s[3] > 0],
               key=lambda s: -(s[1] / s[3]))
print(f"{'rank':>6}{'source':>36}{'share':>9}{'effort':>9}{'per effort':>13}")
print("-" * 73)
for i, (name, share, fix, eff, m) in enumerate(order, 1):
    print(f"{i:>6}{name:>36}{share:>9.0%}{eff:>9.1f}{share / eff:>13.3f}")

print()
print(f"the cheapest remedy is {order[0][0]} at "
      f"{order[0][1] / order[0][3]:.3f}, and it is a config change")

print()
print()
print("Building remedies in payback order.")
print()
print(f"{'after fixing':>36}{'disparity remaining':>22}{'effort so far':>16}"
      f"{'vs floor':>11}")
print("-" * 85)
floor = 1.0 - removable
cur = 1.0
eff = 0.0
path = []
for name, share, fix, e, m in order:
    cur -= share
    eff += e
    path.append((name, cur, eff))
    print(f"{name:>36}{cur:>22.2f}{eff:>16.1f}{cur / floor:>11.2f}x")

print()
print(f"the floor is {floor:.2f} and it is the base-rate term")

print()
print()
print("Now the disparity that exists before the model does: tokenisation.")
print()
# (language, tokens per 1000 characters of equivalent content)
LANGS = [
    ("English",     238),
    ("Spanish",     289),
    ("Portuguese",  301),
    ("Russian",     521),
    ("Hindi",       744),
    ("Thai",        891),
    ("Burmese",    1103),
]
BASE = LANGS[0][1]
PRICE_PER_MTOK = 3.00
CONTEXT = 128_000
REQUESTS_PER_USER_YEAR = 2_400
print(f"{'language':>14}{'tokens/1k chars':>18}{'vs English':>13}"
      f"{'usable context (chars)':>25}{'cost/user/year':>17}")
print("-" * 89)
tok = {}
for name, tpk in LANGS:
    ratio = tpk / BASE
    chars = CONTEXT / tpk * 1000
    cost = REQUESTS_PER_USER_YEAR * (tpk * 2.4) / 1e6 * PRICE_PER_MTOK
    tok[name] = (ratio, chars, cost)
    print(f"{name:>14}{tpk:>18,}{ratio:>12.2f}x{chars:>25,.0f}"
          f"{cost:>17.2f}")

print()
print(f"a Burmese user gets {tok['Burmese'][1] / tok['English'][1]:.2f} times "
      f"the usable context and pays")
print(f"{tok['Burmese'][2] / tok['English'][2]:.2f} times as much, for identical requests")

print()
print()
print("What that does downstream, at a fixed context budget.")
print()
print(f"{'language':>14}{'documents that fit':>21}{'retrieval recall':>19}"
      f"{'answer quality':>17}{'vs English':>13}")
print("-" * 84)
for name, tpk in LANGS:
    docs = max(1, int(CONTEXT * 0.55 / (tpk * 1.8)))
    recall = 1.0 - 0.72 ** docs
    quality = 0.31 + 0.62 * recall
    q_en = 0.31 + 0.62 * (1.0 - 0.72 ** max(1, int(CONTEXT * 0.55 / (BASE * 1.8))))
    print(f"{name:>14}{docs:>21}{recall:>19.3f}{quality:>17.3f}"
          f"{quality / q_en:>12.2f}x")

print()
print("Nothing about the model changed. The context budget is denominated in")
print("tokens and the tokenizer is not neutral.")

print()
print()
print("What can be done about each layer, and what it costs.")
print()
FIXES = [
    ("per-group threshold",            0.15, 0.4,  "config"),
    ("relabel a stratified sample",    0.22, 6.0,  "annotation"),
    ("oversample the smaller group",   0.09, 1.5,  "training"),
    ("collect more data for the group", 0.18, 9.0, "acquisition"),
    ("a tokenizer with better coverage", 0.00, 0.0, "not yours to change"),
    ("price and budget per character",  0.00, 1.0, "product policy"),
    ("larger context for high-fertility languages", 0.00, 2.0, "product policy"),
]
print(f"{'fix':>44}{'disparity removed':>20}{'effort':>9}{'kind':>22}")
print("-" * 95)
for name, rem, eff, kind in FIXES:
    print(f"{name:>44}{rem:>20.0%}{eff:>9.1f}{kind:>22}")

print()
print("The last three remove no measured model disparity and remove the")
print("cost and context disparity, which no fairness metric was measuring.")

print(f"""
The decomposition is the first thing to do with any disparity number, and it changes the
conversation immediately. `{SOURCES[0][0]}` is {SOURCES[0][1]:.0%} of the measured gap and has
**no remedy inside the model** -- it is a fact about the population, and
ch:rai-bias' first listing showed it is also what sets the size of the impossibility.

{removable:.0%} of the disparity has a remedy and {1 - removable:.0%} does not
(eq:disparity-decomposes-and-only-some-parts-are-fixable). A team that reports one number and
sets a target for it has committed to reducing a quantity that is
{1 - removable:.0%} outside its control.

The ranking says where to start. `{order[0][0]}` removes {order[0][1]:.0%} for
{order[0][3]:.1f} units of effort -- {order[0][1] / order[0][3]:.3f} per unit -- and it is a
configuration change. cite:hardt2016equality's post-processing procedure is exactly this: adjust
the threshold per group to enforce a stated criterion, on a model you do not retrain.

Building in payback order takes the disparity from {1.0:.2f} to {path[-1][1]:.2f} for
{path[-1][2]:.1f} units. **The floor is {floor:.2f}**, and no amount of further effort moves
it.

The tokenisation table is the disparity that exists before any of this. cite:petrov2023
measured that tokenizers fragment languages unequally, and the consequences are not subtle: a
Burmese user gets {tok['Burmese'][1] / tok['English'][1]:.2f} times the usable context and pays
{tok['Burmese'][2] / tok['English'][2]:.2f} times as much for identical content
(eq:tokenisation-imposes-a-cost-disparity-before-any-model-runs).

That is a {tok['Burmese'][0]:.2f}x tax on one axis and a
{1 / (tok['Burmese'][1] / tok['English'][1]):.1f}x reduction on the other, and **no fairness
metric in the first listing measures either**, because both are properties of the interface
rather than of the classifier.

The downstream table follows it through. At a fixed context budget, fewer documents fit, so
retrieval recall falls, so answer quality falls -- for reasons that have nothing to do with the
model's competence in the language and everything to do with how many tokens the words became.

The fixes table is the honest ending. The first four remove measured model disparity and cost
between {0.4:.1f} and {9.0:.1f} units. The last three remove **no** measured model disparity
and remove the cost and context disparity entirely: price per character rather than per token,
budget context per character, and give high-fertility languages a larger window.

None of those is a model change and none would show up in a fairness report. **The largest
single equity intervention available to most teams is a billing decision**, and it is invisible
to every metric in this chapter's first listing.""")
