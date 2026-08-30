# -*- coding: utf-8 -*-
# Extracted from: Chapter 146 — Reasoning versus Generation
# Source: src/.../ch146-reasoning-vs-generation.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Three candidate tests for reasoning, and which of them you can run.

The previous listing showed that held-out accuracy cannot distinguish a system
that computes an answer from one fitted to the surface form of the problem, and
that perturbing the problem can. That is useful and it requires ground truth: you
have to know the right answer to the perturbed problem to score it.

This listing looks for criteria that do NOT require ground truth, because those
are the ones you can run on a deployed system against real traffic
(eq:criteria-without-labels).

Three candidates: does the system give the same answer to the same problem phrased
two ways; does its confidence rise when it is out of its depth; and does it
compose operations it has seen into combinations it has not.
"""
import numpy as np

rng = np.random.default_rng(283)

VOCAB = ["has", "gains", "buys", "each", "acquires", "obtains", "total", "sum",
         "boxes", "crates", "apples", "pears", "shop", "store", "sold", "left",
         "friends", "colleagues", "altogether", "combined", "spare", "extra"]
V = len(VOCAB)

# Each RULE has two phrasings. A system that computes gives the same answer to
# both; a system fitted to surface form has no reason to.
RULES = [
    ("a + b*c",   lambda a, b, c: a + b * c,
     [["has", "buys", "each", "total"], ["gains", "acquires", "each", "sum"]]),
    ("a*b + c",   lambda a, b, c: a * b + c,
     [["boxes", "each", "apples", "total"],
      ["crates", "each", "pears", "sum"]]),
    ("a - b*c",   lambda a, b, c: a - b * c,
     [["shop", "sold", "left"], ["store", "sold", "left", "spare"]]),
    ("a*b - c",   lambda a, b, c: a * b - c,
     [["friends", "each", "altogether"],
      ["colleagues", "each", "combined"]]),
]
# Held out entirely: a composition of operations that appear in the seen rules.
NOVEL = ("a*b + a", lambda a, b, c: a * b + a,
         [["boxes", "each", "gains", "total"], ["crates", "each", "obtains",
                                                "sum"]])


def encode(words, nums):
    x = np.zeros(V + 3)
    for w in words:
        x[VOCAB.index(w)] = 1.0
    x[V:] = nums
    return x


LO, HI = 2, 20


def sample(rules, n, phrase=None, lo=LO, hi=HI):
    X, Y, R = [], [], []
    for _ in range(n):
        r = rules[int(rng.integers(len(rules)))]
        p = r[2][phrase if phrase is not None else int(rng.integers(2))]
        nums = rng.integers(lo, hi, size=3).astype(float)
        X.append(encode(p, nums)); Y.append(r[1](*nums)); R.append(nums)
    return np.array(X), np.array(Y, float), np.array(R)


Xtr, Ytr, _ = sample(RULES, 24000)

NF, ENS = 700, 8
MU, SD = Xtr.mean(0), Xtr.std(0) + 1e-9
models = []
for e in range(ENS):
    W = rng.normal(size=(Xtr.shape[1], NF)) * 0.6
    B = rng.uniform(0, 2 * np.pi, NF)
    F = np.cos(((Xtr - MU) / SD) @ W + B)
    c = np.linalg.solve(F.T @ F + 1e-3 * np.eye(NF), F.T @ Ytr)
    models.append((W, B, c))


def predict(X):
    """Ensemble mean and spread. The spread is the model's own uncertainty,
    computed without any labels."""
    P = np.stack([np.cos(((X - MU) / SD) @ W + B) @ c for W, B, c in models])
    return P.mean(0), P.std(0)


def auroc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(len(scores))
    pos, neg = labels == 1, labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    return float((ranks[pos].mean() - (pos.sum() - 1) / 2) / neg.sum())


print("TEST 1 -- consistency under reformulation. The same problem, phrased two")
print("ways. Needs no ground truth: you only need two phrasings.")
print()
print(f"{'condition':>30}{'agree within 0.5':>19}{'mean disagreement':>20}")
print("-" * 69)

consist = {}
for name, lo, hi in (("in-distribution numbers", LO, HI),
                     ("numbers 2x larger", HI, 2 * HI),
                     ("numbers 5x larger", 5 * LO, 5 * HI)):
    agree, gaps = [], []
    for _ in range(3000):
        r = RULES[int(rng.integers(len(RULES)))]
        nums = rng.integers(lo, hi, size=3).astype(float)
        p0, _ = predict(encode(r[2][0], nums)[None])
        p1, _ = predict(encode(r[2][1], nums)[None])
        agree.append(abs(p0[0] - p1[0]) <= 0.5)
        gaps.append(abs(p0[0] - p1[0]))
    consist[name] = (float(np.mean(agree)), float(np.mean(gaps)))
    print(f"{name:>30}{consist[name][0]:>19.1%}{consist[name][1]:>20.2f}")

print()
print("  (the executing system agrees 100% by construction: both phrasings")
print("   invoke the same rule, so it cannot disagree with itself)")

print()
print()
print("TEST 2 -- does confidence rise when the system is out of its depth?")
print("Ensemble spread as an uncertainty signal. Also needs no ground truth.")
print()
print(f"{'condition':>30}{'mean spread':>14}{'vs baseline':>13}"
      f"{'AUROC vs':>11}")
print(f"{'':>30}{'':>14}{'':>13}{'baseline':>11}")
print("-" * 68)

Xb, Yb, _ = sample(RULES, 3000)
_, sb = predict(Xb)
base_spread = float(sb.mean())
unc = {}
for name, rules, lo, hi in (("in-distribution", RULES, LO, HI),
                            ("numbers 2x larger", RULES, HI, 2 * HI),
                            ("numbers 5x larger", RULES, 5 * LO, 5 * HI),
                            ("novel composition", [NOVEL], LO, HI)):
    X, Y, _ = sample(rules, 3000, lo=lo, hi=hi)
    _, sp = predict(X)
    a = auroc(np.concatenate([sb, sp]),
              np.concatenate([np.zeros(len(sb)), np.ones(len(sp))]))
    unc[name] = (float(sp.mean()), float(sp.mean()) / base_spread, a)
    print(f"{name:>30}{unc[name][0]:>14.2f}{unc[name][1]:>12.1f}x"
          f"{a:>11.2f}")

print()
print()
print("TEST 3 -- composition. A rule built from operations the system has seen,")
print("in a combination it has not. Needs ground truth, unlike the first two.")
print()
print(f"{'system':>30}{'exact match':>14}{'rel error':>12}")
print("-" * 56)
Xn, Yn, _ = sample([NOVEL], 3000)
pn, _ = predict(Xn)
comp_l = (float(np.mean(np.abs(pn - Yn) <= 0.5)),
          float(np.mean(np.abs(pn - Yn) / np.maximum(np.abs(Yn), 1.0))))
print(f"{'learned':>30}{comp_l[0]:>14.1%}{comp_l[1]:>12.3f}")
print(f"{'executing, rule not supplied':>30}{0.0:>14.1%}{'--':>12}")

c_in = consist["in-distribution numbers"]
c_2x = consist["numbers 2x larger"]
print(f"""
Test 1 is the one worth having, because it needs nothing you would not already
have: the same question asked twice, differently.

In distribution the two phrasings agree {c_in[0]:.1%} of the time, mean
disagreement {c_in[1]:.2f}. At double the number range they agree
{c_2x[0]:.1%}, mean disagreement {c_2x[1]:.2f}.

The executing system agrees with itself 100% of the time at every range, and not
because it is better -- because a system that maps the problem to a rule and then
evaluates the rule has nothing left that could depend on the phrasing. **Self-
consistency across paraphrase is not a proxy for correctness; it is a proxy for
whether the surface form is entering the computation** (eq:criteria-without-labels).

That makes it deployable in a way accuracy is not. You do not need to know the
right answer, or to have a benchmark, or to construct perturbations that preserve
semantics -- you need one paraphrase, and disagreement is evidence regardless of
which answer was right.

Test 2 comes out far stronger than expected, and the reason it does is the
interesting part.

Ensemble spread rises by {unc['numbers 2x larger'][1]:.0f}x at double the number
range and {unc['novel composition'][1]:.0f}x on a novel composition, separating
out-of-distribution from in-distribution examples with an AUROC of
{unc['numbers 2x larger'][2]:.2f} -- perfect. On this system, ensemble
disagreement is a flawless detector of the conditions that make it wrong.

That result should be read with suspicion rather than enthusiasm, because it
depends on a property of the model class rather than on anything general. Each
ensemble member is a random-feature model, and random features EXTRAPOLATE
DIFFERENTLY: outside the fitted region there is nothing constraining them to
agree, so they fly apart. The disagreement is enormous because the extrapolations
are independent.

Real models do not have that property. Members of an ensemble trained the same way
on the same data, with the same architecture and the same inductive biases,
extrapolate SIMILARLY -- they are wrong in correlated ways, so they agree while
being wrong. **Ensemble disagreement measures diversity of extrapolation, not
distance from the training distribution**, and those two coincide only when the
members are genuinely diverse.

Which is the general form of the caveat, and it applies well beyond this listing.
An uncertainty estimate built from agreement is only as good as the independence
of the things agreeing. Sampling one model several times at a nonzero temperature
gives the weakest version of this, because every sample shares every parameter --
and it is also the version most commonly used.

Test 3 is the criterion that would be most convincing and is the hardest to
apply. On a rule composed of operations the system has seen -- multiply, then add
a quantity already present -- the fitted model scores {comp_l[0]:.1%} with a
relative error of {comp_l[1]:.3f}.

The executing system scores 0%, and this is the honest part of the listing: it
scores 0% because nobody gave it the rule. It cannot induce a rule from examples,
which the fitted model at least attempts. **Neither system is doing the thing the
word "reasoning" is usually meant to name**, and constructing a system that
composes known operations into unseen combinations is the actual open problem
rather than a measurement problem.

So the practical output is a hierarchy of tests ordered by what they cost.

Paraphrase consistency costs one extra call and no labels, and it directly
measures whether the surface form is entering the answer. Run it on production
traffic.

Ensemble spread costs several models and worked perfectly here for a reason that
will not transfer. Run it where the members are genuinely diverse -- different
architectures, different data, different seeds all the way down -- and treat
agreement among near-identical models as almost no evidence at all.

Compositional generalisation costs a hand-built evaluation set and is the only one
that measures the thing people actually mean. Run it when you are choosing between
systems, and expect the answer to be uncomfortable.

None of the three is what a benchmark reports, and a benchmark score with none of
them beside it is a measurement of performance on one distribution, which was the
previous listing's finding stated a second way.""")
