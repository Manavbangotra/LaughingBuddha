# -*- coding: utf-8 -*-
# Extracted from: Chapter 146 — Reasoning versus Generation
# Source: src/.../ch146-reasoning-vs-generation.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Two systems that score the same on a benchmark and are not the same thing.

The word "reasoning" gets applied to any system that produces correct answers to
problems that look like they need reasoning. That is not a definition, and it does
not distinguish a system that COMPUTES the answer from one that has learned what
answers to problems of this shape usually look like.

This listing builds both, on the same task, with the same training data. One
executes the arithmetic the problem describes. The other is fitted to a surface
representation of the problem -- the words present, and the numbers -- exactly as
a learned model sees it.

Then it looks for a measurement that tells them apart, because held-out accuracy
does not (eq:accuracy-does-not-separate).
"""
import numpy as np

rng = np.random.default_rng(281)

# Each template is a word pattern plus an arithmetic rule over three numbers.
VOCAB = ["has", "gives", "buys", "each", "times", "more", "left", "total",
         "boxes", "apples", "friends", "shop", "then", "how", "many", "cost",
         "sold", "bought", "spare", "remaining", "altogether", "twice"]
V = len(VOCAB)

TEMPLATES = [
    (["has", "buys", "each", "total"],        lambda a, b, c: a + b * c),
    (["has", "gives", "left"],                lambda a, b, c: a - b - c),
    (["boxes", "each", "apples", "total"],    lambda a, b, c: a * b + c),
    (["shop", "sold", "remaining"],           lambda a, b, c: a - b * c),
    (["friends", "each", "altogether"],       lambda a, b, c: a * b - c),
    (["bought", "twice", "more", "total"],    lambda a, b, c: a + 2 * b + c),
]
DISTRACTOR = ["spare", "cost", "how", "many"]      # words that carry no rule


def encode(t_idx, nums, distract=False):
    """The surface form a learned model sees: which words are present, and the
    numbers. Nothing tells it which arithmetic rule applies."""
    words, _ = TEMPLATES[t_idx], None
    x = np.zeros(V + 4)
    for w in TEMPLATES[t_idx][0]:
        x[VOCAB.index(w)] = 1.0
    if distract:
        for w in rng.choice(DISTRACTOR, size=2, replace=False):
            x[VOCAB.index(w)] = 1.0
        x[V + 3] = float(rng.integers(2, 30))       # an irrelevant quantity
    x[V:V + 3] = nums
    return x


def make(n, t_pool, lo, hi, distract=False):
    X, Y = [], []
    for _ in range(n):
        t = int(rng.choice(t_pool))
        nums = rng.integers(lo, hi, size=3).astype(float)
        X.append(encode(t, nums, distract))
        Y.append(TEMPLATES[t][1](*nums))
    return np.array(X), np.array(Y, float)


SEEN = [0, 1, 2, 3]           # templates present in training
UNSEEN = [4, 5]               # held out entirely
LO, HI = 2, 20

Xtr, Ytr = make(20000, SEEN, LO, HI)

NF = 900
W = rng.normal(size=(Xtr.shape[1], NF)) * 0.6
B = rng.uniform(0, 2 * np.pi, NF)
MU, SD = Xtr.mean(0), Xtr.std(0) + 1e-9


def feat(X):
    return np.cos(((X - MU) / SD) @ W + B)


coef = np.linalg.solve(feat(Xtr).T @ feat(Xtr) + 1e-3 * np.eye(NF),
                       feat(Xtr).T @ Ytr)


def learned(X):
    return feat(X) @ coef


def reasoner(t_idx, nums):
    """Executes the rule the problem describes. It cannot be wrong about
    arithmetic, and it cannot answer a template it does not know."""
    return TEMPLATES[t_idx][1](*nums)


def evaluate(t_pool, lo, hi, distract=False, n=4000, tol=0.5):
    X, Y = [], []
    ok_r = 0
    for _ in range(n):
        t = int(rng.choice(t_pool))
        nums = rng.integers(lo, hi, size=3).astype(float)
        X.append(encode(t, nums, distract))
        Y.append(TEMPLATES[t][1](*nums))
        ok_r += 1 if t in SEEN + UNSEEN else 0
    X, Y = np.array(X), np.array(Y, float)
    pred = learned(X)
    acc_l = float(np.mean(np.abs(pred - Y) <= tol))
    rel_l = float(np.mean(np.abs(pred - Y) / np.maximum(np.abs(Y), 1.0)))
    return acc_l, ok_r / n, rel_l


print("Two systems, same task, same training data. Exact-match accuracy.")
print("The 'reasoner' executes the arithmetic; the 'learned' model is fitted to")
print("the surface form -- words present, and the numbers.")
print()
print(f"{'test condition':>38}{'learned':>10}{'learned':>11}{'reasoner':>11}")
print(f"{'':>38}{'exact':>10}{'rel error':>11}{'exact':>11}")
print("-" * 70)

CASES = [
    ("held-out, same templates and range", SEEN, LO, HI, False),
    ("numbers 2x larger", SEEN, HI, 2 * HI, False),
    ("numbers 10x larger", SEEN, 10 * LO, 10 * HI, False),
    ("one irrelevant clause added", SEEN, LO, HI, True),
    ("templates never seen in training", UNSEEN, LO, HI, False),
]
res = {}
for name, pool, lo, hi, dis in CASES:
    al, ar, rl = evaluate(pool, lo, hi, dis)
    res[name] = (al, ar, rl)
    print(f"{name:>38}{al:>10.1%}{rl:>11.3f}{ar:>11.1%}")

print()
print()
print("How large does the perturbation have to be? Sweeping the number range.")
print()
print(f"{'number range':>16}{'exact':>11}{'rel error':>12}{'gap from':>11}")
print(f"{'':>16}{'':>11}{'':>12}{'baseline':>11}")
print("-" * 51)
base = res["held-out, same templates and range"][0]
sweep, sweep_rel = {}, {}
for mult in (1.0, 1.5, 2.0, 3.0, 5.0):
    lo, hi = int(LO * mult), int(HI * mult)
    a, _, r = evaluate(SEEN, lo, hi)
    sweep[mult], sweep_rel[mult] = a, r
    print(f"{f'{lo}-{hi}':>16}{a:>11.1%}{r:>12.3f}{base - a:>10.1%}")

h = res["held-out, same templates and range"]
n2 = res["numbers 2x larger"]
irr = res["one irrelevant clause added"]
uns = res["templates never seen in training"]
print(f"""
The first row is the one that would appear in a paper. On held-out problems drawn
from the same distribution as training, the learned model scores {h[0]:.1%} and
the reasoner scores {h[1]:.1%}. **A benchmark cannot tell them apart**
(eq:accuracy-does-not-separate), and neither can any amount of held-out data
drawn the same way.

Every other row separates them completely, and none of the perturbations changes
what the problem MEANS.

Doubling the size of the numbers takes the learned model to {n2[0]:.1%} while the
reasoner is unaffected at {n2[1]:.1%}. Nothing about the arithmetic changed.

But read the relative-error column beside it, because exact match overstates the
collapse. In distribution the learned model's mean relative error is
{h[2]:.3f}; at double the range it is {n2[2]:.3f}. It has not started producing
nonsense -- it is producing answers of roughly the right SIZE and the wrong VALUE,
which is a much more familiar failure than a random one and much harder to notice.

That distinction matters for what this demonstrates. The learned model has
captured the right shape of function and cannot evaluate it outside the region it
was fitted on. It interpolated, which is what fitting does, and outside the
training range there is nothing to interpolate between.

Adding one irrelevant clause -- two extra words and a number the rule never uses
-- takes it to {irr[0]:.1%}. The words are in the vocabulary and the number is in
the feature vector, so they move the prediction. A system that executed the rule
would ignore them because the rule does not mention them; a system fitted to
surface form has no way to know which parts of the surface matter.

That is precisely cite:mirzadeh2024gsmsymbolic's experiment, and it reports the
same shape of result on real models: performance declining when only the numbers
change, and drops of up to 65% from a single irrelevant sentence. This listing
makes the mechanism visible rather than inferred, because here we know exactly
what each system is doing.

The last row is the one that matters most and is easiest to miss. On templates
never seen in training, the learned model scores {uns[0]:.1%} and the reasoner
scores {uns[1]:.1%} -- but the reasoner's score is only that because it was GIVEN
the rule. It cannot induce a rule it has not been told, and a system that could
would be doing something neither of these does.

Which is the honest statement of what this listing shows and does not show. It
demonstrates that accuracy on held-out data cannot distinguish computing from
fitting, and it exhibits three perturbations that can. It does not show that real
models are the learned system -- that is cite:mirzadeh2024gsmsymbolic's job, and
their answer is "partly, and it varies."

The sweep makes the last point quantitative. The degradation is not a cliff at
some boundary; it is smooth in how far the test distribution moves. At
{1.5}x the range the learned model has already lost {base - sweep[1.5]:.1%}, and
at {5.0}x, {base - sweep[5.0]:.1%}. There is no threshold at which the model
stops working and starts working -- which means **any single benchmark score is a
point estimate on a curve whose slope nobody reports.**

So the practical content is a measurement protocol rather than a definition.
Whatever you mean by reasoning, a system that has it should be INVARIANT to
changes that do not change the answer. Generate your evaluation from templates so
you can vary the surface independently of the semantics, report the variance and
not just the mean, and treat a large gap between in-distribution and perturbed
accuracy as the finding rather than as noise.""")
