# -*- coding: utf-8 -*-
# Extracted from: Chapter 134 — Synthetic Data and Data Quality
# Source: src/.../ch134-synthetic-data.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Not all wrong examples are equally wrong.

"Our synthetic data is 95% accurate" is the number every synthetic-data pipeline
reports, and on its own it means very little, because it does not say whether the
5% is scattered or concentrated.

Random errors are close to harmless: they pull in inconsistent directions and a
learner averages over them. Systematic errors are not, because they are LEARNABLE
-- a generator with a consistent misconception produces consistently wrong labels
in one region, and a model trained on them learns the misconception as if it were
the task (eq:systematic-noise-is-learnable).

Synthetic data produces the second kind by construction. This listing measures the
gap, and then measures whether you could detect it from inside the pipeline.
"""
import numpy as np

rng = np.random.default_rng(179)

D, NF = 12, 700
N_TRAIN, N_TEST = 6000, 6000

W_TRUE = rng.normal(size=D)
V_BAD = rng.normal(size=D)               # the direction the generator is wrong in
V_BAD /= np.linalg.norm(V_BAD)


def label(X):
    return (np.sin(1.5 * X @ W_TRUE / np.sqrt(D)) + 0.6 * X[:, 0] > 0).astype(int)


W_RF = rng.normal(size=(D, NF)) * 0.8
B_RF = rng.uniform(0, 2 * np.pi, NF)


def fit(X, y, lam=1e-3):
    P = np.cos(X @ W_RF + B_RF)
    return np.linalg.solve(P.T @ P + lam * len(X) * np.eye(NF),
                           P.T @ (2.0 * y - 1))


def pred(c, X):
    return (np.cos(X @ W_RF + B_RF) @ c > 0).astype(int)


def corrupt_random(X, y, rate):
    y = y.copy()
    idx = rng.permutation(len(y))[:int(rate * len(y))]
    y[idx] = 1 - y[idx]
    return y, np.zeros(len(y), bool)


def corrupt_systematic(X, y, rate):
    """The generator is confidently wrong in one contiguous region -- which is
    what a consistent misconception looks like in the data."""
    y = y.copy()
    s = X @ V_BAD
    idx = np.argsort(-s)[:int(rate * len(y))]
    y[idx] = 1 - y[idx]
    region = np.zeros(len(y), bool); region[idx] = True
    return y, region


X_tr = rng.normal(size=(N_TRAIN, D)); y_tr = label(X_tr)
X_te = rng.normal(size=(N_TEST, D));  y_te = label(X_te)
s_te = X_te @ V_BAD

print(f"{N_TRAIN:,} training examples, {N_TEST:,} clean test examples.\n")
print(f"{'wrong':>7}{'':>3}" + f"{'RANDOM errors':>28}" + f"{'SYSTEMATIC errors':>32}")
print(f"{'labels':>7}{'':>3}{'overall':>10}{'in bad':>9}{'self-':>9}"
      f"{'overall':>11}{'in bad':>10}{'self-':>11}")
print(f"{'':>7}{'':>3}{'':>10}{'region':>9}{'eval':>9}{'':>11}{'region':>10}"
      f"{'eval':>11}")
print("-" * 70)

rows = {}
clean_c = fit(X_tr, y_tr)
base = float((pred(clean_c, X_te) == y_te).mean())

for rate in (0.0, 0.05, 0.10, 0.20, 0.30):
    out = []
    for fn in (corrupt_random, corrupt_systematic):
        y_bad, _ = fn(X_tr, y_tr, rate)
        c = fit(X_tr, y_bad)
        p = pred(c, X_te)
        # The region the systematic generator is wrong about, on the test set.
        cut = np.quantile(s_te, 1 - rate) if rate > 0 else np.inf
        bad_region = s_te >= cut
        overall = float((p == y_te).mean())
        in_bad = float((p == y_te)[bad_region].mean()) if bad_region.any() \
            else float("nan")
        # Self-eval: score against labels the SAME generator would produce.
        y_self, _ = fn(X_te, y_te, rate)
        self_eval = float((p == y_self).mean())
        out.append((overall, in_bad, self_eval))
    rows[rate] = out
    r, sy = out
    def f(v, w):
        return f"{'--':>{w}}" if np.isnan(v) else f"{v:>{w}.3f}"
    print(f"{rate:>7.0%}{'':>3}{r[0]:>10.3f}{f(r[1], 9)}{r[2]:>9.3f}"
          f"{sy[0]:>11.3f}{f(sy[1], 10)}{sy[2]:>11.3f}")

r10, s10 = rows[0.10]
r20, s20 = rows[0.20]
r30, s30 = rows[0.30]
print(f"""
Read across the 10% row. The same fraction of the training labels is wrong in
both halves of the table. Scattered at random, clean-test accuracy is
{r10[0]:.3f}; concentrated in one region, it is {s10[0]:.3f}. And at 30% wrong
labels, random still holds {r30[0]:.3f} while systematic has fallen to
{s30[0]:.3f}.

A learner treats random label noise as noise: the flipped examples disagree with
each other and with the surrounding data, so they raise the loss floor without
moving the decision boundary much. Systematic errors are not noise. They are a
consistent signal about a region, indistinguishable from the truth by anything
inside the training set, and the model learns them because learning them is
exactly what it is for (eq:systematic-noise-is-learnable).

The bad-region column shows where the damage lands, and it is worse than the
overall numbers suggest. At 10% systematic corruption the model scores
{s10[1]:.3f} inside the affected region against {s10[0]:.3f} overall. At 20% it
scores {s20[1]:.3f} -- BELOW CHANCE. The model has not become uncertain about
that region; it has learned the generator's inverted rule and applies it
confidently. Random corruption at the same rate leaves the same region at
{r20[1]:.3f}, barely distinguishable from everywhere else.

This is why "our synthetic data is 95% accurate" is not a useful number. Ninety-
five per cent accurate with the 5% scattered is a mild tax. Ninety-five per cent
accurate with the 5% being every example of one subtopic your generator
misunderstands is a model that is confidently wrong about that subtopic, and the
aggregate accuracy figure is identical in both cases.

Now the self-eval column, which is why this survives review.

Suppose you hold out part of the synthetic data as a test set -- the obvious thing
to do, and what most pipelines do. That test set was produced by the same
generator, so it carries the same misconception, and it AGREES with the model
about the region where both are wrong. At 20% systematic corruption the model
scores {s20[2]:.3f} against generator-produced labels while scoring {s20[1]:.3f}
against the truth in the affected region -- and note the direction of travel:
between 10% and 30% corruption the self-eval score barely moves
({s10[2]:.3f} to {s30[2]:.3f}) while the truth in that region falls from
{s10[1]:.3f} to {s30[1]:.3f}. The held-out synthetic score reports a healthy
model throughout.

Contrast the random column, where self-eval FALLS steeply ({r10[2]:.3f} to
{r30[2]:.3f}). Random errors disagree with each other, so a synthetic test set
built from them punishes the model. Systematic errors agree, so a synthetic test
set built from them rewards it.

The evaluation cannot see the error because it shares it. This is the previous
chapter's eq:metric-inherits-bias in its sharpest form: there, the eval set
inherited the training set's SELECTION; here it inherits the generator's BELIEFS,
which is worse, because a selection bias leaves the missing data missing while a
shared misconception actively certifies the mistake.

Three practical consequences, and none of them is "measure accuracy more
carefully".

Report an error TAXONOMY, not an error rate. Sample the failures, cluster them,
and ask whether they concentrate. A hundred sampled errors that all concern the
same subtopic is a different dataset from a hundred that concern a hundred
subtopics, at identical accuracy.

Validate against something the generator did not produce. A small
human-labelled set, an execution check, a database lookup, a unit test -- any
external oracle breaks the agreement, and none of them needs to be large, because
you are detecting concentration rather than estimating a rate.

And ground the generation, per the previous listing. A generator writing from a
real source document can be checked against that document; a generator writing
from its prior can only be checked against itself.""")
