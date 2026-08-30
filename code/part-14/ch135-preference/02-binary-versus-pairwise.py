# -*- coding: utf-8 -*-
# Extracted from: Chapter 135 — Preference Optimization in Practice
# Source: src/.../ch135-preference.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Pairwise or binary? It depends on one measurable property of your annotators.

cite:ethayarajh2024kto makes the practical case for binary feedback: production
produces thumbs-up and thumbs-down for free, in volumes no comparison campaign
can match. The usual counter-argument is that a comparison carries more
information per judgement.

Both are true, and the comparison people actually face is at equal ANNOTATION
BUDGET rather than equal example count -- a pairwise judgement requires reading
two items, a binary one requires reading one. So binary starts with twice the
judgements for the money.

What decides it is a property of the annotators. Asked "is this good?", each
person applies their own bar, and real annotation is ROUTED: one person takes the
coding queries, another the writing queries, so which bar an item is measured
against is a function of what the item is about. A COMPARISON is immune -- the
annotator's bar appears on both sides and cancels (eq:comparison-cancels-the-bar).
A rating is not.

This listing also tests a hypothesis that turned out to be wrong, and reports it,
because the reason it is wrong is worth knowing.
"""
import numpy as np

rng = np.random.default_rng(199)

D, NF, C = 10, 400, 4             # dims, random features, topic areas
BETA = 5.0

W_Q = rng.normal(size=D)
U_TOPIC = rng.normal(size=(C, D))
W_RF = rng.normal(size=(D, NF)) * 0.9
B_RF = rng.uniform(0, 2 * np.pi, NF)


def quality(X):
    return np.tanh(X @ W_Q / np.sqrt(D)) + 0.35 * X[:, 0]


def topic(X):
    return (X @ U_TOPIC.T).argmax(axis=1)


def feat(X):
    return np.cos(X @ W_RF + B_RF)


def noisy(delta):
    return (rng.random(len(delta)) < 1.0 / (1.0 + np.exp(-BETA * delta)))


def logistic_fit(P, y, steps=600, lr=1.0, lam=1e-3):
    w = np.zeros(P.shape[1])
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-(P @ w)))
        w -= lr * (P.T @ (p - y) / len(y) + lam * w)
    return w


def draw(n):
    X = rng.normal(size=(n, D))
    return X, quality(X), topic(X)


def train_binary(n_items, bars):
    """One reading per judgement. The rating is quality against THIS topic's
    annotator's bar, and nothing in the data identifies that bar."""
    X, q, t = draw(n_items)
    return logistic_fit(feat(X), noisy(q - bars[t]).astype(float))


def build_pairs(n_pairs, cross):
    """Exactly n_pairs, of which `cross` are between different topics. Both
    columns must train on the SAME number of comparisons or the comparison is
    about sample size rather than about spanning."""
    n_cross = int(n_pairs * cross)
    n_within = n_pairs - n_cross
    Xa, qa, ta = draw(n_pairs * 12)
    Xb, qb, tb = draw(n_pairs * 12)
    same = ta == tb
    iw = np.flatnonzero(same)[:n_within]
    ic = np.flatnonzero(~same)[:n_cross]
    i = np.concatenate([iw, ic])
    assert len(i) == n_pairs, (len(i), n_pairs)
    return Xa[i], Xb[i], qa[i], qb[i]


def train_pairwise(n_pairs, bars, cross):
    """Two readings per judgement. An annotator's bar cancels in a comparison,
    so the bars never enter the labels at all."""
    Xa, Xb, qa, qb = build_pairs(n_pairs, cross)
    return logistic_fit(feat(Xa) - feat(Xb), noisy(qa - qb).astype(float))


# Test sets: pairs drawn WITHIN one topic, and pairs drawn ACROSS topics.
def test_pairs(n, want_cross):
    Xa, qa, ta = draw(n * 4)
    Xb, qb, tb = draw(n * 4)
    m = (ta != tb) if want_cross else (ta == tb)
    i = np.flatnonzero(m)[:n]
    return feat(Xa[i]), feat(Xb[i]), (qa[i] > qb[i]).astype(int)


FA_W, FB_W, Y_W = test_pairs(3000, False)
FA_C, FB_C, Y_C = test_pairs(3000, True)


def acc(w):
    return (float((((FA_W - FB_W) @ w > 0).astype(int) == Y_W).mean()),
            float((((FA_C - FB_C) @ w > 0).astype(int) == Y_C).mean()))


BUDGET = 16000
DRIFTS = (0.0, 0.3, 0.6, 1.0)

print(f"{C} topic areas, one annotator each, {BUDGET:,} item-readings of budget.")
print(f"That buys {BUDGET:,} binary ratings or {BUDGET // 2:,} comparisons.\n")
print(f"{'annotator':>10}" + f"{'binary ratings':>22}"
      + f"{'pairwise, within only':>25}" + f"{'pairwise, 30% span':>22}")
print(f"{'bar drift':>10}" + "".join(f"{'within':>11}{'ACROSS':>11}"
                                     for _ in range(3)))
print("-" * 76)

rows = {}
for drift in DRIFTS:
    bars = drift * rng.normal(size=C)
    b = acc(train_binary(BUDGET, bars))
    p0 = acc(train_pairwise(BUDGET // 2, bars, cross=0.0))
    p3 = acc(train_pairwise(BUDGET // 2, bars, cross=0.30))
    rows[drift] = (b, p0, p3)
    print(f"{drift:>10.1f}" + "".join(f"{v[0]:>11.3f}{v[1]:>11.3f}"
                                      for v in (b, p0, p3)))

z, d6, hi = rows[0.0], rows[0.6], rows[1.0]
bz, pz, qz = z
b6, p6, q6 = d6
bh, ph, qh = hi
print(f"""
Start with the drift-zero row, where every topic's annotator happens to share a
bar. Binary wins on both test sets: {bz[0]:.3f} and {bz[1]:.3f} against
pairwise's {pz[0]:.3f} and {pz[1]:.3f}. That is the information-per-cost argument
working as advertised -- a rating costs half the reading of a comparison, so the
same budget buys twice as many, and twice as many noisy absolute judgements beats
half as many clean relative ones.

Now go down the binary column. At drift {0.3} and {0.6} it is still competitive:
{b6[0]:.3f} and {b6[1]:.3f}. Then at drift {1.0} it falls off a cliff, to
{bh[0]:.3f} and {bh[1]:.3f}.

The pairwise columns do not move at all, at any drift. That is not robustness in
the statistical sense -- it is exact cancellation. When an annotator ranks two
items their bar appears on both sides of the comparison and subtracts out, so the
bars never enter the labels (eq:comparison-cancels-the-bar). The pairwise numbers
are flat because the quantity being varied cannot reach them.

Two things are worth taking from the shape of the binary column rather than just
its endpoints.

It is a CLIFF, not a slope. Binary feedback is fine, and better than pairwise, up
to a threshold -- and then it is much worse. That means "how much do our
annotators disagree about where the bar is" is not a nice-to-know: it is the
variable that decides which kind of data to buy, and it has a sharp answer rather
than a gradual one.

And the damage is not confined to the topic boundary. At drift {1.0} the WITHIN-
topic accuracy is {bh[0]:.3f}, barely better than the across-topic {bh[1]:.3f}.
The routed bars do not merely misalign the topics relative to each other; they
corrupt the learned score badly enough that ordering fails inside a topic too,
because the model is fitting one smooth function to labels that four different
questions produced.

Now the hypothesis that failed, which was the reason for the third column.

The expectation was that within-topic comparisons would leave the offsets BETWEEN
topics unidentified -- that a reward model trained only on same-topic pairs would
rank correctly inside each topic and arbitrarily between them, and that a share of
topic-spanning comparisons would be needed to tie the scale together. The third
column adds 30% spanning pairs at identical sample size to test it.

It made no difference: {qh[1]:.3f} against within-only's {ph[1]:.3f} at the
highest drift, and the columns are within noise of each other at every row.

The reason is worth more than the hypothesis was. A reward model is a smooth
function of content, not a lookup table with a free parameter per topic. Topics
overlap in feature space, and an ordering learned in one region constrains the
function in neighbouring regions, so the cross-topic offsets are identified
implicitly even though nothing in the data compares across topics directly.

Which comes with a condition to check rather than assume. That identification
relies on the topics not being genuinely disjoint in whatever representation the
reward model uses. If they were -- separate specialist heads, or topics so
different that no shared feature relates them -- the original worry would apply
and spanning comparisons would be load-bearing. In this setup they were not, and
saying so is more useful than quietly deleting the column.

So the practical rules, in order.

Measure the bar spread before choosing a data format. Have several annotators
rate one shared calibration set and compare their positive rates. That measurement
costs an afternoon and determines whether the free production feedback is an asset
or a liability.

If the spread is tight, take the binary data and the volume that comes with it.
If it is wide, either fix it -- a rubric, worked examples and a calibration round
move the bar spread far more cheaply than either kind of label -- or commission
comparisons, which are immune by construction.

And note the ordering that implies: the rubric is the highest-leverage artefact in
a preference-data project, because it moves the term that decides everything else.
The choice between a KTO-style objective and a DPO-style one is downstream of it.""")
