# -*- coding: utf-8 -*-
# Extracted from: Chapter 135 — Preference Optimization in Practice
# Source: src/.../ch135-preference.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Annotator agreement is the ceiling, and it is also the measuring stick.

part:09 derived the Bradley-Terry model and the DPO objective. This listing is
about the data those objectives consume, and about a number that is almost never
reported alongside a reward-model result: how often two annotators labelling the
same pair give the same answer.

That number matters twice, and the second time is the one people miss. It bounds
what the reward model can LEARN, which is expected. It also bounds what any
evaluation can MEASURE, because the held-out labels come from the same noisy
process (eq:agreement-caps-measurement) -- so a perfect reward model scores the
agreement rate, not 100%.
"""
import numpy as np

rng = np.random.default_rng(191)

D, NF = 10, 400
N_ITEMS = 20000

W_Q = rng.normal(size=D)


def quality(X):
    """The latent quality a preference is 'really' about."""
    return np.tanh(X @ W_Q / np.sqrt(D)) + 0.35 * X[:, 0]


W_RF = rng.normal(size=(D, NF)) * 0.9
B_RF = rng.uniform(0, 2 * np.pi, NF)


def feat(X):
    return np.cos(X @ W_RF + B_RF)


def annotate(qa, qb, beta):
    """A noisy annotator. beta is discrimination: high beta means the annotator
    reliably picks the better item, low beta means close calls are coin flips.
    This is the Bradley-Terry likelihood used as a NOISE model rather than as a
    training objective."""
    p = 1.0 / (1.0 + np.exp(-beta * (qa - qb)))
    return (rng.random(len(p)) < p).astype(int)


def train_rm(Xa, Xb, y, steps=500, lr=1.0, lam=1e-3):
    """Fit a scoring function under the Bradley-Terry loss on pair differences."""
    Pa, Pb = feat(Xa), feat(Xb)
    Dm = Pa - Pb
    w = np.zeros(NF)
    for _ in range(steps):
        z = Dm @ w
        p = 1.0 / (1.0 + np.exp(-z))
        g = Dm.T @ (p - y) / len(y) + lam * w
        w -= lr * g
    return w


def make_pairs(n):
    Xa = rng.normal(size=(n, D)); Xb = rng.normal(size=(n, D))
    return Xa, Xb, quality(Xa), quality(Xb)


BETAS = (0.8, 1.5, 3.0, 8.0)
N_TRAIN = 8000

Xta, Xtb, qta, qtb = make_pairs(6000)
truth_te = (qta > qtb).astype(int)

print(f"{N_TRAIN:,} training pairs, {len(truth_te):,} test pairs. The reward "
      f"model never\nsees the latent quality -- only the annotator's choices.\n")
print(f"{'annotator':>10}{'two annotators':>17}{'RM accuracy':>14}"
      f"{'RM accuracy':>14}{'apparent':>11}")
print(f"{'beta':>10}{'agree':>17}{'vs TRUTH':>14}{'vs LABELS':>14}"
      f"{'shortfall':>11}")
print("-" * 66)

rows = {}
for beta in BETAS:
    Xa, Xb, qa, qb = make_pairs(N_TRAIN)
    y = annotate(qa, qb, beta)
    w = train_rm(Xa, Xb, y)

    pred = ((feat(Xta) - feat(Xtb)) @ w > 0).astype(int)
    lab1 = annotate(qta, qtb, beta)
    lab2 = annotate(qta, qtb, beta)
    agree = float((lab1 == lab2).mean())
    a_truth = float((pred == truth_te).mean())
    a_label = float((pred == lab1).mean())
    rows[beta] = (agree, a_truth, a_label)
    print(f"{beta:>10.1f}{agree:>17.3f}{a_truth:>14.3f}{a_label:>14.3f}"
          f"{a_truth - a_label:>+11.3f}")

print("\n\nDoes more data climb past the annotator?\n")
print(f"{'pairs':>9}" + "".join(f"{'beta=' + str(b):>14}" for b in BETAS))
print(f"{'':>9}" + "".join(f"{'(vs truth)':>14}" for b in BETAS))
print("-" * 65)
scale = {}
for n in (500, 2000, 8000, 32000):
    line = []
    for beta in BETAS:
        Xa, Xb, qa, qb = make_pairs(n)
        w = train_rm(Xa, Xb, annotate(qa, qb, beta))
        acc = float((((feat(Xta) - feat(Xtb)) @ w > 0).astype(int)
                     == truth_te).mean())
        line.append(acc)
        scale[(n, beta)] = acc
    print(f"{n:>9,}" + "".join(f"{v:>14.3f}" for v in line))

lo, hi = rows[BETAS[0]], rows[BETAS[-1]]
b0, b8 = BETAS[0], BETAS[-1]
print(f"""
The first table contains the sentence this listing exists for. Look at the
low-discrimination row: two independent annotators agree {lo[0]:.1%} of the time,
the reward model is {lo[1]:.1%} accurate against the LATENT TRUTH, and it scores
{lo[2]:.1%} when measured against a held-out annotator label.

The reward model is substantially better than its own evaluation says it is. Not
because the evaluation is badly built, but because the held-out labels come from
the same noisy process as the training labels, so a model predicting the truth
perfectly would still disagree with them at the annotator's error rate
(eq:agreement-caps-measurement).

That is a reporting failure that runs in both directions. A reward-model accuracy
of {lo[2]:.0%} sounds poor and might mean the model is nearly as good as the task
permits. The same {lo[2]:.0%} on a task where annotators agree {hi[0]:.0%} of the
time would mean the model is badly broken. The number alone distinguishes
nothing, and the agreement rate is almost never published beside it.

Watch the shortfall column shrink as discrimination rises: {lo[1]-lo[2]:+.3f} at
beta={b0}, {hi[1]-hi[2]:+.3f} at beta={b8:.0f}. Clean labels make the measurement
honest. Noisy labels make it pessimistic by exactly the amount nobody quantifies.

The second table answers the obvious next question, and the answer is not the one
the first table sets you up to expect.

More data does climb past the annotator. In the beta={b0} column, 500 to 32,000
pairs moves accuracy against truth from {scale[(500, b0)]:.3f} to
{scale[(32000, b0)]:.3f}. The clean beta={b8:.0f} column ends at
{scale[(32000, b8)]:.3f}. At 64x the data, the noisy annotator has closed most of
a gap that looked structural.

So annotator noise of this kind is a TAX ON DATA EFFICIENCY, not a ceiling on
capability. The reason is that the noise is unbiased -- an annotator who is
merely uncertain is wrong in both directions with roughly equal probability, and
enough independent draws average that out (eq:noise-is-a-tax). Compare where
beta={b0} and beta={b8:.0f} reach the same accuracy: the clean annotator hits
{scale[(8000, b8)]:.3f} at 8,000 pairs, and the noisy one has not matched it at
32,000. Somewhere past 4x the annotation budget, for the same result.

Now put the two tables together, because the combination is the actionable part
and neither half implies it alone.

You CAN buy your way past noisy preference labels with volume. You can NEVER
measure that you did, because the measurement is capped by the same noise at
{lo[2]:.0%} no matter how good the model becomes. A team in this position sees a
reward model that refuses to score above the high fifties, concludes the approach
is not working, and stops -- while the model is at {lo[1]:.0%} against the thing
they actually care about.

That is a failure of instrumentation producing a wrong decision, and it is
invisible without double-labelling.

Which gives the ordering for a preference-data budget, roughly the reverse of how
these projects usually run. Double-label a small sample FIRST and compute
agreement. If agreement is low, decide deliberately between two different
projects: fix the task definition, rubric, or annotator pool to raise the ceiling
on measurement, or accept the ceiling and buy the volume the second table says
you need. Both are legitimate; conflating them is not.

And report the agreement rate beside every reward-model number, because without it
the number is not interpretable by anyone, including the team that produced it."""
)
