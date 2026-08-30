# -*- coding: utf-8 -*-
# Extracted from: Chapter 181 — Autonomous Experimentation and Report Generation
# Source: src/.../ch181-autonomous.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""When the generator and the judge share a blind spot.

cite:lu2024aiscientist built an end-to-end research pipeline -- idea, code,
experiments, figures, paper, review -- at under $15 per paper, and reported that it
can produce work exceeding a top conference's acceptance threshold AS JUDGED BY ITS
OWN AUTOMATED REVIEWER, which the authors report achieves near-human agreement on
paper scores.

That last clause is the measurement problem, and this book has the apparatus to
price it. ch:as-failures found that agents sharing a base model, a prompt lineage
and a context have correlated errors. A generator and a judge built from the same
model share all three.

So the question is not whether the judge is accurate in general. It is whether the
judge's errors are independent of the generator's -- because a generator that
produces flaws of the kind its own model family cannot see is producing work its
own judge will certify (eq:self-judging-measures-correlation).

Correlation here means one specific thing: the more the generator and judge share,
the more the generator's flaws fall in the region the judge is blind to.
"""
import numpy as np
from math import erf, sqrt

rng = np.random.default_rng(4787)

M = 60000
P_GOOD = 0.18           # share of generated work that is genuinely sound
JUDGE_SENS = 0.80       # judge accepts sound work at this rate
JUDGE_SPEC = 0.78       # judge's detection rate on INDEPENDENT flaws
SHIFT = 1.9             # how far shared bias pushes flaws toward invisibility


def norm_ppf(p):
    """Inverse normal CDF by bisection -- no scipy needed."""
    lo, hi = -9.0, 9.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if 0.5 * (1 + erf(mid / sqrt(2))) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def run(rho, m=M, p_good=P_GOOD, sens=JUDGE_SENS, spec=JUDGE_SPEC, shift=SHIFT):
    """`rho` is how much the generator's flaws are drawn from the region its own
    judge cannot see. At rho=0 flaws are independent of the judge's blind spot;
    at rho=1 they sit squarely inside it.

    Returns (accept rate, precision among accepted, sound accepted,
    unsound accepted).
    """
    good = rng.random(m) < p_good
    # A flaw's visibility TO THIS JUDGE. Calibrated so that at rho = 0 the judge
    # detects `spec` of flaws; shared bias shifts the distribution downward.
    t = norm_ppf(1.0 - spec)
    vis = rng.normal(-shift * rho, 1.0, m)
    detected = vis > t

    accept = np.empty(m, dtype=bool)
    accept[good] = rng.random(int(good.sum())) < sens
    accept[~good] = ~detected[~good]

    tp = float((accept & good).mean())
    fp = float((accept & ~good).mean())
    acc = tp + fp
    return acc, (tp / acc if acc else 0.0), tp, fp


print(f"{M:,} generated papers, {P_GOOD:.0%} of them genuinely sound. The judge")
print(f"accepts sound work {JUDGE_SENS:.0%} of the time and detects {JUDGE_SPEC:.0%}")
print("of flaws that are independent of its own blind spots.")
print()
print(f"{'shared bias':>13}{'flaws caught':>14}{'accept rate':>13}"
      f"{'precision':>11}{'unsound accepted':>18}")
print("-" * 69)
tab = {}
for rho in (0.0, 0.2, 0.5, 0.8, 0.95):
    r = run(rho)
    t = norm_ppf(1.0 - JUDGE_SPEC)
    caught = 1 - 0.5 * (1 + erf((t + SHIFT * rho) / sqrt(2)))
    tab[rho] = r + (caught,)
    print(f"{rho:>13.2f}{caught:>14.1%}{r[0]:>13.1%}{r[1]:>11.1%}{r[3]:>18.1%}")

print()
print()
print("What the pipeline REPORTS against what is true. The reported figure is")
print("the acceptance rate; the true figure is the share that is sound.")
print()
print(f"{'shared bias':>13}{'reported accept':>17}{'actually sound':>16}"
      f"{'overstatement':>15}")
print("-" * 61)
for rho in (0.0, 0.2, 0.5, 0.8, 0.95):
    r = tab[rho]
    print(f"{rho:>13.2f}{r[0]:>17.1%}{r[2]:>16.1%}{r[0] - r[2]:>15.1%}")

print()
print()
print("An independent judge -- a different model family, or a human -- at the")
print("SAME nominal accuracy. Only the shared bias differs.")
print()
print(f"{'judge':>28}{'accept rate':>13}{'precision':>11}")
print("-" * 52)
ind = run(0.0)
dep = run(0.9)
print(f"{'independent (shared 0.00)':>28}{ind[0]:>13.1%}{ind[1]:>11.1%}")
print(f"{'same family (shared 0.90)':>28}{dep[0]:>13.1%}{dep[1]:>11.1%}")
print()
print(f"   Identical nominal sensitivity and specificity. Precision differs by")
print(f"   {(ind[1] - dep[1]) * 100:.1f} points, entirely because the errors line up.")

print()
print()
print("Making the judge more accurate does not rescue it, because the failure")
print("is not an accuracy failure.")
print()
print(f"{'judge specificity':>19}" + "".join(f"{'shared ' + format(r, '.1f'):>15}"
                                            for r in (0.0, 0.5, 0.9)))
print("-" * 64)
sp = {}
for s in (0.70, 0.85, 0.95, 0.99):
    row = tuple(run(r, spec=s)[1] for r in (0.0, 0.5, 0.9))
    sp[s] = row
    print(f"{s:>19.0%}" + "".join(f"{v:>15.1%}" for v in row))


def panel(rho, k, m=M, p_good=P_GOOD, sens=JUDGE_SENS, spec=JUDGE_SPEC,
          shift=SHIFT):
    """k judges vote and the majority decides. Judges from the same family share
    their blind spots WITH EACH OTHER as well as with the generator."""
    good = rng.random(m) < p_good
    t = norm_ppf(1.0 - spec)
    shared = rng.normal(-shift * rho, 1.0, m)      # the common component
    votes = np.zeros(m, dtype=np.int64)
    for _ in range(k):
        vis = np.sqrt(rho) * shared + np.sqrt(1 - rho) * rng.normal(0, 1, m) \
            - shift * rho * (1 - np.sqrt(rho))
        detected = vis > t
        a = np.empty(m, dtype=bool)
        a[good] = rng.random(int(good.sum())) < sens
        a[~good] = ~detected[~good]
        votes += a
    accept = votes > k / 2
    tp = float((accept & good).mean())
    fp = float((accept & ~good).mean())
    acc = tp + fp
    return acc, (tp / acc if acc else 0.0)


print()
print()
print("And a panel of judges, which is ch:as-failures' result arriving here")
print("unchanged: correlated votes do not aggregate.")
print()
print(f"{'judges':>9}" + "".join(f"{'shared ' + format(r, '.1f'):>15}"
                                 for r in (0.0, 0.5, 0.9)))
print("-" * 54)
pn = {}
for k in (1, 3, 7):
    row = tuple(panel(r, k)[1] for r in (0.0, 0.5, 0.9))
    pn[k] = row
    print(f"{k:>9}" + "".join(f"{v:>15.1%}" for v in row))

print(f"""
The first table's accept column rises and the sound column does not.

At zero shared bias the pipeline accepts {tab[0.0][0]:.1%} of what it generates; at
{0.95:.2f} shared bias it accepts {tab[0.95][0]:.1%}. The share that is actually
sound is {tab[0.0][2]:.1%} and {tab[0.95][2]:.1%} -- **the same number, because
nothing about the generator changed.**

What changed is how much the judge and the generator have in common. And the
accept rate, which is the number the pipeline reports, moved
{tab[0.95][0] - tab[0.0][0]:.1f} points on that alone.

**A self-judged pipeline's acceptance rate measures the correlation between its
generator and its judge, not the quality of its output**
(eq:self-judging-measures-correlation). That is a strong claim and the second table
is its direct statement: the overstatement grows from
{tab[0.0][0] - tab[0.0][2]:.1%} to {tab[0.95][0] - tab[0.95][2]:.1%} while the truth
sits still.

The third table isolates it cleanly. Two judges with **identical** nominal
sensitivity and specificity: an independent one delivers {ind[1]:.1%} precision, a
same-family one {dep[1]:.1%}. Nothing about their measured accuracy differs. Only
whether their errors line up with the generator's.

Which means a reported reviewer accuracy -- "our automated reviewer achieves
near-human agreement on paper scores" -- **does not license the conclusion that its
acceptances are sound**, because agreement is measured on a distribution of papers
that is not the distribution its own generator produces. The generator writes
papers of a particular kind, and the judge's competence on THAT kind is the only
relevant number.

The fourth table forecloses the obvious response. At {0.9:.1f} shared bias, driving
the judge's specificity from {0.70:.0%} to {0.99:.0%} moves precision from
{sp[0.70][2]:.1%} to {sp[0.99][2]:.1%}. Independent, the same improvement moves it
from {sp[0.70][0]:.1%} to {sp[0.99][0]:.1%}.

**Making the judge more accurate does not rescue it, because the failure is not an
accuracy failure.** A better reviewer of the same family is better at catching the
flaws it could already catch.

And the last table forecloses the second obvious response. Seven judges at zero
shared bias reach {pn[7][0]:.1%}; seven at {0.9:.1f} reach {pn[7][2]:.1%}, barely
above one judge's {pn[1][2]:.1%}.

That is ch:as-failures' result exactly -- correlated votes do not aggregate,
because they are one opinion restated -- and it means an ensemble of reviewers from
one model family is a reviewer.

The practical consequences are narrow and hard.

**An automated reviewer must be independent of the generator to license anything**,
which in practice means a different model family, a different prompt lineage, or a
human. Same-family review is useful for catching the errors a model makes
carelessly and useless for the errors it makes systematically -- and the systematic
ones are the ones that survive to the output.

**Report the accept rate alongside the shared-bias estimate, or do not report it.**
The number alone is uninterpretable.

**And treat "judged by our own reviewer" as an unverified claim** rather than a
weak one. It is not a low-quality measurement; it is a measurement of a different
quantity than the one being claimed.""")
