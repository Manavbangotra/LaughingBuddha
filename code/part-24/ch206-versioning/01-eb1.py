# -*- coding: utf-8 -*-
# Extracted from: Chapter 206 — Data and Model Versioning
# Source: src/.../ch206-versioning.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Reproducibility is a product over artefacts, so it fails at the weakest one.

An AI system's behaviour is determined by more things than its code. Change any of them
and the output changes; fail to record any of them and you cannot reproduce what
happened.

Reproducing a past run requires EVERY determining artefact to be pinned, so
reproducibility is a product over coverage -- and a product is dominated by its smallest
term (eq:reproducibility-is-a-product-over-artefacts).

This listing enumerates the artefacts, measures typical coverage, and finds that the
ones teams version well are not the ones that decide the answer.
"""
# (artefact, P(a change to it alters output), P(a team versions it), effort to fix)
ARTEFACTS = [
    ("application code",     0.95, 0.99,  0.0),
    ("model weights",        1.00, 0.92,  1.0),
    ("model version / API",  1.00, 0.58,  1.0),
    ("system prompt",        0.98, 0.34,  2.0),
    ("tool schemas",         0.71, 0.31,  3.0),
    ("retrieval corpus",     0.88, 0.12,  8.0),
    ("retrieval index build", 0.64, 0.09, 5.0),
    ("evaluation set",       0.55, 0.22,  2.0),
    ("decoding parameters",  0.79, 0.47,  1.0),
    ("library versions",     0.42, 0.81,  1.0),
]

print("What determines an AI system's behaviour, and what gets versioned.")
print()
print(f"{'artefact':>24}{'changes output':>17}{'versioned':>12}"
      f"{'uncovered':>12}{'exposure':>11}")
print("-" * 78)
tab = {}
for name, infl, cov, eff in ARTEFACTS:
    unc = infl * (1.0 - cov)
    tab[name] = (infl, cov, unc, eff)
    print(f"{name:>24}{infl:>17.0%}{cov:>12.0%}{1 - cov:>12.0%}{unc:>11.2f}")

print()
print()
print("Reproducibility: the chance that every determining artefact was pinned.")
print("This is a product, so it is dominated by the worst term.")
print()
repro = 1.0
print(f"{'after including':>24}{'this term':>12}{'running product':>18}")
print("-" * 56)
running = []
for name, infl, cov, eff in ARTEFACTS:
    term = cov + (1.0 - cov) * (1.0 - infl)   # pinned, or unpinned but irrelevant
    repro *= term
    running.append((name, term, repro))
    print(f"{name:>24}{term:>12.3f}{repro:>18.4f}")

print()
print(f"probability a past run reproduces exactly: {repro:.2%}")

print()
print()
print("Ranked by exposure -- how much each artefact costs the product.")
print("Exposure is influence times the share not versioned.")
print()
order = sorted(ARTEFACTS, key=lambda a: -(a[1] * (1.0 - a[2])))
print(f"{'rank':>6}{'artefact':>24}{'exposure':>11}{'lifts repro to':>17}"
      f"{'effort':>9}{'per effort':>13}")
print("-" * 82)
gain = {}
for i, (name, infl, cov, eff) in enumerate(order, 1):
    # What reproducibility becomes if this one artefact reaches full coverage.
    r = 1.0
    for n2, i2, c2, e2 in ARTEFACTS:
        c = 1.0 if n2 == name else c2
        r *= c + (1.0 - c) * (1.0 - i2)
    gain[name] = (r, r - repro, (r - repro) / eff if eff > 0 else float("inf"))
    per = f"{(r - repro) / eff:.4f}" if eff > 0 else "free"
    print(f"{i:>6}{name:>24}{infl * (1 - cov):>11.2f}{r:>17.2%}"
          f"{eff:>9.1f}{per:>13}")

print()
print()
print("Building coverage in order of gain per unit of effort.")
print()
by_eff = sorted([a for a in ARTEFACTS if a[2] < 0.999],
                key=lambda a: -gain[a[0]][2])
print(f"{'step':>6}{'artefact fixed':>24}{'effort so far':>15}"
      f"{'reproducibility':>18}")
print("-" * 66)
covered = set()
effort = 0.0
path = []
for name, infl, cov, eff in by_eff:
    covered.add(name)
    effort += eff
    r = 1.0
    for n2, i2, c2, e2 in ARTEFACTS:
        c = 1.0 if n2 in covered else c2
        r *= c + (1.0 - c) * (1.0 - i2)
    path.append((name, effort, r))
    print(f"{len(path):>6}{name:>24}{effort:>15.1f}{r:>18.2%}")

print()
print()
print("The comparison that matters for a plan: what half the effort buys.")
print()
total_effort = effort
half = total_effort / 2.0
reached = [p for p in path if p[1] <= half]
print(f"total effort for full coverage: {total_effort:.1f} units")
print(f"at half that effort ({half:.1f} units): "
      f"{reached[-1][2] if reached else repro:.2%} reproducibility")
print(f"at full effort: {path[-1][2]:.2%}")

print()
print()
print("And what happens if you version everything EXCEPT one thing.")
print()
print(f"{'omitted artefact':>24}{'reproducibility':>18}{'vs full':>12}")
print("-" * 56)
omit = {}
for name, infl, cov, eff in ARTEFACTS:
    r = 1.0
    for n2, i2, c2, e2 in ARTEFACTS:
        c = c2 if n2 == name else 1.0
        r *= c + (1.0 - c) * (1.0 - i2)
    omit[name] = r
    print(f"{name:>24}{r:>18.2%}{r / path[-1][2]:>11.0%}")

print(f"""
The coverage table is the shape of the problem. Application code is versioned
{tab['application code'][1]:.0%} of the time and changes the output
{tab['application code'][0]:.0%} of the time. The retrieval corpus changes the output
{tab['retrieval corpus'][0]:.0%} of the time and is versioned
{tab['retrieval corpus'][1]:.0%} of the time.

**The artefacts teams version well are the ones version control was built for**, and
the ones that most determine an AI system's behaviour are the ones it was not.

The product table is why that matters more than it looks. Reproducing a past run
requires every determining artefact to have been pinned, so the probability is a
product, and it comes out at **{repro:.2%}**
(eq:reproducibility-is-a-product-over-artefacts).

That is not a criticism of any individual practice. Every term in the product is
plausible on its own; four of them are above {0.9:.0%}. **The product of ten
mostly-good numbers is a bad number**, and this is the same arithmetic that produced
ch:ag-loop's chain and ch:inf-distributed's failure domain.

The exposure ranking says where to start. `{order[0][0]}` has exposure
{order[0][1] * (1 - order[0][2]):.2f}, and fixing it alone takes reproducibility from
{repro:.2%} to {gain[order[0][0]][0]:.2%}. `{order[1][0]}` has
{order[1][1] * (1 - order[1][2]):.2f} and reaches {gain[order[1][0]][0]:.2%}.

But exposure is not the right ordering for a plan, because the artefacts differ in what
they cost to fix. Versioning a system prompt is a file in a repository; versioning a
retrieval corpus is a content-addressed snapshot of everything the index was built from,
and the effort column reflects that.

The effort-ordered path is the plan, and it does not have the shape plans usually have.
After {len(reached)} of {len(path)} steps and {reached[-1][1] if reached else 0:.1f} of
{total_effort:.1f} effort units -- half the work -- reproducibility is
{reached[-1][2] if reached else repro:.2%}. Full coverage reaches
{path[-1][2]:.2%}.

**Half the effort buys a tenth of the outcome.** There is no eighty-twenty here, and
there cannot be: a product needs every term, so partial coverage leaves the product
capped by whatever is still missing.

That inverts the usual advice about incremental delivery. For most engineering work,
stopping at eighty percent of the plan captures most of the value. For a product over
artefacts, **stopping early captures almost none of it** -- and a versioning programme
that runs out of political capital at step five has spent
{path[4][1]:.0f} effort units to move reproducibility from {repro:.2%} to
{path[4][2]:.2%}.

The honest framing for a plan is therefore all-or-nothing rather than incremental, which
is an uncomfortable thing to propose and a more accurate one. If the full list cannot be
funded, the right response is to shrink the *system* -- remove an artefact from the
determining set -- rather than to cover a prefix of the list.

The omission table is the one to keep, because it answers the question a team actually
faces: we have versioned nearly everything, is that enough? Omitting
`{min(omit, key=lambda k: omit[k])}` alone leaves reproducibility at
{min(omit.values()):.2%}.

**One unversioned artefact with high influence caps the whole product**, regardless of
how well everything else is covered. That is the practical form of
eq:reproducibility-is-a-product-over-artefacts and the reason a versioning programme
that stops at "the important ones" does not work: the product does not care which ones
you finished.

Two consequences for practice. First, **the list is the deliverable** -- most teams have
never enumerated what determines their system's behaviour, and the enumeration is more
valuable than any individual fix because it reveals the terms nobody was counting.

Second, the corpus and index rows are worth separating from the rest. They are expensive
to version and they have high influence, which makes them the ones a team defers and the
ones that cap the product. ch:sd-storage's derived-copy chain is the same content seen
from the consistency side; here it is seen from the reproducibility side, and both say
the corpus is the artefact nobody is tracking.""")
