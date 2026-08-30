# -*- coding: utf-8 -*-
# Extracted from: Chapter 184 — Repository Understanding and Code Retrieval
# Source: src/.../ch184-repository-understanding.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Localisation, which is where issue resolution is actually decided.

cite:jimenez2023swebench requires reading an issue, finding the code that causes
it, and changing it -- across multiple functions, classes and files. The natural
assumption is that writing the fix is the hard part.

This listing decomposes the task the way ch:aids-text-to-sql decomposed
text-to-SQL, and finds the same shape: the generation step is not the bottleneck,
the GROUNDING step is (eq:localisation-caps-the-rest).

There is a second reason to look at localisation specifically.
cite:liang2025swebenchillusion measured models identifying the buggy file from the
issue description ALONE at up to 76% on SWE-bench repositories and up to 53% on
repositories not in SWE-bench. Locating a bug from prose without reading the code
should not be possible at 76%, so part of the reported figure is memory rather than
method -- and the 23-point gap is a measurement of how much.
"""
import numpy as np

rng = np.random.default_rng(5003)

M = 60000

# (sub-problem, success on a benchmark repo, success on an unseen repo)
SUBPROBLEMS = [
    ("localise the change", 0.76, 0.53),
    ("understand the code", 0.84, 0.72),
    ("write the fix",       0.88, 0.86),
    ("not break anything",  0.81, 0.74),
]


def run(profile, m=M, subs=None):
    """profile 0 = benchmark repository, 1 = unseen repository."""
    subs = subs or SUBPROBLEMS
    ok = np.ones(m, dtype=bool)
    blame = np.full(m, -1, dtype=np.int64)
    for i, row in enumerate(subs):
        good = rng.random(m) < row[1 + profile]
        blame[ok & ~good] = i
        ok &= good
    return float(ok.mean()), blame


print("Four sub-problems in resolving an issue, with success rates on a")
print("benchmark repository and on one the model has not seen.")
print()
print(f"{'sub-problem':>22}{'benchmark':>12}{'unseen':>9}{'drop':>8}")
print("-" * 51)
for name, a, b in SUBPROBLEMS:
    print(f"{name:>22}{a:>12.0%}{b:>9.0%}{b - a:>+8.0%}")

tot = {}
for prof, label in ((0, "benchmark repo"), (1, "unseen repo")):
    acc, blame = run(prof)
    tot[label] = (acc, blame)
print()
print(f"   end-to-end, benchmark repository: {tot['benchmark repo'][0]:.1%}")
print(f"   end-to-end, unseen repository:    {tot['unseen repo'][0]:.1%}")

print()
print()
print("As a share of the failures, which is the view that says what to fix.")
print()
print(f"{'first failure at':>22}{'benchmark':>12}{'unseen':>9}")
print("-" * 43)
fs = {}
for i, (name, _, _) in enumerate(SUBPROBLEMS):
    a = float((tot['benchmark repo'][1] == i).mean()) / (1 - tot['benchmark repo'][0])
    b = float((tot['unseen repo'][1] == i).mean()) / (1 - tot['unseen repo'][0])
    fs[name] = (a, b)
    print(f"{name:>22}{a:>12.1%}{b:>9.1%}")

print()
print()
print("Fixing one sub-problem to its benchmark level, on an unseen repository.")
print()
base = tot['unseen repo'][0]
print(f"{'lifted to benchmark level':>27}{'end-to-end':>13}{'gain':>9}")
print("-" * 49)
cf = {}
for i, (name, a, b) in enumerate(SUBPROBLEMS):
    subs = [list(r) for r in SUBPROBLEMS]
    subs[i][2] = subs[i][1]
    v = run(1, subs=[tuple(r) for r in subs])[0]
    cf[name] = (v, v - base)
    print(f"{name:>27}{v:>13.1%}{v - base:>+9.1%}")

print()
print()
print("Localisation is also multiplicative with everything after it: a fix in")
print("the wrong file cannot be right however well it is written.")
print()
print(f"{'localisation accuracy':>23}{'end-to-end':>13}{'ceiling':>10}")
print("-" * 46)
lo = {}
for L in (0.30, 0.53, 0.76, 0.90, 1.00):
    subs = [list(r) for r in SUBPROBLEMS]
    subs[0][2] = L
    v = run(1, subs=[tuple(r) for r in subs])[0]
    lo[L] = v
    print(f"{L:>23.0%}{v:>13.1%}{L:>10.0%}")

print()
print()
print("What the contamination gap costs. Reported performance uses the")
print("benchmark localisation rate; a new repository gets the other one.")
print()
subs_rep = [list(r) for r in SUBPROBLEMS]
subs_rep[0][2] = 0.76
reported = run(1, subs=[tuple(r) for r in subs_rep])[0]
actual = base
print(f"{'setting':>34}{'localisation':>14}{'end-to-end':>13}")
print("-" * 61)
print(f"{'benchmark repository (reported)':>34}{0.76:>14.0%}{reported:>13.1%}")
print(f"{'unseen repository (delivered)':>34}{0.53:>14.0%}{actual:>13.1%}")
print()
print(f"   The 23-point localisation gap becomes "
      f"{(reported - actual) * 100:.1f} points end-to-end,")
print(f"   a {reported / max(actual, 1e-9):.2f}x overstatement.")

print()
print()
print("And what retrieval buys, since localisation is the one sub-problem an")
print("engineering fix can address directly.")
print()
print(f"{'localisation method':>30}{'accuracy':>11}{'end-to-end':>13}")
print("-" * 54)
METHODS = [
    ("issue text alone (unseen repo)", 0.53),
    ("plus keyword search", 0.66),
    ("plus embedding retrieval", 0.71),
    ("plus call-graph expansion", 0.79),
    ("plus a failing test", 0.91),
]
mt = {}
for label, acc in METHODS:
    subs = [list(r) for r in SUBPROBLEMS]
    subs[0][2] = acc
    v = run(1, subs=[tuple(r) for r in subs])[0]
    mt[label] = (acc, v)
    print(f"{label:>30}{acc:>11.0%}{v:>13.1%}")

print(f"""
The failure-share table is ch:aids-text-to-sql's table with different labels.

Writing the fix accounts for {fs['write the fix'][1]:.1%} of failures on an unseen
repository. Localisation accounts for {fs['localise the change'][1]:.1%}.

The counterfactual confirms it: lifting the fix-writing rate to its benchmark level
is worth {cf['write the fix'][1]:+.1%}; lifting localisation is worth
{cf['localise the change'][1]:+.1%}.

**The generation step is not the bottleneck. The grounding step is**
(eq:localisation-caps-the-rest) -- which is the same finding, in the same shape,
that ch:aids-text-to-sql found for queries. In both cases the model can produce the
artefact and cannot reliably work out what the artefact should be about.

The ceiling table says why localisation is special rather than merely large. A fix
in the wrong file cannot be right however well it is written, so localisation
accuracy is a hard cap: at {0.53:.0%} localisation, end-to-end cannot exceed
{0.53:.0%}, and it reaches {lo[0.53]:.1%}.

**Every other sub-problem operates inside whatever localisation leaves**, which is
what makes it worth more than its failure share alone suggests.

Now the contamination table, which is why the benchmark number and the delivered
number differ systematically.

cite:liang2025swebenchillusion measured file identification from the issue text
alone at {0.76:.0%} on SWE-bench repositories and {0.53:.0%} on repositories not in
SWE-bench. Identifying a buggy file from prose, without reading the repository, is
not something method should permit at {0.76:.0%} -- so the gap is a measurement of
memory.

Propagated through the pipeline, {0.76:.0%} localisation gives {reported:.1%}
end-to-end and {0.53:.0%} gives {actual:.1%}: a
{reported / max(actual, 1e-9):.2f}x overstatement of what a new repository will get.

That is not an argument that the benchmark is worthless. It is an argument about
what its number means: **a reported resolution rate is a rate on repositories the
model has seen**, and the transfer to yours depends on a localisation gap that has
now been measured.

The last table is the constructive half, and it is why this chapter exists
separately from ch:aise-swe-agents. Localisation is the one sub-problem an
engineering intervention addresses directly.

Keyword search takes it from {mt['issue text alone (unseen repo)'][0]:.0%} to
{mt['plus keyword search'][0]:.0%}; embeddings to
{mt['plus embedding retrieval'][0]:.0%}; call-graph expansion to
{mt['plus call-graph expansion'][0]:.0%}; and a failing test to
{mt['plus a failing test'][0]:.0%}, taking end-to-end from
{mt['issue text alone (unseen repo)'][1]:.1%} to {mt['plus a failing test'][1]:.1%}.

**A failing test is the best localiser available**, because it identifies the code
by executing it rather than by resembling the issue -- which is
ch:as-specialized's verifier argument arriving as a retrieval technique.""")
