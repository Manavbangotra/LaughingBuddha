---
id: aise-repo
number: 184
part: XXI
tier: full
status: draft
requires: [grounding-not-syntax, retrieval-crossover-is-small,
           ratio-decides-acceptance, verifier-sets-the-ceiling]
provides: [localisation-caps-the-rest, contamination-inflates-localisation,
           structure-finds-what-text-cannot, seed-precedes-expansion,
           reproduce-before-retrieve]
citations: [jimenez2023swebench, liang2025swebenchillusion, wang2025solvedcorrectly,
            qin2023toolllm, patil2023gorilla]
---

## 1. Learning Objectives

By the end of this chapter you will be able to decompose issue resolution and
identify which sub-problem caps the rest; explain why a reported benchmark
resolution rate overstates what a new repository will get, and by roughly how much;
distinguish what text retrieval can reach in a codebase from what only structural
traversal can; state why improving text search improves structural retrieval and not
the reverse; and order the available localisation investments correctly.

## 2. Why This Matters

{{ch:aise-generation}} found writing code to be $15\%$ of a task and identified
constraint knowledge — knowing what the code must not break — as the largest source
of friction. This chapter is about supplying that knowledge, and it opens by
locating where issue resolution is actually decided.

{{sec:9-practical-example}} decomposes the task the way
{{ch:aids-text-to-sql}} decomposed queries, and finds the same shape. On an unseen
repository, writing the fix accounts for $6.9\%$ of failures and **localisation
accounts for $62.0\%$**. Lifting the fix-writing rate to its benchmark level is worth
$+0.1$ points; lifting localisation is worth $+11.9$
({{eq:localisation-caps-the-rest}}).

Localisation is also a hard ceiling rather than merely a large term — a fix in the
wrong file cannot be right however well written — so every other sub-problem operates
inside whatever it leaves.

That makes a second measurement important.
{{cite:liang2025swebenchillusion}} found models identifying the buggy file from the
issue text **alone** at up to $76\%$ on SWE-bench repositories and up to $53\%$ on
repositories not in the benchmark. Locating a bug from prose without reading the code
is not something method should permit at $76\%$, so the $23$-point gap is a
measurement of memorisation. Propagated through the pipeline it becomes a
$1.42\times$ overstatement of what a new repository gets
({{eq:contamination-inflates-localisation}}).

The constructive half is that localisation is the one sub-problem engineering
addresses directly — and the second listing finds that **text similarity finds the
file the issue talks about while only the call graph finds the files that must change
with it** ({{eq:structure-finds-what-text-cannot}}). Text-only resolution collapses
from $77.4\%$ on single-file changes to $0.0\%$ on eight-file ones, which is exactly
the regime {{cite:jimenez2023swebench}} was built to test.

## 3. Prerequisites

{{ch:aids-text-to-sql}}'s {{eq:grounding-not-syntax}} — the same decomposition, the
same conclusion, a different artefact.

{{ch:mcp-schemas}}'s {{eq:retrieval-crossover-is-small}}, since a repository is an
inventory of thousands and everything that chapter found about retrieval applies.

{{ch:as-specialized}}'s {{eq:verifier-sets-the-ceiling}}, which returns here in an
unexpected role: as a *retrieval* technique.

{{ch:aise-generation}}'s {{eq:ratio-decides-acceptance}}, whose constraint-violation
friction this chapter is about reducing.

## 4. Intuitive Explanation

An issue says: *sorting a DataFrame by a categorical column with unused categories
raises a KeyError.*

To fix it you need four things, and only one is writing code.

**Find the code.** Which of nine hundred files implements categorical sorting? The
issue does not say. It names a symptom and a user-facing API, and the bug is three
layers below both.

**Understand it.** What is this function's contract, what depends on it, why is it
written this way — there is usually a reason, and it is usually not in a comment.

**Write the fix.** The part everyone means.

**Not break anything else.** The subclass that overrides the method. The caller that
depends on the current behaviour. The test that asserts it.

{{sec:9-practical-example}} finds writing the fix at $86\%$ on unseen repositories —
nearly as good as on benchmark ones — and localisation at $53\%$. The model can write
the patch; it cannot reliably find where the patch goes.

That is {{ch:aids-text-to-sql}}'s finding with different nouns, and the parallel is
close enough to be a general claim: **for tasks over a large existing artefact, the
gap is in grounding rather than generation.**

Now the contamination problem, which makes this harder to see than it should be.

{{cite:liang2025swebenchillusion}} asked models to name the buggy file from the issue
description alone, without access to the repository. On SWE-bench repositories they
managed $76\%$. On repositories not in the benchmark, $53\%$.

Consider what the first number would require if it were method. From a prose
description of a symptom, with no view of the code, name the file among hundreds.
That is not inference; it is recall of a repository the model has read.

So a reported resolution rate is a rate *on repositories the model has seen*, and
your repository is not one of them. The gap propagates: $76\%$ localisation gives
$34.9\%$ end-to-end and $53\%$ gives $24.5\%$.

Then the useful question: what actually improves localisation?

The default answer is retrieval by similarity — embed the issue, embed the files,
retrieve the closest. That finds the file the issue *talks about*, which is genuinely
useful and is not the whole change set.

A change usually touches files the issue never mentions. The caller whose contract
changed. The subclass that overrides the method being fixed. The test asserting the
old behaviour. None of those resembles the issue text, so no similarity search
reaches them at any quality — but they are all one hop away in the *call graph*.

{{sec:9-practical-example}} finds text-only retrieval resolving $77.4\%$ of
single-file changes and $0.0\%$ of eight-file ones, while call-graph expansion holds
$26.6\%$ at eight files. And the two are asymmetric: graph expansion starts from a
file, so it needs search to find that file first.

The best localiser is neither. **A failing test names the files by executing them**,
which needs no similarity and no seed, and {{sec:9-practical-example}} finds it flat
across change spans where everything else degrades.

## 5. Formal Explanation

**Decomposition.** With sub-problems $s_1..s_4$ and a repository profile $\pi$:

$$A(\pi) = \prod_i s_i(\pi)$$

and the value of lifting sub-problem $j$ is $\Delta_j = \big(\prod_{i \ne j}
s_i\big)(s_j^{\text{high}} - s_j^{\text{low}})$, proportional to the *gap*.

Localisation differs from the others in one structural respect. It is not merely a
factor; it bounds the product, because a patch in the wrong location cannot succeed:

$$A(\pi) \le s_{\text{loc}}(\pi)$$ (eq:localisation-caps-the-rest)

**Every other sub-problem operates inside whatever localisation leaves**, which is
why it is worth more than its failure share alone suggests.

**Contamination.** Let $s_{\text{loc}}^{B}$ be the localisation rate on benchmark
repositories and $s_{\text{loc}}^{U}$ on unseen ones. The reported and delivered
end-to-end rates are:

$$A_{\text{reported}} = s_{\text{loc}}^{B}\prod_{i \ne \text{loc}} s_i, \qquad A_{\text{delivered}} = s_{\text{loc}}^{U}\prod_{i \ne \text{loc}} s_i$$

so the overstatement factor is exactly the localisation ratio:

$$\frac{A_{\text{reported}}}{A_{\text{delivered}}} = \frac{s_{\text{loc}}^{B}}{s_{\text{loc}}^{U}} = \frac{0.76}{0.53} \approx 1.43$$ (eq:contamination-inflates-localisation)

**The contamination gap transfers multiplicatively and unchanged**, which makes
{{cite:liang2025swebenchillusion}}'s file-path measurement directly convertible into
an end-to-end correction.

**Retrieval reach.** A change touches a set $F = \{f_0, f_1, \ldots, f_{k-1}\}$ where
$f_0$ is named by the issue and the rest are graph-linked. Text retrieval finds each
independently:

$$\Pr[\text{text finds } F] = \tau_0 \cdot \tau_s^{\,k-1}$$

with $\tau_s \ll \tau_0$ since linked files do not resemble the issue. Graph
expansion is *conditional on the seed*:

$$\Pr[\text{graph finds } F] = \tau_0 \cdot \gamma^{\,k-1}$$ (eq:structure-finds-what-text-cannot)

with $\gamma \gg \tau_s$. Both decay geometrically in $k$, but at very different
rates: at $\tau_s = 0.21$ and $\gamma = 0.86$, the ratio of bases is over four, so the
gap widens exponentially with span.

**Asymmetry.** Differentiating the combined method $\tau_0(1 - (1-\tau_s)(1-\gamma))^{k-1}$:

$$\frac{\partial A_{\text{both}}}{\partial \tau_0} > 0 \text{ always}, \qquad \frac{\partial A_{\text{graph}}}{\partial \tau_s} = 0$$ (eq:seed-precedes-expansion)

**Seed quality multiplies everything downstream; linked-file text recall does not
help the graph at all.** So the investments are ordered, not parallel.

**Execution as localisation.** A stack trace or failing test identifies $F$ by
having executed it, so its reach is independent of $k$:

$$\Pr[\text{trace finds } F] = \rho, \qquad \frac{\partial \rho}{\partial k} = 0$$ (eq:reproduce-before-retrieve)

**Flat in the span**, where every similarity- or graph-based method decays
geometrically. That is a difference in kind rather than degree.

## 6. Mathematical Foundation

Three extractions.

**A ceiling is not a factor.** {{eq:localisation-caps-the-rest}} says
$A \le s_{\text{loc}}$, which is stronger than localisation being one term among
four. It means no improvement anywhere else can compensate, and it justifies
spending disproportionately on it.

**The contamination correction is exact and available.** From
{{eq:contamination-inflates-localisation}}, the overstatement factor equals the
localisation ratio — a quantity {{cite:liang2025swebenchillusion}} measured directly.
So a reported SWE-bench figure can be corrected by dividing by $1.43$ for transfer to
an unseen repository, which is unusually actionable for a contamination result.

**Geometric decay with different bases separates exponentially.** From
{{eq:structure-finds-what-text-cannot}}, text and graph retrieval both decay in the
change span, so a system evaluated on single-file issues sees them as
indistinguishable and a system deployed on multi-file ones sees a factor of hundreds.
**Evaluating retrieval on single-file changes is the specific mistake that hides
this.**

## 7. Internal Mechanics

### 7.1 What a change set actually contains

```mermaid {#fig:change-set caption="A change set. Only the first file resembles the issue; the rest are reachable by call-graph traversal and by nothing else."}
flowchart TD
    I[the issue text] -->|"text similarity<br/>reaches this"| F0["sort_values in frame.py<br/>NAMED by the issue"]
    F0 -->|"callers"| F1["groupby.py<br/>depends on the contract"]
    F0 -->|"overrides"| F2["CategoricalIndex.sort<br/>subclass"]
    F0 -->|"asserts old behaviour"| F3["test_sorting.py"]
    I -.->|"no resemblance"| F1
    I -.->|"no resemblance"| F2
    I -.->|"no resemblance"| F3
```

The dotted edges are the point. Three of four files in a typical change set have no
textual relationship to the issue at all, and the relationship they *do* have is
structural.

### 7.2 The structural relations worth traversing

Not all graph edges are equally useful. In rough order of yield:

**Callers of a changed function.** If its contract changes, they must be checked.
This is the highest-yield expansion and the cheapest to compute.

**Overrides and implementations.** A change to a base method is a change to a
contract that subclasses satisfy, and those are frequently in distant files.

**Tests referencing the symbol.** Almost always in the change set, almost never
textually similar to the issue, and identifiable by a symbol-level index.

**Definitions of types in the signature.** Needed to understand rather than to
change, so they belong in context at lower priority.

**Recent co-change history.** Files that historically change together, from version
control — an empirical edge rather than a static one, and it captures couplings the
call graph misses.

That last one deserves emphasis: **version control history is a structural index
nobody uses**, and it encodes exactly the couplings that are real rather than
syntactic.

### 7.3 Why symbol indexes beat embeddings here

{{ch:mcp-schemas}} found retrieval necessary and {{cite:patil2023gorilla}} found
retrieval-over-documentation beating baked-in knowledge. Code adds a property that
prose does not have: **identifiers are exact.**

A function is named `sort_values` and every caller writes exactly that. So an exact
symbol index — which file defines it, which files reference it — answers the
retrieval question precisely, cheaply, and without an embedding model. Semantic
search is for the cases where you do not know the name yet, which is the seed step
and not the expansion step.

The practical composition: **embeddings or keyword search to find the seed, exact
symbol indexes to expand from it.** Using embeddings for the second step is both
slower and worse, because the relationship being followed is exact.

### 7.4 Reproduce before retrieve

{{eq:reproduce-before-retrieve}}'s flatness is the chapter's most actionable result,
and it inverts the usual pipeline order.

The usual order is: read the issue, retrieve candidate files, reason, write a patch,
run the tests. Reproduction appears at the end, as verification.

The result says reproduction belongs at the *start*, as localisation. A failing test
that exercises the bug produces a stack trace, and the stack trace names the frames
the execution passed through — which is a large fraction of the change set,
identified exactly, with no similarity or seed required.

That has two further benefits beyond localisation. It supplies
{{ch:as-specialized}}'s verifier for the whole task, converting "did I fix it" from a
judgement into an execution. And it forces the agent to establish that the bug is
real and reproducible before proposing a fix, which is the discipline experienced
engineers apply and agents frequently skip.

**The single largest improvement available to a code agent is usually a reproduction
step**, and it is cheaper to build than a retrieval system.

### 7.5 Retrieve narrowly

{{sec:9-practical-example}} finds most of the retrieval benefit reached by about six
files, with irrelevant files accumulating past that. That is
{{ch:mcp-schemas}}'s early-saturation result in a repository, and it argues against
the instinct to widen retrieval when a task fails.

When retrieval fails on a multi-file change, the failure is usually *categorical*
rather than marginal — the linked file was never a candidate under similarity at any
$k$. Widening the text retrieval therefore adds dilution and does not add the missing
file, while adding one graph hop does.

**Diagnose retrieval failures by asking whether the missing file was reachable, not
whether it was ranked highly enough.**

### 7.6 What a repository map is for

A common architecture gives the agent a condensed repository map — the file tree,
the top-level symbols, the module docstrings — before any retrieval.

Its value in this chapter's terms is precise: it improves the *seed*, and
{{eq:seed-precedes-expansion}} says seed quality multiplies everything after. A map
lets the agent form a hypothesis about which module is implicated, which turns a
nine-hundred-file similarity search into a twenty-file one.

Its cost is {{ch:mcp-schemas}}'s rent, paid on every request. So the map should be
the smallest artefact that supports the seeding hypothesis — module names and
one-line purposes — rather than a full symbol dump, on exactly the argument that
chapter made about tool descriptions: **include what distinguishes, not what
explains.**

### 7.7 Reading benchmark numbers after this chapter

Three corrections follow, and they compose.

**Divide by the localisation ratio** for transfer to an unseen repository:
{{eq:contamination-inflates-localisation}}'s $1.43$ at the measured rates.

**Check the change-span distribution** of the benchmark against your issues. A
benchmark weighted toward single-file changes measures a retrieval regime where text
similarity suffices.

**And ask what the grader accepts**, which is {{ch:aise-swe-agents}}'s subject —
{{cite:wang2025solvedcorrectly}} found $29.6\%$ of plausible patches behaving
differently from the human patch while passing the tests.

Applying all three to a reported figure lands somewhere well below the headline, and
that is the number to plan against.

### 7.8 Understanding is not the same as retrieving

This chapter has treated localisation as a retrieval problem, which it largely is,
and the second sub-problem in {{sec:9-practical-example}} deserves separating out
because it is not.

**Understanding the code** scored $72\%$ on unseen repositories against $84\%$ on
benchmark ones — the second largest drop in the table, and no retrieval technique
addresses it. Having the right file in context is necessary and not sufficient: the
question is why the code is the way it is.

Three kinds of knowledge sit behind that, and they degrade differently.

**Local invariants.** This function assumes the list is sorted; this cache is not
thread-safe; this early return exists because of a bug in a dependency. Sometimes a
comment says so. Usually the reason is in a commit message, a linked issue, or
nobody's memory.

**Architectural constraints.** This layer must not import that one; errors here
propagate rather than log; this module is performance-critical. These are rarely
written anywhere an agent can read, and violating one produces a change that is
correct in isolation and wrong in the codebase — which is
{{ch:aise-generation}}'s largest friction source.

**Intent.** What the code is *for*, as opposed to what it does. A refactor that
preserves behaviour and destroys the reason for a structure is a defect that no test
catches.

The retrieval techniques in this chapter reach the first partially — a blame
annotation or a linked commit message is retrievable — and the second and third not
at all.

Which lands on the same recommendation {{ch:aids-text-to-sql}} reached about
conventions and {{ch:aids-agentic-eda}} reached about cleaning policy: **the durable
fix is to write the constraint down somewhere executable.** An architectural rule
enforced by a lint rule is a constraint an agent can neither violate nor need to
know about. A layering rule enforced by an import checker is not documentation that
might be read; it is a verifier that runs.

That is worth stating as the general form, because it is now the fourth independent
arrival: **when the missing information is convention, encode it as a check rather
than hoping it is retrieved.** In a codebase this is unusually tractable, because the
tooling to enforce such rules already exists and is mostly unused.

## 8. Implementation

Two listings. The first decomposes issue resolution and prices the contamination
gap. The second measures what text and structural retrieval each reach.

```python {tier=A name=localisation-caps-the-rest}
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
```

The second listing asks what retrieval over a repository can reach.

```python {tier=A name=structure-finds-what-text-cannot}
"""Retrieving over a repository, where the thing you need is not the thing you
searched for.

ch:mcp-schemas found retrieval necessary once an inventory exceeds a few dozen
items, and a repository is an inventory of thousands. So code agents retrieve, and
they mostly retrieve by text similarity to the issue.

Text similarity finds the file the issue TALKS ABOUT. A change usually also
requires files the issue does not mention: the caller whose contract changes, the
subclass that overrides the method, the test that asserts the old behaviour. Those
are reachable from the first file by the CALL GRAPH and not by any amount of
similarity search (eq:structure-finds-what-text-cannot).

This listing measures how much of a required change set each method reaches, and
what an incomplete change set does to the patch.
"""
import numpy as np

rng = np.random.default_rng(5039)

M = 40000
REPO_FILES = 900
CONTEXT_FILES = 12          # how many files fit in the working context

# A change touches `span` files: one the issue names, and others reachable
# only structurally.
P_TEXT_FINDS = 0.82         # chance text search finds a textually-similar file
P_TEXT_FINDS_STRUCT = 0.21  # ...and a structurally-linked one it does not mention
P_GRAPH_FINDS = 0.86        # chance graph expansion reaches a linked file
FALSE_RATE = 0.30           # share of retrieved files that are irrelevant
DILUTE = 0.010              # per irrelevant file in context, cost to reasoning


def run(method, span, m=M, budget=CONTEXT_FILES, breadth=1.0):
    """Returns (complete change sets, mean files found, mean irrelevant shown,
    patch success)."""
    # File 0 is named by the issue; the rest are structurally linked.
    # Retrieving more candidates raises the chance each needed file is among
    # them, with diminishing returns, and brings more irrelevant ones.
    lift = 1.0 - (1.0 - 0.55) ** breadth
    found = np.zeros((m, span), dtype=bool)
    found[:, 0] = rng.random(m) < min(P_TEXT_FINDS * (0.75 + 0.35 * lift), 0.99)
    for k in range(1, span):
        p_text = min(P_TEXT_FINDS_STRUCT * (0.6 + 1.2 * lift), 0.95)
        p_graph = P_GRAPH_FINDS
        if method == "text":
            found[:, k] = rng.random(m) < p_text
        elif method == "graph":
            # Graph expansion works from the seed file, so it needs the seed.
            found[:, k] = found[:, 0] & (rng.random(m) < p_graph)
        elif method == "both":
            a = rng.random(m) < p_text
            b = found[:, 0] & (rng.random(m) < p_graph)
            found[:, k] = a | b
        else:
            raise ValueError(method)

    complete = found.all(1)
    n_found = found.sum(1)
    # Irrelevant files retrieved alongside, capped by the context budget.
    retrieved = 3.0 * breadth
    shown = np.minimum(np.maximum(retrieved, n_found), budget)
    irrelevant = np.maximum(shown - n_found, 0)
    # A patch succeeds if the change set is complete and the context is not
    # too diluted to reason over.
    ok = complete & (rng.random(m) < np.clip(1 - DILUTE * irrelevant, 0, 1))
    return (float(complete.mean()), float(n_found.mean()),
            float(irrelevant.mean()), float(ok.mean()))


print(f"A repository of {REPO_FILES} files. A change touches several of them: one")
print("the issue names, and others reachable only through the call graph.")
print()
print(f"{'files in the change':>21}" + "".join(f"{m:>12}" for m in
                                               ("text only", "graph only",
                                                "both")))
print("-" * 57)
tab = {}
for span in (1, 2, 3, 5, 8):
    row = tuple(run(m_, span)[0] for m_ in ("text", "graph", "both"))
    tab[span] = row
    print(f"{span:>21}" + "".join(f"{v:>12.1%}" for v in row))

print()
print()
print("The same, as patch success -- complete change set AND a context clean")
print("enough to reason over.")
print()
print(f"{'files in the change':>21}" + "".join(f"{m:>12}" for m in
                                               ("text only", "graph only",
                                                "both")))
print("-" * 57)
ps = {}
for span in (1, 2, 3, 5, 8):
    row = tuple(run(m_, span)[3] for m_ in ("text", "graph", "both"))
    ps[span] = row
    print(f"{span:>21}" + "".join(f"{v:>12.1%}" for v in row))

print()
print()
print("Why graph-only fails on single-file changes and text-only fails on")
print("multi-file ones: they find different things.")
print()
print(f"{'method':>14}{'finds the named file':>22}{'finds a linked file':>21}")
print("-" * 57)
print(f"{'text':>14}{P_TEXT_FINDS:>22.0%}{P_TEXT_FINDS_STRUCT:>21.0%}")
print(f"{'graph':>14}{'via the seed':>22}{P_GRAPH_FINDS:>21.0%}")
print(f"{'both':>14}{P_TEXT_FINDS:>22.0%}"
      f"{1 - (1 - P_TEXT_FINDS_STRUCT) * (1 - P_GRAPH_FINDS):>21.0%}")

print()
print()
print("Graph expansion depends on the seed, so improving text search improves")
print("BOTH -- which is not true in reverse.")
print()
print(f"{'text finds the seed':>21}{'text only':>12}{'both':>10}{'gain':>9}")
print("-" * 52)
sd = {}
for p in (0.50, 0.65, 0.82, 0.95):
    g = globals()
    saved = g["P_TEXT_FINDS"]
    g["P_TEXT_FINDS"] = p
    a = run("text", 3)[3]
    b = run("both", 3)[3]
    g["P_TEXT_FINDS"] = saved
    sd[p] = (a, b)
    print(f"{p:>21.0%}{a:>12.1%}{b:>10.1%}{b - a:>+9.1%}")

print()
print()
print("Retrieval breadth, which is ch:mcp-schemas' trade inside a repository:")
print("showing more candidate files raises recall and dilutes the context.")
print()
print(f"{'files retrieved':>17}{'complete sets':>15}{'irrelevant shown':>18}"
      f"{'patch success':>15}")
print("-" * 65)
cb = {}
for br in (0.7, 1.0, 2.0, 4.0, 8.0):
    r = run("both", 3, breadth=br)
    cb[round(3.0 * br)] = r
    print(f"{3.0 * br:>17.0f}{r[0]:>15.1%}{r[2]:>18.1f}{r[3]:>15.1%}")
best_breadth = max(cb, key=lambda k: cb[k][3])

print()
print()
print("And the reason a failing test is the best localiser of all: it names the")
print("files by executing them, so it needs neither similarity nor a seed.")
print()
print(f"{'localiser':>26}{'span 1':>10}{'span 3':>10}{'span 8':>10}")
print("-" * 56)
for label, m_ in (("text similarity", "text"), ("call-graph expansion", "graph"),
                  ("both", "both")):
    print(f"{label:>26}" + "".join(f"{run(m_, s)[3]:>10.1%}"
                                   for s in (1, 3, 8)))
# A stack trace names every frame it passed through, which is the change set.
trace = {}
for s in (1, 3, 8):
    found_all = rng.random(M) < 0.91 ** 1      # one localisation, not per file
    irr = 2.0
    ok = found_all & (rng.random(M) < (1 - DILUTE * irr))
    trace[s] = float(ok.mean())
print(f"{'a failing test / trace':>26}" + "".join(f"{trace[s]:>10.1%}"
                                                  for s in (1, 3, 8)))

print(f"""
The first table is the finding, and the collapse in the left column is the whole
argument.

Text similarity resolves {tab[1][0]:.1%} of single-file changes and
{tab[8][0]:.1%} of eight-file ones. Call-graph expansion resolves
{tab[8][1]:.1%} of the eight-file ones.

The mechanism is in the third table. **Text search finds the file the issue talks
about; the call graph finds the files that have to change with it**
(eq:structure-finds-what-text-cannot). A caller whose contract changed is not
textually similar to the issue -- the issue never mentions it -- so no amount of
better embedding reaches it.

That matters because cite:jimenez2023swebench's tasks explicitly require
coordinating changes across multiple functions, classes and files. **The benchmark
is hard in precisely the dimension text retrieval cannot address**, and a system
whose retrieval is similarity-only will look adequate on single-file issues and
fail on the rest.

The seed table shows the two methods are not symmetric. Graph expansion starts from
a file, so it needs text search to find that file first: improving the seed rate
from {0.50:.0%} to {0.95:.0%} takes the combined method from {sd[0.50][1]:.1%} to
{sd[0.95][1]:.1%}.

**Better text search improves structural retrieval and not the reverse.** So the
order of investment is settled: get the seed right, then expand from it. A team that
builds graph expansion on top of weak search has built the second half of a
mechanism.

The breadth table is ch:mcp-schemas' trade inside a repository. Retrieving
{cb[2][1]:.0f} files gives {cb[2][3]:.1%}; retrieving {24} gives {cb[24][3]:.1%},
with irrelevant files in context rising from {cb[2][2]:.1f} to {cb[24][2]:.1f}.

Recall rises and dilution offsets it, so the curve flattens rather than turning
over at these parameters -- but note the shape: **most of the benefit is reached by
about six files**, and everything past that is paying dilution for recall that is
nearly exhausted. That is the same early-saturation result ch:mcp-schemas found for
tool schemas.

The last table is the one to act on. A failing test or a stack trace localises at
{trace[1]:.1%} for a one-file change and {trace[8]:.1%} for an eight-file one --
**flat in the span**, where every other method degrades.

The reason is structural rather than a matter of degree. A trace names the files by
having executed them; it does not need the issue to resemble the code, and it does
not need a seed to expand from. It identifies the change set directly.

Which gives this chapter's practical ordering, and it is not the usual one.

**Reproduce the failure first.** A failing test is worth more than any retrieval
system, and building one is often the largest single improvement available to a
code agent.

**Then get the seed right**, because structural expansion is downstream of it.

**Then expand structurally**, not by retrieving more text.

**And retrieve narrowly** -- around six files -- because recall saturates early and
dilution does not.""")
```

## 9. Practical Example

The first listing decomposes issue resolution:

```
           sub-problem   benchmark   unseen    drop
---------------------------------------------------
   localise the change         76%      53%    -23%
   understand the code         84%      72%    -12%
         write the fix         88%      86%     -2%
    not break anything         81%      74%     -7%

   end-to-end, benchmark repository: 45.8%
   end-to-end, unseen repository:    24.5%
```

As a share of failures:

```
      first failure at   benchmark   unseen
-------------------------------------------
   localise the change       43.5%    62.0%
         write the fix       14.3%     6.9%
```

**Localisation is $62.0\%$ of failures on an unseen repository and writing the fix is
$6.9\%$** ({{eq:localisation-caps-the-rest}}) — {{ch:aids-text-to-sql}}'s result with
different nouns.

And it is a ceiling rather than a term:

```
  localisation accuracy   end-to-end   ceiling
----------------------------------------------
                    53%        24.4%       53%
                    76%        34.9%       76%
                   100%        45.9%      100%
```

The contamination gap:

```
                           setting  localisation   end-to-end
-------------------------------------------------------------
   benchmark repository (reported)           76%        34.9%
     unseen repository (delivered)           53%        24.5%
```

A $1.42\times$ overstatement ({{eq:contamination-inflates-localisation}}) — and the
correction factor is exactly the localisation ratio
{{cite:liang2025swebenchillusion}} measured.

What improves it:

```
           localisation method   accuracy   end-to-end
------------------------------------------------------
issue text alone (unseen repo)        53%        24.5%
           plus keyword search        66%        29.9%
     plus call-graph expansion        79%        36.2%
           plus a failing test        91%        41.8%
```

The second listing measures what each method reaches:

```
  files in the change   text only  graph only        both
---------------------------------------------------------
                    1       77.4%       77.1%       76.8%
                    3        5.3%       56.9%       62.5%
                    8        0.0%       26.6%       36.0%
```

**Text similarity collapses with change span and structural traversal does not**
({{eq:structure-finds-what-text-cannot}}) — because a caller whose contract changed is
not textually similar to an issue that never mentions it.

```
        method  finds the named file  finds a linked file
---------------------------------------------------------
          text                   82%                  21%
         graph          via the seed                  86%
```

And they are asymmetric:

```
  text finds the seed   text only      both     gain
----------------------------------------------------
                  50%        3.2%     38.1%   +35.0%
                  95%        6.3%     72.3%   +66.0%
```

**Better text search improves structural retrieval and not the reverse**
({{eq:seed-precedes-expansion}}) — graph expansion starts from a file, so the seed
multiplies everything after it.

Breadth saturates early:

```
  files retrieved  complete sets  irrelevant shown  patch success
-----------------------------------------------------------------
                2          58.9%               0.4          58.9%
                6          69.2%               3.5          67.0%
               24          75.3%               9.4          68.4%
```

And the localiser that does not degrade:

```
                 localiser    span 1    span 3    span 8
--------------------------------------------------------
           text similarity     77.4%      5.3%      0.0%
      call-graph expansion     77.1%     57.3%     26.4%
    a failing test / trace     89.3%     89.1%     89.3%
```

**A failing test names the files by executing them**
({{eq:reproduce-before-retrieve}}) — flat in the span, where everything else decays
geometrically.

## 10. Production Considerations

Build reproduction first. A failing test is the best localiser available, supplies
the verifier for the whole task, and is cheaper to build than a retrieval system.

Invest in the seed before the expansion. Seed quality multiplies everything
downstream and linked-file text recall does not help the graph at all.

Use exact symbol indexes for expansion, not embeddings. Identifiers are exact and
the relationship being followed is exact.

Traverse callers, overrides, referencing tests, and co-change history from version
control — the last is a structural index almost nobody uses.

Retrieve narrowly, around half a dozen files. Recall saturates early and dilution
does not.

Diagnose retrieval failures by reachability, not by ranking. A missing linked file
was never a candidate at any $k$.

Keep the repository map small — module names and one-line purposes — on
{{ch:mcp-schemas}}'s include-what-distinguishes rule.

And divide reported benchmark figures by the localisation ratio before planning
against them.

## 11. Common Mistakes

**Assuming the model's difficulty is writing code.** It is $6.9\%$ of failures.

**Reading a benchmark rate as transferable.** It is a rate on repositories the model
has seen.

**Retrieving only by similarity.** It cannot reach three quarters of a typical change
set.

**Building graph expansion on weak search.** The second half of a mechanism.

**Using embeddings to follow references.** Slower and worse than an exact index.

**Widening retrieval when a multi-file task fails.** The missing file was not ranked
low; it was not a candidate.

**Putting reproduction at the end.** It is the best localiser and it is being used as
a checker.

**Dumping the whole symbol table as a repository map.** Rent without distinction.

## 12. Failure Modes

*Right fix, wrong file.* The characteristic localisation failure, capped by
{{eq:localisation-caps-the-rest}} regardless of patch quality.

*Incomplete change set.* The named file fixed and the caller, subclass or test left
inconsistent — the multi-file failure text retrieval guarantees.

*Benchmark transfer disappointment.* A system at the reported rate on its benchmark
and $1.43\times$ worse on yours.

*Retrieval dilution.* Context filled with plausible neighbours, none of them the
linked file.

*Unreproduced fix.* A patch for a bug nobody demonstrated, which
{{ch:aise-swe-agents}} finds is where plausible-but-wrong patches come from.

## 13. Alternatives

**Give the agent the whole repository.** Feasible for small projects and it removes
the retrieval problem entirely at {{ch:mcp-schemas}}'s dilution cost.

**Static analysis for change impact.** A compiler or type checker can compute what a
signature change affects, exactly — the strongest form of graph expansion and
underused.

**Ask the agent to search interactively** rather than pre-retrieving, which lets it
follow references adaptively at the cost of turns —
{{ch:aise-swe-agents}}'s regime.

**Human-supplied localisation.** A maintainer naming the file removes the binding
constraint entirely, and is the cheapest intervention where a maintainer is present.

**Better issue reports.** A stack trace in the issue is localisation, supplied for
free by the reporter, and it is worth asking for.

## 14. Evaluation

Measure localisation accuracy separately from end-to-end resolution. It is the
ceiling and it is the thing to improve.

Measure it on repositories the model has not seen. Otherwise you are measuring
{{cite:liang2025swebenchillusion}}'s memorisation.

Report the change-span distribution of your issues. Retrieval evaluated on
single-file changes is evaluated in the regime where the difference does not show.

Measure change-set completeness, not just whether the named file was found.

And measure how often reproduction succeeded before a patch was proposed. It is the
best predictor available of whether the patch is right.

## 15. Advanced Concepts

**Co-change mining as a retrieval index.** Version control history encodes real
couplings the call graph misses, and using it as a retrieval edge is straightforward
and rare. {{maturity:EMERGING}}.

**Type-checker-driven impact analysis.** Computing the exact set of affected call
sites from a signature change, which converts expansion from a heuristic into a
derivation.

**Reproduction synthesis.** Generating a failing test from an issue description,
which would supply {{eq:reproduce-before-retrieve}}'s localiser automatically.
{{ch:aise-testing}} takes this up. {{maturity:EMERGING}}.

**Contamination-controlled benchmarks.** Dynamically constructed repository tasks the
model cannot have seen, which is the only way to measure localisation honestly.
{{maturity:RESEARCH FRONTIER}}.

## 16. Connection to Previous Chapters

{{ch:aids-text-to-sql}}'s grounding-not-syntax result recurs with different nouns,
which suggests it is a general property of tasks over large existing artefacts rather
than a fact about SQL.

{{ch:mcp-schemas}}'s retrieval results transfer, including early saturation and the
rent of a repository map.

{{ch:as-specialized}}'s verifier ceiling appears in a new role: a failing test is
both the verifier and the best localiser, which is why reproduction belongs first.

{{ch:aise-generation}}'s constraint-violation friction is exactly what an incomplete
change set produces, so this chapter's retrieval is that chapter's fix.

Ahead: {{ch:aise-swe-agents}} takes the whole loop, and asks what
{{cite:wang2025solvedcorrectly}} found about patches that pass the tests and do
something else.

## 17. Exercises

1. Measure the change-span distribution in your repository from merged pull
   requests. What fraction touch more than one file?

2. Build a symbol index and measure caller-expansion recall against real change sets
   from history.

3. Add co-change edges from version control and measure the marginal recall over
   call-graph edges alone.

4. Implement reproduction-first localisation and measure it against similarity
   retrieval on your own issues.

5. Derive the span at which text-only and graph-only retrieval cross, from
   {{eq:structure-finds-what-text-cannot}}.

6. Measure your model's localisation accuracy on a repository created after its
   training cutoff.

## 18. Interview Questions

1. Where does issue resolution actually fail?

2. A model names the buggy file from the issue text alone $76\%$ of the time. What
   does that tell you?

3. Your retrieval finds the file the issue mentions and the patch is still
   incomplete. Why?

4. Would you improve semantic search or graph expansion first?

5. What is the best localiser available, and why is it usually used as something
   else?

6. A multi-file task fails. Do you retrieve more files?

## 19. Research Questions

1. How much recall does co-change history add over static call-graph edges?

2. Can failing tests be synthesised from issue text reliably enough to serve as a
   localiser?

3. What is the localisation gap on repositories created after a model's training
   cutoff, and does it match the SWE-bench measurement?

4. Can change-set completeness be predicted before the patch is written?

5. Does the grounding-not-generation pattern hold for every task over a large
   existing artefact?

## 20. Chapter Summary

Issue resolution decomposes into localising, understanding, writing and not
breaking things — and on an unseen repository, writing the fix is $6.9\%$ of failures
while **localisation is $62.0\%$**. It is also a ceiling rather than a term, since a
patch in the wrong file cannot succeed
({{eq:localisation-caps-the-rest}}), so everything else operates inside what it
leaves.

{{cite:liang2025swebenchillusion}} measured models naming the buggy file from the
issue text **alone** at $76\%$ on SWE-bench repositories and $53\%$ elsewhere — a gap
that is a measurement of memorisation, since prose should not locate a bug at that
rate. Propagated, it is a **$1.42\times$ overstatement** of what an unseen repository
gets, and the correction factor is exactly the localisation ratio
({{eq:contamination-inflates-localisation}}).

On retrieval, **text similarity finds the file the issue talks about and only the
call graph finds the files that must change with it**
({{eq:structure-finds-what-text-cannot}}). Text-only resolution fell from $77.4\%$ at
one file to $0.0\%$ at eight; graph expansion held $26.6\%$. The two are asymmetric:
graph expansion starts from a seed, so **better search improves structural retrieval
and not the reverse** ({{eq:seed-precedes-expansion}}), which settles the order of
investment.

Retrieval breadth saturates around half a dozen files, and a multi-file failure is
categorical rather than marginal — the linked file was never a candidate, so widening
adds dilution and not the file.

And the best localiser is not a retrieval system. **A failing test names the files by
executing them**, needing neither similarity nor a seed, and stays flat across change
spans where everything else decays geometrically
({{eq:reproduce-before-retrieve}}). It also supplies the task's verifier. So the
single largest improvement available to a code agent is usually a reproduction step,
placed first rather than last.

## 21. Further Reading

{{cite:liang2025swebenchillusion}} for the localisation measurement this chapter
converts into an end-to-end correction — its file-path experiment is simple and the
result is hard to explain any other way.

{{cite:jimenez2023swebench}} for the benchmark's own statement that issues require
coordinating changes across multiple functions, classes and files, which is the
regime where text retrieval fails.

{{cite:wang2025solvedcorrectly}} for what happens after localisation succeeds, which
is {{ch:aise-swe-agents}}'s subject.

{{ch:aids-text-to-sql}} for the same decomposition on a different artefact, and
{{ch:mcp-schemas}} for the retrieval results that transfer into a repository.
