---
id: ops-versioning
number: 206
part: XXIV
tier: full
status: draft
requires: [period-destroys-attribution, rework-cost-is-set-by-detection-lateness,
           derived-copies-multiply-contradiction, semantic-failure-has-no-instrument]
provides: [reproducibility-is-a-product-over-artefacts, partial-coverage-buys-little,
           diagnosis-cost-grows-with-unpinned-artefacts,
           reproducibility-and-diagnosability-order-differently]
citations: [sculley2015, breck2017, paleyes2020deployment, gama2014]
---

## 1. Learning Objectives

By the end of this chapter you will be able to enumerate the artefacts that determine an
AI system's behaviour and distinguish them from the ones version control was designed
for; compute reproducibility as a product over artefact coverage and explain why it fails
at the weakest term; show why a partially-completed versioning programme buys almost
nothing, and why the same list justified as incident tooling pays incrementally; compute
the candidate space a diagnosis must search and relate it to the lifecycle period; and
explain why optimising for reproducibility and optimising for diagnosability produce
different orderings of the same work.

## 2. Why This Matters

{{ch:ops-lifecycle}} ended on attribution: at a 35-day period, fifteen changes are in
flight and a metric movement has fifteen candidates. This chapter is about what sits
underneath those fifteen, and the answer is worse than the count suggests.

An AI system's behaviour is determined by artefacts that conventional version control
does not cover. {{sec:9-practical-example}} enumerates ten and measures typical coverage:
application code is versioned **99%** of the time, the retrieval corpus **12%**. Since
reproducing a past run requires *every* determining artefact to have been pinned,
reproducibility is a product — and it comes out at **0.27%**
({{eq:reproducibility-is-a-product-over-artefacts}}).

That has an unwelcome consequence for planning. Half the effort of a full versioning
programme buys **10.30%** reproducibility against a complete programme's **100%**
({{eq:partial-coverage-buys-little}}). There is no eighty-twenty, because a product needs
every term.

But the same list justified differently pays differently. Diagnosis cost is the
*logarithm* of the candidate space, so each artefact pinned removes its own contribution
independently — and the candidate space in the worked example is **66,960** combinations
({{eq:diagnosis-cost-grows-with-unpinned-artefacts}}). **Reproducibility is
all-or-nothing; diagnosability is incremental**, and the two rank the same work in
different orders ({{eq:reproducibility-and-diagnosability-order-differently}}).

## 3. Prerequisites

You need {{eq:period-destroys-attribution}} from {{ch:ops-lifecycle}}: the number of
values each artefact took is a function of the window, and the window is the loop period.
The two results multiply.

{{eq:rework-cost-is-set-by-detection-lateness}} from the same chapter explains why the
cost lands during incidents rather than at the moment of change.

{{eq:derived-copies-multiply-contradiction}} from {{ch:sd-storage}} is the same corpus
problem seen from the consistency side; here it is seen from the reproducibility side,
and both conclude the corpus is the artefact nobody tracks.

{{eq:semantic-failure-has-no-instrument}} is why the regression being diagnosed was
noticed late in the first place.

## 4. Intuitive Explanation

Start with a question that sounds trivial and is not: what would you need to have
recorded in order to reproduce, exactly, an answer your system gave three weeks ago?

The code, obviously. The model weights, or the provider's model version if you did not
host it. So far this is ordinary software, and ordinary tooling covers it.

Now keep going. The system prompt — which lives in a string, was edited twice last week,
and is not in any changelog. The tool schemas, which were regenerated when someone added
a field. The retrieval corpus, which has had documents added and removed continuously
and has no snapshot. The index built over that corpus, rebuilt on a schedule nobody
recorded. The decoding temperature, which someone tuned. The evaluation set you compared
against, which grew.

Each of those changes the answer. Miss any one of them and you cannot reproduce the run —
not approximately, not "close enough". You get a different answer and you cannot say why.

That is the shape of the problem, and the shape is what makes it hard. **Reproducibility
is a conjunction.** Every artefact must be pinned. So the probability of reproducing a
past run is a product over how well each one is covered, and a product of ten mostly-good
numbers is a bad number. Four of the ten artefacts in {{sec:9-practical-example}} are
covered above 90%, and the product is under one percent.

This is the fourth or fifth time this book has met that arithmetic — {{ch:ag-loop}}'s
chain, {{ch:sd-retrieval-agents}}'s fan-out, {{ch:inf-distributed}}'s failure domain — and
it behaves the same way every time. The intuition that "most of it is fine" is exactly
the intuition a product punishes.

Now the planning consequence, which is genuinely awkward. Most engineering programmes can
be half-done usefully. This one cannot: half the effort buys a tenth of the outcome,
because whatever you did not finish caps the product. A versioning initiative that runs
out of sponsorship at the halfway point has produced very little that a reproducibility
audit would credit.

That is a hard thing to propose to a sponsor, and it is why the chapter's second half
matters. There is another way to justify exactly the same work, and it has the opposite
shape.

Think about what you actually do during an incident. Something is wrong. You want to know
what changed. Every un-pinned artefact contributes a set of possibilities that has to be
narrowed, and the possibilities multiply — seven unpinned artefacts in the worked example
give sixty-seven thousand combinations.

Nobody searches sixty-seven thousand hypotheses. What happens instead is that the team
tries the two or three they can think of, none of them is right, and the investigation
quietly stops. **The consequence of a large candidate space is not a long search. It is an
abandoned one.**

And here the arithmetic is friendlier. Search cost is logarithmic in the space, so each
artefact you pin removes its own contribution regardless of what else is pinned. Every
step pays. The same backlog, justified as incident tooling rather than as reproducibility,
becomes something you can fund one item at a time — and it happens to build the same
thing.

## 5. Formal Explanation

**Reproducibility.** Let the system's behaviour be determined by artefacts
$a \in A$, where artefact $a$ has influence $\iota_a$ — the probability that a change to
it alters the output — and coverage $\kappa_a$, the probability that it was pinned. A past
run reproduces if, for every artefact, either it was pinned or its unrecorded change did
not matter:

$$ \Pi \;=\; \prod_{a \in A}\Bigl(\kappa_a + (1 - \kappa_a)(1 - \iota_a)\Bigr) $$ (eq:reproducibility-is-a-product-over-artefacts)

Each term is at most one, so **$\Pi$ is bounded above by every individual term**. The
artefact with the smallest term caps the whole product regardless of the others, which is
the formal content of "one gap is enough."

Define artefact $a$'s **exposure** as $\iota_a(1 - \kappa_a)$ — influence times the
uncovered share. Raising $\kappa_a$ to one multiplies $\Pi$ by
$1/(\kappa_a + (1-\kappa_a)(1-\iota_a))$, which is increasing in exposure.

**Partial coverage.** Suppose artefacts are fixed in some order, and after covering a
subset $S$ the reproducibility is $\Pi(S)$. Since $\Pi$ is a product, the fraction of the
final value achieved is

$$ \frac{\Pi(S)}{\Pi(A)} \;=\; \prod_{a \notin S}\Bigl(\kappa_a + (1-\kappa_a)(1-\iota_a)\Bigr) $$ (eq:partial-coverage-buys-little)

**which depends only on what is left, not on what is done.** So progress is measured by
the remaining gaps, and the value curve is convex — nearly flat until the last items are
completed. {{sec:9-practical-example}} measures 10.30% at half the effort.

**Diagnosis.** During a window of length $T$, artefact $a$ took $v_a(T)$ distinct values.
A pinned artefact contributes one known value; an unpinned one contributes $v_a$. The
candidate space is

$$ N \;=\; \prod_{a \text{ unpinned}} v_a(T) $$ (eq:diagnosis-cost-grows-with-unpinned-artefacts)

and diagnosis time, under a search that bisects, is $c\log_2 N$. Since
$\log N = \sum \log v_a$, **pinning artefact $a$ removes exactly $c\log_2 v_a$ hours
regardless of what else is pinned** — an additive, order-independent contribution.

That is the structural difference between the two objectives. Reproducibility is $\Pi$;
diagnosability is $\log \Pi^{-1}$. The first is multiplicative and the second additive,
so:

$$ \text{rank}_{\text{repro}}(a) \propto \iota_a(1-\kappa_a), \qquad \text{rank}_{\text{diag}}(a) \propto \frac{\log v_a}{e_a} $$ (eq:reproducibility-and-diagnosability-order-differently)

for effort $e_a$. **The two orderings differ**, and {{sec:9-practical-example}} finds the
retrieval corpus first by exposure and sixth by payback.

## 6. Mathematical Foundation

The interaction with {{ch:ops-lifecycle}}'s period is worth deriving because it is the
chapter's most actionable composition.

The value count $v_a(T)$ is approximately linear in the window for any artefact that
changes at a steady rate $r_a$: $v_a(T) \approx r_a T$. So the candidate space is

$$ N(T) \;=\; \prod_{a \text{ unpinned}} r_a T \;=\; T^{|U|}\prod_{a \in U} r_a $$

where $U$ is the unpinned set. **The candidate space grows as the period raised to the
number of unpinned artefacts.** With seven unpinned, doubling the period multiplies the
space by $2^7 = 128$.

Diagnosis time is $c\log_2 N = c\bigl(|U|\log_2 T + \sum\log_2 r_a\bigr)$, so it grows
**linearly in the number of unpinned artefacts and logarithmically in the period**. That
asymmetry is useful: pinning an artefact removes a term from a sum that the period
multiplies, so the two interventions compose favourably. Shortening the period helps
every unpinned artefact at once; pinning an artefact removes it from the period's reach
permanently.

{{sec:9-practical-example}} measures the period effect at **16×** the candidate space
between a 3-day and a 35-day window — and only **1.3×** the diagnosis hours, because of
the logarithm. So on diagnosis time alone, the period matters less than the count of
unpinned artefacts, which is a useful ordering for a team that can only do one.

One caution about the log model. It assumes the search can bisect, which requires the
candidate space to be *orderable* — you can test "was it before or after this point".
That holds for time-ordered artefacts and fails for a set of independent configuration
values, where the search is closer to linear. The honest reading is that
$c\log_2 N$ is a lower bound and the truth lies between it and $cN$, which makes the case
for pinning stronger rather than weaker.

## 7. Internal Mechanics

**Why the corpus is the hardest artefact.** Pinning it means content-addressing every
document that contributed to the index at build time, which is a snapshot of a
continuously-changing collection rather than a commit of a file. Storage is cheap and the
plumbing is not: the corpus is usually assembled from several upstream systems, none of
which was built to emit an immutable version. This is
{{eq:derived-copies-multiply-contradiction}}'s derivation chain again, and the same
depth that made staleness expensive makes versioning expensive.

**Why the system prompt is the easiest and most neglected.** It is a string. Putting it in
a repository with the code costs an afternoon, and {{sec:9-practical-example}} ranks it
second by payback. It is neglected because it does not feel like code — it is edited by
people who are not committing, often through a console, and the change never passes a
review that would have caught it. This is exactly {{cite:sculley2015}}'s configuration
debt, with a shorter feedback loop and a larger blast radius.

**Model version is the artefact with the highest payback and the least agency.** When a
provider updates a model behind a stable name, the artefact changed and nobody on your
team did anything. Pinning it means requesting a dated version where the provider offers
one and recording the response metadata where they do not — cheap, and it converts an
unobservable change into a recorded one.

**Why coverage numbers are optimistic.** A team that says it versions its evaluation set
usually means the file is in a repository. Whether the *labels* were regenerated, whether
the sampling changed, and whether the set was extended are separate questions, and the
answer to all three is often unrecorded. **Coverage should be measured by whether a past
run can be reconstructed, not by whether a file exists.**

**Why "record everything" is not the same as versioning.** Many teams log requests and
responses and consider the problem addressed. That gives *inspectability* -- you can see
what happened -- without *reproducibility* -- you cannot make it happen again. The
distinction matters during an investigation with a counterfactual in it: "would this
request have succeeded under last week's prompt?" is answerable only by re-execution, and
a log cannot answer it. Since most interesting diagnostic questions are counterfactual,
logging is a partial substitute that runs out exactly when the investigation gets hard.
It is still worth having, and it should not be mistaken for the thing it resembles.

**Drift is the case where nothing changed and everything did.** {{cite:gama2014}}'s
concept drift is the input distribution moving under a fixed system. Versioning does not
prevent it and does make it diagnosable: with everything pinned, a behaviour change with
no artefact change is *evidence*, and it points at the data rather than the system. An
unversioned system cannot distinguish drift from an unrecorded edit, which is why drift
is so often misdiagnosed.

**{{cite:breck2017}}'s rubric assumes reproducibility.** Almost every test it proposes
requires being able to re-run something and compare. So the rubric is downstream of this
chapter: a team that cannot reproduce cannot score on most of it, and
{{cite:paleyes2020deployment}}'s finding that obstacles appear at every stage is partly a
consequence — an unreproducible pipeline turns every stage's problem into a new
investigation.

## 8. Implementation

The first listing enumerates the determining artefacts and computes reproducibility as a
product.

```python {tier=A name=eb1}
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
```

## 9. Practical Example

What determines behaviour, against what gets versioned:

```
                artefact   changes output   versioned   uncovered   exposure
------------------------------------------------------------------------------
        application code              95%         99%          1%       0.01
           model weights             100%         92%          8%       0.08
     model version / API             100%         58%         42%       0.42
           system prompt              98%         34%         66%       0.65
            tool schemas              71%         31%         69%       0.49
        retrieval corpus              88%         12%         88%       0.77
   retrieval index build              64%          9%         91%       0.58
          evaluation set              55%         22%         78%       0.43
     decoding parameters              79%         47%         53%       0.42
        library versions              42%         81%         19%       0.08
```

**The artefacts teams version well are the ones version control was built for.** The
corpus changes the output **88%** of the time and is versioned **12%** of the time.

The product:

```
         after including   this term   running product
--------------------------------------------------------
        application code       0.991            0.9905
           model weights       0.920            0.9113
     model version / API       0.580            0.5285
           system prompt       0.353            0.1867
            tool schemas       0.510            0.0952
        retrieval corpus       0.226            0.0215
   retrieval index build       0.418            0.0090
          evaluation set       0.571            0.0051
     decoding parameters       0.581            0.0030
        library versions       0.920            0.0027
```

**0.27%** ({{eq:reproducibility-is-a-product-over-artefacts}}). Four of the ten terms are
above 0.9, and the product is under one in three hundred.

Building coverage in payback order:

```
  step          artefact fixed  effort so far   reproducibility
------------------------------------------------------------------
     1        application code            0.0             0.28%
     2           system prompt            2.0             0.78%
     3     model version / API            3.0             1.35%
     4     decoding parameters            4.0             2.32%
     5        retrieval corpus           12.0            10.30%
     6          evaluation set           14.0            18.03%
     7            tool schemas           17.0            35.35%
     8   retrieval index build           22.0            84.66%
     9           model weights           23.0            92.02%
    10        library versions           24.0           100.00%
```

**Half the effort buys a tenth of the outcome**
({{eq:partial-coverage-buys-little}}). There is no eighty-twenty in a conjunction.

```mermaid {#fig:product caption="Reproducibility is a conjunction over artefacts, so it is capped by the weakest term and partial coverage buys little. Diagnosability is the logarithm of the same quantity, so every step pays."}
flowchart TD
  A["ten determining artefacts"] --> B["reproducibility<br/>product of coverage<br/>0.27%"]
  A --> C["candidate space<br/>product of values<br/>66,960"]
  B --> D["all-or-nothing<br/>half effort = 10%"]
  C --> E["log of the space<br/>additive per artefact"]
  E --> F["incrementally fundable"]
```

And what one omission costs:

```
        omitted artefact   reproducibility     vs full
--------------------------------------------------------
        application code            99.05%        99%
     model version / API            58.00%        58%
           system prompt            35.32%        35%
        retrieval corpus            22.56%        23%
```

**Versioning everything except the corpus leaves reproducibility at 22.56%.** One gap
with high influence caps the product regardless of the rest.

The second listing prices what that costs during incidents.

```python {tier=A name=eb2}
"""Un-versioned artefacts are paid for during incidents, at a rate set by search.

The previous listing measured reproducibility as a probability. This one measures what
the missing coverage costs, and the cost is not paid when the artefact changes -- it is
paid weeks later when something is wrong and nobody can say what moved.

Diagnosis is a search over candidate causes. A versioned artefact contributes one known
value; an unversioned one contributes a range that must be explored. So diagnosis time
grows with the size of the candidate space, which grows multiplicatively in the number of
unpinned artefacts (eq:diagnosis-cost-grows-with-unpinned-artefacts).

ch:ops-lifecycle found 15 changes in flight at a 35-day period. This listing shows what
those 15 become when the artefacts underneath them are not pinned.
"""
import math

# (artefact, versioned?, distinct values it could have taken in the window)
ARTEFACTS = [
    ("application code",      True,   14),
    ("model weights",         True,    2),
    ("model version / API",   False,   3),
    ("system prompt",         False,   9),
    ("tool schemas",          False,   4),
    ("retrieval corpus",      False,  31),
    ("retrieval index build", False,   2),
    ("evaluation set",        False,   5),
    ("decoding parameters",   False,   2),
    ("library versions",      True,    6),
]
BISECT_HOURS = 2.6           # cost of testing one hypothesis
INCIDENTS_PER_YEAR = 9.0


def candidates(pinned_set):
    """Size of the space a diagnosis must search."""
    n = 1
    for name, versioned, vals in ARTEFACTS:
        if name in pinned_set or versioned:
            n *= 1          # pinned: exactly one known value
        else:
            n *= vals
    return n


def hours(n_cand):
    """Bisection over an ordered space is log2; unordered is linear in the worst
    case. Real diagnosis is between: assume log2 within an artefact and linear
    across artefacts that must be disambiguated."""
    return BISECT_HOURS * math.log(max(n_cand, 1), 2)


print("Artefacts, and how many distinct values each took during the window a")
print("regression could have been introduced.")
print()
print(f"{'artefact':>24}{'versioned':>12}{'values in window':>19}"
      f"{'contributes':>14}")
print("-" * 72)
for name, versioned, vals in ARTEFACTS:
    print(f"{name:>24}{('yes' if versioned else 'no'):>12}{vals:>19}"
          f"{(1 if versioned else vals):>14}")

base = candidates(set())
print()
print(f"candidate space as-is: {base:,}")
print(f"if everything were pinned: 1")

print()
print()
print("Diagnosis cost, as artefacts are pinned one at a time.")
print()
unpinned = [a for a in ARTEFACTS if not a[1]]
order = sorted(unpinned, key=lambda a: -a[2])
print(f"{'pinned so far':>34}{'candidates':>14}{'diagnosis hrs':>16}"
      f"{'hrs/year':>11}")
print("-" * 78)
pin = set()
path = [(0, "nothing", base, hours(base))]
print(f"{'nothing':>34}{base:>14,}{hours(base):>16.1f}"
      f"{hours(base) * INCIDENTS_PER_YEAR:>11.0f}")
for name, versioned, vals in order:
    pin.add(name)
    c = candidates(pin)
    path.append((len(pin), name, c, hours(c)))
    print(f"{('+ ' + name):>34}{c:>14,}{hours(c):>16.1f}"
          f"{hours(c) * INCIDENTS_PER_YEAR:>11.0f}")

print()
print()
print("What each artefact costs per year, left unpinned.")
print()
print(f"{'artefact':>24}{'values':>9}{'hrs/incident':>15}"
      f"{'hrs/year':>11}{'effort to fix':>16}")
print("-" * 76)
EFFORT = {"model version / API": 1.0, "system prompt": 2.0, "tool schemas": 3.0,
          "retrieval corpus": 8.0, "retrieval index build": 5.0,
          "evaluation set": 2.0, "decoding parameters": 1.0}
cost = {}
for name, versioned, vals in unpinned:
    # Marginal cost: the extra search this artefact alone adds.
    with_it = hours(base)
    without = hours(base / vals)
    cost[name] = (vals, with_it - without,
                  (with_it - without) * INCIDENTS_PER_YEAR, EFFORT[name])
    print(f"{name:>24}{vals:>9}{with_it - without:>15.2f}"
          f"{(with_it - without) * INCIDENTS_PER_YEAR:>11.1f}"
          f"{EFFORT[name]:>16.1f}")

print()
print()
print("Payback: annual hours saved against the effort to pin it.")
print()
rank = sorted(cost, key=lambda k: -(cost[k][2] / cost[k][3]))
print(f"{'rank':>6}{'artefact':>24}{'hrs saved/yr':>15}{'effort':>9}"
      f"{'payback ratio':>16}")
print("-" * 72)
for i, k in enumerate(rank, 1):
    print(f"{i:>6}{k:>24}{cost[k][2]:>15.1f}{cost[k][3]:>9.1f}"
          f"{cost[k][2] / cost[k][3]:>16.1f}")

print()
print()
print("And the interaction with ch:ops-lifecycle's changes-in-flight. A longer")
print("period means more values per artefact, which compounds multiplicatively.")
print()
print(f"{'period days':>13}{'code versions':>16}{'corpus versions':>18}"
      f"{'candidates':>14}{'diagnosis hrs':>16}")
print("-" * 78)
per = {}
for days in (3.0, 7.0, 14.0, 21.0, 35.0):
    code_v = max(1, int(days * 3.0 / 7.0))          # 3 changes a week
    corpus_v = max(1, int(days * 0.9))              # corpus updates daily-ish
    n = 1
    for name, versioned, vals in ARTEFACTS:
        if versioned:
            continue
        if name == "retrieval corpus":
            n *= corpus_v
        else:
            n *= vals
    per[days] = (code_v, corpus_v, n, hours(n))
    print(f"{days:>13.0f}{code_v:>16}{corpus_v:>18}{n:>14,}"
          f"{hours(n):>16.1f}")

print(f"""
The candidate table is the cost of the previous listing's missing coverage, expressed
in the units an incident is measured in. With seven artefacts unpinned and the value
counts shown, a regression could have been introduced by any of **{base:,} distinct
combinations** (eq:diagnosis-cost-grows-with-unpinned-artefacts).

That number is not a search space anyone works through, which is the point. Nobody
enumerates {base:,} hypotheses. What actually happens is that the team
narrows to the two or three they can think of, tries those, and if none is right the
investigation stalls -- so the practical consequence of a large candidate space is not a
long search but an **abandoned** one.

The pinning path prices each step. Pinning `{order[0][0]}` alone takes the space from
{base:,} to {path[1][2]:,} and diagnosis from {path[0][3]:.1f} to {path[1][3]:.1f} hours.
Pinning all seven reaches {path[-1][2]:,} and {path[-1][3]:.1f} hours.

Note the shape, because it is the opposite of the previous listing's. Reproducibility
was a product, and half the effort bought a tenth of the outcome. Diagnosis cost is the
*logarithm* of that product, so **each artefact pinned removes its own log-value from
the total, independently of what else is pinned** -- the reduction is additive rather
than multiplicative.

That difference decides how the work should be justified. If the goal is exact
reproducibility, the programme is all-or-nothing and a half-finished one is nearly
worthless. If the goal is diagnosable incidents, **every step pays its own way**, and
the same list becomes an incrementally-fundable backlog.

Since the second goal is the one that shows up in an incident review, it is usually the
easier case to make -- and it happens to build the same thing.

The per-artefact table ranks by annual cost. `{rank[0]}` is worth
{cost[rank[0]][2]:.1f} hours a year and costs {cost[rank[0]][3]:.1f} to fix, a payback
ratio of {cost[rank[0]][2] / cost[rank[0]][3]:.1f}. `{rank[-1]}` is worth
{cost[rank[-1]][2]:.1f} hours against {cost[rank[-1]][3]:.1f}, a ratio of
{cost[rank[-1]][2] / cost[rank[-1]][3]:.1f}.

**The ranking is not the same as the exposure ranking in the previous listing.** There
the retrieval corpus led because its influence and coverage gap were largest; here the
ordering is driven by value count against effort, and the cheap artefacts with many
versions rise. A team optimising for reproducibility and a team optimising for
diagnosability should build the same things in a different order.

The period table is where this chapter meets ch:ops-lifecycle. The number of distinct
values an artefact took is a function of how long the window is, and the window is the
loop period. At a {3.0:.0f}-day period the candidate space is {per[3.0][2]:,}; at
{35.0:.0f} days it is {per[35.0][2]:,} --
{per[35.0][2] / per[3.0][2]:.0f} times larger.

**The period multiplies the candidate space, and the candidate space is what makes an
incident undiagnosable.** ch:ops-lifecycle argued that a long period destroys
attribution; this table is the mechanism. It is not that fifteen changes are hard to
distinguish. It is that fifteen changes sit on top of thirty-one corpus versions and
nine prompt edits, and the product is what has to be searched.

So the two interventions compose in the direction you would want: shortening the period
reduces the values per artefact, and pinning the artefacts removes them from the product
entirely. Doing both takes the candidate space from {per[35.0][2]:,} to {1} -- there is nothing
to search, because the change that caused it is identified directly.

The practical reading is that **versioning is incident-response tooling that happens to
run continuously.** It is funded as hygiene and it is paid for during outages, which is
why it is chronically under-resourced: the cost lands on a different quarter, a different
metric, and frequently a different team from the one that would have done the work.""")
```

```
                artefact   versioned   values in window   contributes
------------------------------------------------------------------------
        application code         yes                 14             1
     model version / API          no                  3             3
           system prompt          no                  9             9
            tool schemas          no                  4             4
        retrieval corpus          no                 31            31
   retrieval index build          no                  2             2
          evaluation set          no                  5             5
     decoding parameters          no                  2             2
```

**66,960 candidate combinations**
({{eq:diagnosis-cost-grows-with-unpinned-artefacts}}). Nobody searches that; the team
tries two or three hypotheses and the investigation stalls.

```
                     pinned so far    candidates   diagnosis hrs   hrs/year
------------------------------------------------------------------------------
                           nothing        66,960            41.7        375
                + retrieval corpus         2,160            28.8        259
                   + system prompt           240            20.6        185
                  + evaluation set            48            14.5        131
                    + tool schemas            12             9.3         84
             + model version / API             4             5.2         47
           + retrieval index build             2             2.6         23
             + decoding parameters             1             0.0          0
```

**Each step pays its own way**, because diagnosis cost is the logarithm of a product and
a logarithm turns products into sums.

```
  rank                artefact   hrs saved/yr   effort   payback ratio
------------------------------------------------------------------------
     1     model version / API           37.1      1.0            37.1
     2           system prompt           74.2      2.0            37.1
     3          evaluation set           54.3      2.0            27.2
     4     decoding parameters           23.4      1.0            23.4
     5            tool schemas           46.8      3.0            15.6
     6        retrieval corpus          115.9      8.0            14.5
     7   retrieval index build           23.4      5.0             4.7
```

**The retrieval corpus is first by exposure and sixth by payback**
({{eq:reproducibility-and-diagnosability-order-differently}}) — the two objectives order
the same work differently.

And the composition with {{ch:ops-lifecycle}}:

```
  period days   code versions   corpus versions    candidates   diagnosis hrs
------------------------------------------------------------------------------
            3               1                 2         4,320            31.4
            7               3                 6        12,960            35.5
           14               6                12        25,920            38.1
           21               9                18        38,880            39.6
           35              15                31        66,960            41.7
```

The period multiplies the candidate space **16×** between 3 and 35 days — and diagnosis
hours only **1.3×**, because of the logarithm. **On diagnosis time, the count of unpinned
artefacts matters more than the period**, which orders the two interventions.

## 10. Production Considerations

Enumerate the determining artefacts. Most teams have never written the list down, and the
enumeration is worth more than any single fix because it surfaces terms nobody was
counting.

Justify the programme as incident tooling, not as reproducibility. The same backlog is
all-or-nothing under one framing and incrementally fundable under the other, and only one
of those gets funded.

Pin the model version first. It is the cheapest item, it has the highest payback ratio,
and it is the one artefact that changes without anyone on your team acting.

Put the system prompt in the repository with the code. It is a string, it is second by
payback, and it is edited by people who are not committing.

Content-address the corpus at index build time, and record which build served each
request. It is the most expensive item and the one that caps the product; defer it
knowingly rather than by omission.

Measure coverage by reconstructability, not by file existence. "The evaluation set is in
git" and "we can rebuild the evaluation set as it was" are different claims.

Budget the recurring cost separately from the build cost. Corpus and index versioning
are paid on every build, and a plan estimated in one-time effort will under-resource
exactly the high-influence items.

Record the artefact set with every served request, not just at deploy time. Under
continuous deployment the deploy-time snapshot is not what served the request that is
being investigated.

## 11. Common Mistakes

**Assuming code coverage is coverage.** Code is the artefact version control was built
for, and it contributes the smallest term in the product.

**Reading progress as a fraction of the list.** Progress is measured by what remains,
because the value is a product.

**Funding the programme on reproducibility.** It is all-or-nothing and it will be
half-funded.

**Ranking work by exposure when the goal is diagnosability.** The orderings differ.

**Treating "the file is in git" as versioned.** Reconstructability is the test.

**Diagnosing drift in an unversioned system.** You cannot distinguish drift from an
unrecorded edit, so every data-distribution problem is indistinguishable from a
configuration one.

## 12. Failure Modes

**Abandoned investigation.** The candidate space is large enough that the team tries the
obvious hypotheses, misses, and closes the incident as unexplained.

**Silent provider model update.** The behaviour changes, no artefact under the team's
control changed, and nothing in the system records that anything happened.

**Half-finished versioning programme.** Substantial effort spent, reproducibility still
near zero, and the sponsor concludes versioning does not work.

**Corpus reconstruction failure.** A snapshot exists but its upstream sources have been
mutated, so the snapshot cannot actually be rebuilt — reproducibility that is nominal
rather than real.

**Prompt edited in a console.** No commit, no review, no record, and the behaviour
change appears three weeks later as an unexplained metric movement — by which time
the person who made it has forgotten, and nothing in the system remembers.

## 13. Alternatives

**Reduce the determining set.** Remove artefacts rather than pinning them: a fixed
corpus snapshot per release, a pinned model version, a prompt that is genuinely
static. Cheaper than versioning, and it changes the system rather than the tooling —
which is usually the more durable fix, since an artefact that cannot vary needs no
machinery to record that it did not.

**Record inputs and outputs rather than artefacts.** Keep the request, the retrieved
context, and the response, so a past run can be *inspected* even if it cannot be
*re-executed*. Much cheaper and sufficient for many investigations — until the
question becomes counterfactual, which is when investigations get hard. Worth having
and not a substitute.

**Shorten the period instead.** {{ch:ops-lifecycle}}'s intervention reduces every value
count at once. Weaker per unit effort on diagnosis time because of the logarithm, and it
helps the artefacts you have not got to yet.

**Full deterministic replay.** Pin everything and re-execute, which is what this chapter
describes done completely. Correct, expensive, and appropriate where the cost of an
unexplained regression is high.

**Accept unreproducibility and invest in bisection tooling.** The
{{ch:ops-lifecycle}} answer applied here: if the candidate space cannot be shrunk, make
searching it cheaper.

## 14. Evaluation

Compute and publish the reproducibility product. It is a single number derived from a
list, it is comprehensible to a sponsor, and almost no team has it.

Track candidate-space size as a standing metric. It moves with the period and with
coverage, and it predicts how an incident will go.

Measure actual reproduction success on a sample of past requests. The product is an
estimate; attempting ten reproductions is a measurement, and the two frequently disagree
in the pessimistic direction.

Record time-to-attribution per incident and compare it to $c\log_2 N$. A large gap means
the search is not bisecting, and the linear bound applies.

Audit coverage claims by attempting a reconstruction, not by checking for a file.

## 15. Advanced Concepts

The independence assumption in the product is optimistic. Artefacts change together —
a prompt edit often accompanies a tool-schema change, and a corpus refresh often
accompanies an index rebuild — so the effective number of independent unpinned dimensions
is smaller than the count suggests. That makes the candidate space smaller than
{{eq:diagnosis-cost-grows-with-unpinned-artefacts}} implies and reproducibility higher
than the product implies. It also makes attribution *harder* in a different way: when two
artefacts always change together, pinning one does not disambiguate, and the pair must be
treated as a single artefact with the product of their value counts.

The influence parameter $\iota_a$ is treated as a constant, but it is conditional on the
rest of the system. A decoding-temperature change matters enormously for a creative
task and not at all for a constrained extraction task, so a single system with several
surfaces has several different products. The honest treatment computes reproducibility
per surface, which is more work and gives a more useful answer — some surfaces are
already reproducible and some are hopeless, and a single number averages them into
something true of neither.

The effort figures also hide a distinction between one-time and recurring cost.
Pinning a system prompt is a one-time change to where it lives; pinning a corpus is a
recurring cost paid on every index build, in storage and in build time. A plan built on
one-time effort estimates will under-resource the recurring items, and the recurring
items are disproportionately the high-influence ones. **The right unit for this backlog
is cost per year, not cost to build**, and the two rank differently for the same reason
{{eq:reproducibility-and-diagnosability-order-differently}} does.

There is an unexplored relationship between versioning and
{{ch:sd-architecture}}'s interleaving result. A pipeline with more deterministic stages
has more artefacts whose behaviour is exactly reproducible given their inputs, so
interleaving does not merely restore testability — it restores *reproducibility*, for the
same reason. That suggests the determining-artefact count is partly an architectural
choice rather than a fact about the system, and a design that concentrates
nondeterminism into fewer stages is easier to version as well as easier to test.

## 16. Connection to Previous Chapters

{{eq:period-destroys-attribution}} from {{ch:ops-lifecycle}} counted changes in flight.
This chapter counts what sits underneath each one, and the two multiply into the
candidate space.

{{eq:derived-copies-multiply-contradiction}} from {{ch:sd-storage}} found the corpus's
derivation chain expensive for consistency. Here the same chain is expensive for
versioning, and both point at the same neglected artefact.

{{eq:loop-is-not-a-chain}} from {{ch:ag-loop}} appears again as
{{eq:reproducibility-is-a-product-over-artefacts}}. Seventh instance; same arithmetic.

{{eq:semantic-failure-has-no-instrument}} from {{ch:sd-architecture}} explains why the
regression was found late, which is why the window is long, which is why the candidate
space is large.

## 17. Exercises

1. Enumerate the determining artefacts for a system you work on. How many are there, and
   what is the reproducibility product?

2. Show that $\Pi(S)/\Pi(A)$ depends only on the uncovered set, and explain what that
   implies for progress reporting.

3. Compute the candidate space for your own system over a 30-day window. What would it
   be with the two cheapest artefacts pinned?

4. Derive the condition under which the exposure ordering and the payback ordering agree.
   How often does it hold?

5. Model two artefacts that always change together. How does that change both the
   product and the candidate space?

## 18. Interview Questions

1. What would you need to have recorded to reproduce an answer from three weeks ago?

2. We version our code and our model. What is our reproducibility, roughly, and why?

3. Why does a half-completed versioning programme buy almost nothing?

4. Same list of work, two justifications. Why does one get funded and the other not?

5. Our provider updated a model behind a stable name and our metrics moved. What should
   have been in place?

6. We log every request and response. Does that give us reproducibility? What can and
   cannot be answered from a log?

## 19. Research Questions

1. How correlated are artefact changes in practice, and what does that do to the
   effective candidate space?

2. Can reproducibility be measured directly at scale — sampled re-execution of past
   requests — cheaply enough to run continuously?

3. How much does per-surface influence vary, and is a single reproducibility figure ever
   the right summary?

4. Does concentrating nondeterminism into fewer stages measurably improve
   reproducibility, as {{sec:15-advanced-concepts}} conjectures?

## 20. Chapter Summary

An AI system's behaviour is determined by artefacts version control was not built for.
Application code is versioned **99%** of the time; the retrieval corpus **12%**, while
changing the output **88%** of the time.

Reproducibility is a conjunction, so it is a product over coverage — **0.27%** across ten
artefacts of which four are covered above 90%
({{eq:reproducibility-is-a-product-over-artefacts}}). Omitting the corpus alone caps it at
**22.56%**.

That makes the programme all-or-nothing: half the effort buys **10.30%** against a
complete programme's **100%**, because the fraction achieved depends only on what remains
({{eq:partial-coverage-buys-little}}).

The same list justified as incident tooling behaves differently. The candidate space a
diagnosis must search is **66,960** combinations, and diagnosis cost is its logarithm — so
**each artefact pinned removes its own contribution independently**
({{eq:diagnosis-cost-grows-with-unpinned-artefacts}}), and every step pays.

The two objectives order the same work differently: the corpus is first by exposure and
**sixth** by payback ({{eq:reproducibility-and-diagnosability-order-differently}}). And
the period multiplies the candidate space **16×** between 3 and 35 days while raising
diagnosis hours only **1.3×**, so unpinned-artefact count matters more than period.

There is a broader point in the gap between the two framings, and it is worth carrying
beyond versioning. The same body of work had a convex value curve under one objective
and a concave one under another, without changing at all. Which curve a programme is
on is not a property of the work; it is a property of what the work is being measured
against. A team that cannot get a backlog funded has sometimes chosen the wrong
objective to measure it by, and the remedy is to find the objective under which the
same items pay incrementally.

Carry forward: **write down what determines your system's behaviour**, and **fund the
list as incident tooling, because that framing pays incrementally**.

## 21. Further Reading

- {{cite:sculley2015}} — configuration debt, of which the unversioned prompt is the
  purest modern instance.
- {{cite:breck2017}} — a readiness rubric that presupposes the reproducibility this
  chapter builds.
- {{cite:paleyes2020deployment}} — obstacles at every stage, partly a consequence of
  pipelines that cannot be re-run.
- {{cite:gama2014}} — concept drift, which versioning does not prevent and does make
  diagnosable.
