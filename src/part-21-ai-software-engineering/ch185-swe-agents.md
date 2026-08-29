---
id: aise-swe-agents
number: 185
part: XXI
tier: full
status: draft
requires: [localisation-caps-the-rest, retry-needs-a-verifier,
           components-interact-superadditively, guards-cost-the-metric]
provides: [tests-are-a-partial-specification, honest-verifiers-lower-the-metric,
           inflation-scales-with-capability, scaffold-components-interact,
           scaffold-beats-model-improvement]
citations: [jimenez2023swebench, wang2025solvedcorrectly, chan2024mlebench,
            liang2025swebenchillusion, shinn2023reflexion]
---

## 1. Learning Objectives

By the end of this chapter you will be able to explain why passing a repository's
test suite is evidence proportional to its coverage rather than proof of
correctness; quantify the inflation that produces in a reported resolution rate;
explain why the verifier that would remove it lowers the headline number; show that
inflation grows in absolute terms as agents improve; ablate a SWE agent's scaffold
correctly; and state why scaffolding is worth more than a large model improvement.

## 2. Why This Matters

Software engineering has the strongest verifier in this book. Tests execute.
{{cite:jimenez2023swebench}} grades a patch by running the repository's own suite,
which is why coding agents are further along than agents in any other domain and why
this part is the least sceptical in the book.

This chapter measures where that verifier stops.

{{cite:wang2025solvedcorrectly}} applied differential testing to patches from three
state-of-the-art tools on SWE-bench Verified and found **$29.6\%$ of plausible
patches behaving differently from the ground-truth human patch**, $28.6\%$ of those
confirmed incorrect on inspection, $7.8\%$ of patches counted correct while *failing*
the developer-written suite, and reported resolution rates inflated by about **$6.2$
percentage points**.

The mechanism is not a grading defect. A test suite is a **partial specification** —
it pins down the behaviour someone thought to write a test for, and a patch that
satisfies it may do anything on the rest ({{eq:tests-are-a-partial-specification}}).
{{sec:9-practical-example}} reproduces $4.3$ points of inflation from independent
assumptions.

Differential testing against the reference closes most of the gap, and it has the
property that explains its absence: **it lowers the reported number by $2.8$ points
while leaving the number of correct patches exactly where it was**
({{eq:honest-verifiers-lower-the-metric}}). That is {{ch:aids-automl}}'s leakage
guard in a new setting, and its second appearance is enough to state as a general
hazard.

And a result that matters for the next few years: **inflation grows with capability**
({{eq:inflation-scales-with-capability}}). As the plausible-patch rate rises from
$20\%$ to $85\%$, precision holds near $85\%$ and the absolute inflation rises from
$2.1$ to $9.2$ points.

The second listing turns to what makes these agents work at all, and finds
{{cite:chan2024mlebench}}'s result: **the scaffold is worth more than a large model
improvement.**

## 3. Prerequisites

{{ch:aise-repo}}'s {{eq:localisation-caps-the-rest}} — this chapter takes up what
happens after localisation succeeds.

{{ch:ag-recovery}}'s {{eq:retry-needs-a-verifier}}, which is why the agent loop works
here and not in {{ch:aids-agentic-eda}}.

{{ch:as-single-agent}}'s {{eq:components-interact-superadditively}} and
{{eq:ablate-not-add}}, whose methodology the scaffold ablation requires.

{{ch:aids-automl}}'s {{eq:guards-cost-the-metric}}, of which this chapter's
differential-testing result is the second instance.

## 4. Intuitive Explanation

An agent is given an issue, edits some files, and runs the tests. They pass. The
benchmark records a resolution.

Consider what has actually been established. The repository's tests encode the
behaviour its maintainers thought to check. The patch satisfies those checks. On
every behaviour the suite does not exercise — which is most behaviours, in most
codebases — the patch is unconstrained.

So "the tests pass" means "the patch agrees with the specification, on the part of
the specification that was written down". That is real evidence and it is
proportional to coverage.

{{cite:wang2025solvedcorrectly}} measured the gap directly by running the agent's
patch and the human's patch side by side on generated inputs and comparing behaviour.
Nearly a third of plausible patches diverged, and of those, roughly a third were
outright wrong on inspection — some implementing a similar-but-different fix, some
changing more than the issue asked for.

That is the characteristic failure of this whole book, arriving in the domain with
the best verifier: **plausible, well-formed, passes every check anyone runs, and
different from what was wanted.**

The fix exists and is exactly what that paper built: compare against a reference.
{{sec:9-practical-example}} shows it cutting inflation substantially.

The fix also has an awkward property. Since it removes patches that were being
counted as successes, **it lowers your reported resolution rate while changing
nothing about how many patches are correct.** A benchmark that adopted it would score
lower than one that did not, and a team that adopted it would report a regression.

That is the same structure {{ch:aids-automl}} found with leakage guards, and its
second independent appearance suggests a general rule: **when the metric is produced
by the thing being measured, improving the measurement is penalised.**

Then a result about the future rather than the present. The proportion of passing
patches that are wrong stays roughly constant as agents improve — but the *number* of
patches grows. So a more capable agent produces more correct patches and more
plausible-wrong ones, in the same ratio, which means the absolute volume of
subtly-wrong code entering repositories rises with capability unless verification
rises with it.

The second half of the chapter is more encouraging.

A SWE agent is a loop, and the loop has parts: reproduce the bug, find the files,
edit, run the tests, iterate on the result, revert what made things worse.
{{sec:9-practical-example}} ablates them and finds two components worth
approximately *nothing* on their own and eighteen points each in place — because
running tests is useless if you cannot act on the result, and iterating is useless if
nothing tells you whether the last attempt worked.

Which means a team building this incrementally, measuring as it goes, will find the
test runner registering no benefit and remove it — removing the precondition for the
iteration that was about to matter.

And the comparison that should govern where effort goes: a model $20\%$ better with
no scaffold reached $61.0\%$; the baseline model with a full scaffold reached
$91.0\%$.

## 5. Formal Explanation

**Partial specification.** Let a patch be plausible with probability $\pi$ and
correct given plausible with probability $\rho$. A test suite covering a fraction
$\kappa$ of the intended behaviour catches an incorrect patch with probability
$\kappa$. Then:

$$P_{\text{report}} = \pi\big(\rho + (1-\rho)(1-\kappa)\big), \qquad P_{\text{true}} = \pi\rho$$

$$\text{inflation} = \pi(1-\rho)(1-\kappa)$$ (eq:tests-are-a-partial-specification)

**Inflation is proportional to $(1-\kappa)$**, the uncovered share. It vanishes only
at complete coverage, which no repository has, and the precision among passing
patches is:

$$\text{prec} = \frac{\rho}{\rho + (1-\rho)(1-\kappa)}$$

independent of $\pi$ — which is why the listing's precision column is flat.

**Differential testing.** A check comparing behaviour against the reference patch
catches a fraction $\delta$ of the divergent survivors:

$$P_{\text{report}}' = \pi\big(\rho + (1-\rho)(1-\kappa)(1-\delta)\big), \qquad P_{\text{true}}' = \pi\rho$$

$$\frac{\partial P_{\text{report}}}{\partial \delta} < 0, \qquad \frac{\partial P_{\text{true}}}{\partial \delta} = 0$$ (eq:honest-verifiers-lower-the-metric)

**The honest verifier moves the reported number down and the true number not at
all**, which is {{eq:guards-cost-the-metric}}'s sign pattern exactly.

**Scaling with capability.** Since inflation is $\pi(1-\rho)(1-\kappa)$ and precision
is independent of $\pi$:

$$\frac{\partial \,\text{inflation}}{\partial \pi} = (1-\rho)(1-\kappa) > 0, \qquad \frac{\partial\, \text{prec}}{\partial \pi} = 0$$ (eq:inflation-scales-with-capability)

**A more capable agent produces proportionally the same and absolutely more
plausible-wrong patches.** The rate of subtly-wrong code entering repositories
therefore scales with agent capability unless $\kappa$ or $\delta$ scales with it.

**Scaffold interaction.** Let the loop's components be a set $C$. Write $S(C)$ for
resolution. Two components $a, b$ are *mutually contingent* when each is a
precondition for the other's effect:

$$S(\{a\}) - S(\varnothing) \approx 0, \quad S(\{b\}) - S(\varnothing) \approx 0, \quad S(\{a,b\}) - S(\varnothing) \gg 0$$ (eq:scaffold-components-interact)

Running tests and iterating are exactly this pair: a verifier with no loop produces a
signal nobody acts on, and a loop with no verifier is
{{eq:retry-needs-a-verifier}}'s resample. So the one-at-a-time measurement of each is
zero and the joint measurement is large — and **ablation from the full system
recovers the true contribution while addition to the empty one does not.**

**Scaffold against model.** Writing $\mu$ for model skill:

$$S(C_{\text{full}}, \mu) - S(\varnothing, \mu) \gg S(\varnothing, \mu') - S(\varnothing, \mu) \quad\text{for realistic } \mu' > \mu$$ (eq:scaffold-beats-model-improvement)

which is {{cite:chan2024mlebench}}'s finding, and which holds because the scaffold
addresses failures the model's per-attempt skill does not touch: not knowing where to
look, not knowing whether the attempt worked, and not being able to undo damage.

## 6. Mathematical Foundation

Three extractions.

**Precision is independent of capability and inflation is not.** From
{{eq:inflation-scales-with-capability}}, the two most natural summary statistics move
differently: a team reporting precision sees no change as its agent improves, and a
repository receiving the patches sees the absolute defect volume grow. **Which
statistic you report determines whether you notice.**

**The honest-verifier penalty is a sign result, not a magnitude one.**
{{eq:honest-verifiers-lower-the-metric}} says $\partial P_{\text{report}}/\partial
\delta < 0$ unconditionally. There is no coverage level or agent quality at which
better verification improves the headline, which is why the incentive is structural
rather than a matter of degree.

**Mutual contingency makes one-at-a-time measurement return zero.**
{{eq:scaffold-components-interact}} is a stronger statement than superadditivity: it
says the individual measurements are approximately *zero*, not merely understated. A
component that measures zero is not kept, so the standard methodology does not
under-invest in these components — it deletes them.

## 7. Internal Mechanics

### 7.1 The four ways a passing patch is wrong

```mermaid {#fig:patch-divergence caption="A patch that passes the suite. Only the first branch is what the benchmark records it as, and the suite cannot separate them."}
flowchart TD
    P[patch passes the suite] --> A{compared with the human patch}
    A -->|same behaviour| OK["correct"]
    A -->|different| D[divergent]
    D --> D1["similar but different implementation<br/>often fine"]
    D --> D2["broader change than asked<br/>new behaviour nobody wanted"]
    D --> D3["narrower fix<br/>handles the reported case only"]
    D --> D4["passes by coincidence<br/>the test did not exercise the bug"]
```

{{cite:wang2025solvedcorrectly}} reports $46.8\%$ of divergent patches as
similar-but-different implementations and $27.3\%$ as making broader behavioural
changes than the reference. The first group is mostly harmless; the second is the one
that matters, because **a patch that does more than the issue asked has changed
behaviour nobody reviewed.**

That is {{ch:aise-generation}}'s scope creep, at the granularity of a whole change.

### 7.2 Differential testing, concretely

The mechanism {{cite:wang2025solvedcorrectly}} built is worth describing because it
is implementable by any team with a reference implementation.

Take the function or module the patch touches. Generate inputs — fuzzing, property
generation, or replayed production traffic. Run the pre-patch version, the
agent's patch, and (in evaluation) the reference patch. Compare outputs.

Divergence between the agent's patch and the reference on any input is a signal,
and divergence between the agent's patch and the *pre-patch* version on inputs
unrelated to the issue is a stronger one: it means the patch changed behaviour it had
no business changing.

That second form does not need a reference patch, which makes it available in
production rather than only in evaluation. **A patch should change behaviour on the
reported case and nowhere else, and that is checkable without knowing the right
answer.**

It is the strongest verification technique in this part and it is close to unused.

### 7.3 Why the incentive runs the wrong way

{{eq:honest-verifiers-lower-the-metric}} has an organisational consequence worth
being explicit about, since this is now its second appearance.

A benchmark that adopted differential testing would report lower numbers than one
that did not, and would be perceived as harder rather than as more accurate. A
vendor that adopted it internally would watch its resolution rate fall. A team that
adopted it would report a regression to management.

None of those parties is behaving badly. The structure is that **the measurement is
produced by the entity being measured**, so any improvement in measurement quality
appears as a decline in performance.

The available responses are the ones that separate the two. Report both numbers —
raw and differentially-verified — so the improvement is legible as an improvement.
Or have the verification run by a party that does not report the performance, which
is {{ch:aids-autonomous}}'s independence argument at an organisational level.

### 7.4 Building the scaffold in the right order

{{sec:9-practical-example}}'s build-up table gives an order, and it is not the order
teams use.

**Reproduce first.** {{ch:aise-repo}} found it the best localiser; this chapter finds
it worth $+11.8$ points even added last, because it does two jobs.

**Then localisation**, since it caps everything.

**Then the test runner and the iteration loop together.** Not separately — they are
mutually contingent, and measuring either alone returns zero.

**Then revert.** Worth $+4.5$ points, and it is the cheapest component here: keeping a
clean tree so a failed attempt does not poison the next one is version control, not
research.

The failure mode this ordering avoids is the one the build-up table shows directly:
adding a test runner in isolation registers $-0.2$ points, and a team that prunes on
that measurement removes iteration's precondition.

### 7.5 Iteration works here and nowhere else in this book

{{ch:aids-agentic-eda}} found more exploration producing more noise.
{{ch:aids-automl}} found more search producing optimism. {{ch:ag-recovery}} found
retry worthless without a verifier.

Here, iteration takes resolution from $63.8\%$ to $74.8\%$ — and the difference is
entirely that the test suite tells the agent whether the last attempt worked.

**The agent loop is not a general technique that happens to work in software. It is a
technique that requires a verifier, deployed in the one domain that has one.**
{{cite:shinn2023reflexion}}'s reflection has the same precondition, and the same
explanation for why its gains vary so much across settings.

That is the most useful thing to carry from this part into other domains: before
deploying an iterative agent, ask what tells it that an attempt failed. If the answer
is "the model's own judgement", {{ch:aids-autonomous}}'s correlation result applies
and the loop is a resample.

### 7.6 What the reported numbers mean after three corrections

Three multiplicative corrections have now accumulated, and applying them to a
reported figure is worth doing explicitly.

**Contamination.** {{ch:aise-repo}}'s $1.43\times$ localisation ratio for transfer to
an unseen repository ({{cite:liang2025swebenchillusion}}).

**Verification.** This chapter's inflation, about $6.2$ points on the reported rate
({{cite:wang2025solvedcorrectly}}).

**Coverage.** Whether the graded repositories' suites resemble yours in coverage,
which decides how much of the second correction applies to you.

A reported $65\%$ becomes roughly $59\%$ after the verification correction and
roughly $41\%$ after the contamination one — before any adjustment for your
repository being less well tested than a popular open-source Python project.

That is not a debunking. $41\%$ of real issues resolved autonomously would have been
implausible at {{cite:jimenez2023swebench}}'s $1.96\%$, and the trajectory is the
genuinely remarkable thing. It is a statement about what to plan against.

### 7.7 What the benchmark cannot see at all

The three corrections in {{sec:7-internal-mechanics}} adjust a number that is being
measured. There is a further category the benchmark does not measure in either
direction, and it decides whether these systems are usable more than the resolution
rate does.

**Whether the change is one a maintainer would accept.** A patch can be correct,
pass differential testing, and still be wrong for the project: it duplicates a
utility that exists three modules away, it uses a pattern the codebase abandoned two
years ago, it adds a dependency, it solves the symptom where the maintainers wanted
the cause addressed. None of that is a correctness property and all of it decides
whether the patch is merged.

**What the change costs to review.** A minimal patch and a sprawling one can both be
correct. The first takes two minutes to review and the second takes forty, and at
agent volumes review time is the binding resource — {{ch:ag-termination}}'s
habituation applies to a pull-request queue exactly as it applied everywhere else.
A benchmark scores both as a resolution.

**Whether the issue should have been fixed.** Some issues are reports of intended
behaviour. Some are duplicates. Some describe a real problem whose right resolution
is a documentation change or a deprecation. An agent that patches all of them scores
well and is doing harm on a subset, and the judgement involved is
{{ch:aids-oversight}}'s ungradeable framing question wearing different clothes.

These share a property that makes them systematically invisible: **they are
properties of the change relative to a project's intentions, and a benchmark has no
access to intentions.** SWE-bench can compare against what the maintainer did; it
cannot score the difference between a patch that fits the codebase and one that
merely works.

Which suggests the metric a deploying team should actually track, and it is
available from ordinary process data rather than from any benchmark: **the share of
agent-authored pull requests merged without substantive revision.** That number
prices correctness, fit, reviewability and issue triage together, it is what the
work is for, and it is the one measurement in this chapter that nobody has to build
an instrument for.

## 8. Implementation

Two listings. The first measures the gap between passing and correct. The second
ablates the scaffold.

```python {tier=A name=tests-are-a-partial-specification}
"""Passing the tests is not the same as fixing the bug.

Software engineering has the strongest verifier in this book: tests execute, and
cite:jimenez2023swebench grades a patch by running the repository's own suite. That
is a real verifier, which is why coding agents are further along than agents in any
other domain.

cite:wang2025solvedcorrectly measured where it stops. Applying differential testing
to patches from three state-of-the-art tools on SWE-bench Verified, they found 7.8%
of patches counted correct while FAILING the developer-written test suite, 29.6% of
plausible patches behaving differently from the ground-truth human patch, 28.6% of
those divergent patches confirmed incorrect by manual inspection, and reported
resolution rates inflated by about 6.2 percentage points.

The mechanism is that a test suite is a PARTIAL SPECIFICATION. Passing it is
evidence proportional to its coverage, and coverage is never complete
(eq:tests-are-a-partial-specification).
"""
import numpy as np

rng = np.random.default_rng(5077)

M = 80000
P_PATCH_PLAUSIBLE = 0.42    # the agent produces something that compiles and runs
COVERAGE = 0.71             # share of the intended behaviour the suite pins down
P_RIGHT_GIVEN_PLAUSIBLE = 0.63


def run(m=M, coverage=COVERAGE, p_plaus=P_PATCH_PLAUSIBLE,
        p_right=P_RIGHT_GIVEN_PLAUSIBLE, differential=0.0, extra_tests=0.0):
    """Returns (passes the suite, actually correct, precision among passing).

    `differential` is the share of divergent-but-passing patches caught by
    comparing behaviour against the reference patch. `extra_tests` raises
    coverage toward 1.
    """
    cov = coverage + (1.0 - coverage) * extra_tests
    plausible = rng.random(m) < p_plaus
    correct = plausible & (rng.random(m) < p_right)
    wrong = plausible & ~correct
    # A wrong patch fails the suite only where the suite covers the divergence.
    caught_by_tests = wrong & (rng.random(m) < cov)
    passes = plausible & ~caught_by_tests
    # Differential testing against the reference catches divergences the suite
    # does not cover.
    survivors = passes & ~correct
    caught_diff = survivors & (rng.random(m) < differential)
    passes = passes & ~caught_diff

    p = float(passes.mean())
    c = float((passes & correct).mean())
    return p, c, (c / p if p else 0.0)


print(f"{M:,} issues. The agent produces a plausible patch {P_PATCH_PLAUSIBLE:.0%}")
print(f"of the time; {P_RIGHT_GIVEN_PLAUSIBLE:.0%} of plausible patches are right.")
print(f"The repository's test suite pins down {COVERAGE:.0%} of the intended")
print("behaviour.")
print()
print(f"{'':>34}{'share of issues':>17}")
print("-" * 53)
p, c, prec = run()
print(f"{'reported resolved (tests pass)':>34}{p:>17.1%}")
print(f"{'actually correct':>34}{c:>17.1%}")
print(f"{'inflation':>34}{p - c:>17.1%}")
print()
print(f"   Of patches that pass, {1 - prec:.1%} are wrong.")

print()
print()
print("Test coverage is what decides it, because the suite is the specification")
print("and an uncovered behaviour is unspecified.")
print()
print(f"{'coverage':>10}{'reported':>11}{'correct':>10}{'inflation':>12}"
      f"{'precision':>11}")
print("-" * 54)
cv = {}
for k in (0.40, 0.55, 0.71, 0.88, 0.98):
    r = run(coverage=k)
    cv[k] = r
    print(f"{k:>10.0%}{r[0]:>11.1%}{r[1]:>10.1%}{r[0] - r[1]:>12.1%}"
          f"{r[2]:>11.1%}")

print()
print()
print("Differential testing against the reference patch -- run both, compare")
print("behaviour on generated inputs -- catches divergences the suite misses.")
print()
print(f"{'differential catch':>20}{'reported':>11}{'correct':>10}"
      f"{'inflation':>12}{'precision':>11}")
print("-" * 64)
df = {}
for d in (0.0, 0.3, 0.6, 0.85):
    r = run(differential=d)
    df[d] = r
    print(f"{d:>20.0%}{r[0]:>11.1%}{r[1]:>10.1%}{r[0] - r[1]:>12.1%}"
          f"{r[2]:>11.1%}")

print()
print()
print("Which is worth more: more tests, or differential testing against the")
print("reference? Both attack the same gap from different sides.")
print()
print(f"{'intervention':>34}{'reported':>11}{'correct':>10}{'inflation':>12}")
print("-" * 67)
iv = {}
for label, kw in (("as is", {}),
                  ("coverage 71% -> 88%", {"extra_tests": 0.59}),
                  ("differential testing at 60%", {"differential": 0.60}),
                  ("both", {"extra_tests": 0.59, "differential": 0.60})):
    r = run(**kw)
    iv[label] = r
    print(f"{label:>34}{r[0]:>11.1%}{r[1]:>10.1%}{r[0] - r[1]:>12.1%}")

print()
print()
print("Note what differential testing does to the REPORTED number, which is")
print("why it is not adopted.")
print()
print(f"{'':>34}{'reported':>11}{'correct':>10}")
print("-" * 55)
print(f"{'without differential testing':>34}{iv['as is'][0]:>11.1%}"
      f"{iv['as is'][1]:>10.1%}")
print(f"{'with it':>34}{iv['differential testing at 60%'][0]:>11.1%}"
      f"{iv['differential testing at 60%'][1]:>10.1%}")
print()
print(f"   Reported falls {(iv['as is'][0] - iv['differential testing at 60%'][0]) * 100:.1f} points.")
print(f"   Correct is unchanged at {iv['as is'][1]:.1%} -- it was always that.")

print()
print()
print("And how the gap scales with how much the agent attempts. A tool that")
print("produces more plausible patches produces more passing-but-wrong ones.")
print()
print(f"{'plausible patch rate':>22}{'reported':>11}{'correct':>10}"
      f"{'inflation':>12}{'precision':>11}")
print("-" * 66)
pl = {}
for pp in (0.20, 0.42, 0.65, 0.85):
    r = run(p_plaus=pp)
    pl[pp] = r
    print(f"{pp:>22.0%}{r[0]:>11.1%}{r[1]:>10.1%}{r[0] - r[1]:>12.1%}"
          f"{r[2]:>11.1%}")

print(f"""
The first block is the shape of the problem. The pipeline reports {p:.1%} resolved
and {c:.1%} are correct, so **{1 - prec:.1%} of the patches that pass the tests are
wrong**.

cite:wang2025solvedcorrectly measured the real version of this on SWE-bench
Verified: 29.6% of plausible patches behaved differently from the ground-truth human
patch, 28.6% of those were confirmed incorrect on inspection, and reported resolution
rates were inflated by about 6.2 percentage points. This listing produces
{(p - c) * 100:.1f} points of inflation from an independent set of assumptions, which is the
same phenomenon at the same order.

The coverage table says why it happens, and it is not a defect in the grading. A
test suite is a **partial specification**: it pins down the behaviour someone thought
to write a test for, and a patch that satisfies it may do anything at all on the
rest. At {0.40:.0%} coverage the inflation is {cv[0.40][0] - cv[0.40][1]:.1%}; at
{0.98:.0%} it is {cv[0.98][0] - cv[0.98][1]:.1%}
(eq:tests-are-a-partial-specification).

**Passing a test suite is evidence proportional to its coverage**, and every
statement about coding agents' capability inherits whatever coverage the graded
repositories happened to have.

The differential-testing table is the fix, and it works: comparing the agent's patch
against the reference patch on generated inputs -- which is what
cite:wang2025solvedcorrectly's PatchDiff does -- takes inflation from
{df[0.0][0] - df[0.0][1]:.1%} to {df[0.85][0] - df[0.85][1]:.1%} and precision from
{df[0.0][2]:.1%} to {df[0.85][2]:.1%}.

Then the table that explains why it is not standard.

Adding differential testing takes the REPORTED number from {iv['as is'][0]:.1%} to
{iv['differential testing at 60%'][0]:.1%} -- a fall of
{(iv['as is'][0] - iv['differential testing at 60%'][0]) * 100:.1f} points -- while
the number of actually-correct patches does not move. It was always
{iv['as is'][1]:.1%}.

**A better verifier makes your headline worse and your knowledge better**, which is
exactly ch:aids-automl's leakage guard in a new setting. Any team or benchmark
reporting resolution rate is measured against the honest measurement, and the
incentive points away from it.

That is the second independent arrival of this structure in this book, and it is
worth naming as a general hazard: **whenever the metric is produced by the thing
being measured, improving the measurement is penalised.**

The last table is the one to carry forward. As the plausible-patch rate rises from
{0.20:.0%} to {0.85:.0%} -- which is what a better agent looks like -- inflation
rises from {pl[0.20][0] - pl[0.20][1]:.1%} to {pl[0.85][0] - pl[0.85][1]:.1%}
while precision stays flat at about {pl[0.85][2]:.0%}.

**A better agent produces more passing-but-wrong patches in absolute terms.** The
proportion holds and the volume grows, so the amount of plausible-wrong code entering
repositories scales with capability rather than shrinking with it -- unless coverage
or differential testing scales too.""")
```

The second listing asks what the loop around the model is worth.

```python {tier=A name=scaffold-components-interact}
"""The scaffold, which cite:chan2024mlebench found mattered as much as the model.

A SWE agent is a loop around a model, and the loop has parts:

  reproduce   build a failing test before proposing anything
  localise    ch:aise-repo's problem: find the files
  edit        apply a change
  run tests   the verifier, which this domain actually has
  iterate     use the test output to try again
  revert      undo an edit that made things worse

This listing ablates them. The structure is ch:as-single-agent's: components that
each remove a blocker on the others compose superadditively, so measuring them one
at a time understates every one (eq:scaffold-components-interact).

It also measures the thing the loop is for. ch:ag-recovery found retry worthless
without a verifier and valuable with one; a test suite is the verifier, which is why
iteration works here and not in ch:aids-agentic-eda.
"""
import numpy as np

rng = np.random.default_rng(5113)

M = 40000
MAX_ITERS = 6

# Each component's effect, and what it is contingent on.
BASE_LOCALISE = 0.53
BASE_EDIT = 0.80


def run(components, m=M, max_iters=MAX_ITERS, model_skill=1.0):
    """`components` is a set drawn from
    {reproduce, localise, tests, iterate, revert}. Returns
    (resolved, mean iterations used)."""
    have = set(components)

    # Reproduction localises by execution (ch:aise-repo) and supplies a signal.
    loc = BASE_LOCALISE
    if "localise" in have:
        loc = 0.79
    if "reproduce" in have:
        loc = max(loc, 0.91)
    loc = min(loc * model_skill, 0.99)

    located = rng.random(m) < loc
    resolved = np.zeros(m, dtype=bool)
    iters = np.zeros(m, dtype=np.int64)
    damaged = np.zeros(m, dtype=bool)

    n_iter = max_iters if "iterate" in have else 1
    for t in range(n_iter):
        live = located & ~resolved & ~damaged
        idx = np.flatnonzero(live)
        if not len(idx):
            break
        iters[idx] += 1
        # An edit attempt succeeds at the model's rate.
        good = rng.random(len(idx)) < min(BASE_EDIT * model_skill, 0.99)
        if "tests" in have:
            # The suite tells the agent whether the edit worked, which is what
            # makes the next iteration informed rather than a resample.
            resolved[idx[good]] = True
            bad = idx[~good]
            if "revert" not in have:
                # Without revert, a bad edit sometimes leaves the tree worse.
                damaged[bad[rng.random(len(bad)) < 0.22]] = True
        else:
            # No verifier: the agent cannot tell, so it stops after one attempt
            # and ships whatever it produced.
            resolved[idx[good]] = True
            break
    return float(resolved.mean()), float(iters.mean())


ALL = {"reproduce", "localise", "tests", "iterate", "revert"}

print(f"{M:,} issues. Ablating a SWE agent's scaffold, one component at a time.")
print()
print(f"{'scaffold':>34}{'resolved':>11}{'iterations':>13}")
print("-" * 58)
full = run(ALL)
none = run(set())
print(f"{'nothing (single-shot edit)':>34}{none[0]:>11.1%}{none[1]:>13.2f}")
print(f"{'everything':>34}{full[0]:>11.1%}{full[1]:>13.2f}")

print()
print()
print("Each component ADDED to nothing, and REMOVED from everything -- which")
print("ch:as-single-agent showed give very different numbers.")
print()
print(f"{'component':>14}{'added to nothing':>19}{'removed from all':>19}"
      f"{'ratio':>9}")
print("-" * 61)
ab = {}
for c in sorted(ALL):
    added = run({c})[0] - none[0]
    removed = full[0] - run(ALL - {c})[0]
    ab[c] = (added, removed)
    cell = "--" if added <= 0.005 else f"{removed / added:.1f}"
    print(f"{c:>14}{added:>+19.1%}{removed:>+19.1%}{cell:>9}")

print()
print()
print("Building the scaffold up, in the order a team usually builds it.")
print()
print(f"{'scaffold':>44}{'resolved':>11}{'gain':>9}")
print("-" * 64)
ORDER = ["localise", "tests", "iterate", "revert", "reproduce"]
bu = {}
have = set()
prev = none[0]
for c in ORDER:
    have.add(c)
    v = run(set(have))[0]
    bu[c] = (v, v - prev)
    print(f"{('+ ' + c + '  (' + ', '.join(sorted(have)) + ')'):>44}"
          f"{v:>11.1%}{v - prev:>+9.1%}")
    prev = v

print()
print()
print("Iteration is worth nothing without the verifier, which is")
print("ch:ag-recovery's result in the one domain that has a real one.")
print()
print(f"{'configuration':>34}{'resolved':>11}{'iterations':>13}")
print("-" * 58)
it = {}
for label, comp in (("iterate, no tests", {"localise", "iterate"}),
                    ("tests, no iterate", {"localise", "tests"}),
                    ("both", {"localise", "tests", "iterate"})):
    r = run(comp)
    it[label] = r
    print(f"{label:>34}{r[0]:>11.1%}{r[1]:>13.2f}")

print()
print()
print("And scaffold against model. cite:chan2024mlebench found scaffolding")
print("mattering as much as the model; this is that comparison.")
print()
print(f"{'model skill':>13}{'no scaffold':>14}{'full scaffold':>16}{'gap':>9}")
print("-" * 52)
ms = {}
for k in (0.85, 1.00, 1.10, 1.20):
    a = run(set(), model_skill=k)[0]
    b = run(ALL, model_skill=k)[0]
    ms[k] = (a, b)
    print(f"{k:>13.2f}{a:>14.1%}{b:>16.1%}{b - a:>+9.1%}")
print()
print(f"   A {0.20:.0%} better model with no scaffold: {ms[1.20][0]:.1%}")
print(f"   The baseline model with a full scaffold:  {ms[1.00][1]:.1%}")

print(f"""
The ablation table is ch:as-single-agent's methodology result at its most extreme.

Adding TESTS to a bare agent is worth {ab['tests'][0]:+.1%} -- nothing, and
fractionally negative. Removing tests from a full scaffold costs
{ab['tests'][1]:+.1%}. Adding ITERATION alone is worth {ab['iterate'][0]:+.1%};
removing it costs {ab['iterate'][1]:+.1%}.

**Two of the five components are worth approximately zero on their own and eighteen
points each in place** (eq:scaffold-components-interact). The reason is
straightforward once stated: running tests is only useful if you can act on the
result, and iterating is only useful if something tells you whether the last attempt
worked. Each is the other's precondition.

The build-up table shows what that does to a team building this incrementally. The
second row -- adding a test runner -- registers {bu['tests'][1]:+.1%}. A team
measuring as it goes would conclude the test runner does not help and remove it,
which removes the precondition for the iteration that would have been worth
{bu['iterate'][1]:+.1%}.

**Measure by ablation from the full system, not by addition to a bare one**, or you
will delete the components that were about to matter.

The iteration table isolates the dependency. Iterating without tests gives
{it['iterate, no tests'][0]:.1%}; tests without iterating gives
{it['tests, no iterate'][0]:.1%}; both gives {it['both'][0]:.1%}.

That is ch:ag-recovery's finding -- retry needs something to retry AGAINST -- in the
one domain that has a real verifier. Software engineering is where the agent loop
works, and it works for exactly this reason: **a test suite converts a retry from a
resample into a correction.** ch:aids-agentic-eda's exploration had no such thing,
which is why more attempts there produced more noise.

The last table is cite:chan2024mlebench's finding reproduced. A model
{0.20:.0%} better with no scaffold reaches {ms[1.20][0]:.1%}. The baseline model with
a full scaffold reaches {ms[1.00][1]:.1%}.

**The scaffold is worth more than a large model improvement**, and it is available
now, to a team, without waiting for anyone. That is the most actionable finding in
this chapter and it is consistent across the model-skill column: the gap between
scaffolded and unscaffolded stays around {ms[1.00][1] - ms[1.00][0]:.0%} points at
every skill level tested.

Note the high-skill rows are clipped by the model's ceiling and should be read as
"at least", not as a plateau.

The practical ordering, from the build-up table: **reproduce, localise, run tests,
iterate on the results, and revert what makes things worse.** Reproduction is worth
{bu['reproduce'][1]:+.1%} even added last, because it is doing two jobs --
localisation and verification -- which is why ch:aise-repo recommended it first.""")
```

## 9. Practical Example

The first listing runs issues where the agent produces a plausible patch $42\%$ of
the time, $63\%$ of plausible patches are right, and the suite covers $71\%$ of the
intended behaviour:

```
                                   share of issues
-----------------------------------------------------
    reported resolved (tests pass)            31.0%
                 actually correct            26.7%
                        inflation             4.3%
```

**$14\%$ of the patches that pass the tests are wrong**, and the inflation of $4.3$
points reproduces {{cite:wang2025solvedcorrectly}}'s $6.2$ at the same order from
independent assumptions.

Coverage decides it:

```
  coverage   reported   correct   inflation  precision
------------------------------------------------------
       40%      36.0%     26.6%        9.3%      74.1%
       71%      31.0%     26.5%        4.5%      85.5%
       98%      26.7%     26.4%        0.3%      98.8%
```

**Passing a suite is evidence proportional to its coverage**
({{eq:tests-are-a-partial-specification}}).

Differential testing:

```
  differential catch   reported   correct   inflation  precision
----------------------------------------------------------------
                  0%      31.0%     26.7%        4.3%      86.0%
                 60%      28.1%     26.3%        1.8%      93.7%
                 85%      27.1%     26.5%        0.7%      97.5%
```

And why it is not adopted:

```
                                     reported   correct
-------------------------------------------------------
      without differential testing      31.0%     26.5%
                           with it      28.2%     26.3%
```

**The reported number falls $2.8$ points and the correct number does not move**
({{eq:honest-verifiers-lower-the-metric}}) — it was always $26.5\%$.

And the trajectory:

```
  plausible patch rate   reported   correct   inflation  precision
------------------------------------------------------------------
                   20%      14.8%     12.6%        2.1%      85.6%
                   85%      62.9%     53.7%        9.2%      85.3%
```

**Precision is flat and absolute inflation grows**
({{eq:inflation-scales-with-capability}}) — better agents produce more
plausible-wrong patches, not fewer.

The second listing ablates the scaffold:

```
                          scaffold   resolved   iterations
----------------------------------------------------------
        nothing (single-shot edit)      42.4%         0.53
                        everything      91.1%         1.13
```

```
     component   added to nothing   removed from all    ratio
-------------------------------------------------------------
       iterate              +0.1%             +18.3%       --
      localise             +20.6%              +0.0%      0.0
     reproduce             +30.5%             +12.4%      0.4
        revert              +0.3%              +5.2%       --
         tests              -0.2%             +18.6%       --
```

**Tests and iteration are worth approximately zero alone and eighteen points each in
place** ({{eq:scaffold-components-interact}}) — each is the other's precondition.

Which is a trap for incremental development:

```
                      + localise      63.1%   +20.7%
                  + tests              62.9%    -0.2%
       + iterate                       74.7%   +11.8%
+ revert                               79.1%    +4.5%
+ reproduce                            90.9%   +11.8%
```

A team measuring as it builds sees the test runner register $-0.2$ points and removes
it, deleting the precondition for the $+11.8$ that was next.

Iteration's dependency, isolated:

```
                     configuration   resolved   iterations
----------------------------------------------------------
                 iterate, no tests      63.6%         0.79
                 tests, no iterate      63.8%         0.79
                              both      74.8%         0.93
```

{{ch:ag-recovery}}'s result in the one domain with a real verifier: **a test suite
converts a retry from a resample into a correction.**

And scaffold against model:

```
  model skill   no scaffold   full scaffold      gap
----------------------------------------------------
         1.00         42.3%           91.0%   +48.7%
         1.20         61.0%           99.0%   +38.0%
```

A model $20\%$ better with no scaffold reaches $61.0\%$; the baseline model with a
full scaffold reaches $91.0\%$
({{eq:scaffold-beats-model-improvement}}) — {{cite:chan2024mlebench}}'s finding, and
the scaffold is available now.

## 10. Production Considerations

Do not treat "the tests passed" as "the issue is fixed". It is evidence proportional
to coverage.

Implement differential testing against the pre-patch version: a patch should change
behaviour on the reported case and nowhere else. This form needs no reference patch
and is available in production.

Report the raw and verified numbers separately, so better verification reads as an
improvement rather than a regression.

Track absolute inflation, not precision. Precision is flat as agents improve and the
defect volume is not.

Build the scaffold in order: reproduce, localise, tests and iteration *together*,
revert.

Ablate from the full system when measuring components. Adding one at a time returns
zero for the mutually contingent ones and they will be deleted.

Before deploying an iterative agent in any domain, ask what tells it an attempt
failed. If the answer is the model's own judgement, the loop is a resample.

And apply the contamination and verification corrections before planning against a
reported resolution rate.

## 11. Common Mistakes

**Reading a resolution rate as a correctness rate.** About $14\%$ of passing patches
are wrong.

**Assuming better agents produce less bad code.** Precision is flat; volume grows.

**Adopting no differential testing because the numbers would fall.** The correct
patches were always the same number.

**Measuring scaffold components by addition.** Two of five return zero.

**Deleting the test runner because it registered nothing.** It is iteration's
precondition.

**Deploying iteration without a verifier.** It is a resample and this book has
measured that three times.

**Accepting patches that change behaviour beyond the issue.** $27.3\%$ of divergent
patches do exactly that.

## 12. Failure Modes

*Passing-but-wrong patch.* The characteristic failure — plausible, tested, merged,
different from what was wanted.

*Silent scope expansion.* A patch that fixes the issue and changes something else
nobody reviewed.

*Coincidental pass.* A test that did not exercise the bug, satisfied by a patch that
did not fix it — {{cite:wang2025solvedcorrectly}}'s $7.8\%$ counted correct while
failing the developer suite.

*Scaffold pruning.* A mutually contingent component removed on a one-at-a-time
measurement.

*Uninformed iteration.* A loop with no verifier, burning budget on resamples.

*Corrected-number shock.* A system performing at the reported rate on its benchmark
and well below it on an unfamiliar, less-tested repository.

## 13. Alternatives

**Property-based testing of the patched module.** Generates the inputs differential
testing needs, and catches over-broad changes without a reference.

**Requiring a failing test with every patch.** Makes reproduction mandatory and
supplies the coverage the suite lacked, at the cost of the agent needing to write
tests — {{ch:aise-testing}}'s subject and its correlation problem.

**Human review of the diff.** Effective and subject to {{ch:ag-termination}}'s
habituation at agent volumes.

**Restricting agents to well-tested modules.** Directly raises $\kappa$ for the code
they touch, and is the cheapest structural intervention available.

**Shadow deployment.** Run the patched and unpatched versions side by side on real
traffic — the strongest verification and the slowest.

## 14. Evaluation

Measure your inflation directly: sample patches that passed, review them properly,
and count how many were actually right. It is the number this chapter is about and
nobody has it for their own system.

Measure your repositories' coverage. It is the parameter that determines how much of
the published inflation applies to you.

Report resolution rate alongside a differentially-verified rate.

Measure behaviour change outside the issue's scope, which is the strongest available
signal and needs no reference.

Ablate the scaffold from the full system, never by addition.

And measure absolute defect volume as your agent improves, since precision will not
move.

## 15. Advanced Concepts

**Reference-free differential testing.** Comparing against the pre-patch version on
out-of-scope inputs, which is available in production and detects the most damaging
divergence class. {{maturity:EMERGING}}.

**Coverage-aware confidence.** Reporting a patch's confidence as a function of the
coverage of the code it touched, so a patch in untested code is flagged as weakly
verified. Straightforward and unimplemented.

**Automatic scope checking.** Determining whether a patch's behavioural change
exceeds the issue's stated scope, mechanically.

**Contamination-free evaluation.** Repositories created after training cutoffs,
which is the only way to measure {{eq:inflation-scales-with-capability}}'s trajectory
honestly. {{maturity:RESEARCH FRONTIER}}.

## 16. Connection to Previous Chapters

{{ch:as-specialized}}'s verifier ceiling is this chapter's subject at its most
favourable: the strongest verifier in the book, and a measurable gap where it stops.

{{ch:aids-automl}}'s leakage guard reappears exactly — an honest verifier that lowers
the reported number — and the second instance makes it a general hazard rather than a
quirk.

{{ch:as-single-agent}}'s ablation methodology is not merely preferable here but
necessary: two components measure zero by addition.

{{ch:ag-recovery}}'s retry precondition explains why the agent loop works in this
domain and produced noise in {{ch:aids-agentic-eda}}.

{{ch:aise-repo}}'s reproduction result is confirmed from a second direction: it is
worth $+11.8$ points even added last.

Ahead: {{ch:aise-testing}} takes up generating the tests themselves, where the
verifier and the thing verified come from the same model.

## 17. Exercises

1. Derive precision's independence from $\pi$ in
   {{eq:tests-are-a-partial-specification}} and check it against the last table.

2. Implement reference-free differential testing on a real patch: compare pre- and
   post-patch behaviour on inputs unrelated to the issue.

3. Measure coverage for the modules your agent most often touches, and compute your
   own inflation estimate.

4. Ablate a real SWE agent's scaffold from the full configuration and compare with
   the one-at-a-time measurements.

5. Model a scaffold component that is contingent on two others and find what
   measurement recovers its value.

6. Apply all three corrections to a currently reported resolution rate and state what
   you would plan against.

## 18. Interview Questions

1. An agent resolves $65\%$ of SWE-bench Verified. What has been established?

2. Why does adding differential testing look like a regression?

3. Your agent got better and your defect rate did not fall. Why?

4. You add a test runner to your agent and resolution does not improve. Do you keep
   it?

5. Why does iteration work for coding agents and not for exploratory data analysis?

6. Would you spend on a better model or a better scaffold?

## 19. Research Questions

1. How much divergence does reference-free differential testing catch relative to
   reference-based?

2. Can a patch's out-of-scope behavioural change be bounded automatically?

3. What is the inflation on repositories with coverage typical of private codebases
   rather than popular open-source ones?

4. Does the precision-flat, volume-growing pattern hold empirically as agents
   improve?

5. How many scaffold components are mutually contingent, and does the count grow with
   scaffold complexity?

## 20. Chapter Summary

Coding has the strongest verifier in this book and this chapter measures where it
stops. A test suite is a **partial specification**: passing it is evidence
proportional to coverage, and inflation is $\pi(1-\rho)(1-\kappa)$
({{eq:tests-are-a-partial-specification}}). {{cite:wang2025solvedcorrectly}} found
$29.6\%$ of plausible patches diverging behaviourally from the human patch, $28.6\%$
of those confirmed incorrect, and reported rates inflated by about $6.2$ points;
{{sec:9-practical-example}} reproduces $4.3$ from independent assumptions, with
$14\%$ of passing patches wrong.

Differential testing closes most of it — inflation from $4.3$ to $0.7$ points at high
catch rates — and **lowers the reported number by $2.8$ points while leaving the
correct count exactly where it was** ({{eq:honest-verifiers-lower-the-metric}}). That
is {{ch:aids-automl}}'s leakage guard again, and two independent appearances make it a
rule: **when the metric is produced by the thing being measured, improving the
measurement is penalised.**

And the trajectory matters: **precision is independent of capability and absolute
inflation is not** ({{eq:inflation-scales-with-capability}}). From a $20\%$ to an
$85\%$ plausible-patch rate, precision held near $85\%$ while inflation rose from
$2.1$ to $9.2$ points. Better agents write more subtly-wrong code, not less.

On the scaffold, ablation found **tests and iteration worth approximately zero alone
and eighteen points each in place** ({{eq:scaffold-components-interact}}) — each is
the other's precondition, so one-at-a-time measurement returns zero and a team
measuring as it builds will delete them. Iterating without tests gave $63.6\%$ and
with them $74.8\%$: {{ch:ag-recovery}}'s result in the one domain that has a real
verifier, and the reason the agent loop works here and produced noise in
{{ch:aids-agentic-eda}}.

Finally, a model $20\%$ better with no scaffold reached $61.0\%$ and the baseline
model with a full scaffold reached $91.0\%$
({{eq:scaffold-beats-model-improvement}}). **The scaffold is worth more than a large
model improvement, and it is available now.**

## 21. Further Reading

{{cite:wang2025solvedcorrectly}} is the essential paper here and its PatchDiff
technique is implementable by any team — read it for the divergence taxonomy as much
as for the headline number.

{{cite:jimenez2023swebench}} for the benchmark and its $1.96\%$ starting point, which
is the right anchor for how far this moved.

{{cite:chan2024mlebench}} for the scaffolding result this chapter reproduces, and
{{cite:liang2025swebenchillusion}} for the contamination correction that composes with
this chapter's verification one.

{{cite:shinn2023reflexion}} for iteration with a verifier, and
{{ch:ag-recovery}} for why the precondition is the whole story.
