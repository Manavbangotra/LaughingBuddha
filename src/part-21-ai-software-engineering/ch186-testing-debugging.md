---
id: aise-testing
number: 186
part: XXI
tier: full
status: draft
requires: [tests-are-a-partial-specification, self-judging-measures-correlation,
           redundancy-needs-independence, reproduce-before-retrieve]
provides: [same-model-tests-codify-the-bug, independence-not-coverage,
           write-tests-from-the-spec, refactoring-safety-is-coverage,
           worst-code-is-least-covered, characterise-then-refactor]
citations: [wang2025solvedcorrectly, jimenez2023swebench, lu2024aiscientist,
            chan2024mlebench, huang2024selfcorrect]
---

## 1. Learning Objectives

By the end of this chapter you will be able to explain why a test suite generated
from the code it tests achieves high coverage and low defect detection; identify
independence rather than coverage as the binding term and say what a coverage figure
therefore does not tell you; apply the prompt-level change that recovers most of the
independence at almost no cost; explain why debugging and refactoring have opposite
verifier profiles despite looking identical; and state the one use for which
generated-from-code tests are exactly right.

## 2. Why This Matters

{{ch:aise-swe-agents}} established that everything in this part rests on a test suite
being an *independent* partial specification. A developer's suite encodes what they
intended; a patch satisfying it is evidence because the intention and the
implementation came from separate acts.

Generated tests break that, and this chapter measures how badly.

A model asked to write tests for code it just wrote produces tests of what the code
*does* — that is what it is looking at. {{sec:9-practical-example}} finds such a
suite achieving the **highest coverage in the table at $88\%$ and catching $9.8\%$
of defects**, against a human suite's $75\%$ coverage and $68.3\%$ detection
({{eq:same-model-tests-codify-the-bug}}). If the code has a bug, the tests assert the
buggy behaviour and pass.

This is {{ch:aids-autonomous}}'s self-judging result in the domain with the strongest
verifier, and it is worse here than there: the checker is not merely correlated with
the generator, it is reading the generator's output and describing it.

The binding term is independence, and the reported term is coverage. Raising a
generated suite's coverage from $60\%$ to $97\%$ moves detection by four points
({{eq:independence-not-coverage}}) — so **"our generated tests reach $95\%$
coverage" is not the reassurance it sounds like.**

The intervention is nearly free: asking the same model to write tests **from the
specification rather than from the code** takes detection from $9.6\%$ to $34.6\%$
({{eq:write-tests-from-the-spec}}).

The second half turns to refactoring, and finds the same artefact's flaw becoming
its qualification.

## 3. Prerequisites

{{ch:aise-swe-agents}}'s {{eq:tests-are-a-partial-specification}} — this chapter asks
where the specification comes from.

{{ch:aids-autonomous}}'s {{eq:self-judging-measures-correlation}} and
{{ch:as-failures}}'s {{eq:redundancy-needs-independence}}, of which this is the
sharpest instance in the book.

{{ch:aise-repo}}'s {{eq:reproduce-before-retrieve}}, since a failing test turns out
to be the artefact that separates debugging from refactoring.

## 4. Intuitive Explanation

A model writes a function. You ask it to write tests. It writes tests. They pass, and
coverage is high.

Ask what has been established. The model read the function, worked out what it does,
and wrote assertions matching. If the function is correct, the tests are correct and
useful. If the function has an off-by-one error, **the tests assert the off-by-one
and pass**.

A test suite is only evidence to the extent it encodes what the code *should* do
independently of what it does. Generated-from-code tests have almost none of that
independence, which is why {{sec:9-practical-example}} finds them at $9.8\%$ defect
detection against a human suite's $68.3\%$ — with *higher* coverage.

That inversion is the practical hazard. Coverage is the number that gets reported,
generated tests raise it dramatically, and raising it this way makes it stop
measuring anything. A team that moves from $45\%$ to $90\%$ coverage by generating
tests has roughly doubled a number and barely moved its ability to catch a bug.

There is a cheap fix and it is a prompt-level decision. **Do not show the model the
implementation.** Give it the specification — the issue, the docstring, the interface
— and ask for tests. {{sec:9-practical-example}} finds that recovering detection from
$9.6\%$ to $34.6\%$ at a fourteen-point coverage cost, which is an excellent trade
because the coverage was not buying anything.

A different model does better still, which is {{ch:as-failures}}'s decorrelation
result: independence is the input and its sources compose.

Then the second half, which is about two activities that look identical and are not.

**Debugging** starts from a failing test. Somebody has demonstrated the bug. The
verifier is handed to you: change the code until the test passes, and the rest of the
suite guards against collateral damage. This is the strongest position an agent
occupies anywhere in this book, and {{sec:9-practical-example}} finds its success
barely moving with the surrounding coverage.

**Refactoring** starts from passing tests and must keep them passing *for the right
reason*. Behaviour must be preserved, and the only evidence that it was is a suite
written for other purposes covering whatever it happens to cover. Refactoring safety
is coverage ({{eq:refactoring-safety-is-coverage}}), and it moves a lot.

Which produces the unfortunate part. Code gets refactored because it is bad —
tangled, old, poorly understood. That code is also undertested, because undertested
and tangled are the same history ({{eq:worst-code-is-least-covered}}).
{{sec:9-practical-example}} finds refactoring safe $97.2\%$ of the time in
well-maintained code and $83.8\%$ in code nobody understands, which is precisely
inverted from where the value is.

And here the two halves of the chapter meet. The fix for refactoring is
**characterisation tests**: pin what the code does now, before touching anything.
They make no claim about what it *should* do — they are a snapshot.

Which is exactly what a model generating tests from code produces. The property that
disqualifies generated tests as a specification is the property that qualifies them
as a characterisation. **The failure mode in the first listing is the requirement in
the second.**

## 5. Formal Explanation

**Detection requires reach and independence.** Let a suite reach a fraction $r$ of
the code's behaviours and let $\iota$ be the probability that a reached assertion
encodes *intended* rather than *observed* behaviour. A defect is caught when both
hold:

$$\Pr[\text{catch}] = r \cdot \iota$$ (eq:same-model-tests-codify-the-bug)

For a human writing from a specification, $\iota$ is high because the assertion is
derived from intent. For a model writing from the code, $\iota$ is low because the
assertion is derived from the implementation — including its defects.

**Coverage is $r$ and detection is $r\iota$**, so a coverage figure reports one
factor of a product and omits the one that varies most:

$$\frac{\partial \Pr[\text{catch}]}{\partial r} = \iota, \qquad \frac{\partial \Pr[\text{catch}]}{\partial \iota} = r$$ (eq:independence-not-coverage)

At $\iota = 0.11$ and $r = 0.9$, the marginal return on coverage is $0.11$ per point
and on independence is $0.9$ per point — **eight times larger**, and independence is
the unreported one.

**The specification intervention.** Writing from the specification rather than the
code changes $\iota$ and leaves $r$ roughly intact:

$$\Delta \Pr[\text{catch}] = r(\iota_{\text{spec}} - \iota_{\text{code}}) - \iota_{\text{spec}}(r_{\text{code}} - r_{\text{spec}})$$ (eq:write-tests-from-the-spec)

The first term dominates whenever $\iota_{\text{spec}} \gg \iota_{\text{code}}$,
which the measurements support.

**Debugging versus refactoring.** Debugging has an external verifier $V_{\text{fail}}$
supplied before the task, so success is:

$$P_{\text{debug}} = p_{\text{fix}}\big(1 - c(1-r)\big)$$

with $c$ the collateral-damage rate — coverage enters only through the *secondary*
risk. Refactoring has no such artefact, so the suite is the sole verifier:

$$P_{\text{refactor}} = 1 - b(1 - r)$$ (eq:refactoring-safety-is-coverage)

with $b$ the rate at which a refactor changes behaviour it should not. Coverage
enters the *primary* term, which is why
$\partial P_{\text{refactor}}/\partial r \gg \partial P_{\text{debug}}/\partial r$.

**The correlation.** Empirically, refactoring demand and coverage are inversely
related — the same neglect produces both:

$$\text{Cov}\big(\text{refactoring value}, r\big) < 0$$ (eq:worst-code-is-least-covered)

so the highest-value targets sit at the lowest safety.

**Characterisation.** Writing tests that pin current behaviour raises $r$ toward one
*without requiring $\iota$*, because a characterisation test asserts the observed
value by design:

$$r' = r + (1-r)\eta, \qquad \iota \text{ irrelevant}$$ (eq:characterise-then-refactor)

**A characterisation suite needs exactly the property a generated suite has**, which
is why the two listings resolve each other.

## 6. Mathematical Foundation

Three extractions.

**Coverage reports the smaller factor.** From
{{eq:independence-not-coverage}}, the marginal return on independence exceeds that on
coverage by $r/\iota$, which at realistic generated-test values is roughly eight. So
an instrument that measures $r$ and not $\iota$ is measuring the wrong one, and the
tools that raise $r$ fastest are the ones that lower $\iota$ most.

**Two activities differ by where coverage enters.**
{{eq:refactoring-safety-is-coverage}} places $r$ in the primary term for refactoring
and the secondary one for debugging. That is a structural difference, not a matter of
degree, and it justifies treating them as separate capabilities rather than as one
"code change" agent.

**Characterisation decouples $r$ from $\iota$.** From
{{eq:characterise-then-refactor}}, a characterisation suite raises reach while making
no claim requiring independence. It is the one use in this book where a correlated
verifier is not merely acceptable but correct.

## 7. Internal Mechanics

### 7.1 What a generated test actually asserts

```mermaid {#fig:generated-test caption="A test generated from code. The assertion is derived from the implementation, so it agrees with the implementation's defects by construction."}
flowchart TD
    SPEC["the specification<br/>(what it should do)"] -.->|"not consulted"| T
    IMPL["the implementation<br/>(what it does, bugs included)"] --> READ[model reads it]
    READ --> T["assert f(3) == 7"]
    T --> RUN[test runs]
    RUN --> PASS["passes"]
    IMPL -->|"the bug is in here"| PASS
```

The dotted edge is the whole diagram. The specification was available and was not the
input, so the test cannot disagree with the code about anything.

### 7.2 Writing tests from the specification

The intervention is small and worth stating precisely because it is easy to get
almost right.

**Give the model the interface and the intent, not the body.** A docstring, a type
signature, an issue description, an example from documentation. Withhold the
implementation.

**Ask for the cases before the assertions.** Enumerating what should be tested —
boundaries, empty inputs, error conditions — before writing any assertion keeps the
model in specification space rather than in implementation space.

**Then run them against the implementation and expect some to fail.** A
specification-derived suite that passes completely on first run is a signal that the
model saw the implementation anyway, through context it retained or through the
retrieval that supplied it.

That last check is the practical one, because in an agent loop the implementation is
almost always in context already. **Independence has to be arranged rather than
requested**, which is {{ch:ag-security}}'s containment argument again: do not ask the
model not to look, arrange that it has not.

### 7.3 Mutation testing measures the missing term

{{eq:independence-not-coverage}} says $\iota$ is the unreported factor, and there is
an established technique that measures it: **mutation testing.**

Introduce a small deliberate defect into the code — flip a comparison, change a
constant, remove a branch — and run the suite. A suite that catches the mutant has
demonstrated it asserts intent on that behaviour. A suite that passes has
demonstrated it does not.

The mutation score is therefore a direct estimate of $r\iota$, which is the quantity
that matters, rather than of $r$ alone. It is expensive, it has been available for
decades, and it is almost never run.

This chapter is an argument that it should be, and specifically that **a
generated-test pipeline should report mutation score rather than coverage**, because
generated tests are precisely the case where the two diverge most.

### 7.4 Why debugging is the best position in this book

It is worth being explicit about what a failing test supplies, because the list is
longer than "a way to check the fix".

**Localisation**, per {{ch:aise-repo}} — the trace names the files.

**A verifier** that is independent of the model, because the test was written by
whoever reported the bug.

**A termination condition** — {{ch:ag-termination}}'s hardest problem, solved:
the loop stops when the test passes.

**A regression guard** for the future, contributed by the work.

Four of this book's recurring difficulties resolved by one artefact. That is why
{{ch:aise-swe-agents}} found reproduction worth $+11.8$ points even when added last,
and why a bug report with a reproduction is worth so much more than one without.

### 7.5 Refactoring under an agent, safely

{{eq:worst-code-is-least-covered}} says the demand and the safety are inversely
correlated, so the procedure matters more here than anywhere else in the part.

**Characterise first.** Generate tests from the current code, deliberately. Do not
review them for correctness — that is not what they are for. Review them for
*coverage of the region being changed*.

**Refactor in steps small enough to be individually verified.** This is
{{ch:ag-planning}}'s checkpoint result: the governing exponent is the size of the
unverified step, not the size of the change.

**Re-run the characterisation after each step**, not at the end.

**And treat any characterisation test that fails as a stop**, not as a test to
update. Updating a characterisation test to match new behaviour is the exact move
that converts a safety net into a rubber stamp, and an agent will do it readily
because the test looks wrong — from inside the refactor, a test asserting the old
behaviour is indistinguishable from a test that was always incorrect.

There is a legitimate case where the characterisation must change: a refactor that
deliberately alters behaviour is no longer a refactor, and the test should be updated
as part of a change that says so. The discipline is that this is a decision requiring
a human, not a step the agent takes when a test is inconvenient. In practice that
means the characterisation suite lives outside the agent's writable scope, and a
required change to it is a signal to stop and ask rather than a task to complete.

That last point deserves enforcement rather than instruction: **a refactoring agent
should not have permission to modify the characterisation suite.**

### 7.6 The one place a correlated verifier is correct

This book has argued against correlated verifiers in five settings and it is worth
marking the exception clearly, because the distinction is subtle and the
recommendation reverses.

A verifier's job is normally to check work against an *external* standard, and
correlation with the work destroys that. But a characterisation suite's job is to
detect *change*, not to certify correctness. Its reference point is the code as it
stands, so being derived from that code is exactly right — a characterisation test
that disagreed with current behaviour would be a broken characterisation test.

The distinguishing question: **does this artefact assert what the code should do, or
record what it currently does?** The first requires independence and the second
requires fidelity, and they are satisfied by opposite construction methods.

Confusing them is the failure this chapter is about, in both directions: using
generated tests to verify a fix (no independence where it is needed), and writing
characterisation tests from a specification (no fidelity where it is needed).

### 7.7 Where generated tests are genuinely valuable

The measurements in this chapter are unflattering and it would be a
misreading to conclude that generating tests is a bad idea. Four uses survive
{{eq:same-model-tests-codify-the-bug}} intact, and between them they cover a large
fraction of why teams want the capability at all.

**Characterisation before change**, which {{sec:9-practical-example}}'s second
listing measures directly. Fidelity is the requirement and independence is not.

**Regression capture after a fix.** Once a bug is understood and fixed, a test
pinning the corrected behaviour is a characterisation of something now known to be
right. The independence was supplied by the debugging; the test is recording its
outcome.

**Coverage of mechanical branches.** Error paths, argument validation, exhaustive
enumeration handling — cases where "what the code should do" is not in dispute and
the work is tedium. A model that writes forty boundary cases is doing something a
person would do identically and skip.

**Finding crashes rather than wrong answers.** A generated suite that exercises
paths nobody ran will find exceptions, and an exception is self-evidently wrong
regardless of whether the assertion encoded intent. This is fuzzing with a nicer
interface, and it works because the oracle is "did it blow up" rather than "is this
the right value".

That last one generalises usefully. **Generated tests are worth what their oracle is
worth independently of the model.** A crash oracle, a type-invariant oracle, a
"the two implementations agree" oracle, a "this property holds" oracle — each is
external, so a test carrying one has independence the assertion does not need to
supply.

Which reframes the chapter's recommendation from "generate fewer tests" to
**"generate tests around oracles that are not the model's opinion"** — and puts
property-based and metamorphic testing at the top of the list rather than as
alternatives at the bottom.

## 8. Implementation

Two listings. The first measures what generated tests detect. The second measures
what refactoring safety depends on.

```python {tier=A name=same-model-tests-codify-the-bug}
"""Tests written by the model that wrote the code.

ch:aise-swe-agents found a test suite to be a partial specification, and everything
in this part rests on it being an INDEPENDENT one. A suite written by the developer
encodes what they intended; a patch that satisfies it is evidence because the
intention and the implementation came from different acts.

Generated tests break that. A model asked to write tests for code it just wrote
produces tests of what the code DOES, because that is what it is looking at. The
tests pass. They would also have passed on the bug (eq:same-model-tests-codify-the-bug).

This is ch:aids-autonomous's self-judging result in the domain with the strongest
verifier, and the mechanism is identical: a generator and a checker from one model
share the blind spot that produced the defect.
"""
import numpy as np

rng = np.random.default_rng(5147)

M = 60000
P_BUG = 0.22                # the code under test contains a defect
SPEC_COVERAGE = 0.75        # share of intended behaviour any good suite reaches


def run(author, m=M, p_bug=P_BUG, coverage=SPEC_COVERAGE, shown_code=True):
    """`author` is who wrote the tests:
       'human'      wrote them from the specification
       'other'      a different model, given the specification
       'same'       the model that wrote the code, given the code
       'same-spec'  the same model, given the SPECIFICATION and not the code
    Returns (catches the bug, tests pass on buggy code, coverage achieved).
    """
    buggy = rng.random(m) < p_bug

    if author == "human":
        # Written from intent, so a bug is caught wherever the suite reaches.
        reach, independence = coverage, 0.92
    elif author == "other":
        reach, independence = coverage * 0.97, 0.74
    elif author == "same-spec":
        # Same model, but writing from the specification rather than the code.
        reach, independence = coverage * 0.99, 0.46
    elif author == "same":
        # Generated from the code: high coverage of what the code does, and
        # almost no independence from it.
        reach, independence = min(coverage * 1.18, 0.97), 0.11
    else:
        raise ValueError(author)

    # A bug is caught only if the suite reaches that behaviour AND the test
    # encodes the INTENDED result rather than the observed one.
    caught = buggy & (rng.random(m) < reach) & (rng.random(m) < independence)
    passes_buggy = buggy & ~caught
    return (float(caught.mean() / max(p_bug, 1e-9)),
            float(passes_buggy.mean() / max(p_bug, 1e-9)),
            reach)


AUTHORS = [("human, from the spec", "human"),
           ("a different model, from the spec", "other"),
           ("the same model, from the spec", "same-spec"),
           ("the same model, from the code", "same")]

print(f"{M:,} modules, {P_BUG:.0%} of them containing a defect. A suite catches a")
print("defect only if it reaches that behaviour AND asserts what was INTENDED")
print("rather than what the code happens to do.")
print()
print(f"{'tests written by':>34}{'coverage':>11}{'catches the bug':>17}"
      f"{'passes on it':>14}")
print("-" * 76)
tab = {}
for label, a in AUTHORS:
    r = run(a)
    tab[label] = r
    print(f"{label:>34}{r[2]:>11.0%}{r[0]:>17.1%}{r[1]:>14.1%}")

print()
print()
print("Coverage and catch rate move in opposite directions here, which is the")
print("whole problem: generated tests achieve the HIGHEST coverage and the")
print("LOWEST defect detection.")
print()
print(f"{'tests written by':>34}{'coverage':>11}{'catch rate':>13}"
      f"{'coverage per catch':>20}")
print("-" * 78)
for label, a in AUTHORS:
    r = tab[label]
    print(f"{label:>34}{r[2]:>11.0%}{r[0]:>13.1%}"
          f"{r[2] / max(r[0], 1e-9):>20.2f}")

print()
print()
print("Writing tests from the SPECIFICATION rather than from the code recovers")
print("most of the independence, at no cost in coverage.")
print()
print(f"{'':>34}{'from the code':>15}{'from the spec':>15}{'gain':>9}")
print("-" * 73)
a_code = run("same")
a_spec = run("same-spec")
print(f"{'catch rate, same model':>34}{a_code[0]:>15.1%}{a_spec[0]:>15.1%}"
      f"{a_spec[0] - a_code[0]:>+9.1%}")
print(f"{'coverage':>34}{a_code[2]:>15.0%}{a_spec[2]:>15.0%}"
      f"{a_spec[2] - a_code[2]:>+9.0%}")

print()
print()
print("What a generated suite is worth as the verifier in ch:aise-swe-agents'")
print("loop -- the number that decides whether the agent's iteration means")
print("anything.")
print()
print(f"{'suite author':>34}{'usable as a verifier':>22}")
print("-" * 58)
for label, a in AUTHORS:
    r = tab[label]
    print(f"{label:>34}{r[0]:>22.1%}")

print()
print()
print("And what raising the generated suite's coverage buys, which is the")
print("counterintuitive part: 37 points of coverage move the catch rate by 4.")
print()
print(f"{'generated coverage':>20}{'catch rate':>13}{'passes on bug':>16}")
print("-" * 49)
gc = {}
for c in (0.60, 0.75, 0.90, 0.97):
    buggy = rng.random(M) < P_BUG
    caught = buggy & (rng.random(M) < c) & (rng.random(M) < 0.11)
    gc[c] = (float(caught.mean() / P_BUG),
             float((buggy & ~caught).mean() / P_BUG))
    print(f"{c:>20.0%}{gc[c][0]:>13.1%}{gc[c][1]:>16.1%}")

print()
print()
print("The independence term is what matters, and it is the one nobody reports.")
print()
print(f"{'independence':>14}{'catch rate':>13}{'equivalent human coverage':>27}")
print("-" * 54)
ind = {}
for i in (0.11, 0.30, 0.55, 0.92):
    buggy = rng.random(M) < P_BUG
    caught = buggy & (rng.random(M) < 0.90) & (rng.random(M) < i)
    cr = float(caught.mean() / P_BUG)
    ind[i] = cr
    print(f"{i:>14.0%}{cr:>13.1%}{cr / 0.92:>27.0%}")

print(f"""
The first table has the coverage and catch-rate columns moving in opposite
directions, which is the finding.

Tests generated from the code achieve the HIGHEST coverage in the table --
{tab['the same model, from the code'][2]:.0%} -- and catch
{tab['the same model, from the code'][0]:.1%} of defects. Human tests written from
the specification cover {tab['human, from the spec'][2]:.0%} and catch
{tab['human, from the spec'][0]:.1%}.

**A generated suite tests what the code does**, which is why its coverage is
excellent and its detection is not: if the code has a bug, the tests assert the
buggy behaviour and pass ({{eq:same-model-tests-codify-the-bug}}).

That is ch:aids-autonomous's self-judging result in the domain with the strongest
verifier, and the mechanism is identical -- a generator and a checker from one model
share the blind spot that produced the defect. Here it is worse than there, because
the checker is not merely correlated with the generator: it is looking at the
generator's output and describing it.

The coverage-per-catch column makes the trap concrete. Generated tests need
{tab['the same model, from the code'][2] / max(tab['the same model, from the code'][0], 1e-9):.1f}
points of coverage per point of catch rate; human tests need
{tab['human, from the spec'][2] / max(tab['human, from the spec'][0], 1e-9):.1f}.
**Coverage is the metric these tools raise and the metric that stops meaning
anything when they raise it.**

The third table is the intervention, and it is nearly free. Asking the SAME model to
write tests from the SPECIFICATION rather than from the code takes the catch rate
from {a_code[0]:.1%} to {a_spec[0]:.1%} -- {a_spec[0] - a_code[0]:+.1%} -- at a cost
of {a_spec[2] - a_code[2]:+.0%} in coverage.

**Do not show the model the implementation when asking it for tests.** That single
change recovers most of the independence, and it is a prompt-level decision rather
than an architecture.

A different model from the specification does better still at
{tab['a different model, from the spec'][0]:.1%}, which is
ch:as-failures' decorrelation result: independence is the input, and different
sources of it compose.

The fifth table is why "our generated tests reach 95% coverage" is not the
reassurance it sounds like. Moving generated coverage from {0.60:.0%} to
{0.97:.0%} moves the catch rate from {gc[0.60][0]:.1%} to {gc[0.97][0]:.1%} --
{37} points of coverage buying {(gc[0.97][0] - gc[0.60][0]) * 100:.0f} points of
detection.

**Independence is the binding term and coverage is the reported one.** The last
table prices it directly: at {0.11:.0%} independence a suite is worth what a
{ind[0.11] / 0.92:.0%}-coverage human suite would be; at {0.55:.0%} independence,
{ind[0.55] / 0.92:.0%}.

Which gives the rule for anyone using generated tests as the verifier in
ch:aise-swe-agents' loop: **a suite generated from the code is not a verifier**, and
an agent iterating against one is doing ch:ag-recovery's resample with extra steps.""")
```

The second listing separates debugging from refactoring.

```python {tier=A name=refactoring-safety-is-coverage}
"""Refactoring, where the verifier has to already exist.

Debugging and refactoring look like the same activity to an agent -- read code,
change code, run tests -- and they have opposite verifier profiles.

Debugging starts from a FAILING test. The verifier is handed to you: the bug is
demonstrated, and the fix is confirmed when the test goes green. That is the
strongest position an agent occupies anywhere in this book.

Refactoring starts from passing tests and must PRESERVE behaviour. The verifier is
the existing suite, so the safety of a refactor is bounded by the coverage of the
code being refactored (eq:refactoring-safety-is-coverage).

Which produces an unfortunate correlation: the code most in need of refactoring --
old, tangled, poorly understood -- is the code least likely to be well tested
(eq:worst-code-is-least-covered).
"""
import numpy as np

rng = np.random.default_rng(5179)

M = 60000
P_REFACTOR_BREAKS = 0.19    # a refactor changes behaviour it should not
P_FIX_WORKS = 0.80


def debug(coverage, m=M, p_fix=P_FIX_WORKS):
    """A failing test exists. Returns (fixed and verified, shipped broken)."""
    fixed = rng.random(m) < p_fix
    # The failing test confirms the fix directly; the rest of the suite catches
    # collateral damage to the extent it covers it.
    collateral = ~fixed | (rng.random(m) < 0.12)
    caught = collateral & (rng.random(m) < coverage)
    ok = fixed & ~(collateral & ~caught)
    return float(ok.mean()), float((collateral & ~caught).mean())


def refactor(coverage, m=M, p_break=P_REFACTOR_BREAKS, extra=0.0):
    """No failing test; behaviour must be preserved. `extra` is characterisation
    tests written before refactoring."""
    cov = coverage + (1.0 - coverage) * extra
    broke = rng.random(m) < p_break
    caught = broke & (rng.random(m) < cov)
    return float((~broke | caught).mean()), float((broke & ~caught).mean()), cov


print(f"{M:,} changes. Debugging starts from a failing test; refactoring starts")
print("from passing ones and must keep them passing for the right reason.")
print()
print(f"{'coverage':>10}{'debug: verified':>17}{'debug: broken':>15}"
      f"{'refactor: safe':>16}{'refactor: broken':>18}")
print("-" * 76)
tab = {}
for c in (0.25, 0.45, 0.70, 0.90):
    d = debug(c)
    r = refactor(c)
    tab[c] = (d, r)
    print(f"{c:>10.0%}{d[0]:>17.1%}{d[1]:>15.1%}{r[0]:>16.1%}{r[1]:>18.1%}")

print()
print()
print("The asymmetry: debugging's outcome barely moves with coverage because")
print("the failing test IS the verifier. Refactoring's moves a lot.")
print()
print(f"{'':>26}{'at 25% coverage':>18}{'at 90% coverage':>18}{'range':>9}")
print("-" * 71)
print(f"{'debugging verified':>26}{tab[0.25][0][0]:>18.1%}"
      f"{tab[0.90][0][0]:>18.1%}{tab[0.90][0][0] - tab[0.25][0][0]:>+9.1%}")
print(f"{'refactoring safe':>26}{tab[0.25][1][0]:>18.1%}"
      f"{tab[0.90][1][0]:>18.1%}{tab[0.90][1][0] - tab[0.25][1][0]:>+9.1%}")

print()
print()
print("Now the correlation that matters. Code is refactored because it is bad,")
print("and bad code is undertested.")
print()
print(f"{'code quality':>16}{'typical coverage':>18}{'refactor safe':>15}"
      f"{'broken silently':>17}")
print("-" * 66)
QUALITY = [("well-maintained", 0.85), ("ordinary", 0.62),
           ("legacy, tangled", 0.34), ("nobody understands it", 0.15)]
q = {}
for label, cov in QUALITY:
    r = refactor(cov)
    q[label] = r
    print(f"{label:>16}{cov:>18.0%}{r[0]:>15.1%}{r[1]:>17.1%}")

print()
print()
print("Characterisation tests: write tests that pin the CURRENT behaviour")
print("before touching anything. They do not need to be right about intent --")
print("only about what the code does now, which is exactly what a model")
print("generating tests from code is good at.")
print()
print(f"{'starting coverage':>19}{'no characterisation':>21}"
      f"{'with them':>12}{'gain':>9}")
print("-" * 61)
ch = {}
for cov in (0.15, 0.34, 0.62, 0.85):
    a = refactor(cov)[0]
    b = refactor(cov, extra=0.75)[0]
    ch[cov] = (a, b)
    print(f"{cov:>19.0%}{a:>21.1%}{b:>12.1%}{b - a:>+9.1%}")

print()
print()
print("Which resolves ch:aise-testing's first listing. Generated tests are a")
print("bad specification and an excellent CHARACTERISATION -- the failure mode")
print("there is the requirement here.")
print()
print(f"{'use':>34}{'what it needs':>22}{'generated tests':>18}")
print("-" * 74)
print(f"{'verify a fix is correct':>34}{'independent intent':>22}{'poor':>18}")
print(f"{'pin current behaviour':>34}{'accurate observation':>22}"
      f"{'excellent':>18}")

print()
print()
print("And the ordering that follows for a refactoring agent.")
print()
print(f"{'procedure':>44}{'safe':>10}{'broken':>10}")
print("-" * 64)
seq = {}
for label, cov, extra in (("refactor legacy code directly", 0.34, 0.0),
                          ("characterise, then refactor", 0.34, 0.75),
                          ("characterise, then refactor, ordinary code",
                           0.62, 0.75)):
    r = refactor(cov, extra=extra)
    seq[label] = r
    print(f"{label:>44}{r[0]:>10.1%}{r[1]:>10.1%}")

print(f"""
The first two tables are the asymmetry, and it is larger than it looks.

Debugging's verified rate moves {tab[0.90][0][0] - tab[0.25][0][0]:+.1%} across the
whole coverage range, from {tab[0.25][0][0]:.1%} to {tab[0.90][0][0]:.1%}.
Refactoring's moves {tab[0.90][1][0] - tab[0.25][1][0]:+.1%}.

The reason is that **debugging is handed its verifier and refactoring has to find
one already there** (eq:refactoring-safety-is-coverage). A failing test demonstrates
the bug and confirms the fix; it exists before the agent starts. A refactor has no
such artefact -- the only evidence that behaviour was preserved is a suite that was
written for other reasons and covers what it happens to cover.

So debugging is the strongest position an agent occupies anywhere in this book, and
refactoring is one of the weakest, and they look identical from the outside: read
code, change code, run tests.

The third table is the correlation that makes this bite. Well-maintained code
refactors safely {q['well-maintained'][0]:.1%} of the time; code nobody understands,
{q['nobody understands it'][0]:.1%}, with {q['nobody understands it'][1]:.1%} broken
silently.

**The code most in need of refactoring is the code least able to verify a refactor**
(eq:worst-code-is-least-covered). That is not a coincidence -- undertested and
tangled are the same history -- and it means the highest-value refactoring targets
are exactly the ones an agent should be least trusted with.

The fourth table is the resolution, and it is why this chapter's two listings belong
together.

**Characterisation tests** pin what the code does now, before anything is touched.
They make no claim about what it SHOULD do; they are a snapshot, and their only
requirement is that they accurately describe current behaviour.

Which is precisely what the previous listing found generated tests to be good at.
Tests written from the code achieved the highest coverage in that table and the
lowest defect detection -- because they describe the implementation rather than the
intent. **The failure mode there is the requirement here.**

Adding characterisation tests to legacy code takes refactoring safety from
{ch[0.34][0]:.1%} to {ch[0.34][1]:.1%}, and on the worst code from
{ch[0.15][0]:.1%} to {ch[0.15][1]:.1%} -- the largest gains exactly where the
starting coverage is worst.

The last table gives the procedure. Refactoring legacy code directly is
{seq['refactor legacy code directly'][0]:.1%} safe; characterising first takes it to
{seq['characterise, then refactor'][0]:.1%}, which is better than refactoring
ORDINARY code without characterisation.

So the recommendations separate cleanly by task, and they are opposite:

**For debugging, use the failing test as the verifier and do not generate tests from
the code** -- you already have the independent artefact you need.

**For refactoring, generate tests from the code deliberately, before touching it.**
The generated suite's dependence on the implementation is the property that makes it
a correct characterisation.

**And never use a characterisation suite to verify a FIX.** It asserts the old
behaviour, which is what the fix is supposed to change -- the two uses require
opposite properties from the same artefact, and confusing them is the failure this
chapter is about.""")
```

## 9. Practical Example

The first listing runs modules of which $22\%$ contain a defect:

```
                  tests written by   coverage  catches the bug  passes on it
----------------------------------------------------------------------------
              human, from the spec        75%            68.3%         31.7%
  a different model, from the spec        73%            54.1%         45.9%
     the same model, from the spec        74%            34.4%         65.6%
     the same model, from the code        88%             9.8%         90.2%
```

**The highest coverage in the table has the lowest detection**
({{eq:same-model-tests-codify-the-bug}}) — a suite derived from the implementation
asserts the implementation's defects.

The intervention:

```
                                     from the code  from the spec     gain
--------------------------------------------------------------------------
            catch rate, same model           9.6%          34.6%   +25.0%
                          coverage            88%            74%     -14%
```

**Do not show the model the implementation**
({{eq:write-tests-from-the-spec}}) — $+25$ points of detection for $-14$ of coverage
that was not buying anything.

And why coverage is the wrong instrument:

```
  generated coverage   catch rate   passes on bug
-------------------------------------------------
                 60%         6.3%           93.6%
                 97%        10.4%           89.2%
```

Thirty-seven points of coverage buy four points of detection
({{eq:independence-not-coverage}}). Independence is the binding term:

```
  independence   catch rate  equivalent human coverage
------------------------------------------------------
           11%         9.6%                        10%
           55%        49.5%                        54%
```

The second listing separates the two activities:

```
  coverage  debug: verified  debug: broken  refactor: safe  refactor: broken
----------------------------------------------------------------------------
       25%            75.4%          4.4%           85.6%             14.4%
       90%            79.2%          0.6%           98.4%              1.6%
```

```
                          at 25% coverage   at 90% coverage    range
---------------------------------------------------------------------
        debugging verified           75.4%             79.2%    +3.8%
          refactoring safe           85.6%             98.4%   +12.8%
```

**Debugging is handed its verifier and refactoring has to find one already there**
({{eq:refactoring-safety-is-coverage}}).

And the correlation that makes it bite:

```
    code quality  typical coverage  refactor safe  broken silently
------------------------------------------------------------------
 well-maintained               85%          97.2%             2.8%
 legacy, tangled               34%          87.6%            12.4%
  nobody understands it        15%          83.8%            16.2%
```

**The code most in need of refactoring is least able to verify one**
({{eq:worst-code-is-least-covered}}).

Characterisation tests close it:

```
  starting coverage  no characterisation   with them     gain
-------------------------------------------------------------
                15%                84.1%       96.0%   +11.9%
                34%                87.5%       96.9%    +9.4%
                85%                97.2%       99.2%    +2.0%
```

Largest gains exactly where coverage is worst
({{eq:characterise-then-refactor}}). And the resolution:

```
                               use         what it needs   generated tests
--------------------------------------------------------------------------
           verify a fix is correct    independent intent              poor
             pin current behaviour  accurate observation         excellent
```

**The failure mode in the first listing is the requirement in the second.**

```
                                   procedure      safe    broken
----------------------------------------------------------------
               refactor legacy code directly     87.4%     12.6%
                 characterise, then refactor     96.8%      3.2%
```

## 10. Production Considerations

Report mutation score, not coverage, for any generated suite. Coverage is the factor
that generation inflates and mutation score is the product that matters.

Write tests from the specification and withhold the implementation — and arrange it
structurally, since in an agent loop the code is usually already in context.

Expect a specification-derived suite to fail on first run. A clean pass means the
model saw the code.

Use a different model for tests than for code where you can. Independence composes.

Never use a generated-from-code suite as the verifier in an agent's iteration loop.
It is {{ch:ag-recovery}}'s resample with extra steps.

For refactoring: characterise first, refactor in individually-verified steps, re-run
after each, and **deny the agent permission to modify the characterisation suite.**

Treat a failing characterisation test as a stop, never as a test to update.

And ask of every test artefact whether it asserts what the code should do or records
what it does. The two need opposite construction.

## 11. Common Mistakes

**Reading generated coverage as safety.** It is the factor generation inflates.

**Generating tests with the implementation in context.** The independence is lost
before the prompt is read.

**Using generated tests to verify a fix.** They assert the behaviour the fix is
changing.

**Using specification tests as a characterisation.** They lack the fidelity that job
needs.

**Letting a refactoring agent update failing characterisation tests.** Converts a
safety net into a rubber stamp.

**Refactoring untested legacy code with an agent.** The highest value at the lowest
safety.

**Treating debugging and refactoring as one capability.** Coverage enters different
terms.

## 12. Failure Modes

*Tests that codify the bug.* The characteristic failure — high coverage, green
suite, defect intact.

*Coverage theatre.* A number doubled by generation and meaning less than before.

*Silent behaviour change.* A refactor that altered untested behaviour, discovered
when a user does.

*Characterisation drift.* Tests updated as they fail until they assert whatever the
code now does.

*Loop with a correlated verifier.* An agent iterating against tests that cannot
disagree with it.

*Fix verified by a snapshot.* A characterisation suite used to confirm a change it
was written to forbid.

## 13. Alternatives

**Property-based testing.** Assertions about invariants rather than about specific
outputs, which are far harder to derive from an implementation and therefore carry
more independence.

**Metamorphic testing.** Relations between outputs — sorting twice equals sorting
once, a discount applied then reversed returns the original — that hold regardless of
implementation and can be stated from the specification alone. These are unusually
well suited to generation, because enumerating plausible invariants is a task a model
does well and checking them is mechanical.

**Test-first generation.** The model writes tests from the issue *before* seeing or
writing any code, which arranges independence by ordering.

**Human-written tests, agent-written code.** {{ch:aids-oversight}}'s
divide-by-gradeability applied here: the human keeps the specification and delegates
the implementation.

**Differential testing against the pre-change version.**
{{ch:aise-swe-agents}}'s reference-free technique, which is a characterisation
performed at runtime rather than written down.

## 14. Evaluation

Run mutation testing on your generated suites. It is the direct measurement of the
term that matters and it is the number this chapter is about.

Measure detection against seeded defects, separately from coverage. Report both.

Measure how often a specification-derived suite fails on first run against existing
code. A low rate means independence was not achieved.

For refactoring, measure behaviour change in untested regions — the failure that by
construction no test reports.

And track characterisation tests that were modified during a refactor. Each one is a
safety net that was cut.

## 15. Advanced Concepts

**Mutation-guided test generation.** Generating tests specifically to kill surviving
mutants, which optimises $r\iota$ directly rather than $r$.
{{maturity:EMERGING}}.

**Specification extraction from usage.** Deriving intent from call sites and
documentation rather than from the implementation, which would raise $\iota$ without
requiring a written specification.

**Automatic characterisation scoping.** Determining which behaviours a planned
refactor could affect, so characterisation is targeted rather than blanket.

**Independence estimation without ground truth.** Measuring $\iota$ from disagreement
between suites written by different sources, which would make the binding term
reportable. {{maturity:RESEARCH FRONTIER}}.

## 16. Connection to Previous Chapters

{{ch:aids-autonomous}}'s self-judging result reaches its sharpest form here: the
checker is not merely correlated with the generator, it is reading its output.

{{ch:as-failures}}'s decorrelation result explains why a different model helps and
why the sources of independence compose.

{{ch:aise-swe-agents}}'s partial-specification finding acquires its origin story —
the specification is only partial *and* only independent to the extent someone wrote
it from intent.

{{ch:aise-repo}}'s reproduction result is confirmed from a third direction: a failing
test supplies localisation, a verifier, a termination condition and a regression
guard at once.

{{ch:ag-planning}}'s checkpoints return as refactoring in individually-verified
steps.

Ahead: {{ch:aise-cicd}} takes these artefacts into a pipeline, where the question
becomes what an automated change is permitted to do without a human.

## 17. Exercises

1. Run mutation testing against a generated suite and a hand-written one for the
   same module. Report both mutation scores and both coverage figures.

2. Generate tests from a docstring with the implementation withheld, run them, and
   count first-run failures.

3. Measure $\iota$ empirically by seeding defects and computing catch rate divided
   by coverage.

4. Characterise a legacy module, refactor it, and count how many characterisation
   tests you were tempted to modify.

5. Implement property-based tests for a function and compare their mutation score
   with example-based generated tests.

6. Model a refactoring agent permitted to edit its characterisation suite and measure
   how quickly safety degrades.

## 18. Interview Questions

1. Your generated tests reach $92\%$ coverage. What have you learned?

2. Why would a test suite written by the model that wrote the code miss its bugs?

3. What single change to the prompt most improves generated test quality?

4. Why is debugging easier for an agent than refactoring?

5. When is it correct to generate tests from the implementation?

6. A characterisation test fails during a refactor. What should the agent do?

## 19. Research Questions

1. Can independence be estimated from inter-suite disagreement without seeded
   defects?

2. How much independence does specification-derived generation actually recover, and
   does it survive an agent loop where the code is in context?

3. Do property-based and metamorphic tests carry measurably more independence than
   example-based ones?

4. Can characterisation scope be derived automatically from a planned change?

5. How strong is the empirical correlation between refactoring demand and coverage?

## 20. Chapter Summary

Everything in this part rests on a test suite being an *independent* partial
specification, and generated tests are not one. A suite written from the code
achieved the **highest coverage in the table at $88\%$ and detected $9.8\%$ of
defects**, against a human suite's $75\%$ and $68.3\%$
({{eq:same-model-tests-codify-the-bug}}) — because a suite derived from an
implementation asserts that implementation's defects.

Detection is $r\iota$: coverage times independence. **Coverage is the reported factor
and independence is the binding one** ({{eq:independence-not-coverage}}) —
thirty-seven points of generated coverage bought four points of detection, and the
marginal return on independence is roughly eight times larger.

The intervention is a prompt-level decision: writing from the **specification**
rather than the code took detection from $9.6\%$ to $34.6\%$ for a fourteen-point
coverage cost ({{eq:write-tests-from-the-spec}}), and a different model does better
still. Independence has to be arranged structurally, since in an agent loop the
implementation is already in context.

Debugging and refactoring look identical and are not. **Debugging is handed its
verifier** — a failing test supplies localisation, a check, a termination condition
and a regression guard at once — so its success moved $+3.8$ points across the whole
coverage range. **Refactoring must find a verifier already there**, so its safety is
coverage ({{eq:refactoring-safety-is-coverage}}) and moved $+12.8$. And the demand is
inversely correlated with the safety: refactoring succeeded $97.2\%$ of the time in
well-maintained code and $83.8\%$ in code nobody understands
({{eq:worst-code-is-least-covered}}).

The resolution is that the two listings answer each other. **Characterisation tests**
pin current behaviour, requiring fidelity rather than independence — which is
precisely what generated-from-code tests supply. They lifted refactoring safety on
legacy code from $87.5\%$ to $96.9\%$, with the largest gains where coverage was
worst ({{eq:characterise-then-refactor}}).

So the recommendations reverse by task, and the distinguishing question is one
sentence: **does this artefact assert what the code should do, or record what it
currently does?** The first requires independence; the second requires fidelity; they
are satisfied by opposite construction methods, and confusing them is the failure
this chapter is about.

## 21. Further Reading

{{cite:wang2025solvedcorrectly}} for what an incomplete specification permits, which
is this chapter's premise, and for differential testing as characterisation performed
at runtime.

{{cite:lu2024aiscientist}} for the self-judging structure in its purest form, and
{{ch:aids-autonomous}} for the measurement of what correlation does to a reported
number.

{{cite:jimenez2023swebench}} and {{cite:chan2024mlebench}} for benchmarks whose
grading depends entirely on suites written by humans from intent — which is what
makes them evidence.

{{ch:aise-repo}} for reproduction, and {{ch:ag-planning}} for the checkpoint argument
that governs refactoring step size.
