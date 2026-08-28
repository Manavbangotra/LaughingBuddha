---
id: rsn-tool-assisted
number: 151
part: XVI
tier: full
status: draft
requires: [verifier-quality-ceiling, lucky-chain-rate,
           per-step-error-compounding]
provides: [check-turns-coverage-into-accuracy, specification-is-the-ceiling,
           tool-error-reallocation, translation-versus-execution,
           boundary-crossing-cost, green-run-failure]
citations: [gao2023pal, yao2023react, sprague2024tocot, brown2024monkeys,
            deepseek2025r1, lightman2023verify, snell2024testtime,
            huang2024selfcorrect]
---

## 1. Learning Objectives

By the end of this chapter you will be able to say precisely what an executable
check changes about the accuracy-versus-compute curve, and why the change is one
of shape rather than height; recognise the failure mode an incomplete
specification produces and why it is quieter than the one it replaces; decide
whether a tool helps on your task from two measurements rather than from
principle; explain why one program beats $k$ interleaved calls, and when it does
not; and locate the point at which extra compute stops buying accuracy and starts
buying pressure on your specification.

## 2. Why This Matters

This is the chapter where things work, and it is worth being clear about why,
because the reason is narrow.

Every preceding chapter of {{part:16}} ended at the same constraint. Sampling
produces coverage that a selector fails to recover ({{ch:rsn-test-time-compute}}).
Voting is bounded by the generator's mode and reflection by the critic's
correlation with the solver ({{ch:rsn-self-consistency}}). Reward models trained on
outcomes are biased in a way more data does not fix
({{ch:rsn-supervision}}). In each case the binding constraint was a *learned*
judgement about whether an answer is right.

An executable check is not a learned judgement. A test suite, a type checker, a
proof assistant, a constraint solver, an interpreter — these give a verdict that is
correct on the property they check, by construction, and
{{sec:9-practical-example}} shows what that does: delivered accuracy stops being a
fraction of coverage and *becomes* coverage. A learned verifier took a system from
$9.6\%$ to $72.3\%$ over a $128\times$ budget; a complete check took the same
system from $9.6\%$ to $100\%$, tracking the coverage curve exactly at every
budget in between.

That is a different kind of curve, not a better point on the same one. It means
the sample budget is now the binding constraint, and the sample budget is the one
thing in this entire part you can simply buy.

There is a second reason tools belong here, and it comes from
{{ch:rsn-cot}}. {{cite:sprague2024tocot}} found chain-of-thought's gains
concentrated in maths and symbolic reasoning, and that where it helps, it is doing
symbolic execution a real solver does better. {{cite:gao2023pal}} says the same
thing from the other side: models decompose problems correctly and then make
arithmetic mistakes in the solution. So the natural move is to keep the
decomposition and hand the execution away.

But tools are not free, and the second half of this chapter is about the two bills
that arrive. A tool call replaces execution error with *translation* error, and
{{sec:9-practical-example}} finds a design where that trade loses at every chain
length while a different design using the same tool wins from $k=5$. And a check
verifies the property it checks rather than correctness, so an incomplete
specification produces the quietest failure mode in this part: a system that ships
defective work $78\%$ of the time with a green test run on every problem.

## 3. Prerequisites

You need {{ch:rsn-test-time-compute}}'s coverage/selection decomposition and its
verifier-quality parameter $q$ — this chapter is the case $q = 1$, and the
interesting content is what that does to the curve rather than to any single
number.

You need {{ch:rsn-cot}}'s compounding result, {{eq:chain-accuracy-compounds}},
because the entire tool-versus-no-tool comparison is a comparison of two
geometric decays with different bases.

From {{ch:rsn-supervision}}, the lucky-chain rate recurs here in a new costume: a
sample that passes an incomplete check is the same object as a chain that reaches
a right answer by wrong reasoning, and it causes the same trouble for the same
reason.

## 4. Intuitive Explanation

Think about what changes when you can *run* the answer.

Without a check, producing a correct answer is a generation problem. The model has
to get it right, and if you sample a hundred candidates you still have to decide
which one to keep — which is the hard part, because deciding requires the same
knowledge as producing.

With a check, it becomes a search problem. You do not need to know which candidate
is right; you need only to try them until one passes. The knowledge required to
*recognise* success has moved out of the model and into a piece of software, and
software of that kind does not have false positives on the thing it checks.

That single change reorganises the economics. Sampling more was previously
limited: extra candidates raised coverage but a mediocre verifier could not cash
them in, so accuracy flattened. Now every extra candidate is a clean additional
attempt, and coverage — which {{cite:brown2024monkeys}} showed rising log-linearly
over orders of magnitude — becomes accuracy directly. The reason coding and formal
mathematics produced the most dramatic test-time-compute results is not that
models are unusually good at them. It is that those domains ship checkers.

Now the two costs, because they are what separate a working system from a
plausible one.

The first is that using a tool is a skill of its own. The model no longer has to
compute the answer, but it does have to say what it wants computed — write the
program, format the call, get the arguments in the right order — and then read the
result back and use it correctly. Those are new opportunities to be wrong. On a
one-step problem they can easily outweigh the error they replaced, and
{{sec:9-practical-example}} measures exactly that: a per-step tool that is *worse*
than the unaided model at every chain length tested, because it pays the
translate-and-parse cost once per step.

The fix is not to abandon the tool but to cross the boundary less often. Write one
program for the whole problem and run it once, and you pay the parsing cost a
single time while keeping the exact execution throughout. The same tool, arranged
differently, wins from five steps onward. **The question is not whether to use a
tool but how many times to hand control back and forth.**

The second cost is subtler and it is the one that will actually hurt you.

A check tells you whether the answer passes *the check*. If your tests cover
everything that could go wrong, that is the same as correct. If they do not, then
"passes" and "correct" are different sets, and best-of-$n$ against the check is a
search for things in the first set. With enough sampling, it finds the ones that
are not in the second.

What that looks like from outside is worse than it sounds. It does not look like
degraded output — {{sec:9-practical-example}} finds shipped correctness roughly
flat as the budget rises. It looks like a *pass rate going to $100\%$*. Every
problem produces something green, the check stops distinguishing anything, and the
signal you added the tool to obtain has quietly stopped carrying information.

So a tool moves the binding constraint from verification to specification. That is
a genuine and large improvement, because a specification is something you can
read, review, version and extend, and a learned verifier's failure modes are none
of those. It is not the same as removing the constraint.

## 5. Formal Explanation

Take {{ch:rsn-test-time-compute}}'s decomposition. Delivered accuracy is coverage
passed through a selector, and for a verifier of quality $q$:

$$A(n) \;\approx\; q \cdot C(n), \qquad C(n) = 1 - (1-p)^{n}$$ (eq:verifier-recovers-coverage)

An executable check that is *sound and complete* for correctness has $q = 1$
exactly, giving:

$$A_{\text{check}}(n) \;=\; C(n) \;=\; 1 - (1-p)^{n}$$ (eq:check-turns-coverage-into-accuracy)

Note the qualitative difference from a learned verifier. A learned scorer selecting
the maximum over $n$ candidates suffers the extreme-value problem of
{{eq:argmax-extreme-value}}: its effective quality *degrades* with $n$, so
$A(n)$ flattens and can turn over. {{eq:check-turns-coverage-into-accuracy}} has
no such term. It is monotone in $n$ and converges to $1$.

Now drop completeness. Let the possible defects be $b \in \{1, \ldots, B\}$ with
prior $\pi_b$, and let $\mathcal{D}$ be the subset the specification detects. A
candidate passes if it is correct or if its defect is undetected, so:

$$\Pr[\text{pass}] = p + \sum_{b \notin \mathcal{D}} \pi_b, \qquad \Pr[\text{correct} \mid \text{pass}] = \frac{p}{p + \sum_{b \notin \mathcal{D}} \pi_b}$$ (eq:specification-is-the-ceiling)

The right-hand expression is the *precision of the check*, and it does not depend
on $n$. Sampling more raises the probability that something passes and leaves the
quality of what passes unchanged. In the limit:

$$A_{\text{incomplete}}(n) \;\xrightarrow{\;n \to \infty\;}\; \frac{p}{p + \sum_{b \notin \mathcal{D}} \pi_b}$$ (eq:incomplete-check-limit)

So an incomplete specification converts an unbounded compute budget into a bounded
accuracy, and the bound is a property of the specification alone. That is the
formal content of "the specification is the ceiling", and it is why
{{sec:9-practical-example}}'s incomplete checks flatten by $n = 32$ while the
complete one is still climbing.

Now the tool-versus-unaided comparison. Let the model execute one step correctly
with probability $p_e$, translate one step into a call correctly with probability
$p_t$, and read one result back correctly with probability $p_r$. For a $k$-step
problem:

$$A_{\text{unaided}} = p_e^{\,k}, \qquad A_{\text{per-step}} = (p_t p_r)^{k}, \qquad A_{\text{one-call}} = p_t^{\,k} \, p_r$$ (eq:tool-error-reallocation)

Three geometric decays. The comparison is entirely a comparison of bases, with one
constant factor:

- A per-step tool beats the unaided model iff $p_t p_r > p_e$ — a condition
  independent of $k$, so it either wins at every length or loses at every length.
- A single call beats the unaided model iff $(p_t/p_e)^k > 1/p_r$, which *is*
  $k$-dependent, giving a crossover at
  $k^{*} = \ln(1/p_r) / \ln(p_t/p_e)$.

**The parse cost is paid once per boundary crossing, and the translation advantage
compounds per step.** That is the whole design argument, and it says the number of
round trips is the variable to minimise, not the presence of the tool.

The condition $p_t > p_e$ deserves a name because everything rests on it: a tool
helps when the model is better at *asking for* the computation than at *doing* it.
{{cite:gao2023pal}}'s central observation is that this inequality holds for
arithmetic — models decompose correctly and then make arithmetic mistakes — and it
is an empirical claim about a model and an interface, not a general truth.

## 6. Mathematical Foundation

Two consequences of {{eq:tool-error-reallocation}} are worth extracting, because
they are where intuition goes wrong.

**The crossover is sensitive to $p_r$ and insensitive to $k$.** From
$k^{*} = \ln(1/p_r)/\ln(p_t/p_e)$, the numerator moves fast when $p_r$ is near 1 —
$p_r = 0.99$ gives $\ln(1/p_r) \approx 0.010$, $p_r = 0.90$ gives $0.105$, a
tenfold change — while the denominator is a ratio of two numbers that are usually
both close to 1, making it small and therefore making $k^*$ large and volatile. In
practice this means the crossover point is hard to predict and easy to measure, so
measure it.

**Improving translation has an exponential return; improving parsing has a
constant one.** $\partial A_{\text{one-call}} / \partial p_t$ carries a factor of
$k$; $\partial A_{\text{one-call}} / \partial p_r$ does not. If you are choosing
where to spend engineering effort — better prompts and schemas for constructing
the call, versus more robust result parsing — the first scales with problem
complexity and the second does not.

Now the specification side. How large is $\sum_{b \notin \mathcal{D}} \pi_b$ in
practice? It is one minus the fraction of *defect probability mass* your tests
cover, which is not the same as the fraction of defect *types* they cover, and the
difference matters because defects are heavily skewed. Catching the four commonest
of five defect types in {{sec:9-practical-example}} leaves $5\%$ of the mass
uncovered and caps accuracy at $66\%$; catching three leaves considerably more and
caps it at $37\%$.

This has an uncomfortable corollary. Line coverage, branch coverage, and the other
metrics a test suite reports measure how much of the *code* is exercised, not how
much of the defect distribution is detected. They are not estimates of
$\sum_{b \in \mathcal{D}} \pi_b$ and should not be read as one when the test suite
is doubling as an optimisation target.

Finally, combine the two halves. A tool-assisted system with a check has accuracy:

$$A = \Big[\text{coverage under } p_t^{\,k} p_r\Big] \times \Big[\text{precision of the check}\Big]$$ (eq:tool-and-check)

The first factor is what sampling buys and the second is what the specification
caps. They are independent, they fail differently, and a system that measures only
the first will report progress right up to the point where the second binds — at
which point the pass rate is $100\%$ and the accuracy is whatever the
specification allowed.

## 7. Internal Mechanics

### 7.1 What "executable" buys, precisely

The property that matters is not that the checker is fast or deterministic. It is
that its verdict on the property it checks is *not a prediction*. A learned
verifier estimates $\Pr[\text{correct} \mid \text{solution}]$ and has an error rate
on both classes; a type checker either finds the type error or the type error is
not there.

This is why {{eq:check-turns-coverage-into-accuracy}} has no extreme-value term.
Best-of-$n$ against a learned score searches for the sample the score likes most,
which for large $n$ is the sample whose score error is largest. Best-of-$n$ against
a check searches for a sample that passes, and there is no "most passing" — the
verdict is binary and cannot be gamed within the property it covers. All the
gaming moves to the boundary of what it covers, which is
{{eq:specification-is-the-ceiling}}.

### 7.2 Where the boundary crossings are

A round trip to a tool is not one operation. It is: decide to call, construct the
call, serialise it, execute, receive, parse, and integrate the result into the
ongoing state. Each of those is a place a model can go wrong, and
{{eq:tool-error-reallocation}}'s $p_t$ and $p_r$ are aggregates over them.

Two design consequences follow. First, anything that reduces the number of
crossings — one program instead of $k$ calls, a batched query instead of $n$
lookups — multiplies through the whole chain. Second, anything that makes a single
crossing more reliable — a constrained decoding grammar for the call
({{part:8}}), a schema the model has seen in training, a response format with no
ambiguity — raises $p_t$ or $p_r$, and the first of those has the exponential
return.

### 7.3 The serving shape

Interleaved tool use is the worst-shaped request in {{part:15}}'s cost model. Each
call is serial — the next one needs the previous result — so nothing parallelises;
each return re-enters the model with a longer context, so the KV cache grows and
per-token cost rises; and the model is idle during tool execution while its cache
occupies memory.

A single program call has one of everything. {{sec:9-practical-example}} measures
accuracy per round trip at $k=12$: $3.55\%$ for the per-step design against
$66.80\%$ for the single call, a factor of nearly twenty, before any accuracy
comparison at all.

### 7.4 What tools do not fix

They do not fix a problem the model cannot decompose. {{cite:gao2023pal}}'s split
leaves decomposition with the model, and if the decomposition is wrong the
interpreter executes the wrong program perfectly. The failure is then *more*
confident than the unaided version, because there is an authoritative-looking
computed number attached to it.

They do not fix tasks with no checkable property. Most of what production systems
do — summarise, advise, draft, judge — has no interpreter. The whole apparatus of
this chapter is unavailable there, and {{ch:rsn-benchmarks}} is partly about the
consequences of extrapolating from the domains that do have one.

And they do not remove {{ch:rsn-cot}}'s faithfulness problem. A trace containing a
tool call is more legible than one that does not — you can see what was computed —
but the model's *stated reason* for making that call is still produced by the
untied head of {{eq:trace-and-answer-are-untied}}. What you gain is a verifiable
record of a computation, which is genuinely valuable and is not an explanation.

### 7.5 Verifiable rewards

{{cite:deepseek2025r1}}'s training signal is a checker rather than a learned reward
model, which is {{ch:rsn-supervision}}'s bias problem solved by construction: a
checker's label is correct on the checked property, so the lucky-chain rate for
that property is zero.

The same caveat applies with more force during training than during selection. A
selector with an incomplete specification picks candidates that exploit the gap; a
*generator trained against* an incomplete specification is optimised to find the
gap, with far more compute and far more freedom. {{eq:specification-is-the-ceiling}}
is a ceiling for selection and a target for training.

## 8. Implementation

Two listings. The first replaces a learned verifier with an executable check and
measures what happens to the accuracy-versus-compute curve, then makes the
specification incomplete and measures what happens to the same curve and to what
the system ships. The second prices the tool call itself.

```python {tier=A name=check-changes-the-curve}
"""An executable check changes the SHAPE of the accuracy/compute curve.

ch:rsn-test-time-compute measured a verifier of quality q recovering q of the
available coverage, and left the obvious question open: what happens when q is
1? An executable check -- a test suite, a type checker, a solver, an interpreter
-- is the case where it can be, and the interesting part is not that accuracy
goes up. It is that the CURVE changes from one that saturates to one that keeps
climbing with the sample budget (eq:check-turns-coverage-into-accuracy).

The second half is the part that gets skipped. An executable check does not
verify correctness. It verifies the PROPERTY IT CHECKS, and the two coincide only
if the specification is complete. This listing sweeps how complete it is, and the
result is that the same extra compute which is pure profit under a complete check
becomes adversarial pressure under an incomplete one.
"""
import numpy as np

rng = np.random.default_rng(1117)

N_PROB = 5000
NMAX = 128
P_SOLVE = 0.09            # chance one sample is fully correct
N_BUGS = 5                # distinct ways a sample can be wrong


def make_pool(n_prob, ns):
    """Each sample is either correct (0) or carries one of N_BUGS defects.
    Defects are drawn with a skew, so some are far more common than others --
    which is what makes an incomplete specification dangerous rather than merely
    incomplete."""
    correct = rng.random((n_prob, ns)) < P_SOLVE
    w = np.array([0.40, 0.25, 0.18, 0.12, 0.05])
    bug = rng.choice(np.arange(1, N_BUGS + 1), size=(n_prob, ns), p=w)
    return np.where(correct, 0, bug)


POOL = make_pool(N_PROB, NMAX)
ROW = np.arange(N_PROB)


def learned_score(pool, mu):
    """A learned verifier: correct samples score N(mu, 1), wrong ones N(0, 1)."""
    return rng.normal(size=pool.shape) + mu * (pool == 0)


def check_catches(pool, caught):
    """An executable check that detects the defects in `caught` and is silent
    about the rest. A sample PASSES if it is correct, or if its defect is one
    the specification never mentions."""
    passes = pool == 0
    for b in range(1, N_BUGS + 1):
        if b not in caught:
            passes |= (pool == b)
    return passes


BUDGETS = [1, 2, 4, 8, 16, 32, 64, 128]
MU = 1.4                  # a decent learned verifier

print(f"{N_PROB} problems. One sample is fully correct {P_SOLVE:.0%} of the")
print(f"time; otherwise it carries one of {N_BUGS} defects, drawn with a skew.")
print()
print("Selection by a learned verifier, versus by an executable check that")
print("catches every defect.")
print()
print(f"{'samples n':>10}{'coverage':>11}{'learned':>11}{'complete':>11}")
print(f"{'':>10}{'':>11}{'verifier':>11}{'check':>11}")
print("-" * 43)

S_LEARNED = learned_score(POOL, MU)
PASS_ALL = check_catches(POOL, set(range(1, N_BUGS + 1)))
cov, lrn, chk = {}, {}, {}
for n in BUDGETS:
    sub = POOL[:, :n]
    cov[n] = float((sub == 0).any(1).mean())
    idx = S_LEARNED[:, :n].argmax(1)
    lrn[n] = float((sub[ROW, idx] == 0).mean())
    # With a check you keep the FIRST sample that passes; if none passes you are
    # left with the last one you tried.
    p = PASS_ALL[:, :n]
    first = np.where(p.any(1), p.argmax(1), n - 1)
    chk[n] = float((sub[ROW, first] == 0).mean())
    print(f"{n:>10}{cov[n]:>11.1%}{lrn[n]:>11.1%}{chk[n]:>11.1%}")

print()
print()
print("Now make the specification incomplete. The check catches the defects it")
print("knows about and passes the rest. Accuracy at each budget:")
print()
CASES = [("catches all 5", set(range(1, 6))),
         ("catches 4 of 5 (misses the rarest)", {1, 2, 3, 4}),
         ("catches 3 of 5", {1, 2, 3}),
         ("catches 2 of 5 (the two commonest)", {1, 2})]
print(f"{'specification':>36}" + "".join(f"{'n=' + str(n):>9}"
                                         for n in (1, 8, 32, 128)))
print("-" * 72)
inc = {}
for name, caught in CASES:
    p = check_catches(POOL, caught)
    row = {}
    for n in (1, 8, 32, 128):
        pn = p[:, :n]
        first = np.where(pn.any(1), pn.argmax(1), n - 1)
        row[n] = float((POOL[ROW, first][:, None][:, 0] == 0).mean()) \
            if False else float((POOL[:, :n][ROW, first] == 0).mean())
    inc[name] = row
    print(f"{name:>36}" + "".join(f"{row[n]:>9.1%}" for n in (1, 8, 32, 128)))

print()
print()
print("What is the system SHIPPING? Among the samples that passed the check,")
print("what fraction are actually correct, and what does the rest consist of?")
print()
print(f"{'specification':>36}{'shipped':>10}{'shipped':>10}{'and passed':>13}")
print(f"{'':>36}{'at n=1':>10}{'at n=128':>10}{'the check':>13}")
print("-" * 69)
ship = {}
for name, caught in CASES:
    p = check_catches(POOL, caught)
    out = []
    for n in (1, 128):
        pn = p[:, :n]
        first = np.where(pn.any(1), pn.argmax(1), n - 1)
        sel = POOL[:, :n][ROW, first]
        passed = pn[ROW, first]
        out.append(float((sel[passed] == 0).mean()) if passed.any() else 0.0)
    ship[name] = (out[0], out[1], float(p[:, :128].any(1).mean()))
    print(f"{name:>36}{out[0]:>10.1%}{out[1]:>10.1%}{ship[name][2]:>13.1%}")

full = inc["catches all 5"]
four = inc["catches 4 of 5 (misses the rarest)"]
two = inc["catches 2 of 5 (the two commonest)"]
S4 = ship["catches 4 of 5 (misses the rarest)"]
S2 = ship["catches 2 of 5 (the two commonest)"]
print(f"""
The first table is the shape change, and the columns matter more than any single
number in them.

The learned verifier goes {lrn[1]:.1%} -> {lrn[128]:.1%} over a 128x budget. The
complete check goes {chk[1]:.1%} -> {chk[128]:.1%}, tracking coverage
({cov[128]:.1%}) exactly at every budget.

Those are different kinds of curve, not the same curve at different heights. The
learned verifier's gains flatten because it competes against its own false
positives -- ch:rsn-self-consistency's extreme-value problem, where the top score
over a growing pool increasingly belongs to whichever wrong sample scored
luckiest. A complete check has no false positives on the property it checks, so
each extra sample is a clean additional draw and delivered accuracy IS coverage
(eq:check-turns-coverage-into-accuracy).

That is the argument for tool-assisted reasoning stated quantitatively, and it is
not "a checker is a better verifier". It is that a checker moves the binding
constraint OFF the verifier and ONTO the sample budget -- and the sample budget is
the one thing in this whole part that you can simply buy.

It also changes the problem the model is being asked to solve. Without a check
the model must GENERATE a correct answer. With one it must generate a correct
answer somewhere in {NMAX} tries, which is a search problem, and
cite:brown2024monkeys's coverage curve says search is exactly what sampling is
good at.

The second table is what is usually left out, and it changes the conclusion from
"tools fix this" to something more precise.

An executable check does not verify correctness. It verifies the property it
checks. Give it an incomplete specification and the curve does not merely sit
lower -- it stops climbing. Catching four defects out of five reaches
{four[8]:.1%} at n=8 and {four[128]:.1%} at n=128, having gained
{four[128] - four[32]:+.1%} over the last four doublings. Catching two of five
saturates at {two[128]:.1%}.

Compare that with the complete check, which was still gaining
{full[128] - full[32]:+.1%} over the same range and reached {full[128]:.1%}.

**The ceiling is the specification, not the budget.** Once sampling has found a
sample the check accepts, additional compute cannot improve on it, and the check
accepts anything whose defect the specification never mentioned.

The third table says what that means for what actually ships, and it is the
number to take away from this chapter.

At n=128, every specification -- including the ones catching two defects out of
five -- produces a passing sample for {S2[2]:.0%} of problems. The check is green
everywhere. Among those passing samples, the fraction that are actually correct
is {ship['catches all 5'][1]:.1%}, {S4[1]:.1%}, {ship['catches 3 of 5'][1]:.1%}
and {S2[1]:.1%} respectively.

So a system with a two-of-five specification reports a {S2[2]:.0%} pass rate and
ships defective work {1 - S2[1]:.1%} of the time.

Note what did NOT happen, because the usual telling of this story overstates it.
Shipped correctness barely moved with the budget: {S2[0]:.1%} at n=1 against
{S2[1]:.1%} at n=128. Sampling harder did not corrupt the output. What it did was
drive the PASS RATE to {S2[2]:.0%} while correctness stayed flat, which destroys
the check's value as a signal rather than its value as a filter. Before the extra
compute, a failing check told you something. After it, the check passes on
everything and tells you nothing.

That is the quiet failure mode, and it is quieter than the one tools were brought
in to fix. A learned verifier that is wrong gives you a mediocre number. An
incomplete executable check gives you a green run.

Which is the rule this chapter exists to state. **A tool moves the binding
constraint from verification to specification.** That is a large improvement,
because a specification is a thing you can read, version, extend, review and
argue about, and a learned verifier's failure modes are none of those. It is not
the same as removing the constraint. The test suite is now both the evaluation
harness and the optimisation target, and those two roles have different
requirements: a harness needs to be representative, a target needs to be
COMPLETE, and the gap between them is exactly what extra compute will find.""")
```

The second listing sets the check aside and asks what a tool call costs.

```python {tier=A name=tool-error-reallocation}
"""When is a tool worth it? The error budget, reallocated.

cite:sprague2024tocot found chain-of-thought's gains concentrated in maths and
symbolic reasoning, and that where it helps it is doing symbolic execution a real
solver does better. That is an argument for handing the execution to a tool, and
cite:yao2023react and cite:gao2023pal are two of the ways to do it.

The argument is usually made as though tools are free. They are not. Calling a
tool replaces one error mode with another: the model no longer has to EXECUTE the
computation, but it does have to TRANSLATE the problem into a call, and a wrong
call produces a confidently wrong result with an authoritative-looking number
attached (eq:tool-error-reallocation).

This listing measures the trade on a task with an explicit number of steps, so
the crossover is visible rather than asserted.
"""
import numpy as np

rng = np.random.default_rng(1223)

N_PROB = 40000
P_EXEC = 0.96             # chance the model executes one step correctly itself
P_TRANS = 0.97            # chance it translates one step into a correct call
P_PARSE = 0.96            # chance it reads the tool's answer back correctly
LATENCY_TOOL = 1.0        # relative cost of a round trip, per call


def unaided(k):
    """The model does every step itself. Errors compound: the chain is right
    only if every step is (ch:rsn-cot's eq:chain-accuracy-compounds)."""
    return float(np.mean((rng.random((N_PROB, k)) < P_EXEC).all(1)))


def tool_per_step(k):
    """One tool call per step. Execution is exact, so the only ways to fail are
    translating a step into a call and reading the result back."""
    ok = ((rng.random((N_PROB, k)) < P_TRANS) &
          (rng.random((N_PROB, k)) < P_PARSE)).all(1)
    return float(np.mean(ok))


def tool_once(k):
    """One call for the WHOLE problem -- write the program, run it once. The
    translation now has to be right about all k steps at once, which is harder
    per call, but there is only one call and one result to read."""
    p_prog = P_TRANS ** k
    ok = (rng.random(N_PROB) < p_prog) & (rng.random(N_PROB) < P_PARSE)
    return float(np.mean(ok))


KS = [1, 2, 3, 5, 8, 12, 20, 32]

print(f"The model executes a step correctly {P_EXEC:.0%} of the time and")
print(f"translates one into a tool call {P_TRANS:.0%} of the time, reading the")
print(f"result back correctly {P_PARSE:.1%} of the time. Accuracy by chain")
print("length:")
print()
print(f"{'steps k':>9}{'unaided':>11}{'tool per':>11}{'one tool':>11}"
      f"{'calls':>9}")
print(f"{'':>9}{'':>11}{'step':>11}{'call':>11}{'made':>9}")
print("-" * 51)

una, tps, ton = {}, {}, {}
for k in KS:
    una[k], tps[k], ton[k] = unaided(k), tool_per_step(k), tool_once(k)
    print(f"{k:>9}{una[k]:>11.1%}{tps[k]:>11.1%}{ton[k]:>11.1%}{k:>9}")

cross = [k for k in KS if tps[k] > una[k]]
cross1 = [k for k in KS if ton[k] > una[k]]

print()
print()
print("The same comparison as a ratio, which is what decides whether the round")
print("trips are worth paying for.")
print()
print(f"{'steps k':>9}{'per-step tool':>16}{'single call':>14}")
print(f"{'':>9}{'vs unaided':>16}{'vs unaided':>14}")
print("-" * 39)
for k in KS:
    print(f"{k:>9}{tps[k] / una[k]:>15.2f}x{ton[k] / una[k]:>13.2f}x")

print()
print()
print("Now vary how good the model is at using the tool, at k=12. A worse")
print("translator eats the advantage; the question is how fast.")
print()
print(f"{'translation':>13}{'unaided':>10}{'tool per':>11}{'one tool':>11}")
print(f"{'accuracy':>13}{'':>10}{'step':>11}{'call':>11}")
print("-" * 45)
K_STUDY = 12
u12 = una[K_STUDY]
sweep = {}
for pt in (0.999, 0.99, 0.97, 0.95, 0.92, 0.88):
    P_TRANS_SAVE = P_TRANS
    globals()["P_TRANS"] = pt
    a, b = tool_per_step(K_STUDY), tool_once(K_STUDY)
    globals()["P_TRANS"] = P_TRANS_SAVE
    sweep[pt] = (a, b)
    print(f"{pt:>13.1%}{u12:>10.1%}{a:>11.1%}{b:>11.1%}")

print()
print()
print("And the cost side: a per-step tool makes k round trips, a single call")
print("makes one. Accuracy per unit of latency at k=12, relative to unaided.")
print()
print(f"{'approach':>20}{'accuracy':>11}{'round trips':>14}{'per trip':>11}")
print("-" * 56)
print(f"{'unaided':>20}{u12:>11.1%}{0:>14}{'--':>11}")
print(f"{'tool per step':>20}{tps[K_STUDY]:>11.1%}{K_STUDY:>14}"
      f"{tps[K_STUDY] / K_STUDY:>11.2%}")
print(f"{'one tool call':>20}{ton[K_STUDY]:>11.1%}{1:>14}"
      f"{ton[K_STUDY]:>11.2%}")

print(f"""
The first table is the trade, and the first row is the part that gets skipped.

At k=1 the unaided model is {una[1]:.1%} accurate and the per-step tool is
{tps[1]:.1%}. **The tool loses on a one-step problem**, and not because the tool
is bad. The model executes one step correctly {P_EXEC:.0%} of the time, and it
translates-and-reads one step correctly {P_TRANS * P_PARSE:.1%} of the time.
Calling out to a calculator to add two numbers removes one way of being wrong and
introduces two.

{'The per-step tool overtakes the unaided model at k=' + str(cross[0]) + '.' if cross else 'The per-step tool NEVER overtakes the unaided model, at any length swept -- and its ratio gets steadily worse, from ' + format(tps[1] / una[1], '.2f') + 'x at k=1 to ' + format(tps[32] / una[32], '.2f') + 'x at k=32.'}
{'The single call overtakes at k=' + str(cross1[0]) + ', and reaches ' + format(ton[32] / una[32], '.2f') + 'x by k=32.' if cross1 else 'The single call never overtakes over the range swept.'}

Two tool designs, opposite verdicts, from the same tool. That is the result, and
the arithmetic behind it is worth following because it generalises.

Every approach here compounds geometrically; they differ only in the base.
Unaided accuracy is {P_EXEC}^k. A per-step tool is ({P_TRANS} x {P_PARSE})^k =
{P_TRANS * P_PARSE:.4f}^k -- a SMALLER base than {P_EXEC}, so it falls behind and
keeps falling, exponentially. A single call for the whole problem is
{P_TRANS}^k x {P_PARSE} -- the larger base {P_TRANS}, paid for with a one-off
{P_PARSE:.0%} penalty for reading one result.

**The per-step design pays the parse cost k times; the single-call design pays it
once.** At k=1 that difference is nothing and the one-off penalty dominates, so
both lose. As k grows the base wins, which is why the single call crosses over and
the per-step version never does (eq:tool-error-reallocation).

So "should I use a tool" is the wrong question and "how many times do I cross the
boundary" is the right one. Every round trip is a fresh opportunity to
mis-translate and to mis-read, and those opportunities multiply.

That is the difference between cite:gao2023pal's design and an interleaved one.
PAL's own framing is the mechanism this listing measures: models decompose
problems correctly and then make arithmetic mistakes in the solution, so hand the
solution to an interpreter and keep the decomposition. Doing that ONCE, as a
program, is what makes the arithmetic work out. Interleaving calls in the style
of cite:yao2023react buys the ability to let later steps depend on earlier
results, and this table is what that flexibility costs when you do not need it.

The second table is the sensitivity that decides whether any of this survives
contact with your own model. At k={K_STUDY} the unaided baseline is {u12:.1%}.
With translation accuracy at {0.999:.1%} the single call reaches
{sweep[0.999][1]:.1%}; at {0.97:.0%}, {sweep[0.97][1]:.1%}; at {0.95:.0%},
{sweep[0.95][1]:.1%} -- below the unaided baseline; at {0.88:.0%},
{sweep[0.88][1]:.1%}.

The crossover sits near the execution accuracy itself, which gives the summary
worth carrying: **a tool helps exactly when the model is better at calling it than
at doing the work.** That reads as a tautology and is not, because both halves are
routinely assumed rather than measured. A model with strong arithmetic and a
shaky grasp of an unfamiliar API is on the wrong side of the inequality, and the
resulting system is worse than the one that did the sums itself -- while looking
more rigorous, because there is a tool in the trace.

Note also how steep that column is. Dropping translation accuracy from
{0.99:.0%} to {0.95:.0%} costs the single call
{sweep[0.99][1] - sweep[0.95][1]:.1%} at k={K_STUDY}. Anything that degrades
translation -- an API change, an unfamiliar schema, a longer context, a
distribution shift in problem phrasing -- is amplified by the chain length, so
tool-use reliability is not a fixed property of a model but a property of a model
against a particular interface.

The third table adds latency, and it is why the single-call design usually wins
in production even where the accuracy comparison is close. At k={K_STUDY} the
per-step tool makes {K_STUDY} round trips for {tps[K_STUDY]:.1%}; the single call
makes one for {ton[K_STUDY]:.1%}. Per round trip that is
{tps[K_STUDY] / K_STUDY:.2%} against {ton[K_STUDY]:.2%}.

Round trips are serial by construction -- the next call needs the previous result
-- so they do not batch, and each one re-enters the model with a longer context.
On part:15's serving economics that is close to the most expensive shape a
request can have.

So the decision needs three measurements and no philosophy. Measure your model's
per-step execution accuracy on the task. Measure its per-step translation accuracy
against your actual tool, and its parse accuracy on your actual response format.
If translation beats execution, use the tool, by a factor that grows exponentially
in chain length -- and then cross the boundary as few times as you can, which
means one program rather than k calls unless later steps genuinely need earlier
results.""")
```

## 9. Practical Example

The first listing gives $5{,}000$ problems a pool of $128$ candidate samples. One
sample is fully correct $9\%$ of the time; otherwise it carries one of five
defects drawn with a skew, so some defects are far commoner than others.

```
 samples n   coverage    learned   complete
                        verifier      check
-------------------------------------------
         1       9.6%       9.6%       9.6%
         8      53.0%      31.2%      53.0%
        32      95.2%      53.6%      95.2%
       128     100.0%      72.3%     100.0%
```

The learned verifier goes $9.6\% \to 72.3\%$ over a $128\times$ budget. The
complete check goes $9.6\% \to 100\%$, equal to coverage at every budget. Those
are different kinds of curve: the learned verifier competes against its own false
positives, and a check has none on the property it checks
({{eq:check-turns-coverage-into-accuracy}}).

Now make the specification incomplete — the check catches the defects it knows
about and passes the rest:

```
                       specification      n=1      n=8     n=32    n=128
------------------------------------------------------------------------
                       catches all 5     9.6%    53.0%    95.2%   100.0%
  catches 4 of 5 (misses the rarest)     9.6%    45.9%    65.6%    66.3%
                      catches 3 of 5     9.6%    33.0%    36.7%    36.8%
  catches 2 of 5 (the two commonest)     9.6%    21.6%    22.0%    22.0%
```

The incomplete curves do not merely sit lower — they *stop climbing*. Catching four
of five defects gains $0.7$ points over the last four doublings while the complete
check gains $4.8$ and reaches $100\%$. **The ceiling is the specification, not the
budget** ({{eq:incomplete-check-limit}}).

And then what the system actually ships:

```
                       specification   shipped   shipped   and passed
                                        at n=1  at n=128    the check
---------------------------------------------------------------------
                       catches all 5    100.0%    100.0%       100.0%
  catches 4 of 5 (misses the rarest)     68.8%     66.3%       100.0%
                      catches 3 of 5     37.7%     36.8%       100.0%
  catches 2 of 5 (the two commonest)     22.6%     22.0%       100.0%
```

At $n=128$ every specification produces a passing sample for $100\%$ of problems.
A two-of-five specification therefore reports a green run everywhere and ships
defective work $78.0\%$ of the time.

Note what did *not* happen, because the usual telling of this overstates it.
Shipped correctness barely moved with the budget — $22.6\%$ at $n=1$ against
$22.0\%$ at $n=128$. Sampling harder did not corrupt the output. It drove the
*pass rate* to $100\%$ while correctness stayed flat, which destroys the check's
value as a signal rather than its value as a filter. Before the extra compute, a
failing check told you something; after it, the check passes on everything.

That is the quiet failure mode, and it is quieter than the one tools were brought
in to fix. A learned verifier that is wrong gives you a mediocre number. An
incomplete executable check gives you a green run.

The second listing prices the call itself. The model executes a step correctly
$96\%$ of the time, translates one into a tool call $97\%$ of the time, and reads
the result back correctly $96\%$ of the time.

```
  steps k    unaided   tool per   one tool
                           step       call
------------------------------------------
        1      96.1%      93.1%      93.1%
        3      88.7%      80.9%      87.3%
        5      81.4%      70.3%      82.2%
       12      61.5%      42.6%      66.8%
       32      27.0%      10.3%      36.2%
```

Two designs, opposite verdicts, from the same tool. The per-step tool *never*
overtakes the unaided model and gets steadily worse — $0.97\times$ at $k=1$ down to
$0.38\times$ at $k=32$. The single call crosses over at $k=5$ and reaches
$1.34\times$ by $k=32$.

{{eq:tool-error-reallocation}} explains both. Unaided accuracy is $0.96^k$. The
per-step tool is $(0.97 \times 0.96)^k = 0.9312^k$ — a *smaller* base, so it falls
behind exponentially. The single call is $0.97^k \times 0.96$ — the larger base,
paid for with a one-off penalty. **The per-step design pays the parse cost $k$
times; the single-call design pays it once.**

So the design question is not whether to use a tool but how many times to hand
control across the boundary. That is the difference between
{{cite:gao2023pal}}'s program-once approach and an interleaved one, and this table
is what interleaving costs when later steps do not actually need earlier results.

Sweeping translation accuracy at $k=12$, against an unaided baseline of $61.5\%$:

```
  translation   unaided   tool per   one tool
     accuracy                 step       call
---------------------------------------------
        99.9%     61.5%      61.1%      94.7%
        99.0%     61.5%      54.3%      85.0%
        97.0%     61.5%      42.5%      66.6%
        95.0%     61.5%      33.7%      51.6%
        88.0%     61.5%      13.1%      20.7%
```

The crossover sits near the execution accuracy itself: **a tool helps exactly when
the model is better at calling it than at doing the work.** That reads as a
tautology and is not, because both halves are routinely assumed rather than
measured. And the column is steep — dropping translation accuracy from $99\%$ to
$95\%$ costs the single call $33.4$ points at $k=12$ — so tool reliability is a
property of a model *against a particular interface*, not of the model.

Finally the latency, at $k=12$: the per-step tool makes twelve round trips for
$42.6\%$, the single call makes one for $66.8\%$. Per round trip that is $3.55\%$
against $66.80\%$.

## 10. Production Considerations

Use a check wherever one exists, and measure the curve rather than a point. If
accuracy tracks coverage as you raise the budget, your check is complete enough;
if it flattens while the pass rate rises, it is not, and
{{eq:incomplete-check-limit}} tells you what you are converging to.

Monitor the pass rate as a first-class metric alongside accuracy. A pass rate
climbing toward $100\%$ while accuracy is flat is the signature of
{{sec:9-practical-example}}'s failure, and it is invisible in accuracy alone.

Minimise boundary crossings. One program per problem rather than $k$ interleaved
calls, batched queries rather than sequential ones, and interleaving only where a
later step genuinely needs an earlier result. The accuracy argument and the latency
argument point the same way, and they compound.

Measure $p_t$ and $p_e$ before adopting a tool. Per-step translation accuracy
against your actual interface, and per-step execution accuracy on your actual
task. If $p_t \le p_e$ the tool makes things worse while making the trace look
more rigorous.

Constrain the call rather than hoping for it. Grammar-constrained decoding, a
schema the model has seen, an unambiguous response format — these raise $p_t$ and
$p_r$ directly, and {{sec:6-mathematical-foundation}} says the $p_t$ improvement
has the exponential return.

Treat the test suite as an optimisation target, not just a harness. That means
adversarial review of what it *cannot* catch, and it means resisting the
temptation to read line coverage as defect coverage — they are different
quantities and only the second appears in
{{eq:specification-is-the-ceiling}}.

## 11. Common Mistakes

**Reading a green check as a correct answer.** It is a correct answer with
probability {{eq:specification-is-the-ceiling}}, which was $22\%$ for one of
{{sec:9-practical-example}}'s specifications.

**Scaling the sample budget without auditing the specification.** Extra compute
against a complete check is pure profit and against an incomplete one is pressure
on the gap. The two are indistinguishable from the accuracy curve alone until it
flattens.

**Interleaving tool calls by default.** {{sec:9-practical-example}}'s per-step
design loses at every chain length, and it is what most agent scaffolds do.

**Adding a tool for a one-step task.** The translate-and-parse cost is paid whether
the problem needs it or not, and at $k=1$ there is no compounding advantage to
offset it.

**Assuming tool use is a fixed model capability.** $p_t$ is a property of the model
*and* the interface. A schema change can move it several points, and
{{sec:9-practical-example}} shows what several points cost at $k=12$.

**Using line coverage as a proxy for specification completeness.** It measures
which code ran, not which defects would be caught.

## 12. Failure Modes

*The green run.* An incomplete specification with a large sample budget passes
everything and ships defects at the rate {{eq:specification-is-the-ceiling}}
allows. The monitoring that would catch it is the pass rate, which almost nobody
alerts on because a high pass rate looks like success.

*Confidently wrong tool output.* A mis-translated call executes correctly and
returns an authoritative number computed from the wrong inputs. This is harder to
spot than an arithmetic slip because the trace contains a real computation.

*Correct execution of a wrong decomposition.* The interpreter runs the model's
program faithfully; the program answers a different question. {{cite:gao2023pal}}'s
split leaves this entirely with the model.

*Latency blowup from interleaving.* $k$ serial round trips with a growing context
is close to the worst request shape in {{part:15}}'s cost model, and it degrades
non-linearly under load.

*Specification gaming during training.* When the check is the reward
({{cite:deepseek2025r1}}), the generator is optimised to find the specification's
gaps rather than merely to exploit them opportunistically.

## 13. Alternatives

**A learned verifier.** {{ch:rsn-supervision}}'s subject, and the fallback where no
checker exists. Its ceiling is $q$, and {{sec:9-practical-example}} measures the
distance between $q < 1$ and $q = 1$ as the difference between a curve that
flattens at $72\%$ and one that reaches $100\%$.

**Self-consistency.** {{ch:rsn-self-consistency}}: cheaper than a tool, needs no
interface, and bounded by the generator's mode. Where a checker exists it is
strictly dominated; where none does, it is often the best available selector.

**Constrained decoding.** Rather than checking after generation, make the invalid
output unrepresentable ({{part:8}}). This raises $p_t$ toward 1 for the structural
part of a call and is the cheapest available improvement to tool reliability.

**Formal verification.** The complete-specification case taken seriously: a proof
assistant makes $\sum_{b \notin \mathcal{D}} \pi_b$ genuinely zero for the
specified property. The cost is writing the specification, which is the work the
rest of this chapter says you were doing implicitly anyway.

**Training the reasoning in.** {{cite:deepseek2025r1}} uses the checker during
training rather than at inference, converting a per-request cost into a fixed one.
The trade favours it at high request volume, with {{sec:7-internal-mechanics}}'s
caveat about the specification becoming an optimisation target.

## 14. Evaluation

Report accuracy *and* pass rate as a pair across the sample budget. Their
divergence is the specification diagnosis and neither one alone shows it.

Fit accuracy against coverage. If they coincide, your check is complete enough for
the current error distribution; the residual is
{{eq:specification-is-the-ceiling}}'s precision term and it is directly
measurable.

Measure $p_e$, $p_t$ and $p_r$ separately. Three numbers, each cheap, and together
they predict the whole of {{eq:tool-error-reallocation}} including the crossover
you should expect.

Count boundary crossings per request and report accuracy per crossing, not just
per request. {{sec:9-practical-example}}'s twentyfold difference at $k=12$ is
invisible in end-to-end accuracy.

And hold out defect *types*, not just examples. A specification evaluated on the
defects it was written against will always look complete; the useful evaluation
introduces a defect class the tests were not designed for and measures whether the
pass rate notices.

## 15. Advanced Concepts

**Specification synthesis.** If the specification is the ceiling, generating it is
the highest-leverage thing a model could do — write the tests, then write the code
that passes them. The obvious hazard is that a model writing both is
{{ch:rsn-self-consistency}}'s correlated critic in a new costume, and the tests
will be blind to exactly the defects the implementation contains.
{{maturity:EMERGING}}.

**Partial checks as a gradient.** A check need not be binary. Property-based
testing, fuzzing, and runtime assertions each cover a slice of
$\sum_b \pi_b$, and composing several with independent blind spots is the practical
route to raising the ceiling — the same decorrelation argument as
{{eq:recoverable-mass}}, applied to specifications instead of critics.

**Verification-generation gaps.** The entire test-time-compute programme assumes
checking is easier than producing. Where a checker exists this is definitional;
where one does not, the gap is an empirical property of the task and largely
unmeasured. {{maturity:RESEARCH FRONTIER}}, and the question underneath all of
{{part:16}}.

**Tool use as a trained capability.** $p_t$ can be raised by training rather than
prompting, and the exponential return in {{sec:6-mathematical-foundation}} makes
that a better investment than it looks. The complication is that it is
interface-specific, so it must be redone when the interface changes.

## 16. Connection to Previous Chapters

{{ch:rsn-test-time-compute}} introduced the verifier-quality parameter and showed
its value growing with the sample budget. This chapter is the limiting case, and
the interesting result is that $q = 1$ changes the curve's *shape* rather than
lifting it — coverage becomes accuracy, and coverage is buyable.

{{ch:rsn-cot}}'s compounding result is the entire second listing: three geometric
decays differing only in base, with the tool decision reduced to which base is
larger.

{{ch:rsn-supervision}}'s lucky-chain rate recurs as
{{eq:specification-is-the-ceiling}}'s undetected-defect mass. A right answer from
wrong reasoning and a wrong answer that passes the tests are the same object, and
both cap what a selector can deliver regardless of budget.

{{ch:rsn-self-consistency}}'s correlated-critic problem returns in
{{sec:15-advanced-concepts}}: a model that writes both the tests and the code is
its own critic again.

Ahead: {{ch:rsn-benchmarks}} closes {{part:16}} by asking what any of these
numbers mean, with the specification-versus-correctness distinction from this
chapter directly applicable to benchmark scores.

## 17. Exercises

1. In the first listing, change the defect prior to be uniform rather than skewed
   and re-run the incomplete-specification table. Explain why the ordering of the
   rows changes.

2. Derive $k^{*}$ from {{eq:tool-error-reallocation}} and check it against the
   measured crossover in the second listing. Where does the prediction sit
   relative to the observed $k = 5$?

3. Add a third design to the second listing: one call every $m$ steps. Sweep $m$
   and find the optimum at $k = 32$.

4. Make the check *unsound* as well as incomplete — it occasionally fails a correct
   sample — and measure how that differs from incompleteness. Which is worse, and
   why?

5. Implement the composition idea from {{sec:15-advanced-concepts}}: two checks
   with different blind spots. Measure the ceiling as a function of the overlap
   between what they catch.

6. Take a real test suite you own, estimate $\sum_{b \in \mathcal{D}} \pi_b$ by
   introducing defects of several kinds, and compare it with the line-coverage
   number the suite reports.

## 18. Interview Questions

1. Why does an executable check change the *shape* of the accuracy-versus-compute
   curve rather than just its level?

2. Your agent passes its tests on every task and ships bugs. What do you measure
   first?

3. When does adding a calculator make a model worse at arithmetic?

4. Why does one program beat twelve tool calls, given that the per-step
   translation accuracy is the same?

5. A colleague proposes scaling the sample budget tenfold now that you have a test
   suite. What would you want to know first?

6. What is the difference between a test suite as an evaluation harness and as an
   optimisation target?

## 19. Research Questions

1. Can the undetected-defect mass $\sum_{b \notin \mathcal{D}} \pi_b$ be estimated
   without knowing the defects — from disagreement between independently written
   checks, or from the rate at which sampling finds passing-but-different
   solutions?

2. Is there a training signal for specification *completeness* rather than for
   passing a given specification?

3. How large is the verification-generation gap on tasks with no checker, and can
   it be measured without one?

4. Does a model that writes its own tests exhibit measurable blind-spot
   correlation with the code it writes, and can that correlation be reduced by
   architectural separation?

5. Tool-use reliability is interface-specific. Is there an interface design that
   generalises — a schema style or calling convention that a model transfers to
   unseen tools with high $p_t$?

## 20. Chapter Summary

An executable check makes the verifier exact, and the consequence is a change of
shape rather than of level. A learned verifier took a system from $9.6\%$ to
$72.3\%$ across a $128\times$ budget while flattening; a complete check took it to
$100\%$, equal to coverage at every point ({{eq:check-turns-coverage-into-accuracy}}).
The binding constraint moves off verification and onto the sample budget, which is
the one resource in {{part:16}} you can buy.

The constraint that replaces it is the specification. An incomplete check has a
precision that does not depend on $n$ ({{eq:specification-is-the-ceiling}}), so
extra compute converges to a ceiling the specification sets: $66\%$, $37\%$ and
$22\%$ for checks catching four, three and two of five defect types. The failure
is quiet — shipped correctness stayed flat while the pass rate went to $100\%$, so
the system reports a green run on every problem while shipping defects $78\%$ of
the time. **A tool moves the binding constraint from verification to
specification**, which is a large improvement and not the same as removal.

The tool call itself is not free. It trades execution error for translation and
parsing error ({{eq:tool-error-reallocation}}), and the arithmetic is a comparison
of geometric bases. A per-step tool at $(0.97 \times 0.96)^k$ lost to an unaided
model at $0.96^k$ at *every* chain length tested, degrading from $0.97\times$ to
$0.38\times$. The same tool called once per problem, at $0.97^k \times 0.96$,
crossed over at $k=5$ and reached $1.34\times$ by $k=32$ — because the parse cost
is paid once per boundary crossing while the translation advantage compounds per
step.

So the decision needs three measurements: per-step execution accuracy, per-step
translation accuracy against your actual interface, and parse accuracy on your
actual response format. If translation beats execution the tool wins by a margin
exponential in chain length; then cross the boundary as few times as possible. And
whatever the check says, watch the pass rate.

## 21. Further Reading

{{cite:gao2023pal}} is the chapter's spine and its opening observation is the
whole argument: models decompose correctly and then make arithmetic mistakes, so
keep the decomposition and delegate the solving.

{{cite:yao2023react}} is the interleaved design, and worth reading against
{{sec:9-practical-example}}'s second listing — it is the right architecture when
later steps depend on earlier results and an expensive one when they do not.

{{cite:sprague2024tocot}} is the reason this chapter exists: if chain-of-thought's
gains are concentrated where a solver does the job better, the solver is the
answer.

{{cite:brown2024monkeys}} for the coverage curve that a complete check converts
directly into accuracy, and {{cite:deepseek2025r1}} for what happens when the
checker becomes the training signal rather than the inference-time filter.
