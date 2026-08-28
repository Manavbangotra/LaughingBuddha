---
id: rsn-test-time-compute
number: 148
part: XVI
tier: full
status: draft
requires: [per-step-error-compounding, serial-computation-budget,
           decode-is-bandwidth]
provides: [coverage-selection-decomposition, coverage-log-linear,
           verifier-quality-ceiling, systematic-versus-random-error,
           marginal-value-of-samples, adaptive-versus-fixed-allocation,
           early-stopping-allocation]
citations: [brown2024monkeys, snell2024testtime, cobbe2021gsm8k,
            lightman2023verify, wang2023selfconsistency, yao2023tot,
            muennighoff2025s1, deepseek2025r1]
---

## 1. Learning Objectives

By the end of this chapter you will be able to decompose any test-time compute
result into a coverage claim and a selection claim, and say which one a given
paper is actually reporting; predict when majority voting will work and when it
will do nothing, from the shape of the model's errors rather than their rate;
compute the marginal value of one more sample and use it to allocate a fixed
budget; explain why an adaptive policy can beat an allocation that knows every
problem's difficulty in advance; and identify the one component that both
converts samples into answers and tells you when to stop drawing them.

## 2. Why This Matters

{{ch:rsn-cot}} established that intermediate tokens buy serial computation, and
that a chain's accuracy is multiplicative in its length. Both halves point the
same direction: you can spend compute at inference time and get something back.
This chapter is about what you get back, and it opens with a distinction that
most reporting of these results collapses.

When a paper says that sampling 250 times takes a system from $15.9\%$ to $56\%$
— which is what {{cite:brown2024monkeys}} measured on SWE-bench Lite — it is
making a claim about the *generator*: that somewhere in those 250 samples, a
correct one exists. It is not a claim that the system answers correctly $56\%$ of
the time, because answering requires choosing which sample to keep. Those are two
different quantities, they are bounded by two different things, and money spent
improving one does nothing for the other.

The distinction matters because the two have opposite cost curves. Coverage is
cheap and scales predictably: more samples, more coverage, roughly log-linearly
over a wide range. Selection is expensive, does not scale with sampling at all,
and is the binding constraint in nearly every deployed system.
{{sec:9-practical-example}} measures a case where the sample budget rises by a
factor of 256, coverage rises from $9.4\%$ to $89.2\%$, and the delivered accuracy
of the obvious selector *falls slightly*.

There is a second reason this chapter comes before the ones on voting and
verification rather than after. The most common way to pick an answer from a pool
of samples is to take the most frequent one, and the most common belief about it
is that it works because errors cancel. That belief is testable, and the test has
a sharp answer: it works when errors are unsystematic and does nothing when they
are systematic, at identical accuracy on an identical task. Which means the
question "should I sample more?" cannot be answered from a benchmark number. It
depends on the *shape* of your model's errors, and {{sec:9-practical-example}}
shows how to find out.

Finally, this is where the economics of {{part:15}} come due. Test-time compute
is decode-phase compute — memory-bound, poorly batched, billed as output tokens.
A 256× sample budget is a 256× bill on the most expensive part of serving. It is
worth knowing precisely what it buys before signing up for it.

## 3. Prerequisites

You need {{ch:rsn-cot}}'s compounding result, because per-step accuracy is what
sets the per-sample success rate this whole chapter reasons about, and its
serial-computation account, because "more test-time compute" splits into two
genuinely different things — longer chains and more chains — that this chapter
keeps separate.

From {{part:15}} you need the decode phase's cost structure: that generated
tokens are memory-bound and that $n$ parallel samples are $n$ times the decode
work but can share prefill and batch well, which is why parallel sampling is
cheaper per token than its serial equivalent.

Basic probability is assumed: independent trials, the geometric distribution, and
what it means for an estimator to be consistent. That last one carries real weight
in {{sec:5-formal-explanation}}, because the interesting failure of majority
voting is that it is a *perfectly consistent* estimator of the wrong quantity.

## 4. Intuitive Explanation

Imagine handing a problem to a hundred people who are each individually mediocre
at it and collecting their answers. Two questions decide what you can do with that
pile, and they are entirely separate.

First: is the right answer in the pile at all? Call that coverage. It improves
with the number of people, because each one is another chance. It improves in a
very specific way: if each person is right with probability $p$ independently,
the chance that nobody is right is $(1-p)^n$, which falls off fast. This is why
sampling more feels so powerful — you go from one weak attempt to near-certainty
that *somebody* got it.

Second: can you tell which one is right? Call that selection. Nothing about
collecting more answers helps with this. If you had no way to recognise a correct
answer at $n=1$, you have no way to recognise one at $n=100$; you just have more
things to fail to recognise.

That second question is where the entire difficulty lives, and the reason it is
easy to overlook is that the standard workaround looks like it solves it. Take the
most common answer. This feels principled — it is a vote, errors should cancel, the
truth should be the one thing everyone converges on.

Here is where the intuition needs sharpening, because whether that works depends
on something the vote itself cannot see.

Suppose the hundred people are each *randomly* wrong: when they err they err in
different directions, so their wrong answers scatter across many possibilities
while the correct answer is the one thing they agree on. Then the vote works, and
it works spectacularly — even if only $10\%$ of them are right, that $10\%$ is a
bigger bloc than any individual wrong answer, and it grows more decisive with $n$.

Now suppose instead that they all share a misconception. They were taught the same
wrong rule, or the problem contains the same trap for everyone. Now their wrong
answers are not scattered — they pile onto one specific wrong answer, which is the
majority. Voting does not merely fail here; it fails *more confidently* as you add
people, because you are getting a better and better estimate of what the group
believes, and the group is wrong.

That is the whole story of majority voting, and it means the question is never
"how accurate is my model" but "what does my model's error distribution look
like". Two models with identical accuracy can be at opposite ends of this,
which is exactly what {{sec:9-practical-example}} measures: same task, same
$9.4\%$ single-sample accuracy, and at 256 samples the vote delivers $99.6\%$ for
one and $8.8\%$ for the other.

The alternative to voting is having something that can actually check an answer —
a verifier. A test suite, a proof checker, a trained reward model, a second model
asked to find the flaw. With a perfect verifier, selection becomes free: you only
need a correct answer to *exist*, and coverage is exactly your accuracy. With an
imperfect one you get some fraction of the way there, and that fraction — not your
sample budget — is what sets your ceiling.

This reframes the spending decision entirely. Doubling your samples moves coverage
by a fixed number of points per doubling. Improving your verifier moves accuracy
by however much coverage you were failing to cash in. And crucially, those two
investments *multiply*: a better verifier is worth more when you have more samples
(there is more to recover), and more samples are worth more when you have a better
verifier (you can actually use them). Scaling either alone buys half a mechanism.

The second half of the chapter is about how to spread a fixed budget across many
problems, and it has one counterintuitive result worth previewing. The obvious
policy — spend more on the hard problems — loses to spending uniformly. Not
because hard problems do not need more samples, but because a sample aimed at a
problem you will never solve is as wasted as a sample aimed at one you solved on
the first try. The value of a sample is highest in the middle, and the biggest
single win is not giving hard problems more but giving easy problems *less*.

## 5. Formal Explanation

Fix a problem and let $p$ be the probability that one independent sample from the
generator is correct. Coverage at $n$ samples is the probability that at least one
is correct:

$$C(n) = 1 - (1 - p)^{n}$$ (eq:coverage-grows)

Across a population of problems with success rates $p_i$, measured coverage is the
average of {{eq:coverage-grows}}. This is the quantity {{cite:brown2024monkeys}}
reports rising log-linearly, and the log-linearity is not a deep fact about
language models: $1 - (1-p)^n$ plotted against $\log n$ is close to linear over
the middle of its range for any $p$, and a mixture of such curves across a
heterogeneous population extends that middle range considerably.

Delivered accuracy is coverage passed through a selector $\sigma$ that maps a pool
of $n$ samples to one answer:

$$A(n) = \Pr\big[\sigma(y_1, \ldots, y_n) = y^{*}\big] \;\le\; C(n)$$ (eq:coverage-selection-gap)

The inequality is the whole chapter. No selector can return a correct answer that
is not in the pool, so coverage is a hard ceiling on accuracy, and the gap between
them is a property of the selector alone. **Two systems with identical generators
can deliver wildly different accuracy, and two systems with identical accuracy can
have wildly different headroom.**

Take the majority-vote selector. As $n \to \infty$ the empirical frequency of each
answer converges to its true probability under the generator, so:

$$\sigma_{\text{maj}}(y_1, \ldots, y_n) \;\xrightarrow{\;n \to \infty\;}\; \arg\max_{y} \Pr[\,Y = y\,]$$ (eq:vote-converges-to-mode)

The vote is a *consistent estimator of the generator's modal answer*. Whether that
is useful depends entirely on whether the mode is correct, which is a property of
the error distribution and not of the accuracy. Its limiting accuracy is exactly
the fraction of problems whose modal answer is right, and adding samples past the
point where the mode is well estimated changes nothing.

This gives the precise condition. Write $p$ for the probability of the correct
answer and $q_{\max}$ for the largest probability assigned to any single incorrect
answer. Majority voting succeeds in the limit iff:

$$p > q_{\max}$$ (eq:vote-condition)

Note what is absent from {{eq:vote-condition}}: $n$, and the *overall* error rate.
A generator can be right $10\%$ of the time and vote perfectly if its errors spread
over fifty possibilities at $2\%$ each. A generator can be right $40\%$ of the time
and vote at zero if a single wrong answer takes $60\%$. This is the formal content
of "errors must be unsystematic", and it is testable directly by estimating
$q_{\max}$ from a large pool.

For a verifier-based selector, parameterise quality as the probability $q$ that
the verifier identifies a correct sample when the pool contains one:

$$A_{\text{ver}}(n) \approx q \cdot C(n) + (1 - q)\cdot(\text{chance})$$ (eq:verifier-caps-selection)

At $q = 1$ accuracy equals coverage exactly. The practical consequence is in the
derivative: $\partial A / \partial q = C(n)$, so **the value of improving the
verifier is proportional to coverage** and therefore grows with the sample budget.
Verifier quality and sample count are complements, not substitutes.

Now allocation. Given $M$ problems with rates $p_i$ and a total budget $B$, choose
$n_i$ to maximise $\sum_i \big(1 - (1-p_i)^{n_i}\big)$ subject to $\sum_i n_i = B$.
The marginal value of the $(n+1)$th sample on a problem is:

$$\frac{\partial}{\partial n}\Big[1 - (1-p)^n\Big] \;\propto\; (1-p)^{n}\, p$$ (eq:marginal-value-of-a-sample)

which is the key object. It vanishes as $p \to 1$ (the problem is already solved)
*and* as $p \to 0$ (the sample will not land either), and it is maximised at
$p \approx 1/(n+1)$. Since {{eq:marginal-value-of-a-sample}} is decreasing in $n$
for every problem, the objective is concave and the optimum equalises marginal
value across problems — a water-filling solution obtained by finding the $\lambda$
with $\sum_i n_i(\lambda) = B$ where $(1-p_i)^{n_i} p_i = \lambda$.

The important reading of {{eq:marginal-value-of-a-sample}} is that "difficult" and
"worth sampling" are different properties. They coincide only in a middle band,
and a policy that funds the hardest problems is buying the least recoverable part
of the distribution.

## 6. Mathematical Foundation

Two extensions of {{eq:coverage-grows}} are worth having explicitly, because they
are where the simple model stops describing real systems.

**Heterogeneity dominates the shape.** For a single $p$, coverage is a sigmoid in
$\log n$ with a narrow transition. Real populations mix easy and hard problems,
and averaging {{eq:coverage-grows}} over a distribution of $p$ smears the
transition across many more doublings — which is why measured coverage curves look
log-linear over three or four orders of magnitude when no single problem's curve
does. The log-linearity is a statement about the *population*, and reading a
per-problem mechanism out of it is a mistake.

**Independence is the load-bearing assumption, and it fails in the useful
direction and the harmful one.** Samples from one model at one temperature are not
independent; they share every parameter. Write the effective sample count
$n_{\text{eff}} < n$ and {{eq:coverage-grows}} becomes $1 - (1-p)^{n_{\text{eff}}}$,
which is why coverage curves flatten earlier than the independent model predicts.
More sharply: if a problem is one the model is *systematically* wrong about, then
$p \approx 0$ for every sample and no budget reaches it. Systematic error does not
merely defeat the selector; it caps coverage itself, which
{{sec:9-practical-example}} measures at $89.2\%$ versus $100\%$ for two generators
of identical accuracy.

Now the adaptive case, which is where the chapter's most useful result lives. The
allocation in {{eq:marginal-value-of-a-sample}} is *non-adaptive*: it commits
$n_i$ before observing anything. An adaptive policy observes outcomes and
reallocates, and with a verifier the natural one is to stop sampling a problem the
moment a sample is verified correct. Its expected consumption on a problem is:

$$\mathbb{E}[\text{samples used}] = \frac{1 - (1-p)^{n_{\max}}}{p} \;\le\; \min\!\Big(\tfrac{1}{p},\, n_{\max}\Big)$$ (eq:early-stopping-cost)

For $p = 0.83$ that is about $1.2$ samples where a fixed policy might spend 16.
The freed budget flows automatically to problems still failing, with no difficulty
model at all.

An adaptive policy can therefore beat the *non-adaptive optimum*, and this is not
a paradox. The oracle allocation is optimal within the class of policies that must
commit in advance; early stopping is outside that class, because it uses
information — the observed outcomes — that the oracle is not permitted to use.
{{sec:9-practical-example}} measures the gap at $+2.4$ points in favour of the
adaptive policy over an oracle that knows every $p_i$ exactly.

The catch is the same one as everywhere else in this chapter.
{{eq:early-stopping-cost}} requires knowing that a sample is correct, which is a
verifier. The verifier converts samples into answers
({{eq:verifier-caps-selection}}) and it tells you when to stop drawing them
({{eq:early-stopping-cost}}). Both halves of test-time compute reduce to it.

## 7. Internal Mechanics

Test-time compute is not one resource. It is at least three, and they have
different cost structures and different ceilings.

**Longer chains** buy serial depth, at the rate {{ch:rsn-cot}} described: each
token is another pass through the weights. Cost is linear in tokens, entirely in
the decode phase, and it cannot be parallelised — token $t+1$ requires token $t$.
The ceiling is {{ch:rsn-cot}}'s {{eq:length-tradeoff}}: past a point, compounding
error dominates the added depth.

**More samples** buy coverage. Cost is linear in samples, shares prefill across
the batch, and parallelises perfectly, which makes it far cheaper per token than
the serial equivalent on real hardware ({{part:15}}). The ceiling is
{{eq:coverage-selection-gap}}: coverage without selection is not accuracy.

**Search** buys both, at the cost of a scoring call per node. Tree-of-thoughts
methods ({{cite:yao2023tot}}) expand alternatives at each step and prune, which
recovers from a bad prefix instead of committing to it, and beam or lookahead
search against a process reward model ({{cite:snell2024testtime}}) is the same
idea with a learned score. The ceiling is the scorer's quality, which is
{{eq:verifier-caps-selection}} applied per step instead of per sample.

### 7.1 Why the three do not substitute for each other

A team that has already saturated one and starts spending on another usually
discovers this the hard way, so it is worth being explicit about which failure
each one addresses.

Longer chains address *insufficient serial depth*. If the model cannot fit the
computation into the tokens it is producing, more samples will not help — every
sample will hit the same wall.

More samples address *variance*. If the model can sometimes do the problem and
sometimes cannot, coverage grows. If the model is deterministically wrong about
the problem, sampling is buying nothing, and {{sec:9-practical-example}} measures
this as a hard ceiling on coverage itself.

Search addresses *irrecoverable prefixes*. If most failures come from an early
wrong step that the rest of the chain then elaborates faithfully — which is
exactly {{ch:rsn-cot}}'s compounding failure mode — then the ability to back out
of that step is worth more than either of the others.

Diagnosing which one you have is a measurement, not a guess: sample a pool, and
look at whether the failures are the same failure.

### 7.2 The serving shape

Parallel samples share prefill. For a long prompt with a short answer, $n$ samples
cost roughly one prefill plus $n$ decodes, and the decodes batch well because they
are the same shape — which is the regime where continuous batching earns its keep
({{part:15}}). This makes parallel sampling substantially cheaper than the naive
$n\times$ estimate.

Longer chains do not share anything. Each additional token is a full decode step,
memory-bound, and it grows the KV cache that every subsequent step must read.
A 10× longer trace is a 10× decode bill *and* a growing per-token cost.

So the cost-effectiveness ordering usually favours parallel sampling, and the
accuracy ordering usually favours whichever failure mode you actually have. These
two orderings frequently disagree, which is why the choice deserves a measurement
rather than a default.

### 7.3 What a verifier actually is

The word covers a range that spans four orders of magnitude in reliability, and
conflating them is the most common source of disappointment here.

At the reliable end are *executable* checks: a test suite, a compiler, a proof
assistant, a constraint solver. These give $q$ near 1 on the property they check,
which is why coding and formal mathematics are where test-time compute has
produced its most dramatic results. {{cite:brown2024monkeys}}'s SWE-bench numbers
are coverage numbers precisely because the benchmark ships tests.

In the middle are *trained* verifiers: outcome reward models
({{cite:cobbe2021gsm8k}}) and process reward models
({{cite:lightman2023verify}}), which score a solution or its steps. Their $q$ is
empirical, task-specific, and — as {{ch:rsn-supervision}} takes up — depends
heavily on whether they were trained on outcomes or on steps.

At the weak end is *the model itself*, asked whether its answer is right. This is
the cheapest option and it is close to worthless as a selector, for the reason
{{ch:rsn-cot}} gave: the self-assessment and the answer are outputs of the same
system trained on overlapping signals, so the assessment is correlated with the
answer's fluency rather than its correctness. {{ch:rsn-self-consistency}} measures
this directly.

## 8. Implementation

Two listings. The first decomposes accuracy into coverage and selection and
measures what each is bounded by, on a task where the distribution of wrong
answers is produced by the generator's own dynamics rather than specified by hand.
The second spends a fixed budget across a population of problems seven different
ways.

```python {tier=A name=coverage-and-selection}
"""Sampling more does not mean answering better: coverage versus selection.

cite:brown2024monkeys measured coverage -- the fraction of problems solved by AT
LEAST ONE of n samples -- rising log-linearly in n, from 15.9% at one sample to
56% at 250 on SWE-bench Lite. That is a real and large effect, and it is a
statement about the GENERATOR.

Turning it into accuracy requires picking which sample to keep, which is a
statement about the SELECTOR. This listing separates the two on a task where the
distribution of wrong answers is produced by the actual dynamics rather than
specified by hand (eq:coverage-selection-gap).

The task is ch:rsn-cot's: iterate a fixed permutation for k steps. The generator
is a stepper with two kinds of error, because real generators have two kinds.
On most states it is right with probability p and slips at random. On a minority
of CONFUSED states it applies a consistently wrong rule -- a misconception rather
than noise -- which is what makes some wrong answers far more likely than others.
The distribution of wrong answers is then produced by the task's own dynamics.
"""
import numpy as np
from collections import Counter

rng = np.random.default_rng(419)

N = 48
PERM = rng.permutation(N)
WRONG = rng.permutation(N)          # the consistently-wrong rule
CONFUSED = set(rng.choice(N, size=N // 3, replace=False).tolist())
K = 6                       # steps per chain
P_STEP = 0.93               # per-step accuracy on states it is not confused by
P_CONF = 0.12               # per-step accuracy on states it IS confused by
N_PROB = 3000               # problems
MAXS = 256                  # samples per problem


def truth(x, k=K):
    for _ in range(k):
        x = PERM[x]
    return x


def sample_systematic(x):
    """On a confused state the generator usually applies its own wrong rule, so
    many samples land on the SAME wrong answer."""
    for _ in range(K):
        if x in CONFUSED:
            x = PERM[x] if rng.random() < P_CONF else int(WRONG[x])
        else:
            x = PERM[x] if rng.random() < P_STEP else int(rng.integers(N))
    return int(x)


def sample_unsystematic(x, p):
    """The control. Same task, same chain length, errors go somewhere at
    random -- so wrong answers spread instead of piling up."""
    for _ in range(K):
        x = PERM[x] if rng.random() < p else int(rng.integers(N))
    return int(x)


starts = rng.integers(N, size=N_PROB)
answers = np.array([truth(int(s)) for s in starts])

pool = np.array([[sample_systematic(int(s)) for _ in range(MAXS)]
                 for s in starts])
base_acc = float(np.mean(pool[:, 0] == answers))

# Match the control's single-sample accuracy to the systematic generator's, so
# the comparison is about the SHAPE of the errors and not their rate.
lo, hi = 0.3, 0.999
for _ in range(28):
    mid = (lo + hi) / 2
    a = float(np.mean([sample_unsystematic(int(s), mid) == answers[i]
                       for i, s in enumerate(starts)]))
    if a < base_acc:
        lo = mid
    else:
        hi = mid
P_FLAT = (lo + hi) / 2
pool_flat = np.array([[sample_unsystematic(int(s), P_FLAT) for _ in range(MAXS)]
                      for s in starts])


def majority(row):
    return Counter(row.tolist()).most_common(1)[0][0]


def verifier_pick(row, correct, q):
    """A verifier of quality q. With probability q it recognises a correct
    sample when the pool contains one; otherwise it picks uniformly at random.
    q = 1 is the oracle, q = 0 is single-sample-equivalent selection."""
    good = np.flatnonzero(row == correct)
    if len(good) and rng.random() < q:
        return correct
    return int(row[rng.integers(len(row))])


BUDGETS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
QS = [0.5, 0.8, 1.0]

print(f"{N_PROB} problems, {K} steps each on {N} states. The generator is")
print(f"{P_STEP:.0%} accurate per step on ordinary states and {P_CONF:.0%} on the")
print(f"{len(CONFUSED)} states it is confused by, where its errors are SYSTEMATIC.")
print()
print(f"A control generator makes the SAME single-sample accuracy "
      f"({base_acc:.1%}) with")
print(f"errors that go somewhere at random (per-step {P_FLAT:.1%}), so only the")
print("SHAPE of the errors differs between the two generators.")
print()
print(f"{'':>10}{'SYSTEMATIC errors':>25}{'RANDOM errors':>25}")
print(f"{'samples n':>10}{'coverage':>13}{'majority':>12}"
      f"{'coverage':>13}{'majority':>12}")
print("-" * 60)

cov, maj, covf, majf = {}, {}, {}, {}
for n in BUDGETS:
    sub, subf = pool[:, :n], pool_flat[:, :n]
    cov[n] = float(np.mean([(sub[i] == answers[i]).any() for i in range(N_PROB)]))
    maj[n] = float(np.mean([majority(sub[i]) == answers[i] for i in range(N_PROB)]))
    covf[n] = float(np.mean([(subf[i] == answers[i]).any() for i in range(N_PROB)]))
    majf[n] = float(np.mean([majority(subf[i]) == answers[i] for i in range(N_PROB)]))
    print(f"{n:>10}{cov[n]:>13.1%}{maj[n]:>12.1%}{covf[n]:>13.1%}{majf[n]:>12.1%}")

print()
print()
print("Selection on the SYSTEMATIC pool, by verifier quality. q is the chance")
print("the verifier finds a correct sample when the pool contains one.")
print()
print(f"{'samples n':>10}{'coverage':>11}"
      + "".join(f"{'verifier q=' + str(q):>16}" for q in QS))
print("-" * 69)
ver = {q: {} for q in QS}
for n in BUDGETS:
    sub = pool[:, :n]
    row = f"{n:>10}{cov[n]:>11.1%}"
    for q in QS:
        v = float(np.mean([verifier_pick(sub[i], answers[i], q) == answers[i]
                           for i in range(N_PROB)]))
        ver[q][n] = v
        row += f"{v:>16.1%}"
    print(row)

print()
print()
print("Is coverage log-linear in n, as cite:brown2024monkeys reports? Fit")
print("coverage against log2(n) and look at the residuals.")
print()
xs = np.log2(np.array(BUDGETS, float))
ys = np.array([cov[n] for n in BUDGETS])
A = np.stack([xs, np.ones_like(xs)], 1)
slope, icpt = np.linalg.lstsq(A, ys, rcond=None)[0]
print(f"{'samples n':>10}{'coverage':>11}{'log-linear fit':>17}{'residual':>11}")
print("-" * 49)
for n, y in zip(BUDGETS, ys):
    f = slope * np.log2(n) + icpt
    print(f"{n:>10}{y:>11.1%}{f:>17.1%}{y - f:>+11.1%}")
print()
print(f"  fitted slope: {slope:+.3f} coverage per doubling of n")

print()
print()
print("Where does the majority vote's ceiling come from? For each problem, is")
print("the correct answer the MODAL one in a large pool?")
print()
modal = np.array([majority(pool[i]) for i in range(N_PROB)])
modal_right = float(np.mean(modal == answers))
present = float(np.mean([(pool[i] == answers[i]).any() for i in range(N_PROB)]))
share = np.array([float(np.mean(pool[i] == answers[i])) for i in range(N_PROB)])
modal_share = np.array([float(np.mean(pool[i] == modal[i])) for i in range(N_PROB)])
lost = (modal != answers) & (share > 0)
print(f"{'quantity':>46}{'value':>10}")
print("-" * 56)
print(f"{'a correct sample exists somewhere in the pool':>46}{present:>10.1%}")
print(f"{'the correct answer is the modal one':>46}{modal_right:>10.1%}")
print(f"{'correct answer present but OUTVOTED':>46}{float(np.mean(lost)):>10.1%}")
print()
print(f"{'on those outvoted problems:':>46}")
print(f"{'mean share of samples that are correct':>46}"
      f"{float(share[lost].mean()):>10.1%}")
print(f"{'mean share held by the winning wrong answer':>46}"
      f"{float(modal_share[lost].mean()):>10.1%}")

c1, c256 = cov[1], cov[256]
m1, m256 = maj[1], maj[256]
print(f"""
The first table is the decomposition, and it separates two things that a single
accuracy number fuses.

Coverage is what the GENERATOR can do: the fraction of problems where at least
one sample is right. For the systematic generator it goes from {c1:.1%} at one
sample to {c256:.1%} at 256. That is the effect cite:brown2024monkeys reports,
and the log-linear fit below has residuals under five points across two orders of
magnitude.

The majority vote is one particular SELECTOR, and it goes from {m1:.1%} to
{m256:.1%}. It does not climb slowly. It does not climb.

The control generator is why that is a finding rather than a rigged demo. It has
the same single-sample accuracy, {base_acc:.1%}, on the same task with the same
chain length; the search that set its per-step rate to {P_FLAT:.1%} had matching
that number as its only objective. The one difference is that its errors go
somewhere at random instead of piling onto one wrong answer. Its majority vote
goes from {majf[1]:.1%} to {majf[256]:.1%}.

Identical accuracy, identical task, and the vote is worth
{majf[256] - majf[1]:+.1%} in one case and {m256 - m1:+.1%} in the other.
**Majority voting is not a method for turning samples into accuracy. It is a
method for cancelling UNSYSTEMATIC error, and it has no purchase on the
systematic kind.**

There is a second difference in that table which was not part of the plan, and it
is worth more than the one that was. The two coverage columns are not the same
either. The random-error generator reaches {covf[64]:.1%} coverage by n=64 and
saturates; the systematic one is still at {cov[64]:.1%} and reaches only
{c256:.1%} at 256.

So systematic error does not merely defeat the selector. It caps what sampling
can reach at all, because on a problem the generator is reliably confused about,
every sample fails the same way and there is nothing in the pool to select. A
perfect verifier cannot fix that, and neither can a larger budget. That is a
ceiling on the whole test-time-compute strategy, and it is set by the shape of
the model's errors rather than by anything you can buy.

The modal-answer table says exactly where the vote's failure comes from.

A correct sample is present for {present:.1%} of problems, and the correct answer
is modal for {modal_right:.1%}. The gap -- {float(np.mean(lost)):.1%} of problems
-- is answers that are in the pool and outvoted. On those problems the correct
answer holds {float(share[lost].mean()):.1%} of the samples and the winning wrong
answer holds {float(modal_share[lost].mean()):.1%}. It is not close, and it does
not get closer.

That is the ceiling stated exactly. The vote estimates which answer the generator
produces MOST OFTEN, and it estimates it consistently: more samples make it more
confident. Where the mode is wrong, more samples make it more confidently wrong,
and its limit as n goes to infinity is precisely the fraction of problems whose
mode happens to be correct -- {modal_right:.1%} here.

The verifier table is the same statement from the other side, and it is the
constructive half.

A perfect verifier (q=1.0) turns coverage into accuracy exactly: {ver[1.0][256]:.1%}
at n=256, equal to coverage, because all a perfect verifier needs is for a correct
sample to EXIST. At q=0.8 the same pool yields {ver[0.8][256]:.1%} and at q=0.5,
{ver[0.5][256]:.1%}.

So the gap between coverage and delivered accuracy IS the selector's quality
(eq:coverage-selection-gap), which makes it a budget question with a clear answer.
Doubling the sample budget moves coverage {slope:+.1%}. Moving the verifier from
q=0.5 to q=1.0 moves accuracy {ver[1.0][256] - ver[0.5][256]:+.1%} at n=256 --
and only {ver[1.0][8] - ver[0.5][8]:+.1%} at n=8.

Note the direction of that last comparison, because it is the practical
consequence and it runs against the usual intuition that a better verifier is
worth most when you are sampling least. A verifier can only cash in coverage that
exists. At n=8 there is little to recover; at n=256 there is a great deal.
Sampling and verification are complements, and scaling one without the other buys
half a mechanism -- which is ch:rsn-supervision's argument for why the interesting
work is in the verifier.""")
```

The second listing moves from one problem to a population, and asks how to spread
a fixed budget across problems of unequal difficulty.

```python {tier=A name=budget-allocation}
"""A fixed sampling budget, spent seven ways.

cite:snell2024testtime's compute-optimal result says the best way to spend
test-time compute depends on the prompt's difficulty, and that allocating by
difficulty beats a uniform budget substantially. This listing works out what that
means when you cannot see the difficulty and have to estimate it
(eq:marginal-value-of-a-sample), and then what happens when you stop trying to
predict difficulty and simply react to outcomes.

The setup is deliberately favourable to allocation: a population of problems with
widely varying per-sample success rates, so there is a great deal to gain from
spending unevenly.
"""
import numpy as np

rng = np.random.default_rng(523)

M = 4000                  # problems
BUDGET_PER = 16           # mean samples per problem
TOTAL = M * BUDGET_PER

# Per-sample success rate. A realistic spread: many easy, a solid middle, and a
# tail that is effectively hopeless at any budget you can afford.
p = np.concatenate([
    rng.beta(9.0, 1.5, size=M // 3),        # easy
    rng.beta(1.6, 4.0, size=M // 3),        # middle
    rng.beta(0.35, 14.0, size=M - 2 * (M // 3)),   # nearly hopeless
])
rng.shuffle(p)


def solved(alloc, ptrue=p):
    """Expected fraction solved: a problem counts if at least one of its
    allocated samples succeeds."""
    return float(np.mean(1.0 - (1.0 - ptrue) ** alloc))


def marginal(n, ptrue):
    """Increase in P(solved) from one more sample: (1-p)^n * p."""
    return (1.0 - ptrue) ** n * ptrue


def allocate(ptrue, total=TOTAL, cap=4096):
    """Optimal allocation for a concave objective, by water-filling.

    The marginal value of the (n+1)th sample is (1-p)^n * p, which is decreasing
    in n, so the optimum equalises marginal value across problems. Solve
    (1-p)^n * p = lam for n and binary-search lam to hit the budget."""
    ptrue = np.clip(ptrue, 1e-9, 1 - 1e-9)
    lo, hi = 1e-12, float(ptrue.max())

    def alloc_for(lam):
        n = np.log(lam / ptrue) / np.log1p(-ptrue)
        return np.clip(np.round(n), 0, cap).astype(np.int64)

    for _ in range(80):
        mid = (lo + hi) / 2
        if alloc_for(mid).sum() > total:
            lo = mid          # lam too low -> allocating too much
        else:
            hi = mid
    a = alloc_for(hi)
    # Spend any rounding slack on the highest remaining marginal values.
    slack = total - int(a.sum())
    if slack > 0:
        idx = np.argsort(-marginal(a, ptrue))[:slack]
        a[idx] += 1
    return a


def pilot_estimate(ptrue, pilot):
    """What you actually get to see: successes out of `pilot` real samples.
    Laplace-smoothed so a zero-success pilot is not treated as p = 0."""
    s = rng.binomial(pilot, ptrue)
    return (s + 1.0) / (pilot + 2.0)


print(f"{M} problems, a total budget of {TOTAL:,} samples "
      f"({BUDGET_PER} per problem on average).")
print("Per-sample success rates span easy, middle and near-hopeless.")
print()
print(f"{'difficulty band':>22}{'count':>8}{'mean p':>9}"
      f"{'solved at n=16':>16}")
print("-" * 55)
bands = [("p > 0.5 (easy)", p > 0.5),
         ("0.05 < p < 0.5", (p > 0.05) & (p <= 0.5)),
         ("p < 0.05 (hard)", p <= 0.05)]
for name, m in bands:
    print(f"{name:>22}{int(m.sum()):>8}{float(p[m].mean()):>9.3f}"
          f"{solved(np.full(int(m.sum()), BUDGET_PER), p[m]):>16.1%}")

print()
print()
print("Seven ways to spend the same total budget.")
print()

uniform = np.full(M, BUDGET_PER, dtype=np.int64)
oracle = allocate(p)

pilots = {}
for pilot in (2, 4, 8):
    est = pilot_estimate(p, pilot)
    alloc = allocate(est, total=TOTAL - pilot * M)
    # The pilot samples are real attempts and count toward solving.
    pilots[pilot] = alloc + pilot

# Two tempting heuristics, both spending the same total.
order = np.argsort(p)                      # ascending difficulty: hardest first
hardest = np.full(M, BUDGET_PER // 2, dtype=np.int64)
extra = TOTAL - int(hardest.sum())
hardest[order[:M // 4]] += extra // (M // 4)

easiest = np.full(M, BUDGET_PER // 2, dtype=np.int64)
extra = TOTAL - int(easiest.sum())
easiest[order[-(M // 4):]] += extra // (M // 4)


def with_early_stopping(ptrue, total=TOTAL, cap=4096):
    """Sample round-robin over the problems that are still unsolved, and stop
    spending on a problem the moment one of its samples succeeds.

    This needs a VERIFIER -- you cannot stop early unless you can tell that you
    are done -- but it needs no difficulty estimate at all."""
    n = len(ptrue)
    alive = np.ones(n, dtype=bool)
    used = np.zeros(n, dtype=np.int64)
    spent = 0
    while spent < total and alive.any():
        k = int(alive.sum())
        if spent + k > total:
            idx = np.flatnonzero(alive)[: total - spent]
            hit = rng.random(len(idx)) < ptrue[idx]
            used[idx] += 1
            alive[idx[hit]] = False
            break
        idx = np.flatnonzero(alive)
        hit = rng.random(k) < ptrue[idx]
        used[idx] += 1
        alive[idx[hit]] = False
        spent += k
        if used.max() >= cap:
            break
    return float(np.mean(~alive)), used


stop_solved, stop_used = with_early_stopping(p)

print(f"{'strategy':>34}{'solved':>10}{'vs uniform':>13}")
print("-" * 57)
strategies = [
    ("uniform (16 each)", uniform),
    ("oracle allocation (knows p)", oracle),
    ("pilot of 2, then allocate", pilots[2]),
    ("pilot of 4, then allocate", pilots[4]),
    ("pilot of 8, then allocate", pilots[8]),
    ("all spare budget to hardest 25%", hardest),
    ("all spare budget to easiest 25%", easiest),
]
base = solved(uniform)
res = {}
for name, alloc in strategies:
    v = solved(alloc)
    res[name] = v
    print(f"{name:>34}{v:>10.1%}{v - base:>+13.1%}")
res["early stopping (needs a verifier)"] = stop_solved
print(f"{'early stopping (needs a verifier)':>34}{stop_solved:>10.1%}"
      f"{stop_solved - base:>+13.1%}")

print()
print()
print("Where does the budget go? Mean samples per problem by band.")
print()
print(f"{'difficulty band':>22}{'uniform':>10}{'oracle':>10}"
      f"{'pilot of 4':>13}{'early stop':>13}")
print("-" * 68)
for name, m in bands:
    print(f"{name:>22}{BUDGET_PER:>10.1f}{float(oracle[m].mean()):>10.1f}"
          f"{float(pilots[4][m].mean()):>13.1f}"
          f"{float(stop_used[m].mean()):>13.1f}")

o = res["oracle allocation (knows p)"]
p4 = res["pilot of 4, then allocate"]
hd = res["all spare budget to hardest 25%"]
ez = res["all spare budget to easiest 25%"]
easy_m, mid_m, hard_m = bands[0][1], bands[1][1], bands[2][1]
print(f"""
The oracle row is the best a FIXED allocation can do: {o:.1%} against uniform's
{base:.1%}, a gain of {o - base:+.1%} from spending the same {TOTAL:,} samples
differently, with perfect knowledge of every problem's success rate. That is
cite:snell2024testtime's compute-optimal effect in its simplest possible form,
and it is an upper bound only for policies that must commit before seeing any
outcomes.

The allocation table says where the gain comes from, and it is not where the
phrase "spend more on hard problems" points.

The oracle does give the hard band the most -- {float(oracle[hard_m].mean()):.1f}
samples against the middle band's {float(oracle[mid_m].mean()):.1f}. But look at
the easy band: {float(oracle[easy_m].mean()):.1f} samples, down from 16. Those
problems succeed at {float(p[easy_m].mean()):.0%} per sample and are essentially
all solved by the third attempt; the other thirteen samples were buying nothing.
**The gain is almost entirely in not over-sampling the easy problems**, and the
hard band is merely where the freed budget lands.

That distinction is what separates the optimal policy from the heuristic that
sounds like it. Giving all the spare budget to the hardest quarter scores
{hd:.1%}, which is {hd - base:+.1%} against uniform -- it LOSES, because it funds
the hard problems out of the middle band as well as the easy one, and because a
quarter of the problems chosen for being hardest includes the ones that are
hopeless at any budget. Giving it to the easiest quarter is worse still at
{ez:.1%} ({ez - base:+.1%}), which at least fails in the direction you would
expect.

eq:marginal-value-of-a-sample explains both. The value of one more sample is
(1-p)^n * p, which vanishes as p approaches 1 (already solved) and as p
approaches 0 (the sample will not land either). At n={BUDGET_PER} it is maximised
near p = 1/(n+1) = {1/(BUDGET_PER+1):.3f}. Neither end of the difficulty range is
where the money is, and "hard" and "worth sampling" are different properties that
happen to overlap in the middle.

The pilot rows are the practical question, because difficulty is not observable
and estimating it costs samples out of the same budget.

A pilot of 4 gets {p4 - base:+.1%}, which is {(p4 - base) / (o - base):.0%} of the
oracle's {o - base:+.1%}. That is real and it is also a minority of what is
available. Pilots of 2 and 8 give {res['pilot of 2, then allocate'] - base:+.1%}
and {res['pilot of 8, then allocate'] - base:+.1%}, so measuring harder makes
things worse past a point: the pilot spends budget to learn something it then has
less budget to act on. The optimum is interior and shallow, and a third of the
oracle gain is roughly what this approach is worth.

Then there is the last row, which is the one to actually build.

Early stopping scores {stop_solved:.1%}, {stop_solved - base:+.1%} against
uniform. It beats every pilot strategy, and it beats the oracle by
{stop_solved - o:+.1%} while estimating nothing at all.

That is not a bug in the oracle, and the reason is the most useful thing in this
listing. The oracle knows every p and commits its whole allocation up front. Early
stopping knows nothing about p and gets to see OUTCOMES, so it can stop a problem
after one lucky sample and keep feeding one that has failed forty times. An
adaptive policy is allowed to beat the best non-adaptive one, because it is
optimising against information the non-adaptive policy is not permitted to use --
and here it does, by {stop_solved - o:.1%}.

Look at what it does to the bands without being told anything. The easy band
drops to {float(stop_used[easy_m].mean()):.1f} samples on average, the middle to
{float(stop_used[mid_m].mean()):.1f}, and the hard band rises to
{float(stop_used[hard_m].mean()):.1f} -- a more aggressive version of the oracle's
own allocation, arrived at with no model of difficulty whatsoever. A problem that
succeeds immediately consumes one sample whatever its nominal difficulty; a
problem that keeps failing keeps drawing budget until it succeeds or the budget
runs out. The information the pilot spent {4 * M / TOTAL:.0%} of the budget to
buy arrives free as a by-product of doing the work.

The catch is in the parenthesis. Early stopping requires knowing that a sample is
correct, which is a VERIFIER -- the same component whose quality set the ceiling
in the previous listing. So the two halves of this chapter reduce to one
recommendation: the verifier is what converts a sampling budget into answers, and
it is also what tells you when to stop spending. Without it you are choosing
between a uniform budget and a difficulty estimate that costs a third of what it
recovers.""")
```

## 9. Practical Example

The first listing iterates a permutation on 48 states for six steps. The generator
is $93\%$ accurate per step on ordinary states and $12\%$ on the sixteen states it
is *confused* by, where its errors follow a consistently wrong rule. A control
generator was tuned by binary search to the same single-sample accuracy of
$9.4\%$ — per-step rate $64.6\%$ — with errors that go somewhere at random.

```
                  SYSTEMATIC errors            RANDOM errors
 samples n     coverage    majority     coverage    majority
------------------------------------------------------------
         1         9.4%        9.4%         9.9%        9.9%
         2        14.8%        9.4%        18.3%        9.9%
         4        21.6%        9.9%        32.8%       11.3%
         8        30.9%        9.5%        54.8%       17.9%
        16        43.8%        9.0%        79.4%       27.9%
        32        57.2%        8.8%        95.9%       46.0%
        64        69.7%        8.8%        99.8%       71.8%
       128        81.1%        8.8%       100.0%       93.0%
       256        89.2%        8.8%       100.0%       99.6%
```

Coverage on the systematic pool rises from $9.4\%$ to $89.2\%$ — the effect
{{cite:brown2024monkeys}} reports, on a task with entirely different structure —
and a fit against $\log_2 n$ gives $+10.7\%$ per doubling with residuals under
$5.7$ points across two orders of magnitude.

The majority vote goes from $9.4\%$ to $8.8\%$. It does not climb slowly; it does
not climb.

The control column is what makes that a finding rather than a rigged
demonstration. Identical accuracy, identical task, identical chain length; the
only difference is the *shape* of the errors, and the vote is worth $+89.7$ points
for one generator and $-0.6$ for the other. **Majority voting is a method for
cancelling unsystematic error, not a method for turning samples into accuracy.**

The coverage columns carry a second difference that was not part of the design and
matters more than the one that was. The random-error generator saturates at
$100\%$ coverage by $n=128$; the systematic one reaches only $89.2\%$ at $n=256$.
Systematic error does not merely defeat the selector — it caps what sampling can
reach at all, because on a problem the generator is reliably confused about, every
sample fails the same way and there is nothing in the pool to select. No verifier
and no budget recovers that.

The modal-answer breakdown localises the vote's failure exactly:

```
                                      quantity     value
--------------------------------------------------------
 a correct sample exists somewhere in the pool     89.2%
           the correct answer is the modal one      8.8%
           correct answer present but OUTVOTED     80.4%

                   on those outvoted problems:
        mean share of samples that are correct      4.5%
   mean share held by the winning wrong answer     59.8%
```

On $80.4\%$ of problems the right answer is in the pool and loses the vote, at
$4.5\%$ against $59.8\%$. That is not close and it does not get closer:
{{eq:vote-converges-to-mode}} says the vote converges to the generator's mode, so
more samples estimate a wrong answer with greater confidence.

The verifier table is the constructive half:

```
 samples n   coverage  verifier q=0.5  verifier q=0.8  verifier q=1.0
---------------------------------------------------------------------
         1       9.4%            9.4%            9.4%            9.4%
         8      30.9%           19.2%           26.9%           30.9%
        32      57.2%           34.1%           46.9%           57.2%
       128      81.1%           44.6%           68.0%           81.1%
       256      89.2%           49.5%           72.5%           89.2%
```

A perfect verifier turns coverage into accuracy exactly. At $q=0.8$ the same pool
yields $72.5\%$, at $q=0.5$ it yields $49.5\%$. So doubling the sample budget is
worth $+10.7$ points of coverage, while moving the verifier from $q=0.5$ to
$q=1.0$ is worth $+39.7$ points at $n=256$ — and only $+11.7$ at $n=8$.

That last comparison runs against the usual intuition, which is that a good
verifier matters most when you can only afford a few samples. The opposite is
true: a verifier can only cash in coverage that exists, so its value scales with
the budget ({{eq:verifier-caps-selection}}). Sampling and verification are
complements.

The second listing spreads $64{,}000$ samples across $4{,}000$ problems whose
per-sample success rates span three bands: $1{,}507$ easy ($\bar{p}=0.833$),
$1{,}277$ middle ($\bar{p}=0.230$), and $1{,}216$ nearly hopeless
($\bar{p}=0.012$).

```
                          strategy    solved   vs uniform
---------------------------------------------------------
                 uniform (16 each)     71.4%        +0.0%
       oracle allocation (knows p)     78.2%        +6.8%
         pilot of 2, then allocate     73.4%        +2.0%
         pilot of 4, then allocate     73.6%        +2.2%
         pilot of 8, then allocate     73.1%        +1.7%
   all spare budget to hardest 25%     68.8%        -2.6%
   all spare budget to easiest 25%     65.2%        -6.2%
 early stopping (needs a verifier)     80.6%        +9.2%
```

The oracle — perfect knowledge of every $p_i$, committed in advance — is worth
$+6.8$ points. Where that comes from is the surprise:

```
       difficulty band   uniform    oracle   pilot of 4   early stop
--------------------------------------------------------------------
        p > 0.5 (easy)      16.0       2.9          8.5          1.2
        0.05 < p < 0.5      16.0      18.9         17.5          6.5
       p < 0.05 (hard)      16.0      29.3         23.7         44.2
```

The oracle does give the hard band the most. But the movement that generates the
gain is the easy band falling from $16$ samples to $2.9$: those problems are
essentially all solved by the third attempt, and the other thirteen samples were
buying nothing. **The gain is in not over-sampling the easy problems**; the hard
band is merely where the freed budget lands.

Which is why the heuristic that sounds identical *loses*. Giving all spare budget
to the hardest quarter scores $68.8\%$, $2.6$ points below uniform, because it
funds hard problems out of the middle band too and because the hardest quarter
contains problems that are hopeless at any budget.
{{eq:marginal-value-of-a-sample}} is maximised near $p = 1/(n+1) = 0.059$ at
$n=16$: neither end of the difficulty range is where the money is.

The pilot rows are the honest practical answer. A pilot of $4$ recovers $+2.2$
points, or $32\%$ of the oracle's gain, and a pilot of $8$ recovers *less* —
measurement competes with the budget it is measuring for.

And then early stopping, at $+9.2$ points, beating an oracle that knows every
$p_i$ by $2.4$. That is not a bug: the oracle is the optimum among policies that
must commit before observing anything, and early stopping is outside that class.
It uses outcomes. Without any difficulty model it drives the easy band to $1.2$
samples and the hard band to $44.2$ — a more aggressive version of the oracle's own
allocation, discovered online.

The catch is in the parenthesis. Early stopping requires knowing that a sample is
correct, which is a verifier — the same component that set the ceiling in the
first listing. Both halves of this chapter reduce to it.

## 10. Production Considerations

Instrument coverage separately from accuracy. Almost no production system does
this, and it is the difference between knowing you have a generator problem and
knowing you have a selector problem. If you have any way to check correctness
offline, run $n=8$ or $n=16$ on a sample of traffic and record both numbers. The
gap tells you where to spend.

Build the verifier first if you are building anything. It sets the ceiling on
selection, it enables early stopping, and its value grows with every sample you
add. A sampling budget without a verifier is a coverage machine attached to a coin
flip.

Prefer parallel samples to longer chains when you can. They share prefill, batch
cleanly, and parallelise, so $n$ samples cost much less than $n\times$ on real
serving hardware ({{part:15}}), whereas a chain $n$ times longer is $n$ times the
memory-bound decode plus a growing KV cache.

Use early stopping wherever correctness is checkable — code with tests, structured
output with a schema, arithmetic with a checker. It is the highest-return policy
in {{sec:9-practical-example}} and it requires no difficulty model, no router and
no tuning.

Cap per-problem spend explicitly. Early stopping directs budget toward problems
that keep failing, which is right up to the point where a problem is unsolvable
and absorbs everything. A cap converts an unbounded tail into a bounded one.

Watch the tail latency. Adaptive policies have highly variable per-request cost by
construction. If you are serving interactively, the p99 is set by your cap, not by
your mean.

## 11. Common Mistakes

**Reading a coverage number as an accuracy number.** pass@$k$, "solved by at least
one sample", and "with an oracle verifier" are all coverage. They describe the
generator, and the delivered number is whatever your selector recovers from them.

**Assuming majority voting will help.** It helps when
{{eq:vote-condition}} holds — $p > q_{\max}$ — and does nothing when it does not,
regardless of accuracy. Estimate $q_{\max}$ from a large pool before you build on
it.

**Spending the budget on the hardest problems.** {{sec:9-practical-example}}
measures this at $2.6$ points *worse* than uniform. The marginal value of a sample
is near zero at both ends of the difficulty range.

**Scaling samples without scaling the verifier.** The verifier's value grows with
coverage, so a fixed mediocre verifier converts an ever-larger fraction of your
spend into nothing.

**Treating the model's self-assessment as a verifier.** It is generated by the
same system that produced the answer and correlates with fluency rather than
correctness ({{ch:rsn-cot}}).

**Building a difficulty router before trying early stopping.** The router costs a
model, a training set, and a maintenance burden, and in
{{sec:9-practical-example}} it recovers a third of what the parameter-free
adaptive policy does.

**Forgetting that samples are correlated.** $n$ samples from one model at one
temperature are worth substantially less than $n$ independent attempts, which is
why coverage curves flatten earlier than {{eq:coverage-grows}} predicts.

## 12. Failure Modes

*Confidently wrong consensus.* Systematic error concentrates the vote on one wrong
answer, and the vote's margin grows with $n$. The system reports high agreement,
which is usually logged as a confidence signal, and it is precisely backwards.

*Coverage ceilings that no budget clears.* Problems the model is deterministically
wrong about contribute nothing to coverage at any $n$
({{sec:9-practical-example}}: $89.2\%$ versus $100\%$). Sampling curves that
plateau below $100\%$ are diagnosing this, and the fix is a better generator, not
a bigger budget.

*Verifier gaming.* Sampling $n$ times and selecting by a learned verifier is an
optimisation against the verifier, so it finds the verifier's errors — a search
over $256$ candidates for the one that scores highest is exactly the procedure you
would design to break it. The effect grows with $n$, which means the verifier gets
*less* trustworthy as the budget you deployed it for increases.

*Budget exhaustion on unsolvable problems.* Adaptive allocation concentrates spend
on repeated failures. Without a cap, a small number of impossible problems can
consume a large share of a shared budget.

*Latency blowup from serial search.* Tree search is not embarrassingly parallel —
each expansion depends on the previous scoring — so its wall-clock cost is much
worse than its token count suggests.

## 13. Alternatives

**Improve the generator instead.** Every result in this chapter is bounded by the
per-sample success rate. Fine-tuning that raises $p$ improves coverage at every
budget, moves the optimal chain length out ({{ch:rsn-cot}}), and costs nothing at
inference. Where it is available it dominates.

**Self-consistency.** Sampling plus majority voting, which
{{cite:wang2023selfconsistency}} established as the standard baseline.
{{ch:rsn-self-consistency}} takes it seriously — including the conditions under
which it works, which {{sec:9-practical-example}} has already bounded.

**Search over steps.** {{cite:yao2023tot}} and {{cite:snell2024testtime}}'s
beam-search-against-a-PRM: spend the budget on branching rather than on
independent restarts. This is the right choice when failures come from
irrecoverable prefixes rather than from variance.

**Budget forcing.** {{cite:muennighoff2025s1}} showed that appending a
continuation token to make a model keep thinking — or cutting it off — reproduces
much of the test-time scaling curve, which makes chain length a serving parameter
rather than something the model decides.

**Train the reasoning in.** {{cite:deepseek2025r1}} moves the spend from inference
to training. The trade is a fixed cost against a per-request one, and it wins
whenever request volume is high — which is the usual case in production.

## 14. Evaluation

Report coverage and accuracy as two numbers, always. A single accuracy figure at
$n=1$ tells you nothing about headroom, and a pass@$k$ figure tells you nothing
about what the system delivers.

Measure $q_{\max}$ — the largest share held by any single wrong answer — on a large
pool. It is the one number that predicts whether voting will work, it is cheap to
compute, and {{eq:vote-condition}} makes it directly actionable.

Sweep the budget on a log scale and fit. Coverage should be close to log-linear;
where it flattens early you are seeing correlation between samples, and where it
plateaus below $100\%$ you are seeing systematic error.

Evaluate the verifier separately from the system, as a ranking problem: given a
pool containing at least one correct sample, how often does it select one? That is
$q$ in {{eq:verifier-caps-selection}}, and it is the quantity that determines what
your sampling budget is worth.

Report cost with accuracy. A result at $n=256$ and a result at $n=1$ are not
comparable, and test-time compute papers that omit the budget are reporting
coverage under another name.

## 15. Advanced Concepts

**Sequential versus parallel scaling.** {{cite:snell2024testtime}}'s finer result
is that the better use of a budget depends on the prompt: on easier problems,
revising sequentially beats sampling in parallel; on harder ones, the reverse.
This is {{eq:marginal-value-of-a-sample}} with two different generators, and it is
why "how should I spend test-time compute" has no budget-independent answer.

**The verifier-gaming limit.** As $n$ grows, selecting the argmax of an imperfect
verifier converges to the verifier's *maximiser* rather than to the correct
answer — the same structure as {{eq:vote-converges-to-mode}}, with the verifier's
score in place of the generator's frequency. Both selectors are consistent
estimators of something that is not correctness, which suggests the general
principle: any selector that maximises a proxy over a growing candidate pool
eventually optimises the gap between the proxy and the truth.

**Correlation-aware budgeting.** If $n$ samples behave like $n_{\text{eff}}$
independent ones, the right budget question is how to raise $n_{\text{eff}}$ —
temperature, prompt variation, multiple models — rather than how to raise $n$.
Measuring $n_{\text{eff}}$ by fitting {{eq:coverage-grows}} to an observed coverage
curve is cheap and rarely done.

**Where the frontier is.** Learned adaptive stopping — a model that predicts when
further sampling is not worth it, rather than a verifier that confirms success —
would extend early stopping to tasks with no checkable answer. That is
{{maturity:EMERGING}} and it is the piece that would make this chapter's best
result available outside code and mathematics.

## 16. Connection to Previous Chapters

{{ch:rsn-cot}} gave the per-step accuracy that becomes this chapter's $p$, and its
compounding result is why $p$ is usually small enough for coverage to have room to
grow. Its faithfulness result is why {{sec:7-internal-mechanics}} rates the model's
own self-assessment as the weakest verifier.

{{ch:rsn-vs-generation}}'s warning about single-distribution measurement recurs
here in a new form: a single accuracy number fuses a generator property and a
selector property, and improving the wrong one is invisible until you separate
them.

{{part:15}} supplies the cost model that makes parallel sampling the cheaper
option, and the decode-phase economics that make long chains the expensive one.

Ahead: {{ch:rsn-self-consistency}} is the deep treatment of the voting selector
this chapter bounded, {{ch:rsn-supervision}} is about building the verifier this
chapter kept pointing at, and {{ch:rsn-tool-assisted}} is what happens when the
verifier is an interpreter and $q$ goes to 1.

## 17. Exercises

1. Vary `P_CONF` in the first listing from $0.12$ up to $0.93$ and plot the
   majority vote at $n=256$ against it. Where is the transition, and does it match
   {{eq:vote-condition}}?

2. Add a verifier that is *systematically* wrong on the same states the generator
   is confused by — a correlated verifier. Measure how selection degrades, and
   explain why this is the realistic case for a verifier trained on the same data.

3. Fit {{eq:coverage-grows}} to the systematic generator's coverage curve and solve
   for the effective $p$. Compare it with the measured single-sample accuracy. What
   does the difference tell you?

4. In the second listing, replace the pilot's Laplace smoothing with a plain
   frequency estimate and explain the resulting allocation catastrophe in one
   sentence.

5. Add a per-problem cap to early stopping and sweep it. Find the cap that
   maximises problems solved, and explain why an unbounded cap is not optimal.

6. Build a version of early stopping with an *imperfect* verifier that stops on a
   false positive with probability $1-q$. At what $q$ does it stop beating uniform
   allocation?

## 18. Interview Questions

1. A paper reports pass@100 of $80\%$. What have you learned about the deployed
   system's accuracy?

2. Two models have identical accuracy on your benchmark. One benefits enormously
   from self-consistency and the other not at all. What differs, and how would you
   measure it in advance?

3. You have a fixed inference budget. Do you spend it on more samples or longer
   chains? What measurement decides?

4. Why does a better verifier become *more* valuable as you increase the sample
   budget?

5. Explain how an allocation policy that knows nothing about problem difficulty
   can outperform one that knows every problem's difficulty exactly.

6. Your team wants to add a difficulty-prediction model to route reasoning effort.
   What would you try first, and what would you need to measure to justify the
   router?

## 19. Research Questions

1. Can $n_{\text{eff}}$ — the effective number of independent samples — be raised
   cheaply? Temperature and prompt paraphrase both help a little; what would help a
   lot without running multiple models?

2. Is there a selector that does not degrade under its own optimisation pressure,
   given that both voting and verifier-argmax converge to a proxy's maximiser as
   $n$ grows?

3. What predicts, from a model alone, whether its errors on a task are systematic
   or unsystematic? This determines whether sampling is worth anything, and it is
   currently answerable only by sampling.

4. Can adaptive stopping be learned for tasks with no checkable answer, and what
   signal would it use that is not the model's own confidence?

5. {{cite:snell2024testtime}} finds sequential revision better on easy problems and
   parallel sampling better on hard ones. Is there a single quantity, measurable
   before generating, that predicts which side of that boundary a prompt is on?

## 20. Chapter Summary

Accuracy from a pool of samples factors into two independent quantities. Coverage
is what the generator can reach — $1 - (1-p)^n$, log-linear over a wide range,
$9.4\%$ to $89.2\%$ across a 256× budget in {{sec:9-practical-example}}. Selection
is what a selector recovers from it, and {{eq:coverage-selection-gap}} makes
coverage a hard ceiling that no selector can exceed.

Majority voting is a consistent estimator of the generator's *mode*
({{eq:vote-converges-to-mode}}), which is useful only when the mode is correct.
Two generators with identical $9.4\%$ single-sample accuracy on the same task gave
a vote of $99.6\%$ and $8.8\%$ at $n=256$; the difference is entirely the shape of
the errors, and {{eq:vote-condition}} states the criterion as $p > q_{\max}$.
Systematic error also caps coverage itself — $89.2\%$ against $100\%$ — so it
defeats both halves at once.

A verifier of quality $q$ recovers $q$ of the available coverage, which makes its
value proportional to the budget: moving from $q=0.5$ to $q=1.0$ was worth $+11.7$
points at $n=8$ and $+39.7$ at $n=256$. Sampling and verification are
complements.

Across problems, the marginal value of a sample is $(1-p)^n p$
({{eq:marginal-value-of-a-sample}}), which vanishes at both ends of the difficulty
range and peaks near $p = 1/(n+1)$. The optimal fixed allocation was worth $+6.8$
points and got there mainly by cutting easy problems from 16 samples to $2.9$;
the intuitive "spend on the hardest" heuristic *lost* $2.6$ points. Estimating
difficulty with a pilot recovered only about a third of the available gain.

The policy that won was early stopping: sample until verified correct, then stop.
It scored $+9.2$ points, beating an oracle that knows every success rate by $2.4$,
because it uses outcomes rather than predictions — and it needs no difficulty model
at all. Its one requirement is a verifier, which is also what converts coverage
into accuracy. Both halves of test-time compute reduce to the same component, and
{{ch:rsn-supervision}} is about how to build it.

## 21. Further Reading

{{cite:brown2024monkeys}} is the coverage paper, and it is worth reading with this
chapter's decomposition in hand: its central curve is a generator measurement, and
its most interesting sections are the ones about what happens when you have to
select without a test suite.

{{cite:snell2024testtime}} is the allocation paper. Read it for the
difficulty-dependence result and for the comparison against scaling parameters
instead, which is the trade this chapter's economics are really about.

{{cite:cobbe2021gsm8k}} introduced the verifier-plus-sampling recipe and is the
origin of the numbers everything here is built on.
{{cite:lightman2023verify}} is its successor and belongs to
{{ch:rsn-supervision}}.

{{cite:wang2023selfconsistency}} is the self-consistency paper, best read
immediately before {{ch:rsn-self-consistency}}.

{{cite:muennighoff2025s1}} is short and worth it for the budget-forcing idea
alone: chain length as an inference-time control rather than a model property.
