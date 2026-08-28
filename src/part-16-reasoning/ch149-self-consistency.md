---
id: rsn-self-consistency
number: 149
part: XVI
tier: full
status: draft
requires: [coverage-selection-decomposition, systematic-versus-random-error,
           per-step-error-compounding]
provides: [path-marginalisation, self-consistency-temperature,
           weighted-self-consistency, verifier-argmax-gaming,
           critic-error-correlation, reflection-is-voting,
           critic-operating-point]
citations: [wang2023selfconsistency, huang2024selfcorrect, madaan2023selfrefine,
            cobbe2021gsm8k, lightman2023verify, brown2024monkeys,
            snell2024testtime, yao2023tot]
---

## 1. Learning Objectives

By the end of this chapter you will be able to state what self-consistency
actually computes and why it differs from greedy decoding; set the sampling
temperature for aggregation rather than for a single sample, and know what
signal tells you when it is too high; choose between counting votes, taking a
verifier's argmax, and weighting votes by verifier score, from a measured
property of your verifier; explain why picking the highest-scored sample gets
*worse* as the pool grows; and predict whether a reflection loop will help by
measuring one thing about your critic that is not its accuracy.

## 2. Why This Matters

{{ch:rsn-test-time-compute}} bounded two things and left a gap between them.
Coverage is what sampling produces; selection is what you recover from it. This
chapter is about the two selectors that practitioners actually reach for — voting
and self-criticism — and about a result that connects them more tightly than
either literature suggests.

The first is uncontroversial and widely deployed.
{{cite:wang2023selfconsistency}} sample several reasoning paths and keep the most
frequent answer, and it works. What is less widely understood is *why* it works,
and the answer is narrow enough to be actionable: greedy decoding follows the
single most likely reasoning path, while voting returns the answer with the most
total probability behind it. Those differ whenever the correct answer is
reachable by several paths and the most likely single path is not one of them.
That is an argmax-versus-sum distinction and nothing more — no extra thinking, no
error correction, no second look.

The consequence people miss follows immediately. Self-consistency does not sit on
top of your decoding configuration; it *changes what the right configuration is*.
At temperature zero the pool is one answer repeated and voting is exactly greedy
decoding. {{sec:9-practical-example}} measures a system whose single-sample
accuracy is best at temperature 0 and whose vote is best at 1.5, where
single-sample accuracy has fallen by twelve points.

The second technique is reflection, and its literature contains a genuine
disagreement worth resolving. {{cite:madaan2023selfrefine}} report roughly $20\%$
absolute average improvement from a single model acting as generator, critic and
reviser across seven tasks. {{cite:huang2024selfcorrect}} report that intrinsic
self-correction does not improve reasoning and often degrades it, while
correction with external feedback does help. Both are careful papers.

The resolution is not that one of them is wrong, and it is not about how good the
critic is. {{sec:9-practical-example}} builds two critics with *identical
confusion matrices* — the same rate of accepting correct answers, the same rate of
catching wrong ones — and gets different outcomes, because one of them is wrong
about the same problems the solver is wrong about. And it measures where the
self-critic fails: on problems whose modal answer is wrong, it accepts a correct
proposal $10.6\%$ of the time. Where it matters, it is not merely unhelpful. It is
inverted.

That leads to the chapter's most useful result, which is that reflection against
your own judgement converges to your own modal answer — the same quantity voting
returns, obtained sequentially instead of in one parallel batch. The measured
numbers are $36.5\%$ and $36.8\%$.

## 3. Prerequisites

You need {{ch:rsn-test-time-compute}}'s decomposition, and specifically the result
that a majority vote converges to the generator's mode. This chapter measures what
that mode is worth, when it beats greedy decoding, and what else can be done with
the same pool.

You need its distinction between systematic and unsystematic error, which returns
here as a property of the *critic* rather than of the generator, and turns out to
be the variable that decides whether reflection is worth anything.

From {{ch:rsn-cot}}, the compounding result gives the per-sample success rates
these aggregation rules operate on, and the faithfulness result is the reason
{{sec:7-internal-mechanics}} treats a model's judgement of its own answer as a
special case rather than as an ordinary verifier.

Sampling and temperature from {{part:8}} are assumed: what a softmax temperature
does to a distribution, and what happens at its limits.

## 4. Intuitive Explanation

Start with the thing self-consistency is competing against, because the
comparison is the whole explanation.

Greedy decoding, at each step, takes the most likely next token. Over a whole
reasoning chain this approximates following the single most probable *path*
through the model's distribution. It is one route, chosen by being locally
best at every fork.

Now suppose there are several different ways to reach the right answer. Three
different valid approaches to a problem, say, each one moderately likely. And
suppose there is one appealing wrong approach that is individually more likely
than any of the three. Greedy decoding takes the wrong one, because it is the
single most probable path — and it will take it every time, confidently.

But the three correct routes, added together, carry more probability mass than
the wrong one. If you sample many paths and count *answers* rather than paths,
the correct answer wins. It wins because its support was spread across several
routes, which is invisible to any procedure that only ever looks at one.

That is self-consistency. It is a change of what you are maximising: from
$\max_{\text{path}}$ to $\max_{\text{answer}} \sum_{\text{paths to it}}$. It is
marginalisation, and it is the reason the technique is worth its cost.

Notice what this account does *not* say. It does not say the model thinks harder,
or checks itself, or catches errors. Nothing in the procedure asks the model to
evaluate anything. That will matter enormously in the second half of the chapter,
because the other technique — reflection — does exactly that, and it is where the
difficulty lives.

Two things follow from the mechanism.

**Diversity is a required ingredient, and it is not free.** With no diversity
there are no alternative paths and the vote is greedy decoding. With too much,
the chains derail: samples stop being alternative valid routes and become
incoherent, and each derailed chain lands on its own unique wrong answer. So
there is a temperature that is right for voting, it is higher than the one that
is right for a single answer, and it is not the same knob you tuned for
single-sample quality.

**The vote is not the only thing you can do with a pool.** If you have a verifier,
you can pick the highest-scoring sample instead of the most frequent one. This
looks strictly better and is not, for a reason worth internalising: choosing the
maximum of $40$ noisy scores is an extreme-value problem. With an imperfect
verifier, the top score increasingly belongs to whichever *wrong* sample got the
luckiest draw, and the effect gets worse as the pool grows.
{{sec:9-practical-example}} measures argmax accuracy *falling* as diversity rises
while coverage stays flat. The fix is not to abandon the verifier but to combine
the two signals: total each distinct answer's scores, so a lucky score has to
outweigh the accumulated support of an answer several samples reached.

Now reflection. The idea is to ask the model to look at its own answer, criticise
it, and try again. It is intuitive, it is cheap to implement, and it is in nearly
every agent framework.

Here is the problem, and it is structural rather than a question of skill. The
critic's judgement is produced by the same distribution that produced the answer.
Ask "is this right?" and what the model can actually compute is closer to "is this
what I would say?". On a problem where the model's consensus is correct, those two
questions have the same answer and the critic is useful. On a problem where the
consensus is *wrong* — which is precisely the set of problems you were hoping to
fix — the critic will confirm the wrong answer and reject the occasional correct
one.

So the critic's usefulness is anti-correlated with your need for it, and no amount
of making the critic better changes that, because "better" as usually measured
means better on the average problem.

This is why an independent critic of *identical measured quality* does better, and
why an executable check does better still. The property that matters is not
accuracy. It is where the errors fall.

## 5. Formal Explanation

Let $z$ be a latent reasoning path and $y = a(z)$ the answer it produces. The
model defines $p(z \mid x)$, and greedy decoding approximates:

$$\hat{y}_{\text{greedy}} = a\Big(\arg\max_{z} \; p(z \mid x)\Big)$$ (eq:greedy-maximises-paths)

Self-consistency, given samples $z_1, \ldots, z_n \sim p(z \mid x)$, returns:

$$\hat{y}_{\text{SC}} = \arg\max_{y} \; \sum_{i} \mathbb{1}[a(z_i) = y] \;\;\xrightarrow{\;n\to\infty\;}\;\; \arg\max_{y} \sum_{z : a(z) = y} p(z \mid x)$$ (eq:marginalise-over-paths)

The two coincide only when the most likely path leads to the answer with the most
total mass. They differ whenever the correct answer is *many-to-one* in paths and
the leading wrong answer is not — which is the ordinary case for problems with
several valid solution methods.

Temperature enters through $p_\tau(z \mid x) \propto p(z \mid x)^{1/\tau}$. Two
limits bracket the useful range. As $\tau \to 0$ the distribution collapses onto
its mode, every sample is the same path, and {{eq:marginalise-over-paths}} reduces
to {{eq:greedy-maximises-paths}}. As $\tau$ grows, mass spreads to paths that were
never plausible; in a real generator these are *derailed* chains, and each lands on
its own answer rather than joining a bloc.

Model that as a mixture: with probability $d(\tau)$ a sample is derailed and
produces a unique wrong answer, and otherwise it draws from the tempered route
distribution. Expected vote accuracy is then:

$$A_{\text{SC}}(\tau) = \Pr\Big[\,(1 - d(\tau))\,p_\tau(\text{correct}) \;>\; \max_{y \neq \text{correct}} (1 - d(\tau))\,p_\tau(y)\,\Big]$$ (eq:diversity-accuracy-tradeoff)

with $p_\tau(\text{correct})$ generally rising in $\tau$ from its greedy value and
$d(\tau)$ rising too. The product has an interior optimum, and — the operationally
useful part — it is *not* at the $\tau$ that maximises single-sample accuracy,
which falls monotonically in $\tau$ from the start.

Now add a verifier producing scores $s_i$. Three selection rules:

$$\hat{y}_{\text{vote}} = \arg\max_{y} \sum_i \mathbb{1}[y_i = y], \qquad
\hat{y}_{\text{argmax}} = y_{\arg\max_i s_i}, \qquad
\hat{y}_{\text{wt}} = \arg\max_{y} \sum_{i : y_i = y} w(s_i)$$ (eq:three-selection-rules)

The middle one has a failure mode the other two do not. If correct samples score
$\mathcal{N}(\mu, 1)$ and incorrect ones $\mathcal{N}(0, 1)$, then with $k$
incorrect samples in the pool the largest incorrect score grows like
$\sqrt{2 \ln k}$, so argmax is reliable only while:

$$\mu \;\gtrsim\; \sqrt{2 \ln k}$$ (eq:argmax-extreme-value)

The right-hand side grows with the pool. **Selecting the maximum of an imperfect
score over a growing candidate set optimises against the scorer's errors**, and
the pressure increases with the very budget the verifier was deployed to exploit.
Weighted voting ({{eq:three-selection-rules}}, right) is bounded away from this
because a single lucky score must overcome an answer's accumulated support.

For reflection, model a critic as a binary judgement $c(x, y) \in
\{\text{accept}, \text{reject}\}$, characterised not by accuracy but by its two
conditional rates:

$$\alpha = \Pr[\,c = \text{accept} \mid y \text{ correct}\,], \qquad \beta = \Pr[\,c = \text{reject} \mid y \text{ wrong}\,]$$ (eq:critic-operating-point)

Accuracy mixes $\alpha$ and $\beta$ through the base rate, so two critics with
equal accuracy can sit at completely different operating points — which is why
{{sec:9-practical-example}} matches the full pair rather than the summary.

The self-critic's defining property is that $\alpha$ and $\beta$ are *functions of
the problem*, and specifically of whether the model's own mode is correct:

$$\alpha(x) \approx \Pr\big[\text{mode}(x) \text{ is correct}\big] \cdot \text{high} \;+\; \big(1 - \Pr[\cdot]\big) \cdot \text{low}$$ (eq:correlated-critic)

because the operation it can actually perform is "would I produce this again?".
Where the mode is correct, that is a good proxy for correctness. Where the mode is
wrong, it is a good proxy for the wrong answer.

This yields the chapter's structural result. A revision loop that regenerates
*conditioned on* the critique moves the answer toward the critic's preferred
answer, and for a self-critic that preference is the model's own mode. Its fixed
point is therefore:

$$\hat{y}_{\text{reflect}} \;\to\; \arg\max_{y} \; p(y \mid x) \;=\; \hat{y}_{\text{SC}}$$ (eq:reflection-is-voting)

**Intrinsic reflection converges to self-consistency**, sequentially and at higher
cost, and cannot exceed it — because no information enters the loop that was not
already in the distribution being sampled.

## 6. Mathematical Foundation

{{eq:reflection-is-voting}} is the load-bearing claim, so it is worth being
precise about its scope.

It holds when the revision is a regeneration conditioned on the critique and the
critique is computed from the same distribution. It does *not* hold when the
critic has information the generator lacks — a tool result, a retrieved document,
a second model — because then the loop's fixed point is a distribution the
generator alone does not define. That is the formal content of
{{cite:huang2024selfcorrect}}'s split between intrinsic self-correction and
correction with external feedback: the first has a fixed point inside the model,
the second does not.

It also does not bound {{cite:madaan2023selfrefine}}'s results, and the reason is
worth stating because the two papers are usually presented as contradicting each
other. Several of Self-Refine's seven tasks are *generation* tasks — dialogue
responses, code readability, sentiment rewriting — where the criterion is not a
single correct answer but a quality the model can genuinely assess more easily
than it can produce. Recognising that a response is curt is not the same
computation as writing an uncurt one, and where recognition is easier than
generation, {{eq:correlated-critic}}'s correlation is weak and reflection has real
headroom. Where the criterion is *correctness* on a reasoning problem, recognition
and generation are the same computation, and it does not.

So the two findings are compatible and the boundary is checkable in advance: ask
whether your critic is doing a different computation from your generator. If it is
not, expect {{eq:reflection-is-voting}}.

Now the value of an independent critic, quantitatively. Let the solver be wrong
with probability $e_s$ and the critic wrong with probability $e_c$ on the same
judgement. If their errors are independent, the probability that both fail on a
given problem is $e_s e_c$. If perfectly correlated, it is $\min(e_s, e_c)$. The
loop's benefit scales with the fraction of problems where the solver errs and the
critic does not:

$$\text{recoverable} = \Pr[\text{solver wrong} \wedge \text{critic right}] = e_s(1 - e_c) - \text{Cov}$$ (eq:recoverable-mass)

where $\text{Cov}$ is the covariance of the two error indicators. For a self-critic
that covariance is close to its maximum, which drives the recoverable mass toward
zero *without changing either marginal error rate*. This is why matching confusion
matrices in {{sec:9-practical-example}} isolates exactly the right thing: the
marginals are equal by construction, so the entire measured difference is
$\text{Cov}$.

One consequence is worth extracting because it inverts a common instinct. Since
{{eq:recoverable-mass}} is driven by covariance rather than by $e_c$, a *worse*
critic that is differently wrong can beat a better critic that is similarly wrong.
Ensembling a weak second model can outperform improving your first one, and the
measurement that tells you which is the covariance, not the accuracy.

## 7. Internal Mechanics

### 7.1 Why answer extraction is part of the method

{{eq:marginalise-over-paths}} counts equality of answers, so *what counts as the
same answer* is a design decision that changes the result.

Numeric answers need normalisation — $0.5$, $1/2$, and "one half" are one answer.
Free-form answers may have no useful equality at all, which is why
self-consistency is reported almost exclusively on tasks with short canonical
answers. Too fine an equivalence and every sample is its own bloc, so voting
degenerates to picking a random sample; too coarse and distinct answers merge and
the vote becomes meaningless.

This is a real limit on scope rather than an implementation detail. Voting needs
an answer space in which "the same answer" is decidable, and the tasks where
{{cite:sprague2024tocot}} found chain-of-thought helping are largely the same
tasks where that is true — which is not a coincidence, since both properties
follow from the answer being symbolic.

### 7.2 The vote is over answers, not over reasoning

Nothing in {{eq:marginalise-over-paths}} inspects the chains. Two samples that
reach the same answer by entirely different arguments count as agreement; two
samples with near-identical correct reasoning and one arithmetic slip apart count
as disagreement.

That is usually the right behaviour, and it is exactly why voting is immune to the
correlation problem that destroys reflection: it never asks the model to evaluate
anything, so there is no second judgement that could be correlated with the first.
Voting's failure mode is different in kind — it fails when the mode is wrong
({{ch:rsn-test-time-compute}}) — and the two failure modes do not compose, which is
the argument for using both.

### 7.3 Agreement is not confidence

The vote's margin is routinely logged as a confidence signal, and it is a
calibrated one only under the same condition that makes voting work at all. When
errors are unsystematic, a wide margin means the mass is concentrated on one
answer and that answer is usually correct. When errors are systematic, the margin
is widest exactly where the model is confidently wrong, and it *grows* with the
sample budget.

So agreement is a measure of the generator's concentration, not of its
correctness, and using it as a confidence score inherits every property of the
error distribution. If you use it for routing or escalation, calibrate it against
ground truth on your own task rather than assuming monotonicity.

### 7.4 What the model can actually compute about its own answer

{{ch:rsn-cot}} established that a model's stated reasoning and its answer are
outputs with separate objectives and no term tying them. The same argument applies
to a model's judgement of its own answer, with an additional twist.

When asked "is this correct?", the computation available is a forward pass over
the question and the candidate. There is no oracle inside the model to consult,
and no separate faculty for verification — the same weights that produced the
answer produce the judgement. What the judgement can express is how well the
candidate fits the distribution the model would generate, which is
{{eq:correlated-critic}}.

This also explains why *asking more sceptically* does not fix it. Raising the bar
moves the critic along its own $\alpha$–$\beta$ curve — accepting fewer correct
answers as it rejects more wrong ones — but it does not change which problems the
critic is wrong about. {{sec:9-practical-example}} sweeps that bar and finds the
overall critic accuracy improving from $42.7\%$ to $69.0\%$ while the loop's
benefit does not improve at all.

### 7.5 Cost

Self-consistency is $n$ independent generations: one prefill, $n$ decodes, batched
together. On real serving hardware that is substantially cheaper than $n\times$
({{part:15}}).

Reflection is *sequential*. Each round is generate, critique, revise, with each
step depending on the last, so $r$ rounds is at least $2r$ serial round trips and
none of it batches with itself. For a fixed budget, reflection buys far fewer
samples than voting does — and {{eq:reflection-is-voting}} says its ceiling is the
vote's answer anyway. That combination is the practical case against intrinsic
reflection: it is the more expensive way to compute the same thing.

## 8. Implementation

Two listings. The first isolates the mechanism behind self-consistency and then
compares three ways of turning a pool into an answer. The second adds a revision
loop and measures the difference between a critic that is wrong elsewhere and one
that is wrong in the same places, holding critic quality exactly fixed.

```python {tier=A name=self-consistency-mechanism}
"""Why self-consistency works, and what it is actually doing.

cite:wang2023selfconsistency's account is precise and worth taking literally:
sampling several reasoning paths and keeping the most frequent ANSWER
marginalises over the paths. Greedy decoding follows the single most likely
PATH, and those are different objects -- the most likely path need not lead to
the answer with the most total probability behind it, because many paths can
reach one answer while each wrong answer is reached by few.

This listing measures that gap directly (eq:marginalise-over-paths). Each problem
gets a set of latent routes with random weights; some fraction of them reach the
correct answer and the rest reach wrong ones, with some wrong answers reachable
by more than one route. Nothing is hand-tuned: the routes and their weights are
drawn, and greedy and voting are then compared on whatever population results.

Then it asks what else you can do with the same pool once a verifier is
available, and how each rule behaves when the verifier is bad.
"""
import numpy as np
from collections import Counter

rng = np.random.default_rng(733)

N_PROB = 6000
R = 12                      # latent routes per problem
NS = 40                     # samples per problem
P_CORRECT_ROUTE = 0.30      # chance a given route reaches the right answer

# Per-problem route structure. `route_ans[i, r]` is the answer route r produces:
# 0 means the correct answer, and positive integers are distinct wrong answers.
# Wrong routes are grouped, so several routes can share one wrong answer -- which
# is what makes a wrong answer competitive in a vote.
logits = rng.normal(size=(N_PROB, R))
is_correct = rng.random((N_PROB, R)) < P_CORRECT_ROUTE
wrong_group = rng.integers(1, 5, size=(N_PROB, R))       # 4 wrong answers
route_ans = np.where(is_correct, 0, wrong_group)


def derail_rate(tau):
    """Higher temperature also breaks chains outright. A derailed sample leaves
    the route set entirely and produces its own unique wrong answer, so derailed
    samples never form a bloc -- which is what an incoherent chain looks like."""
    return 1.0 - np.exp(-0.35 * tau)


def sample_pool(tau):
    """Sample NS routes per problem from softmax(logits / tau) and record the
    answer each one reaches. tau -> 0 is greedy decoding."""
    if tau <= 1e-6:
        best = logits.argmax(1)
        out = np.repeat(route_ans[np.arange(N_PROB), best][:, None], NS, 1)
        return out
    z = logits / tau
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    c = p.cumsum(1)
    u = rng.random((N_PROB, NS))
    idx = (u[:, :, None] > c[:, None, :]).sum(2).clip(0, R - 1)
    out = np.take_along_axis(route_ans, idx, axis=1)
    d = rng.random((N_PROB, NS)) < derail_rate(tau)
    uniq = 100 + np.arange(N_PROB * NS).reshape(N_PROB, NS)
    return np.where(d, uniq, out)


def vote(row):
    return Counter(row.tolist()).most_common(1)[0][0]


def scores(row, mu):
    """A verifier with a REAL error rate. Correct samples draw N(mu, 1) and
    incorrect ones N(0, 1). Unlike a verifier that merely fails to notice
    correct answers, this one produces FALSE POSITIVES -- some wrong sample
    scores high by chance -- which is what makes selecting over a large pool
    hazardous. mu = 0 is a useless verifier; the ranking accuracy it implies is
    measured below rather than assumed."""
    return rng.normal(size=len(row)) + mu * (row == 0)


def ranking_accuracy(mu, n=200000):
    """The verifier's own quality, measured: how often does it score a random
    correct sample above a random incorrect one?"""
    a = rng.normal(size=n) + mu
    b = rng.normal(size=n)
    return float(np.mean(a > b))


def agg_argmax(row, mu):
    return int(row[int(np.argmax(scores(row, mu)))])


def agg_weighted(row, mu):
    """Weighted self-consistency: total each distinct answer's verifier score
    rather than counting samples or trusting a single top score."""
    s = scores(row, mu)
    e = np.exp(s - s.max())          # softmax weights, so one huge score cannot
    e /= e.sum()                     # outvote a well-supported answer outright
    tot = {}
    for a, w in zip(row.tolist(), e.tolist()):
        tot[a] = tot.get(a, 0.0) + w
    return max(tot.items(), key=lambda kv: kv[1])[0]


TAUS = [0.0, 0.15, 0.3, 0.5, 0.75, 1.0, 1.5, 2.5]
MUS = [0.0, 0.4, 0.8, 1.4, 2.2, 4.0]
MU = 0.8            # the verifier used in the second table
TAU_STUDY = 1.5     # where coverage peaks; the third table is run here
Q = ranking_accuracy(MU)

print(f"{N_PROB} problems, {R} latent routes each, pools of {NS} samples.")
print(f"A route reaches the correct answer with probability {P_CORRECT_ROUTE:.0%};")
print("wrong routes are grouped onto 4 distinct wrong answers, so a wrong answer")
print("can be reached several ways too. tau is the sampling temperature.")
print()
print(f"{'tau':>7}{'single sample':>16}{'coverage':>11}{'majority vote':>16}")
print("-" * 50)

pools, single, cov, vt = {}, {}, {}, {}
for tau in TAUS:
    P = sample_pool(tau)
    pools[tau] = P
    single[tau] = float(np.mean(P[:, 0] == 0))
    cov[tau] = float(np.mean((P == 0).any(1)))
    vt[tau] = float(np.mean(np.array([vote(P[i]) for i in range(N_PROB)]) == 0))
    print(f"{tau:>7.2f}{single[tau]:>16.1%}{cov[tau]:>11.1%}{vt[tau]:>16.1%}")

best_tau = max(TAUS, key=lambda t: vt[t])
best_single = max(TAUS, key=lambda t: single[t])

print()
print()
print("Four ways to use the same pool. Throughout, the verifier ranks a")
print(f"correct sample above an incorrect one {Q:.0%} of the time.")
print()
print(f"{'tau':>7}{'single':>10}{'vote':>10}{'verifier':>11}{'weighted':>11}"
      f"{'coverage':>11}")
print(f"{'':>7}{'':>10}{'':>10}{'argmax':>11}{'vote':>11}{'':>11}")
print("-" * 60)

table = {}
for tau in TAUS:
    P = pools[tau]
    row = {
        "single": single[tau],
        "vote": vt[tau],
        "argmax": float(np.mean([agg_argmax(P[i], MU) == 0
                                 for i in range(N_PROB)])),
        "weighted": float(np.mean([agg_weighted(P[i], MU) == 0
                                   for i in range(N_PROB)])),
    }
    table[tau] = row
    print(f"{tau:>7.2f}{row['single']:>10.1%}{row['vote']:>10.1%}"
          f"{row['argmax']:>11.1%}{row['weighted']:>11.1%}{cov[tau]:>11.1%}")

print()
print()
print(f"How each rule responds to verifier quality, at tau={TAU_STUDY}")
print("-- the temperature at which coverage peaks.")
print()
print(f"{'verifier':>10}{'ranks correct':>15}{'vote':>10}{'argmax':>11}"
      f"{'weighted':>11}")
print(f"{'mu':>10}{'above wrong':>15}{'':>10}{'':>11}{'':>11}")
print("-" * 57)
P = pools[TAU_STUDY]
byq, qof = {}, {}
for mu in MUS:
    qq = ranking_accuracy(mu)
    qof[mu] = qq
    r = {"vote": vt[TAU_STUDY],
         "argmax": float(np.mean([agg_argmax(P[i], mu) == 0
                                  for i in range(N_PROB)])),
         "weighted": float(np.mean([agg_weighted(P[i], mu) == 0
                                    for i in range(N_PROB)]))}
    byq[mu] = r
    print(f"{mu:>10.1f}{qq:>15.1%}{r['vote']:>10.1%}{r['argmax']:>11.1%}"
          f"{r['weighted']:>11.1%}")

print(f"""
The first table is self-consistency working, and the first row is why it works.

At tau=0 the generator is greedy: it follows the single highest-weight route
every time. All {NS} samples are identical, coverage equals single-sample
accuracy at {cov[0.0]:.1%}, and the vote returns exactly what one sample
returned. A pool of forty copies of one answer holds no more information than
the answer does.

Raise the temperature and the vote pulls away from the single sample. At
tau={TAU_STUDY} one sample is {single[TAU_STUDY]:.1%} accurate and the vote is
{vt[TAU_STUDY]:.1%} -- {vt[TAU_STUDY] - single[TAU_STUDY]:+.1%} against a sample
from the same distribution, and {vt[TAU_STUDY] - vt[0.0]:+.1%} against greedy
decoding.

The mechanism is the one cite:wang2023selfconsistency names, and this listing is
built so it is the only mechanism available. Several routes reach the correct
answer. Each one individually may be less likely than the single best wrong
route -- which is exactly what greedy decoding follows -- but their probabilities
SUM when you count answers instead of paths. Voting marginalises over the latent
route (eq:marginalise-over-paths); greedy decoding maximises over it. The two
disagree whenever the correct answer is reachable more ways than the most likely
wrong one.

Nothing here involves the model being more careful, or checking its work, or
noticing a mistake. It is an argmax-versus-sum distinction, and that is the whole
of self-consistency. Keep that in mind for the next listing, where a technique
that DOES ask the model to check its work behaves very differently.

The temperature column contains the configuration error worth knowing about.
Single-sample accuracy is best at tau={best_single}
({single[best_single]:.1%}) and falls monotonically -- {single[2.5]:.1%} by
tau=2.5, because high temperature derails chains outright. The vote moves the
other way, {vt[0.0]:.1%} to {vt[2.5]:.1%}, and flattens out past tau={TAU_STUDY}
rather than turning over sharply within the range swept.

So the two objectives want different settings, and **a system whose temperature
was tuned for single-sample accuracy is at the wrong setting for
self-consistency** -- worth {vt[TAU_STUDY] - vt[best_single]:+.1%} here. Coverage
peaks at tau={TAU_STUDY} ({cov[TAU_STUDY]:.1%}) and then declines, which is the
signal to watch: past that point extra temperature is destroying correct samples
rather than diversifying them, and the vote's flatness is hiding it.

The second table adds a verifier and asks what else the pool is worth.

The pattern in the argmax column is the important one. It rises to
{table[1.0]['argmax']:.1%} at tau=1.0 and then FALLS to {table[2.5]['argmax']:.1%}
at tau=2.5, while coverage over the same range is essentially flat. Picking the
single highest-scoring sample is an extreme-value problem: with {NS} samples and
a verifier that is right {Q:.0%} of the time, the top score increasingly belongs
to whichever wrong sample got the luckiest draw. More diversity gives it more
chances. **Selecting by argmax over a large pool optimises against the verifier's
errors**, which is the effect ch:rsn-test-time-compute warned about, measured.

Weighted voting -- summing verifier scores per distinct answer rather than
trusting a single top score -- peaks at {table[TAU_STUDY]['weighted']:.1%} and is
ahead of argmax everywhere from tau=0.75 up. It is harder to fool because one
lucky score has to outweigh the accumulated support of an answer several samples
reached.

The third table sweeps verifier quality at fixed temperature, and it is where the
recommendation comes from -- including its limit.

With a useless verifier (mu=0, ranking correct above incorrect
{qof[0.0]:.1%} of the time -- chance), argmax collapses to {byq[0.0]['argmax']:.1%},
below even a single sample, because it is now deliberately selecting the most
extreme noise draw. Weighted voting gets {byq[0.0]['weighted']:.1%}. The plain
vote gets {byq[0.0]['vote']:.1%}.

That last comparison is the honest limit of the recommendation, and it is not
what I expected to find. Weighted voting does NOT degrade gracefully into plain
voting when the verifier is worthless -- it degrades into a randomly-weighted
vote, which is a noisier estimator of the same thing, and it loses
{byq[0.0]['vote'] - byq[0.0]['weighted']:.1%} for the privilege.

From mu=0.4 upward -- a verifier that ranks correctly {qof[0.4]:.1%} of the time,
which is a low bar -- weighted voting leads: {byq[0.4]['weighted']:.1%} against
the vote's {byq[0.4]['vote']:.1%} and argmax's {byq[0.4]['argmax']:.1%}. It stays
ahead of argmax until the verifier is nearly perfect ({qof[2.2]:.1%}), where the
two converge, and at mu=4.0 argmax is marginally ahead at
{byq[4.0]['argmax']:.1%} against {byq[4.0]['weighted']:.1%}.

So the rule is not "always weight". It is: measure the verifier's ranking
accuracy, and if it is meaningfully above chance, weight; if it is at chance,
count. Argmax is correct only when the verifier is close to perfect, which in
practice means an executable check rather than a learned one -- and it is the
default in most implementations, which is the wrong default for every learned
verifier in the middle of this table.

One boundary on all of it. Every number here is capped by the coverage column,
and coverage is a property of the generator. Aggregation decides how much of what
sampling produced you actually collect; it cannot produce anything.""")
```

The second listing keeps the same generator and adds a revision loop.

```python {tier=A name=critic-error-correlation}
"""Why voting works and reflection does not, measured side by side.

cite:huang2024selfcorrect's result is that intrinsic self-correction -- a model
revising its own answer using only its own capabilities -- does not improve
reasoning performance and often degrades it, while correction guided by external
feedback does help. Those two findings are usually reported together as a
puzzle: the model can clearly criticise, so why does criticising itself not
work?

This listing takes the previous one's setup and adds a revision loop
(eq:correlated-critic). A proposal is judged by a critic; if the critic rejects
it, the solver proposes again. Three critics, matched so the comparison is about
CORRELATION and not about competence:

  self-check   the solver's own judgement -- sample again and see whether the
               proposal is what it would say
  independent  a critic with an IDENTICAL confusion matrix whose errors fall
               elsewhere
  oracle       knows the answer

The self-check critic is not a weakened version of the others. It accepts correct
answers and catches wrong ones at exactly the same rates. The only thing that
differs is what it is wrong ABOUT.
"""
import numpy as np
from collections import Counter

rng = np.random.default_rng(811)

N_PROB = 8000
R = 12
P_CORRECT_ROUTE = 0.30
TAU = 1.0
M_CHECK = 5                 # samples the self-check critic draws
ROUNDS = 4

logits = rng.normal(size=(N_PROB, R))
is_correct = rng.random((N_PROB, R)) < P_CORRECT_ROUTE
route_ans = np.where(is_correct, 0, rng.integers(1, 5, size=(N_PROB, R)))

z = logits / TAU
z = z - z.max(1, keepdims=True)
PR = np.exp(z)
PR /= PR.sum(1, keepdims=True)
CUM = PR.cumsum(1)


def draw(idx_rows, n):
    """Draw n samples for each problem in idx_rows."""
    u = rng.random((len(idx_rows), n))
    j = (u[:, :, None] > CUM[idx_rows][:, None, :]).sum(2).clip(0, R - 1)
    return np.take_along_axis(route_ans[idx_rows], j, axis=1)


ALL = np.arange(N_PROB)
# Per-problem answer distribution, used to reason about what the pool contains.
p_correct = np.array([PR[i][route_ans[i] == 0].sum() for i in ALL])
pool = draw(ALL, 200)
modal = np.array([Counter(pool[i].tolist()).most_common(1)[0][0]
                  for i in ALL])

print(f"{N_PROB} problems, {R} routes each, temperature {TAU}.")
print(f"A single sample is correct {float(np.mean(draw(ALL, 1)[:, 0] == 0)):.1%}"
      " of the time.")
print(f"The correct answer is the modal one for {float(np.mean(modal == 0)):.1%}"
      " of problems.")
print()
print()


def self_check(idx_rows, proposals, t):
    """The solver judging itself: draw M_CHECK more samples and accept the
    proposal if it reappears at least t times. This is what "are you sure?"
    amounts to when the reviewer and the author are the same distribution, and
    t is how sceptically the question is asked -- t=1 is a gentle "does this
    still look right", t=4 is "convince me"."""
    s = draw(idx_rows, M_CHECK)
    return (s == proposals[:, None]).sum(1) >= t


def oracle_check(idx_rows, proposals):
    return proposals == 0


# Calibrate an INDEPENDENT critic to the self-check critic's measured accuracy.
probe = draw(ALL, 1)[:, 0]
STRICT = [1, 2, 3, 4]
sc_of = {t: self_check(ALL, probe, t) for t in STRICT}
RATES = {t: (float(np.mean(sc_of[t][probe == 0])),
             float(np.mean(~sc_of[t][probe != 0]))) for t in STRICT}


def independent_check(idx_rows, proposals, tpr, tnr):
    """A critic with the SAME confusion matrix as the self-check critic -- same
    rate of accepting correct proposals, same rate of rejecting wrong ones --
    whose errors fall independently of which problem it is looking at.

    Matching the full confusion matrix rather than overall accuracy matters: two
    critics with equal accuracy can sit at completely different operating
    points, and then the comparison measures the operating point instead of the
    correlation."""
    truth = proposals == 0
    u = rng.random(len(proposals))
    return np.where(truth, u < tpr, u > tnr)


print("Each critic, measured on one round of proposals. t is how many of the")
print(f"{M_CHECK} re-samples must match the proposal for the self-check critic to")
print("accept it. The independent critic is given the SAME two rates.")
print()
print(f"{'strictness t':>14}{'accepts a':>13}{'rejects a':>13}{'overall':>11}")
print(f"{'':>14}{'correct one':>13}{'wrong one':>13}{'accuracy':>11}")
print("-" * 51)
good = probe == 0
for t in STRICT:
    tpr, tnr = RATES[t]
    ov = float(np.mean(sc_of[t] == good))
    print(f"{t:>14}{tpr:>13.1%}{tnr:>13.1%}{ov:>11.1%}")

print()
print()
print(f"Revision loop: propose; if the critic rejects, propose again; "
      f"{ROUNDS} rounds.")
print("Accuracy at the end, for the self-check critic and for an independent")
print("critic with an IDENTICAL confusion matrix.")
print()


# An independent critic is a different model, so it has its own preferred
# answer. Give it one that is correct as often as the solver's mode is, but on
# an independently chosen set of problems.
indep_pref = np.where(rng.random(N_PROB) < float(np.mean(modal == 0)),
                      0, rng.integers(1, 5, size=N_PROB))
GAMMA = 0.75          # how strongly a revision moves toward the critique


def run(kind, t, policy="resample", rounds=ROUNDS):
    """policy='resample' redraws from the solver. policy='toward' regenerates
    CONDITIONED on the critique, which pulls the answer toward whatever the
    critic would have said -- which is what "reconsider, given this objection"
    does in practice."""
    cur = draw(ALL, 1)[:, 0]
    tpr, tnr = RATES[t]
    out = [float(np.mean(cur == 0))]
    for _ in range(rounds):
        if kind == "self":
            keep = self_check(ALL, cur, t)
        elif kind == "indep":
            keep = independent_check(ALL, cur, tpr, tnr)
        elif kind == "oracle":
            keep = oracle_check(ALL, cur)
        else:
            keep = np.ones(N_PROB, dtype=bool)
        redo = np.flatnonzero(~keep)
        if len(redo):
            cur = cur.copy()
            fresh = draw(redo, 1)[:, 0]
            if policy == "toward" and kind in ("self", "indep"):
                pref = modal[redo] if kind == "self" else indep_pref[redo]
                pull = rng.random(len(redo)) < GAMMA
                cur[redo] = np.where(pull, pref, fresh)
            else:
                cur[redo] = fresh
        out.append(float(np.mean(cur == 0)))
    return out


none = run("none", 1)
orac = run("oracle", 1)
print(f"{'':>14}{'redraw on reject':>27}{'revise toward critique':>29}")
print(f"{'strictness t':>14}{'self-check':>14}{'independent':>13}"
      f"{'self-check':>15}{'independent':>14}")
print("-" * 70)
selfc, indep, selft, indt = {}, {}, {}, {}
for t in STRICT:
    selfc[t] = run("self", t)
    indep[t] = run("indep", t)
    selft[t] = run("self", t, "toward")
    indt[t] = run("indep", t, "toward")
    print(f"{t:>14}{selfc[t][ROUNDS]:>14.1%}{indep[t][ROUNDS]:>13.1%}"
          f"{selft[t][ROUNDS]:>15.1%}{indt[t][ROUNDS]:>14.1%}")
print(f"{'none':>14}{none[ROUNDS]:>14.1%}{none[ROUNDS]:>13.1%}"
      f"{none[ROUNDS]:>15.1%}{none[ROUNDS]:>14.1%}")
print(f"{'oracle':>14}{orac[ROUNDS]:>14.1%}{'--':>13}{'--':>15}{'--':>14}")

T_HARD = 3
print()
print()
print(f"Where does the self-check critic go wrong? At t={T_HARD}, split by")
print("whether the solver's modal answer is the correct one.")
print()
print(f"{'problems where the mode is':>30}{'count':>8}{'accepts a':>13}"
      f"{'rejects a':>13}")
print(f"{'':>30}{'':>8}{'correct one':>13}{'wrong one':>13}")
print("-" * 64)
sch = sc_of[T_HARD]
split = {}
for name, m in (("correct", modal == 0), ("wrong", modal != 0)):
    g = good & m
    b = (~good) & m
    tpr = float(np.mean(sch[g])) if g.any() else float("nan")
    tnr = float(np.mean(~sch[b])) if b.any() else float("nan")
    split[name] = (int(m.sum()), tpr, tnr)
    print(f"{name:>30}{int(m.sum()):>8}{tpr:>13.1%}{tnr:>13.1%}")

n0, orc = none, orac
vote_acc = float(np.mean(modal == 0))
gaps = ", ".join("%+.1f%%" % (100 * (indep[t][ROUNDS] - selfc[t][ROUNDS]))
                 for t in STRICT)
best_redraw = max(selfc[t][ROUNDS] for t in STRICT)
best_toward = max(selft[t][ROUNDS] for t in STRICT)
best_itoward = max(indt[t][ROUNDS] for t in STRICT)
print(f"""
The first table is the critic, and it behaves sensibly. Asking more sceptically
trades one error for the other in the ordinary way: at t=1 it accepts a correct
proposal {RATES[1][0]:.1%} of the time and catches a wrong one {RATES[1][1]:.1%}
of the time; at t=4 those are {RATES[4][0]:.1%} and {RATES[4][1]:.1%}. Overall
accuracy improves throughout, from {float(np.mean(sc_of[1] == good)):.1%} to
{float(np.mean(sc_of[4] == good)):.1%}. On any summary statistic this is a critic
getting better at its job.

The second table is the revision loop, and it holds three results.

First: with the redraw policy, self-correction helps a little. The best strictness
takes accuracy from {none[ROUNDS]:.1%} to {best_redraw:.1%}. That is real, and it
is a long way below the oracle's {orc[ROUNDS]:.1%} -- which is the same loop,
same solver, same number of rounds, with a critic that actually knows.

Second: an independent critic with an IDENTICAL confusion matrix does better at
every strictness, by {gaps}. Same acceptance rate on correct proposals, same
rejection rate on wrong ones, different outcome. The difference is not
competence; it is WHERE the errors fall (eq:correlated-critic).

Third, and this is the one to carry: under the "revise toward the critique"
policy -- where a rejected answer is regenerated conditioned on the objection
rather than redrawn from scratch, which is what reconsidering actually does --
self-correction reaches {best_toward:.1%}.

Compare that with the majority vote over the same generator, which is
{vote_acc:.1%}.

Those are the same number, and they are the same number for a reason.
**Self-correction against your own judgement is a slow, sequential, expensive
implementation of self-consistency.** Revising toward what you would say on
reflection moves the answer toward your modal answer, and the modal answer is
exactly what voting returns in one parallel batch. The reflection loop cannot
exceed it, because there is nothing in the loop that was not already in the
distribution being sampled. The independent critic under the same policy reaches
{best_itoward:.1%}, and the difference between those two columns is the entire
value of the critic being a different system.

The third table shows why the ceiling sits where it does, and it is the sharpest
number here.

At t={T_HARD}, on problems where the solver's modal answer is already correct, the
self-check critic accepts a correct proposal {split['correct'][1]:.1%} of the time
and rejects a wrong one {split['correct'][2]:.1%} of the time. A useful critic.

On problems where the mode is WRONG, it accepts a correct proposal
{split['wrong'][1]:.1%} of the time.

That is the whole failure in one number. On exactly the problems the system gets
wrong, a correct answer that does turn up is thrown away nine times in ten,
because the critic's test is "is this what I would say?" and what it would say on
those problems is wrong. The critic is not merely unhelpful where it matters. It
is inverted.

So voting and reflection are not two techniques with different strengths. They
are the same information used two ways, and one of them costs a sequential round
trip per revision.

Voting AGGREGATES: it reads the whole distribution and returns its mode. It never
asks the model to evaluate anything, so there is no second judgement that could
be correlated with the first.

Reflection FILTERS: it conditions on a proposal and asks the model to judge it.
That judgement is computed by the distribution that produced the proposal, so
where the distribution is wrong the judgement is wrong in the same way, and the
filter discards the samples that would have rescued it.

One honest note on what this listing does and does not show. It reproduces
cite:huang2024selfcorrect's finding that intrinsic self-correction adds nothing
the model did not already contain, and their finding that external feedback does
help. It does NOT reproduce the outright DEGRADATION they report on some tasks,
and the reason is a mechanism this model omits: a real model asked to reconsider
will sometimes abandon a correct answer simply because it was challenged,
regardless of what its own distribution says. That is instruction-following under
pressure rather than a property of the reasoning distribution, and leaving it out
makes the picture here optimistic rather than pessimistic.

The practical question is therefore not "is my critic any good" -- the first table
shows that can be answered yes while the loop delivers almost nothing. It is
**how correlated is my critic's error with my solver's**, and the cheapest large
improvement available is not a better critic but a differently-wrong one: another
model, another training lineage, or an executable check that is not a model at
all, which is ch:rsn-tool-assisted's subject.""")
```

## 9. Practical Example

The first listing gives each of $6{,}000$ problems twelve latent routes with
random weights. A route reaches the correct answer with probability $0.30$; wrong
routes are grouped onto four distinct wrong answers, so a wrong answer can be
reached several ways too. Pools are $40$ samples.

```
    tau   single sample   coverage   majority vote
--------------------------------------------------
   0.00           29.8%      29.8%           29.8%
   0.15           28.3%      52.7%           30.0%
   0.30           26.8%      72.3%           31.0%
   0.50           24.7%      86.9%           32.3%
   0.75           23.1%      92.6%           34.4%
   1.00           21.3%      95.1%           36.3%
   1.50           17.5%      95.9%           38.2%
   2.50           12.6%      94.7%           38.7%
```

The first row is the argument. At $\tau = 0$ every sample is the same route,
coverage equals single-sample accuracy, and the vote returns precisely what greedy
decoding returns. Diversity is not a nice-to-have; without it there is nothing to
marginalise.

At $\tau = 1.5$ a single sample is $17.5\%$ accurate and the vote is $38.2\%$ —
$+20.7$ points over a sample from the same distribution and $+8.4$ over greedy.
The mechanism is the only one available in this construction: several routes reach
the correct answer, each individually less likely than the best wrong route, and
their probabilities sum when you count answers instead of paths
({{eq:marginalise-over-paths}}).

The temperature columns contain a configuration error worth naming. Single-sample
accuracy is best at $\tau = 0$ and falls monotonically to $12.6\%$; the vote rises
to $38.7\%$. **A system whose temperature was tuned for single-sample accuracy is
at the wrong setting for self-consistency.** The signal for the upper bound is
coverage, which peaks at $\tau = 1.5$ and then declines — past that point extra
temperature destroys correct samples rather than diversifying them, and the vote's
flatness hides it.

Adding a verifier that ranks a correct sample above an incorrect one $72\%$ of the
time:

```
    tau    single      vote   verifier   weighted   coverage
                                argmax       vote           
------------------------------------------------------------
   0.50     24.7%     32.3%      50.5%      46.5%      86.9%
   0.75     23.1%     34.4%      52.4%      53.7%      92.6%
   1.00     21.3%     36.3%      52.6%      59.2%      95.1%
   1.50     17.5%     38.2%      47.9%      61.4%      95.9%
   2.50     12.6%     38.7%      38.1%      57.5%      94.7%
```

The argmax column peaks at $52.6\%$ and then *falls* to $38.1\%$ while coverage
stays flat. That is {{eq:argmax-extreme-value}}: with $40$ samples and an
imperfect scorer, the top score increasingly belongs to whichever wrong sample got
the luckiest draw, and more diversity supplies more chances. Selecting the maximum
of an imperfect score over a growing pool optimises against the scorer's errors.

Weighted voting peaks at $61.4\%$ and leads argmax everywhere from $\tau = 0.75$
up, because a lucky score must outweigh an answer's accumulated support. Sweeping
verifier quality at $\tau = 1.5$:

```
  verifier  ranks correct      vote     argmax   weighted
        mu    above wrong                                
---------------------------------------------------------
       0.0          49.9%     38.2%      18.4%      33.9%
       0.4          61.2%     38.2%      31.3%      48.4%
       0.8          71.5%     38.2%      49.1%      61.3%
       1.4          83.8%     38.2%      71.1%      76.6%
       2.2          94.1%     38.2%      87.9%      88.3%
       4.0          99.8%     38.2%      95.5%      94.9%
```

At chance ($\mu = 0$) argmax collapses to $18.4\%$ — below a single sample, because
it is deliberately selecting the most extreme noise draw. The honest limit of the
recommendation is in the same row: weighted voting gets $33.9\%$ against the plain
vote's $38.2\%$. It does *not* degrade gracefully into voting; it degrades into a
randomly-weighted vote, which is a noisier estimator of the same thing.

From $\mu = 0.4$ — a verifier right $61.2\%$ of the time, a low bar — weighted
voting leads on both, and stays ahead of argmax until the verifier is nearly
perfect. So the rule is: measure the verifier's ranking accuracy; if it is
meaningfully above chance, weight; if it is at chance, count; use argmax only when
the check is essentially exact, which in practice means executable rather than
learned.

The second listing adds a revision loop at $\tau = 1.0$, where a single sample is
$29.5\%$ accurate and the correct answer is modal for $36.8\%$ of problems.

```
  strictness t    accepts a    rejects a    overall
                correct one    wrong one   accuracy
---------------------------------------------------
             1        86.5%        23.9%      42.7%
             2        62.4%        54.2%      56.6%
             3        37.4%        76.4%      64.8%
             4        16.7%        91.3%      69.0%
```

Asking more sceptically trades one error for the other, and overall critic
accuracy improves from $42.7\%$ to $69.0\%$. On any summary statistic this critic
is getting better. Now the loop:

```
                         redraw on reject       revise toward critique
  strictness t    self-check  independent     self-check   independent
----------------------------------------------------------------------
             1         33.3%        36.0%          35.3%         38.3%
             2         34.5%        36.2%          36.5%         39.1%
             3         32.0%        33.2%          36.0%         36.4%
             4         31.4%        31.4%          35.5%         35.8%
          none         29.8%        29.8%          29.8%         29.8%
        oracle         72.7%           --             --            --
```

Three results. Self-correction helps a little — $29.8\%$ to $34.5\%$ — and the
oracle critic, the same loop with a critic that actually knows, reaches $72.7\%$.
That gap is what correlation is costing.

The independent critic beats the self-check critic at every strictness
($+2.7, +1.7, +1.2, +0.0$ points) with an *identical* confusion matrix. Equal
$\alpha$, equal $\beta$, different outcome: the difference is entirely
$\text{Cov}$ in {{eq:recoverable-mass}}.

And under the revise-toward-critique policy, self-correction reaches $36.5\%$
against a majority vote of $36.8\%$. Those are the same number, which is
{{eq:reflection-is-voting}}: revising toward what you would say on reflection
moves the answer toward your modal answer, and the vote returns the modal answer
directly, in one parallel batch instead of four sequential rounds. The independent
critic under the same policy reaches $39.1\%$, and that difference is the whole
value of the critic being a different system.

The last table localises the failure:

```
    problems where the mode is   count    accepts a    rejects a
                                        correct one    wrong one
----------------------------------------------------------------
                       correct    2943        54.2%        91.3%
                         wrong    5057        10.6%        71.2%
```

Where the mode is correct, the self-critic is useful. Where the mode is wrong —
the $5{,}057$ problems you were hoping to fix — it accepts a correct proposal
$10.6\%$ of the time. A correct answer that does turn up is discarded nine times
in ten, because the critic's test is "is this what I would say?" and what it would
say there is wrong.

One caveat stated in the listing and worth repeating: this reproduces
{{cite:huang2024selfcorrect}}'s finding that intrinsic self-correction adds nothing
the model did not already contain, and that external feedback helps. It does not
reproduce the outright *degradation* they report on some tasks, because the model
here omits a real mechanism — abandoning a correct answer merely because it was
challenged. That omission makes these numbers optimistic.

## 10. Production Considerations

Tune temperature for the aggregate, not for one sample. If you deploy
self-consistency on a stack tuned at temperature $0.2$ for single-answer quality,
you are leaving most of the benefit unclaimed. Sweep it against the vote, and use
coverage as the stopping signal: when coverage stops rising, further temperature
is destroying samples.

Normalise answers before counting. Voting is equality on extracted answers, and a
mediocre extractor silently converts self-consistency into random selection from
the pool.

Default to weighted voting when you have any real verifier signal, and measure the
signal first — pairwise ranking accuracy on a held-out set is cheap and it is the
number that picks the rule. Reserve pure argmax for executable checks.

Do not log the vote margin as confidence without calibrating it. It measures the
generator's concentration, and when errors are systematic it is widest exactly
where the system is confidently wrong ({{ch:rsn-test-time-compute}}).

Treat intrinsic reflection loops as a cost centre until proven otherwise. They are
sequential, they are expensive, and on correctness tasks
{{eq:reflection-is-voting}} says their ceiling is a number you can get in one
parallel batch. If you have one in production, the measurement that settles it is
an ablation: same budget, voting versus reflecting.

Where you do want criticism, buy independence rather than quality. A second model
from a different lineage, a differently-prompted critic, or best of all an
executable check will beat a better copy of the same system, and
{{eq:recoverable-mass}} says why.

## 11. Common Mistakes

**Running self-consistency at low temperature.** At $\tau \to 0$ the vote is greedy
decoding with $n$ times the cost. This is the single most common way to deploy the
technique and get nothing.

**Picking the highest-scoring sample.** Intuitive, and it degrades as the pool
grows ({{eq:argmax-extreme-value}}). With a chance-level verifier it is worse than
taking one sample at random.

**Judging a critic by its accuracy.** Accuracy mixes $\alpha$ and $\beta$ through
the base rate, and neither of those is what determines the loop's value.
{{sec:9-practical-example}} improves critic accuracy by 26 points with no
improvement in the loop.

**Assuming a better self-critic will fix reflection.** The limit in
{{eq:reflection-is-voting}} is structural. Improving the critic moves it along its
own operating curve without changing which problems it is wrong about.

**Reading {{cite:madaan2023selfrefine}} and {{cite:huang2024selfcorrect}} as
contradictory.** They measure different task types. Where recognition is an easier
computation than generation, reflection has headroom; where they are the same
computation, it does not.

**Using agreement as a stopping rule in an agent loop.** If the agent's errors are
systematic, agreement arrives fastest on the cases it is wrong about.

## 12. Failure Modes

*Voting at zero temperature.* Silent, expensive, and indistinguishable from
success in a dashboard, because accuracy is unchanged and cost is $n$ times
higher.

*Answer-extraction collapse.* If normalisation is too strict, every sample is its
own answer and the vote returns sample one. Detect it by logging the modal
answer's share; a share near $1/n$ means the vote is not voting.

*Verifier gaming at scale.* Argmax accuracy that improves during evaluation at
$n=8$ and degrades in production at $n=64$ is {{eq:argmax-extreme-value}}, and it
looks like a regression in the model rather than in the selection rule.

*Reflection loops that never terminate.* A critic with low $\alpha$ rejects
correct answers indefinitely. Cap rounds and treat a hit cap as a failure.

*Confidently reinforced consensus.* Reflection under a self-critic drives the
answer toward the mode ({{eq:reflection-is-voting}}), so on problems where the mode
is wrong the loop increases the system's confidence in a wrong answer while
producing an articulate justification for it.

## 13. Alternatives

**A learned verifier.** {{cite:cobbe2021gsm8k}} and {{cite:lightman2023verify}}
train a model to score solutions, which raises $\mu$ in
{{eq:argmax-extreme-value}} and makes selection worth more at every budget.
{{ch:rsn-supervision}} is about how those are trained and why process supervision
differs from outcome supervision.

**Search rather than restart.** {{cite:yao2023tot}} and
{{cite:snell2024testtime}}'s beam search against a process reward model spend the
budget on branching, which addresses irrecoverable prefixes rather than variance.

**More samples.** {{cite:brown2024monkeys}}: if coverage is still climbing, extra
samples are the cheapest available improvement, and they parallelise.

**External feedback loops.** {{cite:huang2024selfcorrect}}'s positive case, and
{{ch:rsn-tool-assisted}}'s subject: an interpreter, a test suite, or a retrieved
document breaks the correlation in {{eq:correlated-critic}} because it is not a
sample from the model.

**Debate between differently-prompted instances.** A cheap approximation of
independence. It is {{maturity:EMERGING}}, and its value should be measured as
error covariance rather than assumed from the framing.

## 14. Evaluation

Report the vote against greedy decoding at the *same* temperature and against
greedy at its own best temperature. Only the second comparison tells you what
self-consistency bought, and only the first tells you whether your temperature is
set correctly.

Log the modal share, not just accuracy. It diagnoses answer-extraction failure, it
is the quantity {{eq:vote-condition}} predicts from, and it is nearly free.

Measure your verifier's pairwise ranking accuracy on a held-out pool containing
both correct and incorrect samples. That single number picks between voting,
weighting and argmax.

For any reflection loop, measure the critic's $\alpha$ and $\beta$ *conditioned on
whether the system was right* — that is the table that predicts the loop's value,
and the aggregate confusion matrix does not.

And run the ablation that matters: the same token budget spent on voting versus
spent on reflecting. {{eq:reflection-is-voting}} predicts a tie on correctness
tasks, and a tie is a strong argument for the cheaper, parallel option.

## 15. Advanced Concepts

**Universal self-consistency.** Where answers have no equality relation, a model
can be asked to cluster the samples and pick the largest cluster, which extends
voting to free-form outputs. It reintroduces exactly the correlation problem this
chapter is about — the clusterer is the generator — and should be measured, not
assumed. {{maturity:EMERGING}}.

**Adaptive sample counts.** Stop sampling once the vote's margin is decisive. This
saves most of the budget on easy problems and is the voting analogue of
{{ch:rsn-test-time-compute}}'s early stopping. The subtlety is that the margin is
not calibrated, so the stopping rule inherits the error-shape dependence.

**Error covariance as the design variable.** {{eq:recoverable-mass}} suggests
building critic ensembles for *decorrelation* rather than accuracy — different
base models, different data, different prompting lineages — which is the same
principle as ensemble diversity in {{ch:rsn-vs-generation}}, arriving for the same
reason.

**Where the boundary really is.** {{eq:reflection-is-voting}} predicts that
reflection helps exactly when recognition is a different computation from
generation. Making that testable in advance — a measurable property of a task that
says whether self-criticism has headroom — is open, and it would settle the
Self-Refine/self-correct disagreement per task rather than per paper.
{{maturity:RESEARCH FRONTIER}}.

## 16. Connection to Previous Chapters

{{ch:rsn-test-time-compute}} showed the vote converging to the generator's mode
and bounded it by {{eq:vote-condition}}. This chapter measured what that mode is
worth against greedy decoding, and found the gap is created by *diversity* — which
is why the same chapter's coverage curve is also the right diagnostic for
temperature.

{{ch:rsn-cot}}'s faithfulness result is the direct ancestor of
{{eq:correlated-critic}}: a model's account of its own answer is computed by the
same weights that produced it, so it cannot be an independent check. Reflection is
that finding applied to a control loop instead of to an explanation.

{{ch:rsn-vs-generation}}'s caution about ensemble disagreement — that it measures
diversity of extrapolation rather than distance from the training distribution —
is the same principle as {{eq:recoverable-mass}}, and both say the useful quantity
is covariance rather than quality.

Ahead: {{ch:rsn-supervision}} builds the verifier that
{{eq:argmax-extreme-value}} says is worth so much, and
{{ch:rsn-tool-assisted}} is the case where the critic is not a model at all, which
is the only clean escape from {{eq:correlated-critic}}.

## 17. Exercises

1. In the first listing, set `P_CORRECT_ROUTE` to $0.6$ so most routes are
   correct. Predict what happens to the gap between greedy and the vote before
   running it, then explain the result in terms of
   {{eq:marginalise-over-paths}}.

2. Change `wrong_group` so all wrong routes map to a *single* wrong answer.
   Measure the vote across temperature and connect the result to
   {{eq:vote-condition}}.

3. Sweep the pool size `NS` at fixed $\mu$ and plot argmax accuracy against it.
   Fit the decline and compare it with the $\sqrt{2 \ln k}$ prediction of
   {{eq:argmax-extreme-value}}.

4. In the second listing, give the independent critic a *lower* accuracy than the
   self-check critic and find the point at which it stops winning. What does that
   crossover tell you about how much independence is worth?

5. Add a fourth revision policy in which the critic's rejection is ignored half
   the time at random. Explain why it can outperform obeying the critic.

6. Construct a task where recognition really is easier than generation and show
   that {{eq:reflection-is-voting}} does not bind. What property of the task made
   the difference?

## 18. Interview Questions

1. Why does self-consistency beat greedy decoding? Answer without using the words
   "checks" or "errors cancel".

2. You enable self-consistency and accuracy does not move. Name three causes and
   the measurement that distinguishes them.

3. Why does picking the highest-scoring sample get worse as you sample more?

4. Two critics have the same accuracy. One improves your reflection loop and one
   does not. What differs, and how would you measure it?

5. {{cite:madaan2023selfrefine}} reports large gains from self-refinement;
   {{cite:huang2024selfcorrect}} reports none. Reconcile them.

6. Your agent framework runs a reflect-and-revise loop. What experiment would you
   run to decide whether to keep it?

## 19. Research Questions

1. Can the error covariance between a solver and a critic be estimated cheaply
   before deployment, given that it is the quantity {{eq:recoverable-mass}} says
   determines a loop's value?

2. Is there an aggregation rule that has voting's floor at chance-level verifiers
   and argmax's ceiling at perfect ones, without the loss that weighted voting
   shows at $\mu = 0$?

3. What measurable property of a task predicts whether recognition is easier than
   generation for a given model — the condition under which
   {{eq:reflection-is-voting}} does not bind?

4. Can universal self-consistency be made robust to the fact that the clustering
   model is the generator, or does it inherit {{eq:correlated-critic}} in full?

5. Does the optimal self-consistency temperature vary per problem in a way that is
   predictable before sampling, and would per-request temperature beat a global
   setting?

## 20. Chapter Summary

Self-consistency works because voting marginalises over reasoning paths while
greedy decoding maximises over them ({{eq:marginalise-over-paths}}). Measured on a
generator where that is the only available mechanism, the vote beat greedy
decoding by $8.4$ points and a temperature-matched single sample by $20.7$. It
involves no checking, no second look, and no extra reasoning — it is argmax versus
sum.

Diversity is the required ingredient and it has a cost. Single-sample accuracy
fell monotonically with temperature while the vote rose, so the temperature that
is right for one answer is wrong for aggregation; coverage, which peaked and then
declined, is the signal for the upper bound.

With a verifier there are three rules and the intuitive one is the worst. Argmax
accuracy *fell* from $52.6\%$ to $38.1\%$ as diversity rose while coverage stayed
flat, because selecting the maximum of an imperfect score over a growing pool
optimises against the scorer's errors ({{eq:argmax-extreme-value}}). Weighted
voting led from a ranking accuracy of $61.2\%$ upward and only conceded to argmax
at near-perfect verifiers — but it lost to plain voting at chance, so the rule is
to measure the verifier first.

Reflection is where the correlation problem bites. Two critics with identical
confusion matrices produced different outcomes, so the variable is not competence
but where the errors fall ({{eq:recoverable-mass}}). The self-critic accepted a
correct proposal $10.6\%$ of the time on problems whose modal answer was wrong —
inverted precisely where it was needed. And revising toward one's own critique
reached $36.5\%$ against a majority vote of $36.8\%$: **intrinsic reflection
converges to self-consistency** ({{eq:reflection-is-voting}}), sequentially, for
more money, with no ceiling above it.

So the operational summary is short. Sample in parallel, aggregate by weighted
vote if your verifier beats chance and by plain vote if it does not, set the
temperature for the aggregate, and if you want criticism, buy a critic that is
wrong somewhere else.

## 21. Further Reading

{{cite:wang2023selfconsistency}} is short and worth reading for its framing:
marginalising over reasoning paths is the correct description, and everything in
the first half of this chapter follows from taking it literally.

{{cite:madaan2023selfrefine}} and {{cite:huang2024selfcorrect}} should be read
back to back, with {{sec:6-mathematical-foundation}}'s reconciliation in hand.
The disagreement is about task type, and both papers are more careful than their
headline numbers suggest.

{{cite:lightman2023verify}} is the natural next step and belongs to
{{ch:rsn-supervision}}: it is about making the verifier that this chapter kept
finding to be the binding constraint.

{{cite:brown2024monkeys}} for the coverage side, if you have not read it after
{{ch:rsn-test-time-compute}}, and {{cite:snell2024testtime}} for what search buys
over restarts.
