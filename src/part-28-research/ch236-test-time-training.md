---
id: res-test-time
number: 236
part: XXVIII
tier: full
status: draft
requires: [coverage-selection-decomposition, verifier-quality-ceiling,
           decode-is-bandwidth-bound, the-training-optimum-is-not-the-deployment-optimum]
provides: [test-time-compute-has-a-ceiling-training-does-not,
           the-axes-cross-at-a-quality-target,
           adaptation-must-amortise-over-reuse,
           per-request-adaptation-forfeits-the-batch]
citations: [snell2024testtime, brown2024monkeys, muennighoff2025s1, hu2021lora]
---

## 1. Learning Objectives

By the end of this chapter you will be able to treat training compute and test-time compute as
two priced axes rather than interchangeable resources; compute the ceiling test-time compute
imposes and show that no sampling budget crosses it; find the crossover request volume between
the two axes and explain why it is usually irrelevant; choose an adaptation *scope* by
amortising its cost over reuse; and show why full-weight test-time training forfeits batching
while small-adapter adaptation does not.

## 2. Why This Matters

{{ch:rsn-test-time-compute}} established the mechanism. This chapter asks the budgeting question
that follows: given a quality target, is it cheaper to train a better model or to think longer?

They are not interchangeable. A model trained with $3\times10^{23}$ FLOPs scores **0.7399**. To
reach 0.90 you can train **4,687×** longer or sample **11** times. To reach 0.97 you can only do
the first, because test-time compute saturates at **0.9636** — set by a verifier at 0.86 — and
no number of samples crosses it
({{eq:test-time-compute-has-a-ceiling-training-does-not}}).

Below the ceiling the two are priced in different units. Reaching 0.90 by training costs
**$1,262 million**; by sampling, **$0.000277** per request. They cross at **4.6 × 10¹²
requests** ({{eq:the-axes-cross-at-a-quality-target}}) — a volume no product reaches.

Then the other way to spend test-time compute: adapting the weights. Gain peaks not on the input
and not on the corpus but on **the session**, at **0.0779**, and adaptation compute must
amortise over reuse ({{eq:adaptation-must-amortise-over-reuse}}).

And the constraint that actually decides it is memory, not FLOPs. Rank-16 adapters cost
**1.00×** throughput; full-weight adaptation costs **591×**, because a batch can only share the
weights its members have in common
({{eq:per-request-adaptation-forfeits-the-batch}}).

## 3. Prerequisites

{{eq:coverage-selection-gap}} from {{ch:rsn-test-time-compute}} is the mechanism this
chapter budgets: coverage rises with samples, selection converts it to accuracy.

{{eq:verifier-caps-selection}} from the same chapter becomes a budgeting instruction here —
spending on the verifier raises a ceiling, spending on samples approaches one.

{{eq:decode-is-bandwidth-bound}} from {{ch:inf-cpu-gpu}} is why adapter *size* rather than
adaptation *cost* decides whether test-time training can be served.

{{eq:the-training-optimum-is-not-the-deployment-optimum}} from {{ch:res-scaling}} is the pattern
this chapter partly breaks: here the ceiling decides rather than the volume.

## 4. Intuitive Explanation

There are two ways to make a system answer better, and organisations usually treat them as the
same decision made by different teams.

You can train a better model. That moves a scaling curve, costs a large amount once, and
benefits every request forever.

Or you can spend more compute at inference — sample several answers and select among them, or
think in more steps. That costs a small amount per request, forever.

Set them side by side and price them.

A model trained with $3\times10^{23}$ FLOPs scores **0.7399** on a hard task. To reach 0.78,
train **4.1×** longer or sample **2** times. To reach 0.85: **112×** or **4** samples. To reach
0.90: **4,687×** or **11** samples. To reach 0.93: **174,187×** or **33**.

Read those columns. On this parameterisation, thinking longer is doing an enormous amount of
work — a handful of samples substitutes for four orders of magnitude of training compute.

Then the row that matters. To reach 0.97: train $1.24\times10^{34}$ FLOPs, or… nothing. Test-time
compute **saturates at 0.9636** ({{cite:brown2024monkeys}}, {{cite:snell2024testtime}}). No
number of samples crosses it, because selection is capped by the verifier and the verifier here
is 0.86.

**That asymmetry is the most important thing in the chapter.** Across most of the useful range
the two axes look like substitutes and they are not the same kind of object: *training moves a
curve; test-time compute moves along one.* One has a ceiling and the other does not.

Where does the useful range end? Look at the returns. One sample, 0.7399. Two, 0.8102 — 31.4% of
everything test-time compute can ever deliver. Four, 53.0%. Sixteen, **77.9%**. Sixty-four,
89.6%. A thousand, 97.7%. Four thousand, 98.9%.

**The first sixteen samples deliver 78% of the total available gain.** That is
{{eq:coverage-selection-gap}}'s log-linear coverage seen against its own asymptote, and
it says the interesting operating points are all small. A budget of thousands of samples is
buying the last two percent of a bounded quantity, at a cost that grows linearly.

Now the cost comparison, and it is more one-sided than the framing suggests.

Reaching 0.90 by training costs **$1,262 million**. Reaching it by sampling costs **$0.000277**
per request. Those cross at **4.6 × 10¹² requests** — four and a half trillion. At 0.93 the
crossover is 5.7 × 10¹³.

No product reaches those volumes. So for every target test-time compute can reach, thinking
longer is cheaper than training longer, by orders of magnitude, at any realistic volume
({{eq:the-axes-cross-at-a-quality-target}}).

**The reason to train a better model is not that it is cheaper. It is that above 0.9636 there is
nothing else on the menu.**

That is a different conclusion from {{ch:res-scaling}}'s and {{ch:res-moe}}'s, which both turned
on serving volume. Here volume barely matters and the *ceiling* decides.

Which makes the last table of the first listing the one to act on. Raise the verifier from 0.55
to 0.95 and the ceiling moves from **0.8830** to **0.9870** — a gain of 0.2470 over the base,
against 0.1430 at the weak verifier. A perfect verifier would put the ceiling at 1.0000.

**The verifier, not the sampler, sets what test-time compute is worth.** Spending on a better
verifier raises a ceiling; spending on more samples only approaches one. That is
{{eq:verifier-caps-selection}} converted from an observation into a budget line.

There is a planner's version of the question too: given one pot of FLOPs, how do you split it?
Across $4\times10^{24}$ FLOPs and $3\times10^8$ requests, the best split spends **50% on
training**, affords **238 samples per request**, and reaches **0.9654** — against **0.8086** for
spending everything on training. The optimum is interior, and it is skewed toward training
because the test-time term is multiplied by the request count and the training term is not.

So much for spending test-time compute on sampling. The other way to spend it is on *training* —
updating parameters using the test input, the session, or the user, and then answering. That is
test-time training proper, and it fails for reasons that have nothing to do with the ones you
would expect.

Start with what to adapt on. Adaptation data has two properties that pull against each other:
**volume** and **relevance**. The request itself is perfectly relevant and 2,000 tokens long.
The whole corpus is 50 million tokens and only 0.45 relevant.

Gain peaks in between. Adapting on the request: **0.0414**. On the session: **0.0779**. On the
user's history: **0.0663**. On the whole corpus: **0.0383**.

**The best thing to adapt on is neither the input nor everything**, which is not where either
camp's intuition points. The striking research framing says adapt on the test input; the
familiar production framing says fine-tune on your corpus; the arithmetic says the session.

Now cost. Adapting on the request itself costs **6.9%** of an inference and is reused exactly
once. Adapting on the session costs more absolutely but is reused 40 times, so it costs **0.6%**
per request. On the user's history, **0.1%**. On the corpus, effectively zero — and that row is
just fine-tuning under a different name.

State it as a rule: **an adaptation must amortise over its reuse count**
({{eq:adaptation-must-amortise-over-reuse}}). At a 5% overhead budget, the request scope needs
1.4 reuses and gets one; the session needs 5.1 and gets 40; the user needs 68.6 and gets 3,000.

So on compute alone, test-time training looks affordable, and even the per-request version is a
7% overhead. **Compute is not the constraint.**

The constraint is batching, and it is the same one {{ch:res-moe}} found from the other side.

A serving step reads the weights that every member of the batch shares. If members use different
weights, they cannot share the read, and batching is where essentially all serving efficiency
comes from.

Price it. Base weights are 40 GB in 80 GB of memory. With shared weights, a batch of 512 runs at
**24,750** tokens per second. With rank-16 adapters ({{cite:hu2021lora}}) — 40 MB each, 1,000 of
them fit alongside the base — throughput is **24,750**, exactly unchanged. With rank-256
adapters, 640 MB each, only 62 fit, and throughput falls to **2,607** — **0.11×**.

With full-weight adaptation, each adapted copy is 40 GB, exactly **one** fits, the effective
batch is **1**, and throughput is **42** tokens per second.

**A factor of 591, and none of it is arithmetic**
({{eq:per-request-adaptation-forfeits-the-batch}}).

That is the result to carry. **Test-time training is affordable exactly to the extent that what
you adapt is small.** The learning algorithm barely matters; the parameter count of the thing
being updated decides whether the technique can be served at all.

One more term, and it closes a loop. Adaptations go stale, because they are fitted to a
distribution that moves. After 1,000 reuses, 84% of the gain remains. After 3,000 — which is
exactly the user scope's amortisation window — **53% of the distribution has drifted and 62% of
the gain remains.** After 10,000, 34%.

**The reuse that pays for the adaptation is the same reuse that erodes it.** The amortisation
window and the freshness window are the same interval pulling in opposite directions, which
means there is an optimal refresh rate rather than a maximal reuse count — and the user scope's
apparent advantage on overhead is bought back by staleness: raw gain 0.0663, after staleness
**0.0411**.

Put the three constraints together and the answer is unglamorous. The session scope wins on raw
gain at **0.0779**, keeps **0.0774** after staleness because its reuse count is small, carries a
**0.6%** overhead, and batches fine with small adapters.

It is also the scope nobody markets, because it is neither the striking research result nor the
familiar production one.

## 5. Formal Explanation

**Two axes.** Write accuracy on the training axis as $a_T(C) = A - \alpha C^{-\gamma}$,
unbounded above only by $A$. On the test-time axis, coverage after $s$ samples is
$1 - (1-a)^{1 + \beta\log_2 s}$ and selection converts a fraction $v$ of coverage into accuracy,
giving $a_S(s) = a + v\,(\text{cov}(s) - a)$. As $s \to \infty$, $\text{cov} \to 1$ and
$a_S \to a + v(1-a)$: a hard ceiling at the verifier's quality.

**The crossing.** Training cost is $c_T$, paid once; test-time cost is $c_S$ per request. For a
target $t$ reachable on both axes, the volumes cross at $R^\star = c_T(t)/c_S(t)$. Because
$c_T$ grows as $t \to A$ like $\left((A-t)/\alpha\right)^{-1/\gamma}$ while $c_S$ grows only
linearly in $s$ and $s$ grows exponentially in the coverage demanded, $R^\star$ is large for
every reachable $t$ — the axes cross far outside any real request volume.

**Adaptation amortisation.** With adaptation cost $F_a$ over $r$ reuses and inference cost
$F_i$, the per-request overhead is $F_a/(rF_i)$. Since $F_a$ grows with the adaptation data and
$r$ grows with the scope's breadth, the overhead falls monotonically as scope widens — while
relevance, and hence gain, falls too. Both are monotone in scope and in opposite directions,
which is why the optimum is interior.

**Batching.** With base weights $W_b$, adapter size $W_a$ and memory $M$, the number of distinct
adapters that fit is $\lfloor (M - W_b)/W_a \rfloor$, and the effective batch is bounded by it.
Throughput is $\min(B/(\text{read}/\text{BW}),\ \text{FLOPS}/(2N))$; as $W_a \to W_b$ the fit
goes to 1 and throughput falls by roughly the batch size.

## 6. Mathematical Foundation

One axis is bounded and the other is not:

$$\lim_{s \to \infty} a_S(s) = a + v(1-a) = 0.9636, \qquad \sup_C a_T(C) = A = 0.980$$ (eq:test-time-compute-has-a-ceiling-training-does-not)

The first 16 samples deliver **77.9%** of everything the test-time axis can supply.

They are priced in different units, and cross far away:

$$R^\star = \frac{c_T(t)}{c_S(t)} = \frac{\$1{,}262\text{M}}{\$0.000277} = 4.6 \times 10^{12} \ \text{requests at } t = 0.90$$ (eq:the-axes-cross-at-a-quality-target)

Adaptation must pay for itself over reuse:

$$\text{overhead} = \frac{F_a}{r\,F_i} = 6.9\% \ (r=1), \ 0.6\% \ (r=40), \ 0.1\% \ (r=3000)$$ (eq:adaptation-must-amortise-over-reuse)

And the batch can only share what its members have in common:

$$B_{\text{eff}} \le \left\lfloor \frac{M - W_b}{W_a} \right\rfloor: \quad 24{,}750 \to 42 \ \text{tokens/s}, \ \text{a factor of } 591$$ (eq:per-request-adaptation-forfeits-the-batch)

## 7. Internal Mechanics

Why does the test-time axis have a ceiling at all? Because it does not create capability, it
selects among capability the model already has. Coverage — the chance that *some* sample is
correct — does approach 1. But the system must choose one, and the chooser is a verifier whose
quality is fixed. At $v = 0.86$, fourteen percent of the coverage gain is thrown away no matter
how large the sample budget, which is exactly the ceiling. Training does not have this structure
because it changes the distribution being sampled rather than the selection over it.

That mechanism explains a practical asymmetry in where to invest. Improving the sampler moves
you along a curve toward a fixed point; improving the verifier moves the fixed point. And
verifiers are usually much cheaper to improve, because a verifier is a narrower object than a
generator — it needs to recognise correctness, not produce it, and on many tasks recognition is
substantially easier.

The adaptation-scope result has a mechanism worth naming because it is the same shape as several
earlier chapters. Relevance is decreasing in scope; volume is increasing; gain is a product of
something like both. Any product of one increasing and one decreasing factor has an interior
maximum, which is why the session — a scope no one proposes as a headline — comes out ahead of
both the input and the corpus.

The batching mechanism deserves the most careful statement because it is the one that kills the
technique when it is killed. Serving throughput at production batch sizes is set by weight
bytes read per token, and a batch amortises one read across all its members. That amortisation
is *conditional on the members sharing weights*. Per-request weights make the batch size one in
everything but name, and the loss is proportional to the batch you would otherwise have run —
here 512, measured as 591 after the compute bound is accounted for.

Small adapters escape this precisely because the base weights, which are the expensive read, stay
shared. The adapter bytes add to the read but do not multiply it, so a 40 MB adapter against 40
GB of base weights is a rounding error even at 512 distinct adapters. **The technique's viability
is a ratio, not an algorithm.**

Finally, staleness and amortisation share an interval, which is unusual and worth noticing. Most
amortisation arguments in this book — {{ch:res-moe}}'s break-even, {{ch:rai-regulation}}'s
evidence — improve monotonically with reuse. This one does not: the same reuse that pays for the
adaptation moves the distribution away from it. That converts a "maximise reuse" instruction into
a "find the refresh interval" one, and the optimum is where the marginal cost of re-adapting
equals the marginal gain lost to drift.

## 8. Implementation

The first listing prices the two axes.

```python {tier=A name=test-time-compute-has-a-ceiling-training-does-not}
"""Two ways to buy accuracy, and only one of them has no ceiling.

ch:rsn-test-time-compute established the mechanism: sample more, select better, and accuracy
rises with the log of the budget until the verifier runs out (cite:brown2024monkeys,
cite:snell2024testtime). This listing asks the budgeting question that follows -- given a
quality target, is it cheaper to train a better model or to think longer?

They are not interchangeable. Training compute moves a scaling curve with no ceiling in the
range of interest; test-time compute moves along a coverage-selection curve that saturates at
the verifier's quality (eq:test-time-compute-has-a-ceiling-training-does-not).

Above that ceiling no amount of thinking substitutes for a better model. Below it the two are
priced in different units -- one paid once, one paid per request -- and the crossing point is
the decision (eq:the-axes-cross-at-a-quality-target).
"""
import math

A_CEIL, A_SCALE, A_EXP = 0.980, 270.6, 0.130
VERIFIER = 0.86
ACTIVE, TOKENS_OUT = 2.0e10, 700
FLOP_COST = 3.20 / 3600 / 9.9e14
BASE_FLOPS = 3.0e23


def acc_train(flops):
    return A_CEIL - A_SCALE * flops ** -A_EXP


def acc_test(base, samples, verifier=VERIFIER):
    """Coverage rises with log samples; selection caps the gain at the verifier."""
    coverage = 1.0 - (1.0 - base) ** (1.0 + 0.28 * math.log(max(samples, 1), 2))
    return base + (coverage - base) * verifier


def samples_for(base, target, verifier=VERIFIER, cap=100_000):
    s = 1
    while s < cap and acc_test(base, s, verifier) < target:
        s += 1
    return s if acc_test(base, s, verifier) >= target else None


BASE_ACC = acc_train(BASE_FLOPS)
CEIL_TEST = acc_test(BASE_ACC, 10 ** 9)

print("What each axis buys on its own.")
print()
print(f"a model trained with {BASE_FLOPS:.1e} FLOPs scores {BASE_ACC:.4f}")
print()
print(f"{'accuracy target':>18}{'training FLOPs needed':>24}{'multiple of base':>19}"
      f"{'samples needed':>17}{'test FLOPs / request':>23}")
print("-" * 101)
need = {}
TARGETS = (0.78, 0.85, 0.90, 0.93, 0.95, 0.97)
for target in TARGETS:
    tf = (max(A_CEIL - target, 1e-9) / A_SCALE) ** (-1 / A_EXP)
    s = samples_for(BASE_ACC, target)
    need[target] = (tf, s)
    ss = f"{s:>17,}" if s else f"{'unreachable':>17}"
    tt = f"{2 * ACTIVE * TOKENS_OUT * s:>23.2e}" if s else f"{'--':>23}"
    print(f"{target:>18.2f}{tf:>24.2e}{tf / BASE_FLOPS:>18.1f}x{ss}{tt}")

print()
print(f"test-time compute saturates at {CEIL_TEST:.4f}, set by the verifier at {VERIFIER:.2f}")
print(f"training has no ceiling below {A_CEIL:.3f}")

print()
print()
print("Diminishing returns on the test-time axis.")
print()
print(f"{'samples':>10}{'accuracy':>12}{'gain over 1':>14}{'gain per doubling':>21}"
      f"{'share of the ceiling':>23}")
print("-" * 80)
prev = None
for s in (1, 2, 4, 16, 64, 256, 1024, 4096):
    a = acc_test(BASE_ACC, s)
    g = f"{a - prev:>20.4f}" if prev is not None else f"{'--':>21}"
    print(f"{s:>10,}{a:>12.4f}{a - BASE_ACC:>14.4f}{g}"
          f"{(a - BASE_ACC) / (CEIL_TEST - BASE_ACC):>23.1%}")
    prev = a

SHARE16 = (acc_test(BASE_ACC, 16) - BASE_ACC) / (CEIL_TEST - BASE_ACC)
print()
print(f"the first 16 samples deliver {SHARE16:.0%} of everything test-time")
print("compute can ever deliver on this model")

print()
print()
print("The two axes are priced in different units.")
print()
print(f"{'accuracy target':>18}{'train-more $M':>21}{'samples':>10}"
      f"{'think-longer $ / request':>27}{'crossover requests':>21}{'reachable?':>15}")
print("-" * 112)
cross = {}
for target in TARGETS:
    tf, s = need[target]
    train_cost = (tf - BASE_FLOPS) * FLOP_COST
    if s is None:
        print(f"{target:>18.2f}{train_cost / 1e6:>21,.1f}{'--':>10}{'--':>27}"
              f"{'--':>21}{'training only':>15}")
        continue
    per_req = 2 * ACTIVE * TOKENS_OUT * s * FLOP_COST
    cross[target] = (train_cost, s, per_req, train_cost / per_req)
    print(f"{target:>18.2f}{train_cost / 1e6:>21,.1f}{s:>10,}{per_req:>27.6f}"
          f"{train_cost / per_req:>21.2e}{'yes':>15}")

print()
print(f"at {0.90:.2f}: training costs {cross[0.90][0] / 1e6:,.1f}M, thinking costs")
print(f"{cross[0.90][2]:.6f} per request, and they cross at"
      f" {cross[0.90][3]:.1e} requests")

print()
print()
print("Splitting a fixed budget between the two.")
print()
TOTAL, REQS = 4.0e24, 3.0e8
print(f"total budget {TOTAL:.1e} FLOPs, {REQS:.0e} requests")
print()
print(f"{'share spent on training':>26}{'train FLOPs':>15}{'base accuracy':>16}"
      f"{'samples affordable':>21}{'final accuracy':>17}")
print("-" * 95)
best = None
for share in (0.20, 0.50, 0.80, 0.90, 0.95, 0.99, 0.999):
    tf = TOTAL * share
    s = max(1, int(TOTAL * (1 - share) / (2 * ACTIVE * TOKENS_OUT * REQS)))
    b = acc_train(tf)
    a = acc_test(b, s)
    if best is None or a > best[1]:
        best = (share, a, s, b)
    print(f"{share:>26.3f}{tf:>15.2e}{b:>16.4f}{s:>21,}{a:>17.4f}")

ALL_TRAIN = acc_train(TOTAL)
print()
print(f"best split: {best[0]:.1%} on training, {best[2]:,} samples per request,"
      f" reaching {best[1]:.4f}")
print(f"spending it all on training gives {ALL_TRAIN:.4f}")

print()
print()
print("What moves the ceiling.")
print()
print(f"{'verifier quality':>19}{'ceiling accuracy':>19}{'samples to 90% of it':>24}"
      f"{'headroom over base':>21}")
print("-" * 83)
ceilings = {}
for v in (0.55, 0.70, 0.86, 0.95, 1.00):
    ceil = acc_test(BASE_ACC, 10 ** 9, v)
    ceilings[v] = ceil
    s = samples_for(BASE_ACC, BASE_ACC + 0.90 * (ceil - BASE_ACC), v)
    print(f"{v:>19.2f}{ceil:>19.4f}{s:>24,}{ceil - BASE_ACC:>21.4f}")

print(f"""
The first table is the budgeting question nobody sets up explicitly. A model trained with
{BASE_FLOPS:.1e} FLOPs scores {BASE_ACC:.4f}. To reach {0.90:.2f} you can train
{need[0.90][0] / BASE_FLOPS:,.0f} times longer, or sample {need[0.90][1]:,} times per request.
Both work.

To reach {0.97:.2f} only one of them works. Test-time compute saturates at **{CEIL_TEST:.4f}**,
set by the verifier's quality at {VERIFIER:.2f}, and no number of samples crosses it
(eq:test-time-compute-has-a-ceiling-training-does-not). Training has no ceiling below
{A_CEIL:.3f}.

That asymmetry is the most important thing in this listing. Across most of the useful range the
two axes look like substitutes, and they are not the same kind of object: **one moves a curve,
the other moves along one.**

The diminishing-returns table says where the useful range ends. The first 16 samples deliver
**{SHARE16:.0%}** of everything test-time compute can ever deliver on this model, and 1,024
deliver {(acc_test(BASE_ACC, 1024) - BASE_ACC) / (CEIL_TEST - BASE_ACC):.0%}. That is
ch:rsn-test-time-compute's `coverage-log-linear` seen against its own asymptote: the interesting
operating points are all small, and a budget of thousands of samples is buying the last two
percent of a bounded quantity.

The cost table is where the decision gets made, and the numbers are more one-sided than the
framing suggests. Reaching {0.90:.2f} by training costs **${cross[0.90][0] / 1e6:,.0f} million**; reaching it by
sampling costs **{cross[0.90][2]:.6f} per request**. Those cross at
**{cross[0.90][3]:.1e} requests** -- a volume no product reaches
(eq:the-axes-cross-at-a-quality-target).

So for every target test-time compute can reach, thinking longer is cheaper than training
longer, by orders of magnitude, at any realistic volume. **The reason to train a better model is
not that it is cheaper. It is that above {CEIL_TEST:.4f} there is nothing else on the menu.**

That is a genuinely different conclusion from ch:res-scaling's and ch:res-moe's, which both
turned on serving volume. Here volume barely matters and the *ceiling* decides -- which is why
the last table is the one to act on.

The joint-budget table confirms the shape at a planner's level of abstraction. Splitting
{TOTAL:.1e} FLOPs across {REQS:.0e} requests, the best split spends **{best[0]:.1%} on training**
and affords {best[2]:,} samples per request, reaching {best[1]:.4f} -- against {ALL_TRAIN:.4f}
for spending it all on training. The optimum is interior and it is skewed heavily toward
training, because the test-time term is multiplied by the request count and the training term is
not.

The last table is the lever everything rests on. Raising the verifier from {0.55:.2f} to
{0.95:.2f} moves the ceiling from {ceilings[0.55]:.4f} to {ceilings[0.95]:.4f} -- a gain of
{ceilings[0.95] - ceilings[0.55]:.4f}, larger than anything the sampling budget can buy at a
fixed verifier.

**The verifier, not the sampler, sets what test-time compute is worth.** That is
ch:rsn-test-time-compute's `verifier-quality-ceiling` restated as a budgeting instruction:
spending on a better verifier raises a ceiling, and spending on more samples only approaches
one.""")
```

## 9. Practical Example

Two ways to reach a target:

```
   accuracy target   training FLOPs needed   multiple of base   samples needed   test FLOPs / request
-----------------------------------------------------------------------------------------------------
              0.78                1.22e+24               4.1x                2               5.60e+13
              0.85                3.36e+25             111.9x                4               1.12e+14
              0.90                1.41e+27            4686.6x               11               3.08e+14
              0.93                5.23e+28          174186.9x               33               9.24e+14
              0.95                2.66e+30         8862258.6x              172               4.82e+15
              0.97                1.24e+34     41467506033.8x      unreachable                     --
```

**Test-time compute saturates at 0.9636; training has no ceiling below 0.980**
({{eq:test-time-compute-has-a-ceiling-training-does-not}}).

```
   samples    accuracy   gain over 1    gain per doubling   share of the ceiling
--------------------------------------------------------------------------------
         1      0.7399        0.0000                   --                   0.0%
         2      0.8102        0.0703               0.0703                  31.4%
        16      0.9141        0.1742               0.0557                  77.9%
       256      0.9526        0.2127               0.0123                  95.1%
     4,096      0.9612        0.2212               0.0027                  98.9%
```

**The first 16 samples deliver 77.9%** of everything the axis can supply.

```
   accuracy target        train-more $M   samples   think-longer $ / request   crossover requests     reachable?
----------------------------------------------------------------------------------------------------------------
              0.78                  0.8         2                   0.000050             1.65e+10            yes
              0.85                 29.9         4                   0.000101             2.97e+11            yes
              0.90              1,262.1        11                   0.000277             4.56e+12            yes
              0.93             46,918.8        33                   0.000830             5.66e+13            yes
              0.97     11,169,698,594.7        --                         --                   --  training only
```

**They cross at 4.6 × 10¹² requests** ({{eq:the-axes-cross-at-a-quality-target}}) — beyond any
product.

```
   share spent on training    train FLOPs   base accuracy   samples affordable   final accuracy
-----------------------------------------------------------------------------------------------
                     0.200       8.00e+23          0.7687                  380           0.9617
                     0.500       2.00e+24          0.7924                  238           0.9654
                     0.900       3.60e+24          0.8062                   47           0.9599
                     0.999       4.00e+24          0.8086                    1           0.8086

   verifier quality   ceiling accuracy    samples to 90% of it   headroom over base
-----------------------------------------------------------------------------------
               0.55             0.8830                      69               0.1430
               0.86             0.9636                      69               0.2236
               0.95             0.9870                      69               0.2470
               1.00             1.0000                      69               0.2600
```

**The verifier moves the ceiling; the sampler only approaches it.**

The second listing prices adaptation.

```python {tier=A name=adaptation-must-amortise-over-reuse}
"""Adapting at test time is cheap in FLOPs and expensive in batching.

The first listing spent test-time compute on sampling. The other way to spend it is on
*training*: update the parameters using the test input, the session, or the user, and then
answer.

Two things decide whether that is a good idea, and neither is the obvious one. The adaptation
compute has to amortise over however many requests reuse the adapted weights, which is a scope
decision rather than an algorithmic one
(eq:adaptation-must-amortise-over-reuse).

And whatever is adapted has to fit in memory *per distinct adapter*, because a batch can only
share the weights its members have in common. Small adapters (cite:hu2021lora) survive this;
full-weight adaptation does not, and batching is where all serving efficiency comes from
(eq:per-request-adaptation-forfeits-the-batch).
"""
import math

ACTIVE, TOKENS_OUT, BYTES = 2.0e10, 700, 2
LORA_PARAMS = 2.0e7
G_MAX, TAU = 0.085, 3.0e3
HBM_BW, FLOPS_S, HBM_GB = 3.35e12, 9.9e14, 80.0
INFER_FLOPS = 2 * ACTIVE * TOKENS_OUT

# (scope, tokens adapted on, passes, reuse count, relevance of that data)
SCOPES = [
    ("no adaptation",     0,          0, 1,           0.00),
    ("the request itself", 2_000,     8, 1,           1.00),
    ("the session",        20_000,    3, 40,          0.92),
    ("the user's history", 400_000,   2, 3_000,       0.78),
    ("the whole corpus",   50_000_000, 1, 1_000_000_000, 0.45),
]


def gain(tokens, relevance):
    if tokens == 0:
        return 0.0
    tau = TAU / max(relevance, 1e-6) ** 2
    return G_MAX * relevance * (1.0 - math.exp(-tokens / tau))


print("Where the adaptation data comes from, and what it is worth.")
print()
print(f"{'adapted on':>22}{'tokens':>14}{'passes':>9}{'relevance':>12}"
      f"{'accuracy gain':>16}{'reuses':>17}")
print("-" * 90)
rows = {}
for name, tok, passes, reuse, rel in SCOPES:
    g = gain(tok, rel)
    flops = 6 * LORA_PARAMS * tok * passes
    rows[name] = (tok, passes, reuse, rel, g, flops)
    print(f"{name:>22}{tok:>14,}{passes:>9}{rel:>12.2f}{g:>16.4f}{reuse:>17,}")

BEST_GAIN = max((n for n in rows if n != "no adaptation"), key=lambda n: rows[n][4])
print()
print(f"largest gain: {BEST_GAIN} at {rows[BEST_GAIN][4]:.4f}")
print("relevance and volume pull in opposite directions, and the optimum is neither end")

print()
print()
print("Adaptation compute, amortised over whatever reuses it.")
print()
print(f"{'adapted on':>22}{'adapt FLOPs':>15}{'reuses':>16}{'FLOPs / request':>18}"
      f"{'share of inference':>21}{'gain per 1% overhead':>23}")
print("-" * 115)
amort = {}
for name, tok, passes, reuse, rel in SCOPES:
    _, _, _, _, g, flops = rows[name]
    per_req = flops / reuse
    share = per_req / INFER_FLOPS
    eff = g / max(share * 100, 1e-9)
    amort[name] = (per_req, share, eff)
    es = f"{eff:>23.4f}" if name != "no adaptation" else f"{'--':>23}"
    print(f"{name:>22}{flops:>15.2e}{reuse:>16,}{per_req:>18.2e}{share:>20.2%}{es}")

print()
print("the last column degenerates as overhead approaches zero, which is the")
print("corpus row's whole story: it is fine-tuning, and fine-tuning is free per request")
print(f"adapting on the request itself costs {amort['the request itself'][1]:.1%}"
      f" of an inference for {rows['the request itself'][4]:.4f}")

print()
print()
print("How many reuses each scope needs before the overhead is under 5%.")
print()
print(f"{'adapted on':>22}{'adapt FLOPs':>15}{'reuses for 5%':>17}"
      f"{'actual reuses':>16}{'amortises?':>13}")
print("-" * 83)
for name, tok, passes, reuse, rel in SCOPES:
    if name == "no adaptation":
        continue
    _, _, _, _, g, flops = rows[name]
    need = flops / (0.05 * INFER_FLOPS)
    print(f"{name:>22}{flops:>15.2e}{need:>17,.1f}{reuse:>16,}"
          f"{('yes' if reuse >= need else 'no'):>13}")

print()
print("(eq:adaptation-must-amortise-over-reuse)")

print()
print()
print("Now the constraint that actually decides it: batching.")
print()
BASE_GB = ACTIVE * BYTES / 1e9
print(f"base weights {BASE_GB:.0f} GB, {HBM_GB:.0f} GB of memory")
print()
MECHS = [
    ("none, shared weights",   0.0,            1),
    ("LoRA rank 16",           LORA_PARAMS,    1),
    ("LoRA rank 256",          LORA_PARAMS * 16, 1),
    ("full-weight adaptation", ACTIVE,         1),
]
print(f"{'adaptation mechanism':>24}{'bytes per adapter':>20}{'adapters that fit':>20}"
      f"{'effective batch':>18}{'tokens/s':>13}{'relative':>11}")
print("-" * 106)
WANT_BATCH = 512
thr = {}
SHARED_TPS = None
for name, params, _ in MECHS:
    ad_bytes = params * BYTES
    room = HBM_GB * 1e9 - BASE_GB * 1e9
    fit = int(room / ad_bytes) if ad_bytes > 0 else WANT_BATCH
    batch = min(WANT_BATCH, max(1, fit))
    read = BASE_GB * 1e9 + ad_bytes * (batch if ad_bytes > 0 else 0)
    mem = batch / (read / HBM_BW)
    comp = FLOPS_S / (2 * ACTIVE)
    tps = min(mem, comp)
    thr[name] = tps
    if SHARED_TPS is None:
        SHARED_TPS = tps
    print(f"{name:>24}{ad_bytes:>20,.0f}{fit:>20,}{batch:>18,}"
          f"{tps:>13,.0f}{tps / SHARED_TPS:>10.2f}x")

SHARED = thr["none, shared weights"]
FULL = thr["full-weight adaptation"]
print()
print(f"shared weights: {SHARED:,.0f} tokens/s; full-weight adaptation:"
      f" {FULL:,.0f}")
print(f"a factor of {SHARED / FULL:,.0f}, and none of it is arithmetic")

print()
print()
print("And adaptations go stale, so the reuse count is not free either.")
print()
print(f"{'requests since adapting':>26}{'drift':>10}{'gain retained':>16}"
      f"{'effective gain':>17}")
print("-" * 69)
G0 = rows["the user's history"][4]
retained = {}
for r in (1, 10, 100, 1_000, 10_000, 100_000):
    drift = 1.0 - math.exp(-r / 4_000.0)
    keep = 1.0 - 0.72 * drift
    retained[r] = keep
    print(f"{r:>26,}{drift:>10.3f}{keep:>16.3f}{G0 * keep:>17.4f}")

print()
print(f"after {3_000:,} reuses -- the user scope's amortisation window --")
print(f"{1.0 - math.exp(-3_000 / 4_000.0):.2f} of the distribution has drifted and"
      f" {1.0 - 0.72 * (1.0 - math.exp(-3_000 / 4_000.0)):.2f} of the gain remains")

print()
print()
print("Putting the three constraints together.")
print()
print(f"{'adapted on':>22}{'raw gain':>11}{'after staleness':>18}"
      f"{'overhead':>11}{'distinct adapters in a 512 batch':>34}{'net verdict':>28}")
print("-" * 124)
DISTINCT = {
    "the request itself": 512,
    "the session":        512,
    "the user's history": 512,
    "the whole corpus":   1,
}
VERDICTS = {
    "the request itself": "no reuse, but batchable",
    "the session":        "the practical sweet spot",
    "the user's history": "amortises, then goes stale",
    "the whole corpus":   "this is just fine-tuning",
}
for name, tok, passes, reuse, rel in SCOPES:
    if name == "no adaptation":
        continue
    g = rows[name][4]
    drift = 1.0 - math.exp(-reuse / 4_000.0)
    net = g * (1.0 - 0.72 * drift)
    print(f"{name:>22}{g:>11.4f}{net:>18.4f}{amort[name][1]:>10.1%}"
          f"{DISTINCT[name]:>34,}{VERDICTS[name]:>28}")

print(f"""
The first table separates two things that get conflated. Adaptation data has a *volume* and a
*relevance*, and they move in opposite directions. The request itself is perfectly relevant and
{2_000:,} tokens long. The whole corpus is {50_000_000:,} tokens and only {0.45:.2f} relevant.

The gain peaks in between: `{BEST_GAIN}` at **{rows[BEST_GAIN][4]:.4f}**, against
{rows['the request itself'][4]:.4f} for the request alone and
{rows['the whole corpus'][4]:.4f} for the corpus. **The best thing to adapt on is neither the
input nor everything**, which is not where either camp's intuition points.

The amortisation table prices it. Adapting on the request itself costs
{amort['the request itself'][1]:.1%} of an inference and is reused exactly once; adapting on the
corpus costs {rows['the whole corpus'][5]:.2e} FLOPs and is reused a billion times, so its
per-request overhead is {amort['the whole corpus'][1]:.4%} -- effectively free, and it is also
just fine-tuning under a different name.

The threshold table states the rule directly (eq:adaptation-must-amortise-over-reuse). Every
scope clears a 5% overhead budget except adapting on the request itself, which needs
{rows['the request itself'][5] / (0.05 * INFER_FLOPS):,.1f} reuses and gets exactly one.

So on compute alone, test-time training looks affordable and even the per-request version is
only a {amort['the request itself'][1]:.0%} overhead. **Compute is not the constraint.**

The batching table is (eq:per-request-adaptation-forfeits-the-batch). A serving step reads the
weights every batch member shares, so a batch can only be as wide as the number of members using
the same weights. With cite:hu2021lora-style adapters that is fine: a rank-16 adapter is
{LORA_PARAMS * BYTES / 1e6:.0f} MB, so {int((HBM_GB * 1e9 - BASE_GB * 1e9) / (LORA_PARAMS * BYTES)):,}
of them fit alongside the base weights and the batch is unaffected.

With full-weight adaptation it is not fine. Each adapted copy is {BASE_GB:.0f} GB, exactly one
fits, and the effective batch is **1**. Throughput falls from {SHARED:,.0f} tokens per second to
{FULL:,.0f} -- **a factor of {SHARED / FULL:,.0f}, none of it arithmetic.**

That is the result to carry out of this listing. **Test-time training is affordable exactly to
the extent that what you adapt is small.** The algorithm barely matters; the parameter count of
the thing being updated decides whether the technique can be served at all, and it is the same
constraint ch:res-moe found from the other direction.

The staleness table adds the third term. An adaptation is fitted to a distribution that moves.
After {3_000:,} reuses -- which is exactly the user scope's amortisation window --
{1.0 - math.exp(-3_000 / 4_000.0):.0%} of the distribution has drifted and
{1.0 - 0.72 * (1.0 - math.exp(-3_000 / 4_000.0)):.0%} of the gain remains.

**The reuse that pays for the adaptation is the same reuse that erodes it**, which means the
amortisation window and the freshness window are the same interval pulling in opposite
directions, and there is an optimal refresh rather than a maximal one.

The summary table is the practical answer. `the session` wins on raw gain, survives staleness
almost intact because its reuse count is small, carries a
{amort['the session'][1]:.1%} overhead, and batches fine. It is also the scope nobody markets,
because it is neither the striking research result nor the familiar production one.""")
```

```
            adapted on        tokens   passes   relevance   accuracy gain           reuses
------------------------------------------------------------------------------------------
    the request itself         2,000        8        1.00          0.0414                1
           the session        20,000        3        0.92          0.0779               40
    the user's history       400,000        2        0.78          0.0663            3,000
      the whole corpus    50,000,000        1        0.45          0.0383    1,000,000,000
```

**Gain peaks on the session** — neither the input nor everything.

```
            adapted on    adapt FLOPs    reuses for 5%   actual reuses   amortises?
-----------------------------------------------------------------------------------
    the request itself       1.92e+12              1.4               1           no
           the session       7.20e+12              5.1              40          yes
    the user's history       9.60e+13             68.6           3,000          yes
```

({{eq:adaptation-must-amortise-over-reuse}})

```
    adaptation mechanism   bytes per adapter   adapters that fit   effective batch     tokens/s   relative
----------------------------------------------------------------------------------------------------------
    none, shared weights                   0                 512               512       24,750      1.00x
            LoRA rank 16          40,000,000               1,000               512       24,750      1.00x
           LoRA rank 256         640,000,000                  62                62        2,607      0.11x
  full-weight adaptation      40,000,000,000                   1                 1           42      0.00x
```

**A factor of 591, and none of it is arithmetic**
({{eq:per-request-adaptation-forfeits-the-batch}}).

```
   requests since adapting     drift   gain retained   effective gain
---------------------------------------------------------------------
                     1,000     0.221           0.841           0.0557
                    10,000     0.918           0.339           0.0225

            adapted on   raw gain   after staleness   overhead  distinct adapters in a 512 batch                 net verdict
----------------------------------------------------------------------------------------------------------------------------
    the request itself     0.0414            0.0414      6.9%                               512     no reuse, but batchable
           the session     0.0779            0.0774      0.6%                               512    the practical sweet spot
    the user's history     0.0663            0.0411      0.1%                               512  amortises, then goes stale
      the whole corpus     0.0383            0.0107      0.0%                                 1    this is just fine-tuning
```

## 10. Production Considerations

Measure your test-time ceiling before budgeting either axis. It is a verifier property and it
tells you which targets are reachable at all.

Spend on the verifier before spending on samples. From 0.55 to 0.95 moves the ceiling by 0.1040
of headroom; no sampling budget does that.

Operate at small sample counts. Sixteen samples deliver 77.9% of the available gain, and a
thousand delivers 97.7% at 64× the cost.

Do not compute a training-versus-thinking crossover expecting it to bind. It sits at 10¹²–10¹⁴
requests; the ceiling is the real constraint.

Choose an adaptation scope by amortisation, not by novelty. Per-request adaptation needs 1.4
reuses and gets one.

Keep adapters small — the technique's viability is a ratio of adapter bytes to base bytes, not a
property of the learning algorithm.

Never ship full-weight test-time adaptation to a batched serving path. It costs 591× throughput
and no amount of engineering recovers it.

Refresh adaptations on an interval, not at a reuse cap. Amortisation and staleness share the
same window and pull opposite ways.

## 11. Common Mistakes

**Treating training and test-time compute as substitutes.** One has a ceiling.

**Budgeting thousands of samples.** The last two percent of a bounded quantity, priced linearly.

**Improving the sampler when the verifier is the ceiling.** Approaches a fixed point instead of
moving it.

**Adapting on the test input alone.** Perfectly relevant, 2,000 tokens, and reused once.

**Fine-tuning on everything and calling it adaptation.** 0.45 relevance and 0.0383 gain.

**Measuring adaptation cost in FLOPs.** Compute is 0.6%; the batching loss is 591×.

**Maximising reuse of an adaptation.** After 3,000 reuses, 62% of the gain remains.

## 12. Failure Modes

**A quality target above the ceiling.** No sampling budget reaches 0.97, and the team keeps
raising $k$.

**A sample budget chosen for a benchmark.** 4,096 samples in evaluation, 4 in production, and
the reported number describes neither.

**Full-weight test-time training that benchmarks well at batch 1.** 24,750 tokens per second
becomes 42 under load.

**Rank-256 adapters chosen for quality.** 62 fit, the batch collapses to 62, and throughput
falls to 0.11×.

**A per-user adaptation never refreshed.** Amortises beautifully and retains 62% of its gain.

**A verifier improved after the sampler.** The order is backwards and the sampler work was
capped the whole time.

## 13. Alternatives

**Train a better model.** The only option above the ceiling, and the honest reason to do it —
not cost.

**Improve the verifier.** Moves the ceiling from 0.8830 to 0.9870 across the range measured
here, and is usually the cheapest lever on the page.

**Distil the sampling procedure into the model.** {{cite:muennighoff2025s1}}'s direction: pay the
test-time cost once during training and serve at single-sample cost.

**Adapt at session scope with small adapters.** 0.0779 gain, 0.6% overhead, no batching cost —
the unglamorous winner in {{sec:9-practical-example}}.

**Retrieval instead of adaptation.** Injects the same relevant material through the context
rather than the weights, keeps weights shared, and is what {{ch:res-memory}} recommended for
related reasons.

## 14. Evaluation

Measure your verifier's quality directly and compute the implied test-time ceiling. Every
sampling decision depends on it.

Sweep sample count on a log grid and report the share-of-ceiling column, not just accuracy. It
tells you where to operate.

Report evaluation and production sample budgets together. A benchmark run at 4,096 samples does
not describe a product served at 4.

Measure adaptation gain at each scope on your own data, and measure relevance separately from
volume. The interior optimum depends on both.

Benchmark adapted serving at production batch size, never at batch 1. That is where the 591×
lives.

Track adaptation age and measure retained gain against it. The refresh interval is an
empirical quantity.

## 15. Advanced Concepts

The ceiling argument assumes the verifier is fixed and independent of the sampler, and neither
holds exactly. A verifier applied to samples from a stronger model faces a harder discrimination
problem — the wrong answers are more plausible — so effective $v$ falls as the generator
improves. That makes the ceiling *move down* as training improves the model, partially cancelling
the gain, and it predicts that the value of test-time compute peaks at some model quality rather
than growing with it. Nothing here measures that, and it is the most consequential missing
number in the chapter.

The sample-budget analysis treats samples as independent draws, which understates what
structured search can do. Sampling with different decompositions, or a search that conditions
later attempts on earlier failures, produces coverage that grows faster than the independent
model and — more importantly — can exceed the independent ceiling, because a sequentially
informed sampler is doing something the verifier's selection is not. The independence assumption
therefore makes {{sec:9-practical-example}}'s ceiling a lower bound on what test-time compute can
achieve, though not on what parallel sampling can.

There is an economic asymmetry between the axes that this chapter's cost model does not capture.
Training compute is capital: spent once, on a schedule, by a research organisation. Test-time
compute is operating expense: spent continuously, by a serving organisation, and *visible in a
monthly bill*. The crossover at 10¹² requests says the two are barely comparable on cost, but a
finance function that treats them differently will make a different choice than the arithmetic
suggests — which is the same governance observation {{ch:res-moe}} ended on, arriving from a
different direction.

Finally, the adaptation analysis assumes updating weights and providing context are alternatives.
They are, mechanically, but they differ in one respect this chapter does not price: an adaptation
persists and a context does not. That makes weight adaptation the only mechanism here that can
accumulate across sessions without paying for the accumulated material at every request — and
it makes staleness and {{ch:rai-privacy}}'s deletion problem apply to it, since a fact learned
into weights is subject to {{eq:deletion-is-a-product-over-derived-artefacts}}'s absorbing term.
**Per-user adaptation is a per-user model, with everything that implies.**

## 16. Connection to Previous Chapters

{{eq:coverage-selection-gap}} from {{ch:rsn-test-time-compute}} is the curve this
chapter budgets against; the addition here is its asymptote, and that 16 samples reach **77.9%**
of it.

{{eq:verifier-caps-selection}} from the same chapter becomes the chapter's central budgeting
instruction: verifier 0.55 to 0.95 moves the ceiling from **0.8830** to **0.9870**.

{{eq:decode-is-bandwidth-bound}} from {{ch:inf-cpu-gpu}} is why adapter *size* decides
feasibility — **591×** for full weights, **1.00×** for rank 16.

{{eq:the-training-optimum-is-not-the-deployment-optimum}} from {{ch:res-scaling}} is the pattern
this chapter breaks: the crossover exists at **4.6 × 10¹²** requests and never binds, so the
ceiling decides instead of the volume.

## 17. Exercises

1. Measure your verifier's selection accuracy and compute the implied test-time ceiling for your
   task.

2. Sweep sample count and find where you reach 80% of your ceiling. How far is that from your
   production setting?

3. Compute the training-versus-sampling crossover for a target you currently miss. Does it bind?

4. Estimate gain, relevance and reuse for three adaptation scopes on your workload, and find the
   interior optimum.

5. Benchmark your serving path with 1, 64 and 512 distinct adapters at fixed batch size. Where
   does throughput break?

6. Model the verifier degradation of {{sec:15-advanced-concepts}} — $v$ falling as the generator
   improves — and find the model quality at which test-time compute's value peaks.

## 18. Interview Questions

1. Should we train a better model or sample more?

2. We are sampling 1,024 times per request. What is that buying?

3. Our accuracy target is above what sampling can reach. How would you know that in advance?

4. Where should the next dollar go: the sampler or the verifier?

5. We want to adapt the model to each user. What does that cost?

6. Why would test-time training that benchmarks well collapse in production?

## 19. Research Questions

1. How does verifier quality degrade as the generator it judges improves, measured on the same
   task?

2. Can sequentially informed search exceed the independent-sampling ceiling, and by how much?

3. What is the empirical relevance-versus-volume curve for adaptation data across scopes?

4. How quickly does a per-user adaptation go stale in production, and what is the optimal refresh
   interval?

## 20. Chapter Summary

Test-time compute and training compute are two priced axes, and the interesting fact about them
is that only one has a ceiling.

A model at $3\times10^{23}$ FLOPs scores **0.7399**. Reaching 0.90 costs **4,687×** the training
compute or **11** samples; reaching 0.97 costs $1.24\times10^{34}$ FLOPs or is impossible,
because the test-time axis saturates at **0.9636** — the verifier's quality, not the sampler's
budget ({{eq:test-time-compute-has-a-ceiling-training-does-not}}). Sixteen samples deliver
**77.9%** of everything that axis can supply, so the interesting operating points are all small.

Below the ceiling the axes barely compete. Reaching 0.90 costs **$1,262 million** to train and
**$0.000277** per request to think, crossing at **4.6 × 10¹² requests**
({{eq:the-axes-cross-at-a-quality-target}}). **The reason to train a better model is not cost —
it is that above the ceiling there is nothing else on the menu**, which makes verifier quality
the lever: 0.55 to 0.95 moves the ceiling from **0.8830** to **0.9870**.

Spending test-time compute on adaptation instead gives a different set of constraints and the
same shape of answer. Gain peaks on the **session** at **0.0779** — not the input (**0.0414**),
not the corpus (**0.0383**) — because relevance and volume pull opposite ways. Adaptation must
amortise: **6.9%** overhead at one reuse, **0.6%** at forty
({{eq:adaptation-must-amortise-over-reuse}}).

And the constraint that actually decides it is not compute. A batch shares only the weights its
members have in common, so rank-16 adapters cost **1.00×** throughput and full-weight adaptation
costs **591×** — 24,750 tokens per second down to 42
({{eq:per-request-adaptation-forfeits-the-batch}}). **Test-time training is affordable exactly
to the extent that what you adapt is small.** Then staleness closes the loop: the reuse that pays
for an adaptation is the reuse that erodes it, and the user scope's 0.0663 becomes **0.0411**.

What runs through the chapter is that both halves are decided by something adjacent to the
technique. Sampling is decided by the verifier, not the sampler. Adaptation is decided by the
adapter's byte count, not the learning rule. In each case the published result improves the part
that is not binding, and the binding part is cheaper to fix.

Carry forward: **the verifier sets the ceiling**, and **adapt something small, at session
scope**.

## 21. Further Reading

- {{cite:snell2024testtime}} — allocating inference compute against model scale.
- {{cite:brown2024monkeys}} — repeated sampling, coverage, and what selection does to it.
- {{cite:muennighoff2025s1}} — spending the test-time budget during training instead.
- {{cite:hu2021lora}} — the low-rank adapters that make per-scope adaptation servable at all.
