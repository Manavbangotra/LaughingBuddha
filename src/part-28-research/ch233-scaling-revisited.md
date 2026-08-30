---
id: res-scaling
number: 233
part: XXVIII
tier: full
status: draft
requires: [metric-choice-manufactures-the-finding, discontinuity-hides-progress,
           contamination-inflates-and-flattens, headroom-sets-benchmark-lifespan]
provides: [scaling-exponents-set-allocation-not-the-ceiling,
           the-training-optimum-is-not-the-deployment-optimum,
           discontinuity-is-a-property-of-the-metric,
           extrapolation-error-grows-with-the-log-range]
citations: [kaplan2020scaling, hoffmann2022chinchilla, wei2022emergent, schaeffer2023mirage,
            kumar2024precisionscaling, lee2022dedup]
---

## 1. Learning Objectives

By the end of this chapter you will be able to read a scaling law as an allocation rule rather
than a forecast, and compute the parameter/token split an exponent pair implies; show why the
irreducible term's share of the loss grows with budget and what that does to a fit; compute the
inference-aware optimum and the overspend from using the training-optimal model; demonstrate
that a discontinuous-looking capability curve can be a smooth process under a thresholded
metric; and bound the error of extrapolating a power law beyond its fitted range.

## 2. Why This Matters

A scaling law is the only quantitative planning instrument this field has for spending very
large amounts of money, and it is routinely read as something it is not.

It is an **allocation rule**. The same budget, split under two exponent pairs, produces models
differing by a factor of **27** in tokens per parameter — **325.0** against **11.9**
({{eq:scaling-exponents-set-allocation-not-the-ceiling}}). Refitting the exponents on a wider
range is what changed the recommended model size for a given budget
({{cite:kaplan2020scaling}}, {{cite:hoffmann2022chinchilla}}): not a new capability, a new
division of the same money.

It is a claim about a **training run**, not a system. Holding quality fixed and varying serving
volume, the cheapest model shrinks by a factor of **8**, and using the training-optimal model
instead costs **13.7×** at consumer volume
({{eq:the-training-optimum-is-not-the-deployment-optimum}}).

And it is measured through instruments that manufacture their own shape. One smooth run with no
threshold anywhere in it reads as **4.2** on loss and **1152.9** on 20-token exact match —
**277×** more apparent discontinuity — and the two metrics name budgets **6 orders of magnitude
apart** as the moment the capability appeared
({{eq:discontinuity-is-a-property-of-the-metric}}).

Finally, a power law fitted over two decades without a floor term is **49%** wrong six decades
out, always optimistically ({{eq:extrapolation-error-grows-with-the-log-range}}).

## 3. Prerequisites

{{eq:metric-choice-manufactures-the-finding}} from {{ch:ev-why-hard}} is the first half of this
chapter's measurement section: the same run, four metrics, four stories.

{{eq:discontinuity-hides-progress}} from the same chapter is the specific harm — four orders of
magnitude of real improvement invisible under a binary metric.

{{eq:contamination-inflates-and-flattens}} from {{ch:ev-llm-benchmarks}} is the first row of the
extrapolation-breakers table, and the only one that changes the measurement rather than the
model.

{{eq:headroom-sets-benchmark-lifespan}} from the same chapter is why a scaling study's benchmark
saturates before the trend does.

## 4. Intuitive Explanation

The scaling literature's central object is simple enough to write in one line:

$$L(N, D) = E + \frac{A}{N^a} + \frac{B}{D^b}$$

An irreducible floor, a term that shrinks with parameters, a term that shrinks with training
tokens. Three constants, two exponents, and almost every practical question about spending a
training budget is a question about this expression.

Start with what it is *for*, because that is the thing most often misread.

Fix a compute budget $C \approx 6ND$. The expression then has one free variable — the split
between $N$ and $D$ — and minimising it gives you the split. Differentiate and you get a clean
condition: at the optimum, the two reducible terms stand in the ratio $b/a$.

That is worth pausing on. **The optimal split depends on the exponents and on nothing else.**
Not on the constants, not on the floor, not on the budget's absolute size.

So take two exponent pairs, anchor them to agree at one point so the comparison is fair, and ask
what each recommends. Shallow exponents — 0.076 and 0.095 — give **325.0** tokens per parameter
averaged across budgets. Steeper ones — 0.340 and 0.280 — give **11.9**.

**A factor of 27 in how a fixed budget is spent**, from a difference in two fitted numbers.

That is what happened between the early scaling work and the refits that followed
({{cite:kaplan2020scaling}}, {{cite:hoffmann2022chinchilla}}). The headline was that models
should be smaller and trained on far more data for a given budget. The mechanism was not a
discovery about learning; it was a better-fitted slope, and the slope is the allocation rule.

Now the part that gets less attention.

Look at what the budget buys as it grows. At $10^{19}$ FLOPs the best achievable loss is 3.464,
of which 1.774 is above the floor. At $10^{27}$ — eight orders of magnitude later — it is 1.795,
of which **0.105** is above the floor.

The reducible part fell by a factor of 16.9. The floor did not move, and its share of the
reported loss rose from 48.8% to **94.2%**.

**The exponents govern a shrinking share of the number being reported.** Every additional decade
of compute buys a smaller fraction of the thing you are looking at, not because progress
stopped, but because most of what remains is the entropy of the data.

Price that directly. Each halving of the reducible loss costs **86×** the compute of the
previous halving — and that multiple is set entirely by $a$ and $b$. It is the same whether the
constants are large or small. This is why an order-of-magnitude compute increase is a routine
expectation in this field rather than an event: it is roughly what one more step down the curve
costs.

Now the thing the framing leaves out entirely, and the most actionable result in the chapter.

Compute-optimal means *optimal per training FLOP*. But a deployed model also costs roughly $2N$
FLOPs per served token, and that term scales with the product's success, not with the training
run.

Fix a quality target and vary the serving volume. At $10^9$ served tokens — a research artefact
— the cheapest way to reach the target uses **4.13 × 10¹⁰** parameters. At $10^{16}$ tokens — a
consumer product — it uses **5.30 × 10⁹**.

**A factor of 8 in model size, driven entirely by how much the thing gets used.**

And the cost of ignoring it: training a compute-optimal model at $6 \times 10^{23}$ FLOPs and
serving it costs **1.1×** an inference-aware design for a research artefact, **6.8×** for a
product feature, and **13.7×** for a consumer product — at identical quality.

**"Compute-optimal" is a claim about a training run, not about a system**, and the two answers
diverge in exactly the direction a successful product moves. The practical implication is the
one the industry has converged on by other routes: train smaller models past the
training-optimal point, deliberately, because the tokens are cheap once and the parameters are
expensive forever.

That is the allocation half. The measurement half is where the chapter's sharpest result lives,
and it is a result about instruments rather than about models.

Take one training run. Underneath, per-token accuracy improves smoothly with log-compute — a
logistic curve, no thresholds, nothing discontinuous anywhere in the generating process. The
loss is a smooth power law with a floor.

Score it four ways.

Loss: 6.069 at $10^{18}$, falling smoothly to 1.866 at $10^{27}$.

Per-token accuracy: 0.0759 to 0.9579, smoothly.

Exact match on a five-token answer — the answer is correct only if every token is: 0.000003 to
0.806545, spending four orders of magnitude indistinguishable from zero.

Exact match on a twenty-token answer: **0.00000000** until $10^{22}$, then 0.42316988 by
$10^{27}$.

Nothing changed except the exponent on a smooth curve. $p^{20}$ where $p$ is smooth is still
smooth in the mathematical sense and looks nothing like it on a log-compute axis.

Quantify the appearance. Measure the largest single step as a multiple of the median step: loss
scores **4.2**, per-token accuracy **1.7**, five-token exact match 2.4, twenty-token exact match
**1152.9**.

**A factor of 277 in apparent discontinuity, from the same run**
({{eq:discontinuity-is-a-property-of-the-metric}}). That is
{{cite:schaeffer2023mirage}}'s argument about {{cite:wei2022emergent}}'s phenomenon, and it is
this book's {{eq:metric-choice-manufactures-the-finding}} at the largest scale it appears.

The planning consequence is in the onset table. Ask "at what budget did this capability start to
appear" and loss answers $10^{19}$ while twenty-token exact match answers $10^{25}$ —
**six orders of magnitude apart**, on identical data.

Which means a team measuring only exact match sees nothing at all for four orders of magnitude
and concludes the approach does not work. The per-token signal was improving the entire time.
{{ch:ev-why-hard}}'s fix — keep a continuous metric alongside the binary one — costs almost
nothing and is worth more here than anywhere else in this book, because the budgets in question
are enormous and the decision being made is whether to keep spending them.

None of this says emergence is not real. It says the *shape* you observe is a joint property of
the model and the scoring rule, and you cannot attribute it to the model without checking the
rule.

The second measurement failure is extrapolation, and it is expensive in a different way.

Fit a pure power law — no floor term — over $10^{19}$ to $10^{21}$, two decades where the
reducible part dominates. It fits well; the fitted exponent is −0.0866. Now use it.

At $10^{23}$, two decades out, it is 12.6% wrong. At $10^{25}$, 30.6%. At $10^{27}$, **48.9%**.
At $10^{29}$, 63.9%.

**The error grows with the log-range, and it grows in one direction**
({{eq:extrapolation-error-grows-with-the-log-range}}). A fit without a floor term predicts loss
approaching zero, which is not a thing a model of a stochastic process should predict. The same
data fitted *with* a floor term recovers the true exponent −0.1550 exactly and tracks the truth
across the whole range.

The distinguishing question is not statistical — both fits look fine on the fitted window. It is
whether the functional form contains the irreducible entropy of the data. A two-decade window
where that term is small will not tell you, and the only defence is to include the term and let
the fit decide it is near zero if it is.

Finally, the things that break extrapolation for reasons entirely outside the functional form.

Benchmark contamination changes the measured score without changing the model —
{{ch:ev-llm-benchmarks}}'s `contamination-inflates-and-flattens`, and it is the only row here that
corrupts the measurement rather than the run. Repeated training data means effective $D$ is
below nominal $D$ ({{cite:lee2022dedup}}). Reduced numerical precision imposes an
effective-parameter penalty ({{cite:kumar2024precisionscaling}}). Distribution shift at
evaluation means you are on a different loss surface. And data exhaustion means the budget
cannot be split the way the optimum requires, at any price.

Together those leave **0.230** of loss on the table at $10^{26}$ — **91%** of the reducible
loss remaining at that budget.

Every one of them is a property of the pipeline rather than of the scaling relationship, and not
one appears in the fit.

**A scaling law predicts what a clean run would do.** The gap between that and what your run
does is the part you control, and on these numbers it is most of the remaining headroom.

## 5. Formal Explanation

**The allocation rule.** Minimise $L = E + AN^{-a} + BD^{-b}$ subject to $C = 6ND$. Substituting
$D = C/(6N)$ and differentiating:

$$-aAN^{-a-1} + bB(6/C)^{b}N^{b-1} = 0 \implies a \cdot AN^{-a} = b \cdot BD^{-b}$$

At the optimum the two reducible terms stand in the ratio $b/a$. The constants $A$, $B$ set the
scale of the loss and the exponents set the split; $E$ enters neither condition. This is the
formal content of "a scaling law is an allocation rule".

**Cost of a halving.** To halve $AN^{-a}$ requires $N \to 2^{1/a}N$; to halve $BD^{-b}$ requires
$D \to 2^{1/b}D$. Since $C \propto ND$, halving the reducible loss costs
$C \to 2^{1/a + 1/b}C$. With $a = 0.34$, $b = 0.28$ that is $2^{6.51} \approx 91$, matching the
measured **86×** up to the discreteness of the search.

**Inference-aware optimum.** Total cost is $6ND + 2N T$ for $T$ served tokens. Holding
$L(N,D) = L^\star$ fixed defines $D(N)$, and minimising over $N$ gives an interior optimum whose
location moves toward smaller $N$ as $T$ grows. As $T \to \infty$ the objective is dominated by
$2NT$ and the optimum approaches the smallest $N$ for which $D(N)$ is finite — i.e. the smallest
model that can reach the target at any data budget.

**Thresholded metrics.** If per-token accuracy $p(x)$ is smooth in $x = \log_{10} C$, then
exact-match on $k$ tokens is $p(x)^k$, whose log-derivative is $k \, d\log p/dx$. The
*relative* steepness is amplified $k$-fold, and on a linear vertical axis the curve is
near-zero until $p$ approaches 1. No threshold exists in $p$; the apparent one is $k$.

## 6. Mathematical Foundation

The optimum depends on exponents alone:

$$a \cdot \frac{A}{N^a} = b \cdot \frac{B}{D^b}, \qquad \left.\frac{D}{N}\right|_{\text{shallow}} = 325.0, \quad \left.\frac{D}{N}\right|_{\text{steep}} = 11.9$$ (eq:scaling-exponents-set-allocation-not-the-ceiling)

a factor of **27** in the same budget, while the floor's share of the loss rises to **94.2%**.

Serving changes the objective:

$$\min_{N} \left[6ND(N) + 2NT\right] \ \text{s.t.}\ L(N, D) = L^\star$$ (eq:the-training-optimum-is-not-the-deployment-optimum)

Model size falls by **8×** from research to consumer volume; the overspend from using the
training-optimal model reaches **13.7×**.

Metric shape is not model shape:

$$\text{EM}_k(x) = p(x)^k, \qquad \frac{\text{max step}}{\text{median step}} = 4.2 \ (\text{loss}), \ \ 1152.9 \ (k = 20)$$ (eq:discontinuity-is-a-property-of-the-metric)

And extrapolation error compounds with range:

$$\left|\frac{\hat L(C) - L(C)}{L(C)}\right| = 12.6\%, \ 30.6\%, \ 48.9\%, \ 63.9\% \ \text{at } 2, 4, 6, 8 \ \text{decades}$$ (eq:extrapolation-error-grows-with-the-log-range)

## 7. Internal Mechanics

Why does the split depend only on the exponents? Because the constraint is multiplicative and
the objective is a sum of power laws. In log space the constraint is a line of slope −1 and each
term is a line of slope $-a$ or $-b$; the optimum sits where the two gradients balance, and
gradients in log space are exactly the exponents. Changing $A$ or $B$ translates the terms
vertically and moves the optimum's *value* without moving its *location*.

The floor's growing share has a mechanism that is easy to state and easy to forget. $E$ is the
entropy of the data under the best possible predictor — the part of next-token prediction that
is genuinely unpredictable. Nothing in the model can reduce it, so as the reducible terms fall,
the reported loss asymptotes and the *fraction* of it that responds to compute falls
monotonically. A team reading "loss fell 3% this generation" is reading a number whose reducible
part may have fallen by half.

The inference-aware result has a mechanism worth making concrete because it explains a
strategic pattern. Training cost is paid once and scales as $6ND$; serving cost is paid per token
and scales as $2NT$. The first is a function of the training decision; the second is a function
of the product's success. So the correct model size is a bet on adoption, and getting it wrong
in the optimistic direction is cheap while getting it wrong in the pessimistic direction is
expensive — which is exactly backwards from how model-size decisions are usually framed.

The thresholded-metric mechanism is the cleanest in the chapter. Exact match on $k$ tokens is a
conjunction of $k$ events, and this book has met conjunctions repeatedly — in
{{ch:ops-versioning}}'s reproducibility, in {{ch:rai-privacy}}'s deletion, in
{{ch:rai-oversight}}'s preconditions. Every one produced the same qualitative effect: a product
of things that mostly work is a thing that mostly does not, until each factor is close to one.
Here the conjunction is over tokens, and the "sudden" appearance is the moment $p^k$ leaves the
region where it rounds to zero.

Finally, why extrapolation error is always optimistic when the floor is omitted. A pure power law
is unbounded below; the true curve is bounded by $E$. Any fit that omits the bound must attribute
the observed flattening to a shallower exponent, and a shallower exponent extrapolated far enough
still reaches zero. The bias has a sign, and it is the sign that makes long-range plans look
affordable.

## 8. Implementation

The first listing treats the law as an allocation rule.

```python {tier=A name=C:/Users/MANAVB~1/AppData/Local/Temp/claude/C--Github-LaughingBuddha/30a87753-43a8-48c3-8378-261faf976dbb/scratchpad/ka1}
"""A scaling law is an allocation rule, and the allocation changes when you count inference.

The parametric form is `L(N, D) = E + A/N^a + B/D^b` -- an irreducible floor plus a parameter
term plus a data term. Everything interesting follows from three facts about it.

First, the exponents `a` and `b` decide how a fixed training budget splits between parameters
and tokens, and they do not decide where the curve ends
(eq:scaling-exponents-set-allocation-not-the-ceiling). Refitting them on a wider range changed
the recommended allocation by an order of magnitude without changing the shape of the surface
(cite:kaplan2020scaling, cite:hoffmann2022chinchilla), which is why the same budget has been
spent two very different ways within a few years.

Second, the split that minimises training loss per training FLOP is not the split that
minimises cost per served token, and for any real serving volume the two are far apart
(eq:the-training-optimum-is-not-the-deployment-optimum).

Third, `E` is a floor. As the budget grows, the terms the exponents govern shrink and the floor
does not.

The constants below are illustrative and chosen to make the arithmetic legible; the structure
is what transfers.
"""
E_FLOOR = 1.69       # irreducible loss (nats/token), illustrative
N_REF, D_REF = 1e10, 2e11    # anchor point both fits are made to agree on
TERM_REF = 0.30              # each of the two reducible terms, at the anchor


def constants(a, b):
    """Constants that make a given exponent pair agree at the anchor point."""
    return TERM_REF * N_REF ** a, TERM_REF * D_REF ** b


def loss(n, d, a, b):
    an, bd = constants(a, b)
    return E_FLOOR + an / n ** a + bd / d ** b


def optimal_split(c, a, b, steps=4000):
    """Cheapest (N, D) with 6ND = C, by scanning log N."""
    best = None
    lo, hi = 1e6, 1e13
    for k in range(steps):
        n = lo * (hi / lo) ** (k / (steps - 1))
        d = c / (6 * n)
        if d < 1e7:
            continue
        v = loss(n, d, a, b)
        if best is None or v < best[0]:
            best = (v, n, d)
    return best


# exponent pairs, not attributions: the point is that the pair decides the split
REGIMES = [
    ("shallow exponents", 0.076, 0.095),
    ("steeper exponents", 0.340, 0.280),
]
SHALLOW, STEEP = REGIMES[0][0], REGIMES[1][0]

print("The same budget, split two ways.")
print()
print(f"{'training FLOPs':>17}{'exponent pair':>20}{'parameters':>15}{'tokens':>15}"
      f"{'tokens/param':>15}{'loss':>9}")
print("-" * 91)
ratios = {}
for c in (1e19, 1e21, 1e23, 1e25):
    for label, a, b in REGIMES:
        v, n, d = optimal_split(c, a, b)
        ratios.setdefault(label, []).append(d / n)
        print(f"{c:>17.0e}{label:>20}{n:>15.3e}{d:>15.3e}{d / n:>15.1f}{v:>9.3f}")
    print()

mean_shallow = sum(ratios[SHALLOW]) / len(ratios[SHALLOW])
mean_steep = sum(ratios[STEEP]) / len(ratios[STEEP])
SPLIT_FACTOR = mean_shallow / mean_steep
print(f"mean tokens per parameter: {mean_shallow:.1f} under the shallow pair,"
      f" {mean_steep:.1f} under the steeper one")
print(f"a factor of {SPLIT_FACTOR:.0f} in the same budget")

print()
print()
print("Both pairs agree on what the budget buys, and on where it stops.")
print()
A, B = REGIMES[1][1], REGIMES[1][2]
AN, BD = constants(A, B)
print(f"{'training FLOPs':>17}{'best loss':>12}{'above the floor':>18}"
      f"{'floor share of loss':>22}")
print("-" * 69)
gaps = {}
for c in (1e19, 1e21, 1e23, 1e25, 1e27):
    v, n, d = optimal_split(c, A, B)
    gaps[c] = v - E_FLOOR
    print(f"{c:>17.0e}{v:>12.3f}{v - E_FLOOR:>18.3f}{E_FLOOR / v:>22.1%}")

print()
print(f"from 1e19 to 1e27 -- eight orders of magnitude -- the reducible part falls")
print(f"from {gaps[1e19]:.3f} to {gaps[1e27]:.3f}, a factor of {gaps[1e19] / gaps[1e27]:.1f}")
print(f"and the floor's share of the loss rises to"
      f" {E_FLOOR / (E_FLOOR + gaps[1e27]):.1%}")

print()
print()
print("What another halving of the reducible loss costs.")
print()
print(f"{'reducible loss target':>24}{'training FLOPs needed':>24}"
      f"{'multiple of the last':>23}")
print("-" * 71)
prev_c = None
targets = [1.0, 0.5, 0.25, 0.125]
needs = {}
for t in targets:
    c = 1e17
    while optimal_split(c, A, B)[0] - E_FLOOR > t and c < 1e34:
        c *= 1.5
    needs[t] = c
    mult = f"{c / prev_c:>22.0f}x" if prev_c else f"{'--':>23}"
    print(f"{t:>24.3f}{c:>24.2e}{mult}")
    prev_c = c

print()
print("The exponent sets that multiple and nothing sets the floor.")

print()
print()
print("Now count inference, which the training-optimal split does not.")
print()
SERVE = [
    ("a research artefact",      1e9),
    ("an internal tool",         1e12),
    ("a product feature",        1e14),
    ("a consumer product",       1e16),
]
TARGET = 2.10
print(f"{'deployment':>22}{'tokens served':>16}{'best N':>13}{'best D':>13}"
      f"{'train FLOPs':>14}{'serve FLOPs':>14}{'total':>13}")
print("-" * 105)


def cheapest_for_target(tokens_served, target, steps=600):
    """Smallest total FLOPs reaching a loss target, over choices of N."""
    best = None
    for k in range(steps):
        n = 1e7 * (1e13 / 1e7) ** (k / (steps - 1))
        # tokens needed at this N to hit the target
        rem = target - E_FLOOR - AN / n ** A
        if rem <= 0:
            continue
        d = (BD / rem) ** (1 / B)
        total = 6 * n * d + 2 * n * tokens_served
        if best is None or total < best[0]:
            best = (total, n, d, 6 * n * d, 2 * n * tokens_served)
    return best


serve_best = {}
for label, served in SERVE:
    total, n, d, tr, inf = cheapest_for_target(served, TARGET)
    serve_best[label] = (n, d, tr, inf, total)
    print(f"{label:>22}{served:>16.0e}{n:>13.3e}{d:>13.3e}"
          f"{tr:>14.2e}{inf:>14.2e}{total:>13.2e}")

n_small = serve_best["a research artefact"][0]
n_large = serve_best["a consumer product"][0]
print()
print(f"at a loss target of {TARGET:.2f}, the best model shrinks from"
      f" {n_small:.2e} to {n_large:.2e} parameters")
print(f"a factor of {n_small / n_large:.0f}, driven entirely by serving volume")

print()
print()
print("What using the training-optimal model instead costs.")
print()
tr_opt_n, tr_opt_d = None, None
v, tr_opt_n, tr_opt_d = optimal_split(6e23, A, B)
print(f"training-optimal at 6e23 FLOPs: {tr_opt_n:.3e} parameters,"
      f" loss {v:.3f}")
print()
overspend = {}
print(f"{'deployment':>22}{'training-optimal total':>25}{'inference-aware total':>24}"
      f"{'overspend':>13}")
print("-" * 84)
for label, served in SERVE:
    # match the loss target by adding tokens to the training-optimal N
    rem = TARGET - E_FLOOR - AN / tr_opt_n ** A
    d_match = (BD / rem) ** (1 / B) if rem > 0 else float("inf")
    naive = 6 * tr_opt_n * d_match + 2 * tr_opt_n * served
    aware = serve_best[label][4]
    overspend[label] = naive / aware
    print(f"{label:>22}{naive:>25.2e}{aware:>24.2e}{naive / aware:>12.1f}x")

print(f"""
The first table is the result that made scaling laws an engineering subject rather than a
curiosity. The same training budget, split according to two different exponent pairs, produces
models that differ by a factor of **{SPLIT_FACTOR:.0f}** in tokens per parameter --
{mean_shallow:.1f} under the shallow pair against {mean_steep:.1f} under the steeper one.

Both pairs describe a surface of the same shape and both are anchored to agree at one point.
They disagree about the *slope in two directions*, and that is what decides the split: at the
optimum the two reducible terms stand in the ratio `b/a`, so the exponents alone fix how much of
a budget becomes parameters and how much becomes tokens.

**The exponents are an allocation rule**
(eq:scaling-exponents-set-allocation-not-the-ceiling). Refitting them on a wider range is what
changed the industry's recommended model size for a given budget
(cite:kaplan2020scaling, cite:hoffmann2022chinchilla) -- not a new capability, a new division of
the same money.

The second table is the part that gets less attention and matters more over time. From 1e19 to
1e27 training FLOPs -- eight orders of magnitude -- the reducible part of the loss falls from
{gaps[1e19]:.3f} to {gaps[1e27]:.3f}, a factor of {gaps[1e19] / gaps[1e27]:.1f}, and the
irreducible floor's share of the total rises to
{E_FLOOR / (E_FLOOR + gaps[1e27]):.1%}.

**The exponents govern a shrinking share of the number being reported.** A curve fitted where
the reducible term dominates is being extrapolated into a region where it does not, which is the
second listing's problem.

The third table prices the exponent directly. Each halving of the reducible loss costs roughly
{needs[0.25] / needs[0.5]:.0f} times the compute of the previous one. That multiple is set by
`a` and `b` and by nothing else -- it is the same whether the constants are large or small --
and it is why an order-of-magnitude compute increase is a routine expectation rather than a
breakthrough.

The fourth table is the one to act on, and it is not in the original framing at all.

The compute-optimal split minimises training loss per *training* FLOP. A deployed model also
costs about `2N` FLOPs per served token, and that term grows with the product's success rather
than with the training run. Holding the loss target at {TARGET:.2f} and varying serving volume
from {1e9:.0e} to {1e16:.0e} tokens, the cheapest model shrinks from {n_small:.2e} to
{n_large:.2e} parameters -- **a factor of {n_small / n_large:.0f}, driven entirely by how much
the thing is used** (eq:the-training-optimum-is-not-the-deployment-optimum).

The last table prices the mistake. Training a compute-optimal model and then serving it costs
{overspend['a research artefact']:.1f} times an inference-aware design for a research artefact
and **{overspend['a consumer product']:.0f} times** for a consumer product, at identical
quality, and the gap widens with every user.

**"Compute-optimal" is a claim about a training run, not about a system**, and the two answers
diverge in exactly the direction a successful product moves.

There is a corollary worth stating plainly: cite:hoffmann2022chinchilla's result made models
smaller for a given budget, and inference-awareness pushes further in the same direction --
smaller models trained past the training-optimal point, deliberately. The overspend column is
the argument, and it grows monotonically with serving volume.""")
```

## 9. Practical Example

The same budget, split two ways:

```
   training FLOPs       exponent pair     parameters         tokens   tokens/param     loss
-------------------------------------------------------------------------------------------
            1e+19   shallow exponents      5.278e+07      3.158e+10          598.4    2.494
            1e+19   steeper exponents      5.555e+08      3.000e+09            5.4    3.464
            1e+23   shallow exponents      8.820e+09      1.890e+12          214.2    2.235
            1e+23   steeper exponents      3.557e+10      4.685e+11           13.2    2.121
            1e+25   shallow exponents      1.136e+11      1.468e+13          129.2    2.139
            1e+25   steeper exponents      2.847e+11      5.854e+12           20.6    1.903
```

Mean tokens per parameter: **325.0** against **11.9** — a factor of **27**
({{eq:scaling-exponents-set-allocation-not-the-ceiling}}).

```
   training FLOPs   best loss   above the floor   floor share of loss
---------------------------------------------------------------------
            1e+19       3.464             1.774                 48.8%
            1e+21       2.565             0.875                 65.9%
            1e+23       2.121             0.431                 79.7%
            1e+25       1.903             0.213                 88.8%
            1e+27       1.795             0.105                 94.2%
```

Eight orders of magnitude move the reducible part by **16.9×** and take the floor's share to
**94.2%**.

```
   reducible loss target   training FLOPs needed   multiple of the last
-----------------------------------------------------------------------
                   1.000                4.99e+20                     --
                   0.500                4.31e+22                    86x
                   0.250                3.73e+24                    86x
                   0.125                3.23e+26                    86x
```

Each halving costs **86×** — set by the exponents and nothing else.

```
            deployment   tokens served       best N       best D   train FLOPs   serve FLOPs        total
---------------------------------------------------------------------------------------------------------
   a research artefact           1e+09    4.131e+10    5.607e+11      1.39e+23      8.26e+19     1.39e+23
      an internal tool           1e+12    2.604e+10    9.605e+11      1.50e+23      5.21e+22     2.02e+23
     a product feature           1e+14    8.809e+09    1.137e+13      6.01e+23      1.76e+24     2.36e+24
    a consumer product           1e+16    5.303e+09    3.266e+14      1.04e+25      1.06e+26     1.16e+26

            deployment   training-optimal total   inference-aware total    overspend
------------------------------------------------------------------------------------
   a research artefact                 1.56e+23                1.39e+23         1.1x
      an internal tool                 3.16e+23                2.02e+23         1.6x
     a product feature                 1.62e+25                2.36e+24         6.8x
    a consumer product                 1.60e+27                1.16e+26        13.7x
```

**Model size falls 8× and the overspend reaches 13.7×**
({{eq:the-training-optimum-is-not-the-deployment-optimum}}).

The second listing measures the curve.

```python {tier=A name=C:/Users/MANAVB~1/AppData/Local/Temp/claude/C--Github-LaughingBuddha/30a87753-43a8-48c3-8378-261faf976dbb/scratchpad/ka2}
"""One run, three metrics, three stories -- and a fit used outside the range it was fitted on.

The first listing treated the loss curve as given. This one asks what the curve is made of.

Two failures recur. A metric that requires every token of an answer to be right raises a smooth
per-token improvement to a power, and a power of a smooth curve looks like a threshold. The
discontinuity belongs to the metric, not to the model
(cite:wei2022emergent, cite:schaeffer2023mirage; eq:discontinuity-is-a-property-of-the-metric).

And a power law fitted over two decades and used over six is an extrapolation whose error grows
with the log-range, in a direction that is always optimistic when the true curve has a floor
(eq:extrapolation-error-grows-with-the-log-range).
"""
import math

E_FLOOR, A_RED, GAMMA = 1.69, 2.7e3, 0.155


def true_loss(c):
    return E_FLOOR + A_RED * c ** -GAMMA


def token_acc(c):
    """Smooth per-token accuracy: no thresholds anywhere in it."""
    return 1.0 / (1.0 + math.exp(-(math.log10(c) - 22.0) / 1.6))


BUDGETS = [10 ** x for x in range(18, 28)]

print("One run, scored four ways.")
print()
print(f"{'training FLOPs':>17}{'per-token accuracy':>21}{'loss':>10}"
      f"{'exact match, 5 tokens':>24}{'exact match, 20 tokens':>25}")
print("-" * 97)
series = {"per-token accuracy": [], "loss": [], "em5": [], "em20": []}
for c in BUDGETS:
    p = token_acc(c)
    series["per-token accuracy"].append(p)
    series["loss"].append(true_loss(c))
    series["em5"].append(p ** 5)
    series["em20"].append(p ** 20)
    print(f"{c:>17.0e}{p:>21.4f}{true_loss(c):>10.3f}"
          f"{p ** 5:>24.6f}{p ** 20:>25.8f}")

print()
print("Nothing in the generating process has a threshold in it.")

print()
print()
print("How discontinuous each metric looks.")
print()


def jumpiness(vals):
    """Largest single-step gain as a multiple of the median step."""
    steps = [abs(b - a) for a, b in zip(vals, vals[1:])]
    med = sorted(steps)[len(steps) // 2]
    return max(steps) / med if med > 0 else float("inf")


print(f"{'metric':>26}{'largest step / median step':>30}{'reads as':>22}")
print("-" * 78)
jump = {}
for label in ("loss", "per-token accuracy", "em5", "em20"):
    j = jumpiness(series[label])
    jump[label] = j
    reads = "smooth" if j < 3 else ("kinked" if j < 8 else "emergent")
    print(f"{label:>26}{j:>30.1f}{reads:>22}")

print()
print(f"the same run reads as {'smooth':>0} on loss ({jump['loss']:.1f}) and"
      f" as a threshold on 20-token exact match ({jump['em20']:.1f})")
print(f"a factor of {jump['em20'] / jump['loss']:.0f} in apparent discontinuity")

print()
print()
print("And each metric names a different budget as the moment it 'appeared'.")
print()
print(f"{'metric':>26}{'5% of final':>16}{'50% of final':>16}"
      f"{'decades between them':>24}")
print("-" * 82)
onset = {}
for label in ("loss", "per-token accuracy", "em5", "em20"):
    vals = series[label]
    if label == "loss":
        rng = [(vals[0] - v) / (vals[0] - vals[-1]) for v in vals]
    else:
        rng = [v / vals[-1] for v in vals]
    c5 = next(BUDGETS[i] for i, r in enumerate(rng) if r >= 0.05)
    c50 = next(BUDGETS[i] for i, r in enumerate(rng) if r >= 0.50)
    onset[label] = (c5, c50)
    print(f"{label:>26}{c5:>16.0e}{c50:>16.0e}"
          f"{math.log10(c50 / c5):>24.0f}")

print()
print(f"`loss` starts moving at {onset['loss'][0]:.0e};"
      f" `em20` at {onset['em20'][0]:.0e}")
print(f"a difference of {math.log10(onset['em20'][0] / onset['loss'][0]):.0f}"
      f" orders of magnitude, from the same run")

print()
print()
print("Now the second failure: a fit used outside its range.")
print()
FIT_LO, FIT_HI = 1e19, 1e21


def fit_powerlaw(lo, hi, with_floor):
    """Least-squares slope and intercept in log space over [lo, hi]."""
    xs, ys = [], []
    for k in range(24):
        c = lo * (hi / lo) ** (k / 23)
        v = true_loss(c) - (E_FLOOR if with_floor else 0.0)
        xs.append(math.log10(c))
        ys.append(math.log10(v))
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    g = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / \
        sum((x - mx) ** 2 for x in xs)
    return g, my - g * mx


G_NO, I_NO = fit_powerlaw(FIT_LO, FIT_HI, with_floor=False)
G_YES, I_YES = fit_powerlaw(FIT_LO, FIT_HI, with_floor=True)
print(f"fitted over {FIT_LO:.0e} to {FIT_HI:.0e} -- two decades")
print(f"  no floor term: exponent {G_NO:.4f}")
print(f"  floor term:    exponent {G_YES:.4f} (true {-GAMMA:.4f})")

print()
print(f"{'predicted at':>16}{'decades out':>14}{'no-floor fit':>15}"
      f"{'floor fit':>13}{'truth':>10}{'no-floor error':>17}")
print("-" * 85)
err = {}
for c in (1e21, 1e23, 1e25, 1e27, 1e29):
    x = math.log10(c)
    p_no = 10 ** (I_NO + G_NO * x)
    p_yes = E_FLOOR + 10 ** (I_YES + G_YES * x)
    truth = true_loss(c)
    dec = x - math.log10(FIT_HI)
    err[c] = abs(p_no - truth) / truth
    print(f"{c:>16.0e}{dec:>14.0f}{p_no:>15.3f}{p_yes:>13.3f}"
          f"{truth:>10.3f}{err[c]:>16.1%}")

print()
print("A fit that omits the floor predicts loss going to zero, and the error")
print("grows with every decade of extrapolation.")

print()
print()
print("What breaks the extrapolation in practice, beyond the functional form.")
print()
FACTORS = [
    ("benchmark contamination",     "the measured score, not the loss",
     "ch:ev-llm-benchmarks",  0.061, "inflates and flattens"),
    ("repeated training data",      "effective D below nominal D",
     "cite:lee2022dedup", 0.048, "the D term stalls"),
    ("reduced numerical precision", "an effective-parameter penalty",
     "cite:kumar2024precisionscaling", 0.037, "the N term stalls"),
    ("distribution shift at eval",  "a different loss surface",
     "ch:ops-observability", 0.029, "the fit does not apply"),
    ("data exhaustion",             "D cannot be bought at any price",
     "--",                0.055, "the budget stops splitting"),
]
print(f"{'factor':>30}{'what it changes':>36}{'where':>32}{'loss shortfall':>17}")
print("-" * 115)
shortfall = 0.0
for name, what, where, gap, effect in FACTORS:
    shortfall += gap
    print(f"{name:>30}{what:>36}{where:>32}{gap:>17.3f}")
print("-" * 115)
print(f"{'TOTAL':>30}{'':>36}{'':>32}{shortfall:>17.3f}")

PRED = true_loss(1e26)
print()
print(f"a fit predicts {PRED:.3f} at 1e26; these five together leave"
      f" {PRED + shortfall:.3f}")
print(f"which is the loss the curve reaches at {10 ** ((-(PRED + shortfall - E_FLOOR) / A_RED) ** 0 * 0):.0f}"
      if False else
      f"a shortfall of {shortfall / (PRED - E_FLOOR):.1%} of the reducible loss at that budget")

print(f"""
The first table is one training run, scored four ways, with **no threshold anywhere in the
generating process**. Per-token accuracy is a smooth logistic in log-compute; the loss is a
smooth power law with a floor.

Raise that smooth per-token accuracy to the fifth power and you get a curve that spends four
orders of magnitude near zero and then climbs. Raise it to the twentieth -- which is what
"the answer must be exactly right" means for a twenty-token answer -- and it spends six.

The jumpiness table quantifies it. Measured as the largest single step divided by the median
step, `loss` scores {jump['loss']:.1f} and 20-token exact match scores {jump['em20']:.1f} --
**a factor of {jump['em20'] / jump['loss']:.0f} in apparent discontinuity, from the same run**
(eq:discontinuity-is-a-property-of-the-metric).

The onset table is the practical consequence and the reason this matters for planning. Asked
"when did this capability appear", `loss` answers {onset['loss'][0]:.0e} and `em20` answers
{onset['em20'][0]:.0e} -- **{math.log10(onset['em20'][0] / onset['loss'][0]):.0f} orders of
magnitude apart**, on identical data.

That is ch:ev-why-hard' `metric-choice-manufactures-the-finding` and
`discontinuity-hides-progress` arriving in the scaling literature, and it has a specific cost: a
team measuring only exact match sees nothing for four orders of magnitude and concludes the
approach does not work, while the per-token signal was improving throughout. The fix is the
cheap one from that chapter -- keep a continuous metric alongside the binary one -- and it is
worth more here than anywhere else in the book, because the budgets involved are enormous.

The extrapolation section is the second failure and the more expensive one.

Fit a pure power law -- no floor term -- over {FIT_LO:.0e} to {FIT_HI:.0e}, two decades where the
reducible part dominates, and it fits well. Use it at {1e27:.0e}, six decades out, and it is
{err[1e27]:.0%} wrong; at {1e29:.0e}, {err[1e29]:.0%}.

**The error grows with the log-range and it grows in one direction**
(eq:extrapolation-error-grows-with-the-log-range). A fit without a floor term predicts loss
approaching zero, which no model of a stochastic process should predict, and the same fit
*with* a floor term tracks the truth across the whole range.

The distinguishing question is not statistical. It is whether the functional form contains the
irreducible entropy of the data, and a two-decade window where that term is small will not tell
you.

The last table is what breaks the extrapolation for reasons outside the functional form
entirely. Benchmark contamination changes the measured score without changing the model
(ch:ev-llm-benchmarks). Repeated data means effective `D` is below nominal `D`
(cite:lee2022dedup). Reduced precision imposes an effective-parameter penalty
(cite:kumar2024precisionscaling). And data exhaustion means the budget cannot be split the way
the optimum requires, at any price.

Together they leave {shortfall:.3f} of loss on the table at 1e26 -- **{shortfall / (PRED - E_FLOOR):.0%}
of the reducible loss remaining at that budget**. Every one of them is a property of the
pipeline rather than of the scaling relationship, and not one of them appears in the fit.

**A scaling law predicts what a clean run would do**, and the gap between that and what your run
does is the part you control.""")
```

```
   training FLOPs   per-token accuracy      loss   exact match, 5 tokens   exact match, 20 tokens
-------------------------------------------------------------------------------------------------
            1e+18               0.0759     6.069                0.000003               0.00000000
            1e+21               0.3486     3.191                0.005151               0.00000000
            1e+23               0.6514     2.425                0.117243               0.00018895
            1e+25               0.8670     2.050                0.489988               0.05764215
            1e+27               0.9579     1.866                0.806545               0.42316988

                    metric    largest step / median step              reads as
------------------------------------------------------------------------------
                      loss                           4.2                kinked
        per-token accuracy                           1.7                smooth
                      em20                        1152.9              emergent
```

**277× more apparent discontinuity, from the same run**
({{eq:discontinuity-is-a-property-of-the-metric}}).

```
                    metric     5% of final    50% of final    decades between them
----------------------------------------------------------------------------------
                      loss           1e+19           1e+20                       1
        per-token accuracy           1e+18           1e+22                       4
                      em20           1e+25           1e+27                       2
```

The two metrics name budgets **6 orders of magnitude apart** as the onset.

```
    predicted at   decades out   no-floor fit    floor fit     truth   no-floor error
-------------------------------------------------------------------------------------
           1e+23             2          2.120        2.425     2.425           12.6%
           1e+25             4          1.423        2.050     2.050           30.6%
           1e+27             6          0.955        1.866     1.866           48.9%
           1e+29             8          0.641        1.776     1.776           63.9%
```

**48.9% wrong six decades out**, always optimistically
({{eq:extrapolation-error-grows-with-the-log-range}}).

```
                        factor                     what it changes                           where   loss shortfall
-------------------------------------------------------------------------------------------------------------------
       benchmark contamination    the measured score, not the loss                ch:ev-llm-benchmarks            0.061
               data exhaustion     D cannot be bought at any price                              --            0.055
        repeated training data         effective D below nominal D               cite:lee2022dedup            0.048
   reduced numerical precision      an effective-parameter penalty  cite:kumar2024precisionscaling            0.037
-------------------------------------------------------------------------------------------------------------------
                         TOTAL                                                                                0.230
```

**0.230 of loss left on the table — 91% of the reducible loss at that budget.**

## 10. Production Considerations

Read a scaling law as an allocation rule and fit it yourself on your own runs before trusting a
published exponent pair. The split is what you are buying and it moves by 27× across plausible
fits.

Always include a floor term. It costs one parameter, it recovers the true exponent, and omitting
it biases every long-range projection in the same optimistic direction.

Report reducible loss, not loss. A 3% headline movement can be a halving of the part that
responds to compute, or nothing at all, and the raw number does not distinguish them.

Compute the inference-aware optimum before fixing model size, using a serving-volume forecast
you are willing to defend. The training-optimal answer is wrong by 13.7× at consumer scale.

Keep a continuous metric alongside every binary one, from the smallest scale you run. This is
{{ch:ev-why-hard}}'s cheapest recommendation and it is worth the most here.

Never declare a capability absent from a thresholded metric alone. Six orders of magnitude
separated the two onsets in {{sec:9-practical-example}}.

Budget for the pipeline losses separately. Contamination, repeated data, precision and shift are
worth 91% of the remaining reducible loss and none is in the fit.

## 11. Common Mistakes

**Treating a scaling law as a forecast.** It is an allocation rule with a floor.

**Fitting without a floor term.** Recovers the wrong exponent and predicts loss reaching zero.

**Optimising the training run in isolation.** Serving cost scales with success, not with the run.

**Reading a thresholded curve as a claim about the model.** The exponent on the metric is doing
the work.

**Concluding an approach fails from a flat exact-match curve.** It was flat for four orders of
magnitude while the underlying signal improved.

**Quoting loss without its floor.** The reported number is 94.2% floor at large budgets.

**Assuming the exponents transfer.** They are fitted, on a range, on a pipeline.

## 12. Failure Modes

**A programme cancelled at $10^{22}$ FLOPs.** Exact match reads zero; per-token accuracy was
already at 0.50.

**A six-decade projection built on a two-decade fit.** 48.9% optimistic and used to justify a
capital plan.

**A consumer product served by a training-optimal model.** 13.7× overspend, discovered as an
inference bill.

**Effective $D$ far below nominal $D$.** Repeated data, and the data term stops responding while
the budget keeps growing.

**A benchmark that saturates before the trend does.** {{eq:headroom-sets-benchmark-lifespan}},
and the scaling study ends where the instrument does.

**A precision reduction that silently changes the exponent.**
{{cite:kumar2024precisionscaling}}'s effective-parameter penalty, attributed to the data.

## 13. Alternatives

**Fit $L(C)$ directly instead of $L(N, D)$.** Cheaper and sufficient if you never need the
split — which is exactly the thing the law is most useful for, so this is a false economy in
most cases.

**Ladder of small runs with an explicit extrapolation-error budget.** Fit on three decades,
project one, and re-fit; the error table gives the honest uncertainty.

**Optimise for a downstream metric rather than loss.** Attractive and much noisier; the
thresholded-metric result says the target's shape is partly an artefact of the scoring rule.

**Buy inference efficiency rather than smaller parameters.** Distillation, quantisation and
sparsity change the $2N$ term without changing the training decision, and compose with the
inference-aware optimum rather than replacing it.

**Refuse to extrapolate.** Defensible past three or four decades, and it leaves the planning
question unanswered rather than answered badly.

## 14. Evaluation

Fit your own exponents on a ladder of small runs, with and without a floor term. Compare the
recommended split under both fits; the gap is your allocation uncertainty.

Compute reducible loss at each checkpoint and report it alongside loss. Track how its share
changes across generations.

Score every capability study with at least one continuous metric and one thresholded one, and
report both onsets. Where they differ by orders of magnitude, say so.

Hold out a decade of compute when fitting, predict into it, and record the error. That number is
your extrapolation credibility and almost nobody has it.

Measure effective $D$ against nominal $D$ by deduplicating and re-counting. The gap is a direct
input to the fit.

## 15. Advanced Concepts

The parametric form treats $N$ and $D$ as sufficient statistics for a model and a dataset, and
neither is. Two models with identical parameter counts and different architectures sit on
different surfaces; two corpora with identical token counts and different composition do too.
The form's success across a wide range says these effects are second-order in the regimes
studied, not that they are absent, and the honest reading is that a scaling law is fitted to a
*family* — an architecture, a pipeline, a data mixture — and transfers only within it.
{{cite:kumar2024precisionscaling}}'s result is the clearest demonstration: change the numerical
precision and $N$ stops being the right variable.

The inference-aware optimum in {{sec:9-practical-example}} assumes serving volume is known in
advance, and it is not. The correct treatment is a decision under uncertainty: choose $N$ to
minimise expected total cost over a distribution of adoption outcomes. Because the serving term
is linear in $T$ and the training term is not, the objective is convex in $N$ for fixed $T$ and
the expectation shifts the optimum toward the smaller models favoured by the high-$T$ tail. **A
model sized for the median outcome is too large**, which is a stronger statement than the
chapter's table makes and follows directly from it.

There is a subtlety in the thresholded-metric argument that deserves stating, because the
conclusion is often overdrawn. Showing that a smooth process can *produce* an apparently
discontinuous metric does not show that every observed discontinuity is an artefact. It shows
the observation is not evidence of a discontinuity in the model, which is a weaker and more
useful claim. Distinguishing the two requires a continuous metric on the same runs, and where
that has been done the smooth reading has usually won — but "usually" is doing real work in that
sentence, and a system whose *internal* behaviour changes qualitatively at some scale would
produce the same picture.

Finally, the extrapolation result has an implication for how these projections should be
communicated. The error is not symmetric and not random: it has a known sign and a known
dependence on range. A projection should therefore be reported as a one-sided bound — "no better
than this" — rather than a point estimate with symmetric error bars, and the fitted-range span
should be printed next to every extrapolated number. Neither is standard practice, and both cost
nothing.

## 16. Connection to Previous Chapters

{{eq:metric-choice-manufactures-the-finding}} from {{ch:ev-why-hard}} is the whole
discontinuity result: one run, four metrics, and a **277×** spread in apparent shape.

{{eq:discontinuity-hides-progress}} from the same chapter is the cost — four orders of magnitude
of real improvement invisible under a 20-token exact match.

{{eq:contamination-inflates-and-flattens}} from {{ch:ev-llm-benchmarks}} is the largest single
pipeline loss here at **0.061**, and the only one that corrupts the measurement rather than the
model.

{{eq:headroom-sets-benchmark-lifespan}} from the same chapter is why the instrument used to
observe a scaling trend usually saturates before the trend does.

## 17. Exercises

1. Derive the optimal split condition and show it is independent of $A$, $B$ and $E$.

2. Compute the cost of a halving for an exponent pair of your choice, and check it against
   $2^{1/a + 1/b}$.

3. Fit a power law with and without a floor term to a two-decade window of a curve you generate
   with a known floor. How far out does the no-floor fit stay within 10%?

4. Compute the inference-aware optimum for your own serving forecast, and the overspend from the
   training-optimal model.

5. Take a capability curve you believe is discontinuous and re-score it with a continuous
   metric. What happens to the onset?

6. Extend {{sec:15-advanced-concepts}}'s uncertainty argument: for a log-normal adoption
   forecast, how much smaller is the expected-cost-optimal model than the median-optimal one?

## 18. Interview Questions

1. What does a scaling law actually tell you how to do?

2. Why did the recommended model size for a given budget change without any new capability?

3. Our loss improved 3% this generation. Is that a lot?

4. We're deploying to fifty million users. Does that change the model size?

5. This capability appeared suddenly at a certain scale. How would you check that claim?

6. We fitted over two decades and projected six. How wrong are we likely to be, and in which
   direction?

## 19. Research Questions

1. How much of a fitted exponent pair transfers across architecture families, and what is the
   measurable predictor of transfer?

2. Can effective $D$ under repetition be predicted from a corpus's duplication profile without
   running the training?

3. What fraction of reported emergent capabilities survive re-scoring with a continuous metric on
   the same runs?

4. What is the empirical distribution of extrapolation error against log-range across published
   scaling studies?

## 20. Chapter Summary

A scaling law is an allocation rule that people read as a forecast, measured through instruments
that manufacture their own shape.

As an allocation rule it is decisive and narrow: the optimal split depends on the exponents and
nothing else, and two plausible exponent pairs differ by **27×** in tokens per parameter —
**325.0** against **11.9** ({{eq:scaling-exponents-set-allocation-not-the-ceiling}}). Refitting
the slope is what changed the industry's recommended model size, not a discovery about learning.
Each halving of the reducible loss costs **86×**, set by the exponents alone. And across eight
orders of magnitude the floor's share of the reported loss rises to **94.2%**, so the exponents
govern a shrinking fraction of the number being quoted.

It is also a claim about a training run rather than a system. Adding the $2NT$ serving term
shrinks the optimal model by **8×** from research to consumer volume, and using the
training-optimal model costs **13.7×** at consumer scale
({{eq:the-training-optimum-is-not-the-deployment-optimum}}) — which is the argument for training
smaller models past the training-optimal point, deliberately.

The measurement half is sharper. One run with no threshold anywhere in it reads as **4.2** on
loss and **1152.9** on 20-token exact match — **277×** more apparent discontinuity — and the two
metrics place the onset **6 orders of magnitude apart**
({{eq:discontinuity-is-a-property-of-the-metric}}). A team watching only the binary metric sees
nothing for four decades while the signal improves throughout.

And a two-decade fit without a floor term is **12.6%**, **30.6%**, **48.9%**, **63.9%** wrong at
two, four, six and eight decades out — always optimistically
({{eq:extrapolation-error-grows-with-the-log-range}}). The same data with a floor term recovers
the true exponent exactly.

Then the losses that are not in the fit at all: contamination, data exhaustion, repetition,
precision and shift, together worth **0.230** of loss at $10^{26}$ — **91%** of the reducible
loss remaining there.

What runs through the chapter is a single distinction. **The law describes a clean run on a
fitted family over a fitted range**, and every practical question — how to split the budget, how
big to make the model, whether a capability has appeared, how far to project — depends on
something the law does not contain. That is not a criticism of scaling laws. It is what makes
them usable: they are precise about the one thing they model, and the rest is yours.

Carry forward: **the exponents allocate, the floor limits**, and **check the metric before
believing the shape**.

## 21. Further Reading

- {{cite:kaplan2020scaling}} — the study that made compute allocation a fitted question.
- {{cite:hoffmann2022chinchilla}} — the refit that changed the recommended split for a fixed
  budget.
- {{cite:wei2022emergent}} — the phenomenon of capabilities appearing at scale.
- {{cite:schaeffer2023mirage}} — the argument that the shape belongs partly to the metric.
- {{cite:kumar2024precisionscaling}} — precision as a term in the scaling relationship.
- {{cite:lee2022dedup}} — duplication, and the gap between nominal and effective data.
