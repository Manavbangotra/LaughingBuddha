---
id: inf-serving-stacks
number: 202
part: XXIII
tier: full
status: draft
requires: [decode-is-bandwidth-bound, static-batching-pays-for-the-longest,
           chunk-size-has-a-cliff, roofline-has-multiple-ridges]
provides: [feature-credit-depends-on-order, overlapping-techniques-are-substitutes,
           launch-overhead-is-a-floor, cheaper-models-raise-the-overhead-share]
citations: [kwon2023pagedattention, agrawal2023sarathi, dao2022flash,
            leviathan2023speculative]
---

## 1. Learning Objectives

By the end of this chapter you will be able to decompose a serving stack's headline
speedup into the inefficiencies its features address, and explain why the decomposition
is order-dependent; recognise that techniques targeting the same inefficiency are
substitutes rather than complements, and choose a subset accordingly; compute the fixed
kernel-launch cost of a decode step and show that it is a constant *share* of every
step in the memory-bound regime rather than something batching dilutes; explain why the
roofline model systematically overpredicts throughput and by how much; and say why every
optimisation that makes a model cheaper makes launch overhead a larger share of what
remains.

## 2. Why This Matters

The previous five chapters measured techniques one at a time. A serving stack ships them
as a bundle, reports one number, and the number is real — {{sec:9-practical-example}}
measures a naive loop going to **4.87×** with everything enabled.

The trouble starts when you ask which parts you need. Adding the same five features in
two different orders gives continuous batching **19%** of the credit in one ordering and
**61%** in the other, and graph capture **34%** and **2%**
({{eq:feature-credit-depends-on-order}}). Same features, same workload, same total.

That is not a measurement error — it is what happens when techniques overlap. Two
features addressing one inefficiency are **substitutes**, so the second one added
recovers only what the first left
({{eq:overlapping-techniques-are-substitutes}}), and a roadmap that budgets them
additively overpromises by exactly the overlap. {{sec:9-practical-example}} finds three
of five features capturing **93%** of the available gain.

The second half of the chapter is about a term this part's models have omitted entirely.
A 32-layer decode step launches **448 kernels**, costing **1.88 ms** before any
arithmetic begins — **31.0%** of a batch-1 step
({{eq:launch-overhead-is-a-floor}}). And because step time is *constant* below the
balance point, that share is **31.0% at batch 8, at batch 32, and at batch 256**. It
does not dilute.

## 3. Prerequisites

You need {{eq:decode-is-bandwidth-bound}} from {{ch:inf-cpu-gpu}} — the constancy of
step time below the balance point is what makes launch overhead a constant share rather
than a shrinking one, and that is the chapter's least intuitive result.

{{eq:static-batching-pays-for-the-longest}} from {{ch:inf-batching}} and
{{eq:chunk-size-has-a-cliff}} supply two of the five features being decomposed.

{{eq:roofline-has-multiple-ridges}} from {{ch:inf-gpu-memory}} is the model that
{{eq:launch-overhead-is-a-floor}} corrects; both refinements are additive to it rather
than replacing it.

Familiarity with {{cite:kwon2023pagedattention}} and {{cite:agrawal2023sarathi}} from the
preceding chapters is assumed.

## 4. Intuitive Explanation

Suppose someone tells you their serving stack is five times faster than a naive
implementation. That is almost certainly true. Now suppose you ask which of its five
features accounts for most of the gain, because you have limited engineering time and
want to build the important ones.

The question has no stable answer, and it is worth understanding why before dismissing
it as a technicality.

Picture the naive loop's time as a pie divided by what it is wasted on: some on slots
held by finished sequences, some on prefills stalling decodes, some on reading weights
you could have read in half the bytes, some on telling the GPU what to do. Each feature
eats a slice.

Now here is the thing. **Continuous batching and paged attention eat the same slice.**
Both address slots not doing useful work. Whichever you enable first gets to remove most
of that slice; the second one arrives to find only crumbs.

So if you enable continuous batching first, it looks enormous and paged attention looks
marginal. Enable paged attention first, and it looks substantial while continuous
batching looks smaller. Neither ordering is wrong. Both report the true marginal
contribution of that feature *given what was already on*.

Which means "feature X gives 3×" is not a fact about feature X. It is a fact about X and
the baseline. A vendor measuring against a naive loop and a vendor measuring against
their previous release are reporting different quantities, both honestly, and the
numbers are not comparable.

The practical consequence is not philosophical. It is that a roadmap listing five
optimisations with their published speedups and multiplying them together will
overpromise, and it will overpromise by exactly the amount they overlap. The way to
catch this before building is to write down, for each item, *which inefficiency it
addresses* — and to notice when two items name the same one.

The second half of the chapter is about a cost the models in this part have simply not
had a term for.

Everything so far has described a step as arithmetic and memory traffic. Both scale with
work. But before any of that happens, someone has to tell the GPU what to do — and a
transformer layer is not one instruction, it is a dozen or more separate kernels, each
launched individually, each with arguments to marshal and dependencies to resolve.

Thirty-two layers times fourteen kernels is nearly four hundred and fifty launches per
token. At a few microseconds each, that is a couple of milliseconds of pure
administration.

Now the part that surprises people. You might expect this to matter at batch 1 and
vanish at batch 256. It does not — and the reason is a result from three chapters back.
Below the balance point, **step time does not change with batch size**: the weights are
read once regardless, and the arithmetic on 32 tokens finishes long before the read
does.

So the step time is constant, and the launch overhead is constant, and a constant
divided by a constant is a constant. Launch overhead is the same **31%** of a decode
step at batch 1 and at batch 256. Batching does not dilute it, because batching does not
make the step longer.

Which explains a common and confusing experience: you compute expected throughput from
memory bandwidth, measure something a third lower, and go looking for the problem in the
memory system. It is not in the memory system. The device is waiting to be told what to
do.

And it is getting worse. Launch cost is fixed; the work it launches is not. Every
optimisation that makes the model cheaper — quantisation, smaller models, better kernels
— shrinks the term the roofline models and leaves the term it does not.

## 5. Formal Explanation

**Feature decomposition.** Let a naive loop's step time be divided among inefficiencies
$c \in C$ with shares $\beta_c$ summing to one, including an irreducible component.
A feature $f$ targets inefficiency $\tau(f)$ and removes a fraction $\phi_f$ of it. With
a set $S$ of features enabled, the remaining time is

$$ T(S) \;=\; \sum_{c \in C}\beta_c \prod_{f \in S:\, \tau(f) = c}(1 - \phi_f) $$

and throughput is $1/T(S)$. The marginal gain of adding $f$ to $S$ is
$T(S)/T(S \cup \{f\})$, which depends on $S$ through the product over features sharing
$\tau(f)$:

$$ \frac{T(S)}{T(S \cup \{f\})} \;=\; \frac{\sum_c \beta_c\prod_{g \in S}(\cdot)}{\sum_c \beta_c\prod_{g \in S \cup \{f\}}(\cdot)} $$ (eq:feature-credit-depends-on-order)

**Two features with $\tau(f) = \tau(g)$ are substitutes**: enabling $g$ reduces the
residual $\beta_{\tau(f)}\prod(1-\phi)$ that $f$ can act on, so $f$'s marginal gain
falls. Features with distinct targets are independent, and their gains compose without
interference:

$$ \tau(f) \ne \tau(g) \;\Longrightarrow\; \frac{T(\emptyset)}{T(\{f,g\})} \;=\; \text{(neither multiplicative nor additive, but monotone in both)} $$ (eq:overlapping-techniques-are-substitutes)

The order-independent attribution is the Shapley value — the average marginal
contribution over all $|S|!$ orderings — which exists and is unique but answers a
different question than "what should I build."

**Launch overhead.** Let a step launch $K$ kernels at fixed cost $\lambda$ each. Adding
to {{ch:inf-cpu-gpu}}'s model,

$$ T(m) \;=\; K\lambda \;+\; \max\!\left(\frac{Pb}{B},\; \frac{2Pm}{F}\right) $$ (eq:launch-overhead-is-a-floor)

Below the balance point the second term is $Pb/B$, **independent of $m$**. So the launch
share is

$$ \frac{K\lambda}{K\lambda + Pb/B} \quad\text{— constant in } m $$

and it falls only once $2Pm/F > Pb/B$, that is $m > FB^{-1}b$ — the same balance point,
now determining where launch overhead begins to dilute.

The floor this imposes is absolute: at batch 1, throughput cannot exceed
$1/(K\lambda + Pb/B) < 1/(K\lambda)$ tokens per second, regardless of hardware.

## 6. Mathematical Foundation

Two consequences of {{eq:launch-overhead-is-a-floor}} deserve separate statement.

**The roofline's systematic error.** The predicted step time is $Pb/B$ and the actual is
$K\lambda + Pb/B$, so the relative error is

$$ \frac{T_{\text{roofline}}}{T_{\text{actual}}} - 1 \;=\; -\frac{K\lambda}{K\lambda + Pb/B} $$

which is exactly the launch share. **The roofline underpredicts step time by precisely
the fraction that launches occupy**, and since that fraction is constant in the
memory-bound regime, the error is a constant offset rather than a converging
approximation. {{sec:9-practical-example}} measures **−31.0%** across every batch from 1
to 256.

**The share rises as models get cheaper.** Differentiating the launch share with respect
to weight bytes $W = Pb$:

$$ \frac{\partial}{\partial W}\left(\frac{K\lambda}{K\lambda + W/B}\right) \;<\; 0 $$ (eq:cheaper-models-raise-the-overhead-share)

so **every reduction in $W$ raises the launch share**. Quantisation halves $W$ and nearly
doubles the overhead fraction; a smaller model does the same. {{sec:9-practical-example}}
finds a 140 GB model at **4.3%** launch share and a 1.5 GB model at **80.8%**.

That is the most operationally useful form of the result. A team that quantises a small
model and measures less than the predicted speedup has not made a mistake in the
quantisation — they have hit a term the memory arithmetic does not contain, and it grows
in proportion to their success at reducing the term it does.

Graph capture attacks $\lambda$ rather than $K$, cutting per-kernel cost by roughly an
order of magnitude. Because both $K\lambda$ and $Pb/B$ are constant in $m$, the speedup
$(K\lambda_{\text{plain}} + Pb/B)/(K\lambda_{\text{graph}} + Pb/B)$ is **also constant in
$m$** — {{sec:9-practical-example}} measures **1.40×** at batch 1, at batch 32, and at
batch 256.

**That makes graph capture the only technique in this part whose benefit is independent
of batch size, context length, and interconnect.** Everything else in Part XXIII needs a
condition; this one does not.

## 7. Internal Mechanics

**What the five features actually are.** Continuous batching
({{cite:kwon2023pagedattention}}) removes slot idleness by letting sequences leave and
join between steps. Paged KV cache removes the fragmentation that caps achievable batch
— which is also slot idleness, hence the overlap. Chunked prefill
({{cite:agrawal2023sarathi}}) removes phase stalls. Quantised kernels reduce weight
traffic. Graph capture reduces launch overhead. Three of the five have a unique target;
two share one.

**Why the overlap is between exactly those two.** Continuous batching and paging both
raise the number of sequences doing useful work per step, from opposite directions — one
by removing waiting, the other by removing wasted reservation. A deployment with uniform
sequence lengths gets little from continuous batching and much from paging; one with
uniform lengths and no fragmentation gets little from either. **The overlap is
workload-dependent**, which is a further reason published attributions do not transfer.

**Where the kernel count comes from.** Per layer: query, key and value projections,
attention, output projection, two layer norms, the MLP's up, gate and down projections,
an activation, a residual add, and cache writes. Fusing reduces the count — a fused
attention kernel ({{cite:dao2022flash}}) replaces several — which is a second attack on
$K\lambda$ orthogonal to graph capture's attack on $\lambda$.

**Why graph capture is not free to adopt.** A captured graph fixes the shapes and the
control flow. Continuous batching changes the batch composition every step, so the
capture must either be re-done, or done for a set of bucketed shapes with padding to the
nearest bucket. Padding wastes exactly the slot utilisation continuous batching
recovered, which is a real interaction the decomposition model does not represent.

**What the stacks differ on.** vLLM's contribution is the paged cache and the scheduler
built on it; TensorRT-LLM's is compiled and fused kernels with aggressive graph capture;
Triton is an inference server that hosts either. The choice is therefore not "which is
fastest" but which set of inefficiencies each addresses well for your workload — a
long-context, high-fragmentation workload and a short-prompt, small-model workload have
different answers.

**Why the irreducible share matters more than it looks.** The budget table reserves 6%
for work that no feature addresses -- the arithmetic that actually has to happen, plus
whatever the framework spends on bookkeeping no optimisation targets. That term sets a
hard ceiling on the whole exercise: with every inefficiency perfectly eliminated, the
speedup would be $1/0.06 \approx 16.7$x, and the five features reach 4.87x. So the
stack captures roughly a third of what is theoretically available, and the remaining
two thirds sits in the *residual* of each inefficiency rather than in anything
unaddressed. Continuous batching removes 92% of slot idleness, not all of it; fp8
removes half of weight traffic, not all. **The gap between a stack's speedup and the
theoretical ceiling is mostly incomplete removal rather than missing features**, which
is a different roadmap than "add a sixth technique."

**Speculative decoding's place in the taxonomy.**
{{cite:leviathan2023speculative}} raises tokens per step, which by
{{eq:decode-is-bandwidth-bound}} raises arithmetic intensity — targeting weight traffic
per useful token. It therefore overlaps with quantisation and not with graph capture,
which is a prediction the decomposition model makes and which is worth testing before
budgeting both.

## 8. Implementation

The first listing decomposes a serving stack's speedup and measures the order
dependence.

```python {tier=A name=db1}
"""What a serving stack buys, decomposed -- and why the decomposition is ambiguous.

A serving stack is a bundle of techniques this part has measured separately: continuous
batching, paged cache, chunked prefill, quantised kernels, graph capture. Vendors report
the bundle's total, which is real and tells you nothing about which parts you need.

This listing adds the features one at a time and measures each one's marginal gain --
then adds them in a different order and gets different numbers for the same features
(eq:feature-credit-depends-on-order).

That is not a measurement error. It is what happens when techniques multiply rather than
add, and it is why "feature X gives 3x" is a claim that requires a baseline to mean
anything.
"""
import itertools

# Each feature's effect is a multiplier on achieved throughput, but several act
# on the SAME inefficiency, so applying one reduces what the next can recover.
# (name, inefficiency it removes, share of that inefficiency it removes)
FEATURES = [
    ("continuous batching", "slot idleness",   0.92),
    ("paged KV cache",      "slot idleness",   0.55),
    ("chunked prefill",     "phase stalls",    0.88),
    ("fp8 weights",         "weight traffic",  0.50),
    ("graph capture",       "launch overhead", 0.85),
]

# How much of a naive loop's time each inefficiency accounts for.
BUDGET = {
    "slot idleness":   0.46,
    "phase stalls":    0.21,
    "weight traffic":  0.18,
    "launch overhead": 0.09,
    "irreducible":     0.06,
}


def throughput(active):
    """Relative throughput with `active` features enabled.

    Time is the sum of each inefficiency's remaining share plus the irreducible
    part. Two features acting on one inefficiency cannot each remove all of it.
    """
    remaining = 0.0
    for cause, share in BUDGET.items():
        if cause == "irreducible":
            remaining += share
            continue
        left = 1.0
        for name, target, frac in FEATURES:
            if name in active and target == cause:
                left *= (1.0 - frac)
        remaining += share * left
    return 1.0 / remaining


print("A naive serving loop's time, by what it is wasted on.")
print()
print(f"{'inefficiency':>18}{'share of time':>16}   {'addressed by':<38}")
print("-" * 76)
for cause, share in BUDGET.items():
    who = ", ".join(n for n, t, _ in FEATURES if t == cause) or "nothing"
    print(f"{cause:>18}{share:>16.0%}   {who:<38}")

base = throughput(set())
full = throughput(set(n for n, _, _ in FEATURES))
print()
print(f"naive loop: {base:.2f}x   everything on: {full:.2f}x")

print()
print()
print("Adding features in the order a vendor's changelog lists them.")
print()
ORDER_A = ["continuous batching", "paged KV cache", "chunked prefill",
           "fp8 weights", "graph capture"]
print(f"{'added':>22}{'cumulative':>13}{'marginal gain':>16}{'credit':>10}")
print("-" * 62)
active = set()
prev = base
creditA = {}
for f in ORDER_A:
    active.add(f)
    now = throughput(active)
    creditA[f] = (now / prev, (now - prev) / (full - base))
    print(f"{f:>22}{now:>12.2f}x{now / prev:>15.2f}x"
          f"{(now - prev) / (full - base):>10.0%}")
    prev = now

print()
print()
print("The same five features, added in reverse order.")
print()
ORDER_B = list(reversed(ORDER_A))
print(f"{'added':>22}{'cumulative':>13}{'marginal gain':>16}{'credit':>10}")
print("-" * 62)
active = set()
prev = base
creditB = {}
for f in ORDER_B:
    active.add(f)
    now = throughput(active)
    creditB[f] = (now / prev, (now - prev) / (full - base))
    print(f"{f:>22}{now:>12.2f}x{now / prev:>15.2f}x"
          f"{(now - prev) / (full - base):>10.0%}")
    prev = now

print()
print()
print("Same features, same total, different attribution.")
print()
print("-" * 71)
print(f"{'feature':>22}{'credit in order A':>20}{'credit in order B':>20}"
      f"{'ratio':>9}")
for f in ORDER_A:
    a = creditA[f][1]
    b = creditB[f][1]
    print(f"{f:>22}{a:>20.0%}{b:>20.0%}"
          f"{max(a, b) / max(min(a, b), 1e-9):>9.1f}x")

print()
print()
print("The order-independent measure: what each feature is worth on its own,")
print("and what removing it costs from the full stack.")
print()
print(f"{'feature':>22}{'alone':>10}{'removing from full':>21}"
      f"{'Shapley share':>16}")
print("-" * 70)
ALL = [n for n, _, _ in FEATURES]
shap = {}
for f in ALL:
    alone = throughput({f}) / base
    without = full / throughput(set(ALL) - {f})
    # Shapley value: average marginal contribution over all orderings.
    total = 0.0
    count = 0
    for perm in itertools.permutations(ALL):
        act = set()
        for g in perm:
            before = throughput(act)
            act.add(g)
            if g == f:
                total += throughput(act) - before
                break
        count += 1
    shap[f] = total / count
    print(f"{f:>22}{alone:>9.2f}x{without:>20.2f}x"
          f"{(total / count) / (full - base):>16.0%}")

print()
print()
print("And the question that actually matters: which subset is worth building?")
print()
print(f"{'subset':>52}{'throughput':>13}{'features':>11}")
print("-" * 78)
best_by_size = {}
for k in range(1, len(ALL) + 1):
    best, bestv = None, 0.0
    for combo in itertools.combinations(ALL, k):
        v = throughput(set(combo))
        if v > bestv:
            best, bestv = combo, v
    best_by_size[k] = (best, bestv)
    label = ", ".join(b.split()[0] for b in best)
    print(f"{label:>52}{bestv:>12.2f}x{k:>11}")

print(f"""
The budget table is where the naive loop's time goes, and it is worth noticing that no
single cause dominates. Slot idleness is {BUDGET['slot idleness']:.0%}, phase stalls
{BUDGET['phase stalls']:.0%}, weight traffic {BUDGET['weight traffic']:.0%}, launch
overhead {BUDGET['launch overhead']:.0%}, and {BUDGET['irreducible']:.0%} is
irreducible.

Turning everything on takes throughput from {base:.2f}x to **{full:.2f}x**. That is the
number a serving stack advertises, and it is honest.

The two ordering tables are where honesty gets complicated. Read the credit column --
each feature's share of the total improvement.

In the first ordering, continuous batching is credited with
{creditA['continuous batching'][1]:.0%} of the gain and graph capture with
{creditA['graph capture'][1]:.0%}. In the reverse ordering, continuous batching gets
{creditB['continuous batching'][1]:.0%} and graph capture gets
{creditB['graph capture'][1]:.0%}.

**Same features, same workload, same total. The attributed credit differs by a factor of
{creditB['continuous batching'][1] / creditA['continuous batching'][1]:.1f} for one
feature and {creditA['graph capture'][1] / creditB['graph capture'][1]:.1f} for another,
purely from the order they were switched on**
(eq:feature-credit-depends-on-order).

The attribution table shows this is not confined to one feature. Whichever feature is
switched on first gets credit for the largest share of time, because it is the only one
operating against the full naive baseline -- and every feature after it works on what is
left.

That has a direct consequence for reading benchmarks. **"Feature X gives Nx" is not a
property of feature X**; it is a property of X and the baseline it was measured
against. A vendor comparing against a naive loop and a vendor comparing against their
previous release are reporting different quantities, and neither is wrong.

The Shapley column is the order-independent answer: average each feature's marginal
contribution over every possible ordering. By that measure
`{max(shap, key=lambda k: shap[k])}` is worth
{shap[max(shap, key=lambda k: shap[k])] / (full - base):.0%} of the total gain and
`{min(shap, key=lambda k: shap[k])}` is worth
{shap[min(shap, key=lambda k: shap[k])] / (full - base):.0%}.

It is also the wrong question for a build decision, which is why the last table exists.
A team is not choosing an attribution; it is choosing a subset to implement. The best
single feature gives {best_by_size[1][1]:.2f}x, the best two give
{best_by_size[2][1]:.2f}x, and the best three give {best_by_size[3][1]:.2f}x against the
full five's {full:.2f}x.

**Three of the five features capture
{(best_by_size[3][1] - base) / (full - base):.0%} of the available gain**, and the two
left out are the ones that overlap with something already chosen. That is the useful
form of the result: not which feature is best, but which ones stop being worth building
once you have the others.

The general lesson is worth stating outside this table. **Techniques that address the
same inefficiency are substitutes, not complements**, and a roadmap that budgets them
additively will overpromise by the amount they overlap. The way to catch it before
building is to name the inefficiency each item addresses -- which is what the first
column of the budget table does, and which almost no roadmap records.""")
```

## 9. Practical Example

Where a naive loop's time goes:

```
      inefficiency   share of time   addressed by                          
----------------------------------------------------------------------------
     slot idleness             46%   continuous batching, paged KV cache   
      phase stalls             21%   chunked prefill                       
    weight traffic             18%   fp8 weights                           
   launch overhead              9%   graph capture                         
       irreducible              6%   nothing                               
```

Everything enabled gives **4.87×**. Adding the features in one order:

```
                 added   cumulative   marginal gain    credit
--------------------------------------------------------------
   continuous batching        1.73x           1.73x       19%
        paged KV cache        1.80x           1.04x        2%
       chunked prefill        2.69x           1.50x       23%
           fp8 weights        3.55x           1.32x       22%
         graph capture        4.87x           1.37x       34%
```

And in reverse:

```
                 added   cumulative   marginal gain    credit
--------------------------------------------------------------
         graph capture        1.08x           1.08x        2%
           fp8 weights        1.20x           1.11x        3%
       chunked prefill        1.54x           1.28x        9%
        paged KV cache        2.53x           1.64x       25%
   continuous batching        4.87x           1.93x       61%
```

```
               feature   credit in order A   credit in order B    ratio
   continuous batching                 19%                 61%      3.2x
        paged KV cache                  2%                 25%     15.6x
       chunked prefill                 23%                  9%      2.6x
           fp8 weights                 22%                  3%      7.3x
         graph capture                 34%                  2%     16.0x
```

**Same features, same total, credit differing by up to 16×**
({{eq:feature-credit-depends-on-order}}). Whichever feature goes first is the only one
operating against the full naive baseline.

```mermaid {#fig:overlap caption="Features targeting the same inefficiency are substitutes: the second one added finds only what the first left. Features with distinct targets compose without interference."}
flowchart TD
  A["naive loop time"] --> B["slot idleness 46%"]
  A --> C["phase stalls 21%"]
  A --> D["weight traffic 18%"]
  A --> E["launch overhead 9%"]
  B --> F["continuous batching"]
  B --> G["paged KV cache"]
  F -.->|"substitutes"| G
  C --> H["chunked prefill"]
  D --> I["fp8 weights"]
  E --> J["graph capture"]
```

The build decision is a subset choice, not an attribution:

```
                                              subset   throughput   features
------------------------------------------------------------------------------
                                         continuous         1.73x           1
                                continuous, chunked         2.69x           2
                           continuous, chunked, fp8         3.55x           3
                    continuous, chunked, fp8, graph         4.87x           4
              continuous, paged, chunked, fp8, graph        4.87x           5
```

**Four of the five features capture the entire gain**
({{eq:overlapping-techniques-are-substitutes}}) — the fifth is the one overlapping with
something already chosen. That is the useful form: not which feature is best, but which
stops being worth building once you have the others.

The second listing turns to a term the roofline omits.

```python {tier=A name=db2}
"""The roofline model has no term for the time before any arithmetic happens.

ch:inf-cpu-gpu and ch:inf-gpu-memory modelled a step as max(traffic/bandwidth,
FLOPs/peak). Both terms are proportional to work. Neither describes the fixed cost of
ASKING the device to do the work: each kernel must be launched, its arguments marshalled,
its dependencies resolved.

A transformer decode step launches hundreds of kernels, and that cost is FIXED -- it
does not scale with batch size. Since ch:inf-cpu-gpu showed the step time is also fixed
throughout the memory-bound regime, the two are a constant ratio: launch overhead is the
same share of every decode step a production system runs
(eq:launch-overhead-is-a-floor).

This listing measures where the floor sits, why it explains benchmarks that undershoot
the roofline, and what graph capture recovers.
"""
LAYERS = 32
KERNELS_PER_LAYER = 14        # projections, norms, activations, cache writes
LAUNCH_US = 4.2               # per-kernel launch and dispatch, unbatched
GRAPH_US = 0.35               # per-kernel cost when the graph is pre-captured
WEIGHT_BYTES = 14.0e9
BANDWIDTH = 3.35e12
PEAK = 9.89e14
PARAMS = 7.0e9
BATCHES = [1, 8, 32, 128, 256, 512, 1024, 2048]

KERNELS = LAYERS * KERNELS_PER_LAYER


def roofline_ms(batch):
    t_mem = WEIGHT_BYTES / BANDWIDTH
    t_flop = 2.0 * PARAMS * batch / PEAK
    return max(t_mem, t_flop) * 1000.0


def launch_ms(per_kernel_us):
    return KERNELS * per_kernel_us / 1000.0


print("A %d-layer decode step launches %d kernels." % (LAYERS, KERNELS))
print("Unbatched launch cost: %.2f us each, %.2f ms total."
      % (LAUNCH_US, launch_ms(LAUNCH_US)))
print("Captured in a graph:   %.2f us each, %.2f ms total."
      % (GRAPH_US, launch_ms(GRAPH_US)))
print()
print("Step time: what the roofline predicts, and what it costs with launches.")
print()
print(f"{'batch':>8}{'roofline ms':>14}{'launch ms':>12}{'actual ms':>12}"
      f"{'launch share':>15}{'roofline error':>17}")
print("-" * 80)
tab = {}
L = launch_ms(LAUNCH_US)
for b in BATCHES:
    rf = roofline_ms(b)
    act = rf + L
    tab[b] = (rf, act, L / act)
    print(f"{b:>8}{rf:>14.2f}{L:>12.2f}{act:>12.2f}{L / act:>15.1%}"
          f"{rf / act - 1.0:>16.1%}")

print()
print()
print("Throughput, which is where the error becomes visible as a benchmark that")
print("undershoots.")
print()
print(f"{'batch':>8}{'roofline tok/s':>17}{'actual tok/s':>15}"
      f"{'shortfall':>12}")
print("-" * 54)
short = {}
for b in BATCHES:
    rf, act, _ = tab[b]
    r_t = b / (rf / 1000.0)
    a_t = b / (act / 1000.0)
    short[b] = (r_t, a_t, 1.0 - a_t / r_t)
    print(f"{b:>8}{r_t:>17.0f}{a_t:>15.0f}{1.0 - a_t / r_t:>11.0%}")

print()
print()
print("Graph capture removes most of the per-launch cost by recording the kernel")
print("sequence once and replaying it.")
print()
G = launch_ms(GRAPH_US)
print(f"{'batch':>8}{'plain ms':>11}{'captured ms':>14}{'speedup':>10}"
      f"{'captured tok/s':>17}{'vs roofline':>14}")
print("-" * 76)
cap = {}
for b in BATCHES:
    rf, act, _ = tab[b]
    capt = rf + G
    cap[b] = (capt, b / (capt / 1000.0))
    print(f"{b:>8}{act:>11.2f}{capt:>14.2f}{act / capt:>9.2f}x"
          f"{b / (capt / 1000.0):>17.0f}{(b / (capt / 1000.0)) / short[b][0]:>13.1%}")

print()
print()
print("Where the floor binds: the batch below which launch overhead exceeds the")
print("arithmetic it is launching.")
print()
print(f"{'per-kernel us':>15}{'launch ms':>12}{'crossover batch':>18}"
      f"{'max tok/s at floor':>21}")
print("-" * 68)
floor = {}
for us in (8.0, 4.2, 2.0, 1.0, GRAPH_US):
    lm = launch_ms(us)
    # Crossover: the batch at which roofline time equals launch time.
    b = 1
    while roofline_ms(b) < lm and b < 100000:
        b *= 2
    floor[us] = (lm, b, 1000.0 / lm)
    print(f"{us:>15.2f}{lm:>12.2f}{b:>18}{1000.0 / lm:>21.0f}")
print()
print("(max tok/s at floor is the ceiling from launches alone, at batch 1)")

print()
print()
print("And why this matters more as models get faster: the arithmetic shrinks and")
print("the launch cost does not.")
print()
print(f"{'model':>14}{'weights GB':>13}{'roofline ms':>14}{'launch ms':>12}"
      f"{'launch share':>15}")
print("-" * 70)
share = {}
for label, gb in (("70B bf16", 140.0), ("13B bf16", 26.0), ("7B bf16", 14.0),
                  ("7B fp8", 7.0), ("3B int4", 1.5)):
    rf = gb * 1e9 / BANDWIDTH * 1000.0
    share[label] = L / (rf + L)
    print(f"{label:>14}{gb:>13.1f}{rf:>14.2f}{L:>12.2f}"
          f"{L / (rf + L):>15.1%}")

print(f"""
The first table is the term the roofline omits. A {LAYERS}-layer step launches
{KERNELS} kernels, and at {LAUNCH_US:.1f} microseconds each that is
{L:.2f}ms of pure dispatch before any arithmetic happens
(eq:launch-overhead-is-a-floor).

At batch {1} the roofline predicts {tab[1][0]:.2f}ms and the step actually takes
{tab[1][1]:.2f}ms -- launch overhead is **{tab[1][2]:.1%} of the step**, and the
roofline is wrong by {1.0 - tab[1][0] / tab[1][1]:.0%}.

Now read down that column, because it does something the intuition does not expect. At
batch {8} launch overhead is {tab[8][2]:.1%}. At batch {32}, {tab[32][2]:.1%}. At batch
{256}, {tab[256][2]:.1%}. **It does not shrink at all.**

The reason is ch:inf-cpu-gpu's result: below the balance point the step time is
*constant* in batch size, because the weight read dominates and it happens once
regardless. Launch overhead is also constant. Two constants have a constant ratio, so
**launch overhead is a fixed share of every decode step in the entire memory-bound
regime** -- which is where decode lives.

It falls only once the batch pushes the step into the compute-bound regime: at batch
{1024} it is {tab[1024][2]:.1%} and at batch {2048}, {tab[2048][2]:.1%}. Those are batch
sizes above what ch:inf-gpu-memory's capacity frontier permits at any useful context
length.

The throughput table is how this shows up in practice, and it inherits the same
flatness: a benchmark undershoots the roofline by {short[1][2]:.0%} at batch {1},
{short[8][2]:.0%} at batch {8}, and {short[256][2]:.0%} at batch {256}.

Teams that compute an expected throughput from bandwidth and measure something lower
usually conclude the memory system is underperforming, and go looking for it in the
memory system. It is not there. **The device is waiting to be told what to do**, and no
amount of batching reveals or fixes it.

Graph capture is the fix, and it is a large one at the batch sizes where it matters. It
records the kernel sequence once and replays it as a unit, cutting per-kernel cost from
{LAUNCH_US:.1f} to {GRAPH_US:.2f} microseconds. At batch {1} that is
{tab[1][1] / cap[1][0]:.2f}x, and it is the same {tab[256][1] / cap[256][0]:.2f}x at
batch {256}, for the same reason the overhead share was flat. At batch {2048}, where
compute finally dominates, it is {tab[2048][1] / cap[2048][0]:.2f}x.

**Graph capture is the one technique in this part whose benefit does not depend on
batch size.** ch:inf-batching's continuous batching and ch:inf-cpu-gpu's
arithmetic-intensity argument both need a large batch to help; ch:inf-parallelism's
dimensions need a fast link. Graph capture needs nothing, and delivers the same
{tab[32][1] / cap[32][0]:.2f}x across the whole operating range.

That makes it a complement to everything else rather than a substitute, which is
unusual in this part -- and it is why the previous listing's overlap analysis put
graph capture in a category of its own.

The floor table gives the operating constraint directly. At {LAUNCH_US:.1f} microseconds
per kernel, the launch cost alone caps a batch-1 deployment at
{floor[LAUNCH_US][2]:.0f} tokens a second no matter what the hardware does. Captured, the
same cap is {floor[GRAPH_US][2]:.0f}.

**That is a ceiling the roofline cannot express**, because it does not scale with any
quantity the roofline measures. A faster device does not raise it. More bandwidth does
not raise it. Only launching fewer kernels or launching them more cheaply does.

The last table is why this is getting worse rather than better. Launch cost is fixed;
the arithmetic it launches is not. A {140.0:.0f} GB model spends
{share['70B bf16']:.1%} of its step on launches; a {1.5:.1f} GB one spends
{share['3B int4']:.1%}.

So **every optimisation that makes the model cheaper makes launch overhead a larger
share of what remains.** Quantisation, smaller models, sparsity, better kernels -- each
one shrinks the term the roofline models and leaves the term it does not. A team that
quantises a small model and measures less speedup than the memory arithmetic predicted
has met this, and the missing factor is in the last column rather than in the
quantisation.""")
```

A 32-layer step launches **448 kernels** at 4.2 μs each — **1.88 ms** before any
arithmetic:

```
   batch   roofline ms   launch ms   actual ms   launch share   roofline error
--------------------------------------------------------------------------------
       1          4.18        1.88        6.06          31.0%          -31.0%
       8          4.18        1.88        6.06          31.0%          -31.0%
      32          4.18        1.88        6.06          31.0%          -31.0%
     128          4.18        1.88        6.06          31.0%          -31.0%
     256          4.18        1.88        6.06          31.0%          -31.0%
     512          7.25        1.88        9.13          20.6%          -20.6%
    1024         14.50        1.88       16.38          11.5%          -11.5%
    2048         28.99        1.88       30.87           6.1%           -6.1%
```

**The share does not shrink with batch** ({{eq:launch-overhead-is-a-floor}}). Below the
balance point, step time is constant and launch cost is constant, so the ratio is
constant — **31.0%** from batch 1 to batch 256. It falls only past the balance point, at
batch sizes {{ch:inf-gpu-memory}}'s capacity frontier does not permit at useful context
lengths.

So the roofline underpredicts by **31%** across the entire production range, and a team
computing expected throughput from bandwidth will find it in the memory system where it
is not.

Graph capture:

```
   batch   plain ms   captured ms   speedup   captured tok/s   vs roofline
----------------------------------------------------------------------------
       1       6.06          4.34     1.40x              231        96.4%
      32       6.06          4.34     1.40x             7380        96.4%
     256       6.06          4.34     1.40x            59042        96.4%
    2048      30.87         29.15     1.06x            70263        99.5%
```

**1.40× at every batch in the memory-bound regime.** Graph capture is the only technique
in this part whose benefit does not depend on batch size, context length, or
interconnect.

And the floor:

```
  per-kernel us   launch ms   crossover batch   max tok/s at floor
--------------------------------------------------------------------
           8.00        3.58               256                  279
           4.20        1.88               128                  531
           2.00        0.90                64                 1116
           1.00        0.45                32                 2232
           0.35        0.16                16                 6378
```

At 4.2 μs per kernel, launches alone cap a batch-1 deployment at **531 tokens/second**
regardless of hardware. A faster device does not raise it; more bandwidth does not raise
it.

The closing table is why this is getting worse:

```
         model   weights GB   roofline ms   launch ms   launch share
----------------------------------------------------------------------
      70B bf16        140.0         41.79        1.88            4.3%
      13B bf16         26.0          7.76        1.88           19.5%
       7B bf16         14.0          4.18        1.88           31.0%
        7B fp8          7.0          2.09        1.88           47.4%
       3B int4          1.5          0.45        1.88           80.8%
```

**Every optimisation that makes the model cheaper makes launch overhead a larger share
of what remains** ({{eq:cheaper-models-raise-the-overhead-share}}). A 3B int4 model
spends **80.8%** of its step telling the GPU what to do.

## 10. Production Considerations

Write down which inefficiency each roadmap item addresses. It is one column, it takes
minutes, and it is the only thing that catches an overlapping pair before both are
budgeted.

Measure your own baseline before believing any published speedup. The number is a
property of the baseline as much as the feature, and yours is not the vendor's.

Compute the launch floor for your model and layer count. It is $K\lambda$, and it caps
batch-1 throughput at a value no hardware purchase changes.

Enable graph capture. It is the only technique here that helps unconditionally, and its
benefit rises as everything else succeeds.

Re-measure after quantising. The predicted speedup from halving weight bytes assumes the
launch term is zero, and {{eq:cheaper-models-raise-the-overhead-share}} says it is not.

Count kernels. Fusion reduces $K$ where graph capture reduces $\lambda$, and the two
compose; a stack reporting a kernel count per token is telling you something useful that
most do not report.

Watch for shape bucketing under graph capture. The padding it requires can consume the
slot utilisation continuous batching recovered, and the interaction is invisible in
either feature's own metrics.

## 11. Common Mistakes

**Multiplying published speedups.** Overlapping features are substitutes; the product
overpromises by the overlap.

**Reading a feature's gain as a property of the feature.** It is a property of the
feature and the baseline.

**Expecting launch overhead to dilute with batch.** It does not, in the memory-bound
regime, because step time does not grow either.

**Debugging a roofline shortfall in the memory system.** The missing term is dispatch,
and it is exactly the size of the shortfall.

**Assuming quantisation's speedup follows the memory arithmetic.** It follows the
memory arithmetic *plus a fixed term that quantisation does not touch*.

**Choosing a stack on a benchmark rather than on which inefficiencies it addresses.**
The benchmark's workload determines which features mattered.

## 12. Failure Modes

**Roadmap overpromise.** Five features with published gains are budgeted multiplicatively
and deliver substantially less, with no single item having underperformed.

**Bucketing-induced utilisation loss.** Graph capture's fixed shapes force padding, the
achieved batch falls, and continuous batching's benefit silently reverses.

**Kernel-count regression.** A model or framework change adds kernels per layer, launch
cost rises proportionally, and the effect appears as a throughput regression with no
change to arithmetic or memory.

**Small-model surprise.** A team moves to a smaller quantised model expecting
proportional speedup and gets a fraction of it, because the launch term now dominates.

**Stack selection on the wrong workload.** A stack chosen on a short-prompt benchmark
underperforms on long-context production traffic, because the inefficiencies differ.

## 13. Alternatives

**Build the subset rather than adopting a stack.** Viable when the workload has one or
two dominant inefficiencies, and it avoids operating a large dependency. The subset
table says how much of the gain a partial build captures.

**Kernel fusion instead of graph capture.** Attacks $K$ rather than $\lambda$;
{{cite:dao2022flash}} is the canonical instance. Composes with capture rather than
replacing it.

**A compiled execution path.** Trades flexibility for a lower kernel count and better
capture, which is TensorRT-LLM's position. Costs recompilation on every shape or model
change.

**Larger batches to dilute overhead.** Does not work below the balance point, which is
the whole finding, and past it is bounded by
{{eq:batch-times-context-is-the-budget}}.

**Accepting the floor.** For a genuinely single-stream deployment, 531 tokens per second
may be enough, and the engineering to change it is substantial. Knowing the number is
what makes that a decision.

## 14. Evaluation

Report speedups against a stated baseline, always. A speedup without a baseline is not a
measurement.

Publish the inefficiency each feature addresses alongside its gain. It is what makes two
teams' numbers comparable.

Measure kernel count and per-kernel launch cost directly. Both are obtainable from a
profiler and neither appears in any standard serving metric.

Compare achieved throughput against the roofline *plus* the launch term, not against the
roofline. The residual after that correction is the real unexplained gap.

Re-run the decomposition after any model change. The inefficiency shares move with model
size, and {{eq:cheaper-models-raise-the-overhead-share}} says they move systematically.

## 15. Advanced Concepts

The decomposition model treats each inefficiency's share as fixed, but the shares depend
on the workload and change as features are enabled. Chunked prefill, for instance,
converts some phase-stall time into slot time, which changes what continuous batching
has left to recover. A faithful model would have features that *move* time between
inefficiencies rather than only removing it, and the resulting attribution would be
order-dependent in a second, harder way. The qualitative conclusion — overlapping
features are substitutes — survives, but the specific credit numbers become even less
transferable.

The launch model assumes launches serialise with computation. Modern runtimes overlap
them: while one kernel executes, the next is being dispatched. In the limit of perfect
overlap the launch term disappears from the critical path entirely, and
{{eq:launch-overhead-is-a-floor}} becomes an upper bound rather than an estimate. Real
overlap is partial, and the degree depends on how long each kernel runs relative to its
dispatch — which for decode's small kernels is unfavourable. The honest statement is
that the launch term lies between zero and $K\lambda$, and measuring where is what a
profiler is for.

A more subtle limitation of the launch model is that it treats $\lambda$ as uniform
across kernels. It is not: a small elementwise kernel and a large matrix multiply have
similar dispatch costs but wildly different durations, so the *share* of a kernel's time
spent on its own launch varies by two orders of magnitude within one layer. That means
the effective $K\lambda$ is dominated by the many small kernels -- norms, activations,
residual adds -- rather than by the few large ones, and it is exactly those small
kernels that fusion targets most easily. So the practical route to reducing $K\lambda$
is not to fuse the expensive operations but the cheap ones, which is the opposite of
where optimisation attention usually goes.

There is a composition worth noting between graph capture and
{{ch:inf-distributed}}'s failure story. A captured graph encodes the collective
operations of a tensor-parallel group, so a rank rejoining after replacement invalidates
the capture and forces a re-capture before serving resumes. That extends the recovery
time beyond the weight-loading term, and it is a cost that appears in neither chapter's
model.

## 16. Connection to Previous Chapters

{{eq:decode-is-bandwidth-bound}} from {{ch:inf-cpu-gpu}} is what makes
{{eq:launch-overhead-is-a-floor}}'s constancy result work. Without the flat step time
below the balance point, launch overhead would dilute with batch and the chapter's
central surprise would not exist.

{{eq:static-batching-pays-for-the-longest}} and {{eq:chunk-size-has-a-cliff}} from
{{ch:inf-batching}} are two of the five features decomposed here, and the decomposition
is what says whether both are worth building.

{{eq:roofline-has-multiple-ridges}} from {{ch:inf-gpu-memory}} refined the model in one
direction; this chapter refines it in another, and both refinements make it predict
lower.

{{eq:batch-times-context-is-the-budget}} bounds the batch that would be needed to dilute
launch overhead, and the bound is below what would be required.

## 17. Exercises

1. Compute the Shapley value for a three-feature system where two share an inefficiency.
   How does it differ from either ordering's credit?

2. For your own model and framework, count kernels per token with a profiler. What is
   $K\lambda$, and what batch-1 throughput does it cap?

3. Derive the batch at which launch overhead falls below 10% of step time, for a 13B
   model at fp8.

4. Extend the first listing so chunked prefill moves time from phase stalls into slot
   idleness. Does the ordering effect get better or worse?

5. Measure the overlap between graph capture and continuous batching in a stack you have
   access to, by enabling each alone and both together.

## 18. Interview Questions

1. A vendor says feature X gives 3×. What do you need to know before using that number?

2. Our throughput is 30% below what the memory bandwidth predicts, at every batch size.
   Where would you look?

3. Why does launch overhead not shrink when you increase the batch?

4. We quantised a 3B model to int4 and got a 1.3× speedup instead of the predicted 4×.
   Explain.

5. Which single feature would you build first, and what would change your answer?

## 19. Research Questions

1. How much launch overhead do modern runtimes actually hide through overlap, and how
   does that vary with kernel duration?

2. Can the inefficiency shares in a decomposition be measured directly rather than
   inferred from ablations?

3. What is the right way to attribute credit among overlapping features for roadmap
   purposes, given that Shapley answers a different question?

4. How much of graph capture's benefit survives the shape bucketing that continuous
   batching forces?

## 20. Chapter Summary

A serving stack bundles techniques this part measured separately, and its headline
number is real: **4.87×** over a naive loop. The decomposition is not.

Adding five features in two orders gives continuous batching **19%** and **61%** of the
credit, and graph capture **34%** and **2%**
({{eq:feature-credit-depends-on-order}}) — differences of up to **16×** from ordering
alone. Features sharing an inefficiency are **substitutes**, so a roadmap that multiplies
published gains overpromises by the overlap
({{eq:overlapping-techniques-are-substitutes}}); four of the five features here capture
the entire gain.

The roofline omits a term. A 32-layer step launches **448 kernels** costing **1.88 ms**,
which is **31.0%** of a batch-1 step — and **31.0%** at batch 8, 32, 128 and 256, because
step time is constant below the balance point and so is launch cost
({{eq:launch-overhead-is-a-floor}}). Batching does not dilute it.

Graph capture cuts it to **1.40×** faster at every batch in that regime, making it the
only technique in Part XXIII whose benefit is unconditional. And the launch floor caps
batch-1 throughput at **531 tokens/second** at 4.2 μs per kernel, regardless of hardware.

Finally, the share grows as models get cheaper: **4.3%** for a 70B bf16 model and
**80.8%** for a 3B int4 one
({{eq:cheaper-models-raise-the-overhead-share}}). Every success at reducing the term the
roofline models enlarges the term it does not.

Both halves of this chapter are about the difference between a number and what it
attributes. A stack's speedup is a real measurement of a bundle; a feature's credit
within that bundle is an artefact of the order it was switched on. The roofline is a
real model of two costs; its error is a third cost it does not name. In both cases
the number is trustworthy and the story attached to it is not, and the correction in
both cases is to name the thing being addressed before quoting the improvement.

Carry forward: **name the inefficiency before budgeting the feature**, and **the roofline
is short by exactly the launch share**.

## 21. Further Reading

- {{cite:kwon2023pagedattention}} — paging and continuous batching, two of the five
  features and the overlapping pair.
- {{cite:agrawal2023sarathi}} — chunked prefill, the phase-stall feature.
- {{cite:dao2022flash}} — kernel fusion, which attacks the kernel count rather than the
  launch cost.
- {{cite:leviathan2023speculative}} — speculative decoding, which the taxonomy predicts
  overlaps with quantisation rather than with capture.
