---
id: ft-qlora-peft
number: 132
part: XIV
tier: full
status: draft
requires: [ft-lora, ft-sft, tf-complexity, dl-optimizers]
provides: [training-memory-budget, optimiser-state-dominance,
           backward-still-full, peft-expressiveness, quantised-base-training,
           serial-versus-parallel-adaptation, peft-selection]
citations: [dettmers2023qlora, houlsby2019adapters, li2021prefixtuning,
            hu2021lora, biderman2024loralearnsless, kirkpatrick2017ewc]
---

## 1. Learning Objectives

By the end of this chapter you will be able to decompose training memory into the
terms that actually bind, and say which method removes which; explain why LoRA and
quantisation compose rather than compete; state why a 99.9% cut in trainable
parameters buys only a **33%** cut in compute; choose between PEFT families by
**what kind of change they can express** rather than by parameter count; and
recognise the conditioning penalty that makes serial adapters worse than parallel
ones on real weight matrices.

## 2. Why This Matters

{{ch:ft-lora}} explained what LoRA *is*. This chapter is about why anyone can
afford to run it, and it opens by taking apart the number everyone quotes.

"0.1% of parameters are trainable" is true and it is the wrong resource.
{{sec:9-practical-example}} decomposes a 7B fine-tune's memory and finds
**126 GB** of model state, of which the weights are **14 GB** and everything the
optimiser needs is **112 GB — 89% of the bill.** The model is the small term.

**That is what LoRA removes**, because optimiser state scales with *trainable*
parameters: 126 GB → **14.1 GB**. And notice what is left standing — the frozen
weights, untouched, because you still have to hold the model.

**Which is exactly what {{cite:dettmers2023qlora}} attacks.** Store the frozen
base at 4 bits instead of 16 and the residual collapses too: **3.6 GB, 0.03× full
fine-tuning.** The two savings target different terms and **compose** — neither
would put a 70B model on one card, and together they do.

Then two corrections the headline hides.

**Activation memory is untouched by any of this**, and at 70B with batch 8 it is
**81.6 GB checkpointed against 35 GB of quantised weights.** The weights fit; the
run may not.

**And compute drops by 33%, not 99.9%**, because the backward pass still traverses
every frozen layer to deliver gradients to the adapters. **These are memory
techniques, not speed techniques**, and budgeting on the parameter ratio produces
schedules wrong by two orders of magnitude.

Finally, the selection question. {{sec:9-practical-example}} shows PEFT methods
are **not points on a quality curve** — they are different hypothesis classes, and
one that cannot express your change **does not improve with budget**: a pure
output offset stays at **0.5856** error for LoRA at rank 1 *and* rank 32.

{{maturity:ESTABLISHED}} LoRA, QLoRA, adapters. {{maturity:MATURE}} The memory
decomposition. {{maturity:EMERGING}} Choosing methods by expressiveness rather
than budget.

## 3. Prerequisites

{{ch:ft-lora}} for the parameterisation and for {{eq:forgetting-quadratic}}, which
this chapter's conditioning result feeds back into; {{ch:ft-sft}} for what a
training step costs; {{ch:tf-complexity}} for activation memory, which this
chapter shows PEFT does not touch; {{ch:dl-optimizers}} for why Adam carries two
moments.

**Quantisation theory is {{part:15}}'s.** This chapter uses the result — a frozen
base can be stored at 4 bits without destroying adaptation — and forward-references
{{ch:q-theory}} for why, and {{ch:q-memory-math}} for the inference-side
arithmetic.

## 4. Intuitive Explanation

### The bill, itemised

Ask "how much memory does fine-tuning need" and the honest answer is four numbers,
not one:

```text
   TERM                 SCALES WITH              REMOVED BY
   ──────────────────   ──────────────────────   ────────────────────
   frozen weights       total parameters         quantisation
   master weights       trainable parameters     LoRA
   gradients            trainable parameters     LoRA
   optimiser moments    trainable parameters     LoRA
   activations          batch x length x depth   nothing here
```

**Once the table is written this way the whole chapter is visible.** LoRA and
quantisation are not competing techniques and not variants of one idea — they
remove *different rows*. That is why QLoRA is their conjunction rather than an
improvement on either.

And it explains the failure everyone hits eventually: the last row has no entry.
You can shrink the model to nothing and still fail at the first backward pass,
because activations depend on the batch and the sequence, not on what is
trainable.

### Why the compute saving is small

The intuition that trainable parameters drive cost is wrong, and the reason is
worth holding onto.

A backward pass computes two things: the gradient with respect to the **inputs**
of each layer, and the gradient with respect to its **weights**. Freezing weights
removes the second. It cannot remove the first, because the adapter in layer 3
gets its gradient from layer 40 — **the chain has to run all the way down.**

Weight gradients are roughly a third of the total work. So:

> **Trainable parameters: down 99.9%. Compute: down 33%.**

A ten-hour full fine-tune becomes about seven hours. Not ten minutes.

### PEFT methods are hypothesis classes, not budget tiers

Here is the comparison the parameter tables cannot make. Three families, on the
same layer:

- **LoRA**, parallel: $y = x(W_0 + BA)$ — adds to the **map**.
- **Adapters** {{cite:houlsby2019adapters}}, serial: $y = (xW_0)(I + B'A')$ —
  transforms the **output**.
- **Prefix/prompt tuning** {{cite:li2021prefixtuning}} and bias-only:
  $y = xW_0 + b$ — adds an **offset**.

{{sec:9-practical-example}} runs all three at matched budget on changes of
different kinds, and the result is not a ranking:

```text
   task                 LoRA r=4   adapter r=4   bias only
   ──────────────────   ────────   ───────────   ─────────
   map change             0.0000        0.0000      0.8328
   output offset          0.5856        0.5856      0.0000
```

**Each is near-exact on the change it can express and useless on the one it
cannot.** And raising the budget does not help: LoRA on the offset task scores
**0.5856 at rank 1 and 0.5856 at rank 32.** The missing capability is not
capacity — it is a term the parameterisation does not have.

**So the selection procedure inverts.** "Which method is best per parameter" is
answerable only after "what kind of change does my task need". Methods acting on
activations *steer a model through behaviour it already has*; methods acting on
the map can *install behaviour that was not reachable*. A task needing the second
will look like a data problem under the first, forever.

### Why parallel beat serial, for a reason nobody cites

For an invertible layer, $W_0 + \Delta = W_0(I + W_0^{-1}\Delta)$, so serial and
parallel adaptation can express the same changes. They are equally expressive —
{{sec:9-practical-example}} confirms it, with identical errors to four decimals.

**But not equally cheap.** The serial form must represent $W_0^{-1}\Delta$, whose
norm grows with the condition number of $W_0$: measured at **1.0×, 4.5×, 32.3×,
306.3×** as $\text{cond}(W_0)$ goes 1 → 1000.

The change is the same change. Expressing it from the output side costs 306×
more movement — which is harder to optimise, and by {{ch:ft-lora}}'s
{{eq:forgetting-quadratic}} **forgets more to achieve the same adaptation.**

Real transformer weight matrices are badly conditioned. **That, not the latency
argument usually given, is the deeper reason the parallel form won.**

## 5. Formal Explanation

### 5.1 The training memory budget

For $P$ total parameters, $P_a$ trainable, mixed-precision Adam:

$$ M_{\text{state}} = \underbrace{b_w P}_{\text{frozen}} + \underbrace{4P_a}_{\text{master}} + \underbrace{4P_a}_{\text{grad}} + \underbrace{8P_a}_{m,\,v} = b_w P + 16 P_a $$ (eq:training-memory)

with $b_w = 2$ for bf16 and $b_w = 0.5$ for 4-bit. **{{eq:training-memory}} is the
whole chapter in one line**: LoRA acts on $P_a$, quantisation acts on $b_w$, and
they are separate factors of separate terms.

Full fine-tuning has $P_a = P$, giving $18P$; the measured 7B figure of 126 GB is
$18 \times 7\text{e}9$.

### 5.2 Activations, and why nothing here helps

$$ M_{\text{act}} \approx c \cdot L \cdot (B \cdot S) \cdot d_{\text{model}} \cdot b $$ (eq:activation-memory-unchanged)

for $L$ layers, batch $B$, sequence $S$. **No term is $P_a$.** Gradient
checkpointing trades an extra forward pass for a roughly $\sqrt{L}$ reduction, and
that is the only lever this chapter leaves you.

$$ M_{\text{total}} = M_{\text{state}} + M_{\text{act}} $$

and past a certain scale the second term dominates, which is why the fine print on
"70B on a consumer card" is always a sequence length.

### 5.3 Compute is not saved

Per token, in units of $P$ FLOPs:

$$ C_{\text{full}} = \underbrace{2}_{\text{fwd}} + \underbrace{2}_{\partial/\partial x} + \underbrace{2}_{\partial/\partial W} = 6, \qquad C_{\text{PEFT}} = 2 + 2 + \epsilon \approx 4 $$ (eq:backward-still-full)

$$ \text{speedup} = \frac{6}{4} = 1.5\times \quad\text{regardless of } P_a $$

**{{eq:backward-still-full}} is independent of rank**, because $\partial/\partial
x$ must traverse every layer to reach any adapter. Quantisation can make it
*worse*: dequantising the frozen weights on each use adds work that a bf16 base
does not pay.

### 5.4 Expressiveness as a subspace

Each method restricts the achievable function set:

$$ \mathcal{F}_{\text{LoRA}} = \{x \mapsto x(W_0 + BA)\}, \quad \mathcal{F}_{\text{bias}} = \{x \mapsto xW_0 + b\} $$

These are **not nested**, and neither contains the other:

$$ x \mapsto xW_0 + c \notin \mathcal{F}_{\text{LoRA}} \;\;\forall r, \qquad x \mapsto x(W_0 + \Delta) \notin \mathcal{F}_{\text{bias}} \;\;\forall b $$ (eq:peft-expressiveness)

**{{eq:peft-expressiveness}} is why the budget sweep is flat.** The error floor is
$\inf_{f \in \mathcal{F}} \|f - f^{*}\|$, a distance to a *set*, and enlarging $r$
enlarges $\mathcal{F}_{\text{LoRA}}$ in a direction that does not approach $f^{*}$.

This is the same shape as {{ch:ft-lora}}'s {{eq:capacity-floor}} with a sharper
cause: there, more rank eventually sufficed; here, no rank ever does.

### 5.5 Serial and parallel are equivalent, and unequal

For invertible $W_0$:

$$ W_0 + \Delta = W_0\left(I + W_0^{-1}\Delta\right), \qquad \text{rank}(W_0^{-1}\Delta) = \text{rank}(\Delta) $$ (eq:serial-parallel-equivalence)

so the two families reach the same functions at the same rank. But

$$ \frac{\|W_0^{-1}\Delta\|}{\|\Delta\|} \le \frac{1}{\sigma_{\min}(W_0)}, \qquad \text{and grows like } \kappa(W_0) \text{ in the worst case} $$ (eq:conditioning-penalty)

**{{eq:conditioning-penalty}} is the measured 306×.** Combined with
{{ch:ft-lora}}'s {{eq:forgetting-quadratic}}, which makes forgetting grow as
$\|\Delta\|^2$, the serial form pays a **squared** penalty in disturbance to the
base model for the same adaptation.

### 5.6 What quantising the base does, and does not, break

{{cite:dettmers2023qlora}}'s claim is that a base stored at 4 bits still supports
full-quality adaptation, because gradients flow to the **adapters**, which stay in
higher precision:

$$ y = x\,\text{dequant}(W_0^{q}) + \tfrac{\alpha}{r} x BA, \qquad \frac{\partial \mathcal{L}}{\partial B}\ \text{computed in bf16} $$ (eq:qlora-forward)

**The quantisation error enters as a fixed perturbation of the frozen map, not as
noise in the gradient.** So the adapter learns to compensate for it — it is
adapting $W_0^{q}$, not $W_0$, and it is trained on exactly the model it will be
used with.

> **IMPORTANT:** That last clause is the practical constraint people violate. An
> adapter trained against a 4-bit base is a delta *for that quantisation*. Merging
> it into the bf16 original, or serving it against a differently-quantised copy,
> is not the model that was trained. {{ch:ft-lora}}'s {{eq:lora-merge}} assumes an
> identity that quantisation breaks.

## 6. Mathematical Foundation

### 6.1 Where the crossover is

From {{eq:training-memory}}, LoRA's state saving is

$$ \frac{b_w P + 16 P_a}{18 P} \xrightarrow{P_a \ll P} \frac{b_w}{18} $$ (eq:peft-memory-limit)

which is $2/18 = 0.11$ for bf16 and $0.5/18 = 0.028$ for 4-bit — **exactly the
0.11× and 0.03× measured.** Note what {{eq:peft-memory-limit}} says: past a very
small $P_a$, **rank stops mattering for memory.** Going from rank 8 to rank 64
changes the total by well under a percent, so the memory argument does not
constrain the rank choice, and {{ch:ft-lora}}'s capacity argument should.

### 6.2 When activations dominate

Setting {{eq:activation-memory-unchanged}} against the quantised state:

$$ c L B S d\, b > 0.5 P \iff B S > \frac{0.5 P}{c L d b} $$ (eq:activation-crossover)

With $P \approx 12 L d^2$ for a transformer, the right side is $\propto d / (c b)$
— **independent of depth.** So the sequence-length budget at which activations
overtake weights is a property of the model's *width*, and a deeper model does not
buy you more room.

### 6.3 The compute ratio, exactly

If a fraction $\phi$ of the model's matmuls have trainable weights,

$$ C = 2 + 2 + 2\phi \quad\Rightarrow\quad \frac{C_{\text{full}}}{C_{\text{PEFT}}} = \frac{6}{4 + 2\phi} $$ (eq:compute-ratio)

At $\phi = 0$, $1.5\times$. **{{eq:compute-ratio}} is bounded above by 1.5
whatever you freeze**, which is the sentence to remember when someone proposes
PEFT as a throughput fix.

> **MATH NOTE:** {{eq:serial-parallel-equivalence}} requires $W_0$ invertible.
> Attention projections are square and generically invertible; MLP
> up-projections are not square, and there the two families genuinely differ in
> reachable set, not merely in cost. The measurement uses a square layer for that
> reason.

## 7. Internal Mechanics

```mermaid {#fig:peft-map caption="The PEFT family arranged by where the trainable parameters act, not by how many there are. The horizontal axis is the one parameter tables show; the vertical axis is the one that decides whether a method can solve your problem at all (eq:peft-expressiveness)."}
flowchart TB
    subgraph ACT["acts on ACTIVATIONS — steers existing behaviour"]
        PT["prompt tuning<br/>(soft tokens)"]
        PX["prefix tuning<br/>(per-layer KV)"]
        BF["bias-only<br/>(BitFit)"]
    end
    subgraph MAP["acts on the MAP — can install new behaviour"]
        LO["LoRA<br/>parallel: W0 + BA"]
        AD["adapters<br/>serial: W0 (I + B'A')"]
        FT["full fine-tuning<br/>unconstrained"]
    end
    ACT -->|"cannot change<br/>what the layer computes"| MAP
    LO -->|"same reachable set<br/>eq:serial-parallel-equivalence"| AD
    AD -->|"but needs 306x the<br/>update norm at cond 1000"| LO
    MAP --> Q["+ quantised base<br/>= QLoRA: orthogonal saving"]
```

### 7.1 The families, priced

| Method | Trainable | Acts on | Inference cost | Notes |
|---|---|---|---|---|
| bias-only | $O(d)$ per layer | offset | none | astonishingly cheap; very limited |
| prompt tuning | $O(kd)$ total | activations | longer sequence | steers only |
| prefix tuning {{cite:li2021prefixtuning}} | $O(Lkd)$ | activations | longer KV | steers only |
| adapters {{cite:houlsby2019adapters}} | $O(Ldr)$ | map, serial | **extra layers** | conditioning penalty |
| LoRA {{cite:hu2021lora}} | $O(Ldr)$ | map, parallel | none if merged | the default |
| QLoRA {{cite:dettmers2023qlora}} | same | map, parallel | dequant overhead | when memory binds |

### 7.2 Selecting, in order

1. **What kind of change?** Format and tone are often reachable by steering.
   New capability is not. {{eq:peft-expressiveness}} decides this and nothing
   else does.
2. **Does the base fit in bf16 with adapters?** {{eq:training-memory}}. If yes,
   LoRA; the dequantisation overhead is not worth paying.
3. **If not, quantise the base** and accept that the adapter belongs to that
   quantisation.
4. **Check activations** against {{eq:activation-crossover}} *before* launching,
   with the real batch and sequence length.
5. **Do not budget time on the parameter ratio** — {{eq:compute-ratio}}.

### 7.3 What QLoRA added beyond "LoRA plus quantisation"

The composition is the idea, but three engineering pieces make it work, and all
three are {{part:15}}'s subject in detail: a normal-float format matched to
weight distributions rather than a uniform grid, quantising the quantisation
constants themselves, and paging optimiser state to host memory when a spike
would otherwise abort a long run.

**The third is the one that shows up in practice as "it ran for six hours then
OOMed on one long batch"**, and it is a memory-*variance* problem rather than a
memory-*mean* problem — which is why the arithmetic in
{{sec:9-practical-example}} can be right and the run still fail.

### 7.4 When activations bind, which they eventually do

{{eq:activation-memory-unchanged}} has no entry in the "removed by" column, so
when it dominates you are outside this chapter's toolkit and into a different set
of trades:

| Lever | Costs | Keeps |
|---|---|---|
| gradient checkpointing | ~30% more compute | batch, sequence, quality |
| shorter sequences | truncation ({{ch:ft-sft}}) | batch, compute |
| smaller batch + accumulation | more steps, worse GPU utilisation | effective batch |
| sequence parallelism | interconnect, complexity | everything, if you have the GPUs |

**The ordering matters and is usually got wrong.** Checkpointing is nearly free in
quality terms and is the first move; shortening sequences is the *last*, because
{{ch:ft-sft}} showed truncation is not a proportional loss of data but a training
signal that teaches the model where to stop. **A memory fix that silently changes
what the model learns is worse than a slower run**, and this is the specific way
that trade gets made without anyone noticing they made it.

## 8. Implementation

```python {tier=A name=training-memory-budget}
"""What LoRA actually saves, and what it does not.

The headline for cite:hu2021lora is a parameter count -- "0.01% of the
parameters are trainable" -- and it is true and misleading, because trainable
parameters are not the resource that binds.

Three different resources are at stake and they behave differently
(eq:training-memory). Optimiser state scales with TRAINABLE parameters, so LoRA
removes nearly all of it. Frozen weights scale with TOTAL parameters, so LoRA
removes none of it and cite:dettmers2023qlora attacks that instead. Activations
scale with batch and sequence length, and NEITHER method touches them.

Compute is the fourth, and the one people get wrong: the backward pass still runs
through every frozen layer, because the adapters need gradients that arrive from
above (eq:backward-still-full).
"""
BYTES = {"fp32": 4, "bf16": 2, "int4": 0.5}


def full_ft(P):
    """Mixed-precision Adam: bf16 weights, fp32 master, fp32 grads, m and v."""
    return {"weights": 2 * P, "master": 4 * P, "grads": 4 * P, "adam": 8 * P}


def lora(P, P_a):
    """Base frozen in bf16; only adapters get master weights, grads, Adam."""
    return {"weights": 2 * P, "master": 4 * P_a, "grads": 4 * P_a,
            "adam": 8 * P_a}


def qlora(P, P_a):
    """Base frozen at 4 bits (part:15 owns how); adapters unchanged."""
    return {"weights": 0.5 * P, "master": 4 * P_a, "grads": 4 * P_a,
            "adam": 8 * P_a}


def gb(x):
    return x / 1e9


MODELS = [("7B", 7e9), ("13B", 13e9), ("70B", 70e9)]
RANK_FRAC = 0.001          # adapters ~0.1% of parameters, a typical r=16 config

print("Memory for the MODEL STATE alone (activations excluded, they come next).\n")
print(f"{'model':>7}{'method':>9}" + "".join(f"{c:>10}" for c in
      ("weights", "master", "grads", "adam", "TOTAL")) + f"{'vs full':>10}")
print("-" * 76)

for name, P in MODELS:
    P_a = P * RANK_FRAC
    base = sum(full_ft(P).values())
    for label, d in (("full", full_ft(P)), ("LoRA", lora(P, P_a)),
                     ("QLoRA", qlora(P, P_a))):
        t = sum(d.values())
        print(f"{name:>7}{label:>9}"
              + "".join(f"{gb(d[c]):>10.1f}" for c in
                        ("weights", "master", "grads", "adam"))
              + f"{gb(t):>10.1f}{t / base:>10.2f}x")
    print()

print("\nActivation memory, which none of the above changes.\n")
print(f"{'model':>7}{'batch x seq':>14}{'no checkpoint':>16}"
      f"{'checkpointed':>15}")
print("-" * 54)
# Rough transformer activation model: ~ layers * tokens * hidden * 2 bytes * k
SHAPES = {"7B": (32, 4096), "13B": (40, 5120), "70B": (80, 8192)}
K = 34          # tensors kept per layer per token, order of magnitude
for name, P in MODELS:
    L, H = SHAPES[name]
    for tokens in (8 * 2048,):
        a = L * tokens * H * 2 * K
        ck = (L ** 0.5) * tokens * H * 2 * K
        print(f"{name:>7}{'8 x 2048':>14}{gb(a):>16.1f}{gb(ck):>15.1f}")

print("\n\nCompute per token, in units of P FLOPs.\n")
print(f"{'method':>10}{'forward':>10}{'grad wrt':>11}{'grad wrt':>11}"
      f"{'TOTAL':>9}{'saving':>9}")
print(f"{'':>10}{'':>10}{'inputs':>11}{'weights':>11}{'':>9}{'':>9}")
print("-" * 60)
print(f"{'full':>10}{2:>10}{2:>11}{2:>11}{6:>9}{'--':>9}")
print(f"{'LoRA':>10}{2:>10}{2:>11}{'~0':>11}{4:>9}{1 - 4/6:>9.0%}")
print(f"{'QLoRA':>10}{2:>10}{2:>11}{'~0':>11}{4:>9}{1 - 4/6:>9.0%}")

p7 = 7e9
tot_full = sum(full_ft(p7).values())
tot_lora = sum(lora(p7, p7 * RANK_FRAC).values())
tot_q = sum(qlora(p7, p7 * RANK_FRAC).values())
print(f"""
Take the 7B rows first. Full fine-tuning needs {gb(tot_full):.0f} GB of model
state before a single activation is stored, and look at where it goes: the
weights are {gb(2*p7):.0f} GB and everything the OPTIMISER needs is
{gb(16*p7):.0f} GB. Adam's two moments, the fp32 master copy and the gradients
are {16/18:.0%} of the bill, and the model itself is the small term
(eq:training-memory).

That is what LoRA removes. Optimiser state scales with TRAINABLE parameters, and
at {RANK_FRAC:.1%} of parameters it effectively vanishes: {gb(tot_lora):.0f} GB
total, {tot_lora/tot_full:.2f} of full fine-tuning. But notice what is left. The
frozen weights are untouched, because you still have to hold the model.

So LoRA converts a problem dominated by optimiser state into one dominated by
frozen weights -- and then cite:dettmers2023qlora attacks exactly that residual,
storing the frozen base at 4 bits instead of 16. {gb(tot_q):.0f} GB, or
{tot_q/tot_full:.2f} of full fine-tuning.

The two savings are independent and they compose, which is the point that gets
lost when QLoRA is described as "LoRA but smaller". They target different terms
of eq:training-memory: LoRA removes optimiser state, quantization removes weight
storage. Neither would be sufficient alone at 70B, and together they put it on
one 80 GB card.

Now the second table, which is where the arithmetic stops being encouraging.
Activation memory depends on batch size, sequence length, and depth -- not on how
many parameters are trainable. Nothing in this chapter reduces it. At 70B with a
modest batch it is larger than the quantized weights, so a QLoRA run that fits on
paper still fails at the first backward pass unless activations are checkpointed,
and checkpointing costs an extra forward pass.

This is why "QLoRA lets you fine-tune 70B on a 24 GB card" comes with a sequence
length in the fine print. The weights fit. Whether the run fits is a question
about activations.

The third table is the one that most often surprises people. Trainable parameters
drop by {1-RANK_FRAC:.1%}, and compute drops by 33%.

The reason is eq:backward-still-full. Gradients for the adapters in layer 3 arrive
from layer 40, which means the backward pass must propagate through every frozen
layer in between. What LoRA skips is only the weight-gradient computation --
roughly a third of the total -- and never the input-gradient chain.

So the honest summary is: LoRA and QLoRA are MEMORY techniques that make training
possible on hardware where it was impossible. They are not, and were never, speed
techniques. A run that took ten hours full fine-tuning takes about seven, not ten
minutes, and budgeting on the parameter ratio produces schedules that are wrong
by two orders of magnitude.""")
```

The memory table says what each method removes. The second listing asks the
question the parameter tables cannot: whether these methods are interchangeable at
matched budget.

```python {tier=A name=peft-expressiveness}
"""The PEFT family is not one axis. It is two, and only one is the parameter count.

Every parameter-efficient method answers "where do the trainable parameters live",
and the usual comparison is a table of parameter counts at matched quality. That
comparison hides the thing that actually decides between them: methods differ in
WHAT KIND OF CHANGE they can express, and a method that cannot express your
change does not improve with budget (eq:peft-expressiveness).

Three families, on one linear layer:

  LoRA (parallel)    y = x (W0 + BA)          -- adds to the MAP
  adapter (serial)   y = (x W0)(I + B'A')     -- transforms the OUTPUT
  bias / prompt      y = x W0 + b             -- adds an OFFSET

This listing puts them on tasks of different kinds at matched budget, then asks a
second question the parameter table cannot see: when the two low-rank families
CAN express the same change, does it cost them the same to do it?
"""
import numpy as np

rng = np.random.default_rng(157)
D, N = 64, 6000


def conditioned_W0(cond):
    """A base map with a chosen condition number."""
    U, _ = np.linalg.qr(rng.normal(size=(D, D)))
    V, _ = np.linalg.qr(rng.normal(size=(D, D)))
    s = np.logspace(0, -np.log10(cond), D)
    return U @ np.diag(s) @ V.T


def rel(P, Y):
    return float(np.linalg.norm(P - Y) / np.linalg.norm(Y))


def fit_lora(W0, X, Y, r, steps=900, lr=0.05):
    A = rng.normal(size=(D, r)) / np.sqrt(D); B = np.zeros((r, D))
    for _ in range(steps):
        G = 2.0 * (X @ (W0 + A @ B) - Y) / len(X); GD = X.T @ G
        gA, gB = GD @ B.T, A.T @ GD
        A -= lr * gA; B -= lr * gB
    return X @ (W0 + A @ B)


def fit_adapter(W0, X, Y, r, steps=900, lr=0.05):
    """Serial: the frozen layer runs first, then a low-rank correction."""
    H = X @ W0
    A = rng.normal(size=(D, r)) / np.sqrt(D); B = np.zeros((r, D))
    for _ in range(steps):
        G = 2.0 * (H + H @ (A @ B) - Y) / len(X); GD = H.T @ G
        gA, gB = GD @ B.T, A.T @ GD
        A -= lr * gA; B -= lr * gB
    return H + H @ (A @ B)


def fit_bias(W0, X, Y):
    """The whole family that acts on activations rather than weights: it can
    shift the output and nothing else."""
    return X @ W0 + (Y - X @ W0).mean(axis=0)


W0 = conditioned_W0(3.0)
X = rng.normal(size=(N, D))

Aq = rng.normal(size=(D, 4)) / np.sqrt(D); Bq = rng.normal(size=(4, D)) / 2
TASKS = {
    "map change (rank 4)": X @ (W0 + Aq @ Bq),
    "output offset":       X @ W0 + rng.normal(size=D) * 0.5,
    "both":                X @ (W0 + Aq @ Bq) + rng.normal(size=D) * 0.5,
}

print(f"D={D}. Matched budget: rank 4 for both low-rank methods "
      f"({2*D*4:,} params), bias has {D} params.\n")
print(f"{'task':>22}{'LoRA r=4':>11}{'adapter r=4':>14}{'bias only':>12}")
print("-" * 59)
for name, Y in TASKS.items():
    print(f"{name:>22}{rel(fit_lora(W0, X, Y, 4), Y):>11.4f}"
          f"{rel(fit_adapter(W0, X, Y, 4), Y):>14.4f}"
          f"{rel(fit_bias(W0, X, Y), Y):>12.4f}")

print("\n\nDoes raising the budget rescue a method that cannot express the "
      "change?\n")
print(f"{'task':>22}{'LoRA r=1':>11}{'r=4':>9}{'r=16':>9}{'r=32':>9}")
print("-" * 60)
for name in ("map change (rank 4)", "output offset"):
    Y = TASKS[name]
    print(f"{name:>22}" + "".join(f"{rel(fit_lora(W0, X, Y, r), Y):>{w}.4f}"
                                  for r, w in ((1, 11), (4, 9), (16, 9),
                                               (32, 9))))

print("\n\nWhen BOTH low-rank families can express the change, what does it "
      "cost each?\n")
print(f"{'cond(W0)':>10}{'LoRA update':>14}{'adapter update':>17}"
      f"{'ratio':>9}")
print(f"{'':>10}{'norm needed':>14}{'norm needed':>17}{'':>9}")
print("-" * 50)
ratios = {}
for cond in (1.0, 10.0, 100.0, 1000.0):
    Wc = conditioned_W0(cond)
    delta = Aq @ Bq                       # the same additive change every time
    serial = np.linalg.pinv(Wc) @ delta   # what a serial adapter must represent
    nl, na = np.linalg.norm(delta), np.linalg.norm(serial)
    ratios[cond] = na / nl
    print(f"{cond:>10.0f}{nl:>14.3f}{na:>17.3f}{na/nl:>9.1f}x")

print(f"""
The first table is the argument. At an identical budget the three methods are not
three points on a quality curve -- they are three different hypothesis classes,
and each is near-exact on the change it can express and useless on the one it
cannot.

LoRA and the serial adapter both handle a change to the MAP, and both fail on a
pure output offset, because x(W0+BA) is linear in x and cannot produce a constant
term. The bias-only method is the mirror image: it nails the offset with {D}
parameters and cannot touch the map at any budget.

The second table makes the failure mode unmistakable. On the map-change task,
LoRA improves with rank exactly as ch:ft-lora predicts. On the offset task it
does not improve at all -- rank 32 is no better than rank 1, because the missing
capability is not capacity. It is a term the parameterisation does not have
(eq:peft-expressiveness).

That is the practical lesson, and it inverts the usual selection procedure.
"Which PEFT method gives the best quality per parameter" is answerable only after
"what kind of change does my task need", and the second question is rarely asked.
Prompt and prefix tuning act on activations, so they steer a model through its
existing behaviour; LoRA and adapters act on the map, so they can install
behaviour that was not reachable. A task that needs the second will look like a
data problem or a capacity problem under the first, and no amount of tuning fixes
it.

Now the third table, which is the subtler result. When both low-rank families CAN
express the same change, they are not equivalent, and the difference is the base
layer's conditioning.

A serial adapter has to represent W0^-1 * delta rather than delta, so its required
update grows with the condition number: {ratios[1.0]:.1f}x at cond 1 rising to
{ratios[1000.0]:.1f}x at cond 1000. The change is the same change. The cost of
expressing it from the output side is not.

Two consequences follow, and both are measured elsewhere in this part. The larger
update is harder to optimise, which shows up as a serial adapter needing more
steps and a smaller learning rate on exactly the tasks where LoRA is
well-behaved. And by ch:ft-lora's eq:forgetting-quadratic, a larger update
disturbs the base model more -- so the serial form forgets more to achieve the
same adaptation, on ill-conditioned layers.

Real transformer weight matrices are not well conditioned. That, rather than the
latency argument usually given, is the deeper reason the parallel form won.""")
```

## 9. Practical Example

**The bill, itemised.** A 7B full fine-tune needs **126 GB** of model state:
weights **14 GB**, everything the optimiser needs **112 GB — 89%.** LoRA takes it
to **14.1 GB (0.11×)** by removing the optimiser terms; QLoRA takes it to **3.6 GB
(0.03×)** by attacking the residual. At 70B: **1,260 GB → 141 GB → 36 GB.**

**The two savings are orthogonal**, which is the point lost when QLoRA is
described as "LoRA but smaller". {{eq:training-memory}} has $P_a$ and $b_w$ as
independent factors of different terms, and {{eq:peft-memory-limit}} predicts
exactly the measured ratios.

**Activation memory is untouched.** At 70B, batch 8 × 2048 tokens: **730 GB
unchecked, 81.6 GB checkpointed — against 35 GB of quantised weights.** The
weights fit and the run does not. **This is why "70B on a 24 GB card" always comes
with a sequence length in the fine print.**

**And compute drops 33%, not 99.9%.** {{eq:backward-still-full}}: forward 2,
input-gradients 2, weight-gradients ≈ 0, so 4 against 6. A ten-hour run becomes
seven hours. **These are memory techniques.**

**PEFT methods are hypothesis classes.** At matched budget, LoRA and adapters
score **0.0000** on a map change and **0.5856** on an output offset; bias-only
scores **0.8328** and **0.0000**. Each is exact on what it can express and useless
otherwise.

**And budget does not rescue expressiveness**: LoRA on the offset task scores
**0.5856 at rank 1, 4, 16 and 32** — perfectly flat, because
{{eq:peft-expressiveness}} makes the floor a distance to a set that rank does not
approach. Contrast the map-change row, which goes **0.6813 → 0.0000** exactly as
{{ch:ft-lora}} predicts.

> **IMPORTANT:** This is the diagnostic that saves the most time in practice. **A
> flat response to budget means the wrong method; a falling response means the
> wrong setting.** They look identical in a loss curve and they have opposite
> fixes.

**Serial and parallel are equally expressive and unequally priced.** Both hit
0.0000 on the map task, as {{eq:serial-parallel-equivalence}} requires. But the
update norm a serial adapter must represent grows **1.0× → 4.5× → 32.3× → 306.3×**
as $\text{cond}(W_0)$ goes 1 → 1000, per {{eq:conditioning-penalty}}.

**Same change, 306× the movement** — harder to optimise, and by
{{eq:forgetting-quadratic}} quadratically more damaging to the base model. Real
weight matrices are badly conditioned, and **this is a better explanation of why
parallel adaptation won than the latency argument usually given.**

## 10. Production Considerations

**Compute {{eq:training-memory}} and {{eq:activation-memory-unchanged}} before
launching**, with the real batch and sequence length. Most failed PEFT runs fail
on the term nobody computed.

**Use bf16 base if it fits.** Quantisation costs dequantisation work and ties the
adapter to a specific quantised copy.

**Treat an adapter as bound to its base's exact quantisation.**
{{eq:lora-merge}}'s identity assumption does not survive a precision change.

**Enable gradient checkpointing by default** at scale, and budget the extra
forward pass.

**Do not schedule on the parameter ratio.** {{eq:compute-ratio}} caps the speedup
at 1.5×.

**Diagnose flat-versus-falling budget response** before adding data or steps.

**Prefer parallel to serial adaptation** unless you need per-task modules at
serving time, and know you are paying {{eq:conditioning-penalty}} if you do not.

**Watch memory variance, not just the mean** — the long-batch OOM six hours in is
a distributional problem.

## 11. Common Mistakes

**Quoting trainable parameters as the memory saving.** {{eq:training-memory}} has
four terms.

**Expecting a speedup proportional to the parameter cut.**
{{eq:backward-still-full}}.

**Forgetting activations exist.** They are frequently the binding term.

**Choosing a PEFT method by parameter efficiency** before asking what change is
needed.

**Raising rank against a flat budget curve** — that is an expressiveness problem.

**Merging a QLoRA adapter into an unquantised base.**

**Assuming higher rank costs meaningful memory.** {{eq:peft-memory-limit}} says it
does not; choose rank on capacity grounds.

**Using serial adapters by default** and paying {{eq:conditioning-penalty}}
invisibly.

## 12. Failure Modes

**OOM despite correct weight arithmetic.** Cause:
{{eq:activation-memory-unchanged}}. Fix: checkpointing, shorter sequences, smaller
batch.

**Intermittent OOM hours in.** Cause: memory variance on long batches. Fix: paged
optimiser state, length-bucketed batches ({{ch:ft-sft}}).

**Adapter produces nonsense on a "same" base.** Cause: different quantisation.

**Training far slower than the parameter ratio suggested.** Cause: expected;
{{eq:compute-ratio}}.

**Quality plateau immune to rank.** Cause: {{eq:peft-expressiveness}} — the method
cannot express the change.

**Serial adapter unstable where LoRA was fine.** Cause:
{{eq:conditioning-penalty}} on an ill-conditioned layer.

**Quantised base degrades a capability nobody tested.** Cause: quantisation error
is a perturbation of the frozen map; the adapter compensates *on the training
distribution* only.

## 13. Alternatives

| Alternative | Trades | When |
|---|---|---|
| full fine-tuning | memory, copies, forgetting | capability that PEFT cannot reach |
| LoRA in bf16 | memory | the base fits; the default |
| QLoRA | dequant overhead, base binding | memory is the binding constraint |
| prompt/prefix tuning | expressiveness | steering only; very small footprint |
| bias-only | expressiveness | offsets and calibration |
| gradient checkpointing | ~30% more compute | activations bind |
| ZeRO / sharding | interconnect, complexity | many GPUs available |
| smaller base model | capability | often the honest answer |

**The last row deserves more consideration than it gets.** A great deal of QLoRA
engineering exists to fine-tune a large model on inadequate hardware, and a
smaller model fine-tuned comfortably is frequently better than a large one
adapted through a straw — a comparison {{ch:ft-when}}'s cost model can actually
settle.

## 14. Evaluation

**Report the memory decomposition**, not the total — the terms tell a reader
whether the result transfers to their hardware.

**Report batch size and sequence length with any memory claim.**

**Report wall-clock alongside parameter counts**, since they diverge by
{{eq:compute-ratio}}.

**Evaluate a quantised-base adapter against the quantised base**, which is the
model that will run.

**Run the budget sweep** to distinguish expressiveness from capacity before
reporting a method as inadequate.

**Compare PEFT families at matched quality, not matched parameters** — matched
parameters compares hypothesis classes that were never comparable.

## 15. Advanced Concepts

**Quantisation error as a perturbation the adapter absorbs.**
{{maturity:MATURE}} {{eq:qlora-forward}} means the adapter is fitted to
$W_0^{q}$, so it partly *corrects* quantisation error on the training
distribution — and does not off it. That asymmetry is under-measured and is a
plausible source of the quiet capability loss people report.

**Rank is free in memory, expensive in forgetting.** {{maturity:MATURE}}
{{eq:peft-memory-limit}} removes the usual reason to keep rank small, leaving
{{ch:ft-lora}}'s {{eq:learns-less-forgets-less}} as the real constraint. **Most
rank sweeps are optimising the wrong objective.**

**Serial adapters remain right for one thing.** {{maturity:ESTABLISHED}}
Modules that stay unmerged compose per task at serving time, which is
{{part:23}}'s concern. {{eq:conditioning-penalty}} is the price.

**PEFT as an alignment-preservation tool.** {{maturity:EMERGING}}
{{cite:kirkpatrick2017ewc}} and {{cite:biderman2024loralearnsless}} together
suggest restricting movement is a *safety* property, not only an efficiency one —
a fine-tune that cannot move far cannot undo far. This is the ordering
{{ch:ft-lora}} argued for, extended.

**Compute-efficient PEFT is an open gap.** {{maturity:RESEARCH FRONTIER}}
{{eq:backward-still-full}}'s 1.5× ceiling holds for every method in this chapter,
because all of them need the full input-gradient chain. Methods that also shorten
that chain would be a genuinely different thing.

## 16. Connection to Previous Chapters

{{ch:ft-lora}} supplies the parameterisation and {{eq:forgetting-quadratic}},
which {{eq:conditioning-penalty}} multiplies into a squared penalty for serial
adapters. {{ch:ft-sft}}'s length bucketing is the fix for this chapter's memory
*variance* problem, which is a second reason to prefer it over packing.
{{ch:tf-complexity}}'s activation accounting is the term
{{eq:activation-memory-unchanged}} shows PEFT cannot touch.
{{ch:dl-optimizers}} explains the $8P_a$ that dominates {{eq:training-memory}}.
Forward: {{ch:q-theory}} and {{ch:q-memory-math}} own the quantisation this
chapter uses as a result; {{ch:ft-training-config}} measures the forgetting;
{{ch:ft-merging}} inherits {{eq:qlora-forward}}'s base-identity constraint.

## 17. Exercises

1. Compute {{eq:training-memory}} for a 34B model at rank 32 in bf16 and at 4
   bits. Which fits on 80 GB before activations?
2. Using {{eq:activation-crossover}}, find the batch × sequence budget at which
   activations overtake quantised weights for a 13B model.
3. Derive {{eq:compute-ratio}} and state the speedup when half the matmuls are
   trainable.
4. Prove {{eq:serial-parallel-equivalence}} and state where it fails for a
   non-square layer.
5. In `peft-expressiveness`, add a method $y = x(W_0 + BA) + b$. Which rows does
   it solve, and what does that suggest about combining families?
6. Raise the condition number in the same listing to $10^4$ and predict the ratio
   from {{eq:conditioning-penalty}} before running it.
7. Explain why a QLoRA adapter should not be merged into a bf16 base, in terms of
   {{eq:qlora-forward}}.
8. Design an experiment that separates quantisation-error compensation from
   genuine adaptation in a QLoRA run.

## 18. Interview Questions

1. Decompose fine-tuning memory. Which term does LoRA remove?
2. Why do LoRA and quantisation compose rather than compete?
3. Trainable parameters fall 99.9%. What happens to compute, and why?
4. Your QLoRA run OOMs even though the weights fit. What did you forget?
5. Quality is flat as you raise the rank. What are the two explanations and how do
   you tell them apart?
6. Are serial adapters and LoRA equally expressive? Equally cheap?
7. Why is a QLoRA adapter bound to a particular quantised base?
8. Does rank affect training memory much? Justify with the arithmetic.
9. When would you choose prefix tuning over LoRA?
10. When is a smaller base model the better answer than QLoRA?

## 19. Research Questions

1. {{eq:qlora-forward}} lets an adapter compensate for quantisation error on the
   training distribution. How large is the off-distribution gap, and does it
   explain reported capability loss?
2. {{eq:backward-still-full}} caps PEFT speedup at 1.5×. Is there a method that
   shortens the input-gradient chain without destroying adaptation quality?
3. {{eq:conditioning-penalty}} predicts serial adapters forget more on
   ill-conditioned layers. Does that hold in real transformers, per layer?
4. {{eq:peft-memory-limit}} makes rank nearly free in memory. If rank is chosen
   purely on {{eq:learns-less-forgets-less}}, how different are the resulting
   configurations from current practice?
5. Can expressiveness class be *diagnosed* from a task's data before training,
   rather than discovered from a flat budget sweep?

## 20. Chapter Summary

**Fine-tuning memory is four terms, not one** ({{eq:training-memory}}). For 7B:
**126 GB total, 14 GB weights, 112 GB — 89% — optimiser.** LoRA removes the terms
that scale with *trainable* parameters (**→ 14.1 GB, 0.11×**); quantisation
removes the term that scales with *total* parameters (**→ 3.6 GB, 0.03×**).

**They compose because they attack different factors**, which is why QLoRA is a
conjunction rather than an improvement, and why neither alone puts 70B on one
card.

**Two corrections the headline hides.** Activations are untouched —
**81.6 GB checkpointed at 70B against 35 GB of quantised weights** — so the
weights fitting does not mean the run fits. And compute falls **33%, not 99.9%**
({{eq:backward-still-full}}), capped at **1.5× by {{eq:compute-ratio}} whatever
you freeze**, because the input-gradient chain must reach every adapter. **These
are memory techniques, not speed techniques.**

**PEFT methods are hypothesis classes, not budget tiers.** At matched budget,
LoRA and adapters score **0.0000** on a map change and **0.5856** on an offset;
bias-only the reverse. **And budget does not rescue expressiveness** —
**0.5856 at rank 1 and at rank 32**, flat, because {{eq:peft-expressiveness}}
makes the floor a distance to a set. **A flat response to budget means the wrong
method; a falling one means the wrong setting**, and the fixes are opposite.

**Serial and parallel adaptation reach the same functions**
({{eq:serial-parallel-equivalence}}) **and do not pay the same price**: the serial
form needs **306× the update norm at $\text{cond}(W_0) = 1000$**
({{eq:conditioning-penalty}}), which by {{eq:forgetting-quadratic}} is a squared
penalty in forgetting. **Real weight matrices are badly conditioned — that is the
deeper reason parallel adaptation won.**

And the constraint to carry forward: **an adapter trained against a quantised base
belongs to that base.** {{eq:qlora-forward}} means it has partly learned to
correct that quantisation, so {{ch:ft-merging}}'s arithmetic inherits an identity
requirement stricter than it looks.

## 21. Further Reading

{{cite:dettmers2023qlora}} for the composition and for the three engineering
pieces — the normal-float format, double quantisation, and paged optimisers —
of which the third is the one that decides whether a long run survives.
{{cite:houlsby2019adapters}} for the serial form, read against
{{eq:conditioning-penalty}}: the paper's own ablations look different once you
know what the update norm is doing.
{{cite:li2021prefixtuning}} for the activation-side family, and as the clearest
example of a method whose limits are expressive rather than budgetary.
{{cite:hu2021lora}} for the parallel form this chapter prices.
{{cite:biderman2024loralearnsless}} and {{cite:kirkpatrick2017ewc}} together for
the argument that constrained movement is a safety property.
{{ch:q-theory}} before relying on a quantised base for anything that matters.
