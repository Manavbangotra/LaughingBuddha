---
id: q-gguf
number: 141
part: XV
tier: full
status: draft
requires: [q-int8-int4, q-theory, tf-complexity]
provides: [decode-is-bandwidth, arithmetic-intensity, memory-bound-crossover,
           dequantization-cost, bit-allocation, weight-only-rationale,
           format-unpacking-tradeoff]
citations: [dettmers2023case4bit, frantar2023gptq, lin2023awq, pope2022inference,
            egiazarian2024aqlm]
---

## 1. Learning Objectives

By the end of this chapter you will be able to derive tokens-per-second from a
model's size and a machine's memory bandwidth; compute the batch size at which
quantization stops helping, and explain why two people can honestly report
opposite results; explain why dequantization cost is free on a GPU and decisive on
a CPU; allocate bits across tensors by measured sensitivity rather than uniformly;
and say where that allocation stops working and why.

## 2. Why This Matters

The usual explanation for weight-only quantization is that a smaller model fits in
less memory. **That is true and it is the smaller half of the story.**

The larger half: **decoding one token reads every weight exactly once and does
almost no arithmetic with each one.** {{sec:9-practical-example}} computes the
arithmetic intensity at batch 1 and 4 bits as **4 FLOPs per byte**, against a
datacentre GPU that needs about **296** to keep its arithmetic units busy.

**Decode runs at roughly one per cent of the hardware's balance point.** The
arithmetic units are idle almost all the time and the only thing that matters is
the bus — so tokens per second is model bytes over memory bandwidth, and
bits-per-weight maps almost exactly onto speed.

Measured: a 7B model on a consumer GPU goes **71.4 → 285.7** tokens per second
from 16 bits to 4. **Four times fewer bits, four times the speed, with identical
arithmetic.**

**On a laptop that is the difference between existing and not**: **5.7** tokens
per second at 16 bits against **22.9** at 4. Quantization did not make local
inference faster — **it made a category of deployment exist.**

**Then the number worth memorising**: the batch size at which decode stops being
memory-bound. On a datacentre GPU at 4 bits it is around **74**. Below it,
halving the bits nearly halves the time; above it, halving the bits does nothing.

> **Which settles a persistent argument.** Someone at batch 1 reports 4-bit
> doubled their throughput. Someone at batch 128 reports it changed nothing.
> **Both measured correctly, on opposite sides of the crossover**, and neither
> result generalises.

**And the correction that decides format design.** Dequantization costs a few
operations per weight, and whether that matters depends entirely on the machine:
at six operations per weight the GPUs do not move at all, and the laptop falls
from **22.9 to 8.9** tokens per second — **61% of the speed the bandwidth
argument promised.**

{{maturity:ESTABLISHED}} Weight-only quantization, GGUF-style formats.
{{maturity:MATURE}} Roofline reasoning for decode.
{{maturity:EMERGING}} Bit allocation by measured sensitivity.

## 3. Prerequisites

{{ch:q-int8-int4}} for weight-only quantization and why it avoids the outlier
problem; {{ch:q-theory}} for {{eq:group-size-dominates}} and the $4^{-b}$ error
law this chapter's allocation rule depends on; {{ch:tf-complexity}} for FLOP
counting.

## 4. Intuitive Explanation

### Decode reads everything and computes almost nothing

A forward pass for one token performs about $2P$ FLOPs against $P$ parameters. So
per byte of weights read at $b$ bits:

$$ \text{intensity} = \frac{2P}{P\,b/8} = \frac{16}{b} \ \text{FLOPs per byte} $$

**At 4 bits that is 4.** Modern accelerators need hundreds. So decode is not
merely memory-bound — it is memory-bound by two orders of magnitude, which is why
the relationship between bits and speed is so nearly linear.

```text
   hardware                7B model:  16-bit   8-bit   4-bit   3-bit
   ─────────────────────              ──────   ─────   ─────   ─────
   laptop CPU, DDR5                      5.7    11.4    22.9    30.5
   Apple M-series, unified              28.6    57.1   114.3   152.4
   consumer GPU, GDDR6X                 71.4   142.9   285.7   381.0
   datacentre GPU, HBM3                239.3   478.6   957.1  1276.2
```

**Nothing about the model changed** across those columns — the same
multiply-accumulates against the same activations. Only how many bytes had to
cross the bus.

### The crossover, and why people disagree

Arithmetic intensity grows with batch, because the same weight read serves every
sequence in the batch:

$$ \text{intensity}(B) = \frac{16B}{b} $$

Setting that equal to the machine's FLOP-per-byte ratio gives the crossover:

```text
   hardware                FLOP/byte    16-bit   8-bit   4-bit
   ─────────────────────   ─────────    ──────   ─────   ─────
   laptop CPU, DDR5                6         6       3       2
   Apple M-series                 18        18       9       4
   consumer GPU                  160       160      80      40
   datacentre GPU                296       296     148      74
```

**Below the crossover, quantization is a speed technique. Above it, it is only a
memory technique.**

And note what quantizing does to the crossover itself: **fewer bits means a lower
crossover**, because the same arithmetic is spread over fewer bytes. **Quantizing
harder moves you closer to compute-bound, so the returns to quantizing shrink as
you quantize.**

### Dequantization is free on a GPU and decisive on a CPU

A quantized weight cannot be multiplied. It must be unpacked from its bit-packed
representation and multiplied by its group's scale first, and that costs
operations the 16-bit path does not pay.

```text
   extra ops/weight    laptop CPU   Apple M   consumer GPU   datacentre GPU
   ────────────────    ──────────   ───────   ────────────   ──────────────
                  0          22.9     114.3          285.7            957.1
                  6           8.9     114.3          285.7            957.1
                 20           3.2      45.5          285.7            957.1
                 60           1.2      16.1          285.7            957.1
```

**Both GPU columns are constant.** There is so much idle arithmetic at batch 1
that even sixty operations per weight is free.

**The laptop loses 61% of its speed at six operations** — an unpack, a shift, a
mask, a multiply, an add. That is not an exotic scheme; it is roughly what any
4-bit format costs.

> **So the cost of quantization falls almost entirely on the machines with the
> least compute — which are exactly the machines quantization exists to serve.**

That explains a set of design choices that otherwise look like conservatism.
Fixed group sizes, byte-aligned packing, scales stored adjacent to the weights
they scale, integer arithmetic wherever possible: on a machine with six FLOPs per
byte, **the unpacking step competes directly with the matmul for the same scarce
resource.**

**And it gives the honest form of the 4-bit argument.** Not that 3-bit is
inaccurate — {{ch:q-int8-int4}} showed methods that handle it. **A format's value
is its bits-per-weight minus its unpacking cost in the local FLOP-per-byte
currency**, and on the hardware where bits matter most, unpacking is most
expensive.

### Not every tensor deserves the same bits

"4-bit quantization" almost always means every tensor gets four bits. **That is a
convention, not a result.**

{{sec:9-practical-example}} measures per-layer sensitivity by quantizing one layer
at a time, and finds a spread of **84.8×** across six layers of one small network.
**And the driver is outlier content, not size or position** — the largest layer is
among the least sensitive.

Allocating bits to equalise the marginal damage gives **1.67×** at an average of
4 bits and **2.34×** at 5, for one forward pass per tensor at quantization time.

**The rule that ships does almost nothing.** "Extra bits on the first and last
tensors" gave **0.88665** against uniform's **0.87784** — because here the
sensitive layers are in the *middle*. **A heuristic that tracks a correlate works
exactly as long as the correlation holds, and fails silently otherwise.**

**And the allocation itself fails at 3 bits**, coming out **worse** than uniform —
because it is derived from the $4^{-b}$ error law that {{ch:q-theory}} said stops
holding below about 3 bits.

## 5. Formal Explanation

### 5.1 The decode roofline

$$ t_{\text{token}} = \max\!\left(\underbrace{\frac{P\,b/8 + M_{\text{kv}}}{\text{BW}}}_{\text{memory}},\ \underbrace{\frac{2PB + F_{\text{attn}}}{C}}_{\text{compute}}\right) $$ (eq:decode-roofline)

At $B = 1$ the first term dominates by two orders of magnitude, so

$$ \text{tokens/s} \;\approx\; \frac{\text{BW}}{P\,b/8} \;\propto\; \frac{1}{b} $$ (eq:decode-is-bandwidth)

**{{eq:decode-is-bandwidth}} is the chapter's central claim** and the measured
71.4 → 285.7 across a 4× bit reduction is it.

### 5.2 Arithmetic intensity and the crossover

$$ I(B, b) = \frac{2PB}{P\,b/8} = \frac{16B}{b} \quad\text{FLOPs per byte} $$ (eq:arithmetic-intensity)

The machine is balanced at $I^{*} = C/\text{BW}$, so decode becomes compute-bound
at

$$ B^{*} = \frac{b}{16}\,\frac{C}{\text{BW}} $$ (eq:memory-bound-crossover)

**{{eq:memory-bound-crossover}} is linear in $b$**, which is the
counterintuitive part: **quantizing lowers the crossover**. A 4-bit model on a
datacentre GPU is compute-bound above batch 74, where the 16-bit model would still
be memory-bound to batch 296.

### 5.3 Dequantization enters the compute term

$$ t_{\text{compute}} = \frac{2PB + d\,P}{C} $$ (eq:dequantization-cost)

for $d$ operations per weight. **The $dP$ term is independent of batch**, so it is
pure overhead at batch 1 and amortised at large batch — the opposite of the usual
intuition that overhead matters less when you are busy.

Quantization is a net win at batch 1 iff

$$ \frac{P b/8}{\text{BW}} \;>\; \frac{2P + dP}{C} \quad\Longleftrightarrow\quad \frac{b}{8} \,\frac{C}{\text{BW}} > 2 + d $$ (eq:dequant-viability)

**{{eq:dequant-viability}} is why the laptop row collapses.** At $C/\text{BW} = 6$
and $b = 4$: the left side is 3, so any $d > 1$ makes the format compute-bound.
On the datacentre GPU the left side is 148 and $d$ would have to exceed 146.

### 5.4 Bit allocation

With per-layer error $\varepsilon_i^2 \approx c_i\,4^{-b_i}$ and $n_i$ parameters,
minimise $\sum_i c_i 4^{-b_i}$ subject to $\sum_i n_i b_i = B_{\text{tot}}$.
The Lagrangian gives

$$ \frac{c_i\,4^{-b_i}\ln 4}{n_i} = \lambda \quad \forall i \quad\Longrightarrow\quad b_i = \tfrac{1}{2}\log_{4}\frac{c_i}{n_i} + \text{const} $$ (eq:bit-allocation)

**{{eq:bit-allocation}} equalises the marginal damage per bit spent.** A layer
whose errors matter 4× more gets exactly one more bit. Greedy assignment on the
ratio $c_i 4^{-b_i}/n_i$ is exact, because the objective is separable and convex.

> **IMPORTANT:** {{eq:bit-allocation}} inherits the $4^{-b}$ model from
> {{ch:q-theory}}'s {{eq:quantization-noise-variance}}, which requires many
> quantization levels. **Below about 3 bits the model fails and so does the
> allocation** — measured at an average of 3 bits, allocation was *worse* than
> uniform. The rule needs a floor.

### 5.5 Why weight-only, and what it gives up

Weight-only quantization dequantises to floating point and does the matmul in
floating point. So it:

- **avoids the activation outlier problem entirely** ({{ch:q-int8-int4}}),
- **gets the full bandwidth benefit** of {{eq:decode-is-bandwidth}},
- **pays $dP$** in {{eq:dequantization-cost}},
- and **never uses integer tensor cores**, so above $B^{*}$ it has no advantage at
  all over a 16-bit model.

$$ \text{weight-only is right} \iff B \ll B^{*} $$ (eq:weight-only-rationale)

**{{eq:weight-only-rationale}} is the entire local-versus-server split**, and it
is a statement about batch size rather than about model quality.

## 6. Mathematical Foundation

### 6.1 The intensity gap, worked

At $B=1$, $b=4$: $I = 4$ FLOPs/byte. A datacentre GPU with 990 TFLOP/s and
3.35 TB/s has $I^{*} = 296$.

$$ \frac{I}{I^{*}} = \frac{4}{296} = 1.4\% $$

**The arithmetic units are idle 98.6% of the time.** That is not an inefficiency
to be optimised away — it is a structural property of autoregressive decode, and
it is the resource that speculative decoding ({{ch:q-throughput-latency}}) spends.

### 6.2 The KV cache term, deferred

{{eq:decode-roofline}} has $M_{\text{kv}}$ in the memory term, and this chapter has
set it to zero. That is valid while $M_{\text{kv}} \ll P b/8$, i.e. while the
context is short relative to the model.

For a 7B model at 4 bits, $P b/8 = 3.5$ GB. A KV cache reaches that at roughly
tens of thousands of tokens across the batch — so **for single-user local
inference at moderate context the approximation holds**, and for a server it does
not. {{ch:q-activation-kv}} and {{ch:q-memory-math}} pick that up.

### 6.3 Where the allocation model breaks

{{eq:bit-allocation}}'s $c_i 4^{-b_i}$ requires $\varepsilon \ll$ signal. At
$b = 2$, {{ch:q-theory}}'s {{eq:effective-levels}} gives one or two usable levels,
the error is comparable to the value, and the $4^{-b}$ curve is no longer even
approximately right.

The measured failure is exactly there: at an average of 3 bits the allocation
pushed two layers to 2 bits to fund the outlier-laden ones, and the result was
**worse than uniform**.

> **MATH NOTE:** This is a general property of Lagrangian allocation and worth
> recognising outside this chapter. The rule optimises a *model* of the
> objective, so it will happily walk into the region where the model is wrong —
> and it does so preferentially, because that region looks cheap. **Any allocation
> derived from an error law needs a floor at the law's domain boundary**, not as a
> safety margin but as part of the derivation.

## 7. Internal Mechanics

```mermaid {#fig:decode-roofline caption="Decode at batch 1 sits far to the left of the roofline's knee: arithmetic intensity is about 4 FLOPs per byte against hardware that needs hundreds (eq:arithmetic-intensity). That is why bits map onto speed. Raising the batch moves right along the axis and crosses the knee at eq:memory-bound-crossover, after which quantization is memory-only. Dequantization cost adds a batch-independent term to the compute side (eq:dequantization-cost), which is free where compute is abundant and decisive where it is not."}
flowchart LR
    W["weights: P x b/8 bytes"] -->|"read once per token"| BUS["memory bandwidth"]
    BUS --> T["time per token"]
    A["arithmetic: 2 P B FLOPs"] --> C["compute"]
    DQ["dequantise: d x P ops<br/>batch-INDEPENDENT"] --> C
    C --> T
    T -->|"max of the two"| OUT["tokens/second"]
    B["batch size"] -->|"raises intensity<br/>16B/b"| KNEE{{"crossover<br/>eq:memory-bound-crossover"}}
    KNEE -->|"below: bits buy speed"| BUS
    KNEE -->|"above: bits buy only memory"| C
```

### 7.1 What a GGUF-style format is actually optimising

Reading the design choices through {{eq:dequant-viability}}:

| Choice | Why |
|---|---|
| fixed group size (32, 64, 256) | unpacking is a shift and mask, not a lookup |
| byte-aligned super-blocks | no cross-byte extraction on the hot path |
| scales adjacent to their weights | one cache line per group, not two streams |
| integer-only dequantisation paths | CPUs have far more integer than float throughput |
| per-tensor bit assignment (k-quants) | {{eq:bit-allocation}}, by heuristic |

**Every row is a $d$ reduction**, and on a machine with $C/\text{BW} = 6$ that is
where the performance is.

### 7.2 Choosing a quantization level, in order

1. **Which side of {{eq:memory-bound-crossover}} are you on?** If above, stop —
   quantize for memory only and pick the highest quality that fits.
2. **What is $d$ for your runtime and format?** {{eq:dequant-viability}} decides
   whether a smaller format is actually faster.
3. **Measure per-tensor sensitivity** and allocate ({{eq:bit-allocation}}), with a
   floor at 3 bits.
4. **Check the group size** — {{ch:q-theory}} showed it is worth more than a bit
   and costs a fraction of one.

### 7.3 Why the mixed formats have the names they do

A file labelled `Q4_K_M` is a bundle of three decisions this chapter has now
separated: a nominal 4-bit width, a k-quant super-block structure with a
particular group size, and a per-tensor allocation that gives some tensors more.
**The "4" is the least informative character in the name.**

The measured allocation in {{sec:9-practical-example}} — `[4, 6, 3, 3, 6, 4]` at
an average of four — is what those formats are approximating by rule. **They get
part of the way there**, and the gap between rule and measurement is the
**0.88665 against 0.52636** in that table.

### 7.4 Prefill is the other half, and it is a different machine

Everything above is about **decode**, and a request has two phases with opposite
characteristics.

**Prefill** processes the whole prompt at once, so every weight read serves as
many token-positions as the prompt is long. A 2,000-token prompt has an arithmetic
intensity of $16 \times 2000 / b$ — tens of thousands of FLOPs per byte, far past
any machine's balance point. **Prefill is compute-bound, always.**

That has three consequences worth holding separately from the rest of the chapter.
Quantization does not speed up prefill, and with a non-trivial $d$ it can slow it
down, because the dequantization work is now competing for a genuinely saturated
resource. Time-to-first-token is therefore governed by compute and prompt length,
while inter-token latency is governed by bandwidth and model size — **two
different bottlenecks in one request**. And a benchmark that reports only
"tokens per second" over a whole request is averaging two regimes, so its answer
depends on the prompt-to-completion ratio of whatever it happened to measure.

**Which is why the useful benchmark reports time-to-first-token and inter-token
latency separately**, and why a format that wins on one can lose on the other.
{{ch:q-throughput-latency}} takes this apart properly; the point here is only that
{{eq:decode-is-bandwidth}} is a claim about one phase.

## 8. Implementation

```python {tier=A name=decode-is-bandwidth}
"""Why 4-bit took over local inference, and it is not about memory.

The usual explanation for weight-only quantization is that a smaller model fits
in less memory. That is true and it is the smaller half of the story.

The larger half is that decoding one token reads EVERY weight exactly once and
does almost no arithmetic with each one. That makes decode memory-bound by a wide
margin, so time per token is essentially the model's size in bytes divided by the
memory bandwidth -- and bits-per-weight maps almost linearly onto tokens per
second (eq:decode-is-bandwidth).

This listing works the roofline for real hardware numbers, finds where the
linearity holds, and then finds the two places it breaks.
"""
import numpy as np

# Rough published figures, used as orders of magnitude rather than as claims
# about any particular part.
HW = [
    ("laptop CPU, DDR5",        0.08e12,   0.5e12),
    ("Apple M-series, unified", 0.40e12,   7.0e12),
    ("consumer GPU, GDDR6X",    1.00e12, 160.0e12),
    ("datacentre GPU, HBM3",    3.35e12, 990.0e12),
]

MODELS = [("7B", 7e9), ("13B", 13e9), ("70B", 70e9)]
BITS = (16, 8, 5, 4, 3)


def bytes_per_token(P, bits):
    return P * bits / 8.0


def flops_per_token(P, batch=1):
    return 2.0 * P * batch


def decode_tps(P, bits, bw, comp, batch=1):
    """Roofline: the slower of reading the weights and doing the arithmetic."""
    t_mem = bytes_per_token(P, bits) / bw
    t_cmp = flops_per_token(P, batch) / comp
    return batch / max(t_mem, t_cmp), t_mem, t_cmp


print("Tokens per second at batch 1, from the roofline alone.")
print()
print(f"{'hardware':>26}{'model':>7}" + "".join(f"{str(b) + '-bit':>10}"
                                                for b in BITS))
print("-" * 83)
tps = {}
for hwname, bw, comp in HW:
    for mname, P in MODELS:
        row = [decode_tps(P, b, bw, comp)[0] for b in BITS]
        tps[(hwname, mname)] = row
        print(f"{hwname:>26}{mname:>7}" + "".join(f"{v:>10.1f}" for v in row))
    print()

print("Arithmetic intensity: FLOPs performed per byte of weights read.")
print()
print(f"{'batch':>7}" + "".join(f"{str(b) + '-bit':>10}" for b in BITS)
      + f"{'':>4}{'hardware needs':>16}")
print("-" * 65)
for batch in (1, 4, 16, 64, 256):
    ai = [2.0 * batch / (b / 8.0) for b in BITS]
    need = HW[3][2] / HW[3][1]
    print(f"{batch:>7}" + "".join(f"{v:>10.1f}" for v in ai)
          + f"{'':>4}{need:>16.0f}")

print()
print()
print("Where does decode stop being memory-bound? Crossover batch size.")
print()
print(f"{'hardware':>26}{'FLOP/byte':>11}" + "".join(f"{str(b) + '-bit':>10}"
                                                     for b in BITS))
print("-" * 87)
cross = {}
for hwname, bw, comp in HW:
    ratio = comp / bw
    xs = [ratio * (b / 8.0) / 2.0 for b in BITS]
    cross[hwname] = xs
    print(f"{hwname:>26}{ratio:>11.0f}" + "".join(f"{v:>10.0f}" for v in xs))

print()
print()
print("The correction: dequantization is work. Whether it matters depends")
print("entirely on the hardware. 7B at 4 bits, batch 1, tokens per second.")
print()
print(f"{'extra ops per weight':>22}" + "".join(f"{n.split(',')[0]:>16}"
                                                for n, _, _ in HW))
print(f"{'to unpack and scale':>22}")
print("-" * 86)
P7 = 7e9
dq_rows = {}
for dq in (0, 2, 6, 20, 60):
    vals = []
    for _, bw, comp in HW:
        t_mem = bytes_per_token(P7, 4) / bw
        t_cmp = (flops_per_token(P7) + dq * P7) / comp
        vals.append(1.0 / max(t_mem, t_cmp))
    dq_rows[dq] = vals
    print(f"{dq:>22}" + "".join(f"{v:>16.1f}" for v in vals))

lap = tps[("laptop CPU, DDR5", "7B")]
gpu = tps[("consumer GPU, GDDR6X", "7B")]
dc70 = tps[("datacentre GPU, HBM3", "70B")]
print(f"""
Read the first table across a row and the scaling is almost exactly linear in
the reciprocal of the bit-width. A 7B model on a consumer GPU: {gpu[0]:.1f}
tokens per second at 16 bits, {gpu[3]:.1f} at 4. Four times fewer bits, almost
exactly four times the speed (eq:decode-is-bandwidth).

Nothing about the model changed. The arithmetic is identical -- the same number of
multiply-accumulates against the same activations. What changed is how many bytes
had to cross the memory bus to perform it, and at batch 1 that is the whole cost.

The laptop row is why this became a movement rather than an optimisation.
{lap[0]:.1f} tokens per second at 16 bits is not usable; {lap[3]:.1f} at 4 bits
is slow but real. Quantization did not make local inference faster -- it made a
category of deployment exist.

The second table says why the effect is so clean. Arithmetic intensity at batch 1
and 4 bits is {2.0 * 1 / 0.5:.0f} FLOPs per byte of weights read. The datacentre
GPU in the table needs about {HW[3][2]/HW[3][1]:.0f} FLOPs per byte to keep its
arithmetic units busy. Decode at batch 1 is running at roughly one per cent of
the hardware's balance point, which is another way of saying the arithmetic units
are idle almost all of the time and the only thing that matters is the bus.

The third table turns that into the number worth memorising: the batch size at
which decode stops being memory-bound. On a datacentre GPU at 4 bits it is around
{cross['datacentre GPU, HBM3'][3]:.0f}. Below that batch, halving the bits nearly
halves the time. Above it, halving the bits does nothing at all, because the
weights are no longer what you are waiting for.

That single number explains a persistent disagreement in practice. Someone
running a model locally at batch 1 reports that 4-bit doubled their throughput
against 8-bit. Someone serving the same model at batch 128 reports it changed
nothing. Both measured correctly. They are on opposite sides of the crossover, and
neither result generalises to the other's setting.

Notice also what the crossover column does with bit-width. Fewer bits means a
LOWER crossover, because the same arithmetic is spread over fewer bytes. Quantizing
harder does not only speed up decode -- it moves you closer to being compute-bound,
so the returns to quantizing shrink as you quantize.

The last table is the correction, and it does not say what the folklore says.

A quantized weight cannot be multiplied. It must first be unpacked from its
bit-packed representation and multiplied by its group's scale, and that costs a
few operations per weight -- operations the 16-bit path does not pay.

On the GPU rows that cost is invisible. Both GPU columns read
{dq_rows[0][2]:.1f} and {dq_rows[0][3]:.1f} at zero extra operations and exactly
the same at SIXTY -- an absurdly expensive unpacking scheme, and the number does
not move at all. There is so much idle arithmetic capacity at batch 1 that the
unpacking is genuinely free.

The Apple row is the intermediate case and it is the instructive one:
{dq_rows[0][1]:.1f} tokens per second at zero, unchanged at six, and
{dq_rows[20][1]:.1f} at twenty. It has enough compute headroom to absorb a cheap
unpacking and not enough to absorb an elaborate one, which is exactly the regime
where format design decisions become visible in benchmarks and arguments start.

On the CPU row it is decisive. The laptop goes from {dq_rows[0][0]:.1f} tokens per
second to {dq_rows[6][0]:.1f} at six operations per weight and {dq_rows[60][0]:.1f}
at sixty. Six operations -- an unpack, a shift, a mask, a multiply, an add -- is
not an exotic scheme; it is roughly what any 4-bit format costs. And it has
already taken away {1 - dq_rows[6][0]/dq_rows[0][0]:.0%} of the speed the
bandwidth argument promised.

That is the finding, and it explains a design decision that otherwise looks like
conservatism. The formats that dominate CPU inference -- fixed group sizes,
byte-aligned packing, scales stored adjacent to the weights they scale, integer
arithmetic wherever possible -- are not aesthetic choices. On a machine with six
FLOPs of compute per byte of bandwidth, the unpacking step is competing directly
with the matmul for the same scarce resource, and a format that is elegant on
paper can arrive slower than a cruder one.

It also explains why the same format can be the obvious choice on a laptop and a
pointless one on a datacentre GPU. The bandwidth argument for quantization holds
everywhere below the crossover batch. The dequantization COST of quantization
falls almost entirely on the machines with the least compute -- which are exactly
the machines quantization exists to serve.

So the honest form of the 4-bit argument is not that 3-bit is inaccurate.
ch:q-int8-int4 showed methods that handle 3 bits and below. It is that a format's
value is its bits-per-weight MINUS its unpacking cost measured in the local
FLOP-per-byte currency, and on the hardware where the bits matter most the
unpacking is most expensive. Four bits with trivial unpacking has held its
position against three bits with clever unpacking for that reason rather than for
an accuracy one.""")
```

The first listing says how many bits to use. The second says where to put them.

```python {tier=A name=bit-allocation}
"""Not every tensor deserves the same number of bits.

"4-bit quantization" almost always means every weight tensor gets four bits. That
is a convention, not a result. Layers differ in how much their errors matter --
by width, by position, by what multiplies them -- and a fixed budget spent
uniformly is a fixed budget spent badly.

The k-quant families in llama.cpp already act on this, giving more bits to some
tensors than others by rule of thumb. This listing does it by MEASUREMENT: derive
each tensor's sensitivity, allocate bits to equalise the marginal damage
(eq:bit-allocation), and compare against uniform at exactly the same average
bits.
"""
import numpy as np

rng = np.random.default_rng(263)

# Layers of deliberately different shapes and scales, as a real network has.
SHAPES = [(64, 256), (256, 256), (256, 512), (512, 256), (256, 256), (256, 64)]
# What actually differs between real weight tensors: outlier content and
# effective rank, not just size. KIND says which structure each layer has.
KIND = ["plain", "outliers", "plain", "lowrank", "outliers", "plain"]
N = 3000


def make_net():
    Ws = []
    for (a, b), kind in zip(SHAPES, KIND):
        W = rng.normal(size=(a, b)) / np.sqrt(a)
        if kind == "outliers":
            cols = rng.choice(b, size=max(1, b // 40), replace=False)
            W[:, cols] *= 20.0
        elif kind == "lowrank":
            r = max(4, min(a, b) // 16)
            U = rng.normal(size=(a, r)); V = rng.normal(size=(r, b))
            W = (U @ V) / np.sqrt(a * r)
        Ws.append(W)
    return Ws


def quantize(W, bits, group=64):
    qmax = 2 ** (bits - 1) - 1
    flat = W.reshape(-1)
    pad = (-flat.size) % group
    f = np.concatenate([flat, np.zeros(pad)]).reshape(-1, group)
    s = np.maximum(np.max(np.abs(f), axis=1, keepdims=True) / qmax, 1e-12)
    q = (np.clip(np.round(f / s), -qmax, qmax) * s).reshape(-1)
    return q[:flat.size].reshape(W.shape)


WS = make_net()
X = rng.normal(size=(N, SHAPES[0][0]))


def forward(Ws):
    h = X
    for i, W in enumerate(Ws):
        h = h @ W
        if i < len(Ws) - 1:
            h = np.tanh(h)
    return h


REF = forward(WS)


def err(Ws):
    return float(np.linalg.norm(forward(Ws) - REF) / np.linalg.norm(REF))


def with_bits(bits_per_layer):
    return [quantize(W, b) for W, b in zip(WS, bits_per_layer)]


SIZES = np.array([a * b for a, b in SHAPES], float)
B_REF = 6

# Sensitivity: quantize ONE layer at the reference width, see what it costs.
sens = []
for i in range(len(WS)):
    Ws = [W.copy() for W in WS]
    Ws[i] = quantize(Ws[i], B_REF)
    sens.append(err(Ws) ** 2)
sens = np.array(sens)

print(f"Per-layer sensitivity, measured by quantizing one layer at {B_REF} bits.")
print()
print(f"{'layer':>7}{'shape':>14}{'params':>10}{'structure':>11}{'error':>12}"
      f"{'relative':>11}")
print("-" * 65)
for i, ((a, b), k) in enumerate(zip(SHAPES, KIND)):
    print(f"{i:>7}{f'{a}x{b}':>14}{int(SIZES[i]):>10,}{k:>11}"
          f"{np.sqrt(sens[i]):>12.5f}{sens[i]/sens.min():>10.1f}x")


def allocate(avg_bits, lo=2, hi=8):
    """Greedy on the marginal return. Raising a layer from b to b+1 cuts its
    error contribution by three quarters and costs SIZES[i] bits, so spend each
    bit where that ratio is largest (eq:bit-allocation). Exact for a separable
    convex objective, which this is."""
    b = np.full(len(WS), float(lo))
    budget = avg_bits * SIZES.sum()
    spent = (b * SIZES).sum()
    while True:
        gain = np.where(b < hi, 0.75 * sens * 4.0 ** (-b) / SIZES, -1.0)
        i = int(np.argmax(gain))
        if gain[i] <= 0 or spent + SIZES[i] > budget:
            break
        b[i] += 1
        spent += SIZES[i]
    return b.astype(int)


def heuristic(avg_bits, lo=2, hi=8):
    """The common rule of thumb: spend extra on the first and last tensors,
    and take it back from the largest ones."""
    b = np.full(len(WS), float(avg_bits))
    b[0] = min(hi, b[0] + 2)
    b[-1] = min(hi, b[-1] + 2)
    budget = avg_bits * SIZES.sum()
    order = np.argsort(-SIZES)
    k = 0
    while (b * SIZES).sum() > budget and k < 10 * len(b):
        i = order[k % len(order)]
        if b[i] > lo:
            b[i] -= 1
        k += 1
    return b.astype(int)


print()
print()
print("Uniform against measured allocation, at exactly the same average bits.")
print()
print(f"{'avg bits':>9}{'uniform':>11}{'heuristic':>12}{'measured':>11}"
      f"{'gain':>8}{'   allocation'}")
print("-" * 78)

for avg in (3, 4, 5, 6):
    bu = np.full(len(WS), avg, int)
    bh = heuristic(avg)
    bm = allocate(avg)
    eu, eh, em = err(with_bits(bu)), err(with_bits(bh)), err(with_bits(bm))
    print(f"{avg:>9}{eu:>11.5f}{eh:>12.5f}{em:>11.5f}{eu/em:>7.2f}x"
          f"   {list(bm)}")

b4 = allocate(4)
e4u, e4m = err(with_bits(np.full(len(WS), 4, int))), err(with_bits(b4))
b3 = allocate(3)
e3u, e3m = err(with_bits(np.full(len(WS), 3, int))), err(with_bits(b3))
b5 = allocate(5)
e5u, e5m = err(with_bits(np.full(len(WS), 5, int))), err(with_bits(b5))
e4h = err(with_bits(heuristic(4)))
print(f"""
The sensitivity table is the first result and the spread is the point. The same
operation -- 6-bit quantization with groups of 64 -- applied to one layer at a
time produces errors differing by a factor of {sens.max()/sens.min():.0f} across
six layers of one small network.

And the reason is legible in the structure column. The two layers with outlier
weights are {sens.max()/sens.min():.0f}x and
{sorted(sens)[-2]/sens.min():.0f}x more sensitive than the least sensitive one.
The low-rank layer is barely above the plain ones. Size hardly matters: the
largest layer here is one of the least sensitive.

That is ch:q-theory's result in a new form. Sensitivity is driven by outlier
content, which sets how coarse the step must be, and not by how big the tensor is
or where it sits. Uniform bit allocation therefore spends the same budget on a
layer whose errors matter eighty times more and one whose errors barely register.

The allocation rule follows from the error model. Quantization error falls like
4^-b, so raising a layer by one bit cuts its contribution by three quarters and
costs its parameter count in storage. Spend each bit where that ratio is largest
and you get the optimum, exactly, because the objective is separable and convex
(eq:bit-allocation).

The comparison table has three things in it and one of them is a failure.

At an average of 4 bits, uniform gives {e4u:.5f} and measured allocation gives
{e4m:.5f} -- a factor of {e4u/e4m:.2f}. At 5 bits, {e5u/e5m:.2f}x. The 4-bit
allocation is {list(b4)}: six bits for each outlier-laden layer, three for the
large plain ones, four for the small plain ones. Not monotone in depth, not
monotone in size, and not something a rule would have produced.

The heuristic column is what actually ships in most quantized model files:
spend extra on the first and last tensors, take it back from the largest. At 4
bits it gives {e4h:.5f} against uniform's {e4u:.5f} -- essentially nothing.

That is worth dwelling on rather than skipping. The rule encodes a positional
correlation: first and last tensors are often smaller and often matter more. Here
the sensitive layers are in the MIDDLE, because sensitivity came from outlier
structure rather than from position, and the rule had no way to know. A heuristic
that tracks a correlate of the thing you care about works exactly as long as the
correlation holds, and fails silently when it does not.

Now the failure row, which is the most useful line in the table. At an average of
3 bits, allocation gives {e3m:.5f} against uniform's {e3u:.5f} -- it is
{e3u/e3m:.2f}x, meaning WORSE. The allocation was {list(b3)}: it took two layers
down to 2 bits to fund the outlier layers, and at 2 bits the error model that
justified the whole procedure has stopped being true.

ch:q-theory flagged this in advance: the 4^-b law comes from treating quantization
error as small uniform noise, and below about 3 bits the error is comparable to
the signal, not uniform, and correlated with the value. The allocation rule is
derived from a model, and it fails precisely where the model does -- which is a
better argument for knowing where a model applies than any amount of warning
about it.

So the practical shape. Bit allocation by measured sensitivity is worth roughly a
factor of two at average widths of four and above, it costs one forward pass per
tensor at quantization time, and it must be floored above the width where the
error model holds. It is a refinement on top of group size and outlier handling
rather than a substitute -- ch:q-theory measured outliers costing 16x at fixed
bit-width, and nothing here approaches that.

And it inherits ch:q-int8-int4's dependency: the sensitivity profile is measured
on data, so a profile from the wrong distribution allocates bits to the layers
that mattered for someone else's workload. The measurement is cheap; using the
right data for it is the part that requires attention.""")
```

## 9. Practical Example

**Decode is bandwidth.** Arithmetic intensity at batch 1 and 4 bits is **4 FLOPs
per byte** against a datacentre GPU's **296** — the arithmetic units are idle
**98.6%** of the time. So {{eq:decode-is-bandwidth}} holds tightly: a 7B model on
a consumer GPU runs **71.4** tokens per second at 16 bits and **285.7** at 4.
**Four times fewer bits, four times the speed, identical arithmetic.**

**On a laptop, 5.7 against 22.9** — the difference between unusable and slow but
real. **Quantization made a category of deployment exist.**

**The crossover decides whether any of that applies.**
{{eq:memory-bound-crossover}}: batch **74** on a datacentre GPU at 4 bits, **40**
on a consumer GPU, **2** on a laptop CPU. Below it, bits buy speed; above it, bits
buy only memory.

> **IMPORTANT:** This settles the recurring disagreement. Batch-1 users report
> 4-bit doubling throughput; batch-128 users report no change. **Both are correct
> and on opposite sides of {{eq:memory-bound-crossover}}.** And note the crossover
> is *linear in $b$*: **quantizing harder moves you closer to compute-bound**, so
> the returns shrink as you go.

**Dequantization cost is free on a GPU and decisive on a CPU.** At six extra
operations per weight, both GPU columns are **unchanged** — even at sixty — while
the laptop falls **22.9 → 8.9**, losing **61%**. The Apple row is the instructive
middle: unchanged at six, **114.3 → 45.5** at twenty.

**{{eq:dequant-viability}} explains it exactly**: at $C/\text{BW} = 6$ and 4 bits
the budget is $d < 1$; at $C/\text{BW} = 296$ it is $d < 146$. **The cost of
quantization falls on the machines quantization exists to serve**, which is why
GGUF-style formats are designed around cheap unpacking rather than around minimal
bits.

**Then bit allocation.** Per-layer sensitivity varies **84.8×** across six layers,
and **the driver is outlier content** — the two outlier-laden layers are 84.8× and
60.8× the least sensitive, while the largest layer is among the least sensitive
and the low-rank one is barely elevated.

**Measured allocation beats uniform by 1.67× at 4 bits and 2.34× at 5**, with the
4-bit allocation coming out `[4, 6, 3, 3, 6, 4]` — not monotone in depth, not
monotone in size.

**The rule that ships does almost nothing.** First-and-last-tensor bonuses gave
**0.88665** against uniform's **0.87784**, because here sensitivity came from
outlier structure rather than position. **A heuristic tracking a correlate works
until the correlation does.**

**And the allocation fails at 3 bits**, giving **1.14539** against uniform's
**1.01485** — it funded the outlier layers by pushing two layers to 2 bits, where
{{ch:q-theory}}'s $4^{-b}$ law no longer holds. **The rule walks into the region
where its own model is wrong, because that region looks cheap.**

## 10. Production Considerations

**Determine which side of {{eq:memory-bound-crossover}} you are on first.** It
decides whether this chapter applies at all.

**Measure $d$ for your runtime**, not for the format in the abstract. The same
file is fast in one implementation and slow in another.

**Benchmark on the target hardware class.** A format comparison on a GPU tells you
nothing about a laptop, per {{eq:dequant-viability}}.

**Report tokens per second with the batch size.** Without it the number is
uninterpretable.

**Allocate bits by measured sensitivity** where the tooling allows, with a floor
at 3 bits.

**Read format names as three decisions**, not one: width, group structure, and
per-tensor allocation.

**Do not neglect the KV cache term** once context grows —
{{ch:q-memory-math}} makes it explicit.

## 11. Common Mistakes

**Explaining weight-only quantization as a memory technique.** It is primarily a
bandwidth technique.

**Generalising a batch-1 benchmark to a serving deployment**, or the reverse.

**Assuming a smaller format is always faster.**
{{eq:dequant-viability}}.

**Comparing formats on the wrong hardware class.**

**Allocating bits uniformly** because the file format's name has one number in it.

**Pushing an allocation below 3 bits** because the arithmetic said to.

**Ignoring that quantizing lowers the crossover**, so the second halving is worth
less than the first.

## 12. Failure Modes

**A 3-bit model is slower than a 4-bit one.** Cause:
{{eq:dequant-viability}} — unpacking cost exceeded the bandwidth saving.

**Quantization gave no speedup on the server.** Cause: above
{{eq:memory-bound-crossover}}. Expected.

**Speed differs by 3× between two runtimes on the same file.** Cause: different
$d$. The format is not the implementation.

**Allocation made quality worse.** Cause: a layer fell below the error model's
domain.

**Local benchmark does not reproduce a published one.** Cause: different
bandwidth, different $d$, or a different batch size — usually all three.

**Long contexts erode the speedup.** Cause: $M_{\text{kv}}$ growing into the
memory term of {{eq:decode-roofline}}.

## 13. Alternatives

| Alternative | Trades | When |
|---|---|---|
| 16-bit weights | bandwidth | above the crossover; quality-critical |
| 8-bit weight-only | half the saving | when 4-bit quality is marginal |
| 4-bit + group scales | unpacking cost | local inference; the default |
| sub-3-bit ({{cite:egiazarian2024aqlm}}) | decode speed | memory-bound above all else |
| W8A8 integer | calibration | above the crossover, on integer hardware |
| a smaller model at 8 bits | capability | {{cite:dettmers2023case4bit}} settled this |
| speculative decoding | a draft model | spends the idle 98.6% instead |

**The last row is the most interesting alternative** and is not really an
alternative: it attacks the same idleness from the other direction, using the
spare arithmetic to verify several tokens at once rather than using less
bandwidth. {{ch:q-throughput-latency}} develops it, and **the two compose.**

## 14. Evaluation

**Report batch size with every throughput number.**

**Report the hardware's bandwidth and FLOP rate**, or the result does not
transfer.

**Report the runtime and version**, since $d$ is an implementation property.

**Report the group size and the per-tensor allocation**, not just the nominal
width.

**Report context length**, because $M_{\text{kv}}$ enters
{{eq:decode-roofline}}.

## 15. Advanced Concepts

**The idle-arithmetic budget is a resource.** {{maturity:MATURE}}
{{sec:6-mathematical-foundation}}'s 98.6% idleness is what speculative decoding,
Medusa-style multi-token heads, and batched verification all spend.
**Recognising decode's idleness as a budget rather than as waste reorganises the
whole optimisation space.**

**Formats are co-designed with kernels.** {{maturity:ESTABLISHED}}
{{eq:dequant-viability}} means a format cannot be evaluated apart from its
unpacking implementation. **A published bits-per-weight figure is half a
specification**, exactly as {{ch:q-theory}}'s group-size argument found for
quality.

**Sensitivity-driven allocation is under-tooled.** {{maturity:EMERGING}}
{{eq:bit-allocation}} is elementary and needs one forward pass per tensor, and
almost no quantization pipeline exposes it. The k-quant heuristics are a
positional proxy for it that {{sec:9-practical-example}} shows can capture
essentially nothing.

**Lagrangian allocation walks into its own model's failure region.**
{{maturity:MATURE}} A general lesson: an optimiser over a modelled objective
prefers the region where the model under-predicts cost. **The floor is part of the
derivation, not a safety margin.**

**Bandwidth is the scaling axis nobody plans for.**
{{maturity:RESEARCH FRONTIER}} Compute has grown far faster than memory
bandwidth, so $C/\text{BW}$ rises with every hardware generation — which raises
{{eq:memory-bound-crossover}} and makes decode *more* memory-bound over time.
**Quantization's value is increasing for structural reasons, independently of any
algorithmic progress.**

## 16. Connection to Previous Chapters

{{ch:q-int8-int4}} explains why weight-only avoids outliers, which is the quality
half of the argument this chapter completes with the speed half.
{{ch:q-theory}}'s $4^{-b}$ law is what {{eq:bit-allocation}} optimises, and its
domain boundary is what the allocation failure at 3 bits runs into. Its
group-size result and this chapter's format-design result are the same claim in
two currencies: **the number people quote is not the number that matters.**
{{ch:tf-complexity}} supplies the FLOP counts.
Forward: {{ch:q-activation-kv}} adds $M_{\text{kv}}$ to
{{eq:decode-roofline}}; {{ch:q-memory-math}} makes the whole budget explicit;
{{ch:q-runtimes}} is about who implements $d$ well; and
{{ch:q-throughput-latency}} spends the idleness this chapter measured.

## 17. Exercises

1. Derive {{eq:arithmetic-intensity}} and compute it for batch 8 at 4 bits.
2. Using {{eq:memory-bound-crossover}}, find the crossover batch for a machine
   with 2 TB/s and 400 TFLOP/s at 4 and 8 bits.
3. From {{eq:dequant-viability}}, find the largest $d$ that keeps a 4-bit format
   memory-bound at batch 1 on the Apple row.
4. Derive {{eq:bit-allocation}} from the Lagrangian and show that a 4× sensitivity
   ratio corresponds to exactly one bit.
5. In `decode-is-bandwidth`, add a KV-cache term for a 32k context and recompute
   the 7B rows. When does it dominate the weights?
6. In `bit-allocation`, raise the floor from 2 bits to 3 and re-run the average-3
   row. Does the failure disappear?
7. Explain why {{eq:memory-bound-crossover}} being linear in $b$ means the second
   halving of bit-width is worth less than the first.
8. For your own machine: measure bandwidth, compute the crossover, and check it
   against a benchmark at two batch sizes.

## 18. Interview Questions

1. Why does 4-bit quantization roughly quadruple decode speed?
2. What is the arithmetic intensity of decode at batch 1, and why does it matter?
3. At what batch size does quantization stop helping speed?
4. Two engineers disagree about whether 4-bit is faster. How do you resolve it?
5. Why is dequantization free on a GPU and not on a CPU?
6. Why is a 3-bit format sometimes slower than a 4-bit one?
7. Why do GGUF-style formats use fixed group sizes and byte alignment?
8. How would you decide bits per tensor?
9. Why does that allocation fail at very low average bit-widths?
10. Why is decode getting *more* memory-bound over hardware generations?

## 19. Research Questions

1. {{eq:dequant-viability}} makes format value hardware-dependent. Is there a
   format family that is Pareto-optimal across the whole $C/\text{BW}$ range, or
   is fragmentation structural?
2. {{eq:bit-allocation}} needs a domain floor. Is there an error model valid below
   3 bits that supports allocation, given codebook methods change the
   representation entirely?
3. How much of the k-quant heuristics' benefit is positional correlation, and how
   much would measurement add on real models?
4. Decode's 98.6% idleness is spent by speculation. What is the theoretical
   maximum recovery, and how close do current methods get?
5. If $C/\text{BW}$ keeps rising, what is the bit-width at which the unpacking
   cost and the bandwidth saving cross for a laptop-class machine in five years?

## 20. Chapter Summary

**Weight-only quantization is a bandwidth technique, not a memory one.** Decode
reads every weight once and performs **4 FLOPs per byte** at 4 bits, against
hardware needing **296** — so the arithmetic units are idle **98.6%** of the time
and {{eq:decode-is-bandwidth}} maps bits onto speed almost exactly: **71.4 →
285.7** tokens per second across a 4× bit reduction, with identical arithmetic.
**On a laptop, 5.7 → 22.9 — the difference between unusable and real.**

**The crossover decides whether that applies at all.**
{{eq:memory-bound-crossover}}: **batch 74** on a datacentre GPU at 4 bits, **40**
on a consumer GPU, **2** on a laptop. **Batch-1 and batch-128 reports of the same
change are both correct and neither transfers** — and because the crossover is
linear in $b$, **quantizing harder shrinks the returns to quantizing.**

**Dequantization cost is free where compute is abundant and decisive where it is
not.** Six operations per weight leaves both GPU columns **unchanged** and costs
the laptop **61%** of its speed ({{eq:dequant-viability}}). **The cost falls
almost entirely on the machines quantization exists to serve**, which is why
GGUF-style formats optimise unpacking rather than bits — and why the honest 4-bit
argument is about $d$, not about accuracy.

**And not every tensor deserves the same bits.** Sensitivity varied **84.8×**
across six layers, driven by **outlier content rather than size or position**.
Measured allocation beat uniform by **1.67×** at 4 bits and **2.34×** at 5, for
one forward pass per tensor. **The shipping heuristic — extra bits at the ends —
captured essentially nothing here (0.88665 against 0.87784)**, because the
sensitive layers were in the middle.

**And the allocation broke at 3 bits**, coming out worse than uniform, because
{{eq:bit-allocation}} optimises {{ch:q-theory}}'s $4^{-b}$ law and walks
preferentially into the region where that law fails. **A floor at the model's
domain boundary is part of the derivation.**

Which leaves the sentence this chapter shares with {{ch:q-theory}}: **the number
people quote is not the number that matters.** There it was group size against
bit-width; here it is unpacking cost and per-tensor allocation against the "4" in
a filename.

## 21. Further Reading

{{cite:dettmers2023case4bit}} for why 4 bits is the default, and read it alongside
{{eq:dequant-viability}}: the paper settles the accuracy question and this chapter
supplies the speed constraint that keeps 4 bits there.
{{cite:pope2022inference}} for the roofline framing, which
{{ch:q-throughput-latency}} develops properly — this chapter uses only its batch-1
corner.
{{cite:frantar2023gptq}} and {{cite:lin2023awq}} for what produces the weights
this chapter serves; both target exactly the weight-only configuration
{{eq:weight-only-rationale}} describes.
{{cite:egiazarian2024aqlm}} for sub-3-bit formats, read against
{{eq:dequant-viability}}: the memory win is real and the decode cost is the
binding constraint, which is the trade this chapter prices.
