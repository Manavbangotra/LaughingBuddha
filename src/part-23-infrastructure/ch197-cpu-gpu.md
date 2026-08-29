---
id: inf-cpu-gpu
number: 197
part: XXIII
tier: full
status: draft
requires: [three-properties-break-the-stack, variance-not-mean-drives-wait,
           streaming-capacity-is-set-by-ttft, access-shape-decides-the-store]
provides: [decode-is-bandwidth-bound, batch-is-the-mechanism-not-an-optimisation,
           kv-traffic-overtakes-weights, phases-want-different-hardware]
citations: [pope2022inference, kwon2023pagedattention, dao2022flash,
            patel2023splitwise]
---

## 1. Learning Objectives

By the end of this chapter you will be able to compute the arithmetic intensity of a
forward pass and place it against a device's balance point; explain why decode is
bandwidth-bound and prefill is compute-bound *in the same request on the same
hardware*, and what follows architecturally; state why batching is the mechanism that
makes a GPU worth using for decode rather than an optimisation applied afterwards;
compute the context length at which KV-cache traffic overtakes weight traffic, and
show why increasing the batch moves that crossover *closer*; and explain why CPU
inference is worse by a factor that depends on batch size rather than by the ratio of
the datasheet FLOP figures.

## 2. Why This Matters

{{part:22}} treated the model as a component with a price and a latency distribution.
This part opens the box, and the first thing inside is a fact that reorganises
everything above it.

**A GPU is a machine for doing arithmetic, and decode barely does any.** Generating
one token requires reading every weight in the model and performing two operations per
weight — an arithmetic intensity of about **1 operation per byte**, against a current
datacentre GPU that needs **295** before its arithmetic units become the constraint
({{eq:decode-is-bandwidth-bound}}).

{{sec:9-practical-example}} measures the consequence: at batch 1, that GPU runs decode
at **0.3% of its peak arithmetic**. Over ninety-nine percent of the silicon is idle,
waiting for weights to arrive.

Batching is what fixes it, and the word "optimisation" understates what it is doing.
Going from batch 1 to batch 256 lifts utilisation from **0.3%** to **86.7%** and
throughput from **239** to **61,257** tokens per second
({{eq:batch-is-the-mechanism-not-an-optimisation}}).

But the second listing finds the limit. KV-cache traffic scales with the batch while
weight traffic does not, so **every increase in batch size moves the crossover
closer** — at batch 1 the cache overtakes the weights at 24,704 tokens of context; at
batch 128, at **193** ({{eq:kv-traffic-overtakes-weights}}).

## 3. Prerequisites

You need {{ch:sd-architecture}}'s expense property
({{eq:three-properties-break-the-stack}}) — this chapter explains where the expense
physically comes from.

{{eq:variance-not-mean-drives-wait}} from {{ch:sd-async}} and
{{eq:streaming-capacity-is-set-by-ttft}} from the same chapter both reappear: the
service-time variance those chapters budgeted around originates in the mechanisms here.

{{eq:access-shape-decides-the-store}} from {{ch:sd-storage}} is the same
reads-per-byte reasoning one level down, and the parallel is worth holding in mind.

{{ch:tf-masking-kv}} supplies the KV cache; {{ch:q-memory-math}} supplies the memory
arithmetic. Neither is re-derived here.

## 4. Intuitive Explanation

Here is the thing that surprises people who arrive from ordinary systems engineering.

You buy a GPU because it does an enormous number of multiply-adds per second. The
datasheet says a current datacentre part does roughly a quadrillion of them. That
number is real. And for the dominant phase of language-model serving, it is almost
entirely irrelevant.

The reason is that arithmetic units are useless without operands, and operands come
from memory. So what actually matters is a ratio: **how many operations do you perform
for each byte you fetch?** Call it arithmetic intensity. Every device has a *balance
point* — the intensity at which its arithmetic units and its memory system are exactly
matched. Below it you are waiting on memory and the FLOP number is decoration; above
it you are doing arithmetic and the bandwidth number is decoration.

Now ask what generating one token involves. The token's representation flows through
every layer, multiplying against every weight matrix. That means reading the entire
model — fourteen gigabytes for a 7-billion-parameter model in 16-bit — and doing about
two operations per weight.

Two operations per two bytes. **An arithmetic intensity of one.** Against a balance
point of nearly three hundred.

This is not an implementation problem. There is no clever kernel that fixes it,
because the weights genuinely all participate and they genuinely each get used once.
You are running a machine built for arithmetic on a workload made almost entirely of
memory movement.

Which is where batching comes in, and it is worth being precise about what it does.
Batching does not make each request faster. It makes the *weight read shared*: if
sixty-four sequences each need their next token, one pass over the weights serves all
sixty-four. The bytes stay the same and the operations multiply by sixty-four, so the
intensity multiplies by sixty-four.

That is why batching is not a tuning knob applied at the end. **It is the mechanism by
which a GPU becomes worth using for decode at all.** A serving system that cannot
batch is running expensive hardware at a fraction of a percent of its capability.

Now the complication, and it is the one that shapes the rest of this part.

Attention does not read weights. It reads the *keys and values of every previous
token*, and that cache is per-sequence — sequence A's history is no use to sequence B.
So when you batch sixty-four sequences, you share one weight read and you perform
sixty-four separate cache reads.

At short context nobody notices, because the cache is tiny next to fourteen gigabytes
of weights. At long context the cache is the whole cost, and batching is amortising a
component that has stopped mattering. **The harder you batch, the sooner you reach
that point** — which is an unpleasant shape, because it means the two things you want
(large batches, long context) fight each other.

## 5. Formal Explanation

Let a model have $P$ parameters stored at $b$ bytes each. A forward pass over $n$
tokens performs approximately $2Pn$ floating-point operations — two per parameter per
token, counting a multiply and an add — and reads $Pb$ bytes of weights, **once,
regardless of $n$**.

Arithmetic intensity is operations per byte:

$$ I(n) \;=\; \frac{2Pn}{Pb} \;=\; \frac{2n}{b} $$ (eq:decode-is-bandwidth-bound)

which is **independent of model size** and linear in the number of tokens in the pass.
At $b = 2$ bytes, $I(n) = n$: one token gives intensity 1, and $n$ tokens give
intensity $n$.

Writing $n_p$ for prompt length and $m$ for decode batch size, the two phases of one
request therefore have intensities

$$ I_{\text{prefill}} = \frac{2n_p}{b}, \qquad I_{\text{decode}} = \frac{2m}{b}, \qquad \frac{I_{\text{prefill}}}{I_{\text{decode}}} = \frac{n_p}{m} $$ (eq:phases-want-different-hardware)

and since a prompt is typically hundreds of tokens while an achievable decode batch is
tens, that ratio is comfortably above one for realistic serving. **The two phases of a
single request differ in arithmetic intensity by an order of magnitude**, which is why
no single device configuration is right for both.

A device with peak throughput $F$ operations per second and memory bandwidth $B$ bytes
per second has balance point $I^\star = F/B$. The time for a pass is

$$ T(n) \;=\; \max\!\left(\frac{Pb}{B},\; \frac{2Pn}{F}\right) $$

— memory-bound when $I(n) < I^\star$ and compute-bound otherwise. Below the balance
point $T$ is *constant in $n$*: processing four tokens together takes the same wall
time as processing one, which is the entire economic argument for batching.

The realised fraction of peak arithmetic is

$$ U(n) \;=\; \frac{2Pn}{T(n)\,F} \;=\; \min\!\left(\frac{I(n)}{I^\star},\; 1\right) $$ (eq:batch-is-the-mechanism-not-an-optimisation)

so utilisation is simply the ratio of intensity to balance point, capped at one.

Now add the cache. Attending over $c$ prior tokens with $L$ layers, $h_{kv}$
key-value heads and head dimension $d_h$ reads

$$ K(c) \;=\; 2\,L\,c\,h_{kv}\,d_h\,b $$

bytes per sequence. For a batch of $m$ sequences the total traffic is
$Pb + mK(c)$ — weights once, cache $m$ times. The crossover context $c^\star$ at
which cache traffic equals weight traffic satisfies

$$ c^\star(m) \;=\; \frac{Pb}{m \cdot 2Lh_{kv}d_h b} \;=\; \frac{c^\star(1)}{m} $$ (eq:kv-traffic-overtakes-weights)

**The crossover is inversely proportional to batch size.** Doubling the batch halves
the context at which batching stops helping.

## 6. Mathematical Foundation

Batching efficiency — throughput at batch $m$ relative to $m$ times throughput at
batch 1 — follows directly:

$$ E(m, c) \;=\; \frac{m\,\bigl(Pb + K(c)\bigr)}{m\,Pb + m\,K(c)} \cdot \frac{1}{1} \;=\; \frac{Pb + K(c)}{Pb/m + K(c)} \cdot \frac{1}{m} \cdot m $$

which simplifies to

$$ E(m, c) \;=\; \frac{Pb + K(c)}{\dfrac{Pb}{m} + K(c)} \cdot \frac{1}{m} \cdot m \;=\; \frac{m\bigl(Pb + K(c)\bigr)}{Pb + mK(c)} \cdot \frac{1}{m} $$

Written cleanly, with $\rho = K(c)/Pb$ the cache-to-weight ratio:

$$ E(m, \rho) \;=\; \frac{1 + \rho}{1 + m\rho} $$

Two limits make this concrete. As $\rho \to 0$ — short context — $E \to 1$ and
batching is free. As $\rho \to \infty$ — long context — $E \to 1/m$, meaning batch
$m$ delivers the throughput of a single sequence and the other $m-1$ sequences'
traffic is pure overhead.

The half-efficiency point, $E = 1/2$, sits at $\rho = 1/(m-2)$, so for large $m$ the
usable context shrinks like $1/m$. This is the formal content of "the harder you
batch, the sooner batching stops helping," and it is why
{{eq:kv-traffic-overtakes-weights}} is an architectural constraint rather than a
tuning observation.

The intervention that moves it is $h_{kv}$. Since $K \propto h_{kv}$, reducing
key-value heads by a factor $g$ scales $\rho$ by $1/g$ and moves $c^\star$ by $g$.
**Grouped-query attention is a serving-throughput decision made at training time**,
and {{sec:9-practical-example}} measures it moving batch-32 efficiency at 32k context
from **5.4%** to **44.8%** at the multi-query limit.

## 7. Internal Mechanics

**Why prefill is different.** A prompt of $n_p$ tokens goes through the model together,
giving intensity $n_p$ by {{eq:decode-is-bandwidth-bound}} — a 900-token prompt has
intensity 900, above every balance point in {{sec:9-practical-example}}'s table. So
**prefill is compute-bound on the same hardware where decode is memory-bound**, in the
same request, seconds apart ({{eq:phases-want-different-hardware}}).
{{cite:pope2022inference}} formalises this split, and {{cite:patel2023splitwise}} acts
on it.

**Where FlashAttention fits.** {{cite:dao2022flash}} attacks a third traffic source:
the $n \times n$ attention score matrix, which during *prefill* would otherwise be
written to and read from high-bandwidth memory. It is a prefill optimisation more than
a decode one, because at decode the score vector is length $c$ rather than a matrix.
This is why FlashAttention's headline speedups are on training and long-prompt
workloads.

**What paging buys and what it cannot.** {{cite:kwon2023pagedattention}}'s paged cache
removes fragmentation, letting sequences of different lengths share memory without
reserving worst-case blocks, and continuous batching lets sequences join and leave a
running batch. Both raise *achievable* $m$ substantially. Neither makes $K(c)$ shared,
because different sequences genuinely have different histories — so both improve the
constant and leave {{eq:kv-traffic-overtakes-weights}}'s shape intact.

**Why CPU inference is not simply worse.** A server CPU has roughly 100× less compute
than a previous-generation datacentre GPU and roughly 5× less bandwidth. For a
bandwidth-bound workload the gap is the *bandwidth* ratio. {{sec:9-practical-example}}
finds batch-1 decode costing **4.3×** more on CPU rather than the **309×** the FLOP
datasheets imply — and the ratio widens with batch size, because batching moves the
work toward the axis where the CPU is weakest.

**Why the balance point is rising.** Successive hardware generations have grown peak
arithmetic faster than memory bandwidth -- in {{sec:9-practical-example}}'s table the
current datacentre part has 3.2x the compute of the previous one and 1.6x the
bandwidth, moving the balance point from 153 to 295. That trend has held for a decade
and there is no sign of it reversing, because adding arithmetic units is a matter of
area while adding bandwidth is a matter of pins and power.

The consequence for this chapter is uncomfortable: **the batch size required to
saturate a device grows with every generation**, so the same workload that used the
previous part well leaves the current one idle. A team that upgrades hardware without
raising batch size can measure a *regression* in cost per token, and
{{sec:12-failure-modes}} lists it for that reason.

**Quantisation as a bandwidth intervention.** Halving $b$ halves weight traffic and,
by {{eq:decode-is-bandwidth-bound}}, doubles intensity. In the memory-bound regime that
is a straight doubling of decode throughput, which is why quantisation pays more at
serving time than its compute-side arithmetic suggests. {{ch:q-memory-math}} has the
detail; the point here is *which* term it acts on.

## 8. Implementation

The first listing computes arithmetic intensity for both phases and places them
against real hardware balance points.

```python {tier=A name=cd1}
"""Decode does almost no arithmetic, which is why a GPU barely helps it.

A GPU is a machine for doing many multiply-adds per byte fetched from memory. Whether
it helps depends entirely on the ARITHMETIC INTENSITY of the work: operations
performed per byte of memory traffic.

Prefill has high intensity -- a whole prompt of tokens multiplies against each weight
matrix, so each weight byte is reused many times. Decode has almost none: one token
multiplies against every weight in the model, so each weight byte is used ONCE
(eq:decode-is-bandwidth-bound).

This listing computes the intensity of both phases and places them against real
hardware, and finds that the two phases of one request belong on opposite sides of
the machine's balance point.
"""
# A 7-billion-parameter model in 16-bit weights.
PARAMS = 7.0e9
BYTES_PER_PARAM = 2.0
WEIGHT_BYTES = PARAMS * BYTES_PER_PARAM

# (device, peak dense FLOP/s at bf16, memory bandwidth bytes/s, price per hour)
DEVICES = [
    ("server CPU, 64 core",   3.2e12,  4.10e11,  2.60),
    ("consumer GPU",          1.65e14, 1.01e12,  0.55),
    ("datacentre GPU, prev",  3.12e14, 2.04e12,  2.20),
    ("datacentre GPU, cur",   9.89e14, 3.35e12,  4.90),
]
BATCHES = [1, 4, 16, 64, 256]
PROMPT = 900


def flops_per_token(n_tokens):
    """Forward-pass FLOPs: about 2 multiply-adds per parameter per token."""
    return 2.0 * PARAMS * n_tokens


def intensity(n_tokens):
    """Arithmetic intensity: FLOPs per byte of weight traffic.

    The weights are read once per forward pass regardless of how many tokens
    are in flight, so intensity rises linearly with tokens processed together.
    """
    return flops_per_token(n_tokens) / WEIGHT_BYTES


print("A %.0fB-parameter model at %.0f bytes per weight: %.1f GB of weights to read"
      % (PARAMS / 1e9, BYTES_PER_PARAM, WEIGHT_BYTES / 1e9))
print("for every forward pass, no matter how many tokens are in it.")
print()
print("Hardware, and the arithmetic intensity each needs to reach peak.")
print()
print(f"{'device':>24}{'TFLOP/s':>11}{'GB/s':>9}{'balance point':>16}{'$/hr':>8}")
print("-" * 68)
balance = {}
for name, fl, bw, price in DEVICES:
    b = fl / bw
    balance[name] = b
    print(f"{name:>24}{fl / 1e12:>11.1f}{bw / 1e9:>9.0f}{b:>13.0f} F/B{price:>8.2f}")

print()
print("The balance point is FLOP/s divided by bytes/s: the arithmetic intensity")
print("at which a device stops being memory-bound and starts being compute-bound.")

print()
print()
print("Arithmetic intensity of each phase, by how many tokens go through together.")
print()
print(f"{'work':>34}{'tokens in pass':>17}{'intensity':>13}")
print("-" * 64)
work = [
    ("decode, batch 1", 1),
    ("decode, batch 4", 4),
    ("decode, batch 16", 16),
    ("decode, batch 64", 64),
    ("decode, batch 256", 256),
    ("prefill, one %d-token prompt" % PROMPT, PROMPT),
]
inten = {}
for label, n in work:
    i = intensity(n)
    inten[label] = i
    print(f"{label:>34}{n:>17}{i:>11.1f} F/B")

print()
print()
print("Placing those against each device. 'memory' means the device is waiting on")
print("memory and its arithmetic units are mostly idle.")
print()
print(f"{'work':>30}" + "".join(f"{d[0][:13]:>15}" for d in DEVICES))
print("-" * 90)
placement = {}
for label, n in work:
    i = intensity(n)
    cells = ""
    row = {}
    for name, fl, bw, price in DEVICES:
        bound = "memory" if i < balance[name] else "compute"
        row[name] = bound
        cells += f"{bound:>15}"
    placement[label] = row
    print(f"{label:>30}{cells}")

print()
print()
print("What that costs in achieved throughput. A memory-bound pass takes")
print("weight-bytes / bandwidth; a compute-bound one takes FLOPs / peak.")
print()


def pass_seconds(name, fl, bw, n_tokens):
    t_mem = WEIGHT_BYTES / bw
    t_flop = flops_per_token(n_tokens) / fl
    return max(t_mem, t_flop)


print(f"{'device':>24}" + "".join(f"{('b=%d' % b):>12}" for b in BATCHES))
print("-" * 84)
tok_s = {}
for name, fl, bw, price in DEVICES:
    row = []
    for b in BATCHES:
        t = pass_seconds(name, fl, bw, b)
        row.append(b / t)
    tok_s[name] = row
    print(f"{name:>24}" + "".join(f"{v:>12.0f}" for v in row))
print()
print("(decode tokens per second, all sequences together)")

print()
print()
print("The utilisation the same table implies: share of peak arithmetic actually")
print("used during decode.")
print()
print(f"{'device':>24}" + "".join(f"{('b=%d' % b):>12}" for b in BATCHES))
print("-" * 84)
util = {}
for name, fl, bw, price in DEVICES:
    row = []
    for b in BATCHES:
        t = pass_seconds(name, fl, bw, b)
        row.append(flops_per_token(b) / t / fl)
    util[name] = row
    print(f"{name:>24}" + "".join(f"{v:>11.1%}" for v in row))

print()
print()
print("And cost per million decoded tokens, which is what the choice comes down to.")
print()
print(f"{'device':>24}" + "".join(f"{('b=%d' % b):>12}" for b in BATCHES))
print("-" * 84)
cost = {}
for name, fl, bw, price in DEVICES:
    row = []
    for i, b in enumerate(BATCHES):
        per_sec = tok_s[name][i]
        row.append(price / 3600.0 / per_sec * 1e6)
    cost[name] = row
    print(f"{name:>24}" + "".join(f"{v:>12.2f}" for v in row))

print(f"""
The balance-point column is the number that explains this part. A current datacentre
GPU needs **{balance['datacentre GPU, cur']:.0f} operations for every byte** it reads
before its arithmetic units are the constraint. Below that it is waiting on memory,
and the FLOP/s figure on the datasheet is irrelevant.

Decode at batch 1 has an arithmetic intensity of **{inten['decode, batch 1']:.1f}**
(eq:decode-is-bandwidth-bound). That is not slightly below the balance point; it is
{balance['datacentre GPU, cur'] / inten['decode, batch 1']:.0f} times below it.

The reason is structural rather than an implementation defect. Generating one token
requires reading **every weight in the model** -- {WEIGHT_BYTES / 1e9:.0f} GB here --
and performing two operations per weight. One token, one pass over the weights, two
operations per byte read. There is no arrangement of the computation that changes
that, because the weights genuinely all participate.

The utilisation table is the same fact stated as waste. At batch 1 a current
datacentre GPU runs decode at **{util['datacentre GPU, cur'][0]:.1%} of its peak
arithmetic**. Over ninety-nine percent of the silicon you are paying for is idle,
waiting for weights to arrive.

Batching is the only lever that changes the ratio, and the table shows exactly how
much. Going from batch 1 to batch {BATCHES[-1]} raises intensity from
{inten['decode, batch 1']:.1f} to {inten['decode, batch 256']:.1f} and utilisation
from {util['datacentre GPU, cur'][0]:.1%} to
{util['datacentre GPU, cur'][-1]:.1%}, and throughput from
{tok_s['datacentre GPU, cur'][0]:.0f} to {tok_s['datacentre GPU, cur'][-1]:.0f}
tokens a second.

**The batch is not an optimisation. It is the mechanism by which a GPU becomes
worth using for decode at all.** ch:inf-batching takes up what it costs in latency.

Prefill sits on the other side. A single {PROMPT}-token prompt has intensity
{inten['prefill, one %d-token prompt' % PROMPT]:.1f}, which is above every device's
balance point in the table -- so prefill is **compute-bound on the same hardware
where decode is memory-bound**, in the same request, seconds apart.

That is the fact ch:inf-distributed is built on. The two phases want different
machines, and cite:patel2023splitwise measured what happens when you give them
different machines.

The cost table has the practical consequence, and it is not the one the FLOP/s
column suggests. The CPU has {DEVICES[2][1] / DEVICES[0][1]:.0f} times less compute
than a previous-generation datacentre GPU but only
{DEVICES[2][2] / DEVICES[0][2]:.0f} times less bandwidth -- so for batch-1 decode,
which is bandwidth-bound, the gap is the bandwidth ratio and not the FLOP ratio.

At batch 1 the CPU costs {cost['server CPU, 64 core'][0]:.2f} per million tokens
against {cost['datacentre GPU, cur'][0]:.2f} for a current datacentre GPU. At batch
{BATCHES[-1]} it costs {cost['server CPU, 64 core'][-1]:.2f} against
{cost['datacentre GPU, cur'][-1]:.2f}.

**CPU inference is not simply worse; it is worse by a factor that depends entirely on
batch size**, and at batch 1 the factor is
{cost['server CPU, 64 core'][0] / cost['datacentre GPU, cur'][0]:.1f}x rather than
the {DEVICES[3][1] / DEVICES[0][1]:.0f}x the datasheets imply. That is the honest
case for local and CPU deployment, and ch:inf-edge takes it up.""")
```

## 9. Practical Example

Four devices and the intensity each needs to reach peak:

```
                  device    TFLOP/s     GB/s   balance point    $/hr
--------------------------------------------------------------------
     server CPU, 64 core        3.2      410            8 F/B    2.60
            consumer GPU      165.0     1010          163 F/B    0.55
    datacentre GPU, prev      312.0     2040          153 F/B    2.20
     datacentre GPU, cur      989.0     3350          295 F/B    4.90
```

And the intensity of the work:

```
                              work   tokens in pass    intensity
----------------------------------------------------------------
                   decode, batch 1                1        1.0 F/B
                   decode, batch 4                4        4.0 F/B
                  decode, batch 16               16       16.0 F/B
                  decode, batch 64               64       64.0 F/B
                 decode, batch 256              256      256.0 F/B
     prefill, one 900-token prompt              900      900.0 F/B
```

Decode at batch 1 has intensity **1.0** against a balance point of **295** — not
slightly below, but **295×** below ({{eq:decode-is-bandwidth-bound}}). Prefill at 900
tokens is above every balance point in the table.

**The two phases of one request sit on opposite sides of the machine's ridge**
({{eq:phases-want-different-hardware}}).

The utilisation this implies:

```
                  device         b=1         b=4        b=16        b=64       b=256
------------------------------------------------------------------------------------
     server CPU, 64 core      12.8%      51.2%     100.0%     100.0%     100.0%
            consumer GPU       0.6%       2.4%       9.8%      39.2%     100.0%
    datacentre GPU, prev       0.7%       2.6%      10.5%      41.8%     100.0%
     datacentre GPU, cur       0.3%       1.4%       5.4%      21.7%      86.7%
```

At batch 1 the current datacentre GPU runs decode at **0.3%** of peak. Batching to 256
reaches **86.7%** ({{eq:batch-is-the-mechanism-not-an-optimisation}}).

Note the CPU column: it reaches full utilisation at batch 16, because its balance
point is only **8**. A weaker device is *easier* to saturate, which is the whole
reason the cost comparison is not the datasheet comparison:

```
                  device         b=1         b=4        b=16        b=64       b=256
------------------------------------------------------------------------------------
     server CPU, 64 core       24.66        6.17        3.16        3.16        3.16
            consumer GPU        2.12        0.53        0.13        0.03        0.01
    datacentre GPU, prev        4.19        1.05        0.26        0.07        0.03
     datacentre GPU, cur        5.69        1.42        0.36        0.09        0.02
```

At batch 1, CPU costs **24.66** per million tokens against **5.69** — a factor of
**4.3**, not the **309** the FLOP figures suggest. At batch 256 it is **3.16** against
**0.02**, a factor of 158. **CPU inference is worse by an amount that depends entirely
on batch size.**

The second listing looks inside the pass.

```python {tier=A name=cd2}
"""Where the time goes inside a forward pass, and why the answer changes with context.

The previous listing treated a forward pass as one lump of arithmetic against one
lump of weight traffic. Inside it, the work divides into operations whose cost scales
with parameters and operations whose cost scales with SEQUENCE LENGTH, and those two
groups behave completely differently as context grows.

Attention over the KV cache reads state proportional to the tokens already generated.
So at short context it is a rounding error and at long context it dominates, and the
crossover is a property of the model shape rather than the hardware
(eq:kv-traffic-overtakes-weights).

This listing finds the crossover, and shows why the standard optimisations stop
working past it.
"""
# A 7B-class decoder. Shapes chosen so parameter count lands near 7e9.
LAYERS = 32
D_MODEL = 4096
D_FF = 11008
N_HEADS = 32
N_KV_HEADS = 32          # multi-head; grouped-query changes this
HEAD_DIM = D_MODEL // N_HEADS
BYTES = 2.0

CONTEXTS = [128, 512, 2048, 8192, 32768, 131072]
BANDWIDTH = 3.35e12      # bytes/s, current datacentre GPU


def weight_bytes():
    """Bytes of weights read per forward pass, per layer times layers."""
    attn = 4.0 * D_MODEL * D_MODEL           # q, k, v, o projections
    ffn = 3.0 * D_MODEL * D_FF               # gated MLP: up, gate, down
    return LAYERS * (attn + ffn) * BYTES


def kv_bytes(ctx, kv_heads=N_KV_HEADS):
    """Bytes of KV cache read to attend over `ctx` prior tokens, one sequence."""
    per_token_per_layer = 2.0 * kv_heads * HEAD_DIM * BYTES   # K and V
    return LAYERS * ctx * per_token_per_layer


W = weight_bytes()
print("A %d-layer model, d_model %d, d_ff %d, %d heads." % (LAYERS, D_MODEL, D_FF,
                                                            N_HEADS))
print("Weights read per forward pass: %.2f GB" % (W / 1e9))
print("KV cache per token: %.2f MB" % (kv_bytes(1) / 1e6))
print()
print()
print("Memory traffic for one decode step, by context length. Batch 1.")
print()
print(f"{'context':>10}{'weight GB':>12}{'KV GB':>10}{'KV share':>11}"
      f"{'total GB':>11}{'step ms':>10}")
print("-" * 64)
tab = {}
for c in CONTEXTS:
    k = kv_bytes(c)
    tot = W + k
    tab[c] = (k, tot, k / tot, tot / BANDWIDTH * 1000.0)
    print(f"{c:>10}{W / 1e9:>12.2f}{k / 1e9:>10.2f}{k / tot:>11.1%}"
          f"{tot / 1e9:>11.2f}{tot / BANDWIDTH * 1000.0:>10.2f}")

print()
print()
print("The same at batch 32. Weights are read ONCE for the whole batch; KV cache")
print("is per sequence, so it scales with the batch and the weights do not.")
print()
B = 32
print(f"{'context':>10}{'weight GB':>12}{'KV GB':>10}{'KV share':>11}"
      f"{'total GB':>11}{'tokens/s':>11}")
print("-" * 65)
batched = {}
for c in CONTEXTS:
    k = kv_bytes(c) * B
    tot = W + k
    t = tot / BANDWIDTH
    batched[c] = (k, tot, k / tot, B / t)
    print(f"{c:>10}{W / 1e9:>12.2f}{k / 1e9:>10.2f}{k / tot:>11.1%}"
          f"{tot / 1e9:>11.2f}{B / t:>11.0f}")

print()
print()
print("Where the crossover sits: the context at which KV traffic equals weight")
print("traffic, by batch size.")
print()
print(f"{'batch':>8}{'crossover context':>20}{'KV GB there':>14}")
print("-" * 42)
cross = {}
for b in (1, 4, 16, 32, 64, 128):
    # W == b * kv_bytes(c)  ->  c = W / (b * kv_bytes(1))
    c = W / (b * kv_bytes(1))
    cross[b] = c
    print(f"{b:>8}{c:>20.0f}{W / 1e9:>14.2f}")

print()
print()
print("What batching buys, by context. This is the number that decides whether")
print("the previous listing's advice still applies.")
print()
print(f"{'context':>10}" + "".join(f"{('b=%d' % b):>12}" for b in (1, 8, 32, 128)))
print("-" * 58)
gain = {}
for c in CONTEXTS:
    row = []
    for b in (1, 8, 32, 128):
        tot = W + kv_bytes(c) * b
        row.append(b / (tot / BANDWIDTH))
    gain[c] = row
    print(f"{c:>10}" + "".join(f"{v:>12.0f}" for v in row))
print()
print("(decode tokens per second across the whole batch)")

print()
print()
print("Batching efficiency: throughput at batch b divided by b times throughput")
print("at batch 1. Perfect batching is 100%.")
print()
print(f"{'context':>10}" + "".join(f"{('b=%d' % b):>12}" for b in (8, 32, 128)))
print("-" * 46)
eff = {}
for c in CONTEXTS:
    base = 1.0 / ((W + kv_bytes(c)) / BANDWIDTH)
    row = []
    for i, b in enumerate((8, 32, 128)):
        tot = W + kv_bytes(c) * b
        row.append((b / (tot / BANDWIDTH)) / (b * base))
    eff[c] = row
    print(f"{c:>10}" + "".join(f"{v:>11.1%}" for v in row))

print()
print()
print("And what grouped-query attention does to it: fewer KV heads means less")
print("cache to read, which moves the crossover.")
print()
print(f"{'KV heads':>10}{'KV MB/token':>14}{'crossover at b=32':>20}"
      f"{'b=32 eff at 32k':>18}")
print("-" * 62)
gqa = {}
for kvh in (32, 8, 4, 1):
    per_tok = kv_bytes(1, kvh)
    c = W / (32 * per_tok)
    tot = W + kv_bytes(32768, kvh) * 32
    base = 1.0 / ((W + kv_bytes(32768, kvh)) / BANDWIDTH)
    e = (32 / (tot / BANDWIDTH)) / (32 * base)
    gqa[kvh] = (per_tok, c, e)
    print(f"{kvh:>10}{per_tok / 1e6:>14.3f}{c:>20.0f}{e:>18.1%}")

print(f"""
The first table is the shape of the problem. At {CONTEXTS[0]} tokens of context the
KV cache is {tab[CONTEXTS[0]][2]:.1%} of memory traffic and the weights are
everything. At {CONTEXTS[-1]} tokens the KV cache is {tab[CONTEXTS[-1]][2]:.1%} and
the weights are the rounding error (eq:kv-traffic-overtakes-weights).

The step time goes from {tab[CONTEXTS[0]][3]:.2f}ms to
{tab[CONTEXTS[-1]][3]:.2f}ms -- a factor of
{tab[CONTEXTS[-1]][3] / tab[CONTEXTS[0]][3]:.0f} -- for the same model on the same
hardware generating the same single token. **Context length is a latency parameter,
not merely a capability one.**

The batched table is where it turns into an architectural constraint. Weights are
read once per pass no matter how large the batch; **KV cache is read per sequence**,
so it scales with the batch. That means the thing batching amortises is exactly the
part that stops mattering as context grows.

The crossover table gives the boundary directly. At batch 1 the KV cache overtakes
the weights at {cross[1]:.0f} tokens of context. At batch {32} it overtakes at
{cross[32]:.0f} tokens. At batch {128}, {cross[128]:.0f} tokens.

**Every increase in batch size moves the crossover closer**, which is the opposite of
convenient: the harder you batch, the sooner batching stops helping.

The efficiency table prices that. At {CONTEXTS[0]} tokens of context, batch {8}
achieves {eff[CONTEXTS[0]][0]:.1%} of ideal batching -- close to free throughput, and
the regime ch:inf-cpu-gpu's first listing was implicitly describing, since it ignored
the cache entirely.

Hold the batch at {8} and grow the context: {eff[CONTEXTS[2]][0]:.1%} at
{CONTEXTS[2]} tokens, {eff[CONTEXTS[4]][0]:.1%} at {CONTEXTS[4]}. Push to batch
{128} at {CONTEXTS[4]} tokens and it is {eff[CONTEXTS[4]][2]:.1%} -- a hundred and
twenty-eight sequences of memory traffic buying
{gain[CONTEXTS[4]][3] / gain[CONTEXTS[4]][0]:.1f} times the throughput of one.

So the standard advice -- batch harder -- has a domain of validity, and the domain is
**short context**. Past the crossover, adding sequences to a batch adds proportional
memory traffic and buys almost nothing, because each sequence brings its own cache
and no work is shared.

This is the honest form of a claim the serving literature sometimes makes loosely.
cite:kwon2023pagedattention's paged cache and continuous batching raise achievable
batch size by removing fragmentation and letting sequences join and leave freely, and
that is a large real gain. What it cannot do is make KV traffic shared, because it
genuinely is not.

The last table is the intervention that does address it. Grouped-query attention
reduces the number of distinct key-value heads, and the cache shrinks in proportion:
{gqa[32][0] / 1e6:.3f} MB per token at {32} KV heads against
{gqa[4][0] / 1e6:.3f} MB at {4}. That moves the batch-32 crossover from
{gqa[32][1]:.0f} to {gqa[4][1]:.0f} tokens, and lifts batch-32 efficiency at
{32768} tokens of context from {gqa[32][2]:.1%} to {gqa[4][2]:.1%}.

**Grouped-query attention is a serving-throughput decision made at training time**,
and the table is why every model shipped for long context has adopted it. The
quality cost is small and paid once; the traffic saving is paid on every token of
every request forever.""")
```

Memory traffic for one decode step, batch 1:

```
   context   weight GB     KV GB   KV share   total GB   step ms
----------------------------------------------------------------
       128       12.95      0.07       0.5%      13.02      3.89
       512       12.95      0.27       2.0%      13.22      3.95
      2048       12.95      1.07       7.7%      14.03      4.19
      8192       12.95      4.29      24.9%      17.25      5.15
     32768       12.95     17.18      57.0%      30.13      8.99
    131072       12.95     68.72      84.1%      81.67     24.38
```

Step time rises **6×** for the same model on the same hardware generating the same
single token. **Context length is a latency parameter, not merely a capability one.**

Weights are read once per pass; cache is read per sequence. So the crossover moves
with batch:

```
   batch   crossover context   KV GB there
------------------------------------------
       1               24704         12.95
       4                6176         12.95
      16                1544         12.95
      32                 772         12.95
      64                 386         12.95
     128                 193         12.95
```

At batch 128 the cache overtakes the weights at **193 tokens** of context
({{eq:kv-traffic-overtakes-weights}}).

```mermaid {#fig:crossover caption="Weight traffic is paid once per pass and amortises across the batch; KV traffic is paid per sequence and does not. So the context at which batching stops helping moves inversely with batch size."}
flowchart TD
  A["forward pass traffic"] --> B["weights: P x b bytes<br/>read ONCE per pass"]
  A --> C["KV cache: K(c) bytes<br/>read PER SEQUENCE"]
  B --> D["amortises across batch"]
  C --> E["scales with batch"]
  D --> F["crossover c* = c*(1) / m"]
  E --> F
```

Batching efficiency makes the consequence plain:

```
   context         b=8        b=32       b=128
----------------------------------------------
       128      96.5%      86.2%      60.4%
       512      87.6%      61.4%      27.9%
      2048      65.1%      29.6%       9.3%
      8192      36.5%      11.5%       3.1%
     32768      20.0%       5.4%       1.4%
    131072      14.5%       3.7%       0.9%
```

At 32,768 tokens, batch 128 achieves **1.4%** of ideal batching — 128 sequences of
memory traffic buying **1.7×** the throughput of one.

Grouped-query attention is what moves it:

```
  KV heads   KV MB/token   crossover at b=32   b=32 eff at 32k
--------------------------------------------------------------
        32         0.524                 772              5.4%
         8         0.131                3088             11.5%
         4         0.066                6176             18.5%
         1         0.016               24704             44.8%
```

Four KV heads instead of 32 moves the batch-32 crossover from **772** to **6176**
tokens and lifts efficiency at 32k context from **5.4%** to **18.5%**. **A
serving-throughput decision made at training time**, and the reason every long-context
model has adopted it.

## 10. Production Considerations

Compute your workload's arithmetic intensity before choosing hardware. It is one
division and it determines whether the FLOP column on the quote matters at all.

Report GPU utilisation as fraction of *peak arithmetic*, not as the occupancy figure
the driver reports. A device can be 100% "busy" at 0.3% of peak, and the second number
is the one that says whether you are wasting money.

Measure your context distribution, not just its mean. By
{{eq:kv-traffic-overtakes-weights}} the batching regime you are in is determined by
context, and a bimodal distribution — short chats and long documents — is operating in
two regimes at once and should probably be two pools.

Choose the batch size against your actual context length, not against a benchmark run
at 512 tokens. The efficiency table falls by an order of magnitude across the range
that real deployments span.

Treat quantisation as a bandwidth intervention first and a memory-footprint one
second. In the memory-bound regime, halving weight bytes doubles decode throughput.

Prefer models with grouped-query or multi-query attention for long-context serving,
and note that this is not a fine-tuning decision — it is fixed at pre-training and
cannot be retrofitted.

Separate prefill and decode capacity in your capacity model even if they run on the
same machines. They scale with different things -- prefill with prompt tokens per
second, decode with concurrent sequences -- and a single "requests per second" figure
cannot express a fleet that is prefill-bound in the morning and decode-bound in the
evening. The two numbers are easy to collect and almost nobody collects them.

Watch for the batch size your scheduler actually achieves, which is usually well below
the one configured. Configured batch size is a ceiling; achieved batch depends on
arrival rate, and a system with a maximum batch of 64 serving eight concurrent users
is running at batch 8 with all the efficiency that implies. The gap between configured
and achieved is often the single largest correctable inefficiency in a deployment, and
it is invisible unless someone logs it.

Price CPU and consumer hardware honestly for your batch size. At batch 1 the gap is
small; the mistake is quoting either the datasheet ratio or the batch-256 ratio for a
workload that runs at neither.

## 11. Common Mistakes

**Choosing hardware on FLOP/s.** Irrelevant below the balance point, which is where
decode lives.

**Treating batching as an optimisation.** It is the mechanism that makes the hardware
worth using; without it you are at fractions of a percent of peak.

**Benchmarking at short context and deploying at long.** The batching efficiency
differs by an order of magnitude.

**Assuming paged attention makes KV traffic shared.** It removes fragmentation; the
histories are still genuinely distinct.

**Reading driver "GPU utilisation" as utilisation.** It measures whether kernels are
resident, not whether arithmetic is happening.

**Quoting the FLOP ratio for CPU inference.** The relevant ratio is bandwidth, and it
is twenty times smaller.

## 12. Failure Modes

**Silent regime change with context growth.** A retrieval change lengthens prompts,
the deployment crosses {{eq:kv-traffic-overtakes-weights}}'s boundary, and throughput
falls with no configuration change.

**Batch-size cargo culting.** A batch size tuned on one context distribution is
applied to another and delivers a fraction of its measured benefit.

**Memory exhaustion from batch times context.** KV cache scales with the product, so
raising both simultaneously exhausts memory at a combination neither raises alone.

**Prefill starvation.** A scheduler that prioritises decode leaves prompts queued,
inflating time-to-first-token — the quantity
{{eq:streaming-capacity-is-set-by-ttft}} showed sets streaming capacity.

**Cost model drift after a hardware upgrade.** A newer device with a higher balance
point can be *worse* per token at low batch, since intensity did not change and
bandwidth grew less than compute. The upgrade looks like a regression and the cause
is not in any code that changed.

**Achieved batch below configured batch.** The scheduler's maximum is not the
operating point; at low arrival rates the achieved batch collapses toward one and
the efficiency with it, while every configuration file still says 64.

## 13. Alternatives

**Speculative decoding.** Raises tokens-per-pass without changing outputs, which by
{{eq:decode-is-bandwidth-bound}} raises intensity directly. The cleanest attack on the
batch-1 problem, and the one that works when batching is unavailable because the
traffic is not there.

**Quantisation.** Halves $b$, halving weight traffic. Compounds with batching rather
than competing with it.

**Sparse models.** Reading only the active experts cuts effective $P$ per token;
{{ch:inf-parallelism}} takes this up.

**CPU or consumer hardware at low batch.** Defensible at **4.3×** rather than
**309×**, especially where the workload is genuinely single-stream.

**Accept low utilisation.** Sometimes correct: a low-traffic internal tool that runs
at batch 1 on a shared GPU is wasting a resource that had no better use, and the
engineering to fix it costs more than the waste.

## 14. Evaluation

Report tokens per second *and* fraction of peak arithmetic. The first alone cannot
distinguish a well-utilised small device from a badly-utilised large one.

Benchmark across the context distribution you actually serve, reporting the batching
efficiency curve rather than a single number.

Measure prefill and decode throughput separately. They are different workloads with
different bottlenecks and averaging them hides both.

Track cost per million tokens at your operating batch size, and recompute it when
either the context distribution or the traffic level moves.

Validate the intensity model against measurement once. If achieved throughput is far
below $B/(Pb)$, the bottleneck is elsewhere — kernel launch overhead, host transfers,
or scheduling — and the analysis in this chapter is not yet the binding one.

## 15. Advanced Concepts

The $2P$ operations-per-token estimate ignores attention's own arithmetic, which is
$O(c)$ per token per layer rather than $O(P)$. At short context that is negligible; at
128k context it is a meaningful additional term, and it is compute rather than
bandwidth — so very long context eventually re-enters the compute-bound regime from
the attention side even as the cache dominates traffic. The two effects have different
crossovers and the interaction is not well captured by either listing.

The balance-point model treats memory as a single tier. Real devices have a hierarchy —
registers, shared memory, L2, HBM — and {{cite:dao2022flash}}'s entire contribution is
exploiting the gap between two of those tiers. A more faithful roofline has multiple
ridges, and an operation can be bound at one level while having ample headroom at
another. {{ch:inf-gpu-memory}} takes this up properly.

The independence of intensity from model size in {{eq:decode-is-bandwidth-bound}} is
worth dwelling on, because it has a counterintuitive consequence. A 70B model and a 7B
model have the *same* arithmetic intensity at the same batch size -- both read their
weights once and do two operations per weight. The larger model is slower in absolute
terms, proportionally to its size, but it is not less efficient, and it saturates the
device at exactly the same batch. So the common intuition that large models are
"harder to serve efficiently" is wrong as stated: they are more expensive, at the same
efficiency. What actually differs is that a larger model leaves less memory for KV
cache, which caps achievable batch and therefore caps the intensity reachable in
practice. **The size penalty is a memory-capacity effect, not a bandwidth-efficiency
one**, and the two call for different responses.

There is a scheduling consequence the arithmetic here understates. Because $T(n)$ is
*constant* below the balance point, adding a sequence to an in-flight batch is
genuinely free in wall-clock terms — not cheap, free. That is what makes continuous
batching ({{cite:kwon2023pagedattention}}) so effective, and it means the marginal
admission decision in a serving system has a discontinuity: free up to the crossover,
proportional after it. Schedulers that model marginal cost as linear get the regime
before the crossover wrong.

## 16. Connection to Previous Chapters

{{eq:three-properties-break-the-stack}}'s expense property is explained here: a model
call is expensive because decode runs the hardware at a fraction of a percent of its
capability, and the price reflects the hardware rather than the work.

{{eq:variance-not-mean-drives-wait}} from {{ch:sd-async}} traces to
{{eq:kv-traffic-overtakes-weights}}: service time depends on context and batch
composition, both of which vary per request.

{{eq:streaming-capacity-is-set-by-ttft}} depends on prefill, which
{{eq:phases-want-different-hardware}} shows is a completely different workload from the
decode that follows it.

{{eq:access-shape-decides-the-store}} from {{ch:sd-storage}} is the same
operations-per-byte reasoning applied to storage rather than to silicon.

## 17. Exercises

1. Compute the balance point for a device with 400 TFLOP/s and 1.6 TB/s. What batch
   size saturates it at 16-bit weights?

2. Derive $E(m, \rho) = (1+\rho)/(1+m\rho)$ from the traffic model and find the $\rho$
   at which batch 64 achieves 50% efficiency.

3. A 70B model at 4-bit, 8 KV heads, 80 layers, head dim 128. Where is the batch-16
   crossover?

4. Modify the second listing to include attention's own FLOPs. At what context does
   decode become compute-bound again?

5. For your own deployment, compute achieved fraction of peak arithmetic. If it is
   below 10%, identify which of the five failure modes applies.

## 18. Interview Questions

1. Why does a GPU's FLOP/s figure barely matter for single-stream decode?

2. What is arithmetic intensity, and what is it for decode at batch 1?

3. Why does increasing batch size make long-context serving *worse* relative to ideal?

4. Prefill and decode run on the same hardware seconds apart. Why might you want
   different hardware for each?

5. Our CPU inference is 4× more expensive than GPU, but the GPU has 300× the FLOPs.
   Explain.

6. We upgraded to newer GPUs and cost per token went up. What would you check first,
   and what does the answer imply about the batch size we are running?

## 19. Research Questions

1. What is the right multi-tier roofline for transformer decode, and does the
   single-ridge model mislead in practice?

2. Can KV cache be made partially shared across sequences — through prefix sharing,
   compression, or approximation — enough to change
   {{eq:kv-traffic-overtakes-weights}}'s shape rather than its constant?

3. Where is the quality-throughput frontier for KV-head count, measured rather than
   assumed, at current model scales?

4. Does the free-below-crossover discontinuity in marginal admission cost admit a
   provably better scheduling policy than continuous batching's greedy admission?

## 20. Chapter Summary

Arithmetic intensity — operations per byte — decides whether a device's compute or its
memory is the constraint. For a forward pass over $n$ tokens at $b$ bytes per weight,
intensity is $2n/b$, **independent of model size**
({{eq:decode-is-bandwidth-bound}}).

Decode at batch 1 has intensity **1.0** against a current datacentre GPU's balance
point of **295**, so the device runs at **0.3%** of peak arithmetic. Batching to 256
reaches **86.7%** and lifts throughput from **239** to **61,257** tokens per second:
**batching is the mechanism that makes the hardware worth using**, not an optimisation
({{eq:batch-is-the-mechanism-not-an-optimisation}}).

Prefill at 900 tokens has intensity 900 and is compute-bound on the same hardware, in
the same request ({{eq:phases-want-different-hardware}}).

Weight traffic is paid once per pass; KV traffic is paid per sequence. So the
crossover context falls as $c^\star(1)/m$ — **24,704** tokens at batch 1, **193** at
batch 128 ({{eq:kv-traffic-overtakes-weights}}) — and at 32k context batch 128
achieves **1.4%** of ideal batching. Grouped-query attention with four KV heads moves
the batch-32 crossover from **772** to **6176** tokens.

And the honest comparison: CPU decode at batch 1 costs **4.3×** a datacentre GPU, not
the **309×** the FLOP datasheets imply, because the binding ratio is bandwidth.

Two of this chapter's results are worth holding separately from their numbers,
because the numbers will age and the structure will not. The first is that a device's
usefulness is decided by a ratio rather than by either of its headline figures, so a
procurement conversation conducted in FLOPs is conducted in the wrong unit. The
second is that the two phases of a single request are different workloads that happen
to run consecutively, and every remaining chapter in this part is in some sense a
response to that fact.

Carry forward: **compute the intensity before choosing the hardware**, and **the batch
size and the context length are one decision, not two**.

## 21. Further Reading

- {{cite:pope2022inference}} — the prefill/decode split and its arithmetic, stated
  properly.
- {{cite:kwon2023pagedattention}} — paged KV cache and continuous batching; what raises
  achievable batch size and what it cannot do.
- {{cite:dao2022flash}} — IO-awareness within the memory hierarchy, and why it is
  mostly a prefill result.
- {{cite:patel2023splitwise}} — acting on {{eq:phases-want-different-hardware}} at the
  fleet level.
