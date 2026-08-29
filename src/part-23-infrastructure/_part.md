---
id: part-23-intro
status: draft
---

## What this part is for

{{part:22}} treated the model as a component with a price and a latency distribution.
This part opens it, and the first thing inside reorganises everything above it.

**A GPU is a machine for doing arithmetic, and decode barely does any.**

Generating one token requires reading every weight in the model and performing two
operations per weight. That is an arithmetic intensity of about **1 operation per
byte**, against a current datacentre GPU that needs **295** before its arithmetic units
become the constraint. At batch 1 the device runs at **0.3% of its peak arithmetic** —
over ninety-nine percent of the silicon you are paying for is idle, waiting for weights
to arrive.

Every technique in this part is a way of raising that ratio, or of arranging hardware so
the two phases of a request each get what they need.

> **The rule adopted for this part: every performance claim is a claim about a regime.**
> A speedup is a property of a batch size, an interconnect, a context length, and a
> baseline. The published numbers are honest and they are routinely quoted outside the
> conditions that produced them, so every chapter states the regime alongside the number.

## Where the numbers land

| what | number | chapter |
|---|---|---|
| Decode arithmetic intensity vs a datacentre GPU's balance point | **1.0** against **295** | {{ch:inf-cpu-gpu}} |
| Utilisation at batch 1, and at batch 256 | **$0.3\%$** to **$86.7\%$** | {{ch:inf-cpu-gpu}} |
| KV-cache crossover context, batch 1 vs batch 128 | **$24{,}704$** to **$193$** tokens | {{ch:inf-cpu-gpu}} |
| FlashAttention's HBM traffic reduction at 8192 tokens | **$65\times$**, for a flat **$4.7\times$** speedup | {{ch:inf-gpu-memory}} |
| Token-slot budget on an 80 GB device | **$476{,}837$**, one curve not two settings | {{ch:inf-gpu-memory}} |
| Static batching utilisation at batch 32 | **$13.4\%$** | {{ch:inf-batching}} |
| Free prefill chunk at batch 32 | **263 tokens**, then a cliff | {{ch:inf-batching}} |
| Tensor parallelism at 8 devices, fast link vs 25G | **$7.76\times$** vs **$0.79\times$** | {{ch:inf-parallelism}} |
| MoE experts touched, batch 1 vs batch 128 | **$3.1\%$** to **$98.3\%$** | {{ch:inf-parallelism}} |
| 16-way tensor group availability vs one device | **$16\times$** the downtime, **$256\times$** at two replicas | {{ch:inf-distributed}} |
| Kernel-launch share of a decode step | **$31.0\%$**, constant from batch 1 to 256 | {{ch:inf-serving-stacks}} |
| Cold start from a container image vs page cache | **486.5s** vs **93.2s** | {{ch:inf-kubernetes}} |
| Self-hosting break-even | **44,000 Mtok/month**, against a naive **15,124** | {{ch:inf-edge}} |

## The organising idea

**Every chapter finds that the quantity people optimise is not the quantity that binds.**

That is a different failure from {{part:22}}'s, where the instrument was silent. Here the
instrument is loud and correct and pointed at the wrong thing.

```text
   CHAPTER                  WHAT PEOPLE OPTIMISE      WHAT ACTUALLY BINDS
   ──────────────────────   ───────────────────────   ────────────────────────
   197 CPU and GPU          FLOP/s on the datasheet   bytes per token
   198 GPU memory           does the model fit        batch x context product
   199 batching             the batch size            how the batch is formed
   200 parallelism          how many devices          which dimension, what link
   201 distributed          routing efficiency        the failure domain
   202 serving stacks       the headline speedup      which inefficiency remains
   203 Kubernetes           the autoscaler            where the weights are stored
   204 cloud and edge       price per token           utilisation and bandwidth
```

Read that last column downward. Not one of those quantities appears on a default
dashboard, in a vendor datasheet, or in a standard configuration file. **Every one of
them has to be computed deliberately**, and each is one division away from data the
system already has.

## The three through-lines

**First: the batch is the axis everything turns on, and the things you want from it
conflict.**

Batching is what makes a GPU worth using for decode at all — **0.3%** to **86.7%** of
peak. But a larger batch moves the KV crossover *closer*
({{ch:inf-cpu-gpu}}), makes static batching *worse* ({{ch:inf-batching}}), erodes MoE
sparsity *faster* ({{ch:inf-parallelism}}), and consumes the token-slot budget that
context length also needs ({{ch:inf-gpu-memory}}). **Almost every technique in this part
either needs a large batch or is destroyed by one**, and the two chapters that escape —
graph capture and speculative decoding — are the ones worth noticing.

**Second: a constant divided by a constant is a constant.**

{{ch:inf-cpu-gpu}} showed decode step time is *flat* below the balance point, because the
weight read happens once regardless. That single fact produces three of this part's most
counterintuitive results: prefill chunks ride free until exactly 263 tokens
({{ch:inf-batching}}), launch overhead is **31%** of every step at every batch size
({{ch:inf-serving-stacks}}), and GPU utilisation reports **100%** at every load
({{ch:inf-kubernetes}}). Three different chapters, one mechanism.

**Third: the same product keeps appearing.**

| Where | The product | Consequence |
|---|---|---|
| {{ch:inf-cpu-gpu}} | coverage across pipeline stages | six stages compose to little |
| {{ch:inf-parallelism}} | expert coverage across a batch | sparsity erodes |
| {{ch:inf-distributed}} | device availability across a group | 16× the downtime |
| {{ch:sd-retrieval-agents}} | per-call reliability across a fan-out | tails amplify |

{{eq:loop-is-not-a-chain}} from {{ch:ag-loop}} appears for the fifth and sixth times in
this book, now as hardware. **The arithmetic does not care what the components are.**

## What this part does not settle

**The chunking-versus-disaggregation question is live.** {{cite:agrawal2023sarathi}} mixes
the phases; {{cite:zhong2024distserve}} and {{cite:patel2023splitwise}} separate them.
{{ch:inf-batching}} finds neither dominates and the crossover sits inside the range real
products occupy. A chapter presenting either as settled would be wrong.

**Correlation is assumed away throughout.** Device failures within a chassis, expert
routing within a batch, parallel-call latencies sharing a fabric — every model here
assumes independence, and each violation is noted and none is measured.

**The balance point is rising.** Successive hardware generations grow arithmetic faster
than bandwidth, so the batch required to saturate a device grows every generation. The
methods survive; the specific numbers have a shelf life, and the direction is known.

## How to read this part

{{ch:inf-cpu-gpu}} is load-bearing for all seven that follow and should not be skipped —
the balance point it defines is the pivot for the chunking cliff, the launch-overhead
constancy, and the autoscaling signal failure.

If you are diagnosing rather than reading through: {{ch:inf-serving-stacks}} contains the
cheapest correction in the part (achieved throughput against roofline *plus* the launch
term), and {{ch:inf-kubernetes}} contains the cheapest capacity fix (where the weights are
stored). Both are computable this afternoon from data you already have.
