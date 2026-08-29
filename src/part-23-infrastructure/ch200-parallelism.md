---
id: inf-parallelism
number: 200
part: XXIII
tier: full
status: draft
requires: [decode-is-bandwidth-bound, batch-is-the-mechanism-not-an-optimisation,
           roofline-has-multiple-ridges, batch-times-context-is-the-budget]
provides: [parallelism-dimension-is-an-interconnect-decision,
           tensor-parallelism-is-in-node, sparsity-erodes-with-batch-size,
           training-economics-are-not-serving-economics]
citations: [shoeybi2019megatron, rajbhandari2020zero, fedus2021switch,
            kwon2023pagedattention]
---

## 1. Learning Objectives

By the end of this chapter you will be able to distinguish the four parallelism
dimensions by what each one buys and what each one costs in communication; compute the
per-step communication volume for tensor and pipeline parallelism and show why the
first is an in-node technique and the second is not; calculate the interconnect
bandwidth a given tensor-parallel split requires, and explain why the requirement grows
roughly with the square of the split; state why data parallelism cannot make a single
request faster; and show why a mixture-of-experts model's sparsity erodes with batch
size until the model costs what its dense equivalent costs.

## 2. Why This Matters

"Add more GPUs" sounds like one action. It is four, with four different communication
costs, and three of them do different things.

{{sec:9-practical-example}} measures the per-step communication for a 70B model at 8
devices: tensor parallelism moves **146.8 MB** per device, pipeline parallelism moves
**3.7 MB**, data parallelism moves nothing
({{eq:parallelism-dimension-is-an-interconnect-decision}}). That factor of **40** comes
from a structural difference — tensor parallelism synchronises twice per layer,
pipeline once per stage boundary, eighty layers against seven.

The consequence is decisive at deployment time. On a fast in-node link, tensor
parallelism at 8 devices gives **7.76×**; on 25G ethernet the identical configuration
gives **0.79× — slower than one device**
({{eq:tensor-parallelism-is-in-node}}), while pipeline still gives **5.35×**.

The second half concerns sparsity, and finds an uncomfortable interaction with
everything this part has argued for. A 600B mixture-of-experts model reads its active
size at batch 1 and **98.3% of its experts at batch 128**
({{eq:sparsity-erodes-with-batch-size}}) — converging to the dense model it
structurally is, at exactly the batch sizes {{ch:inf-cpu-gpu}} showed are necessary.

## 3. Prerequisites

You need {{eq:decode-is-bandwidth-bound}} from {{ch:inf-cpu-gpu}} — every speedup in
this chapter is a reduction in weight bytes read per device, and the communication is
what you pay for it.

{{eq:batch-is-the-mechanism-not-an-optimisation}} is what makes the sparsity result
bite: sparsity and batching compete for the same resource.

{{eq:roofline-has-multiple-ridges}} from {{ch:inf-gpu-memory}} supplies the
interconnect tiers, and the PCIe and network rows are the ones that matter here.

{{eq:batch-times-context-is-the-budget}} is why splitting weights across devices helps
more than it appears: freeing weight memory frees cache capacity.

## 4. Intuitive Explanation

There are exactly four things you can split, and it is worth naming them by what they
split rather than by their conventional names.

**Split each matrix.** Every layer's weight matrices are cut across devices, each
device computes its slice, and the slices are summed. The model runs faster because
each device reads less — but the sum has to happen, twice per layer, and eighty layers
means a hundred and sixty synchronisations per token.

**Split the layers.** Device one holds layers 1–10, device two holds 11–20, and an
activation is passed along. There are only seven handoffs for eight devices, so the
communication is trivial. What you pay instead is idleness: while device one works on
token A, devices two through eight have nothing to do until it arrives. The pipe has
to fill and drain, and that bubble costs a fixed share of every device's time.

**Copy the whole model.** Each device has a complete copy and serves different
requests. No communication at all. Also — and this is the part that gets missed — no
speedup whatsoever for any individual request. A request runs on one device at exactly
the speed it always did. You serve more users; you do not serve any of them faster.

**Split the experts.** In a sparse model, different devices hold different experts and
tokens are routed to wherever their expert lives. Communication is an all-to-all
shuffle rather than a sum, and it happens once per expert layer rather than twice per
layer.

The decision among them is mostly a question about your cables.

Tensor parallelism's hundred and sixty synchronisations per token are free if the
devices are in one chassis connected by a dedicated high-bandwidth fabric. Across an
ordinary network they take longer than the computation they are accelerating, and the
eight-device configuration becomes slower than one device would have been. **This is
why tensor parallelism stops at the node boundary** — not by convention, but because
the arithmetic stops working there.

Pipeline parallelism's seven handoffs are cheap on any link, which is why crossing
nodes means pipelining. Its bubble is the price, and the bubble shrinks as you put more
work in flight — which is another reason batch size matters.

The second half of the chapter is about a claim that sounds like it dissolves this
whole problem: sparse models, where a trillion parameters cost what a small model
costs because each token only visits a few experts.

For one token, exactly right. For a batch, wrong, and wrong for a reason that is
obvious once stated: **different tokens pick different experts.** Two tokens might use
four experts between them. Thirty-two tokens will collectively touch most of them.
And the device has to read every expert any token in the batch needs.

So sparsity is a property of a single token's path, and batching destroys it by
construction. The two techniques this part has spent three chapters recommending —
batch hard, use sparse models — are in direct competition, and at production batch
sizes the sparse model is reading essentially all of itself.

## 5. Formal Explanation

Let a model have $P$ parameters, $L$ layers, hidden dimension $d$, batch $m$, and $b$
bytes per weight. Split across $n$ devices, each device reads $Pb/n$ weight bytes for
the three model-splitting dimensions and $Pb$ for data parallelism.

Communication per device per decode step differs by dimension. An activation is
$a = mdb$ bytes. Tensor parallelism performs two all-reduces per layer; a ring
all-reduce of $a$ bytes over $n$ devices moves $2a(n-1)/n$ per device, giving

$$ C_{\text{tensor}} \;=\; 2L \cdot 2a\frac{n-1}{n} $$

Pipeline parallelism passes one activation per stage boundary:
$C_{\text{pipe}} = a(n-1)$. Expert parallelism performs an all-to-all dispatch and
combine on the expert layers: $C_{\text{expert}} = 2 \cdot \frac{L}{2} \cdot a\frac{n-1}{n}$.
Data parallelism: $C_{\text{data}} = 0$.

The ratio that decides the design is

$$ \frac{C_{\text{tensor}}}{C_{\text{pipe}}} \;=\; \frac{4L(n-1)/n}{n-1} \;=\; \frac{4L}{n} $$ (eq:parallelism-dimension-is-an-interconnect-decision)

**Linear in depth and inverse in device count** — for $L = 80$ and $n = 8$ that is 40,
which is what {{sec:9-practical-example}} measures.

Step time adds communication to the weight read, and pipeline parallelism divides by
its bubble efficiency $\beta = m/(m + n - 1)$:

$$ T \;=\; \frac{1}{\beta}\left(\frac{Pb}{nB_{\text{hbm}}} \;+\; \frac{C}{B_{\text{link}}}\right) $$

For tensor parallelism to be worth adding, communication must stay small relative to
the weight read it saves. Requiring $C/B_{\text{link}} \le \epsilon \cdot Pb/(nB_{\text{hbm}})$
and substituting gives

$$ B_{\text{link}} \;\ge\; \frac{4Lmdb\,(n-1)}{\epsilon\,Pb} \cdot B_{\text{hbm}} $$ (eq:tensor-parallelism-is-in-node)

**The requirement grows with $n$ while the weight read per device shrinks with $n$** —
both terms move the wrong way, so the required bandwidth rises roughly as $n^2/(n)$ per
unit of benefit. {{sec:9-practical-example}} finds 40 GB/s needed at 2 devices and 602
GB/s at 16.

For sparsity, let a model have $E$ experts with top-$k$ routing and dense fraction
$\delta$. A given expert is missed only if all $m$ tokens avoided it, so the expected
number touched is

$$ \mathbb{E}[\text{touched}] \;=\; E\left(1 - \left(1 - \frac{k}{E}\right)^{m}\right) $$ (eq:sparsity-erodes-with-batch-size)

which approaches $E$ exponentially in $m$ with rate $k/E$. The batch at which a
fraction $f$ of experts is touched is $m_f = \log(1-f)/\log(1 - k/E)$.

## 6. Mathematical Foundation

The sparsity result deserves its own statement because it inverts an intuition.

Read bytes at batch $m$ are $b\bigl(\delta P + \mathbb{E}[\text{touched}] \cdot
\frac{(1-\delta)P}{E}\bigr)$, which at $m = 1$ is the active size and as
$m \to \infty$ is $bP$ — the full dense model. Throughput per token is therefore
bounded between the dense-active and dense-total curves, and it crosses from one to
the other over a range of $m$ set entirely by $k/E$.

The half-coverage batch is

$$ m_{1/2} \;=\; \frac{\log 2}{-\log(1 - k/E)} \;\approx\; \frac{E \log 2}{k} $$

so **finer routing erodes faster**, which is the opposite of what "more sparsity is
better" suggests. {{sec:9-practical-example}} finds $m_{1/2} = 21.8$ for 64 experts at
top-2 and $2.4$ for 8 experts at top-2 — both well below the batch sizes
{{ch:inf-cpu-gpu}} argued for.

That gives the chapter's sharpest result. Writing \(R_{\text{serve}}\) for read bytes per token and \(R_{\text{train}}\) for the parameters a gradient step touches,
the two regimes diverge:

$$ \lim_{m \to \infty} R_{\text{serve}}(m) = Pb, \qquad R_{\text{train}} = \left(\delta + \frac{k}{E}(1 - \delta)\right)Pb $$ (eq:training-economics-are-not-serving-economics)

**Serving converges to the full model; training does not.** A gradient step updates
only the experts a token routed to, so sparsity survives batching there -- each token's
gradient is separate. Serving batches tokens into one weight read, and a shared read
must cover every expert any token needs.

At batch 128 the MoE step costs **0.99x** a dense model of the *full* 600B -- neither
cheaper than its active size nor more expensive than its total size. It converges to
the dense model it structurally is.

{{cite:fedus2021switch}}'s claim is a *training* claim: up to **7× pre-training speed
at equal compute**, because during training each token's gradient touches only its
experts and the optimiser state is what dominates. Serving has no equivalent, because
serving batches tokens and gradients do not. **The training economics and the serving
economics are different arguments and they get conflated.**

{{cite:rajbhandari2020zero}}'s super-linear speedup on 400 devices is the same
distinction from the other side: it is super-linear because aggregate *memory* rises
with device count, a capacity effect rather than a compute one, and it applies to
training's optimiser state rather than to serving's weight read.

## 7. Internal Mechanics

**Why tensor parallelism synchronises twice per layer.** The attention block's output
projection and the MLP's down projection both sum contributions computed on different
devices, and neither can proceed until the sum completes. There is no reformulation
that avoids it — the sum is the mathematics, not the implementation — which is why
{{cite:shoeybi2019megatron}}'s design is essentially the only one in use.

**Where the pipeline bubble goes.** With $n$ stages and $m$ microbatches, $n-1$ stages
are idle while the first fills and again while the last drains, giving efficiency
$m/(m+n-1)$. More microbatches shrink it, which is why pipeline parallelism and large
batches go together and why a pipeline-parallel deployment at low load is
disproportionately inefficient.

**The interaction between splitting weights and cache capacity.** Splitting weights
across $n$ devices frees $(n-1)Pb/n$ bytes of weight memory per device *and* provides
$n$ times the total memory, so by
{{eq:batch-times-context-is-the-budget}} the token-slot budget grows superlinearly.
This is a real and frequently under-counted benefit of model parallelism: the speedup
tables understate it because they hold batch fixed.

**Why the two regimes differ.** The asymmetry in
{{eq:training-economics-are-not-serving-economics}} is worth locating precisely,
because it is easy to state and easy to get backwards. Training's per-token work is
separable: token A's gradient touches token A's experts and nothing else, and the
optimiser applies updates expert by expert. Serving's per-step work is *not* separable,
because the weight read is shared -- that sharing is the entire point of batching, and
it is what {{ch:inf-cpu-gpu}} showed makes the hardware usable. So the mechanism that
makes batched serving efficient is exactly the mechanism that destroys sparsity, and no
implementation can have both.

**Expert parallelism's load imbalance.** The communication model assumes uniform
routing. Real routers are not uniform — some experts are popular — so a device holding
a popular expert becomes the bottleneck and the all-to-all completes at the slowest
participant. Auxiliary load-balancing losses during training exist to mitigate this,
and their effectiveness at serving time is workload-dependent.

**Why the four dimensions are not interchangeable.** It is tempting to treat the
choice as an optimisation over one quantity, but the dimensions are not substitutes for
each other -- they solve different problems, and a deployment usually needs several at
once. A 400B model on 80 GB devices needs *at least* eight-way model splitting to fit
at all, which is a capacity requirement with no latency content; whether that eight-way
split is tensor or pipeline is then a separate, latency-driven decision; and whether to
replicate the resulting eight-device unit is a third, throughput-driven one. Treating
these as one search over "how many GPUs" produces configurations that satisfy none of
the three constraints well, and the fix is to answer them in that order: fit, then
latency, then throughput.

**Why data parallelism still matters.** It is the only dimension that scales
throughput without touching latency or requiring an interconnect, so it composes with
everything else: a deployment is typically tensor-parallel within a node, pipeline- or
data-parallel across nodes. The mistake is not using it; it is reaching for it when the
problem is latency.

**Paging under model parallelism.** {{cite:kwon2023pagedattention}}'s block allocator
must be coordinated across tensor-parallel ranks, since every rank holds a slice of the
same sequence's cache. This is why paged attention and tensor parallelism are
co-designed in serving stacks rather than composed, a point {{ch:inf-serving-stacks}}
takes up.

## 8. Implementation

The first listing measures communication volume and achievable speedup for each
dimension at each interconnect speed.

```python {tier=A name=cg1}
"""Four ways to split a model across devices, and each buys a different thing.

When a model does not fit, or does not run fast enough, you add devices. But "add
devices" is four different decisions with four different communication costs, and
picking wrong makes the system slower than the single device you started with.

  tensor    split each layer's matrices; every layer needs an all-reduce
  pipeline  give each device some layers; devices idle waiting for each other
  data      replicate the model; helps throughput, never helps one request
  expert    route tokens to a subset of experts; communication is all-to-all

This listing measures the communication each imposes per decode step and finds which
dimension is viable at which interconnect speed
(eq:parallelism-dimension-is-an-interconnect-decision).
"""
import math

PARAMS = 70.0e9
BYTES = 2.0
LAYERS = 80
D_MODEL = 8192
BATCH = 32
HBM_BW = 3.35e12
PEAK = 9.89e14

# (link, bytes/s, description)
LINKS = [
    ("NVLink in-node",  9.00e11),
    ("PCIe in-node",    6.40e10),
    ("200G ethernet",   2.50e10),
    ("25G ethernet",    3.10e09),
]
DEVICES = [1, 2, 4, 8, 16]


def weight_bytes(n_dev, mode):
    """Weight bytes each device must read per step."""
    if mode in ("tensor", "pipeline", "expert"):
        return PARAMS * BYTES / n_dev
    return PARAMS * BYTES          # data parallel: full copy each


def bubble_factor(n_dev, mode):
    """Pipeline parallelism idles each stage while the pipe fills and drains.

    With `n_dev` stages and BATCH microbatches, the share of time doing useful
    work is BATCH / (BATCH + n_dev - 1). Other dimensions have no bubble.
    """
    if mode != "pipeline" or n_dev == 1:
        return 1.0
    return BATCH / float(BATCH + n_dev - 1)


def comm_bytes(n_dev, mode):
    """Bytes each device sends per decode step."""
    if n_dev == 1:
        return 0.0
    if mode == "tensor":
        # Two all-reduces per layer over the activation, batch times d_model.
        act = BATCH * D_MODEL * BYTES
        return 2.0 * LAYERS * act * 2.0 * (n_dev - 1) / n_dev
    if mode == "pipeline":
        # One activation handoff per stage boundary.
        act = BATCH * D_MODEL * BYTES
        return act * (n_dev - 1)
    if mode == "expert":
        # All-to-all dispatch and combine, once per layer with experts.
        act = BATCH * D_MODEL * BYTES
        return 2.0 * (LAYERS / 2.0) * act * (n_dev - 1) / n_dev
    return 0.0                     # data parallel: no per-step communication


print("A %.0fB model, %d layers, d_model %d, batch %d."
      % (PARAMS / 1e9, LAYERS, D_MODEL, BATCH))
print("Weights are %.0f GB; one activation is %.2f MB."
      % (PARAMS * BYTES / 1e9, BATCH * D_MODEL * BYTES / 1e6))
print()
print("Per-step communication volume by dimension, per device.")
print()
MODES = ["tensor", "pipeline", "data", "expert"]
print(f"{'devices':>9}" + "".join(f"{m:>14}" for m in MODES))
print("-" * 65)
comm = {}
for n in DEVICES:
    row = {}
    cells = ""
    for m in MODES:
        c = comm_bytes(n, m)
        row[m] = c
        cells += f"{c / 1e6:>13.1f}M"
    comm[n] = row
    print(f"{n:>9}{cells}")

print()
print()
print("Time per step, at each interconnect. Compute floor is weight-read time;")
print("communication is added on top.")
print()
for link, bw in LINKS:
    print(f"{link} ({bw / 1e9:.0f} GB/s):")
    print(f"{'devices':>9}" + "".join(f"{m:>14}" for m in MODES))
    print("  " + "-" * 63)
    for n in DEVICES:
        cells = ""
        for m in MODES:
            t_w = weight_bytes(n, m) / HBM_BW
            t_c = comm[n][m] / bw
            t = (t_w + t_c) / bubble_factor(n, m)
            cells += f"{t * 1000:>12.2f}ms"
        print(f"{n:>9}{cells}")
    print()

print()
print("Speedup against one device, which is the number the choice turns on.")
print()
base = PARAMS * BYTES / HBM_BW
for link, bw in LINKS:
    print(f"{link}:")
    print(f"{'devices':>9}" + "".join(f"{m:>14}" for m in MODES))
    print("  " + "-" * 63)
    for n in DEVICES:
        cells = ""
        for m in MODES:
            t_w = weight_bytes(n, m) / HBM_BW
            t_c = comm[n][m] / bw
            t = (t_w + t_c) / bubble_factor(n, m)
            cells += f"{base / t:>13.2f}x"
        print(f"{n:>9}{cells}")
    print()

print()
print("What interconnect tensor parallelism REQUIRES: the bandwidth at which")
print("communication stays under a tenth of the weight read.")
print()
print(f"{'devices':>9}{'comm MB/step':>15}{'weight read ms':>17}"
      f"{'GB/s needed':>14}{'cheapest link that works':>27}")
print("-" * 82)
need = {}
for n in DEVICES:
    if n == 1:
        continue
    c = comm_bytes(n, "tensor")
    t_w = weight_bytes(n, "tensor") / HBM_BW
    bw_needed = c / (0.10 * t_w)
    ok = [nm for nm, bw in LINKS if bw >= bw_needed]
    need[n] = (c, t_w, bw_needed, ok[-1] if ok else "none")
    print(f"{n:>9}{c / 1e6:>15.1f}{t_w * 1000:>17.2f}{bw_needed / 1e9:>14.0f}"
          f"{(ok[-1] if ok else 'none'):>27}")

print()
print("The requirement rises with device count because communication grows while")
print("the weight read per device shrinks -- both move the wrong way.")

print()
print()
print("And what each dimension actually buys, stated plainly.")
print()
print(f"{'dimension':>12}{'fits a bigger model':>22}{'faster single request':>24}"
      f"{'more requests':>16}")
print("-" * 76)
BUYS = [
    ("tensor",   "yes", "yes", "no"),
    ("pipeline", "yes", "no",  "yes"),
    ("data",     "no",  "no",  "yes"),
    ("expert",   "yes", "yes", "no"),
]
for m, a, b, c in BUYS:
    print(f"{m:>12}{a:>22}{b:>24}{c:>16}")

print(f"""
The communication table is the whole decision. At {8} devices, tensor parallelism moves
{comm[8]['tensor'] / 1e6:.1f} MB per device per step; pipeline moves
{comm[8]['pipeline'] / 1e6:.1f} MB; data parallelism moves nothing.

That is a factor of {comm[8]['tensor'] / comm[8]['pipeline']:.0f} between the two
dimensions that both split the model, and it comes from a structural difference:
**tensor parallelism synchronises twice per layer and pipeline parallelism
synchronises once per stage** (eq:parallelism-dimension-is-an-interconnect-decision).
Eighty layers against seven stage boundaries.

The timing tables turn that into a viability question, and the answer changes
completely with the link. On NVLink, tensor parallelism at {8} devices gives
{base / (weight_bytes(8, 'tensor') / HBM_BW + comm[8]['tensor'] / LINKS[0][1]):.2f}x
speedup against pipeline's
{base / ((weight_bytes(8, 'pipeline') / HBM_BW + comm[8]['pipeline'] / LINKS[0][1]) / bubble_factor(8, 'pipeline')):.2f}x
-- tensor wins, because the bubble costs pipeline
{1 - bubble_factor(8, 'pipeline'):.0%} of its time and the fast link makes tensor's
communication nearly free.

On 25G ethernet the same tensor configuration gives
{base / (weight_bytes(8, 'tensor') / HBM_BW + comm[8]['tensor'] / LINKS[3][1]):.2f}x --
**slower than a single device**, if a single device could hold the model -- while
pipeline still gives
{base / ((weight_bytes(8, 'pipeline') / HBM_BW + comm[8]['pipeline'] / LINKS[3][1]) / bubble_factor(8, 'pipeline')):.2f}x.

**Tensor parallelism is an in-node technique.** Not by convention: the table says the
communication is {comm[8]['tensor'] / 1e6:.0f} MB per step, and at
{LINKS[3][1] / 1e9:.1f} GB/s that is
{comm[8]['tensor'] / LINKS[3][1] * 1000:.0f}ms against a
{base * 1000:.1f}ms weight read.

Pipeline parallelism survives the slow link -- {comm[8]['pipeline'] / 1e6:.1f} MB is
{comm[8]['pipeline'] / LINKS[3][1] * 1000:.2f}ms even at
{LINKS[3][1] / 1e9:.1f} GB/s -- and pays instead in bubbles, losing
{1 - bubble_factor(8, 'pipeline'):.0%} at {8} stages and
{1 - bubble_factor(16, 'pipeline'):.0%} at {16}.

**That is the trade, and it is why real deployments use both.** Tensor parallelism
within a node where the link is fast and the bubble would hurt; pipeline parallelism
across nodes where the link is slow and the bubble is the cheaper cost. The topology is
not a preference -- it is what the arithmetic permits.

The requirement table states the constraint as a purchasing decision. Holding tensor
parallelism's communication under a tenth of the weight read needs
{need[2][2] / 1e9:.0f} GB/s at {2} devices and {need[16][2] / 1e9:.0f} GB/s at {16}.

**Both terms move the wrong way**: communication grows with device count while the
weight read per device shrinks, so the required bandwidth rises roughly with the square
of the split. That is the arithmetic behind the industry's node boundary, and it is why
the eight-or-sixteen-device node is a hardware convention that exists because of this
table rather than despite it.

The last table is the part most often confused, and it is worth being blunt about.
**Data parallelism never makes a single request faster.** It replicates the model and
serves more requests, which raises throughput and leaves latency exactly where it was.
A team that adds data-parallel replicas to fix a latency problem has bought nothing,
and the reason the mistake is common is that "add more GPUs" sounds like one action.

It is four actions. Tensor and expert parallelism make a request faster and cost
interconnect. Pipeline parallelism fits a bigger model and costs bubbles. Data
parallelism serves more users and costs memory. Choosing among them starts with which
of those three problems you actually have -- and cite:shoeybi2019megatron's
{0.76:.0%} scaling efficiency at {512} devices is a tensor-and-pipeline result on fast
interconnect, not a general claim about adding hardware.""")
```

## 9. Practical Example

A 70B model, 80 layers, at batch 32:

```
  devices        tensor      pipeline          data        expert
-----------------------------------------------------------------
        1          0.0M          0.0M          0.0M          0.0M
        2         83.9M          0.5M          0.0M         21.0M
        4        125.8M          1.6M          0.0M         31.5M
        8        146.8M          3.7M          0.0M         36.7M
       16        157.3M          7.9M          0.0M         39.3M
```

Tensor parallelism moves **40×** what pipeline parallelism moves at 8 devices
({{eq:parallelism-dimension-is-an-interconnect-decision}}) — eighty layers of
synchronisation against seven stage boundaries.

Speedup on a fast in-node link:

```
  devices        tensor      pipeline          data        expert
        1         1.00x         1.00x         1.00x         1.00x
        2         1.99x         1.94x         1.00x         2.00x
        4         3.95x         3.66x         1.00x         3.99x
        8         7.76x         6.56x         1.00x         7.94x
       16        15.00x        10.86x         1.00x        15.74x
```

And on 25G ethernet:

```
  devices        tensor      pipeline          data        expert
        1         1.00x         1.00x         1.00x         1.00x
        2         0.87x         1.92x         1.00x         1.51x
        4         0.82x         3.49x         1.00x         2.03x
        8         0.79x         5.35x         1.00x         2.45x
       16         0.78x         5.53x         1.00x         2.73x
```

**On the fast link tensor wins (7.76× against 6.56×); on the slow link it is slower
than a single device (0.79×) while pipeline still gives 5.35×**
({{eq:tensor-parallelism-is-in-node}}).

That is the trade in one comparison. Tensor parallelism has no bubble and enormous
communication; pipeline has trivial communication and loses **18%** to the bubble at 8
stages, **32%** at 16. Which is cheaper depends entirely on the link — and it is why
real deployments use tensor parallelism within a node and pipeline parallelism across
nodes.

Stated as a purchasing requirement:

```
  devices   comm MB/step   weight read ms   GB/s needed   cheapest link that works
----------------------------------------------------------------------------------
        2           83.9            20.90            40               PCIe in-node
        4          125.8            10.45           120             NVLink in-node
        8          146.8             5.22           281             NVLink in-node
       16          157.3             2.61           602             NVLink in-node
```

**Both terms move the wrong way** — communication grows with device count while the
weight read per device shrinks — so the required bandwidth rises steeply. This is the
arithmetic behind the industry's node boundary.

```mermaid {#fig:dimensions caption="Four dimensions, four costs. Tensor parallelism trades enormous communication for no bubble; pipeline trades a bubble for trivial communication; data parallelism costs nothing and helps no single request."}
flowchart TD
  A["need more devices"] --> B{"what problem?"}
  B -->|"model does not fit"| C["tensor or pipeline<br/>or expert"]
  B -->|"single request too slow"| D["tensor or expert<br/>needs fast link"]
  B -->|"not enough throughput"| E["data parallel<br/>no communication"]
  C --> F{"link speed?"}
  F -->|"in-node fabric"| G["tensor: 7.76x at 8"]
  F -->|"network"| H["pipeline: 5.35x at 8"]
```

And what each actually buys:

```
   dimension   fits a bigger model   faster single request   more requests
----------------------------------------------------------------------------
      tensor                   yes                     yes              no
    pipeline                   yes                      no             yes
        data                    no                      no             yes
      expert                   yes                     yes              no
```

**Data parallelism never makes a single request faster.** A team that adds replicas to
fix a latency problem has bought nothing.

The second listing turns to sparsity.

```python {tier=A name=cg2}
"""Sparsity breaks the arithmetic-intensity story, and then routing breaks it back.

cite:fedus2021switch decoupled parameter count from per-token compute: a token visits
only a few experts, so a trillion-parameter model can cost what a small one costs.

For serving that sounds decisive. ch:inf-cpu-gpu found decode bound by reading every
weight; if a token reads only its experts, the read shrinks in proportion.

It does -- for one token. For a BATCH of tokens the experts are chosen independently,
so the union of experts touched grows with the batch until it covers everything
(eq:sparsity-erodes-with-batch-size). Sparsity and batching, the two levers this part
has been building on, work against each other.

This listing measures where the erosion bites.
"""
import math

TOTAL_PARAMS = 600.0e9
N_EXPERTS = 64
TOP_K = 2
BYTES = 2.0
HBM_BW = 3.35e12
# Attention and shared layers are dense; only the expert MLPs are sparse.
DENSE_SHARE = 0.22
EXPERT_PARAMS = TOTAL_PARAMS * (1.0 - DENSE_SHARE)
DENSE_PARAMS = TOTAL_PARAMS * DENSE_SHARE
PER_EXPERT = EXPERT_PARAMS / N_EXPERTS

BATCHES = [1, 2, 4, 8, 16, 32, 64, 128]


def experts_touched(batch):
    """Expected distinct experts touched by a batch, tokens routing uniformly.

    Each token picks TOP_K of N_EXPERTS. An expert is missed only if every
    token avoided it.
    """
    p_miss = (1.0 - TOP_K / float(N_EXPERTS)) ** batch
    return N_EXPERTS * (1.0 - p_miss)


def read_bytes(batch):
    return (DENSE_PARAMS + experts_touched(batch) * PER_EXPERT) * BYTES


DENSE_EQUIV = DENSE_PARAMS + TOP_K * PER_EXPERT
print("A %.0fB-parameter sparse model: %d experts, top-%d routing."
      % (TOTAL_PARAMS / 1e9, N_EXPERTS, TOP_K))
print("Dense portion %.0fB; each expert %.1fB; one token activates %.1fB."
      % (DENSE_PARAMS / 1e9, PER_EXPERT / 1e9, DENSE_EQUIV / 1e9))
print()
print("Weights read per decode step, as the batch grows.")
print()
print(f"{'batch':>8}{'experts touched':>18}{'share of experts':>19}"
      f"{'GB read':>11}{'vs dense-equivalent':>22}")
print("-" * 78)
tab = {}
for b in BATCHES:
    e = experts_touched(b)
    r = read_bytes(b)
    tab[b] = (e, r, r / (DENSE_EQUIV * BYTES))
    print(f"{b:>8}{e:>18.1f}{e / N_EXPERTS:>19.1%}{r / 1e9:>11.1f}"
          f"{r / (DENSE_EQUIV * BYTES):>21.2f}x")

print()
print()
print("Throughput per step, against a dense model of the ACTIVE size and a dense")
print("model of the TOTAL size.")
print()
dense_active_ms = DENSE_EQUIV * BYTES / HBM_BW * 1000.0
dense_total_ms = TOTAL_PARAMS * BYTES / HBM_BW * 1000.0
print(f"dense model of active size ({DENSE_EQUIV / 1e9:.0f}B): "
      f"{dense_active_ms:.2f} ms/step")
print(f"dense model of total size ({TOTAL_PARAMS / 1e9:.0f}B):  "
      f"{dense_total_ms:.2f} ms/step")
print()
print(f"{'batch':>8}{'MoE ms':>10}{'tokens/s':>12}{'vs active-dense':>18}"
      f"{'vs total-dense':>17}")
print("-" * 66)
tp = {}
for b in BATCHES:
    ms = read_bytes(b) / HBM_BW * 1000.0
    t = b / (ms / 1000.0)
    tp[b] = (ms, t)
    print(f"{b:>8}{ms:>10.2f}{t:>12.0f}{ms / dense_active_ms:>17.2f}x"
          f"{ms / dense_total_ms:>16.2f}x")

print()
print()
print("The comparison that matters: MoE against a DENSE model of the same active")
print("size, which is what the sparsity claim implicitly promises.")
print()
print(f"{'batch':>8}{'MoE tok/s':>12}{'dense-active tok/s':>21}"
      f"{'MoE advantage':>16}")
print("-" * 58)
adv = {}
for b in BATCHES:
    dense_t = b / (dense_active_ms / 1000.0)
    adv[b] = tp[b][1] / dense_t
    print(f"{b:>8}{tp[b][1]:>12.0f}{dense_t:>21.0f}{tp[b][1] / dense_t:>15.2f}x")

print()
print()
print("Where the erosion sits, by routing width. More experts and lower top-k")
print("mean sparser routing -- and faster erosion.")
print()
print(f"{'experts':>9}{'top-k':>8}{'active B':>11}"
      + "".join(f"{('b=%d' % b):>10}" for b in (1, 8, 32, 128)))
print("-" * 78)
grid = {}
for ne, tk in ((8, 2), (16, 2), (64, 2), (64, 1), (256, 4)):
    per = EXPERT_PARAMS / ne
    active = DENSE_PARAMS + tk * per
    row = []
    for b in (1, 8, 32, 128):
        pm = (1.0 - tk / float(ne)) ** b
        touched = ne * (1.0 - pm)
        r = (DENSE_PARAMS + touched * per) * BYTES
        row.append(r / (active * BYTES))
    grid[(ne, tk)] = row
    print(f"{ne:>9}{tk:>8}{active / 1e9:>11.0f}"
          + "".join(f"{v:>9.2f}x" for v in row))

print()
print()
print("And the batch at which half the experts are touched -- the point past")
print("which the model is effectively dense.")
print()
print(f"{'experts':>9}{'top-k':>8}{'batch for 50%':>16}{'batch for 90%':>16}")
print("-" * 50)
half = {}
for ne, tk in ((8, 2), (16, 2), (64, 2), (64, 1), (256, 4)):
    f = tk / float(ne)
    b50 = math.log(0.5) / math.log(1.0 - f)
    b90 = math.log(0.1) / math.log(1.0 - f)
    half[(ne, tk)] = (b50, b90)
    print(f"{ne:>9}{tk:>8}{b50:>16.1f}{b90:>16.1f}")

print(f"""
The read table is the erosion. At batch 1 the model touches
{tab[1][0]:.1f} experts and reads {tab[1][1] / 1e9:.1f} GB -- close to the
{DENSE_EQUIV / 1e9:.0f}B active size the sparsity claim promises. At batch
{BATCHES[-1]} it touches {tab[128][0]:.1f} of {N_EXPERTS} experts
({tab[128][0] / N_EXPERTS:.0%}) and reads {tab[128][1] / 1e9:.1f} GB
(eq:sparsity-erodes-with-batch-size).

**The sparsity is gone by batch {BATCHES[-1]}.** Not degraded -- gone. The batch
touches essentially every expert, so the step reads essentially every weight, and the
model behaves like the {TOTAL_PARAMS / 1e9:.0f}B dense model it structurally is.

The advantage table states it as the comparison that matters. Against a dense model of
the same ACTIVE size -- which is what "a trillion parameters at the cost of a small
model" implicitly promises -- MoE is {adv[1]:.2f}x at batch 1,
{adv[8]:.2f}x at batch {8}, and {adv[128]:.2f}x at batch {BATCHES[-1]}.

So the promise holds at batch 1 and fails at the batch sizes ch:inf-cpu-gpu showed are
necessary to use the hardware at all. **Sparsity and batching are competing claims on
the same resource**, and a serving design cannot have both.

It is worth being precise about what that does and does not say, because the obvious
reading overshoots.

Look at the other column instead. At batch {BATCHES[-1]} the MoE step costs
{tp[128][0] / dense_total_ms:.2f}x a dense model of the FULL
{TOTAL_PARAMS / 1e9:.0f}B -- essentially identical. So at serving batch sizes, this
model is neither cheaper than its active size nor more expensive than its total size:
**it converges to the dense model it structurally is.**

That is the honest summary. Sparsity does not make a batched MoE cheap; it makes it
cost what a {TOTAL_PARAMS / 1e9:.0f}B dense model costs while having been far cheaper
to TRAIN, which is cite:fedus2021switch's actual claim -- {7}x pre-training speed at
equal compute. **The training economics and the serving economics are different
arguments**, and the serving one does not survive batching.

The routing-width table shows the erosion is a design parameter. A model with
{8} experts at top-{2} is {grid[(8, 2)][3]:.2f}x its active size at batch {128};
one with {256} experts at top-{4} is {grid[(256, 4)][3]:.2f}x. Finer routing erodes
faster, because each token's choice covers a smaller share and the union fills sooner.

The half-coverage table gives the threshold directly. With {64} experts at top-{2},
half the experts are touched by batch {half[(64, 2)][0]:.0f} and ninety percent by
batch {half[(64, 2)][1]:.0f}. With {256} experts at top-{4}, half by batch
{half[(256, 4)][0]:.0f}.

**Those numbers are small.** They sit below every batch size this part has argued for,
which means a production MoE deployment is operating in the dense regime essentially
all the time, and the sparse regime exists only in single-stream benchmarks.

Two consequences follow for design. First, **MoE serving is a memory-capacity problem
rather than a memory-bandwidth one**: the whole model must be resident because the
batch will touch all of it, so ch:inf-gpu-memory's capacity frontier binds and
ch:inf-cpu-gpu's bandwidth story does not simplify. Second, expert parallelism becomes
attractive for a reason unrelated to sparsity -- splitting experts across devices
splits the resident weights, which is the same thing tensor parallelism does with a
different communication pattern, and cg1's table applies.""")
```

A 600B model with 64 experts and top-2 routing:

```
   batch   experts touched   share of experts    GB read   vs dense-equivalent
------------------------------------------------------------------------------
       1               2.0               3.1%      293.2                 1.00x
       8              14.4              22.4%      473.9                 1.62x
      32              40.8              63.8%      861.1                 2.94x
     128              62.9              98.3%     1183.9                 4.04x
```

At batch 128 the model touches **98.3%** of its experts
({{eq:sparsity-erodes-with-batch-size}}). **The sparsity is not degraded — it is
gone.**

Against a dense model of the same active size — what "a trillion parameters at small-model
cost" implicitly promises:

```
   batch   MoE tok/s   dense-active tok/s   MoE advantage
----------------------------------------------------------
       1          11                   11           1.00x
       8          57                   91           0.62x
      32         124                  366           0.34x
     128         362                 1462           0.25x
```

The promise holds at batch 1 and fails at every batch size {{ch:inf-cpu-gpu}} showed is
necessary. But read the other column: at batch 128 the MoE step costs **0.99×** a
dense model of the *full* 600B. **It converges to the dense model it structurally is** —
neither cheaper than its active size nor more expensive than its total size.

{{cite:fedus2021switch}}'s **7×** is a *pre-training* result
({{eq:training-economics-are-not-serving-economics}}). The training and serving
arguments are different, and only one survives batching.

And the erosion is fast:

```
  experts   top-k   batch for 50%   batch for 90%
--------------------------------------------------
        8       2             2.4             8.0
       16       2             5.2            17.2
       64       2            21.8            72.5
      256       4            44.0           146.2
```

Half the experts are touched by batch **21.8** for a 64-expert top-2 model. **A
production MoE deployment operates in the dense regime essentially all the time**; the
sparse regime exists only in single-stream benchmarks.

## 10. Production Considerations

Name the problem before adding devices. Latency, capacity, and throughput have three
different answers and only one of them is data parallelism.

Compute the required interconnect bandwidth before choosing a tensor-parallel degree.
It is one line of arithmetic and it tells you whether the configuration is viable on
the hardware you have.

Keep tensor parallelism within the node and pipeline parallelism across it. This is not
a convention to follow but a conclusion to re-derive when the hardware changes — a
faster network moves the boundary.

Size pipeline microbatches against the bubble. At 8 stages you need well over 32
microbatches for the bubble to be small, which couples pipeline depth to batch size and
therefore to {{ch:inf-gpu-memory}}'s capacity frontier.

Count the capacity benefit of model parallelism, not just the speedup. Splitting weights
frees cache, and the token-slot gain is often larger than the throughput gain.

For MoE, provision memory for the full model and expect dense-model bandwidth at
production batch sizes. Sizing an MoE deployment on active parameters is the single most
expensive mistake available in this chapter.

Re-derive the tensor-parallel degree after any hardware or model change. It depends on
layer count, hidden dimension, batch size, and link bandwidth, and a degree inherited
from a previous model is inherited from different arithmetic.

Measure expert load balance in production. Uniform routing is an assumption, and an
imbalanced router makes the all-to-all complete at the slowest device.

## 11. Common Mistakes

**Adding data-parallel replicas to fix latency.** They cannot; the dimension is
orthogonal to it, and the spend buys throughput nobody asked for.

**Tensor parallelism across nodes.** Slower than one device at ordinary network
speeds, and the configuration looks entirely reasonable on paper.

**Sizing MoE memory on active parameters.** The batch touches nearly all experts, so
the deployment must hold the whole model resident regardless.

**Quoting a sparse model's training efficiency as a serving property.** Different
arguments.

**Ignoring the pipeline bubble at low load.** It is worst exactly when the batch is
small.

**Treating "add GPUs" as one decision.** It is four with different costs.

## 12. Failure Modes

**Silent tensor-parallel degradation after a topology change.** A rack move puts two
ranks on different switches and the step time triples with no configuration change.

**Pipeline stall from an uneven stage split.** One stage holding more layers than
another makes every other stage wait for it; the bubble becomes a bottleneck.

**Expert hotspotting.** A routing skew concentrates tokens on one device, which then
gates the all-to-all for every step.

**MoE memory exhaustion under load.** Provisioned for active parameters, the deployment
runs until batch size grows and then cannot fit the experts the batch requires.

**Capacity loss from over-splitting.** Very high tensor-parallel degree leaves each
device with little weight memory but adds communication buffers, and the net token-slot
gain can be negative.

**Configuration inherited across models.** A parallelism degree that was correct for
one model is applied to another with different depth and hidden dimension, where the
communication arithmetic gives a different answer and nobody re-derived it.

## 13. Alternatives

**A smaller model.** Removes the problem rather than distributing it, and is frequently
the correct answer once the interconnect requirement is priced. A model that fits on
one device needs none of this chapter, and the quality difference is often smaller
than the operational difference.

**Quantisation.** Halving $b$ halves both the weight read and the tensor-parallel
communication, since activations shrink too. Composes with every dimension here.

**Sequence parallelism.** Split along the sequence rather than the hidden dimension,
reducing activation memory. Useful for long context and orthogonal to the four here.

**Offloading to host memory.** {{cite:rajbhandari2020zero}}'s approach, viable for
training where each byte moved amortises over a large gradient computation, and
unviable for decode at {{ch:inf-gpu-memory}}'s 52× PCIe penalty.

**Serving the dense equivalent of an MoE.** If the batch makes the model dense anyway,
a dense model of the same total size is simpler to serve and performs comparably —
worth checking rather than assuming, and rarely checked.

## 14. Evaluation

Report speedup per dimension separately, not aggregate scaling. A deployment that is
tensor-parallel by 8 and data-parallel by 4 has two different efficiency figures and
averaging them hides both.

Measure achieved interconnect bandwidth under real load, not the rated figure. Ring
all-reduce achieves a fraction of link peak, and the fraction depends on message size.

Track pipeline bubble as a first-class metric — it is $1 - m/(m+n-1)$ and it moves with
load, so a system efficient at peak can be badly inefficient at 3am.

For MoE, report experts touched per step alongside batch size. It is the number that
determines the memory read and nothing else reports it.

Measure expert load distribution and report the ratio of the busiest expert to the
mean. That ratio is the all-to-all's effective slowdown.

## 15. Advanced Concepts

The uniform-routing assumption in {{eq:sparsity-erodes-with-batch-size}} is optimistic
in one direction and pessimistic in another. Real routing is *correlated within a
batch* — tokens from one document or one language route similarly — which means fewer
experts are touched than the independent model predicts, and sparsity survives to
larger batches than the arithmetic suggests. Against that, routing is *skewed* across
experts, so the experts that are touched are touched by more tokens, which does not
change the read but does worsen load balance. Batch composition therefore becomes a
serving lever: batching requests from the same domain preserves sparsity that a mixed
batch destroys. As far as the author is aware, no published system routes batch
formation on this basis, and it is the most promising unexplored idea in this chapter.

The communication model treats all-reduce as ring-based with $2a(n-1)/n$ per device.
Tree and hierarchical algorithms trade latency against bandwidth differently, and for
the small activations of decode — a fraction of a megabyte — the latency term dominates
rather than the bandwidth term. So {{eq:tensor-parallelism-is-in-node}}'s bandwidth
framing understates the problem at small batch, where per-message latency rather than
volume is binding. This is why tensor-parallel decode at batch 1 is worse than the
model predicts.

There is a composition question this chapter does not settle. Tensor, pipeline, data,
and expert parallelism can all be applied simultaneously, giving a four-dimensional
configuration space with a communication cost that is not separable across dimensions —
tensor-parallel ranks within a pipeline stage share a link with the pipeline handoff.
Finding the optimum is a constrained optimisation problem over the physical topology,
and it is generally done by search rather than derivation.

## 16. Connection to Previous Chapters

{{eq:decode-is-bandwidth-bound}} from {{ch:inf-cpu-gpu}} is what every dimension here
attacks: each device reads fewer weight bytes, and communication is the price.

{{eq:batch-is-the-mechanism-not-an-optimisation}} is in direct tension with
{{eq:sparsity-erodes-with-batch-size}}. The batch that makes the hardware usable is the
batch that destroys sparsity.

{{eq:roofline-has-multiple-ridges}} from {{ch:inf-gpu-memory}} supplies the tier
hierarchy in which NVLink and ethernet sit, and
{{eq:tensor-parallelism-is-in-node}} is that hierarchy applied to a design decision.

{{eq:batch-times-context-is-the-budget}} explains the under-counted benefit: model
parallelism buys cache capacity as well as speed.

## 17. Exercises

1. Derive $C_{\text{tensor}}/C_{\text{pipe}} = 4L/n$ and evaluate it for a 120-layer
   model at 4 and 32 devices.

2. Compute the interconnect bandwidth required for tensor parallelism at degree 8 for a
   400B model at batch 64. Which links qualify?

3. For a pipeline of 16 stages, find the microbatch count at which the bubble costs less
   than 10%.

4. Find the batch at which a 128-expert top-8 model touches 95% of its experts. Is that
   above or below a realistic serving batch?

5. Model correlated routing — tokens from the same document sharing expert preferences —
   and find how much it delays the erosion.

## 18. Interview Questions

1. We added four more GPUs and latency did not change. What did we probably do?

2. Why does tensor parallelism stop at the node boundary?

3. What does pipeline parallelism cost, and when is that cost worst?

4. A vendor offers a 1T-parameter MoE that "runs at 40B cost." What do you ask?

5. Our MoE deployment ran out of memory at higher load. Explain.

6. We are moving from a 40-layer model to a 100-layer one on the same hardware. What
   changes about the tensor-parallel degree, and why?

## 19. Research Questions

1. How much does batch composition — grouping requests likely to route similarly —
   preserve MoE sparsity in practice?

2. What is the right all-reduce algorithm for decode's small activations, where latency
   rather than bandwidth binds?

3. Can the four-dimensional parallelism configuration be derived from topology rather
   than searched?

4. How skewed is expert routing on production traffic, and what does the skew cost in
   all-to-all completion time?

## 20. Chapter Summary

Four dimensions, four costs. Tensor parallelism synchronises twice per layer and moves
**146.8 MB** per device per step at 8 devices; pipeline synchronises once per stage and
moves **3.7 MB**; data parallelism moves nothing
({{eq:parallelism-dimension-is-an-interconnect-decision}}). The ratio is $4L/n$.

That decides the topology. On a fast in-node link tensor parallelism gives **7.76×** at
8 devices against pipeline's **6.56×**; on 25G ethernet it gives **0.79× — slower than
one device** — while pipeline gives **5.35×**
({{eq:tensor-parallelism-is-in-node}}). Required bandwidth rises from **40 GB/s** at 2
devices to **602 GB/s** at 16, because communication grows while the per-device weight
read shrinks.

**Data parallelism never makes a single request faster.**

Sparsity erodes with batch. A 600B model with 64 experts at top-2 touches **3.1%** of
its experts at batch 1 and **98.3%** at batch 128
({{eq:sparsity-erodes-with-batch-size}}), reaching half coverage by batch **21.8**. At
batch 128 it costs **0.99×** a dense model of the full 600B — it converges to the dense
model it structurally is, and {{cite:fedus2021switch}}'s **7×** was a pre-training
result ({{eq:training-economics-are-not-serving-economics}}).

Both results are instances of one habit worth carrying: check whether a claimed
property survives the operating conditions you actually run in. Tensor parallelism's
speedup is real and it is a property of the link; sparsity's cost saving is real and
it is a property of the batch. Neither is wrong as published, and both are routinely
quoted outside the regime that produced them. The question to ask of any performance
claim in this part is not whether it is true but at what batch size, on what
interconnect, and at what context length.

Carry forward: **name the problem before adding devices**, and **size an MoE for its
total parameters, not its active ones**.

## 21. Further Reading

- {{cite:shoeybi2019megatron}} — tensor parallelism, and the scaling-efficiency figure
  everything else is compared against.
- {{cite:rajbhandari2020zero}} — partitioning optimiser state; the training-side
  capacity argument.
- {{cite:fedus2021switch}} — sparse routing, and the pre-training claim this chapter
  separates from serving.
- {{cite:kwon2023pagedattention}} — paging under model parallelism, where the cache is
  sliced across ranks.
