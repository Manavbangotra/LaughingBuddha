# -*- coding: utf-8 -*-
# Extracted from: Chapter 200 — Parallelism: Tensor, Pipeline, Data, and Expert
# Source: src/.../ch200-parallelism.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

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
