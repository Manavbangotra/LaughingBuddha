# -*- coding: utf-8 -*-
# Extracted from: Chapter 234 — Mixture of Experts and Sparse Models
# Source: src/.../ch234-mixture-of-experts.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Sparsity trades a training cost for a serving cost, and the exchange rate is your traffic.

A mixture-of-experts layer replicates the feed-forward block E times and routes each token to k
of them (cite:shazeer2017moe, cite:fedus2021switch). Total parameters grow by roughly E; FLOPs
per token grow by roughly k. That decoupling is the whole idea and it is real.

What it does not do is make the model cheaper to serve. A generation step reads weights and does
little arithmetic per byte read, so throughput is set by bandwidth
(cite:pope2022inference), and sparse parameters cost exactly as many bytes as dense ones
(eq:sparsity-moves-the-bottleneck-from-flops-to-memory).

So the honest framing is not "cheaper" or "better". It is an exchange: fewer training FLOPs for
a given quality, more bytes moved per served token, and a break-even measured in tokens served
(eq:sparsity-trades-training-cost-for-serving-cost).
"""
D_MODEL, LAYERS = 4096, 32
D2 = D_MODEL ** 2
ATTN_PER_LAYER = 4 * D2          # qkv + out projections
FFN_PER_LAYER = 8 * D2           # up + down, 4x expansion
BYTES = 2                        # bf16 weights
RHO = 0.35                       # sparse parameters are worth this much, in log space
E_FLOOR, A_RED, ALPHA = 1.69, 753.6, 0.34
HBM_BW = 3.35e12                 # bytes/s per accelerator
FLOPS = 9.9e14                   # bf16 FLOP/s per accelerator
GPU_HOUR = 3.20
TRAIN_TOKENS = 4e12


def config(experts, k):
    attn = LAYERS * ATTN_PER_LAYER
    total = attn + LAYERS * experts * FFN_PER_LAYER
    active = attn + LAYERS * k * FFN_PER_LAYER
    return total, active


def effective(total, active):
    """Dense-equivalent parameters: sparse capacity counts, sublinearly."""
    return active * (total / active) ** RHO


def loss_of(n_eff):
    return E_FLOOR + A_RED / n_eff ** ALPHA


def dense_for(target_loss):
    """Parameter count a dense model needs to reach a given loss."""
    return (A_RED / (target_loss - E_FLOOR)) ** (1 / ALPHA)


CONFIGS = [
    ("dense",           1, 1),
    ("8 experts, k=1",  8, 1),
    ("8 experts, k=2",  8, 2),
    ("64 experts, k=2", 64, 2),
    ("128 experts, k=4", 128, 4),
]

print("What sparsity buys and what it costs.")
print()
print(f"{'configuration':>20}{'total params':>15}{'active':>13}{'dense-equiv':>14}"
      f"{'loss':>9}{'weights (GB)':>15}{'80GB GPUs':>12}")
print("-" * 98)
rows = {}
for name, e, k in CONFIGS:
    total, active = config(e, k)
    neff = effective(total, active)
    gb = total * BYTES / 1e9
    rows[name] = (total, active, neff, loss_of(neff), gb, int(-(-gb // 80)), e, k)
    print(f"{name:>20}{total:>15.3e}{active:>13.3e}{neff:>14.3e}"
          f"{loss_of(neff):>9.3f}{gb:>15.1f}{-(-gb // 80):>12.0f}")

DENSE = rows["dense"]
BEST = min(rows, key=lambda n: rows[n][3])
print()
print(f"best loss: {BEST} at {rows[BEST][3]:.3f}, against {DENSE[3]:.3f} dense")
print(f"it uses {rows[BEST][1] / DENSE[1]:.1f}x the FLOPs per token"
      f" and {rows[BEST][0] / DENSE[0]:.1f}x the weights")

print()
print()
print("The fair comparison: a dense model at the same quality.")
print()
print(f"{'configuration':>20}{'loss':>9}{'dense needed':>15}{'active params':>16}"
      f"{'training FLOPs':>17}{'cheaper to train by':>22}")
print("-" * 99)
match = {}
for name, e, k in CONFIGS:
    total, active, neff, l, gb, gpus, _, _ = rows[name]
    dn = dense_for(l)
    tr_sparse = 6 * active * TRAIN_TOKENS
    tr_dense = 6 * dn * TRAIN_TOKENS
    match[name] = (dn, tr_sparse, tr_dense)
    print(f"{name:>20}{l:>9.3f}{dn:>15.3e}{active:>16.3e}"
          f"{tr_sparse:>17.2e}{tr_dense / tr_sparse:>21.1f}x")

print()
print(f"`{BEST}` trains {match[BEST][2] / match[BEST][1]:.1f}x cheaper than the dense")
print(f"model it matches ({match[BEST][0]:.2e} parameters)")

print()
print()
print("Now serving, at system level, where the bytes are charged.")
print()


def touched(experts, k, batch):
    """Expected share of experts at least one token in the batch routes to."""
    if experts == 1:
        return 1.0
    return 1.0 - (1.0 - k / experts) ** batch


def gpus_for(gb):
    return max(1, int(-(-gb // 80)))


def serve(total_gb, attn_bytes, expert_bytes, active, experts, k, batch):
    """System tokens/s and the GPU count holding the weights."""
    g = gpus_for(total_gb)
    read = attn_bytes + expert_bytes * touched(experts, k, batch)
    mem = batch / (read / (g * HBM_BW))
    comp = g * FLOPS / (2 * active)
    return min(mem, comp), g, ("memory" if mem < comp else "compute")


print(f"{'configuration':>20}{'batch':>8}{'experts touched':>18}{'GPUs':>7}"
      f"{'tokens/s':>13}{'bound by':>11}{'$ / 1M tokens':>16}")
print("-" * 93)
thr, cost = {}, {}
for name, e, k in CONFIGS:
    total, active, neff, l, gb, gpus, _, _ = rows[name]
    attn_bytes = LAYERS * ATTN_PER_LAYER * BYTES
    expert_bytes = LAYERS * e * FFN_PER_LAYER * BYTES
    for batch in (1, 32, 512):
        tps, g, bound = serve(gb, attn_bytes, expert_bytes, active, e, k, batch)
        c = g * GPU_HOUR / 3600 / tps * 1e6
        thr[(name, batch)] = tps
        cost[(name, batch)] = c
        print(f"{name:>20}{batch:>8}{touched(e, k, batch):>18.1%}{g:>7}"
              f"{tps:>13.0f}{bound:>11}{c:>16.3f}")
    print()

print()
print("Against the dense model that matches each one's quality.")
print()
print(f"{'configuration':>20}{'loss':>9}{'sparse $ / 1M':>16}"
      f"{'matched dense $ / 1M':>23}{'serving penalty':>18}")
print("-" * 86)
BATCH = 512
penalty = {}
for name, e, k in CONFIGS:
    total, active, neff, l, gb, gpus, _, _ = rows[name]
    dn = match[name][0]
    dgb = dn * BYTES / 1e9
    dtps, dg, _ = serve(dgb, dn * BYTES, 0.0, dn, 1, 1, BATCH)
    dcost = dg * GPU_HOUR / 3600 / dtps * 1e6
    penalty[name] = cost[(name, BATCH)] / dcost
    print(f"{name:>20}{l:>9.3f}{cost[(name, BATCH)]:>16.3f}"
          f"{dcost:>23.3f}{cost[(name, BATCH)] / dcost:>17.2f}x")

print()
print(f"at batch {BATCH}, `{BEST}` serves at {penalty[BEST]:.1f}x the cost of the")
print("dense model it matches, and trains at a fraction of it")

print()
print()
print("So there is a break-even, and it is measured in tokens served.")
print()
print(f"{'configuration':>20}{'training saved ($)':>21}"
      f"{'extra serving $ / 1M tok':>27}{'break-even tokens served':>27}")
print("-" * 95)
breakeven = {}
FLOP_COST = GPU_HOUR / 3600 / FLOPS      # dollars per training FLOP, roughly
for name, e, k in CONFIGS:
    if name == "dense":
        continue
    saved = (match[name][2] - match[name][1]) * FLOP_COST
    dn = match[name][0]
    dgb = dn * BYTES / 1e9
    dtps, dg, _ = serve(dgb, dn * BYTES, 0.0, dn, 1, 1, BATCH)
    dcost = dg * GPU_HOUR / 3600 / dtps * 1e6
    extra = cost[(name, BATCH)] - dcost
    be = saved / extra * 1e6 if extra > 0 else float("inf")
    breakeven[name] = be
    print(f"{name:>20}{saved:>21,.0f}{extra:>27.3f}{be:>27.3e}")

SAVED_BEST = (match[BEST][2] - match[BEST][1]) * FLOP_COST
EXTRA_BEST = SAVED_BEST / breakeven[BEST] * 1e6
print()
print("Below the break-even the sparse model is the cheaper system; above it,")
print("the dense one is.")

print(f"""
The first table is the decoupling that makes sparsity interesting. `{BEST}` reaches a loss of
{rows[BEST][3]:.3f} against the dense baseline's {DENSE[3]:.3f}, using
{rows[BEST][1] / DENSE[1]:.1f} times the FLOPs per token and
**{rows[BEST][0] / DENSE[0]:.1f} times the weights** -- {rows[BEST][4]:.0f} GB against
{DENSE[4]:.0f}, which is {rows[BEST][5]} accelerators against {DENSE[5]}.

The model credits a sparse parameter at {RHO:.2f} of a dense one in log space. That number is
the whole argument's sensitivity and it is not measured here; the structure below is what
transfers.

The matched-quality table is the comparison that should be made and usually is not. Rather than
asking "is this better than a dense model with the same FLOPs", ask what dense model reaches the
same loss. For `{BEST}` that is {match[BEST][0]:.2e} parameters, and the sparse model trains
**{match[BEST][2] / match[BEST][1]:.1f} times cheaper** than it.

**That is the real result, and it is a training result.**

The serving table is the other half (eq:sparsity-moves-the-bottleneck-from-flops-to-memory).
Every configuration is memory-bound at small batch, because a generation step reads weights and
does almost nothing per byte read. A dense model amortises one weight read across the whole
batch. **An MoE reads an expert if any token in the batch routes to it** -- so at batch 1 a
`k = 4` model reads four experts to produce one token, and at batch {BATCH} it reads all of
them.

Expert reuse is therefore a function of batch size, which is a function of concurrent traffic,
which is a product fact rather than a modelling one.

The penalty table puts the two halves together. At batch {BATCH}, `{BEST}` costs
**{penalty[BEST]:.1f} times** the dense model that matches its quality, per served token, while
having cost {match[BEST][2] / match[BEST][1]:.1f} times less to train.

**Sparsity does not make the system cheaper. It moves the cost from training to serving**
(eq:sparsity-trades-training-cost-for-serving-cost), and whether that is a good trade is
arithmetic rather than architecture.

The break-even table does that arithmetic. For `{BEST}` the training saving is worth about
${SAVED_BEST:,.0f} and the serving penalty is {EXTRA_BEST:.3f} per million tokens, which break
even at **{breakeven[BEST]:.2e} tokens served**.

Below that the sparse model is the cheaper system outright; above it the dense one is, and the
gap grows with every served token. Two trillion tokens is a real quantity -- a mid-sized product
reaches it in months and a research artefact never does -- which is exactly why the same
architecture is the right answer for one team and the wrong one for another.

Which reproduces ch:res-scaling's result in a second variable. That chapter found the
training-optimal model is not the deployment-optimal one and that the gap widens with serving
volume. This one finds the same for sparsity, from a completely different mechanism -- bandwidth
rather than FLOPs. **Two independent analyses, one conclusion: the serving forecast is a model
architecture decision**, and it is usually made by people who are not in the room.

None of this touches whether the routing works at all, which is the second listing's problem and
a different kind of failure.""")
