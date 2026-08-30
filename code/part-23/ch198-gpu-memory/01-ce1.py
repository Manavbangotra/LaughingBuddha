# -*- coding: utf-8 -*-
# Extracted from: Chapter 198 — GPU Memory, CUDA, and the Roofline Model
# Source: src/.../ch198-gpu-memory.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""One roofline is not enough: an operation can be starved at one memory tier while
having ample headroom at another.

ch:inf-cpu-gpu used a single balance point -- peak FLOP/s over HBM bandwidth. Real
devices have a hierarchy, and each level has its own bandwidth and its own balance
point. An operation is bound by whichever tier it is worst against
(eq:roofline-has-multiple-ridges).

That matters because the standard fix for one tier is useless for another, and
because the biggest single optimisation in transformer serving -- cite:dao2022flash --
is precisely a move of traffic from one tier to the next one down.

This listing builds the multi-tier roofline and measures where attention sits before
and after tiling.
"""
# Memory tiers on a current datacentre GPU.
# (tier, bandwidth bytes/s, capacity bytes)
TIERS = [
    ("registers",     2.20e14,  2.6e8),
    ("shared / L1",   1.28e14,  2.3e8),
    ("L2 cache",      1.10e13,  5.0e7),
    ("HBM",           3.35e12,  8.0e10),
    ("NVLink peer",   9.00e11,  0.0),
    ("PCIe host",     6.40e10,  0.0),
]
PEAK = 9.89e14           # dense bf16 FLOP/s

print("Memory tiers, each with its own balance point: the arithmetic intensity")
print("needed to keep the arithmetic units fed FROM THAT TIER.")
print()
print(f"{'tier':>16}{'bandwidth GB/s':>18}{'balance point':>16}"
      f"{'easier than HBM':>18}")
print("-" * 62)
bal = {}
hbm = None
for name, bw, cap in TIERS:
    b = PEAK / bw
    bal[name] = b
    if name == "HBM":
        hbm = b
for name, bw, cap in TIERS:
    print(f"{name:>16}{bw / 1e9:>18.0f}{bal[name]:>13.0f} F/B"
          f"{bal['HBM'] / bal[name]:>17.1f}x")

print()
print("A tier with more bandwidth needs LESS intensity to saturate. So an operation")
print("that is memory-bound against HBM may be compute-bound against shared memory,")
print("if you can arrange for it to read from there.")

print()
print()
print("Attention during prefill, the operation cite:dao2022flash addresses.")
print("Sequence length n, head dimension d.")
print()
D_HEAD = 128
BYTES = 2.0
SEQS = [512, 2048, 8192, 32768]


def naive_attention(n, d):
    """Traffic if the n-by-n score matrix is materialised in HBM.

    Write S = QK^T, read it back for softmax, write P, read it back for PV.
    """
    qkv = 3.0 * n * d * BYTES
    scores = 4.0 * n * n * BYTES          # two writes and two reads of n-by-n
    out = n * d * BYTES
    return qkv + scores + out


def tiled_attention(n, d):
    """Traffic if scores never leave on-chip memory (cite:dao2022flash).

    Q, K, V are read from HBM; the n-by-n intermediate lives in SRAM.
    """
    return 3.0 * n * d * BYTES + n * d * BYTES


def attention_flops(n, d):
    """QK^T then PV: two n-by-n-by-d matmuls."""
    return 2.0 * 2.0 * n * n * d


print(f"{'seq len':>10}{'naive HBM MB':>15}{'tiled HBM MB':>15}"
      f"{'reduction':>12}{'GFLOP':>10}")
print("-" * 62)
att = {}
for n in SEQS:
    nv = naive_attention(n, D_HEAD)
    tl = tiled_attention(n, D_HEAD)
    fl = attention_flops(n, D_HEAD)
    att[n] = (nv, tl, fl)
    print(f"{n:>10}{nv / 1e6:>15.1f}{tl / 1e6:>15.1f}{nv / tl:>11.1f}x"
          f"{fl / 1e9:>10.1f}")

print()
print()
print("Arithmetic intensity against HBM, before and after tiling, with the")
print("HBM balance point of %.0f for reference." % bal["HBM"])
print()
print(f"{'seq len':>10}{'naive I':>12}{'tiled I':>12}{'naive bound':>14}"
      f"{'tiled bound':>14}")
print("-" * 62)
inten = {}
for n in SEQS:
    nv, tl, fl = att[n]
    i_nv = fl / nv
    i_tl = fl / tl
    inten[n] = (i_nv, i_tl)
    print(f"{n:>10}{i_nv:>12.1f}{i_tl:>12.1f}"
          f"{('memory' if i_nv < bal['HBM'] else 'compute'):>14}"
          f"{('memory' if i_tl < bal['HBM'] else 'compute'):>14}")

print()
print()
print("What that does to time. A memory-bound op takes traffic/bandwidth; a")
print("compute-bound one takes FLOPs/peak.")
print()
print(f"{'seq len':>10}{'naive ms':>12}{'tiled ms':>12}{'speedup':>11}"
      f"{'tiled % of peak':>18}")
print("-" * 63)
times = {}
for n in SEQS:
    nv, tl, fl = att[n]
    t_nv = max(nv / TIERS[3][1], fl / PEAK)
    t_tl = max(tl / TIERS[3][1], fl / PEAK)
    times[n] = (t_nv, t_tl)
    print(f"{n:>10}{t_nv * 1000:>12.3f}{t_tl * 1000:>12.3f}"
          f"{t_nv / t_tl:>10.1f}x{fl / t_tl / PEAK:>18.1%}")

print()
print()
print("But tiling only works if a tile fits on-chip. Shared memory per streaming")
print("multiprocessor bounds the tile, which bounds how much can stay resident.")
print()
SMEM = 228.0 * 1024      # bytes of shared memory per SM
print(f"shared memory per SM: {SMEM / 1024:.0f} KB")
print()
print(f"{'tile rows':>11}{'tile bytes':>13}{'fits':>8}{'HBM passes over K,V':>22}")
print("-" * 56)
fits = {}
for br in (16, 32, 64, 128, 256, 512):
    # A tile holds Q rows, K rows, V rows and the score block.
    tile = (3.0 * br * D_HEAD + br * br) * BYTES
    ok = tile <= SMEM
    fits[br] = (tile, ok)
    passes = 1 if ok else 0
    print(f"{br:>11}{tile / 1024:>11.1f}K{('yes' if ok else 'no'):>8}"
          f"{(str(1) if ok else 'spills'):>22}")

print()
print()
print("And the tier that is easy to forget. Moving a model's weights across each")
print("link, for a 14 GB model:")
print()
WEIGHTS = 14.0e9
print(f"{'link':>16}{'bandwidth GB/s':>18}{'time to move 14 GB':>22}")
print("-" * 58)
move = {}
for name, bw, cap in TIERS:
    if name in ("registers", "shared / L1", "L2 cache"):
        continue
    t = WEIGHTS / bw
    move[name] = t
    print(f"{name:>16}{bw / 1e9:>18.0f}{t:>20.2f}s")

print(f"""
The tier table is the correction to ch:inf-cpu-gpu's single ridge. HBM needs
{bal['HBM']:.0f} operations per byte to keep the arithmetic units fed. Shared memory
needs {bal['shared / L1']:.0f} -- {bal['HBM'] / bal['shared / L1']:.0f} times less --
because it delivers {TIERS[1][1] / TIERS[3][1]:.0f} times the bandwidth
(eq:roofline-has-multiple-ridges).

**An operation that is hopelessly memory-bound against HBM can be comfortably
compute-bound against shared memory**, and the entire engineering question is whether
its working set can be arranged to live there.

The attention tables are that question answered for the operation where it matters
most. Naive attention at sequence length {SEQS[2]} moves
{att[SEQS[2]][0] / 1e6:.1f} MB through HBM, of which almost all is the
{SEQS[2]}-by-{SEQS[2]} score matrix written out and read back twice. Tiled attention
moves {att[SEQS[2]][1] / 1e6:.1f} MB -- a reduction of
{att[SEQS[2]][0] / att[SEQS[2]][1]:.0f} times -- because the score matrix never leaves
the chip.

Note what did NOT change: the FLOPs. Both do {att[SEQS[2]][2] / 1e9:.1f} GFLOP, and
cite:dao2022flash computes the same exact result. **The speedup is entirely a change
in which tier the traffic goes to**, which is why it required no accuracy trade-off
and why adoption was immediate.

The intensity table shows the regime change. Naive attention at {SEQS[2]} has HBM
intensity {inten[SEQS[2]][0]:.1f}, which is below the HBM balance point of
{bal['HBM']:.0f} -- memory-bound. Tiled has {inten[SEQS[2]][1]:.1f}, far above it --
compute-bound, running at {att[SEQS[2]][2] / times[SEQS[2]][1] / PEAK:.1%} of peak
arithmetic.

**That is what "moving an operation across the ridge" means in practice**, and it is
worth {times[SEQS[2]][0] / times[SEQS[2]][1]:.1f}x here.

Notice that the speedup does NOT keep growing: it is
{times[SEQS[1]][0] / times[SEQS[1]][1]:.1f}x at {SEQS[1]} and
{times[SEQS[-1]][0] / times[SEQS[-1]][1]:.1f}x at {SEQS[-1]}, essentially flat. That
is because tiling has already done everything it can -- the tiled column reaches
{att[SEQS[-1]][2] / times[SEQS[-1]][1] / PEAK:.0%} of peak arithmetic, and there is
nowhere left to go.

**A tier-crossing optimisation has a hard ceiling: peak.** Once an operation is
compute-bound, further reductions in memory traffic buy exactly nothing, which is
worth knowing before spending a quarter on the next one. The remaining lever at that
point is arithmetic -- fewer FLOPs, or cheaper ones -- and that is a different
project entirely.

The tile table is the constraint that makes this an engineering problem rather than a
free win. Shared memory is {SMEM / 1024:.0f} KB per streaming multiprocessor, so a
tile of {[b for b in fits if fits[b][1]][-1]} rows fits and
{[b for b in fits if not fits[b][1]][0]} rows does not. **The tier with the good
balance point is the tier with almost no capacity**, and that trade -- bandwidth
against capacity -- is what the whole hierarchy is.

The last table is the tier people forget until it ruins a design. Moving
{WEIGHTS / 1e9:.0f} GB of weights takes {move['HBM']:.2f}s from HBM,
{move['NVLink peer']:.2f}s across NVLink, and {move['PCIe host']:.2f}s across PCIe.

That last figure is why **model loading, host offload, and cross-node weight movement
are architectural decisions rather than implementation details**. A design that swaps
models per request pays {move['PCIe host']:.2f} seconds every time, which is
{move['PCIe host'] / (14.0e9 / TIERS[3][1]):.0f} times the cost of simply reading the
weights it already has resident -- and ch:inf-kubernetes has to plan capacity around
it.""")
