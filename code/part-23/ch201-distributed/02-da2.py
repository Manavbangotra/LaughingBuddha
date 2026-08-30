# -*- coding: utf-8 -*-
# Extracted from: Chapter 201 — Distributed and Disaggregated Inference
# Source: src/.../ch201-distributed.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Model parallelism multiplies the failure rate, and replication has to pay for it.

A tensor-parallel group is one machine as far as failure is concerned: lose any device
and the whole group stops, because the layer's matrices are split across all of them.

So splitting a model across n devices multiplies its failure rate by n, and the
availability of the group is the device availability raised to the n
(eq:parallel-group-is-one-failure-domain).

That is the same product that has appeared throughout this book -- ch:ag-loop's chain,
ch:sd-retrieval-agents's fan-out -- arriving as a hardware reliability question. This
listing measures what it costs and what replication has to do about it.
"""
import math

MTBF_HOURS = 42000.0        # mean time between failures for one device
REPAIR_HOURS = 3.5          # time to detect, drain, and replace
DEVICE_AVAIL = MTBF_HOURS / (MTBF_HOURS + REPAIR_HOURS)
DEGREES = [1, 2, 4, 8, 16, 32, 64]
TARGET = 0.99999


def group_avail(n):
    return DEVICE_AVAIL ** n


def group_mtbf(n):
    return MTBF_HOURS / n


print("One device: MTBF %.0f hours, repair %.1f hours, availability %.5f."
      % (MTBF_HOURS, REPAIR_HOURS, DEVICE_AVAIL))
print()
print("A tensor-parallel group fails when ANY member fails.")
print()
print(f"{'degree':>8}{'group MTBF hrs':>17}{'group availability':>21}"
      f"{'downtime hrs/yr':>18}{'vs 1 device':>14}")
print("-" * 80)
tab = {}
for n in DEGREES:
    a = group_avail(n)
    down = (1.0 - a) * 8760.0
    tab[n] = (group_mtbf(n), a, down)
    print(f"{n:>8}{group_mtbf(n):>17.0f}{a:>21.5f}{down:>18.1f}"
          f"{down / tab[1][2]:>13.1f}x")

print()
print()
print("Downtime per year against replica count. This is the continuous form of")
print("the redundancy question -- integer replicas make the threshold coarse.")
print()
print(f"{'degree':>8}" + "".join(f"{('r=%d' % r):>16}" for r in (1, 2, 3))
      + f"{'r=2 vs degree 1':>18}")
print("-" * 74)
down = {}
for n in DEGREES:
    av = group_avail(n)
    row = [(1.0 - av) ** r * 8760.0 for r in (1, 2, 3)]
    down[n] = row
    print(f"{n:>8}" + "".join(f"{v:>15.4f}h" for v in row)
          + f"{row[1] / down[1][1]:>17.0f}x")

print()
print()
print("Replicas needed to hold a %.3f%% target." % (TARGET * 100))
print()
print(f"{'degree':>8}{'group avail':>14}{'replicas':>11}"
      f"{'devices total':>16}{'redundancy':>13}")
print("-" * 64)
need = {}
for n in DEGREES:
    av = group_avail(n)
    r = 1
    while 1.0 - (1.0 - av) ** r < TARGET and r < 40:
        r += 1
    need[n] = (r, r * n)
    print(f"{n:>8}{av:>14.5f}{r:>11}{r * n:>16}"
          f"{(r - 1) / float(r):>12.0%}")

print()
print()
print("How repair time moves it. Faster replacement is the cheapest lever,")
print("because it acts on every degree at once.")
print()
print(f"{'repair hours':>14}" + "".join(f"{('d=%d' % n):>10}" for n in
                                        (4, 8, 16, 32, 64)))
print("-" * 66)
rep = {}
for rh in (0.5, 1.0, 3.5, 8.0, 24.0):
    av = MTBF_HOURS / (MTBF_HOURS + rh)
    row = []
    for n in (4, 8, 16, 32, 64):
        a = av ** n
        r = 1
        while 1.0 - (1.0 - a) ** r < TARGET and r < 40:
            r += 1
        row.append(r)
    rep[rh] = row
    print(f"{rh:>14.1f}" + "".join(f"{v:>10}" for v in row))
print()
print("(replicas needed to hold the target)")

print()
print()
print("And the part that is easy to miss: a partial failure does not merely stop")
print("the group, it drops every in-flight sequence's KV cache.")
print()
BATCH = 32
MEAN_DONE = 190          # tokens already generated when the failure lands
STEP_MS = 4.18
print(f"{'degree':>8}{'failures/yr':>14}{'sequences lost/yr':>20}"
      f"{'tokens recomputed/yr':>23}")
print("-" * 68)
loss = {}
for n in DEGREES:
    f_per_yr = 8760.0 / group_mtbf(n)
    seqs = f_per_yr * BATCH
    toks = seqs * MEAN_DONE
    loss[n] = (f_per_yr, seqs, toks)
    print(f"{n:>8}{f_per_yr:>14.2f}{seqs:>20.0f}{toks:>23.0f}")

print()
print()
print("Pipeline parallelism has the same exposure per group but smaller groups")
print("are possible, because stages can be replicated independently.")
print()
print(f"{'topology':>34}{'failure domain':>17}{'availability':>15}"
      f"{'downtime at r=2':>18}")
print("-" * 84)
TOPO = [
    ("16-way tensor, one group",        16),
    ("8-way tensor x 2-way pipeline",    8),
    ("4-way tensor x 4-way pipeline",    4),
    ("2-way tensor x 8-way pipeline",    2),
]
for label, dom in TOPO:
    a = group_avail(dom)
    print(f"{label:>34}{dom:>17}{a:>15.5f}"
          f"{(1.0 - a) ** 2 * 8760.0:>17.4f}h")

print(f"""
The first table is the multiplication. A device with a {MTBF_HOURS:.0f}-hour MTBF and
{DEVICE_AVAIL:.5f} availability, put into a {16}-way tensor-parallel group, gives that
group an availability of {tab[16][1]:.5f} and
{tab[16][2]:.1f} hours of downtime a year against a single device's
{tab[1][2]:.1f} (eq:parallel-group-is-one-failure-domain).

**Sixteen times the downtime, because the group fails if any member does.** That is
ch:ag-loop's chain and ch:sd-retrieval-agents's fan-out in a rack: a product of
per-component reliabilities, and the exponent is the parallelism degree.

The downtime table prices it continuously. With {2} replicas, a degree-{1} service is
down {down[1][1]:.4f} hours a year and a degree-{16} service is down
{down[16][1]:.4f} -- **{down[16][1] / down[1][1]:.0f} times more**, because squaring a
worse number leaves a worse number.

The replica table converts that into hardware. Holding a {TARGET:.3%} target needs
{need[1][0]} replicas at degree 1 and {need[64][0]} at degree {64}, which is
{need[64][1]} devices for {64} devices of capacity --
{(need[64][0] - 1) / float(need[64][0]):.0%} redundancy.

**Model parallelism does not merely cost communication; it costs redundancy**, and the
redundancy cost is the one that does not appear in any throughput benchmark.
ch:inf-parallelism's speedup tables are all computed on a group that is assumed to
work.

The repair table is the cheapest available lever, and it is worth noticing that it is
not a hardware lever. At degree {64}, cutting repair time from {3.5:.1f} hours to
{0.5:.1f} takes the replicas needed from {rep[3.5][4]} to {rep[0.5][4]}; at
{24.0:.0f} hours it would take {rep[24.0][4]}. At degree {32} the same cut takes it
from {rep[8.0][3]} to {rep[0.5][3]}.

**Detection and replacement speed buys redundancy across every degree at once**, which
makes it the highest-leverage reliability investment in a model-parallel fleet --
automated draining, hot spares, and fast health detection rather than more hardware.

The in-flight table is the cost nobody budgets for. A failure does not merely stop the
group; it destroys the KV cache of every sequence in flight. At degree {16} that is
{loss[16][0]:.2f} failures a year, each dropping {BATCH} sequences that have already
generated {MEAN_DONE} tokens -- {loss[16][2]:.0f} tokens a year recomputed, and
{loss[16][1]:.0f} user-visible request failures unless something replays them.

**This is where ch:as-state-machines's replay requirement becomes a hardware
constraint.** A serving system without request-level replay converts every hardware
failure into a batch of user-visible errors, and the batch size is the batch size.

The topology table is the design response. A {16}-way tensor group has availability
{group_avail(16):.5f} and, duplicated, {(1 - group_avail(16)) ** 2 * 8760.0:.4f} hours
of annual downtime. Splitting the same sixteen devices as {4}-way tensor by {4}-way
pipeline gives each failure domain availability {group_avail(4):.5f} and
{(1 - group_avail(4)) ** 2 * 8760.0:.4f} hours --
**{((1 - group_avail(16)) ** 2) / ((1 - group_avail(4)) ** 2):.0f} times better for the
same device count**.

That is a real argument for pipeline parallelism that ch:inf-parallelism's
communication analysis does not contain: **smaller failure domains**. A pipeline stage
can be replaced independently while tensor-parallel ranks cannot, so the topology
decision has a reliability term alongside the bandwidth one -- and on a large fleet the
reliability term is frequently the larger.""")
