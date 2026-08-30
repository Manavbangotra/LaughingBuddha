# -*- coding: utf-8 -*-
# Extracted from: Chapter 200 — Parallelism: Tensor, Pipeline, Data, and Expert
# Source: src/.../ch200-parallelism.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

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
