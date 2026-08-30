# -*- coding: utf-8 -*-
# Extracted from: Chapter 197 — CPU and GPU Inference Fundamentals
# Source: src/.../ch197-cpu-gpu.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

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
