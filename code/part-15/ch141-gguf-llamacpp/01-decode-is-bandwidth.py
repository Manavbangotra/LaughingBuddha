# -*- coding: utf-8 -*-
# Extracted from: Chapter 141 — GGUF, llama.cpp, and Weight-Only Quantization
# Source: src/.../ch141-gguf-llamacpp.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

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
