# -*- coding: utf-8 -*-
# Extracted from: Chapter 132 — QLoRA, PEFT, and Adapters
# Source: src/.../ch132-qlora-peft.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""What LoRA actually saves, and what it does not.

The headline for cite:hu2021lora is a parameter count -- "0.01% of the
parameters are trainable" -- and it is true and misleading, because trainable
parameters are not the resource that binds.

Three different resources are at stake and they behave differently
(eq:training-memory). Optimiser state scales with TRAINABLE parameters, so LoRA
removes nearly all of it. Frozen weights scale with TOTAL parameters, so LoRA
removes none of it and cite:dettmers2023qlora attacks that instead. Activations
scale with batch and sequence length, and NEITHER method touches them.

Compute is the fourth, and the one people get wrong: the backward pass still runs
through every frozen layer, because the adapters need gradients that arrive from
above (eq:backward-still-full).
"""
BYTES = {"fp32": 4, "bf16": 2, "int4": 0.5}


def full_ft(P):
    """Mixed-precision Adam: bf16 weights, fp32 master, fp32 grads, m and v."""
    return {"weights": 2 * P, "master": 4 * P, "grads": 4 * P, "adam": 8 * P}


def lora(P, P_a):
    """Base frozen in bf16; only adapters get master weights, grads, Adam."""
    return {"weights": 2 * P, "master": 4 * P_a, "grads": 4 * P_a,
            "adam": 8 * P_a}


def qlora(P, P_a):
    """Base frozen at 4 bits (part:15 owns how); adapters unchanged."""
    return {"weights": 0.5 * P, "master": 4 * P_a, "grads": 4 * P_a,
            "adam": 8 * P_a}


def gb(x):
    return x / 1e9


MODELS = [("7B", 7e9), ("13B", 13e9), ("70B", 70e9)]
RANK_FRAC = 0.001          # adapters ~0.1% of parameters, a typical r=16 config

print("Memory for the MODEL STATE alone (activations excluded, they come next).\n")
print(f"{'model':>7}{'method':>9}" + "".join(f"{c:>10}" for c in
      ("weights", "master", "grads", "adam", "TOTAL")) + f"{'vs full':>10}")
print("-" * 76)

for name, P in MODELS:
    P_a = P * RANK_FRAC
    base = sum(full_ft(P).values())
    for label, d in (("full", full_ft(P)), ("LoRA", lora(P, P_a)),
                     ("QLoRA", qlora(P, P_a))):
        t = sum(d.values())
        print(f"{name:>7}{label:>9}"
              + "".join(f"{gb(d[c]):>10.1f}" for c in
                        ("weights", "master", "grads", "adam"))
              + f"{gb(t):>10.1f}{t / base:>10.2f}x")
    print()

print("\nActivation memory, which none of the above changes.\n")
print(f"{'model':>7}{'batch x seq':>14}{'no checkpoint':>16}"
      f"{'checkpointed':>15}")
print("-" * 54)
# Rough transformer activation model: ~ layers * tokens * hidden * 2 bytes * k
SHAPES = {"7B": (32, 4096), "13B": (40, 5120), "70B": (80, 8192)}
K = 34          # tensors kept per layer per token, order of magnitude
for name, P in MODELS:
    L, H = SHAPES[name]
    for tokens in (8 * 2048,):
        a = L * tokens * H * 2 * K
        ck = (L ** 0.5) * tokens * H * 2 * K
        print(f"{name:>7}{'8 x 2048':>14}{gb(a):>16.1f}{gb(ck):>15.1f}")

print("\n\nCompute per token, in units of P FLOPs.\n")
print(f"{'method':>10}{'forward':>10}{'grad wrt':>11}{'grad wrt':>11}"
      f"{'TOTAL':>9}{'saving':>9}")
print(f"{'':>10}{'':>10}{'inputs':>11}{'weights':>11}{'':>9}{'':>9}")
print("-" * 60)
print(f"{'full':>10}{2:>10}{2:>11}{2:>11}{6:>9}{'--':>9}")
print(f"{'LoRA':>10}{2:>10}{2:>11}{'~0':>11}{4:>9}{1 - 4/6:>9.0%}")
print(f"{'QLoRA':>10}{2:>10}{2:>11}{'~0':>11}{4:>9}{1 - 4/6:>9.0%}")

p7 = 7e9
tot_full = sum(full_ft(p7).values())
tot_lora = sum(lora(p7, p7 * RANK_FRAC).values())
tot_q = sum(qlora(p7, p7 * RANK_FRAC).values())
print(f"""
Take the 7B rows first. Full fine-tuning needs {gb(tot_full):.0f} GB of model
state before a single activation is stored, and look at where it goes: the
weights are {gb(2*p7):.0f} GB and everything the OPTIMISER needs is
{gb(16*p7):.0f} GB. Adam's two moments, the fp32 master copy and the gradients
are {16/18:.0%} of the bill, and the model itself is the small term
(eq:training-memory).

That is what LoRA removes. Optimiser state scales with TRAINABLE parameters, and
at {RANK_FRAC:.1%} of parameters it effectively vanishes: {gb(tot_lora):.0f} GB
total, {tot_lora/tot_full:.2f} of full fine-tuning. But notice what is left. The
frozen weights are untouched, because you still have to hold the model.

So LoRA converts a problem dominated by optimiser state into one dominated by
frozen weights -- and then cite:dettmers2023qlora attacks exactly that residual,
storing the frozen base at 4 bits instead of 16. {gb(tot_q):.0f} GB, or
{tot_q/tot_full:.2f} of full fine-tuning.

The two savings are independent and they compose, which is the point that gets
lost when QLoRA is described as "LoRA but smaller". They target different terms
of eq:training-memory: LoRA removes optimiser state, quantization removes weight
storage. Neither would be sufficient alone at 70B, and together they put it on
one 80 GB card.

Now the second table, which is where the arithmetic stops being encouraging.
Activation memory depends on batch size, sequence length, and depth -- not on how
many parameters are trainable. Nothing in this chapter reduces it. At 70B with a
modest batch it is larger than the quantized weights, so a QLoRA run that fits on
paper still fails at the first backward pass unless activations are checkpointed,
and checkpointing costs an extra forward pass.

This is why "QLoRA lets you fine-tune 70B on a 24 GB card" comes with a sequence
length in the fine print. The weights fit. Whether the run fits is a question
about activations.

The third table is the one that most often surprises people. Trainable parameters
drop by {1-RANK_FRAC:.1%}, and compute drops by 33%.

The reason is eq:backward-still-full. Gradients for the adapters in layer 3 arrive
from layer 40, which means the backward pass must propagate through every frozen
layer in between. What LoRA skips is only the weight-gradient computation --
roughly a third of the total -- and never the input-gradient chain.

So the honest summary is: LoRA and QLoRA are MEMORY techniques that make training
possible on hardware where it was impossible. They are not, and were never, speed
techniques. A run that took ten hours full fine-tuning takes about seven, not ten
minutes, and budgeting on the parameter ratio produces schedules that are wrong
by two orders of magnitude.""")
