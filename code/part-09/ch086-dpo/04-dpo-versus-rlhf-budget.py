# Extracted from: Chapter 86 — Preference Optimization: DPO and Its Descendants
# Source: src/.../ch086-dpo.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""What each method costs on the hardware a small team actually has."""

PARAMS = 7e9
BYTES_PER_PARAM_BF16 = 2
OPTIMIZER_BYTES_PER_PARAM = 12       # Adam fp32 moments + master weights
GPU_MEMORY_GB = 80
N_PAIRS = 12_000
AVG_TOKENS = 512

weights_gb = PARAMS * BYTES_PER_PARAM_BF16 / 1e9
trainable_gb = weights_gb + PARAMS * OPTIMIZER_BYTES_PER_PARAM / 1e9

# LoRA (Part XIV) trains a small adapter, so the optimiser state is tiny and
# the base weights stay frozen. Included because it is what actually makes any
# of this fit on one device.
LORA_FRACTION = 0.005

CONFIGS = {
    "RLHF (PPO)": dict(trainable=1, frozen=3, generates=True, lora=False),
    "DPO": dict(trainable=1, frozen=1, generates=False, lora=False),
    "DPO, reference cached": dict(trainable=1, frozen=0, generates=False, lora=False),
    "DPO + LoRA, ref cached": dict(trainable=1, frozen=0, generates=False, lora=True),
}

print(f"{PARAMS / 1e9:.0f}B parameters, bf16 weights, Adam optimiser")
print(f"  weights          : {weights_gb:.1f} GB")
print(f"  trainable copy   : {trainable_gb:.1f} GB (weights + optimiser state)\n")

print(f"{'configuration':<24} {'models':>8} {'memory GB':>11} "
      f"{'fits 80GB':>11} {'generates':>11}")
for name, c in CONFIGS.items():
    if c["lora"]:
        # Base weights frozen; optimiser state only for the adapter.
        mem = weights_gb + PARAMS * LORA_FRACTION * (
            BYTES_PER_PARAM_BF16 + OPTIMIZER_BYTES_PER_PARAM) / 1e9
    else:
        mem = c["trainable"] * trainable_gb + c["frozen"] * weights_gb
    n_models = c["trainable"] + c["frozen"]
    print(f"{name:<24} {n_models:>8} {mem:>11.1f} "
          f"{str(mem < GPU_MEMORY_GB):>11} {str(c['generates']):>11}")

# Training cost: forward+backward is ~6ND; a frozen forward pass is ~2ND.
fwd_bwd = 6 * PARAMS
fwd_only = 2 * PARAMS
tokens = N_PAIRS * AVG_TOKENS * 2          # two responses per pair

dpo_flops = tokens * (fwd_bwd + fwd_only)          # policy trained, ref frozen
dpo_cached = tokens * fwd_bwd                      # reference precomputed once
# PPO additionally generates samples (sequential, memory-bound) and runs a
# reward model and a value model over them.
ppo_flops = tokens * (fwd_bwd * 2 + fwd_only * 2) * 4   # x4 for PPO epochs

print(f"\n{'method':<24} {'train FLOPs':>14} {'relative':>10}")
for name, f in [("DPO", dpo_flops), ("DPO, reference cached", dpo_cached),
                ("RLHF (PPO)", ppo_flops)]:
    print(f"{name:<24} {f:>14.2e} {f / dpo_flops:>9.1f}x")

print(f"""
Read the fits-80GB column honestly: FULL fine-tuning of a {PARAMS / 1e9:.0f}B
model does not fit on one 80 GB device under any method, because the Adam state
alone is {PARAMS * OPTIMIZER_BYTES_PER_PARAM / 1e9:.0f} GB. That is the 16N
accounting from ch:tf-complexity, and it is why Part XIV exists.

What the memory column does decide is how many devices. RLHF needs
{140.0 / weights_gb:.0f} model-copies' worth at {140.0:.0f} GB; DPO needs
{112.0:.0f} GB; caching the reference log-probabilities removes the frozen model
and brings it to {98.0:.0f} GB. Only the last row — LoRA on top of a cached
reference — actually fits on one device, and it is the configuration most small
teams really run.

The FLOP column understates the gap, because PPO's cost is dominated by
GENERATION inside the training loop, which is sequential and memory-bandwidth
bound (ch:tf-masking-kv) rather than compute bound. Wall-clock differs by more
than the arithmetic shows.

None of this says DPO produces a better model. It says DPO is a training script
and RLHF is an infrastructure project — which is why the open-weight ecosystem
aligned on DPO regardless of how the quality question comes out.""")
