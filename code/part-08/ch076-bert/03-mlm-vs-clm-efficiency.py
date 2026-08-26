# -*- coding: utf-8 -*-
# Extracted from: Chapter 76 — BERT, RoBERTa, and Masked Language Modeling
# Source: src/.../ch076-bert.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Same compute, different amounts of supervision."""

BATCH, SEQ_LEN, MASK_RATE, STEPS = 256, 512, 0.15, 1_000_000
VOCAB = 30_522
PARAMS = 110e6

tokens_per_step = BATCH * SEQ_LEN
mlm_targets = MASK_RATE * tokens_per_step
clm_targets = tokens_per_step

# Compute is 6ND per token for training (see ch:tf-complexity) and is the same
# for both objectives — the masking changes the loss, not the forward pass.
flops_per_step = 6 * PARAMS * tokens_per_step

print(f"{'objective':<12} {'targets/step':>13} {'targets total':>16} "
      f"{'PFLOPs total':>14} {'targets/PFLOP':>15}")
for name, targets in [("MLM (15%)", mlm_targets), ("causal LM", clm_targets)]:
    total = targets * STEPS
    pflops = flops_per_step * STEPS / 1e15
    print(f"{name:<12} {targets:>13,.0f} {total:>16.3e} "
          f"{pflops:>14,.0f} {total / pflops:>15,.0f}")

print(f"\nsupervision ratio: {clm_targets / mlm_targets:.2f}x in favour of causal LM")
print(f"equation (eq:mlm-sample-efficiency) predicts 1/{MASK_RATE:.2f} = "
      f"{1 / MASK_RATE:.2f}x — the same number")

# What ELECTRA changes: score every position, not 15% of them.
print(f"\nELECTRA scores all {tokens_per_step:,} positions per step rather than "
      f"{mlm_targets:,.0f} — which is exactly this ratio, recovered.")
