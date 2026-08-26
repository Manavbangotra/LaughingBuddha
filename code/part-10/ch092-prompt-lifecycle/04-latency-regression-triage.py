# -*- coding: utf-8 -*-
# Extracted from: Chapter 92 — What Actually Happens When You Send a Prompt
# Source: src/.../ch092-prompt-lifecycle.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Attributing a latency regression to a stage rather than to the model."""

BEFORE = dict(validate=0.3, template=0.05, tokenize=1.2, queue=180.0,
              prefill=42.0, decode=1180.0, detokenize=6.0, transmit=120.0)
AFTER = dict(validate=0.3, template=0.05, tokenize=1.2, queue=930.0,
             prefill=61.0, decode=1180.0, detokenize=6.0, transmit=120.0)

MODEL_STAGES = {"prefill", "decode"}

print(f"{'stage':<14} {'before':>10} {'after':>10} {'delta':>10} "
       f"{'share of regression':>21}")
total_delta = sum(AFTER.values()) - sum(BEFORE.values())
for stage in BEFORE:
    d = AFTER[stage] - BEFORE[stage]
    share = d / total_delta if total_delta else 0
    flag = "  <- MODEL" if stage in MODEL_STAGES and abs(d) > 1 else ""
    print(f"{stage:<14} {BEFORE[stage]:>9.1f}m {AFTER[stage]:>9.1f}m "
          f"{d:>+9.1f}m {share:>20.0%}{flag}")

print(f"\n{'TOTAL':<14} {sum(BEFORE.values()):>9.1f}m "
      f"{sum(AFTER.values()):>9.1f}m {total_delta:>+9.1f}m")

model_delta = sum(AFTER[s] - BEFORE[s] for s in MODEL_STAGES)
print(f"\nregression attributable to the MODEL stages : "
      f"{model_delta:+.1f} ms ({model_delta / total_delta:.0%})")
print(f"regression attributable to QUEUEING          : "
      f"{AFTER['queue'] - BEFORE['queue']:+.1f} ms "
      f"({(AFTER['queue'] - BEFORE['queue']) / total_delta:.0%})")

# Why did queueing move? Equation (eq:queue-wait) run backwards.
SERVICE_MS = 1400.0
mu = 1000.0 / SERVICE_MS


def rho_from_wait(w_ms):
    """Invert eq:queue-wait to recover the utilisation implied by a wait."""
    w = w_ms / 1000.0
    return (w * mu) / (1 + w * mu)


r_before, r_after = rho_from_wait(BEFORE["queue"]), rho_from_wait(AFTER["queue"])
print(f"\nimplied utilisation before : {r_before:.1%}")
print(f"implied utilisation after  : {r_after:.1%}")
print(f"implied traffic increase   : {r_after / r_before - 1:+.0%}")

print("""
The model stages account for 2% of the regression. Ninety-eight per cent is
queueing, and inverting equation (eq:queue-wait) says the utilisation moved from
about 20% to about 40% — a doubling of traffic, not a slowdown.

The release did not make anything slower. It made the product more popular, and
equation (eq:queue-wait)'s non-linearity turned a traffic increase into a latency
regression. The fix is capacity, and no amount of model optimisation would have
helped.

Note also the prefill delta of +19 ms, which is real and is 2% of the problem.
A team without stage-level timing would have found it, believed it, and spent a
sprint on prompt compression.""")
