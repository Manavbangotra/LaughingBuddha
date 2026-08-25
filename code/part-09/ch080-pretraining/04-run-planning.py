# Extracted from: Chapter 80 — Pretraining and Self-Supervised Objectives
# Source: src/.../ch080-pretraining.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Planning a pretraining run: budget, size, duration, and the failure budget."""

DEVICES = 256
DEVICE_FLOPS = 1e15
MFU = 0.45
DAYS = 30
MTBF_HOURS = 20.0          # mean time between failures across the whole fleet
RESTART_MINUTES = 25.0     # detect, reschedule, reload, replay

budget = DEVICES * DEVICE_FLOPS * MFU * DAYS * 86_400
print(f"compute budget: {budget:.3e} FLOPs over {DAYS} days on {DEVICES} devices\n")

# Equation (eq:chinchilla-ratio): C = 6ND with D/N = 20 gives N = sqrt(C/120).
n_opt = (budget / (6 * 20)) ** 0.5
d_opt = 20 * n_opt
print(f"Chinchilla-optimal : N = {n_opt / 1e9:.2f}B params, "
      f"D = {d_opt / 1e9:.0f}B tokens")

print(f"\n{'model':>8} {'tokens':>10} {'D/N':>6} {'train days':>11} "
      f"{'infer TFLOPs/1k tok':>21}")
for n in (1e9, 3e9, 7e9, 13e9, 30e9):
    d = budget / (6 * n)
    days = budget / (DEVICES * DEVICE_FLOPS * MFU) / 86_400
    infer = 2 * n * 1000 / 1e12          # 2N per token, ch:tf-complexity
    print(f"{n / 1e9:>7.0f}B {d / 1e9:>9.0f}B {d / n:>6.1f} {days:>11.1f} "
          f"{infer:>21.2f}")

print("""
Every row spends the same compute — that is what a fixed budget means. The
choice is where to spend it, and the last column is why the answer is usually
smaller than Chinchilla says: inference cost scales with N alone and is paid on
every request forever, while training compute is paid once.""")

# The failure budget: how often to checkpoint.
run_hours = DAYS * 24
expected_failures = run_hours / MTBF_HOURS
print(f"\nexpected interruptions over {run_hours:.0f} h at {MTBF_HOURS} h MTBF: "
      f"{expected_failures:.1f}")

print(f"\n{'checkpoint every':>18} {'lost work/failure':>19} "
      f"{'total lost':>12} {'ckpt overhead':>15} {'total':>9}")
CKPT_WRITE_MIN = 4.0
best = None
for interval_min in (15, 30, 60, 120, 240, 480):
    # On average a failure loses half the interval, plus the restart cost.
    lost_per = interval_min / 2 + RESTART_MINUTES
    total_lost = expected_failures * lost_per / 60
    overhead = (run_hours * 60 / interval_min) * CKPT_WRITE_MIN / 60
    total = total_lost + overhead
    if best is None or total < best[1]:
        best = (interval_min, total)
    print(f"{interval_min:>15} min {lost_per:>16.0f} min {total_lost:>10.1f} h "
          f"{overhead:>13.1f} h {total:>7.1f} h")

print(f"\noptimal checkpoint interval: every {best[0]} minutes "
      f"({best[1]:.1f} h lost in total, {best[1] / run_hours:.1%} of the run)")
print("Checkpoint cadence is a tradeoff between work lost to failures and time "
      "spent writing. It is chosen against the measured MTBF, not by habit.")
