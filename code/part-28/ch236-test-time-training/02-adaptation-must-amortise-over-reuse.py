# -*- coding: utf-8 -*-
# Extracted from: Chapter 236 — Test-Time Training and Test-Time Compute
# Source: src/.../ch236-test-time-training.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Adapting at test time is cheap in FLOPs and expensive in batching.

The first listing spent test-time compute on sampling. The other way to spend it is on
*training*: update the parameters using the test input, the session, or the user, and then
answer.

Two things decide whether that is a good idea, and neither is the obvious one. The adaptation
compute has to amortise over however many requests reuse the adapted weights, which is a scope
decision rather than an algorithmic one
(eq:adaptation-must-amortise-over-reuse).

And whatever is adapted has to fit in memory *per distinct adapter*, because a batch can only
share the weights its members have in common. Small adapters (cite:hu2021lora) survive this;
full-weight adaptation does not, and batching is where all serving efficiency comes from
(eq:per-request-adaptation-forfeits-the-batch).
"""
import math

ACTIVE, TOKENS_OUT, BYTES = 2.0e10, 700, 2
LORA_PARAMS = 2.0e7
G_MAX, TAU = 0.085, 3.0e3
HBM_BW, FLOPS_S, HBM_GB = 3.35e12, 9.9e14, 80.0
INFER_FLOPS = 2 * ACTIVE * TOKENS_OUT

# (scope, tokens adapted on, passes, reuse count, relevance of that data)
SCOPES = [
    ("no adaptation",     0,          0, 1,           0.00),
    ("the request itself", 2_000,     8, 1,           1.00),
    ("the session",        20_000,    3, 40,          0.92),
    ("the user's history", 400_000,   2, 3_000,       0.78),
    ("the whole corpus",   50_000_000, 1, 1_000_000_000, 0.45),
]


def gain(tokens, relevance):
    if tokens == 0:
        return 0.0
    tau = TAU / max(relevance, 1e-6) ** 2
    return G_MAX * relevance * (1.0 - math.exp(-tokens / tau))


print("Where the adaptation data comes from, and what it is worth.")
print()
print(f"{'adapted on':>22}{'tokens':>14}{'passes':>9}{'relevance':>12}"
      f"{'accuracy gain':>16}{'reuses':>17}")
print("-" * 90)
rows = {}
for name, tok, passes, reuse, rel in SCOPES:
    g = gain(tok, rel)
    flops = 6 * LORA_PARAMS * tok * passes
    rows[name] = (tok, passes, reuse, rel, g, flops)
    print(f"{name:>22}{tok:>14,}{passes:>9}{rel:>12.2f}{g:>16.4f}{reuse:>17,}")

BEST_GAIN = max((n for n in rows if n != "no adaptation"), key=lambda n: rows[n][4])
print()
print(f"largest gain: {BEST_GAIN} at {rows[BEST_GAIN][4]:.4f}")
print("relevance and volume pull in opposite directions, and the optimum is neither end")

print()
print()
print("Adaptation compute, amortised over whatever reuses it.")
print()
print(f"{'adapted on':>22}{'adapt FLOPs':>15}{'reuses':>16}{'FLOPs / request':>18}"
      f"{'share of inference':>21}{'gain per 1% overhead':>23}")
print("-" * 115)
amort = {}
for name, tok, passes, reuse, rel in SCOPES:
    _, _, _, _, g, flops = rows[name]
    per_req = flops / reuse
    share = per_req / INFER_FLOPS
    eff = g / max(share * 100, 1e-9)
    amort[name] = (per_req, share, eff)
    es = f"{eff:>23.4f}" if name != "no adaptation" else f"{'--':>23}"
    print(f"{name:>22}{flops:>15.2e}{reuse:>16,}{per_req:>18.2e}{share:>20.2%}{es}")

print()
print("the last column degenerates as overhead approaches zero, which is the")
print("corpus row's whole story: it is fine-tuning, and fine-tuning is free per request")
print(f"adapting on the request itself costs {amort['the request itself'][1]:.1%}"
      f" of an inference for {rows['the request itself'][4]:.4f}")

print()
print()
print("How many reuses each scope needs before the overhead is under 5%.")
print()
print(f"{'adapted on':>22}{'adapt FLOPs':>15}{'reuses for 5%':>17}"
      f"{'actual reuses':>16}{'amortises?':>13}")
print("-" * 83)
for name, tok, passes, reuse, rel in SCOPES:
    if name == "no adaptation":
        continue
    _, _, _, _, g, flops = rows[name]
    need = flops / (0.05 * INFER_FLOPS)
    print(f"{name:>22}{flops:>15.2e}{need:>17,.1f}{reuse:>16,}"
          f"{('yes' if reuse >= need else 'no'):>13}")

print()
print("(eq:adaptation-must-amortise-over-reuse)")

print()
print()
print("Now the constraint that actually decides it: batching.")
print()
BASE_GB = ACTIVE * BYTES / 1e9
print(f"base weights {BASE_GB:.0f} GB, {HBM_GB:.0f} GB of memory")
print()
MECHS = [
    ("none, shared weights",   0.0,            1),
    ("LoRA rank 16",           LORA_PARAMS,    1),
    ("LoRA rank 256",          LORA_PARAMS * 16, 1),
    ("full-weight adaptation", ACTIVE,         1),
]
print(f"{'adaptation mechanism':>24}{'bytes per adapter':>20}{'adapters that fit':>20}"
      f"{'effective batch':>18}{'tokens/s':>13}{'relative':>11}")
print("-" * 106)
WANT_BATCH = 512
thr = {}
SHARED_TPS = None
for name, params, _ in MECHS:
    ad_bytes = params * BYTES
    room = HBM_GB * 1e9 - BASE_GB * 1e9
    fit = int(room / ad_bytes) if ad_bytes > 0 else WANT_BATCH
    batch = min(WANT_BATCH, max(1, fit))
    read = BASE_GB * 1e9 + ad_bytes * (batch if ad_bytes > 0 else 0)
    mem = batch / (read / HBM_BW)
    comp = FLOPS_S / (2 * ACTIVE)
    tps = min(mem, comp)
    thr[name] = tps
    if SHARED_TPS is None:
        SHARED_TPS = tps
    print(f"{name:>24}{ad_bytes:>20,.0f}{fit:>20,}{batch:>18,}"
          f"{tps:>13,.0f}{tps / SHARED_TPS:>10.2f}x")

SHARED = thr["none, shared weights"]
FULL = thr["full-weight adaptation"]
print()
print(f"shared weights: {SHARED:,.0f} tokens/s; full-weight adaptation:"
      f" {FULL:,.0f}")
print(f"a factor of {SHARED / FULL:,.0f}, and none of it is arithmetic")

print()
print()
print("And adaptations go stale, so the reuse count is not free either.")
print()
print(f"{'requests since adapting':>26}{'drift':>10}{'gain retained':>16}"
      f"{'effective gain':>17}")
print("-" * 69)
G0 = rows["the user's history"][4]
retained = {}
for r in (1, 10, 100, 1_000, 10_000, 100_000):
    drift = 1.0 - math.exp(-r / 4_000.0)
    keep = 1.0 - 0.72 * drift
    retained[r] = keep
    print(f"{r:>26,}{drift:>10.3f}{keep:>16.3f}{G0 * keep:>17.4f}")

print()
print(f"after {3_000:,} reuses -- the user scope's amortisation window --")
print(f"{1.0 - math.exp(-3_000 / 4_000.0):.2f} of the distribution has drifted and"
      f" {1.0 - 0.72 * (1.0 - math.exp(-3_000 / 4_000.0)):.2f} of the gain remains")

print()
print()
print("Putting the three constraints together.")
print()
print(f"{'adapted on':>22}{'raw gain':>11}{'after staleness':>18}"
      f"{'overhead':>11}{'distinct adapters in a 512 batch':>34}{'net verdict':>28}")
print("-" * 124)
DISTINCT = {
    "the request itself": 512,
    "the session":        512,
    "the user's history": 512,
    "the whole corpus":   1,
}
VERDICTS = {
    "the request itself": "no reuse, but batchable",
    "the session":        "the practical sweet spot",
    "the user's history": "amortises, then goes stale",
    "the whole corpus":   "this is just fine-tuning",
}
for name, tok, passes, reuse, rel in SCOPES:
    if name == "no adaptation":
        continue
    g = rows[name][4]
    drift = 1.0 - math.exp(-reuse / 4_000.0)
    net = g * (1.0 - 0.72 * drift)
    print(f"{name:>22}{g:>11.4f}{net:>18.4f}{amort[name][1]:>10.1%}"
          f"{DISTINCT[name]:>34,}{VERDICTS[name]:>28}")

print(f"""
The first table separates two things that get conflated. Adaptation data has a *volume* and a
*relevance*, and they move in opposite directions. The request itself is perfectly relevant and
{2_000:,} tokens long. The whole corpus is {50_000_000:,} tokens and only {0.45:.2f} relevant.

The gain peaks in between: `{BEST_GAIN}` at **{rows[BEST_GAIN][4]:.4f}**, against
{rows['the request itself'][4]:.4f} for the request alone and
{rows['the whole corpus'][4]:.4f} for the corpus. **The best thing to adapt on is neither the
input nor everything**, which is not where either camp's intuition points.

The amortisation table prices it. Adapting on the request itself costs
{amort['the request itself'][1]:.1%} of an inference and is reused exactly once; adapting on the
corpus costs {rows['the whole corpus'][5]:.2e} FLOPs and is reused a billion times, so its
per-request overhead is {amort['the whole corpus'][1]:.4%} -- effectively free, and it is also
just fine-tuning under a different name.

The threshold table states the rule directly (eq:adaptation-must-amortise-over-reuse). Every
scope clears a 5% overhead budget except adapting on the request itself, which needs
{rows['the request itself'][5] / (0.05 * INFER_FLOPS):,.1f} reuses and gets exactly one.

So on compute alone, test-time training looks affordable and even the per-request version is
only a {amort['the request itself'][1]:.0%} overhead. **Compute is not the constraint.**

The batching table is (eq:per-request-adaptation-forfeits-the-batch). A serving step reads the
weights every batch member shares, so a batch can only be as wide as the number of members using
the same weights. With cite:hu2021lora-style adapters that is fine: a rank-16 adapter is
{LORA_PARAMS * BYTES / 1e6:.0f} MB, so {int((HBM_GB * 1e9 - BASE_GB * 1e9) / (LORA_PARAMS * BYTES)):,}
of them fit alongside the base weights and the batch is unaffected.

With full-weight adaptation it is not fine. Each adapted copy is {BASE_GB:.0f} GB, exactly one
fits, and the effective batch is **1**. Throughput falls from {SHARED:,.0f} tokens per second to
{FULL:,.0f} -- **a factor of {SHARED / FULL:,.0f}, none of it arithmetic.**

That is the result to carry out of this listing. **Test-time training is affordable exactly to
the extent that what you adapt is small.** The algorithm barely matters; the parameter count of
the thing being updated decides whether the technique can be served at all, and it is the same
constraint ch:res-moe found from the other direction.

The staleness table adds the third term. An adaptation is fitted to a distribution that moves.
After {3_000:,} reuses -- which is exactly the user scope's amortisation window --
{1.0 - math.exp(-3_000 / 4_000.0):.0%} of the distribution has drifted and
{1.0 - 0.72 * (1.0 - math.exp(-3_000 / 4_000.0)):.0%} of the gain remains.

**The reuse that pays for the adaptation is the same reuse that erodes it**, which means the
amortisation window and the freshness window are the same interval pulling in opposite
directions, and there is an optimal refresh rather than a maximal one.

The summary table is the practical answer. `the session` wins on raw gain, survives staleness
almost intact because its reuse count is small, carries a
{amort['the session'][1]:.1%} overhead, and batches fine. It is also the scope nobody markets,
because it is neither the striking research result nor the familiar production one.""")
