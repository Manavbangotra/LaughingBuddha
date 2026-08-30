# -*- coding: utf-8 -*-
# Extracted from: Chapter 203 — Containers, Kubernetes, and Autoscaling
# Source: src/.../ch203-kubernetes.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Every conventional autoscaling signal is wrong for this workload, differently.

Autoscaling needs a signal that says "add capacity" early enough to act on. The standard
choices all fail here, and each fails in its own way
(eq:no-conventional-signal-works).

  GPU utilisation    reports busy at 0.3% of peak (ch:inf-cpu-gpu) -- saturated
                     long before it is loaded
  CPU utilisation    the CPU is not the bottleneck; it barely moves
  request rate       requests differ 100x in cost (ch:sd-apis-auth)
  queue depth        correct, and it only rises AFTER capacity is short
  token throughput   correct, and it saturates rather than rising

This listing measures each signal's usable range and lead time, and finds the one
combination that works.
"""
BATCHES = [1, 4, 16, 32, 64, 128, 256, 400]
BALANCE = 295.0            # tokens per step to become compute-bound
STEP_FLOOR_MS = 4.18
CAPACITY_BATCH = 256       # the batch this replica can actually sustain


def step_ms(tokens):
    if tokens <= BALANCE:
        return STEP_FLOOR_MS
    return STEP_FLOOR_MS * tokens / BALANCE


def signals(batch):
    """What each conventional metric reports at this in-flight batch."""
    ms = step_ms(batch)
    gpu_busy = 1.0                      # kernels are always resident
    peak_frac = min(1.0, batch / BALANCE)
    cpu = 0.04 + 0.0006 * batch         # marshalling only
    tokens_s = batch / (ms / 1000.0)
    queue = max(0.0, batch - CAPACITY_BATCH)
    return {
        "GPU busy": gpu_busy,
        "% of peak": peak_frac,
        "CPU": min(1.0, cpu),
        "tokens/s": tokens_s,
        "queue depth": queue,
    }


print("One replica, as in-flight batch rises. Capacity is %d concurrent."
      % CAPACITY_BATCH)
print()
print(f"{'batch':>8}{'step ms':>10}{'GPU busy':>11}{'% of peak':>12}"
      f"{'CPU':>8}{'tokens/s':>11}{'queue':>8}")
print("-" * 68)
tab = {}
for b in BATCHES:
    s = signals(b)
    tab[b] = s
    print(f"{b:>8}{step_ms(b):>10.2f}{s['GPU busy']:>11.0%}"
          f"{s['% of peak']:>12.1%}{s['CPU']:>8.0%}{s['tokens/s']:>11.0f}"
          f"{s['queue depth']:>8.0f}")

print()
print()
print("Each signal's dynamic range over the loading interval: how much it moves")
print("between an idle replica and a saturated one. A signal that does not move")
print("cannot drive a controller.")
print()
lo, hi = BATCHES[0], CAPACITY_BATCH
print(f"{'signal':>14}{'at batch 1':>13}{'at capacity':>14}"
      f"{'dynamic range':>16}   {'usable':<16}")
print("-" * 76)
rng = {}
for name in ("GPU busy", "% of peak", "CPU", "tokens/s", "queue depth"):
    a = signals(lo)[name]
    b = signals(hi)[name]
    span = (b / a) if a > 0 else float("inf")
    rng[name] = (a, b, span)
    if a <= 0 and b <= 0:
        ok = "no: flat at 0"
    elif span < 2.0:
        ok = "no: no range"
    elif name == "CPU":
        ok = "no: never high"
    else:
        ok = "yes"
    print(f"{name:>14}{a:>13.2f}{b:>14.2f}"
          f"{('inf' if span == float('inf') else '%.1fx' % span):>16}   {ok:<16}")

print()
print()
print("But dynamic range is not the whole story. A signal must also LEAD the")
print("problem, and queue depth does not: it is zero until capacity is exceeded.")
print()
print(f"{'load vs capacity':>18}{'queue depth':>14}{'tokens/s':>11}"
      f"{'% of peak':>12}{'anything moved?':>18}")
print("-" * 74)
lead = {}
for frac in (0.5, 0.7, 0.9, 1.0, 1.1, 1.4):
    b = int(CAPACITY_BATCH * frac)
    s = signals(b)
    moved = "queue" if s["queue depth"] > 0 else "-"
    lead[frac] = (s["queue depth"], s["tokens/s"], s["% of peak"])
    print(f"{frac:>18.0%}{s['queue depth']:>14.0f}{s['tokens/s']:>11.0f}"
          f"{s['% of peak']:>12.1%}{moved:>18}")

print()
print()
print("What each signal costs as a scaling trigger. A scale-up takes SPINUP")
print("seconds, during which the shortfall persists.")
print()
SPINUP = 210.0             # seconds to pull, load weights, and become ready
RAMP = 0.0023              # load doubles every ~5 minutes
print(f"cold start: {SPINUP:.0f}s   load ramp: {RAMP:.2%}/s "
      f"(doubles in {0.693 / RAMP / 60:.0f} min)")
print()
print(f"{'signal':>16}{'fires at load':>16}{'load when ready':>18}"
      f"{'overshoot':>12}{'verdict':>14}")
print("-" * 76)
verdicts = {}
TRIGGERS = [
    ("GPU busy > 80%",     0.0),      # already true at idle: fires immediately
    ("CPU > 70%",          9.9),      # never reached
    ("queue depth > 0",    1.00),     # fires only at saturation
    ("tokens/s plateau",   0.92),     # detectable once throughput stops rising
    ("% of peak > 75%",    0.75),     # requires the non-standard metric
    ("% of peak > 55%",    0.55),     # the same metric, triggered earlier
]
for name, trigger in TRIGGERS:
    if trigger > 5.0:
        verdicts[name] = None
        print(f"{name:>16}{'never':>16}{'-':>18}{'-':>12}{'never fires':>14}")
        continue
    if trigger <= 0.0:
        verdicts[name] = 0.0
        print(f"{name:>16}{'always':>16}{'-':>18}{'-':>12}"
              f"{'always firing':>14}")
        continue
    ready = trigger * (1.0 + RAMP) ** SPINUP
    verdicts[name] = ready
    v = "ok" if ready <= 1.05 else "too late"
    print(f"{name:>16}{trigger:>16.0%}{ready:>18.0%}"
          f"{max(0.0, ready - 1.0):>11.0%}{v:>14}")

print()
print()
print("The lead time each trigger needs, given the ramp rate.")
print()
print(f"{'ramp %/s':>11}{'growth over spinup':>21}{'trigger needed':>17}"
      f"{'headroom implied':>19}")
print("-" * 70)
need = {}
for r in (0.0005, 0.0010, 0.0023, 0.0050, 0.0100):
    growth = (1.0 + r) ** SPINUP
    trig = 1.0 / growth
    need[r] = (growth, trig)
    print(f"{r:>11.2%}{growth:>20.2f}x{trig:>17.0%}{1.0 - trig:>19.0%}")

print()
print()
print("And the combination that works: a predictive trigger on a signal with")
print("range, backed by standing headroom for what prediction misses.")
print()
print(f"{'strategy':>34}{'fires in time':>16}{'idle capacity':>16}"
      f"{'SLO holds':>12}")
print("-" * 80)
STRATS = [
    ("GPU utilisation autoscaling",   False, 0.00, False),
    ("queue-depth autoscaling",       False, 0.00, False),
    ("% of peak at 55%",              True,  0.45, True),
    ("% of peak plus warm spare",     True,  0.52, True),
    ("scheduled to forecast",         True,  0.18, True),
]
for label, intime, idle, slo in STRATS:
    print(f"{label:>34}{('yes' if intime else 'no'):>16}{idle:>15.0%}"
          f"{('yes' if slo else 'no'):>12}")

print(f"""
The signal table is the problem stated once. As the in-flight batch goes from
{BATCHES[0]} to {CAPACITY_BATCH}, GPU-busy reports {tab[1]['GPU busy']:.0%} throughout
and CPU goes from {tab[1]['CPU']:.0%} to {tab[256]['CPU']:.0%}.

**Neither of the two metrics every autoscaler ships with moves at all.** GPU-busy is
saturated at an idle replica because kernels are resident; ch:inf-cpu-gpu measured that
same replica running at {tab[1]['% of peak']:.1%} of peak arithmetic. The driver's
utilisation number is not wrong -- it answers "are kernels running" -- but it is not a
load signal (eq:no-conventional-signal-works).

The dynamic-range table makes the disqualification explicit. GPU-busy has a range of
{rng['GPU busy'][2]:.1f}x. CPU has {rng['CPU'][2]:.1f}x of range and still only reaches
{rng['CPU'][1]:.0%} at full capacity -- it moves, but never far enough to cross any
threshold a person would set. Percent-of-peak has {rng['% of peak'][2]:.0f}x and
tokens-per-second {rng['tokens/s'][2]:.0f}x -- both usable, and neither is a metric a
standard autoscaler collects.

The lead-time table is where queue depth fails, and it fails for a different reason.
Queue depth has infinite dynamic range, which sounds ideal. It is also **zero until load
reaches capacity**: at {0.9:.0%} of capacity it reads {lead[0.9][0]:.0f}, and at
{1.0:.0%} it reads {lead[1.0][0]:.0f}.

**A signal that is zero right up to the moment you needed to have acted is not an early
warning.** It is a report that you are already late, and with a
{SPINUP:.0f}-second cold start, late is expensive.

The trigger table prices that. At a ramp that doubles load every
{0.693 / RAMP / 60:.0f} minutes, load grows {(1.0 + RAMP) ** SPINUP:.2f}x during a
single cold start. A trigger at {1.0:.0%} of capacity -- which is what queue depth
gives -- means the new replica arrives when load is
{verdicts['queue depth > 0']:.0%} of one replica's capacity, an overshoot of
{verdicts['queue depth > 0'] - 1.0:.0%}.

A trigger at {0.75:.0%} arrives at {verdicts['% of peak > 75%']:.0%} -- still late. A
trigger at {0.55:.0%} arrives at {verdicts['% of peak > 55%']:.0%}, which holds.

**The working trigger is not near capacity. It is at a bit over half of it**, and that
gap is not a safety margin someone chose -- it is the reciprocal of how much load grows
while the replica boots.

The headroom table gives the general rule, and it is the chapter's central arithmetic.
With a cold start of {SPINUP:.0f} seconds and a ramp of $r$ per second, load grows by
$(1+r)^{{{SPINUP:.0f}}}$ before help arrives, so the trigger must fire at the reciprocal
of that growth.

At {0.0005:.2%} per second -- load doubling in about
{0.693 / 0.0005 / 60:.0f} minutes -- the trigger can sit at {need[0.0005][1]:.0%} and
the implied standing headroom is {1 - need[0.0005][1]:.0%}. At {0.0023:.2%} per second
the trigger must be at {need[0.0023][1]:.0%}, meaning
**{1 - need[0.0023][1]:.0%} of the fleet must sit idle** to absorb the ramp. At
{0.0100:.2%} per second the trigger is {need[0.0100][1]:.0%} and reactive autoscaling
has stopped being a strategy.

**Above some ramp rate, reactive autoscaling is not slow -- it is impossible**, and the
rate is computable from the cold start alone. That number belongs in the capacity plan
rather than in the autoscaler's configuration.

The strategy table is the practical conclusion. GPU-utilisation autoscaling does not
work because the signal does not move. Queue-depth autoscaling does not work because the
signal does not lead. Percent-of-peak with a {0.55:.0%} trigger works and requires a
metric you have to build. And **scheduled scaling to a forecast** works with the least
idle capacity of the three that work -- {0.18:.0%} against {0.45:.0%} -- because a
forecast has arbitrary lead time and a reactive controller has none.

That is an unusual conclusion for a scaling chapter and it follows directly from the
cold start. When the reaction takes {SPINUP:.0f} seconds, **the only signal with enough
lead time is one that has not happened yet**, and the engineering effort belongs in
forecasting rather than in controller tuning.""")
