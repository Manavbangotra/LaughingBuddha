---
id: inf-kubernetes
number: 203
part: XXIII
tier: full
status: draft
requires: [decode-is-bandwidth-bound, batch-times-context-is-the-budget,
           affinity-fights-balance, variance-not-mean-drives-wait]
provides: [no-conventional-signal-works, trigger-is-the-reciprocal-of-growth,
           cold-start-is-mostly-weight-movement, weight-placement-sets-utilisation]
citations: [kwon2023pagedattention, patel2023splitwise, zhong2024distserve,
            shoeybi2019megatron]
---

## 1. Learning Objectives

By the end of this chapter you will be able to explain why GPU utilisation, CPU
utilisation, request rate and queue depth all fail as autoscaling signals for this
workload, and say what fails about each; compute the scaling trigger a given cold start
and traffic ramp require, and show that it is the reciprocal of the growth during the
cold start; identify the ramp rate above which reactive autoscaling is impossible rather
than merely slow; decompose a cold start and show that weight movement dominates it; and
explain why baking weights into a container image is the worst available placement.

## 2. Why This Matters

Everything about running this workload on a cluster is downstream of one fact: a replica
takes minutes to become useful, and every conventional signal that would tell you to
start one is either flat, absent, or too late.

{{sec:9-practical-example}} measures the standard metrics across a replica's whole
loading range. GPU-busy reports **100%** at an idle replica and **100%** at a saturated
one — a dynamic range of **1.0×**, because kernels are resident whether or not they are
doing anything. CPU moves from **4%** to **19%**, never crossing any threshold a person
would set. Queue depth has infinite range and reads **zero at 90% of capacity**
({{eq:no-conventional-signal-works}}).

The one that works is percent-of-peak-arithmetic, which is not a metric any autoscaler
collects — and even it must fire at **55%** rather than near capacity, because load grows
**1.62×** during a 210-second cold start
({{eq:trigger-is-the-reciprocal-of-growth}}).

The second listing opens the cold start and finds **75%** of it is moving weights
({{eq:cold-start-is-mostly-weight-movement}}). Where those weights live sets the cold
start, which sets the trigger, which sets how much of the fleet must sit idle: **67%**
from a container image against **19%** from the host page cache
({{eq:weight-placement-sets-utilisation}}).

## 3. Prerequisites

You need {{eq:decode-is-bandwidth-bound}} from {{ch:inf-cpu-gpu}}. The reason GPU-busy is
useless here is precisely that chapter's result — the device is 100% occupied at 0.3% of
peak — and without it the first table looks like a broken sensor rather than a correct
one answering the wrong question.

{{eq:batch-times-context-is-the-budget}} from {{ch:inf-gpu-memory}} defines the capacity
a replica actually has, which is what any trigger is a fraction of.

{{eq:affinity-fights-balance}} from {{ch:inf-distributed}} is why draining a node costs
more than its capacity share, which matters for scale-down.

{{eq:variance-not-mean-drives-wait}} supplies the queueing behaviour that makes
overshoot expensive rather than merely untidy.

## 4. Intuitive Explanation

Autoscaling is a control loop. Something measures load, compares it to a threshold, and
starts a replica. For that to work, three things must be true: the measurement must move
with load, it must move *before* you are in trouble, and the replica must arrive while
the measurement is still roughly what it was when you decided.

This workload breaks all three, and it breaks them in ways that each look like a
different problem.

**The measurement does not move.** The natural signal is GPU utilisation, and every
cluster already collects it. But {{ch:inf-cpu-gpu}} showed that a GPU serving one
sequence runs at a fraction of a percent of its arithmetic capability while being
entirely "busy" — kernels are resident, the driver reports occupancy, and the number is
100%. It is 100% at one concurrent request and 100% at two hundred and fifty-six. A
controller cannot act on a constant.

CPU utilisation has the opposite problem: it moves, and it moves from four percent to
nineteen. No threshold anyone would write catches that.

**The measurement arrives too late.** Queue depth is the honest signal — it is exactly
zero while you have capacity and rises the moment you do not. That sounds like a good
alarm and it is a terrible one, because "the moment you do not" is the moment you needed
to have started a replica four minutes ago. A signal that is zero right up to the point
of failure is not an early warning; it is a report.

**The replica arrives too late regardless.** A pod must be scheduled, an image pulled, a
runtime started, and then — the expensive part — a hundred and forty gigabytes of
weights must be fetched from somewhere and pushed across PCIe into device memory. Call
it three and a half minutes. Traffic that is growing does not pause for this.

Which gives the chapter's central arithmetic, and it is simple enough to do in your
head. If load doubles every five minutes and your cold start is three and a half
minutes, load grows about 1.6× while the replica boots. So you must trigger at 1/1.6 —
about 62% of capacity — because by the time help arrives you will be at 100%.

**The trigger is the reciprocal of the growth during the cold start.** Not a safety
margin someone chose; an arithmetic consequence of two numbers.

And that has an uncomfortable corollary. Push the ramp rate up or the cold start out,
and the required trigger falls below anything sensible. At load doubling every seventy
seconds, you would need to trigger at twelve percent — meaning eighty-eight percent of
your fleet sits idle so that the twelve percent can be absorbed. Past some point,
reactive autoscaling is not slow. It is arithmetically impossible, and the only
remaining options are forecasting or standing capacity.

Since the cold start is the villain, the second half of the chapter asks where it goes.
The answer is not where platform teams usually look. Pod scheduling, image pull, and
runtime start together are about a quarter of it. Three quarters is moving weights — and
the time that takes depends almost entirely on *where the weights are stored*, which is a
decision usually made for convenience.

## 5. Formal Explanation

**Signal adequacy.** A metric $\mu$ is usable as a scaling signal if it satisfies two
conditions over the loading interval $[0, m^\star]$ where $m^\star$ is replica capacity:

$$ \text{range: } \frac{\mu(m^\star)}{\mu(\epsilon)} \gg 1, \qquad \text{lead: } \frac{d\mu}{dm} > 0 \text{ for } m < m^\star $$ (eq:no-conventional-signal-works)

GPU-busy fails the first ($\mu \equiv 1$). Queue depth fails the second
($\mu \equiv 0$ for $m < m^\star$, so its derivative is zero exactly where lead time is
needed). CPU satisfies both weakly but never reaches a threshold. Percent-of-peak,
$\mu = \min(1, m/I^\star)$ with $I^\star$ the balance point from
{{eq:decode-is-bandwidth-bound}}, satisfies both — and is not collected by default.

**Trigger placement.** Let load grow at rate $r$ per second and a cold start take $S$
seconds. Load at the moment a replica becomes ready, having triggered at fraction
$\theta$ of capacity, is $\theta(1+r)^S$. Requiring that to be at most capacity:

$$ \theta^\star \;=\; (1+r)^{-S} $$ (eq:trigger-is-the-reciprocal-of-growth)

and the standing headroom is $1 - \theta^\star$. Two consequences follow immediately.

The headroom is **exponential in the cold start**, so halving $S$ does far more than
halving the idle fraction. And $\theta^\star \to 0$ as $rS$ grows, so there is a ramp
rate above which no trigger is early enough — reactive autoscaling has no solution, not
a slow one.

**Cold-start decomposition.** Write $S = S_{\text{fixed}} + W/\beta + W/\pi$ where $W$ is
weight bytes, $\beta$ the source bandwidth, and $\pi$ the host-to-device bandwidth. The
weight-movement share is

$$ \frac{W(1/\beta + 1/\pi)}{S_{\text{fixed}} + W(1/\beta + 1/\pi)} $$ (eq:cold-start-is-mostly-weight-movement)

which approaches 1 as $W$ grows. Since $\pi \gg \beta$ for every realistic source, the
$1/\beta$ term dominates, and **the storage choice is the cold-start choice**.

Composing the two gives the chapter's operational result:

$$ 1 - \theta^\star \;=\; 1 - (1+r)^{-\left(S_{\text{fixed}} + W/\beta + W/\pi\right)} $$ (eq:weight-placement-sets-utilisation)

**Fleet idle capacity is a function of where the weights are stored.** Not of the
autoscaler's configuration, which appears nowhere in that expression.

## 6. Mathematical Foundation

The exponential in {{eq:trigger-is-the-reciprocal-of-growth}} is what makes cold-start
reduction unusually valuable, and it is worth quantifying against the alternative of
buying capacity.

Adding a replica reduces the required headroom *fraction* not at all — $\theta^\star$
depends on $r$ and $S$, not on fleet size — so buying capacity to cover a ramp costs
linearly in fleet size forever. Reducing $S$ changes $\theta^\star$ multiplicatively.
Differentiating,

$$ \frac{\partial(1-\theta^\star)}{\partial S} \;=\; \theta^\star \ln(1+r) \;>\; 0 $$

so the marginal idle capacity saved per second of cold start removed is proportional to
$\theta^\star$ itself — largest when the situation is already good, smallest when it is
dire. That is an awkward shape: **cold-start work pays best in deployments that already
have short cold starts**, and a deployment at 486 seconds must make a large absolute
reduction before the returns become attractive.

The practical reading is that the first big win — moving weights off the container image
— is worth taking as a step change rather than incrementally, and
{{sec:9-practical-example}} measures it at **273 seconds**.

There is also a threshold worth naming. Reactive autoscaling requires
$\theta^\star$ to exceed whatever minimum utilisation the business can tolerate, call it
$\theta_{\min}$. Solving,

$$ rS \;<\; -\ln\theta_{\min} \quad\text{approximately, for small } r $$

so with a tolerance of 50% idle ($\theta_{\min} = 0.5$), reactive scaling works only
while $rS < 0.69$. At $S = 210$ seconds that permits $r < 0.0033$ per second — load
doubling no faster than every 3.5 minutes. **Faster than that and the question stops
being how to tune the autoscaler.**

## 7. Internal Mechanics

**Why percent-of-peak has to be built.** The quantity is achieved FLOP/s divided by
device peak, and no standard exporter reports it. It can be computed from tokens
processed per second and the model's parameter count — $2Pm/\text{step time}$ over peak —
which means the serving process must export token counts, and most do. It is a metric
worth adding for the reason {{ch:inf-cpu-gpu}} gave: it is the only number that says
whether the hardware is being used.

**Why the queue is empty until it is not.** Continuous batching
({{cite:kwon2023pagedattention}}) admits every arriving request into the running batch
while memory permits, so nothing queues until the token-slot budget is exhausted. The
scheduler is doing exactly the right thing and it destroys the queue signal as a side
effect — the same tension {{ch:inf-batching}} noted about admission control.

**What "capacity" means for a replica.** It is the batch at which either the token-slot
budget from {{eq:batch-times-context-is-the-budget}} is exhausted or the latency target
is missed, whichever comes first. Both depend on context length, so **replica capacity
is not a constant** and a trigger expressed as a fraction of it must be recomputed as
the context distribution moves.

**Scale-down is not the mirror of scale-up.** Removing a replica from an affinity-routed
fleet costs the re-prefill of everything it held
({{eq:affinity-fights-balance}}), and removing one mid-generation drops in-flight
sequences unless the system drains gracefully. Graceful drain means refusing new work
while finishing current work, which takes as long as the longest generation — minutes.
So scale-down has its own lag, and an aggressive scale-down policy paired with a slow
scale-up produces oscillation.

**Model-parallel replicas scale in groups.** With tensor parallelism
({{cite:shoeybi2019megatron}}), the scaling unit is the group, so capacity comes in
increments of the parallelism degree and the scheduler must co-schedule all members with
the right topology. A partially-scheduled group is not a partial replica; it is nothing.

**Why the thundering herd is worse here than elsewhere.** A scaling event that starts
ten replicas has them all fetch the same weights from the same source at the same
moment, so the per-replica bandwidth is the source's total divided by ten. That turns a
214-second cold start into something closer to half an hour, and it does so precisely
during the largest scaling events -- the ones that matter. Ordinary web services do not
have this problem because their images are small and cached; here the fetched object is
large enough that concurrent starts genuinely saturate the source. The mitigations are
familiar from other domains -- staggered starts, peer-to-peer distribution between
nodes, or a node-local cache populated ahead of demand -- and none of them is standard
in cluster tooling because no ordinary workload needs them.

**Disaggregation changes the scaling unit.** {{cite:zhong2024distserve}} and
{{cite:patel2023splitwise}} separate prefill and decode, which means the two scale
independently and against different signals — prefill against prompt tokens per second,
decode against concurrent sequences. That is more moving parts and better matched
control, and it is the main operational argument for disaggregation that
{{ch:inf-batching}}'s throughput comparison does not contain.

## 8. Implementation

The first listing measures each conventional signal's range and lead time, and computes
the trigger the cold start requires.

```python {tier=A name=dc1}
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
```

## 9. Practical Example

One replica, as in-flight batch rises to its capacity of 256:

```
   batch   step ms   GPU busy   % of peak     CPU   tokens/s   queue
--------------------------------------------------------------------
       1      4.18       100%        0.3%      4%        239       0
      32      4.18       100%       10.8%      6%       7656       0
     128      4.18       100%       43.4%     12%      30622       0
     256      4.18       100%       86.8%     19%      61244       0
     400      5.67       100%      100.0%     28%      70574     144
```

**GPU-busy is 100% at every load.** CPU reaches 19% at full capacity. Neither of the two
metrics every autoscaler ships with is usable
({{eq:no-conventional-signal-works}}).

```
        signal   at batch 1   at capacity   dynamic range   usable          
----------------------------------------------------------------------------
      GPU busy         1.00          1.00            1.0x   no: no range    
     % of peak         0.00          0.87          256.0x   yes             
           CPU         0.04          0.19            4.8x   no: never high  
      tokens/s       239.23      61244.02          256.0x   yes             
   queue depth         0.00          0.00             inf   no: flat at 0   
```

Queue depth has infinite range and zero lead:

```
  load vs capacity   queue depth   tokens/s   % of peak   anything moved?
--------------------------------------------------------------------------
               50%             0      30622       43.4%                 -
               90%             0      55024       78.0%                 -
              100%             0      61244       86.8%                 -
              110%            25      67225       95.3%             queue
```

**Zero at 90% of capacity.** A signal that is flat until the moment you needed to have
acted is a report, not an alarm.

With a 210-second cold start and load doubling every 5 minutes:

```
          signal   fires at load   load when ready   overshoot       verdict
----------------------------------------------------------------------------
  GPU busy > 80%          always                 -           - always firing
       CPU > 70%           never                 -           -   never fires
 queue depth > 0            100%              162%        62%      too late
tokens/s plateau             92%              149%        49%      too late
 % of peak > 75%             75%              122%        22%      too late
 % of peak > 55%             55%               89%         0%            ok
```

**The working trigger is at 55%, not near capacity** — and that gap is not a chosen
safety margin, it is the reciprocal of growth during the cold start
({{eq:trigger-is-the-reciprocal-of-growth}}).

```
   ramp %/s   growth over spinup   trigger needed   headroom implied
----------------------------------------------------------------------
      0.05%                1.11x              90%                10%
      0.10%                1.23x              81%                19%
      0.23%                1.62x              62%                38%
      0.50%                2.85x              35%                65%
      1.00%                8.08x              12%                88%
```

At 1%/second the trigger is **12%**, meaning **88% of the fleet sits idle**. Above some
ramp rate reactive autoscaling is not slow — it is arithmetically impossible.

```mermaid {#fig:trigger caption="The trigger is the reciprocal of load growth during the cold start. Shortening the cold start moves the trigger multiplicatively; buying capacity does not move it at all."}
flowchart LR
  A["cold start S"] --> B["load grows (1+r)^S"]
  C["ramp rate r"] --> B
  B --> D["trigger at 1/(1+r)^S"]
  D --> E["idle fleet = 1 - trigger"]
  F["shorter S"] -.->|"multiplicative"| D
  G["more replicas"] -.->|"no effect"| D
```

The second listing opens the cold start.

```python {tier=A name=dc2}
"""Where a cold start's minutes actually go, and which of them are avoidable.

The previous listing treated cold start as one number and showed it dominates the
scaling decision. This one opens it up, because the components have very different
costs to remove and a team that attacks the wrong one spends a quarter for nothing
(eq:cold-start-is-mostly-weight-movement).

The finding is that the dominant term is moving weights, that its size is set by where
the weights are stored rather than by anything about the model, and that the standard
container-registry path is the worst available choice by a wide margin.
"""
WEIGHTS_GB = 140.0
# (stage, seconds, whether it scales with model size)
STAGES = [
    ("schedule pod",            8.0,  False),
    ("pull container image",   34.0,  False),
    ("start runtime",          11.0,  False),
    ("fetch weights",           0.0,  True),
    ("load to device memory",   0.0,  True),
    ("capture graphs",         19.0,  False),
    ("warm up and health check", 12.0, False),
]

# Where weights can come from. (source, GB/s achieved)
SOURCES = [
    ("container image layer",  0.35),
    ("object storage",         1.10),
    ("network filesystem",     2.40),
    ("local NVMe",             6.80),
    ("host page cache",       21.00),
]
PCIE_GB_S = 55.0            # host to device


def fetch_s(gb, src_gb_s):
    return gb / src_gb_s


def load_s(gb):
    return gb / PCIE_GB_S


print("Cold start for a %.0f GB model, by stage." % WEIGHTS_GB)
print()
for src, rate in SOURCES:
    total = 0.0
    for name, secs, scales in STAGES:
        if name == "fetch weights":
            secs = fetch_s(WEIGHTS_GB, rate)
        elif name == "load to device memory":
            secs = load_s(WEIGHTS_GB)
        total += secs
    print(f"  weights from {src:<24} {total:>7.1f}s")

print()
print("Breaking down the middle case (object storage):")
print()
print(f"{'stage':>26}{'seconds':>10}{'share':>9}{'scales with model':>20}")
print("-" * 66)
BASE_SRC = 1.10
detail = {}
total = 0.0
for name, secs, scales in STAGES:
    if name == "fetch weights":
        secs = fetch_s(WEIGHTS_GB, BASE_SRC)
    elif name == "load to device memory":
        secs = load_s(WEIGHTS_GB)
    detail[name] = secs
    total += secs
for name, secs, scales in STAGES:
    s = detail[name]
    print(f"{name:>26}{s:>10.1f}{s / total:>9.1%}"
          f"{('yes' if scales else 'no'):>20}")
print("-" * 66)
print(f"{'TOTAL':>26}{total:>10.1f}{1.0:>9.1%}")

print()
print()
print("Weight movement by source. This is the term that dominates and the one")
print("with the widest spread.")
print()
print(f"{'source':>24}{'GB/s':>9}{'fetch s':>10}{'load s':>9}"
      f"{'weight total':>15}{'cold start':>13}")
print("-" * 82)
fixed = sum(s for n, s, sc in STAGES if n not in
            ("fetch weights", "load to device memory"))
bysrc = {}
for src, rate in SOURCES:
    f = fetch_s(WEIGHTS_GB, rate)
    l = load_s(WEIGHTS_GB)
    bysrc[src] = (f, l, f + l, fixed + f + l)
    print(f"{src:>24}{rate:>9.2f}{f:>10.1f}{l:>9.1f}{f + l:>15.1f}"
          f"{fixed + f + l:>13.1f}")

print()
print(f"fixed overhead independent of source: {fixed:.1f}s")

print()
print()
print("By model size, from the best and worst sources.")
print()
print(f"{'model GB':>10}" + "".join(f"{s[0][:14]:>16}" for s in SOURCES))
print("-" * 90)
grid = {}
for gb in (1.5, 14.0, 40.0, 140.0, 400.0):
    row = []
    for src, rate in SOURCES:
        row.append(fixed + fetch_s(gb, rate) + load_s(gb))
    grid[gb] = row
    print(f"{gb:>10.1f}" + "".join(f"{v:>16.1f}" for v in row))
print()
print("(cold start seconds)")

print()
print()
print("What each intervention removes, ranked by seconds bought.")
print()
print(f"{'intervention':>36}{'removes':>10}{'new cold start':>17}"
      f"{'speedup':>10}")
print("-" * 74)
base_total = bysrc["object storage"][3]
INTERVENTIONS = [
    ("bake weights into the image", -1),
    ("pre-pull image to every node", detail["pull container image"]),
    ("skip graph capture", detail["capture graphs"]),
    ("weights on local NVMe", -2),
    ("weights in host page cache", -3),
    ("keep a warm spare", base_total),
]
for label, removed in INTERVENTIONS:
    if removed == -1:
        # Image layer is the slowest source AND makes the pull enormous.
        f = fetch_s(WEIGHTS_GB, 0.35)
        new = fixed + f + load_s(WEIGHTS_GB)
    elif removed == -2:
        new = bysrc["local NVMe"][3]
    elif removed == -3:
        new = bysrc["host page cache"][3]
    elif removed == base_total:
        new = 0.0
    else:
        new = base_total - removed
    sp = (base_total / new) if new > 0 else float("inf")
    print(f"{label:>36}{(base_total - new):>10.1f}{new:>17.1f}"
          f"{('inf' if new == 0 else '%.2fx' % sp):>10}")

print()
print()
print("And what the cold start implies for the headroom from the previous")
print("listing, at a realistic ramp.")
print()
RAMP = 0.0023
print(f"{'weight source':>24}{'cold start s':>14}{'growth over it':>17}"
      f"{'trigger at':>13}{'idle fleet':>13}")
print("-" * 82)
for src, rate in SOURCES:
    cs = bysrc[src][3]
    growth = (1.0 + RAMP) ** cs
    print(f"{src:>24}{cs:>14.1f}{growth:>16.1f}x{1.0 / growth:>13.0%}"
          f"{1.0 - 1.0 / growth:>13.0%}")

print(f"""
The stage breakdown is the first surprise. Of a {total:.0f}-second cold start from
object storage, **{detail['fetch weights'] / total:.0%} is fetching weights** and
{detail['load to device memory'] / total:.0%} is moving them onto the device
(eq:cold-start-is-mostly-weight-movement). Everything a platform team normally
optimises -- pod scheduling, image pull, runtime start -- is
{(detail['schedule pod'] + detail['pull container image'] + detail['start runtime']) / total:.0%}
between them.

The source table is why that matters. Fetching {WEIGHTS_GB:.0f} GB takes
{bysrc['container image layer'][0]:.0f} seconds from a container image layer and
{bysrc['host page cache'][0]:.0f} seconds from the host page cache -- a spread of
{bysrc['container image layer'][0] / bysrc['host page cache'][0]:.0f} times, on the term
that is most of the total.

**Where the weights live is the cold-start decision.** Not the image size, not the
scheduler, not the runtime -- those sum to {fixed:.0f} seconds and do not move.

The intervention table ranks the options by what they actually buy. Baking weights into
the container image is the intuitive move and it is **the worst available choice**: a
container layer delivers {0.35:.2f} GB/s, so it takes the cold start to
{fixed + fetch_s(WEIGHTS_GB, 0.35) + load_s(WEIGHTS_GB):.0f} seconds against object
storage's {base_total:.0f}.

That is worth stating plainly because it is a common instinct. Putting the weights in
the image feels like removing a fetch, and it does -- by moving the same bytes through a
slower path, decompressed layer by layer, on the critical path of every pod start.

Moving weights to local NVMe takes the cold start to
{bysrc['local NVMe'][3]:.0f} seconds; the host page cache takes it to
{bysrc['host page cache'][3]:.0f}. Those are the interventions worth funding, and both
are storage-placement decisions rather than serving ones.

The model-size grid shows the scaling. A {1.5:.1f} GB model cold-starts in
{grid[1.5][1]:.0f} seconds from object storage and a {400.0:.0f} GB one in
{grid[400.0][1]:.0f} -- so **large models do not merely cost more to serve, they cost
more to scale**, and the autoscaling problem from the previous listing gets
proportionally harder with model size.

The last table closes the loop with that listing. At a ramp that doubles load every
{0.693 / RAMP / 60:.0f} minutes, a cold start from the container image forces a trigger
at
{1.0 / ((1 + RAMP) ** bysrc['container image layer'][3]):.0%} of capacity -- meaning
essentially the entire fleet must sit idle. From the host page cache the trigger can sit
at {1.0 / ((1 + RAMP) ** bysrc['host page cache'][3]):.0%}, implying
{1.0 - 1.0 / ((1 + RAMP) ** bysrc['host page cache'][3]):.0%} idle.

**Weight placement and fleet utilisation are the same decision.** A team that cannot
explain why its GPU fleet runs at forty percent should look at where its weights are
stored before it looks at its autoscaler, because the second number is downstream of the
first.""")
```

```
                     stage   seconds    share   scales with model
------------------------------------------------------------------
              schedule pod       8.0     3.7%                   no
      pull container image      34.0    15.9%                   no
             start runtime      11.0     5.1%                   no
             fetch weights     127.3    59.5%                  yes
     load to device memory       2.5     1.2%                  yes
            capture graphs      19.0     8.9%                   no
   warm up and health check      12.0     5.6%                   no
```

**61% is moving weights** ({{eq:cold-start-is-mostly-weight-movement}}); everything a
platform team normally optimises is 25% between them.

```
                  source     GB/s   fetch s   load s   weight total   cold start
----------------------------------------------------------------------------------
   container image layer     0.35     400.0      2.5          402.5         486.5
          object storage     1.10     127.3      2.5          129.8         213.8
      network filesystem     2.40      58.3      2.5           60.8         144.9
              local NVMe     6.80      20.6      2.5           23.1         107.1
         host page cache    21.00       6.7      2.5            9.2          93.2
```

**Baking weights into the container image is the worst available choice** — 486.5
seconds against object storage's 213.8, because a container layer delivers 0.35 GB/s
decompressed on the critical path.

And the loop closes:

```
           weight source  cold start s   growth over it   trigger at   idle fleet
----------------------------------------------------------------------------------
   container image layer         486.5             3.1x          33%          67%
          object storage         213.8             1.6x          61%          39%
      network filesystem         144.9             1.4x          72%          28%
              local NVMe         107.1             1.3x          78%          22%
         host page cache          93.2             1.2x          81%          19%
```

**Weight placement and fleet utilisation are the same decision**
({{eq:weight-placement-sets-utilisation}}) — **67%** idle from a container image against
**19%** from the page cache, with no change to the autoscaler at all.

## 10. Production Considerations

Export percent-of-peak-arithmetic and scale on it. It is computable from token counts
your serving process already has, and it is the only signal with both range and lead.

Compute your trigger from $(1+r)^{-S}$ rather than choosing it. Measure $r$ from your
own traffic and $S$ from your own pods; the number that comes out is not negotiable.

Get the weights off the container image. It is a step change of **273 seconds** in the
worked example, and it is a storage-placement change rather than an engineering project.

Pre-warm the host page cache on nodes that will serve a model. It is the difference
between 93 and 214 seconds, and it costs host memory that is otherwise idle.

Keep a warm spare when the ramp rate makes reactive scaling infeasible. The threshold is
$rS < -\ln\theta_{\min}$, and above it a spare is the cheapest correct answer.

Scale on a forecast where one exists. A forecast has arbitrary lead time; a reactive
controller has none, and no amount of tuning creates lead time that the cold start
consumed.

Drain gracefully and slowly. Scale-down that drops in-flight sequences converts a
capacity decision into user-visible failures, and under affinity routing it also costs
a re-prefill spike.

Co-schedule model-parallel groups atomically. A partially-scheduled tensor-parallel group
provides no capacity and holds devices that another group could have used.

## 11. Common Mistakes

**Scaling on GPU utilisation.** The signal is 100% at idle and 100% at saturation, so
a threshold on it either fires always or never depending on where it is set.

**Scaling on queue depth.** It is zero until you are already late.

**Choosing the trigger by intuition.** It is $(1+r)^{-S}$ and intuition sets it far too
high.

**Baking weights into the container image.** The slowest possible source, on the
critical path of every pod start.

**Optimising pod scheduling and image pull.** Together they are a quarter of the cold
start, and the weight fetch nobody looked at is three fifths of it.

**Treating scale-down as scale-up reversed.** It has its own lag and its own costs.

## 12. Failure Modes

**Oscillation.** Aggressive scale-down paired with slow scale-up removes capacity that
is needed again before it can be replaced, and the cycle repeats at the cold-start
period.

**Thundering herd on weight storage.** Ten replicas starting simultaneously all fetch
140 GB from the same object store, and the achieved per-replica bandwidth collapses —
making a scaling event slower precisely when it is largest.

**Silent capacity change from context drift.** Replica capacity depends on context length
via {{eq:batch-times-context-is-the-budget}}, so a longer-prompt product change lowers
capacity and the trigger, expressed as a fraction, now fires later in absolute terms.
Nothing in the autoscaler changed and nothing in it can detect this.

**Quantised scaling granularity.** Under model parallelism capacity arrives in group-
sized increments, so a fleet forced to round up carries idle capacity the trigger
arithmetic never asked for.

**Partial group scheduling.** A tensor-parallel group missing one member holds devices
and serves nothing, and the cluster autoscaler sees allocated resources.

**Drain-induced prefill spike.** Scaling down an affinity-routed node moves its prefixes
elsewhere, and the resulting prefill burst can trigger a scale-up.

## 13. Alternatives

**Standing capacity with no autoscaling.** Correct whenever $rS$ exceeds the threshold,
and much simpler than a controller that cannot work. It is also easier to reason about
during an incident, which has value the cost comparison does not capture.

**Scheduled scaling to a forecast.** Lowest idle capacity of any approach that works,
because forecast lead time is unbounded. Requires the traffic to be forecastable, which
for most products it substantially is.

**Serverless or scale-to-zero.** Attractive on cost and catastrophic on cold start: the
486-second figure is exactly the scale-to-zero experience for a large model served
from a container image, and it is what the first user after an idle period waits.
Viable for small models from fast storage, where the same arithmetic gives seconds
rather than minutes.

**Multiplexing several models on one replica.** Avoids cold starts by keeping weights
resident for more models than are active, at the cost of
{{eq:batch-times-context-is-the-budget}}'s capacity. Increasingly attractive as model
count grows.

**Smaller models.** Cold start scales with weight bytes, so a smaller model is faster to
scale as well as cheaper to serve — and the grid shows a 1.5 GB model starting in a
fraction of a 400 GB model's time.

## 14. Evaluation

Measure your own cold start end to end, broken down by stage. The decomposition is what
tells you which intervention is worth funding, and almost nobody has it.

Measure the traffic ramp rate at your steepest observed transition, not on average. The
trigger must survive the worst ramp, and the average one will not size it.

Report achieved fleet utilisation against the theoretical $\theta^\star$. A gap means
the trigger is set by something other than arithmetic.

Track replica capacity as a function of the current context distribution rather than as
a constant. It moves, and the trigger moves with it.

Test scale-up under concurrent starts. The thundering-herd bandwidth collapse only
appears when several replicas start at once, which is exactly the case that matters,
and a single-replica cold-start measurement will never reveal it.

Report the scaling granularity alongside the headroom. Under model parallelism the
two compose, and a fleet in groups of eight carries rounding on top of arithmetic.

## 15. Advanced Concepts

The exponential growth model for load is a local approximation and it overstates
sustained ramps badly — real traffic accelerates and then plateaus. A more faithful
treatment would use the peak *derivative* over the cold-start window rather than a
constant rate, which lowers the required headroom for most real traffic shapes. The
qualitative result survives: the trigger is set by how much load can grow during the
cold start, and the cold start is the term you can change.

Cold start and capacity interact through weight residency in an unexplored way. A node
that has served a model recently has its weights in the host page cache, so the *same*
node restarting is far faster than a cold node starting. That makes scale-up
non-uniform — some nodes are warm and some are not — and a scheduler that preferred
previously-warm nodes would cut effective cold start substantially at no hardware cost.
As far as the author is aware, no standard scheduler exposes weight residency as a
scheduling signal, and it would be a cheap and large win.

The trigger formula also assumes a single replica is the scaling increment, which is
false under model parallelism. With a tensor-parallel degree of eight, capacity arrives
in units of eight devices, so the fleet is quantised and the effective headroom is the
computed $1 - \theta^\star$ *plus* whatever rounding the increment forces. For a fleet
of forty devices in groups of eight, the granularity is 20% of the fleet, which is
comparable to the headroom the arithmetic demanded in the first place. **Parallelism
degree is therefore a scaling-granularity decision as well as the bandwidth and
reliability decisions {{ch:inf-parallelism}} and {{ch:inf-distributed}} described** --
a third term in a choice that already had two.

There is a tension between this chapter and {{ch:inf-distributed}} that neither resolves.
Affinity routing wants nodes to be long-lived and specialised, holding particular
prefixes; autoscaling wants them to be interchangeable and disposable. The more
successful the affinity routing, the more expensive every scaling event becomes — and
the more aggressive the autoscaling, the less affinity can accumulate. A fleet that does
both well probably needs a stable core with affinity and an elastic margin without it,
which is a two-tier design neither chapter's model represents.

## 16. Connection to Previous Chapters

{{eq:decode-is-bandwidth-bound}} from {{ch:inf-cpu-gpu}} is the reason GPU utilisation
fails as a signal. The device is fully occupied at a fraction of a percent of its
capability, and the driver correctly reports occupancy.

{{eq:batch-times-context-is-the-budget}} from {{ch:inf-gpu-memory}} defines replica
capacity, which is what every trigger is a fraction of — and which moves with context
length.

{{eq:affinity-fights-balance}} from {{ch:inf-distributed}} makes scale-down expensive and
creates the tension {{sec:15-advanced-concepts}} describes.

{{eq:variance-not-mean-drives-wait}} from {{ch:sd-async}} is why overshoot is costly:
past capacity the queueing term is convex, so being 62% over is much worse than 62%
suggests.

## 17. Exercises

1. Compute $\theta^\star$ for a 90-second cold start and load doubling every 2 minutes.
   What idle fraction does that imply?

2. Derive the maximum ramp rate for which reactive autoscaling holds a 40% idle
   tolerance, at a 150-second cold start.

3. For a model you serve, decompose the cold start by stage. Which intervention buys the
   most seconds?

4. Extend the second listing so ten replicas start concurrently and share source
   bandwidth. How much does the cold start grow?

5. Design a percent-of-peak exporter for a serving process you have access to. What does
   it need from the process, and what does it assume?

## 18. Interview Questions

1. Our GPU utilisation is pinned at 100% and we are not saturated. Explain.

2. Why is queue depth a bad autoscaling signal here when it is a good one elsewhere?

3. How would you choose the scale-up trigger, and what two numbers do you need?

4. Where does a cold start's time actually go for a 140 GB model?

5. A team proposes baking model weights into the container image to speed up starts.
   What do you say?

6. Ten replicas start at once during a traffic spike and each takes four times longer
   than a single start does. What happened, and what would you change?

## 19. Research Questions

1. How much cold start does weight-residency-aware scheduling save on a real fleet, and
   what does it cost in placement quality?

2. What is the right traffic model for setting a trigger — peak derivative, quantile of
   observed ramps, or something else?

3. Can prefill and decode fleets be autoscaled independently in practice, and does the
   improved signal match outweigh the added coordination?

4. What does the two-tier design — stable affinity core plus elastic margin — cost
   relative to either extreme?

## 20. Chapter Summary

Every conventional autoscaling signal fails here. GPU-busy has a dynamic range of
**1.0×** because kernels are resident at any load; CPU reaches **19%** at capacity;
queue depth reads **zero at 90%** of capacity and rises only once you are late
({{eq:no-conventional-signal-works}}). Percent-of-peak-arithmetic works and is not
collected by default.

The trigger is not a choice. With load growing at $r$ and a cold start of $S$, it is
$(1+r)^{-S}$ ({{eq:trigger-is-the-reciprocal-of-growth}}) — **55%** for a 210-second
cold start and load doubling every five minutes. At 1%/second it is **12%**, meaning
**88%** of the fleet idles, and above some ramp rate reactive autoscaling has no
solution rather than a slow one.

Cold start is **61% weight fetching** and **1% device loading**, against **25%** for
everything a platform team normally optimises
({{eq:cold-start-is-mostly-weight-movement}}). Where the weights live sets it: **486.5
seconds** from a container image against **93.2** from the host page cache.

Composing the two gives the operational result. Weight placement determines idle
fleet — **67%** from an image against **19%** from the page cache — with no change to
the autoscaler ({{eq:weight-placement-sets-utilisation}}).

The through-line is that almost nothing in this chapter is a control-theory problem,
which is where the instinct goes. The signals fail for reasons established three
chapters ago about how a GPU reports occupancy; the trigger is fixed by two measured
numbers; the cold start is dominated by a storage-placement decision. A team that
responds to scaling problems by tuning the autoscaler is working on the one part of
the system where there is least to gain.

Carry forward: **the trigger is arithmetic, not judgement**, and **a fleet's utilisation
is set by where its weights are stored**.

## 21. Further Reading

- {{cite:kwon2023pagedattention}} — continuous batching, which admits until memory is
  gone and thereby destroys the queue signal.
- {{cite:zhong2024distserve}} — disaggregation, which gives prefill and decode separate
  scaling signals.
- {{cite:patel2023splitwise}} — heterogeneous fleets, where scaling units differ by
  phase.
- {{cite:shoeybi2019megatron}} — tensor parallelism, which makes the scaling unit a group
  rather than a device.
