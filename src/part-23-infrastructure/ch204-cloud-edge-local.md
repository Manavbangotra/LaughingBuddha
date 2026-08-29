---
id: inf-edge
number: 204
part: XXIII
tier: full
status: draft
requires: [decode-is-bandwidth-bound, trigger-is-the-reciprocal-of-growth,
           weight-placement-sets-utilisation, cache-quantisation-is-the-larger-lever]
provides: [self-hosting-is-a-utilisation-bet, ops-is-most-of-the-fixed-cost,
           device-quality-is-bandwidth-bound, benchmark-is-burst-users-get-sustained]
citations: [kwon2023pagedattention, patel2023splitwise, pope2022inference,
            leviathan2023speculative]
---

## 1. Learning Objectives

By the end of this chapter you will be able to compute the volume at which self-hosting
beats a provider API, and show why the naive break-even understates it by roughly
threefold; identify the operations overhead as the dominant fixed cost rather than the
hardware; explain why a device's binding constraint is memory bandwidth rather than
memory capacity, and compute the largest model a given device can run at reading speed;
distinguish burst from sustained throughput on thermally-limited hardware; and name the
three properties of a deployment choice that no cost comparison expresses.

## 2. Why This Matters

This part has been about making inference fast. This chapter is about where to run it,
and the two halves answer to different constraints entirely.

The cloud question is economic and it has a specific answer. Self-hosting's unit cost is
the hourly rate divided by what you actually push through the hardware, so it is a bet on
utilisation ({{eq:self-hosting-is-a-utilisation-bet}}) — **0.21 per million tokens at
full utilisation against the API's 0.62, and 1.06 at 20%**, which is worse than the API.
{{ch:inf-kubernetes}} already showed utilisation is bounded below one by the trigger
arithmetic.

Adding operations cost gives the real break-even. It is **44,000 million tokens a
month** at a realistic 45% utilisation, against a naive fixed-cost calculation of
**15,124** — the difference being that low utilisation forces more replicas, which
raises fixed cost, which raises the volume needed to justify it. And **the operations
term is 62% of that fixed cost** at a modest estimate
({{eq:ops-is-most-of-the-fixed-cost}}).

The device question is physical. A phone has roughly a seventh of a datacentre GPU's
bandwidth per gigabyte of memory, so an 8B int4 model **fits on every device tested and
runs at 8.7 tokens/second on a flagship phone** — below reading speed
({{eq:device-quality-is-bandwidth-bound}}). And what a benchmark measures is not what a
user gets: a mid-range phone sustains **42%** of its burst rate
({{eq:benchmark-is-burst-users-get-sustained}}).

## 3. Prerequisites

You need {{eq:decode-is-bandwidth-bound}} from {{ch:inf-cpu-gpu}}. The entire device
half of this chapter is that result applied to hardware where bandwidth is scarce
relative to capacity, which inverts which question matters.

{{eq:trigger-is-the-reciprocal-of-growth}} and
{{eq:weight-placement-sets-utilisation}} from {{ch:inf-kubernetes}} supply the
utilisation ceiling that makes self-hosting a bet rather than a calculation.

{{eq:cache-quantisation-is-the-larger-lever}} from {{ch:inf-gpu-memory}} matters
differently here: on a device, quantising weights helps twice, because it reduces both
the footprint and the per-token read.

## 4. Intuitive Explanation

The self-hosting argument is usually made in one line: a token from our own GPU costs a
fraction of what the API charges. That line is arithmetically correct and it omits two
things, both large.

The first is that you rent the GPU by the hour, not by the token. So the cost per token
is the hourly rate divided by the tokens you actually produced in that hour — and if the
machine spent half the hour waiting for traffic, the cost per token doubles.
{{ch:inf-kubernetes}} showed that waiting is not optional: a fleet that must absorb a
traffic ramp has to keep headroom, and the headroom is a computed fraction rather than a
choice. At the utilisation that arithmetic permits, the unit-cost advantage is much
smaller than the datasheet comparison suggests, and at low utilisation it reverses.

The second omission is people. Someone has to operate this: monitor it, be paged by it,
upgrade it, chase the driver bug, recompute the batch size when the context distribution
moves. That cost is a fixed monthly amount, it is invisible in any per-token comparison,
and at a modest estimate it is **most of the fixed cost** — more than the GPU rental.

Put those together and the break-even volume is far higher than instinct suggests. In
{{sec:9-practical-example}}'s figures it is forty-four billion output tokens a month,
which is on the order of a hundred thousand generated answers a day. Most products are
nowhere near it, and the ones that are usually already know.

That is not an argument against self-hosting. It is an argument that the *economic*
case for self-hosting is weaker than it looks, and therefore that most real decisions to
self-host are made for reasons the cost column does not contain — data residency,
latency floors, model control, offline operation. Those are excellent reasons. The
mistake is dressing them in an economic argument that does not survive contact with the
utilisation figure.

The device half of the chapter starts from a different confusion.

The natural question about running a model on a phone is "does it fit". You look at the
model size, you look at the device memory, and the answer is encouraging: quantised
models of respectable size fit on ordinary hardware with room to spare.

That is the wrong first question, and {{ch:inf-cpu-gpu}} explains why. Generating a token
requires reading every weight, so the speed is bandwidth divided by model bytes. A phone
has plenty of memory relative to a model and very little *bandwidth* relative to a
datacentre part — roughly a seventh as much per gigabyte of capacity.

So an eight-billion-parameter model at four bits fits on a phone comfortably, and
produces about nine tokens a second. A person reads faster than that. The model fits and
the product does not work.

And then there is heat. A datacentre GPU runs at its rated clocks indefinitely because
someone engineered the cooling. A phone cannot: it runs fast for a few seconds, warms up,
and throttles. A benchmark that runs for ten seconds measures the fast part. A user
holding the phone for a minute experiences the other one — **less than half** the rate on
a mid-range device.

## 5. Formal Explanation

**Self-hosting economics.** Let a provider charge $\alpha$ per million tokens, a replica
cost $\gamma$ per hour and sustain $\theta$ million tokens per month at full
utilisation, and operations cost $\omega$ per month. At utilisation $u$, one replica
serves $u\theta$ and the unit cost of the hardware alone is

$$ c_{\text{unit}}(u) \;=\; \frac{730\gamma}{u\theta} $$ (eq:self-hosting-is-a-utilisation-bet)

which is **inversely proportional to utilisation** and crosses $\alpha$ at
$u^\times = 730\gamma/(\alpha\theta)$.

For a volume $V$ million tokens per month, the number of replicas is
$n = \lceil V/(u\theta) \rceil$ and total cost is $730\gamma n + \omega$. Self-hosting
wins when

$$ 730\gamma\left\lceil \frac{V}{u\theta} \right\rceil + \omega \;<\; \alpha V $$

Ignoring the ceiling gives the naive break-even $V_0 = (730\gamma + \omega)/\alpha$. The
ceiling matters because low $u$ forces $n > 1$ well before $V_0$, so the true crossover
$V^\star > V_0$, and the gap grows as $u$ falls. {{sec:9-practical-example}} measures
$V_0 = 15{,}124$ and $V^\star = 44{,}000$ at $u = 0.45$.

The operations share of fixed cost is

$$ \frac{\omega}{730\gamma + \omega} $$ (eq:ops-is-most-of-the-fixed-cost)

which exceeds one half whenever $\omega > 730\gamma$ — that is, whenever operations cost
more than about **3,577** a month, which one part-time engineer does.

**Device constraints.** From {{eq:decode-is-bandwidth-bound}}, decode throughput on a
device with bandwidth $B_d$ serving a model of $W$ weight bytes is $B_d/W$ tokens per
second. Requiring at least a reading rate $R$:

$$ W \;\le\; \frac{B_d}{R} \quad\text{while capacity requires}\quad W \;\le\; M_d - \Omega $$ (eq:device-quality-is-bandwidth-bound)

The binding constraint is whichever is smaller. Datacentre hardware has
$B/M \approx 42$ GB/s per GB; a phone has $\approx 5.7$. So for
$R = 12$ tokens/second the bandwidth constraint binds whenever
$B_d/M_d < R$, which is true for every device in {{sec:9-practical-example}}'s
table and false for a datacentre GPU.

**Thermal derating.** Sustained throughput is $\sigma B_d/W$ for a sustained fraction
$\sigma < 1$, so

$$ \frac{\text{sustained}}{\text{burst}} \;=\; \sigma $$ (eq:benchmark-is-burst-users-get-sustained)

and $\sigma$ falls with device size — **0.42** on a mid-range phone against **0.95** on
a workstation GPU.

## 6. Mathematical Foundation

The gap between $V_0$ and $V^\star$ deserves derivation, because it is the chapter's
most useful correction and it is invisible in the naive comparison.

At utilisation $u$, serving $V$ needs $n = \lceil V/(u\theta)\rceil$ replicas. Ignoring
the ceiling, self-hosting cost is $730\gamma V/(u\theta) + \omega$, which is linear in
$V$ with slope $730\gamma/(u\theta) = c_{\text{unit}}(u)$. Self-hosting wins when

$$ V\left(\alpha - c_{\text{unit}}(u)\right) \;>\; \omega \quad\Longrightarrow\quad V^\star \;=\; \frac{\omega}{\alpha - c_{\text{unit}}(u)} $$

**This has no solution when $c_{\text{unit}}(u) \ge \alpha$** — below the crossover
utilisation $u^\times$, no volume makes self-hosting cheaper, because every marginal
token costs more than buying it. That is a stronger statement than "the break-even is
high": at low enough utilisation there is no break-even at all.

And as $u \to u^\times$ from above, $V^\star \to \infty$. The break-even is therefore
extremely sensitive to utilisation near the crossover, which is exactly the region most
deployments occupy. A utilisation estimate that is optimistic by ten points can move the
break-even by a large multiple, and utilisation estimates are usually optimistic.

For the device side, combining the two constraints gives the largest usable model:

$$ W_{\max} \;=\; \min\left(\frac{\sigma B_d}{R},\; M_d - \Omega\right) $$

and the ratio of the two terms is $\sigma B_d/(R(M_d - \Omega))$ — essentially the
bandwidth-per-gigabyte figure divided by the reading rate. **Whenever bandwidth per
gigabyte is below the target token rate, bandwidth binds**, and a phone's 5.7 against a
target of 12 means it always does.

That is a clean design rule: compute $B_d/M_d$ for the target device, compare to the
token rate you need, and if it is smaller then capacity is irrelevant and the model size
is set by $\sigma B_d/R$.

## 7. Internal Mechanics

**Why the API's price is not the API's cost.** A provider runs at utilisation a single
tenant cannot reach, because aggregating many customers' traffic smooths the ramp that
{{eq:trigger-is-the-reciprocal-of-growth}} forces headroom for. That is the structural
advantage, and it is why the price can be below a self-hoster's unit cost while still
being profitable. **The provider is selling statistical multiplexing**, and a
single-tenant deployment cannot manufacture it.

**Where the operations cost actually goes.** Not to keeping the process running, which is
largely automatic, but to the recurring decisions this part has enumerated: recomputing
batch size when the context distribution moves, re-deriving the parallelism degree after
a model change, chasing a kernel regression after a driver upgrade, and being paged when
a tensor-parallel group loses a member. Each is a few hours; together they are a standing
fraction of an engineer.

**Why quantisation helps twice on a device.** It reduces $W$, which by
{{eq:device-quality-is-bandwidth-bound}} raises token rate proportionally *and* frees
capacity for context. In the datacentre {{eq:cache-quantisation-is-the-larger-lever}}
found weight quantisation a minor capacity lever; on a device it is the primary
throughput lever, because the binding constraint is different.

**Prefill on a device is a different problem from decode.**
{{cite:pope2022inference}}'s split means a long prompt is compute-bound, and a phone's
compute is weak in absolute terms even though its bandwidth is the usual constraint. So
on-device time-to-first-token for a long prompt can be poor even when token rate is
acceptable — the two device metrics do not move together.

**Speculative decoding is unusually attractive on-device.**
{{cite:leviathan2023speculative}} raises tokens per weight-read, which is exactly the
scarce quantity here, and the draft model's memory cost is small. It is also easier on a
device than in a datacentre because there is no batch to compete with — the arithmetic
units are idle in a way {{ch:inf-cpu-gpu}} showed batching normally fills.

**Thermal behaviour is workload-shaped.** A device that generates for three seconds and
idles for thirty never throttles; one generating continuously does. So the sustained
fraction is not a device constant but a function of duty cycle, and a product with bursty
generation gets closer to burst rates than the table suggests.

**Why the ceiling function matters more than it looks.** Replicas are integers, and at
low utilisation each one covers a small slice of demand, so the rounding penalty is large
relative to the step. At 45% utilisation a replica covers 7,582 million tokens; a
deployment serving 8,000 buys two replicas and wastes most of the second. That waste is
not a rounding detail -- it is the difference between the naive break-even and the real
one, and it is why the crossover in {{sec:9-practical-example}} sits at 44,000 rather
than 15,124. **The finer the increment, the smaller the penalty**, which is a quiet
argument for smaller replicas and against the large tensor-parallel groups
{{ch:inf-parallelism}}'s bandwidth analysis favours.

**Edge appliances sit between.** {{cite:patel2023splitwise}}'s heterogeneity argument
applies at the edge too: a small appliance can hold a model resident with no cold start
({{eq:weight-placement-sets-utilisation}}) and serve a site's traffic at low utilisation,
which is economically poor and operationally excellent when the alternative is a network
round trip.

## 8. Implementation

The first listing computes the self-hosting crossover including operations cost and
replica rounding.

```python {tier=A name=dd1}
"""Self-hosting beats an API at a specific utilisation, and it is higher than it looks.

The economic case for running your own inference is usually made on unit cost: a token
from your own GPU is cheaper than a token from an API. That is true at full utilisation
and only at full utilisation, because a rented GPU is billed by the hour whether or not
it is decoding (eq:self-hosting-is-a-utilisation-bet).

ch:inf-kubernetes established that utilisation is bounded by the trigger arithmetic --
a fleet that must absorb a ramp cannot run full. So the two results compose into a
crossover, and this listing computes where it sits.
"""
API_PER_MTOK = 0.62          # what a provider charges per million output tokens
GPU_PER_HOUR = 4.90
TOKENS_PER_SEC = 6500.0      # SUSTAINED, including prefill and real batch mix
SECONDS_PER_MONTH = 30.0 * 24.0 * 3600.0
OPS_PER_MONTH = 5800.0       # engineer time, monitoring, on-call, upgrades

MAX_TOK_MONTH = TOKENS_PER_SEC * SECONDS_PER_MONTH / 1e6   # millions


def self_cost(mtok_month, utilisation, replicas=None):
    """Monthly cost of self-hosting `mtok_month` million tokens."""
    capacity_per_replica = MAX_TOK_MONTH * utilisation
    if replicas is None:
        import math
        replicas = max(1, int(math.ceil(mtok_month / capacity_per_replica)))
    return replicas * GPU_PER_HOUR * 730.0 + OPS_PER_MONTH, replicas


print("One replica sustains %.0f output tokens/sec on a real request mix -- well"
      % TOKENS_PER_SEC)
print("below ch:inf-batching's peak, because production batches are not full and")
print("prefill shares the device. That is %.0f million tokens a month at"
      % MAX_TOK_MONTH)
print("100%% utilisation. A GPU costs %.2f an hour; ops overhead is %.0f a month."
      % (GPU_PER_HOUR, OPS_PER_MONTH))
print()
print("Unit cost of self-hosting, by utilisation. The API charges %.2f."
      % API_PER_MTOK)
print()
print(f"{'utilisation':>13}{'Mtok/month':>14}{'GPU cost':>11}"
      f"{'per Mtok':>11}{'vs API':>10}")
print("-" * 60)
unit = {}
for u in (1.00, 0.80, 0.60, 0.40, 0.20, 0.10):
    tok = MAX_TOK_MONTH * u
    gpu = GPU_PER_HOUR * 730.0
    per = gpu / tok
    unit[u] = (tok, gpu, per)
    print(f"{u:>13.0%}{tok:>14.0f}{gpu:>11.0f}{per:>11.4f}"
          f"{per / API_PER_MTOK:>9.2f}x")
print()
print("(GPU cost only -- ops overhead is added per month, not per token)")

print()
print()
print("Total monthly cost including ops, by volume. This is the comparison that")
print("decides it, and the ops term is what moves the crossover.")
print()
print(f"{'Mtok/month':>13}{'API cost':>11}{'replicas':>10}{'self cost':>12}"
      f"{'cheaper':>10}{'ratio':>9}")
print("-" * 66)
UTIL = 0.45                  # what ch:inf-kubernetes's trigger arithmetic permits
cross = None
tab = {}
for mtok in (5, 100, 1600, 6400, 25600, 51200, 102400):
    api = mtok * API_PER_MTOK
    sc, reps = self_cost(mtok, UTIL)
    tab[mtok] = (api, sc, reps)
    who = "API" if api < sc else "self"
    if cross is None and sc < api:
        cross = mtok
    print(f"{mtok:>13}{api:>11.0f}{reps:>10}{sc:>12.0f}{who:>10}"
          f"{max(api, sc) / min(api, sc):>8.2f}x")

# The true crossover, searched, accounting for replica rounding.
true_cross = None
m = 100
while m < 1000000:
    sc, _ = self_cost(m, UTIL)
    if sc < m * API_PER_MTOK:
        true_cross = m
        break
    m += 100
print()
print(f"crossover at {UTIL:.0%} utilisation: {true_cross} Mtok/month")
print(f"one-replica break-even (ignoring rounding): "
      f"{(GPU_PER_HOUR * 730.0 + OPS_PER_MONTH) / API_PER_MTOK:.0f} Mtok/month")

print()
print()
print("Where exactly, by utilisation. The break-even volume is the volume at")
print("which the API bill equals the fleet plus ops.")
print()
print(f"{'utilisation':>13}{'Mtok/replica':>15}{'break-even Mtok':>18}"
      f"{'as % of one replica':>22}{'replicas':>12}")
print("-" * 82)
be = {}
for u in (1.00, 0.80, 0.60, 0.45, 0.30, 0.15):
    cap = MAX_TOK_MONTH * u
    # One replica: cost = GPU*730 + OPS. Break even when mtok*API = that.
    fixed = GPU_PER_HOUR * 730.0 + OPS_PER_MONTH
    bev = fixed / API_PER_MTOK
    be[u] = (cap, bev, bev / cap)
    print(f"{u:>13.0%}{cap:>15.0f}{bev:>18.0f}{bev / cap:>21.0%}"
          f"{('1' if bev <= cap else '%.0f' % (bev / cap + 0.999)):>12}")

print()
print("(break-even is the same volume at every utilisation -- the fixed cost does")
print(" not change -- but whether ONE replica can serve it does)")

print()
print()
print("What the ops term does. It is the part teams underestimate, and it sets")
print("the floor on how small a self-hosted deployment can sensibly be.")
print()
print(f"{'ops per month':>15}{'break-even Mtok':>18}{'API bill there':>17}"
      f"{'ops share':>12}")
print("-" * 64)
opstab = {}
for ops in (0.0, 1500.0, 5800.0, 14000.0, 40000.0):
    fixed = GPU_PER_HOUR * 730.0 + ops
    bev = fixed / API_PER_MTOK
    opstab[ops] = (bev, ops / fixed)
    print(f"{ops:>15.0f}{bev:>18.0f}{bev * API_PER_MTOK:>17.0f}"
          f"{ops / fixed:>12.0%}")

print()
print()
print("And the reason the comparison is not only about money: three properties")
print("the price does not capture.")
print()
print(f"{'deployment':>22}{'per Mtok':>11}{'data leaves':>14}"
      f"{'p50 latency':>14}{'works offline':>15}")
print("-" * 78)
OPTIONS = [
    ("provider API",        API_PER_MTOK, "yes",   140.0, "no"),
    ("self-host, cloud",    unit[UTIL][2] if UTIL in unit else
     GPU_PER_HOUR * 730.0 / (MAX_TOK_MONTH * UTIL), "no", 95.0, "no"),
    ("self-host, on-prem",  0.31,         "no",    60.0,  "no"),
    ("edge appliance",      0.94,         "no",    45.0,  "yes"),
    ("on-device",           0.00,         "no",   310.0,  "yes"),
]
for label, per, leaves, lat, offline in OPTIONS:
    print(f"{label:>22}{per:>11.4f}{leaves:>14}{lat:>13.0f}m{offline:>15}")

print(f"""
The unit-cost table is the argument as usually made, and at the top of it the argument
is correct. At {1.0:.0%} utilisation a self-hosted replica delivers a million tokens for
{unit[1.0][2]:.4f} against the API's {API_PER_MTOK:.2f} --
{API_PER_MTOK / unit[1.0][2]:.0f} times cheaper.

Read down the column. At {0.40:.0%} utilisation it is {unit[0.4][2]:.4f}; at
{0.10:.0%} it is {unit[0.1][2]:.4f}, which is
{unit[0.1][2] / API_PER_MTOK:.2f} times the API price
(eq:self-hosting-is-a-utilisation-bet).

**A rented GPU bills by the hour whether or not it is decoding.** So the unit cost is
the hourly rate divided by whatever you actually put through it, and ch:inf-kubernetes
showed that "whatever you actually put through it" is bounded well below one by the
trigger arithmetic -- {1 - 0.45:.0%} of the fleet idle at a realistic ramp and cold
start.

The total-cost table adds the term the unit comparison omits. Self-hosting costs
{OPS_PER_MONTH:.0f} a month in engineer time, monitoring, on-call and upgrades before
a single token is served, and that is a fixed cost the API does not have.

At {5} million tokens a month the API costs {tab[5][0]:.0f} and self-hosting
{tab[5][1]:.0f} -- **{tab[5][1] / tab[5][0]:.0f} times more**. At {6400} million it is
{tab[6400][0]:.0f} against {tab[6400][1]:.0f}, still favouring the API. Self-hosting
first wins at **{true_cross} million tokens a month**.

To put that in perspective: {true_cross} million output tokens is roughly
{true_cross * 1e6 / 400 / 30 / 1e3:.0f} thousand generated answers a day at four hundred
tokens each. **Most products are nowhere near it**, and the ones that are usually know.

The break-even table separates two things that get conflated. The *fixed-cost*
break-even -- one replica plus ops against the API bill -- is
**{be[1.0][1]:.0f} million tokens a month**, and it does not move with utilisation
because fixed costs do not.

What utilisation changes is whether one replica can actually carry that volume. At
{1.0:.0%} utilisation the break-even volume is {be[1.0][2]:.0%} of a replica's capacity,
so one replica suffices. At {0.45:.0%} it is {be[0.45][2]:.0%} -- three replicas -- and
at {0.15:.0%}, {be[0.15][2]:.0%}, which is seven.

**Every replica added to cover the utilisation shortfall raises the break-even again**,
which is why the true crossover at {UTIL:.0%} utilisation is {true_cross} rather than
{be[1.0][1]:.0f}. The two compound: low utilisation means more replicas, more replicas
mean more fixed cost, more fixed cost means a higher volume is needed to justify it.

The ops table is where estimates go wrong, and it is worth dwelling on because the term
is invisible in every unit-cost comparison. At {0.0:.0f} a month of ops -- the implicit
assumption when someone compares token prices -- break-even is
{opstab[0.0][0]:.0f} million tokens. At a realistic {5800.0:.0f} it is
{opstab[5800.0][0]:.0f}. At {40000.0:.0f}, which is one experienced engineer, it is
{opstab[40000.0][0]:.0f}.

**The ops term is {opstab[5800.0][1]:.0%} of the fixed cost at a modest estimate and
{opstab[40000.0][1]:.0%} at a realistic one.** Self-hosting is mostly a payroll decision
wearing an infrastructure costume, and the GPU is the cheap part.

The last table is the reason none of this settles the question. An API is cheapest below
break-even and it sends your data somewhere else. On-device inference has no marginal
cost at all and a {310.0:.0f}ms latency that reflects a bandwidth-starved device
(ch:inf-cpu-gpu). An edge appliance works with no network.

**Three of the five options are chosen for reasons the cost column cannot express** --
data residency, offline operation, and latency floor -- and for those, the economics
determine only whether you can afford the choice you already had to make. The honest use
of this table is to price a constraint, not to select from it.""")
```

## 9. Practical Example

Unit cost against a 0.62 API price:

```
  utilisation    Mtok/month   GPU cost   per Mtok    vs API
------------------------------------------------------------
         100%         16848       3577     0.2123     0.34x
          80%         13478       3577     0.2654     0.43x
          60%         10109       3577     0.3539     0.57x
          40%          6739       3577     0.5308     0.86x
          20%          3370       3577     1.0616     1.71x
          10%          1685       3577     2.1231     3.42x
```

**At 20% utilisation self-hosting costs 1.71× the API**
({{eq:self-hosting-is-a-utilisation-bet}}). The hourly rate is paid whether or not the
device decodes.

Total cost including operations, at the 45% utilisation
{{eq:trigger-is-the-reciprocal-of-growth}} permits:

```
   Mtok/month   API cost  replicas   self cost   cheaper    ratio
------------------------------------------------------------------
            5          3         1        9377       API 3024.84x
          100         62         1        9377       API  151.24x
         1600        992         1        9377       API    9.45x
         6400       3968         1        9377       API    2.36x
        25600      15872         4       20108       API    1.27x
        51200      31744         7       30839      self    1.03x
       102400      63488        14       55878      self    1.14x
```

The crossover is **44,000 million tokens a month** — against a naive fixed-cost
break-even of **15,124**. Low utilisation forces more replicas, which raises fixed cost,
which raises the volume needed to justify it.

```
  utilisation   Mtok/replica   break-even Mtok   as % of one replica    replicas
----------------------------------------------------------------------------------
         100%          16848             15124                  90%           1
          80%          13478             15124                 112%           2
          60%          10109             15124                 150%           2
          45%           7582             15124                 199%           3
          30%           5054             15124                 299%           4
          15%           2527             15124                 598%           7
```

And the term that dominates:

```
  ops per month   break-even Mtok   API bill there   ops share
----------------------------------------------------------------
              0             5769             3577           0%
           1500             8188             5077          30%
           5800            15124             9377          62%
          14000            28350            17577          80%
          40000            70608            43777          92%
```

**Operations is 62% of fixed cost at a modest estimate and 92% at one experienced
engineer** ({{eq:ops-is-most-of-the-fixed-cost}}). Self-hosting is mostly a payroll
decision; the GPU is the cheap part.

```mermaid {#fig:crossover caption="Self-hosting's unit cost is the hourly rate over realised throughput, so low utilisation forces replicas, replicas raise fixed cost, and fixed cost raises the break-even. The three compound."}
flowchart TD
  A["traffic ramp"] --> B["headroom required<br/>ch:inf-kubernetes"]
  B --> C["utilisation ~45%"]
  C --> D["unit cost rises"]
  C --> E["more replicas for a volume"]
  E --> F["fixed cost rises"]
  D --> G["break-even 44,000 Mtok"]
  F --> G
  H["ops 62% of fixed"] --> G
```

The second listing turns to devices.

```python {tier=A name=dd2}
"""On a device, the constraint is bandwidth and heat -- not capacity, and not FLOPs.

The usual question about on-device inference is "does the model fit". That is the wrong
first question. ch:inf-cpu-gpu established decode is bandwidth-bound, and a phone's
memory bandwidth is two orders of magnitude below a datacentre GPU's while its capacity
is only one order below.

So a model that fits comfortably can still be unusably slow, and the achievable quality
at a given speed is set by bandwidth rather than by memory
(eq:device-quality-is-bandwidth-bound).

There is a second constraint with no datacentre analogue: sustained throughput is
limited by heat, so the number a benchmark reports is not the number a user gets after
the first minute.
"""
# (device, memory GB, bandwidth GB/s, sustained fraction of burst after thermals)
DEVICES = [
    ("phone, mid",        6.0,   34.0, 0.42),
    ("phone, flagship",  12.0,   68.0, 0.51),
    ("laptop, integrated", 24.0, 120.0, 0.63),
    ("laptop, discrete",  16.0,  480.0, 0.78),
    ("workstation GPU",   48.0,  960.0, 0.95),
]
# (model, params B, bytes per param at the quantisation that fits)
MODELS = [
    ("0.5B int4",   0.5, 0.5),
    ("3B int4",     3.0, 0.5),
    ("8B int4",     8.0, 0.5),
    ("8B fp16",     8.0, 2.0),
    ("32B int4",   32.0, 0.5),
]
OVERHEAD_GB = 1.4          # runtime, KV cache, the rest of the operating system
READ_TARGET = 12.0         # tokens/sec a person finds tolerable to read


def weights_gb(params_b, bytes_per):
    return params_b * bytes_per


def burst_tps(params_b, bytes_per, bw_gb_s):
    """Decode tokens/sec: bandwidth over bytes read per token."""
    return bw_gb_s / weights_gb(params_b, bytes_per)


print("Devices, and what each can hold and move.")
print()
print(f"{'device':>20}{'memory GB':>12}{'GB/s':>9}{'sustained':>12}"
      f"{'bandwidth per GB':>19}")
print("-" * 74)
for name, mem, bw, sus in DEVICES:
    print(f"{name:>20}{mem:>12.1f}{bw:>9.0f}{sus:>12.0%}{bw / mem:>19.1f}")

print()
print("A datacentre GPU is 80 GB at 3350 GB/s -- 42 GB/s per GB of memory.")
print("Note that column: devices are memory-rich relative to their bandwidth.")

print()
print()
print("What fits, by model and device. Capacity is the question people ask.")
print()
print(f"{'model':>12}{'weights GB':>13}" +
      "".join(f"{d[0][:12]:>14}" for d in DEVICES))
print("-" * 96)
fits = {}
for m, p, b in MODELS:
    w = weights_gb(p, b)
    row = {}
    cells = ""
    for name, mem, bw, sus in DEVICES:
        ok = (w + OVERHEAD_GB) <= mem
        row[name] = ok
        cells += f"{('fits' if ok else 'no'):>14}"
    fits[m] = row
    print(f"{m:>12}{w:>13.1f}{cells}")

print()
print()
print("What it RUNS at, which is the question that decides the product.")
print("Sustained tokens/sec after thermal throttling.")
print()
print(f"{'model':>12}" + "".join(f"{d[0][:12]:>14}" for d in DEVICES))
print("-" * 84)
speed = {}
for m, p, b in MODELS:
    row = {}
    cells = ""
    for name, mem, bw, sus in DEVICES:
        if not fits[m][name]:
            row[name] = 0.0
            cells += f"{'-':>14}"
            continue
        t = burst_tps(p, b, bw) * sus
        row[name] = t
        cells += f"{t:>14.1f}"
    speed[m] = row
    print(f"{m:>12}{cells}")

print()
print()
print("The largest model each device can run AT READING SPEED (%.0f tok/s)."
      % READ_TARGET)
print()
print(f"{'device':>20}{'largest that fits':>20}{'largest that is usable':>25}"
      f"{'gap':>18}")
print("-" * 84)
usable = {}
for name, mem, bw, sus in DEVICES:
    big_fit = None
    big_use = None
    for m, p, b in MODELS:
        if fits[m][name]:
            big_fit = m
        if fits[m][name] and speed[m][name] >= READ_TARGET:
            big_use = m
    usable[name] = (big_fit, big_use)
    gap = "same" if big_fit == big_use else "smaller"
    print(f"{name:>20}{str(big_fit):>20}{str(big_use or 'none'):>25}{gap:>18}")

print()
print()
print("Burst against sustained. A benchmark measures the first column; a user")
print("after sixty seconds experiences the second.")
print()
print(f"{'device':>20}{'burst tok/s':>14}{'sustained':>12}{'drop':>9}"
      f"{'burst usable?':>16}{'sustained usable?':>20}")
print("-" * 92)
TESTM = "3B int4"
p, b = 3.0, 0.5
therm = {}
for name, mem, bw, sus in DEVICES:
    burst = burst_tps(p, b, bw)
    sust = burst * sus
    therm[name] = (burst, sust, 1.0 - sus)
    print(f"{name:>20}{burst:>14.1f}{sust:>12.1f}{1.0 - sus:>8.0%}"
          f"{('yes' if burst >= READ_TARGET else 'no'):>16}"
          f"{('yes' if sust >= READ_TARGET else 'no'):>20}")

print()
print()
print("And the bandwidth a device would need to run each model at reading speed.")
print()
print(f"{'model':>12}{'GB read/token':>16}{'GB/s needed':>14}"
      f"{'cheapest device that has it':>32}")
print("-" * 76)
need = {}
for m, p, b in MODELS:
    w = weights_gb(p, b)
    req = w * READ_TARGET
    who = [d[0] for d in DEVICES if d[1] * d[3] >= 0 and d[2] * d[3] >= req]
    need[m] = (w, req, who[0] if who else None)
    print(f"{m:>12}{w:>16.2f}{req:>14.0f}"
          f"{(who[0] if who else 'none in this table'):>32}")

print(f"""
The device table has the number that reframes the problem. A datacentre GPU offers
{3350.0 / 80.0:.0f} GB/s per gigabyte of memory. A flagship phone offers
{68.0 / 12.0:.1f}; a mid-range one {34.0 / 6.0:.1f}.

**Devices are memory-rich relative to their bandwidth**, which is the opposite of the
datacentre balance and it inverts which constraint binds
(eq:device-quality-is-bandwidth-bound).

The fits table is the question people ask, and it is close to uninformative. An
{8.0:.0f}B model at int4 is {weights_gb(8.0, 0.5):.1f} GB and fits on
{sum(1 for d in DEVICES if fits['8B int4'][d[0]])} of the
{len(DEVICES)} devices. The capacity question has an encouraging answer nearly
everywhere.

The speed table is the question that decides the product. That same
{8.0:.0f}B int4 model runs at {speed['8B int4']['phone, flagship']:.1f} tokens a second
on a flagship phone -- it fits, comfortably, and it is
{READ_TARGET / speed['8B int4']['phone, flagship']:.1f} times slower than reading speed.

**Fitting and running are different questions with different answers**, and the second
one is not asked nearly often enough.

The usable-model table states the gap directly. A flagship phone can hold
`{usable['phone, flagship'][0]}` and can usefully run
`{usable['phone, flagship'][1]}`. A mid-range phone holds
`{usable['phone, mid'][0]}` and usefully runs `{usable['phone, mid'][1]}`.

That is a product decision disguised as a hardware fact. **The model you can ship
on-device is set by bandwidth, and it is roughly one size class below what fits.**

The thermal table adds a constraint with no datacentre analogue. A phone cannot sustain
its burst clocks: the flagship holds {DEVICES[1][3]:.0%} of burst and the mid-range
{DEVICES[0][3]:.0%}, against a workstation GPU's {DEVICES[4][3]:.0%}.

For a {TESTM} model that is {therm['phone, mid'][0]:.1f} tokens a second in burst and
{therm['phone, mid'][1]:.1f} sustained on a mid-range phone -- and the burst number is
what a benchmark reports, because a benchmark runs for seconds.

**A device benchmark and a device experience differ by the thermal fraction**, and the
difference is largest on the smallest devices, which is exactly where the margin was
thinnest to begin with.

The requirement table gives the design rule. Running a model at reading speed needs
bandwidth equal to its weight bytes times the token rate: {need['3B int4'][1]:.0f} GB/s
for a {3.0:.0f}B int4 model, {need['8B int4'][1]:.0f} for an
{8.0:.0f}B, and {need['32B int4'][1]:.0f} for a {32.0:.0f}B.

Those are the numbers to check against a target device before committing to a model
size, and they explain the industry's convergence on small quantised models for
on-device work. It is not that larger models do not fit. **It is that bandwidth per
gigabyte on a phone is {(3350.0 / 80.0) / (68.0 / 12.0):.0f} times lower than a
datacentre GPU's**, so the model that runs at an acceptable speed is a size class or two
below the one that fits -- and quantisation helps
twice, by shrinking both the footprint and the per-token read.""")
```

```
              device   memory GB     GB/s   sustained   bandwidth per GB
--------------------------------------------------------------------------
          phone, mid         6.0       34         42%                5.7
     phone, flagship        12.0       68         51%                5.7
  laptop, integrated        24.0      120         63%                5.0
    laptop, discrete        16.0      480         78%               30.0
     workstation GPU        48.0      960         95%               20.0
```

A datacentre GPU offers **42 GB/s per GB**; a phone offers **5.7**. **Devices are
memory-rich relative to their bandwidth**, which inverts which constraint binds.

Capacity says everything fits:

```
       model   weights GB    phone, mid  phone, flags  laptop, inte  laptop, disc
------------------------------------------------------------------------------------
   0.5B int4          0.2          fits          fits          fits          fits
     3B int4          1.5          fits          fits          fits          fits
     8B int4          4.0          fits          fits          fits          fits
```

Speed says otherwise:

```
       model    phone, mid  phone, flags  laptop, inte  laptop, disc  workstation 
------------------------------------------------------------------------------------
   0.5B int4          57.1         138.7         302.4        1497.6        3648.0
     3B int4           9.5          23.1          50.4         249.6         608.0
     8B int4           3.6           8.7          18.9          93.6         228.0
```

An 8B int4 model fits on every device and runs at **8.7 tokens/second** on a flagship
phone — **1.4× slower than reading speed**
({{eq:device-quality-is-bandwidth-bound}}).

```
              device   largest that fits   largest that is usable               gap
------------------------------------------------------------------------------------
          phone, mid             8B int4                0.5B int4           smaller
     phone, flagship             8B int4                  3B int4           smaller
  laptop, integrated            32B int4                  8B int4           smaller
    laptop, discrete             8B int4                  8B int4              same
     workstation GPU            32B int4                 32B int4              same
```

**A mid-range phone holds an 8B model and can usefully run a 0.5B one** — a gap of two
size classes, and it is the bandwidth that decides.

And the benchmark gap:

```
              device   burst tok/s   sustained     drop   burst usable?   sustained usable?
--------------------------------------------------------------------------------------------
          phone, mid          22.7         9.5      58%             yes                  no
     phone, flagship          45.3        23.1      49%             yes                 yes
  laptop, integrated          80.0        50.4      37%             yes                 yes
    laptop, discrete         320.0       249.6      22%             yes                 yes
     workstation GPU         640.0       608.0       5%             yes                 yes
```

A mid-range phone bursts at **22.7** and sustains **9.5** — **usable in a benchmark and
not usable in use** ({{eq:benchmark-is-burst-users-get-sustained}}).

## 10. Production Considerations

Compute your utilisation before computing your unit cost. The unit cost is meaningless
without it, and {{eq:trigger-is-the-reciprocal-of-growth}} gives the ceiling.

Include operations in the comparison at a real number. At 62% of fixed cost it is the
larger term, and every unit-cost comparison omits it.

Check whether $c_{\text{unit}}(u) < \alpha$ before looking for a break-even volume. Below
the crossover utilisation there is no volume that works.

State the non-economic reason if there is one. Data residency, latency floor, offline
operation and model control are sufficient reasons on their own; wrapping them in a cost
argument that does not hold weakens a decision that was correct anyway.

For devices, compute bandwidth per gigabyte and compare it to your target token rate. If
it is smaller, capacity is irrelevant and the model size is $\sigma B_d/R$.

Measure sustained rather than burst, over a duty cycle that matches the product. A
ten-second benchmark on a phone measures a state the user will not be in.

Quantise aggressively on-device. It raises token rate proportionally as well as freeing
capacity, which is a stronger argument than the datacentre one.

## 11. Common Mistakes

**Comparing token prices.** The comparison is fixed cost plus utilisation-adjusted unit
cost against the API bill, and the two headline prices are the least informative pair
of numbers available.

**Omitting operations cost.** It is most of the fixed cost, and it is the term a
per-token price comparison structurally cannot contain.

**Assuming a break-even exists.** Below the crossover utilisation there is no volume at
which self-hosting wins, because every marginal token costs more than buying it.

**Asking whether the model fits on the device.** The binding constraint is bandwidth,
and the fit question has an encouraging answer that means nothing.

**Quoting a device benchmark as a user experience.** Benchmarks measure burst.

**Assuming a smaller device merely means a slower version of the same product.** It
means a different, smaller model.

## 12. Failure Modes

**Optimistic utilisation in the business case.** The break-even is hyperbolic in
utilisation near the crossover, so a ten-point error moves it by a multiple.

**Operations cost discovered after migration.** The team that built it is now the team
that runs it, and the capacity that funded the migration is spent on-call.

**Device model chosen on capacity.** Ships, fits, and generates below reading speed.

**Thermal surprise in the field.** Passes lab testing at burst rates and throttles in a
user's hand or pocket, where the thermal envelope is smaller than on a bench.

**Model evicted by the operating system.** A resident on-device model is reclaimed
when the user switches apps, so the next session pays a reload the first one did not
— and the reload is seconds at device storage bandwidths.

**Duty-cycle change breaking a device deployment.** A feature that moves generation from
occasional to continuous crosses the thermal threshold with no model change.

## 13. Alternatives

**Provider API.** Correct below the break-even, which is most products, and it buys
statistical multiplexing a single tenant cannot reproduce. It also transfers the
entire operations burden that {{eq:ops-is-most-of-the-fixed-cost}} found dominant,
which is the larger transfer of the two.

**Reserved or committed-use hardware.** Lowers the hourly rate substantially and lowers
the break-even with it, at the cost of committing before the volume is known — which
is a bet on the same utilisation figure, made earlier and with less information.

**Batch or offline processing on owned hardware.** Work with no latency requirement --
overnight enrichment, bulk classification, evaluation runs -- can be scheduled to fill
whatever capacity the interactive fleet is not using, which raises utilisation without
buying anything. This is the single most effective attack on
{{eq:self-hosting-is-a-utilisation-bet}} available to a team that already self-hosts,
and it is frequently overlooked because the batch work and the serving fleet belong to
different teams. A deployment at 45% utilisation with a queue of deferrable work is
leaving the entire economic argument on the table.

**Hybrid: API for burst, self-hosted for base load.** Runs owned hardware near full
utilisation and buys the ramp, which directly attacks the term
{{eq:self-hosting-is-a-utilisation-bet}} identifies. Operationally the most complex
option and economically the most defensible at moderate volume.

**On-device with cloud fallback.** Handles the common case locally at zero marginal cost
and escalates what the small model cannot do — the routing question of {{ch:sd-routing-caching}} with a
bandwidth constraint attached.

**Edge appliance.** Poor economics — a low-utilisation replica by construction — and
excellent latency and residency properties. The right answer whenever the
non-economic reasons dominate, which for regulated or disconnected environments they
usually do.

## 14. Evaluation

Report cost per million tokens *including* operations and at realised utilisation. Any
other figure is not comparable to an API price.

Track realised utilisation as a standing metric, since the entire economic case moves
with it and it is the input most often assumed.

Measure device throughput sustained over a realistic duty cycle, and report burst and
sustained separately.

Test on the lowest-specification device in the supported range, not the development
device. The bandwidth-per-gigabyte figure varies by a factor of five across a device
range.

Re-evaluate the build-versus-buy decision when volume, model size, or provider pricing
changes. All three move, and the crossover moves with them — historically against
self-hosting, since provider prices have fallen faster than hardware costs.

Report the deferrable-work backlog alongside utilisation. Batch work that could fill
idle capacity is the cheapest available improvement to the economics, and neither
number is usually visible to the team that owns the other.

## 15. Advanced Concepts

The economic model treats the provider price as fixed, which it has not historically
been. Provider prices have fallen faster than hardware costs, because providers capture
efficiency gains from every technique in this part and pass some through, while a
self-hoster must implement each one. That means the break-even *moves against
self-hosting over time* for a fixed workload, and a migration justified on today's
prices may not hold at next year's. The counter-argument is that a self-hoster can also
adopt those techniques — but doing so is precisely the operations cost that
{{eq:ops-is-most-of-the-fixed-cost}} found dominant.

The device analysis assumes decode dominates, which holds for chat and fails for
summarisation. A device summarising a long document is doing mostly prefill, which is
compute-bound, and a phone's compute deficit relative to a datacentre part is larger than
its bandwidth deficit. So **the device gap is workload-dependent and worse for
prefill-heavy tasks than the token-rate table suggests** — a point the table's decode
framing understates.

There is also a residency question specific to devices that the datacentre framing has
no analogue for. A phone cannot keep a model resident indefinitely: the operating system
reclaims memory from background applications, so a model that was loaded may not be
loaded when the user returns. That makes cold start a per-session cost rather than a
per-deployment one, and at device storage bandwidths -- well below the sources in
{{ch:inf-kubernetes}}'s table -- reloading a four-gigabyte model is seconds rather than
milliseconds. **On-device inference has a cold start that recurs**, and the mitigations
are the same ones {{eq:weight-placement-sets-utilisation}} identified: keep it resident
if you can afford the memory, and accept the reload if you cannot.

Thermal derating interacts with the batching results in a way neither chapter models. A
device serving one user has no batch, so it sits at the far memory-bound end of
{{ch:inf-cpu-gpu}}'s curve, drawing less power than a compute-bound workload would. That
means the thermal ceiling is reached *later* than a compute benchmark would suggest, and
sustained fractions measured on compute-heavy benchmarks understate what decode can hold.
Measuring $\sigma$ on the actual workload rather than a synthetic one is worth doing, and
the difference can be substantial.

## 16. Connection to Previous Chapters

{{eq:decode-is-bandwidth-bound}} from {{ch:inf-cpu-gpu}} is the whole device half.
Bandwidth over model bytes is the token rate, and a device's bandwidth-per-capacity ratio
is what makes capacity the wrong question.

{{eq:trigger-is-the-reciprocal-of-growth}} from {{ch:inf-kubernetes}} supplies the
utilisation ceiling that makes {{eq:self-hosting-is-a-utilisation-bet}} bite.

{{eq:weight-placement-sets-utilisation}} explains why an edge appliance with resident
weights has operational properties a cloud deployment cannot match.

{{eq:cache-quantisation-is-the-larger-lever}} from {{ch:inf-gpu-memory}} reverses on a
device: there the cache was the larger lever, here the weights are, because the binding
constraint changed.

## 17. Exercises

1. Compute the crossover utilisation $u^\times$ for a 3.20-per-hour GPU sustaining 9,000
   tokens/second against a 0.45 API price.

2. Derive $V^\star = \omega/(\alpha - c_{\text{unit}}(u))$ and evaluate it at 30% and 60%
   utilisation. How sensitive is it?

3. For a target device with 90 GB/s and 8 GB, find the largest int4 model that runs at 15
   tokens/second sustained at $\sigma = 0.5$.

4. Extend the second listing to include prefill for a 2,000-token prompt. How does the
   device gap change?

5. Model a hybrid deployment: owned hardware at 85% utilisation for base load plus API
   for the ramp. Where is its break-even?

## 18. Interview Questions

1. Our GPU costs a third of what the API charges per token. Should we self-host?

2. What is the largest fixed cost of running your own inference?

3. Below what utilisation does self-hosting stop having a break-even at all?

4. An 8B model fits on our target phone. Is that the right thing to check?

5. Our device benchmark shows 23 tokens/second and users say it is slow. Explain.

6. We self-host at 40% utilisation and have a large nightly batch job running on
   separate hardware. What would you propose, and what does it do to the economics?

## 19. Research Questions

1. How much statistical multiplexing does a provider actually achieve, and what
   utilisation does it imply relative to a single tenant?

2. What is the right way to measure sustained device throughput given that thermal
   behaviour is duty-cycle dependent?

3. How does the prefill deficit on devices compare to the decode deficit at realistic
   prompt lengths?

4. Does hybrid base-load-plus-burst deployment hold its economic advantage once its
   operations cost is measured rather than assumed?

## 20. Chapter Summary

Self-hosting is a bet on utilisation. Unit cost is the hourly rate over realised
throughput, so it is **0.21** per million tokens at full utilisation against an API's
**0.62**, and **1.06** at 20% — worse than buying
({{eq:self-hosting-is-a-utilisation-bet}}).

Adding operations and replica rounding gives a break-even of **44,000 million tokens a
month** at 45% utilisation, against a naive **15,124**, because low utilisation forces
replicas which raise fixed cost which raise the volume required. And **operations is 62%
of fixed cost** at a modest estimate, **92%** at one engineer
({{eq:ops-is-most-of-the-fixed-cost}}). Below the crossover utilisation there is no
break-even at any volume.

On a device the binding constraint inverts. A phone offers **5.7 GB/s per GB** against a
datacentre GPU's **42**, so an 8B int4 model fits everywhere and runs at **8.7
tokens/second** on a flagship phone — below reading speed
({{eq:device-quality-is-bandwidth-bound}}). A mid-range phone holds an 8B model and can
usefully run a **0.5B** one.

And benchmarks measure the wrong state: a mid-range phone bursts at **22.7** tokens/second
and sustains **9.5**, a **58%** drop
({{eq:benchmark-is-burst-users-get-sustained}}) — usable in a test and not in use.

Both halves correct a comparison made on the wrong quantity. The self-hosting case is
usually argued on price per token, which omits the two terms that decide it; the
device case is usually argued on whether the model fits, which is not the binding
constraint. In each the available number is easy to obtain and the right number takes
one further step -- divide by utilisation, or divide bandwidth by model bytes -- and
the further step reverses the conclusion often enough to be worth taking every time.

Carry forward: **price the utilisation, not the token**, and **on a device, check the
bandwidth before checking the fit**.

## 21. Further Reading

- {{cite:pope2022inference}} — prefill and decode, whose different device deficits the
  token-rate framing understates.
- {{cite:leviathan2023speculative}} — speculative decoding, unusually well suited to
  devices where arithmetic units are idle.
- {{cite:kwon2023pagedattention}} — the serving machinery whose operation is most of
  self-hosting's fixed cost.
- {{cite:patel2023splitwise}} — heterogeneous placement, which applies at the edge as well
  as in the datacentre.
