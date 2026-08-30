# -*- coding: utf-8 -*-
# Extracted from: Chapter 211 — Cost, Latency, and Governance
# Source: src/.../ch211-governance.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The mean cost of an agent request is a tail statistic, so budgeting on it is unstable.

An agent runs until it decides to stop. That makes step count a stopping-time
distribution with a long right tail, and cost grows faster than steps because the context
carried into step i grows with i.

The result is a per-request cost distribution where the mean sits far above the median and
is dominated by a small share of runs (eq:agent-cost-is-heavy-tailed).

This listing computes that distribution exactly, shows why an aggregate budget built on
the mean moves when nobody changed anything, and finds the control that does bound it
(eq:per-request-cap-beats-aggregate-budget).
"""
MAX_STEPS = 60
P_STUCK = 0.08                # share of runs that enter a non-terminating pattern
CONT_NORMAL = 0.70            # P(take another step | normal)
CONT_STUCK = 0.94             # P(take another step | stuck)
COST_STEP_0 = 0.0065          # dollars for the first step
CTX_GROWTH = 0.22             # each step carries more context than the last
SUCC_NORMAL = 0.94
SUCC_STUCK = 0.19
REQUESTS_MONTH = 1_260_000.0
LAT_STEP_S = 2.35


def cum_cost(n):
    """Cost of a run that took n steps, with context growing each step."""
    return COST_STEP_0 * sum(1.0 + CTX_GROWTH * i for i in range(n))


def pmf(cont):
    """P(run takes exactly n steps), truncated at MAX_STEPS."""
    out = {}
    for n in range(1, MAX_STEPS + 1):
        p = (cont ** (n - 1)) * (1.0 - cont)
        out[n] = p
    tail = 1.0 - sum(out.values())
    out[MAX_STEPS] += tail       # anything longer is stopped by the framework
    return out


norm, stuck = pmf(CONT_NORMAL), pmf(CONT_STUCK)
JOINT = []
for n in range(1, MAX_STEPS + 1):
    JOINT.append((n, (1 - P_STUCK) * norm[n], SUCC_NORMAL))
    JOINT.append((n, P_STUCK * stuck[n], SUCC_STUCK))

rows = sorted(((cum_cost(n), p, n, s) for n, p, s in JOINT if p > 0))
mean_cost = sum(c * p for c, p, n, s in rows)
mean_steps = sum(n * p for c, p, n, s in rows)

print(f"Agent runs stop when they decide to. Mean {mean_steps:.1f} steps,")
print(f"mean cost ${mean_cost:.4f} a request, {REQUESTS_MONTH:,.0f} requests a month.")
print()
print("The per-request cost distribution.")
print()
print(f"{'percentile':>12}{'steps':>9}{'cost':>11}{'x mean':>10}{'x median':>11}")
print("-" * 53)


def at(q):
    acc = 0.0
    for c, p, n, s in rows:
        acc += p
        if acc >= q:
            return c, n
    return rows[-1][0], rows[-1][2]


median = at(0.50)[0]
pct = {}
for q in (0.50, 0.75, 0.90, 0.99, 0.999, 0.9999):
    c, n = at(q)
    pct[q] = (c, n)
    print(f"{q:>12.2%}{n:>9}{c:>11.4f}{c / mean_cost:>10.1f}{c / median:>11.1f}")

print()
print(f"mean ${mean_cost:.4f} sits at the {sum(p for c, p, n, s in rows if c <= mean_cost):.0%} "
      f"percentile -- above most requests.")

print()
print()
print("Where the money goes: share of spend by share of requests.")
print()
print(f"{'top share of requests':>23}{'share of spend':>17}{'concentration':>16}")
print("-" * 56)
desc = sorted(rows, key=lambda r: -r[0])
conc = {}
for frac in (0.001, 0.01, 0.05, 0.10, 0.25):
    acc_p, acc_c = 0.0, 0.0
    for c, p, n, s in desc:
        take = min(p, frac - acc_p)
        if take <= 0:
            break
        acc_c += c * take
        acc_p += take
    conc[frac] = acc_c / mean_cost
    print(f"{frac:>23.1%}{acc_c / mean_cost:>17.1%}{(acc_c / mean_cost) / frac:>15.1f}x")

print()
print()
print("Why an aggregate budget built on this mean is unstable: the mean is set")
print("by the tail, and the tail moves for reasons nobody is watching.")
print()
print(f"{'change':>38}{'stuck rate':>13}{'cont|stuck':>13}"
      f"{'mean cost':>12}{'monthly':>13}{'vs base':>10}")
print("-" * 99)


def mean_for(p_stuck, cont_stuck, cont_norm=CONT_NORMAL):
    st, nm = pmf(cont_stuck), pmf(cont_norm)
    m = 0.0
    for n in range(1, MAX_STEPS + 1):
        m += cum_cost(n) * ((1 - p_stuck) * nm[n] + p_stuck * st[n])
    return m


SHIFTS = [
    ("baseline",                          P_STUCK, CONT_STUCK, CONT_NORMAL),
    ("a tool gets slower, more retries",  0.11,    CONT_STUCK, CONT_NORMAL),
    ("a prompt change adds one example",  P_STUCK, 0.95,       CONT_NORMAL),
    ("a harder customer segment",         P_STUCK, CONT_STUCK, 0.74),
    ("all three",                         0.11,    0.95,       0.74),
]
shift = {}
for label, ps, cs, cn in SHIFTS:
    m = mean_for(ps, cs, cn)
    shift[label] = (m, m * REQUESTS_MONTH)
    print(f"{label:>38}{ps:>13.0%}{cs:>13.2f}{m:>12.4f}"
          f"{m * REQUESTS_MONTH:>13,.0f}{m / mean_cost:>9.2f}x")

print()
print()
print("The control that does bound it: a cost cap on the individual request.")
print()
print(f"{'cap':>16}{'cap $':>10}{'requests capped':>18}{'spend removed':>16}"
      f"{'successes lost':>17}{'success cost':>15}")
print("-" * 92)
base_succ = sum(p * s for c, p, n, s in rows)
caps = {}
for k in (30, 20, 12, 8, 5, 3):
    cap_c = k * mean_cost
    capped_p = sum(p for c, p, n, s in rows if c > cap_c)
    spend_after = sum(min(c, cap_c) * p for c, p, n, s in rows)
    succ_lost = sum(p * s for c, p, n, s in rows if c > cap_c)
    caps[k] = (cap_c, capped_p, 1 - spend_after / mean_cost, succ_lost)
    per_succ = ((mean_cost - spend_after) * REQUESTS_MONTH) / max(
        succ_lost * REQUESTS_MONTH, 1e-9)
    print(f"{k:>14}x{cap_c:>10.4f}{capped_p:>18.3%}"
          f"{1 - spend_after / mean_cost:>16.1%}{succ_lost:>17.3%}"
          f"{per_succ:>15,.0f}")

print()
print(f"baseline success rate: {base_succ:.1%}")

print()
print()
print("The same cap is a latency control, because steps cost time as well as")
print(f"money ({LAT_STEP_S:.2f}s a step).")
print()
print(f"{'cap':>16}{'max steps':>12}{'max latency':>14}{'p99.9 after':>14}")
print("-" * 56)
for k in (30, 20, 12, 8, 5, 3):
    cap_c = k * mean_cost
    max_n = max((n for c, p, n, s in rows if c <= cap_c), default=1)
    p999_n = min(pct[0.999][1], max_n)
    print(f"{k:>14}x{max_n:>12}{max_n * LAT_STEP_S:>13.0f}s"
          f"{p999_n * LAT_STEP_S:>13.0f}s")

print()
print()
print("Aggregate budget against per-request cap, as controls.")
print()
print(f"{'control':>26}{'bounds the mean?':>19}{'bounds a request?':>20}"
      f"{'acts before spend?':>21}")
print("-" * 86)
for name, a, b, c in (
        ("monthly budget",        "no -- reports it", "no",  "no"),
        ("daily spend cap",       "no",               "no",  "no"),
        ("step limit",            "partly",           "yes", "yes"),
        ("per-request cost cap",  "yes",              "yes", "yes"),
):
    print(f"{name:>26}{a:>19}{b:>20}{c:>21}")

print(f"""
The distribution table is the shape everything else follows from. The median request
costs ${median:.4f} and the mean is ${mean_cost:.4f} --
**{mean_cost / median:.1f} times the median** -- while the 99.9th percentile is
${pct[0.999][0]:.4f}, or {pct[0.999][0] / mean_cost:.0f} times the mean
(eq:agent-cost-is-heavy-tailed).

The mean sits at the {sum(p for c, p, n, s in rows if c <= mean_cost):.0%} percentile, which
is the compact statement of the problem: **the average request is more expensive than most
requests**, so any plan phrased in averages is a plan about a request that rarely happens.

The concentration table says where the money is. The most expensive
{0.01:.0%} of requests are {conc[0.01]:.0%} of spend --
{conc[0.01] / 0.01:.0f} times their share -- and the top {0.10:.0%} are
{conc[0.10]:.0%}.

That is not an anomaly to be cleaned up. It is what a stopping-time distribution looks
like when cost grows with the step index, and it will be true of any agent that decides
its own length.

The stability table is the part that makes aggregate budgeting fail in practice rather
than merely in principle. Nobody has to do anything wrong for the monthly bill to move.
A tool that gets slower and triggers more retries takes the stuck rate from
{P_STUCK:.0%} to {0.11:.0%} and the monthly spend from
{shift['baseline'][1]:,.0f} to {shift['a tool gets slower, more retries'][1]:,.0f}. A
prompt change that adds one example -- which lengthens context and makes continuation
marginally more attractive -- takes it to
{shift['a prompt change adds one example'][1]:,.0f}. All three together,
{shift['all three'][1]:,.0f}, or {shift['all three'][0] / mean_cost:.2f} times baseline.

**None of the three was a cost change.** One was a dependency's latency, one was a prompt
edit -- and ch:ops-prompt-versioning established that a prompt edit passes no gate at all
-- and one was who showed up. The budget moved
{shift['all three'][0] / mean_cost - 1:.0%} and nothing in the cost control system has an
opinion about any of these events, because none of them is filed as a cost event.

The cap table is the control that does work, and its numbers are better than they have
any right to be. A cap at {8}x the mean -- ${8 * mean_cost:.4f} -- truncates
{caps[8][1]:.2%} of requests, removes {caps[8][2]:.1%} of spend, and loses
{caps[8][3]:.2%} of successful requests
(eq:per-request-cap-beats-aggregate-budget).

The reason the trade is so favourable is in the model rather than in the arithmetic: the
expensive tail is disproportionately the stuck mode, which succeeds
{SUCC_STUCK:.0%} of the time against {SUCC_NORMAL:.0%} for normal runs. **The requests you
truncate are mostly the ones that were not going to work**, which is why capping is not
the blunt instrument it sounds like.

Note also what the {0.999:.1%} and {0.9999:.2%} percentiles have in common: both are
{pct[0.999][1]} steps, the framework's own step limit. The limit does not remove the tail,
it **piles it up at the boundary** -- a point mass of maximum-cost runs, every one of
which spent the full budget before being stopped.

The implied price is the number to put in a design review: at an {8}x cap you are paying
${((mean_cost - sum(min(c, 8 * mean_cost) * p for c, p, n, s in rows)) * REQUESTS_MONTH) / max(caps[8][3] * REQUESTS_MONTH, 1e-9):,.2f}
per additional success by *not* capping -- against the
$0.62 a blocked request was worth in the previous listing. Somebody should have to say
both numbers out loud before deciding the cap is too aggressive.

The latency table is the free consequence. The same cap bounds run length, so it bounds
the tail latency that ch:sd-latency spent a chapter attacking -- an
{8}x cost cap holds the run to {max((n for c, p, n, s in rows if c <= 8 * mean_cost), default=1)} steps and
{max((n for c, p, n, s in rows if c <= 8 * mean_cost), default=1) * LAT_STEP_S:.0f} seconds.

One control, two budgets, and they were being managed by separate teams with separate
dashboards.

The last table is the summary a governance document should contain. A monthly budget
bounds nothing and reports afterwards. A step limit bounds a request but only partly
bounds the mean, because ch:sd-apis-auth's result applies -- a count is not a cost. Only a
per-request cost cap bounds both, acts before the money is spent, and needs nothing that
the request handler does not already know.""")
