---
id: as-long-running
number: 167
part: XVIII
tier: full
status: draft
requires: [replay-needs-idempotence, state-must-be-sufficient,
           checkpoints-cap-the-exponent, habituation]
provides: [horizon-changes-the-failure, recovery-converts-failure-to-cost,
           drift-is-silent, revalidation-is-cheapest,
           placement-beats-frequency, oversight-has-a-horizon-limit,
           oversight-decays-in-frequency]
citations: [cemri2025mast, liu2024agentbench, zhou2024webarena,
            shinn2023reflexion, greshake2023indirect, yao2023react]
---

## 1. Learning Objectives

By the end of this chapter you will be able to explain why the failure mode that
dominates a long-running workflow is not the one that dominates a short one; state
what recovery actually buys and what it converts step failure into; recognise
silent drift as the failure that produces no error; schedule assumption
re-validation and defend the interval; and place human oversight by consequence
rather than by frequency — which the measurements say is worth an eightfold larger
review budget.

## 2. Why This Matters

Everything before this chapter measured runs that lasted seconds to minutes. This
one is about runs that last days: a research task that crawls a hundred sources, a
migration that touches ten thousand files, a monitoring agent that watches a system
for a week.

The intuition most people carry over is that a long run is a short run with more
steps, so the thing to fix is per-step reliability. That intuition fails
immediately, and the arithmetic says why: at $98.5\%$ per step, a $300$-step run
completes $1\%$ of the time. **A long-running system is only possible at all
because {{ch:ag-recovery}} exists** — a failed step is retried rather than fatal.

Recovery does not remove step failure. It converts it into budget consumption
({{eq:recovery-converts-failure-to-cost}}), and once the budget absorbs that, the
thing that binds is something else entirely.

{{sec:9-practical-example}} measures it. With a generous budget, exhaustion is
$0.0\%$ of outcomes at every horizon tested. **Silent drift** — an assumption made
early that stopped being true, with the run continuing confidently on it — is
$94.3\%$ of failures at horizon $10$ and still $43.7\%$ at horizon $1000$.

A drifted run *completes*. Every step returned success. The answer is wrong because
a premise expired around step forty and nothing looked again.

The fix is cheap enough to be embarrassing: re-validating assumptions takes horizon
$300$ from $0.7\%$ correct to $76.8\%$ for about $50\%$ more steps. And the second
listing finds the same shape in human oversight — twelve pauses placed before
consequential steps match a hundred placed uniformly
({{eq:placement-beats-frequency}}), because {{ch:ag-termination}}'s habituation
destroys most of the value of frequency.

## 3. Prerequisites

{{ch:as-state-machines}} supplies durability, without which none of this survives a
restart, and its finding that the tried set matters most applies with more force
over days than minutes.

{{ch:ag-recovery}}'s retry is what makes a long horizon feasible;
{{eq:recovery-converts-failure-to-cost}} is that chapter's mechanism restated as an
economic claim.

{{ch:ag-termination}}'s budget is the resource retry consumes, and its
human-in-the-loop gate is what {{sec:9-practical-example}}'s second listing places.

{{ch:ag-termination}}'s {{eq:habituation}} is the constraint that makes
placement matter — attention per pause falls as pauses get more frequent.

## 4. Intuitive Explanation

Consider an agent given a week to migrate a codebase.

On day one it reads the build configuration and forms a picture: this project uses
a particular test runner, this module is the dependency root, this API is the one
being replaced. Everything it does afterwards rests on that picture.

On day three, someone merges a change to the build configuration.

Nothing in the agent's loop notices. There is no error. Every subsequent step
succeeds — it edits files, runs commands, gets exit code zero. It is doing correct
work against a world that no longer exists, and it will keep doing so until it
finishes and hands you something that does not build.

That is drift, and it is the characteristic failure of a long horizon. It is
invisible in exactly the way the failures of a short horizon are visible: a
short-horizon failure throws, and drift returns success.

Why does it dominate? Because the other failure modes get handled. Step failures
are retried, which is what recovery is for. Budget exhaustion is monitored, because
it costs money and dashboards show money. Nobody monitors "the premises this run
was working from" because there is no such field anywhere.

And the exposure grows with the horizon in a way per-step reliability does not.
Each additional step is another chance for the world to change underneath, and
because staleness is *sticky* — once a premise expires it stays expired — the
probability of finishing on at least one stale premise rises much faster than any
individual step's failure probability.

The fix is to look again. Periodically re-read the things you assumed, and check
whether they still hold. It costs a step. {{sec:9-practical-example}} prices the
intervention at roughly $50\%$ more steps for a $76$-point swing in correctness,
which is the best trade measured anywhere in this part.

Then the second question, which is where a human goes.

The reflex is to pause more often for approval on a long run, because there is more
that can go wrong. That reflex is wrong for the reason {{ch:ag-security}} found: a
reviewer asked constantly stops reading. In {{sec:9-practical-example}}'s model a
reviewer catching $85\%$ when attentive is down to $4.8\%$ when asked at every step,
and the $200$ pauses cost $800$ hours of wall-clock delay.

So frequency does not work. What works is placement. Ask before the steps where
being off-course actually costs something — the irreversible ones, {{ch:ag-security}}'s
own criterion. Twelve such pauses matched a hundred uniform ones at an eighth of the
delay, because a uniform pause interrupts at a random moment and most moments do not
matter yet.

## 5. Formal Explanation

Let a run have horizon $H$, per-step success $p$, and a hard-failure probability
$q \ll 1-p$ that retry cannot fix. Without retry, completion is $p^H$. With retry
and a budget $B$, completion is instead governed by whether the expected work fits:

$$\mathbb{E}[\text{steps}] = \frac{H}{p}, \qquad \Pr[\text{exhausted}] = \Pr\!\left[\text{Neg-Binom}(H, p) > B\right]$$ (eq:recovery-converts-failure-to-cost)

which for $B = \beta H$ with $\beta > 1/p$ is small and shrinking in $H$ by
concentration. **Recovery converts a multiplicative reliability problem into an
additive budget problem**, and the additive one is easy.

Now drift. Let the run depend on $A$ standing assumptions, each expiring with
probability $\delta$ per step. Staleness is absorbing without intervention, so:

$$\Pr[\text{stale at } H] = 1 - (1-\delta)^{AH}$$ (eq:drift-is-silent)

Compare the two exponents. Step failure enters as $p^H$ only *without* recovery;
with recovery it enters as a $\sqrt{H}$-scale fluctuation around $H/p$. Drift enters
as $\delta A H$ in the exponent regardless. So:

$$\lim_{H \to \infty} \frac{\Pr[\text{drift}]}{\Pr[\text{exhausted}]} \to \infty$$ (eq:horizon-changes-the-failure)

**The dominant failure mode changes with the horizon**, and the change is not a
matter of parameter tuning — it follows from recovery handling one term and nothing
handling the other.

Re-validation every $\kappa$ steps with repair probability $\rho$ makes staleness
non-absorbing. The stationary stale probability becomes approximately:

$$\pi_{\text{stale}} \approx \frac{\delta A \kappa}{\delta A \kappa + \rho}$$ (eq:revalidation-is-cheapest)

linear in $\kappa$ for small $\kappa$, against a cost of $H/\kappa$ extra steps. The
benefit is first-order in $\kappa$ and the cost is $O(1/\kappa)$, so for cheap checks
the optimum is at the smallest feasible $\kappa$ — which is what
{{sec:9-practical-example}} finds, and why the usual "there must be a trade-off"
instinct misleads here.

Now oversight. Let a gate fire on a set $G \subseteq \{1..H\}$ of steps with
$|G| = g$, and let the reviewer's catch rate decay in the ask rate:

$$c(g) = \frac{c_0}{1 + g/g_{1/2}}$$ (eq:oversight-decays-in-frequency)

Harm occurs when an off-course run reaches a consequential step. Expected catches
before harm is $\sum_{t \in G} c(g)\Pr[\text{off at } t]$, and the *useful* catches
are only those occurring between going off-course and reaching a consequential step.
A uniform $G$ places mass proportionally to $H$; a $G$ concentrated on consequential
steps places all of it in the window that matters:

$$\frac{\text{useful}(G_{\text{targeted}})}{\text{useful}(G_{\text{uniform}})} \approx \frac{1}{\Pr[\text{consequential}]}$$ (eq:placement-beats-frequency)

which at $6\%$ consequential steps is a factor of roughly $16$ in efficiency per
pause — the mechanism behind the eightfold budget equivalence measured.

Finally, a limit. Since $c(g) \to 0$ as $g$ grows and harm probability rises with
$H$, there is an $H^{*}$ beyond which no gate schedule holds harm below a target:

$$\exists\, H^{*} : \forall G,\ \Pr[\text{harm}] > \tau \quad \text{for } H > H^{*}$$ (eq:oversight-has-a-horizon-limit)

**Gate-based oversight has a horizon past which it does not work at any budget.**

## 6. Mathematical Foundation

Three extractions.

**Budget sizing is not the interesting decision.** From
{{eq:recovery-converts-failure-to-cost}}, once $\beta > 1/p$ with margin, exhaustion
concentrates away and further budget buys nothing.
{{sec:9-practical-example}} confirms it: $0.0\%$ exhausted at $\beta = 1.6$ across
every horizon. Teams tune budget because it is visible on a bill; the measurements
say it is the wrong dial once it is roughly right.

**The re-validation optimum is a corner unless checks are expensive.** The
first-order-benefit / inverse-cost structure of {{eq:revalidation-is-cheapest}} puts
the optimum at minimum $\kappa$ whenever a check costs $O(1)$ steps. It moves
inward only when a check costs a constant fraction of the work — re-running an
expensive query, say. Knowing which regime you are in is a one-line calculation and
decides the whole schedule.

**Placement efficiency scales inversely with consequence density.** From
{{eq:placement-beats-frequency}}, the sparser the consequential steps, the larger
targeting's advantage — because uniform scheduling wastes a larger fraction of its
pauses. A workflow where every step is irreversible gains nothing from targeting;
one where $1\%$ of steps matter gains enormously. That density is a property you can
measure from a trace.

## 7. Internal Mechanics

### 7.1 What a re-validation actually checks

"Re-validate assumptions" is only actionable if the assumptions are written down,
which is the real work.

```mermaid {#fig:revalidation caption="Assumptions recorded at the moment they are formed, and re-checked on a schedule. The check is cheap; recording what to check is the design work."}
flowchart TD
    A[step forms a premise] --> R[record: source, value, how to recheck]
    R --> W[work continues]
    W --> C{recheck due?}
    C -- no --> W
    C -- yes --> V[re-read source]
    V --> M{still equal?}
    M -- yes --> W
    M -- no --> P[repair: replan from current truth]
    P --> W
```

The recorded triple is (source, observed value, re-read procedure). A premise
without a re-read procedure cannot be re-validated, so the discipline is to refuse
to record one — if you cannot say how you would check it again, you have an
assumption you cannot manage.

This is {{ch:ag-memory}}'s scratchpad with a freshness field, and
{{ch:as-state-machines}} says it must be durable or a resume loses the whole
mechanism.

### 7.2 Why drift is absorbing and why that is the whole problem

A step failure is transient: it happens, you retry, the state is unchanged. A stale
premise is absorbing: once it expires, every subsequent step inherits it, and
nothing in the loop returns it to validity.

That asymmetry is the entire reason {{eq:horizon-changes-the-failure}} holds.
Transient failures do not accumulate under retry; absorbing ones accumulate under
everything. Re-validation is precisely the operation that makes the absorbing state
non-absorbing, which is why its effect size is so large relative to its cost.

### 7.3 Consequence density, and how to measure it

Targeting needs a definition of "consequential", and the useful one is
{{ch:ag-termination}}'s: irreversible, or expensive to reverse. External writes,
messages sent, money moved, resources destroyed.

That is a property of the *tool*, not of the step, so it is a column in the same
audit {{ch:as-state-machines}} asked for on idempotence. Both questions — is this
idempotent, is this reversible — are answered per tool, once, and they drive
durability and oversight respectively.

Consequence density is then the fraction of steps in a typical trace that call such
a tool. {{eq:placement-beats-frequency}} says the lower it is, the more targeting is
worth.

### 7.4 Batching gates so the human sees a decision, not a step

A targeted gate still interrupts, and a human interrupted for a single tool call has
no context to judge it. The practical form is to batch: pause before a *group* of
consequential steps and present the plan for that group.

This trades a small amount of granularity for a large amount of reviewability, and
it reduces $g$ in {{eq:oversight-decays-in-frequency}}, which raises the catch rate on the
pauses that remain. It is the same argument {{ch:ag-planning}} made for
segment-level rather than step-level verification.

### 7.5 What to do past the oversight horizon

{{eq:oversight-has-a-horizon-limit}} says gates stop working past some $H^{*}$.
{{sec:9-practical-example}} finds every design at or under $1.2\%$ by horizon $600$.

Three responses, in order of preference. **Shorten the horizon** by decomposing into
independently-completing runs, each of which reports and terminates — this is
{{ch:ag-planning}}'s decomposition arriving for a third reason. **Reduce consequence
density** by making steps reversible, which moves the harm threshold rather than the
catch rate. **Re-validate rather than gate**, since the first listing's mechanism
does not habituate: an automated freshness check has the same catch rate on its
thousandth firing as its first.

That last point is the one worth carrying: **automated re-validation scales with the
horizon and human gates do not.**

### 7.6 Wall-clock is a first-class cost here

A pause that costs four hours of human latency is invisible in every metric this
book has used so far, all of which count steps or tokens.

On a long-running workflow it is often the dominant cost: {{sec:9-practical-example}}'s
every-step schedule costs $800$ hours — five weeks — of delay on a run whose compute
takes minutes. **A design that is cheap in tokens and expensive in wall-clock will be
abandoned by its users**, and the abandonment will be recorded as a product problem
rather than an architecture one.

### 7.7 Drift in the environment versus drift in the goal

Two things go stale and they need different treatments.

**Environment drift** is what the listings model: a fact about the world changed.
Re-validation fixes it because the world can be re-read.

**Goal drift** is the run's own understanding of what it was asked, degrading across
summarisation and resumes — {{ch:as-state-machines}} found omitting the verbatim goal
costs $10$ points. Re-validation cannot fix it, because there is no external source
to re-read; only the durable verbatim goal can. Storing it is the entire mitigation,
and it is one field.

### 7.8 Autonomy is a claim about the oversight schedule, not the agent

"Autonomous" is used as though it were a property of a system, and it is more
useful as a description of one number: how much consequential action happens between
human decisions.

Under that reading the designs in {{sec:9-practical-example}} are all autonomous to
different degrees, and the degree is set by the gate schedule rather than by
anything about the model. A run with a gate before every consequential step is
supervised regardless of how capable the agent is; a run with gates every fifty
steps is autonomous across those fifty regardless of how cautious it is.

This matters because autonomy is usually discussed as a capability threshold — the
model is good enough to be trusted, or it is not — and the measurements say the
question is badly posed. The relevant quantity is the *expected consequence between
decisions*, which is a product of consequence density and gate spacing, and both are
design choices.

It also gives a rule for increasing autonomy safely: widen the gate spacing only as
fast as you reduce consequence density. A system that made every step reversible
could run unsupervised for a very long time at the same expected harm as a
tightly-gated irreversible one. That is a more actionable programme than waiting for
a capability threshold, and it is the one {{sec:7-internal-mechanics}} recommends
past the oversight horizon.

The corollary is uncomfortable for how these systems are usually sold. **A claim
that a workflow is autonomous is a claim about how much irreversible action it takes
unattended**, and it can be checked from a trace rather than taken on faith.

## 8. Implementation

Two listings. The first decomposes long-horizon failure and prices re-validation.
The second places human oversight and measures placement against frequency.

```python {tier=A name=horizon-changes-the-failure}
"""What actually fails as the horizon gets longer.

Short-horizon agents fail because a step fails: ch:ag-loop's per-step reliability
compounds and the run dies. That framing does not survive a long horizon, for a
reason worth stating first.

At 98.5% per step, a 300-step run completes 1% of the time. A long-running system
is therefore only possible at all because ch:ag-recovery exists: a failed step is
RETRIED rather than fatal. Recovery does not make step failure go away -- it
converts it into budget consumption (eq:recovery-converts-failure-to-cost).

So the three things that can end a long run are:

  exhaustion   retries and work exceed ch:ag-termination's budget
  drift        an assumption made early stopped being true, and the run continued
               confidently on a stale premise -- every step 'succeeded'
  step failure a step fails in a way retry cannot fix

They scale differently in the horizon, so the dominant one changes
(eq:horizon-changes-the-failure).
"""
import numpy as np

rng = np.random.default_rng(3571)

M = 40000
P_STEP = 0.97           # per-step success; a failure is retried, not fatal
P_HARD = 0.0008         # per-step chance the failure is unrecoverable
P_INVALIDATE = 0.0025   # per-step chance a given standing assumption goes stale
N_ASSUME = 6
BUDGET_MULT = 1.6       # budget as a multiple of the nominal horizon


def run(horizon, m=M, recheck=0, p_step=P_STEP, p_inv=P_INVALIDATE,
        budget_mult=BUDGET_MULT, repair=0.9, recheck_cost=1):
    """Walk `horizon` steps under a budget. A failed step is retried and costs
    budget; a hard failure ends the run. Independently, standing assumptions go
    stale silently -- a run that finishes on a stale assumption produces a wrong
    answer with no error anywhere. A recheck re-validates and mostly repairs."""
    budget = int(horizon * budget_mult)
    alive = np.ones(m, dtype=bool)
    stale = np.zeros(m, dtype=bool)
    pos = np.zeros(m, dtype=np.int64)
    used = np.zeros(m, dtype=np.int64)
    hard = np.zeros(m, dtype=bool)
    for t in range(budget):
        live = alive & (pos < horizon) & (used < budget)
        idx = np.flatnonzero(live)
        if not len(idx):
            break
        used[idx] += 1
        # A hard failure is unrecoverable; a soft one just costs this step.
        h = rng.random(len(idx)) < P_HARD
        alive[idx[h]] = False
        hard[idx[h]] = True
        rest = idx[~h]
        good = rest[rng.random(len(rest)) < p_step]
        pos[good] += 1
        # Assumptions go stale as wall-clock passes, whether or not work advanced.
        went = rng.random(len(rest)) < (1 - (1 - p_inv) ** N_ASSUME)
        stale[rest[went]] = True
        if recheck and ((t + 1) % recheck == 0):
            fixed = rng.random(len(rest)) < repair
            stale[rest[fixed]] = False
            used[rest] += recheck_cost
    finished = alive & (pos >= horizon)
    correct = finished & ~stale
    exhausted = alive & (pos < horizon)
    return (float(correct.mean()), float((finished & stale).mean()),
            float(exhausted.mean()), float(hard.mean()), float(used.mean()))


HORIZONS = [10, 30, 100, 300, 1000]

print(f"{M:,} runs. Per-step success {P_STEP:.0%} with retry, {P_HARD:.2%} chance")
print(f"a failure is unrecoverable, {N_ASSUME} standing assumptions each going")
print(f"stale at {P_INVALIDATE:.2%} per step, budget {BUDGET_MULT:.1f}x horizon.")
print()
print(f"{'horizon':>9}{'correct':>10}{'silent drift':>14}{'exhausted':>11}"
      f"{'hard failure':>14}")
print("-" * 58)
base = {}
for h in HORIZONS:
    r = run(h)
    base[h] = r
    print(f"{h:>9}{r[0]:>10.1%}{r[1]:>14.1%}{r[2]:>11.1%}{r[3]:>14.1%}")

print()
print()
print("As a share of the FAILURES, which is the view that says what to work on.")
print()
print(f"{'horizon':>9}{'silent drift':>15}{'exhausted':>12}{'hard failure':>15}")
print("-" * 51)
share = {}
for h in HORIZONS:
    r = base[h]
    tot = r[1] + r[2] + r[3]
    row = (r[1] / tot, r[2] / tot, r[3] / tot)
    share[h] = row
    print(f"{h:>9}{row[0]:>15.1%}{row[1]:>12.1%}{row[2]:>15.1%}")

print()
print()
print("Rechecking assumptions repairs drift and costs budget. Sweeping the")
print("interval at horizon 300:")
print()
print(f"{'recheck every':>15}{'correct':>10}{'silent drift':>14}"
      f"{'exhausted':>11}{'steps used':>12}")
print("-" * 62)
rc = {}
for k in (0, 100, 50, 25, 10, 5, 2):
    r = run(300, recheck=k)
    rc[k] = r
    label = "never" if k == 0 else str(k)
    print(f"{label:>15}{r[0]:>10.1%}{r[1]:>14.1%}{r[2]:>11.1%}{r[4]:>12.0f}")

print()
print()
print("The optimum moves with the horizon: a longer run has more to go stale and")
print("less budget slack to spend on checking.")
print()
print(f"{'horizon':>9}{'never':>9}{'every 50':>11}{'every 25':>11}"
      f"{'every 10':>11}{'every 5':>10}{'best':>11}")
print("-" * 72)
opt = {}
INTERVALS = (0, 50, 25, 10, 5)
NAMES = ["never", "every 50", "every 25", "every 10", "every 5"]
for h in (30, 100, 300, 1000):
    row = [run(h, recheck=k)[0] for k in INTERVALS]
    best = NAMES[int(np.argmax(row))]
    opt[h] = (row, best)
    print(f"{h:>9}" + "".join(f"{v:>{w}.1%}" for v, w in
                              zip(row, (9, 11, 11, 11, 10))) + f"{best:>11}")

print()
print()
print("And against the rate the world changes, which is a property of the")
print("environment rather than of the agent. Horizon 300:")
print()
print(f"{'staleness rate':>16}{'never':>9}{'every 25':>11}{'every 10':>11}"
      f"{'best gain':>12}")
print("-" * 59)
sw = {}
for pi in (0.0005, 0.0025, 0.008, 0.02):
    row = [run(300, recheck=k, p_inv=pi)[0] for k in (0, 25, 10)]
    sw[pi] = row
    print(f"{pi:>16.2%}{row[0]:>9.1%}{row[1]:>11.1%}{row[2]:>11.1%}"
          f"{max(row[1:]) - row[0]:>+12.1%}")

print(f"""
The first table's most useful column is the one that stays near zero. EXHAUSTED is
{base[300][2]:.1%} at horizon 300 and {base[1000][2]:.1%} at 1000, because a
{BUDGET_MULT:.1f}x budget absorbs the retries comfortably.

That is worth stating plainly, because budget is the thing long-running systems are
usually tuned on. **With recovery in place, the budget is not what binds** -- it is
what converts step failure into a cost you can afford
(eq:recovery-converts-failure-to-cost).

What binds instead is the second column. At horizon {10}, silent drift is
{share[10][0]:.1%} of all failures. At {100} it is {share[100][0]:.1%}. At
{1000} it is {share[1000][0]:.1%}, having handed the lead to unrecoverable step
failure at {share[1000][2]:.1%}.

**The dominant failure mode changes with the horizon** (eq:horizon-changes-the-
failure), and for the range most production workflows live in -- tens to a few
hundred steps -- it is the one that produces no error at all. A drifted run
completes. Every step returned success. The answer is wrong because a premise
stopped being true somewhere around step 40 and nothing looked again.

The third table is the fix, and the size of it is the point. At horizon 300,
never rechecking gives {rc[0][0]:.1%}; rechecking every 2 steps gives
{rc[2][0]:.1%}. Even rechecking every 100 steps -- twice in the whole run -- gets
{rc[100][0]:.1%}.

The cost is small: {rc[0][4]:.0f} steps against {rc[2][4]:.0f}, about
{rc[2][4] / rc[0][4] - 1:+.0%}, for a {rc[2][0] - rc[0][0]:+.1%} swing in
correctness. **Re-validating assumptions is the cheapest intervention in this
chapter by a wide margin**, and almost nothing does it, because there is no error
to prompt it.

The fourth table looked like it would show an interior optimum and does not. The
most frequent interval tested wins at every horizon up to 1000. **At these
parameters there is no rechecking frequency that is too often** -- the budget cost
of a check is simply much smaller than the expected cost of continuing on a stale
premise.

That will not hold if a recheck is expensive; if re-validating means re-running a
query that costs as much as the work itself, the optimum moves inward and has to be
computed. But the default should be to check far more often than feels necessary,
because the failure it prevents is invisible until the end.

The last table shows the value peaking in the middle, which is the least obvious
result here. Rechecking is worth {sw[0.0005][1] - sw[0.0005][0]:+.1%} when the world
changes at {0.0005:.2%} per step, {sw[0.0025][2] - sw[0.0025][0]:+.1%} at
{0.0025:.2%}, and {sw[0.02][2] - sw[0.02][0]:+.1%} at {0.02:.2%}.

**Oversight of drift is worth most in a moderately unstable environment.** In a
stable one there is little to catch; in a violently unstable one the premise goes
stale again immediately after the check, and the right response is to shorten the
horizon rather than to check harder.""")
```

The second listing asks where the human should be.

```python {tier=A name=placement-beats-frequency}
"""Where to put the human in a run that lasts days.

ch:ag-termination put a human at the approval gate and ch:ag-security found the
gate habituates: a reviewer asked constantly approves reflexively, so the catch
rate per pause FALLS as pauses get more frequent.

A long-running workflow makes that worse in a way a short one does not. Each pause
costs wall-clock -- the human answers in hours, not milliseconds -- so pausing
often can mean the run takes a week. And the drift ch:as-long-running's first
listing measured is exactly what a human WOULD catch, if asked at the right moment.

So there are two questions, and only one of them is the one teams argue about:

  how often   pause every k steps -- the frequency question
  where       pause before consequential steps -- the placement question

This listing measures both (eq:placement-beats-frequency).
"""
import numpy as np

rng = np.random.default_rng(3607)

M = 40000
HORIZON = 200
P_WRONG = 0.010         # per-step chance the run goes off-course
P_CONSEQ = 0.06         # share of steps that are consequential (irreversible)
CATCH_0 = 0.85          # catch rate of an attentive reviewer
HALF = 12               # pauses per run at which attention has halved
HOURS = 4.0             # wall-clock hours a pause costs


def catch_rate(n_pauses):
    """ch:ag-termination's habituation: attention decays with how often you ask."""
    return CATCH_0 / (1.0 + n_pauses / HALF)


def run(every=0, placement="uniform", m=M, horizon=HORIZON, p_wrong=P_WRONG,
        conseq_only=False):
    """Walk the horizon. Off-course states accumulate; a pause may catch and
    repair one. `placement='targeted'` pauses only before consequential steps."""
    conseq = rng.random((m, horizon)) < P_CONSEQ
    if placement == "targeted":
        gate = conseq.copy()
    elif every:
        gate = np.zeros((m, horizon), dtype=bool)
        gate[:, ::every] = True
    else:
        gate = np.zeros((m, horizon), dtype=bool)
    n_pauses = gate.sum(1).mean()
    cr = catch_rate(n_pauses)
    off = np.zeros(m, dtype=bool)
    harm = np.zeros(m, dtype=bool)
    for t in range(horizon):
        went = rng.random(m) < p_wrong
        off |= went
        # A gate fires BEFORE the step, and may catch an off-course run.
        g = gate[:, t] & off
        caught = g & (rng.random(m) < cr)
        off &= ~caught
        # A consequential step taken while still off-course does real damage.
        harm |= off & conseq[:, t]
    return (float((~harm).mean()), float(off.mean()), float(n_pauses),
            float(n_pauses * HOURS), float(cr))


print(f"{M:,} runs of a {HORIZON}-step workflow. Each step has a {P_WRONG:.1%}")
print(f"chance of going off-course; {P_CONSEQ:.0%} of steps are consequential, and")
print(f"an off-course run reaching one does harm. An attentive reviewer catches")
print(f"{CATCH_0:.0%}, halving every {HALF} pauses per run (ch:ag-security).")
print()
print(f"{'pause every':>13}{'no harm':>10}{'pauses':>9}{'catch rate':>13}"
      f"{'delay (h)':>12}")
print("-" * 57)
freq = {}
for k in (0, 50, 25, 10, 5, 2, 1):
    r = run(every=k)
    freq[k] = r
    label = "never" if k == 0 else str(k)
    print(f"{label:>13}{r[0]:>10.1%}{r[2]:>9.1f}{r[4]:>13.1%}{r[3]:>12.0f}")

print()
print()
print("The same budget of human attention, spent on consequential steps only")
print("rather than uniformly. Both rows pause a similar number of times.")
print()
print(f"{'placement':>22}{'no harm':>10}{'pauses':>9}{'catch rate':>13}"
      f"{'delay (h)':>12}")
print("-" * 66)
tgt = run(placement="targeted")
# the uniform interval that produces the closest pause count
k_match = max(1, int(round(HORIZON / tgt[2])))
uni = run(every=k_match)
print(f"{('uniform, every ' + str(k_match)):>22}{uni[0]:>10.1%}{uni[2]:>9.1f}"
      f"{uni[4]:>13.1%}{uni[3]:>12.0f}")
print(f"{'targeted':>22}{tgt[0]:>10.1%}{tgt[2]:>9.1f}{tgt[4]:>13.1%}"
      f"{tgt[3]:>12.0f}")

print()
print()
print("Frequency against placement across horizons, since a longer run has more")
print("chances to drift and more consequential steps to reach.")
print()
print(f"{'horizon':>9}{'never':>9}{'uniform 10':>13}{'uniform 2':>12}"
      f"{'targeted':>11}{'best':>11}")
print("-" * 65)
hz = {}
for h in (50, 200, 600):
    row = (run(every=0, horizon=h)[0], run(every=10, horizon=h)[0],
           run(every=2, horizon=h)[0], run(placement="targeted", horizon=h)[0])
    names = ["never", "uniform 10", "uniform 2", "targeted"]
    hz[h] = (row, names[int(np.argmax(row))])
    print(f"{h:>9}{row[0]:>9.1%}{row[1]:>13.1%}{row[2]:>12.1%}{row[3]:>11.1%}"
          f"{hz[h][1]:>11}")

print()
print()
print("What habituation costs. Same designs, with the decay switched off -- an")
print("idealised reviewer whose attention never falls.")
print()
print(f"{'design':>16}{'with decay':>13}{'no decay':>11}{'loss':>10}")
print("-" * 50)
hab = {}
SAVE = HALF
for name, kw in [("uniform 10", dict(every=10)), ("uniform 2", dict(every=2)),
                 ("uniform 1", dict(every=1)),
                 ("targeted", dict(placement="targeted"))]:
    a = run(**kw)[0]
    globals()["HALF"] = 10 ** 9
    b = run(**kw)[0]
    globals()["HALF"] = SAVE
    hab[name] = (a, b)
    print(f"{name:>16}{a:>13.1%}{b:>11.1%}{a - b:>+10.1%}")

print(f"""
Read the first table as a cost curve rather than a benefit curve.

Going from never pausing to pausing every 50 steps buys
{freq[50][0] - freq[0][0]:+.1%}. Going from every 50 to every 1 -- a
{freq[1][2] / freq[50][2]:.0f}x increase in human interruptions and
{freq[1][3] - freq[50][3]:.0f} extra hours of wall-clock -- buys
{freq[1][0] - freq[50][0]:+.1%} more.

The reason is in the catch-rate column: {freq[50][4]:.1%} at every 50 and
{freq[1][4]:.1%} at every 1. **ch:ag-termination's habituation converts additional
oversight into additional noise**, and past a fairly early point the marginal pause
is close to worthless while its cost is undiminished.

The second table is the chapter's result, and it is about placement rather than
frequency.

Spending the SAME twelve pauses on consequential steps only gives
{tgt[0]:.1%} against the uniform schedule's {uni[0]:.1%}. Identical human cost,
identical habituation, {tgt[0] - uni[0]:+.1%} in harm avoided
(eq:placement-beats-frequency).

Better: targeted's {tgt[0]:.1%} on {tgt[2]:.0f} pauses beats uniform-every-2's
{freq[2][0]:.1%} on {freq[2][2]:.0f} pauses. **The targeted schedule matches an
eightfold larger review budget**, at {tgt[3]:.0f} hours of delay against
{freq[2][3]:.0f}.

The reason is structural rather than statistical. A uniform pause catches an
off-course run at a random moment, and most moments are ones where being off-course
costs nothing yet. A pause before a consequential step is asked exactly when the
answer matters -- it is ch:ag-termination's irreversibility criterion used as a
scheduling rule rather than as a policy.

The third table says the ordering holds across horizons, and it says something less
comfortable too. At horizon {600} every design is at or under
{max(hz[600][0]):.1%}. **Past some horizon, gate-based oversight cannot keep a run
on course at any review budget**, because the number of opportunities to drift
outgrows any schedule of checks. The response there is not more gates; it is the
first listing's re-validation, or a shorter horizon.

The last table prices habituation directly by switching it off. An idealised
reviewer whose attention never decayed would take uniform-every-1 to
{hab['uniform 1'][1]:.1%}; the real one gets {hab['uniform 1'][0]:.1%}, a loss of
{hab['uniform 1'][0] - hab['uniform 1'][1]:.1%}.

**Nearly all of the theoretical value of frequent oversight is destroyed by the
frequency itself.** That is the sharpest way to state why placement is the lever:
you cannot buy attention with volume, so the only remaining move is to spend the
attention you have where it changes an outcome.""")
```

## 9. Practical Example

The first listing runs workflows at horizons from $10$ to $1000$, with retry, a
$1.6\times$ budget, and six standing assumptions each expiring at $0.25\%$ per step.

```
  horizon   correct  silent drift  exhausted  hard failure
----------------------------------------------------------
       10     84.8%         14.4%       0.0%          0.9%
      100     19.7%         72.2%       0.0%          8.1%
      300      0.8%         77.2%       0.0%         22.0%
     1000      0.0%         43.7%       0.0%         56.3%
```

The exhausted column is $0.0\%$ everywhere. **With recovery in place, budget is not
what binds** ({{eq:recovery-converts-failure-to-cost}}) — it is what makes step
failure affordable.

As a share of failures:

```
  horizon   silent drift   exhausted   hard failure
---------------------------------------------------
       10          94.3%        0.0%           5.7%
      100          90.0%        0.0%          10.0%
      300          77.8%        0.0%          22.2%
     1000          43.7%        0.0%          56.3%
```

**The dominant failure mode changes with the horizon**
({{eq:horizon-changes-the-failure}}), and across the range most production
workflows occupy it is the one that produces no error at all.

Re-validation, at horizon $300$:

```
  recheck every   correct  silent drift  exhausted  steps used
--------------------------------------------------------------
          never      0.7%         76.9%       0.0%         273
            100     62.4%         15.7%       0.0%         276
             10     71.4%          6.5%       0.0%         301
              2     76.8%          0.9%       0.1%         410
```

$0.7\%$ to $76.8\%$ for $50\%$ more steps — and even checking *twice in the whole
run* gets $62.4\%$. **Re-validating assumptions is the cheapest intervention in this
part** ({{eq:revalidation-is-cheapest}}), and almost nothing does it, because
there is no error to prompt it.

Sweeping the interval against the horizon, the most frequent option tested wins
everywhere:

```
  horizon    never   every 50   every 25   every 10   every 5       best
------------------------------------------------------------------------
       30    61.3%      61.2%      86.3%      94.7%     95.5%    every 5
      300     0.8%      64.1%      65.6%      71.3%     75.0%    every 5
     1000     0.0%      26.0%      37.1%      40.2%     42.1%    every 5
```

There is no interior optimum at these parameters. That changes only if a check
costs a substantial fraction of the work itself.

And the value peaks in the middle of the staleness range:

```
  staleness rate    never   every 25   every 10   best gain
-----------------------------------------------------------
           0.05%    30.9%      75.4%      76.9%      +46.0%
           0.25%     0.8%      65.6%      71.1%      +70.3%
           2.00%     0.0%      24.9%      43.8%      +43.8%
```

**Oversight of drift is worth most in a moderately unstable environment.** In a
violently unstable one the premise expires again immediately, and the answer is a
shorter horizon rather than harder checking.

The second listing places a human. Pausing more often:

```
  pause every   no harm   pauses   catch rate   delay (h)
---------------------------------------------------------
        never     15.8%      0.0        85.0%           0
           50     22.4%      4.0        63.8%          16
           10     32.3%     20.0        31.9%          80
            1     37.1%    200.0         4.8%         800
```

The catch rate falls from $63.8\%$ to $4.8\%$
({{eq:oversight-decays-in-frequency}}): **habituation converts additional oversight into
additional noise.**

Spending the *same* twelve pauses by placement rather than by schedule:

```
             placement   no harm   pauses   catch rate   delay (h)
------------------------------------------------------------------
     uniform, every 17     29.9%     12.0        42.5%          48
              targeted     37.0%     12.0        42.5%          48
```

$+7.1$ points at identical human cost — and targeted's $37.0\%$ on twelve pauses
matches uniform-every-2's $35.7\%$ on a hundred, at $48$ hours of delay against
$400$. **The targeted schedule matches an eightfold larger review budget**
({{eq:placement-beats-frequency}}).

Across horizons, with an uncomfortable last row:

```
  horizon    never   uniform 10   uniform 2   targeted       best
-----------------------------------------------------------------
       50    71.0%        83.8%       89.3%      90.4%   targeted
      200    16.1%        32.3%       36.3%      37.1%   targeted
      600     0.3%         1.1%        1.1%       1.2%   targeted
```

**Past some horizon, gate-based oversight cannot keep a run on course at any review
budget** ({{eq:oversight-has-a-horizon-limit}}).

And habituation priced by switching it off:

```
          design   with decay   no decay      loss
--------------------------------------------------
      uniform 10        31.8%      57.2%    -25.4%
       uniform 1        36.6%      97.9%    -61.2%
        targeted        37.3%      78.1%    -40.8%
```

An idealised reviewer would take every-step pausing to $97.9\%$; the real one gets
$36.6\%$. **Nearly all of the theoretical value of frequent oversight is destroyed
by the frequency itself.**

## 10. Production Considerations

Record every premise as (source, value, re-read procedure) at the moment it is
formed, and refuse to record one you cannot say how to re-check.

Re-validate on a schedule, far more often than feels necessary. The measurements say
the corner is the optimum unless a check is expensive; compute which regime you are
in rather than guessing.

Size the budget once, at comfortably above $1/p$, and stop tuning it. It is not
what binds.

Audit tools for reversibility in the same pass as
{{ch:as-state-machines}}'s idempotence audit. Both questions are per-tool and both
drive a scheduling decision.

Place gates before consequential steps rather than on an interval, and batch them so
the human sees a decision rather than a call.

Track wall-clock delay as a first-class metric. A design that is cheap in tokens and
costs five weeks of human latency will be abandoned.

Store the verbatim goal durably. It is the only mitigation for goal drift, and it is
one field.

And measure your consequence density — it tells you how much targeting is worth
before you build it.

## 11. Common Mistakes

**Treating a long run as a short run with more steps.** The dominant failure mode
differs, so the fix does too.

**Tuning the budget.** Once it comfortably exceeds $H/p$, further budget buys
nothing measurable.

**Assuming success means correctness.** A drifted run completes with every step
green.

**Pausing more often on long runs.** Habituation means this buys little and costs
enormously in wall-clock.

**Gating uniformly.** Most moments do not matter yet; a pause spent there is wasted
attention that cannot be spent later.

**Recording premises without a re-read procedure.** An assumption you cannot re-check
is one you cannot manage.

**Reporting wall-clock cost nowhere.** It is often the dominant cost and appears on
no dashboard this book has otherwise used.

## 12. Failure Modes

*Silent drift to a wrong answer.* The characteristic long-horizon failure: complete,
green, wrong.

*Goal drift across resumes.* {{ch:as-state-machines}}'s missing verbatim goal,
compounding over days.

*Rubber-stamped approvals.* {{ch:ag-termination}}'s habituation at scale — a reviewer
who has approved four hundred times this week.

*Wall-clock abandonment.* A correct design nobody uses because it takes a week.

*Runaway cost under retry.* The other side of
{{eq:recovery-converts-failure-to-cost}}: if $p$ degrades, retry converts it into
spend rather than into failure, and the alert fires on the bill.

*Persisted injection over a long horizon.* {{cite:greshake2023indirect}}'s vector
with days of dwell time and many retrieval opportunities.

## 13. Alternatives

**Shorter runs that report and terminate.** The strongest alternative: decompose so
no single run reaches the horizon where drift and oversight limits bind.

**Event-driven rather than long-polling.** The agent sleeps and wakes on a change,
so wall-clock passes without steps accumulating — and the change itself is a
re-validation signal.

**Continuous re-planning.** {{cite:shinn2023reflexion}}'s reflection applied on a
timer rather than on failure, which subsumes re-validation at higher cost.

**Automated invariant checks instead of human gates.** They do not habituate, which
{{sec:7-internal-mechanics}} argues is the property that matters past $H^{*}$.

**Human as author rather than approver.** Rather than gating an autonomous run, have
the human approve a plan once and the agent execute a bounded version of it —
{{ch:ag-planning}}'s plan-and-execute used as an oversight structure.

## 14. Evaluation

Measure drift directly: at the end of a run, re-read the premises it recorded and
count how many had expired. This is the only measurement of the dominant failure
mode, and nothing gives it to you by default.

Report the failure decomposition — drift, exhaustion, hard failure — rather than a
single success rate. The decomposition is what says which lever to pull.

Measure catch rate per pause over time, not in aggregate. Habituation is a trend and
an average hides it.

Track wall-clock delay alongside token cost.

Measure consequence density from real traces.

And evaluate at your production horizon. {{eq:horizon-changes-the-failure}} says a
system validated at fifty steps tells you very little about six hundred.

## 15. Advanced Concepts

**Adaptive re-validation intervals.** Checking more often for premises observed to be
volatile and less for stable ones turns a global $\kappa$ into a per-premise one,
and the volatility is measurable from the check history itself.
{{maturity:EMERGING}}.

**Attention budgets as an explicit resource.** {{eq:oversight-decays-in-frequency}} implies a
reviewer has a finite catch-rate-integral per period. Scheduling against that
explicitly — rather than against step counts — would make oversight a resource
allocation problem with known structure. Nothing does this.

**Estimating $H^{*}$ from a trace.** {{eq:oversight-has-a-horizon-limit}} is an
existence claim; computing where the horizon limit falls for a given system would
tell teams when to stop adding gates and start decomposing.

**Drift-aware benchmarks.** {{cite:liu2024agentbench}} and
{{cite:zhou2024webarena}} evaluate against static environments, so drift cannot
occur and the dominant long-horizon failure is by construction unmeasurable.
{{maturity:RESEARCH FRONTIER}}.

## 16. Connection to Previous Chapters

{{ch:ag-recovery}}'s retry is what makes long horizons possible, and this chapter
restates what it buys: a conversion from reliability to cost.

{{ch:ag-termination}}'s budget turns out not to bind once it is roughly right, which
is a mild correction to that chapter's emphasis.

{{ch:ag-termination}}'s habituation is the constraint that makes placement the lever,
and its irreversibility criterion is what the placement uses.

{{ch:as-state-machines}}'s durable verbatim goal is the only mitigation for goal
drift, and its tool audit extends naturally to reversibility.

{{ch:ag-planning}}'s decomposition arrives here for a third distinct reason: it is
the response to the oversight horizon limit.

Ahead: {{ch:as-specialized}} looks at what changes when the agent is specialised to a
domain, and {{ch:as-failures}} returns to {{cite:cemri2025mast}}'s taxonomy with
this chapter's modes as instances.

## 17. Exercises

1. Derive the exhaustion probability from {{eq:recovery-converts-failure-to-cost}}
   and find the $\beta$ at which it drops below $1\%$ for $H=300$, $p=0.97$.

2. Make re-validation expensive — a check costing ten steps — and find where the
   interior optimum appears.

3. Add per-premise volatility and implement adaptive intervals. How much does that
   beat the best global $\kappa$?

4. Compute $H^{*}$ numerically for the second listing's parameters, and check the
   $600$-step row against it.

5. Model batched gates: pausing before groups of consequential steps. Does the lower
   $g$ recover enough catch rate to beat unbatched targeting?

6. Combine the two listings — re-validation and targeted gates in one run — and
   check whether the effects are additive or whether re-validation makes gates
   redundant.

## 18. Interview Questions

1. Why does per-step reliability stop being the right thing to optimise on a long
   run?

2. Your long-running agent completes and returns a wrong answer with no errors
   logged. What happened?

3. What does retry actually buy, and what does it convert step failure into?

4. Would you pause more often for approval on a week-long run than an hour-long one?

5. You have budget for twelve human approvals per run. Where do you put them?

6. At what point do you stop adding oversight gates, and what do you do instead?

## 19. Research Questions

1. Can premise volatility be estimated well enough online to drive adaptive
   re-validation?

2. What is the right formalisation of a reviewer's attention budget, and can
   oversight be scheduled against it?

3. Can $H^{*}$ be estimated from a trace rather than assumed?

4. How would an agent benchmark introduce controlled environment drift, and what
   would current systems score?

5. Does re-validation subsume human gating past some horizon, or are the mechanisms
   catching disjoint failure sets?

## 20. Chapter Summary

Long-running workflows fail differently, and the difference follows from recovery.
Retry converts step failure into budget consumption
({{eq:recovery-converts-failure-to-cost}}), and with a $1.6\times$ budget exhaustion
was $0.0\%$ at every horizon tested. **Budget is not what binds.**

What binds is **silent drift** — $94.3\%$ of failures at horizon $10$, $43.7\%$ at
$1000$ ({{eq:horizon-changes-the-failure}}). A drifted run completes with every step
green, because a premise expired and nothing looked again
({{eq:drift-is-silent}}).

Re-validation fixes it, and cheaply: horizon $300$ went from $0.7\%$ to $76.8\%$
correct for about $50\%$ more steps, with even two checks in the whole run reaching
$62.4\%$ ({{eq:revalidation-is-cheapest}}). At these parameters there is no interval
that is too frequent, and its value peaks in a *moderately* unstable environment
rather than a violently unstable one.

For human oversight, frequency fails: the catch rate fell from $63.8\%$ to $4.8\%$
as pauses went from four to two hundred, at $800$ hours of delay
({{eq:oversight-decays-in-frequency}}). **Placement is the lever** — twelve pauses before
consequential steps matched a hundred uniform ones at an eighth of the delay
({{eq:placement-beats-frequency}}), because a uniform pause interrupts at a moment
where being off-course does not yet cost anything.

And a limit: by horizon $600$ every gate design was at or under $1.2\%$.
**Gate-based oversight has a horizon past which it does not work at any budget**
({{eq:oversight-has-a-horizon-limit}}). Automated re-validation scales with the
horizon; human attention does not.

## 21. Further Reading

{{cite:shinn2023reflexion}} for reflection as a recovery mechanism, which
{{sec:13-alternatives}} treats as re-validation's more expensive generalisation.

{{cite:liu2024agentbench}} and {{cite:zhou2024webarena}} for long-horizon
evaluation — and read them against {{sec:15-advanced-concepts}}'s observation that
their static environments make drift unmeasurable by construction.

{{cite:cemri2025mast}} for the failure taxonomy this chapter's modes instantiate,
and {{cite:yao2023react}} for the loop whose per-step framing this chapter argues
does not survive a long horizon.

{{ch:ag-termination}} for the habituation result, and {{ch:as-state-machines}} for the
durability without which none of this survives the first restart.
