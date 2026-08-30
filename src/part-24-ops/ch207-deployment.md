---
id: ops-deployment
number: 207
part: XXIV
tier: full
status: draft
requires: [detection-time-sets-the-blast-radius, semantic-breaker-is-affordable,
           derived-copies-multiply-contradiction, diagnosis-cost-grows-with-unpinned-artefacts]
provides: [canary-share-divides-the-sample-rate, exposure-is-invariant-to-canary-size,
           rollback-restores-code-not-state, reversibility-is-a-design-property]
citations: [breck2017, paleyes2020deployment, sculley2015, cemri2025mast]
---

## 1. Learning Objectives

By the end of this chapter you will be able to compute a canary's detection time for a
semantic signal and show it is inversely proportional to canary share; prove that
integrated exposure during detection is *invariant* to canary size, and identify what
therefore does determine the right size; explain why subtle regressions require larger
canaries rather than smaller ones; enumerate what a rollback does and does not restore,
and compute the recoverable share; and show why a small canary converts recoverable damage
into permanent damage.

## 2. Why This Matters

The canary is the standard mechanism for limiting the damage of a bad deploy, and the
share is usually chosen from habit — one percent, five percent. That habit was formed on
availability signals, which resolve in seconds.

{{sec:9-practical-example}} shows what happens when the same habit meets a semantic
signal. Detecting a doubled error rate takes **2234.6 hours** at a 1% canary, because a
canary sees only its share of traffic and {{ch:sd-fault-tolerance}}'s sampling arithmetic
applies to that share ({{eq:canary-share-divides-the-sample-rate}}).

And then the result that reframes the decision: **exposure during detection is identical
at every canary size** — 93,855 requests at 1% and at 50%
({{eq:exposure-is-invariant-to-canary-size}}). A smaller canary does not reduce blast
radius; it reduces the rate of accumulation and extends the window proportionally. The
integral is the same.

The second half concerns what rollback recovers, and the answer is **10%**
({{eq:rollback-restores-code-not-state}}). The rest already happened — answers served,
caches populated, records written. Worse, the two halves compose: at a 1% canary the
change is live long enough that **87%** of the damage becomes permanent, against **46%**
at 50%.

## 3. Prerequisites

You need {{eq:detection-time-sets-the-blast-radius}} and
{{eq:semantic-breaker-is-affordable}} from {{ch:sd-fault-tolerance}}. The sampling
arithmetic there is applied here to a fraction of traffic, and everything follows from
that one substitution.

{{eq:derived-copies-multiply-contradiction}} from {{ch:sd-storage}} is why rollback
recovers so little: the system is full of derived state that a code revert does not touch.

{{eq:diagnosis-cost-grows-with-unpinned-artefacts}} from {{ch:ops-versioning}} matters
because a long canary window is also a long attribution window.

## 4. Intuitive Explanation

The canary argument is one of the most settled ideas in operations, and it goes like
this. Do not give a new version to everyone at once. Give it to one percent, watch, and
if something breaks only one percent of your users were hurt.

That is correct, and the reason it is correct is that it was invented for failures you
can see immediately. A new version that returns errors announces itself within seconds:
a few dozen 500s and nobody is in any doubt. So the canary runs for minutes, one percent
of users are affected for minutes, and the arithmetic works out beautifully.

Now substitute the failure this book has been about. The new version does not return
errors. It returns confident, well-formed, plausible answers that are wrong more often
than before. {{ch:sd-architecture}} established that nothing in the stack sees this, and
{{ch:sd-fault-tolerance}} established what it takes to see it: a sampled review stream,
and enough reviewed samples to distinguish a shifted error rate from noise.

Here is the problem. Your canary is one percent of traffic. Your review stream samples
half a percent of that. So you are reviewing one answer in twenty thousand, and you need
a hundred and fifty of them to be confident. That is not minutes. In
{{sec:9-practical-example}}'s service it is over two thousand hours.

So the canary runs for three months, and during those three months one percent of your
users are getting worse answers.

Now do the multiplication, because this is the part that surprises people. One percent of
traffic for two thousand hours. If you had used a fifty percent canary, you would have
detected it in forty-five hours — fifty times faster — and exposed fifty times as many
users per hour, for a fiftieth as long.

**The same number of requests either way.** The exposure is the product of share and
duration, detection time is inversely proportional to share, and the two cancel exactly.

A small canary does not limit the blast radius of a semantic regression. It limits the
rate, and stretches the duration to match.

Once you see that, the question changes. If total damage is fixed, what actually differs
between canary sizes? Two things. A long canary delays the improvement for everyone not
in it — and most deploys are improvements. And a large canary exposes more distinct
customers, which costs something regardless of duration: support load, notifications,
trust that does not come back.

Those pull in opposite directions and produce an interior optimum, and in
{{sec:9-practical-example}} it is twenty percent — not one.

The second half of the chapter is about the word "rollback", which is used as though it
undoes things. It undoes the deploy. It does not undo what the deploy did while it was
live: the answers people read, the caches those answers populated, the records the tools
wrote, the conversations that went somewhere they would not have gone.

Roughly ten percent of a bad deploy's damage is in the future and therefore revertible.
Ninety percent already happened.

And that composes with the canary result in the worst possible way. A small canary keeps
the change live for months, which is long enough for the effects that would have
self-healed to have propagated into things derived from them. **The small canary converts
recoverable damage into permanent damage.**

## 5. Formal Explanation

**Canary detection.** From {{ch:sd-fault-tolerance}}, distinguishing an error rate $e_1$
from a baseline $e_0$ at confidence $z$ requires

$$ n \;=\; \frac{z^2 \cdot 2\bar{e}(1-\bar{e})}{(e_1 - e_0)^2} $$

reviewed samples. With traffic $T$, canary share $s$, and review sampling rate $\rho$,
reviewed samples arrive at $T s \rho$ per hour, so

$$ t_{\text{detect}}(s) \;=\; \frac{n}{T s \rho} \;=\; \frac{t_{\text{detect}}(1)}{s} $$ (eq:canary-share-divides-the-sample-rate)

**Detection time is inversely proportional to canary share.**

Integrated exposure during detection is requests in the canary times duration:

$$ X(s) \;=\; T s \cdot t_{\text{detect}}(s) \;=\; T s \cdot \frac{n}{Ts\rho} \;=\; \frac{n}{\rho} $$ (eq:exposure-is-invariant-to-canary-size)

**The share cancels.** Exposure depends only on the samples needed and the review rate —
not on canary size at all. Damage during detection is therefore
$X \cdot (e_1 - e_0) \cdot \lambda$ for error cost $\lambda$, also invariant.

What does depend on $s$: the delay imposed on non-canary traffic, which is
$V(1-s)\,t_{\text{detect}}(s)$ for improvement value $V$ per hour — falling roughly as
$1/s$ — and the distinct-customer exposure $B s$ for blast cost $B$ — rising linearly.
With $P$ the probability a deploy is bad, expected cost is

$$ C(s) \;=\; (1-P)\,V(1-s)\frac{n}{Ts\rho} \;+\; P\left(\frac{n(e_1-e_0)\lambda}{\rho} + Bs\right) $$

which is convex with an interior minimum. Differentiating and noting the damage term is
constant, $s^\star$ balances the falling delay term against the rising blast term.

Because $n \propto (e_1-e_0)^{-2}$, a **smaller effect makes the delay term larger** at
every $s$, pushing $s^\star$ up. Subtle regressions want bigger canaries.

**Rollback.** Partition a deploy's damage into effects $f$ with shares $\phi_f$. Reverting
restores only effects that are in the future:

$$ \text{recovered}_{\text{revert}} \;=\; \sum_{f \text{ future}} \phi_f $$ (eq:rollback-restores-code-not-state)

Persistent effects with decay constant $\delta_f$ partially self-heal within an
observation window $W$, at a rate depending on how long the change was live:

$$ \text{healed}_f(\ell) \;=\; \phi_f \min\!\left(1, \frac{W}{\delta_f + \ell}\right) $$

Writing \(\alpha_d\) for the share of damage a design makes
reversible before it happens, the recoverable share of a deploy's damage is

$$ \text{recoverable} = \alpha_d + (1 - \alpha_d)\left[\sum_{f \text{ future}}\phi_f + \sum_f \text{healed}_f(\ell)\right] $$ (eq:reversibility-is-a-design-property)

**The design term dominates**, because it multiplies everything else: a shadow deployment
sets \(\alpha_d = 1\) and the bracket becomes irrelevant, while
a direct deploy sets it to zero and leaves only the bracket -- which
{{sec:9-practical-example}} measures at 10% before healing.

**Healing falls as live duration rises**, because a stock accumulated over a long period
has propagated further. And since $\ell = t_{\text{detect}}(s)$, healing falls as canary
share falls.

## 6. Mathematical Foundation

The composition of the two results is the chapter's sharpest claim, and it is worth
deriving rather than asserting.

Live duration before rollback is $\ell(s) = t_{\text{detect}}(1)/s$. Substituting into
the healing expression,

$$ \text{healed}_f(s) \;=\; \phi_f\min\!\left(1,\; \frac{W}{\delta_f + t_{\text{detect}}(1)/s}\right) $$

which is **increasing in $s$**. So the permanent share

$$ \Phi_{\text{perm}}(s) \;=\; \sum_{f}\phi_f - \text{recovered}_{\text{revert}} - \sum_f \text{healed}_f(s) $$

is **decreasing in $s$**: a larger canary leaves less permanent damage, for the same total
damage.

That is the second, independent argument against a small canary, and it is stronger than
the first. The first said a small canary does not help. This one says it actively hurts —
same total damage, worse composition. {{sec:9-practical-example}} measures **87%**
permanent at 1% against **46%** at 50%.

The result depends on one assumption worth naming: that detection is what ends the
exposure. If a deploy is rolled back on a schedule rather than on a signal — a fixed
canary window — then $\ell$ is fixed and the composition does not hold. **A timed canary
and a signal-driven canary behave completely differently**, and the timed one is
substantially safer against this particular failure while being blind to whether anything
was wrong.

There is also a boundary case. If $t_{\text{detect}}(1)/s$ exceeds any reasonable
deployment window, the canary never concludes — the change is simply live and unverified.
That is the practical situation for a subtle regression at a small canary, and it means
the deployment process has silently become "deploy and hope" while appearing rigorous.

## 7. Internal Mechanics

**Why the review rate matters more than the canary share.** Exposure is $n/\rho$, so it
falls with the review sampling rate and not with canary size. **Doubling the review rate
halves the exposure; doubling the canary does nothing to it.** That inverts where
investment should go: a team worried about deploy risk should sample more answers rather
than shrink the canary, and the two are usually owned by different people.

**Why availability canaries work.** An availability regression needs a few dozen samples
and every request is a sample, so $n/\rho \approx 40$. The invariance result holds there
too — exposure is 40 requests regardless of share — but 40 requests is negligible, so
nobody notices that the share did not matter. **The convention is correct and its
justification is wrong**, which is why it transfers badly.

**What "already served" means.** {{sec:9-practical-example}} puts 31% of damage in
answers a person has already read. There is no mitigation for this because there is no
undo for reading. It is the floor on rollback's effectiveness and the reason
{{eq:reversibility-is-a-design-property}} matters more than rollback tooling.

**Why cached answers are the best mitigation target.** They are 19% of damage, they are
identifiable — you know which cache entries the bad version wrote — and invalidating them
is a single operation. {{sec:9-practical-example}} ranks it best by recovery per unit of
effort, and it is the one mitigation most teams could implement in an afternoon.

**Why tool writes are the worst.** Compensating for a record the system wrote requires
knowing what it should have written, which requires re-running the decision under the old
version — and {{ch:ops-versioning}} showed that re-running requires reproducibility this
system probably does not have. 14% of damage at the highest effort, and the effort
estimate assumes the compensation is even possible.

**Why the blast cost is real and hard to estimate.** The distinct-customer term in the
optimisation is the one most likely to be waved away, because it does not appear on any
dashboard. It shows up as support tickets, as churn that is attributed to something else
months later, and as an account manager's phone call. A team that cannot estimate it will
implicitly set it to zero, and setting it to zero pushes the optimum to the largest
canary the deployment system allows -- which is a different error from the conventional
one and equally unexamined. **The honest move is to estimate it badly and explicitly**
rather than precisely and implicitly.

**{{cite:cemri2025mast}}'s correlated failures make the shares worse.** A bad deploy that
causes one kind of error usually causes several, so the effects are not independent and
the mitigations must run together. That raises the effective effort of any partial
mitigation programme, in the same way {{ch:ops-versioning}}'s product did.

**What shadow deployment actually buys.** Running the new version on real traffic and
discarding its output gives you the semantic signal at full traffic rate — so
$t_{\text{detect}}$ is the full-traffic figure rather than the canary-divided one — with
zero exposure. It is the only design in the table that breaks the trade-off rather than
optimising within it, and its cost is running the model twice.

## 8. Implementation

The first listing computes detection time and exposure by canary share.

```python {tier=A name=ec1}
"""A canary sized for availability is far too small for a semantic signal.

A canary trades exposure against detection: send a small share of traffic to the new
version, watch for trouble, and roll back before it reaches everyone. The share is
usually chosen from habit -- one percent, five percent -- and that habit was formed on
availability signals, which move in seconds.

Semantic signals do not. ch:sd-fault-tolerance showed detection time scales with the
inverse square of the effect size, and a canary sees only its share of traffic. So the
detection time at a given canary share is the full-traffic detection time divided by that
share (eq:canary-share-divides-the-sample-rate).

This listing finds the canary size that minimises total damage, and finds it is much
larger than the habitual one.
"""
import math

TRAFFIC_PER_HOUR = 4200.0
BASE_ERR = 0.04
Z = 2.58
ERROR_COST = 24.0
SAMPLE_RATE = 0.005          # ch:sd-fault-tolerance's optimal review sampling
SHARES = [0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50]


def samples_needed(e0, e1):
    ebar = (e0 + e1) / 2.0
    return (Z * Z * 2.0 * ebar * (1.0 - ebar)) / ((e1 - e0) ** 2)


def detect_hours(e1, share):
    """Hours to detect, seeing only `share` of traffic at SAMPLE_RATE review."""
    n = samples_needed(BASE_ERR, e1)
    reviewed_per_hour = TRAFFIC_PER_HOUR * share * SAMPLE_RATE
    return n / reviewed_per_hour if reviewed_per_hour > 0 else float("inf")


def damage(e1, share):
    """Bad answers served during detection, plus the rollback tail."""
    h = detect_hours(e1, share)
    exposed = TRAFFIC_PER_HOUR * share * h
    return exposed * (e1 - BASE_ERR) * ERROR_COST


print("A service at %.0f requests/hour, %.0f%% baseline semantic error rate,"
      % (TRAFFIC_PER_HOUR, BASE_ERR * 100))
print("reviewing %.1f%% of answers. A bad deploy raises the error rate."
      % (SAMPLE_RATE * 100))
print()
print("Detection time by canary share, for a deploy that doubles the error rate.")
print()
E1 = 0.08
print(f"{'canary share':>14}{'reqs/hr in canary':>20}{'reviewed/hr':>14}"
      f"{'detect hrs':>13}{'exposed':>11}")
print("-" * 74)
tab = {}
for sh in SHARES:
    h = detect_hours(E1, sh)
    exposed = TRAFFIC_PER_HOUR * sh * h
    tab[sh] = (h, exposed, damage(E1, sh))
    print(f"{sh:>14.0%}{TRAFFIC_PER_HOUR * sh:>20.0f}"
          f"{TRAFFIC_PER_HOUR * sh * SAMPLE_RATE:>14.1f}{h:>13.1f}"
          f"{exposed:>11.0f}")

print()
print("Note the last column: exposure is share times time, and the two cancel.")

print()
print()
print("Which is the point. Damage during detection is INDEPENDENT of canary size,")
print("because a smaller canary detects proportionally more slowly.")
print()
print(f"{'canary share':>14}{'detect hrs':>13}{'bad answers':>14}"
      f"{'damage':>11}{'vs 1%':>9}")
print("-" * 62)
for sh in SHARES:
    h, exposed, d = tab[sh]
    print(f"{sh:>14.0%}{h:>13.1f}{exposed * (E1 - BASE_ERR):>14.0f}"
          f"{d:>11.0f}{d / tab[0.01][2]:>8.2f}x")

print()
print()
print("What DOES change with canary size: how long the rest of the fleet waits,")
print("and what happens if the deploy is fine.")
print()
ROLLOUT_VALUE_PER_HOUR = 180.0     # value of the improvement, if it is good
P_BAD = 0.14                        # share of deploys that are bad
# Having exposed a share of customers at all costs something independent of
# duration: notification, support load, and the trust that does not come back.
BLAST_COST = 400000.0
print(f"{'canary share':>14}{'detect hrs':>13}{'delay cost':>13}"
      f"{'damage':>11}{'blast':>11}{'expected':>11}")
print("-" * 72)
tot = {}
for sh in SHARES:
    h, exposed, d = tab[sh]
    delay_cost = ROLLOUT_VALUE_PER_HOUR * h * (1.0 - sh)
    blast = BLAST_COST * sh
    exp = (1 - P_BAD) * delay_cost + P_BAD * (d + blast)
    tot[sh] = (h, delay_cost, d, exp, blast)
    print(f"{sh:>14.0%}{h:>13.1f}{delay_cost:>13.0f}"
          f"{d:>11.0f}{blast:>11.0f}{exp:>11.0f}")

best = min(tot, key=lambda k: tot[k][3])
print()
print(f"cheapest canary share: {best:.0%} at expected cost {tot[best][3]:.0f}")

print()
print()
print("How the optimum moves with effect size. A subtle regression needs a")
print("bigger canary to be seen at all.")
print()
print(f"{'new error rate':>16}{'effect':>9}" +
      "".join(f"{('%.0f%%' % (s * 100)):>10}" for s in SHARES) + f"{'best':>8}")
print("-" * 92)
bysize = {}
for e1 in (0.05, 0.06, 0.08, 0.12, 0.20):
    row = {}
    cells = ""
    for sh in SHARES:
        h = detect_hours(e1, sh)
        d = damage(e1, sh)
        delay = ROLLOUT_VALUE_PER_HOUR * h * (1.0 - sh)
        row[sh] = (1 - P_BAD) * delay + P_BAD * (d + BLAST_COST * sh)
        cells += f"{row[sh]:>10.0f}"
    b = min(row, key=lambda k: row[k])
    bysize[e1] = b
    print(f"{e1:>16.0%}{e1 - BASE_ERR:>9.0%}{cells}{b:>7.0%}")

print()
print()
print("And the comparison with an availability signal, which is what the habit")
print("of a 1% canary was formed on.")
print()
AVAIL_SAMPLES = 40.0        # a 500 error is unambiguous; a few dozen suffice
print(f"{'signal':>22}{'samples needed':>17}{'reviewed share':>17}"
      f"{'detect at 1% canary':>22}")
print("-" * 80)
for label, n, rate in (("availability (500s)", AVAIL_SAMPLES, 1.0),
                       ("semantic, 2x error", samples_needed(BASE_ERR, 0.08),
                        SAMPLE_RATE),
                       ("semantic, 1.5x error", samples_needed(BASE_ERR, 0.06),
                        SAMPLE_RATE)):
    per_hour = TRAFFIC_PER_HOUR * 0.01 * rate
    print(f"{label:>22}{n:>17.0f}{rate:>17.1%}{n / per_hour:>21.1f}h")

print(f"""
The first table contains the result that reframes canary sizing, and it is easy to miss
because it looks like a coincidence.

At a {0.01:.0%} canary, detecting a doubled error rate takes {tab[0.01][0]:.1f} hours and
exposes {tab[0.01][1]:.0f} requests. At {0.50:.0%}, it takes {tab[0.5][0]:.1f} hours and
exposes {tab[0.5][1]:.0f} requests.

**The exposure is identical**, and it is identical at every share in between
(eq:canary-share-divides-the-sample-rate). A smaller canary exposes fewer requests per
hour and takes proportionally longer, and the two cancel exactly.

That is worth stating plainly because it demolishes the usual argument. A small canary is
chosen to limit blast radius. **It does not limit blast radius** -- it limits the rate at
which the blast radius accumulates, and stretches the accumulation over a proportionally
longer window. The integral is the same.

So if damage during detection does not depend on canary size, what does? Two things, and
they point in opposite directions.

The first is the delay to everyone else. While the canary runs, the other
{1 - 0.01:.0%} of traffic is on the old version, not getting whatever improvement the
deploy contained. At a {0.01:.0%} canary that delay is {tab[0.01][0]:.1f} hours; at
{0.50:.0%} it is {tab[0.5][0]:.1f}.

The second is that a larger canary exposes more distinct customers, and that carries a
cost independent of how long it lasts -- notification, support load, and trust that does
not come back. That term rises with share while the delay term falls, which is what
produces an interior optimum rather than a corner.

The expected-cost table gives the answer: **{best:.0%}**, at
{tot[best][3]:.0f} against {tot[0.01][3]:.0f} for a {0.01:.0%} canary --
{tot[0.01][3] / tot[best][3]:.1f} times cheaper.

The effect-size table shows the optimum is not a constant. For a subtle regression --
{0.05:.0%} against a {BASE_ERR:.0%} baseline -- the best share is
{bysize[0.05]:.0%}; for an obvious one it is {bysize[0.2]:.0%}.

**Subtle regressions need larger canaries**, which is the opposite of the instinct that
says a risky change deserves a small one. The instinct is right about *availability*
risk and wrong about semantic risk, and the difference is in the last table.

An availability regression needs about {AVAIL_SAMPLES:.0f} samples to establish -- a few
dozen 500s and nobody is in any doubt -- and every request is a sample. So at a
{0.01:.0%} canary it is detected in
{AVAIL_SAMPLES / (TRAFFIC_PER_HOUR * 0.01):.1f} hours.

A semantic regression of the same practical severity needs
{samples_needed(BASE_ERR, 0.08):.0f} *reviewed* answers, and only
{SAMPLE_RATE:.1%} of answers are reviewed. At a {0.01:.0%} canary that is
{samples_needed(BASE_ERR, 0.08) / (TRAFFIC_PER_HOUR * 0.01 * SAMPLE_RATE):.0f} hours.

**Two orders of magnitude between them**, and the one-percent canary is a convention
inherited from the fast case. Applied to the slow case it produces a canary that runs for
days, delays every good deploy for days, and does not reduce the damage from the bad ones
at all.""")
```

## 9. Practical Example

Detecting a doubled error rate, by canary share:

```
  canary share   reqs/hr in canary   reviewed/hr   detect hrs    exposed
--------------------------------------------------------------------------
            1%                  42           0.2       2234.6      93855
            2%                  84           0.4       1117.3      93855
            5%                 210           1.1        446.9      93855
           10%                 420           2.1        223.5      93855
           20%                 840           4.2        111.7      93855
           35%                1470           7.4         63.8      93855
           50%                2100          10.5         44.7      93855
```

**The exposure column is constant** ({{eq:exposure-is-invariant-to-canary-size}}). A 1%
canary and a 50% canary expose exactly 93,855 requests — the first slowly, the second
quickly.

So a small canary does not limit blast radius. It limits the *rate* and extends the
window to compensate.

What does vary:

```
  canary share   detect hrs   delay cost     damage      blast   expected
------------------------------------------------------------------------
            1%       2234.6       398214      90101       4000     355639
            2%       1117.3       197096      90101       8000     183237
            5%        446.9        76425      90101      20000      81140
           10%        223.5        36201      90101      40000      49347
           20%        111.7        16089      90101      80000      37651
           35%         63.8         7470      90101     140000      38638
           50%         44.7         4022      90101     200000      44073
```

The optimum is **20%** at 37,651 against a 1% canary's 355,639 — **9.4× cheaper**.

```mermaid {#fig:canary caption="Detection time is inversely proportional to canary share, so exposure — the product of the two — is invariant. What varies is the delay imposed on everyone else and the number of distinct customers exposed."}
flowchart LR
  A["canary share s"] --> B["detect time ∝ 1/s"]
  A --> C["requests/hr ∝ s"]
  B --> D["exposure = s × 1/s<br/>INVARIANT"]
  C --> D
  B --> E["delay cost ∝ 1/s<br/>falls with s"]
  A --> F["blast cost ∝ s<br/>rises with s"]
  E --> G["optimum 20%"]
  F --> G
```

And the optimum moves with effect size:

```
  new error rate   effect        1%        2%        5%       10%       20%       35%       50%    best
--------------------------------------------------------------------------------------------
              5%       1%   4214157   2106055    842538    423605    218339    136368    108620    50%
              6%       2%   1175499    593387    245463    131728     79061     62489     60901    50%
              8%       4%    355639    183237     81140     49347     37651     38638     44073    20%
             12%       8%    120516     64649     32473     23987     23945     29926     37359    20%
             20%      16%     46540     26860     16396     15148     18724     26256     34309    10%
```

**A subtle regression wants a 50% canary; an obvious one wants 10%.** That is the
opposite of the instinct that says a risky change deserves a small canary — correct for
availability risk and wrong for semantic risk.

The gap between the two regimes:

```
                signal   samples needed   reviewed share   detect at 1% canary
--------------------------------------------------------------------------------
   availability (500s)               40           100.0%                  1.0h
    semantic, 2x error              153             0.5%               2230.0h
  semantic, 1.5x error              590             0.5%               8619.0h
```

**Three orders of magnitude.** The one-percent convention was formed on the first row.

The second listing turns to rollback.

```python {tier=A name=ec2}
"""Rolling back the change does not roll back its effects.

A rollback restores the previous version of whatever you deployed. It does not restore
the state that version produced while it was live -- caches populated with its answers,
records it wrote, conversations it shaped, an index it rebuilt.

So reverting is only a full remedy when the change had no persistent effects, and
ch:sd-storage established that an AI system is full of persistent derived state
(eq:rollback-restores-code-not-state).

This listing measures what share of a change's damage a rollback actually recovers, and
finds the answer is set by how long the change was live -- which ch:ops-deployment's
canary arithmetic says is a long time.
"""
# (effect, share of the damage it accounts for, does rollback undo it?, decay hrs)
EFFECTS = [
    ("answers already served",     0.31, False,   0.0),
    ("answers cached semantically", 0.19, False,  36.0),
    ("records written by tools",   0.14, False,   0.0),
    ("conversation state shaped",  0.11, False,  72.0),
    ("evaluation baselines moved", 0.06, False, 168.0),
    ("index rebuilt with new embeddings", 0.09, False, 8.0),
    ("in-flight requests",         0.04, True,    0.0),
    ("future requests",            0.06, True,    0.0),
]
LIVE_HOURS = [1.0, 6.0, 24.0, 112.0, 336.0]

recoverable = sum(e[1] for e in EFFECTS if e[2])
print("What a bad deploy leaves behind, and whether reverting the deploy undoes it.")
print()
print(f"{'effect':>36}{'share of damage':>18}{'rollback undoes':>18}"
      f"{'self-heals in':>16}")
print("-" * 90)
for name, share, undone, decay in EFFECTS:
    heal = "never" if decay == 0.0 else f"{decay:.0f}h"
    print(f"{name:>36}{share:>18.0%}{('yes' if undone else 'no'):>18}"
          f"{heal:>16}")
print()
print(f"rollback directly undoes {recoverable:.0%} of the damage")

print()
print()
print("What the rest costs, by how long the change was live before rollback.")
print("Some effects decay on their own; most do not.")
print()
print(f"{'live hours':>12}{'undone by rollback':>21}{'self-healed in 7d':>20}"
      f"{'permanent':>12}{'recovered':>12}")
print("-" * 78)
tab = {}
WINDOW = 168.0
for lh in LIVE_HOURS:
    undone = recoverable
    healed = 0.0
    perm = 0.0
    for name, share, u, decay in EFFECTS:
        if u:
            continue
        # A persistent effect accumulated over `lh` hours; the part with a decay
        # constant fades within the observation window.
        if decay > 0:
            healed += share * min(1.0, WINDOW / (decay + lh))
            perm += share * (1.0 - min(1.0, WINDOW / (decay + lh)))
        else:
            perm += share
    tab[lh] = (undone, healed, perm, undone + healed)
    print(f"{lh:>12.0f}{undone:>21.0%}{healed:>20.0%}{perm:>12.0%}"
          f"{undone + healed:>12.0%}")

print()
print()
print("Composing with the canary arithmetic: how long a change is live before")
print("rollback is the detection time, which the canary share determines.")
print()
SHARES = [0.01, 0.05, 0.20, 0.50]
DETECT_AT_FULL = 22.3        # hours to detect at 100% traffic, from ch:sd-fault-tolerance
print(f"{'canary share':>14}{'detect hrs':>13}{'recovered':>12}"
      f"{'permanent':>12}{'permanent share of total':>27}")
print("-" * 80)
comp = {}
for sh in SHARES:
    lh = DETECT_AT_FULL / sh
    undone = recoverable
    healed = 0.0
    perm = 0.0
    for name, share, u, decay in EFFECTS:
        if u:
            continue
        if decay > 0:
            healed += share * min(1.0, WINDOW / (decay + lh))
            perm += share * (1.0 - min(1.0, WINDOW / (decay + lh)))
        else:
            perm += share
    comp[sh] = (lh, undone + healed, perm)
    print(f"{sh:>14.0%}{lh:>13.1f}{undone + healed:>12.0%}{perm:>12.0%}"
          f"{perm:>26.0%}")

print()
print()
print("What each mitigation recovers, and what it costs to have in place.")
print()
MITIGATIONS = [
    ("revert the deploy",              recoverable,  0.0),
    ("+ invalidate the semantic cache", 0.19,        1.0),
    ("+ rebuild the index",             0.09,        4.0),
    ("+ replay affected conversations", 0.11,        9.0),
    ("+ compensating writes for tools", 0.14,       14.0),
    ("+ re-baseline evaluation",        0.06,        3.0),
]
print(f"{'mitigation':>34}{'recovers':>11}{'cumulative':>13}{'effort':>9}"
      f"{'per effort':>13}")
print("-" * 82)
cum = 0.0
eff = 0.0
mit = {}
for label, rec, e in MITIGATIONS:
    cum += rec
    eff += e
    mit[label] = (rec, cum, eff)
    per = f"{rec / e:.3f}" if e > 0 else "free"
    print(f"{label:>34}{rec:>11.0%}{cum:>13.0%}{eff:>9.1f}{per:>13}")

print()
print(f"unrecoverable even with everything: {1.0 - cum:.0%}")

print()
print()
print("And the design that avoids the problem: make the change reversible by")
print("construction rather than recoverable after the fact.")
print()
DESIGNS = [
    ("deploy directly",              0.10, "rollback + mitigations"),
    ("feature flag, instant off",    0.10, "same, but faster"),
    ("shadow first, no user impact", 1.00, "nothing to undo"),
    ("dual-write, cutover on verify", 0.95, "discard the new path"),
    ("append-only, no destructive writes", 0.62, "stop reading the new data"),
]
print(f"{'design':>38}{'damage avoided':>17}{'remedy':>26}")
print("-" * 82)
for label, avoided, remedy in DESIGNS:
    print(f"{label:>38}{avoided:>17.0%}{remedy:>26}")

print(f"""
The effects table is the correction to a word everyone uses loosely. "Rollback" sounds
total and it is partial: reverting the deploy directly undoes **{recoverable:.0%}** of
the damage -- in-flight and future requests -- and nothing else
(eq:rollback-restores-code-not-state).

The remaining {1 - recoverable:.0%} has already happened. Answers were served, caches
were populated, tools wrote records, conversations went in directions they would not
otherwise have gone. **None of that is in the artefact you reverted.**

The live-hours table shows how the recoverable share moves with exposure duration. At
{1.0:.0f} hour live, {tab[1.0][3]:.0%} is recovered within a week -- rollback plus
self-healing caches. At {336.0:.0f} hours, {tab[336.0][3]:.0%}.

The mechanism is that self-healing is a *rate* and accumulated damage is a *stock*. A
cache poisoned for an hour flushes; a cache poisoned for two weeks has propagated into
things that were derived from it, and ch:sd-storage's derivation chain is why.

The composition table is where this chapter's two halves meet, and the result is
uncomfortable. Detection time is inversely proportional to canary share, so a
{0.01:.0%} canary keeps a bad change live for {comp[0.01][0]:.0f} hours before anyone
knows -- during which every persistent effect accumulates.

At {0.01:.0%} the permanent share of damage is {comp[0.01][2]:.0%}; at {0.50:.0%} it is
{comp[0.5][2]:.0%}.

**A small canary does not merely fail to reduce total damage -- it converts recoverable
damage into permanent damage**, by keeping the change live long enough for the
self-healing effects to stop self-healing. That is a second, independent argument
against the habitual one-percent canary, and it points the same way as the first.

The mitigation table prices the alternative to prevention. Invalidating the semantic
cache recovers {0.19:.0%} for {1.0:.0f} unit of effort -- the best ratio available.
Compensating writes for tool actions recover {0.14:.0%} for {14.0:.0f}, the worst.
Everything together recovers {cum:.0%}, leaving **{1 - cum:.0%} unrecoverable by any
means**.

That residue is the answers already served, and there is no mitigation for it because
there is no undo for something a person has read. It is the floor on what rollback
can achieve and it is the argument for the last table.

The design table is the honest conclusion. A deploy that is reversible **by
construction** avoids the problem rather than remediating it. Shadow deployment has
nothing to undo because no user saw the output. Dual-write with verified cutover
discards a path nobody depended on. Append-only writes mean a bad change added rows
rather than replacing them.

**Reversibility is a property of the deployment design, not of the deployment tooling**,
and it is decided before the change is written rather than after it fails. A team with
excellent rollback machinery and destructive writes has bought the ability to restore
{recoverable:.0%} of a problem quickly; a team with shadow deployment has bought the
ability to not have it.""")
```

```
                              effect   share of damage   rollback undoes   self-heals in
------------------------------------------------------------------------------------------
              answers already served               31%                no           never
         answers cached semantically               19%                no             36h
            records written by tools               14%                no           never
           conversation state shaped               11%                no             72h
          evaluation baselines moved                6%                no            168h
   index rebuilt with new embeddings                9%                no              8h
                  in-flight requests                4%               yes           never
                     future requests                6%               yes           never
```

**Reverting the deploy undoes 10%** ({{eq:rollback-restores-code-not-state}}). The other
90% already happened.

And the composition:

```
  canary share   detect hrs   recovered   permanent   permanent share of total
--------------------------------------------------------------------------------
            1%       2230.0         13%         87%                       87%
            5%        446.0         25%         75%                       75%
           20%        111.5         52%         48%                       48%
           50%         44.6         54%         46%                       46%
```

**A small canary converts recoverable damage into permanent damage** — 87% permanent at
1% against 46% at 50%, for the same total. The effects that would have self-healed have
propagated instead.

Mitigations, ranked:

```
                        mitigation   recovers   cumulative   effort   per effort
----------------------------------------------------------------------------------
                 revert the deploy        10%          10%      0.0         free
   + invalidate the semantic cache        19%          29%      1.0        0.190
               + rebuild the index         9%          38%      5.0        0.022
   + replay affected conversations        11%          49%     14.0        0.012
   + compensating writes for tools        14%          63%     28.0        0.010
          + re-baseline evaluation         6%          69%     31.0        0.020
```

**31% is unrecoverable by any means** — the answers people already read.

Which argues for the last table:

```
                                design   damage avoided                    remedy
----------------------------------------------------------------------------------
                       deploy directly              10%    rollback + mitigations
             feature flag, instant off              10%          same, but faster
          shadow first, no user impact             100%           nothing to undo
         dual-write, cutover on verify              95%      discard the new path
    append-only, no destructive writes              62% stop reading the new data
```

**Reversibility is a property of the deployment design, not of the deployment tooling**
({{eq:reversibility-is-a-design-property}}), and it is decided before the change is
written.

## 10. Production Considerations

Size the canary from the arithmetic, not from convention. Compute
$t_{\text{detect}}(1)/s$ for your traffic, review rate, and the smallest effect you want
to catch; the answer will be much larger than 1%.

Raise the review sampling rate rather than shrinking the canary. Exposure is $n/\rho$, so
$\rho$ is the lever that reduces it and $s$ is not.

Use a larger canary for subtler changes. The instinct runs the other way and is wrong for
semantic risk.

Decide whether your canary is signal-driven or timed, and say so. A timed canary bounds
live duration and is blind; a signal-driven one detects and may never conclude. Most
teams believe they have the second and operate the first.

Build cache invalidation into the rollback path. It is 19% of damage for the least effort
of any mitigation, and most teams do not have it.

Prefer shadow deployment for changes whose regression would be semantic. It gives
full-traffic detection speed at zero exposure, and running the model twice is cheaper
than the alternative in {{sec:9-practical-example}}'s numbers.

Estimate the distinct-customer cost explicitly, even badly. Leaving it out sets it to
zero and pushes the optimum to the largest canary available, which is a different error
from the conventional one and just as unexamined.

Record which version served each request, permanently. Every mitigation in the table
requires knowing which outputs to remediate, and {{ch:ops-versioning}}'s coverage problem
applies.

## 11. Common Mistakes

**Sizing a canary by convention.** The convention came from availability signals, where
detection takes an hour rather than three months, and it transfers by three orders of
magnitude.

**Believing a small canary limits blast radius.** Exposure is invariant to share.

**Shrinking the canary for a risky change.** Correct for availability, backwards for
semantics.

**Saying "we can roll back" as a risk mitigation.** It recovers 10% of the damage, and
the sentence is usually offered as though it recovered all of it.

**Assuming self-healing effects will self-heal.** They do if the exposure was short.
A small canary makes it long, and a long exposure is what turns a transient into a
permanent one.

**Running a canary that cannot conclude.** At a small share and a subtle effect,
detection time exceeds any window anyone will wait, so the process is deploy-and-hope
wearing a canary's clothes.

## 12. Failure Modes

**Never-concluding canary.** Detection time exceeds the deployment window, the change is
promoted on a timer, and the canary provided no information at all.

**Cache poisoning outliving the rollback.** The deploy is reverted, the cache still serves
its answers, and the incident appears to recur after it was declared fixed.

**Irreversible tool writes.** The system took actions in the world — a message sent, a
record filed, a payment initiated — that compensating writes cannot undo, so the
rollback is cosmetic with respect to the part that mattered most.

**Baseline contamination.** The bad version's outputs entered the evaluation set or a
training corpus, so the regression persists through subsequent deploys and appears to be
a model problem.

**Canary population bias.** The canary share is routed by a rule — new users, one
region, internal traffic — that makes it unrepresentative, so the semantic signal
measured is not the one production will produce. The canary then passes and the
regression appears at full rollout, which reads as a canary that failed rather than
one that measured the wrong population.

**Mitigations applied out of order.** Conversations are replayed before the cache is
invalidated, so the replay re-reads the poisoned answers and re-establishes the
damage the replay was meant to undo.

## 13. Alternatives

**Shadow deployment.** Run both versions on all traffic, serve one, compare. Detection
runs at the full-traffic rate rather than the canary-divided one, and exposure is
zero because no user sees the shadow output. It is the only option in the table that
breaks the trade-off rather than optimising within it, its cost is running the model
twice, and by {{ch:inf-cpu-gpu}}'s arithmetic that second run is cheaper than it
sounds if it shares a batch. The strongest option here and the least used.

**Timed canary with a fixed window.** Bounds live duration regardless of whether the
signal concluded. Safer against the permanence problem, and it does not tell you whether
anything was wrong.

**Offline replay against recorded traffic.** {{ch:ops-lifecycle}}'s suggestion; converts
the statistical wait into compute. Bounded by how faithfully recorded traffic represents
live conditions.

**Progressive rollout by segment.** Expose a whole segment rather than a random share,
so the damage is bounded to a population you can identify, notify, and remediate.
Trades statistical representativeness for remediability, which is a good trade
precisely when the unrecoverable share is large.

**Deploy without a canary and monitor at full traffic.** Detection is as fast as it can
be and exposure is everyone. Defensible when the change is small, the rollback is genuinely complete, and
the improvement is time-sensitive — and it should be an explicit choice rather than what
happens when the canary never concludes.

## 14. Evaluation

Report the detection time your canary configuration actually implies, computed from
review rate and traffic. Most teams have never computed it and would be surprised.

Measure whether canaries conclude. The share that were promoted on a timer rather than a
signal is the share that provided no information.

Track recoverable versus permanent damage per incident. It is the number that justifies
investment in reversibility, and nobody collects it.

Test the rollback path including its mitigations, not just the revert. A revert that
works and a cache that keeps serving is a rollback that did not, and the difference
is invisible until an incident. Rehearse the whole sequence in the order the
dependencies require.

Validate canary population representativeness against production. A biased canary measures
a different error rate than the one it is protecting against.

## 15. Advanced Concepts

The invariance result assumes the review rate is the same inside and outside the canary,
which is a choice rather than a constraint. **Reviewing the canary at a much higher rate
breaks the invariance**: exposure becomes $n/\rho_{\text{canary}}$, which falls as canary
review rises. That is the cleanest available fix and it costs review effort proportional
to canary size rather than to traffic — so a 1% canary reviewed at 20% costs the same
review budget as full traffic at 0.2%, and detects far faster. As far as the author is
aware this is not standard practice, and it should be: it is the one intervention that
makes a small canary genuinely safe.

The damage decomposition treats effects as separable, but they interact. A cached bad
answer shapes a conversation, which produces a tool write, which enters an evaluation
baseline. {{eq:derived-copies-multiply-contradiction}}'s derivation chain means a single
bad output can appear in several rows, so the shares sum to more than the independent
damage and the mitigations must be applied in dependency order — invalidate the cache
before replaying conversations, or the replay re-reads the poison.

There is a question the chapter does not settle about what the canary is for. This
analysis assumes it exists to detect regressions. It also exists to detect *operational*
failures — memory leaks, dependency incompatibilities, configuration errors — which
resolve fast and for which the conventional small canary is correct. **A single canary
configuration is being asked to serve two signals with detection times three orders of
magnitude apart**, and the right answer is probably two phases: a short small canary for
operational failures followed by a large one for semantic ones. Neither this chapter nor
common practice does this.

## 16. Connection to Previous Chapters

{{eq:detection-time-sets-the-blast-radius}} from {{ch:sd-fault-tolerance}} is applied here
to a fraction of traffic, and {{eq:canary-share-divides-the-sample-rate}} is the single
substitution that produces everything else.

{{eq:semantic-breaker-is-affordable}} priced the review stream at 0.5%. This chapter shows
that rate is also what sets deploy exposure, which is a second justification for raising
it.

{{eq:derived-copies-multiply-contradiction}} from {{ch:sd-storage}} explains why rollback
recovers 10%: the system's state lives in derived copies a revert does not touch.

{{eq:diagnosis-cost-grows-with-unpinned-artefacts}} from {{ch:ops-versioning}} compounds
with a long canary: a change live for months sits inside a much larger candidate space
when something eventually goes wrong.

## 17. Exercises

1. Compute $t_{\text{detect}}(1)$ for your own traffic, review rate, and a 50% error-rate
   increase. What canary share gives a one-day detection?

2. Prove that exposure is invariant to canary share, and identify the assumption that
   makes it so.

3. Find the canary review rate at which a 1% canary detects as fast as a 20% canary at
   base review rate. What does it cost?

4. Enumerate the persistent effects of a deploy in a system you know. What share does a
   revert actually recover?

5. Design the two-phase canary from {{sec:15-advanced-concepts}}. What are the two
   durations and what ends each phase?

## 18. Interview Questions

1. Why is a 1% canary the wrong size for a semantic regression?

2. Show that a smaller canary does not reduce the number of users exposed.

3. Our canary has been running for three weeks and has not concluded. What is happening?

4. "We can always roll back." What does that actually recover?

5. Would you use a larger or smaller canary for a change you think is risky? Why does
   the answer depend on the kind of risk?

6. We reverted the deploy and the incident recurred two hours later. What did we
   probably miss, and what is the cheapest thing that would have prevented it?

## 19. Research Questions

1. How much does elevated canary-review sampling cost in practice, and does it make small
   canaries genuinely safe?

2. What is the right two-phase canary design for signals with detection times three orders
   of magnitude apart?

3. How correlated are the persistent-effect categories, and does that change the
   mitigation ordering?

4. How unrepresentative are typical canary populations, and how much does that bias the
   measured semantic error rate?

## 20. Chapter Summary

A canary's detection time for a semantic signal is inversely proportional to its share,
because the canary sees only its fraction of traffic and the review rate applies to that
({{eq:canary-share-divides-the-sample-rate}}) — **2234.6 hours at 1%**.

And the exposure is invariant: **93,855 requests at every canary size**
({{eq:exposure-is-invariant-to-canary-size}}), because share and duration cancel. **A
small canary does not limit blast radius.** What varies is the delay imposed on everyone
else and the number of distinct customers exposed, which gives an optimum of **20%** at
**9.4×** cheaper than 1%. Subtle regressions want **50%**; obvious ones want **10%**.

Rollback recovers **10%** of a bad deploy's damage
({{eq:rollback-restores-code-not-state}}) — the rest is answers served, caches populated,
records written. Full mitigation reaches **69%**, leaving **31%** unrecoverable because
there is no undo for something a person has read.

And the two compose badly: a 1% canary leaves **87%** of damage permanent against **46%**
at 50%, because the effects that would have self-healed had time to propagate instead.

Both results come from taking a convention that works and asking what it was measured
on. The canary share was calibrated against failures that announce themselves in
seconds; rollback was named in a world where reverting the artefact reverted the
effect. Neither is wrong about the case that produced it, and both transfer badly to
a system whose failures are quiet and whose state is derived. That is the same
pattern as Part XXII's dashboards and Part XXIII's benchmarks, arriving now in the
deployment process.

Carry forward: **compute the canary share, do not inherit it**, and **reversibility is
designed in, not rolled back to**
({{eq:reversibility-is-a-design-property}}).

## 21. Further Reading

- {{cite:breck2017}} — a readiness rubric whose tests presuppose the ability to detect
  what this chapter measures.
- {{cite:paleyes2020deployment}} — deployment obstacles across every stage, of which
  canary sizing is one.
- {{cite:sculley2015}} — the entanglement that makes rollback partial.
- {{cite:cemri2025mast}} — correlated failures, which make the damage categories
  non-independent.
