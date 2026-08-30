---
id: sec-permissions
number: 227
part: XXVI
tier: full
status: draft
requires: [approval-must-sit-at-the-outcome-not-the-call, agent-authority-exceeds-requester-authority,
           guardrail-precision-is-set-by-the-base-rate, cause-distance-drives-triage-cost]
provides: [approval-quality-falls-with-volume, a-low-rejection-rate-trains-approval,
           delegation-preserves-authority-unless-attenuated, audit-completeness-requires-the-principal-chain]
citations: [beurerkellner2025patterns, hou2025mcp, cemri2025mast, breck2017]
---

## 1. Learning Objectives

By the end of this chapter you will be able to compute an approval queue's effective catch
rate from its volume and its rejection density, and locate the volume that maximises bad
actions caught; explain why a wider gate can catch fewer bad things than a narrower one;
design a risk-weighted sampling review and compare it against reviewing everything; show that
delegation preserves authority unless something explicitly attenuates it; compare permission
models by what principal they can express; and identify the audit fields required to answer
"why" rather than "who".

## 2. Why This Matters

{{ch:sec-tool-abuse}} concluded that approval must sit at the outcome rather than the call.
This chapter is about the human on the other end of that queue, and about the permission
system underneath it.

A reviewer with **360 minutes a day** and a queue of **20,640** approvals gets **0.7 seconds
an item**. Detection at that budget is **0.012**
({{eq:approval-quality-falls-with-volume}}). And a queue rejecting **0.04%** of items teaches,
correctly, that the next one is fine: the care multiplier is **0.185** against **0.784** at an
8% rejection rate ({{eq:a-low-rejection-rate-trains-approval}}).

Both terms move together, so a wider gate covering **99%** of bad actions catches **0.02** a
day while a narrow one covering **44%** catches **1.10** — **55× more from 109× fewer
approvals.**

The permission half has a matching inversion. Along a six-hop chain, the user holds **0.08**
of the available authority and the request executes with **1.00**, because nothing between the
entry point and the backend removes anything
({{eq:delegation-preserves-authority-unless-attenuated}}). **A chain of trust is as weak as its
weakest link; a chain of authority is as strong as its most privileged member.**

And audit records answer the wrong question. Acting identity is recorded **98%** of the time
and the full principal chain **19%**; together the top two settle **39%** of audit questions
against **93%** for all six ({{eq:audit-completeness-requires-the-principal-chain}}).

## 3. Prerequisites

{{eq:approval-must-sit-at-the-outcome-not-the-call}} from {{ch:sec-tool-abuse}} is the design
this chapter validates from the human side: it was recommended there because a per-call gate
cannot express a composition, and it is recommended here because a per-call gate cannot be
read.

{{eq:agent-authority-exceeds-requester-authority}} from the same chapter is the starting point
for the delegation half — this chapter follows the excess along the chain and finds nothing
removing it.

{{eq:guardrail-precision-is-set-by-the-base-rate}} from {{ch:sec-jailbreaks}} is the same
arithmetic in a queue rather than a classifier: a rare event produces alarms that are mostly
wrong, and here the consequence is habituation rather than refusals.

{{eq:cause-distance-drives-triage-cost}} from {{ch:ops-agent-tracing}} is why the audit fields
that matter are the ones about *why* rather than *who* — the cause sits several steps back from
the acting identity.

{{cite:breck2017}}'s readiness rubric is the closest prior art for the governance artefacts in
{{sec:9-practical-example}}'s second listing.

## 4. Intuitive Explanation

Human approval is the strongest control anywhere in this part. It is also the one most likely
to be silently disabled — not by a decision, but by degradation.

Two things degrade it and they compound.

The first is arithmetic. A reviewer has a time budget. Six productive hours is 360 minutes. If
the queue has 20 items, that is 18 minutes each and the reviewer reads carefully. If the queue
has 20,640 items — which is what a per-call gate on a moderately busy agent generates — that is
0.7 seconds each.

Nobody decided to stop reviewing. The time per item is the queue's reciprocal, and the queue is
set by where the gate sits.

The second is a reinforcement schedule. How carefully someone reads depends on how often
reading carefully changed the outcome. A queue where 0.04% of items are rejected has, over four
thousand consecutive approvals, taught that the next one is probably fine. That lesson is
*correct*, and it is not undone by a training day.

In the model here the care multiplier at a 0.04% rejection rate is 0.185. At 8% it is 0.784 —
four times the effective catch from the same person on the same items in the same time.

Now put both together, and there is an interesting consequence.

A wider gate covers more of the bad actions — it intercepts more of them — and generates more
items, which spends scrutiny. So the number of bad actions actually caught has an interior
maximum.

Run it: a gate at 20 approvals a day covers only 6% of bad actions and catches 0.22 a day. A
gate at 20,640 covers 100% and catches 0.02, because each item gets 0.7 seconds. The maximum is
1.10 at around 200 approvals a day.

**Widening a gate buys coverage and spends scrutiny, and past a point the second term
dominates.**

Apply that to {{ch:sec-tool-abuse}}'s queue designs. Approving every tool call has the best
coverage on the list — 99% — and catches 0.02 bad actions a day. Approving only where taint
reaches covers 44% and catches 1.10. **Fifty-five times more, from a hundred and nine times
fewer approvals.**

That is the same conclusion that chapter reached from the composition side, arriving here from
the human side, and the agreement is worth noting: a per-call gate fails for two independent
reasons.

There is a third design that sounds like giving up and is not. Instead of reviewing everything
shallowly, review a *sample* deeply — and weight the sample by risk. Reading 2% of items
risk-weighted gives 0.87 minutes each and reaches 31% of the bad actions, catching more than
reviewing all of them at 0.7 seconds apiece.

That is {{ch:ops-observability}}'s sampling result in the approval queue: uniform coverage of a
rare event is the expensive way to see nothing.

Before leaving the queue, one honest note. A rubber-stamped approval is not worthless. It still
produces a decision record. It still puts a human in the causal chain. It still imposes a delay
during which the action can be cancelled. Two of those are why the control was funded and all
three survive habituation.

What does not survive is the review and the rejection — the control itself. That distinction
belongs in the design document, because a queue producing the first three and not the last two
is an accountability mechanism carrying a security mechanism's name, and the confusion is what
prevents anyone from fixing the volume.

Now the permission system underneath.

A user calls an agent. The agent calls a planning sub-agent, which calls a retrieval
sub-agent, which calls a tool server, which calls a backend API. Five hops.

At each hop, some authority is passed along. The default — in every system that has not
thought about it — is to pass all of it, because attenuating requires deciding what to remove
and the code that forwards a request has no opinion about that.

Count it. The user holds 0.08 of the available authority. The orchestrator runs as a service
account and holds 0.94. Every hop after that holds 0.94 too, and the backend holds 1.00.

The request executes with 1.00. The user who asked for it holds 0.08.

Look at where the authority enters: the user-to-orchestrator transition is +0.86, and every
transition after it is +0.00 or, at one hop, negative by accident because the tool server
happens to have less — which is immediately undone when the server uses its own credential.

**The only narrowing on the chain is accidental.**

That is an inversion of the usual intuition and it is worth saying carefully. A chain of
*trust* is as weak as its weakest link: if any component is compromised, the whole thing is. A
chain of *authority* is as strong as its most privileged member, because privilege flows
downhill and nothing removes it.

So what would remove it? That depends on what the permission model can express.

RBAC on the service account scores 0.11 on expressiveness. Its principal is the *agent*, so
every request from every user looks identical to the policy engine — the policy literally
cannot mention the user.

RBAC on the user scores 0.44: better, and it does not survive the hops, because the sub-agent
does not know who the user was.

ABAC with request attributes reaches 0.71. A delegated on-behalf-of token reaches 0.88 and can
attenuate. A capability minted per task reaches 0.97.

**Only the last two carry the user's identity to the far end, and only they can narrow on the
way.** Which is the same delegation result {{ch:sd-apis-auth}} reached from the API side and
{{ch:sec-tool-abuse}} from the authority side, arriving now as a statement about what the
policy engine can see.

Finally, governance, which in practice means: after something goes wrong, can you reconstruct
why?

Six fields matter. The acting identity — who called the API. The originating user — who asked.
The full principal chain — every hop in order. The task the chain served. The content that
triggered it. The authority actually exercised.

Recorded in practice: acting identity 98%, originating user 62%, principal chain 19%, task 14%,
triggering content 9%, exercised authority 7%.

Together the top two settle 39% of audit questions. All six settle 93%.

The gap is the questions that begin "why did it do that" rather than "who did it" — and those
are the questions an incident actually asks. {{ch:ops-agent-tracing}} found the cause sits
several steps back from where the failure became visible; an audit log recording only the
acting identity has recorded the last step.

Closing the gap is cheap. Propagating a chain header settles 44% for half a unit of effort —
the best ratio on the list. Four additions totalling four units take answerability to 93%.

One of those additions deserves a flag. "Record the content that triggered it" is the same
field {{ch:ops-agent-tracing}} argued for on triage grounds and {{ch:sec-data-leakage}} counted
as the largest single leak source. **The same field is the most valuable audit record and the
biggest privacy exposure**, and the resolution is the one that chapter reached — record it,
redact at emit — rather than a choice between the two.

## 5. Formal Explanation

**Approval effectiveness.** With review budget $M$ minutes and queue volume $v$, time per item
is $M/v$ and detection is $d(M/v)$, increasing and saturating. With rejection rate $r$,
habituation multiplies scrutiny by $h(r)$, increasing and saturating. If a gate at volume $v$
intercepts a share $c(v)$ of the $B$ bad actions, the expected catch is

$$K(v) = B\,c(v)\,d\!\left(\tfrac{M}{v}\right) h\!\left(\tfrac{B c(v)}{v}\right).$$

$c$ is increasing and $d$ decreasing in $v$, and $h$'s argument $Bc(v)/v$ is decreasing for
concave $c$. So two of three factors fall with $v$ and $K$ has an interior maximum.

**Sampling.** Reviewing a share $s$ of items with bad-item coverage $\beta(s)$ gives
$K = B\beta(s)d(M/(sv))h(B\beta(s)/(sv))$. Under uniform sampling $\beta(s) = s$; under
risk-weighted sampling $\beta(s) \gg s$. Since $d$ and $h$ both improve as $s$ falls, the
risk-weighted design dominates on all three factors simultaneously.

**Delegation.** Let hop $i$ hold authority $a_i$ and pass forward $\pi_i(a)$. Preservation is
$\pi_i(a) = \max(a, a_i)$; attenuation is $\pi_i(a) = \gamma a$ for $\gamma < 1$;
least-privilege is $\pi_i(a) = a_{\text{user}}$. Under preservation the executed authority is
$\max_i a_i$, independent of the requester — **the chain's authority is a maximum, not a
minimum**, which is the opposite of the composition rule for trust.

**Expressiveness.** A policy is a predicate over the information available at the decision
point. If the principal presented is the agent, no predicate can distinguish requests
originating from different users, so the achievable policy set is bounded by what the
credential carries. Expressiveness is therefore a property of the *token*, not of the policy
language.

**Audit completeness.** Let question classes $j$ be settled by field sets $F_j$. Answerability
with recorded set $R$ is $\sum_j w_j \mathbf{1}[F_j \subseteq R]$. Because "why" questions
require the chain, the task and the trigger, and "who" questions require only the acting
identity, a deployment recording the cheapest fields answers the cheapest questions.

## 6. Mathematical Foundation

Approval catch as a product of three terms, two of which fall with volume:

$$K(v) = B\,c(v)\,d\!\left(\tfrac{M}{v}\right)h\!\left(\tfrac{Bc(v)}{v}\right), \qquad \frac{\partial d}{\partial v} < 0,\ \frac{\partial h}{\partial v} < 0$$ (eq:approval-quality-falls-with-volume)

At $M = 360$, $B = 8.3$: maximum **1.10** at $v = 200$; **0.02** at $v = 20{,}640$ with
**0.7 seconds** an item.

Habituation as a function of density alone:

$$h(r) = 0.18 + 0.82\left(1 - e^{-r/0.06}\right), \qquad \frac{h(0.08)}{h(0.0005)} = 4.2$$ (eq:a-low-rejection-rate-trains-approval)

Same reviewer, same items, same 0.88 minutes each — **4.2× the effective catch from density**.

Authority under default forwarding:

$$a_{\text{exec}} = \max_i a_i = 1.00 \quad \text{while} \quad a_{\text{user}} = 0.08$$ (eq:delegation-preserves-authority-unless-attenuated)

with attenuation at $\gamma = 0.72$ per hop reaching **0.015**.

And audit answerability:

$$A(R) = \sum_j w_j \mathbf{1}[F_j \subseteq R], \qquad A(\{\text{acting}, \text{user}\}) = 39\%, \quad A(\text{all six}) = 93\%$$ (eq:audit-completeness-requires-the-principal-chain)

## 7. Internal Mechanics

Why do approval queues grow rather than shrink? Because each addition is individually
justified. An incident happens, a review finds the action should have been approved, and the
action class is added to the gate. Nothing removes a class, because removing one requires
arguing that a failure is now acceptable. So the queue is a monotone accumulation of past
incidents, and its volume is a function of organisational history rather than of current risk.

**The gate's volume is set by a process with no downward pressure**, which is why it reliably
ends up past the maximum of $K(v)$.

The habituation mechanism has a property that makes it hard to detect from inside. A reviewer
who is approving without reading feels the same as one who is reading and finding nothing
wrong — the observable behaviour is identical, and both produce a queue that empties. The only
distinguishing measurement is time-per-item, or a planted item that should be rejected. Neither
is standard, and the first is often available in the tooling and never looked at.

On the delegation side, the reason preservation is the default is worth being precise about.
Forwarding a request means constructing an outbound call from an inbound one, and the simplest
correct implementation reuses whatever credential the process holds. Attenuating requires the
forwarding code to know *which subset* of authority this particular downstream call needs,
which is a per-call-site decision that nobody has made and that changes when the downstream
changes. **Preservation is what you get when you do not have a policy; attenuation requires
one per hop.**

That interacts badly with sub-agent architectures. {{cite:cemri2025mast}}'s multi-agent failure
taxonomy is about coordination; the security version is that each additional agent is another
hop that preserves authority, so a system decomposed into more agents has a longer chain with
the same maximum. **Decomposition does not attenuate.**

On audit, the reason the cheap fields dominate is that they are the ones a framework emits by
default. Acting identity is in every request log because the auth layer put it there. The
principal chain is not, because nothing in a standard stack constructs one — it requires a
header propagated through every hop, and any hop that drops it breaks the chain silently. This
is the same fragility {{ch:ops-agent-tracing}} found for trace context, and it has the same
remedy: propagate at the framework level rather than at the call site.

Finally, the tension between the audit field and the leak source. Recording the triggering
content is the single most valuable audit field — it settles 51% of questions — and
{{ch:sec-data-leakage}} counted payload-bearing logs as 44.9% of leaked records. Those are the
same bytes. The resolution is field-level redaction at emit, which preserves the diagnostic
structure and removes the sensitive substrings, and it requires knowing which fields are
sensitive — which is why it is skipped and why both chapters end up recommending it.

## 8. Implementation

The first listing measures the approval queue.

```python {tier=A name=C:/Users/MANAVB~1/AppData/Local/Temp/claude/C--Github-LaughingBuddha/30a87753-43a8-48c3-8378-261faf976dbb/scratchpad/ig1}
"""An approval queue where almost everything is fine trains the reviewer to approve.

Human approval is the strongest control in ch:sec-tool-abuse's table and the one most likely
to be quietly disabled. It is not disabled by a decision; it degrades, because the quantity
that determines how carefully a reviewer reads is how often reading carefully changed the
outcome.

Two mechanisms. Volume: a fixed time budget spread over more items gives less time per item,
so scrutiny falls (eq:approval-quality-falls-with-volume). And base rate: a queue where almost
nothing is rejected provides no reinforcement for care
(eq:a-low-rejection-rate-trains-approval).

A wider gate covers more of the bad actions and destroys the reviewer's ability to see any of
them, so the product has an interior maximum -- and it is far below what a per-call gate
generates.
"""
import math

REVIEW_MINUTES_PER_DAY = 6.0 * 60.0
BAD_PER_DAY = 8.3              # genuinely bad actions attempted per day


def scrutiny(volume):
    """Minutes per item, and the detection that supports."""
    mins = REVIEW_MINUTES_PER_DAY / volume
    return mins, 1.0 - math.exp(-mins / 1.4)


def habituation(reject_rate):
    """Care multiplier: a reviewer who never rejects stops looking."""
    return 0.18 + 0.82 * (1.0 - math.exp(-reject_rate / 0.06))


def coverage(volume):
    """A wider gate intercepts more of the bad actions."""
    return 1.0 - math.exp(-volume / 330.0)


print(f"{REVIEW_MINUTES_PER_DAY:.0f} reviewer-minutes a day, "
      f"{BAD_PER_DAY:.1f} bad actions attempted.")
print()
print(f"{'approvals/day':>15}{'coverage':>10}{'min/item':>11}{'scrutiny':>10}"
      f"{'reject rate':>13}{'habituation':>13}{'catch':>8}{'bad caught':>12}")
print("-" * 92)
tab = {}
for v in (20, 60, 200, 600, 2000, 6000, 20640):
    mins, d = scrutiny(v)
    cov = coverage(v)
    rej = BAD_PER_DAY * cov / v
    h = habituation(rej)
    eff = d * h
    caught = BAD_PER_DAY * cov * eff
    tab[v] = (cov, mins, d, rej, h, eff, caught)
    print(f"{v:>15,}{cov:>10.3f}{mins:>11.2f}{d:>10.3f}{rej:>13.2%}"
          f"{h:>13.3f}{eff:>8.3f}{caught:>12.2f}")

best_v = max(tab, key=lambda v: tab[v][6])
print()
print(f"maximum bad actions caught: {tab[best_v][6]:.2f} a day at "
      f"{best_v:,} approvals")
print(f"at {20640:,} approvals: {tab[20640][6]:.2f} caught, "
      f"{tab[20640][2] * 60:.1f} seconds an item")

print()
print()
print("The two terms separately, at a fixed volume.")
print()
V = 410
mins_v, d_v = scrutiny(V)
print(f"At {V} approvals a day: {mins_v:.2f} minutes an item, "
      f"scrutiny {d_v:.3f}.")
print()
print(f"{'rejection rate':>16}{'habituation':>14}{'effective catch':>18}"
      f"{'vs 0.05%':>11}")
print("-" * 59)
rej_tab = {}
for r in (0.0005, 0.004, 0.02, 0.08, 0.25):
    h = habituation(r)
    rej_tab[r] = (h, d_v * h)
    print(f"{r:>16.2%}{h:>14.3f}{d_v * h:>18.3f}"
          f"{(d_v * h) / (d_v * habituation(0.0005)):>10.1f}x")

print()
print("Same reviewer, same items, same time budget. Only the density changed.")

print()
print()
print("Queue designs from ch:sec-tool-abuse, scored on both terms.")
print()
QUEUES = [
    ("approve every tool call",           20640, 0.99),
    ("approve non-reversible calls",       5250, 0.91),
    ("approve on a taint path only",        190, 0.44),
    ("approve by outcome class",            410, 0.62),
    ("approve outcome class, high-risk",     84, 0.28),
]
print(f"{'queue design':>36}{'items/day':>12}{'coverage':>11}"
      f"{'reject rate':>14}{'catch':>8}{'bad caught/day':>17}")
print("-" * 98)
q = {}
for name, v, cov in QUEUES:
    mins, d = scrutiny(v)
    rej = BAD_PER_DAY * cov / v
    h = habituation(rej)
    eff = d * h
    q[name] = (v, cov, rej, eff, BAD_PER_DAY * cov * eff)
    print(f"{name:>36}{v:>12,}{cov:>11.2f}{rej:>14.2%}{eff:>8.3f}"
          f"{BAD_PER_DAY * cov * eff:>17.2f}")

best_q = max(q, key=lambda n: q[n][4])
print()
print(f"best: {best_q} at {q[best_q][4]:.2f} bad actions caught a day")
print(f"worst: approve every tool call at "
      f"{q['approve every tool call'][4]:.2f}")

print()
print()
print("The alternative nobody proposes: sample deeply instead of reviewing all.")
print()
TOTAL = 20640
print(f"{'policy':>34}{'reviewed':>11}{'min/item':>11}{'catch':>8}"
      f"{'bad-item coverage':>20}{'bad caught/day':>17}")
print("-" * 101)
SAMPLES = [
    ("review everything",         1.000, 0.99),
    ("review 10%, uniform",       0.100, 0.099),
    ("review 2%, uniform",        0.020, 0.020),
    ("review 2%, risk-weighted",  0.020, 0.310),
    ("review 0.5%, risk-weighted", 0.005, 0.140),
]
for name, share, badcov in SAMPLES:
    n = TOTAL * share
    mins, d = scrutiny(n)
    rej = BAD_PER_DAY * badcov / n
    h = habituation(rej)
    eff = d * h
    print(f"{name:>34}{n:>11,.0f}{mins:>11.2f}{eff:>8.3f}"
          f"{badcov:>20.1%}{BAD_PER_DAY * badcov * eff:>17.2f}")

print()
print("Risk-weighted sampling reads 2% of items and reaches 31% of the bad")
print("ones, because the sample is not uniform.")

print()
print()
print("And what a rubber-stamped approval still produces.")
print()
ARTEFACTS = [
    ("a decision record",           "yes", "yes", "compliance, forensics"),
    ("a human in the causal chain", "yes", "yes", "accountability"),
    ("a delay before the action",   "yes", "yes", "a cancellation window"),
    ("an actual review",            "yes", "no",  "the control itself"),
    ("a rejection when warranted",  "yes", "no",  "the control itself"),
]
print(f"{'what approval produces':>32}{'careful review':>17}"
      f"{'rubber stamp':>15}{'why it matters':>25}")
print("-" * 89)
for name, careful, stamp, why in ARTEFACTS:
    print(f"{name:>32}{careful:>17}{stamp:>15}{why:>25}")

print(f"""
The main table is both mechanisms at once and the last column is the finding. A gate
generating {20:,} approvals a day gives {tab[20][1]:.1f} minutes an item and catches
{tab[20][6]:.2f} bad actions -- limited by coverage, since it only intercepts
{tab[20][0]:.1%} of them. A gate generating {20640:,} covers {tab[20640][0]:.1%} and catches
{tab[20640][6]:.2f}, because each item gets {tab[20640][2] * 60:.1f} seconds
(eq:approval-quality-falls-with-volume).

**The maximum is {tab[best_v][6]:.2f} bad actions a day at {best_v:,} approvals**, and both
ends of the table are far worse than the middle. Widening a gate buys coverage and spends
scrutiny, and past a point the second term dominates.

The rejection table isolates the other mechanism, holding volume and time fixed. Same
reviewer, same {mins_v:.2f} minutes an item, same items. At a {0.0005:.2%} rejection rate the
care multiplier is {rej_tab[0.0005][0]:.3f}; at {0.08:.0%} it is {rej_tab[0.08][0]:.3f} --
**{rej_tab[0.08][1] / rej_tab[0.0005][1]:.1f} times the effective catch from density alone**
(eq:a-low-rejection-rate-trains-approval).

That is not a criticism of reviewers. It is what a reinforcement schedule does. A queue in
which four thousand consecutive items were fine has taught, correctly, that the next one
probably is, and no amount of training-day emphasis survives that gradient.

The queue table applies both to ch:sec-tool-abuse's designs. `approve every tool call` has the
best coverage on the list at {q['approve every tool call'][1]:.2f} and catches
{q['approve every tool call'][4]:.2f} bad actions a day. `{best_q}` covers
{q[best_q][1]:.2f} and catches {q[best_q][4]:.2f} --
**{q[best_q][4] / q['approve every tool call'][4]:.0f} times more, from
{q['approve every tool call'][0] / q[best_q][0]:.0f} times fewer approvals.**

Which is the same recommendation ch:sec-tool-abuse reached from the composition side, arriving
here from the human side. It was recommended there because a per-call gate cannot express a
composition; it is recommended here because a per-call gate cannot be read.

The sampling table is the design that sounds like giving up and is not. Reviewing
{0.02:.0%} of items risk-weighted gives {scrutiny(TOTAL * 0.02)[0]:.2f} minutes each, reaches
{0.31:.0%} of the bad actions, and catches more than reviewing everything at
{scrutiny(TOTAL)[0] * 60:.1f} seconds an item.

**A deep review of a biased sample beats a shallow review of everything**, which is
ch:ops-observability's sampling result in the approval queue: uniform coverage of a rare event
is the expensive way to see nothing.

The last table is the honest accounting of a rubber stamp, because the answer is not nothing.
It still produces a decision record, still puts a human in the causal chain, and still imposes
a delay during which the action can be cancelled. Two of those are why the control was funded
and all three survive habituation.

What does not survive is the review and the rejection -- **the control itself**. Which is worth
writing down in the design document, because a queue producing the first three and not the last
two is an accountability mechanism carrying a security mechanism's name, and that confusion is
what prevents anybody from fixing the volume.""")
```

## 9. Practical Example

Both mechanisms at once, at 360 reviewer-minutes and 8.3 bad actions a day:

```
  approvals/day  coverage   min/item  scrutiny  reject rate  habituation   catch  bad caught
--------------------------------------------------------------------------------------------
             20     0.059      18.00     1.000        2.44%        0.454   0.454        0.22
            200     0.455       1.80     0.724        1.89%        0.401   0.290        1.10
            600     0.838       0.60     0.349        1.16%        0.324   0.113        0.79
          2,000     0.998       0.18     0.121        0.41%        0.235   0.028        0.23
         20,640     1.000       0.02     0.012        0.04%        0.185   0.002        0.02
```

**Maximum 1.10 at 200 approvals; 0.02 at 20,640** with 0.7 seconds an item
({{eq:approval-quality-falls-with-volume}}). Widening buys coverage and spends scrutiny.

```
  rejection rate   habituation   effective catch   vs 0.05%
-----------------------------------------------------------
           0.05%         0.187             0.087       1.0x
           2.00%         0.412             0.192       2.2x
           8.00%         0.784             0.365       4.2x
          25.00%         0.987             0.460       5.3x
```

Same reviewer, same items, same 0.88 minutes each — **4.2× from density alone**
({{eq:a-low-rejection-rate-trains-approval}}).

```
                        queue design   items/day   coverage   reject rate   catch   bad caught/day
--------------------------------------------------------------------------------------------------
             approve every tool call      20,640       0.99         0.04%   0.002             0.02
        approve non-reversible calls       5,250       0.91         0.14%   0.010             0.07
        approve on a taint path only         190       0.44         1.92%   0.300             1.10
            approve by outcome class         410       0.62         1.26%   0.156             0.80
    approve outcome class, high-risk          84       0.28         2.77%   0.460             1.07
```

**55× more caught from 109× fewer approvals** — the same conclusion
{{ch:sec-tool-abuse}} reached from composition, arriving from the human side.

```
                            policy   reviewed   min/item   catch   bad-item coverage   bad caught/day
-----------------------------------------------------------------------------------------------------
                 review everything     20,640       0.02   0.002               99.0%             0.02
               review 10%, uniform      2,064       0.17   0.022                9.9%             0.02
          review 2%, risk-weighted        413       0.87   0.121               31.0%             0.31
        review 0.5%, risk-weighted        103       3.49   0.294               14.0%             0.34
```

**A deep review of a biased sample beats a shallow review of everything.**

```
      what approval produces   careful review   rubber stamp           why it matters
--------------------------------------------------------------------------------------
           a decision record              yes            yes    compliance, forensics
 a human in the causal chain              yes            yes           accountability
   a delay before the action              yes            yes    a cancellation window
              an actual review            yes             no       the control itself
    a rejection when warranted            yes             no       the control itself
```

The second listing follows the authority.

```python {tier=A name=C:/Users/MANAVB~1/AppData/Local/Temp/claude/C--Github-LaughingBuddha/30a87753-43a8-48c3-8378-261faf976dbb/scratchpad/ig2}
"""Delegation preserves authority by default, so a chain is as strong as its strongest link.

When a user calls an agent, which calls a sub-agent, which calls a tool server, each hop
passes some authority along. The default in every system that does not think about it is to
pass *all* of it, because attenuating requires a decision about what to remove and nobody made
one (eq:delegation-preserves-authority-unless-attenuated).

That inverts the usual security intuition. A chain of trust is as weak as its weakest link; a
chain of *authority* is as strong as its most privileged member, and the privilege flows
downhill to whatever is at the end.

The second half is what an audit needs to reconstruct the decision, and the answer is the
whole principal chain rather than the acting identity
(eq:audit-completeness-requires-the-principal-chain).
"""
# (hop, authority it holds alone, does it attenuate by default?)
HOPS = [
    ("the user",            0.08, "-"),
    ("the orchestrator",    0.94, "no"),
    ("a planning sub-agent", 0.94, "no"),
    ("a retrieval sub-agent", 0.94, "no"),
    ("an MCP tool server",  0.61, "no"),
    ("the backend API",     1.00, "-"),
]

print("Authority along a call chain, with and without attenuation.")
print()
print(f"{'hop':>24}{'own authority':>16}{'preserved':>12}{'attenuated':>13}"
      f"{'least-privilege':>18}")
print("-" * 83)
preserved, atten, lp = 0.08, 0.08, 0.08
rows = []
for name, own, att in HOPS:
    preserved = max(preserved, own) if name != "the user" else own
    atten = atten * 0.72 if name != "the user" else own
    lp = 0.08
    rows.append((name, own, preserved, atten, lp))
    print(f"{name:>24}{own:>16.2f}{preserved:>12.2f}{atten:>13.3f}"
          f"{lp:>18.2f}")

print()
print(f"the user holds {HOPS[0][1]:.2f} and the request executes with "
      f"{preserved:.2f}")
print(f"attenuating 28% a hop would end at {atten:.3f}")

print()
print()
print("What each hop can do that the previous one could not.")
print()
GAINS = [
    ("the user -> orchestrator",     0.08, 0.94, "the service account"),
    ("orchestrator -> planner",      0.94, 0.94, "nothing, and nothing removed"),
    ("planner -> retriever",         0.94, 0.94, "nothing, and nothing removed"),
    ("retriever -> tool server",     0.94, 0.61, "narrower, by accident"),
    ("tool server -> backend",       0.61, 1.00, "the server's own credential"),
]
print(f"{'transition':>30}{'before':>10}{'after':>9}{'delta':>9}"
      f"{'what changed':>32}")
print("-" * 90)
for name, a, b, why in GAINS:
    print(f"{name:>30}{a:>10.2f}{b:>9.2f}{b - a:>+9.2f}{why:>32}")

print()
print("Only one transition narrows authority, and it does so because the")
print("tool server happens to have less, not because anything attenuated.")

print()
print()
print("Permission models, and what each can express about an agent.")
print()
MODELS = [
    ("RBAC on the service account", "the agent's role", 0.11, 0.0, "no"),
    ("RBAC on the user",            "the user's role",  0.44, 1.0, "no"),
    ("ABAC with request attributes", "user + resource + action", 0.71, 2.5, "partly"),
    ("delegated token (on-behalf-of)", "user, via the agent", 0.88, 3.5, "yes"),
    ("capability per task",          "this task's resources", 0.97, 5.0, "yes"),
]
print(f"{'model':>34}{'principal':>28}{'expressiveness':>16}"
      f"{'effort':>9}{'attenuates?':>13}")
print("-" * 100)
mod = {}
for name, prin, expr, eff, att in MODELS:
    mod[name] = (expr, eff, att)
    print(f"{name:>34}{prin:>28}{expr:>16.2f}{eff:>9.1f}{att:>13}")

print()
print("Only the last two carry the user's identity to the far end, and only")
print("they can narrow on the way.")

print()
print()
print("Audit reconstruction: what a record must contain to answer 'why'.")
print()
FIELDS = [
    ("acting identity",            "who called the API",        0.98, 0.11),
    ("originating user",           "who asked",                 0.62, 0.31),
    ("the full principal chain",   "every hop, in order",       0.19, 0.44),
    ("the task the chain served",  "what it was for",           0.14, 0.29),
    ("the content that triggered it", "which input, verbatim",  0.09, 0.51),
    ("the authority actually used", "which scope was exercised", 0.07, 0.38),
]
print(f"{'field':>32}{'what it answers':>28}{'recorded in practice':>23}"
      f"{'share of questions it settles':>32}")
print("-" * 115)
cum_miss = 1.0
for name, what, recorded, settles in FIELDS:
    cum_miss *= (1 - settles)
    print(f"{name:>32}{what:>28}{recorded:>23.0%}{settles:>32.0%}")

print()
print(f"if all six were recorded, {1 - cum_miss:.0%} of audit questions are")
print("answerable; in practice the top two carry most deployments")

top2 = 1 - (1 - FIELDS[0][3]) * (1 - FIELDS[1][3])
print(f"acting identity plus originating user alone: {top2:.0%}")

print()
print()
print("What it costs to record the missing four.")
print()
ADD = [
    ("propagate a chain header",       0.44, 0.5),
    ("attach the task id at entry",    0.29, 0.3),
    ("record the triggering content",  0.51, 2.0),
    ("record the scope exercised",     0.38, 1.2),
]
print(f"{'addition':>34}{'settles':>11}{'effort':>9}{'per effort':>13}"
      f"{'cumulative answerable':>24}")
print("-" * 92)
cum = top2
for name, settles, eff in ADD:
    cum = 1 - (1 - cum) * (1 - settles)
    print(f"{name:>34}{settles:>11.0%}{eff:>9.1f}{settles / eff:>13.3f}"
          f"{cum:>24.0%}")

print()
print(f"four additions, {sum(e for n, s, e in ADD):.1f} units of effort, "
      f"{cum:.0%} answerable")

print()
print()
print("And the governance question underneath: what a policy can be about.")
print()
LEVELS = [
    ("a tool",            "may the agent call it",     "static",  "no"),
    ("a resource",        "may this record be touched", "static", "partly"),
    ("a principal chain", "did the user authorise this", "dynamic", "yes"),
    ("an outcome",        "is money moving",           "dynamic", "yes"),
    ("a task",            "is this within the ask",    "dynamic", "yes"),
]
print(f"{'policy is about':>22}{'the question it asks':>32}{'evaluation':>13}"
      f"{'survives a new tool?':>23}")
print("-" * 90)
for name, q, ev, surv in LEVELS:
    print(f"{name:>22}{q:>32}{ev:>13}{surv:>23}")

print(f"""
The chain table is the default and it is worth staring at. The user holds
{HOPS[0][1]:.2f} of the available authority. The request executes with {preserved:.2f},
because the orchestrator runs as a service account and **nothing between the entry point and
the backend removes anything** (eq:delegation-preserves-authority-unless-attenuated).

An explicit attenuation of {1 - 0.72:.0%} a hop would end at {atten:.3f}. Least privilege
would end at {HOPS[0][1]:.2f}. Neither is what happens by default, because attenuating requires
somebody to decide what to remove at each hop and the code that forwards a request has no
opinion.

The transition table shows where authority enters. `{GAINS[0][0]}` is
{GAINS[0][2] - GAINS[0][1]:+.2f} -- the service account -- and every hop after it is
{GAINS[1][2] - GAINS[1][1]:+.2f}. **The only narrowing on the list happens by accident**,
because the tool server happens to hold less, and it is undone at the next hop when the server
uses its own credential.

That is the inversion worth naming. A chain of *trust* is as weak as its weakest link. A chain
of *authority* is as strong as its most privileged member, and privilege flows downhill.

The models table is what can be expressed. `RBAC on the service account` scores
{mod['RBAC on the service account'][0]:.2f} on expressiveness because its principal is the
agent -- so every request from every user looks identical to the policy engine, and the policy
cannot mention the user at all.

`{MODELS[3][0]}` scores {mod[MODELS[3][0]][0]:.2f} and `{MODELS[4][0]}` scores
{mod[MODELS[4][0]][0]:.2f}. **Only the last two carry the user's identity to the far end, and
only they can narrow on the way** -- which is the same delegation result ch:sd-apis-auth
reached from the API side and ch:sec-tool-abuse from the authority side.

The audit table is the governance half. Six fields; the acting identity is recorded
{FIELDS[0][2]:.0%} of the time and the full principal chain {FIELDS[2][2]:.0%}.

Together the top two settle {top2:.0%} of audit questions
(eq:audit-completeness-requires-the-principal-chain), and all six settle
{1 - cum_miss:.0%}. The gap is the questions that begin "why did it do that" rather than
"who did it", and those are the ones an incident actually asks.

The addition table prices closing it. Propagating a chain header settles
{ADD[0][1]:.0%} for {ADD[0][2]:.1f} units of effort -- the best ratio on the list -- and four
additions totalling {sum(e for n, s, e in ADD):.1f} units take answerability to {cum:.0%}.

Note what `record the triggering content` is: it is ch:ops-agent-tracing's payload field,
already argued for on triage grounds and already argued *against* in ch:sec-data-leakage's leak
accounting. **The same field is the most valuable audit record and the largest leak source**,
and the resolution is the same one that chapter reached -- record it, redact at emit -- rather
than a choice between the two.

The last table is the governance point. A policy about a *tool* is static and does not survive
the next integration; a policy about an *outcome* or a *task* is dynamic and does. That is
ch:sec-tool-abuse's approval result in policy form, and it is the reason a permission system
built around a tool list has to be rewritten every time the product grows and one built around
outcomes does not.""")
```

```
                     hop   own authority   preserved   attenuated   least-privilege
-----------------------------------------------------------------------------------
                the user            0.08        0.08        0.080              0.08
        the orchestrator            0.94        0.94        0.058              0.08
    a planning sub-agent            0.94        0.94        0.041              0.08
      an MCP tool server            0.61        0.94        0.021              0.08
         the backend API            1.00        1.00        0.015              0.08
```

The user holds **0.08** and the request executes with **1.00**
({{eq:delegation-preserves-authority-unless-attenuated}}) — **a chain of authority is as strong
as its most privileged member.**

```
                    transition    before    after    delta                    what changed
------------------------------------------------------------------------------------------
      the user -> orchestrator      0.08     0.94    +0.86             the service account
       orchestrator -> planner      0.94     0.94    +0.00    nothing, and nothing removed
      retriever -> tool server      0.94     0.61    -0.33           narrower, by accident
        tool server -> backend      0.61     1.00    +0.39     the server's own credential
```

**The only narrowing is accidental, and it is undone at the next hop.**

```
                             model                   principal  expressiveness   effort  attenuates?
----------------------------------------------------------------------------------------------------
       RBAC on the service account            the agent's role            0.11      0.0           no
                  RBAC on the user             the user's role            0.44      1.0           no
      ABAC with request attributes    user + resource + action            0.71      2.5       partly
    delegated token (on-behalf-of)         user, via the agent            0.88      3.5          yes
               capability per task       this task's resources            0.97      5.0          yes
```

**Expressiveness is a property of the token, not of the policy language.**

```
                           field             what it answers   recorded in practice   share of questions it settles
-------------------------------------------------------------------------------------------------------------------
                 acting identity          who called the API                    98%                             11%
                originating user                   who asked                    62%                             31%
        the full principal chain         every hop, in order                    19%                             44%
   the content that triggered it       which input, verbatim                     9%                             51%
     the authority actually used   which scope was exercised                     7%                             38%
```

Top two settle **39%**; all six settle **93%**
({{eq:audit-completeness-requires-the-principal-chain}}) — the gap is every question beginning
"why".

```
                          addition   settles   effort   per effort   cumulative answerable
-------------------------------------------------------------------------------------------
        propagate a chain header       44%      0.5        0.880                      66%
     attach the task id at entry       29%      0.3        0.967                      76%
   record the triggering content       51%      2.0        0.255                      88%
    record the scope exercised        38%      1.2        0.317                      93%
```

## 10. Production Considerations

Measure your approval queue's volume, rejection rate and time-per-item. All three are in the
tooling and none is on a dashboard.

Compute the volume that maximises bad actions caught, and move the gate rather than the
reviewer. The current volume is a record of past incidents, not of current risk.

Weight the review sample by risk instead of reviewing everything. Two percent risk-weighted
beats a hundred percent shallow.

Plant an item that should be rejected, monthly. It is the only measurement that distinguishes
reading from approving.

Propagate a principal chain header at the framework level. It settles 44% of audit questions
for half a unit of effort and any hop that drops it breaks the chain silently.

Attenuate at each hop explicitly. Preservation is what you get by default and the default
executes at the maximum authority on the chain.

Record the triggering content and redact at emit. It is the most valuable audit field and the
largest leak source, and the resolution is the same one {{ch:sec-data-leakage}} reached.

## 11. Common Mistakes

**Adding approval classes after every incident.** The process has no downward pressure and the
volume ends past the maximum.

**Reading queue-empty as queue-reviewed.** They look identical from outside.

**Reviewing everything.** 0.7 seconds an item catches 0.02 bad actions a day.

**Assuming decomposition attenuates.** More agents is a longer chain with the same maximum.

**Choosing a policy language before a token.** Expressiveness is bounded by what the credential
carries.

**Auditing the acting identity.** It settles 11% of questions and is recorded 98% of the time.

## 12. Failure Modes

**Approval queue that empties in minutes.** Volume is high, rejection rate is near zero, and
every item was approved.

**Sub-agent with full authority.** Each hop forwarded what it held and nobody wrote an
attenuation policy.

**Chain header dropped at one hop.** The audit shows the last actor and the incident asks about
the first.

**Policy engine that cannot see the user.** The service account is the principal and every
request looks the same.

**Approval added, never removed.** Three years of incidents encoded as a queue nobody reads —
{{cite:cemri2025mast}}'s coordination failure with a governance consequence.

**Triggering content dropped for privacy.** The leak source was closed and the audit
answerability fell from 88% to 66%.

## 13. Alternatives

**Two-person approval on the highest class.** Doubles cost on a small volume and raises catch
substantially where damage is concentrated.

**Time-delayed execution with a cancel window.** No reviewer at all; the action happens unless
someone stops it. Keeps the delay and the record, drops the review.

**Post-hoc review with reversal.** Approve everything, review a sample afterwards, reverse what
was wrong. Works exactly as far as reversibility does — {{ch:sec-tool-abuse}}'s 57% permanent
damage bounds it.

**Capability tokens minted per task.** {{cite:beurerkellner2025patterns}}'s design and the
highest-expressiveness row here. Removes the need for most approvals by removing the authority.

**Policy-as-code with simulation.** Evaluate the policy against historical traffic before
deploying it. Standard in network security, rare in agent permissions, and it is what would
have caught the volume problem before the queue existed.

## 14. Evaluation

Measure time-per-item and rejection rate weekly. The product of the two predicts your catch
rate better than any statement about reviewer diligence.

Plant rejectable items and measure the catch. Monthly, unannounced, and it is the only direct
measurement available.

Compute $K(v)$ for your own numbers and find the maximum. Compare against your current volume.

Trace a request end to end and check whether the principal chain survives every hop. Any drop
is silent.

Audit a past incident against your recorded fields and count which questions you could not
answer. That number is your answerability.

## 15. Advanced Concepts

The habituation model treats care as a function of the aggregate rejection rate, and reviewers
are more sophisticated than that: they habituate *per class*. A queue mixing a high-density
class with a low-density one produces high care on the first and low on the second, and the
aggregate rate predicts neither. That argues for splitting queues by expected density rather
than by risk — the counterintuitive consequence being that **a class where almost everything is
fine should not share a queue with one where it is not**, because the shared rate teaches the
wrong lesson about both.

The interior maximum of $K(v)$ assumes a fixed review budget $M$. If $M$ scales with volume —
more reviewers hired as the queue grows — then $d(M/v)$ is constant and $K$ becomes increasing
in $v$, so widening the gate always helps. That is the implicit assumption behind "just staff
the queue," and it fails for the reason {{ch:ops-agent-tracing}} found for triage: the headcount
required grows linearly with a quantity that grows with traffic, and the resulting number is
one nobody funds. **The interior maximum is a consequence of a fixed budget, and the budget is
fixed for economic reasons that do not change.**

The delegation analysis assumes each hop either preserves or attenuates uniformly. Real
attenuation is *typed*: a retrieval sub-agent should lose write authority entirely rather than
having all authority scaled by 0.72. Typed attenuation is strictly better — it removes whole
capabilities rather than shrinking all of them — and it requires a capability model rich enough
to name what is being removed, which is the same requirement that makes
{{ch:sec-tool-abuse}}'s outcome classes work. **The permission model and the approval model
want the same vocabulary**, and building one twice is the common failure.

Finally, on audit and {{cite:hou2025mcp}}'s runtime tool addition. If tools can join at
runtime, the principal chain must be reconstructable for a hop that did not exist when the
policy was written. That rules out any audit design keyed on an enumerated component list and
requires the chain to be self-describing — each hop appending its own identity and the
authority it exercised. It is a small protocol requirement with a large consequence, and it is
the kind of thing that is nearly free to specify early and very expensive to retrofit.

## 16. Connection to Previous Chapters

{{eq:approval-must-sit-at-the-outcome-not-the-call}} from {{ch:sec-tool-abuse}} is confirmed by
a second, independent argument: a per-call gate cannot express a composition *and* cannot be
read.

{{eq:agent-authority-exceeds-requester-authority}} from the same chapter is followed here along
the chain, where nothing removes the excess and each additional agent extends it.

{{eq:guardrail-precision-is-set-by-the-base-rate}} from {{ch:sec-jailbreaks}} appears as
habituation: a rare event produces a queue that is mostly fine, and the consequence is a
reviewer who has learned that.

{{eq:cause-distance-drives-triage-cost}} from {{ch:ops-agent-tracing}} is why the audit fields
that settle questions are the ones about the chain and the trigger, not the acting identity.

## 17. Exercises

1. Measure your approval queue's volume, rejection rate and time-per-item, and compute $K(v)$.
   Where is the maximum relative to where you are?

2. Plant three rejectable items in a week. How many were caught?

3. Trace a request through every hop and record what authority each held. Where does the
   maximum enter?

4. Audit your last incident against the six fields. What could you not answer?

5. Split one mixed-density queue into two by expected rejection rate, per
   {{sec:15-advanced-concepts}}, and measure the catch rate on each.

## 18. Interview Questions

1. Our approval queue has 20,000 items a day and empties every afternoon. Is it working?

2. Why might a narrower approval gate catch more bad things than a wider one?

3. A user with read-only access asks the agent to do something. What authority does the request
   execute with?

4. Why does adding a sub-agent not reduce authority?

5. Our audit log records who called the API. What can we not reconstruct?

6. Which permission model would you choose, and what does the choice depend on?

## 19. Research Questions

1. How does habituation behave in mixed-density queues, and does splitting by density improve
   aggregate catch?

2. What are realistic rejection rates and time-per-item in deployed agent approval systems?

3. Can typed attenuation be specified generically enough to apply across heterogeneous tool
   protocols?

4. How often does principal-chain propagation survive a full production request path, and where
   does it break?

## 20. Chapter Summary

An approval queue degrades for two reasons and a permission chain fails for a third.

**Volume**: 360 reviewer-minutes over 20,640 approvals is **0.7 seconds** an item, and
detection at that budget is **0.012** ({{eq:approval-quality-falls-with-volume}}).
**Density**: a **0.04%** rejection rate gives a care multiplier of **0.185** against **0.784**
at 8% — **4.2×** from density alone ({{eq:a-low-rejection-rate-trains-approval}}).

Both move with volume, so catch has an interior maximum: **1.10 bad actions a day at 200
approvals** against **0.02 at 20,640**. Across {{ch:sec-tool-abuse}}'s designs, the narrow gate
catches **55× more from 109× fewer approvals**. And a **2% risk-weighted** sample beats
reviewing everything.

The permission half inverts a familiar rule. Along a six-hop chain the user holds **0.08** and
the request executes with **1.00**, because the only narrowing is accidental and is undone at
the next hop ({{eq:delegation-preserves-authority-unless-attenuated}}). **A chain of trust is
as weak as its weakest link; a chain of authority is as strong as its most privileged
member** — and decomposing into more agents lengthens the chain without attenuating it.

What a policy can express is bounded by the token: **0.11** for RBAC on the service account,
**0.97** for a per-task capability. And audit records answer the wrong question — acting
identity is recorded **98%** of the time and settles **11%**, while the full principal chain is
recorded **19%** and settles **44%**. Top two together: **39%**; all six: **93%**
({{eq:audit-completeness-requires-the-principal-chain}}).

What ties the chapter together is that all three failures are defaults rather than decisions.
Nobody chose a 20,000-item queue; it accumulated. Nobody chose to execute at maximum authority;
forwarding does that when no policy says otherwise. Nobody chose to record only the acting
identity; the framework emitted it. In each case the fix is to make an explicit choice where
there is currently an implicit one, and in each case the explicit choice is cheaper than the
control it replaces.

Carry forward: **a queue nobody can read is not a control**, and **attenuate at every hop or
execute at the maximum**.

## 21. Further Reading

- {{cite:beurerkellner2025patterns}} — capability-scoped designs, and the utility cost of
  removing authority rather than reviewing its use.
- {{cite:hou2025mcp}} — runtime tool addition, and why an audit chain must be self-describing.
- {{cite:cemri2025mast}} — multi-agent coordination failures, including the accumulation that
  turns a gate into a log line.
- {{cite:breck2017}} — a readiness rubric whose governance section is the closest prior art for
  the audit-field accounting here.
