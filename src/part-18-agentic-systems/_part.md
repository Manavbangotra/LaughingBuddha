---
id: part-18-intro
status: final
---

## What this part is for

{{part:17}} built one agent. This part asks what happens when you build several, and
it is the part of the book with the widest gap between what the architecture diagrams
promise and what the measurements deliver.

**The hazard here is that multi-agent systems are easy to draw and hard to price.**
A diagram with a planner, three workers and a critic looks like more capability than
a diagram with one box. Whether it *is* depends on numbers almost nobody collects:
error correlation, handoff cost, verifier detection rate, tool reversibility.

> **The rule adopted for this part: every architecture is compared at equal cost,
> against one agent.** Not against a bare loop — against {{ch:as-single-agent}}'s
> properly-built single agent, which completes $89.6\%$ of a task the bare loop
> completes $6.8\%$ of. Most published multi-agent comparisons use the wrong
> baseline, and the difference is the entire result.

## The organising idea

**A second agent buys exactly one thing: decorrelation.** Everything else it appears
to buy is either available more cheaply from one agent, or is a cost being mistaken
for a benefit.

```text
   CHAPTER                   THE DECISION IT OWNS       WHAT DECIDES IT
   ───────────────────────   ────────────────────────   ─────────────────────────
   162 single agent          the baseline               component interactions
   163 multi-agent           one agent or several       error correlation
   164 roles                 label roles or capability  what the role changes
   165 graph orchestration   graph or free loop         branch count and tail mass
   166 state machines        what survives a crash      the agent's own state
   167 long-running          how long a run may be      how fast the world moves
   168 specialization        which domains work         the domain's verifier
   169 failure modes         where the checks go        coverage, then freshness
```

The through-line: **the topology is not the variable.** Seven of these eight
chapters found the outcome decided by a property of the environment or the tooling —
correlation, tail mass, verifier detection, reversibility, staleness rate — rather
than by the shape of the agent graph. A team that has drawn its architecture and
measured none of those has made no decision at all.

**And a second through-line, which {{part:17}} also had and which is stronger
here.** In chapter after chapter, the elaborate structure lost:

| Chapter | The elaborate thing | What beat it |
|---|---|---|
| {{ch:as-multi-agent}} | a debate panel | one agent with diverse tools |
| {{ch:as-roles}} | planner / worker / critic | one agent at equal cost |
| {{ch:as-graph}} | a drawn control-flow graph | a free loop, at zero tail mass |
| {{ch:as-state-machines}} | frequent checkpointing | a deduplication key |
| {{ch:as-long-running}} | pausing a hundred times | pausing twelve times, placed well |
| {{ch:as-failures}} | nine voters | five voters that disagree |

That is one finding restated six times. **Structure that adds components adds
correlated components**, and correlated components do not aggregate. The structures
that *did* win — capability partitions, deduplication keys, spread critics — all
work by bounding what can go wrong rather than by adding judgement.

## Ten things worth knowing before you start

**A second agent's entire value is decorrelation.** An agent with identical errors
buys $-0.0$ points; one with independent errors buys $+18.8$. The residual failure of
a good single agent is $25\%$ capability, $72\%$ correlated, $4\%$ verification — and
no amount of architecture touches the capability term.

**Evaluate by ablation, not by addition.** Informative errors were worth $+23.4$
added to nothing and $+43.6$ removed from everything. The standard one-at-a-time
methodology understates every contingent component, which is most of them, and leads
teams to ship the bare loop.

**A role is a prompt, and a prompt does not decorrelate anything.** Three
role-structured agents scored $18.2\%$ against one agent's $35.1\%$ at equal cost.
Role separation earns its handoffs only when the roles carry different
*capabilities* — a reader that cannot act, an actor that cannot read.

**A graph lost to a free loop at zero tail mass**, which is the graph's best case.
Branch decisions are an exponent: three branches at $96\%$ is $88.5\%$ before any
step runs. A graph's real justification is that control flow becomes a reviewable
artefact, and it should be argued for on those terms rather than on reliability.

**An at-least-once execution guarantee is an at-least-once side effect guarantee.**
With no idempotent steps, $33.5\%$ of runs ended with a duplicated effect and nothing
malfunctioned. A deduplication key takes every configuration to $100\%$ — it removes
the term where checkpointing only shrinks it.

**The durable field that matters most is the one no engine stores.** Omitting the
tried set cost $38.6$ points; omitting the position cost $3.3$. Engines persist what
*engines* need; the tried set, the derived values and the verbatim goal are what the
*agent* needs to be the same agent it was.

**Long runs fail silently.** With recovery in place, budget exhaustion was $0.0\%$
at every horizon — and silent drift was $94.3\%$ of failures at horizon 10. A drifted
run completes with every step green. Re-validating assumptions took a 300-step run
from $0.7\%$ to $76.8\%$ for $50\%$ more steps, the best trade measured in this part.

**Place oversight by consequence, not by frequency.** Twelve pauses before
irreversible steps matched a hundred placed uniformly, at an eighth of the delay —
because habituation destroys most of frequency's value. An idealised reviewer would
take every-step pausing to $97.9\%$; the real one gets $36.6\%$.

**A domain's ceiling is its verifier, not its difficulty.** Detection correlates
$0.96$ with task success and per-step difficulty $0.71$. Research has the *highest*
per-step success in the table and loses to coding by thirty-five points, because
`pytest` exists and source agreement does not.

**Nine agents that share a cause are worth about one.** A vote of nine turned $85\%$
into $99.4\%$ independent and $86.2\%$ correlated. Meanwhile the chance of *all*
agents failing together rose from $0.01\%$ to $9.53\%$ — the tail the $r^k$
calculation hides while it is being too pessimistic about your pipeline.

## What this part deliberately does not cover

**Tool protocols and discovery.** {{part:19}}. This part assumes agents already have
tools; how those tools are published, versioned and found across organisational
boundaries is MCP's subject.

**Training or fine-tuning for coordination.** The models are given here.
{{ch:as-specialized}} concedes that domain conventions are a genuine model property
and then sets them aside, because they do not explain the domain ordering.

**Human-team analogies.** Deliberately. The reason a human team benefits from roles
— people have genuinely different knowledge and genuinely uncorrelated judgement — is
exactly the property {{ch:as-roles}} finds absent when the roles are prompts over one
model. The analogy is the source of most of the bad architecture in this space.

**Serving economics.** {{part:15}}'s. Note only that multi-agent systems are the
worst request shape there is: many sequential decodes, each with a growing cache,
multiplied by the agent count.

## How to read it

{{ch:as-single-agent}} is not optional. It is the baseline every other chapter
measures against, and reading the rest without it produces the same error the
literature makes — comparing an elaborate system to a bad simple one.

{{ch:as-multi-agent}}, {{ch:as-roles}} and {{ch:as-failures}} are one argument in
three parts, and the third supplies the mechanism for the first two.
{{ch:as-multi-agent}} found decorrelation cheap; {{ch:as-failures}} explains why
($k_{\text{eff}}$ is linear in $1-\rho$ and flat in $k$). Read them together if you
read nothing else here.

{{ch:as-graph}} and {{ch:as-state-machines}} are the structural pair — what a graph
buys and gives up, and what has to survive a crash — and they share the observation
that a state machine is a graph whose edges are checkable, so the branch penalty
vanishes and the tail is surrendered completely.

{{ch:as-long-running}} and {{ch:as-specialized}} are the environment chapters, and
both find the same shape: something outside the agent — the staleness rate, the
verifier — setting the ceiling.

> **One thing to notice on a second reading**: {{ch:as-state-machines}} finds a
> deduplication key beating more checkpoints, {{ch:as-long-running}} finds placed
> gates beating frequent ones, {{ch:as-failures}} finds spread critics beating
> concentrated ones, and {{ch:as-specialized}} finds staging beating caution.
> **All four are the same finding**: bounding where a failure can reach beats
> reducing how often it occurs. {{part:17}} closed on the single-agent version of
> that sentence; this part is what it looks like across a system.
