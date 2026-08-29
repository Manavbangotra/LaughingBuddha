---
id: part-17-intro
status: final
---

## What this part is for

{{part:16}} made a model reason. This part is about letting it *act*, and it is the
part of the book where the distance between the marketing and the measurement is
greatest.

**The hazard here is that "agent" is a product category, and the literature is
mostly framework documentation.** A part organised around frameworks would be
obsolete before it rendered. Organised around one structural property — who chooses
the next action — it teaches something that survives the next release.

> **The rule adopted for this part: every recommendation is priced, and the
> baseline is always the trivial thing.** Blind retry, uniform budgets, no gate at
> all. Several chapters here found the sophisticated intervention losing to the
> trivial one, and those are the pages worth reading twice.

## The organising idea

**An agent is a control loop with a stochastic policy and no guarantee of
termination.** Everything durable follows from that sentence.

```text
   CHAPTER                   THE DECISION IT OWNS       WHAT DECIDES IT
   ───────────────────────   ────────────────────────   ─────────────────────────
   153 (the definition)      agent or workflow          your tail mass
   154 tool calling          which tool, what arguments tool DISTINCTNESS
   155 the loop              when to stop               the false-stop rate
   156 reason and act        plan or interleave         observation informativeness
   157 planning              plan or checkpoint         which loss dominates
   158 memory                context, pad, or store     which dependency you have
   159 recovery              retry, resume, or diagnose where it failed
   160 termination           who ends the run           reviewer attention
   161 security              what it may reach          the capability union
```

The through-line: **almost every question in this part turns out to be a
measurement about your environment rather than a choice about your architecture.**
Tail mass, informativeness, drift, dependency distance, feedback quality, reviewer
load, capability union. Nine chapters, nine numbers, and a team that has measured
none of them is choosing by taste.

**And a second through-line, which was not planned.** In chapter after chapter the
naive baseline beat the considered intervention:

| Chapter | The sophisticated thing | What beat it |
|---|---|---|
| {{ch:ag-loop}} | tuning action accuracy | fixing the stopping threshold |
| {{ch:ag-planning}} | a better plan | a checkpoint |
| {{ch:ag-recovery}} | self-assessment | retrying blindly |
| {{ch:ag-termination}} | difficulty-aware budgets | a shared pool with no model |
| {{ch:ag-security}} | an injection detector | removing the capability |

That is not a coincidence about agents. It is what happens when a component is
inserted between a decision and its default: **an unreliable signal that GATES is
worse than no signal at all**, because it forfeits the default. That sentence
explains four of the five rows.

## Ten things worth knowing before you start

**An agent is a workflow whose control flow the model writes.** That is binary, not
a spectrum, and it changes the correctness argument: {{ch:ag-what-is-an-agent}}
counts $531{,}441$ execution paths at horizon 12, of which 500 tests cover
$0.09\%$. A workflow's argument is coverage; an agent's has to be statistical —
ten paths out of $6{,}561$ covered $85.4\%$ of runs.

**And the choice between them is arithmetic.** The agent overtook a six-branch
router on success at about $20\%$ tail mass — and never overtook it on success per
model call. The deciding number is the cost of a failed task over the cost of a
model call.

**Tool count is nearly free; tool overlap is not.** Selection was $100.0\%$ at four
tools and $100.0\%$ at $128$, because distinct points in a high-dimensional space
stay far apart. The same $128$ tools in two families: $67.1\%$.

**The error message is the cheapest selector in the book.** Three retries against
an opaque error bought $0.9$ points; against one naming the field and listing valid
values, $16.1$. **A bad signature with good errors beat a good signature with bad
errors while making fewer calls.**

**A loop with slack is not a chain.** With a perfect stopping judgement, agents at
$75\%$ and $99\%$ per action both completed $100\%$ of tasks, differing only in
steps. Retries convert reliability into cost, so {{eq:chain-accuracy-compounds}}
stops governing — and success is decided by the stopping classifier instead, where
cutting false stops $5\% \to 1\%$ bought $+19.6$ points against $+2.4$ for taking
actions from $90\%$ to $99\%$.

**A loop is a fixed point, not a malfunction.** A naive agent spent $35\%$ of its
budget re-running actions that had already failed, because after a failure the
context is nearly unchanged. Refusing to re-issue a failed action beat improving the
model from $82\%$ to $96\%$, at fifteen lines of code.

**The plan-versus-interleave crossover does not depend on task length.** Both shapes
are per-step bases raised to $k$, so $k$ cancels — which removes "complexity" from
the decision entirely. What decides it is how often an observation tells you
something you could not have predicted, and replanning on surprise beat both pure
strategies at every level.

**Checkpoints are the largest lever in the part.** Cutting a 12-step task into six
verified segments took the same agent from $48.7\%$ to $97.5\%$, because the
governing exponent changes from task length to segment length. It beat a five-point
model improvement — and adding checkpoints without raising the budget took
completion to **zero**.

**Extending the context window made recall of recent facts worse** — $21.0\%$ to
$10.5\%$ — because dilution applies to everything inside. Three mechanisms are
called memory and they are not substitutes: the right one at a small context beat
the wrong one at a large context in every row.

**Confirming everything is close to confirming nothing.** Three reviewers gating
three thousand actions a day dropped to a $2.2\%$ catch rate: seventy-five human
hours to avoid $3\%$ of the harm, while producing an audit trail saying those
actions were reviewed. At a fixed budget, harm avoided per human hour ranged from
$0.07$ to $34.52$ depending only on the selection criterion.

## What this part deliberately does not cover

**Multi-agent systems.** {{part:18}}. This part stops at one agent, and the one
multi-agent pattern it does recommend — the reader/actor split in
{{ch:ag-security}} — arrives for a security reason rather than an orchestration one.

**Tool protocols and ecosystems.** {{part:19}}. {{ch:ag-tool-calling}} is about what
makes a tool callable, not about how tools are published or discovered.

**Training agents.** {{part:9}}'s alignment material and
{{ch:rsn-supervision}}'s. This part treats the model as given.

**Serving economics.** {{part:15}}'s, used here rather than rederived — an agent
loop is many sequential decode calls with a growing cache, which is the most
expensive request shape there is.

## How to read it

{{ch:ag-what-is-an-agent}} and {{ch:ag-loop}} are the foundation.
{{eq:chain-accuracy-compounds}} arrives from {{part:16}} and then
{{eq:loop-is-not-a-chain}} explains where it stops applying, which is one of the few
places in this book where an earlier part's arithmetic is explicitly bounded.

{{ch:ag-tool-calling}} is the reliability chapter and its results feed everything
after it — the error-message finding turns up again in {{ch:ag-loop}} as a
loop-breaking mechanism and in {{ch:ag-recovery}} as the diagnosis term.

{{ch:ag-react}}, {{ch:ag-planning}} and {{ch:ag-memory}} are three views of the same
question — what should the agent carry forward, and in what form — and they should be
read together.

{{ch:ag-recovery}}, {{ch:ag-termination}} and {{ch:ag-security}} are the control
chapters, and they share one structure: a decision, a signal that might inform it,
and the observation that a weak signal wired to a gate is worse than no signal.

> **One thing to notice on a second reading**: {{ch:ag-planning}} finds
> checkpoints worth more than plan quality, {{ch:ag-recovery}} finds localisation
> worth thirteen times diagnosis, and {{ch:ag-security}} finds capability
> partitioning worth more than any detector. **All three are the same finding**:
> structure that bounds what can go wrong beats effort spent making it less likely.
> That is the most transferable idea in {{part:17}} and no chapter states it alone.
