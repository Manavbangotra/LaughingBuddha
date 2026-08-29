# Part XVIII research notes — Agentic AI Systems

Research pass done 2026-08-29. Eight chapters, ch162–ch169.

## What this part has to avoid

Multi-agent is where the framework marketing is thickest and the evidence
thinnest. {{cite:cemri2025mast}} opens by stating that **multi-agent performance
gains on popular benchmarks are often minimal**, which contradicts the premise
most multi-agent tooling is sold on, and then supplies 1600+ annotated traces
across seven frameworks and a validated 14-mode taxonomy.

**The organising commitment: a multi-agent system is a distributed system whose
nodes are unreliable, whose messages are natural language, and whose failures are
mostly coordination failures rather than capability failures.** That framing makes
the whole of distributed systems available and it is what makes the part durable.

Every chapter must beat a **single-agent baseline at equal cost**. That is the
comparison {{cite:cemri2025mast}} says is usually missing, and this part will run
it in every chapter.

## The genuinely live questions

### 1. When does a second agent help at all?

Two mechanisms are conflated. **Ensembling** — independent attempts reconciled,
which is {{cite:du2023debate}}'s debate and is {{ch:rsn-self-consistency}}'s
voting with extra steps. And **division of labour** — different agents doing
different sub-tasks, which is a decomposition argument and is
{{ch:ag-planning}}'s checkpoints under another name.

Only the first has a theoretical justification that does not reduce to something
cheaper. {{ch:ag-planning}} already showed decomposition is the big lever, and it
does not require separate agents — a single agent with checkpoints gets it.

**Listing (ch162 or ch163):** single agent with checkpoints vs two agents with a
handoff, at equal call budget, sweeping how independent the two agents' errors
are. Expect: the multi-agent version wins only through error decorrelation, and
loses the decomposition argument entirely because the single agent can decompose
too.

### 2. What does a handoff cost?

This is the part's central arithmetic and nobody states it. Passing work between
agents is a serialisation: the sending agent must summarise its state into a
message, and the receiving agent must reconstruct enough of it to continue. Both
are lossy.

That is exactly {{ch:rsn-cot}}'s token bottleneck and
{{ch:ag-memory}}'s scratchpad/context distinction, at the level of a whole agent.

**Listing:** measure task success against the number of handoffs, holding total
work constant. Expect a per-handoff multiplicative penalty, so success is
$p_{\text{handoff}}^{h}$ — and therefore the number of agents should be minimised
for the same reason the number of tool boundary crossings should be
({{ch:rsn-tool-assisted}}).

### 3. Do roles help, or are they prompt decoration?

Supervisor/worker/critic is the standard taxonomy. The critic role is
{{ch:rsn-self-consistency}}'s correlated critic unless the critic is a genuinely
different system — same finding, new costume. The supervisor role is
{{ch:ag-planning}}'s planner and inherits its weakness.

**Listing:** measure a role-labelled multi-agent system against a single agent
running the same prompts sequentially. Expect roles to buy nothing unless they
carry different *capabilities* or different *lineage* — which is
{{ch:ag-security}}'s capability partition arriving as an architecture.

### 4. What does graph orchestration buy over a loop?

A graph makes the control flow explicit and inspectable, which is
{{ch:ag-what-is-an-agent}}'s testability argument: a graph has enumerable paths
and a loop does not. That is a real benefit and it is a *correctness-argument*
benefit rather than a performance one.

**Listing:** path counts and mass coverage for a graph versus a free-running loop
at the same capability, reusing {{ch:ag-what-is-an-agent}}'s machinery. The
honest finding is likely that graphs recover testability at the cost of the tail
coverage that motivated the agent.

### 5. What does durable execution actually solve?

Retries after a crash. This is {{ch:ag-planning}}'s checkpoint with a persistence
requirement, and the interesting content is what has to be in the state for a
resume to be correct — and the observation that **an agent step is usually not
idempotent**, so replay is not free.

**Listing:** measure correctness of resume-after-crash against what fraction of
steps are idempotent, and show where at-least-once delivery produces duplicate
side effects.

### 6. What breaks in long-running autonomous systems?

Drift over time, accumulating memory staleness ({{ch:ag-memory}}), and the
absence of a person noticing. The distinctive failure is that **nothing is
wrong at any single step**.

**Listing:** a long-horizon run where each step is individually fine and the
aggregate drifts, plus the detector that would catch it.

### 7. Do specialised agents beat a generalist?

The claim is that a research agent, a coding agent and a data agent each do
better than one agent doing all three. That is a routing argument
({{ch:ag-what-is-an-agent}}'s router) plus a prompt-specialisation argument, and
the second is testable.

**Listing:** specialist vs generalist at equal cost, sweeping how distinct the
sub-tasks are. Expect specialisation to matter only when the sub-tasks need
genuinely different tools — which is again a capability argument.

### 8. What are the actual failure modes?

{{cite:cemri2025mast}}'s three categories: system design, inter-agent
misalignment, task verification. The third is {{part:16}}'s verification problem
and the second is new to this part.

**Listing:** reproduce the shape of the taxonomy — measure a multi-agent system
where each agent is individually correct and the system fails through
misalignment, and show that no per-agent metric detects it.

## Chapter plan

### 162 — Single-Agent Architectures
The baseline every later chapter must beat. Consolidates {{part:17}} into one
architecture and measures it honestly.
**Listing:** the reference single-agent design, and its cost/success frontier.

### 163 — Multi-Agent Architectures and Communication
Per live questions 1 and 2. The handoff cost is the chapter.
**Listing:** handoff penalty against handoff count; ensembling vs division of
labour at equal budget.

### 164 — Supervisor, Worker, Planner, and Critic Roles
Per live question 3. Expect a negative-leaning chapter.
**Listing:** roles against the same prompts run sequentially by one agent.

### 165 — Graph-Based Orchestration
Per live question 4. Testability recovered, tail coverage surrendered.
**Listing:** path enumeration and mass coverage, graph vs loop.

### 166 — State Machines, Events, and Durable Execution
Per live question 5. Idempotence is the chapter.
**Listing:** resume-after-crash correctness against idempotent-step fraction.

### 167 — Long-Running and Autonomous Workflows
Per live question 6. Drift with no single bad step.
**Listing:** aggregate drift detection.

### 168 — Specialized Agents
Per live question 7. Routing plus capability, not personality.
**Listing:** specialist vs generalist against sub-task distinctness.

### 169 — Multi-Agent Failure Modes
Per live question 8. {{cite:cemri2025mast}}'s taxonomy, reproduced in shape.
**Listing:** per-agent-correct system-wrong; the metric that detects it.

## Cross-part bookkeeping

- {{part:17}} owns the single agent. Every chapter here compares against it.
- {{part:16}} owns verification; {{cite:cemri2025mast}}'s third failure category
  is that problem and should defer.
- {{part:19}} owns tool protocols; agent-to-agent protocols belong here only as
  far as the handoff cost.
- {{part:23}} owns serving; the cost of n agents is n times the decode.

## Citations verified this pass

`cemri2025mast`, `du2023debate` — verified against arXiv abstract pages on
2026-08-29. `yao2023react`, `shinn2023reflexion`, `liu2024agentbench`,
`zhou2024webarena`, `greshake2023indirect`, `huang2024selfcorrect` already in the
bibliography.
