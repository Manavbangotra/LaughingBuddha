# Part XVII research notes — AI Agents

Research pass done 2026-08-29. Nine chapters, ch153–ch161.

## What this part has to avoid

Agents are the most product-marketed topic in the book, and the literature is
mostly framework documentation. A part organised around frameworks would be
obsolete before it was rendered.

**The organising commitment: an agent is a control loop with a stochastic
policy and no formal guarantee of termination.** Everything durable follows
from that sentence — the loop, the state it carries, why it fails to stop, why
its errors compound, and why giving it authority is a security decision rather
than a UX one.

Every chapter must leave the reader able to *compute* something: the expected
number of steps, the probability a $k$-step task completes, the blast radius of
a tool, the cost of a retry policy.

## The genuinely live questions

### 1. Is "agent" a meaningful category, or a spectrum?

The useful distinction is **who chooses the next step**. In a workflow the
control flow is written by a human and the model fills slots. In an agent the
model chooses the next action, including when to stop. That is a binary
distinction about where control lives, and it predicts the failure modes: a
workflow fails where its author did not anticipate a case, an agent fails by
looping, wandering, or stopping early.

{{ch:ag-what-is-an-agent}} must make this crisp and resist the spectrum framing,
because the spectrum framing hides the fact that **the two have different
correctness arguments**. A workflow can be tested by enumerating paths. An agent
cannot.

**Listing:** the same task solved three ways — fixed pipeline, router, agent loop
— on a task distribution with a long tail. Measure success, cost, and variance
against the tail fraction. The expected finding: the pipeline wins on the head
and the agent wins on the tail, and the crossover is a function of tail mass, so
"should this be an agent" is a measurement about your traffic rather than an
architecture preference.

### 2. What actually fails in tool calling?

{{cite:schick2023toolformer}} names four decisions: which API, when, what
arguments, how to use the result. That decomposition is the chapter, because the
four fail at very different rates and the aggregate "tool use accuracy" number
hides which.

{{ch:rsn-tool-assisted}} already established the arithmetic — $p_t$ against $p_e$,
and the boundary-crossing cost. {{ch:ag-tool-calling}} is the *design* half: what
makes a tool easy to call correctly.

**Listing:** decompose tool-call failure into the four stages and measure each
separately as the tool inventory grows. Expect selection error to grow with
inventory size (a discrimination problem over an expanding set) while argument
error stays flat — which would mean **tool count is the variable to control, not
tool quality**. Then measure the effect of overlapping tool descriptions.

### 3. Why do agent loops not terminate?

This is the chapter that has to carry real arithmetic. An agent loop is a Markov
chain over states with an absorbing "done" state, and the questions are: what is
the expected number of steps, what is the probability of absorption, and what
happens when the transition probabilities are slightly wrong.

Key asymmetry to establish: **the probability of completing a $k$-step task
compounds ({{eq:chain-accuracy-compounds}} again), but the cost of NOT
terminating does not compound — it is unbounded.** So budgets are not an
optimisation, they are the thing that makes the expected cost finite.

**Listing:** a loop with per-step success and a per-step chance of entering a
non-productive cycle. Measure completion rate and expected steps against horizon,
and show where a step budget converts an infinite expected cost into a finite
one at a small cost in completion rate.

### 4. Does interleaving reasoning and acting help, and why?

{{cite:yao2023react}} is the canonical answer and {{ch:rsn-tool-assisted}}
complicated it: interleaving costs a boundary crossing per step, and the single-
call design won from $k=5$ in {{ch:rsn-tool-assisted}}'s listing.

The reconciliation, and {{ch:ag-react}} must state it: **interleaving buys the
ability to condition on observations you could not predict.** Where the
environment is known in advance, emit a program. Where it is not, you must look
before deciding, and the boundary crossing is what you are buying.

**Listing:** the same task under two environments — one fully predictable, one
where each observation carries information — and measure plan-then-execute
against interleaved. Expect the crossover to be a function of *observation
informativeness*, which is measurable.

### 5. Does planning help?

Suspect this is the weakest-supported common practice in the part. A plan
generated before any observation is a prediction, and
{{cite:liu2024agentbench}} identifies long-horizon reasoning as a primary
obstacle — which means plans are exactly the thing models are worst at.

**Listing:** plan-and-execute against reactive, with a sweep over how much the
environment deviates from what the plan assumed. Expect planning to win when
deviation is low and lose when it is high, and — the more useful result — expect
*replanning frequency* to matter more than plan quality.

### 6. What is agent memory actually for?

Three things get called memory: the context window, a scratchpad the agent
writes to, and a retrieval store. They solve different problems and
{{ch:ag-memory}} should refuse the umbrella term.

{{cite:shinn2023reflexion}}'s episodic memory of past failures is the
interesting case, because it is the one with a measured result — and its result
depends on the feedback being external (test failures), which is
{{ch:rsn-self-consistency}}'s correlation condition arriving in the agent
setting.

**Listing:** measure what memory buys against what it costs. Expect a
non-monotone curve: accumulated memory helps up to a point and then degrades
performance by filling the context with stale or wrong entries. If so, **memory
needs eviction more than it needs capacity**, which is a testable and unfashionable
claim.

### 7. Can agents recover from their own errors?

{{ch:rsn-self-consistency}} answered the general form: intrinsic self-correction
converges to the model's mode. The agent setting has an important difference —
the environment supplies external feedback, which is exactly the condition under
which reflection works ({{cite:huang2024selfcorrect}}).

{{ch:ag-recovery}} must state the boundary precisely: **an agent can recover from
errors the environment reports and cannot recover from errors it must detect
itself.** That single sentence predicts which agent tasks work.

**Listing:** recovery with environment feedback, with noisy environment feedback,
and with self-assessment only. Reuse {{ch:rsn-self-consistency}}'s
correlated-critic machinery in an episodic setting.

### 8. When should an agent stop, and who decides?

Termination is three separate decisions that get conflated: the task is done, the
budget is exhausted, and this requires a human. The third is the one with a
design literature and no measurements.

**Listing:** the economics of human-in-the-loop. A confirmation gate has a cost
(latency, human attention) and a benefit (blocked bad actions), and the optimum
depends on the action's blast radius and the agent's precision. Expect the
finding that **confirming everything is worse than confirming nothing** on any
realistic attention budget, because habituation makes the human a rubber stamp —
which argues for selective gating keyed to reversibility.

### 9. What is the actual security model?

{{cite:greshake2023indirect}} is the paper. The structural fact: an LLM has no
channel separation between instructions and data, so **any content the agent
reads can issue commands**, and the abstract's note that mitigations are lacking
has aged well.

{{ch:ag-agent-security}} must be concrete about the design response rather than
gesturing at "be careful": least privilege per tool, reversibility as the gating
criterion, and the observation that **the blast radius of an agent is the union
of its tools' blast radii, not the maximum**.

**Listing:** measure the difference between a permission model keyed to tools and
one keyed to reversibility, on a task distribution with injected adversarial
content. The interesting quantity is not detection rate — detection is
unreliable — but the fraction of successful injections that produce an
irreversible effect.

## Chapter plan

### 153 — What an AI Agent Is
Per live question 1. The control-location distinction, the three architectures,
and the measurement that chooses between them.
**Listing:** pipeline vs router vs agent across a tail-mass sweep.

### 154 — Tool Calling and Tool Design
Per live question 2. The four decisions, and what makes a tool callable.
**Listing:** four-stage failure decomposition against inventory size, plus the
effect of description overlap.

### 155 — The Agent Loop
Per live question 3. The loop as an absorbing Markov chain, with arithmetic.
**Listing:** completion and expected steps against horizon; budgets making
expected cost finite.

### 156 — ReAct and Interleaved Reasoning and Acting
Per live question 4. Reconciles {{cite:yao2023react}} with
{{ch:rsn-tool-assisted}}'s boundary-crossing cost.
**Listing:** plan-then-execute vs interleaved against observation
informativeness.

### 157 — Planning and Plan-and-Execute
Per live question 5. Expect a negative-leaning chapter, honestly reported.
**Listing:** deviation sweep; replanning frequency against plan quality.

### 158 — Agent Memory
Per live question 6. Three distinct mechanisms under one word.
**Listing:** the non-monotone memory curve, and eviction.

### 159 — Reflection, Replanning, and Error Recovery
Per live question 7. The external-feedback boundary.
**Listing:** three feedback regimes in an episodic setting.

### 160 — Termination, Budgets, and Human-in-the-Loop
Per live question 8. Three decisions, and the economics of gating.
**Listing:** confirmation-gate optimum against blast radius and precision.

### 161 — Agent Security and Excessive Agency
Per live question 9. {{cite:greshake2023indirect}}'s structural fact and the
design response.
**Listing:** tool-keyed against reversibility-keyed permissions under injection.

## Cross-part bookkeeping

- {{part:16}} owns the reasoning mechanisms. This part uses
  {{eq:chain-accuracy-compounds}}, {{eq:tool-error-reallocation}} and
  {{eq:correlated-critic}} rather than rederiving them.
- {{part:18}} owns multi-agent systems. This part stops at one agent.
- {{part:15}} owns serving economics; agent loops are many sequential decode
  calls and that is where their cost lives.
- {{part:12}} owns retrieval; agent memory that is retrieval should defer to it.

## Citations verified this pass

`schick2023toolformer`, `shinn2023reflexion`, `greshake2023indirect`,
`zhou2024webarena`, `liu2024agentbench` — all verified against arXiv abstract
pages on 2026-08-29. `yao2023react`, `huang2024selfcorrect`, `gao2023pal`,
`sprague2024tocot` already in the bibliography from earlier parts.
