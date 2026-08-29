---
id: ag-react
number: 156
part: XVII
tier: full
status: draft
requires: [loop-is-not-a-chain, four-decisions, boundary-crossing-cost]
provides: [observation-informativeness, replan-on-surprise,
           crossover-independent-of-length, thought-buys-composition,
           selective-thinking, interface-decides-architecture]
citations: [yao2023react, gao2023pal, sprague2024tocot, schick2023toolformer,
            liu2024agentbench, zhou2024webarena, shinn2023reflexion]
---

## 1. Learning Objectives

By the end of this chapter you will be able to reconcile the case for interleaving
with the case against it, and name the one environment property that decides
between them; show that the crossover does not depend on task length, and say what
that rules out; explain why replanning on surprise dominates both pure strategies
and what it requires that they do not; predict from composition depth whether a
thought before an action will pay for its tokens; and identify the tool-interface
property that changes which architecture is correct.

## 2. Why This Matters

This chapter resolves a genuine tension between two things this book has already
established, and the resolution is a measurement rather than a preference.

{{cite:yao2023react}} interleaves reasoning and acting — think, act, observe,
repeat — and it is the default shape of nearly every agent framework.
{{ch:rsn-tool-assisted}} measured that shape and found it losing to a single
up-front program at *every* chain length, because each boundary crossing is paid
twice: once to construct the call, once to read the result back.

Both results are correct, and {{sec:9-practical-example}} finds the variable that
separates them. It is **observation informativeness**: how often a step's correct
action depends on something you could not have known before taking the previous
one. Below about $10\%$ the plan wins; above it the interleaving does. That is a
property of your environment, and it is measurable on a task distribution you
already have.

Three further results come out of the sweep, and two of them were not what the
experiment was built to show.

**The crossover does not move with task length.** Both shapes are products of a
per-step base raised to $k$, so the comparison is between two bases and $k$
cancels. That rules out the most common form of the advice — "use ReAct for
complex tasks" — because complexity does not enter the comparison at all.

**Replanning on surprise beats both pure strategies at every informativeness level
swept**, because it pays the observation cost when a surprise occurs rather than
unconditionally or never. It is the only one of the three whose cost is
proportional to what it buys.

**The reasoning half is a separate decision from the acting half**, and its
benefit is governed by composition depth. A thought before an action that needs
one fact from the context buys $+0.3$ points for $3.4\times$ the output tokens; a
thought before an action that combines four facts buys $+42.3$ points.

Together these say that "should I use ReAct" is four questions, not one, and that
the answers point in different directions depending on two numbers you can
measure.

## 3. Prerequisites

You need {{ch:ag-loop}}'s framing of the loop and, in particular, its result that
a loop with slack is not a chain — the arithmetic in this chapter is about
per-step *bases*, and which base is larger is what decides everything.

From {{ch:rsn-tool-assisted}}, the boundary-crossing cost: an interleaved step
pays a translation and a parse, and those are the terms interleaving is spending.

From {{ch:rsn-cot}}, the serial-computation account of intermediate tokens, which
is what the second half of this chapter applies to an agent's decisions.

And {{cite:sprague2024tocot}}'s scope result — chain-of-thought helps where the
difficulty is compositional depth and almost nowhere else — because that transfers
directly, and it is the reason a thought before every action is the wrong default.

## 4. Intuitive Explanation

Consider two ways to run an errand.

The first is to plan it at the kitchen table. Post office, then the bank, then the
shop, then home. You commit to the sequence before leaving, and you execute it.

The second is to decide each leg when you get there. Go to the post office, see
what happened, decide where to go next.

Planning is obviously cheaper — you thought once instead of four times — and it is
obviously better when the world behaves as expected. If the post office is open,
the bank has no queue, and the shop has what you need, the plan executes
perfectly and the deliberation at every corner was wasted.

It is obviously worse when the world surprises you. If the post office is closed,
every remaining leg of the plan was built on an assumption that is now false. You
are executing against a state that does not exist.

So the choice depends on one thing: **how often does what you find out change what
you should do next?** That is not a property of the plan or of the planner. It is a
property of the environment, and it is the number this chapter measures.

Here is the part that surprised me. You might expect longer errands to favour
checking as you go — more legs, more chances to be surprised, more value in
looking. That intuition is wrong, and the reason is arithmetic rather than
psychology. Both approaches succeed only if every step works, so both are a
per-step reliability raised to the number of steps. Which one wins is a comparison
of the two per-step numbers, and the exponent applies equally to both. Length
amplifies the gap; it does not change its sign.

That matters because it kills the most common version of the advice. "Use an
interleaved agent for complex multi-step tasks" attaches the decision to
complexity, and complexity is not in the comparison.

The third thing is that neither pure strategy is what you should build. Plan the
errand — and when you find the post office closed, replan from there. You get the
plan's cheapness when the world cooperates and the interleaver's adaptability when
it does not, and you pay for looking only when there was something to see.
{{sec:9-practical-example}} finds this beating both alternatives at every level of
surprise. What it needs, and what the pure strategies do not, is the ability to
*notice* that the plan no longer applies — which is a judgement, with all of
{{ch:ag-loop}}'s problems attached.

The last idea is separate, and it is the "Re" in ReAct rather than the "Act".

Writing a thought before each action is justified by {{ch:rsn-cot}}: intermediate
tokens buy serial computation the forward pass does not have. But
{{cite:sprague2024tocot}} bounded where that helps, and the bound transfers. If
the action is "search for the account number the user gave", there is nothing to
compose and the thought buys nothing. If it is "transfer the amount from the first
message to the account found in step two, in the currency implied by step four",
that is a composition of three retrieved facts, and doing it across emitted tokens
is far more reliable than doing it in one pass.

So the thought pays in proportion to how much combining the step requires — which
means the right policy is neither "always think" nor "never think", but a
structural rule about which kind of step gets one.

## 5. Formal Explanation

Let a task require $k$ steps. Define **observation informativeness** $\iota$ as the
probability that a step's correct action depends on state not observable before
that step:

$$\iota = \Pr[\text{step's correct action depends on an unobserved fact}]$$ (eq:observation-informativeness)

For plan-then-execute, a step succeeds if the planner got it right *and* it was
not surprised, so with per-planned-step reliability $p_{\text{plan}}$:

$$S_{\text{plan}} = \big(p_{\text{plan}}(1 - \iota)\big)^{k}$$ (eq:plan-success)

For interleaving, the observation removes the surprise term and adds the boundary
cost of {{ch:rsn-tool-assisted}} — a translation $p_t$ and a parse $p_r$ per step,
on top of the action itself $p_a$:

$$S_{\text{inter}} = \big(p_a\, p_t\, p_r\big)^{k}$$ (eq:interleave-success)

Setting them equal gives the crossover, and the $k$ cancels:

$$\iota^{*} = 1 - \frac{p_a\, p_t\, p_r}{p_{\text{plan}}}$$ (eq:crossover-independent-of-length)

**The architecture choice is a comparison of two per-step reliabilities and
contains no $k$.** Task length amplifies whichever side is already winning and
never changes which side that is. That is the chapter's cleanest result and it
eliminates "complexity" from the decision.

Now the hybrid. Replanning executes the plan and rewrites it when a surprise
occurs, so it pays the observation cost only on surprised steps:

$$S_{\text{replan}} = p_{\text{plan}}^{\,k} \cdot p_r^{\,\iota k}, \qquad C_{\text{replan}} = 2 + \iota k$$ (eq:replan-on-surprise)

Compare the exponents. Interleaving pays $p_r$ (and $p_t$) $k$ times; replanning
pays $p_r$ only $\iota k$ times. Planning pays $(1-\iota)$ $k$ times and
replanning pays it never. **Replanning takes the smaller exponent on both terms**,
which is why it dominates rather than interpolates.

The cost is a requirement neither pure strategy has: a detector for "the plan no
longer applies". Write its sensitivity and false-positive rate as
$\alpha_d, \beta_d$; then {{eq:replan-on-surprise}} is the $\alpha_d = 1$,
$\beta_d = 0$ case, and the realistic version degrades toward interleaving as
$\beta_d$ rises (replanning on nothing) and toward planning as $\alpha_d$ falls
(missing the surprise). It is {{ch:ag-loop}}'s stopping classifier again, with the
same correlated-critic ceiling.

For the reasoning half, model a step's action choice as retrieving $d$ facts from
the context and combining them. Retrieval succeeds at $p_\ell$ per fact.
Combination happens either inside one forward pass, where {{ch:rsn-cot}}'s depth
bound applies and difficulty compounds, or across emitted tokens, where each
combination gets its own pass:

$$p_{\text{step}} = p_\ell^{\,d} \times \begin{cases} p_c^{\,d-1} & \text{one pass} \\ p_c'^{\,d-1} & \text{with a thought, } p_c' > p_c \end{cases}$$ (eq:thought-buys-composition)

The gain from thinking is $\big(p_c'/p_c\big)^{d-1}$, which is exactly $1$ at
$d = 1$ and grows geometrically in depth. **A thought's value is a function of
composition depth and of nothing else** — not of task importance, not of step
count, not of how hard the task feels.

Against a token cost $\tau$ per thought, the selective policy that thinks only on
steps with $d > d^{*}$ dominates both constant policies whenever deep steps are a
minority, and the saving is $(1 - \text{deep share}) \cdot \tau k$.

## 6. Mathematical Foundation

{{eq:crossover-independent-of-length}} rewards a second look, because two of its
terms are things you control.

$p_t$ and $p_r$ are the tool interface: how reliably the model can compose a call
and read a result. {{ch:ag-tool-calling}} showed both are dominated by design
choices — enumerated arguments, schemas enforced at the sampler, response formats
with the needed field at the top. Substituting into
{{eq:crossover-independent-of-length}}:

$$\frac{\partial \iota^{*}}{\partial (p_t p_r)} = -\frac{p_a}{p_{\text{plan}}} < 0$$ (eq:interface-decides-architecture)

**Improving the tool interface moves the crossover left**, which means it does not
merely make interleaving better — it makes interleaving *correct* on environments
where it previously was not. {{sec:9-practical-example}} measures the effect
directly: taking the per-step overhead from $90\%$ to $99.5\%$ takes interleaved
success from $14.7\%$ to $60.2\%$ at fixed informativeness, crossing the plan's
$56.4\%$ on the way.

That is a stronger claim than "good tools help". It says the same team, on the
same task, should choose different architectures depending on how rigorous their
tool layer is, and that investing in the tool layer is also an architectural
decision.

Two boundaries on the length result. {{eq:crossover-independent-of-length}}'s
cancellation assumes both shapes fail the same way — one bad step ruins the task.
Where a plan can be *partially* salvaged, planning's exponent is effectively
smaller and the crossover moves right. And where a surprise invalidates all
*remaining* steps rather than one, planning is worse than
{{eq:plan-success}} models. Real tasks sit between those, so the measured
crossover should be estimated rather than computed.

Finally, note what {{eq:thought-buys-composition}} does *not* contain: any term for
the number of steps. Thinking is a per-step decision with a per-step justification,
and aggregating it into "this agent uses chain-of-thought" discards the only
variable that matters.

## 7. Internal Mechanics

### 7.1 The three shapes

```mermaid {#fig:react-shapes caption="Three arrangements of the same work. The difference is where the observation enters, and how many times."}
flowchart TD
    subgraph plan [plan-then-execute]
        P1[plan all k steps] --> P2[execute 1..k] --> P3[done]
    end
    subgraph inter [interleaved]
        I1[observe] --> I2[think] --> I3[act] --> I1
    end
    subgraph re [replan on surprise]
        R1[plan] --> R2[execute step]
        R2 --> R3{surprised?}
        R3 -- no --> R2
        R3 -- yes --> R1
    end
```

The middle shape observes $k$ times; the top observes zero times; the bottom
observes $\iota k$ times. That count is the entire cost difference and most of the
accuracy difference.

### 7.2 What "surprise" means operationally

{{eq:replan-on-surprise}} requires detecting that the plan no longer applies, and
the detector is the hard part. Three implementations, in increasing order of
strength:

**Structural**: the step's precondition is checkable and failed — the file does not
exist, the account is closed, the API returned 404. Cheap, reliable, and covers
only the surprises that happen to be typed.

**Comparative**: the observation differs from what the plan expected, which
requires the plan to have recorded an expectation. This is a good reason to make
plans state expected outcomes rather than only actions, and it is under-used.

**Judged**: the model reads the observation and decides. Most general, weakest
guarantee, and subject to {{ch:rsn-self-consistency}}'s correlation — the model
that wrote the plan is the one deciding whether it still holds.

The second is the one most systems are missing and could add cheaply.

### 7.3 Why the thought is also a memory

{{eq:thought-buys-composition}} scores a thought purely on whether it makes the
current step more reliable. That undercounts, because the thought stays in the
context and is available to every later step.

A step that writes "the account is 4471, the branch is Lisbon, so the currency is
EUR" has done a composition once and left the result where the next step can read
it as a single fact — turning a depth-3 step into a depth-1 step later on. This is
{{ch:ag-memory}}'s subject and it is a legitimate reason to emit a thought whose
immediate accuracy benefit is zero.

### 7.4 The thought as observability

The other thing a trace buys is a human-readable record of what the agent believed
at each step, which is often the only debugging surface a deployed agent has.

{{ch:rsn-cot}}'s faithfulness result applies in full — the thought is not
necessarily why the action was taken — so it should be read as *what a plausible
justification looks like* rather than as an explanation. But for locating where a
run went wrong it is genuinely useful, and "this thought buys no accuracy" is not
the same as "this thought is worthless".

### 7.5 Serving shape

Interleaving is the worst-shaped request in {{part:15}}'s cost model: $k$ serial
round trips, each re-entering the model with a longer context. Planning is one
long generation plus one execution phase. Replanning is one long generation plus
$\iota k$ short ones.

This compounds the accuracy argument rather than opposing it: the shape that wins
on accuracy in a predictable environment also wins on latency and cost there, and
the shape that wins in an unpredictable one is expensive in exactly the
environment where it is necessary.

## 8. Implementation

Two listings. The first sweeps observation informativeness and finds the crossover
between the three shapes. The second separates the reasoning half from the acting
half and prices it against composition depth.

```python {tier=A name=observation-informativeness}
"""When is it worth looking before you act?

cite:yao2023react interleaves reasoning and acting: think, act, observe, repeat.
ch:rsn-tool-assisted measured the cost of that shape and found it losing to a
single up-front program at every chain length, because each boundary crossing is
paid twice -- once to construct the call and once to read the result back.

Both are right, and the variable that separates them is how much an observation
tells you that you could not have predicted (eq:observation-informativeness).

A plan written before any observation is a prediction about the environment. Where
the environment is predictable the prediction is accurate and the plan is free.
Where it is not, every step of the plan after the first surprise is executing
against a state that no longer exists.

This listing sweeps that one variable and finds the crossover.
"""
import numpy as np

rng = np.random.default_rng(1877)

N = 60000
K = 7                    # steps in the task
P_ACT = 0.94             # executing a correctly-chosen action
P_TRANS = 0.96           # composing a call for an interleaved step
P_PARSE = 0.97           # reading one observation back correctly
P_PLAN_STEP = 0.97       # getting one step of an up-front plan right


def run(info, k=K):
    """`info` is observation informativeness: the chance that a step's correct
    action depends on something only observable at that step.

    plan-then-execute: commits to k actions up front. A step whose action
    depended on an unobserved fact is wrong, and everything after a wrong step
    in a dependent plan is executing from a bad state.

    interleaved: observes before each action, so informativeness costs it
    nothing -- but it constructs a call and parses a result at every step.
    """
    # PLAN: each step is right if the plan got it right AND it did not depend on
    # something unobservable at planning time.
    surprises = rng.random((N, k)) < info
    plan_ok = (rng.random((N, k)) < P_PLAN_STEP) & ~surprises
    plan_done = plan_ok.all(1)
    plan_calls = np.full(N, 1.0 + 1.0)          # write the plan, execute it

    # INTERLEAVED: sees the state before choosing, so surprises are handled.
    inter_ok = ((rng.random((N, k)) < P_ACT) &
                (rng.random((N, k)) < P_TRANS) &
                (rng.random((N, k)) < P_PARSE)).all(1)
    inter_calls = np.full(N, float(k))

    # REPLAN ON SURPRISE: execute the plan, and rewrite it when surprised.
    # Costs one extra call per surprise; the rewritten steps are then informed.
    n_sur = surprises.sum(1)
    # Every step still has to be planned correctly; a surprise
    # additionally costs one observation to read, and only surprises do.
    parse_ok = (rng.random((N, k)) < P_PARSE) | ~surprises
    replan_ok = (rng.random((N, k)) < P_PLAN_STEP).all(1) & parse_ok.all(1)
    replan_calls = 2.0 + n_sur

    return {
        "plan": (float(plan_done.mean()), float(plan_calls.mean())),
        "interleaved": (float(inter_ok.mean()), float(inter_calls.mean())),
        "replan": (float(replan_ok.mean()), float(replan_calls.mean())),
    }


INFOS = [0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.60]

print(f"A {K}-step task. `informativeness` is the chance that a step's correct")
print("action depends on something only visible at that step. Plan-then-execute")
print(f"commits up front ({P_PLAN_STEP:.0%} per planned step); interleaving")
print(f"observes first but pays {P_TRANS:.0%} to compose each call and")
print(f"{P_PARSE:.0%} to read each result.")
print()
print(f"{'info':>7}{'plan-then-execute':>24}{'interleaved':>20}"
      f"{'replan on surprise':>24}")
print(f"{'':>7}{'success':>13}{'calls':>11}{'success':>11}{'calls':>9}"
      f"{'success':>14}{'calls':>10}")
print("-" * 75)

res = {}
for i in INFOS:
    r = run(i)
    res[i] = r
    print(f"{i:>7.0%}{r['plan'][0]:>13.1%}{r['plan'][1]:>11.1f}"
          f"{r['interleaved'][0]:>11.1%}{r['interleaved'][1]:>9.1f}"
          f"{r['replan'][0]:>14.1%}{r['replan'][1]:>10.1f}")

cross = [i for i in INFOS if res[i]["interleaved"][0] > res[i]["plan"][0]]

print()
print()
print("Success per model call -- what each shape returns for its cost.")
print()
print(f"{'info':>7}{'plan':>10}{'interleaved':>14}{'replan':>10}{'best':>14}")
print("-" * 55)
eff = {}
for i in INFOS:
    e = {k2: res[i][k2][0] / res[i][k2][1] for k2 in res[i]}
    eff[i] = e
    print(f"{i:>7.0%}{e['plan']:>10.3f}{e['interleaved']:>14.3f}"
          f"{e['replan']:>10.3f}{max(e, key=e.get):>14}")

print()
print()
print("Does the crossover move with task length? Sweep k at fixed")
print(f"informativeness of {0.05:.0%} and {0.20:.0%}.")
print()
print(f"{'steps k':>9}{'info 5%':>24}{'info 20%':>24}")
print(f"{'':>9}{'plan':>12}{'interleaved':>12}{'plan':>12}{'interleaved':>12}")
print("-" * 57)
len_tab = {}
for k in (2, 4, 7, 12, 20):
    a, b = run(0.05, k), run(0.20, k)
    len_tab[k] = (a, b)
    print(f"{k:>9}{a['plan'][0]:>12.1%}{a['interleaved'][0]:>12.1%}"
          f"{b['plan'][0]:>12.1%}{b['interleaved'][0]:>12.1%}")

print()
print()
print("What if the boundary crossing gets cheaper? Interleaving's cost is two")
print("multiplications per step; sweep them together.")
print()
print(f"{'per-step overhead':>19}{'interleaved success':>22}{'vs plan at 5%':>16}")
print("-" * 57)
P_T_SAVE, P_P_SAVE = P_TRANS, P_PARSE
ov = {}
plan5 = res[0.05]["plan"][0]
for q in (0.90, 0.94, 0.96, 0.98, 0.995):
    P_TRANS = P_PARSE = q
    v = run(0.05)["interleaved"][0]
    ov[q] = v
    print(f"{q:>19.1%}{v:>22.1%}{v - plan5:>+16.1%}")
P_TRANS, P_PARSE = P_T_SAVE, P_P_SAVE

print(f"""
The first table is the reconciliation, and the two ends of the informativeness
column are the two papers.

At {0:.0%} informativeness -- a fully predictable environment -- planning up front
scores {res[0.0]['plan'][0]:.1%} in {res[0.0]['plan'][1]:.0f} calls and
interleaving scores {res[0.0]['interleaved'][0]:.1%} in
{res[0.0]['interleaved'][1]:.0f}. Interleaving loses on both axes, which is
ch:rsn-tool-assisted's result: it is paying {K} boundary crossings for
information it could have predicted.

At {0.2:.0%} informativeness the same comparison is
{res[0.2]['plan'][0]:.1%} against {res[0.2]['interleaved'][0]:.1%}. The plan is
now a prediction about an environment that keeps surprising it, and each surprise
invalidates a step.

{'The crossover is at about ' + format(cross[0], '.0%') + ' informativeness.' if cross else 'Interleaving does not overtake over the range swept.'}

**So "should I interleave" is a question about the environment, not about the
architecture.** The number to estimate is how often a step's correct action
depends on something you could not have known before taking the previous one, and
that is measurable on a task distribution you already have.

The replan column is the result, and it is neither of the two shapes the
literature argues about. It writes a plan, executes it, and rewrites when
surprised -- so it pays for informativeness only when informativeness occurs.

It beats BOTH pure strategies at every informativeness level swept.
{res[0.0]['replan'][0]:.1%} at {0:.0%}, where it matches planning because there is
nothing to replan; {res[0.6]['replan'][0]:.1%} at {0.6:.0%}, where planning has
collapsed to {res[0.6]['plan'][0]:.1%} and interleaving sits at
{res[0.6]['interleaved'][0]:.1%}. Its call count rises from
{res[0.0]['replan'][1]:.1f} to {res[0.6]['replan'][1]:.1f} -- **a call per
surprise rather than a call per step**, which is the whole difference.

The second table says the same thing in cost terms: replanning is the best of the
three at all {len(INFOS)} informativeness levels swept.

That is a stronger result than the sweep was built to produce, and the reason is
worth extracting. Interleaving pays the observation cost unconditionally, on the
assumption that every step might be surprising. Planning pays it never, on the
assumption that none is. **Replanning pays it exactly when a surprise occurs**,
which is the only one of the three policies whose cost is proportional to the
thing it is buying.

The catch is in the word "when". Replanning requires DETECTING that the plan no
longer applies, which is a classifier -- ch:ag-loop's stopping decision wearing a
different hat, with the same correlated-critic ceiling. This listing gives it a
perfect detector. A real one is not, and the gap between these numbers and a
deployed system is almost entirely that detector, which is ch:ag-planning's
subject and its hardest problem.

The third table checks whether the conclusion depends on task length, and the
answer is cleaner than I expected: it does not.

At {0.05:.0%} informativeness the plan leads at every length -- {2} steps
({len_tab[2][0]['plan'][0]:.1%} against
{len_tab[2][0]['interleaved'][0]:.1%}) through {20}
({len_tab[20][0]['plan'][0]:.1%} against
{len_tab[20][0]['interleaved'][0]:.1%}). At {0.2:.0%} interleaving leads at every
length. The winner never changes.

The reason is that both shapes are products of a per-step base raised to k, so
the comparison is between the two bases and k cancels out. Planning's base is
{P_PLAN_STEP:.2f}(1 - info); interleaving's is
{P_ACT * P_TRANS * P_PARSE:.3f}. They cross where
{P_PLAN_STEP:.2f}(1 - info) = {P_ACT * P_TRANS * P_PARSE:.3f}, at
info = {1 - (P_ACT * P_TRANS * P_PARSE) / P_PLAN_STEP:.1%}, and that expression
contains no k.

**So the architecture choice is decided entirely by a comparison of two per-step
reliabilities, and task length only amplifies whichever one is already winning.**
That is a much simpler rule than "use ReAct for complex tasks": complexity does
not enter. What enters is how predictable the environment is and how lossy your
tool boundary is.

The last table asks what would change the answer, and it is the actionable one.
Interleaving's disadvantage is entirely the per-step overhead: composing a call
and parsing a result, {K} times. Take that overhead from {0.90:.0%} to
{0.995:.0%} and interleaved success goes {ov[0.90]:.1%} to {ov[0.995]:.1%},
against the plan's {plan5:.1%} at the same informativeness.

**Every point of per-step overhead is multiplied by the horizon**, which is
ch:ag-tool-calling's finding stated as an architectural argument: constrained
decoding, enumerated arguments and unambiguous response formats do not merely
improve tool calls, they change which agent architecture is correct. A team with
a rigorous tool interface should interleave; a team without one should plan and
replan, because it cannot afford {K} round trips through a lossy boundary.""")
```

The second listing holds the acting shape fixed and asks about the thought.

```python {tier=A name=thought-buys-composition}
"""Does the "reasoning" half of ReAct earn its tokens?

cite:yao2023react's contribution is two things bolted together, and they are
usually evaluated as one. The ACTING half is the interleaving the previous listing
measured. The REASONING half is emitting a thought before each action, and it has
its own justification, its own cost, and its own scope.

ch:rsn-cot established what a thought buys: serial computation the forward pass
does not have. cite:sprague2024tocot established where that helps: tasks whose
difficulty is the DEPTH of a composition, and almost nowhere else. This listing
applies both to the agent setting (eq:thought-buys-composition).

The variable is how many facts already in the context an action's choice has to
combine. Choosing "call search" needs one. Choosing "call transfer with the
account found in step 2, the amount from the user's first message, and the
currency implied by the branch in step 4" needs three, and it is a composition.
"""
import numpy as np

rng = np.random.default_rng(1933)

N = 60000
K = 6
P_LOOKUP = 0.985         # retrieving one fact from context correctly
P_COMPOSE_NOTHOUGHT = 0.90   # combining facts inside one forward pass
P_COMPOSE_THOUGHT = 0.985    # combining them across emitted tokens
THOUGHT_TOKENS = 60      # a thought costs this many output tokens
ACTION_TOKENS = 25


def step_ok(depth, thought):
    """A step succeeds if every fact is retrieved AND they are combined
    correctly. A thought turns one deep composition into a sequence of shallow
    ones (ch:rsn-cot), which is why its benefit scales with depth."""
    look = (rng.random((N, max(depth, 1))) < P_LOOKUP).all(1)
    if depth <= 1:
        comp = np.ones(N, dtype=bool)
    elif thought:
        # Each combination is done in its own emitted step.
        comp = (rng.random((N, depth - 1)) < P_COMPOSE_THOUGHT).all(1)
    else:
        # All combinations must happen in one pass; difficulty compounds.
        comp = rng.random(N) < P_COMPOSE_NOTHOUGHT ** (depth - 1)
    return look & comp


def task(depth, thought, k=K):
    ok = np.ones(N, dtype=bool)
    for _ in range(k):
        ok &= step_ok(depth, thought)
    tok = k * (ACTION_TOKENS + (THOUGHT_TOKENS if thought else 0))
    return float(ok.mean()), float(tok)


DEPTHS = [1, 2, 3, 4, 6]

print(f"A {K}-step task. `depth` is how many facts from the context a step's")
print(f"action has to combine. Retrieving one fact is {P_LOOKUP:.1%} reliable;")
print(f"combining two inside one forward pass is {P_COMPOSE_NOTHOUGHT:.0%},")
print(f"and combining two across emitted tokens is {P_COMPOSE_THOUGHT:.1%}.")
print()
print(f"{'depth':>7}{'no thought':>24}{'with thought':>24}{'gain':>9}")
print(f"{'':>7}{'success':>12}{'tokens':>12}{'success':>12}{'tokens':>12}"
      f"{'':>9}")
print("-" * 76)

tab = {}
for d in DEPTHS:
    a, ta = task(d, False)
    b, tb = task(d, True)
    tab[d] = (a, ta, b, tb)
    print(f"{d:>7}{a:>12.1%}{ta:>12.0f}{b:>12.1%}{tb:>12.0f}{b - a:>+9.1%}")

print()
print()
print("Success per thousand output tokens -- the cost side of the same table.")
print()
print(f"{'depth':>7}{'no thought':>14}{'with thought':>15}{'better':>14}")
print("-" * 50)
eff = {}
for d in DEPTHS:
    a, ta, b, tb = tab[d]
    e = (a / (ta / 1000), b / (tb / 1000))
    eff[d] = e
    print(f"{d:>7}{e[0]:>14.2f}{e[1]:>15.2f}"
          f"{('thought' if e[1] > e[0] else 'no thought'):>14}")

print()
print()
print("A mixed task: most steps are shallow, a few are deep. Sweep the share of")
print("deep steps, and compare always-think against think-only-when-deep.")
print()
print(f"{'deep share':>12}{'never think':>14}{'always think':>15}"
      f"{'think when deep':>18}")
print("-" * 59)
mix = {}
for share in (0.0, 0.10, 0.25, 0.50, 1.0):
    deep = rng.random((N, K)) < share
    never = np.ones(N, dtype=bool)
    always = np.ones(N, dtype=bool)
    sel = np.ones(N, dtype=bool)
    tok_n = tok_a = tok_s = 0.0
    for j in range(K):
        d_deep = 4
        sn = np.where(deep[:, j], step_ok(d_deep, False), step_ok(1, False))
        sa = np.where(deep[:, j], step_ok(d_deep, True), step_ok(1, True))
        ss = np.where(deep[:, j], step_ok(d_deep, True), step_ok(1, False))
        never &= sn
        always &= sa
        sel &= ss
    tok_n = K * ACTION_TOKENS
    tok_a = K * (ACTION_TOKENS + THOUGHT_TOKENS)
    tok_s = K * ACTION_TOKENS + share * K * THOUGHT_TOKENS
    mix[share] = (float(never.mean()), float(always.mean()), float(sel.mean()),
                  tok_n, tok_a, tok_s)
    print(f"{share:>12.0%}{mix[share][0]:>14.1%}{mix[share][1]:>15.1%}"
          f"{mix[share][2]:>18.1%}")

print()
print()
print("And what that costs, at the same three policies.")
print()
print(f"{'deep share':>12}{'never':>10}{'always':>10}{'selective':>12}"
      f"{'selective saves':>18}")
print("-" * 62)
for share in (0.0, 0.10, 0.25, 0.50, 1.0):
    m = mix[share]
    print(f"{share:>12.0%}{m[3]:>10.0f}{m[4]:>10.0f}{m[5]:>12.0f}"
          f"{(m[4] - m[5]) / m[4]:>18.0%}")

print(f"""
The first table is cite:sprague2024tocot's finding transplanted into an agent
loop, and the depth column is the whole result.

At depth {1} -- an action that needs one fact from the context -- thinking first
buys {tab[1][2] - tab[1][0]:+.1%} and costs
{tab[1][3] / tab[1][1]:.1f} times the output tokens. There is nothing to compose,
so there is nothing for the extra serial steps to do, and the tokens are pure
overhead.

At depth {6} it buys {tab[6][2] - tab[6][0]:+.1%}, taking the task from
{tab[6][0]:.1%} to {tab[6][2]:.1%}.

**The benefit of a thought is a function of composition depth and nothing else**,
which is exactly ch:rsn-cot's account: intermediate tokens buy serial steps, and
serial steps are worth something only when the computation needs them. An agent
step that selects a tool by matching a description needs no depth; one that
assembles arguments from four places in the history is a composition, and it is
where the thought pays.

The second table adds the token cost, and it flips the recommendation at the
shallow end. Per thousand output tokens, not thinking scores
{eff[1][0]:.2f} against thinking's {eff[1][1]:.2f} at depth {1} -- roughly
{eff[1][0] / eff[1][1]:.1f} times better -- and the ordering reverses by depth
{[d for d in DEPTHS if eff[d][1] > eff[d][0]][0] if [d for d in DEPTHS if eff[d][1] > eff[d][0]] else 'never'}.

So "always think before acting" is the wrong default and "never think" is also the
wrong default, which is the setup for the third table.

Real tasks are mixed: most steps are shallow and a few are not. At a
{0.25:.0%} share of deep steps, never thinking scores {mix[0.25][0]:.1%}, always
thinking scores {mix[0.25][1]:.1%}, and thinking only on the deep steps scores
{mix[0.25][2]:.1%} -- statistically the same as always thinking, at
{(mix[0.25][4] - mix[0.25][5]) / mix[0.25][4]:.0%} fewer output tokens.

**Selective thinking gets all of the benefit for a quarter of the cost**, and the
saving grows as deep steps get rarer -- which is the direction real distributions
run.

The catch, and it is the same catch as everywhere in this part: selecting requires
knowing which steps are deep, and the thing that would decide is the model, before
it has done the composition. In practice the decision is made structurally rather
than by judgement -- **a step that assembles tool arguments from history gets a
thought; a step that picks a tool from a description does not** -- and the depth
column says how much that structural rule is worth.

One boundary on all of it. This models a thought as buying reliable composition
and nothing else. It does not model the two other things a thought does in a real
agent: it becomes part of the context for later steps, which is
ch:ag-memory's subject and is sometimes the whole point of writing it; and it is
the artefact a human reads when the run goes wrong. **A thought that buys nothing
in accuracy can still be the cheapest observability you have**, and that is a
legitimate reason to emit one that this listing has no way to score.""")
```

## 9. Practical Example

The first listing runs a seven-step task at varying observation informativeness.
Planning commits up front at $97\%$ per planned step; interleaving observes first
but pays $96\%$ to compose each call and $97\%$ to read each result.

```
   info       plan-then-execute         interleaved      replan on surprise
             success      calls    success    calls       success     calls
---------------------------------------------------------------------------
     0%        80.5%        2.0      39.1%      7.0         80.7%       2.0
     5%        56.4%        2.0      39.3%      7.0         79.8%       2.3
    10%        38.7%        2.0      39.6%      7.0         79.1%       2.7
    20%        16.9%        2.0      39.1%      7.0         77.7%       3.4
    60%         0.1%        2.0      39.3%      7.0         71.2%       6.2
```

At $0\%$ informativeness planning scores $80.5\%$ in two calls and interleaving
$39.1\%$ in seven — {{ch:rsn-tool-assisted}}'s result, with interleaving paying
seven boundary crossings for information it could have predicted. At $20\%$ the
comparison is $16.9\%$ against $39.1\%$: the plan is now a prediction about an
environment that keeps surprising it.

The crossover is at about $10\%$ informativeness. **"Should I interleave" is a
question about the environment, not the architecture.**

The replan column is the result the sweep was not built to produce. It beats
*both* pure strategies at every level, matching planning at $0\%$ (nothing to
replan) and reaching $71.2\%$ at $60\%$ where planning has collapsed to $0.1\%$.
Its call count goes $2.0 \to 6.2$: **a call per surprise rather than a call per
step**, which {{eq:replan-on-surprise}} explains as taking the smaller exponent on
both terms.

Per model call it is best at all seven levels swept:

```
   info      plan   interleaved    replan          best
-------------------------------------------------------
     0%     0.403         0.056     0.403        replan
    10%     0.194         0.057     0.293        replan
    60%     0.001         0.056     0.115        replan
```

Does the crossover move with task length?

```
  steps k               info 5%                info 20%
                 plan interleaved       plan  interleaved
---------------------------------------------------------
        2       84.9%       76.6%      59.6%        76.4%
        7       56.5%       39.2%      17.0%        39.0%
       20       19.5%        7.0%       0.6%         7.0%
```

No. At $5\%$ planning leads at every length; at $20\%$ interleaving leads at every
length. The winner never changes, because both are per-step bases raised to $k$
and $k$ cancels ({{eq:crossover-independent-of-length}}). **Task length amplifies
whichever side is already winning**, which eliminates "complexity" from the
decision.

And what would change the answer:

```
  per-step overhead   interleaved success   vs plan at 5%
---------------------------------------------------------
              90.0%                 14.7%          -41.7%
              96.0%                 36.8%          -19.6%
              99.5%                 60.2%           +3.8%
```

Taking the per-step overhead from $90\%$ to $99.5\%$ takes interleaving from
losing by $41.7$ points to winning by $3.8$, at fixed informativeness.
**Improving the tool interface does not merely make interleaving better; it makes
it correct where it previously was not** ({{eq:interface-decides-architecture}}).

The second listing turns to the thought. `depth` is how many facts from the
context a step's action must combine.

```
  depth              no thought            with thought     gain
            success      tokens     success      tokens         
----------------------------------------------------------------------------
      1       91.2%         150       91.5%         510    +0.3%
      2       44.3%         150       76.6%         510   +32.3%
      4       10.4%         150       52.7%         510   +42.3%
      6        2.5%         150       37.0%         510   +34.6%
```

At depth 1 a thought buys $+0.3$ points for $3.4\times$ the output tokens. At
depth 4 it buys $+42.3$. **The benefit is a function of composition depth and
nothing else** ({{eq:thought-buys-composition}}) — which is
{{cite:sprague2024tocot}}'s scope result arriving in an agent loop.

Per thousand output tokens, the ordering reverses at depth 4:

```
  depth    no thought   with thought        better
--------------------------------------------------
      1          6.08           1.79    no thought
      3          1.43           1.24    no thought
      4          0.69           1.03       thought
      6          0.16           0.73       thought
```

So neither constant policy is right, which is what the mixed case is for:

```
  deep share   never think   always think   think when deep
-----------------------------------------------------------
          0%         91.3%          91.4%             91.5%
         10%         75.7%          86.6%             86.6%
         25%         57.5%          80.3%             80.0%
        100%         10.4%          53.2%             53.0%
```

Thinking only on deep steps matches always-thinking at every share — $80.0\%$
against $80.3\%$ at a quarter deep — while saving $75\%$ of the thought tokens at
that share. **Selective thinking gets all of the benefit for a quarter of the
cost**, and the saving grows as deep steps get rarer, which is the direction real
distributions run.

The selection has to be structural rather than judged, because the thing that
would decide is the model, before it has done the composition. In practice: a step
that assembles tool arguments from history gets a thought; a step that picks a tool
from a description does not.

## 10. Production Considerations

Measure your observation informativeness. Sample real traces, and for each step
ask whether the correct action could have been determined before the previous
step's result. That fraction and
{{eq:crossover-independent-of-length}} decide the architecture.

Build the hybrid, not either pure shape. Plan, execute, replan on surprise. It
dominated both at every level swept and its cost scales with surprises rather than
with steps.

Make plans record expected outcomes, not just actions. That converts surprise
detection from a judgement into a comparison, which is the cheapest available
improvement to the detector {{eq:replan-on-surprise}} depends on.

Invest in the tool interface before choosing an architecture. Enumerated
arguments, enforced schemas, and unambiguous responses move $p_t$ and $p_r$, and
{{eq:interface-decides-architecture}} says those change which architecture is
correct.

Emit thoughts selectively, by a structural rule on the kind of step. Argument
assembly from history gets one; tool selection from a description does not.

But do not remove thoughts purely on the accuracy argument. They are also memory
({{sec:7-internal-mechanics}}) and they are usually your only debugging surface —
neither of which this chapter's model scores.

Watch the serving shape. Interleaving is $k$ serial round trips with a growing
context, which is {{part:15}}'s most expensive request shape, and it is required
exactly where the environment is least predictable.

## 11. Common Mistakes

**Choosing interleaving because the task is complex.** Complexity does not appear
in {{eq:crossover-independent-of-length}}. Predictability does.

**Interleaving on a predictable environment.** $39.1\%$ against $80.5\%$ at $0\%$
informativeness, for $3.5\times$ the calls.

**Planning on an unpredictable one.** $0.1\%$ at $60\%$ informativeness — the plan
is a prediction and the prediction is wrong.

**Treating ReAct as one decision.** The acting shape and the thinking policy are
independent, with different governing variables.

**Thinking before every action.** At depth 1 it buys $+0.3$ points for $3.4\times$
the tokens.

**Never thinking.** At depth 4 it costs $42.3$ points.

**Replanning without a surprise detector.** The hybrid's dominance in
{{sec:9-practical-example}} assumes perfect detection; a poor detector degrades it
toward whichever pure strategy its errors resemble.

## 12. Failure Modes

*Executing a stale plan.* The surprise happened and was not detected, so every
remaining step runs against a state that no longer exists. This is the failure
{{eq:plan-success}} models and it is silent — the steps all "succeed".

*Replanning on noise.* A false surprise detection rewrites a working plan, costing
a call and possibly losing progress. The $\beta_d$ direction, and the reason the
detector should be structural where possible.

*Thought inflation.* Emitting long thoughts before trivial steps, which is
expensive and also crowds the context that later steps depend on.

*Interleaving with a lossy interface.* $k$ round trips through a boundary with
$90\%$ per-step overhead reaches $14.7\%$ — the architecture amplifies the
interface's weakness.

*Debugging loss from removing thoughts.* An accuracy-motivated removal of the
reasoning trace leaves a run that cannot be diagnosed, which is a cost that shows
up weeks later.

## 13. Alternatives

**One program, executed once.** {{cite:gao2023pal}} and
{{ch:rsn-tool-assisted}}: the extreme of the planning end, correct where the
environment is fully predictable and the steps are computational.

**Hierarchical plans.** Plan coarsely, expand each step when reached. This lowers
effective informativeness at the top level while keeping adaptivity at the bottom,
and it is {{ch:ag-planning}}'s subject.

**Speculative execution.** Execute the plan while checking preconditions in
parallel, and roll back on a surprise. Trades wasted work for latency, and needs
reversible actions.

**Trained action selection.** {{cite:schick2023toolformer}}: raise $p_a$ and $p_t$
by training rather than prompting, which moves the crossover as well as the level.

**Fixed workflow.** {{ch:ag-what-is-an-agent}}: if informativeness is near zero and
the shapes are enumerable, none of this is needed.

## 14. Evaluation

Report observation informativeness as a property of your task distribution. It is
the number that selects the architecture and almost nobody measures it.

Report surprise-detection sensitivity and false-positive rate separately, for the
same reason {{ch:ag-loop}} reports the two stopping errors separately: they cause
opposite failures.

Evaluate the architectures at matched call budgets, not matched step counts.
Interleaving spends $k$ calls to planning's two, and a comparison that ignores
that is measuring the wrong thing.

Measure composition depth per step type. It predicts where a thought pays, and it
is recoverable from traces by counting how many prior observations an action's
arguments reference.

And evaluate after any tool-interface change.
{{eq:interface-decides-architecture}} says the correct architecture moves when
$p_t p_r$ moves, so an interface improvement should trigger an architecture
review.

## 15. Advanced Concepts

**Learned surprise detection.** {{eq:replan-on-surprise}}'s detector is a binary
classifier over (expected outcome, observed outcome) pairs, which is a small
supervised problem with abundant labels from traces. Decorrelating it from the
planner is the same argument as {{ch:ag-loop}}'s learned stopping, and it is the
single highest-value component this chapter identifies.
{{maturity:EMERGING}}.

**Informativeness as a routing signal.** If $\iota$ varies per request rather than
per task family, the architecture could be chosen per request. Estimating $\iota$
before acting is the open part.

**Depth estimation for selective thinking.** The structural rule in
{{sec:9-practical-example}} is a proxy for composition depth. Estimating depth
directly — counting the distinct prior observations an action's arguments depend
on — is computable from a plan and would make the selection principled rather than
heuristic.

**Partial plan salvage.** {{eq:crossover-independent-of-length}}'s cancellation
assumes a surprise ruins the task. A representation in which a plan degrades
gracefully — a dependency graph rather than a sequence — changes the exponent and
moves the crossover, and it is the formal case for graph-shaped plans that
{{part:18}} takes up. {{maturity:RESEARCH FRONTIER}}.

## 16. Connection to Previous Chapters

{{ch:rsn-tool-assisted}}'s finding that interleaving loses to a single program was
correct at the informativeness its listing implied — zero — and this chapter
supplies the axis along which it stops being correct.

{{ch:ag-tool-calling}}'s interface results reappear as
{{eq:interface-decides-architecture}}: enumerated arguments and enforced schemas
are architectural decisions, not just reliability ones.

{{ch:ag-loop}}'s stopping classifier is structurally the same object as this
chapter's surprise detector, with the same two asymmetric errors and the same
correlated-critic ceiling.

{{ch:rsn-cot}}'s serial-computation account and
{{cite:sprague2024tocot}}'s scope bound are what
{{eq:thought-buys-composition}} applies per agent step.

Ahead: {{ch:ag-planning}} takes the plan seriously and asks whether it survives
contact with an environment; {{ch:ag-memory}} takes up the thought-as-memory
observation; and {{ch:ag-recovery}} develops the surprise detector into a general
recovery mechanism.

## 17. Exercises

1. Derive {{eq:crossover-independent-of-length}} and compute $\iota^{*}$ for the
   listing's constants. Check it against the measured crossover.

2. Give the surprise detector a false-positive rate and re-run. At what $\beta_d$
   does replanning stop beating pure planning at low informativeness?

3. Model partial plan salvage — a surprise invalidates only the steps that depend
   on it — and show how the crossover moves.

4. In the second listing, let a thought reduce the depth of *later* steps by
   recording its result. Measure how much that changes the selective policy's
   advantage.

5. Estimate composition depth from a trace you own by counting prior observations
   referenced by each action's arguments. What fraction of your steps are deep?

6. Construct a task where interleaving wins at short lengths and loses at long
   ones. What did you have to break in the model to do it?

## 18. Interview Questions

1. When is planning better than interleaving? Answer in terms of one measurable
   property.

2. Does the plan-versus-interleave crossover move with task length? Why?

3. Why does replanning on surprise beat both pure strategies, and what does it
   need that they do not?

4. Your team improves the tool schemas. Should that change your agent
   architecture?

5. When does emitting a thought before an action buy nothing?

6. Give two reasons to emit a thought that have nothing to do with accuracy.

## 19. Research Questions

1. Can observation informativeness be estimated from traces without ground truth
   about what was knowable in advance?

2. What is the best learnable surprise detector, and how far can it be
   decorrelated from the planner that produced the expectation?

3. Does composition depth predict thought value on real tasks as cleanly as
   {{eq:thought-buys-composition}} implies, and is it measurable before acting?

4. Under what plan representations does a surprise degrade the plan gracefully
   rather than invalidating it, and how much does that move
   {{eq:crossover-independent-of-length}}?

5. Is per-request architecture selection worth its routing cost, given that
   informativeness varies within a task family?

## 20. Chapter Summary

The case for interleaving and the case against it are both correct, and the
variable that separates them is **observation informativeness**
({{eq:observation-informativeness}}): how often a step's correct action depends on
something unknowable beforehand. Below about $10\%$ planning wins; above it
interleaving does. {{sec:9-practical-example}} measures $80.5\%$ against $39.1\%$
at $\iota = 0$ and $16.9\%$ against $39.1\%$ at $\iota = 0.2$.

**The crossover does not move with task length.** Both shapes are per-step bases
raised to $k$, so $k$ cancels ({{eq:crossover-independent-of-length}}) and length
only amplifies whichever side is already ahead. That removes "complexity" from the
decision entirely.

**Replanning on surprise beat both pure strategies at every level swept** — $80.7\%$
at $\iota=0$, $71.2\%$ at $\iota = 0.6$ where planning reached $0.1\%$ — and best
per model call at all seven. It takes the smaller exponent on both terms
({{eq:replan-on-surprise}}), paying a call per surprise rather than per step. What
it requires, and the pure strategies do not, is a surprise detector, which is
{{ch:ag-loop}}'s classifier problem again.

The tool interface is an architectural decision. Taking per-step overhead from
$90\%$ to $99.5\%$ moved interleaving from losing by $41.7$ points to winning by
$3.8$ at fixed informativeness ({{eq:interface-decides-architecture}}).

And the reasoning half is a separate question with a separate answer. A thought's
value is $\left(p_c'/p_c\right)^{d-1}$ in composition depth
({{eq:thought-buys-composition}}) — $+0.3$ points at depth 1 for $3.4\times$ the
tokens, $+42.3$ at depth 4. Thinking only on deep steps matched always-thinking at
every mix while saving three-quarters of the thought tokens. **Neither constant
policy is right**, and the selection has to be structural because the thing that
would judge it is the model, before it has done the composition.

## 21. Further Reading

{{cite:yao2023react}} is the paper, and it is worth reading with this chapter's
decomposition in hand: its two contributions have different justifications and
different scopes, and the paper does not separate them.

{{cite:gao2023pal}} is the opposite end, and {{ch:rsn-tool-assisted}} is where
this book measured it. The two together bracket the axis.

{{cite:sprague2024tocot}} bounds where the reasoning half helps, and its
conclusion transfers to agent steps unchanged.

{{cite:liu2024agentbench}} and {{cite:zhou2024webarena}} for what these
architectures achieve on environments with real informativeness, which is the
calibration for how much any of this buys.
