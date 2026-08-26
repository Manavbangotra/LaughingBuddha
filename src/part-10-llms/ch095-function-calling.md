---
id: llm-function-calling
number: 95
part: X
tier: full
status: draft
requires: [llm-structured-output, llm-prompt-lifecycle, llm-prompting,
           fm-instruction-tuning, nlp-extraction, llm-decoding]
provides: [function-calling, tool-schema, tool-dispatch-loop, tool-result-injection,
           tool-selection, argument-hallucination, parallel-tool-calls,
           tool-error-handling, action-not-taken]
citations: [schick2023, willard2023, wei2022cot, brown2020, ouyang2022,
            ji2023survey, min2022, liu2023lost]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Explain what function calling is mechanically — and why the model never
   executes anything.
2. Describe the dispatch loop and identify which parts are the model's
   responsibility and which are yours.
3. Distinguish the four failure modes of tool use and say which are fixable by
   constrained decoding.
4. Explain why tool results must be injected as context rather than as anything
   privileged.
5. Design tool schemas that a model can select between reliably.
6. Handle tool errors so the model can recover rather than repeat.
7. Decide how many tools is too many, from measurement.

## 2. Why This Matters

**This is the mechanism underneath every agent in {{part:17}}**, and it is
simpler than it appears: structured output plus a loop. Understanding it here,
where the whole thing fits on a page, makes the agent chapters comprehensible
rather than mystical.

**The model does not call anything.** It emits text that your code parses and
acts on. Every security property, every failure mode, and every design decision
follows from that one fact — and it is the fact most often obscured by the
terminology.

**The failure surface is different from ordinary generation.** A malformed
sentence is a bad answer; a malformed tool call is *an action not taken*, or
worse, the wrong action taken. {{ch:llm-structured-output}} eliminates the
malformed-argument class entirely, and the interesting failures are the ones it
cannot touch: the right format, the wrong tool.

**And the loop has properties nobody designed.** Each tool result is appended to
the context, so a multi-step interaction accumulates tokens quadratically in
cost and runs into {{cite:liu2023lost}}'s position effects. Those are
consequences of the architecture rather than choices, and knowing them prevents a
class of surprise.

## 3. Prerequisites

{{ch:llm-structured-output}} for constrained generation — a tool call is a
schema, and everything in that chapter applies. {{ch:llm-prompt-lifecycle}} for
the request loop this wraps. {{ch:llm-prompting}} for how tools are described.
{{ch:fm-instruction-tuning}} for the template, which is where tool definitions
and results are placed. {{ch:nlp-extraction}} for structured extraction, which
argument-filling is a form of. {{ch:llm-decoding}} for why $T=0$ is the right
setting here.

## 4. Intuitive Explanation

You want the model to check the weather. It cannot: it has no network, no
clock, and no ability to run code. What it *can* do is emit text.

**So you agree on a convention.** You describe the available functions in the
prompt — names, parameters, types — and the model, when it judges a function
useful, emits a structured object naming one and supplying arguments. Your code
parses that, calls the actual function, and puts the result back into the
conversation. The model continues, now with the answer in its context.

> NOTE: The model never executes anything. It emits a *request* to execute, and
> your code decides whether to honour it. This is the single most important
> sentence in the chapter, and it is why "the model called a function" is a
> harmful shorthand — it obscures that every call is your code's decision, which
> is the entire basis of {{part:26}}'s permission model.

**Why the model is any good at this** is {{ch:fm-instruction-tuning}}: models
are trained on examples of tool use, so emitting a well-formed call in the
expected format is a learned behaviour like any other.
{{cite:schick2023}} showed the model can even learn *when* calls are useful from
self-supervised data.

**The loop.** Send the conversation with tool definitions. If the response is a
tool call, execute it, append the result, and send again. Repeat until the model
answers without calling anything. **The loop is your code**, and its termination
condition, error handling, and iteration limit are your design decisions.

**Four ways it fails**, and they are genuinely different:

1. **Malformed arguments** — invalid JSON, missing fields, wrong types. Fully
   solved by constrained decoding.
2. **Hallucinated tool** — a call to a function that does not exist. Also solved
   by constraining the name to the available set.
3. **Wrong tool** — a valid call to the wrong function. Not solvable by
   constraints, because it is semantic.
4. **Wrong arguments** — the right tool with values that are plausible and
   incorrect. Also semantic, and the hardest.

**The first two vanish with a grammar. The last two are the real problem**, and
they are the same structural-versus-semantic division
{{tbl:constraint-scope}} drew.

**The mental model:** function calling is structured extraction
({{ch:nlp-extraction}}) where the extracted structure is an action, wrapped in a
loop your code controls. Where it breaks down: the model cannot observe whether
its call succeeded unless you tell it, so error handling is a *conversational*
problem, not an exception-handling one.

## 5. Formal Explanation

### 5.1 The tool schema

A tool is a name, a description, and a parameter schema:

$$
\tau = \big(\text{name},\ \text{description},\ \mathcal{S}\big)
$$ (eq:tool-definition)

The model's task, given a conversation $c$ and tool set $T = \{\tau_1,\dots,
\tau_k\}$, is to emit either a final answer or a call:

$$
\text{call} = \big(\text{name} \in \{\tau_i.\text{name}\},\
 \text{args} \models \mathcal{S}_{\text{name}}\big)
$$ (eq:tool-call)

**Both components are structural constraints** — the name comes from a finite
set, the arguments from a schema — so both are enforceable by
{{ch:llm-structured-output}}'s machinery.

### 5.2 The dispatch loop

$$
\begin{aligned}
&c_0 = \text{system} \cdot \text{tools} \cdot \text{user}\\
&\textbf{for } i = 1 \dots I_{\max}:\\
&\quad r_i \sim P(\cdot \given c_{i-1})\\
&\quad \textbf{if } r_i \text{ is a final answer: } \textbf{return } r_i\\
&\quad o_i = \text{execute}(r_i)\\
&\quad c_i = c_{i-1}\cdot r_i \cdot o_i
\end{aligned}
$$ (eq:dispatch-loop)

**Three things are your code's responsibility and not the model's**: the
iteration limit $I_{\max}$, the `execute` function including its permission
checks, and the decision to append $o_i$ rather than something else.

### 5.3 Context growth is quadratic in cost

Each iteration appends the call and its result. If each round adds $m$ tokens,
after $i$ rounds the context is $|c_0| + im$, and the *cumulative* prefill cost
across the loop is

$$
\sum_{i=1}^{I} \big(|c_0| + im\big)
 = I|c_0| + m\frac{I(I+1)}{2}
 = O(I^2 m)
$$ (eq:tool-loop-cost)

$\square$

**A ten-round tool interaction costs far more than ten single-turn requests.**
Prefix caching ({{ch:llm-inference}}) helps enormously here because each round's
context is a strict extension of the previous one — the shared prefix is
everything but the last exchange, so the marginal cost per round approaches $m$
rather than $|c_0| + im$.

### 5.4 Tool results are ordinary context

The result $o_i$ is inserted into the conversation as text. It has **no special
status**: the model weighs it against everything else in the context, including
the system prompt and the user's message.

> WARNING: This is the mechanism of indirect prompt injection. If a tool returns
> attacker-controlled content — a fetched web page, a retrieved document, an
> email body — that content enters the context on equal terms with your
> instructions. The model has no way to distinguish "data returned by a tool"
> from "instructions from the operator", because at the token level there is no
> difference. {{part:26}} treats this properly; the point here is that it
> follows directly from {{eq:dispatch-loop}} rather than being an implementation
> flaw.

### 5.5 The four failure modes

{#tbl:tool-failure-modes caption="Tool-use failures by whether a grammar can prevent them. The division is the same structural/semantic split as tbl:constraint-scope, and it decides where engineering effort should go."}

| Failure | Example | Preventable by constraint | Detectable |
|---|---|---|---|
| Malformed arguments | invalid JSON, missing field | **yes** | at parse |
| Hallucinated tool | calls `get_wether` | **yes** | at dispatch |
| Wrong tool | searches when it should calculate | no | only by outcome |
| Wrong arguments | right tool, plausible wrong value | no | sometimes, by validation |
| Unnecessary call | calls a tool when it knew the answer | no | by cost |
| Missing call | answers from memory when a tool was needed | no | by correctness |

**The last two are worth naming separately** because they are invisible to every
check that examines the call itself — the call is well-formed and correct in
isolation, and wrong only relative to what should have happened.

## 6. Mathematical Foundation

### 6.1 Tool selection degrades with tool count

Treat selection as classification over $k$ tools. Suppose the model's score for
the correct tool is $s^* $ and for each distractor is drawn from a distribution
with mean $\mu < s^*$ and variance $\sigma^2$.

Correct selection requires the true tool to beat all $k-1$ distractors. As $k$
grows, the expected maximum distractor score grows as

$$
\E\big[\max_{j\ne *} s_j\big] \approx \mu + \sigma\sqrt{2\ln(k-1)}
$$ (eq:max-distractor)

so accuracy falls once

$$
s^* - \mu \lesssim \sigma\sqrt{2\ln(k-1)}
$$ (eq:selection-degradation)

$\square$

**The degradation is logarithmic in $k$, which is slow — and the variance term
is what matters.** Two tools with similar descriptions have small $s^*-\mu$ and
fail at small $k$; well-differentiated tools survive large $k$. **The number of
tools matters less than how distinguishable they are**, which is the actionable
form of the result.

### 6.2 Error compounding across rounds

If each round succeeds with probability $p$, a $k$-round task succeeds with

$$
P(\text{success}) = p^{k}
$$ (eq:tool-chain-success)

At $p = 0.95$: three rounds give 0.857, ten rounds give 0.599.

$\square$

**This is {{eq:exact-match-composition}} again** — the emergence chapter's
observation that exact-match over $k$ steps compresses a smooth quantity. Here
it says something operational: **a multi-step tool task is far less reliable
than its per-step accuracy suggests**, and the only way to make long chains work
is to raise $p$ very close to 1 or to allow recovery from failures, which is why
error feedback in {{sec:7-internal-mechanics}} matters so much.

### 6.3 When to call a tool at all

The model should call a tool when the expected value of the information exceeds
the cost. With $a_0$ the accuracy of answering directly, $a_1$ with the tool, and
$\kappa$ the tool's cost in units of answer value:

$$
\text{call if } a_1 - a_0 > \kappa
$$ (eq:call-decision)

**Nothing in the model's training optimises this.** It emits a call when the
context makes a call likely, which correlates with usefulness and is not the
same thing. That is why unnecessary calls are common and why the cost side of
{{eq:call-decision}} has to be enforced by the *prompt* — "do not search for
things you know" — or by the loop.

### 6.4 A worked reliability calculation

A task requiring 4 tool rounds, per-round success 0.93.

$$
P(\text{success}) = 0.93^4 = 0.748
$$

Now add error feedback: a failed round is retried once with the error message in
context, succeeding with probability 0.7 on retry. Per-round effective success:

$$
p' = 0.93 + 0.07\times 0.7 = 0.979
$$

$$
P(\text{success}) = 0.979^4 = 0.918
$$

**Error feedback moved a 4-round task from 75% to 92%**, at the cost of some
extra rounds. That is the single highest-leverage intervention in tool-using
systems, and it consists of putting the error message back into the context
rather than swallowing it.

## 7. Internal Mechanics

```mermaid {#fig:tool-loop caption="The dispatch loop. Everything outside the model box is your code — including the permission check, which is the only place a dangerous action can be prevented, since the model merely requests."}
graph TD
  A["conversation + tool schemas"] --> B["model generates"]
  B --> C{"tool call<br/>or answer?"}
  C -- answer --> D["return to user"]
  C -- call --> E["parse + validate<br/>ch:llm-structured-output"]
  E --> F{"permitted?"}
  F -- no --> G["inject refusal as<br/>tool result"]
  F -- yes --> H["EXECUTE<br/>your code"]
  H --> I["inject result as context"]
  G --> I
  I --> J{"iteration<br/>limit?"}
  J -- no --> B
  J -- yes --> K["terminate"]
  style H fill:#dfe,stroke:#5a5
  style F fill:#fde,stroke:#c69
```

**Error messages must go back to the model.** A tool that fails should return
its error *as the tool result*, not raise. The model can then correct — a wrong
date format, a missing filter, an out-of-range value — and
{{sec:6-mathematical-foundation}} showed this moving a four-round task from 75%
to 92%. Swallowing the error and retrying blind loses the only information that
would fix it.

**Errors must be specific enough to act on.** "Invalid input" tells the model
nothing; "start_date must be ISO 8601, received '03/04/2024'" tells it exactly
what to change. **The error message is a prompt**, and it should be written like
one.

**Parallel calls.** When several independent calls are needed, emitting them
together saves rounds — and {{eq:tool-loop-cost}} makes that saving quadratic
rather than linear. The model must be able to recognise independence, which it
does unreliably, and the loop must handle partial failure across a batch.

**Tool descriptions are prompts and are subject to everything in
{{ch:llm-prompting}}.** They compete for attention with the conversation, they
are sensitive to phrasing, and they occupy context on every request. A
twenty-tool schema set can be thousands of tokens of prefill paid per turn,
which prefix caching largely recovers.

**Temperature should be 0.** Tool calling is a task with a correct answer
({{ch:llm-decoding}}), and sampling introduces variation into argument values
for no benefit. This is one of the clearest cases in the book where the provider
default is wrong for the task.

**Where the permission check belongs.** Between parse and execute, and nowhere
else. Checking before generation cannot work — the model has not decided yet;
checking after execution is too late. {{fig:tool-loop}}'s diamond is the only
correct location, and it is your code.

**A refused call must still produce a tool result.** When the permission check
denies a call, the loop cannot simply drop it: the model is waiting for a result
and will otherwise repeat the request or hallucinate an outcome. Injecting an
explicit refusal — "permission denied: this tool requires user confirmation" —
keeps the conversation coherent and lets the model route around the restriction
or explain it to the user. {{fig:tool-loop}} shows this path explicitly for that
reason, and it is the same principle as returning errors rather than raising.

**Idempotency is the loop's unstated assumption.** {{eq:dispatch-loop}} may
retry, and the model may repeat a call it believes failed. For a read that is
harmless; for `send_email` or `create_ticket` it is not. **Any tool with side
effects needs an idempotency key supplied by the loop rather than by the
model**, because the model has no reliable notion of whether it has already
acted — its only evidence is what is in its context, and a timed-out call may
have succeeded without leaving one.

**Tool results should be summarised, not pasted.** A search returning fifty
results consumes the context window for a turn and every turn after it
({{eq:tool-loop-cost}}), and {{cite:liu2023lost}}'s position effects mean the
middle of a long result is poorly used anyway. Truncating or summarising before
injection is a design decision with a real quality effect, and the naive
implementation — inject everything — degrades a long conversation quietly.

## 8. Implementation

The full dispatch loop, with the four failure modes made visible.

```python {tier=A name=tool-dispatch-loop}
"""A complete tool-calling loop, with the four failure modes distinguished."""
import json

TOOLS = {
    "get_weather": {
        "description": "Current weather for a city.",
        "params": {"city": str, "units": str},
        "required": ["city"],
        "enum": {"units": ["celsius", "fahrenheit"]},
    },
    "search_docs": {
        "description": "Full-text search over the internal documentation.",
        "params": {"query": str, "limit": int},
        "required": ["query"],
        "enum": {},
    },
    "calculate": {
        "description": "Evaluate an arithmetic expression.",
        "params": {"expression": str},
        "required": ["expression"],
        "enum": {},
    },
}


def validate_call(call):
    """Returns (ok, message). This is the parse+validate box of fig:tool-loop."""
    if not isinstance(call, dict) or "name" not in call:
        return False, "malformed: no 'name' field"
    name = call["name"]
    if name not in TOOLS:
        return False, (f"hallucinated tool {name!r}; "
                       f"available: {sorted(TOOLS)}")
    spec = TOOLS[name]
    args = call.get("arguments", {})
    if not isinstance(args, dict):
        return False, "malformed: 'arguments' is not an object"
    for req in spec["required"]:
        if req not in args:
            return False, f"missing required parameter {req!r}"
    for key, val in args.items():
        if key not in spec["params"]:
            return False, f"unknown parameter {key!r} for {name}"
        want = spec["params"][key]
        if not isinstance(val, want):
            return False, (f"parameter {key!r} must be {want.__name__}, "
                           f"received {type(val).__name__}")
        if key in spec["enum"] and val not in spec["enum"][key]:
            return False, (f"parameter {key!r} must be one of "
                           f"{spec['enum'][key]}, received {val!r}")
    return True, "ok"


CANDIDATE_CALLS = [
    ({"name": "get_weather", "arguments": {"city": "Paris"}}, "correct"),
    ({"name": "get_wether", "arguments": {"city": "Paris"}}, "hallucinated tool"),
    ({"name": "get_weather", "arguments": {}}, "missing required"),
    ({"name": "get_weather", "arguments": {"city": 42}}, "wrong type"),
    ({"name": "get_weather", "arguments": {"city": "Paris",
                                           "units": "kelvin"}}, "bad enum"),
    ({"name": "search_docs", "arguments": {"query": "weather in Paris"}},
     "WRONG TOOL — valid, and not what was asked"),
]

print(f"{'call':<52} {'valid':>7}  diagnosis")
for call, label in CANDIDATE_CALLS:
    ok, msg = validate_call(call)
    shown = json.dumps(call)[:50]
    print(f"{shown:<52} {str(ok):>7}  {msg if not ok else label}")

print("""
The last row is the important one. It passes every structural check — real tool,
required parameter present, correct types — and it is the wrong tool for the
question. No validator catches that, because the call is well-formed and wrong
only relative to intent.

That is tbl:tool-failure-modes' division: the first four rows are structural and
a grammar makes them unreachable; the last is semantic and survives.""")


# --- the loop itself -------------------------------------------------------
def execute(call):
    """Your code. Returns a STRING result, including for errors — the model
    can only see what is put back into its context."""
    name, args = call["name"], call.get("arguments", {})
    if name == "get_weather":
        if args["city"] not in ("Paris", "London"):
            return f"ERROR: unknown city {args['city']!r}; try Paris or London"
        return f"{args['city']}: 14C, overcast"
    if name == "calculate":
        expr = args["expression"]
        if not set(expr) <= set("0123456789+-*/(). "):
            return f"ERROR: expression contains disallowed characters"
        try:
            return f"result: {eval(expr, {'__builtins__': {}}, {})}"
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"
    if name == "search_docs":
        return f"3 results for {args['query']!r}"
    return "ERROR: not implemented"


def dispatch_loop(scripted_responses, max_iterations=6, verbose=True):
    """Equation (eq:dispatch-loop). `scripted_responses` stands in for the
    model so the loop's behaviour is deterministic and inspectable."""
    context, rounds = [], 0
    for step, response in enumerate(scripted_responses):
        if response.get("final"):
            if verbose:
                print(f"  [{step}] final answer: {response['final']}")
            return response["final"], rounds, context
        rounds += 1
        if rounds > max_iterations:
            return "ERROR: iteration limit", rounds, context
        call = response["call"]
        ok, msg = validate_call(call)
        result = execute(call) if ok else f"ERROR: {msg}"
        context.append({"call": call, "result": result})
        if verbose:
            print(f"  [{step}] call {call['name']}({call.get('arguments')}) "
                  f"-> {result}")
    return "ERROR: ran out of scripted responses", rounds, context


print("\nA loop where the model gets it wrong, sees the error, and recovers:\n")
answer, rounds, ctx = dispatch_loop([
    {"call": {"name": "get_weather", "arguments": {"city": "Pariss"}}},
    {"call": {"name": "get_weather", "arguments": {"city": "Paris"}}},
    {"final": "It is 14C and overcast in Paris."},
])
print(f"\n  rounds used: {rounds}, context entries: {len(ctx)}")
print("The error string went back into the context, and the next call was "
      "corrected. That recovery is only possible because execute() RETURNED "
      "the error rather than raising it.")
```

Now the reliability arithmetic that governs multi-step tasks:

```python {tier=A name=tool-chain-reliability}
"""Multi-round reliability, and what error feedback buys. Eq (eq:tool-chain-success)."""
import numpy as np

print(f"{'per-round p':>12} " + " ".join(f"{'k=' + str(k):>8}" for k in
                                          (1, 2, 3, 5, 10, 20)))
for p in (0.99, 0.97, 0.95, 0.90, 0.80):
    row = " ".join(f"{p ** k:>8.3f}" for k in (1, 2, 3, 5, 10, 20))
    print(f"{p:>12.2f} {row}")

print("""
Read across a row. At 95% per-round reliability — which sounds excellent — a
ten-round task succeeds 60% of the time and a twenty-round task 36%. This is
eq:exact-match-composition from the emergence chapter, and it is why long
autonomous chains are hard in a way that per-step benchmarks never reveal.""")

# What error feedback does — equation in section 6.4.
print(f"\n{'rounds':>7} {'no feedback':>13} {'with feedback':>15} {'gain':>8}")
P_BASE, P_RECOVER = 0.93, 0.70
p_effective = P_BASE + (1 - P_BASE) * P_RECOVER
for k in (1, 2, 4, 8, 16):
    a, b = P_BASE ** k, p_effective ** k
    print(f"{k:>7} {a:>13.3f} {b:>15.3f} {b - a:>+8.3f}")

print(f"\nper-round success rises {P_BASE:.2f} -> {p_effective:.3f} with one "
      f"retry at {P_RECOVER:.0%} recovery")
print("""
Error feedback is the highest-leverage intervention available, and it is almost
free: return the error as the tool result instead of raising, and make the
message specific enough to act on. A four-round task goes from 75% to 92%.

Note the gain GROWS with chain length, which is the opposite of most
interventions — the longer the task, the more feedback is worth.""")

# Iteration limits: necessary, and they cost you the tail.
print(f"\n{'limit':>7} {'tasks completed':>18} {'cut off':>10}")
rng = np.random.default_rng(0)
needed = rng.geometric(0.35, size=20000)          # rounds a task actually needs
for limit in (2, 3, 5, 8, 12, 20):
    done = float((needed <= limit).mean())
    print(f"{limit:>7} {done:>18.1%} {1 - done:>10.1%}")

print("""
An iteration limit is mandatory — without one a confused model loops forever —
and it truncates the tail of genuinely long tasks. The limit is a product
decision about which tasks you are willing to fail, not a safety valve that
costs nothing.""")
```

And the tool-selection result, which is the actionable one:

```python {tier=A name=tool-selection}
"""How many tools is too many? Equation (eq:selection-degradation)."""
import numpy as np

rng = np.random.default_rng(2)
TRIALS = 4000


def selection_accuracy(n_tools, separation, noise=1.0):
    """The correct tool scores `separation` above the distractors' mean;
    all scores are noisy. Selection succeeds if the correct one wins."""
    wins = 0
    for _ in range(TRIALS):
        correct = separation + rng.normal(0, noise)
        distractors = rng.normal(0, noise, n_tools - 1)
        if correct > distractors.max(initial=-np.inf):
            wins += 1
    return wins / TRIALS


print("Tool-selection accuracy against tool count\n")
print(f"{'tools':>7} " + " ".join(f"{'sep=' + str(s):>10}"
                                   for s in (0.5, 1.0, 2.0, 3.0)))
for k in (2, 5, 10, 25, 50, 100):
    row = " ".join(f"{selection_accuracy(k, s):>10.3f}"
                   for s in (0.5, 1.0, 2.0, 3.0))
    print(f"{k:>7} {row}")

print("""
Read down the columns rather than across the rows.

With poorly differentiated tools (sep=0.5) accuracy is bad at five tools and
hopeless at fifty. With well-differentiated ones (sep=3.0) it is still good at a
hundred. The degradation with tool COUNT is logarithmic and slow
(eq:max-distractor); the dependence on SEPARATION is what actually decides it.

So 'how many tools can a model handle' is the wrong question. The right one is
'are my tool descriptions distinguishable', and the fix for a system that
selects badly is usually to merge overlapping tools and sharpen descriptions
rather than to reduce the count.""")

# What separation looks like in practice.
print(f"\n{'tool pair':<44} {'overlap'}")
PAIRS = [
    ("search_docs / query_knowledge_base", "high — merge them"),
    ("search_docs / calculate", "none"),
    ("get_weather / get_forecast", "high — one tool, a time parameter"),
    ("send_email / send_notification", "moderate — clarify in descriptions"),
]
for pair, note in PAIRS:
    print(f"{pair:<44} {note}")

# The context cost of carrying tools.
print(f"\n{'tools':>7} {'schema tokens':>15} {'prefill/turn @2N':>18} "
      f"{'per 1M turns':>14}")
TOKENS_PER_TOOL, N = 120, 7e9
for k in (5, 20, 50, 100):
    toks = k * TOKENS_PER_TOOL
    flops = 2 * N * toks
    print(f"{k:>7} {toks:>15,} {flops:>18.2e} {flops * 1e6:>14.2e}")

print("""
Tool schemas are prompt tokens paid on every turn (ch:llm-inference). A
hundred-tool schema set is 12,000 tokens of prefill before the conversation
starts, which is both a latency cost and a large share of the context window.

Prefix caching recovers most of the compute — the schema block is byte-identical
across turns — but not the context-window space, and the space competes with
conversation history and retrieved content.""")
```

## 9. Practical Example

A team's assistant has 34 tools and users report it "picks the wrong one". The
instinct is to write better descriptions. The measurement says the problem is
structural.

```python {tier=A name=tool-set-audit}
"""Auditing a tool set: overlap, context cost, and where selection fails."""
import numpy as np

rng = np.random.default_rng(7)

# A realistic tool set, with the overlaps real systems accumulate.
TOOLS = {
    "search_documents":      "search",
    "search_knowledge_base": "search",
    "find_files":            "search",
    "lookup_policy":         "search",
    "get_user":              "user",
    "get_user_profile":      "user",
    "get_customer_record":   "user",
    "send_email":            "notify",
    "send_slack_message":    "notify",
    "create_ticket":         "write",
    "update_ticket":         "write",
    "close_ticket":          "write",
    "calculate":             "compute",
    "run_report":            "compute",
}
# Add filler tools to reach a realistic count.
for i in range(20):
    TOOLS[f"misc_tool_{i}"] = f"unique_{i}"

groups = {}
for name, family in TOOLS.items():
    groups.setdefault(family, []).append(name)

print(f"{len(TOOLS)} tools in {len(groups)} functional families\n")
print(f"{'family':<12} {'tools':>6}  {'members'}")
for fam, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    if len(members) > 1:
        print(f"{fam:<12} {len(members):>6}  {', '.join(members)}")

confusable = sum(len(m) for m in groups.values() if len(m) > 1)
print(f"\n{confusable} of {len(TOOLS)} tools sit in a confusable family "
      f"({confusable / len(TOOLS):.0%})")


def accuracy(n_competitors, separation, trials=4000):
    wins = 0
    for _ in range(trials):
        correct = separation + rng.normal()
        if n_competitors == 0:
            wins += 1
            continue
        if correct > rng.normal(size=n_competitors).max():
            wins += 1
    return wins / trials


# Within-family separation is low; across-family separation is high.
WITHIN_SEP, ACROSS_SEP = 0.6, 2.8
print(f"\n{'query targets':<28} {'competitors':>12} {'separation':>12} "
      f"{'accuracy':>10}")
for fam, members in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:4]:
    n_within = len(members) - 1
    sep = WITHIN_SEP if n_within else ACROSS_SEP
    acc = accuracy(len(TOOLS) - 1, sep) if n_within == 0 else accuracy(
        n_within, WITHIN_SEP)
    print(f"{fam:<28} {n_within:>12} {sep:>12.1f} {acc:>10.3f}")

# The intervention: merge within families, keep the parameter distinction.
merged = len(groups)
print(f"\n{'configuration':<30} {'tools':>7} {'schema tokens':>15} "
      f"{'est. selection acc':>20}")
for label, k, sep in [("as-is (34 tools)", len(TOOLS), WITHIN_SEP),
                      ("merged by family", merged, ACROSS_SEP)]:
    print(f"{label:<30} {k:>7} {k * 120:>15,} "
          f"{accuracy(k - 1, sep):>20.3f}")

print("""
Look at what changed and what did not. The tool count fell from 34 to 25 — barely
a quarter — while estimated selection accuracy went from 0.088 to 0.781. The
count is not what fixed it.

eq:max-distractor says count costs only logarithmically. What was actually wrong
is that fourteen tools sat in families whose members are barely distinguishable,
so within a family the separation is small and selection is close to a coin
flip.

Merging each family into one tool with a parameter — search(scope=...) rather
than four search tools — raises separation and cuts schema tokens at the same
time. Writing better descriptions for four tools that genuinely do the same
thing does not, because no description makes two identical functions
distinguishable.""")
```

> PRODUCTION TIP: Audit tool sets for families before writing descriptions. Two
> tools a human would struggle to choose between are two tools the model will
> also struggle with, and the fix is usually one tool with a parameter.

## 10. Production Considerations

**Constrain the tool name and arguments.** Both are structural
({{eq:tool-call}}) and {{ch:llm-structured-output}} eliminates them as failure
modes entirely.

**Return errors as tool results.** The single highest-leverage intervention —
75% to 92% on a four-round task — and it requires only not raising.

**Write error messages as prompts.** Specific enough to act on.

**Set $T = 0$.** Tool calling has a correct answer.

**Enforce an iteration limit, and treat it as a product decision.**
`tool-chain-reliability` shows what each limit costs in completed tasks.

**Put the permission check between parse and execute.** It is the only correct
location and it is your code.

**Treat tool output as untrusted.** {{eq:dispatch-loop}} injects it as ordinary
context with no privileged status ({{part:26}}).

**What to monitor:** calls per conversation, tool-selection distribution,
validation-failure rate by type, error-recovery rate, iteration-limit hits, and
schema token count. The selection distribution is the one that reveals a
confusable family — a tool that is never chosen usually has a twin.

## 11. Common Mistakes

**Beginners:**

*Believing the model executes the function.* It emits a request; your code
decides ({{sec:4-intuitive-explanation}}).

*Raising on tool errors.* The model cannot see an exception.

*Using the default temperature.* Sampling argument values is variation without
benefit.

**Experienced practitioners:**

*Adding tools without auditing overlap.* `tool-set-audit` shows count costing
logarithmically and confusability costing a great deal.

*Omitting the iteration limit.* A confused model loops until the budget is gone.

*Trusting tool output.* It enters the context on equal terms with your
instructions.

*Measuring per-step accuracy and inferring task reliability.*
{{eq:tool-chain-success}} — 95% per round is 60% over ten.

*Constraining the whole response.* It prevents reasoning before the call, losing
{{eq:cot-depth}}'s benefit — constrain the call, not the turn.

*Letting the model supply idempotency keys.* It cannot know whether a timed-out
call succeeded; the loop must supply the key
({{sec:7-internal-mechanics}}).

## 12. Failure Modes

**Wrong tool selected.** *Symptom:* well-formed calls producing irrelevant
results. *Detection:* selection distribution against expectation. *Cause:*
usually a confusable family, not a bad description.

**Argument hallucination.** Plausible values not grounded in the conversation —
an invented ID, a guessed date. *Detection:* validate arguments against the
conversation where possible; this is
{{ch:nlp-extraction}}'s grounding check applied to arguments.

**Infinite loop.** The model repeats a failing call. *Cause:* an error message
that does not say what to change. *Detection:* repeated identical calls within a
conversation.

**Context exhaustion mid-loop.** {{eq:tool-loop-cost}} grows quadratically and
long loops hit the window. *Symptom:* failures correlated with round count.

**Unnecessary calls.** {{eq:call-decision}} is not optimised by anything.
*Detection:* calls per conversation against a baseline.

**Indirect prompt injection.** Tool output containing instructions.
*This is not a bug in the loop; it is what the loop does.* {{part:26}}.

## 13. Alternatives

{#tbl:tool-integration-patterns caption="Ways to connect a model to external capability. The first row is this chapter; the others trade flexibility against reliability, and the last two remove the model's discretion entirely."}

| Pattern | Model decides | Reliability | Where used |
|---|---|---|---|
| Function calling | which tool, when, arguments | moderate | general assistants |
| Constrained function calling | same, structurally guaranteed | better | production |
| Fixed pipeline with LLM steps | nothing about control flow | high | known workflows |
| Retrieval always-on | nothing | high | RAG ({{part:12}}) |
| Code generation + execution | everything | low, powerful | data analysis |
| Toolformer-style training | learned when to call | moderate | {{cite:schick2023}} |

**What genuinely differs.** The first two are the same mechanism with different
guarantees. **The third is the important alternative and it is under-used**: if
you know the workflow, encoding it as a pipeline with the model filling in steps
is far more reliable than letting the model choose the control flow — and
{{eq:tool-chain-success}} says why, since a fixed pipeline has no selection step
to get wrong.

**Code generation is the extreme.** Instead of $k$ tools, one tool that runs
arbitrary code. Maximum flexibility, and the permission check becomes
intractable — which is why it appears in sandboxed analysis contexts and not in
general assistants.

## 14. Evaluation

**Is the loop correct?**

1. **Validation catches all four structural failures**, tested with deliberately
   malformed calls — the table in `tool-dispatch-loop`.
2. **Errors reach the model** as results, verified by a test where the first
   call fails and the second succeeds.
3. **The iteration limit terminates**, tested with a model that always calls.
4. **Permission checks run before execution**, tested with a forbidden call.

**Is tool use working?**

1. **Task success end to end**, not per-call accuracy —
   {{eq:tool-chain-success}} makes the difference large.
2. **Selection accuracy per tool**, which reveals confusable families.
3. **Recovery rate** after an error.
4. **Calls per task** against a reasonable baseline, for
   {{eq:call-decision}}'s cost side.

**The measurement that matters is end-to-end.** A system with 97% per-call
accuracy and a four-round average task succeeds 88% of the time, and reporting
the 97% is reporting the wrong number.

## 15. Advanced Concepts

**Parallel tool calls.** {{maturity:ESTABLISHED}} Emitting independent calls
together, saving rounds — and {{eq:tool-loop-cost}} makes the saving quadratic.
Requires the model to recognise independence and the loop to handle partial
failure.

**Learned tool use.** {{maturity:EMERGING}} {{cite:schick2023}}'s Toolformer
learns *when* a call helps from self-supervised data, which is
{{eq:call-decision}} being optimised rather than hoped for — the only work that
directly addresses the unnecessary-call problem.

**Tool retrieval.** {{maturity:EMERGING}} With hundreds of tools, retrieve the
relevant few into context rather than carrying all
({{part:11}}). Attacks both the schema-token cost and the selection problem, and
introduces a retrieval failure mode.

**Sandboxed code execution.** {{maturity:ESTABLISHED}} One tool that runs code,
in an isolated environment. Powerful, and it moves the entire permission
question into sandbox design.

**Model Context Protocol.** {{maturity:EMERGING}} Standardising tool definition
and transport so tools are portable across models and hosts.
{{part:19}}'s subject, and the reason tool schemas are converging on a common
shape.

## 16. Connection to Previous Chapters

**Backwards.** {{ch:llm-structured-output}} provides the guarantee for
{{eq:tool-call}}'s two structural components, and
{{tbl:constraint-scope}}'s structural/semantic split becomes
{{tbl:tool-failure-modes}}. {{ch:nlp-extraction}}'s grounding check applies to
argument hallucination. {{eq:tool-chain-success}} is
{{eq:exact-match-composition}} from {{ch:fm-emergence}}.
{{ch:llm-inference}}'s prefix caching is what makes
{{eq:tool-loop-cost}} tolerable. {{ch:llm-decoding}} is why $T=0$.
{{ch:fm-instruction-tuning}} is why the model produces well-formed calls at all.

**Forwards.** {{part:17}} is this loop with planning and memory added — agents
are not a new mechanism, they are {{eq:dispatch-loop}} with more structure
around it. {{part:19}} standardises tool definitions. {{part:26}} treats the
untrusted-tool-output problem this chapter identifies. {{part:12}} is retrieval
as an always-on tool.

## 17. Exercises

**Beginner**

1. Does the model execute functions? Explain what it actually does.
2. List the four structural failure modes and which are constraint-preventable.
3. Why must tool errors be returned rather than raised?

**Intermediate**

4. Using {{eq:tool-chain-success}}, compute success for 6 rounds at $p=0.92$.
5. Compute the effective per-round success with one retry at 60% recovery from a
   base of 0.90.
6. Using {{eq:tool-loop-cost}}, compare cumulative prefill for 8 rounds with and
   without prefix caching.

**Advanced**

7. Derive {{eq:max-distractor}} and explain why separation matters more than
   count.
8. Explain why the permission check cannot be placed before generation or after
   execution.
9. Design a scheme for detecting argument hallucination, and state its limits.

**Implementation**

10. Extend `tool-dispatch-loop` with constrained decoding over the tool name and
    schema, and show the first two failure modes becoming unreachable.
11. Implement parallel tool calls with partial-failure handling, and measure the
    round saving on a task needing three independent lookups.
12. Implement tool retrieval: embed tool descriptions, retrieve the top 5 per
    turn, and measure selection accuracy against carrying all of them.
13. Build the loop-detection check from {{sec:12-failure-modes}} and verify it
    catches a model repeating a failing call.

**Reasoning**

14. Your assistant makes well-formed calls to the wrong tools. Rank the possible
    causes and say what you would measure first.
15. Explain why a fixed pipeline is more reliable than tool calling for a known
    workflow, using {{eq:tool-chain-success}}.

## 18. Interview Questions

**Beginner**

1. What is function calling and what does the model actually emit?
2. What are the failure modes of tool use?
3. Why is temperature 0 right here?

**Intermediate**

4. Walk through the dispatch loop. What is your code's responsibility?
5. Why does context grow quadratically in cost across rounds?
6. How does per-round reliability relate to task reliability?

**Senior**

7. Your system has 40 tools and selects badly. Diagnose and fix.
8. How would you make a ten-round tool task reliable?
9. Where does the permission check go, and why nowhere else?

**Systems**

10. Design tool-call observability. What do you log per call?
11. How would you handle a tool that returns attacker-controlled content?

## 19. Research Questions

**Can the call/no-call decision be optimised?** {{eq:call-decision}} is not
optimised by any training objective, and {{cite:schick2023}} is the only serious
attempt. Measure how far from optimal current models are on tasks where $a_0$,
$a_1$ and $\kappa$ are all known.

**How much does tool retrieval cost in selection accuracy?** Retrieving a subset
attacks both context cost and confusability, and introduces a retrieval miss.
Characterise the tradeoff as a function of tool-set size and description
quality.

**Is argument hallucination detectable in general?** Grounding works when
arguments should appear in the conversation, and many should not — a computed
value, a default. Whether a general detector is possible, or only a per-schema
one, is open.

**Does constrained tool calling cost selection quality?** Constraining the name
to the valid set removes hallucinated tools and could in principle push
probability toward a *wrong* valid tool. Measure whether it does, with the
unconstrained baseline's hallucinated calls counted as failures.

## 20. Chapter Summary

Function calling is structured output plus a loop. **The model never executes
anything** — it emits a request, your code parses it, checks permission, and
decides. Every security property in {{part:26}} follows from that.

{{eq:tool-call}}'s two components — a name from a finite set and arguments
matching a schema — are both *structural*, so
{{ch:llm-structured-output}}'s machinery makes malformed arguments and
hallucinated tool names unreachable. **What survives is semantic**: the right
format calling the wrong tool, or the right tool with plausible wrong arguments,
and {{tbl:tool-failure-modes}} is {{tbl:constraint-scope}}'s division applied to
actions.

**Reliability compounds badly.** {{eq:tool-chain-success}} says a ten-round task
at 95% per-round reliability succeeds 60% of the time — the same
exact-match-over-$k$-steps structure as {{ch:fm-emergence}}'s emergence curves.
The highest-leverage fix is **error feedback**: returning the error as the tool
result instead of raising moves a four-round task from 75% to 92%, and its value
*grows* with chain length, which is unusual among interventions.

**Tool count is not the problem; confusability is.**
{{eq:max-distractor}} makes degradation logarithmic in the number of tools and
sharply dependent on how distinguishable they are — `tool-set-audit` shows a
34-tool system failing because fourteen tools sit in families whose members do
the same thing. **No description makes two identical functions
distinguishable**, so the fix is merging into one tool with a parameter.

Two consequences of {{eq:dispatch-loop}} that nobody designed: context grows so
that cumulative prefill is $O(I^2m)$ {{eq:tool-loop-cost}}, largely recovered by
prefix caching; and tool results enter the context **with no privileged status**,
which is exactly the indirect prompt injection surface — not an implementation
flaw but a direct consequence of how the loop must work.

## 21. Further Reading

{{cite:schick2023}} is the one research paper here that addresses a question the
engineering does not: *when* is a call worth making. Its self-supervised
approach — insert candidate calls, keep the ones that reduce perplexity on the
continuation — is elegant and is the only serious attack on
{{eq:call-decision}}.

{{cite:willard2023}} from the previous chapter is the practical foundation, since
a tool call is a schema and everything in it applies unchanged.

Beyond those, the useful sources are provider API documentation and the source
of open agent frameworks, for the same reason as
{{ch:llm-prompt-lifecycle}}: this is engineering practice that has not been
written up carefully, and reading a dispatch loop is worth more than reading
about one.

**Where to go next:** {{ch:llm-hallucination}} takes up the semantic failures
that constraints and validation leave behind — including the argument
hallucination this chapter identified and could not solve.
