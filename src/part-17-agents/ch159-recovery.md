---
id: ag-recovery
number: 159
part: XVII
tier: full
status: draft
requires: [checkpoints-cap-the-exponent, error-message-as-selector,
           stopping-is-a-classifier]
provides: [feedback-must-be-external, gating-costs-a-retry,
           feedback-quality-threshold, localise-before-diagnose,
           localisation-is-a-free-option, recovery-scales-with-length]
citations: [huang2024selfcorrect, shinn2023reflexion, madaan2023selfrefine,
            yao2023react, liu2024agentbench, zhou2024webarena,
            greshake2023indirect]
---

## 1. Learning Objectives

By the end of this chapter you will be able to say why an agent that assesses its
own work can do *worse* than one that retries blindly, and identify the mechanism;
compute the feedback-quality threshold below which acting on a signal is negative;
separate detection, localisation and diagnosis and rank what each is worth; explain
why a failure localiser beats restarting even when it is usually wrong; and give the
build order for a recovery system, which is not the one most teams follow.

## 2. Why This Matters

{{ch:rsn-self-consistency}} established that intrinsic self-correction converges to
the model's own mode, and {{cite:huang2024selfcorrect}} found it does not improve
reasoning while external feedback does. An agent looks like it escapes that,
because it has an *environment*: the tool errors, the test fails, the page does not
load. That is genuinely external and it is exactly the missing ingredient.

This chapter measures how much of the escape survives, and finds two results that
change what to build.

The first is that **blind retry beats self-assessment** — $88.8\%$ against $73.9\%$
in {{sec:9-practical-example}}. An agent that knows nothing about its failure does
better than one that tries to work out what happened. The mechanism is not
information content; it is *gating*. Self-assessment supplies the decision to retry
as well as the diagnosis, and on the attempts the agent botched, its judgement is
the judgement that is unreliable. A failed self-assessment does not merely fail to
help — it throws away a retry that would otherwise have been free.

That generalises past self-assessment. Environment feedback falls below blind retry
too, once its quality drops far enough: **a feedback signal that gates a retry has
to be better than the retry it suppresses**, which is a much higher bar than
"better than nothing".

The second result reorders the engineering. "The agent noticed it failed and tried
again" bundles three separate things — detection, localisation, diagnosis — worth
very different amounts. Detection with a restart takes completion from $35.9\%$ to
$83.1\%$. Localisation, resuming from the last good state, takes it to $98.6\%$
*while spending fewer steps*. Diagnosis, additionally knowing why, adds $1.2$.

**Knowing where is worth about thirteen times knowing why**, which is the opposite
of where attention usually goes: teams write error taxonomies and reflection prompts
on top of a retry that starts from the beginning.

And localisation turns out to be a free option. Even a localiser that is wrong four
times in five still beats restarting, because a wrong guess costs exactly what
restarting always costs.

## 3. Prerequisites

You need {{ch:rsn-self-consistency}}'s correlated-critic result — it is the
mechanism behind the self-assessment column and behind
{{ch:ag-loop}}'s stopping classifier.

From {{ch:ag-planning}}, checkpoints: a verified segment boundary *is* the
known-good state this chapter's localisation resumes from, so a system with
checkpoints has localisation for free and one without cannot buy it.

From {{ch:ag-tool-calling}}, the error-message result: an informative error is what
turns a retry from a fresh sample into a conditioned one, which is this chapter's
diagnosis term.

And {{ch:ag-loop}}'s budget, because everything here is measured under one.

## 4. Intuitive Explanation

An agent fails at something. What happens next depends on three questions, and they
are usually collapsed into one.

**Does it know it failed?** Detection. Without this nothing happens at all.

**Does it know where it failed?** Localisation. This decides whether the retry
starts over or resumes.

**Does it know why?** Diagnosis. This decides whether the retry is a fresh attempt
or an informed one.

Start with the first, because it contains the chapter's most counterintuitive
result.

Suppose the agent judges its own work. When it correctly notices a failure, it
retries — good. When it wrongly believes it succeeded, it stops — and that is the
problem. The judgement gates the retry, and on exactly the attempts where the agent
did badly, its judgement of its own work is also bad. That is the correlated critic
from {{ch:rsn-self-consistency}}, arriving where it does the most damage.

Now compare with an agent that just retries every time without asking. It never
misses a failure, because it never checks. It spends more attempts, and if attempts
are affordable it wins.

The self-assessing agent's problem is not that its assessment is uninformative. It
is that the assessment is being used to *decide*, and a decision made on an
unreliable signal is worse than not making the decision at all when the default is
cheap. {{sec:9-practical-example}} measures blind retry beating self-assessment by
$14.9$ points.

The same logic applies to the environment, which is the part that transfers. If the
tool's error report is reliable, acting on it is excellent. If it is weak — a bare
`error`, an ambiguous status — and you use it to decide whether to continue, you can
end up worse off than an agent that ignores it and retries anyway. There is a
threshold, and it is measurable on your own tools.

Now the second half, which is about what the failure tells you rather than whether
you believe it.

Imagine an eight-step task where step six went wrong. Restarting means redoing
steps one to five, which were fine, and each of them can fail again this time. The
retry pays the whole task's failure probability. Resuming from step six pays only
the remainder.

That difference is large and it grows with task length, because the wasted prefix
grows. On a twenty-two-step task {{sec:9-practical-example}} measures restarting at
$17.0\%$ and resuming at $98.7\%$.

Against that, knowing *why* step six failed improves step six's odds — one step out
of the remaining three. Useful, and much smaller.

The last idea makes localisation cheap to adopt. You might reasonably object that
you cannot reliably tell which step failed. You do not need to. A wrong guess sends
the agent back to the start, which is exactly what it would have done anyway; a
right guess saves the prefix. **The downside is bounded by the alternative and the
upside is not**, so a bad localiser is strictly better than none.

## 5. Formal Explanation

Let an attempt succeed with probability $p$, and let a failed attempt be retried
$r-1$ times. Write $\phi$ for the probability that the retry is *informed* — that it
knows the cause and is therefore drawn from a better distribution $g > p$. Then
under an ungated policy:

$$S_{\text{blind}} = 1 - (1-p)^{r}$$ (eq:blind-retry)

Now gate the retry on a detector with sensitivity $\alpha$ — the probability it
notices a failure. Retries happen only when the detector fires:

$$S_{\text{gated}} = 1 - (1-p)\big(1 - \alpha\,[\,\phi g + (1-\phi)p\,]\big)^{\,r-1}$$ (eq:gating-costs-a-retry)

Compare the two. Gating multiplies the per-retry improvement by $\alpha$, so it is
better than blind retry only if the conditioning gain compensates:

$$\alpha\,\big[\phi g + (1-\phi)p\big] \;>\; p$$ (eq:feedback-quality-threshold)

**A feedback signal that gates a retry must clear a threshold set by the retry it
suppresses**, not by zero. For a self-critic, $\alpha$ is itself a function of the
agent's competence on that instance — the correlated-critic property — so $\alpha$ is
lowest exactly where the retry was most needed, and
{{eq:feedback-quality-threshold}} fails hardest there.

That is the formal content of {{cite:huang2024selfcorrect}}'s "can make things
worse", and it is more specific than the usual reading: intrinsic self-correction is
harmful *when it gates*, and merely useless when it only advises.

Now decompose what a failure signal contains. For a $k$-step task with per-step
success $p$, failing at step $j$:

$$S_{\text{restart}} \text{ re-runs } k \text{ steps}, \qquad S_{\text{resume}} \text{ re-runs } k - j \text{ steps}$$ (eq:localise-before-diagnose)

Each attempt's cost and failure exposure scale with the number of steps re-run, so
resumption's advantage is proportional to the expected wasted prefix
$\mathbb{E}[j]$. Under a geometric failure position that is roughly
$\min(1/(1-p), k)$, which grows with $k$ — hence:

$$\frac{S_{\text{resume}}}{S_{\text{restart}}} \text{ increases with } k$$ (eq:recovery-scales-with-length)

Diagnosis, by contrast, raises $p \to g$ for a *single* step. Its contribution is
$O(1)$ in $k$ while localisation's is $O(k)$, which is why the measured ratio is
about thirteen to one at $k = 8$ and widens.

Finally, localisation's robustness. Let the localiser name the right step with
probability $\lambda$ and otherwise force a restart:

$$S_{\text{loc}}(\lambda) = \lambda\, S_{\text{resume}} + (1-\lambda)\, S_{\text{restart}} \;\ge\; S_{\text{restart}}$$ (eq:localisation-is-a-free-option)

The inequality is unconditional for any $\lambda \ge 0$, because the failure mode of
a bad guess is exactly the baseline. **Localisation is a free option**: no accuracy
threshold, no downside, and the only cost is whatever it takes to produce a guess.

## 6. Mathematical Foundation

Three consequences.

**The threshold in {{eq:feedback-quality-threshold}} is computable from two
measurements.** $\alpha$ is how often your detector fires on genuine failures, and
$\phi g$ is the success rate of retries that acted on the signal. Both come from
logs. {{sec:9-practical-example}} finds the environment crossing below blind retry
somewhere between $45\%$ and $25\%$ feedback quality, and no team checks which side
of that its tools are on.

**Self-assessment's damage is worst at middling competence.** $\alpha$ rises with the
agent's ability and the need for retries falls with it, so the product peaks in the
middle. {{sec:9-practical-example}} measures the gap over no-retry at $+16.6$ points
at $25\%$ base competence, $+30.2$ at $55\%$, and $+13.6$ at $85\%$. The
environment column, by contrast, is above $96\%$ at every competence — **external
feedback rescues a weak agent and self-assessment cannot**.

**Localisation's value and checkpoint value are the same quantity.**
{{eq:localise-before-diagnose}}'s $k - j$ is the segment remainder in
{{ch:ag-planning}}'s {{eq:checkpoints-cap-the-exponent}}. A checkpoint provides a
verified $j$; a localiser guesses one. That means the two chapters are describing one
mechanism from two directions — success exponent and recovery cost — and a team that
builds checkpoints for either reason gets the other for free.

One thing the model omits and {{sec:12-failure-modes}} restores: retries here are
independent draws. A retry that repeats the same action is
{{ch:ag-loop}}'s non-productive cycle and contributes nothing, so blind retry's
advantage in {{sec:9-practical-example}} assumes the deduplication that chapter
recommended is already in place. Without it, blind retry degenerates.

## 7. Internal Mechanics

### 7.1 The three signals, and where each comes from

```mermaid {#fig:recovery-signals caption="What a failure can tell you, and the component that supplies each. Only the first is usually built deliberately."}
flowchart TD
    F[attempt fails] --> D[detection: something is wrong]
    D --> L[localisation: at step j]
    L --> X[diagnosis: because of x]
    D -. from .-> D1[test result, exception, checkpoint]
    L -. from .-> L1[checkpoint boundary, or a guess]
    X -. from .-> X1[the tool's error message]
```

Detection comes from the environment or from a judgement. Localisation comes from
{{ch:ag-planning}}'s checkpoints, or from a guess. Diagnosis comes from
{{ch:ag-tool-calling}}'s error messages. Each has a different owner, which is part
of why they get conflated in the retry code.

### 7.2 Advising versus gating

The distinction that decides whether a weak signal helps or hurts.

A signal **advises** when it conditions the retry but does not decide whether to
take it: "you failed, probably at step four" feeding into a retry that was going to
happen anyway. Its value is bounded below by zero — bad advice wastes an attempt
that was already budgeted.

A signal **gates** when it decides: "you succeeded, stop". Its value can be
negative, by {{eq:feedback-quality-threshold}}, because a wrong stop forfeits an
attempt.

**So route weak signals to advice and strong signals to gates.** An agent's own
judgement of its work is the canonical weak signal, and the canonical mistake is
wiring it to the gate.

### 7.3 Where the retry budget should be spent

Since {{eq:blind-retry}} says unconditional retries compound, and
{{eq:localise-before-diagnose}} says resumption makes each retry cheaper, the two
compose: a system with checkpoints can afford more retries because each one costs a
segment rather than a task.

That interacts with {{ch:ag-loop}}'s budget in a specific way. A budget expressed in
*steps* rewards resumption automatically. A budget expressed in *attempts* does not,
and will let a restarting policy burn the allowance. Express budgets in steps.

### 7.4 What reflection is actually for

{{cite:shinn2023reflexion}} keeps a written record of what went wrong and reads it
on the next attempt. This chapter's arithmetic says where its value comes from, and
it is not the insight.

It is that the record changes the context after a failure, which is
{{eq:context-change-breaks-loops}} — the reason the retry is not a repeat. A
reflection that says something wrong but *different* still breaks the cycle. That
reframes the quality bar: reflection needs to be varied more than it needs to be
correct, which is a much easier target.

{{cite:madaan2023selfrefine}}'s positive results sit here too, and
{{ch:rsn-self-consistency}}'s reconciliation applies: on tasks where recognising a
flaw is genuinely easier than producing an unflawed output, the self-critic is not
correlated in the damaging way. Generation tasks often qualify; correctness tasks
usually do not.

### 7.5 Recovery and untrusted content

A retry conditioned on an error message is a retry conditioned on text the tool
produced. If any part of that text can be influenced by an attacker — an error that
echoes user input, a page that returns a crafted message —
{{cite:greshake2023indirect}}'s injection vector runs straight through the recovery
path, which is one of the least-audited parts of an agent.

Treat error text as untrusted input to the same degree as tool output, because that
is what it is.

### 7.6 Why recovery is not the same as retrying

The two words are used interchangeably and the distinction is the whole chapter.

A **retry** re-runs work. Its expected value is bounded by the first attempt's
success rate, because it is another draw from the same distribution, and
{{eq:blind-retry}} says that compounds usefully but slowly.

**Recovery** changes something about the situation before re-running: where the
run resumes from, or what the next attempt knows, or which action is now
forbidden. Its expected value is bounded by how much it changed, which is why
{{eq:localise-before-diagnose}} and {{eq:context-change-breaks-loops}} are the two
equations that matter and {{eq:blind-retry}} is the baseline they beat.

That distinction predicts which interventions are worth building. Anything that
merely re-runs is a retry and its ceiling is the compounding in
{{eq:blind-retry}}. Anything that alters the starting state, the available
actions, or the information available is recovery, and it has no such ceiling.

It also explains a puzzle in {{sec:9-practical-example}}'s first table. Blind
retry looks like it is doing well for something so unintelligent, and it is —
because four independent draws at $42\%$ is genuinely $88.8\%$. The lesson is not
that blindness is good; it is that **the bar a recovery system has to clear is
higher than teams assume**, because the trivial baseline is already strong.

Anything you build should be measured against blind retry at the same step budget,
and a surprising amount of published agent tooling has never been.

## 8. Implementation

Two listings. The first compares recovery under three feedback regimes and finds
where gating turns negative. The second separates detection, localisation and
diagnosis and prices each.

```python {tier=A name=feedback-must-be-external}
"""Recovery works when something outside the agent grades it.

cite:huang2024selfcorrect separates intrinsic self-correction -- a model revising
using only its own capabilities -- from correction guided by external feedback, and
finds the first does not improve reasoning while the second does.
ch:rsn-self-consistency measured the mechanism: a self-critic's errors are
correlated with the solver's, so it is least reliable exactly where it is needed.

An agent looks like it escapes this, because it has an ENVIRONMENT. The tool
returns an error; the test fails; the page does not load. That is external, it is
not a sample from the model, and it is exactly the ingredient
cite:huang2024selfcorrect says is missing.

This listing asks how much of that escape survives the environment's feedback being
imperfect (eq:feedback-must-be-external), which it always is.
"""
import numpy as np

rng = np.random.default_rng(2281)

N = 60000
P_FIRST = 0.42          # chance the first attempt succeeds
ATTEMPTS = 4


def run(kind, quality=1.0, attempts=ATTEMPTS, p_first=P_FIRST):
    """kind:
       'none'  -- no retry at all
       'blind' -- always retry, learning nothing from the failure
       'env'   -- the environment reports the failure; `quality` is how often
                  that report is correct and useful
       'self'  -- the agent judges its own attempt. Its judgement is CORRELATED
                  with its competence: on attempts it got wrong, it is also
                  wrong about whether it got them wrong.
    """
    ok = rng.random(N) < p_first
    for _ in range(attempts - 1):
        if kind == "none":
            break
        failed = ~ok
        if kind == "blind":
            told = failed                     # knows it failed, learns nothing
            informed = np.zeros(N, dtype=bool)
        elif kind == "env":
            told = failed & (rng.random(N) < quality)
            informed = told
        else:  # self
            # Detecting its own failure is itself a skill it lacks on the
            # attempts it botched: correlate the detection with the outcome.
            told = failed & (rng.random(N) < quality * p_first)
            informed = told & (rng.random(N) < p_first)
        # An informed retry is conditioned on the fault and much more likely to
        # land; an uninformed one is a fresh draw from the same distribution.
        gain = np.where(informed, 0.75, p_first)
        ok |= told & (rng.random(N) < gain)
    return float(ok.mean())


print(f"First attempt succeeds {P_FIRST:.0%} of the time; up to {ATTEMPTS}")
print("attempts. An informed retry -- one that knows what went wrong -- lands")
print(f"{0.75:.0%} of the time; an uninformed one is a fresh {P_FIRST:.0%} draw.")
print()
print(f"{'recovery mode':>34}{'final success':>15}{'vs no retry':>13}")
print("-" * 62)
base = run("none")
modes = {}
for name, args in [("no retry", ("none",)),
                   ("blind retry", ("blind",)),
                   ("self-assessment", ("self",)),
                   ("environment feedback", ("env",))]:
    v = run(*args)
    modes[name] = v
    print(f"{name:>34}{v:>15.1%}{v - base:>+13.1%}")

print()
print()
print("Environment feedback is not perfect either. Sweep how often the")
print("environment's report is correct and actionable.")
print()
print(f"{'feedback quality':>18}{'environment':>14}{'self-assessment':>18}"
      f"{'blind retry':>14}")
print("-" * 64)
q_tab = {}
for q in (1.0, 0.85, 0.65, 0.45, 0.25):
    a = run("env", quality=q)
    b = run("self", quality=q)
    c = run("blind")
    q_tab[q] = (a, b, c)
    print(f"{q:>18.0%}{a:>14.1%}{b:>18.1%}{c:>14.1%}")

print()
print()
print("Where does self-assessment break? Sweep the agent's base competence,")
print("holding feedback quality at 85%.")
print()
print(f"{'first-attempt':>15}{'no retry':>11}{'self':>10}{'environment':>14}"
      f"{'self gap':>11}")
print("-" * 61)
comp = {}
for p in (0.25, 0.40, 0.55, 0.70, 0.85):
    n = run("none", p_first=p)
    s = run("self", quality=0.85, p_first=p)
    e = run("env", quality=0.85, p_first=p)
    comp[p] = (n, s, e)
    print(f"{p:>15.0%}{n:>11.1%}{s:>10.1%}{e:>14.1%}{s - n:>+11.1%}")

print()
print()
print("And how many attempts are worth having, under each regime.")
print()
print(f"{'attempts':>10}{'blind':>10}{'self':>10}{'environment':>14}")
print("-" * 44)
at = {}
for a in (1, 2, 3, 5, 8):
    at[a] = (run("blind", attempts=a), run("self", quality=0.85, attempts=a),
             run("env", quality=0.85, attempts=a))
    print(f"{a:>10}{at[a][0]:>10.1%}{at[a][1]:>10.1%}{at[a][2]:>14.1%}")

print(f"""
The first table has the result, and it is not the one the chapter was outlined to
find.

Environment feedback is the best mode at {modes['environment feedback']:.1%},
which is cite:huang2024selfcorrect's positive case and no surprise. Self-assessment
reaches {modes['self-assessment']:.1%}, well short of it, which is
cite:huang2024selfcorrect's negative case and also no surprise.

The surprise is the middle row. **Blind retry -- retrying with no idea what went
wrong -- scores {modes['blind retry']:.1%}, beating self-assessment by
{modes['blind retry'] - modes['self-assessment']:.1%}.**

An agent that knows nothing about its failure does better than one that tries to
work out what happened. That is worth understanding precisely, because it is not
about information content.

The mechanism is gating. Self-assessment does not just supply a diagnosis; it
supplies the DECISION TO RETRY. An agent that judges it succeeded does not try
again -- and on the attempts it botched, its judgement is exactly the judgement
that is unreliable (ch:rsn-self-consistency's correlated critic). So a failed
self-assessment does not merely fail to help. It throws away a retry the agent
would otherwise have taken for free.

Blind retry never makes that mistake, because it never asks. It retries
unconditionally, and even an uninformed retry is a fresh {P_FIRST:.0%} draw, which
compounds across attempts into {modes['blind retry']:.1%}.

**A feedback signal that gates a retry has to be better than the retry it
suppresses**, and that is a much higher bar than "better than nothing"
(eq:feedback-must-be-external).

The second table shows the same bar applying to the environment, which is the part
that transfers to real systems. Environment feedback falls from
{q_tab[1.0][0]:.1%} at perfect quality to {q_tab[0.25][0]:.1%} at {0.25:.0%}, while
blind retry sits flat at about {q_tab[0.25][2]:.1%} because it does not depend on
quality at all.

They cross. Somewhere between {0.45:.0%} and {0.25:.0%} feedback quality,
**consulting the environment becomes worse than ignoring it** -- not because the
environment is lying, but because acting on a weak signal costs the unconditional
retries that a policy with no signal would have taken.

That is a testable property of your own tools. A test suite that reports precisely
is far above the crossover; a tool that returns "error" and a system that treats
that as a verdict on whether to continue may be below it.

The third table locates where self-assessment does its most damage, and the shape
is not monotone. The gap between self-assessment and no retry is
{comp[0.25][1] - comp[0.25][0]:+.1%} at {0.25:.0%} base competence,
{comp[0.55][1] - comp[0.55][0]:+.1%} at {0.55:.0%}, and
{comp[0.85][1] - comp[0.85][0]:+.1%} at {0.85:.0%}.

Self-assessment helps most in the middle and least at both ends, for two different
reasons. A weak agent cannot tell that it failed, so it rarely retries. A strong
agent rarely needs to. Meanwhile the environment column is
{comp[0.25][2]:.1%} even at {0.25:.0%} competence -- **external feedback rescues a
weak agent and self-assessment cannot**, which is the practically important half of
cite:huang2024selfcorrect's distinction.

The fourth table says how many attempts are worth buying under each regime, and the
curves separate rather than converge. At {8} attempts blind reaches
{at[8][0]:.1%}, self-assessment {at[8][1]:.1%}, environment {at[8][2]:.1%}. More
attempts help every regime and they do not close the gap, because the gap is a
per-attempt conditioning difference and attempts multiply it.

So the design conclusions, in order of how much they contradict common practice.

**Do not gate a retry on the agent's own judgement.** If you can afford the retry,
take it unconditionally. Blind beat self-assessed here by
{modes['blind retry'] - modes['self-assessment']:.1%}, and the only thing
self-assessment added was an opportunity to skip.

**Use the agent's judgement for DIAGNOSIS, never for DETECTION.** Detection is what
gates, and it is the correlated part. Once you have decided to retry anyway,
whatever the agent thinks went wrong is free information that can only condition
the next attempt.

**Measure your environment's feedback quality**, because the second table says it
has a threshold below which acting on it is negative, and nobody checks.

One honest boundary. This models a retry as costless. It is not: ch:ag-loop's step
budget is finite, and an unconditional-retry policy spends it on attempts that had
already succeeded. The real recommendation is therefore narrower than "always
retry" -- it is that the decision should be made by a budget policy rather than by
the agent's opinion of its own work, which is ch:ag-termination's subject.""")
```

The second listing holds the feedback source fixed and varies what it tells you.

```python {tier=A name=localise-before-diagnose}
"""Detection, localisation, diagnosis: three different things a failure can tell you.

"The agent noticed it failed and tried again" collapses three separate pieces of
information into one sentence, and they are worth very different amounts
(eq:localise-before-diagnose).

  DETECTION    -- something went wrong
  LOCALISATION -- it went wrong at step j
  DIAGNOSIS    -- it went wrong at step j BECAUSE of x

Detection alone forces a restart from the beginning, which throws away every step
that was fine. Localisation lets the retry resume from the last good state, which
is exactly ch:ag-planning's checkpoint. Diagnosis conditions the retry, which is
ch:ag-tool-calling's error message.

This listing gives an agent each in turn and measures what each one buys, under a
fixed step budget.
"""
import numpy as np

rng = np.random.default_rng(2357)

N = 40000
K = 8                   # steps in the task
P_STEP = 0.88
BUDGET = 26
P_INFORMED = 0.97       # a retry that knows the cause


def run(mode, budget=BUDGET, k=K, p_step=P_STEP, loc_acc=1.0):
    """mode: 'none' | 'detect' | 'localise' | 'diagnose'.
    loc_acc is how often localisation identifies the right step."""
    done = np.zeros(N, dtype=bool)
    spent = np.zeros(N, dtype=np.int64)
    resume = np.zeros(N, dtype=np.int64)     # step to restart from
    for _ in range(budget):
        live = ~done & (spent < budget)
        idx = np.flatnonzero(live)
        if not len(idx):
            break
        start = resume[idx]
        # Run from `start` to the end, or until a step fails.
        fail_at = np.full(len(idx), k)
        for j in range(k):
            active = (start <= j) & (fail_at == k)
            if not active.any():
                break
            p = np.where((mode == "diagnose") & (resume[idx] == j),
                         P_INFORMED, p_step)
            bad = active & (rng.random(len(idx)) >= p)
            fail_at = np.where(bad, j, fail_at)
        spent[idx] += (k - start).clip(min=1)
        won = fail_at == k
        done[idx[won]] = True
        lost = ~won
        if mode == "none" or mode == "detect":
            resume[idx[lost]] = 0                       # restart from scratch
        else:
            guessed = fail_at
            if loc_acc < 1.0:
                miss = rng.random(len(idx)) >= loc_acc
                guessed = np.where(miss, 0, fail_at)    # a bad guess restarts
            resume[idx[lost]] = guessed[lost]
        if mode == "none":
            break                                       # a single attempt
    return float(done.mean()), float(spent.mean())


print(f"A {K}-step task at {P_STEP:.0%} per step, budget {BUDGET} steps. A retry")
print(f"that knows the cause succeeds at {P_INFORMED:.0%} on the failed step.")
print()
print(f"{'what the failure tells you':>32}{'completed':>12}{'steps':>9}"
      f"{'per step':>11}")
print("-" * 64)
tab = {}
for name, mode in [("nothing (one attempt)", "none"),
                   ("detection: restart", "detect"),
                   ("localisation: resume at j", "localise"),
                   ("diagnosis: resume, informed", "diagnose")]:
    s, c = run(mode)
    tab[name] = (s, c)
    print(f"{name:>32}{s:>12.1%}{c:>9.1f}{s / max(c, 1e-9):>11.3f}")

print()
print()
print("Localisation is a guess. Sweep how often it names the right step;")
print("a wrong guess sends the agent back to the beginning.")
print()
print(f"{'localisation accuracy':>23}{'localise':>11}{'diagnose':>11}"
      f"{'detect only':>14}")
print("-" * 59)
la = {}
det = run("detect")[0]
for a in (1.0, 0.85, 0.65, 0.45, 0.20):
    x = run("localise", loc_acc=a)[0]
    y = run("diagnose", loc_acc=a)[0]
    la[a] = (x, y)
    print(f"{a:>23.0%}{x:>11.1%}{y:>11.1%}{det:>14.1%}")

print()
print()
print("How the three scale with the budget -- the resource localisation saves.")
print()
print(f"{'budget':>8}{'detect':>10}{'localise':>11}{'diagnose':>11}")
print("-" * 40)
bd = {}
for b in (10, 16, 26, 40, 60):
    bd[b] = (run("detect", budget=b)[0], run("localise", budget=b)[0],
             run("diagnose", budget=b)[0])
    print(f"{b:>8}{bd[b][0]:>10.1%}{bd[b][1]:>11.1%}{bd[b][2]:>11.1%}")

print()
print()
print("And how it scales with task length, at a budget of 3x the task.")
print()
print(f"{'steps k':>9}{'detect':>10}{'localise':>11}{'gain':>9}")
print("-" * 39)
kl = {}
for k in (4, 8, 14, 22):
    a = run("detect", budget=3 * k, k=k)[0]
    b = run("localise", budget=3 * k, k=k)[0]
    kl[k] = (a, b)
    print(f"{k:>9}{a:>10.1%}{b:>11.1%}{b - a:>+9.1%}")

print(f"""
The first table is the ordering, and the size of the steps between rows is the
result.

Detection alone -- knowing that something went wrong and starting over -- takes
completion from {tab['nothing (one attempt)'][0]:.1%} to
{tab['detection: restart'][0]:.1%}. That is a large gain and it is the one most
systems have.

Localisation takes it to {tab['localisation: resume at j'][0]:.1%}, and it does so
while spending FEWER steps: {tab['localisation: resume at j'][1]:.1f} against
{tab['detection: restart'][1]:.1f}. Better outcome, cheaper.

Diagnosis -- additionally knowing why -- adds
{tab['diagnosis: resume, informed'][0] - tab['localisation: resume at j'][0]:+.1%}.

**Knowing WHERE is worth roughly thirteen times knowing WHY**, at these
parameters, and that ordering is the opposite of where engineering attention
usually goes. Teams write elaborate error taxonomies and reflection prompts to
extract the cause, on top of a retry that starts from the beginning.

The reason is arithmetic rather than psychology. A restart re-runs every step that
was already fine, so it pays the whole task's failure probability again -- and
those steps fail at the same rate they did the first time. Resuming from the last
good state pays only the remainder. Diagnosis improves ONE step's odds; localisation
removes all the steps before it from the retry.

The second table is the objection: localisation is a guess, and a wrong guess sends
you back to the start. So how good does the guess have to be?

At {1.0:.0%} accuracy localisation scores {la[1.0][0]:.1%}. At {0.2:.0%} -- a guess
that is wrong four times in five -- it still scores {la[0.2][0]:.1%}, against
detection-only's {det:.1%}.

**Localisation beats restarting even when it is usually wrong**, because a wrong
guess costs exactly what detection-only always costs, and a right guess saves a
great deal. The downside is bounded by the alternative and the upside is not, which
makes it a free option in the strict sense.

That is a strong practical statement: you do not need a reliable failure localiser
to benefit from having one, so the usual reason for not building it -- "we could
not identify the failing step reliably" -- is not a reason.

The third table shows what localisation is actually buying, which is budget. At a
budget of {16} steps -- twice the task length -- detection reaches
{bd[16][0]:.1%} and localisation {bd[16][1]:.1%}. Detection needs a budget of
{60} to reach {bd[60][0]:.1%}, which localisation reaches at {16}.

**Localisation converts a budget problem into a non-problem**, and budget is the
scarce resource in every chapter of this part. Restarting a partially-completed task
is the single largest source of wasted steps an agent has.

The fourth table is why this grows in importance rather than staying constant. The
gain from localisation is {kl[4][1] - kl[4][0]:+.1%} on a {4}-step task and
{kl[22][1] - kl[22][0]:+.1%} on a {22}-step one, where detection-only completes
{kl[22][0]:.1%} and localisation {kl[22][1]:.1%}.

The mechanism is the same exponent as everywhere else: a restart re-runs $k$ steps
and resumption re-runs the remainder, so the gap widens with $k$. **On long tasks,
the difference between restarting and resuming is most of the outcome.**

Which connects this chapter to two others and makes the recommendation concrete.

Localisation is what ch:ag-planning's checkpoints provide. A verified segment
boundary IS a known-good state to resume from, so a system with checkpoints has
localisation for free and a system without one cannot have it at any price. That is
a second, independent argument for checkpoints, arrived at from recovery rather
than from the success exponent.

Diagnosis is what ch:ag-tool-calling's error messages provide, and this listing
puts it in its place: worth having, worth much less than localisation, and worth
almost nothing if the retry restarts from the beginning anyway.

So the build order is: resume before you diagnose. Get a known-good state to return
to, and only then invest in working out what went wrong.""")
```

## 9. Practical Example

The first listing gives an agent a $42\%$ first-attempt success rate and up to four
attempts.

```
                     recovery mode  final success  vs no retry
--------------------------------------------------------------
                          no retry          41.8%        -0.7%
                       blind retry          88.8%       +46.3%
                   self-assessment          73.9%       +31.4%
              environment feedback          99.1%       +56.7%
```

Environment feedback wins and self-assessment trails it — both expected from
{{cite:huang2024selfcorrect}}. The surprise is the middle row: **blind retry beats
self-assessment by $14.9$ points.** An agent that knows nothing about its failure
outperforms one that tries to work out what happened.

The mechanism is gating ({{eq:gating-costs-a-retry}}). Self-assessment supplies the
*decision to retry* as well as the diagnosis, and on the attempts the agent botched
its judgement is the unreliable one. A failed self-assessment throws away a retry
that would have been free.

The same bar applies to the environment:

```
  feedback quality   environment   self-assessment   blind retry
----------------------------------------------------------------
              100%         99.1%             74.0%         88.8%
               65%         92.3%             64.7%         88.7%
               45%         83.2%             58.3%         88.6%
               25%         68.7%             51.4%         88.9%
```

They cross between $45\%$ and $25\%$: **consulting the environment becomes worse
than ignoring it**, not because it lies but because acting on a weak signal costs
the unconditional retries a signal-free policy would have taken
({{eq:feedback-quality-threshold}}). A test suite that reports precisely is far
above the crossover; a tool that returns `error`, treated as a verdict on whether to
continue, may be below it.

Where self-assessment does most damage:

```
  first-attempt   no retry      self   environment   self gap
-------------------------------------------------------------
            25%      25.1%     41.7%         96.5%     +16.6%
            55%      54.9%     85.1%         97.8%     +30.2%
            85%      85.1%     98.6%         99.2%     +13.6%
```

Its benefit peaks in the middle — a weak agent cannot tell it failed, a strong one
rarely needs to. The environment column is above $96\%$ at every competence:
**external feedback rescues a weak agent and self-assessment cannot.**

The second listing holds the source fixed and varies what the signal contains:

```
      what the failure tells you   completed    steps   per step
----------------------------------------------------------------
           nothing (one attempt)       35.9%      8.0      0.045
              detection: restart       83.1%     18.5      0.045
       localisation: resume at j       98.6%     12.8      0.077
     diagnosis: resume, informed       99.8%     11.7      0.085
```

Detection with a restart buys $+47.2$ points. Localisation buys another $+15.5$
*while spending $5.7$ fewer steps* — better outcome, cheaper. Diagnosis adds
$+1.2$.

**Knowing where is worth about thirteen times knowing why.** The reason is
{{eq:localise-before-diagnose}}: a restart re-runs every step that was fine and pays
their failure probability again, while diagnosis improves one step's odds.

And localisation does not need to be reliable:

```
  localisation accuracy   localise   diagnose   detect only
-----------------------------------------------------------
                   100%      98.6%      99.8%         83.1%
                    45%      90.1%      93.4%         83.1%
                    20%      86.3%      89.9%         83.1%
```

At $20\%$ accuracy — wrong four times in five — it still beats restarting by $3.2$
points, because a wrong guess costs exactly what restarting always costs
({{eq:localisation-is-a-free-option}}). **"We could not identify the failing step
reliably" is not a reason not to try.**

What localisation actually buys is budget:

```
  budget    detect   localise   diagnose
----------------------------------------
      10     59.4%      70.8%      77.7%
      16     58.8%      92.3%      98.1%
      60     97.2%     100.0%     100.0%
```

Detection needs a budget of $60$ to reach what localisation reaches at $16$.

And it grows with task length:

```
  steps k    detect   localise     gain
---------------------------------------
        4     93.6%      99.5%    +5.8%
       14     41.8%      98.7%   +57.0%
       22     17.0%      98.7%   +81.7%
```

$+5.8$ points at four steps, $+81.7$ at twenty-two
({{eq:recovery-scales-with-length}}). **On long tasks the difference between
restarting and resuming is most of the outcome.**

## 10. Production Considerations

Never gate a retry on the agent's own judgement. Route it to advice — conditioning a
retry you were taking anyway — and let a budget policy decide whether to retry.

Measure your environment's feedback quality and check which side of
{{eq:feedback-quality-threshold}} you are on. Two numbers from logs: how often the
detector fires on genuine failures, and how often signal-guided retries succeed.

Build localisation before diagnosis. It is worth an order of magnitude more, it
saves steps rather than spending them, and it needs no accuracy to be worth having.

Get localisation from checkpoints where you have them
({{ch:ag-planning}}). A verified boundary is a known-good resume point, and the two
chapters' arguments are the same mechanism.

Express budgets in steps, not attempts, so that a resuming policy is rewarded and a
restarting one is charged.

Keep reflection, and lower its quality bar. Its measured value is that it changes
the context after a failure ({{eq:context-change-breaks-loops}}); varied beats
correct.

Treat tool error text as untrusted input. The recovery path conditions on it and is
rarely audited ({{cite:greshake2023indirect}}).

## 11. Common Mistakes

**Letting the agent decide whether to retry.** Blind retry beat self-assessment by
$14.9$ points precisely because it never asks.

**Assuming any feedback is better than none.** It has a threshold
({{eq:feedback-quality-threshold}}), and weak signals that gate are negative.

**Investing in diagnosis before localisation.** Thirteen to one the other way, and
localisation also reduces cost.

**Requiring a reliable localiser before building one.** It is a free option at any
accuracy ({{eq:localisation-is-a-free-option}}).

**Restarting from the beginning.** It re-runs the steps that worked and pays their
failure probability again — the largest source of wasted steps an agent has.

**Budgeting in attempts.** It hides the cost difference between resuming and
restarting.

**Retrying without deduplication.** Blind retry's advantage assumes retries are
fresh draws; without {{ch:ag-loop}}'s dedupe they are repeats.

## 12. Failure Modes

*Silent early exit.* Self-assessment declares success, the run ends, and a partial
result is returned confidently — {{ch:ag-loop}}'s false stop reached through the
recovery path.

*Retry storms.* Detection fires, the retry restarts, it fails at the same step, and
the budget drains. The signature is a flat failure position across attempts.

*Corrupted resume point.* Localisation names a step *after* the real failure, so the
retry resumes from a bad state and every subsequent attempt inherits it. This is
{{ch:ag-planning}}'s approved-corruption failure, and it is the one case where a
localiser is worse than restarting.

*Injected recovery.* Error text influenced by an attacker steers the retry
({{cite:greshake2023indirect}}).

*Reflection that repeats.* A reflection that says the same thing every attempt does
not change the context and therefore does not break the cycle, which is the failure
{{eq:context-change-breaks-loops}} predicts.

## 13. Alternatives

**No recovery, more samples.** {{ch:rsn-test-time-compute}}: run the whole task $n$
times independently and select. Simpler, parallelises, and wastes the prefix every
time — which is exactly the cost localisation removes.

**Checkpoint-driven resume.** {{ch:ag-planning}}: get localisation structurally
rather than by inference. Strictly better where the task admits verified boundaries.

**Human escalation.** Convert a failure into a delay rather than a retry.
{{ch:ag-termination}} prices it, and it is the right answer when
{{eq:feedback-quality-threshold}} says your signals are below threshold.

**Transactional rollback.** Where the environment supports undo, a failed segment
can be reversed rather than resumed-around, which removes the corrupted-resume
failure mode entirely.

**Better first attempts.** Raising $p$ helps every row of every table here, and
{{ch:ag-tool-calling}}'s interventions are usually cheaper than a recovery system.

## 14. Evaluation

Measure detector sensitivity and the success rate of signal-guided retries
separately. Those two numbers evaluate {{eq:feedback-quality-threshold}} and decide
whether your feedback should gate or advise.

Report the failure *position* distribution, not just the failure rate. It is what
tells you how much prefix a restart is wasting, and it is the input to
{{eq:localise-before-diagnose}}.

Measure localiser accuracy — but do not gate adoption on it, since
{{eq:localisation-is-a-free-option}} says any accuracy is positive.

Report steps consumed alongside completion. Localisation improved both in
{{sec:9-practical-example}}, and a completion-only metric hides that.

And evaluate at your real task length. The localisation gain went from $+5.8$ to
$+81.7$ points between four and twenty-two steps.

## 15. Advanced Concepts

**Learned localisation.** Predicting which step failed from a trace is a supervised
problem with free labels — every run that later succeeded from a resume point
confirms one. Because {{eq:localisation-is-a-free-option}} makes any accuracy
positive, this can be deployed before it is good. {{maturity:EMERGING}}.

**Feedback routing.** Automatically deciding whether a given signal should gate or
advise, based on its measured position relative to
{{eq:feedback-quality-threshold}}. This is a small policy layer that no framework
currently exposes.

**Reflection diversity as the objective.** If reflection's value is context change
rather than insight ({{sec:7-internal-mechanics}}), then optimising reflections for
*variety* rather than accuracy is the right target — and it is measurable as the
distance between successive reflections.

**Partial rollback.** {{sec:12-failure-modes}}'s corrupted-resume failure disappears
if the environment can undo. Characterising which agent domains admit it, and
designing tools so that they do, is {{maturity:RESEARCH FRONTIER}} and would remove
the only case where localisation loses.

## 16. Connection to Previous Chapters

{{ch:rsn-self-consistency}}'s correlated critic is why the self-assessment column
underperforms, and {{cite:huang2024selfcorrect}}'s "can make things worse" gets a
mechanism here: it is harmful when it gates and merely useless when it advises.

{{ch:ag-planning}}'s checkpoints and this chapter's localisation are the same
quantity — the wasted prefix $k-j$ — approached from the success exponent and from
recovery cost respectively.

{{ch:ag-tool-calling}}'s error message is the diagnosis term, and this chapter puts
it in its place: real, and worth an order of magnitude less than resuming.

{{ch:ag-loop}}'s deduplication is what makes blind retry a fresh draw rather than a
repeat, and its budget is the resource localisation conserves.

Ahead: {{ch:ag-termination}} decides who owns the retry decision this chapter took
away from the agent; {{ch:ag-security}} takes up error text as an injection surface.

## 17. Exercises

1. Derive {{eq:feedback-quality-threshold}} and compute the crossover feedback
   quality for the listing's constants. Check it against the measured $45$–$25\%$
   band.

2. Make the self-critic's detection *anti*-correlated with competence — best where
   the agent did worst — and show how much of the gap to environment feedback
   closes.

3. Add the corrupted-resume failure to the second listing: localisation sometimes
   names a step after the true failure. At what rate does localisation stop beating
   restarting?

4. Model reflection as changing the context by a measurable amount and show that
   its benefit tracks the change rather than the correctness.

5. Combine both listings: an agent with checkpoints, environment feedback of
   measured quality, and a budget. Which single upgrade is worth most?

6. Take a real agent trace set and plot the failure-position distribution. How much
   prefix is your current retry policy wasting?

## 18. Interview Questions

1. Why can an agent that checks its own work do worse than one that does not?

2. When is acting on tool feedback worse than ignoring it?

3. Rank detection, localisation and diagnosis by value, and justify the ordering.

4. Why should you build a failure localiser even if it is usually wrong?

5. Your retry budget is in attempts. What is wrong with that?

6. What is reflection actually buying, if not insight?

## 19. Research Questions

1. Can a failure localiser be trained from free labels (successful resumes) and how
   accurate does it get before the corrupted-resume mode starts to bind?

2. Is the gate-versus-advise routing decision automatable from online measurement of
   {{eq:feedback-quality-threshold}}?

3. Does optimising reflections for diversity rather than accuracy improve recovery,
   and by how much?

4. Which agent environments admit partial rollback, and can tool interfaces be
   designed to provide it as a contract?

5. How correlated is a self-critic's detection with instance difficulty in practice,
   and does that correlation weaken for tasks where recognition is easier than
   generation?

## 20. Chapter Summary

An agent has an environment, which is the external feedback
{{cite:huang2024selfcorrect}} says self-correction lacks. This chapter measures how
much of that escape survives imperfect feedback, and finds two things.

**Blind retry beat self-assessment**, $88.8\%$ against $73.9\%$. The mechanism is
gating ({{eq:gating-costs-a-retry}}): self-assessment supplies the decision to retry,
and on the attempts the agent botched its judgement is the unreliable one, so a
failed assessment discards a free retry. The same bar applies to the environment —
below about $35\%$ feedback quality, acting on it became worse than ignoring it
({{eq:feedback-quality-threshold}}). **A signal that gates must beat the retry it
suppresses**, which is a far higher bar than beating nothing. Route weak signals to
advice and strong ones to gates.

Self-assessment's benefit peaked at middling competence ($+30.2$ points at $55\%$)
and external feedback stayed above $96\%$ at every competence: **external feedback
rescues a weak agent and self-assessment cannot.**

The second half separates what a failure tells you. Detection with a restart bought
$+47.2$ points; localisation bought another $+15.5$ *while spending $5.7$ fewer
steps*; diagnosis added $+1.2$. **Knowing where is worth about thirteen times
knowing why** ({{eq:localise-before-diagnose}}), because a restart re-runs every
step that was fine while diagnosis improves one.

Localisation is a free option: at $20\%$ accuracy it still beat restarting, since a
wrong guess costs exactly the baseline ({{eq:localisation-is-a-free-option}}). And
its value grows with task length — $+5.8$ points at four steps, $+81.7$ at
twenty-two ({{eq:recovery-scales-with-length}}).

So the build order is: resume before you diagnose, get the resume point from
checkpoints, budget in steps, and keep the agent's opinion of its own work away from
the gate.

## 21. Further Reading

{{cite:huang2024selfcorrect}} is the paper this chapter measures, and worth
re-reading with {{eq:feedback-quality-threshold}} in hand — "can make things worse"
has a mechanism.

{{cite:shinn2023reflexion}} for the reflection loop, read with
{{sec:7-internal-mechanics}}'s reframing: its value is context change rather than
insight, which lowers the quality bar considerably.

{{cite:madaan2023selfrefine}} for the positive case, and
{{ch:rsn-self-consistency}} for when recognition really is easier than generation.

{{cite:liu2024agentbench}} and {{cite:zhou2024webarena}} for the long-horizon
environments where {{eq:recovery-scales-with-length}} does the most work.
