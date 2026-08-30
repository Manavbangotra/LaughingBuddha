---
id: ev-agents
number: 218
part: XXV
tier: full
status: draft
requires: [reference-scoring-penalises-valid-answers, cause-distance-drives-triage-cost,
           most-rag-failures-are-invisible-end-to-end, optimising-against-a-judge-diverges]
provides: [pass-k-separates-capability-from-reliability, retry-does-not-help-a-deterministic-failure,
           outcome-evaluation-credits-lucky-trajectories, trajectory-matching-inherits-the-answer-space-problem]
citations: [yao2024taubench, jimenez2023swebench, cemri2025mast, deshpande2025trail]
---

## 1. Learning Objectives

By the end of this chapter you will be able to reconcile a single-run success rate with a
pass^k figure and infer the correlation structure between attempts; show that two
improvements with identical effects on pass^1 can differ by an order of magnitude on pass^8;
allocate a retry budget conditionally and explain why a fast failure is a better retry signal
than a slow one; separate outcome correctness from trajectory soundness and quantify how much
outcome evaluation over-credits; explain why trajectory matching inherits and worsens the
reference-sampling problem; and assemble a reference-free invariant suite that catches the
failures outcome scoring cannot see.

## 2. Why This Matters

{{cite:yao2024taubench}} reported state-of-the-art function-calling agents succeeding on
under 50% of tasks and — the number that matters more — **pass^8 below 25%** in the retail
domain: the probability that all eight attempts at the same task succeed.

Under independence those numbers cannot both be true. In the population modelled here,
single-run success is **58%** and pass^8 is **27%**, where independence would give **1.21%**
({{eq:pass-k-separates-capability-from-reliability}}). **The gap is correlation between
attempts**: some tasks work every time and some never do, and the single-run rate averages
them into a number that describes neither.

That distinction reorders development. Two improvements that both move pass^1 by **+0.035**
move pass^8 by **+0.005** and **+0.055** — **11×** apart on the metric users experience, and
tied on the metric anyone would report.

It also changes what a retry buys. Five attempts take success from **0.576 to 0.801** and
attempts per success from **1.74 to 2.75**, with `out of reach` tasks consuming **32.7%** of
attempts to produce **1.8%** of successes ({{eq:retry-does-not-help-a-deterministic-failure}}).

The second half is what "succeed" should mean. **24% of outcome-evaluation passes are correct
for a reason that will not recur** ({{eq:outcome-evaluation-credits-lucky-trajectories}}), and
that is enough to invert a comparison between two agents. The obvious fix — grade the
trajectory — fails harder: debugging a repository has on the order of **2,600** valid
trajectories, so exact matching marks **99.96%** of correct routes wrong
({{eq:trajectory-matching-inherits-the-answer-space-problem}}).

## 3. Prerequisites

{{eq:reference-scoring-penalises-valid-answers}} from {{ch:ev-why-hard}} is the result that
kills trajectory matching before it starts. A trajectory is a sequence of choices, so the
acceptable-trajectory space multiplies at every step.

{{eq:cause-distance-drives-triage-cost}} from {{ch:ops-agent-tracing}} is the operational
companion: the cause sits several steps back from the visible failure, which is why an
outcome-level pass or fail carries almost no diagnostic information.

{{eq:most-rag-failures-are-invisible-end-to-end}} from {{ch:ev-rag}} is the same instrument
argument one system up — a single end-to-end verdict cannot distinguish failure modes that
need different fixes.

{{eq:optimising-against-a-judge-diverges}} from {{ch:ev-llm-judge}} governs the step-level
judge in {{sec:9-practical-example}}'s instrument table, and it is why the invariant checks
are preferable where they apply.

{{cite:cemri2025mast}}'s multi-agent failure taxonomy and {{cite:deshpande2025trail}}'s
localisation benchmark are the empirical backdrop for both halves.

## 4. Intuitive Explanation

Agent evaluation has two problems that single-turn evaluation does not, and they are usually
conflated.

The first is that **an agent's success rate is not a rate.** Run a language model on a
classification task a thousand times and the errors are roughly independent — a hard item is
hard, but the randomness is per-sample. Run an agent on the same task eight times and the
outcomes are strongly correlated: if it can do the task, it usually does it every time; if it
cannot, it usually fails every time.

{{cite:yao2024taubench}} measured exactly this. Under 50% single-run success, and pass^8 —
all eight attempts succeeding — below 25%. If attempts were independent, a 50% agent would
have a pass^8 of 0.4%. Observing 25% means most of the successes come from tasks that work
reliably, and most of the failures from tasks that never work.

Which means the single-run number is a mixture and reports itself as a capability. Two agents
with identical 58% single-run rates can have completely different populations behind them,
and users will experience them completely differently.

pass^k is the instrument that separates them. Raise k and you progressively filter out
everything except the tasks the agent does consistently. In the numbers here, the reliably
solvable class supplies 52% of pass^1 successes and 90% of pass^8 successes. **pass^8 is
asking "what can this agent be depended on for," which is the question a product needs
answered.**

The development consequence is sharp. Consider two improvements. Improvement A makes the
coin-flip tasks a bit more likely to work — better prompting, a slightly stronger model.
Improvement B converts some coin-flip tasks into reliable ones — better tool schemas, a
retry-with-verification step, a state check that catches a specific recurring failure.

Both move pass^1 by exactly +0.035. On pass^8, A moves +0.005 and B moves +0.055.

**B is worth eleven times A on the metric that describes reliability, and ties it on the
metric anyone would report.** A team optimising the headline number is indifferent between
them.

Now retries, which is how products cope with unreliability.

Allowing five attempts takes success from 0.576 to 0.801. That looks like a good trade until
you look at what it costs: attempts per success rises from 1.74 to 2.75, and — more
importantly — where those attempts go. The `out of reach` class consumes 32.7% of all
attempts and produces 1.8% of successes. **Retrying a task the agent cannot do burns five
attempts to arrive back where it started**, and the user waits through all five.

The fix is a conditional retry, and there is a nice result about how to condition it. You do
not need to identify the recoverable tasks. You need to *stop* on the unrecoverable ones, and
a long failing trajectory is a usable signal: an agent that fails after thirty steps has
usually exhausted a real approach, while an agent that fails after three has often hit
something transient. Retrying only after a fast failure reaches 0.735 for 1.29 attempts —
41% fewer attempts for 0.066 less success, at essentially the same attempts-per-success as
never retrying at all.

**A cheap negative filter beats an expensive positive one.**

That is the reliability half. The other half is what "succeed" should mean.

Take an agent that books a flight. It books the right flight. Outcome evaluation says pass.

Except it called the booking API twice and the second call failed with a duplicate error,
which is why there is only one booking. Or it read the customer's payment history when it did
not need to. Or it guessed a date format that happened to work on this endpoint and will not
work on the next one.

The outcome is correct. The trajectory is unsound. It passed, and it will not pass on a
neighbouring task.

Splitting outcome from trajectory gives four cells. Correct and sound is the good case, 44%
here. Correct and unsound is 14% — a quarter of everything outcome evaluation passes.
Incorrect and sound is 11%: the agent did everything right and the task could not be done, and
outcome evaluation scores that as a failure. Incorrect and unsound is the rest.

So the outcome score is wrong in both directions at once — over-crediting luck and penalising
sound attempts at impossible tasks — and the errors do not cancel, because they are different
tasks.

That matters most in comparisons. Agent P has 0.44 sound passes and 0.14 lucky ones; agent Q
has 0.40 and 0.21. Outcome score: Q wins, 0.61 to 0.58. Dependable passes: P wins, 0.44 to
0.40. **The agent that gets luckier ranks higher**, and luck does not transfer.

The obvious response is: evaluate the trajectory, not just the outcome. Compare what the
agent did against what it should have done.

This fails, and it fails worse than the single-answer version. {{ch:ev-why-hard}} showed a
single-reference metric marks 99.4% of correct summaries wrong because there are 180
acceptable summaries. A trajectory is a *sequence* of choices, so the space multiplies at
every step: on the order of 2,600 valid routes for debugging a repository. Exact trajectory
matching credits 0.04% of correct routes.

Trajectory matching is the worst instrument in this chapter and it is what "evaluate the
reasoning, not just the answer" becomes when implemented literally.

The way out is the same one {{ch:ev-why-hard}} found for answers: **state a predicate instead
of writing a reference.** Do not ask whether the trajectory matched. Ask whether it violated
anything.

Were all tool arguments valid? Was any write repeated with the same effect? Did any read leave
the authorised scope? Is every claim in the answer traceable to a tool result? Does the
terminal state match what was requested? Does any step contradict an earlier result?

Every one of those is checkable on the trajectory alone, with no reference. Between them they
take the lucky-pass share from 0.14 to 0.033, and **the reported score falls from 0.58 to
0.47** — which is the right direction. An evaluation that gets stricter and lower is one that
started too high.

And the economics are good. Test-graded outcomes — {{cite:jimenez2023swebench}}'s design —
plus invariant checks plus a side-effect diff cost 89 per thousand trajectories against 3,400
for human outcome checking alone, and they measure three things instead of one.

## 5. Formal Explanation

**pass^k under heterogeneity.** Let tasks be drawn from classes $c$ with shares $w_c$ and
per-attempt success probabilities $p_c$, with attempts independent *within* a task. Then

$$\text{pass}^k = \sum_c w_c\, p_c^{\,k}.$$

Since $x \mapsto x^k$ is convex, $\text{pass}^k \ge (\text{pass}^1)^k$ by Jensen, with
equality only if all $p_c$ are equal. The observed excess of $\text{pass}^k$ over
$(\text{pass}^1)^k$ is therefore a direct measure of heterogeneity — of how much the
population separates into reliable and unreliable tasks.

The contribution of class $c$ to $\text{pass}^k$ is $w_c p_c^k / \text{pass}^k$, which is
increasing in $p_c$ and increasingly so with $k$: raising $k$ reweights the population toward
its reliable tail.

**Two kinds of improvement.** Raising $p_c$ for a class contributes $w_c k p_c^{k-1} \delta$
to $\text{pass}^k$, which vanishes for small $p_c$ and large $k$. Moving mass $m$ from class
$c$ to class $c'$ contributes $m(p_{c'}^k - p_c^k)$, which is large when $p_{c'}$ is near one.
Hence **improving a marginal class helps pass^1; promoting a class helps pass^k**, and the
two are indistinguishable at $k=1$.

**Retries.** With $r$ attempts, success is $\sum_c w_c[1 - (1-p_c)^r]$ and expected attempts
is $\sum_c w_c \sum_{j=1}^{r}(1-p_c)^{j-1}$. The marginal attempt on class $c$ costs
$(1-p_c)^{r-1}$ and yields $p_c(1-p_c)^{r-1}$, so the *yield per attempt* is exactly $p_c$ —
independent of $r$. A retry budget allocated without regard to $p_c$ therefore spends in
proportion to failure probability and gains in proportion to success probability, which is
maximally misaligned.

**Outcome versus trajectory.** Let $O$ be outcome correctness and $T$ trajectory soundness.
The outcome score is $\Pr[O]$; the dependable share is $\Pr[O \cap T]$. The over-credit is
$\Pr[O \cap \bar T]$ and the under-credit is $\Pr[\bar O \cap T]$; these are different task
populations and do not offset. Ranking two agents by $\Pr[O]$ rather than $\Pr[O \cap T]$
inverts whenever $\Pr[O\cap\bar T]$ differs by more than $\Pr[O \cap T]$ does.

**Trajectory matching.** For a task with $|A_\tau|$ acceptable trajectories and $r$
references, exact-match credit is $\min(1, r/|A_\tau|)$. Since a trajectory of length $n$ with
$b$ acceptable choices per step has $|A_\tau| \sim b^n$, the credit falls exponentially in
trajectory length.

## 6. Mathematical Foundation

pass^k as a probe of heterogeneity:

$$\text{pass}^k = \sum_c w_c p_c^{\,k} \;\ge\; \left(\text{pass}^1\right)^k, \qquad \frac{\partial}{\partial k}\left[\frac{w_c p_c^k}{\text{pass}^k}\right] > 0 \iff p_c > \bar p_k$$ (eq:pass-k-separates-capability-from-reliability)

At $\text{pass}^1 = 0.58$: independence predicts $\text{pass}^8 = 1.21\%$; the observed
$27\%$ is the heterogeneity.

The misallocation of an unconditional retry budget:

$$\frac{\text{attempts on } c}{\text{successes from } c} = \frac{1}{p_c}, \qquad \text{out of reach: } 32.7\% \text{ of attempts}, \; 1.8\% \text{ of successes}$$ (eq:retry-does-not-help-a-deterministic-failure)

Outcome scoring's two-sided error:

$$\Pr[O] - \Pr[O \cap T] = \Pr[O \cap \bar T] = 0.14, \qquad \frac{0.14}{0.58} = 24\%$$ (eq:outcome-evaluation-credits-lucky-trajectories)

with an under-credit of $\Pr[\bar O \cap T] = 0.11$ on a different task population.

And why the obvious fix is worse:

$$|A_\tau| \sim b^{\,n}, \qquad \text{credit} = \min\!\left(1, \frac{r}{|A_\tau|}\right) = 0.0004 \ \text{at}\ |A_\tau| = 2600$$ (eq:trajectory-matching-inherits-the-answer-space-problem)

## 7. Internal Mechanics

Why are agent attempts correlated? Because the sources of failure are mostly not stochastic.
A tool whose schema the agent misreads will be misread every time. A task requiring a
capability the model lacks will fail every time. A policy the agent does not know about will
be violated every time. The genuinely random component — sampling temperature, a race in the
environment, a transient API error — is a minority of the failure mass, and it is exactly the
minority that retries address.

That explains the shape of the retry result. Retries work on the stochastic component and the
stochastic component is small, so retries buy less than the naive arithmetic suggests and cost
more than anyone budgets.

It also explains why the fast-failure heuristic works. A trajectory that fails in three steps
has usually hit something environmental — a timeout, a malformed response, a transient error.
A trajectory that fails after thirty has usually exhausted a genuine approach, which means the
agent's model of the task is wrong and a fresh sample will build the same wrong model. **The
length of the failing trajectory is a proxy for whether the failure was stochastic**, and it
is free to observe.

On the trajectory side, the reason `correct but unsound` is so common has to do with how
agents recover. An agent that makes a mistake often notices and corrects — retries the tool
call with different arguments, re-reads the state, tries another path. That recovery is
genuinely valuable and it is also what produces a correct outcome from an unsound trajectory:
the double write happened, the out-of-scope read happened, and the final state is right
anyway. **Recovery hides its own causes**, which is why side effects need a separate
instrument from outcomes.

There is a connection to {{ch:ops-agent-tracing}} worth making explicit. That chapter found
the cause of an agent failure sits 2.7 steps back from where it becomes visible, and that
per-step correctness checks catch only 27% because the causing step *succeeded*. The
invariants in {{sec:9-practical-example}} are constructed to catch precisely those: a
duplicate write succeeded, an out-of-scope read succeeded, a claim untraceable to any tool
result was produced by a step that returned normally. **The invariant suite is the evaluation
form of that chapter's recorded fields**, and the two should be built together because they
consume the same instrumentation.

Finally, the invariants are cheap for a structural reason rather than an accidental one. Each
is a predicate over the recorded trajectory, evaluable by a few lines of code or a small model
call, with no reference and no ground truth. That is the same escape
{{cite:jimenez2023swebench}} used at the outcome level — a test defines the acceptable set
rather than sampling it — applied to the process instead of the result.

## 8. Implementation

The first listing reconciles single-run success with pass^k and prices retries.

```python {tier=A name=pass-k-separates-capability-from-reliability}
"""A single-run success rate is a capability measurement. Users experience reliability.

cite:yao2024taubench reported state-of-the-art function-calling agents succeeding on under
50% of tasks, and -- the number that matters more -- pass^8 below 25% in the retail domain:
the probability that all eight independent attempts at the same task succeed.

Under independence those two numbers are irreconcilable. At p = 0.5, pass^8 would be 0.4%.
Observing 25% means the trials are strongly correlated: some tasks succeed every time and
some fail every time, and the single-run average is a mixture of the two
(eq:pass-k-separates-capability-from-reliability).

Which changes what a retry buys. Retrying a coin-flip task helps; retrying a task the agent
cannot do wastes the attempt and the user's patience
(eq:retry-does-not-help-a-deterministic-failure).
"""
# (task class, share, per-attempt success probability)
CLASSES = [
    ("reliably solvable", 0.31, 0.97),
    ("usually solvable",  0.19, 0.78),
    ("coin flip",         0.22, 0.47),
    ("rarely solvable",   0.13, 0.16),
    ("out of reach",      0.15, 0.02),
]

print("A task population with very different per-attempt reliabilities.")
print()
print(f"{'task class':>21}{'share':>9}{'p(success)':>13}"
      f"{'pass^1':>10}{'pass^4':>10}{'pass^8':>10}")
print("-" * 73)
for name, sh, p in CLASSES:
    print(f"{name:>21}{sh:>9.0%}{p:>13.2f}"
          f"{p:>10.2f}{p ** 4:>10.2f}{p ** 8:>10.2f}")


def pass_k(k):
    return sum(sh * p ** k for name, sh, p in CLASSES)


print("-" * 73)
print(f"{'POPULATION':>21}{1.0:>9.0%}{'':>13}"
      f"{pass_k(1):>10.2f}{pass_k(4):>10.2f}{pass_k(8):>10.2f}")

print()
print(f"single-run success {pass_k(1):.0%}, pass^8 {pass_k(8):.0%}")
print(f"under independence, {pass_k(1):.0%}^8 would be "
      f"{pass_k(1) ** 8:.2%} -- the gap is the correlation")

print()
print()
print("Where each level of pass^k gets its mass.")
print()
print(f"{'task class':>21}", end="")
for k in (1, 2, 4, 8, 16):
    print(f"{('share of pass^' + str(k)):>18}", end="")
print()
print("-" * 111)
contrib = {}
for name, sh, p in CLASSES:
    print(f"{name:>21}", end="")
    for k in (1, 2, 4, 8, 16):
        c = sh * p ** k / pass_k(k)
        contrib[(name, k)] = c
        print(f"{c:>18.1%}", end="")
    print()

print()
print(f"at pass^1 the reliably-solvable class is "
      f"{contrib[('reliably solvable', 1)]:.0%} of successes;")
print(f"at pass^8 it is {contrib[('reliably solvable', 8)]:.0%}")

print()
print()
print("Two improvements with the same effect on the headline number.")
print()


def population(classes):
    return {k: sum(sh * p ** k for n, sh, p in classes) for k in (1, 4, 8, 16)}


BASE = list(CLASSES)
# A: make the coin-flip tasks a bit more likely to work.
A = [(n, sh, p + (0.16 if n == "coin flip" else 0.0)) for n, sh, p in CLASSES]
# B: convert some coin-flip tasks into reliably solvable ones.
B = []
moved = 0.07
for n, sh, p in CLASSES:
    if n == "coin flip":
        B.append((n, sh - moved, p))
    elif n == "reliably solvable":
        B.append((n, sh + moved, p))
    else:
        B.append((n, sh, p))

print(f"{'model':>34}{'pass^1':>10}{'pass^4':>10}{'pass^8':>10}"
      f"{'pass^16':>11}")
print("-" * 75)
res = {}
for label, cls in (("baseline", BASE),
                   ("A: coin-flip tasks +0.16", A),
                   ("B: 7% of coin flips made reliable", B)):
    r = population(cls)
    res[label] = r
    print(f"{label:>34}{r[1]:>10.3f}{r[4]:>10.3f}{r[8]:>10.3f}"
          f"{r[16]:>11.3f}")

print()
print(f"A improves pass^1 by {res['A: coin-flip tasks +0.16'][1] - res['baseline'][1]:+.3f} "
      f"and pass^8 by {res['A: coin-flip tasks +0.16'][8] - res['baseline'][8]:+.3f}")
print(f"B improves pass^1 by {res['B: 7% of coin flips made reliable'][1] - res['baseline'][1]:+.3f} "
      f"and pass^8 by {res['B: 7% of coin flips made reliable'][8] - res['baseline'][8]:+.3f}")

print()
print()
print("What a user gets, if the product retries on failure.")
print()
print(f"{'retries allowed':>17}{'task succeeds':>16}{'attempts spent':>17}"
      f"{'attempts per success':>23}")
print("-" * 73)
retry = {}
for r in (1, 2, 3, 5, 8):
    succ = sum(sh * (1.0 - (1.0 - p) ** r) for n, sh, p in CLASSES)
    spent = sum(sh * sum((1.0 - p) ** (j - 1) for j in range(1, r + 1))
                for n, sh, p in CLASSES)
    retry[r] = (succ, spent, spent / succ)
    print(f"{r:>17}{succ:>16.3f}{spent:>17.2f}{spent / succ:>23.2f}")

print()
print()
print("And where those retry attempts go.")
print()
print(f"{'task class':>21}{'share of attempts at r=5':>27}"
       f"{'share of successes':>21}{'ratio':>9}")
print("-" * 78)
r5_spent = {}
for name, sh, p in CLASSES:
    spent = sh * sum((1.0 - p) ** (j - 1) for j in range(1, 6))
    gained = sh * (1.0 - (1.0 - p) ** 5)
    r5_spent[name] = (spent / retry[5][1], gained / retry[5][0])
    print(f"{name:>21}{spent / retry[5][1]:>27.1%}"
          f"{gained / retry[5][0]:>21.1%}"
          f"{(spent / retry[5][1]) / (gained / retry[5][0]):>9.2f}")

print()
print()
print("A retry budget spent where it pays: retry only the classes that")
print("respond to it, if you can tell them apart.")
print()
print(f"{'policy':>34}{'success':>11}{'attempts':>11}"
      f"{'attempts per success':>23}")
print("-" * 79)
POLICIES = [
    ("one attempt, no retries", lambda n, p: 1),
    ("five attempts, everything", lambda n, p: 5),
    ("five attempts if first fails fast", lambda n, p: 5 if p > 0.30 else 1),
    ("retry only the coin flips", lambda n, p: 5 if n == "coin flip" else 1),
]
pol = {}
for label, f in POLICIES:
    succ, spent = 0.0, 0.0
    for name, sh, p in CLASSES:
        r = f(name, p)
        succ += sh * (1.0 - (1.0 - p) ** r)
        spent += sh * sum((1.0 - p) ** (j - 1) for j in range(1, r + 1))
    pol[label] = (succ, spent, spent / succ)
    print(f"{label:>34}{succ:>11.3f}{spent:>11.2f}{spent / succ:>23.2f}")

print(f"""
The population table is cite:yao2024taubench's two numbers reconciled. Single-run success is
{pass_k(1):.0%} and pass^8 is {pass_k(8):.0%}; under independence the second would be
{pass_k(1) ** 8:.2%}. **The gap is entirely correlation between attempts**
(eq:pass-k-separates-capability-from-reliability), and the correlation is not mysterious:
some tasks are within the agent's reach every time and some are outside it every time.

Which means the single-run rate is measuring a mixture and reporting it as a capability. Two
systems with identical {pass_k(1):.0%} single-run rates can have completely different
populations behind them, and the user experience differs entirely.

The contribution table shows what pass^k is actually selecting for. At k=1 the
reliably-solvable class supplies {contrib[('reliably solvable', 1)]:.0%} of successes; at
k=8 it supplies {contrib[('reliably solvable', 8)]:.0%}. **Raising k filters out everything
except the tasks the agent can do consistently**, which is the correct definition of what an
agent can do.

The two-improvements table is the design consequence. Improvement A raises the coin-flip
tasks' success probability by {0.16:.2f} and moves pass^1 by
{res['A: coin-flip tasks +0.16'][1] - res['baseline'][1]:+.3f}; improvement B converts
{moved:.0%} of coin-flip tasks into reliable ones and moves pass^1 by
{res['B: 7% of coin flips made reliable'][1] - res['baseline'][1]:+.3f}.

On the headline number they are indistinguishable. On pass^8, A moves
{res['A: coin-flip tasks +0.16'][8] - res['baseline'][8]:+.3f} and B moves
{res['B: 7% of coin flips made reliable'][8] - res['baseline'][8]:+.3f} --
**B is worth {(res['B: 7% of coin flips made reliable'][8] - res['baseline'][8]) / (res['A: coin-flip tasks +0.16'][8] - res['baseline'][8]):.0f} times as much** on the
metric that describes what users can depend on, and it ties the comparison anyone would
actually run.

The retry table is the other half. Allowing {5} attempts takes success from
{retry[1][0]:.3f} to {retry[5][0]:.3f}, which sounds like a good trade until the last column:
attempts per success rises from {retry[1][2]:.2f} to {retry[5][2]:.2f}.

The allocation table says where they go. `out of reach` tasks are
{r5_spent['out of reach'][0]:.0%} of the attempts spent and
{r5_spent['out of reach'][1]:.0%} of the successes gained --
{r5_spent['out of reach'][0] / r5_spent['out of reach'][1]:.1f} times more consumption than
production. **Retrying a task the agent cannot do burns five attempts to arrive back where it
started** (eq:retry-does-not-help-a-deterministic-failure), and the user waits through all
five.

The policy table prices the fix and its ordering is instructive. Retrying everything reaches
{pol['five attempts, everything'][0]:.3f} at {pol['five attempts, everything'][1]:.2f}
attempts per task. Retrying only when the first attempt failed *fast* -- rather than after a
long doomed trajectory -- reaches {pol['five attempts if first fails fast'][0]:.3f} for
{pol['five attempts if first fails fast'][1]:.2f}, which is
{pol['five attempts, everything'][0] - pol['five attempts if first fails fast'][0]:.3f} less
success for {1 - pol['five attempts if first fails fast'][1] / pol['five attempts, everything'][1]:.0%}
fewer attempts.

Per success it is {pol['five attempts if first fails fast'][2]:.2f} against
{pol['five attempts, everything'][2]:.2f} -- **essentially the same as never retrying at
all**, while capturing most of the retry benefit.

The narrower policy in the last row, retrying only the class that responds best, is worse on
both axes than the fast-failure heuristic. That is the useful lesson: you do not need to
identify the recoverable tasks, only to *stop* on the unrecoverable ones, and a long failing
trajectory is a usable signal for that. A cheap negative filter beats an expensive positive
one.

Two things to carry out of this. **Report pass^k, not pass^1**, because the number users
experience is the one that requires the task to work every time. And **make retry
conditional**, because an unconditional retry budget is spent mostly on tasks that will not
respond to it. The next listing takes up what "succeed" should mean in the first place.""")
```

## 9. Practical Example

A task population with heterogeneous reliability:

```
           task class    share   p(success)    pass^1    pass^4    pass^8
-------------------------------------------------------------------------
    reliably solvable      31%         0.97      0.97      0.89      0.78
     usually solvable      19%         0.78      0.78      0.37      0.14
            coin flip      22%         0.47      0.47      0.05      0.00
      rarely solvable      13%         0.16      0.16      0.00      0.00
         out of reach      15%         0.02      0.02      0.00      0.00
-------------------------------------------------------------------------
           POPULATION     100%                   0.58      0.36      0.27
```

Single-run **58%**, pass^8 **27%**, where independence would give **1.21%** — **the gap is
the correlation** ({{eq:pass-k-separates-capability-from-reliability}}).

```
           task class   share of pass^1   share of pass^4   share of pass^8  share of pass^16
---------------------------------------------------------------------------------------------
    reliably solvable             52.2%             77.2%             90.1%             98.2%
     usually solvable             25.7%             19.8%              9.7%              1.8%
            coin flip             17.9%              3.0%              0.2%              0.0%
```

Raising $k$ filters the population toward what the agent does consistently: **52% → 90%** from
the reliable class.

```
                             model    pass^1    pass^4    pass^8    pass^16
---------------------------------------------------------------------------
                          baseline     0.576     0.356     0.270      0.194
          A: coin-flip tasks +0.16     0.611     0.380     0.274      0.194
 B: 7% of coin flips made reliable     0.611     0.414     0.324      0.237
```

Both move pass^1 by **+0.035**. On pass^8, A moves **+0.005** and B **+0.055** — **11×**
apart, and tied on the metric anyone reports.

```
  retries allowed   task succeeds   attempts spent   attempts per success
-------------------------------------------------------------------------
                1           0.576             1.00                   1.74
                5           0.801             2.20                   2.75
                8           0.839             2.76                   3.29

           task class   share of attempts at r=5   share of successes    ratio
------------------------------------------------------------------------------
    reliably solvable                      14.5%                38.7%     0.37
            coin flip                      20.3%                26.3%     0.77
      rarely solvable                      21.4%                 9.4%     2.27
         out of reach                      32.7%                 1.8%    18.16
```

`out of reach` consumes **32.7%** of attempts for **1.8%** of successes — **18×** more
consumption than production ({{eq:retry-does-not-help-a-deterministic-failure}}).

```
                            policy    success   attempts   attempts per success
-------------------------------------------------------------------------------
           one attempt, no retries      0.576       1.00                   1.74
         five attempts, everything      0.801       2.20                   2.75
 five attempts if first fails fast      0.735       1.29                   1.76
         retry only the coin flips      0.683       1.23                   1.80
```

Retrying only after a *fast* failure: **41% fewer attempts** for 0.066 less success, at
essentially the same attempts-per-success as never retrying. **A cheap negative filter beats
an expensive positive one.**

The second listing asks what "succeed" should mean.

```python {tier=A name=outcome-evaluation-credits-lucky-trajectories}
"""An agent that reaches the right answer by the wrong route has passed your evaluation.

cite:jimenez2023swebench grades by running the repository's tests, which is the strongest
form of outcome evaluation available -- and even there, a patch that passes the tests for
the wrong reason passes. For agents with side effects, the gap is wider: the outcome can be
correct while the trajectory issued a refund twice, read a record it should not have, or
succeeded on a coincidence that will not recur
(eq:outcome-evaluation-credits-lucky-trajectories).

The obvious fix is to evaluate the trajectory. That requires a reference trajectory, which
puts you back in ch:ev-why-hard's position: there are many correct routes and the reference
is one of them (eq:trajectory-matching-inherits-the-answer-space-problem).

This listing measures both errors and finds the instrument that avoids both.
"""
# (outcome correct?, trajectory sound?, share, what it is, generalises?)
CELLS = [
    (True,  True,  0.44, "correct, for the right reason",  True),
    (True,  False, 0.14, "correct, by luck or side effect", False),
    (False, True,  0.11, "sound attempt, task not doable",  True),
    (False, False, 0.31, "wrong, and wrongly",              False),
]

print("Outcome and trajectory are two questions with four answers.")
print()
print(f"{'outcome':>10}{'trajectory':>13}{'share':>9}"
      f"{'what it is':>34}{'will recur?':>13}")
print("-" * 79)
for ok, sound, sh, desc, gen in CELLS:
    print(f"{('pass' if ok else 'fail'):>10}"
          f"{('sound' if sound else 'unsound'):>13}{sh:>9.2f}"
          f"{desc:>34}{('yes' if gen else 'no'):>13}")

outcome_pass = sum(sh for ok, s, sh, d, g in CELLS if ok)
truly_good = sum(sh for ok, s, sh, d, g in CELLS if ok and s)
print("-" * 79)
print(f"{'outcome score':>10}{'':>13}{outcome_pass:>9.2f}")
print(f"{'dependable':>10}{'':>13}{truly_good:>9.2f}")
print()
print(f"outcome evaluation over-credits by "
      f"{outcome_pass - truly_good:.2f} -- "
      f"{(outcome_pass - truly_good) / outcome_pass:.0%} of its passes")

print()
print()
print("What that does to a comparison between two agents.")
print()
AGENTS = [
    ("agent P", 0.44, 0.14),
    ("agent Q", 0.40, 0.21),
]
print(f"{'agent':>10}{'sound passes':>15}{'lucky passes':>15}"
      f"{'outcome score':>16}{'dependable':>13}{'rank flip?':>13}")
print("-" * 82)
by_out = sorted(AGENTS, key=lambda a: -(a[1] + a[2]))
by_dep = sorted(AGENTS, key=lambda a: -a[1])
for name, sound, lucky in AGENTS:
    flip = (by_out.index((name, sound, lucky))
            != by_dep.index((name, sound, lucky)))
    print(f"{name:>10}{sound:>15.2f}{lucky:>15.2f}"
          f"{sound + lucky:>16.2f}{sound:>13.2f}"
          f"{('yes' if flip else 'no'):>13}")
print()
print(f"outcome ranking: {by_out[0][0]} first; dependable ranking: "
      f"{by_dep[0][0]} first")

print()
print()
print("So evaluate the trajectory. How many correct trajectories are there?")
print()
TASKS = [
    ("look up one record",              1.0),
    ("look up and summarise",           3.0),
    ("book a flight to spec",          14.0),
    ("resolve a support ticket",       120.0),
    ("debug and patch a repository", 2600.0),
]
print(f"{'task':>32}{'valid trajectories':>21}"
      f"{'exact-match credit':>21}{'valid marked wrong':>21}")
print("-" * 95)
traj = {}
for name, a in TASKS:
    hit = min(1.0, 1.0 / a)
    traj[name] = (a, hit, 1.0 - hit)
    print(f"{name:>32}{a:>21.0f}{hit:>21.3f}{1.0 - hit:>21.2%}")

print()
print("Trajectory matching is ch:ev-why-hard's reference problem with a")
print("bigger answer space, because a trajectory is a sequence of choices")

print()
print()
print("The instrument that avoids both errors: check invariants, not routes.")
print()
INVARIANTS = [
    ("no tool called with invalid arguments",   0.19, 0.3),
    ("no write repeated with the same effect",  0.22, 0.4),
    ("no read outside the authorised scope",    0.11, 0.5),
    ("every claim traceable to a tool result",  0.26, 1.2),
    ("terminal state matches the request",      0.31, 2.0),
    ("no step contradicts an earlier result",   0.17, 3.0),
]
LUCKY = sum(sh for ok, s, sh, d, g in CELLS if ok and not s)
print(f"{'invariant':>40}{'catches of the lucky':>22}"
      f"{'effort':>9}{'per effort':>13}")
print("-" * 84)
inv = {}
for name, catch, eff in INVARIANTS:
    inv[name] = (catch, eff, catch / eff)
    print(f"{name:>40}{catch:>22.0%}{eff:>9.1f}{catch / eff:>13.3f}")

order = sorted(INVARIANTS, key=lambda i: -(i[1] / i[2]))
print()
print("Applied in payback order:")
print()
print(f"{'after adding':>40}{'lucky passes remaining':>25}"
      f"{'effort so far':>16}{'reported score':>17}")
print("-" * 98)
rem = LUCKY
eff = 0.0
path = []
for name, catch, e in order:
    rem *= (1.0 - catch)
    eff += e
    path.append((name, rem, eff))
    print(f"{name:>40}{rem:>25.4f}{eff:>16.1f}"
          f"{truly_good + rem:>17.4f}")

print()
print(f"the reported score falls from {outcome_pass:.4f} to "
      f"{truly_good + path[-1][1]:.4f} and gets more true")

print()
print()
print("None of these needs a reference trajectory. Which do?")
print()
NEEDS = [
    ("outcome check",              "a correct answer",       "yes"),
    ("trajectory match",           "a reference trajectory", "yes"),
    ("invariant checks",           "nothing",                "no"),
    ("side-effect diff",           "a clean environment",    "no"),
    ("step-level judge",           "a rubric",               "no"),
    ("cause-distance triage",      "recorded state",         "no"),
]
print(f"{'instrument':>26}{'what it needs':>26}{'needs ground truth?':>22}")
print("-" * 74)
for name, needs, gt in NEEDS:
    print(f"{name:>26}{needs:>26}{gt:>22}")

print()
print()
print("Cost per 1000 evaluated trajectories.")
print()
COSTS = [
    ("outcome check (human)",      3400.0, "over-credits by 24%"),
    ("outcome check (tests)",        12.0, "where tests exist"),
    ("trajectory match",           4100.0, "and penalises valid routes"),
    ("invariant checks",             31.0, "catches most lucky passes"),
    ("side-effect diff",             46.0, "catches the dangerous ones"),
    ("step-level judge",            190.0, "ch:ev-llm-judge applies"),
]
print(f"{'instrument':>26}{'cost/1000':>12}{'note':>34}")
print("-" * 72)
cost = {}
for name, c, note in COSTS:
    cost[name] = c
    print(f"{name:>26}{c:>12,.0f}{note:>34}")

combo = cost["outcome check (tests)"] + cost["invariant checks"] + cost["side-effect diff"]
print()
print(f"tests + invariants + side-effect diff: {combo:,.0f} per 1000")
print(f"against human outcome checking alone: "
      f"{cost['outcome check (human)']:,.0f} "
      f"({cost['outcome check (human)'] / combo:.0f}x)")

print(f"""
The quadrant table is the problem in four rows. Outcome evaluation reports
{outcome_pass:.2f} and the dependable share is {truly_good:.2f}: **{(outcome_pass - truly_good) / outcome_pass:.0%} of
its passes are correct for a reason that will not recur**
(eq:outcome-evaluation-credits-lucky-trajectories).

The third row is worth noticing too. {CELLS[2][2]:.0%} of trajectories are sound attempts at
tasks that could not be completed -- the environment was wrong, the record did not exist, the
policy forbade it -- and outcome evaluation scores those as failures. So the outcome score is
wrong in both directions at once, and the errors do not cancel because they are different
tasks.

The comparison table is where this stops being an accounting complaint. Agent P has
{AGENTS[0][1]:.2f} sound passes and {AGENTS[0][2]:.2f} lucky ones; agent Q has
{AGENTS[1][1]:.2f} and {AGENTS[1][2]:.2f}. On the outcome score Q wins
{AGENTS[1][1] + AGENTS[1][2]:.2f} to {AGENTS[0][1] + AGENTS[0][2]:.2f}; on dependable passes
P wins {AGENTS[0][1]:.2f} to {AGENTS[1][1]:.2f}.

**The agent that gets luckier ranks higher**, and luck does not survive contact with a
different distribution of tasks.

The trajectory table is the obvious fix failing. Debugging a repository has on the order of
{2600:.0f} valid trajectories, so exact trajectory matching credits {traj['debug and patch a repository'][1]:.3f}
of correct routes and marks {traj['debug and patch a repository'][2]:.2%} of them wrong
(eq:trajectory-matching-inherits-the-answer-space-problem).

That is ch:ev-why-hard's reference-sampling problem with a much larger answer space, because a
trajectory is a *sequence* of choices and the space multiplies at every step. Trajectory
matching is a worse instance of a problem that was already severe for single answers.

The invariant table is the way out and it is the useful part of this listing. Each row is a
property that can be checked on a trajectory **without any reference at all**: were the tool
arguments valid, was a write repeated, did a read leave the authorised scope, is every claim
traceable to a tool result, does the terminal state match the request, does any step
contradict an earlier one.

`{order[0][0]}` catches {order[0][1]:.0%} of the lucky passes for {order[0][2]:.1f} units of
effort. Applied in payback order, the lucky share falls from {LUCKY:.4f} to
{path[-1][1]:.4f} and **the reported score falls from {outcome_pass:.4f} to
{truly_good + path[-1][1]:.4f}** -- which is the right direction. An evaluation that gets
stricter and lower is an evaluation that started too high.

The dependency table is the structural point. Of six instruments, two need ground truth and
four do not, and the four that do not are the ones that catch the failure modes outcome
scoring cannot see. **Reference-free does not mean weak here**, which is the opposite of the
situation in ch:ev-why-hard, and the reason is that an invariant is a predicate rather than a
comparison -- the same escape that made execution grading work.

The cost table closes it. Test-graded outcomes plus invariant checks plus a side-effect diff
cost {combo:,.0f} per thousand trajectories against {cost['outcome check (human)']:,.0f} for
human outcome checking alone -- {cost['outcome check (human)'] / combo:.0f} times cheaper,
and it measures three things instead of one.

The instrument that is *not* on that list is trajectory matching, at
{cost['trajectory match']:,.0f} and penalising every valid route it did not anticipate. It is
the most expensive option and the only one that is wrong by construction, and it is what
"evaluate the reasoning, not just the answer" turns into when implemented literally.""")
```

```
   outcome   trajectory    share                        what it is  will recur?
-------------------------------------------------------------------------------
      pass        sound     0.44     correct, for the right reason          yes
      pass      unsound     0.14   correct, by luck or side effect           no
      fail        sound     0.11    sound attempt, task not doable          yes
      fail      unsound     0.31                wrong, and wrongly           no
-------------------------------------------------------------------------------
outcome score                  0.58
dependable                  0.44
```

**24% of outcome passes will not recur** ({{eq:outcome-evaluation-credits-lucky-trajectories}}),
and 11% of sound attempts are scored as failures — wrong in both directions, on different
tasks.

```
     agent   sound passes   lucky passes   outcome score   dependable   rank flip?
----------------------------------------------------------------------------------
   agent P           0.44           0.14            0.58         0.44          yes
   agent Q           0.40           0.21            0.61         0.40          yes
```

**The agent that gets luckier ranks higher.**

```
                            task   valid trajectories   exact-match credit   valid marked wrong
-----------------------------------------------------------------------------------------------
           look up and summarise                    3                0.333               66.67%
           book a flight to spec                   14                0.071               92.86%
        resolve a support ticket                  120                0.008               99.17%
    debug and patch a repository                 2600                0.000               99.96%
```

Trajectory matching marks **99.96%** of correct routes wrong
({{eq:trajectory-matching-inherits-the-answer-space-problem}}) — the reference problem with a
space that multiplies at every step.

```
                               invariant  catches of the lucky   effort   per effort
------------------------------------------------------------------------------------
   no tool called with invalid arguments                   19%      0.3        0.633
  no write repeated with the same effect                   22%      0.4        0.550
      terminal state matches the request                   31%      2.0        0.155

                            after adding   lucky passes remaining   effort so far   reported score
--------------------------------------------------------------------------------------------------
   no tool called with invalid arguments                   0.1134             0.3           0.5534
  no write repeated with the same effect                   0.0885             0.7           0.5285
   no step contradicts an earlier result                   0.0334             7.4           0.4734
```

Invariants take lucky passes from **0.14 to 0.033** and **the reported score from 0.58 to
0.47** — stricter and lower, which is the right direction.

```
                instrument             what it needs   needs ground truth?
--------------------------------------------------------------------------
             outcome check          a correct answer                   yes
          trajectory match    a reference trajectory                   yes
          invariant checks                   nothing                    no
          side-effect diff       a clean environment                    no
```

**Reference-free does not mean weak here** — an invariant is a predicate, not a comparison,
which is the same escape execution grading uses.

## 10. Production Considerations

Report pass^k, not pass^1. The number users experience requires the task to work every time,
and pass^1 averages the reliable and the impossible into one figure.

Choose $k$ from your product's retry policy. If the product does not retry, $k$ is the number
of times a user will attempt before giving up.

Make retries conditional on failure speed. A long failing trajectory means the agent's model
of the task is wrong and a fresh sample will build the same wrong model.

Build the invariant suite before the trajectory-matching harness. It is cheaper, needs no
reference, and catches the failures outcome scoring misses.

Diff side effects against a clean environment. Duplicate writes and out-of-scope reads are
the failures that survive a correct outcome, and they are the ones with consequences.

Score `sound attempt, impossible task` separately from `wrong answer`. They call for
different work and outcome scoring merges them.

Instrument once for {{ch:ops-agent-tracing}} and {{ch:ev-agents}} together. The invariants
here consume exactly the fields that chapter argued for.

## 11. Common Mistakes

**Reporting single-run success as a capability.** It is a mixture of reliable and impossible
tasks with no way to tell which moved.

**Comparing agents on pass^1.** Two improvements 11× apart on reliability tie on it.

**Unconditional retry budgets.** A third of the attempts go to tasks that will never
succeed.

**Treating a correct outcome as a passing trajectory.** A quarter of passes here are luck or
side effect.

**Building trajectory matching.** The most expensive instrument in the chapter and the only
one wrong by construction.

**Scoring impossible tasks as agent failures.** 11% of trajectories were sound and the task
could not be done.

## 12. Failure Modes

**Reliability regression shipped as an improvement.** pass^1 rose, pass^8 fell, and only
pass^1 was on the dashboard.

**Retry storm on an unreachable capability.** Every user request for a feature the agent
cannot do consumes five attempts and a minute of waiting.

**Duplicate side effects passing evaluation.** The refund was issued twice, the second call
errored, the final state is correct, and the outcome check passed.

**Valid trajectory rejected by the matcher.** The agent found a better route than the
reference and the evaluation recorded a regression.

**Invariants that encode the current implementation.** The suite asserts the tool order the
agent happens to use, and now it is a trajectory matcher with extra steps.

**Environment-caused failures counted against the agent.** The record did not exist, the API
was down, and the agent's sound attempt is in the failure bucket.

## 13. Alternatives

**Execution-graded outcomes.** {{cite:jimenez2023swebench}}'s design — real tests as the
grader. The strongest outcome instrument available and it still admits the lucky pass.

**Step-level LLM judge.** Score each step against a rubric.
{{eq:optimising-against-a-judge-diverges}} applies in full, and it is 6× the cost of the
invariant suite.

**Simulated-user protocols.** {{cite:yao2024taubench}}'s design — a simulated user and a
domain policy, with pass^k reported. The best available reliability instrument, and building
the simulator is most of the work.

**Failure taxonomy annotation.** {{cite:cemri2025mast}}-style classification of what went
wrong. Diagnostic rather than gating, and the annotation cost is real.

**Trace localisation.** {{cite:deshpande2025trail}}'s task — find the causing step. The right
instrument for triage rather than for evaluation, and at 11% automated accuracy it is not yet
a gate.

## 14. Evaluation

Compute pass^k for $k \in \{1, 2, 4, 8\}$ on your task set and compare pass^8 against
$(\text{pass}^1)^8$. The ratio is your heterogeneity.

Bucket your tasks by measured per-attempt success rate and report the shares. That
distribution, not the mean, is what your product experiences.

Measure the correlation between failure-trajectory length and retry success. If it holds, you
have a free retry filter.

Audit a sample of passing trajectories for invariant violations. The lucky-pass share is the
number that tells you how much your outcome score is inflated.

Diff side effects for every evaluation run against a clean environment snapshot, and count
violations separately from outcome failures.

## 15. Advanced Concepts

The within-task independence assumed in the pass^k model is itself questionable, and it fails
in the direction that makes things worse. Attempts share a prompt, a tool set, and a model, so
even within a task the failures correlate — an agent that misreads a schema on attempt one
usually misreads it on attempt two. That means observed pass^k is *higher* than the model
predicts for a given pass^1, and the correct interpretation is that heterogeneity is somewhat
smaller than the naive fit suggests while within-task correlation makes up the difference.
Both effects argue against retries and for the same fixes, so the practical conclusions are
robust; the decomposition is not.

The task-class model also treats $p_c$ as a property of the task, when much of it is a
property of the *interface*. A task that is a coin flip against one tool schema is reliably
solvable against a clearer one, which is why "promote a class" improvements — the B-type in
{{sec:9-practical-example}} — are so often schema work rather than model work.
**The most valuable agent improvements are usually changes to what the agent is given rather
than to the agent**, and pass^k is the metric that makes them visible.

There is a subtlety about the invariant suite that needs guarding against. Invariants are
predicates over trajectories, which makes them reference-free — but a badly chosen invariant
encodes the current implementation's habits and becomes a trajectory matcher wearing a
predicate's clothes. The test is whether a genuinely better trajectory could violate the
invariant. "No duplicate write with the same effect" survives that test; "the search tool is
called before the fetch tool" does not. **Write invariants about consequences, never about
sequences.**

Finally, the two halves of this chapter interact in a way neither states alone. Lucky passes
are, by construction, less repeatable than sound ones — a correct outcome from an unsound
trajectory depends on a coincidence that will not recur. So the lucky share is concentrated in
the low-$p_c$ classes, which means **pass^k partially corrects for trajectory unsoundness
without being told about it.** Raising $k$ filters out both the unreliable tasks and the lucky
passes, and that is an argument for reporting pass^8 even in systems where trajectory
instrumentation does not exist yet.

## 16. Connection to Previous Chapters

{{eq:reference-scoring-penalises-valid-answers}} from {{ch:ev-why-hard}} is what defeats
trajectory matching, with an answer space that multiplies at every step rather than being
fixed per item.

{{eq:cause-distance-drives-triage-cost}} from {{ch:ops-agent-tracing}} explains why the
invariants that work are the ones checking consequences of steps that *succeeded* — the same
class of failure that defeats per-step correctness monitoring.

{{eq:most-rag-failures-are-invisible-end-to-end}} from {{ch:ev-rag}} is the identical
instrument argument one system up: a single verdict cannot distinguish failure modes needing
different fixes.

{{eq:optimising-against-a-judge-diverges}} from {{ch:ev-llm-judge}} is why the step-level
judge is the last resort in the instrument table rather than the first.

## 17. Exercises

1. Compute pass^k for $k \in \{1,2,4,8\}$ on your agent's task set. How does pass^8 compare to
   $(\text{pass}^1)^8$?

2. Bucket your tasks by measured per-attempt success rate. What share is `out of reach`, and
   what is your retry policy spending on it?

3. Test whether failing-trajectory length predicts retry success. What does a fast-failure
   filter buy you?

4. Audit fifty passing trajectories for invariant violations. What is your lucky-pass share?

5. Take one task class currently at coin-flip reliability and try to promote it by changing
   the tool schema rather than the model. How far does pass^8 move?

## 18. Interview Questions

1. Our agent succeeds on 60% of tasks. What do you ask next?

2. Why can a 50% single-run agent have a pass^8 of 25%?

3. Two changes both improve success by 3 points. How would you decide between them?

4. We retry three times on failure. What is that costing?

5. Our agent booked the right flight. Did it pass?

6. Why not just compare the trajectory against a reference?

## 19. Research Questions

1. How much of observed pass^k excess is task heterogeneity and how much is within-task
   correlation, and can they be separated?

2. How well does failing-trajectory length predict retry success across domains?

3. What share of coin-flip tasks can be promoted to reliable by interface changes alone,
   holding the model fixed?

4. Which invariants generalise across agent architectures, and which encode implementation
   habits?

## 20. Chapter Summary

Agent evaluation fails in two ways that single-turn evaluation does not.

**A success rate is not a rate.** {{cite:yao2024taubench}}'s under-50% single-run and
sub-25% pass^8 are irreconcilable under independence; in the population here, **58%** and
**27%** against an independent prediction of **1.21%**, and the gap is heterogeneity
({{eq:pass-k-separates-capability-from-reliability}}). Raising $k$ reweights toward what the
agent does consistently — the reliable class goes from **52% to 90%** of successes — and two
improvements tied at **+0.035** on pass^1 differ **11×** on pass^8.

Retries land badly for the same reason: `out of reach` tasks take **32.7%** of attempts and
give **1.8%** of successes ({{eq:retry-does-not-help-a-deterministic-failure}}). Retrying only
after a fast failure costs **41% fewer attempts** for 0.066 less success, because failure speed
is a free proxy for whether the failure was stochastic.

**And an outcome is not a pass.** **24%** of outcome-evaluation passes are correct for reasons
that will not recur ({{eq:outcome-evaluation-credits-lucky-trajectories}}), 11% of sound
attempts are scored as failures, and the two errors invert a comparison between agents.
Trajectory matching makes it worse — **99.96%** of correct routes marked wrong
({{eq:trajectory-matching-inherits-the-answer-space-problem}}) — while reference-free
invariants take lucky passes from **0.14 to 0.033** and the reported score from **0.58 to
0.47**, for **38×** less than human outcome checking.

What both halves share is a distinction between what a measurement averages and what a user
depends on. pass^1 averages over tasks; the user has one task. An outcome score averages over
routes; the user gets the route that was taken, side effects included. The instruments that
work — pass^k, invariants, side-effect diffs — are all ways of refusing to average, and they
all report *lower* numbers than the ones they replace. That is the tell: an agent evaluation
that gets stricter and reports less is usually the first one that was measuring the right
thing.

Carry forward: **report pass^k**, and **check consequences, not routes**.

## 21. Further Reading

- {{cite:yao2024taubench}} — the pass^k protocol and the reliability result that organises
  this chapter.
- {{cite:jimenez2023swebench}} — execution-graded outcomes at scale, the strongest available
  outcome instrument.
- {{cite:cemri2025mast}} — a taxonomy of multi-agent failure modes, which is where an
  invariant suite's contents come from.
- {{cite:deshpande2025trail}} — localising the causing step inside a trace, the triage
  counterpart to this chapter's gating instruments.
