---
id: sd-routing-caching
number: 190
part: XXII
tier: full
status: draft
requires: [three-properties-break-the-stack, semantic-failure-has-no-instrument,
           retry-needs-a-verifier, model-belongs-interleaved]
provides: [cascade-is-a-verifier-bet, cache-threshold-is-an-error-cost-decision,
           tight-cache-is-a-stability-layer, hit-rate-is-not-the-objective]
citations: [chen2023frugalgpt, hu2024routerbench, kwon2023pagedattention]
---

## 1. Learning Objectives

By the end of this chapter you will be able to state why a model cascade is a bet on
a judge rather than a bet on a cheap model, and compute the judge recall at which
the bet starts paying; show why a perfect judge produces an answer *better* than the
expensive model alone rather than merely cheaper; apply the random-split test that
catches a cascade being justified by its cost column; derive a semantic cache's
similarity threshold from the error cost of the surface it serves rather than from
its hit rate; and explain why a tight semantic cache improves accuracy while a loose
one destroys it.

## 2. Why This Matters

{{ch:sd-architecture}} established that the model call is expensive enough to be a
first-order design concern rather than an optimisation. This chapter is the two
techniques everyone reaches for in response — route to a cheaper model when you can,
and cache what you have already answered — and both of them turn out to be governed
by something other than the quantity their dashboards report.

{{cite:chen2023frugalgpt}} reports matching the best individual model with up to 98%
cost reduction through cascading, and {{cite:hu2024routerbench}} made the trade-off
measurable rather than assertable with over 405,000 precomputed inference outcomes.
Those are real results. What neither headline carries is the **precondition**: a
cascade escalates when something decides the cheap answer is insufficient, and that
something is a verifier.

{{sec:9-practical-example}} finds that a cascade with a perfect judge is right
**97.5%** of the time — better than the expensive model's 91% — at **33%** of its
cost, while a cascade with a poor judge is right **74.2%** and is beaten by randomly
splitting traffic ({{eq:cascade-is-a-verifier-bet}}). Same models, same prices. The
judge is the entire product.

The cache result has the same shape. Sweeping a semantic cache's similarity
threshold across three domains that differ **only** in what a wrong answer costs
moves the optimum from 0.82 to 0.94, and the hit rate at the optimum from **57%** to
**15%** ({{eq:cache-threshold-is-an-error-cost-decision}}). A single threshold
shipped across a product with all three surfaces is wrong on at least two.

## 3. Prerequisites

You need {{ch:sd-architecture}}'s three properties
({{eq:three-properties-break-the-stack}}) and, specifically, its finding that
caching survives at 19% — this chapter is what to do with the 19%.

You need {{eq:semantic-failure-has-no-instrument}}, because both of this chapter's
failure modes are instances of it: a cascade dashboard reports cost, a cache
dashboard reports hit rate, and neither reports the thing that decides whether the
configuration is right.

{{eq:retry-needs-a-verifier}} from {{ch:ag-recovery}} is the result this chapter
generalises. There, a retry without a verifier was a fresh sample. Here, an
escalation without a verifier is the same thing with a price tag.

{{eq:model-belongs-interleaved}} is assumed but not required; routing composes with
interleaving rather than replacing it.

## 4. Intuitive Explanation

Start with routing, and with the mental model that makes cascades look obviously
good: most requests are easy, so send them to a cheap model, and only pay for the
expensive one on the hard ones. Cost falls, quality holds.

The step that model skips is how you know which ones are hard. You do not get to
inspect the request and see difficulty written on it. What you actually do is run
the cheap model, look at what it produced, and decide whether it is good enough. That
decision is a judgement about an answer's correctness made without knowing the
correct answer — which is the verifier problem, and this book has spent several
chapters establishing that it is hard.

So a cascade is not "cheap model plus expensive model." It is "cheap model plus
expensive model plus a verifier," and the verifier is the component you did not plan
for, did not budget for, and cannot evaluate with the metrics you set up for the
other two.

Here is what makes it worth the trouble anyway. If the judge were perfect, you would
keep every cheap answer that happens to be right and escalate exactly the ones that
are wrong. That is not the expensive model's performance — it is **better**, because
you have combined two models and kept the best of each. A cascade with a good judge
is an ensemble that happens to be cheap. A cascade with a bad judge is an expensive
way to be wrong.

Caching has the same structure with a different name for the threshold. A semantic
cache serves a stored answer when a new question is *similar enough* to an old one.
Tighten the definition of "enough" and you serve fewer queries from cache but the
ones you serve are right. Loosen it and the hit rate climbs while you increasingly
answer questions nobody asked.

The instinct is to tune this by watching the hit rate, because hit rate is what
cache dashboards show and it moves satisfyingly when you loosen the threshold. But
hit rate is a measure of how often you avoided work, not of whether avoiding it was
correct. The quantity that decides the right threshold is **what a wrong answer
costs**, and that number lives outside the cache entirely.

## 5. Formal Explanation

Let a cascade have a cheap model with accuracy $a_c$ and cost $\kappa_c$, an
expensive model with accuracy $a_e > a_c$ and cost $\kappa_e \gg \kappa_c$, and a
judge characterised by two rates: $\tau$, the probability it escalates given the
cheap answer is wrong (recall), and $\phi$, the probability it escalates given the
cheap answer is right (false positive rate).

Every request runs the cheap model. The escalation rate is

$$ E \;=\; (1 - a_c)\,\tau \;+\; a_c\,\phi $$

and the cascade's accuracy and cost are

$$ A \;=\; a_c(1 - \phi) \;+\; E\,a_e, \qquad K \;=\; \kappa_c \;+\; E\,\kappa_e $$ (eq:cascade-is-a-verifier-bet)

The first term of $A$ is what makes cascades interesting: cheap answers that were
right and were kept. A perfect judge ($\tau = 1$, $\phi = 0$) gives
$A = a_c + (1 - a_c)a_e$, which **exceeds** $a_e$ whenever $a_c > 0$. The cascade is
an oracle-gated ensemble.

As $\tau$ falls, wrong cheap answers survive; as $\phi$ rises, right cheap answers
are needlessly escalated at full price. Both degrade $A$, and only $\phi$ raises $K$
for no return.

The comparison that keeps a cascade honest is the **random split**: send a share $p$
of traffic to the expensive model with no judge at all, giving
$A_{\text{rand}} = (1-p)a_c + p\,a_e$ at cost $(1-p)\kappa_c + p\kappa_e$. Solving
for the $p$ that matches the cascade's accuracy gives the cheapest judge-free way to
buy that accuracy. **A cascade that costs more than its random-split equivalent is
not a cost optimisation — it is a tax.**

For the cache, let queries arrive with similarity $s$ to the nearest cached entry,
distributed with density $f(s)$, and let $r(s)$ be the probability a stored answer at
similarity $s$ is correct for the new query. Serving from cache above threshold $\theta$
gives accuracy and spend

$$ A(\theta) = \int_{\theta}^{1} r(s) f(s)\,ds + a_f \int_{0}^{\theta} f(s)\,ds, \qquad K(\theta) = \kappa_h F(\theta^+) + \kappa_m F(\theta^-) $$

with $a_f$ the accuracy of a fresh generation. Neither is the objective. The
objective is total cost including errors:

$$ J(\theta) \;=\; K(\theta) \;+\; \bigl(1 - A(\theta)\bigr)\,\lambda $$ (eq:cache-threshold-is-an-error-cost-decision)

where $\lambda$ is the cost of one wrong answer. The optimum $\theta^\star$ minimises
$J$, and **$\theta^\star$ depends on $\lambda$** — a parameter of the product
surface, not of the cache.

## 6. Mathematical Foundation

Two structural facts follow from {{eq:cascade-is-a-verifier-bet}}.

**The oracle bonus.** At $\tau = 1, \phi = 0$, accuracy is $a_c + (1-a_c)a_e$ and the
gain over the expensive model alone is $a_c(1 - a_e)$ — the cheap model's correct
answers on cases the expensive model would have got wrong. This scales with the
cheap model's accuracy, so **a better cheap model makes the cascade's ceiling higher
even though it is the expensive model doing the hard work.**

**The break-even recall.** Setting $A = a_e$ and solving for $\tau$ under a coupling
$\phi = \gamma\tau$ gives the recall at which the cascade stops costing accuracy.
Rearranging,

$$ \tau^\star \;=\; \frac{a_c - a_e}{a_c\gamma(a_e - 1) - (1 - a_c)a_e + a_c\gamma \cdot a_e - a_c\gamma} $$

which is easier to obtain numerically than to read, and {{sec:9-practical-example}}
does so. The point is that $\tau^\star$ is a **number you can compute before building
anything**, and it converts "should we cascade?" into "can we build a judge with
recall above $\tau^\star$ on our traffic?"

For the cache, differentiating {{eq:cache-threshold-is-an-error-cost-decision}} at
the threshold gives the marginal condition: raising $\theta$ by $d\theta$ moves
$f(\theta)d\theta$ of queries from cache to fresh generation, costing
$(\kappa_m - \kappa_h)f(\theta)d\theta$ and buying
$(a_f - r(\theta))f(\theta)d\theta \cdot \lambda$ in avoided error. Setting these
equal,

$$ a_f - r(\theta^\star) \;=\; \frac{\kappa_m - \kappa_h}{\lambda} $$

Two consequences of that condition are worth writing down as results in their own
right, because they are the ones teams get wrong.

The first is that the accuracy contributed by the cached region is not bounded above
by fresh generation. Writing the cached region's mean correctness as
$\bar{r}(\theta) = \mathbb{E}[r(s) \mid s \ge \theta]$, the cache changes accuracy by

$$ \Delta A(\theta) \;=\; F(\theta^{+})\,\bigl(\bar{r}(\theta) - a_f\bigr) $$ (eq:tight-cache-is-a-stability-layer)

which is **positive whenever $\bar{r}(\theta) > a_f$** — that is, whenever the
threshold is tight enough that stored answers are more reliable than fresh ones.
Since $r$ is decreasing in $s$ and $r(1) \to 1$, such a $\theta$ always exists. A
sufficiently tight cache is an accuracy improvement, not a compromise.

The second concerns what happens if the threshold is tuned on hit rate. Hit rate is
$H(\theta) = F(\theta^{+})$, which is monotonically decreasing in $\theta$, so
maximising $H$ drives $\theta$ to its floor. But the objective is $J$, and

$$ \frac{\partial J}{\partial \theta}\bigg|_{\theta \to 0} \;=\; -f(0)\Bigl[(\kappa_m - \kappa_h) - \bigl(a_f - r(0)\bigr)\lambda\Bigr] \;<\; 0 $$ (eq:hit-rate-is-not-the-objective)

whenever $\lambda > (\kappa_m - \kappa_h) / (a_f - r(0))$. **The hit-rate optimum
and the cost optimum coincide only when errors are free.** For any $\lambda$ above
that bound, the threshold that maximises hit rate sits strictly on the wrong side of
the minimum, and the gap grows with $\lambda$ — which is what
{{sec:9-practical-example}} measures as 1.24, 3.87 and 5.51 times optimal.

**The optimal threshold is where the stored answer's accuracy deficit equals the
generation cost divided by the error cost.** As $\lambda \to \infty$ the right-hand
side vanishes and $\theta^\star$ rises until $r(\theta^\star) = a_f$ — a high-stakes
surface caches only what is nearly certain. As $\lambda \to 0$ the threshold falls to
wherever the cache is cheaper at all, which is everywhere.

## 7. Internal Mechanics

**What the judge actually sees.** An escalation judge inspects a generated answer
and a query, without a reference. The signals available are self-reported
confidence (poorly calibrated), output properties (length, hedging, refusal
markers), consistency across samples (expensive — it costs extra generations), and
a small verifier model trained on labelled escalation decisions. Each has a
different cost, and the cost of the judge is part of $\kappa_c$, which the formal
model above quietly folded in. A judge that costs as much as the expensive model
is not a cascade.

**Why $\phi$ hurts twice.** A false positive escalates an answer that was already
right. You pay $\kappa_e$ and, because the expensive model is not perfect either,
you sometimes replace a right answer with a wrong one. This is why $\phi$ appears
negatively in $A$ as well as positively in $K$, and why judges tuned purely for
recall underperform.

**Calibration is the judge's real problem.** The signals a judge has access to are
mostly monotone in the model's confidence, and a model's confidence is well known to
be miscalibrated in the specific direction that hurts here: it is most confident on
fluent, plausible, wrong answers. A judge reading confidence therefore has its
lowest recall exactly on the errors that matter most — the ones a user will act on.
This is why $\tau$ measured on a random sample of traffic overstates $\tau$ on the
subset where escalation would have changed the outcome, and why judge evaluation
should be stratified by error severity rather than pooled.

**The judge's cost is not free and not fixed.** A consistency-based judge that
samples the cheap model $k$ times costs $k\kappa_c$ and buys recall that improves
roughly with $\sqrt{k}$. Since the cascade's whole premise is $\kappa_c \ll
\kappa_e$, there is room for $k$ to be several before the economics turn — but the
turn is sharp, and a judge at $k = 8$ against a 20x price ratio has spent nearly half
the cascade's margin before a single escalation.

**Cache key construction.** A semantic cache keys on an embedding of the query, but
the answer usually depends on more than the query — user identity, permissions,
time, retrieved documents. An entry that ignores those is not merely stale, it is
serving one user's answer to another. In practice the key must be
`embedding(query) + hash(everything the answer depended on)`, and the second term
is what turns a semantic cache from a liability into an asset.

**Where the cache sits.** Under {{eq:model-belongs-interleaved}}, the deterministic
stages between model stages are cacheable with conventional caching and exact keys.
The model stages need semantic caching. These are different systems with different
correctness arguments, and conflating them is a common source of the failures in
{{sec:12-failure-modes}}.

**Serving-layer interaction.** {{cite:kwon2023pagedattention}}'s memory management
determines how much concurrency a given hardware budget supports, which sets
$\kappa_e$ in real terms. Routing decisions made against list prices rather than
achieved throughput can be wrong by a large factor.

## 8. Implementation

The first listing measures a two-model cascade as the judge's quality varies, and
applies the random-split test.

```python {tier=A name=bv1}
"""A cascade is a bet on a judge, not a bet on a cheap model.

Cascade routing sends every request to a cheap model first, and escalates to an
expensive one only when the cheap answer looks insufficient. cite:chen2023frugalgpt
reports large savings from this. But the escalation decision is made by something,
and that something is a verifier -- so the cascade inherits every property of the
verifier problem from ch:ag-recovery.

This listing measures what the judge's quality does to the cascade's economics
(eq:cascade-is-a-verifier-bet). The question is not "does a cascade save money" but
"how good does the judge have to be before it saves anything at all".
"""
# Cheap model: right on the easy majority, wrong on the hard tail.
# Expensive model: better everywhere, at 20x the price.
A_CHEAP = 0.72
A_EXP = 0.91
C_CHEAP = 1.0
C_EXP = 20.0


def cascade(judge_tpr, judge_fpr):
    """Every request goes to the cheap model. The judge inspects the cheap answer
    and escalates when it thinks the answer is wrong.

    judge_tpr: P(escalate | cheap answer IS wrong)   -- catching real errors
    judge_fpr: P(escalate | cheap answer is RIGHT)   -- escalating needlessly

    Returns (accuracy, cost, escalation rate).
    """
    wrong = 1 - A_CHEAP
    right = A_CHEAP

    # Four outcomes, by whether the cheap answer was right and whether we escalate.
    esc_from_wrong = wrong * judge_tpr
    esc_from_right = right * judge_fpr
    keep_wrong = wrong * (1 - judge_tpr)
    keep_right = right * (1 - judge_fpr)

    escalated = esc_from_wrong + esc_from_right
    # An escalated request is answered by the expensive model at its own accuracy.
    acc = keep_right + escalated * A_EXP
    cost = C_CHEAP + escalated * C_EXP
    return acc, cost, escalated


print("A two-model cascade. The cheap model is right %.0f%% of the time and costs"
      % (A_CHEAP * 100))
print("%.0f; the expensive model is right %.0f%% and costs %.0f."
      % (C_CHEAP, A_EXP * 100, C_EXP))
print()
print("Baselines, with no cascade at all:")
print()
print(f"{'policy':>22}{'accuracy':>11}{'cost':>9}")
print("-" * 42)
print(f"{'always cheap':>22}{A_CHEAP:>11.1%}{C_CHEAP:>9.2f}")
print(f"{'always expensive':>22}{A_EXP:>11.1%}{C_EXP:>9.2f}")

print()
print()
print("Now the cascade, as the judge gets better at spotting a wrong cheap answer.")
print("A perfect judge escalates every error and nothing else.")
print()
print(f"{'judge recall':>14}{'judge FPR':>12}{'accuracy':>11}{'cost':>9}"
      f"{'escalated':>12}{'vs always-exp':>15}")
print("-" * 73)
rows = {}
for tpr, fpr in ((1.00, 0.00), (0.90, 0.05), (0.75, 0.10),
                 (0.60, 0.20), (0.40, 0.30), (0.20, 0.45)):
    acc, cost, esc = cascade(tpr, fpr)
    rows[tpr] = (acc, cost, esc)
    # Positive means the cascade is cheaper AND at least as accurate.
    verdict = ("saves %.0f%%" % ((1 - cost / C_EXP) * 100)
               if acc >= A_EXP - 0.005 else "loses %.1f pts" % ((A_EXP - acc) * 100))
    print(f"{tpr:>14.0%}{fpr:>12.0%}{acc:>11.1%}{cost:>9.2f}{esc:>12.1%}"
          f"{verdict:>15}")

print()
print()
print("The accuracy the cascade gives up, and the money it saves, side by side.")
print("A cascade is only worth it if the first column is small enough to buy the")
print("second.")
print()
print(f"{'judge recall':>14}{'accuracy given up':>19}{'cost saved':>12}"
      f"{'pts per 1x saved':>18}")
print("-" * 63)
eff = {}
for tpr in sorted(rows, reverse=True):
    acc, cost, esc = rows[tpr]
    lost = A_EXP - acc
    saved = C_EXP - cost
    eff[tpr] = lost / saved if saved > 0 else float("inf")
    print(f"{tpr:>14.0%}{lost:>19.1%}{saved:>12.2f}{lost / saved:>18.4f}")

print()
print()
print("The threshold. Below some judge recall, the cascade is dominated: you could")
print("buy the same accuracy more cheaply by sending a random SHARE of traffic to")
print("the expensive model and skipping the judge entirely.")
print()
print(f"{'judge recall':>14}{'cascade acc':>13}{'cascade cost':>14}"
      f"{'random-split cost':>19}{'verdict':>14}")
print("-" * 74)


def random_split_cost(target_acc):
    """Cheapest way to hit an accuracy with no judge: send share p to expensive."""
    # acc = (1-p)*A_CHEAP + p*A_EXP  ->  p = (acc - A_CHEAP) / (A_EXP - A_CHEAP)
    p = (target_acc - A_CHEAP) / (A_EXP - A_CHEAP)
    p = max(0.0, min(1.0, p))
    return (1 - p) * C_CHEAP + p * C_EXP


beat = {}
for tpr in sorted(rows, reverse=True):
    acc, cost, esc = rows[tpr]
    rc = random_split_cost(acc)
    beat[tpr] = cost < rc
    print(f"{tpr:>14.0%}{acc:>13.1%}{cost:>14.2f}{rc:>19.2f}"
          f"{('cascade wins' if cost < rc else 'random wins'):>14}")

worst_win = min([t for t in beat if beat[t]], default=None)

print()
print()
print("The break-even: the judge recall at which the cascade first matches the")
print("expensive model's accuracy. Below it the cascade is a cost decision that")
print("costs accuracy; above it the cascade dominates on both axes.")
print()
print(f"{'judge recall':>14}{'judge FPR':>12}{'accuracy':>11}{'cost':>9}"
      f"{'vs always-exp':>16}")
print("-" * 62)
be = None
for i in range(101):
    tpr = i / 100.0
    fpr = tpr * 0.10          # a judge that catches more also over-escalates more
    acc, cost, esc = cascade(tpr, fpr)
    if be is None and acc >= A_EXP:
        be = (tpr, fpr, acc, cost)
for tpr in (0.50, 0.65, 0.80, 0.95):
    acc, cost, esc = cascade(tpr, tpr * 0.10)
    print(f"{tpr:>14.0%}{tpr * 0.10:>12.0%}{acc:>11.1%}{cost:>9.2f}"
          f"{(acc - A_EXP):>+16.1%}")
print()
print(f"break-even judge recall: {be[0]:.0%} (FPR {be[1]:.0%}), "
      f"accuracy {be[2]:.1%} at cost {be[3]:.2f}")
print(f"at break-even the cascade costs {be[3] / C_EXP:.0%} of always-expensive")

print(f"""
The two baselines set the range. Always-cheap costs {C_CHEAP:.2f} and is right
{A_CHEAP:.0%} of the time; always-expensive costs {C_EXP:.0f} and is right
{A_EXP:.0%}. Twenty times the money buys {(A_EXP - A_CHEAP) * 100:.0f} points.

Now the finding, which is not the one the cost-saving literature leads with. With a
**perfect** judge the cascade is right {rows[1.0][0]:.1%} of the time -- better than
the expensive model alone -- at a cost of {rows[1.0][1]:.2f}, which is
{rows[1.0][1] / C_EXP:.0%} of always-expensive.

That is not a cheaper way to get the expensive model's answer. It is a **better**
answer for a third of the price, because a perfect judge turns the pair into an
oracle: keep the cheap model's correct answers, escalate exactly its errors.
cite:chen2023frugalgpt's "+4% accuracy at the same cost" is this effect.

But the accuracy column collapses as the judge degrades. At {0.75:.0%} recall the
cascade is {rows[0.75][0]:.1%} -- it has given back the entire oracle bonus and is
now {(A_EXP - rows[0.75][0]) * 100:.1f} points BELOW always-expensive. At {0.40:.0%} recall
it is {rows[0.4][0]:.1%}, and at {0.20:.0%} recall it is {rows[0.2][0]:.1%} --
barely above the cheap model it started from, at {rows[0.2][1]:.2f} times the
cheap model's cost.

**The cascade's entire value lives in the judge** (eq:cascade-is-a-verifier-bet).
The cheap model's accuracy sets the ceiling; the judge decides how much of the gap
to the expensive model you actually capture, and it can capture more than all of it
or almost none.

The break-even table makes the operating requirement explicit. Coupling the false
positive rate to recall at a tenth -- a judge that catches more errors also
escalates more good answers -- the cascade first matches always-expensive at
**{be[0]:.0%} recall**, where it costs {be[3]:.2f}, or {be[3] / C_EXP:.0%} of the
expensive baseline.

So the deployment question has a number attached: **can you build an escalation
judge with better than {be[0]:.0%} recall on your own traffic?** If yes, the cascade
is close to free money. If no, you are paying for a judge to make you worse.

The last table is the check that stops a bad cascade from looking good. A cascade
that loses accuracy can always be compared against the trivial policy of sending a
random share of traffic to the expensive model -- no judge, no infrastructure. The
cascade beats that policy down to {0.40:.0%} recall, and loses at {0.20:.0%}, where
random-split buys the same {rows[0.2][0]:.1%} accuracy for {random_split_cost(rows[0.2][0]):.2f}
against the cascade's {rows[0.2][1]:.2f}.

**A cascade whose judge is worse than a coin flip is beaten by a coin flip**, and it
is beaten while costing more to build and more to run. That comparison belongs in
every routing design review, because it is cheap to compute and it is the one thing
that catches a cascade being justified by its cost column alone.""")
```

## 9. Practical Example

The baselines set the range. A cheap model right **72%** of the time costs 1.00; an
expensive model right **91%** costs 20.00. Twenty times the money buys 19 points.

Now the cascade as the judge degrades:

```
  judge recall   judge FPR   accuracy     cost   escalated  vs always-exp
-------------------------------------------------------------------------
          100%          0%      97.5%     6.60       28.0%      saves 67%
           90%          5%      94.6%     6.76       28.8%      saves 66%
           75%         10%      90.5%     6.64       28.2%  loses 0.5 pts
           60%         20%      86.0%     7.24       31.2%  loses 5.0 pts
           40%         30%      80.2%     7.56       32.8% loses 10.8 pts
           20%         45%      74.2%     8.60       38.0% loses 16.8 pts
```

With a perfect judge the cascade is right **97.5%** of the time — **better than the
expensive model alone** — at **6.60**, or 33% of always-expensive. That is the
oracle bonus: keep the cheap model's correct answers, escalate exactly its errors.
{{cite:chen2023frugalgpt}}'s "+4% accuracy at the same cost" is this effect.

The accuracy column then collapses. At 75% recall the cascade is **90.5%** — the
entire oracle bonus given back — and at 20% recall it is **74.2%**, barely above the
cheap model it started from, while costing 8.6 times the cheap model.

**The cascade's entire value lives in the judge** ({{eq:cascade-is-a-verifier-bet}}).
Coupling false positives to recall at a tenth, the cascade first matches
always-expensive at **77% recall**, costing 6.42 — **32%** of the expensive
baseline. That converts the design question into a buildable one: *can you get an
escalation judge above 77% recall on your own traffic?*

The random-split test catches the failure case:

```
  judge recall  cascade acc  cascade cost  random-split cost       verdict
--------------------------------------------------------------------------
          100%        97.5%          6.60              20.00  cascade wins
           90%        94.6%          6.76              20.00  cascade wins
           75%        90.5%          6.64              19.46  cascade wins
           60%        86.0%          7.24              14.99  cascade wins
           40%        80.2%          7.56               9.25  cascade wins
           20%        74.2%          8.60               3.18   random wins
```

At 20% recall, randomly splitting traffic buys the same **74.2%** accuracy for
**3.18** against the cascade's **8.60**. A cascade whose judge is worse than a coin
flip is beaten by a coin flip — while costing more to build and more to run.

The second listing turns to caching.

```python {tier=A name=bv2}
"""A semantic cache threshold is an error-cost decision, not a hit-rate decision.

ch:sd-architecture found caching survives at 19% under the three properties. What
survives is the utility; what breaks is the guarantee. A semantic cache restores
some of the utility by serving a stored answer when a NEW query is similar enough
to an old one -- and "similar enough" is a threshold somebody has to pick.

Loosening the threshold raises the hit rate and raises the rate of serving an answer
that was right for a different question. This listing finds where the optimum sits,
and what moves it (eq:cache-threshold-is-an-error-cost-decision).
"""
# Query stream: a similarity distribution against the cache, and the probability
# that a stored answer is still CORRECT for the new query at that similarity.
# (similarity band, share of queries, P(stored answer is right for this query))
BANDS = [
    (0.98, 0.06, 0.99),   # near-duplicate
    (0.94, 0.09, 0.96),
    (0.90, 0.12, 0.88),
    (0.86, 0.14, 0.74),
    (0.82, 0.16, 0.55),
    (0.78, 0.17, 0.34),
    (0.74, 0.26, 0.12),   # merely topical
]
C_MISS = 1.00      # cost of generating a fresh answer
C_HIT = 0.02       # cost of serving from cache
A_FRESH = 0.91     # a freshly generated answer is right this often


def evaluate(threshold, error_cost):
    """Serve from cache when similarity >= threshold. Returns
    (hit rate, mean cost, accuracy, total cost including errors)."""
    hits = 0.0
    right = 0.0
    spend = 0.0
    for sim, share, p_right in BANDS:
        if sim >= threshold:
            hits += share
            right += share * p_right
            spend += share * C_HIT
        else:
            right += share * A_FRESH
            spend += share * C_MISS
    total = spend + (1 - right) * error_cost
    return hits, spend, right, total


print("A semantic cache over a query stream. Loosening the similarity threshold")
print("buys hit rate and sells accuracy.")
print()
print(f"{'similarity band':>17}{'share of queries':>18}"
      f"{'stored answer right':>21}")
print("-" * 56)
for sim, share, p in BANDS:
    print(f"{sim:>17.2f}{share:>18.0%}{p:>21.0%}")

print()
print()
print("Sweeping the threshold. A miss costs %.2f and is right %.0f%% of the time;"
      % (C_MISS, A_FRESH * 100))
print("a hit costs %.2f and is right as often as the band allows." % C_HIT)
print()
print(f"{'threshold':>11}{'hit rate':>11}{'spend':>9}{'accuracy':>11}"
      f"{'vs no cache':>14}")
print("-" * 56)
sweep = {}
for t in (1.01, 0.98, 0.94, 0.90, 0.86, 0.82, 0.78, 0.74):
    hits, spend, acc, _ = evaluate(t, 0.0)
    sweep[t] = (hits, spend, acc)
    label = "no cache" if t > 1.0 else "%.2f" % t
    print(f"{label:>11}{hits:>11.0%}{spend:>9.3f}{acc:>11.1%}"
          f"{acc - A_FRESH:>+14.1%}")

print()
print()
print("Now price the errors. The optimum threshold is wherever total cost --")
print("generation plus the cost of a wrong answer -- is lowest. Three domains,")
print("differing ONLY in what a wrong answer costs.")
print()
DOMAINS = [
    ("suggesting a tag",       2.0),
    ("answering a support Q", 20.0),
    ("quoting a price",      200.0),
]
THRESHOLDS = [1.01, 0.98, 0.94, 0.90, 0.86, 0.82, 0.78, 0.74]
print(f"{'domain':>22}{'error cost':>12}" + "".join(f"{t:>8.2f}" if t <= 1.0
                                                    else f"{'none':>8}"
                                                    for t in THRESHOLDS))
print("-" * 98)
best = {}
for name, ec in DOMAINS:
    totals = {}
    for t in THRESHOLDS:
        totals[t] = evaluate(t, ec)[3]
    b = min(totals, key=lambda k: totals[k])
    best[name] = (b, totals[b], sweep[b][0])
    cells = "".join(f"{totals[t]:>8.2f}" for t in THRESHOLDS)
    print(f"{name:>22}{ec:>12.0f}{cells}")

print()
print(f"{'domain':>22}{'best threshold':>16}{'hit rate there':>16}"
      f"{'total cost':>12}")
print("-" * 66)
for name, ec in DOMAINS:
    b, tot, hr = best[name]
    label = "no cache" if b > 1.0 else "%.2f" % b
    print(f"{name:>22}{label:>16}{hr:>16.0%}{tot:>12.2f}")

print()
print()
print("What the hit-rate-maximising choice costs in each domain -- the mistake of")
print("tuning a cache on its own dashboard.")
print()
LOOSE = 0.74
print(f"{'domain':>22}{'best total':>12}{'loose total':>13}{'penalty':>10}"
      f"{'as multiple':>13}")
print("-" * 70)
pen = {}
for name, ec in DOMAINS:
    b, tot, hr = best[name]
    loose = evaluate(LOOSE, ec)[3]
    pen[name] = (loose - tot, loose / tot)
    print(f"{name:>22}{tot:>12.2f}{loose:>13.2f}{loose - tot:>10.2f}"
          f"{loose / tot:>13.2f}x")

print(f"""
The sweep table is the trade with no error pricing at all, and it looks like an
easy win: at threshold {0.74:.2f} the cache serves {sweep[0.74][0]:.0%} of queries
and spend falls from {sweep[1.01][1]:.3f} to {sweep[0.74][1]:.3f} -- a
{(1 - sweep[0.74][1] / sweep[1.01][1]):.0%} reduction. Any cost dashboard would
call that a success.

The accuracy column is what the cost dashboard does not show. At that same
threshold accuracy is {sweep[0.74][2]:.1%}, against {A_FRESH:.1%} with no cache --
a loss of {(A_FRESH - sweep[0.74][2]) * 100:.1f} points. This is
ch:sd-architecture's semantic failure arriving through the cache: every one of
those is a 200 response containing an answer to a question the user did not ask.

There is a second thing in that column worth pausing on, because it runs the other
way. At threshold {0.94:.2f} accuracy is {sweep[0.94][2]:.1%} -- **above** the
{A_FRESH:.1%} of generating every answer fresh. A near-duplicate's stored answer is
right {0.96:.0%} to {0.99:.0%} of the time, which is better than a fresh sample.

So a tight semantic cache is not a quality compromise made for cost reasons. It is
a quality **improvement** that also happens to cost less, and the mechanism is
ch:sd-architecture's nondeterminism turned around: when the same question is asked
twice, re-sampling is a fresh chance to be wrong, and serving the vetted stored
answer avoids it. The cache is acting as a stability layer.

That only holds while the threshold is tight. It is the interior of the curve that
matters, and both ends are worse than the middle.

Pricing the errors is what makes the decision well-posed, and the result is the
finding. **The optimal threshold is not a property of the cache. It is a property of
what a wrong answer costs** (eq:cache-threshold-is-an-error-cost-decision).

Across three domains that differ in nothing but error cost, the optimum moves from
`{best['suggesting a tag'][0] if best['suggesting a tag'][0] <= 1 else 'no cache'}`
to `{best['quoting a price'][0] if best['quoting a price'][0] <= 1 else 'no cache'}`:

  - suggesting a tag (error costs {2.0:.0f}): best threshold
    {best['suggesting a tag'][0]:.2f}, hit rate {best['suggesting a tag'][2]:.0%}
  - answering a support question (error costs {20.0:.0f}): best threshold
    {best['answering a support Q'][0]:.2f}, hit rate
    {best['answering a support Q'][2]:.0%}
  - quoting a price (error costs {200.0:.0f}): best threshold
    {best['quoting a price'][0]:.2f}, hit rate {best['quoting a price'][2]:.0%}

The same cache, the same query stream, the same similarity model -- and hit rates
{best['suggesting a tag'][2]:.0%}, {best['answering a support Q'][2]:.0%} and
{best['quoting a price'][2]:.0%}. A team that ships one threshold across a product
with all three surfaces has it wrong on at least two of them.

The last table prices that mistake. Tuning for hit rate -- taking the loosest
threshold, which any cache dashboard rewards -- costs
{pen['suggesting a tag'][1]:.2f} times optimal in the cheap-error domain,
{pen['answering a support Q'][1]:.2f} times in the middle one, and
{pen['quoting a price'][1]:.2f} times when a wrong answer costs {200.0:.0f}.

**The penalty for over-caching grows with the thing the cache dashboard cannot
see.** That is the same structural failure as ch:sd-architecture's availability
graph, one layer down: an instrument that is accurate about its own quantity and
silent about the one that decides whether the choice was right.

The practical rule this gives is short. Set the threshold per surface, from the
error cost of that surface, and hold the cache team accountable for total cost
rather than hit rate. A cache reporting {sweep[0.74][0]:.0%} hits is not reporting
good news until someone has priced the {(A_FRESH - sweep[0.74][2]) * 100:.1f} points
it gave up to get there.""")
```

Sweeping the similarity threshold with no error pricing looks like an easy win:

```
  threshold   hit rate    spend   accuracy   vs no cache
--------------------------------------------------------
   no cache         0%    1.000      91.0%         +0.0%
       0.98         6%    0.941      91.5%         +0.5%
       0.94        15%    0.853      91.9%         +0.9%
       0.90        27%    0.735      91.6%         +0.6%
       0.86        41%    0.598      89.2%         -1.8%
       0.82        57%    0.441      83.4%         -7.6%
       0.78        74%    0.275      73.7%        -17.3%
       0.74       100%    0.020      53.2%        -37.8%
```

At threshold 0.74 the cache serves **100%** of queries and spend falls **98%**. Any
cost dashboard calls that a success. Accuracy there is **53.2%** against 91.0% — a
loss of **37.8 points**, every one of them a 200 response answering a question the
user did not ask.

But note the top of the column. At threshold 0.94 accuracy is **91.9%**, *above* the
91.0% of generating everything fresh. A near-duplicate's stored answer is right 96%
to 99% of the time, better than a fresh sample. **A tight semantic cache is a
quality improvement that also costs less** — {{ch:sd-architecture}}'s nondeterminism
turned around, with the cache acting as a stability layer
({{eq:tight-cache-is-a-stability-layer}}).

Pricing errors makes the decision well-posed. Across three domains differing in
**nothing but error cost**:

```
                domain  best threshold  hit rate there  total cost
------------------------------------------------------------------
      suggesting a tag            0.82             57%        0.77
 answering a support Q            0.90             27%        2.42
       quoting a price            0.94             15%       16.99
```

Same cache, same query stream, same similarity model — and hit rates **57%**, **27%**
and **15%** ({{eq:cache-threshold-is-an-error-cost-decision}}).

```mermaid {#fig:threshold caption="Total cost is a U in the threshold. The minimum moves right as the cost of a wrong answer rises, so each product surface has its own optimum and hit rate is not the objective on any of them."}
flowchart TD
  A["loose threshold<br/>high hit rate"] --> B["spend falls"]
  A --> C["wrong answers rise"]
  D["tight threshold<br/>low hit rate"] --> E["spend rises"]
  D --> F["wrong answers fall"]
  B --> G["total cost"]
  C --> G
  E --> G
  F --> G
  G --> H["minimum moves right<br/>as error cost rises"]
```

Tuning for hit rate — what every cache dashboard rewards — costs **1.24 times**
optimal in the cheap-error domain, **3.87 times** in the middle, and **5.51 times**
when a wrong answer costs 200 ({{eq:hit-rate-is-not-the-objective}}). The penalty
for over-caching grows with exactly the quantity the cache dashboard cannot see.

## 10. Production Considerations

Compute $\tau^\star$ before building a cascade. It is arithmetic on four numbers you
either know or can measure cheaply, and it tells you whether the judge you can
actually build clears the bar. Teams routinely build the cascade first and discover
the bar afterwards.

Measure the judge separately, on labelled escalation decisions, and report recall
and false positive rate as first-class metrics beside cost. A cascade whose judge is
not independently measured is a cascade whose accuracy is unmonitored, and
{{eq:semantic-failure-has-no-instrument}} says nothing else will catch it.

Run the random-split comparison in every routing review. It is a few lines of
arithmetic and it is the only check that reliably catches a cascade justified by its
cost column alone.

Set cache thresholds per surface, not per system. The three domains in
{{sec:9-practical-example}} are one product in most companies — a tagging feature, a
support answer, and a price quote — and the correct thresholds differ by 12 points of
similarity and 42 points of hit rate.

Include everything the answer depended on in the cache key. Permissions and
retrieved-document identity especially; a semantic cache that ignores them is a data
leak with a latency benefit.

Version the cache alongside the prompt and the model. A prompt change alters the
distribution of correct answers, which invalidates stored entries in a way no TTL
expresses; entries generated under a previous prompt are not stale, they are wrong.
The cheapest correct policy is to include a prompt-and-model fingerprint in the key,
so a deploy naturally partitions the cache rather than silently reusing it.

Watch the escalation rate as an operational signal in its own right. It is the one
number that moves when either the traffic distribution or the judge shifts, and
because it sits directly upstream of spend it is usually already instrumented. A
rising escalation rate at flat traffic means the cheap model is being asked harder
questions; a falling one at flat traffic usually means the judge has degraded, which
is the more dangerous direction and the one nobody alerts on.

Hold cache and routing owners accountable for **total cost including errors**, not
for hit rate or spend. This is an organisational control, not a technical one, and
it is the one that determines whether the rest of this chapter gets applied.

## 11. Common Mistakes

**Treating the cascade as a two-model decision.** It is a three-component system and
the third one decides the outcome.

**Tuning the judge for recall alone.** False positives cost money *and* accuracy,
because the expensive model sometimes replaces a right answer with a wrong one.

**Reporting cascade savings without the accuracy column.** Every cascade saves money;
only some of them are worth it.

**Tuning a cache on hit rate.** Hit rate measures avoided work, not correct work.

**Shipping one similarity threshold across a whole product.** Correct on at most one
surface.

**Assuming a cache can only hurt quality.** A tight cache improves it, and teams that
believe otherwise leave the stability benefit on the table.

## 12. Failure Modes

**Judge drift.** The escalation judge's recall degrades as the traffic distribution
moves, with no error surfacing — the cascade silently converts into a cheap-model
deployment at cascade prices.

**Cache poisoning via incomplete keys.** An entry keyed only on query embedding is
served across permission boundaries or after the underlying documents changed. Grows
with hit rate.

**Threshold drift by dashboard.** Successive loosening, each justified by hit-rate
improvement, walks the threshold down the wrong side of the U in
{{fig:threshold}}.

**Cost-driven cascade collapse.** Under budget pressure the escalation rate is capped,
which is equivalent to reducing $\tau$ — accuracy falls with no configuration change
anyone recorded.

**Oracle-bonus illusion in evaluation.** Measuring a cascade with a judge that had
access to labels during evaluation reproduces the 97.5% row and ships a system that
performs at the 75% row.

## 13. Alternatives

**Single-model with a smaller model.** No judge, no infrastructure, and the accuracy
is what it is. Correct whenever $\tau^\star$ is out of reach.

**Random split.** The judge-free baseline from {{sec:9-practical-example}}. Its
existence is what makes the cascade's value measurable; occasionally it is also the
right answer.

**Difficulty prediction on the input.** Route before generating, using query
features rather than answer inspection. Cheaper than a judge (no cheap-model call is
wasted) but strictly less informed — it cannot see the answer, which is where the
evidence of difficulty actually is. {{cite:hu2024routerbench}} evaluates this family.

**Exact-match caching only.** Gives up the semantic hit rate and every risk that
comes with it. Reasonable for very high $\lambda$ surfaces, where
{{sec:6-mathematical-foundation}} shows $\theta^\star \to 1$ anyway.

**Speculative decoding.** Cheap-model-drafts-expensive-model-verifies at the token
level rather than the request level, with the crucial difference that verification
is **exact** rather than judged, so the oracle bonus is guaranteed rather than
purchased. Different mechanism, same intuition.

## 14. Evaluation

Report cascade accuracy and cost together, always. Either alone is misleading in a
predictable direction.

Evaluate the judge on held-out labelled escalations, and report $\tau$ and $\phi$
separately. A single "judge accuracy" number hides the asymmetry that drives the
economics.

Compare every cascade against its random-split equivalent and publish the
comparison. If the cascade does not beat it, the finding is the deliverable.

For caches, report total cost including a priced error term, and state the $\lambda$
used. A cache evaluation without a stated $\lambda$ has not evaluated anything —
{{eq:cache-threshold-is-an-error-cost-decision}} says the answer is undefined until
$\lambda$ is fixed.

Track $r(s)$ — stored-answer correctness by similarity band — from sampled review.
It is the parameter the whole cache decision rests on and the one most often assumed
rather than measured.

## 15. Advanced Concepts

The two-model cascade generalises to $n$ tiers with a judge between each, and the
economics compound in both directions: each tier's oracle bonus adds, and each
judge's error rate multiplies. With independent judges of recall $\tau$, the
probability an error survives $k$ tiers is $(1-\tau)^k$, which argues for more tiers
— but the false-positive cost accumulates linearly in escalations, which argues for
fewer. The optimum is usually two or three.

The independence assumption between judge errors and model errors is optimistic. A
judge built from the same family as the cheap model tends to be confident exactly
where the cheap model is confidently wrong, so $\tau$ measured on adversarial cases
is well below $\tau$ measured on average traffic. This is
{{eq:agent-errors-correlate}}'s correlation appearing in the routing layer.

The cache and the cascade interact. A cache hit is an escalation the cascade never
sees, so caching changes the difficulty distribution reaching the cheap model —
generally making it *harder*, since easy repeated queries are absorbed by the cache.
A cascade tuned before a cache is deployed will have its judge operating
off-distribution afterwards, and $\tau^\star$ needs recomputing.

## 16. Connection to Previous Chapters

{{eq:retry-needs-a-verifier}} from {{ch:ag-recovery}} is this chapter's spine.
Escalation is retry with a price tag, and the same verifier requirement governs it.

{{eq:three-properties-break-the-stack}} from {{ch:sd-architecture}} said caching
survives at 19%. {{eq:cache-threshold-is-an-error-cost-decision}} is what the 19%
looks like when it is engineered properly, and
{{eq:tight-cache-is-a-stability-layer}} recovers more than the 19% suggested.

{{eq:semantic-failure-has-no-instrument}} explains why both failure modes here are
silent: cost dashboards and hit-rate dashboards are accurate and irrelevant.

{{eq:model-belongs-interleaved}} determines where each kind of cache goes —
conventional caching in the deterministic gaps, semantic caching on the model stages.

## 17. Exercises

1. Compute $\tau^\star$ for $a_c = 0.85$, $a_e = 0.93$, $\gamma = 0.1$. Is a cascade
   easier or harder to justify when the cheap model is already good?

2. Extend the first listing to a three-tier cascade. At what judge recall does the
   third tier stop paying for itself?

3. Derive the $\lambda$ at which the optimal cache threshold for the listing's bands
   reaches 0.98. What kind of product surface has that error cost?

4. Modify the second listing so cache entries expire, making $r(s)$ decay with age.
   How does $\theta^\star$ move?

5. Build the random-split comparison for a routing system you have access to. Does
   it clear the bar?

## 18. Interview Questions

1. A team reports their cascade cut inference cost 70%. What is the first question
   you ask?

2. Why can a cascade with a perfect judge beat the expensive model on accuracy?

3. Your semantic cache has a 60% hit rate and the team is proud of it. Under what
   circumstances is that bad news?

4. Two product surfaces share a cache. One quotes prices, one suggests tags. What is
   wrong with that architecture?

5. How does deploying a cache in front of a cascade change the cascade's tuning?

## 19. Research Questions

1. How well can escalation judges be built from signals that cost nothing extra
   (logits, output structure) versus signals that cost a second generation?

2. Is there a judge architecture whose errors are uncorrelated with the cheap
   model's, and what does decorrelation buy in $\tau$ on adversarial traffic?

3. Can $r(s)$ — stored-answer correctness by similarity — be predicted from query
   and corpus properties rather than measured per deployment?

4. Does the two-or-three-tier optimum hold when tier costs are geometric rather than
   linear, as they are across current model families?

## 20. Chapter Summary

Routing and caching are the two obvious responses to an expensive model, and both
are governed by a quantity their dashboards do not show.

A cascade is a bet on a judge ({{eq:cascade-is-a-verifier-bet}}). With a perfect
judge it is right **97.5%** against the expensive model's **91%**, at **33%** of the
cost — an oracle-gated ensemble, not a cheaper way to get the same answer. At 75%
judge recall the entire bonus is gone; at 20% the cascade is beaten by randomly
splitting traffic, **8.60** against **3.18** for the same accuracy. The break-even is
**77% recall**, and it is computable before anything is built.

A semantic cache's threshold is an error-cost decision
({{eq:cache-threshold-is-an-error-cost-decision}}). Across three domains differing
only in what a wrong answer costs, optimal hit rates are **57%**, **27%** and
**15%**. Tuning for hit rate instead costs **1.24×**, **3.87×** and **5.51×** optimal
({{eq:hit-rate-is-not-the-objective}}).

And the counterintuitive one: a **tight** cache raises accuracy to **91.9%** above
fresh generation's **91.0%**, because serving a vetted stored answer avoids a fresh
chance to be wrong ({{eq:tight-cache-is-a-stability-layer}}).

Carry forward: **compute the break-even before building the cascade**, and **set the
cache threshold from the surface's error cost, never from its hit rate**.

## 21. Further Reading

- {{cite:chen2023frugalgpt}} — the cascade result, and the source of the oracle-bonus
  effect measured here.
- {{cite:hu2024routerbench}} — 405,000+ precomputed outcomes; the benchmark that made
  routing policies comparable offline.
- {{cite:kwon2023pagedattention}} — serving-layer memory management, which sets the
  real cost ratio the routing arithmetic depends on.
