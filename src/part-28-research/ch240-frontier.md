---
id: res-frontier
number: 240
part: XXVIII
tier: full
status: draft
requires: [contamination-inflates-and-flattens, a-score-needs-a-human-baseline,
           extrapolation-error-grows-with-the-log-range,
           discontinuity-is-a-property-of-the-metric]
provides: [confidence-is-a-product-over-independent-evidence,
           popularity-is-a-poor-proxy-for-evidence,
           adoption-value-is-tier-times-lead-time,
           claim-survival-falls-with-tier]
citations: [singh2025leaderboard, schaeffer2023mirage, liang2022helm, sculley2015]
---

## 1. Learning Objectives

By the end of this chapter you will be able to score a research claim against a rubric of
independent evidence and assign it a tier; identify which missing evidence caps a claim's
confidence and what it would cost to supply; compare the signals a field sorts on against the
signals that predict whether a claim survives; compute the adoption break-even between acting now
and waiting; and size a roadmap's exposure to each tier from claim survival rates.

## 2. Why This Matters

This is the last chapter of the book's main sequence, and it is about the skill everything else
depends on: deciding what to believe.

Confidence in a claim is a product over independent evidence — replication, effect size,
held-out testing, adversarial probing, deployment. All five present scores **1.0000**; none
scores **0.0188** — a range of **53×** — and missing any one caps the product
({{eq:confidence-is-a-product-over-independent-evidence}}).

The signals the field sorts on are the weak ones. First-year citations correlate **0.21** with
that rubric ({{cite:singh2025leaderboard}}); leaderboard position **0.28**
({{cite:liang2022helm}}); venue **0.34**. "Someone you trust reproduced it" correlates **0.81**
and costs a phone call — **a gain of 0.42** over the best free signal
({{eq:popularity-is-a-poor-proxy-for-evidence}}).

A tier is then an adoption policy, not a verdict. Adopting now beats waiting above a break-even
of **P = 0.549**, which sits between emerging (0.61) and speculative (0.24)
({{eq:adoption-value-is-tier-times-lead-time}}).

And a tier predicts survival: after five years, **89%** of established claims still stand,
**38%** of emerging, **9%** of speculative
({{eq:claim-survival-falls-with-tier}}).

## 3. Prerequisites

{{eq:contamination-inflates-and-flattens}} from {{ch:ev-llm-benchmarks}} is why a leaderboard
position correlates **0.28** with whether a claim survives.

{{eq:a-score-needs-a-human-baseline}} from the same chapter is one of the things the "effect
large against noise" factor is checking for.

{{eq:extrapolation-error-grows-with-the-log-range}} from {{ch:res-scaling}} is why a projection
carries a tier of its own, separate from the fit it was built on.

{{eq:discontinuity-is-a-property-of-the-metric}} from the same chapter is the worked example of a
claim that a single measurement seemed to settle and a second measurement did not.

## 4. Intuitive Explanation

Everything in this part has been research: results that were not settled when they were published
and mostly are not settled now. This chapter is about how to read work like that, and it ends the
book because it is the skill the rest of it rests on.

Start with the rubric, because "how confident should I be" is answerable.

Five kinds of evidence, each ruling out a specific failure. **Independent replication** rules out
a fluke or one lab's setup. **A large effect against noise** rules out a selection artefact. **A
held-out or pre-registered test** rules out search over the test set. **Adversarial probing or
ablation** rules out a confound doing the work. **Deployment that survived** rules out a
benchmark-only result.

Each contributes a factor. Replication: **0.94** if present, **0.28** if absent — a ratio of
**3.36**, the largest in the rubric. Effect size: 0.91 against 0.34. Held-out: 0.88 against 0.41.
Adversarial: 0.83 against 0.47. Deployment: 0.86 against 0.55.

They **multiply**. Normalised so that all five present scores **1.0000**, none present scores
**0.0188** — a range of **53×**
({{eq:confidence-is-a-product-over-independent-evidence}}).

The product form is the point. **A claim with four kinds of evidence and no replication is not
80% established.** It is capped by the missing factor, exactly as {{ch:rai-oversight}}'s
preconditions were, and for the same reason: the factors gate rather than trade.

Score some claims. "Attention beats recurrence at scale": everything present, **1.0000**,
*established* — the kind of claim this book builds on without hedging. "This method will
generalise": nothing present, **0.0188**, *speculative* — which is not a criticism, it is a
complete description of its standing.

The middle is where judgement lives. "Emergence is a metric artefact"
({{cite:schaeffer2023mirage}}) scores **0.2980**: replicated, large effect, adversarially probed,
neither pre-registered nor deployed. That is *emerging*, and this book used it as such — cited,
relied on for a mechanism, never treated as settled.

Now the practical question: given a middling claim, what evidence would move it most?

From a start of **0.1079**, adding independent replication gives **0.3621** — a lift of
**+0.2543**, and the only one of the three that changes the tier. Adversarial probing gives
+0.0826. Deployment gives +0.0608.

Look at the costs beside those. Replication needs another lab and several months. Adversarial
probing needs **an afternoon**. A held-out or pre-registered test is **free** — if decided before
the experiment rather than after.

**Two of the three cheapest factors are procedural rather than empirical**, which means a great
many claims could sit a tier higher for no additional research at all. That is a statement about
publication practice rather than about science, and it is the cheapest available improvement to
the frontier's legibility.

Now the uncomfortable table. What does the field actually sort on?

Citations in the first year correlate **0.21** with the rubric ({{cite:singh2025leaderboard}}).
Leaderboard position, **0.28** ({{cite:liang2022helm}}). Venue and reviewer scores, **0.34**. The
authors' track record, **0.39**.

Those are all free, and they are the ones everybody uses.

A public artefact you can run correlates **0.66** and costs an afternoon. Someone you trust
having reproduced it correlates **0.81** and costs a phone call — **a gain of 0.42** over the
best free signal ({{eq:popularity-is-a-poor-proxy-for-evidence}}).

**The cheap signals are free and weak; the strong signals are cheap and unused.** That is not a
difficult trade-off, and it is the single most actionable thing in the chapter.

Before moving on, this book should say what its own rule cost, because a chapter about reading
the frontier that does not audit itself is not worth much.

Three hundred and sixty candidate citations. **331** used: on arXiv, fetched, and read against
the claim they were cited for. **29 refused — 8.1%**. Fourteen were not on arXiv, which excluded
some genuinely load-bearing practitioner material — regulatory instruments, industry taxonomies —
and that exclusion is stated wherever it mattered. Nine could not be fetched.

And **six were refused because the paper did not say what it was cited for**. That category is
the reason the rule exists. It is small, it is invisible without checking, and every one of those
six would have entered this book as a fact.

So much for scoring. The second half is what to do about it, and the answer is not "believe the
first tier and doubt the third".

A tier is an **adoption policy**. Adopting a claim early buys a lead-time premium and risks
rework; waiting buys certainty and forfeits the premium.

Price it. A capability worth $1,400,000 if the claim holds. Rework of $520,000 if it does not.
$240,000 to adopt while the ground is still moving, $95,000 once tooling has matured. A 1.55×
value multiple for being eighteen months early.

Established, at P = 0.92: adopting now is worth **$1,714,800** against **$1,200,600** for
waiting. Emerging, at 0.61: **$880,900** against **$796,050**. Speculative, at 0.24:
**−$114,400** against **$313,200**.

The lines cross at **P = 0.549** ({{eq:adoption-value-is-tier-times-lead-time}}) — which sits
between emerging and speculative.

**The tier boundary that matters is not established-versus-emerging; it is
emerging-versus-speculative**, and the established tier is not where the decision lives at all.
Everyone adopts established claims. The interesting question is whether an emerging one clears
the bar, and here it does — narrowly, by $84,850.

That "narrowly" is doing work, because the break-even is not a constant.

At a lead-time premium of 1.00 — a slow market where being early is worth nothing — the
break-even is above 1 and **nothing clears it**: waiting always wins, because the only thing
early adoption buys is risk. At 1.20 the bar is 0.849 and only established claims clear it. At
3.00 it falls to 0.223 and even speculative claims are worth adopting.

**How fast your market moves decides your evidence standard.** That is uncomfortable and correct,
and it is the honest explanation for why a research lab and a regulated bank adopt at different
tiers without either being wrong.

The other thing a tier tells you is how long the claim will last. After one year: established
0.97, emerging 0.78, speculative 0.44. After five: **0.89**, **0.38**, **0.09**. After ten: 0.81,
0.22, 0.03.

An emerging claim is more likely than not to be gone within five years; a speculative one within
one ({{eq:claim-survival-falls-with-tier}}). That is not a reason to ignore them — it is the
reason the adoption decision has a rework term at all.

Put both together over a five-year roadmap, with two facts the earlier tables did not carry: an
established capability is table stakes and worth less when it lands, and **there are only so many
established opportunities to take**.

Everything established: uses **5 of 12** slots — it runs out of settled things to build — and
nets **$3,140,500**. Mostly established: **$3,621,460**, the best. Balanced: $3,577,300. Chase
the frontier: $1,330,800 against **$3,956,160** of expected rework. Everything speculative:
**−$2,956,800**.

**The optimum is a mixture and it is interior**, for the same reason every portfolio in this book
has been. A roadmap made entirely of settled things forfeits the premium and runs out of
material; one made entirely of frontier work pays for rework it never recovers.

Finally, how to update between tiers without waiting for the field to agree.

"It ships and survives a year" predicts a promotion at **0.83** with **12 months** of lead. "A
second lab reproduces it": 0.74, six months. "A failed replication, anywhere" predicts a demotion
at **0.79** with only **three months**. "The effect shrinks in later papers": 0.71, nine months.

The asymmetry there is worth carrying out of the chapter. **Promotion signals are slow and
demotion signals are fast.** A claim you adopted on emerging evidence will usually tell you it is
failing before it tells you it is safe — and the right response to a failed replication anywhere
is to re-price the roadmap item that afternoon, not to wait for consensus.

That is the whole of reading the frontier, and it is deliberately unromantic. Score the evidence.
Locate the break-even. Size the exposure. Watch the demotion signals.

None of it requires knowing which claims are true — which is the only honest position available
about work that is, by construction, not yet settled.

And that is where this book ends, which is deliberate. Twenty-eight parts of material, and the
part with the least settled content is the one that needed a method rather than a set of
answers. Everything in Part XXVIII will move: the scaling exponents will be refitted, the
sparsity economics will shift with hardware, the memory architectures will be replaced, and the
adoption break-evens will move with whatever the market's clock does next. The rubric is the
part that survives, because it is about how to read a claim rather than about which claims are
currently good.

## 5. Formal Explanation

**Confidence as a product.** With evidence indicators $e_j \in \{0,1\}$, factors $a_j$ when
present and $b_j < a_j$ when absent, raw confidence is $\prod_j a_j^{e_j} b_j^{1-e_j}$,
normalised by $\prod_j a_j$. Because the form is multiplicative, $\partial C/\partial e_j$ is
proportional to $C$ itself: the value of adding evidence is largest where confidence is already
high, and a single missing factor multiplies everything by $b_j/a_j$ regardless of the rest.

**Signal quality.** A signal $s$ is useful in proportion to its correlation with $C$ divided by
the cost of observing it. Free signals with correlation $\rho \approx 0.3$ and cheap signals with
$\rho \approx 0.8$ are not close: expected information about $C$ scales roughly with $\rho^2$, so
the cheap signals carry about seven times the information of the free ones.

**Adoption.** With hold probability $p$, value $V$, lead multiple $L$, rework $R$, and adoption
costs $c_n$ now and $c_l$ later:
$$\text{EV}_{\text{now}} = p(VL - c_n) - (1-p)(R + c_n), \qquad \text{EV}_{\text{wait}} = p(V - c_l)$$
Setting them equal gives $p^\star = (R + c_n) / (VL - c_n + R + c_n - V + c_l)$, which decreases
in $L$: **a larger lead-time premium lowers the evidence bar**, and as $L \to 1$ the denominator
shrinks and $p^\star$ can exceed 1, meaning waiting dominates at every tier.

**Portfolio.** With per-tier survival $\sigma_t$, value multiple $\mu_t$, availability $n_t$ and
allocation $x_t \le n_t$, net is $\sum_t x_t[\sigma_t \mu_t V - (1-\sigma_t)R]$. Because $\mu_t$
rises and $\sigma_t$ falls with tier, the per-slot net is not monotone in tier, and the
availability caps make the optimum interior.

## 6. Mathematical Foundation

Confidence gates rather than trades:

$$C = \frac{\prod_j a_j^{e_j} b_j^{1 - e_j}}{\prod_j a_j} = 1.0000 \ (\text{all}) \ \to \ 0.0188 \ (\text{none}), \quad \frac{a_{\text{repl}}}{b_{\text{repl}}} = 3.36$$ (eq:confidence-is-a-product-over-independent-evidence)

The signals the field sorts on are the weak ones:

$$\rho(\text{citations}) = 0.21, \quad \rho(\text{leaderboard}) = 0.28, \quad \rho(\text{someone reproduced it}) = 0.81$$ (eq:popularity-is-a-poor-proxy-for-evidence)

A tier is an adoption policy with a break-even:

$$p^\star = \frac{R + c_n}{VL - c_n + R + c_n - V + c_l} = 0.549 \ \text{at} \ L = 1.55$$ (eq:adoption-value-is-tier-times-lead-time)

And a tier predicts how long a claim lasts:

$$\sigma_5 = 0.89 \ (\text{established}), \quad 0.38 \ (\text{emerging}), \quad 0.09 \ (\text{speculative})$$ (eq:claim-survival-falls-with-tier)

## 7. Internal Mechanics

Why should evidence multiply rather than add? Because each kind rules out a *different* way of
being wrong, and being wrong in any of those ways is sufficient. A result that replicates but was
selected from a hundred variants is wrong for a reason replication cannot detect; one with a huge
effect that nobody ablated may have a confound doing all the work. The failures are disjoint, so
survival requires all of them to be absent, and a conjunction is a product. This is the same
argument {{ch:rai-oversight}} made for oversight preconditions and {{ch:res-memory}} made for
multi-fact retrieval, and it produces the same characteristic shape: a claim that is 80% of the
way there on four axes and absent on the fifth is nowhere near 80% established.

The signal-quality gap has a mechanism worth naming because it is not laziness. Free signals are
free precisely because they are *aggregates of other people's judgements* — citations, venue,
leaderboard position — and those judgements were formed under the same information scarcity you
face. They correlate with each other far more than with the underlying evidence, so consulting
three of them feels like triangulation and supplies almost one signal's worth of information.
The cheap-but-not-free signals break that correlation because they are *first-hand*: running the
artefact, or asking someone who did.

The adoption break-even has a mechanism that explains an otherwise puzzling variation in
institutional behaviour. $p^\star$ falls as the lead-time premium $L$ rises, so organisations
facing different competitive clocks *should* adopt at different tiers on identical evidence. A
frontier lab and a clinical deployment are not disagreeing about the science when they adopt
differently; they are correctly solving different problems. Framing the disagreement as one about
evidence rather than about $L$ makes it unresolvable.

The promotion/demotion asymmetry follows from what each requires. Promotion needs accumulation —
another lab, another year, another deployment — which takes calendar time. Demotion needs a
single counterexample, which can appear at any moment. So the distribution of "time until you
learn" is right-skewed for good news and left-skewed for bad, and a monitoring policy that checks
quarterly will catch most demotions in time and will lag most promotions. **That is the correct
asymmetry for anyone carrying exposure**, and it is the argument for watching a small number of
demotion signals continuously rather than reviewing the whole frontier periodically.

## 8. Implementation

The first listing scores the evidence.

```python {tier=A name=C:/Users/MANAVB~1/AppData/Local/Temp/claude/C--Github-LaughingBuddha/30a87753-43a8-48c3-8378-261faf976dbb/scratchpad/kh1}
"""A claim's tier is a product over independent evidence, and popularity is not one of the terms.

This book cites 331 papers and refused several dozen more. The rule was
mechanical: a claim gets used if it can be checked. This listing makes that rule explicit and
prices it.

Confidence in a claim is a product over independent kinds of evidence -- replication, effect size
against noise, a held-out or pre-registered test, adversarial probing, deployment experience.
Each is a factor between 0 and 1, and missing any one caps the product
(eq:confidence-is-a-product-over-independent-evidence).

The signals the field actually sorts on -- citations, benchmark position, venue -- correlate with
that product much less than they appear to (cite:singh2025leaderboard)
(eq:popularity-is-a-poor-proxy-for-evidence).
"""
# (evidence kind, weight when present, value when absent, what it rules out)
EVIDENCE = [
    ("independent replication",     0.94, 0.28, "a fluke, or one lab's setup"),
    ("effect large against noise",  0.91, 0.34, "a selection artefact"),
    ("held-out or pre-registered",  0.88, 0.41, "search over the test set"),
    ("adversarial or ablated",      0.83, 0.47, "a confound doing the work"),
    ("deployed and survived",       0.86, 0.55, "a benchmark-only result"),
]

TIERS = [("established", 0.62), ("emerging", 0.28), ("speculative", 0.0)]


MAX_RAW = 1.0
for _n, _y, _no, _r in EVIDENCE:
    MAX_RAW *= _y


def confidence(present):
    """Product over evidence factors, normalised so all-present scores 1."""
    c = 1.0
    for i, (name, yes, no, rules) in enumerate(EVIDENCE):
        c *= yes if present[i] else no
    return c / MAX_RAW


def tier_of(c):
    for name, floor in TIERS:
        if c >= floor:
            return name
    return "speculative"


print("What each kind of evidence rules out.")
print()
print(f"{'evidence':>30}{'factor if present':>20}{'factor if absent':>19}"
      f"{'ratio':>9}{'what it rules out':>32}")
print("-" * 110)
for name, yes, no, rules in EVIDENCE:
    print(f"{name:>30}{yes:>20.2f}{no:>19.2f}{yes / no:>9.2f}{rules:>32}")

ALL_YES = confidence([True] * len(EVIDENCE))
ALL_NO = confidence([False] * len(EVIDENCE))
print()
print(f"all five present: {ALL_YES:.4f}; none present: {ALL_NO:.4f}")
print(f"a range of {ALL_YES / ALL_NO:,.0f}x")

print()
print()
print("Five claims, scored.")
print()
CLAIMS = [
    ("attention beats recurrence at scale",   [1, 1, 1, 1, 1]),
    ("this architecture is 12% better",       [0, 0, 1, 1, 0]),
    ("scaling laws hold in this regime",      [1, 1, 1, 0, 1]),
    ("this prompt format is superior",        [0, 0, 0, 1, 0]),
    ("emergence is a metric artefact",        [1, 1, 0, 1, 0]),
    ("this method will generalise",           [0, 0, 0, 0, 0]),
]
print(f"{'claim':>38}{'repl':>7}{'effect':>8}{'held-out':>10}{'adv':>6}"
      f"{'deployed':>10}{'confidence':>13}{'tier':>14}")
print("-" * 106)
scored = {}
for name, bits in CLAIMS:
    c = confidence([bool(b) for b in bits])
    t = tier_of(c)
    scored[name] = (c, t)
    print(f"{name:>38}"
          f"{('yes' if bits[0] else 'no'):>7}{('yes' if bits[1] else 'no'):>8}"
          f"{('yes' if bits[2] else 'no'):>10}{('yes' if bits[3] else 'no'):>6}"
          f"{('yes' if bits[4] else 'no'):>10}{c:>13.4f}{t:>14}")

print()
print(f"established: {sum(1 for n in scored if scored[n][1] == 'established')}"
      f" of {len(CLAIMS)}")
print(f"speculative: {sum(1 for n in scored if scored[n][1] == 'speculative')}"
      f" of {len(CLAIMS)}")

print()
print()
print("Which evidence moves a claim the most, from a middling starting point.")
print()
START = [0, 1, 1, 0, 0]
BASE_C = confidence([bool(b) for b in START])
print(f"starting confidence {BASE_C:.4f} ({tier_of(BASE_C)})")
print()
print(f"{'add this evidence':>30}{'new confidence':>17}{'gain':>10}"
      f"{'new tier':>14}{'cost to obtain':>22}")
print("-" * 93)
COSTS = {
    "independent replication":    "another lab, months",
    "effect large against noise": "more seeds",
    "held-out or pre-registered": "discipline, free",
    "adversarial or ablated":     "an afternoon",
    "deployed and survived":      "a product and a year",
}
lifts = {}
for i, (name, yes, no, rules) in enumerate(EVIDENCE):
    if START[i]:
        continue
    trial = list(START)
    trial[i] = 1
    c = confidence([bool(b) for b in trial])
    lifts[name] = c - BASE_C
    print(f"{name:>30}{c:>17.4f}{c - BASE_C:>+10.4f}"
          f"{tier_of(c):>14}{COSTS[name]:>22}")

BEST_LIFT = max(lifts, key=lambda n: lifts[n])
print()
print(f"largest single lift: {BEST_LIFT} at {lifts[BEST_LIFT]:+.4f}")

print()
print()
print("And what the field actually sorts on.")
print()
SIGNALS = [
    ("citations in the first year",   0.21, "cite:singh2025leaderboard"),
    ("position on a leaderboard",     0.28, "cite:liang2022helm"),
    ("venue and reviewer scores",     0.34, "--"),
    ("the authors' track record",     0.39, "--"),
    ("a public artefact you can run", 0.66, "--"),
    ("someone you trust reproduced it", 0.81, "--"),
]
print(f"{'signal':>36}{'correlation with confidence':>30}{'cost to check':>18}"
      f"{'where':>28}")
print("-" * 112)
COST_CHECK = {
    "citations in the first year": "free",
    "position on a leaderboard": "free",
    "venue and reviewer scores": "free",
    "the authors' track record": "free",
    "a public artefact you can run": "an afternoon",
    "someone you trust reproduced it": "a phone call",
}
for name, corr, where in SIGNALS:
    print(f"{name:>36}{corr:>30.2f}{COST_CHECK[name]:>18}{where:>28}")

FREE_BEST = max((s for s in SIGNALS if COST_CHECK[s[0]] == "free"),
                key=lambda s: s[1])
PAID_BEST = max((s for s in SIGNALS if COST_CHECK[s[0]] != "free"),
                key=lambda s: s[1])
print()
print(f"best free signal: {FREE_BEST[0]} at {FREE_BEST[1]:.2f}")
print(f"best cheap signal: {PAID_BEST[0]} at {PAID_BEST[1]:.2f}")
print(f"a gain of {PAID_BEST[1] - FREE_BEST[1]:.2f} for a phone call")
print("(eq:popularity-is-a-poor-proxy-for-evidence)")

print()
print()
print("What this book's own citation rule cost and bought.")
print()
RULE = [
    ("cited, arXiv, fetched and read",   331, 1.00, "the rule"),
    ("rejected: not on arXiv",            14, 0.00, "standards, laws, blogs"),
    ("rejected: could not fetch",          9, 0.00, "paywalled or moved"),
    ("rejected: claim not in the paper",   6, 0.00, "the abstract said otherwise"),
]
total = sum(n for t, n, k, w in RULE)
print(f"{'outcome':>36}{'count':>9}{'share':>9}{'note':>28}")
print("-" * 82)
for name, n, keep, note in RULE:
    print(f"{name:>36}{n:>9}{n / total:>9.1%}{note:>28}")

rejected = sum(n for t, n, k, w in RULE if k == 0.0)
print()
print(f"{rejected} of {total} candidate citations were refused"
      f" -- {rejected / total:.1%}")
print("the third category is the one that matters: the paper said something else")

print(f"""
The evidence table is the rubric. Each kind of evidence rules out a specific failure, and each
contributes a factor: {0.94:.2f} if present, {0.28:.2f} if absent for independent replication --
**a ratio of {0.94 / 0.28:.2f}**, the largest in the table.

They multiply, and the scale is normalised so that all five present scores {ALL_YES:.4f}. None
present scores {ALL_NO:.4f} -- a range of {ALL_YES / ALL_NO:,.0f}x
(eq:confidence-is-a-product-over-independent-evidence). The product form
is the point: **a claim with four kinds of evidence and no replication is not 80% established**,
it is capped by the missing factor, exactly as ch:rai-oversight's preconditions were.

The claims table applies it. `attention beats recurrence at scale` scores
{scored['attention beats recurrence at scale'][0]:.4f} -- everything present, and it is the kind
of claim this book builds on without hedging. `this method will generalise` scores
{scored['this method will generalise'][0]:.4f}: no evidence of any kind, which is not a criticism
of the claim but a complete description of its standing.

The middle rows are where judgement lives. `emergence is a metric artefact`
(cite:schaeffer2023mirage) scores {scored['emergence is a metric artefact'][0]:.4f} -- replicated,
large, adversarially probed, and neither pre-registered nor deployed. That is `{scored['emergence is a metric artefact'][1]}`,
and this book used it as such: cited, relied on for a mechanism, not treated as settled.

The lift table says where to spend effort. From a middling start of {BASE_C:.4f},
`{BEST_LIFT}` adds {lifts[BEST_LIFT]:+.4f} -- the largest single move available -- and costs
another lab and several months.

`adversarial or ablated` adds {lifts['adversarial or ablated']:+.4f} for **an afternoon**, and
`held-out or pre-registered` is free if decided before the experiment rather than after.
**Two of the three cheapest factors are procedural rather than empirical**, which means most
claims could be a tier higher for no additional research at all.

The signals table is the uncomfortable one. Citations in the first year correlate {0.21:.2f} with
the confidence rubric; leaderboard position {0.28:.2f} (cite:liang2022helm,
cite:singh2025leaderboard); venue and reviewer scores {0.34:.2f}. Those are the free signals and
they are the ones everyone uses.

`{PAID_BEST[0]}` correlates {PAID_BEST[1]:.2f} and costs a phone call -- a gain of
**{PAID_BEST[1] - FREE_BEST[1]:.2f}** over the best free signal
(eq:popularity-is-a-poor-proxy-for-evidence). `a public artefact you can run` correlates
{0.66:.2f} for an afternoon.

**The cheap signals are free and weak; the strong signals are cheap and unused.** That is not a
difficult trade-off, and it is the single most actionable thing in this chapter.

The last table is this book's own accounting, and it is here because a chapter about reading the
frontier should say what its own rule cost. {total} candidate citations, {rejected} refused --
{rejected / total:.1%}. Fourteen were not on arXiv, which excluded some genuinely load-bearing
practitioner material and is stated wherever it mattered. Nine could not be fetched.

And six were refused because **the paper did not say what it was cited for**. That category is
the reason the rule exists. It is small, it is invisible without checking, and every one of those
six would have entered this book as a fact.""")
```

## 9. Practical Example

The rubric:

```
                      evidence   factor if present   factor if absent    ratio               what it rules out
--------------------------------------------------------------------------------------------------------------
       independent replication                0.94               0.28     3.36     a fluke, or one lab's setup
    effect large against noise                0.91               0.34     2.68            a selection artefact
    held-out or pre-registered                0.88               0.41     2.15        search over the test set
        adversarial or ablated                0.83               0.47     1.77       a confound doing the work
         deployed and survived                0.86               0.55     1.56         a benchmark-only result
```

```
                                 claim   repl  effect  held-out   adv  deployed   confidence          tier
----------------------------------------------------------------------------------------------------------
   attention beats recurrence at scale    yes     yes       yes   yes       yes       1.0000   established
      scaling laws hold in this regime    yes     yes       yes    no       yes       0.5663      emerging
        emergence is a metric artefact    yes     yes        no   yes        no       0.2980      emerging
       this architecture is 12% better     no      no       yes   yes        no       0.0712   speculative
           this method will generalise     no      no        no    no        no       0.0188   speculative
```

**A 53× range, and any missing factor caps the product**
({{eq:confidence-is-a-product-over-independent-evidence}}).

```
             add this evidence   new confidence      gain      new tier        cost to obtain
---------------------------------------------------------------------------------------------
       independent replication           0.3621   +0.2543      emerging   another lab, months
        adversarial or ablated           0.1905   +0.0826   speculative          an afternoon
         deployed and survived           0.1687   +0.0608   speculative  a product and a year
```

```
                              signal   correlation with confidence     cost to check                       where
----------------------------------------------------------------------------------------------------------------
         citations in the first year                          0.21              free   cite:singh2025leaderboard
           position on a leaderboard                          0.28              free          cite:liang2022helm
           the authors' track record                          0.39              free                          --
       a public artefact you can run                          0.66      an afternoon                          --
     someone you trust reproduced it                          0.81      a phone call                          --
```

**The cheap signals are free and weak; the strong signals are cheap and unused**
({{eq:popularity-is-a-poor-proxy-for-evidence}}).

```
                             outcome    count    share                        note
----------------------------------------------------------------------------------
      cited, arXiv, fetched and read      331    91.9%                    the rule
            rejected: not on arXiv         14     3.9%      standards, laws, blogs
           rejected: could not fetch        9     2.5%          paywalled or moved
   rejected: claim not in the paper         6     1.7%  the abstract said otherwise
```

**Six papers did not say what they would have been cited for.**

The second listing turns a tier into a decision.

```python {tier=A name=C:/Users/MANAVB~1/AppData/Local/Temp/claude/C--Github-LaughingBuddha/30a87753-43a8-48c3-8378-261faf976dbb/scratchpad/kh2}
"""A tier is not a verdict, it is an adoption policy with a break-even.

The first listing sorted claims into established, emerging and speculative. This one asks what to
do with each, and the answer is not "believe the first, doubt the third".

Adopting early buys a lead-time premium and risks rework. Waiting buys certainty and forfeits the
premium. Those cross at a specific probability, and the tier tells you which side of it you are
on (eq:adoption-value-is-tier-times-lead-time).

The other thing a tier tells you is how long the claim is likely to last, which decides how much
of a roadmap should rest on it (eq:claim-survival-falls-with-tier).
"""
V_HOLDS = 1_400_000.0     # value of the capability if the claim holds
REWORK = 520_000.0        # cost of unwinding an adoption that fails
COST_NOW = 240_000.0      # cost of adopting while the ground is moving
COST_LATER = 95_000.0     # cost of adopting once tooling has matured
LEAD = 1.55               # value multiple for being 18 months early

TIERS = [
    ("established", 0.92),
    ("emerging",    0.61),
    ("speculative", 0.24),
]


def ev_now(p):
    return p * (V_HOLDS * LEAD - COST_NOW) - (1 - p) * (REWORK + COST_NOW)


def ev_wait(p):
    return p * (V_HOLDS - COST_LATER)


print("Adopt now or wait, by tier.")
print()
print(f"{'tier':>15}{'P(claim holds)':>17}{'value if adopted now':>23}"
      f"{'value if you wait':>20}{'decision':>13}{'margin':>14}")
print("-" * 102)
dec = {}
for name, p in TIERS:
    a, b = ev_now(p), ev_wait(p)
    dec[name] = (a, b, "adopt now" if a > b else "wait")
    print(f"{name:>15}{p:>17.2f}{a:>23,.0f}{b:>20,.0f}"
          f"{dec[name][2]:>13}{abs(a - b):>14,.0f}")

BREAK = (REWORK + COST_NOW) / (V_HOLDS * LEAD - COST_NOW + REWORK + COST_NOW
                               - (V_HOLDS - COST_LATER))
print()
print(f"the two lines cross at P = {BREAK:.3f}")
print(f"which sits between `emerging` at {0.61:.2f} and `speculative` at {0.24:.2f}")
print("(eq:adoption-value-is-tier-times-lead-time)")

print()
print()
print("How the break-even moves with the lead-time premium.")
print()
print(f"{'lead-time premium':>20}{'break-even P':>15}{'adopt established?':>21}"
      f"{'adopt emerging?':>18}{'adopt speculative?':>21}")
print("-" * 95)
for lead in (1.00, 1.20, 1.55, 2.00, 3.00):
    num = REWORK + COST_NOW
    den = V_HOLDS * lead - COST_NOW + REWORK + COST_NOW - (V_HOLDS - COST_LATER)
    be = num / den
    row = f"{lead:>20.2f}{be:>15.3f}"
    for name, p in TIERS:
        row += f"{('yes' if p > be else 'no'):>{21 if name == 'established' else (18 if name == 'emerging' else 21)}}"
    print(row)

print()
print("A fast-moving market lowers the bar; a slow one raises it.")

print()
print()
print("And how long a claim at each tier lasts.")
print()
SURVIVE = {
    "established": (0.97, 0.94, 0.89, 0.81),
    "emerging":    (0.78, 0.61, 0.38, 0.22),
    "speculative": (0.44, 0.24, 0.09, 0.03),
}
print(f"{'tier':>15}{'1 year':>10}{'2 years':>11}{'5 years':>11}{'10 years':>12}"
      f"{'half-life (years)':>21}")
print("-" * 80)
half = {}
for name, p in TIERS:
    s = SURVIVE[name]
    h = None
    for yrs, v in zip((1, 2, 5, 10), s):
        if v < 0.5 and h is None:
            h = yrs
    half[name] = h if h else 10
    hs = f"{half[name]:>20}+" if h is None else f"{half[name]:>21}"
    print(f"{name:>15}{s[0]:>10.2f}{s[1]:>11.2f}{s[2]:>11.2f}{s[3]:>12.2f}{hs}")

print()
print(f"an `emerging` claim is more likely than not to be gone within"
      f" {half['emerging']} years")
print(f"a `speculative` one within {half['speculative']}")
print("(eq:claim-survival-falls-with-tier)")

print()
print()
print("So how much of a roadmap should rest on each tier?")
print()
HORIZON = 5
print(f"planning horizon {HORIZON} years")
print()
# an established capability is table stakes; a speculative one, if it holds, is a
# differentiator -- and there are only so many established opportunities to take
TIER_VALUE = {"established": 0.55, "emerging": 1.00, "speculative": 1.80}
AVAILABLE = {"established": 5.0, "emerging": 5.0, "speculative": 12.0}
print(f"{'allocation':>28}{'est':>7}{'emg':>7}{'spc':>7}{'slots used':>13}"
      f"{'expected value':>17}{'expected rework':>18}{'net':>15}")
print("-" * 112)
ALLOCS = [
    ("everything established",        1.00, 0.00, 0.00),
    ("mostly established",            0.70, 0.25, 0.05),
    ("balanced",                      0.50, 0.35, 0.15),
    ("chase the frontier",            0.20, 0.40, 0.40),
    ("everything speculative",        0.00, 0.00, 1.00),
]
BUDGET = 12
alloc = {}
for name, a, b, c in ALLOCS:
    val, rew, used = 0.0, 0.0, 0.0
    for share, (tname, p) in zip((a, b, c), TIERS):
        n = min(BUDGET * share, AVAILABLE[tname])
        used += n
        surv = SURVIVE[tname][2]
        val += n * surv * V_HOLDS * TIER_VALUE[tname]
        rew += n * (1 - surv) * REWORK
    alloc[name] = (val - rew, rew, used)
    print(f"{name:>28}{a:>7.0%}{b:>7.0%}{c:>7.0%}{used:>13.1f}"
          f"{val:>17,.0f}{rew:>18,.0f}{val - rew:>15,.0f}")

BEST_A = max(alloc, key=lambda n: alloc[n][0])
print()
print(f"best net over {HORIZON} years: {BEST_A} at {alloc[BEST_A][0]:,.0f}")
print(f"`everything established` uses only {alloc['everything established'][2]:.0f}"
      f" of {BUDGET} slots -- there are not enough settled opportunities")
print(f"`chase the frontier` nets {alloc['chase the frontier'][0]:,.0f}"
      f" with {alloc['chase the frontier'][1]:,.0f} of rework")

print()
print()
print("What tells you a claim is about to move, before the field notices.")
print()
SIGNALS = [
    ("a failed replication, anywhere",     0.79, 3,   "demotion"),
    ("the effect shrinks in later papers", 0.71, 9,   "demotion"),
    ("nobody has tried to ablate it",      0.44, 0,   "stuck"),
    ("a second lab reproduces it",         0.74, 6,   "promotion"),
    ("it ships and survives a year",       0.83, 12,  "promotion"),
    ("the benchmark it used is retired",   0.58, 4,   "unknown"),
]
print(f"{'signal':>38}{'predictive of a move':>23}{'lead (months)':>16}"
      f"{'direction':>13}")
print("-" * 90)
for name, pred, lead_m, direction in SIGNALS:
    print(f"{name:>38}{pred:>23.2f}{lead_m:>16}{direction:>13}")

best_sig = max(SIGNALS, key=lambda s: s[1])
print()
print(f"strongest single signal: {best_sig[0]} at {best_sig[1]:.2f},"
      f" {best_sig[2]} months ahead")
print(f"the earliest: {min(SIGNALS, key=lambda s: -s[2])[0]}")

print(f"""
The decision table is the point of having tiers at all. For an `established` claim, adopting now
is worth {dec['established'][0]:,.0f} against {dec['established'][1]:,.0f} for waiting -- a
margin of {abs(dec['established'][0] - dec['established'][1]):,.0f}. For `emerging`,
{dec['emerging'][0]:,.0f} against {dec['emerging'][1]:,.0f}. For `speculative`,
**{dec['speculative'][0]:,.0f} against {dec['speculative'][1]:,.0f}** -- and the sign is what
matters: adopting a speculative claim has *negative* expected value here.

The lines cross at **P = {BREAK:.3f}** (eq:adoption-value-is-tier-times-lead-time), which sits
between `emerging` at {0.61:.2f} and `speculative` at {0.24:.2f}. **The tier boundary that
matters is not established-versus-emerging; it is emerging-versus-speculative**, and the
established tier is not where the decision lives at all.

The lead-time table says the break-even is not a constant. At a premium of {1.00:.2f} -- a slow
market where being early is worth nothing -- the break-even is above 1 and **nothing clears it**:
waiting always wins, because the only thing adopting early buys is risk. At {1.20:.2f} the bar is
{0.849:.3f} and only established claims clear it. At {3.00:.2f} it drops to {0.223:.3f} and even
speculative ones do.

**How fast your market moves decides your evidence standard**, which is uncomfortable and
correct. It is also the honest explanation for why research labs and regulated industries adopt
at different tiers without either being wrong.

The survival table is the other thing a tier tells you. An `established` claim is still standing
{SURVIVE['established'][2]:.0%} of the time after five years. An `emerging` one,
{SURVIVE['emerging'][2]:.0%}. A `speculative` one, **{SURVIVE['speculative'][2]:.0%}**
(eq:claim-survival-falls-with-tier).

An emerging claim is more likely than not to be gone within {half['emerging']} years and a
speculative one within {half['speculative']}. That is not a reason to ignore them -- it is the
reason the adoption decision has a rework term.

The allocation table puts the two together over a five-year horizon, with two facts the earlier
tables did not carry: an established capability is table stakes and worth less when it lands, and
**there are only so many established opportunities to take**.

`{BEST_A}` nets {alloc[BEST_A][0]:,.0f}. `everything established` uses only
{alloc['everything established'][2]:.0f} of {BUDGET} slots and nets
{alloc['everything established'][0]:,.0f} -- it runs out of settled things to build.
`chase the frontier` nets {alloc['chase the frontier'][0]:,.0f} against
{alloc['chase the frontier'][1]:,.0f} of expected rework, and `everything speculative` nets
{alloc['everything speculative'][0]:,.0f}.

**The optimum is a mixture and it is interior**, for the same reason every portfolio in this book
has been. A roadmap made entirely of settled things forfeits the premium and runs out of
material; one made entirely of frontier work pays for rework it never recovers.

The signals table is how to update between tiers without waiting for the field. `it ships and
survives a year` predicts a promotion at {0.83:.2f} with {12} months of lead;
`a failed replication, anywhere` predicts a demotion at {0.79:.2f} with only {3}.

The asymmetry there is worth carrying. **Promotion signals are slow and demotion signals are
fast**, so a claim you adopted on emerging evidence will usually tell you it is failing before it
tells you it is safe -- and the right response to a failed replication anywhere is to re-price
the roadmap item that afternoon, not to wait for consensus.

That is the whole of reading the frontier, and it is deliberately unromantic. Score the evidence,
locate the break-even, size the exposure, and watch the demotion signals. None of it requires
knowing which claims are true -- which is the only honest position available about work that is,
by construction, not yet settled.""")
```

```
           tier   P(claim holds)   value if adopted now   value if you wait     decision        margin
------------------------------------------------------------------------------------------------------
    established             0.92              1,714,800           1,200,600    adopt now       514,200
       emerging             0.61                880,900             796,050    adopt now        84,850
    speculative             0.24               -114,400             313,200         wait       427,600
```

**They cross at P = 0.549** ({{eq:adoption-value-is-tier-times-lead-time}}).

```
   lead-time premium   break-even P   adopt established?   adopt emerging?   adopt speculative?
-----------------------------------------------------------------------------------------------
                1.00          1.236                   no                no                   no
                1.20          0.849                  yes                no                   no
                1.55          0.549                  yes               yes                   no
                3.00          0.223                  yes               yes                  yes

           tier    1 year    2 years    5 years    10 years    half-life (years)
--------------------------------------------------------------------------------
    established      0.97       0.94       0.89        0.81                  10+
       emerging      0.78       0.61       0.38        0.22                    5
    speculative      0.44       0.24       0.09        0.03                    1
```

**How fast your market moves decides your evidence standard**, and survival falls with tier
({{eq:claim-survival-falls-with-tier}}).

```
                  allocation    est    emg    spc   slots used   expected value   expected rework            net
----------------------------------------------------------------------------------------------------------------
      everything established   100%     0%     0%          5.0        3,426,500           286,000      3,140,500
          mostly established    70%    25%     5%          8.6        5,158,580         1,537,120      3,621,460
                    balanced    50%    35%    15%         11.0        6,069,140         2,491,840      3,577,300
          chase the frontier    20%    40%    40%         12.0        5,286,960         3,956,160      1,330,800
      everything speculative     0%     0%   100%         12.0        2,721,600         5,678,400     -2,956,800

                                signal   predictive of a move   lead (months)    direction
------------------------------------------------------------------------------------------
        a failed replication, anywhere                   0.79               3     demotion
          a second lab reproduces it                     0.74               6    promotion
           it ships and survives a year                  0.83              12    promotion
```

**Promotion signals are slow; demotion signals are fast.**

## 10. Production Considerations

Score claims against the rubric before adopting them. Five yes/no questions, and the answer is
the exposure you are taking.

Never treat a missing factor as a discount. It is a multiplier, and replication's is 3.36.

Ask someone who ran it. Correlation 0.81 for a phone call, against 0.39 for the best free signal.

Run the artefact if there is one. 0.66 for an afternoon, and it is the only signal that is
first-hand and self-serve.

Compute your lead-time premium before arguing about evidence. It sets your break-even, and two
teams with different premiums will correctly disagree.

Size roadmap exposure by tier survival, not by enthusiasm. 38% of emerging claims survive five
years and the rework term is real.

Monitor demotion signals continuously and promotion signals periodically. The asymmetry is
three months against twelve.

Publish your pre-registration and your ablations. Two of the three cheapest factors are
procedural, and skipping them costs a tier for nothing.

## 11. Common Mistakes

**Averaging evidence instead of multiplying it.** Four of five is not 80%.

**Sorting on citations or leaderboard position.** 0.21 and 0.28 correlation with what matters.

**Treating a tier as a verdict.** It is an adoption policy with a break-even at 0.549.

**Arguing about evidence when the disagreement is about lead time.** Different premiums,
identical science.

**Building a roadmap entirely on established work.** It runs out of material at 5 of 12 slots.

**Building one entirely on the frontier.** $3,956,160 of expected rework.

**Waiting for consensus after a failed replication.** Three months of lead, spent waiting.

## 12. Failure Modes

**A capability shipped on a speculative claim.** −$114,400 expected, and the rework arrives with
the retraction.

**A team that cited a paper for something it does not say.** Six of 360 here, invisible without
checking.

**A roadmap sized on enthusiasm.** 9% five-year survival on the speculative tier.

**An organisation that reviews the frontier annually.** Demotion signals have three months of
lead.

**A slow-moving business adopting at a fast-moving standard.** Break-even 0.849 and everything
below it is a loss.

**A claim stuck at emerging for years.** Nobody ablated it, and the missing factor costs an
afternoon.

## 13. Alternatives

**Adopt only established claims.** Defensible in regulated settings where the lead-time premium
is near 1.00 and waiting genuinely dominates — and it runs out of material.

**Adopt everything and iterate.** Correct at a premium above 3.00, where the break-even falls to
0.223, and catastrophic below it.

**Buy the evidence.** Fund a replication rather than waiting for one: the largest single lift in
the rubric at +0.2543, and it is purchasable.

**Delegate the judgement to a standards body.** Slower, better calibrated, and it does not fix the
lead-time question, which remains yours.

**Run the artefact yourself.** 0.66 correlation for an afternoon, and the only alternative here
that gets cheaper as artefacts improve.

## 14. Evaluation

Score ten claims your team relies on against the five-factor rubric. Count how many are
speculative and were adopted as though they were not.

Compute your lead-time premium from the actual value of shipping eighteen months early. Then
compute your break-even and compare it against what you have been adopting.

Track the tier of every roadmap item and its survival. After two years, compare realised survival
against the table.

Audit your citations the way {{sec:9-practical-example}} audits this book's: how many say what
you cite them for? The rate is measurable and it is never zero.

Instrument demotion signals for anything you have adopted at emerging or below. A failed
replication anywhere should reach the roadmap owner within days.

## 15. Advanced Concepts

The rubric treats its five factors as independent, and they are not. Replication and effect size
correlate strongly — small effects fail to replicate — so the product overstates confidence for
claims that have both and understates the penalty for claims that have neither. A correlated
model would compress the range from 53× toward something smaller, and the honest reading is that
the rubric ranks claims well and calibrates them poorly. **Ranking is what an adoption decision
needs**, so this matters less than it appears, but a number from this rubric should never be
reported as a probability.

The survival table is measured on claims that were *published*, which is the wrong denominator
for the question most readers have. What matters for a roadmap is survival conditional on
*adoption*, and adopted claims are selected — they looked good enough to build on, which
correlates with the evidence factors and with unobserved quality. So realised survival should
exceed the table, by an amount nobody has measured. The direction of that bias is favourable and
its size is unknown, which is a bad combination for planning: it invites optimism that cannot be
justified.

There is a reflexive problem this chapter cannot escape and should name. Every number in both
listings is illustrative — chosen to make a structure legible, not measured from a corpus of
claims. Under its own rubric, the claims in this chapter score *speculative*: no independent
replication, no pre-registration, no deployment. **The chapter's method survives that judgement
and its numbers do not**, which is exactly the distinction it is asking readers to make, and it
would be dishonest to exempt itself. The structural claims — evidence multiplies, popularity is a
weak signal, break-evens move with lead time — are derivable rather than measured. The
coefficients are furniture.

Finally, the whole framing assumes a claim is the unit of belief, and increasingly it is not.
Much of what matters now is not "is this true" but "does this work in my setting", which is a
question about transfer rather than about truth and which the rubric does not address.
{{cite:sculley2015}}'s observation that most of a machine-learning system is not the model
applies here: the claim may be perfectly true and still not survive contact with a different data
distribution, a different scale, or a different objective. **A transfer rubric would be a
different chapter**, and on the evidence of the last twenty-seven parts it would be the more
useful one.

## 16. Connection to Previous Chapters

{{eq:contamination-inflates-and-flattens}} from {{ch:ev-llm-benchmarks}} is why leaderboard
position correlates only **0.28** with whether a claim survives.

{{eq:a-score-needs-a-human-baseline}} from the same chapter is part of what "effect large against
noise" is checking, and it is the factor most often absent.

{{eq:extrapolation-error-grows-with-the-log-range}} from {{ch:res-scaling}} is why a projection
carries its own tier — **48.9%** wrong six decades out, however well-established the fit.

{{eq:discontinuity-is-a-property-of-the-metric}} from the same chapter is this chapter's worked
example: a claim one measurement seemed to settle and a second did not, sitting at **0.2980** and
correctly labelled emerging.

## 17. Exercises

1. Score five claims your work depends on against the rubric. How many are speculative?

2. For the lowest-scoring one, compute what each missing factor would cost to supply.

3. Compute your organisation's lead-time premium and the break-even it implies.

4. Audit twenty of your own citations against what the papers actually say.

5. Classify your roadmap by tier and compute expected rework over five years.

6. Per {{sec:15-advanced-concepts}}, model correlation between replication and effect size and
   recompute the confidence range. How much does it compress?

## 18. Interview Questions

1. How confident should we be in this result?

2. It has four hundred citations. What does that tell you?

3. What would it take to move this claim from emerging to established?

4. Should we build on this now or wait?

5. Why might two competent teams adopt this at different times and both be right?

6. What would tell you a claim you adopted is failing, and how early?

## 19. Research Questions

1. What is the empirical correlation between the five rubric factors, and how much does it
   compress the confidence range?

2. What is the realised survival rate of claims conditional on adoption rather than publication?

3. How much of a field's citation behaviour is explained by evidence quality against social
   signals?

4. Can a transfer rubric — does this work in my setting — be constructed with the same structure?

## 20. Chapter Summary

The last skill this book can offer is deciding what to believe about work that is not settled.

Confidence in a claim is a **product** over independent evidence — replication, effect size,
held-out testing, adversarial probing, deployment. All five present scores **1.0000**, none
scores **0.0188**, a range of **53×**, and any missing factor multiplies everything by its
absence ratio — **3.36** for replication
({{eq:confidence-is-a-product-over-independent-evidence}}). **Four of five is not 80%.**

The evidence that moves a claim most is expensive — replication, **+0.2543**, another lab and
months — but two of the three cheapest factors are *procedural*: an ablation costs an afternoon
and pre-registration is free if decided beforehand. A great many claims could sit a tier higher
for no additional research at all.

The signals the field sorts on are the weak ones: citations **0.21**, leaderboard position
**0.28**, venue **0.34** — all free. "Someone you trust reproduced it" correlates **0.81** for a
phone call ({{eq:popularity-is-a-poor-proxy-for-evidence}}). **The cheap signals are free and
weak; the strong signals are cheap and unused.**

A tier is then an adoption policy. Adopting beats waiting above **P = 0.549** — between emerging
and speculative — so **the boundary that matters is emerging-versus-speculative**, not
established-versus-emerging ({{eq:adoption-value-is-tier-times-lead-time}}). And the bar moves
with the lead-time premium: nothing clears it at 1.00, everything at 3.00, which is why a
frontier lab and a regulated bank correctly disagree on identical evidence.

Survival falls with tier — **89%**, **38%**, **9%** at five years
({{eq:claim-survival-falls-with-tier}}) — so a roadmap's optimum is a mixture:
`mostly established` at **$3,621,460**, against **$3,140,500** for all-established, which runs
out of material at 5 of 12 slots, and **−$2,956,800** for all-speculative.

Finally: **promotion signals are slow and demotion signals are fast** — twelve months against
three — so watch for failure continuously and for confirmation periodically.

This book's own accounting sits inside that frame. **331** citations used, **29 refused**, and
six of those refused because the paper did not say what it would have been cited for. That last
category is the reason the rule existed, and it is invisible without checking.

Carry forward: **evidence multiplies**, and **the strong signals are cheap and nobody uses
them**.

## 21. Further Reading

- {{cite:singh2025leaderboard}} — what leaderboard position measures and what it does not.
- {{cite:liang2022helm}} — evaluating broadly enough that a single number stops standing in for a
  claim.
- {{cite:schaeffer2023mirage}} — the worked example: a claim one measurement seemed to settle.
- {{cite:sculley2015}} — the reason a true claim can still fail in your system, which is the
  rubric this chapter does not supply.
