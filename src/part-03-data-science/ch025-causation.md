---
id: ds-causation
number: 25
part: III
tier: focused
status: reviewed
requires: [ds-eda, math-covariance, math-probability]
provides: [confounding, simpsons-paradox, causal-effect, counterfactual,
           randomisation, collider-bias]
citations: [simpson1951]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. State the fundamental problem of causal inference and why it is
   fundamental.
2. Define a causal effect in terms of potential outcomes.
3. Explain confounding precisely, and identify confounders in a described
   scenario.
4. Construct Simpson's paradox and explain why no statistical rule resolves it.
5. Explain why randomisation licenses a causal claim, and prove it.
6. Recognise collider bias and explain why controlling for the wrong variable
   creates a spurious association.
7. Decide which variables to adjust for, and justify the choice.
8. State what can and cannot be concluded from observational data.

## 2. Why This Matters

This is the most consequential chapter in Part III.

Almost every data-science claim that turns out to be wrong is a causal claim
supported only by an association. "Users who engage with feature X retain
better, so we should promote X." "Customers contacted by sales convert more, so
we should contact more." "The model shows that having a premium plan reduces
churn." Each of these may be true. None of them follow from the data cited.

The reason is not statistical subtlety. It is that **the data does not contain
the information required**. An association is compatible with the causal story
you have in mind, with the reverse causal story, and with a third variable
producing both. Distinguishing them requires knowledge of how the data was
generated — which is not in the table ({{ch:ds-what-it-is}}).

Simpson's paradox {{cite:simpson1951}} is the sharpest demonstration. It
produces two arithmetically correct answers that point in opposite directions,
and no purely statistical criterion selects between them. What selects between
them is a claim about causal structure, supplied by you.

This matters commercially because the actions taken on causal claims are
expensive. Promoting a feature that correlates with retention because engaged
users find it, rather than because it causes retention, spends a roadmap on
nothing.

## 3. Prerequisites

{{ch:math-covariance}} for correlation and the fact that it measures linear
association only. {{ch:math-probability}} for conditional probability, which is
the language of confounding. {{ch:ds-eda}} for the disaggregation habit, which
this chapter takes to its conclusion.

## 4. Intuitive Explanation

### 4.1 The fundamental problem

A {{term:causal-effect}} is a comparison between what happened and what would
have happened otherwise, **for the same unit**.

For user $i$, let $Y_i(1)$ be the outcome if they receive the treatment and
$Y_i(0)$ if they do not. The individual causal effect is $Y_i(1) - Y_i(0)$.

You can never observe both. The user either received the treatment or did not,
and the other value — the {{term:counterfactual}} — is permanently
unavailable. This is the **fundamental problem of causal inference**, and it is
not a limitation of your data collection. It is logical.

```text
                 Y(0)      Y(1)      effect
  treated user     ?        14         ?
  control user     9         ?         ?
                   ▲         ▲
              observed   observed — but never both for the same person
```

Everything in causal inference is a strategy for estimating an *average* effect
despite never observing an individual one.

### 4.2 Why comparing groups usually fails

The obvious move is to compare treated users to untreated ones:

$$
\E[Y \mid T=1] - \E[Y \mid T=0]
$$

This equals the causal effect only if the two groups are comparable in every
other respect. They usually are not, because **something decided who got
treated**, and that something is often related to the outcome.

Users who adopted the new feature are more engaged. Customers contacted by sales
were selected because they looked promising. Patients given the aggressive
treatment were sicker. In each case the treated and untreated groups differ
before the treatment, and the observed difference mixes the treatment's effect
with that pre-existing difference.

### 4.3 Confounding

A {{term:confounding}} variable causes both the treatment and the outcome:

```mermaid {#fig:confounding caption="A confounder Z causes both treatment and outcome, opening a non-causal path between them. The observed association mixes the true effect with the path through Z."}
graph LR
  Z[Z: engagement] --> T[T: uses feature X]
  Z --> Y[Y: retention]
  T -->|the effect we want| Y
  style Z fill:#fde68a,stroke:#ca8a04
```

The association between $T$ and $Y$ now has two sources: the causal arrow
$T \to Y$, and the **back-door path** $T \leftarrow Z \to Y$. The observed
correlation sums both, and no amount of data separates them without knowing $Z$
exists.

If you can measure $Z$, you can adjust for it — compare treated and untreated
users *within* levels of engagement. If you cannot, or do not know it exists,
the estimate is biased by an unknown amount.

### 4.4 Three explanations for every correlation

Whenever $X$ and $Y$ are associated, at least four explanations are live:

1. $X$ causes $Y$ — the one you had in mind.
2. $Y$ causes $X$ — reverse causation. Do people buy more because they are
   loyal, or become loyal because they bought more?
3. $Z$ causes both — confounding.
4. Coincidence, or selection into the sample ({{ch:ds-collection}}).

The data is equally consistent with all four. Ruling out the alternatives
requires either an experiment, or an argument from how the world works.

## 5. Formal Explanation

### 5.1 Potential outcomes

For each unit $i$ define $Y_i(1)$ and $Y_i(0)$. The **average treatment effect**
is

$$
\text{ATE} = \E[Y(1) - Y(0)] = \E[Y(1)] - \E[Y(0)]
$$ (eq:ate)

using linearity of expectation ({{ch:math-random-vars}}) — note that this holds
even though the two terms are never observed on the same unit.

The observed difference in means is

$$
\Delta_{\text{obs}} = \E[Y \mid T=1] - \E[Y \mid T=0]
  = \E[Y(1) \mid T=1] - \E[Y(0) \mid T=0]
$$ (eq:observed-difference)

The second equality uses **consistency**: the observed outcome for a treated
unit is its treated potential outcome.

Adding and subtracting $\E[Y(0) \mid T=1]$ decomposes the observed difference:

$$
\Delta_{\text{obs}} =
\underbrace{\E[Y(1) - Y(0) \mid T=1]}_{\text{effect on the treated}}
+ \underbrace{\E[Y(0) \mid T=1] - \E[Y(0) \mid T=0]}_{\text{selection bias}}
$$ (eq:causal-decomposition)

{{eq:causal-decomposition}} is the central identity of the chapter. The second
term compares the *untreated* outcomes of the two groups — how they would have
differed anyway. It is zero exactly when the groups are comparable, and it is
what confounding makes nonzero.

### 5.2 Why randomisation works

{{term:randomisation}} assigns $T$ by a coin flip, independently of everything
else. Formally:

$$
T \perp \{Y(0), Y(1)\}
$$ (eq:ignorability)

Treatment is independent of the potential outcomes. Therefore

$$
\E[Y(0) \mid T=1] = \E[Y(0) \mid T=0] = \E[Y(0)]
$$ (eq:randomisation-consequence)

and the selection-bias term in {{eq:causal-decomposition}} is exactly zero. The
observed difference *is* the average treatment effect.

> IMPORTANT: The power of randomisation is that {{eq:ignorability}} holds for
> **unmeasured** variables too. You do not need to know what the confounders
> are, or even that they exist. This is why a randomised experiment is
> qualitatively stronger evidence than any observational analysis, however
> sophisticated — and why {{ch:ds-experiments}} exists.

### 5.3 Adjustment, and what it requires

Without randomisation, you can still identify the effect **if** you can measure
every confounder. Under conditional ignorability,
$T \perp \{Y(0), Y(1)\} \mid Z$, the adjustment formula gives

$$
\E[Y(t)] = \sum_{z} \E[Y \mid T=t, Z=z]\,\Prob(Z=z)
$$ (eq:adjustment-formula)

Compare within strata of $Z$, then average the strata weighted by their
population frequency — *not* by their frequency within each treatment group,
which is the mistake that produces Simpson's paradox.

The assumption is untestable. "I measured all the confounders" is a claim about
the world, and no diagnostic on the data can verify it.

### 5.4 Colliders: when adjusting makes things worse

The instinct that controlling for more variables is safer is wrong.

A **collider** is a variable caused by both $X$ and $Y$. Conditioning on it
*creates* an association that did not exist:

```mermaid {#fig:collider caption="A collider is a common effect. X and Y are independent, but conditioning on C — by controlling for it, or by selecting on it — induces a spurious association between them."}
graph LR
  X[X: talent] --> C[C: admitted to the programme]
  Y[Y: connections] --> C
  style C fill:#fecaca,stroke:#c94b4b
```

Suppose talent and connections are independent in the population, and either one
suffices for admission. Among the admitted, someone lacking connections must be
talented — so within the admitted group the two are *negatively* correlated. The
association is manufactured entirely by conditioning.

{{term:collider-bias}} is why "control for everything" is bad advice. Whether a
variable should be adjusted for depends on its causal role:

{#tbl:adjustment-rules caption="Whether to adjust for a variable depends on its causal position, which is not determinable from the data."}

| Role of Z | Adjust? | Why |
|---|---|---|
| Confounder (Z → T, Z → Y) | **yes** | closes the back-door path |
| Collider (T → Z ← Y) | **no** | conditioning creates a spurious path |
| Mediator (T → Z → Y) | **no**, for the total effect | removes part of the effect you want |
| Descendant of the outcome | **no** | a form of leakage ({{ch:ds-leakage}}) |
| Predictor of Y only | optional | reduces variance, no bias effect |

The consequence worth internalising: **you cannot decide what to adjust for by
looking at the data.** Two variables with identical statistical profiles may
require opposite treatment depending on causal structure. The decision requires
a causal model, and the causal model comes from domain knowledge.

## 6. Mathematical Foundation

### 6.1 Constructing Simpson's paradox

Take two treatments and two subgroups. The numbers below are constructed to make
the reversal exact.

{#tbl:simpson-numbers caption="Recovery rates by treatment and severity. Treatment A wins in both subgroups and loses overall."}

| | Mild cases | Severe cases | Combined |
|---|---|---|---|
| **Treatment A** | 81/87 = **93%** | 192/263 = **73%** | 273/350 = **78%** |
| **Treatment B** | 234/270 = 87% | 55/80 = 69% | 289/350 = **83%** |

Treatment A has the higher recovery rate among mild cases (93% vs 87%) *and*
among severe cases (73% vs 69%). Pooled, it has the lower rate (78% vs 83%).

Nothing is miscalculated. The reversal happens because the *allocation* differs:
A was given mostly to severe cases (263 of 350), B mostly to mild ones (270 of
350). Severity affects both which treatment you received and whether you
recovered — it is a confounder.

Formally, the pooled rate for treatment $t$ is a weighted average of the
subgroup rates:

$$
\Prob(\text{recover} \mid t) = \sum_{s} \Prob(\text{recover} \mid t, s)\,
  \Prob(s \mid t)
$$ (eq:pooled-rate)

The weights $\Prob(s \mid t)$ differ between treatments. Treatment A's average
is dragged down by being weighted toward the harder cases.

The adjustment formula {{eq:adjustment-formula}} reweights by the *population*
severity distribution instead:

$$
\Prob(\text{recover} \mid do(t)) = \sum_{s} \Prob(\text{recover} \mid t, s)\,
  \Prob(s)
$$ (eq:adjusted-rate)

which recovers A's advantage. {{sec:7-implementation}} computes both.

> IMPORTANT: The critical point is that {{eq:pooled-rate}} and
> {{eq:adjusted-rate}} are both correct arithmetic on the same table. Which one
> answers your question depends on whether severity is a confounder (adjust) or
> a consequence of treatment (do not). **That is not a fact about the data.**
> If severity were measured *after* treatment and affected by it, adjusting
> would be wrong and the pooled figure would be right. The table is identical in
> both worlds.

### 6.2 The magnitude of confounding bias

For a binary confounder $Z$, the bias in the naive comparison is

$$
\text{bias} = \big(\E[Y \mid Z=1] - \E[Y \mid Z=0]\big)\,
  \big(\Prob(Z=1 \mid T=1) - \Prob(Z=1 \mid T=0)\big)
$$ (eq:confounding-bias)

A product of two terms: how much $Z$ affects the outcome, and how differently
$Z$ is distributed between the treatment groups.

Both must be nonzero for confounding to bite. A variable strongly related to the
outcome but balanced across groups causes no bias — which is exactly what
randomisation guarantees in expectation. A variable badly imbalanced but
unrelated to the outcome also causes none.

This gives a practical diagnostic: for each candidate confounder, estimate both
terms. Their product bounds how much your estimate could move.

### 6.3 Deriving collider bias

Let $X$ and $Y$ be independent binary variables, each 1 with probability 0.5,
and let $C = \max(X, Y)$ — admitted if either qualifies.

Unconditionally, $\Prob(X=1 \mid Y=1) = \Prob(X=1) = 0.5$: independent.

Condition on $C = 1$. The possible cases are $(1,0), (0,1), (1,1)$, each with
prior probability 0.25, renormalised to $1/3$ each. Then

$$
\Prob(X=1 \mid Y=1, C=1) = \frac{\Prob(1,1)}{\Prob(1,1) + \Prob(0,1)}
  = \frac{1/3}{2/3} = 0.5
$$

$$
\Prob(X=1 \mid Y=0, C=1) = \frac{\Prob(1,0)}{\Prob(1,0)} = 1.0
$$

Among the admitted, knowing $Y=0$ guarantees $X=1$. The variables are now
strongly dependent, and the dependence was created entirely by conditioning on
their common effect.

This is the mechanism behind several well-known effects. Selecting on a
composite outcome, restricting analysis to a filtered subgroup, or "controlling
for" a downstream variable all condition on a collider — and
{{ch:ds-collection}}'s selection bias is the special case where the collider is
sample membership itself.

## 7. Implementation

```python {tier=A name=simpsons-paradox}
"""Simpson's paradox, confounding bias, and collider bias — all constructed.
"""
import numpy as np
import pandas as pd

# --- section 6.1: the reversal, exactly ------------------------------------
print("=" * 72)
print("Simpson's paradox: A wins in both subgroups and loses overall")
print("=" * 72)

data = pd.DataFrame([
    ("A", "mild",   81,  87),
    ("A", "severe", 192, 263),
    ("B", "mild",   234, 270),
    ("B", "severe", 55,  80),
], columns=["treatment", "severity", "recovered", "total"])
data["rate"] = data["recovered"] / data["total"]

pivot = data.pivot(index="treatment", columns="severity",
                   values=["recovered", "total", "rate"])
print(f"{'':<11} {'mild':>18} {'severe':>18} {'combined':>18}")
for t in ("A", "B"):
    sub = data[data.treatment == t]
    combined = sub["recovered"].sum() / sub["total"].sum()
    cells = []
    for s in ("mild", "severe"):
        row = sub[sub.severity == s].iloc[0]
        cells.append(f"{int(row.recovered)}/{int(row.total)} = {row.rate:.0%}")
    tot = f"{sub['recovered'].sum()}/{sub['total'].sum()} = {combined:.0%}"
    print(f"treatment {t:<2} {cells[0]:>18} {cells[1]:>18} {tot:>18}")

a_mild = data.query("treatment=='A' and severity=='mild'").iloc[0]
b_mild = data.query("treatment=='B' and severity=='mild'").iloc[0]
a_sev = data.query("treatment=='A' and severity=='severe'").iloc[0]
b_sev = data.query("treatment=='B' and severity=='severe'").iloc[0]
a_all = data.query("treatment=='A'")["recovered"].sum() / 350
b_all = data.query("treatment=='B'")["recovered"].sum() / 350

print(f"\nA beats B on mild   : {a_mild.rate:.1%} vs {b_mild.rate:.1%}  "
      f"({a_mild.rate > b_mild.rate})")
print(f"A beats B on severe : {a_sev.rate:.1%} vs {b_sev.rate:.1%}  "
      f"({a_sev.rate > b_sev.rate})")
print(f"A beats B overall   : {a_all:.1%} vs {b_all:.1%}  "
      f"({a_all > b_all})   <- REVERSED")

# --- eq. 25.6: why. the allocation differs ----------------------------------
print(f"\nwhy: severity is distributed very differently across treatments")
for t in ("A", "B"):
    sub = data[data.treatment == t]
    frac_severe = sub[sub.severity == "severe"]["total"].iloc[0] / sub["total"].sum()
    print(f"  treatment {t}: {frac_severe:.0%} of its patients were severe")

# --- eq. 25.7: adjust by the POPULATION distribution ------------------------
pop = data.groupby("severity")["total"].sum()
pop_frac = pop / pop.sum()
print(f"\npopulation severity mix: "
      f"{ {k: f'{v:.0%}' for k, v in pop_frac.items()} }")

print(f"\n{'treatment':<11} {'pooled (eq. 25.6)':>20} "
      f"{'adjusted (eq. 25.7)':>22}")
for t in ("A", "B"):
    sub = data[data.treatment == t].set_index("severity")
    pooled = sub["recovered"].sum() / sub["total"].sum()
    adjusted = sum(sub.loc[s, "rate"] * pop_frac[s] for s in pop_frac.index)
    print(f"{t:<11} {pooled:>20.1%} {adjusted:>22.1%}")

print("\nAdjusting for severity recovers A's advantage. Both computations are")
print("arithmetically correct on the same table; which one answers your")
print("question depends on whether severity is a confounder or a consequence")
print("of treatment — and the table cannot tell you which.")

# --- eq. 25.8: the magnitude of confounding bias ----------------------------
print("\n" + "=" * 72)
print("confounding bias is a product of two terms (eq. 25.8)")
print("=" * 72)

rng = np.random.default_rng(0)
n = 400_000
print(f"{'Z effect on Y':>14} {'Z imbalance':>13} {'predicted bias':>16} "
      f"{'measured':>10}")
for z_effect in (0.0, 0.3):
    for imbalance in (0.0, 0.5):
        # Z is a confounder: it drives assignment and the outcome.
        z = rng.random(n) < 0.5
        p_treat = 0.5 + imbalance * (z - 0.5)
        t = rng.random(n) < p_treat
        true_effect = 0.10
        y = (0.3 + true_effect * t + z_effect * z
             + rng.normal(0, 0.05, n))

        naive = y[t].mean() - y[~t].mean()
        measured_bias = naive - true_effect
        p_z_given_t1 = z[t].mean()
        p_z_given_t0 = z[~t].mean()
        predicted = z_effect * (p_z_given_t1 - p_z_given_t0)
        print(f"{z_effect:>14.2f} {imbalance:>13.2f} {predicted:>16.4f} "
              f"{measured_bias:>10.4f}")

print("\nBias requires BOTH: Z must affect the outcome AND be distributed")
print("differently across treatment groups. Either term at zero, no bias —")
print("which is exactly what randomisation guarantees for the second term.")

# --- section 5.2: randomisation eliminates the selection term ---------------
print("\n" + "=" * 72)
print("randomisation zeroes the selection-bias term of eq. 25.3")
print("=" * 72)

n = 300_000
engagement = rng.normal(0, 1, n)              # an UNMEASURED confounder
y0 = 0.20 + 0.15 * engagement                 # baseline retention
y1 = y0 + 0.05                                # true effect: +5pp for everyone
true_ate = (y1 - y0).mean()

# Observational: engaged users self-select into the treatment.
t_obs = rng.random(n) < 1 / (1 + np.exp(-2 * engagement))
y_obs = np.where(t_obs, y1, y0)
naive_obs = y_obs[t_obs].mean() - y_obs[~t_obs].mean()
selection_term = y0[t_obs].mean() - y0[~t_obs].mean()

# Randomised: assignment is a coin flip, ignoring engagement entirely.
t_rct = rng.random(n) < 0.5
y_rct = np.where(t_rct, y1, y0)
naive_rct = y_rct[t_rct].mean() - y_rct[~t_rct].mean()
selection_rct = y0[t_rct].mean() - y0[~t_rct].mean()

print(f"true ATE                          : {true_ate:>+8.4f}")
print(f"\n{'design':<16} {'observed diff':>15} {'selection term':>16} "
      f"{'error':>9}")
print(f"{'observational':<16} {naive_obs:>+15.4f} {selection_term:>+16.4f} "
      f"{naive_obs - true_ate:>+9.4f}")
print(f"{'randomised':<16} {naive_rct:>+15.4f} {selection_rct:>+16.4f} "
      f"{naive_rct - true_ate:>+9.4f}")
print("\nThe observational estimate is roughly "
      f"{naive_obs/true_ate:.0f}x the true effect. Randomisation drives the")
print("selection term to zero WITHOUT measuring engagement — which is why it")
print("protects against confounders you do not know exist (eq. 25.5).")

# --- section 6.3: collider bias ---------------------------------------------
print("\n" + "=" * 72)
print("collider bias: conditioning CREATES an association")
print("=" * 72)

n = 200_000
talent = rng.normal(0, 1, n)
connections = rng.normal(0, 1, n)             # independent by construction
admitted = (talent + connections) > 1.2       # a common effect

print(f"correlation in the whole population : "
      f"{np.corrcoef(talent, connections)[0,1]:+.4f}")
print(f"correlation among the ADMITTED      : "
      f"{np.corrcoef(talent[admitted], connections[admitted])[0,1]:+.4f}")
print(f"correlation among the REJECTED      : "
      f"{np.corrcoef(talent[~admitted], connections[~admitted])[0,1]:+.4f}")
print(f"\n({admitted.mean():.0%} were admitted)")
print("\nTalent and connections are independent by construction. Restricting")
print("to the admitted manufactures a negative association: among those who")
print("got in, someone with few connections must have had talent.")
print("\nThis is the same mechanism as selection bias (Chapter 22) and is why")
print("'control for everything' is bad advice (table 25.1).")
```

## 8. Practical Example

Deciding what to adjust for is the practical skill. The example below works a
realistic scenario where the naive answer, the over-adjusted answer, and the
correct answer all differ.

```python {tier=A name=adjustment-decisions}
"""Choosing adjustment variables from causal structure, not from the data.

Four candidate covariates with identical statistical prominence and four
different correct treatments.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(4)
n = 250_000

# --- the true causal structure ----------------------------------------------
# tenure    -> confounder: drives both feature adoption and retention
# adopted   -> the treatment
# engagement-> mediator:   adoption causes engagement causes retention
# support   -> collider:   both adoption and retention affect contacting support
# region    -> predictor of the outcome only

tenure = rng.normal(0, 1, n)
adopted = (rng.random(n) < 1 / (1 + np.exp(-(0.9 * tenure))))
TRUE_DIRECT = 0.04
engagement = 0.5 * adopted + 0.3 * tenure + rng.normal(0, 0.5, n)
region_effect = rng.normal(0, 0.3, n)
retention = (0.30
             + TRUE_DIRECT * adopted            # the direct effect
             + 0.06 * engagement                # the mediated path
             + 0.10 * tenure                    # confounding path
             + region_effect
             + rng.normal(0, 0.1, n))
support = 0.4 * adopted + 0.4 * retention + rng.normal(0, 0.4, n)  # collider

df = pd.DataFrame({"adopted": adopted.astype(int), "tenure": tenure,
                   "engagement": engagement, "support": support,
                   "region_effect": region_effect, "retention": retention})

# The total effect of adoption = direct + through engagement.
TRUE_TOTAL = TRUE_DIRECT + 0.06 * 0.5
print(f"true DIRECT effect of adoption : {TRUE_DIRECT:+.4f}")
print(f"true TOTAL effect (direct + via engagement) : {TRUE_TOTAL:+.4f}\n")


def estimate(adjust_for):
    """Regress retention on adoption, adjusting for the named covariates."""
    cols = ["adopted"] + list(adjust_for)
    X = np.column_stack([np.ones(n)] + [df[c].to_numpy() for c in cols])
    beta, *_ = np.linalg.lstsq(X, df["retention"].to_numpy(), rcond=None)
    return beta[1]


scenarios = [
    ("nothing (naive)",              [],                              TRUE_TOTAL),
    ("tenure (the confounder)",      ["tenure"],                      TRUE_TOTAL),
    ("tenure + engagement",          ["tenure", "engagement"],        TRUE_DIRECT),
    ("tenure + support (collider)",  ["tenure", "support"],           TRUE_TOTAL),
    ("tenure + region",              ["tenure", "region_effect"],     TRUE_TOTAL),
    ("everything",                   ["tenure", "engagement",
                                      "support", "region_effect"],    TRUE_DIRECT),
]

print(f"{'adjusting for':<30} {'estimate':>10} {'target':>9} {'error':>9}")
print("-" * 62)
for label, cols, target in scenarios:
    est = estimate(cols)
    print(f"{label:<30} {est:>+10.4f} {target:>+9.4f} {est - target:>+9.4f}")

print("\ninterpretation:")
print("  nothing            — biased upward: the tenure back-door path is open")
print("  tenure             — correct for the TOTAL effect; the back door is closed")
print("  tenure+engagement  — correct for the DIRECT effect; adjusting for a")
print("                       mediator removes the path you may have wanted")
print("  tenure+support     — the SIGN FLIPS. Support is a collider, and")
print("                       conditioning on it opens a spurious path strong")
print("                       enough to turn a real positive effect negative.")
print("                       An analyst who 'controlled for support contacts'")
print("                       would conclude the feature HARMS retention.")
print("  tenure+region      — unchanged estimate, smaller variance: adjusting")
print("                       for an outcome-only predictor is free precision")
print("  everything         — the 'control for everything' default, which here")
print("                       silently answers a different question AND is")
print("                       contaminated by the collider")

# --- the variance benefit of adjusting for an outcome predictor -------------
print("\n" + "=" * 72)
print("adjusting for an outcome-only predictor reduces variance")
print("=" * 72)
ests_without, ests_with = [], []
for _ in range(300):
    idx = rng.choice(n, 4000, replace=False)
    sub = df.iloc[idx]
    for cols, store in ((["tenure"], ests_without),
                        (["tenure", "region_effect"], ests_with)):
        X = np.column_stack([np.ones(len(sub))]
                            + [sub[c].to_numpy() for c in ["adopted"] + cols])
        b, *_ = np.linalg.lstsq(X, sub["retention"].to_numpy(), rcond=None)
        store.append(b[1])

print(f"{'adjustment':<26} {'mean estimate':>15} {'sd of estimate':>16}")
print(f"{'tenure only':<26} {np.mean(ests_without):>+15.4f} "
      f"{np.std(ests_without):>16.4f}")
print(f"{'tenure + region':<26} {np.mean(ests_with):>+15.4f} "
      f"{np.std(ests_with):>16.4f}")
print(f"\nSame estimate, {np.std(ests_without)/np.std(ests_with):.1f}x tighter. "
      f"Adjusting for a variable that predicts")
print("the outcome but not the treatment costs nothing and buys precision.")

print("\n" + "=" * 72)
print("the point")
print("=" * 72)
print("All four covariates look similar in the data: each correlates with")
print("both adoption and retention. Their correct treatment is opposite in")
print("three of the four cases, and nothing in the table distinguishes them.")
print("The decision comes from a causal model, which comes from knowing how")
print("the system works.")
```

## 9. Common Mistakes

**Reading a correlation as a causal effect.** Four explanations are always live.

**Controlling for everything available.** Colliders and mediators make
adjustment actively harmful.

**Adjusting for a mediator when you want the total effect.** Removes the path
you were measuring.

**Adjusting for a post-treatment variable.** It is downstream of the treatment
and conditioning on it is a form of leakage ({{ch:ds-leakage}}).

**Assuming a large sample rules out confounding.** Confounding bias does not
shrink with $n$ ({{eq:confounding-bias}}).

**Believing "we adjusted for confounders" settles it.** The assumption that you
measured them all is untestable.

**Choosing the pooled or subgroup answer by which is more convenient.** The
choice is a causal claim, and should be argued as one.

**Interpreting regression coefficients causally by default.** A coefficient is a
conditional association; it is causal only under assumptions you should state.

**Concluding causation from a large effect.** Magnitude is not evidence of
mechanism.

**Restricting the analysis to a subgroup defined by an outcome.** Collider bias,
by another name.

## 10. Connection to Previous Chapters

{{ch:math-covariance}} established that correlation measures linear association
and nothing more; this chapter establishes that even a perfect measure of
association would not establish causation. {{ch:math-probability}} supplies the
conditional probability that {{eq:adjustment-formula}} is written in, and the
collider derivation in {{sec:6-mathematical-foundation}} is Bayes' theorem
applied to a common effect. {{ch:ds-eda}} established the habit of
disaggregating; Simpson's paradox is what happens when you do and the answer
reverses. {{ch:ds-collection}} covered selection bias, which is collider bias
where the collider is sample membership.

Forward: {{ch:ds-experiments}} makes randomisation practical, and is the
constructive response to this chapter's negative results. {{ch:ds-leakage}}
covers post-treatment variables from the predictive side.

Beyond Part III: {{ch:ml-linear-regression}} returns to interpreting
coefficients; {{ch:rai-interpretability}} distinguishes what a model attends to
from what causes the outcome, which is the same distinction at a different
level. {{cite:simpson1951}} is the original.

## 11. Exercises

**Beginner**

1. State the fundamental problem of causal inference in one sentence.
2. Ice cream sales correlate with drownings. Give all four explanations from
   {{sec:4-intuitive-explanation}}.
3. Define a confounder and give an example from a domain you know.
4. Why does randomisation work even for unmeasured confounders?
5. What is a collider, and why is conditioning on one harmful?

**Intermediate**

6. Verify the arithmetic of {{tbl:simpson-numbers}} and compute the adjusted
   rates using {{eq:adjustment-formula}}.
7. Using {{eq:confounding-bias}}, compute the bias when $Z$ raises the outcome by
   0.2 and is present in 70% of the treated and 30% of the untreated.
8. Explain why adjusting for a mediator answers a different question, and name
   both questions.
9. A study finds that among hospitalised patients, smoking is associated with
   *lower* COVID severity. Give a collider explanation.
10. Your treated and control groups have identical distributions on ten measured
    covariates. Does that establish no confounding? Why not?
11. Explain why a large sample does not reduce confounding bias.

**Advanced**

12. Derive {{eq:causal-decomposition}} from {{eq:observed-difference}}.
13. Prove that randomisation implies the selection-bias term is zero, stating
    where {{eq:ignorability}} is used.
14. Derive {{eq:confounding-bias}} for a binary confounder.
15. Extend the collider derivation of {{sec:6-mathematical-foundation}} to
    continuous variables and explain why the induced correlation is negative.
16. Construct a dataset exhibiting Simpson's paradox where the *correct* answer
    is the pooled one, and explain what makes it correct there.

**Implementation**

17. Write a function that, given a treatment, outcome and candidate confounder,
    reports both terms of {{eq:confounding-bias}} and their product.
18. Simulate a mediator and show empirically that adjusting for it recovers the
    direct effect and not the total.
19. Build a simulation where an unmeasured confounder reverses the sign of the
    estimated effect, and show that randomisation recovers the truth.
20. Implement inverse-probability weighting for a binary treatment and compare
    it against stratified adjustment on the same simulated data.

**Reasoning**

21. An analyst says "we controlled for all available covariates, so the effect
    is causal." What is wrong with the reasoning?
22. When is an observational estimate good enough to act on? What would you
    require?

## 12. Chapter Summary

A causal effect compares an outcome to what would have happened otherwise for
the same unit. Only one of the two is ever observed — the fundamental problem of
causal inference — so every method estimates an average effect despite never
observing an individual one.

The observed difference between treated and untreated groups decomposes into the
effect on the treated plus a selection term comparing what the two groups would
have done anyway. Confounding is exactly this second term being nonzero, and its
magnitude is the product of how much the confounder affects the outcome and how
unevenly it is distributed across treatment groups. It does not shrink with
sample size.

Randomisation makes treatment independent of the potential outcomes, driving the
selection term to zero. Its decisive advantage is that this holds for
confounders you did not measure and do not know about, which is why an
experiment is qualitatively stronger evidence than any observational analysis.

Simpson's paradox produces two arithmetically correct answers pointing in
opposite directions. Which one is right depends on whether the subgroup variable
is a confounder or a consequence of treatment — a causal question the table
cannot answer. This is the clearest demonstration that data alone does not
determine a causal conclusion.

Adjusting for more variables is not safer. Confounders should be adjusted for;
colliders must not be, because conditioning on a common effect manufactures an
association that did not exist; mediators must not be if the total effect is
wanted; and outcome-only predictors are optional and buy precision for free. The
correct treatment depends on causal role, and two variables with identical
statistical profiles can require opposite handling.
