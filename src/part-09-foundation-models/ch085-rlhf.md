---
id: fm-rlhf
number: 85
part: IX
tier: full
status: draft
requires: [fm-instruction-tuning, fm-pretraining, dl-losses, math-probability,
           ml-logistic, math-optimization, dl-optimizers]
provides: [rlhf, bradley-terry, reward-model, preference-pair, kl-penalty,
           reward-hacking, over-optimisation, alignment-tax, policy-optimisation,
           reference-policy, constitutional-ai, rlaif]
citations: [ouyang2022, christiano2017, stiennon2020, schulman2017ppo, bai2022,
            wei2022flan, brown2020, hoffmann2022chinchilla]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Explain why demonstrations cannot express a preference, and what comparisons
   add.
2. Derive the Bradley–Terry model and its log-likelihood, and fit a reward model
   from pairwise comparisons.
3. State the KL-regularised RLHF objective and explain what each term is for.
4. Explain reward over-optimisation and demonstrate it as a measurement.
5. Describe the three-stage pipeline and say what each stage costs.
6. Explain what {{cite:bai2022}} substitutes and why an explicit constitution is
   a different kind of artefact from preference labels.
7. State honestly what RLHF achieves and what is contested about how.

## 2. Why This Matters

**This is the stage that made language models into products.**
{{cite:ouyang2022}}'s headline result deserves to be memorised: a 1.3B parameter
aligned model was preferred by human raters to the 175B base model. Alignment
bought more perceived quality than two orders of magnitude of scale — the single
most striking cost-effectiveness result in this part.

**It teaches something demonstrations cannot.**
{{ch:fm-instruction-tuning}} ended on this: a demonstration shows one good
answer and cannot express that one acceptable answer is *better* than another.
Preferences are comparative, and capturing them requires comparisons.

**Its core failure mode is the most general lesson in the part.** Optimising
against a learned proxy for what you want works until you push too hard, at
which point true quality falls while the proxy keeps rising
({{cite:stiennon2020}}). That is Goodhart's law with a measurement attached, and
it recurs everywhere in this book — in {{ch:ml-metrics}}, in
{{ch:fm-emergence}}'s benchmarks, and in every evaluation in {{part:25}}.

**And the machinery is contested.** {{ch:fm-dpo}} shows the whole RL apparatus
can be replaced by one supervised loss with comparable results, which raises the
question of whether the RL was ever where the value lay. This chapter derives it
properly so the next chapter's simplification is legible rather than magical.

## 3. Prerequisites

{{ch:fm-instruction-tuning}} for the SFT stage this builds on, and for the gap
it identified. {{ch:fm-pretraining}} for the causal objective.
{{ch:dl-losses}} for cross-entropy and KL divergence — both are load-bearing.
{{ch:math-probability}} for the probabilistic model in
{{sec:6-mathematical-foundation}}. {{ch:ml-logistic}} for the logistic
likelihood, which Bradley–Terry turns out to be. {{ch:math-optimization}} for
constrained optimisation. {{ch:dl-optimizers}} for the policy-gradient step.

## 4. Intuitive Explanation

You want the model to be helpful. Write that as a loss function.

You cannot. There is no differentiable expression for helpfulness, no labelled
dataset of helpfulness scores, and no agreement on a definition. This is the
situation alignment is always in: **the objective cannot be written down.**

{{cite:christiano2017}}'s answer, developed on Atari games five years before
anyone applied it to language: **do not write the objective — learn it.** People
are bad at scoring things absolutely and good at comparing two things. So show
them pairs, ask which is better, and fit a function that agrees with their
choices. Then optimise the model against that function.

Three stages follow:

**Stage 1, supervised fine-tuning.** {{ch:fm-instruction-tuning}}. Get the model
answering questions at all, because the next stage needs it to produce plausible
candidates to compare.

**Stage 2, the reward model.** Sample several responses to each prompt, have
people rank them, and fit a model $r_\phi$ that assigns a scalar to any
(prompt, response) pair such that preferred responses score higher.

**Stage 3, policy optimisation.** Now you have a differentiable stand-in for
"good". Optimise the language model to maximise it.

**And immediately, a problem.** The reward model is a *learned approximation* to
human preference, fitted on a finite sample. Push the policy hard enough against
it and the policy finds regions where the reward model is wrong — high predicted
reward, poor actual quality. {{cite:stiennon2020}} documented this carefully:
true quality rises, peaks, and falls, while predicted reward climbs throughout.

> NOTE: This is not a bug in the reward model that a better one would fix. It is
> what happens whenever you optimise a proxy far from where it was fitted. The
> standard defence is not a better proxy but a *leash*: penalise the policy for
> straying from the model it started as, which keeps it in the region where the
> reward model has evidence.

**The leash is a KL penalty**, and it is the term that makes the whole thing
work. Without it the policy drifts into gibberish that the reward model happens
to like. With it, the policy improves within a neighbourhood of the SFT model.

**The mental model:** RLHF replaces an unwritable objective with a learned one,
then constrains optimisation to the region where the learned objective is
trustworthy. Where it breaks down: "the region where the reward model is
trustworthy" is not observable, so the KL coefficient is a hyperparameter
standing in for a quantity nobody can measure.

## 5. Formal Explanation

### 5.1 The Bradley–Terry preference model

Given a prompt $x$ and two responses $y_w$ (preferred) and $y_l$, assume a
latent reward $r^*(x,y)$ and that human choices follow

$$
\Prob\big[y_w \succ y_l \given x\big]
 = \frac{\exp r^*(x,y_w)}{\exp r^*(x,y_w) + \exp r^*(x,y_l)}
 = \sigma\big(r^*(x,y_w) - r^*(x,y_l)\big)
$$ (eq:bradley-terry)

**This is a logistic model on the reward difference** ({{ch:ml-logistic}}), and
that is the whole of the modelling assumption: preferences depend only on the
difference of two scalars.

Fitting $r_\phi$ by maximum likelihood over a comparison dataset
$\Data = \{(x, y_w, y_l)\}$ gives

$$
\Loss_{\text{RM}}(\phi) = -\E_{(x,y_w,y_l)\sim\Data}
 \Big[\log\sigma\big(r_\phi(x,y_w) - r_\phi(x,y_l)\big)\Big]
$$ (eq:reward-model-loss)

> IMPORTANT: $r^*$ is only identified up to an additive function of $x$. Adding
> any $c(x)$ to both responses' rewards leaves {{eq:bradley-terry}} unchanged.
> The absolute scale of a reward model's outputs is therefore meaningless, and
> comparing reward values across prompts is an error — a common one, and it
> produces confident nonsense in dashboards.

### 5.2 The KL-regularised objective

With $r_\phi$ fitted, optimise the policy $\pi_\theta$:

$$
\max_{\theta}\ \E_{x\sim\Data,\ y\sim\pi_\theta(\cdot\given x)}
 \Big[r_\phi(x,y)\Big]
 - \beta\,\KL\Big(\pi_\theta(\cdot\given x)\ \Big\|\ \pi_{\text{ref}}(\cdot\given x)\Big)
$$ (eq:rlhf-objective)

where $\pi_{\text{ref}}$ is the SFT model, frozen.

Three components, each doing distinct work:

- **The reward term** pushes toward responses the reward model scores highly.
- **The KL term** penalises divergence from the reference, bounding how far the
  policy can travel from where the reward model has evidence.
- **$\beta$** trades them off. Small $\beta$ optimises harder and
  over-optimises sooner; large $\beta$ barely moves from SFT.

**{{cite:ouyang2022}} adds a third term**: a fraction of the pretraining
gradient, mixed in to limit capability regression — the *alignment tax* of
{{sec:12-failure-modes}}.

### 5.3 Why PPO

{{eq:rlhf-objective}} is optimised by a policy-gradient method, in practice PPO
({{cite:schulman2017ppo}}). The reason is stability: an unconstrained policy
gradient on a language model takes steps large enough to destroy it, and PPO's
clipped surrogate bounds the update:

$$
\Loss^{\text{CLIP}} = \E\Big[\min\big(\rho_t \hat{A}_t,\
 \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\hat{A}_t\big)\Big],
\qquad
\rho_t = \frac{\pi_\theta(a_t\given s_t)}{\pi_{\theta_{\text{old}}}(a_t\given s_t)}
$$ (eq:ppo-clip)

The clip removes the incentive to move the probability ratio far from 1 in a
single update. **There are now two constraints on how far the policy can move**
— the KL term against the reference, and the clip against the previous
iterate — and they do different jobs: one bounds total drift, the other bounds
per-step drift.

### 5.4 The full pipeline and its costs

{#tbl:rlhf-stages caption="The three stages of RLHF and what each requires. The last stage is where the operational complexity lives: it needs sampling from the policy during training, which means generation inside the training loop."}

| Stage | Data | Models in memory | Cost driver |
|---|---|---|---|
| 1. SFT | demonstrations | 1 | small ({{ch:fm-instruction-tuning}}) |
| 2. Reward model | comparisons | 1 | annotation, not compute |
| 3. Policy optimisation | prompts only | **4** | generation in the loop |

**Stage 3 holds four models**: the policy being trained, the frozen reference for
the KL term, the reward model, and (for PPO) a value model. That memory
requirement, plus the need to *generate* samples at every step, is the practical
burden {{ch:fm-dpo}} removes.

### 5.5 Constitutional AI

{{cite:bai2022}} replaces human harmfulness labels with model-generated
critiques against an explicit written constitution. The model critiques and
revises its own outputs according to stated principles, and the resulting
preference pairs train the reward model.

**The substitution is not primarily about cost.** With human preference labels,
the normative content of alignment is implicit in a set of annotations nobody
can read. With a constitution it is **a document that can be inspected, argued
about, and versioned.** That is a change in the kind of artefact alignment
produces, and it is the reason the idea matters beyond the labelling saving.

## 6. Mathematical Foundation

### 6.1 The optimal policy under the KL-regularised objective

The optimum of {{eq:rlhf-objective}} has a closed form, and deriving it here is
what makes {{ch:fm-dpo}} possible.

For a fixed prompt $x$, maximise over distributions $\pi$:

$$
J(\pi) = \sum_y \pi(y) r(x,y) - \beta \sum_y \pi(y)\log\frac{\pi(y)}{\pi_{\text{ref}}(y)}
$$

subject to $\sum_y \pi(y) = 1$. With a Lagrange multiplier $\lambda$:

$$
\frac{\partial}{\partial \pi(y)}
 \Big[J(\pi) + \lambda\big(\textstyle\sum_y \pi(y) - 1\big)\Big]
 = r(x,y) - \beta\Big(\log\frac{\pi(y)}{\pi_{\text{ref}}(y)} + 1\Big) + \lambda = 0
$$

Solving for $\pi(y)$:

$$
\log\frac{\pi(y)}{\pi_{\text{ref}}(y)} = \frac{r(x,y) + \lambda}{\beta} - 1
\implies
\pi(y) = \pi_{\text{ref}}(y)\exp\!\Big(\frac{r(x,y)}{\beta}\Big)\cdot e^{\frac{\lambda}{\beta}-1}
$$

The final factor is constant in $y$ and fixed by normalisation, giving

$$
\pi^*(y\given x) = \frac{1}{Z(x)}\,\pi_{\text{ref}}(y\given x)
 \exp\!\Big(\frac{1}{\beta}r(x,y)\Big),
\qquad
Z(x) = \sum_y \pi_{\text{ref}}(y\given x)e^{r(x,y)/\beta}
$$ (eq:rlhf-optimal-policy)

$\square$

**Read what this says.** The optimal aligned policy is the reference policy
*reweighted* by the exponentiated reward. It does not invent new behaviour; it
tilts the existing distribution toward higher reward, exactly as
{{eq:continuation-mixture}} described instruction tuning. And $\beta$ controls
the sharpness of the tilt: as $\beta\to\infty$, $\pi^*\to\pi_{\text{ref}}$; as
$\beta\to 0$, $\pi^*$ collapses onto the argmax of $r$.

**This equation is the whole of {{ch:fm-dpo}}.** Inverting it to express $r$ in
terms of $\pi^*$ is what lets preference data train the policy directly.

### 6.2 Why over-optimisation is inevitable

Let $r^*$ be true reward and $r_\phi$ the fitted approximation, with error
$\varepsilon(x,y) = r_\phi(x,y) - r^*(x,y)$. The policy maximises $r_\phi$, so
it maximises $r^* + \varepsilon$.

Over a large response space, the responses maximising $r_\phi$ are
disproportionately those where $\varepsilon$ is large and positive — the
selection is on the sum, and errors that happen to be favourable are selected
along with genuine quality. This is the same **winner's-curse** structure as the
optimism of a hyperparameter search's reported best in {{ch:mle-hpo}}.

Expected true reward under the optimised policy is therefore

$$
\E_{\pi}[r^*] = \E_{\pi}[r_\phi] - \E_{\pi}[\varepsilon]
$$ (eq:over-optimisation)

and $\E_\pi[\varepsilon]$ *grows* as the policy moves further from the fitting
distribution, because that is where $\varepsilon$ is both larger and less
constrained by data.

$\square$

**Hence the shape**: true quality rises while the genuine-improvement term
dominates, then falls once the selected-error term overtakes it. The KL penalty
works by bounding how far the policy can move, which bounds $\E_\pi[\varepsilon]$
— it does not make the reward model better, it keeps the policy where the reward
model is right.

### 6.3 A worked Bradley–Terry calculation

Two responses with fitted rewards $r_\phi(x,y_1) = 2.4$ and
$r_\phi(x,y_2) = 1.1$.

$$
\Prob[y_1 \succ y_2] = \sigma(2.4 - 1.1) = \sigma(1.3)
 = \frac{1}{1+e^{-1.3}} = 0.786
$$

Now add 5 to both: $\sigma(7.4 - 6.1) = \sigma(1.3) = 0.786$. **Unchanged**,
which is the non-identifiability of {{sec:5-formal-explanation}} in one line.

And the loss contribution if this pair was labelled $y_1 \succ y_2$:

$$
-\log\sigma(1.3) = -\log 0.786 = 0.241
$$

If it was labelled the other way: $-\log\sigma(-1.3) = 1.541$ — about six times
larger, which is the gradient pressure to correct a confidently wrong ordering.

## 7. Internal Mechanics

```mermaid {#fig:rlhf-pipeline caption="The three-stage RLHF pipeline. Stage 3 is the operationally hard one: it holds four models and generates samples inside the training loop, which is the complexity that DPO removes in the next chapter."}
graph TD
  A["base model"] --> B["STAGE 1: SFT<br/>demonstrations"]
  B --> C["SFT model"]
  C --> D["sample k responses<br/>per prompt"]
  D --> E["humans rank them"]
  E --> F["STAGE 2: fit reward model<br/>eq:reward-model-loss"]
  C -.->|"frozen copy"| G["reference policy"]
  C --> H["STAGE 3: policy optimisation"]
  F --> H
  G -->|"KL penalty"| H
  H --> I["aligned model"]
  H -.->|"generate, score, update"| H
  style H fill:#fde,stroke:#c69
  style G fill:#dfe,stroke:#5a5
```

**Where the reward model comes from.** It is usually the SFT model with the
unembedding replaced by a scalar head, so it inherits the language
understanding and only has to learn the ranking. Training it is cheap; the
expensive input is the comparison data.

**How comparisons are collected.** Sample $k$ responses per prompt and ask for a
ranking, which yields $\binom{k}{2}$ pairs from one annotation session — a
substantial efficiency over collecting pairs independently. The pairs from one
prompt are correlated, and treating them as independent in
{{eq:reward-model-loss}} slightly overstates the effective sample size.

**Why generation in the loop is expensive.** Stage 3 must sample from the
current policy to score it, so every optimisation step contains a generation
step. Generation is memory-bandwidth-bound and sequential
({{ch:tf-masking-kv}}), so the training loop inherits inference's worst
performance characteristics.

**What the KL is computed over.** The full token-level KL between policy and
reference, accumulated across the generated sequence. In practice it is often
estimated from the sampled tokens rather than summed over the vocabulary, and
which estimator is used affects the effective $\beta$ — a detail that makes
published $\beta$ values hard to compare.

## 8. Implementation

Fitting a reward model from comparisons, and verifying it recovers a known
latent ordering.

```python {tier=A name=bradley-terry-reward-model}
"""Fit a reward model from pairwise comparisons. Equation (eq:reward-model-loss)."""
import numpy as np

rng = np.random.default_rng(0)

N_ITEMS, D, N_PAIRS = 60, 8, 4000

# A latent reward that we will try to recover from comparisons alone.
features = rng.normal(size=(N_ITEMS, D))
true_w = rng.normal(size=D)
true_reward = features @ true_w

# Humans compare pairs and choose stochastically per Bradley-Terry
# (eq:bradley-terry) — they are noisy, not deterministic.
i = rng.integers(0, N_ITEMS, N_PAIRS)
j = rng.integers(0, N_ITEMS, N_PAIRS)
keep = i != j
i, j = i[keep], j[keep]
p_i_wins = 1 / (1 + np.exp(-(true_reward[i] - true_reward[j])))
i_wins = rng.random(len(i)) < p_i_wins
winner = np.where(i_wins, i, j)
loser = np.where(i_wins, j, i)

print(f"{N_ITEMS} responses, {len(winner)} comparisons")
agree = float(np.mean((true_reward[winner] > true_reward[loser])))
print(f"labeller agreement with the latent reward: {agree:.3f} "
      f"(noisy by construction)\n")


def loss_and_grad(w):
    """Negative log likelihood of eq:reward-model-loss, and its gradient."""
    r = features @ w
    diff = r[winner] - r[loser]
    sig = 1 / (1 + np.exp(-diff))
    loss = -np.mean(np.log(sig + 1e-12))
    # d/dw of -log sigma(rw - rl) = -(1 - sigma) * (phi_w - phi_l)
    coef = (sig - 1)[:, None]
    grad = (coef * (features[winner] - features[loser])).mean(0)
    return loss, grad


w = np.zeros(D)
for step in range(1, 3001):
    loss, grad = loss_and_grad(w)
    w -= 1.0 * grad
    if step in (1, 500, 1500, 3000):
        print(f"step {step:>4}: reward-model loss {loss:.4f}")

fitted = features @ w

# Does the fitted reward rank items the way the latent one does?
order_true = np.argsort(true_reward)
order_fit = np.argsort(fitted)
rank_true = np.empty(N_ITEMS); rank_true[order_true] = np.arange(N_ITEMS)
rank_fit = np.empty(N_ITEMS); rank_fit[order_fit] = np.arange(N_ITEMS)
spearman = float(np.corrcoef(rank_true, rank_fit)[0, 1])

pairs = [(a, b) for a in range(N_ITEMS) for b in range(a + 1, N_ITEMS)]
concordant = np.mean([(true_reward[a] > true_reward[b]) ==
                      (fitted[a] > fitted[b]) for a, b in pairs])

print(f"\nrank correlation with the latent reward : {spearman:.4f}")
print(f"pairwise ordering agreement             : {concordant:.4f}")
assert spearman > 0.9, "the reward model should recover the latent ordering"

# The non-identifiability of section 5.1, demonstrated.
shifted = fitted + 7.3
print(f"\nadding a constant to every reward:")
print(f"  mean reward   {fitted.mean():+.3f} -> {shifted.mean():+.3f}")
print(f"  ordering agreement unchanged: "
      f"{np.mean([(fitted[a] > fitted[b]) == (shifted[a] > shifted[b]) for a, b in pairs]):.3f}")
print("Absolute reward values carry no information. Only differences do — so a "
      "dashboard tracking mean reward across prompts is tracking an artefact.")
```

Now the failure mode the chapter is really about:

```python {tier=A name=reward-over-optimisation}
"""True quality rises, peaks, then falls, while predicted reward climbs."""
import numpy as np

rng = np.random.default_rng(1)
N_CANDIDATES = 40_000

# The model of the world this listing assumes, stated explicitly:
#
#   1. Moving away from the reference policy buys real quality at first and
#      then costs it — text far from the SFT distribution degrades. So TRUE
#      quality is concave in the distance travelled.
#   2. The reward model's error GROWS with that distance, because it was fitted
#      on samples near the reference and has no evidence further out.
#
# The policy sees only fitted reward and maximises it.

def true_quality_mean(d):
    """Concave in distance: genuine gains, then degradation."""
    return 3.0 * np.sqrt(d) - 0.55 * d


def error_scale(d):
    """The reward model's error, growing with distance from its fitting data."""
    return 0.25 * d


print(f"{'KL budget':>10} {'predicted reward':>18} {'TRUE reward':>13} "
      f"{'E[error]':>10}")
results = []
for kl in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0):
    # Candidate responses available at this distance from the reference.
    true_r = true_quality_mean(kl) + rng.normal(0, 1.0, N_CANDIDATES)
    err = rng.normal(0, error_scale(kl) + 1e-9, N_CANDIDATES)
    fitted_r = true_r + err

    chosen = int(np.argmax(fitted_r))          # the policy maximises FITTED
    results.append((kl, float(fitted_r[chosen]), float(true_r[chosen]),
                    float(err[chosen])))
    print(f"{kl:>10.1f} {fitted_r[chosen]:>18.3f} {true_r[chosen]:>13.3f} "
          f"{err[chosen]:>10.3f}")

kls = [r[0] for r in results]
fits = [r[1] for r in results]
trues = [r[2] for r in results]
peak = int(np.argmax(trues))

print(f"\ntrue reward peaks at KL = {kls[peak]:.1f} "
      f"({trues[peak]:.3f}) and declines to {trues[-1]:.3f}")
print(f"predicted reward rises throughout: {fits[0]:.2f} -> {fits[-1]:.2f}")

assert trues[peak] > trues[-1], "true reward must decline past the peak"
assert fits[-1] > fits[0], "predicted reward must keep rising"
assert peak < len(kls) - 1, "the peak must be interior, not at the boundary"

print("""
This is eq:over-optimisation as a measurement. Two things drive it and both are
necessary: true quality eventually degrades with distance from the reference,
and the reward model's error grows there because it has no data. The policy
selects on the SUM, so past a point it is selecting error rather than quality —
note the E[error] column climbing steadily.

The practical consequence is in the second and third columns. Predicted reward,
the curve visible during training, rises monotonically and gives NO indication
that quality has begun to fall. The only instrument that sees the peak is
held-out human evaluation, which is why the KL penalty is set conservatively
rather than tuned against the reward.""")
```

And the effect of the KL coefficient, which is how the leash is set:

```python {tier=A name=kl-penalty-tradeoff}
"""What beta buys and what it costs. Equation (eq:rlhf-optimal-policy)."""
import numpy as np

rng = np.random.default_rng(2)
N = 200

# A reference policy over N candidate responses, and a fitted reward whose
# error grows with distance from the reference's mass.
ref_logits = rng.normal(size=N)
ref = np.exp(ref_logits) / np.exp(ref_logits).sum()

true_r = rng.normal(size=N)
# Error is larger for responses the reference rarely produces — exactly the
# responses the reward model has little data on.
rarity = -np.log(ref + 1e-12)
rarity = (rarity - rarity.min()) / (rarity.max() - rarity.min())
fitted_r = true_r + rng.normal(size=N) * rarity * 1.8


def optimal_policy(beta):
    """Equation (eq:rlhf-optimal-policy), computed exactly."""
    logits = np.log(ref + 1e-12) + fitted_r / beta
    logits -= logits.max()
    p = np.exp(logits)
    return p / p.sum()


def kl(p, q):
    return float(np.sum(p * np.log((p + 1e-12) / (q + 1e-12))))


print(f"{'beta':>8} {'KL(pi||ref)':>12} {'E[fitted r]':>13} {'E[TRUE r]':>11} "
      f"{'verdict':<22}")
best_true, best_beta = -np.inf, None
for beta in (10.0, 3.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02):
    p = optimal_policy(beta)
    d = kl(p, ref)
    ef, et = float(p @ fitted_r), float(p @ true_r)
    if et > best_true:
        best_true, best_beta = et, beta
    verdict = "barely moved" if d < 0.1 else ("over-optimised" if et < 0 else "")
    print(f"{beta:>8.2f} {d:>12.3f} {ef:>13.3f} {et:>11.3f} {verdict:<22}")

print(f"\ntrue reward is maximised at beta = {best_beta} "
      f"(E[true r] = {best_true:.3f})")
print(f"at beta = 0.02 the policy has KL {kl(optimal_policy(0.02), ref):.2f} "
      f"from the reference and E[true r] = {optimal_policy(0.02) @ true_r:+.3f}")

print("""
Both ends of the beta range are bad and for different reasons. Large beta keeps
the policy on top of the reference and captures almost none of the available
improvement. Small beta lets the policy chase the reward model into the region
where it is wrong, and true reward falls even as fitted reward rises.

The optimum is interior, and — this is the difficult part — it cannot be found
by watching the fitted reward, which is monotone in 1/beta. Setting beta
requires held-out evaluation with real judges, which is expensive, which is why
in practice it is set conservatively and rarely tuned.""")
```

## 9. Practical Example

A team has an instruction-tuned support assistant. Users complain it is
"technically correct but unhelpful" — it answers the question asked rather than
the question meant, and it hedges. Demonstrations have not fixed it, because
every demonstration is *one* answer and the problem is a preference between
answers that are all defensible.

This is the case RLHF is for. The question is what it costs and where it fails.

```python {tier=A name=preference-data-planning}
"""Planning a preference-data collection, and the reward model's ceiling."""
import numpy as np

rng = np.random.default_rng(5)

PROMPTS = 4_000
K_RESPONSES = 4                    # sampled per prompt
COST_PER_RANKING = 2.20            # one annotator ranking k responses
ANNOTATOR_AGREEMENT = 0.72         # measured on a doubly-labelled subset

pairs_per_prompt = K_RESPONSES * (K_RESPONSES - 1) // 2
total_pairs = PROMPTS * pairs_per_prompt
cost = PROMPTS * COST_PER_RANKING

print(f"{PROMPTS:,} prompts x {K_RESPONSES} responses")
print(f"  pairs per prompt : {pairs_per_prompt}")
print(f"  total pairs      : {total_pairs:,}")
print(f"  annotation cost  : ${cost:,.0f}")
print(f"  cost per pair    : ${cost / total_pairs:.3f}  "
      f"(ranking k responses is much cheaper per pair than collecting pairs)\n")

# The ceiling: a reward model cannot be more accurate than its labels.
# With agreement a, the fraction of pairs where the label matches the true
# preference is a; the rest are noise the model can only fit or ignore.
print(f"annotator agreement: {ANNOTATOR_AGREEMENT:.0%}")
print(f"-> a reward model scoring above {ANNOTATOR_AGREEMENT:.0%} pairwise "
      f"accuracy on this data is fitting label noise\n")

# How does reward-model accuracy scale with the number of comparisons?
D = 16
true_w = rng.normal(size=D)


def fit_rm(n_pairs, agreement):
    items = rng.normal(size=(600, D))
    r = items @ true_w
    i, j = rng.integers(0, 600, n_pairs), rng.integers(0, 600, n_pairs)
    m = i != j
    i, j = i[m], j[m]
    correct = r[i] > r[j]
    # Annotators disagree with the latent preference at rate (1 - agreement).
    flip = rng.random(len(i)) > agreement
    i_wins = np.where(flip, ~correct, correct)
    win, lose = np.where(i_wins, i, j), np.where(i_wins, j, i)

    w = np.zeros(D)
    for _ in range(600):
        diff = items[win] @ w - items[lose] @ w
        sig = 1 / (1 + np.exp(-diff))
        grad = ((sig - 1)[:, None] * (items[win] - items[lose])).mean(0)
        w -= 2.0 * grad

    test_i, test_j = rng.integers(0, 600, 4000), rng.integers(0, 600, 4000)
    m = test_i != test_j
    test_i, test_j = test_i[m], test_j[m]
    pred = (items[test_i] @ w) > (items[test_j] @ w)
    latent = r[test_i] > r[test_j]
    # Held-out LABELS are noisy in exactly the same way the training ones were.
    flip = rng.random(len(test_i)) > agreement
    labels = np.where(flip, ~latent, latent)
    return float(np.mean(pred == latent)), float(np.mean(pred == labels))


print(f"{'comparisons':>13} {'vs LATENT preference':>22} {'vs held-out LABELS':>21}")
for n in (500, 2_000, 8_000, 24_000):
    vs_latent, vs_labels = fit_rm(n, ANNOTATOR_AGREEMENT)
    print(f"{n:>13,} {vs_latent:>22.3f} {vs_labels:>21.3f}")

print(f"""
The two columns are different quantities and confusing them is common.

Accuracy against the LATENT preference keeps climbing toward 1.0. Noisy labels
still identify a consistent underlying ordering given enough of them — noise
that is symmetric averages out, which is why more comparisons help even when
each one is unreliable.

Accuracy against held-out LABELS saturates near the annotator agreement of
{ANNOTATOR_AGREEMENT:.0%}, and cannot exceed it: the held-out labels are wrong
{1 - ANNOTATOR_AGREEMENT:.0%} of the time, so a perfect model disagrees with
them exactly that often.

This matters because the second column is the one you can actually measure. A
reward model scoring {ANNOTATOR_AGREEMENT:.2f} against held-out labels may be
perfect or may be mediocre, and the number alone cannot tell you — which is why
annotator agreement must be measured separately, as the interpretation key for
every reward-model number you will ever report.""")
```

> PRODUCTION TIP: Measure inter-annotator agreement before measuring the reward
> model. It is the ceiling, it is cheap to establish on a doubly-labelled
> subset, and without it a reward-model accuracy number cannot be interpreted at
> all — which is the same argument {{ch:nlp-extraction}} made about NER labels,
> at higher stakes.

## 10. Production Considerations

**Set $\beta$ conservatively and verify with human evaluation.** The
`kl-penalty-tradeoff` listing shows the optimum is interior and invisible in the
fitted reward. There is no substitute for held-out judging, and the cost of that
is why $\beta$ is usually inherited rather than tuned.

**Monitor KL from the reference during training.** It is the leading indicator
of over-optimisation, and unlike true quality it is free to compute. A KL that
climbs steadily is a policy walking away from where the reward model has
evidence.

**Track response length as a first-class metric.** Length bias enters at
{{ch:fm-instruction-tuning}} and is *amplified* here, because human raters
prefer longer answers and the reward model learns that preference. A reward
model that has learned "longer is better" produces a policy that grows without
bound.

**Keep a general capability evaluation the alignment data never touches.** The
alignment tax is real and is invisible in preference metrics by construction.
{{cite:ouyang2022}}'s pretraining-gradient mixing is the standard mitigation.

**Version the reward model with the policy.** A policy is only meaningful
relative to the reward model it was optimised against, and re-running with a
different reward model is a different experiment.

**What to monitor:** KL from reference, mean and distribution of response
length, refusal rate, reward-model score distribution, and the untouched
capability set. Refusal rate is the one that moves in the direction nobody
wants and is easiest to miss.

## 11. Common Mistakes

**Beginners:**

*Comparing reward values across prompts.* {{eq:bradley-terry}} identifies reward
only up to an additive function of the prompt, so cross-prompt comparisons are
meaningless — {{sec:6-mathematical-foundation}} demonstrates it in one line.

*Treating the reward model as ground truth.* It is a fitted approximation, and
{{eq:over-optimisation}} says optimising it hard is precisely what exposes its
errors.

*Skipping the KL penalty.* The policy drifts into text the reward model likes
and people do not.

**Experienced practitioners:**

*Tuning $\beta$ against the fitted reward.* It is monotone in $1/\beta$, so this
procedure always says "optimise harder". Tune against held-out human judgement
or not at all.

*Ignoring length bias.* It is the most reliably learned spurious feature in
preference data, and it compounds across stages.

*Treating ranked pairs from one prompt as independent.*
{{eq:reward-model-loss}} assumes independence; $\binom{k}{2}$ pairs from one
ranking are correlated, and the effective sample size is smaller than the pair
count suggests.

*Not measuring annotator agreement.* Without it, reward-model accuracy is
uninterpretable — you cannot tell a good model from one fitting noise.

*Assuming more preference data always helps.* The `preference-data-planning`
listing shows saturation, and the ceiling is the annotators.

## 12. Failure Modes

**Reward hacking.** The policy finds responses scoring highly that people
dislike — repetitive phrasings, flattery, confident hedging. *Symptom:* rising
reward, flat or falling human evaluation. *Detection:* held-out judging;
KL as an early proxy.

**Over-optimisation.** The general case {{eq:over-optimisation}}, of which
reward hacking is the visible form. *Mitigation:* KL penalty, early stopping on
human evaluation.

**Alignment tax.** Capability regression on tasks alignment did not cover.
*Detection:* the untouched capability set. *Mitigation:* pretraining-gradient
mixing.

**Length inflation.** Responses grow without a corresponding quality gain.
*Detection:* length as a tracked metric; length-controlled human evaluation.

**Refusal over-generalisation.** Safety preferences teach refusal of benign
requests sharing surface features. *Symptom:* rising refusal rate on innocuous
inputs. *This is the failure users notice most and it is directly optimised for
by well-intentioned preference data.*

**Annotator drift.** Guidelines shift over a months-long collection, so early
and late labels encode different preferences. *Detection:* agreement measured
per batch, not once.

**Mode collapse.** The policy converges on a narrow response style that scores
well. *Detection:* diversity statistics over generations; a falling entropy of
response openings.

## 13. Alternatives

{#tbl:alignment-methods caption="Ways to align a model to preferences. The first two need no preference data at all; the middle rows are this chapter and the next; the last changes where the normative content lives."}

| Method | Data | Stages | Models in memory | Where treated |
|---|---|---|---|---|
| Prompting / system prompt | none | 0 | 1 | {{ch:llm-prompting}} |
| SFT on curated demonstrations | demonstrations | 1 | 1 | {{ch:fm-instruction-tuning}} |
| Best-of-$n$ at inference | comparisons (for the RM) | 2 | 2 | {{sec:15-advanced-concepts}} |
| RLHF with PPO | comparisons | 3 | **4** | this chapter |
| DPO | comparisons | 2 | 2 | {{ch:fm-dpo}} |
| Constitutional AI / RLAIF | a written constitution | 3 | 4 | {{cite:bai2022}} |

**What genuinely differs.** SFT and prompting cannot express a preference
between two acceptable answers — that is the gap this chapter fills. RLHF and
DPO optimise the *same objective* {{eq:rlhf-objective}} and differ in
machinery, which is the next chapter's argument. Constitutional AI changes the
*source* of the preference signal, and with it the kind of artefact the
alignment target is.

**Best-of-$n$ deserves more attention than it gets.** Sample $n$ responses and
return the one the reward model scores highest. No policy training at all, and
it captures a surprising fraction of RLHF's benefit — at the cost of $n$ times
the inference. It is the correct baseline for any RLHF project, and it is
frequently skipped.

## 14. Evaluation

**Is the reward model any good?**

1. **Pairwise accuracy on held-out comparisons**, read against annotator
   agreement — the ceiling from `preference-data-planning`.
2. **Calibration**: does a reward gap of $\Delta$ predict a win rate of
   $\sigma(\Delta)$? {{eq:bradley-terry}} says it should, and departures
   indicate the model is not Bradley–Terry-shaped.
3. **Length correlation.** If reward correlates strongly with length, the model
   has learned a proxy.

**Is the policy better?**

1. **Human preference against the SFT baseline**, length-controlled. This is the
   only measurement that matters and everything else is a proxy for it.
2. **KL from reference**, as the over-optimisation indicator.
3. **Capability on an untouched set**, for the tax.
4. **Refusal and diversity statistics**, for the failures users notice.

**What not to do.** Do not evaluate the policy with the reward model it was
optimised against. That number rises by construction
({{eq:over-optimisation}}) and carries no information about quality — it is the
most common self-deception in this stage.

## 15. Advanced Concepts

**Best-of-$n$ and rejection sampling.** {{maturity:ESTABLISHED}} Inference-time
alignment with no policy training. Strong baseline, linear inference cost, and
directly comparable to RLHF by the KL it implicitly induces.

**Reward-model ensembles.** {{maturity:EMERGING}} Averaging several reward
models, or penalising their disagreement, to reduce the $\varepsilon$ term of
{{eq:over-optimisation}}. Disagreement is a usable proxy for "outside the
fitting distribution".

**Process supervision.** {{maturity:EMERGING}} Rewarding intermediate reasoning
steps rather than only final answers, which gives denser signal and less room
for a correct answer reached badly. Central in {{part:16}}.

**RLAIF and constitutional methods.** {{maturity:ESTABLISHED}}
{{cite:bai2022}}'s AI-generated feedback against written principles. Cheaper,
more consistent than human labels, and it makes the normative content
inspectable — which is the more interesting property.

**Online versus offline preference data.** {{maturity:EMERGING}} Collecting
comparisons on the *current* policy's outputs rather than a fixed set. This is
where online methods retain an advantage over DPO, and it is the strongest
remaining argument for the RL machinery.

## 16. Connection to Previous Chapters

**Backwards.** {{ch:fm-instruction-tuning}} produced $\pi_{\text{ref}}$ and
identified the gap this chapter fills. {{ch:ml-logistic}} is
{{eq:bradley-terry}} — the Bradley–Terry model is logistic regression on a
reward difference. {{ch:dl-losses}} supplies the KL that leashes the policy.
{{ch:math-optimization}} supplies the Lagrangian that yields
{{eq:rlhf-optimal-policy}}. {{ch:mle-hpo}}'s winner's curse is exactly
{{eq:over-optimisation}}'s selection-on-error structure, one level up.
{{ch:nlp-extraction}}'s insistence on annotator agreement as a ceiling applies
here at much higher cost.

**Forwards.** {{ch:fm-dpo}} inverts {{eq:rlhf-optimal-policy}} and removes the
reward model and the RL loop entirely — the derivation in
{{sec:6-mathematical-foundation}} is what makes that possible.
{{part:16}} applies process supervision to reasoning. {{part:25}} builds the
human-evaluation discipline this chapter depends on and cannot supply.
{{part:26}} is about prompts that defeat the alignment this chapter installs,
and {{part:27}} about whose preferences the comparisons encode.

## 17. Exercises

**Beginner**

1. Why can a demonstration not express a preference?
2. Given rewards 3.2 and 1.9, compute the Bradley–Terry win probability.
3. What is the KL penalty for, in one sentence?

**Intermediate**

4. Show that adding $c(x)$ to both rewards leaves {{eq:bradley-terry}}
   unchanged, and say what that implies for reporting.
5. Compute the loss contribution of a pair with reward gap $-0.8$ that was
   labelled in favour of the lower-reward response.
6. With $k=5$ sampled responses, how many pairs does one ranking yield? Why is
   treating them as independent optimistic?

**Advanced**

7. Derive {{eq:rlhf-optimal-policy}} in full, stating where the constraint and
   the multiplier enter.
8. Explain {{eq:over-optimisation}} and relate it to the winner's curse in
   {{ch:mle-hpo}}. What does the KL penalty bound?
9. Show that as $\beta\to 0$ the optimal policy in
   {{eq:rlhf-optimal-policy}} concentrates on $\argmax_y r(x,y)$, and say why
   that is undesirable.

**Implementation**

10. Extend `bradley-terry-reward-model` with a held-out set and plot accuracy
    against the number of comparisons, marking the annotator-agreement ceiling.
11. Add a length feature to the reward model's inputs, make the synthetic
    annotators mildly prefer longer responses, and show the reward model
    learning length as a proxy.
12. Implement best-of-$n$ against the fitted reward model and compare its true
    reward against the KL-optimal policy at matched KL.
13. Implement reward-model ensembling in `reward-over-optimisation` and show
    that penalising disagreement pushes the true-reward peak to a higher KL.

**Reasoning**

14. Your reward model scores 0.91 pairwise accuracy and annotators agree 0.74 of
    the time. What is happening?
15. Argue for and against replacing human preference labels with a written
    constitution, in terms of what each makes inspectable.

## 18. Interview Questions

**Beginner**

1. What is RLHF and what problem does it solve?
2. What is a reward model?
3. Why is there a KL penalty?

**Intermediate**

4. Derive the Bradley–Terry loss.
5. What is reward hacking and how would you detect it?
6. Why does stage 3 need four models in memory?

**Senior**

7. How would you set $\beta$? What would you measure?
8. Your aligned model scores better on the reward model and worse with users.
   Diagnose it.
9. When would you use best-of-$n$ instead of RLHF?

**Systems**

10. Design the preference-collection pipeline: sampling, ranking interface,
    agreement measurement, and versioning.
11. What do you monitor during policy optimisation, and what triggers a stop?

## 19. Research Questions

**How much of RLHF's benefit is the RL?** {{ch:fm-dpo}} suggests little. Compare
RLHF, DPO, and best-of-$n$ at matched KL from the reference, with
length-controlled human evaluation. Matched KL is the control that makes the
comparison meaningful and is rarely applied.

**Can over-optimisation be detected without human evaluation?** Reward-model
disagreement and KL are proxies. Measure how well each predicts the true-reward
peak across settings — a reliable proxy would remove the main cost of tuning
$\beta$.

**Whose preferences do reward models encode?** Annotator pools are not
representative, and {{eq:bradley-terry}} assumes a single latent reward for
everyone. Fit per-annotator rewards and measure the disagreement structure; if
it is large, the single-reward assumption is doing real damage that pooling
hides.

**Does the alignment tax have to exist?** It is observed and mitigated, not
explained. Is it capacity reallocation, distribution shift, or an artefact of
how the tax is measured? Different answers imply different fixes.

## 20. Chapter Summary

Alignment is the situation where **the objective cannot be written down**.
{{cite:christiano2017}}'s answer is to learn it: people compare reliably even
when they cannot score, so fit a reward function to their comparisons and
optimise against that.

**The Bradley–Terry model** {{eq:bradley-terry}} is the bridge — logistic
regression on a reward difference — and it identifies reward only up to an
additive function of the prompt, so absolute reward values carry no information
and cross-prompt comparisons are meaningless.

**The objective is reward minus a KL leash** {{eq:rlhf-objective}}, and its
optimum has a closed form: the reference policy reweighted by the exponentiated
reward {{eq:rlhf-optimal-policy}}. That equation says alignment *tilts an
existing distribution* rather than creating behaviour — the same conclusion
{{eq:continuation-mixture}} reached for instruction tuning — and it is the
equation {{ch:fm-dpo}} inverts to remove the RL entirely.

**Over-optimisation is structural, not a defect.** The policy maximises
$r^* + \varepsilon$ and therefore selects responses where the fitted reward's
error is favourable {{eq:over-optimisation}}. True quality rises, peaks, and
falls while predicted reward climbs throughout — and the predicted curve, the
one visible during training, gives no warning. The KL penalty does not improve
the reward model; it keeps the policy where the reward model has evidence.

**Setting $\beta$ is therefore not tunable against anything cheap.** The
`kl-penalty-tradeoff` listing shows both extremes fail for different reasons and
the optimum is interior — but fitted reward is monotone in $1/\beta$, so tuning
against it always says "optimise harder". Only held-out human evaluation
locates the peak.

The result that justifies all of it: {{cite:ouyang2022}}'s 1.3B aligned model
preferred to the 175B base. And the caveats that come with it — length bias
amplified from the previous stage, refusal over-generalisation, an alignment tax
invisible in preference metrics, and a ceiling set by annotator agreement rather
than by the model.

## 21. Further Reading

{{cite:ouyang2022}} is the paper. Read §3 for the pipeline and §4.1 for the
result, and note how much of the paper is about *data collection* rather than
algorithms — that ratio is the honest picture of where the work is.

{{cite:christiano2017}} is worth reading for its origin outside language
entirely. The method was developed on Atari and simulated robotics, which
separates the idea — learn the reward from comparisons — from the domain it is
now associated with.

{{cite:stiennon2020}} is the over-optimisation paper and its §4.3 is the figure
this chapter's second listing reproduces. It is also the cleanest demonstration
that the KL penalty is doing necessary work rather than being a regulariser
someone added out of habit.

{{cite:schulman2017ppo}} for the optimiser. Read §3 for the clipped objective;
the rest is about continuous control and is not what RLHF uses it for. Note that
it has no peer-reviewed venue, which for a paper this influential is worth
registering.

{{cite:bai2022}} for constitutional methods, and specifically for the appendix
containing the constitution itself — reading the actual principles is more
informative than the method description, and it is the clearest illustration of
what becomes inspectable when the normative content is written down.

**Where to go next:** {{ch:fm-dpo}} takes {{eq:rlhf-optimal-policy}}, inverts it,
and shows that the reward model and the entire RL loop can be removed — leaving
a single supervised loss on preference pairs.
