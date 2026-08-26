---
id: fm-dpo
number: 86
part: IX
tier: full
status: draft
requires: [fm-rlhf, fm-instruction-tuning, dl-losses, math-probability,
           ml-logistic, math-optimization]
provides: [direct-preference-optimisation, implicit-reward, dpo-loss,
           offline-preference-learning, online-versus-offline, preference-descendants,
           likelihood-displacement, reference-free-alignment]
citations: [rafailov2023, ouyang2022, christiano2017, stiennon2020,
            schulman2017ppo, bai2022, touvron2023llama]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Invert {{eq:rlhf-optimal-policy}} to express reward in terms of the policy.
2. Derive the DPO loss from the Bradley–Terry likelihood and explain why the
   partition function cancels.
3. Explain what the implicit reward is and verify it recovers a latent ordering.
4. State precisely what DPO removes from the RLHF pipeline and what it keeps.
5. Explain the online/offline distinction and why it is the strongest remaining
   argument for RL-based methods.
6. Describe likelihood displacement and why DPO can reduce the probability of
   preferred responses.
7. Choose between DPO and RLHF on evidence rather than fashion.

## 2. Why This Matters

**This is the most satisfying derivation in the part.** Three pages of algebra
remove a reward model, a value model, a sampling loop, and PPO — replacing them
with a single supervised loss on preference pairs. Nothing is approximated. The
result is exact given the same assumptions RLHF already makes.

**It changed what a small team can do.** {{ch:fm-rlhf}}'s stage 3 holds four
models in memory and generates samples inside the training loop
({{tbl:rlhf-stages}}). DPO holds two and generates nothing. That is the
difference between alignment being a research-lab capability and a
fine-tuning-script capability, and it is why open-weight models are aligned at
all.

**And it raises an uncomfortable question about the previous chapter.** If a
three-stage RL pipeline can be replaced by one classification loss with
comparable results, the RL machinery was not where the value lay. **The value is
in the preference data.** That conclusion is worth stating plainly because the
secondary literature treats RLHF's complexity as essential rather than
incidental.

**The honest qualification is equally important.** Frontier labs largely still
use online methods, and the evidence that DPO closes the gap is stronger for
open models than at the very top. {{sec:13-alternatives}} gives both, and
{{sec:19-research-questions}} says what would settle it.

## 3. Prerequisites

{{ch:fm-rlhf}} is essential and this chapter is unreadable without it —
specifically {{eq:bradley-terry}}, {{eq:rlhf-objective}}, and
{{eq:rlhf-optimal-policy}}, which is the equation this chapter inverts.
{{ch:fm-instruction-tuning}} for the SFT model that becomes the reference.
{{ch:dl-losses}} for cross-entropy and log-probabilities.
{{ch:math-probability}} and {{ch:ml-logistic}} for the Bradley–Terry likelihood.
{{ch:math-optimization}} for the constrained optimisation being inverted.

## 4. Intuitive Explanation

{{ch:fm-rlhf}} ended with a closed-form result that deserved more attention than
it got. The optimal policy under the KL-regularised objective is

$$
\pi^*(y\given x) \propto \pi_{\text{ref}}(y\given x)\exp\big(r(x,y)/\beta\big)
$$

**Read it backwards.** If the optimal policy is determined by the reward, then
the reward is determined by the optimal policy. Rearranging:

$$
r(x,y) = \beta\log\frac{\pi^*(y\given x)}{\pi_{\text{ref}}(y\given x)} + \beta\log Z(x)
$$

So **any policy implicitly defines a reward function** — the scaled log-ratio
between it and the reference. There is no need to fit a separate reward model;
the policy is one.

Now substitute this into the Bradley–Terry likelihood
{{eq:bradley-terry}}, which depends only on the *difference* of two rewards for
the same prompt. The awkward $\beta\log Z(x)$ term depends on $x$ alone, so **it
appears in both rewards and cancels.** What remains is a loss you can compute
from log-probabilities of the policy and the reference on the two responses:

$$
\Loss_{\text{DPO}} = -\log\sigma\left(
\beta\log\frac{\pi_\theta(y_w\given x)}{\pi_{\text{ref}}(y_w\given x)}
- \beta\log\frac{\pi_\theta(y_l\given x)}{\pi_{\text{ref}}(y_l\given x)}\right)
$$

**That is the whole method.** Four forward passes per example — policy and
reference, on the preferred and dispreferred response — and a logistic loss. No
reward model, no sampling, no PPO, no value function.

> NOTE: The derivation is not an approximation of RLHF. It is the same objective
> solved differently: the reward model in RLHF is an intermediate representation
> that turns out to be eliminable. Whether that changes the *result* is an
> empirical question and the answer is "mostly not, and the exceptions are
> interesting" — {{sec:13-alternatives}}.

**What is genuinely lost.** RLHF samples from the *current* policy and gets
feedback on those samples. DPO trains on a fixed dataset of comparisons
collected from some other policy. That is the difference between online and
offline learning, and it matters: as the policy moves away from whatever
generated the training pairs, the data becomes progressively less relevant to
where the policy now is.

**The mental model:** DPO recognises that the policy already encodes a reward,
so fitting a separate one is redundant. Where it breaks down: the identity holds
at the *optimum*, and gradient descent on the DPO loss is not guaranteed to
travel there in the way the derivation implies — {{sec:12-failure-modes}}'s
likelihood displacement is what that looks like in practice.

## 5. Formal Explanation

### 5.1 Inverting the optimal policy

From {{eq:rlhf-optimal-policy}}:

$$
\pi^*(y\given x) = \frac{1}{Z(x)}\pi_{\text{ref}}(y\given x)
 \exp\!\Big(\frac{1}{\beta}r(x,y)\Big)
$$

Take logs and solve for $r$:

$$
r(x,y) = \beta\log\frac{\pi^*(y\given x)}{\pi_{\text{ref}}(y\given x)}
 + \beta\log Z(x)
$$ (eq:implicit-reward)

**This is exact**, and it says the reward is recoverable from the policy up to a
function of $x$ alone — which is precisely the non-identifiability
{{ch:fm-rlhf}} noted for Bradley–Terry. The two facts are the same fact.

### 5.2 The cancellation

Substitute {{eq:implicit-reward}} into {{eq:bradley-terry}}:

$$
\Prob[y_w \succ y_l\given x] = \sigma\big(r(x,y_w) - r(x,y_l)\big)
$$

$$
= \sigma\!\left(
 \beta\log\frac{\pi^*(y_w\given x)}{\pi_{\text{ref}}(y_w\given x)}
 + \beta\log Z(x)
 - \beta\log\frac{\pi^*(y_l\given x)}{\pi_{\text{ref}}(y_l\given x)}
 - \beta\log Z(x)\right)
$$

$$
= \sigma\!\left(
 \beta\log\frac{\pi^*(y_w\given x)}{\pi_{\text{ref}}(y_w\given x)}
 - \beta\log\frac{\pi^*(y_l\given x)}{\pi_{\text{ref}}(y_l\given x)}\right)
$$ (eq:dpo-probability)

**$Z(x)$ cancels because it does not depend on $y$.** That single observation is
what makes the method possible: $Z(x)$ is a sum over all possible responses and
is completely intractable, and it never has to be computed.

### 5.3 The DPO loss

Maximising the likelihood of the observed preferences under
{{eq:dpo-probability}}, over a dataset of comparisons:

$$
\Loss_{\text{DPO}}(\theta) = -\E_{(x,y_w,y_l)\sim\Data}
 \left[\log\sigma\!\left(
 \beta\log\frac{\pi_\theta(y_w\given x)}{\pi_{\text{ref}}(y_w\given x)}
 - \beta\log\frac{\pi_\theta(y_l\given x)}{\pi_{\text{ref}}(y_l\given x)}
 \right)\right]
$$ (eq:dpo-loss)

Define the **implicit reward**:

$$
\hat{r}_\theta(x,y) = \beta\log\frac{\pi_\theta(y\given x)}{\pi_{\text{ref}}(y\given x)}
$$ (eq:dpo-implicit-reward)

so {{eq:dpo-loss}} is exactly {{eq:reward-model-loss}} with $\hat{r}_\theta$ in
place of a separately-fitted $r_\phi$. **DPO is reward-model training where the
reward model is parameterised by the policy.**

### 5.4 The gradient, and what it does

Differentiating {{eq:dpo-loss}}:

$$
\nabla_\theta\Loss_{\text{DPO}} = -\beta\,\E\Big[
 \sigma\big(\hat{r}_\theta(x,y_l) - \hat{r}_\theta(x,y_w)\big)
 \big(\nabla_\theta\log\pi_\theta(y_w\given x)
 - \nabla_\theta\log\pi_\theta(y_l\given x)\big)\Big]
$$ (eq:dpo-gradient)

Two components:

- **The direction** raises the log-probability of $y_w$ and lowers that of
  $y_l$ — a contrastive update, structurally identical to
  {{eq:negsampling-gradient}} from {{ch:nlp-static-embeddings}}.
- **The weight** $\sigma(\hat{r}_l - \hat{r}_w)$ is large when the implicit
  reward has the pair *ordered wrongly* and near zero when it already has them
  right. **The loss automatically focuses on examples the model currently gets
  wrong**, which is the same self-weighting as negative sampling.

### 5.5 What is removed and what remains

{#tbl:dpo-versus-rlhf caption="RLHF against DPO, component by component. The right-hand column is what the derivation eliminates; the last two rows are what it does not."}

| Component | RLHF | DPO |
|---|---|---|
| SFT stage | required | required |
| Preference data | required | required |
| Reward model | trained separately | **implicit in the policy** |
| Value model | required for PPO | **none** |
| Sampling in the loop | required | **none** |
| Models in memory | 4 | 2 |
| KL control | explicit penalty | **implicit via $\beta$ and the reference** |
| Online feedback | yes | **no** |
| Hyperparameters | many (PPO's) | $\beta$, and the usual optimiser settings |

**The last two rows are the substance of the remaining debate.** DPO's KL
control is implicit — there is no term explicitly bounding divergence, only the
reference appearing in the ratio — and its data is fixed, so it cannot obtain
feedback on responses the current policy would actually produce.

## 6. Mathematical Foundation

### 6.1 The derivation, complete

Collecting the argument in one place, since it is the chapter.

**Step 1.** The RLHF objective {{eq:rlhf-objective}} has optimum
{{eq:rlhf-optimal-policy}}:

$$
\pi^*(y\given x) = \frac{\pi_{\text{ref}}(y\given x)e^{r(x,y)/\beta}}{Z(x)}
$$

**Step 2.** Invert for $r$, giving {{eq:implicit-reward}}.

**Step 3.** Bradley–Terry {{eq:bradley-terry}} depends only on reward
differences at fixed $x$. Substituting, the $\beta\log Z(x)$ terms cancel,
giving {{eq:dpo-probability}}.

**Step 4.** Maximum likelihood over the comparison dataset gives
{{eq:dpo-loss}}.

$\square$

**Where each assumption enters.** Step 1 assumes the KL-regularised objective is
the right one. Step 3 assumes Bradley–Terry — that preferences depend on a
difference of scalars. **DPO makes no assumption RLHF does not already make**,
which is what distinguishes it from an approximation.

### 6.2 Why $\beta$ still controls the leash

There is no explicit KL term in {{eq:dpo-loss}}, which invites the belief that
DPO has no KL control. It does, implicitly.

From {{eq:dpo-implicit-reward}}, the loss is a function of
$\beta\log(\pi_\theta/\pi_{\text{ref}})$. To move the loss appreciably, the
log-ratio must change by $O(1/\beta)$. **Large $\beta$ therefore means small
log-ratio changes suffice**, so the policy stays near the reference; small
$\beta$ requires large ratio changes and permits large drift.

The relationship to RLHF's explicit penalty is exact at the optimum — both
reach {{eq:rlhf-optimal-policy}} for the same $\beta$ — and inexact along the
optimisation path, which is one source of the practical differences observed
between the methods.

### 6.3 Likelihood displacement

A property of {{eq:dpo-gradient}} that surprises people, and it follows directly
from the algebra.

The gradient raises $\log\pi_\theta(y_w)$ and lowers $\log\pi_\theta(y_l)$. But
the loss depends only on the *difference* of the implicit rewards. **Nothing in
the objective requires $\pi_\theta(y_w)$ to increase in absolute terms** — the
loss falls just as well if both probabilities decrease and $y_l$'s decreases
faster.

$$
\hat{r}_w - \hat{r}_l \uparrow
\quad\text{is satisfiable by}\quad
\pi_\theta(y_w)\downarrow,\ \pi_\theta(y_l)\downarrow\downarrow
$$ (eq:likelihood-displacement)

$\square$

**This is observed in practice**: DPO training frequently *decreases* the
probability of preferred responses while the loss falls and the accuracy of the
implied ordering improves. The mass has to go somewhere, and it goes to
responses not in the dataset — which may be better, worse, or degenerate, and
the objective is indifferent.

This is the clearest example in the chapter of the gap between the derivation's
optimum and gradient descent's path.

### 6.4 A worked calculation

A preference pair with reference log-probabilities
$\log\pi_{\text{ref}}(y_w) = -12.0$, $\log\pi_{\text{ref}}(y_l) = -11.0$ — the
reference actually prefers the *dispreferred* response, which is why this pair
is informative.

At initialisation $\pi_\theta = \pi_{\text{ref}}$, so both implicit rewards are
zero and

$$
\Loss = -\log\sigma(0) = \log 2 = 0.693
$$

After training, suppose $\log\pi_\theta(y_w) = -10.5$ and
$\log\pi_\theta(y_l) = -13.0$. With $\beta = 0.1$:

$$
\hat{r}_w = 0.1(-10.5 + 12.0) = 0.15,
\qquad
\hat{r}_l = 0.1(-13.0 + 11.0) = -0.20
$$

$$
\Loss = -\log\sigma(0.15 + 0.20) = -\log\sigma(0.35) = 0.554
$$

**Note that the loss at initialisation is always $\log 2$**, for every pair,
because the policy starts equal to the reference. That is the diagnostic
equivalent of {{eq:init-loss}} from {{ch:fm-pretraining}}: a DPO run whose loss
does not start at 0.693 has loaded the wrong reference or is computing
log-probabilities incorrectly.

## 7. Internal Mechanics

```mermaid {#fig:dpo-mechanics caption="One DPO step. Four forward passes and a logistic loss; the reference model is frozen and its two passes can be precomputed once for the whole dataset, reducing the training loop to two passes per example."}
graph TD
  A["preference pair<br/>(x, y_w, y_l)"] --> B["policy: log pi(y_w|x)"]
  A --> C["policy: log pi(y_l|x)"]
  A --> D["reference: log pi_ref(y_w|x)"]
  A --> E["reference: log pi_ref(y_l|x)"]
  B --> F["implicit reward r_w<br/>eq:dpo-implicit-reward"]
  D --> F
  C --> G["implicit reward r_l"]
  E --> G
  F --> H["logistic loss on r_w - r_l<br/>eq:dpo-loss"]
  G --> H
  H --> I["gradient step on the policy only"]
  style D fill:#dfe,stroke:#5a5
  style E fill:#dfe,stroke:#5a5
```

**The reference passes are precomputable.** $\pi_{\text{ref}}$ is frozen, so its
log-probabilities on the dataset can be computed once and cached. The training
loop then needs only two forward passes per example, and the reference model
need not be resident at all — which halves the memory requirement again.

**Sequence-level log-probabilities.** $\log\pi_\theta(y\given x)$ is the sum over
the response's tokens of the per-token log-probability, conditioned on the
prompt. This is the same masked computation as
{{eq:loss-masking}} in {{ch:fm-instruction-tuning}}, without the averaging —
and using the *mean* instead of the *sum* is a common implementation variant
that changes the effective $\beta$ per example by the response length. Whether
to normalise by length is a real decision with a length-bias consequence.

**Why $\beta$ values do not transfer.** Between the sum/mean choice, tokenizer
differences, and the KL estimator, a $\beta$ that works in one implementation
may be wrong by an order of magnitude in another. Published values should be
treated as starting points and re-tuned.

**The SFT stage is not optional.** The reference must be a model that already
produces plausible responses, both because the implicit reward is relative to it
and because the preference data was collected on outputs from something like it.
Running DPO from a base model is a common mistake and produces poor results for
reasons that have nothing to do with the method.

**Numerical care in the log-ratio.** {{eq:dpo-implicit-reward}} is a difference
of two sequence log-probabilities, each a sum of several hundred token
log-probabilities in the range $[-20, 0]$. The difference is typically small
relative to its operands, so the computation is a subtraction of two large
similar numbers — the classic catastrophic-cancellation shape. In practice this
is handled by keeping both terms in float32 even when the model runs in bf16,
and it is worth knowing because the failure is a silently noisy gradient rather
than an overflow anyone would notice.

**The gradient vanishes on easy pairs, and this is a feature until it is not.**
{{eq:dpo-gradient}}'s weight $\sigma(\hat{r}_l - \hat{r}_w)$ falls toward zero
once a pair is confidently ordered, so late in training most of the batch
contributes almost nothing and the effective batch size collapses to the
disputed examples. That is the correct behaviour — there is nothing left to
learn from a pair the model already has right — but it means the *observed* loss
flattens while a small subset of examples continues to drive large updates.
Monitoring the fraction of the batch with a non-negligible weight is a cheap and
informative diagnostic, and it is what distinguishes "converged" from
"dominated by a handful of mislabelled pairs".

**Why the reference cannot simply be dropped.** It is tempting to note that
$\pi_{\text{ref}}$ appears only as a constant offset per response and conclude
it could be absorbed. It cannot: the offset differs *per response*, so removing
it changes which responses the loss considers improved. Reference-free methods
({{sec:15-advanced-concepts}}) do exist, but they replace the reference with a
different anchor rather than eliminating the need for one — without any anchor,
nothing distinguishes raising the preferred response from lowering everything
else, which is {{eq:likelihood-displacement}} with no counterweight at all.

## 8. Implementation

The loss, the implicit reward, and the verification that it recovers a latent
ordering without ever fitting a reward model.

```python {tier=A name=dpo-loss-and-implicit-reward}
"""DPO from scratch, on an explicit small distribution where everything is exact."""
import numpy as np

rng = np.random.default_rng(0)

N_PROMPTS, N_RESPONSES, BETA = 6, 12, 0.5

# A reference policy: a proper distribution over responses for each prompt.
ref_logits = rng.normal(size=(N_PROMPTS, N_RESPONSES))
ref = np.exp(ref_logits) / np.exp(ref_logits).sum(1, keepdims=True)

# A latent reward we never give the algorithm. DPO must recover its ORDERING
# from comparisons alone, with no reward model anywhere.
true_reward = rng.normal(size=(N_PROMPTS, N_RESPONSES))


def sample_pairs(n):
    """Comparisons drawn per Bradley-Terry on the latent reward."""
    out = []
    for _ in range(n):
        x = rng.integers(N_PROMPTS)
        a, b = rng.choice(N_RESPONSES, size=2, replace=False)
        p_a = 1 / (1 + np.exp(-(true_reward[x, a] - true_reward[x, b])))
        if rng.random() < p_a:
            out.append((x, a, b))
        else:
            out.append((x, b, a))
    return out


pairs = sample_pairs(6000)
print(f"{N_PROMPTS} prompts x {N_RESPONSES} responses, {len(pairs)} comparisons")
print(f"beta = {BETA}\n")

# The policy is parameterised by logits, initialised AT the reference — which
# is what makes the initial loss exactly log 2.
policy_logits = ref_logits.copy()


def policy_probs(logits):
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def implicit_reward(logits):
    """Equation (eq:dpo-implicit-reward)."""
    return BETA * (np.log(policy_probs(logits) + 1e-12) - np.log(ref + 1e-12))


def dpo_loss_and_grad(logits):
    """Equation (eq:dpo-loss) and its gradient with respect to the logits."""
    p = policy_probs(logits)
    logp = np.log(p + 1e-12)
    r = BETA * (logp - np.log(ref + 1e-12))

    total, grad = 0.0, np.zeros_like(logits)
    for x, w, l in pairs:
        margin = r[x, w] - r[x, l]
        total += -np.log(1 / (1 + np.exp(-margin)) + 1e-12)
        # d/d margin of -log sigma(margin) = -(1 - sigma(margin))
        coef = -(1 - 1 / (1 + np.exp(-margin))) * BETA
        # d logp[x,i] / d logits[x,j] = delta_ij - p[x,j]
        grad[x] += coef * ((np.eye(N_RESPONSES)[w] - p[x])
                           - (np.eye(N_RESPONSES)[l] - p[x]))
    return total / len(pairs), grad / len(pairs)


loss0, _ = dpo_loss_and_grad(policy_logits)
print(f"loss at initialisation : {loss0:.6f}")
print(f"log 2                  : {np.log(2):.6f}   <- section 6.4's diagnostic")
assert abs(loss0 - np.log(2)) < 1e-6, \
    "with policy == reference every implicit reward is 0, so the loss is log 2"

for step in range(1, 4001):
    loss, grad = dpo_loss_and_grad(policy_logits)
    policy_logits -= 12.0 * grad
    if step in (1, 500, 2000, 4000):
        print(f"step {step:>4}: DPO loss {loss:.4f}")

# Did the implicit reward recover the latent ORDERING, with no reward model?
r_hat = implicit_reward(policy_logits)
agree, total = 0, 0
for x in range(N_PROMPTS):
    for a in range(N_RESPONSES):
        for b in range(a + 1, N_RESPONSES):
            total += 1
            agree += ((true_reward[x, a] > true_reward[x, b])
                      == (r_hat[x, a] > r_hat[x, b]))
print(f"\nimplicit-reward ordering agreement with the latent reward: "
      f"{agree / total:.4f}")
assert agree / total > 0.85, "the implicit reward should recover the ordering"

# And does the policy match the closed form eq:rlhf-optimal-policy?
analytic = ref * np.exp(true_reward / BETA)
analytic /= analytic.sum(1, keepdims=True)
learned = policy_probs(policy_logits)
corr = float(np.corrcoef(analytic.ravel(), learned.ravel())[0, 1])
print(f"correlation with the closed-form optimal policy: {corr:.4f}")

print("""
No reward model was fitted anywhere in this listing. The ordering was recovered
from the policy's own log-ratio against the reference, which is what
eq:implicit-reward says must be possible — and the learned policy tracks the
closed form of eq:rlhf-optimal-policy that the previous chapter derived.""")
```

Now the property that surprises people, which falls straight out of
{{eq:likelihood-displacement}}:

```python {tier=A name=likelihood-displacement}
"""DPO can lower the probability of the PREFERRED response. Here is why."""
import numpy as np

rng = np.random.default_rng(0)
N, D, BETA = 40, 16, 0.5

# The setup that matters: y_w and y_l are SIMILAR. Both are plausible answers
# to the same prompt differing in a detail, which is the normal case in real
# preference data — and they share parameters, so a gradient that pushes one
# down drags the other with it.
feat = rng.normal(size=(N, D))
shared = rng.normal(size=D)
feat[0] = shared + 0.25 * rng.normal(size=D)      # preferred
feat[1] = shared + 0.25 * rng.normal(size=D)      # dispreferred, very similar

W, L = 0, 1
cos_wl = float(feat[W] @ feat[L]
               / (np.linalg.norm(feat[W]) * np.linalg.norm(feat[L])))
print(f"cosine similarity between the two responses: {cos_wl:.3f}")
print("(this is the crux — dissimilar responses do not displace)\n")

theta0 = rng.normal(size=D) * 0.3


def probs(theta):
    z = feat @ theta
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()


ref = probs(theta0)
theta = theta0.copy()

print(f"{'step':>6} {'loss':>9} {'P(preferred)':>14} {'P(dispreferred)':>17} "
      f"{'P(all others)':>15}")
for step in range(0, 801):
    p = probs(theta)
    margin = BETA * (np.log(p[W] / ref[W]) - np.log(p[L] / ref[L]))
    loss = -np.log(1 / (1 + np.exp(-margin)) + 1e-12)
    if step in (0, 50, 200, 400, 800):
        print(f"{step:>6} {loss:>9.4f} {p[W]:>14.6f} {p[L]:>17.6f} "
              f"{1 - p[W] - p[L]:>15.6f}")
    # Gradient of the margin: d log p[i] / d theta = feat[i] - E_p[feat]
    coef = (1 - 1 / (1 + np.exp(-margin))) * BETA
    mean_feat = p @ feat
    theta += 3.0 * coef * ((feat[W] - mean_feat) - (feat[L] - mean_feat))

final = probs(theta)
print(f"\nP(preferred)    {ref[W]:.6f} -> {final[W]:.6f}   "
      f"{'DOWN' if final[W] < ref[W] else 'up'}")
print(f"P(dispreferred) {ref[L]:.6f} -> {final[L]:.6f}")
print(f"P(all others)   {1 - ref[W] - ref[L]:.6f} -> "
      f"{1 - final[W] - final[L]:.6f}")

assert final[W] < ref[W], "this configuration should displace the preferred response"

print("""
The loss fell from log 2 to near zero and the implied ordering is correct — the
method did exactly what it was asked. And the probability of the PREFERRED
response went down.

Equation (eq:likelihood-displacement) is why: the objective constrains only the
DIFFERENCE of the two implicit rewards, so it is satisfied by pushing the
dispreferred response down hard, and because the two responses are similar and
share parameters, the preferred one is dragged down with it. The displaced mass
lands on responses that appear in NO comparison, about which the preference data
says nothing at all.

Note the dependence on similarity. Repeat this with two unrelated responses and
the preferred one rises as expected — displacement is a consequence of
preference pairs being NEARLY THE SAME, which is exactly what good preference
data looks like. That is why DPO implementations monitor the absolute
log-probability of chosen responses and not only the loss.""")
```

And the online/offline distinction, which is the real remaining argument:

```python {tier=A name=online-versus-offline}
"""Why a fixed preference dataset degrades as the policy moves away from it."""
import numpy as np

rng = np.random.default_rng(7)
D, N_CANDIDATES = 10, 3000

true_w = rng.normal(size=D)


def quality(v):
    return v @ true_w


# Preference data is collected by sampling from SOME policy. Offline DPO uses
# a fixed dataset collected from the SFT model; online methods re-collect from
# the CURRENT policy at every round.
sft_centre = np.zeros(D)


def collect(centre, n=400):
    """Comparisons between responses sampled around `centre`."""
    a = centre + rng.normal(size=(n, D))
    b = centre + rng.normal(size=(n, D))
    return a, b, quality(a) > quality(b)


def informative_fraction(data_centre, policy_centre):
    """What share of a dataset collected at `data_centre` discriminates between
    responses the policy at `policy_centre` would actually produce?"""
    a, b, _ = collect(data_centre, N_CANDIDATES)
    # Responses the current policy plausibly generates: those near its centre.
    near = (np.linalg.norm(a - policy_centre, axis=1) < 3.2) & \
           (np.linalg.norm(b - policy_centre, axis=1) < 3.2)
    return float(near.mean())


print(f"{'policy drift from SFT':>22} {'offline data relevance':>24} "
      f"{'online data relevance':>23}")
for drift in (0.0, 1.0, 2.0, 3.0, 4.0, 6.0):
    direction = true_w / np.linalg.norm(true_w)
    policy_centre = sft_centre + drift * direction
    offline = informative_fraction(sft_centre, policy_centre)
    online = informative_fraction(policy_centre, policy_centre)
    print(f"{drift:>22.1f} {offline:>24.3f} {online:>23.3f}")

print("""
Offline relevance collapses as the policy moves; online relevance does not,
because the data is re-collected where the policy now is.

This is the one thing DPO's derivation does not give it. The algebra is exact,
the reward model is genuinely redundant, and none of that addresses the fact
that a fixed dataset describes preferences over responses the policy has since
stopped producing. It is also why iterated DPO — alternate training and
re-collecting comparisons from the current policy — recovers much of the gap,
and why frontier labs, which can afford continuous collection, have less reason
to abandon online methods than a small team does.""")
```

## 9. Practical Example

A team has an SFT model, 12,000 preference pairs, and two A100s. They want
alignment. RLHF's stage 3 needs four models resident and generation in the
loop; DPO needs two and no generation. The question is what the choice actually
costs in memory, time, and quality.

```python {tier=A name=dpo-versus-rlhf-budget}
"""What each method costs on the hardware a small team actually has."""

PARAMS = 7e9
BYTES_PER_PARAM_BF16 = 2
OPTIMIZER_BYTES_PER_PARAM = 12       # Adam fp32 moments + master weights
GPU_MEMORY_GB = 80
N_PAIRS = 12_000
AVG_TOKENS = 512

weights_gb = PARAMS * BYTES_PER_PARAM_BF16 / 1e9
trainable_gb = weights_gb + PARAMS * OPTIMIZER_BYTES_PER_PARAM / 1e9

# LoRA (Part XIV) trains a small adapter, so the optimiser state is tiny and
# the base weights stay frozen. Included because it is what actually makes any
# of this fit on one device.
LORA_FRACTION = 0.005

CONFIGS = {
    "RLHF (PPO)": dict(trainable=1, frozen=3, generates=True, lora=False),
    "DPO": dict(trainable=1, frozen=1, generates=False, lora=False),
    "DPO, reference cached": dict(trainable=1, frozen=0, generates=False, lora=False),
    "DPO + LoRA, ref cached": dict(trainable=1, frozen=0, generates=False, lora=True),
}

print(f"{PARAMS / 1e9:.0f}B parameters, bf16 weights, Adam optimiser")
print(f"  weights          : {weights_gb:.1f} GB")
print(f"  trainable copy   : {trainable_gb:.1f} GB (weights + optimiser state)\n")

print(f"{'configuration':<24} {'models':>8} {'memory GB':>11} "
      f"{'fits 80GB':>11} {'generates':>11}")
for name, c in CONFIGS.items():
    if c["lora"]:
        # Base weights frozen; optimiser state only for the adapter.
        mem = weights_gb + PARAMS * LORA_FRACTION * (
            BYTES_PER_PARAM_BF16 + OPTIMIZER_BYTES_PER_PARAM) / 1e9
    else:
        mem = c["trainable"] * trainable_gb + c["frozen"] * weights_gb
    n_models = c["trainable"] + c["frozen"]
    print(f"{name:<24} {n_models:>8} {mem:>11.1f} "
          f"{str(mem < GPU_MEMORY_GB):>11} {str(c['generates']):>11}")

# Training cost: forward+backward is ~6ND; a frozen forward pass is ~2ND.
fwd_bwd = 6 * PARAMS
fwd_only = 2 * PARAMS
tokens = N_PAIRS * AVG_TOKENS * 2          # two responses per pair

dpo_flops = tokens * (fwd_bwd + fwd_only)          # policy trained, ref frozen
dpo_cached = tokens * fwd_bwd                      # reference precomputed once
# PPO additionally generates samples (sequential, memory-bound) and runs a
# reward model and a value model over them.
ppo_flops = tokens * (fwd_bwd * 2 + fwd_only * 2) * 4   # x4 for PPO epochs

print(f"\n{'method':<24} {'train FLOPs':>14} {'relative':>10}")
for name, f in [("DPO", dpo_flops), ("DPO, reference cached", dpo_cached),
                ("RLHF (PPO)", ppo_flops)]:
    print(f"{name:<24} {f:>14.2e} {f / dpo_flops:>9.1f}x")

print(f"""
Read the fits-80GB column honestly: FULL fine-tuning of a {PARAMS / 1e9:.0f}B
model does not fit on one 80 GB device under any method, because the Adam state
alone is {PARAMS * OPTIMIZER_BYTES_PER_PARAM / 1e9:.0f} GB. That is the 16N
accounting from ch:tf-complexity, and it is why Part XIV exists.

What the memory column does decide is how many devices. RLHF needs
{140.0 / weights_gb:.0f} model-copies' worth at {140.0:.0f} GB; DPO needs
{112.0:.0f} GB; caching the reference log-probabilities removes the frozen model
and brings it to {98.0:.0f} GB. Only the last row — LoRA on top of a cached
reference — actually fits on one device, and it is the configuration most small
teams really run.

The FLOP column understates the gap, because PPO's cost is dominated by
GENERATION inside the training loop, which is sequential and memory-bandwidth
bound (ch:tf-masking-kv) rather than compute bound. Wall-clock differs by more
than the arithmetic shows.

None of this says DPO produces a better model. It says DPO is a training script
and RLHF is an infrastructure project — which is why the open-weight ecosystem
aligned on DPO regardless of how the quality question comes out.""")
```

> PRODUCTION TIP: Cache the reference log-probabilities before training. They
> are constant, they take one pass over the dataset, and caching them removes an
> entire model from the training loop's memory footprint. Teams routinely keep
> the reference resident out of habit.

## 10. Production Considerations

**Verify the initial loss is $\log 2 = 0.693$.** With $\pi_\theta =
\pi_{\text{ref}}$ every implicit reward is zero, so the loss is exactly
$\log 2$ for every pair. Anything else means the reference is wrong or the
log-probabilities are being computed incorrectly — the cheapest bug-catch in the
chapter, and the direct analogue of {{eq:init-loss}}.

**Monitor absolute log-probabilities, not only the loss.**
{{eq:likelihood-displacement}} means the loss can fall while the preferred
response's probability falls too. Track $\log\pi_\theta(y_w)$ directly.

**Re-tune $\beta$ for your implementation.** The sum-versus-mean choice on
sequence log-probabilities changes the effective $\beta$ by a factor of the
response length. Published values do not transfer.

**Do not run DPO from a base model.** The reference must produce plausible
responses. This is a common and expensive mistake.

**Cache the reference log-probabilities.** One pass, then the reference model
is not needed at all.

**Consider iterating.** Train, sample fresh responses from the new policy,
collect new comparisons, repeat. This recovers much of what offline learning
gives up, at the cost of a collection round — and it is the honest middle ground
between DPO and full online RLHF.

**What to monitor:** loss, implicit-reward margin, $\log\pi_\theta(y_w)$ and
$\log\pi_\theta(y_l)$ separately, KL from the reference (computable even though
it is not in the objective), and response length.

## 11. Common Mistakes

**Beginners:**

*Skipping SFT.* The reference must be a competent model.

*Using a mismatched reference.* $\pi_{\text{ref}}$ must be the model the policy
was initialised from. Using a different checkpoint makes the implicit reward
meaningless.

*Copying $\beta$ from a paper.* See above — it does not transfer across
implementations.

**Experienced practitioners:**

*Monitoring only the loss.* Likelihood displacement is invisible in it by
construction.

*Averaging instead of summing token log-probabilities without adjusting
$\beta$.* This silently makes $\beta$ length-dependent and introduces a length
bias of its own.

*Assuming DPO and RLHF are interchangeable.* They optimise the same objective
and differ in what data they can use. The online/offline gap is real and is the
reason the debate is not settled.

*Training too long.* DPO overfits preference data readily, and the symptom is
displacement — the margin keeps improving while the model degrades. Early
stopping on held-out generation quality, not on the loss.

## 12. Failure Modes

**Likelihood displacement.** Preferred-response probability falls while the loss
improves {{eq:likelihood-displacement}}. *Detection:* track absolute
log-probabilities. *Mitigation:* larger $\beta$, early stopping, or an added SFT
term on the preferred responses.

**Distribution shift from offline data.** The policy moves away from what
generated the comparisons, and the data stops being informative.
*Detection:* KL from reference, plus the relevance measurement in
`online-versus-offline`. *Mitigation:* iterate.

**Overfitting the preference set.** Margins keep growing on training pairs and
generation quality falls. *Detection:* held-out generation evaluation, not
held-out loss.

**Degenerate output from over-optimisation.** Small $\beta$ and long training
produce repetitive or truncated text. *This is {{ch:fm-rlhf}}'s
over-optimisation appearing in a method with no explicit KL term* — the leash is
implicit and easy to loosen accidentally.

**Reference mismatch.** Silent, and it makes every implicit reward wrong.
*Detection:* the $\log 2$ initial-loss check.

**Length bias from the sum/mean choice.** Summing favours short responses under
some configurations and averaging changes the effective $\beta$ per example.
*Detection:* correlate the implicit-reward margin against response length.

## 13. Alternatives

{#tbl:preference-methods caption="Preference-optimisation methods. The first three optimise the same objective by different routes; the rest adjust an assumption in the derivation. All of them need the same preference data, which is the expensive input."}

| Method | Removes | Keeps | Adjusts |
|---|---|---|---|
| RLHF (PPO) | — | everything | — |
| DPO {{cite:rafailov2023}} | reward model, value model, sampling | Bradley–Terry, KL via $\beta$ | — |
| Best-of-$n$ | policy training entirely | reward model | inference cost instead |
| IPO | — | — | drops Bradley–Terry for a bounded objective |
| KTO | paired comparisons | — | uses unpaired thumbs-up/down signals |
| ORPO | the reference model | — | folds preference into the SFT loss |
| Iterated DPO | sampling loop, not collection | — | re-collects data between rounds |

**What genuinely differs.** DPO and RLHF optimise the *same objective* and
differ in machinery. IPO changes the objective, arguing Bradley–Terry
over-optimises when preferences are near-deterministic. KTO changes the *data
requirement*, which matters enormously in practice — unpaired feedback is far
cheaper to collect than rankings. ORPO removes the reference model, folding the
preference term into SFT.

**The honest summary of the state of play.** For open models with a fixed
preference dataset, DPO or a descendant is the default and the quality gap
against RLHF is small or absent. Frontier labs largely retain online methods,
and the best explanation is the online/offline distinction rather than anything
about the reward model — which is exactly what `online-versus-offline` measures.

## 14. Evaluation

**Is the implementation correct?**

1. **Initial loss $= \log 2$** {{sec:6-mathematical-foundation}}. This catches
   reference mismatches and log-probability bugs.
2. **Implicit-reward ordering** on held-out pairs — the accuracy of
   $\hat{r}_\theta$ at ranking comparisons it was not trained on.
3. **Reference is frozen** — assert its parameters do not change.
4. **Log-probability computation** matches a reference implementation on a
   handful of sequences.

**Is the model better?**

1. **Held-out human preference against the SFT baseline**, length-controlled.
   Same standard as {{ch:fm-rlhf}} and for the same reason.
2. **Absolute log-probabilities of preferred responses**, for displacement.
3. **KL from the reference**, computed even though it is not optimised — it is
   the comparable quantity across methods, and comparing DPO against RLHF at
   *matched KL* is the only fair comparison.
4. **General capability on an untouched set**, for the alignment tax.

**On comparing methods.** Most published DPO-versus-RLHF comparisons do not
control for KL from the reference, which means they may be comparing different
points on the same tradeoff curve rather than different curves. That is the
standing question of this book — *what was held fixed?* — applied to the
chapter's own central claim.

## 15. Advanced Concepts

**Iterated and online DPO.** {{maturity:ESTABLISHED}} Alternate training with
fresh comparison collection from the current policy. Recovers much of the
online advantage and is now common practice.

**IPO and bounded objectives.** {{maturity:EMERGING}} Bradley–Terry's likelihood
is unbounded as preferences become deterministic, so DPO can push margins
arbitrarily far on easy pairs. IPO replaces it with a bounded objective.

**KTO and unpaired feedback.** {{maturity:EMERGING}} Learns from independent
thumbs-up/down rather than rankings, which transforms the data-collection
economics — production systems already collect unpaired signals for free.

**Reference-free methods.** {{maturity:EMERGING}} ORPO and relatives remove
$\pi_{\text{ref}}$ entirely by folding a preference term into the SFT loss.
Simpler still, with a weaker theoretical grounding.

**Token-level credit assignment.** {{maturity:RESEARCH FRONTIER}} DPO treats a
response as one unit. Assigning credit to the tokens actually responsible is
the same idea as process supervision in {{part:16}} and is largely open.

## 16. Connection to Previous Chapters

**Backwards.** This chapter is an inversion of {{ch:fm-rlhf}}'s
{{eq:rlhf-optimal-policy}}, and everything else follows from that one algebraic
step. {{eq:bradley-terry}} supplies the likelihood; the reward's
non-identifiability there is the same fact as $Z(x)$ cancelling here.
{{eq:dpo-gradient}}'s self-weighting is structurally
{{eq:negsampling-gradient}} from {{ch:nlp-static-embeddings}} — attract the
positive, repel the negative, weighted by how wrong you currently are.
{{ch:fm-instruction-tuning}}'s masked log-probability is what
$\log\pi_\theta(y\given x)$ computes.

**Forwards.** {{part:14}} makes this parameter-efficient, which is what puts it
on a single consumer GPU. {{part:16}} applies preference optimisation to
reasoning traces with token-level credit. {{part:25}} supplies the
length-controlled human evaluation that {{sec:14-evaluation}} requires and this
chapter cannot provide.

## 17. Exercises

**Beginner**

1. Why is the DPO loss exactly $\log 2$ at initialisation?
2. What does DPO remove from the RLHF pipeline? List each component.
3. What is the implicit reward?

**Intermediate**

4. Show that $Z(x)$ cancels in {{eq:dpo-probability}} and say why that is the
   key step.
5. With $\beta=0.2$, reference log-probs $-8.0$ and $-7.5$, and policy log-probs
   $-6.0$ and $-9.0$, compute the implicit rewards and the loss.
6. Explain why $\beta$ controls drift despite there being no KL term in
   {{eq:dpo-loss}}.

**Advanced**

7. Derive {{eq:dpo-gradient}} and interpret the weighting factor.
8. Prove {{eq:likelihood-displacement}}: show the loss can decrease while
   $\pi_\theta(y_w)$ decreases.
9. DPO and RLHF share an optimum. Explain why they can still reach different
   models, and what that implies for comparing them.

**Implementation**

10. Extend `dpo-loss-and-implicit-reward` with a held-out comparison set and
    plot implicit-reward accuracy against training steps, marking where it
    begins to overfit.
11. Implement IPO's bounded objective alongside DPO on the same data and compare
    the margins each produces on easy pairs.
12. Implement iterated DPO in `online-versus-offline`: re-collect comparisons at
    the current policy every round and show the relevance staying high.
13. Add an SFT term on the preferred responses to `likelihood-displacement` and
    find the weight at which displacement stops.

**Reasoning**

14. If DPO matches RLHF, what does that say about where the value in RLHF lay?
    What evidence would change your answer?
15. Your DPO run's loss falls steadily and generation quality degrades. Give the
    three most likely causes in order.

## 18. Interview Questions

**Beginner**

1. What is DPO and what problem does it solve?
2. What is the implicit reward?
3. Why does DPO need fewer models in memory?

**Intermediate**

4. Derive the DPO loss from the RLHF optimum.
5. Why does the partition function cancel?
6. What is likelihood displacement?

**Senior**

7. DPO or RLHF for your team? Walk through the decision.
8. How would you compare them fairly?
9. What does DPO give up, and how would you recover it?

**Systems**

10. Design a DPO training pipeline for a 7B model on limited hardware, including
    the reference-caching optimisation.
11. What do you monitor, and what triggers early stopping?

## 19. Research Questions

**Does DPO match RLHF at matched KL?** Most comparisons do not control for
divergence from the reference, so they may be comparing points on one curve.
Run the comparison with KL held fixed and length-controlled human evaluation.
This is the experiment the field's central claim rests on and it is rarely done
properly.

**How much of the gap is online data?** `online-versus-offline` measures data
relevance collapsing with drift. Quantify how much of RLHF's remaining advantage
iterated DPO recovers, as a function of iteration count.

**Can likelihood displacement be prevented without weakening the objective?**
Adding an SFT term works and dilutes the preference signal. A principled fix
would need to constrain absolute probabilities without changing the optimum.

**Which of DPO's descendants actually help?** IPO, KTO, ORPO and others each
adjust an assumption. A controlled comparison on identical data with identical
tuning budget would establish which adjustments matter — and, given this book's
recurring theme, some of the reported differences are probably tuning.

## 20. Chapter Summary

DPO begins from the observation that {{ch:fm-rlhf}}'s closed-form optimum can be
read backwards. If the optimal policy is the reference reweighted by the
exponentiated reward, then **the reward is recoverable from the policy**:
$r(x,y) = \beta\log(\pi^*/\pi_{\text{ref}}) + \beta\log Z(x)$
{{eq:implicit-reward}}.

Substituting that into the Bradley–Terry likelihood, **the intractable partition
function cancels** — it depends on the prompt alone and appears in both rewards
of a comparison {{eq:dpo-probability}}. What remains is a logistic loss on
log-probability ratios {{eq:dpo-loss}}, computable with four forward passes and
no reward model, no value model, no sampling, and no PPO.

**Nothing is approximated.** DPO makes no assumption RLHF does not already make;
it solves the same objective by a different route, and the reward model turns
out to be an eliminable intermediate. The gradient {{eq:dpo-gradient}} is
contrastive with a self-weighting that focuses on currently-misordered pairs —
structurally the same update as negative sampling in
{{ch:nlp-static-embeddings}}, ten years and several fields apart.

**Two things do not follow from the algebra.** {{eq:likelihood-displacement}}
shows the objective constrains only the *difference* of implicit rewards, so the
loss falls just as well when both probabilities drop and the dispreferred one
drops faster — DPO routinely reduces the probability of preferred responses, and
the displaced mass goes to responses no comparison covers. And the derivation
says nothing about *which* data: a fixed offline dataset describes preferences
over responses the policy has since stopped producing, which
`online-versus-offline` shows degrading with drift.

**The practical consequence is unambiguous even where the quality question is
not.** RLHF's stage 3 needs four models resident and generation inside the
training loop; DPO needs two, or one with the reference cached. That is the
difference between an infrastructure project and a training script, and it is
why open-weight alignment converged on DPO regardless of how the comparison
comes out.

And the uncomfortable inference stands: if one classification loss matches a
three-stage RL pipeline, the RL was not where the value lay. **The value is in
the preference data** — which is the expensive input to every method in
{{tbl:preference-methods}}.

## 21. Further Reading

{{cite:rafailov2023}} is short and §4 is the derivation. Read it with
{{ch:fm-rlhf}}'s {{eq:rlhf-optimal-policy}} beside you; the entire method is
three algebraic steps from that equation, and seeing it that way is more
valuable than the experiments.

{{cite:ouyang2022}} is worth rereading here specifically for its pipeline
diagram, now that you can see which of its components are eliminable and which
are not. The preference data survives; almost everything else does not.

{{cite:stiennon2020}} for over-optimisation, which does not go away because the
reward model became implicit. The KL leash in DPO is $\beta$, it is easier to
loosen by accident, and the failure looks the same.

{{cite:touvron2023llama}} and its successors document real alignment recipes on
downloadable models, several of which use DPO or a descendant. Reading a recipe
you can reproduce is worth more than a method paper you cannot.

**Where to go next:** {{ch:fm-distillation}} is the last chapter of this part and
the one that makes everything before it affordable — turning a large aligned
model into a small one that can actually be served.
