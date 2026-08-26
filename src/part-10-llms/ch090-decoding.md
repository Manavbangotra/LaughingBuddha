---
id: llm-decoding
number: 90
part: X
tier: full
status: draft
requires: [llm-next-token, llm-anatomy, math-probability, math-random-vars,
           dl-losses, nlp-subword]
provides: [greedy-decoding, temperature-sampling, top-k-sampling, nucleus-sampling,
           beam-search, degeneration, repetition-penalty, sampler-chain,
           likelihood-quality-gap, stochastic-decoding]
citations: [holtzman2020, fan2018, radford2019, brown2020, ouyang2022,
            wei2022cot, ji2023survey, touvron2023llama]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Implement greedy, temperature, top-k, top-p and beam search from a logit
   vector.
2. Explain why maximising sequence likelihood produces degenerate text, and
   reproduce the effect.
3. Derive what temperature does to the distribution and to its entropy.
4. Explain why top-p adapts to distributional sharpness and top-k does not.
5. Order a sampler chain correctly and say why the order matters.
6. Choose a decoding configuration from the task rather than by convention.
7. Explain which "model quality" complaints are actually decoding complaints.

## 2. Why This Matters

**This is the chapter with the highest ratio of practical value to conceptual
difficulty in the book.** The techniques are twenty lines each, they are applied
to every token any model ever produces, and getting them wrong is the single
most common cause of output that people attribute to the model.

**It contains a genuinely counter-intuitive empirical result.**
{{cite:holtzman2020}} showed that the *highest-probability* continuation is
repetitive, degenerate text that no human would write — and that human text sits
in a region of *moderate* surprise rather than maximal likelihood. Maximising the
objective the model was trained on produces output nobody wants. That deserves
to be understood rather than memorised.

**Almost every knob a user is given lives here.** Temperature, top-p, top-k,
repetition penalty, and beam width are decoding parameters, not model
parameters ({{tbl:not-in-the-model}}). A team that cannot reason about them is
tuning blind.

**And the task determines the answer.** Code generation, factual question
answering, creative writing and structured extraction want genuinely different
configurations, and the standard defaults are a compromise chosen for chat.

## 3. Prerequisites

{{ch:llm-next-token}} for what the probabilities mean and for the length bias in
{{eq:log-sequence-probability}}, which is why beam search behaves as it does.
{{ch:llm-anatomy}} for {{eq:next-token-distribution}} and
{{eq:scale-is-temperature}} — the latter is the whole of temperature.
{{ch:math-probability}} and {{ch:math-random-vars}} for sampling and entropy.
{{ch:nlp-subword}} for why token-level operations have odd effects on words.

## 4. Intuitive Explanation

The model gives you a distribution over 128,000 tokens. You must pick one. That
is the entire problem, and there are more sensible answers than people expect.

**Greedy: take the argmax.** Deterministic, fast, and it produces remarkably bad
text for anything open-ended. It also gets stuck: once the model repeats a
phrase, that phrase becomes more likely, so it repeats again.

**Sampling: draw from the distribution.** More varied, and it occasionally
selects something from the far tail — of 128,000 tokens, the bottom 100,000 hold
a small total probability, but "small" times "every token" is a near-certainty
of eventually picking something absurd.

**Truncation fixes that.** Restrict to the plausible tokens and renormalise.
Two ways: keep the top $k$ ({{cite:fan2018}}), or keep the smallest set whose
probability sums to $p$ ({{cite:holtzman2020}}).

**The difference between them is the whole argument for top-p.** Consider two
positions. After "The capital of France is" the distribution is sharp — one
token holds 0.95. After "She opened the door and saw" it is flat — hundreds of
continuations are reasonable. A fixed $k = 50$ admits 49 bad tokens in the first
case and cuts off hundreds of good ones in the second. **Top-p adapts, because
the *set size* varies with the distribution's shape while the probability mass
stays fixed.**

**Temperature reshapes before truncating.** Dividing logits by $T$ flattens the
distribution for $T>1$ and sharpens it for $T<1$. At $T\to 0$ it becomes greedy.

**And beam search is the interesting failure.** It keeps several candidate
sequences and extends the highest-scoring ones, approximating the *most likely
sequence* rather than sampling. That is the right objective for translation,
where there is a correct answer, and the wrong one for open-ended text — because
{{cite:holtzman2020}} showed the most likely sequence is degenerate.

> NOTE: The finding to sit with is this. The model was trained to maximise the
> likelihood of human text. Yet the highest-likelihood sequences *under that
> model* are not human-like at all. Human text is consistently more surprising
> than the model's most probable continuation, which means likelihood and
> quality diverge systematically rather than occasionally.

**The mental model:** decoding is a policy for converting a distribution into a
choice, and the policy has to be matched to whether the task has one right
answer. Where it breaks down: the policies interact — temperature changes what
top-p admits, and repetition penalties change what temperature does — so the
order of operations is part of the specification.

## 5. Formal Explanation

### 5.1 The decoding problem

Given $p(\cdot\given x, y_{<t})$ from {{eq:next-token-distribution}}, a decoding
strategy is a map from that distribution to a token. Repeated, it produces a
sequence.

**Two families**, with different objectives:

$$
\text{search:}\quad \hat{\vec{y}} = \argmax_{\vec{y}} \log P(\vec{y}\given x)
\qquad
\text{sampling:}\quad y_t \sim q\big(p(\cdot\given x,y_{<t})\big)
$$ (eq:decoding-families)

Search targets the single most likely sequence; sampling draws from a
transformed distribution. The choice between them is a choice about whether the
task has one right answer.

### 5.2 Temperature

$$
p_T(v) = \frac{\exp(z_v / T)}{\sum_w \exp(z_w / T)}
$$ (eq:decoding-temperature)

which by {{eq:scale-is-temperature}} is identical to scaling logits by $1/T$.

**Limits.** As $T\to 0^+$, $p_T$ concentrates on $\argmax_v z_v$ — greedy. As
$T\to\infty$, $p_T\to$ uniform. $T = 1$ is the model's own distribution.

Temperature is monotone in the entropy of the result
({{sec:6-mathematical-foundation}}), so it is a single dial from "deterministic"
to "random" with the model's own beliefs at 1.

### 5.3 Top-k

Keep the $k$ highest-probability tokens, renormalise:

$$
V_k = \operatorname{top-}k(p),
\qquad
p'(v) = \begin{cases}
 p(v)/\sum_{w\in V_k} p(w) & v \in V_k\\
 0 & \text{otherwise}
\end{cases}
$$ (eq:top-k)

**The flaw is a fixed set size against a varying distribution.** With
$k = 50$ at a position where the top token holds 0.99, the other 49 collectively
hold 0.01 and are all noise. At a position where the distribution is flat, 50 is
an arbitrary cut through a continuum of reasonable options.

### 5.4 Top-p (nucleus)

{{cite:holtzman2020}}. Sort by descending probability and keep the smallest
prefix whose cumulative probability reaches $p$:

$$
V_p = \min\Big\{ V' \subseteq V \ :\ \sum_{v\in V'} p(v) \ge p \Big\}
$$ (eq:top-p)

**The set size is now a function of the distribution.** Sharp distributions give
small nuclei; flat ones give large. That is the adaptivity top-k lacks, and it
is the entire contribution.

> IMPORTANT: Top-p does not bound the set size. On a near-uniform distribution
> at $p = 0.95$, the nucleus can contain tens of thousands of tokens. That is
> usually correct behaviour and occasionally not, which is why production
> configurations often apply top-k *and* top-p — the $k$ acting as a safety
> bound on an otherwise unbounded set.

### 5.5 Beam search

Maintain $B$ partial sequences; at each step extend every beam by every token,
score, and keep the best $B$. The score is
{{eq:length-normalised-score}}, because raw log-probability favours short
sequences ({{ch:llm-next-token}}).

$$
\text{beams}_{t} = \operatorname{top-}B\Big\{
 s(\vec{y}_{<t}\!\cdot\! v)\ :\ \vec{y}_{<t}\in\text{beams}_{t-1},\ v\in V\Big\}
$$ (eq:beam-search)

**Beam search is right for constrained tasks and wrong for open-ended ones.**
Translation, summarisation and structured extraction have a target the model
should find; conversation and creative writing do not, and searching harder for
the mode makes them worse.

### 5.6 The sampler chain and its order

Production decoders apply several transformations in sequence. **Order changes
the result**, and the conventional order is:

1. **Repetition/frequency penalties** — modify logits based on what has already
   been generated.
2. **Temperature** — scale logits.
3. **Top-k** — truncate by rank.
4. **Top-p** — truncate by mass.
5. **Sample** from what remains.

Penalties must precede temperature, because they operate on logits and
temperature rescales them; applying temperature first changes the penalty's
effective strength. Top-k before top-p bounds the set the nucleus is computed
over. **Different libraries use different orders**, which is one reason the same
nominal settings behave differently across stacks.

## 6. Mathematical Foundation

### 6.1 Temperature is monotone in entropy

Let $H(T)$ be the entropy of {{eq:decoding-temperature}}. Write
$\beta = 1/T$ and note $p_\beta(v) \propto e^{\beta z_v}$ is an exponential
family with natural parameter $\beta$ and sufficient statistic $z$.

For such a family,

$$
\frac{\partial}{\partial\beta}\E_\beta[z] = -\Var_\beta[z] \le 0
$$

and entropy relates to $\beta$ by

$$
\frac{\dd H}{\dd\beta} = -\beta\,\Var_\beta[z]
$$ (eq:entropy-temperature-derivative)

$\square$

**Since $\Var \ge 0$ and $\beta > 0$, entropy is non-increasing in $\beta$ and
therefore non-decreasing in $T$.** Temperature is a well-behaved single dial:
raising it always flattens, never sharpens, and the rate is proportional to the
logits' variance — so it has the most effect exactly where the model is most
opinionated.

### 6.2 Why maximum likelihood degenerates

The model assigns probability to sequences. Consider a repeated phrase
$\vec{r}$ appearing $n$ times. Because language models condition on their own
context, and repetition in the context raises the probability of further
repetition, we have approximately

$$
P(\vec{r}\ \text{at step } n+1 \given \vec{r}\ \text{repeated } n\ \text{times})
 \ \text{increasing in } n
$$ (eq:repetition-feedback)

This is a positive feedback loop. Any decoding strategy that always takes the
locally most probable token will enter it and cannot leave, because leaving
requires selecting a lower-probability token.

$\square$

**Greedy and beam search are therefore structurally susceptible and sampling is
not.** A sampler occasionally selects off-mode and breaks the loop; a maximiser
never does. That is the mechanism behind {{cite:holtzman2020}}'s degeneration
result, and it explains why the fix is *stochasticity* rather than a better
search.

### 6.3 The likelihood–quality gap

{{cite:holtzman2020}}'s empirical claim, stated precisely: let $H_t$ be the
per-token surprise $-\log p(y_t\given y_{<t})$ of human-written text under the
model. Then

$$
\E[H_t^{\text{human}}] \gg \E[H_t^{\text{greedy}}]
$$ (eq:human-surprise-gap)

and, more strikingly, human text's surprise *fluctuates* while greedy text's is
uniformly low.

**Human language is not maximally predictable and does not try to be.** A
speaker who only ever said the most predictable next word would be
uninformative — there is an information-theoretic argument here that the
communicative function of language requires surprise. Whatever the explanation,
the operational consequence is that **matching the model's own distribution
produces more human-like text than maximising it**, which is why sampling at
$T\approx 1$ beats search for open-ended generation.

### 6.4 A worked truncation calculation

Logits over six tokens: $(6.0, 5.6, 3.1, 3.0, 2.9, -1.0)$.

At $T = 1$, $\softmax$ gives approximately

$$
p = (0.478,\ 0.320,\ 0.026,\ 0.024,\ 0.021,\ 0.0004)
$$

with the remaining mass on the rest of the vocabulary.

**Top-$k$ with $k=2$:** keep $\{0.478, 0.320\}$, sum $0.798$, renormalise to
$(0.599, 0.401)$.

**Top-$p$ with $p=0.8$:** cumulative is $0.478$, then $0.798$ — which is just
under $0.8$, so the third token is included: three tokens, sum $0.824$,
renormalised $(0.580, 0.388, 0.032)$.

**Now flatten with $T = 1.5$.** Logits become $(4.0, 3.73, 2.07, 2.0, 1.93,
-0.67)$ and

$$
p \approx (0.395,\ 0.302,\ 0.057,\ 0.054,\ 0.050,\ 0.004)
$$

Top-$p$ at $0.8$ now needs four tokens ($0.395+0.302+0.057+0.054 = 0.808$).

**Temperature changed the nucleus size.** That is why the order in
{{sec:5-formal-explanation}} matters: applying temperature after truncation
would sample from a differently-sized set than applying it before.

## 7. Internal Mechanics

```mermaid {#fig:sampler-chain caption="The decoding pipeline for one token. Every stage transforms the logits or the support before a single draw is taken. Order is part of the specification — penalties act on raw logits, and truncation must see the post-temperature distribution."}
graph LR
  A["logits (|V|,)<br/>from ch:llm-anatomy"] --> B["repetition /<br/>frequency penalty"]
  B --> C["temperature<br/>z / T"]
  C --> D["top-k<br/>keep k by rank"]
  D --> E["top-p<br/>keep by mass"]
  E --> F["renormalise"]
  F --> G["draw one token"]
  G --> H["append, repeat"]
  H -.->|"next step"| A
  style C fill:#fde,stroke:#c69
  style G fill:#dfe,stroke:#5a5
```

**Repetition penalties, and their token-level pathology.** The common form
divides the logit of any already-generated token by a factor $\rho > 1$ (or
multiplies if negative). It works, and it operates on *tokens*, not words —
so with subword tokenization ({{ch:nlp-subword}}) penalising a token can
suppress unrelated words sharing a fragment, and a legitimately repeated word
(a name, a variable identifier in code) is penalised as heavily as filler.
**This is why repetition penalties are actively harmful for code generation**
and why structured tasks usually set them to 1.0.

**Numerical detail that matters.** Truncation sets probabilities to zero, which
is $-\infty$ in log space. Implementations mask by setting logits to a large
negative number rather than true $-\infty$, because $-\infty$ propagates NaN
through some softmax implementations when an entire row is masked. The
conventional value is around $-10^{4}$ in fp16 — large enough to zero the
probability, finite enough to be safe.

**Stop conditions are outside all of this.** Generation ends on an end-of-
sequence token, a maximum length, or a user-supplied stop string. Stop strings
are checked on *detokenized text*, which means a stop sequence can be split
across token boundaries and missed — a real bug class, and the reason production
stacks buffer a few tokens before emitting.

**Why the first token is special.** At step 1 the model has only the prompt to
condition on, so its distribution is typically flatter than at any later step —
the prompt constrains what follows far less than the prompt *plus forty
generated tokens* does. A nucleus computed at $p=0.9$ is therefore widest
exactly where the model knows least, and an unlucky first token propagates
through the entire generation because everything after it is conditioned on it.
This is the mechanism behind a familiar observation: **re-running a generation
usually changes it completely rather than slightly.** The divergence is not
accumulated drift, it is one early branch.

**Logit bias, and what it is really doing.** Most APIs expose a per-token
additive bias applied before everything else in
{{fig:sampler-chain}}. Since it acts on logits, a bias of $b$ multiplies that
token's odds by $e^{b}$ — so the scale is exponential and a bias of 10 is
effectively a hard include, while $-100$ is a hard ban. It is the crudest form
of the constraint that {{ch:llm-structured-output}} makes principled, and it is
worth recognising as the same operation: **both work by making certain tokens
unreachable, and differ only in whether the choice of which tokens is computed
by a grammar or supplied by hand.**

**Seeding.** Sampling is stochastic, so reproducibility requires a seed. Most
APIs expose one, and even with it, batching can change results: floating-point
reduction order varies with batch composition, which perturbs logits in the last
bits, which occasionally changes which token is drawn near a boundary.
**Bit-exact reproducibility across batch sizes is not generally available**, and
promising it to a user is a mistake.

## 8. Implementation

Every sampler, implemented from a logit vector.

```python {tier=A name=samplers-from-scratch}
"""Greedy, temperature, top-k, top-p — the complete set, from logits."""
import numpy as np

rng = np.random.default_rng(0)


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def apply_temperature(logits, T):
    """Equation (eq:decoding-temperature). T -> 0 is greedy, T -> inf uniform."""
    if T <= 0:
        out = np.full_like(logits, -1e9)
        out[logits.argmax()] = 0.0
        return out
    return logits / T


def top_k_filter(logits, k):
    """Equation (eq:top-k). Mask all but the k highest logits."""
    if k <= 0 or k >= len(logits):
        return logits
    threshold = np.partition(logits, -k)[-k]
    return np.where(logits < threshold, -1e9, logits)


def top_p_filter(logits, p):
    """Equation (eq:top-p). Keep the smallest prefix reaching cumulative p."""
    if not (0 < p < 1):
        return logits
    probs = softmax(logits)
    order = np.argsort(-probs)
    cumulative = np.cumsum(probs[order])
    # Keep everything up to and INCLUDING the token that crosses p.
    n_keep = int(np.searchsorted(cumulative, p) + 1)
    keep = order[:n_keep]
    out = np.full_like(logits, -1e9)
    out[keep] = logits[keep]
    return out


# A realistic-looking distribution: a few plausible tokens and a long tail.
V = 2000
logits = np.concatenate([
    np.array([6.0, 5.6, 3.1, 3.0, 2.9]),
    rng.normal(-2.0, 1.0, V - 5),
])

base = softmax(logits)
print(f"vocabulary {V}, top-5 probabilities: "
      f"{np.round(np.sort(base)[::-1][:5], 4).tolist()}")
print(f"mass in the tail beyond the top 5: {1 - np.sort(base)[::-1][:5].sum():.4f}")
print("That tail is small per token and there are 1,995 of them — which is why "
      "unfiltered sampling eventually picks something absurd.\n")


def nucleus_size(logits, p):
    probs = softmax(logits)
    return int(np.searchsorted(np.cumsum(np.sort(probs)[::-1]), p) + 1)


print(f"{'T':>5} {'entropy':>9} {'top-1 prob':>12} {'nucleus @0.9':>14} "
      f"{'nucleus @0.95':>15}")
for T in (0.2, 0.5, 0.7, 1.0, 1.3, 2.0):
    z = apply_temperature(logits, T)
    pr = softmax(z)
    ent = float(-(pr * np.log(pr + 1e-12)).sum())
    print(f"{T:>5.1f} {ent:>9.4f} {pr.max():>12.4f} "
          f"{nucleus_size(z, 0.90):>14} {nucleus_size(z, 0.95):>15}")

print("""
Entropy rises monotonically with T, which is equation
(eq:entropy-temperature-derivative). And the nucleus GROWS with temperature —
so temperature and top-p are not independent knobs. Raising temperature while
holding p fixed widens the sampled set twice over: once by flattening the
distribution and once by admitting more tokens into the nucleus.""")

# Top-k against top-p on distributions of different sharpness.
sharp = np.concatenate([np.array([9.0, 2.0, 1.5]), rng.normal(-3.0, 1.0, V - 3)])
flat = np.concatenate([np.linspace(3.0, 2.0, 40), rng.normal(-1.0, 1.0, V - 40)])

print(f"\n{'distribution':<12} {'top-1':>8} {'entropy':>9} "
      f"{'top-k=50 keeps':>16} {'top-p=0.9 keeps':>17}")
for name, lg in [("sharp", sharp), ("flat", flat)]:
    pr = softmax(lg)
    ent = float(-(pr * np.log(pr + 1e-12)).sum())
    print(f"{name:<12} {pr.max():>8.4f} {ent:>9.4f} {50:>16} "
          f"{nucleus_size(lg, 0.9):>17}")

print("""
This is the argument for top-p in one table. A fixed k=50 keeps fifty tokens
whatever the distribution looks like: on the sharp one that admits 49 tokens of
noise, and on the flat one it truncates a genuine continuum. Top-p keeps a
handful on the sharp distribution and many more on the flat one, because it
fixes the MASS and lets the set size follow.""")

# The full chain, in the order of section 5.6.
def sample(logits, T=1.0, k=0, p=1.0, generator=None):
    z = apply_temperature(logits, T)
    z = top_k_filter(z, k)
    z = top_p_filter(z, p)
    probs = softmax(z)
    g = generator or rng
    return int(g.choice(len(probs), p=probs))


counts = {}
for _ in range(4000):
    tok = sample(logits, T=1.0, k=0, p=0.9)
    counts[tok] = counts.get(tok, 0) + 1
print(f"\n4,000 draws at T=1.0, p=0.9 selected {len(counts)} distinct tokens "
      f"(nucleus size {nucleus_size(logits, 0.9)})")
assert len(counts) <= nucleus_size(logits, 0.9), "sampling must stay in the nucleus"
print("Sampling never left the nucleus — truncation is a hard constraint, not "
      "a bias.")
```

Now the result the chapter is built around:

```python {tier=A name=degeneration}
"""Maximising likelihood produces repetitive text. Sampling does not."""
import numpy as np

rng = np.random.default_rng(1)
V, CONTEXT = 60, 6


def make_model():
    """A toy autoregressive model with the ONE property that matters here:
    a token becomes more likely the more it has recently appeared, which is
    equation (eq:repetition-feedback)."""
    base = rng.normal(size=(V, V))          # base bigram preferences

    def logits(history):
        z = base[history[-1]].copy()
        recent = history[-CONTEXT:]
        for tok in recent:
            z[tok] += 1.1                   # self-reinforcement
        return z
    return logits


model = make_model()


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def generate(strategy, n=140, T=1.0, p=0.9, seed=0):
    g = np.random.default_rng(seed)
    hist = [int(g.integers(V))]
    for _ in range(n):
        z = model(hist)
        if strategy == "greedy":
            nxt = int(z.argmax())
        elif strategy == "sample":
            pr = softmax(z / T)
            nxt = int(g.choice(V, p=pr))
        elif strategy == "nucleus":
            pr = softmax(z / T)
            order = np.argsort(-pr)
            cum = np.cumsum(pr[order])
            keep = order[:int(np.searchsorted(cum, p) + 1)]
            q = pr[keep] / pr[keep].sum()
            nxt = int(g.choice(keep, p=q))
        hist.append(nxt)
    return hist[1:]


def repetition_rate(seq, n=3):
    """Fraction of n-grams that are repeats."""
    grams = [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]
    return 1 - len(set(grams)) / len(grams)


def distinct_ratio(seq):
    return len(set(seq)) / len(seq)


def mean_surprise(seq):
    """Equation (eq:human-surprise-gap): -log p of each chosen token."""
    hist, total = [seq[0]], 0.0
    for tok in seq[1:]:
        pr = softmax(model(hist))
        total += -np.log(pr[tok] + 1e-12)
        hist.append(tok)
    return total / (len(seq) - 1)


print(f"{'strategy':<22} {'distinct':>10} {'3-gram repeat':>15} "
      f"{'mean surprise':>15}")
results = {}
for label, kwargs in [("greedy", dict(strategy="greedy")),
                      ("sample T=0.7", dict(strategy="sample", T=0.7)),
                      ("sample T=1.0", dict(strategy="sample", T=1.0)),
                      ("nucleus T=1.0 p=0.9", dict(strategy="nucleus", T=1.0, p=0.9))]:
    seq = generate(**kwargs)
    r = (distinct_ratio(seq), repetition_rate(seq), mean_surprise(seq))
    results[label] = r
    print(f"{label:<22} {r[0]:>10.3f} {r[1]:>15.3f} {r[2]:>15.4f}")

g = results["greedy"]
s = results["sample T=1.0"]
print(f"\ngreedy repeats {g[1]:.1%} of its 3-grams; sampling at T=1 repeats "
      f"{s[1]:.1%}")
print(f"greedy's mean surprise is {g[2]:.3f} nats, sampling's is {s[2]:.3f}")
assert g[1] > s[1], "greedy must repeat more than sampling"
assert g[2] < s[2], "greedy must produce lower-surprise (higher-likelihood) text"

# Show the loop forming.
seq = generate(strategy="greedy", n=60)
print(f"\ngreedy output (token ids): {seq[:40]}")
tail = seq[-20:]
print(f"last 20 tokens          : {tail}")
print(f"distinct tokens in tail : {len(set(tail))} of 20")

print("""
This is holtzman2020's result in miniature. Greedy decoding produces text that
is MORE probable under the model — lower mean surprise, by construction, since
it takes the argmax every step — and it is degenerate: it falls into a loop and
cannot leave, because leaving requires choosing a lower-probability token.

Sampling produces higher-surprise, less probable text that does not degenerate.
The fix for degeneration is stochasticity, not a better search, and equation
(eq:repetition-feedback) is why: a maximiser is structurally unable to escape a
positive feedback loop.""")
```

And beam search, showing where it is right and where it is wrong:

```python {tier=A name=beam-search}
"""Beam search: correct for constrained tasks, wrong for open-ended ones."""
import numpy as np

rng = np.random.default_rng(2)


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def beam_search(logit_fn, n_steps, beam_width, V, alpha=1.0):
    """Equation (eq:beam-search), with length normalisation."""
    beams = [([], 0.0)]
    for _ in range(n_steps):
        cands = []
        for seq, score in beams:
            lp = np.log(softmax(logit_fn(seq)) + 1e-12)
            for v in range(V):
                cands.append((seq + [v], score + float(lp[v])))
        cands.sort(key=lambda sc: sc[1] / (len(sc[0]) ** alpha), reverse=True)
        beams = cands[:beam_width]
    return beams[0]


def greedy_decode(logit_fn, n_steps):
    seq, total = [], 0.0
    for _ in range(n_steps):
        lp = np.log(softmax(logit_fn(seq)) + 1e-12)
        v = int(lp.argmax())
        total += float(lp[v])
        seq.append(v)
    return seq, total


# ---------------------------------------------------------------------------
# TASK A: a constrained task with a correct answer, containing a GREEDY TRAP —
# a token that is locally best at step 0 and leads nowhere. This is the exact
# situation beam search exists for, and note that the model is a DETERMINISTIC
# function of the history: beam search over a stochastic scorer is meaningless.
# ---------------------------------------------------------------------------
VA, TARGET, TRAP = 12, [2, 5, 8], 3


def constrained_logits(history):
    z = np.full(VA, -4.0)
    step = len(history)
    if step == 0:
        z[TRAP] = 2.0            # locally the best choice...
        z[TARGET[0]] = 1.6       # ...and this one is second best
    elif step == 1:
        if history[0] == TRAP:
            z[:] = -1.0          # ...but the trap leads to a flat, poor region
            z[4] = 0.2
        elif history[0] == TARGET[0]:
            z[TARGET[1]] = 3.2   # while the target path is rich
    else:
        if history[:2] == TARGET[:2]:
            z[TARGET[2]] = 3.4
        elif history[0] == TRAP:
            z[6] = 0.1
    return z


print("TASK A — constrained: there IS a correct answer\n")
print(f"target: {TARGET}   (step 0 has a trap at token {TRAP})")
print(f"{'method':<12} {'output':<14} {'log P':>9} {'exact match':>13}")
out, lp = greedy_decode(constrained_logits, len(TARGET))
print(f"{'greedy':<12} {str(out):<14} {lp:>9.3f} {str(out == TARGET):>13}")
for width in (2, 3, 5):
    out_b, lp_b = beam_search(constrained_logits, len(TARGET), width, VA)
    print(f"{'beam ' + str(width):<12} {str(out_b):<14} {lp_b:>9.3f} "
          f"{str(out_b == TARGET):>13}")

print("""
Greedy takes the locally-best token at step 0 and is then stuck in a region
where everything is mediocre. Beam search keeps the second-best prefix alive
long enough to discover that it leads somewhere much better, and finds a
sequence with more than twice the log-probability.

This is what beam search is FOR, and it is why it remains standard in
translation and structured extraction — tasks where a target exists and local
choices can be traps.""")

# ---------------------------------------------------------------------------
# TASK B: open-ended, with the repetition feedback of eq:repetition-feedback.
# ---------------------------------------------------------------------------
VB = 24
base = rng.normal(size=(VB, VB))


def open_logits(history):
    z = (base[history[-1]] if history else base[0]).copy()
    for tok in history[-5:]:
        z[tok] += 1.2                     # eq:repetition-feedback
    return z


def rep_rate(seq, n=3):
    grams = [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]
    return 1 - len(set(grams)) / len(grams)


print("\nTASK B — open-ended: there is NO correct answer\n")
print(f"{'method':<14} {'distinct':>10} {'3-gram repeat':>15} "
      f"{'mean log p':>12}")
for label, width in [("greedy", 1), ("beam 3", 3), ("beam 10", 10)]:
    if width == 1:
        out, lp = greedy_decode(open_logits, 50)
    else:
        out, lp = beam_search(open_logits, 50, width, VB)
    print(f"{label:<14} {len(set(out)):>10} {rep_rate(out):>15.3f} "
          f"{lp / len(out):>12.4f}")

g = np.random.default_rng(3)
seq = []
for _ in range(50):
    pr = softmax(open_logits(seq))
    seq.append(int(g.choice(VB, p=pr)))
print(f"{'sampling T=1':<14} {len(set(seq)):>10} {rep_rate(seq):>15.3f} "
      f"{'-':>12}")

print("""
On the open-ended task, searching harder buys nothing. Beam search finds
sequences with roughly four times greedy's mean log-probability and they are
JUST AS REPETITIVE — both collapse to two distinct tokens and repeat 96% of
their 3-grams. The extra search effort located a higher point inside the same
degenerate region, because equation (eq:repetition-feedback) means the mode IS
that region.

Sampling finds the LEAST probable text of the three and is the only one that
stays varied: sixteen distinct tokens and 19% repetition. That is the inversion
— on Task A more search meant a better answer, and here it means a
higher-scoring version of the same failure.

The whole rule: beam search when the task has a right answer, sampling when it
does not.""")
```

## 9. Practical Example

A team runs one endpoint for four features: code generation, factual Q&A over
retrieved documents, marketing copy, and JSON extraction. All four use the
provider's defaults. Three of them are misconfigured, and the symptoms have been
attributed to the model.

```python {tier=A name=decoding-by-task}
"""Decoding settings are a task decision. The defaults are a chat compromise."""

TASKS = {
    "code generation": dict(
        determinism="high", diversity="none", failure="subtle wrong logic",
        temperature=0.2, top_p=0.95, top_k=0, repetition_penalty=1.0,
        note="penalties break repeated identifiers; low T for determinism"),
    "factual Q&A (RAG)": dict(
        determinism="high", diversity="none", failure="fabrication",
        temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
        note="greedy: the answer is in the context, do not invent alternatives"),
    "marketing copy": dict(
        determinism="low", diversity="high", failure="bland or repetitive",
        temperature=1.0, top_p=0.95, top_k=0, repetition_penalty=1.05,
        note="sampling at T~1 matches the model's distribution (eq:human-surprise-gap)"),
    "JSON extraction": dict(
        determinism="total", diversity="none", failure="unparseable",
        temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
        note="greedy + constrained decoding (ch:llm-structured-output)"),
}

DEFAULTS = dict(temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0)

print(f"{'task':<20} {'T':>5} {'top_p':>7} {'rep pen':>9} {'matches default':>17}")
for name, cfg in TASKS.items():
    matches = all(cfg[k] == v for k, v in DEFAULTS.items())
    print(f"{name:<20} {cfg['temperature']:>5.1f} {cfg['top_p']:>7.2f} "
          f"{cfg['repetition_penalty']:>9.2f} {str(matches):>17}")

print(f"\n{'task':<20} {'primary failure mode':<24} {'why this config'}")
for name, cfg in TASKS.items():
    print(f"{name:<20} {cfg['failure']:<24} {cfg['note']}")

n_wrong = sum(1 for c in TASKS.values()
              if not all(c[k] == v for k, v in DEFAULTS.items()))
print(f"\n{n_wrong} of {len(TASKS)} tasks need something other than the "
      f"defaults.")

# What the wrong setting costs, per task.
print(f"\n{'task':<20} {'symptom if left at defaults':<46}")
SYMPTOMS = {
    "code generation": "nondeterministic output; occasional invalid syntax",
    "factual Q&A (RAG)": "answers not grounded in the retrieved passage",
    "marketing copy": "fine — this is what the defaults were chosen for",
    "JSON extraction": "intermittent parse failures under load",
}
for name, sym in SYMPTOMS.items():
    print(f"{name:<20} {sym:<46}")

print("""
Every one of these symptoms would normally be reported as a model problem, and
three of the four are decoding problems fixable in a config file.

The pattern to internalise: ask whether the task has a CORRECT answer. If it
does — code, extraction, grounded Q&A — you want determinism, and temperature
above zero is actively harmful because it introduces variation into something
that should not vary. If it does not, you want the model's own distribution,
which is T near 1 with nucleus truncation.

The provider defaults are tuned for open-ended chat, which is exactly one of
these four cases.""")
```

> PRODUCTION TIP: Set decoding parameters per endpoint, explicitly, and record
> them alongside the prompt in your evaluation harness. A prompt evaluated at
> $T=0$ and served at $T=1$ has not been evaluated.

## 10. Production Considerations

**Set parameters per task, not per provider.** `decoding-by-task` shows three of
four typical workloads needing something other than the defaults.

**Use $T = 0$ for anything with a correct answer.** Grounded Q&A, extraction,
code and classification all want determinism. It also makes evaluation
reproducible, which is worth as much as the quality difference.

**Do not use repetition penalties on code.** They operate on tokens and
legitimately repeated identifiers are penalised as heavily as filler.

**Record the sampler order.** Different libraries apply penalties, temperature
and truncation in different orders ({{sec:5-formal-explanation}}), so the same
nominal settings do not transfer across stacks.

**Log the settings with every generation** in any system where outputs are
reviewed later. A reported bad output is uninterpretable without them.

**Do not promise bit-exact reproducibility.** Seeding fixes the sampler;
batching still perturbs logits in the last bits
({{sec:7-internal-mechanics}}), and near a probability boundary that flips a
token.

## 11. Common Mistakes

**Beginners:**

*Raising temperature to fix repetition.* It helps and it is the crude
instrument. Nucleus sampling addresses the cause — a maximiser trapped by
{{eq:repetition-feedback}} — without flattening everything.

*Using defaults for every task.* They are a chat compromise.

*Believing $T=0$ and greedy differ.* They are the same thing
({{eq:decoding-temperature}} as $T\to0$).

**Experienced practitioners:**

*Tuning temperature and top-p independently.* They interact:
`samplers-from-scratch` shows the nucleus growing with temperature, so raising
$T$ at fixed $p$ widens the sampled set twice.

*Using beam search for open-ended generation.* `beam-search` shows wider beams
producing more repetitive output on exactly the tasks people reach for them.

*Evaluating at one temperature and serving at another.*

*Assuming top-p bounds the candidate set.* It bounds mass, not size, and on a
flat distribution the nucleus can be enormous. Pair it with a $k$ if you need a
bound.

## 12. Failure Modes

**Degeneration.** Repetitive loops. *Cause:*
{{eq:repetition-feedback}} plus a maximising decoder. *Fix:* sampling with
nucleus truncation. *Detection:* $n$-gram repetition rate, which is one line and
almost never monitored.

**Tail selection.** An absurd token from the far tail. *Cause:* untruncated
sampling over a large vocabulary. *Fix:* top-p.

**Over-truncation.** Output that is bland and safe. *Cause:* $p$ or $T$ too low.
*Symptom:* the model "refuses to be creative", which is read as alignment and is
usually decoding.

**Nondeterminism where determinism was assumed.** *Cause:* $T > 0$ on a task
with a correct answer. *Symptom:* flaky evaluations and irreproducible bug
reports.

**Stop-sequence miss.** A stop string split across token boundaries.
*Symptom:* generation running past where it should end, intermittently.
*Fix:* buffer and check detokenized text.

**Settings drift.** A provider changes a default. *Detection:* log the settings
you send *and* what the response reports, and alert on divergence.

## 13. Alternatives

{#tbl:decoding-strategies caption="Decoding strategies by objective and cost. The first two maximise likelihood, which equation (eq:human-surprise-gap) shows is the wrong target for open-ended text; the middle three sample; the last two are constrained or assisted variants treated elsewhere."}

| Strategy | Objective | Cost | Right for |
|---|---|---|---|
| Greedy | local argmax | 1x | tasks with a correct answer |
| Beam search | sequence likelihood | $B$x | translation, extraction |
| Pure sampling | model distribution | 1x | nothing, in practice |
| Top-k {{cite:fan2018}} | truncated by rank | 1x | superseded by top-p |
| Top-p {{cite:holtzman2020}} | truncated by mass | 1x | open-ended generation |
| Constrained decoding | valid strings only | 1x | structured output ({{ch:llm-structured-output}}) |
| Speculative decoding | same distribution, faster | <1x | serving ({{part:23}}) |

**What genuinely differs.** The first two search and the middle three sample —
a difference in objective, not in quality. **Constrained decoding is the only
row that changes what is reachable**: it makes invalid outputs impossible rather
than unlikely, which is a guarantee none of the others provide. And speculative
decoding is the odd one out: it targets the *identical* distribution as the
sampler it accelerates, so it is a systems optimisation rather than a decoding
choice.

**Pure sampling's row is worth noting.** It is the only strategy that samples
exactly the model's distribution, which sounds correct and is unusable —
{{cite:holtzman2020}}'s argument is that the model's tail is badly estimated,
so faithfully sampling it is faithfully sampling noise.

## 14. Evaluation

**Is the sampler correct?** Four checks:

1. **Truncation is respected** — sampled tokens never fall outside the nucleus.
   The assertion in `samplers-from-scratch`.
2. **$T \to 0$ reproduces greedy** exactly.
3. **Probabilities sum to 1** after each transformation.
4. **Order is as specified** — verifiable by applying temperature before and
   after truncation and confirming the results differ.

**Is the configuration right for the task?**

1. **Repetition rate** on generated output, which catches degeneration.
2. **Determinism** where required — generate twice and compare.
3. **Task metric under the settings you actually serve**, never under defaults.
4. **Distinct-$n$ or self-BLEU** for diversity where diversity is the goal.

**And attribute correctly.** Before investigating a model, re-run with $T=0$. If
the complaint disappears, it was a decoding complaint —
{{tbl:not-in-the-model}} again, and it resolves a surprising share of reported
model problems.

## 15. Advanced Concepts

**Min-p and typical sampling.** {{maturity:EMERGING}} Alternatives to nucleus
truncation. Min-p keeps tokens above a fraction of the top probability, which is
scale-relative rather than rank- or mass-relative; typical sampling keeps tokens
whose surprise is near the distribution's entropy, targeting
{{eq:human-surprise-gap}} directly rather than incidentally.

**Contrastive decoding.** {{maturity:EMERGING}} Score with a large model minus a
small one, suppressing the degenerate behaviours both share and keeping what the
large model knows and the small one does not.

**Speculative decoding.** {{maturity:ESTABLISHED}} A small model drafts several
tokens; the large model verifies them in one forward pass. Provably preserves the
target distribution, so it is free quality-wise ({{part:23}}).

**Self-consistency.** {{maturity:ESTABLISHED}} Sample $n$ chains of thought and
take the majority answer ({{ch:llm-prompting}}). Uses sampling's diversity as a
feature rather than a cost.

**Watermarking.** {{maturity:EMERGING}} Biasing the sampler on a secret key so
generated text is statistically detectable. It is a decoding-time intervention,
which is why it survives paraphrase poorly and cannot be applied retroactively —
and why it degrades exactly when the distribution is sharp, since a sharp
distribution leaves no room to encode a signal without changing the output.

**Structured sampling for reasoning.** {{maturity:EMERGING}} Sampling several
chains and selecting among them ({{ch:llm-prompting}}'s self-consistency) turns
the diversity this chapter treats as a cost into the mechanism. Note what that
implies about temperature: a self-consistency system wants $T$ high enough that
the chains genuinely differ, which is the opposite of the setting the same task
would use for a single generation. **The optimal temperature depends on how many
samples you intend to take**, which is a coupling that is rarely made explicit
and which {{ch:llm-routing}}'s cost model has to account for.

## 16. Connection to Previous Chapters

**Backwards.** {{ch:llm-anatomy}} produced the logits and
{{eq:scale-is-temperature}} *is* temperature — the two are the same operation
seen from different sides. {{ch:llm-next-token}}'s length bias
{{eq:log-sequence-probability}} is why beam search needs
{{eq:length-normalised-score}}, and its calibration discussion is why the tail
is untrustworthy enough to truncate. {{ch:nlp-subword}} explains repetition
penalties' token-level pathology. {{ch:fm-pretraining}}'s objective is what
{{eq:human-surprise-gap}} shows diverging from quality.

**Forwards.** {{ch:llm-prompt-lifecycle}} places this loop in the serving path.
{{ch:llm-structured-output}} adds a constraint that makes invalid output
unreachable. {{ch:llm-hallucination}} uses temperature as one of its levers.
{{ch:llm-routing}} uses the entropy this chapter manipulates. {{part:23}} makes
the loop fast without changing its distribution.

## 17. Exercises

**Beginner**

1. Compute the softmax of $(3, 1, 0)$ at $T = 1$ and $T = 0.5$.
2. Given $p = (0.5, 0.25, 0.15, 0.1)$, what does top-$p$ at $0.9$ keep?
3. Why does greedy decoding repeat?

**Intermediate**

4. For the logits in {{sec:6-mathematical-foundation}}, compute the nucleus at
   $p = 0.95$ for $T = 0.8$ and $T = 1.2$.
5. Explain why penalties must be applied before temperature.
6. A team reports the model "is not creative". Give three decoding causes.

**Advanced**

7. Derive {{eq:entropy-temperature-derivative}} and interpret the variance
   factor.
8. Prove that a maximising decoder cannot escape the feedback loop of
   {{eq:repetition-feedback}}, and say what property of sampling breaks it.
9. Explain why top-p does not bound the candidate set, and construct a
   distribution where it admits most of the vocabulary.

**Implementation**

10. Add min-p and typical sampling to `samplers-from-scratch` and compare their
    set sizes against top-p across sharp and flat distributions.
11. Implement a repetition penalty and measure its effect on a sequence
    containing a legitimately repeated token.
12. Extend `beam-search` with diverse beam search and show whether it recovers
    any of the open-ended quality gap.
13. Measure the interaction: sweep $T$ and $p$ jointly and plot the resulting
    effective set size and repetition rate.

**Reasoning**

14. A RAG system fabricates details not in its retrieved context. Which decoding
    change would you try first, and why is it likely to help?
15. Explain why "sample from the model's own distribution" is both the most
    principled strategy and a bad idea in practice.

## 18. Interview Questions

**Beginner**

1. What does temperature do?
2. What is the difference between top-k and top-p?
3. Why is greedy decoding bad for open-ended text?

**Intermediate**

4. Explain nucleus sampling and why it adapts.
5. Why does beam search hurt open-ended generation?
6. In what order should sampler transformations be applied?

**Senior**

7. Choose decoding settings for four different features and justify each.
8. A user reports nondeterministic output. Walk through the diagnosis.
9. What is degeneration and what actually fixes it?

**Systems**

10. What would you log per generation to make output reviewable later?
11. Why can't you promise bit-exact reproducibility with a seed?

## 19. Research Questions

**Is there a principled truncation rule?** Top-k, top-p, min-p and typical
sampling are all heuristics with tuned parameters. {{eq:human-surprise-gap}}
suggests targeting the entropy directly. Compare them on a controlled quality
metric with matched effective set size — the matched-set-size control is what
most comparisons omit.

**How much of "model quality" is decoding?** Take a fixed model and sweep
decoding configurations across tasks, measuring the spread in task metrics. If
the spread is comparable to the gap between model generations, a great deal of
model comparison in the literature is confounded by decoding.

**Why is human text at moderate surprise?** {{eq:human-surprise-gap}} is
well-established empirically and its explanation is not. Whether it follows from
communicative efficiency, from the model's tail being mis-estimated, or from
both is testable and unresolved.

**Does constrained decoding cost content quality?**
{{ch:llm-structured-output}} makes invalid output unreachable. The obvious
concern is that masking degrades what remains. Measure it with the control that
is usually missing: compare against an unconstrained baseline *including* its
parse failures, not only its successes.

## 20. Chapter Summary

Decoding converts a distribution over tokens into a choice, and the strategy is
a separate design decision from the model. **Everything a user tunes lives
here.**

**Maximising likelihood is the wrong objective for open-ended text.**
{{cite:holtzman2020}}'s result is that the highest-probability continuation is
repetitive and degenerate, and human text sits at *moderate* surprise
{{eq:human-surprise-gap}}. The mechanism is {{eq:repetition-feedback}}: a
repeated phrase raises its own probability, creating a feedback loop that a
maximising decoder is structurally unable to leave, because leaving requires
choosing a lower-probability token. **Sampling escapes it; a better search does
not.**

**Temperature** {{eq:decoding-temperature}} is a single monotone dial —
{{eq:entropy-temperature-derivative}} shows entropy is non-decreasing in $T$,
at a rate proportional to the logits' variance. **Top-k**
{{eq:top-k}} truncates by rank and **top-p** {{eq:top-p}} by mass, and the
adaptivity is the whole argument: a fixed $k$ admits noise on sharp
distributions and truncates continua on flat ones, while a fixed mass lets the
set size follow the distribution's shape.

**The knobs interact.** Raising temperature widens the nucleus as well as
flattening the distribution, so $T$ and $p$ cannot be tuned independently — and
because different libraries apply penalties, temperature and truncation in
different orders, identical nominal settings do not transfer across stacks.

**Beam search inverts.** On a task with a correct answer, wider beams find it.
On open-ended text, wider beams find higher-probability sequences and those are
*more repetitive* — `beam-search` shows the ordering reversing between the two
tasks. The rule is simply whether the task has a right answer.

Finally, the attribution point that makes this chapter operational: three of the
four typical workloads in `decoding-by-task` need something other than the
provider defaults, and every one of their symptoms — nondeterministic code,
ungrounded answers, intermittent parse failures — would ordinarily be reported
as a model problem. **Before investigating a model, re-run at $T=0$.**

## 21. Further Reading

{{cite:holtzman2020}} is the essential paper and it is unusually readable. §3 is
the diagnosis and §4 introduces nucleus sampling, but the figures are the real
content: the plot of human versus machine per-token surprise is the single most
convincing artefact in the decoding literature and makes
{{eq:human-surprise-gap}} immediate.

{{cite:fan2018}} for top-k, which matters as the predecessor that nucleus
sampling fixes. Reading them in order makes clear that top-p is not a new idea
but a correction of a specific, identifiable flaw.

{{cite:radford2019}} §3 for how decoding choices were reported at the time —
worth reading to see how much of what is now standard was then unremarked.

**Where to go next:** {{ch:llm-inference}} follows what happens to the KV cache
as this loop runs, and why the second token costs so much less than the first.
