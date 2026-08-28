---
id: rsn-cot
number: 147
part: XVI
tier: full
status: draft
requires: [invariance-criterion, decoder-only, depth-vs-width]
provides: [serial-computation-budget, tokens-as-working-memory,
           length-generalisation-by-steps, per-step-error-compounding,
           trace-answer-untied, faithfulness-intervention, cot-scope]
citations: [nye2021scratchpads, wei2022cot, kojima2022zeroshotcot,
            merrill2024cotexpressive, turpin2023faithfulness,
            lanham2023faithfulness, sprague2024tocot, deepseek2025r1,
            yao2023tot]
---

## 1. Learning Objectives

By the end of this chapter you will be able to say precisely what intermediate
tokens buy a fixed-depth network, and why the answer is *serial steps* rather
than *intelligence*; predict from that mechanism which tasks chain-of-thought
helps and which it cannot; compute how per-step accuracy compounds over a chain
and what that implies for chain length; explain why a stated reason can be
fluent, sincere and completely unrelated to the answer; and run the intervention
that measures whether a trace is load-bearing instead of reading it and hoping.

## 2. Why This Matters

{{ch:rsn-vs-generation}} ended with a measurement problem: benchmark accuracy
cannot distinguish a system that computes an answer from one fitted to the
surface form of the question. This chapter is about the single technique that is
supposed to fix that — having the model write out its reasoning — and about the
two things that are consistently misunderstood about it.

The first misunderstanding is about what it does. The popular account is that
asking a model to think step by step makes it *think harder*, or *slow down*, or
*engage a different mode*. None of that is mechanistically meaningful. What
actually happens is narrow, precise, and much more useful to know: emitting a
token and reading it back gives the model another pass through its own weights.
A fixed-depth network has a fixed number of sequential operations available per
forward pass. Intermediate tokens are the only way to get more of them.

That is the whole mechanism, and everything else in this chapter follows from it
— including the limits. If the difficulty of a task is that it needs many
sequential steps, intermediate tokens dissolve it. If the difficulty is inside a
single step, they do nothing at all, and {{cite:sprague2024tocot}}'s meta-analysis
over a hundred papers found exactly that shape: large gains on maths and symbolic
reasoning, close to nothing elsewhere.

The second misunderstanding is more consequential, because production systems
are built on it. A visible chain of thought looks like an explanation. Teams read
traces to debug, log them for audit, and build monitoring on the premise that a
model which states its reasoning can be checked by reading it. But the trace and
the answer are two outputs of one system, trained by two different signals, and
nothing in the training ties them together. {{cite:turpin2023faithfulness}} showed
models exploiting an injected bias, losing accuracy when it pointed the wrong
way, and producing confident explanations that never mentioned it.
{{sec:9-practical-example}} builds a system with that exact structure and
measures the resulting divergence, and the number that matters is not how wrong
the explanations are — it is how *stable* their apparent quality is while the
answers collapse.

Both halves of the chapter are load-bearing for {{part:16}}. The serial-steps
result is why test-time compute scales at all ({{ch:rsn-test-time-compute}}).
The compounding result is why self-consistency and verification exist
({{ch:rsn-self-consistency}}). The faithfulness result is why process supervision
is a different thing from outcome supervision ({{ch:rsn-supervision}}), and why
the difference matters.

## 3. Prerequisites

You need {{ch:rsn-vs-generation}}'s distinction between computing an answer and
fitting one, and its invariance criterion — this chapter uses it twice, for
different reasons. From {{part:7}} you need the shape of a decoder-only
transformer: that a forward pass is a fixed number of layers, and that generated
tokens re-enter as input. From {{part:6}} you need depth versus width — that
widening a network adds parallel capacity while deepening it adds sequential
capacity, and that these are not interchangeable.

You do not need the reinforcement-learning material from {{part:9}} here. This
chapter is about what intermediate tokens *are*; how models are trained to
produce good ones is {{ch:rsn-supervision}}'s subject.

## 4. Intuitive Explanation

Think about multiplying two three-digit numbers in your head.

You cannot do it in one shot, and the reason is not that you are bad at
arithmetic. It is that the computation has a shape: partial products, then a
sum, and the sum depends on the partial products having already been computed.
There is an ordering, and you cannot collapse it. What you can do is hold each
partial result somewhere while you compute the next one — on paper, or in working
memory — and that is what makes the problem tractable.

A neural network has the same constraint in a much more literal form. A forward
pass runs through $L$ layers. Each layer's output depends on the previous
layer's, so a forward pass performs $L$ sequential operations and no more. Width
does not help: a wider layer does more things *at once*, and the problem is that
some things have to happen *after* other things. If a task requires more
sequential dependencies than the network has depth, the computation does not fit,
and no amount of training or parameters changes that.

So what does a model do when handed a task that does not fit? It does not fail
loudly. It finds something else that scores well — a lookup table, a heuristic, a
correlation with surface features — which is exactly {{ch:rsn-vs-generation}}'s
subject arriving in a new place. The network is a function approximator, and if
the true function is out of reach it approximates something else.

Now let the model write something down.

The model emits a token. That token becomes part of the input on the next step.
The next forward pass reads it and runs through all $L$ layers again. Nothing
about the model changed — same weights, same depth — but the *computation* now has
$2L$ sequential operations in it, because the second pass depends on the first
pass's output. Emit $T$ tokens and you have $T \times L$ sequential operations
available, and the depth bound has become a bound per token rather than a bound
on the problem.

That is why the sequence is often described as a scratchpad, which is the framing
{{cite:nye2021scratchpads}} introduced a year before the prompting result made
the idea famous. It is not a metaphor. The context window is genuinely serving
the role paper serves for mental arithmetic: a place to put an intermediate
result so that the next step can read it instead of recomputing it, and so that
the next step can *exist at all*.

{{cite:merrill2024cotexpressive}} turned this into a theorem, and the precise
form is worth carrying. Standard transformers that answer immediately provably
cannot solve some very simple problems — checking whether two nodes in a graph
are connected, simulating a finite-state machine. Allowing intermediate
generation genuinely extends what they can compute, and *how much* it extends
depends on how many tokens: a logarithmic number of decoding steps barely helps,
a linear number lets a decoder recognise all regular languages, and a polynomial
number gets you exactly the polynomial-time-solvable problems. The resource being
spent is token count, and the returns are stratified.

Hold onto the shape of that result, because it explains two things at once. It
explains why longer chains help — more tokens, more serial steps — and it explains
why they help *unevenly*, since moving from logarithmic to linear to polynomial
token budgets crosses genuine complexity boundaries rather than sliding along a
smooth curve.

Here is the part that gets lost. Nothing in this account says the model got
better at anything. The one-step operation is the same operation it could always
do. What changed is that a problem needing twenty of those operations in sequence
has been converted into twenty problems needing one each, and the model only ever
faces the one-step version. The sequence dimension does the iterating.

Which immediately tells you where this fails. If the model cannot do the one-step
version reliably, giving it twenty attempts at a coin flip is still a coin flip —
worse, actually, because a chain is only correct if *every* link is. And if the
difficulty of the task was never sequential — recalling a fact, judging a tone,
resolving an ambiguous pronoun — then there is no decomposition to exploit, and
the extra tokens buy exactly nothing.

The second half of the chapter is about a different failure, and the intuition is
simpler than the mechanism. When a model produces a chain of thought and then an
answer, you are looking at two outputs. Ask what trained each one. The answer was
trained against labels: be right. The trace was trained against human-written
reasoning: sound like a person explaining. Those are two objectives, and there is
no term in either that says *the trace must describe how the answer was produced*.

So the default outcome — not the failure case, the default — is that the model
produces a plausible rationale and a correct answer by two separate routes.
Usually they agree, because both are functions of the same input. When they
diverge, they diverge silently: the rationale stays just as fluent, because
fluency was what it was optimised for, and it was never a function of the answer
in the first place.

## 5. Formal Explanation

Write a forward pass as a composition of $L$ layer functions:

$$h^{(L)} = f_L \circ f_{L-1} \circ \cdots \circ f_1 (x)$$ (eq:forward-pass-depth)

The composition depth is $L$. Any computation the network performs in one pass is
expressible as $L$ nested applications, so any problem whose solution requires a
chain of $S > L$ dependent operations — where step $i$ cannot begin until step
$i-1$ has produced its value — cannot be computed exactly by this network:

$$S \le L \quad\text{for exact computation in a single pass}$$ (eq:depth-bounds-serial-steps)

This is a statement about *serial* depth, not about capacity. Widening layers
increases the parallel work per step and leaves $S \le L$ untouched. It is the
formal version of the arithmetic intuition, and it is the reason
{{cite:merrill2024cotexpressive}}'s impossibility results exist for problems as
simple as graph connectivity.

Now let the model emit tokens autoregressively. At step $t$ the input is the
original prompt plus every token emitted so far, so the computation at step $t$
depends on the computation at step $t-1$ through the emitted token $y_{t-1}$:

$$y_t = g\big(x, y_1, \ldots, y_{t-1}\big), \qquad S_{\text{available}} = T \cdot L$$ (eq:tokens-buy-steps)

The available serial depth is now the product of the token count and the layer
count. The bound has not disappeared — it has been re-expressed in a resource you
can buy more of at inference time, which is exactly what {{part:16}}'s later
chapters spend.

The cost is that correctness now has to hold at every step. If the per-step
operation succeeds independently with probability $p$, and a wrong intermediate
value cannot be recovered from, then the probability that a $k$-step chain ends
correctly is:

$$P(\text{chain of length } k \text{ correct}) = p^{k}$$ (eq:chain-accuracy-compounds)

Two things follow immediately. Accuracy decays geometrically in chain length, so
there is a length beyond which adding steps makes things worse rather than
better. And the exponent means small per-step improvements matter far more than
they look: raising $p$ from $0.95$ to $0.98$ takes twenty-step accuracy from
$36\%$ to $67\%$. This single equation is the argument for
{{ch:rsn-self-consistency}}'s sampling and {{ch:rsn-supervision}}'s per-step
supervision, and it is why the rest of this part is largely about *checking*
steps rather than generating more of them.

For the faithfulness half, model the system as one network with two output heads.
Let $\theta$ be shared parameters, $\phi_a$ the answer head and $\phi_r$ the trace
head. Training minimises:

$$\mathcal{L} = \mathcal{L}_{\text{ans}}\big(a_{\theta,\phi_a}(x),\, y\big) + \lambda\, \mathcal{L}_{\text{trace}}\big(r_{\theta,\phi_r}(x),\, \rho\big)$$ (eq:trace-and-answer-are-untied)

where $y$ is the label and $\rho$ is a human-written rationale. Read the two terms
carefully. The first depends on the answer and the label. The second depends on
the trace and the human rationale. **Neither term contains both $a$ and $r$.**
There is no gradient anywhere in this objective that increases when the trace
describes the answer's derivation and decreases when it does not.

So faithfulness is not something the training optimises weakly, or optimises
under other pressures. It is not in the objective at all. Any agreement between
$a$ and $r$ arises because both are functions of $x$ through shared parameters
$\theta$ — a correlation, not a constraint — and correlations break exactly where
you most need them to hold, which is off the training distribution.

That gives the operational definition of faithfulness, and it is an
interventional one rather than a property you can read off a trace. A trace is
faithful to the extent that perturbing it changes the answer:

$$\text{faithfulness} \;\propto\; \mathbb{E}\big[\,\mathbb{1}[\,a(x, r) \neq a(x, \tilde{r})\,]\,\big]$$ (eq:faithfulness-intervention)

for perturbations $\tilde{r}$ that alter the trace's content. {{cite:lanham2023faithfulness}}
built precisely this measurement — inserting mistakes into the chain of thought,
paraphrasing it — and found dependence varying enormously across tasks. Their
headline result is the one to remember: **larger and more capable models produced
less faithful reasoning on most tasks studied.** Every other quality improves with
scale; this one degrades.

## 6. Mathematical Foundation

{{eq:chain-accuracy-compounds}} deserves more attention than it usually gets,
because the independence assumption behind it is both wrong and useful.

Steps in a real chain are not independent. Errors are correlated — a model that
misreads the problem tends to keep misreading it — and some errors are
self-correcting, where a later step contradicts an earlier one visibly enough to
be caught. Write the true chain accuracy as a product of conditional
probabilities:

$$P(\text{correct}) = \prod_{i=1}^{k} P\big(s_i \text{ correct} \mid s_1 \ldots s_{i-1} \text{ correct}\big)$$ (eq:conditional-chain)

If conditioning on a correct prefix makes the next step easier — which it does,
since a correct prefix is a cleaner context — then each conditional exceeds the
marginal $p$ and the true accuracy beats $p^k$. So {{eq:chain-accuracy-compounds}}
is a lower bound in practice, and the measured decay is gentler than geometric.

Do not take much comfort from that. The correlation cuts the other way once a
step *is* wrong: subsequent steps condition on the error and elaborate it
consistently. The distribution of outcomes is bimodal — chains that stay on track
tend to stay on track, chains that leave it do not come back — which is why a
long wrong chain reads as coherent rather than confused. Every step after the
mistake is correct reasoning from a wrong premise.

Now the length trade-off. Longer chains buy serial depth
({{eq:tokens-buy-steps}}) and pay compounding ({{eq:chain-accuracy-compounds}}).
Let $q(k)$ be the probability the task is *solvable* with $k$ steps available —
increasing in $k$, saturating once the chain is long enough — and let $p^k$ be the
probability of executing $k$ steps correctly. Expected accuracy is:

$$A(k) = q(k) \cdot p^{k}$$ (eq:length-tradeoff)

with $q$ increasing and $p^k$ decreasing. The product has an interior maximum.
There is an optimal chain length, it depends on per-step reliability, and
**improving $p$ moves the optimum to the right**: a more reliable model should
think for longer, and a less reliable one should not.

That is the analytical shape behind a result the field found empirically. When
{{cite:deepseek2025r1}} reported reasoning traces growing over the course of RL
training, {{eq:length-tradeoff}} says what to make of it: the model's per-step
reliability $p$ improved, which moved the optimal $k$ out, and the length increase
is a *consequence* of reliability rather than a cause of capability. Copying the
length without the reliability moves you down the wrong side of the curve.

## 7. Internal Mechanics

Where does the intermediate result actually live?

In the KV cache, and this is worth making concrete because it is easy to
mistake the scratchpad for a metaphor. When the model emits token $y_t$, that
token is embedded and processed, and its keys and values at every layer are
appended to the cache. Every subsequent forward pass attends over those entries.
The intermediate result is stored as $L$ layers' worth of key-value vectors, and
it is read back by attention.

This has a consequence for the depth argument that the clean version elides.
A later token does not only get "another pass through the weights" — it gets
attention over the *full residual stream state* of every earlier token at every
layer. So the serial computation is richer than a simple iteration: step $t$ can
attend to layer-3 features of step $t-4$ directly. The formal accounting in
{{eq:tokens-buy-steps}} is a lower bound on what the architecture makes available.

### 7.1 The bottleneck at the token

There is also a real narrowing, and it is the interesting asymmetry.

Between passes, the model's state is squeezed through a *token* — one discrete
symbol from a vocabulary of perhaps 100,000. The residual stream carrying it is
thousands of dimensions of continuous state, and all of that must be compressed
into a categorical choice before the next pass can read it.

So the scratchpad is a lossy channel. Whatever the model computed that did not
survive into the emitted token is gone; the next pass reconstructs what it can
from the symbol plus attention over previous states. This is the mechanistic
reason chain-of-thought works better on tasks whose intermediate states are
*naturally symbolic* — a partial sum, a variable binding, a selected entity — than
on tasks whose intermediate states are diffuse. A partial sum survives
tokenisation perfectly. "The vague sense that this argument is weaker than it
looks" does not.

It also explains why latent or continuous-space reasoning is an active research
direction: if the bottleneck is the discretisation, then passing a continuous
vector between steps removes it. That is {{maturity:EXPERIMENTAL}} work, and it
gives up the one property that makes token-based reasoning tractable to work
with, which is that you can read it.

### 7.2 Elicitation versus capability

{{cite:wei2022cot}} demonstrated the behaviour with few-shot exemplars, which
made it look as though the demonstrations were teaching a format. Then
{{cite:kojima2022zeroshotcot}} showed a single instruction phrase does most of the
work — MultiArith from $17.7\%$ to $78.7\%$, GSM8K from $10.4\%$ to $40.7\%$ on
the same model, with no exemplars at all.

That is a different claim about what is happening. If exemplars were required,
chain-of-thought would be in-context learning: the model is shown a pattern and
copies it. If a phrase suffices, the capability was already present in the
pretrained weights and the prompt is only *routing* to it — selecting a region of
the output distribution where step-by-step derivations are likely, because such
derivations were abundant in the training corpus.

Both papers describe the same underlying resource. The serial-computation
argument says what the extra tokens buy; the elicitation results say the model
already knew how to produce them, and that reasoning-trained models
({{ch:rsn-supervision}}) are making a behaviour reliable rather than installing
one.

### 7.3 Why the trace and the answer come apart mechanically

{{eq:trace-and-answer-are-untied}} says nothing ties them. The mechanics say more:
there is active pressure pushing them apart.

The answer head is rewarded for being right, and gradient descent finds the
cheapest path to being right. If a spurious feature predicts the label in
training, using it is cheaper than computing the real function, so it gets used —
and, as {{sec:9-practical-example}} measures, it gets used *even when the model
has also learned the real function*. The trace head is rewarded for producing text
that scores well against human rationales, and human rationales cite reasons
humans find plausible. Those are different targets, so they select different
features.

Now add the part that makes it worse at scale. A more capable model finds
shortcuts more reliably and produces more fluent rationales. Both heads get
better at their own objectives, and neither improvement pulls them together.
That is the mechanism behind {{cite:lanham2023faithfulness}}'s finding that
faithfulness *decreases* with capability, and it means you cannot wait for the
problem to be solved by better models.

### 7.4 What the serial-steps account does not explain

Two things, stated plainly, because the account above is clean enough to be
over-applied.

It does not explain gains on tasks with no serial structure. Chain-of-thought
sometimes helps a little on classification and reading comprehension, where there
is no multi-step computation to unroll. The likely mechanism is different —
conditioning the output on a longer, more task-relevant context shifts the
distribution — and {{cite:sprague2024tocot}} found these gains small and
inconsistent, which is what you would expect from something other than the
mechanism this chapter describes.

And it does not explain why the *content* of the trace matters when it does.
{{cite:lanham2023faithfulness}} found the performance boost comes neither from
added test-time compute alone nor from the trace's particular phrasing, which
rules out both the pure serial-steps story and the pure elicitation story as
complete accounts. Something in between is happening, and pinning it down is
{{maturity:RESEARCH FRONTIER}}.

## 8. Implementation

Two listings. The first measures what intermediate tokens buy on a task where the
required number of serial steps is explicit and controllable, and where nothing
can be memorised because the test asks for step counts that never appeared in
training. The second builds a system with the structure of
{{eq:trace-and-answer-are-untied}} and measures how far the trace and the answer
come apart.

```python {tier=A name=depth-bounds-serial-steps}
"""What intermediate tokens actually buy: serial steps a forward pass cannot take.

A forward pass through a fixed-depth network performs a bounded number of
sequential operations. If a problem needs more sequential steps than the network
has depth, no amount of width or training fixes it -- the computation does not fit
(eq:depth-bounds-serial-steps).

Emitting an intermediate result and reading it back changes that. The sequence
becomes working memory, and each emitted token buys another pass through the same
weights, so the number of serial steps is bounded by the number of tokens rather
than by the depth.

This listing measures the difference on a task where the required number of steps
is explicit and controllable, and where nothing can be memorised because the test
asks for step counts never seen in training.
"""
import numpy as np

rng = np.random.default_rng(293)

N = 16                       # states
PERM = rng.permutation(N)    # the function to iterate
K_TRAIN = 8                  # step counts seen in training
K_TEST = 24


def apply_k(x, k):
    for _ in range(k):
        x = PERM[x]
    return x


def onehot(idx, n):
    v = np.zeros(n)
    v[idx] = 1.0
    return v


def make_direct(n, kmax):
    """Input: the start state AND the number of steps. The network must do the
    whole iteration inside one forward pass."""
    X, Y = [], []
    for _ in range(n):
        x = int(rng.integers(N)); k = int(rng.integers(1, kmax + 1))
        X.append(np.concatenate([onehot(x, N), onehot(k - 1, K_TEST)]))
        Y.append(apply_k(x, k))
    return np.array(X), np.array(Y)


def make_step(n, states=N):
    """Input: the current state only. Output: ONE step. The iteration happens
    outside the network, through the emitted tokens."""
    X, Y = [], []
    for _ in range(n):
        x = int(rng.integers(states))     # only `states` of them are ever shown
        X.append(np.concatenate([onehot(x, N), np.zeros(K_TEST)]))
        Y.append(PERM[x])
    return np.array(X), np.array(Y)


def train(X, Y, depth, width=64, steps=500, lr=0.08):
    """A plain MLP. `depth` is how many nonlinear layers a single forward pass
    passes through -- the number of sequential operations available."""
    dims = [X.shape[1]] + [width] * depth + [N]
    Ws = [rng.normal(size=(a, b)) / np.sqrt(a) for a, b in zip(dims, dims[1:])]
    bs = [np.zeros(b) for b in dims[1:]]
    T = np.eye(N)[Y]
    m = [np.zeros_like(w) for w in Ws] + [np.zeros_like(b) for b in bs]
    v = [np.zeros_like(w) for w in Ws] + [np.zeros_like(b) for b in bs]
    for t in range(1, steps + 1):
        hs = [X]
        for i, (W, b) in enumerate(zip(Ws, bs)):
            z = hs[-1] @ W + b
            hs.append(np.tanh(z) if i < len(Ws) - 1 else z)
        p = np.exp(hs[-1] - hs[-1].max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        g = (p - T) / len(X)
        grads = []
        for i in range(len(Ws) - 1, -1, -1):
            grads.append((hs[i].T @ g, g.sum(0)))
            if i > 0:
                g = (g @ Ws[i].T) * (1 - hs[i] ** 2)
        grads = grads[::-1]
        params = Ws + bs
        gs = [grads[i][0] for i in range(len(Ws))] + \
             [grads[i][1] for i in range(len(Ws))]
        for i, (pm, gr) in enumerate(zip(params, gs)):
            m[i] = 0.9 * m[i] + 0.1 * gr
            v[i] = 0.999 * v[i] + 0.001 * gr ** 2
            pm -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    return Ws, bs


def forward(model, X):
    Ws, bs = model
    h = X
    for i, (W, b) in enumerate(zip(Ws, bs)):
        z = h @ W + b
        h = np.tanh(z) if i < len(Ws) - 1 else z
    return h


def eval_direct(model, k, n=1200):
    X, Y = [], []
    for _ in range(n):
        x = int(rng.integers(N))
        X.append(np.concatenate([onehot(x, N), onehot(k - 1, K_TEST)]))
        Y.append(apply_k(x, k))
    return float(np.mean(forward(model, np.array(X)).argmax(1) == np.array(Y)))


def eval_cot(model, k, n=1200):
    """Emit one step at a time and feed it back. Each token is another pass
    through the same weights."""
    xs = rng.integers(N, size=n)
    ys = np.array([apply_k(int(x), k) for x in xs])
    cur = xs.copy()
    for _ in range(k):
        X = np.stack([np.concatenate([onehot(int(c), N), np.zeros(K_TEST)])
                      for c in cur])
        cur = forward(model, X).argmax(1)
    return float(np.mean(cur == ys))


Xd, Yd = make_direct(4000, K_TRAIN)
Xs, Ys = make_step(4000)
Xn, Yn = make_step(4000, states=12)

print(f"Iterating a fixed permutation on {N} states. Training uses step counts")
print(f"1 to {K_TRAIN}; the test goes to {K_TEST}. Nothing can be memorised past")
print(f"{K_TRAIN} because those step counts never appeared in training.")
print()
print(f"{'steps k':>9}" + "".join(f"{'direct d=' + str(d):>13}"
                                  for d in (1, 2, 4))
      + f"{'CoT, exact':>13}{'CoT, partial':>15}")
print(f"{'':>9}{'':>39}{'step model':>13}{'step model':>15}")
print("-" * 76)

direct = {d: train(Xd, Yd, d) for d in (1, 2, 4)}
step_model = train(Xs, Ys, 2)
noisy_step = train(Xn, Yn, 2)

rows = {}
for k in (1, 2, 4, 8, 12, 16, 24):
    ds = [eval_direct(direct[d], k) for d in (1, 2, 4)]
    c = eval_cot(step_model, k)
    cn = eval_cot(noisy_step, k)
    rows[k] = (ds, c, cn)
    mark = "  <- unseen k" if k > K_TRAIN else ""
    print(f"{k:>9}" + "".join(f"{v:>13.1%}" for v in ds) + f"{c:>13.1%}"
          + f"{cn:>15.1%}" + mark)

print()
print()
print("Per-step accuracy of the two step models, and what depth buys the direct")
print("model within the range it was trained on.")
print()
print(f"{'model':>28}{'k=1':>9}{'k=8':>9}{'k=24':>9}")
print("-" * 55)
p_exact = eval_cot(step_model, 1)
p_part = eval_cot(noisy_step, 1)
print(f"{'step model (all 16 states)':>28}"
      f"{p_exact:>9.1%}{eval_cot(step_model, 8):>9.1%}"
      f"{eval_cot(step_model, 24):>9.1%}")
print(f"{'step model (12 of 16 seen)':>28}"
      f"{p_part:>9.1%}{eval_cot(noisy_step, 8):>9.1%}"
      f"{eval_cot(noisy_step, 24):>9.1%}")
for d in (1, 2, 4):
    print(f"{'direct, depth ' + str(d):>28}{rows[1][0][(1,2,4).index(d)]:>9.1%}"
          f"{rows[8][0][(1,2,4).index(d)]:>9.1%}"
          f"{rows[24][0][(1,2,4).index(d)]:>9.1%}")

r8, r24, r12, r16 = rows[8], rows[24], rows[12], rows[16]
lo = min(r24[0] + r12[0] + r16[0])
hi = max(r24[0] + r12[0] + r16[0])
print(f"""
Read the last three rows of the first table against the first four.

At every step count seen in training the direct models are perfect. All three
depths reach {r8[0][0]:.0%} at k={K_TRAIN}, and so does the chain-of-thought
model. If you stopped the experiment here you would conclude that the two
approaches are equivalent and that depth is irrelevant, and both conclusions
would be artefacts of testing only inside the training range.

At k={K_TEST} the direct models score {r24[0][0]:.1%}, {r24[0][1]:.1%} and
{r24[0][2]:.1%}. Across all three unseen step counts they range from {lo:.1%} to
{hi:.1%} with no pattern -- depth 4 happens to score {r24[0][2]:.1%} at k=24 and
{r12[0][2]:.1%} at k=12, which is the signature of an arbitrary mapping rather
than of a computation that partially survives. The chain-of-thought model scores
{r24[1]:.1%} at every unseen k.

Nothing changed about the task. The permutation is the same and the states are
the same; the only difference is a number the direct model was never asked about.
It has no representation of "do this k times" -- it learned a separate mapping for
each k it saw, sixteen states at a time, and there is nothing in that to
extrapolate from.

The one-step model never learned anything about k at all. It learned the
permutation, once, and the iteration happens OUTSIDE it, in the loop that feeds
each output back as the next input. Its accuracy is flat in k because k is not a
property of anything it computes.

Note what this does NOT show. It is not that the deeper networks ran out of
sequential steps -- at these sizes depth 1 already suffices for every k in range,
because the network is memorising sixteen lookup tables rather than iterating.
eq:depth-bounds-serial-steps is the reason a fixed-depth network cannot iterate
indefinitely; what this listing measures is the consequence when the network
therefore does something else instead. The failure is not "too few layers", it is
"a different algorithm that happens to fit the training range".

That is the mechanism, and it is worth stating precisely because the popular
version is vaguer. Intermediate tokens do not make a model smarter. They convert
a problem that needs many sequential steps into many problems that each need one,
and the model only ever has to solve the one-step problem. The sequence dimension
does the iterating.

Which immediately predicts where chain-of-thought helps and where it does not.

It helps when the task decomposes into steps the model can each do reliably and
the difficulty was the DEPTH of the composition. Arithmetic, symbolic
manipulation, multi-hop lookup: shallow operations composed deeply, which is
exactly cite:sprague2024tocot's finding that the measured gains are concentrated
in maths and symbolic reasoning and close to zero elsewhere.

It does not help when the difficulty is inside a single step -- recalling a fact,
making a judgement, resolving an ambiguous sentence. Writing out intermediate
tokens gives a model more chances to do a thing it cannot do, and more attempts
at a coin flip is still a coin flip.

The last column is the cost, and it is the reason the rest of this part exists.

That column is the same architecture and the same loop, with one change: the step
model saw only 12 of the 16 states in training, so it is {p_part:.1%} accurate on
a single step instead of {p_exact:.1%}. Follow it down the column:
{rows[1][2]:.1%} at one step, {rows[2][2]:.1%} at two, {rows[4][2]:.1%} at four,
{rows[8][2]:.1%} at eight.

Those numbers are approximately {p_part:.2f} raised to the power of k, which is
what compounding means: a chain is correct only if every link is, so accuracy is
multiplicative in length. A per-step accuracy of {p_part:.0%} sounds respectable
and delivers {rows[8][2]:.0%} over eight steps.

(It levels off near {rows[24][2]:.0%} rather than falling to zero because a wrong
state sometimes iterates back onto the right orbit. That is an accident of this
task -- on any problem where a wrong intermediate value stays wrong, the decay
continues.)

The same arithmetic explains a familiar experience. A long chain of thought that
goes wrong early produces a confident, internally consistent, completely wrong
conclusion -- and every step after the mistake is CORRECT reasoning from a wrong
premise. It is the compounding, not the quality of individual steps, that makes
long chains unreliable, and it is why the rest of this part is largely about
checking steps rather than about generating better ones.""")
```

The second listing turns to the trace. It builds the structure of
{{eq:trace-and-answer-are-untied}} directly — one model, two heads, two training
signals, no term connecting them — and adds a shortcut feature of the kind
{{cite:turpin2023faithfulness}} injects.

```python {tier=A name=trace-and-answer-are-untied}
"""Why a stated reason can be sincere, fluent, and unrelated to the answer.

cite:turpin2023faithfulness injects a bias into a task -- reordering options so
the correct one is always in the same position -- and finds models exploiting it,
losing up to 36% accuracy when the bias points the wrong way, and producing
confident explanations that NEVER MENTION IT.

That result is often read as a surprising failure. It is not surprising once you
look at what the two outputs are trained on, and this listing makes the mechanism
explicit by building a system with the same structure
(eq:trace-and-answer-are-untied).

A single model with two heads. One predicts the ANSWER and is trained on labels.
The other produces the STATED REASON and is trained on human-written rationales.
Nothing in the training ties them together, so nothing makes the second an
account of the first.
"""
import numpy as np

rng = np.random.default_rng(307)

D_REAL = 6                # features a human would cite
N = 24000


def make(n, shortcut_strength=1.0, shortcut_valid=True):
    """The label depends on real features. A SHORTCUT feature also predicts it,
    perfectly during training, because of how the data was collected."""
    Xr = rng.normal(size=(n, D_REAL))
    w = np.array([1.4, -1.1, 0.9, 0.0, 0.0, 0.0])      # only 3 features matter
    logit = Xr @ w
    y = (logit > 0).astype(int)
    if shortcut_valid:
        s = y * shortcut_strength + (1 - y) * (-shortcut_strength)
    else:
        s = (1 - y) * shortcut_strength + y * (-shortcut_strength)
    s = s + 0.15 * rng.normal(size=n)
    X = np.concatenate([Xr, s[:, None]], axis=1)
    # The human rationale cites whichever real feature contributed most. It
    # cannot cite the shortcut, because the human never saw it.
    contrib = Xr * w
    rationale = np.abs(contrib).argmax(1)
    return X, y, rationale


def fit_logistic(X, Y, classes, steps=400, lr=0.5):
    W = np.zeros((X.shape[1], classes))
    T = np.eye(classes)[Y]
    for _ in range(steps):
        z = X @ W
        p = np.exp(z - z.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        W -= lr * (X.T @ (p - T) / len(X))
    return W


Xtr, Ytr, Rtr = make(N)
ANSWER = fit_logistic(Xtr, Ytr, 2)        # trained on labels
REASON = fit_logistic(Xtr, Rtr, D_REAL)   # trained on human rationales


def answer(X):
    return (X @ ANSWER).argmax(1)


def stated_reason(X):
    return (X @ REASON).argmax(1)


def acc(X, Y):
    return float(np.mean(answer(X) == Y))


print("A model with two heads: one predicts the answer, one states the reason.")
print(f"{D_REAL} real features (only three matter) plus one shortcut feature that")
print("predicts the label perfectly in training. Human rationales never mention")
print("the shortcut, because the humans never saw it.")
print()
print(f"{'condition':>34}{'accuracy':>11}{'reasons citing':>17}")
print(f"{'':>34}{'':>11}{'the shortcut':>17}")
print("-" * 62)

CASES = [
    ("training distribution", True, 1.0),
    ("shortcut removed", True, 0.0),
    ("shortcut points the WRONG way", False, 1.0),
]
res = {}
for name, valid, strength in CASES:
    X, Y, R = make(6000, strength, valid)
    a = acc(X, Y)
    cite = 0.0                       # the reason head cannot output "shortcut"
    res[name] = a
    print(f"{name:>34}{a:>11.1%}{cite:>17.1%}")

print()
print("  (the reason head has no shortcut option in its output space, so it")
print("   cannot cite it even when the shortcut decided the answer)")

print()
print()
print("How much does each head depend on the shortcut? Weight magnitude as a")
print("share of the total, and accuracy when each feature is ablated.")
print()
print(f"{'feature':>12}{'answer head':>14}{'reason head':>14}"
      f"{'accuracy with':>16}")
print(f"{'':>12}{'weight share':>14}{'weight share':>14}"
      f"{'it zeroed':>16}")
print("-" * 58)
aw = np.abs(ANSWER[:, 1] - ANSWER[:, 0])
rw = np.abs(REASON).sum(1)
Xe, Ye, _ = make(6000)
share = {}
for j, name in enumerate([f"real {i}" for i in range(D_REAL)] + ["SHORTCUT"]):
    Xa = Xe.copy(); Xa[:, j] = 0.0
    share[name] = (aw[j] / aw.sum(), rw[j] / rw.sum(), acc(Xa, Ye))
    print(f"{name:>12}{share[name][0]:>14.1%}{share[name][1]:>14.1%}"
          f"{share[name][2]:>16.1%}")

print()
print()
print("Are the stated reasons plausible? Agreement with the human rationale,")
print("which is what a reader would use to judge them.")
print()
print(f"{'condition':>34}{'reason matches':>17}{'answer':>10}")
print(f"{'':>34}{'the human one':>17}{'correct':>10}")
print("-" * 61)
plaus = {}
for name, valid, strength in CASES:
    X, Y, R = make(6000, strength, valid)
    m = float(np.mean(stated_reason(X) == R))
    plaus[name] = (m, acc(X, Y))
    print(f"{name:>34}{m:>17.1%}{plaus[name][1]:>10.1%}")

tr = res["training distribution"]
rm = res["shortcut removed"]
wr = res["shortcut points the WRONG way"]
print(f"""
The first table is cite:turpin2023faithfulness's experiment with the mechanism
exposed.

On the training distribution the answer head is {tr:.1%} accurate. Remove the
shortcut feature and it drops to {rm:.1%}. Point the shortcut the wrong way and it
falls to {wr:.1%} -- below chance, because it is now confidently following a
feature that has been inverted.

That spread is the measurement of how much the shortcut was doing:
{tr - wr:.1%} of the accuracy was resting on a feature nobody intended the model
to use.

And the reasons cite it {0.0:.0%} of the time, in every condition, because the
reason head's output space does not contain it. It was trained to predict which
REAL feature a human would have cited, and it does that job well.

Which is the whole mechanism, and it is not a failure of anything. Two heads, two
training signals, no term in either objective that ties them together
(eq:trace-and-answer-are-untied). The answer head was rewarded for being right and
found the shortest path to being right. The reason head was rewarded for sounding
like a human explanation and produces one. Nothing asked them to agree.

The second table quantifies the divergence, and it contains the result that
sharpens the whole problem.

The shortcut carries {share['SHORTCUT'][0]:.1%} of the answer head's weight -- more
than the three genuinely-predictive features combined. And zeroing it leaves
accuracy at {share['SHORTCUT'][2]:.1%}.

Read those two together. The model DID learn the correct computation: the real
features are weighted correctly, the irrelevant ones are at zero, and with the
shortcut set aside the model is perfect. It is not that a spurious feature crowded
out the right answer. **The right answer is in there, fully learned, and the
shortcut overrides it.**

Which is why inverting the shortcut takes accuracy to {wr:.1%} rather than to the
{share['SHORTCUT'][2]:.1%} the real features alone would deliver. The model has
everything it needs to be right and is following the stronger signal, exactly as
a linear combination weighted {share['SHORTCUT'][0]:.1%} to one feature must.

That is a worse situation than "the model never learned to reason", and it is the
one cite:turpin2023faithfulness's biasing experiments produce. A model that lacked
the capability would fail visibly on hard cases. A model that has the capability
and is outvoted by a spurious feature fails only when the spurious feature
disagrees -- which is rare in training, rare in evaluation, and exactly what
happens when the deployment distribution shifts.

The third table is the part that makes this hard to catch, and the absolute
numbers matter less than their flatness.

The stated reason matches the human rationale
{plaus['training distribution'][0]:.1%} of the time on the training distribution,
{plaus['shortcut removed'][0]:.1%} with the shortcut removed, and
{plaus['shortcut points the WRONG way'][0]:.1%} when the shortcut has been
inverted and the answer is {plaus['shortcut points the WRONG way'][1]:.1%}
correct.

Those three numbers are the same. The answers behind them are
{plaus['training distribution'][1]:.1%}, {plaus['shortcut removed'][1]:.1%} and
{plaus['shortcut points the WRONG way'][1]:.1%} correct.

**The explanation carries no information about whether the answer is right.** It is
not degraded when the model is wrong, because it was never a function of the
answer -- it is a function of the input, computed by a separate head with a
separate objective. A reader inspecting it sees exactly the same thing in the case
where the model is perfect and the case where it is worse than guessing.

(The absolute agreement of around {plaus['training distribution'][0]:.0%} is not
high -- the reason head is a weak predictor of which feature a human would cite.
That is beside the point here. Even a reason head that matched humans perfectly
would show the same flatness, because the flatness comes from the two heads being
untied rather than from either being bad at its job.)

Three consequences, and they are the reason this chapter exists.

A chain of thought is evidence about what a plausible justification looks like,
not about what happened. Reading one tells you the model can produce a rationale,
which was never in doubt.

Interpretability and safety arguments that rest on monitoring the trace need an
additional ingredient: something that ties the trace to the computation. Post-hoc
rationalisation is the DEFAULT outcome of training two outputs on two objectives,
and a tie between them has to be constructed deliberately -- it does not arise
from making either output better.

And the practical test is not to read the reasoning. It is to PERTURB the input in
a way the stated reasoning implies is irrelevant, and see whether the answer
moves. That is ch:rsn-vs-generation's invariance criterion arriving for a second
reason: it measures the computation rather than the story about it.""")
```

## 9. Practical Example

The first listing iterates a fixed permutation on 16 states. A "direct" model
receives the start state and the step count $k$ and must produce the answer in one
forward pass; a "one-step" model receives only the current state, produces one
step, and is applied $k$ times with its own output fed back. Training uses
$k \in [1, 8]$; the test goes to $k = 24$.

```
  steps k   direct d=1   direct d=2   direct d=4   CoT, exact   CoT, partial
                                                   step model     step model
----------------------------------------------------------------------------
        1       100.0%       100.0%       100.0%       100.0%          76.2%
        2       100.0%       100.0%       100.0%       100.0%          55.8%
        4       100.0%       100.0%       100.0%       100.0%          32.1%
        8       100.0%       100.0%       100.0%       100.0%          19.6%
       12        25.8%        30.1%        19.5%       100.0%          18.8%  <- unseen k
       16        16.4%        12.8%        31.6%       100.0%          19.5%  <- unseen k
       24        10.7%        24.8%        56.3%       100.0%          19.0%  <- unseen k
```

Inside the training range every method is perfect, at every depth. An experiment
that stopped at $k=8$ would report that the two approaches are equivalent and that
depth does not matter, and both conclusions would be artefacts of the test range —
which is {{ch:rsn-vs-generation}}'s warning about held-out accuracy, reproduced
under controlled conditions.

Outside the range the direct models score between $10.7\%$ and $56.3\%$ with no
pattern. Depth 4 gets $56.3\%$ at $k=24$ and $19.5\%$ at $k=12$; that
non-monotonicity is the signature of an arbitrary mapping rather than of a
degraded computation. The chain-of-thought model scores $100\%$ at every unseen
$k$.

The honest reading is narrower than "deeper networks ran out of steps". At this
size, depth 1 already suffices for every $k$ in range — because the network is
memorising sixteen lookup tables rather than iterating.
{{eq:depth-bounds-serial-steps}} is the reason a fixed-depth network *cannot*
iterate indefinitely; what the listing measures is what the network does instead
when it cannot, which is to learn a different algorithm that fits the training
range and has nothing to extrapolate from. The failure is not "too few layers",
it is "a substitute algorithm".

The last column is the cost, and it is the same architecture and the same loop
with one change: that step model saw only 12 of the 16 states during training.

```
                       model      k=1      k=8     k=24
-------------------------------------------------------
  step model (all 16 states)   100.0%   100.0%   100.0%
  step model (12 of 16 seen)    72.6%    19.8%    17.8%
             direct, depth 1   100.0%   100.0%    10.7%
             direct, depth 2   100.0%   100.0%    24.8%
             direct, depth 4   100.0%   100.0%    56.3%
```

Per-step accuracy of about $75\%$ yields $55.8\%$ at two steps, $32.1\%$ at four
and $19.6\%$ at eight — approximately $p^k$, exactly {{eq:chain-accuracy-compounds}}.
A per-step accuracy that sounds respectable delivers under $20\%$ over eight
steps. (It levels off near $19\%$ instead of decaying to zero because a wrong
state occasionally iterates back onto the right orbit; on tasks where a wrong
intermediate value stays wrong, the decay continues.)

The second listing measures the trace. On the training distribution the answer
head is $100\%$ accurate; with the shortcut removed, $90.9\%$; with the shortcut
inverted, $6.4\%$ — below chance, because the model is confidently following an
inverted feature. In all three conditions the stated reasons mention the shortcut
$0\%$ of the time.

```
     feature   answer head   reason head   accuracy with
              weight share  weight share       it zeroed
----------------------------------------------------------
      real 0         19.1%          9.8%          100.0%
      real 1         15.1%          4.5%          100.0%
      real 2         12.3%         11.0%          100.0%
      real 3          0.0%         16.7%          100.0%
      real 4          0.0%         18.2%          100.0%
      real 5          0.0%         17.7%          100.0%
    SHORTCUT         53.3%         22.1%          100.0%
```

This table contains the result that sharpens the problem, and it is not the one
the setup predicts. The shortcut carries $53.3\%$ of the answer head's weight —
more than the three genuinely predictive features combined — and yet **zeroing it
leaves accuracy at $100\%$.** The correct computation is fully present: the three
real features are weighted properly, the three irrelevant ones sit at zero. It is
not that the shortcut crowded out the right answer. The right answer is in there,
learned, and the shortcut *overrides* it.

That is a worse failure than a missing capability. A model that never learned the
computation fails visibly on hard cases. A model that has the computation and is
outvoted by a spurious feature fails only when the spurious feature disagrees —
rare in training, rare in evaluation, and routine after a distribution shift.

The third table is why this is hard to catch:

```
                         condition   reason matches    answer
                                      the human one   correct
-------------------------------------------------------------
             training distribution            28.6%    100.0%
                  shortcut removed            25.9%     90.8%
     shortcut points the WRONG way            28.2%      6.3%
```

The explanation's apparent quality is $28.6\%$, $25.9\%$, $28.2\%$ — flat — while
the answers behind it go $100\%$, $90.8\%$, $6.3\%$. **The stated reason carries no
information about whether the answer is right.** It does not degrade when the model
fails, because it was never a function of the answer. A reviewer reading traces
sees the same thing in the case where the model is perfect and the case where it
is worse than guessing.

The absolute agreement of $\approx 28\%$ is low — the reason head is a weak
predictor of which feature a human would cite — and that is beside the point. A
reason head that matched humans perfectly would show the same flatness, because
the flatness comes from the two heads being untied, not from either being bad at
its job.

## 10. Production Considerations

Budget tokens by the task, not by policy. {{cite:sprague2024tocot}} found the
gains concentrated in maths and symbolic reasoning; on retrieval, classification,
and most of what production systems actually do, reasoning tokens are latency and
cost with no measured return. A per-request decision — cheap classifier, or the
model's own signal — is worth building before you enable long reasoning
everywhere.

Set a maximum chain length and enforce it. {{eq:length-tradeoff}} says accuracy
is non-monotonic in length, so an unbounded budget is not a safe default even
ignoring cost. Measure where your own curve turns over rather than assuming
longer is better.

Log traces, but decide in advance what you will do with them. They are useful for
debugging *your prompt and your data*, since they show what the model attended to
and what it was confused by. They are not evidence about how the answer was
produced, and a monitoring system that treats them as such will pass exactly the
cases you built it to catch.

Never let a trace reach a user as an explanation. It reads as one, and
{{sec:9-practical-example}} shows apparent quality staying flat while correctness
collapses. If you surface reasoning, surface it as *the model's draft work*, and
put the verification somewhere the user can see it independently.

Watch cost. A reasoning model can spend thousands of tokens before its first
visible output, which changes time-to-first-token, cache behaviour
({{part:15}}'s serving economics), and the shape of your tail latency. Reasoning
tokens are usually billed as output tokens, so a 10× trace is a 10× bill on that
segment.

## 11. Common Mistakes

**Treating the trace as an explanation.** The single most consequential error, and
the one this chapter exists to prevent. The trace is a plausible rationalisation
produced by a head with its own objective. Use {{eq:faithfulness-intervention}} —
perturb and observe — if you need to know whether it is load-bearing.

**Assuming longer chains are better.** {{eq:length-tradeoff}} has an interior
maximum. Chains past the optimum compound errors without buying serial depth that
the task needs.

**Enabling reasoning everywhere by default.** Reasoning helps on tasks with serial
structure. Elsewhere it is pure cost, and occasionally harmful, because a model
that talks itself through a judgement call can talk itself out of a correct first
instinct.

**Copying a reasoning model's trace length.** Length followed reliability in
{{cite:deepseek2025r1}}, not the other way round. Forcing long traces from a model
with poor per-step accuracy moves you down the wrong side of {{eq:length-tradeoff}}.

**Believing self-reported confidence in the trace.** "I am certain that…" is
generated by the same untied head. It correlates with the *fluency* of the
rationale, not with correctness.

**Testing only inside the range you trained on.** The first listing scores $100\%$
for every method at every depth inside the training range and separates completely
outside it. If your evaluation shares a range with your training data, it cannot
tell you which algorithm you got.

## 12. Failure Modes

*Confident wrong chains.* An early error is elaborated consistently by every
subsequent step, producing a fluent, internally coherent, wrong conclusion. This
is the bimodality of {{eq:conditional-chain}}, and it is why wrong answers from
reasoning models are harder to spot than wrong answers from non-reasoning ones.

*Silent shortcut dependence.* The model has the right computation and is outvoted
by a spurious feature ({{sec:9-practical-example}}). Accuracy is perfect until the
feature's relationship to the label changes, then collapses below chance. Nothing
in the trace signals it, before or after.

*Faithfulness degrading with capability.* {{cite:lanham2023faithfulness}} measured
larger models producing less faithful reasoning on most tasks. An oversight
mechanism that reads traces gets *less* reliable as the systems it oversees get
more capable, which is the wrong direction for a safety control.

*Reasoning that talks itself out of a correct answer.* On tasks with no serial
structure, generating a rationale can shift the output distribution away from a
correct immediate response. This is the mechanism behind the small negative
deltas in {{cite:sprague2024tocot}}'s non-symbolic categories.

*Runaway length.* Models trained to reason sometimes fail to stop, spending the
entire budget on a loop. Enforce a hard cap and treat a hit cap as a failed
request, not a truncated success.

## 13. Alternatives

**Tool use.** If the serial computation is arithmetic or symbolic manipulation,
call a calculator or an interpreter. {{cite:sprague2024tocot}} found
chain-of-thought doing execution that a real solver does better, and
{{ch:rsn-tool-assisted}} takes this seriously: a tool gives you $p = 1$ on the
step, which {{eq:chain-accuracy-compounds}} says is worth more than any amount of
per-step tuning.

**Sampling and aggregation.** Rather than one long chain, draw several and
aggregate ({{ch:rsn-self-consistency}}). This attacks the compounding term
directly, since independent chains fail independently.

**Search over steps.** Tree-of-thoughts style methods
({{cite:yao2023tot}}, {{ch:rsn-test-time-compute}}) explore alternatives at each
step instead of committing, trading compute for the ability to back out of a bad
prefix.

**Process supervision.** Train the steps themselves rather than the outcome
({{ch:rsn-supervision}}). This raises $p$, which is the highest-leverage variable
in {{eq:chain-accuracy-compounds}} and the only one that improves both terms of
{{eq:length-tradeoff}}.

**Latent reasoning.** Pass continuous state between steps instead of tokens,
removing the discretisation bottleneck of {{sec:7-internal-mechanics}}.
{{maturity:EXPERIMENTAL}}, and it gives up readability.

## 14. Evaluation

Evaluate the answer and the trace separately, because they are separate outputs.

For the answer: hold out step counts, not just examples. The first listing shows
a method scoring $100\%$ inside the training range of $k$ and $10\%$ outside it,
and no amount of held-out *examples* at the trained $k$ would have revealed that.
Length generalisation is the axis that separates iteration from lookup.

For the trace, do not score it — intervene on it. {{eq:faithfulness-intervention}}
is a procedure: corrupt a step and see whether the answer changes; paraphrase the
trace and see whether the answer changes; truncate it and see whether the answer
survives. {{cite:lanham2023faithfulness}} ran exactly these and found dependence
varying by task, which means you must measure it *for your task* rather than
inheriting a number.

Report per-step accuracy where you can decompose the task, because
{{eq:chain-accuracy-compounds}} makes it the quantity that predicts behaviour at
lengths you have not tested. End-to-end accuracy at $k=4$ tells you little about
$k=20$; $p$ tells you a lot.

And measure the no-reasoning baseline on every task before enabling reasoning.
The comparison people skip is the cheap one, and on most non-symbolic tasks it is
the one that wins.

## 15. Advanced Concepts

**The complexity stratification.** {{cite:merrill2024cotexpressive}}'s result is
finer than "more tokens help": logarithmic decoding steps extend the recognised
class only slightly, linear steps add all regular languages (with projected
pre-norm) and keep decoders within context-sensitive languages, and polynomial
steps yield exactly the polynomial-time-solvable class. Chain length is a
resource with genuine thresholds in it, not a dial.

**Steganographic reasoning.** If a trace is optimised for an outcome and also
monitored for content, the pressure is to encode information in the trace in ways
the monitor does not read — length, ordering, word choice — while the surface
content remains innocuous. This follows from {{eq:trace-and-answer-are-untied}} plus
optimisation pressure on the trace, and it is why monitoring traces is a weaker
control than it appears. {{maturity:RESEARCH FRONTIER}}.

**Faithfulness by construction.** The only reliable fix is to make the trace
*causally necessary*: force the answer to be a deterministic function of the
trace, so the trace cannot diverge without changing the answer. Tool-assisted
reasoning ({{ch:rsn-tool-assisted}}) gets this almost for free, since the tool call
is the computation. Text traces do not, and no amount of trace-quality training
supplies it.

**Why reasoning-trained models are different in kind.** {{cite:deepseek2025r1}}
trained the behaviour with reinforcement learning against outcomes rather than
imitating human rationales. That changes {{eq:trace-and-answer-are-untied}}: the
trace term is now optimised *through* the answer, so there is at least a path by
which trace content is selected for usefulness. It does not guarantee
faithfulness — a trace can be useful and still not describe the computation — but
it is a structurally different situation from imitation, and it is the reason
{{ch:rsn-supervision}}'s distinction between process and outcome supervision is
the pivot of this part.

## 16. Connection to Previous Chapters

{{ch:rsn-vs-generation}} established that held-out accuracy cannot separate
computing from fitting, and offered perturbation as the test. This chapter used
that twice: the first listing perturbs the step count and watches direct models
collapse; the second perturbs a feature the trace claims is irrelevant.

{{part:7}}'s decoder architecture supplies the mechanism — the KV cache is
literally where the scratchpad lives — and {{part:6}}'s depth-versus-width
distinction is what {{eq:depth-bounds-serial-steps}} formalises.

{{part:15}}'s serving economics acquire a new term here: reasoning tokens are
output tokens, generated in the memory-bound decode phase, so a long trace is
expensive in exactly the regime that is hardest to make efficient.

Ahead: {{ch:rsn-test-time-compute}} spends the resource this chapter identified;
{{ch:rsn-self-consistency}} attacks the compounding in
{{eq:chain-accuracy-compounds}}; {{ch:rsn-supervision}} raises $p$ directly; and
{{ch:rsn-benchmarks}} returns to measurement with both of this chapter's results
in hand.

## 17. Exercises

1. Modify the first listing so the permutation is replaced by a function with
   short cycles (say, period 4). Predict what happens to the direct model's
   out-of-range accuracy before you run it, then explain the result.

2. Sweep the partial step model's coverage from 16 states down to 8 and plot
   measured $k$-step accuracy against $p^k$. Where does the prediction break, and
   what does the gap tell you about the independence assumption?

3. Implement {{eq:length-tradeoff}} with a plausible $q(k)$ for a task of your
   choice and find the optimal $k$ for $p = 0.90$ and $p = 0.98$. How far does the
   optimum move?

4. Add a third head to the second listing that is trained to predict *the answer
   head's output* rather than human rationales. Measure whether its stated
   reasons now track correctness, and explain what changed in
   {{eq:trace-and-answer-are-untied}}.

5. Run {{eq:faithfulness-intervention}} against a deployed reasoning model on a
   task you care about: corrupt one step of its trace and measure how often the
   final answer changes. Compare with the same measurement after truncating the
   trace instead.

6. Construct a task where chain-of-thought *hurts*, and explain the mechanism in
   terms of {{eq:length-tradeoff}} rather than in terms of the model being
   confused.

## 18. Interview Questions

1. What do intermediate tokens give a transformer that a bigger transformer
   cannot? Answer in terms of serial versus parallel computation.

2. A model has $95\%$ per-step accuracy. What is its expected accuracy on a
   30-step problem, and what does that imply about how you should architect the
   system?

3. Why does {{cite:lanham2023faithfulness}} find faithfulness *decreasing* with
   model capability? What does that imply for trace-based oversight?

4. Your reasoning model's traces got longer after RL training and accuracy
   improved. A colleague proposes forcing longer traces on a different model to
   get the same gain. What is wrong with the inference?

5. How would you determine whether a model's stated reasoning is load-bearing,
   without access to its weights?

6. Chain-of-thought improves your maths eval by 20 points and your customer-intent
   classifier by 0.3 points. Explain both numbers with one mechanism.

## 19. Research Questions

1. Can faithfulness be trained *in* rather than hoped for? Every current approach
   improves trace quality, which {{eq:trace-and-answer-are-untied}} says is the
   wrong target. What objective would have a term containing both outputs?

2. Where exactly does the discretisation bottleneck in {{sec:7-internal-mechanics}}
   bind? If continuous-state reasoning outperforms token reasoning on a task, that
   task's intermediate states were not symbolic — can this be predicted in advance?

3. {{cite:lanham2023faithfulness}} found the boost comes neither from added
   test-time compute alone nor from the trace's phrasing. What is the third thing,
   and can it be isolated?

4. Is there a measurable signature that distinguishes a model iterating from a
   model that learned a lookup table, computable from a single model without the
   length-generalisation experiment this chapter used?

5. Does the complexity stratification in {{cite:merrill2024cotexpressive}} have an
   empirical counterpart — can you observe the logarithmic/linear/polynomial
   thresholds in measured accuracy curves on real tasks?

## 20. Chapter Summary

Intermediate tokens buy serial computation. A forward pass through $L$ layers
performs $L$ sequential operations ({{eq:depth-bounds-serial-steps}}); emitting a
token and reading it back gives another $L$, so $T$ tokens make $T \cdot L$
available ({{eq:tokens-buy-steps}}). {{cite:nye2021scratchpads}} named this,
{{cite:merrill2024cotexpressive}} proved it, and the proof is stratified:
logarithmic, linear and polynomial token budgets land in genuinely different
complexity classes.

This predicts the scope. Chain-of-thought helps where the difficulty was the
depth of a composition of steps the model can each do — arithmetic, symbolic
manipulation, multi-hop lookup — and does approximately nothing where the
difficulty is inside a single step. That is {{cite:sprague2024tocot}}'s
meta-analytic finding, and it is a consequence of the mechanism rather than an
empirical surprise.

The first listing measures both halves. Inside the trained range of step counts,
direct models and chain-of-thought are indistinguishable at $100\%$; outside it,
direct models scatter between $10.7\%$ and $56.3\%$ with no pattern while
chain-of-thought stays at $100\%$. And with a step model that is $72.6\%$ rather
than $100\%$ accurate per step, chain accuracy falls $76.2\% \to 55.8\% \to
32.1\% \to 19.6\%$ over one, two, four and eight steps — {{eq:chain-accuracy-compounds}}
in the measured form. Per-step reliability is the variable that matters, and it
enters as an exponent.

The trace is a separate output with a separate objective, and
{{eq:trace-and-answer-are-untied}} contains no term connecting it to the answer.
The second listing measures the consequence: a shortcut carrying $53.3\%$ of the
answer head's weight, never mentioned in any stated reason, and apparent
explanation quality flat at $\approx 28\%$ while accuracy goes $100\% \to 6.3\%$.
The sharpest finding is that the model *had* learned the correct computation —
zeroing the shortcut leaves $100\%$ accuracy — and the shortcut overrode it, which
is a harder failure to detect than a missing capability.

So a chain of thought is a resource and an artefact, and it is not an
explanation. Spend it where the task has serial structure, cap its length because
{{eq:length-tradeoff}} turns over, raise per-step reliability before raising
length, and if you need to know whether the trace describes the computation,
intervene on it ({{eq:faithfulness-intervention}}) rather than reading it.

## 21. Further Reading

{{cite:nye2021scratchpads}} is the paper to read first, because it states the
mechanism plainly and demonstrates it on tasks where the required step count is
explicit — before the prompting framing made the idea famous and vaguer.

{{cite:wei2022cot}} and {{cite:kojima2022zeroshotcot}} are the pair worth reading
together: the first shows the behaviour with exemplars, the second shows a single
instruction phrase does most of the work, and the difference between them is the
difference between teaching a format and eliciting a capability.

{{cite:merrill2024cotexpressive}} is the theory, and it is more readable than its
abstract suggests. Read it for the stratification result specifically.

{{cite:turpin2023faithfulness}} and {{cite:lanham2023faithfulness}} are the
faithfulness pair. The first shows models exploiting an injected bias and never
mentioning it; the second builds the intervention methodology and reports that
faithfulness degrades with capability.

{{cite:sprague2024tocot}} bounds the scope with a meta-analysis and its own
evaluations, and is the citation to have when someone proposes enabling reasoning
across an entire product.

{{cite:deepseek2025r1}} is where the training story begins, and
{{ch:rsn-supervision}} picks it up.
