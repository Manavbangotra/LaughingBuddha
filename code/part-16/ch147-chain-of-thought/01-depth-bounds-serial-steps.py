# -*- coding: utf-8 -*-
# Extracted from: Chapter 147 — Chain-of-Thought and Its Mechanics
# Source: src/.../ch147-chain-of-thought.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

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
