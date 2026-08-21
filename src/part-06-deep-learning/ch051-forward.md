---
id: dl-forward
number: 51
part: VI
tier: full
status: reviewed
requires: [dl-neural-networks, dl-activations, py-numpy, math-matrices]
provides: [computational-graph, forward-pass, epoch, mini-batch, tensor-shapes,
           broadcasting-dl, static-vs-dynamic-graph]
citations: [rumelhart1986, pedregosa2011]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Represent a network as a computational graph and explain why that
   representation is the one that matters.
2. Write a forward pass in batched matrix form and get every shape right.
3. Explain what must be stored during the forward pass and why.
4. Distinguish static from dynamic graph construction and state the trade-off.
5. Count the FLOPs and the memory of a forward pass.
6. Explain why batching is what makes networks fast, and what limits the batch
   size.
7. Debug a shape error by reasoning about the graph rather than by guessing.

## 2. Why This Matters

**The computational graph is the data structure that makes automatic
differentiation possible.** {{ch:dl-backprop}} derives backpropagation as a
traversal of this graph in reverse. Before that traversal can be described, the
graph has to exist as an object rather than as a mental picture, and building
it is the whole content of this chapter.

**Almost all deep learning bugs are shape bugs.** Not conceptual errors —
transposed matrices, a batch dimension in the wrong place, a broadcast that
silently did something other than intended. A discipline for reasoning about
shapes is worth more per hour than almost any other skill in this material, and
{{sec:11-common-mistakes}} catalogues the ones that fail *silently*.

**Batching is why any of this is fast.** A network processes examples one at a
time conceptually and in batches physically, and the difference is two orders
of magnitude of throughput. Understanding why — arithmetic intensity, not
parallelism per se — is what lets you predict whether a change will help.

## 3. Prerequisites

{{ch:dl-neural-networks}} for the layer and {{eq:mlp-forward}}.
{{ch:dl-activations}} for the elementwise functions the graph contains.
{{ch:py-numpy}} for arrays, strides and broadcasting — the mechanics are
identical here. {{ch:math-matrices}} for shapes and matrix products.

## 4. Intuitive Explanation

### 4.1 A network is a graph of operations

Forget layers for a moment. What a framework actually sees is a directed acyclic
graph whose nodes are tensors and whose edges are operations:

```text
    x ──▶[matmul W₁]──▶ z₁ ──▶[+ b₁]──▶ a₁ ──▶[relu]──▶ h₁ ──┐
                                                              │
                        ┌─────────────────────────────────────┘
                        ▼
                   [matmul W₂]──▶ z₂ ──▶[+ b₂]──▶ ŷ ──▶[loss]──▶ L
                                                          ▲
                                                          y
```

Two things follow from taking this literally.

**"Layer" is a convenience, not a primitive.** A dense layer is a matmul and an
add. The graph does not know about layers, and neither does the differentiation
algorithm. That is why frameworks can differentiate arbitrary code rather than
only recognised layer types.

**The graph is what gets differentiated.** {{ch:dl-backprop}} walks it
backwards. Every design decision in this chapter — what to store, when to
build the graph, how to batch — is really a decision about that traversal.

### 4.2 The forward pass has a side effect

Evaluating the graph produces the output. It also produces something less
obvious and more expensive: **the intermediate values that the backward pass
will need.**

```text
   inference:  x ──▶ ... ──▶ ŷ           discard everything as you go
   training:   x ──▶ ... ──▶ ŷ           KEEP h₁, a₁, z₁, ... for later
```

That difference is why training a model needs several times the memory of
serving it, and it is a fact about the algorithm rather than about any
framework. {{sec:7-internal-mechanics}} quantifies it and
{{ch:dl-backprop}} explains exactly which values are needed.

### 4.3 Batching, and why it is not just parallelism

The naive reason to batch is "do many examples at once". The real reason is
**arithmetic intensity** — the ratio of arithmetic performed to bytes moved.

```text
   one example :  matmul (1, n) x (n, m)
                  work = 2nm      data = n + nm + m  ≈ nm
                  intensity ≈ 2 operations per element loaded

   B examples  :  matmul (B, n) x (n, m)
                  work = 2Bnm     data = Bn + nm + Bm
                  intensity ≈ 2B operations per element loaded  (for large nm)
```

The weight matrix is loaded **once** and used for all $B$ examples. Modern
accelerators can perform hundreds of arithmetic operations in the time it takes
to load one number from memory, so an intensity of 2 leaves the machine almost
entirely idle and an intensity of 200 does not.

That is why batch size 1 is catastrophically slow and batch size 256 is not
256 times slower than batch size 1. {{sec:9-practical-example}} measures the
curve, and it has a knee rather than a slope.

### 4.4 Static and dynamic graphs

Two ways to obtain the graph, and the difference has shaped every framework:

**Define-and-run (static).** Build the graph once as a data structure, then
execute it many times. Enables whole-graph optimisation — operator fusion,
memory reuse, constant folding — and makes control flow awkward, because an
`if` on a tensor value is not something a fixed graph can express.

**Define-by-run (dynamic).** Build the graph as the code executes, by recording
operations on a tape. Ordinary Python control flow works, debugging works,
and the graph is rebuilt every iteration, which costs overhead and forfeits
whole-graph optimisation.

The 2026 position is that this is no longer a choice between frameworks but a
mode within one: write dynamically, then compile the hot path to a static graph
for execution. That resolution has been the direction of travel since roughly
2019 and it is now the default in the major frameworks.

## 5. Formal Explanation

### 5.1 The graph

A computational graph is a DAG $G = (V, E)$ where each node $v$ carries a
tensor value and each non-input node has an operation $f_v$ and an ordered list
of parents:

$$
v = f_v\big(u_1, u_2, \dots, u_{k}\big), \qquad (u_i, v) \in E
$$ (eq:graph-node)

The forward pass evaluates nodes in **topological order**, so every node's
parents are computed before it. Because the graph is acyclic such an order
exists, and because it is a *partial* order there are many valid ones — a
freedom that schedulers use for memory and parallelism.

Three properties matter for what follows:

**Acyclicity** is what makes topological order exist and is what
{{ch:dl-backprop}}'s reverse traversal requires. Recurrent networks appear
cyclic and are not: {{ch:dl-rnns}} unrolls them into an acyclic graph over time.

**Nodes may have several consumers.** A tensor used twice appears as the parent
of two nodes, and the gradient contributions from both must be *summed* — the
multivariate chain rule, and the source of a great many bugs in hand-written
autodiff.

**Operations are the primitive, not layers.** The set of operations a framework
can differentiate defines what it can train.

### 5.2 The batched forward pass

For a dense layer with input $\mat{H} \in \R^{B \times n_{\text{in}}}$ and
weights $\mat{W} \in \R^{n_{\text{out}} \times n_{\text{in}}}$:

$$
\mat{Z} = \mat{H}\mat{W}\T + \vec{1}_B\vec{b}\T, \qquad
\mat{H}' = \phi(\mat{Z})
$$ (eq:batched-layer)

with $\mat{Z}, \mat{H}' \in \R^{B \times n_{\text{out}}}$.

The bias term is written with an explicit outer product to make the shapes
honest, and in code it is a broadcast. **The batch dimension is first by
convention** in most frameworks, and that convention is worth following
mechanically, because half of all shape bugs are a disagreement about it.

> IMPORTANT: The shape rule that removes most confusion is to read a matmul as
> consuming the *last* axis of the left operand against the *first* axis of the
> right, leaving all other axes untouched. Under that reading a batch dimension
> is not special — it is simply an axis nothing consumed. Extending to sequences
> $(B, T, d)$ or images $(B, C, H, W)$ then requires no new rule.

### 5.3 What must be stored

For each operation, the backward pass needs whatever its local derivative
depends on:

{#tbl:forward-storage caption="What the forward pass must retain for each operation. The third column is what actually drives training memory, and it is why activation memory scales with batch size while parameters do not."}

| Operation | Backward needs | Stored size |
|---|---|---|
| $\mat{Z} = \mat{H}\mat{W}\T$ | $\mat{H}$ and $\mat{W}$ | $B n_{\text{in}}$ (activation) |
| bias add | nothing | 0 |
| ReLU | sign of input, or the output | $B n_{\text{out}}$ (or 1 bit each) |
| sigmoid, tanh | the *output* | $B n_{\text{out}}$ |
| GELU, SiLU | the *input* | $B n_{\text{out}}$ |
| softmax + CE | the probabilities and labels | $B C$ |

The parameters are stored anyway. The activations are extra, they scale with
$B$, and they are usually the larger number. {{sec:7-internal-mechanics}}
derives the total.

### 5.4 Epochs, steps and batches

Three units, routinely confused:

- A **step** (or iteration) is one parameter update — one forward, one
  backward, one optimiser step.
- A **mini-batch** is the group of examples in one step.
- An **epoch** is $\lceil N/B \rceil$ steps: one pass over the data.

> WARNING: **The epoch is a poor unit for comparing runs.** Two runs at
> different batch sizes see the same data per epoch and take different numbers
> of *updates*, and updates are what change the model. Comparing "10 epochs at
> batch 32" against "10 epochs at batch 512" compares 16× different amounts of
> optimisation. Report steps, or tokens, or examples-seen — and note that
> large-model work abandoned the epoch entirely, because such models see most
> data once.

### 5.5 Inference is a different computation

Not merely training with the backward pass removed. Four differences:

**No activation storage**, which is the memory difference above.

**Different behaviour from some layers.** Dropout is off; batch normalisation
uses running statistics rather than the batch's ({{ch:dl-normalization}}).
Frameworks expose this as a train/eval mode flag, and forgetting to set it is
one of the most common production bugs in deep learning.

**Different batching economics.** Serving often has a batch of one, which is
exactly the low-arithmetic-intensity regime of
{{sec:4-intuitive-explanation}}, and it is why inference servers batch requests
across users.

**Different numerics are acceptable.** Inference tolerates far more
quantisation than training, because there is no gradient to corrupt
({{part:15}}).

## 6. Mathematical Foundation

### 6.1 Why topological order is necessary and sufficient

**Sufficient.** If nodes are evaluated in an order where every node follows its
parents, then when $f_v$ is applied its arguments are available by construction.

**Necessary.** If $v$ is evaluated before some parent $u$, then $u$'s value is
undefined at that moment, and no amount of cleverness recovers it — the value
does not exist yet.

**Existence.** A finite DAG always has a topological order, by induction: a
finite DAG has at least one node with no incoming edges (otherwise following
edges backwards forever from any node would revisit one, giving a cycle);
remove it and recurse.

The reverse pass of {{ch:dl-backprop}} needs the *reverse* topological order,
which exists for the same reason applied to the reversed graph. This
symmetry — forward in topological order, backward in reverse — is the entire
structure of automatic differentiation, and everything else is bookkeeping.

### 6.2 FLOPs of a forward pass

A matrix product $(B \times n) \times (n \times m)$ performs $Bnm$
multiply-accumulates, conventionally counted as $2Bnm$ floating-point
operations. Summing over a network:

$$
F_{\text{fwd}} = 2B\sum_{l=1}^{L} n_{l}n_{l-1}
 \;+\; \underbrace{O\Big(B\sum_l n_l\Big)}_{\text{bias, activation}}
$$ (eq:forward-flops)

The second term is smaller by a factor of the layer width and is conventionally
ignored in FLOP counts — which is why reported FLOP figures understate the
*time* taken by elementwise operations, since those are bandwidth-bound rather
than arithmetic-bound. **FLOP counts predict compute-bound time and mislead
about memory-bound time**, and knowing which regime you are in is the point of
the next section.

### 6.3 Arithmetic intensity and the roofline

For an operation performing $W$ arithmetic operations while moving $Q$ bytes,
the **arithmetic intensity** is $I = W/Q$. A machine with peak throughput
$\pi$ operations per second and bandwidth $\beta$ bytes per second achieves at
best

$$
\text{throughput} = \min(\pi,\; I\beta)
$$ (eq:roofline)

so an operation is memory-bound when $I < \pi/\beta$ and compute-bound
otherwise. The ratio $\pi/\beta$ — the machine's **ridge point** — is of order
100 or more for contemporary accelerators in single precision, and higher still
for reduced precision.

Applying this to a dense layer at batch $B$, with 4 bytes per element:

$$
I = \frac{2Bnm}{4(Bn + nm + Bm)}
$$ (eq:layer-intensity)

For $B = 1$ and large $n, m$, the $nm$ term dominates the denominator and
$I \approx 1/2$ — two hundred times below the ridge point, so the machine runs
at under one per cent of peak. For $B \gg 1$ the $Bn$ and $Bm$ terms dominate
and $I \to \frac{2nm}{4(n+m)}$, which for $n = m = 1024$ is about 256 —
comfortably compute-bound.

**That is the whole argument for batching**, and it explains the shape of the
measured curve in {{sec:9-practical-example}}: throughput rises steeply with
batch size until the ridge point is crossed and then flattens, because past
that point the machine is arithmetic-limited and no longer waiting for memory.

### 6.4 Activation memory

Summing the storage column of {{tbl:forward-storage}}, a network storing one
activation tensor per layer needs

$$
M_{\text{act}} = B\sum_{l=1}^{L} n_l \cdot s
$$ (eq:activation-memory)

bytes, for element size $s$. Compare the parameters, $M_{\text{par}} = s\sum_l
n_l n_{l-1}$.

The ratio is instructive:

$$
\frac{M_{\text{act}}}{M_{\text{par}}}
 = \frac{B\sum_l n_l}{\sum_l n_l n_{l-1}}
 \approx \frac{B}{\bar{n}}
$$ (eq:memory-ratio)

for typical width $\bar{n}$. So activations dominate whenever the batch size
exceeds the layer width — routine for small models, and the reason that a wide
model at batch 8 is parameter-dominated while a narrow one at batch 1024 is
activation-dominated. In transformers the picture shifts again because
attention adds a term quadratic in sequence length
({{ch:tf-complexity}}).

## 7. Internal Mechanics

### 7.1 A tape

The simplest implementation of a dynamic graph is a list. Every operation
appends a record of itself:

```text
   tape = [
     (matmul, inputs=(x, W1),  output=z1),
     (add,    inputs=(z1, b1), output=a1),
     (relu,   inputs=(a1,),    output=h1),
     ...
   ]
```

The forward pass appends; the backward pass iterates the list in reverse. That
is the entire mechanism, and {{ch:dl-backprop}} implements it in about eighty
lines.

Two details that turn out to matter. The tape holds **references** to the input
tensors, which is why they stay alive in memory — the activation cost is a
consequence of the tape holding them, and clearing the tape is what frees them.
And the tape must record enough to compute local derivatives, which for most
operations means the inputs, and for a few (sigmoid, tanh, softmax) can be the
cheaper output.

### 7.2 Where the time goes

For a dense network on a modern accelerator, a rough breakdown of a training
step:

```text
   forward  matmuls         ~30%   compute-bound, near peak
   backward matmuls         ~60%   two matmuls per forward one (Ch 53)
   elementwise ops          ~5%    memory-bound; more if unfused
   optimiser update         ~5%    memory-bound, proportional to parameters
```

The 1:2 ratio between forward and backward matmuls is derived in
{{ch:dl-backprop}} and gives the familiar rule that **a training step costs
about three times a forward pass**.

The elementwise share is the one that varies wildly. Unfused, a chain of
add-bias, activation and dropout reads and writes the same tensor three times
and can consume a quarter of the step. Fused into the matmul epilogue, it
nearly vanishes. This is why the same model can be twice as fast in one
framework as another with identical mathematics.

### 7.3 Memory layout and why transposes are not free

A tensor is a flat buffer plus a shape and strides, exactly as in
{{ch:py-numpy}}. A transpose usually just swaps strides and copies nothing —
which is free — but the resulting array is no longer contiguous, and a matmul
kernel that requires contiguous input will then trigger a hidden copy.

The practical consequence is that `A @ B.T` and `A @ B` can differ measurably
in speed for identical FLOP counts, and which is faster depends on the library
and the layout. It is one of several reasons that predicting deep learning
performance from arithmetic alone fails.

### 7.4 In-place operations and the graph

An in-place operation overwrites its input. That is a memory saving and it
destroys a value the backward pass might need, which is why frameworks either
forbid it for such operations or raise an error at backward time.

The rule is mechanical: an operation may be performed in place only if its
input is not required by any backward function. ReLU qualifies when the
backward uses the *output*; GELU does not, because {{tbl:forward-storage}} says
its backward needs the input.

## 8. Implementation

```python {tier=A name=graph-and-forward}
"""A computational graph as an explicit object, evaluated in topological
order — the structure Chapter 53 will differentiate.
"""
import numpy as np

rng = np.random.default_rng(0)


# --- the graph --------------------------------------------------------------
class Node:
    """One tensor in the graph, with the operation that produced it."""
    _counter = 0

    def __init__(self, op=None, parents=(), name=None, value=None):
        self.op, self.parents, self.value = op, tuple(parents), value
        self.name = name or f"n{Node._counter}"
        Node._counter += 1

    def __repr__(self):
        shape = None if self.value is None else self.value.shape
        return f"<{self.name} {self.op or 'input'} {shape}>"


OPS = {
    "matmul": lambda a, b: a @ b,
    "add": lambda a, b: a + b,
    "relu": lambda a: np.maximum(0.0, a),
    "tanh": lambda a: np.tanh(a),
    "mse": lambda a, b: np.array([[np.mean((a - b) ** 2)]]),
}


def topological_order(node):
    """Section 6.1: parents before children. Depth-first with a visited set,
    which is O(V + E) and detects nothing about cycles — the graph is assumed
    acyclic, which is what 'DAG' asserts."""
    order, seen = [], set()

    def visit(n):
        if id(n) in seen:
            return
        seen.add(id(n))
        for p in n.parents:
            visit(p)
        order.append(n)

    visit(node)
    return order


def evaluate(node, feeds):
    """Evaluate the graph, returning (value, tape).

    The tape is what section 7.1 describes and what Chapter 53 walks
    backwards. Building it is the ONLY thing that distinguishes a training
    forward pass from an inference one.
    """
    order = topological_order(node)
    tape = []
    for n in order:
        if n.op is None:
            n.value = feeds[n.name]
        else:
            n.value = OPS[n.op](*[p.value for p in n.parents])
            tape.append((n.op, n.parents, n))
    return node.value, tape, order


# --- build a two-layer network as a graph -----------------------------------
B, D_IN, H, D_OUT = 8, 5, 4, 1
x = Node(name="x")
W1 = Node(name="W1")
b1 = Node(name="b1")
W2 = Node(name="W2")
b2 = Node(name="b2")
y = Node(name="y")

z1 = Node("matmul", (x, W1), "z1")
a1 = Node("add", (z1, b1), "a1")
h1 = Node("relu", (a1,), "h1")
z2 = Node("matmul", (h1, W2), "z2")
yhat = Node("add", (z2, b2), "yhat")
loss = Node("mse", (yhat, y), "loss")

feeds = {
    "x": rng.normal(size=(B, D_IN)),
    "W1": rng.normal(0, np.sqrt(2 / D_IN), (D_IN, H)),
    "b1": np.zeros((1, H)),
    "W2": rng.normal(0, np.sqrt(2 / H), (H, D_OUT)),
    "b2": np.zeros((1, D_OUT)),
    "y": rng.normal(size=(B, D_OUT)),
}

value, tape, order = evaluate(loss, feeds)

print("=" * 72)
print("the graph, in topological order (section 6.1)")
print("=" * 72)
for i, n in enumerate(order):
    parents = ", ".join(p.name for p in n.parents) or "-"
    print(f"  {i:>2}. {n.name:<6} {(n.op or 'input'):<8} "
          f"parents=({parents:<10}) shape={n.value.shape}")
print(f"\nloss = {value.item():.6f}")
print(f"tape length (differentiable operations): {len(tape)}")

print("\nEvery node appears after all of its parents. That ordering is what")
print("makes the forward pass well defined, and its REVERSE is what makes")
print("the backward pass of Chapter 53 well defined — the same property")
print("read in the other direction.")

# --- a node with two consumers, which is where hand-written autodiff breaks --
print("\n" + "=" * 72)
print("a tensor used twice: the gradient contributions must SUM")
print("=" * 72)
Node._counter = 0
u = Node(name="u")
v1 = Node("tanh", (u,), "v1")
v2 = Node("relu", (u,), "v2")
w = Node("add", (v1, v2), "w")
out = Node("mse", (w, Node(name="t")), "out")
feeds2 = {"u": rng.normal(size=(4, 3)), "t": np.zeros((4, 3))}
_, _, order2 = evaluate(out, feeds2)

consumers = {}
for n in order2:
    for p in n.parents:
        consumers.setdefault(p.name, []).append(n.name)
print("consumers of each node:")
for name, cs in consumers.items():
    flag = "  <-- used more than once" if len(cs) > 1 else ""
    print(f"  {name:<6} -> {cs}{flag}")

print("\nWhen a tensor feeds two operations, the chain rule says its gradient")
print("is the SUM of the contributions from both paths. A hand-written")
print("backward pass that assigns rather than accumulates silently drops one")
print("of them, and the symptom is a gradient that is wrong by exactly the")
print("magnitude of the missing branch — small enough to look like a")
print("learning-rate problem.")

# --- section 5.3: what the forward pass must keep ---------------------------
print("\n" + "=" * 72)
print("training keeps what inference discards (table 51.1)")
print("=" * 72)


def forward_memory(sizes, batch, bytes_per=4, training=True):
    """Bytes for parameters and (if training) stored activations."""
    params = sum(sizes[i] * sizes[i + 1] + sizes[i + 1]
                 for i in range(len(sizes) - 1))
    acts = batch * sum(sizes[1:]) if training else 0
    return params * bytes_per, acts * bytes_per


print(f"{'network':<26} {'batch':>6} {'params MB':>11} {'activations MB':>16} "
      f"{'ratio':>8}")
for label, sizes in (("small MLP  [784,256,256,10]", [784, 256, 256, 10]),
                     ("wide MLP   [784,4096,4096,10]", [784, 4096, 4096, 10])):
    for batch in (1, 32, 512):
        p, a = forward_memory(sizes, batch)
        print(f"{label:<26} {batch:>6} {p / 1e6:>11.2f} {a / 1e6:>16.2f} "
              f"{a / max(p, 1):>8.3f}")

print("\nEq. 51.7 predicts the ratio is about B/n_bar, and the table follows")
print("it. At batch 1 the parameters dominate completely. The small network")
print("has a typical width around 500, so eq. 51.7 says activations should")
print("reach parity at a batch around 500 — and at batch 512 the measured")
print("ratio is 0.99. The wide network has a typical width eight times")
print("larger, so at the same batch it is still firmly parameter-dominated.")
print("\nThat is the memory a framework consumes that a napkin calculation")
print("from the parameter count misses entirely — and it is why the same")
print("model trains on one accelerator and serves comfortably on a much")
print("smaller one.")
```

```python {tier=A name=batching-and-shapes}
"""Arithmetic intensity, the batching curve, and the shape rules that
prevent most deep learning bugs.
"""
import time

import numpy as np

rng = np.random.default_rng(1)


# --- section 6.3: the roofline, measured ------------------------------------
def intensity(B, n, m, bytes_per=4):
    """Eq. 51.6: arithmetic operations per byte moved."""
    work = 2 * B * n * m
    data = bytes_per * (B * n + n * m + B * m)
    return work / data


print("=" * 72)
print("arithmetic intensity and the batching curve (eqs. 51.5, 51.6)")
print("=" * 72)
n = m = 1024
A_w = rng.normal(size=(n, m)).astype(np.float32)
print(f"a {n}x{m} dense layer, float32\n")
print(f"{'batch':>7} {'intensity':>11} {'GFLOP/s':>10} "
      f"{'us per example':>16} {'regime':<18}")

ridge_guess = None
prev_per_ex = None
for B in (1, 2, 8, 32, 128, 512, 2048):
    X = rng.normal(size=(B, n)).astype(np.float32)
    reps = max(20, int(4e9 / (2 * B * n * m)))
    for _ in range(5):
        X @ A_w                                    # warm up
    # median of five timed blocks: a single block is badly contaminated by
    # thread-pool wake-up and frequency scaling at small batch sizes
    trials = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(reps):
            X @ A_w
        trials.append((time.perf_counter() - t0) / reps)
    dt = float(np.median(trials))
    gflops = 2 * B * n * m / dt / 1e9
    per_ex = dt / B * 1e6
    I = intensity(B, n, m)
    regime = "memory-bound" if I < 30 else "approaching compute"
    print(f"{B:>7} {I:>11.1f} {gflops:>10.1f} {per_ex:>16.2f} {regime:<18}")

print("\nThe per-example cost falls steeply and then flattens — a knee, not")
print("a slope. Eq. 51.6 explains it: at batch 1 the weight matrix is loaded")
print("from memory to be used once, giving an intensity below 1, and the")
print("machine spends nearly all its time waiting. Each additional example")
print("reuses that same loaded matrix.")
print("\nPast the knee the operation is arithmetic-limited and further")
print("batching buys nothing per example — it only costs memory. That is the")
print("real reason to batch, and it is a bandwidth argument rather than a")
print("parallelism one.")
print("\nTwo honest caveats about this measurement. The very small batches")
print("are noisy even with the median of five timed blocks, because the")
print("per-call overhead of the library's threading is comparable to the")
print("work itself. And the largest batch does not improve on the previous")
print("one, because once the operation is compute-bound the only thing left")
print("to gain would be better cache behaviour, and a larger working set")
print("makes that worse rather than better.")

# --- the same argument in reverse: why serving batches requests -------------
print("\n" + "=" * 72)
print("the inference consequence")
print("=" * 72)
X1 = rng.normal(size=(1, n)).astype(np.float32)
X64 = rng.normal(size=(64, n)).astype(np.float32)
for label, X in (("one request at a time", X1), ("64 requests batched", X64)):
    reps = 200
    X @ A_w
    t0 = time.perf_counter()
    for _ in range(reps):
        X @ A_w
    dt = (time.perf_counter() - t0) / reps
    print(f"{label:<24} {dt * 1e6:>9.1f} us total   "
          f"{dt / len(X) * 1e6:>8.2f} us per request")
print("\nThat is the trade an inference server makes, and it is worth being")
print("precise about which direction each number moves. Batching 64 requests")
print("makes each individual request take LONGER end to end — the whole")
print("batch must finish before any of it returns — while cutting the cost")
print("per request severalfold. Throughput improves; per-request latency")
print("degrades. Section 5.5's note that serving lives at batch 1 is why the")
print("trade is usually worth making anyway, and the batching window is")
print("chosen so that the added latency stays inside the service objective.")

# --- shape discipline -------------------------------------------------------
print("\n" + "=" * 72)
print("the shape rule that prevents most bugs (section 5.2)")
print("=" * 72)
print("Read a matmul as: consume the LAST axis of the left operand against")
print("the FIRST axis of the right; leave every other axis alone.\n")
cases = [
    ("dense, batched", (32, 784), (784, 256)),
    ("sequence, batched", (8, 128, 512), (512, 2048)),
    ("image features", (16, 49, 768), (768, 768)),
]
for label, sa, sb in cases:
    a, b = np.zeros(sa), np.zeros(sb)
    out = a @ b
    print(f"{label:<20} {str(sa):>18} @ {str(sb):<12} -> {str(out.shape)}")
print("\nNo new rule was needed for the three-dimensional cases. A batch or")
print("sequence axis is simply an axis that nothing consumed.")

# --- the broadcasts that fail SILENTLY --------------------------------------
print("\n" + "=" * 72)
print("broadcasts that do the wrong thing without erroring")
print("=" * 72)

B_, C = 6, 4
logits = rng.normal(size=(B_, C))
bias_ok = rng.normal(size=(C,))                # per-CLASS bias, correct
bias_bad = rng.normal(size=(B_,))              # per-EXAMPLE, a mistake

print(f"logits {logits.shape}, correct bias {bias_ok.shape}")
print(f"  logits + bias  -> {(logits + bias_ok).shape}   correct\n")
print(f"logits {logits.shape}, wrong bias {bias_bad.shape}")
try:
    r = logits + bias_bad
    print(f"  logits + bias  -> {r.shape}")
except ValueError as e:
    print(f"  raises: {str(e)[:60]}")
print(f"  ...but reshaped to {(B_, 1)} it broadcasts happily:")
print(f"  logits + bias[:, None] -> {(logits + bias_bad[:, None]).shape}   "
      f"WRONG, and silent")

print("\nThe second form adds a per-example constant to every class, which")
print("leaves the softmax output completely unchanged (it is shift-invariant")
print("per row) — so the bug produces no error, no shape mismatch, and no")
print("visible symptom beyond a bias vector that never learns anything.")

# a subtler one: the accidental outer product
print("\nthe accidental outer product:")
pred = rng.normal(size=(B_,))
targ = rng.normal(size=(B_,))
correct = float(np.mean((pred - targ) ** 2))
wrong = float(np.mean((pred[:, None] - targ[None, :]) ** 2))
print(f"  mean((pred - targ)**2)                     = {correct:.4f}  correct")
print(f"  mean((pred[:,None] - targ[None,:])**2)     = {wrong:.4f}  WRONG")
print(f"  the second compares every prediction against every target:")
print(f"  it computed a {B_}x{B_} matrix where a length-{B_} vector was meant")
print("\nBoth of these run, produce a plausible float, and train to a")
print("plausible-looking loss curve. Shape assertions at layer boundaries")
print("are cheap and catch all of them:\n")


def assert_shape(t, expected, name):
    if t.shape != expected:
        raise AssertionError(f"{name}: expected {expected}, got {t.shape}")
    return t


try:
    assert_shape(pred[:, None] - targ[None, :], (B_,), "residual")
except AssertionError as e:
    print(f"  caught: {e}")
```

## 9. Practical Example

```python {tier=A name=forward-pass-in-practice}
"""A complete forward pass with shape tracking, FLOP accounting and a
train/eval mode — the three things a real implementation must get right.
"""
import numpy as np

rng = np.random.default_rng(5)


class Layer:
    """A dense layer that reports its own shapes, FLOPs and stored memory."""

    def __init__(self, n_in, n_out, act="relu", name="fc", seed=0):
        rs = np.random.default_rng(seed)
        self.W = rs.normal(0, np.sqrt(2.0 / n_in), (n_in, n_out))
        self.b = np.zeros(n_out)
        self.act, self.name = act, name
        self.cache = None

    def forward(self, X, training=True):
        Z = X @ self.W + self.b
        if self.act == "relu":
            H = np.maximum(0.0, Z)
        elif self.act == "tanh":
            H = np.tanh(Z)
        else:
            H = Z
        # section 4.2: the side effect. Only training keeps this.
        self.cache = (X, Z) if training else None
        return H

    def flops(self, B):
        return 2 * B * self.W.shape[0] * self.W.shape[1]

    def params(self):
        return self.W.size + self.b.size

    def stored_bytes(self, B, bytes_per=4):
        if self.cache is None:
            return 0
        return bytes_per * sum(a.size for a in self.cache)


class Net:
    def __init__(self, sizes, seed=0):
        self.layers = [
            Layer(sizes[i], sizes[i + 1],
                  act="relu" if i < len(sizes) - 2 else "linear",
                  name=f"fc{i + 1}", seed=seed + i)
            for i in range(len(sizes) - 1)]

    def forward(self, X, training=True, trace=False):
        if trace:
            print(f"{'layer':<8} {'input':>16} {'weights':>16} "
                  f"{'output':>16} {'MFLOPs':>9} {'stored MB':>11}")
            print(f"{'input':<8} {str(X.shape):>16} {'-':>16} "
                  f"{str(X.shape):>16} {'-':>9} {'-':>11}")
        h = X
        for L in self.layers:
            prev = h.shape
            h = L.forward(h, training=training)
            if trace:
                print(f"{L.name:<8} {str(prev):>16} {str(L.W.shape):>16} "
                      f"{str(h.shape):>16} {L.flops(len(X)) / 1e6:>9.2f} "
                      f"{L.stored_bytes(len(X)) / 1e6:>11.3f}")
        return h

    def totals(self, B, training=True):
        f = sum(L.flops(B) for L in self.layers)
        p = sum(L.params() for L in self.layers)
        m = sum(L.stored_bytes(B) for L in self.layers) if training else 0
        return f, p, m


# --- a traced forward pass --------------------------------------------------
print("=" * 72)
print("a forward pass, traced (eqs. 51.4, 51.6)")
print("=" * 72)
net = Net([784, 512, 512, 10], seed=1)
X = rng.normal(size=(64, 784))
out = net.forward(X, training=True, trace=True)
f, p, m = net.totals(64)
print(f"\ntotal: {f / 1e6:.1f} MFLOPs forward, {p:,} parameters, "
      f"{m / 1e6:.2f} MB stored")
print(f"a training step is roughly 3x the forward FLOPs (section 7.2): "
      f"{3 * f / 1e6:.1f} MFLOPs")

# --- training vs inference memory -------------------------------------------
print("\n" + "=" * 72)
print("training keeps activations; inference does not (section 5.5)")
print("=" * 72)
print(f"{'batch':>7} {'training MB':>13} {'inference MB':>14} "
      f"{'x more to train':>17}")
for B in (1, 8, 64, 256, 1024):
    Xb = rng.normal(size=(B, 784))
    net.forward(Xb, training=True)
    _, params, m_train = net.totals(B, training=True)
    net.forward(Xb, training=False)
    m_inf = sum(L.stored_bytes(B) for L in net.layers)
    par_mb = params * 4 / 1e6
    print(f"{B:>7} {(par_mb + m_train / 1e6):>13.2f} "
          f"{(par_mb + m_inf / 1e6):>14.2f} "
          f"{(par_mb + m_train / 1e6) / (par_mb + m_inf / 1e6):>17.2f}x")

print("\nAt batch 1 the two are nearly identical — the parameters dominate.")
print("By batch 1024 training needs several times the memory, and every byte")
print("of the difference is activations being held for a backward pass that")
print("inference never performs.")

# --- the train/eval mode bug ------------------------------------------------
print("\n" + "=" * 72)
print("the train/eval mode bug (section 5.5)")
print("=" * 72)


class Dropout:
    """Deliberately written to show what forgetting the mode flag costs."""

    def __init__(self, p=0.5):
        self.p = p

    def forward(self, X, training=True, rs=None):
        if not training:
            return X                                  # identity at eval
        mask = (rs.random(X.shape) > self.p) / (1 - self.p)
        return X * mask


rs = np.random.default_rng(9)
h = rng.normal(size=(2000, 64))
drop = Dropout(0.5)

eval_out = drop.forward(h, training=False)
train_outs = np.array([drop.forward(h, training=True, rs=rs).mean()
                       for _ in range(50)])
print(f"activation mean, eval mode      : {eval_out.mean():>9.5f}")
print(f"activation mean, train mode     : {train_outs.mean():>9.5f} "
      f"(mean over 50 masks)")
print(f"activation mean, train mode SD  : {train_outs.std():>9.5f}")
print(f"\nthe expectations match — inverted dropout rescales by 1/(1-p) so")
print(f"that they do — but a single train-mode forward pass differs from")
print(f"the eval-mode one by a random amount:")
one = drop.forward(h, training=True, rs=rs)
print(f"  max |single train-mode pass - eval-mode pass| = "
      f"{np.abs(one - eval_out).max():.4f}")
print(f"  fraction of activations zeroed = {(one == 0).mean():.4f}")

print("\nServing a model left in training mode gives a DIFFERENT answer for")
print("the same input on every call, with half the activations missing. The")
print("expectation is right, so aggregate metrics computed over a large")
print("evaluation set look almost correct — which is what makes this bug")
print("survive. The symptom is per-request nondeterminism, not a bad score.")

# --- and the FLOP/time gap of section 6.2 -----------------------------------
print("\n" + "=" * 72)
print("FLOPs predict compute-bound time and mislead about the rest")
print("=" * 72)
import time

B = 256
Xb = rng.normal(size=(B, 784))
t0 = time.perf_counter()
for _ in range(20):
    net.forward(Xb, training=True)
dt_full = (time.perf_counter() - t0) / 20

Wc = [L.W for L in net.layers]
t0 = time.perf_counter()
for _ in range(20):
    h = Xb
    for W in Wc:
        h = h @ W                                     # matmuls ONLY
dt_mm = (time.perf_counter() - t0) / 20

f, _, _ = net.totals(B)
print(f"matmuls only          : {dt_mm * 1e3:>7.2f} ms  "
      f"({f / dt_mm / 1e9:>6.1f} GFLOP/s)")
print(f"full forward pass     : {dt_full * 1e3:>7.2f} ms")
print(f"elementwise overhead  : {(dt_full - dt_mm) * 1e3:>7.2f} ms "
      f"({(dt_full - dt_mm) / dt_full:.0%} of the total)")
print("\nThe FLOP count in eq. 51.4 counts only the matmuls, and they are")
print("not the whole time. The bias adds and activations contribute a")
print("negligible number of FLOPs and a non-negligible fraction of the")
print("runtime, because they are bandwidth-bound (section 6.3).")
print("\nThat gap is what operator fusion closes, and it is why two")
print("frameworks running identical mathematics can differ by a factor of")
print("two in wall-clock time.")
```

## 10. Production Considerations

**Batch size is a systems parameter as much as a learning one.** It sets
arithmetic intensity (throughput), activation memory (feasibility), and gradient
variance (learning). {{ch:dl-optimizers}} covers the third; the first two are
measured here, and the knee in the throughput curve is the number to find for
your hardware.

**Serving batches requests.** The measurement shows the per-request cost falling
sharply from batch 1 to batch 64. Production inference servers buffer requests
for a few milliseconds to form a batch, trading a little latency for a large
throughput gain — and the right buffer window is derived from where your own
curve's knee sits.

**Always set eval mode explicitly.** The measured dropout example produces a
different answer per call with no error. The failure is per-request
nondeterminism rather than a degraded aggregate metric, which is exactly the
kind of thing {{ch:mle-drift}}'s prediction-distribution monitoring catches and
an accuracy dashboard does not.

**Watch activation memory, not parameter count.** The measured ratio grows with
batch size, and out-of-memory failures during training are almost always
activations. Reducing batch size, gradient checkpointing
({{ch:dl-backprop}}) and activation offloading are the levers.

**Profile before optimising.** The measured elementwise overhead is a
substantial fraction of a small model's step and nearly zero in a fused
implementation. FLOP counts will not tell you which situation you are in.

## 11. Common Mistakes

**Transposing a weight matrix by guessing.** Apply the last-axis/first-axis
rule from {{sec:5-formal-explanation}} instead.

**A bias with the batch shape rather than the feature shape.** The measured
example adds a per-example constant that a softmax then discards entirely — no
error, no symptom, a parameter that never learns.

**The accidental outer product.** `pred[:, None] - targ[None, :]` produces a
$B \times B$ matrix where a length-$B$ vector was meant, and the loss is a
plausible number.

**Comparing runs by epoch at different batch sizes.** Different numbers of
updates; report steps.

**Forgetting eval mode.** Measured above.

**Assuming FLOPs predict time.** The measured elementwise overhead is
invisible in a FLOP count.

**Estimating training memory from the parameter count.** Activations dominate
past a modest batch size.

**Performing an in-place operation on a value the backward pass needs.**
{{sec:7-internal-mechanics}} gives the rule.

## 12. Failure Modes

**Silent broadcasting.** The most dangerous class in this chapter, because
NumPy and every framework will happily broadcast shapes you did not intend.
Both measured examples run cleanly and produce a trainable loss.

**Out-of-memory at a batch size that worked yesterday.** Activation memory
scales with batch size *and* with sequence length; a longer input than usual can
exhaust memory without any code change.

**Throughput collapse at small batch.** A pipeline change that reduces the
effective batch — gradient accumulation misconfigured, or a data loader
starving the device — produces a model that trains correctly and ten times too
slowly, with no error.

**Non-contiguous tensors triggering hidden copies.** A transpose is free and
the copy the next kernel makes is not, so a reshape in the wrong place can cost
more than the operation it feeds.

**Graph retained across iterations.** In a dynamic framework, keeping a
reference to a loss tensor beyond the step keeps its entire tape alive, and
therefore every activation. The symptom is memory growing linearly with
iteration until it fails.

## 13. Alternatives

**Static graph frameworks** trade flexibility for whole-graph optimisation.
Historically the defining difference between frameworks; now a compilation mode
within them.

**Symbolic differentiation** computes an explicit derivative expression rather
than evaluating one numerically. Exact, and the expression can blow up
exponentially in size for deep compositions — which is why autodiff, not
symbolic differentiation, is what frameworks implement.

**Forward-mode automatic differentiation** propagates derivatives alongside
values in the forward pass, costing $O(n_{\text{inputs}})$ passes. Ideal when
inputs are few and outputs many; a loss function is the opposite case, which is
why reverse mode wins ({{ch:dl-backprop}}).

**Numerical differentiation** by finite differences needs no graph at all,
costs one forward pass per parameter, and is far too slow for training. It
remains essential for *verifying* an autodiff implementation, and
{{ch:dl-backprop}} uses it for exactly that.

## 14. Evaluation

**Assert shapes at layer boundaries.** One line per layer, and it catches every
silent broadcast in {{sec:8-implementation}}.

**Trace shapes once per architecture change.** The traced table in
{{sec:9-practical-example}} takes seconds to produce and makes an entire class
of bug impossible.

**Measure the throughput curve.** Find your hardware's knee rather than
inheriting a batch size.

**Profile the elementwise fraction.** If it is large, the fix is fusion, not a
faster function.

**Check memory against the prediction.** If measured training memory greatly
exceeds {{eq:activation-memory}}, something is being retained that should not
be — usually a graph held across iterations.

## 15. Advanced Concepts

**Operator fusion.** Combining several elementwise operations, or an activation
into a matmul's epilogue, so the intermediate tensor is never written to memory.
This is where most of the gap between naive and optimised implementations lives,
and it is the reason {{ch:tf-efficient}}'s FlashAttention is a memory argument.

**Recomputation as a memory/compute trade.** Gradient checkpointing stores a
subset of activations and recomputes the rest, reducing memory from $O(L)$ to
$O(\sqrt{L})$ for roughly one extra forward pass ({{ch:dl-backprop}}).

**Graph-level memory planning.** With a static graph, buffers can be reused for
tensors whose lifetimes do not overlap, which is an interval-graph colouring
problem and can reduce peak memory substantially.

**Mixed-precision graphs.** Different operations in different precisions:
matmuls in `bfloat16`, reductions and the optimiser state in `float32`. The
partition is chosen by numerical sensitivity, and {{part:15}} treats it.

**Vectorising over the batch automatically.** Some frameworks provide a
transform that lifts a single-example function to a batched one without the
author writing batch dimensions at all — which is a compiler operating on the
graph, and a good demonstration that the graph is the real object.

## 16. Connection to Previous Chapters

{{ch:dl-neural-networks}} gave {{eq:mlp-forward}}; this chapter makes it a graph
and adds the shapes, the costs and the storage.
{{ch:dl-activations}} supplied the elementwise nodes, and
{{tbl:forward-storage}} records which of them need their input rather than
their output — the memory consequence of that chapter's analysis.
{{ch:py-numpy}} supplied broadcasting and strides, which are the same mechanics
here, and the silent-broadcast failures measured in
{{sec:8-implementation}} are the deep learning form of that chapter's warning.
{{ch:mle-reproducibility}} supplied the summation-order argument that makes
`float32` accumulation necessary.

Forward: {{ch:dl-backprop}} traverses this graph in reverse and is the reason
every design choice here was made. {{ch:dl-normalization}} adds nodes whose
behaviour differs between train and eval, making the mode flag consequential.
{{ch:tf-complexity}} redoes {{eq:activation-memory}} for attention, where the
sequence-length term changes the conclusion.

## 17. Exercises

**Beginner**

1. What is a computational graph, and what are its nodes?
2. Why must nodes be evaluated in topological order?
3. What does a training forward pass store that an inference one does not?
4. Define step, batch and epoch.
5. Why is comparing runs by epoch misleading?

**Intermediate**

6. Using {{eq:forward-flops}}, compute the forward FLOPs of a
   $512 \to 2048 \to 2048 \to 512$ network at batch 128.
7. Using {{eq:layer-intensity}}, compute the arithmetic intensity of a
   $1024 \times 1024$ layer at batch 1 and batch 256.
8. Using {{eq:memory-ratio}}, find the batch size at which activations exceed
   parameters for a network of typical width 512.
9. Explain why a tensor with two consumers needs its gradients summed.
10. Give two operations that can be performed in place and two that cannot.
11. Why does an inference server batch requests from different users?

**Advanced**

12. Prove that a finite DAG has a topological order.
13. Derive {{eq:layer-intensity}} and find the batch size at which a
    $n \times n$ layer becomes compute-bound on a machine with ridge point
    $\pi/\beta = 100$.
14. Explain why symbolic differentiation can produce exponentially large
    expressions where autodiff does not.
15. Design a memory-planning scheme that reuses buffers for tensors with
    disjoint lifetimes, and say what makes it hard in a dynamic graph.

**Implementation**

16. Extend the graph class with shape inference that runs before execution and
    reports mismatches with node names.
17. Implement a context manager that asserts every layer's output shape against
    a declared specification.
18. Measure your own hardware's throughput curve and locate the knee.
19. Instrument a training loop to report peak activation memory and compare it
    against {{eq:activation-memory}}.

**Reasoning**

20. A model trains at 30% of the expected throughput with no errors. List your
    hypotheses in order.
21. Memory usage grows linearly with iteration until the job dies. What is the
    most likely cause?

## 18. Interview Questions

**"Why does training need more memory than inference?"** — Stored activations
for the backward pass. A strong answer adds that they scale with batch size
while parameters do not, and gives the crossover.

**"Why batch?"** — Arithmetic intensity, not parallelism. The weight matrix is
loaded once and reused, so the machine stops waiting on memory. A candidate who
says "to use the GPU better" is not wrong and has not said why.

**"What is the difference between a static and a dynamic graph?"** — When the
graph is built. The strong answer adds that the distinction is now a mode rather
than a framework choice.

**"How many FLOPs is a training step relative to a forward pass?"** — About
three, with the reason: the backward pass performs two matmuls per forward one.

**"You have a shape error. How do you find it?"** — Apply the
last-axis/first-axis rule and trace shapes forward from the input. Guessing
transposes until it runs is the wrong answer, and is what most people do.

**"Your model gives different predictions for the same input on successive
calls. What is wrong?"** — Training mode left on: dropout or batch-norm batch
statistics. A good answer notes that aggregate metrics can look fine because
the expectation is correct.

## 19. Research Questions

**How much can compilers close the fusion gap automatically?** Hand-written
fused kernels still beat compiler output on the operations that matter most.
Whether that gap is fundamental or an engineering lag is unresolved.
{{maturity:EMERGING}}

**What is the right intermediate representation for deep learning?** Several
compiler stacks target the problem with different abstractions, and none has
become the obvious answer the way LLVM's IR did for scalar code.
{{maturity:EMERGING}}

**Can activation memory be reduced without recomputation?** Reversible
architectures allow activations to be reconstructed from later ones rather than
stored, eliminating the memory cost at some constraint on the architecture.
Interesting and not widely adopted, which is itself informative.
{{maturity:EMERGING}}

## 20. Chapter Summary

A network is a directed acyclic graph of tensors and operations, and that
representation — not the layer abstraction — is what gets differentiated. The
forward pass evaluates it in topological order; {{ch:dl-backprop}} traverses the
reverse. The existence of both orders is the same fact read twice, and it is the
whole structure of automatic differentiation.

The forward pass has a side effect: it stores the intermediate values the
backward pass will need. {{tbl:forward-storage}} records which operations need
their input and which can use their output, and the measured memory table shows
activations growing from negligible at batch 1 to several times the parameters
at batch 1024 — exactly the $B/\bar{n}$ ratio of {{eq:memory-ratio}}.

Batching is a bandwidth argument, not a parallelism one. The weight matrix is
loaded once and reused across the batch, so arithmetic intensity rises linearly
with $B$ until the machine's ridge point is crossed. The measured throughput
curve has a knee rather than a slope, and that knee is the batch size worth
finding for your hardware. The same argument in reverse is why inference servers
batch requests from different users.

The shape rule that prevents most bugs is to read a matmul as consuming the last
axis of the left operand against the first axis of the right. Under that reading
batch and sequence axes need no special treatment, because they are simply axes
nothing consumed.

Two broadcast failures were measured and both are silent. A bias with the batch
shape adds a per-example constant that a softmax discards entirely, so the
parameter never learns and nothing errors. An accidental outer product computes
a $B \times B$ matrix where a length-$B$ vector was meant and yields a plausible
loss. Shape assertions at layer boundaries cost one line and catch both.

FLOP counts predict compute-bound time and mislead about the rest: the measured
elementwise overhead was a meaningful fraction of a small model's step despite
contributing almost no FLOPs, because those operations are bandwidth-bound.
Fusion is what closes that gap, and it is why identical mathematics can run at
very different speeds.

Finally, inference is not training minus the backward pass. It stores nothing,
some layers behave differently, it usually runs at batch 1, and it tolerates
lower precision. The measured dropout example shows the cost of forgetting the
mode flag: a different answer on every call, with correct expectations, so
aggregate metrics look fine and only per-request behaviour is wrong.

## 21. Further Reading

{{cite:rumelhart1986}} is worth reading again at this point, specifically for
how it describes the forward computation before introducing the backward one.
The graph framing is implicit there and became explicit only with the first
autodiff frameworks.

{{cite:pedregosa2011}} for the `fit`/`transform`/`predict` separation, which is
the same train/eval distinction as {{sec:5-formal-explanation}}'s mode flag,
solved by a different convention. Comparing the two designs is instructive:
scikit-learn puts the mode in the method name, frameworks put it in object
state, and the second is why the bug in {{sec:9-practical-example}} exists.

**On the systems material**, the roofline model of {{eq:roofline}} is the
standard tool for reasoning about whether an operation is memory- or
compute-bound, and it is worth knowing outside deep learning entirely. Any
treatment of high-performance computing will cover it; the deep learning
application is unusually clean because the two regimes are so sharply separated.

**Where to go next:** {{ch:dl-losses}} completes the forward pass by specifying
the final node, then {{ch:dl-backprop}} differentiates everything built here.
Those three chapters are best read consecutively.
