# -*- coding: utf-8 -*-
# Extracted from: Chapter 51 — Forward Propagation and Computational Graphs
# Source: src/.../ch051-forward.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

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
