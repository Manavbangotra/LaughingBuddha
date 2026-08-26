# -*- coding: utf-8 -*-
# Extracted from: Chapter 11 — Derivatives, Partial Derivatives, Gradients, and the Chain Rule
# Source: src/.../ch011-derivatives.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A minimal reverse-mode automatic differentiation engine.

Each Value node stores its data, its parents, and a closure that knows how to
push gradient to those parents. Calling backward() topologically sorts the
graph and runs those closures in reverse — which is exactly what PyTorch does,
minus the tensors, kernels and device management.
"""
import math


class Value:
    def __init__(self, data, parents=(), op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._parents = set(parents)
        self._op = op

    def __repr__(self):
        return f"Value({self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # Addition passes the gradient through unchanged, to BOTH parents.
            # The += is essential: a node used twice must accumulate.
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # Each parent receives the gradient times the OTHER parent's value,
            # which is why forward values must be retained.
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1.0 - s) * out.grad      # eq. 2.16

        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other if isinstance(other, Value) else Value(-other))

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def backward(self):
        """Topologically sort, then run each node's backward closure in reverse."""
        order, visited = [], set()

        def build(node):
            if node not in visited:
                visited.add(node)
                for parent in node._parents:
                    build(parent)
                order.append(node)

        build(self)
        self.grad = 1.0                    # d(self)/d(self) = 1
        for node in reversed(order):
            node._backward()


# --- reproduce the hand computation of section 6.2 --------------------------
x = Value(2.0)
y = Value(-5.0)
f = (x + y) * x.relu()
f.backward()
print("f(x, y) = (x + y) * relu(x)  at x=2, y=-5")
print(f"  f      = {f.data}")
print(f"  df/dx  = {x.grad}   (hand-computed: -1)")
print(f"  df/dy  = {y.grad}   (hand-computed:  2)")
assert x.grad == -1.0 and y.grad == 2.0

# --- a full logistic-regression step, differentiated automatically ----------
print("\nlogistic regression on 3 features, gradient by autodiff:")
w = [Value(0.5), Value(-0.3), Value(0.8)]
xs = [1.2, -0.7, 0.4]
y_true = 1.0

z = w[0] * xs[0] + w[1] * xs[1] + w[2] * xs[2]
p = z.sigmoid()
# Binary cross-entropy for y = 1 reduces to -log(p).
loss = -p.log()
loss.backward()

p_val = p.data
print(f"  z = {z.data:.6f}, p = {p_val:.6f}, loss = {loss.data:.6f}")
print(f"  autodiff gradient : {[round(wi.grad, 6) for wi in w]}")
analytic = [(p_val - y_true) * xi for xi in xs]      # eq. 11.19
print(f"  analytic (p-y)*x  : {[round(a, 6) for a in analytic]}")
for wi, a in zip(w, analytic):
    assert abs(wi.grad - a) < 1e-9
print("  they agree exactly.")

# --- branching: a node used twice accumulates gradient ----------------------
a = Value(3.0)
out = a * a + a          # d/da (a^2 + a) = 2a + 1 = 7
out.backward()
print(f"\nd/da (a*a + a) at a=3 : {a.grad}  (expected 7)")
print("The += in each _backward is what makes this correct — a node with")
print("several consumers must SUM its incoming gradients (eq. 11.6).")
assert a.grad == 7.0
