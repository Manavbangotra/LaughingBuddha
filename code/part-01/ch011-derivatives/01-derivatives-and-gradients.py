# Extracted from: Chapter 11 — Derivatives, Partial Derivatives, Gradients, and the Chain Rule
# Source: src/.../ch011-derivatives.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Derivatives, gradients, the chain rule, and backpropagation by hand.

Every analytic derivative is checked against a numerical one.
"""
import numpy as np

# --- eq. 11.1: the derivative as a limit ------------------------------------
def f(x):
    return x ** 2


print(f"{'h':>10} {'difference quotient':>22}")
for h in (1.0, 0.1, 0.01, 0.001, 1e-6):
    print(f"{h:>10} {(f(3 + h) - f(3)) / h:>22.9f}")
print("converging to 6 = 2x at x = 3\n")


def numerical_gradient(fn, x, h=1e-6):
    """Central-difference gradient. More accurate than the forward difference:
    its error is O(h^2) rather than O(h)."""
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    for i in range(x.size):
        step = np.zeros_like(x)
        step.flat[i] = h
        grad.flat[i] = (fn(x + step) - fn(x - step)) / (2 * h)
    return grad


# --- partial derivatives and the gradient (eq. 11.3) ------------------------
def g(v):
    x, y = v
    return x**2 * y + 3 * y


def g_grad(v):
    x, y = v
    return np.array([2 * x * y, x**2 + 3])


point = np.array([2.0, 5.0])
print(f"analytic  grad g at {point}: {g_grad(point)}")
print(f"numerical grad g at {point}: {np.round(numerical_gradient(g, point), 6)}")
assert np.allclose(g_grad(point), numerical_gradient(g, point), atol=1e-5)

# --- section 6.1: the gradient really is the steepest direction -------------
grad = g_grad(point)
unit_grad = grad / np.linalg.norm(grad)
print(f"\nrate of increase in 2000 random unit directions vs the gradient:")
best_rate, best_dir = -np.inf, None
rng = np.random.default_rng(0)
for _ in range(2000):
    u = rng.normal(size=2)
    u /= np.linalg.norm(u)
    rate = grad @ u                              # eq. 11.5
    if rate > best_rate:
        best_rate, best_dir = rate, u
print(f"  best random direction : rate {best_rate:.6f}")
print(f"  the gradient direction: rate {grad @ unit_grad:.6f}  <- larger")
print(f"  ||grad||              : {np.linalg.norm(grad):.6f}  <- eq. 11.10")
assert grad @ unit_grad >= best_rate - 1e-9

# Perpendicular to the gradient, nothing changes — a contour line.
perp = np.array([-unit_grad[1], unit_grad[0]])
print(f"  perpendicular direction: rate {grad @ perp:+.2e}  <- zero")

# --- section 6.2: backpropagation by hand -----------------------------------
print("\n" + "=" * 62)
print("backpropagation on f(x, y) = (x + y) * max(0, x)  at x=2, y=-5")
print("=" * 62)

x, y = 2.0, -5.0

# forward
a = x + y
b = max(0.0, x)
out = a * b
print(f"forward : a = {a}, b = {b}, f = {out}")

# backward
d_out = 1.0
d_a = d_out * b                  # multiplication: gradient x the OTHER input
d_b = d_out * a
d_x_via_a = d_a * 1.0            # addition passes gradient through
d_y = d_a * 1.0
d_x_via_b = d_b * (1.0 if x > 0 else 0.0)    # relu gates it
d_x = d_x_via_a + d_x_via_b      # eq. 11.13: branching SUMS

print(f"backward: df/da = {d_a}, df/db = {d_b}")
print(f"          df/dx via a = {d_x_via_a}, via b = {d_x_via_b}, "
      f"total = {d_x}")
print(f"          df/dy = {d_y}")


def f_xy(v):
    return (v[0] + v[1]) * max(0.0, v[0])


num = numerical_gradient(f_xy, np.array([x, y]))
print(f"\nhand-computed : [{d_x}, {d_y}]")
print(f"numerical     : {np.round(num, 6)}")
assert np.allclose([d_x, d_y], num, atol=1e-5)

# --- eq. 11.19: the logistic-regression gradient ----------------------------
print("\n" + "=" * 62)
print("logistic regression gradient: prediction error times input")
print("=" * 62)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def loss(w, xi, yi):
    p = sigmoid(w @ xi)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(yi * np.log(p) + (1 - yi) * np.log(1 - p))


rng = np.random.default_rng(1)
w = rng.normal(size=5)
xi = rng.normal(size=5)
yi = 1.0

p_hat = sigmoid(w @ xi)
analytic = (p_hat - yi) * xi                    # eq. 11.19
numeric = numerical_gradient(lambda v: loss(v, xi, yi), w)

print(f"p_hat = {p_hat:.6f}, y = {yi}")
print(f"analytic (p_hat - y) * x : {np.round(analytic, 6)}")
print(f"numerical                : {np.round(numeric, 6)}")
print(f"max abs difference       : {np.abs(analytic - numeric).max():.2e}")
assert np.allclose(analytic, numeric, atol=1e-6)

# --- the chain rule as multiplication, and why gradients vanish -------------
print("\n" + "=" * 62)
print("the chain rule multiplies — which is why depth is dangerous")
print("=" * 62)
print(f"{'depth':>7} {'sigmoid (x0.25)':>18} {'relu (x1.0)':>14} "
      f"{'slightly >1 (x1.1)':>20}")
for depth in (1, 10, 30, 60):
    print(f"{depth:>7} {0.25**depth:>18.3e} {1.0**depth:>14.3e} "
          f"{1.1**depth:>20.3e}")
print("\nBelow 1 the product vanishes; above 1 it explodes. Keeping the")
print("per-layer factor near 1 is the whole job of initialisation and")
print("normalisation (Part VI).")

# --- eq. 11.7: the Jacobian, and eq. 11.8: chain rule as matmul -------------
def h1(v):
    return np.array([v[0] ** 2, v[0] * v[1], np.sin(v[1])])


def jac_h1(v):
    return np.array([[2 * v[0], 0.0],
                     [v[1], v[0]],
                     [0.0, np.cos(v[1])]])


pt = np.array([1.5, 0.7])
J_analytic = jac_h1(pt)
J_numeric = np.stack([numerical_gradient(lambda v: h1(v)[i], pt) for i in range(3)])
print(f"\nJacobian is {J_analytic.shape} for f: R^2 -> R^3")
assert np.allclose(J_analytic, J_numeric, atol=1e-5)
print("analytic and numerical Jacobians agree")

# eq. 11.8: the Jacobian of a composition is the product of the Jacobians.
A = rng.normal(size=(4, 3))
B = rng.normal(size=(3, 2))
composed = lambda v: A @ (B @ v)
v0 = rng.normal(size=2)
J_comp = np.stack([numerical_gradient(lambda v: composed(v)[i], v0)
                   for i in range(4)])
assert np.allclose(J_comp, A @ B, atol=1e-5)
print(f"J_(f o g) == J_f J_g : {np.allclose(J_comp, A @ B, atol=1e-5)}")
