# -*- coding: utf-8 -*-
# Extracted from: Chapter 132 — QLoRA, PEFT, and Adapters
# Source: src/.../ch132-qlora-peft.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The PEFT family is not one axis. It is two, and only one is the parameter count.

Every parameter-efficient method answers "where do the trainable parameters live",
and the usual comparison is a table of parameter counts at matched quality. That
comparison hides the thing that actually decides between them: methods differ in
WHAT KIND OF CHANGE they can express, and a method that cannot express your
change does not improve with budget (eq:peft-expressiveness).

Three families, on one linear layer:

  LoRA (parallel)    y = x (W0 + BA)          -- adds to the MAP
  adapter (serial)   y = (x W0)(I + B'A')     -- transforms the OUTPUT
  bias / prompt      y = x W0 + b             -- adds an OFFSET

This listing puts them on tasks of different kinds at matched budget, then asks a
second question the parameter table cannot see: when the two low-rank families
CAN express the same change, does it cost them the same to do it?
"""
import numpy as np

rng = np.random.default_rng(157)
D, N = 64, 6000


def conditioned_W0(cond):
    """A base map with a chosen condition number."""
    U, _ = np.linalg.qr(rng.normal(size=(D, D)))
    V, _ = np.linalg.qr(rng.normal(size=(D, D)))
    s = np.logspace(0, -np.log10(cond), D)
    return U @ np.diag(s) @ V.T


def rel(P, Y):
    return float(np.linalg.norm(P - Y) / np.linalg.norm(Y))


def fit_lora(W0, X, Y, r, steps=900, lr=0.05):
    A = rng.normal(size=(D, r)) / np.sqrt(D); B = np.zeros((r, D))
    for _ in range(steps):
        G = 2.0 * (X @ (W0 + A @ B) - Y) / len(X); GD = X.T @ G
        gA, gB = GD @ B.T, A.T @ GD
        A -= lr * gA; B -= lr * gB
    return X @ (W0 + A @ B)


def fit_adapter(W0, X, Y, r, steps=900, lr=0.05):
    """Serial: the frozen layer runs first, then a low-rank correction."""
    H = X @ W0
    A = rng.normal(size=(D, r)) / np.sqrt(D); B = np.zeros((r, D))
    for _ in range(steps):
        G = 2.0 * (H + H @ (A @ B) - Y) / len(X); GD = H.T @ G
        gA, gB = GD @ B.T, A.T @ GD
        A -= lr * gA; B -= lr * gB
    return H + H @ (A @ B)


def fit_bias(W0, X, Y):
    """The whole family that acts on activations rather than weights: it can
    shift the output and nothing else."""
    return X @ W0 + (Y - X @ W0).mean(axis=0)


W0 = conditioned_W0(3.0)
X = rng.normal(size=(N, D))

Aq = rng.normal(size=(D, 4)) / np.sqrt(D); Bq = rng.normal(size=(4, D)) / 2
TASKS = {
    "map change (rank 4)": X @ (W0 + Aq @ Bq),
    "output offset":       X @ W0 + rng.normal(size=D) * 0.5,
    "both":                X @ (W0 + Aq @ Bq) + rng.normal(size=D) * 0.5,
}

print(f"D={D}. Matched budget: rank 4 for both low-rank methods "
      f"({2*D*4:,} params), bias has {D} params.\n")
print(f"{'task':>22}{'LoRA r=4':>11}{'adapter r=4':>14}{'bias only':>12}")
print("-" * 59)
for name, Y in TASKS.items():
    print(f"{name:>22}{rel(fit_lora(W0, X, Y, 4), Y):>11.4f}"
          f"{rel(fit_adapter(W0, X, Y, 4), Y):>14.4f}"
          f"{rel(fit_bias(W0, X, Y), Y):>12.4f}")

print("\n\nDoes raising the budget rescue a method that cannot express the "
      "change?\n")
print(f"{'task':>22}{'LoRA r=1':>11}{'r=4':>9}{'r=16':>9}{'r=32':>9}")
print("-" * 60)
for name in ("map change (rank 4)", "output offset"):
    Y = TASKS[name]
    print(f"{name:>22}" + "".join(f"{rel(fit_lora(W0, X, Y, r), Y):>{w}.4f}"
                                  for r, w in ((1, 11), (4, 9), (16, 9),
                                               (32, 9))))

print("\n\nWhen BOTH low-rank families can express the change, what does it "
      "cost each?\n")
print(f"{'cond(W0)':>10}{'LoRA update':>14}{'adapter update':>17}"
      f"{'ratio':>9}")
print(f"{'':>10}{'norm needed':>14}{'norm needed':>17}{'':>9}")
print("-" * 50)
ratios = {}
for cond in (1.0, 10.0, 100.0, 1000.0):
    Wc = conditioned_W0(cond)
    delta = Aq @ Bq                       # the same additive change every time
    serial = np.linalg.pinv(Wc) @ delta   # what a serial adapter must represent
    nl, na = np.linalg.norm(delta), np.linalg.norm(serial)
    ratios[cond] = na / nl
    print(f"{cond:>10.0f}{nl:>14.3f}{na:>17.3f}{na/nl:>9.1f}x")

print(f"""
The first table is the argument. At an identical budget the three methods are not
three points on a quality curve -- they are three different hypothesis classes,
and each is near-exact on the change it can express and useless on the one it
cannot.

LoRA and the serial adapter both handle a change to the MAP, and both fail on a
pure output offset, because x(W0+BA) is linear in x and cannot produce a constant
term. The bias-only method is the mirror image: it nails the offset with {D}
parameters and cannot touch the map at any budget.

The second table makes the failure mode unmistakable. On the map-change task,
LoRA improves with rank exactly as ch:ft-lora predicts. On the offset task it
does not improve at all -- rank 32 is no better than rank 1, because the missing
capability is not capacity. It is a term the parameterisation does not have
(eq:peft-expressiveness).

That is the practical lesson, and it inverts the usual selection procedure.
"Which PEFT method gives the best quality per parameter" is answerable only after
"what kind of change does my task need", and the second question is rarely asked.
Prompt and prefix tuning act on activations, so they steer a model through its
existing behaviour; LoRA and adapters act on the map, so they can install
behaviour that was not reachable. A task that needs the second will look like a
data problem or a capacity problem under the first, and no amount of tuning fixes
it.

Now the third table, which is the subtler result. When both low-rank families CAN
express the same change, they are not equivalent, and the difference is the base
layer's conditioning.

A serial adapter has to represent W0^-1 * delta rather than delta, so its required
update grows with the condition number: {ratios[1.0]:.1f}x at cond 1 rising to
{ratios[1000.0]:.1f}x at cond 1000. The change is the same change. The cost of
expressing it from the output side is not.

Two consequences follow, and both are measured elsewhere in this part. The larger
update is harder to optimise, which shows up as a serial adapter needing more
steps and a smaller learning rate on exactly the tasks where LoRA is
well-behaved. And by ch:ft-lora's eq:forgetting-quadratic, a larger update
disturbs the base model more -- so the serial form forgets more to achieve the
same adaptation, on ill-conditioned layers.

Real transformer weight matrices are not well conditioned. That, rather than the
latency argument usually given, is the deeper reason the parallel form won.""")
