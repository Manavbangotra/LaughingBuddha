# Extracted from: Chapter 87 — Distillation and Model Specialization
# Source: src/.../ch087-distillation.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Does a distilled student beat an identically-sized model trained on labels?"""
import numpy as np

rng = np.random.default_rng(0)

N_CLASSES, D_IN, N_TRAIN, N_TEST = 8, 24, 300, 4000


def softmax(z, T=1.0):
    s = z / T
    s = s - s.max(-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(-1, keepdims=True)


# Ground truth with real class STRUCTURE: two families of four classes, so the
# teacher's confusions carry information a one-hot label cannot.
true_W = rng.normal(size=(D_IN, N_CLASSES))
family = np.array([0, 0, 0, 0, 1, 1, 1, 1])
true_W += np.outer(rng.normal(size=D_IN), family) * 1.5


def make_data(n):
    X = rng.normal(size=(n, D_IN))
    logits = X @ true_W
    y = np.array([rng.choice(N_CLASSES, p=row) for row in softmax(logits)])
    return X, y, logits


def train_to_convergence(X, y, epochs=2000, lr=0.5):
    W = np.zeros((X.shape[1], N_CLASSES))
    onehot = np.eye(N_CLASSES)[y]
    for _ in range(epochs):
        W -= lr * X.T @ (softmax(X @ W) - onehot) / len(X)
    return W


# --- the teacher: plenty of data and full access to the features ------------
Xbig, ybig, _ = make_data(20_000)
teacher_W = train_to_convergence(Xbig, ybig)
Xtr, ytr, _ = make_data(N_TRAIN)
Xte, yte, _ = make_data(N_TEST)
teacher_acc = float((softmax(Xte @ teacher_W).argmax(1) == yte).mean())
print(f"teacher (20,000 examples, full features): {teacher_acc:.4f}")
print(f"chance level with {N_CLASSES} classes    : {1 / N_CLASSES:.4f}\n")

# The student is CAPACITY LIMITED — it sees a compressed view of the input and
# has only N_TRAIN examples. This is the regime distillation is for.
proj = rng.normal(size=(D_IN, 14))
Str, Ste = Xtr @ proj, Xte @ proj
teacher_logits_tr = Xtr @ teacher_W


def train_student(S, y, teacher_logits=None, T=1.0, alpha=1.0,
                  epochs=4000, lr=0.5):
    """Every configuration gets the same optimisation budget, and the soft
    gradient is divided by T so that temperature changes the TARGET rather
    than the effective step size — otherwise the comparison across T would be
    a learning-rate sweep in disguise."""
    W = np.zeros((S.shape[1], N_CLASSES))
    onehot = np.eye(N_CLASSES)[y]
    pt = softmax(teacher_logits, T) if teacher_logits is not None else None
    for _ in range(epochs):
        grad = np.zeros_like(W)
        if alpha < 1.0:
            grad += (1 - alpha) * S.T @ (softmax(S @ W) - onehot) / len(S)
        if pt is not None and alpha > 0.0:
            grad += alpha * S.T @ (softmax(S @ W, T) - pt) / len(S)
        W -= lr * grad
    return W


def accuracy(W):
    return float((softmax(Ste @ W).argmax(1) == yte).mean())


hard = accuracy(train_student(Str, ytr, alpha=0.0))
print(f"{'student trained on':<32} {'test accuracy':>14} {'vs hard labels':>16}")
print(f"{'hard labels only':<32} {hard:>14.4f} {'-':>16}")

results = {}
for T in (1.0, 2.0, 3.0, 4.0, 6.0):
    acc = accuracy(train_student(Str, ytr, teacher_logits_tr, T=T, alpha=1.0))
    results[T] = acc
    print(f"{'soft targets, T=' + f'{T:.0f}':<32} {acc:>14.4f} "
          f"{acc - hard:>+16.4f}")

for T in (2.0, 4.0):
    acc = accuracy(train_student(Str, ytr, teacher_logits_tr, T=T, alpha=0.7))
    print(f"{'mixed, T=' + f'{T:.0f}' + ', alpha=0.7':<32} {acc:>14.4f} "
          f"{acc - hard:>+16.4f}")

best_T = max(results, key=results.get)
print(f"\nbest temperature: T={best_T:.0f} ({results[best_T]:.4f}), "
      f"{results[best_T] - hard:+.4f} over hard labels")
assert results[best_T] > hard, \
    "distillation should beat hard-label training at equal student capacity"
assert best_T > 1.0, "the benefit should come from RAISING the temperature"

print("""
Same student capacity, same 300 training examples, same optimiser and the same
number of steps. The only thing that differs is what the student was asked to
match — and matching the teacher's distribution beats matching the labels by a
wide margin.

Note that T=1 already helps a little and higher T helps much more, peaking
around T=4 and declining slightly after. That shape is the whole argument of
section 6.2: the useful signal is in the SMALL probabilities, T=1 leaves them
negligible, and too much T flattens the distribution until the target carries
no information about the correct class either.""")
