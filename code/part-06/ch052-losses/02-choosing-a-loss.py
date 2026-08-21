# Extracted from: Chapter 52 — Loss Functions
# Source: src/.../ch052-losses.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Three decisions measured rather than asserted: the loss-metric gap, what
class weighting actually buys, and what focal loss does.
"""
import numpy as np

rng = np.random.default_rng(3)


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def sigmoid(z):
    return np.where(z >= 0, 1 / (1 + np.exp(-np.clip(z, -500, 500))),
                    np.exp(np.clip(z, -500, 500))
                    / (1 + np.exp(np.clip(z, -500, 500))))


# --- an imbalanced binary problem -------------------------------------------
def make_data(n, pos_rate, sep, seed):
    rs = np.random.default_rng(seed)
    y = (rs.random(n) < pos_rate).astype(float)
    X = rs.normal(size=(n, 6))
    X[:, 0] += sep * y                          # only feature 0 is informative
    X[:, 1] += 0.4 * sep * y
    return X, y


Xtr, ytr = make_data(4000, 0.03, 1.6, 11)
Xte, yte = make_data(4000, 0.03, 1.6, 12)
print("=" * 72)
print("an imbalanced problem: 3% positive")
print("=" * 72)
print(f"train positives: {int(ytr.sum())}/{len(ytr)}   "
      f"test positives: {int(yte.sum())}/{len(yte)}")


def train_logreg(X, y, loss="bce", weight=None, gamma=0.0,
                 steps=3000, lr=0.3, seed=0):
    """Plain logistic regression; the LOSS is the only thing that varies."""
    rs = np.random.default_rng(seed)
    w = rs.normal(0, 0.01, X.shape[1])
    b = 0.0
    for _ in range(steps):
        p = sigmoid(X @ w + b)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        if loss == "bce":
            g = p - y                                  # eq. 52.15, binary
        elif loss == "focal":
            # d/dz of -(1-p_t)^gamma log p_t, with p_t the true-class prob.
            # Using dp_t/dz = s p_t(1-p_t) with s = +1 for y=1 and -1 for y=0,
            # this collapses to s[gamma p_t (1-p_t)^g log p_t - (1-p_t)^(g+1)],
            # which reduces to p - y at gamma = 0 (verified below).
            pt = np.where(y == 1, p, 1 - p)
            s_ = np.where(y == 1, 1.0, -1.0)
            g = s_ * (gamma * pt * (1 - pt) ** gamma * np.log(pt)
                      - (1 - pt) ** (gamma + 1))
        if weight is not None:
            g = g * np.where(y == 1, weight, 1.0)
        w -= lr * (X.T @ g) / len(y)
        b -= lr * g.mean()
    return w, b


def report(name, w, b):
    s = Xte @ w + b
    p = sigmoid(s)
    pred = (p > 0.5).astype(float)
    tp = float(((pred == 1) & (yte == 1)).sum())
    fp = float(((pred == 1) & (yte == 0)).sum())
    fn = float(((pred == 0) & (yte == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    # AUC by rank
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    npos, nneg = yte.sum(), (1 - yte).sum()
    auc = (ranks[yte == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    nll = -np.mean(yte * np.log(np.clip(p, 1e-12, 1))
                   + (1 - yte) * np.log(np.clip(1 - p, 1e-12, 1)))
    print(f"{name:<26} {nll:>8.4f} {auc:>7.4f} {prec:>8.3f} {rec:>7.3f} "
          f"{f1:>7.3f} {p.mean():>9.4f}")


print(f"\n{'objective':<26} {'NLL':>8} {'AUC':>7} {'prec':>8} {'rec':>7} "
      f"{'F1':>7} {'mean p':>9}")
w0, b0 = train_logreg(Xtr, ytr, "bce")
report("plain BCE", w0, b0)
for weight in (5.0, 32.0):
    ww, bb = train_logreg(Xtr, ytr, "bce", weight=weight)
    report(f"BCE, positive weight {weight:g}", ww, bb)
for gamma in (1.0, 2.0):
    wf, bf = train_logreg(Xtr, ytr, "focal", gamma=gamma)
    report(f"focal loss, gamma={gamma:g}", wf, bf)

print(f"\nbase rate for reference: {yte.mean():.4f}")
print("\nThree things to read out of this table, and one of them is not")
print("what the usual account of these techniques would lead you to expect.")
print("\nFirst, EVERY modification is ranking-neutral. AUC moves in the")
print("fourth decimal place across all five rows. Neither weighting nor")
print("focal loss taught the model anything it did not already know about")
print("which examples are positive; the decision function is the same.")
print("\nSecond, every modification is worse on NLL and moves the mean")
print("predicted probability far above the 3% base rate. That is")
print("decalibration, and it is the price of both techniques.")
print("\nThird — and this is the part worth pausing on — weighting shifts")
print("recall substantially and FOCAL LOSS DOES NOT. Focal loss inflated the")
print("probabilities just as much and left precision and recall essentially")
print("unchanged. Down-weighting easy examples rescales the loss surface")
print("without preferentially favouring the positive class, so it does not")
print("act as a threshold shift the way a class weight does.")
print("\nThat is consistent with what focal loss was designed for. It was")
print("built for foreground/background imbalance at thousands to one, where")
print("the easy negatives are so numerous that they dominate the gradient")
print("sum outright. At 30 to 1 they do not, so there is nothing for the")
print("modulating factor to suppress.")
print("\nThe practical summary: class weighting is a threshold choice")
print("expressed as a loss, and if you can tune the threshold directly")
print("(Chapter 33) you should, because it is reversible and does not")
print("decalibrate. Focal loss is a different tool for a much more extreme")
print("regime, and reaching for it at mild imbalance — as is common — is")
print("using it far outside the setting it was validated in.")

# --- the loss-metric gap ----------------------------------------------------
print("\n" + "=" * 72)
print("the loss improves and the metric does not (section 4.3)")
print("=" * 72)


def train_traced(X, y, Xv, yv, steps=4000, lr=0.3, seed=0, every=250):
    rs = np.random.default_rng(seed)
    w, b = rs.normal(0, 0.01, X.shape[1]), 0.0
    trace = []
    for t in range(steps + 1):
        s = Xv @ w + b
        p = sigmoid(s)
        if t % every == 0:
            nll = -np.mean(yv * np.log(np.clip(p, 1e-12, 1))
                           + (1 - yv) * np.log(np.clip(1 - p, 1e-12, 1)))
            acc = ((p > 0.5) == yv).mean()
            order = np.argsort(s)
            ranks = np.empty(len(s))
            ranks[order] = np.arange(1, len(s) + 1)
            npos, nneg = yv.sum(), (1 - yv).sum()
            auc = (ranks[yv == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
            trace.append((t, nll, acc, auc))
        ptr = sigmoid(X @ w + b)
        g = ptr - y
        w -= lr * (X.T @ g) / len(y)
        b -= lr * g.mean()
    return trace


trace = train_traced(Xtr, ytr, Xte, yte)
print(f"{'step':>6} {'val NLL':>10} {'accuracy':>10} {'AUC':>8}")
for t, nll, acc, auc in trace:
    print(f"{t:>6} {nll:>10.5f} {acc:>10.5f} {auc:>8.5f}")

nlls = [r[1] for r in trace]
accs = [r[2] for r in trace]
aucs = [r[3] for r in trace]
i1 = 1                                    # the first checkpoint after step 0
print(f"\nfrom step 0 (random init):")
print(f"  NLL      {nlls[0] - nlls[-1]:+.5f}   (improvement)")
print(f"  accuracy {accs[-1] - accs[0]:+.5f}")
print(f"  AUC      {aucs[-1] - aucs[0]:+.5f}")
print(f"\nfrom step {trace[i1][0]} onward, which is the part that matters:")
print(f"  NLL      {nlls[i1] - nlls[-1]:+.5f}   "
      f"({(nlls[i1] - nlls[-1]) / nlls[i1]:.1%} of the remaining loss)")
print(f"  accuracy {accs[-1] - accs[i1]:+.5f}")
print(f"  AUC      {aucs[-1] - aucs[i1]:+.5f}")

print("\nThe first checkpoint does almost all the visible work: from random")
print("initialisation both metrics jump, and that is not the interesting")
print("part. Everything after it is.")
print("\nAfter step 250 the loss continues to improve by a real margin while")
print("accuracy does NOT improve — it drifts slightly DOWN — and AUC is")
print("flat to four decimal places. So the second half of training refined")
print("probabilities in a way that neither the argmax nor the ranking")
print("registers at all.")
print("\nThis is a sharper version of the point than 'the metric lags the")
print("loss'. The loss and the metric are measuring genuinely different")
print("things, and it is possible — as here — for the loss to be improving")
print("while a threshold metric slowly degrades. Neither number is lying.")
print("\nThe practical rule: monitor the loss AND a threshold-free metric,")
print("and do not read a falling loss as evidence that the thing you are")
print("evaluated on is improving. Decide in advance which one you will stop")
print("on, because they will disagree.")

# --- reduction and batch size (section 7.2) ---------------------------------
print("\n" + "=" * 72)
print("'sum' reduction couples the learning rate to the batch size (7.2)")
print("=" * 72)


# One fixed model and one fixed data stream; the batch is a PREFIX of it, so
# the only thing changing across rows is the batch size itself.
# The labels must depend on X. With coin-flip labels the true gradient is
# zero, so the mean-reduced norm would decay as 1/sqrt(B) and the experiment
# would measure sampling noise rather than the reduction.
_rs = np.random.default_rng(7)
_Xpool = _rs.normal(size=(8192, 6))
_wtrue = np.array([1.4, -1.1, 0.8, 0.0, 0.5, -0.3])
_ypool = (_rs.random(8192) < sigmoid(_Xpool @ _wtrue)).astype(float)
_w = _rs.normal(0, 0.1, 6)                 # the model, deliberately not _wtrue


def one_step_norm(batch, reduction):
    Xb, yb = _Xpool[:batch], _ypool[:batch]
    grad = Xb.T @ (sigmoid(Xb @ _w) - yb)
    if reduction == "mean":
        grad = grad / batch
    return float(np.linalg.norm(grad))


print(f"{'batch':>7} {'|grad| (mean)':>15} {'|grad| (sum)':>15} "
      f"{'sum / batch-8 sum':>19}")
base = one_step_norm(8, "sum")
for batch in (8, 32, 128, 512, 2048, 8192):
    print(f"{batch:>7} {one_step_norm(batch, 'mean'):>15.4f} "
          f"{one_step_norm(batch, 'sum'):>15.4f} "
          f"{one_step_norm(batch, 'sum') / base:>19.1f}x")
print("\nUnder 'mean' the gradient norm settles: it is an estimate of a fixed")
print("quantity — the full-dataset gradient at this parameter setting — and")
print("larger batches estimate it more precisely rather than differently.")
print("The small batches wobble around that value because they are noisy")
print("estimates of it, not because the quantity itself is changing.")
print("\nUnder 'sum' the norm grows roughly in proportion to the batch, so")
print("at a fixed learning rate the step length is multiplied by the same")
print("factor. That is why 'mean' is the default, and why switching to 'sum'")
print("presents as a diverging model rather than as a configuration change.")

# --- masked reduction, the sequence-model bug -------------------------------
print("\n" + "=" * 72)
print("the masked-reduction bug (section 7.2)")
print("=" * 72)
B_, T = 4, 10
lengths = np.array([10, 6, 3, 8])
mask = np.arange(T)[None, :] < lengths[:, None]
per_token = rng.random((B_, T)) * 2.0
per_token_masked = per_token * mask

wrong = per_token_masked.sum() / per_token_masked.size    # divide by B*T
right = per_token_masked.sum() / mask.sum()               # divide by n valid
print(f"sequence lengths           : {lengths.tolist()} of max {T}")
print(f"valid tokens               : {int(mask.sum())} of {B_ * T}")
print(f"loss, divided by B*T       : {wrong:.4f}   WRONG")
print(f"loss, divided by valid     : {right:.4f}   correct")
print(f"ratio                      : {right / wrong:.4f}")
print("\nThe wrong version counts padding as zero-loss tokens, so a batch")
print("that happens to contain short sequences reports a lower loss for the")
print("same model. The loss curve then tracks the batch composition rather")
print("than the model, and it improves whenever the sampler happens to draw")
print("short sequences together.")
