# Extracted from: Chapter 57 — Normalization: Batch, Layer, and RMSNorm
# Source: src/.../ch057-normalization.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The property that distinguishes batch normalisation from everything else:
its output depends on the other examples in the batch.
"""
import numpy as np

rng = np.random.default_rng(1)


def bn_train(x, gamma, beta, eps=1e-5):
    mu, var = x.mean(axis=0), x.var(axis=0)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta


def bn_eval(x, gamma, beta, mu, var, eps=1e-5):
    return gamma * (x - mu) / np.sqrt(var + eps) + beta


def ln(x, gamma, beta, eps=1e-5):
    mu = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta


D = 16
gamma, beta = np.ones(D), np.zeros(D)

# --- the same example, different batches ------------------------------------
print("=" * 72)
print("one example, many batches: how much does the output move?")
print("=" * 72)
probe = rng.normal(size=(1, D))
pool = rng.normal(size=(20000, D))

print(f"{'batch size':>11} {'BN output sd':>14} {'LN output sd':>14} "
      f"{'BN noise / signal':>19}")
for B in (2, 4, 8, 32, 128, 512):
    outs_bn, outs_ln = [], []
    for _ in range(400):
        others = pool[rng.integers(0, len(pool), B - 1)]
        batch = np.vstack([probe, others])
        outs_bn.append(bn_train(batch, gamma, beta)[0])
        outs_ln.append(ln(batch, gamma, beta)[0])
    bn_sd = float(np.mean(np.std(outs_bn, axis=0)))
    ln_sd = float(np.mean(np.std(outs_ln, axis=0)))
    bn_mean = float(np.mean(np.abs(np.mean(outs_bn, axis=0))))
    print(f"{B:>11} {bn_sd:>14.5f} {ln_sd:>14.5f} "
          f"{bn_sd / max(bn_mean, 1e-12):>19.4f}")

print("\nThe SAME input produces a different output every time, depending")
print("on which other examples happened to share its batch. Layer norm's")
print("output does not move at all, because it never looks at them.")
print("\nThat variability is noise injection, and it is the regularisation")
print("effect of section 6.4. It shrinks as the batch grows — roughly as")
print("1/sqrt(B), which the column follows — so a large-batch run gets less")
print("regularisation from its batch norm than a small-batch one does.")
print("\nThis is why swapping BatchNorm for a batch-independent")
print("normalisation can COST accuracy for reasons that have nothing to do")
print("with optimisation: you removed a regulariser you did not know you")
print("were relying on.")

# --- the train/eval gap -----------------------------------------------------
print("\n" + "=" * 72)
print("the train/eval divergence (section 5.1 warning)")
print("=" * 72)
mu_run = pool.mean(axis=0)
var_run = pool.var(axis=0)
print(f"{'batch size':>11} {'mean |train - eval| output':>28} "
      f"{'relative to output scale':>26}")
for B in (1, 2, 8, 32, 256):
    diffs = []
    for _ in range(200):
        batch = pool[rng.integers(0, len(pool), B)]
        t = bn_train(batch, gamma, beta)
        e = bn_eval(batch, gamma, beta, mu_run, var_run)
        diffs.append(np.abs(t - e).mean())
    scale = np.abs(bn_eval(pool[:1000], gamma, beta, mu_run,
                           var_run)).mean()
    print(f"{B:>11} {float(np.mean(diffs)):>28.5f} "
          f"{float(np.mean(diffs)) / scale:>26.4f}")

print("\nAt batch size 1 the training-mode output is ZERO for every feature")
print("— one example has zero variance about its own mean — while the")
print("eval-mode output is whatever the running statistics say. The two")
print("modes compute completely different things.")
print("\nThe gap closes as the batch grows and never reaches zero, because")
print("a finite batch's statistics are an estimate. This is a permanent")
print("property of the design, not a bug to be fixed.")

# --- what a distribution shift does to the running statistics ---------------
print("\n" + "=" * 72)
print("running statistics assume the inference distribution matches (7.2)")
print("=" * 72)
print(f"{'shift (sd)':>11} {'output mean':>14} {'output sd':>12} "
      f"{'target: 0 and 1':>17}")
for shift in (0.0, 0.5, 1.0, 3.0):
    xs = pool[:2000] + shift
    out = bn_eval(xs, gamma, beta, mu_run, var_run)
    print(f"{shift:>11.1f} {out.mean():>14.4f} {out.std():>12.4f} "
          f"{'':>17}")

print("\nThe layer's whole purpose is to hand the next layer something with")
print("mean 0 and standard deviation 1. Under a distribution shift it hands")
print("over something else, and every layer above was trained assuming it")
print("would not.")
print("\nThat makes batch normalisation an amplifier of the covariate shift")
print("of Chapter 48 rather than a defence against it, which is worth")
print("noting given the technique's original name. Layer norm has no such")
print("exposure: it recomputes from the input it is actually given.")

# --- gradient accumulation is NOT exact with batch norm ---------------------
print("\n" + "=" * 72)
print("gradient accumulation is NOT exact with batch norm (section 7.4)")
print("=" * 72)
print("Chapter 53 measured accumulation reproducing a full batch to 1e-16.")
print("Here is the same test with a batch-norm layer in the way.\n")
Xb = pool[:256]
full = bn_train(Xb, gamma, beta)
print(f"{'micro-batch':>12} {'max |accumulated - full|':>26} "
      f"{'relative':>11}")
for micro in (256, 64, 32, 8):
    parts = [bn_train(Xb[i:i + micro], gamma, beta)
             for i in range(0, 256, micro)]
    acc = np.vstack(parts)
    print(f"{micro:>12} {np.abs(acc - full).max():>26.4e} "
          f"{np.abs(acc - full).max() / np.abs(full).max():>11.4f}")

print("\nThe difference is not floating-point; it is a different function.")
print("Each micro-batch normalises by its own statistics, so an example's")
print("output depends on which micro-batch it landed in. Chapter 53's")
print("exactness result required the loss to be a mean over INDEPENDENT")
print("examples, and batch normalisation is precisely the construction that")
print("breaks that independence.")
