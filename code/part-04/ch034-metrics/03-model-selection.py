# Extracted from: Chapter 34 — Evaluation Metrics and the Bias–Variance Tradeoff
# Source: src/.../ch034-metrics.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Model selection that does not lie: nested CV, one-SE rule, and a
measurement of the optimism from evaluating too many candidates.
"""
import numpy as np

rng = np.random.default_rng(21)


def make_data(n):
    X = rng.normal(size=(n, 12))
    z = 1.4 * X[:, 0] - 1.1 * X[:, 1] + 0.8 * X[:, 2] * X[:, 3] - 0.4
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(float)
    return X, y


def sigmoid(z):
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1 / (1 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1 + e)
    return out


def fit_logistic(X, y, lam, n_iter=60):
    """Newton with an L2 penalty; ridge-stabilised so it never fails."""
    A = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(A.shape[1])
    for _ in range(n_iter):
        p = sigmoid(A @ w)
        g = A.T @ (p - y) / len(y)
        g[1:] += 2 * lam * w[1:]
        S = np.maximum(p * (1 - p), 1e-9)
        H = (A * S[:, None]).T @ A / len(y) + (2 * lam + 1e-8) * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    return w


def score_logistic(w, X, y):
    p = sigmoid(np.column_stack([np.ones(len(X)), X]) @ w)
    return float(np.mean((p >= 0.5) == (y == 1)))


def kfold(n, k, seed):
    idx = np.random.default_rng(seed).permutation(n)
    return [(np.concatenate([idx[:i * n // k], idx[(i + 1) * n // k:]]),
             idx[i * n // k:(i + 1) * n // k]) for i in range(k)]


GRID = np.logspace(-4, 1, 12)      # log grid, per section 5.6 of Chapter 32

# --- 1. plain CV, and the optimism it hides ---------------------------------
X, y = make_data(600)
print("=" * 72)
print("1. cross-validation on a grid of 12 lambdas")
print("=" * 72)
folds = kfold(len(y), 5, 1)
means, ses = [], []
for lam in GRID:
    accs = [score_logistic(fit_logistic(X[tr], y[tr], lam), X[va], y[va])
            for tr, va in folds]
    means.append(np.mean(accs))
    ses.append(np.std(accs, ddof=1) / np.sqrt(len(accs)))
means, ses = np.array(means), np.array(ses)

best_i = int(means.argmax())
print(f"{'lambda':>10} {'CV accuracy':>13} {'SE':>8}")
for i, lam in enumerate(GRID):
    mark = "  <-- best" if i == best_i else ""
    print(f"{lam:>10.5f} {means[i]:>13.4f} {ses[i]:>8.4f}{mark}")

# the one-standard-error rule: simplest model within 1 SE of the best
threshold = means[best_i] - ses[best_i]
one_se_i = int(np.max(np.where(means >= threshold)[0]))   # largest lambda
print(f"\nbest lambda      : {GRID[best_i]:.5f}  (CV {means[best_i]:.4f})")
print(f"one-SE rule picks: {GRID[one_se_i]:.5f}  (CV {means[one_se_i]:.4f})")
print("The one-SE rule takes the strongest regularisation whose score is")
print("statistically indistinguishable from the best, on the grounds that")
print("the argmax of a noisy grid is itself a noisy quantity.")

# --- 2. the CV score of the SELECTED model is biased ------------------------
print("\n" + "=" * 72)
print("2. how optimistic is that CV score? (eq. 34.11)")
print("=" * 72)
Xh, yh = make_data(20000)          # a large fresh sample as ground truth
true_best = score_logistic(fit_logistic(X, y, GRID[best_i]), Xh, yh)
true_1se = score_logistic(fit_logistic(X, y, GRID[one_se_i]), Xh, yh)
print(f"CV score of the chosen model     : {means[best_i]:.4f}")
print(f"its accuracy on 20,000 fresh rows: {true_best:.4f}")
print(f"optimism                         : {means[best_i] - true_best:+.4f}")
print(f"\none-SE choice, fresh accuracy    : {true_1se:.4f}")

# --- 3. optimism grows with the number of candidates ------------------------
print("\n" + "=" * 72)
print("3. optimism grows with how many models you try")
print("=" * 72)
print("Candidates are RANDOM SEEDS for the same model on the same data, so")
print("every genuine difference between them is exactly zero. Any apparent")
print("winner is noise by construction.")
print("\nOne run of this experiment says nothing — the whole point is that a")
print("single maximum is a lucky draw. So it is repeated on 25 independent")
print("datasets and the optimism averaged.\n")

N_REPS_SEL, K_MAX = 25, 64
Xh2, yh2 = make_data(20000)
ks = (1, 2, 4, 8, 16, 32, 64)
opt_by_k = {k: [] for k in ks}
se_samples = []

for rep in range(N_REPS_SEL):
    X2, y2 = make_data(300)
    folds2 = kfold(len(y2), 5, 5000 + rep)
    scored = []
    for c in range(K_MAX):
        # identical model, different bootstrap of the training folds: the
        # candidates are interchangeable, so all differences are sampling noise
        g = np.random.default_rng(90000 + 1000 * rep + c)
        accs = []
        for tr, va in folds2:
            boot = g.choice(len(tr), len(tr), replace=True)
            w = fit_logistic(X2[tr][boot], y2[tr][boot], 0.01)
            accs.append(score_logistic(w, X2[va], y2[va]))
        scored.append((float(np.mean(accs)), boot, w))
        se_samples.append(np.std(accs, ddof=1) / np.sqrt(5))
    for k in ks:
        cv_best, _, w_best = max(scored[:k], key=lambda t: t[0])
        opt_by_k[k].append(cv_best - score_logistic(w_best, Xh2, yh2))

se_typical = float(np.mean(se_samples))
print(f"typical fold-to-fold SE of one candidate: {se_typical:.4f}\n")
base = float(np.mean(opt_by_k[1]))
print(f"{'candidates':>11} {'mean optimism':>15} {'rise over k=1':>15} "
      f"{'SE x sqrt(2 log k)':>20}")
for k in ks:
    predicted = se_typical * np.sqrt(2 * np.log(k)) if k > 1 else 0.0
    print(f"{k:>11} {np.mean(opt_by_k[k]):>15.4f} "
          f"{np.mean(opt_by_k[k]) - base:>15.4f} {predicted:>20.4f}")

print("\nRead the third column, not the second. The absolute level starts")
print("negative because each candidate is fitted on a bootstrap of its")
print("training folds and is therefore slightly worse than a full fit — a")
print("constant offset that says nothing about selection. What matters is")
print("that the winner's optimism RISES monotonically with the number of")
print("candidates it beat, by 0.038 accuracy points from k=1 to k=64, while")
print("not one of these models is genuinely better than another.")
print("\nThe last column over-predicts by about a factor of two, and the")
print("reason is instructive: eq. 34.11 assumes k INDEPENDENT candidates.")
print("These share a dataset and a model, so their scores are strongly")
print("correlated and the effective number of independent draws is far")
print("below 64. The formula is an upper bound in practice, and the")
print("qualitative claim — sqrt(log k) growth, never zero — survives.")
print("\nThis is why the test set is touched once, and why 'we tried 300")
print("configurations' should make you trust a reported improvement LESS.")

# --- 4. nested CV: the honest estimate of the whole procedure ---------------
print("\n" + "=" * 72)
print("4. nested cross-validation")
print("=" * 72)
print("The selection space is now realistic: 6 lambdas x 8 feature subsets")
print("= 48 candidates, on 300 rows. Every subset keeps the four informative")
print("features and adds three of the eight noise features, so the")
print("candidates are genuinely interchangeable — as in section 3, any")
print("winner is a winner by luck. Section 3 says a search this wide will")
print("produce a visibly optimistic score.\n")

Xs, ys = make_data(300)
LAM_GRID = np.logspace(-3, 0, 6)
SUBSETS = [np.concatenate([[0, 1, 2, 3],
                           np.random.default_rng(7000 + s).choice(
                               np.arange(4, 12), 3, replace=False)])
           for s in range(8)]
CANDIDATES = [(lam, cols) for lam in LAM_GRID for cols in SUBSETS]


def select(Xsel, ysel, k_inner, seed):
    """Run the search. Returns the winner, its CV score, and the per-fold
    models that produced that score — keeping them lets us re-score exactly
    those fits on fresh data, with no difference in training-set size to
    confound the comparison."""
    inner = kfold(len(ysel), k_inner, seed)
    best, best_score, best_models = None, -np.inf, None
    for lam, cols in CANDIDATES:
        models = [fit_logistic(Xsel[itr][:, cols], ysel[itr], lam)
                  for itr, _ in inner]
        a = [score_logistic(m, Xsel[iva][:, cols], ysel[iva])
             for m, (_, iva) in zip(models, inner)]
        if np.mean(a) > best_score:
            best, best_score, best_models = (lam, cols), float(np.mean(a)), models
    return best, best_score, best_models


def nested_cv(Xn, yn, k_outer, seed):
    """Inner loop selects, outer loop scores what the inner loop chose."""
    scores = []
    for oi, (tr, te) in enumerate(kfold(len(yn), k_outer, seed)):
        (lam_i, cols_i), _, _ = select(Xn[tr], yn[tr], 4, seed + 1 + oi)
        scores.append(score_logistic(
            fit_logistic(Xn[tr][:, cols_i], yn[tr], lam_i),
            Xn[te][:, cols_i], yn[te]))
    return float(np.mean(scores))


# All three estimators, on the same 12 independent datasets, so they are
# directly comparable. The honest figure re-scores the winner's OWN per-fold
# fits on 20,000 fresh rows — same models, same training-set size, so the only
# difference is whether the scoring data was also used to choose them.
naive_scores, honest_scores, nested_scores = [], [], []
for rep in range(12):
    Xr_, yr_ = make_data(300)
    (lam_r, cols_r), inner_r, models_r = select(Xr_, yr_, 5, 6000 + rep)
    naive_scores.append(inner_r)
    honest_scores.append(np.mean([score_logistic(m, Xh[:, cols_r], yh)
                                  for m in models_r]))
    nested_scores.append(nested_cv(Xr_, yr_, 5, 8000 + 20 * rep))
    print(f"  run {rep + 1:>2}: naive {inner_r:.4f}   "
          f"nested {nested_scores[-1]:.4f}   honest {honest_scores[-1]:.4f}")


def se(v):
    return float(np.std(v, ddof=1) / np.sqrt(len(v)))


naive_bias = np.array(naive_scores) - np.array(honest_scores)
nested_bias = np.array(nested_scores) - np.array(honest_scores)

print(f"\naveraged over 12 independent runs of the 48-candidate search:")
print(f"{'estimator':<42} {'value':>8} {'bias':>9} {'SE of bias':>11}")
print(f"{'naive: winner CV score (what gets quoted)':<42} "
      f"{np.mean(naive_scores):>8.4f} {np.mean(naive_bias):>+9.4f} "
      f"{se(naive_bias):>11.4f}")
print(f"{'nested CV':<42} "
      f"{np.mean(nested_scores):>8.4f} {np.mean(nested_bias):>+9.4f} "
      f"{se(nested_bias):>11.4f}")
print(f"{'honest: same fits on 20,000 fresh rows':<42} "
      f"{np.mean(honest_scores):>8.4f} {0.0:>+9.4f} {'-':>11}")

print("\nRead this carefully, because it does not say what the slogan says.")
print("\nThe naive score is biased UPWARD, as predicted: same models, same")
print("training-set size, same everything — the only difference is that the")
print("data used to score them also chose them.")
print("\nNested CV is biased DOWNWARD, and by more. That is not a bug and it")
print("is well known: each outer fold selects and fits on 4/5 of the data,")
print("so nested CV estimates the quality of the procedure applied to a")
print("SMALLER dataset than the one you will actually train on. It is a")
print("conservative estimator, not an unbiased one — and 'conservative' is")
print("usually the direction you want to be wrong in.")
print("\nBoth biases here are a fraction of a point, and comparable to their")
print("own standard errors, because the 48 candidates were deliberately")
print("interchangeable and 300 rows is not tiny. With a genuinely wide")
print("search over scarce data the naive bias grows (section 3) while the")
print("nested penalty does not, which is when nested CV earns its 20x cost.")
print("\nThe durable lesson is not 'always use nested CV'. It is that a score")
print("computed on data that participated in choosing the model is not a")
print("measurement, and that every honest alternative costs something —")
print("compute, data, or a conservative bias. Pick which one you can afford.")
