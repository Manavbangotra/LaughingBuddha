# Extracted from: Chapter 34 — Evaluation Metrics and the Bias–Variance Tradeoff
# Source: src/.../ch034-metrics.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The bias-variance decomposition, measured by resampling (eq. 34.1).
"""
import numpy as np

rng = np.random.default_rng(0)

TRUE_SIGMA = 0.30


def true_f(x):
    return np.sin(1.6 * x) + 0.35 * x


def sample(n):
    x = rng.uniform(-3, 3, n)
    return x, true_f(x) + rng.normal(0, TRUE_SIGMA, n)


def fit_poly(x, y, degree):
    A = np.vander(x, degree + 1)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta


def predict_poly(beta, x):
    return np.vander(x, len(beta)) @ beta


# a fixed grid of test points at which we measure bias and variance
x_test = np.linspace(-2.6, 2.6, 60)
f_test = true_f(x_test)

N_TRAIN, N_REPS = 40, 400

print(f"true noise variance sigma^2 = {TRUE_SIGMA ** 2:.4f}")
print(f"{N_REPS} independent training sets of {N_TRAIN} points each\n")
print(f"{'degree':>7} {'bias^2':>10} {'variance':>10} {'noise':>9} "
      f"{'total':>10} {'measured MSE':>14}")

results = {}
for degree in (0, 1, 2, 3, 5, 9, 15):
    preds = np.empty((N_REPS, len(x_test)))
    for r in range(N_REPS):
        xt, yt = sample(N_TRAIN)
        preds[r] = predict_poly(fit_poly(xt, yt, degree), x_test)

    mean_pred = preds.mean(axis=0)
    bias2 = np.mean((mean_pred - f_test) ** 2)
    var = np.mean(preds.var(axis=0))
    total = bias2 + var + TRUE_SIGMA ** 2

    # measure the same thing directly: fresh noisy targets at the test points
    y_fresh = f_test[None, :] + rng.normal(0, TRUE_SIGMA, preds.shape)
    mse = np.mean((preds - y_fresh) ** 2)

    results[degree] = (bias2, var, total, mse)
    print(f"{degree:>7} {bias2:>10.4f} {var:>10.4f} "
          f"{TRUE_SIGMA ** 2:>9.4f} {total:>10.4f} {mse:>14.4f}")

best = min(results, key=lambda d: results[d][2])
print(f"\nbias^2 + variance + noise reproduces the measured MSE to within")
print(f"sampling error at every degree — eq. 34.1 is an identity, not an")
print(f"analogy. The total is minimised at degree {best}.")
print("\nTwo details worth not glossing over. Bias barely improves from")
print("degree 1 to 2 (0.475 -> 0.467): the target sin(1.6x) + 0.35x is an")
print("ODD function, so an added x^2 term buys almost nothing and the added")
print("x^3 term at degree 3 buys a great deal. Complexity helps only when it")
print("is the right kind. And at degree 15 — 16 parameters for 40 points —")
print("variance is 900x the noise floor and bias^2 has RISEN, because wild")
print("fits distort the average prediction too. Past the point of collapse")
print("the neat monotone story stops holding.")

print("\nthe irreducible floor:")
print(f"  no model can beat MSE = {TRUE_SIGMA ** 2:.4f} on this problem.")
print("  A validation score below the noise floor is leakage, not skill.")

# --- more data moves the variance term, not the bias term -------------------
print("\n" + "=" * 72)
print("what more data does to each term")
print("=" * 72)
print(f"{'N':>6} " + " ".join(f"{'deg ' + str(d) + ' bias2':>14}"
                              for d in (1, 9)) +
      " " + " ".join(f"{'deg ' + str(d) + ' var':>13}" for d in (1, 9)))
for n_train in (20, 40, 100, 400, 2000):
    row = {}
    for degree in (1, 9):
        preds = np.empty((150, len(x_test)))
        for r in range(150):
            xt, yt = sample(n_train)
            preds[r] = predict_poly(fit_poly(xt, yt, degree), x_test)
        row[degree] = (np.mean((preds.mean(0) - f_test) ** 2),
                       np.mean(preds.var(0)))
    print(f"{n_train:>6} " +
          " ".join(f"{row[d][0]:>14.4f}" for d in (1, 9)) + " " +
          " ".join(f"{row[d][1]:>13.4f}" for d in (1, 9)))

print("\nVariance falls roughly as 1/N for both degrees: a hundredfold more")
print("data cuts it about a hundredfold. Degree 1's bias does not move at")
print("all — 0.470 at N=20 and 0.480 at N=2000 — because bias is a property")
print("of the hypothesis space (Chapter 31), not of the sample.")
print("\nDegree 9's bias column needs a caveat: it reads 3.43 at N=20 and")
print("~0 thereafter. That is not bias falling with data. With 10 parameters")
print("and 20 points the fits are so unstable that the AVERAGE prediction is")
print("itself garbage, and measured bias absorbs it. Once there is enough")
print("data to fit the model at all, degree 9's bias is ~0 and stays there.")
print("\nThe usable conclusion is unchanged: more data buys down variance and")
print("never buys down bias, which is why it fixes overfitting and never")
print("fixes underfitting — and why the learning curve below can tell you")
print("which one you have before you spend the money.")

# --- the learning curve, which is the practical form of the above -----------
print("\n" + "=" * 72)
print("learning curves: would more data help?")
print("=" * 72)
x_big, y_big = sample(4000)
x_val, y_val = sample(4000)

for degree, label in ((1, "degree 1 (too rigid)"), (9, "degree 9 (flexible)")):
    print(f"\n{label}")
    print(f"{'N':>6} {'train RMSE':>12} {'val RMSE':>10} {'gap':>8}")
    for n_train in (10, 20, 50, 200, 1000, 4000):
        beta = fit_poly(x_big[:n_train], y_big[:n_train], degree)
        tr = np.sqrt(np.mean((predict_poly(beta, x_big[:n_train])
                              - y_big[:n_train]) ** 2))
        va = np.sqrt(np.mean((predict_poly(beta, x_val) - y_val) ** 2))
        print(f"{n_train:>6} {tr:>12.4f} {va:>10.4f} {va - tr:>8.4f}")

print(f"\nnoise floor RMSE = {TRUE_SIGMA:.4f}")
print("Degree 1 converges quickly to a validation RMSE well ABOVE the noise")
print("floor and the gap closes: high bias, and more data is wasted money.")
print("Degree 9 starts with a large gap that keeps closing towards the floor:")
print("high variance, and more data is exactly what it needs.")
