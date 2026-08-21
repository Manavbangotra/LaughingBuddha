# Extracted from: Chapter 20 — Debugging, Logging, Testing, Async, and Performance
# Source: src/.../ch020-engineering.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Logging, testing, profiling, and Amdahl's law — measured.
"""
import cProfile
import io
import logging
import pstats
import random
import time

import numpy as np

# --- logging: levels, lazy formatting, and the hierarchy --------------------
print("=" * 66)
print("logging")
print("=" * 66)

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)-8s %(name)s | %(message)s",
                    force=True)
logger = logging.getLogger("demo.pipeline")

logger.debug("this is suppressed at INFO level")
logger.info("epoch %d complete, loss=%.4f", 3, 0.2143)
logger.warning("retrying after %s", "connection reset")

# Lazy formatting is not a style preference; it is a cost.
class Expensive:
    def __str__(self):
        time.sleep(0.001)          # pretend this is costly to render
        return "expensive-value"


t0 = time.perf_counter()
for _ in range(200):
    logger.debug("value is %s", Expensive())     # __str__ never called
lazy = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(200):
    logger.debug(f"value is {Expensive()}")      # __str__ ALWAYS called
eager = time.perf_counter() - t0

print(f"\n200 suppressed DEBUG calls:")
print(f"  lazy  'msg %s', arg : {lazy*1000:>7.1f} ms")
print(f"  eager f'msg {{arg}}'  : {eager*1000:>7.1f} ms   <- {eager/lazy:.0f}x")
print("The f-string formats the message even though nothing is emitted.")

# --- a minimal property-based tester ----------------------------------------
print("\n" + "=" * 66)
print("property-based testing, from scratch")
print("=" * 66)


def standardise(values):
    arr = np.asarray(values, dtype=float)
    std = arr.std()
    return (arr - arr.mean()) / (std if std else 1.0)


def check_property(prop, generator, trials=400, seed=0):
    """Search for a counterexample, then shrink it to something minimal."""
    rng = random.Random(seed)
    for _ in range(trials):
        case = generator(rng)
        try:
            if prop(case):
                continue
        except Exception:
            pass
        # Found a failure. Shrink by repeatedly trying smaller inputs.
        shrunk = case
        improved = True
        while improved:
            improved = False
            for candidate in ([shrunk[:len(shrunk)//2], shrunk[1:],
                               shrunk[:-1]] if len(shrunk) > 1 else []):
                try:
                    ok = prop(candidate)
                except Exception:
                    ok = False
                if not ok and len(candidate) < len(shrunk):
                    shrunk, improved = candidate, True
                    break
        return shrunk
    return None


def gen_floats(rng):
    n = rng.randint(1, 8)
    return [rng.choice([rng.uniform(-100, 100), 0.0, 1e9, -1e9])
            for _ in range(n)]


# Property 1: standardising always centres the data. TRUE, and it passes.
cx = check_property(lambda v: abs(standardise(v).mean()) < 1e-6,
                    gen_floats)
print(f"property 'mean is zero'      : "
      f"{'PASSED' if cx is None else f'failed on {cx}'}")

# Property 2: the result always has unit variance. FALSE — constant input.
cx = check_property(lambda v: abs(standardise(v).std() - 1.0) < 1e-6,
                    gen_floats)
print(f"property 'std is one'        : "
      f"{'PASSED' if cx is None else f'FAILED, shrunk to {cx}'}")
print("  The framework found the constant-input case, which the guard in")
print("  standardise() handles by returning zeros — a real edge case that an")
print("  example-based test would only cover if you had thought of it.")

# --- gradient checking as a test (Chapter 11) --------------------------------
print("\n" + "=" * 66)
print("testing what is testable in ML code")
print("=" * 66)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def loss_and_grad(w, X, y):
    p = np.clip(sigmoid(X @ w), 1e-12, 1 - 1e-12)
    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    grad = X.T @ (p - y) / len(y)
    return loss, grad


rng = np.random.default_rng(0)
X = rng.normal(size=(40, 5))
y = (rng.random(40) < 0.5).astype(float)
w = rng.normal(size=5)

# Test 1: shapes and invariants.
loss, grad = loss_and_grad(w, X, y)
assert grad.shape == w.shape, "gradient must match parameter shape"
assert loss >= 0, "cross-entropy is non-negative"
print(f"shape and invariant tests    : PASSED (loss={loss:.4f})")

# Test 2: the analytic gradient matches finite differences.
h = 1e-6
numeric = np.array([
    (loss_and_grad(w + h * np.eye(5)[i], X, y)[0]
     - loss_and_grad(w - h * np.eye(5)[i], X, y)[0]) / (2 * h)
    for i in range(5)])
err = np.abs(grad - numeric).max()
assert err < 1e-6, f"gradient check failed: {err}"
print(f"gradient check               : PASSED (max error {err:.2e})")

# Test 3: determinism under a fixed seed.
def run(seed):
    r = np.random.default_rng(seed)
    return float(r.normal(size=100).mean())


assert run(42) == run(42)
print(f"determinism under fixed seed : PASSED")

# Test 4: loss decreases — the cheapest useful smoke test.
w_train = np.zeros(5)
losses = []
for _ in range(30):
    l, g = loss_and_grad(w_train, X, y)
    losses.append(l)
    w_train -= 0.5 * g
assert losses[-1] < losses[0], "loss must decrease"
print(f"loss decreases over 30 steps : PASSED "
      f"({losses[0]:.4f} -> {losses[-1]:.4f})")

# --- profiling, and eq. 20.1 -------------------------------------------------
print("\n" + "=" * 66)
print("profile before optimising")
print("=" * 66)


def slow_component(n):
    return sum(i * i for i in range(n))          # the dominant cost


def fast_component(n):
    return sum(range(n))                          # the minor cost


def workload():
    total = 0
    for _ in range(30):
        total += slow_component(20_000)
        total += fast_component(20_000)
    return total


buf = io.StringIO()
profiler = cProfile.Profile()
profiler.enable()
workload()
profiler.disable()
pstats.Stats(profiler, stream=buf).sort_stats("cumtime").print_stats(6)
lines = [l for l in buf.getvalue().splitlines()
         if "component" in l or "cumtime" in l]
print("\n".join(lines[:4]))

# Measure the true split, then apply Amdahl.
t0 = time.perf_counter()
for _ in range(30):
    slow_component(20_000)
t_slow = time.perf_counter() - t0
t0 = time.perf_counter()
for _ in range(30):
    fast_component(20_000)
t_fast = time.perf_counter() - t0
p_slow = t_slow / (t_slow + t_fast)

print(f"\nmeasured split: slow_component is {p_slow:.1%} of runtime")
print(f"\n{'optimise':<18} {'by 10x':>9} {'by inf':>9}   (eq. 20.1)")
for label, p in (("the slow part", p_slow), ("the fast part", 1 - p_slow)):
    s10 = 1 / ((1 - p) + p / 10)
    smax = 1 / (1 - p)
    print(f"{label:<18} {s10:>8.2f}x {smax:>8.2f}x")
print("\nMaking the fast component infinitely fast barely helps. This is why")
print("guessing where the time goes is not a strategy.")
