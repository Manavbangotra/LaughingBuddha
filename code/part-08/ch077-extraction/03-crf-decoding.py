# Extracted from: Chapter 77 — Classification, Named Entity Recognition, and Information Extraction
# Source: src/.../ch077-extraction.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A linear-chain CRF: exact partition function, and constrained decoding."""
import numpy as np
from itertools import product

TAGS = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC"]
K = len(TAGS)
rng = np.random.default_rng(0)


def build_transitions(constrain):
    """Transitions, plus start and stop vectors — equation (eq:crf-score).

    The start vector is easy to forget and is exactly where the constraint
    leaks: without start[I-*] = -inf a sequence may BEGIN with a continuation
    tag, which no pairwise transition can forbid.
    """
    A = rng.normal(0, 0.5, (K, K))
    start = rng.normal(0, 0.5, K)
    stop = rng.normal(0, 0.5, K)
    if constrain:
        for j, tag in enumerate(TAGS):
            if tag.startswith("I-"):
                typ = tag[2:]
                start[j] = -np.inf                      # cannot open with I-
                for i, prev in enumerate(TAGS):
                    if prev not in (f"B-{typ}", f"I-{typ}"):
                        A[i, j] = -np.inf               # cannot continue nothing
    return A, start, stop


def logsumexp(a, axis=None):
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)       # an all -inf row contributes nothing
    return np.squeeze(m, axis=axis) + np.log(np.exp(a - m).sum(axis=axis))


def log_partition(E, A, start, stop):
    """The forward algorithm — equation (eq:forward-recurrence), in log space."""
    alpha = E[0] + start
    for t in range(1, len(E)):
        alpha = E[t] + logsumexp(alpha[:, None] + A, axis=0)
    return logsumexp(alpha + stop)


def brute_force_log_partition(E, A, start, stop):
    """Sum over every one of K^T sequences. Tractable only because T is tiny."""
    scores = []
    for seq in product(range(K), repeat=len(E)):
        s = sum(E[t, y] for t, y in enumerate(seq)) + start[seq[0]] + stop[seq[-1]]
        s += sum(A[seq[t - 1], seq[t]] for t in range(1, len(seq)))
        scores.append(s)
    return logsumexp(np.array(scores))


def viterbi(E, A, start, stop):
    """Best path — the same dynamic program as (eq:viterbi-recurrence)."""
    T = len(E)
    delta = E[0] + start
    back = np.zeros((T, K), dtype=int)
    for t in range(1, T):
        scores = delta[:, None] + A
        back[t] = np.argmax(scores, axis=0)
        delta = E[t] + np.max(scores, axis=0)
    delta = delta + stop
    path = [int(np.argmax(delta))]
    for t in range(T - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    return [TAGS[i] for i in reversed(path)]


def illegal_count(tags):
    n, prev = 0, None
    for tag in tags:
        if tag.startswith("I-") and prev not in (f"B-{tag[2:]}", f"I-{tag[2:]}"):
            n += 1
        prev = tag
    return n


T = 6
E = rng.normal(0, 1.0, (T, K))         # emissions from the encoder head

# 1. The forward algorithm is exact, not an approximation — check it both with
#    and without the -inf entries, since those are where a log-space bug hides.
print(f"log Z over {K ** T:,} sequences")
for label, constrain in [("free transitions", False), ("constrained", True)]:
    A, st, sp = build_transitions(constrain)
    exact = brute_force_log_partition(E, A, st, sp)
    fast = log_partition(E, A, st, sp)
    print(f"  {label:<18} brute force {exact:>10.6f}   "
          f"forward {fast:>10.6f}   ({T} x {K}^2 = {T * K * K} operations)")
    assert abs(exact - fast) < 1e-9

# 2. Independent argmax versus free Viterbi versus constrained Viterbi.
A_free, st_free, sp_free = build_transitions(False)
A_con, st_con, sp_con = build_transitions(True)

print()
for name, tags in [
        ("independent argmax", [TAGS[i] for i in E.argmax(1)]),
        ("CRF, free transitions", viterbi(E, A_free, st_free, sp_free)),
        ("CRF, constrained", viterbi(E, A_con, st_con, sp_con))]:
    print(f"{name:<24} {' '.join(f'{t:<6}' for t in tags)}  "
          f"illegal: {illegal_count(tags)}")

# 3. The property, over many random emission matrices.
n_trials, bad_independent, bad_constrained = 2000, 0, 0
for _ in range(n_trials):
    Ei = rng.normal(0, 1.0, (T, K))
    bad_independent += illegal_count([TAGS[i] for i in Ei.argmax(1)]) > 0
    bad_constrained += illegal_count(viterbi(Ei, A_con, st_con, sp_con)) > 0

print(f"\nover {n_trials:,} random emission matrices:")
print(f"  independent argmax produced ill-formed output "
      f"{bad_independent / n_trials:.1%} of the time")
print(f"  constrained Viterbi produced ill-formed output "
      f"{bad_constrained / n_trials:.1%} of the time")
assert bad_constrained == 0
