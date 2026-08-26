# -*- coding: utf-8 -*-
# Extracted from: Chapter 90 — Decoding: Softmax, Temperature, Top-k, Top-p, and Beam Search
# Source: src/.../ch090-decoding.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Beam search: correct for constrained tasks, wrong for open-ended ones."""
import numpy as np

rng = np.random.default_rng(2)


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def beam_search(logit_fn, n_steps, beam_width, V, alpha=1.0):
    """Equation (eq:beam-search), with length normalisation."""
    beams = [([], 0.0)]
    for _ in range(n_steps):
        cands = []
        for seq, score in beams:
            lp = np.log(softmax(logit_fn(seq)) + 1e-12)
            for v in range(V):
                cands.append((seq + [v], score + float(lp[v])))
        cands.sort(key=lambda sc: sc[1] / (len(sc[0]) ** alpha), reverse=True)
        beams = cands[:beam_width]
    return beams[0]


def greedy_decode(logit_fn, n_steps):
    seq, total = [], 0.0
    for _ in range(n_steps):
        lp = np.log(softmax(logit_fn(seq)) + 1e-12)
        v = int(lp.argmax())
        total += float(lp[v])
        seq.append(v)
    return seq, total


# ---------------------------------------------------------------------------
# TASK A: a constrained task with a correct answer, containing a GREEDY TRAP —
# a token that is locally best at step 0 and leads nowhere. This is the exact
# situation beam search exists for, and note that the model is a DETERMINISTIC
# function of the history: beam search over a stochastic scorer is meaningless.
# ---------------------------------------------------------------------------
VA, TARGET, TRAP = 12, [2, 5, 8], 3


def constrained_logits(history):
    z = np.full(VA, -4.0)
    step = len(history)
    if step == 0:
        z[TRAP] = 2.0            # locally the best choice...
        z[TARGET[0]] = 1.6       # ...and this one is second best
    elif step == 1:
        if history[0] == TRAP:
            z[:] = -1.0          # ...but the trap leads to a flat, poor region
            z[4] = 0.2
        elif history[0] == TARGET[0]:
            z[TARGET[1]] = 3.2   # while the target path is rich
    else:
        if history[:2] == TARGET[:2]:
            z[TARGET[2]] = 3.4
        elif history[0] == TRAP:
            z[6] = 0.1
    return z


print("TASK A — constrained: there IS a correct answer\n")
print(f"target: {TARGET}   (step 0 has a trap at token {TRAP})")
print(f"{'method':<12} {'output':<14} {'log P':>9} {'exact match':>13}")
out, lp = greedy_decode(constrained_logits, len(TARGET))
print(f"{'greedy':<12} {str(out):<14} {lp:>9.3f} {str(out == TARGET):>13}")
for width in (2, 3, 5):
    out_b, lp_b = beam_search(constrained_logits, len(TARGET), width, VA)
    print(f"{'beam ' + str(width):<12} {str(out_b):<14} {lp_b:>9.3f} "
          f"{str(out_b == TARGET):>13}")

print("""
Greedy takes the locally-best token at step 0 and is then stuck in a region
where everything is mediocre. Beam search keeps the second-best prefix alive
long enough to discover that it leads somewhere much better, and finds a
sequence with more than twice the log-probability.

This is what beam search is FOR, and it is why it remains standard in
translation and structured extraction — tasks where a target exists and local
choices can be traps.""")

# ---------------------------------------------------------------------------
# TASK B: open-ended, with the repetition feedback of eq:repetition-feedback.
# ---------------------------------------------------------------------------
VB = 24
base = rng.normal(size=(VB, VB))


def open_logits(history):
    z = (base[history[-1]] if history else base[0]).copy()
    for tok in history[-5:]:
        z[tok] += 1.2                     # eq:repetition-feedback
    return z


def rep_rate(seq, n=3):
    grams = [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]
    return 1 - len(set(grams)) / len(grams)


print("\nTASK B — open-ended: there is NO correct answer\n")
print(f"{'method':<14} {'distinct':>10} {'3-gram repeat':>15} "
      f"{'mean log p':>12}")
for label, width in [("greedy", 1), ("beam 3", 3), ("beam 10", 10)]:
    if width == 1:
        out, lp = greedy_decode(open_logits, 50)
    else:
        out, lp = beam_search(open_logits, 50, width, VB)
    print(f"{label:<14} {len(set(out)):>10} {rep_rate(out):>15.3f} "
          f"{lp / len(out):>12.4f}")

g = np.random.default_rng(3)
seq = []
for _ in range(50):
    pr = softmax(open_logits(seq))
    seq.append(int(g.choice(VB, p=pr)))
print(f"{'sampling T=1':<14} {len(set(seq)):>10} {rep_rate(seq):>15.3f} "
      f"{'-':>12}")

print("""
On the open-ended task, searching harder buys nothing. Beam search finds
sequences with roughly four times greedy's mean log-probability and they are
JUST AS REPETITIVE — both collapse to two distinct tokens and repeat 96% of
their 3-grams. The extra search effort located a higher point inside the same
degenerate region, because equation (eq:repetition-feedback) means the mode IS
that region.

Sampling finds the LEAST probable text of the three and is the only one that
stays varied: sixteen distinct tokens and 19% repetition. That is the inversion
— on Task A more search meant a better answer, and here it means a
higher-scoring version of the same failure.

The whole rule: beam search when the task has a right answer, sampling when it
does not.""")
