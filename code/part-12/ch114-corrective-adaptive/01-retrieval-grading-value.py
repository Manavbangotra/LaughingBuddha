# -*- coding: utf-8 -*-
# Extracted from: Chapter 114 — Corrective and Adaptive RAG
# Source: src/.../ch114-corrective-adaptive.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Grading retrieval before generating on it, and what the grader's own errors cost.

Every chapter so far has tried to make retrieval better. None has asked what the
system does when retrieval is bad anyway -- and the default answer is the worst
one available: pass the bad context to the generator, which grounds a confident
wrong answer in it and cites the irrelevant documents (ch:rag-generation).

cite:yan2024crag inserts a grader between the two stages. The grader is itself a
classifier with errors, and eq:grading-value says its value depends on those
errors as much as on the retrieval failure rate. This listing measures the whole
decision -- coverage, accuracy, HARM (confident wrong answers), answer precision,
and retrieval cost -- as the grader gets noisier.

Every grading decision here is made on the GRADER's score, never on the true
quality, including the choice between a first and a second attempt. A harness
that peeks at ground truth when picking the better attempt would report a
corrective loop that nobody can build.
"""
import numpy as np

rng = np.random.default_rng(7)

N_QUERY = 60_000
P_BAD = 0.30            # share of queries whose retrieval genuinely fails
TAU = 0.50              # grade threshold


def draw_quality(n):
    """Retrieval quality per query: a bimodal mix, which is what measured
    retrieval looks like -- mostly fine, with a hard tail that is not close."""
    bad = rng.random(n) < P_BAD
    return np.where(bad, rng.beta(1.6, 6.0, n), rng.beta(6.0, 1.6, n))


def grade(q, sigma):
    """The grader observes quality through noise. sigma = 0 is the oracle grader
    the papers implicitly assume."""
    return q if sigma == 0 else q + rng.normal(scale=sigma, size=len(q))


def report(name, q, answer, retrievals):
    """An answered query is correct with probability q. Every wrong answer is
    HARM: fluent, cited, and unmarked as ungrounded."""
    correct = (rng.random(len(q)) < q) & answer
    n_ans = answer.sum()
    print(f"{name:<27}{n_ans / N_QUERY:>10.3f}{correct.sum() / N_QUERY:>11.3f}"
          f"{(n_ans - correct.sum()) / N_QUERY:>10.3f}"
          f"{(correct.sum() / n_ans if n_ans else 0):>12.3f}"
          f"{retrievals / N_QUERY:>12.2f}")


q1 = draw_quality(N_QUERY)
q2 = draw_quality(N_QUERY)          # the second attempt, if one is made
all_yes = np.ones(N_QUERY, dtype=bool)

print(f"{N_QUERY:,} queries; {P_BAD:.0%} with genuinely failed retrieval; "
      f"mean quality {q1.mean():.3f}\n")
print(f"{'policy':<27}{'coverage':>10}{'accuracy':>11}{'harm':>10}"
      f"{'precision':>12}{'retrievals':>12}")
print("-" * 82)

report("generate always", q1, all_yes, N_QUERY)

for sigma in (0.0, 0.10, 0.20, 0.40):
    g1 = grade(q1, sigma)
    passed = g1 >= TAU

    # 1. Abstain on a failing grade. A TERMINAL handler: a false reject costs
    #    the whole answer (eq:terminal-handler).
    report(f"  abstain      s={sigma:.2f}", q1, passed, N_QUERY)

    # 2. Retry on a failing grade, then keep whichever attempt the GRADER
    #    prefers. A RECOVERABLE handler: a false reject costs one retrieval
    #    (eq:recoverable-handler).
    g2 = grade(q2, sigma)
    retried = ~passed
    keep2 = retried & (g2 > g1)
    q_best = np.where(keep2, q2, q1)
    g_best = np.where(keep2, g2, g1)
    n_retr = N_QUERY + retried.sum()
    report(f"  retry        s={sigma:.2f}", q_best, all_yes, n_retr)

    # 3. Retry, then abstain if the surviving attempt still fails the grade
    #    (eq:composed-policy).
    report(f"  retry+abstain s={sigma:.2f}", q_best, g_best >= TAU, n_retr)

print("""
Read the first row as the baseline every RAG tutorial ships. It answers
everything, so coverage is 1.000 and answer precision equals accuracy: about
38% of everything the system says is a confident, cited, wrong answer. There is
no handler, so a failed retrieval is not an error condition -- it is an ordinary
input that produces ordinary-looking output.

Now the oracle rows (s=0.00), which is the regime the papers evaluate in.
Abstention raises answer precision from 0.615 to 0.803 and cuts harm by nearly
two thirds -- and it does that by LOWERING accuracy, from 0.615 to 0.549.
Refusing to answer means refusing some questions you would have got right. That
trade is the entire content of an abstention policy, and which side of it you
want is a product decision, not a modelling one.

Retry improves both axes at once: accuracy 0.750 against 0.615, harm 0.250
against 0.385, for 32% more retrieval calls. It is not a compromise between
answering and abstaining, and it is the same grader doing the deciding. A second
retrieval attempt simply has an independent chance of succeeding where the first
failed, and only the 32% that failed the grade pay for one.

The result worth carrying is what happens as the grader degrades, because the two
handlers degrade at very different rates. From s=0.00 to s=0.40, abstention loses
0.102 of accuracy (0.549 to 0.447) and 0.057 of precision. Retry loses 0.047 of
accuracy (0.750 to 0.703) -- less than half as much.

The reason is structural. Under abstention, a false reject costs the ENTIRE
answer: the grader's mistake is terminal. Under retry, a false reject costs one
extra retrieval, and the grader then gets a second chance to notice it preferred
the wrong attempt. Same grader, same error rate, one-quarter of the damage.

State it as a design rule, because it generalises well past retrieval: WHEN THE
DECIDER IS UNRELIABLE, MAKE ITS MISTAKES RECOVERABLE. A terminal handler inherits
the decider's error rate; a recoverable handler absorbs it.

Finally, compare the third row of each block against the first. At every noise
level, retry-then-abstain dominates abstain-alone on coverage and accuracy at
essentially identical precision -- at s=0.40 it is 0.842 coverage and 0.628
accuracy against 0.599 and 0.447, with precision 0.745 against 0.746. Retrying
BEFORE abstaining buys back most of the coverage abstention gives up and costs
nothing in precision to do it. If you implement one thing from this chapter, it
is that ordering.""")
