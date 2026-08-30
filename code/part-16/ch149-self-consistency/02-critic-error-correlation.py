# -*- coding: utf-8 -*-
# Extracted from: Chapter 149 — Self-Consistency, Reflection, and Critic Models
# Source: src/.../ch149-self-consistency.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Why voting works and reflection does not, measured side by side.

cite:huang2024selfcorrect's result is that intrinsic self-correction -- a model
revising its own answer using only its own capabilities -- does not improve
reasoning performance and often degrades it, while correction guided by external
feedback does help. Those two findings are usually reported together as a
puzzle: the model can clearly criticise, so why does criticising itself not
work?

This listing takes the previous one's setup and adds a revision loop
(eq:correlated-critic). A proposal is judged by a critic; if the critic rejects
it, the solver proposes again. Three critics, matched so the comparison is about
CORRELATION and not about competence:

  self-check   the solver's own judgement -- sample again and see whether the
               proposal is what it would say
  independent  a critic with an IDENTICAL confusion matrix whose errors fall
               elsewhere
  oracle       knows the answer

The self-check critic is not a weakened version of the others. It accepts correct
answers and catches wrong ones at exactly the same rates. The only thing that
differs is what it is wrong ABOUT.
"""
import numpy as np
from collections import Counter

rng = np.random.default_rng(811)

N_PROB = 8000
R = 12
P_CORRECT_ROUTE = 0.30
TAU = 1.0
M_CHECK = 5                 # samples the self-check critic draws
ROUNDS = 4

logits = rng.normal(size=(N_PROB, R))
is_correct = rng.random((N_PROB, R)) < P_CORRECT_ROUTE
route_ans = np.where(is_correct, 0, rng.integers(1, 5, size=(N_PROB, R)))

z = logits / TAU
z = z - z.max(1, keepdims=True)
PR = np.exp(z)
PR /= PR.sum(1, keepdims=True)
CUM = PR.cumsum(1)


def draw(idx_rows, n):
    """Draw n samples for each problem in idx_rows."""
    u = rng.random((len(idx_rows), n))
    j = (u[:, :, None] > CUM[idx_rows][:, None, :]).sum(2).clip(0, R - 1)
    return np.take_along_axis(route_ans[idx_rows], j, axis=1)


ALL = np.arange(N_PROB)
# Per-problem answer distribution, used to reason about what the pool contains.
p_correct = np.array([PR[i][route_ans[i] == 0].sum() for i in ALL])
pool = draw(ALL, 200)
modal = np.array([Counter(pool[i].tolist()).most_common(1)[0][0]
                  for i in ALL])

print(f"{N_PROB} problems, {R} routes each, temperature {TAU}.")
print(f"A single sample is correct {float(np.mean(draw(ALL, 1)[:, 0] == 0)):.1%}"
      " of the time.")
print(f"The correct answer is the modal one for {float(np.mean(modal == 0)):.1%}"
      " of problems.")
print()
print()


def self_check(idx_rows, proposals, t):
    """The solver judging itself: draw M_CHECK more samples and accept the
    proposal if it reappears at least t times. This is what "are you sure?"
    amounts to when the reviewer and the author are the same distribution, and
    t is how sceptically the question is asked -- t=1 is a gentle "does this
    still look right", t=4 is "convince me"."""
    s = draw(idx_rows, M_CHECK)
    return (s == proposals[:, None]).sum(1) >= t


def oracle_check(idx_rows, proposals):
    return proposals == 0


# Calibrate an INDEPENDENT critic to the self-check critic's measured accuracy.
probe = draw(ALL, 1)[:, 0]
STRICT = [1, 2, 3, 4]
sc_of = {t: self_check(ALL, probe, t) for t in STRICT}
RATES = {t: (float(np.mean(sc_of[t][probe == 0])),
             float(np.mean(~sc_of[t][probe != 0]))) for t in STRICT}


def independent_check(idx_rows, proposals, tpr, tnr):
    """A critic with the SAME confusion matrix as the self-check critic -- same
    rate of accepting correct proposals, same rate of rejecting wrong ones --
    whose errors fall independently of which problem it is looking at.

    Matching the full confusion matrix rather than overall accuracy matters: two
    critics with equal accuracy can sit at completely different operating
    points, and then the comparison measures the operating point instead of the
    correlation."""
    truth = proposals == 0
    u = rng.random(len(proposals))
    return np.where(truth, u < tpr, u > tnr)


print("Each critic, measured on one round of proposals. t is how many of the")
print(f"{M_CHECK} re-samples must match the proposal for the self-check critic to")
print("accept it. The independent critic is given the SAME two rates.")
print()
print(f"{'strictness t':>14}{'accepts a':>13}{'rejects a':>13}{'overall':>11}")
print(f"{'':>14}{'correct one':>13}{'wrong one':>13}{'accuracy':>11}")
print("-" * 51)
good = probe == 0
for t in STRICT:
    tpr, tnr = RATES[t]
    ov = float(np.mean(sc_of[t] == good))
    print(f"{t:>14}{tpr:>13.1%}{tnr:>13.1%}{ov:>11.1%}")

print()
print()
print(f"Revision loop: propose; if the critic rejects, propose again; "
      f"{ROUNDS} rounds.")
print("Accuracy at the end, for the self-check critic and for an independent")
print("critic with an IDENTICAL confusion matrix.")
print()


# An independent critic is a different model, so it has its own preferred
# answer. Give it one that is correct as often as the solver's mode is, but on
# an independently chosen set of problems.
indep_pref = np.where(rng.random(N_PROB) < float(np.mean(modal == 0)),
                      0, rng.integers(1, 5, size=N_PROB))
GAMMA = 0.75          # how strongly a revision moves toward the critique


def run(kind, t, policy="resample", rounds=ROUNDS):
    """policy='resample' redraws from the solver. policy='toward' regenerates
    CONDITIONED on the critique, which pulls the answer toward whatever the
    critic would have said -- which is what "reconsider, given this objection"
    does in practice."""
    cur = draw(ALL, 1)[:, 0]
    tpr, tnr = RATES[t]
    out = [float(np.mean(cur == 0))]
    for _ in range(rounds):
        if kind == "self":
            keep = self_check(ALL, cur, t)
        elif kind == "indep":
            keep = independent_check(ALL, cur, tpr, tnr)
        elif kind == "oracle":
            keep = oracle_check(ALL, cur)
        else:
            keep = np.ones(N_PROB, dtype=bool)
        redo = np.flatnonzero(~keep)
        if len(redo):
            cur = cur.copy()
            fresh = draw(redo, 1)[:, 0]
            if policy == "toward" and kind in ("self", "indep"):
                pref = modal[redo] if kind == "self" else indep_pref[redo]
                pull = rng.random(len(redo)) < GAMMA
                cur[redo] = np.where(pull, pref, fresh)
            else:
                cur[redo] = fresh
        out.append(float(np.mean(cur == 0)))
    return out


none = run("none", 1)
orac = run("oracle", 1)
print(f"{'':>14}{'redraw on reject':>27}{'revise toward critique':>29}")
print(f"{'strictness t':>14}{'self-check':>14}{'independent':>13}"
      f"{'self-check':>15}{'independent':>14}")
print("-" * 70)
selfc, indep, selft, indt = {}, {}, {}, {}
for t in STRICT:
    selfc[t] = run("self", t)
    indep[t] = run("indep", t)
    selft[t] = run("self", t, "toward")
    indt[t] = run("indep", t, "toward")
    print(f"{t:>14}{selfc[t][ROUNDS]:>14.1%}{indep[t][ROUNDS]:>13.1%}"
          f"{selft[t][ROUNDS]:>15.1%}{indt[t][ROUNDS]:>14.1%}")
print(f"{'none':>14}{none[ROUNDS]:>14.1%}{none[ROUNDS]:>13.1%}"
      f"{none[ROUNDS]:>15.1%}{none[ROUNDS]:>14.1%}")
print(f"{'oracle':>14}{orac[ROUNDS]:>14.1%}{'--':>13}{'--':>15}{'--':>14}")

T_HARD = 3
print()
print()
print(f"Where does the self-check critic go wrong? At t={T_HARD}, split by")
print("whether the solver's modal answer is the correct one.")
print()
print(f"{'problems where the mode is':>30}{'count':>8}{'accepts a':>13}"
      f"{'rejects a':>13}")
print(f"{'':>30}{'':>8}{'correct one':>13}{'wrong one':>13}")
print("-" * 64)
sch = sc_of[T_HARD]
split = {}
for name, m in (("correct", modal == 0), ("wrong", modal != 0)):
    g = good & m
    b = (~good) & m
    tpr = float(np.mean(sch[g])) if g.any() else float("nan")
    tnr = float(np.mean(~sch[b])) if b.any() else float("nan")
    split[name] = (int(m.sum()), tpr, tnr)
    print(f"{name:>30}{int(m.sum()):>8}{tpr:>13.1%}{tnr:>13.1%}")

n0, orc = none, orac
vote_acc = float(np.mean(modal == 0))
gaps = ", ".join("%+.1f%%" % (100 * (indep[t][ROUNDS] - selfc[t][ROUNDS]))
                 for t in STRICT)
best_redraw = max(selfc[t][ROUNDS] for t in STRICT)
best_toward = max(selft[t][ROUNDS] for t in STRICT)
best_itoward = max(indt[t][ROUNDS] for t in STRICT)
print(f"""
The first table is the critic, and it behaves sensibly. Asking more sceptically
trades one error for the other in the ordinary way: at t=1 it accepts a correct
proposal {RATES[1][0]:.1%} of the time and catches a wrong one {RATES[1][1]:.1%}
of the time; at t=4 those are {RATES[4][0]:.1%} and {RATES[4][1]:.1%}. Overall
accuracy improves throughout, from {float(np.mean(sc_of[1] == good)):.1%} to
{float(np.mean(sc_of[4] == good)):.1%}. On any summary statistic this is a critic
getting better at its job.

The second table is the revision loop, and it holds three results.

First: with the redraw policy, self-correction helps a little. The best strictness
takes accuracy from {none[ROUNDS]:.1%} to {best_redraw:.1%}. That is real, and it
is a long way below the oracle's {orc[ROUNDS]:.1%} -- which is the same loop,
same solver, same number of rounds, with a critic that actually knows.

Second: an independent critic with an IDENTICAL confusion matrix does better at
every strictness, by {gaps}. Same acceptance rate on correct proposals, same
rejection rate on wrong ones, different outcome. The difference is not
competence; it is WHERE the errors fall (eq:correlated-critic).

Third, and this is the one to carry: under the "revise toward the critique"
policy -- where a rejected answer is regenerated conditioned on the objection
rather than redrawn from scratch, which is what reconsidering actually does --
self-correction reaches {best_toward:.1%}.

Compare that with the majority vote over the same generator, which is
{vote_acc:.1%}.

Those are the same number, and they are the same number for a reason.
**Self-correction against your own judgement is a slow, sequential, expensive
implementation of self-consistency.** Revising toward what you would say on
reflection moves the answer toward your modal answer, and the modal answer is
exactly what voting returns in one parallel batch. The reflection loop cannot
exceed it, because there is nothing in the loop that was not already in the
distribution being sampled. The independent critic under the same policy reaches
{best_itoward:.1%}, and the difference between those two columns is the entire
value of the critic being a different system.

The third table shows why the ceiling sits where it does, and it is the sharpest
number here.

At t={T_HARD}, on problems where the solver's modal answer is already correct, the
self-check critic accepts a correct proposal {split['correct'][1]:.1%} of the time
and rejects a wrong one {split['correct'][2]:.1%} of the time. A useful critic.

On problems where the mode is WRONG, it accepts a correct proposal
{split['wrong'][1]:.1%} of the time.

That is the whole failure in one number. On exactly the problems the system gets
wrong, a correct answer that does turn up is thrown away nine times in ten,
because the critic's test is "is this what I would say?" and what it would say on
those problems is wrong. The critic is not merely unhelpful where it matters. It
is inverted.

So voting and reflection are not two techniques with different strengths. They
are the same information used two ways, and one of them costs a sequential round
trip per revision.

Voting AGGREGATES: it reads the whole distribution and returns its mode. It never
asks the model to evaluate anything, so there is no second judgement that could
be correlated with the first.

Reflection FILTERS: it conditions on a proposal and asks the model to judge it.
That judgement is computed by the distribution that produced the proposal, so
where the distribution is wrong the judgement is wrong in the same way, and the
filter discards the samples that would have rescued it.

One honest note on what this listing does and does not show. It reproduces
cite:huang2024selfcorrect's finding that intrinsic self-correction adds nothing
the model did not already contain, and their finding that external feedback does
help. It does NOT reproduce the outright DEGRADATION they report on some tasks,
and the reason is a mechanism this model omits: a real model asked to reconsider
will sometimes abandon a correct answer simply because it was challenged,
regardless of what its own distribution says. That is instruction-following under
pressure rather than a property of the reasoning distribution, and leaving it out
makes the picture here optimistic rather than pessimistic.

The practical question is therefore not "is my critic any good" -- the first table
shows that can be answered yes while the loop delivers almost nothing. It is
**how correlated is my critic's error with my solver's**, and the cheapest large
improvement available is not a better critic but a differently-wrong one: another
model, another training lineage, or an executable check that is not a model at
all, which is ch:rsn-tool-assisted's subject.""")
