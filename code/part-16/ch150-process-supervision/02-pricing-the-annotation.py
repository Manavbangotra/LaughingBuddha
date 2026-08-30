# -*- coding: utf-8 -*-
# Extracted from: Chapter 150 — Process versus Outcome Supervision
# Source: src/.../ch150-process-supervision.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Pricing the annotation.

A process label costs K times an outcome label, because there are K steps. The
previous listing showed process supervision is worth MORE when a wrong
derivation often reaches a right answer, and worth nothing -- or less than
nothing -- when it does not. This one asks the budget question directly: given a
fixed number of LABELS, not a fixed number of examples, which signal should you
buy (eq:label-budget)?

It also prices the shortcut everyone reaches for first. If step labels are
expensive, impute them: take the chains whose answer was right and mark all their
steps correct, which is the labelling rule behind cite:zelikman2022star's
rejection-sampling loop and behind bootstrapped process reward models. It is free.
It is also wrong on exactly the lucky chains, and this listing measures what that
costs.
"""
import numpy as np

rng = np.random.default_rng(1013)

K = 6
VALS = np.arange(-3, 4)
P_STEP_OK = 0.80
SIGMA = 0.9
LABEL_NOISE = 0.08
N_TEST = 4000
NS = 16


def make(n, spread=1):
    true_steps = rng.choice(VALS, size=(n, K))
    ok = rng.random((n, K)) < P_STEP_OK
    wrong = rng.choice(VALS * spread, size=(n, K))
    taken = np.where(ok, true_steps, wrong)
    sound = ok.all(1)
    correct = taken.sum(1) == true_steps.sum(1)
    obs = (taken - true_steps) + SIGMA * rng.normal(size=(n, K))
    obs_fin = (taken.sum(1) - true_steps.sum(1)) + SIGMA * rng.normal(size=n)
    return taken, ok, sound, correct, obs, obs_fin


def feats_chain(taken, obs, obs_fin):
    run = np.cumsum(taken, axis=1)
    orun = np.cumsum(obs, axis=1)
    return np.concatenate([taken, obs, run, orun,
                           obs_fin[:, None], np.abs(obs_fin)[:, None],
                           obs.sum(1)[:, None], np.abs(obs).sum(1)[:, None],
                           np.ones((len(taken), 1))], axis=1)


def feats_step(taken, obs):
    n = len(taken)
    pos = np.tile(np.arange(K), (n, 1))
    prev = np.concatenate([np.zeros((n, 1)), np.cumsum(taken, 1)[:, :-1]], 1)
    return np.stack([taken, obs, np.abs(obs), pos, prev,
                     np.ones((n, K))], axis=2).reshape(n * K, 6)


NF = 300
W_C = rng.normal(size=(4 * K + 5, NF)) * 0.7
B_C = rng.uniform(0, 2 * np.pi, NF)
W_S = rng.normal(size=(6, NF)) * 0.7
B_S = rng.uniform(0, 2 * np.pi, NF)


def fit(X, y, W, B, lam=1e-2):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    F = np.cos(((X - mu) / sd) @ W + B)
    return (mu, sd, W, B,
            np.linalg.solve(F.T @ F + lam * np.eye(NF), F.T @ y))


def score(m, X):
    mu, sd, W, B, c = m
    return np.cos(((X - mu) / sd) @ W + B) @ c


def evaluate(model, kind, spread=1):
    """Best-of-NS selection accuracy and soundness on held-out chains."""
    tk, ok, sd_, cr, ob, of = make(N_TEST * NS, spread=spread)
    if kind == "orm":
        s = score(model, feats_chain(tk, ob, of)).reshape(N_TEST, NS)
    else:
        s = score(model, feats_step(tk, ob)).reshape(N_TEST, NS, K).min(2)
    idx = s.argmax(1)
    r = np.arange(N_TEST)
    return (float(cr.reshape(N_TEST, NS)[r, idx].mean()),
            float(sd_.reshape(N_TEST, NS)[r, idx].mean()))


def train_orm(n_chains, spread=1):
    tk, ok, sd_, cr, ob, of = make(n_chains, spread=spread)
    return fit(feats_chain(tk, ob, of), cr.astype(float), W_C, B_C)


def train_prm(n_chains, spread=1, impute=False):
    """impute=False buys real step labels (K per chain, noisy).
    impute=True buys only OUTCOME labels and marks every step of an
    answer-correct chain as correct -- free, and wrong on the lucky ones."""
    tk, ok, sd_, cr, ob, of = make(n_chains, spread=spread)
    if impute:
        lab = np.repeat(cr, K).astype(float)
    else:
        lab = ok.reshape(-1).astype(float)
        flip = rng.random(lab.shape) < LABEL_NOISE
        lab = np.where(flip, 1.0 - lab, lab)
    return fit(feats_step(tk, ob), lab, W_S, B_S)


BUDGETS = [900, 1800, 3600, 7200, 14400, 28800]

print(f"Chains of {K} steps, so one process-labelled chain costs {K} labels and")
print("one outcome-labelled chain costs 1. Both models are then used to pick")
print(f"the best of {NS} candidates. Selection accuracy at equal LABEL budget:")
print()
print(f"{'':>10}{'chains bought':>24}{'answer accuracy':>27}")
print(f"{'labels':>10}{'outcome':>12}{'process':>12}{'outcome':>13}"
      f"{'process':>14}")
print("-" * 61)

tab = {}
for B in BUDGETS:
    o = train_orm(B)
    p = train_prm(max(B // K, 20))
    ro, rp = evaluate(o, "orm"), evaluate(p, "prm")
    tab[B] = (ro, rp)
    print(f"{B:>10}{B:>12}{B // K:>12}{ro[0]:>13.1%}{rp[0]:>14.1%}")

print()
print()
print("The same budgets, scored on SOUNDNESS of the selected derivation.")
print()
print(f"{'labels':>10}{'outcome':>13}{'process':>14}{'gap':>10}")
print("-" * 47)
for B in BUDGETS:
    ro, rp = tab[B]
    print(f"{B:>10}{ro[1]:>13.1%}{rp[1]:>14.1%}{rp[1] - ro[1]:>+10.1%}")

print()
print()
print("Imputed step labels: mark every step of an answer-correct chain correct.")
print("Costs one outcome label per chain, so the same budget buys K times as")
print("many chains as real step annotation does.")
print()
print(f"{'':>10}{'answer accuracy':>40}{'soundness':>26}")
print(f"{'labels':>10}{'outcome':>13}{'imputed':>13}{'real':>14}"
      f"{'imputed':>13}{'real':>13}")
print("-" * 76)
imp = {}
for B in BUDGETS:
    pi = train_prm(B, impute=True)
    ri = evaluate(pi, "prm")
    ro, rp = tab[B]
    imp[B] = ri
    print(f"{B:>10}{ro[0]:>13.1%}{ri[0]:>13.1%}{rp[0]:>14.1%}"
          f"{ri[1]:>13.1%}{rp[1]:>13.1%}")

print()
print()
print("And the same three signals where lucky chains are RARE (spread=4),")
print(f"at a budget of {BUDGETS[-1]} labels.")
print()
print(f"{'signal':>22}{'answer':>11}{'soundness':>12}")
print("-" * 45)
B = BUDGETS[-1]
far = {}
for name, m, kind in (
        ("outcome", train_orm(B, spread=4), "orm"),
        ("process (real)", train_prm(B // K, spread=4), "prm"),
        ("process (imputed)", train_prm(B, spread=4, impute=True), "prm")):
    r = evaluate(m, kind, spread=4)
    far[name] = r
    print(f"{name:>22}{r[0]:>11.1%}{r[1]:>12.1%}")

Bs = BUDGETS[0]
Bm = BUDGETS[3]
Bl = BUDGETS[-1]
orm_plateau = tab[Bl][0][0] - tab[Bm][0][0]
prm_plateau = tab[Bl][1][0] - tab[Bm][1][0]
print(f"""
The first table is the budget question stated properly -- equal LABELS, not equal
examples -- and it does not come out as the crossover I expected.

At {Bs} labels the outcome model buys {Bs} chains and reaches {tab[Bs][0][0]:.1%};
the process model buys only {Bs // K} chains and still reaches
{tab[Bs][1][0]:.1%}. At {Bl} labels it is {tab[Bl][0][0]:.1%} against
{tab[Bl][1][0]:.1%}. The process signal leads at every budget swept, despite
buying {K} times fewer examples throughout.

The second column pair says why, and it is the real finding. Between {Bm} and
{Bl} labels -- a fourfold increase -- the outcome model improves by
{orm_plateau:+.1%} and the process model by {prm_plateau:+.1%}. The outcome
model has stopped learning.

**Outcome supervision is not a noisier version of process supervision. It is a
BIASED version**, and more data does not fix a bias. Its labels say that lucky
chains are good, they say it consistently, and a larger sample estimates that
same wrong target more precisely (eq:label-budget). The plateau is where the
model has fully learned the signal it was given, and the signal was partly wrong.

That reframes the annotation question. The comparison is not "is a process label
worth {K} outcome labels" -- at a high enough budget it is worth an unlimited
number of them, because they are buying a target that is off by a fixed amount.
The comparison is against the ceiling, and the ceiling is set by the lucky-chain
rate rather than by the budget.

The soundness table is the same story with a wider gap:
{tab[Bl][1][1] - tab[Bl][0][1]:+.1%} at the largest budget, and no sign of the
outcome model closing it.

The third table is where the practical advice lives, and it is the one to act on.

Imputing step labels from outcomes -- take the chains whose answer was right and
mark every step correct, which is the labelling rule inside
cite:zelikman2022star's loop and inside every bootstrapped process reward model --
costs one label per chain instead of {K}. At {Bs} labels it reaches
{imp[Bs][0]:.1%}, beating both real process labels ({tab[Bs][1][0]:.1%}) and the
outcome model ({tab[Bs][0][0]:.1%}). At {Bl} labels it reaches {imp[Bl][0]:.1%}
against real process labels' {tab[Bl][1][0]:.1%}.

So a free heuristic gets within {tab[Bl][1][0] - imp[Bl][0]:.1%} of human step
annotation on answers, for one {K}th of the price, and it is strictly the best
option when labels are scarce -- because it converts the same outcome budget into
{K} times as many training rows without needing anyone to grade a step.

It is not a free lunch, and the soundness columns show the tax:
{imp[Bl][1]:.1%} against real step labels' {tab[Bl][1][1]:.1%}. The imputation is
wrong on exactly the lucky chains -- it hands {K} confident "this step is correct"
labels to steps that were not -- so it inherits a weakened form of the bias it was
meant to remove. It closes most of the gap to real process supervision and cannot
close all of it, and the residue is proportional to the same lucky-chain rate as
everything else in this chapter.

The fourth table removes the lucky chains entirely. At spread=4 the three signals
land at {far['outcome'][0]:.1%}, {far['process (real)'][0]:.1%} and
{far['process (imputed)'][0]:.1%} on answers, and {far['outcome'][1]:.1%},
{far['process (real)'][1]:.1%} and {far['process (imputed)'][1]:.1%} on
soundness. The outcome model and real process supervision converge, as they
should: with nothing for the outcome label to be wrong about, one bit per chain
is very nearly a soundness label, and paying {K}x buys {far['process (real)'][0] - far['outcome'][0]:+.1%}.

The imputed model does NOT converge with them -- it drops to
{far['process (imputed)'][0]:.1%}, {far['process (imputed)'][0] - far['outcome'][0]:+.1%}
against the plain outcome model it was built from. That reversal is worth
understanding, because it is the flaw in the heuristic showing from the other
side. Imputation marks every step of a correct chain correct, which is now
accurate; and every step of an INCORRECT chain incorrect, which never was. In an
unsound chain most steps are usually fine and one is not, so the negative labels
are mostly false, and when luck is rare there are more unsound chains for the
heuristic to mislabel. It trades one bias for another rather than removing it.

Which sharpens the advice rather than reversing it: imputation is a good deal
where lucky chains are COMMON, because that is where the outcome signal it
replaces is at its worst, and a bad deal where they are rare, because there the
outcome signal was already nearly right and the imputation adds noise of its
own.

So the decision procedure needs one measurement rather than a policy, and it is
the same measurement the previous listing pointed at.

Grade a few hundred of your own solutions both ways and compute the share of
correct answers that came from faulty reasoning. That share is the entire
headroom. If it is small, buy outcome labels. If it is large, do NOT jump
straight to human step annotation: impute step labels from the outcomes you
already have, measure again, and buy real ones only for the residue -- which this
listing prices at a few points of soundness and almost nothing on answers.

And keep measuring soundness separately from accuracy, because every plateau in
this listing is invisible in the answer column until you look at the other
one.""")
