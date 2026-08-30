# -*- coding: utf-8 -*-
# Extracted from: Chapter 149 — Self-Consistency, Reflection, and Critic Models
# Source: src/.../ch149-self-consistency.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Why self-consistency works, and what it is actually doing.

cite:wang2023selfconsistency's account is precise and worth taking literally:
sampling several reasoning paths and keeping the most frequent ANSWER
marginalises over the paths. Greedy decoding follows the single most likely
PATH, and those are different objects -- the most likely path need not lead to
the answer with the most total probability behind it, because many paths can
reach one answer while each wrong answer is reached by few.

This listing measures that gap directly (eq:marginalise-over-paths). Each problem
gets a set of latent routes with random weights; some fraction of them reach the
correct answer and the rest reach wrong ones, with some wrong answers reachable
by more than one route. Nothing is hand-tuned: the routes and their weights are
drawn, and greedy and voting are then compared on whatever population results.

Then it asks what else you can do with the same pool once a verifier is
available, and how each rule behaves when the verifier is bad.
"""
import numpy as np
from collections import Counter

rng = np.random.default_rng(733)

N_PROB = 6000
R = 12                      # latent routes per problem
NS = 40                     # samples per problem
P_CORRECT_ROUTE = 0.30      # chance a given route reaches the right answer

# Per-problem route structure. `route_ans[i, r]` is the answer route r produces:
# 0 means the correct answer, and positive integers are distinct wrong answers.
# Wrong routes are grouped, so several routes can share one wrong answer -- which
# is what makes a wrong answer competitive in a vote.
logits = rng.normal(size=(N_PROB, R))
is_correct = rng.random((N_PROB, R)) < P_CORRECT_ROUTE
wrong_group = rng.integers(1, 5, size=(N_PROB, R))       # 4 wrong answers
route_ans = np.where(is_correct, 0, wrong_group)


def derail_rate(tau):
    """Higher temperature also breaks chains outright. A derailed sample leaves
    the route set entirely and produces its own unique wrong answer, so derailed
    samples never form a bloc -- which is what an incoherent chain looks like."""
    return 1.0 - np.exp(-0.35 * tau)


def sample_pool(tau):
    """Sample NS routes per problem from softmax(logits / tau) and record the
    answer each one reaches. tau -> 0 is greedy decoding."""
    if tau <= 1e-6:
        best = logits.argmax(1)
        out = np.repeat(route_ans[np.arange(N_PROB), best][:, None], NS, 1)
        return out
    z = logits / tau
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    c = p.cumsum(1)
    u = rng.random((N_PROB, NS))
    idx = (u[:, :, None] > c[:, None, :]).sum(2).clip(0, R - 1)
    out = np.take_along_axis(route_ans, idx, axis=1)
    d = rng.random((N_PROB, NS)) < derail_rate(tau)
    uniq = 100 + np.arange(N_PROB * NS).reshape(N_PROB, NS)
    return np.where(d, uniq, out)


def vote(row):
    return Counter(row.tolist()).most_common(1)[0][0]


def scores(row, mu):
    """A verifier with a REAL error rate. Correct samples draw N(mu, 1) and
    incorrect ones N(0, 1). Unlike a verifier that merely fails to notice
    correct answers, this one produces FALSE POSITIVES -- some wrong sample
    scores high by chance -- which is what makes selecting over a large pool
    hazardous. mu = 0 is a useless verifier; the ranking accuracy it implies is
    measured below rather than assumed."""
    return rng.normal(size=len(row)) + mu * (row == 0)


def ranking_accuracy(mu, n=200000):
    """The verifier's own quality, measured: how often does it score a random
    correct sample above a random incorrect one?"""
    a = rng.normal(size=n) + mu
    b = rng.normal(size=n)
    return float(np.mean(a > b))


def agg_argmax(row, mu):
    return int(row[int(np.argmax(scores(row, mu)))])


def agg_weighted(row, mu):
    """Weighted self-consistency: total each distinct answer's verifier score
    rather than counting samples or trusting a single top score."""
    s = scores(row, mu)
    e = np.exp(s - s.max())          # softmax weights, so one huge score cannot
    e /= e.sum()                     # outvote a well-supported answer outright
    tot = {}
    for a, w in zip(row.tolist(), e.tolist()):
        tot[a] = tot.get(a, 0.0) + w
    return max(tot.items(), key=lambda kv: kv[1])[0]


TAUS = [0.0, 0.15, 0.3, 0.5, 0.75, 1.0, 1.5, 2.5]
MUS = [0.0, 0.4, 0.8, 1.4, 2.2, 4.0]
MU = 0.8            # the verifier used in the second table
TAU_STUDY = 1.5     # where coverage peaks; the third table is run here
Q = ranking_accuracy(MU)

print(f"{N_PROB} problems, {R} latent routes each, pools of {NS} samples.")
print(f"A route reaches the correct answer with probability {P_CORRECT_ROUTE:.0%};")
print("wrong routes are grouped onto 4 distinct wrong answers, so a wrong answer")
print("can be reached several ways too. tau is the sampling temperature.")
print()
print(f"{'tau':>7}{'single sample':>16}{'coverage':>11}{'majority vote':>16}")
print("-" * 50)

pools, single, cov, vt = {}, {}, {}, {}
for tau in TAUS:
    P = sample_pool(tau)
    pools[tau] = P
    single[tau] = float(np.mean(P[:, 0] == 0))
    cov[tau] = float(np.mean((P == 0).any(1)))
    vt[tau] = float(np.mean(np.array([vote(P[i]) for i in range(N_PROB)]) == 0))
    print(f"{tau:>7.2f}{single[tau]:>16.1%}{cov[tau]:>11.1%}{vt[tau]:>16.1%}")

best_tau = max(TAUS, key=lambda t: vt[t])
best_single = max(TAUS, key=lambda t: single[t])

print()
print()
print("Four ways to use the same pool. Throughout, the verifier ranks a")
print(f"correct sample above an incorrect one {Q:.0%} of the time.")
print()
print(f"{'tau':>7}{'single':>10}{'vote':>10}{'verifier':>11}{'weighted':>11}"
      f"{'coverage':>11}")
print(f"{'':>7}{'':>10}{'':>10}{'argmax':>11}{'vote':>11}{'':>11}")
print("-" * 60)

table = {}
for tau in TAUS:
    P = pools[tau]
    row = {
        "single": single[tau],
        "vote": vt[tau],
        "argmax": float(np.mean([agg_argmax(P[i], MU) == 0
                                 for i in range(N_PROB)])),
        "weighted": float(np.mean([agg_weighted(P[i], MU) == 0
                                   for i in range(N_PROB)])),
    }
    table[tau] = row
    print(f"{tau:>7.2f}{row['single']:>10.1%}{row['vote']:>10.1%}"
          f"{row['argmax']:>11.1%}{row['weighted']:>11.1%}{cov[tau]:>11.1%}")

print()
print()
print(f"How each rule responds to verifier quality, at tau={TAU_STUDY}")
print("-- the temperature at which coverage peaks.")
print()
print(f"{'verifier':>10}{'ranks correct':>15}{'vote':>10}{'argmax':>11}"
      f"{'weighted':>11}")
print(f"{'mu':>10}{'above wrong':>15}{'':>10}{'':>11}{'':>11}")
print("-" * 57)
P = pools[TAU_STUDY]
byq, qof = {}, {}
for mu in MUS:
    qq = ranking_accuracy(mu)
    qof[mu] = qq
    r = {"vote": vt[TAU_STUDY],
         "argmax": float(np.mean([agg_argmax(P[i], mu) == 0
                                  for i in range(N_PROB)])),
         "weighted": float(np.mean([agg_weighted(P[i], mu) == 0
                                    for i in range(N_PROB)]))}
    byq[mu] = r
    print(f"{mu:>10.1f}{qq:>15.1%}{r['vote']:>10.1%}{r['argmax']:>11.1%}"
          f"{r['weighted']:>11.1%}")

print(f"""
The first table is self-consistency working, and the first row is why it works.

At tau=0 the generator is greedy: it follows the single highest-weight route
every time. All {NS} samples are identical, coverage equals single-sample
accuracy at {cov[0.0]:.1%}, and the vote returns exactly what one sample
returned. A pool of forty copies of one answer holds no more information than
the answer does.

Raise the temperature and the vote pulls away from the single sample. At
tau={TAU_STUDY} one sample is {single[TAU_STUDY]:.1%} accurate and the vote is
{vt[TAU_STUDY]:.1%} -- {vt[TAU_STUDY] - single[TAU_STUDY]:+.1%} against a sample
from the same distribution, and {vt[TAU_STUDY] - vt[0.0]:+.1%} against greedy
decoding.

The mechanism is the one cite:wang2023selfconsistency names, and this listing is
built so it is the only mechanism available. Several routes reach the correct
answer. Each one individually may be less likely than the single best wrong
route -- which is exactly what greedy decoding follows -- but their probabilities
SUM when you count answers instead of paths. Voting marginalises over the latent
route (eq:marginalise-over-paths); greedy decoding maximises over it. The two
disagree whenever the correct answer is reachable more ways than the most likely
wrong one.

Nothing here involves the model being more careful, or checking its work, or
noticing a mistake. It is an argmax-versus-sum distinction, and that is the whole
of self-consistency. Keep that in mind for the next listing, where a technique
that DOES ask the model to check its work behaves very differently.

The temperature column contains the configuration error worth knowing about.
Single-sample accuracy is best at tau={best_single}
({single[best_single]:.1%}) and falls monotonically -- {single[2.5]:.1%} by
tau=2.5, because high temperature derails chains outright. The vote moves the
other way, {vt[0.0]:.1%} to {vt[2.5]:.1%}, and flattens out past tau={TAU_STUDY}
rather than turning over sharply within the range swept.

So the two objectives want different settings, and **a system whose temperature
was tuned for single-sample accuracy is at the wrong setting for
self-consistency** -- worth {vt[TAU_STUDY] - vt[best_single]:+.1%} here. Coverage
peaks at tau={TAU_STUDY} ({cov[TAU_STUDY]:.1%}) and then declines, which is the
signal to watch: past that point extra temperature is destroying correct samples
rather than diversifying them, and the vote's flatness is hiding it.

The second table adds a verifier and asks what else the pool is worth.

The pattern in the argmax column is the important one. It rises to
{table[1.0]['argmax']:.1%} at tau=1.0 and then FALLS to {table[2.5]['argmax']:.1%}
at tau=2.5, while coverage over the same range is essentially flat. Picking the
single highest-scoring sample is an extreme-value problem: with {NS} samples and
a verifier that is right {Q:.0%} of the time, the top score increasingly belongs
to whichever wrong sample got the luckiest draw. More diversity gives it more
chances. **Selecting by argmax over a large pool optimises against the verifier's
errors**, which is the effect ch:rsn-test-time-compute warned about, measured.

Weighted voting -- summing verifier scores per distinct answer rather than
trusting a single top score -- peaks at {table[TAU_STUDY]['weighted']:.1%} and is
ahead of argmax everywhere from tau=0.75 up. It is harder to fool because one
lucky score has to outweigh the accumulated support of an answer several samples
reached.

The third table sweeps verifier quality at fixed temperature, and it is where the
recommendation comes from -- including its limit.

With a useless verifier (mu=0, ranking correct above incorrect
{qof[0.0]:.1%} of the time -- chance), argmax collapses to {byq[0.0]['argmax']:.1%},
below even a single sample, because it is now deliberately selecting the most
extreme noise draw. Weighted voting gets {byq[0.0]['weighted']:.1%}. The plain
vote gets {byq[0.0]['vote']:.1%}.

That last comparison is the honest limit of the recommendation, and it is not
what I expected to find. Weighted voting does NOT degrade gracefully into plain
voting when the verifier is worthless -- it degrades into a randomly-weighted
vote, which is a noisier estimator of the same thing, and it loses
{byq[0.0]['vote'] - byq[0.0]['weighted']:.1%} for the privilege.

From mu=0.4 upward -- a verifier that ranks correctly {qof[0.4]:.1%} of the time,
which is a low bar -- weighted voting leads: {byq[0.4]['weighted']:.1%} against
the vote's {byq[0.4]['vote']:.1%} and argmax's {byq[0.4]['argmax']:.1%}. It stays
ahead of argmax until the verifier is nearly perfect ({qof[2.2]:.1%}), where the
two converge, and at mu=4.0 argmax is marginally ahead at
{byq[4.0]['argmax']:.1%} against {byq[4.0]['weighted']:.1%}.

So the rule is not "always weight". It is: measure the verifier's ranking
accuracy, and if it is meaningfully above chance, weight; if it is at chance,
count. Argmax is correct only when the verifier is close to perfect, which in
practice means an executable check rather than a learned one -- and it is the
default in most implementations, which is the wrong default for every learned
verifier in the middle of this table.

One boundary on all of it. Every number here is capped by the coverage column,
and coverage is a property of the generator. Aggregation decides how much of what
sampling produced you actually collect; it cannot produce anything.""")
