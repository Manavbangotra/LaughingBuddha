# -*- coding: utf-8 -*-
# Extracted from: Chapter 134 — Synthetic Data and Data Quality
# Source: src/.../ch134-synthetic-data.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Diversity collapse, measured -- and why quality filtering accelerates it.

cite:wang2023selfinstruct made instruction data a compute problem, and the
compute is real. The hazard is that a model generating its own training data
samples from its own distribution, so each generation inherits the previous
generation's modes and, crucially, its SAMPLING ERROR in the tail
(eq:collapse-recursion).

This listing runs the recursion. A true distribution has eight modes with
Zipf-like weights; a model is fitted, sampled, refitted on its own samples, and so
on. Three regimes are compared: plain resampling, resampling with a quality filter
that keeps the highest-likelihood samples, and resampling with a small share of
real data mixed back in.

The middle regime is the one worth the listing. Filtering for quality is the
standard remedy for synthetic-data noise, and it is applied to a distribution
whose problem is that the tail is disappearing.
"""
import numpy as np

rng = np.random.default_rng(173)

K = 8
MEANS = np.linspace(-9.0, 9.0, K)
SIG = 0.55
W_TRUE = np.array([0.30, 0.22, 0.16, 0.12, 0.08, 0.06, 0.04, 0.02])
N_GEN = 4000
GENERATIONS = 8


def sample_true(n):
    c = rng.choice(K, size=n, p=W_TRUE)
    return MEANS[c] + SIG * rng.normal(size=n)


def em_fit(x, k=K, iters=60):
    """Fit a k-component 1-D Gaussian mixture. Components that stop attracting
    mass are what 'losing a mode' looks like mechanically."""
    w = np.full(k, 1.0 / k)
    mu = np.quantile(x, np.linspace(0.05, 0.95, k))
    var = np.full(k, x.var() / k + 1e-3)
    for _ in range(iters):
        d = x[:, None] - mu[None, :]
        logp = (-0.5 * d ** 2 / var[None, :] - 0.5 * np.log(2 * np.pi * var)
                [None, :] + np.log(np.maximum(w, 1e-300))[None, :])
        m = logp.max(axis=1, keepdims=True)
        r = np.exp(logp - m); r /= r.sum(axis=1, keepdims=True)
        nk = r.sum(axis=0) + 1e-10
        w = nk / len(x)
        mu = (r * x[:, None]).sum(axis=0) / nk
        var = np.maximum((r * (x[:, None] - mu[None, :]) ** 2).sum(axis=0) / nk,
                         1e-3)
    return w, mu, var


def sample_model(model, n):
    w, mu, var = model
    c = rng.choice(len(w), size=n, p=w / w.sum())
    return mu[c] + np.sqrt(var[c]) * rng.normal(size=n)


def logpdf(model, x):
    w, mu, var = model
    d = x[:, None] - mu[None, :]
    lp = (-0.5 * d ** 2 / var[None, :] - 0.5 * np.log(2 * np.pi * var)[None, :]
          + np.log(np.maximum(w, 1e-300))[None, :])
    m = lp.max(axis=1, keepdims=True)
    return (m[:, 0] + np.log(np.exp(lp - m).sum(axis=1)))


def modes_alive(x, tol=2.0, floor=0.005):
    """A true mode counts as alive if at least `floor` of the sample lands
    within `tol` standard deviations of it."""
    hits = np.abs(x[:, None] - MEANS[None, :]) < tol * SIG
    return int((hits.mean(axis=0) >= floor).sum())


def tail_mass(x):
    """Share of samples belonging to the four RAREST true modes -- 20% of the
    true distribution, and the part that vanishes first."""
    near = np.abs(x[:, None] - MEANS[None, :]).argmin(axis=1)
    return float(np.isin(near, np.arange(K // 2, K)).mean())


def run(real_share=0.0, keep=1.0, n_gen=N_GEN):
    x = sample_true(n_gen)
    out = [(modes_alive(x), float(x.std()), tail_mass(x))]
    for _ in range(GENERATIONS):
        model = em_fit(x)
        n_syn = int(n_gen * (1 - real_share))
        draw = sample_model(model, int(n_syn / keep))
        if keep < 1.0:                       # keep the highest-likelihood share
            thr = np.quantile(logpdf(model, draw), 1 - keep)
            draw = draw[logpdf(model, draw) >= thr][:n_syn]
        x = draw if real_share == 0 else np.concatenate(
            [draw[:n_syn], sample_true(n_gen - n_syn)])
        out.append((modes_alive(x), float(x.std()), tail_mass(x)))
    return out


REGIMES = [
    ("pure synthetic, 4,000 per generation", dict()),
    ("pure synthetic, 60 per generation", dict(n_gen=60)),
    ("filtered for quality: keep top 70%", dict(keep=0.7)),
    ("filtered for quality: keep top 40%", dict(keep=0.4)),
    ("filtered top 70%, but 10% real data mixed in",
     dict(keep=0.7, real_share=0.10)),
    ("filtered top 70%, but 30% real data mixed in",
     dict(keep=0.7, real_share=0.30)),
]

print(f"True distribution: {K} modes, weights {W_TRUE.min():.0%} to "
      f"{W_TRUE.max():.0%}. {N_GEN:,} examples per generation.\n")

results = {}
for name, kw in REGIMES:
    r = run(**kw)
    results[name] = r
    print(f"{name}")
    print(f"  {'generation':>11}" + "".join(f"{g:>6}" for g in
                                            range(GENERATIONS + 1)))
    print(f"  {'modes alive':>11}" + "".join(f"{v[0]:>6}" for v in r))
    print(f"  {'std':>11}" + "".join(f"{v[1]:>6.2f}" for v in r))
    print(f"  {'tail mass':>11}" + "".join(f"{v[2]:>6.0%}" for v in r))
    print()

pure = results["pure synthetic, 4,000 per generation"]
small = results["pure synthetic, 60 per generation"]
f70 = results["filtered for quality: keep top 70%"]
f40 = results["filtered for quality: keep top 40%"]
mix = results["filtered top 70%, but 10% real data mixed in"]
mix30 = results["filtered top 70%, but 30% real data mixed in"]

print(f"""
The first block is not what the folklore predicts, and it is worth sitting with.
Eight generations of a model trained on nothing but its own output, and the
distribution is essentially intact: {pure[-1][0]} of {pure[0][0]} modes alive,
spread {pure[0][1]:.2f} to {pure[-1][1]:.2f}, tail mass {pure[0][2]:.0%} to
{pure[-1][2]:.0%}.

At {N_GEN:,} examples per generation the sampling error is small enough that each
refit recovers the distribution it was drawn from, so the recursion has nothing
to compound. Self-training is not intrinsically degenerative.

The second block shows the mechanism that IS intrinsic, at a sample size small
enough to see it. At 60 examples per generation the same procedure drifts down to
{small[-1][0]} modes, and note how NOISY that column is -- modes disappear and
come back, because at this sample size a rare mode's survival is close to a coin
flip each round (eq:collapse-recursion). This is real degradation, it is slow,
and it is a sample-size problem, which means it is the one you can buy your way
out of.

Now the blocks that collapse at {N_GEN:,} examples, where pure recursion did not.

Quality filtering is the standard hygiene for synthetic data, and the standard
implementation keeps the samples the model scores highest. At keep-70%, eight
generations leave {f70[-1][0]} mode alive with a spread of {f70[-1][1]:.2f}. At
keep-40%, {f40[-1][0]} mode and {f40[-1][1]:.2f}. The distribution has become a
point.

Look at how fast the tail goes: {f70[1][2]:.0%} after ONE generation in both
filtered regimes, while pure recursion still had {pure[-1][2]:.0%} of it after
eight.

The filter did not malfunction. It did exactly what it was asked, and what it was
asked was to delete the tail. A rare mode has low likelihood BECAUSE it is rare,
so a likelihood-based quality filter is a rareness filter wearing different
clothing (eq:quality-filters-rareness). Each pass concentrates mass on what the
model already does confidently, which is a precise description of the failure the
filter was installed to prevent.

That is the finding to carry out of this listing, and it inverts the usual
advice. In a synthetic-data pipeline the dangerous component is not the
recursion. It is the filter placed there to make the recursion safe.

The generalisation past this toy is direct. A reward model, an LLM-as-judge, a
perplexity threshold, a heuristic for well-formedness -- each scores typicality
alongside quality and cannot separate the two, because inside the model's own
distribution an unusual-but-correct example and an unusual-but-wrong one look
alike. Filtering synthetic data for quality reliably raises the average example
and reliably narrows the set, and only the first half is usually measured.

The last two blocks are the fix, and they are more interesting than a fix usually
is, because they show its limit.

Keep the identical keep-70% filter and hold back 10% of each generation as real
data: {mix[-1][0]} modes survive instead of {f70[-1][0]}, and the spread holds at
{mix[-1][1]:.2f} instead of {f70[-1][1]:.2f}. At 30% real, all {mix30[-1][0]}
modes survive. The real fraction is the only term in the recursion that does not
depend on the model's current beliefs, so it is the only term that can
reintroduce a mode the model has already lost (eq:grounding-is-the-fix).

But read the tail row before declaring victory. True tail mass is
{W_TRUE[K//2:].sum():.0%}. With 10% real it sits at {mix[-1][2]:.0%}; with 30%
real, {mix30[-1][2]:.0%}. Grounding restores the MODES and does not restore their
MASS, because the filter is still deleting the tail every round and the real data
is only refilling a fraction of it.

So the honest summary is a hierarchy rather than a fix. Grounding prevents
collapse; it does not make aggressive filtering safe. If you need the tail -- and
the previous chapter's coverage argument says you do -- the filter is the thing to
change, not the amount of real data used to compensate for it.

Which gives the rule for building these pipelines. Generate FROM something rather
than from the model's prior: a document, a log, a schema, a customer record, a
real ticket. The synthetic examples then inherit that material's diversity instead
of the model's. And filter for CORRECTNESS against that source rather than for
typicality under the generator, because those two are the same operation only when
the generator is already right about everything -- in which case there was nothing
to fix.""")
