# -*- coding: utf-8 -*-
# Extracted from: Chapter 150 — Process versus Outcome Supervision
# Source: src/.../ch150-process-supervision.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""What outcome supervision actually selects for.

cite:uesato2022process ran the first careful comparison of the two supervision
signals and found something more specific than "process supervision is better":
pure outcome supervision gives SIMILAR final-answer error rates with LESS label
supervision, while process supervision is what buys correct reasoning -- reasoning
error among final-answer-correct solutions fell from 14.0% to 3.4%.

Two different criteria, two different answers. This listing reproduces the shape
of that result on a task where soundness and correctness can be measured
separately (eq:lucky-chain).

The task: a chain of k additions. A step is either correct or wrong; a wrong step
adds the wrong number. The chain's ANSWER is right when the total is right, which
happens when every step is right -- and also when two wrong steps happen to
cancel. Those lucky chains are the whole story, because outcome supervision
labels them positive.
"""
import numpy as np

rng = np.random.default_rng(907)

K = 6                      # steps per chain
VALS = np.arange(-3, 4)    # the numbers a step can add
N_TRAIN = 24000
N_TEST = 4000
NMAX = 32
P_STEP_OK = 0.80           # chance the generator gets a step right
SIGMA = 0.9                # how noisily a grader can read one step
LABEL_NOISE = 0.08         # disagreement rate on step-level annotation


def make(n, p_ok=P_STEP_OK, spread=1):
    """A chain of K steps. `spread` scales how far a wrong step strays: at
    spread=1 wrong steps draw from the same small range and often cancel; at
    higher spread they scatter and cancellation becomes rare."""
    true_steps = rng.choice(VALS, size=(n, K))
    ok = rng.random((n, K)) < p_ok
    wrong = rng.choice(VALS * spread, size=(n, K))
    taken = np.where(ok, true_steps, wrong)
    sound = ok.all(1)                       # every step correct
    correct = taken.sum(1) == true_steps.sum(1)
    # What a grader can actually read off the page. A step is checkable but not
    # perfectly: the derivation is prose, and reading it is noisy. Both reward
    # models see this and only this -- neither is handed the answer key.
    obs = (taken - true_steps) + SIGMA * rng.normal(size=(n, K))
    # An outcome grader reads the FINAL answer, once, with its own noise -- it
    # does not have to add up six noisy step readings. Giving it a clean single
    # observation of the thing it is graded on is what makes the comparison
    # fair: each signal is read as directly as it is actually available.
    obs_fin = (taken.sum(1) - true_steps.sum(1)) + SIGMA * rng.normal(size=n)
    return true_steps, taken, ok, sound, correct, obs, obs_fin


def feats_chain(taken, obs, obs_fin):
    """What a reward model sees for a whole chain: the emitted steps and a noisy
    reading of each one, plus the running totals. Nothing here states whether a
    step was right -- that has to be inferred, imperfectly, from `obs`."""
    run = np.cumsum(taken, axis=1)
    orun = np.cumsum(obs, axis=1)
    return np.concatenate([taken, obs, run, orun,
                           obs_fin[:, None], np.abs(obs_fin)[:, None],
                           obs.sum(1)[:, None], np.abs(obs).sum(1)[:, None],
                           np.ones((len(taken), 1))], axis=1)


def feats_step(taken, obs):
    """The same noisy reading, one row per (chain, step). This is the view a
    step-level annotator has: this step, in its place, and how it looks."""
    n = len(taken)
    pos = np.tile(np.arange(K), (n, 1))
    prev = np.concatenate([np.zeros((n, 1)), np.cumsum(taken, 1)[:, :-1]], 1)
    return np.stack([taken, obs, np.abs(obs), pos, prev,
                     np.ones((n, K))], axis=2).reshape(n * K, 6)


def rf(X, W, B):
    return np.cos(X @ W + B)


def fit(X, y, nf=300, lam=1e-2):
    """Random-feature ridge regression: enough capacity to fit either signal,
    little enough to make the fit depend on what the labels say."""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    W = rng.normal(size=(X.shape[1], nf)) * 0.7
    B = rng.uniform(0, 2 * np.pi, nf)
    F = rf((X - mu) / sd, W, B)
    c = np.linalg.solve(F.T @ F + lam * np.eye(nf), F.T @ y)
    return (mu, sd, W, B, c)


def score(model, X):
    mu, sd, W, B, c = model
    return rf((X - mu) / sd, W, B) @ c


tr_true, tr_taken, tr_ok, tr_sound, tr_correct, tr_obs, tr_fin = make(N_TRAIN)

# OUTCOME reward model: one label per chain -- did the final answer come out
# right? It never learns which step was at fault, because nobody told it.
ORM = fit(feats_chain(tr_taken, tr_obs, tr_fin), tr_correct.astype(float))

# PROCESS reward model: one label per STEP, K times the annotation, and the
# labels disagree with the truth LABEL_NOISE of the time, as human step-level
# annotation does.
step_lab = tr_ok.reshape(-1).astype(float)
flip = rng.random(step_lab.shape) < LABEL_NOISE
step_lab = np.where(flip, 1.0 - step_lab, step_lab)
PRM = fit(feats_step(tr_taken, tr_obs), step_lab)


def prm_chain_score(taken, obs):
    """A chain's process score is its weakest step, which is what a process
    reward model is for: one bad step invalidates the derivation."""
    s = score(PRM, feats_step(taken, obs)).reshape(len(taken), K)
    return s.min(1)


def orm_chain_score(taken, obs, obs_fin):
    return score(ORM, feats_chain(taken, obs, obs_fin))


print(f"Chains of {K} additions. The generator gets a step right "
      f"{P_STEP_OK:.0%} of the time.")
print()
print("How often does a chain come out right, and how often is it SOUND?")
print()
print(f"{'quantity':>44}{'value':>10}")
print("-" * 54)
print(f"{'chains with every step correct (sound)':>44}"
      f"{float(tr_sound.mean()):>10.1%}")
print(f"{'chains with the correct final answer':>44}"
      f"{float(tr_correct.mean()):>10.1%}")
lucky = tr_correct & ~tr_sound
print(f"{'correct answer from an UNSOUND chain':>44}{float(lucky.mean()):>10.1%}")
print(f"{'  ... as a share of correct answers':>44}"
      f"{float(lucky.sum() / tr_correct.sum()):>10.1%}")

print()
print()
print("Both reward models, used to pick the best of n sampled chains.")
print("Two criteria: is the ANSWER right, and is the DERIVATION sound?")
print()
print(f"{'':>10}{'outcome RM':>25}{'process RM':>25}")
print(f"{'best of n':>10}{'answer':>12}{'sound':>13}{'answer':>12}{'sound':>13}")
print("-" * 60)

# One pool of NMAX chains per problem, scored once and reused at every budget,
# so the columns are comparable rather than independently noisy.
po_true, po_taken, po_ok, po_sound, po_correct, po_obs, po_fin = make(N_TEST * NMAX)
SO = orm_chain_score(po_taken, po_obs, po_fin).reshape(N_TEST, NMAX)
SP = prm_chain_score(po_taken, po_obs).reshape(N_TEST, NMAX)
CC = po_correct.reshape(N_TEST, NMAX)
SS = po_sound.reshape(N_TEST, NMAX)
BUD = (1, 2, 4, 8, 16, 32)
ROW = np.arange(N_TEST)

res = {}
for n in BUD:
    io_ = SO[:, :n].argmax(1)
    ip = SP[:, :n].argmax(1)
    r = (float(CC[ROW, io_].mean()), float(SS[ROW, io_].mean()),
         float(CC[ROW, ip].mean()), float(SS[ROW, ip].mean()))
    res[n] = r
    print(f"{n:>10}{r[0]:>12.1%}{r[1]:>13.1%}{r[2]:>12.1%}{r[3]:>13.1%}")

print()
print()
print("cite:uesato2022process's metric: among the selections whose ANSWER is")
print("right, how many contain a reasoning error?")
print()
print(f"{'best of n':>10}{'outcome RM':>14}{'process RM':>14}")
print("-" * 38)
rerr = {}
for n in BUD:
    out = []
    for idx in (SO[:, :n].argmax(1), SP[:, :n].argmax(1)):
        c, sd = CC[ROW, idx], SS[ROW, idx]
        out.append(float(np.mean(~sd[c])) if c.any() else float("nan"))
    rerr[n] = tuple(out)
    print(f"{n:>10}{out[0]:>14.1%}{out[1]:>14.1%}")

print()
print()
print("What happens when lucky cancellation gets rarer? Same reward models,")
print("test chains whose wrong steps stray further, so errors stop cancelling.")
print()
print(f"{'':>10}{'lucky-answer':>15}{'outcome RM':>25}{'process RM':>25}")
print(f"{'spread':>10}{'rate':>15}{'answer':>12}{'sound':>13}"
      f"{'answer':>12}{'sound':>13}")
print("-" * 75)
shift = {}
NS = 16
for spread in (1, 2, 4):
    t2, k2, o2, s2, c2, ob2, of2 = make(N_TEST * NS, spread=spread)
    so = orm_chain_score(k2, ob2, of2).reshape(N_TEST, NS)
    sp = prm_chain_score(k2, ob2).reshape(N_TEST, NS)
    C, S = c2.reshape(N_TEST, NS), s2.reshape(N_TEST, NS)
    lk = float(np.mean(c2 & ~s2))
    io_, ip = so.argmax(1), sp.argmax(1)
    r = (lk, float(C[ROW, io_].mean()), float(S[ROW, io_].mean()),
         float(C[ROW, ip].mean()), float(S[ROW, ip].mean()))
    shift[spread] = r
    print(f"{spread:>10}{r[0]:>15.1%}{r[1]:>12.1%}{r[2]:>13.1%}"
          f"{r[3]:>12.1%}{r[4]:>13.1%}")

# How good is the process reward model at its own job? Report it, so the
# reader can tell a limitation of the method from a limitation of this fit.
_t, _k, _o, _s, _c, _ob, _f = make(20000)
_ps = score(PRM, feats_step(_k, _ob))
_lab = _o.reshape(-1)
_auc = float(np.mean(rng.choice(_ps[_lab], 200000) >
                     rng.choice(_ps[~_lab], 200000)))
print()
print()
print(f"The process reward model's own quality: it ranks a correct step above")
print(f"an incorrect one {_auc:.1%} of the time. It is good, not perfect, which")
print(f"is the realistic case -- step labels are noisy ({LABEL_NOISE:.0%} here)")
print(f"and a step is read imperfectly.")

print(f"""
The first table sets up the whole problem in one number:
{float(lucky.sum() / tr_correct.sum()):.1%} of the chains with a correct final
answer contain at least one wrong step.

Those are the chains outcome supervision labels POSITIVE. It has one bit per
chain and that bit says the total came out right, so a derivation that went wrong
twice and cancelled is indistinguishable, to that signal, from one that never
went wrong at all (eq:lucky-chain).

The second table is what the two reward models select for. At best-of-32 the
process model reaches {res[32][2]:.1%} on answers and {res[32][3]:.1%} on
soundness; the outcome model reaches {res[32][0]:.1%} and {res[32][1]:.1%}. The
process model leads on both, at {K}x the annotation cost.

The third table is cite:uesato2022process's own metric, and it is where this
listing FAILS to reproduce their result, which is worth stating plainly rather
than burying. Among selections whose answer is right, the fraction containing a
reasoning error is {rerr[32][0]:.1%} for the outcome model and {rerr[32][1]:.1%}
for the process model. Uesato et al. report 14.0% falling to 3.4%.

The gap here is real but small, and the reason is visible in the process model's
own quality: it ranks a correct step above an incorrect one {_auc:.1%} of the
time, not {1.0:.0%}. A chain's score is its weakest step, so six imperfect
readings compound, and a process model at this quality cannot cleanly separate
sound chains from lucky ones. PRM800K is 800,000 human step labels; this is a
ridge fit on noisy synthetic ones. **The direction reproduces and the magnitude
does not, and the magnitude is a function of how good your process model is** --
which is the first thing to measure before budgeting for one.

The fourth table is the one that changed my mind about how to present this
chapter, because it runs against the way the process-versus-outcome question is
usually framed.

As `spread` rises, wrong steps stray further and stop cancelling, so the
lucky-answer rate falls from {shift[1][0]:.1%} to {shift[4][0]:.1%}. The reward
models are unchanged; only the test distribution moves.

At spread=1 the process model leads on answers by
{shift[1][3] - shift[1][1]:+.1%}. At spread=2 the lead is
{shift[2][3] - shift[2][1]:+.1%}. At spread=4 it is
{shift[4][3] - shift[4][1]:+.1%} -- the outcome model is AHEAD.

The same reversal happens on soundness: {shift[1][4] - shift[1][2]:+.1%},
{shift[2][4] - shift[2][2]:+.1%}, {shift[4][4] - shift[4][2]:+.1%}.

So the advantage of process supervision is not a property of process supervision.
**It is proportional to the rate at which a wrong derivation reaches a right
answer.** When that rate is {shift[1][0]:.1%}, the outcome label is a badly
corrupted proxy for soundness and the process label is worth paying for. When it
is {shift[4][0]:.1%}, the outcome label is very nearly a soundness label already,
and paying {K}x per example buys a slightly worse model -- worse because the
process model's own noise is now the dominant error and the outcome signal has
stopped being misleading.

That gives the decision rule this chapter exists to produce, and it is a
measurement you can make on your own data before spending anything: sample
solutions, grade a few hundred of them BOTH ways, and compute the fraction of
correct answers that came from faulty reasoning. That number is the expected
value of process supervision. Everything else -- how long your chains are, how
many discrete answers there are, how much the steps interact -- matters only
through it.

It also explains why the literature disagrees rather than one side being wrong.
cite:uesato2022process worked on GSM8K, where solutions are short and a wrong
step usually produces a wrong number, and found outcome supervision competitive
on final answers at lower cost. cite:lightman2023verify worked on MATH, where
derivations are long and there is far more room for a flawed path to arrive
somewhere right, and found process supervision substantially better. Those are
the two ends of this table.

One caveat on the whole framing. Both models here are selectors, scoring
candidates the generator produced. Using the same signals to TRAIN the generator
is a different problem with a worse failure mode, because an outcome-trained
generator is being optimised to produce lucky chains rather than merely to have
them accepted -- and it will find ways to be lucky that no test distribution in
this listing contains. That is ch:rsn-benchmarks's subject, and it is the reason
the reasoning-error metric is worth tracking even when the answer metric says
the two signals are equivalent.""")
