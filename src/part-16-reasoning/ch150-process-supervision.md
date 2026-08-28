---
id: rsn-supervision
number: 150
part: XVI
tier: full
status: draft
requires: [verifier-quality-ceiling, critic-error-correlation,
           per-step-error-compounding]
provides: [lucky-chain-rate, outcome-signal-bias, soundness-versus-accuracy,
           label-budget-crossover, imputed-step-labels,
           process-reward-aggregation, supervision-decision-rule]
citations: [lightman2023verify, uesato2022process, zelikman2022star,
            cobbe2021gsm8k, snell2024testtime, deepseek2025r1,
            huang2024selfcorrect, brown2024monkeys]
---

## 1. Learning Objectives

By the end of this chapter you will be able to name the single measurement that
decides whether process supervision is worth buying, and take it on your own
data; explain why outcome supervision is a *biased* signal rather than a noisy
one, and what that implies about collecting more of it; compare two supervision
signals at equal annotation budget rather than equal example count; predict when
imputing step labels from outcomes is a good deal and when it makes things worse;
and separate two criteria — is the answer right, is the derivation sound — that a
single accuracy number fuses.

## 2. Why This Matters

Every chapter of {{part:16}} so far has ended in the same place. Test-time compute
is bounded by the selector ({{ch:rsn-test-time-compute}}). Aggregation is bounded
by the verifier ({{ch:rsn-self-consistency}}). Reflection fails because the critic
is the solver ({{ch:rsn-self-consistency}} again). The verifier is the component
everything else waits on, and this chapter is about how it gets trained.

There are two signals available and they differ by a factor of $k$ in cost. You
can label the *outcome* — did the final answer come out right — which is one bit
per solution and often free, because you already have an answer key. Or you can
label the *process* — was each step right — which is $k$ bits per solution and
requires someone who can grade a derivation. {{cite:lightman2023verify}} released
$800{,}000$ human step-level labels to do this, which conveys the scale of the
commitment.

The literature appears to disagree about whether it is worth it, and the
disagreement is more interesting than either position.
{{cite:lightman2023verify}} found process supervision substantially better on
MATH. {{cite:uesato2022process}} found, on GSM8K, that *pure outcome supervision
produces similar final-answer error rates with less label supervision* — while
also finding that process supervision is what buys correct reasoning, cutting
reasoning error among final-answer-correct solutions from $14.0\%$ to $3.4\%$.

Both are right, and {{sec:9-practical-example}} identifies the variable that
separates them. It is the rate at which a wrong derivation reaches a right answer.
When that rate is high, the outcome label is a badly corrupted proxy for soundness
and process supervision is worth a great deal. When it is low, the outcome label
*is* nearly a soundness label, and paying $k\times$ buys a slightly worse model.
That is one number, it is cheap to measure on your own data, and it is the whole
decision.

The chapter also produces a result that changed how I would advise a team.
Outcome supervision does not merely learn more slowly — it *plateaus*. In
{{sec:9-practical-example}} a fourfold increase in outcome labels moves accuracy
by $0.6$ points while the same increase in process labels moves it by $3.1$.
Outcome labels are biased, not noisy, and more of a biased signal estimates the
wrong target more precisely.

## 3. Prerequisites

You need {{ch:rsn-test-time-compute}}'s verifier-quality result — that selection
accuracy is bounded by the verifier and that its value grows with the sample
budget — because this chapter is about producing that verifier.

You need {{ch:rsn-self-consistency}}'s distinction between a critic's accuracy and
its error structure. The same lesson recurs here in a different form: a reward
model's overall accuracy does not tell you what it selects for.

From {{ch:rsn-cot}}, the compounding result explains why chains contain multiple
independent opportunities to go wrong, which is what makes lucky cancellation
possible at all.

Standard supervised learning is assumed: labels, bias versus variance, and what
it means for a model to be well specified. The bias-versus-variance distinction
does real work in {{sec:5-formal-explanation}}.

## 4. Intuitive Explanation

Imagine grading a stack of maths homework with a very limited amount of time.

The cheap option is to check the final answer against the key. One mark per page,
fast, and you already have the key. The expensive option is to read the working
and mark each step. Six times the effort on a six-step problem, and it requires a
grader who understands the subject rather than one who can compare two numbers.

Now ask what each mark actually tells you.

If a student's answer is wrong, both methods agree: something went wrong. If the
answer is right and the working is right, both agree again. The interesting case
is the third one — the answer is right and the working is not. Two errors that
cancelled. A wrong method that happens to give the right number on this input. A
step skipped that did not matter here.

Outcome grading marks that page correct. It has no way not to; the only thing it
looked at was the number at the bottom.

That single case is the entire subject of this chapter, and everything follows
from how often it happens. If it almost never happens, outcome grading is nearly
as good as reading the working and costs a sixth as much, so you should obviously
use it. If it happens a quarter of the time, then a quarter of your positive
labels are telling a model that faulty reasoning is fine.

Here is the part that is easy to get wrong, and it is why the chapter has a
listing rather than an argument. You might expect that outcome labels are just
*noisier* — that they are right on average and you can fix them by collecting
more. They are not. They are wrong in a consistent direction: they systematically
approve of exactly the lucky chains, and they approve of them every time. A model
trained on more of them learns that pattern more confidently.

That is the difference between noise and bias, and it decides the shape of the
whole trade. With noise, more data helps and eventually wins. With bias, more data
converges on the wrong answer, and {{sec:9-practical-example}} shows the outcome
model's accuracy going flat while the process model's keeps climbing.

There is one more idea in the chapter and it is the practical one. Step labels are
expensive, so the obvious shortcut is to *manufacture* them: take the solutions
whose answers were right and declare all their steps correct. That is free — it
uses only the answer key you already have — and it is the labelling rule inside
{{cite:zelikman2022star}}'s bootstrapping loop and inside most process reward
models trained without human annotation.

It works surprisingly well and it fails in an instructive way. It is wrong on
exactly the lucky chains, marking their faulty steps as good. So it inherits a
weakened version of the bias it was introduced to remove — which means it helps
most in the regime where the outcome signal was worst, and can actively hurt in
the regime where the outcome signal was already fine.

## 5. Formal Explanation

Let a solution be a sequence of $k$ steps, each correct or not. Write $s_i \in
\{0,1\}$ for step $i$'s correctness. Two derived quantities:

$$\text{sound} = \prod_{i=1}^{k} s_i, \qquad \text{correct} = \mathbb{1}\big[\,a(\text{steps}) = a^{*}\,\big]$$ (eq:sound-versus-correct)

Soundness implies correctness for a deterministic task. The converse does not
hold, and the gap is the object of interest:

$$\ell \;=\; \Pr\big[\,\text{correct} \wedge \neg\,\text{sound}\,\big] \Big/ \Pr[\,\text{correct}\,]$$ (eq:lucky-chain)

$\ell$ is the *lucky-chain rate*: the share of right answers that came from wrong
reasoning. It rises with $k$, because more steps mean more opportunities to err
and to cancel, and falls as the answer space grows, because a random error is less
likely to land on the right value.

Outcome supervision trains a model on $(\text{solution}, \text{correct})$ pairs.
Process supervision trains on $(\text{step}, s_i)$ pairs, $k$ per solution. Now
write what each one is estimating. The outcome-supervised model converges to:

$$f_{\text{out}}(\text{solution}) \;\to\; \Pr[\,\text{correct} \mid \text{solution}\,]$$ (eq:outcome-target)

which is the right target if correctness is what you want, and is a *biased*
estimate of soundness with bias exactly $\ell$ on the positive class. This is the
key structural fact:

$$\mathbb{E}\big[f_{\text{out}} \mid \text{sound}\big] - \mathbb{E}\big[f_{\text{out}} \mid \neg\,\text{sound} \wedge \text{correct}\big] \;=\; 0$$ (eq:outcome-signal-bias)

The outcome signal assigns the same expected score to a sound chain and to a lucky
one, *by construction*, because they carry the same label. No amount of data
separates them. That is what makes this bias rather than noise, and it is why
{{sec:9-practical-example}} finds a plateau: the model reaches
{{eq:outcome-target}} and stops, because {{eq:outcome-target}} is all the signal
contains.

Process supervision has no such term. Its target is $\Pr[s_i = 1 \mid \text{step}]$,
which is a property of the step and does not reference the other steps or the
final answer. It pays instead in *variance*: at a fixed label budget $B$ you get
$B$ outcome-labelled solutions or $B/k$ process-labelled ones, so

$$n_{\text{out}} = B, \qquad n_{\text{proc}} = B/k$$ (eq:label-budget)

and the process model is fitted on $k$ times fewer examples. The comparison is
therefore bias against variance, and the standard intuition — more data wins
eventually — applies only to the variance side.

To score a whole chain from step scores, aggregate. The natural choice follows
from {{eq:sound-versus-correct}}'s product form: a chain is sound only if every
step is, so

$$F_{\text{proc}}(\text{chain}) = \min_{i} f_{\text{proc}}(\text{step}_i)$$ (eq:process-aggregation)

and the minimum is what {{sec:9-practical-example}} uses. A product of
probabilities is the other defensible choice and behaves similarly; a *mean* is
not, because it lets five good steps outvote one fatal one, which is precisely the
property you were paying to avoid.

Finally, the imputation heuristic. Set $\hat{s}_i = \text{correct}$ for every
step — all steps of a right answer marked right, all steps of a wrong answer marked
wrong. Its label error is:

$$\Pr[\hat{s}_i \neq s_i] = \underbrace{\Pr[\text{correct} \wedge \neg s_i]}_{\text{lucky chains}} + \underbrace{\Pr[\neg\,\text{correct} \wedge s_i]}_{\text{good steps in failed chains}}$$ (eq:imputation-error)

Two error terms pulling in opposite directions as $\ell$ varies, and this is the
reason the heuristic's value is non-monotone. When $\ell$ is large the first term
dominates and the imputation is still much better than the outcome signal it
replaces. When $\ell$ is small the first term vanishes but the *second* does not —
a failed chain usually has one bad step and $k-1$ good ones, all of which the
heuristic marks wrong — and it becomes worse than plain outcome supervision.

## 6. Mathematical Foundation

How large is $\ell$ in practice, and what controls it?

Model a chain of $k$ steps, each correct with probability $p$, with a wrong step
producing an error drawn from a distribution over $m$ effective values. The answer
is correct if the errors sum to zero. For a single error that requires the error
to be zero, which it is not by definition; for two errors it requires them to
cancel, with probability roughly $1/m$. Then:

$$\ell \;\approx\; \frac{\sum_{j \ge 2} \binom{k}{j} (1-p)^{j} p^{\,k-j}\, c_j}{p^{k} + \sum_{j \ge 2} \binom{k}{j}(1-p)^{j} p^{\,k-j} c_j}$$ (eq:lucky-rate-model)

where $c_j \approx m^{-(j-1)}$ is the chance that $j$ errors cancel. Three
readings, and they are the practical content of the equation.

$\ell$ grows with $k$. Longer derivations have more pairs of steps that can cancel,
and the binomial term grows faster than $p^{k}$ shrinks over the relevant range.
This is the single largest driver, and it is why MATH
({{cite:lightman2023verify}}) and GSM8K ({{cite:uesato2022process}}) land in
different regimes — the derivations differ in length by a large factor.

$\ell$ falls with $m$, the effective size of the answer space. Multiple-choice
questions have $m \approx 4$ and enormous $\ell$; open-ended numeric answers have
large $m$ and small $\ell$. **Outcome supervision on a multiple-choice task is
close to useless as a soundness signal**, and this follows from arithmetic rather
than from anything about models.

$\ell$ is not monotone in $p$. As $p \to 1$ there are no errors to cancel and
$\ell \to 0$; as $p \to 0$ almost nothing is correct and the denominator collapses
too. It peaks in the middle, which is where deployed systems live.

Now the budget comparison. Let the outcome model's asymptotic soundness error be
$\varepsilon_{\text{bias}} \propto \ell$ and the process model's variance error
scale as $\sigma^2 k / B$. Then process supervision wins whenever:

$$\frac{\sigma^{2} k}{B} \;<\; \varepsilon_{\text{bias}}(\ell)$$ (eq:supervision-crossover)

The left side falls with budget; the right side does not move. So for any $\ell >
0$ there is a budget beyond which process supervision wins, and below which
outcome supervision does. {{sec:9-practical-example}} finds that in its setting
the crossover is below the smallest budget swept — process wins everywhere,
despite buying six times fewer examples — which says $\varepsilon_{\text{bias}}$
was large relative to the variance term throughout.

That is the shape to carry: **the question is never which signal is better, it is
where your budget sits relative to $\ell$.**

## 7. Internal Mechanics

### 7.1 What a process reward model is actually scoring

A step-level label answers "is this step correct given what came before". Note
what that does *not* require: it does not require the step to be on a path to the
answer, and it does not require the preceding steps to have been right.

Both looseness matter in practice. A model that has already made an error can
continue making individually valid steps from a wrong premise, and a strict
step-correctness annotator will mark them correct — which is right by the letter
of the definition and unhelpful for selection.
{{cite:lightman2023verify}}'s annotation guidelines have to take a position on
this, and different positions produce meaningfully different reward models from
the same solutions.

The aggregation in {{eq:process-aggregation}} partly compensates: taking the
minimum means the one bad step still condemns the chain. But it also makes the
reward model maximally sensitive to its own worst-case error, which is why a
process model with modest per-step accuracy can score chains poorly. In
{{sec:9-practical-example}} the process model ranks a correct step above an
incorrect one $83.3\%$ of the time, and taking a minimum over six such judgements
is a much weaker signal than that number suggests.

### 7.2 Why the plateau is the diagnostic

If you plot selection accuracy against annotation budget on a log scale, a
variance-limited model keeps improving and a bias-limited one flattens. That shape
is the cheapest available diagnosis of which regime you are in, and it requires no
soundness labels at all — only the answer key you already have and a willingness
to train the same model at several budgets.

{{sec:9-practical-example}} shows it directly: from $7{,}200$ to $28{,}800$ outcome
labels, accuracy moves $67.1\% \to 67.7\%$. The model has learned everything the
signal contains.

### 7.3 Selection versus training

Everything above treats the reward model as a *selector*, scoring candidates the
generator produced. Using the same signal to *train the generator* — RL against a
reward model, or rejection-sampling fine-tuning — is a strictly worse situation.

A selector with a biased score picks lucky chains that the generator happened to
produce. A generator trained against that score is being optimised *to produce
them*, and it will find ways to be lucky that no evaluation distribution contains.
The bias in {{eq:outcome-signal-bias}} becomes an optimisation target rather than a
measurement error.

That is the strongest argument for process supervision that does not depend on
caring about soundness for its own sake, and it is the reason
{{cite:deepseek2025r1}}'s use of *verifiable* rewards — a real checker, not a
learned model — is a different kind of thing from RL against a learned reward
model. {{ch:rsn-tool-assisted}} takes that up.

### 7.4 Where the labels come from

{{cite:zelikman2022star}}'s loop is the template for producing reasoning
supervision without step annotation: generate rationales, keep the ones that
reached the right answer, fine-tune on those, repeat — and when the model fails,
rationalise backwards from the known answer so hard problems stay in the training
set.

It needs only answers. It is also, in the language of this chapter, an outcome
filter, so it selects for correct answers including the lucky ones and trains the
generator on their faulty reasoning. The filter's precision is exactly $1 - \ell$.

That is not an argument against it — it works, and it made the annotation problem
tractable — but it is the reason a STaR-style loop should be monitored with a
soundness metric rather than an accuracy one. Accuracy will look fine while the
proportion of laundered faulty reasoning in the training set rises.

### 7.5 Cost, honestly

A process label is not simply $k$ times an outcome label. It requires a grader
who can evaluate a derivation rather than compare two numbers, which is a
different and more expensive labour pool, and it produces more disagreement
between annotators — the $8\%$ label noise in {{sec:9-practical-example}} is not
pessimistic. The true multiplier is often well above $k$.

Against that, outcome labels are frequently free: any task with an answer key, a
test suite, or a checkable output supplies them at zero marginal cost. So the
realistic comparison is not $1$ against $k$ but $0$ against something expensive,
which raises the bar considerably and is why {{sec:9-practical-example}}'s
imputation result matters.

## 8. Implementation

Two listings. The first asks what each supervision signal selects for, on a task
where soundness and correctness can be measured separately, and then moves the
lucky-chain rate to find out what the difference between them depends on. The
second prices the two signals at equal annotation budget and prices the free
shortcut.

```python {tier=A name=what-outcome-supervision-selects-for}
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
```

The second listing fixes the number of *labels* rather than the number of
examples, and adds the imputation heuristic.

```python {tier=A name=pricing-the-annotation}
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
```

## 9. Practical Example

The task is a chain of six additions. A step is either correct or wrong; a wrong
step adds the wrong number. The answer is right when the total is right — which
happens when every step is right, and also when two wrong steps cancel. Both
reward models see a *noisy* reading of each step and of the final answer; neither
is handed the answer key.

```
                                    quantity     value
------------------------------------------------------
      chains with every step correct (sound)     26.2%
        chains with the correct final answer     34.9%
        correct answer from an UNSOUND chain      8.7%
           ... as a share of correct answers     25.0%
```

That last number is $\ell$: a quarter of the correct answers came from faulty
reasoning, and outcome supervision labels every one of them positive.

```
                         outcome RM               process RM
 best of n      answer        sound      answer        sound
------------------------------------------------------------
         1       35.3%        27.1%       35.3%        27.1%
         4       55.2%        43.0%       64.7%        51.8%
         8       63.2%        49.4%       71.8%        57.4%
        32       71.8%        56.9%       78.1%        62.9%
```

At best-of-32 the process model leads on both criteria, at six times the
annotation cost per example.

On {{cite:uesato2022process}}'s own metric — reasoning error among selections whose
answer is right — this listing gets $20.7\%$ for the outcome model against
$19.5\%$ for the process model, where they report $14.0\%$ falling to $3.4\%$.
**The direction reproduces and the magnitude does not**, and the reason is
measurable in the same run: the process model ranks a correct step above an
incorrect one $83.3\%$ of the time, not perfectly. A chain's score is its weakest
step ({{eq:process-aggregation}}), so six imperfect readings compound, and a
process model at this quality cannot cleanly separate sound chains from lucky
ones. PRM800K is $800{,}000$ human labels; this is a ridge fit on noisy synthetic
ones. The magnitude of the process-supervision advantage is a function of how good
your process model is, which is the first thing to measure before budgeting for
one.

The fourth table is the result that determines how the rest of the chapter is
framed. The reward models are unchanged; only the test distribution moves, so that
wrong steps stray further and stop cancelling:

```
             lucky-answer               outcome RM               process RM
    spread           rate      answer        sound      answer        sound
---------------------------------------------------------------------------
         1           8.6%       68.1%        53.2%       76.0%        62.4%
         2           4.5%       74.3%        66.9%       74.4%        67.1%
         4           2.0%       76.0%        72.2%       73.9%        71.8%
```

The process model's lead on answers goes $+7.9$, $+0.1$, $-2.1$ points as the
lucky rate falls from $8.6\%$ to $2.0\%$. It *reverses*.

So the advantage of process supervision is not a property of process supervision.
**It is proportional to $\ell$.** When a wrong derivation often reaches a right
answer, the outcome label is a badly corrupted proxy for soundness and the process
label is worth paying for. When it rarely does, the outcome label is very nearly a
soundness label already, and paying six times per example buys a slightly worse
model — worse because the process model's own noise now dominates and the outcome
signal has stopped being misleading.

The second listing fixes the *label* budget. One process-labelled chain costs six
labels; one outcome-labelled chain costs one.

```
                     chains bought            answer accuracy
    labels     outcome     process      outcome       process
-------------------------------------------------------------
       900         900         150        55.4%        61.6%
      3600        3600         600        64.1%        69.0%
      7200        7200        1200        67.1%        71.7%
     14400       14400        2400        67.2%        74.8%
     28800       28800        4800        67.7%        74.8%
```

I expected a crossover and there is none: the process signal leads at every budget
swept, while buying six times fewer examples throughout.

The reason is in the last two rows. From $7{,}200$ to $28{,}800$ labels — a
fourfold increase — the outcome model improves by $0.6$ points and the process
model by $3.1$. **The outcome model has stopped learning.** It is not a noisier
version of the process signal; it is a biased one ({{eq:outcome-signal-bias}}), and
more data estimates the same wrong target more precisely. The plateau is where the
model has fully absorbed a signal that was partly wrong.

On soundness the gap is wider — $+7.3$ points at the largest budget — and shows no
sign of closing.

Then the shortcut, which is the practically useful result:

```
                                   answer accuracy                 soundness
    labels      outcome      imputed          real      imputed         real
----------------------------------------------------------------------------
       900        55.4%        67.8%         61.6%        53.9%        49.0%
      7200        67.1%        71.8%         71.7%        57.3%        58.0%
     28800        67.7%        74.3%         74.8%        58.4%        60.5%
```

Imputing step labels from outcomes — mark every step of an answer-correct chain
correct — costs one label per chain instead of six. At $900$ labels it beats
*both* alternatives, because it converts the same outcome budget into six times as
many training rows. At $28{,}800$ it lands within $0.5$ points of human step
annotation on answers, for a sixth of the price, and pays $2.1$ points on
soundness.

And then the case that shows the heuristic's own bias:

```
                signal     answer   soundness
---------------------------------------------
               outcome      92.5%       88.7%
        process (real)      92.9%       89.8%
     process (imputed)      86.2%       83.2%
```

At `spread=4`, where lucky chains are rare, outcome and real process supervision
converge as {{eq:supervision-crossover}} predicts — but the imputed model *drops
below both*, to $86.2\%$. The reason is the second term of
{{eq:imputation-error}}. Imputation marks every step of a correct chain correct,
which is now accurate; and every step of an *incorrect* chain incorrect, which
never was. A failed chain usually has one bad step and five good ones, all marked
wrong, and when luck is rare there are more failed chains to mislabel.

Which sharpens the advice rather than reversing it. Imputation is a good deal
where lucky chains are common — precisely where the outcome signal it replaces is
at its worst — and a bad deal where they are rare, because there the outcome signal
was already nearly right and the imputation adds a bias of its own.

## 10. Production Considerations

Measure $\ell$ before you budget. Take a few hundred of your own solutions, grade
them both ways, and compute the share of correct answers that came from faulty
reasoning. That single number is the expected value of process supervision, and
every other consideration — chain length, answer-space size, model quality —
matters only through it.

Plot accuracy against annotation budget on a log scale for the signal you already
have. A plateau means you are bias-limited and more outcome labels are wasted
money; a continuing climb means you are variance-limited and they are the cheapest
improvement available.

Try imputation before you buy human step labels. It uses only the answer key,
{{sec:9-practical-example}} puts it within half a point of real step annotation on
answers at high budget and *ahead* of everything at low budget, and it tells you
whether the residue is worth purchasing.

Aggregate step scores with a minimum or a product, never a mean. A mean lets good
steps outvote a fatal one, which discards the property you paid for
({{eq:process-aggregation}}).

Track soundness as a separate metric in production, not only accuracy. Every
plateau and every regression in {{sec:9-practical-example}} is larger and earlier
in the soundness column than in the answer column, so a soundness metric is an
early-warning signal even when correctness is all you ship.

Be much more careful when the reward model trains the generator than when it
selects. A biased selector picks bad chains that already exist; a biased training
signal manufactures them ({{sec:7-internal-mechanics}}).

## 11. Common Mistakes

**Treating "process supervision is better" as budget-independent.** It is better
per *example*, and the question is always whether it is better per *label*
({{eq:label-budget}}). Both papers this chapter builds on are careful about which
they measured; most citations of them are not.

**Collecting more outcome labels to fix a plateau.** The plateau is bias. More
data will not move it, and {{sec:9-practical-example}} shows a fourfold increase
buying $0.6$ points.

**Reading a correct answer as a correct derivation.** That inference is wrong
$\ell$ of the time, and $\ell$ was $25\%$ here.

**Using outcome supervision on multiple-choice tasks and expecting soundness.**
{{eq:lucky-rate-model}} says $\ell$ is enormous when the answer space is small.
A four-way choice makes lucky agreement the norm.

**Averaging step scores.** See {{eq:process-aggregation}}. It is the most common
implementation error in a process reward model, and it silently converts it into
something closer to an outcome model.

**Assuming imputation is strictly better than outcome labels.**
{{eq:imputation-error}} has two terms, and the second one makes imputation *worse*
than plain outcome supervision when $\ell$ is small — measured at $-6.3$ points in
{{sec:9-practical-example}}.

## 12. Failure Modes

*Silent laundering in a self-training loop.* A {{cite:zelikman2022star}}-style
filter keeps solutions with correct answers, so the fraction of faulty reasoning
in the fine-tuning set is $\ell$ and it compounds across rounds. Accuracy looks
fine throughout; the derivations degrade.

*Reward hacking against a biased selector.* When the reward model trains the
generator, {{eq:outcome-signal-bias}}'s indifference between sound and lucky chains
becomes something to optimise, and the generator will find lucky-chain strategies
no evaluation set contains.

*Annotation drift on step labels.* "Is this step correct" is under-specified for
steps that are locally valid but follow an earlier error. Different annotators
resolve it differently, and the resulting reward model changes character without
anyone changing the code.

*Process models that are worse than their per-step accuracy suggests.* Aggregating
by minimum over $k$ steps compounds the model's worst case;
{{sec:9-practical-example}}'s $83.3\%$ per-step ranking accuracy produces a much
weaker chain-level signal.

*Imputation in the wrong regime.* Deploying the free heuristic on a task with small
$\ell$ makes things worse than doing nothing, and it will be reported as "the PRM
didn't help".

## 13. Alternatives

**A real verifier.** If the task has an executable check — tests, a type checker, a
proof assistant, a solver — you do not need a learned reward model at all, and $q$
goes to $1$ in {{ch:rsn-test-time-compute}}'s terms. This dominates everything in
this chapter where it is available, and it is {{ch:rsn-tool-assisted}}'s subject.

**Outcome-only, honestly.** If $\ell$ is small and your product consumes only the
answer, outcome supervision is cheaper and at equal budget can be better
({{sec:9-practical-example}}, `spread=4`). Do not buy soundness you do not need.

**Imputed step labels.** {{cite:zelikman2022star}}'s labelling rule, priced above.
The right first move whenever $\ell$ is large.

**Model-generated step labels.** Have a strong model grade steps instead of a
human. Cheaper, and it reintroduces {{ch:rsn-self-consistency}}'s correlation
problem if the grader shares a lineage with the generator. Measure the covariance
before trusting it.

**Monte-Carlo step values.** Estimate a step's quality by how often completions
from it reach a correct answer. This needs only outcomes and produces step-level
signal, at the cost of many rollouts per step. It is the main practical route to
process supervision without annotation, and it inherits $\ell$ in its own way,
since a step that often leads to *lucky* answers scores well.

## 14. Evaluation

Report soundness and accuracy as two numbers. Everything interesting in this
chapter is invisible in accuracy alone, and the gap between them is $\ell$
restated.

Measure the reward model's per-step ranking accuracy separately from the chain
score it produces. {{sec:9-practical-example}} shows a large distance between
those two, and only the first one tells you whether your reward model is good.

Sweep the annotation budget and look at the *shape*. A plateau is a bias
diagnosis, and it is available from data you already have.

Evaluate under a shifted error distribution if you can construct one. Two reward
models that agree in distribution can reverse under shift
({{sec:9-practical-example}}'s fourth table), and which one you want depends on how
stable your generator's failure modes are.

And for any self-training loop, report the soundness of the *training set* it is
accumulating, not just the accuracy of the model it produces.

## 15. Advanced Concepts

**Process supervision without annotation.** Monte-Carlo estimates of step value —
roll out completions from each prefix, score by outcome — turn outcome labels into
step-level signal automatically. This is how most large-scale process reward models
are now built, and its bias is inherited from $\ell$ rather than removed: a step
that leads to lucky answers is scored well. {{maturity:MATURE}} as a technique,
under-analysed as a statistical object.

**Verifiable rewards.** {{cite:deepseek2025r1}}'s training signal is a checker
rather than a model, which sets $\ell = 0$ by construction for the checked
property. The scope question is what fraction of tasks admit one, and the
frontier is extending it beyond code and mathematics.

**The generator/verifier gap.** Everything in {{part:16}} assumes verification is
easier than generation. Where that holds, the whole test-time-compute programme
works; where it does not — open-ended writing, judgement calls, most of what
production systems do — none of it does. Quantifying that gap per task, rather
than assuming it, is the {{maturity:RESEARCH FRONTIER}} question underneath this
part.

**Step granularity as a hyperparameter.** What counts as "a step" changes both $k$
and the annotation cost, and therefore both sides of
{{eq:supervision-crossover}}. Coarser steps are cheaper to label and give a weaker
signal. Nobody tunes this and it is directly tunable.

## 16. Connection to Previous Chapters

{{ch:rsn-test-time-compute}} established that a verifier converts coverage into
accuracy and that its value grows with the sampling budget. This chapter is how
that verifier is built, and it adds the constraint that the *signal* used to build
it can be biased in a way no amount of data fixes.

{{ch:rsn-self-consistency}} showed a critic's error structure mattering more than
its accuracy. The same lesson appears here as bias versus variance: a reward model's
overall accuracy does not tell you what it selects for, and the outcome-supervised
model in {{sec:9-practical-example}} is a good model of the wrong thing.

{{ch:rsn-cot}}'s compounding result is why $\ell > 0$ at all — multiple independent
opportunities to err are also multiple opportunities to cancel — and its
faithfulness result is the same phenomenon at the level of an explanation rather
than a derivation.

Ahead: {{ch:rsn-tool-assisted}} is the case where the verifier is executable and
$\ell$ is zero by construction, which is why it is the chapter where things work.
{{ch:rsn-benchmarks}} returns to measurement with $\ell$ in hand, because a
benchmark that scores answers is subject to exactly this bias.

## 17. Exercises

1. Vary `K` in the first listing from 3 to 12 and plot the measured lucky-chain
   rate against it. Compare the shape with {{eq:lucky-rate-model}}.

2. Shrink the answer space by taking the total modulo 4, simulating a
   multiple-choice task. Predict what happens to $\ell$ and to the two reward
   models before running it.

3. Replace {{eq:process-aggregation}}'s minimum with a mean and re-run. Explain the
   result in one sentence.

4. In the second listing, find the label budget at which imputation stops beating
   real step labels on answers, and explain which term of
   {{eq:imputation-error}} is responsible.

5. Implement the Monte-Carlo step-value estimator described in
   {{sec:15-advanced-concepts}} — score a step by the outcome rate of completions
   from it — and compare its cost and quality with both signals here.

6. Build a version where the reward model *trains* the generator rather than
   selecting from it, and measure how the lucky-chain rate moves over rounds.

## 18. Interview Questions

1. Your model gets 80% on a maths benchmark. What does that tell you about the
   correctness of its derivations?

2. You double your outcome-label budget and accuracy does not move. What do you
   conclude, and what do you do next?

3. When is outcome supervision *better* than process supervision, not just
   cheaper?

4. Why is averaging step scores the wrong way to score a chain?

5. You cannot afford step-level annotation. What would you do instead, and in
   which regime would your substitute make things worse?

6. A colleague proposes a self-training loop that fine-tunes on all solutions that
   reached the correct answer. What would you monitor?

## 19. Research Questions

1. Can $\ell$ be estimated without step-level grading — from the geometry of the
   answer distribution, or from agreement between independently sampled
   derivations?

2. Monte-Carlo step values inherit $\ell$ through their outcome signal. Is there an
   annotation-free step signal that does not?

3. What is the right definition of "correct step" for a step that is locally valid
   but follows an earlier error, and does the choice change downstream selection
   quality measurably?

4. Step granularity affects both sides of {{eq:supervision-crossover}}. Is there an
   optimal granularity, and is it predictable from the task?

5. For which task families does verification remain easier than generation as model
   capability rises? If the gap closes, the whole verification-centred approach of
   {{part:16}} closes with it.

## 20. Chapter Summary

Two supervision signals differ by a factor of $k$ in cost, and the choice between
them turns on one number: $\ell$, the share of correct answers that came from
faulty reasoning ({{eq:lucky-chain}}). It was $25\%$ in
{{sec:9-practical-example}}, it grows with chain length, and it is enormous when
the answer space is small.

Outcome supervision assigns the same expected score to a sound chain and a lucky
one, by construction ({{eq:outcome-signal-bias}}). That makes it a *biased* signal
rather than a noisy one, and the consequence is measurable: a fourfold increase in
outcome labels moved accuracy $0.6$ points while the same increase in process
labels moved it $3.1$. The outcome model plateaus because it has learned
everything its signal contains.

At equal *label* budget — not equal examples — process supervision led at every
budget swept, despite buying six times fewer examples. But move the lucky-chain
rate down from $8.6\%$ to $2.0\%$ and the advantage reverses to $-2.1$ points.
**The value of process supervision is proportional to $\ell$**, which reconciles
{{cite:uesato2022process}}'s finding on short GSM8K derivations with
{{cite:lightman2023verify}}'s on long MATH ones: those are the two ends of the
same table.

The free shortcut — imputing step labels from outcomes, the rule inside
{{cite:zelikman2022star}}'s loop — came within $0.5$ points of human step
annotation on answers at high budget and beat everything at low budget. It also
*lost* $6.3$ points to plain outcome supervision when lucky chains were rare,
because its second error term ({{eq:imputation-error}}) marks the five good steps
of a failed chain as wrong. It helps where the outcome signal is worst and hurts
where the outcome signal is fine.

One caveat carried throughout: this listing reproduced the *direction* of
{{cite:uesato2022process}}'s reasoning-error result and not its magnitude
($20.7\%$ against $19.5\%$, where they report $14.0\%$ against $3.4\%$), because
its process model ranks steps correctly only $83.3\%$ of the time. How much
process supervision buys you is a function of how good your process model is.

So: measure $\ell$, plot accuracy against budget and look for the plateau, try
imputation before buying annotation, aggregate with a minimum, and track soundness
separately from accuracy.

## 21. Further Reading

{{cite:uesato2022process}} and {{cite:lightman2023verify}} should be read together
and in that order. The first is the careful comparison and contains the finding
that outcome supervision is competitive on final answers at lower cost; the second
is the large-scale result in the opposite direction. This chapter's claim is that
they are the same finding at two values of $\ell$.

{{cite:zelikman2022star}} is short, and worth reading for the rationalisation
trick as much as for the loop: generating a rationale *backwards* from the known
answer is how hard problems stay in the training set, and it is also where the
laundering risk is highest.

{{cite:cobbe2021gsm8k}} is the origin of verifier-plus-sampling and the baseline
both later papers measure against.

{{cite:deepseek2025r1}} for what happens when the reward is a checker rather than
a model, which is the escape from everything in this chapter and the bridge to
{{ch:rsn-tool-assisted}}.
