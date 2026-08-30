# -*- coding: utf-8 -*-
# Extracted from: Chapter 130 — Supervised Fine-Tuning
# Source: src/.../ch130-sft.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Truncation is not a small loss of data. It is a training signal.

Every fine-tuning run sets a maximum sequence length, and every dataset has
examples longer than it. The usual mental model is that a few per cent of
examples get shortened and the effect is proportional and small.

It is neither, for two reasons this listing measures.

First, truncation is not uniform over the example. The prompt comes first, so the
cut always lands on the COMPLETION -- the part being trained on -- and long
examples are exactly the ones with long completions
(eq:truncation-hits-completions).

Second, and worse: a truncated example is not dropped. It is trained on, with an
end-of-sequence position that is not where the answer ended. The model is being
shown, repeatedly, that a good answer stops mid-sentence at exactly the token
budget (eq:truncation-teaches-stopping).
"""
import numpy as np

rng = np.random.default_rng(139)

N = 40000
PROMPT_FRAC = 0.35            # of an example's tokens, roughly, are the prompt


def dataset():
    """Heavy-tailed total lengths; completion length correlates with total, as it
    does in practice -- long questions get long answers."""
    total = np.clip(rng.lognormal(mean=6.0, sigma=0.85, size=N), 48, 32000)
    prompt = total * (PROMPT_FRAC + 0.10 * rng.normal(size=N)).clip(0.15, 0.6)
    completion = total - prompt
    return total.astype(int), prompt.astype(int), completion.astype(int)


TOTAL, PROMPT, COMP = dataset()

print(f"{N:,} examples. median length {int(np.median(TOTAL))}, "
      f"p95 {int(np.percentile(TOTAL, 95))}, p99 {int(np.percentile(TOTAL, 99))}\n")
print(f"{'max_len':>9}{'% examples':>14}{'% completion':>16}"
      f"{'trained tokens from':>22}")
print(f"{'':>9}{'truncated':>14}{'tokens lost':>16}"
      f"{'a truncated example':>22}")
print("-" * 62)

rows = {}
for L in (512, 1024, 2048, 4096, 8192):
    trunc = TOTAL > L
    # Tokens of completion that survive: whatever is left after the prompt.
    kept_comp = np.clip(L - PROMPT, 0, COMP)
    lost_comp = COMP - kept_comp
    ex_frac = float(trunc.mean())
    tok_frac = float(lost_comp.sum() / COMP.sum())
    # Of the completion tokens the model DOES train on, what share come from a
    # truncated example -- i.e. end at a fake stopping point?
    fake_stop = float(kept_comp[trunc].sum() / kept_comp.sum())
    rows[L] = (ex_frac, tok_frac, tok_frac, fake_stop)
    print(f"{L:>9}{ex_frac:>14.1%}{tok_frac:>16.1%}{fake_stop:>22.1%}")

print(f"\n{'max_len':>9}{'amplification: lost tokens / truncated examples':>50}")
print("-" * 60)
for L in (512, 1024, 2048, 4096, 8192):
    e, t, _, _ = rows[L]
    print(f"{L:>9}{(t / e if e else 0):>50.2f}x")

a2, a8 = rows[2048], rows[8192]
print(f"""
Read the first two columns together at max_len 2048: {a2[0]:.1%} of examples are
truncated, and {a2[1]:.1%} of completion tokens are lost. Those are not the same
number, and the ratio in the second table is why -- the examples that get
truncated are the LONG ones, so each truncated example loses far more than an
average example contains (eq:truncation-hits-completions).

The amplification factor makes this concrete. At every budget, the share of
training signal lost is several times the share of examples affected. "We only
truncate a few per cent of examples" is a true statement that describes a much
larger loss than it sounds like, and it is the sentence that usually ends the
conversation.

Note that the amplification RISES with max_len, from 2.39x to 3.72x. That is
counter-intuitive and worth pausing on: raising the limit reduces the total loss
while making each remaining truncation worse, because the examples still being
cut are drawn from ever further out in the tail.

The direction of the loss compounds it. Truncation cuts from the END, and the
prompt is at the start, so the tokens removed are always completion tokens -- the
ones the loss is computed on. A truncated example does not lose 30% of itself
evenly; it keeps its entire prompt and loses the tail of its answer.

Now the last column, which is the part that is not a data-loss problem at all.
Those truncated examples are not discarded. They are trained on, and their final
token -- the one the model learns to follow with an end of sequence -- sits
wherever the budget ran out, mid-sentence. At max_len 2048, {a2[3]:.1%} of the
completion tokens the model trains on come from an example that ends at a fake
stopping point (eq:truncation-teaches-stopping).

The model is not merely missing those answers. It is being taught, on a
meaningful share of its training signal, that a good answer stops abruptly at
around the token budget. The symptom in production is a model that trails off on
long generations, and the diagnosis usually offered is "it needs a longer context
window" -- which is exactly backwards, because the behaviour was installed by
training rather than limited by capacity.

Raising max_len to 8192 improves every column ({a8[0]:.1%} of examples,
{a8[1]:.1%} of tokens, {a8[3]:.1%} fake stops) and costs quadratically in
attention. So the budget is a real constraint and the question is what to do
within it, which has a better answer than picking a number.

DROP the examples you cannot fit, rather than truncating them. Dropping loses
exactly the same completion tokens -- they were never going to be trained on --
and it removes the fake stopping points entirely. The cost is a slightly smaller,
slightly shorter-skewed dataset; the benefit is that nothing in the training data
is a lie about where answers end.

If long examples matter to your task, the answer is not truncation either. Split
them into multiple training examples at natural boundaries, so each one ends where
something actually ended. Both options are cheap. Truncation is the default in
most tooling, and it is the only one of the three that actively teaches something
false.""")
