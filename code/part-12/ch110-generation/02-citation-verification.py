# -*- coding: utf-8 -*-
# Extracted from: Chapter 110 — Prompt Construction, Generation, and Citation
# Source: src/.../ch110-generation.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""An unverified citation is decoration. Verification is what makes it evidence.

Every claim in a generated answer carries a citation. Some claims are genuinely
supported by the cited chunk; some are not -- drawn from parametric memory, from a
different chunk, or fabricated. Nothing in the generation step distinguishes
them, because the model was asked to cite and it emitted a citation.

We simulate answers at a known unsupported rate, run a verifier over each claim,
and measure eq:post-verification-rate: what survives, and at what cost in dropped
good claims. The verifier is a lexical-overlap proxy; a real one is an NLI model
or an LLM judge, both stronger and both far cheaper than the generation.
"""
import numpy as np

rng = np.random.default_rng(17)

N_ANSWER, CLAIMS_PER_ANSWER, VOCAB = 1500, 6, 400
UNSUPPORTED_RATE = 0.22        # fraction of claims not entailed by their citation
CHUNK_LEN, CLAIM_LEN = 60, 12


def make_case():
    """One claim with its cited chunk.

    A SUPPORTED claim is a PARAPHRASE of the chunk, not a copy of it -- so it
    shares most of the chunk's content words and introduces some of its own.
    An UNSUPPORTED claim is about the right topic and says something the chunk
    does not, so it shares fewer.

    The two distributions therefore OVERLAP, which is what makes the threshold
    sweep a real trade-off rather than an exercise. A verifier that separated
    them perfectly would not need a threshold, and would not resemble any
    verifier you can build.
    """
    chunk = rng.choice(VOCAB, CHUNK_LEN, replace=False)
    outside = np.setdiff1d(np.arange(VOCAB), chunk)
    supported = rng.random() > UNSUPPORTED_RATE
    if supported:
        n_shared = int(rng.integers(5, CLAIM_LEN + 1))    # 42%-100% overlap
    else:
        n_shared = int(rng.integers(2, 11))               # 17%-83% overlap
    claim = np.concatenate([
        rng.choice(chunk, n_shared, replace=False),
        rng.choice(outside, CLAIM_LEN - n_shared, replace=False)])
    return set(chunk.tolist()), set(claim.tolist()), supported


def coverage(chunk, claim):
    """Fraction of the claim's content present in the cited span. A crude but
    real verifier: an NLI model or LLM judge does the same job better."""
    return len(claim & chunk) / len(claim)


cases = [make_case() for _ in range(N_ANSWER * CLAIMS_PER_ANSWER)]
scores = np.array([coverage(c, m) for c, m, _ in cases])
truth = np.array([s for _, _, s in cases])

baseline = 1 - truth.mean()
print(f"claims: {len(cases):,}   genuinely unsupported: {baseline:.1%}")
print(f"\n{'threshold':>10}{'claims kept':>13}{'unsupported kept':>18}"
      f"{'good dropped':>14}{'unsupported caught':>20}")
print("-" * 76)

for thr in [0.0, 0.40, 0.50, 0.58, 0.67, 0.75, 0.84, 0.92]:
    keep = scores >= thr
    if keep.sum() == 0:
        continue
    kept_rate = keep.mean()
    unsupported_kept = (~truth[keep]).mean()
    good_dropped = (truth & ~keep).sum() / truth.sum()
    caught = (~truth & ~keep).sum() / (~truth).sum()
    print(f"{thr:>10.2f}{kept_rate:>13.1%}{unsupported_kept:>18.1%}"
          f"{good_dropped:>14.1%}{caught:>20.1%}")

print(f"""
The threshold=0.00 row is the system almost everyone ships: every claim is kept,
every citation is displayed, and {baseline:.0%} of what the user reads is a
confident statement attached to a source that does not support it. Nothing about
the output signals which ones. The citation is not merely uninformative there --
it is actively harmful, because it converts a reader's healthy scepticism into
misplaced confidence.

Now walk down the threshold column, and note that it has two regimes rather than
one.

The first rows are a FREE REGION. Up to the lowest coverage any genuine
paraphrase achieves, the filter catches a third of the fabrications and drops
nothing -- because no supported claim scores that low. Every system should take
this region unconditionally; it is a pure gain and it requires no judgement about
how to value the trade.

Past it the trade turns, and turns UNFAVOURABLY: each further point of
unsupported claims caught costs several points of good claims dropped. That is
worth stating plainly, because the tidy story would be that the curve stays
favourable throughout and it does not. Reaching zero unsupported claims here
means discarding three quarters of the good ones, which is not a system anyone
would ship.

So the recommendation is conditional rather than universal. Take the free region
always. Beyond it, eq:verification-asymmetry is the argument for leaning
aggressive -- a dropped good claim costs COMPLETENESS while a kept bad claim
costs a confident cited falsehood, and those are not comparable harms -- but it
is a genuine trade with a genuine cost, and where you sit on it is a product
decision about how much incompleteness your users will tolerate.

Note also what this measurement does NOT require: no ground truth about whether
the claim is TRUE. It asks only whether the cited text supports it, which is
checkable from material you already have. That is ch:llm-hallucination's point
that groundedness is measurable while truth is not, arriving with an
implementation -- and it is why this is deployable as a runtime filter rather
than only as an offline evaluation.

The verifier here is deliberately crude -- lexical overlap. An NLI model or an LLM
judge does the same job substantially better and still costs a fraction of the
generation being checked. The reason to show the crude version is that even it
separates the two populations well, so the usual objection -- that verification
is too expensive or too unreliable to be worth it -- does not survive contact with
the numbers.""")
