# -*- coding: utf-8 -*-
# Extracted from: Chapter 117 — RAG Failure Modes and How to Debug Them
# Source: src/.../ch117-failure-modes.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Diagnosing ONE failing query: why the symptom does not tell you the stage.

The previous listing decided which stage to work on across a whole evaluation
set. This one is the other job, and the one that actually arrives on a Tuesday:
a user reports a bad answer, and something has to decide which stage lost it.

The obstacle is that a RAG pipeline emits the same few symptoms whatever broke.
A fabricated answer looks identical whether the document was never ingested,
never retrieved, or retrieved and ignored -- because the generator writes fluent
prose in all three cases (eq:symptom-collapse). This listing measures how much a
symptom is worth, then measures four cheap PROBES against it
(eq:diagnostic-probes).
"""
import numpy as np

rng = np.random.default_rng(2024)

N = 60_000
STAGES = ["ingestion", "indexing", "retrieval", "generation"]
SYMPTOMS = ["fabricated", "says-not-found", "contradicts-source", "partial"]

# How often each stage is the true culprit.
PRIOR = np.array([0.22, 0.13, 0.40, 0.25])

# P(symptom | stage). Rows are stages. The first three rows are nearly identical
# in their first two columns, and that is the whole problem: the symptoms a user
# can report do not separate the stages that produce them.
LIKELIHOOD = np.array([
    [0.50, 0.38, 0.02, 0.10],      # ingestion: gone from the corpus entirely
    [0.46, 0.36, 0.04, 0.14],      # indexing: present but unfindable
    [0.48, 0.34, 0.04, 0.14],      # retrieval: findable but not found
    [0.20, 0.06, 0.46, 0.28],      # generation: it had the text and misused it
])

# Probe reliability. Probes read the SYSTEM, not the answer, which is why they
# separate stages that the answer cannot (eq:state-versus-output).
PROBE_ACC = {"in_corpus": 0.97, "in_index": 0.93, "in_topk": 0.98, "used": 0.90}

truth = rng.choice(len(STAGES), size=N, p=PRIOR)
symptom = np.array([rng.choice(len(SYMPTOMS), p=LIKELIHOOD[t]) for t in truth])


def guess_from_symptom():
    """The best possible symptom-only classifier: maximum a posteriori under the
    true generative model (eq:map-triage). No real triage does better."""
    post = PRIOR[:, None] * LIKELIHOOD          # stage x symptom
    best = post.argmax(axis=0)
    return best[symptom]


def guess_from_probes():
    """Four checks, run in pipeline order, each answering one yes/no question
    about the system rather than about the answer:

      in_corpus : does the gold text exist anywhere in the parsed corpus?
      in_index  : is it in a chunk that the index actually holds?
      in_topk   : did that chunk come back for this query?
      used      : given the gold chunk in the prompt, does the model answer?
    """
    out = np.empty(N, dtype=int)
    noisy = {k: rng.random(N) < v for k, v in PROBE_ACC.items()}
    for i in range(N):
        t = truth[i]
        # A probe fails at the broken stage; upstream probes pass. Each probe
        # reports correctly with its own accuracy (eq:probe-attribution).
        if t == 0:
            out[i] = 0 if noisy["in_corpus"][i] else 1
        elif t == 1:
            out[i] = 1 if noisy["in_index"][i] else 2
        elif t == 2:
            out[i] = 2 if noisy["in_topk"][i] else 3
        else:
            out[i] = 3 if noisy["used"][i] else 2
    return out


def confusion(pred, title):
    print(f"\n{title}   accuracy {float((pred == truth).mean()):.3f}")
    print(f"{'true \\ called':<16}" + "".join(f"{s[:9]:>12}" for s in STAGES))
    for i, s in enumerate(STAGES):
        row = [(float(((truth == i) & (pred == j)).sum())
                / max((truth == i).sum(), 1)) for j in range(len(STAGES))]
        print(f"{s:<16}" + "".join(f"{v:>12.2f}" for v in row))


print(f"{N:,} failing queries. True stage prior: "
      + ", ".join(f"{s} {p:.0%}" for s, p in zip(STAGES, PRIOR)))

confusion(guess_from_symptom(), "SYMPTOM ONLY (optimal MAP classifier)")
confusion(guess_from_probes(), "FOUR PROBES, in pipeline order")

print("""
The first matrix is the ceiling on what the reported symptom can tell you, and it
is low. It is not a bad classifier -- it is the OPTIMAL one under the true
generative model, so no amount of prompt engineering on a triage rubric beats it.
Read the first three rows: ingestion, indexing and retrieval failures produce
almost the same symptom distribution, so the classifier gives up and assigns
nearly all of them to whichever is most common. Two entire stages are effectively
invisible -- the ingestion and indexing COLUMNS are empty, which means this
procedure will never once name them, no matter how many tickets it processes.

The one useful signal is in the last row. "The answer contradicts the text that
was retrieved" genuinely does discriminate, because it is the only symptom that
requires the right text to have ARRIVED. That is worth building into a triage
form, and ch:rag-generation's citation verification computes it automatically.
Everything else a user can tell you is close to noise.

The second matrix is what four yes/no probes buy, and the jump is the point of
the chapter: 0.514 to 0.952. Each probe interrogates the SYSTEM rather than the
answer -- is the text in the corpus, is it in the index, did it come back, was it
used. None requires a judgement about quality, all four are cheap, and together
they locate the failure almost exactly.

Note why they work where the symptom does not. The stages are indistinguishable
in their OUTPUT and perfectly distinguishable in their STATE -- a chunk is either
in the index or it is not. So the diagnostic move is always the same: stop
studying the answer and go and look at the artefact the stage was supposed to
produce.

The probes are also cheap in the order given. Run them in pipeline order and stop
at the first failure: most investigations end after one or two, and the last and
most expensive probe -- re-running generation with the gold chunk supplied -- is
needed only for queries that got all the way through.""")
