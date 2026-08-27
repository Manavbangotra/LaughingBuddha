# -*- coding: utf-8 -*-
# Extracted from: Chapter 106 — Why RAG Exists
# Source: src/.../ch106-why-rag.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""How fast does a frozen model go stale, and when does retrieval overtake it?

A MODEL, clearly labelled as one: a corpus of facts, each with a probability per
month of being superseded. A parametric model answers from its training snapshot;
a RAG system answers from the current corpus but only when retrieval finds the
fact.

The question is not whether RAG wins -- it obviously does eventually -- but HOW
SOON, because that is what decides whether staleness is an architectural problem
or a scheduling one.
"""
import numpy as np

rng = np.random.default_rng(19)

N_FACTS, N_TRIALS = 5000, 40
RECALL_AT_K = 0.85           # the retriever's ceiling, eq:rag-ceiling
PARAM_ACCURACY = 0.92        # accuracy on facts the model DID learn correctly
MONTHS = [0, 3, 6, 12, 18, 24, 36]

# Monthly probability that a given fact is superseded, by domain.
CHURN = {"stable (regulations)": 0.005,
         "moderate (product docs)": 0.03,
         "fast (pricing, staffing)": 0.10}

print(f"{'domain / architecture':<30}{'churn':>8}"
      + "".join(f"{'m' + str(m):>8}" for m in MONTHS))
print("-" * (38 + 8 * len(MONTHS)))

for label, churn in CHURN.items():
    param_row, rag_row = [], []
    for months in MONTHS:
        p_current = (1 - churn) ** months        # fact unchanged since cutoff
        acc_param = np.mean([
            np.mean(rng.random(N_FACTS) < p_current * PARAM_ACCURACY)
            for _ in range(N_TRIALS)])
        acc_rag = np.mean([
            np.mean(rng.random(N_FACTS) < RECALL_AT_K * PARAM_ACCURACY)
            for _ in range(N_TRIALS)])
        param_row.append(acc_param)
        rag_row.append(acc_rag)
    print(f"{label:<30}{churn:>8.3f}"
          + "".join(f"{v:>8.3f}" for v in param_row) + "   parametric")
    print(f"{'':<30}{'':>8}"
          + "".join(f"{v:>8.3f}" for v in rag_row) + "   RAG")
    # First month at which RAG overtakes the frozen model.
    cross = next((m for m, p, r in zip(MONTHS, param_row, rag_row) if r > p), None)
    verdict = f"month {cross}" if cross is not None else "not within 36 months"
    print(f"{'  -> RAG overtakes at ' + verdict:<30}\n")

print(f"""
Read the crossover line, which is the only number in the table that changes a
decision.

At month zero the parametric model WINS in every domain: it answers from a
perfect memory of its training snapshot, while RAG pays the eq:rag-ceiling tax of
{RECALL_AT_K:.0%} recall on every query. Retrieval is not free accuracy -- it
trades a retrieval failure mode for a staleness failure mode.

How long that trade takes to pay off is entirely a property of the DOMAIN, not of
the technology. In the fast-churn row the frozen model is overtaken within a
couple of quarters; in the stable row it stays ahead for years, and a RAG system
there is buying attribution and access control rather than accuracy.

That is the useful decomposition. Freshness is one of the four properties in
section 5.3 and it is the only one with a clock on it -- so measure your corpus's
churn rate before assuming staleness is your problem. Teams frequently build RAG
for freshness when what they actually needed was citations.""")
