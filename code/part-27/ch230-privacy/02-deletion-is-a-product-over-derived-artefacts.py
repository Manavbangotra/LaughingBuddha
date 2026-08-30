# -*- coding: utf-8 -*-
# Extracted from: Chapter 230 — Privacy, Data Governance, and Copyright
# Source: src/.../ch230-privacy.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Deleting a record means deleting it everywhere it went, and one destination cannot.

A deletion request is a conjunction: the record is gone only if it is gone from the source
store, the index, the embeddings, the caches, the logs, the analytics warehouse, the backups,
the partner exports -- and the model weights
(eq:deletion-is-a-product-over-derived-artefacts).

Every destination except the last has a delete operation. The weights do not, and the only
mechanism that removes a training example from them is retraining, which is priced per training
run rather than per request.

The second half is copyright, which is often filed next to privacy and is a different question
with the same measurement. Whether a model reproduces protected material is the memorisation
rate ch:sec-data-leakage already computed
(eq:copyright-exposure-is-the-memorisation-rate).
"""
import math

# (destination, deletable?, cost per request, latency days, share of copies)
DESTINATIONS = [
    ("the source record store",   True,  0.02,  0.01, 1.00),
    ("the search index",          True,  0.04,  0.04, 1.00),
    ("embedding vectors",         True,  0.06,  0.08, 1.00),
    ("the summary cache",         True,  0.03,  0.30, 0.62),
    ("the semantic answer cache", True,  0.05,  0.30, 0.41),
    ("conversation histories",    True,  0.11,  1.20, 0.74),
    ("application logs",          True,  0.09,  2.00, 0.88),
    ("the analytics warehouse",   True,  0.22,  7.00, 0.55),
    ("nightly backups",           True,  0.34, 35.00, 1.00),
    ("a partner export",          False, 0.00,  0.00, 0.19),
    ("the model weights",         False, 0.00,  0.00, 1.00),
]

print("Where one record goes, and whether it can be removed.")
print()
print(f"{'destination':>28}{'deletable':>12}{'cost/request':>15}"
      f"{'latency (days)':>17}{'share holding it':>19}")
print("-" * 91)
per_request = 0.0
for name, dele, cost, lat, share in DESTINATIONS:
    per_request += cost * share
    print(f"{name:>28}{('yes' if dele else 'no'):>12}{cost:>15.2f}"
          f"{lat:>17.2f}{share:>19.0%}")
print("-" * 91)
print(f"{'TOTAL PER REQUEST':>28}{'':>12}{per_request:>15.2f}")

undeletable = [n for n, d, c, l, s in DESTINATIONS if not d]
print()
print(f"deletable destinations: {len(DESTINATIONS) - len(undeletable)}")
print(f"undeletable: {', '.join(undeletable)}")
print(f"longest latency: {max(l for n, d, c, l, s in DESTINATIONS):.0f} days")

print()
print()
print("Completeness as a product, since the record is gone only if it is gone")
print("everywhere.")
print()
print(f"{'after deleting from':>28}{'per-destination success':>26}"
      f"{'cumulative completeness':>26}")
print("-" * 80)
SUCCESS = {
    "the source record store": 0.999, "the search index": 0.995,
    "embedding vectors": 0.990, "the summary cache": 0.940,
    "the semantic answer cache": 0.910, "conversation histories": 0.880,
    "application logs": 0.820, "the analytics warehouse": 0.760,
    "nightly backups": 0.700, "a partner export": 0.000,
    "the model weights": 0.000,
}
cum = 1.0
cum_deletable = None
for name, dele, cost, lat, share in DESTINATIONS:
    s = SUCCESS[name] if dele else 1.0 - share
    cum *= s
    if name == "nightly backups":
        cum_deletable = cum
    print(f"{name:>28}{s:>26.3f}{cum:>26.4f}")

print()
print(f"across the nine deletable destinations: {cum_deletable:.4f}")
print(f"including the partner export:           {cum_deletable * 0.81:.4f}")
print(f"including the model weights:            {cum:.4f}")

print()
print()
print("What removing a record from the weights would cost.")
print()
TRAIN_COST = 1_400_000.0
REQUESTS_PER_YEAR = 62_000
print(f"{'approach':>34}{'cost per request':>19}{'latency':>16}"
      f"{'feasible?':>12}")
print("-" * 81)
APPROACHES = [
    ("retrain from scratch per request", TRAIN_COST, "6 weeks", "no"),
    ("batch retrain quarterly",  TRAIN_COST * 4 / REQUESTS_PER_YEAR,
     "up to 90 days", "maybe"),
    ("batch retrain annually",   TRAIN_COST / REQUESTS_PER_YEAR,
     "up to 365 days", "yes"),
    ("machine unlearning",       TRAIN_COST * 0.04 / 60, "hours", "research"),
    ("never train on it",        0.0, "n/a", "yes"),
]
for name, cost, lat, feas in APPROACHES:
    print(f"{name:>34}{cost:>19,.2f}{lat:>16}{feas:>12}")

print()
print(f"an annual batch retrain costs {TRAIN_COST / REQUESTS_PER_YEAR:,.2f} per")
print("request and takes up to a year; per-request retraining is not a policy")

print()
print()
print("So the design decision is upstream, and it has three options.")
print()
OPTIONS = [
    ("train on everything, delete elsewhere", 1.00, cum, "the weights retain it"),
    ("train only on consented data",          0.71, 0.9994, "smaller corpus"),
    ("retrieval only, never fine-tune",       0.83, 0.9994, "context, not weights"),
    ("train with DP at epsilon 3",            0.91, 0.9994, "a formal bound instead"),
]
print(f"{'option':>40}{'capability kept':>18}{'deletion completeness':>24}"
      f"{'what it trades':>24}")
print("-" * 106)
for name, cap, comp2, trade in OPTIONS:
    print(f"{name:>40}{cap:>18.0%}{comp2:>24.4f}{trade:>24}")

print()
print("The first row is what most systems do and the last column is why the")
print("other three exist.")

print()
print()
print("Now copyright, which is a different question with the same measurement.")
print()
LICENCES = [
    ("public domain",          0.11, "none",            "none"),
    ("permissive open licence", 0.19, "attribution",    "low"),
    ("copyleft",               0.06, "share-alike",     "medium"),
    ("all rights reserved, crawled", 0.48, "unresolved", "high"),
    ("licensed for training",  0.09, "contractual",     "none"),
    ("provenance unknown",     0.07, "unresolved",      "unknown"),
]
print(f"{'licence class':>32}{'share of corpus':>18}{'obligation':>16}"
      f"{'exposure':>12}")
print("-" * 78)
unresolved = 0.0
for name, share, obl, exp in LICENCES:
    if obl == "unresolved":
        unresolved += share
    print(f"{name:>32}{share:>18.0%}{obl:>16}{exp:>12}")

print()
print(f"{unresolved:.0%} of the corpus has an unresolved obligation")

print()
print()
print("And the technical question, which is the memorisation rate.")
print()
print(f"{'content type':>30}{'occurrences':>13}{'distinctiveness':>18}"
      f"{'P(verbatim reproduction)':>27}")
print("-" * 88)
REPRO = [
    ("a common phrase",          410_000, 0.04),
    ("a book paragraph",              14, 0.71),
    ("a song lyric",                 890, 0.66),
    ("a code snippet, unique",          1, 0.94),
    ("an image caption",               3, 0.58),
]
for name, occ, dist in REPRO:
    base = 1.0 - math.exp(-0.42 * occ ** 0.55)
    p = min(0.995, base * (0.25 + 0.75 * dist) * 0.61)
    print(f"{name:>30}{occ:>13,}{dist:>18.2f}{p:>27.3f}")

print()
print("These are the same numbers ch:sec-data-leakage computed for secrets,")
print("because it is the same mechanism.")

print()
print()
print("What each control does on each axis.")
print()
CONTROLS = [
    ("deletion pipeline",       "privacy", 0.71, 0.00, "removes copies"),
    ("differential privacy",    "privacy", 0.93, 0.44, "bounds influence"),
    ("licence filtering at ingest", "copyright", 0.00, 0.62, "excludes classes"),
    ("output verbatim filter",  "both",    0.31, 0.58, "blocks reproduction"),
    ("provenance manifest",     "both",    0.11, 0.34, "records what went in"),
    ("never train on it",       "both",    1.00, 1.00, "removes the question"),
]
print(f"{'control':>32}{'axis':>12}{'privacy':>10}{'copyright':>12}"
      f"{'what it does':>24}")
print("-" * 90)
for name, axis, pr, cp, what in CONTROLS:
    print(f"{name:>32}{axis:>12}{pr:>10.2f}{cp:>12.2f}{what:>24}")

print(f"""
The destination table is the shape of a deletion request. One record reaches
{len(DESTINATIONS)} places, {len(DESTINATIONS) - len(undeletable)} of which support deletion, at
a total marginal cost of {per_request:.2f} and a longest latency of
{max(l for n, d, c, l, s in DESTINATIONS):.0f} days -- the backup rotation.

The two that do not are `{undeletable[0]}` and `{undeletable[1]}`.

The completeness table is why "we deleted it" is a claim requiring evidence. Deletion succeeds
per destination at between {min(SUCCESS[n] for n, d, c, l, s in DESTINATIONS if d):.2f} and
{max(SUCCESS.values()):.3f}, and the record is gone only if every one succeeded -- a product
(eq:deletion-is-a-product-over-derived-artefacts).

Across the nine deletable destinations that gives **{cum_deletable:.4f}**. Including the
partner export, {cum_deletable * 0.81:.4f}. Including the weights, **{cum:.4f}** -- exactly
zero, because a record used in training is not removed by any operation in the table.

That last number is the honest one and it is the one nobody reports. The number teams do report
is the first: nine pipelines, each with a delete, each mostly working. **The compliance claim is
made over the destinations that have a delete operation**, and the destination that does not is
excluded from the accounting rather than from the system.

It is also ch:ops-versioning's conjunction result and ch:sd-storage's derived-copy result
arriving together. Nothing is broken in any individual pipeline; the composite is what it is
because conjunctions punish.

The retraining table prices the destination that has no delete. Removing a record from the
weights by retraining from scratch costs {TRAIN_COST:,.0f} per request, which is not a policy.
Batched annually it is {TRAIN_COST / REQUESTS_PER_YEAR:,.2f} per request -- affordable -- and
the latency is **up to a year**, which is not a deletion guarantee any regulation would accept.

Machine unlearning is the research direction and it is labelled research here because the
guarantees available are weaker than "the record is gone", which is what the request asked for.

So the design decision is upstream, and the options table has three of them.
`train only on consented data` keeps {0.71:.0%} of capability. `retrieval only, never
fine-tune` keeps {0.83:.0%} and moves the exposure to ch:sec-data-leakage's inference-time
categories. `train with DP at epsilon 3` keeps {0.91:.0%} and substitutes a formal bound for a
deletion promise -- which is a different assurance and, for membership questions, a stronger
one.

**The first row is what most systems do**, and its last column is why the other three exist.

The licence table is the copyright axis and it is not a technical question. Forty-eight percent
of this corpus is all-rights-reserved material obtained by crawling and seven percent has
unknown provenance -- {unresolved:.0%} with an unresolved obligation. No engineering control
resolves that; a licensing decision does.

What *is* technical is the reproduction question, and the last table is the point of this
chapter's second half. The probability that a model reproduces a specific piece of protected
content verbatim is computed exactly the way ch:sec-data-leakage computed extraction of a
secret, because **it is the same mechanism**
(eq:copyright-exposure-is-the-memorisation-rate). A unique code snippet appearing once has a
{min(0.995, (1.0 - math.exp(-0.42 * 1 ** 0.55)) * (0.25 + 0.75 * 0.94) * 0.61):.3f}
reproduction probability; a book paragraph appearing fourteen times,
{min(0.995, (1.0 - math.exp(-0.42 * 14 ** 0.55)) * (0.25 + 0.75 * 0.71) * 0.61):.3f}.

That is a useful unification because it means one measurement serves both programmes. A canary
methodology built for memorisation measures copyright exposure; a deduplication programme
reduces both; an output verbatim filter blocks both.

The control table makes the overlap explicit and marks the one row that is not a compromise.
`never train on it` scores {1.00:.2f} on both axes, and every other row is partial on at least
one. `differential privacy` is {0.93:.2f} on privacy and {0.44:.2f} on copyright, because a
bound on individual influence is not a bound on reproducing a passage that appears in many
documents.

**Privacy and copyright share a mechanism and not a remedy**, which is why filing them together
produces programmes that half-address both.""")
