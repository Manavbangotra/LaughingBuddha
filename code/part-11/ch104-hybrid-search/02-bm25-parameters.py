# -*- coding: utf-8 -*-
# Extracted from: Chapter 104 — Sparse Retrieval, BM25, and Hybrid Search
# Source: src/.../ch104-hybrid-search.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""What k1 and b actually do, and why neither endpoint is right.

Saturation (eq:bm25-saturation) and length normalisation are usually quoted.
Here they are plotted as functions and then tested on a corpus with a planted
pathology: one document that repeats a query term many times without being about
it, and one that is genuinely relevant but long.
"""
import numpy as np

K1_VALUES = [0.0001, 0.5, 1.2, 2.0, 8.0]
B_VALUES = [0.0, 0.25, 0.75, 1.0]


def saturation(f, k1):
    return f * (k1 + 1) / (f + k1)


print("Term-frequency saturation: sat(f) = f(k1+1)/(f+k1)   (eq:bm25-saturation)")
print(f"{'k1':>10}" + "".join(f"{'f=' + str(f):>9}" for f in [1, 2, 5, 20, 100])
      + f"{'limit':>9}")
print("-" * 64)
for k1 in K1_VALUES:
    row = "".join(f"{saturation(f, k1):>9.3f}" for f in [1, 2, 5, 20, 100])
    print(f"{k1:>10.4f}{row}{k1 + 1:>9.3f}")

print("""
Every row equals exactly 1.000 at f=1 -- that is the calibration point of
eq:saturation-limits -- and every row is bounded by k1+1 no matter how large f
gets. The bound is what makes BM25 immune to keyword stuffing with no special
handling: repeating a term a hundred times cannot buy more than k1+1.

Read the extremes. At k1 -> 0 the function is a step: present or absent, count
ignored. At k1 = 8 it is still climbing at f=100. The conventional 1.2 sits far
closer to the step than to raw counts, which IS the empirical finding -- whether
a term occurs matters enormously and how often matters remarkably little.
""")

# ---- The pathology, on a small corpus ------------------------------------
CORPUS = {
    "rel-short":  {"kubernetes": 3, "scheduler": 2, "pod": 2},
    "rel-long":   {"kubernetes": 6, "scheduler": 4, "pod": 5, "filler": 85},
    "stuffed":    {"kubernetes": 40, "buy": 30, "cheap": 30},
    "off-topic":  {"database": 4, "index": 3},
}
QUERY = ["kubernetes", "scheduler"]
N_D = len(CORPUS)
lengths = {d: sum(c.values()) for d, c in CORPUS.items()}
avgdl = float(np.mean(list(lengths.values())))
idf = {}
for term in QUERY:
    n_t = sum(1 for c in CORPUS.values() if term in c)
    idf[term] = np.log(1 + (N_D - n_t + 0.5) / (n_t + 0.5))


def score(doc, k1, b):
    c = CORPUS[doc]
    total = 0.0
    for term in QUERY:
        f = c.get(term, 0)
        if f:
            total += idf[term] * f * (k1 + 1) / (
                f + k1 * (1 - b + b * lengths[doc] / avgdl))
    return total


print(f"query: {QUERY}   lengths: "
      + ", ".join(f"{d}={lengths[d]}" for d in CORPUS))
print(f"\n{'setting':<32}" + "".join(f"{d:>11}" for d in CORPUS) + "   winner")
print("-" * 80)
table = {}
for label, k1, b in [("k1->0    (presence only)", 0.0001, 0.75),
                     ("k1=1.2,  b=0    (no len norm)", 1.2, 0.0),
                     ("k1=1.2,  b=0.75 (standard)", 1.2, 0.75),
                     ("k1=1.2,  b=1    (full len norm)", 1.2, 1.0),
                     ("k1=8,    b=0.75 (weak satur.)", 8.0, 0.75)]:
    s = {d: score(d, k1, b) for d in CORPUS}
    table[label] = s
    print(f"{label:<32}" + "".join(f"{s[d]:>11.3f}" for d in CORPUS)
          + f"   {max(s, key=s.get)}")

std = table["k1=1.2,  b=0.75 (standard)"]
weak = table["k1=8,    b=0.75 (weak satur.)"]
no_norm = table["k1=1.2,  b=0    (no len norm)"]
full_norm = table["k1=1.2,  b=1    (full len norm)"]

print(f"""
Follow the STUFFED column. That document contains "kubernetes" forty times and is
about nothing. Under standard saturation it scores {std['stuffed']:.3f}, well
below both genuinely relevant documents. Weaken the saturation to k1=8 and it
jumps to {weak['stuffed']:.3f} -- a factor of
{weak['stuffed'] / std['stuffed']:.1f} -- close enough to the relevant long
document's {weak['rel-long']:.3f} to start displacing real results. The cap at
k1+1 in eq:saturation-limits is doing the anti-spam work, and it needs no rule
about repetition to do it.

Now the two length settings, on the two documents that are BOTH genuinely
relevant. With b=0 the long document wins ({no_norm['rel-long']:.3f} against
{no_norm['rel-short']:.3f}) purely by containing more of everything -- the
spurious advantage of section 5.3. Turn normalisation on and the ordering flips:
at the standard b=0.75 the short document leads {std['rel-short']:.3f} to
{std['rel-long']:.3f}, and at b=1 the gap widens further to
{full_norm['rel-short']:.3f} against {full_norm['rel-long']:.3f}.

That widening is the argument against b=1. Full normalisation divides length out
entirely, which assumes a document is long ONLY because it is padded -- so it
penalises a thorough document exactly as hard as a stuffed one. The conventional
0.75 is the claim that most of a long document's advantage is spurious but not
all of it, and b is the parameter genuinely worth tuning per corpus: uniform
product descriptions and mixed abstracts want different answers.""")
