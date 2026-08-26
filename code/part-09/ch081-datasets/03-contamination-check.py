# -*- coding: utf-8 -*-
# Extracted from: Chapter 81 — Pretraining Dataset Construction and Curation
# Source: src/.../ch081-datasets.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""n-gram contamination detection, and what it cannot see."""

N_GRAM = 8

BENCHMARK = [
    "what is the capital city of the republic of france in europe",
    "compute the derivative of the function f of x equals x squared plus three",
    "who wrote the novel one hundred years of solitude in nineteen sixty seven",
]

TRAINING_DOCS = {
    "clean-1": "the weather in paris is generally mild during the spring months",
    "clean-2": "derivatives measure how quickly a function changes at a point",
    "verbatim": ("a quiz question asks what is the capital city of the republic "
                 "of france in europe and the answer is paris"),
    "paraphrase": ("france is a european republic whose capital city is paris, "
                   "a fact commonly tested in geography examinations"),
    "answer-only": ("the answer to the famous solitude authorship question is "
                    "gabriel garcia marquez who published it in 1967"),
}


def ngrams(text, n=N_GRAM):
    toks = text.split()
    return {" ".join(toks[i:i + n]) for i in range(max(0, len(toks) - n + 1))}


bench_ngrams = set()
for item in BENCHMARK:
    bench_ngrams |= ngrams(item)

print(f"benchmark: {len(BENCHMARK)} items, {len(bench_ngrams)} distinct "
      f"{N_GRAM}-grams\n")
print(f"{'document':<14} {'overlap':>9} {'flagged':>9}  note")
for name, doc in TRAINING_DOCS.items():
    overlap = len(ngrams(doc) & bench_ngrams)
    flagged = overlap > 0
    note = ""
    if name == "paraphrase" and not flagged:
        note = "<- CONTAMINATED but invisible to n-grams"
    if name == "answer-only" and not flagged:
        note = "<- teaches the answer, no string overlap"
    print(f"{name:<14} {overlap:>9} {str(flagged):>9}  {note}")

detected = sum(1 for n, d in TRAINING_DOCS.items() if ngrams(d) & bench_ngrams)
truly_contaminated = 3          # verbatim, paraphrase, answer-only
print(f"\ndetected {detected} of {truly_contaminated} genuinely contaminated "
      f"documents -> recall {detected / truly_contaminated:.0%}")

# Sensitivity to n: shorter n-grams catch more and produce false positives.
print(f"\n{'n':>4} {'flagged docs':>13} {'note':<40}")
for n in (5, 8, 13, 20):
    bench_n = set()
    for item in BENCHMARK:
        bench_n |= ngrams(item, n)
    hits = sum(1 for d in TRAINING_DOCS.values() if ngrams(d, n) & bench_n)
    note = ("catches more, risks false positives" if n <= 5
            else "misses paraphrase and restatement")
    print(f"{n:>4} {hits:>13} {note:<40}")

print("""
Equation (eq:contamination) detects verbatim overlap and nothing else. The
paraphrase and the bare answer are contamination by any reasonable definition
and neither leaves an n-gram trace, so every published contamination rate is a
LOWER BOUND. "We decontaminated against benchmark X" means exact n-gram matches
were removed — which is worth doing and is not the same as a clean evaluation.""")
