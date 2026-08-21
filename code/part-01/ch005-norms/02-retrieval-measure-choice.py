# Extracted from: Chapter 5 — Norms, Distances, and Similarity Measures
# Source: src/.../ch005-norms.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""How the choice of similarity measure changes retrieval results.

Documents differ in length, which shows up as embedding magnitude. Cosine
ignores that; the raw dot product does not.
"""
import numpy as np

rng = np.random.default_rng(3)

# Two topics as directions, plus a length factor that scales the magnitude.
topic_a = np.array([1.0, 0.2, 0.0, 0.1]); topic_a /= np.linalg.norm(topic_a)
topic_b = np.array([0.0, 0.1, 1.0, 0.2]); topic_b /= np.linalg.norm(topic_b)

corpus = [
    ("short, exactly on topic A",   topic_a, 1.0),
    ("long, exactly on topic A",    topic_a, 4.0),
    ("short, mostly topic A",       0.85 * topic_a + 0.15 * topic_b, 1.0),
    ("very long, topic B",          topic_b, 6.0),
    ("medium, topic B",             topic_b, 2.5),
]
names = [c[0] for c in corpus]
D = np.stack([length * (v / np.linalg.norm(v)) for _, v, length in corpus])

query = topic_a.copy()

dot = D @ query
cos = dot / (np.linalg.norm(D, axis=1) * np.linalg.norm(query))
euc = np.linalg.norm(D - query, axis=1)

print(f"{'document':<28} {'|d|':>6} {'dot':>7} {'cos':>7} {'euclid':>8}")
for i, name in enumerate(names):
    print(f"{name:<28} {np.linalg.norm(D[i]):>6.2f} {dot[i]:>7.3f} "
          f"{cos[i]:>7.3f} {euc[i]:>8.3f}")

print("\nTop result by each measure:")
print(f"  dot product : {names[int(np.argmax(dot))]}")
print(f"  cosine      : {names[int(np.argmax(cos))]}")
print(f"  euclidean   : {names[int(np.argmin(euc))]}")

print("\nThe dot product ranks the LONG topic-B document above the short")
print("topic-A one purely because it is longer. Cosine gets it right.")
print("Euclidean prefers the short document because the query is short —")
print("it is measuring length agreement as well as topical agreement.")
