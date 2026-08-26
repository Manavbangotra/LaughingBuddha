# Extracted from: Chapter 84 — Instruction Tuning
# Source: src/.../ch084-instruction-tuning.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Held-out task clusters: does instruction tuning teach tasks, or following?"""
import numpy as np

rng = np.random.default_rng(0)

CLUSTERS = ["sentiment", "nli", "summarisation", "qa", "translation",
            "classification", "reasoning", "extraction"]
D = 32


def cluster_vector(name):
    r = np.random.default_rng(abs(hash(name)) % (2 ** 31))
    v = r.normal(size=D)
    return v / np.linalg.norm(v)


# Two components of task performance:
#   - a task-specific part, only learnable from that cluster's examples
#   - a shared "follow the instruction" part, learnable from ANY cluster
task_dirs = {c: cluster_vector(c) for c in CLUSTERS}
follow_dir = np.ones(D) / np.sqrt(D)


def train(seen_clusters):
    """Returns learned task-specific strength per cluster, and follow strength."""
    task_strength = {c: (1.0 if c in seen_clusters else 0.0) for c in CLUSTERS}
    # Instruction-following accrues with the NUMBER OF DISTINCT clusters seen,
    # with diminishing returns — this is wei2022flan's secondary finding.
    follow = 1 - np.exp(-len(seen_clusters) / 3.0)
    return task_strength, follow


def performance(task_strength, follow, cluster):
    base = 0.25                                   # chance level
    return base + 0.45 * task_strength[cluster] + 0.30 * follow


print("Held-out cluster evaluation (eq:held-out-cluster)\n")
print(f"{'clusters trained on':>20} {'held-out perf':>15} {'seen-task perf':>16}")
held_out = "reasoning"
pool = [c for c in CLUSTERS if c != held_out]
for k in range(0, len(pool) + 1):
    seen = pool[:k]
    ts, fol = train(seen)
    held = performance(ts, fol, held_out)
    seen_perf = (np.mean([performance(ts, fol, c) for c in seen])
                 if seen else float("nan"))
    print(f"{k:>20} {held:>15.3f} {seen_perf:>16.3f}")

ts0, f0 = train([])
ts_all, f_all = train(pool)
print(f"\nheld-out '{held_out}' with no instruction tuning : "
      f"{performance(ts0, f0, held_out):.3f}")
print(f"held-out '{held_out}' after 7 other clusters      : "
      f"{performance(ts_all, f_all, held_out):.3f}")
print(f"improvement on a task type never seen             : "
      f"{performance(ts_all, f_all, held_out) - performance(ts0, f0, held_out):+.3f}")

assert performance(ts_all, f_all, held_out) > performance(ts0, f0, held_out)

print("""
The held-out column rises even though not one example of that task type was in
the training data. Nothing task-specific was learned for it — the model's
task_strength for 'reasoning' is zero throughout. What improved is the shared
instruction-following component, which accrues from cluster DIVERSITY and
saturates.

That is wei2022flan's result in miniature, and it is the reason instruction
tuning is a general capability unlock rather than a way of teaching specific
tasks: the thing being learned is 'a request should be answered', and that
transfers to requests you never trained on.""")
