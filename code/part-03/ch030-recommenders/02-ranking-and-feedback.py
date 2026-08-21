# Extracted from: Chapter 30 — Recommendation Systems
# Source: src/.../ch030-recommenders.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Ranking metrics, and a simulation of the popularity feedback loop.
"""
import numpy as np

rng = np.random.default_rng(3)

# --- ranking metrics ---------------------------------------------------------
def dcg_at_k(rels, k):
    rels = np.asarray(rels)[:k]
    return float(np.sum((2 ** rels - 1) / np.log2(np.arange(2, len(rels) + 2))))


def ndcg_at_k(rels, k):
    ideal = dcg_at_k(sorted(rels, reverse=True), k)
    return dcg_at_k(rels, k) / ideal if ideal > 0 else 0.0


def precision_at_k(rels, k):
    return float(np.mean(np.asarray(rels)[:k] > 0))


print("=" * 72)
print("ranking metrics: position matters (eqs. 30.10-30.11)")
print("=" * 72)
lists = {
    "relevant first":  [3, 2, 1, 0, 0],
    "relevant last":   [0, 0, 1, 2, 3],
    "one great at #1": [3, 0, 0, 0, 0],
    "three mediocre":  [1, 1, 1, 0, 0],
}
print(f"{'ranking':<20} {'P@5':>7} {'DCG@5':>9} {'NDCG@5':>9}")
for label, rels in lists.items():
    print(f"{label:<20} {precision_at_k(rels,5):>7.2f} "
          f"{dcg_at_k(rels,5):>9.3f} {ndcg_at_k(rels,5):>9.3f}")
print("\nThe first two lists contain identical items and have identical")
print("precision. NDCG separates them, because users read from the top.")

# --- eq. 30.9: the popularity feedback loop ---------------------------------
print("\n" + "=" * 72)
print("the feedback loop: exposure determines the next training set")
print("=" * 72)

n_items = 200
true_quality = rng.beta(2, 5, n_items)          # intrinsic click propensity
true_quality /= true_quality.sum()


def simulate(rounds=40, users_per_round=2000, explore=0.0, seed=1):
    r = np.random.default_rng(seed)
    interactions = np.ones(n_items)             # weak uniform prior
    exposure_history = []
    for _ in range(rounds):
        popularity = interactions / interactions.sum()
        # epsilon-greedy: mostly recommend by popularity, sometimes explore
        probs = (1 - explore) * popularity + explore / n_items
        shown = r.choice(n_items, size=users_per_round, p=probs)
        clicked = r.random(users_per_round) < true_quality[shown] * 8
        np.add.at(interactions, shown[clicked], 1)
        exposure_history.append(np.bincount(shown, minlength=n_items))
    return interactions, np.array(exposure_history)


print(f"{'exploration':>12} {'top-10 exposure share':>23} "
      f"{'items never shown':>19} {'quality-exposure corr':>23}")
for explore in (0.0, 0.05, 0.20):
    inter, hist = simulate(explore=explore)
    final = hist[-1] / hist[-1].sum()
    top10 = np.sort(final)[-10:].sum()
    never = int((hist.sum(axis=0) == 0).sum())
    corr = np.corrcoef(true_quality, final)[0, 1]
    print(f"{explore:>12.0%} {top10:>23.1%} {never:>19} {corr:>23.3f}")

print("\nWith no exploration, ten items out of 200 take most of the exposure")
print("and many are never shown at all — so their quality is never learned.")
print("Exploration costs short-term clicks and buys a catalogue the system")
print("actually knows something about (eq. 30.9).")

# --- offline evaluation is biased toward the incumbent ----------------------
print("\n" + "=" * 72)
print("why offline evaluation flatters the model that generated the logs")
print("=" * 72)

# The logging policy favours the first 50 items; the "new" model favours others.
logging_pref = np.zeros(n_items)
logging_pref[:50] = 1.0
logging_probs = (logging_pref + 0.05) / (logging_pref + 0.05).sum()

n_log = 40_000
shown = rng.choice(n_items, size=n_log, p=logging_probs)
clicked = rng.random(n_log) < true_quality[shown] * 8
logged = {"item": shown, "click": clicked}

# Two candidate models: one mimics the logging policy, one ranks by true quality.
incumbent_scores = logging_probs
challenger_scores = true_quality

def offline_ctr(scores, top_n=40):
    """Estimate CTR from logs by restricting to this model's top items —
    the standard naive offline evaluation."""
    top = set(np.argsort(-scores)[:top_n].tolist())
    m = np.isin(logged["item"], list(top))
    return logged["click"][m].mean() if m.sum() > 30 else float("nan"), int(m.sum())


inc_ctr, inc_n = offline_ctr(incumbent_scores)
cha_ctr, cha_n = offline_ctr(challenger_scores)

# The truth: expected CTR if each model's top items were actually shown.
inc_true = true_quality[np.argsort(-incumbent_scores)[:40]].mean() * 8
cha_true = true_quality[np.argsort(-challenger_scores)[:40]].mean() * 8

print(f"{'model':<14} {'offline CTR':>13} {'logged rows':>13} "
      f"{'true CTR if deployed':>22}")
print(f"{'incumbent':<14} {inc_ctr:>13.4f} {inc_n:>13,} {inc_true:>22.4f}")
print(f"{'challenger':<14} {cha_ctr:>13.4f} {cha_n:>13,} {cha_true:>22.4f}")

print(f"\nThe challenger is genuinely better ({cha_true:.3f} vs {inc_true:.3f}")
print(f"true CTR) but is evaluated on only {cha_n:,} logged rows — the few")
print("times its preferred items happened to be shown. Offline evaluation")
print("has far less evidence about the model that differs from the logger,")
print("which is exactly the model you are trying to assess.")
print("\nThis is why recommenders are decided by A/B test (Chapter 26), and")
print("why logging the propensity of each impression is worth the effort.")
