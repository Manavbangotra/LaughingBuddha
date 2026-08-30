# -*- coding: utf-8 -*-
# Extracted from: Chapter 234 — Mixture of Experts and Sparse Models
# Source: src/.../ch234-mixture-of-experts.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Routing is a load-balancing problem with a quality term, and the two pull opposite ways.

An MoE layer only works if tokens spread across experts. Learned routing does not spread them:
gates that specialise are gates that concentrate, and concentration is what makes an expert
useful.

Two consequences follow. Hardware needs a fixed per-expert capacity, so an imbalanced batch
either drops tokens or wastes compute, and the capacity factor picks the mix
(eq:capacity-factor-trades-dropped-tokens-for-wasted-compute).

And the auxiliary loss that forces balance is a direct tax on the specialisation that motivated
the architecture (cite:shazeer2017moe, cite:fedus2021switch). Push it hard enough and every
expert becomes the average expert (eq:balance-and-quality-are-opposed).
"""
import math

EXPERTS, TOPK, TOKENS = 64, 2, 4096
BASE_LOSS, SPEC_GAIN, DROP_PENALTY = 2.050, 0.300, 0.550
MAX_SKEW = 1.20


def loads(skew):
    """Expected tokens per expert under a power-law routing distribution."""
    w = [(i + 1) ** -skew for i in range(EXPERTS)]
    z = sum(w)
    return [TOKENS * TOPK * x / z for x in w]


def capacity_split(skew, factor):
    """Dropped and wasted token-slots at a given capacity factor."""
    ld = loads(skew)
    cap = factor * TOKENS * TOPK / EXPERTS
    dropped = sum(max(0.0, x - cap) for x in ld)
    wasted = sum(max(0.0, cap - x) for x in ld)
    allocated = factor * TOKENS * TOPK
    return dropped / (TOKENS * TOPK), wasted / allocated, cap


def specialisation(skew):
    """How concentrated the routing is, normalised against the steepest gate."""
    return min(1.0, skew / MAX_SKEW)


print("How unevenly a learned router actually spreads tokens.")
print()
print(f"{'routing skew':>15}{'busiest expert':>17}{'quietest':>12}"
      f"{'max / mean':>13}{'specialisation':>17}")
print("-" * 74)
for s in (0.0, 0.3, 0.6, 0.9, 1.2):
    ld = loads(s)
    mean = sum(ld) / EXPERTS
    print(f"{s:>15.2f}{max(ld):>17.1f}{min(ld):>12.1f}"
          f"{max(ld) / mean:>13.2f}{specialisation(s):>17.3f}")

print()
print(f"a perfectly uniform router puts {TOKENS * TOPK / EXPERTS:.0f} tokens on each")
print("of 64 experts; a mildly skewed one puts several times that on the first")

print()
print()
print("Capacity is fixed per expert, so imbalance costs one of two ways.")
print()
SKEW = 0.6
print(f"at routing skew {SKEW:.2f}")
print()
print(f"{'capacity factor':>18}{'slots per expert':>19}{'tokens dropped':>17}"
      f"{'capacity wasted':>18}{'compute multiple':>19}")
print("-" * 91)
for cf in (1.0, 1.1, 1.25, 1.5, 2.0, 3.0):
    dropped, wasted, cap = capacity_split(SKEW, cf)
    print(f"{cf:>18.2f}{cap:>19.0f}{dropped:>17.2%}{wasted:>18.2%}"
          f"{cf:>19.2f}x")

print()
print(f"at capacity factor 1.0, {capacity_split(SKEW, 1.0)[0]:.1%} of routed tokens are dropped")
print(f"at 2.0, {capacity_split(SKEW, 2.0)[0]:.1%} still are, and"
      f" {capacity_split(SKEW, 2.0)[1]:.1%} of the allocated compute is idle")

print()
print()
print("The auxiliary balance loss, which is the usual fix.")
print()


def skew_under(alpha):
    """Routing skew as the balance loss is strengthened."""
    return 0.90 * math.exp(-3.2 * alpha)


CF = 1.25
print(f"at capacity factor {CF:.2f}")
print()
print(f"{'balance weight':>17}{'routing skew':>15}{'max / mean':>13}"
      f"{'dropped':>10}{'specialisation':>17}{'model loss':>13}")
print("-" * 85)
results = {}
for alpha in (0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.60, 0.80):
    s = skew_under(alpha)
    ld = loads(s)
    mean = sum(ld) / EXPERTS
    dropped, wasted, cap = capacity_split(s, CF)
    spec = specialisation(s)
    l = BASE_LOSS - SPEC_GAIN * spec ** 0.5 + DROP_PENALTY * dropped
    results[alpha] = (s, max(ld) / mean, dropped, spec, l)
    print(f"{alpha:>17.2f}{s:>15.3f}{max(ld) / mean:>13.2f}"
          f"{dropped:>10.2%}{spec:>17.3f}{l:>13.4f}")

BEST_A = min(results, key=lambda a: results[a][4])
print()
print(f"best loss at balance weight {BEST_A:.2f}: {results[BEST_A][4]:.4f}")
print(f"no balance loss at all: {results[0.0][4]:.4f}"
      f" (+{results[0.0][4] - results[BEST_A][4]:.4f}, {results[0.0][2]:.1%} dropped)")
print(f"heavy balance loss:     {results[0.80][4]:.4f}"
      f" (+{results[0.80][4] - results[BEST_A][4]:.4f},"
      f" specialisation {results[0.80][3]:.3f})")

print()
print()
print("Both failure directions cost about the same, for opposite reasons.")
print()
print(f"{'setting':>26}{'what goes wrong':>37}{'loss above best':>18}")
print("-" * 81)
for alpha, why in ((0.00, "a quarter of tokens dropped"),
                   (0.10, "still dropping heavily"),
                   (0.20, "nearly balanced, still specialised"),
                   (0.30, "the compromise"),
                   (0.60, "experts becoming similar"),
                   (0.80, "every expert is the average expert")):
    tag = "  <- best" if alpha == BEST_A else ""
    print(f"{f'balance weight {alpha:.2f}':>26}{why:>37}"
          f"{results[alpha][4] - results[BEST_A][4]:>+18.4f}{tag}")

print()
print("The optimum is interior and neither end is safe")
print("(eq:balance-and-quality-are-opposed).")

print()
print()
print("And one more thing the router decides: how much of the model a single")
print("request touches.")
print()
print(f"{'sequence length':>18}{'expected experts touched':>27}"
      f"{'share of weights read':>24}{'effective sparsity':>21}")
print("-" * 90)
for seq in (1, 8, 64, 512, 4096):
    frac = 1.0 - (1.0 - TOPK / EXPERTS) ** seq
    print(f"{seq:>18}{frac * EXPERTS:>27.1f}{frac:>24.1%}"
          f"{1 / max(frac, 1e-9):>20.1f}x")

SEQ = 512
FRAC = 1.0 - (1.0 - TOPK / EXPERTS) ** SEQ
print()
print(f"a single {SEQ}-token request touches {FRAC:.1%} of the experts")
print(f"the model is {EXPERTS // TOPK}x sparse per token and"
      f" {1 / FRAC:.1f}x sparse per request")

print(f"""
The first table is the fact that makes routing hard. A uniform router would put
{TOKENS * TOPK / EXPERTS:.0f} tokens on each of {EXPERTS} experts. A router with skew
{SKEW:.2f} -- mild by any standard -- puts {max(loads(SKEW)) / (TOKENS * TOPK / EXPERTS):.2f}
times the mean on its busiest expert and specialises at {specialisation(SKEW):.3f}.

Those two columns are the same phenomenon. **A gate that has learned something puts related
tokens together**, and putting related tokens together is exactly what unbalances the load.
Balance and specialisation are not independent objectives that happen to conflict; they are the
same quantity read in two directions.

The capacity table is what that costs on hardware
(eq:capacity-factor-trades-dropped-tokens-for-wasted-compute). Each expert gets a fixed number
of slots, `capacity factor x tokens x k / experts`. At factor 1.0 the allocation exactly matches
the average and **{capacity_split(SKEW, 1.0)[0]:.1%} of routed tokens are dropped** -- they skip
the layer entirely, which is a silent quality loss rather than an error. Doubling the capacity
to factor 2.0 costs twice the compute, still drops {capacity_split(SKEW, 2.0)[0]:.1%}, and
leaves {capacity_split(SKEW, 2.0)[1]:.1%} of the allocated slots idle.

**Paying twice for the compute removes less than two thirds of the drops**, because the
distribution has a tail and capacity is uniform. There is no setting that avoids both, the
capacity factor only picks the mix, and the usual choice of about {CF:.2f} is a compromise
arrived at by measuring rather than by principle.

The balance-loss table is the standard remedy and it has the shape the section title promised.
With no auxiliary loss, {results[0.0][2]:.1%} of tokens are dropped and the model loses
**{results[0.0][4] - results[BEST_A][4]:.4f}** against the best setting. With a heavy one,
specialisation falls to {results[0.80][3]:.3f} and it loses
**{results[0.80][4] - results[BEST_A][4]:.4f}** -- comparable damage, for the opposite reason.

**The optimum is interior and both ends are bad** (eq:balance-and-quality-are-opposed). At
balance weight {BEST_A:.2f} the model reaches {results[BEST_A][4]:.4f}, with
{results[BEST_A][2]:.1%} dropped and specialisation {results[BEST_A][3]:.3f}.

That is worth stating as a warning, because the two failures look completely different in
practice. Too little balance shows up as throughput collapse and dropped tokens -- visible,
alarming, quickly fixed. Too much shows up as a model that trains fine, serves fine, balances
beautifully, and is quietly worth less than a dense model of the same active size. **One failure
pages you and the other does not**, and the second is the one that survives to production.

The last table returns to the serving question from the first listing, and sharpens it. The
model is {EXPERTS // TOPK}x sparse per token -- {TOPK} experts of {EXPERTS}. But routing is
per-token, so a single {SEQ}-token request touches **{FRAC:.1%} of the experts**, and its
effective sparsity is {1 / FRAC:.1f}x rather than {EXPERTS // TOPK}x.

**Sparsity is a property of a token and density is a property of a request.** Everything about
serving -- the weights that must be resident, the bytes read, the accelerators required -- is
charged at the request level, which is why the first listing's serving penalty exists at all.""")
