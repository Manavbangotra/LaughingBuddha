# -*- coding: utf-8 -*-
# Extracted from: Chapter 214 — LLM Evaluation: Benchmarks and Their Limits
# Source: src/.../ch214-llm-benchmarks.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A benchmark score has no scale until somebody measures a human on the same items.

39% sounds bad. 72% sounds good. Neither statement is meaningful without knowing what a
domain expert scores on the same questions -- and cite:rein2023gpqa is the standard example
of supplying that: experts reach 65%, skilled non-experts with unrestricted web access
reach 34% (eq:a-score-needs-a-human-baseline).

The second problem is aggregation. A suite reports one number over many scenarios, and two
models with the same aggregate gain can have improved on completely different things
(eq:aggregate-hides-which-scenario-moved).

And underneath both, cite:liang2022helm's coverage result: before a deliberate effort,
models were compared on 17.9% of the same scenarios, which means most published comparisons
were between different measurements.
"""
# (scenario, weight, base model score, non-expert baseline, expert baseline)
SCENARIOS = [
    ("factual recall",       0.14, 0.81, 0.42, 0.94),
    ("multi-step reasoning", 0.16, 0.54, 0.31, 0.88),
    ("code generation",      0.13, 0.63, 0.22, 0.91),
    ("summarisation",        0.11, 0.77, 0.61, 0.86),
    ("graduate science",     0.09, 0.39, 0.34, 0.65),
    ("legal reasoning",      0.10, 0.58, 0.36, 0.83),
    ("tool use",             0.14, 0.47, 0.55, 0.90),
    ("long-context recall",  0.13, 0.71, 0.48, 0.92),
]

base = {n: s for n, w, s, ne, ex in SCENARIOS}
weight = {n: w for n, w, s, ne, ex in SCENARIOS}
nonexp = {n: ne for n, w, s, ne, ex in SCENARIOS}
expert = {n: ex for n, w, s, ne, ex in SCENARIOS}
agg = sum(weight[n] * base[n] for n in base)

print(f"An eight-scenario suite. Aggregate score: {agg:.4f}")
print()
print(f"{'scenario':>22}{'weight':>9}{'score':>8}{'non-expert':>13}"
      f"{'expert':>9}{'headroom':>11}{'% closed':>11}")
print("-" * 83)
for n, w, s, ne, ex in SCENARIOS:
    head = ex - s
    closed = (s - ne) / (ex - ne) if ex > ne else 1.0
    print(f"{n:>22}{w:>9.2f}{s:>8.3f}{ne:>13.3f}{ex:>9.3f}"
          f"{head:>11.3f}{closed:>11.0%}")

print()
print("The raw-score ranking and the headroom-closed ranking are different")
print("orderings of the same table.")
print()
by_raw = sorted(base, key=lambda n: -base[n])
by_closed = sorted(base, key=lambda n:
                   -((base[n] - nonexp[n]) / (expert[n] - nonexp[n])))
print(f"{'rank':>6}{'by raw score':>24}{'by headroom closed':>26}")
print("-" * 56)
for i in range(len(SCENARIOS)):
    print(f"{i + 1:>6}{by_raw[i]:>24}{by_closed[i]:>26}")

print()
print()
print("Two models with the same aggregate gain, improving different things.")
print()
MODEL_A = {"tool use": 0.14, "multi-step reasoning": 0.11}
MODEL_B = {"factual recall": 0.10, "summarisation": 0.08,
           "long-context recall": 0.07, "code generation": 0.02,
           "legal reasoning": 0.02}


def gain(deltas):
    return sum(weight[n] * d for n, d in deltas.items())


print(f"{'model':>9}{'scenarios improved':>21}{'aggregate gain':>17}"
      f"{'new aggregate':>16}")
print("-" * 63)
for name, d in (("A", MODEL_A), ("B", MODEL_B)):
    g = gain(d)
    print(f"{name:>9}{len(d):>21}{g:>17.4f}{agg + g:>16.4f}")

print()
print("Same headline movement. Four defensible summaries of the same pair.")
print()


def closed(n, d=0.0):
    s = min(0.999, base[n] + d)
    return (s - nonexp[n]) / (expert[n] - nonexp[n])


print(f"{'summary':>36}{'model A':>12}{'model B':>12}{'says':>10}")
print("-" * 70)
summ = {}
for label, f in (
        ("aggregate gain",
         lambda d: gain(d)),
        ("weighted headroom closed",
         lambda d: sum(weight[n] * (closed(n, d.get(n, 0.0)) - closed(n))
                       for n in base)),
        ("worst remaining gap to expert",
         lambda d: max(expert[n] - min(0.999, base[n] + d.get(n, 0.0))
                       for n in base)),
        ("scenarios below non-expert",
         lambda d: float(sum(1 for n in base
                             if base[n] + d.get(n, 0.0) < nonexp[n]))),
):
    a, b = f(MODEL_A), f(MODEL_B)
    summ[label] = (a, b)
    if abs(a - b) < 1e-3:
        verdict = "tie"
    elif label == "worst remaining gap to expert":
        verdict = "A" if a < b else "B"
    elif label == "scenarios below non-expert":
        verdict = "A" if a < b else "B"
    else:
        verdict = "A" if a > b else "B"
    print(f"{label:>36}{a:>12.4f}{b:>12.4f}{verdict:>10}")

print()
print("Three different answers from four summaries, and only the first is")
print("ever reported.")

print()
print()
print("Where each model's aggregate gain actually came from.")
print()
print(f"{'scenario':>22}{'A contributes':>16}{'share of A':>13}"
      f"{'B contributes':>16}{'share of B':>13}")
print("-" * 80)
ga, gb = gain(MODEL_A), gain(MODEL_B)
for n, w, s, ne, ex in SCENARIOS:
    ca = weight[n] * MODEL_A.get(n, 0.0)
    cb = weight[n] * MODEL_B.get(n, 0.0)
    print(f"{n:>22}{ca:>16.4f}{ca / ga:>13.0%}{cb:>16.4f}{cb / gb:>13.0%}")

top_a = max(MODEL_A, key=lambda n: weight[n] * MODEL_A[n])
top_b = max(MODEL_B, key=lambda n: weight[n] * MODEL_B[n])
sh_a = weight[top_a] * MODEL_A[top_a] / ga
sh_b = weight[top_b] * MODEL_B[top_b] / gb
print()
print(f"largest single contributor: A {sh_a:.0%} ({top_a}), "
      f"B {sh_b:.0%} ({top_b})")

print()
print()
print("And the coverage problem: three models evaluated on different subsets.")
print()
COVER = {
    "model P": ["factual recall", "summarisation", "long-context recall",
                "code generation", "tool use"],
    "model Q": ["multi-step reasoning", "graduate science", "legal reasoning",
                "tool use", "code generation"],
    "model R": ["factual recall", "summarisation", "code generation",
                "multi-step reasoning", "tool use"],
}
SKILL = {"model P": 0.00, "model Q": 0.09, "model R": 0.05}
print(f"{'model':>10}{'true skill offset':>20}{'reported mean':>16}"
      f"{'reported rank':>16}{'scenarios':>12}")
print("-" * 74)
rep = {}
for m, sc in COVER.items():
    uniq = sorted(set(sc))
    rep[m] = sum(min(0.999, base[n] + SKILL[m]) for n in uniq) / len(uniq)
order = sorted(rep, key=lambda m: -rep[m])
for m in COVER:
    print(f"{m:>10}{SKILL[m]:>20.2f}{rep[m]:>16.4f}"
          f"{order.index(m) + 1:>16}{len(set(COVER[m])):>12}")

common = set.intersection(*[set(v) for v in COVER.values()])
print()
print(f"common scenarios across all three: {len(common)} of {len(SCENARIOS)} "
      f"({len(common) / len(SCENARIOS):.1%})")
print()
fair = {m: sum(min(0.999, base[n] + SKILL[m]) for n in common) / len(common)
        for m in COVER}
forder = sorted(fair, key=lambda m: -fair[m])
print(f"{'model':>10}{'on common set':>16}{'fair rank':>12}"
      f"{'reported rank':>16}{'moved':>8}")
print("-" * 62)
for m in COVER:
    print(f"{m:>10}{fair[m]:>16.4f}{forder.index(m) + 1:>12}"
          f"{order.index(m) + 1:>16}"
          f"{order.index(m) - forder.index(m):>8}")

print(f"""
The scenario table is where the chapter's first point lives, in the last two columns.
`graduate science` reports {base['graduate science']:.3f}, which reads as a failure. Against
a non-expert baseline of {nonexp['graduate science']:.3f} and an expert ceiling of
{expert['graduate science']:.3f} it has closed
{(base['graduate science'] - nonexp['graduate science']) / (expert['graduate science'] - nonexp['graduate science']):.0%}
of the available distance (eq:a-score-needs-a-human-baseline).

`tool use` reports {base['tool use']:.3f} -- higher -- and has closed
{(base['tool use'] - nonexp['tool use']) / (expert['tool use'] - nonexp['tool use']):.0%},
because it is *below the non-expert baseline*. A number that looks better is describing a
system that is worse than an untrained human at the task.

Without the two human columns those two rows are 0.39 and 0.47 and the ordering is obvious
and wrong. **A benchmark score is a raw measurement and the baselines are its units**, which
is why cite:rein2023gpqa's decision to measure experts and non-experts under the same
conditions matters more than its headline difficulty.

The two rankings confirm it. Sorting by raw score and sorting by headroom closed produce
different orders of the same eight rows, and a roadmap built from the first is aimed at the
scenarios where the model is already doing well relative to what is achievable.

The model-comparison table is the aggregation problem. Model A improves
{len(MODEL_A)} scenarios and model B improves {len(MODEL_B)}, and the aggregate moves by
{ga:.4f} and {gb:.4f} -- **indistinguishable in the headline**
(eq:aggregate-hides-which-scenario-moved).

The four-summary table is the uncomfortable part. By the aggregate the two models are
tied -- {summ['aggregate gain'][0]:.4f} against {summ['aggregate gain'][1]:.4f}.

Measured as *headroom closed*, B is ahead
({summ['weighted headroom closed'][1]:.4f} against
{summ['weighted headroom closed'][0]:.4f}), because points are cheapest in headroom terms
exactly where headroom is smallest: improving a scenario already near its ceiling closes a
larger *fraction* of what remained.

Measured by the worst remaining gap to expert, A is ahead
({summ['worst remaining gap to expert'][0]:.2f} against
{summ['worst remaining gap to expert'][1]:.2f}), because A attacked what was furthest
behind. And by scenarios still below the non-expert baseline, A leaves
{summ['scenarios below non-expert'][0]:.0f} and B leaves
{summ['scenarios below non-expert'][1]:.0f} -- B never touched `tool use`, so it ships a
suite containing a task the model does worse than an untrained human.

**Four defensible summaries, three different answers, and only the first one is ever
reported.** That is worth sitting with, because the usual reaction to a hidden aggregate is
"report a better aggregate," and this table says there is no better aggregate -- there are
questions, and each summary answers one.

There is a Goodhart consequence hiding in the headroom row that is worth naming. If a team
optimises the aggregate, points are cheapest where the model is already strong, so the
optimisation pushes effort toward scenarios that are nearly finished and away from the ones
that are far behind. **The suite rewards polishing**, and no individual decision inside that
process looks wrong.

The contribution table says something further that the aggregate cannot. Model A's gain is
{sh_a:.0%} concentrated in `{top_a}` and model B's is {sh_b:.0%} in `{top_b}`, spread across
{len(MODEL_B)} scenarios. Report the aggregate alone and a reader assumes broad improvement
in both cases; one of them is broad and the other is two scenarios, and those are different
claims about what the model can now do.

**The decomposition costs nothing** -- it is the same numbers, not summed -- and almost no
report includes it.

The coverage table is the third failure and it is cite:liang2022helm's contribution.
Three models are evaluated on overlapping but different scenario subsets, which is how
public comparison actually works when each lab reports what it chose to run. The reported
ranking is {' > '.join(order)}.

On the {len(common)} scenarios all three actually share -- {len(common) / len(SCENARIOS):.0%}
of the suite -- the ranking is {' > '.join(forder)}.

Model Q reports worst and is best on the shared items; it is the strongest model in the
table and it ran on the hardest subset. The reported ordering is partly a fact about which
scenarios each model was run on. That is
the whole of cite:liang2022helm's coverage argument: taking coverage from
{0.179:.1%} to {0.960:.1%} was not a completeness exercise, it was **the precondition for the
comparisons to be comparisons at all.**

And cite:singh2025leaderboard's audit adds the adversarial version of the same point.
Unequal access to a leaderboard's data lets one participant fit the evaluation distribution
better than another, with relative gains reported up to {1.12:.0%} on the arena
distribution. Coverage differences and access differences are the same failure at different
scales: **the score partly measures the conditions under which it was collected.**

One thing to keep hold of. None of this says benchmarks are useless -- it says a benchmark
number is a measurement that has not yet been given units, a decomposition, or a matched
comparison set. All three are cheap. Two of them are free.""")
