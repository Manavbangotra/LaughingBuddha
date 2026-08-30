---
id: res-moe
number: 234
part: XXVIII
tier: full
status: draft
requires: [sparsity-erodes-with-batch-size, training-economics-are-not-serving-economics,
           decode-is-bandwidth-bound, the-training-optimum-is-not-the-deployment-optimum]
provides: [sparsity-moves-the-bottleneck-from-flops-to-memory,
           sparsity-trades-training-cost-for-serving-cost,
           capacity-factor-trades-dropped-tokens-for-wasted-compute,
           balance-and-quality-are-opposed]
citations: [shazeer2017moe, fedus2021switch, pope2022inference, shoeybi2019megatron]
---

## 1. Learning Objectives

By the end of this chapter you will be able to compare a sparse model against the dense model
that matches its quality rather than its FLOPs, and compute the training saving; show why sparse
parameters cost the same bytes as dense ones and what that does to serving throughput; compute
the break-even serving volume at which a sparse design stops being the cheaper system; derive
the drop/waste split a capacity factor implies from a routing load distribution; and locate the
interior optimum of an auxiliary balance loss, explaining why both ends are bad.

## 2. Why This Matters

A mixture-of-experts layer replicates the feed-forward block $E$ times and routes each token to
$k$ of them ({{cite:shazeer2017moe}}, {{cite:fedus2021switch}}). Parameters grow by $E$, FLOPs
by $k$. That decoupling is real, and the conclusion usually drawn from it is not.

Compared against the dense model that reaches the same loss — not the same FLOPs — a
128-expert, $k=4$ design trains **3.2× cheaper**. That is the genuine win and it is a training
win.

Serving runs the other way. Weights are read from memory and sparse bytes cost what dense bytes
cost ({{eq:sparsity-moves-the-bottleneck-from-flops-to-memory}}): **1103.8 GB** against
**12.9**, **14** accelerators against **1**, and **5.10×** the cost per served token against the
dense model it matches.

So there is a break-even, and it is a quantity a product manager owns:
**2.02 × 10¹² tokens served** ({{eq:sparsity-trades-training-cost-for-serving-cost}}).

Then routing, which has to work at all. A mildly skewed router puts **5.67×** the mean load on
its busiest expert; at capacity factor 1.0 that drops **25.75%** of routed tokens, and doubling
the compute still drops **10.90%**
({{eq:capacity-factor-trades-dropped-tokens-for-wasted-compute}}). The balance loss that fixes
it taxes the specialisation that motivated the architecture — no balance costs **+0.0597** loss,
heavy balance costs **+0.0456** ({{eq:balance-and-quality-are-opposed}}).

## 3. Prerequisites

{{eq:sparsity-erodes-with-batch-size}} from {{ch:inf-parallelism}} is the mechanism this chapter
prices: expected experts touched rises with batch, and the sparse model converges to the dense
one it contains.

{{eq:training-economics-are-not-serving-economics}} from the same chapter is the qualitative
result; the break-even in tokens served is the quantitative form.

{{eq:decode-is-bandwidth-bound}} from {{ch:inf-cpu-gpu}} is why bytes and not FLOPs decide
serving throughput, and it is the reason a 3× FLOP increase can be an 86× byte increase and
still be described as cheap.

{{eq:the-training-optimum-is-not-the-deployment-optimum}} from {{ch:res-scaling}} reached the
same conclusion through model size; this chapter reaches it through sparsity, and the agreement
between two unrelated mechanisms is the strongest signal in the part.

## 4. Intuitive Explanation

Here is the pitch, and it is a good one. A transformer's feed-forward blocks hold most of its
parameters and each token uses all of them. That looks wasteful: a token about protein folding
and a token about Portuguese grammar are processed by the same weights.

So replicate the block $E$ times, add a small learned gate, and send each token to the $k$
experts the gate likes best. Parameters go up by roughly $E$. FLOPs per token go up by roughly
$k$. If $k \ll E$ you have bought a much larger model for almost no extra arithmetic.

That is true, and the arithmetic supports it. In {{sec:9-practical-example}}, a 128-expert
$k=4$ configuration uses **3.0×** the FLOPs per token of the dense baseline and reaches a loss
of **1.851** against **2.038** — a real gain that a dense model would need far more compute to
match.

Now the question almost nobody asks in that comparison: *how much* more?

The standard presentation compares a sparse model against a dense model with the **same FLOPs**,
which is the comparison sparsity wins by construction. The useful comparison is against the
dense model with the **same loss**.

Do that. The 128-expert design reaches 1.851, and a dense model needs **6.25 × 10¹⁰**
parameters to reach the same number. The sparse model has 1.93 × 10¹⁰ active parameters, so it
trains **3.2× cheaper** than the dense model it matches.

**That is the real result, and it should be stated as a training result**, because everything
else about it goes the other way.

Look at the weights. The sparse model holds **5.52 × 10¹¹** parameters — **1103.8 GB** at two
bytes each, against the baseline's **12.9 GB**. That is **14** accelerators to hold the weights
instead of one, before a single request arrives.

Why does that matter so much? Because of {{eq:decode-is-bandwidth-bound}} from
{{ch:inf-cpu-gpu}}: generating a token reads weights from memory and does very little arithmetic
per byte read. Throughput is set by bandwidth, not by FLOPs.

And **a sparse parameter costs exactly as many bytes as a dense one**
({{eq:sparsity-moves-the-bottleneck-from-flops-to-memory}}). The whole saving was in a currency
the serving path does not spend.

There is a second effect on top, and {{ch:inf-parallelism}} named it. A dense model amortises one
weight read across the whole batch. An MoE reads an expert if *any* token in the batch routes to
it. At batch 1 with $k=4$ you read four experts to produce one token; at batch 512 with 64
experts you read all of them ({{eq:sparsity-erodes-with-batch-size}}).

Put the numbers together at batch 512. The dense baseline serves **76,834** tokens per second
and is compute-bound — the good regime. The 128-expert model serves **21,755** tokens per second
across 14 accelerators and is memory-bound. Per served token: **$0.012** against **$0.572**.

Against the dense model that *matches its quality*, the sparse model costs **5.10×** per served
token.

So: **3.2× cheaper to train, 5.10× more expensive to serve.** Sparsity did not make the system
cheaper. It moved the cost from one budget to another.

Which means there is a break-even, and it is computable. The training saving here is worth about
**$929,650**. The serving penalty is **$0.460** per million tokens. Those cross at
**2.02 × 10¹² tokens served** ({{eq:sparsity-trades-training-cost-for-serving-cost}}).

The other rows put that number in context, and they do not line up the way intuition suggests.
The 8-expert $k=1$ configuration breaks even at **6.98 × 10¹²** tokens and the 8-expert $k=2$
at **1.50 × 10¹³** — *further out* than the far larger 128-expert design, because their serving
penalties are small enough that the modest training saving takes a long time to be eaten. The
aggressive configurations are the ones whose economics resolve soonest, in both directions:
they save the most up front and lose the most per token afterwards. **Sparsity's risk profile
sharpens with expert count**, and a cautious 8-expert design is not a smaller version of the
same bet.

Two trillion tokens is a real quantity with a real meaning. A mid-sized product reaches it in
months. A research artefact never does. **The same architecture is the correct decision for one
team and the wrong one for another, and the deciding number is a traffic forecast.**

That is {{ch:res-scaling}}'s result reached through a different mechanism — that chapter through
FLOPs and model size, this one through bandwidth and expert count — and the agreement is worth
more than either finding alone.

Now the half that decides whether any of it works: the routing.

An MoE only functions if tokens spread across experts. Learned routing does not spread them, and
the reason is structural rather than a training bug.

A gate that has learned something puts related tokens together. That is what specialisation
*is*. And putting related tokens together is exactly what unbalances the load. Balance and
specialisation are not independent goals that happen to conflict; **they are one quantity read
in two directions.**

Measure it. A uniform router puts **128** tokens on each of 64 experts. A router with a mild
skew of 0.60 puts **5.67×** the mean on its busiest expert.

Hardware cannot accommodate that, because an expert's compute is allocated in advance. Each
expert gets a fixed number of slots — the *capacity factor* times the average load. Tokens
beyond an expert's capacity are dropped: they skip the layer entirely, which is a silent quality
loss rather than an error anything reports.

At capacity factor 1.0 — allocate exactly the average — **25.75%** of routed tokens are dropped,
and **25.75%** of allocated slots sit idle at the same time, because the same skew that overflows
the head starves the tail.

Raise the factor. At 1.25, 20.11% dropped and 36.09% wasted. At 2.0 — paying twice the compute —
**10.90%** still dropped and **55.45%** of slots idle.

**Paying twice for the compute removes less than two thirds of the drops**, because the load
distribution has a tail and capacity is uniform. There is no setting that avoids both
({{eq:capacity-factor-trades-dropped-tokens-for-wasted-compute}}); the capacity factor only
picks the mix, and the conventional choice near 1.25 is a compromise arrived at by measuring.

So force the router to balance. That is the auxiliary balance loss, and it works — it is the
standard remedy for exactly this reason.

It also taxes the thing you built the architecture for.

Sweep it at capacity factor 1.25. With no balance loss, skew 0.900, **36.66%** of tokens
dropped, specialisation 0.750, loss **1.9918**. Turn it up: at weight 0.30, skew 0.345, 7.80%
dropped, specialisation 0.287, loss **1.9322** — the best in the sweep. Turn it up further: at
0.80, skew 0.070, **nothing** dropped, specialisation 0.058, loss **1.9778**.

**Both ends are bad and the optimum is interior** ({{eq:balance-and-quality-are-opposed}}). No
balance costs **+0.0597**; heavy balance costs **+0.0456** — comparable damage, opposite causes.

That symmetry in the numbers hides a dangerous asymmetry in practice, and it is the warning this
chapter exists to give.

Too little balance is loud. Tokens drop, throughput collapses, capacity errors appear in logs,
somebody is paged. It gets fixed within a day.

Too much balance is silent. The model trains fine. Load is beautifully even. Utilisation charts
look excellent. Nothing is dropped, nothing errors, and the model is quietly worth less than a
dense model of the same active size — because every expert has converged toward the average
expert and you are paying 86× the bytes for a model that has stopped specialising.

**One failure mode pages you and the other does not**, and the second is the one that survives
to production. The only instrument that catches it is a specialisation measurement — routing
entropy, or expert-conditional domain distributions — which is not in any default dashboard.

One last number, which sharpens the serving story. The model is **32× sparse per token**: 2
experts out of 64. But routing happens per token, so a request of any length touches more. At 8
tokens, 22.4% of experts. At 64 tokens, 86.9%. At **512 tokens, 100.0%**.

**Sparsity is a property of a token; density is a property of a request.** And every serving
cost — resident weights, bytes read, accelerators required — is charged per request.

It is worth being clear about what that does and does not undermine. The architecture is not
failing at anything it promised. Each token really is processed by two experts out of
sixty-four, the arithmetic really is what a much smaller model would do, and the training saving
really is banked. What has happened is that the unit of accounting changed underneath the claim.
The research result is stated per token because per token is where the mechanism lives; the bill
is computed per request because a request is what a user sends. Nobody misrepresented anything,
and the two numbers differ by a factor of thirty-two.

Which is the whole chapter in one line: the sparsity is real where the training happens, and it
has largely evaporated by the time anyone is billed for it.

## 5. Formal Explanation

**Parameter and FLOP accounting.** For a transformer with $L$ layers, model dimension $d$,
attention parameters $4d^2$ per layer and a feed-forward block of $8d^2$ per layer:

$$N_{\text{total}} = L(4d^2 + 8Ed^2), \qquad N_{\text{active}} = L(4d^2 + 8kd^2)$$

Serving reads $N_{\text{total}}$ scaled by expert coverage; training and per-token arithmetic
scale with $N_{\text{active}}$. The two quantities diverge by roughly $E/k$.

**Quality credit.** Model sparse capacity as contributing sublinearly:
$N_{\text{eff}} = N_{\text{active}} (N_{\text{total}}/N_{\text{active}})^{\rho}$ with
$\rho \in (0,1)$, then apply a dense scaling law to $N_{\text{eff}}$. Everything in this chapter
is monotone in $\rho$ and the qualitative conclusions do not depend on its value; $\rho$ sets
where the break-even sits, not whether one exists.

**Expert coverage.** With $m$ tokens routed independently to $k$ of $E$ experts, the expected
covered fraction is $1 - (1 - k/E)^m$ — {{eq:sparsity-erodes-with-batch-size}}. It reaches 1
exponentially in $m$, so coverage is complete long before batch sizes reach production values.

**Capacity.** With per-expert loads $\ell_i$ and capacity $c = f \cdot \bar\ell$, dropped tokens
are $\sum_i (\ell_i - c)^+$ and idle slots $\sum_i (c - \ell_i)^+$. Both are positive whenever
the load distribution is non-degenerate; only $f \to \infty$ eliminates drops, and it eliminates
none of the waste.

**The balance trade.** Writing loss as
$L = L_0 - g(\sigma) + p \cdot \text{drop}(\sigma, f)$ for specialisation $\sigma$, with $g$
increasing and drop increasing in $\sigma$, the stationary condition is
$g'(\sigma^\star) = p \cdot \partial\text{drop}/\partial\sigma$. Both terms are positive, so the
optimum is interior for any $p > 0$ and any strictly increasing $g$.

## 6. Mathematical Foundation

Sparse parameters are free in FLOPs and full price in bytes:

$$\text{FLOPs} \propto N_{\text{active}} = 3.0\times, \qquad \text{bytes} \propto N_{\text{total}} = 85.7\times$$ (eq:sparsity-moves-the-bottleneck-from-flops-to-memory)

**1103.8 GB** against **12.9 GB**, **14** accelerators against **1**.

Which converts one budget into another:

$$T^\star = \frac{\text{training saved}}{\text{serving penalty per token}} = 2.02 \times 10^{12} \ \text{tokens}$$ (eq:sparsity-trades-training-cost-for-serving-cost)

**3.2× cheaper to train, 5.10× more expensive to serve.**

Capacity allocation is uniform and load is not:

$$\text{drop}(f) = \sum_i (\ell_i - f\bar\ell)^+, \qquad \text{idle}(f) = \sum_i (f\bar\ell - \ell_i)^+$$ (eq:capacity-factor-trades-dropped-tokens-for-wasted-compute)

At $f = 1.0$: **25.75%** dropped. At $f = 2.0$: **10.90%** dropped, **55.45%** idle.

And the balance loss has an interior optimum:

$$g'(\sigma^\star) = p \cdot \frac{\partial\,\text{drop}}{\partial \sigma}, \qquad \Delta L = +0.0597 \ (\text{no balance}), \ +0.0456 \ (\text{heavy})$$ (eq:balance-and-quality-are-opposed)

## 7. Internal Mechanics

Why is the FLOP-matched comparison the wrong one? Because it holds fixed the resource sparsity
was designed to economise on, which guarantees the answer. It is the same error as comparing two
compression schemes at equal decode time: the comparison is well-defined and it is not the
decision anyone faces. The decision is "what is the cheapest system that reaches quality $q$",
and that requires matching on $q$.

The bandwidth mechanism deserves restating because it is counterintuitive from the training
side. During training, arithmetic intensity is high — large batches, activations reused,
matrices large — so FLOPs dominate and sparsity's saving is fully realised. During decode,
arithmetic intensity is low: each step produces one token per sequence and reads all the weights
that token needs. **The same architecture sits on opposite sides of the roofline in its two
lifecycle phases**, which is why a design evaluated in one phase misprices itself in the other.

Expert coverage's exponential form has a practical consequence worth naming. Because coverage is
$1 - (1-k/E)^m$, it saturates *fast*: with $k/E = 1/32$, coverage passes 87% by 64 tokens. There
is no operating regime at production batch sizes where the sparse model reads meaningfully fewer
bytes than the dense one it contains. The sparsity is real per token and essentially absent per
request, and no scheduling trick recovers it — routing is content-dependent, so batching by
expert affinity fights load balance, which is {{ch:inf-parallelism}}'s `affinity-fights-balance`.

The capacity mechanism is a queueing result in disguise. Uniform capacity against non-uniform
demand always wastes and always drops; only the split moves. This is the same structure as
{{ch:sec-permissions}}' approval queue and {{ch:rai-oversight}}'s review budget — a fixed
service capacity meeting a skewed arrival process — and it produces the same shape of answer:
there is no free setting, only a priced trade.

Finally, why the balance optimum is interior for *any* positive drop penalty. Specialisation
gain is concave and increasing in skew; drop cost is increasing and convex in skew. A concave
increasing benefit against a convex increasing cost has an interior maximum whenever both
derivatives are positive at zero. The result is structural, so a team that finds their optimum at
an endpoint has almost certainly mis-measured one of the two terms — usually specialisation,
because nobody measures it.

## 8. Implementation

The first listing prices the exchange.

```python {tier=A name=sparsity-moves-the-bottleneck-from-flops-to-memory}
"""Sparsity trades a training cost for a serving cost, and the exchange rate is your traffic.

A mixture-of-experts layer replicates the feed-forward block E times and routes each token to k
of them (cite:shazeer2017moe, cite:fedus2021switch). Total parameters grow by roughly E; FLOPs
per token grow by roughly k. That decoupling is the whole idea and it is real.

What it does not do is make the model cheaper to serve. A generation step reads weights and does
little arithmetic per byte read, so throughput is set by bandwidth
(cite:pope2022inference), and sparse parameters cost exactly as many bytes as dense ones
(eq:sparsity-moves-the-bottleneck-from-flops-to-memory).

So the honest framing is not "cheaper" or "better". It is an exchange: fewer training FLOPs for
a given quality, more bytes moved per served token, and a break-even measured in tokens served
(eq:sparsity-trades-training-cost-for-serving-cost).
"""
D_MODEL, LAYERS = 4096, 32
D2 = D_MODEL ** 2
ATTN_PER_LAYER = 4 * D2          # qkv + out projections
FFN_PER_LAYER = 8 * D2           # up + down, 4x expansion
BYTES = 2                        # bf16 weights
RHO = 0.35                       # sparse parameters are worth this much, in log space
E_FLOOR, A_RED, ALPHA = 1.69, 753.6, 0.34
HBM_BW = 3.35e12                 # bytes/s per accelerator
FLOPS = 9.9e14                   # bf16 FLOP/s per accelerator
GPU_HOUR = 3.20
TRAIN_TOKENS = 4e12


def config(experts, k):
    attn = LAYERS * ATTN_PER_LAYER
    total = attn + LAYERS * experts * FFN_PER_LAYER
    active = attn + LAYERS * k * FFN_PER_LAYER
    return total, active


def effective(total, active):
    """Dense-equivalent parameters: sparse capacity counts, sublinearly."""
    return active * (total / active) ** RHO


def loss_of(n_eff):
    return E_FLOOR + A_RED / n_eff ** ALPHA


def dense_for(target_loss):
    """Parameter count a dense model needs to reach a given loss."""
    return (A_RED / (target_loss - E_FLOOR)) ** (1 / ALPHA)


CONFIGS = [
    ("dense",           1, 1),
    ("8 experts, k=1",  8, 1),
    ("8 experts, k=2",  8, 2),
    ("64 experts, k=2", 64, 2),
    ("128 experts, k=4", 128, 4),
]

print("What sparsity buys and what it costs.")
print()
print(f"{'configuration':>20}{'total params':>15}{'active':>13}{'dense-equiv':>14}"
      f"{'loss':>9}{'weights (GB)':>15}{'80GB GPUs':>12}")
print("-" * 98)
rows = {}
for name, e, k in CONFIGS:
    total, active = config(e, k)
    neff = effective(total, active)
    gb = total * BYTES / 1e9
    rows[name] = (total, active, neff, loss_of(neff), gb, int(-(-gb // 80)), e, k)
    print(f"{name:>20}{total:>15.3e}{active:>13.3e}{neff:>14.3e}"
          f"{loss_of(neff):>9.3f}{gb:>15.1f}{-(-gb // 80):>12.0f}")

DENSE = rows["dense"]
BEST = min(rows, key=lambda n: rows[n][3])
print()
print(f"best loss: {BEST} at {rows[BEST][3]:.3f}, against {DENSE[3]:.3f} dense")
print(f"it uses {rows[BEST][1] / DENSE[1]:.1f}x the FLOPs per token"
      f" and {rows[BEST][0] / DENSE[0]:.1f}x the weights")

print()
print()
print("The fair comparison: a dense model at the same quality.")
print()
print(f"{'configuration':>20}{'loss':>9}{'dense needed':>15}{'active params':>16}"
      f"{'training FLOPs':>17}{'cheaper to train by':>22}")
print("-" * 99)
match = {}
for name, e, k in CONFIGS:
    total, active, neff, l, gb, gpus, _, _ = rows[name]
    dn = dense_for(l)
    tr_sparse = 6 * active * TRAIN_TOKENS
    tr_dense = 6 * dn * TRAIN_TOKENS
    match[name] = (dn, tr_sparse, tr_dense)
    print(f"{name:>20}{l:>9.3f}{dn:>15.3e}{active:>16.3e}"
          f"{tr_sparse:>17.2e}{tr_dense / tr_sparse:>21.1f}x")

print()
print(f"`{BEST}` trains {match[BEST][2] / match[BEST][1]:.1f}x cheaper than the dense")
print(f"model it matches ({match[BEST][0]:.2e} parameters)")

print()
print()
print("Now serving, at system level, where the bytes are charged.")
print()


def touched(experts, k, batch):
    """Expected share of experts at least one token in the batch routes to."""
    if experts == 1:
        return 1.0
    return 1.0 - (1.0 - k / experts) ** batch


def gpus_for(gb):
    return max(1, int(-(-gb // 80)))


def serve(total_gb, attn_bytes, expert_bytes, active, experts, k, batch):
    """System tokens/s and the GPU count holding the weights."""
    g = gpus_for(total_gb)
    read = attn_bytes + expert_bytes * touched(experts, k, batch)
    mem = batch / (read / (g * HBM_BW))
    comp = g * FLOPS / (2 * active)
    return min(mem, comp), g, ("memory" if mem < comp else "compute")


print(f"{'configuration':>20}{'batch':>8}{'experts touched':>18}{'GPUs':>7}"
      f"{'tokens/s':>13}{'bound by':>11}{'$ / 1M tokens':>16}")
print("-" * 93)
thr, cost = {}, {}
for name, e, k in CONFIGS:
    total, active, neff, l, gb, gpus, _, _ = rows[name]
    attn_bytes = LAYERS * ATTN_PER_LAYER * BYTES
    expert_bytes = LAYERS * e * FFN_PER_LAYER * BYTES
    for batch in (1, 32, 512):
        tps, g, bound = serve(gb, attn_bytes, expert_bytes, active, e, k, batch)
        c = g * GPU_HOUR / 3600 / tps * 1e6
        thr[(name, batch)] = tps
        cost[(name, batch)] = c
        print(f"{name:>20}{batch:>8}{touched(e, k, batch):>18.1%}{g:>7}"
              f"{tps:>13.0f}{bound:>11}{c:>16.3f}")
    print()

print()
print("Against the dense model that matches each one's quality.")
print()
print(f"{'configuration':>20}{'loss':>9}{'sparse $ / 1M':>16}"
      f"{'matched dense $ / 1M':>23}{'serving penalty':>18}")
print("-" * 86)
BATCH = 512
penalty = {}
for name, e, k in CONFIGS:
    total, active, neff, l, gb, gpus, _, _ = rows[name]
    dn = match[name][0]
    dgb = dn * BYTES / 1e9
    dtps, dg, _ = serve(dgb, dn * BYTES, 0.0, dn, 1, 1, BATCH)
    dcost = dg * GPU_HOUR / 3600 / dtps * 1e6
    penalty[name] = cost[(name, BATCH)] / dcost
    print(f"{name:>20}{l:>9.3f}{cost[(name, BATCH)]:>16.3f}"
          f"{dcost:>23.3f}{cost[(name, BATCH)] / dcost:>17.2f}x")

print()
print(f"at batch {BATCH}, `{BEST}` serves at {penalty[BEST]:.1f}x the cost of the")
print("dense model it matches, and trains at a fraction of it")

print()
print()
print("So there is a break-even, and it is measured in tokens served.")
print()
print(f"{'configuration':>20}{'training saved ($)':>21}"
      f"{'extra serving $ / 1M tok':>27}{'break-even tokens served':>27}")
print("-" * 95)
breakeven = {}
FLOP_COST = GPU_HOUR / 3600 / FLOPS      # dollars per training FLOP, roughly
for name, e, k in CONFIGS:
    if name == "dense":
        continue
    saved = (match[name][2] - match[name][1]) * FLOP_COST
    dn = match[name][0]
    dgb = dn * BYTES / 1e9
    dtps, dg, _ = serve(dgb, dn * BYTES, 0.0, dn, 1, 1, BATCH)
    dcost = dg * GPU_HOUR / 3600 / dtps * 1e6
    extra = cost[(name, BATCH)] - dcost
    be = saved / extra * 1e6 if extra > 0 else float("inf")
    breakeven[name] = be
    print(f"{name:>20}{saved:>21,.0f}{extra:>27.3f}{be:>27.3e}")

SAVED_BEST = (match[BEST][2] - match[BEST][1]) * FLOP_COST
EXTRA_BEST = SAVED_BEST / breakeven[BEST] * 1e6
print()
print("Below the break-even the sparse model is the cheaper system; above it,")
print("the dense one is.")

print(f"""
The first table is the decoupling that makes sparsity interesting. `{BEST}` reaches a loss of
{rows[BEST][3]:.3f} against the dense baseline's {DENSE[3]:.3f}, using
{rows[BEST][1] / DENSE[1]:.1f} times the FLOPs per token and
**{rows[BEST][0] / DENSE[0]:.1f} times the weights** -- {rows[BEST][4]:.0f} GB against
{DENSE[4]:.0f}, which is {rows[BEST][5]} accelerators against {DENSE[5]}.

The model credits a sparse parameter at {RHO:.2f} of a dense one in log space. That number is
the whole argument's sensitivity and it is not measured here; the structure below is what
transfers.

The matched-quality table is the comparison that should be made and usually is not. Rather than
asking "is this better than a dense model with the same FLOPs", ask what dense model reaches the
same loss. For `{BEST}` that is {match[BEST][0]:.2e} parameters, and the sparse model trains
**{match[BEST][2] / match[BEST][1]:.1f} times cheaper** than it.

**That is the real result, and it is a training result.**

The serving table is the other half (eq:sparsity-moves-the-bottleneck-from-flops-to-memory).
Every configuration is memory-bound at small batch, because a generation step reads weights and
does almost nothing per byte read. A dense model amortises one weight read across the whole
batch. **An MoE reads an expert if any token in the batch routes to it** -- so at batch 1 a
`k = 4` model reads four experts to produce one token, and at batch {BATCH} it reads all of
them.

Expert reuse is therefore a function of batch size, which is a function of concurrent traffic,
which is a product fact rather than a modelling one.

The penalty table puts the two halves together. At batch {BATCH}, `{BEST}` costs
**{penalty[BEST]:.1f} times** the dense model that matches its quality, per served token, while
having cost {match[BEST][2] / match[BEST][1]:.1f} times less to train.

**Sparsity does not make the system cheaper. It moves the cost from training to serving**
(eq:sparsity-trades-training-cost-for-serving-cost), and whether that is a good trade is
arithmetic rather than architecture.

The break-even table does that arithmetic. For `{BEST}` the training saving is worth about
${SAVED_BEST:,.0f} and the serving penalty is {EXTRA_BEST:.3f} per million tokens, which break
even at **{breakeven[BEST]:.2e} tokens served**.

Below that the sparse model is the cheaper system outright; above it the dense one is, and the
gap grows with every served token. Two trillion tokens is a real quantity -- a mid-sized product
reaches it in months and a research artefact never does -- which is exactly why the same
architecture is the right answer for one team and the wrong one for another.

Which reproduces ch:res-scaling's result in a second variable. That chapter found the
training-optimal model is not the deployment-optimal one and that the gap widens with serving
volume. This one finds the same for sparsity, from a completely different mechanism -- bandwidth
rather than FLOPs. **Two independent analyses, one conclusion: the serving forecast is a model
architecture decision**, and it is usually made by people who are not in the room.

None of this touches whether the routing works at all, which is the second listing's problem and
a different kind of failure.""")
```

## 9. Practical Example

What sparsity buys and costs:

```
       configuration   total params       active   dense-equiv     loss   weights (GB)   80GB GPUs
--------------------------------------------------------------------------------------------------
               dense      6.442e+09    6.442e+09     6.442e+09    2.038           12.9           1
      8 experts, k=1      3.651e+10    6.442e+09     1.182e+10    1.973           73.0           1
      8 experts, k=2      3.651e+10    1.074e+10     1.648e+10    1.943           73.0           1
     64 experts, k=2      2.770e+11    1.074e+10     3.349e+10    1.889          554.1           7
    128 experts, k=4      5.519e+11    1.933e+10     6.247e+10    1.851         1103.8          14
```

**3.0× the FLOPs per token and 85.7× the weights**
({{eq:sparsity-moves-the-bottleneck-from-flops-to-memory}}).

```
       configuration     loss   dense needed   active params   training FLOPs   cheaper to train by
---------------------------------------------------------------------------------------------------
               dense    2.038      6.442e+09       6.442e+09         1.55e+23                  1.0x
      8 experts, k=2    1.943      1.648e+10       1.074e+10         2.58e+23                  1.5x
     64 experts, k=2    1.889      3.349e+10       1.074e+10         2.58e+23                  3.1x
    128 experts, k=4    1.851      6.247e+10       1.933e+10         4.64e+23                  3.2x
```

Against the dense model at **the same loss**, the sparse one trains **3.2× cheaper**.

```
       configuration   batch   experts touched   GPUs     tokens/s   bound by   $ / 1M tokens
---------------------------------------------------------------------------------------------
               dense       1            100.0%      1          260     memory           3.419
               dense     512            100.0%      1        76834    compute           0.012
    128 experts, k=4       1              3.1%     14         1213     memory          10.257
    128 experts, k=4     512            100.0%     14        21755     memory           0.572
```

```
       configuration     loss   sparse $ / 1M   matched dense $ / 1M   serving penalty
--------------------------------------------------------------------------------------
               dense    2.038           0.012                  0.012             1.00x
      8 experts, k=1    1.973           0.038                  0.021             1.78x
     64 experts, k=2    1.889           0.287                  0.060             4.77x
    128 experts, k=4    1.851           0.572                  0.112             5.10x
```

**5.10× the serving cost of the dense model it matches.**

```
       configuration   training saved ($)   extra serving $ / 1M tok   break-even tokens served
-----------------------------------------------------------------------------------------------
      8 experts, k=1              115,938                      0.017                  6.981e+12
      8 experts, k=2              123,713                      0.008                  1.500e+13
     64 experts, k=2              490,377                      0.227                  2.160e+12
    128 experts, k=4              929,650                      0.460                  2.022e+12
```

**Break-even at 2.02 × 10¹² tokens served**
({{eq:sparsity-trades-training-cost-for-serving-cost}}).

The second listing prices the routing.

```python {tier=A name=capacity-factor-trades-dropped-tokens-for-wasted-compute}
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
```

```
   routing skew   busiest expert    quietest   max / mean   specialisation
--------------------------------------------------------------------------
           0.00            128.0       128.0         1.00            0.000
           0.30            321.3        92.3         2.51            0.250
           0.60            726.0        59.9         5.67            0.500
           0.90           1427.5        33.8        11.15            0.750
```

```
   capacity factor   slots per expert   tokens dropped   capacity wasted   compute multiple
-------------------------------------------------------------------------------------------
              1.00                128           25.75%            25.75%               1.00x
              1.25                160           20.11%            36.09%               1.25x
              2.00                256           10.90%            55.45%               2.00x
              3.00                384            5.33%            68.44%               3.00x
```

**Doubling the compute removes less than two thirds of the drops**
({{eq:capacity-factor-trades-dropped-tokens-for-wasted-compute}}).

```
   balance weight   routing skew   max / mean   dropped   specialisation   model loss
-------------------------------------------------------------------------------------
             0.00          0.900        11.15    36.66%            0.750       1.9918
             0.20          0.475         4.10    13.74%            0.395       1.9369
             0.30          0.345         2.86     7.80%            0.287       1.9322
             0.60          0.132         1.52     0.74%            0.110       1.9546
             0.80          0.070         1.25     0.00%            0.058       1.9778
```

**Both ends bad: +0.0597 with no balance, +0.0456 with heavy balance**
({{eq:balance-and-quality-are-opposed}}).

```
   sequence length   expected experts touched   share of weights read   effective sparsity
------------------------------------------------------------------------------------------
                 1                        2.0                    3.1%                32.0x
                 8                       14.4                   22.4%                 4.5x
                64                       55.6                   86.9%                 1.2x
               512                       64.0                  100.0%                 1.0x
```

**32× sparse per token, 1.0× sparse per request.**

## 10. Production Considerations

Compare against the dense model at matched **loss**, never matched FLOPs. The FLOP-matched
comparison is the one sparsity wins by construction.

State the decision as a break-even in tokens served and get a traffic forecast in writing. Two
trillion tokens separates the right answer from the wrong one.

Budget the weights before the arithmetic. 1103.8 GB is 14 accelerators standing idle at low
traffic, and that cost is paid whether or not anyone sends a request.

Measure specialisation continuously, not just balance. Load balance is on every dashboard and
routing entropy is on none, and the silent failure is the one only entropy sees.

Tune the capacity factor by measuring drop and idle together. Both are positive at every
setting, and reporting only one is how a team convinces itself a factor is free.

Treat dropped tokens as a quality metric, not an error rate. They do not raise exceptions and
they do change outputs.

Do not batch by expert affinity to recover sparsity. It fights load balance
({{ch:inf-parallelism}}), and coverage saturates by 64 tokens anyway.

Re-run the break-even after every serving-efficiency improvement. Quantisation and better
batching change the serving side and move the crossing point.

## 11. Common Mistakes

**Comparing at equal FLOPs.** Guarantees the conclusion; answers no real question.

**Calling a sparse model "cheap".** It is cheap in FLOPs and expensive in bytes, and serving
spends bytes.

**Ignoring resident weight cost at low traffic.** Fourteen accelerators, 1213 tokens/second, and
a bill that does not care.

**Reporting balance without specialisation.** The failure that survives to production is
invisible in the balance metric.

**Choosing a capacity factor from the drop rate alone.** Idle slots are the other half and they
are larger.

**Assuming per-token sparsity is per-request sparsity.** 32× becomes 1.0× by 512 tokens.

## 12. Failure Modes

**A sparse model shipped to a consumer product.** 5.10× serving cost, discovered as an inference
bill after launch.

**Capacity factor 1.0 chosen for efficiency.** 25.75% of tokens silently skip the layer.

**Balance loss turned up until the dashboards look clean.** Specialisation 0.058, no alerts, and
the architecture is now an expensive dense model.

**Expert-affinity batching added to recover throughput.** Load skews, drops rise, and the fix
made it worse.

**A sparse model evaluated only during training.** High arithmetic intensity, sparsity fully
realised, and the serving profile never measured.

**A break-even computed once.** The serving side improves and the crossing point moves.

## 13. Alternatives

**A dense model at the matched size.** Simpler, one accelerator here, and 5.10× cheaper per
served token; the right answer above the break-even volume.

**Fewer, larger experts.** Reduces total bytes and coverage saturation at the cost of
specialisation granularity; the 8-expert rows show the shape.

**Shared-plus-routed experts.** Keeps a dense path so no token is unserved when capacity is
exceeded, which converts drops into a quality gradient rather than a cliff.

**Quantise the experts.** Attacks the actual constraint — bytes — rather than the FLOPs, and
composes with everything else here.

**Distil the sparse model into a dense one.** Pays the training saving once and serves at dense
economics; the natural move for anything past the break-even.

## 14. Evaluation

Fit your own $\rho$ — the sublinear credit sparse parameters earn — from a ladder of small runs.
Everything in this chapter is monotone in it and nobody publishes theirs.

Measure serving cost per token at your real batch distribution, not at a benchmark batch.
{{ch:inf-edge}}'s point applies: benchmarks measure burst and users get sustained.

Compute your break-even and compare it against your twelve-month token forecast. If they are
within a factor of two, the architecture decision is a forecasting decision.

Log routing entropy and per-expert load together, every run. The pair distinguishes the two
failure modes and neither number alone does.

Measure dropped-token rate and correlate it with output quality on a held-out set. If nothing
changes, your capacity factor is too high.

## 15. Advanced Concepts

The $\rho$ parameter carries the whole quantitative argument and it is almost certainly not a
constant. Sparse capacity should help most where the data is heterogeneous — many domains, many
languages, many formats — and least where it is uniform, because specialisation has nothing to
specialise on. That predicts $\rho$ rising with corpus diversity, which would mean **the case
for sparsity is strongest exactly for the general-purpose models that are hardest to serve**.
Nothing here measures that, and it is the single most useful missing number in the chapter.

The capacity analysis assumes drops are uniformly harmful, and they are not. A dropped token
skips one layer's feed-forward block and keeps its residual stream, so the damage depends on how
much that layer contributes for that token — which is exactly the tokens the router assigned
confidently. **Drops are concentrated on the tokens the routing was most sure about**, which is
the worst possible correlation and makes the uniform-damage model in {{sec:9-practical-example}}
optimistic. A capacity policy that dropped low-confidence assignments first would be strictly
better and requires only a sort.

There is a deeper question the chapter sidesteps. Everything here treats expert count as a
hyperparameter and routing as a mechanism for using it, but the interesting version is
conditional computation *in general*: depth-adaptive models, early exit, and cascades all trade
the same currencies — arithmetic for bytes for latency — and all face the same request-level
density problem. The MoE formulation is the one with production evidence, and the analysis in
this chapter transfers to the others with the coverage term replaced.

Finally, the break-even framing assumes training cost and serving cost are fungible, and in most
organisations they are not. Training is capital, decided once, by a research organisation.
Serving is operating expense, decided continuously, by a different one. **A break-even that is
correct in aggregate can be irrelevant to both parties**, which is the honest reason sparse
models get built more often than the arithmetic supports — and it is a governance problem rather
than a technical one.

## 16. Connection to Previous Chapters

{{eq:sparsity-erodes-with-batch-size}} from {{ch:inf-parallelism}} is the coverage term that
takes 32× sparsity to 1.0× by 512 tokens; this chapter prices what that erosion costs.

{{eq:training-economics-are-not-serving-economics}} from the same chapter is the qualitative
statement; **2.02 × 10¹² tokens** is the quantitative one.

{{eq:decode-is-bandwidth-bound}} from {{ch:inf-cpu-gpu}} is why 85.7× the bytes matters more
than 3.0× the FLOPs.

{{eq:the-training-optimum-is-not-the-deployment-optimum}} from {{ch:res-scaling}} reached this
conclusion through model size and this chapter reaches it through sparsity — two mechanisms, one
answer.

## 17. Exercises

1. Recompute the matched-quality comparison for $\rho = 0.2$ and $\rho = 0.5$. How far does the
   break-even move?

2. Derive expected expert coverage and find the sequence length at which it exceeds 95% for your
   $E$ and $k$.

3. Compute drop and idle fractions for your measured load distribution across capacity factors
   1.0 to 3.0. Where is your operating point?

4. Sweep an auxiliary balance weight and locate the interior optimum. How wide is the flat
   region around it?

5. Implement {{sec:15-advanced-concepts}}'s confidence-ordered drop policy and measure the
   quality difference against uniform dropping at the same capacity.

6. Compute the break-even after applying 4-bit expert quantisation. Does the architecture
   decision reverse?

## 18. Interview Questions

1. Our MoE has 500 billion parameters and the FLOPs of a 20-billion model. What does it cost to
   serve?

2. Why compare against a dense model at the same loss rather than the same FLOPs?

3. At what traffic volume does this sparse model stop being the cheaper system?

4. Our expert load is beautifully balanced. Is that good?

5. What is a dropped token and how would you know it happened?

6. We're at batch 1. How sparse is the model really?

## 19. Research Questions

1. How does the sparse-parameter credit $\rho$ vary with corpus diversity, measured directly?

2. Do confidence-ordered capacity policies measurably outperform uniform dropping at equal
   capacity?

3. What is the relationship between routing entropy and downstream quality, across balance-loss
   settings?

4. Can expert-affinity batching and load balance be jointly optimised, or is the trade
   fundamental?

## 20. Chapter Summary

A mixture of experts decouples parameters from arithmetic, and the decoupling is real in exactly
one currency.

Against the dense model that matches its **loss** — not its FLOPs — a 128-expert $k=4$ design
trains **3.2× cheaper**. That is the genuine result and it is a training result.

Serving runs the other way, because {{ch:inf-cpu-gpu}}'s decode step reads weights and sparse
bytes cost what dense bytes cost ({{eq:sparsity-moves-the-bottleneck-from-flops-to-memory}}):
**1103.8 GB** against **12.9**, **14** accelerators against **1**, **21,755** tokens/second
against **76,834**, and **5.10×** the cost per served token. So the architecture does not make
the system cheaper — **it moves the cost from training to serving**, with a break-even at
**2.02 × 10¹² tokens served** ({{eq:sparsity-trades-training-cost-for-serving-cost}}). That is a
traffic forecast, not an architecture opinion.

Routing supplies the second trade. A gate that specialises is a gate that concentrates, so
balance and specialisation are one quantity read two ways. A skew of 0.60 puts **5.67×** the
mean on the busiest expert; at capacity factor 1.0 that drops **25.75%** of tokens and idles
**25.75%** of slots, and paying double removes less than two thirds of the drops
({{eq:capacity-factor-trades-dropped-tokens-for-wasted-compute}}). The balance loss that fixes
it has an interior optimum: **+0.0597** with none, **+0.0456** with too much
({{eq:balance-and-quality-are-opposed}}) — and only the first failure is loud.

And the sparsity itself mostly evaporates before anyone is billed: **32× per token, 1.0× per
512-token request**.

What runs through the chapter is that every headline number for this architecture is measured in
the phase where it looks best. FLOPs rather than bytes. Training rather than serving. Per token
rather than per request. Balance rather than specialisation. None of those measurements is
wrong, and each of them is taken on the favourable side of a trade the other side of which is
paid by someone else in the organisation.

Carry forward: **match on quality, then count bytes**, and **measure specialisation, not just
balance**.

## 21. Further Reading

- {{cite:shazeer2017moe}} — the sparsely-gated layer and the load-balancing problem it creates.
- {{cite:fedus2021switch}} — routing simplified to $k = 1$, with the capacity factor made
  explicit.
- {{cite:pope2022inference}} — the serving-side arithmetic that decides whether any of this is
  affordable.
- {{cite:shoeybi2019megatron}} — the parallelism machinery a model of this size requires before
  it serves a single request.
