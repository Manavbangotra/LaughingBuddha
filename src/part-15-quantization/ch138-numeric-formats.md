---
id: q-formats
number: 138
part: XV
tier: full
status: draft
requires: [ft-qlora-peft, dl-optimizers, math-random-vars]
provides: [format-is-a-budget, underflow-is-not-error, optimal-split,
           metric-picks-the-format, scale-factor-as-exponent, loss-scaling,
           fp8-division-of-labour]
citations: [micikevicius2018mixed, micikevicius2022fp8, dettmers2022int8,
            kumar2024precisionscaling]
---

## 1. Learning Objectives

By the end of this chapter you will be able to derive a format's range and
resolution from its exponent/mantissa split alone; explain why a value rounded to
zero is a **qualitatively different** failure from a value rounded imprecisely;
state the design rule for choosing the split, and why the two obvious scoring
metrics disagree about it; explain what a per-tensor scale factor *is* in
information terms; and say why the interesting parameter in the rest of this part
is not bits-per-weight.

## 2. Why This Matters

{{ch:ft-qlora-peft}} stored a frozen base model at 4 bits and treated the
possibility as given. **This part is where that gets earned**, and it starts with
the observation that a numeric format has exactly one design decision.

**A format with $B$ bits splits them between exponent — which buys dynamic range
— and mantissa — which buys resolution. Nothing else is available.** FP16 and
BF16 are both 16 bits and differ only in where the line falls: 5/10 against 8/7.
{{sec:9-practical-example}} shows every consequence following from that single
difference — BF16 reaching **5.2 × 10³³ times** higher at the top, and paying with
**8× coarser** resolution.

**And the trade is settled by the distribution, not in the abstract.** On
transformer weights FP16's relative error is **1.79e-04** against BF16's
**1.41e-03** — 8× better, with neither losing a value. On gradients spanning eight
decades, FP16 **silently zeroes 9.33%** of them and BF16 zeroes none.

> **Those are not the same kind of failure.** A gradient rounded imprecisely is
> still a gradient. A gradient rounded to zero is **absent**, and the parameter it
> belonged to receives no update at all.

**Then the design question, swept directly.** At 16 bits, sweeping every possible
split gives **two answers that disagree completely**. Under signal-to-noise, more
mantissa wins at *every* dynamic range — **85.3 dB at e=3** against 55.7 at e=8.
Under the count of values a format cannot reach, e=3 loses **46.7%** of an
eight-decade distribution.

**Both are right**, and which one matters is decided by what happens downstream —
which is why two 16-bit formats exist and neither is better.

**Finally, the result that governs the rest of the part.** At 8 bits, adding a
single **per-tensor scale factor** improves the best achievable score from
**34.4 dB to 37.4 dB** — one number, stored once, worth more than any
redistribution of the per-value bits. **An exponent field is a per-value scale;
a scale factor is a per-tensor exponent.**

{{maturity:ESTABLISHED}} All formats discussed. {{maturity:MATURE}} FP8 in
production. {{maturity:EMERGING}} Precision as a term in scaling laws.

## 3. Prerequisites

{{ch:ft-qlora-peft}} for {{eq:training-memory}}, whose $b_w$ term this chapter
explains; {{ch:dl-optimizers}} for why gradients and weights have different
statistics; {{ch:math-random-vars}} for distributions and their dynamic range.

> **NOTE:** *Precision* is overloaded. {{part:10}} and {{part:11}} use it for the
> retrieval metric. **In this part it always means numeric precision**, and the
> retrieval sense is never intended.

## 4. Intuitive Explanation

### One decision, two consequences

A floating-point number is stored as sign, exponent, mantissa:

$$ x = (-1)^{s} \times 1.m \times 2^{e - \text{bias}} $$

The exponent says **where on the number line** you are; the mantissa says **how
precisely** you are located there. Given a fixed total width, every bit you give
one is taken from the other.

```text
   format      bits   exp   man        largest    smallest     decades
   ────────    ────   ───   ───   ────────────  ──────────   ─────────
   FP32          32     8    23      3.40e+38    1.40e-45        83.4
   FP16          16     5    10      6.55e+04    5.96e-08        12.0
   BF16          16     8     7      3.39e+38    9.18e-41        78.6
   FP8 E4M3       8     4     3      2.40e+02    1.95e-03         5.1
   FP8 E5M2       8     5     2      5.73e+04    1.53e-05         9.6
```

**FP16 and BF16 are the clean comparison** because the total is identical. BF16
covers 78.6 decades against FP16's 12.0; FP16 resolves 8× more finely.

### Two failures that are not the same failure

This is the idea to carry out of the chapter.

**Insufficient resolution** degrades every value a little. The downstream
computation usually absorbs it — a matmul sums many terms and independent rounding
errors partly cancel.

**Insufficient range** removes some values entirely. And **nothing downstream can
distinguish a value that rounded to zero from a value that was zero.**

{{sec:9-practical-example}} measures both on a gradient distribution: FP16 zeroes
**9.33%**, BF16 zeroes **0%**, and FP16's relative error on the survivors is
*better*. **The better number is the wrong number.**

### The design rule, and why one sweep gives two answers

Sweeping every 16-bit split produces a genuine disagreement:

```text
   decades          e=3     e=4     e=5     e=6     e=7     e=8
   ────────  ────  ────    ────    ────    ────    ────    ────
        0.5   SNR  85.3    79.7    73.7    67.6    61.6    55.6
             dead   0.0%    0.0%    0.0%    0.0%    0.0%    0.0%
        8.0   SNR  80.5    79.6    73.7    67.6    61.4    55.7
             dead  46.7%   35.6%    9.2%    0.0%    0.0%    0.0%
```

**Under SNR, e=3 wins everywhere.** Under dead-value count, e=3 loses nearly half
of a wide distribution.

**The disagreement has a cause, and it is the useful part.** SNR weights each
value by its **energy**, and in a distribution spanning eight decades essentially
all the energy is in the top decade. Zeroing a value at $10^{-7}$ costs SNR almost
nothing because there was almost nothing there. Counting dead values weights every
element **equally**.

**Composed, they give a rule:** spend the *smallest* exponent that reaches your
distribution, then put every remaining bit in the mantissa. The dead-value row
sets a floor; the SNR row says never exceed it.

Measured: half a decade needs **e=3**, four decades needs **e=5**, eight decades
needs **e=6**.

### Which is why both 16-bit formats exist

**For activations feeding a matmul**, the output is a weighted sum, so an error's
effect scales with its magnitude. SNR is the right score; more mantissa is the
right answer. **That is FP16.**

**For gradients**, every element is one parameter's update. A gradient rounded to
zero means that parameter does not move — and it does not matter that it was
small, because small consistent updates are how most parameters learn anything.
Dead values are the right score; more exponent is the right answer. **That is
BF16.**

> **Neither format is better.** They are optimal under different loss functions,
> and the loss function is chosen by the tensor's *role*, not by its statistics.

### The scale factor, and what it actually is

At 8 bits the problem is severe enough that the formats barely work unscaled.
{{sec:9-practical-example}} finds E4M3 scoring **1.02e-01** relative error on
weights — *worse* than E5M2's 4.53e-02, which looks like a contradiction of its
design.

**It is not.** E4M3's smallest normal is **1.56e-02** and these weights have
standard deviation 0.02, so most of the distribution falls into E4M3's subnormal
region. **The format does not lack resolution; it lacks the range to place the
distribution where its resolution lives.**

Divide by one constant so the distribution fills the format's range, quantize,
multiply back — and the designed behaviour appears: E4M3 goes to **2.25e-02**,
now beating E5M2's **4.48e-02** on weights, while E5M2 zeroes **0.21%** of
gradients against E4M3's **42.94%**.

**So {{cite:micikevicius2022fp8}}'s division of labour is not a convention. It is
that table, and it only appears once the scale factor is present.**

### And that reframes the rest of the part

**An exponent field is a per-value scale, carried by every element. A scale factor
is a per-tensor exponent, carried once.** If values within a tensor share most of
their magnitude information — and weights within a layer usually do — then an
exponent per element pays repeatedly for something nearly constant.

Push it to the conclusion and you get **integer quantization: no exponent bits at
all, every bit in the mantissa, one shared scale.**

> **So the interesting parameter is not "how many bits per weight". It is "how
> many weights share a scale factor."** Every scheme in {{ch:q-int8-int4}} is a
> position on that axis, and the bit-width everyone quotes is the less
> interesting half of the specification.

## 5. Formal Explanation

### 5.1 The budget

For $e$ exponent bits and $m$ mantissa bits with bias $2^{e-1}-1$:

$$ x_{\max} = (2 - 2^{-m})\,2^{\,2^{e}-2-\text{bias}}, \qquad x_{\min}^{\text{norm}} = 2^{\,1-\text{bias}}, \qquad x_{\min}^{\text{sub}} = 2^{\,1-\text{bias}-m} $$ (eq:format-is-a-budget)

with $1 + e + m = B$ fixed. **{{eq:format-is-a-budget}} is the whole design
space**: range spans roughly $2^{2^{e}}$ and relative resolution is $2^{-m}$, and
increasing either decreases the other.

Relative rounding error for a normal value is bounded by the machine epsilon:

$$ \frac{|x - Q(x)|}{|x|} \le 2^{-(m+1)} $$ (eq:relative-resolution)

**Independent of $e$** — resolution is purely a mantissa property, which is why
the two axes are genuinely separate.

### 5.2 Underflow is not rounding

For $|x| < x_{\min}^{\text{sub}}/2$, $Q(x) = 0$ and

$$ \frac{|x - Q(x)|}{|x|} = 1 $$ (eq:underflow-is-not-error)

**{{eq:underflow-is-not-error}} is why the two failures need separate accounting.**
Relative error saturates at 1 and stops distinguishing "slightly wrong" from
"gone", and the downstream consequences differ completely: a matmul absorbs a 1%
error in a summand and cannot recover a summand that is missing.

### 5.3 The two scores

$$ \text{SNR} = 10\log_{10}\frac{\sum_i x_i^2}{\sum_i (x_i - Q(x_i))^2}, \qquad \text{dead} = \frac{1}{N}\big|\{i : x_i \ne 0,\ Q(x_i) = 0\}\big| $$

SNR is **energy-weighted**; dead is **count-weighted**. For a distribution whose
magnitudes span $D$ decades roughly uniformly in $\log$,

$$ \frac{\text{energy below } 10^{-k}}{\text{total energy}} \approx 10^{-2k} $$ (eq:energy-concentrates)

**{{eq:energy-concentrates}} is why the scores disagree**: at $k=4$, values four
decades down carry $10^{-8}$ of the energy and $1/D$ of the *count*.

### 5.4 The design rule

$$ e^{*} = \min\{e : x_{\min}^{\text{sub}}(e, B-1-e) \le \min_i |x_i|\}, \qquad m^{*} = B - 1 - e^{*} $$ (eq:optimal-split)

**{{eq:optimal-split}} is the composition of the two scores**: the constraint
comes from the dead count and the objective from SNR. Measured optima — e=3 at
half a decade, e=5 at four, e=6 at eight — follow from it directly.

### 5.5 Why the metric is a modelling choice

Let $g$ be a downstream functional. The relevant error is $|g(x) - g(Q(x))|$, and

$$ g(x) = \textstyle\sum_i w_i x_i \;\Rightarrow\; \text{error} \propto \|x - Q(x)\|_2 \quad\text{(SNR)} $$

$$ g(x) = \text{"does parameter } i \text{ move?"} \;\Rightarrow\; \text{error} \propto \text{dead} $$ (eq:metric-picks-the-format)

**{{eq:metric-picks-the-format}} is the chapter's central claim**: the format
follows from the tensor's *role* in the computation, not from its histogram. Two
tensors with identical distributions can want different formats.

### 5.6 A scale factor is an exponent, amortised

Storing $s = \max_i|x_i| / x_{\max}$ once and quantizing $x/s$:

$$ \text{bits} = N \cdot (1 + e + m) + 32 \quad\text{versus}\quad N \cdot (1 + e' + m'),\ e' > e $$ (eq:scale-factor-as-exponent)

**{{eq:scale-factor-as-exponent}} shows the amortisation.** A shared scale costs
32 bits total; an extra exponent bit costs $N$ bits. For $N = 64$ the shared scale
is cheaper than *half* an exponent bit per element, and it does the same job
whenever the tensor's internal dynamic range is small.

The condition is precisely that:

$$ \frac{\max_i |x_i|}{\min_i |x_i|} \ \lesssim\ 2^{\,2^{e}} \quad \text{within the group sharing } s $$ (eq:scale-group-condition)

> **IMPORTANT:** {{eq:scale-group-condition}} is where {{ch:q-int8-int4}} lives.
> The group is the unit of the assumption, and shrinking it from a tensor to a
> channel to 64 weights weakens the assumption at a storage cost of
> $32/N$ bits per weight.

### 5.7 Loss scaling is a manual exponent

Multiplying by $2^k$ before rounding and dividing after:

$$ Q_{\text{scaled}}(x) = 2^{-k}\,Q(2^{k}x) $$ (eq:loss-scaling)

shifts the representable window by $k$ powers of two without changing the format.
**{{eq:loss-scaling}} is {{eq:scale-factor-as-exponent}} applied to gradients at
training time**, and the measurement shows it working: FP16 underflow falls from
**9.33%** at $k=0$ to **0.05%** at $k=8$ and **0.00%** at $k=16$.

**BF16 needs none of it at any $k$**, because it has the exponent bits to begin
with — which is the whole reason it replaced FP16 for training.

## 6. Mathematical Foundation

### 6.1 The FP16/BF16 ratio, worked

$$ \frac{x_{\max}^{\text{BF16}}}{x_{\max}^{\text{FP16}}} = \frac{(2-2^{-7})2^{127}}{(2-2^{-10})2^{15}} \approx 2^{112} \approx 5.2\times10^{33} $$

$$ \frac{\epsilon_{\text{BF16}}}{\epsilon_{\text{FP16}}} = \frac{2^{-8}}{2^{-11}} = 8 $$

**Three exponent bits bought $2^{112}$ of range and cost a factor of 8 in
resolution.** The asymmetry is not a coincidence: range is *exponential* in $e$
and resolution is exponential in $m$, but range is exponential in $2^e$ while
resolution is exponential in $m$ alone.

$$ \text{range} \sim 2^{2^{e}}, \qquad \text{resolution} \sim 2^{-m} $$ (eq:double-exponential)

**{{eq:double-exponential}} explains why exponent bits are so much more
"valuable" per bit**, and therefore why formats cluster at small $e$: a couple of
exponent bits go a very long way, and after that the mantissa is the better
purchase.

### 6.2 Why 8 bits needs a scale and 16 does not

From {{eq:format-is-a-budget}}, an 8-bit format has at most $2^{2^{e}}$ of range
with $7-e$ mantissa bits. At $e=4$: 5.1 decades of range and 3 mantissa bits. A
weight tensor with $\sigma = 0.02$ has magnitudes centred at $10^{-1.7}$ —
**below E4M3's smallest normal**, so the distribution sits in the subnormal
region where the step is fixed at $1.95\times10^{-3}$, i.e. a tenth of the
standard deviation.

$$ \frac{\text{step}}{\sigma} = \frac{1.95\times10^{-3}}{0.02} \approx 0.1 $$

**A tenth of a standard deviation is a catastrophic step size**, and it explains
the measured 1.02e-01 exactly. At 16 bits the same distribution sits comfortably
in the normal range of every split, which is why nobody scales FP16 weights.

> **MATH NOTE:** {{eq:relative-resolution}} bounds error only for *normal* values.
> In the subnormal region the step is absolute rather than relative, so relative
> error grows without bound as $|x| \to 0$. Most surprises attributed to "FP8
> being lossy" are really this: the distribution was in the subnormals, and a
> scale factor moves it out.

### 6.3 Precision as a scaling-law term

{{cite:kumar2024precisionscaling}} makes precision a variable in the scaling law
rather than an implementation choice, with two consequences worth stating early:

1. Training at reduced precision **reduces effective parameter count**, so a
   low-precision model of size $N$ behaves like a smaller full-precision one.
2. **Post-training quantization damage grows with pretraining tokens** — past a
   point, more pretraining makes a model *worse* after quantization.

**The second inverts the usual intuition** and is developed in {{ch:q-theory}}. Its
practical form: a quantization recipe is validated on a **checkpoint**, not on an
architecture.

## 7. Internal Mechanics

```mermaid {#fig:format-budget caption="A fixed bit budget and the two things it can buy. Range is double-exponential in the exponent width (eq:double-exponential), so a few exponent bits go a long way; resolution is exponential in the mantissa width. A per-tensor scale factor is a third option that buys range without spending per-value bits (eq:scale-factor-as-exponent), which is why every 8-bit scheme has one and no 16-bit scheme needs one."}
flowchart TB
    B["B bits per value"] --> E["exponent: e bits"]
    B --> M["mantissa: m bits"]
    E -->|"range ~ 2^(2^e)"| R["reaches small values"]
    M -->|"resolution ~ 2^-m"| P["locates them precisely"]
    R --> D{{"dead values?<br/>count-weighted"}}
    P --> S{{"rounding error?<br/>energy-weighted"}}
    D -->|"sets a FLOOR on e"| RULE["eq:optimal-split"]
    S -->|"says never exceed it"| RULE
    SC[("per-tensor scale:<br/>32 bits, shared")] -->|"buys range<br/>off-budget"| R
```

### 7.1 The formats, and what each is for

| Format | Split | Designed for | Because |
|---|---|---|---|
| FP32 | 8/23 | anything | 83 decades and 24-bit resolution; no trade needed |
| TF32 | 8/10 | matmul accumulation | FP32 range, FP16-ish resolution, hardware-native |
| FP16 | 5/10 | activations, weights | resolution matters, range is known |
| BF16 | 8/7 | gradients, general training | range unknown; underflow is unacceptable |
| FP8 E4M3 | 4/3 | forward pass | resolution-leaning half of the FP8 pair |
| FP8 E5M2 | 5/2 | backward pass | range-leaning half |
| INT8/INT4 | 0/$m$ | stored weights | $e=0$; the scale factor does all the range work |

**The last row is the point of the table.** Integer quantization is not a
different kind of thing from floating point — it is the $e = 0$ corner of the same
design space, with the range delegated entirely to a shared scale.

### 7.2 What to check when a format "does not work"

1. **Is the distribution in the subnormals?** Compare
   $\min|x|$ against $x_{\min}^{\text{norm}}$. This is the most common cause and
   a scale factor fixes it.
2. **Are values being zeroed?** Count them; do not infer from relative error,
   which saturates at 1 ({{eq:underflow-is-not-error}}).
3. **Are values being clipped at the top?** Rarer, and it shows as a hard cap
   rather than as noise.
4. **Is the failure energy-weighted or count-weighted?**
   {{eq:metric-picks-the-format}} — the answer determines which direction to move.

### 7.3 Accumulation precision is a separate decision

**Storage precision and accumulation precision are independent**, and conflating
them causes real bugs. A matmul may read FP8 operands and accumulate in FP32,
which is what tensor cores do: the product of two 8-bit values needs more than 8
bits to represent, and summing thousands of them needs more still.

$$ \text{Var}\big[\textstyle\sum_{i=1}^{K} \epsilon_i\big] = K\,\text{Var}[\epsilon] $$ (eq:accumulation-error)

**{{eq:accumulation-error}} says accumulation error grows as $\sqrt{K}$**, so a
long reduction in low precision is a different failure from lossy storage — and
it is why "we quantized to 8 bits" almost always means storage.

## 8. Implementation

```python {tier=A name=format-is-a-budget}
"""Every float format is one bit budget split two ways.

A format with B bits spends some on the EXPONENT, which buys dynamic range, and
the rest on the MANTISSA, which buys resolution. Nothing else is available. FP16
and BF16 are both 16 bits and differ only in where the line is drawn, and every
consequence people attribute to them follows from that line
(eq:format-is-a-budget).

This listing implements the rounding for each format directly, so the numbers
come from the arithmetic rather than from a table, and then applies them to two
distributions with very different shapes -- transformer weights, which are tightly
clustered, and gradients, which span many orders of magnitude.
"""
import numpy as np

rng = np.random.default_rng(229)


def fp_round(x, e_bits, m_bits):
    """Round to the nearest value representable with `e_bits` of exponent and
    `m_bits` of mantissa. Subnormals fall out of clamping the exponent at its
    minimum, which is exactly what the hardware does."""
    bias = 2 ** (e_bits - 1) - 1
    emin, emax = 1 - bias, (2 ** e_bits - 2) - bias
    maxval = (2.0 - 2.0 ** (-m_bits)) * 2.0 ** emax
    ax = np.abs(x)
    safe = np.where(ax > 0, ax, 1.0)
    e = np.clip(np.floor(np.log2(safe)), emin, emax)
    step = 2.0 ** (e - m_bits)
    q = np.round(x / step) * step
    q = np.clip(q, -maxval, maxval)
    return np.where(ax > 0, q, 0.0)


def int_round(x, bits):
    """Symmetric per-tensor integer quantization, for contrast: a FIXED step
    everywhere rather than a step that scales with magnitude."""
    qmax = 2 ** (bits - 1) - 1
    s = np.max(np.abs(x)) / qmax
    return np.clip(np.round(x / s), -qmax, qmax) * s


FORMATS = [
    ("FP32", 8, 23), ("FP16", 5, 10), ("BF16", 8, 7),
    ("FP8 E4M3", 4, 3), ("FP8 E5M2", 5, 2),
]


def describe(e_bits, m_bits):
    bias = 2 ** (e_bits - 1) - 1
    emin, emax = 1 - bias, (2 ** e_bits - 2) - bias
    return ((2.0 - 2.0 ** (-m_bits)) * 2.0 ** emax,      # largest normal
            2.0 ** emin,                                  # smallest normal
            2.0 ** (emin - m_bits))                       # smallest subnormal


print("What each format can represent, from the exponent/mantissa split alone.\n")
print(f"{'format':>10}{'bits':>6}{'exp':>5}{'man':>5}{'largest':>12}"
      f"{'smallest':>12}{'smallest':>12}{'decades of':>12}")
print(f"{'':>10}{'':>6}{'':>5}{'':>5}{'normal':>12}{'normal':>12}"
      f"{'subnormal':>12}{'range':>12}")
print("-" * 74)
for name, e, m in FORMATS:
    hi, lo, sub = describe(e, m)
    print(f"{name:>10}{1+e+m:>6}{e:>5}{m:>5}{hi:>12.2e}{lo:>12.2e}"
          f"{sub:>12.2e}{np.log10(hi/sub):>12.1f}")

W = rng.normal(0, 0.02, size=200000)                      # transformer weights
G = rng.normal(0, 1.0, size=200000) * 10.0 ** rng.uniform(-8, 0, size=200000)

print("\n\nApplied to two real distribution shapes.\n")
print(f"{'':>16}{'TRANSFORMER WEIGHTS  N(0, 0.02)':>30}"
      f"{'GRADIENTS  spanning 1e-8 to 1':>30}")
print(f"{'format':>16}{'rel err':>12}{'zeroed':>10}{'over':>8}"
      f"{'rel err':>12}{'zeroed':>10}{'over':>8}")
print("-" * 76)


def report(q, x, maxval=np.inf):
    nz = x != 0
    rel = float(np.mean(np.abs(q[nz] - x[nz]) / np.abs(x[nz])))
    zeroed = float(np.mean((q == 0) & nz))          # fell below the format
    over = float(np.mean(np.abs(x) > maxval))        # rose above it
    return rel, zeroed, over


def scaled(x, e_bits, m_bits):
    """What real FP8 does: divide by a per-tensor scale so the distribution
    uses the format's range, quantize, multiply back. The scale is stored
    alongside in higher precision and is part of the format in practice."""
    hi = describe(e_bits, m_bits)[0]
    s = np.max(np.abs(x)) / hi
    return fp_round(x / s, e_bits, m_bits) * s


rows = {}


def line(name, a, b):
    rows[name] = (a, b)
    print(f"{name:>16}{a[0]:>12.2e}{a[1]:>10.2%}{a[2]:>8.2%}"
          f"{b[0]:>12.2e}{b[1]:>10.2%}{b[2]:>8.2%}")


for name, e, m in FORMATS:
    hi = describe(e, m)[0]
    line(name, report(fp_round(W, e, m), W, hi),
         report(fp_round(G, e, m), G, hi))

print(f"{'':>16}{'--- with a per-tensor scale factor ---':>70}")
for name, e, m in FORMATS[3:]:
    line(name + " +scale", report(scaled(W, e, m), W), report(scaled(G, e, m), G))
line("INT8 +scale", report(int_round(W, 8), W), report(int_round(G, 8), G))

print("\n\nLoss scaling: what multiplying by a constant before rounding buys.\n")
print(f"{'scale':>10}{'FP16 zeroed':>14}{'BF16 zeroed':>14}"
      f"{'FP16 rel err':>15}")
print("-" * 53)
for k in (0, 8, 16, 24):
    s = 2.0 ** k
    f16 = report(fp_round(G * s, 5, 10) / s, G)
    b16 = report(fp_round(G * s, 8, 7) / s, G)
    print(f"{'2^' + str(k):>10}{f16[1]:>14.2%}{b16[1]:>14.2%}{f16[0]:>15.2e}")

w16, wbf = rows["FP16"][0], rows["BF16"][0]
g16, gbf = rows["FP16"][1], rows["BF16"][1]
e4, e5 = rows["FP8 E4M3"], rows["FP8 E5M2"]
e4s, e5s = rows["FP8 E4M3 +scale"], rows["FP8 E5M2 +scale"]
i8s = rows["INT8 +scale"]
print(f"""
The first table is the whole design space in one sentence: at equal total width,
every bit given to the exponent is a bit taken from the mantissa. FP16 and BF16
are both 16 bits. FP16 spends 5 on the exponent and 10 on the mantissa; BF16
spends 8 and 7. Everything else in that table follows -- BF16 reaches
{describe(8,7)[0]/describe(5,10)[0]:.0f}x higher at the top and far lower at the
bottom, and pays {2**(10-7):.0f}x coarser resolution for it
(eq:format-is-a-budget).

The second table shows that the trade is not settled in the abstract. It is
settled by the DISTRIBUTION you intend to store.

On transformer weights -- tightly clustered, spanning a few decades -- FP16's
extra mantissa bits win: {w16[0]:.2e} against BF16's {wbf[0]:.2e}, about
{wbf[0]/w16[0]:.0f}x better, with neither format losing a value. The range BF16
bought is range this distribution never uses.

On gradients spanning eight decades the ranking reverses, and it reverses in the
column that matters rather than the one you were watching. FP16 silently zeroes
{g16[1]:.1%} of them; BF16 zeroes {gbf[1]:.1%}. FP16's relative error on the
survivors is still the better number and it is the wrong number, because a
gradient rounded to zero is not imprecise -- it is ABSENT, and the parameter it
belonged to receives no update at all (eq:underflow-is-not-error).

That distinction is the most useful idea in the chapter and it generalises past
floats. Resolution failures degrade every value slightly and the process usually
absorbs them. Range failures remove some values completely, and nothing
downstream can tell a value that rounded to zero from a value that was zero.

Now the block with the scale factors, which changes the reading of the two FP8
rows above it entirely.

Unscaled, E4M3 is WORSE than E5M2 on weights: {e4[0][0]:.2e} against
{e5[0][0]:.2e}. That looks like a contradiction of its design and it is not.
E4M3's smallest normal value is {describe(4,3)[1]:.2e}, and these weights have a
standard deviation of 0.02, so most of the distribution falls into E4M3's
subnormal region where the step size is fixed and coarse. The format does not lack
resolution; it lacks the RANGE to place this distribution where its resolution
lives.

Give each format a single per-tensor scale factor -- divide by a constant so the
distribution fills the format's range, quantize, multiply back -- and the designed
behaviour appears. E4M3 goes from {e4[0][0]:.2e} to {e4s[0][0]:.2e} on weights,
now beating E5M2's {e5s[0][0]:.2e}. On gradients E5M2 zeroes {e5s[1][1]:.2%}
against E4M3's {e4s[1][1]:.1%}.

So the division of labour cite:micikevicius2022fp8 specifies -- E4M3 for weights
and activations, E5M2 for gradients -- is not a convention. It is this table, and
it only appears once the scale factor is present.

Which is the practical lesson: at 8 bits, the scale factor is not an
implementation detail, it is part of the format. Eight bits is not enough to
carry both the location of a distribution and its shape, so the location is
factored out and stored separately at higher precision. Every scheme in the
chapters that follow is an argument about how FINELY to factor it out -- per
tensor, per channel, per block of 64 weights -- and that granularity choice
matters more than the bit-width people quote.

The INT8 row makes the same point from the other side. Integer quantization has no
exponent field at all, so its steps are uniform rather than log-spaced. With a
scale factor it reaches {i8s[0][0]:.2e} on weights against E4M3's
{e4s[0][0]:.2e}: an 8-bit FLOAT beats an 8-bit INTEGER on a bell-shaped
distribution, because log-spaced steps put resolution where a bell curve puts its
mass. On the eight-decade gradients it zeroes {i8s[1][1]:.1%}, because one uniform
step cannot serve eight decades at any scale.

The last table is the historical footnote that turns out to be the clearest
demonstration in the listing. Loss scaling multiplies gradients by a large
constant before storing them in FP16 and divides afterwards, and underflow falls
from {g16[1]:.1%} toward nothing as the constant grows.

That is a manual exponent adjustment, applied because the format's own exponent
field is too small. BF16 needs none of it at any scale.
cite:micikevicius2018mixed introduced loss scaling as a necessary part of FP16
training; BF16 removed the need by moving the bits; cite:micikevicius2022fp8 then
encoded the same lesson into two formats rather than into the training loop. One
idea, learned three times, each time pushed further down the stack.""")
```

The first listing takes the formats as given. The second asks where the line
should go.

```python {tier=A name=optimal-split}
"""Where should the line between exponent and mantissa go?

The previous listing took the existing formats as given. This one asks the design
question directly: at a fixed total width, sweep every possible split and measure
which one wins.

The interesting part is that the sweep gives two different answers depending on
how you score it, and the disagreement is not a flaw in the experiment -- it is
the reason more than one format exists (eq:metric-picks-the-format). Then it asks
a second question that matters more in practice: what happens to the whole
picture once a per-tensor scale factor is allowed?
"""
import numpy as np

rng = np.random.default_rng(233)
N = 120000


def fp_round(x, e_bits, m_bits):
    bias = 2 ** (e_bits - 1) - 1
    emin, emax = 1 - bias, (2 ** e_bits - 2) - bias
    maxval = (2.0 - 2.0 ** (-m_bits)) * 2.0 ** emax
    ax = np.abs(x)
    safe = np.where(ax > 0, ax, 1.0)
    e = np.clip(np.floor(np.log2(safe)), emin, emax)
    step = 2.0 ** (e - m_bits)
    q = np.clip(np.round(x / step) * step, -maxval, maxval)
    return np.where(ax > 0, q, 0.0)


def int_round(x, bits):
    qmax = 2 ** (bits - 1) - 1
    s = np.max(np.abs(x)) / qmax
    return np.clip(np.round(x / s), -qmax, qmax) * s


def snr_db(x, q):
    """Weights every value by its ENERGY, so large values dominate and small
    ones are nearly free to lose."""
    err = np.sum((x - q) ** 2)
    return float(10 * np.log10(np.sum(x ** 2) / max(err, 1e-300)))


def dead(x, q):
    """Fraction of nonzero values that rounded to zero -- values the format
    could not reach at all, weighted equally regardless of magnitude."""
    nz = x != 0
    return float(np.mean((q[nz] == 0)))


def dist(decades):
    """Values spanning `decades` orders of magnitude below the largest."""
    mag = 10.0 ** rng.uniform(-decades, 0, size=N)
    return rng.normal(0, 1, size=N) * mag


TOTAL = 16
E_RANGE = (3, 4, 5, 6, 7, 8, 9, 10)
DECADES = (0.5, 1, 2, 4, 8)

print("Fixed budget of 16 bits: 1 sign, e exponent, 15 - e mantissa.")
print("Two scores, because they disagree. SNR in dB (higher better) weights by")
print("energy. 'dead' is the share of values the format cannot reach at all,")
print("and its 'best' column is the SMALLEST exponent that loses nothing.")
print()
print(f"{'decades':>9}{'':>6}" + "".join(f"{'e=' + str(e):>9}" for e in E_RANGE)
      + f"{'best':>7}")
print("-" * 94)

best_snr, best_dead, tab = {}, {}, {}
for d in DECADES:
    x = dist(d)
    qs = [fp_round(x, e, TOTAL - 1 - e) for e in E_RANGE]
    sn = [snr_db(x, q) for q in qs]
    dd = [dead(x, q) for q in qs]
    best_snr[d] = E_RANGE[int(np.argmax(sn))]
    ok = [e for e, v in zip(E_RANGE, dd) if v < 5e-4]
    best_dead[d] = ok[0] if ok else E_RANGE[int(np.argmin(dd))]
    tab[d] = (sn, dd)
    print(f"{d:>9.1f}{'SNR':>6}" + "".join(f"{v:>9.1f}" for v in sn)
          + f"{'e=' + str(best_snr[d]):>7}")
    print(f"{'':>9}{'dead':>6}" + "".join(f"{v:>9.1%}" for v in dd)
          + f"{'e=' + str(best_dead[d]):>7}")

print()
print()
print("Same question at 8 bits, with and without a per-tensor scale factor.")
print()
E8 = (2, 3, 4, 5, 6)
print(f"{'decades':>9}{'':>6}" + "".join(f"{'e=' + str(e):>9}" for e in E8)
      + f"{'INT8':>9}{'best':>7}")
print("-" * 70)

raw8, sc8 = {}, {}
for d in DECADES:
    x = dist(d)
    raw = [snr_db(x, fp_round(x, e, 7 - e)) for e in E8]
    sc = []
    for e in E8:
        hi = (2.0 - 2.0 ** -(7 - e)) * 2.0 ** ((2 ** e - 2) - (2 ** (e - 1) - 1))
        ss = np.max(np.abs(x)) / hi
        sc.append(snr_db(x, fp_round(x / ss, e, 7 - e) * ss))
    i8 = snr_db(x, int_round(x, 8))
    raw8[d], sc8[d] = (raw, i8), sc
    print(f"{d:>9.1f}{'raw':>6}" + "".join(f"{v:>9.1f}" for v in raw)
          + f"{i8:>9.1f}" + f"{'e=' + str(E8[int(np.argmax(raw))]):>7}")
    print(f"{'':>9}{'+sc':>6}" + "".join(f"{v:>9.1f}" for v in sc)
          + f"{i8:>9.1f}" + f"{'e=' + str(E8[int(np.argmax(sc))]):>7}")

s_lo, d_lo = tab[0.5]
s_hi, d_hi = tab[8]
print(f"""
Read the 16-bit table one row-pair at a time and the two scores disagree
completely.

Under SNR, e=3 wins at EVERY dynamic range -- {s_lo[0]:.1f} dB at half a decade
and {s_hi[0]:.1f} dB at eight decades, against e=8's {s_hi[5]:.1f} dB. More
mantissa always wins, no matter how much range the distribution spans.

Under the dead-value count, the picture is a hard constraint rather than a
ranking. At half a decade every split reaches everything, so e={best_dead[0.5]}
suffices. At four decades you need e={best_dead[4]}, and at eight you need
e={best_dead[8]}. Below that threshold the losses are not marginal: e=3 cannot
reach {d_hi[0]:.1%} of the eight-decade values and e=4 cannot reach
{d_hi[1]:.1%}.

Both numbers are correct, and the reason they disagree is worth more than either
of them. SNR is an ENERGY-weighted score, and in a distribution spanning eight
decades essentially all the energy sits in the top decade. Zeroing a value at
1e-7 costs SNR almost nothing, because there was almost nothing there to lose.
Counting dead values weights every element EQUALLY, so the same rounding is a
total loss.

Put the two together and they are not really in conflict -- they compose into a
design rule. Spend the SMALLEST exponent that reaches your distribution, then put
every remaining bit in the mantissa (eq:optimal-split). The dead-value row sets
the floor; the SNR row says never exceed it.

What that rule needs is knowledge of the distribution's dynamic range, and whether
you have it is the actual difference between the formats. FP16's e=5 is the right
answer if you know your values span a few decades. BF16's e=8 is the right answer
if you do not know, or if the range varies between tensors and you want one format
for all of them.

And which score matters downstream decides how much risk the floor carries
(eq:metric-picks-the-format).

For an activation tensor feeding a matmul, the output is a weighted sum, so an
error's effect is roughly proportional to its magnitude -- SNR is the right score,
and more mantissa is the right answer. That is FP16.

For a gradient tensor, every element is the update for one parameter. A gradient
rounded to zero means that parameter does not move, and it does not matter that
the gradient was small -- over many steps, small consistent updates are how most
parameters learn anything. Dead values are the right score, and more exponent is
the right answer. That is BF16.

So the two 16-bit formats are not a historical accident and neither is better.
They are optimal under two different loss functions, and the loss function is
chosen by the tensor's role rather than by its statistics.

The 8-bit table then makes the practical point that governs the rest of
{{part:15}}.

Compare each raw row against the +sc row below it. At eight decades the best raw
split reaches {max(raw8[8][0]):.1f} dB; with a single per-tensor scale factor the
best reaches {max(sc8[8]):.1f} dB. One number, stored once for the whole tensor,
is worth more than any redistribution of the per-value bits.

The reason is that an exponent field is a PER-VALUE scale, carried by every
element. A scale factor is a PER-TENSOR exponent, carried once. If the values in a
tensor share most of their magnitude information -- and weights within one layer
usually do -- then an exponent field per element pays repeatedly for something
nearly constant.

Push that to its conclusion and you get integer quantization: no exponent bits at
all, every bit in the mantissa, one shared scale. The INT8 column is competitive
at low dynamic range ({raw8[0.5][1]:.1f} dB against the best float's
{max(raw8[0.5][0]):.1f}) and falls behind as the range widens, which is exactly
what that reading predicts.

And it reframes the question the next chapters answer. The interesting parameter
is not "how many bits per weight" but "how many weights share a scale factor". A
tensor-wide scale is free and assumes the whole tensor has one dynamic range. A
per-channel or per-64-weight scale costs a little storage and assumes much less.
Every INT4 and INT8 scheme in {{ch:q-int8-int4}} is a position on that axis, and
the bit-width everyone quotes is the less interesting half of the
specification.""")
```

## 9. Practical Example

**The design space, from the split alone.** FP32 covers 83.4 decades, FP16 12.0,
BF16 78.6, FP8 E4M3 5.1, FP8 E5M2 9.6. **BF16 reaches $5.2\times10^{33}$ times
higher than FP16 at the top and pays 8× in resolution** — three exponent bits, per
{{eq:double-exponential}}.

**And the trade is settled by the distribution.** On transformer weights: FP16
**1.79e-04** relative error against BF16's **1.41e-03**, neither losing a value.
On eight-decade gradients: FP16 **zeroes 9.33%**, BF16 zeroes none — and FP16's
error on the survivors is still better.

> **IMPORTANT:** The better number is the wrong number.
> {{eq:underflow-is-not-error}} — a gradient rounded to zero is not imprecise, it
> is **absent**, and the parameter it belonged to does not move.

**The design sweep gives two answers.** Under SNR, e=3 wins at every dynamic range
(**85.3 dB** at half a decade, **80.5** at eight, against e=8's 55.7). Under dead
values, e=3 loses **46.7%** of an eight-decade distribution, e=4 loses **35.6%**,
e=5 loses **9.2%**.

**{{eq:energy-concentrates}} is why they disagree** — essentially all the energy
in a wide distribution is in the top decade. **Composed via
{{eq:optimal-split}}**: e=3 suffices at half a decade, **e=5** is needed at four,
**e=6** at eight.

**Which is why both 16-bit formats exist.** {{eq:metric-picks-the-format}}:
activations feed weighted sums, so SNR is right and FP16's mantissa wins;
gradients are per-parameter updates, so dead values are right and BF16's exponent
wins. **Neither format is better — they are optimal under different loss
functions, chosen by the tensor's role.**

**At 8 bits the scale factor becomes part of the format.** Unscaled, E4M3 scores
**1.02e-01** on weights — *worse* than E5M2's 4.53e-02, because its smallest
normal is 1.56e-02 and the weights have σ = 0.02, putting the distribution in the
subnormals at a step of a tenth of a standard deviation.

**Add one per-tensor scale and the design appears**: E4M3 → **2.25e-02**, beating
E5M2's **4.48e-02** on weights, while E5M2 zeroes **0.21%** of gradients against
E4M3's **42.94%**. {{cite:micikevicius2022fp8}}'s division of labour is that
table.

**And the scale factor is worth more than the bits.** At eight decades the best
8-bit split reaches **34.4 dB** raw and **37.4 dB** scaled —
{{eq:scale-factor-as-exponent}}: 32 bits shared beats redistributing $N$ bits.
**An 8-bit float also beats an 8-bit integer on a bell curve** (2.25e-02 against
**4.38e-02**), because log-spaced steps put resolution where a bell curve puts its
mass.

**Finally, loss scaling as a manual exponent.** FP16 gradient underflow falls
**9.33% → 0.05% → 0.00%** at scales $2^0, 2^8, 2^{16}$; BF16 needs none of it.
{{cite:micikevicius2018mixed}} introduced it as necessary for FP16 training; BF16
removed the need by moving bits; {{cite:micikevicius2022fp8}} then encoded the
lesson into two formats. **One idea learned three times, each time pushed further
down the stack.**

## 10. Production Considerations

**Use BF16 for training** unless you have a specific reason not to. It removes
loss scaling and the whole class of underflow bugs.

**Check the subnormal boundary before blaming a format.** Compare $\min|x|$
against $x_{\min}^{\text{norm}}$ — this is the most common cause of "FP8 is
lossy".

**Count dead values, do not infer them from relative error**, which saturates.

**Treat the scale factor as part of an 8-bit format**, never as an optimisation.

**Keep accumulation in FP32** unless you have measured otherwise
({{eq:accumulation-error}}).

**Record the format of every stored tensor**, including the KV cache — mixed
assumptions across a pipeline are a recurring source of silent degradation.

**Do not assume a quantization result transfers across checkpoints**
({{cite:kumar2024precisionscaling}}); this is developed in {{ch:q-theory}}.

## 11. Common Mistakes

**Comparing formats by bit-width** rather than by split.

**Using relative error as the only metric**, so underflow is invisible.

**Assuming FP8 works like FP16 with fewer bits** — without a scale factor it does
not work at all on typical weights.

**Confusing storage precision with accumulation precision.**

**Applying loss scaling under BF16**, where it does nothing.

**Choosing a format from a tensor's histogram** rather than from its role
({{eq:metric-picks-the-format}}).

**Quoting "4-bit" without the group size**, which is the more consequential half.

## 12. Failure Modes

**Training loss stops improving under FP16.** Cause: gradient underflow
({{eq:underflow-is-not-error}}). Fix: loss scaling, or BF16.

**FP8 weights are far worse than expected.** Cause: the distribution is in the
subnormal region. Fix: a per-tensor scale.

**Quality fine on average, broken on rare inputs.** Cause: an energy-weighted
metric hid a count-weighted failure.

**Long reductions lose accuracy despite adequate storage precision.** Cause:
{{eq:accumulation-error}}.

**A format that worked on one checkpoint fails on its successor.** Cause:
{{cite:kumar2024precisionscaling}} — robustness is a checkpoint property.

**Clipping at the top of the range, showing as saturated outputs.** Cause: the
scale factor was computed on unrepresentative data.

## 13. Alternatives

| Alternative | Trades | When |
|---|---|---|
| FP32 everywhere | 2× memory, 2× bandwidth | numerical debugging, small models |
| BF16 | resolution | training; the default |
| FP16 + loss scaling | operational complexity | hardware without BF16 |
| TF32 | resolution | matmul on hardware that offers it |
| FP8 with per-tensor scale | resolution, calibration | inference on supporting hardware |
| INT8/INT4 with group scales | range flexibility | storage; {{ch:q-int8-int4}} |
| stochastic rounding | determinism | very low precision accumulation |

**The last row deserves a mention it rarely gets.** Round-to-nearest is biased
when the same value is rounded repeatedly — as in accumulation — and stochastic
rounding removes the bias at the cost of reproducibility. It matters most exactly
where {{eq:accumulation-error}} bites.

## 14. Evaluation

**Report the format's split, not just its width.**

**Report both an energy-weighted and a count-weighted error**, always.

**Report the scale granularity** for any sub-16-bit format.

**Report accumulation precision separately from storage precision.**

**Report the checkpoint**, not just the architecture, for any quantization result.

## 15. Advanced Concepts

**Integer quantization is the $e=0$ corner.** {{maturity:ESTABLISHED}}
Treating INT and FP as one design space rather than two families makes
{{ch:q-int8-int4}}'s choices legible: group size is where the range went.

**Microscaling formats.** {{maturity:EMERGING}} Formats that build a small shared
exponent into the block structure — a scale per 32 elements, encoded in the format
rather than alongside it — are {{eq:scale-factor-as-exponent}} taken to its logical
end, and are appearing in hardware.

**Stochastic rounding.** {{maturity:MATURE}} Unbiased under repeated rounding,
which matters for low-precision accumulation and optimiser states. Under-used
because it breaks reproducibility.

**Precision in the scaling law.** {{maturity:EMERGING}}
{{cite:kumar2024precisionscaling}} makes low-precision training a reduction in
effective parameters, which turns "what precision" into a term in the compute
budget rather than a deployment decision.

**Outliers as a range problem.** {{maturity:MATURE}}
{{cite:dettmers2022int8}}'s emergent features are, in this chapter's terms, a
violation of {{eq:scale-group-condition}}: a few coordinates blow up the group's
dynamic range and force the shared scale to serve a range it cannot.
**{{ch:q-int8-int4}} is largely about that one inequality.**

## 16. Connection to Previous Chapters

{{ch:ft-qlora-peft}}'s {{eq:training-memory}} has $b_w$ as a free parameter; this
chapter says what values it can take and what each costs.
{{ch:dl-optimizers}} explains why gradients and weights have different dynamic
ranges, which {{eq:metric-picks-the-format}} turns into different formats.
{{ch:math-random-vars}} supplies the distributions.
Forward: {{ch:q-theory}} asks why the resulting errors are tolerable at all;
{{ch:q-int8-int4}} is {{eq:scale-group-condition}} in practice;
{{ch:q-memory-math}} uses {{eq:format-is-a-budget}} to answer "will it fit";
{{ch:q-activation-kv}} applies all of it to tensors whose statistics are set by
the data rather than by training.

## 17. Exercises

1. Derive $x_{\max}$, $x_{\min}^{\text{norm}}$ and $x_{\min}^{\text{sub}}$ for
   E4M3 and E5M2 from {{eq:format-is-a-budget}} and check against the listing.
2. A tensor has $\sigma = 0.02$ and $\max|x| = 0.15$. Which of FP16, BF16, E4M3,
   E5M2 need a scale factor, and why?
3. Using {{eq:double-exponential}}, compute how much range one more exponent bit
   buys at $e=4$ and at $e=8$. Why are the answers so different?
4. Show from {{eq:scale-factor-as-exponent}} the group size at which a shared
   32-bit scale costs the same as one extra bit per element.
5. In `format-is-a-budget`, add a distribution with $\sigma = 10^{-4}$. Which
   formats need a scale now?
6. In `optimal-split`, change the dead-value threshold to count values losing more
   than 50% of their magnitude rather than exactly zero. Does the design rule
   change?
7. Explain why an 8-bit float beat an 8-bit integer on the weight distribution and
   would lose on a uniform one.
8. Derive {{eq:accumulation-error}} and estimate the accumulation width needed for
   a $K = 4096$ reduction of FP8 products.

## 18. Interview Questions

1. FP16 and BF16 are both 16 bits. What is the difference and what follows from
   it?
2. Why is a value rounded to zero a different problem from a value rounded
   imprecisely?
3. Your FP16 training run stalls. What do you check?
4. Why does BF16 not need loss scaling?
5. Why do E4M3 and E5M2 both exist?
6. Your FP8 weights are far worse than FP8 activations. What is the likely cause?
7. What is a scale factor, in information terms?
8. Why is integer quantization the same design space as floating point?
9. What is the difference between storage precision and accumulation precision?
10. Why is "4-bit" an incomplete specification?

## 19. Research Questions

1. {{eq:optimal-split}} assumes you know the distribution's range. How well can it
   be predicted per tensor before training, and what would a per-tensor format
   assignment buy?
2. Microscaling formats embed the scale in the block. What is the optimal block
   size as a function of a tensor's internal dynamic range, and does it match the
   sizes hardware has chosen?
3. {{eq:metric-picks-the-format}} argues the format follows from the tensor's
   role. Can that role be inferred automatically from the computation graph?
4. Stochastic rounding is unbiased but non-reproducible. Is there a
   deterministic-per-seed variant that keeps the bias properties and the
   debuggability?
5. {{cite:kumar2024precisionscaling}} makes precision a scaling-law term. What
   does the compute-optimal precision look like when inference cost is included in
   the objective rather than only training cost?

## 20. Chapter Summary

**A numeric format has one design decision**: how to split a fixed bit budget
between exponent, which buys range, and mantissa, which buys resolution
({{eq:format-is-a-budget}}). FP16 and BF16 are both 16 bits; BF16's three extra
exponent bits buy **$5.2\times10^{33}$** of range and cost a factor of **8** in
resolution, because {{eq:double-exponential}} makes range double-exponential in
$e$.

**Range failures and resolution failures are not the same failure.** FP16 gave
**8× better** relative error on weights and **silently zeroed 9.33%** of
gradients. {{eq:underflow-is-not-error}}: a zeroed gradient is not imprecise, it
is absent, and relative error saturates at 1 rather than reporting it.

**The design sweep gives two answers that disagree.** SNR picks **e=3
everywhere**; the dead-value count needs **e=5** at four decades and **e=6** at
eight. {{eq:energy-concentrates}} explains it — nearly all energy sits in the top
decade — and {{eq:optimal-split}} composes them: **the smallest exponent that
reaches your distribution, then everything else in the mantissa.**

**Which is why both 16-bit formats exist and neither is better.**
{{eq:metric-picks-the-format}}: activations feed weighted sums so SNR is the right
score and FP16 wins; gradients are per-parameter updates so dead values are the
right score and BF16 wins. **The format follows from the tensor's role, not from
its histogram.**

**At 8 bits the scale factor is part of the format.** Unscaled E4M3 scored
**1.02e-01** on weights — worse than E5M2 — because the distribution sat in its
subnormals at a step of a tenth of a standard deviation. **Scaled, E4M3 reaches
2.25e-02 and beats E5M2**, while E5M2 zeroes **0.21%** of gradients against
**42.94%**. The designed division of labour only appears once the scale is there.

**And a shared scale is worth more than the bits it saves.** **34.4 → 37.4 dB**
from one 32-bit constant, because {{eq:scale-factor-as-exponent}} amortises what
an exponent field pays per element. **An exponent field is a per-value scale; a
scale factor is a per-tensor exponent.**

Which sets up the whole part: **integer quantization is the $e = 0$ corner of this
same design space**, with range delegated entirely to shared scales — so the
interesting parameter is not bits-per-weight but **how many weights share a
scale**, and {{eq:scale-group-condition}} is the inequality {{ch:q-int8-int4}}
spends its length on.

## 21. Further Reading

{{cite:micikevicius2018mixed}} for the mixed-precision pattern that every training
stack still uses, and for loss scaling as the clearest example of a manual
exponent adjustment.
{{cite:micikevicius2022fp8}} for E4M3 and E5M2, read alongside
{{sec:9-practical-example}}'s scaled rows: the division of labour is a measurement,
not a convention.
{{cite:dettmers2022int8}} for outliers, which this chapter's vocabulary describes
as a violation of {{eq:scale-group-condition}} — the framing
{{ch:q-int8-int4}} builds on.
{{cite:kumar2024precisionscaling}} for precision as a scaling-law term, and for
the result that reframes quantization robustness as a checkpoint property.
