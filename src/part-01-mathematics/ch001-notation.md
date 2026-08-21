---
id: math-notation
number: 1
part: I
tier: focused
status: reviewed
requires: []
provides: [summation-notation, indicator-function]
citations: [deisenroth2020]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Read an equation and identify the *type* of every symbol — scalar, vector,
   matrix, set, function — from its typography alone.
2. Expand and evaluate summation and product notation, including nested and
   conditional sums.
3. Use set notation, interval notation, and the membership and subset relations
   correctly.
4. Read function signatures of the form $f: \R^n \to \R^m$ and state the shape
   of the input and output.
5. Use indicator functions to convert a counting statement into a sum.
6. Translate freely between mathematical notation and NumPy code.
7. Apply a systematic procedure to an unfamiliar equation instead of stalling.

## 2. Why This Matters

Most people who believe they cannot do mathematics are, in fact, stuck on
notation. The underlying ideas in this book are not beyond anyone with ordinary
patience. The symbols, however, are a genuine second language, and nobody
learns a language by being handed a novel.

This is not a trivial obstacle. Consider a line you will meet in Part IX:

$$
\Loss(\theta) = -\frac{1}{N}\sum_{i=1}^{N} \log p_{\theta}(y_i \given \vec{x}_i)
$$ (eq:nll)

A reader fluent in notation sees this immediately: *the average, over N
examples, of the negative log-probability the model assigns to the correct
answer.* A reader who is not fluent sees eight unfamiliar symbols and concludes
the subject is inaccessible. The difference is not intelligence and not
mathematical ability. It is vocabulary, and vocabulary is learnable in an
afternoon.

The second reason this chapter exists is precision. Mathematical notation is
compact not to be obscure but because natural language is too ambiguous for the
job. "The average error over the dataset" leaves at least three questions open:
average over what index, error measured how, and is the dataset the training set
or all of it? {{eq:nll}} answers all three in one line and cannot be
misunderstood.

## 3. Prerequisites

Arithmetic, and the idea of a variable standing for a number. Nothing else.

This is the first chapter of the book, and it assumes no mathematics beyond
secondary school. If you have forgotten what you once knew, that is fine — this
chapter and the next rebuild the parts that matter.

## 4. Intuitive Explanation

### 4.1 Typography is type information

The most useful thing to know about mathematical notation, and the thing least
often said explicitly, is that **the way a symbol is printed tells you what kind
of object it is.** This is not decoration. It is a type system, and once you
know it you can read the structure of an equation before understanding any of
its content.

{#tbl:typography caption="The typographic convention used throughout this book. It is close to universal in machine learning, though not every paper follows it; when one does not, this book says so."}

| Looks like | Is a | Example |
|---|---|---|
| lowercase italic | scalar — a single number | $a$, $n$, $\eta$, $\alpha$ |
| lowercase **bold** | vector — an ordered list of numbers | $\vec{x}$, $\vec{q}$ |
| uppercase **bold** | matrix — a rectangular grid | $\mat{A}$, $\mat{W}$ |
| uppercase italic | a set, or a count | $S$, $N$, $V$ |
| blackboard bold | a standard set | $\R$, $\N$, $\Z$ |
| calligraphic | a collection or a functional | $\Data$, $\Loss$ |
| Greek lowercase | usually a parameter | $\theta$, $\lambda$, $\sigma$ |

Now look again at {{eq:nll}}. Without knowing what any of it means, the
typography already tells you that $\Loss$ is a functional, $\theta$ is a
parameter, $N$ is a count, $y_i$ is a scalar, and $\vec{x}_i$ is a vector. That
is most of the structure, extracted from the fonts alone.

> IMPORTANT: When you meet an unfamiliar equation, read its typography before
> you read its content. Establishing what kind of object each symbol is takes
> ten seconds and makes the rest of the equation tractable.

### 4.2 Sums are loops

The single most common piece of notation in this book is the summation sign, and
it is exactly a `for` loop with an accumulator.

$$
\sum_{i=1}^{5} i^{2}
$$ (eq:sum-example)

reads: *start with $i = 1$; add $i^2$; increase $i$ by one; stop after $i = 5$.*
That is $1 + 4 + 9 + 16 + 25 = 55$. In Python:

```python {tier=C name=sum-as-loop}
total = 0
for i in range(1, 6):      # i = 1, 2, 3, 4, 5
    total += i ** 2
# total == 55
```

Three parts, always in the same places: **below** the sigma is the index
variable and where it starts; **above** is where it stops; **to the right** is
the thing being added, which usually mentions the index.

Everything else about summation notation is a variation on this. A sum with no
upper limit runs over whatever set is named below it. A sum with a condition
below it includes only the terms satisfying that condition. A double sum is a
nested loop.

### 4.3 Functions are contracts

A function signature such as

$$
f: \R^{n} \to \R^{m}
$$ (eq:signature)

is a type declaration, in exactly the sense a programmer means. It says: give
this function a vector of $n$ real numbers, and it returns a vector of $m$ real
numbers. It does not say what the function *does*; it says what shapes go in and
out.

This is worth taking seriously, because a large fraction of the bugs you will
write in Parts VI onward are shape errors, and a large fraction of the equations
you will fail to parse are equations whose shapes you have not checked. A
neural network layer is a function $\R^{d_{\text{in}}} \to \R^{d_{\text{out}}}$;
a loss is a function $\R^{m} \times \R^{m} \to \R$; a whole model is a
composition of such functions. Reading signatures is how you keep track.

## 5. Formal Explanation

### 5.1 Sets

A **set** is an unordered collection of distinct objects. Two notations define
them:

- By listing: $S = \{2, 3, 5, 7\}$.
- By a rule: $S = \{x \in \N : x < 10 \text{ and } x \text{ is prime}\}$, read
  "the set of natural numbers $x$ such that $x$ is less than 10 and prime". The
  colon is read "such that"; a vertical bar is sometimes used instead.

{#tbl:set-notation caption="Set notation used in this book."}

| Notation | Meaning |
|---|---|
| $x \in S$ | $x$ is an element of $S$ |
| $x \notin S$ | $x$ is not an element of $S$ |
| $A \subseteq B$ | every element of $A$ is in $B$ |
| $A \cup B$ | union: in $A$, or $B$, or both |
| $A \cap B$ | intersection: in both |
| $A \setminus B$ | in $A$ but not in $B$ |
| $\varnothing$ | the empty set |
| $\lvert S \rvert$ | cardinality: the number of elements |
| $A \times B$ | all ordered pairs $(a, b)$ with $a \in A$, $b \in B$ |

The standard number sets are $\N$ (natural numbers, taken here to start at 1),
$\Z$ (integers), and $\R$ (reals). Superscripts build tuples:
$\R^{n}$ is the set of ordered $n$-tuples of reals — which is to say, vectors of
length $n$ — and $\R^{m \times n}$ is the set of $m$-by-$n$ real matrices.

> NOTE: Sets are unordered and have no repeats, so $\{1, 2\}$ and $\{2, 1\}$ are
> the same set and $\{1, 1, 2\}$ is just $\{1, 2\}$. When order matters, the
> object is a *tuple*, written with parentheses: $(1, 2) \neq (2, 1)$. A vector
> is a tuple, not a set. This distinction is why $\R^{n}$ is defined with a
> Cartesian product rather than as a set of values.

**Intervals** are sets of real numbers between two bounds. Square brackets
include the endpoint, round brackets exclude it: $[0, 1]$ contains both 0 and 1,
$(0, 1)$ contains neither, and $[0, 1)$ contains 0 but not 1. This matters more
than it looks: a probability lies in $[0, 1]$, but the output of a logistic
function lies in $(0, 1)$ — it approaches the endpoints without ever reaching
them, which is why a model can never be *exactly* certain.

### 5.2 Summation and product notation

Formally,

$$
\sum_{i=a}^{b} f(i) = f(a) + f(a+1) + \cdots + f(b)
$$ (eq:sum-def)

with the convention that the sum is **empty**, and therefore equal to 0, when
$b < a$. The corresponding product is

$$
\prod_{i=a}^{b} f(i) = f(a) \cdot f(a+1) \cdots f(b)
$$ (eq:prod-def)

which is empty and equal to **1** when $b < a$ — the identity of the operation
in each case.

Variations you will meet constantly:

- **Over a set:** $\sum_{x \in S} f(x)$ adds $f(x)$ for every element of $S$.
  Since sets are unordered this is only well defined because addition is
  commutative.
- **Conditional:** $\sum_{i : y_i = 1} x_i$ adds $x_i$ over exactly those $i$
  for which $y_i = 1$.
- **Nested:** $\sum_{i=1}^{m}\sum_{j=1}^{n} A_{ij}$ is a double loop, summing
  every entry of an $m \times n$ grid.
- **Implicit range:** $\sum_i x_i$ omits the bounds when they are obvious from
  context. Common, and occasionally a source of genuine ambiguity.

Three properties get used constantly and are worth stating once:

$$
\sum_{i} c\,f(i) = c\sum_{i} f(i), \qquad
\sum_{i}\big(f(i) + g(i)\big) = \sum_{i} f(i) + \sum_{i} g(i)
$$ (eq:sum-linearity)

$$
\sum_{i=1}^{m}\sum_{j=1}^{n} a_{ij} = \sum_{j=1}^{n}\sum_{i=1}^{m} a_{ij}
$$ (eq:sum-swap)

{{eq:sum-linearity}} is the linearity of summation: constants pull out, and sums
of sums split. {{eq:sum-swap}} says the order of nested finite sums does not
matter. Both are used without comment throughout the book, and
{{eq:sum-linearity}} is the reason expectation is linear
({{ch:math-random-vars}}).

> WARNING: {{eq:sum-swap}} holds for *finite* sums unconditionally. For infinite
> sums it requires absolute convergence, and there exist conditionally
> convergent double series where swapping the order changes the answer. This
> book only ever sums finitely many terms, so the issue does not arise — but it
> is worth knowing the caveat exists, because it is the same caveat that makes
> expectations of heavy-tailed distributions misbehave.

### 5.3 Functions

A **function** $f: X \to Y$ assigns to each element of the **domain** $X$
exactly one element of the **codomain** $Y$. "Exactly one" is the whole content
of the definition: a rule that might return two different answers for the same
input is not a function.

Notation for the same function varies:

$$
f(x) = x^{2} + 1
\qquad\text{or}\qquad
f : x \mapsto x^{2} + 1
$$ (eq:function-notation)

The barred arrow $\mapsto$ maps an *element* to an element; the plain arrow
$\to$ maps a *set* to a set. So $f: \R \to \R$ and $f: x \mapsto x^2 + 1$ say
different things about the same object — the first its type, the second its
rule.

Argument conventions worth recognising:

- $f(x; \theta)$ — a semicolon separates inputs from parameters. Same as
  $f_{\theta}(x)$; both mean $x$ varies and $\theta$ is fixed for now.
- $f(x \given y)$ — the vertical bar is *conditioning*, used for probabilities
  ({{ch:math-probability}}). $p(x \given y)$ is a distribution over $x$ for each
  fixed $y$.
- $f \circ g$ — composition, meaning $f(g(x))$. Right to left, which is a common
  early confusion ({{ch:math-functions}}).

### 5.4 Indicator functions

The **indicator function** turns a true-or-false condition into a number:

$$
\mathbb{1}[P] = \begin{cases} 1 & \text{if } P \text{ is true} \\ 0 & \text{otherwise}\end{cases}
$$ (eq:indicator)

It exists because it converts counting into summation. "The number of examples
whose label is 1" is awkward in prose and immediate in symbols:

$$
\lvert \{i : y_i = 1\} \rvert = \sum_{i=1}^{N} \mathbb{1}[y_i = 1]
$$ (eq:indicator-count)

Accuracy — the fraction of predictions that are correct — is then a single
expression:

$$
\text{accuracy} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}[\hat{y}_i = y_i]
$$ (eq:accuracy)

The indicator's other important property, which {{ch:math-random-vars}} uses, is
that its expectation is a probability: $\E[\mathbb{1}[P]] = \Prob(P)$.

### 5.5 Symbols used throughout

{#tbl:common-symbols caption="Symbols that appear across the whole book. The complete table is in the notation appendix; these are the ones worth recognising immediately."}

| Symbol | Read as | Note |
|---|---|---|
| $\approx$ | approximately equal to | |
| $\propto$ | proportional to | equal up to a constant factor |
| $\triangleq$ | is defined as | distinguishes a definition from a derived equality |
| $\forall$ | for all | |
| $\exists$ | there exists | |
| $\implies$ | implies | |
| $\iff$ | if and only if | implication in both directions |
| $\to$ | maps to / tends to | context distinguishes them |
| $\lvert x \rvert$ | absolute value | for a set, cardinality |
| $\norm{\vec{x}}$ | norm | the size of a vector ({{ch:math-norms}}) |
| $\hat{y}$ | "y hat" | an estimate, never the truth |
| $\bar{x}$ | "x bar" | a sample mean |
| $x^{*}$ | "x star" | an optimal value |
| $\E[\cdot]$ | expectation | ({{ch:math-random-vars}}) |
| $\argmax_{x}$ | the $x$ that maximises | returns an argument, not a value |

> IMPORTANT: $\max$ and $\argmax$ are different and the difference matters. For
> $f(x) = -(x-3)^2$: $\max_x f(x) = 0$ is the largest *value*, while
> $\argmax_x f(x) = 3$ is the *input* achieving it. Classification returns an
> argmax — the label — not a max.

## 6. Mathematical Foundation

### 6.1 Reading a real equation

The point of this chapter is a procedure, not a list. Here it is applied to
{{eq:nll}}, the cross-entropy loss you will meet properly in
{{ch:math-optimization}} and use constantly from Part VI onward.

$$
\Loss(\theta) = -\frac{1}{N}\sum_{i=1}^{N} \log p_{\theta}(y_i \given \vec{x}_i)
$$

**Step 1 — types.** $\Loss$ is calligraphic: a functional, returning a scalar.
$\theta$ is Greek lowercase: parameters. $N$ is uppercase italic: a count.
$\vec{x}_i$ is bold lowercase: a vector. $y_i$ is italic lowercase: a scalar.
$p_{\theta}$ is a function subscripted by parameters.

**Step 2 — the outermost structure.** Strip everything inside the sum. What
remains is $\Loss(\theta) = -\frac{1}{N}\sum(\cdots)$: a negated average.
Whatever is inside, this is a mean.

**Step 3 — the index.** $i$ runs from 1 to $N$, and appears in both $y_i$ and
$\vec{x}_i$. So the sum is over examples, each of which has an input vector and
a scalar label.

**Step 4 — the innermost term.** $p_{\theta}(y_i \given \vec{x}_i)$ is a
probability: the model's assigned probability of the true label $y_i$, given
input $\vec{x}_i$. Take its log.

**Step 5 — reassemble.** The average, over the dataset, of the negative log of
the probability the model gave the correct answer.

**Step 6 — sanity-check the sign.** A probability lies in $(0, 1]$, so its log
is $\le 0$, so the negative of it is $\ge 0$. The loss is non-negative, and it
is zero exactly when the model assigns probability 1 to every correct answer.
Minimising it therefore means making the model confident and right.

That last step is the one to internalise. Checking the sign and the extreme
cases costs fifteen seconds and catches both your misreadings and, occasionally,
genuine errors in what you are reading.

### 6.2 A worked numerical example

Take $N = 3$ examples where the model assigns the true label probabilities
$0.9$, $0.6$ and $0.2$.

$$
\Loss = -\tfrac{1}{3}\big(\log 0.9 + \log 0.6 + \log 0.2\big)
$$

Term by term, in natural logs: $\log 0.9 = -0.1054$,
$\log 0.6 = -0.5108$, $\log 0.2 = -1.6094$. Their sum is $-2.2256$; divided by 3
gives $-0.7419$; negated, $\Loss = 0.7419$.

Two things to notice. The confident-and-correct example contributes almost
nothing ($0.105$), while the one the model nearly got wrong contributes
$1.609$ — more than fifteen times as much. Cross-entropy punishes confident
errors disproportionately, and that asymmetry is a property of the logarithm
rather than a design decision anyone made. And if any single probability were
0, the loss would be infinite, which is why implementations clamp probabilities
away from zero.

## 7. Implementation

Notation exists to be translated into code. The correspondence is close enough
that reading one should suggest the other.

```python {tier=A name=notation-to-numpy}
"""Mathematical notation and its NumPy equivalents, with the identities of
section 5.2 checked numerically rather than asserted.
"""
import numpy as np

# --- summation: sum_{i=1}^{5} i^2 ------------------------------------------
print("sum i^2, i=1..5 :", sum(i**2 for i in range(1, 6)),
      "| numpy:", int((np.arange(1, 6) ** 2).sum()))

# --- product: prod_{i=1}^{5} i  (= 5!) -------------------------------------
print("prod i, i=1..5  :", int(np.prod(np.arange(1, 6))))

# --- the empty-sum and empty-product conventions ---------------------------
# A sum with no terms is 0; a product with no terms is 1. NumPy agrees, and it
# matters: it is why an unnormalised probability of "no evidence" is 1, not 0.
print("empty sum       :", np.sum(np.array([])),
      "| empty product:", np.prod(np.array([])))

# --- conditional sum: sum over i where y_i == 1 ----------------------------
x = np.array([10.0, 20.0, 30.0, 40.0])
y = np.array([1, 0, 1, 1])
print("sum x_i where y_i==1 :", x[y == 1].sum())

# --- indicator functions and eq. 1.10 --------------------------------------
y_true = np.array([1, 0, 1, 1, 0])
y_pred = np.array([1, 0, 0, 1, 0])
indicator = (y_pred == y_true).astype(float)     # 1[y_hat == y]
print("indicators      :", indicator, "| accuracy:", indicator.mean())

# --- nested sums, and eq. 1.7 (order does not matter for finite sums) -------
A = np.arange(12).reshape(3, 4)
print("sum over i then j:", A.sum(axis=0).sum(),
      "| j then i:", A.sum(axis=1).sum(),
      "| all at once:", A.sum())
assert A.sum(axis=0).sum() == A.sum(axis=1).sum() == A.sum()

# --- eq. 1.6: linearity of summation ---------------------------------------
f = np.array([1.0, 2.0, 3.0])
g = np.array([10.0, 20.0, 30.0])
c = 2.5
assert np.isclose((c * f).sum(), c * f.sum())
assert np.isclose((f + g).sum(), f.sum() + g.sum())
print("linearity of summation verified")

# --- max vs argmax ----------------------------------------------------------
scores = np.array([0.1, 0.7, 0.2])
print("max  :", scores.max(), "(the value)")
print("argmax:", scores.argmax(), "(the index) — a classifier returns this")

# --- eq. 1.1 on the worked example of section 6.2 ---------------------------
p_true = np.array([0.9, 0.6, 0.2])
loss = -np.mean(np.log(p_true))
print(f"\ncross-entropy loss: {loss:.4f}")
assert np.isclose(loss, 0.7419, atol=1e-4)
print("per-example contributions:", np.round(-np.log(p_true), 4))
print(f"the near-miss contributes {np.log(0.2) / np.log(0.9):.1f}x as much "
      f"as the confident-correct example")
```

> NOTE: One-based mathematics and zero-based code are reconciled by convention,
> not by cleverness. A sum written $\sum_{i=1}^{N}$ becomes `range(N)` or a
> whole-array operation, and the index shifts by one. This book writes
> mathematics the way the literature does and code the way Python runs, and
> flags the transition wherever both appear in the same passage — off-by-one
> errors introduced exactly here are among the most common bugs in
> implementations of positional encoding and attention masks.

## 8. Practical Example

Here is a fragment from a real paper's method section, of the kind you will be
able to read by the end of Part I:

$$
\hat{y} = \argmax_{c \in \mathcal{C}} \; p(c \given \vec{x}), \qquad
p(c \given \vec{x}) = \frac{\exp(z_c)}{\sum_{c' \in \mathcal{C}} \exp(z_{c'})}
$$ (eq:classifier)

Apply the procedure. $\hat{y}$ has a hat: a prediction. $\mathcal{C}$ is
calligraphic: a set — of classes. $\vec{x}$ is a vector: the input. $z_c$ is a
scalar indexed by class: a score. The second equation divides $\exp$ of one
score by the sum of $\exp$ over all scores, so its outputs are positive and sum
to one: a probability distribution over classes. The first equation takes the
argmax, returning the class itself rather than its probability.

Read as prose: *convert the per-class scores into probabilities by exponentiating
and normalising, then predict whichever class has the highest one.* The second
equation is the softmax; you now have its definition, and
{{ch:math-functions}} will explain why exponentiating first is the right move.

Note also the primed index $c'$ in the denominator. The prime distinguishes the
summation variable from the fixed $c$ in the numerator — without it the
expression would be ambiguous. This is a small convention that trips up many
first-time readers and appears in almost every softmax written down.

## 9. Common Mistakes

**Reading $\sum$ as a single quantity.** It is an operator with three parts. The
index below, the limit above, and the summand to the right are all essential;
skipping any of them produces a misreading.

**Confusing $\max$ with $\argmax$.** One returns a value, the other an argument.
{{ch:math-optimization}} depends on the distinction and so does every
classifier.

**Ignoring typography.** Treating $\vec{x}$ and $x$ as the same object leads to
shape errors that no amount of algebra will fix.

**Missing the difference between $\in$ and $\subseteq$.** $x \in S$ says $x$ is
an *element*; $A \subseteq S$ says $A$ is a *subset*. For $S = \{1, 2, 3\}$:
$1 \in S$ is true, $\{1\} \in S$ is false, $\{1\} \subseteq S$ is true.

**Assuming $[0,1]$ and $(0,1)$ are interchangeable.** A logistic output lies in
the open interval and can never equal 0 or 1 exactly, which is why a saturated
sigmoid produces vanishing gradients rather than a clean answer.

**Reading composition left to right.** $f \circ g$ means apply $g$ first. The
notation genuinely reads backwards relative to the order of operations.

**Not checking the sign.** Losses are conventionally minimised and are usually
non-negative; log-likelihoods are maximised and are usually negative. Getting
the sign backwards produces a model that trains confidently in the wrong
direction, and the symptom — a loss that rises steadily — is easy to
misdiagnose.

## 10. Connection to Previous Chapters

This is the first chapter, so there is nothing behind it. What lies ahead:

{{ch:math-functions}} uses the function notation of
{{sec:5-formal-explanation}} to develop the exponential and logarithm, which
{{sec:6-mathematical-foundation}} has already used informally.
{{ch:math-vectors}} gives the dot product, which is the summation notation of
this chapter applied to two vectors at once. {{ch:math-random-vars}} defines
expectation, whose linearity is exactly {{eq:sum-linearity}}, and shows that the
expectation of an indicator is a probability. {{ch:math-optimization}} returns to
{{eq:nll}} and derives it rather than merely reading it.

Beyond Part I, the notation is used everywhere without further comment. The
notation appendix is the reference; this chapter is the tutorial.

## 11. Exercises

**Beginner**

1. Evaluate $\sum_{i=1}^{4} (2i + 1)$ by hand.
2. Evaluate $\prod_{i=1}^{4} i$ and $\sum_{i=5}^{3} i$. Explain the second
   answer.
3. For $S = \{1, 2, 3, 4\}$ and $T = \{3, 4, 5\}$, write out $S \cup T$,
   $S \cap T$, $S \setminus T$, and $\lvert S \times T \rvert$.
4. State whether each is true: $2 \in S$; $\{2\} \in S$; $\{2\} \subseteq S$;
   $\varnothing \subseteq S$.
5. For $f(x) = 3x - 2$, write the signature $f: ? \to ?$ and compute $f(4)$.

**Intermediate**

6. Write, using summation and indicator notation, "the number of examples in a
   dataset of size $N$ whose label is 1 and whose prediction is 0". This
   quantity has a name you will meet in Part IV.
7. Expand $\sum_{i=1}^{2}\sum_{j=1}^{3} (i \cdot j)$ fully, then verify
   {{eq:sum-swap}} by computing it in the other order.
8. A function has signature $f: \R^{5} \to \R^{3}$. What is the shape of its
   input, its output, and its Jacobian ({{ch:math-derivatives}})?
9. Rewrite $\frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2$ in words, then state
   what it measures and whether it should be maximised or minimised.
10. Explain the role of the primed index $c'$ in {{eq:classifier}}. What would
    go wrong without it?

**Advanced**

11. Prove {{eq:sum-linearity}} from the associativity and commutativity of
    addition.
12. Show that $\E[\mathbb{1}[P]] = \Prob(P)$, given that expectation is a
    probability-weighted average. (You may take the definition of expectation
    from {{ch:math-random-vars}} on faith for now.)
13. Give an example of a rule from $\R$ to $\R$ that is *not* a function, and
    say precisely which part of the definition it violates.

**Implementation**

14. Write a function `nll(probs)` computing {{eq:nll}} from an array of
    probabilities assigned to the true labels. Handle the case where a
    probability is 0 in a way you can defend, and explain your choice.
15. Implement {{eq:accuracy}} without using `==` on arrays — that is, with an
    explicit loop and an indicator — then verify it against the vectorised
    version on random data.
16. Write a function that takes a NumPy array `A` of shape $(m, n)$ and returns
    $\sum_{i}\sum_{j} A_{ij}^2$ three different ways: nested loops, one
    `np.sum`, and a dot product with a flattened copy. Confirm all three agree.

**Reasoning**

17. Why does mathematical notation index from 1 while nearly every programming
    language indexes from 0? What class of bug does the mismatch cause, and
    where in this book would you expect it to appear?
18. {{eq:nll}} divides by $N$. What would change, in practice, if it did not?
    Consider both the numerical value and the effect on the learning rate you
    would need.

## 12. Chapter Summary

Mathematical notation is a type system. Typography tells you what kind of object
each symbol is — scalar, vector, matrix, set, function, parameter — before you
understand anything about its meaning, and reading the types first is the single
most useful habit for parsing unfamiliar equations.

Summation notation is a `for` loop with an accumulator: index and start below,
stop above, summand to the right. Empty sums are 0 and empty products are 1.
Summation is linear, and the order of nested finite sums does not matter.

A function signature $f: X \to Y$ is a contract about shapes, not behaviour.
Keeping track of signatures is how you avoid the shape errors that dominate
practical work with tensors.

Indicator functions convert counting into summation, which is how accuracy,
error counts, and many probabilistic quantities get written compactly. The
expectation of an indicator is a probability.

The procedure for an unfamiliar equation is: identify the types, strip to the
outermost structure, find what the index ranges over, read the innermost term,
reassemble in words, then check the sign and the extreme cases. That last step
takes seconds and catches most misreadings.

For a second treatment of this material, {{cite:deisenroth2020}} covers the same
ground with a different emphasis and is freely available.
