---
id: fm-notation
status: final
---

## Why notation is fixed in advance

In a book this long, written over a long period, notation drifts unless
something stops it. The symbol table is therefore fixed before the first chapter
and enforced by the build: {{ch:app-notation}} is generated from the same file
the chapters draw on, so a symbol cannot mean one thing in Part VI and another
in Part XV without the change being visible in one place.

This section explains the conventions behind the table. The table itself is
{{ch:app-notation}}.

## Conventions

**Typography carries type information.** A lowercase italic letter is a scalar
($a$, $\alpha$). Lowercase bold is a column vector ($\vec{x}$). Uppercase bold is
a matrix ($\mat{A}$). Calligraphic letters are sets or functionals ($\Data$ for a
dataset, $\Loss$ for a loss). Once you internalise this, an equation's shapes are
readable without checking the surrounding text.

**Vectors are columns.** Every vector is a column vector unless it is written
transposed. An inner product is therefore $\vec{x}\T\vec{y}$, and an outer product
is $\vec{x}\vec{y}\T$.

**Tokens index rows.** This is the one convention most likely to trip you up
when moving between this book and the literature. A batch of token
representations is $\mat{X} \in \R^{n \times d_{\text{model}}}$: row $i$ is the
vector for position $i$. This matches how tensors are laid out in PyTorch and
how attention is implemented in practice, and it is why attention scores are
written $\mat{Q}\mat{K}\T$ rather than $\mat{K}\T\mat{Q}$.

> WARNING: Several important papers use the opposite convention, with features
> indexing rows and tokens indexing columns. Their equations are the transpose
> of the ones here. Whenever the book quotes an equation from a paper that uses
> the other convention, it says so at that point and gives both forms. If you
> derive a result and find your shapes do not match, check this first — it is
> the most common cause.

**A hat means an estimate.** $y$ is the truth, $\hat{y}$ is what a model
produced. This extends to parameters: $\theta$ is the true or optimal value,
$\hat{\theta}$ is what an estimator returned.

**Subscripts index, superscripts identify.** $\vec{x}_i$ is the $i$-th example
or the $i$-th coordinate, depending on context stated locally. $\mat{W}^{Q}$ is
the query projection — the superscript names which projection it is, not a power
or an index. Powers are written unambiguously, and transpose is always
$\phantom{}\T$.

**Learned parameters are collected in $\theta$.** When an equation does not need
to distinguish between individual weight matrices, it writes $\theta$ for all of
them and $\nabla_{\theta}\Loss$ for the gradient with respect to all of them. The
gradient always has the same shape as the thing it differentiates with respect
to.

**Distributions: $p$ is real, $q$ is a model.** Where both a data distribution
and a model distribution appear, $p$ is the one that exists in the world and $q$
is the one being fitted to it. A model distribution with explicit parameters is
$p_{\theta}$ when there is no competing $p$ in the same equation.

**Dimensions have names, not letters.** Attention involves at least four
different dimensions, and calling them all $d$ makes the equations unreadable.
This book writes $d_{\text{model}}$ for the residual-stream width, $d_k$ for the
query/key dimension, $d_v$ for the value dimension, and $d_{\text{ff}}$ for the
feed-forward hidden width, always. Sequence length is $n$, head count is $h$,
layer count is $L$, vocabulary size is $V$.

**Logs are natural.** $\log$ means $\log_e$ throughout, and entropies are in nats
unless a passage explicitly works in bits. Where a result is conventionally
quoted in bits — as some information-theoretic and compression results are — the
conversion is stated.

**Indices are one-based in mathematics, zero-based in code.** This book does not
try to reconcile the two. Equations are written the way they appear in the
literature; code is written the way Python runs. Where a chapter moves between
them in the same passage, the transition is called out explicitly, because
off-by-one errors introduced at exactly this boundary are among the most common
bugs in implementations of attention and positional encoding.

## Reused symbols

A few symbols carry more than one standard meaning, and there is no way to avoid
this without inventing notation nobody else uses. The book keeps the standard
meanings and disambiguates by context, stating which is meant on first use in
each chapter:

- $\sigma$ is a standard deviation in probability contexts and a singular value
  in linear-algebra contexts. It is also, in some literature, the logistic
  function — this book writes that as $\operatorname{sigmoid}$ instead.
- $L$ is the number of layers. A loss is always $\Loss$ or $\ell$, never $L$.
- $\tau$ is sampling temperature. Where a time constant is needed, it is named
  explicitly.
- $n$ is sequence length in Parts VII onward and a generic count in Parts I–V.
  Dataset size is always $N$.

## Reading an unfamiliar equation

When an equation resists you, the most reliable technique is to stop reading it
as a statement and start reading it as a shape.

Write the dimension of every symbol above it. Check that every product is
conformable. Identify which index is being summed over — that is almost always
where the meaning is. Then set every dimension to 1 and see what the scalar case
says; if the scalar case is obvious, the general case is usually the same idea
applied along an axis.

{{ch:tf-scaled-dot-product}} does exactly this, at length, for the attention
equation, and the same procedure works on most of what follows.
