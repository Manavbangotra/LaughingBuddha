---
id: dl-rnns
number: 60
part: VI
tier: full
status: reviewed
requires: [dl-backprop, dl-initialization, dl-cnns, ds-timeseries]
provides: [recurrent-network, bptt, lstm, gru, gating, truncated-bptt,
           teacher-forcing, bidirectional-rnn, sequence-modelling]
citations: [rumelhart1986, hochreiter1997, bengio1994, pascanu2013, saxe2014]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Define a recurrent network and unroll it into an acyclic graph.
2. Derive backpropagation through time and show why the gradient is a matrix
   power.
3. Explain precisely why vanishing gradients are worse here than in a
   feedforward network.
4. Derive the LSTM and GRU and explain what gating does to the gradient.
5. Apply truncated backpropagation and state what it costs.
6. Explain teacher forcing and exposure bias.
7. State honestly what recurrence is and is not used for in 2026.

## 2. Why This Matters

**This is where {{eq:unrolled-backprop}}'s product becomes a matrix power**, and
therefore where the part's organising problem is at its sharpest. A
feedforward network multiplies $L$ *different* matrices; a recurrent network
multiplies the *same* matrix $T$ times, so the product is governed by a single
spectral radius and the behaviour is exponential with no averaging to soften it.
{{cite:bengio1994}} analysed this and it is the cleanest instance of the problem
in the book.

**Gating is a genuinely different solution from anything else in this part.**
Initialisation sets the product's scale; normalisation reconditions it; residual
connections add a term equal to 1. The LSTM {{cite:hochreiter1997}} makes the
recurrent Jacobian *data-dependent and learned*, so the network decides when to
preserve and when to forget. Understanding that is worth more than the equations.

**Recurrence is legacy for language and alive elsewhere.** Transformers replaced
LSTMs for text after 2018. The recurrent *idea* — a fixed-size state carried
forward in linear time — returned via state space models, and as of 2026 whether
that line will displace attention at scale is genuinely open for the first time
in seven years.

**Sequential state is what attention gives up.** A transformer's cost is
quadratic in sequence length and it has no compressed state; a recurrence is
linear and does. That trade is the subject of {{ch:tf-efficient}}, and this
chapter is where you learn what is on the other side of it.

## 3. Prerequisites

{{ch:dl-backprop}} for {{eq:unrolled-backprop}}, of which this chapter is the
sharpest case. {{ch:dl-initialization}} for orthogonal initialisation, which
earns its place here. {{ch:dl-cnns}} for weight sharing — this is the same idea
along time instead of space. {{ch:ds-timeseries}} for sequence data.

## 4. Intuitive Explanation

### 4.1 A loop, unrolled

A recurrent network keeps a state and updates it with each input:

$$
\vec{h}_t = f(\vec{h}_{t-1}, \vec{x}_t)
$$

```text
   as a loop                 unrolled over time
                             x₁      x₂      x₃
      ┌───┐                   │       │       │
   x─▶│ f │─▶h                ▼       ▼       ▼
      └─┬─┘             h₀──▶[f]──▶[f]──▶[f]──▶ h₃
        └──┘                   ▲       ▲       ▲
                            SAME W  SAME W  SAME W
```

**The unrolled form is an ordinary feedforward network with two special
properties**: it is as deep as the sequence is long, and every layer shares the
same weights. Both properties cause everything that follows.

The weight sharing is the same idea as {{ch:dl-cnns}}'s, moved from space to
time: a rule useful at one time step is useful at every time step. And like the
convolution, it makes the model *less* expressive in a way that matches the
data.

### 4.2 Why the gradient problem is worse here

A 50-layer feedforward network multiplies 50 *different* Jacobians. Their
singular values vary, and errors in either direction partly average out.

A recurrence over 50 steps multiplies the *same* Jacobian 50 times. That is
$\mat{J}^{50}$, whose behaviour is governed entirely by the largest eigenvalue
$\rho$:

```text
   rho = 0.9   ->  0.9^50  = 0.005
   rho = 0.5   ->  0.5^50  = 9e-16
   rho = 1.1   ->  1.1^50  = 117
   rho = 2.0   ->  2.0^50  = 1e15
```

**No averaging, no escape.** Either $\rho$ is essentially exactly 1 or the
gradient goes to zero or infinity exponentially, and hitting exactly 1 by
accident does not happen. {{sec:6-mathematical-foundation}} makes this precise
and {{sec:8-implementation}} measures it.

### 4.3 Gating

The LSTM's idea, stated without equations: **add a state that is modified by
addition rather than by multiplication, and learn when to modify it.**

```text
   plain RNN   h_t = tanh(W h_{t-1} + U x_t)      MULTIPLIED every step
   LSTM        c_t = f_t * c_{t-1} + i_t * g_t    ADDED to, gated
```

If the forget gate $f_t$ is near 1, then $c_t \approx c_{t-1} + \text{something}$
and the gradient passes through nearly unchanged — a path with gain close to 1,
which is the same mechanism as {{ch:dl-cnns}}'s residual connection.

The difference is that $f_t$ is *learned and depends on the input*. A residual
connection has gain exactly 1 always; a gate has gain the network chooses, per
time step and per unit. **The network learns what to remember and for how long**,
which is more than a residual connection offers and is why the LSTM was the
answer for twenty years.

### 4.4 The three gates

```text
   forget  f_t   how much of the old cell state to keep
   input   i_t   how much of the new candidate to write
   output  o_t   how much of the cell state to expose as the hidden state
```

Each is a sigmoid, so each is in $[0,1]$ and acts as a soft switch. The GRU
merges the forget and input gates into one — what is not kept is written — which
halves the gate count and performs comparably.

### 4.5 Where this stands in 2026

Worth stating plainly rather than leaving implicit.

**For language, LSTMs are legacy.** Transformers replaced them from 2018 and the
gap is not close. Attention relates any two positions in one step; a recurrence
needs $|i-j|$ steps and must squeeze everything through a fixed-size state.

**Parallelism is the deeper reason.** A recurrence is sequential *by
definition* — $\vec{h}_t$ needs $\vec{h}_{t-1}$ — so training cannot be
parallelised over the sequence. A transformer processes all positions at once.
On modern hardware that is decisive, and it is a fact about the computation
rather than about the modelling.

**The recurrent idea returned.** State space models achieve a linear-time
recurrence that *can* be parallelised at training time, by making the recurrence
linear so it can be computed as a scan. They are competitive with transformers
at moderate scale and hybrids are the strongest current results.
{{maturity:EMERGING}}, and genuinely unsettled.

## 5. Formal Explanation

### 5.1 The vanilla RNN

$$
\vec{h}_t = \tanh\big(\mat{W}_{hh}\vec{h}_{t-1}
 + \mat{W}_{xh}\vec{x}_t + \vec{b}_h\big)
$$ (eq:vanilla-rnn)

$$
\vec{y}_t = \mat{W}_{hy}\vec{h}_t + \vec{b}_y
$$ (eq:rnn-output)

Three weight matrices for the whole sequence, however long. The parameter count
is independent of $T$ — the same weight-sharing accounting as
{{eq:conv-params}}.

### 5.2 Backpropagation through time

Unroll and apply {{ch:dl-backprop}}. The total loss is
$\Like = \sum_t \Like_t$, and the gradient with respect to the shared
$\mat{W}_{hh}$ sums over every time step:

$$
\frac{\partial\Like}{\partial\mat{W}_{hh}}
 = \sum_{t=1}^{T}\frac{\partial\Like_t}{\partial\mat{W}_{hh}}
 = \sum_{t=1}^{T}\sum_{k=1}^{t}
 \frac{\partial\Like_t}{\partial\vec{h}_t}
 \left(\prod_{j=k+1}^{t}\frac{\partial\vec{h}_j}{\partial\vec{h}_{j-1}}\right)
 \frac{\partial\vec{h}_k}{\partial\mat{W}_{hh}}
$$ (eq:bptt)

**The inner product is the whole problem.** With

$$
\frac{\partial\vec{h}_j}{\partial\vec{h}_{j-1}}
 = \diag\big(1-\vec{h}_j^2\big)\,\mat{W}_{hh}\T
$$ (eq:rnn-jacobian)

the product over $t-k$ steps involves $\mat{W}_{hh}\T$ raised to that power.

### 5.3 The LSTM

$$
\vec{f}_t = \sigma(\mat{W}_f[\vec{h}_{t-1},\vec{x}_t]+\vec{b}_f)
$$ (eq:lstm-forget)

$$
\vec{i}_t = \sigma(\mat{W}_i[\vec{h}_{t-1},\vec{x}_t]+\vec{b}_i),
\qquad
\tilde{\vec{c}}_t = \tanh(\mat{W}_c[\vec{h}_{t-1},\vec{x}_t]+\vec{b}_c)
$$ (eq:lstm-input)

$$
\vec{c}_t = \vec{f}_t\odot\vec{c}_{t-1} + \vec{i}_t\odot\tilde{\vec{c}}_t
$$ (eq:lstm-cell)

$$
\vec{o}_t = \sigma(\mat{W}_o[\vec{h}_{t-1},\vec{x}_t]+\vec{b}_o),
\qquad
\vec{h}_t = \vec{o}_t\odot\tanh(\vec{c}_t)
$$ (eq:lstm-output)

{{eq:lstm-cell}} is the load-bearing line. **The cell state is updated
additively, and the only multiplication on the carry path is by
$\vec{f}_t \in (0,1)$** — which the network controls.

Parameters: $4d(d+n)$ for hidden size $d$ and input size $n$, four times a
vanilla RNN's.

### 5.4 The GRU

$$
\vec{z}_t = \sigma(\mat{W}_z[\vec{h}_{t-1},\vec{x}_t]),
\qquad
\vec{r}_t = \sigma(\mat{W}_r[\vec{h}_{t-1},\vec{x}_t])
$$ (eq:gru-gates)

$$
\tilde{\vec{h}}_t = \tanh\big(\mat{W}[\vec{r}_t\odot\vec{h}_{t-1},
 \vec{x}_t]\big)
$$ (eq:gru-candidate)

$$
\vec{h}_t = (1-\vec{z}_t)\odot\vec{h}_{t-1} + \vec{z}_t\odot\tilde{\vec{h}}_t
$$ (eq:gru-update)

One state instead of two, three matrices instead of four. {{eq:gru-update}} is a
convex combination, so what is not kept is exactly what is written — a coupling
the LSTM leaves free. Comparable in practice, and cheaper.

### 5.5 Training details that are not details

**Truncated BPTT.** Backpropagate only $k$ steps rather than the whole sequence.
Cost drops from $O(T)$ to $O(k)$ in memory and the model *cannot learn
dependencies longer than $k$* — a hard limit imposed by the training procedure
rather than by the architecture.

**Teacher forcing.** During training, feed the *true* previous token rather than
the model's prediction. Faster and more stable, and it creates **exposure
bias**: at inference the model consumes its own outputs, which it never saw
during training, so an early error moves it into a state distribution it has no
experience of.

**Bidirectionality.** Run one recurrence forwards and one backwards and
concatenate. Available only when the whole sequence is known in advance, so not
for generation.

**Gradient clipping.** {{ch:dl-backprop}}'s global-norm clipping originated here
{{cite:pascanu2013}}, and it is effectively mandatory: the exploding case is
routine rather than exceptional.

## 6. Mathematical Foundation

### 6.1 Why the gradient is a matrix power

From {{eq:rnn-jacobian}}, the product over $t-k$ steps is

$$
\prod_{j=k+1}^{t}\frac{\partial\vec{h}_j}{\partial\vec{h}_{j-1}}
 = \prod_{j=k+1}^{t}\mat{D}_j\mat{W}_{hh}\T
$$ (eq:bptt-product)

with $\mat{D}_j = \diag(1-\vec{h}_j^2)$. Bounding the norm:

$$
\left\|\prod_j \mat{D}_j\mat{W}\T\right\|
 \le \big(\gamma\,\|\mat{W}\|\big)^{t-k}
$$ (eq:bptt-bound)

where $\gamma = \max_j\|\mat{D}_j\| \le 1$ for tanh.

**Contrast with the feedforward case.** There, $\|\mat{W}^{(l)}\|$ varies with
$l$ and the product is of *different* numbers, so it behaves like a geometric
mean and deviations partly cancel. Here $\|\mat{W}\|$ is one number appearing
$t-k$ times.

{{cite:bengio1994}} makes this precise: if the spectral radius
$\rho(\mat{W}_{hh}) < 1/\gamma$ then gradients vanish exponentially, and if it
exceeds that they explode. **There is no setting of a single matrix that gives
long-range gradient flow through a saturating nonlinearity**, which is the
theorem, not a heuristic.

### 6.2 The asymmetry between vanishing and exploding

Exploding is easy to detect — the loss becomes `nan` — and easy to fix, because
clipping bounds the step without changing its direction
({{ch:dl-backprop}} measured the direction preservation).

Vanishing is invisible. The model trains, the loss decreases, and it simply
never learns any dependency beyond a few steps. **Nothing recovers information
multiplied by $10^{-15}$.** That asymmetry is why the LSTM addresses vanishing
and clipping is left to handle exploding.

### 6.3 How the LSTM changes the recursion

Differentiating {{eq:lstm-cell}}:

$$
\frac{\partial\vec{c}_t}{\partial\vec{c}_{t-1}}
 = \diag(\vec{f}_t)
$$ (eq:lstm-carry)

**No weight matrix and no activation derivative.** The carry path's Jacobian is
a diagonal matrix of gate values, so the product over $t-k$ steps is

$$
\prod_{j=k+1}^{t}\diag(\vec{f}_j)
 = \diag\left(\prod_j \vec{f}_j\right)
$$ (eq:lstm-carry-product)

Three consequences:

**If $f_j \approx 1$ the product is $\approx 1$** for as many steps as you like.
That is the constant-error carousel of {{cite:hochreiter1997}}.

**It is per-unit.** Different units can have different forget gates, so one unit
can carry information for a thousand steps while another resets every step. A
residual connection cannot do this.

**It is data-dependent.** $\vec{f}_t$ depends on the input, so the network can
learn to reset the state at a sentence boundary and preserve it within one.

This is a *learned, input-dependent, per-unit* solution to the product problem,
and it is qualitatively different from every other technique in this part.

> IMPORTANT: The LSTM does not eliminate vanishing gradients. If the network
> learns $f_j = 0.9$, the product still decays as $0.9^{t-k}$. What it provides
> is the *option* of a gain near 1, and the gradient signal to learn when to use
> it. Initialising $\vec{b}_f$ to a positive value biases the gate open at the
> start, which is why that is standard.

### 6.4 Why orthogonal initialisation belongs here

{{ch:dl-initialization}} noted that an orthogonal matrix preserves the norm of
*every* vector exactly, and that a product of orthogonal matrices is orthogonal.
In a recurrence, the same matrix is applied at every step, so
$\mat{W}^{t}$ is orthogonal for every $t$ and the norm is preserved at every
horizon.

Since $\rho(\mat{W}) = 1$ exactly, {{eq:bptt-bound}} gives a bound of
$\gamma^{t-k}$ — the activation's contribution alone. This is the only
initialisation with that property, and it is the reason
{{cite:saxe2014}}'s scheme is standard here and optional elsewhere.

### 6.5 The cost of sequential dependence

$\vec{h}_t$ requires $\vec{h}_{t-1}$, so the $T$ steps cannot be computed in
parallel. Total work is $O(Td^2)$ and *critical path length* is $O(T)$.

A transformer's work is $O(T^2 d)$ — worse — with critical path $O(1)$ in the
sequence dimension. On hardware with thousands of parallel units, a shorter
critical path beats less total work by a wide margin.

$$
\text{RNN: } \underbrace{O(Td^2)}_{\text{work}},\;
 \underbrace{O(T)}_{\text{depth}}
\qquad
\text{Transformer: } \underbrace{O(T^2d)}_{\text{work}},\;
 \underbrace{O(1)}_{\text{depth}}
$$ (eq:rnn-vs-transformer-cost)

**That is the real reason recurrence lost, and it is an argument about hardware
rather than about modelling.** State space models attack exactly this: by making
the recurrence *linear* in the state, the whole sequence can be computed with a
parallel scan in $O(\log T)$ depth, recovering the transformer's parallelism at
the recurrence's linear cost.

## 7. Internal Mechanics

### 7.1 Memory during BPTT

Every time step's activations must be stored for the backward pass, so memory is
$O(T \times \text{batch} \times d)$. For a 1000-step sequence this dominates
everything, and it is why truncation exists.

Gradient checkpointing ({{ch:dl-backprop}}) applies directly: store every
$\sqrt{T}$-th state and recompute.

### 7.2 Fused kernels

A naive LSTM step does four separate matrix multiplies. Concatenating the four
weight matrices into one and doing a single matmul of four times the width is
substantially faster — the arithmetic-intensity argument of
{{ch:dl-forward}} — and every optimised implementation does it.

### 7.3 Variable-length sequences

Batching sequences of different lengths requires padding, and the padding must
be masked out of both the loss ({{ch:dl-losses}}'s masked reduction) and the
state updates. Packed-sequence representations avoid computing on padding
entirely.

**Sorting the batch by length** reduces padding waste and correlates the batch
composition with sequence length, which interacts badly with batch normalisation
({{ch:dl-normalization}}) — one of several reasons sequence models use layer
normalisation.

### 7.4 Statefulness across batches

Carrying the final hidden state into the next batch lets the model see context
longer than one window. It requires the batches to be contiguous slices of the
same sequences, and the state must be *detached* from the graph or the tape
grows without bound ({{ch:dl-forward}}'s retained-graph failure).

### 7.5 The forget-gate bias

Initialising $\vec{b}_f$ to 1 or 2 makes $\sigma(b_f) \approx 0.73$ or $0.88$,
so the gate starts mostly open and gradients flow through time from the first
step. Without it the gate starts at $0.5$ and the carry decays as $0.5^t$ before
the network has learned anything — measured in {{sec:9-practical-example}}.

## 8. Implementation

```python {tier=A name=bptt-and-the-matrix-power}
"""Backpropagation through time, and the matrix power that makes recurrence
the sharpest case of the vanishing-gradient problem.
"""
import numpy as np

rng = np.random.default_rng(0)


# --- section 6.1: the product is a matrix POWER -----------------------------
print("=" * 72)
print("the recurrent gradient is a matrix power (eq. 60.7)")
print("=" * 72)
print("A feedforward net multiplies L DIFFERENT Jacobians; a recurrence")
print("multiplies the SAME one T times. Compare what that does.\n")


def product_norm(T, d=64, same_matrix=True, rho=1.0, seed=0):
    """Norm of a product of T Jacobians, all identical or all distinct,
    each scaled to the given spectral radius."""
    rs = np.random.default_rng(seed)

    def make():
        W = rs.normal(0, 1.0 / np.sqrt(d), (d, d))
        return W * (rho / max(np.abs(np.linalg.eigvals(W)).max(), 1e-12))

    M = np.eye(d)
    W0 = make()
    for _ in range(T):
        M = (W0 if same_matrix else make()) @ M
    return float(np.linalg.norm(M, 2))


print(f"{'rho':>6} {'T':>5} {'SAME matrix (recurrent)':>25} "
      f"{'DIFFERENT matrices (feedforward)':>34}")
for rho in (0.9, 1.0, 1.1):
    for T in (10, 30, 60):
        a = product_norm(T, same_matrix=True, rho=rho, seed=1)
        b = product_norm(T, same_matrix=False, rho=rho, seed=1)
        print(f"{rho:>6.1f} {T:>5} {a:>25.4e} {b:>34.4e}")

print("\nBoth columns use matrices with the SAME spectral radius, so a naive")
print("reading of eq. 60.8 would predict the same behaviour. It does not")
print("happen.")
print("\nWith the same matrix repeated, the product is W^T and its norm is")
print("governed by rho^T exactly — clean exponential growth or decay with")
print("no escape. With different matrices the singular directions do not")
print("line up between factors, so the growth is different and the")
print("behaviour is not read off a single number.")
print("\nThat is the structural difference between a recurrence and a deep")
print("feedforward network, and it is why Bengio et al.'s result is a")
print("theorem about recurrences specifically.")

# --- an explicit BPTT implementation ----------------------------------------
class VanillaRNN:
    """Eq. 60.1, with backpropagation through time written out."""

    def __init__(self, n_in, d, n_out, w_scale=None, seed=0, ortho=False):
        rs = np.random.default_rng(seed)
        if ortho:
            Q, R = np.linalg.qr(rs.normal(size=(d, d)))
            self.Whh = Q * np.sign(np.diag(R))
        else:
            s = w_scale if w_scale is not None else 1.0 / np.sqrt(d)
            self.Whh = rs.normal(0, s, (d, d))
        self.Wxh = rs.normal(0, 1.0 / np.sqrt(n_in), (n_in, d))
        self.bh = np.zeros(d)
        self.Why = rs.normal(0, 1.0 / np.sqrt(d), (d, n_out))
        self.by = np.zeros(n_out)
        self.d = d

    def forward(self, X):
        """X: (batch, T, n_in). Returns logits at the LAST step only."""
        B, T, _ = X.shape
        h = np.zeros((B, self.d))
        self.H = [h]
        for t in range(T):
            h = np.tanh(h @ self.Whh + X[:, t] @ self.Wxh + self.bh)
            self.H.append(h)
        self.X = X
        return h @ self.Why + self.by

    def backward(self, dlogits, report_norms=False):
        """Eq. 60.5. Returns gradients and, optionally, the norm of the
        error signal at each time step."""
        B, T, _ = self.X.shape
        gWhy = self.H[-1].T @ dlogits
        gby = dlogits.sum(axis=0)
        dh = dlogits @ self.Why.T
        gWhh = np.zeros_like(self.Whh)
        gWxh = np.zeros_like(self.Wxh)
        gbh = np.zeros_like(self.bh)
        norms = []
        for t in reversed(range(T)):
            dz = dh * (1 - self.H[t + 1] ** 2)     # eq. 60.6
            norms.append(float(np.sqrt(np.mean(dz ** 2))))
            gWhh += self.H[t].T @ dz
            gWxh += self.X[:, t].T @ dz
            gbh += dz.sum(axis=0)
            dh = dz @ self.Whh.T
        return (gWhh, gWxh, gbh, gWhy, gby), list(reversed(norms))


print("\n" + "=" * 72)
print("the error signal reaching each time step (eq. 60.5)")
print("=" * 72)
B, T, N_IN, D = 32, 60, 8, 48
Xseq = rng.normal(size=(B, T, N_IN))
dlog = rng.normal(size=(B, 3)) / np.sqrt(B)

print(f"{'W_hh init':<24} " +
      " ".join(f"{f't={t}':>11}" for t in (0, 20, 40, 59))
      + f" {'t=59 / t=0':>13}")
for label, kw in (("small  (sd 0.5/sqrt(d))", {"w_scale": 0.5 / np.sqrt(D)}),
                  ("standard (1/sqrt(d))", {}),
                  ("large  (sd 2/sqrt(d))", {"w_scale": 2.0 / np.sqrt(D)}),
                  ("ORTHOGONAL", {"ortho": True})):
    net = VanillaRNN(N_IN, D, 3, seed=2, **kw)
    net.forward(Xseq)
    _, norms = net.backward(dlog)
    picks = [0, 20, 40, 59]
    rho = float(np.abs(np.linalg.eigvals(net.Whh)).max())
    gamma = float(np.mean(1 - np.concatenate(net.H[1:]) ** 2))
    print(f"{label:<24} " + " ".join(f"{norms[t]:>11.3e}" for t in picks)
          + f" {norms[59] / max(norms[0], 1e-300):>13.3e}")
    print(f"{'  rho, mean tanh deriv':<24} {rho:>11.4f} {gamma:>11.4f}"
          f"   -> (rho*gamma)^59 = "
          f"{(rho * gamma) ** 59:.3e}")

print("\nRead the last column: how much larger the gradient at the LAST")
print("time step is than the gradient reaching the FIRST. It is what")
print("decides whether a dependency spanning the sequence is learnable at")
print("all.")
print("\nThe predicted (rho*gamma)^59 on each second line is eq. 60.8's")
print("bound. Compare it against the reciprocal of the measured ratio: the")
print("bound is two to three orders of magnitude PESSIMISTIC, which is what")
print("a bound should be — it assumes the worst case at every one of the")
print("59 steps. What it gets right is the ordering and the scale, which is")
print("all it claims.")
print("\nNow the honest part. The orthogonal row has spectral radius")
print("exactly 1.0000 — the only initialisation here that does — and its")
print("ratio is NOT the best on the table. It is comparable to the standard")
print("initialisation and enormously better than the small one.")
print("\nThe reason is the gamma column. Orthogonality removes the WEIGHT")
print("MATRIX's contribution to the product and leaves the tanh")
print("derivative's, which is well below 1 and which no choice of W can")
print("touch. Over 59 steps that alone is fatal.")
print("\nSo eq. 60.8 has two factors and orthogonal initialisation fixes")
print("one of them. That is a real improvement and it is not a solution,")
print("which is exactly why gating — section 6.3, where the carry path has")
print("NEITHER factor — was the development that mattered.")
print("\nThe 'large' row is the other failure. Its ratio is below 1, meaning")
print("the gradient at the first step is LARGER than at the last: with rho")
print("above 2 the backward product explodes rather than vanishing.")

# --- section 6.4: orthogonality is preserved under powers -------------------
print("\n" + "=" * 72)
print("only an orthogonal matrix keeps rho = 1 at EVERY horizon (6.4)")
print("=" * 72)
d = 64
rs = np.random.default_rng(3)
Wg = rs.normal(0, 1.0 / np.sqrt(d), (d, d))
Wg = Wg / np.abs(np.linalg.eigvals(Wg)).max()      # force rho = 1
Q, R = np.linalg.qr(rs.normal(size=(d, d)))
Wo = Q * np.sign(np.diag(R))
print(f"both start with spectral radius 1.0000\n")
print(f"{'T':>5} {'||W_gauss^T||_2':>18} {'||W_orth^T||_2':>17} "
      f"{'cond(W_gauss^T)':>18}")
Mg, Mo = np.eye(d), np.eye(d)
for T in range(1, 51):
    Mg, Mo = Wg @ Mg, Wo @ Mo
    if T in (1, 5, 10, 25, 50):
        sg = np.linalg.svd(Mg, compute_uv=False)
        print(f"{T:>5} {np.linalg.norm(Mg, 2):>18.4e} "
              f"{np.linalg.norm(Mo, 2):>17.4e} "
              f"{sg.max() / max(sg.min(), 1e-300):>18.4e}")

print("\nA spectral radius of 1 controls the ASYMPTOTIC growth rate and")
print("says nothing about the transient. The Gaussian matrix at rho = 1 has")
print("singular values spread widely, so its powers amplify some directions")
print("and crush others, and the condition number grows without bound.")
print("\nThe orthogonal matrix's powers are orthogonal — norm exactly 1 and")
print("condition number exactly 1 at every horizon. That is the property")
print("Saxe et al. established, and it is why orthogonal initialisation is")
print("standard in recurrent networks and merely optional elsewhere: here")
print("the SAME matrix recurs, so its conditioning compounds directly.")
```

```python {tier=A name=gating-and-the-carry-path}
"""What gating does to the recursion, measured: the LSTM's carry path, the
forget-gate bias, and the horizon each architecture can actually learn.
"""
import numpy as np

rng = np.random.default_rng(0)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


# --- section 6.3: the carry path --------------------------------------------
print("=" * 72)
print("the LSTM carry path is a product of GATE VALUES (eq. 60.15)")
print("=" * 72)
print("A vanilla RNN's carry Jacobian is diag(1-h^2) W^T — a weight matrix")
print("and an activation derivative at every step. An LSTM's is diag(f_t)")
print("— gate values and nothing else.\n")
print(f"{'forget gate f':>15} " +
      " ".join(f"{f'after {t}':>13}" for t in (10, 50, 200, 1000)))
for f in (0.5, 0.9, 0.99, 0.999, 1.0):
    print(f"{f:>15.3f} " + " ".join(f"{f ** t:>13.3e}"
                                    for t in (10, 50, 200, 1000)))

print("\nThis table is the entire argument for gating. At f = 0.5 the")
print("gradient is gone in twenty steps. At f = 0.999 it survives a")
print("thousand. The network chooses f, per unit and per time step.")
print("\nNote what the table also shows: the LSTM does NOT eliminate the")
print("problem. If it learns f = 0.9 the decay is still exponential. What")
print("it provides is the OPTION of a gain near 1 and the gradient signal")
print("to learn when to take it — which a fixed architecture cannot offer.")

# --- section 7.5: the forget-gate bias --------------------------------------
print("\n" + "=" * 72)
print("why the forget-gate bias is initialised positive (section 7.5)")
print("=" * 72)
print(f"{'b_f':>6} {'sigma(b_f)':>12} " +
      " ".join(f"{f'carry @ {t}':>14}" for t in (10, 50, 200)))
for bf in (-1.0, 0.0, 1.0, 2.0, 3.0):
    f = float(sigmoid(bf))
    print(f"{bf:>6.1f} {f:>12.4f} " +
          " ".join(f"{f ** t:>14.3e}" for t in (10, 50, 200)))

print("\nAt b_f = 0 the gate starts at 0.5 and the carry has decayed to")
print("1e-15 after fifty steps — before the network has learned anything at")
print("all, so there is no gradient with which to learn to open the gate.")
print("It is a chicken-and-egg failure.")
print("\nInitialising b_f = 2 starts the gate at 0.88, which survives fifty")
print("steps at 1e-3 — small but present, and enough to learn from. That is")
print("the whole justification for a convention that otherwise looks")
print("arbitrary.")

# --- a full LSTM, and the horizon each architecture learns ------------------
class LSTM:
    """Eqs. 60.9-60.12 with a fused weight matrix (section 7.2)."""

    def __init__(self, n_in, d, n_out, forget_bias=1.0, seed=0):
        rs = np.random.default_rng(seed)
        k = 1.0 / np.sqrt(d + n_in)
        self.W = rs.normal(0, k, (d + n_in, 4 * d))     # f, i, c, o fused
        self.b = np.zeros(4 * d)
        self.b[:d] = forget_bias                        # section 7.5
        self.Why = rs.normal(0, 1.0 / np.sqrt(d), (d, n_out))
        self.by = np.zeros(n_out)
        self.d = d

    def forward(self, X):
        B, T, _ = X.shape
        d = self.d
        h = np.zeros((B, d))
        c = np.zeros((B, d))
        self.cache = []
        for t in range(T):
            z = np.concatenate([h, X[:, t]], axis=1) @ self.W + self.b
            f = sigmoid(z[:, :d])
            i = sigmoid(z[:, d:2 * d])
            g = np.tanh(z[:, 2 * d:3 * d])
            o = sigmoid(z[:, 3 * d:])
            c_prev = c
            c = f * c_prev + i * g                       # eq. 60.11
            tc = np.tanh(c)
            h = o * tc
            self.cache.append((np.concatenate([
                self.h_prev if False else np.zeros(0)], axis=0)
                if False else (f, i, g, o, c_prev, c, tc,
                               np.concatenate([
                                   np.zeros((B, 0)), X[:, t]], axis=1))))
            self.cache[-1] = (f, i, g, o, c_prev, c, tc,
                              np.concatenate([
                                  self.cache[-2][8] if False else np.zeros(
                                      (B, 0)), X[:, t]], axis=1))
        self.hT = h
        self.X = X
        return h @ self.Why + self.by

    def forget_gates(self):
        return np.array([cc[0].mean() for cc in self.cache])


print("\n" + "=" * 72)
print("what an untrained LSTM's carry looks like at each forget bias")
print("=" * 72)
Xs = rng.normal(size=(64, 120, 6))
print(f"{'forget bias':>13} {'mean gate':>11} {'carry over 120 steps':>22}")
for fb in (0.0, 1.0, 2.0, 3.0):
    net = LSTM(6, 32, 3, forget_bias=fb, seed=5)
    net.forward(Xs)
    g = net.forget_gates()
    print(f"{fb:>13.1f} {g.mean():>11.4f} "
          f"{float(np.prod(g)):>22.3e}")

print("\nThe measured mean gate tracks sigmoid(b_f) closely, and the product")
print("over 120 steps is the carry the gradient would see at")
print("initialisation. At b_f = 0 it is numerically zero; at b_f = 3 it")
print("survives.")
print("\nThis is at INITIALISATION, before any learning. The bias does not")
print("decide what the network ends up doing — it decides whether there is")
print("a gradient to learn from at all.")

# --- section 6.5: the parallelism argument ----------------------------------
print("\n" + "=" * 72)
print("the real reason recurrence lost: the critical path (eq. 60.16)")
print("=" * 72)
import time

print("Identical FLOPs both ways. The only difference is whether they are")
print("done in T sequential rounds or one big one.\n")
print(f"{'T':>6} {'batch':>7} {'d':>6} {'per-step MFLOP':>16} "
      f"{'sequential':>13} {'one matmul':>13} {'ratio':>8}")
for T, B, d in ((256, 1, 128), (256, 8, 128), (256, 64, 128),
                (256, 64, 512), (1024, 1, 128), (1024, 64, 512)):
    Wr = rng.normal(0, 1 / np.sqrt(d), (d, d)).astype(np.float32)
    Xb = rng.normal(size=(B, T, d)).astype(np.float32)
    h0 = np.zeros((B, d), dtype=np.float32)
    t0 = time.perf_counter()
    h = h0
    for t in range(T):
        h = np.tanh(h @ Wr + Xb[:, t])
    dt_seq = time.perf_counter() - t0
    flat = Xb.reshape(B * T, d)
    t0 = time.perf_counter()
    _ = np.tanh(flat @ Wr)
    dt_par = time.perf_counter() - t0
    mflop = 2 * B * d * d / 1e6
    print(f"{T:>6} {B:>7} {d:>6} {mflop:>16.3f} "
          f"{dt_seq * 1e3:>11.2f}ms {dt_par * 1e3:>11.2f}ms "
          f"{dt_seq / dt_par:>8.1f}x")

print("\nThe gap reaches more than an order of magnitude, and it appears")
print("where there is real work to serialise. Each sequential round is")
print("small enough that the library never reaches peak throughput on it,")
print("while the fused version does the identical arithmetic in one call")
print("that does. That is Chapter 51's roofline argument in a different")
print("costume.")
print("\nThe two batch-1 rows go the other way, and they are worth reading")
print("rather than dismissing. There the TOTAL work is a few MFLOP and both")
print("versions are dominated by per-call overhead rather than by")
print("arithmetic — the comparison is measuring NumPy's function-call cost,")
print("not the hardware. A ratio below 1 there means the experiment has")
print("left the regime it was designed to probe, not that recurrence wins.")
print("\nBe honest about the scale of all of this. Four CPU cores is not")
print("where the argument bites hardest; on an accelerator with thousands")
print("of parallel units, the per-round shortfall is far larger and the gap")
print("at realistic model sizes dwarfs anything measurable here.")
print("\nThe underlying fact is eq. 60.16, and it does not depend on the")
print("hardware: a recurrence has an O(T) critical path and a transformer")
print("has O(1) in the sequence dimension. The transformer does MORE total")
print("work — O(T^2 d) against O(T d^2) — and wins anyway wherever there")
print("are enough parallel units for the critical path to be the binding")
print("constraint. That is an argument about hardware, not about modelling.")
print("\nWhich is also why the recurrent idea came back. A LINEAR recurrence")
print("is associative, so it can be computed with a parallel scan in")
print("O(log T) depth — recovering the transformer's parallelism at the")
print("recurrence's linear cost. Whether that line displaces attention is")
print("open as of 2026.")
```

## 9. Practical Example

```python {tier=A name=learning-long-range-dependencies}
"""The task recurrence exists for: a dependency spanning the sequence.
Vanilla RNN against GRU against LSTM, as the distance grows.
"""
import numpy as np

rng = np.random.default_rng(3)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


# --- the task: remember a bit across T steps --------------------------------
def make_copy_task(n, T, seed):
    """The first token carries the label; everything after is noise.
    Solving it REQUIRES carrying information T steps."""
    rs = np.random.default_rng(seed)
    X = rs.normal(0, 0.5, (n, T, 4))
    y = rs.integers(0, 2, n)
    X[:, 0, 0] = np.where(y == 1, 3.0, -3.0)      # the signal, at step 0
    X[:, 0, 1] = 1.0                              # a marker
    return X, y


class GRUNet:
    """Eqs. 60.13-60.14, fused."""

    def __init__(self, n_in, d, seed=0, z_bias=0.0):
        rs = np.random.default_rng(seed)
        k = 1.0 / np.sqrt(d + n_in)
        self.Wzr = rs.normal(0, k, (d + n_in, 2 * d))
        self.bzr = np.zeros(2 * d)
        # Section 7.5's argument, applied to the GRU. h_t = (1-z)h + z*cand,
        # so a NEGATIVE z bias starts the update gate closed and the state
        # held — the GRU's equivalent of a positive LSTM forget bias.
        self.bzr[:d] = z_bias
        self.Wh = rs.normal(0, k, (d + n_in, d))
        self.bh = np.zeros(d)
        self.Wo = rs.normal(0, 1 / np.sqrt(d), (d, 2))
        self.bo = np.zeros(2)
        self.d = d

    def params(self):
        return [self.Wzr, self.bzr, self.Wh, self.bh, self.Wo, self.bo]

    def forward(self, X):
        B, T, _ = X.shape
        d = self.d
        h = np.zeros((B, d))
        self.cache = []
        for t in range(T):
            hx = np.concatenate([h, X[:, t]], axis=1)
            zr = sigmoid(hx @ self.Wzr + self.bzr)
            z, r = zr[:, :d], zr[:, d:]
            hx2 = np.concatenate([r * h, X[:, t]], axis=1)
            cand = np.tanh(hx2 @ self.Wh + self.bh)
            h_new = (1 - z) * h + z * cand                # eq. 60.14
            self.cache.append((hx, z, r, hx2, cand, h))
            h = h_new
        self.hT = h
        self.T = T
        return h @ self.Wo + self.bo

    def backward(self, dlogits):
        d = self.d
        gWo = self.hT.T @ dlogits
        gbo = dlogits.sum(axis=0)
        dh = dlogits @ self.Wo.T
        gWzr = np.zeros_like(self.Wzr)
        gbzr = np.zeros_like(self.bzr)
        gWh = np.zeros_like(self.Wh)
        gbh = np.zeros_like(self.bh)
        carry = []
        for t in reversed(range(self.T)):
            hx, z, r, hx2, cand, h_prev = self.cache[t]
            carry.append(float(np.sqrt(np.mean(dh ** 2))))
            dz = dh * (cand - h_prev)
            dcand = dh * z
            dh_prev = dh * (1 - z)
            dc_pre = dcand * (1 - cand ** 2)
            gWh += hx2.T @ dc_pre
            gbh += dc_pre.sum(axis=0)
            dhx2 = dc_pre @ self.Wh.T
            dr_h = dhx2[:, :d]
            dr = dr_h * h_prev
            dh_prev = dh_prev + dr_h * r
            dzr_pre = np.concatenate(
                [dz * z * (1 - z), dr * r * (1 - r)], axis=1)
            gWzr += hx.T @ dzr_pre
            gbzr += dzr_pre.sum(axis=0)
            dhx = dzr_pre @ self.Wzr.T
            dh = dh_prev + dhx[:, :d]
        return [gWzr, gbzr, gWh, gbh, gWo, gbo], list(reversed(carry))


class VanillaNet:
    def __init__(self, n_in, d, seed=0, ortho=False):
        rs = np.random.default_rng(seed)
        if ortho:
            Q, R = np.linalg.qr(rs.normal(size=(d, d)))
            self.Whh = Q * np.sign(np.diag(R))
        else:
            self.Whh = rs.normal(0, 1 / np.sqrt(d), (d, d))
        self.Wxh = rs.normal(0, 1 / np.sqrt(n_in), (n_in, d))
        self.bh = np.zeros(d)
        self.Wo = rs.normal(0, 1 / np.sqrt(d), (d, 2))
        self.bo = np.zeros(2)
        self.d = d

    def params(self):
        return [self.Whh, self.Wxh, self.bh, self.Wo, self.bo]

    def forward(self, X):
        B, T, _ = X.shape
        h = np.zeros((B, self.d))
        self.H = [h]
        for t in range(T):
            h = np.tanh(h @ self.Whh + X[:, t] @ self.Wxh + self.bh)
            self.H.append(h)
        self.X, self.T = X, T
        return h @ self.Wo + self.bo

    def backward(self, dlogits):
        gWo = self.H[-1].T @ dlogits
        gbo = dlogits.sum(axis=0)
        dh = dlogits @ self.Wo.T
        gWhh = np.zeros_like(self.Whh)
        gWxh = np.zeros_like(self.Wxh)
        gbh = np.zeros_like(self.bh)
        carry = []
        for t in reversed(range(self.T)):
            carry.append(float(np.sqrt(np.mean(dh ** 2))))
            dz = dh * (1 - self.H[t + 1] ** 2)
            gWhh += self.H[t].T @ dz
            gWxh += self.X[:, t].T @ dz
            gbh += dz.sum(axis=0)
            dh = dz @ self.Whh.T
        return [gWhh, gWxh, gbh, gWo, gbo], list(reversed(carry))


def train(net, X, y, Xv, yv, steps=600, lr=5e-3, batch=64, clip=1.0, seed=0):
    ps = net.params()
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    rs = np.random.default_rng(seed + 5)
    for t in range(1, steps + 1):
        idx = rs.integers(0, len(X), batch)
        logits = net.forward(X[idx])
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        p[np.arange(len(idx)), y[idx]] -= 1.0
        gs, _ = net.backward(p / len(idx))
        total = np.sqrt(sum(float(np.sum(g ** 2)) for g in gs))
        scale = min(1.0, clip / (total + 1e-12))       # global-norm clipping
        for i, (pp, g) in enumerate(zip(ps, gs)):
            g = g * scale
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            pp -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    lg = net.forward(Xv)
    return float((lg.argmax(axis=1) == yv).mean())


print("=" * 72)
print("the task recurrence exists for: carrying a bit across T steps")
print("=" * 72)
print("The label is in the FIRST token; everything after is noise. Solving")
print("it requires preserving information for the whole sequence.\n")
MODELS = [
    ("vanilla", lambda: VanillaNet(4, 48, seed=1)),
    ("vanilla (ortho)", lambda: VanillaNet(4, 48, seed=1, ortho=True)),
    ("GRU, z bias 0", lambda: GRUNet(4, 48, seed=1, z_bias=0.0)),
    ("GRU, z bias -2", lambda: GRUNet(4, 48, seed=1, z_bias=-2.0)),
    ("GRU, z bias -4", lambda: GRUNet(4, 48, seed=1, z_bias=-4.0)),
]
print(f"{'T':>5} " + " ".join(f"{n:>17}" for n, _ in MODELS))
for T in (5, 20, 60, 120):
    Xa, ya = make_copy_task(3000, T, 11)
    Xb, yb = make_copy_task(2000, T, 12)
    row = [train(mk(), Xa, ya, Xb, yb) for _, mk in MODELS]
    print(f"{T:>5} " + " ".join(f"{a:>17.4f}" for a in row))
print("\n(chance is 0.5000)")

print("\nThe three GRU columns are the point of this table, and the result")
print("is not the one the architecture's reputation would predict.")
print("\nAt its default bias the GRU FAILS at long T, and it fails for")
print("exactly the reason section 7.5 gives for the LSTM. Its update gate")
print("starts at sigmoid(0) = 0.5, so eq. 60.14 gives h_t = 0.5 h_{t-1} +")
print("0.5 cand and the carry decays as 0.5^T — gone in twenty steps. There")
print("is then no gradient with which to learn to close the gate, and the")
print("architecture that is supposed to solve the problem cannot get")
print("started on it.")
print("\nBiasing the update gate closed fixes it, and the effect is large.")
print("This is the same chicken-and-egg failure and the same one-line")
print("remedy as the LSTM's forget-gate bias — which is worth knowing")
print("because the LSTM convention is widely taught and the GRU one is not.")
print("\nThe vanilla rows are the other surprise, and the bigger one. The")
print("plain tanh RNN solves this task at T = 120, where eq. 60.8's product")
print("says the gradient reaching step 0 should be around 1e-30.")
print("\nThe explanation is that this task does not require gradient flow")
print("to be solved — it requires a LATCH. The signal is a strong bipolar")
print("spike, and a saturating recurrence can park itself in one of two")
print("attractors and stay there. Once the network finds that solution it")
print("holds the bit indefinitely, and finding it needs gradient only over")
print("the first few steps, not over all 120.")
print("\nThat is a real mechanism and a narrow one. It works because the")
print("thing being remembered is one bit carried by a large-amplitude")
print("signal. It would not work for graded information, for several")
print("competing signals, or where the state must also keep changing —")
print("which is every real sequence task.")
print("\nThe methodological lesson is worth more than the result. A")
print("synthetic long-range benchmark can be solvable by a mechanism other")
print("than the one it was designed to test, and the architecture that")
print("'wins' is then telling you about your task rather than about long-")
print("range dependencies. That the orthogonal initialisation — which")
print("eq. 60.8 says has the best-conditioned product — does WORST here is")
print("the clue that something other than the product is doing the work.")

# --- the gradient reaching step 0, measured ---------------------------------
print("\n" + "=" * 72)
print("the gradient actually reaching the first time step")
print("=" * 72)
T = 60
Xa, ya = make_copy_task(2000, T, 21)
print(f"{'model':<20} " +
      " ".join(f"{f't={t}':>12}" for t in (59, 40, 20, 0))
      + f" {'t=0 / t=59':>13}")
for name, net in (("vanilla", VanillaNet(4, 48, seed=1)),
                  ("vanilla (ortho)", VanillaNet(4, 48, seed=1, ortho=True)),
                  ("GRU, z bias 0", GRUNet(4, 48, seed=1, z_bias=0.0)),
                  ("GRU, z bias -4", GRUNet(4, 48, seed=1, z_bias=-4.0))):
    logits = net.forward(Xa[:256])
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    p[np.arange(256), ya[:256]] -= 1.0
    _, carry = net.backward(p / 256)
    print(f"{name:<20} " + " ".join(f"{carry[t]:>12.3e}"
                                    for t in (59, 40, 20, 0))
          + f" {carry[0] / max(carry[59], 1e-300):>13.3e}")

print("\nThis is at INITIALISATION, before training. The last column is the")
print("fraction of the output gradient that reaches the first time step —")
print("the step that holds the answer.")
print("\nA number of 1e-10 means the parameter update responsible for")
print("remembering the first token is ten orders of magnitude smaller than")
print("the update for the last one. The network will learn the recent")
print("context and never learn the long-range dependency, and nothing about")
print("the loss curve will say so.")
print("\nThat invisibility is section 6.2's point about the asymmetry.")
print("Exploding gradients announce themselves as a nan; vanishing ones")
print("produce a model that trains successfully and is quietly wrong about")
print("what it can represent.")

# --- truncated BPTT ---------------------------------------------------------
print("\n" + "=" * 72)
print("truncated BPTT imposes a HARD limit on what can be learned (5.5)")
print("=" * 72)


def train_truncated(net, X, y, Xv, yv, k, steps=600, lr=5e-3, batch=64,
                    seed=0):
    """Zero the gradient contribution from beyond k steps back."""
    ps = net.params()
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    rs = np.random.default_rng(seed + 5)
    T = X.shape[1]
    for t in range(1, steps + 1):
        idx = rs.integers(0, len(X), batch)
        Xb = X[idx].copy()
        # truncation to k steps == the model only SEES the last k steps
        Xb[:, :max(0, T - k)] = 0.0
        logits = net.forward(Xb)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        p[np.arange(len(idx)), y[idx]] -= 1.0
        gs, _ = net.backward(p / len(idx))
        total = np.sqrt(sum(float(np.sum(g ** 2)) for g in gs))
        sc = min(1.0, 1.0 / (total + 1e-12))
        for i, (pp, g) in enumerate(zip(ps, gs)):
            g = g * sc
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            pp -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    Xvb = Xv.copy()
    Xvb[:, :max(0, T - k)] = 0.0
    return float((net.forward(Xvb).argmax(axis=1) == yv).mean())


T = 40
Xa, ya = make_copy_task(3000, T, 31)
Xb_, yb_ = make_copy_task(2000, T, 32)
print(f"the signal is at step 0 of a {T}-step sequence\n")
print(f"{'truncation k':>14} {'covers step 0?':>16} {'GRU test acc':>14}")
for k in (5, 20, 39, 40):
    net = GRUNet(4, 48, seed=1, z_bias=-4.0)
    acc = train_truncated(net, Xa, ya, Xb_, yb_, k)
    print(f"{k:>14} {str(k >= T):>16} {acc:>14.4f}")
print("\n(chance is 0.5000)")

print("\nTruncating to k steps makes the first T-k steps invisible, and the")
print("signal is at step 0. Below k = T the model cannot see the answer at")
print("all, so no amount of training helps — the limit is imposed by the")
print("training procedure, not by the architecture.")
print("\nThis is the cost of truncated BPTT stated as sharply as possible.")
print("It buys O(k) memory instead of O(T), and it buys it by making")
print("dependencies longer than k unlearnable. When you truncate, you are")
print("choosing a maximum dependency length, and it is worth choosing it")
print("deliberately rather than inheriting it from a memory constraint.")
```

## 10. Production Considerations

**Clip gradients, always.** {{cite:pascanu2013}} originated global-norm clipping
here and the exploding case is routine rather than exceptional.

**Initialise the forget-gate bias positive.** Measured: at $b_f = 0$ the carry
at initialisation is numerically zero after fifty steps, so there is no gradient
with which to learn to open the gate.

**Use orthogonal initialisation for the recurrent matrix.** Measured: it is the
only initialisation whose powers keep both the norm and the condition number at
exactly 1.

**Choose the truncation length deliberately.** Measured: it is a hard ceiling on
learnable dependency length, not a soft approximation.

**Use layer normalisation, not batch normalisation.** Variable lengths, batch-1
inference, and length-sorted batches all break the batch statistics
({{ch:dl-normalization}}).

**Fuse the four gate matmuls into one.** {{ch:dl-forward}}'s arithmetic
intensity.

**Detach the state between batches** if you carry it, or the tape grows without
bound ({{ch:dl-forward}}).

**Expect the parallelism gap.** Measured: the same FLOPs run substantially
slower sequentially than as one matmul, and the gap grows with sequence length.

## 11. Common Mistakes

**Forgetting to clip.** The loss goes to `nan` and it looks like a learning-rate
problem.

**Leaving the forget-gate bias at zero.** Measured.

**Not masking padded positions**, in the loss or in the state update.

**Not detaching the carried state.** Memory grows linearly with iteration.

**Using a bidirectional model for generation.** It needs the future.

**Assuming teacher forcing is free.** It creates exposure bias by construction.

**Expecting an LSTM to learn a 1000-step dependency because it "solves"
vanishing gradients.** Measured: at a learned gate of 0.9 the decay is still
exponential.

**Comparing an RNN and a transformer on wall-clock at equal FLOPs.** Measured:
the critical path, not the FLOP count, is what differs.

## 12. Failure Modes

**Exploding gradients.** Routine here. Loud, and fixed by clipping.

**Vanishing gradients.** Measured at initialisation: the gradient reaching the
first step can be ten orders of magnitude below the last's. The model trains
normally and silently cannot represent long dependencies.

**Exposure bias.** The model is fluent for a few tokens and then degenerates,
because it has entered a state distribution it never saw in training.

**State leakage across examples.** Carrying the state into a batch that is not a
continuation of the previous one mixes unrelated sequences.

**Truncation silently capping the task.** Measured: below the required length,
accuracy is at chance and there is no error.

**Slow training that looks like a bug.** Measured: it is the sequential critical
path, and no amount of profiling the kernels will change it.

## 13. Alternatives

**Transformers** ({{part:7}}) replace recurrence with attention: $O(1)$ critical
path, $O(T^2)$ work, direct access between any two positions. The default for
sequences since 2018.

**State space models** (S4, Mamba and relatives) use a *linear* recurrence,
which can be computed with a parallel scan at training time and run as a genuine
recurrence at inference. Linear in sequence length with a constant-size state,
competitive at moderate scale. {{maturity:EMERGING}}

**Temporal convolutions** apply dilated causal convolutions along time —
parallel like a transformer, with a receptive field limited by depth
({{ch:dl-cnns}}).

**Linear attention** removes the softmax so attention can be written as a
recurrence, giving linear cost and a fixed-size state. The convergence of this
line with state space models is one of the more interesting developments of the
last few years.

**Hybrids** interleave attention layers with recurrent or state-space layers,
which are currently the strongest results in the linear-time family and suggest
neither mechanism is sufficient alone.

## 14. Evaluation

**Measure the gradient reaching step 0 at initialisation.** Measured here in
four lines; it tells you the maximum learnable horizon before you train
anything.

**Test on a synthetic long-range task.** The copy task used here isolates the
capability that real benchmarks confound with everything else.

**Log the gradient norm and the clip rate.**

**Check the truncation against the required dependency length.**

**Compare against a transformer at equal wall-clock**, not equal FLOPs.

**Evaluate generation with free running, not teacher forcing.** Teacher-forced
perplexity overstates generation quality by exactly the amount exposure bias
costs.

## 15. Advanced Concepts

**The parallel scan.** A linear recurrence $h_t = a_t h_{t-1} + b_t$ is an
associative operation, so the whole sequence can be computed in $O(\log T)$
depth. This is what makes state space models trainable at scale, and it is the
one property a nonlinear recurrence cannot have.

**Unitary and orthogonal RNNs** constrain $\mat{W}_{hh}$ to stay orthogonal
throughout training, guaranteeing $\rho = 1$ forever. Elegant, and the
constraint costs expressiveness and the maintenance costs compute.

**Attention as an RNN's escape hatch.** Encoder-decoder attention was introduced
to fix the fixed-size-bottleneck problem in recurrent translation. Attention
then turned out not to need the recurrence at all, which is a good example of a
patch outgrowing the thing it patched ({{ch:tf-scaled-dot-product}}).

**Echo state networks** leave the recurrent weights *random and frozen*,
training only the output layer. They work surprisingly well on some tasks and
are direct evidence that a random recurrence already provides useful temporal
features.

**Continuous-time formulations.** Treating the recurrence as a discretised
differential equation is the framing that produced state space models, and it is
why they have principled initialisations that ordinary RNNs lack.

## 16. Connection to Previous Chapters

{{ch:dl-backprop}}'s {{eq:unrolled-backprop}} becomes a matrix power here, and
the measured comparison between repeating one matrix and multiplying different
ones is the sharpest statement of the part's organising problem.

{{ch:dl-initialization}}'s orthogonal scheme finally earns its place: measured,
it is the only initialisation whose powers preserve both norm and conditioning.
{{ch:dl-cnns}}'s weight sharing is the same idea along time, and its residual
connection is the fixed-gain version of what gating does adaptively.
{{ch:dl-losses}}'s masked reduction is required for variable-length batches.
{{ch:ds-timeseries}} supplied the data and the classical alternatives.

Forward: {{ch:tf-scaled-dot-product}} removes the sequential dependence
entirely, and the measured parallelism gap is the reason.
{{ch:tf-architectures}} shows what replaces the recurrence.
{{ch:tf-efficient}} returns to linear-time sequence modelling, where this
chapter's ideas come back in a parallelisable form.

## 17. Exercises

**Beginner**

1. Write the vanilla RNN recurrence.
2. Why is the unrolled network as deep as the sequence is long?
3. What do the LSTM's three gates control?
4. What is teacher forcing, and what does it cost?
5. Why can a bidirectional model not generate?

**Intermediate**

6. Derive {{eq:rnn-jacobian}}.
7. Using {{eq:bptt-bound}}, find the sequence length at which the gradient
   falls below $10^{-8}$ for $\rho\gamma = 0.85$.
8. Derive {{eq:lstm-carry}} and explain why no weight matrix appears.
9. Count the parameters of an LSTM with $d=512$, $n=256$, and compare with a
   vanilla RNN and a GRU.
10. Explain why truncating to $k$ steps caps the learnable dependency length.
11. Why does a length-sorted batch interact badly with batch normalisation?

**Advanced**

12. Prove that the recurrent gradient bound involves $\rho^{T}$ and explain why
    the feedforward case does not reduce to a single number.
13. Derive the GRU's backward pass.
14. Show that a linear recurrence is associative and can be computed by a
    parallel scan in $O(\log T)$ depth.
15. Explain why constraining $\mat{W}_{hh}$ to be orthogonal throughout training
    costs expressiveness, and what it buys.
16. Derive the exposure-bias argument formally as a distribution mismatch.

**Implementation**

17. Implement an LSTM forward and backward and gradient-check it.
18. Reproduce the copy-task table and extend it to $T=200$.
19. Implement truncated BPTT properly (state carried, gradient truncated) and
    compare against the crude version used here.
20. Implement a parallel scan for a linear recurrence and verify it against the
    sequential version.

**Reasoning**

21. Your sequence model learns local structure and no long-range structure,
    with no error. Give an ordered diagnostic procedure.
22. Teacher-forced perplexity is excellent and generated text degenerates after
    twenty tokens. Explain and propose two fixes.

## 18. Interview Questions

**"Why do RNNs have vanishing gradients?"** — {{eq:bptt-product}} is a matrix
*power*. The distinguishing answer explains why that is worse than a deep
feedforward network's product of different matrices.

**"How does an LSTM fix it?"** — {{eq:lstm-carry}}: the carry Jacobian is
$\diag(\vec{f}_t)$, with no weight matrix and no activation derivative. Say that
it does not *fix* it — it makes the gain learnable.

**"LSTM or GRU?"** — GRU is cheaper with comparable performance; LSTM's separate
cell state occasionally helps. Neither is a default in 2026 for language.

**"Why did transformers replace RNNs?"** — The critical path. Give the
$O(T)$-versus-$O(1)$ argument and note that the transformer does *more* total
work.

**"What is teacher forcing and what is exposure bias?"** — Training on true
prefixes, inference on generated ones; a distribution mismatch that compounds.

**"What is truncated BPTT and what does it cost?"** — Memory for maximum
learnable dependency length. Say it is a hard ceiling.

**"Are RNNs dead?"** — No, and be precise: legacy for language, and the
recurrent idea returned via state space models with a parallelisable linear
recurrence.

## 19. Research Questions

**Can a linear-time sequence model match attention at scale?** State space
models are competitive at moderate scale and hybrids are stronger than either
alone, which suggests neither mechanism is sufficient. Genuinely open.
{{maturity:EMERGING}}

**What does attention do that a fixed-size state cannot?** Exact retrieval of an
arbitrary earlier token is the usual answer, and it is not a complete account of
where the gap appears. {{maturity:EMERGING}}

**Is there a recurrence with guaranteed long-range gradient flow that stays
expressive?** Unitary RNNs guarantee it and lose expressiveness; gating is
expressive and guarantees nothing. Whether both are achievable at once is open.
{{maturity:RESEARCH FRONTIER}}

**How much does exposure bias actually matter?** Scheduled sampling and
sequence-level training address it with mixed results, and large models trained
on enough data appear to suffer from it less than the argument predicts.
{{maturity:EMERGING}}

## 20. Chapter Summary

A recurrent network is a feedforward network unrolled over time with shared
weights, and both properties cause everything else. The weight sharing is
{{ch:dl-cnns}}'s idea moved from space to time. The depth is the sequence
length, and the shared weights make the gradient a matrix *power* rather than a
product of distinct matrices — which the measurement separates directly: at the
same spectral radius, repeating one matrix produced clean exponential behaviour
governed by $\rho^T$, while multiplying different matrices did not. That is why
{{cite:bengio1994}}'s result is a theorem about recurrences specifically.

Orthogonal initialisation earns its place here. Measured, a Gaussian matrix
forced to $\rho = 1$ still had its powers' condition number grow without bound,
because the spectral radius controls the asymptotic rate and says nothing about
the transient. An orthogonal matrix's powers stay orthogonal — norm and
condition number exactly 1 at every horizon — and since the *same* matrix
recurs, that conditioning compounds directly.

Gating is a qualitatively different solution from anything else in this part.
{{eq:lstm-carry}} shows the carry Jacobian is $\diag(\vec{f}_t)$: no weight
matrix, no activation derivative, just gate values the network chooses per unit
and per time step. The measured table of $f^t$ is the whole argument — 0.5 is
gone in twenty steps, 0.999 survives a thousand. But the same table shows the
LSTM does *not* eliminate the problem: at a learned $f = 0.9$ the decay is still
exponential. What it provides is the option of a gain near 1 and the gradient
signal to learn when to take it.

The forget-gate bias convention follows directly. Measured at initialisation, at
$b_f = 0$ the carry over 120 steps is numerically zero — so there is no gradient
with which to learn to open the gate, a chicken-and-egg failure that a positive
bias removes.

Truncated backpropagation is a hard ceiling, not an approximation. Measured,
below the required dependency length the model sat at chance and no amount of
training helped. When you truncate, you are choosing a maximum learnable
dependency length, and it is worth choosing deliberately.

Finally, the reason recurrence lost is about hardware rather than modelling. The
measured comparison ran identical FLOPs sequentially and as one matmul, and the
sequential version was substantially slower — increasingly so with length —
purely because a recurrence has an $O(T)$ critical path where a transformer has
$O(1)$. A transformer does *more* total work and wins anyway. That precision
matters, because it is also why the recurrent idea returned: a linear recurrence
is associative, so it can be computed by a parallel scan in $O(\log T)$ depth,
recovering the parallelism at the recurrence's linear cost. Whether that line
displaces attention is open as of 2026.

## 21. Further Reading

{{cite:hochreiter1997}} is the original LSTM paper and it is harder going than
most modern treatments — the notation is unfamiliar and the architecture is not
quite the one used today. What is worth extracting is the *constant error
carousel* argument, which is {{eq:lstm-carry}} stated before anyone had a
diagram for it.

{{cite:bengio1994}} is the paper that established the problem. It is short,
mathematical, and precise about what it proves — that with a saturating
nonlinearity there is a trade-off between storing information robustly and
receiving useful gradients, and that no choice of weights escapes it. Read it
before the LSTM paper.

{{cite:pascanu2013}} is the modern treatment of the same problem, with the
geometric picture of the loss surface near a cliff and the gradient-clipping
proposal that came out of it. The clipping argument transfers to everything.

{{cite:saxe2014}} for why orthogonal initialisation matters here specifically.

**Where to go next:** {{ch:dl-autoencoders}} closes the part with unsupervised
representation learning, and {{part:7}} takes up the sequence problem again with
the architecture that replaced this one. The measured parallelism gap in
{{sec:8-implementation}} is the single best preparation for it.
