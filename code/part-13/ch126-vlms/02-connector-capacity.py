# -*- coding: utf-8 -*-
# Extracted from: Chapter 126 — Vision-Language Models
# Source: src/.../ch126-vlms.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The connector is a bottleneck, and a fixed-length one is a hard ceiling.

Between the vision tower and the language model sits a connector that turns P
patch features into T tokens the LLM will read. The design space is small and the
consequences are not:

  FULL PROJECTION   T = P. Every patch becomes a token (cite:liu2023llava's
                    linear projector). Nothing is discarded and the context bill
                    is the full patch count.
  FIXED QUERIES     T is a constant, independent of the image
                    (cite:li2023blip2's Q-Former). Cheap, and it imposes a
                    capacity ceiling that does not move with content
                    (eq:connector-capacity).
  POOLING           T = P/k. A middle position with the same shape of limit.

The question this listing answers is not "which is more accurate" but "how many
distinct things can an image convey through each", because that is what a fixed T
bounds. Facts are placed in the image, passed through each connector, and then
recovered by a linear decoder -- so what is measured is whether the INFORMATION
survived, not whether some particular model used it.
"""
import numpy as np

rng = np.random.default_rng(97)

P, D = 64, 16          # patches, feature dimension per patch
N_IMG = 3000
NOISE = 0.35


def make_images(n_facts):
    """Each image contains a random subset of n_facts distinct facts. A fact
    lives in one patch and has its own direction."""
    where = rng.integers(0, P, size=n_facts)
    dirs = rng.normal(size=(n_facts, D))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    Y = (rng.random((N_IMG, n_facts)) < 0.5).astype(float)
    X = NOISE * rng.normal(size=(N_IMG, P, D))
    for f in range(n_facts):
        X[:, where[f], :] += Y[:, f:f + 1] * dirs[f]
    return X, Y


def connector(kind, T):
    """Return a (T, P) mixing matrix."""
    if kind == "full":
        return np.eye(P)
    if kind == "pool":
        A = np.zeros((T, P))
        g = P // T
        for t in range(T):
            A[t, t * g:(t + 1) * g] = 1.0 / g
        return A
    # Fixed learned queries, modelled as a dense random mixing over patches --
    # each query reads the whole image and emits one token.
    A = rng.normal(size=(T, P)) / np.sqrt(P)
    return A


def recoverable(X, Y, A):
    """Mean R^2 of a ridge decoder recovering each fact's presence from the
    connector's OUTPUT. This measures information survival, not model skill."""
    Z = np.einsum("tp,npd->ntd", A, X).reshape(len(X), -1)
    Z = np.hstack([Z, np.ones((len(Z), 1))])
    ridge = 1e-3 * np.eye(Z.shape[1])
    W = np.linalg.solve(Z.T @ Z + ridge, Z.T @ Y)
    pred = Z @ W
    ss_res = ((Y - pred) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0)
    return float(np.mean(1.0 - ss_res / ss_tot))


FACTS = (4, 16, 64, 256, 1024)
SETUPS = [("full projection (T=64)", "full", P),
          ("fixed queries, T=8", "query", 8),
          ("fixed queries, T=32", "query", 32),
          ("pooling to T=8", "pool", 8)]

print(f"{P} patches of {D} dims; facts recovered by a linear decoder from the "
      f"connector output\n")
print(f"{'connector':<26}{'tokens':>8}" + "".join(f"{str(m) + ' facts':>12}"
                                                  for m in FACTS))
print("-" * 94)

res = {}
for name, kind, T in SETUPS:
    row = []
    for m in FACTS:
        X, Y = make_images(m)
        row.append(recoverable(X, Y, connector(kind, T)))
    res[name] = row
    print(f"{name:<26}{T:>8}" + "".join(f"{v:>12.3f}" for v in row))

q8, q32 = res["fixed queries, T=8"], res["fixed queries, T=32"]
full, pool8 = res["full projection (T=64)"], res["pooling to T=8"]
print(f"""
The first thing to read is a column, not a row, and it corrects the intuition
this listing was built to test. The expectation was a KNEE -- a fixed-length
connector holding up while the content fits inside T tokens and collapsing once
it does not. There is no knee. At 4 facts, where eight tokens ought to be
plentiful, T=8 already recovers only {q8[0]:.3f} against full projection's
{full[0]:.3f}.

So a fixed-length connector is not "adequate until the image gets busy". It is
LOSSY ALWAYS, by an amount set by the compression ratio T/P, and the loss is
there in the simplest image. The ceiling is a uniform tax rather than a cliff
(eq:uniform-tax).

That distinction changes the diagnosis in a useful way. If it were a cliff you
would expect a connector to work and then fail as documents got denser, and you
would look for the threshold. What actually happens is that the information is
missing from the start and you only NOTICE when a task needs the part that was
discarded. Captioning does not need it -- a caption conveys a handful of facts, so
eight tokens carry enough of them and cite:li2023blip2's efficiency argument was
correct for what it was measured on. Document work needs it, which is where the
same connector looked like it had regressed and had not changed at all.

Read across the rows for the second, separate effect. Content density degrades
every connector -- full projection falls from {full[0]:.3f} to {full[-1]:.3f} as
facts go from 4 to 1024 -- because facts start sharing patches and superpose
within a fixed-dimension feature. That is a SECOND ceiling, belonging to the patch
representation rather than the connector, and it is why more visual tokens
eventually stop helping even with no connector at all.

The two ceilings compound unevenly, which is the practically important part. Over
the same sweep the T=8 connector falls {(1 - q8[-1]/q8[0]) * 100:.0f}% while full
projection falls {(1 - full[-1]/full[0]) * 100:.0f}%. Compression does not merely
subtract a constant; it makes the system more fragile to exactly the density that
makes compression tempting (eq:compounding-ceilings).

The pooling row is the control. It compresses 64 patches to 8 tokens by a
completely different mechanism -- averaging neighbours rather than learned global
queries -- and lands at {pool8[0]:.3f} against the query connector's {q8[0]:.3f}.
Essentially the same. What sets the ceiling is the token count, and how you get
there changes the constant rather than the shape.

So the design question is not which connector is more elegant. It is: how many
distinct things must one image convey, and can I afford that many tokens? A
caption needs few, a spreadsheet needs many, and that number -- not FLOPs and not
architecture -- is what should set T. cite:liu2023llava's plain linear projection
won for documents not because it is cleverer but because it declines to answer
the question in advance.""")
