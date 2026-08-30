# -*- coding: utf-8 -*-
# Extracted from: Chapter 129 — When to Fine-Tune and When Not To
# Source: src/.../ch129-when-to-fine-tune.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Fine-tuning teaches format reliably and facts unreliably. The same run shows both.

ch:fm-what-they-are argued that adaptation carries far less information than
pretraining, so a fine-tune cannot install much that is new. This listing makes
the distinction sharper than "less": it separates what fine-tuning teaches into
two kinds and shows they behave completely differently on the SAME training run.

  FORMAT   a rule that applies to every input -- answer in JSON, always cite,
           refuse this category. Depends on the request TYPE, not its content, so
           one example teaches it for all inputs (eq:format-generalises).
  FACTS    a mapping from a specific key to a specific value. There is no rule to
           learn; each pair must be memorised, and memorising one says nothing
           about the next (eq:facts-do-not-generalise).

A tiny model is trained on inputs carrying both, and evaluated on keys it saw and
keys it did not.
"""
import numpy as np

rng = np.random.default_rng(131)

N_KEY, N_TYPE = 400, 6
N_FORMAT, N_VALUE = 4, 25
D_EMB, D_HID = 24, 64
EPOCHS, LR = 60, 0.08

# The two targets. FORMAT depends only on the request type; VALUE depends only
# on the key, and is an arbitrary lookup with no structure to generalise from.
format_of_type = rng.integers(0, N_FORMAT, size=N_TYPE)
value_of_key = rng.integers(0, N_VALUE, size=N_KEY)

# Keys the model will be trained on, and keys held out entirely.
perm = rng.permutation(N_KEY)
SEEN, UNSEEN = perm[:N_KEY // 2], perm[N_KEY // 2:]


def make_batch(keys, n):
    k = rng.choice(keys, size=n)
    t = rng.integers(0, N_TYPE, size=n)
    return k, t, format_of_type[t], value_of_key[k]


class Model:
    """Key embedding + type embedding -> hidden -> two heads. The only way to
    get VALUE right is to store it in the key embedding; FORMAT can be read off
    the type embedding alone."""

    def __init__(self):
        self.Ek = rng.normal(scale=0.3, size=(N_KEY, D_EMB))
        self.Et = rng.normal(scale=0.3, size=(N_TYPE, D_EMB))
        self.W1 = rng.normal(scale=np.sqrt(2 / (2 * D_EMB)), size=(2 * D_EMB, D_HID))
        self.b1 = np.zeros(D_HID)
        self.Wf = rng.normal(scale=np.sqrt(2 / D_HID), size=(D_HID, N_FORMAT))
        self.bf = np.zeros(N_FORMAT)
        self.Wv = rng.normal(scale=np.sqrt(2 / D_HID), size=(D_HID, N_VALUE))
        self.bv = np.zeros(N_VALUE)

    def forward(self, k, t):
        self.k, self.t = k, t
        self.x = np.hstack([self.Ek[k], self.Et[t]])
        self.h = np.maximum(self.x @ self.W1 + self.b1, 0)
        return self.h @ self.Wf + self.bf, self.h @ self.Wv + self.bv

    def step(self, gf, gv, lr):
        gWf, gbf = self.h.T @ gf, gf.sum(0)
        gWv, gbv = self.h.T @ gv, gv.sum(0)
        gh = (gf @ self.Wf.T + gv @ self.Wv.T) * (self.h > 0)
        gW1, gb1 = self.x.T @ gh, gh.sum(0)
        gx = gh @ self.W1.T
        np.add.at(self.Ek, self.k, -lr * gx[:, :D_EMB])
        np.add.at(self.Et, self.t, -lr * gx[:, D_EMB:])
        for p, g in ((self.W1, gW1), (self.b1, gb1), (self.Wf, gWf),
                     (self.bf, gbf), (self.Wv, gWv), (self.bv, gbv)):
            p -= lr * g


def ce(logits, y):
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    g = p.copy(); g[np.arange(len(y)), y] -= 1
    return g / len(y)


def evaluate(m, keys):
    k, t, yf, yv = make_batch(keys, 4000)
    lf, lv = m.forward(k, t)
    return float((lf.argmax(1) == yf).mean()), float((lv.argmax(1) == yv).mean())


print(f"{N_KEY // 2} keys seen in training, {N_KEY // 2} never seen.")
print(f"FORMAT is a function of the request type ({N_TYPE} types, "
      f"{N_FORMAT} formats).")
print(f"FACT is a function of the key ({N_KEY} keys, {N_VALUE} values, "
      f"chance = {1/N_VALUE:.3f}).\n")
print(f"{'examples per seen key':>23}{'':>3}{'FORMAT':>18}{'':>4}{'FACT':>18}")
print(f"{'':>23}{'':>3}{'seen':>9}{'unseen':>9}{'':>4}{'seen':>9}{'unseen':>9}")
print("-" * 76)

rows = {}
for per_key in (1, 4, 16, 64, 256):
    m = Model()
    n_batch = max(per_key * len(SEEN) // EPOCHS, 32)
    for _ in range(EPOCHS):
        for _ in range(max(n_batch // 128, 1)):
            k, t, yf, yv = make_batch(SEEN, 128)
            lf, lv = m.forward(k, t)
            m.step(ce(lf, yf), ce(lv, yv), LR)
    fs, vs = evaluate(m, SEEN)
    fu, vu = evaluate(m, UNSEEN)
    rows[per_key] = (fs, fu, vs, vu)
    print(f"{per_key:>23}{'':>3}{fs:>9.3f}{fu:>9.3f}{'':>4}{vs:>9.3f}{vu:>9.3f}")

lo, hi = rows[1], rows[256]
print(f"""
Read the two FORMAT columns first, and read them together. They are
approximately equal at every training budget -- {hi[0]:.3f} on keys the model
trained on and {hi[1]:.3f} on keys it has never seen. The format rule transferred
completely to inputs that were not in the training set, because there was a rule
to learn: format depends on the request type, and the types were all covered
(eq:format-generalises).

That is what makes format cheap. Once the model has seen each request type a few
times, it has the rule, and the rule applies to every future input regardless of
content. This is why instruction tuning works on a thousand examples
(cite:zhou2023lima) and why "always answer in this shape" is the thing
fine-tuning is genuinely good at.

Now the FACT columns, which behave in the opposite way. On seen keys, accuracy
climbs with repetition -- {lo[2]:.3f} at one example per key up to {hi[2]:.3f} at
256 -- which is memorisation working as memorisation does. On UNSEEN keys it sits
at {hi[3]:.3f} against a chance rate of {1/N_VALUE:.3f} and does not move at any
budget.

Not "worse". Not "needs more data". Flat at chance, permanently, because there is
nothing to generalise (eq:facts-do-not-generalise). The mapping from key to value
is arbitrary by construction -- as facts about your customers, your inventory or
your policies are arbitrary with respect to each other -- so learning four hundred
of them tells the model nothing whatsoever about the four hundred and first.

Those two behaviours from one training run are the whole argument for the
architecture of the previous two parts. If a capability is a RULE, fine-tuning
installs it cheaply and it generalises. If it is a FACT, fine-tuning memorises the
ones you showed it, generalises to none, and every new fact requires another
training run -- which is exactly the churn term that this chapter's first listing
shows decides the economics.

Retrieval has the complementary shape: it is poor at changing behaviour and
excellent at supplying a fact that was never in the weights, including one created
this morning. So the two are not competitors to be ranked. They address the two
halves of this table, and a system that needs both should use both -- fine-tune
the format, retrieve the facts.""")
