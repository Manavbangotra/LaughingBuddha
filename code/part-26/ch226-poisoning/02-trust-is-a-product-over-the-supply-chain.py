# -*- coding: utf-8 -*-
# Extracted from: Chapter 226 — Data Poisoning, RAG Poisoning, and Supply-Chain Attacks
# Source: src/.../ch226-poisoning.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Trust in an AI system is a product over its supply chain, and one unsigned link zeroes it.

A deployed system is a chain: base weights from one party, a fine-tune from another, an
adapter, an embedding model, a retrieval corpus, a set of tool servers, a package tree. Each
link is trusted or it is not, and the composite trust is the product
(eq:trust-is-a-product-over-the-supply-chain).

Verification helps and it has a hard edge: it covers what is signed and attests to what the
signature actually claims, which is usually "this is the artefact the publisher published"
and not "this artefact is not malicious"
(eq:provenance-covers-only-signed-links).

This listing computes the composite, prices detection methods that do not need provenance,
and shows which link is worth fixing.
"""
# (link, P(this link is clean), can it be signed?, cost to verify)
CHAIN = [
    ("base model weights",         0.985, True,  0.5),
    ("pretraining corpus",         0.870, False, 0.0),
    ("fine-tuning dataset",        0.940, True,  2.0),
    ("adapter or LoRA weights",    0.975, True,  0.4),
    ("embedding model",            0.990, True,  0.3),
    ("retrieval corpus",           0.820, False, 3.0),
    ("tool server endpoints",      0.910, True,  1.5),
    ("package dependency tree",    0.960, True,  1.0),
]

print("Trust as a product over the chain.")
print()
print(f"{'link':>30}{'P(clean)':>11}{'signable':>11}{'cumulative':>13}"
      f"{'contribution to loss':>23}")
print("-" * 88)
cum = 1.0
contrib = {}
for name, p, sign, cost in CHAIN:
    prev = cum
    cum *= p
    contrib[name] = (p, sign, cost, prev - cum)
    print(f"{name:>30}{p:>11.3f}{('yes' if sign else 'no'):>11}"
          f"{cum:>13.4f}{prev - cum:>23.4f}")
print("-" * 88)
print(f"{'COMPOSITE':>30}{'':>11}{'':>11}{cum:>13.4f}")

print()
print(f"eight links averaging {sum(p for n, p, s, c in CHAIN) / len(CHAIN):.3f} "
      f"compose to {cum:.4f}")
print(f"P(something in the chain is compromised) = {1 - cum:.1%}")

print()
print()
print("What verifying the signable links buys.")
print()
VERIFY_LIFT = 0.65        # signing raises P(clean) toward 1 by this fraction of the gap
print(f"{'link':>30}{'P before':>11}{'P after':>10}{'cost':>8}"
      f"{'gain per cost':>16}")
print("-" * 75)
lift = {}
for name, p, sign, cost in CHAIN:
    if not sign:
        lift[name] = (p, p, cost, 0.0)
        print(f"{name:>30}{p:>11.3f}{p:>10.3f}{'--':>8}{'--':>16}")
        continue
    after = p + VERIFY_LIFT * (1.0 - p)
    lift[name] = (p, after, cost, (after - p) / cost)
    print(f"{name:>30}{p:>11.3f}{after:>10.3f}{cost:>8.1f}"
          f"{(after - p) / cost:>16.4f}")

signed_cum = 1.0
for name, p, sign, cost in CHAIN:
    signed_cum *= lift[name][1]
print()
print(f"composite after verifying every signable link: {signed_cum:.4f}")
print(f"compromise probability falls from {1 - cum:.1%} to {1 - signed_cum:.1%}")

unsigned = [n for n, p, s, c in CHAIN if not s]
unsigned_p = 1.0
for n in unsigned:
    unsigned_p *= dict((x[0], x[1]) for x in CHAIN)[n]
print()
print(f"the two unsignable links alone contribute {1 - unsigned_p:.1%}")
print(f"which is {(1 - unsigned_p) / (1 - signed_cum):.0%} of the residual")

print()
print()
print("What a signature actually attests to.")
print()
ATTESTS = [
    ("model weight signature",  "this is what the publisher published", "no"),
    ("package lockfile hash",   "this is the version you resolved",     "no"),
    ("SBOM entry",              "this component was included",          "no"),
    ("build provenance",        "this came from that source and CI",    "partly"),
    ("reproducible build",      "the source produces these bytes",      "partly"),
    ("corpus manifest hash",    "this is the corpus you indexed",       "no"),
]
print(f"{'artefact':>28}{'what it attests':>42}"
      f"{'says it is safe?':>19}")
print("-" * 89)
for name, says, safe in ATTESTS:
    print(f"{name:>28}{says:>42}{safe:>19}")

print()
print("Every row answers 'is this the thing you meant to get'. None answers")
print("'is the thing you meant to get malicious'.")

print()
print()
print("Detection methods, for the links provenance cannot reach.")
print()
DETECT = [
    ("held-out clean evaluation",   0.14, 0.71, 1.2),
    ("loss anomaly during training", 0.09, 0.44, 0.8),
    ("activation clustering",        0.31, 0.22, 4.0),
    ("nearest-neighbour corpus audit", 0.26, 0.18, 2.5),
    ("canary trigger probes",        0.55, 0.06, 3.0),
    ("output monitoring in production", 0.19, 0.62, 2.0),
]
print(f"{'method':>34}{'catches targeted':>19}{'catches broad':>16}"
      f"{'effort':>9}{'per effort':>13}")
print("-" * 91)
det = {}
for name, targ, broad, eff in DETECT:
    combined = 0.75 * targ + 0.25 * broad
    det[name] = (targ, broad, eff, combined / eff)
    print(f"{name:>34}{targ:>19.0%}{broad:>16.0%}{eff:>9.1f}"
          f"{combined / eff:>13.3f}")

print()
print("Weighted 3:1 toward targeted attacks, because those are the cheap ones.")

best_det = max(det, key=lambda n: det[n][3])
print(f"best: {best_det} at {det[best_det][3]:.3f}")

print()
print()
print("Composite miss rate against the two attack classes.")
print()
miss_t, miss_b = 1.0, 1.0
for name, targ, broad, eff in DETECT:
    miss_t *= (1 - targ)
    miss_b *= (1 - broad)
print(f"{'attack class':>24}{'all six methods miss':>24}"
      f"{'cost of one attack':>22}{'expected loss':>16}")
print("-" * 86)
for label, miss, dmg in (("targeted backdoor", miss_t, 9.0),
                         ("broad degradation", miss_b, 8.0)):
    print(f"{label:>24}{miss:>24.1%}{dmg:>22.1f}{miss * dmg:>16.2f}")

print()
print()
print("And the ranking that follows: fix the link, do not chase the poison.")
print()
ACTIONS = [
    ("verify every signable link",      1 - signed_cum, 5.7),
    ("re-host the retrieval corpus",    0.11,           4.0),
    ("train on a curated corpus only",  0.13,           30.0),
    ("run all six detection methods",   0.06,           13.5),
    ("pin and re-verify tool servers",  0.05,           1.5),
]
print(f"{'action':>34}{'compromise probability after':>31}{'effort':>9}"
      f"{'reduction per effort':>23}")
print("-" * 97)
act = {}
for name, after, eff in ACTIONS:
    red = max(0.0, (1 - cum) - after)
    act[name] = (after, eff, red / eff)
    print(f"{name:>34}{after:>31.1%}{eff:>9.1f}{red / eff:>23.4f}")
best_act = max(act, key=lambda n: act[n][2])

print(f"""
The chain table is the arithmetic that should open every AI supply-chain conversation. Eight
links, each individually respectable -- the lowest is {min(p for n, p, s, c in CHAIN):.3f} --
compose to {cum:.4f} (eq:trust-is-a-product-over-the-supply-chain).

**A {1 - cum:.1%} chance that something in the chain is compromised**, from links that would
each pass a review. That is ch:ops-versioning's product-over-artefacts result with a different
quantity in the product, and it has the same shape: the composite is dominated by the weakest
links and no single link's improvement rescues it.

The contribution column names them. `{CHAIN[5][0]}` and `{CHAIN[1][0]}` contribute
{contrib[CHAIN[5][0]][3] + contrib[CHAIN[1][0]][3]:.4f} of the loss between them, and both are
in the `signable: no` column.

The verification table is the good news and its limit. Signing every signable link takes the
composite from {cum:.4f} to {signed_cum:.4f} -- compromise probability from
{1 - cum:.1%} to {1 - signed_cum:.1%} -- for {sum(c for n, p, s, c in CHAIN if s):.1f} units
of effort.

Then read the line under it. **The two unsignable links alone contribute
{1 - unsigned_p:.1%}**, which is {(1 - unsigned_p) / (1 - signed_cum):.0%} of what remains
(eq:provenance-covers-only-signed-links). Verification is worth doing and it converges to a
floor set by the links nobody can sign: a pretraining corpus you did not assemble and a
retrieval corpus that changes every day.

The attestation table is the part most often over-read. Every row answers **"is this the thing
you meant to get?"** A model signature says the publisher published these bytes. A lockfile
hash says this is the version you resolved. An SBOM entry says the component was included.

**None of them says the thing you meant to get is not malicious.** Build provenance and
reproducible builds get partway -- they tie the artefact to a source and a process -- and even
they attest to origin rather than to behaviour. Provenance answers a substitution question, and
poisoning is not a substitution attack.

The detection table is what is left for the unsignable links, and the numbers are modest.
`{best_det}` returns {det[best_det][3]:.3f} per unit of effort, weighted three-to-one toward
targeted attacks because ch:sec-poisoning's first listing showed those are the cheap ones.

Notice the shape of that table. Methods that catch *broad* degradation --
`held-out clean evaluation` at {DETECT[0][2]:.0%}, `output monitoring` at
{DETECT[5][2]:.0%} -- are cheap and effective. Methods that catch *targeted* backdoors are
expensive and weak, except canary probes, which catch {DETECT[4][1]:.0%} of the triggers **you
thought of**.

The composite miss table is the honest summary. All six methods together miss
{miss_t:.1%} of targeted backdoors and {miss_b:.1%} of broad degradation. The cheap attack is
the one that gets through.

Which produces the ranking in the last table and the recommendation of this chapter.
**Fix the link, do not chase the poison.** `{best_act}` returns
{act[best_act][2]:.4f} of compromise-probability reduction per unit of effort, against
{act['run all six detection methods'][2]:.4f} for running every detector in the literature --
{act[best_act][2] / act['run all six detection methods'][2]:.0f} times better.

Re-hosting the retrieval corpus -- taking a corpus you do not control and making it one you do
-- is the single largest move available, because it converts an unsignable link into a signable
one.

Detection is the control you build for the links you cannot re-host, and there are always
some. It is a residual measure and it should be budgeted as one.""")
