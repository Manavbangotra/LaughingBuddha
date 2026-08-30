# -*- coding: utf-8 -*-
# Extracted from: Chapter 203 — Containers, Kubernetes, and Autoscaling
# Source: src/.../ch203-kubernetes.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Where a cold start's minutes actually go, and which of them are avoidable.

The previous listing treated cold start as one number and showed it dominates the
scaling decision. This one opens it up, because the components have very different
costs to remove and a team that attacks the wrong one spends a quarter for nothing
(eq:cold-start-is-mostly-weight-movement).

The finding is that the dominant term is moving weights, that its size is set by where
the weights are stored rather than by anything about the model, and that the standard
container-registry path is the worst available choice by a wide margin.
"""
WEIGHTS_GB = 140.0
# (stage, seconds, whether it scales with model size)
STAGES = [
    ("schedule pod",            8.0,  False),
    ("pull container image",   34.0,  False),
    ("start runtime",          11.0,  False),
    ("fetch weights",           0.0,  True),
    ("load to device memory",   0.0,  True),
    ("capture graphs",         19.0,  False),
    ("warm up and health check", 12.0, False),
]

# Where weights can come from. (source, GB/s achieved)
SOURCES = [
    ("container image layer",  0.35),
    ("object storage",         1.10),
    ("network filesystem",     2.40),
    ("local NVMe",             6.80),
    ("host page cache",       21.00),
]
PCIE_GB_S = 55.0            # host to device


def fetch_s(gb, src_gb_s):
    return gb / src_gb_s


def load_s(gb):
    return gb / PCIE_GB_S


print("Cold start for a %.0f GB model, by stage." % WEIGHTS_GB)
print()
for src, rate in SOURCES:
    total = 0.0
    for name, secs, scales in STAGES:
        if name == "fetch weights":
            secs = fetch_s(WEIGHTS_GB, rate)
        elif name == "load to device memory":
            secs = load_s(WEIGHTS_GB)
        total += secs
    print(f"  weights from {src:<24} {total:>7.1f}s")

print()
print("Breaking down the middle case (object storage):")
print()
print(f"{'stage':>26}{'seconds':>10}{'share':>9}{'scales with model':>20}")
print("-" * 66)
BASE_SRC = 1.10
detail = {}
total = 0.0
for name, secs, scales in STAGES:
    if name == "fetch weights":
        secs = fetch_s(WEIGHTS_GB, BASE_SRC)
    elif name == "load to device memory":
        secs = load_s(WEIGHTS_GB)
    detail[name] = secs
    total += secs
for name, secs, scales in STAGES:
    s = detail[name]
    print(f"{name:>26}{s:>10.1f}{s / total:>9.1%}"
          f"{('yes' if scales else 'no'):>20}")
print("-" * 66)
print(f"{'TOTAL':>26}{total:>10.1f}{1.0:>9.1%}")

print()
print()
print("Weight movement by source. This is the term that dominates and the one")
print("with the widest spread.")
print()
print(f"{'source':>24}{'GB/s':>9}{'fetch s':>10}{'load s':>9}"
      f"{'weight total':>15}{'cold start':>13}")
print("-" * 82)
fixed = sum(s for n, s, sc in STAGES if n not in
            ("fetch weights", "load to device memory"))
bysrc = {}
for src, rate in SOURCES:
    f = fetch_s(WEIGHTS_GB, rate)
    l = load_s(WEIGHTS_GB)
    bysrc[src] = (f, l, f + l, fixed + f + l)
    print(f"{src:>24}{rate:>9.2f}{f:>10.1f}{l:>9.1f}{f + l:>15.1f}"
          f"{fixed + f + l:>13.1f}")

print()
print(f"fixed overhead independent of source: {fixed:.1f}s")

print()
print()
print("By model size, from the best and worst sources.")
print()
print(f"{'model GB':>10}" + "".join(f"{s[0][:14]:>16}" for s in SOURCES))
print("-" * 90)
grid = {}
for gb in (1.5, 14.0, 40.0, 140.0, 400.0):
    row = []
    for src, rate in SOURCES:
        row.append(fixed + fetch_s(gb, rate) + load_s(gb))
    grid[gb] = row
    print(f"{gb:>10.1f}" + "".join(f"{v:>16.1f}" for v in row))
print()
print("(cold start seconds)")

print()
print()
print("What each intervention removes, ranked by seconds bought.")
print()
print(f"{'intervention':>36}{'removes':>10}{'new cold start':>17}"
      f"{'speedup':>10}")
print("-" * 74)
base_total = bysrc["object storage"][3]
INTERVENTIONS = [
    ("bake weights into the image", -1),
    ("pre-pull image to every node", detail["pull container image"]),
    ("skip graph capture", detail["capture graphs"]),
    ("weights on local NVMe", -2),
    ("weights in host page cache", -3),
    ("keep a warm spare", base_total),
]
for label, removed in INTERVENTIONS:
    if removed == -1:
        # Image layer is the slowest source AND makes the pull enormous.
        f = fetch_s(WEIGHTS_GB, 0.35)
        new = fixed + f + load_s(WEIGHTS_GB)
    elif removed == -2:
        new = bysrc["local NVMe"][3]
    elif removed == -3:
        new = bysrc["host page cache"][3]
    elif removed == base_total:
        new = 0.0
    else:
        new = base_total - removed
    sp = (base_total / new) if new > 0 else float("inf")
    print(f"{label:>36}{(base_total - new):>10.1f}{new:>17.1f}"
          f"{('inf' if new == 0 else '%.2fx' % sp):>10}")

print()
print()
print("And what the cold start implies for the headroom from the previous")
print("listing, at a realistic ramp.")
print()
RAMP = 0.0023
print(f"{'weight source':>24}{'cold start s':>14}{'growth over it':>17}"
      f"{'trigger at':>13}{'idle fleet':>13}")
print("-" * 82)
for src, rate in SOURCES:
    cs = bysrc[src][3]
    growth = (1.0 + RAMP) ** cs
    print(f"{src:>24}{cs:>14.1f}{growth:>16.1f}x{1.0 / growth:>13.0%}"
          f"{1.0 - 1.0 / growth:>13.0%}")

print(f"""
The stage breakdown is the first surprise. Of a {total:.0f}-second cold start from
object storage, **{detail['fetch weights'] / total:.0%} is fetching weights** and
{detail['load to device memory'] / total:.0%} is moving them onto the device
(eq:cold-start-is-mostly-weight-movement). Everything a platform team normally
optimises -- pod scheduling, image pull, runtime start -- is
{(detail['schedule pod'] + detail['pull container image'] + detail['start runtime']) / total:.0%}
between them.

The source table is why that matters. Fetching {WEIGHTS_GB:.0f} GB takes
{bysrc['container image layer'][0]:.0f} seconds from a container image layer and
{bysrc['host page cache'][0]:.0f} seconds from the host page cache -- a spread of
{bysrc['container image layer'][0] / bysrc['host page cache'][0]:.0f} times, on the term
that is most of the total.

**Where the weights live is the cold-start decision.** Not the image size, not the
scheduler, not the runtime -- those sum to {fixed:.0f} seconds and do not move.

The intervention table ranks the options by what they actually buy. Baking weights into
the container image is the intuitive move and it is **the worst available choice**: a
container layer delivers {0.35:.2f} GB/s, so it takes the cold start to
{fixed + fetch_s(WEIGHTS_GB, 0.35) + load_s(WEIGHTS_GB):.0f} seconds against object
storage's {base_total:.0f}.

That is worth stating plainly because it is a common instinct. Putting the weights in
the image feels like removing a fetch, and it does -- by moving the same bytes through a
slower path, decompressed layer by layer, on the critical path of every pod start.

Moving weights to local NVMe takes the cold start to
{bysrc['local NVMe'][3]:.0f} seconds; the host page cache takes it to
{bysrc['host page cache'][3]:.0f}. Those are the interventions worth funding, and both
are storage-placement decisions rather than serving ones.

The model-size grid shows the scaling. A {1.5:.1f} GB model cold-starts in
{grid[1.5][1]:.0f} seconds from object storage and a {400.0:.0f} GB one in
{grid[400.0][1]:.0f} -- so **large models do not merely cost more to serve, they cost
more to scale**, and the autoscaling problem from the previous listing gets
proportionally harder with model size.

The last table closes the loop with that listing. At a ramp that doubles load every
{0.693 / RAMP / 60:.0f} minutes, a cold start from the container image forces a trigger
at
{1.0 / ((1 + RAMP) ** bysrc['container image layer'][3]):.0%} of capacity -- meaning
essentially the entire fleet must sit idle. From the host page cache the trigger can sit
at {1.0 / ((1 + RAMP) ** bysrc['host page cache'][3]):.0%}, implying
{1.0 - 1.0 / ((1 + RAMP) ** bysrc['host page cache'][3]):.0%} idle.

**Weight placement and fleet utilisation are the same decision.** A team that cannot
explain why its GPU fleet runs at forty percent should look at where its weights are
stored before it looks at its autoscaler, because the second number is downstream of the
first.""")
