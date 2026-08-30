# -*- coding: utf-8 -*-
# Extracted from: Chapter 204 — Cloud, Edge, and Local Deployment
# Source: src/.../ch204-cloud-edge-local.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""On a device, the constraint is bandwidth and heat -- not capacity, and not FLOPs.

The usual question about on-device inference is "does the model fit". That is the wrong
first question. ch:inf-cpu-gpu established decode is bandwidth-bound, and a phone's
memory bandwidth is two orders of magnitude below a datacentre GPU's while its capacity
is only one order below.

So a model that fits comfortably can still be unusably slow, and the achievable quality
at a given speed is set by bandwidth rather than by memory
(eq:device-quality-is-bandwidth-bound).

There is a second constraint with no datacentre analogue: sustained throughput is
limited by heat, so the number a benchmark reports is not the number a user gets after
the first minute.
"""
# (device, memory GB, bandwidth GB/s, sustained fraction of burst after thermals)
DEVICES = [
    ("phone, mid",        6.0,   34.0, 0.42),
    ("phone, flagship",  12.0,   68.0, 0.51),
    ("laptop, integrated", 24.0, 120.0, 0.63),
    ("laptop, discrete",  16.0,  480.0, 0.78),
    ("workstation GPU",   48.0,  960.0, 0.95),
]
# (model, params B, bytes per param at the quantisation that fits)
MODELS = [
    ("0.5B int4",   0.5, 0.5),
    ("3B int4",     3.0, 0.5),
    ("8B int4",     8.0, 0.5),
    ("8B fp16",     8.0, 2.0),
    ("32B int4",   32.0, 0.5),
]
OVERHEAD_GB = 1.4          # runtime, KV cache, the rest of the operating system
READ_TARGET = 12.0         # tokens/sec a person finds tolerable to read


def weights_gb(params_b, bytes_per):
    return params_b * bytes_per


def burst_tps(params_b, bytes_per, bw_gb_s):
    """Decode tokens/sec: bandwidth over bytes read per token."""
    return bw_gb_s / weights_gb(params_b, bytes_per)


print("Devices, and what each can hold and move.")
print()
print(f"{'device':>20}{'memory GB':>12}{'GB/s':>9}{'sustained':>12}"
      f"{'bandwidth per GB':>19}")
print("-" * 74)
for name, mem, bw, sus in DEVICES:
    print(f"{name:>20}{mem:>12.1f}{bw:>9.0f}{sus:>12.0%}{bw / mem:>19.1f}")

print()
print("A datacentre GPU is 80 GB at 3350 GB/s -- 42 GB/s per GB of memory.")
print("Note that column: devices are memory-rich relative to their bandwidth.")

print()
print()
print("What fits, by model and device. Capacity is the question people ask.")
print()
print(f"{'model':>12}{'weights GB':>13}" +
      "".join(f"{d[0][:12]:>14}" for d in DEVICES))
print("-" * 96)
fits = {}
for m, p, b in MODELS:
    w = weights_gb(p, b)
    row = {}
    cells = ""
    for name, mem, bw, sus in DEVICES:
        ok = (w + OVERHEAD_GB) <= mem
        row[name] = ok
        cells += f"{('fits' if ok else 'no'):>14}"
    fits[m] = row
    print(f"{m:>12}{w:>13.1f}{cells}")

print()
print()
print("What it RUNS at, which is the question that decides the product.")
print("Sustained tokens/sec after thermal throttling.")
print()
print(f"{'model':>12}" + "".join(f"{d[0][:12]:>14}" for d in DEVICES))
print("-" * 84)
speed = {}
for m, p, b in MODELS:
    row = {}
    cells = ""
    for name, mem, bw, sus in DEVICES:
        if not fits[m][name]:
            row[name] = 0.0
            cells += f"{'-':>14}"
            continue
        t = burst_tps(p, b, bw) * sus
        row[name] = t
        cells += f"{t:>14.1f}"
    speed[m] = row
    print(f"{m:>12}{cells}")

print()
print()
print("The largest model each device can run AT READING SPEED (%.0f tok/s)."
      % READ_TARGET)
print()
print(f"{'device':>20}{'largest that fits':>20}{'largest that is usable':>25}"
      f"{'gap':>18}")
print("-" * 84)
usable = {}
for name, mem, bw, sus in DEVICES:
    big_fit = None
    big_use = None
    for m, p, b in MODELS:
        if fits[m][name]:
            big_fit = m
        if fits[m][name] and speed[m][name] >= READ_TARGET:
            big_use = m
    usable[name] = (big_fit, big_use)
    gap = "same" if big_fit == big_use else "smaller"
    print(f"{name:>20}{str(big_fit):>20}{str(big_use or 'none'):>25}{gap:>18}")

print()
print()
print("Burst against sustained. A benchmark measures the first column; a user")
print("after sixty seconds experiences the second.")
print()
print(f"{'device':>20}{'burst tok/s':>14}{'sustained':>12}{'drop':>9}"
      f"{'burst usable?':>16}{'sustained usable?':>20}")
print("-" * 92)
TESTM = "3B int4"
p, b = 3.0, 0.5
therm = {}
for name, mem, bw, sus in DEVICES:
    burst = burst_tps(p, b, bw)
    sust = burst * sus
    therm[name] = (burst, sust, 1.0 - sus)
    print(f"{name:>20}{burst:>14.1f}{sust:>12.1f}{1.0 - sus:>8.0%}"
          f"{('yes' if burst >= READ_TARGET else 'no'):>16}"
          f"{('yes' if sust >= READ_TARGET else 'no'):>20}")

print()
print()
print("And the bandwidth a device would need to run each model at reading speed.")
print()
print(f"{'model':>12}{'GB read/token':>16}{'GB/s needed':>14}"
      f"{'cheapest device that has it':>32}")
print("-" * 76)
need = {}
for m, p, b in MODELS:
    w = weights_gb(p, b)
    req = w * READ_TARGET
    who = [d[0] for d in DEVICES if d[1] * d[3] >= 0 and d[2] * d[3] >= req]
    need[m] = (w, req, who[0] if who else None)
    print(f"{m:>12}{w:>16.2f}{req:>14.0f}"
          f"{(who[0] if who else 'none in this table'):>32}")

print(f"""
The device table has the number that reframes the problem. A datacentre GPU offers
{3350.0 / 80.0:.0f} GB/s per gigabyte of memory. A flagship phone offers
{68.0 / 12.0:.1f}; a mid-range one {34.0 / 6.0:.1f}.

**Devices are memory-rich relative to their bandwidth**, which is the opposite of the
datacentre balance and it inverts which constraint binds
(eq:device-quality-is-bandwidth-bound).

The fits table is the question people ask, and it is close to uninformative. An
{8.0:.0f}B model at int4 is {weights_gb(8.0, 0.5):.1f} GB and fits on
{sum(1 for d in DEVICES if fits['8B int4'][d[0]])} of the
{len(DEVICES)} devices. The capacity question has an encouraging answer nearly
everywhere.

The speed table is the question that decides the product. That same
{8.0:.0f}B int4 model runs at {speed['8B int4']['phone, flagship']:.1f} tokens a second
on a flagship phone -- it fits, comfortably, and it is
{READ_TARGET / speed['8B int4']['phone, flagship']:.1f} times slower than reading speed.

**Fitting and running are different questions with different answers**, and the second
one is not asked nearly often enough.

The usable-model table states the gap directly. A flagship phone can hold
`{usable['phone, flagship'][0]}` and can usefully run
`{usable['phone, flagship'][1]}`. A mid-range phone holds
`{usable['phone, mid'][0]}` and usefully runs `{usable['phone, mid'][1]}`.

That is a product decision disguised as a hardware fact. **The model you can ship
on-device is set by bandwidth, and it is roughly one size class below what fits.**

The thermal table adds a constraint with no datacentre analogue. A phone cannot sustain
its burst clocks: the flagship holds {DEVICES[1][3]:.0%} of burst and the mid-range
{DEVICES[0][3]:.0%}, against a workstation GPU's {DEVICES[4][3]:.0%}.

For a {TESTM} model that is {therm['phone, mid'][0]:.1f} tokens a second in burst and
{therm['phone, mid'][1]:.1f} sustained on a mid-range phone -- and the burst number is
what a benchmark reports, because a benchmark runs for seconds.

**A device benchmark and a device experience differ by the thermal fraction**, and the
difference is largest on the smallest devices, which is exactly where the margin was
thinnest to begin with.

The requirement table gives the design rule. Running a model at reading speed needs
bandwidth equal to its weight bytes times the token rate: {need['3B int4'][1]:.0f} GB/s
for a {3.0:.0f}B int4 model, {need['8B int4'][1]:.0f} for an
{8.0:.0f}B, and {need['32B int4'][1]:.0f} for a {32.0:.0f}B.

Those are the numbers to check against a target device before committing to a model
size, and they explain the industry's convergence on small quantised models for
on-device work. It is not that larger models do not fit. **It is that bandwidth per
gigabyte on a phone is {(3350.0 / 80.0) / (68.0 / 12.0):.0f} times lower than a
datacentre GPU's**, so the model that runs at an acceptable speed is a size class or two
below the one that fits -- and quantisation helps
twice, by shrinking both the footprint and the per-token read.""")
