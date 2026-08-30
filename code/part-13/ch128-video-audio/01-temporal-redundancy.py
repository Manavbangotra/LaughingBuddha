# -*- coding: utf-8 -*-
# Extracted from: Chapter 128 — Video, Audio, and Spatial Reasoning
# Source: src/.../ch128-video-audio.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""How many frames? The answer is set by the shortest thing you must notice.

cite:tong2022videomae masks 90-95% of a video and still learns from it, which is
the quantitative statement of how little independent information a frame adds:
adjacent frames are nearly identical, so a video is far less information than its
frame count suggests (eq:temporal-redundancy).

The practical question is what sampling rate to use, and the usual answers ("one
frame per second", "eight frames per clip") are stated as though they were
properties of video. They are not. They are properties of the TASK -- specifically
of the shortest event that must be caught (eq:event-catch-probability).

This listing measures two task types on the same videos and finds their frame
requirements differ by more than an order of magnitude.
"""
import numpy as np

rng = np.random.default_rng(113)

DURATION = 60.0            # seconds of video
N_VIDEO = 8000
TAU = 6.0                  # seconds over which the scene meaningfully changes


def sample_times(n):
    """Uniform sampling, the standard scheme."""
    return (np.arange(n) + 0.5) * DURATION / n


def action_accuracy(n, trials=N_VIDEO):
    """A sustained action fills the video. Each sampled frame is a noisy view of
    one persistent latent state, and frames decorrelate over TAU seconds -- so
    extra frames within one TAU add almost nothing (eq:effective-samples)."""
    t = sample_times(n)
    # Effective independent samples: a frame counts fully only to the extent it
    # has decorrelated from the previous one. With one frame there are no gaps,
    # so n_eff is exactly 1.
    if n == 1:
        n_eff = 1.0
    else:
        rho = np.exp(-np.diff(t) / TAU)
        n_eff = 1.0 + float(np.sum(1.0 - rho))
    # Classification accuracy improves with the square root of effective samples.
    return float(1.0 - 0.5 * np.exp(-0.9 * np.sqrt(n_eff))), n_eff


def event_accuracy(n, event_s):
    """A brief event happens once, at a uniformly random time. It is caught only
    if a sampled frame falls inside its window."""
    t = sample_times(n)
    hits = 0
    for _ in range(N_VIDEO):
        start = rng.uniform(0.0, DURATION - event_s)
        hits += int(((t >= start) & (t <= start + event_s)).any())
    return hits / N_VIDEO


FRAMES = (1, 2, 4, 8, 16, 32, 64, 128, 256)

print(f"{DURATION:.0f}-second video; the scene decorrelates over ~{TAU:.0f}s\n")
print(f"{'frames':>8}{'eff. independent':>19}{'sustained action':>19}"
      f"{'3s event':>11}{'0.5s event':>13}")
print("-" * 70)

rows = {}
for n in FRAMES:
    acc, n_eff = action_accuracy(n)
    e3 = event_accuracy(n, 3.0)
    e05 = event_accuracy(n, 0.5)
    rows[n] = (n_eff, acc, e3, e05)
    print(f"{n:>8}{n_eff:>19.2f}{acc:>19.3f}{e3:>11.3f}{e05:>13.3f}")

print(f"""
The effective-independence column is the whole reason video is cheaper than it
looks. Going from 8 frames to 256 -- a factor of 32 in tokens, latency and cost --
raises the number of genuinely independent views from {rows[8][0]:.2f} to
{rows[256][0]:.2f}. Not 32 times more information. About
{rows[256][0] / rows[8][0]:.1f} times, because frames sampled closer together
than the scene's own timescale are near-duplicates of each other
(eq:temporal-redundancy).

That is why the sustained-action column is nearly flat. It reaches
{rows[8][1]:.3f} at 8 frames and {rows[256][1]:.3f} at 256, so thirty-two times
the compute buys {rows[256][1] - rows[8][1]:.3f} of accuracy. For any task whose
evidence persists across the clip -- what activity is this, what room is it in,
who is present -- a handful of frames is genuinely enough, and the instinct to
sample densely is spending a great deal for nothing.

Now the event columns, which is where the flatness stops and where the usual
advice breaks. A 3-second event in a 60-second video is caught
{rows[8][2]:.3f} of the time at 8 frames and {rows[32][2]:.3f} at 32. A
0.5-second event is caught {rows[32][3]:.3f} of the time at 32 frames,
{rows[64][3]:.3f} at 64, and reaches {rows[128][3]:.3f} only at 128 -- the point
at which the sampling interval, {DURATION/128:.2f}s, finally drops below the
event's duration. Before that the curve is not converging; it is just
proportional to the frame count, because each frame is an independent lottery
ticket on a window it either lands in or does not.

Nothing about the video changed between those columns. The task did.
eq:event-catch-probability says catching an event of length d requires a sampling
interval shorter than d, so the frame count scales as DURATION/d -- and it does
not care at all about the redundancy that made the action column flat, because
the question is no longer "what is the scene" but "did this instant occur".

So the two task families have requirements that differ by more than an order of
magnitude on identical footage, and one number cannot serve both. The rule that
follows is concrete: sample at DURATION/d frames where d is the SHORTEST event
you must not miss, and if you do not know d, that is the measurement to make
before choosing a frame rate.

One consequence worth stating because it is uncomfortable. For genuinely brief
events -- a hand leaving a shelf, a single frame of a licence plate -- uniform
sampling is the wrong tool at any affordable rate, and the answer is not more
frames. It is a cheap detector run at full frame rate to propose candidate
moments, with the expensive model looking only at those. That is
ch:rag-corrective's cascade, applied along the time axis.""")
