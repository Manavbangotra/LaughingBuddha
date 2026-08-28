---
id: mm-video-audio
number: 128
part: XIII
tier: full
status: draft
requires: [mm-vlms, mm-vit, mm-segmentation, mm-multimodal-rag,
           llm-long-context, rag-corrective]
provides: [temporal-redundancy, effective-frame-count, event-catch-probability,
           video-token-budget, frame-sampling-policy, audio-as-text-front-end,
           temporal-cascade]
citations: [tong2022videomae, radford2022whisper, ravi2024sam2,
            wang2024qwen2vl, faysse2025colpali, kirillov2023sam]
---

## 1. Learning Objectives

By the end of this chapter you will be able to quantify **temporal redundancy** and
explain why a video contains far less information than its frame count suggests;
compute how many frames a task needs from the **shortest event you must not
miss**, rather than from a convention; show that the frame requirements of two
task families on identical footage differ by more than an order of magnitude;
allocate a fixed token budget between frames and per-frame resolution, and
recognise when no split is adequate; and say what audio contributes as a
front-end and why it is the easiest modality to add.

## 2. Why This Matters

Video looks like the expensive modality and mostly is not — it looks expensive
because people count frames, and frames are the wrong unit.

{{cite:tong2022videomae}} masks **90–95%** of a video and still learns from it,
which is the quantitative statement of the situation: adjacent frames are nearly
identical, so a frame is not an independent observation.
{{sec:9-practical-example}} measures the consequence: going from 8 sampled frames
to 256 — **32× the tokens, latency and cost** — raises the number of genuinely
independent views from **5.99 to 10.77**, a factor of **1.8**.

**So the design question in video is almost never "how do I model time". It is
"which frames do I keep", and the honest answer for most tasks is remarkably
few.**

And the right number is not a property of video. {{sec:9-practical-example}}
measures two task families on identical footage. A sustained action is classified
at **0.945 with 8 frames** and **0.974 with 256** — thirty-two times the compute
for three points. A **0.5-second event** in the same 60-second video is caught
**0.069** of the time at 8 frames and reaches certainty only at **128**.

**Nothing about the video changed between those columns. The task did.**

Then video makes {{ch:mm-vlms}}'s budget two-dimensional — frames times
tokens-per-frame — and the two axes conflict. At a 256-token budget the best split
for a spatial question is **1 frame** and for a temporal one **32**: opposite ends
of the sweep, same footage, same budget.

{{maturity:MATURE}} Frame-sampling VLMs and speech front-ends.
{{maturity:EMERGING}} Promptable temporal propagation
({{cite:ravi2024sam2}}) and long-horizon video understanding.

## 3. Prerequisites

{{ch:mm-vlms}} for the visual token budget, which this chapter multiplies by a
frame count; {{ch:mm-vit}} for {{eq:patch-compression}}, which sets the spatial
axis; {{ch:llm-long-context}} for what a very long context does;
{{ch:mm-segmentation}} for masks, which {{cite:ravi2024sam2}} propagates through
time; {{ch:rag-corrective}} for the cascade this chapter recommends;
{{ch:mm-multimodal-rag}} for indexing what comes out.

## 4. Intuitive Explanation

### A frame is not an observation

Thirty frames per second, sixty seconds: 1800 frames. Feed them all to a VLM at a
few hundred tokens each and you have half a million visual tokens for one minute
of video — clearly impossible, and clearly unnecessary, because frames 400 and 401
are the same picture.

**The right question is how many *independent* observations the video contains**,
and the answer is governed by how fast the scene changes.
{{sec:9-practical-example}} puts it starkly: 8 frames give 5.99 effective
independent views and 256 give 10.77. **The additional 248 frames added 4.8
independent observations.**

{{cite:tong2022videomae}}'s 90–95% masking ratio is the same fact from the
training side: you can throw away almost all of a video and still have a solvable
reconstruction task, which would be impossible if frames carried independent
content.

### Which frames, and how many, depends on what you are looking for

Here is the distinction that replaces "one frame per second":

> **Sustained evidence** — what activity is this, what room, who is present. The
> answer is visible in most frames, so a handful suffices and more adds almost
> nothing.
>
> **Momentary evidence** — did the hand leave the shelf, did the sign appear, how
> many times did the door open. The answer exists in a short window, and you must
> *land a frame inside it*.

The second is a coverage problem, not an accuracy problem. Catching an event of
duration $d$ needs a sampling interval shorter than $d$, so the frame count scales
as $\text{duration}/d$ — **and no amount of redundancy helps, because the question
is not "what is the scene" but "did this instant occur".**

Measured: a 3-second event is caught with certainty at 32 frames; a 0.5-second
event needs **128**. Same video, same model, different question.

**So "how many frames" is answerable only after "what is the shortest thing I
must not miss".** If you do not know that number, measuring it is the work to do
before choosing a sampling rate.

### Video makes the budget two-dimensional

{{ch:mm-vlms}} spent a budget on one image. Video splits it:

$$ \text{budget} = \text{frames} \times \text{tokens per frame} $$

Every token spent on temporal coverage is one not spent on spatial detail.
{{sec:9-practical-example}} sweeps the split and finds the two optima at
**opposite ends**: 1 frame for reading something small, 32 frames for counting
events, at the same budget on the same footage.

**There is no compromise setting that is good at both.** A question needing detail
*and* timing — read the licence plate of the car that ran the light — scores
**0.109** at its best split on a 256-token budget. That is not a tuning failure;
the budget cannot serve the question.

**And unusually, spending more resolves it.** The joint best rises to **0.309** at
4096 tokens. Most trades in this book do not improve by buying more; this one
does, which makes it a budgeting decision rather than an architectural one.

### When no split works, stop sampling uniformly

If the required product exceeds your budget, the answer is not a better split. It
is to **stop spending evenly on a video whose interesting content is not evenly
distributed**.

Run something cheap at full frame rate — motion detection, a small classifier, a
scene-cut detector — to propose candidate moments, then run the expensive model at
high resolution on only those. That is {{ch:rag-corrective}}'s cascade applied
along the time axis, and it is the standard architecture for any video system with
a real budget.

### Audio is the easy modality

Speech recognition is a solved-enough commodity ({{cite:radford2022whisper}}), and
that changes what audio *is* architecturally: **a front-end that converts a
modality into text**, after which everything in {{part:12}} applies unchanged.

That is a much easier integration than vision, and it is why audio pipelines are
usually transcribe-then-treat-as-text rather than a joint audio-language model.
**The cost is everything speech recognition discards** — tone, emphasis, speaker
identity, overlap, non-speech sound — which matters enormously for some tasks and
not at all for most.

## 5. Formal Explanation

### 5.1 Temporal redundancy

Model the scene as a latent state $z(t)$ with autocorrelation decaying over a
timescale $\tau$. Two frames separated by $\Delta t$ have correlation
$\rho = e^{-\Delta t/\tau}$, so a frame contributes new information in proportion
to $1 - \rho$:

$$ n_{\text{eff}}(n) = 1 + \sum_{i=1}^{n-1}\left(1 - e^{-\Delta t_i/\tau}\right) $$ (eq:effective-samples)

For uniform sampling of a video of length $T$, $\Delta t = T/n$, so

$$ n_{\text{eff}} \;\xrightarrow[n \to \infty]{}\; 1 + \frac{T}{\tau} $$ (eq:temporal-redundancy)

**{{eq:temporal-redundancy}} is a hard ceiling.** A 60-second video with a
6-second timescale contains about **11** independent observations no matter how
finely you sample it — measured 10.77 at 256 frames. **Sampling past
$n \approx T/\tau$ buys essentially nothing for sustained-evidence tasks.**

### 5.2 Catching a momentary event

An event of duration $d$ at a uniformly random start time is caught by uniform
sampling at interval $\Delta t = T/n$ when at least one sample lands inside it:

$$ \Prob[\text{caught}] \approx \min\!\left(1,\; \frac{d}{\Delta t}\right) = \min\!\left(1,\; \frac{n\,d}{T}\right) $$ (eq:event-catch-probability)

**Linear in $n$ until it saturates**, with the saturation point at

$$ n^{*} = \frac{T}{d} $$ (eq:frames-for-event)

At $T = 60$, $d = 0.5$: $n^* = 120$, and the measurement reaches 1.000 at 128 —
the first sweep point above 120.

**Note what {{eq:event-catch-probability}} does not contain: $\tau$.** Redundancy
is irrelevant here. The two task families are governed by different equations, and
that is why one number cannot serve both.

### 5.3 The two-dimensional budget

$$ B = n_{\text{frames}} \times t_{\text{per frame}} $$ (eq:video-token-budget)

with the spatial axis set by {{ch:mm-vit}}'s
{{eq:patch-compression}} — a feature of $s$ pixels on a $W$-pixel frame survives
when $W/\sqrt{t} \le s$, so

$$ t^{*} = \left(\frac{W}{s}\right)^{2} $$ (eq:tokens-for-feature)

and the temporal axis by {{eq:frames-for-event}}. The budget a task genuinely
needs is therefore the **product of two independently measurable quantities**:

$$ B^{*} = \frac{T}{d} \times \left(\frac{W}{s}\right)^{2} $$ (eq:required-video-budget)

**{{eq:required-video-budget}} is the chapter's design equation.** Both factors
come from the task, not the model, and both can be measured before choosing
anything.

### 5.4 When the budget is short

If $B < B^{*}$, no allocation satisfies both. Scoring a joint task by the weaker
axis:

$$ \text{joint}(n) = \min\!\left(\frac{n d}{T},\; \frac{s\sqrt{B/n}}{W}\right) $$ (eq:joint-video-score)

which is maximised where the two are equal, and its value at that point rises with
$B$ — measured **0.109 → 0.167 → 0.309** across budgets of 256, 1024 and 4096.

### 5.5 The temporal cascade

Uniform sampling assumes interesting content is uniformly distributed in time. It
is not. Let a cheap detector with recall $r_c$ and cost $c_c$ per frame propose a
fraction $\phi$ of moments:

$$ \text{cost} = c_c N_{\text{all}} + \phi N_{\text{all}} \, C_{\text{VLM}}, \qquad \text{recall} \le r_c $$ (eq:temporal-cascade)

**The expensive model now sees a fraction $\phi$ of the frames at full
resolution**, so the effective budget per examined frame rises by $1/\phi$. This
is {{ch:emb-reranking}}'s cascade and {{ch:rag-corrective}}'s escalation with time
as the axis, and it inherits their property: **the ceiling is $r_c$**, the cheap
stage's recall, and no downstream quality recovers a moment it never proposed.

### 5.6 Audio as a front-end

$$ \text{audio} \xrightarrow{\ \text{ASR}\ } \text{text} \xrightarrow{\ \text{everything in Part XII}\ } \text{answer} $$ (eq:audio-front-end)

{{cite:radford2022whisper}}'s 680,000 hours of weak supervision made the first
arrow reliable enough to treat as infrastructure. The consequence is
architectural: **audio does not need a joint model, it needs a transcript**, and
after transcription the corpus is text.

What {{eq:audio-front-end}} discards is real and task-dependent — prosody, speaker
identity, overlapping speech, non-speech events — and the right question is
whether your task depends on any of it. **For most tasks it does not, which is why
this is the cheapest modality to add and the one most often over-engineered.**

## 6. Mathematical Foundation

### 6.1 The redundancy ceiling, checked

{{eq:temporal-redundancy}} predicts $1 + T/\tau = 1 + 60/6 = 11$ effective
observations. Measured at 256 frames: **10.77**, approaching the ceiling from
below as expected.

At 8 frames, {{eq:effective-samples}} gives $1 + 7(1 - e^{-7.5/6}) = 1 + 7(0.713)
= 5.99$ — matching the measurement exactly.

**So the marginal value of the 249th frame is computable in advance**, and it is
approximately zero.

### 6.2 The two families, side by side

| frames | $n_{\text{eff}}$ | sustained action | 3 s event | 0.5 s event |
|---|---|---|---|---|
| 8 | 5.99 | **0.945** | 0.425 | 0.069 |
| 32 | 9.32 | 0.968 | **1.000** | 0.259 |
| 128 | 10.54 | 0.973 | 1.000 | **1.000** |
| 256 | 10.77 | **0.974** | 1.000 | 1.000 |

**Read the first and last rows of the action column**: 32× the compute for 0.029
accuracy. **Then read the last column**: the same 32× is the difference between
0.069 and 1.000.

The event columns are linear in $n$ until saturation, exactly as
{{eq:event-catch-probability}} says — 0.069, 0.140, 0.259, 0.538 doubles with each
doubling of $n$. **That curve is not converging; each frame is an independent
lottery ticket on a window it either lands in or does not.**

> **MATH NOTE:** {{eq:event-catch-probability}} assumes the event is caught if any
> frame lands inside it, which is optimistic — a frame at the very edge of a
> motion may not show it clearly. In practice the effective $d$ is somewhat
> shorter than the nominal event duration, so {{eq:frames-for-event}} is a lower
> bound on the frames required. The direction of the error is the unsafe one, so
> sample above $T/d$ rather than at it.

### 6.3 The required budget, worked

For "read a licence plate (14 px on a 1024 px frame) at the moment a car passes
(2.5 s in a 60 s video)":

$$ t^* = (1024/14)^2 = 5350, \qquad n^* = 60/2.5 = 24, \qquad B^* = 128{,}000 $$

**Which is why that task is not solved by prompting a VLM with a video.** The
budget it needs is two orders of magnitude beyond what a general model spends on a
clip, and {{eq:joint-video-score}}'s best joint score at a realistic 4096 tokens is
**0.309**.

The cascade is not an optimisation here; it is the only affordable architecture.
Detect the car cheaply at full frame rate, then spend 5350 tokens on the two or
three frames that matter — total cost a few tens of thousands of tokens instead of
128,000, with better coverage.

## 7. Internal Mechanics

```mermaid {#fig:video-pipeline caption="Uniform sampling against a temporal cascade. The top path spends its budget evenly on a video whose informative content is not evenly distributed; the bottom path spends almost nothing on most frames and the full per-frame budget on the few that matter (eq:temporal-cascade). The cheap detector's recall is the ceiling — a moment it does not propose is never seen."}
flowchart TB
    V["video: T seconds"] --> U["uniform sample n frames"]
    U --> VLM1["VLM at B/n tokens per frame"]
    VLM1 --> A1["answer"]
    V --> CH["cheap detector at full frame rate<br/>motion, scene cut, small classifier"]
    CH --> CAND["candidate moments<br/>fraction phi of frames"]
    CAND --> VLM2["VLM at FULL tokens per frame"]
    VLM2 --> A2["answer"]
    CH -.->|"recall r_c is the ceiling"| CAND
    AUD["audio track"] --> ASR["ASR (cite:radford2022whisper)"]
    ASR --> TXT["transcript -> Part XII applies"]
    TXT --> A2
```

### 7.1 Choosing frames better than uniformly

Uniform sampling is the default and rarely the best use of a budget:

| Policy | Good for | Cost |
|---|---|---|
| uniform | unknown content, sustained evidence | none |
| scene-cut aligned | edited video, slides, screencasts | trivial |
| motion-triggered | surveillance, sparse activity | small |
| detector-proposed | a known target object or event | a model per frame |
| audio-triggered | anything where speech marks the moment | ASR, already free |

**The last row is under-used.** In a meeting recording, a lecture, or a screencast,
the *transcript* tells you where the interesting moments are, and aligning frame
sampling to speech is nearly free because the transcript was being made anyway.

### 7.2 Masks through time

{{cite:kirillov2023sam}} segments an image on a prompt;
{{cite:ravi2024sam2}} adds a streaming memory so a mask prompted once is carried
forward. **That converts tracking from a separate discipline into a segmentation
prompt**, and its practical effect is that "follow this object" stopped needing a
tracker.

Note the same redundancy argument applies: propagation is cheap precisely because
consecutive frames barely differ, so the memory has little to update.

### 7.3 What audio adds beyond the words

Transcription is the easy 90%. The rest is where the remaining engineering is:

- **Diarisation** — who spoke. Usually needed, usually harder than the
  transcription.
- **Timestamps** — required to align speech with frames, and the mechanism behind
  audio-triggered sampling.
- **Non-speech events** — a door, an alarm, a machine fault. Outside ASR entirely
  and needing a separate classifier.
- **Prosody** — tone and emphasis. Discarded by {{eq:audio-front-end}} and
  genuinely lost.

**Decide which of these your task needs before choosing an architecture**, because
"we have a transcript" is often taken to mean "we have the audio" and it does not.

## 8. Implementation

```python {tier=A name=temporal-redundancy}
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
```

The first listing sets the frame count. The second shows what it costs, because
frames are only half the budget.

```python {tier=A name=video-token-budget}
"""Video makes ch:mm-vlms's token budget two-dimensional, and the axes conflict.

A VLM spends N visual tokens on one image. A video spends frames x tokens-per-frame,
so at a fixed budget B the two multiply out (eq:video-token-budget):

    B = n_frames * tokens_per_frame

Every token spent on temporal coverage is a token not spent on spatial detail,
and vice versa. There is no setting that is generous on both, which makes this a
genuine allocation problem rather than a tuning one.

The two task families from the previous listing want opposite splits. A SPATIAL
question -- read the sign, identify the small object -- needs resolution and
tolerates few frames. A TEMPORAL question -- count the events, order them -- needs
frames and tolerates low resolution. This listing sweeps the split and finds both
optima.
"""
import numpy as np

DURATION = 60.0
BUDGETS = (256, 1024, 4096)
SPLITS = (1, 2, 4, 8, 16, 32, 64, 128)      # frames; per-frame tokens = B / n

FEATURE_PX = 14.0        # the thing to read is 14 px on a 1024 px frame
FRAME_PX = 1024.0
EVENT_S = 2.5            # the thing to count lasts 2.5 seconds


def spatial_score(tokens_per_frame):
    """ch:mm-vit's eq:patch-compression: a feature survives if the patch is no
    bigger than it. Patch side = frame / sqrt(tokens). See eq:tokens-for-feature."""
    if tokens_per_frame < 1:
        return 0.0
    patch = FRAME_PX / np.sqrt(tokens_per_frame)
    return float(np.clip(FEATURE_PX / patch, 0.0, 1.0))


def temporal_score(n_frames):
    """eq:event-catch-probability: an event of EVENT_S is caught when the
    sampling interval is shorter than it."""
    interval = DURATION / n_frames
    return float(np.clip(EVENT_S / interval, 0.0, 1.0))


print(f"{DURATION:.0f}s video, frames {FRAME_PX:.0f}px; spatial target "
      f"{FEATURE_PX:.0f}px, temporal target {EVENT_S:.1f}s\n")

best = {}
for B in BUDGETS:
    print(f"budget B = {B} visual tokens")
    print(f"{'frames':>8}{'tok/frame':>11}{'spatial':>10}{'temporal':>11}"
          f"{'both (min)':>12}")
    print("-" * 52)
    rows = []
    for n in SPLITS:
        tpf = B // n
        sp, tp = spatial_score(tpf), temporal_score(n)
        rows.append((n, tpf, sp, tp, min(sp, tp)))
        print(f"{n:>8}{tpf:>11}{sp:>10.3f}{tp:>11.3f}{min(sp, tp):>12.3f}")
    bs = max(rows, key=lambda r: r[2])
    bt = max(rows, key=lambda r: r[3])
    bb = max(rows, key=lambda r: r[4])
    best[B] = (bs, bt, bb)
    print(f"  best for SPATIAL: {bs[0]} frames    "
          f"best for TEMPORAL: {bt[0]} frames    "
          f"best for BOTH: {bb[0]} frames (score {bb[4]:.3f})\n")

b_small, b_large = best[BUDGETS[0]], best[BUDGETS[-1]]
print(f"""
Within any single budget block, the spatial and temporal columns move in opposite
directions -- one rises as the other falls, because they are reading the same
number from two ends (eq:video-token-budget). The best split for a spatial
question is {b_small[0][0]} frame at B={BUDGETS[0]} and the best for a temporal
one is {b_small[1][0]} -- opposite ends of the sweep, on the same footage and the
same budget. There is no compromise setting that is good at both; there is only a
choice about which question you are asking.

The "both" column makes that concrete by scoring the WORSE of the two, which is
what a question needing detail AND timing actually experiences. At B={BUDGETS[0]}
its best is {b_small[2][4]:.3f} -- the budget simply cannot serve a question that
needs to read something small at a moment it must not miss.

Now compare across budgets. At B={BUDGETS[-1]} the joint best rises to
{b_large[2][4]:.3f} at {b_large[2][0]} frames, so the conflict is not permanent --
it is what a small budget looks like. Buying tokens buys both axes at once, which
is unusual and worth noticing: most trades in this book do not resolve by
spending more.

The practical shape of the answer is therefore to stop treating "how many frames"
as a video question. It is two questions with two different answers:

  What is the smallest thing I must SEE?   -> sets tokens per frame, via
                                              ch:mm-vit's eq:patch-compression.
  What is the shortest thing I must NOTICE? -> sets frame count, via
                                              eq:event-catch-probability.

Both are measurable from the task before any model is chosen, and their product
is the budget you need. If that product exceeds what you can afford, the answer is
not a compromise split -- it is a cascade: a cheap detector at full frame rate to
find candidate moments, then the expensive model at high resolution on only those.
Uniform sampling spends its budget evenly over a video whose interesting content
is not evenly distributed, and that is the assumption to break first.""")
```

## 9. Practical Example

**A frame is not an observation.** Going from 8 sampled frames to 256 — **32× the
tokens, latency and cost** — raises effective independent views from **5.99 to
10.77**, a factor of **1.8**. {{eq:temporal-redundancy}} predicted the ceiling at
$1 + T/\tau = 11$, and {{eq:effective-samples}} predicted 5.99 at eight frames
exactly.

**So the sustained-action column is nearly flat**: 0.945 at 8 frames, 0.974 at 256.
**Thirty-two times the compute for 0.029 of accuracy.** For any task whose evidence
persists — what activity, what room, who is present — a handful of frames is
genuinely enough, and dense sampling is spending a great deal for nothing.

**And then the event columns, where that stops being true.** A 3-second event is
caught 0.425 of the time at 8 frames and with certainty at 32. A **0.5-second
event** is caught **0.069** at 8, **0.259** at 32, **0.538** at 64, and reaches
1.000 only at **128** — where the sampling interval, 0.47 s, finally drops below
the event.

> **IMPORTANT:** Look at that sequence — 0.069, 0.140, 0.259, 0.538 — **doubling
> with each doubling of frames.** That curve is not converging. Each frame is an
> independent lottery ticket on a window it either lands in or does not, exactly
> as {{eq:event-catch-probability}} says, and $\tau$ does not appear in that
> equation at all. **The redundancy that made the action column flat is simply
> irrelevant to the event column.** Two task families, two equations, one video.

**So "how many frames" is unanswerable without "the shortest thing I must not
miss".** {{eq:frames-for-event}} gives $T/d$ — 120 for a half-second event in a
minute of video — and the measurement saturates at the first sweep point above it.

**Video then makes {{ch:mm-vlms}}'s budget two-dimensional, and the axes fight.**
At a 256-token budget the best split is **1 frame** for a spatial question and
**32 frames** for a temporal one: opposite ends of the sweep, same footage, same
budget. A question needing both scores **0.109** at its best split.

**That is not a tuning failure — the budget cannot serve the question.** And
{{eq:required-video-budget}} says how far short it is: reading a 14-pixel plate at
a 2.5-second moment needs $24 \times 5350 = 128{,}000$ tokens.

**Unusually, spending more does resolve it**: the joint best rises **0.109 → 0.167
→ 0.309** across budgets of 256, 1024 and 4096. Most trades in this book do not
improve by buying more, which makes this a budgeting decision rather than an
architectural one — **until the required product exceeds what you can afford, at
which point the cascade is the only affordable architecture**, not an
optimisation.

## 10. Production Considerations

**Measure $d$, the shortest event you must not miss**, before choosing a frame
rate. {{eq:frames-for-event}} then gives the number.

**Measure $s$, the smallest thing you must see**, and use
{{eq:tokens-for-feature}}. Their product is
{{eq:required-video-budget}}, and it tells you in advance whether uniform sampling
can work at all.

**Do not sample densely for sustained-evidence tasks.**
{{eq:temporal-redundancy}} caps the return at $1 + T/\tau$.

**Use a temporal cascade whenever $B^* $ exceeds your budget**, and measure the
cheap stage's *recall* — it is the ceiling ({{eq:temporal-cascade}}), and a moment
it never proposes is never seen.

**Align sampling to something.** Scene cuts, motion, or the transcript beat uniform
sampling at no extra cost, and the transcript is usually already there.

**Transcribe audio and treat it as text** unless your task needs prosody, speaker
identity, or non-speech events — decide which explicitly.

**Get timestamps from ASR**, because they are what makes audio-triggered frame
selection possible.

**Report frames and tokens-per-frame separately** in any video benchmark. A single
"token budget" hides the allocation, which is the actual design decision.

## 11. Common Mistakes

**Quoting a frame rate as though it were a property of video** rather than of the
task.

**Sampling densely for action recognition**, where the return is capped.

**Sampling uniformly for brief-event detection**, where uniform sampling is the
wrong tool at any affordable rate.

**Treating the video budget as one number** instead of a product of two.

**Assuming more frames always helps** — {{eq:temporal-redundancy}}.

**Ignoring the cheap stage's recall** in a cascade, then tuning the expensive
stage.

**Believing a transcript is the audio.** It discards prosody, speakers and
non-speech events.

## 12. Failure Modes

**Missed brief events.** Symptom: the system describes videos well and misses the
one moment that mattered. Cause: {{eq:event-catch-probability}} at too low a
frame rate. Detect by measuring recall against a labelled set of *short* events
specifically.

**Wasted budget on redundant frames.** Symptom: cost scales with video length and
accuracy does not. Cause: sampling past $T/\tau$.

**Detail lost at high frame counts.** Symptom: dense sampling makes reading worse,
not better. Cause: {{eq:video-token-budget}} — frames were bought with resolution.

**Cascade recall ceiling.** Symptom: the expensive model is excellent on what it
sees and overall recall is poor. Cause: the cheap detector never proposed the
moment.

**Transcript-only blindness.** Symptom: a meeting system misses who said something
or that an alarm sounded.

**Timestamp drift.** Symptom: frames and transcript disagree about when something
happened, so audio-triggered sampling picks the wrong frames.

**Context overflow on long video.** Symptom: quality degrades with clip length
even at a fixed frame rate. Cause: {{ch:llm-long-context}}'s dilution, with
frames as the distractors.

## 13. Alternatives

| Approach | Trades away | When it wins |
|---|---|---|
| uniform sampling | budget efficiency | unknown content, sustained evidence |
| temporal cascade | cheap-stage recall | brief events, tight budgets — usually |
| dedicated video model | VLM generality | high-volume, one task |
| transcript-only | everything visual | meetings, lectures, podcasts |
| mask propagation ({{cite:ravi2024sam2}}) | needs a prompt | tracking a known object |
| frame-level retrieval ({{ch:mm-multimodal-rag}}) | temporal reasoning | "find the clip where..." |

**The fourth row is worth stating plainly**: for a great many "video" tasks the
video is not needed. A meeting summary, a lecture Q&A, a podcast search index —
all of these are text problems wearing a video interface, and
{{eq:audio-front-end}} solves them for the cost of transcription.

## 14. Evaluation

**Evaluate brief-event recall separately.** It is governed by a different equation
and an aggregate hides it entirely.

**Report frames and tokens-per-frame**, not just a total.

**Sweep the frame count and report the curve**, because the flat region is where
the savings are and it is invisible at a single operating point.

**Measure the cheap stage's recall** in any cascade, independently of end-to-end
accuracy.

**Evaluate on your video lengths.** Results on 10-second clips say little about
10-minute recordings, since $T/\tau$ and $T/d$ both scale with $T$.

**For audio, evaluate diarisation and timestamps separately** from word accuracy.

## 15. Advanced Concepts

**Redundancy as a training signal.** {{maturity:MATURE}}
{{cite:tong2022videomae}}'s 90–95% masking works *because* of
{{eq:temporal-redundancy}} — the redundancy that makes video expensive to process
is the same property that makes it cheap to learn from, since a high mask ratio
still leaves a solvable task.

**Token merging along time.** {{maturity:EMERGING}} Since adjacent frames' tokens
are near-duplicates, merging them across the temporal axis attacks
{{eq:video-token-budget}} directly — more frames at no extra context cost. The
most promising direction for long-video understanding, and a direct application of
{{ch:mm-vlms}}'s token-pruning idea.

**Streaming memory.** {{maturity:EMERGING}} {{cite:ravi2024sam2}}'s design
processes video as a stream with a bounded memory rather than a fixed clip, which
is what makes unbounded video length tractable — and is the same architectural
move as {{part:17}}'s agent memory.

**Learned frame selection.** {{maturity:EMERGING}} The cascade's cheap stage can
be learned end-to-end against the downstream task rather than hand-designed, which
turns {{eq:temporal-cascade}}'s $r_c$ into something optimisable. This is
{{ch:rag-corrective}}'s learned grader, one modality over.

**Audio is under-exploited.** {{maturity:MATURE}} The transcript localises events
in time for free, and almost no video pipeline uses it to choose frames. The
cheapest available improvement to most video systems is to sample where someone is
talking.

## 16. Connection to Previous Chapters

{{ch:mm-vlms}}'s visual token budget becomes two-dimensional here
({{eq:video-token-budget}}), and {{ch:mm-vit}}'s
{{eq:patch-compression}} supplies its spatial axis via
{{eq:tokens-for-feature}}. {{ch:llm-long-context}}'s dilution is what caps the
frame count from the language side, as it did in
{{ch:mm-vlms}}. {{ch:mm-segmentation}}'s masks are what
{{cite:ravi2024sam2}} propagates. {{ch:rag-corrective}}'s escalation and
{{ch:emb-reranking}}'s cascade become {{eq:temporal-cascade}} with time as the
axis, and inherit the cheap-stage recall ceiling.
{{ch:mm-multimodal-rag}} is where the resulting frames and transcripts get
indexed, and {{part:12}} applies unchanged to whatever
{{eq:audio-front-end}} produces.

## 17. Exercises

1. Derive {{eq:temporal-redundancy}} from {{eq:effective-samples}} and compute the
   ceiling for a 10-minute video with $\tau = 4$ s.
2. Use {{eq:frames-for-event}} to find the frames needed for a 0.2-second event in
   a 5-minute video. Is uniform sampling viable?
3. In `temporal-redundancy`, set `TAU = 1.0`. Which column changes and which does
   not, and why?
4. Add an event of 10 seconds. At what frame count does it saturate, and does that
   match {{eq:frames-for-event}}?
5. Compute {{eq:required-video-budget}} for reading a 20-pixel timestamp burned
   into a 1920-pixel frame, at a 1-second event, in a 5-minute video.
6. In `video-token-budget`, add B = 16384. Does the joint optimum keep rising, and
   what does {{eq:joint-video-score}} predict as $B \to \infty$?
7. Model a cascade with {{eq:temporal-cascade}}: cheap detector recall 0.9 firing
   on 5% of frames. Compare total cost and recall against uniform sampling at
   equal cost.
8. Take a video task you have. Measure $d$ and $s$, compute $B^*$, and compare
   against what your model actually spends.

## 18. Interview Questions

1. Why is a video less information than its frame count suggests?
2. How many frames should you sample, and what determines it?
3. Why do action recognition and event detection need different frame rates?
4. What caps the return from sampling more frames?
5. Explain the two-dimensional video token budget.
6. A task needs to read something small at a precise moment. What do you build?
7. What is the ceiling on a temporal cascade's recall?
8. When is a "video" task actually a text task?
9. What does a transcript discard, and when does it matter?
10. Your video system misses brief events and describes clips well. Diagnose.

## 19. Research Questions

1. {{eq:temporal-redundancy}} uses a single timescale $\tau$. Real video has many
   — camera motion, object motion, scene cuts. What is the right
   multi-timescale version and does it change sampling policy?
2. Token merging along time promises more frames at fixed context. What is the
   achievable merge ratio before temporal information is lost, and does it depend
   on $\tau$?
3. {{eq:temporal-cascade}}'s ceiling is $r_c$. Can the cheap stage be trained
   against downstream task loss, and how much recall does that buy over a
   hand-designed detector?
4. Audio timestamps localise events for free. How much of the frame budget can be
   saved by transcript-aligned sampling, across task types?
5. {{eq:event-catch-probability}} assumes a frame anywhere in the window suffices.
   What is the true effective event duration for a VLM, and how does it depend on
   motion blur and the event's visual salience?

## 20. Chapter Summary

**Video looks expensive because people count frames, and frames are the wrong
unit.** {{eq:temporal-redundancy}} caps the independent content of a $T$-second
video at $1 + T/\tau$: measured, 32× the frames bought **1.8×** the independent
observations, and {{cite:tong2022videomae}}'s 90–95% masking is the same fact from
the training side.

**So sustained-evidence tasks are cheap.** Action classification went 0.945 → 0.974
across a 32× increase in compute — three points for thirty-two times the bill.

**And momentary-evidence tasks obey a different equation entirely.** A 0.5-second
event was caught 0.069, 0.140, 0.259, 0.538 as frames doubled — **linear in $n$,
not converging** — reaching certainty only at 128, where the sampling interval
finally dropped below the event. {{eq:event-catch-probability}} does not contain
$\tau$: **the redundancy that made one column flat is irrelevant to the other.**

**Which makes "how many frames" unanswerable on its own.**
{{eq:frames-for-event}} gives $T/d$, and $d$ — the shortest thing you must not
miss — is a property of the task that has to be measured.

**Video then makes {{ch:mm-vlms}}'s budget two-dimensional and the axes conflict.**
At 256 tokens the best split is 1 frame for a spatial question and 32 for a
temporal one, and a question needing both scores **0.109**.
{{eq:required-video-budget}} says why: reading a 14-pixel plate at a 2.5-second
moment needs **128,000 tokens**, two orders of magnitude beyond what a general
model spends on a clip.

**Unusually, buying more resolves the conflict** — the joint best rose 0.109 →
0.167 → 0.309 across budgets — **until the required product exceeds what you can
afford**, at which point the temporal cascade is not an optimisation but the only
affordable architecture. Its ceiling is the cheap stage's recall
({{eq:temporal-cascade}}), and a moment never proposed is never seen.

**And audio is the easy modality precisely because it reduces to text.**
{{eq:audio-front-end}}: transcribe, and {{part:12}} applies unchanged. What it
discards — prosody, speakers, non-speech events — is real and usually irrelevant,
which is why this is the modality most often over-engineered. Its underused gift
is *timestamps*: the transcript says where the interesting moments are, for free,
and almost no video pipeline uses it to choose frames.

Which closes the part where it began. **Resolution is the budget** — in pixels in
{{ch:mm-cv-fundamentals}}, in patches in {{ch:mm-vit}}, in visual tokens in
{{ch:mm-vlms}}, and here in tokens *and* frames at once. Every architecture in
this part is a different answer to the same question: what is the unit of
information, and how many can you afford?

## 21. Further Reading

{{cite:tong2022videomae}} for temporal redundancy quantified — the 90–95% masking
ratio is the number to remember, and it explains both why video is cheap to learn
from and why it is expensive to process.
{{cite:ravi2024sam2}} for streaming memory and mask propagation, which turned
tracking into a segmentation prompt.
{{cite:radford2022whisper}} for the front-end that makes audio a text problem,
and note the 680,000 hours: this is the same weak-supervision-at-scale lesson as
{{cite:radford2021clip}}, in a different modality.
{{cite:wang2024qwen2vl}} for how a general VLM handles video, and for the
resolution machinery it reuses.
{{cite:kirillov2023sam}} for the image-level segmentation SAM 2 extends, and
{{cite:faysse2025colpali}} for the retrieval end, developed in
{{ch:mm-multimodal-rag}}.
