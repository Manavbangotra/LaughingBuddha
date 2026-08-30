# -*- coding: utf-8 -*-
# Extracted from: Chapter 151 — Tool-Assisted and Verified Reasoning
# Source: src/.../ch151-tool-assisted.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""When is a tool worth it? The error budget, reallocated.

cite:sprague2024tocot found chain-of-thought's gains concentrated in maths and
symbolic reasoning, and that where it helps it is doing symbolic execution a real
solver does better. That is an argument for handing the execution to a tool, and
cite:yao2023react and cite:gao2023pal are two of the ways to do it.

The argument is usually made as though tools are free. They are not. Calling a
tool replaces one error mode with another: the model no longer has to EXECUTE the
computation, but it does have to TRANSLATE the problem into a call, and a wrong
call produces a confidently wrong result with an authoritative-looking number
attached (eq:tool-error-reallocation).

This listing measures the trade on a task with an explicit number of steps, so
the crossover is visible rather than asserted.
"""
import numpy as np

rng = np.random.default_rng(1223)

N_PROB = 40000
P_EXEC = 0.96             # chance the model executes one step correctly itself
P_TRANS = 0.97            # chance it translates one step into a correct call
P_PARSE = 0.96            # chance it reads the tool's answer back correctly
LATENCY_TOOL = 1.0        # relative cost of a round trip, per call


def unaided(k):
    """The model does every step itself. Errors compound: the chain is right
    only if every step is (ch:rsn-cot's eq:chain-accuracy-compounds)."""
    return float(np.mean((rng.random((N_PROB, k)) < P_EXEC).all(1)))


def tool_per_step(k):
    """One tool call per step. Execution is exact, so the only ways to fail are
    translating a step into a call and reading the result back."""
    ok = ((rng.random((N_PROB, k)) < P_TRANS) &
          (rng.random((N_PROB, k)) < P_PARSE)).all(1)
    return float(np.mean(ok))


def tool_once(k):
    """One call for the WHOLE problem -- write the program, run it once. The
    translation now has to be right about all k steps at once, which is harder
    per call, but there is only one call and one result to read."""
    p_prog = P_TRANS ** k
    ok = (rng.random(N_PROB) < p_prog) & (rng.random(N_PROB) < P_PARSE)
    return float(np.mean(ok))


KS = [1, 2, 3, 5, 8, 12, 20, 32]

print(f"The model executes a step correctly {P_EXEC:.0%} of the time and")
print(f"translates one into a tool call {P_TRANS:.0%} of the time, reading the")
print(f"result back correctly {P_PARSE:.1%} of the time. Accuracy by chain")
print("length:")
print()
print(f"{'steps k':>9}{'unaided':>11}{'tool per':>11}{'one tool':>11}"
      f"{'calls':>9}")
print(f"{'':>9}{'':>11}{'step':>11}{'call':>11}{'made':>9}")
print("-" * 51)

una, tps, ton = {}, {}, {}
for k in KS:
    una[k], tps[k], ton[k] = unaided(k), tool_per_step(k), tool_once(k)
    print(f"{k:>9}{una[k]:>11.1%}{tps[k]:>11.1%}{ton[k]:>11.1%}{k:>9}")

cross = [k for k in KS if tps[k] > una[k]]
cross1 = [k for k in KS if ton[k] > una[k]]

print()
print()
print("The same comparison as a ratio, which is what decides whether the round")
print("trips are worth paying for.")
print()
print(f"{'steps k':>9}{'per-step tool':>16}{'single call':>14}")
print(f"{'':>9}{'vs unaided':>16}{'vs unaided':>14}")
print("-" * 39)
for k in KS:
    print(f"{k:>9}{tps[k] / una[k]:>15.2f}x{ton[k] / una[k]:>13.2f}x")

print()
print()
print("Now vary how good the model is at using the tool, at k=12. A worse")
print("translator eats the advantage; the question is how fast.")
print()
print(f"{'translation':>13}{'unaided':>10}{'tool per':>11}{'one tool':>11}")
print(f"{'accuracy':>13}{'':>10}{'step':>11}{'call':>11}")
print("-" * 45)
K_STUDY = 12
u12 = una[K_STUDY]
sweep = {}
for pt in (0.999, 0.99, 0.97, 0.95, 0.92, 0.88):
    P_TRANS_SAVE = P_TRANS
    globals()["P_TRANS"] = pt
    a, b = tool_per_step(K_STUDY), tool_once(K_STUDY)
    globals()["P_TRANS"] = P_TRANS_SAVE
    sweep[pt] = (a, b)
    print(f"{pt:>13.1%}{u12:>10.1%}{a:>11.1%}{b:>11.1%}")

print()
print()
print("And the cost side: a per-step tool makes k round trips, a single call")
print("makes one. Accuracy per unit of latency at k=12, relative to unaided.")
print()
print(f"{'approach':>20}{'accuracy':>11}{'round trips':>14}{'per trip':>11}")
print("-" * 56)
print(f"{'unaided':>20}{u12:>11.1%}{0:>14}{'--':>11}")
print(f"{'tool per step':>20}{tps[K_STUDY]:>11.1%}{K_STUDY:>14}"
      f"{tps[K_STUDY] / K_STUDY:>11.2%}")
print(f"{'one tool call':>20}{ton[K_STUDY]:>11.1%}{1:>14}"
      f"{ton[K_STUDY]:>11.2%}")

print(f"""
The first table is the trade, and the first row is the part that gets skipped.

At k=1 the unaided model is {una[1]:.1%} accurate and the per-step tool is
{tps[1]:.1%}. **The tool loses on a one-step problem**, and not because the tool
is bad. The model executes one step correctly {P_EXEC:.0%} of the time, and it
translates-and-reads one step correctly {P_TRANS * P_PARSE:.1%} of the time.
Calling out to a calculator to add two numbers removes one way of being wrong and
introduces two.

{'The per-step tool overtakes the unaided model at k=' + str(cross[0]) + '.' if cross else 'The per-step tool NEVER overtakes the unaided model, at any length swept -- and its ratio gets steadily worse, from ' + format(tps[1] / una[1], '.2f') + 'x at k=1 to ' + format(tps[32] / una[32], '.2f') + 'x at k=32.'}
{'The single call overtakes at k=' + str(cross1[0]) + ', and reaches ' + format(ton[32] / una[32], '.2f') + 'x by k=32.' if cross1 else 'The single call never overtakes over the range swept.'}

Two tool designs, opposite verdicts, from the same tool. That is the result, and
the arithmetic behind it is worth following because it generalises.

Every approach here compounds geometrically; they differ only in the base.
Unaided accuracy is {P_EXEC}^k. A per-step tool is ({P_TRANS} x {P_PARSE})^k =
{P_TRANS * P_PARSE:.4f}^k -- a SMALLER base than {P_EXEC}, so it falls behind and
keeps falling, exponentially. A single call for the whole problem is
{P_TRANS}^k x {P_PARSE} -- the larger base {P_TRANS}, paid for with a one-off
{P_PARSE:.0%} penalty for reading one result.

**The per-step design pays the parse cost k times; the single-call design pays it
once.** At k=1 that difference is nothing and the one-off penalty dominates, so
both lose. As k grows the base wins, which is why the single call crosses over and
the per-step version never does (eq:tool-error-reallocation).

So "should I use a tool" is the wrong question and "how many times do I cross the
boundary" is the right one. Every round trip is a fresh opportunity to
mis-translate and to mis-read, and those opportunities multiply.

That is the difference between cite:gao2023pal's design and an interleaved one.
PAL's own framing is the mechanism this listing measures: models decompose
problems correctly and then make arithmetic mistakes in the solution, so hand the
solution to an interpreter and keep the decomposition. Doing that ONCE, as a
program, is what makes the arithmetic work out. Interleaving calls in the style
of cite:yao2023react buys the ability to let later steps depend on earlier
results, and this table is what that flexibility costs when you do not need it.

The second table is the sensitivity that decides whether any of this survives
contact with your own model. At k={K_STUDY} the unaided baseline is {u12:.1%}.
With translation accuracy at {0.999:.1%} the single call reaches
{sweep[0.999][1]:.1%}; at {0.97:.0%}, {sweep[0.97][1]:.1%}; at {0.95:.0%},
{sweep[0.95][1]:.1%} -- below the unaided baseline; at {0.88:.0%},
{sweep[0.88][1]:.1%}.

The crossover sits near the execution accuracy itself, which gives the summary
worth carrying: **a tool helps exactly when the model is better at calling it than
at doing the work.** That reads as a tautology and is not, because both halves are
routinely assumed rather than measured. A model with strong arithmetic and a
shaky grasp of an unfamiliar API is on the wrong side of the inequality, and the
resulting system is worse than the one that did the sums itself -- while looking
more rigorous, because there is a tool in the trace.

Note also how steep that column is. Dropping translation accuracy from
{0.99:.0%} to {0.95:.0%} costs the single call
{sweep[0.99][1] - sweep[0.95][1]:.1%} at k={K_STUDY}. Anything that degrades
translation -- an API change, an unfamiliar schema, a longer context, a
distribution shift in problem phrasing -- is amplified by the chain length, so
tool-use reliability is not a fixed property of a model but a property of a model
against a particular interface.

The third table adds latency, and it is why the single-call design usually wins
in production even where the accuracy comparison is close. At k={K_STUDY} the
per-step tool makes {K_STUDY} round trips for {tps[K_STUDY]:.1%}; the single call
makes one for {ton[K_STUDY]:.1%}. Per round trip that is
{tps[K_STUDY] / K_STUDY:.2%} against {ton[K_STUDY]:.2%}.

Round trips are serial by construction -- the next call needs the previous result
-- so they do not batch, and each one re-enters the model with a longer context.
On part:15's serving economics that is close to the most expensive shape a
request can have.

So the decision needs three measurements and no philosophy. Measure your model's
per-step execution accuracy on the task. Measure its per-step translation accuracy
against your actual tool, and its parse accuracy on your actual response format.
If translation beats execution, use the tool, by a factor that grows exponentially
in chain length -- and then cross the boundary as few times as you can, which
means one program rather than k calls unless later steps genuinely need earlier
results.""")
