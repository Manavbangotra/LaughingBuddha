---
id: part-21-intro
status: final
---

## What this part is for

{{part:20}} looked at a domain with weak verifiers and found automation concentrated
in the third of the work that could be graded. This part looks at the domain with
the **strongest** verifier in the book — tests execute, compilers reject, types
constrain — and asks how far that gets you.

Further than anywhere else, and the boundary is measurable.

> **The rule adopted for this part: every capability claim is corrected before it is
> used.** Three corrections apply to published coding-agent numbers — contamination,
> patch verification, and coverage — and they compose multiplicatively. A reported
> $65\%$ becomes roughly $41\%$ after two of them, before any adjustment for your
> repository being less well tested than a popular open-source Python project.

## Where the published numbers sit

| what | number | source |
|---|---|---|
| SWE-bench at launch | **$1.96\%$** of $2{,}294$ real issues resolved | {{cite:jimenez2023swebench}} |
| "Solved" issues that diverge from the human patch | **$29.6\%$**, of which **$28.6\%$** confirmed wrong; rates inflated **$\sim6.2$ points** | {{cite:wang2025solvedcorrectly}} |
| Buggy-file identification from issue text alone | **$76\%$** on benchmark repos vs **$53\%$** elsewhere | {{cite:liang2025swebenchillusion}} |
| Measured developer productivity | forecast **$-24\%$**, self-reported **$-20\%$**, measured **$+19\%$** | {{cite:becker2025devproductivity}} |

That last row is the most important thing in this part and the least known. The
$39$-point gap between what developers reported and what was measured is a larger
finding than the slowdown itself, because it means practitioner testimony about these
tools is not evidence about their effect.

## The organising idea

**Coding agents work because software has a verifier, and every limit in this part is
a place where that verifier stops.**

```text
   CHAPTER                   THE DECISION IT OWNS       WHAT DECIDES IT
   ───────────────────────   ────────────────────────   ─────────────────────────
   183 generation            which suggestions to take  defect cost over saving
   184 repository            where to invest            localisation, not writing
   185 SWE agents            what a resolution means    test coverage
   186 testing               where the tests come from  independence, not coverage
   187 CI/CD                 which changes need a human blast radius
   188 autonomy              how much to delegate       verify times reverse
```

The through-line: **the pipeline is the scaffold.** Four chapters converge on it
independently — reproduction is the best localiser, the test runner and the iteration
loop are mutually contingent, the suite's independence decides what iteration means,
and the automated catch rate decides what needs a human. A team improving CI is
improving the agent.

**And a second through-line this part shares with {{part:20}}.** Twice more, an
honest verifier turned out to lower the reported number:

| Chapter | The honest verifier | What it does to the headline |
|---|---|---|
| {{ch:aids-automl}} | a leakage guard | $-0.177$ reported, $+0.202$ deployed |
| {{ch:aise-swe-agents}} | differential testing | $-2.8$ points reported, correct count unchanged |

Two independent instances make it a rule: **when the metric is produced by the thing
being measured, improving the measurement is penalised.**

## Ten things worth knowing before you start

**Writing code is $15\%$ of a task.** Making it *free* — not fast, free — gives a
$1.17\times$ speedup. The stage everyone means by "AI coding" is the stage that was
already small, because being checkable is what made it efficient first.

**Acceptance rate is not a quality measure.** It is recorded at the moment the
correct and plausible-wrong branches are indistinguishable. Across the sweep, apparent
time per block fell the whole way while true cost including debugging *rose*.

**Generated code is reviewed less carefully than written code.** Reading permits a
decision per block where writing forces one per token, and a plausible function invites
agreement where a blank line demands a decision. At high acceptance with shallow
review, the apparent saving was $+66\%$ and the true saving $-171\%$.

**The gap in issue resolution is localisation, not writing.** On an unseen repository,
writing the fix is $6.9\%$ of failures and localisation is $62.0\%$ — and localisation
is a *ceiling*, since a patch in the wrong file cannot succeed however well written.

**Text similarity cannot reach three quarters of a change set.** It finds the file the
issue talks about; the caller whose contract changed is not textually similar to an
issue that never mentions it. Text-only resolution fell from $77.4\%$ at one file to
$0.0\%$ at eight.

**A failing test is the best localiser available** — flat across change spans where
everything else decays geometrically, because it names the files by executing them.
It also supplies the verifier, the termination condition and a regression guard.

**Passing the tests is evidence proportional to coverage.** About $14\%$ of patches
that pass are wrong, and **inflation grows in absolute terms as agents improve** while
precision stays flat. Better agents write more subtly-wrong code, not less.

**Tests generated from code achieve the highest coverage and the lowest detection** —
$88\%$ and $9.8\%$, against a human suite's $75\%$ and $68.3\%$. Writing them from the
*specification* instead recovers $+25$ points of detection.

**Gate by blast radius, not by author.** A gate on schema migrations returns $33.8$
units per review-minute; a gate on documentation returns $0.00$. Volume cancels out of
the expression entirely.

**Autonomy is an environment property.** Today's model with seven prerequisites in
place reaches $18.7\%$ safe autonomy; a model $60\%$ better with none of them reaches
$5.8\%$. The largest single step is a **rollback path**, which no model improvement
touches.

## What this part deliberately does not cover

**Programming.** The reader is assumed to write software. This part is about what
changes when a machine writes some of it.

**Particular tools or frameworks.** They will have changed. The properties measured
here — coverage, reversibility, localisation, independence — will not.

**Training code models.** {{part:14}}'s. The models are given.

**The redesign effect.** Conceded in {{ch:aise-autonomy}} and measured nowhere here:
every listing prices a fixed workload done differently, and cannot see the value of
changes that became worth attempting because they got cheap. It plausibly points the
opposite way from most of this part's caution.

## How to read it

{{ch:aise-generation}} and {{ch:aise-autonomy}} are the two chapters about *effect* —
what these tools do to a working developer and to an organisation — and they bracket
the part. Reading them together, first and last, gives the argument without the
mechanism.

{{ch:aise-repo}}, {{ch:aise-swe-agents}} and {{ch:aise-testing}} are the mechanism,
and they should be read in order: localisation caps resolution, resolution is graded
by a suite, and the suite's origin decides what the grade means.

{{ch:aise-cicd}} is the practical chapter and the one to read first if you operate a
pipeline rather than write code.

> **One thing to notice on a second reading**: {{ch:aise-repo}} concludes that
> reproduction should come first, {{ch:aise-testing}} concludes that a failing test
> supplies four things at once, {{ch:aise-swe-agents}} finds it worth $+11.8$ points
> even added last, and {{ch:aise-autonomy}} puts it third in the build order.
> **All four are the same recommendation.** A reproduction is not a verification step
> that happens to come early — it is simultaneously the localiser, the verifier, the
> termination condition and the regression guard, and it is the single artefact that
> most changes what an agent can do in a codebase.
