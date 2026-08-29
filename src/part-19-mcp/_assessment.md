---
id: part-19-assessment
status: final
---

## How to use this

Four sections. The knowledge check takes about ninety minutes. The assignment is
to **build a server and audit a host**, because this part's findings are mostly
things a server author or host operator can act on directly. The challenge problems
are open-ended. The interview section is what to rehearse.

No answers are provided. Every question is answerable from the chapters.

## Knowledge check

**Why protocols**

1. State the $N \times M$ argument, then state what is wrong with the usual version
   of it.
2. A small ecosystem loses on build cost. Derive the term that reverses the verdict
   and say by what year.
3. Explain why a protocol gets adopted before it is break-even in total cost.
4. State {{eq:support-window-beats-upgrade-pressure}} and explain why a threshold
   parameter can saturate a probability where a scale parameter cannot.
5. An eight-revision support window scored $57.8\%$ and a three-revision window
   $68.6\%$. What was missing?
6. Why should a version number increment only on breaking change, and what does
   that cost?

**Architecture**

7. Name MCP's participants and say which one the model is not.
8. What changed between the `2025-11-25` and `2026-07-28` revisions, and why is it
   an operational change rather than an aesthetic one?
9. State {{eq:stateless-removes-the-chain}} and explain why the gap grows with
   session length.
10. Your fleet is imbalanced under sticky routing and you add replicas. Predict
    what happens and justify it.
11. Two deployments both report $99.6\%$ availability. What single additional
    number distinguishes them, and what are its values here?
12. Explain why replication is under-provisioned by an availability-driven capacity
    plan.
13. Why does `server/discover` have to be implemented but not called?

**Primitives**

14. State {{eq:primitive-is-a-controller-choice}} and give the practical test for
    classifying a capability.
15. Why is a wrong tool call worse than a failed one, and what is the cheapest fix?
16. Demand for your context is highly concentrated. Should you preload more or
    less? Derive it.
17. Your selection reliability is $55\%$. What follows for the preload fraction?
18. State {{eq:resources-go-stale}} and compute the volatility threshold above
    which nothing else matters.
19. Why does the freshness threshold fall over time without anything changing?

**Schemas and budgets**

20. {{ch:ag-tool-calling}} found tool count nearly free. Reconcile that with a
    two-thousand-tool inventory scoring $0.1\%$.
21. Compute the token rent of a hundred-tool inventory at typical description
    length.
22. Why is showing eight tools better than showing sixty-four when the inventory is
    two thousand?
23. At what inventory size does retrieval start winning, and why is that earlier
    than intuition?
24. Name the three claimants on a context budget and say which one arrives without
    a decision being made.
25. Success peaked near $24{,}000$ tokens. Explain the shape, and say what happens
    to the optimal split as the budget grows.

**Security**

26. State the three token requirements the specification makes and the one
    structural claim they reduce to.
27. Derive {{eq:passthrough-is-quadratic}} and explain why the specification says
    MUST rather than SHOULD.
28. You implemented audience binding. What is still unbounded, and what fixes it?
29. Rank the defences against tool poisoning by measured effect and justify the
    order.
30. Why is parameter visibility weaker inside an agent than in a conventional
    application?
31. State {{eq:approval-is-a-snapshot}} and give the cheapest mitigation.
32. Why do stdio implementations not use the authorization specification?

**Building and production**

33. What state does an MCP server keep between requests, and what does that remove
    from the implementation?
34. Give three things a server author controls that no client can recover.
35. What is wrong with replacing the scope set on a step-up challenge?
36. State {{eq:review-does-not-scale}} and give its limit as submissions grow.
37. A registry blocks fourteen good servers per bad one. What determines whether
    that is a good policy?
38. State {{eq:marginal-server-turns-negative}} and give the condition at the
    optimum in words.
39. Why does a retrieval layer change how many servers to connect rather than just
    improving a large inventory?
40. Explain the connection between registry governance and host capability.

## Assignment: build a server, audit a host

**Part 1: build.** Implement an MCP server for a system you have access to,
directly against the wire format rather than with an SDK. It must implement
`server/discover`, per-request version negotiation with a window of at least two
revisions, an `UnsupportedProtocolVersionError` carrying the supported list, and
the tool and resource families.

**Part 2: make the errors earn their keep.** Every failure path must name the
field, state what was wrong, and enumerate what would be valid. Then measure it:
inject malformed calls and record recovery rate against an opaque-error baseline.
Report both numbers.

**Part 3: publish the audit.** Annotate every tool with `readOnlyHint` and
`idempotentHint`, derived from what the tool actually does rather than asserted.
Take an idempotency key on every write tool and record it before the effect. Prove
the replay suppression with a test.

**Part 4: cost it.** Report your `tools/list` token count. Rewrite the descriptions
to the distinguish-only standard and report it again.

**Part 5: audit a host.** For a host you operate or can inspect: count connected
servers, total schema tokens, and the fraction of context they consume. For every
token it holds, record which server it is bound to and which scopes it carries;
identify any server that would accept a token issued for another.

**Part 6: compute the optimum.** Estimate $\sigma$ by ablating servers over a task
sample. Compute the marginal coverage of each connected server and identify any
whose marginal coverage is under one percent.

**Part 7: the recommendation.** Which servers should be disconnected, which tokens
re-scoped, and whether a retrieval layer is warranted. Justify each with a number.

Deliverable: working code, plus six pages with the ablation curve as figure one.

## Challenge problems

**A. Error-quality conformance.** Design and implement a conformance suite that
scores error informativeness rather than error presence — does the message name the
field, state the constraint, enumerate the alternatives. Run it against several
public MCP servers and report the distribution. This is the ecosystem's cheapest
unclaimed reliability lever.

**B. Schema compression.** Design a compact wire form for tool schemas with shared
type definitions, expanded only for retrieved candidates. Measure the rent
reduction against selection accuracy on a realistic inventory.

**C. Volatility-aware resources.** Extend a server to publish a per-resource change
rate and a host to apply {{eq:resources-go-stale}}'s threshold automatically.
Measure against a fixed preload policy on content of mixed volatility.

**D. Attested definitions.** Implement signed tool definitions pinned at approval,
and design the update path so legitimate changes are possible without reopening the
rug-pull gap. Where does the trust actually rest?

**E. Reputation-weighted admission.** Combine a structural filter with a
usage-derived reputation signal and test whether it achieves scale-invariance and
tail coverage together. Specify what behaviour signal is required and who could
collect it.

**F. Measure the parameters.** $\pi$ (poisoned fraction), $\lambda$ (hostile-turn
rate), $\sigma$ (coverage scale) and $\delta$ (dilution coefficient) govern every
result in this part, and none has a published estimate. Measure any one of them for
a real ecosystem or model and publish the method.

## Interview preparation

Rehearse until the mechanism comes out before the technique's name.

1. Make the case for a tool protocol to someone with three integrations.
2. Why do protocols spread before they are economically break-even?
3. Your ecosystem's connectivity is $40\%$. What are your levers?
4. Why did MCP move from sessions to self-contained requests?
5. You add replicas and the imbalance gets worse. Explain.
6. Two deployments report identical availability. What do you ask next?
7. When is a tool the wrong primitive?
8. Your agent returns confidently wrong answers with all context preloaded and no
   errors logged. What do you check?
9. Tool count is nearly free. Why is your two-thousand-tool host unusable?
10. Would you show eight tools or sixty-four?
11. You moved to a window eight times larger and quality fell. What happened?
12. Why does the specification forbid token passthrough rather than discourage it?
13. You implemented audience binding. What is still unbounded?
14. Rank the tool-poisoning defences.
15. A server was approved six months ago. What have you assumed?
16. What are the three highest-leverage things a server author can do?
17. Why can't a registry staff its way out of a review backlog?
18. How many servers should a host connect, and how would you find out?
