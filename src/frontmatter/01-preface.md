---
id: fm-preface
status: final
---

## Why this book exists

There is no shortage of material about artificial intelligence. There is a
severe shortage of material that explains how it works.

The two most common formats both fail the reader who wants to build things. The
first is the tutorial: here is a library, here are twelve lines that call it,
here is a result. It produces someone who can assemble systems from parts they
cannot reason about, and who is helpless the moment a part behaves unexpectedly
— which, in this field, is constantly. The second is the paper: rigorous,
correct, and written for people who already know the surrounding twenty years of
context. It produces nothing at all for a reader who lacks that context.

This book is an attempt at the missing middle: rigorous enough that you can read
papers afterwards, concrete enough that you can ship systems, and complete
enough that you do not need a third source to fill the gaps between the two.

## What "understanding" means here

A recurring pattern in this field is the *plausible explanation* — a description
that is memorable, roughly correct in outline, and useless the moment you need
to predict behaviour. "Attention lets the model focus on relevant parts of the
input" is such an explanation. It is not wrong. It also will not tell you why
the scores are divided by the square root of the key dimension, why long
contexts degrade in the particular way they do, why the KV cache is the binding
constraint on serving cost, or what to change when your retrieval system returns
the right documents and the model still answers incorrectly.

So the standard applied throughout is this: after reading a chapter you should
be able to predict how the thing behaves in a situation the chapter did not
describe. That requires the mechanics, the mathematics behind the mechanics, and
the engineering consequences of both. Where a derivation is needed, the
derivation is here. Where the honest answer is that the field does not know, the
book says so rather than supplying a satisfying story.

## What is permanent and what is not

Most of what is currently exciting about AI will be obsolete within a few years.
Almost none of what is currently *fundamental* will be.

Specific model names, framework APIs, benchmark leaderboards, context-window
sizes, and the relative merits of particular vendors all have short half-lives.
Attention, optimisation, representation learning, retrieval, evaluation
methodology, distributed inference, and the architecture of agentic systems do
not. This book is organised so the durable material carries the weight, and
anything tied to a moment in time is quarantined into clearly marked sections
that can be replaced without disturbing the surrounding argument.

That is also why maturity labels appear throughout. A technique that has been
independently reproduced for a decade and a technique from a preprint published
last quarter are both worth knowing about, and it would be dishonest to present
them in the same voice.

## What this book will not do

It will not tell you that AI is transforming everything. It will tell you what
changed, why it changed, how the change works, and what the engineering
consequences are.

It will not pretend the field is more settled than it is. Several areas covered
here — agent reliability, evaluation of open-ended generation, mechanistic
interpretability, long-horizon autonomy — are genuinely unsolved. Chapters on
those subjects explain the current state of the argument rather than manufacture
a conclusion.

It will not treat frameworks as the subject. Frameworks appear, because you will
use them and because pretending otherwise would be precious. But every important
algorithm is implemented from scratch before any library is called, on the
principle that you cannot debug an abstraction you have never seen the inside
of.

## How the book was built

The book is written as Markdown, checked by an automated test suite, and
rendered to HTML and PDF by a build pipeline in the repository.

The checks exist because a book of this length written over many months will
drift unless something prevents it. Every chapter declares the concepts it
requires and the concepts it provides; the build refuses to proceed if a chapter
depends on one that comes later, or if the dependency graph develops a cycle.
Every equation, figure and table label is resolved book-wide, so a broken
cross-reference is a build failure rather than a reader's problem. Every glossary
term has exactly one definition, and the glossary appendix is generated from it.
Every citation is checked against the bibliography, and every bibliography entry
records the primary source it was verified against and the date — an unverified
citation is rendered as **UNVERIFIED** in the text rather than quietly trusted.
Code listings are extracted to real files and executed where the hardware
permits.

None of this makes the book correct. It makes a specific and otherwise very
likely set of errors impossible.
