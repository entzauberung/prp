# Thesis

## Claim

Agent reasoning should be a protocol of public facts, not a transcript of
private thought.

Large language models already produce actions, reflections, and plans.
Those traces are not a process: they can be rewritten, they do not say
when to stop, and "I am not sure" is usually stored as failure or hidden
as success. PRP treats planning, execution, verification, revision, and
stopping as typed state.

## What "progressive" means here

In this protocol, progress is a new graph version justified by new
evidence. It is not another sample from the same prompt.

That distinction is the research object:

- Reflexion and Self-Refine continue because the model says so.
- Tree of Thoughts and Graph of Thoughts expand a search tree in model
  space.
- PRP revises a persistent graph only when a deterministic trigger,
  a finite budget, and a non-regressing comparison all hold.

The protocol intentionally leaves model routing, tool execution, storage, and
transport to an Agent runtime. This keeps the research claim testable: a
runtime can be replaced without changing the law that decides whether the
next graph version is allowed.

## Hypotheses

H1. Evidence-gated revision reduces undetected false completion relative
    to unconstrained self-reflection, because `INCONCLUSIVE` cannot be
    stored as `PASS`.

H2. Fingerprint-conservative reuse reduces redundant work without
    accepting a node whose public contract changed.

H3. Explicit stop reasons make failure modes comparable across models,
    because the process ends for a protocol reason rather than a hidden
    generation choice.

These hypotheses are not results. They are the questions this repository
is allowed to study.

## Method boundary

A fair test of PRP is a conformance and process evaluation:

- illegal transitions never occur
- terminal facts are not rewritten
- revision requires trigger and budget
- reuse is refused on missing facts
- stop reasons are stable under replay

A product test of a cloud-local agent is a different evaluation. Mixing
the two hides whether the protocol or the topology did the work.
