# Release Preparation

## 0.0.2 (2026-09-12)

- Added the public `prp.decide_progressive` entry point.
- Added comparison-aware Progressive decisions and conformance coverage.
- Clarified the independent protocol identity and Iskrov Agent boundary.

## 0.0.1 working-tree completion (2026-09-12)

- Added the stable `prp.decide_progressive` reference entry point.
- Clarified that Progressive revision is the protocol core and other routing
  strategies belong to Agent runtimes.
- Expanded the identity, non-goals, thesis, and README boundaries.
- Conformance suite: 22 tests passing locally.

## Identity

- Project: Progressive Reasoning Protocol
- Package: `prp-protocol`
- Current version: `0.0.2`
- Distribution status: prepared locally, not published

## Scope

This repository contains only the Progressive strategy as a protocol
research artifact: its vocabulary, state machines, revision and conservative
reuse laws, specification notes, and conformance tests.

The reference entry point is `prp.decide_progressive`. The package is a
small executable protocol kernel, not a production Agent and not a benchmark
result.

It intentionally does not contain the cloud-local agent's HTTP server,
SQLite store, provider adapters, bridge client, workspace tools, or CLI.

## Pre-publish gate

- [ ] Review the protocol wording and normative/non-normative boundaries
- [ ] Run the complete offline test suite
- [ ] Run `ruff check .`
- [ ] Build sdist and wheel in a clean environment
- [ ] Inspect archive contents for credentials, caches, and local files
- [ ] Set the final public repository URL in `pyproject.toml`
- [ ] Add a versioned changelog entry
- [ ] Obtain explicit approval before any push or package publication

No publish or push is performed by this preparation.
