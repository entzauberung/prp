# Release Preparation

## Identity

- Project: Progressive Reasoning Protocol
- Package: `prp-protocol`
- Current version: `0.0.1`
- Distribution status: prepared locally, not published

## Scope

This repository contains only the Progressive strategy as a protocol
research artifact: its vocabulary, state machines, revision and conservative
reuse laws, specification notes, and conformance tests.

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
