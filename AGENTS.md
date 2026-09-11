# PRP Protocol

This repository is the Progressive Reasoning Protocol.

It is not a cloud-local agent, not a scheduler, and not a CLI product.
Do not add HTTP servers, provider adapters, SQLite, tool runtimes, or
bridge clients here. Those belong in the agent implementation.

Work in this repository must stay:

- protocol facts, state machines, and revision laws
- conformance tests for those laws
- research notes that do not claim unimplemented results

The current agent implementation lives at `/home/ognev/prp` and was
copied, unmodified, to `/home/ognev/cloud-local-agent`.
