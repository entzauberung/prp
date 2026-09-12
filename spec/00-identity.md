# 0. Identity

## Object

Progressive Reasoning Protocol (PRP) specifies how a reasoning process
advances through evidence-backed graph versions. Progressive revision is the
protocol object. It does not specify where the process runs, which model is
called, or which operating-system tools are available.

## Normative layers

1. Vocabulary: closed enumerations for status, verdict, comparison, and stop.
2. Facts: immutable records whose shape is part of the protocol.
3. Machines: legal status transitions. Illegal transitions are errors.
4. Revision: whether a new graph version may be created.

An implementation may add transport, storage, and tools. Those extensions
are not PRP.

## Closed world

Unknown fields are protocol errors. A fact that cannot be decided is
`INCONCLUSIVE`. Silence is not success.

## Non-identity

The following names are out of this repository's identity:

- cloud-local agent
- bridge client
- provider adapter
- native HTTP API
- workspace sandbox

They may implement PRP. They are not PRP.
