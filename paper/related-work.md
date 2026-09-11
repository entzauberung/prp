# Related work

These papers are neighbouring research. They are not sources of this
protocol, and listing them is not a reproduction or benchmark claim.

## Acting and reflecting

- Yao et al., *ReAct*, 2023. https://arxiv.org/abs/2210.03629
  Interleaves thought and action. PRP stores action and evidence as
  facts; thought is not a protocol object.

- Shinn et al., *Reflexion*, 2023. https://arxiv.org/abs/2303.11366
  Verbal reinforcement from a model's own reflection. PRP forbids
  revision without a deterministic trigger and a declared ceiling.

- Madaan et al., *Self-Refine*, 2023. https://arxiv.org/abs/2303.17651
  Iterative refinement from self-feedback. PRP compares rounds on
  public evidence, not on a self-score.

## Search over thoughts

- Yao et al., *Tree of Thoughts*, 2023. https://arxiv.org/abs/2305.10601
- Besta et al., *Graph of Thoughts*, 2024. https://arxiv.org/abs/2308.09687

Both search in thought space. PRP searches in a versioned execution
graph with isolated writes and three-way merge as an implementation
technique, not as a thought operator.

## Tools and software agents

- Schick et al., *Toolformer*, 2023. https://arxiv.org/abs/2302.04761
- Yang et al., *SWE-agent*, 2024. https://arxiv.org/abs/2405.15793

Tool use and agent-computer interfaces are orthogonal. PRP does not
define a tool set. It defines when a result may be accepted, reused,
revised, or stopped.

## Position

PRP's bet is that the scarce object is not a better prompt loop, but a
process whose states can be audited after the model is gone.
