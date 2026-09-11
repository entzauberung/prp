# Progressive Reasoning Protocol

> 推理不是一次回答，而是可验证、可停止、不可改写历史的过程。

PRP 是一份协议，不是一个云原生 Agent 产品。

本仓库只研究一件事：一次任务如何被计划、执行、验证、修订和停止。
它不调度机器，不调用模型供应商，不在本机跑工具。

当前身份：`0.0.1` · Python `3.12+` · 无运行时依赖 · Apache-2.0

## 为什么要拆开

现有代码仓库已经是一个 cloud-local agent：

```text
云端：模型、规划、审批、证据、合并
本机：封闭工具，无模型
```

那是产品拓扑。PRP 不是这个拓扑。

协议回答的问题是：

- 这一步为什么可以执行
- 什么证据证明它完成
- 失败后是重试、切换、修订还是停止
- 修订是否产生了新证据，而不是再问模型一次
- 重启后哪些事实仍然成立

位置（CLOUD / BRIDGE / LOCAL）是实现选择。协议在任何位置上都相同。

## 核心主张

1. **事实，不是思路。** Run、WorkUnit、Attempt、Artifact、Evidence、Event 是协议对象。Chain-of-thought 不是。
2. **INCONCLUSIVE 是一等结果。** 不能判定不得记为通过，也不得假装成失败。
3. **修订必须有新证据。** Progressive 不是“再问一遍”。没有触发、没有预算、没有增益，就不能开新的 graph version。
4. **历史不可改写。** 终端状态不再回到运行态。修订创建新的 WorkUnit 和新的 graph version。
5. **复用必须保守。** lineage、fingerprint 或依赖事实缺失时，只能重算，不能复用。
6. **停止由规则决定。** 预算、取消、通过、无增益、回退，都是协议事件，不是模型心情。

## 四种策略

控制强度单向递增，禁止降级：

| 策略 | 强度 | 协议行为 |
|---|---|---|
| `DIRECT` | 0 | 一个 WorkUnit，一次 Attempt，一次验证 |
| `CASCADE` | 1 | 仅在可重试失败时进入下一 profile |
| `PLANNED` | 2 | Planner 提案 DAG，Worker 按依赖执行 |
| `PROGRESSIVE` | 3 | 执行、合并、验证、复用，有限修订 |

## 本仓库有什么

```text
spec/     规范性协议文本
paper/    研究定位与相关工作，不声称已完成实验
src/prp/  纯协议核：词表、状态机、修订法则
tests/    对法则的符合性测试
```

没有 HTTP、SQLite、CLI、provider、sandbox、bridge。

## 不是什么

- 不是 cloud-local agent
- 不是 MCP / A2A / Codex / Claude Code 兼容层
- 不是模型训练或评测排行榜
- 不是对 ReAct、Tree of Thoughts、Reflexion 的复现声明

一份 runtime 可以实现 PRP。实现不是协议本身。

## 最小核

```python
from prp.revision import decide_revision
from prp.vocabulary import RevisionStopReason, RunStatus, VerificationResult

decision = decide_revision(
    run_status=RunStatus.RUNNING,
    verification_result=VerificationResult.PASS,
    revision_count=0,
    graph_version=1,
    max_plan_revisions=3,
)
assert decision.stop_reason is RevisionStopReason.PASS
```

## 许可证

Apache-2.0。见 [LICENSE-APACHE](LICENSE-APACHE) 与 [NOTICE](NOTICE)。
