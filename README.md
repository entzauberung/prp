# Progressive Reasoning Protocol

> 证据门控的渐进式修订协议。

Progressive Reasoning Protocol（PRP）是一个独立的协议研究项目和 Python 参考内核，用来规定 Agent 如何依据公开事实、确定性验证和有限预算推进到下一版执行图。

PRP 研究的是“什么时候允许继续修订”，不是“让模型多想几次”。它不定义模型、Provider、HTTP、数据库、工具、Bridge 或云端拓扑。如果你需要完整的云端 Agent 产品，请使用 [Iskrov Agent](https://github.com/entzauberung/iskrov-agent)。

## 它解决什么问题

普通模型调用通常是：

```text
发送请求 -> 获得回答
```

可审计的 Agent 还必须回答：

- 当前结果是否有公开证据支持；
- 失败后是否真的允许继续；
- 新一轮是否带来了增益；
- 历史结果是否可以安全复用；
- 什么时候必须停止。

PRP 将这些决定表示为公开、持久、可重放的协议事实，不保存私有思维链。

## 最小生命周期

```text
第 N 个 graph version
        |
        v
公开事实 -> 确定性验证 -> 与上一轮比较
                              |
       PASS / NO_GAIN / REGRESSION / BUDGET -> STOP
                              |
       明确触发 + revision budget -> 第 N+1 个 graph version
```

一次修订必须同时满足：

1. 有记录的确定性失败、可比较的不确定结果或可重试 Provider 失败；
2. 明确声明了有限的修订预算；
3. token、deadline 等资源仍然可用；
4. 比较结果没有出现 `NO_GAIN` 或 `REGRESSION`。

`INCONCLUSIVE` 是独立的结论，不能被默认为成功或失败。

## 协议定义的内容

- Run、WorkUnit、Attempt、Artifact、Evidence 和 Event 等公开事实；
- 严格的运行、工作单元和尝试状态机；
- 不可回退的终态和不可改写的历史版本；
- 基于 lineage、fingerprint、依赖事实和证据的保守复用；
- 明确的 revision decision 和 stop reason。

`DIRECT`、`CASCADE` 和 `PLANNED` 可以是运行时的调度策略，但不是 PRP 的研究核心。PRP 的核心只有 Progressive revision。

## 参考实现

仓库提供无运行时依赖的 Python 参考内核：

```python
from prp import VerificationResult, decide_progressive

decision = decide_progressive(
    verification_result=VerificationResult.FAIL,
    revision_count=0,
    graph_version=1,
    max_plan_revisions=3,
)

assert decision.next_graph_version == 2
```

该函数是纯决策：不调用模型、不写文件、不访问网络，也不修改数据库。

## 仓库结构

```text
spec/     协议身份、事实、状态机和边界规范
paper/    论文思路、研究假设和相关工作
src/prp/  无依赖参考内核
tests/    协议法则的 conformance tests
docs/     实施计划和补充说明
```

本仓库不包含 HTTP 服务、数据库、Provider 适配器、Bridge、本地工作区、工具执行器或调度器。

## 研究状态

当前版本 `0.0.2` 是协议参考实现和可执行法则集合，不是生产 Agent，也不声称拥有 benchmark 优势、模型质量优势或生产 SLA。研究假设见 [paper/thesis.md](paper/thesis.md)，相关论文仅作为研究背景。

## 开发

```bash
python -m pip install -e '.[dev]'
pytest -q
ruff check .
```

要求 Python 3.12 或更高版本。

## 英文说明

英文完整说明见 [README.en.md](README.en.md)。

## 许可证

PRP 使用 [Apache-2.0](LICENSE-APACHE) 许可证。署名和品牌边界见 [NOTICE](NOTICE)。
