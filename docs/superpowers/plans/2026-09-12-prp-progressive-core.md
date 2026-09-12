# PRP Progressive Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 收敛 PRP 为只描述 Progressive 渐进式推理的独立、可验证协议，并补齐其规范、参考实现和 README。

**Architecture:** PRP 保持无依赖纯函数内核。协议只定义公开事实、版本化 round、证据比较、受限修订、保守复用和停止原因；DIRECT/CASCADE/PLANNED 作为实现策略，而不是 PRP 的研究对象。

**Tech Stack:** Python 3.12+, dependency-free reference kernel, pytest, Markdown.

**Spec:** `spec/00-identity.md` through `spec/04-non-goals.md` and `paper/thesis.md`.

## Global Constraints

- PRP 不增加 HTTP、数据库、Provider、Bridge、工具运行时或调度器。
- 不存储私有模型推理文本，只存储可审计公开事实。
- 终态不可回退；修订必须产生新 graph version。
- `INCONCLUSIVE` 不得隐式转换为 `PASS` 或 `FAIL`。

### Task 1: 收敛协议身份和规范

**Files:**
- Modify: `README.md`
- Modify: `spec/00-identity.md`
- Modify: `spec/04-non-goals.md`
- Modify: `paper/thesis.md`

- [ ] 明确 Progressive 是唯一研究核心；把其他策略标为 runtime concerns。
- [ ] 增加最小 round/evidence/comparison/revision 生命周期说明。
- [ ] 运行 Markdown 文本检查，确保没有把产品能力宣称为 PRP 能力。

### Task 2: 补齐参考内核的公开协议入口

**Files:**
- Create: `src/prp/progressive.py`
- Modify: `src/prp/__init__.py`
- Create: `tests/test_progressive_api.py`

- [ ] 暴露一个稳定的 `ProgressiveDecision` 和 `decide_progressive` 入口。
- [ ] 用现有 revision law 组合实现，保持纯函数和闭合集合。
- [ ] 为 PASS、INCONCLUSIVE、FAIL、预算缺失、revision limit 编写 conformance tests。

### Task 3: 验证和发布说明

**Files:**
- Modify: `RELEASE.md`

- [ ] 运行 `pytest -q`。
- [ ] 运行 `python -m compileall src tests`。
- [ ] 记录当前版本是 reference kernel，不宣称生产就绪或 benchmark 结果。
