# 连续施工蓝图执行协议

## 运行目标

连续执行 ai/ 中的完整施工蓝图。

执行器只根据 ai/STATE.md 找到当前 WO 和当前 ST，不把全部蓝图一次性读取到上下文。

## 启动顺序

1. 读取 AGENTS.md。
2. 读取 ai/CONTROL.md。
3. 读取 ai/STATE.md。
4. 读取本文件。
5. 根据 ai/STATE.md 的 CURRENT_WO 和 CURRENT_ST 读取对应的工单文件和子任务文件。
6. 只执行当前子任务。

## 连续推进

完成当前 ST 后，不要等待用户确认，也不要结束会话。

必须按以下顺序执行：

1. 进行当前 ST 文件中列出的定向验证。
2. 最多执行两轮“修改 -> 定向验证 -> 修复”。
3. 覆盖写入 ai/LATEST-REPORT.md。
4. 覆盖更新 ai/STATE.md。
5. 检查下一个 ST 的前置依赖。
6. 如果依赖已满足，读取下一个 ST 文件并继续执行。
7. 当前 WO 的所有 ST 完成后，继续读取下一个满足依赖的 WO。
8. 所有 WO 完成后，将 ai/STATE.md 标记为 COMPLETED，并停止。

## 状态文件规则

ai/STATE.md 是唯一的当前进度指针。

状态必须使用以下值之一：

- READY
- RUNNING
- BLOCKED
- COMPLETED

每次更新 STATE.md 必须写明：

- BLUEPRINT_ID
- STATUS
- CURRENT_WO
- CURRENT_ST
- COMPLETED_WORK_ORDERS
- COMPLETED_TASKS
- NEXT_WO
- NEXT_ST
- BLOCKER
- LAST_REPORT
- LAST_UPDATED

不要通过聊天历史判断当前进度。

## 资源规则

R0：

- 不运行命令。
- 只读取和修改当前任务允许的文件。

R1：

- 只运行当前 ST 文件中明确列出的定向验证。
- 一次只运行一个命令。
- 不允许并行启动测试、构建或服务。
- 命令超过任务规定时间后停止并报告。

R2：

- 全量测试、全量构建、发布构建、桌面打包和数据库迁移。
- R2 必须停止并等待用户明确批准。
- 未获得批准不得执行 R2。

## 禁止的重型操作

除非当前 ST 明确属于 R2 且已获得批准，否则禁止：

- npm install
- npm ci
- npm test
- npm run build
- pnpm install
- pnpm test
- pnpm build
- cargo build
- cargo test
- cargo check --workspace
- cargo test --workspace
- cargo clean
- tauri build
- Docker build
- 启动开发服务器
- watch 模式
- release 构建
- workspace 全量检查
- 同时启动多个 Cargo、Node 或测试进程

Rust 定向验证如果被当前 ST 明确允许，必须优先使用：

CARGO_BUILD_JOBS=2

## 失败与停止

遇到以下情况时，立即覆盖更新 STATE.md 为 BLOCKED，并覆盖写入 LATEST-REPORT.md：

- 需要修改当前 ST 允许范围之外的文件。
- 需要修改依赖、锁文件、数据结构或架构。
- 同一个测试失败连续出现两次。
- 两轮修复后仍未通过。
- 发生 OOM、系统明显卡顿、进程被系统杀死或磁盘空间不足。
- 当前工单与实际代码不一致。
- 前置工单没有真正完成。
- 当前 ST 没有明确的允许验证命令。
- 需要执行 R2 操作。

BLOCKED 后不得自行继续，不得跳过当前任务，不得进入下一个工单。

## 完成报告

每个 ST 完成后覆盖写入 ai/LATEST-REPORT.md，内容必须包含：

- 完成的 WO/ST。
- 修改文件。
- 实际执行的命令。
- 每条命令的结果。
- 未执行的验证。
- 与蓝图的偏差。
- 剩余风险。
- 下一 WO/ST。

不得提交 Git，不得推送，不得自动开始蓝图之外的任务。
