# CLAUDE.md

> [!URGENT]
> **研究性项目 (Research Project)**
> 1. 本项目为 MVP（最小可行性产品），严禁过度工程化。
> 2. 你的所有思考过程和回复必须使用 **简体中文**。

## 1. 项目元数据 (Metadata)
- **核心目标**: 构建基于群体智能与进化算法的多Agent系统，解决复杂代码问题的自动化能力
- **项目类型**: MVP / 研究性项目
- **前端架构**: 无（纯后端系统）
- **后端架构**: Python 3.12 + asyncio
- **多Agent框架**: LangGraph (StateGraph + Native Tool Calling)
- **数据存储**: 无特殊存储需求（基于内存与文件系统）
- **版本管理**: Git

## 2. 常用命令 (Commands)


## 3. 标准作业程序 (Standard Operating Procedure)
> **Agent 必须严格遵守以下生命周期执行任务：**

### Phase 1: 规划与设计 (Planning)
1. **查阅规格 (Read Specs)**: 在撰写计划前，**必须**仔细阅读 `docs/` 下对应的功能规格说明书，确保理解架构师的设计意图。
2. **草稿 (Drafting)**: 如果有任何思路不清晰或需要推演复杂逻辑，**务必优先**在 `draft.md` 中撰写草稿，理清思路。
3. **计划 (Plan)**: 正式编码前，**必须**输出开发计划（Artifact: implementation_plan.md），内容必须严格包含：
   - **1.1 摘要 (Summary)**: 1-2句话的简单总结。
   - **1.2 审查点 (User Review Required)**: 明确列出整个计划中不清楚、需要用户审查和确认的部分。若无，请注明"无"。
   - **1.3 拟议变更 (Proposed Changes)**:
     - 以 **文件名 + 修改内容** 的形式列出。
     - 修改内容必须精确到 **函数/方法级别 (Function-level)**。
     - 明确标识 `[NEW]`, `[MODIFY]`, `[DELETE]`。
   - **1.4 验证计划 (Verification Plan)**: 具体描述如何验证修改是否成功（如具体的测试命令、预期日志输出等）。
4. **等待 (Wait)**: **必须** 暂停并等待用户审核开发计划。用户批准后方可进入下一阶段。

### Phase 2: 执行与验证 (Execution & Verification)
1. **编码 (Coding)**: 审核通过后，开始编写代码。
2. **验证 (Verify)**: 运行验证命令。
   - *失败*: 回到编码阶段修复，直到通过。
   - *成功*: 进入下一步。

### Phase 3: 收尾与交付 (Finalization)
1. **文档同步 (Docs Sync)**: **关键步骤**。
   - **必须** 使用 `doc-updater` Agent 检查并更新相关文档。
   - Agent 自动扫描范围：
     - `README.md` 项目说明
     - `docs/CODEMAPS/` 架构图
     - `docs/GUIDES/` 使用指南
   - 确保文档与代码实现保持一致。

## 4. 核心规则 (Rules)

### 4.1 代码开发规范 (Code Style)
- **类型系统**: 强制所有函数签名包含完整类型注解 (`Union`, `Dict`, `Optional` 等)。
- **文档**: 所有模块、类、方法必须包含 **中文 Docstring** (功能、参数、返回值、关键实现细节)。
- **MVP原则**:
  - **必须** 必须在`tests/`目录下编写测试代码。
  - **严禁** 使用默认参数掩盖仅需逻辑（必须显式传递关键参数）。
  - **必需** 运行时检查：关键维度、设备一致性必须通过 assertion 或 if 验证。
- **代码组织**:
  - 使用阶段化注释 (`# Phase 1`, `# Phase 2`) 组织复杂逻辑。
  - 接口返回值需包含完整诊断信息（输出、损失、统计），使用条件标志控制。
- **命名与依赖**:
  - 类名 `PascalCase`，变量描述性命名，私有变量前缀 `_`。
  - 导入顺序：标准库 → 第三方库 → 项目内部。
- **日志与错误处理**:
  - **必须**使用 `utils/logger_system.py` 提供的 `log_msg()` 和 `log_json()` 函数进行日志记录和错误处理。
  - 禁止直接使用 `print()` 进行调试或日志输出（除非在 logger 未初始化的 fallback 代码中）。
  - 使用 `log_msg("ERROR", "...")` 记录错误会自动抛出异常，无需额外 `raise`。
  - 结构化数据（如Agent调用信息、任务状态）应使用 `log_json()` 记录到 JSON 日志文件。
  - 示例：
    ```python
    from utils.logger_system import log_msg, log_json

    # 文本日志
    log_msg("INFO", "Agent 开始执行任务")
    log_msg("WARNING", "检测到潜在问题")
    log_msg("ERROR", "任务执行失败: xxx")  # 自动 raise Exception

    # JSON 日志
    log_json({"agent_name": "Agent1", "step": 3, "action": "tool_call"})
    ```
- **功能修改**:
  - **必须** 不考虑向后兼容，直接修改原文件。代码简洁性优先。

### 4.2 Git 工作流规范
- **提交**: 使用 `/commit` 命令自动生成并提交（详见 `.claude/commands/commit.md`）
- **严禁** 在 commit message 中添加任何 AI 标识

## 5. 上下文获取与迷途指南 (Context & Navigation)
> **当如果你感到困惑或需要更多信息时，请参考以下路径：**

- **想要了解项目整体目标与背景？**
  - -> 请阅读 `README.md`。
  - *注意*: 这里包含核心业务逻辑与项目定性。

- **想要了解具体的实施细节或技术规范？**
  - -> 请查阅 `docs/` 目录下的文档或者 `.claude`中的记忆。
  - *注意*: 这里包含特定模块的详细设计与 API 说明。

- **觉得当前任务复杂，需要梳理思路？**
  - -> 请立即使用 `draft.md`。
  - *行动*: 在此文件中列出你的思考过程、待办事项和草稿代码，清理工作记忆。

## 6. 输出规范
- 所有输出语言: **中文**
`