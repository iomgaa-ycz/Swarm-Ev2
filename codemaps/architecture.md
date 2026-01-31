# Swarm-Ev2 架构概览

**更新时间:** 2026-01-31
**项目阶段:** Phase 2 核心功能完成 (Phase 2.4 Orchestrator ✅)

---

## 系统架构

```
┌─────────────────────────────────────────────┐
│          入口层 (main.py)                     │  ← Phase 2.4 ✅
├─────────────────────────────────────────────┤
│      编排层 (Orchestrator 427行)              │  ← Phase 2.4 ✅
│  · 三阶段父节点选择 (初稿/修复/改进)             │
│  · Function Calling Review (GLM-4.6)        │
│  · 双向指标比较 (lower_is_better)             │
├─────────────────────────────────────────────┤
│       Agent 层 (CoderAgent 272行)            │  ← Phase 2.3 ✅
│  · 5次LLM重试 + 响应解析                       │
│  · Memory 机制 (Journal.generate_summary)   │
├─────────────────────────────────────────────┤
│       执行层 (Interpreter + Workspace)       │  ← Phase 2.1 ✅
├─────────────────────────────────────────────┤
│     核心数据层 (Node + Journal + Task)        │  ← Phase 1 ✅
├─────────────────────────────────────────────┤
│    后端抽象层 (OpenAI + Anthropic + GLM)     │  ← Phase 1 ✅
├─────────────────────────────────────────────┤
│  基础设施 (Config + Logger + FileUtils)      │  ← Phase 1 ✅
└─────────────────────────────────────────────┘
```

---

## 模块关系图

```
main.py
  ├── utils/config.py → Config
  ├── utils/workspace_builder.py → 数据集验证 + 工作空间构建
  ├── utils/logger_system.py → 日志初始化
  └── core/orchestrator.py → Orchestrator
        ├── agents/coder_agent.py → CoderAgent
        │     ├── agents/base_agent.py → BaseAgent 抽象
        │     ├── utils/prompt_builder.py → Prompt 构建
        │     ├── core/backend/ → LLM 查询
        │     ├── core/executor/interpreter.py → 代码执行
        │     └── utils/response.py → 响应解析
        ├── core/state/journal.py → Journal (DAG 管理)
        │     └── core/state/node.py → Node
        ├── core/executor/workspace.py → 工作空间管理
        └── core/backend/ → Function Calling Review
```

---

## 核心模块统计

| 层级 | 模块数 | 代码行数 | 状态 |
|------|--------|---------|------|
| 入口层 | 1 | 161 | ✅ main.py |
| 编排层 | 1 | 427 | ✅ Orchestrator |
| Agent 层 | 3 | 506 | ✅ BaseAgent + CoderAgent + PromptBuilder |
| 执行层 | 2 | 357 | ✅ Interpreter + Workspace |
| 数据层 | 3 | 476 | ✅ Node + Journal + Task |
| 后端层 | 4 | 522 | ✅ 统一接口 + OpenAI + Anthropic + Utils |
| 基础层 | 4 | 965 | ✅ Config + Logger + FileUtils + DataPreview |
| 工具层 | 3 | 373 | ✅ Metric + Response + WorkspaceBuilder |
| **总计** | **21** | **~3787** | **20 个核心模块 + main.py** |

---

## 数据流（端到端）

```
1. 环境准备
   .env + config/default.yaml → Config

2. 工作空间构建
   data_dir/ → workspace/input/ (symlink)
   workspace_builder.py → task_description

3. 组件初始化
   Config → Logger, Interpreter, PromptBuilder, CoderAgent, Journal, Orchestrator

4. 主循环 (Orchestrator.run)
   ┌─────────────────────────────────────────┐
   │ 每个 step (最多 50 步, 12 小时)            │
   │                                         │
   │ 1. _prepare_step()                      │
   │    清理 submission/ 目录                 │
   │                                         │
   │ 2. _select_parent_node()                │
   │    初稿/修复/改进 三阶段策略               │
   │                                         │
   │ 3. agent.generate(context)              │
   │    → CoderAgent 生成代码                 │
   │                                         │
   │ 4. _execute_code()                      │
   │    → Workspace 路径重写                  │
   │    → Interpreter 执行                    │
   │                                         │
   │ 5. _review_node()                       │
   │    → Function Calling (GLM-4.6)        │
   │    → 解析 submit_review 工具调用         │
   │                                         │
   │ 6. journal.append(node)                 │
   │    _update_best_node(node)              │
   │    _save_best_solution()                │
   │                                         │
   └─────────────────────────────────────────┘

5. 结果输出
   workspace/best_solution/ → solution.py + submission.csv
   logs/ → system.log + metrics.json
```

---

## 三阶段父节点选择策略

```
_select_parent_node()
│
├── Phase 1: 初稿模式 (return None)
│   条件: len(journal.draft_nodes) < config.search.num_drafts
│   效果: Agent 生成全新方案（无历史上下文）
│
├── Phase 2: 修复模式 (return buggy_leaf)
│   条件: random() < config.search.debug_prob
│   操作: journal.build_dag() → 查找 buggy 叶子节点
│   效果: Agent 修复 bug（带错误输出上下文）
│
└── Phase 3: 改进模式 (return best_node)
    条件: 默认
    操作: journal.get_best_node(only_good=True)
    效果: Agent 改进最佳方案（带正常输出上下文）
```

---

## Function Calling Review 机制

```
_review_node(node)
│
├── 构建 Review 消息:
│   ├── 任务描述 (task_desc)
│   ├── 代码 (node.code)
│   └── 执行输出 (node.term_out)
│
├── 调用 LLM (glm-4.6):
│   model: config.llm.feedback.model
│   tools: [submit_review]
│   tool_choice: {"type": "function", "function": {"name": "submit_review"}}
│
├── submit_review schema:
│   ├── is_bug: bool          是否有 bug
│   ├── has_csv_submission: bool  是否生成 submission.csv
│   ├── summary: string       2-3 句话摘要
│   ├── metric: number|null   验证集指标值
│   └── lower_is_better: bool 指标方向 (RMSE=true, Accuracy=false)
│
└── 更新节点:
    ├── node.analysis = summary
    ├── node.is_buggy = is_bug || exc_type != None
    ├── node.metric_value = metric
    └── node.lower_is_better = lower_is_better
```

---

## 双向指标比较逻辑

```
_update_best_node(node)
│
├── 过滤: is_buggy=True 或 metric_value=None → 跳过
│
├── 初始化: best_node=None → 直接设置
│
└── 比较:
    ├── lower_is_better=True  (RMSE, MAE):  new < current → 更新
    └── lower_is_better=False (Accuracy, F1): new > current → 更新
```

---

## 配置系统优先级

```
高 ──────────────────────────────── 低

CLI 参数          环境变量            YAML 配置
--key=value       export VAR=val     config/default.yaml
     │               │                    │
     └───────────────┼────────────────────┘
                     ↓
              OmegaConf.merge()
                     ↓
               Config 对象
```

---

## 工作空间目录结构

```
workspace/
├── input/          # 输入数据 (symlink → data_dir/)
├── working/        # 临时执行目录 (Interpreter)
├── submission/     # 预测结果 (submission_{node_id}.csv)
├── archives/       # 归档文件 (node_{node_id}.zip)
├── best_solution/  # 最佳方案 (solution.py + submission.csv)
└── logs/           # 日志输出 (system.log + metrics.json)
```

---

## Phase 实施状态

| Phase | 名称 | 状态 | 核心交付物 |
|-------|------|------|-----------|
| 1 | 基础设施 | ✅ | Config, Logger, FileUtils, Node, Journal, Task, Backend |
| 2.1 | 执行层 | ✅ | Interpreter, WorkspaceManager |
| 2.2 | 工具增强 | ✅ | DataPreview, Metric, Response, PromptBuilder |
| 2.3 | CoderAgent | ✅ | BaseAgent, CoderAgent (5次重试, 92% 覆盖) |
| 2.4 | Orchestrator | ✅ | 编排器 (三阶段选择, Function Calling Review) |
| 3 | 搜索算法 | 🔴 | MCTS, GA, ParallelEvaluator |
| 4 | 进化机制 | 🔴 | AgentEvolution, SolutionEvolution |
| 5 | 端到端 | 🔴 | MLE-Bench 适配, 性能测试 |

---

## 关联文档

| 文档 | 路径 |
|------|------|
| 后端详解 | codemaps/backend.md |
| 数据流详解 | codemaps/data.md |
| 开发规范 | CLAUDE.md |
| 项目说明 | README.md |
