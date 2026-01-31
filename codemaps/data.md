# 数据流与配置管理

**更新时间:** 2026-01-31
**范围:** 配置系统、工作空间、日志系统

---

## 配置优先级

```
高 ──────────────────────────────── 低

  CLI 参数          环境变量            YAML 配置
  --key=value       export VAR=val     config/default.yaml
       │               │                    │
       │    ┌──────────┤                    │
       │    │ 系统环境变量                    │
       │    │ (export VAR=val)              │
       │    │    ↑                          │
       │    │  .env 文件                     │
       │    │  (override=False)             │
       │    └──────────┘                    │
       │               │                    │
       └───────────────┼────────────────────┘
                       ↓
                OmegaConf.merge()
                       ↓
                Config 对象
```

---

## 配置加载流程

```
load_config()
│
├── load_dotenv(override=False)
│   .env 文件不覆盖系统环境变量
│
├── OmegaConf.load("config/default.yaml")
│   YAML 配置，支持 ${env:VAR, default} 插值
│
├── OmegaConf.from_cli()
│   CLI 参数最高优先级
│
├── validate_config()
│   ├── 必填字段检查 (data_dir, desc/goal, provider)
│   ├── 路径解析 (resolve)
│   ├── 目录创建 (mkdir)
│   ├── exp_name 生成 (YYYYMMDD_HHMMSS_xxxx)
│   └── API Key 检查
│
└── return Config
```

---

## 配置文件结构

### config/default.yaml

```yaml
project:
  name: "Swarm-Ev2"
  version: "0.1.0"
  workspace_dir: "./workspace"
  log_dir: "./logs"
  exp_name: null  # 自动生成

data:
  data_dir: null              # [必填] 数据目录
  desc_file: null             # 与 goal 二选一
  goal: null                  # 与 desc_file 二选一
  eval: null                  # 评估指标
  preprocess_data: true
  copy_data: false            # false=symlink, true=复制

llm:
  code:                       # CoderAgent 使用
    provider: ${env:LLM_PROVIDER, "openai"}  # [必填]
    model: ${env:LLM_MODEL, "gpt-4-turbo"}
    temperature: 0.5
    api_key: ${env:OPENAI_API_KEY}
    base_url: ${env:OPENAI_BASE_URL, "https://api.openai.com/v1"}
    max_tokens: ${env:MAX_TOKENS, null}

  feedback:                   # Orchestrator Review 使用
    provider: ${env:LLM_PROVIDER, "openai"}
    model: ${env:LLM_MODEL, "glm-4.6"}  # Function Calling
    temperature: 0.5
    api_key: ${env:OPENAI_API_KEY}
    base_url: ${env:OPENAI_BASE_URL, "https://open.bigmodel.cn/api/coding/paas/v4"}
    max_tokens: ${env:MAX_TOKENS, null}

execution:
  timeout: 3600               # 单次超时 (秒)
  agent_file_name: "runfile.py"
  format_tb_ipython: false

agent:
  max_steps: 50               # 最大迭代步数
  time_limit: 43200           # 总时间限制 (12 小时)
  k_fold_validation: 5
  expose_prediction: false
  data_preview: true
  convert_system_to_user: false

search:
  strategy: "mcts"
  max_debug_depth: 3
  debug_prob: 0.5             # 修复模式触发概率
  num_drafts: 5               # 初稿数量
  parallel_num: 3

logging:
  level: "INFO"
  console_output: true
  file_output: true
```

### .env.example

```bash
# LLM 配置
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-openai-api-key-here

# 可选: API 端点配置
# OPENAI_BASE_URL=https://api.openai.com/v1
# MAX_TOKENS=4096

# 可选: GLM (智谱 AI)
# OPENAI_BASE_URL=https://open.bigmodel.cn/api/coding/paas/v4

# 可选: 其他第三方 OpenAI 兼容 API
# Moonshot:   OPENAI_BASE_URL=https://api.moonshot.cn/v1
# DeepSeek:   OPENAI_BASE_URL=https://api.deepseek.com/v1
```

---

## 工作空间目录结构

```
workspace/                    # project.workspace_dir
├── input/                    # 输入数据 (只读)
│   ├── train.csv            # symlink → data_dir/train.csv
│   ├── test.csv             # symlink → data_dir/test.csv
│   └── ...
│
├── working/                  # Agent 临时工作目录
│   └── _temp_script.py      # Interpreter 执行文件
│
├── submission/               # 预测结果目录
│   ├── submission_{node_id}.csv  # 各节点的提交文件
│   └── ...
│
├── archives/                 # 归档目录
│   ├── node_{node_id}.zip   # 节点归档 (solution.py + submission.csv)
│   └── ...
│
└── best_solution/            # 最佳方案 (Orchestrator 维护)
    ├── solution.py           # 最佳代码
    └── submission.csv        # 最佳提交文件
```

### 数据准备模式

```
copy_data: false (默认)
─────────────────────────────
workspace/input/ → symlink → data_dir/
  优点: 节省空间, 只读保护
  适用: macOS/Linux

copy_data: true
─────────────────────────────
workspace/input/ ← 复制 ← data_dir/
  优点: 完全隔离, 不影响源数据
  适用: Windows 环境
```

---

## 日志系统

### 双通道输出

```
logs/                        # project.log_dir
├── system.log               # 文本日志 (追加写入)
└── metrics.json             # 结构化日志 (完整重写)
```

### system.log 格式

```
[2026-01-31 10:00:00] [INFO] 加载配置文件: config/default.yaml
[2026-01-31 10:00:01] [INFO] Orchestrator 初始化完成
[2026-01-31 10:00:01] [INFO] === Step 1/50 ===
[2026-01-31 10:00:02] [INFO] 查询 LLM: model=gpt-4-turbo
[2026-01-31 10:00:10] [INFO] Function Calling 响应: submit_review
[2026-01-31 10:00:10] [INFO] Review 完成: metric=0.85
[2026-01-31 10:00:10] [INFO] 新的最佳节点: abc123, metric=0.85
```

### metrics.json 格式

```json
[
    {
        "step": 1,
        "action": "draft",
        "metric": 0.72,
        "exec_time": 45.2
    },
    {
        "step": 2,
        "action": "improve",
        "metric": 0.85,
        "exec_time": 38.7
    }
]
```

---

## 完整数据流 (端到端)

```
用户输入                       系统输出
───────                       ───────

data_dir/ ──→ workspace/input/ (symlink)
              ↓
config.yaml ──→ Config 对象
              ↓
.env ──→ API Keys ──→ LLM Backend
              ↓
        ┌── main.py ──────────────────────────┐
        │                                     │
        │ 1. load_config()                    │
        │ 2. build_workspace()                │
        │ 3. init_logger()                    │
        │ 4. 初始化组件:                       │
        │    - Interpreter                    │
        │    - PromptBuilder                  │
        │    - CoderAgent                     │
        │    - Journal                        │
        │    - Orchestrator                   │
        │                                     │
        │ 5. orchestrator.run()               │
        │    ┌────────────────────────────┐   │
        │    │  每个 step:                 │   │
        │    │  1. _select_parent_node()  │   │
        │    │  2. agent.generate()       │   │
        │    │  3. _execute_code()        │   │
        │    │  4. _review_node()         │   │
        │    │  5. journal.append()       │   │
        │    │  6. _update_best_node()    │   │
        │    │  7. _save_best_solution()  │   │
        │    └────────────────────────────┘   │
        │                                     │
        │ 6. 结果展示                          │
        │                                     │
        └─────────────────────────────────────┘
              ↓
workspace/best_solution/ ──→ 最终结果
              ↓
logs/system.log ──→ 文本日志
logs/metrics.json ──→ 结构化日志
```

---

## Orchestrator 单步数据流

```
Step N 开始
    ↓
_prepare_step()
    清空 submission/ 目录
    ↓
_select_parent_node()
    ├── draft 不足 → None (初稿)
    ├── random < debug_prob → buggy_leaf (修复)
    └── 默认 → best_node (改进)
    ↓
AgentContext(
    task_type="explore",
    parent_node=selected_node,
    journal=self.journal,
    config=self.config,
    start_time=self.start_time,
    current_step=self.current_step
)
    ↓
agent.generate(context)
    ├── 1. _generate_data_preview()
    ├── 2. journal.generate_summary() (Memory)
    ├── 3. _calculate_remaining() (时间/步数)
    ├── 4. prompt_builder.build_explore_prompt()
    ├── 5. _call_llm_with_retry(max_retries=5)
    ├── 6. _parse_response_with_retry()
    ├── 7. interpreter.run(code)
    └── 8. return AgentResult(node)
    ↓
_execute_code(node.code, node.id)
    ├── workspace.rewrite_submission_path()
    └── interpreter.run()
    ↓
_review_node(node)
    ├── _build_review_messages()
    ├── backend.query(
    │     model=config.llm.feedback.model,
    │     tools=[submit_review],
    │     tool_choice={"type": "function", ...}
    │   )
    ├── json.loads(response)
    └── 更新 node 字段:
        ├── node.analysis = summary
        ├── node.is_buggy = is_bug || exc_type != None
        ├── node.metric_value = metric
        └── node.lower_is_better = lower_is_better
    ↓
journal.append(node)
    node.step = len(journal.nodes)
    ↓
_update_best_node(node)
    if node.is_buggy or metric_value is None:
        跳过
    else:
        ├── lower_is_better=True:  new < old → 更新
        └── lower_is_better=False: new > old → 更新
    ↓
_save_best_solution()
    workspace/best_solution/
    ├── solution.py (最佳代码)
    └── submission.csv (最佳提交)
```

---

## 配置系统数据流

```
.env 文件
    ↓ load_dotenv(override=False)
os.environ
    ↓ OmegaConf resolver: ${env:VAR, default}
config/default.yaml
    ↓ OmegaConf.load()
DictConfig (base)
    ↓ OmegaConf.merge(base, cli)
DictConfig (merged)
    ↓ validate_config()
    ├── 必填字段检查
    ├── Provider 验证 (openai | anthropic)
    ├── 路径解析
    ├── 目录创建
    ├── exp_name 生成
    └── API Key 检查
    ↓ OmegaConf.to_container()
Config(@dataclass)
```

---

## 配置验证规则

| 规则 | 检查内容 | 失败行为 |
|------|---------|---------|
| 必填: data_dir | `cfg.data.data_dir is None` | ValueError |
| 必填: desc/goal | 两者均为 None | ValueError |
| 必填: provider | `not in {"openai", "anthropic"}` | ValueError |
| 路径: data_dir | 目录不存在 | ValueError |
| 路径: desc_file | 文件不存在 | ValueError |
| 目录: workspace | 不存在 | 自动创建 |
| 目录: log_dir | 不存在 | 自动创建 |
| 名称: exp_name | 为 None | 自动生成 YYYYMMDD_HHMMSS_xxxx |
| 密钥: api_key | 未解析的 `${env:}` | 记录 WARNING |

---

## 关键配置值说明

### Orchestrator 相关

| 配置项 | 值 | 说明 |
|--------|-----|------|
| `agent.max_steps` | 50 | 主循环最大步数 |
| `agent.time_limit` | 43200 | 总时间限制 (12 小时) |
| `search.num_drafts` | 5 | 初稿数量 |
| `search.debug_prob` | 0.5 | 修复模式触发概率 |
| `llm.feedback.model` | "glm-4.6" | Review 模型 (Function Calling) |
| `execution.timeout` | 3600 | 单次执行超时 (1 小时) |

### 双阶段 LLM 配置

```
llm.code: CoderAgent 使用
├── model: gpt-4-turbo
├── 用途: 生成 ML 代码
└── 调用方: CoderAgent._call_llm_with_retry()

llm.feedback: Orchestrator Review 使用
├── model: glm-4.6 (支持 Function Calling)
├── 用途: 评估代码执行结果
├── 调用方: Orchestrator._review_node()
└── base_url: https://open.bigmodel.cn/api/coding/paas/v4
```

---

## 使用示例

### 基础用法

```bash
# 使用默认配置
conda run -n Swarm-Evo python main.py \
  --data.data_dir=./datasets/titanic
```

### 完整覆盖

```bash
conda run -n Swarm-Evo python main.py \
  --data.data_dir=./datasets/house-prices \
  --data.goal="预测房价" \
  --llm.code.model=gpt-3.5-turbo \
  --llm.feedback.model=glm-4.6 \
  --agent.max_steps=30 \
  --agent.time_limit=7200 \
  --search.num_drafts=3 \
  --search.debug_prob=0.3
```

### Python 代码

```python
from utils.config import load_config
from utils.workspace_builder import build_workspace
from utils.logger_system import init_logger
from core.orchestrator import Orchestrator
from agents.coder_agent import CoderAgent
from core.state import Journal

# 1. 加载配置
config = load_config(use_cli=True)

# 2. 构建工作空间
task_desc = build_workspace(
    data_dir=config.data.data_dir,
    workspace_dir=config.project.workspace_dir,
    copy_data=config.data.copy_data
)

# 3. 初始化日志
init_logger(str(config.project.log_dir))

# 4. 初始化组件
agent = CoderAgent(...)
journal = Journal()
orchestrator = Orchestrator(
    agent=agent,
    config=config,
    journal=journal,
    task_desc=task_desc
)

# 5. 运行
best_node = orchestrator.run()
```

---

## 关联文档

| 文档 | 路径 |
|------|------|
| 架构概览 | codemaps/architecture.md |
| 后端详解 | codemaps/backend.md |
| 开发规范 | CLAUDE.md |
