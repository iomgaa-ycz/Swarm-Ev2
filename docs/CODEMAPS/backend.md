# 后端模块详细说明

**Last Updated:** 2026-01-31
**模块范围:** utils/, core/state/, core/backend/, core/executor/, agents/, config/, tests/

---

## 1. 模块概览

| 模块 | 文件 | 职责 | 状态 |
|------|------|------|------|
| 配置系统 | `utils/config.py` | OmegaConf 配置加载与验证 | 已完成 |
| 日志系统 | `utils/logger_system.py` | 双通道日志输出 | 已完成 |
| 文件工具 | `utils/file_utils.py` | 目录复制/链接 | 已完成 |
| **Node 数据类** | `core/state/node.py` | 解决方案 DAG 节点 | **已完成** |
| **Journal 数据类** | `core/state/journal.py` | DAG 容器与查询 | **已完成** |
| **Task 数据类** | `core/state/task.py` | Agent 任务定义 | **已完成** |
| **后端抽象层** | `core/backend/__init__.py` | 统一 LLM 查询接口 | **已完成** |
| **OpenAI 后端** | `core/backend/backend_openai.py` | OpenAI + GLM 支持 | **已完成** |
| **Anthropic 后端** | `core/backend/backend_anthropic.py` | Claude 系列支持 | **已完成** |
| **后端工具** | `core/backend/utils.py` | 消息格式化 + 重试机制 | **已完成** |
| **代码执行器** | `core/executor/interpreter.py` | 沙箱执行 + 超时控制 | **已完成** |
| **工作空间管理** | `core/executor/workspace.py` | 目录管理 + 文件归档 | **已完成** |
| **数据预览** | `utils/data_preview.py` | EDA 预览生成 | **已完成** |
| **指标工具** | `utils/metric.py` | 评估指标容器 | **已完成** |
| **响应解析** | `utils/response.py` | LLM 响应提取 | **已完成** |
| **Prompt 构建器** | `utils/prompt_builder.py` | 统一 Prompt 生成逻辑 | **已完成** |
| **Agent 基类** | `agents/base_agent.py` | Agent 抽象基类 | **已完成** |
| **CoderAgent** | `agents/coder_agent.py` | 代码生成 Agent（LLM重试+响应解析） | **已完成** |
| YAML 配置 | `config/default.yaml` | 项目主配置 | 已完成 |
| 环境变量 | `.env.example` | API Keys 模板 | 已完成 |

---

## 2. 配置系统 (`utils/config.py`)

### 2.1 架构设计

```
┌─────────────────────────────────────────────────────┐
│                    load_config()                     │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐  │
│  │  .env 文件  │ + │ YAML 配置  │ + │ CLI 参数   │  │
│  │  (低优先)   │   │  (中优先)   │   │  (高优先)   │  │
│  └─────┬──────┘   └─────┬──────┘   └─────┬──────┘  │
│        └────────────────┼────────────────┘          │
│                         ↓                            │
│                 validate_config()                    │
│                         ↓                            │
│                   Config 对象                        │
└─────────────────────────────────────────────────────┘
```

### 2.2 数据类结构

```python
@dataclass
class Config(Hashable):
    """顶层配置类"""
    project: ProjectConfig    # 项目基础配置
    data: DataConfig         # 数据配置
    llm: LLMConfig           # LLM 后端配置
    execution: ExecutionConfig # 执行配置
    agent: AgentConfig       # Agent 配置
    search: SearchConfig     # 搜索算法配置
    logging: LoggingConfig   # 日志配置
```

#### 子配置类

| 类名 | 字段 | 说明 |
|------|------|------|
| `ProjectConfig` | name, version, workspace_dir, log_dir, exp_name | 项目元信息 |
| `DataConfig` | data_dir, desc_file, goal, eval, preprocess_data, copy_data | 数据路径与选项 |
| `LLMStageConfig` | provider, model, temperature, api_key, base_url, max_tokens | 单阶段 LLM 配置（provider 必填） |
| `LLMConfig` | code, feedback | 双阶段 LLM 配置 |
| `ExecutionConfig` | timeout, agent_file_name, format_tb_ipython | 执行选项 |
| `AgentConfig` | max_steps, time_limit, k_fold_validation, ... | Agent 行为参数 |
| `SearchConfig` | strategy, max_debug_depth, debug_prob, num_drafts, parallel_num | 搜索策略参数 |
| `LoggingConfig` | level, console_output, file_output | 日志输出控制 |

### 2.3 核心函数

| 函数 | 签名 | 说明 |
|------|------|------|
| `load_config` | `(config_path?, use_cli?, env_file?) -> Config` | 加载并验证配置 |
| `validate_config` | `(cfg: DictConfig) -> Config` | 验证配置完整性 |
| `generate_exp_name` | `() -> str` | 生成实验名称 `YYYYMMDD_HHMMSS_xxxx` |
| `print_config` | `(cfg: Config) -> None` | Rich 美观打印配置 |
| `setup_workspace` | `(cfg: Config) -> None` | 初始化工作空间目录 |

### 2.4 验证规则

```
必填字段:
├── data.data_dir 必须存在
├── data.desc_file 或 data.goal 至少提供一个
└── llm.*.provider 必须为 "openai" 或 "anthropic"

路径处理:
├── 相对路径 → 绝对路径 (Path.resolve())
├── 不存在的目录 → 自动创建
└── exp_name 为空 → 自动生成

API Key 检查:
└── ${env:VAR} 未解析 → 记录 WARNING

Provider 验证:
├── llm.code.provider 必须在 {"openai", "anthropic"} 中
└── llm.feedback.provider 必须在 {"openai", "anthropic"} 中
└── 无效值 → ValueError
```

### 2.5 使用示例

```python
from utils.config import load_config, setup_workspace, print_config

# 基础用法
cfg = load_config()
print_config(cfg)

# 自定义配置文件
cfg = load_config(config_path=Path("custom.yaml"))

# 禁用 CLI 合并（用于测试）
cfg = load_config(use_cli=False)

# 指定 .env 文件
cfg = load_config(env_file=Path(".env.prod"))

# 初始化工作空间
setup_workspace(cfg)
```

---

## 3. 日志系统 (`utils/logger_system.py`)

### 3.1 双通道架构

```
┌─────────────────────────────────────────────────┐
│                 LoggerSystem                     │
│  ┌──────────────────┐  ┌──────────────────┐    │
│  │   text_log()     │  │   json_log()     │    │
│  │  ↓               │  │  ↓               │    │
│  │ logs/system.log  │  │ logs/metrics.json│    │
│  │  + 终端输出       │  │  (JSON 数组)     │    │
│  └──────────────────┘  └──────────────────┘    │
└─────────────────────────────────────────────────┘
```

### 3.2 核心函数

| 函数 | 签名 | 说明 |
|------|------|------|
| `init_logger` | `(log_dir: Path) -> LoggerSystem` | 初始化全局日志系统 |
| `log_msg` | `(level: str, message: str) -> None` | 记录文本日志 |
| `log_json` | `(data: Dict) -> None` | 记录 JSON 数据 |
| `ensure` | `(condition: bool, error_msg: str) -> None` | 断言工具 |
| `log_exception` | `(exc: Exception, context?: str) -> None` | 记录异常堆栈 |

### 3.3 日志级别

| 级别 | 用途 | 行为 |
|------|------|------|
| `DEBUG` | 调试信息 | 仅文件 |
| `INFO` | 一般信息 | 文件 + 终端 |
| `WARNING` | 警告 | 文件 + 终端 |
| `ERROR` | 错误 | 文件 + 终端（不自动 raise） |

### 3.4 Phase 1 重构要点

```python
# 旧版行为（已废弃）
log_msg("ERROR", "错误")  # 自动 raise Exception

# 新版行为
log_msg("ERROR", "错误")  # 仅记录，不抛出
raise ValueError("错误")  # 需显式抛出

# 推荐: 使用 ensure()
ensure(condition, "错误消息")  # 条件失败时 raise AssertionError
```

### 3.5 使用示例

```python
from utils.logger_system import init_logger, log_msg, log_json, ensure, log_exception

# 初始化
init_logger(Path("logs"))

# 文本日志
log_msg("INFO", "开始执行任务")
log_msg("WARNING", "检测到潜在问题")
log_msg("ERROR", "任务失败")  # 不抛出

# 断言
ensure(config.is_valid(), "配置无效")

# 异常记录
try:
    risky_operation()
except Exception as e:
    log_exception(e, "执行风险操作时")
    raise

# JSON 日志
log_json({
    "agent_name": "Agent1",
    "step": 3,
    "action": "tool_call",
    "metric": 0.85
})
```

---

## 4. 文件工具 (`utils/file_utils.py`)

### 4.1 核心函数

| 函数 | 签名 | 说明 |
|------|------|------|
| `copytree` | `(src, dst, use_symlinks=True) -> None` | 复制/链接目录树 |
| `_set_readonly_recursive` | `(path) -> None` | 递归设置只读权限 |

### 4.2 copytree 行为

```
use_symlinks=True (默认)
├── 创建符号链接指向源目录
├── 设置只读权限（平台支持时）
└── 失败时自动降级为复制

use_symlinks=False
├── 使用 shutil.copytree 复制
├── 支持 dirs_exist_ok=True 增量复制
└── 递归设置只读权限
```

### 4.3 跨平台兼容性

| 平台 | symlink 支持 | 回退方案 |
|------|-------------|---------|
| Linux | 完全支持 | - |
| macOS | 完全支持 | - |
| Windows | 需管理员权限 | 自动降级为复制 |

### 4.4 使用示例

```python
from utils.file_utils import copytree
from pathlib import Path

# 符号链接模式（推荐，节省空间）
copytree(src=Path("data"), dst=Path("workspace/input"), use_symlinks=True)

# 复制模式（完全隔离）
copytree(src=Path("data"), dst=Path("workspace/input"), use_symlinks=False)
```

---

## 5. 测试架构 (`tests/`)

### 5.1 目录结构

```
tests/
├── __init__.py
├── unit/                              # 单元测试
│   ├── __init__.py
│   ├── test_config.py                 # 配置系统测试 (7 个测试)
│   ├── test_config_priority.py        # 配置优先级测试 (4 个测试)
│   ├── test_file_utils.py             # 文件工具测试 (5 个测试)
│   ├── test_node.py                   # Node 数据类测试 (7 个测试)
│   ├── test_journal.py                # Journal 数据类测试 (12 个测试)
│   ├── test_task.py                   # Task 数据类测试 (5 个测试)
│   ├── test_state_integration.py      # State 集成测试 (1 个测试)
│   ├── test_backend_provider.py       # Backend Provider 测试 (6 个测试)
│   └── test_agents.py                 # CoderAgent 测试 (12 个测试) ★NEW★
└── integration/                       # 集成测试（待添加）
    └── __init__.py
```

### 5.2 测试类汇总

| 测试类 | 测试方法 | 覆盖功能 |
|--------|---------|---------|
| `TestLoadConfig` | 2 | 默认配置加载, CLI 覆盖 |
| `TestValidateConfig` | 2 | 必填字段检查 |
| `TestGenerateExpName` | 2 | 格式验证, 唯一性 |
| `TestSetupWorkspace` | 1 | 目录创建 |
| `TestConfigHashable` | 2 | dict key, set member |
| `TestConfigPriority` | 4 | .env 文件, 系统环境变量优先级, 完整链 |
| `TestCopytree` | 5 | symlink, copy, 异常, 只读, 替换 |
| **`TestNode`** | **7** | 创建, 序列化, 相等性, stage_name, has_exception, children_ids, metadata |
| **`TestJournal`** | **12** | append, get_node_by_id, get_children, get_siblings, draft_nodes, buggy/good_nodes, get_best_node, build_dag, 序列化, parse_solution_genes |
| **`TestTask`** | **5** | 创建, __str__, 序列化, 类型, dependencies |
| **`TestStateIntegration`** | **1** | 完整工作流：创建节点 -> 构建 DAG -> 查询 -> 序列化 |
| **`TestBackendProvider`** | **6** | Provider 参数验证、必填检查、base_url 传递、映射正确性 |
| **`TestCoderAgentInit`** | **1** | CoderAgent 初始化参数验证 |
| **`TestCoderAgentGenerate`** | **2** | generate() 方法、merge 类型处理 |
| **`TestCoderAgentExplore`** | **3** | 初稿/修复/改进模式 |
| **`TestCoderAgentLLMRetry`** | **2** | LLM 重试机制、重试次数耗尽 |
| **`TestCoderAgentResponseParsing`** | **2** | 代码执行失败、响应解析失败 |
| **`TestCoderAgentMemory`** | **1** | Memory 机制 |
| **`TestCoderAgentTimeStepsCalculation`** | **1** | 时间步数计算 |

### 5.3 运行测试

```bash
# 运行所有单元测试
pytest tests/unit/ -v

# 运行并查看覆盖率（utils + core）
pytest tests/unit/ --cov=utils --cov=core --cov-report=term-missing

# 运行特定测试文件
pytest tests/unit/test_config.py -v
pytest tests/unit/test_node.py -v
pytest tests/unit/test_journal.py -v
pytest tests/unit/test_task.py -v

# 运行特定测试类
pytest tests/unit/test_config.py::TestLoadConfig -v
pytest tests/unit/test_journal.py::TestJournal -v
```

---

## 6. 核心数据结构（Phase 1 已完成）

### 6.1 Node (`core/state/node.py`)

解决方案 DAG 中的单个节点，包含代码、执行结果、评估信息及 MCTS/GA 统计。

```python
@dataclass(eq=False)
class Node(DataClassJsonMixin):
    # ---- 代码 ----
    code: str                              # [必填] Python 代码
    plan: str = ""                         # 实现计划
    genes: Dict[str, str] = {}             # 基因组件（由 parse_solution_genes 解析）

    # ---- 通用属性 ----
    step: int = 0                          # Journal 中的序号
    id: str = uuid4().hex                  # 唯一 ID
    ctime: float = time()                  # 创建时间戳
    parent_id: Optional[str] = None        # 父节点 ID
    children_ids: list[str] = []           # 子节点 ID 列表（build_dag 构建）
    task_type: str = "draft"               # 任务类型
    metadata: Dict = {}                    # 额外元数据

    # ---- 执行信息 ----
    logs: str = ""                         # 执行日志
    term_out: str = ""                     # 终端输出
    exec_time: float = 0.0                 # 执行时间（秒）
    exc_type: Optional[str] = None         # 异常类型
    exc_info: Optional[Dict] = None        # 异常详情

    # ---- 评估 ----
    analysis: str = ""                     # LLM 分析结果
    metric_value: Optional[float] = None   # 评估指标值
    is_buggy: bool = False                 # 是否包含 bug
    is_valid: bool = True                  # 是否有效

    # ---- MCTS ----
    visits: int = 0                        # MCTS 访问次数
    total_reward: float = 0.0              # MCTS 累计奖励

    # ---- GA ----
    generation: Optional[int] = None       # GA 进化代数
    fitness: Optional[float] = None        # GA 适应度值
```

**方法:**

| 方法 | 返回类型 | 说明 |
|------|----------|------|
| `__eq__(other)` | `bool` | 基于 ID 比较相等性 |
| `__hash__()` | `int` | 基于 ID 生成哈希值 |
| `stage_name` (property) | `Literal["draft","debug","improve","unknown"]` | 推导节点阶段 |
| `has_exception` (property) | `bool` | 检查是否有执行异常 |

**使用示例:**

```python
from core.state import Node, parse_solution_genes

# 创建节点
node = Node(code="x = 1", plan="初始方案")

# 解析基因
node.genes = parse_solution_genes(node.code)

# 序列化
json_dict = node.to_dict()
restored = Node.from_dict(json_dict)
```

### 6.2 Journal (`core/state/journal.py`)

解决方案节点集合，表示搜索树/DAG，提供节点管理和树结构查询。

```python
@dataclass
class Journal(DataClassJsonMixin):
    nodes: list[Node] = []
```

**方法:**

| 方法 | 签名 | 说明 | 复杂度 |
|------|------|------|--------|
| `__len__` | `() -> int` | 返回节点数量 | O(1) |
| `__getitem__` | `(idx: int) -> Node` | 通过索引访问节点 | O(1) |
| `append` | `(node: Node) -> None` | 添加节点（自动设置 step） | O(1) |
| `get_node_by_id` | `(node_id: str) -> Optional[Node]` | 通过 ID 查找节点 | O(n) |
| `get_children` | `(node_id: str) -> list[Node]` | 获取子节点列表 | O(k) |
| `get_siblings` | `(node_id: str) -> list[Node]` | 获取兄弟节点（不含自身） | O(n) |
| `get_best_node` | `(only_good: bool = True) -> Optional[Node]` | 返回 metric_value 最高节点 | O(n) |
| `build_dag` | `() -> None` | 根据 parent_id 构建 children_ids | O(n) |
| `draft_nodes` (property) | `-> list[Node]` | 所有无父节点的节点 | O(n) |
| `buggy_nodes` (property) | `-> list[Node]` | 所有 is_buggy=True 的节点 | O(n) |
| `good_nodes` (property) | `-> list[Node]` | 所有 is_buggy=False 的节点 | O(n) |

**工具函数:**

| 函数 | 签名 | 说明 |
|------|------|------|
| `parse_solution_genes` | `(code: str) -> Dict[str, str]` | 解析 `# [SECTION: NAME]` 标记，分割代码为基因组件 |

**使用示例:**

```python
from core.state import Journal, Node, parse_solution_genes

journal = Journal()

# 添加节点
root = Node(code="# [SECTION: DATA]\nimport pandas as pd", plan="初始方案")
root.genes = parse_solution_genes(root.code)
journal.append(root)

child = Node(code="x = 2", parent_id=root.id, metric_value=0.85)
journal.append(child)

# 构建 DAG 并查询
journal.build_dag()
best = journal.get_best_node()
children = journal.get_children(root.id)
```

### 6.3 Task (`core/state/task.py`)

Agent 任务定义，用于任务队列和调度系统。

```python
TaskType = Literal["explore", "merge", "select", "review"]

@dataclass
class Task(DataClassJsonMixin):
    # ---- 核心字段 ----
    type: TaskType                              # [必填] 任务类型
    node_id: str                                # [必填] 关联节点 ID

    # ---- 元数据 ----
    description: str = ""                       # 任务描述
    id: str = uuid4().hex                       # 唯一标识符
    created_at: float = time()                  # 创建时间戳

    # ---- 调度信息 ----
    agent_name: Optional[str] = None            # 分配的 Agent 名称
    dependencies: Optional[Dict[str, str]] = None  # 任务依赖 {名称: 任务ID}
    payload: Dict = {}                          # 任务上下文数据
```

**Task 类型说明:**

| 类型 | 说明 |
|------|------|
| `explore` | 探索新方案，生成新的解决方案节点 |
| `merge` | 合并多个方案，融合不同节点的优点 |
| `select` | 选择最佳方案，从候选节点中筛选 |
| `review` | 审查方案质量，评估节点的有效性 |

**使用示例:**

```python
from core.state import Task

task = Task(
    type="explore",
    node_id="node_abc123",
    description="基于最佳节点探索新方案",
    agent_name="explorer_agent",
    payload={"parent_metric": 0.90},
)
print(task)  # Task(type=explore, node_id=node_abc...)
```

---

## 7. 外部依赖

| 包 | 版本 | 用途 |
|-----|------|------|
| `omegaconf` | >=2.3.0 | 配置管理 |
| `python-dotenv` | >=1.0.0 | .env 文件加载 |
| `rich` | >=13.0.0 | 终端美化输出 |
| `dataclasses-json` | >=0.6.0 | 数据类序列化 |
| `pydantic` | >=2.0.0 | 数据验证 |
| `pytest` | >=8.0.0 | 测试框架 |
| `pytest-asyncio` | >=0.23.0 | 异步测试 |
| `pytest-cov` | >=4.0.0 | 覆盖率报告 |
| `ruff` | >=0.3.0 | 代码格式化 |

---

## 8. 后端抽象层 (`core/backend/`)

### 8.1 架构设计

```
┌─────────────────────────────────────────────────────────┐
│                      query()                              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              provider 参数（必填）                    │ │
│  │  "openai" → backend_openai.query()                  │ │
│  │  "anthropic" → backend_anthropic.query()            │ │
│  └─────────────────────────────────────────────────────┘ │
│                         ↓                                 │
│  ┌──────────────────┐   ┌──────────────────────────────┐ │
│  │ backend_openai   │   │ backend_anthropic            │ │
│  │ - OpenAI GPT     │   │ - Claude 3.x                 │ │
│  │ - 第三方兼容 API  │   │ - 特殊消息处理               │ │
│  │ - 自定义 base_url │   │                              │ │
│  └──────────────────┘   └──────────────────────────────┘ │
│                         ↓                                 │
│  ┌─────────────────────────────────────────────────────┐ │
│  │                 utils.backoff_create()              │ │
│  │           指数退避重试: 1.5^n 秒, max 60s           │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

**设计变更 (2026-01-31)**:
- 删除 `determine_provider()` 函数（基于模型名自动推断）
- `provider` 参数改为必填，由配置文件显式指定
- 支持任意 OpenAI 兼容 API（通过 `base_url` 参数）

### 8.2 核心函数

| 函数 | 文件 | 签名 | 说明 |
|------|------|------|------|
| `query` | `__init__.py` | `(system_message, user_message, model, provider, ...) -> str` | 统一 LLM 查询入口（provider 必填） |
| `backend_openai.query` | `backend_openai.py` | 同上 | OpenAI 兼容 API 调用 |
| `backend_anthropic.query` | `backend_anthropic.py` | 同上 | Anthropic API 调用 |
| `opt_messages_to_list` | `utils.py` | `(system, user) -> list[dict]` | 消息格式转换 |
| `backoff_create` | `utils.py` | `(fn, exceptions, *args) -> Any` | 带重试的 API 调用 |

### 8.3 支持的提供商

| Provider | 说明 | 支持的第三方 API |
|----------|------|-----------------|
| `openai` | OpenAI 及兼容 API | GLM (智谱)、Moonshot、DeepSeek 等 |
| `anthropic` | Anthropic Claude | - |

**第三方 OpenAI 兼容 API 示例**:

| 服务 | base_url | 模型示例 |
|------|----------|---------|
| GLM (智谱) | `https://open.bigmodel.cn/api/paas/v4/` | `glm-4`, `glm-4-flash` |
| Moonshot | `https://api.moonshot.cn/v1` | `moonshot-v1-8k` |
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |

### 8.4 LLMStageConfig 配置

```python
@dataclass
class LLMStageConfig:
    """LLM 阶段配置（code/feedback）。"""
    provider: str                              # [必填] "openai" | "anthropic"
    model: str                                 # 模型名称
    temperature: float                         # 采样温度
    api_key: str                               # API 密钥
    base_url: str = "https://api.openai.com/v1"  # API 端点（第三方 API 覆盖此值）
    max_tokens: int | None = None              # 最大生成 token 数
```

### 8.5 配置示例

**YAML 配置 (config/default.yaml)**:

```yaml
llm:
  code:
    provider: ${env:LLM_PROVIDER, "openai"}  # 必填
    model: ${env:LLM_MODEL, "gpt-4-turbo"}
    temperature: 0.5
    api_key: ${env:OPENAI_API_KEY}
    base_url: ${env:OPENAI_BASE_URL, "https://api.openai.com/v1"}
    max_tokens: ${env:MAX_TOKENS, null}
```

**环境变量 (.env)**:

```bash
# 使用 OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key

# 使用第三方 API (如 Moonshot)
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-moonshot-key
OPENAI_BASE_URL=https://api.moonshot.cn/v1

# 使用 Anthropic
LLM_PROVIDER=anthropic
OPENAI_API_KEY=sk-ant-your-key
```

### 8.6 使用示例

```python
from core.backend import query

# 1. OpenAI GPT-4
response = query(
    system_message="You are a helpful assistant",
    user_message="Hello",
    model="gpt-4-turbo",
    provider="openai",           # 必填
    temperature=0.7,
    api_key="sk-...",
)

# 2. 第三方 OpenAI 兼容 API (Moonshot)
response = query(
    system_message="你是智能助手",
    user_message="介绍 Python",
    model="moonshot-v1-8k",
    provider="openai",           # 使用 openai provider
    api_key="sk-moonshot-...",
    base_url="https://api.moonshot.cn/v1",  # 指定第三方端点
)

# 3. Anthropic Claude
response = query(
    system_message="You are a helpful assistant",
    user_message="Hello",
    model="claude-3-opus-20240229",
    provider="anthropic",        # 必填
    api_key="sk-ant-...",
)

# 4. 错误处理
try:
    response = query(...)
except ValueError as e:
    # 不支持的 provider
    print(f"Provider 错误: {e}")
except TypeError as e:
    # 缺少必填参数 provider
    print(f"参数错误: {e}")
except Exception as e:
    # API 调用失败
    print(f"API 错误: {e}")
```

### 8.7 重试机制

```python
# 自动重试的异常类型:
# OpenAI: RateLimitError, APIConnectionError, APITimeoutError, InternalServerError
# Anthropic: 同上

# 重试策略: 指数退避
# 间隔: 1.5^n 秒 (n = 重试次数)
# 最大间隔: 60 秒
```

---

## 9. 执行层模块 (`core/executor/`)

### 9.1 Interpreter (`core/executor/interpreter.py`)

Python 代码执行沙箱，使用独立的 subprocess 执行代码。

```python
@dataclass
class ExecutionResult:
    """代码执行结果容器。"""
    term_out: List[str]              # 终端输出（stdout + stderr）
    exec_time: float                 # 执行时间（秒）
    exc_type: Optional[str]          # 异常类型（如 "ValueError"）
    exc_info: Optional[str]          # 异常详情（traceback 最后 5 行）
    success: bool                    # 执行是否成功
    timeout: bool                    # 是否超时

class Interpreter:
    """Python 代码执行沙箱。"""
    def __init__(self, working_dir: Path, timeout: int = 300):
        """初始化执行器。"""

    def run(self, code: str, reset_session: bool = True) -> ExecutionResult:
        """执行 Python 代码。"""
```

**核心功能:**

| 方法 | 说明 |
|------|------|
| `run(code)` | 在独立进程中执行代码，捕获输出和异常 |
| `_execute_in_subprocess(code)` | 创建临时脚本并执行 |
| `_capture_exception(stderr)` | 从 stderr 解析异常类型和 traceback |

**异常处理:**
- `TimeoutError`: 执行超过 timeout 限制
- `UnknownError`: 无法识别的异常类型
- 自动提取 traceback 最后 5 行作为详情

**使用示例:**

```python
from core.executor import Interpreter
from pathlib import Path

interp = Interpreter(working_dir=Path("workspace/working"), timeout=300)

result = interp.run(code="print('hello')")
if result.success:
    print(f"执行成功，耗时 {result.exec_time:.2f}s")
else:
    print(f"执行失败: {result.exc_type} - {result.exc_info}")
```

### 9.2 WorkspaceManager (`core/executor/workspace.py`)

工作空间管理器，负责目录结构管理和文件归档。

```python
class WorkspaceManager:
    """工作空间管理器。

    目录结构：
    workspace/
    ├── input/          # 输入数据（符号链接）
    ├── working/        # 临时工作目录
    ├── submission/     # 预测结果（每个 node 一个文件）
    ├── archives/       # 归档文件（每个 node 一个 zip）
    └── best_solution/  # 最佳解决方案
    """
```

**核心方法:**

| 方法 | 签名 | 说明 |
|------|------|------|
| `setup()` | `() -> None` | 创建工作空间目录结构 |
| `link_input_data(source_dir)` | `(Path?) -> None` | 链接/复制输入数据到 workspace/input/ |
| `rewrite_submission_path(code, node_id)` | `(str, str) -> str` | 重写代码中的 submission 路径 |
| `archive_node_files(node_id, code)` | `(str, str) -> Path?` | 打包节点文件为 zip |
| `cleanup_submission()` | `() -> None` | 清空 submission 目录 |
| `cleanup_working()` | `() -> None` | 清空 working 目录 |

**路径重写示例:**

```python
manager.rewrite_submission_path(
    code='df.to_csv("./submission/submission.csv")',
    node_id='abc123'
)
# 输出: 'df.to_csv("./submission/submission_abc123.csv")'
```

**归档文件格式:**

```
archives/node_abc123.zip
├── solution.py         # 代码内容
└── submission.csv      # 预测结果（如果存在）
```

---

## 10. 工具模块增强 (`utils/`)

### 10.1 数据预览 (`utils/data_preview.py`)

基于 aideml/ML-Master 实现，生成用于 LLM Prompt 的数据集预览。

**核心函数:**

| 函数 | 签名 | 说明 |
|------|------|------|
| `generate(base_path, include_file_details, simple)` | `(Path, bool, bool) -> str` | 生成目录的文本预览 |
| `file_tree(path, depth, max_dirs)` | `(Path, int, int) -> str` | 生成目录树结构 |
| `preview_csv(p, file_name, simple)` | `(Path, str, bool) -> str` | 生成 CSV 文件预览 |
| `preview_json(p, file_name)` | `(Path, str) -> str` | 生成 JSON 文件预览（自动生成 Schema） |

**CSV 预览模式:**

- **简化模式** (`simple=True`): shape + 前 15 列列名
- **详细模式** (`simple=False`): 每列统计
  - bool 列：True/False 百分比
  - 低基数列（< 10 个唯一值）：显示所有唯一值
  - 数值列：min-max 范围 + 缺失值数量
  - 字符串列：唯一值数量 + 前 4 个常见值

**自动长度控制:**
- 输出 > 6000 字符：自动降级到 simple 模式
- 仍 > 6000 字符：截断并添加 "... (truncated)"

**使用示例:**

```python
from utils.data_preview import generate
from pathlib import Path

preview_text = generate(
    base_path=Path("workspace/input"),
    include_file_details=True,
    simple=False
)
print(preview_text)
```

### 10.2 评估指标 (`utils/metric.py`)

提供指标值容器和比较工具。

```python
@dataclass
class MetricValue:
    """评估指标值容器。"""
    value: Optional[float]           # 指标值
    lower_is_better: bool = False   # 优化方向

    def is_better_than(self, other: "MetricValue") -> bool:
        """比较两个指标值。"""
```

**工具函数:**

| 函数 | 签名 | 说明 |
|------|------|------|
| `WorstMetricValue(lower_is_better)` | `(bool) -> MetricValue` | 返回最差值（Infinity 或 -Infinity） |
| `compare_metrics(a, b)` | `(MetricValue, MetricValue) -> int` | 三路比较（-1 / 0 / 1） |

**使用示例:**

```python
from utils.metric import MetricValue, WorstMetricValue

# 准确率（越高越好）
m1 = MetricValue(value=0.85, lower_is_better=False)
m2 = MetricValue(value=0.90, lower_is_better=False)
print(m2.is_better_than(m1))  # True

# 初始化最差值
worst = WorstMetricValue(lower_is_better=False)
print(worst.value)  # -inf
```

### 10.3 响应解析 (`utils/response.py`)

从 LLM 响应中提取代码和文本的工具。

**核心函数:**

| 函数 | 签名 | 说明 |
|------|------|------|
| `extract_code(response)` | `(str) -> str` | 提取第一个 Python 代码块 |
| `extract_text_up_to_code(response)` | `(str) -> str` | 提取代码块之前的文本（plan/说明） |
| `trim_long_string(text, max_length)` | `(str, int) -> str` | 截断长字符串并添加省略号 |

**支持的代码块格式:**
- ` ```python ... ``` `
- ` ```py ... ``` `
- ` ``` ... ``` ` (无语言标记)

**使用示例:**

```python
from utils.response import extract_code, extract_text_up_to_code

response = """
Here is my plan:
1. Load data
2. Train model

```python
import pandas as pd
df = pd.read_csv('train.csv')
```
"""

plan = extract_text_up_to_code(response)
# "Here is my plan:\n1. Load data\n2. Train model"

code = extract_code(response)
# "import pandas as pd\ndf = pd.read_csv('train.csv')"
```

---

## 11. Agent 抽象层 (`agents/`, `utils/prompt_builder.py`) ★NEW★

### 11.1 BaseAgent (`agents/base_agent.py`)

Agent 抽象基类，定义统一的 Agent 接口。

```python
class BaseAgent(ABC):
    """Agent 抽象基类。"""
    def __init__(self, name: str, config: Config, prompt_builder: PromptBuilder):
        """初始化 BaseAgent。"""

    @abstractmethod
    def generate(self, context: AgentContext) -> AgentResult:
        """主入口：根据 task_type 分发到具体实现。"""

    @abstractmethod
    def _explore(self, context: AgentContext) -> Node:
        """探索新方案（统一方法）。"""
```

**数据类:**

| 类名 | 职责 | 字段 |
|------|------|------|
| `AgentContext` | Agent 执行上下文 | task_type, parent_node, journal, config, start_time, current_step |
| `AgentResult` | Agent 执行结果 | node, success, error |

**设计要点:**
- 不显式区分初稿/改进/修复，让 LLM 根据上下文判断
- `parent_node=None` → LLM 知道要生成初稿
- `parent_node.is_buggy=True` → LLM 看到错误输出，自动修复
- `parent_node.is_buggy=False` → LLM 看到正常输出，自动改进

**使用示例:**

```python
from agents.base_agent import AgentContext, AgentResult, BaseAgent
from core.state import Journal, Node
from utils.config import load_config
from utils.prompt_builder import PromptBuilder

# 初始化
config = load_config()
prompt_builder = PromptBuilder()
agent = CoderAgent(name="coder", config=config, prompt_builder=prompt_builder)

# 执行
context = AgentContext(
    task_type="explore",
    parent_node=None,  # 初稿模式
    journal=Journal(),
    config=config,
    start_time=time.time(),
    current_step=0
)
result = agent.generate(context)

if result.success:
    print(f"生成节点: {result.node.id}")
else:
    print(f"执行失败: {result.error}")
```

### 11.2 PromptBuilder (`utils/prompt_builder.py`)

统一 Prompt 构建逻辑，根据上下文动态调整。

**核心方法:**

| 方法 | 签名 | 说明 |
|------|------|------|
| `build_explore_prompt` | `(task_desc, parent_node, memory, data_preview, time_remaining, steps_remaining) -> str` | 构建统一的 explore prompt |
| `_get_role_intro` | `() -> str` | 获取角色介绍 |
| `_get_guidelines` | `(time_remaining, steps_remaining) -> str` | 获取实现指南 |
| `_get_response_format` | `() -> str` | 获取响应格式说明 |
| `_format_time` | `(seconds) -> str` | 格式化时间（秒 → 人类可读） |

**自适应 Prompt 逻辑:**

```python
# 场景 1: 初稿模式
parent_node = None
→ Prompt 不包含 "Previous Attempt"
→ LLM 知道要生成初稿

# 场景 2: 修复模式
parent_node.is_buggy = True
→ Prompt 包含 "Previous Attempt + 错误输出"
→ LLM 看到异常信息，自动修复

# 场景 3: 改进模式
parent_node.is_buggy = False
→ Prompt 包含 "Previous Attempt + 正常输出"
→ LLM 看到正常执行，自动改进
```

**混淆模式:**
- `obfuscate=True`: 隐藏 Kaggle 背景，通用 ML 工程师
- `obfuscate=False`: 显式 Kaggle 大师身份

**使用示例:**

```python
from utils.prompt_builder import PromptBuilder

builder = PromptBuilder(obfuscate=False)

# 初稿模式
prompt = builder.build_explore_prompt(
    task_desc="预测房价",
    parent_node=None,
    memory="",
    data_preview="train.csv: 1460 rows, 81 columns",
    time_remaining=3600,
    steps_remaining=50
)

# 修复模式
prompt = builder.build_explore_prompt(
    task_desc="预测房价",
    parent_node=buggy_node,  # is_buggy=True
    memory="",
    data_preview="train.csv: 1460 rows, 81 columns",
    time_remaining=3500,
    steps_remaining=49
)
```

### 11.3 CoderAgent (`agents/coder_agent.py`) ★NEW★

代码生成 Agent，继承 BaseAgent，负责调用 LLM 生成代码并执行。

```python
class CoderAgent(BaseAgent):
    """代码生成 Agent。"""
    def __init__(self, name: str, config, prompt_builder, interpreter: Interpreter):
        """初始化 CoderAgent。"""

    def generate(self, context: AgentContext) -> AgentResult:
        """主入口：根据 task_type 分发到具体实现。"""

    def _explore(self, context: AgentContext) -> Node:
        """探索新方案（6 阶段流程）。"""

    def _call_llm_with_retry(self, prompt: str, max_retries: int = 5) -> str:
        """调用 LLM 并实现重试机制。"""

    def _parse_response_with_retry(self, response: str, max_retries: int = 5) -> Tuple[str, str]:
        """解析 LLM 响应并实现重试机制。"""
```

**核心方法:**

| 方法 | 签名 | 说明 |
|------|------|------|
| `generate` | `(context: AgentContext) -> AgentResult` | 主入口，分发到 `_explore` |
| `_explore` | `(context: AgentContext) -> Node` | 完整的代码生成→执行→评估流程 |
| `_call_llm_with_retry` | `(prompt: str, max_retries: int) -> str` | LLM 调用（5次重试，指数退避） |
| `_parse_response_with_retry` | `(response: str, max_retries: int) -> Tuple[str, str]` | 响应解析（硬格式失败不重试） |
| `_generate_data_preview` | `() -> Optional[str]` | 生成数据预览（Phase 2.4 完善） |
| `_calculate_remaining` | `(context) -> Tuple[int, int]` | 计算剩余时间和步数 |

**LLM 重试机制:**

```
调用失败 → 等待 10s → 重试
        → 等待 20s → 重试
        → 等待 40s → 重试
        → 等待 80s → 重试
        → 抛出异常（5次全部失败）
```

**响应解析策略:**

| 情况 | 处理方式 |
|------|---------|
| 硬格式失败（无代码块） | 直接抛出 ValueError，不重试 |
| 软格式失败（代码不合理） | Phase 2 简化：只要有代码块就成功 |

**_explore 6 阶段流程:**

1. **Phase 1: 准备上下文** - 数据预览、Memory、剩余时间/步数
2. **Phase 2: 构建 Prompt** - 调用 PromptBuilder
3. **Phase 3: 调用 LLM** - 带重试机制
4. **Phase 4: 解析响应** - 提取 plan 和 code
5. **Phase 5: 执行代码** - 调用 Interpreter
6. **Phase 6: 创建 Node** - 封装执行结果

**使用示例:**

```python
from agents.coder_agent import CoderAgent
from agents.base_agent import AgentContext
from core.executor import Interpreter
from core.state import Journal
from utils.config import load_config
from utils.prompt_builder import PromptBuilder
import time

# 初始化
config = load_config()
prompt_builder = PromptBuilder()
interpreter = Interpreter(working_dir=Path("workspace/working"))
agent = CoderAgent(
    name="coder",
    config=config,
    prompt_builder=prompt_builder,
    interpreter=interpreter
)

# 执行
context = AgentContext(
    task_type="explore",
    parent_node=None,  # 初稿模式
    journal=Journal(),
    config=config,
    start_time=time.time(),
    current_step=0
)
result = agent.generate(context)

if result.success:
    print(f"生成节点: {result.node.id}")
    print(f"代码执行: {'成功' if not result.node.is_buggy else '失败'}")
else:
    print(f"Agent 执行失败: {result.error}")
```

**测试覆盖 (92%):**

| 测试类 | 测试数量 | 覆盖功能 |
|--------|---------|---------|
| `TestCoderAgentInit` | 1 | 初始化参数验证 |
| `TestCoderAgentGenerate` | 2 | generate() 方法、merge 类型处理 |
| `TestCoderAgentExplore` | 3 | 初稿/修复/改进模式 |
| `TestCoderAgentLLMRetry` | 2 | LLM 重试、重试次数耗尽 |
| `TestCoderAgentResponseParsing` | 2 | 代码执行失败、响应解析失败 |
| `TestCoderAgentMemory` | 1 | Memory 机制 |
| `TestCoderAgentTimeStepsCalculation` | 1 | 时间步数计算 |

---

## 12. 模块依赖图

```mermaid
graph TD
    subgraph "基础设施层"
        LOG[utils/logger_system.py]
        CFG[utils/config.py]
        FU[utils/file_utils.py]
    end

    subgraph "数据结构层"
        NODE[core/state/node.py]
        JOURNAL[core/state/journal.py]
        TASK[core/state/task.py]
    end

    subgraph "后端抽象层"
        BACKEND[core/backend/__init__.py]
        OPENAI[core/backend/backend_openai.py]
        ANTHRO[core/backend/backend_anthropic.py]
        BUTILS[core/backend/utils.py]
    end

    subgraph "执行层"
        INTERP[core/executor/interpreter.py]
        WS[core/executor/workspace.py]
    end

    subgraph "工具增强"
        DP[utils/data_preview.py]
        METRIC[utils/metric.py]
        RESP[utils/response.py]
        PB[utils/prompt_builder.py]
    end

    subgraph "Agent 层"
        AGENT[agents/base_agent.py]
        CODER[agents/coder_agent.py ★NEW★]
    end

    %% 依赖关系
    CODER --> AGENT
    CODER --> BACKEND
    CODER --> INTERP
    CODER --> RESP
    AGENT --> CFG
    AGENT --> PB
    AGENT --> NODE
    AGENT --> JOURNAL

    PB --> NODE
    JOURNAL --> NODE
    TASK -.-> LOG

    INTERP --> LOG
    WS --> LOG
    WS --> CFG

    DP --> LOG
    METRIC -.-> LOG
    RESP -.-> LOG

    BACKEND --> LOG
    BACKEND --> OPENAI
    BACKEND --> ANTHRO

    OPENAI --> BUTILS
    OPENAI --> LOG

    ANTHRO --> BUTILS
    ANTHRO --> LOG

    BUTILS --> LOG

    CFG --> LOG
    FU --> LOG

    style LOG fill:#e1f5ff
    style NODE fill:#fff4e6
    style JOURNAL fill:#fff4e6
    style TASK fill:#fff4e6
    style BACKEND fill:#e8f5e9
    style INTERP fill:#fff3e0
    style WS fill:#fff3e0
    style DP fill:#f3e5f5
    style METRIC fill:#f3e5f5
    style RESP fill:#f3e5f5
```

**依赖层级**:
1. 基础层: `logger_system.py` (0 依赖)
2. 配置层: `config.py`, `file_utils.py` (依赖基础层)
3. 数据层: `Node`, `Journal`, `Task` (自包含，轻量依赖)
4. 后端层: `backend/*` (依赖基础层)
5. **执行层**: `executor/*` (依赖基础层 + 配置层)
6. **工具层**: `data_preview`, `metric`, `response`, **`prompt_builder`** (依赖基础层 + 数据层)
7. **Agent 层**: **`base_agent`**, **`coder_agent`** (依赖配置层 + 数据层 + 工具层 + 执行层 + 后端层)

**关键设计原则**:
- 单向依赖：下层不依赖上层
- 最小耦合：数据结构独立于配置和后端
- 易测试性：每层可独立测试

---

## 13. 关联文档

| 文档 | 路径 |
|------|------|
| 架构概览 | `docs/CODEMAPS/architecture.md` |
| 数据流与配置 | `docs/CODEMAPS/data.md` |
| Phase 1 详细计划 | `docs/plans/phase1_infrastructure.md` |
| 开发规范 | `CLAUDE.md` |
