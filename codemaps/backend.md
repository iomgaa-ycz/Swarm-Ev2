# 后端模块详解

**更新时间:** 2026-01-31
**范围:** utils/, core/, agents/

---

## 模块清单

| 模块 | 路径 | 行数 | 职责 | 状态 |
|------|------|------|------|------|
| **基础设施层** |||||
| 配置系统 | utils/config.py | 486 | OmegaConf 配置加载与验证 | ✅ |
| 日志系统 | utils/logger_system.py | 180 | 双通道日志输出 (文本+JSON) | ✅ |
| 文件工具 | utils/file_utils.py | 113 | 目录复制/链接 (跨平台) | ✅ |
| 工作空间构建 | utils/workspace_builder.py | 186 | 数据集验证 + 工作空间初始化 | ✅ |
| **数据结构层** |||||
| Node | core/state/node.py | 121 | 解决方案 DAG 节点 (22字段) | ✅ |
| Journal | core/state/journal.py | 293 | DAG 容器 (11方法 + generate_summary) | ✅ |
| Task | core/state/task.py | 62 | Agent 任务定义 | ✅ |
| **后端抽象层** |||||
| 统一接口 | core/backend/__init__.py | 137 | query() 支持 Function Calling | ✅ |
| OpenAI | core/backend/backend_openai.py | 163 | GPT + GLM (tools 参数) | ✅ |
| Anthropic | core/backend/backend_anthropic.py | 142 | Claude 系列 | ✅ |
| Utils | core/backend/utils.py | 80 | 消息格式化 + 指数退避重试 | ✅ |
| **执行层** |||||
| Interpreter | core/executor/interpreter.py | 176 | subprocess 沙箱 + 超时控制 | ✅ |
| WorkspaceManager | core/executor/workspace.py | 181 | 目录管理 + 路径重写 + 归档 | ✅ |
| **工具层** |||||
| DataPreview | utils/data_preview.py | 269 | EDA 预览生成 | ✅ |
| Metric | utils/metric.py | 117 | 评估指标容器 | ✅ |
| Response | utils/response.py | 89 | LLM 响应提取 | ✅ |
| PromptBuilder | utils/prompt_builder.py | 167 | 统一 Prompt 生成 | ✅ |
| **Agent 层** |||||
| BaseAgent | agents/base_agent.py | 117 | Agent 抽象基类 | ✅ |
| CoderAgent | agents/coder_agent.py | 272 | 代码生成 Agent (5次重试) | ✅ |
| **编排层** |||||
| Orchestrator | core/orchestrator.py | 427 | 任务编排器 (三阶段 + Review) | ✅ |
| **入口层** |||||
| Main | main.py | 161 | 白盒调试入口 | ✅ |

**总计:** 21 个核心模块 | ~3787 行代码

---

## 后端抽象层架构

### query() 统一接口

```python
from core.backend import query

# 支持的 provider: "openai" | "anthropic"
response = query(
    system_message="你是智能助手",
    user_message="1+1=?",
    model="gpt-4-turbo",
    provider="openai",
    api_key="sk-...",
    # Function Calling 参数 (可选)
    tools=[{"type": "function", "function": {...}}],
    tool_choice={"type": "function", "function": {"name": "tool_name"}}
)

# 返回值:
# - 无 tools: 返回 LLM 响应文本
# - 有 tools: 返回 tool call 的参数 JSON 字符串
```

### 提供商映射

```
PROVIDER_TO_QUERY = {
    "openai": backend_openai.query,      # OpenAI + GLM (智谱)
    "anthropic": backend_anthropic.query # Claude 系列
}
```

### Function Calling 支持

```
backend_openai.query() 新增参数:
├── tools: list[dict] | None          Function 定义列表
└── tool_choice: dict | str | None    工具选择策略

返回处理:
├── 无 tools: response.choices[0].message.content
└── 有 tools: response.choices[0].message.tool_calls[0].function.arguments
```

---

## Orchestrator 编排器

### 核心职责

```
Orchestrator (427 行)
├── 主循环控制 (run)
├── 三阶段父节点选择 (_select_parent_node)
├── Agent 代码生成调度 (agent.generate)
├── 代码执行 (_execute_code)
├── Function Calling Review (_review_node)
└── 最佳节点维护 (_update_best_node, _save_best_solution)
```

### 主循环流程

```
run(max_steps: int = 50)
│
├── for step in range(max_steps):
│   │
│   ├── if elapsed >= time_limit: break  (12 小时超时)
│   │
│   ├── _prepare_step()                  清理 submission/
│   │
│   ├── _select_parent_node()            三阶段策略
│   │
│   ├── agent.generate(context)          CoderAgent 生成代码
│   │
│   ├── _execute_code()                  执行 + 路径重写
│   │
│   ├── _review_node()                   Function Calling Review
│   │
│   ├── journal.append(node)             记录节点
│   │
│   ├── _update_best_node()              更新最佳节点
│   │
│   └── _save_best_solution()            保存到 best_solution/
│
└── return self.best_node
```

### 三阶段父节点选择

```
_select_parent_node() -> Optional[Node]
│
├── Phase 1: 初稿模式
│   条件: len(journal.draft_nodes) < config.search.num_drafts
│   返回: None → Agent 生成全新方案
│
├── Phase 2: 修复模式
│   条件: random() < config.search.debug_prob
│   操作: journal.build_dag() → 查找 buggy 叶子节点
│   返回: random_buggy_leaf → Agent 修复 bug
│
└── Phase 3: 改进模式
    条件: 默认
    操作: journal.get_best_node(only_good=True)
    返回: best_node → Agent 改进最佳方案
```

### Function Calling Review

```
_review_node(node: Node)
│
├── 构建 review_message:
│   ├── 任务描述 (self.task_desc)
│   ├── 代码 (node.code)
│   └── 执行输出 (node.term_out 或异常信息)
│
├── 调用 LLM:
│   model: config.llm.feedback.model (默认 glm-4.6)
│   provider: config.llm.feedback.provider
│   tools: [_get_review_tool_schema()]
│   tool_choice: {"type": "function", "function": {"name": "submit_review"}}
│
├── submit_review schema:
│   {
│     "name": "submit_review",
│     "description": "提交代码审查结果",
│     "parameters": {
│       "type": "object",
│       "properties": {
│         "is_bug": {"type": "boolean"},
│         "has_csv_submission": {"type": "boolean"},
│         "summary": {"type": "string"},
│         "metric": {"type": "number"},
│         "lower_is_better": {"type": "boolean"}
│       },
│       "required": ["is_bug", "has_csv_submission", "summary"]
│     }
│   }
│
├── 解析响应:
│   response = backend.query(...) → JSON 字符串
│   review_data = json.loads(response)
│
└── 更新节点:
    ├── node.analysis = review_data["summary"]
    ├── node.is_buggy = review_data["is_bug"] or node.exc_type is not None
    ├── node.metric_value = review_data.get("metric")
    └── node.lower_is_better = review_data.get("lower_is_better", False)
```

### 双向指标比较

```
_update_best_node(node: Node)
│
├── 过滤:
│   if node.is_buggy or node.metric_value is None:
│       return  # 跳过无效节点
│
├── 初始化:
│   if self.best_node is None:
│       self.best_node = node
│       return
│
└── 比较:
    ├── if node.lower_is_better:  # RMSE, MAE 等
    │   if node.metric_value < self.best_node.metric_value:
    │       self.best_node = node
    │
    └── else:  # Accuracy, F1 等
        if node.metric_value > self.best_node.metric_value:
            self.best_node = node
```

---

## CoderAgent 架构

### 核心方法

```
CoderAgent.generate(context: AgentContext) -> AgentResult
│
├── _explore(context)
│   │
│   ├── Phase 1: 准备上下文
│   │   ├── _generate_data_preview()       EDA 预览
│   │   ├── journal.generate_summary()     Memory 机制
│   │   └── _calculate_remaining()         剩余时间/步数
│   │
│   ├── Phase 2: 构建 Prompt
│   │   └── prompt_builder.build_explore_prompt()
│   │
│   ├── Phase 3: LLM 调用 (5 次重试)
│   │   └── _call_llm_with_retry(max_retries=5)
│   │       指数退避: 10s, 20s, 40s, 80s
│   │
│   ├── Phase 4: 响应解析 (带重试)
│   │   └── _parse_response_with_retry()
│   │       ├── 硬格式失败 (无代码块) → 不重试
│   │       └── 软格式失败 (代码不合理) → 重试
│   │
│   ├── Phase 5: 执行代码
│   │   └── interpreter.run(code, reset_session=True)
│   │
│   └── Phase 6: 创建 Node 对象
│       └── Node(code, plan, exec_result, is_buggy)
│
└── AgentResult(node, success, error)
```

### LLM 重试机制

```
_call_llm_with_retry(prompt: str, max_retries: int = 5)
│
├── for attempt in range(max_retries):
│   │
│   ├── try:
│   │   response = backend.query(...)
│   │   return response
│   │
│   ├── except RateLimitError:
│   │   delay = min(10 * (2 ** attempt), 80)  # 指数退避
│   │   time.sleep(delay)
│   │
│   └── except Exception:
│       if attempt == max_retries - 1:
│           raise
│
└── raise RuntimeError("LLM 调用失败")
```

### 响应解析重试

```
_parse_response_with_retry(response: str, max_retries: int = 3)
│
├── for attempt in range(max_retries):
│   │
│   ├── code, plan = response.parse_response(response)
│   │
│   ├── if code is None:  # 硬格式失败
│   │   raise ValueError("无代码块")
│   │
│   ├── if len(code) < 50:  # 软格式失败
│   │   if attempt < max_retries - 1:
│   │       response = _call_llm_with_retry(...)
│   │       continue
│   │
│   └── return code, plan
│
└── return code or "", plan or ""
```

---

## PromptBuilder 自适应逻辑

### 场景识别

```python
# 场景 1: 初稿模式
parent_node = None
# → Prompt 不包含 "Previous Attempt"
# → LLM 自动识别为初稿任务

# 场景 2: 修复模式
parent_node.is_buggy = True
# → Prompt 包含 "Previous Attempt + 错误输出"
# → LLM 看到异常信息，自动修复

# 场景 3: 改进模式
parent_node.is_buggy = False
# → Prompt 包含 "Previous Attempt + 正常输出"
# → LLM 看到正常执行，自动改进
```

### Prompt 结构

```
build_explore_prompt(context: AgentContext) -> str
│
├── 任务描述 (task_desc)
│
├── 数据预览 (data_preview, 可选)
│
├── 历史记录 (Memory)
│   └── journal.generate_summary(include_code=False)
│
├── 上一次尝试 (如果 parent_node 存在)
│   ├── 代码: parent_node.code
│   └── 输出: parent_node.term_out 或异常信息
│
├── 剩余资源
│   ├── 剩余时间: X 秒
│   └── 剩余步数: Y 步
│
└── 输出格式要求
    ```python
    # Implementation Plan
    ...

    # Code
    ...
    ```
```

---

## 核心数据结构

### Node (22 字段 + 4 方法)

```python
@dataclass
class Node:
    # 代码
    code: str                              # [必填] Python 代码
    plan: str = ""                         # 实现计划
    genes: Dict[str, str] = {}             # 基因组件

    # 通用属性
    step: int = 0                          # Journal 序号
    id: str = uuid4().hex                  # 唯一 ID
    ctime: float = time()                  # 创建时间戳
    parent_id: Optional[str] = None        # 父节点 ID
    children_ids: list[str] = []           # 子节点 ID
    task_type: str = "explore"             # 任务类型
    metadata: Dict = {}                    # 额外元数据

    # 执行信息
    logs: str = ""                         # 执行日志
    term_out: str = ""                     # 终端输出
    exec_time: float = 0.0                 # 执行时间 (秒)
    exc_type: Optional[str] = None         # 异常类型
    exc_info: Optional[Dict] = None        # 异常详情

    # 评估
    analysis: str = ""                     # LLM 分析
    metric_value: Optional[float] = None   # 指标值
    is_buggy: bool = False                 # 是否有 bug
    is_valid: bool = True                  # 是否有效
    lower_is_better: bool = False          # 指标方向

    # MCTS
    visits: int = 0                        # 访问次数
    total_reward: float = 0.0              # 累计奖励

    # GA
    generation: Optional[int] = None       # 进化代数
    fitness: Optional[float] = None        # 适应度值
```

### Journal (11 方法)

```python
@dataclass
class Journal:
    nodes: list[Node] = []

    # 核心方法:
    append(node: Node)                     # 添加节点
    get_node_by_id(node_id: str)          # 通过 ID 查找
    get_children(node_id: str)            # 获取子节点
    get_siblings(node_id: str)            # 获取兄弟节点
    get_best_node(only_good=True)         # 返回最佳节点
    build_dag()                            # 构建 children_ids
    generate_summary(include_code=False)   # Memory 机制

    # 属性:
    draft_nodes                            # 无父节点的节点
    buggy_nodes                            # is_buggy=True 的节点
    good_nodes                             # is_buggy=False 的节点
```

---

## 执行层模块

### Interpreter (沙箱执行)

```python
class Interpreter:
    def __init__(self, working_dir: Path, timeout: int = 300): ...

    def run(self, code: str, reset_session: bool = True) -> ExecutionResult:
        """执行 Python 代码 (subprocess)。

        特性:
        - 独立进程隔离
        - 超时控制 (默认 300s)
        - 捕获 stdout + stderr
        - 异常信息提取
        """
```

### WorkspaceManager (目录管理)

```python
class WorkspaceManager:
    # 核心方法:
    setup()                                # 创建目录结构
    link_input_data(source_dir)            # 链接/复制输入数据
    rewrite_submission_path(code, node_id) # 重写 submission 路径
    archive_node_files(node_id, code)      # 打包节点文件为 zip
    cleanup_submission()                   # 清空 submission/
    cleanup_working()                      # 清空 working/
```

---

## 配置系统

### 优先级

```
高 ──────────────────────────────── 低

CLI 参数          环境变量            YAML 配置
--key=value       export VAR=val     config/default.yaml
```

### Config 数据类

```python
@dataclass
class Config:
    project: ProjectConfig    # 项目基础配置
    data: DataConfig         # 数据配置
    llm: LLMConfig           # LLM 后端配置 (code + feedback)
    execution: ExecutionConfig # 执行配置
    agent: AgentConfig       # Agent 配置 (time_limit=43200)
    search: SearchConfig     # 搜索策略配置
    logging: LoggingConfig   # 日志配置
```

### 双阶段 LLM 配置

```yaml
llm:
  code:                       # CoderAgent 使用
    provider: openai
    model: gpt-4-turbo
    api_key: ${env:OPENAI_API_KEY}

  feedback:                   # Orchestrator Review 使用
    provider: openai
    model: glm-4.6            # 支持 Function Calling
    api_key: ${env:OPENAI_API_KEY}
    base_url: https://open.bigmodel.cn/api/coding/paas/v4
```

---

## 日志系统

### 双通道输出

```
logs/
├── system.log        # 文本日志 (追加写入)
└── metrics.json      # 结构化日志 (完整重写)
```

### 使用示例

```python
from utils.logger_system import log_msg, log_json, ensure

# 文本日志
log_msg("INFO", "Orchestrator 初始化完成")
log_msg("ERROR", "LLM 调用失败")  # 不自动抛异常

# 结构化日志
log_json({"step": 1, "metric": 0.85, "exec_time": 45.2})

# 断言工具
ensure(config.is_valid(), "配置无效")  # 失败时抛 AssertionError
```

---

## 测试架构

### 目录结构

```
tests/
├── unit/ (16 个测试文件)
│   ├── test_config.py
│   ├── test_node.py
│   ├── test_journal.py
│   ├── test_backend_provider.py
│   ├── test_interpreter.py
│   ├── test_workspace.py
│   ├── test_agents.py
│   ├── test_orchestrator.py
│   └── ...
└── integration/ (待添加)
```

### 运行测试

```bash
# 所有单元测试
conda run -n Swarm-Evo pytest tests/unit/ -v

# 测试覆盖率
conda run -n Swarm-Evo pytest tests/unit/ \
  --cov=utils --cov=core --cov=agents \
  --cov-report=term-missing
```

---

## 关联文档

| 文档 | 路径 |
|------|------|
| 架构概览 | codemaps/architecture.md |
| 数据流详解 | codemaps/data.md |
| 开发规范 | CLAUDE.md |
