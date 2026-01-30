# Swarm-Ev2 架构设计草稿

## 一、现状分析

### 1.1 Swarm-Evo (旧版) 的主要问题

#### 架构层面
- **高耦合**: `IterationController` 职责过重 (363行)，同时负责任务调度、Prompt 构建、Node 创建
- **配置分散**: `.env` + `agent.json` + YAML，缺乏统一管理
- **缺乏抽象**: 任务类型硬编码，工具注册不灵活
- **错误处理不一致**: `log_msg(ERROR)` 自动 raise 导致混乱

#### 代码质量问题
- **代码重复**: 任务类型判断逻辑重复多次
- **性能瓶颈**: 串行执行所有 Agent，理由是"避免 workspace 冲突"（过于保守）
- **内存效率**: Journal 所有节点在内存，长时间运行可能 OOM

#### 依赖管理
- 依赖过重 (327 个包)
- 版本固定 (==)，不够灵活

---

### 1.2 AIDE 项目的优秀设计

#### 核心亮点
1. **后端抽象层** (`backend/__init__.py`)
   - 统一 `query()` 接口，自动识别 LLM 提供商
   - 支持 OpenAI/Anthropic/Google/OpenRouter

2. **Solution Space Tree Search**
   - 树形搜索探索解决方案空间
   - 搜索策略: draft → debug (buggy) / improve (good)

3. **数据结构设计**
   - `@dataclass` + `DataClassJsonMixin` 自动序列化
   - `MetricValue` 实现 `@total_ordering`，自动比较

4. **配置系统**
   - OmegaConf: YAML + CLI 参数合并
   - 分层配置 (data/exec/agent/search)

5. **Function Calling**
   - `FunctionSpec` 确保 LLM 输出结构化可靠

6. **Memory 机制**
   - 历史经验提炼为 Prompt 上下文

#### 可借鉴的文件
- `/Reference/aideml-main/aide/backend/__init__.py` (后端抽象)
- `/Reference/aideml-main/aide/journal.py` (数据结构)
- `/Reference/aideml-main/aide/utils/config.py` (配置管理)
- `/Reference/aideml-main/aide/backend/utils.py` (FunctionSpec)

---

### 1.3 ML-Master 项目的优秀设计

#### 核心亮点
1. **MCTS (蒙特卡洛树搜索)**
   - Selection (UCT) → Expansion → Simulation → Backpropagation
   - 动态探索常数衰减 (4 种策略)

2. **并行搜索架构** ⭐⭐⭐
   - `ThreadPoolExecutor` + `FIRST_COMPLETED` 策略
   - 细粒度锁保证并发安全

3. **Memory 机制** ⭐⭐
   - Child Memory: 兄弟节点经验
   - Parent Memory: 父节点成功设计
   - Prompt 中注入历史避免重复探索

4. **多进程代码执行**
   - CPU 亲和性绑定
   - 超时控制 (SIGINT → SIGKILL)

5. **Steerable Reasoning**
   - 为开源模型注入推理链

#### 可借鉴的文件
- `/Reference/ML-Master-main/agent/mcts_agent.py` (MCTS 算法)
- `/Reference/ML-Master-main/search/mcts_node.py` (节点设计)
- `/Reference/ML-Master-main/interpreter/interpreter_parallel.py` (并行执行)
- `/Reference/ML-Master-main/main_mcts.py` (并行搜索架构)

---

## 二、Swarm-Ev2 架构设计

### 2.1 设计目标

1. **解耦**: 清晰的分层架构，职责单一
2. **可扩展**: 易于添加新 Agent、新工具、新搜索策略
3. **高性能**: 支持并发执行，避免不必要的串行瓶颈
4. **可配置**: 统一配置管理，支持 YAML + CLI
5. **可维护**: 代码简洁，类型注解完整，测试覆盖 80%+
6. **MVP 原则**: 避免过度工程化，专注核心功能

---

### 2.2 架构层次设计

```
Swarm-Ev2/
├── agents/                    # Agent 层
│   ├── base_agent.py         # 基础 Agent 抽象类
│   ├── coder_agent.py        # 代码生成 Agent
│   ├── reviewer_agent.py     # 代码评审 Agent
│   └── registry.py           # Agent 注册表
│
├── core/                      # 核心层
│   ├── state/                # 状态管理
│   │   ├── node.py          # Node 数据类 (借鉴 AIDE)
│   │   ├── journal.py       # Journal (解决方案树)
│   │   └── task.py          # Task 数据类
│   ├── backend/              # LLM 后端 (复用 AIDE)
│   │   ├── __init__.py      # 统一 query 接口
│   │   ├── openai.py
│   │   └── anthropic.py
│   ├── executor/             # 执行层
│   │   ├── interpreter.py   # 代码执行沙箱 (借鉴 ML-Master)
│   │   └── workspace.py     # 工作空间管理
│   └── orchestrator.py       # 编排器 (拆分 IterationController 职责)
│
├── search/                    # 搜索算法层
│   ├── base_strategy.py      # 搜索策略抽象类
│   ├── mcts_strategy.py      # MCTS 策略 (借鉴 ML-Master)
│   ├── genetic_strategy.py   # 遗传算法策略 (新增)
│   └── parallel_search.py    # 并行搜索框架 (借鉴 ML-Master)
│
├── tools/                     # 工具层
│   ├── base_tool.py          # 工具抽象类
│   ├── file_tools.py         # 文件操作工具
│   ├── shell_tool.py         # Shell 工具
│   └── registry.py           # 工具注册表
│
├── utils/                     # 工具模块
│   ├── config.py             # 配置管理 (OmegaConf)
│   ├── logger_system.py      # 日志系统 (保留)
│   ├── prompt_builder.py     # Prompt 构建器 (拆分出来)
│   └── metrics.py            # 度量系统
│
├── config/                    # 配置文件
│   ├── default.yaml          # 默认配置
│   ├── agents/               # Agent 配置
│   └── tools/                # 工具配置
│
├── tests/                     # 测试 (80%+ 覆盖率)
│   ├── test_agents/
│   ├── test_search/
│   └── test_integration/
│
└── main.py                    # 主入口
```

---

### 2.3 核心设计决策

#### Decision 1: 拆分 IterationController
**问题**: 旧版 363 行，职责过重

**方案**:
```python
# core/orchestrator.py (任务调度)
class Orchestrator:
    def __init__(self, journal, agent_pool, search_strategy):
        self.journal = journal
        self.agent_pool = agent_pool
        self.search_strategy = search_strategy

    def run(self):
        while not self.is_done():
            # 使用搜索策略选择节点
            node = self.search_strategy.select_node(self.journal)

            # 分配给合适的 Agent
            agent = self.agent_pool.get_agent_for_task(node.task_type)

            # 执行任务
            result = await agent.execute(node)

            # 更新 Journal
            self.journal.add_node(result)

# utils/prompt_builder.py (Prompt 构建)
class PromptBuilder:
    def build_context(self, node: Node, journal: Journal) -> dict:
        # 构建 Prompt 上下文
        pass

# agents/base_agent.py (Node 创建)
class BaseAgent:
    def execute(self, node: Node) -> Node:
        # 执行任务并创建新 Node
        pass
```

**好处**:
- 单一职责原则
- 易于测试
- 易于扩展

---

#### Decision 2: 统一配置管理
**问题**: 旧版配置分散在 `.env` + `agent.json` + YAML

**方案**: 复用 AIDE 的 OmegaConf 设计
```yaml
# config/default.yaml
project:
  name: "Swarm-Ev2"
  workspace: "./workspace"
  log_dir: "./logs"

llm:
  api_key: ${env:OPENAI_API_KEY}
  model: "gpt-4-turbo"
  temperature: 0.7

search:
  strategy: "mcts"  # mcts | genetic | pso
  max_steps: 50
  parallel_num: 3

agents:
  coder:
    max_steps: 10
    timeout: 3600
  reviewer:
    max_steps: 5
    timeout: 600

tools:
  enabled:
    - file_read
    - file_write
    - shell
```

**加载方式**:
```python
from omegaconf import OmegaConf

cfg = OmegaConf.load("config/default.yaml")
cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())  # CLI 覆盖
```

---

#### Decision 3: 工具注册表模式
**问题**: 旧版工具硬编码在 `agent_pool.py`

**方案**:
```python
# tools/registry.py
class ToolRegistry:
    _tools: dict[str, BaseTool] = {}

    @classmethod
    def register(cls, name: str, tool: BaseTool):
        cls._tools[name] = tool

    @classmethod
    def get_tools(cls, names: list[str]) -> list[BaseTool]:
        return [cls._tools[name] for name in names]

# config/tools/default.yaml
tools:
  file_read:
    enabled: true
    max_size: 1000000
  file_write:
    enabled: true
  shell:
    enabled: true
    timeout: 300

# main.py
for tool_name, tool_cfg in cfg.tools.items():
    if tool_cfg.enabled:
        ToolRegistry.register(tool_name, create_tool(tool_name, tool_cfg))
```

---

#### Decision 4: 搜索策略抽象
**问题**: 旧版只有隐式的搜索策略

**方案**: 策略模式
```python
# search/base_strategy.py
class SearchStrategy(ABC):
    @abstractmethod
    def select_node(self, journal: Journal) -> Node:
        """选择下一个要扩展的节点"""
        pass

    @abstractmethod
    def should_stop(self, journal: Journal) -> bool:
        """判断是否停止搜索"""
        pass

# search/mcts_strategy.py
class MCTSStrategy(SearchStrategy):
    def select_node(self, journal: Journal) -> Node:
        # UCT 选择
        pass

# search/genetic_strategy.py
class GeneticStrategy(SearchStrategy):
    def select_node(self, journal: Journal) -> Node:
        # 遗传算法选择
        pass
```

**配置切换**:
```yaml
search:
  strategy: "mcts"  # 一行配置切换算法
```

---

#### Decision 5: 并行搜索框架
**问题**: 旧版串行执行，性能瓶颈

**方案**: 借鉴 ML-Master 的并行架构
```python
# search/parallel_search.py
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

class ParallelSearch:
    def __init__(self, orchestrator, max_workers=3):
        self.orchestrator = orchestrator
        self.max_workers = max_workers

    def run(self, total_steps: int):
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.orchestrator.step)
                for _ in range(min(self.max_workers, total_steps))
            }

            completed = 0
            while completed < total_steps:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)

                for fut in done:
                    node = fut.result()
                    with self.lock:
                        self.save_checkpoint()
                    completed += 1

                # 动态补充新任务
                if completed + len(futures) < total_steps:
                    futures.add(executor.submit(self.orchestrator.step))
```

**优势**:
- Review 任务可并发（只读）
- 动态任务调度
- 线程安全

---

#### Decision 6: 数据结构设计
**方案**: 借鉴 AIDE 的 `@dataclass` 设计

```python
# core/state/node.py
from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin
from typing import Optional

@dataclass
class Node(DataClassJsonMixin):
    """解决方案节点"""
    id: str
    parent_id: Optional[str]
    code: str
    plan: str

    # 执行信息
    output: str
    exec_time: float
    is_buggy: bool

    # 评估结果
    score: Optional[float]
    analysis: str

    # MCTS 相关 (可选)
    visits: int = 0
    total_reward: float = 0.0

    def to_dict(self) -> dict:
        return self.to_dict()  # DataClassJsonMixin 提供

    @classmethod
    def from_dict(cls, data: dict) -> "Node":
        return cls.from_dict(data)  # DataClassJsonMixin 提供
```

---

#### Decision 7: 后端抽象层
**方案**: 复用 AIDE 的 `backend/` 模块

```python
# core/backend/__init__.py (直接复用 AIDE)
def query(system_message, user_message, model, ...):
    provider = determine_provider(model)
    query_func = provider_to_query_func[provider]
    return query_func(...)
```

**好处**:
- 支持多 LLM 提供商
- 统一接口
- 自动重试机制

---

#### Decision 8: Memory 机制
**方案**: 借鉴 ML-Master 的 Memory 设计

```python
# utils/prompt_builder.py
class PromptBuilder:
    def build_memory(self, node: Node, journal: Journal) -> str:
        """构建历史经验上下文"""
        # 1. 兄弟节点经验（避免重复）
        siblings = journal.get_siblings(node)
        sibling_memory = [
            f"Design: {s.plan}\nResult: {s.analysis}"
            for s in siblings
        ]

        # 2. 父节点成功经验（继承）
        if node.parent_id:
            parent = journal.get_node(node.parent_id)
            parent_memory = f"Parent Design: {parent.plan}\nScore: {parent.score}"

        return f"""
## Previous Attempts (Avoid Repetition)
{chr(10).join(sibling_memory)}

## Successful Design to Build Upon
{parent_memory if node.parent_id else "None"}
"""
```

---

### 2.4 数据流设计

```
main.py 启动
    ↓
[1] 加载配置 (OmegaConf)
    ↓
[2] 初始化日志系统
    ↓
[3] 创建 Backend (LLM 客户端)
    ↓
[4] 注册工具 (ToolRegistry)
    ↓
[5] 创建 Agent Pool
    ↓
[6] 初始化 Journal
    ↓
[7] 创建搜索策略 (MCTS/Genetic)
    ↓
[8] 创建 Orchestrator
    ↓
[9] 创建 ParallelSearch
    ↓
[10] 运行搜索
    ├─ select_node (搜索策略)
    ├─ get_agent (Agent Pool)
    ├─ execute (Agent)
    └─ add_node (Journal)
    ↓
[11] 输出最佳方案
```

---

### 2.5 与旧版的对比

| 维度 | Swarm-Evo (旧) | Swarm-Ev2 (新) |
|-----|----------------|----------------|
| **配置管理** | 分散 (.env + JSON + YAML) | 统一 (OmegaConf YAML) |
| **任务调度** | IterationController (363行) | Orchestrator + SearchStrategy |
| **并发性能** | 串行执行 | ThreadPoolExecutor 并行 |
| **工具系统** | 硬编码 | ToolRegistry 动态注册 |
| **搜索策略** | 隐式规则 | 显式策略模式 (可切换) |
| **Memory** | 无 | Child + Parent Memory |
| **类型注解** | 部分 | 完整 (Pydantic/dataclass) |
| **测试覆盖** | 无 | 80%+ |

---

## 三、实施计划

### Phase 1: 基础设施 (P0)
1. 配置系统 (config/)
2. 日志系统 (utils/logger_system.py 保留)
3. Backend 抽象层 (复用 AIDE)
4. 数据结构 (Node, Journal, Task)

### Phase 2: 核心功能 (P0)
5. BaseAgent 抽象类
6. Orchestrator (任务调度)
7. Interpreter (代码执行)
8. PromptBuilder (Prompt 构建)

### Phase 3: 搜索算法 (P1)
9. SearchStrategy 抽象类
10. MCTSStrategy 实现
11. ParallelSearch 框架

### Phase 4: 扩展功能 (P2)
12. Memory 机制
13. GeneticStrategy 实现
14. 工具注册表
15. Agent 注册表

### Phase 5: 测试与文档 (P1)
16. 单元测试 (80%+ 覆盖)
17. 集成测试
18. 文档同步

---

## 四、风险与缓解

### Risk 1: 并发执行的 Workspace 冲突
**缓解**:
- 为每个 Agent 分配独立 workspace 子目录
- 使用文件锁保护关键资源

### Risk 2: 配置系统学习成本
**缓解**:
- 提供详细的配置示例
- 默认配置覆盖 90% 场景

### Risk 3: 搜索策略抽象过度
**缓解**:
- MVP 阶段只实现 MCTS
- 预留扩展点但不过度设计

---

## 五、待确认问题

1. **是否需要保留 LangGraph?**
   - AIDE 没用 LangGraph，直接用 Python 实现
   - ML-Master 也是纯 Python
   - 建议: MVP 阶段不使用 LangGraph，减少依赖

2. **是否需要实现遗传算法?**
   - MCTS 已足够强大
   - 建议: MVP 阶段只实现 MCTS，预留接口

3. **是否需要 MLE-Bench 适配?**
   - 旧版有 `run_mle_adapter.py`
   - 建议: 保留，但作为可选功能

4. **日志系统是否需要重构?**
   - 现有 `logger_system.py` 有自动 raise 的问题
   - 建议: 去除自动 raise，统一异常处理

---

## 六、下一步行动

1. **等待用户确认**:
   - 是否使用 LangGraph?
   - 是否实现遗传算法?
   - 是否保留 MLE-Bench 适配?

2. **输出正式开发计划** (implementation_plan.md)
   - 摘要
   - 拟议变更 (文件级 + 函数级)
   - 验证计划

3. **开始 Phase 1 实施**
