# Phase 4: 扩展功能详细计划

## 1. 目标

1) **Memory 机制增强（Parent + Child Memory）**
- 在 Phase 3 最小 Memory 基础上，增加“父节点记忆 + 子节点记忆”两层结构：用于**减少重复探索**、**记录失败模式**、并在 Agent 生成（draft/improve/debug）时**注入到 Prompt**。

2) **工具注册表（Tool Registry）**
- 提供一个轻量、可测试、可动态加载的工具注册表：支持工具注册、列举、按名获取、从模块路径动态加载（MVP：importlib 加载 + allowlist）。

3) **Agent 注册表（Agent Registry）**
- 提供统一的 Agent 构建入口：支持按类型名创建 Agent（如 `coder` / `reviewer`），并支持从模块路径动态加载扩展 Agent（MVP：不做复杂插件系统）。

4) **HybridStrategy（混合策略）**（Phase 4 重点）
- 在不引入 LangGraph/重型框架的前提下，实现 **MCTS 与 GA 的协作/混合**：
  - MVP 推荐：**“代际 GA + 精英个体局部 MCTS 微调”** 的协作方式（可控、实现成本低、可日志化、可单测）。

---

## 2. 文件清单

> 说明：当前仓库工具扫描结果显示主要为 `docs/` 与 `Reference/`，Phase2/3 的代码文件可能尚未落盘或路径与规划不同。以下路径严格对齐 Phase2/3 规划中出现的约定路径；若实际代码路径不同，需在实现阶段同步调整，但本计划仍保持函数级别精确描述。

### 2.1 新建文件 [NEW]

- `core/memory/__init__.py`
  - 职责说明：Memory 子系统入口导出（层级记忆、文本构建器、指纹工具）。
  - 关键函数/类签名：
    - `from .hierarchical import HierarchicalMemory`
    - `from .builder import MemoryPromptBuilder`

- `core/memory/fingerprints.py`
  - 职责说明：生成“去重签名/失败指纹”的工具函数（MVP：字符串/正则；可选：AST）。
  - 关键函数/类签名：
    - `def code_signature(*, code: str) -> str`
    - `def plan_signature(*, plan: str) -> str`
    - `def error_fingerprint(*, term_out: str, exc_type: str | None) -> str`

- `core/memory/models.py`
  - 职责说明：Memory 记录的数据结构（成功/失败/重复等），便于序列化与日志化。
  - 关键函数/类签名：
    - `@dataclass(frozen=True) class MemoryRecord: ...`
    - `@dataclass(frozen=True) class FailurePattern: ...`

- `core/memory/hierarchical.py`
  - 职责说明：实现 Parent/Child 两层 Memory 索引与写入接口；支持按 parent_id 生成 child 约束。
  - 关键函数/类签名：
    - `class HierarchicalMemory:`
      - `def __init__(self, *, max_records_per_parent: int, max_sibling_records: int) -> None`
      - `def seen_code(self, *, signature: str) -> bool`
      - `def record_attempt(self, *, parent_id: str | None, node_id: str, plan: str, code: str) -> None`
      - `def record_result(self, *, parent_id: str | None, node_id: str, reward: float, is_buggy: bool, term_out: str, exc_type: str | None) -> None`
      - `def build_parent_context(self, *, parent_id: str | None) -> str`
      - `def build_child_context(self, *, parent_id: str | None) -> str`

- `core/memory/builder.py`
  - 职责说明：将层级 Memory 转换为可注入 Prompt 的文本（长度裁剪、结构化分段）。
  - 关键函数/类签名：
    - `class MemoryPromptBuilder:`
      - `def __init__(self, *, max_chars: int, max_items: int) -> None`
      - `def build_for_draft(self, *, global_summary: str, parent_ctx: str, child_ctx: str) -> str`
      - `def build_for_improve(self, *, parent_ctx: str, child_ctx: str, failure_ctx: str) -> str`
      - `def build_for_debug(self, *, failure_ctx: str) -> str`

- `tools/registry.py`
  - 职责说明：Tool Registry（注册/获取/列举/动态加载）。
  - 关键函数/类签名：
    - `@dataclass(frozen=True) class ToolSpec:`
      - `name: str`
      - `description: str`
      - `callable: Callable[..., object]`
      - `input_schema: type[BaseModel] | None`
    - `class ToolRegistry:`
      - `def register(self, *, spec: ToolSpec) -> None`
      - `def get(self, *, name: str) -> ToolSpec`
      - `def list_names(self) -> list[str]`
      - `def load_from_module(self, *, module_path: str, allowed_tools: set[str] | None) -> int`

- `agents/registry.py`
  - 职责说明：Agent Registry（注册 builder / 创建 agent / 动态加载 agent 类型）。
  - 关键函数/类签名：
    - `AgentBuilder = Callable[..., BaseAgent]`
    - `class AgentRegistry:`
      - `def register(self, *, agent_type: str, builder: AgentBuilder) -> None`
      - `def create(self, *, agent_type: str, name: str, config: Config, prompt_builder: PromptBuilder, tool_registry: ToolRegistry | None) -> BaseAgent`
      - `def load_from_module(self, *, module_path: str, allowed_agents: set[str] | None) -> int`

- `core/strategies/hybrid.py`
  - 职责说明：HybridStrategy 实现（GA 全局探索 + MCTS 局部强化），并与 HierarchicalMemory 协作。
  - 关键函数/类签名：
    - `class HybridStrategy(BaseSearchStrategy):`
      - `async def initialize(self, *, journal: Journal) -> None`
      - `async def step(self, *, journal: Journal) -> list[Node]`
      - `def is_finished(self, *, journal: Journal) -> bool`
      - `def _select_elites(self, *, population: list[Node], elite_k: int) -> list[Node]`
      - `async def _run_local_mcts(self, *, journal: Journal, seed_node: Node) -> list[Node]`

- `tests/test_memory/test_hierarchical_memory.py`
  - 职责说明：验证 Parent/Child 记忆生成、失败指纹、裁剪策略、去重效果。

- `tests/test_tools/test_tool_registry.py`
  - 职责说明：验证 register/get/list、模块动态加载、allowlist 拦截、日志输出。

- `tests/test_agents/test_agent_registry.py`
  - 职责说明：验证 agent 注册/创建、动态加载、错误处理。

- `tests/test_strategies/test_hybrid_strategy.py`
  - 职责说明：用 stub evaluator + stub agent 验证混合调度与“精英交换/微调”行为可重复。

- `tests/test_integration/test_phase4_hybrid_flow.py`
  - 职责说明：最小集成：Orchestrator + HybridStrategy + Memory + Registry（mock LLM / mock interpreter）。

### 2.2 修改文件 [MODIFY]

- `core/strategies/memory.py` 或迁移点（若 Phase3 已存在）
  - 修改内容（函数级别）：
    - [DELETE/REPLACE] Phase3 `MemoryStore/MemoryBuilder` 的单层实现（如已存在）
    - [MODIFY] 改为调用 `core/memory/HierarchicalMemory` 与 `MemoryPromptBuilder`

- `core/strategies/factory.py`
  - 修改内容（函数级别）：
    - [MODIFY] `def build_search_strategy(...) -> BaseSearchStrategy`：新增 `hybrid` 分支
    - [MODIFY] 注入 `HierarchicalMemory` 到策略实例（mcts/genetic/hybrid 共用）

- `core/orchestrator.py`
  - 修改内容（函数级别）：
    - [MODIFY] `__init__`：引入 `AgentRegistry` 与 `ToolRegistry` 的可选注入/构建
    - [MODIFY] `_evaluate_candidate_sync(...) -> EvaluationResult`：评估后写入 `HierarchicalMemory.record_result(...)` 并 `log_json(...)`
    - [MODIFY] `run_async(...)`/`step(...)`：打通 Memory 的 parent/child prompt 注入路径

- `agents/base_agent.py`
  - 修改内容（函数级别）：
    - [MODIFY] `AgentContext`：增加 `tools: ToolRegistry | None`（显式传入）
    - [MODIFY] `BaseAgent.generate(...)`：记录 `log_json`（task_type、parent_id、memory_chars、tool_names 等诊断信息）

- `agents/coder_agent.py`
  - 修改内容（函数级别）：
    - [MODIFY] `_draft/_improve/_debug`：使用 Phase4 的分段 memory（global/parent/child/failure）文本

- `utils/config.py`（或实际配置模块）
  - 修改内容（函数级别）：
    - [MODIFY] `SearchConfig`：允许 `search.strategy in {"mcts","genetic","hybrid"}`
    - [NEW/MODIFY] `HybridConfig`：添加混合策略所需最小字段
    - [MODIFY] `validate_config(...)`：校验 hybrid 参数边界（如 `elite_k <= population_size`）

---

## 3. 详细设计

### 3.1 Memory 增强 (core/strategies/memory.py 或 core/memory/)

#### 3.1.1 设计原则（MVP）
- **两层记忆**：Parent Memory + Child Memory
- **写入点清晰**：候选生成时 `record_attempt`；评估完成后 `record_result`
- **文本注入可控**：长度上限裁剪，避免 prompt 爆炸
- **强日志**：所有写入与裁剪都 `log_json()`

#### 3.1.2 数据结构（建议）
```python
# core/memory/models.py

from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class FailurePattern:
    """失败模式，用于在后续 prompt 中避免重复踩坑。"""
    fingerprint: str
    exc_type: str | None
    snippet: str
    count: int

@dataclass(frozen=True)
class MemoryRecord:
    """单次尝试记录（候选->评估）。"""
    parent_id: str | None
    node_id: str
    plan_sig: str
    code_sig: str
    reward: float | None
    is_buggy: bool | None
    error_fp: str | None
    meta: dict[str, Any]
```

#### 3.1.3 HierarchicalMemory（核心行为）
```python
# core/memory/hierarchical.py

class HierarchicalMemory:
    """层级记忆：Parent + Child。"""

    def record_attempt(self, *, parent_id: str | None, node_id: str, plan: str, code: str) -> None: ...
    def record_result(self, *, parent_id: str | None, node_id: str, reward: float, is_buggy: bool, term_out: str, exc_type: str | None) -> None: ...

    def build_parent_context(self, *, parent_id: str | None) -> str: ...
    def build_child_context(self, *, parent_id: str | None) -> str: ...
```

- MVP 约定：`parent_id is None` 视为 root 维度。
- 失败模式以 `error_fingerprint(...)` 聚合计数，生成 Top-N 常见错误。

#### 3.1.4 MemoryPromptBuilder（注入模板）
- 输出分段：Parent Memory / Sibling Attempts / Frequent Failure Patterns
- 统一裁剪（max_chars/max_items）

可选增强（后续）：AST 级签名、跨 run 持久化、失败模式自动分类。

---

### 3.2 Tool Registry (tools/registry.py)

#### 3.2.1 MVP 范围
- 手动注册 + importlib 动态加载
- allowlist 限制可加载工具名

#### 3.2.2 关键接口
```python
# tools/registry.py

from dataclasses import dataclass
from typing import Callable
from pydantic import BaseModel

@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    callable: Callable[..., object]
    input_schema: type[BaseModel] | None

class ToolRegistry:
    def register(self, *, spec: ToolSpec) -> None: ...
    def get(self, *, name: str) -> ToolSpec: ...
    def list_names(self) -> list[str]: ...
    def load_from_module(self, *, module_path: str, allowed_tools: set[str] | None) -> int: ...
```

日志：`log_json({"event": "tool_registry.load", "module": module_path, "loaded": n})`

可选增强（后续）：工具调用审计、强制 schema。

---

### 3.3 Agent Registry (agents/registry.py)

#### 3.3.1 MVP 范围
- 统一 create 入口
- allowlist 动态加载

#### 3.3.2 关键接口
```python
# agents/registry.py

from typing import Callable
from agents.base_agent import BaseAgent
from tools.registry import ToolRegistry
from utils.config import Config
from utils.prompt_builder import PromptBuilder

AgentBuilder = Callable[[str, Config, PromptBuilder, ToolRegistry | None], BaseAgent]

class AgentRegistry:
    def register(self, *, agent_type: str, builder: AgentBuilder) -> None: ...
    def create(self, *, agent_type: str, name: str, config: Config, prompt_builder: PromptBuilder, tool_registry: ToolRegistry | None) -> BaseAgent: ...
    def load_from_module(self, *, module_path: str, allowed_agents: set[str] | None) -> int: ...
```

可选增强（后续）：多 Agent 协作、能力标签。

---

### 3.4 HybridStrategy (core/strategies/hybrid.py)

#### 3.4.1 MVP 混合方式（推荐）
**GA 全局探索 + 精英本地 MCTS 微调**：
- GA 推进一代得到 population
- 选 elite_k 个精英，对每个精英跑 `local_mcts_rollouts_per_elite` 次小预算 MCTS
- 将局部改进结果回灌到候选池/种群

#### 3.4.2 配置（Phase4 MVP 最小字段）
- `population_size: int`
- `elite_k: int`
- `offspring_k: int`
- `mutation_rate: float`
- `crossover_rate: float`
- `local_mcts_rollouts_per_elite: int`
- `local_mcts_expand_k: int`
- `parallel_eval: int`

#### 3.4.3 关键实现接口
```python
# core/strategies/hybrid.py

class HybridStrategy(BaseSearchStrategy):
    async def initialize(self, *, journal: Journal) -> None: ...
    async def step(self, *, journal: Journal) -> list[Node]: ...
    def _select_elites(self, *, population: list[Node], elite_k: int) -> list[Node]: ...
    async def _run_local_mcts(self, *, journal: Journal, seed_node: Node) -> list[Node]: ...
    def is_finished(self, *, journal: Journal) -> bool: ...
```

Memory 注入点：GA mutation/improve 与 local MCTS expansion 都使用 parent/child/failure 约束。

可选增强（后续）：自适应权重调度、跨策略共享池。

---

### 3.5 Orchestrator / StrategyFactory 接入

- `core/strategies/factory.py`
  - 增加 `hybrid` 分支并注入 memory

- `core/orchestrator.py`
  - 构建 ToolRegistry/AgentRegistry
  - 评估完成后写入 memory + log_json

---

## 4. 验证计划

### 4.1 单元测试
- `pytest tests/test_memory/test_hierarchical_memory.py -v`
- `pytest tests/test_tools/test_tool_registry.py -v`
- `pytest tests/test_agents/test_agent_registry.py -v`
- `pytest tests/test_strategies/test_hybrid_strategy.py -v`

### 4.2 最小集成测试
- `pytest tests/test_integration/test_phase4_hybrid_flow.py -v`

### 4.3 覆盖率
- `pytest tests --cov=. --cov-report=term-missing`

---

## 5. 风险与缓解

1) **Memory 过强导致探索受限（中）**
- 缓解：MVP 仅同 parent 强去重 + 常见失败提示；memory 文本裁剪；参数可调。

2) **动态加载带来安全与可控性问题（中-高）**
- 缓解：强制 allowlist；加载/注册全量 `log_json`；错误用 `log_msg("ERROR", ...)` 失败。

3) **Hybrid 调度复杂，难定位收益（中）**
- 缓解：结构化日志记录 GA 与 local MCTS 产出与 reward；stub 测试保证可复现。

4) **与 MCTS/GA 接口耦合不当（中）**
- 缓解：HybridStrategy 作为顶层策略，内部复用最小子流程；避免让 MCTS/GA 互相依赖。

5) **Prompt 注入膨胀影响输出质量（中）**
- 缓解：MemoryPromptBuilder 统一裁剪；只输出要点列表。

---

**Phase 4 MVP 交付清单**
- [MVP] `HierarchicalMemory + MemoryPromptBuilder`
- [MVP] `ToolRegistry`（allowlist 动态加载）
- [MVP] `AgentRegistry`（allowlist 动态加载）
- [MVP] `HybridStrategy`（GA 代际 + 精英 local MCTS）
- [MVP] 工厂与 Orchestrator 接入 + 单测/最小集成测

**可选增强（后续）**
- [Optional] AST 级签名、跨 run 持久化 memory
- [Optional] 工具调用统一审计与沙箱
- [Optional] reviewer/debugger 多 Agent 协作、Hybrid 自适应调度
