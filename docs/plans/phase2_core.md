# Phase 2: 核心功能重构详细计划

## 1. 目标

Phase 2 的核心目标是实现 Swarm-Ev2 的执行引擎层，包括：

1. **BaseAgent 抽象类** - 定义 Agent 的统一接口，支持多种任务类型（draft/improve/debug）
2. **CoderAgent 实现** - 基于 BaseAgent 的代码生成 Agent 实现
3. **Orchestrator** - 任务编排器，协调 Agent 执行流程，替代旧版 363 行的 IterationController
4. **Interpreter** - 代码执行沙箱，支持并行执行和文件隔离
5. **WorkspaceManager** - 工作空间管理，实现动态文件名重写避免并发冲突
6. **PromptBuilder** - Prompt 构建器，提供简洁的模板渲染能力

**核心设计原则**：
- 不使用 LangGraph，纯 Python + asyncio 实现
- 借鉴 AIDE 的简洁设计（单一 Agent + 搜索策略）
- 借鉴 ML-Master 的并行执行能力（ThreadPoolExecutor + 文件名重写）
- 拆分旧版 IterationController 的职责到多个模块

---

## 2. 文件清单

### 2.1 新建文件 [NEW]

| 文件路径 | 职责说明 | 关键函数/类 |
|---------|---------|------------|
| `agents/__init__.py` | Agent 模块入口 | `CoderAgent` 导出 |
| `agents/base_agent.py` | Agent 抽象基类 | `BaseAgent`, `AgentContext`, `AgentResult` |
| `agents/coder_agent.py` | 代码生成 Agent | `CoderAgent`, `_draft()`, `_improve()`, `_debug()` |
| `core/orchestrator.py` | 任务编排器 | `Orchestrator`, `step()`, `run()` |
| `core/executor/__init__.py` | 执行器模块入口 | `Interpreter`, `WorkspaceManager` 导出 |
| `core/executor/interpreter.py` | 代码执行沙箱 | `Interpreter`, `ExecutionResult`, `run()` |
| `core/executor/workspace.py` | 工作空间管理 | `WorkspaceManager`, `setup()`, `rewrite_submission_path()` |
| `utils/prompt_builder.py` | Prompt 构建器 | `PromptBuilder`, `build_draft_prompt()`, `build_improve_prompt()` |

### 2.2 修改文件 [MODIFY]

| 文件路径 | 修改内容 |
|---------|---------|
| `core/state/node.py` | 新增 `absorb_exec_result()` 方法、`term_out` 属性 |
| `core/state/journal.py` | 新增 `generate_summary()` 方法用于 Memory 机制 |

---

## 3. 详细设计

### 3.1 BaseAgent 抽象类

**文件路径**: `agents/base_agent.py`

**设计理念**:
- 参考 AIDE 的 Agent 类设计，但不使用 LangGraph
- 支持同步和异步执行
- 通过组合模式注入 Backend 和 PromptBuilder

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Callable, Literal
import time

from core.state.node import Node
from core.state.journal import Journal
from core.backend import query
from utils.config import Config
from utils.prompt_builder import PromptBuilder

TaskType = Literal["draft", "improve", "debug"]

@dataclass
class AgentContext:
    """Agent 执行上下文，包含任务信息和运行状态。

    Attributes:
        task_type: 任务类型 (draft/improve/debug)
        parent_node: 父节点（improve/debug 时必填）
        journal: 解决方案日志
        config: 配置对象
        start_time: 搜索开始时间
        current_step: 当前步数
    """
    task_type: TaskType
    parent_node: Optional[Node]
    journal: Journal
    config: Config
    start_time: float = field(default_factory=time.time)
    current_step: int = 0


@dataclass
class AgentResult:
    """Agent 执行结果。

    Attributes:
        node: 生成的新节点
        success: 是否成功生成
        error: 错误信息（如果失败）
    """
    node: Optional[Node]
    success: bool
    error: Optional[str] = None


class BaseAgent(ABC):
    """Agent 抽象基类，定义 Agent 的统一接口。

    所有 Agent 实现必须继承此类并实现抽象方法。

    Attributes:
        name: Agent 名称
        config: 配置对象
        prompt_builder: Prompt 构建器
    """

    def __init__(
        self,
        name: str,
        config: Config,
        prompt_builder: PromptBuilder
    ) -> None:
        """初始化 Agent。

        Args:
            name: Agent 名称
            config: 配置对象
            prompt_builder: Prompt 构建器
        """
        self.name = name
        self.config = config
        self.prompt_builder = prompt_builder

    @abstractmethod
    def generate(self, context: AgentContext) -> AgentResult:
        """生成新的解决方案节点。

        Args:
            context: Agent 执行上下文

        Returns:
            AgentResult 包含生成的节点或错误信息
        """
        pass

    @abstractmethod
    def _draft(self, context: AgentContext) -> Node:
        """生成初始草稿。

        Args:
            context: Agent 执行上下文

        Returns:
            新生成的 Node
        """
        pass

    @abstractmethod
    def _improve(self, context: AgentContext) -> Node:
        """改进现有解决方案。

        Args:
            context: Agent 执行上下文

        Returns:
            改进后的 Node
        """
        pass

    @abstractmethod
    def _debug(self, context: AgentContext) -> Node:
        """修复有 bug 的解决方案。

        Args:
            context: Agent 执行上下文

        Returns:
            修复后的 Node
        """
        pass
```

**关键方法说明**:

1. `generate(context: AgentContext) -> AgentResult`
   - 功能: Agent 的主入口，根据 task_type 分发到对应的方法
   - 实现: 包含 try-except 错误处理，记录日志

2. `_draft()`, `_improve()`, `_debug()`
   - 抽象方法，子类必须实现
   - 分别对应三种任务类型的具体逻辑

---

### 3.2 CoderAgent 实现

**文件路径**: `agents/coder_agent.py`

**设计理念**:
- 继承 BaseAgent
- 参考 AIDE 的 Agent 类实现三阶段逻辑
- 使用 PromptBuilder 构建 Prompt
- 调用 Backend 获取 LLM 响应

```python
from typing import Tuple
import logging

from agents.base_agent import BaseAgent, AgentContext, AgentResult
from core.state.node import Node
from core.backend import query
from utils.config import Config
from utils.prompt_builder import PromptBuilder
from utils.response import extract_code, extract_text_up_to_code
from utils.logger_system import log_msg

logger = logging.getLogger("swarm-ev2")


class CoderAgent(BaseAgent):
    """代码生成 Agent，负责生成、改进和调试 ML 解决方案。

    实现三阶段任务：
    - draft: 生成初始解决方案
    - improve: 改进现有解决方案
    - debug: 修复有 bug 的解决方案
    """

    def __init__(
        self,
        name: str,
        config: Config,
        prompt_builder: PromptBuilder
    ) -> None:
        """初始化 CoderAgent。

        Args:
            name: Agent 名称
            config: 配置对象
            prompt_builder: Prompt 构建器
        """
        super().__init__(name, config, prompt_builder)
        self.data_preview: str | None = None

    def generate(self, context: AgentContext) -> AgentResult:
        """生成新的解决方案节点。

        根据 context.task_type 分发到对应的方法：
        - draft: 调用 _draft()
        - improve: 调用 _improve()
        - debug: 调用 _debug()

        Args:
            context: Agent 执行上下文

        Returns:
            AgentResult 包含生成的节点或错误信息
        """
        try:
            if context.task_type == "draft":
                node = self._draft(context)
            elif context.task_type == "improve":
                node = self._improve(context)
            elif context.task_type == "debug":
                node = self._debug(context)
            else:
                raise ValueError(f"未知的任务类型: {context.task_type}")

            return AgentResult(node=node, success=True)

        except Exception as e:
            log_msg("ERROR", f"Agent {self.name} 执行失败: {e}")
            return AgentResult(node=None, success=False, error=str(e))

    def _draft(self, context: AgentContext) -> Node:
        """生成初始草稿。

        构建 draft prompt，调用 LLM 生成代码，解析响应。

        Args:
            context: Agent 执行上下文

        Returns:
            新生成的 Node
        """
        log_msg("INFO", f"Agent {self.name} 开始生成初始草稿")

        prompt = self.prompt_builder.build_draft_prompt(
            task_desc=context.config.data.goal or "",
            memory=context.journal.generate_summary(),
            data_preview=self.data_preview,
            time_remaining=self._calc_time_remaining(context),
            steps_remaining=self._calc_steps_remaining(context)
        )

        plan, code = self._plan_and_code_query(prompt)

        node = Node(code=code, plan=plan)
        log_msg("INFO", f"Agent {self.name} 生成草稿节点 {node.id}")
        return node

    def _improve(self, context: AgentContext) -> Node:
        """改进现有解决方案。

        基于父节点的代码和执行结果，生成改进版本。

        Args:
            context: Agent 执行上下文

        Returns:
            改进后的 Node
        """
        parent = context.parent_node
        assert parent is not None, "improve 任务必须提供 parent_node"

        log_msg("INFO", f"Agent {self.name} 开始改进节点 {parent.id}")

        prompt = self.prompt_builder.build_improve_prompt(
            task_desc=context.config.data.goal or "",
            parent_code=parent.code,
            parent_output=parent.term_out,
            memory=context.journal.generate_summary(),
            time_remaining=self._calc_time_remaining(context),
            steps_remaining=self._calc_steps_remaining(context)
        )

        plan, code = self._plan_and_code_query(prompt)

        node = Node(code=code, plan=plan, parent_id=parent.id)
        log_msg("INFO", f"Agent {self.name} 改进节点 {parent.id} -> {node.id}")
        return node

    def _debug(self, context: AgentContext) -> Node:
        """修复有 bug 的解决方案。

        基于父节点的代码和错误信息，生成修复版本。

        Args:
            context: Agent 执行上下文

        Returns:
            修复后的 Node
        """
        parent = context.parent_node
        assert parent is not None, "debug 任务必须提供 parent_node"

        log_msg("INFO", f"Agent {self.name} 开始调试节点 {parent.id}")

        prompt = self.prompt_builder.build_debug_prompt(
            task_desc=context.config.data.goal or "",
            buggy_code=parent.code,
            error_output=parent.term_out,
            data_preview=self.data_preview,
            time_remaining=self._calc_time_remaining(context),
            steps_remaining=self._calc_steps_remaining(context)
        )

        plan, code = self._plan_and_code_query(prompt)

        node = Node(code=code, plan=plan, parent_id=parent.id)
        log_msg("INFO", f"Agent {self.name} 调试节点 {parent.id} -> {node.id}")
        return node

    def _plan_and_code_query(
        self,
        prompt: str,
        retries: int = 3
    ) -> Tuple[str, str]:
        """调用 LLM 生成 plan + code 并解析。

        Args:
            prompt: 完整的 prompt 字符串
            retries: 重试次数

        Returns:
            (plan, code) 元组
        """
        for attempt in range(retries):
            try:
                response = query(
                    system_message=prompt,
                    user_message=None,
                    model=self.config.llm.code.model,
                    temperature=self.config.llm.code.temperature
                )

                code = extract_code(response)
                plan = extract_text_up_to_code(response)

                if code and plan:
                    return plan, code

                log_msg("WARNING", f"解析失败，重试 {attempt + 1}/{retries}")

            except Exception as e:
                log_msg("WARNING", f"LLM 调用失败: {e}，重试 {attempt + 1}/{retries}")

        log_msg("ERROR", "LLM 调用和解析全部失败")
        return "", response if 'response' in locals() else ""

    def _calc_time_remaining(self, context: AgentContext) -> int:
        """计算剩余时间（秒）。"""
        import time
        elapsed = time.time() - context.start_time
        return max(0, int(context.config.agent.time_limit - elapsed))

    def _calc_steps_remaining(self, context: AgentContext) -> int:
        """计算剩余步数。"""
        return max(0, context.config.agent.max_steps - context.current_step)

    def update_data_preview(self, workspace_dir) -> None:
        """更新数据预览。

        Args:
            workspace_dir: 工作空间目录
        """
        from utils.data_preview import generate
        self.data_preview = generate(workspace_dir)
```

**关键方法说明**:

1. `generate(context)` - 主入口，根据任务类型分发
2. `_draft()` - 生成初始草稿，使用 draft prompt
3. `_improve()` - 改进方案，需要父节点的代码和输出
4. `_debug()` - 修复 bug，需要错误信息
5. `_plan_and_code_query()` - LLM 调用和响应解析

---

### 3.3 Orchestrator 任务编排器

**文件路径**: `core/orchestrator.py`

**设计理念**:
- 拆分旧版 IterationController 的职责
- 只负责任务调度和流程控制
- 搜索策略抽象到 Phase 3 实现

```python
import time
import shutil
from pathlib import Path
from typing import Optional, Callable
import logging

from agents.base_agent import BaseAgent, AgentContext
from core.state.node import Node
from core.state.journal import Journal
from core.executor.interpreter import Interpreter, ExecutionResult
from core.executor.workspace import WorkspaceManager
from utils.config import Config
from utils.logger_system import log_msg
from utils.metric import MetricValue, WorstMetricValue

logger = logging.getLogger("swarm-ev2")

# 执行回调类型
ExecCallbackType = Callable[[str, str, bool], ExecutionResult]


class Orchestrator:
    """任务编排器，协调 Agent 执行流程。

    负责：
    - 管理搜索循环
    - 调用 Agent 生成代码
    - 调用 Interpreter 执行代码
    - 解析执行结果
    - 更新 Journal
    - 保存最佳解决方案

    Attributes:
        agent: 代码生成 Agent
        config: 配置对象
        journal: 解决方案日志
        interpreter: 代码执行器
        workspace: 工作空间管理器
    """

    def __init__(
        self,
        agent: BaseAgent,
        config: Config,
        journal: Journal,
        task_desc: str
    ) -> None:
        """初始化编排器。

        Args:
            agent: 代码生成 Agent
            config: 配置对象
            journal: 解决方案日志
            task_desc: 任务描述
        """
        self.agent = agent
        self.config = config
        self.journal = journal
        self.task_desc = task_desc

        self.start_time = time.time()
        self.current_step = 0
        self.best_node: Optional[Node] = None

        # 初始化执行器和工作空间
        self.workspace = WorkspaceManager(config)
        self.interpreter = Interpreter(
            working_dir=config.project.workspace_dir,
            timeout=config.execution.timeout
        )

    def run(self, max_steps: Optional[int] = None) -> Optional[Node]:
        """运行搜索循环。

        Args:
            max_steps: 最大步数（可选，默认使用配置值）

        Returns:
            最佳节点
        """
        steps = max_steps or self.config.agent.max_steps

        log_msg("INFO", f"开始搜索循环，最大步数: {steps}")

        for step in range(steps):
            self.current_step = step

            # 检查时间限制
            elapsed = time.time() - self.start_time
            if elapsed >= self.config.agent.time_limit:
                log_msg("INFO", f"已达时间限制 {self.config.agent.time_limit}s")
                break

            log_msg("INFO", f"=== Step {step + 1}/{steps} ===")

            self.step()

        log_msg("INFO", f"搜索完成，共 {len(self.journal)} 个节点")

        if self.best_node:
            log_msg("INFO", f"最佳节点: {self.best_node.id}, 指标: {self.best_node.metric_value}")

        return self.best_node

    def step(self) -> None:
        """执行单步搜索。

        流程：
        1. 清理 submission 目录
        2. 选择搜索策略（draft/improve/debug）
        3. 调用 Agent 生成代码
        4. 执行代码并解析结果
        5. 更新 Journal 和最佳节点
        """
        # Phase 1: 清理并准备
        self._prepare_step()

        # Phase 2: 选择任务类型和父节点
        task_type, parent_node = self._search_policy()

        # Phase 3: 生成代码
        context = AgentContext(
            task_type=task_type,
            parent_node=parent_node,
            journal=self.journal,
            config=self.config,
            start_time=self.start_time,
            current_step=self.current_step
        )

        result = self.agent.generate(context)

        if not result.success or result.node is None:
            log_msg("WARNING", f"Agent 生成失败: {result.error}")
            return

        node = result.node

        # Phase 4: 执行代码
        exec_result = self._execute_code(node.code, node.id)
        node = self._parse_exec_result(node, exec_result)

        # Phase 5: 更新状态
        self._update_journal(node)
        self._update_best_node(node)

    def _prepare_step(self) -> None:
        """准备单步执行。"""
        submission_dir = self.config.project.workspace_dir / "submission"
        shutil.rmtree(submission_dir, ignore_errors=True)
        submission_dir.mkdir(exist_ok=True)

    def _search_policy(self) -> tuple[str, Optional[Node]]:
        """选择搜索策略。

        简单策略（Phase 2 默认实现）：
        - 如果草稿数 < num_drafts，则 draft
        - 如果有 buggy 节点且概率命中，则 debug
        - 否则 improve 最佳节点

        Returns:
            (task_type, parent_node) 元组
        """
        search_cfg = self.config.search

        # 初始 drafting
        if len(self.journal.draft_nodes) < search_cfg.num_drafts:
            log_msg("INFO", "[search_policy] drafting 新节点 (草稿不足)")
            return "draft", None

        # debugging
        import random
        if random.random() < search_cfg.debug_prob:
            buggy_leaves = [
                n for n in self.journal.buggy_nodes
                if n.is_leaf and n.debug_depth <= search_cfg.max_debug_depth
            ]
            if buggy_leaves:
                node = random.choice(buggy_leaves)
                log_msg("INFO", f"[search_policy] debugging 节点 {node.id}")
                return "debug", node

        # improving
        good_nodes = self.journal.good_nodes
        if not good_nodes:
            log_msg("INFO", "[search_policy] drafting 新节点 (无可用节点)")
            return "draft", None

        best = self.journal.get_best_node()
        if best:
            log_msg("INFO", f"[search_policy] improving 节点 {best.id}")
            return "improve", best

        return "draft", None

    def _execute_code(self, code: str, node_id: str) -> ExecutionResult:
        """执行代码。

        Args:
            code: 待执行的代码
            node_id: 节点 ID（用于文件名重写）

        Returns:
            执行结果
        """
        # 重写 submission 文件路径
        modified_code = self.workspace.rewrite_submission_path(code, node_id)

        return self.interpreter.run(modified_code, reset_session=True)

    def _parse_exec_result(
        self,
        node: Node,
        exec_result: ExecutionResult
    ) -> Node:
        """解析执行结果。

        Args:
            node: 待更新的节点
            exec_result: 执行结果

        Returns:
            更新后的节点
        """
        node.absorb_exec_result(exec_result)

        # TODO: Phase 2 简化版，后续增加 LLM 评估
        # 简单判断：有异常或无 submission 文件则为 buggy
        submission_path = (
            self.config.project.workspace_dir /
            "submission" /
            f"submission_{node.id}.csv"
        )

        if node.exc_type is not None:
            node.is_buggy = True
            node.metric_value = None
            log_msg("INFO", f"节点 {node.id} 执行出错: {node.exc_type}")
        elif not submission_path.exists():
            node.is_buggy = True
            node.metric_value = None
            log_msg("INFO", f"节点 {node.id} 未生成 submission.csv")
        else:
            node.is_buggy = False
            # TODO: 从输出中提取 metric
            node.metric_value = self._extract_metric(node.term_out)
            log_msg("INFO", f"节点 {node.id} 执行成功, metric: {node.metric_value}")

        return node

    def _extract_metric(self, term_out: str) -> Optional[float]:
        """从终端输出提取指标值。

        简单实现：查找最后一个浮点数。

        Args:
            term_out: 终端输出

        Returns:
            提取的指标值
        """
        import re
        numbers = re.findall(r"[-+]?\d*\.?\d+", term_out)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass
        return None

    def _update_journal(self, node: Node) -> None:
        """更新 Journal。"""
        self.journal.append(node)

    def _update_best_node(self, node: Node) -> None:
        """更新最佳节点。"""
        if node.is_buggy or node.metric_value is None:
            return

        if self.best_node is None or (
            node.metric_value > self.best_node.metric_value
        ):
            log_msg("INFO", f"新的最佳节点: {node.id}")
            self.best_node = node
            self._save_best_solution(node)

    def _save_best_solution(self, node: Node) -> None:
        """保存最佳解决方案。"""
        best_dir = self.config.project.workspace_dir / "best_solution"
        best_dir.mkdir(exist_ok=True, parents=True)

        # 复制 submission
        submission_src = (
            self.config.project.workspace_dir /
            "submission" /
            f"submission_{node.id}.csv"
        )
        if submission_src.exists():
            shutil.copy(submission_src, best_dir / "submission.csv")

        # 保存代码
        with open(best_dir / "solution.py", "w") as f:
            f.write(node.code)

        # 记录节点 ID
        with open(best_dir / "node_id.txt", "w") as f:
            f.write(node.id)
```

**关键方法说明**:

1. `run(max_steps)` - 主循环，控制步数和时间限制
2. `step()` - 单步执行，包含完整流程
3. `_search_policy()` - 搜索策略（简单版，Phase 3 扩展为 MCTS/GA）
4. `_execute_code()` - 调用 Interpreter 执行
5. `_parse_exec_result()` - 解析执行结果，判断是否 buggy
6. `_update_best_node()` - 更新并保存最佳节点

---

### 3.4 Interpreter 代码执行沙箱

**文件路径**: `core/executor/interpreter.py`

**设计理念**:
- 复用 AIDE 的 Interpreter 设计
- 支持超时控制
- 捕获 stdout/stderr 和异常

```python
"""代码执行沙箱，支持超时控制和输出捕获。"""

import logging
import os
import queue
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Optional

import humanize
from dataclasses_json import DataClassJsonMixin

from utils.logger_system import log_msg

logger = logging.getLogger("swarm-ev2")


def trim_long_string(s: str, threshold: int = 5000, k: int = 2000) -> str:
    """截断过长的字符串。

    Args:
        s: 原始字符串
        threshold: 截断阈值
        k: 保留首尾字符数

    Returns:
        截断后的字符串
    """
    if len(s) > threshold:
        truncated = len(s) - 2 * k
        return f"{s[:k]}\n...[{truncated} 字符已截断]...\n{s[-k:]}"
    return s


@dataclass
class ExecutionResult(DataClassJsonMixin):
    """代码执行结果。

    Attributes:
        term_out: 终端输出列表
        exec_time: 执行时间（秒）
        exc_type: 异常类型（无异常则为 None）
        exc_info: 异常详情
        exc_stack: 异常堆栈
    """
    term_out: list[str]
    exec_time: float
    exc_type: Optional[str] = None
    exc_info: Optional[dict] = None
    exc_stack: Optional[list[tuple]] = None


class RedirectQueue:
    """将输出重定向到队列。"""

    def __init__(self, q: Queue) -> None:
        self.queue = q

    def write(self, msg: str) -> None:
        self.queue.put(msg)

    def flush(self) -> None:
        pass


class Interpreter:
    """代码执行沙箱，在独立进程中执行代码。

    功能：
    - 捕获 stdout/stderr
    - 超时控制
    - 异常捕获

    Attributes:
        working_dir: 工作目录
        timeout: 超时时间（秒）
        agent_file_name: 临时代码文件名
    """

    def __init__(
        self,
        working_dir: Path | str,
        timeout: int = 3600,
        agent_file_name: str = "runfile.py"
    ) -> None:
        """初始化解释器。

        Args:
            working_dir: 工作目录
            timeout: 超时时间（秒）
            agent_file_name: 临时代码文件名
        """
        self.working_dir = Path(working_dir).resolve()
        assert self.working_dir.exists(), f"工作目录不存在: {self.working_dir}"

        self.timeout = timeout
        self.agent_file_name = agent_file_name
        self.process: Optional[Process] = None

    def _child_proc_setup(self, result_outq: Queue) -> None:
        """子进程初始化。"""
        import shutup
        shutup.mute_warnings()

        os.chdir(str(self.working_dir))
        sys.path.append(str(self.working_dir))
        sys.stdout = sys.stderr = RedirectQueue(result_outq)

    def _run_session(
        self,
        code_inq: Queue,
        result_outq: Queue,
        event_outq: Queue
    ) -> None:
        """子进程执行会话。"""
        self._child_proc_setup(result_outq)

        global_scope: dict = {"__name__": "__main__"}

        while True:
            code = code_inq.get()
            os.chdir(str(self.working_dir))

            with open(self.agent_file_name, "w") as f:
                f.write(code)

            event_outq.put(("state:ready",))

            try:
                exec(compile(code, self.agent_file_name, "exec"), global_scope)
            except BaseException as e:
                tb_str, exc_cls, exc_info, exc_stack = self._exception_summary(e)
                result_outq.put(tb_str)

                if exc_cls == "KeyboardInterrupt":
                    exc_cls = "TimeoutError"

                event_outq.put(("state:finished", exc_cls, exc_info, exc_stack))
            else:
                event_outq.put(("state:finished", None, None, None))

            # 清理临时文件
            if os.path.exists(self.agent_file_name):
                os.remove(self.agent_file_name)

            result_outq.put("<|EOF|>")

    def _exception_summary(self, e: BaseException) -> tuple:
        """生成异常摘要。"""
        tb_lines = traceback.format_exception(e)
        tb_str = "".join(
            l for l in tb_lines
            if "swarm-ev2/" not in l and "importlib" not in l
        )
        tb_str = tb_str.replace(
            str(self.working_dir / self.agent_file_name),
            self.agent_file_name
        )

        exc_info = {}
        if hasattr(e, "args"):
            exc_info["args"] = [str(a) for a in e.args]

        tb = traceback.extract_tb(e.__traceback__)
        exc_stack = [(t.filename, t.lineno, t.name, t.line) for t in tb]

        return tb_str, e.__class__.__name__, exc_info, exc_stack

    def _create_process(self) -> tuple[Queue, Queue, Queue]:
        """创建子进程。"""
        code_inq, result_outq, event_outq = Queue(), Queue(), Queue()

        self.process = Process(
            target=self._run_session,
            args=(code_inq, result_outq, event_outq)
        )
        self.process.start()

        return code_inq, result_outq, event_outq

    def _cleanup_session(self) -> None:
        """清理子进程。"""
        if self.process is None:
            return

        self.process.terminate()
        self.process.join(timeout=2)

        if self.process.exitcode is None:
            logger.warning("子进程未能优雅终止，强制杀死")
            self.process.kill()
            self.process.join()

        self.process.close()
        self.process = None

    def run(self, code: str, reset_session: bool = True) -> ExecutionResult:
        """执行代码。

        Args:
            code: 待执行的 Python 代码
            reset_session: 是否重置会话

        Returns:
            执行结果
        """
        log_msg("INFO", f"Interpreter 执行代码 (reset_session={reset_session})")

        if reset_session:
            if self.process is not None:
                self._cleanup_session()
            code_inq, result_outq, event_outq = self._create_process()
        else:
            assert self.process is not None, "首次执行必须 reset_session=True"

        assert self.process.is_alive()

        code_inq.put(code)

        # 等待子进程准备就绪
        try:
            state = event_outq.get(timeout=10)
        except queue.Empty:
            msg = "子进程启动超时"
            log_msg("ERROR", msg)
            self._cleanup_session()
            return ExecutionResult(
                term_out=[msg],
                exec_time=0,
                exc_type="RuntimeError"
            )

        assert state[0] == "state:ready", state
        start_time = time.time()

        child_in_overtime = False

        while True:
            try:
                state = event_outq.get(timeout=1)
                assert state[0] == "state:finished", state
                exec_time = time.time() - start_time
                break
            except queue.Empty:
                if not child_in_overtime and not self.process.is_alive():
                    msg = "子进程意外终止"
                    log_msg("ERROR", msg)
                    self._cleanup_session()
                    return ExecutionResult(
                        term_out=[msg],
                        exec_time=0,
                        exc_type="RuntimeError"
                    )

                if self.timeout is None:
                    continue

                running_time = time.time() - start_time
                if running_time > self.timeout:
                    os.kill(self.process.pid, signal.SIGINT)
                    child_in_overtime = True

                    if running_time > self.timeout + 60:
                        logger.warning("子进程超时后仍未终止，强制杀死")
                        self._cleanup_session()
                        state = (None, "TimeoutError", {}, [])
                        exec_time = self.timeout
                        break

        # 收集输出
        output: list[str] = []
        while not result_outq.empty() or not output or output[-1] != "<|EOF|>":
            output.append(result_outq.get())
        output.pop()  # 移除 EOF 标记

        exc_type, exc_info, exc_stack = state[1:]

        if exc_type == "TimeoutError":
            output.append(
                f"TimeoutError: 执行超过时间限制 {humanize.naturaldelta(self.timeout)}"
            )
        else:
            output.append(
                f"执行时间: {humanize.naturaldelta(exec_time)} "
                f"(限制: {humanize.naturaldelta(self.timeout)})"
            )

        return ExecutionResult(
            term_out=output,
            exec_time=exec_time,
            exc_type=exc_type,
            exc_info=exc_info,
            exc_stack=exc_stack
        )
```

---

### 3.5 WorkspaceManager 工作空间管理

**文件路径**: `core/executor/workspace.py`

**设计理念**:
- 参考 ML-Master 的动态文件名重写
- 避免并发执行时的文件冲突
- 管理输入数据的 symlink

```python
"""工作空间管理器，负责文件隔离和路径重写。"""

import os
import re
import shutil
from pathlib import Path
from typing import Optional

from utils.config import Config
from utils.logger_system import log_msg


class WorkspaceManager:
    """工作空间管理器。

    负责：
    - 初始化工作空间目录结构
    - 动态重写 submission 文件路径避免冲突
    - 管理输入数据的符号链接

    Attributes:
        config: 配置对象
        workspace_dir: 工作空间根目录
    """

    def __init__(self, config: Config) -> None:
        """初始化工作空间管理器。

        Args:
            config: 配置对象
        """
        self.config = config
        self.workspace_dir = Path(config.project.workspace_dir).resolve()

    def setup(self) -> None:
        """初始化工作空间目录结构。

        创建以下目录：
        - input/: 输入数据（symlink 或复制）
        - working/: 临时文件
        - submission/: 提交文件
        - best_solution/: 最佳解决方案
        """
        # 创建目录
        (self.workspace_dir / "input").mkdir(parents=True, exist_ok=True)
        (self.workspace_dir / "working").mkdir(parents=True, exist_ok=True)
        (self.workspace_dir / "submission").mkdir(parents=True, exist_ok=True)
        (self.workspace_dir / "best_solution").mkdir(parents=True, exist_ok=True)

        # 链接或复制输入数据
        if self.config.data.data_dir:
            self._setup_input_data()

        log_msg("INFO", f"工作空间初始化完成: {self.workspace_dir}")

    def _setup_input_data(self) -> None:
        """设置输入数据（symlink 或复制）。"""
        src = Path(self.config.data.data_dir).resolve()
        dst = self.workspace_dir / "input"

        if not src.exists():
            log_msg("WARNING", f"输入数据目录不存在: {src}")
            return

        if self.config.data.copy_data:
            # 复制数据
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            log_msg("INFO", f"已复制输入数据: {src} -> {dst}")
        else:
            # 使用符号链接
            for item in src.iterdir():
                link_path = dst / item.name
                if link_path.exists() or link_path.is_symlink():
                    if link_path.is_symlink():
                        link_path.unlink()
                    elif link_path.is_dir():
                        shutil.rmtree(link_path)
                    else:
                        link_path.unlink()

                os.symlink(item, link_path)

            log_msg("INFO", f"已创建输入数据符号链接: {src} -> {dst}")

    def rewrite_submission_path(self, code: str, node_id: str) -> str:
        """重写代码中的 submission 文件路径。

        将 submission.csv 替换为 submission_{node_id}.csv，
        避免并发执行时的文件冲突。

        Args:
            code: 原始代码
            node_id: 节点 ID

        Returns:
            重写后的代码
        """
        submission_file = f"submission_{node_id}.csv"
        modified = code

        # 替换各种可能的写法
        patterns = [
            (r"submission/submission\.csv", f"submission/{submission_file}"),
            (r"/submission\.csv", f"/{submission_file}"),
            (r"to_csv\(['\"]submission\.csv", f"to_csv('submission/{submission_file}"),
            (r"['\"]submission\.csv['\"]", f"'{submission_file}'"),
        ]

        for pattern, replacement in patterns:
            modified = re.sub(pattern, replacement, modified)

        if modified != code:
            log_msg("INFO", f"已重写 submission 路径: submission_{node_id}.csv")

        return modified

    def get_submission_path(self, node_id: str) -> Path:
        """获取节点的 submission 文件路径。

        Args:
            node_id: 节点 ID

        Returns:
            submission 文件路径
        """
        return self.workspace_dir / "submission" / f"submission_{node_id}.csv"

    def cleanup_submission(self, node_id: Optional[str] = None) -> None:
        """清理 submission 文件。

        Args:
            node_id: 节点 ID（如果为 None，清理所有）
        """
        submission_dir = self.workspace_dir / "submission"

        if node_id:
            path = submission_dir / f"submission_{node_id}.csv"
            if path.exists():
                path.unlink()
        else:
            shutil.rmtree(submission_dir, ignore_errors=True)
            submission_dir.mkdir(exist_ok=True)
```

---

### 3.6 PromptBuilder Prompt 构建器

**文件路径**: `utils/prompt_builder.py`

**设计理念**:
- 简化版 Prompt 构建（不使用 Jinja2，直接字符串拼接）
- 参考 AIDE 的 prompt 结构
- 支持三种任务类型的 prompt

```python
"""Prompt 构建器，负责生成 LLM 调用的 prompt。"""

from typing import Optional
import humanize


class PromptBuilder:
    """Prompt 构建器。

    负责为不同任务类型构建结构化的 prompt：
    - draft: 初始方案生成
    - improve: 方案改进
    - debug: bug 修复
    """

    def __init__(self, obfuscate: bool = False) -> None:
        """初始化 Prompt 构建器。

        Args:
            obfuscate: 是否隐藏 Kaggle 竞赛相关描述
        """
        self.obfuscate = obfuscate

    def _get_role_intro(self, task_type: str) -> str:
        """获取角色介绍。"""
        if self.obfuscate:
            base = "You are an expert machine learning engineer attempting a task."
        else:
            base = "You are a Kaggle grandmaster attending a competition."

        if task_type == "draft":
            return (
                f"{base} In order to win this competition, you need to come up with "
                "an excellent and creative plan for a solution and then implement "
                "this solution in Python. We will now provide a description of the task."
            )
        elif task_type == "improve":
            return (
                f"{base} You are provided with a previously developed solution below "
                "and should improve it in order to further increase the (test time) performance. "
                "For this you should first outline a brief plan in natural language for how "
                "the solution can be improved and then implement this improvement in Python "
                "based on the provided previous solution."
            )
        elif task_type == "debug":
            return (
                f"{base} Your previous solution had a bug and/or did not produce a submission.csv, "
                "so based on the information below, you should revise it in order to fix this. "
                "Your response should be an implementation outline in natural language, "
                "followed by a single markdown code block which implements the bugfix/solution."
            )
        else:
            return base

    def _get_environment_info(self) -> str:
        """获取环境信息。"""
        packages = [
            "numpy", "pandas", "scikit-learn", "statsmodels",
            "xgboost", "lightGBM", "torch", "torchvision",
            "torch-geometric", "bayesian-optimization", "timm"
        ]
        import random
        random.shuffle(packages)
        pkg_str = ", ".join(f"`{p}`" for p in packages)

        return (
            f"**Installed Packages**: Your solution can use any relevant machine learning "
            f"packages such as: {pkg_str}. Feel free to use any other packages too "
            "(all packages are already installed!). For neural networks we suggest using "
            "PyTorch rather than TensorFlow."
        )

    def _get_guidelines(
        self,
        time_remaining: int,
        steps_remaining: int
    ) -> str:
        """获取实现指南。"""
        return f"""**Implementation Guidelines**:
- <TOTAL_TIME_REMAINING: {humanize.naturaldelta(time_remaining)}>
- <TOTAL_STEPS_REMAINING: {steps_remaining}>
- The code should **implement the proposed solution**, **print the value of the evaluation metric computed on a hold-out validation set**
- **SAVE PREDICTIONS ON THE PROVIDED UNLABELED TEST DATA IN A `submission.csv` FILE IN THE ./submission/ DIRECTORY**
- The code should be a single-file python program that is self-contained and can be executed as-is
- Your response should only contain a single code block
- All the provided input data is stored in "./input" directory
- You can use the "./working" directory to store any temporary files
- REMEMBER THE ./submission/submission.csv FILE!"""

    def _get_response_format(self) -> str:
        """获取响应格式说明。"""
        return (
            "**Response Format**: Your response should be a brief outline/sketch of your "
            "proposed solution in natural language (3-5 sentences), followed by a single "
            "markdown code block (wrapped in ```) which implements this solution and prints "
            "out the evaluation metric. There should be no additional headings or text in "
            "your response."
        )

    def build_draft_prompt(
        self,
        task_desc: str,
        memory: str,
        data_preview: Optional[str],
        time_remaining: int,
        steps_remaining: int
    ) -> str:
        """构建初始草稿 prompt。

        Args:
            task_desc: 任务描述
            memory: 历史解决方案摘要
            data_preview: 数据预览
            time_remaining: 剩余时间（秒）
            steps_remaining: 剩余步数

        Returns:
            完整的 prompt 字符串
        """
        sections = [
            f"# Introduction\n{self._get_role_intro('draft')}",
            f"# Task Description\n{task_desc}",
        ]

        if memory:
            sections.append(f"# Memory\n{memory}")

        if data_preview:
            sections.append(f"# Data Overview\n{data_preview}")

        sections.extend([
            f"# Environment\n{self._get_environment_info()}",
            f"# Guidelines\n{self._get_guidelines(time_remaining, steps_remaining)}",
            f"# Response Format\n{self._get_response_format()}",
            """# Solution Sketch Guidelines
- This first solution design should be relatively simple, without ensembling or hyper-parameter optimization
- Take the Memory section into consideration when proposing the design, don't propose the same modelling solution
- The solution sketch should be 3-5 sentences
- Propose an evaluation metric that is reasonable for this task
- Don't suggest to do EDA
- The data is already prepared and available in the `./input` directory"""
        ])

        return "\n\n".join(sections)

    def build_improve_prompt(
        self,
        task_desc: str,
        parent_code: str,
        parent_output: str,
        memory: str,
        time_remaining: int,
        steps_remaining: int
    ) -> str:
        """构建改进 prompt。

        Args:
            task_desc: 任务描述
            parent_code: 父节点代码
            parent_output: 父节点执行输出
            memory: 历史解决方案摘要
            time_remaining: 剩余时间（秒）
            steps_remaining: 剩余步数

        Returns:
            完整的 prompt 字符串
        """
        sections = [
            f"# Introduction\n{self._get_role_intro('improve')}",
            f"# Task Description\n{task_desc}",
        ]

        if memory:
            sections.append(f"# Memory\n{memory}")

        sections.extend([
            f"# Previous Solution\n```python\n{parent_code}\n```",
            f"# Previous Execution Output\n```\n{parent_output}\n```",
            f"# Guidelines\n{self._get_guidelines(time_remaining, steps_remaining)}",
            f"# Response Format\n{self._get_response_format()}",
            """# Solution Improvement Guidelines
- The solution sketch should be a brief natural language description of how the previous solution can be improved
- You should be very specific and should only propose a single actionable improvement
- This improvement should be atomic so that we can experimentally evaluate the effect of the proposed change
- Take the Memory section into consideration when proposing the improvement
- The solution sketch should be 3-5 sentences
- Don't suggest to do EDA"""
        ])

        return "\n\n".join(sections)

    def build_debug_prompt(
        self,
        task_desc: str,
        buggy_code: str,
        error_output: str,
        data_preview: Optional[str],
        time_remaining: int,
        steps_remaining: int
    ) -> str:
        """构建调试 prompt。

        Args:
            task_desc: 任务描述
            buggy_code: 有 bug 的代码
            error_output: 错误输出
            data_preview: 数据预览
            time_remaining: 剩余时间（秒）
            steps_remaining: 剩余步数

        Returns:
            完整的 prompt 字符串
        """
        sections = [
            f"# Introduction\n{self._get_role_intro('debug')}",
            f"# Task Description\n{task_desc}",
            f"# Previous (Buggy) Implementation\n```python\n{buggy_code}\n```",
            f"# Execution Output\n```\n{error_output}\n```",
        ]

        if data_preview:
            sections.append(f"# Data Overview\n{data_preview}")

        sections.extend([
            f"# Guidelines\n{self._get_guidelines(time_remaining, steps_remaining)}",
            f"# Response Format\n{self._get_response_format()}",
            """# Bugfix Guidelines
- You should write a brief natural language description (3-5 sentences) of how the issue in the previous implementation can be fixed
- Don't suggest to do EDA"""
        ])

        return "\n\n".join(sections)
```

---

### 3.7 Node 数据类修改

**文件路径**: `core/state/node.py`

**修改内容**: 新增 `absorb_exec_result()` 方法和 `term_out` 属性

```python
# 在现有 Node 类中添加以下内容

from core.executor.interpreter import ExecutionResult

@dataclass(eq=False)
class Node(DataClassJsonMixin):
    # ... 现有字段 ...

    # 新增执行信息字段
    _term_out: list[str] = field(default_factory=list, kw_only=True)

    def absorb_exec_result(self, exec_result: ExecutionResult) -> None:
        """吸收执行结果到节点。

        Args:
            exec_result: 执行结果对象
        """
        self._term_out = exec_result.term_out
        self.exec_time = exec_result.exec_time
        self.exc_type = exec_result.exc_type
        self.exc_info = exec_result.exc_info

    @property
    def term_out(self) -> str:
        """获取截断后的终端输出。"""
        from utils.response import trim_long_string
        return trim_long_string("".join(self._term_out))

    @property
    def is_leaf(self) -> bool:
        """是否为叶子节点。"""
        # 需要 Journal 上下文判断，Phase 2 简化实现
        return True

    @property
    def debug_depth(self) -> int:
        """调试深度（连续调试的次数）。"""
        if self.parent_id is None:
            return 0
        # 需要 Journal 上下文，Phase 2 简化返回 0
        return 0
```

---

### 3.8 Journal 数据类修改

**文件路径**: `core/state/journal.py`

**修改内容**: 新增 `generate_summary()` 方法用于 Memory 机制

```python
# 在现有 Journal 类中添加以下方法

def generate_summary(self, include_code: bool = False) -> str:
    """生成 Journal 摘要用于 Memory 机制。

    Args:
        include_code: 是否包含代码

    Returns:
        摘要字符串
    """
    if not self.good_nodes:
        return "No previous successful solutions."

    summaries = []
    for node in self.good_nodes:
        parts = [f"Design: {node.plan}"]

        if include_code:
            parts.append(f"Code: {node.code}")

        parts.append(f"Results: {node.analysis}")

        if node.metric_value is not None:
            parts.append(f"Validation Metric: {node.metric_value}")

        summaries.append("\n".join(parts))

    return "\n\n-------------------------------\n\n".join(summaries)
```

---

## 4. 验证计划

### 4.1 单元测试

#### 测试文件: `tests/test_agents.py`

```python
import pytest
from agents.base_agent import BaseAgent, AgentContext, AgentResult
from agents.coder_agent import CoderAgent
from core.state.journal import Journal
from utils.config import load_config
from utils.prompt_builder import PromptBuilder

class TestCoderAgent:
    """CoderAgent 测试类。"""

    def test_agent_init(self):
        """测试 Agent 初始化。"""
        cfg = load_config()
        builder = PromptBuilder()
        agent = CoderAgent("test_agent", cfg, builder)

        assert agent.name == "test_agent"
        assert agent.config == cfg

    @pytest.mark.asyncio
    async def test_draft_generation(self, mock_llm):
        """测试草稿生成（需要 Mock LLM）。"""
        # ... 使用 mock 测试
        pass
```

#### 测试文件: `tests/test_orchestrator.py`

```python
import pytest
from core.orchestrator import Orchestrator
from agents.coder_agent import CoderAgent
from core.state.journal import Journal
from utils.config import load_config
from utils.prompt_builder import PromptBuilder

class TestOrchestrator:
    """Orchestrator 测试类。"""

    def test_orchestrator_init(self):
        """测试 Orchestrator 初始化。"""
        cfg = load_config()
        journal = Journal()
        builder = PromptBuilder()
        agent = CoderAgent("test", cfg, builder)

        orch = Orchestrator(agent, cfg, journal, "test task")

        assert orch.current_step == 0
        assert orch.best_node is None

    def test_search_policy_initial(self):
        """测试初始搜索策略。"""
        # ... 验证初始状态返回 draft
        pass
```

#### 测试文件: `tests/test_interpreter.py`

```python
import pytest
from pathlib import Path
from core.executor.interpreter import Interpreter, ExecutionResult

class TestInterpreter:
    """Interpreter 测试类。"""

    def test_simple_execution(self, tmp_path):
        """测试简单代码执行。"""
        interpreter = Interpreter(tmp_path, timeout=10)

        result = interpreter.run("print('hello')")

        assert result.exc_type is None
        assert "hello" in "".join(result.term_out)

    def test_timeout(self, tmp_path):
        """测试超时处理。"""
        interpreter = Interpreter(tmp_path, timeout=1)

        result = interpreter.run("import time; time.sleep(10)")

        assert result.exc_type == "TimeoutError"

    def test_exception_capture(self, tmp_path):
        """测试异常捕获。"""
        interpreter = Interpreter(tmp_path, timeout=10)

        result = interpreter.run("raise ValueError('test error')")

        assert result.exc_type == "ValueError"
```

#### 测试文件: `tests/test_workspace.py`

```python
import pytest
from pathlib import Path
from core.executor.workspace import WorkspaceManager
from utils.config import load_config

class TestWorkspaceManager:
    """WorkspaceManager 测试类。"""

    def test_rewrite_submission_path(self, tmp_path):
        """测试 submission 路径重写。"""
        cfg = load_config()
        cfg.project.workspace_dir = tmp_path

        ws = WorkspaceManager(cfg)

        code = 'df.to_csv("submission/submission.csv")'
        node_id = "abc123"

        result = ws.rewrite_submission_path(code, node_id)

        assert f"submission_{node_id}.csv" in result
```

### 4.2 集成测试

#### 测试文件: `tests/integration/test_phase2.py`

```python
import pytest
from pathlib import Path

class TestPhase2Integration:
    """Phase 2 集成测试。"""

    def test_full_step_cycle(self, tmp_path, mock_llm):
        """测试完整的单步执行周期。"""
        from utils.config import load_config
        from core.state.journal import Journal
        from agents.coder_agent import CoderAgent
        from core.orchestrator import Orchestrator
        from utils.prompt_builder import PromptBuilder

        # 设置配置
        cfg = load_config()
        cfg.project.workspace_dir = tmp_path

        # 初始化组件
        journal = Journal()
        builder = PromptBuilder()
        agent = CoderAgent("test", cfg, builder)
        orch = Orchestrator(agent, cfg, journal, "test task")

        # 执行单步
        orch.step()

        # 验证
        assert len(journal) == 1
```

### 4.3 手动验证

```bash
# 1. 运行单元测试
pytest tests/test_agents.py -v
pytest tests/test_orchestrator.py -v
pytest tests/test_interpreter.py -v
pytest tests/test_workspace.py -v

# 2. 运行集成测试
pytest tests/integration/test_phase2.py -v

# 3. 检查覆盖率
pytest tests/ --cov=agents --cov=core --cov-report=term-missing

# 4. 简单端到端测试
python -c "
from utils.config import load_config
from core.state.journal import Journal
from agents.coder_agent import CoderAgent
from core.orchestrator import Orchestrator
from utils.prompt_builder import PromptBuilder

cfg = load_config()
journal = Journal()
builder = PromptBuilder()
agent = CoderAgent('test', cfg, builder)
orch = Orchestrator(agent, cfg, journal, 'Test task')

print('Orchestrator 初始化成功')
print(f'工作目录: {cfg.project.workspace_dir}')
"
```

---

## 5. 风险与缓解

### Risk 1: Interpreter 进程管理复杂
**影响**: 高
**概率**: 中
**缓解**:
- 参考 AIDE 的成熟实现
- 添加详细的日志记录
- 完善进程清理逻辑

### Risk 2: 搜索策略过于简单
**影响**: 中
**概率**: 低（Phase 3 扩展）
**缓解**:
- Phase 2 只实现简单策略
- 预留接口供 Phase 3 扩展
- 通过配置切换策略

### Risk 3: Prompt 质量影响结果
**影响**: 高
**概率**: 中
**缓解**:
- 复用 AIDE 验证过的 prompt 结构
- 支持 obfuscate 模式
- 后续可通过遗传算法优化

### Risk 4: 并发执行文件冲突
**影响**: 高（Phase 2 暂不并发）
**概率**: 低
**缓解**:
- Phase 2 使用串行执行
- 已实现文件名重写机制
- Phase 3 启用并行时激活

---

## 6. 成功标准

### 6.1 功能完整性
- [ ] Orchestrator 可成功执行单步搜索
- [ ] CoderAgent 可生成 draft/improve/debug 代码
- [ ] Interpreter 可执行代码并捕获输出
- [ ] WorkspaceManager 可正确重写文件路径
- [ ] PromptBuilder 可生成有效 prompt

### 6.2 测试覆盖率
- [ ] 单元测试覆盖率 >= 80%
- [ ] 所有集成测试通过
- [ ] 手动验证步骤全部通过

### 6.3 代码质量
- [ ] Ruff 格式化通过
- [ ] 所有函数包含中文 Docstring
- [ ] 无 print() 语句（使用 log_msg）

---

## 7. 文件清单汇总

**新增文件** (8 个):
```
agents/__init__.py                     [NEW]
agents/base_agent.py                   [NEW]
agents/coder_agent.py                  [NEW]
core/orchestrator.py                   [NEW]
core/executor/__init__.py              [NEW]
core/executor/interpreter.py           [NEW]
core/executor/workspace.py             [NEW]
utils/prompt_builder.py                [NEW]
```

**修改文件** (2 个):
```
core/state/node.py                     [MODIFY] - 新增 absorb_exec_result(), term_out
core/state/journal.py                  [MODIFY] - 新增 generate_summary()
```

**测试文件** (5 个):
```
tests/test_agents.py                   [NEW]
tests/test_orchestrator.py             [NEW]
tests/test_interpreter.py              [NEW]
tests/test_workspace.py                [NEW]
tests/integration/test_phase2.py       [NEW]
```

---

**计划版本**: Phase2-v1.0
**创建日期**: 2026-01-29
**状态**: 待审查
