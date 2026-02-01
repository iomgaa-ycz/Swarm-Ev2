"""Orchestrator 任务编排器模块。

负责控制主循环、选择父节点、协调 Agent 生成代码、执行代码、Review 评估、更新最佳节点等核心流程。
支持双层进化：Solution 层（Step 循环）+ Agent 层（Epoch 循环）。
"""

import time
import random
import json
import shutil
from typing import Optional, Dict, TYPE_CHECKING

from agents.base_agent import BaseAgent, AgentContext
from core.state import Node, Journal
from core.executor.interpreter import Interpreter, ExecutionResult
from core.executor.workspace import WorkspaceManager
from core.backend import query as backend_query
from utils.config import Config
from utils.logger_system import log_msg, log_exception

if TYPE_CHECKING:
    from core.evolution.agent_evolution import AgentEvolution


class Orchestrator:
    """任务编排器（双层进化模式）。

    控制主循环与搜索流程，协调 Agent、Interpreter、Review 等模块。
    支持双层进化：
        - Solution 层：每个 Epoch 内多个 Step，生成/评估/更新节点
        - Agent 层：每 N 个 Epoch 触发 Agent 进化（Role/Strategy 变异 + Skill 池更新）

    Attributes:
        agent: 代码生成 Agent 实例
        config: 全局配置对象
        journal: 历史节点记录
        task_desc: 任务描述字符串
        agent_evolution: Agent 层进化器（可选，P3.5 使用）
        start_time: 任务开始时间（用于计算剩余时间）
        current_step: 当前步数
        current_epoch: 当前 Epoch 编号
        best_node: 当前最佳节点
        workspace: 工作空间管理器
        interpreter: 代码执行器
    """

    def __init__(
        self,
        agent: BaseAgent,
        config: Config,
        journal: Journal,
        task_desc: str,
        agent_evolution: Optional["AgentEvolution"] = None,
    ):
        """初始化 Orchestrator。

        Args:
            agent: 代码生成 Agent 实例
            config: 全局配置对象
            journal: 历史节点记录
            task_desc: 任务描述字符串
            agent_evolution: Agent 层进化器（可选，P3.5 使用）
        """
        self.agent = agent
        self.config = config
        self.journal = journal
        self.task_desc = task_desc
        self.agent_evolution = agent_evolution

        self.start_time = time.time()
        self.current_step = 0
        self.current_epoch = 0
        self.best_node: Optional[Node] = None

        # 初始化工作空间管理器
        self.workspace = WorkspaceManager(config)

        # 初始化代码执行器
        self.interpreter = Interpreter(
            working_dir=str(config.project.workspace_dir / "working"),
            timeout=config.execution.timeout,
        )

        log_msg(
            "INFO",
            f"Orchestrator 初始化完成: task={task_desc[:50]}..., "
            f"agent_evolution={'启用' if agent_evolution else '禁用'}",
        )

    def run(
        self,
        num_epochs: int = 1,
        steps_per_epoch: Optional[int] = None,
    ) -> Optional[Node]:
        """主循环入口（双层进化模式）。

        双层循环结构：
            - 外层：Epoch 循环，每个 Epoch 结束时触发 Agent 层进化
            - 内层：Step 循环，每个 Step 执行 Solution 层进化（生成/评估/更新节点）

        Args:
            num_epochs: Epoch 数量（默认 1，兼容原有逻辑）
            steps_per_epoch: 每个 Epoch 的步数（默认使用 config.evolution.solution.steps_per_epoch
                            或 config.agent.max_steps）

        Returns:
            最佳节点对象（可能为 None）
        """
        # 确定每个 Epoch 的步数
        if steps_per_epoch is None:
            # 优先使用 evolution.solution.steps_per_epoch，否则使用 agent.max_steps
            if hasattr(self.config, "evolution") and hasattr(
                self.config.evolution, "solution"
            ):
                steps_per_epoch = self.config.evolution.solution.steps_per_epoch
            else:
                steps_per_epoch = self.config.agent.max_steps

        total_steps = num_epochs * steps_per_epoch
        log_msg(
            "INFO",
            f"Orchestrator 开始运行: num_epochs={num_epochs}, "
            f"steps_per_epoch={steps_per_epoch}, total_steps={total_steps}",
        )

        # Epoch 循环
        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # 检查时间限制
            if self._check_time_limit():
                break

            log_msg("INFO", f"===== Epoch {epoch + 1}/{num_epochs} 开始 =====")

            # Step 循环（Solution 层进化）
            epoch_completed = self._run_single_epoch(steps_per_epoch)

            if not epoch_completed:
                # 时间限制触发，提前退出
                break

            # Agent 层进化（每个 Epoch 结束时）
            if self.agent_evolution:
                self.agent_evolution.evolve(epoch)

            # Epoch 结束日志
            best = self.journal.get_best_node()
            log_msg(
                "INFO",
                f"===== Epoch {epoch + 1}/{num_epochs} 完成 | "
                f"最佳 metric: {best.metric_value if best else 'N/A'} =====",
            )

        log_msg(
            "INFO",
            f"Orchestrator 运行完成: best_node={'存在' if self.best_node else '不存在'}",
        )
        return self.best_node

    def run_legacy(self, max_steps: Optional[int] = None) -> Optional[Node]:
        """原有主循环入口（兼容旧接口）。

        Args:
            max_steps: 最大步数，默认使用 config.agent.max_steps

        Returns:
            最佳节点对象（可能为 None）
        """
        steps = max_steps or self.config.agent.max_steps
        return self.run(num_epochs=1, steps_per_epoch=steps)

    def _run_single_epoch(self, steps_per_epoch: int) -> bool:
        """运行单个 Epoch（Solution 层进化）。

        Args:
            steps_per_epoch: 该 Epoch 的步数

        Returns:
            是否正常完成（False 表示因时间限制提前退出）
        """
        for step in range(steps_per_epoch):
            self.current_step = self.current_epoch * steps_per_epoch + step

            # 检查时间限制
            if self._check_time_limit():
                return False

            log_msg(
                "INFO",
                f"=== Epoch {self.current_epoch + 1} | "
                f"Step {step + 1}/{steps_per_epoch} ===",
            )
            self.step()

        return True

    def _check_time_limit(self) -> bool:
        """检查是否达到时间限制。

        Returns:
            是否已达时间限制
        """
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= self.config.agent.time_limit:
            log_msg(
                "INFO",
                f"已达时间限制 {self.config.agent.time_limit}s，停止运行",
            )
            return True
        return False

    def step(self) -> None:
        """单步执行流程。

        流程：
        1. 清理 submission 目录
        2. 选择父节点
        3. 调用 Agent 生成代码
        4. 执行代码
        5. Review 评估
        6. 保存节点代码和输出
        7. 更新 Journal 和 best_node
        """
        try:
            # Phase 1: 准备环境
            self._prepare_step()

            # Phase 2: 选择父节点
            parent_node = self._select_parent_node()

            # Phase 3: 生成代码
            context = AgentContext(
                task_type="explore",
                parent_node=parent_node,
                journal=self.journal,
                config=self.config,
                start_time=self.start_time,
                current_step=self.current_step,
                task_desc=self.task_desc,
            )
            result = self.agent.generate(context)

            if not result.success or result.node is None:
                log_msg("WARNING", f"Agent 生成失败: {result.error}")
                return

            node = result.node

            # Phase 4: 执行代码
            exec_result = self._execute_code(node.code, node.id)
            node.term_out = "\n".join(exec_result.term_out)
            node.exec_time = exec_result.exec_time
            node.exc_type = exec_result.exc_type
            node.exc_info = str(exec_result.exc_info) if exec_result.exc_info else None

            # Phase 5: Review 评估
            self._review_node(node)

            # Phase 6: 保存节点代码和输出
            self._save_node_solution(node)

            # Phase 7: 更新状态
            self.journal.append(node)
            self._update_best_node(node)

            # Phase 8: 打印评估结果
            self._print_node_summary(node)

        except Exception as e:
            log_exception(e, "Orchestrator step() 执行失败")

    def _select_parent_node(self) -> Optional[Node]:
        """选择父节点（搜索策略）。

        三阶段策略：
        1. 初稿模式：draft 数量不足时生成初稿
        2. 修复模式：概率触发，修复 buggy 叶子节点
        3. 改进模式：选择 best_node 进行改进

        Returns:
            - None: 初稿模式
            - buggy node: 修复模式
            - best node: 改进模式
        """
        # Phase 1: 初稿模式（draft 数量不足）
        if len(self.journal.draft_nodes) < self.config.search.num_drafts:
            log_msg("INFO", "[search_policy] 初稿模式")
            return None

        # Phase 2: 修复模式（概率触发，优先修复 buggy 叶子节点）
        if random.random() < self.config.search.debug_prob:
            # 构建 DAG 以获取 children_ids
            self.journal.build_dag()

            # 查找 buggy 叶子节点
            buggy_leaves = [n for n in self.journal.buggy_nodes if not n.children_ids]

            if buggy_leaves:
                node = random.choice(buggy_leaves)
                log_msg("INFO", f"[search_policy] 修复模式: 节点 {node.id[:8]}")
                return node

        # Phase 3: 改进模式（选择 best_node）
        best = self.journal.get_best_node(only_good=True)
        if best:
            log_msg("INFO", f"[search_policy] 改进模式: 节点 {best.id[:8]}")
            return best
        else:
            log_msg("INFO", "[search_policy] 初稿模式（无可用节点）")
            return None

    def _prepare_step(self) -> None:
        """准备单步执行环境。

        清理 submission 目录，避免文件冲突。
        """
        submission_dir = self.config.project.workspace_dir / "submission"
        if submission_dir.exists():
            # 清空目录（保留目录本身）
            for item in submission_dir.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)

    def _execute_code(self, code: str, node_id: str) -> ExecutionResult:
        """执行代码。

        Args:
            code: Python 代码字符串
            node_id: 节点 ID（用于重写 submission 路径）

        Returns:
            ExecutionResult 对象
        """
        # 使用 WorkspaceManager 重写 submission 路径
        modified_code = self.workspace.rewrite_submission_path(code, node_id)

        # 执行代码（reset_session=True 确保每次独立执行）
        return self.interpreter.run(modified_code, reset_session=True)

    def _review_node(self, node: Node) -> None:
        """Review 评估节点（使用 Function Calling）。

        Args:
            node: 待评估的节点对象

        Side effects:
            更新 node 的 analysis, is_buggy, metric_value, lower_is_better 字段
        """
        try:
            # 构建 messages
            messages_content = self._build_review_messages(node)

            # 获取 tool schema
            tool_schema = self._get_review_tool_schema()

            # 调用 LLM（Function Calling）
            response = backend_query(
                system_message=None,
                user_message=messages_content,
                model=self.config.llm.feedback.model,
                provider=self.config.llm.feedback.provider,
                temperature=self.config.llm.feedback.temperature,
                api_key=self.config.llm.feedback.api_key,
                base_url=getattr(self.config.llm.feedback, "base_url", None),
                tools=[{"type": "function", "function": tool_schema}],
                tool_choice={
                    "type": "function",
                    "function": {"name": "submit_review"},
                },
            )

            # 解析 Function Calling 响应（已经是 JSON 字符串）
            review_data = json.loads(response)

            # 更新节点信息
            node.analysis = review_data.get("summary", "")
            node.is_buggy = (
                review_data.get("is_bug", False) or node.exc_type is not None
            )
            node.metric_value = review_data.get("metric")
            node.lower_is_better = review_data.get("lower_is_better", False)

            log_msg(
                "INFO",
                f"Review 完成: 节点 {node.id[:8]}, metric={node.metric_value}, lower_is_better={node.lower_is_better}",
            )

        except Exception as e:
            log_exception(e, "Review 评估失败")
            node.analysis = f"Review 失败: {str(e)}"
            node.is_buggy = True
            node.metric_value = None
            node.lower_is_better = False

    def _build_review_messages(self, node: Node) -> str:
        """构建 Review 消息（用于 Function Calling）。

        Args:
            node: 节点对象

        Returns:
            消息内容字符串
        """
        return f"""You are evaluating a machine learning solution.

**Task Description:**
{self.task_desc}

**Code:**
```python
{node.code}
```

**Execution Output:**
```
{node.term_out}
```

**Execution Status:**
- Execution Time: {node.exec_time:.2f}s
- Exception: {node.exc_type or "None"}

---

Please analyze the execution results and call the `submit_review` function with your assessment.
"""

    def _get_review_tool_schema(self) -> Dict:
        """获取 Review Function Calling 的 schema。

        Returns:
            tool schema 字典
        """
        return {
            "name": "submit_review",
            "description": "提交代码评估结果",
            "parameters": {
                "type": "object",
                "properties": {
                    "is_bug": {
                        "type": "boolean",
                        "description": "代码是否有 bug 或执行失败",
                    },
                    "has_csv_submission": {
                        "type": "boolean",
                        "description": "代码是否生成了 submission.csv 文件",
                    },
                    "summary": {
                        "type": "string",
                        "description": "2-3 句话的结果摘要",
                    },
                    "metric": {
                        "type": "number",
                        "description": "验证集指标值（如准确率、RMSE），失败时为 null",
                        "nullable": True,
                    },
                    "lower_is_better": {
                        "type": "boolean",
                        "description": "指标是否越小越好（如 RMSE=true, Accuracy=false）",
                    },
                },
                "required": ["is_bug", "summary", "lower_is_better"],
            },
        }

    def _update_best_node(self, node: Node) -> None:
        """更新最佳节点（支持 lower_is_better）。

        策略：
        1. 优先选择非 buggy 节点
        2. 如果都是 buggy，选择指标最好的（作为参考）
        3. 如果没有指标，不更新

        Args:
            node: 候选节点对象
        """
        # 跳过无指标节点
        if node.metric_value is None:
            return

        # 初始化 best_node（即使是 buggy）
        if self.best_node is None:
            log_msg(
                "INFO",
                f"初始化最佳节点: {node.id[:8]}, metric={node.metric_value}, buggy={node.is_buggy}",
            )
            self.best_node = node
            self._save_best_solution(node)
            return

        # 优先级：非buggy > buggy
        current_is_good = not node.is_buggy
        best_is_good = not self.best_node.is_buggy

        should_update = False

        if current_is_good and not best_is_good:
            # 当前节点成功，最佳节点失败 -> 必须更新
            should_update = True
        elif not current_is_good and best_is_good:
            # 当前节点失败，最佳节点成功 -> 不更新
            should_update = False
        else:
            # 都成功 或 都失败 -> 比较指标
            if self._is_better(node, self.best_node):
                should_update = True

        if should_update:
            direction = "↓" if node.lower_is_better else "↑"
            status = "✅ 成功" if current_is_good else "⚠️  失败（参考）"
            log_msg(
                "INFO",
                f"新的最佳节点: {node.id[:8]}, metric={node.metric_value} {direction}, {status}",
            )
            self.best_node = node
            self._save_best_solution(node)

    def _save_best_solution(self, node: Node) -> None:
        """保存最佳解决方案到文件。

        Args:
            node: 最佳节点对象
        """
        try:
            best_dir = self.config.project.workspace_dir / "best_solution"
            best_dir.mkdir(exist_ok=True, parents=True)

            # 保存代码
            with open(best_dir / "solution.py", "w", encoding="utf-8") as f:
                f.write(node.code)

            # 复制 submission 文件（如果存在）
            submission_src = (
                self.config.project.workspace_dir
                / "submission"
                / f"submission_{node.id}.csv"
            )
            if submission_src.exists():
                shutil.copy(submission_src, best_dir / "submission.csv")

            log_msg("INFO", f"最佳方案已保存到 {best_dir}")

        except Exception as e:
            log_exception(e, "保存最佳方案失败")

    def _save_node_solution(self, node: Node) -> None:
        """保存节点代码和输出到独立目录。

        Args:
            node: 节点对象
        """
        try:
            # 创建节点专属目录
            node_dir = (
                self.config.project.workspace_dir
                / "working"
                / f"solution_{node.id[:8]}"
            )
            node_dir.mkdir(exist_ok=True, parents=True)

            # 保存代码
            with open(node_dir / "solution.py", "w", encoding="utf-8") as f:
                f.write(node.code)

            # 保存执行输出
            with open(node_dir / "output.txt", "w", encoding="utf-8") as f:
                f.write(f"执行时间: {node.exec_time:.2f}s\n")
                f.write(f"异常类型: {node.exc_type or 'None'}\n")
                f.write(f"是否有Bug: {node.is_buggy}\n")
                f.write(f"评估指标: {node.metric_value}\n")
                f.write("\n=== 终端输出 ===\n")
                f.write(node.term_out or "")
                if node.exc_info:
                    f.write("\n\n=== 异常信息 ===\n")
                    f.write(node.exc_info)

            # 复制 submission 文件（如果存在）
            submission_src = (
                self.config.project.workspace_dir
                / "submission"
                / f"submission_{node.id}.csv"
            )
            if submission_src.exists():
                shutil.copy(submission_src, node_dir / "submission.csv")

            log_msg("INFO", f"节点 {node.id[:8]} 已保存到 {node_dir}")

        except Exception as e:
            log_exception(e, f"保存节点 {node.id[:8]} 失败")

    def _print_node_summary(self, node: Node) -> None:
        """打印节点评估摘要（日志+控制台）。

        Args:
            node: 节点对象
        """
        # 构建评估信息
        status = "❌ BUGGY" if node.is_buggy else "✅ SUCCESS"
        metric_str = f"{node.metric_value}" if node.metric_value is not None else "N/A"
        direction = "↓ (越小越好)" if node.lower_is_better else "↑ (越大越好)"

        summary = (
            f"{status} | 节点 {node.id[:8]} | "
            f"指标: {metric_str} {direction} | "
            f"执行: {node.exec_time:.2f}s"
        )

        # 打印到日志和控制台
        log_msg("INFO", f"[评估] {summary}")
        print(f"\n  {summary}")

        # 如果是最佳节点，额外高亮
        if not node.is_buggy and node.metric_value is not None:
            if self.best_node is None or self._is_better(node, self.best_node):
                print("  🎉 新的最佳节点！")
                log_msg("INFO", f"[最佳] 节点 {node.id[:8]} 成为新的最佳方案")

    def _is_better(self, node: Node, best_node: Node) -> bool:
        """判断节点是否优于最佳节点。

        Args:
            node: 候选节点
            best_node: 当前最佳节点

        Returns:
            是否更好
        """
        if node.metric_value is None or best_node.metric_value is None:
            return False

        if node.lower_is_better:
            return node.metric_value < best_node.metric_value
        else:
            return node.metric_value > best_node.metric_value
