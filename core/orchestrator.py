"""Orchestrator 任务编排器模块（并行版本）。

负责控制主循环、选择父节点、协调多个 Agent 并行生成代码、执行代码、Review 评估、更新最佳节点等核心流程。
支持双层进化：Solution 层（Step 循环）+ Agent 层（Epoch 循环）。

参考: Reference/ML-Master-main/main_mcts.py
"""

import difflib
import random
import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from typing import Optional, Dict, List, TYPE_CHECKING

from agents.base_agent import BaseAgent, AgentContext
from core.state import Node, Journal
from core.executor.interpreter import Interpreter, ExecutionResult
from core.executor.workspace import WorkspaceManager
from core.backend import query as backend_query
from utils.config import Config
from utils.logger_system import log_msg, log_exception
from utils.system_info import (
    get_hardware_description,
    get_conda_packages,
    get_conda_python_path,
)

if TYPE_CHECKING:
    from core.evolution.agent_evolution import AgentEvolution


class Orchestrator:
    """任务编排器（并行执行模式）。

    控制主循环与搜索流程，协调多个 Agent 并行工作。
    支持双层进化：
        - Solution 层：每个 Epoch 内多个 Step 并行执行
        - Agent 层：每 N 个 Epoch 触发 Agent 进化

    Attributes:
        agents: Agent 列表（多个 Agent 并行工作）
        config: 全局配置对象
        journal: 历史节点记录
        task_desc: 任务描述字符串
        agent_evolution: Agent 层进化器（可选）
        start_time: 任务开始时间
        best_node: 当前最佳节点
        workspace: 工作空间管理器
        interpreter: 代码执行器（支持并行）
        max_workers: 最大并行工作线程数
        journal_lock: Journal 访问锁
        save_lock: 文件保存锁
    """

    def __init__(
        self,
        agents: List[BaseAgent],
        config: Config,
        journal: Journal,
        task_desc: str,
        agent_evolution: Optional["AgentEvolution"] = None,
        task_dispatcher=None,
        experience_pool=None,
        gene_registry=None,
    ):
        """初始化 Orchestrator。

        Args:
            agents: Agent 列表（支持多 Agent 并行）
            config: 全局配置对象
            journal: 历史节点记录
            task_desc: 任务描述字符串
            agent_evolution: Agent 层进化器（可选）
            task_dispatcher: 任务分发器（Phase 3）
            experience_pool: 经验池（Phase 3）
            gene_registry: 基因注册表（信息素驱动交叉时必需）
        """
        from utils.text_utils import compress_task_desc

        self.agents = agents
        self.config = config
        self.journal = journal
        self.task_desc = task_desc
        self._task_desc_compressed = compress_task_desc(task_desc)
        self.agent_evolution = agent_evolution
        self.task_dispatcher = task_dispatcher
        self.experience_pool = experience_pool
        self.gene_registry = gene_registry

        self.start_time = time.time()
        self.current_epoch = 0
        self.best_node: Optional[Node] = None

        # 并行配置
        self.max_workers = config.search.parallel_num

        # 线程安全锁
        self.journal_lock = threading.Lock()
        self.save_lock = threading.Lock()

        # 初始化工作空间管理器
        self.workspace = WorkspaceManager(config)

        # 获取并缓存环境信息（一次性，启动时获取）
        self.device_info = get_hardware_description()
        self.conda_env_name = getattr(
            getattr(config, "environment", None), "conda_env_name", "Swarm-Evo"
        )

        # 获取 conda 环境的 Python 路径
        conda_python = get_conda_python_path(self.conda_env_name)
        if conda_python:
            log_msg("INFO", f"使用 conda Python: {conda_python}")
        else:
            log_msg(
                "WARNING",
                f"无法获取 conda 环境 '{self.conda_env_name}' 的 Python，使用当前解释器",
            )

        # 初始化代码执行器（使用 conda Python）
        self.interpreter = Interpreter(
            working_dir=str(config.project.workspace_dir),
            timeout=config.execution.timeout,
            max_parallel_run=self.max_workers,
            python_path=conda_python,
        )
        self.conda_packages = get_conda_packages(self.conda_env_name)

        log_msg("INFO", f"环境信息: {self.device_info}")

        log_msg(
            "INFO",
            f"Orchestrator 初始化完成: task={task_desc[:50]}..., "
            f"agents={len(agents)}, parallel={self.max_workers}, "
            f"agent_evolution={'启用' if agent_evolution else '禁用'}",
        )

    def run(
        self,
        num_epochs: int = 1,
        steps_per_epoch: Optional[int] = None,
    ) -> Optional[Node]:
        """主循环入口（并行执行模式）。

        双层循环结构：
            - 外层：Epoch 循环，每个 Epoch 结束时触发 Agent 层进化
            - 内层：Step 循环，并行执行多个任务

        Args:
            num_epochs: Epoch 数量（默认 1）
            steps_per_epoch: 每个 Epoch 的步数

        Returns:
            最佳节点对象（可能为 None）
        """
        # 确定每个 Epoch 的步数
        if steps_per_epoch is None:
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

            # Step 循环（并行执行）
            epoch_completed = self._run_single_epoch(steps_per_epoch)

            if not epoch_completed:
                log_msg("INFO", "Epoch 执行过程中检测到时间限制，停止运行")
                break

            # Agent 层进化
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

        # 清理所有进程
        self.interpreter.cleanup_session(-1)

        return self.best_node

    def run_legacy(self, max_steps: Optional[int] = None) -> Optional[Node]:
        """原有主循环入口（兼容旧接口）。

        Args:
            max_steps: 最大步数

        Returns:
            最佳节点对象
        """
        steps = max_steps or self.config.agent.max_steps
        return self.run(num_epochs=1, steps_per_epoch=steps)

    def _run_single_epoch(self, steps_per_epoch: int) -> bool:
        """运行单个 Epoch（并行执行）。

        使用 ThreadPoolExecutor 并行提交任务，完成一个后立即提交新任务。

        Args:
            steps_per_epoch: 该 Epoch 的步数

        Returns:
            是否正常完成（False 表示因时间限制提前退出）
        """
        # 记录 Epoch 开始时的节点数，用于计算当前 Epoch 完成的任务数
        with self.journal_lock:
            epoch_start_count = len(self.journal.nodes)

        completed = 0  # 当前 Epoch 完成的任务数（局部计数）

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 初始提交任务
            initial_count = min(self.max_workers, steps_per_epoch)
            futures = {
                executor.submit(self._step_task, None) for _ in range(initial_count)
            }

            log_msg("INFO", f"初始提交 {initial_count} 个并行任务")

            while completed < steps_per_epoch:
                # 检查时间限制
                if self._check_time_limit():
                    for fut in futures:
                        fut.cancel()
                    return False

                # 等待任意任务完成
                done, _ = wait(futures, timeout=5.0, return_when=FIRST_COMPLETED)

                for fut in done:
                    futures.remove(fut)

                    try:
                        fut.result()  # 获取结果以触发异常传播
                    except Exception as e:
                        log_msg("WARNING", f"任务执行失败: {e}")

                    # 更新完成计数（当前 Epoch 的完成数 = 总节点数 - Epoch 开始时的节点数）
                    with self.journal_lock:
                        total_nodes = len(self.journal.nodes)
                    completed = total_nodes - epoch_start_count

                    log_msg(
                        "INFO",
                        f"=== Epoch {self.current_epoch + 1} | "
                        f"Step {completed}/{steps_per_epoch} 完成 ===",
                    )

                    # 提交新任务
                    remaining = steps_per_epoch - completed - len(futures)
                    if remaining > 0:
                        parent = self._select_parent_node()
                        futures.add(executor.submit(self._step_task, parent))

        return True

    def _step_task(self, parent_node: Optional[Node]) -> Optional[Node]:
        """执行单个搜索任务（线程安全）。

        Args:
            parent_node: 父节点（None 表示初稿模式）

        Returns:
            生成的节点
        """
        try:
            # Phase 1: 选择 Agent（优先使用 TaskDispatcher，否则随机选择）
            task_type = "explore"  # 默认任务类型

            if self.task_dispatcher:
                agent = self.task_dispatcher.select_agent(task_type)
            else:
                agent = random.choice(self.agents)

            log_msg(
                "INFO",
                f"{agent.name} 开始 {'explore' if parent_node is None else 'improve'} (parent_id={parent_node.id[:8] if parent_node else None})",
            )

            # Phase 2: 准备环境
            self._prepare_step()

            # Phase 3: 生成代码
            with self.journal_lock:
                current_step = len(self.journal.nodes)

            context = AgentContext(
                task_type="explore",  # 统一使用 explore，PromptManager 根据 parent_node 自动适配
                parent_node=parent_node,
                journal=self.journal,
                config=self.config,
                start_time=self.start_time,
                current_step=current_step,
                task_desc=self.task_desc,
                device_info=self.device_info,
                conda_packages=self.conda_packages,
                conda_env_name=self.conda_env_name,
                experience_pool=self.experience_pool,
            )

            result = agent.generate(context)

            if not result.success or result.node is None:
                log_msg("WARNING", f"{agent.name} 生成失败: {result.error}")
                return None

            node = result.node

            # Phase 4: 执行代码（并行安全）
            exec_result = self._execute_code(node.code, node.id)
            node.term_out = "\n".join(exec_result.term_out)
            node.exec_time = exec_result.exec_time
            node.exc_type = exec_result.exc_type
            node.exc_info = str(exec_result.exc_info) if exec_result.exc_info else None

            # Phase 5: Review 评估
            self._review_node(node, parent_node=parent_node)

            # Phase 6: 保存节点
            with self.save_lock:
                self._save_node_solution(node)

            # Phase 7: 线程安全追加到 Journal
            with self.journal_lock:
                self.journal.append(node)
                self._update_best_node(node)

            # Phase 8: 写入经验池
            if self.experience_pool:
                self._write_experience_pool(agent.name, task_type, node)

            # Phase 9: 打印结果
            self._print_node_summary(node)

            log_msg(
                "INFO",
                f"{agent.name} 完成 {'explore' if parent_node is None else 'improve'}: is_buggy={node.is_buggy}, exec_time={node.exec_time:.2f}s",
            )

            return node

        except Exception as e:
            log_exception(e, "Orchestrator _step_task() 执行失败")
            return None

    def _check_time_limit(self) -> bool:
        """检查是否达到时间限制。

        Returns:
            是否已达时间限制
        """
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= self.config.agent.time_limit:
            log_msg("INFO", f"已达时间限制 {self.config.agent.time_limit}s，停止运行")
            return True
        return False

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
        with self.journal_lock:
            # Phase 1: 初稿模式
            if len(self.journal.draft_nodes) < self.config.search.num_drafts:
                log_msg("INFO", "[search_policy] 初稿模式")
                return None

            # Phase 2: 修复模式
            if random.random() < self.config.search.debug_prob:
                self.journal.build_dag()
                buggy_leaves = [
                    n for n in self.journal.buggy_nodes if not n.children_ids
                ]

                if buggy_leaves:
                    node = random.choice(buggy_leaves)
                    log_msg("INFO", f"[search_policy] 修复模式: 节点 {node.id[:8]}")
                    return node

            # Phase 3: 改进模式
            best = self.journal.get_best_node(only_good=True)
            if best:
                log_msg("INFO", f"[search_policy] 改进模式: 节点 {best.id[:8]}")
                return best
            else:
                log_msg("INFO", "[search_policy] 初稿模式（无可用节点）")
                return None

    def _prepare_step(self) -> None:
        """准备单步执行环境（线程安全）。

        Note: submission 目录不再清空，因为每个节点使用独立的 submission_{node_id}.csv
        """
        # submission 目录确保存在
        submission_dir = self.config.project.workspace_dir / "submission"
        submission_dir.mkdir(exist_ok=True)

    def _execute_code(self, code: str, node_id: str) -> ExecutionResult:
        """执行代码。

        Args:
            code: Python 代码字符串
            node_id: 节点 ID

        Returns:
            ExecutionResult 对象
        """
        # 使用 WorkspaceManager 重写 submission 路径
        modified_code = self.workspace.rewrite_submission_path(code, node_id)

        # 执行代码（并行安全）
        return self.interpreter.run(modified_code, node_id=node_id, reset_session=True)

    def _generate_code_diff(
        self,
        parent_code: Optional[str],
        current_code: str,
        context_lines: int = 3,
    ) -> str:
        """生成父子代码的 unified diff。

        Args:
            parent_code: 父节点代码（None 表示首次生成）
            current_code: 当前节点代码
            context_lines: diff 上下文行数（默认 3）

        Returns:
            unified diff 格式字符串，首次生成返回 "(Initial solution, no diff)"
        """
        if parent_code is None:
            return "(Initial solution, no diff)"

        parent_lines = parent_code.splitlines(keepends=True)
        current_lines = current_code.splitlines(keepends=True)

        diff = difflib.unified_diff(
            parent_lines,
            current_lines,
            fromfile="parent_solution.py",
            tofile="current_solution.py",
            n=context_lines,
        )

        diff_text = "".join(diff)

        # 如果 diff 过长，截断并提示
        max_diff_lines = 100
        diff_lines = diff_text.splitlines()
        if len(diff_lines) > max_diff_lines:
            diff_text = "\n".join(diff_lines[:max_diff_lines])
            diff_text += (
                f"\n... (truncated, {len(diff_lines) - max_diff_lines} more lines)"
            )

        return diff_text if diff_text.strip() else "(No changes detected)"

    def _review_node(
        self,
        node: Node,
        parent_node: Optional[Node] = None,
        gene_plan: Optional[Dict] = None,
    ) -> None:
        """Review 评估节点（多层验证 + 回退机制）。

        流程:
        0. 生成变更上下文（根据任务类型选择策略）
        1. 检查 submission 文件是否存在
        2. 调用 LLM Function Calling
        3. 失败时回退到无 Tool 方案
        4. 验证 LLM 响应
        5. 检测异常指标值
        6. 综合判断 is_buggy
        7. 强耦合：is_buggy=True 时 metric_value=None

        Args:
            node: 待评估的节点对象
            parent_node: 父节点（用于 explore/mutate 的代码 diff）
            gene_plan: 基因选择计划（用于 merge 的变更上下文）
        """
        # Phase 0: 生成变更上下文（根据任务类型选择策略）
        if gene_plan:
            # merge 模式：gene_plan 已是 Markdown 字符串，直接用作变更上下文
            change_context = gene_plan if isinstance(gene_plan, str) else str(gene_plan)
        elif parent_node:
            # explore/mutate 模式：代码 diff
            change_context = self._generate_code_diff(parent_node.code, node.code)
        else:
            # 初稿模式
            change_context = "(Initial solution, no diff)"

        # Phase 1: 文件存在检查
        has_submission = self._check_submission_exists(node.id)

        # Phase 2: LLM Function Calling（收集调试数据）
        review_data = None
        review_debug = {
            "method": None,
            "input": None,
            "output_raw": None,
            "output_parsed": None,
            "error": None,
        }

        # 构建 Review 输入
        review_input = self._build_review_messages(node, change_context, parent_node)
        review_debug["input"] = {
            "user_message": review_input,
            "model": self.config.llm.feedback.model,
            "provider": self.config.llm.feedback.provider,
        }

        try:
            review_debug["method"] = "function_calling"
            raw_response, parsed_data = self._call_review_with_tool_debug(
                node, change_context, review_input
            )
            review_debug["output_raw"] = raw_response
            review_debug["output_parsed"] = parsed_data
            review_data = parsed_data
        except Exception as e:
            log_msg("WARNING", f"Function Calling 失败: {e}，尝试回退方案")
            review_debug["error"] = str(e)

        # Phase 3: 回退到无 Tool 方案
        if review_data is None:
            try:
                review_debug["method"] = "fallback"
                raw_response, parsed_data = self._review_node_without_tool_debug(node)
                review_debug["output_raw"] = raw_response
                review_debug["output_parsed"] = parsed_data
                review_data = parsed_data
            except Exception as e:
                log_exception(e, "回退方案也失败")
                review_debug["error"] = str(e)
                review_data = {
                    "is_bug": True,
                    "metric": None,
                    "summary": f"Review 完全失败: {str(e)}",
                    "lower_is_better": False,
                    "has_csv_submission": False,
                }
                review_debug["output_parsed"] = review_data

        # Phase 4: 验证响应
        review_data = self._validate_review_response(review_data, node, has_submission)

        # Phase 5: 异常值检测
        metric_value = review_data.get("metric")
        is_metric_plausible = True
        if metric_value is not None and self.best_node is not None:
            is_metric_plausible = self._check_metric_plausibility(metric_value)
            if not is_metric_plausible:
                log_msg(
                    "WARNING",
                    f"节点 {node.id[:8]} 指标值异常 ({metric_value})，标记为 buggy",
                )

        # Phase 6: 综合判断 is_buggy（5 条件 OR）
        node.is_buggy = (
            review_data.get("is_bug", False)  # 条件 1: LLM 判断
            or node.exc_type is not None  # 条件 2: 执行异常
            or metric_value is None  # 条件 3: 指标缺失
            or not has_submission  # 条件 4: 文件不存在
            or not is_metric_plausible  # 条件 5: 指标异常
        )

        # Phase 7: 强耦合 - is_buggy=True 时 metric_value=None
        if node.is_buggy:
            node.metric_value = None
            log_msg("INFO", f"节点 {node.id[:8]} 标记为 BUGGY，metric_value 设为 None")
        else:
            node.metric_value = metric_value

        node.lower_is_better = review_data.get("lower_is_better", False)
        node.analysis = review_data.get("key_change", "")  # 兼容旧字段
        node.analysis_detail = {
            "key_change": review_data.get("key_change", ""),
            "insight": review_data.get("insight", ""),
            "bottleneck": review_data.get("bottleneck"),
            "suggested_direction": review_data.get("suggested_direction"),
        }

        # Phase 8: 存储 Review 调试数据（用于排查问题）
        node.metadata["review_debug"] = review_debug

        log_msg(
            "INFO",
            f"Review 完成: 节点 {node.id[:8]}, is_buggy={node.is_buggy}, "
            f"metric={node.metric_value}, lower_is_better={node.lower_is_better}",
        )

        # Phase 9: 计算节点信息素并更新基因注册表
        if self.gene_registry and not node.is_buggy and node.metric_value is not None:
            from core.evolution.pheromone import (
                compute_node_pheromone,
                ensure_node_stats,
            )

            scores = [
                n.metric_value
                for n in self.journal.nodes
                if not n.is_buggy and n.metric_value is not None
            ]
            if scores:
                ensure_node_stats(node)
                pheromone = compute_node_pheromone(
                    node,
                    current_step=len(self.journal.nodes),
                    score_min=min(scores),
                    score_max=max(scores),
                )
                node.metadata["pheromone_node"] = pheromone
                self.gene_registry.update_from_reviewed_node(node)

    def _call_review_with_tool(self, node: Node, change_context: str) -> Dict:
        """使用 Function Calling 调用 Review LLM。

        Args:
            node: 待评估的节点
            change_context: 变更上下文（diff 或 gene selection）

        Returns:
            LLM 返回的 review 数据 dict

        Raises:
            Exception: LLM 调用或解析失败
        """
        messages_content = self._build_review_messages(node, change_context)
        tool_schema = self._get_review_tool_schema()

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

        return json.loads(response)

    def _call_review_with_tool_debug(
        self, node: Node, change_context: str, review_input: str
    ) -> tuple:
        """Function Calling 调用 Review LLM（返回调试信息）。

        Args:
            node: 待评估的节点
            change_context: 变更上下文
            review_input: 已构建的输入消息

        Returns:
            (原始响应字符串, 解析后的 dict) 元组
        """
        tool_schema = self._get_review_tool_schema()

        response = backend_query(
            system_message=None,
            user_message=review_input,
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

        return response, json.loads(response)

    def _review_node_without_tool_debug(self, node: Node) -> tuple:
        """回退方案（返回调试信息）。

        Args:
            node: 待评估的节点

        Returns:
            (原始响应字符串, 解析后的 dict) 元组
        """
        from utils.response import extract_review

        prompt = self._build_review_prompt_without_tool(node)

        response_text = backend_query(
            system_message=None,
            user_message=prompt,
            model=self.config.llm.feedback.model,
            provider=self.config.llm.feedback.provider,
            temperature=self.config.llm.feedback.temperature,
            api_key=self.config.llm.feedback.api_key,
            base_url=getattr(self.config.llm.feedback, "base_url", None),
            tools=None,
            tool_choice=None,
        )

        return response_text, extract_review(response_text)

    def _review_node_without_tool(self, node: Node) -> Dict:
        """回退方案：无 Tool 的 LLM 调用 + JSON 提取。

        参考: ML-Master parse_exec_result_without_tool()

        Args:
            node: 待评估的节点

        Returns:
            解析后的 review 数据 dict
        """
        from utils.response import extract_review

        prompt = self._build_review_prompt_without_tool(node)

        response_text = backend_query(
            system_message=None,
            user_message=prompt,
            model=self.config.llm.feedback.model,
            provider=self.config.llm.feedback.provider,
            temperature=self.config.llm.feedback.temperature,
            api_key=self.config.llm.feedback.api_key,
            base_url=getattr(self.config.llm.feedback, "base_url", None),
            tools=None,  # 无 Tool
            tool_choice=None,
        )

        return extract_review(response_text)

    def _build_review_prompt_without_tool(self, node: Node) -> str:
        """构建回退方案的 Prompt（压缩版，要求 LLM 输出 JSON）。

        Args:
            node: 节点对象

        Returns:
            Prompt 字符串
        """
        return f"""You are evaluating a ML solution.

**Task Summary:**
{self._task_desc_compressed}

**Execution Output:**
```
{node.term_out}
```

**Status**: Time={node.exec_time:.2f}s, Exception={node.exc_type or "None"}

---

Respond with JSON:
```json
{{
    "is_bug": <bool>,
    "has_csv_submission": <bool>,
    "metric": <number|null>,
    "lower_is_better": <bool>
}}
```
"""

    def _validate_review_response(
        self, response: Dict, node: Node, has_submission: bool
    ) -> Dict:
        """验证 LLM 返回的 review 数据。

        验证规则:
        1. metric 必须是 float/int/None
        2. is_bug=True 时，metric 应为 None
        3. has_csv_submission 与实际文件存在性交叉验证

        Args:
            response: LLM 返回的原始 response
            node: 节点对象
            has_submission: 实际文件是否存在

        Returns:
            验证/修正后的 response dict
        """
        validated = dict(response)

        # 规则 1: 类型检查
        metric = validated.get("metric")
        if metric is not None and not isinstance(metric, (float, int)):
            log_msg("WARNING", f"metric 类型无效: {type(metric)}，设为 None")
            validated["metric"] = None
        elif isinstance(metric, (float, int)):
            validated["metric"] = float(metric)

        # 规则 2: is_bug 与 metric 一致性
        if validated.get("is_bug", False) and validated.get("metric") is not None:
            log_msg("WARNING", "is_bug=True 但 metric 非空，强制设为 None")
            validated["metric"] = None

        # 规则 3: 文件存在交叉验证
        llm_says_has_csv = validated.get("has_csv_submission", False)
        if llm_says_has_csv and not has_submission:
            log_msg("WARNING", "LLM 声称有 submission 但文件不存在，覆盖为 False")
            validated["has_csv_submission"] = False

        return validated

    def _check_submission_exists(self, node_id: str) -> bool:
        """检查 submission 文件是否存在。

        Args:
            node_id: 节点 ID

        Returns:
            文件是否存在
        """
        submission_path = (
            self.config.project.workspace_dir
            / "submission"
            / f"submission_{node_id}.csv"
        )
        exists = submission_path.exists()
        if not exists:
            log_msg("DEBUG", f"submission_{node_id}.csv 不存在")
        return exists

    def _check_metric_plausibility(self, metric: float) -> bool:
        """检测指标是否在合理范围内（防止 LLM 幻觉）。

        参考: ML-Master check_metric_valid()

        规则: 如果新指标与 best_node 指标相差超过 upper_bound 倍，视为异常。

        Args:
            metric: 待检测的指标值

        Returns:
            True 如果在合理范围内，False 如果异常
        """
        if self.best_node is None or self.best_node.metric_value is None:
            return True  # 无参考值，默认通过

        best_value = self.best_node.metric_value

        # 获取阈值（默认 50，支持配置覆盖）
        upper_bound = getattr(self.config.search, "invalid_metric_upper_bound", 50)

        # 避免除零
        if best_value == 0 or metric == 0:
            return abs(best_value - metric) <= upper_bound

        # 相对比率检查
        ratio = max(abs(best_value), abs(metric)) / min(abs(best_value), abs(metric))
        return ratio <= upper_bound

    def _build_review_messages(
        self,
        node: Node,
        change_context: str,
        parent_node: Optional[Node] = None,
    ) -> str:
        """构建 Review 消息（压缩版，减少 token 消耗）。

        优化点:
        1. 使用压缩后的 task_desc（~500B vs ~7KB）
        2. 有 Diff 时省略完整代码（节省 4-8KB）
        3. 包含父节点基准信息（metric + 异常状态）

        Args:
            node: 节点对象
            change_context: 变更上下文（diff 或 gene selection）
            parent_node: 父节点（用于基准对比）

        Returns:
            消息内容字符串
        """
        best_metric = self.best_node.metric_value if self.best_node else None
        best_metric_str = f"{best_metric:.4f}" if best_metric else "N/A"

        # 父节点基准信息
        if parent_node:
            p_metric = (
                f"{parent_node.metric_value:.4f}" if parent_node.metric_value else "N/A"
            )
            p_exc = parent_node.exc_type or "None"
            baseline_str = f"**Baseline (Parent)**: metric={p_metric}, exc={p_exc}"
        else:
            baseline_str = "**Baseline (Parent)**: N/A (Initial)"

        # 判断是否为初稿（无 Diff）
        is_initial = change_context == "(Initial solution, no diff)"

        # 核心模板
        prompt = f"""You are evaluating a ML solution.

**Task Summary:**
{self._task_desc_compressed}

{baseline_str}
**Current Best**: {best_metric_str}

---

## Code Changes

```diff
{change_context}
```
"""

        # 仅初稿时包含完整代码（因为无 Diff 可参考）
        if is_initial:
            prompt += f"""
## Solution Code

```python
{node.code}
```
"""

        # 执行输出（必须保留）
        prompt += f"""
## Execution Output

```
{node.term_out}
```

**Status**: Time={node.exec_time:.2f}s, Exception={node.exc_type or "None"}

---

Call `submit_review` with your analysis.
"""
        return prompt

    def _get_review_tool_schema(self) -> Dict:
        """获取 Review Function Calling 的 schema（增强版）。

        Returns:
            tool schema 字典
        """
        return {
            "name": "submit_review",
            "description": "提交代码评估结果（包含详细分析）",
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
                    "metric": {
                        "type": "number",
                        "description": "验证集指标值（如 RMSE），失败时为 null",
                        "nullable": True,
                    },
                    "lower_is_better": {
                        "type": "boolean",
                        "description": "指标是否越小越好（如 RMSE=true）",
                    },
                    # ===== 新增字段 =====
                    "key_change": {
                        "type": "string",
                        "description": "本次方案的核心改动点（基于 diff 总结，1-2 句话）",
                    },
                    "insight": {
                        "type": "string",
                        "description": "从本次实验得到的洞察（什么有效/无效，为什么）",
                    },
                    "bottleneck": {
                        "type": "string",
                        "description": "当前方案的主要瓶颈或限制",
                        "nullable": True,
                    },
                    "suggested_direction": {
                        "type": "string",
                        "description": "建议的下一步优化方向",
                        "nullable": True,
                    },
                },
                "required": [
                    "is_bug",
                    "has_csv_submission",
                    "metric",
                    "lower_is_better",
                    "key_change",
                    "insight",
                ],
            },
        }

    def _update_best_node(self, node: Node) -> None:
        """更新最佳节点（线程安全，需在 journal_lock 内调用）。

        修改点：
        1. 跳过 metric_value=None 的节点
        2. 跳过 buggy 节点
        3. 使用 _is_better() 比较

        Args:
            node: 候选节点对象
        """
        # 跳过无效指标
        if node.metric_value is None:
            return

        # 跳过 buggy 节点（双重保险，因为 is_buggy=True 时 metric_value 应为 None）
        if node.is_buggy:
            return

        if self.best_node is None:
            log_msg(
                "INFO",
                f"初始化最佳节点: {node.id[:8]}, metric={node.metric_value}",
            )
            self.best_node = node
            self._save_best_solution(node)
            return

        # best_node 也应该是 good 节点
        if self.best_node.is_buggy or self.best_node.metric_value is None:
            log_msg("INFO", f"替换无效的 best_node: {node.id[:8]}")
            self.best_node = node
            self._save_best_solution(node)
            return

        # 正常比较
        if self._is_better(node, self.best_node):
            direction = "↓" if node.lower_is_better else "↑"
            log_msg(
                "INFO",
                f"新的最佳节点: {node.id[:8]}, metric={node.metric_value} {direction}",
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

            with open(best_dir / "solution.py", "w", encoding="utf-8") as f:
                f.write(node.code)

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
            node_dir = (
                self.config.project.workspace_dir
                / "working"
                / f"solution_{node.id[:8]}"
            )
            node_dir.mkdir(exist_ok=True, parents=True)

            with open(node_dir / "solution.py", "w", encoding="utf-8") as f:
                f.write(node.code)

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

            submission_src = (
                self.config.project.workspace_dir
                / "submission"
                / f"submission_{node.id}.csv"
            )
            if submission_src.exists():
                shutil.copy(submission_src, node_dir / "submission.csv")

            # 保存 prompt.json（LLM 请求数据）
            if node.prompt_data:
                with open(node_dir / "prompt.json", "w", encoding="utf-8") as f:
                    json.dump(node.prompt_data, f, ensure_ascii=False, indent=2)

            # 保存 plan.txt（方案说明）
            if node.plan:
                with open(node_dir / "plan.txt", "w", encoding="utf-8") as f:
                    f.write(node.plan)

            # 保存 review.json（Review 调试数据）
            if node.metadata.get("review_debug"):
                with open(node_dir / "review.json", "w", encoding="utf-8") as f:
                    json.dump(
                        node.metadata["review_debug"], f, ensure_ascii=False, indent=2
                    )

            log_msg("INFO", f"节点 {node.id[:8]} 已保存到 {node_dir}")

        except Exception as e:
            log_exception(e, f"保存节点 {node.id[:8]} 失败")

    def _print_node_summary(self, node: Node) -> None:
        """打印节点评估摘要。

        Args:
            node: 节点对象
        """
        status = "❌ BUGGY" if node.is_buggy else "✅ SUCCESS"
        metric_str = f"{node.metric_value}" if node.metric_value is not None else "N/A"
        direction = "↓ (越小越好)" if node.lower_is_better else "↑ (越大越好)"

        summary = (
            f"{status} | 节点 {node.id[:8]} | "
            f"指标: {metric_str} {direction} | "
            f"执行: {node.exec_time:.2f}s"
        )

        log_msg("INFO", f"[评估] {summary}")

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

    def _write_experience_pool(self, agent_id: str, task_type: str, node: Node) -> None:
        """写入经验池（Phase 3）。

        Args:
            agent_id: Agent ID
            task_type: 任务类型
            node: 生成的节点
        """
        try:
            import hashlib
            import time
            from core.evolution.experience_pool import TaskRecord

            # 计算输入哈希
            input_hash = hashlib.sha256(f"{task_type}_{node.id}".encode()).hexdigest()[
                :16
            ]

            # 计算输出质量（基于 metric_value）
            output_quality = float(node.metric_value or 0.0)

            # 提取策略摘要（从 plan）
            strategy_summary = node.plan[:200] if node.plan else "No plan"

            record = TaskRecord(
                agent_id=agent_id,
                task_type=task_type,
                input_hash=input_hash,
                output_quality=output_quality,
                strategy_summary=strategy_summary,
                timestamp=time.time(),
            )

            self.experience_pool.add(record)
            log_msg("DEBUG", f"经验池记录已添加: agent={agent_id}, task={task_type}")

        except Exception as e:
            log_msg("WARNING", f"写入经验池失败: {e}")

    def execute_merge_task(
        self, parent_a: Node, parent_b: Node, gene_plan: Dict
    ) -> Optional[Node]:
        """执行 merge 任务（基因交叉）。

        Args:
            parent_a: 父代 A
            parent_b: 父代 B
            gene_plan: 基因交叉计划

        Returns:
            合成的子代节点（失败时返回 None）
        """
        try:
            # 选择 Agent
            if self.task_dispatcher:
                agent = self.task_dispatcher.select_agent("merge")
            else:
                agent = random.choice(self.agents)

            log_msg(
                "INFO",
                f"{agent.name} 开始 merge (parent_a={parent_a.id[:8]}, parent_b={parent_b.id[:8]})",
            )

            # 构建上下文
            with self.journal_lock:
                current_step = len(self.journal.nodes)

            context = AgentContext(
                task_type="merge",
                parent_node=None,  # merge 不使用 parent_node
                journal=self.journal,
                config=self.config,
                start_time=self.start_time,
                current_step=current_step,
                task_desc=self.task_desc,
                device_info=self.device_info,
                conda_packages=self.conda_packages,
                conda_env_name=self.conda_env_name,
                parent_a=parent_a,
                parent_b=parent_b,
                gene_plan=gene_plan,
                experience_pool=self.experience_pool,
            )

            # 生成代码
            result = agent.generate(context)

            if not result.success or result.node is None:
                log_msg("WARNING", f"{agent.name} merge 失败: {result.error}")
                return None

            node = result.node

            # 执行代码
            exec_result = self._execute_code(node.code, node.id)
            node.term_out = "\n".join(exec_result.term_out)
            node.exec_time = exec_result.exec_time
            node.exc_type = exec_result.exc_type
            node.exc_info = str(exec_result.exc_info) if exec_result.exc_info else None

            # Review 评估（merge 使用基因选择方案而非代码 diff）
            self._review_node(node, gene_plan=gene_plan)

            # 保存节点
            with self.save_lock:
                self._save_node_solution(node)

            # 追加到 Journal
            with self.journal_lock:
                self.journal.append(node)
                self._update_best_node(node)

            # 写入经验池
            if self.experience_pool:
                self._write_experience_pool(agent.name, "merge", node)

            log_msg(
                "INFO",
                f"{agent.name} merge 完成: is_buggy={node.is_buggy}, exec_time={node.exec_time:.2f}s",
            )

            return node

        except Exception as e:
            log_exception(e, "execute_merge_task() 失败")
            return None

    def execute_mutate_task(self, parent: Node, target_gene: str) -> Optional[Node]:
        """执行 mutate 任务（基因变异）。

        Args:
            parent: 父代节点
            target_gene: 目标基因块名称

        Returns:
            变异后的节点（失败时返回 None）
        """
        try:
            # 选择 Agent
            if self.task_dispatcher:
                agent = self.task_dispatcher.select_agent("mutate")
            else:
                agent = random.choice(self.agents)

            log_msg(
                "INFO",
                f"{agent.name} 开始 mutate (parent={parent.id[:8]}, gene={target_gene})",
            )

            # 构建上下文
            with self.journal_lock:
                current_step = len(self.journal.nodes)

            context = AgentContext(
                task_type="mutate",
                parent_node=parent,
                journal=self.journal,
                config=self.config,
                start_time=self.start_time,
                current_step=current_step,
                task_desc=self.task_desc,
                device_info=self.device_info,
                conda_packages=self.conda_packages,
                conda_env_name=self.conda_env_name,
                target_gene=target_gene,
                experience_pool=self.experience_pool,
            )

            # 生成代码
            result = agent.generate(context)

            if not result.success or result.node is None:
                log_msg("WARNING", f"{agent.name} mutate 失败: {result.error}")
                return None

            node = result.node

            # 执行代码
            exec_result = self._execute_code(node.code, node.id)
            node.term_out = "\n".join(exec_result.term_out)
            node.exec_time = exec_result.exec_time
            node.exc_type = exec_result.exc_type
            node.exc_info = str(exec_result.exc_info) if exec_result.exc_info else None

            # Review 评估（mutate 使用代码 diff）
            self._review_node(node, parent_node=parent)

            # 保存节点
            with self.save_lock:
                self._save_node_solution(node)

            # 追加到 Journal
            with self.journal_lock:
                self.journal.append(node)
                self._update_best_node(node)

            # 写入经验池
            if self.experience_pool:
                self._write_experience_pool(agent.name, "mutate", node)

            log_msg(
                "INFO",
                f"{agent.name} mutate 完成: is_buggy={node.is_buggy}, exec_time={node.exec_time:.2f}s",
            )

            return node

        except Exception as e:
            log_exception(e, "execute_mutate_task() 失败")
            return None


# 兼容旧接口的工厂函数
def create_orchestrator(
    agent: BaseAgent,
    config: Config,
    journal: Journal,
    task_desc: str,
    agent_evolution: Optional["AgentEvolution"] = None,
) -> Orchestrator:
    """兼容旧接口的工厂函数（单 Agent）。

    Args:
        agent: 单个 Agent
        config: 配置
        journal: Journal
        task_desc: 任务描述
        agent_evolution: Agent 进化器

    Returns:
        Orchestrator 实例
    """
    return Orchestrator(
        agents=[agent],
        config=config,
        journal=journal,
        task_desc=task_desc,
        agent_evolution=agent_evolution,
    )
