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
from pathlib import Path
from typing import Optional, Dict, List, TYPE_CHECKING

from agents.base_agent import BaseAgent, AgentContext
from core.state import Node, Journal
from core.executor.interpreter import Interpreter, ExecutionResult
from core.executor.workspace import WorkspaceManager
from core.backend import query as backend_query
from utils.config import Config
import re

from core.evolution.gene_parser import parse_solution_genes
from utils.logger_system import log_msg, log_json, log_exception
from utils.system_info import (
    get_hardware_description,
    get_conda_packages,
    get_conda_python_path,
)

if TYPE_CHECKING:
    from core.evolution.agent_evolution import AgentEvolution

# Metric 合理性范围（用于防止 LLM Review 幻觉）
# 格式: { metric_keyword: (min_val, max_val) }
# min_val/max_val 为 None 表示不限制该方向
METRIC_BOUNDS = {
    # Bounded [0, 1]
    "auc": (0.0, 1.0),
    "accuracy": (0.0, 1.0),
    "f1": (0.0, 1.0),
    "precision": (0.0, 1.0),
    "recall": (0.0, 1.0),
    "map": (0.0, 1.0),
    "qwk": (-1.0, 1.0),
    "kappa": (-1.0, 1.0),
    # Unbounded non-negative
    "rmse": (0.0, None),
    "rmsle": (0.0, None),
    "mae": (0.0, None),
    "mse": (0.0, None),
    # Log loss: 理论 (0, +inf)，实际合理 (epsilon, 15)
    "logloss": (1e-7, 15.0),
    "log_loss": (1e-7, 15.0),
}

# Metric 方向映射表（确定性来源，修复 P0-1 Bug）
# True = lower_is_better（越小越好: loss/error 类）
# False = higher_is_better（越大越好: score/accuracy 类）
METRIC_DIRECTION: Dict[str, bool] = {
    # === Lower is better ===
    "rmse": True,
    "root mean squared error": True,
    "rmsle": True,
    "mae": True,
    "mean absolute error": True,
    "mse": True,
    "mean squared error": True,
    "logloss": True,
    "log_loss": True,
    "log loss": True,
    "logarithmic loss": True,
    "cross-entropy": True,
    "cross entropy": True,
    "mcrmse": True,
    "medae": True,
    "mape": True,
    "smape": True,
    "pinball loss": True,
    "hinge loss": True,
    # === Higher is better ===
    "mean column-wise roc auc": False,
    "column-wise roc auc": False,
    "area under the roc curve": False,
    "area under the receiver operating characteristic": False,
    "mean auc": False,
    "roc auc": False,
    "roc_auc": False,
    "auc": False,
    "accuracy": False,
    "categorization accuracy": False,
    "f1": False,
    "f1-score": False,
    "f1 score": False,
    "precision": False,
    "recall": False,
    "sensitivity": False,
    "specificity": False,
    "map": False,
    "mean average precision": False,
    "mean column-wise auc": False,
    "qwk": False,
    "quadratic weighted kappa": False,
    "kappa": False,
    "cohen's kappa": False,
    "ndcg": False,
    "r2": False,
    "r-squared": False,
    "r²": False,
    "spearman": False,
    "pearson": False,
    "correlation": False,
    "iou": False,
    "dice": False,
    "bleu": False,
    "rouge": False,
    "mean column-wise log loss": True,
    "multiclass loss": True,
}


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

        # P0-1 修复：全局 metric 方向（启动时从 task_desc 检测，首次 review 时锁定）
        self._global_lower_is_better: Optional[bool] = None
        self._detect_metric_direction()

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

        # 计算自适应超时
        adaptive_timeout = self._estimate_timeout()

        # 初始化代码执行器（使用 conda Python + 自适应超时）
        self.interpreter = Interpreter(
            working_dir=str(config.project.workspace_dir),
            timeout=adaptive_timeout,
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

        # Phase 1: 文件存在 + 格式校验
        has_submission = self._check_submission_exists(node.id)
        if has_submission:
            submission_check = self._validate_submission_format(node.id)
            if not submission_check["valid"]:
                log_msg(
                    "WARNING",
                    f"Submission 格式异常: {submission_check['errors']}",
                )
                # [NaN回传] 仅当代码执行成功时才回传格式错误（执行失败的节点不应收到 NaN 提示）
                if node.exc_type is None:
                    error_details = "; ".join(submission_check["errors"])
                    node.term_out = (node.term_out or "") + (
                        f"\n\n[SUBMISSION VALIDATION FAILED]: {error_details}\n"
                        f"Your code ran without errors but produced an invalid submission.\n"
                        f"Fix the root cause in your code (do NOT use fillna as a patch)."
                    )
                has_submission = False  # 格式无效等同于不存在

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

        # Phase 5.5: stdout metric 数据收集（LLM 优先，仅记录差异）
        stdout_metric = self._parse_metric_from_stdout(node.term_out)
        if stdout_metric is not None and metric_value is not None:
            diff = abs(stdout_metric - metric_value)
            if diff > max(abs(stdout_metric) * 0.01, 1e-6):
                log_json(
                    {
                        "event": "metric_mismatch",
                        "node_id": node.id,
                        "llm_metric": metric_value,
                        "stdout_metric": stdout_metric,
                        "diff": diff,
                    }
                )
                log_msg(
                    "WARNING",
                    f"Metric 不一致: LLM={metric_value}, stdout={stdout_metric}（保留 LLM 值）",
                )
        elif stdout_metric is not None and metric_value is None:
            # LLM 未提取到但 stdout 有值：用 stdout 补位
            log_msg("INFO", f"LLM 未提取 metric，使用 stdout 值: {stdout_metric}")
            metric_value = stdout_metric
            review_data["metric"] = stdout_metric

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

        # P0-1 修复：尝试从本次 review 锁定全局方向（仅首次生效，幂等操作）
        if not node.is_buggy:
            self._lock_metric_direction(review_data)

        # 节点级 lower_is_better 始终使用全局值（保持序列化兼容性）
        node.lower_is_better = (
            self._global_lower_is_better
            if self._global_lower_is_better is not None
            else review_data.get("lower_is_better", False)
        )

        node.analysis = review_data.get("key_change", "")  # 兼容旧字段
        node.analysis_detail = {
            "key_change": review_data.get("key_change", ""),
            "insight": review_data.get("insight", ""),
            "bottleneck": review_data.get("bottleneck"),
            "suggested_direction": review_data.get("suggested_direction"),
        }

        # Phase 7.5: 提取 approach_tag（仅非 buggy 节点）
        if not node.is_buggy:
            approach_tag = review_data.get("approach_tag")
            if approach_tag:
                node.approach_tag = approach_tag
                log_msg("DEBUG", f"节点 {node.id[:8]} approach_tag: {approach_tag}")

        # Phase 8: 存储 Review 调试数据（用于排查问题）
        node.metadata["review_debug"] = review_debug

        log_msg(
            "INFO",
            f"Review 完成: 节点 {node.id[:8]}, is_buggy={node.is_buggy}, "
            f"metric={node.metric_value}, lower_is_better={node.lower_is_better}",
        )

        # Phase 8.5: 解析节点基因（信息素机制的前提）
        if not node.is_buggy and node.code:
            node.genes = parse_solution_genes(node.code)

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

        三层校验:
        1. 绝对范围检查（基于 METRIC_BOUNDS，匹配 task_desc 中的关键词）
        2. logloss=0 特殊检查（logloss 值严格大于 0）
        3. 相对比率检查（与 best_node 比较，阈值默认 50 倍）

        Args:
            metric: 待检测的指标值

        Returns:
            True 如果在合理范围内，False 如果异常
        """
        # Phase 1: 绝对范围检查
        task_lower = (self._task_desc_compressed or "").lower()
        for keyword, (min_val, max_val) in METRIC_BOUNDS.items():
            if keyword in task_lower:
                if min_val is not None and metric < min_val:
                    log_msg(
                        "WARNING",
                        f"Metric {metric} 低于 {keyword} 下界 {min_val}",
                    )
                    return False
                if max_val is not None and metric > max_val:
                    log_msg(
                        "WARNING",
                        f"Metric {metric} 超过 {keyword} 上界 {max_val}",
                    )
                    return False
                break  # 只匹配第一个关键词

        # Phase 2: metric=0.0 特殊检查
        if metric == 0.0 and self.best_node and self.best_node.metric_value:
            if self.best_node.metric_value > 0.01:
                log_msg(
                    "WARNING",
                    f"Metric=0.0 疑似虚假值（best={self.best_node.metric_value}）",
                )
                return False

        # Phase 3: 相对比率检查（原有逻辑）
        if self.best_node is None or self.best_node.metric_value is None:
            return True

        best_value = self.best_node.metric_value
        upper_bound = getattr(self.config.search, "invalid_metric_upper_bound", 50)

        if best_value == 0 or metric == 0:
            return abs(best_value - metric) <= upper_bound

        ratio = max(abs(best_value), abs(metric)) / min(abs(best_value), abs(metric))
        return ratio <= upper_bound

    def _detect_metric_direction(self) -> Optional[bool]:
        """从 task_desc 中检测 metric 方向（启动时调用一次）。

        策略: 在 task_desc 中搜索 METRIC_DIRECTION 所有 key，
        按 key 长度降序匹配（优先匹配更具体的名称如 "log loss" 而非 "loss"）。

        Returns:
            检测到的方向（True=lower_is_better, False=higher_is_better），未检测到返回 None
        """
        text = (self.task_desc or "").lower()
        sorted_keys = sorted(METRIC_DIRECTION.keys(), key=len, reverse=True)

        for key in sorted_keys:
            # P0-A 修复：使用单词边界匹配，避免 "mse" 匹配 "themselves" 等子串
            if re.search(r'\b' + re.escape(key) + r'\b', text):
                direction = METRIC_DIRECTION[key]
                self._global_lower_is_better = direction
                log_msg(
                    "INFO",
                    f"[metric_direction] 从 task_desc 检测到: '{key}' → lower_is_better={direction}",
                )
                return direction

        log_msg("WARNING", "[metric_direction] task_desc 中未匹配到已知 metric")
        return None

    def _lock_metric_direction(self, review_data: Dict) -> None:
        """从 review 结果中尝试锁定 metric 方向（仅在未锁定时生效）。

        借鉴 ML-Master 的一致性保证：方向一旦锁定，终生不可变（幂等操作）。

        Args:
            review_data: LLM Review 返回的数据 dict
        """
        if self._global_lower_is_better is not None:
            return  # 已锁定，幂等返回

        # 策略 1: metric_name → 查表（确定性来源）
        metric_name = (review_data.get("metric_name") or "").lower().strip()
        if metric_name:
            sorted_keys = sorted(METRIC_DIRECTION.keys(), key=len, reverse=True)
            for key in sorted_keys:
                # P0-A 修复：使用单词边界匹配，防止同类子串假匹配
                if re.search(r'\b' + re.escape(key) + r'\b', metric_name):
                    self._global_lower_is_better = METRIC_DIRECTION[key]
                    log_msg(
                        "INFO",
                        f"[metric_direction] 从 review metric_name 锁定: "
                        f"'{metric_name}' → lower_is_better={self._global_lower_is_better}",
                    )
                    return

        # 策略 2: 直接使用 review 的 lower_is_better（fallback，与 ML-Master 同级）
        lib = review_data.get("lower_is_better")
        if isinstance(lib, bool):
            self._global_lower_is_better = lib
            log_msg(
                "WARNING",
                f"[metric_direction] 使用 LLM 首次 review 的 lower_is_better={lib} "
                f"（未能从 metric_name 查表确认，fallback 到 LLM 判断）",
            )

    def _validate_submission_format(self, node_id: str) -> Dict:
        """校验 submission.csv 的基本格式。

        检查项:
        1. 文件是否可读
        2. 是否有 NaN 值
        3. 行数是否与 sample_submission 一致

        Args:
            node_id: 节点 ID

        Returns:
            {"valid": bool, "errors": list[str], "row_count": int}
        """
        result: Dict = {"valid": True, "errors": [], "row_count": 0}

        submission_path = (
            self.config.project.workspace_dir
            / "submission"
            / f"submission_{node_id}.csv"
        )
        if not submission_path.exists():
            result["valid"] = False
            result["errors"].append("submission.csv 不存在")
            return result

        try:
            import pandas as pd

            sub_df = pd.read_csv(submission_path)
            result["row_count"] = len(sub_df)

            # P0-2 修复：使用 glob 模式匹配 sample_submission 文件
            input_dir = self.config.project.workspace_dir / "input"
            candidates = (
                list(input_dir.glob("sample_submission*.csv"))
                + list(input_dir.glob("sampleSubmission*.csv"))
                + list(input_dir.glob("sample_Submission*.csv"))
            )
            sample_path = candidates[0] if candidates else None

            if sample_path is not None and sample_path.exists():
                sample_df = pd.read_csv(sample_path)
                # 行数检查
                if len(sub_df) != len(sample_df):
                    result["valid"] = False
                    result["errors"].append(
                        f"行数不匹配: submission={len(sub_df)}, sample={len(sample_df)}"
                    )
                # P0-B 修复：集合比较（不依赖顺序），列存在但顺序不同时自动重排写回
                sub_cols = set(sub_df.columns)
                sample_cols = set(sample_df.columns)
                if sub_cols != sample_cols:
                    result["valid"] = False
                    result["errors"].append(
                        f"列名不匹配: submission={sorted(sub_cols)}, "
                        f"sample={sorted(sample_cols)}"
                    )
                elif list(sub_df.columns) != list(sample_df.columns):
                    sub_df = sub_df[list(sample_df.columns)]
                    sub_df.to_csv(submission_path, index=False)
                    log_msg(
                        "INFO",
                        f"[P0-B] 列顺序已自动调整为 sample 顺序并写回: {node_id}",
                    )
                # P0-B 修复：NaN 检查仅针对目标列（非 id 列），避免特征列为空误判
                id_like = {"id", "image_id", "row_id", "uuid", "pid"}
                target_cols = [
                    c for c in sample_df.columns if c.lower() not in id_like
                ]
                if target_cols:
                    target_col = target_cols[0]
                    nan_count = int(sub_df[target_col].isna().sum())
                    if nan_count > 0:
                        result["valid"] = False
                        result["errors"].append(
                            f"目标列 '{target_col}' 包含 {nan_count} 个 NaN 值"
                        )
            else:
                # 无 sample 文件时，全表 NaN 检查（降级保留原逻辑）
                nan_count = int(sub_df.isnull().sum().sum())
                if nan_count > 0:
                    result["valid"] = False
                    result["errors"].append(f"submission 包含 {nan_count} 个 NaN 值")
        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"读取 submission 失败: {e}")

        return result

    def _parse_metric_from_stdout(self, term_out: str) -> Optional[float]:
        """从终端输出中正则提取 Validation metric 值。

        匹配格式: "Validation metric: {number}"
        如果有多行匹配（如多折输出），取最后一个（通常是平均值）。

        Args:
            term_out: 终端输出文本

        Returns:
            提取的 metric 值，未匹配返回 None
        """
        if not term_out:
            return None
        matches = re.findall(
            r"Validation metric:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            term_out,
        )
        if matches:
            try:
                return float(matches[-1])
            except ValueError:
                return None
        return None

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

**Metric Alignment Check**: Verify that the validation metric printed in output matches the competition's evaluation metric. If the code uses a different loss function for training (e.g., Focal Loss) but reports that loss as the validation metric instead of the actual competition metric (e.g., log_loss), set `is_bug=true` and explain in `key_change`.

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
                    "metric_name": {
                        "type": "string",
                        "description": (
                            "验证集使用的评估指标名称（如 log_loss, auc, rmse）。"
                            "必须与竞赛要求一致。如果代码使用了不同的指标"
                            "（如用 Focal Loss 代替 log_loss），请说明。"
                        ),
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
                    "approach_tag": {
                        "type": "string",
                        "description": (
                            "本方案的核心方法摘要（1 句话，如 'LightGBM + 5-fold CV + log1p feature'），"
                            "供后续 Draft 多样性引导使用。非 buggy 节点必填。"
                        ),
                        "nullable": True,
                    },
                },
                "required": [
                    "is_bug",
                    "has_csv_submission",
                    "metric",
                    "lower_is_better",
                    "metric_name",
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
            # P0-1 修复：使用全局方向
            lower = (
                self._global_lower_is_better
                if self._global_lower_is_better is not None
                else node.lower_is_better
            )
            direction = "↓" if lower else "↑"
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
                # [持久化] 同时同步到容器标准提交路径（防止进程被杀导致文件丢失）
                SUBMISSION_OUTPUT = Path("/home/submission/submission.csv")
                try:
                    SUBMISSION_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(submission_src, SUBMISSION_OUTPUT)
                    log_msg("INFO", f"[持久化] 最佳提交已同步到 {SUBMISSION_OUTPUT}")
                except Exception as sync_err:
                    log_msg("WARNING", f"[持久化] 同步到 /home/submission/ 失败: {sync_err}")

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

    def _estimate_timeout(self) -> int:
        """根据数据集大小估算合理超时时间（自适应超时）。

        Returns:
            超时时间（秒）
        """
        if not self.config.execution.adaptive_timeout:
            log_msg("INFO", f"自适应超时已禁用，使用固定超时: {self.config.execution.timeout}s")
            return self.config.execution.timeout

        base_timeout = self.config.execution.timeout  # 3600s
        max_timeout = self.config.execution.timeout_max  # 7200s

        # 计算数据集总大小
        input_dir = self.config.project.workspace_dir / "input"
        if not input_dir.exists():
            log_msg("WARNING", "input 目录不存在，使用基础超时")
            return base_timeout

        try:
            total_size_bytes = sum(
                f.stat().st_size for f in input_dir.rglob("*") if f.is_file()
            )
            total_size_mb = total_size_bytes / (1024 * 1024)
        except Exception as e:
            log_msg("WARNING", f"计算数据集大小失败: {e}，使用基础超时")
            return base_timeout

        # 根据数据集大小确定倍数
        multiplier = 1.0
        if total_size_mb > 500:  # 大数据集 (>500MB)
            multiplier = 2.0
        elif total_size_mb > 100:  # 中等数据集 (>100MB)
            multiplier = 1.5

        estimated_timeout = int(base_timeout * multiplier)
        final_timeout = min(estimated_timeout, max_timeout)

        log_msg(
            "INFO",
            f"自适应超时: dataset={total_size_mb:.1f}MB, multiplier={multiplier:.1f}x, "
            f"timeout={final_timeout}s (base={base_timeout}s, max={max_timeout}s)",
        )
        return final_timeout

    def _is_better(self, node: Node, best_node: Node) -> bool:
        """判断节点是否优于最佳节点（P0-1 修复：使用全局方向）。

        Args:
            node: 候选节点
            best_node: 当前最佳节点

        Returns:
            是否更好
        """
        if node.metric_value is None or best_node.metric_value is None:
            return False

        # 优先使用全局方向，fallback 到节点级方向
        lower = (
            self._global_lower_is_better
            if self._global_lower_is_better is not None
            else node.lower_is_better
        )

        if lower:
            return node.metric_value < best_node.metric_value
        else:
            return node.metric_value > best_node.metric_value

    def _debug_chain(
        self,
        node: Node,
        agent: BaseAgent,
        context: AgentContext,
        max_attempts: Optional[int] = None,
    ) -> Node:
        """链式 Debug：最多 max_attempts 次迭代修复，耗尽后标记 node.dead=True。

        与废弃的 _try_immediate_debug() 的区别：
        - 支持多次迭代（每次将上一次输出作为下一次输入）
        - 全部失败后设置 node.dead = True
        - debug_attempts 计入 node.debug_attempts

        Args:
            node: 执行后的节点
            agent: 执行该节点的 Agent
            context: Agent 执行上下文
            max_attempts: 最大 debug 次数（默认使用 config.evolution.solution.debug_max_attempts）

        Returns:
            修复成功的节点；或 dead=True 的原节点（所有尝试失败）
        """
        if max_attempts is None:
            max_attempts = getattr(
                self.config.evolution.solution, "debug_max_attempts", 2
            )

        # 不可恢复的错误类型跳过 debug（TimeoutExpired 为旧版兼容别名）
        skip_exc_types = {None, "TimeoutError", "TimeoutExpired", "MemoryError"}
        if node.exc_type in skip_exc_types:
            return node

        current = node
        for attempt in range(1, max_attempts + 1):
            log_msg(
                "INFO",
                f"Debug chain attempt {attempt}/{max_attempts}: "
                f"node={node.id[:8]}, exc_type={current.exc_type}",
            )

            debug_context = {
                "buggy_code": current.code,
                "exc_type": current.exc_type or "",
                "term_out": current.term_out if isinstance(current.term_out, str) else "",
                "task_desc": context.task_desc,
                "data_preview": "",
                "device_info": context.device_info,
                "conda_packages": context.conda_packages,
                "conda_env_name": context.conda_env_name,
            }

            fixed_node = agent._debug(debug_context)

            if fixed_node and fixed_node.code and fixed_node.code != current.code:
                fix_exec_result = self._execute_code(fixed_node.code, fixed_node.id)
                fixed_node.term_out = "\n".join(fix_exec_result.term_out)
                fixed_node.exec_time = fix_exec_result.exec_time
                fixed_node.exc_type = fix_exec_result.exc_type
                fixed_node.exc_info = (
                    str(fix_exec_result.exc_info) if fix_exec_result.exc_info else None
                )
                # 保留原始 parent_id 和 task_type
                fixed_node.parent_id = node.parent_id
                fixed_node.task_type = node.task_type
                fixed_node.debug_attempts = attempt

                if fixed_node.exc_type is None:
                    log_msg("INFO", f"Debug chain 成功（第 {attempt} 次）")
                    return fixed_node

                # 仍有错误，继续下一轮（用新输出作为下一轮输入）
                log_msg(
                    "INFO",
                    f"Debug attempt {attempt} 仍有错误 ({fixed_node.exc_type})，继续",
                )
                current = fixed_node
            else:
                log_msg("INFO", f"Debug attempt {attempt} 未产生有效代码，停止")
                break

        # 所有尝试耗尽，标记为 dead
        node.dead = True
        node.debug_attempts = max_attempts
        log_msg(
            "WARNING",
            f"Debug chain 耗尽（{max_attempts} 次），节点 {node.id[:8]} 标记为 dead",
        )
        return node

    def _build_draft_history(self) -> List[str]:
        """收集已有方案的 approach_tag 列表（用于 Phase 1 多样性引导）。

        从 Journal 中提取所有非死节点的 approach_tag，去重后按出现顺序返回。

        Returns:
            approach_tag 列表（已去重）
        """
        seen: set = set()
        history: List[str] = []
        with self.journal_lock:
            for node in self.journal.nodes:
                tag = node.approach_tag
                if tag and not node.dead and tag not in seen:
                    seen.add(tag)
                    history.append(tag)
        return history

    def _draft_step(self, draft_history: Optional[List[str]] = None) -> Optional[Node]:
        """执行单个 draft 步骤（Phase 1 专用）。

        与 _step_task() 的区别：
        - 始终使用 task_type="draft"（对应 draft.j2 模板，无父代）
        - 注入 draft_history（已用方法列表，用于多样性引导）
        - 使用 _debug_chain() 替代单次 debug

        Args:
            draft_history: 已用方法标签列表（None 表示首次）

        Returns:
            生成的节点（失败时返回 None）
        """
        try:
            # draft 使用 explore 类型 Agent（功能相同）
            if self.task_dispatcher:
                agent = self.task_dispatcher.select_agent("explore")
            else:
                agent = random.choice(self.agents)

            self._prepare_step()

            with self.journal_lock:
                current_step = len(self.journal.nodes)

            context = AgentContext(
                task_type="draft",
                parent_node=None,
                journal=self.journal,
                config=self.config,
                start_time=self.start_time,
                current_step=current_step,
                task_desc=self.task_desc,
                device_info=self.device_info,
                conda_packages=self.conda_packages,
                conda_env_name=self.conda_env_name,
                experience_pool=self.experience_pool,
                draft_history=draft_history,
            )

            result = agent.generate(context)

            if not result.success or result.node is None:
                log_msg("WARNING", f"{agent.name} draft 生成失败: {result.error}")
                return None

            node = result.node

            exec_result = self._execute_code(node.code, node.id)
            node.term_out = "\n".join(exec_result.term_out)
            node.exec_time = exec_result.exec_time
            node.exc_type = exec_result.exc_type
            node.exc_info = str(exec_result.exc_info) if exec_result.exc_info else None

            node = self._debug_chain(node, agent, context)

            self._review_node(node, parent_node=None)

            with self.save_lock:
                self._save_node_solution(node)

            with self.journal_lock:
                self.journal.append(node)
                self._update_best_node(node)

            if self.experience_pool:
                self._write_experience_pool(agent.name, "draft", node)

            self._print_node_summary(node)

            log_msg(
                "INFO",
                f"{agent.name} draft 完成: is_buggy={node.is_buggy}, dead={node.dead}",
            )
            return node

        except Exception as e:
            log_exception(e, "Orchestrator _draft_step() 执行失败")
            return None

    def run_epoch_draft(self, steps: int) -> List[Node]:
        """执行一个 Phase 1 Draft Epoch（固定步数，终止条件由调用方控制）。

        与旧版的区别：
        - 旧版：内部 while 循环，自行检查 valid_pool 目标和总预算
        - 新版：跑满 steps 步后返回，valid_pool 检测和 Agent 进化由 main.py 的 while 循环负责

        Args:
            steps: 本次 epoch 执行的步数

        Returns:
            本 epoch 生成的所有 Node 列表（含 dead 节点）
        """
        generated: List[Node] = []

        log_msg(
            "INFO",
            f"===== Phase 1 Draft Epoch 开始 (steps={steps}) =====",
        )

        for _ in range(steps):
            if self._check_time_limit():
                break

            draft_history = self._build_draft_history() or None
            node = self._draft_step(draft_history)

            if node:
                generated.append(node)

        with self.journal_lock:
            current_valid = len(
                [n for n in self.journal.nodes if not n.is_buggy and not n.dead]
            )
        log_msg(
            "INFO",
            f"===== Phase 1 Draft Epoch 完成: 生成 {len(generated)} 个节点, "
            f"valid={current_valid} =====",
        )
        return generated

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
        self,
        primary_parent: Node,
        gene_plan: Dict,
        gene_sources: Optional[Dict[str, str]] = None,
    ) -> Optional[Node]:
        """执行 merge 任务（基因交叉）。

        Args:
            primary_parent: 贡献基因最多的父代（用于 merge.j2 参考框架）
            gene_plan: 基因交叉计划（pheromone_with_degenerate_check 的输出）
            gene_sources: {locus: source_node_id} 字典（可选，用于 node.metadata 记录）

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
                f"{agent.name} 开始 merge (primary_parent={primary_parent.id[:8]})",
            )

            # 构建上下文
            with self.journal_lock:
                current_step = len(self.journal.nodes)

            context = AgentContext(
                task_type="merge",
                parent_node=None,
                journal=self.journal,
                config=self.config,
                start_time=self.start_time,
                current_step=current_step,
                task_desc=self.task_desc,
                device_info=self.device_info,
                conda_packages=self.conda_packages,
                conda_env_name=self.conda_env_name,
                primary_parent=primary_parent,
                gene_plan=gene_plan,
                gene_sources=gene_sources,
                experience_pool=self.experience_pool,
            )

            # 生成代码
            result = agent.generate(context)

            if not result.success or result.node is None:
                log_msg("WARNING", f"{agent.name} merge 失败: {result.error}")
                return None

            node = result.node
            # 记录基因来源（用于分析）
            if gene_sources:
                node.metadata["gene_sources"] = gene_sources

            # 执行代码
            exec_result = self._execute_code(node.code, node.id)
            node.term_out = "\n".join(exec_result.term_out)
            node.exec_time = exec_result.exec_time
            node.exc_type = exec_result.exc_type
            node.exc_info = str(exec_result.exc_info) if exec_result.exc_info else None

            # 链式 Debug（非超时/OOM 错误才重试）
            node = self._debug_chain(node, agent, context)

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

            # 链式 Debug（非超时/OOM 错误才重试）
            node = self._debug_chain(node, agent, context)

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
