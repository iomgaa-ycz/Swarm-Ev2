"""并行评估器模块。

使用多线程并发执行和评估多个 Solution。
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional

from agents.base_agent import BaseAgent, AgentContext, AgentResult
from core.state import Node
from core.executor.interpreter import Interpreter
from core.executor.workspace import WorkspaceManager
from core.evolution.gene_registry import GeneRegistry
from core.evolution.pheromone import compute_node_pheromone, ensure_node_stats
from search.fitness import normalize_fitness
from utils.config import Config
from utils.logger_system import log_msg, log_exception


class ParallelEvaluator:
    """并行评估器。

    使用线程池并发执行和评估多个 Solution，提高效率。
    """

    def __init__(
        self,
        max_workers: int,
        workspace: WorkspaceManager,
        interpreter: Interpreter,
        gene_registry: GeneRegistry,
        config: Config,
    ):
        """初始化并行评估器。

        Args:
            max_workers: 最大并发线程数（使用 config.search.parallel_num）
            workspace: 工作空间管理器
            interpreter: 代码执行器
            gene_registry: 基因注册表（用于更新基因信息素）
            config: 全局配置
        """
        self.max_workers = max_workers
        self.workspace = workspace
        self.interpreter = interpreter
        self.gene_registry = gene_registry
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        log_msg("INFO", f"ParallelEvaluator 初始化: max_workers={max_workers}")

    def batch_generate(
        self, tasks: List[Tuple[BaseAgent, AgentContext]]
    ) -> List[AgentResult]:
        """并行生成多个 Solution（用于初始化种群）。

        Args:
            tasks: (Agent, AgentContext) 列表

        Returns:
            AgentResult 列表
        """
        log_msg("INFO", f"开始并行生成 {len(tasks)} 个 Solution")

        futures = []
        for agent, context in tasks:
            future = self.executor.submit(agent.generate, context)
            futures.append(future)

        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                log_msg("ERROR", f"Agent 生成失败: {e}")
                # 创建失败的 AgentResult
                results.append(
                    AgentResult(node=None, success=False, error=str(e))
                )

        log_msg("INFO", f"并行生成完成: 成功 {sum(r.success for r in results)}/{len(results)}")
        return results

    def batch_evaluate(self, nodes: List[Node], current_step: int) -> None:
        """并行评估多个节点。

        执行代码、解析 metric、更新节点和基因信息素。

        Args:
            nodes: 待评估的节点列表
            current_step: 当前步骤编号
        """
        log_msg("INFO", f"开始并行评估 {len(nodes)} 个节点（step={current_step}）")

        # 提交评估任务
        futures = {}
        for node in nodes:
            future = self.executor.submit(self._evaluate_one, node)
            futures[future] = node

        # 收集结果
        for future in as_completed(futures):
            node = futures[future]
            try:
                metric_value = future.result()
                node.metric_value = metric_value
                log_msg("INFO", f"节点 {node.id[:6]} 评估完成: metric={metric_value:.4f}")
            except Exception as e:
                log_msg("ERROR", f"节点 {node.id[:6]} 评估失败: {e}")
                node.is_buggy = True
                node.metric_value = -1e9

        # 更新信息素
        self._update_pheromones(nodes, current_step)

        success_count = sum(1 for n in nodes if not n.is_buggy)
        log_msg("INFO", f"并行评估完成: 成功 {success_count}/{len(nodes)}")

    def _evaluate_one(self, node: Node) -> float:
        """评估单个节点。

        执行代码并解析 metric。

        Args:
            node: 待评估的节点

        Returns:
            metric 值（buggy 节点返回 -1e9）
        """
        # 重写 submission 路径（避免并发冲突）
        code = self.workspace.rewrite_submission_path(node.code, node.id)

        # 执行代码
        exec_result = self.interpreter.run(code)

        # 更新节点执行信息
        node.term_out = "\n".join(exec_result.term_out)
        node.exec_time = exec_result.exec_time
        node.exc_type = exec_result.exc_type
        node.exc_info = exec_result.exc_info

        # 检查执行状态
        if not exec_result.success:
            node.is_buggy = True
            log_msg(
                "WARNING",
                f"节点 {node.id[:6]} 执行失败: {exec_result.exc_type}",
            )
            return -1e9

        # 解析 metric
        try:
            metric_value = self._parse_metric(node.term_out)
            node.is_buggy = False
            return metric_value
        except Exception as e:
            log_msg("WARNING", f"节点 {node.id[:6]} metric 解析失败: {e}")
            node.is_buggy = False  # 执行成功但无 metric（设为 0.0）
            return 0.0

    def _parse_metric(self, term_out: str) -> float:
        """从执行输出中提取 metric。

        支持的格式：
        - "Metric: 0.85"
        - "Score: 0.85"
        - "Accuracy: 0.85"
        - "RMSE: 0.15"

        Args:
            term_out: 终端输出字符串

        Returns:
            解析到的 metric 值

        Raises:
            ValueError: 无法解析 metric
        """
        # 正则匹配常见 metric 格式
        patterns = [
            r"Metric:\s*([-+]?\d*\.?\d+)",
            r"Score:\s*([-+]?\d*\.?\d+)",
            r"Accuracy:\s*([-+]?\d*\.?\d+)",
            r"F1:\s*([-+]?\d*\.?\d+)",
            r"AUC:\s*([-+]?\d*\.?\d+)",
            r"RMSE:\s*([-+]?\d*\.?\d+)",
            r"MAE:\s*([-+]?\d*\.?\d+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, term_out, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                return value

        raise ValueError("无法从输出中解析 metric")

    def _update_pheromones(self, nodes: List[Node], current_step: int) -> None:
        """更新节点和基因信息素。

        Args:
            nodes: 节点列表
            current_step: 当前步骤
        """
        # 计算得分范围
        valid_nodes = [n for n in nodes if not n.is_buggy and n.metric_value is not None]
        if not valid_nodes:
            log_msg("WARNING", "没有有效节点，跳过信息素更新")
            return

        scores = [n.metric_value for n in valid_nodes]
        score_min = min(scores)
        score_max = max(scores)

        # 更新节点信息素
        for node in nodes:
            # 确保节点有统计字段
            ensure_node_stats(node)

            # 计算节点信息素
            pheromone = compute_node_pheromone(
                node,
                current_step=current_step,
                score_min=score_min,
                score_max=score_max,
            )
            node.metadata["pheromone_node"] = pheromone

            # 更新基因注册表
            if not node.is_buggy and node.metric_value is not None:
                self.gene_registry.update_from_reviewed_node(node)

        log_msg("INFO", f"信息素更新完成: {len(valid_nodes)} 个有效节点")

    def shutdown(self) -> None:
        """关闭线程池。

        确保所有任务完成并释放资源。
        """
        log_msg("INFO", "关闭 ParallelEvaluator 线程池")
        self.executor.shutdown(wait=True)
