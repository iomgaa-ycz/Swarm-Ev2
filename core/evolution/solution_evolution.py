"""Solution 层遗传算法实现。

实现种群初始化、锦标赛选择、基因交叉（merge）、基因变异（mutate）等操作。
两阶段进化架构：Phase 1 由 Orchestrator.run_epoch_draft() 驱动，Phase 2 由 run_epoch() 驱动。
"""

from __future__ import annotations

import random
from typing import List, Dict, Any, Optional, Tuple

from core.state import Node, Journal
from core.evolution.gene_parser import select_mutation_target, validate_genes
from core.evolution.gene_registry import GeneRegistry
from core.evolution.gene_selector import (
    pheromone_with_degenerate_check,
    get_primary_parent,
    LOCUS_TO_FIELD,
)
from utils.config import Config
from utils.logger_system import log_msg


class SolutionEvolution:
    """Solution 层遗传算法（两阶段进化架构 Phase 2）。

    Phase 2 进化策略：
    - 50% 概率 merge（pheromone 全局 TOP-1 + 退化检测，无 Tournament 选父）
    - 50% 概率 mutate（Tournament 选父 + 非 stub 基因选择）

    触发条件：valid_pool（非 buggy，非 dead）数量 >= ga_trigger_threshold。

    注意：Phase 1（draft 模式）由 Orchestrator.run_epoch_draft() 驱动。
    """

    def __init__(
        self,
        config: Config,
        journal: Journal,
        orchestrator=None,
        gene_registry: Optional[GeneRegistry] = None,
    ):
        """初始化遗传算法（两阶段进化版）。

        Args:
            config: 全局配置
            journal: 节点历史记录
            orchestrator: Orchestrator 实例（用于执行任务）
            gene_registry: 基因注册表（信息素驱动交叉时必需）
        """
        self.config = config
        self.journal = journal
        self.orchestrator = orchestrator
        self.gene_registry = gene_registry

        # GA 参数
        self.population_size = config.evolution.solution.population_size
        self.elite_size = config.evolution.solution.elite_size
        self.crossover_rate = config.evolution.solution.crossover_rate
        self.mutation_rate = config.evolution.solution.mutation_rate
        self.tournament_k = config.evolution.solution.tournament_k
        self.crossover_strategy = config.evolution.solution.crossover_strategy
        # GA 触发阈值（解耦于 population_size，允许更早触发进化）
        self.ga_trigger_threshold = config.evolution.solution.ga_trigger_threshold
        # 两阶段进化参数（P1 新增）
        self.phase1_target_nodes = getattr(
            config.evolution.solution, "phase1_target_nodes", 8
        )

        # 当前种群
        self.population: List[Node] = []

        log_msg(
            "INFO",
            f"SolutionEvolution 初始化（两阶段进化）: population_size={self.population_size}, "
            f"crossover_strategy={self.crossover_strategy}, "
            f"phase1_target_nodes={self.phase1_target_nodes}",
        )

    def run_epoch(self, steps_per_epoch: int) -> Optional[Node]:
        """运行 Phase 2 进化 Epoch（50% merge + 50% mutate）。

        触发条件：valid_pool（非 buggy，非 dead）数量 >= ga_trigger_threshold。
        merge 策略：pheromone 全局 TOP-1 + 退化检测，无 Tournament 选父。
        mutate 策略：Tournament 选父代 + 非 stub 基因选择。

        Args:
            steps_per_epoch: 本轮进化步数（merge + mutate 共同计入）

        Returns:
            本 Epoch 结束后 Journal 中的全局最佳节点
        """
        log_msg("INFO", "===== SolutionEvolution: Phase 2 run_epoch 开始 =====")

        # 获取 valid_pool（非 buggy，非 dead，且基因结构完整）
        valid_pool = [
            n for n in self.journal.nodes
            if not n.is_buggy and not n.dead and validate_genes(n.genes)
        ]

        if len(valid_pool) < self.ga_trigger_threshold:
            log_msg(
                "WARNING",
                f"valid_pool 不足 ({len(valid_pool)}/{self.ga_trigger_threshold})，"
                f"跳过 Phase 2，继续等待 Phase 1 填充",
            )
            return None

        actual_size = min(self.population_size, len(valid_pool))
        self.population = valid_pool[-actual_size:]
        log_msg("INFO", f"Phase 2 种群: {len(self.population)} 个节点")

        merge_count = 0
        mutate_count = 0

        for step in range(steps_per_epoch):
            if random.random() < 0.5:
                node = self._run_merge_step()
                if node:
                    merge_count += 1
            else:
                node = self._run_mutate_step()
                if node:
                    mutate_count += 1

        log_msg(
            "INFO",
            f"Phase 2 完成: steps={steps_per_epoch}, "
            f"merge={merge_count}, mutate={mutate_count}",
        )

        global_direction = (
            self.orchestrator._global_lower_is_better
            if self.orchestrator
            and hasattr(self.orchestrator, "_global_lower_is_better")
            else None
        )
        best_node = self.journal.get_best_node(
            only_good=True, lower_is_better=global_direction
        )

        if best_node:
            log_msg(
                "INFO",
                f"===== Phase 2 run_epoch 完成 | 最佳 metric: {best_node.metric_value} =====",
            )
        else:
            log_msg("WARNING", "===== Phase 2 run_epoch 完成 | 未找到有效节点 =====")

        return best_node

    def _is_lower_better(self) -> bool:
        """判断当前任务的 metric 方向（P0-1 修复：优先使用全局方向）。

        优先级：
        1. Orchestrator 的全局方向（确定性来源）
        2. 种群中第一个有效节点的方向（fallback）

        Returns:
            True 表示越小越好（如 RMSE/logloss），False 表示越大越好（如 AUC）
        """
        # 优先使用 Orchestrator 的全局方向（确定性来源）
        if self.orchestrator and hasattr(self.orchestrator, "_global_lower_is_better"):
            if self.orchestrator._global_lower_is_better is not None:
                return self.orchestrator._global_lower_is_better

        # Fallback: 种群中第一个有效节点
        for node in self.population:
            if node.metric_value is not None:
                return node.lower_is_better

        return False  # 默认 higher_is_better

    def _tournament_select(self) -> Node:
        """锦标赛选择单个父代（正确处理 metric 方向）。

        随机抽取 k 个个体，返回其中最优者。

        Returns:
            选中的父代节点
        """
        tournament = random.sample(self.population, k=self.tournament_k)
        lower = self._is_lower_better()
        if lower:
            winner = min(
                tournament,
                key=lambda n: (
                    n.metric_value if n.metric_value is not None else float("inf")
                ),
            )
        else:
            winner = max(
                tournament,
                key=lambda n: (
                    n.metric_value if n.metric_value is not None else float("-inf")
                ),
            )
        return winner

    def _build_gene_plan_markdown_from_pheromone(
        self, raw_plan: Dict[str, Any]
    ) -> Tuple[str, Dict[str, str]]:
        """将信息素选择结果格式化为统一 Markdown 和 gene_sources 字典。

        Args:
            raw_plan: pheromone_with_degenerate_check() 的返回值

        Returns:
            (gene_plan_md, gene_sources) 元组
            - gene_plan_md: Markdown 格式的基因计划字符串
            - gene_sources: {locus: source_node_id} 字典
        """
        lines: List[str] = []
        gene_sources: Dict[str, str] = {}

        for locus, field_name in LOCUS_TO_FIELD.items():
            item = raw_plan.get(field_name)
            if not item:
                continue
            node_id = item["source_node_id"][:8]
            full_node_id = item["source_node_id"]
            score = item.get("source_score", 0.0)
            code = item["code"]
            lines.append(f"### {locus} (from {node_id}, fitness={score:.4f})")
            lines.append(f"```python\n{code}\n```\n")
            gene_sources[locus] = full_node_id

        return "\n".join(lines), gene_sources

    def _run_merge_step(self) -> Optional[Node]:
        """执行一次 merge 操作（Phase 2 内部使用）。

        策略：
        1. pheromone_with_degenerate_check() 选 4 个全局最优基因
        2. get_primary_parent() 推断主父代（贡献基因最多的节点）
        3. 调用 Orchestrator 执行 merge 任务

        Returns:
            生成的子代节点（失败时返回 None）
        """
        if not self.orchestrator or not self.gene_registry:
            log_msg("WARNING", "Orchestrator 或 GeneRegistry 未初始化，跳过 merge")
            return None

        current_step = len(self.journal.nodes)

        # 信息素 TOP-1 + 退化检测
        raw_plan = pheromone_with_degenerate_check(
            self.journal, self.gene_registry, current_step
        )

        # 推断主父代
        try:
            primary_parent = get_primary_parent(raw_plan, self.journal)
        except ValueError as e:
            log_msg("WARNING", f"无法推断 primary_parent: {e}，跳过 merge")
            return None

        # 构建 gene_plan Markdown 和 gene_sources
        gene_plan_md, gene_sources = self._build_gene_plan_markdown_from_pheromone(
            raw_plan
        )

        log_msg("INFO", f"Merge: primary_parent={primary_parent.id[:8]}")
        return self.orchestrator.execute_merge_task(
            primary_parent, gene_plan_md, gene_sources
        )

    def _run_mutate_step(self) -> Optional[Node]:
        """执行一次 mutate 操作（Phase 2 内部使用）。

        策略：
        1. Tournament 选父代（质量最优）
        2. select_mutation_target() 选非 stub 基因块和子方面
        3. 调用 Orchestrator 执行 mutate 任务

        Returns:
            变异后的节点（失败时返回 None）
        """
        if not self.orchestrator:
            log_msg("WARNING", "Orchestrator 未初始化，跳过 mutate")
            return None

        if len(self.population) < self.tournament_k:
            log_msg(
                "WARNING",
                f"种群过小（{len(self.population)} < {self.tournament_k}），跳过 mutate",
            )
            return None

        parent = self._tournament_select()
        target_gene, mutation_aspect = select_mutation_target(parent)

        log_msg("INFO", f"Mutate: parent={parent.id[:8]}, gene={target_gene}, aspect={mutation_aspect}")
        return self.orchestrator.execute_mutate_task(parent, target_gene, mutation_aspect)
