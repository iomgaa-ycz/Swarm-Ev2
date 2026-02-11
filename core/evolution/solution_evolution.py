"""Solution 层遗传算法实现。

实现种群初始化、精英保留、锦标赛选择、基因交叉、基因变异等遗传算法核心操作。
"""

from __future__ import annotations

import random
from typing import List, Dict, Any, Optional

from core.state import Node, Journal
from core.evolution.gene_parser import REQUIRED_GENES
from core.evolution.gene_registry import GeneRegistry
from core.evolution.gene_selector import select_gene_plan
from utils.config import Config
from utils.logger_system import log_msg


class SolutionEvolution:
    """Solution 层遗传算法。

    实现基于遗传算法的 Solution 进化，支持随机交叉和信息素驱动交叉两种策略。

    注意：MVP 阶段简化实现，使用 Orchestrator 执行任务而非直接调用 Evaluator。
    """

    def __init__(
        self,
        config: Config,
        journal: Journal,
        orchestrator=None,
        gene_registry: Optional[GeneRegistry] = None,
    ):
        """初始化遗传算法（MVP 简化版）。

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

        # 当前种群
        self.population: List[Node] = []

        log_msg(
            "INFO",
            f"SolutionEvolution 初始化（MVP）: population_size={self.population_size}, "
            f"crossover_strategy={self.crossover_strategy}",
        )

    def run_epoch(self, steps_per_epoch: int) -> Optional[Node]:
        """运行单个 Epoch 的 Solution 层进化（MVP 主入口）。

        流程:
        1. 获取当前种群（从 Journal 的 good_nodes）
        2. 精英保留
        3. 锦标赛选择 + 交叉（merge）
        4. 变异（mutate）
        5. 返回最佳节点

        Args:
            steps_per_epoch: 每个 Epoch 的步数

        Returns:
            本 Epoch 最佳节点
        """
        log_msg("INFO", "===== SolutionEvolution: run_epoch 开始 =====")

        # [1] 获取当前种群（从 Journal）
        good_nodes = [n for n in self.journal.nodes if not n.is_buggy]

        if len(good_nodes) < self.population_size:
            log_msg(
                "WARNING",
                f"当前 good_nodes 数量不足 ({len(good_nodes)}/{self.population_size})，"
                f"跳过进化，继续探索",
            )
            return None

        # 取最近的 population_size 个节点作为当前种群
        self.population = good_nodes[-self.population_size :]
        log_msg("INFO", f"当前种群: {len(self.population)} 个节点")

        # [2] 精英保留
        elites = self._select_elites()
        log_msg("INFO", f"精英保留: {len(elites)} 个节点")

        # [3] 交叉：生成 (population_size - elite_size) 个子代
        num_offspring = self.population_size - self.elite_size
        offspring_count = 0

        for _ in range(num_offspring):
            # 锦标赛选择父代对
            parent_a = self._tournament_select()
            parent_b = self._tournament_select()

            # 概率触发交叉
            if random.random() < self.crossover_rate:
                child = self._crossover_mvp(parent_a, parent_b)
                if child:
                    offspring_count += 1
            else:
                log_msg("DEBUG", "跳过交叉（概率未触发）")

        log_msg("INFO", f"交叉完成: {offspring_count} 个子代")

        # [4] 变异：对 Journal 中最后 num_offspring 个节点进行变异
        mutation_count = 0
        recent_nodes = [n for n in self.journal.nodes if not n.is_buggy][
            -num_offspring:
        ]

        for node in recent_nodes:
            if random.random() < self.mutation_rate:
                self._mutate_mvp(node)
                mutation_count += 1

        log_msg("INFO", f"变异完成: {mutation_count} 个节点")

        # [5] 返回最佳节点
        best_node = self.journal.get_best_node(only_good=True)

        if best_node:
            log_msg(
                "INFO",
                f"===== SolutionEvolution: run_epoch 完成 | 最佳 metric: {best_node.metric_value} =====",
            )
        else:
            log_msg(
                "WARNING",
                "===== SolutionEvolution: run_epoch 完成 | 未找到有效节点 =====",
            )

        return best_node

    def _is_lower_better(self) -> bool:
        """判断当前任务的 metric 方向。

        使用种群中第一个有 metric 的节点的 lower_is_better 属性。
        同一任务内所有节点方向一致。

        Returns:
            True 表示越小越好（如 RMSE/logloss），False 表示越大越好（如 AUC）
        """
        for node in self.population:
            if node.metric_value is not None:
                return node.lower_is_better
        return False  # 默认 higher_is_better

    def _select_elites(self) -> List[Node]:
        """精英保留（返回 top-K 节点，正确处理 metric 方向）。

        Returns:
            精英节点列表
        """
        lower = self._is_lower_better()
        if lower:
            # lower_is_better: 升序排列，取最小值
            sorted_pop = sorted(
                self.population,
                key=lambda n: (
                    n.metric_value if n.metric_value is not None else float("inf")
                ),
            )
        else:
            # higher_is_better: 降序排列，取最大值
            sorted_pop = sorted(
                self.population,
                key=lambda n: (
                    n.metric_value if n.metric_value is not None else float("-inf")
                ),
                reverse=True,
            )
        return sorted_pop[: self.elite_size]

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

    def _crossover_mvp(self, parent_a: Node, parent_b: Node) -> Optional[Node]:
        """基因交叉（MVP 简化版，调用 Orchestrator）。

        根据 crossover_strategy 选择随机或信息素驱动策略，
        两种策略均输出统一的 Markdown 格式 gene_plan。

        Args:
            parent_a: 父代 A
            parent_b: 父代 B

        Returns:
            子代节点（失败时返回 None）
        """
        if not self.orchestrator:
            log_msg("WARNING", "Orchestrator 未初始化，跳过交叉")
            return None

        if self.crossover_strategy == "pheromone" and self.gene_registry:
            # 信息素策略：从全局历史中选最优基因
            current_step = len(self.journal.nodes)
            raw_plan = select_gene_plan(self.journal, self.gene_registry, current_step)
            gene_plan_md = self._build_gene_plan_markdown_from_pheromone(raw_plan)
            log_msg("INFO", f"信息素驱动交叉: {len(REQUIRED_GENES)} 个基因位点")
        else:
            # 随机策略：从两个父代中随机选基因
            gene_plan_md = self._build_gene_plan_markdown_from_random(
                parent_a, parent_b
            )
            log_msg("INFO", f"随机交叉: {len(REQUIRED_GENES)} 个基因位点")

        # 调用 Orchestrator 执行 merge 任务（gene_plan 为 Markdown 字符串）
        child = self.orchestrator.execute_merge_task(parent_a, parent_b, gene_plan_md)

        return child

    def _build_gene_plan_markdown_from_pheromone(self, raw_plan: Dict[str, Any]) -> str:
        """将信息素选择结果格式化为统一 Markdown。

        Args:
            raw_plan: select_gene_plan() 的返回值

        Returns:
            Markdown 格式的 gene_plan 字符串
        """
        from core.evolution.gene_selector import LOCUS_TO_FIELD

        lines: List[str] = []
        for locus, field_name in LOCUS_TO_FIELD.items():
            item = raw_plan.get(field_name)
            if not item:
                continue
            node_id = item["source_node_id"][:8]
            score = item.get("source_score", 0.0)
            code = item["code"]
            lines.append(f"### {locus} (from {node_id}, fitness={score:.4f})")
            lines.append(f"```python\n{code}\n```\n")

        return "\n".join(lines)

    def _build_gene_plan_markdown_from_random(
        self, parent_a: Node, parent_b: Node
    ) -> str:
        """将随机交叉结果格式化为统一 Markdown。

        从两个父代中随机选基因，提取对应代码片段。

        Args:
            parent_a: 父代 A
            parent_b: 父代 B

        Returns:
            Markdown 格式的 gene_plan 字符串
        """
        lines: List[str] = []
        genes_a = parent_a.genes or {}
        genes_b = parent_b.genes or {}

        for gene in REQUIRED_GENES:
            chosen = random.choice(["A", "B"])
            if chosen == "A":
                source_node = parent_a
                code = genes_a.get(gene, "# (no code)")
            else:
                source_node = parent_b
                code = genes_b.get(gene, "# (no code)")

            node_id = source_node.id[:8]
            score = source_node.metric_value or 0.0
            lines.append(f"### {gene} (from {node_id}, fitness={score:.4f})")
            lines.append(f"```python\n{code}\n```\n")

        return "\n".join(lines)

    def _mutate_mvp(self, node: Node) -> None:
        """基因变异（MVP 简化版，调用 Orchestrator）。

        Args:
            node: 待变异的节点（原地修改）
        """
        if not self.orchestrator:
            log_msg("WARNING", "Orchestrator 未初始化，跳过变异")
            return

        # 随机选择一个基因块进行变异
        target_gene = random.choice(REQUIRED_GENES)

        log_msg("INFO", f"变异目标基因: {target_gene}")

        # 调用 Orchestrator 执行 mutate 任务
        mutated_node = self.orchestrator.execute_mutate_task(node, target_gene)

        if mutated_node:
            log_msg("DEBUG", f"变异成功: {mutated_node.id[:8]}")
        else:
            log_msg("WARNING", "变异失败")
