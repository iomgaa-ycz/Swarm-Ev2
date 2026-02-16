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

        # [5] 返回最佳节点（P0-1 修复：传入全局方向）
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
                f"===== SolutionEvolution: run_epoch 完成 | 最佳 metric: {best_node.metric_value} =====",
            )
        else:
            log_msg(
                "WARNING",
                "===== SolutionEvolution: run_epoch 完成 | 未找到有效节点 =====",
            )

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
        增加兼容性预检：检测框架冲突并注入警告。

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

            # 兼容性预检（仅随机策略，信息素策略已从全局优选）
            gene_plan_md = self._inject_compatibility_warnings(
                parent_a, parent_b, gene_plan_md
            )

        # 调用 Orchestrator 执行 merge 任务（gene_plan 为 Markdown 字符串）
        child = self.orchestrator.execute_merge_task(parent_a, parent_b, gene_plan_md)

        return child

    def _inject_compatibility_warnings(
        self, parent_a: Node, parent_b: Node, gene_plan_md: str
    ) -> str:
        """检测基因兼容性并注入警告到 gene_plan。

        当两个父代使用不同框架（如 torch vs sklearn）时，
        在 gene_plan 前注入详细警告，让 LLM 自行选择与修改。

        Args:
            parent_a: 父代 A
            parent_b: 父代 B
            gene_plan_md: 原始 gene_plan Markdown

        Returns:
            可能增加了警告前缀的 gene_plan Markdown
        """
        from core.evolution.gene_compatibility import check_gene_compatibility

        genes_a = parent_a.genes or {}
        genes_b = parent_b.genes or {}

        # 构建基因选择方案（从 gene_plan_md 中解析）
        gene_plan_choices: Dict[str, str] = {}
        for gene in REQUIRED_GENES:
            # 从 Markdown 中匹配 "### GENE (from node_id, ...)"
            if f"### {gene}" in gene_plan_md:
                # 简单判断来自哪个父代
                idx = gene_plan_md.index(f"### {gene}")
                section = gene_plan_md[idx : idx + 200]
                if parent_a.id[:8] in section:
                    gene_plan_choices[gene] = "A"
                elif parent_b.id[:8] in section:
                    gene_plan_choices[gene] = "B"
                else:
                    gene_plan_choices[gene] = random.choice(["A", "B"])

        if not gene_plan_choices:
            return gene_plan_md

        compat = check_gene_compatibility(
            parent_a_code=parent_a.code or "",
            parent_b_code=parent_b.code or "",
            genes_a=genes_a,
            genes_b=genes_b,
            gene_plan_choices=gene_plan_choices,
        )

        if compat.conflicts:
            warning_text = "\n".join(f"⚠️ {c}" for c in compat.conflicts)
            gene_plan_md = (
                f"# ⚠️ Compatibility Warnings\n"
                f"The following conflicts were detected. You MUST resolve them:\n"
                f"{warning_text}\n\n"
                f"Suggestion: Choose components from ONE framework consistently, "
                f"or adapt the conflicting blocks to be compatible.\n\n"
                f"{gene_plan_md}"
            )
            log_msg(
                "WARNING",
                f"Merge 兼容性警告: {len(compat.conflicts)} 个冲突已注入 prompt",
            )

        return gene_plan_md

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
