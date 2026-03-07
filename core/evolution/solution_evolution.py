"""Solution 层遗传算法实现。

实现种群初始化、锦标赛选择、两亲本交叉（merge）、适应度共享变异（mutate）等操作。
两阶段进化架构：Phase 1 由 Orchestrator.run_epoch_draft() 驱动，Phase 2 由 run_epoch() 驱动。

V7 改进：
- Merge：从信息素 per-locus TOP-1 贪心 → 两亲本交叉（同物种内锦标赛选择 + uniform crossover）
- Mutate：从 top-K 截断种群 → 全 valid_pool + 适应度共享（保护稀有物种）
- 种群管理：移除 top-K 截断，使用全部有效节点
"""

from __future__ import annotations

import random
from typing import List, Dict, Optional, Tuple

from core.state import Node, Journal
from core.evolution.gene_parser import select_mutation_target, validate_genes
from core.evolution.gene_registry import GeneRegistry
from core.evolution.gene_compatibility import detect_framework, check_gene_compatibility
from utils.config import Config
from utils.logger_system import log_msg

# 基因位点列表
GENE_LOCI = ["DATA", "MODEL", "TRAIN", "POSTPROCESS"]


class SolutionEvolution:
    """Solution 层遗传算法（两阶段进化架构 Phase 2）。

    Phase 2 进化策略（V7）：
    - 50% 概率 merge（同物种两亲本交叉 + uniform crossover）
    - 50% 概率 mutate（适应度共享锦标赛选择 + 非 stub 基因变异）

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
            f"SolutionEvolution 初始化（两阶段进化 V7）: population_size={self.population_size}, "
            f"crossover_strategy={self.crossover_strategy}, "
            f"phase1_target_nodes={self.phase1_target_nodes}",
        )

    def run_epoch(self, steps_per_epoch: int) -> Optional[Node]:
        """运行 Phase 2 进化 Epoch（50% merge + 50% mutate）。

        V7 改进：种群使用全部 valid_pool，不截断为 top-K。

        Args:
            steps_per_epoch: 本轮进化步数（merge + mutate 共同计入）

        Returns:
            本 Epoch 结束后 Journal 中的全局最佳节点
        """
        log_msg("INFO", "===== SolutionEvolution: Phase 2 run_epoch 开始 =====")

        # 获取 valid_pool（非 buggy，非 dead，且基因结构完整）
        valid_pool = [
            n
            for n in self.journal.nodes
            if not n.is_buggy and not n.dead and validate_genes(n.genes)
        ]

        if len(valid_pool) < self.ga_trigger_threshold:
            log_msg(
                "WARNING",
                f"valid_pool 不足 ({len(valid_pool)}/{self.ga_trigger_threshold})，"
                f"跳过 Phase 2，继续等待 Phase 1 填充",
            )
            return None

        # V7: 使用全部 valid_pool，不截断
        self.population = valid_pool
        log_msg("INFO", f"Phase 2 种群: {len(self.population)} 个节点（全 valid_pool）")

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

    def run_single_ga_step(self) -> Optional[Node]:
        """执行单步 GA 操作（供混合模式调用）。

        包含 valid_pool 检查 + 种群刷新 + 50% merge / 50% mutate。

        Returns:
            生成的子代节点（条件不满足或失败时返回 None）
        """
        valid_pool = [
            n
            for n in self.journal.nodes
            if not n.is_buggy and not n.dead and validate_genes(n.genes)
        ]

        if len(valid_pool) < self.ga_trigger_threshold:
            log_msg(
                "DEBUG",
                f"GA 单步: valid_pool 不足 ({len(valid_pool)}/{self.ga_trigger_threshold})，跳过",
            )
            return None

        # V7: 使用全部 valid_pool，不截断
        self.population = valid_pool

        if random.random() < 0.5:
            return self._run_merge_step()
        else:
            return self._run_mutate_step()

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

    def _tournament_select(
        self,
        pool: Optional[List[Node]] = None,
        use_shared_fitness: bool = False,
    ) -> Node:
        """锦标赛选择单个父代（正确处理 metric 方向）。

        V7 增强：支持自定义 pool 和适应度共享。

        Args:
            pool: 候选池（默认 self.population）
            use_shared_fitness: 是否使用适应度共享值

        Returns:
            选中的父代节点
        """
        if pool is None:
            pool = self.population

        k = min(self.tournament_k, len(pool))
        tournament = random.sample(pool, k=k)

        lower = self._is_lower_better()

        def fitness_key(n: Node) -> float:
            """计算节点适应度值（用于锦标赛比较）。"""
            if use_shared_fitness:
                return self._shared_fitness(n)
            if n.metric_value is not None:
                return n.metric_value
            return float("inf") if lower else float("-inf")

        if lower:
            winner = min(tournament, key=fitness_key)
        else:
            winner = max(tournament, key=fitness_key)
        return winner

    def _detect_species(self, node: Node) -> str:
        """检测节点所属物种（基于代码框架）。

        复用 gene_compatibility.detect_framework()。

        Args:
            node: 待检测的节点

        Returns:
            "torch" | "sklearn" | "tensorflow" | "unknown"
        """
        if not node.code:
            return "unknown"
        framework = detect_framework(node.code)
        return framework or "unknown"

    def _shared_fitness(self, node: Node) -> float:
        """计算适应度共享值：同物种个体分摊适应度。

        公式：shared_fitness = metric / niche_count
        niche_count = 同物种在种群中的个体数

        Args:
            node: 待计算的节点

        Returns:
            共享适应度值（lower_is_better 时取相反数再共享）
        """
        if node.metric_value is None:
            lower = self._is_lower_better()
            return float("inf") if lower else float("-inf")

        species = self._detect_species(node)
        niche_count = sum(
            1 for n in self.population if self._detect_species(n) == species
        )
        niche_count = max(1, niche_count)

        lower = self._is_lower_better()
        if lower:
            # lower_is_better: 值越小越好，共享后应按同样方向比较
            # metric / niche_count 会使值更小，反而更优，需要反转
            # 用 metric * niche_count 让同物种的"有效惩罚"更大
            return node.metric_value * niche_count
        else:
            # higher_is_better: 值越大越好，同物种分摊
            return node.metric_value / niche_count

    def _build_crossover_gene_plan(
        self, parent_a: Node, parent_b: Node, crossover: Dict[str, str]
    ) -> Tuple[str, Dict[str, str]]:
        """从两亲本的基因按交叉计划构建 Markdown gene_plan 和 gene_sources。

        空基因处理：若 crossover 分配的亲本对该 locus 无基因，自动 fallback 到另一亲本。
        兼容性检查：调用 check_gene_compatibility() 检测框架冲突，有冲突时注入警告。

        Args:
            parent_a: 亲本 A
            parent_b: 亲本 B
            crossover: {locus: "A" | "B"} 交叉分配

        Returns:
            (gene_plan_md, gene_sources) 元组
        """
        lines: List[str] = []
        gene_sources: Dict[str, str] = {}
        genes_a = parent_a.genes or {}
        genes_b = parent_b.genes or {}

        for locus in GENE_LOCI:
            choice = crossover.get(locus, "A")
            # 获取基因代码
            if choice == "A":
                code = genes_a.get(locus, "")
                source_id = parent_a.id
                source_label = "A"
                # 空基因 fallback
                if not code or code.strip() in ("", "# (no code)"):
                    code = genes_b.get(locus, "")
                    source_id = parent_b.id
                    source_label = "B"
            else:
                code = genes_b.get(locus, "")
                source_id = parent_b.id
                source_label = "B"
                # 空基因 fallback
                if not code or code.strip() in ("", "# (no code)"):
                    code = genes_a.get(locus, "")
                    source_id = parent_a.id
                    source_label = "A"

            metric = (
                parent_a.metric_value if source_label == "A" else parent_b.metric_value
            )
            metric_str = f"{metric:.4f}" if metric is not None else "N/A"
            lines.append(
                f"### {locus} (from Parent {source_label} [{source_id[:8]}], fitness={metric_str})"
            )
            if code and code.strip():
                lines.append(f"```python\n{code}\n```\n")
            else:
                lines.append("```python\n# (no code available)\n```\n")
            gene_sources[locus] = source_id

        # 兼容性检查
        compat = check_gene_compatibility(
            parent_a.code or "",
            parent_b.code or "",
            genes_a,
            genes_b,
            crossover,
        )
        if compat.conflicts:
            lines.append("### ⚠️ Compatibility Warnings")
            for conflict in compat.conflicts:
                lines.append(f"- {conflict}")
            lines.append("")

        return "\n".join(lines), gene_sources

    def _run_merge_step(self) -> Optional[Node]:
        """执行一次 merge 操作（V7：两亲本交叉）。

        策略：
        1. 按物种分组种群
        2. 选最优物种（含最高分节点的物种）
        3. 从该物种内锦标赛选 parent_a 和 parent_b
        4. Uniform crossover：每个 locus 50% 概率来自 A 或 B
        5. 兼容性检查 + gene_plan 构建
        6. 调用 Orchestrator 执行 merge 任务

        Returns:
            生成的子代节点（失败时返回 None）
        """
        if not self.orchestrator:
            log_msg("WARNING", "Orchestrator 未初始化，跳过 merge")
            return None

        if len(self.population) < 2:
            log_msg("WARNING", f"种群过小（{len(self.population)} < 2），跳过 merge")
            return None

        # Phase 1: 按物种分组
        species_groups: Dict[str, List[Node]] = {}
        for node in self.population:
            sp = self._detect_species(node)
            species_groups.setdefault(sp, []).append(node)

        # Phase 2: 选最优物种（含最高分节点的物种）
        lower = self._is_lower_better()
        best_species = None
        best_metric = None
        for sp, nodes in species_groups.items():
            for n in nodes:
                if n.metric_value is None:
                    continue
                if best_metric is None:
                    best_metric = n.metric_value
                    best_species = sp
                elif lower and n.metric_value < best_metric:
                    best_metric = n.metric_value
                    best_species = sp
                elif not lower and n.metric_value > best_metric:
                    best_metric = n.metric_value
                    best_species = sp

        if best_species is None:
            log_msg("WARNING", "无法确定最优物种，跳过 merge")
            return None

        species_pool = species_groups[best_species]

        # Phase 3: 锦标赛选择两个亲本
        parent_a = self._tournament_select(pool=species_pool)

        # 选 parent_b（≠ parent_a）
        if len(species_pool) >= 2:
            # 从同物种中排除 parent_a 选择
            b_pool = [n for n in species_pool if n.id != parent_a.id]
            if not b_pool:
                b_pool = species_pool  # fallback
            parent_b = self._tournament_select(pool=b_pool)
        else:
            # 物种内节点数 < 2，fallback 到全种群选第二亲本
            b_pool = [n for n in self.population if n.id != parent_a.id]
            if not b_pool:
                log_msg("WARNING", "无法选择第二亲本，跳过 merge")
                return None
            parent_b = self._tournament_select(pool=b_pool)

        # Phase 4: Uniform crossover
        crossover: Dict[str, str] = {}
        for locus in GENE_LOCI:
            crossover[locus] = "A" if random.random() < 0.5 else "B"

        # Phase 5: 构建 gene_plan
        gene_plan_md, gene_sources = self._build_crossover_gene_plan(
            parent_a, parent_b, crossover
        )

        # 确定 primary_parent（贡献基因更多的亲本）
        a_count = sum(1 for v in gene_sources.values() if v == parent_a.id)
        b_count = sum(1 for v in gene_sources.values() if v == parent_b.id)
        if a_count >= b_count:
            primary_parent = parent_a
            secondary_parent = parent_b
        else:
            primary_parent = parent_b
            secondary_parent = parent_a

        log_msg(
            "INFO",
            f"Merge: parent_a={parent_a.id[:8]} ({self._detect_species(parent_a)}), "
            f"parent_b={parent_b.id[:8]} ({self._detect_species(parent_b)}), "
            f"crossover={crossover}",
        )

        return self.orchestrator.execute_merge_task(
            primary_parent,
            secondary_parent=secondary_parent,
            gene_plan=gene_plan_md,
            gene_sources=gene_sources,
        )

    def _run_mutate_step(self) -> Optional[Node]:
        """执行一次 mutate 操作（V7：适应度共享选择）。

        策略：
        1. 适应度共享锦标赛选父代（保护稀有物种）
        2. select_mutation_target() 选非 stub 基因块和子方面
        3. 调用 Orchestrator 执行 mutate 任务

        Returns:
            变异后的节点（失败时返回 None）
        """
        if not self.orchestrator:
            log_msg("WARNING", "Orchestrator 未初始化，跳过 mutate")
            return None

        if not self.population:
            log_msg("WARNING", "种群为空，跳过 mutate")
            return None

        parent = self._tournament_select(use_shared_fitness=True)
        target_gene, mutation_aspect = select_mutation_target(parent)

        log_msg(
            "INFO",
            f"Mutate: parent={parent.id[:8]} ({self._detect_species(parent)}), "
            f"gene={target_gene}, aspect={mutation_aspect}",
        )
        return self.orchestrator.execute_mutate_task(
            parent, target_gene, mutation_aspect
        )
