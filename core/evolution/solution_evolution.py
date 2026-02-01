"""Solution 层遗传算法实现。

实现种群初始化、精英保留、锦标赛选择、基因交叉、基因变异等遗传算法核心操作。
"""

from __future__ import annotations

import random
from typing import List, Tuple, Dict, Any, Optional

from agents.base_agent import BaseAgent, AgentContext, AgentResult
from core.state import Node, Journal
from core.evolution.task_dispatcher import TaskDispatcher
from core.evolution.experience_pool import ExperiencePool, TaskRecord
from core.evolution.gene_parser import parse_solution_genes, REQUIRED_GENES
from core.evolution.gene_registry import GeneRegistry
from core.evolution.gene_selector import select_gene_plan
from search.parallel_evaluator import ParallelEvaluator
from search.fitness import normalize_fitness
from utils.config import Config
from utils.logger_system import log_msg, log_json


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
    ):
        """初始化遗传算法（MVP 简化版）。

        Args:
            config: 全局配置
            journal: 节点历史记录
            orchestrator: Orchestrator 实例（用于执行任务）
        """
        self.config = config
        self.journal = journal
        self.orchestrator = orchestrator

        # GA 参数
        self.population_size = config.evolution.solution.population_size
        self.elite_size = config.evolution.solution.elite_size
        self.crossover_rate = config.evolution.solution.crossover_rate
        self.mutation_rate = config.evolution.solution.mutation_rate
        self.tournament_k = config.evolution.solution.tournament_k

        # 当前种群
        self.population: List[Node] = []

        log_msg(
            "INFO",
            f"SolutionEvolution 初始化（MVP）: population_size={self.population_size}",
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
        log_msg("INFO", f"===== SolutionEvolution: run_epoch 开始 =====")

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

    def initialize_population(self, start_time: float, task_desc: str) -> None:
        """初始化种群（并行生成初始 Solution）。

        Args:
            start_time: 任务开始时间
            task_desc: 任务描述
        """
        log_msg("INFO", f"开始初始化种群: {self.population_size} 个 Solution")

        # 创建 explore 任务列表
        tasks = []
        for i in range(self.population_size):
            agent = self.task_dispatcher.select_agent("explore")
            context = AgentContext(
                task_type="explore",
                parent_node=None,  # 从零生成
                journal=self.journal,
                config=self.config,
                start_time=start_time,
                current_step=self.current_step,
                task_desc=task_desc,
            )
            tasks.append((agent, context))

        # 并行生成
        results = self.evaluator.batch_generate(tasks)

        # 收集成功生成的节点
        self.population = []
        for result in results:
            if result.success and result.node is not None:
                # 解析基因
                result.node.genes = parse_solution_genes(result.node.code)
                self.population.append(result.node)
                self.journal.append(result.node)

                # 记录到经验池
                self._record_to_experience_pool(
                    agent_id=self.task_dispatcher.select_agent("explore").name,
                    task_type="explore",
                    node=result.node,
                )

        # 并行评估种群
        self.evaluator.batch_evaluate(self.population, self.current_step)

        success_count = len(self.population)
        log_msg("INFO", f"种群初始化完成: {success_count}/{self.population_size} 成功")

        if success_count < self.population_size:
            log_msg(
                "WARNING",
                f"初始化失败 {self.population_size - success_count} 个节点",
            )

    def step(self, start_time: float, task_desc: str) -> None:
        """单步进化。

        执行精英保留 + 选择 + 交叉 + 变异 + 评估 + 截断。

        Args:
            start_time: 任务开始时间
            task_desc: 任务描述
        """
        self.current_step += 1
        log_msg("INFO", f"===== Step {self.current_step}: 开始进化 =====")

        # [1] 精英保留
        elites = self._select_elites()
        log_msg("INFO", f"精英保留: {len(elites)} 个节点")

        # [2] 锦标赛选择父代对
        num_offspring = self.population_size - self.elite_size
        parent_pairs = self._select_parent_pairs(num_offspring)
        log_msg("INFO", f"选择父代: {len(parent_pairs)} 对")

        # [3] 基因交叉
        offspring = []
        for parent_a, parent_b in parent_pairs:
            if random.random() < self.crossover_rate:
                child = self._crossover(parent_a, parent_b, start_time, task_desc)
                if child is not None:
                    offspring.append(child)
            else:
                # 不交叉，随机选择一个父代作为子代
                offspring.append(random.choice([parent_a, parent_b]))

        log_msg("INFO", f"交叉完成: {len(offspring)} 个子代")

        # [4] 基因变异
        mutation_count = 0
        for node in offspring:
            if random.random() < self.mutation_rate:
                self._mutate(node, start_time, task_desc)
                mutation_count += 1

        log_msg("INFO", f"变异完成: {mutation_count} 个节点")

        # [5] 并行评估新个体
        self.evaluator.batch_evaluate(offspring, self.current_step)

        # [6] 合并精英 + 新个体，截断到种群大小
        combined = elites + offspring
        combined.sort(key=lambda n: n.metric_value or -1e9, reverse=True)
        self.population = combined[: self.population_size]

        # 输出统计
        best_metric = self.population[0].metric_value if self.population else None
        avg_metric = (
            sum(n.metric_value or 0 for n in self.population) / len(self.population)
            if self.population
            else 0
        )

        log_json(
            {
                "step": self.current_step,
                "best_metric": best_metric,
                "avg_metric": avg_metric,
                "elite_count": len(elites),
                "offspring_count": len(offspring),
                "mutation_count": mutation_count,
            }
        )

        log_msg("INFO", f"===== Step {self.current_step}: 进化完成 =====")

    def _select_elites(self) -> List[Node]:
        """精英保留（返回 top-K 节点）。

        Returns:
            精英节点列表
        """
        # 按 metric_value 降序排序
        sorted_pop = sorted(
            self.population, key=lambda n: n.metric_value or -1e9, reverse=True
        )
        return sorted_pop[: self.elite_size]

    def _select_parent_pairs(self, num_pairs: int) -> List[Tuple[Node, Node]]:
        """锦标赛选择父代对。

        Args:
            num_pairs: 需要选择的父代对数量

        Returns:
            父代对列表
        """
        pairs = []
        for _ in range(num_pairs):
            parent_a = self._tournament_select()
            parent_b = self._tournament_select()
            pairs.append((parent_a, parent_b))
        return pairs

    def _tournament_select(self) -> Node:
        """锦标赛选择单个父代。

        随机抽取 k 个个体，返回其中最优者。

        Returns:
            选中的父代节点
        """
        tournament = random.sample(self.population, k=self.tournament_k)
        winner = max(tournament, key=lambda n: n.metric_value or -1e9)
        return winner

    def _crossover(
        self, parent_a: Node, parent_b: Node, start_time: float, task_desc: str
    ) -> Optional[Node]:
        """基因交叉（调用 merge Agent）。

        Args:
            parent_a: 父代 A
            parent_b: 父代 B
            start_time: 任务开始时间
            task_desc: 任务描述

        Returns:
            子代节点（失败时返回 None）
        """
        # 生成交叉计划
        gene_plan = self._generate_crossover_plan(parent_a, parent_b)

        # 选择 merge Agent
        agent = self.task_dispatcher.select_agent("merge")

        # 构建上下文
        context = AgentContext(
            task_type="merge",
            parent_node=parent_a,
            journal=self.journal,
            config=self.config,
            start_time=start_time,
            current_step=self.current_step,
            task_desc=task_desc,
        )

        # 将交叉计划传递给 Agent（通过 metadata）
        # 注意：这需要 Agent 实现支持从 metadata 读取 gene_plan
        # 暂时简化实现，直接调用 Agent 生成
        result = agent.generate(context)

        if result.success and result.node is not None:
            # 解析基因
            result.node.genes = parse_solution_genes(result.node.code)
            self.journal.append(result.node)

            # 记录到经验池
            self._record_to_experience_pool(agent.name, "merge", result.node)

            return result.node
        else:
            log_msg("WARNING", f"交叉失败: {result.error}")
            return None

    def _mutate(self, node: Node, start_time: float, task_desc: str) -> None:
        """基因变异（调用 mutate Agent，原地修改）。

        Args:
            node: 待变异的节点
            start_time: 任务开始时间
            task_desc: 任务描述
        """
        # 随机选择一个基因块进行变异
        target_gene = random.choice(REQUIRED_GENES)

        # 选择 mutate Agent
        agent = self.task_dispatcher.select_agent("mutate")

        # 构建上下文
        context = AgentContext(
            task_type="mutate",
            parent_node=node,
            journal=self.journal,
            config=self.config,
            start_time=start_time,
            current_step=self.current_step,
            task_desc=task_desc,
        )

        # 调用 Agent 生成变异后的代码
        # 注意：需要将 target_gene 传递给 Agent（通过 metadata）
        result = agent.generate(context)

        if result.success and result.node is not None:
            # 原地更新节点
            node.code = result.node.code
            node.plan = result.node.plan
            node.genes = parse_solution_genes(result.node.code)

            # 记录到经验池
            self._record_to_experience_pool(agent.name, "mutate", node)
        else:
            log_msg("WARNING", f"变异失败: {result.error}")

    def _generate_crossover_plan(
        self, parent_a: Node, parent_b: Node
    ) -> Dict[str, Any]:
        """生成交叉计划（根据配置选择策略）。

        Args:
            parent_a: 父代 A
            parent_b: 父代 B

        Returns:
            交叉计划字典
        """
        if self.crossover_strategy == "random":
            return self._generate_random_plan()
        elif self.crossover_strategy == "pheromone":
            return self._generate_pheromone_plan(parent_a, parent_b)
        else:
            raise ValueError(f"未知的交叉策略: {self.crossover_strategy}")

    def _generate_random_plan(self) -> Dict[str, str]:
        """生成随机交叉计划（每个基因 50% 概率选 A 或 B）。

        Returns:
            交叉计划字典 {gene: "A" or "B"}
        """
        gene_plan = {}
        for gene in REQUIRED_GENES:
            gene_plan[gene] = random.choice(["A", "B"])

        log_msg("INFO", f"随机交叉计划: {gene_plan}")
        return gene_plan

    def _generate_pheromone_plan(
        self, parent_a: Node, parent_b: Node
    ) -> Dict[str, Any]:
        """生成信息素驱动交叉计划（从 Journal 选择质量最高的基因）。

        Args:
            parent_a: 父代 A
            parent_b: 父代 B

        Returns:
            交叉计划字典
        """
        # 调用基因选择器
        gene_plan = select_gene_plan(
            journal=self.journal,
            gene_registry=self.gene_registry,
            current_step=self.current_step,
        )

        log_msg("INFO", f"信息素驱动交叉计划: 选择了 {len(gene_plan)} 个基因位点")
        return gene_plan

    def _record_to_experience_pool(
        self, agent_id: str, task_type: str, node: Node
    ) -> None:
        """记录执行结果到经验池。

        Args:
            agent_id: Agent ID
            task_type: 任务类型
            node: 生成的节点
        """
        import hashlib
        import time

        # 计算输入哈希
        input_hash = hashlib.sha256(
            f"{task_type}_{self.current_step}".encode()
        ).hexdigest()[:16]

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

    def _crossover_mvp(self, parent_a: Node, parent_b: Node) -> Optional[Node]:
        """基因交叉（MVP 简化版，调用 Orchestrator）。

        Args:
            parent_a: 父代 A
            parent_b: 父代 B

        Returns:
            子代节点（失败时返回 None）
        """
        if not self.orchestrator:
            log_msg("WARNING", "Orchestrator 未初始化，跳过交叉")
            return None

        # 生成随机交叉计划（MVP: 50% 概率选 A 或 B）
        gene_plan = {}
        for gene in REQUIRED_GENES:
            gene_plan[gene] = random.choice(["A", "B"])

        log_msg("INFO", f"交叉计划: {gene_plan}")

        # 调用 Orchestrator 执行 merge 任务
        child = self.orchestrator.execute_merge_task(parent_a, parent_b, gene_plan)

        return child

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
