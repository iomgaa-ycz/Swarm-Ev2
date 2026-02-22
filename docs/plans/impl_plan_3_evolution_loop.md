# 实施计划 P3：进化主循环层

**范围**: `solution_evolution.py` 完整重写（Phase 2 逻辑）+ `main.py` 主循环改造。
**依赖**: P1、P2 均完成后才能执行
**估计改动量**: 2 个文件，约 120 行修改（含删除）

---

## 3.1 `core/evolution/solution_evolution.py` [MODIFY - 大幅重写]

### 修改位置 1：文件顶部 import（第 12–16 行）

**旧：**
```python
from core.state import Node, Journal
from core.evolution.gene_parser import REQUIRED_GENES
from core.evolution.gene_registry import GeneRegistry
from core.evolution.gene_selector import select_gene_plan
from utils.config import Config
```
**新：**
```python
from core.state import Node, Journal
from core.evolution.gene_parser import REQUIRED_GENES, select_non_stub_gene
from core.evolution.gene_registry import GeneRegistry
from core.evolution.gene_selector import (
    pheromone_with_degenerate_check,
    get_primary_parent,
    LOCUS_TO_FIELD,
)
from utils.config import Config
```

注意：删除 `from core.evolution.gene_selector import select_gene_plan`（新架构不直接调用，而是通过 `pheromone_with_degenerate_check` 调用）。

### 修改位置 2：`__init__()` 更新（第 42–64 行）

在 `self.ga_trigger_threshold` 赋值之后追加：
```python
        # 两阶段进化参数（P1 新增）
        self.phase1_target_nodes = getattr(
            config.evolution.solution, "phase1_target_nodes", 8
        )
```

### 修改位置 3：完整重写 `run_epoch()`（第 66–158 行）

**删除整个旧 `run_epoch()`，替换为：**

```python
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

        # 获取 valid_pool（非 buggy，非 dead）
        valid_pool = [
            n for n in self.journal.nodes if not n.is_buggy and not n.dead
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
```

### 修改位置 4：删除旧方法，新增 `_run_merge_step()` 和 `_run_mutate_step()`

#### 删除的方法（直接删除整个函数体）

- `_select_elites()` (第 182–206 行) — 精英保留不再显式执行
- `_crossover_mvp()` (第 234–273 行) — 被 `_run_merge_step()` 取代
- `_inject_compatibility_warnings()` (第 275–337 行) — 不再需要
- `_build_gene_plan_markdown_from_random()` (第 363–395 行) — 不再使用随机策略

#### 修改的方法

**`_is_lower_better()` 和 `_tournament_select()` 保留不变**（用于 mutate 父代选择）。

**`_build_gene_plan_markdown_from_pheromone()`（第 339–361 行）追加 gene_sources 返回**

由于新架构需要同时返回 `gene_plan_md` 和 `gene_sources`，修改返回值：

**旧：**
```python
    def _build_gene_plan_markdown_from_pheromone(self, raw_plan: Dict[str, Any]) -> str:
        ...
        return "\n".join(lines)
```
**新：**
```python
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
```

注意：需要在文件顶部 import 中添加 `Tuple`：
```python
from typing import List, Dict, Any, Optional, Tuple
```

**`_mutate_mvp()` 重命名为 `_run_mutate_step()` 并简化**

删除整个旧 `_mutate_mvp()` (第 397–418 行)，替换为：

```python
    def _run_mutate_step(self) -> Optional[Node]:
        """执行一次 mutate 操作（Phase 2 内部使用）。

        策略：
        1. Tournament 选父代（质量最优）
        2. select_non_stub_gene() 选非 stub 基因块
        3. 调用 Orchestrator 执行 mutate 任务

        Returns:
            变异后的节点（失败时返回 None）
        """
        if not self.orchestrator:
            log_msg("WARNING", "Orchestrator 未初始化，跳过 mutate")
            return None

        if len(self.population) < self.tournament_k:
            log_msg("WARNING", f"种群过小（{len(self.population)} < {self.tournament_k}），跳过 mutate")
            return None

        parent = self._tournament_select()
        target_gene = select_non_stub_gene(parent)

        log_msg("INFO", f"Mutate: parent={parent.id[:8]}, gene={target_gene}")
        return self.orchestrator.execute_mutate_task(parent, target_gene)
```

**新增 `_run_merge_step()`**

在 `_run_mutate_step()` 之前插入：

```python
    def _run_merge_step(self) -> Optional[Node]:
        """执行一次 merge 操作（Phase 2 内部使用）。

        策略：
        1. pheromone_with_degenerate_check() 选 7 个全局最优基因
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
```

---

## 3.2 `main.py` [MODIFY]

### 修改位置 1：Phase 4 主循环（第 471–538 行）

**删除旧的 Phase 4 循环（第 471–538 行），替换为：**

```python
        # ============================================================
        # Phase 4: 两阶段进化主循环
        # ============================================================
        print("\n[4/6] 运行两阶段进化主循环...")

        total_budget = config.agent.max_steps
        phase1_budget = max(
            config.evolution.solution.phase1_target_nodes * 3,  # 3x 目标节点数作为预算
            total_budget // 2,                                   # 至多使用一半预算
        )
        phase2_budget = total_budget - phase1_budget
        steps_per_epoch = config.evolution.solution.steps_per_epoch
        num_epochs = max(1, phase2_budget // steps_per_epoch)

        print(f"  Phase 1 Draft 预算: {phase1_budget} 步")
        print(f"  Phase 2 Evolution 预算: {phase2_budget} 步 ({num_epochs} epochs)")
        print(f"  Phase 1 目标 valid_pool: {config.evolution.solution.phase1_target_nodes}")
        print("")

        # --- Phase 1: Draft（纯探索，无父代）---
        log_msg("INFO", "===== 开始 Phase 1: Draft 模式 =====")
        orchestrator.run_epoch_draft(phase1_budget)

        # --- Phase 2: 进化（merge + mutate）---
        log_msg("INFO", "===== 开始 Phase 2: 进化模式（merge + mutate）=====")
        best_node = None

        for epoch in range(num_epochs):
            if orchestrator._check_time_limit():
                log_msg("INFO", "时间限制已达，停止 Phase 2 进化")
                break

            log_msg("INFO", f"===== Phase 2 Epoch {epoch + 1}/{num_epochs} =====")
            epoch_best = solution_evolution.run_epoch(steps_per_epoch)

            if epoch_best and epoch_best.metric_value is not None:
                if best_node is None or best_node.metric_value is None:
                    best_node = epoch_best
                else:
                    lower = orchestrator._global_lower_is_better or False
                    is_better = (
                        epoch_best.metric_value < best_node.metric_value
                        if lower
                        else epoch_best.metric_value > best_node.metric_value
                    )
                    if is_better:
                        best_node = epoch_best

            # Agent 层进化（每 3 Epoch）
            if agent_evolution and (epoch + 1) % 3 == 0:
                log_msg("INFO", "触发 Agent 层进化")
                agent_evolution.evolve(epoch)

            current_best = journal.get_best_node(
                lower_is_better=orchestrator._global_lower_is_better
            )
            log_msg(
                "INFO",
                f"Phase 2 Epoch {epoch + 1}/{num_epochs} 完成 | "
                f"最佳 metric: {current_best.metric_value if current_best else 'N/A'}",
            )

        best_node = journal.get_best_node(
            only_good=True, lower_is_better=orchestrator._global_lower_is_better
        )
        log_msg(
            "INFO",
            f"两阶段进化完成: best_node={'存在' if best_node else '不存在'}",
        )
```

### 修改位置 2：`generate_markdown_report()` 表头更新（第 208–214 行）

**旧：**
```python
| Agent | explore | merge | mutate |
|-------|---------|-------|--------|
"""

    scores = task_dispatcher.get_specialization_matrix()
    for agent_id, task_scores in scores.items():
        content += f"| {agent_id} | {task_scores['explore']:.3f} | {task_scores['merge']:.3f} | {task_scores['mutate']:.3f} |\n"
```
**新：**
```python
| Agent | draft | merge | mutate |
|-------|-------|-------|--------|
"""

    scores = task_dispatcher.get_specialization_matrix()
    for agent_id, task_scores in scores.items():
        content += (
            f"| {agent_id} "
            f"| {task_scores.get('draft', task_scores.get('explore', 0)):.3f} "
            f"| {task_scores.get('merge', 0):.3f} "
            f"| {task_scores.get('mutate', 0):.3f} |\n"
        )
```

注意：`task_scores.get('draft', task_scores.get('explore', 0))` 向后兼容——优先取 "draft" 分数，若无则取 "explore" 分数。

### 修改位置 3：Phase 4 控制台打印（第 302–306 行）

**旧：**
```python
        print(
            f"  {agent_id}: explore={task_scores['explore']:.3f}, merge={task_scores['merge']:.3f}, mutate={task_scores['mutate']:.3f}"
        )
```
**新：**
```python
        print(
            f"  {agent_id}: "
            f"draft={task_scores.get('draft', task_scores.get('explore', 0)):.3f}, "
            f"merge={task_scores.get('merge', 0):.3f}, "
            f"mutate={task_scores.get('mutate', 0):.3f}"
        )
```

### 修改位置 4：Phase 1 控制台打印更新（第 465–469 行）

**旧：**
```python
        print("\n📋 配置摘要:")
        print(f"  Agent 数量: {config.evolution.agent.num_agents}")
        print(f"  每 Epoch 步数: {config.evolution.solution.steps_per_epoch}")
        print(f"  探索率: {config.evolution.agent.epsilon}")
```
**新（追加 Phase 1 配置）：**
```python
        print("\n📋 配置摘要:")
        print(f"  Agent 数量: {config.evolution.agent.num_agents}")
        print(f"  每 Epoch 步数: {config.evolution.solution.steps_per_epoch}")
        print(f"  探索率: {config.evolution.agent.epsilon}")
        print(f"  Phase 1 目标节点: {config.evolution.solution.phase1_target_nodes}")
        print(f"  Debug 最大次数: {config.evolution.solution.debug_max_attempts}")
```

---

## 3.3 验证步骤

```bash
# 1. solution_evolution 导入与方法验证
conda run -n Swarm-Evo python -c "
from core.evolution.solution_evolution import SolutionEvolution
import inspect
src = inspect.getsource(SolutionEvolution)
assert '_run_merge_step' in src, '_run_merge_step 缺失'
assert '_run_mutate_step' in src, '_run_mutate_step 缺失'
assert '_crossover_mvp' not in src, '_crossover_mvp 应已删除'
assert '_select_elites' not in src, '_select_elites 应已删除'
assert 'pheromone_with_degenerate_check' in src, 'pheromone 调用缺失'
assert 'select_non_stub_gene' in src, 'select_non_stub_gene 调用缺失'
print('solution_evolution 重写 OK')
"

# 2. main.py 导入验证
conda run -n Swarm-Evo python -c "
import ast
src = open('main.py').read()
tree = ast.parse(src)
print('main.py 语法解析 OK')
"

# 3. Phase 1 配置打印验证（需要 default.yaml 有 phase1_target_nodes）
conda run -n Swarm-Evo python -c "
from utils.config import load_config
c = load_config('config/default.yaml', use_cli=False)
assert hasattr(c.evolution.solution, 'phase1_target_nodes'), 'phase1_target_nodes 缺失'
assert hasattr(c.evolution.solution, 'debug_max_attempts'), 'debug_max_attempts 缺失'
print('配置验证 OK')
print('phase1_target_nodes:', c.evolution.solution.phase1_target_nodes)
print('debug_max_attempts:', c.evolution.solution.debug_max_attempts)
"

# 4. 完整导入链验证
conda run -n Swarm-Evo python -c "
from core.evolution.solution_evolution import SolutionEvolution
from core.orchestrator import Orchestrator
from agents.base_agent import AgentContext
print('导入链 OK')
"

# 5. gene_sources 字典格式验证
conda run -n Swarm-Evo python -c "
from core.evolution.gene_selector import LOCUS_TO_FIELD
print('LOCUS_TO_FIELD keys:', list(LOCUS_TO_FIELD.keys()))
# 验证 gene_sources 字典应包含 7 个 locus
assert len(LOCUS_TO_FIELD) == 7, f'期望 7 个 locus，实际 {len(LOCUS_TO_FIELD)}'
print('LOCUS_TO_FIELD OK')
"
```

---

## 3.4 变更摘要对照

| 旧方法/逻辑 | 新方法/逻辑 | 说明 |
|-------------|-------------|------|
| `_crossover_mvp()` + Tournament | `_run_merge_step()` + pheromone | merge 不再用 Tournament |
| `_mutate_mvp()` + random.choice(REQUIRED_GENES) | `_run_mutate_step()` + `select_non_stub_gene()` | mutate 跳过 stub 基因 |
| `_select_elites()` | 无（implicit） | Phase 2 直接使用 valid_pool |
| `_inject_compatibility_warnings()` | 无 | pheromone 策略不需要兼容性检测 |
| `_build_gene_plan_markdown_from_random()` | 无 | 随机策略已废弃 |
| `select_gene_plan` 直接调用 | `pheromone_with_degenerate_check` | 增加退化检测 |
| main.py: 单阶段循环（explore epoch） | 两阶段（Phase 1 draft + Phase 2 evolve） | 新架构主循环 |

## 3.5 注意事项

### Phase 1 预算计算

`phase1_budget = max(phase1_target * 3, total_budget // 2)` 是一个启发式默认值。
当 `phase1_target_nodes = 8`，`max_steps = 30` 时，`phase1_budget = max(24, 15) = 24`，留给 Phase 2 只有 6 步（仅够 1 个 epoch）。
如需调整，可在 `config.yaml` 中增加 `phase1_budget` 字段，或在调试时手动设置 `max_steps`。

### `_build_gene_plan_markdown_from_pheromone()` 返回值改变

从返回 `str` 改为返回 `Tuple[str, Dict[str, str]]`，所有调用方需同步更新。
目前调用方只有 `_run_merge_step()`（P3 新增），以及旧的 `_crossover_mvp()`（P3 已删除），无遗留调用方。

### journal.get_node() 存在性

`get_primary_parent()` 在 P1 计划中已说明：若 `journal.get_node()` 不存在，需使用 `next((n for n in journal.nodes if n.id == node_id), None)` 替代。P3 执行前请确认 P1 的 gene_selector.py 中此问题已处理。
