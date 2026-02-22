# 实施计划 P1：基础数据层

**范围**: 纯增量变更——数据结构、配置、工具函数。无架构改动，不破坏现有接口。
**依赖**: 无（可独立完成）
**估计改动量**: 6 个文件，约 60 行新增

---

## 1.1 `core/state/node.py` [MODIFY]

### 修改位置：第 85–88 行（GA 字段块末尾）

**现有**（第 85-88 行）：
```python
    # ---- GA ----
    generation: Optional[int] = field(default=None, kw_only=True)
    fitness: Optional[float] = field(default=None, kw_only=True)
```

**改为**（在 fitness 字段后追加 3 个字段）：
```python
    # ---- GA ----
    generation: Optional[int] = field(default=None, kw_only=True)
    fitness: Optional[float] = field(default=None, kw_only=True)

    # ---- 两阶段进化 ----
    dead: bool = field(default=False, kw_only=True)            # 1主操作+2次debug全失败
    debug_attempts: int = field(default=0, kw_only=True)       # 已执行的 debug_chain 次数
    approach_tag: Optional[str] = field(default=None, kw_only=True)  # Review 写入，用于 Phase 1 多样性引导
```

### 同步修改：Docstring（第 27–45 行）

在 Attributes 列表末尾（`fitness` 条目之后）追加：
```
        dead: 是否为死节点（1 次主操作 + 2 次 debug_chain 全失败）
        debug_attempts: 已执行的 debug_chain 次数
        approach_tag: Review 写入的方法摘要（如 "LightGBM + 5-fold CV"），用于 Phase 1 多样性引导
```

### 同步修改：task_type 默认值（第 58 行）

```python
# 旧
task_type: str = field(default="explore", kw_only=True)
# 新
task_type: str = field(default="draft", kw_only=True)
```

---

## 1.2 `utils/config.py` [MODIFY]

### 修改位置：SolutionEvolutionConfig（第 122–132 行）

**追加 2 个字段**（在 `crossover_strategy` 之后）：
```python
@dataclass
class SolutionEvolutionConfig:
    """Solution 层遗传算法配置。"""
    population_size: int
    ga_trigger_threshold: int
    elite_size: int
    crossover_rate: float
    mutation_rate: float
    tournament_k: int
    steps_per_epoch: int
    crossover_strategy: str = "random"
    # ---- 两阶段进化新增 ----
    phase1_target_nodes: int = 8       # Phase 1 结束条件：valid_pool 达到此数量
    debug_max_attempts: int = 2        # debug_chain 最大次数（两阶段共用）
```

### 修改位置：validate_config schema（第 432–441 行）

在 `SolutionEvolutionConfig(...)` 构造调用中追加两个字段：
```python
solution=SolutionEvolutionConfig(
    population_size=0,
    ga_trigger_threshold=0,
    elite_size=0,
    crossover_rate=0.0,
    mutation_rate=0.0,
    tournament_k=0,
    steps_per_epoch=0,
    crossover_strategy="random",
    phase1_target_nodes=8,      # ← 新增
    debug_max_attempts=2,       # ← 新增
),
```

---

## 1.3 `config/default.yaml` [MODIFY]

### 修改位置：第 103–111 行（`evolution.solution` 块）

**追加 2 行**（在 `crossover_strategy` 之后）：
```yaml
  solution:
    population_size: 12
    ga_trigger_threshold: 4
    elite_size: 3
    crossover_rate: 0.8
    mutation_rate: 0.2
    tournament_k: 3
    steps_per_epoch: 10
    crossover_strategy: "random"
    phase1_target_nodes: 8       # Phase 1 目标：valid_pool 达此数量
    debug_max_attempts: 2        # debug_chain 最大次数
```

---

## 1.4 `config/mle_bench.yaml` [MODIFY]

### 修改位置：第 71–79 行（`evolution.solution` 块）

**追加 2 行**（在 `crossover_strategy` 之后）：
```yaml
  solution:
    population_size: 12
    ga_trigger_threshold: 4
    elite_size: 3
    crossover_rate: 0.8
    mutation_rate: 0.2
    tournament_k: 3
    steps_per_epoch: 10
    crossover_strategy: "random"
    phase1_target_nodes: 8       # Phase 1 目标
    debug_max_attempts: 2        # debug_chain 最大次数
```

---

## 1.5 `core/evolution/gene_parser.py` [MODIFY]

### 在文件末尾追加新函数（第 163 行之后）

```python
def select_non_stub_gene(node: "Node", stub_threshold: int = 20) -> str:
    """从节点中随机选择一个非 stub 的基因位点（用于 mutate 目标选择）。

    内容 strip 后长度 < stub_threshold 视为 stub（如 "# Not applicable..."）。
    对表格任务，LOSS/OPTIMIZER/INITIALIZATION 通常是 stub，自动跳过。

    Args:
        node: 待变异的节点
        stub_threshold: 内容长度阈值，低于此值视为 stub（默认 20）

    Returns:
        随机选中的非 stub 基因位点名称；若所有基因均为 stub，则 fallback 到随机选择
    """
    candidates = [
        gene for gene in REQUIRED_GENES
        if len((node.genes.get(gene) or "").strip()) >= stub_threshold
    ]
    if candidates:
        return random.choice(candidates)
    # fallback: 所有基因均为 stub（极端情况），随机选一个
    return random.choice(REQUIRED_GENES)
```

### 同步修改：文件顶部添加 import（第 6 行）

```python
import random   # ← 新增（原文件无此 import）
from typing import Dict, TYPE_CHECKING   # 修改 typing import

if TYPE_CHECKING:
    from core.state.node import Node
```

注意：`from typing import Dict` 原文件已有，只需追加 `TYPE_CHECKING` 和条件 import。

---

## 1.6 `core/evolution/gene_selector.py` [MODIFY]

### 在文件末尾追加两个新函数（第 323 行之后）

```python
# ══════════════════════════════════════════════════════
# Pheromone + 退化检测（新架构 Phase 2 Merge 使用）
# ══════════════════════════════════════════════════════

from collections import Counter  # 添加到文件顶部 import


def pheromone_with_degenerate_check(
    journal: Journal,
    gene_registry: GeneRegistry,
    current_step: int,
) -> Dict[str, Any]:
    """Pheromone 全局 TOP-1 选择 + 退化检测。

    若 7 个基因全来自同一节点（退化），则将每个位点替换为
    来自其他节点的次优基因（second-best），以避免 merge 退化为复制。

    Args:
        journal: Journal 对象
        gene_registry: 基因注册表
        current_step: 当前步骤

    Returns:
        基因计划字典（与 select_gene_plan 格式兼容）
    """
    # Step 1: 标准 pheromone 全局 TOP-1（不传父代，全局视野）
    gene_plan = select_gene_plan(journal, gene_registry, current_step)

    # Step 2: 退化检测
    source_ids = {
        v["source_node_id"]
        for v in gene_plan.values()
        if isinstance(v, dict) and "source_node_id" in v
    }

    if len(source_ids) != 1:
        return gene_plan  # 来源多样，无退化

    dominant_id = next(iter(source_ids))
    log_msg("WARNING", f"[DEGENERATE] 7 基因全来自节点 {dominant_id[:8]}，启动退化检测替换")

    pools = build_decision_gene_pools(journal, gene_registry, current_step)

    for locus, field_name in LOCUS_TO_FIELD.items():
        others = [
            item for item in pools.get(locus, [])
            if item.source_node_id != dominant_id
        ]
        if others:
            second_best = max(others, key=lambda x: x.quality)
            gene_plan[field_name] = {
                "locus": locus,
                "source_node_id": second_best.source_node_id,
                "gene_id": second_best.gene_id,
                "code": second_best.raw_content,
                "source_score": second_best.source_score,
            }
            log_msg(
                "INFO",
                f"[DEGENERATE FIX] {locus}: 替换为 {second_best.source_node_id[:8]} "
                f"(quality={second_best.quality:.4f})",
            )
        # 若无其他来源候选，保留原 dominant（不强行替换）

    return gene_plan


def get_primary_parent(gene_plan: Dict[str, Any], journal: Journal) -> "Node":
    """从基因计划中推断主父代（贡献基因数最多的节点）。

    用于 merge.j2 的单一参考父代。选择规则：
    1. 统计每个 source_node_id 贡献的基因数量
    2. 取贡献最多的节点
    3. 并列时选 metric_value 最高的节点

    Args:
        gene_plan: pheromone_with_degenerate_check() 返回的基因计划
        journal: Journal 对象（用于查找节点）

    Returns:
        贡献基因最多的 Node 对象

    Raises:
        ValueError: gene_plan 为空或 journal 中找不到对应节点
    """
    # 统计各 source_node_id 出现次数
    source_counts = Counter(
        v["source_node_id"]
        for v in gene_plan.values()
        if isinstance(v, dict) and "source_node_id" in v
    )

    if not source_counts:
        raise ValueError("gene_plan 中无有效的 source_node_id")

    max_count = max(source_counts.values())
    candidates = [nid for nid, cnt in source_counts.items() if cnt == max_count]

    # 快速路径：唯一候选
    if len(candidates) == 1:
        node = journal.get_node(candidates[0])
        if node is None:
            raise ValueError(f"Journal 中找不到节点: {candidates[0][:8]}")
        return node

    # 并列时选 metric 最高的节点
    best_id = None
    best_metric = float("-inf")
    for nid in candidates:
        node = journal.get_node(nid)
        if node is None:
            continue
        metric = node.metric_value or 0.0
        if metric > best_metric:
            best_metric = metric
            best_id = nid

    if best_id is None:
        # fallback: 取 gene_plan 中第一个出现的 source_node_id
        for v in gene_plan.values():
            if isinstance(v, dict) and "source_node_id" in v:
                best_id = v["source_node_id"]
                break

    node = journal.get_node(best_id)
    if node is None:
        raise ValueError(f"Journal 中找不到节点: {best_id[:8]}")
    return node
```

### 同步修改：文件顶部 import（第 15–24 行区域）

在现有 import 后追加：
```python
from collections import Counter   # ← 新增
```

---

## 1.7 验证步骤

```bash
# 1. 配置字段加载验证
conda run -n Swarm-Evo python -c "
from utils.config import load_config
c = load_config('config/default.yaml', use_cli=False)
print('phase1_target_nodes:', c.evolution.solution.phase1_target_nodes)
print('debug_max_attempts:', c.evolution.solution.debug_max_attempts)
"

# 2. Node 新字段验证
conda run -n Swarm-Evo python -c "
from core.state.node import Node
n = Node(code='test')
print('dead:', n.dead)
print('debug_attempts:', n.debug_attempts)
print('approach_tag:', n.approach_tag)
"

# 3. select_non_stub_gene 验证
conda run -n Swarm-Evo python -c "
from core.state.node import Node
from core.evolution.gene_parser import select_non_stub_gene, REQUIRED_GENES
n = Node(code='test')
n.genes = {g: '# Not applicable' for g in REQUIRED_GENES}
n.genes['DATA'] = 'import pandas as pd\ndf = pd.read_csv(\"./input/train.csv\")'
result = select_non_stub_gene(n)
assert result == 'DATA', f'Expected DATA, got {result}'
print('select_non_stub_gene OK:', result)
"

# 4. 单元测试
conda run -n Swarm-Evo pytest tests/unit/test_gene_parser.py -v -k stub 2>/dev/null || echo "测试文件待创建"
conda run -n Swarm-Evo pytest tests/unit/test_gene_selector.py -v -k "degenerate or primary" 2>/dev/null || echo "测试文件待创建"
```

---

## 1.8 注意事项

- **`journal.get_node(node_id)`** 方法需要确认存在于 `core/state/journal.py`。若不存在，`get_primary_parent()` 中需改为 `next((n for n in journal.nodes if n.id == node_id), None)`。
- `select_non_stub_gene` 加 `import random` 时注意不要与 `gene_parser.py` 的现有 import 冲突（当前文件只有 `re` 和 `typing`）。
- `gene_selector.py` 的 `from collections import Counter` 需加到文件顶部，不能在函数内 import（ruff 格式规范）。
