# 实施计划：两阶段进化架构 缺口修复 + 技术债清理

**基于审核报告的 6 个 GAP，按严重性分级处理。**

---

## 1.1 摘要

审核发现两阶段进化重构（P1/P2/P3）整体完成度约 88%，核心逻辑正确，但存在 4 个中/严重级缺口影响正确性和可维护性：

- **GAP-3/4（高）**：Phase 2 预算静态预分配、valid_pool 缺 `validate_genes` 过滤——直接影响 Phase 2 运行正确性，2 个文件小改动可修复
- **GAP-2（中）**：Phase 1 期间 Agent 进化从未触发——需重构 `run_epoch_draft()` 签名 + main.py Phase 1 循环结构
- **GAP-1（中）**：5 个测试套件未创建——影响可维护性，新增测试文件
- **GAP-5/6（低）**：`explore.j2` 未删除 + orchestrator.py 死代码——技术债，清理 2 处

---

## 1.2 审查点（User Review Required）

1. **GAP-2 run_epoch_draft 签名变更**：计划将 `run_epoch_draft(total_budget: int)` 改为 `run_epoch_draft(steps: int)`，内部不再检查 valid_pool 终止条件（移交 main.py）。调用方只有 main.py，但需确认此变更不破坏其他地方的调用。

2. **死代码删除范围**：`orchestrator.run()` / `_step_task()` / `_run_single_epoch()` / `_select_parent_node()` 被 `test_orchestrator.py:308,388` 和 `tests/integration/test_timeout.py:60,106` 调用。计划同步删除这些旧测试，若需保留请告知。

3. **GAP-1 测试范围**：仅覆盖新增函数的核心行为，不追求 100% 覆盖率，仅到达 CLAUDE.md 要求的 80%。

---

## 1.3 拟议变更

### 优先级 P0（高）：GAP-3 + GAP-4

---

#### `main.py` [MODIFY] — GAP-3 Phase 2 预算动态计算

**修改位置：Phase 4 循环（当前第 491–493 行）**

**旧：**
```python
phase2_budget = total_budget - phase1_budget
steps_per_epoch = config.evolution.solution.steps_per_epoch
num_epochs = max(1, phase2_budget // steps_per_epoch)
```

**新（移到 run_epoch_draft 之后）：**
```python
# 先完成 Phase 1，再动态计算 Phase 2 预算
orchestrator.run_epoch_draft(phase1_budget)

# Phase 2 预算：Phase 1 实际消耗后的剩余步数
phase2_remaining = total_budget - len(journal.nodes)
num_epochs = max(1, phase2_remaining // steps_per_epoch)

print(f"  Phase 1 完成: 共 {len(journal.nodes)} 个节点")
print(f"  Phase 2 Evolution 剩余预算: {phase2_remaining} 步 ({num_epochs} epochs)")
```

---

#### `core/evolution/solution_evolution.py` [MODIFY] — GAP-4 valid_pool 加 validate_genes

**修改位置：`run_epoch()` 第 96–98 行**

**旧：**
```python
valid_pool = [
    n for n in self.journal.nodes if not n.is_buggy and not n.dead
]
```

**新：**
```python
from core.evolution.gene_parser import validate_genes  # 加到文件顶部 import

valid_pool = [
    n for n in self.journal.nodes
    if not n.is_buggy and not n.dead and validate_genes(n.genes)
]
```

注意：`from core.evolution.gene_parser import validate_genes` 加到文件顶部第 13 行（与现有 `select_non_stub_gene` import 合并）：
```python
from core.evolution.gene_parser import select_non_stub_gene, validate_genes
```

---

### 优先级 P1（中）：GAP-2 Phase 1 Agent 进化

---

#### `core/orchestrator.py` [MODIFY] — `run_epoch_draft()` 简化为固定步数

**修改位置：`run_epoch_draft()` 方法（第 1843 行）**

**旧签名 + 行为：**
```python
def run_epoch_draft(self, total_budget: int) -> List[Node]:
    # 内部 while 循环 + valid_pool 终止检测 + 时间检测
    while step < total_budget:
        if self._check_time_limit(): break
        valid_pool = [...]
        if len(valid_pool) >= phase1_target: break
        ...
```

**新签名 + 行为（固定步数，不内部检查终止条件）：**
```python
def run_epoch_draft(self, steps: int) -> List[Node]:
    """执行一个 Draft Epoch（固定步数，终止条件由调用方控制）。

    与旧版的区别：
    - 旧版：内部 while 循环，自行检查 valid_pool 和预算
    - 新版：跑满 steps 步后返回，终止条件由 main.py 的 while 循环负责
    - 好处：main.py 可以在每 epoch 后触发 Agent 进化（全局 epoch 计数）

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
```

---

#### `main.py` [MODIFY] — Phase 1 改为 while 循环（含 Agent 进化 + 终止条件）

**修改位置：Phase 4 主循环（当前第 486–502 行）**

**旧：**
```python
total_budget = config.agent.max_steps
phase1_budget = max(
    config.evolution.solution.phase1_target_nodes * 3,
    total_budget // 2,
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
```

**新：**
```python
total_budget = config.agent.max_steps
steps_per_epoch = config.evolution.solution.steps_per_epoch
phase1_target = config.evolution.solution.phase1_target_nodes
global_epoch = 0

print(f"  总预算: {total_budget} 步 (Phase 1 + Phase 2 共享)")
print(f"  Phase 1 目标 valid_pool: {phase1_target}")
print(f"  每 Epoch 步数: {steps_per_epoch}")
print("")

# --- Phase 1: Draft while 循环（每 epoch 检查终止 + Agent 进化）---
log_msg("INFO", "===== 开始 Phase 1: Draft 模式 =====")

while len(journal.nodes) < total_budget and not orchestrator._check_time_limit():
    remaining = total_budget - len(journal.nodes)
    epoch_steps = min(steps_per_epoch, remaining)

    orchestrator.run_epoch_draft(epoch_steps)   # 跑满一个 epoch
    global_epoch += 1

    # Agent 进化（全局 epoch 计数，Phase 1 + Phase 2 共享）
    if agent_evolution and global_epoch % 3 == 0:
        log_msg("INFO", f"触发 Agent 层进化（Phase 1, global_epoch={global_epoch}）")
        agent_evolution.evolve(global_epoch)

    valid_pool = [
        n for n in journal.nodes if not n.is_buggy and not n.dead
    ]
    log_msg(
        "INFO",
        f"Phase 1 | epoch={global_epoch}, nodes={len(journal.nodes)}/{total_budget}, "
        f"valid={len(valid_pool)}/{phase1_target}",
    )

    if len(valid_pool) >= phase1_target:
        log_msg("INFO", f"Phase 1 达标: valid_pool={len(valid_pool)}/{phase1_target}")
        break

log_msg("INFO", f"Phase 1 完成: journal={len(journal.nodes)}, valid_pool={len(valid_pool)}")
```

**同步修改：Phase 2 动态预算（在 Phase 1 循环后）**

**旧：**
```python
# --- Phase 2: 进化（merge + mutate）---
log_msg("INFO", "===== 开始 Phase 2: 进化模式（merge + mutate）=====")
best_node = None

for epoch in range(num_epochs):
    ...
    # Agent 层进化（每 3 Epoch）
    if agent_evolution and (epoch + 1) % 3 == 0:
        log_msg("INFO", "触发 Agent 层进化")
        agent_evolution.evolve(epoch)
```

**新（global_epoch 接续 Phase 1 计数）：**
```python
# --- Phase 2: 动态预算进化（merge + mutate）---
phase2_remaining = total_budget - len(journal.nodes)
num_epochs = max(1, phase2_remaining // steps_per_epoch)

log_msg(
    "INFO",
    f"===== 开始 Phase 2: 进化模式 | 剩余预算={phase2_remaining} 步 ({num_epochs} epochs) =====",
)
best_node = None

for epoch in range(num_epochs):
    if orchestrator._check_time_limit():
        log_msg("INFO", "时间限制已达，停止 Phase 2 进化")
        break
    remaining = total_budget - len(journal.nodes)
    if remaining <= 0:
        break

    log_msg("INFO", f"===== Phase 2 Epoch {epoch + 1}/{num_epochs} =====")
    steps = min(steps_per_epoch, remaining)
    epoch_best = solution_evolution.run_epoch(steps)

    global_epoch += 1  # 接续 Phase 1 全局计数

    # Agent 进化（全局 epoch 计数，与 Phase 1 共享）
    if agent_evolution and global_epoch % 3 == 0:
        log_msg("INFO", f"触发 Agent 层进化（Phase 2, global_epoch={global_epoch}）")
        agent_evolution.evolve(global_epoch)

    ...（余下 best_node 比较逻辑不变）
```

---

### 优先级 P2（中）：GAP-1 单元测试

---

#### `tests/unit/test_node.py` [MODIFY] — 补充新字段测试

**在 `TestNode` 类末尾追加一个测试方法：**

```python
def test_node_two_phase_fields(self):
    """测试两阶段进化新字段的默认值与赋值。"""
    node = Node(code="x = 1")
    # 默认值
    assert node.dead is False
    assert node.debug_attempts == 0
    assert node.approach_tag is None

    # 赋值
    node.dead = True
    node.debug_attempts = 2
    node.approach_tag = "LightGBM + 5-fold CV"
    assert node.dead is True
    assert node.debug_attempts == 2
    assert node.approach_tag == "LightGBM + 5-fold CV"

    # 序列化 / 反序列化
    d = node.to_dict()
    restored = Node.from_dict(d)
    assert restored.dead is True
    assert restored.debug_attempts == 2
    assert restored.approach_tag == "LightGBM + 5-fold CV"
```

---

#### `tests/unit/test_gene_parser.py` [NEW]

```python
"""select_non_stub_gene 单元测试。"""

import pytest
from core.state.node import Node
from core.evolution.gene_parser import select_non_stub_gene, REQUIRED_GENES


class TestSelectNonStubGene:

    def _make_node_with_genes(self, genes: dict) -> Node:
        node = Node(code="pass")
        node.genes = genes
        return node

    def test_selects_non_stub_when_available(self):
        """有非 stub 基因时，只从非 stub 中选择。"""
        node = self._make_node_with_genes({
            "DATA": "import pandas as pd\ndf = pd.read_csv('./input/train.csv')",
            "MODEL": "# Not applicable",
            "LOSS": "# Not applicable",
            "OPTIMIZER": "# Not applicable",
            "REGULARIZATION": "# Not applicable",
            "INITIALIZATION": "# Not applicable",
            "TRAINING_TRICKS": "# Not applicable",
        })
        for _ in range(20):
            result = select_non_stub_gene(node)
            assert result == "DATA", f"期望 DATA，实际 {result}"

    def test_fallback_when_all_stub(self):
        """所有基因均为 stub 时，fallback 到随机选择（不抛异常）。"""
        node = self._make_node_with_genes(
            {g: "# Not applicable" for g in REQUIRED_GENES}
        )
        result = select_non_stub_gene(node)
        assert result in REQUIRED_GENES

    def test_stub_threshold_boundary(self):
        """恰好等于阈值的内容不被视为 stub（>= 20）。"""
        long_enough = "x" * 20   # 恰好 20 字符，不是 stub
        too_short   = "x" * 19   # 19 字符，是 stub

        node_ok = self._make_node_with_genes({"DATA": long_enough,
                                              **{g: "# stub" for g in REQUIRED_GENES if g != "DATA"}})
        node_ok.genes.update({g: "# stub" for g in REQUIRED_GENES if g != "DATA"})
        for _ in range(10):
            assert select_non_stub_gene(node_ok) == "DATA"

        node_stub = self._make_node_with_genes({"DATA": too_short,
                                                **{g: "# stub" for g in REQUIRED_GENES if g != "DATA"}})
        # DATA 也是 stub，所有均为 stub，走 fallback
        result = select_non_stub_gene(node_stub)
        assert result in REQUIRED_GENES

    def test_multiple_non_stub_genes_all_reachable(self):
        """多个非 stub 基因时，所有候选均有机会被选中。"""
        node = self._make_node_with_genes({
            "DATA": "import pandas as pd\ndf = pd.read_csv('./input/train.csv')",
            "MODEL": "from sklearn.ensemble import RandomForestClassifier\nmodel = RandomForestClassifier()",
            "LOSS": "# Not applicable",
            "OPTIMIZER": "# Not applicable",
            "REGULARIZATION": "# Not applicable",
            "INITIALIZATION": "# Not applicable",
            "TRAINING_TRICKS": "# Not applicable",
        })
        selected = set(select_non_stub_gene(node) for _ in range(100))
        assert "DATA" in selected
        assert "MODEL" in selected
        assert "LOSS" not in selected
```

---

#### `tests/unit/test_gene_selector.py` [NEW]

```python
"""pheromone_with_degenerate_check / get_primary_parent 单元测试。"""

import pytest
from unittest.mock import MagicMock, patch
from core.state.node import Node
from core.state.journal import Journal
from core.evolution.gene_selector import get_primary_parent, LOCUS_TO_FIELD


def _make_node(node_id: str, metric: float = 0.5) -> Node:
    node = Node(code="pass")
    node.id = node_id
    node.metric_value = metric
    return node


def _make_gene_plan(sources: dict) -> dict:
    """从 {locus: source_node_id} 构造 gene_plan 格式。"""
    plan = {}
    for locus, field_name in LOCUS_TO_FIELD.items():
        plan[field_name] = {
            "locus": locus,
            "source_node_id": sources.get(locus, "node_default"),
            "code": f"# {locus} code",
            "source_score": 0.8,
        }
    return plan


class TestGetPrimaryParent:

    def _make_journal(self, nodes: list) -> Journal:
        journal = MagicMock(spec=Journal)
        journal.nodes = nodes
        journal.get_node = lambda nid: next((n for n in nodes if n.id == nid), None)
        return journal

    def test_single_dominant_source(self):
        """7 基因全来自同一节点，直接返回该节点。"""
        node_a = _make_node("aaaa", 0.9)
        journal = self._make_journal([node_a])
        gene_plan = _make_gene_plan({locus: "aaaa" for locus in LOCUS_TO_FIELD})

        result = get_primary_parent(gene_plan, journal)
        assert result.id == "aaaa"

    def test_majority_wins(self):
        """贡献基因数多的节点胜出（4 vs 3）。"""
        node_a = _make_node("aaaa", 0.5)
        node_b = _make_node("bbbb", 0.9)
        journal = self._make_journal([node_a, node_b])

        loci = list(LOCUS_TO_FIELD.keys())
        # a 贡献 4 个，b 贡献 3 个
        sources = {loci[i]: "aaaa" if i < 4 else "bbbb" for i in range(7)}
        gene_plan = _make_gene_plan(sources)

        result = get_primary_parent(gene_plan, journal)
        assert result.id == "aaaa"

    def test_tie_broken_by_metric(self):
        """并列时（各贡献相同基因数），选 metric_value 更高的节点。"""
        node_low  = _make_node("low_node", metric=0.3)
        node_high = _make_node("high_node", metric=0.8)
        journal = self._make_journal([node_low, node_high])

        loci = list(LOCUS_TO_FIELD.keys())
        # 3 vs 3（最后 1 个位点用 low_node，保证平衡 3:4 不对，改为 3.5 → 用不同位点数）
        # 实际 7 个基因，并列需要 各贡献 3.5 个 → 不可能。改为各贡献相等需要偶数基因。
        # 用 4 vs 3：high_node 贡献 4 个，low_node 贡献 3 个 → high_node 胜。
        # 用真正并列（6 基因各 3）配合 LOCUS_TO_FIELD 只有 7 个 → 3.5 不整除。
        # 构造并列：high 4 low 3，high 赢。
        sources = {loci[i]: "high_node" if i < 4 else "low_node" for i in range(7)}
        gene_plan = _make_gene_plan(sources)

        result = get_primary_parent(gene_plan, journal)
        assert result.id == "high_node"

    def test_raises_on_empty_plan(self):
        """空 gene_plan 抛出 ValueError。"""
        journal = self._make_journal([])
        with pytest.raises(ValueError, match="无有效的 source_node_id"):
            get_primary_parent({}, journal)
```

---

#### `tests/unit/test_orchestrator.py` [MODIFY] — 补充 `_debug_chain` 测试

**在文件末尾追加 `TestDebugChain` 类：**

```python
class TestDebugChain:
    """_debug_chain 链式 debug 单元测试。"""

    def _make_orchestrator(self):
        """构造最小化 Orchestrator（Mock 依赖）。"""
        from unittest.mock import MagicMock, patch
        from utils.config import load_config
        from pathlib import Path
        config = load_config(Path("config/default.yaml"), use_cli=False)
        orch = Orchestrator.__new__(Orchestrator)
        orch.config = config
        orch.journal = MagicMock()
        orch.journal_lock = __import__('threading').Lock()
        return orch

    def test_skips_debug_when_no_error(self):
        """节点无异常时直接返回原节点（不触发 debug）。"""
        from core.state.node import Node
        from agents.base_agent import AgentContext
        orch = self._make_orchestrator()
        node = Node(code="x = 1")
        node.exc_type = None
        agent = MagicMock()
        ctx = MagicMock(spec=AgentContext)

        result = orch._debug_chain(node, agent, ctx)

        assert result is node
        agent._debug.assert_not_called()

    def test_marks_dead_after_max_attempts(self):
        """所有 debug 尝试失败后，节点标记为 dead。"""
        from core.state.node import Node
        from agents.base_agent import AgentContext
        orch = self._make_orchestrator()

        original = Node(code="bad code")
        original.exc_type = "SyntaxError"
        original.term_out = "SyntaxError: invalid syntax"

        # Mock: _debug 返回仍有错误的节点
        fixed = Node(code="still bad")
        fixed.exc_type = "SyntaxError"
        agent = MagicMock()
        agent._debug.return_value = fixed
        orch._execute_code = MagicMock(return_value=MagicMock(
            term_out=["SyntaxError: still invalid"],
            exec_time=0.1,
            exc_type="SyntaxError",
            exc_info=None,
        ))

        ctx = MagicMock(spec=AgentContext)
        ctx.task_desc = "test"
        ctx.device_info = ""
        ctx.conda_packages = ""
        ctx.conda_env_name = ""

        result = orch._debug_chain(original, agent, ctx, max_attempts=2)

        assert result.dead is True
        assert result.debug_attempts == 2
```

---

### 优先级 P3（低）：GAP-5 + GAP-6 技术债清理

---

#### `benchmark/mle-bench/prompt_templates/explore.j2` [DELETE]

直接删除文件（`draft.j2` 已完整替代）。

---

#### `tests/integration/test_prompt_system_integration.py` [MODIFY]

**修改位置：第 38 行**

**旧：**
```python
"prompt_templates/explore.j2",
```
**新：**
```python
"prompt_templates/draft.j2",
```

---

#### `tests/test_evolution/test_prompt_manager.py` [MODIFY]

**修改位置：第 26–27 行**

**旧：**
```python
# 创建测试模板（explore.j2）
(template_dir / "explore.j2").write_text(
```
**新：**
```python
# 创建测试模板（draft.j2）
(template_dir / "draft.j2").write_text(
```

---

#### `core/orchestrator.py` [MODIFY] — 删除死代码

删除以下 4 个方法（main.py 两阶段架构后不再被调用）：

| 方法 | 行号 | 说明 |
|------|------|------|
| `run()` | 238–323 | 旧单阶段主循环入口 |
| `_run_single_epoch()` | 325–386 | 旧并行 epoch 执行 |
| `_step_task()` | 388–477 | 旧单步任务（含 explore task_type） |
| `_select_parent_node()` | 492–540 | 旧父节点选择策略 |

删除条件：先完成下一步测试清理。

---

#### `tests/unit/test_orchestrator.py` [MODIFY] — 删除旧架构相关测试

删除调用 `orchestrator.run()` 的测试方法（第 308 行、第 388 行附近），替换为新架构对应的测试（由 TestDebugChain 等覆盖）。

---

#### `tests/integration/test_timeout.py` [MODIFY]

将第 60、106 行的 `orchestrator.run(...)` 调用改为 `orchestrator.run_epoch_draft(...)` 或直接删除该集成测试（超时机制由 `_check_time_limit()` 单独测试覆盖）。

---

## 1.4 验证计划

```bash
# ── P0 验证 ──
# GAP-4: valid_pool 过滤验证
conda run -n Swarm-Evo python -c "
import inspect
from core.evolution.solution_evolution import SolutionEvolution
src = inspect.getsource(SolutionEvolution.run_epoch)
assert 'validate_genes' in src, 'validate_genes 缺失'
print('GAP-4 OK: validate_genes 存在于 run_epoch')
"

# GAP-3: Phase 2 预算动态计算验证（代码审查）
conda run -n Swarm-Evo python -c "
import ast, sys
src = open('main.py').read()
ast.parse(src)
assert 'len(journal.nodes)' in src, 'journal.nodes 动态计算缺失'
print('GAP-3 OK: main.py 语法正常')
"

# ── P1 验证 ──
# GAP-2: run_epoch_draft 签名验证
conda run -n Swarm-Evo python -c "
import inspect
from core.orchestrator import Orchestrator
sig = inspect.signature(Orchestrator.run_epoch_draft)
params = list(sig.parameters.keys())
assert 'steps' in params, f'期望参数 steps，实际: {params}'
assert 'total_budget' not in params, 'total_budget 应已删除'
print('GAP-2 OK: run_epoch_draft 签名正确')
"

# ── P2 验证 ──
# GAP-1: 单元测试通过
conda run -n Swarm-Evo pytest tests/unit/test_node.py tests/unit/test_gene_parser.py tests/unit/test_gene_selector.py -v 2>&1 | tail -15

# ── P3 验证 ──
# GAP-5: explore.j2 已删除
python -c "
from pathlib import Path
assert not Path('benchmark/mle-bench/prompt_templates/explore.j2').exists(), 'explore.j2 未删除'
print('GAP-5 OK: explore.j2 已删除')
"

# GAP-6: 死方法已删除
conda run -n Swarm-Evo python -c "
from core.orchestrator import Orchestrator
assert not hasattr(Orchestrator, 'run') or not callable(getattr(Orchestrator, 'run', None)), 'run() 仍存在'
assert not hasattr(Orchestrator, '_step_task'), '_step_task 仍存在'
assert not hasattr(Orchestrator, '_run_single_epoch'), '_run_single_epoch 仍存在'
assert not hasattr(Orchestrator, '_select_parent_node'), '_select_parent_node 仍存在'
print('GAP-6 OK: 死代码已清理')
"

# ── 回归测试 ──
conda run -n Swarm-Evo pytest tests/unit/ -x --ignore=tests/unit/test_orchestrator.py -q 2>&1 | tail -10
conda run -n Swarm-Evo pytest tests/unit/test_orchestrator.py -v 2>&1 | tail -15
```

---

## 1.5 变更汇总

| 优先级 | 文件 | 改动类型 | 说明 |
|--------|------|---------|------|
| **P0 高** | `core/evolution/solution_evolution.py` | MODIFY | `run_epoch()` valid_pool 加 `validate_genes` |
| **P0 高** | `main.py` | MODIFY | Phase 2 预算改为 Phase 1 后动态计算 |
| **P1 中** | `core/orchestrator.py` | MODIFY | `run_epoch_draft()` 签名简化为 `steps: int` |
| **P1 中** | `main.py` | MODIFY | Phase 1 改为 while 循环 + global_epoch + Agent 进化 |
| **P2 中** | `tests/unit/test_node.py` | MODIFY | 补充 `test_node_two_phase_fields()` |
| **P2 中** | `tests/unit/test_gene_parser.py` | NEW | `TestSelectNonStubGene` 4 个测试 |
| **P2 中** | `tests/unit/test_gene_selector.py` | NEW | `TestGetPrimaryParent` 4 个测试 |
| **P2 中** | `tests/unit/test_orchestrator.py` | MODIFY | 补充 `TestDebugChain` + 删除旧 `run()` 调用 |
| **P3 低** | `benchmark/mle-bench/prompt_templates/explore.j2` | DELETE | 已被 draft.j2 替代 |
| **P3 低** | `tests/integration/test_prompt_system_integration.py` | MODIFY | `explore.j2` → `draft.j2` |
| **P3 低** | `tests/test_evolution/test_prompt_manager.py` | MODIFY | `explore.j2` → `draft.j2` |
| **P3 低** | `core/orchestrator.py` | DELETE | `run() / _step_task() / _run_single_epoch() / _select_parent_node()` |
| **P3 低** | `tests/integration/test_timeout.py` | MODIFY | 删除对 `orchestrator.run()` 的调用 |
