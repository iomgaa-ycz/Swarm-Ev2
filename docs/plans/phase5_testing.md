# Phase 5: 测试与文档详细计划

## 1. 目标

1) **单元测试与集成测试**
- 使用 `pytest` + `pytest-asyncio`，并通过 `pytest-cov` 约束覆盖率 **>= 80%**。
- 测试优先覆盖：配置/日志、Node/Journal、Interpreter/Workspace、Strategy(MCTS/GA/Hybrid)/ParallelEvaluator、Memory、Registry、MLE-Bench adapter。

2) **关键流程验证（端到端 smoke test）**
至少包含一次完整闭环（允许 mock/最小数据）：
“策略驱动 -> 生成代码 -> 执行 -> 解析 metric -> 产出 `submission_{node_id}.csv` -> `best_solution` 更新”。

3) **文档同步（按 SOP）**
- 必须使用 **doc-updater agent** 扫描并更新/校验：`README.md`、`docs/CODEMAPS/`、`docs/GUIDES/`。
- 若目录不存在：给出**创建策略**与**最小内容**，避免过度工程化。

4) **MLE-Bench 适配验证**
- 提供可执行的验证步骤：跑通 adapter + 产出 submission（不要求真实刷榜）。

---

## 2. 文件清单

> 说明：当前仓库扫描结果显示业务代码可能尚未全部落盘（主要为 `docs/` 与 `Reference/`）。以下文件路径以 Phase1-4 规划的约定结构为基准；实现时若实际路径不同，应同步调整测试导入路径与覆盖率统计目标。

### 2.1 新建文件 [NEW]

#### 测试基础设施
- `tests/conftest.py`
  - 职责说明：统一 fixtures（临时 workspace、最小 Config、FakeAgent、FakeStrategy、日志隔离等）。
  - 关键测试点 / 关键函数：
    - `workspace_root(tmp_path) -> Path`
    - `minimal_config(workspace_root) -> Config`
    - `fake_agent_factory() -> BaseAgent`
    - `monkeypatch_no_print()`（可选）

- `pyproject.toml`（或 `pytest.ini` + `.coveragerc` 二选一；推荐 `pyproject.toml`）
  - 职责说明：配置 pytest markers、coverage omit（跳过 `Reference/`）、以及 `--cov-fail-under=80`。

#### 单元测试
- `tests/test_utils/test_logger_system.py`
- `tests/test_utils/test_config_validation.py`
- `tests/test_state/test_node.py`
- `tests/test_state/test_journal.py`
- `tests/test_executor/test_workspace_manager.py`
- `tests/test_executor/test_interpreter.py`
- `tests/test_strategies/test_parallel_evaluator.py`
- `tests/test_strategies/test_mcts_strategy.py`
- `tests/test_strategies/test_genetic_strategy.py`
- `tests/test_strategies/test_hybrid_strategy.py`
- `tests/test_memory/test_hierarchical_memory.py`
- `tests/test_registry/test_tool_registry.py`
- `tests/test_registry/test_agent_registry.py`

#### 集成/端到端 smoke
- `tests/test_integration/test_smoke_strategy_to_submission.py`
  - 职责说明：核心流程闭环测试（Strategy → Agent → submission 产出）。
  - 关键测试点：submission 产出、metric 解析、best_solution 更新。

- `tests/test_integration/test_main_smoke.py`
  - 职责说明：`main.py` 白盒端到端验证（CLI 解析、workspace 生命周期、调试输出）。
  - 关键测试点：CLI 参数传递、workspace 清理重建、submission 文件产出。

- `tests/test_integration/test_mle_bench_adapter_smoke.py`
  - 职责说明：`mle_bench_adapter.py` 黑盒端到端验证（环境变量、固定路径、结果整理）。
  - 关键测试点：路径合规 `/home/submission/submission.csv`、`code/` `logs/` 目录结构、无 print 泄漏。

- `tests/test_integration/test_dual_entry_consistency.py`
  - 职责说明：验证两入口在相同输入 + 相同 seed 下产出一致的 Solution。
  - 关键测试点：CSV 内容一致性。

#### 文档最小骨架（若不存在则创建）
- `README.md`
- `docs/CODEMAPS/architecture.md`
- `docs/GUIDES/testing.md`
- `docs/GUIDES/mle_bench.md`

### 2.2 修改文件 [MODIFY]

- `core/orchestrator.py`
  - [MODIFY] `Orchestrator._extract_metric(self, term_out: str) -> float | None`：保证可单测。
  - [MODIFY] `Orchestrator._save_best_solution(self, node: Node) -> None`：写入路径可控，并用 `log_json` 记录。
  - [MODIFY] `Orchestrator._execute_code(self, code: str, node_id: str) -> ExecutionResult`：强制经 rewrite。

- `core/executor/workspace.py`
  - [MODIFY] `WorkspaceManager.rewrite_submission_path(self, code: str, node_id: str) -> str`：一致性增强与诊断日志。

- `utils/logger_system.py`
  - [MODIFY] 确认 `log_msg/log_json` 可隔离日志目录，便于测试。

- `docs/implementation_plan.md`
  - [MODIFY] 同步 Phase5 验证命令与文档链接。

---

## 3. 详细设计

### 3.1 测试总策略与分层（unit/integration/smoke）

- **Unit**：确定性、无外部依赖。
- **Integration**：模块协作但仍 mock LLM。
- **Smoke**：最小闭环一次，保证关键产物存在。

统一 mock 原则：
- patch `core.backend.query` 或 `BaseAgent.generate`，禁止真实 LLM。
- `tmp_path` 构造最小 `input/` 与 `submission/`。
- Interpreter 可真实执行最小代码（常数预测写 CSV）。

### 3.2 单元测试：核心模块覆盖

- Config/Logger：边界校验、结构化日志写入。
- Node/Journal：best 选择、树遍历、signature/统计字段。
- Workspace/Interpreter：重写路径、执行/异常/超时。
- Strategy/ParallelEvaluator：闭环、早停、回传、elitism。

### 3.3 集成测试：Orchestrator + Strategy + Evaluator

- `test_smoke_strategy_to_submission.py`：
  - FakeAgent 返回可执行代码，写入 `submission/submission.csv`（由 rewrite 变为 `submission_{node_id}.csv`）。
  - 代码打印 `metric: <float>`，确保 `_extract_metric` 可解析。
  - 断言 best_solution 更新。

### 3.4 MLE-Bench 适配层验证（双入口策略）

#### 3.4.1 两入口职责与测试重点

| 维度 | main.py（白盒） | mle_bench_adapter.py（黑盒） | tests/（单模块） |
|------|-----------------|---------------------------|----------------|
| **场景** | 本地开发调试 | MLE-Bench 生产评测 | 质量保证 |
| **路径** | 相对路径 `workspace/` | 固定路径 `/home/` | 临时路径 `tmp_path` |
| **日志** | print() + log_msg() 混合 | 仅结构化日志 | pytest capture |
| **测试重点** | CLI 参数解析、workspace 生命周期、调试输出完整性 | 环境变量注入、路径合规、结果文件整理 | 单模块隔离、边界条件 |

两入口共享核心流程：`Orchestrator → Strategy → ParallelEvaluator → Agent → Solution`

#### 3.4.2 main.py 测试策略

```
test_main_smoke.py
├── test_cli_args_parsing          # --competition, --steps, --seed 正确传递到 Config
├── test_workspace_clean_start     # workspace/ 可删除重建
├── test_submission_output         # workspace/submission/submission_{node_id}.csv 产出
└── test_debug_output_richness     # stdout 包含阶段性进度信息
```

伪代码示例：
```python
async def test_main_end_to_end(tmp_path, monkeypatch):
    """main.py 端到端 smoke：CLI → Orchestrator → submission 产出。"""
    monkeypatch.chdir(tmp_path)
    # 准备最小 input 数据
    (tmp_path / "workspace" / "input").mkdir(parents=True)
    write_minimal_csv(tmp_path / "workspace" / "input" / "train.csv")

    # mock LLM 返回常数预测代码
    with patch("core.backend.query", return_value=CONSTANT_PREDICTION_CODE):
        exit_code = await run_main(["--competition=test", "--steps=1", "--seed=42"])

    assert exit_code == 0
    submissions = list((tmp_path / "workspace" / "submission").glob("submission_*.csv"))
    assert len(submissions) >= 1
```

#### 3.4.3 mle_bench_adapter.py 测试策略

```
test_mle_bench_adapter_smoke.py
├── test_env_vars_injection        # 环境变量 → Config 映射正确
├── test_fixed_path_compliance     # 产出在 /home/submission/submission.csv
├── test_result_file_organization  # submission/, code/, logs/ 三目录结构
└── test_no_print_statements       # 无 print() 泄漏（仅结构化日志）
```

伪代码示例：
```python
async def test_adapter_path_compliance(tmp_path, monkeypatch):
    """mle_bench_adapter.py：验证固定路径合规性。"""
    fake_home = tmp_path / "home"
    (fake_home / "input").mkdir(parents=True)
    write_minimal_csv(fake_home / "input" / "train.csv")

    # 重定向 /home/ → tmp_path/home/
    monkeypatch.setenv("HOME_DIR", str(fake_home))

    with patch("core.backend.query", return_value=CONSTANT_PREDICTION_CODE):
        await run_adapter()

    # 产物验证
    assert (fake_home / "submission" / "submission.csv").exists()
    assert (fake_home / "code").is_dir()
    assert (fake_home / "logs").is_dir()
```

#### 3.4.4 双入口一致性验证

```python
async def test_dual_entry_consistency(tmp_path, monkeypatch):
    """相同输入 + 相同 seed → 两入口产出相同 Solution 内容。"""
    seed = 42
    with patch("core.backend.query", return_value=CONSTANT_PREDICTION_CODE):
        result_main = await run_main(["--steps=1", f"--seed={seed}"], workdir=tmp_path / "a")
        result_adapter = await run_adapter(home_dir=tmp_path / "b", seed=seed)

    csv_main = read_csv(tmp_path / "a" / "workspace" / "submission" / "submission_0.csv")
    csv_adapter = read_csv(tmp_path / "b" / "submission" / "submission.csv")
    assert csv_main.equals(csv_adapter)
```

### 3.5 文档同步（doc-updater）

- 必须运行 doc-updater agent，同步：README、CODEMAPS、GUIDES。
- 若目录缺失：创建最小文件，记录运行/测试/adapter 验证命令。

---

## 4. 验证计划

### 4.1 测试命令

```bash
# 全量测试 + 覆盖率
pytest tests -v --cov=agents --cov=core --cov=utils --cov=tools --cov-report=term-missing --cov-fail-under=80

# 核心流程 smoke
pytest tests/test_integration/test_smoke_strategy_to_submission.py -v

# main.py 白盒验证
pytest tests/test_integration/test_main_smoke.py -v
python main.py --competition=titanic --steps=5 --seed=42
ls workspace/submission/submission_*.csv

# mle_bench_adapter.py 黑盒验证
pytest tests/test_integration/test_mle_bench_adapter_smoke.py -v
python mle_bench_adapter.py                         # 模拟 docker 环境
ls /home/submission/submission.csv                   # 合规产物
ls /home/code/ /home/logs/                           # 结果整理目录

# 双入口一致性
pytest tests/test_integration/test_dual_entry_consistency.py -v
```

### 4.2 mock 方案

- Mock LLM：patch `core.backend.query` 或 FakeAgent。
- 隔离 workspace：`config.project.workspace_dir = tmp_path/"workspace"`。
- adapter 路径重定向：通过 `monkeypatch.setenv("HOME_DIR", str(tmp_path))` 避免真实写入 `/home/`。
- 并发：smoke 默认并发=1；并发冲突用单独测试覆盖。

### 4.3 文档验证

- doc-updater 后检查：README/GUIDES 命令可执行、路径一致、submission 命名规则与 best_solution 更新明确。
- 双入口文档：确认 `docs/GUIDES/mle_bench.md` 包含两入口的使用说明和产物说明。

---

## 5. 风险与缓解

1) 路径与规划不一致导致测试导入失败：集中在 `tests/conftest.py` 修正；coverage 显式指定目录。
2) smoke 不稳定：固定随机种子、并发=1、强制 rewrite。
3) metric 解析不稳定：抽成可单测函数；输出固定格式 `metric: <float>`。
4) adapter 依赖过重：使用最小 CSV 与常数预测；禁止网络。
5) 规范违规（缺类型注解/缺中文 Docstring/使用 print）：加入自检清单，必要时在测试中拦截 print。
