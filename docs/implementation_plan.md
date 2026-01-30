# Swarm-Ev2 架构重构实施计划

## 1. 摘要

本项目构建基于**双层群体智能**的多 Agent 系统，用于 MLE-Bench 刷榜任务。架构分为两层：**Agent 层**（群体智能，优化"如何设计方案"）和 **Solution 层**（遗传算法，优化"方案本身性能"）。两层通过经验池形成正反馈循环。新架构采用清晰的分层设计、统一配置管理、并行执行框架，支持涌现式 Agent 分工与基因级 Solution 进化。

**关键改进**:
- 双层优化：Agent 层元学习 + Solution 层直接优化
- 涌现式分工：Agent 无预定义角色，根据历史表现自然分化专长
- 基因级进化：Solution 由 7 个基因块组成，支持精细交叉与变异
- 解耦设计：拆分 363 行 IterationController 为 Orchestrator + PromptBuilder + SearchStrategy
- 统一配置：OmegaConf 单一 YAML 配置
- 并行执行：复用 ML-Master 的 ThreadPoolExecutor 架构

---

## 2. 审查点

### 2.1 已确认的设计决策
- **不使用 LangGraph**: 纯 Python 实现，更灵活
- **双层群体智能**: Agent 层群体智能 + Solution 层遗传算法
- **涌现式分工**: Agent 无预定义角色，通过经验池隐式协作
- **保留 MLE-Bench**: 核心刷榜目标
- **重构日志系统**: 去除 `log_msg(ERROR)` 自动 raise
- **文件冲突方案**: 采用 ML-Master 的动态文件名重写
- **双入口设计**: `main.py`（白盒调试）+ `mle_bench_adapter.py`（黑盒评测），共享核心流程 `Orchestrator → Strategy → ParallelEvaluator → Agent → Solution`，仅在路径配置、日志策略、环境初始化上分化

### 2.2 已确认的架构参数

| 参数 | 值 | 说明 |
|------|-----|------|
| Agent 种群规模 | 4 | Agent 层个体数 |
| Solution 种群规模 | 12 | Solution 层个体数 |
| 基因块数量 | 7 | DATA, MODEL, LOSS, OPTIMIZER, REGULARIZATION, INITIALIZATION, TRAINING_TRICKS |
| 精英保留 | top-3 | Solution 层精英策略 |
| 锦标赛 k | 3 | Solution 层选择压力 |
| 变异概率 | 20% | 单基因块变异 |
| Prompt 进化周期 | 每 3 Epoch | Agent 层进化频率 |
| 精英保留 (Agent) | top-2 | Agent 层精英策略 |
| 任务分配策略 | 70/30 epsilon-greedy | 70% 擅长 + 30% 探索 |

### 2.3 遗传算法的基因定义

solution.py 由**七个基因组成部分**构成，每个基因通过注释标签区分：

1. **DATA** - 数据加载、预处理、数据增强
2. **MODEL** - 模型架构（Backbone/Head）
3. **LOSS** - 损失函数
4. **OPTIMIZER** - 优化器和调度器
5. **REGULARIZATION** - 正则化技术
6. **INITIALIZATION** - 权重初始化
7. **TRAINING_TRICKS** - 训练技巧

**特殊处理：DATA 基因的固定与可进化区域**

为保证实验可比性，DATA 基因内部划分为固定区域和可进化区域：

```python
# [SECTION: DATA]

# [FIXED: DATA_SPLIT]
# 训练集/测试集划分（固定，所有实验共享）
train_idx, test_idx = train_test_split(range(len(data)), test_size=0.2, random_state=42)

# [EVOLVABLE: DATA_LOADING]
# 数据加载逻辑（可进化）
def load_data(path):
    return pd.read_csv(path)

# [EVOLVABLE: DATA_AUGMENTATION]
# 数据增强（可进化）
def augment_data(data):
    return data
```

**实现要点**：
- 在 Merge 阶段，`# [FIXED: ...]` 部分强制从任意父代原样复制（保证一致性）
- `# [EVOLVABLE: ...]` 部分根据 gene_plan 正常参与交叉
- `parse_solution_genes` 函数需识别 FIXED 和 EVOLVABLE 子区域
- Merge Prompt 模板中添加约束，确保 LLM 遵守固定区域规则

---

## 3. Phase 概览

实施计划分为 5 个 Phase，按依赖关系顺序执行：

### Phase 1: 基础设施重构 (P0 - 必须)
**目标**: 建立新架构的基础设施
- 统一配置系统（OmegaConf + YAML）
- 重构日志系统（去除 auto-raise）
- 后端抽象层（复用 AIDE）
- 数据结构定义（Node, Journal, Task）

**详细计划**: [docs/plans/phase1_infrastructure.md](./plans/phase1_infrastructure.md)

**预计工作量**: 8-12 个文件，约 1000 行代码

---

### Phase 2: 核心功能重构 (P0 - 必须)
**目标**: 实现核心执行引擎
- BaseAgent 抽象类
- Orchestrator（任务编排）
- Interpreter（代码执行沙箱）
- WorkspaceManager（文件隔离）
- PromptBuilder（Prompt 构建）

**详细计划**: [docs/plans/phase2_core.md](./plans/phase2_core.md)

**预计工作量**: 10-15 个文件，约 1500 行代码

---

### Phase 3: 双层群体智能实现 (P0 - 必须)
**目标**: 实现双层群体智能架构
- **Agent 层**: 共享经验池、Prompt 进化、动态任务分配
- **Solution 层**: 遗传算法（精英保留、锦标赛选择、基因交叉、基因变异）
- **两层协同**: 经验池反馈、正反馈循环
- ParallelEvaluator（并行评估框架）

**详细计划**: [docs/plans/phase3_search.md](./plans/phase3_search.md)

**预计工作量**: 8-12 个文件，约 1500 行代码

---

### Phase 4: 扩展功能 (P1 - 重要)
**目标**: 增强系统能力
- Memory 机制增强（Hierarchical Memory）
- 工具注册表（动态加载）
- Agent 注册表（可扩展）
- 高级进化策略（自适应变异率、多目标优化）

**详细计划**: [docs/plans/phase4_extensions.md](./plans/phase4_extensions.md)

**预计工作量**: 5-7 个文件，约 800 行代码

---

### Phase 5: 测试与文档 (P1 - 重要)
**目标**: 确保质量与可维护性
- 单元测试（80%+ 覆盖率）
- 集成测试
- 文档同步（README, CODEMAPS, GUIDES）
- 双入口 MLE-Bench 适配验证

**双入口职责**:

| 入口文件 | 场景 | 路径 | 日志 |
|----------|------|------|------|
| `main.py` | 本地开发调试 | 相对路径 `workspace/` | print() + log_msg() 混合 |
| `mle_bench_adapter.py` | MLE-Bench 生产评测 | 固定路径 `/home/` | 仅结构化日志 |

两者共享核心流程：`Orchestrator → Strategy → ParallelEvaluator → Agent → Solution`

**详细计划**: [docs/plans/phase5_testing.md](./plans/phase5_testing.md)

**预计工作量**: 15-20 个测试文件，约 2000 行测试代码

---

## 4. 依赖关系图

```
Phase 1 (基础设施)
    |
Phase 2 (核心功能)
    |
Phase 3 (双层群体智能)
    |
Phase 4 (扩展功能)
    |
Phase 5 (测试与文档)
```

**并行机会**:
- Phase 1 完成后，Phase 2 的部分模块可以并行开发
- Phase 3 的 Agent 层和 Solution 层可以并行实现
- Phase 5 的测试可以与 Phase 4 部分重叠

---

## 5. 风险与缓解

### Risk 1: 遗传算法交叉后代码语法错误
**影响**: 高
**概率**: 中
**缓解**:
1. 使用 Python AST 验证生成的代码
2. 交叉操作保持函数/类完整性
3. 失败的个体标记为 buggy，触发 debug 流程

### Risk 2: 并发执行的 Workspace 冲突
**影响**: 高
**概率**: 低（已有解决方案）
**缓解**:
1. 采用 ML-Master 的动态文件名重写
2. 输入数据只读（symlink）
3. 输出文件带 node_id 后缀

### Risk 3: Agent 涌现分工可能不收敛
**影响**: 中
**概率**: 中
**缓解**:
1. epsilon-greedy 保证探索多样性（30% 随机分配）
2. 经验池记录完整历史，Prompt 进化基于统计数据
3. 设置最小分化阈值，避免所有 Agent 趋同

### Risk 4: 配置系统学习成本
**影响**: 低
**概率**: 中
**缓解**:
1. 提供详细的配置示例和文档
2. 默认配置覆盖 90% 场景
3. 配置验证提供清晰错误提示

### Risk 5: 两层协同的反馈延迟
**影响**: 中
**概率**: 中
**缓解**:
1. Agent 层每 3 Epoch 进化一次，Solution 层每代进化
2. 经验池实时写入，无需等待进化周期
3. 结构化日志记录协同效果，便于调参

---

## 6. 验证计划

### 6.1 Phase 验证标准

#### Phase 1: 基础设施
- [ ] 配置文件可正确加载（YAML + CLI 覆盖）
- [ ] 日志系统正常工作（不自动 raise）
- [ ] Backend 可调用多个 LLM 提供商
- [ ] Node/Journal/Task 可序列化/反序列化

#### Phase 2: 核心功能
- [ ] Orchestrator 可成功调度任务
- [ ] Interpreter 可执行代码并捕获输出
- [ ] WorkspaceManager 可正确隔离文件
- [ ] PromptBuilder 可生成有效 Prompt

#### Phase 3: 双层群体智能
- [ ] Agent 层：经验池正确记录、Prompt 进化可执行、任务分配符合 epsilon-greedy
- [ ] Solution 层：精英保留正确、锦标赛选择有效、基因交叉产生合法代码、变异改进单基因块
- [ ] 两层协同：Solution 表现正确写入经验池、Agent 进化基于经验池数据
- [ ] ParallelEvaluator 可并发执行多个评估任务

#### Phase 4: 扩展功能
- [ ] Memory 机制正确注入 Prompt
- [ ] 工具可动态注册和加载
- [ ] Agent 可动态注册和加载
- [ ] 高级进化策略可配置切换

#### Phase 5: 测试与文档
- [ ] 单元测试覆盖率 >= 80%
- [ ] 所有集成测试通过
- [ ] 文档与代码同步
- [ ] `main.py` 端到端验证：`python main.py --competition=titanic --steps=5` 产出 `workspace/submission/` 下的结果文件
- [ ] `mle_bench_adapter.py` 端到端验证：模拟 `/home/` 目录结构，产出 `/home/submission/submission.csv`
- [ ] 两入口共享逻辑一致性：相同输入 + 相同 seed 产出相同 Solution

### 6.2 最终验证

**成功标准**:
1. 在 MLE-Bench 上运行完整流程（至少 1 个竞赛）
2. 生成有效的 `submission.csv`
3. 所有测试通过（单元 + 集成）
4. 代码覆盖率 >= 80%
5. 文档完整（README, CODEMAPS, GUIDES）

**验证命令**:
```bash
# 1. 运行测试
pytest tests/ --cov=agents --cov=core --cov=utils --cov=tools --cov-report=term-missing --cov-fail-under=80

# 2. main.py 白盒验证（本地开发）
python main.py --competition=titanic --steps=5
ls workspace/submission/submission_*.csv
cat logs/system.log

# 3. mle_bench_adapter.py 黑盒验证（模拟 MLE-Bench 环境）
python mle_bench_adapter.py
ls /home/submission/submission.csv
cat /home/logs/system.log

# 4. 双入口一致性验证
diff <(python main.py --competition=titanic --steps=1 --seed=42 2>/dev/null && cat workspace/submission/submission_*.csv) \
     <(python mle_bench_adapter.py --seed=42 2>/dev/null && cat /home/submission/submission.csv)
```

---

## 7. 下一步行动

### 立即行动
1. 用户审查本计划
2. 确认双层架构参数（已列于 2.2 节）
3. 开始 Phase 1 实施

### Phase 1 启动前
1. 创建 `config/` 目录结构
2. 设计配置 Schema（YAML 格式）
3. 准备 Backend 模块（从 AIDE 复用）

---

## 8. 附录

### 8.1 目录结构

```
Swarm-Ev2/
├── main.py                    # 白盒入口：本地开发调试（相对路径，混合日志）
├── mle_bench_adapter.py       # 黑盒入口：MLE-Bench 生产评测（/home/ 固定路径，结构化日志）
├── agents/                    # Agent 层（群体智能）
│   ├── base_agent.py         # Agent 基类
│   ├── swarm_agent.py        # 群体 Agent（含 Prompt 进化）
│   └── registry.py           # Agent 注册表
├── core/
│   ├── state/                # 数据结构
│   ├── backend/              # LLM 抽象
│   ├── executor/             # 代码执行
│   ├── orchestrator.py       # 编排器
│   └── evolution/            # 进化机制
│       ├── agent_evolution.py    # Agent 层进化
│       ├── solution_evolution.py # Solution 层遗传算法
│       ├── experience_pool.py    # 共享经验池
│       └── gene_parser.py       # 基因解析器
├── search/                    # 搜索与评估
│   ├── parallel_evaluator.py # 并行评估器
│   └── fitness.py            # 适应度计算
├── tools/                     # 工具注册表
├── utils/                     # 工具模块
├── config/                    # 统一配置
└── tests/                     # 测试（80%+ 覆盖率）
```

#### 双入口对比

```
main.py                              mle_bench_adapter.py
───────────────────────              ─────────────────────────────
CLI 参数 + YAML 配置                  环境变量预设（docker 容器适配）
workspace/ (相对路径)                  /home/ (固定绝对路径)
print() + log_msg() 混合               仅 log_msg() / log_json()
可删除重建 workspace（干净启动）         结果自动整理到 submission/ code/ logs/
                                      合规输出 /home/submission/submission.csv
        ↓                                     ↓
        └──── 共享核心流程 ────────────────────┘
              Orchestrator → Strategy
              → ParallelEvaluator → Agent → Solution
```

### 8.2 代码行数估算

| Phase | 新增代码 | 修改代码 | 删除代码 | 测试代码 |
|-------|---------|---------|---------|---------|
| Phase 1 | ~1000 | ~200 | ~100 | ~500 |
| Phase 2 | ~1500 | ~300 | ~200 | ~800 |
| Phase 3 | ~1500 | ~200 | ~100 | ~700 |
| Phase 4 | ~800 | ~150 | ~50 | ~400 |
| Phase 5 | ~200 | ~100 | ~50 | ~700 |
| **总计** | **~5000** | **~950** | **~500** | **~3100** |

---

## 9. 参考资料

- **AIDE 项目**: `/Reference/aideml-main/`
  - 后端抽象层: `aide/backend/`
  - 配置系统: `aide/utils/config.py`
  - 数据结构: `aide/journal.py`

- **ML-Master 项目**: `/Reference/ML-Master-main/`
  - 并行执行: `interpreter/interpreter_parallel.py`, `main_mcts.py`
  - 配置管理: `utils/config_mcts.py`
  - Memory 机制: `agent/mcts_agent.py`

- **Swarm-Evo (旧版)**: `/Reference/Swarm-Evo/`
  - 架构分析: 见 `draft.md` 第一部分

- **双层群体智能架构**: `docs/architecture/swarm_intelligence.md`

---

**计划版本**: v2.0
**创建日期**: 2026-01-29
**状态**: 已确认
