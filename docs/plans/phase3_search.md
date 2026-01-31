# Phase 3: 双层群体智能实现

**Last Updated:** 2026-01-31
**Status:** 🔴 待实现
**Dependencies:** Phase 1 ✅, Phase 2 ✅

---

## 1. 目标

实现 Swarm-Ev2 的核心创新架构 -- **双层群体智能**：

| 层级 | 优化对象 | 机制 | 种群规模 |
|------|---------|------|---------|
| **Agent 层** | 如何设计方案（元学习） | 经验池 + Prompt 进化 + 动态任务分配 | 4 个 Agent |
| **Solution 层** | 方案本身性能（直接优化） | 遗传算法（精英保留 + 锦标赛 + 交叉 + 变异） | 12 个 solution.py |

两层通过经验池反馈形成正循环：好 Prompt -> 好 Solution -> 好经验 -> 更好 Prompt。

---

## 2. 第一部分：Agent 层群体智能

### 2.1 Agent 种群设计

4 个 Agent 共享同一任务空间，无预定义角色，通过历史表现自然分化专长（涌现式分工）。

**Agent 状态结构：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `agent_id` | `str` | 唯一标识 |
| `system_prompt` | `str` | 当前系统提示词（可进化） |
| `specialization_scores` | `dict[str, float]` | 各任务类型的擅长度得分 |
| `generation` | `int` | 当前所属代数 |

> **注意**：`task_history` 不再存储在 Agent 中，由经验池统一管理。

---

### 2.2 任务类型（3 种）

| 任务 | 英文名称 | 说明 | 输入 | 输出 | 对应遗传算法操作 |
|------|---------|------|------|------|----------------|
| **完整方案探索** | `explore` | 从零生成或整体改进方案 | 任务描述 + 父节点（可选） | 完整 solution.py | 种群初始化 / 大范围变异 |
| **基因融合** | `merge` | 从两个父代合成子代 | 父代 A + 父代 B + gene_plan | 合成后的 solution.py | **交叉（Crossover）** |
| **基因变异** | `mutate` | 局部改进某个基因块 | 父代 + target_gene | 改进后的 solution.py | **变异（Mutation）** |

**设计说明：**

1. **`explore`** - 整体级别的探索
   - 包含内部 Review 环节（由 Orchestrator 的 `_review_node()` 完成）
   - 可以从零生成（parent_node=None）或基于父节点改进
   - 自由度最高，可以大幅修改代码

2. **`merge`** - 基因交叉
   - 包含内部 Select 环节（锦标赛选择父代）
   - 根据 gene_plan 从两个父代选择基因块
   - LLM 负责解决命名冲突和兼容性

3. **`mutate`** - 基因变异
   - 只修改指定的基因块（如 `MODEL`）
   - 其他基因块必须保持不变
   - 约束性最强，确保局部优化

**与原设计的变化：**
- ❌ 删除 `select` 任务 → 成为 `merge` 的内部环节（锦标赛选择）
- ❌ 删除 `review` 任务 → 成为所有任务的通用后处理步骤（Function Calling）
- ✅ 新增 `mutate` 任务 → 独立的基因变异操作

---

### 2.3 共享经验池（Experience Pool）

经验池是 Agent 层隐式协作的核心数据结构。所有 Agent 向同一个池写入，从同一个池读取。

```
Experience Pool
+--------------------------------------------------+
| TaskRecord                                        |
|   agent_id: str       # 执行者                     |
|   task_type: str      # explore/merge/mutate      |
|   input_hash: str     # 输入摘要指纹               |
|   output_quality: float  # 产出质量（fitness 变化）  |
|   strategy_summary: str  # 策略摘要                 |
|   timestamp: float                                 |
+--------------------------------------------------+
```

**读写规则：**

| 操作 | 时机 | 数据 |
|------|------|------|
| **写入** | 每次任务执行完毕 | TaskRecord（含策略摘要和质量评分） |
| **读取** | Agent 生成方案前 | 同类任务的 Top-K 成功记录 + 常见失败模式 |
| **聚合** | Prompt 进化时 | 按 agent_id 聚合统计（成功率、平均质量） |

---

### 2.4 Prompt 进化

**架构原则**: Agent层双进化机制 = Role变异（个性化） + Skill池更新（共享知识）

#### Agent层双进化机制

```
Agent层进化 = Role变异（个性化） + Skill池更新（共享知识）
```

| 维度 | Role变异 | Skill池更新 |
|------|---------|------------|
| 对象 | 4个Agent的role.md | 全局Skill池 |
| 触发频率 | 每3个Epoch | 每3个Epoch（同步） |
| 评估依据 | Agent个体表现 | 经验池整体数据 |
| 变异方式 | top-2保留，bottom-2变异 | 增量/全量更新 |
| 作用范围 | Agent特定 | 所有Agent共享 |

---

#### Prompt 模板化架构（Jinja2 + Markdown + 类 XML）

**核心设计：**

```
Jinja2 模板 (.j2)  ← 框架，定义 Prompt 结构
    ↓ 加载
Markdown Skill (.md)  ← 内容片段，可独立维护
    ↓ 渲染
完整 Prompt（带类 XML 分隔）  ← 最终发送给 LLM
```

**类 XML 分隔示例：**

```jinja2
<!-- SECTION: ROLE [EVOLVABLE] -->
{{ load_agent_config(agent_id, "role") }}
<!-- END SECTION: ROLE -->

<!-- SECTION: FORMAT [STATIC_SKILL] -->
{{ load_skill("static/output_format") }}
<!-- END SECTION: FORMAT -->

<!-- SECTION: STRATEGY [HYBRID] -->
{{ load_skill("static/ml_best_practices") }}
{{ load_agent_config(agent_id, "strategy_explore") }}
<!-- END SECTION: STRATEGY -->

<!-- SECTION: EXAMPLES [DYNAMIC_SKILL] -->
{{ inject_top_k_skills(task_type="explore", k=5) }}
<!-- END SECTION: EXAMPLES -->
```

**优势：**
- ✅ **可追踪性** - 清晰标识每个部分来源
- ✅ **可调试性** - 快速定位问题 Skill
- ✅ **可解析性** - 可用正则提取特定部分
- ✅ **版本控制友好** - Git diff 清晰显示变更

---

#### Prompt 7 层结构

| 层级 | 内容 | 可变性 | 存储方式 | 说明 |
|------|------|-------|---------|------|
| 1. Role | 角色定位 | ✅ 可变 | `agent_configs/{agent_id}/role.md` | Agent 个性，支持涌现分工 |
| 2. Format | 输出格式约束 | ❌ 不可变 | 静态 Skill: `output_format.md` | JSON Schema、代码块标记 |
| 3. Constraints | 硬性约束 | ❌ 不可变 | 静态 Skill: `workspace_rules.md` | 路径规则、安全约束 |
| 4. Task | 任务描述 | ❌ 不可变 | 运行时注入 | 竞赛目标、评估指标 |
| 5. Context | 动态上下文 | 🔄 运行时 | 运行时生成 | 目录树、文件预览、执行历史 |
| 6. Strategy | 策略指导 | ⚡ 部分可变 | 静态 Skill + Agent 特定策略 | 通用策略外置，个性策略可变 |
| 7. Examples | 历史案例 | 🔄 动态 Skill | 从经验池提取 | 成功模式、失败教训 |

---

#### 文件组织结构

```
benchmark/mle-bench/
├── prompt_templates/
│   ├── explore.j2          # Explore 任务主模板
│   ├── merge.j2            # Merge 任务主模板
│   └── mutate.j2           # Mutate 任务主模板
│
├── skills/
│   ├── static/             # 静态 Skill（通用规范）
│   │   ├── output_format.md
│   │   ├── workspace_rules.md
│   │   ├── ml_best_practices.md
│   │   └── code_style.md
│   │
│   ├── by_task_type/       # 按任务类型组织
│   │   ├── explore/
│   │   │   ├── success_patterns/     # 成功模式（动态生成）
│   │   │   └── failure_lessons/      # 失败教训（动态生成）
│   │   ├── merge/
│   │   │   ├── crossover_strategies.md
│   │   │   └── conflict_resolution.md
│   │   └── mutate/
│   │       ├── mutation_strategies.md
│   │       └── local_optimization.md
│   │
│   ├── deprecated/         # 已淘汰Skill
│   └── meta/
│       ├── skill_index.json          # 全局索引
│       ├── skill_lineage.json        # 演化谱系
│       └── update_history.json       # 更新日志
│
└── agent_configs/          # Agent 个性化配置
    ├── agent_0/
    │   ├── role.md                   # 角色定位（可变）
    │   ├── strategy_explore.md       # Explore 策略（可变）
    │   ├── strategy_merge.md         # Merge 策略（可变）
    │   └── strategy_mutate.md        # Mutate 策略（可变）
    ├── agent_1/
    ├── agent_2/
    └── agent_3/
```

---

#### Skill池生成与更新

##### 生命周期

```
初始化（种子知识） → 增量更新 → 质量评估 → 演化（新增/合并/淘汰）
```

##### 初始化策略

| 阶段 | Skill来源 | 数量 | 状态 |
|------|----------|------|------|
| Bootstrap | 预置种子（从AIDE/ML-Master提取） | 5-10个 | seed |
| 首次更新 | 经验池提取（Epoch 3） | +5-8个 | active |
| 成熟期 | 持续演化 | 40-60个（稳定） | active/deprecated |

##### 更新触发机制

混合触发策略：

```
每个Epoch结束：
    if 距离上次更新≥3个Epoch or 新增记录≥50:
        触发Skill池更新
```

##### 提取Pipeline

```
经验池（TaskRecord × N）
    ↓ 按task_type分组
成功案例 / 失败案例
    ↓ 提取strategy_summary
策略文本向量化（Embedding）
    ↓ HDBSCAN聚类（min_cluster_size=5）
策略簇识别
    ↓ LLM总结（生成Skill草稿）
Skill候选
    ↓ 质量评估（覆盖度、成功率、质量增益）
过滤低质量Skill（success_rate < 0.5）
    ↓ 检测重复（语义相似度 > 0.85）
Skill池更新（新增/合并/淘汰）
```

##### 质量评估体系

| 指标 | 计算公式 | 权重 | 作用 |
|------|---------|------|------|
| 覆盖度 | 匹配案例数 | 0.3 | 衡量适用广度 |
| 成功率 | 成功案例数 / 总案例数 | 0.4 | 衡量有效性 |
| 质量增益 | mean(output_quality) | 0.2 | 衡量效果 |
| 新鲜度 | exp(-衰减系数 × 天数) | 0.1 | 衡量时效性 |

综合评分公式：
```
Skill得分 = 0.4 × success_rate
          + 0.3 × log(1 + coverage) / log(10)
          + 0.2 × avg_quality
          + 0.1 × freshness
```

##### 演化机制

| 演化类型 | 触发条件 | 处理方式 |
|---------|---------|---------|
| **新增** | 新聚类簇出现，size≥5 | 创建新Skill |
| **合并** | 语义相似度 > 0.85 | 合并为更通用Skill，保留lineage |
| **分裂** | cluster内部方差过大 | 拆分为细粒度Skill |
| **淘汰** | 连续5 Epoch未匹配 or 成功率<0.4 | 移至deprecated/ |
| **升级** | 种子Skill累积足够验证 | 状态: seed → active |

##### Skill注入策略

动态Top-K选择：

```
对于task_type的任务：
    候选Skill = 过滤(status="active", task_type匹配)

    for Skill in 候选Skill:
        得分 = 综合评分公式

    排序(按得分降序)
    选择Top-5 → 注入Prompt
```

##### 关键优化

| 问题 | 优化策略 | 效果 |
|------|---------|------|
| LLM调用成本高 | 批处理生成 + 缓存 + 阈值过滤（cluster≥10才调用） | 降低50%成本 |
| 更新计算量大 | 增量更新为主，每10 Epoch全量一次 | 快速响应 + 全局优化 |
| Skill质量波动 | 多维度评估 + 自动淘汰低效Skill | 持续优化 |

---

#### Role变异机制

**变异范围**: 只变异可变部分（Role + Strategy），保护不可变部分（Format + Constraints）

**进化流程**:

```
每 3 个 Epoch 结束
    ↓
评估所有 Agent（成功率 × 平均质量）
    ↓
排序：top-2 精英，bottom-2 弱者
    ↓
精英保留（Role + Strategy 不变）
    ↓
弱者变异
    ├─ Role 变异（角色定位调整）
    └─ Strategy 变异（分 task_type 独立变异）
```

**变异指令设计**（元 Prompt）:

```
变异目标：agent_configs/{weak_agent_id}/strategy_explore.md

输入信息：
1. 当前策略文本
2. 精英策略文本（随机选择一个）
3. 该 Agent 在 explore 任务上的表现摘要：
   - 成功率、平均质量
   - Top-3 成功案例的策略描述
   - Top-5 失败案例的错误模式

约束条件：
1. 保持 Markdown 格式
2. 保留成功策略要素
3. 针对失败模式增加规避建议
4. 学习精英策略但保持差异性（diversity）

LLM 生成 → 新策略文本 → 验证渲染 → 写入文件
```

**Role 变异逻辑**:

```
当前角色 + 历史表现 → 涌现式角色定位

示例：
- Agent_0: 成功率 91%（explore）、68%（merge）
  → 进化为"探索者型"：注重创新，偏好大胆尝试

- Agent_2: 成功率 62%（explore）、81%（mutate）
  → 进化为"精化型"：擅长局部优化，倾向保守改进
```

---

#### 双进化协同机制

```
每3个Epoch触发进化：

    [并行执行]
    ├─ 进程1: Role变异
    │   ├─ 评估4个Agent表现（成功率 × 平均质量）
    │   ├─ 排序：top-2精英，bottom-2弱者
    │   ├─ 精英Role保留
    │   └─ 弱者Role变异（LLM生成）
    │
    └─ 进程2: Skill池更新
        ├─ 从经验池提取策略文本
        ├─ 聚类分析（HDBSCAN）
        ├─ LLM生成Skill草稿
        ├─ 质量评估与过滤
        └─ 更新Skill池

    [同步点]
    所有Agent重新加载Skill池 → 下一个Epoch开始
```

协同效果：

| 时间点 | Role状态 | Skill池状态 | 效果 |
|-------|---------|------------|------|
| Epoch 1-3 | 初始Role | 种子Skill | 探索阶段 |
| Epoch 3 | 2个Role变异 | 首次经验Skill | 开始分化 |
| Epoch 6 | 再次变异 | Skill丰富 | 涌现分工 |
| Epoch 9+ | 角色稳定 | Skill成熟 | 高效协作 |

---

#### PromptManager 实现

**核心职责**:
1. 加载静态 Skill
2. 动态生成 Skill（从经验池）
3. 加载 Agent 特定配置（Role + Strategy）
4. 渲染完整 Prompt（基于 Jinja2）
5. 管理Skill池演化

**接口设计**:

```python
class PromptManager:
    def __init__(self, template_dir: Path, skills_dir: Path):
        """初始化 Jinja2 环境和 Skill 目录。"""

    def load_skill(self, skill_path: str) -> str:
        """加载 Skill 文件。

        Args:
            skill_path: 相对于 skills_dir 的路径（如 "static/output_format"）

        Returns:
            Skill 文件内容
        """

    def load_agent_config(self, agent_id: str, section: str) -> str:
        """加载 Agent 配置。

        Args:
            agent_id: Agent ID（如 "agent_0"）
            section: 配置部分（"role" | "strategy_explore" | "strategy_merge" | "strategy_mutate"）

        Returns:
            配置文件内容
        """

    def inject_top_k_skills(
        self,
        task_type: str,
        k: int = 5,
        **filters
    ) -> str:
        """注入 Top-K 动态 Skill（从经验池提取）。

        Args:
            task_type: 任务类型（"explore" | "merge" | "mutate"）
            k: 返回数量
            **filters: 额外过滤条件（如 target_gene="MODEL"）

        Returns:
            拼接后的 Skill 文本
        """

    def build_prompt(
        self,
        task_type: str,  # "explore" | "merge" | "mutate"
        agent_id: str,
        context: Dict,
    ) -> str:
        """渲染完整 Prompt。

        Args:
            task_type: 任务类型
            agent_id: Agent ID
            context: 运行时上下文（task_desc, parent_node, journal, etc.）

        Returns:
            完整 Prompt 文本
        """

    def update_skill_pool(self, experience_pool) -> None:
        """从经验池更新Skill池（提取+评估+演化）。"""

    def mutate_agent_config(self, agent_id: str, section: str, new_content: str) -> None:
        """变异 Agent 配置（供进化算法调用）。"""

    def evaluate_skill_quality(self, skill_id: str) -> float:
        """计算Skill综合评分。"""
```

---

#### 关键优势

| 维度 | 效果 |
|------|------|
| Prompt 长度 | 减少 50%（8000+ → 4000 tokens） |
| 知识复用 | 静态 Skill 共享，避免重复 |
| 经验传承 | 动态 Skill 结构化历史智慧 |
| 进化效率 | 双轨并行（Role+Skill） |
| 可维护性 | 模块化，易于调试和扩展 |
| 自适应性 | Skill池自动演化，持续优化 |

---

### 2.5 动态任务分配（Epsilon-Greedy）

```
收到新任务 task_type
        |
   random() < 0.3 ?
   /            \
  YES            NO
  |               |
随机选择        选择 specialization_scores[task_type]
任意 Agent      最高的 Agent（擅长者优先）
```

**擅长度得分更新：**

```python
# 指数移动平均
alpha = 0.3
agent.specialization_scores[task_type] = (
    (1 - alpha) * agent.specialization_scores[task_type]
    + alpha * task_quality
)
```

---

## 3. 第二部分：Solution 层遗传算法

### 3.1 种群与基因定义

**种群规模**: 12 个 solution.py

**基因结构**: 每个 solution.py 由 7 个基因块通过注释标签标识：

| 基因块 | 标签 | 说明 | 典型内容 |
|--------|------|------|---------|
| DATA | `# [SECTION: DATA]` | 数据处理 | 加载、预处理、增强 |
| MODEL | `# [SECTION: MODEL]` | 模型架构 | Backbone、Head、层配置 |
| LOSS | `# [SECTION: LOSS]` | 损失函数 | CrossEntropy、Focal、组合 |
| OPTIMIZER | `# [SECTION: OPTIMIZER]` | 优化策略 | Adam/SGD、学习率调度 |
| REGULARIZATION | `# [SECTION: REGULARIZATION]` | 正则化 | Dropout、权重衰减、BatchNorm |
| INITIALIZATION | `# [SECTION: INITIALIZATION]` | 初始化 | He/Xavier、预训练权重 |
| TRAINING_TRICKS | `# [SECTION: TRAINING_TRICKS]` | 训练技巧 | 混合精度、梯度裁剪、EMA |

**DATA 基因特殊处理**:
- 内部通过注释标注 `[FIXED]` 和 `[EVOLVABLE]` 两个区域
- `[FIXED]`: 数据划分逻辑（train/test split），保证实验可比性
- `[EVOLVABLE]`: 数据加载和增强逻辑，可自由修改
- **约束方式**: 通过 Prompt 指导 LLM 不要修改 `[FIXED]` 区域（而非代码层面强制）

**示例代码：**

```python
# [SECTION: DATA]

# [FIXED] - Do not modify this region
# This ensures consistent train/test split across all experiments
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("input/train.csv")
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["target"]
)

# [EVOLVABLE] - You can modify below
# Data preprocessing and feature engineering
train_df["feature_1"] = train_df["col_a"] * train_df["col_b"]
train_df = train_df.fillna(train_df.mean())

X_train = train_df.drop("target", axis=1)
y_train = train_df["target"]

# [END SECTION: DATA]
```

---

### 3.2 进化流程（单代）

```
当前种群 (12 个体)
        |
   [1] 精英保留 -----> top-3 直接进入下一代
        |
   [2] 锦标赛选择 ---> 从剩余中选出父代对 (tournament_k=3)
        |
   [3] 基因交叉 -----> merge 任务：按 gene_plan 交叉
        |
   [4] 基因变异 -----> mutate 任务：20% 概率变异单个基因块
        |
   [5] 并行评估 -----> ParallelEvaluator 执行所有新个体
        |
   [6] 适者生存 -----> 合并精英 + 新个体，截断到 12
        |
下一代种群 (12 个体)
```

---

### 3.3 关键操作详解

#### 精英保留

```python
elites = sorted(population, key=lambda x: x.fitness, reverse=True)[:3]
# 精英直接进入下一代，不参与交叉变异
```

---

#### 锦标赛选择

```python
def tournament_select(population, k=3):
    """从种群中随机抽取 k 个，返回最优者。"""
    candidates = random.sample(population, k)
    return max(candidates, key=lambda x: x.fitness)

# 生成父代对
parent_pairs = [
    (tournament_select(population), tournament_select(population))
    for _ in range(num_offspring)
]
```

---

#### 基因交叉（Merge 任务）

```
Parent A: [DATA_a, MODEL_a, LOSS_a, OPT_a, REG_a, INIT_a, TRICK_a]
Parent B: [DATA_b, MODEL_b, LOSS_b, OPT_b, REG_b, INIT_b, TRICK_b]
            |
     [Step 1] 生成 gene_plan:
     {"DATA": "A", "MODEL": "B", "LOSS": "A", "OPTIMIZER": "B",
      "REGULARIZATION": "A", "INITIALIZATION": "B", "TRAINING_TRICKS": "A"}
            |
     [Step 2] Merge Agent 执行:
     - 输入: Parent A + Parent B + gene_plan
     - Prompt: 指导 LLM 按 gene_plan 选择基因块
     - LLM 任务: 解决命名冲突，生成完整可运行的 solution.py
            |
Child:   [DATA_a, MODEL_b, LOSS_a, OPT_b, REG_a, INIT_b, TRICK_a]
```

**交叉约束（通过 Prompt 实现）：**

```markdown
## Merge Prompt 关键约束

1. **按 gene_plan 严格选择基因块**
   - 示例: "MODEL": "A" → 使用 Parent A 的 MODEL 部分

2. **DATA 的 [FIXED] 区域特殊处理**
   - ⚠️ **始终使用 Parent A 的 [FIXED] 区域**（即使 gene_plan 说用 B）
   - 确保所有实验的数据划分一致

3. **解决命名冲突**
   - 如果两个父代使用不同变量名，统一命名并更新引用
   - 确保合成后的代码语法正确

4. **保留基因块边界**
   - 保持所有 `# [SECTION: ...]` 标记
```

**实现说明：**
- 使用 `parse_solution_genes()` 简单提取父代基因块（返回 dict[str, str]）
- 约束完全由 Prompt 控制，无需代码层面的嵌套解析

---

#### 基因变异（Mutate 任务）

```
20% 概率触发变异
        |
随机选择 1 个基因块 (如 MODEL)
        |
Mutate Agent 执行:
  - 输入: 当前代码 + target_gene="MODEL"
  - 输出: 改进后的完整 solution.py
```

**变异约束（通过 Prompt 实现）：**

```markdown
## Mutate Prompt 关键约束

⚠️ **CRITICAL:** You MUST follow these rules:

1. **只修改指定的基因块**
   - 目标: `[SECTION: {{ target_gene }}]`
   - 可以完全重写该基因块内的逻辑

2. **其他基因块保持不变**
   - 所有其他 SECTION（DATA, MODEL, LOSS, etc.）**必须原样保留**
   - 不要改动任何其他部分的代码

3. **DATA 的 [FIXED] 区域特殊处理**
   - 即使你看到 DATA 部分，**绝对不要修改 [FIXED] 区域**
   - 数据划分必须保持一致

4. **保留基因块边界**
   - 保持所有 `# [SECTION: ...]` 标记
   - 确保输出是完整可运行的 Python 代码
```

**实现说明：**
- LLM 看到完整代码上下文，更容易理解各部分关系
- 约束完全由 Prompt 控制，简化代码实现

---

### 3.4 适应度与评估

| 项目 | 说明 |
|------|------|
| **fitness** | `metric_value`（越大越好；若原始指标是 loss，评估层统一取反） |
| **buggy 个体** | `fitness = -1e9`（自然淘汰） |
| **评估方式** | ParallelEvaluator 并行执行 solution.py，解析 metric 输出 |
| **超时处理** | 超时标记为 buggy |

---

### 3.5 基因解析器（简单提取）

**设计原则：简单提取 + Prompt 约束**

```python
import re
from typing import Dict

def parse_solution_genes(code: str) -> Dict[str, str]:
    """解析 solution.py 的 7 个基因块（简单提取）。

    提取每个基因块的完整代码，不做嵌套解析。
    对于 DATA 的 [FIXED] 区域等约束，完全由 Prompt 控制。

    Args:
        code: solution.py 完整代码

    Returns:
        字典，键为基因块名称，值为完整基因块代码
        {"DATA": "# [SECTION: DATA]\n...\n", "MODEL": "# [SECTION: MODEL]\n...\n", ...}

    示例:
        >>> code = '''
        ... # [SECTION: DATA]
        ... import pandas as pd
        ... data = pd.read_csv("train.csv")
        ...
        ... # [SECTION: MODEL]
        ... model = nn.Sequential(...)
        ... '''
        >>> genes = parse_solution_genes(code)
        >>> print(genes["DATA"])
        # [SECTION: DATA]
        import pandas as pd
        data = pd.read_csv("train.csv")
    """
    sections = {}

    # 正则匹配所有 SECTION 标签
    pattern = r'# \[SECTION: (\w+)\]'
    matches = list(re.finditer(pattern, code))

    for i, match in enumerate(matches):
        section_name = match.group(1)
        start = match.start()

        # 下一个 SECTION 的起始位置（或代码结尾）
        end = matches[i + 1].start() if i + 1 < len(matches) else len(code)

        # 提取完整基因块代码（包含 SECTION 标记）
        sections[section_name] = code[start:end].strip()

    return sections
```

**为什么选择简单提取？**

| 维度 | 嵌套提取（复杂） | 简单提取（推荐） |
|------|----------------|----------------|
| **实现复杂度** | 高（需要嵌套正则） | 低（简单正则即可） |
| **数据结构** | 复杂（需要 GeneBlock 类） | 简单（dict[str, str]） |
| **LLM 上下文** | 碎片化（只看子区域） | 完整（看到全部基因块） |
| **约束灵活性** | 低（硬编码在代码中） | 高（Prompt 动态调整） |
| **维护成本** | 高（新约束需改代码） | 低（只改 Prompt） |
| **错误风险** | 高（解析可能失败） | 低（LLM 自然理解） |

**使用示例：**

```python
# 在 Merge Agent 中使用
parent_a_genes = parse_solution_genes(parent_a.code)
parent_b_genes = parse_solution_genes(parent_b.code)

# 根据 gene_plan 选择基因块
gene_plan = {"DATA": "A", "MODEL": "B", "LOSS": "A", ...}
selected_genes = {
    name: parent_a_genes[name] if source == "A" else parent_b_genes[name]
    for name, source in gene_plan.items()
}

# 构建 Merge Prompt（包含约束）
prompt = build_merge_prompt(
    parent_a=parent_a,
    parent_b=parent_b,
    gene_plan=gene_plan,
    # Prompt 中会约束 DATA 的 [FIXED] 区域
)

# LLM 生成合成后的代码
merged_code = llm.generate(prompt)
```

---

### 3.6 Prompt 约束策略（核心设计）

**设计哲学：约束在 Prompt，而非代码**

本系统的基因操作约束（如"只修改某个基因块"、"不修改 DATA 的 FIXED 区域"）完全通过 Prompt 实现，而非代码层面的强制解析。

**优势分析：**

| 维度 | 代码层面约束 | Prompt 层面约束（本设计） |
|------|-------------|------------------------|
| **实现复杂度** | 高（需要嵌套解析、AST 操作） | 低（只需构建清晰的 Prompt） |
| **灵活性** | 低（新约束需要改代码） | 高（只需调整 Prompt 模板） |
| **LLM 理解** | 差（只看到代码片段） | 好（看到完整上下文） |
| **调试难度** | 高（解析错误难定位） | 低（直接检查 LLM 输出） |
| **可维护性** | 差（代码耦合度高） | 好（Prompt 与代码解耦） |

**核心 Prompt 约束示例：**

#### Mutate 任务约束

```markdown
⚠️ **CRITICAL CONSTRAINTS:**

1. **Target Section:** Only modify `[SECTION: {{ target_gene }}]`
2. **Other Sections:** Keep ALL other sections exactly as they are
3. **DATA [FIXED] Region:** Never modify the `[FIXED]` region in DATA section
4. **Section Boundaries:** Preserve all `# [SECTION: ...]` markers
5. **Output:** Return the complete solution.py (not just the modified part)
```

#### Merge 任务约束

```markdown
⚠️ **CRITICAL CONSTRAINTS:**

1. **Gene Plan:** Strictly follow the gene selection plan
   - Example: "MODEL": "A" → Use MODEL section from Parent A

2. **DATA [FIXED] Special Rule:**
   - ⚠️ Always use Parent A's [FIXED] region (ignore gene plan for this part)
   - Ensures consistent data split across all experiments

3. **Naming Conflicts:** Resolve variable name conflicts and update references
4. **Completeness:** Output must be a complete, runnable solution.py
```

#### Explore 任务约束

```markdown
## Constraints

1. **Follow Section Structure:** Use the 7-section template:
   - DATA, MODEL, LOSS, OPTIMIZER, REGULARIZATION, INITIALIZATION, TRAINING_TRICKS

2. **DATA [FIXED] Region:** Must include a `[FIXED]` region for data split
   ```python
   # [FIXED] - Do not modify in future mutations
   train_df, test_df = train_test_split(..., random_state=42)
   ```

3. **Section Markers:** Include all `# [SECTION: ...]` boundaries
```

**失败案例与 Prompt 改进：**

| 失败模式 | 原 Prompt 问题 | 改进方案 |
|---------|--------------|---------|
| LLM 修改了其他基因块 | 约束不够明确 | 添加 "⚠️ CRITICAL" 标记，用 markdown 强调 |
| LLM 删除了 [FIXED] 区域 | 未说明后果 | 解释原因："确保数据划分一致性" |
| LLM 只返回修改部分 | 未明确输出格式 | 明确要求 "Return the **complete** solution.py" |
| LLM 破坏了基因块边界 | 未强调保留标记 | 添加约束："Preserve all `# [SECTION: ...]` markers" |

**Prompt 验证流程：**

```python
def validate_gene_constraints(original_code: str, modified_code: str, task_type: str) -> bool:
    """验证 LLM 输出是否符合基因操作约束。

    这是一个轻量级的后验检查，主要依赖 Prompt 的前验约束。
    """
    # 1. 检查基因块标记是否完整
    original_sections = set(re.findall(r'# \[SECTION: (\w+)\]', original_code))
    modified_sections = set(re.findall(r'# \[SECTION: (\w+)\]', modified_code))

    if original_sections != modified_sections:
        log_msg("WARNING", f"基因块标记不一致: {original_sections} vs {modified_sections}")
        return False

    # 2. 对于 Mutate 任务，检查目标基因块是否真的被修改
    if task_type == "mutate":
        # 简单检查：目标基因块的代码是否有变化
        # 详细验证交给 AST 解析（可选）
        pass

    return True
```

---

## 4. 第三部分：两层协同机制

### 4.1 协同数据流

```
+------------------+                    +--------------------+
|   Agent 层 (4)   |                    |  Solution 层 (12)   |
|                  |   Agent 执行任务    |                    |
|  Agent_0 --------+---> explore ------>+-> solution_new     |
|  Agent_1 --------+---> merge -------->+-> solution_child   |
|  Agent_2 --------+---> mutate ------->+-> solution_mutated |
|  Agent_3         |                    |                    |
|                  |   Solution 反馈    |                    |
|  experience_pool <--------------------+-- fitness_delta    |
|  prompt_evolve   <--------------------+-- success/failure  |
+------------------+                    +--------------------+
```

---

### 4.2 正反馈循环

```
[1] Agent 执行任务 (explore/merge/mutate)
         |
[2] 产出/改进 Solution
         |
[3] Solution 评估得到 fitness
         |
[4] fitness 变化写入经验池 (TaskRecord)
         |
[5] 经验池数据影响:
    +--- Agent 擅长度得分更新 (即时)
    +--- Prompt 进化的评估依据 (每 3 Epoch)
         |
[6] 更好的 Prompt / 更优的任务分配
         |
[7] 回到 [1]，产出更好的 Solution
```

---

### 4.3 Epoch 内工作流

一个 Epoch 的完整执行流程：

```
Epoch N 开始
    |
[1] 初始化/继承 Solution 种群 (首次: 由 explore Agent 并行生成)
    |
[2] 并行评估当前种群 -> 得到 fitness
    |
[3] Solution 层进化:
    a. 精英保留 top-3
    b. 锦标赛选择父代对
    c. Merge Agent 执行基因交叉
    d. Mutate Agent 执行基因变异 (20% 概率)
    e. Orchestrator 对所有新节点执行 Review
    |
[4] 并行评估新种群 -> 更新 fitness
    |
[5] 合并精英 + 新个体 -> 截断到 12
    |
[6] 所有 Agent 执行结果写入经验池
    |
[7] 若 N % 3 == 0: 触发 Agent 层 Prompt 进化
    |
Epoch N 结束
```

---

### 4.4 Journal 生命周期与全局最优追踪

**关键设计原则：**

```
✅ Journal 在整个程序运行期间永不重置
✅ Orchestrator.best_node 持续追踪全局最优（跨 epoch）
✅ Epoch 只是时间分段，用于触发进化，不影响历史记录
```

**实现方式：**

```python
# main.py 或 Orchestrator
journal = Journal()  # ✅ 初始化一次，全局共享

for epoch in range(num_epochs):
    log_msg("INFO", f"===== Epoch {epoch} 开始 =====")

    for step in range(steps_per_epoch):
        node = agent.generate(...)
        journal.append(node)  # ✅ 持续累积，不重置
        orchestrator._update_best_node(node)  # ✅ 更新全局最优

    # 每 3 个 epoch 进化一次
    if epoch % 3 == 0:
        agent_evolution.evolve()  # 基于 journal 全部历史
        solution_evolution.step()

    log_msg("INFO", f"Epoch {epoch} 当前最佳: {orchestrator.best_node.metric_value}")

# 程序结束时，orchestrator.best_node 指向全局最优（跨所有 epoch）
```

**为什么这样设计？**

| 问题 | 解决方案 |
|------|---------|
| 如果 Journal 每个 epoch 重置，会丢失全局最优 | ✅ Journal 永不重置，累积所有历史 |
| 全局最优可能出现在早期 epoch | ✅ Orchestrator.best_node 始终指向历史最佳 |
| 内存占用问题（长期运行） | 可选：定期归档旧节点到磁盘 |

---

### 4.5 关键约束

| 约束 | 原因 | 实现方式 |
|------|------|---------|
| 经验池写入实时 | Agent 立即获得最新反馈 | 每次任务完成后同步写入 |
| Prompt 进化延迟 | 需要足够样本量 | 每 3 Epoch 批量评估 |
| fitness 单调化 | 遗传算法需统一比较方向 | 评估层统一转换为"越大越好" |
| DATA_SPLIT 固定 | 实验可比性 | 基因解析器识别 FIXED 标签 |
| 并发安全 | 多 Agent 并行写入经验池 | 线程安全的经验池实现 |
| Journal 永不重置 | 追踪全局最优 | Orchestrator 全局管理 |

---

## 5. 文件清单

### 5.1 新建文件 [NEW]

| 文件 | 职责 |
|------|------|
| `core/evolution/__init__.py` | 进化机制子系统入口 |
| `core/evolution/experience_pool.py` | 共享经验池（TaskRecord 存储与查询） |
| `core/evolution/agent_evolution.py` | Agent 层进化（Prompt 进化 + 擅长度更新） |
| `core/evolution/solution_evolution.py` | Solution 层遗传算法（选择/交叉/变异/精英） |
| `core/evolution/gene_parser.py` | 基因解析器（解析 solution.py 的 7 个基因块） |
| `core/evolution/task_dispatcher.py` | 动态任务分配（epsilon-greedy） |
| `utils/prompt_manager.py` | **Prompt 管理器（Jinja2 + Markdown Skill）** |
| `core/evolution/skill_extractor.py` | Skill 提取器（经验池聚类 + LLM生成） |
| `core/evolution/skill_manager.py` | Skill 池管理器（质量评估 + 演化 + 索引） |
| `search/parallel_evaluator.py` | 并行评估器（ThreadPoolExecutor + FIRST_COMPLETED） |
| `search/fitness.py` | 适应度计算与单调化 |
| `tests/test_evolution/test_experience_pool.py` | 经验池单元测试 |
| `tests/test_evolution/test_agent_evolution.py` | Agent 进化单元测试 |
| `tests/test_evolution/test_solution_evolution.py` | Solution 遗传算法单元测试 |
| `tests/test_evolution/test_gene_parser.py` | 基因解析器单元测试 |
| `tests/test_evolution/test_prompt_manager.py` | Prompt 管理器单元测试 |
| `tests/test_evolution/test_skill_extractor.py` | Skill 提取器单元测试 |
| `tests/test_evolution/test_skill_manager.py` | Skill 管理器单元测试 |
| `tests/test_search/test_parallel_evaluator.py` | 并行评估器单元测试 |
| `benchmark/mle-bench/prompt_templates/explore.j2` | Explore 任务 Jinja2 模板 |
| `benchmark/mle-bench/prompt_templates/merge.j2` | Merge 任务 Jinja2 模板 |
| `benchmark/mle-bench/prompt_templates/mutate.j2` | Mutate 任务 Jinja2 模板 |
| `benchmark/mle-bench/skills/static/output_format.md` | 静态 Skill: 输出格式 |
| `benchmark/mle-bench/skills/static/workspace_rules.md` | 静态 Skill: 工作空间规则 |
| `benchmark/mle-bench/skills/static/ml_best_practices.md` | 静态 Skill: ML 最佳实践 |
| `benchmark/mle-bench/skills/static/code_style.md` | 静态 Skill: 代码风格 |
| `benchmark/mle-bench/skills/by_task_type/merge/crossover_strategies.md` | Merge 策略 Skill |
| `benchmark/mle-bench/skills/by_task_type/merge/conflict_resolution.md` | 命名冲突解决 Skill |
| `benchmark/mle-bench/skills/by_task_type/mutate/mutation_strategies.md` | Mutate 策略 Skill |
| `benchmark/mle-bench/skills/by_task_type/mutate/local_optimization.md` | 局部优化 Skill |
| `benchmark/mle-bench/skills/meta/skill_index.json` | Skill 全局索引（id、评分、状态） |
| `benchmark/mle-bench/skills/meta/skill_lineage.json` | Skill 演化谱系（合并/分裂历史） |
| `benchmark/mle-bench/skills/meta/update_history.json` | Skill 更新日志 |

---

### 5.2 修改文件 [MODIFY]

| 文件 | 修改内容 |
|------|---------|
| `agents/base_agent.py` | [MODIFY] `AgentContext.task_type` 改为 `Literal["explore", "merge", "mutate"]` |
| `core/state/journal.py` | [NEW] 新增 `get_best_k(k: int, only_good: bool = True) -> list[Node]` 方法 |
| `core/orchestrator.py` | [MODIFY] 接入双层进化调度，委托给 AgentEvolution + SolutionEvolution |
| `utils/config.py` | [MODIFY] 新增 `evolution` 配置区（Agent 层 + Solution 层参数） |
| `utils/prompt_builder.py` | [MODIFY] 改为加载 `PromptManager`，调用其 `build_prompt()` 方法 |

---

## 6. 配置文件更新

### 6.1 `config/default.yaml` 新增 Evolution 配置区

```yaml
# ============================================================
# 进化算法配置（Phase 3）
# ============================================================
evolution:
  # Agent 层群体智能
  agent:
    population_size: 4           # Agent 数量
    evolve_interval: 3           # 每 N 个 epoch 进化一次
    epsilon: 0.3                 # Epsilon-Greedy 探索率
    specialization_alpha: 0.3    # 擅长度得分更新率（指数移动平均）
    elite_count: 2               # 精英保留数量（top-2）

  # Solution 层遗传算法
  solution:
    population_size: 12          # Solution 种群大小
    elite_size: 3                # 精英保留数量
    mutation_rate: 0.2           # 变异概率
    tournament_k: 3              # 锦标赛选择参数
    crossover_rate: 0.8          # 交叉概率

  # Epoch 控制
  epochs: 10                     # 总 epoch 数量
  steps_per_epoch: 5             # 每个 epoch 步数

  # 经验池
  experience_pool:
    max_size: 1000               # 最大记录数
    top_k_inject: 5              # Prompt 注入 Top-K 经验
    min_cluster_size: 5          # Skill 提取最小簇大小
    similarity_threshold: 0.85   # Skill 合并相似度阈值

  # Prompt 模板路径
  prompt:
    template_dir: "benchmark/mle-bench/prompt_templates"
    skills_dir: "benchmark/mle-bench/skills"
    agent_configs_dir: "benchmark/mle-bench/agent_configs"
```

---

## 7. 验证计划

### 7.1 单元测试

```bash
# 经验池
conda run -n Swarm-Evo pytest tests/test_evolution/test_experience_pool.py -v

# Agent 进化
conda run -n Swarm-Evo pytest tests/test_evolution/test_agent_evolution.py -v

# Solution 遗传算法
conda run -n Swarm-Evo pytest tests/test_evolution/test_solution_evolution.py -v

# 基因解析器
conda run -n Swarm-Evo pytest tests/test_evolution/test_gene_parser.py -v

# Prompt 管理器
conda run -n Swarm-Evo pytest tests/test_evolution/test_prompt_manager.py -v

# Skill 提取器
conda run -n Swarm-Evo pytest tests/test_evolution/test_skill_extractor.py -v

# Skill 管理器
conda run -n Swarm-Evo pytest tests/test_evolution/test_skill_manager.py -v

# 并行评估器
conda run -n Swarm-Evo pytest tests/test_search/test_parallel_evaluator.py -v
```

---

### 7.2 集成验证

```bash
# 运行完整双层进化（最小配置）
conda run -n Swarm-Evo python main.py \
  --evolution.agent.population_size=4 \
  --evolution.solution.population_size=8 \
  --evolution.epochs=6 \
  --evolution.agent.evolve_interval=3
```

**预期**:
- ✅ 经验池有写入记录（`log_json` 输出 TaskRecord）
- ✅ Agent 在第 3、6 Epoch 触发 Prompt 进化
- ✅ Skill池在第 3、6 Epoch 触发更新
- ✅ Solution 种群 fitness 呈上升趋势
- ✅ 日志记录 Agent 擅长度得分变化
- ✅ Skill池生成新Skill并记录在 `skill_index.json`
- ✅ Skill质量评估输出（覆盖度、成功率、质量增益）
- ✅ Journal 持续累积，不重置
- ✅ Orchestrator.best_node 追踪全局最优

**Skill池验证**:
- [ ] Skill池正确生成和更新
- [ ] Skill质量评估有效（评分公式计算正确）
- [ ] Skill注入Prompt后效果提升（对比实验）
- [ ] 演化机制正确（新增/合并/淘汰/升级）
- [ ] Top-K注入策略有效（高质量Skill优先）

**Journal 验证**:
- [ ] Journal 在所有 epoch 持续累积
- [ ] `orchestrator.best_node` 始终指向全局最优
- [ ] `journal.get_best_k(k=3)` 返回正确的 Top-3 节点

---

### 7.3 覆盖率

```bash
conda run -n Swarm-Evo pytest tests \
  --cov=core/evolution \
  --cov=search \
  --cov=utils/prompt_manager \
  --cov-report=term-missing
```

**目标覆盖率**: **80%+**

---

## 8. 风险与缓解

| 风险 | 等级 | 缓解措施 |
|------|------|---------|
| 基因交叉产出不可运行代码 | 高 | AST 验证 + buggy 自然淘汰 + debug 流程 |
| Agent 涌现分工不收敛 | 中 | 30% 随机探索 + 最小分化阈值 |
| 经验池并发写入竞争 | 中 | 线程安全数据结构 (threading.Lock) |
| Prompt 进化效果不明显 | 中 | 记录进化前后的 Agent 表现对比日志 |
| 种群多样性丧失 | 中 | 变异率 20% + explore 任务保证新鲜血液 |
| 并行评估 submission 冲突 | 高 | WorkspaceManager 强制 node_id 后缀 |
| Skill提取质量不高 | 中 | 人工审核样本 + 调整聚类参数（min_cluster_size） |
| Skill池过度膨胀 | 中 | 定期清理 + 严格淘汰标准（连续5 Epoch未匹配） |
| Skill语义相似度计算误差 | 低 | 使用robust embedding模型 + 阈值敏感性测试 |
| LLM生成Skill不稳定 | 中 | 多样本生成取consensus + 格式验证 + 质量过滤 |
| Journal 内存占用过大 | 低 | 可选：定期归档旧节点到磁盘（保留最近 N 个 epoch） |

---

## 9. 实施建议

### 9.1 开发顺序（推荐）

| 阶段 | 模块 | 优先级 | 预计工时 |
|------|------|--------|---------|
| **1. 基础设施** | `gene_parser.py` | P0 | 4h |
| | `experience_pool.py` | P0 | 6h |
| | `utils/prompt_manager.py` | P0 | 8h |
| | 更新 `config/default.yaml` | P0 | 1h |
| | 更新 `AgentContext.task_type` | P0 | 0.5h |
| | 新增 `Journal.get_best_k()` | P0 | 1h |
| **2. Agent 层** | `agent_evolution.py` | P1 | 10h |
| | `task_dispatcher.py` | P1 | 4h |
| | `skill_extractor.py` | P1 | 8h |
| | `skill_manager.py` | P1 | 6h |
| **3. Solution 层** | `solution_evolution.py` | P1 | 12h |
| | `search/parallel_evaluator.py` | P1 | 6h |
| | `search/fitness.py` | P1 | 2h |
| **4. 集成** | 更新 `orchestrator.py` | P1 | 8h |
| | 创建 Prompt 模板（.j2） | P1 | 6h |
| | 创建 Skill 文件（.md） | P1 | 4h |
| **5. 测试** | 单元测试（8 个文件） | P2 | 16h |
| | 集成测试 | P2 | 8h |

**总计**: 约 110 小时（~14 工作日，单人）

---

### 9.2 里程碑

| 里程碑 | 完成标志 | 时间点 |
|--------|---------|--------|
| **M1: 基础设施** | PromptManager + ExperiencePool + Config 完成 | Day 3 |
| **M2: Agent 层** | Agent 进化 + Skill 池可运行 | Day 7 |
| **M3: Solution 层** | 遗传算法 + 并行评估可运行 | Day 10 |
| **M4: 双层集成** | 完整流程跑通（单 epoch） | Day 12 |
| **M5: 测试完成** | 覆盖率 80%+ | Day 14 |

---

### 9.3 验收标准

- [ ] **所有单元测试通过**（覆盖率 80%+）
- [ ] **集成测试通过**（最小配置运行 6 个 epoch）
- [ ] **Journal 持续累积**（不重置，追踪全局最优）
- [ ] **经验池正常写入**（每次任务后记录 TaskRecord）
- [ ] **Agent 进化触发**（第 3、6 epoch 触发 Role 变异）
- [ ] **Skill 池更新**（第 3、6 epoch 触发 Skill 提取）
- [ ] **Solution 种群进化**（fitness 呈上升趋势）
- [ ] **Prompt 模板正确渲染**（类 XML 分隔清晰）
- [ ] **3 种任务类型正常执行**（explore, merge, mutate）
- [ ] **基因解析正确**（7 个基因块完整提取）

---

## 10. 附录

### 10.1 关键术语表

| 术语 | 说明 |
|------|------|
| **Epoch** | 进化算法的一个完整迭代周期 |
| **Agent 层** | 元学习层，优化"如何设计方案" |
| **Solution 层** | 直接优化层，优化方案本身性能 |
| **Experience Pool** | 共享经验池，记录所有 Agent 执行历史 |
| **Skill** | 可复用的 Prompt 片段（静态/动态） |
| **Role** | Agent 的角色定位（可进化） |
| **Task Type** | 任务类型（explore, merge, mutate） |
| **Gene Block** | 基因块（solution.py 的 7 个模块） |
| **Fitness** | 适应度值（metric_value） |
| **Elite** | 精英个体（fitness 最高的个体） |

---

### 10.2 相关文档

| 文档 | 路径 |
|------|------|
| 架构概览 | `docs/CODEMAPS/architecture.md` |
| 后端模块 | `docs/CODEMAPS/backend.md` |
| Phase 1 计划 | `docs/plans/phase1_infrastructure.md` |
| Phase 2 计划 | `docs/plans/phase2_core.md` |
| 开发规范 | `CLAUDE.md` |

---

**文档版本**: v2.0
**更新日期**: 2026-01-31
**下一步**: 执行 Phase 3.1 - 基础设施模块实现
