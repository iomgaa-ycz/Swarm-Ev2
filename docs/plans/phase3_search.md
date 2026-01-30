# Phase 3: 双层群体智能实现

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
| `task_history` | `list[TaskRecord]` | 历史执行记录 |
| `specialization_scores` | `dict[str, float]` | 各任务类型的擅长度得分 |
| `generation` | `int` | 当前所属代数 |

**任务类型：**

| 任务 | 说明 | 输入 | 输出 |
|------|------|------|------|
| `explore` | 从零生成新方案 | 任务描述 + 经验池摘要 | 完整 solution.py |
| `select` | 评估并选择优秀基因 | 候选方案集合 | 基因选择计划 (gene_plan) |
| `merge` | 交叉合成新方案 | 父代基因 + gene_plan | 合成后的 solution.py |
| `review` | 审查并改进方案 | 现有方案 + 评估结果 | 改进后的 solution.py |

### 2.2 共享经验池（Experience Pool）

经验池是 Agent 层隐式协作的核心数据结构。所有 Agent 向同一个池写入，从同一个池读取。

```
Experience Pool
+--------------------------------------------------+
| TaskRecord                                        |
|   agent_id: str       # 执行者                     |
|   task_type: str      # explore/select/merge/review|
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

### 2.3 Prompt 进化

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

#### Prompt 组织架构

使用 **Jinja2 主模板** + **XML 注释分隔** + **模块化加载**：

```jinja2
<!-- main_prompt.j2 -->
<!-- SECTION: ROLE [EVOLVABLE] -->
{{ load_markdown(f"agent_configs/{agent_id}/role.md") }}

<!-- SECTION: FORMAT [STATIC_SKILL] -->
{{ load_skill("static/output_format.md") }}

<!-- SECTION: CONSTRAINTS [STATIC_SKILL] -->
{{ load_skill("static/workspace_rules.md") }}

<!-- SECTION: TASK [RUNTIME] -->
{{ task_description }}

<!-- SECTION: CONTEXT [RUNTIME] -->
{{ render_context(directory_tree, file_previews, device_info, ...) }}

<!-- SECTION: STRATEGY [HYBRID] -->
{{ load_skill("static/ml_best_practices.md") }}
{{ load_markdown(f"agent_configs/{agent_id}/strategy_{task_type}.md") }}

<!-- SECTION: EXAMPLES [DYNAMIC_SKILL] -->
{{ inject_top_k_skills(task_type, k=5) }}
```

#### 文件组织结构

```
benchmark/mle-bench/
├── prompt_templates/
│   └── main_prompt.j2                        # 主模板框架
├── skills/
│   ├── static/                              # 静态 Skill（通用规范）
│   │   ├── output_format.md
│   │   ├── workspace_rules.md
│   │   └── ml_best_practices.md
│   ├── by_task_type/                         # 按任务类型组织
│   │   ├── explore/
│   │   │   ├── success_patterns/             # 成功模式
│   │   │   └── failure_lessons/              # 失败教训
│   │   ├── select/, merge/, review/
│   ├── deprecated/                           # 已淘汰Skill
│   └── meta/
│       ├── skill_index.json                  # 全局索引
│       ├── skill_lineage.json                # 演化谱系
│       └── update_history.json               # 更新日志
└── agent_configs/                           # Agent 个性化配置
    ├── agent_0/
    │   ├── role.md                          # 角色定位（可变）
    │   ├── strategy_explore.md              # Explore 策略（可变）
    │   ├── strategy_select.md
    │   ├── strategy_merge.md
    │   └── strategy_review.md
    ├── agent_1/
    ├── agent_2/
    └── agent_3/
```

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
  → 进化为"守门员型"：注重代码质量，偏好保守策略

- Agent_2: 成功率 62%（explore）、81%（select）
  → 进化为"评审型"：擅长方案评估，倾向精细分析
```

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

#### PromptManager 实现

**核心职责**:
1. 加载静态 Skill
2. 动态生成 Skill（从经验池）
3. 加载 Agent 特定配置（Role + Strategy）
4. 渲染完整 Prompt
5. 管理Skill池演化

**接口设计**:

```python
class PromptManager:
    def load_skill(type: str, name: str) -> str:
        """加载 Skill 文件。type: "static" | "by_task_type" """

    def update_skill_pool(experience_pool) -> None:
        """从经验池更新Skill池（提取+评估+演化）。"""

    def load_agent_config(agent_id: str, section: str) -> str:
        """加载 Agent 配置。section: "role" | "strategy_explore" | ..."""

    def build_prompt(agent_id, task_type, runtime_context) -> str:
        """渲染完整 Prompt（含动态Top-K Skill注入）。"""

    def mutate_agent_config(agent_id, section, new_content) -> None:
        """变异 Agent 配置（供进化算法调用）。"""

    def evaluate_skill_quality(skill_id: str) -> float:
        """计算Skill综合评分。"""
```

#### 关键优势

| 维度 | 效果 |
|------|------|
| Prompt 长度 | 减少 50%（8000+ → 4000 tokens） |
| 知识复用 | 静态 Skill 共享，避免重复 |
| 经验传承 | 动态 Skill 结构化历史智慧 |
| 进化效率 | 双轨并行（Role+Skill） |
| 可维护性 | 模块化，易于调试和扩展 |
| 自适应性 | Skill池自动演化，持续优化 |

### 2.4 动态任务分配（Epsilon-Greedy）

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

**DATA 基因特殊处理**: 内部分为 `[FIXED]`（数据划分，保证可比性）和 `[EVOLVABLE]`（加载/增强逻辑）两个子区域。

### 3.2 进化流程（单代）

```
当前种群 (12 个体)
        |
   [1] 精英保留 -----> top-3 直接进入下一代
        |
   [2] 锦标赛选择 ---> 从剩余中选出父代对 (tournament_k=3)
        |
   [3] 基因交叉 -----> 随机选择每个基因块的来源父代，LLM 合成
        |
   [4] 基因变异 -----> 20% 概率，随机选择 1 个基因块改进
        |
   [5] 并行评估 -----> ParallelEvaluator 执行所有新个体
        |
   [6] 适者生存 -----> 合并精英 + 新个体，截断到 12
        |
下一代种群 (12 个体)
```

### 3.3 关键操作详解

#### 精英保留

```python
elites = sorted(population, key=lambda x: x.fitness, reverse=True)[:3]
# 精英直接进入下一代，不参与交叉变异
```

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

#### 基因交叉（LLM 合成）

```
Parent A: [DATA_a, MODEL_a, LOSS_a, OPT_a, REG_a, INIT_a, TRICK_a]
Parent B: [DATA_b, MODEL_b, LOSS_b, OPT_b, REG_b, INIT_b, TRICK_b]
            |
     随机生成 gene_plan:
     {"DATA": "A", "MODEL": "B", "LOSS": "A", "OPTIMIZER": "B",
      "REGULARIZATION": "A", "INITIALIZATION": "B", "TRAINING_TRICKS": "A"}
            |
     LLM 合成: 按 gene_plan 提取基因块，解决命名冲突，
              生成完整可运行的 solution.py
            |
Child:   [DATA_a, MODEL_b, LOSS_a, OPT_b, REG_a, INIT_b, TRICK_a]
```

**交叉约束：**
- DATA 基因的 `[FIXED]` 区域强制保持一致
- LLM 负责解决跨基因块的命名冲突与兼容性
- 交叉前用 `parse_solution_genes()` 解析父代基因

#### 基因变异

```
20% 概率触发变异
        |
随机选择 1 个基因块 (如 MODEL)
        |
LLM 改进该基因块:
  - 输入: 当前基因块代码 + 评估反馈
  - 约束: 只修改该基因块，其余保持不变
  - 输出: 改进后的完整 solution.py
```

### 3.4 适应度与评估

| 项目 | 说明 |
|------|------|
| **fitness** | `metric_value`（越大越好；若原始指标是 loss，评估层统一取反） |
| **buggy 个体** | `fitness = -1e9`（自然淘汰） |
| **评估方式** | ParallelEvaluator 并行执行 solution.py，解析 metric 输出 |
| **超时处理** | 超时标记为 buggy |

### 3.5 基因解析器

```python
def parse_solution_genes(code: str) -> dict[str, GeneBlock]:
    """
    解析 solution.py 的 7 个基因块。

    返回: {"DATA": GeneBlock(...), "MODEL": GeneBlock(...), ...}
    每个 GeneBlock 包含:
      - section_name: str
      - code: str
      - is_fixed: bool  (仅 DATA 的子区域)
      - start_line: int
      - end_line: int
    """
```

---

## 4. 第三部分：两层协同机制

### 4.1 协同数据流

```
+------------------+                    +--------------------+
|   Agent 层 (4)   |                    |  Solution 层 (12)   |
|                  |   Agent 执行任务    |                    |
|  Agent_0 --------+---> explore ------>+-> solution_new     |
|  Agent_1 --------+---> select ------->+-> gene_plan        |
|  Agent_2 --------+---> merge -------->+-> solution_child   |
|  Agent_3 --------+---> review ------->+-> solution_improved|
|                  |                    |                    |
|                  |   Solution 反馈    |                    |
|  experience_pool <--------------------+-- fitness_delta    |
|  prompt_evolve   <--------------------+-- success/failure  |
+------------------+                    +--------------------+
```

### 4.2 正反馈循环

```
[1] Agent 执行任务 (explore/select/merge/review)
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
    c. Agent(select) 选择基因计划
    d. Agent(merge) 执行基因交叉
    e. 20% 概率基因变异
    f. Agent(review) 审查改进
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

### 4.4 关键约束

| 约束 | 原因 | 实现方式 |
|------|------|---------|
| 经验池写入实时 | Agent 立即获得最新反馈 | 每次任务完成后同步写入 |
| Prompt 进化延迟 | 需要足够样本量 | 每 3 Epoch 批量评估 |
| fitness 单调化 | 遗传算法需统一比较方向 | 评估层统一转换为"越大越好" |
| DATA_SPLIT 固定 | 实验可比性 | 基因解析器识别 FIXED 标签 |
| 并发安全 | 多 Agent 并行写入经验池 | 线程安全的经验池实现 |

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
| `core/evolution/prompt_manager.py` | Prompt 管理（Skill 加载 + 模板渲染 + 变异） |
| `core/evolution/skill_extractor.py` | Skill 提取器（经验池聚类 + LLM生成） |
| `core/evolution/skill_manager.py` | Skill 池管理器（质量评估 + 演化 + 索引） |
| `core/strategies/parallel.py` | 并行评估器（ThreadPoolExecutor + FIRST_COMPLETED） |
| `core/strategies/fitness.py` | 适应度计算与单调化 |
| `tests/test_evolution/test_experience_pool.py` | 经验池单元测试 |
| `tests/test_evolution/test_agent_evolution.py` | Agent 进化单元测试 |
| `tests/test_evolution/test_solution_evolution.py` | Solution 遗传算法单元测试 |
| `tests/test_evolution/test_gene_parser.py` | 基因解析器单元测试 |
| `tests/test_evolution/test_prompt_manager.py` | Prompt 管理器单元测试 |
| `tests/test_evolution/test_skill_extractor.py` | Skill 提取器单元测试 |
| `tests/test_evolution/test_skill_manager.py` | Skill 管理器单元测试 |
| `tests/test_strategies/test_parallel_evaluator.py` | 并行评估器单元测试 |
| `benchmark/mle-bench/prompt_templates/main_prompt.j2` | Jinja2 主模板 |
| `benchmark/mle-bench/skills/static/output_format.md` | 静态 Skill: 输出格式 |
| `benchmark/mle-bench/skills/static/workspace_rules.md` | 静态 Skill: 工作空间规则 |
| `benchmark/mle-bench/skills/static/ml_best_practices.md` | 静态 Skill: ML 最佳实践 |
| `benchmark/mle-bench/skills/meta/skill_index.json` | Skill 全局索引（id、评分、状态） |
| `benchmark/mle-bench/skills/meta/skill_lineage.json` | Skill 演化谱系（合并/分裂历史） |
| `benchmark/mle-bench/skills/meta/update_history.json` | Skill 更新日志 |

### 5.2 修改文件 [MODIFY]

| 文件 | 修改内容 |
|------|---------|
| `core/state/node.py` | [MODIFY] Node 增加 `fitness`, `generation`, `gene_blocks` 字段 |
| `core/state/journal.py` | [NEW] `get_population()`, `get_best_k()` 方法 |
| `core/orchestrator.py` | [MODIFY] 接入双层进化调度，委托给 AgentEvolution + SolutionEvolution |
| `utils/config.py` | [MODIFY] 新增 `evolution` 配置区（Agent 层 + Solution 层参数） |
| `utils/prompt_builder.py` | [NEW] `build_crossover_prompt()`, `build_mutation_prompt()`, `build_explore_prompt()` |
| `agents/base_agent.py` | [MODIFY] 增加 `specialization_scores`, `system_prompt` 可变字段 |

---

## 6. 验证计划

### 6.1 单元测试

```bash
# 经验池
pytest tests/test_evolution/test_experience_pool.py -v

# Agent 进化
pytest tests/test_evolution/test_agent_evolution.py -v

# Solution 遗传算法
pytest tests/test_evolution/test_solution_evolution.py -v

# 基因解析器
pytest tests/test_evolution/test_gene_parser.py -v

# Prompt 管理器
pytest tests/test_evolution/test_prompt_manager.py -v

# 并行评估器
pytest tests/test_strategies/test_parallel_evaluator.py -v
```

### 6.2 集成验证

```bash
# 运行完整双层进化（最小配置）
python main.py \
  --evolution.agent.population_size=4 \
  --evolution.solution.population_size=8 \
  --evolution.epochs=6 \
  --evolution.agent.evolve_interval=3
```

**预期**:
- 经验池有写入记录（`log_json` 输出 TaskRecord）
- Agent 在第 3、6 Epoch 触发 Prompt 进化
- Skill池在第 3、6 Epoch 触发更新
- Solution 种群 fitness 呈上升趋势
- 日志记录 Agent 擅长度得分变化
- Skill池生成新Skill并记录在 `skill_index.json`
- Skill质量评估输出（覆盖度、成功率、质量增益）

**Skill池验证**:
- [ ] Skill池正确生成和更新
- [ ] Skill质量评估有效（评分公式计算正确）
- [ ] Skill注入Prompt后效果提升（对比实验）
- [ ] 演化机制正确（新增/合并/淘汰/升级）
- [ ] Top-K注入策略有效（高质量Skill优先）

### 6.3 覆盖率

```bash
pytest tests --cov=core/evolution --cov=core/strategies --cov-report=term-missing
```

---

## 7. 风险与缓解

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
