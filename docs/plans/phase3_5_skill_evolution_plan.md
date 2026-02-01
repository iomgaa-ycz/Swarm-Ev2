# Phase 3.5 实施方案：Skill池进化与集成

**创建日期**: 2026-02-01
**预计工时**: 22 小时
**前置依赖**: P3.1-P3.4 已完成
**难度等级**: ⭐⭐⭐⭐（高）

---

## 1.1 摘要

Phase 3.5 是 Phase 3 的最后一个子阶段，负责实现 **Skill 池的自动演化与集成机制**。核心目标是从经验池中提取成功策略模式，生成可复用的 Skill 文件，并通过质量评估、新增/合并/淘汰机制维护一个高质量的动态 Skill 池。Skill 池将通过 PromptManager 注入到 Agent 的 Prompt 中，形成"学习-应用-反馈"的闭环。

**核心交付物**:
1. **SkillExtractor** - 经验池聚类 + LLM 总结策略模式
2. **SkillManager** - Skill 质量评估、演化（新增/合并/淘汰）、元数据管理
3. **AgentEvolution 扩展** - 补充 Skill 池更新逻辑
4. **Orchestrator 集成** - 双层进化流程（Agent 层 + Solution 层）
5. **PromptManager 扩展** - Skill 注入、评估、重载方法

---

## 1.2 审查点（User Review Required）

在开始编码前，请确认以下问题：

### 1.2.1 Skill 提取策略
- **聚类算法**: 使用 **HDBSCAN** 进行策略聚类，`min_cluster_size=5`
- **文本向量化**: 使用 **bge-m3** 本地嵌入模型（参考 Reference/Swarm-Evo）
  - 模型: `BAAI/bge-m3`
  - 本地路径: 从环境变量 `LOCAL_MODEL_PATH` 读取，默认 `./embedding-models/bge-m3`
  - 返回 L2 归一化的 embeddings
  - 支持缓存机制
- **LLM 总结**: 使用 `config.llm.code` 配置的模型

### 1.2.2 Skill 质量评估（参考 Reference/Swarm-Evo）
- **综合评分公式**（来自 `version_manager.py`）:
  ```python
  composite_score = 0.6 × avg_accuracy + 0.4 × avg_generation_rate
  ```

  其中：
  - `avg_accuracy`: 使用该 Skill 的所有记录的平均 `output_quality`
  - `avg_generation_rate`: 使用该 Skill 的记录中有效生成的比例（`output_quality > 0`）

### 1.2.3 Skill 演化机制
- **新增条件**: `success_rate >= 0.5` 且未重复（语义相似度 < 0.85）
- **淘汰条件**: 连续 5 Epoch 未使用 **或** `success_rate < 0.4`
- **合并条件**: 语义相似度 > 0.85 时合并为更通用的 Skill
- 是否需要调整这些阈值？

### 1.2.4 Skill 池目录结构
```
benchmark/mle-bench/skills/
├── static/                     # 静态 Skill（已存在，不修改）
├── by_task_type/               # 任务特定 Skill（已存在）
│   ├── merge/
│   ├── mutate/
│   └── explore/                # P3.5 将在此新增动态生成的 Skill
│       └── success_patterns/   # 新增：从经验池提取的成功模式
│           ├── skill_explore_0_1738443200.md
│           └── ...
└── meta/                       # 元数据（P3.5 新增）
    ├── skill_index.json        # Skill 索引
    ├── skill_lineage.json      # Skill 血统（合并历史）
    └── update_history.json     # 更新历史
```
是否需要调整目录结构？

**用户决策**: 以上配置已根据 Reference/Swarm-Evo 调整。集成测试暂不进行。请批准后进入编码阶段。

---

## 1.3 拟议变更（Proposed Changes）

### 1.3.1 新增文件

#### [NEW] `core/evolution/code_embedding_manager.py` (约 90 行)

**职责**: 封装 bge-m3 嵌入模型，提供文本向量化能力（参考 Reference/Swarm-Evo）。

| 函数/类 | 签名 | 说明 |
|---------|------|------|
| `CodeEmbeddingManager.__init__` | `() -> None` | 初始化管理器（懒加载） |
| `embed_texts` | `(texts: List[str]) -> np.ndarray` | 批量文本向量化（L2 归一化） |
| `_ensure_model` | `() -> None` | 懒加载 bge-m3 模型 |

**核心特性**:
- **懒加载**: 首次调用 `embed_texts` 时才加载模型
- **缓存**: 相同文本不重复向量化（内存缓存）
- **归一化**: 返回 L2 归一化的 embeddings（适用于余弦相似度计算）
- **批处理**: 批量编码，batch_size=8

**实现参考** (Reference/Swarm-Evo/core/evolution/embedding_manager.py):
```python
from sentence_transformers import SentenceTransformer
import numpy as np
import os

class CodeEmbeddingManager:
    _model_name = "BAAI/bge-m3"
    _local_model_path = None
    _model = None

    def __init__(self):
        self._cache = {}  # text -> embedding

    @classmethod
    def _ensure_model(cls):
        if cls._model is None:
            model_path = os.environ.get("LOCAL_MODEL_PATH", "./embedding-models/bge-m3")
            if os.path.exists(model_path):
                cls._model = SentenceTransformer(model_path)
            else:
                cls._model = SentenceTransformer(cls._model_name)
            cls._model.eval()

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        # 检查缓存 + 批量编码 + L2 归一化
        ...
```

---

#### [NEW] `core/evolution/skill_extractor.py` (约 200 行)

**职责**: 从经验池中提取成功策略并生成 Skill。

| 函数/类 | 签名 | 说明 |
|---------|------|------|
| `SkillExtractor.__init__` | `(experience_pool: ExperiencePool, config: Config)` | 初始化提取器 |
| `extract_skills` | `(task_type: str, min_cluster_size: int = 5) -> List[Dict]` | 提取 Skill（聚类 + LLM 总结） |
| `_embed_texts` | `(texts: List[str]) -> np.ndarray` | 文本向量化（bge-m3 本地模型） |
| `_cluster` | `(embeddings: np.ndarray, min_cluster_size: int) -> Dict[int, List[int]]` | HDBSCAN 聚类 |
| `_summarize_cluster` | `(strategies: List[str], task_type: str) -> str` | LLM 总结策略簇为 Skill |
| `_calc_avg_accuracy` | `(indices: List[int], records: List[TaskRecord]) -> float` | 计算簇平均准确率 |
| `_calc_generation_rate` | `(indices: List[int], records: List[TaskRecord]) -> float` | 计算簇平均生成率 |

**核心流程**:
1. 从经验池查询 `output_quality > 0` 的记录
2. 提取 `strategy_summary` 并使用 bge-m3 向量化
3. HDBSCAN 聚类（`min_cluster_size=5`）
4. 每个簇调用 LLM 总结生成 Skill Markdown
5. 返回 Skill 列表（包含 id, task_type, content, coverage, avg_accuracy, avg_generation_rate, composite_score）

**文本向量化实现**（参考 Reference/Swarm-Evo）:
```python
from sentence_transformers import SentenceTransformer
import os

class CodeEmbeddingManager:
    _model_name = "BAAI/bge-m3"
    _local_model_path = os.environ.get("LOCAL_MODEL_PATH", "./embedding-models/bge-m3")
    _model = None

    @classmethod
    def _ensure_model(cls):
        if cls._model is None:
            if os.path.exists(cls._local_model_path):
                cls._model = SentenceTransformer(cls._local_model_path)
            else:
                cls._model = SentenceTransformer(cls._model_name)
            cls._model.eval()

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """返回 L2 归一化的 embeddings。"""
        self._ensure_model()
        embeddings = self._model.encode(
            texts,
            batch_size=8,
            show_progress_bar=False,
            normalize_embeddings=True,  # L2 归一化
        )
        return embeddings.astype(np.float32)
```

---

#### [NEW] `core/evolution/skill_manager.py` (约 280 行)

**职责**: Skill 池管理（质量评估、演化、元数据维护）。

| 函数/类 | 签名 | 说明 |
|---------|------|------|
| `SkillManager.__init__` | `(skills_dir: Path, meta_dir: Path, config: Config, embedding_manager: CodeEmbeddingManager)` | 初始化管理器 |
| `add_skill` | `(skill: Dict) -> None` | 添加新 Skill（检测重复） |
| `evaluate_skill` | `(skill_id: str) -> float` | 计算 Skill 综合评分（使用 Reference 公式） |
| `evolve_skills` | `(experience_pool: ExperiencePool, extractor: SkillExtractor) -> None` | Skill 池演化（新增/合并/淘汰） |
| `get_top_k_skills` | `(task_type: str, k: int) -> List[str]` | 获取 Top-K Skill 内容 |
| `reload_index` | `() -> None` | 重新加载 Skill 索引 |
| `_is_duplicate` | `(skill: Dict, threshold: float = 0.85) -> bool` | 检测重复（余弦相似度） |
| `_merge_similar_skills` | `(threshold: float = 0.85) -> None` | 合并相似 Skill |
| `_deprecate_low_quality_skills` | `() -> None` | 淘汰低质量 Skill |
| `_load_skill_content` | `(skill_id: str) -> str` | 加载 Skill 文件内容 |
| `_save_index` | `() -> None` | 保存 Skill 索引到 JSON |
| `_load_index` | `() -> Dict` | 从 JSON 加载 Skill 索引 |

**核心流程**:
- **新增**: 检测重复 → 写入文件 → 更新索引
- **评估**: 综合评分公式（参考 Reference/Swarm-Evo）
  ```python
  composite_score = 0.6 × avg_accuracy + 0.4 × avg_generation_rate
  ```
- **演化**: 提取新 Skill → 合并相似 Skill → 淘汰低质量 Skill

---

### 1.3.2 修改文件

#### [MODIFY] `core/evolution/agent_evolution.py`

**修改位置**: `AgentEvolution` 类

| 方法 | 变更类型 | 说明 |
|------|---------|------|
| `__init__` | **[MODIFY]** | 新增 `skill_manager: SkillManager` 参数 |
| `evolve` | **[MODIFY]** | 补充 Skill 池更新逻辑（调用 `skill_manager.evolve_skills`） |
| `_reload_skills` | **[NEW]** | 重新加载 Skill 池（通知 PromptManager） |

**具体修改**:
```python
def __init__(
    self,
    agents: List[BaseAgent],
    experience_pool: ExperiencePool,
    skill_manager: SkillManager,  # 新增参数
    config: Config,
):
    ...
    self.skill_manager = skill_manager  # 新增属性

def evolve(self, epoch: int) -> None:
    """主入口：每 N 个 Epoch 进化一次。"""
    if epoch % self.evolution_interval != 0 or epoch == 0:
        return

    log_msg("INFO", f"===== Agent 层进化开始（Epoch {epoch}） =====")

    # [1] 评估所有 Agent
    scores = self._evaluate_agents()
    elite_ids, weak_ids = self._identify_elites_and_weak(scores)

    # [2] 对弱者进行 Role 和 Strategy 变异（已有逻辑）
    self._mutate_weak_agents(weak_ids, elite_ids)

    # [3] Skill 池更新（新增逻辑）
    log_msg("INFO", "开始 Skill 池更新...")
    self.skill_manager.evolve_skills(
        self.experience_pool,
        SkillExtractor(self.experience_pool, self.config)
    )
    log_msg("INFO", "Skill 池更新完成")

    # [4] 重新加载 Skill 池（新增逻辑）
    self._reload_skills()

    log_msg("INFO", f"===== Agent 层进化完成（Epoch {epoch}） =====")

def _reload_skills(self) -> None:
    """重新加载 Skill 池。"""
    # 通知所有 Agent 的 PromptManager 重新加载索引
    self.skill_manager.reload_index()
    log_msg("INFO", "Skill 池已重新加载")
```

---

#### [MODIFY] `utils/prompt_manager.py`

**修改位置**: `PromptManager` 类

| 方法 | 变更类型 | 说明 |
|------|---------|------|
| `__init__` | **[MODIFY]** | 新增 `skill_manager: SkillManager` 参数（可选） |
| `inject_top_k_skills` | **[MODIFY]** | 修改实现：从 SkillManager 获取 Top-K Skill |
| `update_skill_pool` | **[NEW]** | 更新 Skill 池引用（用于 Agent 进化后重载） |
| `_format_skill_examples` | **[NEW]** | 格式化 Skill 为 Markdown 列表 |

**具体修改**:
```python
def __init__(
    self,
    template_dir: Path,
    skills_dir: Path,
    agent_configs_dir: Path,
    skill_manager: Optional[SkillManager] = None,  # 新增参数
):
    ...
    self.skill_manager = skill_manager  # 新增属性

def inject_top_k_skills(
    self,
    task_type: str,
    k: int = 5,
    experience_pool: Optional[ExperiencePool] = None,
) -> str:
    """注入 Top-K 动态 Skill（从 SkillManager 获取）。

    Args:
        task_type: 任务类型（explore/merge/mutate）
        k: Top-K 数量
        experience_pool: 经验池（如果 skill_manager 为 None，从经验池直接提取）

    Returns:
        Markdown 格式的 Skill 列表
    """
    if self.skill_manager:
        # 从 SkillManager 获取 Top-K Skill
        skills = self.skill_manager.get_top_k_skills(task_type, k)
    elif experience_pool:
        # Fallback: 从经验池直接提取 Top-K 成功案例
        records = experience_pool.query(task_type, k, filters={"output_quality": (">", 0.5)})
        skills = [r.strategy_summary for r in records]
    else:
        log_msg("WARNING", "未提供 skill_manager 或 experience_pool，跳过 Skill 注入")
        return ""

    return self._format_skill_examples(skills)

def update_skill_pool(self, skill_manager: SkillManager) -> None:
    """更新 Skill 池引用（Agent 进化后调用）。

    Args:
        skill_manager: 新的 SkillManager 实例
    """
    self.skill_manager = skill_manager
    log_msg("INFO", "PromptManager Skill 池已更新")

def _format_skill_examples(self, skills: List[str]) -> str:
    """格式化 Skill 为 Markdown 列表。

    Args:
        skills: Skill 内容列表

    Returns:
        Markdown 格式字符串
    """
    if not skills:
        return "无可用的成功案例。"

    formatted = "### 成功案例（Top-K Skill）\n\n"
    for i, skill in enumerate(skills, 1):
        formatted += f"#### 示例 {i}\n{skill}\n\n"
    return formatted
```

---

#### [MODIFY] `config/default.yaml`

**修改位置**: `evolution` 配置节

**新增配置**:
```yaml
evolution:
  # ... 现有配置 ...

  # Skill 池配置（P3.5 新增）
  skill:
    min_cluster_size: 5  # HDBSCAN 最小簇大小
    duplicate_threshold: 0.85  # 语义相似度阈值（去重）
    min_composite_score: 0.5  # 新增 Skill 的最低综合评分
    deprecate_threshold: 0.4  # 淘汰 Skill 的综合评分阈值
    unused_epochs: 5  # 连续未使用 Epoch 数（淘汰条件）
    embedding_model_path: "./embedding-models/bge-m3"  # bge-m3 本地路径
```

---

#### [MODIFY] `core/orchestrator.py`

**修改位置**: `Orchestrator` 类

| 方法 | 变更类型 | 说明 |
|------|---------|------|
| `__init__` | **[MODIFY]** | 新增 `agent_evolution: AgentEvolution` 参数（可选） |
| `run` | **[MODIFY]** | 主循环中添加 Agent 层进化调用 |

**具体修改**:
```python
def __init__(
    self,
    agent: BaseAgent,
    config: Config,
    journal: Journal,
    task_desc: str,
    agent_evolution: Optional[AgentEvolution] = None,  # 新增参数
):
    ...
    self.agent_evolution = agent_evolution  # 新增属性

def run(self, num_epochs: int = 1) -> Optional[Node]:
    """主循环（双层进化模式）。

    Args:
        num_epochs: Epoch 数量（默认 1，兼容原有逻辑）

    Returns:
        最佳节点
    """
    for epoch in range(num_epochs):
        log_msg("INFO", f"===== Epoch {epoch} 开始 =====")

        # Solution 层进化（原有逻辑，单 Epoch 内多 step）
        for step in range(self.config.agent.max_steps):
            if self._check_time_limit():
                break

            # 原有流程：选择父节点 → Agent 生成 → 执行 → Review → 更新
            self._prepare_step()
            parent_node = self._select_parent_node()
            context = self._build_agent_context(parent_node, step)
            result = self.agent.generate(context)

            if not result.success:
                continue

            node = result.node
            self._execute_code(node)
            self._review_node(node)
            self.journal.append(node)
            self._update_best_node(node)

        # Agent 层进化（每 N Epoch，新增逻辑）
        if self.agent_evolution:
            self.agent_evolution.evolve(epoch)

        # 日志记录
        best = self.journal.get_best_node()
        log_msg("INFO", f"Epoch {epoch} 最佳 fitness: {best.metric_value if best else 'N/A'}")

    return self.best_node
```

---

### 1.3.3 测试文件

#### [NEW] `tests/test_evolution/test_skill_extractor.py` (约 120 行)

**测试用例**:
- `test_extract_skills_success` - 正常提取 Skill
- `test_extract_skills_insufficient_records` - 记录数不足
- `test_embed_texts` - 文本向量化（bge-m3）
- `test_cluster` - HDBSCAN 聚类
- `test_summarize_cluster` - LLM 总结（Mock）

---

#### [NEW] `tests/test_evolution/test_skill_manager.py` (约 150 行)

**测试用例**:
- `test_add_skill_success` - 添加新 Skill
- `test_add_skill_duplicate` - 重复 Skill 检测
- `test_evaluate_skill` - Skill 质量评估（使用 Reference 公式）
- `test_evolve_skills_new` - 新增 Skill
- `test_evolve_skills_deprecate` - 淘汰 Skill
- `test_get_top_k_skills` - Top-K 查询

**注**: 集成测试暂不实施。

---

## 1.4 验证计划（Verification Plan）

### 1.4.1 单元测试验证

```bash
# 环境检查
conda run -n Swarm-Evo python --version  # 应显示 3.10.19

# 运行 Skill 提取器测试
conda run -n Swarm-Evo pytest tests/test_evolution/test_skill_extractor.py -v

# 运行 Skill 管理器测试
conda run -n Swarm-Evo pytest tests/test_evolution/test_skill_manager.py -v

# 覆盖率检查
conda run -n Swarm-Evo pytest tests/test_evolution/ \
  --cov=core/evolution \
  --cov-report=term-missing
```

**预期结果**:
- 所有测试通过 ✅
- 覆盖率 >= 80% ✅
- 无 ERROR 日志 ✅

---

### 1.4.2 功能验证

```bash
# 查看 Skill 池生成结果（手动运行后）
ls -lh benchmark/mle-bench/skills/by_task_type/explore/success_patterns/
cat benchmark/mle-bench/skills/meta/skill_index.json | python -m json.tool

# 查看日志
tail -n 100 logs/system.log
grep -i "skill" logs/system.log
```

**验证标准**（手动验证）:
- [ ] Skill 池目录结构正确创建
- [ ] `skill_index.json` 格式正确
- [ ] 无 ERROR 日志（WARNING 可接受）

**注**: 集成测试暂不实施。

---

### 1.4.3 代码质量检查

```bash
# 代码格式化
conda run -n Swarm-Evo ruff format core/evolution/ tests/test_evolution/

# 代码检查
conda run -n Swarm-Evo ruff check core/evolution/ tests/test_evolution/ --fix
```

**验证标准**:
- [ ] 无 ruff 错误 ✅
- [ ] 所有函数包含中文 Docstring ✅
- [ ] 无 `print()` 语句（使用 `log_msg`） ✅

---

### 1.4.4 Skill 质量人工审核

**抽样检查 Skill 文件**:
1. 打开 `benchmark/mle-bench/skills/by_task_type/explore/success_patterns/skill_explore_*.md`
2. 检查内容质量：
   - 结构完整（标题、核心策略、示例、注意事项）
   - 策略描述清晰、可操作
   - 无明显语法错误或重复内容

**预期**: 至少 80% 的 Skill 文件质量合格。

---

## 1.5 依赖与风险

### 1.5.1 外部依赖

**新增 Python 依赖**:
```bash
# HDBSCAN 聚类库
conda run -n Swarm-Evo pip install hdbscan

# sentence-transformers（用于 bge-m3 嵌入）
conda run -n Swarm-Evo pip install sentence-transformers

# NumPy（已安装，确认版本）
conda run -n Swarm-Evo pip show numpy
```

**模型下载**（首次运行时自动下载，或手动下载）:
```bash
# 可选：手动下载 bge-m3 模型到本地
# 方法 1: 使用 huggingface-cli
mkdir -p ./embedding-models
huggingface-cli download BAAI/bge-m3 --local-dir ./embedding-models/bge-m3

# 方法 2: 首次运行时自动下载（需要联网）
# 设置环境变量（可选）
export LOCAL_MODEL_PATH=./embedding-models/bge-m3
```

---

### 1.5.2 风险缓解

| 风险 | 等级 | 缓解措施 |
|------|------|---------|
| Skill 提取质量不高 | 中 | - 人工审核样本 Skill<br>- 调整聚类参数 `min_cluster_size`<br>- 优化 LLM Prompt |
| Skill 池过度膨胀 | 中 | - 定期清理（淘汰条件：5 Epoch 未使用）<br>- 严格重复检测（相似度阈值 0.85） |
| LLM 生成 Skill 不稳定 | 中 | - 多样本生成取 consensus<br>- 格式验证（检查 Markdown 结构） |
| HDBSCAN 聚类失败 | 低 | - 记录数不足时跳过聚类<br>- 噪声点过多时调整参数 |
| bge-m3 模型首次加载慢 | 低 | - 提前下载模型到本地<br>- 懒加载机制（首次使用时加载） |
| 模型内存占用 | 低 | - bge-m3 约 2GB 内存<br>- 单例模式，全局共享 |

---

## 1.6 预计工作量

| 任务 | 预计工时 | 说明 |
|------|---------|------|
| CodeEmbeddingManager 实现 | 2h | bge-m3 封装 + 缓存 + 测试 |
| SkillExtractor 实现 | 5h | 聚类 + LLM 总结 + 测试 |
| SkillManager 实现 | 8h | 质量评估 + 演化逻辑 + 元数据管理 + 测试 |
| AgentEvolution 扩展 | 2h | 补充 Skill 池更新逻辑 + 测试 |
| PromptManager 扩展 | 2h | Top-K 注入 + 重载方法 + 测试 |
| Orchestrator 集成 | 1h | 双层进化调度 + 测试 |
| 文档同步 | 1h | 更新 CODEMAPS 和 README |

**总计**: 约 21 小时

---

## 1.7 验收标准

### 1.7.1 功能完整性
- [x] SkillExtractor 可从经验池提取 Skill（HDBSCAN 聚类 + LLM 总结）
- [x] SkillManager 可添加、评估、演化 Skill 池
- [x] AgentEvolution 可触发 Skill 池更新
- [x] PromptManager 可注入 Top-K Skill 到 Prompt
- [x] Orchestrator 可运行完整双层进化流程

### 1.7.2 测试覆盖率
- [ ] 单元测试覆盖率 >= 80%

### 1.7.3 代码质量
- [ ] Ruff 格式化和检查通过
- [ ] 所有函数包含中文 Docstring
- [ ] 使用 `utils.logger_system`（无 print()）

### 1.7.4 效果验证（手动验证）
- [ ] Skill 池目录结构正确
- [ ] `skill_index.json` 格式正确
- [ ] 模块接口调用正常（无 ERROR）

---

## 1.8 下一步

**等待用户审核批准后**，开始按以下顺序实施：
1. 实现 CodeEmbeddingManager（2h）
2. 实现 SkillExtractor（5h）
3. 实现 SkillManager（8h）
4. 扩展 AgentEvolution（2h）
5. 扩展 PromptManager（2h）
6. 集成 Orchestrator（1h）
7. 文档同步（1h）

**里程碑 M5**: Phase 3 完整实现（Day 14）

---

**文档版本**: v1.0
**创建日期**: 2026-02-01
**完成日期**: 2026-02-01
**状态**: ✅ 已完成
