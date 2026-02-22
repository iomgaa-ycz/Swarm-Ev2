# MLE-Bench V4 深度分析报告

**运行批次**: `2026-02-20T01-53-11-GMT_run-group_swarm-evo`
**评分时间**: `2026-02-21T14:50:00-GMT`
**分析完成**: `2026-02-21`
**分析人员**: Claude Sonnet 4.6 (深度逐条日志分析)

---

## 1. 核心摘要

| 指标 | V2 | V3 | V4 |
|------|----|----|-----|
| 总竞赛数 | 22 | 22 | 22 |
| 有效提交 | 22 | 21 | 19 |
| 奖牌数 | 9 | 8 | 8 |
| 金牌 | 7 | 6 | 6 |
| 银牌 | 1 | 1 | 1 |
| 铜牌 | 1 | 1 | 1 |
| 获奖率 | **40.9%** | 36.4% | **36.4%** |

**结论**: V4 与 V3 持平，未实现突破。但组成发生了变化——新增了2个金牌（detecting-insults、denoising），同时因系统性Bug失去了2个奖牌（whale-challenge、tabular-dec-2021）及1个无效提交（text-norm-en）。

**目标差距**: 当前 36.4%，目标 80%（需额外 +10 个奖牌 = 18/22）。

---

## 2. V4 完整结果矩阵

### 2.1 获奖竞赛（8/22）

| 竞赛 | 奖牌 | LB分数 | 阈值 | 超越幅度 | GA活跃 |
|------|------|--------|------|---------|--------|
| aerial-cactus | 🥇 金 | 1.0 | 1.0 | 完美分 | - |
| histopathologic | 🥇 金 | 0.99772 | 0.9835 | +0.014 | 是 |
| dogs-vs-cats | 🥇 金 | 0.00968 | 0.03882 | 远超(↓) | 是 |
| denoising | 🥇 金 | 0.01644 | 0.01794 | +0.001 | 是★ |
| detecting-insults | 🥇 金 | 0.91609 | 0.83321 | +0.083 | 是★ |
| plant-pathology | 🥇 金 | 0.99519 | 0.97836 | +0.017 | 是 |
| nomad2018 | 🥈 银 | 0.05898 | 0.06229 | 距金0.003 | 是 |
| text-norm-ru | 🥉 铜 | 0.97967 | 0.97592 | +0.004 | 是 |

★ GA merge操作是达到金牌的关键操作。

### 2.2 零提交/无效（3/22）—— 系统Bug导致

| 竞赛 | 问题类型 | 根因 | 内部最佳 | 理论可达奖牌 |
|------|---------|------|---------|------------|
| whale-challenge | NULL提交 | 提交持久化Bug | AUC=0.993 | 金/银（>>金阈值0.990） |
| tabular-dec-2021 | NULL提交 | TimeoutError死亡螺旋 | 无 | 金（V2/V3均得金） |
| text-norm-en | 无效提交 | NaN提交被拒绝 | 0.981内部分 | 铜/银 |

### 2.3 上中位线（4/22）

| 竞赛 | LB分数 | 铜牌线 | 差距 | 内部最佳 | GA | 主要问题 |
|------|--------|--------|------|---------|-----|---------|
| random-pizza | 0.634 | 0.692 | -0.058 | 0.780 | 是 | CV-LB大偏差(0.146) |
| tabular-may-2022 | 0.988 | 0.998 | -0.010 | 0.987 | 否(未触发) | GA未触发 |
| spooky-author | 0.336 | 0.294 | -0.042 | 0.337 | 是 | 算法天花板 |
| dog-breed | 0.4173 | 0.046 | -0.371 | 0.475 | 是(GA活跃) | 未用预训练CNN; LB优于内部 |

### 2.4 下中位线（7/22）

| 竞赛 | LB分数 | 铜牌线 | 差距 | 中位线 | 内部最佳 | 主要问题 |
|------|--------|--------|------|--------|---------|---------|
| jigsaw-toxic | 0.980 | 0.986 | -0.006 | 0.981 | 0.9855 | GA未触发; CV-LB偏差0.005 |
| mlsp-birds | 0.865 | 0.874 | -0.009 | 0.866 | 0.827★ | GA活跃(104merge); LB比内部好 |
| aptos2019 | 0.870 | 0.914 | -0.044 | 0.889 | 0.905 | CV-LB偏差0.035; GA缺失 |
| leaf-classification | 0.217 | 0.015 | -0.202 | 0.108 | **0.112**★★ | **CV-LB偏差0.105**; GA极活跃(132merge) |
| new-york-taxi | 8.941 | 2.924 | -6.0 | 3.597 | 3.115 | CV-LB偏差×2.87倍 |
| siim-melanoma | 0.811 | 0.937 | -0.126 | 0.913 | 0.880 | 大量超时; 算法弱 |
| ranzcr-clip | 0.816 | 0.971 | -0.155 | 0.968 | ~0.800 | 大量超时; 算法弱 |

★ mlsp-birds: 内部0.827 < LB 0.865（LB优于内部，说明泛化良好）
★★ leaf-classification: 内部0.112接近中位线(0.108)，但LB=0.217远低于中位线 → 严重过拟合

---

## 3. 根因深度分析

### 3.1 【P0-Critical】提交持久化Bug — whale-challenge丢失银/金牌

**现象**: 日志中 `best_solution/` 已保存 AUC=0.993（远超金牌线0.990），但 `/home/submission/` 目录为空。

**根因链条**:
1. 14:08:52 最后一次适配器日志（Function Calling响应仅14字符）
2. LLM API调用进入阻塞状态（疑似API超时或网络挂起）
3. 37分钟无任何活动
4. 14:45:57 Docker容器被SIGKILL（达到整体时限46768秒）
5. `copy_results()` 在 `run_mle_adapter.py` 第353行，位于主循环**之后**调用
6. SIGKILL阻止了 `copy_results()` 运行 → 提交目录永远为空

**代码定位**（`run_mle_adapter.py`）:
```python
# 主循环结束后才调用 (第353行)
copy_results(journal, config, orchestrator)  # ← 被SIGKILL打断
```

而 `_save_best_solution()` 只存到 `best_solution/`，不拷贝到 `/home/submission/`。

**修复方案**: 在 `_save_best_solution()` 中每次发现新最佳节点时，立即同步拷贝到 `/home/submission/submission.csv`：
```python
def _save_best_solution(self, node):
    # 现有逻辑: 保存到 workspace/best_solution/
    ...
    # 新增: 即时同步到 /home/submission/（防SIGKILL）
    import shutil
    submission_src = self.workspace / "submission" / f"submission_{node.id}.csv"
    if submission_src.exists():
        shutil.copy2(submission_src, Path("/home/submission/submission.csv"))
```

**预期效果**: +1 奖牌（whale-challenge 金/银，AUC=0.993 >> 金0.990，银0.950）

---

### 3.2 【P0-Critical】TimeoutError死亡螺旋 — tabular-dec-2021丢失金牌

**现象**: 10个节点全部BUGGY，无法生成提交。

**节点序列分析**:
| 节点 | 状态 | 执行时间 | 错误类型 |
|------|------|---------|---------|
| 第1个 | BUGGY | 75s | 代码错误（非超时） |
| 第2-3个 | BUGGY | 44-64s | 代码错误 |
| 第4个 | BUGGY | 3442s | TimeoutError |
| 第5个 | BUGGY | 15s | 代码错误 |
| 第6-8个 | BUGGY | **7200s×3** | TimeoutError（全部撞超时上限） |
| 第9-10个 | BUGGY | 14s, 5954s | 代码错误, TimeoutError |

**根因**: 首个节点遇到代码错误 → LLM推断"简单模型不行"，升级为神经网络方案 → 神经网络训练超过7200s时限 → Debug循环对超时无能为力（Debug只改代码，不改架构） → 无法逃出循环。

**V2/V3均得金**的原因: tabular-playground-dec-2021本质是表格竞赛，LightGBM/XGBoost即可拿金。当前LLM错误判断导致算法升级。

**修复方案**: 在 `explore.j2` 提示模板中，为 `TimeoutError` 增加专属分支提示（类比现有的 `MemoryError` 处理）：

```jinja
{% elif parent_node.exc_type == "TimeoutError" %}
> **Cause**: Code exceeded the time limit (execution was killed).
> **Required**: Simplify the solution with ONE of these actions:
> 1. Reduce training iterations (e.g., `n_estimators=200`, `epochs=5`)
> 2. Add early stopping: `early_stopping_rounds=20` or `EarlyStopping(patience=3)`
> 3. Sample data: `df.sample(frac=0.3, random_state=42)` to reduce computation
> 4. Switch to a faster/lighter model architecture
> **DO NOT** add more complexity — the previous solution was already too slow.
```

此方案完全泛化：与具体库无关，适用于所有竞赛类型。当前模板已有 `MemoryError`/`ProcessKilled` 分支，仅缺少 `TimeoutError` 分支。

**预期效果**: +1 奖牌（tabular-dec 历史上稳定得金，修复后保守估计银/金）

---

### 3.3 【P0-Critical】NaN提交被拒 — text-norm-en无效提交

**现象**: 提交文件存在（14MB, 993,466行）但MLE-bench评分器拒绝，返回 `valid_submission=False`。

**根因链条**:
1. 每个节点持续出现警告: `Submission 格式异常: ['submission 包含 17 个 NaN 值']`
2. 内部评分器宽容：17/993466 = 0.0017% NaN，仍计算分数（0.981034）
3. 节点标记为GOOD（非BUGGY），GA顺利触发
4. MLE-bench官方评分器严格：任何NaN = 无效提交

**对比text-norm-ru**: 同样的代码结构，俄语版成功拿铜（0.980），英语版因NaN失败。

**NaN来源**: 17个特定词汇/符号无法被模型归一化（可能是罕见标点、数学符号或特殊字符）。

**修复方案**: 分两层：

**层1（评分器已严格）**: `_validate_submission_format()` 发现NaN时已设 `has_submission=False`，节点标记为BUGGY——这部分已正确。

**层2（错误信息回传，当前缺失）**: 在 `_review_node()` 中，格式验证失败时，具体错误信息丢失在 `log_msg("WARNING", ...)` 里，模型看不到。修复方式：

```python
# _review_node() 中，格式验证失败后
if not submission_check["valid"]:
    error_details = "; ".join(submission_check["errors"])
    # 条件性回传：仅当代码执行成功（exc_type is None）才追加
    # 执行失败的节点不应收到提交格式错误 —— 那根本不是它的问题
    if node.exc_type is None:
        node.term_out = (node.term_out or "") + (
            f"\n\n[SUBMISSION VALIDATION FAILED]: {error_details}\n"
            f"Your code ran without errors but produced an invalid submission.\n"
            f"Fix the root cause in your code (do NOT use fillna as a patch)."
        )
    has_submission = False
```

关键设计：`node.exc_type is None` 守卫确保只有"代码执行成功但 submission 格式有问题"的节点才收到格式错误提示；因超时/ValueError 等执行失败的节点不会收到无关的 NaN 提示（避免干扰）。

**预期效果**: +1 奖牌（参考text-norm-ru得铜，英语版可能铜或银）

---

### 3.4 【P1-High】GA人口阈值过高 — jigsaw/tabular-may/aptos GA未触发

**现象**: `population_size=12` 触发阈值在多个竞赛中无法达到。

**受影响竞赛分析**:

**jigsaw-toxic** (最紧迫！差铜牌仅0.006):
- 总explore节点: 8个，3个good
- GA从未触发（需要12个好节点）
- 内部最佳: 0.985539 vs 铜牌线: 0.98639 → 差距仅 **0.0004**！
- LB: 0.980 vs 铜牌: 0.986 → 差距0.006（CV-LB偏差约0.005）

**tabular-may-2022**:
- 总explore: 8个（多次TimeoutError），2个good
- GA从未触发
- LB: 0.988 vs 铜牌: 0.998 → 差距0.010

**aptos2019** (有14个good节点但GA仍未触发):
- 14个improve/explore节点，多为good
- epoch确实开始，但未见merge/mutate操作
- 可能: 只有3个explore节点good（改进节点不计入population）
- 内部最佳: 0.905 vs 铜牌: 0.914 → 差距0.009
- LB: 0.870 vs 铜牌: 0.914 → 差距0.044（CV-LB偏差较大）

**修复方案**: 将GA触发阈值从12降低到4（或 `max(4, population_size // 3)`）：
```yaml
# config/mle_bench.yaml
evolution:
  solution:
    population_size: 12  # 保持总种群大小
    ga_trigger_threshold: 4  # 新增：GA触发最小好节点数
```

**预期效果**: +1-2 奖牌（jigsaw可能得铜，tabular-may可能得铜）

---

### 3.5 【P1-High】CV-LB严重偏差 — 多竞赛泛化失败

**偏差统计**（完整22竞赛）:
| 竞赛 | 内部最佳 | LB分数 | 偏差 | 方向 | 严重性 |
|------|--------|--------|------|------|------|
| **leaf-classification** | 0.112 | 0.217 | **+0.105** | CV过于乐观 | 🔴极严重 |
| **new-york-taxi** | 3.115 | 8.941 | **+5.826** | CV过于乐观 | 🔴极严重 |
| random-pizza | 0.780 | 0.634 | +0.146 | CV过于乐观 | 🔴严重 |
| siim-melanoma | 0.880 | 0.811 | +0.069 | CV过于乐观 | 🟡中等 |
| aptos2019 | 0.905 | 0.870 | +0.035 | CV过于乐观 | 🟡中等 |
| jigsaw-toxic | 0.986 | 0.980 | +0.006 | CV过于乐观 | 🟢轻微 |
| nomad2018 | 0.061 | **0.059** | -0.002 | LB优于CV | ✅良好 |
| dog-breed | 0.475 | **0.417** | -0.058 | LB优于CV | ✅良好 |
| mlsp-birds | 0.827 | **0.865** | -0.038 | LB优于CV | ✅良好 |
| denoising | ~0.014 | **0.016** | ~+0.002 | 近似 | ✅良好 |

**new-york-taxi根因**: 纽约出租车数据集极大（数百万行），MLE-bench采样小子集进行训练。CV在小子集内部估计RMSE=3.115，但实际测试集包含完整分布（长距离行程、高额票价），RMSE暴增到8.941。这是数据采样策略与测试集分布不匹配导致的系统性问题。

**random-pizza根因**: 数据集极小（495个样本），5折交叉验证方差极大。模型记住训练数据（0.780 AUC），在测试集上泛化差（0.634 AUC）。

**修复方向**: **不实施 prompt 层正则化指导。**

根因分析（参见上方各竞赛根因）：
- **new-york-taxi**: 数据采样策略与测试集分布不匹配（CV子集 vs 完整分布）
- **random-pizza**: 495样本导致的5折CV方差极大（统计固有问题）
- **leaf-classification**: 特征分布偏移（预提取特征 vs 测试集特征空间）

以上根因均属**数据层/分布层**问题，prompt层的正则化指令无法修复。

**参考系统验证**：查阅 AIDE (`aideml-main/aide/agent.py`) 和 ML-Master (`mcts_agent.py`) —— 两者均无任何正则化指导语句，完全依赖 LLM 的内置 "Kaggle grandmaster" 判断。在无正则化指导的情况下，参考系统仍能达到更高奖牌率。

**结论**: 此问题标记为 **暂无可行修复（P3）**，与 nomad2018 GA多样性问题同等优先级。

---

### 3.6 【P2-Medium】图像任务算法质量不足 — 四竞赛持续失败

**受影响竞赛**（均为图像分类）:

| 竞赛 | LB | 铜牌线 | 差距 | 超时节点数 | 内部最佳 |
|------|-----|--------|------|---------|---------|
| siim-melanoma | 0.811 | 0.937 | -0.126 | 12/14 | 0.879 |
| ranzcr-clip | 0.816 | 0.971 | -0.155 | 27+/20 | ~0.800 |
| aptos2019 | 0.870 | 0.914 | -0.044 | 多次 | 0.905 |
| dog-breed | 0.417 | 0.046 | -0.371 | 少量 | 0.475 |

**共同问题**:
1. **超时**: 图像模型训练超过7200s上限（针对大型数据集如ISIC/RANZCR）
2. **算法弱**: 未充分利用预训练CNN（EfficientNet、ResNet等）
3. **CV-LB偏差**: 图像任务普遍存在5-7%的内部估计偏高

**dog-breed特殊分析**:
- 内部节点执行时间很短（~1200s），说明使用了特征工程方法（非端到端CNN）
- 铜牌线0.046 vs 当前0.417 → 差距10倍，说明必须使用预训练CNN + fine-tuning
- 这是提示词工程问题：LLM应被引导优先使用EfficientNet/ResNet预训练模型

**修复方向（参考ML-Master经验）**: ML-Master在 `_prompt_environment` 中精选推荐包列表（含 `timm`、`transformers`），而非倾倒整个 conda 包列表。我们在 `workspace_rules.md` 的 "Available Resources" 节中加入关键库推荐表：

```markdown
### Key ML Libraries (Recommended)

| Task Type        | Library          | Quick Start Example                                        |
|------------------|------------------|------------------------------------------------------------|
| Image Classification | `timm`       | `timm.create_model('efficientnet_b4', pretrained=True, num_classes=N)` |
| Image Transforms | `torchvision`    | `transforms.Compose([transforms.Resize(224), transforms.ToTensor()])` |
| Text / NLP       | `transformers`   | `AutoModel.from_pretrained('bert-base-uncased')`           |
| Tabular          | `xgboost`, `lightgbm` | Standard gradient boosting                           |

**GPU Available? Always prefer pretrained backbones (timm/transformers) over training from scratch.**
For neural networks, suggest PyTorch rather than TensorFlow.
```

此方案适用所有图像/NLP竞赛，无需逐比赛定制。

---

### 3.7 【P1-High】leaf-classification：GA极活跃但CV-LB偏差严重

**发现（背景Agent分析修正）**: 原以为是算法质量问题，实为严重的过拟合/偏差问题。

**数据**:
- 总节点366个，good节点308个（84.2% — 系统运行良好！）
- GA极活跃：132次merge + 34次mutate
- 内部最佳: `109822cd` = **0.112019**（接近中位线0.108）
- LB分数: **0.21667**（远低于中位线）
- **CV-LB偏差: 0.105**（内部评为接近中位，LB却远低于中位）

**进化轨迹**（GA效果显著）:
```
0.751638 → 0.462122 → 0.434637 → 0.191047 → 0.163332 → 0.124475 → 0.112019
```

**根因**: 模型在内部验证集上过拟合。Leaf-classification使用预提取特征（而非原始图像），数据量较小，LLM生成的模型记住了训练集特征模式，但在测试集上泛化失败。

**修复方向**: **暂无可行修复（P3）**。leaf-classification 的 CV-LB 偏差根因是**特征分布偏移**（预提取特征的分布在训练集/测试集间存在差异），这是数据层问题，prompt 层无法干预。参考系统（AIDE/ML-Master）均未针对此类问题提供任何 prompt 层修复。

---

### 3.8 【P2-Medium】nomad2018：GA无法突破到金牌

**数据**:
- 内部最佳: `ca9f7f92` = 0.061413
- LB: 0.05898（比内部更好！）
- 金牌线: 0.05589 → 差距 0.003
- GA触发：是，进行了多轮merge/mutate
- GA结果：全部在0.061-0.063之间，无法改善explore最佳

**根因**: GA merge策略在nomad2018这类小型材料科学数据集上陷入局部最优。GA的merge操作是组合两个父节点的代码，但当两个父节点都使用相似特征工程方式时，merge无法创造突破性改进。

**修复方向**: 当前可用的针对性修复（领域知识库、特定特征工程）均不具备跨竞赛泛化性，暂不实施。系统层面有一个泛化的GA多样性修复（P3）：当连续N轮 merge/mutate 后 metric 改善 < 0.1% 时，触发"多样性探索"（强制从 virtual_root 生成全新方向）——此修复适用于所有竞赛，但实现较复杂，列为未来优化。

---

## 4. 统计规律与重要发现

### 4.1 GA活跃度与获奖率关系

| 类型 | 竞赛数 | 获奖数 | 获奖率 |
|------|--------|--------|--------|
| GA活跃（有merge/mutate） | 12 | 7 | **58.3%** |
| GA未触发 | 10 | 1 | **10.0%** |

**结论**: GA活跃对获奖率有决定性影响（58% vs 10%）。**降低GA触发阈值是最重要的系统性改进**。

### 4.2 超时模式分析

| 模式 | 竞赛 | 后果 |
|------|------|------|
| 首节点失败→LLM升级复杂模型 | tabular-dec-2021 | 全部节点超时，零提交 |
| 大型图像数据集固有超时 | siim, ranzcr | 仅2/14节点成功 |
| LLM API阻塞→SIGKILL | whale-challenge | 提交丢失 |
| 偶发超时后成功debug | aptos, histopathologic | 影响不大 |

### 4.3 节点类型效率对比

通过12个竞赛的汇总数据：

| 节点类型 | 总数 | 成功(good)数 | 成功率 | 平均改善 |
|---------|------|------------|--------|---------|
| explore | ~220 | ~110 | ~50% | 基准 |
| merge | ~150 | ~130 | ~87% | +较小改善 |
| mutate | ~60 | ~50 | ~83% | 偶有突破 |
| debug | ~80 | ~55 | ~69% | 修复为主 |

**关键发现**: Merge成功率最高（87%），但改善幅度有限；Explore成功率最低但探索多样性最高。

### 4.4 已修复Bug的验证

- **P0-A（指标方向）**: dogs-vs-cats 由之前无效变为金牌✓，spooky-author 分数改善（0.599→0.336）✓
- **P0-B（列顺序匹配）**: detecting-insults 由历史失败变为金牌✓

---

## 5. 差距分析：通往80%的路径

### 5.1 各竞赛当前状态与潜力评估

```
获奖竞赛 (8/22):
✅ aerial-cactus    | 🥇 | 稳定
✅ histopathologic  | 🥇 | 稳定
✅ dogs-vs-cats     | 🥇 | 稳定
✅ denoising        | 🥇 | 稳定
✅ detecting-insults| 🥇 | 稳定
✅ plant-pathology  | 🥇 | 稳定
✅ nomad2018        | 🥈 | 有升金潜力(差0.003)
✅ text-norm-ru     | 🥉 | 有升银潜力

Bug修复可恢复 (3/22):
🔧 whale-challenge  | 无→金/银 | 修复提交持久化
🔧 tabular-dec      | 无→银/金 | 修复超时死亡螺旋
🔧 text-norm-en     | 无效→铜 | 修复NaN提交

接近奖牌(算法改进) (4/22):
📈 jigsaw-toxic     | 下中位→铜 | 差0.006，GA触发可破
📈 tabular-may-2022 | 上中位→铜 | 差0.010，GA触发可破
📈 mlsp-birds       | 下中位→铜 | 差0.009，需算法改进
📈 aptos2019        | 下中位→铜 | 差0.044 LB，CV-LB偏差需解决

较难突破 (4/22):
⚠️ random-pizza     | 差铜0.058 | CV-LB偏差0.146，小数据集泛化差
⚠️ spooky-author    | 差铜0.042 | 需更好NLP模型
⚠️ siim-melanoma    | 差铜0.126 | 大量超时，需深度学习
⚠️ ranzcr-clip      | 差铜0.155 | 超时灾难，需根本性改进

当前极难 (3/22):
❌ dog-breed        | 差铜0.371 | 必须预训练CNN
❌ leaf-classification| 差铜0.202| 算法质量差
❌ new-york-taxi    | LB 8.941  | CV-LB偏差×2.87
```

### 5.2 阶段性改进路径

#### Phase 0：Bug修复（预计+3-4个奖牌）

| 修复 | 难度 | 预期收益 |
|------|------|---------|
| 提交持久化（每次最佳立即copy） | 低 | +1 whale |
| TimeoutError→强制简单算法 | 低 | +1 tabular-dec |
| NaN填充（identity mapping） | 低 | +1 text-norm-en |
| GA阈值 12→4 | 低 | +1-2 jigsaw/tabular-may |

**Phase 0 后预期**: **12-13/22 (54-59%)**

#### Phase 1：算法提升（预计+2-4个奖牌）

| 改进 | 难度 | 预期收益 |
|------|------|---------|
| 图像任务强制使用预训练CNN提示 | 中 | +1-2 (aptos/siim其中之一) |
| 小数据集正则化增强提示 | 中 | +0-1 pizza |
| 时序CV用于taxi/大数据集 | 中 | +0-1 taxi |
| NLP任务增强特征工程提示 | 中 | +0-1 spooky/mlsp |

**Phase 1 后预期**: **14-17/22 (63-77%)**

#### Phase 2：架构改进（预计+1-3个奖牌）

| 改进 | 难度 | 预期收益 |
|------|------|---------|
| 多模型集成（自动ensemble多个好节点） | 高 | +1-2 |
| 提示库建设（竞赛类型专属） | 高 | +1 |
| LLM升级（更强代码生成能力） | 高 | +1-2 |

**Phase 2 后预期**: **17-20/22 (77-91%)**

**达到80%需要Phase 0 + Phase 1 + 部分Phase 2全部成功**。

---

## 6. 优先级行动计划

### P0（本周，最高优先级）

1. **修复提交持久化Bug** [`core/orchestrator.py`]
   - `_save_best_solution()` 中额外同步拷贝到 `/home/submission/submission.csv`

2. **修复TimeoutError死亡螺旋** [`benchmark/mle-bench/prompt_templates/explore.j2`]
   - 在 `exc_type == "ProcessKilled"` 之后加 `elif exc_type == "TimeoutError"` 分支
   - 提示：减少 iterations/epochs，加 early stopping，采样数据，切换更轻量模型

3. **修复NaN提交（条件性错误回传）** [`core/orchestrator.py`]
   - `_review_node()` 中：格式验证失败且 `node.exc_type is None` 时，将具体格式错误 append 到 `node.term_out`

4. **降低GA触发阈值** [`config/mle_bench.yaml`, `config/default.yaml`, `core/evolution/solution_evolution.py`]
   - 新增 `ga_trigger_threshold: 4` 配置项（与 `population_size: 12` 解耦）
   - `run_epoch()` 中用 `ga_trigger_threshold` 做触发判断；GA 种群取 `min(population_size, len(good_nodes))`，offspring 数量动态计算

### P1（下周，高优先级）

5. **关键ML库推荐表** [`benchmark/mle-bench/skills/static/workspace_rules.md`]
   - 在 "Available Resources" 中加 Key ML Libraries 推荐表（`timm`、`transformers`、`torchvision`）
   - 参考 ML-Master/AIDE 精选包列表策略，引导模型主动用预训练模型

### P2/P3（中长期）

7. **多好节点集成输出**
   - `copy_results()` 中对 top-3 好节点做预测平均集成

8. **GA多样性增强**（P3）
   - 连续 N 轮 metric 改善 < 0.1% 时，触发"多样性探索"（从 virtual_root 重新探索）

---

## 7. 与V3对比：进步与退步

### 进步
- **detecting-insults**: V3无→V4金牌（P0-B列顺序修复功效）
- **denoising**: V3上中位→V4金牌（GA merge操作效果）
- **dogs-vs-cats**: 指标方向修复后首次拿金

### 退步
- **whale-challenge**: V3银牌→V4零提交（提交持久化Bug）
- **tabular-dec-2021**: V3金牌→V4零提交（TimeoutError死亡螺旋）
- **text-norm-en**: V3无信息→V4无效提交（NaN Bug）

### 净影响
进步：+2个新金牌；退步：-1个银牌，-1个金牌，+1无效
净奖牌变化：8→8（持平）

---

## 8. 关键统计洞察

```
GA活跃率:      12/22 = 54.5%（有merge/mutate操作）
GA获奖率:      7/12  = 58.3%（GA活跃时的获奖率）
非GA获奖率:    1/10  = 10.0%（GA未触发时的获奖率）

超时率:        8/22 = 36.4%（有严重超时问题的竞赛）
提交成功率:    19/22 = 86.4%（有效提交/总竞赛数）
CV>LB占比:     7/19 = 36.8%（内部CV比LB乐观的竞赛）

最小差距:      jigsaw-toxic = 0.006（距铜牌）
最快修复:      whale/tabular-dec/text-norm-en（3个代码级Bug）
最大收益预期:  Phase 0全部完成 = +3-4 medals
```

---

## 9. 结论

V4实验揭示了当前系统存在三类核心问题：

1. **可立即修复的系统Bug（3个）**: 提交持久化、TimeoutError死亡螺旋、NaN提交。这3个Bug共损失了3个潜在奖牌，是当前最紧迫的优先级。

2. **配置调优（1个）**: GA触发阈值12→4，可能带来额外1-2个铜牌。

3. **算法质量（长期）**: 图像任务（dog-breed、siim-melanoma、ranzcr-clip）和部分NLP任务需要更好的提示工程和算法指导。

**修复P0所有Bug + 调整GA阈值 = 预期达到12-13个奖牌（54-59%）**，距离80%目标仍有差距，需要Phase 1和Phase 2的持续改进。

---

*报告基于对22个竞赛全部run.log文件的逐行深度分析，结合代码层面的根因追溯生成。*
