# MLE-Bench 第二轮实验结果深度分析报告

> **实验配置**: glm-4.7 模型 + 群体进化 | 22 个 Kaggle 竞赛 | 10 Epoch
>
> **总成绩**: 22 提交 / 20 有效 / 9 奖牌 (4金1银4铜) / 13 超中位数
>
> **对比基线**: 上一轮 13 竞赛 / 12 有效 / 4 奖牌 (3金1银) / 7 超中位数
>
> **分析日期**: 2026-02-15
>
> **目标**: 从 9 枚奖牌（41%）提升到 80%+ 获奖率

---

## 一、总览仪表盘

### 1.1 全部 22 竞赛成绩表

| # | 竞赛 | 类型 | 得分 | 方向 | 中位线 | 铜牌线 | 金牌线 | 结果 | 新/旧 |
|---|------|------|------|------|--------|--------|--------|------|-------|
| 1 | aerial-cactus | 图像分类 | 1.0 | ↑ | 0.9991 | 1.0 | 1.0 | **金牌** | 旧 |
| 2 | aptos2019 | 细粒度CV | 0.893 | ↑ | 0.889 | 0.914 | 0.931 | 超中位 | 旧 |
| 3 | denoising | 图像去噪 | 0.081 | ↓ | 0.073 | 0.045 | 0.018 | 低于中位 | 旧 |
| 4 | detecting-insults | NLP | null | ↑ | 0.778 | 0.791 | 0.833 | **无效** | 旧 |
| 5 | dog-breed | 细粒度CV | 0.460 | ↓ | 0.472 | 0.046 | 0.001 | 超中位 | 旧 |
| 6 | dogs-vs-cats | 图像分类 | 0.064 | ↓ | 0.122 | 0.061 | 0.039 | 超中位 | 旧 |
| 7 | histopathologic | 图像分类 | 0.998 | ↑ | 0.948 | 0.974 | 0.984 | **金牌** | 旧 |
| 8 | jigsaw-toxic | NLP | 0.979 | ↑ | 0.981 | 0.986 | 0.987 | 低于中位 | 旧 |
| 9 | leaf-classification | 表格多分类 | 0.266 | ↓ | 0.108 | 0.015 | 0.0 | 低于中位 | 旧 |
| 10 | mlsp-birds | 音频 | 0.887 | ↑ | 0.866 | 0.874 | 0.935 | **铜牌** | 旧 |
| 11 | nomad2018 | 表格回归 | 0.059 | ↓ | 0.070 | 0.066 | 0.056 | **银牌** | 旧 |
| 12 | plant-pathology | 图像分类 | 0.989 | ↑ | 0.949 | 0.974 | 0.978 | **金牌** | 旧 |
| 13 | random-acts-pizza | 表格二分类 | 0.703 | ↑ | 0.600 | 0.692 | 0.979 | **铜牌** | 旧 |
| 14 | new-york-taxi | 大规模回归 | 5.708 | ↓ | 3.597 | 2.924 | 2.834 | 低于中位 | 新 |
| 15 | ranzcr-clip | 医学CV | 0.844 | ↑ | 0.968 | 0.971 | 0.974 | 低于中位 | 新 |
| 16 | siim-melanoma | 医学CV | 0.860 | ↑ | 0.913 | 0.937 | 0.946 | 低于中位 | 新 |
| 17 | spooky-author | NLP | 0.844 | ↓ | 0.419 | 0.294 | 0.165 | 低于中位 | 新 |
| 18 | tabular-dec-2021 | 表格 | 0.963 | ↑ | 0.953 | 0.957 | 0.957 | **金牌** | 新 |
| 19 | tabular-may-2022 | 表格 | 0.977 | ↑ | 0.973 | 0.998 | 0.998 | 超中位 | 新 |
| 20 | text-norm-en | 文本规范化 | null | ↑ | 0.990 | 0.990 | 0.997 | **无效** | 新 |
| 21 | text-norm-ru | 文本规范化 | 0.977 | ↑ | 0.976 | 0.976 | 0.990 | **铜牌** | 新 |
| 22 | whale-challenge | 音频特殊 | 0.929 | ↑ | 0.865 | 0.905 | 0.990 | **铜牌** | 新 |

### 1.2 汇总统计

| 指标 | 本轮(22题) | 上轮(13题) | 变化 |
|------|-----------|-----------|------|
| 有效提交 | 20/22 (91%) | 12/13 (92%) | ≈持平 |
| 奖牌数 | 9 (4G1S4B) | 4 (3G1S) | +5 |
| 获奖率 | **40.9%** | 30.8% | **+10.1%** |
| 超中位数 | 13/22 (59%) | 7/13 (54%) | +5% |
| 全局成功率 | **73.0%** | 31.5% | **+41.5%** |
| explore 成功率 | 62.5% | 29.5% | +33.0% |
| merge 成功率 | 91.7% | 46.7% | +45.0% |
| mutate 成功率 | 96.4% | 42.9% | +53.5% |

**关键结论**: 代码验证和 Debug 循环的引入使全局成功率从 31.5% 飙升至 73.0%，但获奖率仅从 30.8% 提升到 40.9%，**距离 80% 目标仍有巨大差距**。说明当前瓶颈不在代码执行质量，而在**方案质量和系统级 bug**。

---

## 二、新旧竞赛对比（13 个重复竞赛）

| 竞赛 | 上轮得分 | 本轮得分 | 上轮结果 | 本轮结果 | 变化 |
|------|---------|---------|---------|---------|------|
| aerial-cactus | 0.99999 | 1.0 | 超中位 | **金牌** | **↑ 升级** |
| aptos2019 | 0.886 | 0.893 | 低于中位 | **超中位** | **↑ 改善** |
| denoising | null | 0.081 | 无效 | 低于中位 | **↑ 有效化** |
| detecting-insults | **0.917** | null | **金牌** | **无效** | **↓↓ 严重退化** |
| dog-breed | 0.512 | 0.460 | 低于中位 | **超中位** | **↑ 改善** |
| dogs-vs-cats | 0.068 | 0.064 | 超中位 | 超中位 | → 持平 |
| histopathologic | 0.998 | 0.998 | **金牌** | **金牌** | → 持平 |
| jigsaw-toxic | 0.979 | 0.979 | 低于中位 | 低于中位 | → 持平 |
| leaf-classif | 0.995 | 0.266 | 低于中位 | 低于中位 | ↑ 改善但仍差 |
| mlsp-birds | 0.763 | 0.887 | 低于中位 | **铜牌** | **↑↑ 大幅改善** |
| nomad2018 | 0.061 | 0.059 | **银牌** | **银牌** | → 持平 |
| plant-pathology | 0.996 | 0.989 | **金牌** | **金牌** | → 持平 |
| random-pizza | 0.665 | 0.703 | 超中位 | **铜牌** | **↑ 升级** |

### 对比汇总

- **改善/升级**: 6 个（aerial-cactus↑金, aptos↑超中位, denoising↑有效, dog-breed↑超中位, mlsp-birds↑铜, random-pizza↑铜）
- **持平**: 5 个（dogs-vs-cats, histopathologic, jigsaw-toxic, nomad2018, plant-pathology）
- **退化**: 1 个（**detecting-insults: 金牌→无效，最严重退化**）
- **仍差**: 1 个（leaf-classif: 仍低于中位）

**重复竞赛奖牌率**: 上轮 4/13=30.8% → 本轮 6/13=46.2%（+15.4%）

---

## 三、发现的关键系统 Bug

### 3.1 [CRITICAL] `lower_is_better` 极性翻转 Bug

**这是本轮分析中发现的最关键系统 Bug，影响 6+ 个竞赛，是导致获奖率低的首要原因。**

#### Bug 描述

系统通过 LLM（glm-4.7）的 Function Calling 来判断每个竞赛的 metric 方向（越大越好 or 越小越好）。但 LLM 的判断**在同一竞赛内不一致** —— 对同一个 logloss 指标，有些节点返回 `lower_is_better=True`，有些返回 `lower_is_better=False`。

#### Bug 位置

- `core/orchestrator.py:675`: `node.lower_is_better = review_data.get("lower_is_better", False)` — 每个节点独立使用 LLM 返回值
- `core/orchestrator.py:1372-1388`: `_is_better()` 使用节点级 `lower_is_better` 而非全局一致的方向
- `core/state/journal.py:144-172`: `get_best_node()` 仅用第一个有效节点的方向作为参考

#### 受影响竞赛证据

| 竞赛 | True 次数 | False 次数 | 正确方向 | 影响 |
|------|----------|-----------|---------|------|
| **dogs-vs-cats** | ~50% | ~50% | True(↓) | 选择了 0.208 而非最佳 0.022 → **直接丢失金牌** |
| **spooky-author** | 42% | 58% | True(↓) | 选择了 0.868 而非最佳 0.424 → **巨大偏差** |
| **dog-breed** | 15 | 16 | True(↓) | 系统曾选 logloss=15.36 作为"最佳" |
| **leaf-classif** | 混合 | 混合 | True(↓) | 进化方向反复翻转 |
| **new-york-taxi** | 混合 | 混合 | True(↓) | 错误方向选择影响提交 |
| **denoising** | 混合 | 混合 | True(↓) | 内部 CV 0.016 vs 提交 0.081（5x gap） |

#### 最典型案例：dogs-vs-cats

- journal.json 记录 Best 方向: **↑ (higher is better)** — 对 logloss 完全错误
- 系统"最佳"节点: logloss = **0.208417**（越来越差的方向！轨迹: 0.031→0.037→0.208）
- 实际最佳节点 `edd472fe`: logloss = **0.022**（如提交可获**金牌**，金牌线 0.039）
- 实际提交: logloss = **0.064**（碰巧来自一个较好节点，但非最优）

#### 第二典型案例：spooky-author

- journal.json 记录 Best 方向: **↑ (higher is better)** — 对 logloss 完全错误
- 174 个节点，169 个成功（**97.1% 成功率，全竞赛最高**）
- 但进化轨迹仅 **1 次更新**: 第一个节点 logloss=0.868856 后，系统认为"越大越好"，所有后续更小的 logloss 值都被视为"退化"
- 实际最佳节点 logloss ≈ 0.424（接近中位线 0.419），但从未被选为 best

#### 预估影响

**修复此 bug 预计直接挽回 +3~5 枚奖牌**:
- dogs-vs-cats: 0.022 可获金牌（当前超中位）
- spooky-author: 0.424 可超中位，可能接近铜牌（当前远低于中位）
- dog-breed: 更好的节点选择可大幅降低 logloss
- denoising: 更好的节点选择可接近铜牌

### 3.2 [HIGH] 容器网络限制导致无法下载预训练权重

Docker 容器禁止网络访问（PyTorch Hub、torchvision 等均不可达），导致所有依赖运行时下载的预训练模型被迫使用 `weights=None`（随机初始化）。

**受影响竞赛**:
- **ranzcr-clip**: ResNet18 with weights=None, AUC 0.844 vs median 0.968（差距 0.124）
- **siim-melanoma**: MobileNetV2 随机初始化, AUC 0.860 vs median 0.913

**预估影响**: 修复后 +1~2 枚奖牌

### 3.3 [HIGH] sample_submission.csv 被 Agent 代码删除

`tabular-may-2022` 竞赛中，Agent 生成的代码意外删除了 `sample_submission.csv`，导致后续 30+ 个节点全部 `FileNotFoundError` 失败。

- 成功率仅 **6%**（3/50 节点成功，全竞赛最低）
- 最佳方案在前 2 步即找到，之后 10+ 小时完全浪费

### 3.4 [MEDIUM] API 超时级联

`jigsaw-toxic` 竞赛中出现 **14 次连续 API 超时**（从 12:38 到 14:55，持续 2+ 小时），随后触发"请求参数太长"错误。Epoch 2 的所有节点均超时（3600s），总运行时间 45449s 中大量时间浪费在等待 API 响应。

### 3.5 [MEDIUM] text-norm-en 评分类型错误

提交文件格式正确（993465 行，id+after 两列），但 Docker 容器中的 grader 在 `sort_values('id')` 时抛出 `TypeError: '<' not supported between instances of 'str' and 'float'`。根因是 pandas 版本差异导致 id 列混合了 str 和 float 类型。手动评分显示 accuracy=0.958（低于铜牌线 0.990）。

---

## 四、逐竞赛 BCD 层深度分析

### D层 — 无效提交（2 个）

#### D1: detecting-insults — 金牌退化为无效（最严重退化）

| 指标 | 上轮 | 本轮 |
|------|------|------|
| 得分 | 0.917（金牌） | null（无效） |
| 方案 | 5模型 Stacking | TF-IDF + LR + SVM |
| 结果 | 金牌 (+0.083) | 无效提交 |

**根因**: 提交 CSV 格式错误 + 系统验证器文件名匹配失败

1. **提交列缺失**: 期望 3 列 `Comment, Date, Insult`，实际仅 1 列 `Insult`。LLM 在代码注释中写了 *"Create submission with ONLY the 'Insult' column to satisfy format requirements"* — 这是 LLM hallucination
2. **验证器绕过**: 该竞赛的 sample_submission 文件名为 `sample_submission_null.csv`（非标准命名），但 `orchestrator.py:1025-1027` 只查找 `sample_submission.csv` / `sampleSubmission.csv`，导致**列名/列数校验完全被跳过**
3. **NaN 误报**: 其他节点的 524 NaN 值也源于此 — 评估框架按 3 列读取只有 1 列的 CSV，缺失列被填充为 NaN

**方案质量**: 内部 CV AUC=0.910784（TF-IDF word+char bigram + LogisticRegression + 5-Fold），方案本身合理，如修复提交格式预计得分 0.90~0.92（保持金牌）。

**系统修复**:
- `orchestrator.py` 改用 `glob("sample_submission*.csv")` 匹配所有变体
- submission 验证器在找到 sample_submission 后强制校验列名一致性

**教训**: 上轮金牌方案（5模型 Stacking）未被保留/复用，系统从零探索导致退化。

---

#### D2: text-normalization-en — 评分系统类型错误

| 指标 | 数值 |
|------|------|
| 手动评分 accuracy | 0.958 |
| 铜牌线 | 0.990 |
| 提交行数 | 993,465（正确） |
| 提交列 | id, after（正确） |

**根因**: MLE-Bench 评分系统 + pandas 3.0 兼容性 bug（**非提交格式问题**）

1. **answers.csv 含 16 个 NaN**: MLE-Bench `prepare.py` 从原始数据拆分时，16 条 token 的 `after` 列为空值
2. **pandas 3.0 StringDtype**: 主机 pandas 3.0 默认 `future.infer_string=True`，`read_csv` 将字符串列推断为 `StringDtype`，NaN 表示为 `pd.NA`
3. **`.astype(str)` 不转换 pd.NA**: grade.py 中 `.astype(str)` 在 StringDtype 下保持 `pd.NA`，`.to_numpy()` 再将其转为 `float('nan')`
4. **混合类型排序崩溃**: `accuracy_score` 内部 `np.unique()` 遇到 str + float 混合类型，抛出 `TypeError: '<' not supported between instances of 'str' and 'float'`

**提交文件本身完全正确**: 993465 行，`id,after` 两列，无 NaN，ID 100% 匹配。手动评分 accuracy = **0.958**（低于铜牌线 0.990）。

**修复**: grade.py 中 `answers[col].fillna('').astype(str)` 即可。但即使修复，accuracy 0.958 仍远低于铜牌线 0.990，实际影响有限。

---

### C层 — 低于中位数（7 个）

#### C1: spooky-author — logloss 0.844 vs 中位 0.419（偏差最大）

| 指标 | 数值 |
|------|------|
| 提交得分 | 0.844 (logloss) |
| 中位线 | 0.419 |
| 铜牌线 | 0.294 |
| 内部最佳 | 0.424 |
| 总节点数 | ~100 |
| `lower_is_better` True/False | 42% / 58% |

**根因**: **`lower_is_better` 极性翻转 bug**

系统内部最佳节点的 logloss = 0.424（超过中位线 0.419），但由于 58% 的 review 返回 `lower_is_better=False`（错误！logloss 应该越小越好），系统选择了 logloss=0.868 的劣质节点作为最佳方案。

**方案**: TF-IDF (word 1-3gram + char 2-5gram) + LogisticRegression (multi_class='multinomial')。方案本身合理，是经典 NLP 基线。

**修复 lower_is_better 后**: 提交 0.424 → 超中位，接近铜牌（差距 0.130）

---

#### C2: new-york-taxi — RMSE 5.708 vs 中位 3.597

| 指标 | 数值 |
|------|------|
| 提交得分 | 5.708 (RMSE) |
| 中位线 | 3.597 |
| 铜牌线 | 2.924 |
| 内部最佳 CV | 2.79 |
| CV-LB Gap | **2.92** (104% 偏差!) |

**根因**: 多重问题叠加
1. **`lower_is_better` bug**: 影响节点选择
2. **数据采样不一致**: 训练用 200万行采样，CV 只用 10% validation，test 有 960万行 — 数据分布偏移
3. **行数不匹配**: 部分 submission 行数少于预期（大数据集处理超时截断）
4. **缺少关键特征**: 无地理距离(haversine)、无时间周期特征

**方案**: 最佳节点(2b078cf3)使用 XGBoost + haversine/manhattan距离 + 机场距离 + PCA 特征，内部 CV RMSE=2.79（达到铜牌级别），但测试集泛化严重失败。

**额外发现**: `code/solution.py` 是另一个版本（含 KMeans 聚类），与实际最佳节点的代码**不一致**。如果 MLE-Bench 重新运行 solution.py 而非直接使用 submission.csv，得分会更差。

---

#### C3: ranzcr-clip — AUC 0.844 vs 中位 0.968（差距 0.124）

| 指标 | 数值 |
|------|------|
| 提交得分 | 0.844 (AUC) |
| 中位线 | 0.968 |
| 铜牌线 | 0.971 |
| 模型 | ResNet18 (weights=None) |
| 分辨率 | 256x256 |
| 训练 epochs | 4 |

**根因**: **容器网络限制 + 训练不足**
1. **无预训练权重**: `torchvision.models.resnet18(weights=None)` — 随机初始化的 ResNet18 在医学图像上性能极差
2. **分辨率过低**: 256x256 无法捕获导管线细节（catheter line 是细线条特征）
3. **训练不足**: 仅 4 epoch，3600s 超时限制
4. **单通道处理**: 将 3 通道灰度医学图像错误处理
5. **缺少数据增强**: 无旋转、翻转、裁剪等增强

**注**: 容器网络限制导致无法下载预训练权重，此问题暂无可行解决方案

---

#### C4: siim-melanoma — AUC 0.860 vs 中位 0.913

| 指标 | 数值 |
|------|------|
| 提交得分 | 0.860 (AUC) |
| 中位线 | 0.913 |
| 铜牌线 | 0.937 |
| 模型 | MobileNetV2 |
| 训练 epochs/fold | 1 |
| Buggy 率 | **73%** (11/15) |
| 成功率 | **26.7%** |

**根因**: 模型太弱 + 训练不足 + 高 buggy 率
1. **MobileNetV2 太弱**: 轻量模型不适合细粒度医学图像（黑色素瘤 vs 普通痣区分困难）
2. **仅 1 epoch/fold**: 3600s 超时限制，3-fold 每 fold 只能训练 1 epoch
3. **极端正负样本失衡**: pos_weight=55.5 导致过度预测阳性
4. **缺少元数据特征**: 未使用 age, sex, anatomy_site 等临床信息
5. **73% buggy 率**: 大多数因超时失败，系统缺乏"计算预算感知"

---

#### C5: leaf-classification — logloss 0.266 vs 中位 0.108

| 指标 | 上轮 | 本轮 |
|------|------|------|
| 得分 | 0.995 | 0.266 |
| 改善 | - | +73% (好转) |
| 仍差距 | - | 中位 0.108 |

**根因**: `lower_is_better` bug + 模型选择不当
1. **`lower_is_better` 翻转**: 同上轮类似问题，进化方向反复混乱
2. **使用 LightGBM 而非深度学习**: 数据集仅 990 个样本（99 类），LightGBM 在如此少样本多分类上表现差
3. **上轮"丢弃 192 维原始特征"bug 仍存在**: 系统从零探索，未复用上轮教训

**正面**: 比上轮 0.995 改善到 0.266（+73%），说明代码质量提升确实有效

---

#### C6: denoising — RMSE 0.081 vs 中位 0.073

| 指标 | 上轮 | 本轮 |
|------|------|------|
| 得分 | null(无效) | 0.081 |
| 改善 | - | 有效化 |
| 仍差距 | - | 中位 0.073, 铜牌 0.045 |

**根因**: 3 个技术问题
1. **Resize 破坏宽高比**: 原始图像 540x258 被 Resize 到 512x512，严重扭曲文档图像
2. **PIL 量化精度损失**: 整数化 → /255 导致精度降低
3. **`lower_is_better` bug**: 内部 CV 0.016 vs 提交 0.081（5x gap），部分因节点选择错误
4. **上轮双重归一化 bug 已修复**: `/255.0` 和行数问题已解决

**正面**: 从无效提交变为有效（上轮的 P0 修复生效）

---

#### C7: jigsaw-toxic — AUC 0.979 vs 中位 0.981（差距仅 0.002）

| 指标 | 上轮 | 本轮 |
|------|------|------|
| 得分 | 0.979 | 0.979 |
| 改善 | - | 无变化 |
| 差距 | 0.001 | 0.002 |

**根因**: API 超时级联 + prompt 过长
1. **总节点仅 13 个**: 对比其他竞赛动辄 30-100 节点，搜索空间极度受限
2. **Epoch 2 全部超时**: 14 次连续 API 超时（每次 ~10 分钟），随后"请求参数太长"错误
3. **内部最佳 0.985677**: 方案本身合理（TF-IDF + LR），但内部-测试 gap 0.007

**方案**: TF-IDF + LogisticRegression，与上轮完全相同。系统未突破 TF-IDF 天花板。

**最接近中位线的竞赛**（差 0.002），微幅改进即可突破。

---

### B层 — 超中位未获奖（4 个）

#### B1: aptos2019 — QWK 0.893 vs 铜牌 0.914（差距 0.021）

| 指标 | 上轮 | 本轮 |
|------|------|------|
| 得分 | 0.886 | 0.893 |
| 结果 | 低于中位 | **超中位** |
| 差铜牌 | 0.028 | 0.021 |

**根因**: 方案同质化
- 所有成功节点都使用 **EfficientNet-B0 @ 224x224, 3-fold, 6 epochs**
- 无架构多样性（没有 ResNet50、DenseNet 等替代方案）
- 进化操作未产生质的突破，仅在同一方案上微调

**改进**: 更大模型（EfficientNet-B3/B4）+ 更高分辨率（384+）+ 阈值优化

---

#### B2: dog-breed — logloss 0.460 vs 铜牌 0.046（差距 10x）

| 指标 | 上轮 | 本轮 |
|------|------|------|
| 得分 | 0.512 | 0.460 |
| 改善 | - | +10% |
| Focal Loss 问题 | 存在 | **已修复** |

**上轮 P0 修复效果验证**:
- 上轮发现的 "Focal Loss 与 logloss 不匹配" bug **已修复**：本轮使用 CrossEntropyLoss
- 得分从 0.512 改善到 0.460（确认修复有效）

**仍存在的问题**:
1. **`lower_is_better` bug**: 15 个 True vs 16 个 False — 系统曾选 logloss=15.36 作为"最佳"
2. **模型过弱**: ResNet50 + DenseNet121（2015 年模型），对 120 品种细粒度分类远远不够
3. **差铜牌 10 倍**: 即使修复所有 bug，当前方案也难达铜牌（需要 EfficientNet/ViT 级别模型）

---

#### B3: dogs-vs-cats — logloss 0.064 vs 铜牌 0.061（差距 0.003）

| 指标 | 上轮 | 本轮 |
|------|------|------|
| 得分 | 0.068 | 0.064 |
| 改善 | - | 微幅 |
| 差铜牌 | 0.007 | 0.003 |
| 系统最佳节点 | - | **0.022**（金牌级！） |

**这是 `lower_is_better` bug 造成损失最大的竞赛**:
- 系统内部最佳节点 `edd472fe`: logloss = **0.022**
- 金牌线: 0.039，银牌线: 0.050，铜牌线: 0.061
- **如果提交最佳节点，直接获得金牌！**
- 但 `lower_is_better` 翻转导致系统选择了 metric=0.208 的节点
- 最终提交的 0.064 还是碰巧来自一个较好但不是最好的节点

**修复 lower_is_better 后**: 0.022 → **金牌**（+1 奖牌，从超中位直接跳到金牌）

---

#### B4: tabular-may-2022 — AUC 0.977 vs 铜牌 0.998（差距 0.021）

| 指标 | 数值 |
|------|------|
| 提交得分 | 0.977 (AUC) |
| 铜牌线 | 0.998 |
| 成功率 | **6%** (3/50) |
| 根因 | sample_submission.csv 被删除 |

**根因**: Agent 代码删除了 `sample_submission.csv`
- 第 1~2 步找到最佳方案（AUC 0.977）
- 之后某节点代码中 `pd.read_csv('sample_submission.csv')` 后又写回了错误格式，导致后续所有节点 `FileNotFoundError`
- 50 个节点中仅 3 个成功（6%），实质上进化被完全瘫痪

**缺少关键特征工程**: 未对 f_27 等高 importance 特征做非线性变换和交互

---

## 五、上轮改进建议跟踪

### 已实施的改进

| 改进项 | 上轮优先级 | 实施状态 | 效果评估 |
|--------|----------|---------|---------|
| **降低 Buggy 率（代码模板/验证）** | P1 | **已实施** — 代码静态预验证 + 即时 Debug 循环 + submission 格式验证 | **成功率 31.5% → 73.0%**，效果显著 |
| **修复 denoising submission 格式** | P0 | **已实施** — 双重归一化和行数问题修复 | 从无效变为有效提交 |
| **增加 metric 合理性校验** | P0 | **部分实施** — 添加了预验证模块 | leaf-classif 的 metric=0.0 事件未再出现，但新问题(lower_is_better)暴露 |

### 未实施的改进

| 改进项 | 上轮优先级 | 当前状态 | 本轮影响 |
|--------|----------|---------|---------|
| **loss/metric 对齐检查** | P0 | **未实施** | dog-breed 的 Focal Loss 问题虽已修复但系统级未防范 |
| **图像分类默认 TTA** | P1 | **未实施** | aerial-cactus 靠方案质量获金牌，但其他图像竞赛未受益 |
| **进化停滞检测 + restart** | P2 | **未实施** | 多个竞赛后期仍在浪费时间 |
| **动态任务类型比例** | P2 | **未实施** | 探索/利用比例固定 |
| **降低 elite_size** | P2 | **未实施** | 种群同质化仍存在 |

### 效果评估总结

- **已实施 P1 "降低 Buggy 率"** 是本轮最大改善来源，直接贡献了 +5 枚奖牌（主要通过提高代码执行成功率）
- **新发现的 `lower_is_better` bug** 是上轮未识别的系统性问题，其影响超过所有上轮 P0 改进项

---

## 六、系统级进化效率分析

### 6.1 全局成功率对比

| 指标 | 上轮 | 本轮 | 提升 |
|------|------|------|------|
| 全局成功率 | 31.5% | **73.0%** | +41.5% |
| explore 成功率 | 29.5% | **62.5%** | +33.0% |
| merge 成功率 | 46.7% | **91.7%** | +45.0% |
| mutate 成功率 | 42.9% | **96.4%** | +53.5% |

**代码预验证 + Debug 循环效果卓著**，但成功率的提升未完全转化为奖牌率提升，说明瓶颈已从"代码能否运行"转移到"方案质量能否竞争"。

### 6.2 成功率异常值

| 竞赛 | 成功率 | 异常原因 |
|------|--------|---------|
| tabular-may-2022 | **6%** | sample_submission.csv 被删除 |
| siim-melanoma | **27%** | 3600s 超时 + 大数据集 |
| jigsaw-toxic | **62%** | API 超时级联 |

### 6.3 进化有效窗口

基于上轮发现的"前 60% 时间完成所有突破，后 40% 浪费"规律，本轮同样成立:
- jigsaw-toxic: 仅 Epoch 1 产出有效方案，Epoch 2 全部超时
- tabular-may-2022: 前 2 步后完全停滞
- 多个竞赛的最终提交方案在前 3-5 步内即确定

---

## 七、任务类型维度分析

| 类型 | 竞赛数 | 奖牌数 | 奖牌率 | 上轮奖牌率 | 变化 |
|------|-------|--------|--------|-----------|------|
| 图像分类(简单) | 3 | 2(G+G) | **67%** | 33% | ↑↑ |
| 图像分类(细粒度) | 3 | 1(G) | 33% | 33% | → |
| 医学图像 | 2 | 0 | **0%** | N/A | 新类型，完败 |
| NLP 分类 | 3 | 0 | **0%** | 50% | ↓↓ |
| 表格数据 | 5 | 3(G+S+B) | **60%** | 33% | ↑↑ |
| 大规模回归 | 1 | 0 | 0% | N/A | 新类型 |
| 文本规范化 | 2 | 1(B) | 50% | N/A | 新类型 |
| 音频/特殊 | 3 | 2(B+B) | **67%** | 0% | ↑↑ |

**关键发现**:
- **医学图像完败**(0/2): 根因是容器无预训练权重 + 超时限制
- **NLP 从 50% 降到 0%**: detecting-insults 退化 + jigsaw/spooky 均未突破
- **表格数据是最强项**(60%): LightGBM + 领域特征工程策略有效
- **音频/特殊从 0% 到 67%**: mlsp-birds 和 whale 的大幅改善

---

## 八、改进方向优先级排序（按 ROI，目标 80%+）

### 当前到目标的差距分析

- 当前: 9/22 = 40.9% 获奖
- 目标: 18/22 = 81.8% 获奖
- **需要额外 +9 枚奖牌**

### 各改进项预估收益

| 优先级 | 改进项 | 影响竞赛 | 预期新增奖牌 | 实现难度 | ROI |
|--------|--------|---------|------------|---------|-----|
| **P0** | **修复 `lower_is_better` bug** | dogs-vs-cats, spooky, dog-breed, leaf, denoising, new-york-taxi | **+2~4** | **极低** — 改为全局固定方向 | **极高** |
| **P0** | **修复 submission 格式验证** | detecting-insults, text-norm-en | **+1** | 低 — 加列名/列数校验 | 高 |
| **P0** | **保护输入文件(chmod 444)** | tabular-may-2022 | +0~1 | **极低** — 1行代码 | 高 |
| **P1** | **延长/自适应超时** | siim-melanoma, jigsaw-toxic, ranzcr-clip | **+1** | 低 — 改配置 | 中 |
| **P2** | **API 超时处理 + prompt 截断** | jigsaw-toxic, 全局 | +0~1 | 中 | 中 |
| **P2** | **进化停滞检测 + restart** | 全局 | +0~1 | 低 | 中 |
| **P2** | **方案库/经验复用** | detecting-insults 等重复竞赛 | +0~1 | 高 | 低 |
| **P3** | **NLP 引入 Transformer 基线** | jigsaw, spooky, detecting-insults | +1~2 | 高 — 需 GPU 资源 | 低 |

### 累计预估效果

| 实施范围 | 累计新增奖牌 | 总奖牌 | 获奖率 |
|---------|------------|--------|--------|
| 仅 P0 | +3~5 | 12~14 | 55~64% |
| P0 + P1 | +4~6 | 13~15 | 59~68% |
| P0 + P1 + P2 | +5~8 | 14~17 | **64~77%** |

---

## 九、具体 Action Items

### 立即执行（P0，预期 +3~5 枚奖牌）

#### 1. 修复 `lower_is_better` 极性翻转 bug [最高优先级]

**改动文件**: `core/orchestrator.py`

**方案**: 废弃 LLM 逐节点判断，改为**全局一次性确定 metric 方向**:

```python
# 方案 A（推荐）: 硬编码常见 metric 方向映射
METRIC_DIRECTION = {
    "auc": False, "accuracy": False, "f1": False, "qwk": False,  # higher is better
    "logloss": True, "rmse": True, "mae": True, "rmsle": True,   # lower is better
}

# 方案 B: 竞赛开始时 LLM 判断一次，后续所有节点共享
# 在 _initialize_competition() 中确定，存入 state
```

**关键改动**:
- `orchestrator.py:675`: 删除逐节点 `lower_is_better` 赋值
- `orchestrator.py:1372-1388`: `_is_better()` 使用全局方向
- `journal.py:144-172`: `get_best_node()` 使用全局方向

#### 2. 加强 submission 格式验证

**改动 A**: 修复 sample_submission 文件名匹配（`orchestrator.py:1025-1027`）:

```python
# 当前: 只查找固定文件名
# 修复: glob 匹配所有变体（解决 detecting-insults 的 sample_submission_null.csv 问题）
import glob
candidates = list(input_dir.glob("sample_submission*.csv")) + \
             list(input_dir.glob("sampleSubmission*.csv"))
sample_path = candidates[0] if candidates else None
```

**改动 B**: 找到 sample_submission 后强制校验列名/列数:

```python
def validate_submission(submission_path, sample_path):
    sample = pd.read_csv(sample_path, nrows=5)
    sub = pd.read_csv(submission_path, nrows=5)
    assert list(sub.columns) == list(sample.columns), f"列名不匹配"
    # 检查行数在 ±5% 范围内
```

#### 3. 保护输入文件

```python
import os, stat
for f in input_files:
    os.chmod(f, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)  # 444
```

### 短期改进（P1，预期 +1 枚奖牌）

#### 4. 自适应超时

```python
# 根据数据集大小和模型类型动态调整
timeout = base_timeout  # 3600s
if dataset_size > 100000: timeout *= 2  # 大数据集
if "deep_learning" in solution_type: timeout *= 1.5  # DL 训练
```

### 中期优化（P2-P3）

6. **API 超时处理**: 设置 prompt 长度上限，超限自动截断历史上下文
7. **进化停滞检测**: 连续 20 步无 best 更新时触发 restart
8. **方案库复用**: 对重复竞赛可加载上轮最佳方案作为初始解
9. **NLP 任务引入 Transformer**: 添加 distilbert/roberta 预训练模型支持

---

## 十、附录：关键数据

### 各竞赛 `lower_is_better` 翻转统计

| 竞赛 | 正确方向 | True 比例 | False 比例 | 翻转严重度 |
|------|---------|----------|-----------|----------|
| dogs-vs-cats | True(↓) | ~50% | ~50% | **极严重** |
| spooky-author | True(↓) | 42% | 58% | **严重** |
| dog-breed | True(↓) | 48% | 52% | **严重** |
| leaf-classif | True(↓) | 混合 | 混合 | 严重 |
| new-york-taxi | True(↓) | 混合 | 混合 | 严重 |
| denoising | True(↓) | 混合 | 混合 | 中等 |

### 各竞赛精确统计（来自 journal.json）

| 竞赛 | 总节点 | 成功 | 成功率 | Best方向 | 进化轨迹更新次数 | 结果 |
|------|--------|------|--------|---------|---------------|------|
| aerial-cactus | 108 | 80 | 74.1% | ↑ | 7 | **金牌** |
| detecting-insults | 175 | 127 | 72.6% | ↑ | 10 | **无效** |
| nomad2018 | 181 | 147 | 81.2% | ↓ | 13 | **银牌** |
| random-pizza | 187 | 158 | 84.5% | ↑ | 14 | **铜牌** |
| spooky-author | 174 | 169 | **97.1%** | ↑(**错**) | **1** | 低于中位 |
| mlsp-birds | 123 | 88 | 71.5% | ↑ | 9 | **铜牌** |
| denoising | 78 | 55 | 70.5% | ↓ | 10 | 低于中位 |
| new-york-taxi | 62 | 41 | 66.1% | ↓ | 9 | 低于中位 |
| text-norm-en | 79 | 32 | 40.5% | ↑ | 5 | **无效** |
| text-norm-ru | 52 | 25 | 48.1% | ↑ | 8 | **铜牌** |
| whale-challenge | 51 | 36 | 70.6% | ↑ | 10 | **铜牌** |
| leaf-classif | 50 | 41 | 82.0% | ↓ | 9 | 低于中位 |
| tabular-may-2022 | 50 | 3 | **6.0%** | ↑ | 2 | 超中位 |
| dogs-vs-cats | 34 | 26 | 76.5% | ↑(**错**) | 3 | 超中位 |
| dog-breed | 31 | 30 | 96.8% | ↓ | 8 | 超中位 |
| plant-pathology | 31 | 27 | 87.1% | ↑ | 8 | **金牌** |
| tabular-dec-2021 | 22 | 9 | 40.9% | ↑ | 5 | **金牌** |
| ranzcr-clip | 19 | 10 | 52.6% | ↑ | 4 | 低于中位 |
| aptos2019 | 19 | 15 | 78.9% | ↑ | 6 | 超中位 |
| histopathologic | 18 | 15 | 83.3% | ↑ | 4 | **金牌** |
| siim-melanoma | 15 | 4 | **26.7%** | ↑ | 1 | 低于中位 |
| jigsaw-toxic | 13 | 9 | 69.2% | ↑ | 5 | 低于中位 |

**重要**: spooky-author 成功率 97.1% 但仅 1 次轨迹更新 — 因为 `lower_is_better` 方向错误(↑)，系统认为 logloss 越大越好，第一个节点 0.868856 后再无"更好"值。dogs-vs-cats 同理（方向↑错误），轨迹从 0.031 上升到 0.208（越来越差）。

### 改进项实施优先级矩阵

```
                 高影响
                  │
    P0:lower_is_better ●     P1:自适应超时 ●
                  │
    P0:submission验证 ●
                  │
    P0:输入保护 ●
                  │
  ──────────────── ┼ ────────────────────
                  │
    P2:停滞检测 ●            P3:Transformer ●
                  │
    P2:API超时 ●             P3:方案库 ●
                  │
                 低影响
    低难度 ←──────┼──────→ 高难度
```

---

> **总结**: 本轮实验通过代码预验证+Debug循环将成功率从 31.5% 提升到 73.0%，是重大进步。但获奖率仅 40.9%，主要受限于 **`lower_is_better` 系统 bug**（影响 6+ 竞赛，直接丢失 2~4 枚奖牌）和 **submission 格式问题**（2 个无效提交）。修复 P0+P1 改进项后，预计获奖率可达 **59~68%**。**注**：容器网络限制导致无法下载预训练权重问题（影响 ranzcr-clip、siim-melanoma 两个医学图像竞赛）暂无可行解决方案。
