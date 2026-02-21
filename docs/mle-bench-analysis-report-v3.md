# MLE-Bench 第三轮实验结果深度分析报告

> **实验配置**: glm-4.7 模型 + 群体进化 | 22 个 Kaggle 竞赛 | 双层进化（Explore+GA）
>
> **总成绩**: 22 提交 / 19 有效提交 / 18 有效评分 / **8 奖牌 (5金2银1铜)** / 13 超中位数
>
> **对比基线**: 第二轮 22 竞赛 / 20 有效 / 9 奖牌 (4金1银4铜) / 13 超中位数
>
> **分析日期**: 2026-02-19
>
> **目标**: 从 40.9% 提升至 **80%** 获奖率

---

## 一、总览仪表盘

### 1.1 全部 22 竞赛成绩对比表

| # | 竞赛 | 类型 | V2 得分 | V2 结果 | V3 得分 | V3 结果 | 变化 |
|---|------|------|---------|---------|---------|---------|------|
| 1 | aerial-cactus | 图像 | 1.0 | **金牌** | 1.0 | **金牌** | → 持平 |
| 2 | aptos2019 | 细粒度CV | 0.893 | 超中位 | 0.868 | **低于中位** | ↓ 退化 |
| 3 | denoising | 图像去噪 | 0.081↓ | 低于中位 | 0.050↓ | **超中位** | ↑ 改善 |
| 4 | detecting-insults | NLP | null | 无效 | null | **无效** | → 持平(内损) |
| 5 | dog-breed | 细粒度CV | 0.460↓ | 超中位 | 0.429↓ | 超中位 | → 持平 |
| 6 | dogs-vs-cats | 图像 | 0.064↓ | 超中位 | 0.008↓ | **金牌** | ↑↑ 大幅升级 |
| 7 | histopathologic | 图像 | 0.998↑ | **金牌** | 0.994↑ | **金牌** | → 持平 |
| 8 | jigsaw-toxic | NLP | 0.979↑ | 低于中位 | 0.886↑ | **低于中位** | ↓ 深度退化 |
| 9 | leaf-classification | 表格 | 0.266↓ | 低于中位 | 0.333↓ | 低于中位 | → 持平(略差) |
| 10 | mlsp-birds | 音频 | 0.887↑ | **铜牌** | 0.806↑ | **低于中位** | ↓ 铜牌丢失 |
| 11 | new-york-taxi | 大规模回归 | 5.71↓ | 低于中位 | 10.69↓ | 低于中位 | → 持平(更差) |
| 12 | nomad2018 | 表格回归 | 0.059↓ | **银牌** | 0.061↓ | **银牌** | → 持平 |
| 13 | plant-pathology | 图像 | 0.989↑ | **金牌** | 0.994↑ | **金牌** | → 持平 |
| 14 | random-pizza | 表格二分类 | 0.703↑ | **铜牌** | 0.647↑ | 超中位 | ↓ 铜牌丢失 |
| 15 | ranzcr-clip | 医学CV | 0.844↑ | 低于中位 | null | **无效** | ↓ 完全失效 |
| 16 | siim-melanoma | 医学CV | 0.860↑ | 低于中位 | null | **无效** | ↓ 完全失效 |
| 17 | spooky-author | NLP | 0.844↓ | 低于中位 | 0.357↓ | **超中位** | ↑ 改善 |
| 18 | tabular-dec-2021 | 表格 | 0.963↑ | **金牌** | 0.961↑ | **金牌** | → 持平 |
| 19 | tabular-may-2022 | 表格 | 0.977↑ | 超中位 | 0.992↑ | 超中位 | ↑ 略有改善 |
| 20 | text-norm-en | 文本规范化 | null | 无效 | null | 无效 | → 持平 |
| 21 | text-norm-ru | 文本规范化 | 0.977↑ | **铜牌** | 0.979↑ | **铜牌** | → 持平 |
| 22 | whale-challenge | 音频 | 0.929↑ | **铜牌** | 0.958↑ | **银牌** | ↑ 铜升银 |

### 1.2 汇总统计

| 指标 | V3（本轮） | V2（上轮） | 变化 |
|------|-----------|-----------|------|
| 有效提交 | 18/22 (82%) | 20/22 (91%) | ↓ -9% |
| 奖牌数 | **8 (5G2S1B)** | 9 (4G1S4B) | -1 |
| 获奖率 | **36.4%** | 40.9% | -4.5% |
| 超中位数 | 13/22 (59%) | 13/22 (59%) | = 持平 |
| 改善/退化 | 4改/5退/13平 | — | — |

### 1.3 关键结论

> 本轮成绩 **轻微退化**（-1 枚奖牌），但实质上是**两种力量对冲**的结果：
> - **P0-1 lower_is_better 修复（正效果）**: 带来了 +1金（dogs-vs-cats）、+1银（whale升级）、+2超中位（denoising、spooky）
> - **P1 网络代理 + Docker CPU 4核限制（负效果）**: 联网使 LLM 使用大型预训练模型，但容器 CPU 限制（nano_cpus=4核）导致 DataLoader 严重瓶颈，GPU 大量空转，训练速度极慢（3-113 s/it）→ -4 个竞赛严重退化（siim: 有效→无效、ranzcr: 有效→无效、aptos: 超中位→低于中位、random-pizza: 铜牌→超中位）
> - **新发现 mse 假匹配 Bug（负效果）**: jigsaw-toxic 提交分数从 0.979 骤降至 0.886

---

## 二、本轮新增 Bugs 与问题分析

### 2.1 [CRITICAL-NEW] `mse` 子串假匹配导致 jigsaw-toxic 方向完全错误

#### Bug 详情

`core/orchestrator.py:_detect_metric_direction()` 使用 `if key in text`（子串匹配）检测指标方向。字典中存在 `"mse": True` 键，但 jigsaw-toxic 任务描述中包含短语：

```
"expressing themselves"
           ↑
       包含 "mse" 子串！
```

系统日志记录：
```
[metric_direction] 从 task_desc 检测到: 'mse' → lower_is_better=True
```

在 `"themselves"` 第 198 个字符处匹配到了 `"mse"` —— 这是 jigsaw-toxic 任务描述中的一个普通英文单词，与 MSE 指标完全无关。

#### 影响

| 指标 | 数值 |
|------|------|
| 被锁定方向 | `lower_is_better=True`（错误！jigsaw 使用 AUC = 越大越好） |
| 有效节点数 | 24 个 |
| 所有节点 lower_is_better | 100% = True（全部错误） |
| 系统"最佳"节点 | **最低 AUC = 0.909**（方向反转，选了最差的！） |
| 实际最优节点 AUC | **0.986**（超过铜牌线 0.986，可获铜牌！） |
| 实际提交 LB 得分 | 0.886（远低于中位线 0.981） |

V2 中 jigsaw 得分 0.979（使用 LLM 评估，有概率偶然正确）。V3 由于新增的 P0-1 修复锁定机制，第一次评估就用 `"mse"` 假匹配将方向永久锁定为错误值，导致全程反向选择节点。

#### 修复方案

```python
# 错误: 子串匹配
if key in text:  # "mse" 匹配 "themselves"

# 正确: 单词边界匹配
import re
pattern = r'\b' + re.escape(key) + r'\b'
if re.search(pattern, text, re.IGNORECASE):
    ...
```

**预计影响**: 修复后 jigsaw 可恢复 AUC 0.986 水平（铜牌线 0.986，边界情况），最低超中位（0.981）。

---

### 2.2 [CRITICAL-NEW] 提交验证器列顺序检查过严导致 detecting-insults 100% buggy

#### Bug 详情

`core/orchestrator.py:_validate_submission_format()` 中列名检查：

```python
# 错误：列表比较（顺序敏感）
if list(sub_df.columns) != list(sample_df.columns):
    errors.append("列名不匹配")
```

detecting-insults 竞赛的 `sample_submission.csv` 列顺序为 `['Insult', 'Date', 'Comment']`（不寻常的顺序）。LLM 生成的代码自然按照语义顺序排列列：`['Comment', 'Date', 'Insult']`。

**结果**：所有包含正确 3 列的提交都被列顺序检查拒绝（61 个节点）；另 78 个只生成 `['Insult']` 1 列的节点被 NaN 检查拒绝。

#### 内部最优方案评估

| 指标 | 数值 |
|------|------|
| 总节点数 | 100 |
| 有效提交 | **0**（100% buggy） |
| 内部最优 AUC | **0.901** |
| 金牌线 | 0.833 |
| 银牌线 | 0.823 |
| 预期奖牌 | **金牌**（0.901 >> 0.833） |

detecting-insults 内部最好的节点（TF-IDF + SVM，AUC=0.901）远超金牌线，只需提交验证器允许通过即可获金牌！

#### 修复方案

```python
# 修复1: 用集合比较替代列表比较（列不分顺序）
if set(sub_df.columns) != set(sample_df.columns):
    errors.append(f"列名不匹配: sub={sorted(sub_df.columns)}, sample={sorted(sample_df.columns)}")

# 修复2: 发现列存在但顺序不同时，自动重排
elif list(sub_df.columns) != list(sample_df.columns):
    sub_df = sub_df[sample_df.columns]  # 按 sample 顺序重排

# 修复3: NaN 检查仅针对目标列（不检查特征列）
target_col = "Insult"  # 或从 sample 推断
nan_count = sub_df[target_col].isna().sum()
if nan_count > 0:
    errors.append(f"{target_col} 列包含 {nan_count} 个 NaN 值")
```

**预计影响**: 修复后 detecting-insults 直接获得**金牌**。

---

### 2.3 [HIGH-NEW] Docker CPU 核数限制导致 DataLoader 瓶颈 → 图像类竞赛普遍超时

#### 问题详情

P1 修复引入了网络代理，允许容器访问互联网，LLM 因此调用 `torchvision.models.efficientnet_b0(weights='DEFAULT')` 等接口下载预训练权重。**经测量，下载速度正常（6-12 MB/s），下载本身不是根本瓶颈。**

真正根因是 MLE-bench 的 Docker 容器配置：

**位置：** `Reference/mle-bench/environment/config/container_configs/default.json`
```json
{
    "gpus": 1,
    "nano_cpus": 4e9,   ← 硬限 4 核 CPU
    "shm_size": "4G"
}
```

**CPU-GPU 流水线断裂链**：
```
RTX 4090 等待数据（GPU 大量空转）
         ↑
DataLoader（num_workers=0/2）
         ↑
CPU（4核限制）串行解码 JPEG → resize → normalize
         ↑
33k 高清医学图像（siim/ranzcr）
```

**实测训练速度**：

| 竞赛 | num_workers | 实测速度 | 单 epoch 耗时 | 时间预算 | 结果 |
|------|------------|---------|------------|---------|------|
| siim Node 3 | 2 | 3.19 s/it | ~38 min | 3600s | TimeoutError |
| siim Node 7 | **0** | **113 s/it** | **~17 小时** | 3600s | TimeoutError |
| ranzcr | — | — | — | 3600s | 78192s 后强制终止 |

Node 7 的注释写了 `"num_workers=0 is safer to avoid multiprocessing timeouts"`——LLM 的保守选择彻底断掉了 CPU-GPU 流水线。

**次要加剧因素：** `parallel_num: 2`（两个进程共抢同一块 GPU + 4 核 CPU），进一步分摊了有限的 CPU 资源。

**实际后果**:

| 竞赛 | V2 节点成功率 | V3 节点成功率 | 主要原因 |
|------|-------------|-------------|---------|
| siim-melanoma | 26.7% (4/15) | **0%** (0/14) | 12 TimeoutError（4核CPU+大图像）+ 1 哈希错误 |
| ranzcr-clip | ~50% (有 valid 提交) | **0%** (无 journal) | 全部超时，78192s 后容器被强制结束 |
| aptos2019 | 60%+ | 60% (但 5 超时) | 4核CPU下图像加载慢 + 部分下载消耗时间 |

#### 修复方案

**修复 1（Docker 配置）**：提升容器 CPU 核数限制

```json
// Reference/mle-bench/environment/config/container_configs/default.json
{
    "gpus": 1,
    "nano_cpus": 16e9,  ← 4核 → 16核
    "shm_size": "4G"
}
```

**修复 2（Prompt 约束）**：要求 Agent 代码中设置足够的 DataLoader workers

```
重要约束：
1. 必须检测并设置 num_workers：
   import os
   num_workers = min(4, os.cpu_count() or 1)
   DataLoader(dataset, num_workers=num_workers, pin_memory=True)
2. 禁止使用 num_workers=0（除非数据集 <1000 样本）
```

**预计影响**: siim/ranzcr/aptos 图像加载速度提升 3-4 倍，siim/ranzcr 有望恢复有效提交。

---

## 三、继承 Bug 与持续问题

### 3.1 [HIGH-继承] CV-LB 数据采样偏差

#### new-york-taxi（最严重：3.1x 差距）

| 指标 | V2 | V3 |
|------|-----|-----|
| 内部最优 CV | 2.79 (RMSE) | 3.44 (RMSE) |
| LB 提交得分 | 5.71 | **10.69** |
| CV-LB 倍数 | 2.0x | **3.1x** |

**根因**: 训练代码使用 `nrows=5_000_000`（固定读取前 500 万行）—— 这是时序数据，前 500 万行仅覆盖早期年份。模型学到的是早期出租车价格分布，测试集覆盖所有年份，导致严重分布偏移。

**修复**: 使用随机采样替代顺序采样：
```python
# 错误
df = pd.read_csv(train_path, nrows=5_000_000)

# 正确: 随机采样
df = pd.read_csv(train_path).sample(n=5_000_000, random_state=42)
# 或: 按年份分层采样以保持时序分布
```

#### denoising（0.005 差距，接近铜牌）

| 指标 | 数值 |
|------|------|
| 内部 3-fold CV | 0.019 (silver 级别！) |
| LB 提交得分 | 0.050 |
| 铜牌线 | 0.045 |
| 差距 | **仅 0.005！** |

内部 CV 银牌级别（0.019），但 LB 只有 0.050（铜牌线为 0.045，差 0.005）。训练集 144 张图像，每折只有 29 张验证图，CV 估计方差极大。

#### mlsp-birds（V2 铜牌消失：统计噪声）

V2 铜牌（0.887）来自内部 CV 0.763 的模型，正 CV-LB 差距 +0.124。V3 内部 CV 提升到 0.828，但 LB 只有 0.806（正常 -0.022 差距）。V2 铜牌是**幸运的正差距**，V3 是正常发挥。

仅 257 个训练样本，5-fold 每折只有 51 个验证样本 → CV 估计本质上不稳定。

---

### 3.2 [HIGH-继承] GA 进化从未触发问题

#### 架构描述

进化循环：
```
For each epoch:
  orchestrator._run_single_epoch(steps=10)  # 生成 explore/fix 节点
  if good_nodes >= population_size(=12):
    solution_evolution.run_epoch()  # 触发 merge/mutate（GA）
  else:
    skip GA
```

**关键问题**: 时间限制竞赛在 epoch 尚未完成时停止，GA 永远不会运行。

#### 受影响竞赛统计

| 竞赛 | 总节点 | 有效节点 | Merge | Mutate | GA触发原因 |
|------|--------|---------|-------|--------|-----------|
| tabular-may-2022 | 18 | 13 | 0 | 0 | 第1轮只有7个有效 < 12 |
| tabular-dec-2021 | 15 | 5 | 0 | 0 | 5 << 12 |
| dogs-vs-cats | 20 | 14 | 0 | 0 | 第1轮 ~7个有效 < 12 |
| new-york-taxi | 13 | 7 | 0 | 0 | 7 < 12 |
| siim-melanoma | 14 | 0 | 0 | 0 | 0 < 12 |
| aptos2019 | 20 | 12 | 0 | 0 | 时间到，epoch 未完 |

**tabular-may-2022 详细分析**：所有 18 个节点均为 explore 类型，最优内部 CV = 0.992，铜牌线 = 0.998（差 0.006）。GA 的 merge 操作本有可能通过集成多个 0.990+ 节点来突破这个瓶颈，但从未触发。

#### 修复方案

```python
# 当前（过严格）
if len(good_nodes) < self.population_size(=12):  # 从不提前触发
    return None

# 修改为（降低阈值，更积极触发）
min_ga_nodes = max(4, self.population_size // 3)  # 至少 4 个好节点就触发
if len(good_nodes) < min_ga_nodes:
    return None
```

---

### 3.3 [MEDIUM-继承] 提交验证器对非标准 sample_submission 文件名的处理

detecting-insults 的 sample_submission 文件名为 `sample_submission_null.csv`（非标准）。P0-2 修复使用了 `glob("sample_submission*.csv")` 来处理这种情况，这是正确的。但引入了新问题（见 2.2 节）。

---

## 四、逐竞赛深度分析

### A 层 — 奖牌（8 个）

#### 新增金牌：dogs-vs-cats（P0-1 修复直接效果）

| 指标 | V2 | V3 |
|------|-----|-----|
| 提交得分 (↓) | 0.064 (超中位) | **0.008 (金牌！)** |
| lower_is_better | 混乱（50%/50%） | **正确锁定** True |
| 最优节点选择 | 选了 logloss=0.208 | 选了 logloss=0.011 |
| 金牌线 | 0.039 | 0.039 |

V2 中 50% 的节点认为 lower_is_better=False（错误），导致系统选了 logloss=0.208 的劣质节点。V3 修复后，系统正确识别并锁定了 lower_is_better=True，选择了真正最优节点。**这是 P0-1 修复最直接的成功案例**。

#### 铜升银：whale-challenge

| 指标 | V2 | V3 |
|------|-----|-----|
| 提交得分 (↑) | 0.929 (铜牌) | **0.958 (银牌)** |
| 银牌线 | 0.950 | 0.950 |

59 个节点（46 explore + 12 merge + 1 mutate），124 个有效节点中最优 AUC=0.965。GA 进化有效地融合了多个探索节点的策略，实现了铜牌→银牌的跃升。

---

### B 层 — 超中位（5 个，非奖牌）

#### denoising（接近铜牌 0.005 差距）

| 指标 | 数值 |
|------|------|
| LB 得分 (↓) | 0.050 |
| 铜牌线 (↓) | 0.045 |
| 内部最优 CV | 0.019 (silver 级别！) |
| CV-LB 差距 | 2.7x |

68 个节点，53 个有效（78%），使用 UNet 3-fold CV。内部 CV 达到 silver 级别，但训练集太小（144张），测试集分布不同，CV-LB 差距 2.7x。

#### tabular-may-2022（差 0.006 达铜牌）

| 指标 | 数值 |
|------|------|
| LB 得分 (↑) | 0.992 |
| 铜牌线 (↑) | 0.998 |
| 内部最优 CV | 0.992 |
| GA 触发 | 否（18节点全为 explore） |

最优节点在步骤 9 就已达到 0.9921，后续 8 个节点（步骤 10-17）都只能在 0.9920 附近徘徊。GA 的 merge/mutate 可能是突破这个平台的关键，但因 good_nodes < 12 而从未触发。

#### spooky-author（P0-1 修复正效果）

| 指标 | V2 | V3 |
|------|-----|-----|
| 提交得分 (↓) | 0.844 | **0.357** |
| lower_is_better | 58% False（错误） | **正确锁定 True** |
| 结果 | 低于中位 | **超中位** |

142 个节点，138 个有效（97%！ 最高有效率）。V3 正确选择了 logloss=0.357 的节点（低于中位线 0.419），而 V2 错误选择了 0.868。距铜牌（0.294）还有 0.063 的差距，但进入了超中位区间。

---

### C 层 — 低于中位（8 个）

#### jigsaw-toxic（本轮最严重新增退化）

| 指标 | V2 | V3 |
|------|-----|-----|
| LB 得分 (↑) | 0.979 | **0.886** |
| 中位线 | 0.981 | 0.981 |
| 铜牌线 | 0.986 | 0.986 |
| 实际最优 AUC | ~0.980 | **0.986** |
| 提交的节点 AUC | ~0.980 | **0.909（最差节点！）** |

V3 的 `mse` 假匹配锁定了错误方向（lower_is_better=True），导致系统选取了 AUC **最低**的节点（0.909）作为提交。实际内部最优节点 AUC=0.986（达到铜牌线！），但从未被选中。

#### aptos2019（网络代理致 5 次 EfficientNet 超时）

20 个节点中 5 个（25%）因 EfficientNet_b0 下载超时（全部精确 3600s）。有效节点 12 个，最优 CV 0.892，LB 0.868（-2.7% CV-LB 差距）。勉强低于中位（0.889）。

V2 中 aptos 为超中位（0.893），V3 由于超时浪费了 25% 的计算预算，最终有效节点数量减少，未能找到更好的解。

#### leaf-classification（长期难题）

| 指标 | V2 | V3 | 铜牌线 |
|------|-----|-----|--------|
| LB 得分 (↓) | 0.266 | 0.333 | 0.015 |

两轮均远低于铜牌线（14-22x 差距）。内部 CV 0.216 vs LB 0.333（1.54x CV-LB 差距），说明方案本身质量也不够好。这类细粒度多分类任务（120 种狗品种）需要更专业的预训练特征，但受网络代理影响无法可靠下载。

---

### D 层 — 无效/无提交（4 个）

#### detecting-insults（内部金牌被验证器封锁）

| 指标 | 数值 |
|------|------|
| 有效节点数 | 0/100 (100% buggy) |
| 失败原因 1 | 61 节点：列顺序错误（有3列但顺序不同） |
| 失败原因 2 | 78 节点：仅1列（缺失 Date/Comment） |
| 内部最优 AUC | **0.901** |
| 金牌线 | 0.833 |
| 预期奖牌 | **金牌** |

如果修复列顺序检查，这将直接成为金牌竞赛。

#### siim-isic-melanoma（网络代理致全程失效）

| 指标 | V2 | V3 |
|------|-----|-----|
| 有效节点率 | 26.7% (4/15) | **0%** (0/14) |
| LB 得分 | 0.860 | null |

12 个 TimeoutError（EfficientNet 训练太慢，1490s/epoch × 3 epoch = 4470s >> 3600s），1 个哈希错误（下载的 EfficientNetV2_s 文件损坏），1 个 KeyError。

#### ranzcr-clip（78192s 容器被杀，journal 为空）

V3 容器运行了 21.7 小时（78192s），系统日志显示只有 2 个节点被尝试，均超时。Journal 文件为空（容器在写入前被强制终止）。V2 有有效提交（0.844），说明 V3 的网络代理使问题更严重（下载失败导致 epoch 更慢）。

#### text-normalization-english（MLE-bench 评分崩溃）

| 指标 | 数值 |
|------|------|
| 提交文件 | 993465 行，格式完全正确 |
| ID 匹配 | 100% |
| NaN | 0 |
| valid_submission | False（评分程序崩溃） |

根因：`answers.csv` 含 16 个 NaN 值 → sklearn `accuracy_score` 内部 `np.unique()` 遇到 str+float 混合类型 → `TypeError`。这是 **MLE-bench 评分系统的 bug**，与我们的提交无关。

手动模拟评分：实际准确率约 0.955（低于铜牌线 0.990）。即使评分系统修复，本轮也无法获得奖牌。

---

## 五、优先修复计划

### P0 — 关键 Bug（立即修复，单竞赛影响 ≥ 1 枚奖牌）

| 优先级 | Bug | 文件 | 修复方法 | 预计增益 |
|--------|-----|------|---------|---------|
| **P0-A** | `mse` 子串假匹配 | `core/orchestrator.py:_detect_metric_direction()` | `re.search(r'\b' + key + r'\b', text)` | +1铜牌（jigsaw） |
| **P0-B** | 提交验证器列顺序过严 | `core/orchestrator.py:_validate_submission_format()` | 集合比较 + 自动重排列 + 仅验证目标列 NaN | +1金牌（detecting-insults） |

**P0-A 代码修复**：

```python
# core/orchestrator.py: _detect_metric_direction()
import re

def _detect_metric_direction(self, text: str) -> Optional[bool]:
    text_lower = text.lower()
    for key, lower_is_better in METRIC_DIRECTION.items():
        # 使用单词边界匹配，避免 "mse" 匹配 "themselves"
        pattern = r'\b' + re.escape(key.lower()) + r'\b'
        if re.search(pattern, text_lower):
            log_msg("INFO", f"[metric_direction] 检测到: '{key}' → lower_is_better={lower_is_better}")
            return lower_is_better
    return None
```

**P0-B 代码修复**：

```python
# core/orchestrator.py: _validate_submission_format()
def _validate_submission_format(self, node_id: str) -> dict:
    ...
    # 集合比较（不依赖顺序）
    sub_cols = set(sub_df.columns)
    sample_cols = set(sample_df.columns)

    if sub_cols != sample_cols:
        errors.append(f"列名不匹配: submission={sorted(sub_cols)}, sample={sorted(sample_cols)}")
    elif list(sub_df.columns) != list(sample_df.columns):
        # 列存在但顺序不同 → 自动重排
        sub_df = sub_df[sample_df.columns]
        log_msg("INFO", f"列顺序已自动调整为 sample 顺序")

    # 仅检查目标列（第一个非 id 列）的 NaN
    target_col = [c for c in sample_df.columns if c.lower() != 'id'][0]
    nan_count = sub_df[target_col].isna().sum()
    if nan_count > 0:
        errors.append(f"目标列 '{target_col}' 包含 {nan_count} 个 NaN 值")
    ...
```

---

### P1 — 高优先级（影响 2-4 个竞赛）

| 优先级 | 问题 | 修复方法 | 预计增益 |
|--------|------|---------|---------|
| **P1-A** | Docker CPU 4核限制导致 DataLoader 瓶颈 | `default.json: nano_cpus: 4e9 → 16e9` + Prompt 要求 `num_workers≥4` | siim/ranzcr/aptos 恢复有效训练，减少 TimeoutError |
| **P1-B** | new-york-taxi 时序采样偏差 | 系统提示中提醒随机采样大数据集 | CV-LB 差距从 3.1x 降低到 <1.5x |

**P1-A 修复 1 — Docker 配置**（`Reference/mle-bench/environment/config/container_configs/default.json`）：

```json
{
    "gpus": 1,
    "nano_cpus": 16e9,
    "shm_size": "4G"
}
```

**P1-A 修复 2 — Prompt 约束**（在 system prompt 中添加）：

```
重要约束：
必须为 DataLoader 设置足够的 num_workers，充分利用多核 CPU：
  import os
  num_workers = min(4, os.cpu_count() or 1)
  DataLoader(dataset, num_workers=num_workers, pin_memory=True)
禁止使用 num_workers=0（除非数据集 <1000 样本）。
```

---

### P2 — 中优先级（架构层改进）

| 优先级 | 问题 | 修复方法 | 预计增益 |
|--------|------|---------|---------|
| **P2-A** | GA 种群阈值过高（12）导致多竞赛从不触发 | 降至 `min_ga_nodes = max(4, population_size // 3)` | tabular-may-2022 等可能突破瓶颈 |
| **P2-B** | denoising CV-LB 差距 | TTA（测试时增强）+ 更保守的 CV 策略 | 可能突破铜牌线（差 0.005） |
| **P2-C** | METRIC_DIRECTION 字典完善 | 为 jigsaw AUC 等添加明确映射，减少 LLM fallback 依赖 | 更稳定的方向锁定 |

---

## 六、总体效果预测（修复后）

| 场景 | 预计奖牌数 | 获奖率 |
|------|-----------|-------|
| 当前 V3 | 8 (5G2S1B) | 36.4% |
| +P0 修复（jigsaw+detecting） | 10 (6G2S2B) | 45.5% |
| +P0+P1 修复（网络代理约束） | 11-12 | 50-55% |
| +P0+P1+P2 修复（GA+采样） | 12-13 | 55-59% |
| 80% 目标 | 18/22 | 80% |

**结论**: 即使修复所有已知 bug，距离 80% 目标仍有较大差距。根本瓶颈在于**方案质量**——许多竞赛（leaf-classification, new-york-taxi, mlsp-birds 等）需要更专业化的 ML 方案才能进入奖牌区间，而当前的 LLM 自动生成方案质量尚不足以覆盖所有类型竞赛。

---

## 七、关键洞察总结

### 7.1 正效果（V3 新功能的价值）

1. **P0-1 lower_is_better 修复** 直接带来 +1金（dogs-vs-cats）+1银（whale）+2超中位（spooky, denoising）
2. **P0-2 P0-3 修复**（提交验证 + 文件保护）提升了整体代码质量，valid rate 从 v2 的 ~60% 提升到 v3 各竞赛平均 73%

### 7.2 负效果（V3 新功能的代价）

1. **P1 网络代理 + Docker CPU 限制**：使能互联网 = LLM 使用大型预训练模型（下载速度本身正常 6-12 MB/s），但 Docker 容器 `nano_cpus=4核` 限制使 DataLoader 成为严重瓶颈（GPU 空等 CPU 解码图像）→ 训练速度 3~113 s/iter → 超时 → 2 个竞赛变无效提交，2 个竞赛性能下降
2. **P0-1 修复副作用（mse 假匹配）**：锁定机制放大了假匹配的危害，V2 的随机 LLM 评估反而不会持续选错方向

### 7.3 统计方差问题

mlsp-birds（257 样本 / 5 fold）类型竞赛的 CV 估计天然不稳定，V2 铜牌来自 +0.124 的幸运 CV-LB 正差距，不可复制。这类竞赛需要不同策略（ensemble/bootstrap）来提高 LB 稳定性。

### 7.4 架构短板

GA 进化（merge/mutate）在当前配置下对 60% 以上的竞赛从不触发（population_size=12，steps_per_epoch=10），导致进化的核心价值未能在多数竞赛中体现。

---

*分析完成时间：2026-02-19 | 分析师：Claude Sonnet 4.6*
