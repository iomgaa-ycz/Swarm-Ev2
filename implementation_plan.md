# MLE-Bench 适配修改计划

## 1.1 摘要

为 Swarm-Ev2 项目创建 MLE-Bench 评测框架适配层，包括 Docker 容器化、环境桥接适配器、专用配置文件，使项目能在 MLE-Bench 标准化容器中运行并输出合规的提交结果。

## 1.2 审查点

**已确认决策：**

| # | 决策项 | 结论 |
|---|--------|------|
| 1 | Agent ID | 沿用 `swarm-evo` |
| 2 | BGE-M3 Embedding 模型 | **必须预下载**（详见附录 A） |
| 3 | Python 版本 | 使用 3.11（与基础镜像一致） |
| 4 | 数据验证 | 允许放宽 |
| 5 | 默认步数/时间 | 100 步 / 12 小时（43200 秒） |

**待确认：无**

## 1.3 拟议变更

### 架构总览

```
MLE-Bench 容器运行流程:
┌───────────────────────────────────────────────────────┐
│ mlebench-env (基础镜像)                                │
│  ├── /entrypoint.sh → 启动 grading_server (port 5000) │
│  ├── /home/data/     ← 竞赛数据（只读挂载）             │
│  ├── /home/description.md ← 竞赛描述                   │
│  └── /private/data/  ← 私有数据（只读挂载）             │
├───────────────────────────────────────────────────────┤
│ Swarm-Ev2 Agent 层 (本次新增)                          │
│  ├── /home/agent/    ← 项目全部代码                     │
│  ├── /home/agent/start.sh → 激活 conda + 启动适配器    │
│  └── /home/agent/run_mle_adapter.py                   │
│       ├── 1. 映射环境变量（API_KEY→OPENAI_API_KEY 等）  │
│       ├── 2. 加载 config/mle_bench.yaml (OmegaConf)   │
│       ├── 3. 构建 MLE-Bench 兼容工作空间               │
│       ├── 4. 初始化 Agent + 进化组件 (复用 main.py)     │
│       ├── 5. 运行双层进化主循环                          │
│       └── 6. 复制结果 → /home/submission/               │
├───────────────────────────────────────────────────────┤
│ 输出                                                    │
│  ├── /home/submission/submission.csv ← 必须             │
│  ├── /home/code/solution.py                             │
│  └── /home/logs/system.log + journal.json               │
└───────────────────────────────────────────────────────┘
```

---

### 配置系统隔离设计（关键）

两套配置系统**完全独立运作，互不干扰**：

```
项目根目录/
├── config.yaml              ← mle-bench Registry 专用
│   解析器: yaml.safe_load()
│   语法:   ${{ secrets.API_KEY }}
│   读者:   mle-bench registry.py (glob "**/config.yaml")
│
├── config/
│   ├── default.yaml         ← Swarm-Ev2 本地开发
│   │   解析器: OmegaConf.load()
│   │   语法:   ${env:OPENAI_API_KEY}
│   │   读者:   utils/config.py load_config()
│   │
│   └── mle_bench.yaml       ← Swarm-Ev2 容器运行
│       解析器: OmegaConf.load()
│       语法:   ${env:OPENAI_API_KEY}
│       读者:   run_mle_adapter.py → load_config(path)
```

**不冲突的原因：**
1. mle-bench Registry 只扫描文件名 = `config.yaml` 的文件 → `default.yaml`/`mle_bench.yaml` 不会被扫描
2. `load_config()` 显式指定路径 → 不会加载根目录的 `config.yaml`
3. 两种 env var 语法完全不同 → 即使误读也不会崩溃（`yaml.safe_load` 把 `${env:XX}` 当字符串）

**隔离规则：** `config/` 子目录下的文件**绝对不能**命名为 `config.yaml`

---

### 新建文件清单 (8 个)

#### 1. `run_mle_adapter.py` `[NEW]` — MLE-Bench 适配器入口

> 核心桥接文件，将 MLE-Bench 容器环境映射到 Swarm-Ev2 的执行管线。

**关键函数:**

| 函数 | 功能 |
|------|------|
| `map_env_vars()` | 将 MLE-Bench 环境变量映射为 Swarm-Ev2 格式。**必须在任何 import 之前执行**。 |
| `setup_workspace(config, data_dir, description)` | 替代 `build_workspace()`，直接构建 MLE-Bench 兼容的工作空间。 |
| `run_adapter()` | 主执行函数（同步），10 个阶段。 |
| `copy_results(journal, workspace_dir)` | 从 Journal 获取最佳方案，复制结果到 MLE-Bench 标准目录。 |

**环境变量映射逻辑（必须在任何 import 之前）:**

```python
# map_env_vars() 伪代码
API_KEY       → OPENAI_API_KEY
API_BASE      → OPENAI_BASE_URL
MODEL_NAME    → LLM_MODEL
# TIME_LIMIT_SECS 和 STEP_LIMIT 直接由 mle_bench.yaml 的 ${env:} 消费
```

**与 `main.py` 的关键差异:**

| 方面 | main.py | run_mle_adapter.py |
|------|---------|-------------------|
| 配置加载 | `load_config()` → default.yaml | `load_config(config/mle_bench.yaml, use_cli=False)` |
| 数据验证 | `validate_dataset()` 严格校验 | 放宽（MLE-Bench 保证数据完整） |
| 工作空间 | `build_workspace()` 从 data_dir 复制 | `setup_workspace()` 从 /home/data symlink |
| description | 从 data_dir/description.md | 从 /home/description.md 或 /home/data/description.md |
| 结果输出 | 屏幕 + 报告文件 | 复制到 /home/submission/ + /home/code/ + /home/logs/ |
| 错误处理 | raise + print | try/except 兜底（容器内不能 crash 无输出） |

---

#### 2. `Dockerfile` `[NEW]` — Agent 容器镜像定义

```dockerfile
FROM mlebench-env

ARG SUBMISSION_DIR=/home/submission
ARG LOGS_DIR=/home/logs
ARG CODE_DIR=/home/code
ARG AGENT_DIR=/home/agent

RUN mkdir -p ${LOGS_DIR} ${CODE_DIR} ${AGENT_DIR}

ARG CONDA_ENV_NAME=agent

# 1. 复制依赖清单（利用 Docker 层缓存）
COPY requirements_agent.txt ${AGENT_DIR}/requirements.txt

# 2. 安装系统级依赖（OpenCV 等）
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 && rm -rf /var/lib/apt/lists/*

# 3. 安装 Python 依赖（基础镜像已有 PyTorch 2.2, sklearn 等）
RUN conda run -n ${CONDA_ENV_NAME} pip install -r ${AGENT_DIR}/requirements.txt && \
    conda clean -afy

# 4. 预下载 BGE-M3 Embedding 模型（Skill 进化必需）
ENV HF_ENDPOINT=https://hf-mirror.com
ENV MODEL_SAVE_PATH=${AGENT_DIR}/embedding-models/bge-m3
RUN mkdir -p ${MODEL_SAVE_PATH}
COPY scripts/download_model.py ${AGENT_DIR}/scripts/download_model.py
RUN conda run -n ${CONDA_ENV_NAME} python ${AGENT_DIR}/scripts/download_model.py && \
    chmod -R 555 ${MODEL_SAVE_PATH}

# 5. 复制项目代码
COPY . ${AGENT_DIR}
```

**关键设计:**
- 基础镜像 `mlebench-env` 已含 PyTorch 2.2、TensorFlow 2.17、sklearn
- 分层构建：requirements → 模型下载 → 代码（缓存友好）
- conda 环境名 `agent`（与基础镜像一致，Python 3.11）

---

#### 3. `start.sh` `[NEW]` — 容器启动脚本

```bash
#!/bin/bash
cd /home/agent
export PYTHONPATH=$PYTHONPATH:/home/agent
eval "$(conda shell.bash hook)"
conda activate agent
mkdir -p /home/code /home/logs
python /home/agent/run_mle_adapter.py
```

---

#### 4. `config.yaml` `[NEW]` — MLE-Bench Agent 注册配置（根目录）

> ⚠️ 此文件使用 `${{ secrets.XX }}` 语法，由 mle-bench Registry 的 `yaml.safe_load()` 解析。
> 与 Swarm-Ev2 的 OmegaConf 配置系统**完全无关**。

```yaml
vars:
  time_limit: &time_limit 43200  # 12 小时
  step_limit: &step_limit 100

defaults: &defaults
  start: swarm-evo/start.sh
  dockerfile: swarm-evo/Dockerfile
  env_vars: &env_vars
    TIME_LIMIT_SECS: *time_limit
    STEP_LIMIT: *step_limit

swarm-evo:
  <<: *defaults
  env_vars:
    <<: *env_vars
    API_KEY: ${{ secrets.API_KEY }}
    API_BASE: ${{ secrets.API_BASE }}
    MODEL_NAME: ${{ secrets.MODEL_NAME }}

swarm-evo/dev:
  <<: *defaults
  env_vars:
    <<: *env_vars
    API_KEY: ${{ secrets.API_KEY }}
    API_BASE: ${{ secrets.API_BASE }}
    MODEL_NAME: ${{ secrets.MODEL_NAME }}
    STEP_LIMIT: 10
```

---

#### 5. `config/mle_bench.yaml` `[NEW]` — MLE-Bench 专用 OmegaConf 配置

> 完整的 YAML 配置，基于 `default.yaml` 覆盖容器路径和参数。
> ⚠️ 文件名**不能**叫 `config.yaml`（避免被 mle-bench Registry 扫描）。

**与 default.yaml 的差异：**

| 配置项 | default.yaml | mle_bench.yaml | 原因 |
|--------|-------------|----------------|------|
| `project.workspace_dir` | `./workspace` | `/home/workspace` | 容器内路径 |
| `project.log_dir` | `./logs` | `/home/logs` | 容器日志目录 |
| `data.data_dir` | `./datasets/public` | `/home/data` | MLE-Bench 数据挂载点 |
| `environment.conda_env_name` | `Swarm-Evo` | `agent` | 容器 conda 环境 |
| `agent.max_steps` | `150` | `${env:STEP_LIMIT, 100}` | 从 MLE-Bench 读取 |
| `agent.time_limit` | `43200` | `${env:TIME_LIMIT_SECS, 43200}` | 从 MLE-Bench 读取 |
| `evolution.skill.embedding_model_path` | `./embedding-models/bge-m3` | `/home/agent/embedding-models/bge-m3` | 容器内模型路径 |
| `llm.feedback.model` | `glm-4.6` | `${env:LLM_MODEL}` | MLE-Bench 统一用同一模型 |

其余所有配置项（search、logging、evolution.experience_pool、evolution.solution、evolution.agent）保持与 default.yaml 一致。

---

#### 6. `requirements_agent.txt` `[NEW]` — Docker 锁定依赖

生成方式:
```bash
conda run -n Swarm-Evo pip freeze | grep -v -E "^(pytest|ruff|mypy|pytest-)" > requirements_agent.txt
```

**需额外安装的核心依赖（基础镜像未包含）:**

| 包 | 用途 |
|----|------|
| `omegaconf>=2.3.0` | 配置管理 |
| `openai>=1.0.0` | LLM 客户端 |
| `anthropic>=0.20.0` | Anthropic LLM |
| `langgraph` | Agent 编排（LangGraph） |
| `langchain-openai` | LangChain OpenAI |
| `sentence-transformers>=2.6.0` | BGE-M3 推理 |
| `rich>=13.0.0` | 美化输出 |
| `python-dotenv>=1.0.0` | .env 加载 |
| `backoff>=2.2.0` | 重试机制 |
| `dataclasses-json>=0.6.0` | 数据序列化 |
| `humanize>=4.0.0` | 人性化格式 |
| `genson>=1.2.0` | JSON Schema |

---

#### 7. `.dockerignore` `[NEW]`

```
.git
workspace
logs
tests
docs
Reference
embedding-models
datasets
__pycache__
*.pyc
.DS_Store
.vscode
.env
draft.md
implementation_plan.md
```

---

#### 8. `scripts/download_model.py` `[NEW]`

从参考项目复制：`huggingface_hub.snapshot_download()` 下载 `BAAI/bge-m3` 到 `MODEL_SAVE_PATH`。

---

### 修改文件清单 (2 个)

#### 9. `utils/workspace_builder.py` `[MODIFY]`

| 函数 | 修改内容 |
|------|---------|
| `validate_dataset()` | 放宽数据文件检查：".csv 或 .zip 或 .parquet 或包含子目录" |
| `build_workspace()` | `/home/data` 只读场景下对目录也创建 symlink（当前仅处理文件） |

#### 10. `utils/config.py` `[MODIFY]`

| 函数 | 修改内容 |
|------|---------|
| `validate_config()` | description.md 自动检测增加 `/home/description.md` 候选路径 |
| `validate_config()` | data_dir 存在但无 description.md 时改为 WARNING 而非 ERROR |

---

### 不修改的文件

| 文件/模块 | 原因 |
|-----------|------|
| `main.py` | 本地运行入口保持不变 |
| `agents/*` | Agent 实现与运行环境无关 |
| `core/*` | 编排/状态/进化逻辑不变 |
| `utils/logger_system.py` | 日志系统通过 config 指定路径即可 |
| `utils/prompt_manager.py` | Prompt 管理不变 |

---

## 1.4 验证计划

### 阶段 1: 本地配置验证

```bash
# 验证 mle_bench.yaml 可被 OmegaConf 正常加载
conda run -n Swarm-Evo python -c "
from utils.config import load_config
import os
os.environ['OPENAI_API_KEY'] = 'test-key'
os.environ['OPENAI_BASE_URL'] = 'https://api.test.com'
os.environ['LLM_MODEL'] = 'gpt-4'
cfg = load_config(config_path='config/mle_bench.yaml', use_cli=False)
assert str(cfg.data.data_dir) == '/home/data'
assert cfg.environment.conda_env_name == 'agent'
print('✅ 配置加载验证通过')
"
```

### 阶段 2: Docker 构建验证

```bash
# 构建 Agent 镜像
docker build --platform=linux/amd64 -t swarm-evo .

# 验证文件存在
docker run --rm swarm-evo ls /home/agent/run_mle_adapter.py
docker run --rm swarm-evo ls /home/agent/embedding-models/bge-m3/

# 验证依赖可导入
docker run --rm swarm-evo conda run -n agent python -c \
  "import omegaconf, langgraph, sentence_transformers; print('✅ 依赖验证通过')"
```

### 阶段 3: MLE-Bench 集成验证

```bash
python run_agent.py \
  --agent-id swarm-evo/dev \
  --competition-set experiments/splits/spaceship-titanic.txt \
  --n-workers 1

# 验证: /home/submission/submission.csv 存在
```

### 阶段 4: 评分验证

```bash
mlebench grade --submission runs/<latest>/submissions.jsonl
# 验证: 生成 grading_report.json
```

---

## 附录 A: BGE-M3 模型使用分析

**结论：必须预下载。**

BGE-M3 在 Skill 进化链路中被调用：

```
Agent 进化触发（每 3 Epoch, 需 ≥20 条经验池记录）
  └→ _update_skill_pool()
       └→ skill_manager.evolve_skills()
            ├→ SkillExtractor._embed_texts()      ← BGE-M3
            ├→ SkillManager._is_duplicate()         ← BGE-M3
            └→ SkillManager._merge_similar_skills() ← BGE-M3
```

MLE-Bench 默认 100 步（steps_per_epoch=10 → 10 Epochs），经验池可积累 ~100 条记录。
第 3/6/9 Epoch 均会触发 Agent 进化，模型**必定被加载和调用**。

`CodeEmbeddingManager` 采用懒加载，但首次调用时若模型不存在且无网络 → 运行时崩溃。

---

## 附录 B: 环境变量映射表

| MLE-Bench 传入 | 适配器映射目标 | mle_bench.yaml 消费 |
|---------------|-------------|---------------------|
| `API_KEY` | → `OPENAI_API_KEY` | `${env:OPENAI_API_KEY}` |
| `API_BASE` | → `OPENAI_BASE_URL` | `${env:OPENAI_BASE_URL}` |
| `MODEL_NAME` | → `LLM_MODEL` | `${env:LLM_MODEL}` |
| `TIME_LIMIT_SECS` | 直接使用 | `${env:TIME_LIMIT_SECS, 43200}` |
| `STEP_LIMIT` | 直接使用 | `${env:STEP_LIMIT, 100}` |
| `COMPETITION_ID` | adapter 内部 | 用于日志和兜底描述 |

## 附录 C: 容器内目录结构

```
/home/
├── data/                      # 竞赛数据（只读挂载）
├── description.md             # 竞赛描述（只读）
├── instructions.txt           # MLE-Bench 指令（只读）
├── submission/                # ← submission.csv 输出至此
├── code/                      # ← solution.py 输出至此
├── logs/                      # ← system.log + journal.json
├── workspace/                 # Swarm-Ev2 工作空间（adapter 创建）
│   ├── input/                 # → /home/data symlink
│   ├── working/
│   ├── submission/
│   ├── best_solution/
│   └── logs/
└── agent/                     # Swarm-Ev2 项目代码
    ├── run_mle_adapter.py     # 适配器入口
    ├── start.sh               # 启动脚本
    ├── config.yaml            # mle-bench 注册（不被 OmegaConf 读取）
    ├── config/
    │   ├── default.yaml       # 本地配置（不被容器使用）
    │   └── mle_bench.yaml     # 容器配置（adapter 显式加载）
    ├── core/
    ├── agents/
    ├── utils/
    ├── benchmark/
    ├── embedding-models/bge-m3/
    └── scripts/download_model.py
```
