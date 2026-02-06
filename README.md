# Swarm-Ev2

**双层群体智能驱动的自动化 ML 系统**

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Phase](https://img.shields.io/badge/Phase-2%20%E6%A0%B8%E5%BF%83Agent-yellow.svg)]()
[![测试覆盖率](https://img.shields.io/badge/%E6%B5%8B%E8%AF%95%E8%A6%86%E7%9B%96%E7%8E%87-80%25-brightgreen.svg)]()

---

## 项目概述

Swarm-Ev2 是一个基于**双层群体智能**（Agent 层 + Solution 层）与**进化算法**的多 Agent 系统，旨在自动化解决复杂的机器学习问题（如 Kaggle 竞赛、MLE-Bench 评测）。

### 核心特性

- 🧠 **双层群体智能**: Agent 群体协作 + Solution 群体进化
- 🔄 **自我进化**: Agent 能力持续提升，Solution 基因池演化
- 🎯 **目标驱动**: 自动探索 + 评估 + 优化，无需人工干预
- 📊 **可观测性**: 完整的日志系统和性能指标追踪
- 🧪 **可测试**: TDD 驱动开发，测试覆盖率 80%+

---

## 快速开始

### 环境要求

- Python 3.10+
- Conda（推荐）

### 安装

```bash
# 1. 克隆仓库
git clone <repository-url>
cd Swarm-Ev2

# 2. 创建 Conda 环境
conda create -n Swarm-Evo python=3.10.19
conda activate Swarm-Evo

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env 填写 API Keys
```

### 配置 API Keys

在 `.env` 文件中配置：

```bash
# OpenAI API Key (必填)
OPENAI_API_KEY=sk-your-openai-api-key-here

# Anthropic API Key (可选，如需使用 Claude 模型)
ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key-here

# GLM API Key (可选，智谱 AI，用于 glm-4.6 等模型)
GLM_API_KEY=your-glm-api-key-here
```

### 运行测试

```bash
# 运行所有测试
pytest tests/unit/ -v

# 查看测试覆盖率（utils + core 模块）
pytest tests/unit/ --cov=utils --cov=core --cov-report=term-missing

# 代码格式化和检查
ruff format utils/ core/ tests/
ruff check utils/ core/ tests/ --fix
```

---

## 项目结构

```
Swarm-Ev2/
├── config/                    # 配置文件
│   └── default.yaml          # 主配置文件
├── agents/                    # Agent 层
│   ├── __init__.py           # 模块导出
│   ├── base_agent.py         # Agent 抽象基类 + AgentContext
│   └── coder_agent.py        # 代码生成 Agent (LLM重试+响应解析)
├── core/                      # 核心模块
│   ├── state/                # 核心数据结构
│   │   ├── __init__.py       # 导出 Node, Journal, Task
│   │   ├── node.py           # 解决方案节点 (22字段 + 4方法)
│   │   ├── journal.py        # 解决方案 DAG (11方法 + parse_solution_genes)
│   │   └── task.py           # 任务定义 (8字段)
│   ├── backend/              # LLM 后端抽象层
│   │   ├── __init__.py       # 统一查询接口 (query)
│   │   ├── backend_openai.py # OpenAI + GLM 后端
│   │   ├── backend_anthropic.py # Anthropic 后端
│   │   └── utils.py          # 消息格式化 + 重试机制
│   └── executor/             # 代码执行
│       ├── interpreter.py    # 执行沙箱 (超时控制)
│       └── workspace.py      # 工作空间管理
├── utils/                     # 工具模块
│   ├── config.py             # 配置管理 (OmegaConf + YAML)
│   ├── logger_system.py      # 日志系统 (双通道输出)
│   ├── file_utils.py         # 文件操作工具
│   ├── data_preview.py       # 数据预览生成
│   ├── metric.py             # 评估指标工具
│   ├── response.py           # LLM 响应解析
│   └── prompt_builder.py     # Prompt 构建器
├── tests/                     # 测试目录
│   ├── unit/                 # 单元测试 (59 个测试用例)
│   └── integration/          # 集成测试
├── docs/                      # 文档
│   ├── CODEMAPS/             # 架构文档
│   │   ├── architecture.md   # 整体架构
│   │   ├── backend.md        # 后端模块详解
│   │   └── data.md           # 数据流与配置
│   └── plans/                # 实施计划
│       └── phase1_infrastructure.md
├── logs/                      # 日志输出 (自动生成)
│   ├── system.log            # 文本日志
│   └── metrics.json          # 结构化日志
├── workspace/                 # 工作空间 (自动生成)
│   ├── input/                # 输入数据
│   ├── working/              # Agent 工作目录
│   └── submission/           # 提交文件
├── .env.example              # 环境变量模板
├── requirements.txt          # Python 依赖
├── CLAUDE.md                 # AI Agent 开发规范
└── README.md                 # 本文件
```

---

## 配置管理

### 配置优先级（从高到低）

1. **CLI 参数** (`--key=value`) - 最高优先级
2. **系统环境变量** (`export VAR=value`)
3. **.env 文件** (`VAR=value`)
4. **YAML 配置文件** (`key: value`) - 最低优先级

### 示例

```bash
# 使用默认配置
python main.py --data.data_dir=./datasets/titanic

# 覆盖配置
python main.py \
  --data.data_dir=./datasets/titanic \
  --llm.code.model=gpt-3.5-turbo \
  --agent.max_steps=30
```

详细配置说明参见 [docs/CODEMAPS/data.md](docs/CODEMAPS/data.md)。

---

## 开发指南

### Phase 实施状态

| Phase | 状态 | 说明 |
|-------|------|------|
| Phase 1 | 🟢 已完成 | 配置系统、日志系统、文件工具 ✅<br>核心数据结构（Node/Journal/Task）✅<br>后端抽象层（OpenAI/Anthropic/GLM）✅ |
| Phase 2 | 🟡 进行中 | 执行层（Interpreter/WorkspaceManager）✅<br>工具增强（data_preview/metric/response）✅<br>Agent 抽象（BaseAgent/PromptBuilder）✅<br>**CoderAgent（5次LLM重试，92%覆盖）✅**<br>Orchestrator 待实现 |
| Phase 3 | 🔴 未开始 | 搜索算法（MCTS/GA）、并行评估 |
| Phase 4 | 🔴 未开始 | 进化算法、经验池、Meta-Agent |
| Phase 5 | 🔴 未开始 | 端到端测试、MLE-Bench 适配 |

### 架构文档

- [项目架构概览](docs/CODEMAPS/architecture.md) - 分层架构、模块依赖
- [后端模块详解](docs/CODEMAPS/backend.md) - 配置、日志、测试系统
- [数据流与配置管理](docs/CODEMAPS/data.md) - 配置加载、工作空间

### 开发规范

**必读**: [CLAUDE.md](CLAUDE.md) - AI Agent 和人类开发者的统一规范

核心原则：
- ✅ **MVP 优先**: 严禁过度工程化
- ✅ **TDD 驱动**: 先写测试，后写实现
- ✅ **类型注解**: 强制所有函数包含完整类型
- ✅ **中文文档**: 所有 Docstring 和注释使用简体中文
- ✅ **测试覆盖**: 最低 80% 覆盖率

---

## 核心功能

### Phase 1: 基础设施（已完成）
- [x] **配置系统** - OmegaConf + YAML，支持 CLI/环境变量覆盖
- [x] **日志系统** - 双通道输出（文本 + JSON），不自动 raise
- [x] **文件工具** - 目录复制/链接，跨平台兼容
- [x] **Node** - 解决方案 DAG 节点（代码、执行结果、评估、MCTS/GA 统计）
- [x] **Journal** - DAG 容器（节点管理、树查询、序列化）
- [x] **Task** - 任务定义（explore/merge/select/review）
- [x] **parse_solution_genes** - 基因组件解析器
- [x] **后端抽象层** - 统一 LLM 接口（OpenAI/Anthropic/GLM 4.7）

### Phase 2: 核心 Agent 系统
- [x] Interpreter 执行器 + WorkspaceManager
- [x] 工具增强（data_preview, metric, response）
- [x] BaseAgent 抽象类 + PromptBuilder
- [x] **CoderAgent 实现（5次LLM重试、响应解析、代码执行、92%测试覆盖）**
- [ ] Orchestrator 编排器

### Phase 3: 搜索与评估
- [ ] MCTS 搜索算法
- [ ] 遗传算法
- [ ] 并行评估框架

### Phase 4: 进化与学习
- [ ] Agent 能力进化
- [ ] Solution 基因进化
- [ ] 经验池与记忆系统
- [ ] Meta-Agent 自我优化

### Phase 5: 集成与评测
- [ ] 端到端测试
- [ ] MLE-Bench 适配器
- [ ] 性能基准测试

---

## 日志系统

### 双通道输出

- **文本日志** (`logs/system.log`): 人类可读的时间戳日志
- **JSON 日志** (`logs/metrics.json`): 结构化指标数据

### 使用示例

```python
from utils.logger_system import log_msg, log_json, ensure

# 文本日志
log_msg("INFO", "Agent 开始执行任务")

# 结构化日志
log_json({"agent_name": "Agent1", "step": 3, "score": 0.92})

# 断言工具
ensure(config.is_valid(), "配置无效")  # 失败时抛出 AssertionError
```

**重要变更（Phase 1）**: `log_msg("ERROR", ...)` 不再自动抛出异常，需要显式处理。

---

## 测试

### 运行测试

```bash
# 单元测试
pytest tests/unit/ -v

# 集成测试
pytest tests/integration/ -v

# 测试覆盖率
pytest tests/unit/ --cov=utils --cov-report=html
open htmlcov/index.html  # 查看覆盖率报告
```

### 测试组织

- `tests/unit/`: 单元测试（80%+ 覆盖率）
- `tests/integration/`: 集成测试（待添加）
- `tests/e2e/`: 端到端测试（Phase 5）

---

## MLE-Bench 评测

MLE-Bench 是由 OpenAI 构建的机器学习工程能力评估基准，涵盖 75 个真实 Kaggle 竞赛，要求 Agent 在标准化 Docker 容器环境中完成从数据理解、特征工程到模型训练与提交的全流程。Swarm-Ev2 通过 `run_mle_adapter.py` 适配器桥接 MLE-Bench 容器环境与双层进化主循环。

### 前置条件

- Docker Desktop（已启用）
- Kaggle 账号及 API 凭证
- 足够的磁盘空间（Lite 版数据集约 158GB）

### 配置步骤

1. **克隆 MLE-Bench 仓库**

```bash
cd ..
git clone https://github.com/openai/mle-bench.git
cd mle-bench
```

2. **修改容器配置**

编辑 `environment/config/container_configs/default.json`，替换为：

```json
{
    "gpus": 1,
    "mem_limit": null,
    "shm_size": "4G",
    "nano_cpus": 4e9,
    "runtime": "runc"
}
```

3. **构建 MLE-Bench 基础镜像**（仅需一次）

```bash
docker build --platform=linux/amd64 -t mlebench-env -f environment/Dockerfile .
```

4. **配置 Kaggle 凭证**

从 https://www.kaggle.com/account 下载 `kaggle.json`：

```bash
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

5. **下载 MLE-Bench 数据集**

```bash
conda create -n mlebench python=3.11 -y
conda activate mlebench
pip install -e .
mlebench prepare --lite
```

6. **构建 Swarm-Ev2 Agent 镜像**（每次更新代码后需重新构建）

```bash
# 将 Swarm-Ev2 代码同步到 mle-bench 的 agents 目录
rsync -av --progress \
  --exclude='workspace' --exclude='.git' --exclude='Reference' \
  ../Swarm-Ev2/ ./agents/swarm-evo/

# 构建 Agent 镜像
docker build --no-cache -t swarm-evo ./agents/swarm-evo
```

7. **运行评测**

```bash
API_KEY="your-api-key" \
API_BASE="https://api.openai.com/v1" \
MODEL_NAME="gpt-4-turbo" \
python run_agent.py \
  --agent-id swarm-evo \
  --competition-set experiments/splits/low.txt \
  --n-workers 4
```

### 关键文件说明

| 文件 | 说明 |
|------|------|
| `Dockerfile` | Agent 容器构建文件，基于 `mlebench-env` 基础镜像 |
| `start.sh` | 容器内入口脚本，激活 conda 环境并启动适配器 |
| `run_mle_adapter.py` | MLE-Bench 适配器，桥接环境变量、构建工作空间、运行进化主循环 |
| `config.yaml` | MLE-Bench Agent 注册配置（时间限制、环境变量等） |
| `config/mle_bench.yaml` | 容器内专用运行配置（路径、LLM、进化参数） |
| `requirements_agent.txt` | 容器内额外 Python 依赖 |
| `scripts/download_model.py` | 构建阶段预下载 BGE-M3 Embedding 模型 |

### 环境变量映射

适配器会自动将 MLE-Bench 环境变量映射为 Swarm-Ev2 格式：

| MLE-Bench 变量 | Swarm-Ev2 变量 | 说明 |
|----------------|---------------|------|
| `API_KEY` | `OPENAI_API_KEY` | LLM API 密钥 |
| `API_BASE` | `OPENAI_BASE_URL` | LLM API 地址 |
| `MODEL_NAME` | `LLM_MODEL` | 模型名称 |

### 注意事项

- 基础镜像 `mlebench-env` 仅需构建一次，Agent 镜像在代码更新后需重新构建
- 如果使用智谱 GLM 等兼容 OpenAI 格式的模型，修改 `API_BASE` 和 `MODEL_NAME` 即可
- `config/mle_bench.yaml` 中的路径已适配容器环境（`/home/` 前缀），无需修改

---

## 贡献指南

### 开发工作流

1. **阅读规范**: 仔细阅读 [CLAUDE.md](CLAUDE.md)
2. **创建分支**: `git checkout -b feature/your-feature`
3. **TDD 开发**:
   - 先写测试 (RED)
   - 写最小实现 (GREEN)
   - 重构优化 (REFACTOR)
4. **代码检查**:
   ```bash
   ruff format .
   ruff check . --fix
   pytest tests/ --cov=utils --cov-report=term-missing
   ```
5. **提交代码**: 遵循 [Conventional Commits](https://www.conventionalcommits.org/)
   ```bash
   git commit -m "feat: 添加 Agent 基类实现"
   ```
6. **创建 PR**: 提交 Pull Request 并等待审核

### Commit Message 格式

```
<type>: <description>

[optional body]
```

**Types**: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`

---

## 技术栈

| 类别 | 技术 |
|------|------|
| 语言 | Python 3.10 |
| 配置 | OmegaConf + YAML |
| 日志 | Rich + JSON |
| LLM | OpenAI API, Anthropic API, GLM API (智谱 AI) |
| 测试 | pytest + pytest-asyncio + pytest-cov |
| 代码质量 | Ruff (formatter + linter) |
| 类型检查 | Mypy |

---

## 许可证

[MIT License](LICENSE)

---

## 致谢

本项目受以下项目启发：
- [AIDE](https://github.com/WecoAI/aideml) - Agent 设计与后端抽象
- [Swarm-Evo](https://github.com/ML-Master/Swarm-Evo) - 群体智能算法

---

## 联系方式

- **Issue Tracker**: [GitHub Issues](../../issues)
- **Discussions**: [GitHub Discussions](../../discussions)

---

**最后更新**: 2026-01-31
