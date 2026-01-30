# Phase 1: 基础设施重构详细实施计划

## 1.1 摘要 (Summary)

Phase 1 的目标是建立 Swarm-Ev2 架构的基础设施层，包括：统一配置系统（OmegaConf + YAML）、重构日志系统（去除自动 raise）、LLM 后端抽象层（复用 AIDE）、以及核心数据结构定义（Node, Journal, Task）。这一阶段为后续的核心功能和搜索算法实现奠定坚实基础。

**预计工作量**: 8-12 个文件，约 1000 行新增代码，200 行修改代码。

---

## 1.2 审查点 (User Review Required)

**已确认**:
- 使用 OmegaConf 作为配置管理工具
- 去除 `log_msg(ERROR)` 的自动 raise 行为
- 从 AIDE 复用后端抽象层设计
- 使用 `@dataclass` + `DataClassJsonMixin` 定义数据结构

**需确认**:
1. **配置文件位置**: 是否采用 `config/default.yaml` 作为主配置文件？
2. **环境变量管理**: 是否使用 `${env:OPENAI_API_KEY}` 的 OmegaConf 环境变量插值？
3. **日志文件路径**: 保持 `logs/system.log` 和 `logs/metrics.json` 的双通道输出？
4. **MetricValue 实现**: 是否需要实现 `@total_ordering` 用于自动比较（参考 AIDE）？

---

## 2. 拟议变更 (Proposed Changes)

### 2.1 配置系统模块

#### [NEW] `config/default.yaml`
**功能**: 项目主配置文件

**内容结构**:
```yaml
# 项目基础配置
project:
  name: "Swarm-Ev2"
  version: "0.1.0"
  workspace_dir: "./workspace"
  log_dir: "./logs"
  exp_name: null  # 自动生成

# 数据配置
data:
  data_dir: null  # 必填，CLI 提供
  desc_file: null
  goal: null
  eval: null
  preprocess_data: true
  copy_data: false  # 使用 symlink

# LLM 后端配置
llm:
  code:
    model: "gpt-4-turbo"
    temperature: 0.5
    api_key: ${env:OPENAI_API_KEY}
  feedback:
    model: "gpt-4-turbo"
    temperature: 0.5
    api_key: ${env:OPENAI_API_KEY}

# 执行配置
execution:
  timeout: 3600
  agent_file_name: "runfile.py"
  format_tb_ipython: false

# Agent 配置
agent:
  max_steps: 50
  time_limit: 86400  # 24 hours
  k_fold_validation: 5
  expose_prediction: false
  data_preview: true
  convert_system_to_user: false

# 搜索算法配置（Phase 3 使用）
search:
  strategy: "mcts"  # mcts | genetic
  max_debug_depth: 3
  debug_prob: 0.5
  num_drafts: 5
  parallel_num: 3

# 日志配置
logging:
  level: "INFO"
  console_output: true
  file_output: true
```

**依赖**: OmegaConf, Python 3.12+

---

#### [NEW] `utils/config.py`
**功能**: 配置加载、验证和管理

**类定义**:
```python
@dataclass
class ProjectConfig:
    """项目基础配置。"""
    name: str
    version: str
    workspace_dir: Path
    log_dir: Path
    exp_name: str | None

@dataclass
class DataConfig:
    """数据配置。"""
    data_dir: Path | None
    desc_file: Path | None
    goal: str | None
    eval: str | None
    preprocess_data: bool
    copy_data: bool

@dataclass
class LLMStageConfig:
    """LLM 阶段配置（code/feedback）。"""
    model: str
    temperature: float
    api_key: str

@dataclass
class LLMConfig:
    """LLM 配置。"""
    code: LLMStageConfig
    feedback: LLMStageConfig

@dataclass
class ExecutionConfig:
    """执行配置。"""
    timeout: int
    agent_file_name: str
    format_tb_ipython: bool

@dataclass
class AgentConfig:
    """Agent 配置。"""
    max_steps: int
    time_limit: int
    k_fold_validation: int
    expose_prediction: bool
    data_preview: bool
    convert_system_to_user: bool

@dataclass
class SearchConfig:
    """搜索算法配置。"""
    strategy: str
    max_debug_depth: int
    debug_prob: float
    num_drafts: int
    parallel_num: int

@dataclass
class LoggingConfig:
    """日志配置。"""
    level: str
    console_output: bool
    file_output: bool

@dataclass
class Config(Hashable):
    """顶层配置类。"""
    project: ProjectConfig
    data: DataConfig
    llm: LLMConfig
    execution: ExecutionConfig
    agent: AgentConfig
    search: SearchConfig
    logging: LoggingConfig

    def __hash__(self) -> int:
        return hash(self.project.exp_name)
```

**函数定义**:

1. **load_config(config_path: Path | None = None, use_cli: bool = True) -> Config**
   - 功能: 加载 YAML 配置并合并 CLI 参数
   - 参数:
     - `config_path`: 配置文件路径，默认 `config/default.yaml`
     - `use_cli`: 是否合并 CLI 参数
   - 返回: 验证后的 `Config` 对象
   - 实现:
     ```python
     def load_config(config_path: Path | None = None, use_cli: bool = True) -> Config:
         if config_path is None:
             config_path = Path(__file__).parent.parent / "config" / "default.yaml"

         cfg = OmegaConf.load(config_path)
         if use_cli:
             cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())

         return validate_config(cfg)
     ```

2. **validate_config(cfg: DictConfig) -> Config**
   - 功能: 验证配置完整性和合法性
   - 参数: OmegaConf DictConfig 对象
   - 返回: 类型化的 `Config` 对象
   - 实现:
     ```python
     def validate_config(cfg: DictConfig) -> Config:
         # 必填字段检查
         if cfg.data.data_dir is None:
             raise ValueError("`data.data_dir` 必须提供")

         if cfg.data.desc_file is None and cfg.data.goal is None:
             raise ValueError("必须提供 `desc_file` 或 `goal`")

         # 路径解析和创建
         cfg.data.data_dir = Path(cfg.data.data_dir).resolve()
         cfg.project.workspace_dir = Path(cfg.project.workspace_dir).resolve()
         cfg.project.log_dir = Path(cfg.project.log_dir).resolve()

         # 生成实验名称
         if cfg.project.exp_name is None:
             cfg.project.exp_name = generate_exp_name()

         # 创建目录
         cfg.project.workspace_dir.mkdir(parents=True, exist_ok=True)
         cfg.project.log_dir.mkdir(parents=True, exist_ok=True)

         # 类型化转换
         cfg_schema: Config = OmegaConf.structured(Config)
         cfg = OmegaConf.merge(cfg_schema, cfg)

         return cast(Config, cfg)
     ```

3. **generate_exp_name() -> str**
   - 功能: 生成实验名称（时间戳 + 随机后缀）
   - 返回: 实验名称字符串
   - 实现:
     ```python
     def generate_exp_name() -> str:
         from datetime import datetime
         import random
         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
         suffix = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=4))
         return f"{timestamp}_{suffix}"
     ```

4. **print_config(cfg: Config) -> None**
   - 功能: 美观打印配置（用于调试）
   - 参数: Config 对象
   - 实现:
     ```python
     def print_config(cfg: Config) -> None:
         from rich import print as rprint
         from rich.syntax import Syntax
         yaml_str = OmegaConf.to_yaml(cfg)
         syntax = Syntax(yaml_str, "yaml", theme="paraiso-dark")
         rprint(syntax)
     ```

5. **setup_workspace(cfg: Config) -> None**
   - 功能: 初始化工作空间目录结构
   - 参数: Config 对象
   - 实现:
     ```python
     def setup_workspace(cfg: Config) -> None:
         (cfg.project.workspace_dir / "input").mkdir(parents=True, exist_ok=True)
         (cfg.project.workspace_dir / "working").mkdir(parents=True, exist_ok=True)
         (cfg.project.workspace_dir / "submission").mkdir(parents=True, exist_ok=True)

         # 复制或链接数据
         from utils.file_utils import copytree
         copytree(
             cfg.data.data_dir,
             cfg.project.workspace_dir / "input",
             use_symlinks=not cfg.data.copy_data
         )
     ```

**依赖**: OmegaConf, dataclasses, pathlib, typing, rich

---

#### [NEW] `utils/file_utils.py`
**功能**: 文件操作工具（从 AIDE 复用）

**函数定义**:

1. **copytree(src: Path, dst: Path, use_symlinks: bool = True) -> None**
   - 功能: 复制或链接目录树
   - 参数:
     - `src`: 源目录
     - `dst`: 目标目录
     - `use_symlinks`: 是否使用符号链接
   - 实现: 参考 AIDE 的 `utils/__init__.py`

---

### 2.2 日志系统重构

#### [MODIFY] `utils/logger_system.py`
**修改目标**: 去除 `log_msg(ERROR)` 的自动 raise 行为

**修改内容**:

1. **text_log(level: str, message: str, **kwargs) -> None** [MODIFY]
   - 当前行为: `level == "ERROR"` 时自动 `raise Exception(message)`
   - 修改后: 只记录日志，不自动 raise
   - 修改代码:
     ```python
     # 旧版（删除）:
     if level == "ERROR":
         logger.error(message, **kwargs)
         raise Exception(message)

     # 新版（保留）:
     if level == "ERROR":
         logger.error(message, **kwargs)
         # 不再自动 raise
     ```

2. **log_msg(level: str, message: str, **kwargs) -> None** [MODIFY]
   - 修改: 调用 `text_log()`，不再负责 raise
   - 代码:
     ```python
     def log_msg(level: str, message: str, **kwargs) -> None:
         """记录文本日志到 system.log。"""
         text_log(level, message, **kwargs)
     ```

3. **ensure(condition: bool, error_msg: str) -> None** [NEW]
   - 功能: 断言工具，条件为 False 时记录错误并抛出异常
   - 参数:
     - `condition`: 断言条件
     - `error_msg`: 错误消息
   - 实现:
     ```python
     def ensure(condition: bool, error_msg: str) -> None:
         """断言工具，失败时记录错误并抛出异常。

         Args:
             condition: 断言条件
             error_msg: 错误消息

         Raises:
             AssertionError: 条件为 False 时抛出
         """
         if not condition:
             log_msg("ERROR", error_msg)
             raise AssertionError(error_msg)
     ```

4. **log_exception(exc: Exception, context: str = "") -> None** [NEW]
   - 功能: 记录异常信息（带堆栈跟踪）
   - 参数:
     - `exc`: 异常对象
     - `context`: 上下文描述
   - 实现:
     ```python
     def log_exception(exc: Exception, context: str = "") -> None:
         """记录异常信息和堆栈跟踪。

         Args:
             exc: 异常对象
             context: 上下文描述（可选）
         """
         import traceback
         error_msg = f"{context}: {exc}" if context else str(exc)
         traceback_str = ''.join(traceback.format_tb(exc.__traceback__))
         log_msg("ERROR", f"{error_msg}\n{traceback_str}")
     ```

**影响范围**: 所有调用 `log_msg("ERROR", ...)` 的代码需要显式处理异常。

**迁移指南**:
```python
# 旧版写法:
log_msg("ERROR", "配置无效")  # 自动 raise

# 新版写法:
log_msg("ERROR", "配置无效")
raise ValueError("配置无效")

# 或使用新的 ensure():
ensure(config.is_valid(), "配置无效")
```

---

### 2.3 后端抽象层

#### [NEW] `core/backend/__init__.py`
**功能**: 统一 LLM 查询接口（从 AIDE 复用）

**函数定义**:

1. **determine_provider(model: str) -> str**
   - 功能: 根据模型名称判断 LLM 提供商
   - 参数: `model` - 模型名称（如 "gpt-4-turbo"）
   - 返回: 提供商名称（"openai" | "anthropic" | "gdm" | "openrouter"）
   - 实现:
     ```python
     def determine_provider(model: str) -> str:
         """根据模型名称判断 LLM 提供商。"""
         if model.startswith("gpt-") or model.startswith("o1-"):
             return "openai"
         elif model.startswith("claude-"):
             return "anthropic"
         elif model.startswith("gemini-"):
             return "gdm"
         else:
             return "openrouter"
     ```

2. **query(system_message: str | None, user_message: str | None, model: str, temperature: float | None = None, max_tokens: int | None = None, **kwargs) -> str**
   - 功能: 统一 LLM 查询接口
   - 参数:
     - `system_message`: 系统消息
     - `user_message`: 用户消息
     - `model`: 模型名称
     - `temperature`: 采样温度
     - `max_tokens`: 最大 token 数
     - `**kwargs`: 额外参数
   - 返回: LLM 响应文本
   - 实现:
     ```python
     from utils.logger_system import log_msg

     def query(
         system_message: str | None,
         user_message: str | None,
         model: str,
         temperature: float | None = None,
         max_tokens: int | None = None,
         **kwargs
     ) -> str:
         """统一 LLM 查询接口。"""
         provider = determine_provider(model)
         query_func = PROVIDER_TO_QUERY[provider]

         log_msg("INFO", f"查询 LLM: model={model}, provider={provider}")

         response = query_func(
             system_message=system_message,
             user_message=user_message,
             model=model,
             temperature=temperature,
             max_tokens=max_tokens,
             **kwargs
         )

         log_msg("INFO", f"LLM 响应: {response[:100]}...")
         return response
     ```

3. **PROVIDER_TO_QUERY: dict[str, Callable]** [变量]
   - 功能: 提供商到查询函数的映射
   - 定义:
     ```python
     PROVIDER_TO_QUERY = {
         "openai": backend_openai.query,
         "anthropic": backend_anthropic.query,
         "gdm": backend_gdm.query,
         "openrouter": backend_openrouter.query,
     }
     ```

**依赖**: typing, utils.logger_system

---

#### [NEW] `core/backend/backend_openai.py`
**功能**: OpenAI 后端实现（从 AIDE 复用）

**函数定义**:

1. **query(system_message: str | None, user_message: str | None, model: str, **kwargs) -> str**
   - 功能: 调用 OpenAI API
   - 参数: 同父接口
   - 返回: API 响应
   - 实现: 直接复用 AIDE 的 `backend_openai.py` 实现

**依赖**: openai, os

---

#### [NEW] `core/backend/backend_anthropic.py`
**功能**: Anthropic 后端实现（从 AIDE 复用）

**函数定义**:
1. **query(...)** - 同上，复用 AIDE 实现

**依赖**: anthropic, os

---

### 2.4 数据结构定义

#### [NEW] `core/state/node.py`
**功能**: Node 数据类（参考 AIDE 设计）

**类定义**:
```python
from dataclasses import dataclass, field
from dataclasses_json import DataClassJsonMixin
from typing import Optional
import uuid
import time

@dataclass(eq=False)
class Node(DataClassJsonMixin):
    """解决方案树中的单个节点，包含代码、执行结果和评估信息。"""

    # ---- 代码与计划 ----
    code: str
    plan: str = field(default="", kw_only=True)

    # ---- 通用属性 ----
    step: int = field(default=0, kw_only=True)
    id: str = field(default_factory=lambda: uuid.uuid4().hex, kw_only=True)
    ctime: float = field(default_factory=time.time, kw_only=True)
    parent_id: Optional[str] = field(default=None, kw_only=True)

    # ---- 执行信息 ----
    term_out: str = field(default="", kw_only=True)
    exec_time: float = field(default=0.0, kw_only=True)
    exc_type: str | None = field(default=None, kw_only=True)
    exc_info: dict | None = field(default=None, kw_only=True)

    # ---- 评估结果 ----
    analysis: str = field(default="", kw_only=True)
    metric_value: float | None = field(default=None, kw_only=True)
    is_buggy: bool = field(default=False, kw_only=True)
    is_valid: bool = field(default=True, kw_only=True)

    # ---- MCTS 特定属性（Phase 3 使用）----
    visits: int = field(default=0, kw_only=True)
    total_reward: float = field(default=0.0, kw_only=True)

    # ---- GA 特定属性（Phase 3 使用）----
    genes: dict[str, str] = field(default_factory=dict, kw_only=True)
    generation: int | None = field(default=None, kw_only=True)
    fitness: float | None = field(default=None, kw_only=True)

    def __eq__(self, other) -> bool:
        """节点相等性比较（基于 ID）。"""
        return isinstance(other, Node) and self.id == other.id

    def __hash__(self) -> int:
        """节点哈希（基于 ID）。"""
        return hash(self.id)

    @property
    def stage_name(self) -> str:
        """返回节点阶段名称: draft | debug | improve。"""
        if self.parent_id is None:
            return "draft"
        # 需要访问 Journal 判断父节点是否 buggy
        # Phase 2 实现
        return "unknown"

    @property
    def has_exception(self) -> bool:
        """是否有异常。"""
        return self.exc_type is not None
```

**方法说明**:
- `__eq__` 和 `__hash__`: 使节点可以作为 dict key 和 set 成员
- `stage_name`: 阶段名称（依赖 Journal 上下文，Phase 2 完善）
- `has_exception`: 快速判断是否有异常

**依赖**: dataclasses, dataclasses_json, typing, uuid, time

---

#### [NEW] `core/state/journal.py`
**功能**: Journal 数据类（参考 AIDE 设计）

**类定义**:
```python
from dataclasses import dataclass, field
from dataclasses_json import DataClassJsonMixin
from typing import Optional
from core.state.node import Node

@dataclass
class Journal(DataClassJsonMixin):
    """解决方案树的容器，包含所有节点和树结构信息。"""

    nodes: list[Node] = field(default_factory=list)

    def __len__(self) -> int:
        """返回节点数量。"""
        return len(self.nodes)

    def __getitem__(self, idx: int) -> Node:
        """通过索引访问节点。"""
        return self.nodes[idx]

    def append(self, node: Node) -> None:
        """添加新节点。

        Args:
            node: 待添加的节点
        """
        node.step = len(self.nodes)
        self.nodes.append(node)

    def get_node_by_id(self, node_id: str) -> Optional[Node]:
        """通过 ID 获取节点。

        Args:
            node_id: 节点 ID

        Returns:
            找到的节点，不存在则返回 None
        """
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_children(self, node_id: str) -> list[Node]:
        """获取指定节点的所有子节点。

        Args:
            node_id: 父节点 ID

        Returns:
            子节点列表
        """
        return [n for n in self.nodes if n.parent_id == node_id]

    def get_siblings(self, node_id: str) -> list[Node]:
        """获取指定节点的所有兄弟节点。

        Args:
            node_id: 节点 ID

        Returns:
            兄弟节点列表（不包含自己）
        """
        node = self.get_node_by_id(node_id)
        if node is None or node.parent_id is None:
            return []

        siblings = [
            n for n in self.nodes
            if n.parent_id == node.parent_id and n.id != node_id
        ]
        return siblings

    @property
    def draft_nodes(self) -> list[Node]:
        """返回所有初始草稿节点（无父节点）。"""
        return [n for n in self.nodes if n.parent_id is None]

    @property
    def buggy_nodes(self) -> list[Node]:
        """返回所有被标记为 buggy 的节点。"""
        return [n for n in self.nodes if n.is_buggy]

    @property
    def good_nodes(self) -> list[Node]:
        """返回所有非 buggy 的节点。"""
        return [n for n in self.nodes if not n.is_buggy]

    def get_best_node(self, only_good: bool = True) -> Optional[Node]:
        """返回得分最高的节点。

        Args:
            only_good: 是否只考虑非 buggy 节点

        Returns:
            最佳节点，不存在则返回 None
        """
        candidates = self.good_nodes if only_good else self.nodes

        if not candidates:
            return None

        # 过滤掉没有 metric_value 的节点
        valid_candidates = [n for n in candidates if n.metric_value is not None]

        if not valid_candidates:
            return None

        return max(valid_candidates, key=lambda n: n.metric_value)
```

**方法说明**:
- `append()`: 自动设置 step 编号
- `get_node_by_id()`: O(n) 查找（MVP 阶段可接受）
- `get_children()`: 树结构遍历
- `get_siblings()`: Memory 机制使用
- `get_best_node()`: 返回最优解

**工具函数**（同一文件内定义）:
```python
def parse_solution_genes(code: str) -> dict[str, dict[str, str]]:
    """
    解析 solution.py 代码为基因组件。

    解析规则：
    1. 识别 `# [SECTION: NAME]` 标记的七个主基因块
    2. 识别 `# [FIXED: NAME]` 和 `# [EVOLVABLE: NAME]` 子区域

    Args:
        code: solution.py 的完整代码

    Returns:
        基因字典，结构如：
        {
            "DATA": {
                "FIXED": {"DATA_SPLIT": "...code..."},
                "EVOLVABLE": {"DATA_LOADING": "...", "DATA_AUGMENTATION": "..."}
            },
            "MODEL": {
                "EVOLVABLE": {"MODEL_DEFINITION": "..."}
            },
            ...
        }
    """
    # 实现细节见 Phase 1 实施阶段
```

**依赖**: dataclasses, dataclasses_json, typing, re, core.state.node

---

#### [NEW] `core/state/task.py`
**功能**: Task 数据类

**类定义**:
```python
from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin
from typing import Literal

TaskType = Literal["draft", "debug", "improve", "review"]

@dataclass
class Task(DataClassJsonMixin):
    """任务定义，描述 Agent 需要执行的操作。"""

    type: TaskType
    node_id: str
    description: str = ""

    def __str__(self) -> str:
        """任务字符串表示。"""
        return f"Task(type={self.type}, node_id={self.node_id[:8]}...)"
```

**说明**:
- `TaskType`: 任务类型枚举（扩展自 AIDE 的三阶段）
- 简洁设计，Phase 2 根据需要扩展

**依赖**: dataclasses, dataclasses_json, typing

---

### 2.5 依赖管理

#### [NEW] `requirements.txt`
**功能**: Python 依赖声明

**内容**:
```txt
# 核心依赖
python>=3.12
omegaconf>=2.3.0
dataclasses-json>=0.6.0
rich>=13.0.0

# LLM 客户端
openai>=1.0.0
anthropic>=0.20.0

# 工具库
pydantic>=2.0.0
pydantic-settings>=2.0.0

# 测试（Phase 5）
pytest>=8.0.0
pytest-asyncio>=0.23.0
pytest-cov>=4.0.0

# 代码质量
ruff>=0.3.0
mypy>=1.8.0
```

**版本策略**: 使用 `>=` 保持灵活性（MVP 原则）

---

## 3. 实施顺序 (Implementation Order)

### Step 1: 配置系统 [优先级: P0]
**顺序**: 最先实施，因为所有模块都依赖配置

1. 创建 `config/default.yaml`
2. 实现 `utils/config.py`:
   - 数据类定义
   - `load_config()`
   - `validate_config()`
   - `generate_exp_name()`
3. 实现 `utils/file_utils.py`:
   - `copytree()`
4. 编写配置加载测试

**验证**:
```bash
python -c "from utils.config import load_config; cfg = load_config(); print(cfg)"
```

---

### Step 2: 日志系统重构 [优先级: P0]
**顺序**: 第二，日志系统在开发过程中需要使用

1. 修改 `utils/logger_system.py`:
   - 去除 `text_log()` 的 auto-raise
   - 新增 `ensure()`
   - 新增 `log_exception()`
2. 更新现有调用（如果存在）
3. 编写日志系统测试

**验证**:
```bash
python -c "from utils.logger_system import log_msg, ensure; log_msg('ERROR', 'test'); print('未抛出异常')"
```

---

### Step 3: 数据结构定义 [优先级: P0]
**顺序**: 第三，为后端抽象层提供类型支持

1. 实现 `core/state/node.py`
2. 实现 `core/state/journal.py`
3. 实现 `core/state/task.py`
4. 编写序列化/反序列化测试

**验证**:
```python
from core.state.node import Node
from core.state.journal import Journal

node = Node(code="print('hello')", plan="测试")
journal = Journal()
journal.append(node)

# 序列化测试
json_str = journal.to_json()
journal2 = Journal.from_json(json_str)
assert len(journal2) == 1
```

---

### Step 4: 后端抽象层 [优先级: P0]
**顺序**: 最后，依赖配置和日志系统

1. 创建 `core/backend/__init__.py`:
   - `determine_provider()`
   - `query()`
2. 复用 AIDE 的 `backend_openai.py`
3. 复用 AIDE 的 `backend_anthropic.py`
4. 编写后端集成测试（需要真实 API Key）

**验证**:
```python
from core.backend import query

response = query(
    system_message="你是一个助手",
    user_message="说 Hello",
    model="gpt-4-turbo",
    temperature=0.7
)
print(response)
```

---

## 4. 验证计划 (Verification Plan)

### 4.1 单元测试

#### 测试文件: `tests/test_config.py`
**测试内容**:
1. `test_load_config_default()` - 加载默认配置
2. `test_load_config_with_cli()` - CLI 参数覆盖
3. `test_validate_config_missing_data_dir()` - 缺少必填字段
4. `test_generate_exp_name()` - 实验名称生成
5. `test_setup_workspace()` - 工作空间创建

**运行**:
```bash
pytest tests/test_config.py -v
```

---

#### 测试文件: `tests/test_logger.py`
**测试内容**:
1. `test_log_msg_no_auto_raise()` - ERROR 级别不自动 raise
2. `test_ensure_success()` - 条件为 True 时不抛出
3. `test_ensure_failure()` - 条件为 False 时抛出
4. `test_log_exception()` - 异常记录
5. `test_log_json()` - JSON 日志输出

**运行**:
```bash
pytest tests/test_logger.py -v
```

---

#### 测试文件: `tests/test_state.py`
**测试内容**:
1. `test_node_creation()` - 节点创建
2. `test_node_serialization()` - 序列化/反序列化
3. `test_journal_append()` - 添加节点
4. `test_journal_get_best_node()` - 获取最佳节点
5. `test_journal_get_siblings()` - 获取兄弟节点

**运行**:
```bash
pytest tests/test_state.py -v
```

---

#### 测试文件: `tests/test_backend.py`
**测试内容**:
1. `test_determine_provider()` - 提供商识别
2. `test_query_openai()` - OpenAI 查询（需要 API Key）
3. `test_query_anthropic()` - Anthropic 查询（需要 API Key）

**运行**:
```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
pytest tests/test_backend.py -v
```

---

### 4.2 集成测试

#### 测试文件: `tests/integration/test_phase1.py`
**测试内容**:
```python
def test_phase1_integration():
    """Phase 1 完整流程集成测试。"""
    # 1. 加载配置
    cfg = load_config(config_path=Path("config/default.yaml"))
    assert cfg.project.name == "Swarm-Ev2"

    # 2. 设置日志
    log_msg("INFO", "Phase 1 集成测试开始")

    # 3. 创建数据结构
    node = Node(code="print('test')", plan="测试节点")
    journal = Journal()
    journal.append(node)

    # 4. 序列化
    json_str = journal.to_json()
    assert len(json_str) > 0

    # 5. 后端查询（Mock）
    response = query(
        system_message="你是助手",
        user_message="说 Hello",
        model="gpt-4-turbo"
    )
    assert len(response) > 0

    log_msg("INFO", "Phase 1 集成测试完成")
```

**运行**:
```bash
pytest tests/integration/test_phase1.py -v
```

---

### 4.3 手动验证

#### 验证 1: 配置系统
```bash
# 1. 默认加载
python -c "from utils.config import load_config, print_config; cfg = load_config(); print_config(cfg)"

# 2. CLI 覆盖
python -c "from utils.config import load_config; cfg = load_config()" \
  --project.name="Test" \
  --llm.code.model="gpt-3.5-turbo"

# 3. 检查生成的实验名称
python -c "from utils.config import generate_exp_name; print(generate_exp_name())"
```

**预期输出**:
- 美观的 YAML 格式配置
- CLI 参数正确覆盖
- 实验名称格式: `YYYYMMDD_HHMMSS_xxxx`

---

#### 验证 2: 日志系统
```bash
# 1. 测试 ERROR 不 raise
python -c "from utils.logger_system import log_msg; log_msg('ERROR', '测试错误'); print('程序继续运行')"

# 2. 测试 ensure()
python -c "from utils.logger_system import ensure; ensure(1 + 1 == 2, '数学错误')"

# 3. 检查日志文件
cat logs/system.log
cat logs/metrics.json
```

**预期输出**:
- ERROR 日志不导致程序终止
- `ensure()` 条件满足时不抛出
- 日志文件正确生成

---

#### 验证 3: 数据结构
```python
# test_manual_state.py
from core.state.node import Node
from core.state.journal import Journal

# 创建节点
node1 = Node(code="x = 1", plan="初始化")
node2 = Node(code="x = 2", plan="修改", parent_id=node1.id)

# 创建 Journal
journal = Journal()
journal.append(node1)
journal.append(node2)

# 测试树结构
assert len(journal) == 2
assert len(journal.get_children(node1.id)) == 1

# 序列化
json_str = journal.to_json()
journal2 = Journal.from_json(json_str)
assert len(journal2) == 2

print("✓ 数据结构验证通过")
```

**运行**:
```bash
python test_manual_state.py
```

---

#### 验证 4: 后端抽象层
```python
# test_manual_backend.py
from core.backend import query, determine_provider

# 测试提供商识别
assert determine_provider("gpt-4-turbo") == "openai"
assert determine_provider("claude-3-opus") == "anthropic"

# 测试 OpenAI 查询
response = query(
    system_message="你是一个友好的助手",
    user_message="用一句话介绍 Python",
    model="gpt-4-turbo",
    temperature=0.7
)

print(f"LLM 响应: {response}")
print("✓ 后端抽象层验证通过")
```

**运行**:
```bash
export OPENAI_API_KEY=sk-...
python test_manual_backend.py
```

---

## 5. 依赖与风险 (Dependencies & Risks)

### 5.1 依赖关系

```
配置系统 (Step 1)
    ↓
日志系统 (Step 2)
    ↓
数据结构 (Step 3)
    ↓
后端抽象层 (Step 4)
```

**关键路径**: 配置系统 → 后续所有模块

---

### 5.2 风险分析

#### Risk 1: OmegaConf 学习曲线
**影响**: 中
**概率**: 低
**缓解**:
- 提供详细配置示例（`config/default.yaml`）
- 参考 AIDE 的实现（已验证）
- 默认配置覆盖 90% 场景

---

#### Risk 2: 日志系统重构影响现有代码
**影响**: 中
**概率**: 低（项目初始阶段）
**缓解**:
- 当前项目无遗留代码
- 提供迁移指南
- 新增 `ensure()` 工具简化异常处理

---

#### Risk 3: 后端 API Key 配置错误
**影响**: 高（阻塞开发）
**概率**: 中
**缓解**:
- 配置验证时检查环境变量
- 提供清晰的错误提示
- 文档说明 API Key 配置步骤

---

#### Risk 4: 数据结构设计不足
**影响**: 高（后续重构成本高）
**概率**: 低
**缓解**:
- 参考 AIDE 和 ML-Master 的成熟设计
- Phase 1 只实现核心字段
- 预留扩展点（kw_only 参数）

---

#### Risk 5: 文件操作（symlink）跨平台兼容性
**影响**: 中
**概率**: 中（Windows 平台）
**缓解**:
- 提供 `copy_data` 配置选项
- 自动检测平台并回退到复制模式
- 文档说明平台差异

---

## 6. 成功标准 (Success Criteria)

Phase 1 完成的标准:

### 6.1 功能完整性
- [ ] 配置文件可正确加载（YAML + CLI 覆盖）
- [ ] 日志系统正常工作（双通道输出，不自动 raise）
- [ ] Backend 可成功调用 OpenAI 和 Anthropic
- [ ] Node/Journal/Task 可序列化/反序列化

### 6.2 测试覆盖率
- [ ] 单元测试覆盖率 ≥ 80%
- [ ] 所有集成测试通过
- [ ] 手动验证步骤全部通过

### 6.3 代码质量
- [ ] Ruff 格式化通过
- [ ] Mypy 类型检查通过（无错误）
- [ ] 所有函数包含中文 Docstring

### 6.4 文档完整性
- [ ] 配置文件包含详细注释
- [ ] README.md 更新（Phase 1 完成状态）
- [ ] API 文档生成（Sphinx/MkDocs）

---

## 7. 下一步行动 (Next Steps)

### 立即行动
1. **用户审核本计划** ← **当前步骤**
2. 确认审查点问题（1.2 节）
3. 批准后开始实施

### Phase 1 启动检查清单
- [ ] 创建 `config/` 目录
- [ ] 创建 `core/state/` 目录
- [ ] 创建 `core/backend/` 目录
- [ ] 创建 `tests/` 目录
- [ ] 安装依赖: `pip install -r requirements.txt`
- [ ] 配置环境变量: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`

### Phase 1 完成后
1. 运行完整验证流程（第 4 节）
2. 提交 Git commit（符合 Conventional Commits 格式）
3. 更新 `docs/implementation_plan.md` 状态
4. 开始 Phase 2 规划

---

## 8. 附录 (Appendix)

### 8.1 文件清单

**新增文件** (10 个):
```
config/default.yaml                        [NEW]
utils/config.py                            [NEW]
utils/file_utils.py                        [NEW]
core/state/node.py                         [NEW]
core/state/journal.py                      [NEW]
core/state/task.py                         [NEW]
core/backend/__init__.py                   [NEW]
core/backend/backend_openai.py             [NEW]
core/backend/backend_anthropic.py          [NEW]
requirements.txt                           [NEW]
```

**修改文件** (1 个):
```
utils/logger_system.py                     [MODIFY]
```

**测试文件** (5 个):
```
tests/test_config.py                       [NEW]
tests/test_logger.py                       [NEW]
tests/test_state.py                        [NEW]
tests/test_backend.py                      [NEW]
tests/integration/test_phase1.py           [NEW]
```

---

### 8.2 代码行数估算

| 模块 | 新增代码 | 修改代码 | 测试代码 | 总计 |
|------|---------|---------|---------|------|
| 配置系统 | ~350 | 0 | ~150 | 500 |
| 日志系统 | ~80 | ~50 | ~100 | 230 |
| 数据结构 | ~300 | 0 | ~150 | 450 |
| 后端抽象层 | ~200 | 0 | ~100 | 300 |
| 工具函数 | ~70 | 0 | 0 | 70 |
| **总计** | **~1000** | **~50** | **~500** | **1550** |

---

### 8.3 参考资源

**AIDE 项目**:
- `/Users/yuchengzhang/Desktop/Code/Swarm-Ev2/Reference/aideml-main/aide/utils/config.py`
- `/Users/yuchengzhang/Desktop/Code/Swarm-Ev2/Reference/aideml-main/aide/backend/__init__.py`
- `/Users/yuchengzhang/Desktop/Code/Swarm-Ev2/Reference/aideml-main/aide/journal.py`
- `/Users/yuchengzhang/Desktop/Code/Swarm-Ev2/Reference/aideml-main/aide/utils/config.yaml`

**ML-Master 项目**:
- `/Users/yuchengzhang/Desktop/Code/Swarm-Ev2/Reference/ML-Master-main/utils/config_mcts.py`
- `/Users/yuchengzhang/Desktop/Code/Swarm-Ev2/Reference/ML-Master-main/search/node.py`

**现有文档**:
- `/Users/yuchengzhang/Desktop/Code/Swarm-Ev2/draft.md` (架构设计草稿)
- `/Users/yuchengzhang/Desktop/Code/Swarm-Ev2/docs/implementation_plan.md` (总体计划)

---

**计划版本**: Phase1-v1.0
**创建日期**: 2026-01-29
**状态**: 待审查
**预计工期**: 3-5 天（单人开发）
