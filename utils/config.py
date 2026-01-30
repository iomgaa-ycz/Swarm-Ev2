"""配置管理模块。

提供基于 OmegaConf + YAML 的统一配置加载、验证和管理功能。
"""

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Hashable
from omegaconf import OmegaConf, DictConfig
from datetime import datetime
import random
import os
from dotenv import load_dotenv


# 注册环境变量解析器（支持 ${env:VAR} 语法）
OmegaConf.register_new_resolver("env", lambda var: os.getenv(var, ""))


# ============================================================
# 配置数据类定义
# ============================================================


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
    """顶层配置类。

    实现 Hashable 接口，可用作 dict key 和 set 成员。
    """

    project: ProjectConfig
    data: DataConfig
    llm: LLMConfig
    execution: ExecutionConfig
    agent: AgentConfig
    search: SearchConfig
    logging: LoggingConfig

    def __hash__(self) -> int:
        """基于实验名称的哈希值。

        Returns:
            实验名称的哈希值
        """
        return hash(self.project.exp_name)


# ============================================================
# 配置加载与验证函数
# ============================================================


def load_config(
    config_path: Path | None = None, use_cli: bool = True, env_file: Path | None = None
) -> Config:
    """加载 YAML 配置并合并 CLI 参数和环境变量。

    配置优先级（从高到低）:
        1. CLI 参数（--key=value）
        2. 环境变量（.env 文件或系统环境变量）
        3. YAML 配置文件

    Args:
        config_path: 配置文件路径，默认为 config/default.yaml
        use_cli: 是否合并 CLI 参数（通过 OmegaConf.from_cli()）
        env_file: .env 文件路径，默认为项目根目录的 .env 文件

    Returns:
        验证后的 Config 对象

    Raises:
        FileNotFoundError: 配置文件不存在
        ValueError: 配置验证失败

    示例:
        >>> cfg = load_config()  # 加载默认配置
        >>> cfg = load_config(Path("custom.yaml"))  # 加载自定义配置
        >>> cfg = load_config(use_cli=False)  # 不合并 CLI 参数
        >>> cfg = load_config(env_file=Path(".env.prod"))  # 使用生产环境变量
    """
    from utils.logger_system import log_msg

    # 步骤 1: 加载 .env 文件到环境变量
    if env_file is None:
        env_file = Path(__file__).parent.parent / ".env"

    if env_file.exists():
        load_dotenv(env_file, override=False)  # override=False: 不覆盖已存在的环境变量
        log_msg("INFO", f"加载环境变量文件: {env_file}")
    else:
        log_msg("INFO", "未找到 .env 文件，使用系统环境变量")

    # 步骤 2: 确定配置文件路径
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "default.yaml"

    if not config_path.exists():
        error_msg = f"配置文件不存在: {config_path}"
        log_msg("ERROR", error_msg)
        raise FileNotFoundError(error_msg)

    log_msg("INFO", f"加载配置文件: {config_path}")

    # 步骤 3: 加载 YAML 配置（会自动解析 ${env:VAR} 插值）
    cfg = OmegaConf.load(config_path)

    # 步骤 4: 合并 CLI 参数（优先级最高）
    if use_cli:
        cli_cfg = OmegaConf.from_cli()
        if cli_cfg:
            log_msg("INFO", f"合并 CLI 参数: {OmegaConf.to_yaml(cli_cfg)}")
            cfg = OmegaConf.merge(cfg, cli_cfg)

    # 步骤 5: 验证配置
    validated_cfg = validate_config(cfg)
    log_msg("INFO", "配置加载并验证成功")

    return validated_cfg


def validate_config(cfg: DictConfig) -> Config:
    """验证配置完整性和合法性。

    Args:
        cfg: OmegaConf DictConfig 对象

    Returns:
        类型化的 Config 对象

    Raises:
        ValueError: 配置验证失败（缺少必填字段、路径不存在等）

    验证规则:
        1. 必填字段检查: data.data_dir 必须提供
        2. 二选一检查: data.desc_file 或 data.goal 至少提供一个
        3. 路径解析: 将相对路径转为绝对路径
        4. 目录创建: 创建必要的工作目录
        5. 实验名称生成: 如果未提供则自动生成
        6. API Key 检查: 确保环境变量已设置
    """
    from utils.logger_system import log_msg

    # ---- 必填字段检查 ----
    if cfg.data.data_dir is None:
        error_msg = "`data.data_dir` 必须提供（通过配置文件或 CLI 参数）"
        log_msg("ERROR", error_msg)
        raise ValueError(error_msg)

    if cfg.data.desc_file is None and cfg.data.goal is None:
        error_msg = "必须提供 `data.desc_file` 或 `data.goal` 之一"
        log_msg("ERROR", error_msg)
        raise ValueError(error_msg)

    # ---- 路径解析和创建 ----
    cfg.data.data_dir = Path(cfg.data.data_dir).resolve()

    # 验证数据目录存在
    if not cfg.data.data_dir.exists():
        error_msg = f"数据目录不存在: {cfg.data.data_dir}"
        log_msg("ERROR", error_msg)
        raise ValueError(error_msg)

    # 解析其他路径
    if cfg.data.desc_file is not None:
        cfg.data.desc_file = Path(cfg.data.desc_file).resolve()
        if not cfg.data.desc_file.exists():
            error_msg = f"描述文件不存在: {cfg.data.desc_file}"
            log_msg("ERROR", error_msg)
            raise ValueError(error_msg)

    cfg.project.workspace_dir = Path(cfg.project.workspace_dir).resolve()
    cfg.project.log_dir = Path(cfg.project.log_dir).resolve()

    # 创建必要目录
    cfg.project.workspace_dir.mkdir(parents=True, exist_ok=True)
    cfg.project.log_dir.mkdir(parents=True, exist_ok=True)
    log_msg("INFO", f"工作目录: {cfg.project.workspace_dir}")
    log_msg("INFO", f"日志目录: {cfg.project.log_dir}")

    # ---- 生成实验名称 ----
    if cfg.project.exp_name is None:
        cfg.project.exp_name = generate_exp_name()
        log_msg("INFO", f"生成实验名称: {cfg.project.exp_name}")

    # ---- API Key 检查 ----
    if not cfg.llm.code.api_key or cfg.llm.code.api_key.startswith("${env:"):
        log_msg("WARNING", "LLM API Key 未设置或环境变量未解析，请检查环境变量配置")

    # ---- 类型化转换 ----
    # 创建结构化配置模板
    cfg_schema = OmegaConf.structured(
        Config(
            project=ProjectConfig(
                name="", version="", workspace_dir=Path(), log_dir=Path(), exp_name=None
            ),
            data=DataConfig(
                data_dir=None,
                desc_file=None,
                goal=None,
                eval=None,
                preprocess_data=True,
                copy_data=False,
            ),
            llm=LLMConfig(
                code=LLMStageConfig(model="", temperature=0.0, api_key=""),
                feedback=LLMStageConfig(model="", temperature=0.0, api_key=""),
            ),
            execution=ExecutionConfig(
                timeout=0, agent_file_name="", format_tb_ipython=False
            ),
            agent=AgentConfig(
                max_steps=0,
                time_limit=0,
                k_fold_validation=0,
                expose_prediction=False,
                data_preview=False,
                convert_system_to_user=False,
            ),
            search=SearchConfig(
                strategy="",
                max_debug_depth=0,
                debug_prob=0.0,
                num_drafts=0,
                parallel_num=0,
            ),
            logging=LoggingConfig(level="", console_output=False, file_output=False),
        )
    )

    # 合并配置
    cfg_merged = OmegaConf.merge(cfg_schema, cfg)

    # 转换为 Python 对象
    cfg_dict = OmegaConf.to_container(cfg_merged, resolve=True)

    # 手动构造 Config 对象（确保 Path 类型正确）
    return Config(
        project=ProjectConfig(**cfg_dict["project"]),
        data=DataConfig(**cfg_dict["data"]),
        llm=LLMConfig(
            code=LLMStageConfig(**cfg_dict["llm"]["code"]),
            feedback=LLMStageConfig(**cfg_dict["llm"]["feedback"]),
        ),
        execution=ExecutionConfig(**cfg_dict["execution"]),
        agent=AgentConfig(**cfg_dict["agent"]),
        search=SearchConfig(**cfg_dict["search"]),
        logging=LoggingConfig(**cfg_dict["logging"]),
    )


def generate_exp_name() -> str:
    """生成实验名称（时间戳 + 随机后缀）。

    Returns:
        实验名称字符串，格式: YYYYMMDD_HHMMSS_xxxx

    示例:
        >>> name = generate_exp_name()
        >>> print(name)
        20260130_143022_abcd
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=4))
    return f"{timestamp}_{suffix}"


def print_config(cfg: Config) -> None:
    """美观打印配置（用于调试）。

    Args:
        cfg: Config 对象

    实现细节:
        - 使用 rich 库高亮显示 YAML 格式
        - 使用 paraiso-dark 主题
    """
    from rich import print as rprint
    from rich.syntax import Syntax

    # 转换为字典以便序列化
    cfg_dict = {
        "project": {
            "name": cfg.project.name,
            "version": cfg.project.version,
            "workspace_dir": str(cfg.project.workspace_dir),
            "log_dir": str(cfg.project.log_dir),
            "exp_name": cfg.project.exp_name,
        },
        "data": {
            "data_dir": str(cfg.data.data_dir) if cfg.data.data_dir else None,
            "desc_file": str(cfg.data.desc_file) if cfg.data.desc_file else None,
            "goal": cfg.data.goal,
            "eval": cfg.data.eval,
            "preprocess_data": cfg.data.preprocess_data,
            "copy_data": cfg.data.copy_data,
        },
        "llm": {
            "code": {
                "model": cfg.llm.code.model,
                "temperature": cfg.llm.code.temperature,
                "api_key": "***" if cfg.llm.code.api_key else None,  # 隐藏 API Key
            },
            "feedback": {
                "model": cfg.llm.feedback.model,
                "temperature": cfg.llm.feedback.temperature,
                "api_key": "***" if cfg.llm.feedback.api_key else None,
            },
        },
        "execution": {
            "timeout": cfg.execution.timeout,
            "agent_file_name": cfg.execution.agent_file_name,
            "format_tb_ipython": cfg.execution.format_tb_ipython,
        },
        "agent": {
            "max_steps": cfg.agent.max_steps,
            "time_limit": cfg.agent.time_limit,
            "k_fold_validation": cfg.agent.k_fold_validation,
            "expose_prediction": cfg.agent.expose_prediction,
            "data_preview": cfg.agent.data_preview,
            "convert_system_to_user": cfg.agent.convert_system_to_user,
        },
        "search": {
            "strategy": cfg.search.strategy,
            "max_debug_depth": cfg.search.max_debug_depth,
            "debug_prob": cfg.search.debug_prob,
            "num_drafts": cfg.search.num_drafts,
            "parallel_num": cfg.search.parallel_num,
        },
        "logging": {
            "level": cfg.logging.level,
            "console_output": cfg.logging.console_output,
            "file_output": cfg.logging.file_output,
        },
    }

    yaml_str = OmegaConf.create(cfg_dict)
    yaml_str = OmegaConf.to_yaml(yaml_str)
    syntax = Syntax(yaml_str, "yaml", theme="paraiso-dark", line_numbers=True)
    rprint(syntax)


def setup_workspace(cfg: Config) -> None:
    """初始化工作空间目录结构。

    Args:
        cfg: Config 对象

    实现细节:
        1. 创建工作空间子目录:
           - input: 输入数据（链接或复制自 data_dir）
           - working: Agent 工作目录
           - submission: 提交文件目录
        2. 复制或链接数据（根据 cfg.data.copy_data 决定）
    """
    from utils.logger_system import log_msg
    from utils.file_utils import copytree

    # 创建工作空间子目录
    input_dir = cfg.project.workspace_dir / "input"
    working_dir = cfg.project.workspace_dir / "working"
    submission_dir = cfg.project.workspace_dir / "submission"

    input_dir.mkdir(parents=True, exist_ok=True)
    working_dir.mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(parents=True, exist_ok=True)

    log_msg("INFO", "创建工作空间子目录: input, working, submission")

    # 复制或链接数据到 input 目录
    if cfg.data.data_dir is not None:
        copytree(
            src=cfg.data.data_dir, dst=input_dir, use_symlinks=not cfg.data.copy_data
        )
        log_msg("INFO", f"数据已准备: {input_dir}")
