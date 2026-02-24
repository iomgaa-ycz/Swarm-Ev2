"""Agent 抽象基类模块。

定义统一的 Agent 接口和数据结构。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Literal

from dataclasses_json import DataClassJsonMixin

from core.state import Node, Journal
from utils.config import Config

if TYPE_CHECKING:
    from utils.prompt_manager import PromptManager


@dataclass
class AgentContext(DataClassJsonMixin):
    """Agent 执行上下文。

    封装 Agent 执行所需的所有上下文信息。

    Attributes:
        task_type: 任务类型（"draft"、"explore"、"merge" 或 "mutate"）
        parent_node: 父节点（None=初稿, buggy=修复, normal=改进）
        journal: 历史节点记录（用于 Memory 机制）
        config: 全局配置
        start_time: 任务开始时间（用于计算剩余时间）
        current_step: 当前步数（用于计算剩余步数）
        task_desc: 任务描述字符串
        device_info: 硬件描述字符串（CPU/RAM/GPU）
        conda_packages: Conda 环境包信息描述
        conda_env_name: Conda 环境名称
        parent_a: merge 任务专用 - 父代 A
        parent_b: merge 任务专用 - 父代 B
        gene_plan: merge 任务专用 - 基因交叉计划
        primary_parent: merge 任务专用 - 贡献基因最多的父代（取代 parent_a 作为语义主父代）
        gene_sources: merge 任务专用 - {locus: source_node_id} 字典，记录每个位点的来源
        target_gene: mutate 任务专用 - 目标基因块名称
        mutation_aspect: mutate 任务专用 - 目标基因的子方面（如 "optimizer", "architecture"）
        draft_history: draft 任务专用 - 已用方法标签列表，用于多样性引导
    """

    task_type: Literal["draft", "explore", "merge", "mutate"]
    parent_node: Optional[Node]
    journal: Journal
    config: Config
    start_time: float
    current_step: int
    task_desc: str
    device_info: str = ""
    conda_packages: str = ""
    conda_env_name: str = ""
    # merge 任务专用字段
    parent_a: Optional[Node] = None       # 保留：兼容旧 execute_merge_task 过渡期
    parent_b: Optional[Node] = None       # 保留：兼容旧 execute_merge_task 过渡期
    primary_parent: Optional[Node] = None  # 新增：贡献基因最多的父代
    gene_plan: Optional[dict] = None
    gene_sources: Optional[Dict[str, str]] = None  # 新增：{locus: source_node_id}
    # mutate 任务专用字段
    target_gene: Optional[str] = None
    mutation_aspect: Optional[str] = None
    # draft 任务专用字段
    draft_history: Optional[List[str]] = None  # 新增：已用方法标签列表
    # 经验池（用于动态 Skill 注入）
    experience_pool: Optional[Any] = field(
        default=None, metadata={"dataclasses_json": {"exclude": lambda _: True}}
    )


@dataclass
class AgentResult(DataClassJsonMixin):
    """Agent 执行结果。

    封装 Agent 执行后的结果。

    Attributes:
        node: 生成的节点（失败时为 None）
        success: 执行是否成功
        error: 错误信息（成功时为 None）
    """

    node: Optional[Node]
    success: bool
    error: Optional[str] = None


class BaseAgent(ABC):
    """Agent 抽象基类。

    定义统一的 Agent 接口，所有具体 Agent 必须继承此类。

    Attributes:
        name: Agent 名称
        config: 全局配置
        prompt_manager: Prompt 管理器
    """

    def __init__(self, name: str, config: Config, prompt_manager: "PromptManager"):
        """初始化 BaseAgent。

        Args:
            name: Agent 名称
            config: 全局配置
            prompt_manager: PromptManager 实例
        """
        self.name = name
        self.config = config
        self.prompt_manager = prompt_manager

    @abstractmethod
    def generate(self, context: AgentContext) -> AgentResult:
        """主入口：根据 task_type 分发到具体实现。

        Args:
            context: Agent 执行上下文

        Returns:
            AgentResult 对象，包含生成的节点或错误信息

        注意:
            - 具体实现需要根据 context.task_type 分发到 _explore 或 _merge
            - Phase 2 只实现 explore，merge 留给 Phase 3
        """
        pass

    @abstractmethod
    def _explore(self, context: AgentContext) -> Node:
        """探索新方案（统一方法）。

        根据 parent_node 自动适配：
        - None: 生成初稿
        - is_buggy: 修复 bug
        - 正常: 改进方案

        Args:
            context: Agent 执行上下文

        Returns:
            生成的 Node 对象

        注意:
            - 不显式区分初稿/改进/修复，让 LLM 根据上下文判断
            - Prompt 会根据 parent_node 动态插入上下文
        """
        pass
