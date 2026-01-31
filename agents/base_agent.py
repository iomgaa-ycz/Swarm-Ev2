"""Agent 抽象基类模块。

定义统一的 Agent 接口和数据结构。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Literal

from dataclasses_json import DataClassJsonMixin

from core.state import Node, Journal
from utils.config import Config

if TYPE_CHECKING:
    from utils.prompt_builder import PromptBuilder


@dataclass
class AgentContext(DataClassJsonMixin):
    """Agent 执行上下文。

    封装 Agent 执行所需的所有上下文信息。

    Attributes:
        task_type: 任务类型（"explore" 或 "merge"）
        parent_node: 父节点（None=初稿, buggy=修复, normal=改进）
        journal: 历史节点记录（用于 Memory 机制）
        config: 全局配置
        start_time: 任务开始时间（用于计算剩余时间）
        current_step: 当前步数（用于计算剩余步数）
    """

    task_type: Literal["explore", "merge"]
    parent_node: Optional[Node]
    journal: Journal
    config: Config
    start_time: float
    current_step: int


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
        prompt_builder: Prompt 构建器
    """

    def __init__(self, name: str, config: Config, prompt_builder: "PromptBuilder"):
        """初始化 BaseAgent。

        Args:
            name: Agent 名称
            config: 全局配置
            prompt_builder: Prompt 构建器实例
        """
        self.name = name
        self.config = config
        self.prompt_builder = prompt_builder

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
