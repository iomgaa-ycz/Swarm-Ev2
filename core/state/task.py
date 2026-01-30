"""
Task 数据类模块。

定义任务类型和任务数据结构，用于 Agent 系统的任务调度。
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Literal, Optional, Dict

from dataclasses_json import DataClassJsonMixin


# Task 类型定义（Swarm-Evo 标准）
TaskType = Literal["explore", "merge", "select", "review"]


@dataclass
class Task(DataClassJsonMixin):
    """Agent 任务定义。

    用于任务队列和调度系统，支持依赖管理和上下文传递。

    Attributes:
        id: 任务唯一标识符
        type: 任务类型（explore/merge/select/review）
        node_id: 关联的节点 ID
        description: 任务描述
        created_at: 创建时间戳
        agent_name: 分配的 Agent 名称
        dependencies: 任务依赖关系（格式：{依赖名称: 任务ID}）
        payload: 任务上下文数据

    Task 类型说明:
        - explore: 探索新方案，生成新的解决方案节点
        - merge: 合并多个方案，融合不同节点的优点
        - select: 选择最佳方案，从候选节点中筛选
        - review: 审查方案质量，评估节点的有效性
    """

    # ---- 核心字段 ----
    type: TaskType
    node_id: str

    # ---- 元数据 ----
    description: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)

    # ---- 调度信息 ----
    agent_name: Optional[str] = None
    dependencies: Optional[Dict[str, str]] = None  # {依赖名称: 任务ID}
    payload: Dict = field(default_factory=dict)  # 任务上下文

    def __str__(self) -> str:
        """返回任务的字符串表示。

        Returns:
            格式化的任务信息字符串
        """
        return f"Task(type={self.type}, node_id={self.node_id[:8]}...)"
