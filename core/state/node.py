"""
Node 数据类模块。

定义了解决方案搜索 DAG 中的节点结构，包含代码、执行结果、评估信息等。
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, Dict, Literal

from dataclasses_json import DataClassJsonMixin


@dataclass(eq=False)
class Node(DataClassJsonMixin):
    """解决方案 DAG 中的单个节点。

    包含代码、执行结果、评估信息以及 MCTS/GA 算法所需的统计数据。

    Attributes:
        code: Python 代码（必填）
        plan: 实现计划
        genes: 基因组件（由 parse_solution_genes 解析）
        step: Journal 中的序号
        id: 唯一标识符
        ctime: 创建时间戳
        parent_id: 父节点 ID
        children_ids: 子节点 ID 列表（用于 DAG 遍历）
        task_type: 任务类型（explore/merge）
        metadata: 额外元数据
        logs: 执行日志
        term_out: 终端输出
        exec_time: 执行时间（秒）
        exc_type: 异常类型
        exc_info: 异常详情
        analysis: LLM 分析结果
        metric_value: 评估指标值
        is_buggy: 是否包含 bug
        is_valid: 是否有效
        visits: MCTS 访问次数
        total_reward: MCTS 累计奖励
        generation: GA 进化代数
        fitness: GA 适应度值
    """

    # ---- 代码 ----
    code: str
    plan: str = field(default="", kw_only=True)
    genes: Dict[str, str] = field(default_factory=dict, kw_only=True)

    # ---- 通用属性 ----
    step: int = field(default=0, kw_only=True)
    id: str = field(default_factory=lambda: uuid.uuid4().hex, kw_only=True)
    ctime: float = field(default_factory=time.time, kw_only=True)
    parent_id: Optional[str] = field(default=None, kw_only=True)
    children_ids: list[str] = field(default_factory=list, kw_only=True)
    task_type: str = field(default="explore", kw_only=True)
    metadata: Dict = field(default_factory=dict, kw_only=True)

    # ---- 执行信息 ----
    logs: str = field(default="", kw_only=True)
    term_out: str = field(default="", kw_only=True)
    exec_time: float = field(default=0.0, kw_only=True)
    exc_type: Optional[str] = field(default=None, kw_only=True)
    exc_info: Optional[Dict] = field(default=None, kw_only=True)
    # Phase 2 可能启用的字段：
    # exc_stack: Optional[list[tuple]] = field(default=None, kw_only=True)

    # ---- 评估 ----
    analysis: str = field(default="", kw_only=True)
    metric_value: Optional[float] = field(default=None, kw_only=True)
    is_buggy: bool = field(default=False, kw_only=True)
    is_valid: bool = field(default=True, kw_only=True)

    # ---- MCTS ----
    visits: int = field(default=0, kw_only=True)
    total_reward: float = field(default=0.0, kw_only=True)

    # ---- GA ----
    generation: Optional[int] = field(default=None, kw_only=True)
    fitness: Optional[float] = field(default=None, kw_only=True)

    def __eq__(self, other) -> bool:
        """基于 ID 比较节点相等性。"""
        return isinstance(other, Node) and self.id == other.id

    def __hash__(self) -> int:
        """基于 ID 生成哈希值。"""
        return hash(self.id)

    @property
    def stage_name(self) -> Literal["initial", "bugfix", "improve", "unknown"]:
        """返回节点的生成模式。

        生成模式推导规则：
        - initial: 无父节点（初始方案）
        - bugfix: 父节点有 bug（修复模式）
        - improve: 父节点无 bug（改进模式）
        - unknown: 其他情况

        Returns:
            生成模式字符串

        注意: Phase 2 简化实现，需要 Journal 上下文才能准确判断父节点状态
        """
        if self.parent_id is None:
            return "initial"
        # Phase 2 完善：需要从 Journal 中查找父节点的 is_buggy 状态
        # 当前 MVP 阶段暂时返回 unknown
        return "unknown"

    @property
    def has_exception(self) -> bool:
        """检查节点执行是否产生异常。

        Returns:
            True 如果有异常，否则 False
        """
        return self.exc_type is not None
