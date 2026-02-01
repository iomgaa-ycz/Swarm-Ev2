"""共享经验池模块。

提供线程安全的 Agent 执行记录存储、查询和 JSON 持久化功能。
"""

import json
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional

from utils.config import Config
from utils.logger_system import log_msg, log_exception


@dataclass
class TaskRecord:
    """Agent 执行记录。

    Attributes:
        agent_id: Agent 唯一标识
        task_type: 任务类型（"explore" | "merge" | "mutate"）
        input_hash: 输入的哈希值（用于去重）
        output_quality: 适应度值（归一化后的质量评分）
        strategy_summary: Agent 的策略摘要（从 node.plan 提取）
        timestamp: 记录创建时间戳
    """

    agent_id: str
    task_type: str
    input_hash: str
    output_quality: float
    strategy_summary: str
    timestamp: float


class ExperiencePool:
    """共享经验池（线程安全 + JSON 持久化）。

    存储所有 Agent 的执行记录，支持按任务类型查询 Top-K 记录。

    Attributes:
        config: 全局配置
        records: 记录列表
        lock: 线程锁（保护并发写入）
        save_path: JSON 文件保存路径
    """

    def __init__(self, config: Config):
        """初始化经验池。

        Args:
            config: 全局配置对象

        注意:
            - 初始化时自动从 JSON 文件加载历史记录（如果存在）
            - 如果文件损坏或不存在，从空白开始
        """
        self.config = config
        self.records: List[TaskRecord] = []
        self.lock = threading.Lock()

        # 获取保存路径
        save_path_str = config.evolution.experience_pool.save_path
        self.save_path = Path(save_path_str)

        # 确保目录存在
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        # 加载历史记录
        self.load()

    def add(self, record: TaskRecord) -> None:
        """添加新记录到经验池（线程安全）。

        Args:
            record: 要添加的 TaskRecord 对象

        注意:
            - 超过 max_records 时删除最旧记录（FIFO）
            - 添加后自动持久化到 JSON 文件
        """
        with self.lock:
            self.records.append(record)

            # 检查记录数上限
            max_records = self.config.evolution.experience_pool.max_records
            if len(self.records) > max_records:
                # 删除最旧记录（按 timestamp 排序）
                self.records.sort(key=lambda r: r.timestamp)
                self.records = self.records[-max_records:]
                log_msg(
                    "INFO",
                    f"经验池记录数超过上限 {max_records}，已删除最旧记录",
                )

            # 持久化
            self.save()

    def query(
        self,
        task_type: Optional[str] = None,
        k: int = 10,
        **filters,
    ) -> List[TaskRecord]:
        """查询指定任务类型的 Top-K 记录。

        Args:
            task_type: 任务类型（"explore" | "merge" | "mutate"），None 表示查询所有任务类型
            k: 返回前 k 个记录
            **filters: 过滤条件，支持以下格式:
                - output_quality=(">", 0.5): 质量大于 0.5
                - agent_id="agent_0": Agent ID 等于 "agent_0"

        Returns:
            Top-K 记录列表，按 output_quality 降序排列

        时间复杂度: O(n log n)

        示例:
            >>> pool = ExperiencePool(config)
            >>> # 查询 explore 任务的 Top-5 高质量记录
            >>> results = pool.query("explore", k=5, output_quality=(">", 0.7))
            >>> len(results) <= 5
            True
            >>> all(r.task_type == "explore" for r in results)
            True
            >>> # 查询所有任务类型的 Top-3 记录
            >>> results = pool.query(task_type=None, k=3, agent_id="agent_0")
            >>> len(results) <= 3
            True
        """
        with self.lock:
            # [1] 过滤任务类型（如果指定）
            if task_type is not None:
                candidates = [r for r in self.records if r.task_type == task_type]
            else:
                candidates = self.records.copy()

            # [2] 应用额外过滤条件
            for field, condition in filters.items():
                candidates = self._apply_filter(candidates, field, condition)

            # [3] 按 output_quality 降序排序
            candidates.sort(key=lambda r: r.output_quality, reverse=True)

            # [4] 返回 Top-K
            return candidates[:k]

    def get_agent_stats(self, agent_id: str) -> Dict[str, Any]:
        """获取指定 Agent 的统计信息。

        Args:
            agent_id: Agent 唯一标识

        Returns:
            统计字典，包含以下字段:
                - total_count: 总记录数
                - success_count: 成功记录数（output_quality > 0）
                - avg_quality: 平均质量
                - success_rate: 成功率

        时间复杂度: O(n)

        示例:
            >>> stats = pool.get_agent_stats("agent_0")
            >>> stats.keys()
            dict_keys(['total_count', 'success_count', 'avg_quality', 'success_rate'])
        """
        with self.lock:
            agent_records = [r for r in self.records if r.agent_id == agent_id]

            total_count = len(agent_records)
            if total_count == 0:
                return {
                    "total_count": 0,
                    "success_count": 0,
                    "avg_quality": 0.0,
                    "success_rate": 0.0,
                }

            # 成功定义：output_quality > 0
            success_records = [r for r in agent_records if r.output_quality > 0]
            success_count = len(success_records)

            # 平均质量
            total_quality = sum(r.output_quality for r in agent_records)
            avg_quality = total_quality / total_count

            # 成功率
            success_rate = success_count / total_count

            return {
                "total_count": total_count,
                "success_count": success_count,
                "avg_quality": avg_quality,
                "success_rate": success_rate,
            }

    def save(self) -> None:
        """保存经验池到 JSON 文件。

        注意:
            - 使用 lock 保护，确保线程安全
            - 失败时记录错误但不抛出异常
        """
        try:
            # 转换为字典列表
            data = [asdict(record) for record in self.records]

            # 写入 JSON 文件
            with open(self.save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            log_msg(
                "DEBUG",
                f"经验池已保存: {self.save_path} ({len(self.records)} 条记录)",
            )

        except Exception as e:
            log_exception(e, "保存经验池失败")

    def load(self) -> None:
        """从 JSON 文件加载经验池。

        注意:
            - 如果文件不存在或损坏，从空白开始
            - 失败时记录警告但不抛出异常
        """
        if not self.save_path.exists():
            log_msg("INFO", f"经验池文件不存在，从空白开始: {self.save_path}")
            return

        try:
            with open(self.save_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 转换为 TaskRecord 对象
            self.records = [TaskRecord(**record) for record in data]

            log_msg(
                "INFO",
                f"经验池已加载: {self.save_path} ({len(self.records)} 条记录)",
            )

        except json.JSONDecodeError as e:
            log_msg(
                "WARNING",
                f"经验池文件损坏，从空白开始: {e}",
            )
            self.records = []

        except Exception as e:
            log_exception(e, "加载经验池失败，从空白开始")
            self.records = []

    def _apply_filter(
        self,
        records: List[TaskRecord],
        field: str,
        condition: Any,
    ) -> List[TaskRecord]:
        """应用单个过滤条件。

        Args:
            records: 待过滤的记录列表
            field: 字段名（如 "output_quality", "agent_id"）
            condition: 过滤条件，支持:
                - 简单值: "agent_0" → 相等过滤
                - 元组: (">", 0.5) → 比较过滤

        Returns:
            过滤后的记录列表

        时间复杂度: O(n)
        """
        if isinstance(condition, tuple):
            # 比较过滤（如 (">", 0.5)）
            op, value = condition
            return self._filter_by_comparison(records, field, op, value)
        else:
            # 相等过滤（如 "agent_0"）
            return [r for r in records if getattr(r, field, None) == condition]

    def _filter_by_comparison(
        self,
        records: List[TaskRecord],
        field: str,
        op: str,
        value: Any,
    ) -> List[TaskRecord]:
        """按比较运算符过滤记录。

        Args:
            records: 待过滤的记录列表
            field: 字段名
            op: 运算符（">", ">=", "<", "<=", "==", "!="）
            value: 比较值

        Returns:
            过滤后的记录列表

        时间复杂度: O(n)
        """
        ops: Dict[str, Callable[[Any, Any], bool]] = {
            ">": lambda a, b: a > b,
            ">=": lambda a, b: a >= b,
            "<": lambda a, b: a < b,
            "<=": lambda a, b: a <= b,
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
        }

        if op not in ops:
            log_msg("WARNING", f"不支持的比较运算符: {op}，跳过过滤")
            return records

        compare_func = ops[op]
        return [r for r in records if compare_func(getattr(r, field, None), value)]
