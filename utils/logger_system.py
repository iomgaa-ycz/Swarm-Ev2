"""日志系统模块。

提供文本日志和 JSON 日志的双通道输出功能。

修改说明（Phase 1 重构）:
- 去除 log_msg("ERROR") 的自动 raise 行为
- 新增 ensure() 断言工具
- 新增 log_exception() 异常记录函数
"""

import json
import datetime
from typing import Dict, Any, List
from pathlib import Path


class LoggerSystem:
    """日志系统类。

    提供文本日志（system.log）和 JSON 日志（metrics.json）的双通道输出。
    """

    def __init__(self, log_dir: str | Path):
        """初始化日志系统。

        Args:
            log_dir: 日志目录路径
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.text_log_path = self.log_dir / "system.log"
        self.json_log_path = self.log_dir / "metrics.json"

        # 初始化 JSON 日志列表
        self.json_data: List[Dict[str, Any]] = []
        if self.json_log_path.exists():
            try:
                content = self.json_log_path.read_text(encoding="utf-8")
                if content:
                    self.json_data = json.loads(content)
            except json.JSONDecodeError:
                self.json_data = []  # 损坏时重置

    def text_log(self, level: str, message: str) -> None:
        """记录文本日志到 system.log 并打印到终端。

        Args:
            level: 日志级别（INFO, WARNING, ERROR 等）
            message: 日志消息

        注意:
            Phase 1 重构后，ERROR 级别不再自动抛出异常。
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}\n"

        # 写入文件
        with open(self.text_log_path, "a", encoding="utf-8") as f:
            f.write(log_entry)

        # 打印到终端
        print(log_entry.strip())

    def json_log(self, data: Dict[str, Any]) -> None:
        """记录字典数据到 metrics.json。

        Args:
            data: 待记录的字典数据
        """
        self.json_data.append(data)

        with open(self.json_log_path, "w", encoding="utf-8") as f:
            json.dump(self.json_data, f, indent=4, ensure_ascii=False)


# ============================================================
# 全局日志实例
# ============================================================

logger: LoggerSystem | None = None


def init_logger(log_dir: str | Path) -> LoggerSystem:
    """初始化全局日志系统。

    Args:
        log_dir: 日志目录路径

    Returns:
        初始化的 LoggerSystem 实例
    """
    global logger
    logger = LoggerSystem(log_dir)
    return logger


# ============================================================
# 便捷日志函数
# ============================================================


def log_msg(level: str, message: str) -> None:
    """记录文本日志（线程安全）。

    Args:
        level: 日志级别（INFO, WARNING, ERROR 等）
        message: 日志消息

    注意:
        - 如果 logger 未初始化，回退到 print
        - Phase 1 重构后，ERROR 级别不再自动抛出异常
    """
    if logger:
        logger.text_log(level, message)
    else:
        # 回退模式：直接打印
        print(f"[{level}] {message}")


def log_json(data: Dict[str, Any]) -> None:
    """记录 JSON 数据。

    Args:
        data: 待记录的字典数据

    注意:
        如果 logger 未初始化，回退到打印 JSON 字符串
    """
    if logger:
        logger.json_log(data)
    else:
        # 回退模式：打印格式化 JSON
        print(f"[JSON] {json.dumps(data, indent=2, ensure_ascii=False)}")


# ============================================================
# Phase 1 新增工具函数
# ============================================================


def ensure(condition: bool, error_msg: str) -> None:
    """断言工具，失败时记录错误并抛出异常。

    Args:
        condition: 断言条件
        error_msg: 错误消息

    Raises:
        AssertionError: 条件为 False 时抛出

    示例:
        >>> ensure(x > 0, "x 必须为正数")
        >>> ensure(cfg.data_dir.exists(), "数据目录不存在")
    """
    if not condition:
        log_msg("ERROR", error_msg)
        raise AssertionError(error_msg)


def log_exception(exc: Exception, context: str = "") -> None:
    """记录异常信息和堆栈跟踪。

    Args:
        exc: 异常对象
        context: 上下文描述（可选）

    示例:
        >>> try:
        ...     risky_operation()
        ... except Exception as e:
        ...     log_exception(e, "执行风险操作时")
    """
    import traceback

    error_msg = f"{context}: {exc}" if context else str(exc)
    traceback_str = "".join(traceback.format_tb(exc.__traceback__))
    full_msg = f"{error_msg}\n{traceback_str}"

    log_msg("ERROR", full_msg)
