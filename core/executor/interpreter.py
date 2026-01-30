"""
代码执行沙箱模块。

使用 subprocess 在隔离环境中执行 Python 代码，提供超时控制和输出捕获。
"""

import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

from utils.logger_system import log_msg


@dataclass
class ExecutionResult:
    """代码执行结果容器。

    Attributes:
        term_out: 终端输出（stdout + stderr 合并）
        exec_time: 执行时间（秒）
        exc_type: 异常类型（如 "ValueError", "KeyError"），None 表示无异常
        exc_info: 异常详情（traceback 最后几行）
        success: 执行是否成功（无异常且无超时）
        timeout: 是否超时
    """

    term_out: List[str] = field(default_factory=list)
    exec_time: float = 0.0
    exc_type: Optional[str] = None
    exc_info: Optional[str] = None
    success: bool = True
    timeout: bool = False


class Interpreter:
    """Python 代码执行沙箱。

    使用独立的 subprocess 执行代码，提供超时控制和异常捕获。
    """

    def __init__(self, working_dir: Path, timeout: int = 300):
        """初始化执行器。

        Args:
            working_dir: 工作目录（代码在此目录执行）
            timeout: 超时时间（秒，默认 300）
        """
        self.working_dir = Path(working_dir)
        self.timeout = timeout

        if not self.working_dir.exists():
            self.working_dir.mkdir(parents=True, exist_ok=True)

        log_msg(
            "INFO",
            f"Interpreter 初始化: working_dir={self.working_dir}, timeout={self.timeout}s",
        )

    def run(self, code: str, reset_session: bool = True) -> ExecutionResult:
        """执行 Python 代码。

        Args:
            code: Python 代码字符串
            reset_session: 是否重置会话（Phase 2 始终为 True，即每次启动新进程）

        Returns:
            ExecutionResult 对象
        """
        start_time = time.time()
        result = ExecutionResult()

        try:
            # 执行代码
            stdout, stderr, returncode = self._execute_in_subprocess(code)

            # 合并输出
            result.term_out = [stdout, stderr]

            # 检查执行结果
            if returncode != 0:
                result.success = False
                # 解析异常信息
                exc_type, exc_info = self._capture_exception(stderr)
                result.exc_type = exc_type
                result.exc_info = exc_info

        except subprocess.TimeoutExpired:
            result.success = False
            result.timeout = True
            result.exc_type = "TimeoutError"
            result.exc_info = f"代码执行超过 {self.timeout} 秒"
            result.term_out.append(f"TimeoutError: 执行超过 {self.timeout} 秒")
            log_msg("WARNING", f"代码执行超时: {self.timeout}s")

        except Exception as e:
            result.success = False
            result.exc_type = type(e).__name__
            result.exc_info = str(e)
            result.term_out.append(f"执行器内部错误: {e}")
            log_msg("ERROR", f"执行器内部错误: {e}")

        # 记录执行时间
        result.exec_time = time.time() - start_time

        return result

    def _execute_in_subprocess(self, code: str) -> tuple[str, str, int]:
        """在子进程中执行代码。

        Args:
            code: Python 代码字符串

        Returns:
            (stdout, stderr, returncode)

        Raises:
            subprocess.TimeoutExpired: 如果执行超时
        """
        # 创建临时脚本文件
        script_path = self.working_dir / "_temp_script.py"
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)

        try:
            # 执行脚本
            proc = subprocess.run(
                ["python", str(script_path)],
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                encoding="utf-8",
                errors="replace",  # 处理编码错误
            )

            return proc.stdout, proc.stderr, proc.returncode

        finally:
            # 清理临时文件
            if script_path.exists():
                script_path.unlink()

    def _capture_exception(self, stderr: str) -> tuple[Optional[str], Optional[str]]:
        """从 stderr 中解析异常信息。

        Args:
            stderr: 标准错误输出

        Returns:
            (异常类型, 异常详情)

        Examples:
            >>> _capture_exception("ValueError: invalid literal\\n  File ...")
            ("ValueError", "ValueError: invalid literal\\n  File ...")
        """
        if not stderr:
            return None, None

        # 提取异常类型（最后一行通常是异常信息）
        lines = stderr.strip().split("\n")
        last_line = lines[-1] if lines else ""

        # 匹配异常类型（如 "ValueError:", "KeyError:", etc.）
        exc_type_match = re.match(r"^(\w+Error|\w+Exception):", last_line)
        if exc_type_match:
            exc_type = exc_type_match.group(1)
        else:
            exc_type = "UnknownError"

        # 提取 traceback 最后 5 行作为详情
        exc_info = "\n".join(lines[-5:])

        return exc_type, exc_info
