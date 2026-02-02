"""
代码执行沙箱模块（subprocess 版本）。

使用 subprocess.Popen 在指定 Python 环境中执行代码，支持：
- 显式指定 Python 解释器路径（解决 conda 环境问题）
- 超时控制和输出捕获
- 并行执行（通过槽位管理）
"""

import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Optional, List, Tuple

from utils.logger_system import log_msg


def trim_long_string(string: str, threshold: int = 5100, k: int = 2500) -> str:
    """截断过长的字符串，保留首尾 k 个字符。

    Args:
        string: 输入字符串
        threshold: 超过此长度才截断
        k: 保留首尾各 k 个字符

    Returns:
        截断后的字符串
    """
    if len(string) > threshold:
        first_k = string[:k]
        last_k = string[-k:]
        truncated_len = len(string) - 2 * k
        return f"{first_k}\n ... [{truncated_len} 字符被截断] ... \n{last_k}"
    return string


@dataclass
class ExecutionResult:
    """代码执行结果容器。

    Attributes:
        term_out: 终端输出（stdout + stderr 合并）
        exec_time: 执行时间（秒）
        exc_type: 异常类型（如 "ValueError", "KeyError"），None 表示无异常
        exc_info: 异常详情（字典或字符串）
        exc_stack: 异常堆栈（文件名、行号、函数名、代码行）
        success: 执行是否成功（无异常且无超时）
        timeout: 是否超时
    """

    term_out: List[str] = field(default_factory=list)
    exec_time: float = 0.0
    exc_type: Optional[str] = None
    exc_info: Optional[dict] = None
    exc_stack: Optional[List[Tuple]] = None
    success: bool = True
    timeout: bool = False


class Interpreter:
    """Python 代码执行沙箱（subprocess 版本）。

    使用 subprocess.Popen 执行代码，支持指定 Python 解释器路径。
    """

    def __init__(
        self,
        working_dir: Path,
        timeout: int = 3600,
        max_parallel_run: int = 3,
        python_path: Optional[str] = None,
    ):
        """初始化执行器。

        Args:
            working_dir: 工作目录（代码在此目录执行）
            timeout: 超时时间（秒，默认 3600）
            max_parallel_run: 最大并行进程数（默认 3）
            python_path: Python 解释器路径，None 则使用 sys.executable
        """
        self.working_dir = Path(working_dir).resolve()
        self.timeout = timeout
        self.max_parallel_run = max_parallel_run
        self.python_path = python_path or sys.executable

        if not self.working_dir.exists():
            self.working_dir.mkdir(parents=True, exist_ok=True)

        # 进程管理
        self.process: List[Optional[subprocess.Popen]] = [None] * max_parallel_run
        self.status_map = [0] * max_parallel_run  # 0=空闲, 1=占用
        self.current_parallel_run = 0

        # 进程分配锁
        self.lock = Lock()

        # 脚本文件名（避免并行冲突）
        self.script_names = [f"runfile_{i}.py" for i in range(max_parallel_run)]

        log_msg(
            "INFO",
            f"Interpreter 初始化: working_dir={self.working_dir}, "
            f"timeout={self.timeout}s, python={self.python_path}",
        )

    def check_available(self) -> bool:
        """检查是否有可用的执行槽位。"""
        return self.current_parallel_run < self.max_parallel_run

    def _replace_submission_name(self, code: str, node_id: str) -> str:
        """将 submission.csv 替换为 submission_{node_id}.csv。

        防止并行执行时多个进程写入同一文件。
        """
        submission_file = f"submission_{node_id}.csv"
        replacements = [
            ("submission/submission.csv", f"submission/{submission_file}"),
            ("/submission.csv", f"/{submission_file}"),
            ("to_csv('submission.csv", f"to_csv('submission/{submission_file}"),
            ('to_csv("submission.csv', f'to_csv("submission/{submission_file}'),
            ('"submission.csv"', f'"{submission_file}"'),
            ("'submission.csv'", f"'{submission_file}'"),
        ]
        for old, new in replacements:
            code = code.replace(old, new)
        return code

    def _parse_exception(
        self, output: str, script_name: str
    ) -> Tuple[Optional[str], Optional[dict], Optional[List[Tuple]]]:
        """从输出中解析异常信息。

        Args:
            output: 终端输出
            script_name: 脚本文件名

        Returns:
            (异常类型, 异常信息字典, 异常堆栈)
        """
        # 匹配 Python 异常格式: ExceptionType: message
        exc_pattern = r"(\w+Error|\w+Exception|KeyboardInterrupt): (.+?)(?:\n|$)"
        match = re.search(exc_pattern, output)

        if not match:
            return None, None, None

        exc_type = match.group(1)
        exc_msg = match.group(2).strip()

        # 构建异常信息
        exc_info = {"args": [exc_msg], "msg": exc_msg}
        if "'" in exc_msg:
            # 提取模块名（如 "No module named 'xgboost'" -> name='xgboost'）
            name_match = re.search(r"'(\w+)'", exc_msg)
            if name_match:
                exc_info["name"] = name_match.group(1)

        # 解析堆栈（简化版）
        exc_stack = []
        tb_pattern = r'File "([^"]+)", line (\d+), in (\w+)'
        for m in re.finditer(tb_pattern, output):
            filename, lineno, funcname = m.groups()
            # 简化路径
            if script_name in filename:
                filename = script_name
            exc_stack.append((filename, int(lineno), funcname, ""))

        return exc_type, exc_info, exc_stack if exc_stack else None

    def _cleanup_process(self, process_id: int) -> None:
        """清理指定进程。"""
        proc = self.process[process_id]
        if proc is None:
            return

        try:
            if proc.poll() is None:  # 仍在运行
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    log_msg("WARNING", f"进程 {process_id} 未能优雅终止，强制终止")
                    proc.kill()
                    proc.wait()
        except Exception as e:
            log_msg("WARNING", f"清理进程 {process_id} 时出错: {e}")
        finally:
            self.process[process_id] = None

    def cleanup_session(self, process_id: int) -> None:
        """清理指定进程（兼容旧接口）。

        Args:
            process_id: 进程 ID，-1 表示清理所有进程
        """
        if process_id == -1:
            for pid in range(self.max_parallel_run):
                self._cleanup_process(pid)
        else:
            self._cleanup_process(process_id)

    def run(
        self,
        code: str,
        node_id: str = "",
        reset_session: bool = True,
    ) -> ExecutionResult:
        """执行 Python 代码。

        Args:
            code: Python 代码字符串
            node_id: 节点 ID（用于 submission 文件隔离）
            reset_session: 是否重置会话（subprocess 版本始终为独立执行）

        Returns:
            ExecutionResult 对象
        """
        process_id = None

        # Phase 1: 分配槽位
        with self.lock:
            for idx in range(self.max_parallel_run):
                if self.status_map[idx] == 0:
                    self.status_map[idx] = 1
                    process_id = idx
                    self.current_parallel_run += 1
                    break

            if process_id is None:
                log_msg("ERROR", "达到最大并行数限制")
                return ExecutionResult(
                    term_out=["错误: 达到最大并行数限制"],
                    exc_type="RuntimeError",
                    success=False,
                )

        try:
            # Phase 2: 替换 submission 文件名
            if node_id:
                code = self._replace_submission_name(code, node_id)

            # Phase 3: 写入脚本文件
            script_name = self.script_names[process_id]
            script_path = self.working_dir / script_name
            script_path.write_text(code, encoding="utf-8")

            log_msg(
                "DEBUG",
                f"执行代码 (slot={process_id}, node={node_id[:8] if node_id else 'N/A'})",
            )

            # Phase 4: 启动子进程
            start_time = time.time()
            self.process[process_id] = subprocess.Popen(
                [self.python_path, script_name],
                cwd=str(self.working_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )

            # Phase 5: 等待完成（带超时）
            try:
                stdout, _ = self.process[process_id].communicate(timeout=self.timeout)
                exec_time = time.time() - start_time
                return_code = self.process[process_id].returncode
                is_timeout = False
            except subprocess.TimeoutExpired:
                # 超时处理
                log_msg("WARNING", f"执行超时 ({self.timeout}s)，终止进程")
                os.kill(self.process[process_id].pid, signal.SIGINT)
                try:
                    stdout, _ = self.process[process_id].communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process[process_id].kill()
                    stdout, _ = self.process[process_id].communicate()
                exec_time = self.timeout
                return_code = -1
                is_timeout = True

            # Phase 6: 解析输出
            output_lines = [stdout] if stdout else []
            exc_type, exc_info, exc_stack = None, None, None

            if is_timeout:
                exc_type = "TimeoutError"
                exc_info = {"args": [f"执行超过 {self.timeout} 秒"]}
                output_lines.append(f"TimeoutError: 执行超过 {self.timeout} 秒")
            elif return_code != 0:
                exc_type, exc_info, exc_stack = self._parse_exception(
                    stdout or "", script_name
                )
                if exc_type == "KeyboardInterrupt":
                    exc_type = "TimeoutError"

            output_lines.append(
                f"执行时间: {exec_time:.2f} 秒 (限制: {self.timeout} 秒)"
            )

            # Phase 7: 构建结果
            return ExecutionResult(
                term_out=output_lines,
                exec_time=exec_time,
                exc_type=exc_type,
                exc_info=exc_info,
                exc_stack=exc_stack,
                success=return_code == 0 and not is_timeout,
                timeout=is_timeout,
            )

        finally:
            # Phase 8: 清理
            with self.lock:
                self.current_parallel_run -= 1
                self.status_map[process_id] = 0
            self._cleanup_process(process_id)

            # 清理脚本文件
            script_path = self.working_dir / self.script_names[process_id]
            if script_path.exists():
                try:
                    script_path.unlink()
                except Exception:
                    pass

    def __del__(self):
        """析构函数，清理所有进程。"""
        try:
            self.cleanup_session(-1)
        except Exception:
            pass
