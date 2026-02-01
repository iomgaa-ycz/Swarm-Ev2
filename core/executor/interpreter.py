"""
代码执行沙箱模块（并行版本）。

使用 multiprocessing 在隔离环境中并行执行 Python 代码，提供超时控制和输出捕获。
参考: Reference/ML-Master-main/interpreter/interpreter_parallel.py
"""

import os
import queue
import signal
import sys
import time
import traceback
from dataclasses import dataclass, field
from multiprocessing import Process, Queue, Lock
from pathlib import Path
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


class RedirectQueue:
    """重定向输出到队列的类。"""

    def __init__(self, q: Queue):
        self.queue = q

    def write(self, msg: str):
        self.queue.put(msg)

    def flush(self):
        pass


def exception_summary(
    e: BaseException,
    working_dir: Path,
    exec_file_name: str,
) -> Tuple[str, str, dict, List[Tuple]]:
    """生成异常摘要信息。

    Args:
        e: 异常对象
        working_dir: 工作目录
        exec_file_name: 执行文件名

    Returns:
        (traceback字符串, 异常类名, 异常信息字典, 异常堆栈列表)
    """
    tb_lines = traceback.format_exception(e)
    # 过滤掉框架内部的堆栈信息
    tb_str = "".join(
        [line for line in tb_lines if "swarm-ev2/" not in line.lower() and "importlib" not in line]
    )

    # 替换完整路径为文件名
    tb_str = tb_str.replace(str(working_dir / exec_file_name), exec_file_name)

    exc_info = {}
    if hasattr(e, "args"):
        exc_info["args"] = [str(i) for i in e.args]
    for att in ["name", "msg", "obj"]:
        if hasattr(e, att):
            exc_info[att] = str(getattr(e, att))

    tb = traceback.extract_tb(e.__traceback__)
    exc_stack = [(t.filename, t.lineno, t.name, t.line) for t in tb]

    return tb_str, e.__class__.__name__, exc_info, exc_stack


class Interpreter:
    """Python 代码执行沙箱（支持并行执行）。

    使用独立的 multiprocessing.Process 执行代码，支持多进程并行。
    """

    def __init__(
        self,
        working_dir: Path,
        timeout: int = 3600,
        max_parallel_run: int = 3,
    ):
        """初始化执行器。

        Args:
            working_dir: 工作目录（代码在此目录执行）
            timeout: 超时时间（秒，默认 3600）
            max_parallel_run: 最大并行进程数（默认 3）
        """
        self.working_dir = Path(working_dir).resolve()
        self.timeout = timeout
        self.max_parallel_run = max_parallel_run

        if not self.working_dir.exists():
            self.working_dir.mkdir(parents=True, exist_ok=True)

        # 进程管理
        self.process: List[Optional[Process]] = [None] * max_parallel_run
        self.status_map = [0] * max_parallel_run  # 0=空闲, 1=占用
        self.current_parallel_run = 0

        # 队列通信
        self.code_inq: List[Optional[Queue]] = [None] * max_parallel_run
        self.result_outq: List[Optional[Queue]] = [None] * max_parallel_run
        self.event_outq: List[Optional[Queue]] = [None] * max_parallel_run

        # 进程分配锁
        self.lock = Lock()

        # 脚本文件名（避免并行冲突）
        self.agent_file_name = [f"runfile_{i}.py" for i in range(max_parallel_run)]

        log_msg(
            "INFO",
            f"Interpreter 初始化: working_dir={self.working_dir}, "
            f"timeout={self.timeout}s, max_parallel={max_parallel_run}",
        )

    def check_available(self) -> bool:
        """检查是否有可用的执行槽位。

        Returns:
            True 表示有空闲槽位
        """
        return self.current_parallel_run < self.max_parallel_run

    def _replace_submission_name(self, code: str, node_id: str) -> str:
        """将 submission.csv 替换为 submission_{node_id}.csv。

        防止并行执行时多个进程写入同一文件。

        Args:
            code: 原始代码
            node_id: 节点 ID

        Returns:
            替换后的代码
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

    @staticmethod
    def _child_proc_setup(result_outq: Queue, working_dir: Path) -> None:
        """子进程初始化设置。

        Args:
            result_outq: 结果输出队列
            working_dir: 工作目录
        """
        try:
            # 禁用警告
            import warnings

            warnings.filterwarnings("ignore")

            # 设置工作目录
            os.chdir(str(working_dir))

            # 添加到 sys.path
            sys.path.insert(0, str(working_dir))

            # 重定向 stdout/stderr
            sys.stdout = sys.stderr = RedirectQueue(result_outq)

        except Exception:
            result_outq.put(f"[子进程初始化错误] {traceback.format_exc()}")
            raise

    @staticmethod
    def _run_session(
        code_inq: Queue,
        result_outq: Queue,
        event_outq: Queue,
        process_id: int,
        working_dir: Path,
        agent_file_name: str,
    ) -> None:
        """子进程主循环（在独立进程中运行）。

        Args:
            code_inq: 代码输入队列
            result_outq: 结果输出队列
            event_outq: 事件输出队列
            process_id: 进程 ID
            working_dir: 工作目录
            agent_file_name: 脚本文件名
        """
        Interpreter._child_proc_setup(result_outq, working_dir)

        global_scope: dict = {"__name__": "__main__"}

        while True:
            code, node_id = code_inq.get()

            # 确保在正确的工作目录
            os.chdir(str(working_dir))

            # 写入临时脚本
            script_path = working_dir / agent_file_name
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code)

            event_outq.put(("state:ready",))

            try:
                exec(
                    compile(code, agent_file_name, "exec"),
                    global_scope,
                )
                event_outq.put(("state:finished", None, None, None))
            except BaseException as e:
                tb_str, e_cls_name, exc_info, exc_stack = exception_summary(
                    e, working_dir, agent_file_name
                )
                result_outq.put(tb_str)

                if e_cls_name == "KeyboardInterrupt":
                    e_cls_name = "TimeoutError"

                event_outq.put(("state:finished", e_cls_name, exc_info, exc_stack))

            # 清理脚本文件（避免被 data_preview 包含）
            if script_path.exists():
                try:
                    os.remove(script_path)
                except Exception:
                    pass

            # EOF 标记
            result_outq.put("<|EOF|>")

    def create_process(self, process_id: int) -> None:
        """创建子进程。

        Args:
            process_id: 进程 ID
        """
        log_msg("DEBUG", f"创建进程 {process_id}")

        self.code_inq[process_id] = Queue()
        self.result_outq[process_id] = Queue()
        self.event_outq[process_id] = Queue()

        self.process[process_id] = Process(
            target=Interpreter._run_session,
            args=(
                self.code_inq[process_id],
                self.result_outq[process_id],
                self.event_outq[process_id],
                process_id,
                self.working_dir,
                self.agent_file_name[process_id],
            ),
        )
        self.process[process_id].start()

    def cleanup_session(self, process_id: int) -> None:
        """清理指定进程。

        Args:
            process_id: 进程 ID，-1 表示清理所有进程
        """
        if process_id == -1:
            # 清理所有进程
            for pid in range(self.max_parallel_run):
                self._cleanup_single_process(pid)
        else:
            self._cleanup_single_process(process_id)

    def _cleanup_single_process(self, process_id: int) -> None:
        """清理单个进程。

        Args:
            process_id: 进程 ID
        """
        if self.process[process_id] is None:
            return

        try:
            # 优雅终止
            self.process[process_id].terminate()
            self.process[process_id].join(timeout=2)

            # 强制终止
            if self.process[process_id].exitcode is None:
                log_msg("WARNING", f"进程 {process_id} 未能优雅终止，强制终止")
                self.process[process_id].kill()
                self.process[process_id].join()

            # 清理资源
            self.process[process_id].close()
        except Exception as e:
            log_msg("WARNING", f"清理进程 {process_id} 时出错: {e}")
        finally:
            self.process[process_id] = None

    def run(
        self,
        code: str,
        node_id: str = "",
        reset_session: bool = True,
    ) -> ExecutionResult:
        """执行 Python 代码（并行安全）。

        Args:
            code: Python 代码字符串
            node_id: 节点 ID（用于 submission 文件隔离）
            reset_session: 是否重置会话（默认 True）

        Returns:
            ExecutionResult 对象
        """
        log_msg(
            "DEBUG",
            f"执行代码 (reset_session={reset_session}, node_id={node_id[:8] if node_id else 'N/A'})",
        )

        process_id = None

        # Phase 1: 分配进程 ID
        with self.lock:
            self.current_parallel_run += 1
            for idx in range(self.max_parallel_run):
                if self.status_map[idx] == 0:
                    self.status_map[idx] = 1
                    process_id = idx
                    log_msg("DEBUG", f"分配进程 ID: {process_id}")
                    break

            if process_id is None:
                self.current_parallel_run -= 1
                log_msg("ERROR", "达到最大并行数限制")
                return ExecutionResult(
                    term_out=["错误: 达到最大并行数限制"],
                    exec_time=0,
                    exc_type="RuntimeError",
                    success=False,
                )

        try:
            # Phase 2: 创建/重置进程
            if reset_session:
                if self.process[process_id] is not None:
                    try:
                        self.cleanup_session(process_id)
                    except Exception as e:
                        log_msg("WARNING", f"重置进程时出错: {e}")

                self.create_process(process_id)
            else:
                assert self.process[process_id] is not None

            assert self.process[process_id].is_alive()

            # Phase 3: 替换 submission 文件名
            if node_id:
                code = self._replace_submission_name(code, node_id)

            # Phase 4: 发送代码到子进程
            self.code_inq[process_id].put((code, node_id))

            # Phase 5: 等待子进程准备就绪
            try:
                state = self.event_outq[process_id].get(timeout=30)
            except queue.Empty:
                msg = "子进程启动超时"
                log_msg("ERROR", msg)
                queue_dump = self._drain_queue(self.result_outq[process_id])
                self.cleanup_session(process_id)
                self.current_parallel_run -= 1
                self.status_map[process_id] = 0
                return ExecutionResult(
                    term_out=[msg, queue_dump],
                    exec_time=0,
                    exc_type="RuntimeError",
                    success=False,
                )

            assert state[0] == "state:ready", state

            # Phase 6: 等待执行完成
            start_time = time.time()
            child_in_overtime = False

            while True:
                try:
                    state = self.event_outq[process_id].get(timeout=1)
                    assert state[0] == "state:finished", state
                    exec_time = time.time() - start_time
                    break
                except queue.Empty:
                    # 检查子进程是否存活
                    if (
                        not child_in_overtime
                        and not self.process[process_id].is_alive()
                    ):
                        msg = "子进程意外终止"
                        log_msg("ERROR", msg)
                        queue_dump = self._drain_queue(self.result_outq[process_id])
                        self.cleanup_session(process_id)
                        self.current_parallel_run -= 1
                        self.status_map[process_id] = 0
                        return ExecutionResult(
                            term_out=[msg, queue_dump],
                            exec_time=0,
                            exc_type="RuntimeError",
                            success=False,
                        )

                    # 检查超时
                    if self.timeout is not None:
                        running_time = time.time() - start_time
                        if running_time > self.timeout:
                            assert reset_session, "超时发生在交互式会话中"
                            os.kill(self.process[process_id].pid, signal.SIGINT)
                            child_in_overtime = True

                            # 超时超过 1 分钟，强制终止
                            if running_time > self.timeout + 60:
                                log_msg("WARNING", "子进程超时未响应，强制终止")
                                self.cleanup_session(process_id)
                                state = (None, "TimeoutError", {}, [])
                                exec_time = self.timeout
                                break

            # Phase 7: 收集输出
            output: List[str] = []
            while (
                not self.result_outq[process_id].empty()
                or not output
                or output[-1] != "<|EOF|>"
            ):
                try:
                    res = self.result_outq[process_id].get(timeout=1)
                    output.append(res)
                except queue.Empty:
                    break

            if output and output[-1] == "<|EOF|>":
                output.pop()

            e_cls_name, exc_info, exc_stack = state[1:]

            if e_cls_name == "TimeoutError":
                output.append(f"TimeoutError: 执行超过 {self.timeout} 秒")
            else:
                output.append(f"执行时间: {exec_time:.2f} 秒 (限制: {self.timeout} 秒)")

            # Phase 8: 构建结果
            result = ExecutionResult(
                term_out=output,
                exec_time=exec_time,
                exc_type=e_cls_name,
                exc_info=exc_info,
                exc_stack=exc_stack,
                success=e_cls_name is None,
                timeout=e_cls_name == "TimeoutError",
            )

            return result

        finally:
            # Phase 9: 清理
            self.current_parallel_run -= 1
            self.status_map[process_id] = 0
            if reset_session:
                self.cleanup_session(process_id)

    def _drain_queue(self, q: Queue) -> str:
        """清空队列并返回内容。

        Args:
            q: 队列

        Returns:
            队列中的所有内容拼接
        """
        items = []
        while not q.empty():
            try:
                items.append(str(q.get_nowait()))
            except queue.Empty:
                break
        return "\n".join(items)

    def __del__(self):
        """析构函数，清理所有进程。"""
        try:
            self.cleanup_session(-1)
        except Exception:
            pass
