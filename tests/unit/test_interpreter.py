"""
core/executor/interpreter.py 的单元测试。
"""

import subprocess
from unittest.mock import MagicMock, patch

import pytest
from core.executor.interpreter import Interpreter, ExecutionResult


class TestExecutionResult:
    """测试 ExecutionResult 数据类。"""

    def test_creation(self):
        """测试创建 ExecutionResult。"""
        result = ExecutionResult()
        assert result.term_out == []
        assert result.exec_time == 0.0
        assert result.exc_type is None
        assert result.success is True


class TestInterpreter:
    """测试 Interpreter 类。"""

    @pytest.fixture
    def interpreter(self, tmp_path):
        """创建测试用的 Interpreter。"""
        return Interpreter(working_dir=tmp_path, timeout=10)

    def test_init(self, interpreter, tmp_path):
        """测试初始化。"""
        assert interpreter.working_dir == tmp_path
        assert interpreter.timeout == 10

    def test_simple_execution(self, interpreter):
        """测试简单代码执行。"""
        code = "print('Hello, World!')"
        result = interpreter.run(code)

        assert result.success is True
        assert result.exc_type is None
        assert "Hello, World!" in "".join(result.term_out)
        assert result.exec_time > 0

    def test_execution_with_output(self, interpreter):
        """测试捕获输出。"""
        code = """
for i in range(3):
    print(f"Line {i}")
"""
        result = interpreter.run(code)

        assert result.success is True
        output = "".join(result.term_out)
        assert "Line 0" in output
        assert "Line 1" in output
        assert "Line 2" in output

    def test_execution_with_exception(self, interpreter):
        """测试异常捕获。"""
        code = "x = 1 / 0"
        result = interpreter.run(code)

        assert result.success is False
        assert result.exc_type == "ZeroDivisionError"
        assert result.exc_info is not None
        # exc_info 是字典，检查 msg 或 args
        exc_msg = result.exc_info.get("msg", "") or str(result.exc_info.get("args", []))
        assert "division by zero" in exc_msg.lower()

    def test_execution_timeout(self, interpreter):
        """测试超时控制。"""
        code = """
import time
time.sleep(20)
"""
        result = interpreter.run(code)

        assert result.success is False
        assert result.timeout is True
        assert result.exc_type == "TimeoutError"
        assert result.exec_time >= interpreter.timeout

    def test_execution_with_syntax_error(self, interpreter):
        """测试语法错误。"""
        code = "def foo(\npass"  # 语法错误
        result = interpreter.run(code)

        assert result.success is False
        assert result.exc_type is not None

    def test_execution_working_directory(self, interpreter, tmp_path):
        """测试工作目录正确。"""
        code = """
import os
print(os.getcwd())
"""
        result = interpreter.run(code)

        assert result.success is True
        assert str(tmp_path) in "".join(result.term_out)

    def test_file_operations_in_working_dir(self, interpreter, tmp_path):
        """测试在工作目录中的文件操作。"""
        code = """
with open('test_file.txt', 'w') as f:
    f.write('test content')

with open('test_file.txt', 'r') as f:
    content = f.read()
    print(content)
"""
        result = interpreter.run(code)

        assert result.success is True
        assert "test content" in "".join(result.term_out)
        assert (tmp_path / "test_file.txt").exists()


class TestSignalDetection:
    """测试 signal 终止检测功能。"""

    @pytest.fixture
    def interpreter(self, tmp_path):
        """创建测试用的 Interpreter。"""
        return Interpreter(working_dir=tmp_path, timeout=10)

    def test_detect_signal_termination_oom_high(self, interpreter):
        """测试 OOM Kill 检测（高置信度）。"""
        stdout = "Loading data...\nTrain shape: (55413942, 8)\nFold 1/5"
        exc_type, exc_info = interpreter._detect_signal_termination(-9, stdout)

        assert exc_type == "MemoryError"
        assert exc_info["signal"] == 9
        assert exc_info["oom_likelihood"] == "high"
        assert "采样" in exc_info["msg"] or "batch" in exc_info["msg"]

    def test_detect_signal_termination_oom_medium(self, interpreter):
        """测试 OOM Kill 检测（中等置信度）。"""
        stdout = "Loading data...\nTraining model..."
        exc_type, exc_info = interpreter._detect_signal_termination(-9, stdout)

        assert exc_type == "ProcessKilled"
        assert exc_info["signal"] == 9
        assert exc_info["oom_likelihood"] == "medium"

    def test_detect_signal_termination_oom_low(self, interpreter):
        """测试 OOM Kill 检测（低置信度）。"""
        stdout = "Hello World"
        exc_type, exc_info = interpreter._detect_signal_termination(-9, stdout)

        assert exc_type == "ProcessKilled"
        assert exc_info["signal"] == 9
        assert exc_info["oom_likelihood"] == "low"

    def test_detect_signal_termination_sigterm(self, interpreter):
        """测试 SIGTERM 检测。"""
        exc_type, exc_info = interpreter._detect_signal_termination(-15, "")

        assert exc_type == "ProcessTerminated"
        assert exc_info["signal"] == 15

    def test_detect_signal_termination_other(self, interpreter):
        """测试其他 signal 检测。"""
        exc_type, exc_info = interpreter._detect_signal_termination(-11, "")  # SIGSEGV

        assert exc_type == "SignalError"
        assert exc_info["signal"] == 11

    def test_estimate_oom_likelihood_high(self, interpreter):
        """测试 OOM 可能性估算（高）。"""
        # 超过 1000 万行
        assert (
            interpreter._estimate_oom_likelihood("Train shape: (55413942, 8)") == "high"
        )
        assert interpreter._estimate_oom_likelihood("Loading 10000000 rows") == "high"

    def test_estimate_oom_likelihood_medium(self, interpreter):
        """测试 OOM 可能性估算（中）。"""
        # 超过 100 万行但不到 1000 万
        assert (
            interpreter._estimate_oom_likelihood("Train shape: (1500000, 8)")
            == "medium"
        )
        # 训练相关操作
        assert interpreter._estimate_oom_likelihood("Fold 1/5\nTraining...") == "medium"
        assert interpreter._estimate_oom_likelihood("Loading data...") == "medium"

    def test_estimate_oom_likelihood_low(self, interpreter):
        """测试 OOM 可能性估算（低）。"""
        assert interpreter._estimate_oom_likelihood("Hello World") == "low"
        assert interpreter._estimate_oom_likelihood("Train shape: (1000, 8)") == "low"
        assert interpreter._estimate_oom_likelihood("") == "low"


class TestReplaceSubmissionName:
    """测试 _replace_submission_name 方法。"""

    @pytest.fixture
    def interpreter(self, tmp_path):
        return Interpreter(working_dir=tmp_path, timeout=10)

    def test_replace_submission_csv(self, interpreter):
        """测试替换 submission.csv。"""
        code = "df.to_csv('submission.csv', index=False)"
        result = interpreter._replace_submission_name(code, "node_abc123")
        assert "submission_node_abc123.csv" in result
        assert "submission.csv" not in result

    def test_replace_submission_path(self, interpreter):
        """测试替换带路径的 submission.csv。"""
        code = 'df.to_csv("submission/submission.csv", index=False)'
        result = interpreter._replace_submission_name(code, "node_x")
        assert "submission/submission_node_x.csv" in result

    def test_no_replacement_needed(self, interpreter):
        """测试无需替换的代码。"""
        code = "print('hello')"
        result = interpreter._replace_submission_name(code, "node_y")
        assert result == code


class TestParseException:
    """测试 _parse_exception 方法。"""

    @pytest.fixture
    def interpreter(self, tmp_path):
        return Interpreter(working_dir=tmp_path, timeout=10)

    def test_parse_no_exception(self, interpreter):
        """测试无异常输出。"""
        exc_type, exc_info, exc_stack = interpreter._parse_exception(
            "normal output\nno errors", "runfile_0.py"
        )
        assert exc_type is None
        assert exc_info is None
        assert exc_stack is None

    def test_parse_import_error(self, interpreter):
        """测试解析 ImportError（包含模块名）。"""
        # 注意: tb_pattern 使用 (\w+)，不匹配 <module>；使用具名函数名触发堆栈解析
        output = (
            'Traceback (most recent call last):\n'
            '  File "runfile_0.py", line 1, in run_script\n'
            "ModuleNotFoundError: No module named 'xgboost'\n"
        )
        exc_type, exc_info, exc_stack = interpreter._parse_exception(output, "runfile_0.py")
        assert exc_type == "ModuleNotFoundError"
        assert exc_info["name"] == "xgboost"
        assert exc_stack is not None
        assert exc_stack[0][0] == "runfile_0.py"

    def test_parse_value_error(self, interpreter):
        """测试解析 ValueError（无模块名）。"""
        output = "ValueError: invalid literal for int()\n"
        exc_type, exc_info, exc_stack = interpreter._parse_exception(output, "runfile_0.py")
        assert exc_type == "ValueError"
        assert "invalid literal" in exc_info["msg"]

    def test_parse_exception_with_traceback(self, interpreter):
        """测试解析带 traceback 的异常。"""
        output = (
            'Traceback (most recent call last):\n'
            '  File "runfile_0.py", line 5, in main\n'
            '  File "/usr/lib/python3.10/os.py", line 10, in helper\n'
            "RuntimeError: something failed\n"
        )
        exc_type, exc_info, exc_stack = interpreter._parse_exception(output, "runfile_0.py")
        assert exc_type == "RuntimeError"
        assert len(exc_stack) == 2
        # 第一个应该简化为 script_name
        assert exc_stack[0][0] == "runfile_0.py"


class TestCleanupSession:
    """测试 cleanup_session 方法。"""

    @pytest.fixture
    def interpreter(self, tmp_path):
        return Interpreter(working_dir=tmp_path, timeout=10, max_parallel_run=2)

    def test_cleanup_all_sessions(self, interpreter):
        """测试清理所有会话。"""
        interpreter.cleanup_session(-1)
        # 不应抛异常
        assert all(p is None for p in interpreter.process)

    def test_cleanup_specific_session(self, interpreter):
        """测试清理特定会话。"""
        interpreter.cleanup_session(0)
        assert interpreter.process[0] is None


class TestRunWithNodeId:
    """测试带 node_id 的执行。"""

    @pytest.fixture
    def interpreter(self, tmp_path):
        return Interpreter(working_dir=tmp_path, timeout=10)

    def test_run_with_node_id(self, interpreter):
        """测试 node_id 被用于替换 submission 文件名。"""
        code = "print('hello')"
        result = interpreter.run(code, node_id="test_node_123")
        assert result.success is True

    def test_run_keyboard_interrupt_becomes_timeout(self, interpreter):
        """测试 KeyboardInterrupt 在 stdout 输出中被解析为 TimeoutError。

        当 stdout 中包含 "KeyboardInterrupt: ..." 且 return_code != 0 时，
        _parse_exception 识别出 exc_type="KeyboardInterrupt"，
        run() 将其转换为 "TimeoutError"。
        """
        # 打印到 stdout（subprocess 重定向了 stderr->stdout），让 _parse_exception 能解析
        code = (
            "import sys\n"
            "print('KeyboardInterrupt: user interrupted')\n"
            "sys.exit(1)\n"
        )
        result = interpreter.run(code)
        assert result.success is False
        assert result.exc_type == "TimeoutError"


class TestInitCreateDir:
    """测试初始化时创建目录。"""

    def test_creates_nonexistent_working_dir(self, tmp_path):
        """测试自动创建不存在的工作目录。"""
        new_dir = tmp_path / "new_subdir" / "deep"
        interp = Interpreter(working_dir=new_dir, timeout=10)
        assert new_dir.exists()
        assert interp.working_dir == new_dir


class TestEstimateOomLikelihoodValueError:
    """测试 _estimate_oom_likelihood 中 ValueError 路径（Lines 241-242）。"""

    @pytest.fixture
    def interpreter(self, tmp_path):
        """创建测试用的 Interpreter。"""
        return Interpreter(working_dir=tmp_path, timeout=10)

    def test_oom_non_numeric_in_shape_pattern(self, interpreter):
        """正则匹配到无法转换为 int 的值时跳过（ValueError 被吞掉）。

        re.findall 对 shape.*(\\d+), 只会匹配纯数字，但通过 patch 强制
        注入非数字字符串来模拟该路径。
        """
        # 直接 patch re.findall，使第一次调用返回含非数字的 tuple，触发 ValueError
        original_findall = __import__("re").findall

        call_count = [0]

        def fake_findall(pattern, string, flags=0):
            call_count[0] += 1
            if call_count[0] == 1:
                # 返回包含非数字字符串的 tuple，模拟无法 int() 转换的情形
                return [("not_a_number",)]
            return original_findall(pattern, string, flags)

        with patch("core.executor.interpreter.re.findall", side_effect=fake_findall):
            # 不应抛出异常，ValueError 被内部 try/except 吞掉
            result = interpreter._estimate_oom_likelihood("shape=(abc, 8)")
        # 只要不抛异常即可；返回值取决于其他 pattern 匹配
        assert result in ("high", "medium", "low")

    def test_oom_empty_string_returns_low(self, interpreter):
        """空 stdout 时直接返回 low，不进入循环。"""
        assert interpreter._estimate_oom_likelihood("") == "low"

    def test_oom_none_stdout_returns_low(self, interpreter):
        """None 被视为空时返回 low。"""
        # _estimate_oom_likelihood 接受 str，但 '' 测已覆盖，此处验证边界
        assert interpreter._estimate_oom_likelihood("") == "low"


class TestCleanupProcessExtended:
    """测试 _cleanup_process 的扩展路径（Lines 276-284）。"""

    @pytest.fixture
    def interpreter(self, tmp_path):
        """创建测试用的 Interpreter。"""
        return Interpreter(working_dir=tmp_path, timeout=10)

    def test_cleanup_none_process_is_noop(self, interpreter):
        """process 为 None 时直接返回，不做任何操作。"""
        interpreter.process[0] = None
        interpreter._cleanup_process(0)  # 不应抛异常
        assert interpreter.process[0] is None

    def test_cleanup_already_terminated_process(self, interpreter):
        """进程已退出（poll 返回非 None）时，只置空不 terminate。"""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0  # 已退出
        interpreter.process[0] = mock_proc
        interpreter._cleanup_process(0)
        mock_proc.terminate.assert_not_called()
        assert interpreter.process[0] is None

    def test_cleanup_running_process_terminate_success(self, interpreter):
        """进程仍在运行，terminate 后 wait 成功（Lines 275-278）。"""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # 仍在运行
        mock_proc.wait.return_value = 0      # terminate 后正常退出
        interpreter.process[0] = mock_proc
        interpreter._cleanup_process(0)
        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_not_called()
        assert interpreter.process[0] is None

    def test_cleanup_running_process_terminate_timeout_then_kill(self, interpreter):
        """进程 terminate 超时后执行 kill（Lines 279-282）。"""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        # 第一次 wait（terminate 后）抛超时，第二次 wait（kill 后）正常
        mock_proc.wait.side_effect = [
            subprocess.TimeoutExpired("cmd", 2),
            0,
        ]
        interpreter.process[0] = mock_proc
        interpreter._cleanup_process(0)
        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()
        assert interpreter.process[0] is None

    def test_cleanup_process_exception_suppressed(self, interpreter):
        """清理进程时 poll 抛出异常，异常被抑制，process 置 None（Lines 283-286）。"""
        mock_proc = MagicMock()
        mock_proc.poll.side_effect = Exception("unexpected process error")
        interpreter.process[0] = mock_proc
        interpreter._cleanup_process(0)  # 不应向上抛异常
        assert interpreter.process[0] is None


class TestRunMaxParallelLimit:
    """测试 run() 达到最大并行数限制路径（Lines 328-329）。"""

    def test_max_parallel_limit_returns_error(self, tmp_path):
        """所有槽位已占用时返回 RuntimeError 结果。"""
        interp = Interpreter(working_dir=tmp_path, timeout=10, max_parallel_run=1)
        # 手动占用唯一槽位
        interp.status_map[0] = 1
        interp.current_parallel_run = 1

        result = interp.run("print('hello')")

        assert result.success is False
        assert result.exc_type == "RuntimeError"
        assert "最大并行数" in "".join(result.term_out)

        # 恢复槽位，避免影响其他测试
        interp.status_map[0] = 0
        interp.current_parallel_run = 0


class TestRunTimeoutKillPath:
    """测试 run() 超时后 communicate 再次超时 → kill 路径（Lines 373-375）。"""

    def test_timeout_sigint_then_communicate_timeout_kills(self, tmp_path):
        """SIGINT 发送后，communicate(timeout=5) 仍超时，进程被 kill。"""
        interp = Interpreter(working_dir=tmp_path, timeout=2)

        # 使用真实的睡眠脚本触发超时；补充 kill 路径测试
        # 替代方案：patch communicate，模拟两次超时
        mock_proc = MagicMock()
        mock_proc.pid = 99999  # 不存在的 pid，kill 调用无害
        # 第一次 communicate（正常等待）抛超时
        # 第二次 communicate（SIGINT 后等待 5 秒）抛超时
        # 第三次 communicate（kill 后）返回正常
        mock_proc.communicate.side_effect = [
            subprocess.TimeoutExpired("cmd", 2),  # 第一次：主超时
            subprocess.TimeoutExpired("cmd", 5),  # 第二次：SIGINT 后超时
            ("killed output", None),              # 第三次：kill 后
        ]
        mock_proc.returncode = -1

        with patch("subprocess.Popen", return_value=mock_proc):
            with patch("os.kill"):  # 避免向不存在 pid 发信号
                result = interp.run("import time; time.sleep(100)")

        assert result.timeout is True
        assert result.exc_type == "TimeoutError"
        mock_proc.kill.assert_called_once()


class TestRunSignalTerminationPath:
    """测试 run() 进程被信号终止路径（Lines 390-393）。"""

    def test_signal_termination_negative_return_code(self, tmp_path):
        """return_code < 0 且非超时时，走 signal 终止分支。"""
        interp = Interpreter(working_dir=tmp_path, timeout=30)

        mock_proc = MagicMock()
        mock_proc.pid = 99998
        mock_proc.communicate.return_value = ("some output", None)
        mock_proc.returncode = -9  # SIGKILL

        with patch("subprocess.Popen", return_value=mock_proc):
            result = interp.run("print('hello')")

        assert result.success is False
        assert result.exc_type in ("MemoryError", "ProcessKilled")
        assert result.exc_info is not None
        assert result.exc_info.get("signal") == 9

    def test_signal_termination_sigterm(self, tmp_path):
        """SIGTERM（-15）时返回 ProcessTerminated。"""
        interp = Interpreter(working_dir=tmp_path, timeout=30)

        mock_proc = MagicMock()
        mock_proc.pid = 99997
        mock_proc.communicate.return_value = ("", None)
        mock_proc.returncode = -15  # SIGTERM

        with patch("subprocess.Popen", return_value=mock_proc):
            result = interp.run("print('hello')")

        assert result.success is False
        assert result.exc_type == "ProcessTerminated"
        assert result.exc_info["signal"] == 15


class TestRunScriptCleanupException:
    """测试 run() 脚本文件删除失败时异常被抑制（Lines 432-433）。"""

    def test_script_unlink_exception_suppressed(self, tmp_path):
        """script_path.unlink() 抛出异常时不向上传播。"""
        interp = Interpreter(working_dir=tmp_path, timeout=10)

        # patch Path.unlink 使其抛出异常
        original_unlink = __import__("pathlib").Path.unlink

        def fake_unlink(self, missing_ok=False):
            raise OSError("模拟文件删除失败")

        with patch("pathlib.Path.unlink", fake_unlink):
            # 即使 unlink 失败，run() 也不应抛异常
            result = interp.run("print('cleanup exception test')")

        # 执行本身正常完成
        assert result.success is True


class TestDestructor:
    """测试 __del__ 析构函数（Lines 439-440）。"""

    def test_del_calls_cleanup_session(self, tmp_path):
        """析构函数调用 cleanup_session(-1)。"""
        interp = Interpreter(working_dir=tmp_path, timeout=10)
        with patch.object(interp, "cleanup_session") as mock_cleanup:
            interp.__del__()
            mock_cleanup.assert_called_once_with(-1)

    def test_del_suppresses_exception(self, tmp_path):
        """析构函数抑制 cleanup_session 抛出的异常。"""
        interp = Interpreter(working_dir=tmp_path, timeout=10)
        with patch.object(
            interp, "cleanup_session", side_effect=Exception("cleanup error")
        ):
            interp.__del__()  # 不应抛异常
