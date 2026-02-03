"""
core/executor/interpreter.py 的单元测试。
"""

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
        assert interpreter._estimate_oom_likelihood("Train shape: (55413942, 8)") == "high"
        assert interpreter._estimate_oom_likelihood("Loading 10000000 rows") == "high"

    def test_estimate_oom_likelihood_medium(self, interpreter):
        """测试 OOM 可能性估算（中）。"""
        # 超过 100 万行但不到 1000 万
        assert interpreter._estimate_oom_likelihood("Train shape: (1500000, 8)") == "medium"
        # 训练相关操作
        assert interpreter._estimate_oom_likelihood("Fold 1/5\nTraining...") == "medium"
        assert interpreter._estimate_oom_likelihood("Loading data...") == "medium"

    def test_estimate_oom_likelihood_low(self, interpreter):
        """测试 OOM 可能性估算（低）。"""
        assert interpreter._estimate_oom_likelihood("Hello World") == "low"
        assert interpreter._estimate_oom_likelihood("Train shape: (1000, 8)") == "low"
        assert interpreter._estimate_oom_likelihood("") == "low"
