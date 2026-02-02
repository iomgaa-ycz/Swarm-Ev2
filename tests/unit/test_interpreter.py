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
