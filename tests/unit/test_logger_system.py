"""utils/logger_system.py 的单元测试。"""

import json
import pytest

from utils.logger_system import (
    LoggerSystem,
    init_logger,
    log_msg,
    log_json,
    ensure,
    log_exception,
)


class TestLoggerSystem:
    """测试 LoggerSystem 类。"""

    def test_init_creates_dir(self, tmp_path):
        """测试初始化时创建日志目录。"""
        log_dir = tmp_path / "logs" / "subdir"
        ls = LoggerSystem(log_dir)

        assert log_dir.exists()
        assert ls.text_log_path == log_dir / "system.log"
        assert ls.json_log_path == log_dir / "metrics.json"

    def test_init_loads_existing_json(self, tmp_path):
        """测试初始化时加载已有 JSON 日志。"""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        json_path = log_dir / "metrics.json"
        json_path.write_text('[{"step": 1, "metric": 0.5}]', encoding="utf-8")

        ls = LoggerSystem(log_dir)

        assert len(ls.json_data) == 1
        assert ls.json_data[0]["metric"] == 0.5

    def test_init_handles_corrupted_json(self, tmp_path):
        """测试损坏的 JSON 文件被重置。"""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        (log_dir / "metrics.json").write_text("{invalid json", encoding="utf-8")

        ls = LoggerSystem(log_dir)

        assert ls.json_data == []

    def test_init_handles_empty_json(self, tmp_path):
        """测试空 JSON 文件被重置。"""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        (log_dir / "metrics.json").write_text("", encoding="utf-8")

        ls = LoggerSystem(log_dir)

        assert ls.json_data == []

    def test_text_log_writes_to_file(self, tmp_path):
        """测试文本日志写入文件。"""
        ls = LoggerSystem(tmp_path / "logs")
        ls.text_log("INFO", "Test message")

        content = ls.text_log_path.read_text(encoding="utf-8")
        assert "[INFO] Test message" in content

    def test_text_log_appends(self, tmp_path):
        """测试多次写入追加到文件。"""
        ls = LoggerSystem(tmp_path / "logs")
        ls.text_log("INFO", "First")
        ls.text_log("WARNING", "Second")

        content = ls.text_log_path.read_text(encoding="utf-8")
        assert "First" in content
        assert "Second" in content

    def test_json_log_appends_and_persists(self, tmp_path):
        """测试 JSON 日志追加并持久化。"""
        ls = LoggerSystem(tmp_path / "logs")
        ls.json_log({"step": 1, "metric": 0.5})
        ls.json_log({"step": 2, "metric": 0.8})

        assert len(ls.json_data) == 2

        # 验证文件内容
        persisted = json.loads(ls.json_log_path.read_text(encoding="utf-8"))
        assert len(persisted) == 2
        assert persisted[1]["metric"] == 0.8


class TestGlobalFunctions:
    """测试全局便捷函数。"""

    def test_init_logger(self, tmp_path):
        """测试 init_logger 创建全局实例。"""
        import utils.logger_system as mod

        old_logger = mod.logger
        try:
            result = init_logger(tmp_path / "logs")
            assert isinstance(result, LoggerSystem)
            assert mod.logger is result
        finally:
            mod.logger = old_logger

    def test_log_msg_with_logger(self, tmp_path):
        """测试 log_msg 有 logger 时写入文件。"""
        import utils.logger_system as mod

        old_logger = mod.logger
        try:
            init_logger(tmp_path / "logs")
            log_msg("INFO", "test message via global")

            content = mod.logger.text_log_path.read_text(encoding="utf-8")
            assert "test message via global" in content
        finally:
            mod.logger = old_logger

    def test_log_msg_without_logger(self, capsys):
        """测试 log_msg 无 logger 时回退到 print。"""
        import utils.logger_system as mod

        old_logger = mod.logger
        try:
            mod.logger = None
            log_msg("WARNING", "fallback message")

            captured = capsys.readouterr()
            assert "[WARNING] fallback message" in captured.out
        finally:
            mod.logger = old_logger

    def test_log_json_with_logger(self, tmp_path):
        """测试 log_json 有 logger 时写入文件。"""
        import utils.logger_system as mod

        old_logger = mod.logger
        try:
            init_logger(tmp_path / "logs")
            log_json({"key": "value"})

            assert len(mod.logger.json_data) >= 1
        finally:
            mod.logger = old_logger

    def test_log_json_without_logger(self, capsys):
        """测试 log_json 无 logger 时回退到 print。"""
        import utils.logger_system as mod

        old_logger = mod.logger
        try:
            mod.logger = None
            log_json({"key": "value"})

            captured = capsys.readouterr()
            assert "[JSON]" in captured.out
            assert "value" in captured.out
        finally:
            mod.logger = old_logger


class TestEnsure:
    """测试 ensure 断言工具。"""

    def test_ensure_passes(self):
        """测试条件为 True 时通过。"""
        ensure(True, "should not raise")

    def test_ensure_fails(self):
        """测试条件为 False 时抛出 AssertionError。"""
        with pytest.raises(AssertionError, match="条件失败"):
            ensure(False, "条件失败")


class TestLogException:
    """测试 log_exception 函数。"""

    def test_log_exception_with_context(self, tmp_path):
        """测试带上下文的异常记录。"""
        import utils.logger_system as mod

        old_logger = mod.logger
        try:
            init_logger(tmp_path / "logs")

            try:
                raise ValueError("test error")
            except Exception as e:
                log_exception(e, "执行测试时")

            content = mod.logger.text_log_path.read_text(encoding="utf-8")
            assert "执行测试时: test error" in content
        finally:
            mod.logger = old_logger

    def test_log_exception_without_context(self, tmp_path):
        """测试不带上下文的异常记录。"""
        import utils.logger_system as mod

        old_logger = mod.logger
        try:
            init_logger(tmp_path / "logs")

            try:
                raise RuntimeError("runtime fail")
            except Exception as e:
                log_exception(e)

            content = mod.logger.text_log_path.read_text(encoding="utf-8")
            assert "runtime fail" in content
        finally:
            mod.logger = old_logger
