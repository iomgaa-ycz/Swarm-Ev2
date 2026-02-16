"""自适应超时功能测试。"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from core.orchestrator import Orchestrator
from utils.config import Config, ExecutionConfig, ProjectConfig


class TestAdaptiveTimeout:
    """自适应超时测试。"""

    @pytest.fixture
    def mock_config(self, tmp_path):
        """创建 mock 配置。"""
        config = Mock(spec=Config)
        config.execution = ExecutionConfig(
            timeout=3600,
            timeout_max=7200,
            adaptive_timeout=True,
            agent_file_name="runfile.py",
            format_tb_ipython=False,
        )
        config.project = Mock(spec=ProjectConfig)
        config.project.workspace_dir = tmp_path
        return config

    def test_estimate_timeout_disabled(self, mock_config):
        """测试禁用自适应超时。"""
        mock_config.execution.adaptive_timeout = False

        with patch.object(Orchestrator, "__init__", lambda x, **kwargs: None):
            orchestrator = Orchestrator()
            orchestrator.config = mock_config
            timeout = orchestrator._estimate_timeout()

        assert timeout == 3600

    def test_estimate_timeout_small_dataset(self, mock_config, tmp_path):
        """测试小数据集（<100MB）。"""
        # 创建小文件
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "test.csv").write_text("a,b,c\n1,2,3\n" * 100)  # <1MB

        with patch.object(Orchestrator, "__init__", lambda x, **kwargs: None):
            orchestrator = Orchestrator()
            orchestrator.config = mock_config
            timeout = orchestrator._estimate_timeout()

        assert timeout == 3600  # 1.0x

    def test_estimate_timeout_medium_dataset(self, mock_config, tmp_path):
        """测试中等数据集（100~500MB）。"""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        # 模拟 200MB 数据
        with patch("pathlib.Path.rglob") as mock_rglob:
            mock_file = Mock()
            mock_file.is_file.return_value = True
            mock_file.stat.return_value = Mock(st_size=200 * 1024 * 1024)
            mock_rglob.return_value = [mock_file]

            with patch.object(Orchestrator, "__init__", lambda x, **kwargs: None):
                orchestrator = Orchestrator()
                orchestrator.config = mock_config
                timeout = orchestrator._estimate_timeout()

        assert timeout == 5400  # 1.5x

    def test_estimate_timeout_large_dataset(self, mock_config, tmp_path):
        """测试大数据集（>500MB）。"""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        # 模拟 800MB 数据
        with patch("pathlib.Path.rglob") as mock_rglob:
            mock_file = Mock()
            mock_file.is_file.return_value = True
            mock_file.stat.return_value = Mock(st_size=800 * 1024 * 1024)
            mock_rglob.return_value = [mock_file]

            with patch.object(Orchestrator, "__init__", lambda x, **kwargs: None):
                orchestrator = Orchestrator()
                orchestrator.config = mock_config
                timeout = orchestrator._estimate_timeout()

        assert timeout == 7200  # 2.0x, capped at max

    def test_estimate_timeout_no_input_dir(self, mock_config, tmp_path):
        """测试 input 目录不存在。"""
        with patch.object(Orchestrator, "__init__", lambda x, **kwargs: None):
            orchestrator = Orchestrator()
            orchestrator.config = mock_config
            timeout = orchestrator._estimate_timeout()

        assert timeout == 3600  # fallback to base
