"""
core/executor/workspace.py 的单元测试。
"""

import pytest
import zipfile
from pathlib import Path
from unittest.mock import Mock
from core.executor.workspace import WorkspaceManager


@pytest.fixture
def mock_config(tmp_path):
    """创建 mock 配置对象。"""
    config = Mock()
    config.project.workspace_dir = tmp_path / "workspace"
    config.data.input_dir = tmp_path / "data"
    return config


class TestWorkspaceManager:
    """测试 WorkspaceManager 类。"""

    def test_init(self, mock_config):
        """测试初始化。"""
        manager = WorkspaceManager(mock_config)
        assert manager.workspace_dir == Path(mock_config.project.workspace_dir)

    def test_setup_creates_directories(self, mock_config):
        """测试 setup 创建目录结构。"""
        manager = WorkspaceManager(mock_config)
        manager.setup()

        # 验证所有目录已创建
        assert (manager.workspace_dir / "input").exists()
        assert (manager.workspace_dir / "working").exists()
        assert (manager.workspace_dir / "submission").exists()
        assert (manager.workspace_dir / "archives").exists()
        assert (manager.workspace_dir / "best_solution").exists()

    def test_rewrite_submission_path(self, mock_config):
        """测试路径重写。"""
        manager = WorkspaceManager(mock_config)

        code = 'df.to_csv("./submission/submission.csv")'
        rewritten = manager.rewrite_submission_path(code, "abc123")
        assert "submission_abc123.csv" in rewritten

    def test_rewrite_submission_path_various_formats(self, mock_config):
        """测试各种路径格式的重写。"""
        manager = WorkspaceManager(mock_config)

        test_cases = [
            ('save("./submission/submission.csv")', "submission_test.csv"),
            ("save('./submission/submission.csv')", "submission_test.csv"),
            ('save("submission.csv")', "submission_test.csv"),
        ]

        for original, expected in test_cases:
            rewritten = manager.rewrite_submission_path(original, "test")
            assert expected in rewritten

    def test_link_input_data(self, mock_config, tmp_path):
        """测试输入数据链接。"""
        # 创建数据源目录
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "train.csv").write_text("a,b\n1,2")

        manager = WorkspaceManager(mock_config)
        manager.setup()
        manager.link_input_data(data_dir)

        # 验证链接已创建
        input_link = manager.workspace_dir / "input"
        assert input_link.exists()
        assert (input_link / "train.csv").exists()

    def test_archive_node_files_with_submission(self, mock_config):
        """测试打包包含 submission 的节点。"""
        manager = WorkspaceManager(mock_config)
        manager.setup()

        # 创建 submission 文件
        submission_path = (
            manager.workspace_dir / "submission" / "submission_test123.csv"
        )
        submission_path.write_text("prediction\n1\n2\n3")

        code = "print('Hello')"
        zip_path = manager.archive_node_files("test123", code)

        # 验证 zip 文件
        assert zip_path is not None
        assert zip_path.exists()

        with zipfile.ZipFile(zip_path, "r") as zf:
            assert "solution.py" in zf.namelist()
            assert "submission.csv" in zf.namelist()
            assert zf.read("solution.py").decode() == code

    def test_archive_node_files_without_submission(self, mock_config):
        """测试打包只有代码的节点。"""
        manager = WorkspaceManager(mock_config)
        manager.setup()

        code = "x = 1"
        zip_path = manager.archive_node_files("test456", code)

        # 验证 zip 文件
        assert zip_path is not None
        assert zip_path.exists()

        with zipfile.ZipFile(zip_path, "r") as zf:
            assert "solution.py" in zf.namelist()
            assert "submission.csv" not in zf.namelist()

    def test_cleanup_submission(self, mock_config):
        """测试清空 submission 目录。"""
        manager = WorkspaceManager(mock_config)
        manager.setup()

        # 创建文件
        (manager.workspace_dir / "submission" / "file.csv").write_text("data")

        # 清空
        manager.cleanup_submission()

        # 验证目录为空
        submission_dir = manager.workspace_dir / "submission"
        assert submission_dir.exists()
        assert len(list(submission_dir.iterdir())) == 0

    def test_cleanup_working(self, mock_config):
        """测试清空 working 目录。"""
        manager = WorkspaceManager(mock_config)
        manager.setup()

        # 创建文件
        (manager.workspace_dir / "working" / "temp.txt").write_text("temp")

        # 清空
        manager.cleanup_working()

        # 验证目录为空
        working_dir = manager.workspace_dir / "working"
        assert working_dir.exists()
        assert len(list(working_dir.iterdir())) == 0

    def test_preprocess_input_extracts_zip(self, mock_config, tmp_path):
        """测试 preprocess_input 解压 zip 文件。"""
        manager = WorkspaceManager(mock_config)
        manager.setup()

        # 创建 input 目录和 zip 文件
        input_dir = manager.workspace_dir / "input"
        zip_path = input_dir / "data.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("train.csv", "a,b\n1,2")

        # 预处理
        manager.preprocess_input()

        # 验证
        assert not zip_path.exists()
        assert (input_dir / "data" / "train.csv").read_text() == "a,b\n1,2"

    def test_preprocess_input_cleans_macosx(self, mock_config, tmp_path):
        """测试 preprocess_input 清理 macOS 垃圾文件。"""
        manager = WorkspaceManager(mock_config)
        manager.setup()

        # 创建垃圾文件
        input_dir = manager.workspace_dir / "input"
        (input_dir / "__MACOSX").mkdir()
        (input_dir / "__MACOSX" / "._file").write_text("meta")
        (input_dir / ".DS_Store").write_text("data")
        (input_dir / "train.csv").write_text("a,b")

        # 预处理
        manager.preprocess_input()

        # 验证
        assert not (input_dir / "__MACOSX").exists()
        assert not (input_dir / ".DS_Store").exists()
        assert (input_dir / "train.csv").read_text() == "a,b"

    def test_preprocess_input_converts_symlink(self, mock_config, tmp_path):
        """测试 preprocess_input 将符号链接转换为复制。"""
        # 创建数据源
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "file.txt").write_text("content")

        manager = WorkspaceManager(mock_config)
        manager.setup()

        # 创建符号链接
        input_link = manager.workspace_dir / "input"
        input_link.rmdir()  # 删除 setup 创建的空目录
        input_link.symlink_to(data_dir)
        assert input_link.is_symlink()

        # 预处理
        manager.preprocess_input()

        # 验证：符号链接已转换为实际目录
        assert not input_link.is_symlink()
        assert input_link.is_dir()
        assert (input_link / "file.txt").read_text() == "content"

    def test_prepare_workspace_full_flow(self, mock_config, tmp_path):
        """测试 prepare_workspace 一站式流程。"""
        # 创建数据源
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # 创建 zip 文件
        zip_path = data_dir / "train.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("data.csv", "a,b\n1,2")

        # 创建垃圾文件
        (data_dir / ".DS_Store").write_text("junk")

        # 配置启用预处理
        mock_config.data.preprocess_data = True
        mock_config.data.data_dir = data_dir

        manager = WorkspaceManager(mock_config)
        manager.prepare_workspace(data_dir)

        # 验证目录结构
        assert (manager.workspace_dir / "input").exists()
        assert (manager.workspace_dir / "working").exists()
        assert (manager.workspace_dir / "submission").exists()

        # 验证解压和清理
        input_dir = manager.workspace_dir / "input"
        assert (input_dir / "train" / "data.csv").read_text() == "a,b\n1,2"
        assert not (input_dir / ".DS_Store").exists()

    def test_prepare_workspace_preprocess_disabled(self, mock_config, tmp_path):
        """测试禁用预处理时不执行解压。"""
        # 创建数据源
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # 创建 zip 文件
        zip_path = data_dir / "train.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("data.csv", "a,b\n1,2")

        # 禁用预处理
        mock_config.data.preprocess_data = False
        mock_config.data.data_dir = data_dir

        manager = WorkspaceManager(mock_config)
        manager.prepare_workspace(data_dir)

        # 验证：zip 文件仍然存在（未解压，因为是符号链接模式）
        assert zip_path.exists()
