"""输入文件保护机制的单元测试（P0-3 修复验证）。

测试范围:
    - WorkspaceManager.protect_input_files(): 将 input/ 下文件设为只读
    - 跳过 symlink 和目录
    - 保护后尝试写入应失败
"""

import pytest
import stat
import os
from pathlib import Path
from unittest.mock import Mock

from core.executor.workspace import WorkspaceManager


class TestProtectInputFiles:
    """测试输入文件保护功能。"""

    def test_protect_normal_files(self, tmp_path):
        """input/ 下有 csv 文件 → chmod 变为 444。"""
        config = Mock()
        config.project.workspace_dir = tmp_path

        workspace = WorkspaceManager(config)
        workspace.workspace_dir = tmp_path

        # 创建测试文件
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        test_file = input_dir / "train.csv"
        test_file.write_text("id,label\n1,0\n2,1\n")

        # 初始权限应该是可写的（默认 0o644 或 0o666）
        initial_mode = test_file.stat().st_mode & 0o777
        assert initial_mode != 0o444

        # 执行保护
        workspace.protect_input_files()

        # 验证权限变为 0o444
        protected_mode = test_file.stat().st_mode & 0o777
        assert protected_mode == 0o444

    def test_protect_skip_symlinks(self, tmp_path):
        """input/ 下有 symlink → 不修改权限。"""
        config = Mock()
        config.project.workspace_dir = tmp_path

        workspace = WorkspaceManager(config)
        workspace.workspace_dir = tmp_path

        # 创建测试文件和 symlink
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        # 源文件
        source_file = tmp_path / "external.csv"
        source_file.write_text("data")
        original_mode = source_file.stat().st_mode & 0o777

        # 创建 symlink
        symlink = input_dir / "link.csv"
        symlink.symlink_to(source_file)

        # 执行保护
        workspace.protect_input_files()

        # symlink 指向的源文件权限应该不变
        assert (source_file.stat().st_mode & 0o777) == original_mode
        # symlink 本身也不应该被修改
        assert symlink.is_symlink()

    def test_protect_empty_dir(self, tmp_path):
        """input/ 为空 → 无操作，不报错。"""
        config = Mock()
        config.project.workspace_dir = tmp_path

        workspace = WorkspaceManager(config)
        workspace.workspace_dir = tmp_path

        input_dir = tmp_path / "input"
        input_dir.mkdir()

        # 执行保护（空目录）
        workspace.protect_input_files()  # 应该不抛异常

    def test_write_after_protect(self, tmp_path):
        """保护后尝试写入 → PermissionError。"""
        config = Mock()
        config.project.workspace_dir = tmp_path

        workspace = WorkspaceManager(config)
        workspace.workspace_dir = tmp_path

        # 创建测试文件
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        test_file = input_dir / "data.csv"
        test_file.write_text("original content")

        # 执行保护
        workspace.protect_input_files()

        # 尝试写入应该失败
        with pytest.raises(PermissionError):
            test_file.write_text("modified content")

    def test_protect_skip_directories(self, tmp_path):
        """input/ 下有子目录 → 跳过目录本身。"""
        config = Mock()
        config.project.workspace_dir = tmp_path

        workspace = WorkspaceManager(config)
        workspace.workspace_dir = tmp_path

        # 创建子目录
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        sub_dir = input_dir / "subdir"
        sub_dir.mkdir()

        original_mode = sub_dir.stat().st_mode & 0o777

        # 执行保护
        workspace.protect_input_files()

        # 目录权限应该不变（需要 execute bit 才能进入）
        assert (sub_dir.stat().st_mode & 0o777) == original_mode

    def test_protect_nested_files(self, tmp_path):
        """input/ 下嵌套子目录中的文件 → 也应被保护。"""
        config = Mock()
        config.project.workspace_dir = tmp_path

        workspace = WorkspaceManager(config)
        workspace.workspace_dir = tmp_path

        # 创建嵌套文件
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        sub_dir = input_dir / "images"
        sub_dir.mkdir()
        nested_file = sub_dir / "image.jpg"
        nested_file.write_text("image data")

        # 执行保护
        workspace.protect_input_files()

        # 嵌套文件应该被保护
        assert (nested_file.stat().st_mode & 0o777) == 0o444


class TestProtectIntegrationWithPrepare:
    """测试 protect_input_files() 与 prepare_workspace() 的集成。"""

    def test_prepare_workspace_calls_protect(self, tmp_path):
        """prepare_workspace() 应该在预处理后调用 protect_input_files()。"""
        config = Mock()
        config.project.workspace_dir = tmp_path
        config.data.data_dir = tmp_path / "source"
        config.data.input_dir = tmp_path / "source" / "input"
        config.data.preprocess_data = False  # 禁用预处理简化测试

        # 创建源数据
        source_input_dir = tmp_path / "source" / "input"
        source_input_dir.mkdir(parents=True)
        test_file = source_input_dir / "data.csv"
        test_file.write_text("test data")

        workspace = WorkspaceManager(config)

        # 执行 prepare_workspace
        workspace.prepare_workspace(source_dir=source_input_dir)

        # 验证 input/ 下的文件被保护
        protected_file = tmp_path / "input" / "data.csv"
        assert protected_file.exists()
        assert (protected_file.stat().st_mode & 0o777) == 0o444
