"""文件工具单元测试。"""

import pytest
from pathlib import Path
import os
import stat


from utils.file_utils import copytree


class TestCopytree:
    """copytree 函数测试类。"""

    def test_copytree_with_symlink(self, tmp_path: Path) -> None:
        """测试符号链接模式。"""
        # 创建源目录和文件
        src_dir = tmp_path / "source"
        src_dir.mkdir()
        (src_dir / "file1.txt").write_text("content1")
        (src_dir / "file2.txt").write_text("content2")

        dst_dir = tmp_path / "destination_symlink"

        # 使用符号链接模式
        copytree(src=src_dir, dst=dst_dir, use_symlinks=True)

        # 验证目标是符号链接
        assert dst_dir.exists()
        assert dst_dir.is_symlink()
        assert dst_dir.resolve() == src_dir.resolve()

    def test_copytree_with_copy(self, tmp_path: Path) -> None:
        """测试复制模式。"""
        # 创建源目录和文件
        src_dir = tmp_path / "source"
        src_dir.mkdir()
        (src_dir / "file1.txt").write_text("content1")
        (src_dir / "subdir").mkdir()
        (src_dir / "subdir" / "file2.txt").write_text("content2")

        dst_dir = tmp_path / "destination_copy"

        # 使用复制模式
        copytree(src=src_dir, dst=dst_dir, use_symlinks=False)

        # 验证目标是目录（非符号链接）
        assert dst_dir.exists()
        assert dst_dir.is_dir()
        assert not dst_dir.is_symlink()

        # 验证文件内容一致
        assert (dst_dir / "file1.txt").read_text() == "content1"
        assert (dst_dir / "subdir" / "file2.txt").read_text() == "content2"

        # 验证目录是只读的
        dst_stat = dst_dir.stat()
        # 检查没有写权限（owner, group, others）
        assert not (dst_stat.st_mode & stat.S_IWUSR)
        assert not (dst_stat.st_mode & stat.S_IWGRP)
        assert not (dst_stat.st_mode & stat.S_IWOTH)

    def test_copytree_src_not_exist(self, tmp_path: Path) -> None:
        """测试源目录不存在时抛出异常。"""
        src_dir = tmp_path / "nonexistent"
        dst_dir = tmp_path / "destination"

        # 验证抛出 FileNotFoundError
        with pytest.raises(FileNotFoundError, match="源目录不存在"):
            copytree(src=src_dir, dst=dst_dir, use_symlinks=False)

    def test_copytree_src_not_directory(self, tmp_path: Path) -> None:
        """测试源路径不是目录时抛出异常。"""
        src_file = tmp_path / "source.txt"
        src_file.write_text("content")
        dst_dir = tmp_path / "destination"

        # 验证抛出 NotADirectoryError
        with pytest.raises(NotADirectoryError, match="源路径不是目录"):
            copytree(src=src_file, dst=dst_dir, use_symlinks=False)

    def test_copytree_readonly_protection(self, tmp_path: Path) -> None:
        """测试复制后的目录具有只读保护。"""
        # 创建源目录
        src_dir = tmp_path / "source"
        src_dir.mkdir()
        (src_dir / "file1.txt").write_text("content1")

        dst_dir = tmp_path / "destination"

        # 复制目录
        copytree(src=src_dir, dst=dst_dir, use_symlinks=False)
        assert (dst_dir / "file1.txt").exists()

        # 验证目录和文件都是只读的
        dst_stat = dst_dir.stat()
        assert not (dst_stat.st_mode & stat.S_IWUSR), "目录应该是只读的"

        file_stat = (dst_dir / "file1.txt").stat()
        assert not (file_stat.st_mode & stat.S_IWUSR), "文件应该是只读的"

    def test_copytree_replaces_existing_symlink(self, tmp_path: Path) -> None:
        """测试替换已存在的符号链接。"""
        # 创建第一个源目录
        src_dir1 = tmp_path / "source1"
        src_dir1.mkdir()
        (src_dir1 / "file1.txt").write_text("content1")

        # 创建第二个源目录
        src_dir2 = tmp_path / "source2"
        src_dir2.mkdir()
        (src_dir2 / "file2.txt").write_text("content2")

        dst_dir = tmp_path / "destination"

        # 第一次创建符号链接
        copytree(src=src_dir1, dst=dst_dir, use_symlinks=True)
        assert dst_dir.resolve() == src_dir1.resolve()

        # 第二次创建符号链接（指向不同源）
        copytree(src=src_dir2, dst=dst_dir, use_symlinks=True)
        assert dst_dir.resolve() == src_dir2.resolve()
