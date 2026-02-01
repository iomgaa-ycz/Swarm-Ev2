"""文件工具单元测试。"""

import pytest
from pathlib import Path
import stat
import zipfile


from utils.file_utils import copytree, extract_archives, clean_up_dataset


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


class TestExtractArchives:
    """extract_archives 函数测试类。"""

    def test_extract_single_zip(self, tmp_path: Path) -> None:
        """测试解压单个 zip 文件。"""
        # 创建测试 zip 文件
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        zip_path = data_dir / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("file1.txt", "content1")
            zf.writestr("subdir/file2.txt", "content2")

        # 解压
        count = extract_archives(data_dir)

        # 验证
        assert count == 1
        assert not zip_path.exists()  # 原始 zip 已删除
        assert (data_dir / "test" / "file1.txt").read_text() == "content1"
        assert (data_dir / "test" / "subdir" / "file2.txt").read_text() == "content2"

    def test_extract_multiple_zips(self, tmp_path: Path) -> None:
        """测试解压多个 zip 文件。"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # 创建多个 zip 文件
        for i in range(3):
            zip_path = data_dir / f"archive{i}.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr(f"file{i}.txt", f"content{i}")

        # 解压
        count = extract_archives(data_dir)

        # 验证
        assert count == 3
        for i in range(3):
            assert (
                data_dir / f"archive{i}" / f"file{i}.txt"
            ).read_text() == f"content{i}"

    def test_extract_nested_zip_directory(self, tmp_path: Path) -> None:
        """测试解压嵌套目录（zip 中只有一个同名目录）。"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # 创建嵌套结构的 zip: train.zip -> train/ -> data.csv
        zip_path = data_dir / "train.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("train/data.csv", "col1,col2\n1,2")

        # 解压
        count = extract_archives(data_dir)

        # 验证：嵌套目录应该被展开
        assert count == 1
        assert (data_dir / "train" / "data.csv").read_text() == "col1,col2\n1,2"

    def test_skip_existing_directory(self, tmp_path: Path) -> None:
        """测试跳过已存在的目录。"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # 创建同名目录
        existing_dir = data_dir / "test"
        existing_dir.mkdir()
        (existing_dir / "existing.txt").write_text("existing")

        # 创建 zip 文件
        zip_path = data_dir / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("new.txt", "new content")

        # 解压
        count = extract_archives(data_dir)

        # 验证：跳过解压，保留原有内容
        assert count == 0
        assert (existing_dir / "existing.txt").read_text() == "existing"
        assert not (existing_dir / "new.txt").exists()

    def test_extract_empty_directory(self, tmp_path: Path) -> None:
        """测试空目录不报错。"""
        data_dir = tmp_path / "empty"
        data_dir.mkdir()

        count = extract_archives(data_dir)
        assert count == 0


class TestCleanUpDataset:
    """clean_up_dataset 函数测试类。"""

    def test_clean_macosx_directory(self, tmp_path: Path) -> None:
        """测试清理 __MACOSX 目录。"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # 创建 __MACOSX 目录
        macosx_dir = data_dir / "__MACOSX"
        macosx_dir.mkdir()
        (macosx_dir / "._file.txt").write_text("metadata")

        # 创建正常文件
        (data_dir / "file.txt").write_text("content")

        # 清理
        count = clean_up_dataset(data_dir)

        # 验证
        assert count >= 1
        assert not macosx_dir.exists()
        assert (data_dir / "file.txt").read_text() == "content"

    def test_clean_ds_store(self, tmp_path: Path) -> None:
        """测试清理 .DS_Store 文件。"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # 创建 .DS_Store 文件
        (data_dir / ".DS_Store").write_text("binary data")

        # 创建子目录中的 .DS_Store
        subdir = data_dir / "subdir"
        subdir.mkdir()
        (subdir / ".DS_Store").write_text("binary data")

        # 清理
        count = clean_up_dataset(data_dir)

        # 验证
        assert count >= 2
        assert not (data_dir / ".DS_Store").exists()
        assert not (subdir / ".DS_Store").exists()

    def test_clean_mixed_junk(self, tmp_path: Path) -> None:
        """测试同时清理多种垃圾文件。"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # 创建各种垃圾
        (data_dir / "__MACOSX").mkdir()
        (data_dir / "__MACOSX" / "._file").write_text("meta")
        (data_dir / ".DS_Store").write_text("data")

        # 创建正常内容
        (data_dir / "train.csv").write_text("a,b,c")

        # 清理
        count = clean_up_dataset(data_dir)

        # 验证
        assert count >= 2
        assert not (data_dir / "__MACOSX").exists()
        assert not (data_dir / ".DS_Store").exists()
        assert (data_dir / "train.csv").read_text() == "a,b,c"

    def test_clean_empty_directory(self, tmp_path: Path) -> None:
        """测试空目录不报错。"""
        data_dir = tmp_path / "empty"
        data_dir.mkdir()

        count = clean_up_dataset(data_dir)
        assert count == 0
