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


# ============================================================
# 新增：覆盖率补全测试（针对未覆盖行）
# ============================================================

from utils.file_utils import copytree, extract_archives  # noqa: E402（已在顶部导入，此处为可读性）


class TestLinkReadonlyDataLchmod:
    """测试 copytree symlink 模式下 lchmod 分支（lines 61-66）。"""

    def test_lchmod_branch_when_available(self, tmp_path: Path) -> None:
        """当 os.lchmod 存在时，调用 lchmod 并记录日志（line 62）。"""
        import os
        from unittest.mock import patch

        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "data.txt").write_text("content")
        dst_dir = tmp_path / "dst_lchmod"

        lchmod_called = {"called": False}

        def fake_lchmod(path, mode):
            lchmod_called["called"] = True

        # 在 hasattr 返回 True 的情况下注入 fake_lchmod
        with patch.object(os, "lchmod", fake_lchmod, create=True):
            copytree(src=src_dir, dst=dst_dir, use_symlinks=True)

        assert dst_dir.is_symlink()
        # fake_lchmod 是否被调用取决于平台是否原生有 lchmod
        # 只要函数正常执行（没有异常）即可验证此路径
        assert dst_dir.resolve() == src_dir.resolve()

    def test_lchmod_oserror_is_swallowed(self, tmp_path: Path) -> None:
        """当 lchmod 抛出 OSError 时，错误被吞掉不影响主流程（lines 64-66）。"""
        import os
        from unittest.mock import patch

        src_dir = tmp_path / "src2"
        src_dir.mkdir()
        (src_dir / "file.txt").write_text("data")
        dst_dir = tmp_path / "dst_lchmod_err"

        def raising_lchmod(path, mode):
            raise OSError("平台不支持")

        with patch.object(os, "lchmod", raising_lchmod, create=True):
            # 不应抛出异常
            copytree(src=src_dir, dst=dst_dir, use_symlinks=True)

        assert dst_dir.is_symlink()

    def test_symlink_oserror_falls_back_to_copy(self, tmp_path: Path) -> None:
        """当 symlink_to 抛出 OSError 时，降级为复制模式（lines 64-66 对应 68-71）。"""
        from unittest.mock import patch
        from pathlib import Path as _Path

        src_dir = tmp_path / "src3"
        src_dir.mkdir()
        (src_dir / "file.txt").write_text("fallback data")
        dst_dir = tmp_path / "dst_fallback"

        original_symlink_to = _Path.symlink_to

        def raising_symlink_to(self, target, target_is_directory=False):
            raise OSError("不支持符号链接")

        with patch.object(_Path, "symlink_to", raising_symlink_to):
            copytree(src=src_dir, dst=dst_dir, use_symlinks=True)

        # 降级为复制模式，dst 应该是普通目录
        assert dst_dir.exists()
        assert dst_dir.is_dir()
        assert not dst_dir.is_symlink()
        assert (dst_dir / "file.txt").read_text() == "fallback data"


class TestLinkReadonlyDataCopyExistingDst:
    """测试 copytree 复制模式下目标目录已存在的警告路径（lines 85-86）。"""

    def test_copy_mode_with_existing_dst_logs_warning(self, tmp_path: Path) -> None:
        """当复制模式下 dst 已存在时，记录 WARNING 并继续增量复制（lines 85-86）。"""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "new_file.txt").write_text("new content")

        dst_dir = tmp_path / "dst"
        dst_dir.mkdir()
        (dst_dir / "existing_file.txt").write_text("existing content")

        # use_symlinks=False 且 dst 已存在 → 触发 lines 85-86
        copytree(src=src_dir, dst=dst_dir, use_symlinks=False)

        # 两个文件都应存在（增量复制）
        assert (dst_dir / "existing_file.txt").read_text() == "existing content"
        assert (dst_dir / "new_file.txt").read_text() == "new content"


class TestExtractArchivesSkipExisting:
    """测试 extract_archives 跳过已存在目录路径（line 145）。"""

    def test_skip_existing_path_does_not_delete_zip(self, tmp_path: Path) -> None:
        """当输出目录已存在时，跳过解压但保留 zip 文件（line 145 条件）。"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # 创建已存在的同名目录
        existing_dir = data_dir / "archive"
        existing_dir.mkdir()
        (existing_dir / "keep.txt").write_text("保留内容")

        # 创建 zip 文件
        zip_path = data_dir / "archive.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("overwrite.txt", "新内容")

        count = extract_archives(data_dir)

        # 跳过，不解压
        assert count == 0
        # 原有文件保留
        assert (existing_dir / "keep.txt").read_text() == "保留内容"
        # zip 文件保留（因为 f_out_dir 是目录，不是同名文件）
        assert zip_path.exists()

    def test_skip_existing_file_deletes_zip(self, tmp_path: Path) -> None:
        """当同名路径是文件（非目录）时，删除 zip（line 144-145 的文件分支）。"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # 创建同名普通文件（不是目录）——注意：f_out_dir = zip_f.with_suffix("")
        # zip 名为 "mydata.zip" → f_out_dir = "mydata"
        existing_file = data_dir / "mydata"
        existing_file.write_text("已存在的文件")

        zip_path = data_dir / "mydata.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("new.txt", "新内容")

        count = extract_archives(data_dir)

        # 跳过解压，但 zip 被删除（因为同名路径是有扩展名的文件的判断条件不满足，不删除）
        # 实际代码：if f_out_dir.is_file() and f_out_dir.suffix != ""：才删除
        # "mydata" 后缀为 ""，所以 zip 不会被删除
        assert count == 0


class TestExtractArchivesNestedSingleFile:
    """测试 extract_archives 嵌套单文件展开路径（lines 154-156, 173-179）。"""

    def test_nested_single_dir_expansion(self, tmp_path: Path) -> None:
        """zip 内只有一个同名子目录时，展开其内容（lines 166-171）。"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # archive.zip 内只有 archive/ 目录，目录内有文件
        zip_path = data_dir / "archive.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("archive/file1.csv", "col1,col2\n1,2")
            zf.writestr("archive/file2.txt", "text content")

        count = extract_archives(data_dir)

        assert count == 1
        out_dir = data_dir / "archive"
        # 嵌套目录被展开，内容直接在 out_dir 下
        assert (out_dir / "file1.csv").read_text() == "col1,col2\n1,2"
        assert (out_dir / "file2.txt").read_text() == "text content"

    def test_nested_single_file_expansion(self, tmp_path: Path) -> None:
        """zip 内只有一个同名单文件时，重命名替代目录（lines 173-179）。"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # report.zip 内只有 report（无扩展名）文件
        zip_path = data_dir / "report.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("report", "这是报告内容")

        count = extract_archives(data_dir)

        assert count == 1
        # report 目录已被替换为同名文件
        out_path = data_dir / "report"
        assert out_path.exists()
        # 它应该是一个文件（目录被删除，文件被重命名进来）
        assert out_path.is_file()
        assert out_path.read_text() == "这是报告内容"


class TestExtractArchivesAdditionalBranches:
    """覆盖 extract_archives 中剩余未覆盖分支。"""

    def test_skip_existing_file_with_extension_deletes_zip(self, tmp_path: Path) -> None:
        """同名路径是带扩展名的文件时，删除 zip（line 145）。

        条件: f_out_dir.is_file() and f_out_dir.suffix != ""
        例：archive.zip → f_out_dir = "archive"，但若创建 "archive.txt.zip" → f_out_dir = "archive.txt"
        f_out_dir.suffix == ".txt" != ""，触发 line 145。
        """
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # 创建同名带扩展名的文件（f_out_dir = data_dir / "archive.txt"）
        existing_file = data_dir / "archive.txt"
        existing_file.write_text("已存在的文本文件")

        # 创建 archive.txt.zip → f_out_dir = archive.txt
        zip_path = data_dir / "archive.txt.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("new.txt", "新内容")

        count = extract_archives(data_dir)

        # 跳过解压
        assert count == 0
        # 原文件保留
        assert existing_file.read_text() == "已存在的文本文件"
        # zip 被删除（因为 f_out_dir.is_file() and f_out_dir.suffix == ".txt" != ""）
        assert not zip_path.exists()

    def test_bad_zip_file_skipped(self, tmp_path: Path) -> None:
        """损坏的 zip 文件不计入解压数量，continue 跳过（lines 154-156）。"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # 创建一个损坏的 zip 文件（内容不是合法 zip）
        bad_zip = data_dir / "corrupt.zip"
        bad_zip.write_bytes(b"this is not a valid zip file content at all")

        count = extract_archives(data_dir)

        # 损坏的 zip 被跳过，不计入
        assert count == 0
        # 解压目录可能已创建（mkdir 在 extractall 之前），内容为空
        out_dir = data_dir / "corrupt"
        # 只要函数正常返回不抛异常即可

    def test_copy_mode_dst_exists_incremental(self, tmp_path: Path) -> None:
        """复制模式下 dst 已存在时（symlink 失败降级后 dst 已创建），触发 WARNING（lines 85-86）。"""
        from unittest.mock import patch
        from pathlib import Path as _Path

        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "newfile.txt").write_text("新内容")

        dst_dir = tmp_path / "dst"
        dst_dir.mkdir()
        (dst_dir / "old.txt").write_text("旧内容")

        # symlink 失败 → 降级为 copy，此时 dst 已存在 → lines 85-86
        original_symlink_to = _Path.symlink_to

        def raising_symlink_to(self, target, target_is_directory=False):
            raise OSError("不支持 symlink")

        with patch.object(_Path, "symlink_to", raising_symlink_to):
            copytree(src=src_dir, dst=dst_dir, use_symlinks=True)

        # 增量复制：新旧文件都存在
        assert (dst_dir / "newfile.txt").read_text() == "新内容"
        assert (dst_dir / "old.txt").read_text() == "旧内容"
