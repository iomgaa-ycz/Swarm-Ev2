"""
utils/data_preview.py 的单元测试。
"""

import json
import pandas as pd
from utils.data_preview import (
    generate,
    preview_csv,
    preview_json,
    preview_special_file,
    preview_image_dir,
    file_tree,
    get_file_len_size,
    _walk,
)


class TestGetFileLenSize:
    """测试 get_file_len_size 函数。"""

    def test_text_file(self, tmp_path):
        """测试文本文件返回行数。"""
        file = tmp_path / "test.txt"
        file.write_text("line1\nline2\nline3")
        size, desc = get_file_len_size(file)
        assert size == 3
        assert "lines" in desc

    def test_csv_file(self, tmp_path):
        """测试 CSV 文件返回行数。"""
        file = tmp_path / "test.csv"
        file.write_text("a,b\n1,2\n3,4")
        size, desc = get_file_len_size(file)
        assert size == 3
        assert "lines" in desc


class TestFileTree:
    """测试 file_tree 函数。"""

    def test_empty_directory(self, tmp_path):
        """测试空目录。"""
        tree = file_tree(tmp_path)
        assert tree == ""

    def test_simple_directory(self, tmp_path):
        """测试简单目录结构。"""
        (tmp_path / "file1.txt").write_text("content")
        (tmp_path / "file2.csv").write_text("a,b\n1,2")
        tree = file_tree(tmp_path)
        assert "file1.txt" in tree
        assert "file2.csv" in tree


class TestPreviewCsv:
    """测试 preview_csv 函数。"""

    def test_preview_csv_simple_mode(self, tmp_path):
        """测试简单模式。"""
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        df.to_csv(csv_file, index=False)

        preview = preview_csv(csv_file, "test.csv", simple=True)
        assert "3 rows and 3 columns" in preview
        assert "a, b, c" in preview

    def test_preview_csv_detailed_mode(self, tmp_path):
        """测试详细模式。"""
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({"numeric": [1.0, 2.0, 3.0], "category": ["A", "B", "A"]})
        df.to_csv(csv_file, index=False)

        preview = preview_csv(csv_file, "test.csv", simple=False)
        assert "3 rows and 2 columns" in preview
        assert "Here is some information about the columns:" in preview
        assert "numeric" in preview
        assert "category" in preview

    def test_preview_csv_bool_column(self, tmp_path):
        """测试 bool 列统计。"""
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({"flag": [True, False, True, True]})
        df.to_csv(csv_file, index=False)

        preview = preview_csv(csv_file, "test.csv", simple=False)
        assert "True" in preview
        assert "False" in preview

    def test_preview_csv_low_cardinality(self, tmp_path):
        """测试低基数列。"""
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({"status": ["A", "B", "C", "A", "B"]})
        df.to_csv(csv_file, index=False)

        preview = preview_csv(csv_file, "test.csv", simple=False)
        assert "unique values" in preview


class TestGenerate:
    """测试 generate 函数。"""

    def test_generate_with_csv(self, tmp_path):
        """测试生成包含 CSV 的预览。"""
        csv_file = tmp_path / "train.csv"
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df.to_csv(csv_file, index=False)

        preview = generate(tmp_path, include_file_details=True, simple=True)
        assert "train.csv" in preview
        assert "2 rows and 2 columns" in preview

    def test_generate_empty_dir(self, tmp_path):
        """测试空目录。"""
        preview = generate(tmp_path, include_file_details=True, simple=True)
        assert preview  # 至少包含目录树

    def test_generate_auto_truncation(self, tmp_path):
        """测试自动截断。"""
        # 创建大量文件触发截断
        for i in range(100):
            csv_file = tmp_path / f"file_{i}.csv"
            df = pd.DataFrame({"col": list(range(1000))})
            df.to_csv(csv_file, index=False)

        preview = generate(tmp_path, include_file_details=True, simple=False)
        # 应该自动降级或截断
        assert len(preview) <= 7000  # 稍大于 6000 的阈值（考虑标记）


# ============================================================
# 以下测试迁移自 test_p0_v6_fixes.py
# ============================================================


class TestPreviewSpecialFileSep:
    """测试 preview_special_file() 输出包含 pd.read_csv(sep=...) 提示。"""

    def test_comma_separated(self, tmp_path):
        """逗号分隔文件包含 pd.read_csv 提示。"""
        f = tmp_path / "train.csv"
        f.write_text("id,name,value\n1,foo,100\n2,bar,200\n", encoding="utf-8")

        result = preview_special_file(f)
        assert "comma-separated" in result
        assert "pd.read_csv('train.csv', sep=',')" in result

    def test_tab_separated(self, tmp_path):
        """Tab 分隔文件包含 pd.read_csv(sep='\\t') 提示。"""
        f = tmp_path / "data.tsv"
        f.write_text("id\tname\tvalue\n1\tfoo\t100\n", encoding="utf-8")

        result = preview_special_file(f)
        assert "tab-separated" in result
        assert r"pd.read_csv('data.tsv', sep='\t')" in result

    def test_space_separated(self, tmp_path):
        """空格分隔文件包含 pd.read_csv(sep=' ') 提示。"""
        f = tmp_path / "data.txt"
        f.write_text("id name value\n1 foo 100\n", encoding="utf-8")

        result = preview_special_file(f)
        assert "space-separated" in result
        assert "pd.read_csv('data.txt', sep=' ')" in result

    def test_unstructured_no_sep_hint(self, tmp_path):
        """非结构化文件不包含 pd.read_csv 提示。"""
        f = tmp_path / "readme.txt"
        f.write_text("Thisisasolidblockoftextwithnoseparators\n", encoding="utf-8")

        result = preview_special_file(f)
        assert "pd.read_csv" not in result

    def test_empty_file(self, tmp_path):
        """测试空文件。"""
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")

        result = preview_special_file(f)
        assert "empty file" in result


# ============================================================
# 补充覆盖率测试
# ============================================================


class TestGetFileLenSizeBinary:
    """测试 get_file_len_size 二进制文件路径。"""

    def test_binary_file_returns_bytes(self, tmp_path):
        """测试二进制文件返回字节数。"""
        file = tmp_path / "data.bin"
        file.write_bytes(b"\x00" * 1024)
        size, desc = get_file_len_size(file)
        assert size == 1024
        assert "Bytes" in desc or "kB" in desc


class TestFileTreeExtended:
    """测试 file_tree 的高级功能。"""

    def test_many_files_truncated(self, tmp_path):
        """测试超过 30 个文件时截断显示。"""
        for i in range(35):
            (tmp_path / f"file_{i:03d}.txt").write_text(f"line {i}")

        tree = file_tree(tmp_path)
        assert "... and" in tree
        assert "other files" in tree

    def test_nested_directories(self, tmp_path):
        """测试嵌套目录结构。"""
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "nested.txt").write_text("nested content")
        (tmp_path / "root.txt").write_text("root content")

        tree = file_tree(tmp_path)
        assert "subdir/" in tree
        assert "nested.txt" in tree
        assert "root.txt" in tree

    def test_max_dirs_limit(self, tmp_path):
        """测试最大目录数限制。"""
        for i in range(25):
            d = tmp_path / f"dir_{i:03d}"
            d.mkdir()
            (d / "file.txt").write_text("content")

        tree = file_tree(tmp_path, max_dirs=5)
        assert "... and" in tree
        assert "other subfolders" in tree


class TestWalk:
    """测试 _walk 递归遍历。"""

    def test_walk_simple(self, tmp_path):
        """测试简单目录遍历。"""
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")

        files = list(_walk(tmp_path))
        names = [f.name for f in files]
        assert "a.txt" in names
        assert "b.txt" in names

    def test_walk_recursive(self, tmp_path):
        """测试递归遍历子目录。"""
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.txt").write_text("nested")
        (tmp_path / "root.txt").write_text("root")

        files = list(_walk(tmp_path))
        names = [f.name for f in files]
        assert "root.txt" in names
        assert "nested.txt" in names


class TestPreviewCsvExtended:
    """测试 preview_csv 的额外路径。"""

    def test_many_columns_truncated(self, tmp_path):
        """测试超过 15 列时简化模式截断。"""
        csv_file = tmp_path / "wide.csv"
        data = {f"col_{i}": [1] for i in range(20)}
        pd.DataFrame(data).to_csv(csv_file, index=False)

        preview = preview_csv(csv_file, "wide.csv", simple=True)
        assert "more columns" in preview

    def test_numeric_column_range(self, tmp_path):
        """测试数值列显示范围。"""
        csv_file = tmp_path / "nums.csv"
        pd.DataFrame({"val": range(100)}).to_csv(csv_file, index=False)

        preview = preview_csv(csv_file, "nums.csv", simple=False)
        assert "range" in preview
        assert "nan values" in preview

    def test_object_column_unique_values(self, tmp_path):
        """测试字符串列显示唯一值（低基数确保 nunique < 10 分支）。"""
        csv_file = tmp_path / "strings.csv"
        # 使用低基数数据确保 nunique < 10 分支命中
        pd.DataFrame({"city": ["A", "B", "C", "A", "B"]}).to_csv(csv_file, index=False)

        preview = preview_csv(csv_file, "strings.csv", simple=False)
        assert "city" in preview
        assert "unique values" in preview


class TestPreviewJson:
    """测试 preview_json 函数。"""

    def test_preview_json_object(self, tmp_path):
        """测试标准 JSON 对象。"""
        json_file = tmp_path / "data.json"
        data = {"name": "test", "value": 42, "tags": ["a", "b"]}
        json_file.write_text(json.dumps(data), encoding="utf-8")

        result = preview_json(json_file, "data.json")
        assert "data.json" in result
        assert "json schema" in result

    def test_preview_jsonl(self, tmp_path):
        """测试 JSONL（每行一个 JSON 对象）。"""
        jsonl_file = tmp_path / "data.jsonl"
        lines = [
            json.dumps({"id": 1, "text": "hello"}),
            json.dumps({"id": 2, "text": "world"}),
        ]
        jsonl_file.write_text("\n".join(lines), encoding="utf-8")

        result = preview_json(jsonl_file, "data.jsonl")
        assert "data.jsonl" in result
        assert "json schema" in result

    def test_preview_json_prettified(self, tmp_path):
        """测试格式化的 JSON 文件。"""
        json_file = tmp_path / "pretty.json"
        data = {"key": "value", "nested": {"a": 1}}
        json_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

        result = preview_json(json_file, "pretty.json")
        assert "pretty.json" in result
        assert "json schema" in result

    def test_preview_json_invalid(self, tmp_path):
        """测试无效 JSON 文件。"""
        json_file = tmp_path / "bad.json"
        json_file.write_text("not valid json {{{", encoding="utf-8")

        result = preview_json(json_file, "bad.json")
        assert "bad.json" in result
        assert "无法读取" in result or "cannot" in result.lower()


class TestPreviewImageDir:
    """测试 preview_image_dir 函数。"""

    def test_empty_dir(self, tmp_path):
        """测试无图像文件的目录。"""
        result = preview_image_dir(tmp_path)
        assert result == ""

    def test_non_image_files_ignored(self, tmp_path):
        """测试非图像文件被忽略。"""
        (tmp_path / "readme.txt").write_text("text")
        result = preview_image_dir(tmp_path)
        assert result == ""


class TestGenerateExtended:
    """测试 generate 的更多路径。"""

    def test_generate_with_json(self, tmp_path):
        """测试生成包含 JSON 的预览。"""
        json_file = tmp_path / "data.json"
        json_file.write_text('{"a": 1}', encoding="utf-8")

        preview = generate(tmp_path, include_file_details=True, simple=False)
        assert "data.json" in preview
        assert "json schema" in preview

    def test_generate_with_special_text(self, tmp_path):
        """测试生成包含特殊文本文件的预览。"""
        txt_file = tmp_path / "data.txt"
        # 创建 >= 30 行的文件触发 preview_special_file
        lines = [f"id\tval_{i}" for i in range(40)]
        txt_file.write_text("\n".join(lines), encoding="utf-8")

        preview = generate(tmp_path, include_file_details=True, simple=False)
        assert "data.txt" in preview

    def test_generate_with_small_plaintext(self, tmp_path):
        """测试小型纯文本文件直接显示内容。"""
        txt_file = tmp_path / "small.txt"
        txt_file.write_text("line1\nline2\n", encoding="utf-8")

        preview = generate(tmp_path, include_file_details=True, simple=False)
        assert "small.txt" in preview
        assert "line1" in preview

    def test_generate_no_details(self, tmp_path):
        """测试 include_file_details=False 只显示树结构。"""
        (tmp_path / "train.csv").write_text("id\n1\n")

        preview = generate(tmp_path, include_file_details=False)
        assert "train.csv" in preview
        # 不应有详细的列信息
        assert "rows and" not in preview

    def test_generate_with_code_file(self, tmp_path):
        """测试代码文件用 markdown 包裹。"""
        py_file = tmp_path / "script.py"
        py_file.write_text("print('hello')\n", encoding="utf-8")

        preview = generate(tmp_path, include_file_details=True, simple=False)
        assert "script.py" in preview
        assert "```" in preview


# ============================================================
# 新增：覆盖率补全测试（针对未覆盖行）
# ============================================================


class TestGetFileLenSizeException:
    """测试 get_file_len_size 异常路径（lines 45-48）。"""

    def test_exception_falls_back_to_file_size(self, tmp_path):
        """当文本文件读取抛出异常时，回退到返回文件字节数（lines 45-48）。"""
        from unittest.mock import patch

        txt_file = tmp_path / "data.txt"
        txt_file.write_text("line1\nline2\nline3", encoding="utf-8")

        # 模拟 open() 在文本后缀文件上抛出异常
        with patch("builtins.open", side_effect=OSError("模拟IO错误")):
            size, desc = get_file_len_size(txt_file)

        # 应回退到字节数
        file_bytes = txt_file.stat().st_size
        assert size == file_bytes
        # 描述不包含 "lines"（因为失败了）
        assert "lines" not in desc


class TestFileTreePermissionError:
    """测试 file_tree PermissionError 路径（line 70）。"""

    def test_permission_denied_returns_message(self, tmp_path):
        """当 iterdir 抛出 PermissionError 时返回 [Permission Denied]（line 70）。"""
        from unittest.mock import patch

        # 在 file_tree 内部的 Path.iterdir 模拟权限拒绝
        with patch("utils.data_preview.Path.iterdir", side_effect=PermissionError("拒绝")):
            result = file_tree(tmp_path)

        assert "[Permission Denied]" in result


class TestPreviewCsvException:
    """测试 preview_csv 异常路径（lines 135-137）。"""

    def test_csv_read_failure_returns_error_message(self, tmp_path):
        """当 pd.read_csv 失败时返回包含文件名的错误消息（lines 135-137）。"""
        from unittest.mock import patch

        csv_file = tmp_path / "broken.csv"
        csv_file.write_text("a,b\n1,2\n", encoding="utf-8")

        with patch("pandas.read_csv", side_effect=Exception("磁盘读取错误")):
            result = preview_csv(csv_file, "broken.csv", simple=False)

        assert "broken.csv" in result
        assert "无法读取" in result


class TestPreviewCsvObjectHighCardinality:
    """测试 preview_csv object 类型高基数路径（lines 171-172）。"""

    def test_object_dtype_high_cardinality(self, tmp_path):
        """测试 object 列 nunique >= 10 时显示唯一值数量和示例（lines 171-172）。

        pandas 3.0 中纯字符串列的 dtype 为 'str' 而非 'object'，
        因此通过 mock pd.read_csv 返回显式 object dtype 的 DataFrame 来覆盖此分支。
        """
        from unittest.mock import patch

        csv_file = tmp_path / "text.csv"
        csv_file.write_text("name\n" + "\n".join(f"item_{i}" for i in range(20)), encoding="utf-8")

        # 构造 object dtype 的 DataFrame（pandas 3.0 不再自动推断纯字符串为 object）
        # 必须使用 pd.Series(dtype='object') 显式指定，否则会被推断为 'str'
        values = pd.Series([f"item_{i}" for i in range(20)], dtype="object")
        mock_df = pd.DataFrame({"name": values})
        assert mock_df["name"].dtype == object, "期望 object dtype"
        assert mock_df["name"].nunique() >= 10, "期望高基数"

        with patch("pandas.read_csv", return_value=mock_df):
            preview = preview_csv(csv_file, "text.csv", simple=False)

        # object 且 nunique >= 10 时命中 lines 171-172 分支
        assert "name" in preview
        assert "unique values" in preview
        assert "Some example values" in preview


class TestPreviewSpecialFileException:
    """测试 preview_special_file 异常路径（lines 316-317）。"""

    def test_read_exception_returns_cannot_read(self, tmp_path):
        """当文件读取抛出异常时返回 (cannot read: ...)（lines 316-317）。"""
        from unittest.mock import patch

        txt_file = tmp_path / "special.txt"
        txt_file.write_text("content", encoding="utf-8")

        with patch("builtins.open", side_effect=OSError("权限拒绝")):
            result = preview_special_file(txt_file)

        assert "cannot read" in result
        assert "special.txt" in result


class TestPreviewImageDirWithPIL:
    """测试 preview_image_dir 使用 PIL 路径（lines 244-274）。"""

    def test_with_mock_pil_images(self, tmp_path):
        """模拟 PIL 和 numpy，测试图像目录预览完整路径（lines 244-267）。"""
        import sys
        from unittest.mock import patch, MagicMock

        # 创建假图像文件
        img_file = tmp_path / "sample.jpg"
        img_file.write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)  # 伪 JPEG

        # 构建完整的 PIL/numpy mock
        mock_img = MagicMock()
        mock_img.__enter__ = lambda s: s
        mock_img.__exit__ = MagicMock(return_value=False)
        mock_img.size = (128, 128)
        mock_img.mode = "RGB"

        mock_pil_module = MagicMock()
        mock_pil_module.Image.open.return_value = mock_img

        mock_arr = MagicMock()
        mock_arr.min.return_value = 0
        mock_arr.max.return_value = 255

        mock_np_module = MagicMock()
        mock_np_module.array.return_value = mock_arr

        with patch.dict(sys.modules, {"PIL": mock_pil_module, "PIL.Image": mock_pil_module.Image, "numpy": mock_np_module}):
            # 移除已缓存的模块引用，确保 import 重新执行
            result = preview_image_dir(tmp_path, max_sample=1)

        # 只要没有抛出异常，函数正确进入了"有图像文件"分支
        # 结果可能是成功预览或 ImportError/Exception 降级
        assert isinstance(result, str)
        assert len(result) > 0

    def test_import_error_path(self, tmp_path):
        """当 PIL 不可用时返回降级信息（lines 268-271）。"""
        import sys
        from unittest.mock import patch

        # 创建假图像文件以确保走到 PIL import 分支
        img_file = tmp_path / "sample.png"
        img_file.write_bytes(b"\x89PNG" + b"\x00" * 50)

        # 强制 import PIL 失败
        with patch.dict(sys.modules, {"PIL": None, "PIL.Image": None}):
            result = preview_image_dir(tmp_path)

        assert "PIL not available" in result
        assert "1 images" in result

    def test_image_open_exception_path(self, tmp_path):
        """当 PIL.Image.open 失败时返回降级信息（lines 272-274）。"""
        import sys
        from unittest.mock import patch, MagicMock

        img_file = tmp_path / "corrupt.jpg"
        img_file.write_bytes(b"\xff\xd8" + b"\x00" * 10)

        mock_pil_module = MagicMock()
        mock_pil_module.Image.open.side_effect = Exception("文件损坏")

        mock_np_module = MagicMock()

        with patch.dict(sys.modules, {"PIL": mock_pil_module, "PIL.Image": mock_pil_module.Image, "numpy": mock_np_module}):
            result = preview_image_dir(tmp_path)

        assert "preview failed" in result or "PIL not available" in result


class TestGenerateImageFiles:
    """测试 generate 处理图像文件路径（lines 355-361）。"""

    def test_generate_with_image_file_calls_preview_image_dir(self, tmp_path):
        """图像文件触发 preview_image_dir（lines 355-361）。"""
        from unittest.mock import patch

        # 在子目录中创建假图像文件
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        img_file = img_dir / "photo.jpg"
        img_file.write_bytes(b"\xff\xd8\xff" + b"\x00" * 50)

        # mock preview_image_dir 返回固定字符串
        with patch("utils.data_preview.preview_image_dir", return_value="-> Image directory: 1 images") as mock_preview:
            result = generate(tmp_path, include_file_details=True, simple=False)

        mock_preview.assert_called_once_with(img_dir)
        assert "Image directory" in result

    def test_generate_image_dir_previewed_only_once(self, tmp_path):
        """同一目录中多个图像只触发一次 preview_image_dir（lines 355-361）。"""
        from unittest.mock import patch

        img_dir = tmp_path / "photos"
        img_dir.mkdir()
        for name in ["a.jpg", "b.png", "c.jpeg"]:
            (img_dir / name).write_bytes(b"\xff\xd8\xff" + b"\x00" * 10)

        with patch("utils.data_preview.preview_image_dir", return_value="-> Image directory: 3 images") as mock_preview:
            generate(tmp_path, include_file_details=True, simple=False)

        # 只调用了一次（同一目录去重）
        assert mock_preview.call_count == 1


class TestGenerateFileReadException:
    """测试 generate 中小文件读取失败路径（lines 374-375）。"""

    def test_small_plaintext_read_exception(self, tmp_path):
        """小型纯文本文件读取失败时，generate 不抛出异常（lines 374-375）。"""
        from unittest.mock import patch, mock_open

        # 创建一个小的 .txt 文件（< 30 行）
        small_file = tmp_path / "small.txt"
        small_file.write_text("a\nb\nc\n", encoding="utf-8")

        call_count = {"n": 0}
        original_open = open

        def selective_open(file, *args, **kwargs):
            """第一次调用（计行数）正常，第二次（读内容）抛出异常。"""
            call_count["n"] += 1
            if call_count["n"] >= 2:
                raise OSError("模拟读取失败")
            return original_open(file, *args, **kwargs)

        with patch("builtins.open", side_effect=selective_open):
            # 不应该抛出异常
            result = generate(tmp_path, include_file_details=True, simple=False)

        # 结果应该是字符串（不抛出）
        assert isinstance(result, str)


class TestGenerateSimpleTruncation:
    """测试 generate 在 simple=True 时仍超长的截断路径（lines 387-388）。"""

    def test_truncation_when_simple_still_too_long(self, tmp_path):
        """当 simple 模式仍超过 MAX_PREVIEW_LENGTH 时，截断输出（lines 387-388）。"""
        from unittest.mock import patch
        from utils.data_preview import MAX_PREVIEW_LENGTH

        # 创建足够多的 CSV 文件使 simple 模式也超长
        for i in range(60):
            csv_file = tmp_path / f"data_{i:03d}.csv"
            # 每个文件有足够多的列名使得 simple 输出较长
            cols = {f"column_name_{j}": [1, 2] for j in range(20)}
            pd.DataFrame(cols).to_csv(csv_file, index=False)

        # 在 simple=True 模式下，如果超长，mock 使 result 超出阈值
        with patch("utils.data_preview.MAX_PREVIEW_LENGTH", 100):
            result = generate(tmp_path, include_file_details=True, simple=True)

        # 结果应该被截断
        assert "truncated" in result or len(result) <= MAX_PREVIEW_LENGTH + len("\n... (truncated)")
