"""
utils/data_preview.py 的单元测试。
"""

import pandas as pd
from utils.data_preview import (
    generate,
    preview_csv,
    file_tree,
    get_file_len_size,
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
