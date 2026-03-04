"""utils/workspace_builder.py 的单元测试。"""

import pytest
from pathlib import Path

from utils.workspace_builder import build_workspace, validate_dataset, _setup_input_data


class TestBuildWorkspace:
    """测试 build_workspace 函数。"""

    def _make_data_dir(self, tmp_path: Path) -> Path:
        """创建标准测试数据目录。"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "description.md").write_text("# Competition\nPredict target.", encoding="utf-8")
        (data_dir / "train.csv").write_text("id,value\n1,100\n2,200\n", encoding="utf-8")
        return data_dir

    def test_build_workspace_creates_structure(self, tmp_path):
        """测试工作空间目录结构创建。"""
        data_dir = self._make_data_dir(tmp_path)
        workspace = tmp_path / "workspace"

        result = build_workspace(data_dir, workspace)

        assert (workspace / "input").is_dir()
        assert (workspace / "working").is_dir()
        assert (workspace / "submission").is_dir()
        assert (workspace / "logs").is_dir()
        assert (workspace / "description.md").exists()
        assert "# Competition" in result

    def test_build_workspace_returns_description(self, tmp_path):
        """测试返回 description.md 内容。"""
        data_dir = self._make_data_dir(tmp_path)
        workspace = tmp_path / "workspace"

        content = build_workspace(data_dir, workspace)

        assert "Predict target" in content

    def test_build_workspace_symlinks_data(self, tmp_path):
        """测试默认使用软链接。"""
        data_dir = self._make_data_dir(tmp_path)
        workspace = tmp_path / "workspace"

        build_workspace(data_dir, workspace, copy_data=False)

        input_train = workspace / "input" / "train.csv"
        assert input_train.is_symlink()

    def test_build_workspace_copies_data(self, tmp_path):
        """测试 copy_data=True 复制文件。"""
        data_dir = self._make_data_dir(tmp_path)
        workspace = tmp_path / "workspace"

        build_workspace(data_dir, workspace, copy_data=True)

        input_train = workspace / "input" / "train.csv"
        assert input_train.exists()
        assert not input_train.is_symlink()
        assert input_train.read_text() == "id,value\n1,100\n2,200\n"

    def test_build_workspace_missing_data_dir(self, tmp_path):
        """测试数据目录不存在时抛异常。"""
        with pytest.raises(AssertionError, match="数据目录不存在"):
            build_workspace(tmp_path / "nonexistent", tmp_path / "workspace")

    def test_build_workspace_missing_description(self, tmp_path):
        """测试缺少 description.md 时抛异常。"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "train.csv").write_text("id\n1\n")

        with pytest.raises(AssertionError, match="description.md 不存在"):
            build_workspace(data_dir, tmp_path / "workspace")

    def test_build_workspace_idempotent(self, tmp_path):
        """测试重复构建不报错。"""
        data_dir = self._make_data_dir(tmp_path)
        workspace = tmp_path / "workspace"

        build_workspace(data_dir, workspace)
        content = build_workspace(data_dir, workspace)

        assert "# Competition" in content


class TestSetupInputData:
    """测试 _setup_input_data 函数。"""

    def test_copy_directory(self, tmp_path):
        """测试复制子目录。"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        sub_dir = data_dir / "images"
        sub_dir.mkdir()
        (sub_dir / "img1.png").write_bytes(b"\x89PNG")
        (data_dir / "description.md").write_text("desc")

        input_dir = tmp_path / "input"
        input_dir.mkdir()

        _setup_input_data(data_dir, input_dir, copy_data=True)

        assert (input_dir / "images").is_dir()
        assert (input_dir / "images" / "img1.png").exists()

    def test_overwrite_existing_symlink(self, tmp_path):
        """测试覆盖已存在的软链接。"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "description.md").write_text("desc")
        (data_dir / "train.csv").write_text("id\n1\n")

        input_dir = tmp_path / "input"
        input_dir.mkdir()

        # 先创建一个旧的软链接
        old_target = tmp_path / "old_train.csv"
        old_target.write_text("old")
        (input_dir / "train.csv").symlink_to(old_target)

        _setup_input_data(data_dir, input_dir, copy_data=False)

        link = input_dir / "train.csv"
        assert link.is_symlink()
        assert link.resolve() == (data_dir / "train.csv").resolve()


class TestValidateDataset:
    """测试 validate_dataset 函数。"""

    def test_valid_dataset(self, tmp_path):
        """测试有效数据集。"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "description.md").write_text("desc")
        (data_dir / "train.csv").write_text("id\n1\n")

        valid, msg = validate_dataset(data_dir)
        assert valid is True
        assert msg == ""

    def test_missing_directory(self, tmp_path):
        """测试目录不存在。"""
        valid, msg = validate_dataset(tmp_path / "nonexistent")
        assert valid is False
        assert "数据目录不存在" in msg

    def test_missing_description(self, tmp_path):
        """测试缺少 description.md。"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "train.csv").write_text("id\n1\n")

        valid, msg = validate_dataset(data_dir)
        assert valid is False
        assert "description.md" in msg

    def test_no_data_files(self, tmp_path):
        """测试没有数据文件。"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "description.md").write_text("desc")

        valid, msg = validate_dataset(data_dir)
        assert valid is False
        assert "未找到任何数据文件" in msg

    def test_hidden_files_ignored(self, tmp_path):
        """测试隐藏文件被忽略。"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "description.md").write_text("desc")
        (data_dir / ".gitkeep").write_text("")  # 隐藏文件

        valid, msg = validate_dataset(data_dir)
        assert valid is False
        assert "未找到任何数据文件" in msg
