"""Submission 格式验证的单元测试（P0-2 修复验证）。

测试范围:
    - _validate_submission_format(): glob 模式匹配 sample_submission 文件
    - 列名不匹配时标记为 invalid（升级从 warning）
"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch

from core.orchestrator import Orchestrator


class TestSampleSubmissionGlobMatching:
    """测试 sample_submission 文件的 glob 模式匹配。"""

    def test_glob_standard_name(self, tmp_path, mock_orchestrator_with_workspace):
        """标准文件名 'sample_submission.csv' → 正常匹配。"""
        orch = mock_orchestrator_with_workspace
        orch.config.project.workspace_dir = tmp_path

        # 创建测试文件
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "sample_submission.csv").write_text("id,label\n1,0.5\n2,0.5\n")

        submission_dir = tmp_path / "submission"
        submission_dir.mkdir()
        node_id = "test_node"
        (submission_dir / f"submission_{node_id}.csv").write_text("id,label\n1,0.5\n2,0.5\n")

        # 执行验证
        result = orch._validate_submission_format(node_id)

        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_glob_camelcase(self, tmp_path, mock_orchestrator_with_workspace):
        """驼峰式文件名 'sampleSubmission.csv' → 正常匹配。"""
        orch = mock_orchestrator_with_workspace
        orch.config.project.workspace_dir = tmp_path

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "sampleSubmission.csv").write_text("id,label\n1,0.5\n2,0.5\n")

        submission_dir = tmp_path / "submission"
        submission_dir.mkdir()
        node_id = "test_node"
        (submission_dir / f"submission_{node_id}.csv").write_text("id,label\n1,0.5\n2,0.5\n")

        result = orch._validate_submission_format(node_id)

        assert result["valid"] is True

    def test_glob_null_variant(self, tmp_path, mock_orchestrator_with_workspace):
        """变体文件名 'sample_submission_null.csv' → 正常匹配。"""
        orch = mock_orchestrator_with_workspace
        orch.config.project.workspace_dir = tmp_path

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "sample_submission_null.csv").write_text("id,label\n1,0.5\n2,0.5\n")

        submission_dir = tmp_path / "submission"
        submission_dir.mkdir()
        node_id = "test_node"
        (submission_dir / f"submission_{node_id}.csv").write_text("id,label\n1,0.5\n2,0.5\n")

        result = orch._validate_submission_format(node_id)

        assert result["valid"] is True

    def test_glob_no_match(self, tmp_path, mock_orchestrator_with_workspace):
        """无 sample_submission 文件 → sample_path=None，跳过对比。"""
        orch = mock_orchestrator_with_workspace
        orch.config.project.workspace_dir = tmp_path

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        # 不创建 sample_submission 文件

        submission_dir = tmp_path / "submission"
        submission_dir.mkdir()
        node_id = "test_node"
        (submission_dir / f"submission_{node_id}.csv").write_text("id,label\n1,0.5\n2,0.5\n")

        result = orch._validate_submission_format(node_id)

        # 无 sample_submission → 仅检查 NaN，应该通过
        assert result["valid"] is True

    def test_column_mismatch_invalid(self, tmp_path, mock_orchestrator_with_workspace):
        """列名不匹配 → valid=False（P0-2 修复：升级为 invalid）。"""
        orch = mock_orchestrator_with_workspace
        orch.config.project.workspace_dir = tmp_path

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "sample_submission.csv").write_text("id,label\n1,0.5\n2,0.5\n")

        submission_dir = tmp_path / "submission"
        submission_dir.mkdir()
        node_id = "test_node"
        # 列名不同: "id,prediction" vs "id,label"
        (submission_dir / f"submission_{node_id}.csv").write_text(
            "id,prediction\n1,0.5\n2,0.5\n"
        )

        result = orch._validate_submission_format(node_id)

        # P0-2 修复后应该标记为 invalid
        assert result["valid"] is False
        assert any("列名不匹配" in err for err in result["errors"])


class TestSubmissionRowMismatch:
    """测试行数不匹配的情况。"""

    def test_row_mismatch_invalid(self, tmp_path, mock_orchestrator_with_workspace):
        """行数不匹配 → valid=False。"""
        orch = mock_orchestrator_with_workspace
        orch.config.project.workspace_dir = tmp_path

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "sample_submission.csv").write_text("id,label\n1,0.5\n2,0.5\n3,0.5\n")

        submission_dir = tmp_path / "submission"
        submission_dir.mkdir()
        node_id = "test_node"
        # 只有 2 行，sample 有 3 行
        (submission_dir / f"submission_{node_id}.csv").write_text("id,label\n1,0.5\n2,0.5\n")

        result = orch._validate_submission_format(node_id)

        assert result["valid"] is False
        assert any("行数不匹配" in err for err in result["errors"])


class TestSubmissionNaNCheck:
    """测试 NaN 值检查。"""

    def test_nan_values_invalid(self, tmp_path, mock_orchestrator_with_workspace):
        """包含 NaN 值 → valid=False。"""
        orch = mock_orchestrator_with_workspace
        orch.config.project.workspace_dir = tmp_path

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "sample_submission.csv").write_text("id,label\n1,0.5\n2,0.5\n")

        submission_dir = tmp_path / "submission"
        submission_dir.mkdir()
        node_id = "test_node"
        # 第二行有 NaN
        (submission_dir / f"submission_{node_id}.csv").write_text("id,label\n1,0.5\n2,\n")

        result = orch._validate_submission_format(node_id)

        assert result["valid"] is False
        assert any("NaN" in err for err in result["errors"])


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def mock_orchestrator_with_workspace():
    """创建带 workspace 路径的 mock Orchestrator。"""
    orch = Mock(spec=Orchestrator)
    orch.config = Mock()
    orch.config.project = Mock()

    # 绑定真实方法
    orch._validate_submission_format = lambda node_id: Orchestrator._validate_submission_format(
        orch, node_id
    )

    return orch
