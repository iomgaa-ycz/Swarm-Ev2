"""Submission 格式验证模块单元测试。"""

import pytest
import pandas as pd
from pathlib import Path

from utils.submission_validator import validate_submission


@pytest.fixture
def tmp_dir(tmp_path):
    """创建临时目录。"""
    return tmp_path


class TestValidateSubmission:
    """validate_submission() 测试。"""

    def test_valid_submission(self, tmp_dir):
        """有效的 submission 应通过。"""
        sub_path = tmp_dir / "submission.csv"
        pd.DataFrame({"id": [1, 2, 3], "target": [0.1, 0.2, 0.3]}).to_csv(
            sub_path, index=False
        )
        is_valid, msg = validate_submission(sub_path)
        assert is_valid is True
        assert msg == "OK"

    def test_file_not_found(self, tmp_dir):
        """文件不存在应失败。"""
        is_valid, msg = validate_submission(tmp_dir / "not_exist.csv")
        assert is_valid is False
        assert "not found" in msg

    def test_empty_submission(self, tmp_dir):
        """空 submission 应失败。"""
        sub_path = tmp_dir / "submission.csv"
        pd.DataFrame(columns=["id", "target"]).to_csv(sub_path, index=False)
        is_valid, msg = validate_submission(sub_path)
        assert is_valid is False
        assert "empty" in msg

    def test_nan_values(self, tmp_dir):
        """含 NaN 值应失败。"""
        sub_path = tmp_dir / "submission.csv"
        pd.DataFrame({"id": [1, 2], "target": [0.1, None]}).to_csv(
            sub_path, index=False
        )
        is_valid, msg = validate_submission(sub_path)
        assert is_valid is False
        assert "NaN" in msg

    def test_row_count_mismatch(self, tmp_dir):
        """行数不匹配应失败。"""
        sub_path = tmp_dir / "submission.csv"
        sample_path = tmp_dir / "sample_submission.csv"

        pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]}).to_csv(sub_path, index=False)
        pd.DataFrame({"id": [1, 2, 3], "target": [0.1, 0.2, 0.3]}).to_csv(
            sample_path, index=False
        )

        is_valid, msg = validate_submission(sub_path, sample_path)
        assert is_valid is False
        assert "Row count" in msg

    def test_column_mismatch(self, tmp_dir):
        """列名不匹配应失败。"""
        sub_path = tmp_dir / "submission.csv"
        sample_path = tmp_dir / "sample_submission.csv"

        pd.DataFrame({"id": [1, 2], "prediction": [0.1, 0.2]}).to_csv(
            sub_path, index=False
        )
        pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]}).to_csv(
            sample_path, index=False
        )

        is_valid, msg = validate_submission(sub_path, sample_path)
        assert is_valid is False
        assert "Column mismatch" in msg

    def test_matching_sample(self, tmp_dir):
        """与 sample 完全匹配应通过。"""
        sub_path = tmp_dir / "submission.csv"
        sample_path = tmp_dir / "sample_submission.csv"

        data = {"id": [1, 2, 3], "target": [0.1, 0.2, 0.3]}
        pd.DataFrame(data).to_csv(sub_path, index=False)
        pd.DataFrame(data).to_csv(sample_path, index=False)

        is_valid, msg = validate_submission(sub_path, sample_path)
        assert is_valid is True

    def test_no_sample_provided(self, tmp_dir):
        """不提供 sample 时，只检查基本格式。"""
        sub_path = tmp_dir / "submission.csv"
        pd.DataFrame({"id": [1], "target": [0.5]}).to_csv(sub_path, index=False)

        is_valid, msg = validate_submission(sub_path, None)
        assert is_valid is True
