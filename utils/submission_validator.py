"""Submission 格式验证模块。

纯本地实现（基于 pandas），不依赖外部服务。
参考 ML-Master 的 check_format 机制。
"""

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from utils.logger_system import log_msg


def validate_submission(
    submission_path: Path,
    sample_submission_path: Optional[Path] = None,
) -> Tuple[bool, str]:
    """验证 submission.csv 的格式。

    检查项:
    1. 文件存在性
    2. 非空
    3. 无 NaN 值
    4. 行数匹配 sample_submission（如果提供）
    5. 列名匹配 sample_submission（如果提供）

    Args:
        submission_path: submission.csv 路径
        sample_submission_path: sample_submission.csv 路径（可选）

    Returns:
        (is_valid, error_message) 元组
    """
    if not submission_path.exists():
        return False, f"submission.csv not found at {submission_path}"

    try:
        sub = pd.read_csv(submission_path)
    except Exception as e:
        return False, f"Cannot read submission.csv: {e}"

    if len(sub) == 0:
        return False, "submission.csv is empty"

    if sub.isnull().any().any():
        nan_cols = sub.columns[sub.isnull().any()].tolist()
        nan_count = int(sub.isnull().sum().sum())
        return False, (
            f"submission.csv has {nan_count} NaN values in columns: {nan_cols}"
        )

    if sample_submission_path and sample_submission_path.exists():
        try:
            sample = pd.read_csv(sample_submission_path)

            # 列名检查
            if list(sub.columns) != list(sample.columns):
                return False, (
                    f"Column mismatch: submission has {list(sub.columns)}, "
                    f"expected {list(sample.columns)}"
                )

            # 行数检查
            if len(sub) != len(sample):
                return False, (
                    f"Row count mismatch: submission has {len(sub)} rows, "
                    f"expected {len(sample)}"
                )
        except Exception as e:
            log_msg("WARNING", f"sample_submission 比对失败: {e}")

    return True, "OK"
