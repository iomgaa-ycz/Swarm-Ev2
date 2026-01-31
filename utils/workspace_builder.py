"""工作空间构建工具模块。

负责构建 Kaggle 竞赛的工作空间目录结构。
"""

import shutil
from pathlib import Path
from typing import Tuple

from utils.logger_system import log_msg, ensure


def build_workspace(
    data_dir: Path, workspace_dir: Path, copy_data: bool = False
) -> str:
    """构建完整的工作空间目录结构。

    参数:
        data_dir: 竞赛数据源目录（包含 description.md, train.csv 等）
        workspace_dir: 工作空间目标目录
        copy_data: 是否复制数据（False 时使用软链接）

    返回:
        description.md 的内容

    异常:
        FileNotFoundError: 当源文件不存在时抛出
        AssertionError: 当参数无效时抛出

    工作空间结构:
        ./workspace/
        ├── description.md            # 竞赛描述（只读）- 从数据源复制
        ├── input/                    # 输入数据（只读）- 软链接或复制
        ├── working/                  # Agent 工作目录（读写）
        ├── submission/               # 提交输出（读写）
        └── logs/                     # 日志输出（读写）
    """
    # Phase 1: 参数验证
    ensure(data_dir.exists(), f"数据目录不存在: {data_dir}")
    description_file = data_dir / "description.md"
    ensure(description_file.exists(), f"description.md 不存在: {description_file}")

    log_msg("INFO", f"开始构建工作空间: {workspace_dir}")

    # Phase 2: 创建目录结构
    workspace_dir.mkdir(parents=True, exist_ok=True)

    input_dir = workspace_dir / "input"
    working_dir = workspace_dir / "working"
    submission_dir = workspace_dir / "submission"
    logs_dir = workspace_dir / "logs"

    input_dir.mkdir(exist_ok=True)
    working_dir.mkdir(exist_ok=True)
    submission_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    # Phase 3: 复制 description.md
    description_target = workspace_dir / "description.md"
    shutil.copy2(description_file, description_target)
    log_msg("INFO", "✅ description.md 已复制")

    # Phase 4: 复制或链接数据文件到 input/ 目录
    _setup_input_data(data_dir, input_dir, copy_data)

    # Phase 5: 读取并返回 description.md 内容
    with open(description_file, "r", encoding="utf-8") as f:
        description_content = f.read()

    log_msg("INFO", "✅ 工作空间构建完成")
    return description_content


def _setup_input_data(data_dir: Path, input_dir: Path, copy_data: bool) -> None:
    """设置输入数据（复制或软链接）。

    参数:
        data_dir: 数据源目录
        input_dir: 输入目标目录
        copy_data: 是否复制数据（False 时使用软链接）
    """
    data_files = [
        f for f in data_dir.iterdir() if f.name != "description.md" and f.is_file()
    ]

    for data_file in data_files:
        target_path = input_dir / data_file.name

        # 如果已存在，先删除
        if target_path.exists() or target_path.is_symlink():
            target_path.unlink()

        if copy_data:
            # 复制数据
            shutil.copy2(data_file, target_path)
            log_msg("INFO", f"  [COPY] {data_file.name}")
        else:
            # 创建软链接
            target_path.symlink_to(data_file.resolve())
            log_msg("INFO", f"  [LINK] {data_file.name}")


def validate_dataset(data_dir: Path) -> Tuple[bool, str]:
    """验证数据集完整性。

    参数:
        data_dir: 数据目录路径

    返回:
        (是否有效, 错误消息)
    """
    # Phase 1: 检查目录存在性
    if not data_dir.exists():
        return False, f"数据目录不存在: {data_dir}"

    # Phase 2: 检查必需文件
    required_files = ["description.md"]
    for filename in required_files:
        if not (data_dir / filename).exists():
            return False, f"缺少必需文件: {filename}"

    # Phase 3: 检查数据文件（至少有一个 .csv 文件）
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        return False, "未找到任何 CSV 数据文件"

    return True, ""
