"""
工作空间管理模块。

负责工作空间目录结构管理、文件路径重写、节点文件归档等功能。
"""

import re
import shutil
import zipfile
from pathlib import Path
from typing import Optional, Any

from utils.logger_system import log_msg


class WorkspaceManager:
    """工作空间管理器。

    目录结构：
    workspace/
    ├── input/          # 输入数据（符号链接到原始数据）
    ├── working/        # 临时工作目录
    ├── submission/     # 预测结果（每个 node 一个文件）
    │   ├── submission_<node_id>.csv
    │   └── ...
    ├── archives/       # 归档文件（每个 node 一个 zip）
    │   ├── node_<node_id>.zip
    │   └── ...
    └── best_solution/  # 最佳解决方案
        ├── solution.py
        └── submission.csv
    """

    def __init__(self, config: Any):
        """初始化工作空间管理器。

        Args:
            config: 配置对象（包含 project.workspace_dir 等）
        """
        self.config = config
        self.workspace_dir = Path(config.project.workspace_dir)

    def setup(self) -> None:
        """创建工作空间目录结构。

        创建目录：
        - input/
        - working/
        - submission/
        - archives/
        - best_solution/
        """
        dirs = ["input", "working", "submission", "archives", "best_solution"]
        for dir_name in dirs:
            (self.workspace_dir / dir_name).mkdir(parents=True, exist_ok=True)

        log_msg("INFO", f"工作空间已创建: {self.workspace_dir}")

    def link_input_data(self, source_dir: Optional[Path] = None) -> None:
        """链接输入数据到 workspace/input/。

        优先使用符号链接，Windows 上降级为目录复制。

        Args:
            source_dir: 数据源目录（默认使用 config.data.input_dir）
        """
        if source_dir is None:
            source_dir = Path(self.config.data.input_dir)

        if not source_dir.exists():
            log_msg("ERROR", f"数据源目录不存在: {source_dir}")
            raise FileNotFoundError(f"数据源目录不存在: {source_dir}")

        input_link = self.workspace_dir / "input"

        # 如果已存在链接/目录，先删除
        if input_link.exists() or input_link.is_symlink():
            if input_link.is_symlink():
                input_link.unlink()
            else:
                shutil.rmtree(input_link)

        # 尝试创建符号链接
        try:
            input_link.symlink_to(source_dir, target_is_directory=True)
            log_msg("INFO", f"已创建符号链接: {input_link} -> {source_dir}")
        except (OSError, NotImplementedError):
            # Windows 上可能失败，降级为目录复制
            log_msg("WARNING", "符号链接创建失败，降级为目录复制")
            shutil.copytree(source_dir, input_link)
            log_msg("INFO", f"已复制数据到: {input_link}")

    def rewrite_submission_path(self, code: str, node_id: str) -> str:
        """重写代码中的 submission 路径。

        将 './submission/submission.csv' 替换为 './submission/submission_{node_id}.csv'

        Args:
            code: 原始代码
            node_id: 节点 ID

        Returns:
            修改后的代码

        Examples:
            >>> manager.rewrite_submission_path('df.to_csv("./submission/submission.csv")', 'abc123')
            'df.to_csv("./submission/submission_abc123.csv")'
        """
        # 匹配各种可能的写法
        patterns = [
            (
                r'(["\'])\.?/submission/submission\.csv\1',
                f"\\1./submission/submission_{node_id}.csv\\1",
            ),
            (r'(["\'])submission\.csv\1', f"\\1submission_{node_id}.csv\\1"),
        ]

        modified_code = code
        for pattern, replacement in patterns:
            modified_code = re.sub(pattern, replacement, modified_code)

        return modified_code

    def archive_node_files(self, node_id: str, code: str) -> Optional[Path]:
        """打包节点的 solution.py 和 submission.csv 为 zip。

        Args:
            node_id: 节点 ID
            code: 代码内容

        Returns:
            zip 文件路径（如果打包成功），否则 None
        """
        archives_dir = self.workspace_dir / "archives"
        archives_dir.mkdir(exist_ok=True)

        zip_path = archives_dir / f"node_{node_id}.zip"
        submission_path = (
            self.workspace_dir / "submission" / f"submission_{node_id}.csv"
        )

        try:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                # 添加 solution.py
                zf.writestr("solution.py", code)

                # 添加 submission.csv（如果存在）
                if submission_path.exists():
                    zf.write(submission_path, "submission.csv")
                    log_msg(
                        "INFO", f"已归档节点 {node_id}: solution.py + submission.csv"
                    )
                else:
                    log_msg(
                        "INFO", f"已归档节点 {node_id}: solution.py (无 submission)"
                    )

            return zip_path

        except Exception as e:
            log_msg("ERROR", f"归档节点 {node_id} 失败: {e}")
            return None

    def cleanup_submission(self) -> None:
        """清空 submission 目录。

        注意：通常只在 run 开始时调用一次，而非每步调用。
        """
        submission_dir = self.workspace_dir / "submission"
        if submission_dir.exists():
            shutil.rmtree(submission_dir)
            submission_dir.mkdir(exist_ok=True)
            log_msg("INFO", "已清空 submission 目录")

    def cleanup_working(self) -> None:
        """清空 working 目录。"""
        working_dir = self.workspace_dir / "working"
        if working_dir.exists():
            shutil.rmtree(working_dir)
            working_dir.mkdir(exist_ok=True)
            log_msg("INFO", "已清空 working 目录")
