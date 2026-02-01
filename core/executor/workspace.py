"""
工作空间管理模块。

负责工作空间目录结构管理、文件路径重写、节点文件归档、数据预处理等功能。
"""

import re
import shutil
import zipfile
from pathlib import Path
from typing import Optional, Any

from utils.logger_system import log_msg
from utils.file_utils import extract_archives, clean_up_dataset


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

    def preprocess_input(self) -> None:
        """预处理输入数据：解压压缩包 + 清理垃圾文件。

        预处理内容:
            1. 解压 workspace/input/ 下的所有 .zip 文件
            2. 清理 __MACOSX、.DS_Store 等垃圾文件

        注意:
            - 需要在 link_input_data() 之后调用
            - 如果使用 symlink 模式，会先转换为复制模式再预处理
        """
        input_dir = self.workspace_dir / "input"

        if not input_dir.exists():
            log_msg("WARNING", "输入目录不存在，跳过预处理")
            return

        # 如果是符号链接，需要先转换为实际复制
        if input_dir.is_symlink():
            log_msg("INFO", "检测到符号链接，转换为复制模式以支持预处理")
            source = input_dir.resolve()
            input_dir.unlink()
            shutil.copytree(source, input_dir)
            log_msg("INFO", f"已复制数据: {source} -> {input_dir}")

        # 解压压缩包
        extracted = extract_archives(input_dir)
        if extracted > 0:
            log_msg("INFO", f"预处理完成: 解压 {extracted} 个压缩包")

        # 清理垃圾文件
        cleaned = clean_up_dataset(input_dir)
        if cleaned > 0:
            log_msg("INFO", f"预处理完成: 清理 {cleaned} 个垃圾文件")

    def prepare_workspace(self, source_dir: Optional[Path] = None) -> None:
        """一站式准备工作空间。

        执行流程:
            1. setup() - 创建目录结构
            2. link_input_data() - 复制/链接输入数据
            3. preprocess_input() - 预处理（如果配置启用）

        Args:
            source_dir: 数据源目录（默认使用 config.data.data_dir）

        配置项:
            - data.preprocess_data: 是否启用预处理（默认 true）
        """
        # Phase 1: 创建目录结构
        self.setup()

        # Phase 2: 链接/复制输入数据
        self.link_input_data(source_dir)

        # Phase 3: 预处理（根据配置）
        preprocess_enabled = getattr(self.config.data, "preprocess_data", True)
        if preprocess_enabled:
            self.preprocess_input()
        else:
            log_msg("INFO", "数据预处理已禁用（config.data.preprocess_data=false）")
