"""文件操作工具模块。

提供文件和目录的复制、链接等操作，支持跨平台兼容性。
"""

from pathlib import Path
import shutil
import os
import stat


def copytree(src: Path, dst: Path, use_symlinks: bool = True) -> None:
    """复制或链接目录树。

    Args:
        src: 源目录（必须存在）
        dst: 目标目录（自动创建）
        use_symlinks: 是否使用符号链接（Windows 平台或权限不足时自动降级为复制）

    Raises:
        FileNotFoundError: 源目录不存在

    实现细节:
        - 使用 symlink 时会设置只读权限，防止误修改源数据
        - Windows 平台或权限不足时自动降级为复制模式
        - 复制模式使用 shutil.copytree，支持增量复制
    """
    from utils.logger_system import log_msg

    # 验证源目录
    if not src.exists():
        error_msg = f"源目录不存在: {src}"
        log_msg("ERROR", error_msg)
        raise FileNotFoundError(error_msg)

    if not src.is_dir():
        error_msg = f"源路径不是目录: {src}"
        log_msg("ERROR", error_msg)
        raise NotADirectoryError(error_msg)

    # 确保目标父目录存在
    dst.parent.mkdir(parents=True, exist_ok=True)

    # 如果目标已存在且是符号链接，先删除
    if dst.exists() and dst.is_symlink():
        dst.unlink()
        log_msg("INFO", f"删除已存在的符号链接: {dst}")

    if use_symlinks:
        try:
            # 创建符号链接（使用绝对路径）
            src_abs = src.resolve()
            dst.symlink_to(src_abs, target_is_directory=True)
            log_msg("INFO", f"创建符号链接: {dst} -> {src_abs}")

            # 设置只读权限（仅对 symlink 本身生效，实际文件由源目录控制）
            # 注意: symlink 的权限在大多数系统上不可修改，这里主要是标记意图
            try:
                # 在支持的平台上尝试设置 symlink 的只读属性
                if hasattr(os, "lchmod"):
                    os.lchmod(dst, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
                log_msg("INFO", f"符号链接已设置为只读: {dst}")
            except (OSError, AttributeError) as e:
                # 某些平台不支持 lchmod，忽略错误
                log_msg("WARNING", f"无法设置符号链接只读权限（平台不支持）: {e}")

        except (OSError, NotImplementedError) as e:
            # 符号链接失败（Windows 权限问题或平台不支持），降级为复制
            log_msg("WARNING", f"符号链接失败（{e}），降级为复制模式")
            use_symlinks = False

    if not use_symlinks:
        # 复制模式
        if dst.exists():
            log_msg("WARNING", f"目标目录已存在，将进行增量复制: {dst}")

        shutil.copytree(src, dst, dirs_exist_ok=True)
        log_msg("INFO", f"数据复制完成: {src} -> {dst}")

        # 设置复制后的目录为只读
        try:
            _set_readonly_recursive(dst)
            log_msg("INFO", f"目录已设置为只读: {dst}")
        except Exception as e:
            log_msg("WARNING", f"设置只读权限失败: {e}")


def _set_readonly_recursive(path: Path) -> None:
    """递归设置目录及其所有内容为只读。

    Args:
        path: 目标路径（目录或文件）

    实现细节:
        - 移除写权限（owner, group, others）
        - 保留读权限和执行权限（对目录）
    """
    if path.is_file():
        # 文件: 设置为只读
        current_mode = path.stat().st_mode
        readonly_mode = current_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
        path.chmod(readonly_mode)
    elif path.is_dir():
        # 目录: 设置为只读+可执行（允许进入目录）
        current_mode = path.stat().st_mode
        readonly_mode = (
            current_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
        ) | stat.S_IXUSR
        path.chmod(readonly_mode)

        # 递归处理子项
        for child in path.iterdir():
            _set_readonly_recursive(child)
