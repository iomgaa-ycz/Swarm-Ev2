"""
数据预览生成模块。

基于 aideml/ML-Master 实现，提供详细的 EDA 预览功能。
生成数据集的文本预览，用于插入 LLM Prompt。
"""

import json
from pathlib import Path

import humanize
import pandas as pd
from genson import SchemaBuilder
from pandas.api.types import is_numeric_dtype

from utils.logger_system import log_msg

# 代码文件类型（markdown 包裹）
code_files = {".py", ".sh", ".yaml", ".yml", ".md", ".html", ".xml", ".log", ".rst"}
# 文本文件类型
plaintext_files = {".txt", ".csv", ".json", ".tsv"} | code_files


def get_file_len_size(f: Path) -> tuple[int, str]:
    """计算文件大小。

    Args:
        f: 文件路径

    Returns:
        (行数/字节数, 人类可读字符串)
        - 文本文件：返回行数 + 文件大小（便于 LLM 推理文件用途）
        - 二进制文件：返回字节数
    """
    if f.suffix in plaintext_files:
        try:
            num_lines = sum(1 for _ in open(f, encoding="utf-8", errors="ignore"))
            file_size = humanize.naturalsize(f.stat().st_size)
            return num_lines, f"{num_lines} lines, {file_size}"
        except Exception as e:
            log_msg("WARNING", f"读取文件 {f} 行数失败: {e}")
            s = f.stat().st_size
            return s, humanize.naturalsize(s)
    else:
        s = f.stat().st_size
        return s, humanize.naturalsize(s)


def file_tree(path: Path, depth: int = 0, max_dirs: int = 20) -> str:
    """生成目录树结构。

    Args:
        path: 目录路径
        depth: 当前递归深度
        max_dirs: 最大显示子目录数（避免过长）

    Returns:
        树形结构文本
    """
    result = []
    try:
        files = [p for p in Path(path).iterdir() if not p.is_dir()]
        dirs = [p for p in Path(path).iterdir() if p.is_dir()]
    except PermissionError:
        return f"{' ' * depth * 4}[Permission Denied]"

    # 显示文件（限制数量）
    max_n = 4 if len(files) > 30 else 8
    for p in sorted(files)[:max_n]:
        result.append(f"{' ' * depth * 4}{p.name} ({get_file_len_size(p)[1]})")
    if len(files) > max_n:
        result.append(f"{' ' * depth * 4}... and {len(files) - max_n} other files")

    # 递归显示子目录（限制数量）
    for p in sorted(dirs)[:max_dirs]:
        result.append(f"{' ' * depth * 4}{p.name}/")
        result.append(file_tree(p, depth + 1, max_dirs))

    if len(dirs) > max_dirs:
        result.append(
            f"{' ' * depth * 4}... and {len(dirs) - max_dirs} other subfolders"
        )

    return "\n".join(result)


def _walk(path: Path):
    """递归遍历目录（类似 os.walk）。

    Args:
        path: 目录路径

    Yields:
        文件路径
    """
    for p in sorted(Path(path).iterdir()):
        if p.is_dir():
            yield from _walk(p)
            continue
        yield p


def preview_csv(p: Path, file_name: str, simple: bool = True) -> str:
    """生成 CSV 文件的文本预览。

    Args:
        p: CSV 文件路径
        file_name: 文件名（用于显示）
        simple: 是否使用简化模式

    Returns:
        格式化的预览文本

    简化模式 (simple=True):
        - shape + 前 15 列列名

    详细模式 (simple=False):
        - shape + 每列详细统计：
          - bool 列：True/False 百分比
          - 低基数列（< 10 个唯一值）：显示所有唯一值
          - 数值列：min-max 范围 + 缺失值数量
          - 字符串列：唯一值数量 + 前 4 个常见值
    """
    try:
        # 先获取总行数（快速方法，减去 CSV 表头行）
        total_rows = get_file_len_size(p)[0] - 1

        # 只读取前 1000 行用于预览（避免大文件超时）
        df = pd.read_csv(p, nrows=1000)
    except Exception as e:
        log_msg("WARNING", f"读取 CSV 文件 {file_name} 失败: {e}")
        return f"-> {file_name} (无法读取: {e})"

    out = []
    out.append(f"-> {file_name} has {total_rows} rows and {df.shape[1]} columns.")

    if simple:
        # 简化模式：只显示列名
        cols = df.columns.tolist()
        sel_cols = 15
        cols_str = ", ".join(cols[:sel_cols])
        res = f"The columns are: {cols_str}"
        if len(cols) > sel_cols:
            res += f"... and {len(cols) - sel_cols} more columns"
        out.append(res)
    else:
        # 详细模式：每列统计
        out.append("Here is some information about the columns:")
        for col in sorted(df.columns):
            dtype = df[col].dtype
            name = f"{col} ({dtype})"

            nan_count = df[col].isnull().sum()

            if dtype == "bool":
                v = df[col][df[col].notnull()].mean()
                out.append(f"{name} is {v * 100:.2f}% True, {100 - v * 100:.2f}% False")
            elif df[col].nunique() < 10:
                out.append(
                    f"{name} has {df[col].nunique()} unique values: {df[col].unique().tolist()}"
                )
            elif is_numeric_dtype(df[col]):
                out.append(
                    f"{name} has range: {df[col].min():.2f} - {df[col].max():.2f}, {nan_count} nan values"
                )
            elif dtype == "object":
                out.append(
                    f"{name} has {df[col].nunique()} unique values. "
                    f"Some example values: {df[col].value_counts().head(4).index.tolist()}"
                )

    return "\n".join(out)


def preview_json(p: Path, file_name: str) -> str:
    """生成 JSON 文件的预览（自动生成 JSON Schema）。

    Args:
        p: JSON 文件路径
        file_name: 文件名

    Returns:
        JSON Schema 文本
    """
    builder = SchemaBuilder()
    try:
        with open(p, encoding="utf-8") as f:
            first_line = f.readline().strip()

            try:
                first_object = json.loads(first_line)

                if not isinstance(first_object, dict):
                    raise json.JSONDecodeError(
                        "The first line isn't JSON", first_line, 0
                    )

                # 检查是否是 JSONL 文件（第二行非空）
                second_line = f.readline().strip()
                if second_line:
                    f.seek(0)  # 重置并逐行读取
                    for line in f:
                        builder.add_object(json.loads(line.strip()))
                else:
                    builder.add_object(first_object)

            except json.JSONDecodeError:
                # 首行不是 JSON，可能是 prettified，读取完整文件
                f.seek(0)
                builder.add_object(json.load(f))

        return f"-> {file_name} has auto-generated json schema:\n" + builder.to_json(
            indent=2
        )

    except Exception as e:
        log_msg("WARNING", f"读取 JSON 文件 {file_name} 失败: {e}")
        return f"-> {file_name} (无法读取: {e})"


def generate(
    base_path: Path, include_file_details: bool = True, simple: bool = False
) -> str:
    """生成目录的文本预览。

    Args:
        base_path: 基础路径（通常是 workspace/input/）
        include_file_details: 是否包含文件详情
        simple: 是否使用简化模式

    Returns:
        格式化的预览文本

    自动长度控制：
        - 输出 > 6000 字符：自动降级到 simple 模式
        - 仍 > 6000 字符：截断并添加 "... (truncated)"
    """
    # 添加强调性声明，确保 LLM 使用正确的文件名
    header = "**IMPORTANT: Use the exact file names listed below in your code.**\n\n"
    tree = f"{header}```\n{file_tree(base_path)}\n```"
    out = [tree]

    if include_file_details:
        for fn in _walk(base_path):
            file_name = str(fn.relative_to(base_path))

            if fn.suffix == ".csv":
                out.append(preview_csv(fn, file_name, simple=simple))
            elif fn.suffix == ".json":
                out.append(preview_json(fn, file_name))
            elif fn.suffix in plaintext_files:
                # 小文件直接显示内容
                if get_file_len_size(fn)[0] < 30:
                    try:
                        with open(fn, encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                            if fn.suffix in code_files:
                                content = f"```\n{content}\n```"
                            out.append(f"-> {file_name} has content:\n\n{content}")
                    except Exception as e:
                        log_msg("WARNING", f"读取文件 {file_name} 失败: {e}")

    result = "\n\n".join(out)

    # 自动长度控制
    if len(result) > 6_000 and not simple:
        log_msg("INFO", "预览过长，自动降级到简化模式")
        return generate(
            base_path, include_file_details=include_file_details, simple=True
        )

    if len(result) > 6_000 and simple:
        log_msg("WARNING", "预览仍过长，截断输出")
        return result[:6_000] + "\n... (truncated)"

    return result
