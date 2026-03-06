"""
数据预览生成模块。

基于 aideml/ML-Master 实现，提供详细的 EDA 预览功能。
生成数据集的文本预览，用于插入 LLM Prompt。
支持 CSV、JSON、图像目录、特殊文本文件的预览。
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
# 图像文件扩展名
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
# 音频文件扩展名
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".aif", ".aiff", ".ogg"}
# 长度阈值（从 6000 提升到 8000，给图像/特殊文件预览留空间）
MAX_PREVIEW_LENGTH = 8000


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


def preview_image_dir(dir_path: Path, max_sample: int = 3) -> str:
    """采样图像目录，返回关键统计信息。

    信息包括：图像数量、尺寸、通道数、值范围、格式。
    同时注入 PyTorch ToTensor 归一化警告。

    Args:
        dir_path: 图像目录路径
        max_sample: 最大采样图像数

    Returns:
        图像目录预览文本
    """
    images = [f for f in dir_path.iterdir() if f.suffix.lower() in IMAGE_EXTS]

    if not images:
        return ""

    try:
        from PIL import Image
        import numpy as np

        samples = images[:max_sample]
        sizes = []
        modes = []
        for img_path in samples:
            with Image.open(img_path) as img:
                sizes.append(img.size)
                modes.append(img.mode)

        # 检查值范围（只对第一张采样）
        with Image.open(samples[0]) as img:
            arr = np.array(img)
            val_min, val_max = int(arr.min()), int(arr.max())

        return (
            f"-> Image directory: {len(images)} images, format: {samples[0].suffix}\n"
            f"   Sample sizes: {sizes}, mode: {set(modes)}\n"
            f"   Pixel value range: [{val_min}, {val_max}]\n"
            f"   NOTE: torchvision.transforms.ToTensor() converts [0,255] → [0.0,1.0]. "
            f"Do NOT divide by 255 again after ToTensor()."
        )
    except ImportError:
        return (
            f"-> Image directory: {len(images)} images (PIL not available for preview)"
        )
    except Exception as e:
        log_msg("WARNING", f"图像预览失败: {e}")
        return f"-> Image directory: {len(images)} images (preview failed: {e})"


def preview_audio_dir(dir_path: Path, max_sample: int = 3) -> str:
    """采样音频目录，返回关键统计信息。

    信息包括：文件数量、格式、采样率、时长、通道数。
    librosa 不可用时降级为仅报告文件数量。

    Args:
        dir_path: 音频目录路径
        max_sample: 最大采样文件数

    Returns:
        音频目录预览文本
    """
    audio_files = [f for f in dir_path.iterdir() if f.suffix.lower() in AUDIO_EXTS]

    if not audio_files:
        return ""

    try:
        import librosa

        samples = audio_files[:max_sample]
        durations = []
        srs = []
        channels_list = []
        for audio_path in samples:
            y, sr = librosa.load(audio_path, sr=None, mono=False)
            srs.append(sr)
            duration = librosa.get_duration(y=y, sr=sr)
            durations.append(round(duration, 2))
            # mono: y.ndim == 1, stereo: y.ndim == 2
            ch = 1 if y.ndim == 1 else y.shape[0]
            channels_list.append(ch)

        return (
            f"-> Audio directory: {len(audio_files)} files, format: {samples[0].suffix}\n"
            f"   Sample rates: {set(srs)}, channels: {set(channels_list)}\n"
            f"   Sample durations (sec): {durations}"
        )
    except ImportError:
        return (
            f"-> Audio directory: {len(audio_files)} files "
            f"(librosa not available for preview)"
        )
    except Exception as e:
        log_msg("WARNING", f"音频预览失败: {e}")
        return f"-> Audio directory: {len(audio_files)} files (preview failed: {e})"


def _is_train_dir(dir_path: Path) -> bool:
    """判断目录名是否包含 'train'（不区分大小写）。

    覆盖 train/、train2/、training/ 等常见命名。

    Args:
        dir_path: 目录路径

    Returns:
        是否为训练目录
    """
    return "train" in dir_path.name.lower()


def detect_filename_labels(dir_path: Path, ext_filter: set[str]) -> str:
    """分析文件名中的标签模式，报告类别分布。

    检测两种模式：
    1. 后缀模式: filename_<label>.ext（如 train100_0.aif）
    2. 前缀模式: <label>_filename.ext（如 0_train100.aif）

    仅当文件数 >= 10 且标签基数 <= 20 时报告。

    Args:
        dir_path: 目录路径
        ext_filter: 文件扩展名过滤集合

    Returns:
        标签检测报告文本，无标签时返回空字符串
    """
    files = [f for f in dir_path.iterdir() if f.suffix.lower() in ext_filter]
    if len(files) < 10:
        return ""

    # 提取文件名 stem 的后缀标签（最后一个 _ 后的部分）
    suffix_labels: dict[str, int] = {}
    prefix_labels: dict[str, int] = {}

    for f in files:
        stem = f.stem
        # 后缀模式: name_<label>
        if "_" in stem:
            suffix_part = stem.rsplit("_", 1)[1]
            suffix_labels[suffix_part] = suffix_labels.get(suffix_part, 0) + 1
            # 前缀模式: <label>_name
            prefix_part = stem.split("_", 1)[0]
            prefix_labels[prefix_part] = prefix_labels.get(prefix_part, 0) + 1

    total = len(files)

    # 选择最佳模式（基数最低且 <= 20 的模式）
    for pattern_name, labels in [("suffix", suffix_labels), ("prefix", prefix_labels)]:
        if not labels or len(labels) > 20:
            continue

        # 构建分布字符串
        sorted_labels = sorted(labels.items(), key=lambda x: -x[1])
        dist_parts = []
        for label, count in sorted_labels:
            pct = count / total * 100
            dist_parts.append(f"'{label}': {count} ({pct:.1f}%)")
        dist_str = ", ".join(dist_parts)

        # 检查是否存在类别不平衡（最大类 > 70%）
        max_pct = max(count / total for count in labels.values())
        imbalance_warning = ""
        if max_pct > 0.7:
            imbalance_warning = (
                "\n   IMPORTANT: Handle class imbalance "
                "(e.g., class_weight, oversampling)."
            )

        pattern_desc = (
            f"_{'{'}label{'}'}" if pattern_name == "suffix" else f"{'{'}label{'}'}_"
        )
        return (
            f"   ⚠ LABEL DETECTED in filenames ({pattern_name} pattern '{pattern_desc}'):\n"
            f"   Distribution: {dist_str}{imbalance_warning}"
        )

    return ""


def preview_special_file(file_path: Path, max_lines: int = 5) -> str:
    """预览非标准文本文件（.txt, .tsv 等）。

    读取前 N 行，推断分隔符和列数。

    Args:
        file_path: 文件路径
        max_lines: 最大读取行数

    Returns:
        文件预览文本
    """
    try:
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            lines = [f.readline() for _ in range(max_lines)]

        # 过滤空行
        lines = [line for line in lines if line.strip()]
        if not lines:
            return f"-> {file_path.name}: empty file"

        # 推断分隔符
        first_line = lines[0].strip()
        sep_to_repr = {"\t": "\\t", ",": ",", " ": " "}
        for sep, name in [("\t", "tab"), (",", "comma"), (" ", "space")]:
            if sep in first_line:
                num_cols = len(first_line.split(sep))
                preview = "\n".join(line.rstrip() for line in lines)
                sep_repr = sep_to_repr[sep]
                return (
                    f"-> {file_path.name}: {name}-separated, ~{num_cols} columns\n"
                    f"   Usage: pd.read_csv('{file_path.name}', sep='{sep_repr}')\n"
                    f"   First {len(lines)} lines:\n```\n{preview}\n```"
                )

        preview = "".join(lines)
        return (
            f"-> {file_path.name}: unstructured text, first lines:\n```\n{preview}\n```"
        )
    except Exception as e:
        return f"-> {file_path.name}: (cannot read: {e})"


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

    # 已预览的图像/音频目录（避免重复）
    previewed_image_dirs = set()
    previewed_audio_dirs: set[Path] = set()

    if include_file_details:
        for fn in _walk(base_path):
            file_name = str(fn.relative_to(base_path))

            if fn.suffix == ".csv":
                out.append(preview_csv(fn, file_name, simple=simple))
            elif fn.suffix == ".json":
                out.append(preview_json(fn, file_name))
            elif fn.suffix.lower() in IMAGE_EXTS:
                # 图像文件：对其所在目录做一次采样预览
                img_dir = fn.parent
                if img_dir not in previewed_image_dirs:
                    previewed_image_dirs.add(img_dir)
                    img_preview = preview_image_dir(img_dir)
                    if img_preview:
                        rel_dir = str(img_dir.relative_to(base_path))
                        block = f"[{rel_dir}/]\n{img_preview}"
                        if _is_train_dir(img_dir):
                            label_info = detect_filename_labels(img_dir, IMAGE_EXTS)
                            if label_info:
                                block += f"\n{label_info}"
                        out.append(block)
            elif fn.suffix.lower() in AUDIO_EXTS:
                # 音频文件：对其所在目录做一次采样预览
                audio_dir = fn.parent
                if audio_dir not in previewed_audio_dirs:
                    previewed_audio_dirs.add(audio_dir)
                    audio_preview = preview_audio_dir(audio_dir)
                    if audio_preview:
                        rel_dir = str(audio_dir.relative_to(base_path))
                        block = f"[{rel_dir}/]\n{audio_preview}"
                        if _is_train_dir(audio_dir):
                            label_info = detect_filename_labels(audio_dir, AUDIO_EXTS)
                            if label_info:
                                block += f"\n{label_info}"
                        out.append(block)
            elif fn.suffix in {".txt", ".tsv"} and get_file_len_size(fn)[0] >= 30:
                # 较大的特殊文本文件：推断格式
                out.append(preview_special_file(fn))
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

    # 自动长度控制（使用 MAX_PREVIEW_LENGTH 阈值）
    if len(result) > MAX_PREVIEW_LENGTH and not simple:
        log_msg("INFO", "预览过长，自动降级到简化模式")
        return generate(
            base_path, include_file_details=include_file_details, simple=True
        )

    if len(result) > MAX_PREVIEW_LENGTH and simple:
        log_msg("WARNING", "预览仍过长，截断输出")
        return result[:MAX_PREVIEW_LENGTH] + "\n... (truncated)"

    return result
