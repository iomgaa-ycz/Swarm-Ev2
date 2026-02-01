"""系统信息获取工具模块。

提供自动检测和获取系统硬件配置信息以及 Conda 环境信息的功能。
参考: Reference/Swarm-Evo/utils/system_info.py（MVP 简化版）

关键设计决策:
    - 任何获取失败都返回合理默认值，而非抛出异常
    - macOS 平台返回 Apple Silicon 相关信息
    - 非 Mac 且无 GPU 时返回 "无 GPU"
"""

import json
import os
import platform
import subprocess
from collections import Counter
from typing import Any, Dict, Optional

from utils.logger_system import log_msg


def get_cpu_count() -> int:
    """获取 CPU 核心数。

    Returns:
        CPU 核心数，失败时返回 1
    """
    try:
        count = os.cpu_count()
        if count is None:
            log_msg("WARNING", "os.cpu_count() 返回 None，使用默认值 1")
            return 1
        return count
    except Exception as e:
        log_msg("WARNING", f"获取 CPU 核心数失败: {e}，使用默认值 1")
        return 1


def get_memory_info() -> Dict[str, float]:
    """获取内存信息（单位：GB）。

    Returns:
        包含 total 和 available 的字典，失败时返回默认值
    """
    default_value = {"total": 8.0, "available": 8.0}

    # 优先尝试 Linux /proc/meminfo
    if platform.system() == "Linux":
        try:
            if os.path.exists("/proc/meminfo"):
                with open("/proc/meminfo") as f:
                    meminfo = f.read()

                for line in meminfo.split("\n"):
                    if line.startswith("MemTotal:"):
                        mem_kb = int(line.split()[1])
                        mem_gb = mem_kb / (1024 * 1024)
                        return {
                            "total": round(mem_gb, 2),
                            "available": round(mem_gb, 2),
                        }
        except Exception as e:
            log_msg("WARNING", f"读取 /proc/meminfo 失败: {e}")

    # 尝试使用 psutil（如果已安装）
    try:
        import psutil

        mem = psutil.virtual_memory()
        return {
            "total": round(mem.total / (1024**3), 2),
            "available": round(mem.available / (1024**3), 2),
        }
    except ImportError:
        log_msg("WARNING", "psutil 未安装，使用默认内存值")
    except Exception as e:
        log_msg("WARNING", f"psutil 获取内存失败: {e}")

    return default_value


def _is_macos() -> bool:
    """检测是否为 macOS 平台。

    Returns:
        是否为 macOS
    """
    return platform.system() == "Darwin"


def _get_apple_silicon_info() -> Optional[str]:
    """获取 Apple Silicon 信息（仅 macOS）。

    Returns:
        Apple 芯片描述（如 "Apple M1"），非 Apple Silicon 返回 None
    """
    if not _is_macos():
        return None

    try:
        # 使用 sysctl 获取芯片信息
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0 and result.stdout.strip():
            brand = result.stdout.strip()
            # 检查是否为 Apple Silicon
            if "Apple" in brand:
                return brand
    except Exception as e:
        log_msg("WARNING", f"获取 Apple Silicon 信息失败: {e}")

    # 备选方案：检查架构
    try:
        arch = platform.machine()
        if arch == "arm64":
            return "Apple Silicon (arm64)"
    except Exception:
        pass

    return None


def get_gpu_info() -> Optional[str]:
    """获取 GPU 信息。

    返回策略:
        - NVIDIA GPU: 返回 GPU 名称和显存（如 "NVIDIA RTX 3090 24GB"）
        - macOS Apple Silicon: 返回 Apple 芯片信息（如 "Apple M1 (GPU integrated)"）
        - macOS 无 Apple Silicon: 返回 None
        - 其他平台无 GPU: 返回 None

    Returns:
        GPU 描述字符串，无 GPU 时返回 None
    """
    # 优先尝试 NVIDIA GPU
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split("\n")
            gpu_list = []

            for line in lines:
                if line.strip():
                    parts = line.split(",")
                    if len(parts) >= 2:
                        gpu_name = parts[0].strip()
                        gpu_memory = parts[1].strip()
                        gpu_list.append(f"{gpu_name} {gpu_memory}")

            if gpu_list:
                if len(gpu_list) == 1:
                    return gpu_list[0]
                else:
                    return f"{len(gpu_list)}x {gpu_list[0]}"

    except FileNotFoundError:
        # nvidia-smi 不存在，正常情况
        pass
    except subprocess.TimeoutExpired:
        log_msg("WARNING", "nvidia-smi 命令超时")
    except Exception as e:
        log_msg("WARNING", f"nvidia-smi 执行失败: {e}")

    # macOS 平台：尝试获取 Apple Silicon 信息
    if _is_macos():
        apple_info = _get_apple_silicon_info()
        if apple_info:
            return f"{apple_info} (GPU integrated)"
        # macOS 但非 Apple Silicon，返回 None
        return None

    # 非 macOS 且无 NVIDIA GPU
    return None


def get_hardware_description() -> str:
    """获取完整的硬件配置描述字符串。

    Returns:
        格式化的硬件配置描述，例如：
            - "CPU: 8 cores, RAM: 32GB, GPU: NVIDIA RTX 3090 24GB"
            - "CPU: 8 cores, RAM: 16GB, GPU: Apple M1 (GPU integrated)"
            - "CPU: 4 cores, RAM: 8GB"（无 GPU）
    """
    cpu_count = get_cpu_count()
    memory_info = get_memory_info()
    gpu_info = get_gpu_info()

    description_parts = [
        f"CPU: {cpu_count} cores",
        f"RAM: {int(memory_info['total'])}GB",
    ]

    if gpu_info:
        description_parts.append(f"GPU: {gpu_info}")

    return ", ".join(description_parts)


def get_conda_packages(env_name: Optional[str] = None) -> str:
    """获取 Conda 环境中所有包的摘要描述。

    Args:
        env_name: Conda 环境名称，None 则使用当前激活的环境

    Returns:
        适合 LLM 理解的自然语言描述

    注意:
        失败时返回默认描述，不抛出异常
    """
    default_description = (
        "Conda environment contains common ML packages including numpy, pandas, "
        "scikit-learn, torch, etc. All packages are pre-installed."
    )

    try:
        # 构建命令
        if env_name:
            cmd = ["conda", "list", "--name", env_name, "--json"]
        else:
            cmd = ["conda", "list", "--json"]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)

        if result.returncode != 0:
            log_msg("WARNING", f"conda list 命令失败: {result.stderr}")
            return default_description

        # 解析 JSON 输出
        try:
            packages_data: list[Dict[str, Any]] = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            log_msg("WARNING", f"解析 conda list 输出失败: {e}")
            return default_description

        if not isinstance(packages_data, list):
            return default_description

        # 统计信息
        python_info: Optional[str] = None
        channel_counter: Counter[str] = Counter()
        highlighted_packages: list[tuple[str, str]] = []

        # 核心 ML 包列表
        core_packages = {
            "numpy",
            "pandas",
            "scipy",
            "scikit-learn",
            "torch",
            "torchvision",
            "tensorflow",
            "xgboost",
            "lightgbm",
        }

        for package in packages_data:
            name = package.get("name", "")
            version = package.get("version", "unknown")
            channel = package.get("channel", "unknown")

            channel_counter[channel] += 1

            if name.lower() == "python":
                python_info = f"python {version}"
            elif name.lower() in core_packages:
                highlighted_packages.append((name, f"{name} {version}"))

        # 构建描述
        environment_name = env_name or "current"
        package_total = len(packages_data)

        description_parts = []

        # Python 版本
        python_str = python_info or "python (version unknown)"
        description_parts.append(
            f"Conda environment '{environment_name}' contains {package_total} packages, "
            f"Python version is {python_str}."
        )

        # 核心包列表
        if highlighted_packages:
            # 按名称排序，取前 6 个
            sorted_packages = sorted(set(highlighted_packages), key=lambda x: x[0])[:6]
            packages_str = ", ".join([sig for _, sig in sorted_packages])
            remaining = len(highlighted_packages) - len(sorted_packages)

            if remaining > 0:
                packages_str += f" (+{remaining} more)"

            description_parts.append(f"Key ML packages: {packages_str}.")

        # PyTorch 优先提示
        has_torch = any(name.lower() == "torch" for name, _ in highlighted_packages)
        has_tf = any(name.lower() == "tensorflow" for name, _ in highlighted_packages)

        if has_torch:
            description_parts.append(
                "PyTorch is available and recommended for neural network tasks."
            )
        elif has_tf:
            description_parts.append(
                "TensorFlow is available for neural network tasks."
            )

        return " ".join(description_parts)

    except subprocess.TimeoutExpired:
        log_msg("WARNING", "conda list 命令超时")
        return default_description
    except FileNotFoundError:
        log_msg("WARNING", "conda 命令不可用")
        return default_description
    except Exception as e:
        log_msg("WARNING", f"获取 conda 包信息时发生未知错误: {e}")
        return default_description
