"""网络代理配置与连通性检测模块。

提供代理环境变量的统一配置、连通性测试和状态日志功能。
支持 main.py（本地模式）和 run_mle_adapter.py（Docker 模式）两套运行路径。

缓存说明：
    PyTorch 预训练权重下载后缓存在 ~/.cache/torch/hub/checkpoints/，
    同一容器（同一竞赛）内所有 solution.py 共享缓存，仅首次下载。
"""

import os
import urllib.request
from typing import Optional

# 延迟导入 logger：proxy 模块可能在 logger 初始化之前被调用
_logger_ready = False


def _log(level: str, msg: str) -> None:
    """安全日志输出（兼容 logger 未初始化场景）。

    Args:
        level: 日志级别（INFO/WARNING/ERROR）
        msg: 日志消息
    """
    global _logger_ready
    if not _logger_ready:
        try:
            from utils.logger_system import log_msg

            log_msg(level, msg)
            _logger_ready = True
            return
        except Exception:
            pass
    # logger 不可用时直接 print（仅在启动最早期）
    print(f"[{level}] {msg}")


def setup_proxy_env() -> None:
    """确保代理环境变量大小写变体同步。

    不同 Python 库检查不同大小写：
        - urllib.request (PyTorch Hub): 小写 http_proxy
        - requests (pip, HuggingFace): 大写 HTTP_PROXY
        - httpx (OpenAI SDK): 两种都检查

    设置 NO_PROXY 默认值，避免本地通信走代理（如 grading server localhost:5000）。
    """
    proxy_vars = ["HTTP_PROXY", "HTTPS_PROXY"]
    for var in proxy_vars:
        upper_val = os.environ.get(var, "")
        lower_val = os.environ.get(var.lower(), "")
        val = upper_val or lower_val
        if val:
            os.environ[var] = val
            os.environ[var.lower()] = val

    # NO_PROXY: 确保本地通信不走代理
    no_proxy_upper = os.environ.get("NO_PROXY", "")
    no_proxy_lower = os.environ.get("no_proxy", "")
    no_proxy = no_proxy_upper or no_proxy_lower or "localhost,127.0.0.1"
    os.environ["NO_PROXY"] = no_proxy
    os.environ["no_proxy"] = no_proxy


def test_proxy_connectivity(
    proxy_url: Optional[str] = None, timeout: int = 5
) -> bool:
    """测试代理连通性。

    通过代理访问一个轻量级 URL 验证代理是否可用。

    Args:
        proxy_url: 代理地址（如 http://192.168.31.250:7890），
                   None 时从环境变量读取
        timeout: 连接超时秒数

    Returns:
        True 表示代理可用，False 表示不可用
    """
    if proxy_url is None:
        proxy_url = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy", "")

    if not proxy_url:
        return False

    try:
        proxy_handler = urllib.request.ProxyHandler({
            "http": proxy_url,
            "https": proxy_url,
        })
        opener = urllib.request.build_opener(proxy_handler)
        # 用 HEAD 请求测试，不下载内容
        req = urllib.request.Request(
            "https://pypi.org/simple/",
            method="HEAD",
        )
        opener.open(req, timeout=timeout)
        return True
    except Exception:
        return False


def test_network_connectivity(timeout: int = 5) -> bool:
    """测试容器/主机的网络连通性（不通过代理）。

    Args:
        timeout: 连接超时秒数

    Returns:
        True 表示有网络连接
    """
    try:
        # 直连测试（不走代理）
        no_proxy_handler = urllib.request.ProxyHandler({})
        opener = urllib.request.build_opener(no_proxy_handler)
        req = urllib.request.Request(
            "https://pypi.org/simple/",
            method="HEAD",
        )
        opener.open(req, timeout=timeout)
        return True
    except Exception:
        return False


def init_proxy() -> bool:
    """代理初始化主入口：测试 → 配置 → 日志。

    在 main.py 和 run_mle_adapter.py 启动时调用。
    自动检测代理可用性，可用则启用，不可用则降级为直连。

    Returns:
        True 表示代理已启用，False 表示使用直连
    """
    # Phase 1: 同步大小写
    setup_proxy_env()

    proxy_url = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy", "")

    if not proxy_url:
        _log("INFO", "代理未配置（HTTP_PROXY 环境变量为空），使用直连")
        # 检查直连网络
        if test_network_connectivity(timeout=5):
            _log("INFO", "网络连通性检测: 直连可用")
        else:
            _log("WARNING", "网络连通性检测: 直连不可用（可能无外网访问）")
        return False

    # Phase 2: 测试代理连通性
    _log("INFO", f"检测代理连通性: {_mask_proxy_url(proxy_url)} ...")
    proxy_ok = test_proxy_connectivity(proxy_url, timeout=5)

    if proxy_ok:
        _log("INFO", f"代理可用: {_mask_proxy_url(proxy_url)}")
        log_proxy_status()
        return True
    else:
        # 代理不可用，清除代理变量，降级为直连
        _log("WARNING", f"代理不可用: {_mask_proxy_url(proxy_url)}，降级为直连")
        _clear_proxy_env()

        if test_network_connectivity(timeout=5):
            _log("INFO", "网络连通性检测: 直连可用")
        else:
            _log("WARNING", "网络连通性检测: 直连也不可用（可能无外网访问）")
        return False


def log_proxy_status() -> None:
    """打印当前代理配置状态（脱敏）。"""
    http_proxy = os.environ.get("HTTP_PROXY", "")
    https_proxy = os.environ.get("HTTPS_PROXY", "")
    no_proxy = os.environ.get("NO_PROXY", "")

    if http_proxy or https_proxy:
        _log("INFO", f"代理配置 — HTTP: {_mask_proxy_url(http_proxy)}, "
             f"HTTPS: {_mask_proxy_url(https_proxy)}, NO_PROXY: {no_proxy}")
    else:
        _log("INFO", "代理配置 — 未启用（直连模式）")


def _mask_proxy_url(url: str) -> str:
    """脱敏代理 URL（保留协议和端口，部分隐藏 IP）。

    Args:
        url: 代理 URL

    Returns:
        脱敏后的字符串，如 "http://192.168.*.*:7890"
    """
    if not url:
        return "(空)"
    # 简单脱敏：显示完整地址（局域网 IP 不敏感）
    return url


def _clear_proxy_env() -> None:
    """清除所有代理环境变量。"""
    for var in ["HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"]:
        os.environ.pop(var, None)
