"""后端抽象层模块。

提供统一的 LLM 查询接口，支持多种 LLM 提供商。

支持的提供商:
- OpenAI (GPT 系列: gpt-4-turbo, gpt-3.5-turbo, o1-*)
- GLM (智谱 AI: glm-4.6，兼容 OpenAI 格式)
- Anthropic (Claude 系列: claude-3-opus, claude-3-sonnet, claude-3-haiku)

示例:
    >>> from core.backend import query
    >>> response = query(
    ...     system_message="You are a helpful assistant",
    ...     user_message="Hello",
    ...     model="gpt-4-turbo",
    ...     provider="openai",
    ...     api_key="sk-..."
    ... )
"""

from typing import Any, Callable

from utils.logger_system import log_msg, log_exception
from . import backend_openai, backend_anthropic


# 提供商到查询函数的映射
PROVIDER_TO_QUERY: dict[str, Callable] = {
    "openai": backend_openai.query,
    "anthropic": backend_anthropic.query,
}


def query(
    system_message: str | None,
    user_message: str | None,
    model: str,
    provider: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> str:
    """统一 LLM 查询接口。

    根据 provider 参数选择对应的后端实现。

    Args:
        system_message: 系统消息（定义 AI 角色和行为）
        user_message: 用户消息（用户输入）
        model: 模型名称
        provider: 提供商名称（必填，"openai" 或 "anthropic"）
        temperature: 采样温度（0-1，越高越随机）
        max_tokens: 最大生成 token 数
        api_key: API 密钥（从 Config 传入）
        **kwargs: 额外的模型参数（如 base_url）

    Returns:
        LLM 生成的文本响应

    Raises:
        ValueError: 不支持的 provider
        Exception: API 调用失败

    示例:
        >>> # 使用 GPT-4
        >>> response = query(
        ...     system_message="你是智能助手",
        ...     user_message="1+1=?",
        ...     model="gpt-4-turbo",
        ...     provider="openai",
        ...     api_key="sk-..."
        ... )

        >>> # 使用第三方 OpenAI 兼容 API
        >>> response = query(
        ...     system_message="你是智能助手",
        ...     user_message="介绍 Python",
        ...     model="moonshot-v1-8k",
        ...     provider="openai",
        ...     api_key="sk-...",
        ...     base_url="https://api.moonshot.cn/v1"
        ... )

        >>> # 使用 Claude
        >>> response = query(
        ...     system_message="You are a helpful assistant",
        ...     user_message="Hello",
        ...     model="claude-3-opus-20240229",
        ...     provider="anthropic",
        ...     api_key="sk-ant-..."
        ... )
    """
    try:
        # 验证 provider
        if provider not in PROVIDER_TO_QUERY:
            raise ValueError(
                f"不支持的 provider: {provider}。支持: {list(PROVIDER_TO_QUERY.keys())}"
            )

        log_msg("INFO", f"查询 LLM: model={model}, provider={provider}")

        # 获取对应的查询函数
        query_func = PROVIDER_TO_QUERY[provider]

        # 调用后端
        response = query_func(
            system_message=system_message,
            user_message=user_message,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            **kwargs,
        )

        log_msg("INFO", f"LLM 响应完成: {len(response)} 字符")
        return response

    except ValueError as e:
        log_msg("ERROR", f"Provider 验证失败: {e}")
        raise
    except Exception as e:
        log_exception(e, f"LLM 查询失败 (model={model}, provider={provider})")
        raise


# 导出公共接口
__all__ = ["query", "PROVIDER_TO_QUERY"]
