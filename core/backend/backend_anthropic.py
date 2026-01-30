"""Anthropic 后端实现。

支持 Claude 系列模型。
"""

from typing import Any
import anthropic
from funcy import notnone, select_values

from utils.logger_system import log_msg, log_exception
from .utils import opt_messages_to_list, backoff_create


# 全局客户端实例
_client: anthropic.Anthropic | None = None


# 需要重试的异常类型
ANTHROPIC_TIMEOUT_EXCEPTIONS = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
)


def _setup_anthropic_client(api_key: str | None = None) -> anthropic.Anthropic:
    """初始化 Anthropic 客户端。

    Args:
        api_key: API 密钥（可选，不提供则从环境变量读取）

    Returns:
        Anthropic 客户端实例
    """
    global _client
    if _client is None or api_key is not None:
        _client = anthropic.Anthropic(api_key=api_key, max_retries=0)
    return _client


def query(
    system_message: str | None,
    user_message: str | None,
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    api_key: str | None = None,
    **model_kwargs: Any,
) -> str:
    """调用 Anthropic API。

    Args:
        system_message: 系统消息
        user_message: 用户消息
        model: 模型名称（如 "claude-3-opus-20240229"）
        temperature: 采样温度
        max_tokens: 最大生成 token 数
        api_key: API 密钥（从 Config 传入）
        **model_kwargs: 额外的模型参数

    Returns:
        LLM 生成的文本响应

    Raises:
        Exception: API 调用失败时抛出

    注意:
        - Anthropic 必须有 user message（无则用 system message 代替）
        - system message 作为单独参数传递，不在 messages 中
        - 默认 max_tokens=4096

    示例:
        >>> response = query(
        ...     system_message="You are a helpful assistant",
        ...     user_message="Hello",
        ...     model="claude-3-opus-20240229",
        ...     api_key="sk-ant-..."
        ... )
    """
    # 初始化客户端
    client = _setup_anthropic_client(api_key)

    # 构建参数字典
    filtered_kwargs: dict[str, Any] = select_values(
        notnone,
        {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **model_kwargs,
        },
    )

    # Anthropic 默认 max_tokens
    if "max_tokens" not in filtered_kwargs:
        filtered_kwargs["max_tokens"] = 4096
        log_msg("INFO", f"Claude 模型 {model} 使用默认 max_tokens=4096")

    # Anthropic 特殊逻辑：必须有 user message
    # 如果只有 system message，将其作为 user message
    if system_message is not None and user_message is None:
        log_msg(
            "INFO",
            "检测到只有 system message，将其转换为 user message（Anthropic 要求）",
        )
        system_message, user_message = None, system_message

    # Anthropic 的 system message 作为单独参数传递
    if system_message is not None:
        filtered_kwargs["system"] = system_message

    # 构建消息列表（不包含 system message）
    messages = opt_messages_to_list(None, user_message)

    log_msg("INFO", f"调用 Anthropic API: model={model}, messages={len(messages)} 条")

    try:
        # 带重试的 API 调用
        message = backoff_create(
            client.messages.create,
            ANTHROPIC_TIMEOUT_EXCEPTIONS,
            messages=messages,
            **filtered_kwargs,
        )

        # 提取响应文本
        if not message.content or len(message.content) == 0:
            log_msg("ERROR", "API 返回空响应")
            raise ValueError("API 返回空响应")

        response_text = message.content[0].text

        log_msg("INFO", f"API 响应成功: {len(response_text)} 字符")
        return response_text

    except ANTHROPIC_TIMEOUT_EXCEPTIONS as e:
        log_exception(e, "Anthropic API 调用失败（重试后仍失败）")
        raise
    except Exception as e:
        log_exception(e, "Anthropic API 调用出现未预期错误")
        raise
