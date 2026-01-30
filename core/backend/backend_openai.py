"""OpenAI 后端实现。

支持 OpenAI GPT 系列和兼容 OpenAI 格式的模型（如 GLM）。
"""

from typing import Any
import openai
from funcy import notnone, select_values

from utils.logger_system import log_msg, log_exception
from .utils import opt_messages_to_list, backoff_create


# 全局客户端实例
_client: openai.OpenAI | None = None


# 需要重试的异常类型
OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


def _setup_openai_client(
    api_key: str | None = None, base_url: str | None = None
) -> openai.OpenAI:
    """初始化 OpenAI 客户端。

    Args:
        api_key: API 密钥（可选，不提供则从环境变量读取）
        base_url: API 基础 URL（可选，用于兼容 GLM 等第三方 API）

    Returns:
        OpenAI 客户端实例
    """
    global _client
    if _client is None or api_key is not None or base_url is not None:
        kwargs = {"api_key": api_key, "max_retries": 0}
        if base_url is not None:
            kwargs["base_url"] = base_url
        _client = openai.OpenAI(**kwargs)
    return _client


def query(
    system_message: str | None,
    user_message: str | None,
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    **model_kwargs: Any,
) -> str:
    """调用 OpenAI API（或兼容的 API）。

    Args:
        system_message: 系统消息
        user_message: 用户消息
        model: 模型名称（如 "gpt-4-turbo", "glm-4.6"）
        temperature: 采样温度
        max_tokens: 最大生成 token 数
        api_key: API 密钥（从 Config 传入）
        base_url: API 基础 URL（用于 GLM 等第三方 API）
        **model_kwargs: 额外的模型参数

    Returns:
        LLM 生成的文本响应

    Raises:
        Exception: API 调用失败时抛出

    示例:
        >>> response = query(
        ...     system_message="You are a helpful assistant",
        ...     user_message="Hello",
        ...     model="gpt-4-turbo",
        ...     api_key="sk-..."
        ... )
    """
    # 初始化客户端
    client = _setup_openai_client(api_key, base_url)

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

    # GLM 模型默认参数
    if model.startswith("glm-") and "max_tokens" not in filtered_kwargs:
        filtered_kwargs["max_tokens"] = 64000
        log_msg("INFO", f"GLM 模型 {model} 使用默认 max_tokens=64000")

    # 构建消息列表
    messages = opt_messages_to_list(system_message, user_message)

    log_msg("INFO", f"调用 OpenAI API: model={model}, messages={len(messages)} 条")

    try:
        # 带重试的 API 调用
        completion = backoff_create(
            client.chat.completions.create,
            OPENAI_TIMEOUT_EXCEPTIONS,
            messages=messages,
            **filtered_kwargs,
        )

        # 提取响应文本
        response_text = completion.choices[0].message.content

        if response_text is None:
            log_msg("ERROR", "API 返回空响应")
            raise ValueError("API 返回空响应")

        log_msg("INFO", f"API 响应成功: {len(response_text)} 字符")
        return response_text

    except OPENAI_TIMEOUT_EXCEPTIONS as e:
        log_exception(e, "OpenAI API 调用失败（重试后仍失败）")
        raise
    except Exception as e:
        log_exception(e, "OpenAI API 调用出现未预期错误")
        raise
