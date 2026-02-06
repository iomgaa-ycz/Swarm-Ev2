"""OpenAI 后端实现。

支持 OpenAI GPT 系列和兼容 OpenAI 格式的模型（如 GLM）。
支持多 Key 轮询：逗号分隔的 api_key 自动启用 Round-Robin + 限流切换。
"""

from typing import Any
import openai
from funcy import notnone, select_values

from utils.logger_system import log_msg, log_exception
from .utils import opt_messages_to_list, backoff_create
from .key_pool import get_pool


# 客户端缓存：按 (api_key, base_url) 复用
_clients: dict[tuple[str, str], openai.OpenAI] = {}

# 非限流类重试异常（连接、超时、服务器错误）
_RETRY_EXCEPTIONS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)

# 全部可重试异常（含限流，单 Key 场景使用）
OPENAI_TIMEOUT_EXCEPTIONS = (*_RETRY_EXCEPTIONS, openai.RateLimitError)


def _get_client(
    api_key: str, base_url: str | None = None
) -> openai.OpenAI:
    """获取或创建 OpenAI 客户端（按 key+url 缓存）。

    Args:
        api_key: API 密钥
        base_url: API 基础 URL（用于 GLM 等第三方 API）

    Returns:
        OpenAI 客户端实例
    """
    cache_key = (api_key, base_url or "")
    if cache_key not in _clients:
        kwargs: dict[str, Any] = {"api_key": api_key, "max_retries": 0}
        if base_url is not None:
            kwargs["base_url"] = base_url
        _clients[cache_key] = openai.OpenAI(**kwargs)
    return _clients[cache_key]


def query(
    system_message: str | None,
    user_message: str | None,
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    tools: list[dict] | None = None,
    tool_choice: dict | str | None = None,
    **model_kwargs: Any,
) -> str:
    """调用 OpenAI API（或兼容的 API），支持 Function Calling 和多 Key 轮询。

    多 Key 模式：api_key 含逗号时自动启用。RateLimitError 触发时标记当前 Key
    冷却并切换下一个，其他异常仍走指数退避重试。

    Args:
        system_message: 系统消息
        user_message: 用户消息
        model: 模型名称（如 "gpt-4-turbo", "glm-4.6"）
        temperature: 采样温度
        max_tokens: 最大生成 token 数
        api_key: API 密钥，支持逗号分隔多个（如 "key1,key2,key3"）
        base_url: API 基础 URL（用于 GLM 等第三方 API）
        tools: Function Calling 工具列表（可选）
        tool_choice: 工具选择策略（可选）
        **model_kwargs: 额外的模型参数

    Returns:
        - 无 tools: 返回 LLM 响应文本
        - 有 tools: 返回 tool call 的参数 JSON 字符串
    """
    pool = get_pool(api_key or "")

    # 构建参数字典
    filtered_kwargs: dict[str, Any] = select_values(
        notnone,
        {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tools": tools,
            "tool_choice": tool_choice,
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

    # 多 Key: RateLimitError 由外层循环处理；单 Key: 交给 backoff 重试
    retry_exc = _RETRY_EXCEPTIONS if pool.size > 1 else OPENAI_TIMEOUT_EXCEPTIONS
    last_err: Exception | None = None

    for attempt in range(pool.size):
        current_key = pool.get_key()
        client = _get_client(current_key, base_url)

        try:
            completion = backoff_create(
                client.chat.completions.create,
                retry_exc,
                messages=messages,
                **filtered_kwargs,
            )

            message = completion.choices[0].message

            # Function Calling 响应
            if tools is not None and message.tool_calls:
                tool_call = message.tool_calls[0]
                arguments = tool_call.function.arguments
                log_msg(
                    "INFO",
                    f"Function Calling 响应: {tool_call.function.name}, {len(arguments)} 字符",
                )
                return arguments

            # 普通文本响应
            response_text = message.content
            if response_text is None:
                log_msg("ERROR", "API 返回空响应")
                raise ValueError("API 返回空响应")

            log_msg("INFO", f"API 响应成功: {len(response_text)} 字符")
            return response_text

        except openai.RateLimitError as e:
            pool.mark_rate_limited(current_key)
            last_err = e
            log_msg(
                "WARNING",
                f"Key ...{current_key[-4:]} 限流，切换下一个 ({attempt + 1}/{pool.size})",
            )
            continue

        except OPENAI_TIMEOUT_EXCEPTIONS as e:
            log_exception(e, "OpenAI API 调用失败（重试后仍失败）")
            raise

        except Exception as e:
            log_exception(e, "OpenAI API 调用出现未预期错误")
            raise

    # 所有 Key 均被限流
    log_msg("ERROR", f"所有 {pool.size} 个 Key 均被限流")
    raise last_err  # type: ignore[misc]
