"""Anthropic 后端实现。

支持 Claude 系列模型。
支持多 Key 轮询：逗号分隔的 api_key 自动启用 Round-Robin + 限流切换。
"""

from typing import Any
import anthropic
from funcy import notnone, select_values

from utils.logger_system import log_msg, log_exception
from .utils import opt_messages_to_list, backoff_create
from .key_pool import get_pool


# 客户端缓存：按 api_key 复用
_clients: dict[str, anthropic.Anthropic] = {}

# 非限流类重试异常
_RETRY_EXCEPTIONS = (
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
)

# 全部可重试异常（含限流，单 Key 场景使用）
ANTHROPIC_TIMEOUT_EXCEPTIONS = (*_RETRY_EXCEPTIONS, anthropic.RateLimitError)


def _get_client(api_key: str) -> anthropic.Anthropic:
    """获取或创建 Anthropic 客户端（按 key 缓存）。

    Args:
        api_key: API 密钥

    Returns:
        Anthropic 客户端实例
    """
    if api_key not in _clients:
        _clients[api_key] = anthropic.Anthropic(api_key=api_key, max_retries=0)
    return _clients[api_key]


def query(
    system_message: str | None,
    user_message: str | None,
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    api_key: str | None = None,
    **model_kwargs: Any,
) -> str:
    """调用 Anthropic API，支持多 Key 轮询。

    多 Key 模式：api_key 含逗号时自动启用。RateLimitError 触发时标记当前 Key
    冷却并切换下一个，其他异常仍走指数退避重试。

    Args:
        system_message: 系统消息
        user_message: 用户消息
        model: 模型名称（如 "claude-3-opus-20240229"）
        temperature: 采样温度
        max_tokens: 最大生成 token 数
        api_key: API 密钥，支持逗号分隔多个
        **model_kwargs: 额外的模型参数

    Returns:
        LLM 生成的文本响应

    注意:
        - Anthropic 必须有 user message（无则用 system message 代替）
        - system message 作为单独参数传递，不在 messages 中
        - 默认 max_tokens=4096
    """
    pool = get_pool(api_key or "")

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
    if system_message is not None and user_message is None:
        log_msg(
            "INFO",
            "检测到只有 system message，将其转换为 user message（Anthropic 要求）",
        )
        system_message, user_message = None, system_message

    # system message 作为单独参数传递
    if system_message is not None:
        filtered_kwargs["system"] = system_message

    # 构建消息列表（不包含 system message）
    messages = opt_messages_to_list(None, user_message)

    log_msg("INFO", f"调用 Anthropic API: model={model}, messages={len(messages)} 条")

    # 多 Key: RateLimitError 由外层循环处理；单 Key: 交给 backoff 重试
    retry_exc = _RETRY_EXCEPTIONS if pool.size > 1 else ANTHROPIC_TIMEOUT_EXCEPTIONS
    last_err: Exception | None = None

    for attempt in range(pool.size):
        current_key = pool.get_key()
        client = _get_client(current_key)

        try:
            message = backoff_create(
                client.messages.create,
                retry_exc,
                messages=messages,
                **filtered_kwargs,
            )

            if not message.content or len(message.content) == 0:
                log_msg("ERROR", "API 返回空响应")
                raise ValueError("API 返回空响应")

            response_text = message.content[0].text
            log_msg("INFO", f"API 响应成功: {len(response_text)} 字符")
            return response_text

        except anthropic.RateLimitError as e:
            pool.mark_rate_limited(current_key)
            last_err = e
            log_msg(
                "WARNING",
                f"Key ...{current_key[-4:]} 限流，切换下一个 ({attempt + 1}/{pool.size})",
            )
            continue

        except ANTHROPIC_TIMEOUT_EXCEPTIONS as e:
            log_exception(e, "Anthropic API 调用失败（重试后仍失败）")
            raise

        except Exception as e:
            log_exception(e, "Anthropic API 调用出现未预期错误")
            raise

    # 所有 Key 均被限流
    log_msg("ERROR", f"所有 {pool.size} 个 Key 均被限流")
    raise last_err  # type: ignore[misc]
