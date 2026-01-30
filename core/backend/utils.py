"""后端工具函数模块。

提供消息格式转换和 API 重试机制。
"""

from typing import Callable, Any
import backoff

from utils.logger_system import log_msg


def opt_messages_to_list(
    system_message: str | None,
    user_message: str | None,
) -> list[dict[str, str]]:
    """将可选的 system 和 user 消息转换为消息列表。

    Args:
        system_message: 系统消息（可选）
        user_message: 用户消息（可选）

    Returns:
        消息列表，格式为 [{"role": "system", "content": "..."}, ...]

    示例:
        >>> opt_messages_to_list("You are a helpful assistant", "Hello")
        [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"}
        ]
    """
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    if user_message:
        messages.append({"role": "user", "content": user_message})
    return messages


@backoff.on_predicate(
    wait_gen=backoff.expo,
    max_value=60,
    factor=1.5,
)
def backoff_create(
    create_fn: Callable,
    retry_exceptions: tuple[type[Exception], ...],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """带指数退避的 API 调用包装器。

    使用指数退避策略自动重试失败的 API 调用。
    重试间隔: 1.5^n 秒，最大 60 秒。

    Args:
        create_fn: 要调用的函数
        retry_exceptions: 需要重试的异常类型元组
        *args: 传递给 create_fn 的位置参数
        **kwargs: 传递给 create_fn 的关键字参数

    Returns:
        API 调用的返回值

    Raises:
        如果不是指定的重试异常，直接抛出

    示例:
        >>> def api_call():
        ...     return client.create(...)
        >>> result = backoff_create(
        ...     api_call,
        ...     (RateLimitError, APIConnectionError),
        ... )
    """
    try:
        return create_fn(*args, **kwargs)
    except retry_exceptions as e:
        log_msg("WARNING", f"API 调用失败，准备重试: {e}")
        return False  # 返回 False 触发 backoff 重试
