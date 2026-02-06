"""API Key 轮询池。

支持逗号分隔的多 Key 配置，线程安全的 Round-Robin 分配 + 限流冷却。
单 Key 时行为与原逻辑完全一致（零开销）。
"""

import threading
import time

from utils.logger_system import log_msg


class APIKeyPool:
    """线程安全的 API Key 轮询池。

    Args:
        keys: API Key 列表（至少 1 个）

    示例:
        >>> pool = APIKeyPool(["key-a", "key-b", "key-c"])
        >>> pool.get_key()  # → "key-a"
        >>> pool.get_key()  # → "key-b"
        >>> pool.mark_rate_limited("key-a", cooldown=60)
        >>> pool.get_key()  # → "key-c"（跳过冷却中的 key-a）
    """

    def __init__(self, keys: list[str]) -> None:
        assert keys, "至少需要 1 个 API Key"
        self._keys = keys
        self._index = 0
        self._cooldowns: dict[str, float] = {}
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        """Key 总数。"""
        return len(self._keys)

    def get_key(self) -> str:
        """Round-Robin 获取下一个可用 Key（跳过冷却中的）。"""
        if self.size == 1:
            return self._keys[0]
        with self._lock:
            now = time.time()
            for _ in range(self.size):
                key = self._keys[self._index % self.size]
                self._index += 1
                if self._cooldowns.get(key, 0) <= now:
                    return key
            # 全部冷却中 → 返回冷却最早结束的
            return min(self._keys, key=lambda k: self._cooldowns.get(k, 0))

    def mark_rate_limited(self, key: str, cooldown: int = 60) -> None:
        """标记 Key 进入冷却期。

        Args:
            key: 被限流的 Key
            cooldown: 冷却秒数（默认 60s）
        """
        with self._lock:
            self._cooldowns[key] = time.time() + cooldown


# ============================================================
# 模块级缓存：相同原始字符串复用同一 Pool 实例
# ============================================================

_pools: dict[str, APIKeyPool] = {}
_pools_lock = threading.Lock()


def get_pool(raw: str) -> APIKeyPool:
    """获取或创建 Key 池。

    解析逗号分隔的 Key 字符串，相同字符串复用同一实例。

    Args:
        raw: 原始 api_key 字符串（可含逗号分隔的多个 Key）

    Returns:
        APIKeyPool 实例
    """
    with _pools_lock:
        if raw not in _pools:
            keys = [k.strip() for k in raw.split(",") if k.strip()]
            if not keys:
                raise ValueError("API Key 未配置")
            _pools[raw] = APIKeyPool(keys)
            if len(keys) > 1:
                log_msg("INFO", f"API Key Pool 初始化: {len(keys)} 个 Key")
        return _pools[raw]
