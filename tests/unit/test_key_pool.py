"""API Key Pool 单元测试。

测试 core/backend/key_pool.py 的轮询、冷却、缓存逻辑。
"""

import time
import threading
import pytest

from core.backend.key_pool import APIKeyPool, get_pool, _pools


class TestAPIKeyPool:
    """APIKeyPool 核心逻辑测试。"""

    def test_single_key_always_returns_same(self):
        """单 Key 场景：始终返回同一个 Key。"""
        pool = APIKeyPool(["key-a"])
        assert pool.get_key() == "key-a"
        assert pool.get_key() == "key-a"
        assert pool.size == 1

    def test_round_robin(self):
        """多 Key 场景：Round-Robin 轮询。"""
        pool = APIKeyPool(["key-a", "key-b", "key-c"])
        results = [pool.get_key() for _ in range(6)]
        assert results == ["key-a", "key-b", "key-c", "key-a", "key-b", "key-c"]

    def test_skip_rate_limited_key(self):
        """冷却中的 Key 被跳过。"""
        pool = APIKeyPool(["key-a", "key-b", "key-c"])
        pool.get_key()  # → key-a，推进 index
        pool.mark_rate_limited("key-b", cooldown=60)

        key = pool.get_key()  # 应跳过 key-b
        assert key == "key-c"

    def test_all_rate_limited_returns_earliest_expiry(self):
        """全部冷却时返回最早过期的 Key。"""
        pool = APIKeyPool(["key-a", "key-b"])
        pool.mark_rate_limited("key-a", cooldown=100)
        pool.mark_rate_limited("key-b", cooldown=200)

        key = pool.get_key()
        assert key == "key-a"  # key-a 冷却先结束

    def test_cooldown_expires(self):
        """冷却过期后 Key 恢复可用。"""
        pool = APIKeyPool(["key-a", "key-b"])
        pool.mark_rate_limited("key-a", cooldown=0)  # 立即过期
        time.sleep(0.01)

        # key-a 应该已经恢复
        keys = {pool.get_key() for _ in range(4)}
        assert "key-a" in keys

    def test_empty_keys_raises(self):
        """空 Key 列表应抛出 AssertionError。"""
        with pytest.raises(AssertionError):
            APIKeyPool([])

    def test_thread_safety(self):
        """多线程并发 get_key 不崩溃。"""
        pool = APIKeyPool(["key-a", "key-b", "key-c"])
        results: list[str] = []
        lock = threading.Lock()

        def worker():
            for _ in range(100):
                key = pool.get_key()
                with lock:
                    results.append(key)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 400
        assert all(k in ("key-a", "key-b", "key-c") for k in results)


class TestGetPool:
    """get_pool 模块缓存测试。"""

    def setup_method(self):
        """每个测试前清空缓存。"""
        _pools.clear()

    def test_single_key_parse(self):
        """单 Key 字符串解析。"""
        pool = get_pool("sk-abc123")
        assert pool.size == 1
        assert pool.get_key() == "sk-abc123"

    def test_multi_key_parse(self):
        """逗号分隔多 Key 解析。"""
        pool = get_pool("sk-aaa, sk-bbb, sk-ccc")
        assert pool.size == 3

    def test_cache_reuse(self):
        """相同字符串复用同一 Pool 实例。"""
        pool1 = get_pool("sk-aaa,sk-bbb")
        pool2 = get_pool("sk-aaa,sk-bbb")
        assert pool1 is pool2

    def test_different_string_different_pool(self):
        """不同字符串创建不同 Pool。"""
        pool1 = get_pool("sk-aaa")
        pool2 = get_pool("sk-bbb")
        assert pool1 is not pool2

    def test_empty_string_raises(self):
        """空字符串应抛出 ValueError。"""
        with pytest.raises(ValueError, match="API Key 未配置"):
            get_pool("")

    def test_whitespace_trimmed(self):
        """Key 两端空白被清除。"""
        pool = get_pool("  sk-aaa , sk-bbb  ")
        assert pool.size == 2
        assert pool.get_key() == "sk-aaa"
        assert pool.get_key() == "sk-bbb"
