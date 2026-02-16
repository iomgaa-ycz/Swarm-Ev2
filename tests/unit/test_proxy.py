"""utils.proxy 模块单元测试。"""

import os
import pytest
from utils.proxy import setup_proxy_env, test_proxy_connectivity, test_network_connectivity


class TestProxySetup:
    """代理配置测试。"""

    def test_setup_proxy_env_sync_case(self, monkeypatch):
        """测试大小写同步。"""
        # 设置大写环境变量
        monkeypatch.setenv("HTTP_PROXY", "http://192.168.31.250:7890")
        monkeypatch.setenv("HTTPS_PROXY", "http://192.168.31.250:7890")

        # 清除小写
        monkeypatch.delenv("http_proxy", raising=False)
        monkeypatch.delenv("https_proxy", raising=False)

        setup_proxy_env()

        # 验证小写也被设置
        assert os.environ.get("http_proxy") == "http://192.168.31.250:7890"
        assert os.environ.get("https_proxy") == "http://192.168.31.250:7890"

    def test_setup_proxy_env_no_proxy_default(self, monkeypatch):
        """测试 NO_PROXY 默认值。"""
        monkeypatch.delenv("NO_PROXY", raising=False)
        monkeypatch.delenv("no_proxy", raising=False)

        setup_proxy_env()

        assert os.environ.get("NO_PROXY") == "localhost,127.0.0.1"
        assert os.environ.get("no_proxy") == "localhost,127.0.0.1"


class TestNetworkConnectivity:
    """网络连通性测试（需要实际网络）。"""

    @pytest.mark.skipif(
        not os.environ.get("TEST_NETWORK"), reason="需要网络连接，设置 TEST_NETWORK=1 启用"
    )
    def test_network_connectivity(self):
        """测试直连网络（跳过，除非环境变量启用）。"""
        result = test_network_connectivity(timeout=5)
        assert isinstance(result, bool)

    @pytest.mark.skipif(
        not os.environ.get("TEST_PROXY"), reason="需要代理，设置 TEST_PROXY=1 启用"
    )
    def test_proxy_connectivity(self):
        """测试代理连通性（跳过，除非环境变量启用）。"""
        proxy_url = "http://192.168.31.250:7890"
        result = test_proxy_connectivity(proxy_url, timeout=5)
        assert isinstance(result, bool)
