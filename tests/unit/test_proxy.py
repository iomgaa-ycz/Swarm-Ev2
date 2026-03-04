"""utils.proxy 模块单元测试。"""

import os
import pytest
from unittest.mock import patch, MagicMock

from utils.proxy import (
    setup_proxy_env,
    test_proxy_connectivity,
    test_network_connectivity,
    init_proxy,
    log_proxy_status,
    _mask_proxy_url,
    _clear_proxy_env,
    _log,
)


class TestProxySetup:
    """代理配置测试。"""

    def test_setup_proxy_env_sync_case(self, monkeypatch):
        """测试大小写同步。"""
        monkeypatch.setenv("HTTP_PROXY", "http://192.168.31.250:7890")
        monkeypatch.setenv("HTTPS_PROXY", "http://192.168.31.250:7890")
        monkeypatch.delenv("http_proxy", raising=False)
        monkeypatch.delenv("https_proxy", raising=False)

        setup_proxy_env()

        assert os.environ.get("http_proxy") == "http://192.168.31.250:7890"
        assert os.environ.get("https_proxy") == "http://192.168.31.250:7890"

    def test_setup_proxy_env_no_proxy_default(self, monkeypatch):
        """测试 NO_PROXY 默认值。"""
        monkeypatch.delenv("NO_PROXY", raising=False)
        monkeypatch.delenv("no_proxy", raising=False)

        setup_proxy_env()

        assert os.environ.get("NO_PROXY") == "localhost,127.0.0.1"
        assert os.environ.get("no_proxy") == "localhost,127.0.0.1"

    def test_setup_proxy_env_lowercase_sync(self, monkeypatch):
        """测试小写环境变量同步到大写。"""
        monkeypatch.delenv("HTTP_PROXY", raising=False)
        monkeypatch.delenv("HTTPS_PROXY", raising=False)
        monkeypatch.setenv("http_proxy", "http://10.0.0.1:8080")
        monkeypatch.setenv("https_proxy", "http://10.0.0.1:8080")

        setup_proxy_env()

        assert os.environ.get("HTTP_PROXY") == "http://10.0.0.1:8080"
        assert os.environ.get("HTTPS_PROXY") == "http://10.0.0.1:8080"

    def test_setup_proxy_env_preserves_existing_no_proxy(self, monkeypatch):
        """测试保留已有的 NO_PROXY 值。"""
        monkeypatch.setenv("NO_PROXY", "localhost,10.0.0.0/8")
        monkeypatch.delenv("no_proxy", raising=False)

        setup_proxy_env()

        assert os.environ.get("NO_PROXY") == "localhost,10.0.0.0/8"


class TestProxyConnectivityMock:
    """代理连通性测试（使用 mock）。"""

    def test_no_proxy_url_returns_false(self, monkeypatch):
        """没有代理 URL 时返回 False。"""
        monkeypatch.delenv("HTTP_PROXY", raising=False)
        monkeypatch.delenv("http_proxy", raising=False)
        assert test_proxy_connectivity(proxy_url=None) is False

    def test_empty_proxy_url_returns_false(self):
        """空代理 URL 返回 False。"""
        assert test_proxy_connectivity(proxy_url="") is False

    @patch("urllib.request.build_opener")
    def test_proxy_connectivity_success(self, mock_opener):
        """代理可用时返回 True。"""
        mock_opener.return_value.open.return_value = MagicMock()
        assert test_proxy_connectivity("http://10.0.0.1:7890") is True

    @patch("urllib.request.build_opener")
    def test_proxy_connectivity_failure(self, mock_opener):
        """代理不可用时返回 False。"""
        mock_opener.return_value.open.side_effect = Exception("Connection refused")
        assert test_proxy_connectivity("http://10.0.0.1:7890") is False


class TestNetworkConnectivityMock:
    """网络连通性测试（使用 mock）。"""

    @patch("urllib.request.build_opener")
    def test_network_ok(self, mock_opener):
        """网络可用时返回 True。"""
        mock_opener.return_value.open.return_value = MagicMock()
        assert test_network_connectivity() is True

    @patch("urllib.request.build_opener")
    def test_network_fail(self, mock_opener):
        """网络不可用时返回 False。"""
        mock_opener.return_value.open.side_effect = Exception("Network unreachable")
        assert test_network_connectivity() is False


class TestInitProxy:
    """init_proxy() 测试。"""

    @patch("utils.proxy.test_network_connectivity", return_value=True)
    def test_no_proxy_configured_direct_ok(self, mock_net, monkeypatch):
        """无代理配置且直连可用。"""
        monkeypatch.delenv("HTTP_PROXY", raising=False)
        monkeypatch.delenv("http_proxy", raising=False)
        monkeypatch.delenv("HTTPS_PROXY", raising=False)
        monkeypatch.delenv("https_proxy", raising=False)

        result = init_proxy()
        assert result is False

    @patch("utils.proxy.test_network_connectivity", return_value=False)
    def test_no_proxy_configured_direct_fail(self, mock_net, monkeypatch):
        """无代理配置且直连也不可用。"""
        monkeypatch.delenv("HTTP_PROXY", raising=False)
        monkeypatch.delenv("http_proxy", raising=False)
        monkeypatch.delenv("HTTPS_PROXY", raising=False)
        monkeypatch.delenv("https_proxy", raising=False)

        result = init_proxy()
        assert result is False

    @patch("utils.proxy.log_proxy_status")
    @patch("utils.proxy.test_proxy_connectivity", return_value=True)
    def test_proxy_configured_and_ok(self, mock_proxy, mock_log, monkeypatch):
        """代理配置且可用。"""
        monkeypatch.setenv("HTTP_PROXY", "http://10.0.0.1:7890")

        result = init_proxy()
        assert result is True

    @patch("utils.proxy.test_network_connectivity", return_value=True)
    @patch("utils.proxy.test_proxy_connectivity", return_value=False)
    def test_proxy_configured_but_fail_fallback(self, mock_proxy, mock_net, monkeypatch):
        """代理配置但不可用，降级为直连。"""
        monkeypatch.setenv("HTTP_PROXY", "http://10.0.0.1:7890")

        result = init_proxy()
        assert result is False
        # 代理变量应被清除
        assert os.environ.get("HTTP_PROXY", "") == ""


class TestLogProxyStatus:
    """log_proxy_status() 测试。"""

    def test_with_proxy(self, monkeypatch):
        """有代理时记录状态。"""
        monkeypatch.setenv("HTTP_PROXY", "http://10.0.0.1:7890")
        monkeypatch.setenv("HTTPS_PROXY", "http://10.0.0.1:7890")
        monkeypatch.setenv("NO_PROXY", "localhost")

        # 不应抛异常
        log_proxy_status()

    def test_without_proxy(self, monkeypatch):
        """无代理时记录直连模式。"""
        monkeypatch.delenv("HTTP_PROXY", raising=False)
        monkeypatch.delenv("HTTPS_PROXY", raising=False)

        log_proxy_status()


class TestMaskProxyUrl:
    """_mask_proxy_url() 测试。"""

    def test_empty_url(self):
        """空 URL 返回 '(空)'。"""
        assert _mask_proxy_url("") == "(空)"

    def test_normal_url(self):
        """正常 URL 返回原值。"""
        assert _mask_proxy_url("http://10.0.0.1:7890") == "http://10.0.0.1:7890"


class TestClearProxyEnv:
    """_clear_proxy_env() 测试。"""

    def test_clears_all_proxy_vars(self, monkeypatch):
        """清除所有代理环境变量。"""
        monkeypatch.setenv("HTTP_PROXY", "http://10.0.0.1:7890")
        monkeypatch.setenv("http_proxy", "http://10.0.0.1:7890")
        monkeypatch.setenv("HTTPS_PROXY", "http://10.0.0.1:7890")
        monkeypatch.setenv("https_proxy", "http://10.0.0.1:7890")

        _clear_proxy_env()

        assert "HTTP_PROXY" not in os.environ
        assert "http_proxy" not in os.environ
        assert "HTTPS_PROXY" not in os.environ
        assert "https_proxy" not in os.environ


class TestLogFunction:
    """_log() 安全日志函数测试。"""

    def test_log_fallback_to_print(self, capsys):
        """logger 不可用时回退到 print。"""
        import utils.proxy as proxy_mod

        old_ready = proxy_mod._logger_ready
        try:
            proxy_mod._logger_ready = True  # 跳过 logger 导入尝试
            _log("INFO", "test fallback")

            captured = capsys.readouterr()
            assert "[INFO] test fallback" in captured.out
        finally:
            proxy_mod._logger_ready = old_ready
