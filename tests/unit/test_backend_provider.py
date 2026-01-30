"""后端 Provider 参数测试。

测试 core/backend/__init__.py 中 provider 必填参数的行为。
"""

import pytest
from unittest.mock import patch, MagicMock
from core.backend import query, PROVIDER_TO_QUERY


class TestBackendProvider:
    """后端 Provider 参数测试类。"""

    @patch("core.backend.PROVIDER_TO_QUERY")
    def test_valid_provider_openai(self, mock_provider_map):
        """测试有效的 openai provider。"""
        mock_query_func = MagicMock(return_value="test response")
        mock_provider_map.__getitem__.return_value = mock_query_func
        mock_provider_map.__contains__.return_value = True

        result = query(
            system_message="test system",
            user_message="test user",
            model="gpt-4-turbo",
            provider="openai",
            api_key="sk-test",
        )

        assert result == "test response"
        mock_query_func.assert_called_once()

    @patch("core.backend.PROVIDER_TO_QUERY")
    def test_valid_provider_anthropic(self, mock_provider_map):
        """测试有效的 anthropic provider。"""
        mock_query_func = MagicMock(return_value="test response")
        mock_provider_map.__getitem__.return_value = mock_query_func
        mock_provider_map.__contains__.return_value = True

        result = query(
            system_message="test system",
            user_message="test user",
            model="claude-3-opus-20240229",
            provider="anthropic",
            api_key="sk-ant-test",
        )

        assert result == "test response"
        mock_query_func.assert_called_once()

    def test_invalid_provider(self):
        """测试无效的 provider（应抛出 ValueError）。"""
        with pytest.raises(ValueError, match="不支持的 provider: invalid"):
            query(
                system_message="test",
                user_message="test",
                model="test-model",
                provider="invalid",
                api_key="sk-test",
            )

    def test_missing_provider(self):
        """测试缺失 provider 参数（应抛出 TypeError）。"""
        with pytest.raises(TypeError):
            # 缺少必填参数 provider
            query(  # type: ignore
                system_message="test",
                user_message="test",
                model="gpt-4-turbo",
                api_key="sk-test",
            )

    @patch("core.backend.PROVIDER_TO_QUERY")
    def test_provider_with_base_url(self, mock_provider_map):
        """测试 provider 配合 base_url 参数（第三方 API）。"""
        mock_query_func = MagicMock(return_value="moonshot response")
        mock_provider_map.__getitem__.return_value = mock_query_func
        mock_provider_map.__contains__.return_value = True

        result = query(
            system_message="test",
            user_message="test",
            model="moonshot-v1-8k",
            provider="openai",
            api_key="sk-moonshot",
            base_url="https://api.moonshot.cn/v1",
        )

        assert result == "moonshot response"
        # 验证 base_url 被传递
        call_kwargs = mock_query_func.call_args[1]
        assert call_kwargs["base_url"] == "https://api.moonshot.cn/v1"

    def test_provider_to_query_mapping(self):
        """测试 PROVIDER_TO_QUERY 映射正确性。"""
        assert "openai" in PROVIDER_TO_QUERY
        assert "anthropic" in PROVIDER_TO_QUERY
        assert len(PROVIDER_TO_QUERY) == 2
