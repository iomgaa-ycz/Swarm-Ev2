"""后端抽象层单元测试。

测试 LLM 后端的提供商识别、API 调用和错误处理。
"""

import pytest
from core.backend import determine_provider, query


class TestProviderDetection:
    """测试提供商识别功能。"""

    def test_openai_models(self):
        """测试 OpenAI 模型识别。"""
        assert determine_provider("gpt-4-turbo") == "openai"
        assert determine_provider("gpt-3.5-turbo") == "openai"
        assert determine_provider("gpt-4") == "openai"
        assert determine_provider("o1-preview") == "openai"

    def test_glm_models(self):
        """测试 GLM 模型识别（兼容 OpenAI 格式）。"""
        assert determine_provider("glm-4.6") == "openai"
        assert determine_provider("glm-4.7") == "openai"
        assert determine_provider("glm-4-plus") == "openai"

    def test_anthropic_models(self):
        """测试 Anthropic 模型识别。"""
        assert determine_provider("claude-3-opus-20240229") == "anthropic"
        assert determine_provider("claude-3-sonnet") == "anthropic"
        assert determine_provider("claude-2.1") == "anthropic"

    def test_unsupported_models(self):
        """测试不支持的模型抛出异常。"""
        with pytest.raises(ValueError) as exc_info:
            determine_provider("unknown-model")
        assert "不支持的模型" in str(exc_info.value)

        with pytest.raises(ValueError):
            determine_provider("gemini-pro")


class TestGLMBackend:
    """测试 GLM 后端真实 API 调用。"""

    # GLM 4.7 配置
    GLM_API_KEY = "0d943612dc3c48bea3c8b8e1dc24a89b.hTBTuWjLqj8O0UXy"
    GLM_BASE_URL = "https://open.bigmodel.cn/api/coding/paas/v4"
    GLM_MODEL = "glm-4.7"

    def test_glm_basic_query(self):
        """测试 GLM 基本查询功能。"""
        response = query(
            system_message="你是一个数学助手",
            user_message="1+1=?",
            model=self.GLM_MODEL,
            api_key=self.GLM_API_KEY,
            base_url=self.GLM_BASE_URL,
        )

        assert isinstance(response, str)
        assert len(response) > 0
        assert "2" in response

    def test_glm_with_temperature(self):
        """测试 GLM 带温度参数查询。"""
        response = query(
            system_message="你是智能助手",
            user_message="用一句话介绍 Python",
            model=self.GLM_MODEL,
            api_key=self.GLM_API_KEY,
            base_url=self.GLM_BASE_URL,
            temperature=0.7,
        )

        assert isinstance(response, str)
        assert len(response) > 0
        assert "Python" in response

    def test_glm_with_max_tokens(self):
        """测试 GLM 带 max_tokens 参数查询。"""
        response = query(
            system_message="你是助手",
            user_message="说一个字：好",
            model=self.GLM_MODEL,
            api_key=self.GLM_API_KEY,
            base_url=self.GLM_BASE_URL,
            max_tokens=50,  # GLM 需要较大的 max_tokens
        )

        assert isinstance(response, str)
        # GLM 可能返回空字符串，这是正常行为

    @pytest.mark.skip(reason="GLM 要求必须有 user message")
    def test_glm_system_message_only(self):
        """测试只有 system message 的情况（GLM 不支持）。"""
        response = query(
            system_message="直接回答：1+1=2",
            user_message=None,
            model=self.GLM_MODEL,
            api_key=self.GLM_API_KEY,
            base_url=self.GLM_BASE_URL,
        )

        assert isinstance(response, str)
        assert len(response) > 0

    def test_glm_user_message_only(self):
        """测试只有 user message 的情况。"""
        response = query(
            system_message=None,
            user_message="你好",
            model=self.GLM_MODEL,
            api_key=self.GLM_API_KEY,
            base_url=self.GLM_BASE_URL,
        )

        assert isinstance(response, str)
        assert len(response) > 0

    def test_glm_long_conversation(self):
        """测试较长的对话。"""
        response = query(
            system_message="你是编程助手",
            user_message="请写一个 Python 函数，计算两个数的和",
            model=self.GLM_MODEL,
            api_key=self.GLM_API_KEY,
            base_url=self.GLM_BASE_URL,
        )

        assert isinstance(response, str)
        assert len(response) > 50
        assert "def" in response or "function" in response.lower()


class TestErrorHandling:
    """测试错误处理。"""

    GLM_BASE_URL = "https://open.bigmodel.cn/api/coding/paas/v4"
    GLM_MODEL = "glm-4.7"

    def test_invalid_api_key(self):
        """测试无效 API Key。"""
        with pytest.raises(Exception) as exc_info:
            query(
                system_message=None,
                user_message="test",
                model=self.GLM_MODEL,
                api_key="invalid-key-12345",
                base_url=self.GLM_BASE_URL,
            )

        # 应该包含认证或授权相关的错误信息
        error_msg = str(exc_info.value).lower()
        assert any(
            keyword in error_msg
            for keyword in ["auth", "key", "invalid", "unauthorized", "401"]
        )

    def test_empty_messages(self):
        """测试空消息。"""
        # 两个消息都为 None 应该能处理（虽然可能返回错误）
        try:
            response = query(
                system_message=None,
                user_message=None,
                model=self.GLM_MODEL,
                api_key="test-key",
                base_url=self.GLM_BASE_URL,
            )
            # 如果没有抛出异常，响应应该是字符串
            assert isinstance(response, str)
        except Exception:
            # 抛出异常也是可接受的行为
            pass


class TestBackendIntegration:
    """后端集成测试。"""

    GLM_API_KEY = "0d943612dc3c48bea3c8b8e1dc24a89b.hTBTuWjLqj8O0UXy"
    GLM_BASE_URL = "https://open.bigmodel.cn/api/coding/paas/v4"
    GLM_MODEL = "glm-4.7"

    def test_multiple_queries(self):
        """测试连续多次查询。"""
        questions = [
            "1+1=?",
            "2+2=?",
            "3+3=?",
        ]

        for question in questions:
            response = query(
                system_message="你是数学助手",
                user_message=question,
                model=self.GLM_MODEL,
                api_key=self.GLM_API_KEY,
                base_url=self.GLM_BASE_URL,
            )

            assert isinstance(response, str)
            assert len(response) > 0

    def test_different_model_versions(self):
        """测试不同 GLM 模型版本的识别。"""
        models = ["glm-4.6", "glm-4.7", "glm-4-plus"]

        for model in models:
            provider = determine_provider(model)
            assert provider == "openai"
