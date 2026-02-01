"""CodeEmbeddingManager 单元测试。"""

import numpy as np

from core.evolution.code_embedding_manager import CodeEmbeddingManager


class TestCodeEmbeddingManager:
    """测试 CodeEmbeddingManager 类。"""

    def test_init(self):
        """测试初始化。"""
        manager = CodeEmbeddingManager()
        assert manager._cache == {}

    def test_embed_texts_single(self):
        """测试单个文本向量化。"""
        manager = CodeEmbeddingManager()
        texts = ["这是一个测试文本"]

        embeddings = manager.embed_texts(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 1
        assert embeddings.dtype == np.float32

        # 检查 L2 归一化（向量范数应接近 1）
        norm = np.linalg.norm(embeddings[0])
        assert abs(norm - 1.0) < 0.01

    def test_embed_texts_multiple(self):
        """测试多个文本向量化。"""
        manager = CodeEmbeddingManager()
        texts = [
            "第一个测试文本",
            "第二个测试文本",
            "第三个测试文本",
        ]

        embeddings = manager.embed_texts(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 3
        assert embeddings.dtype == np.float32

        # 检查每个向量都是 L2 归一化的
        for i in range(3):
            norm = np.linalg.norm(embeddings[i])
            assert abs(norm - 1.0) < 0.01

    def test_embed_texts_empty(self):
        """测试空文本列表。"""
        manager = CodeEmbeddingManager()
        texts = []

        embeddings = manager.embed_texts(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (0, 0)

    def test_embed_texts_cache(self):
        """测试缓存机制。"""
        manager = CodeEmbeddingManager()
        text = "测试缓存文本"

        # 第一次嵌入
        embeddings1 = manager.embed_texts([text])

        # 检查缓存
        assert text in manager._cache

        # 第二次嵌入（应使用缓存）
        embeddings2 = manager.embed_texts([text])

        # 结果应完全相同（使用缓存）
        np.testing.assert_array_equal(embeddings1, embeddings2)

    def test_embed_texts_mixed_cache(self):
        """测试部分缓存场景。"""
        manager = CodeEmbeddingManager()

        # 第一次嵌入
        texts1 = ["文本1", "文本2"]
        embeddings1 = manager.embed_texts(texts1)

        # 第二次嵌入（部分缓存）
        texts2 = ["文本2", "文本3"]
        embeddings2 = manager.embed_texts(texts2)

        # "文本2" 的嵌入应该相同
        np.testing.assert_array_equal(embeddings1[1], embeddings2[0])

    def test_embed_texts_code_prefix(self):
        """测试代码前缀添加。"""
        manager = CodeEmbeddingManager()
        text = "def foo(): pass"

        # 嵌入（会添加 "code:" 前缀）
        embeddings = manager.embed_texts([text])

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 1

        # 缓存中的 key 不应包含前缀（使用原文本）
        assert text in manager._cache

    def test_model_lazy_loading(self):
        """测试模型懒加载。"""
        # 清空类级别的模型
        CodeEmbeddingManager._model = None

        manager = CodeEmbeddingManager()

        # 初始化时不应加载模型
        assert CodeEmbeddingManager._model is None

        # 第一次调用 embed_texts 时加载模型
        manager.embed_texts(["test"])

        # 模型应已加载
        assert CodeEmbeddingManager._model is not None

    def test_model_singleton(self):
        """测试模型单例模式。"""
        # 清空类级别的模型
        CodeEmbeddingManager._model = None

        manager1 = CodeEmbeddingManager()
        manager2 = CodeEmbeddingManager()

        # 触发模型加载
        manager1.embed_texts(["test"])

        # 两个实例应共享同一个模型
        assert CodeEmbeddingManager._model is not None
        assert manager1._model is manager2._model
