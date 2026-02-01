"""代码嵌入管理器模块。

基于 bge-m3 模型的文本向量化工具，支持懒加载和缓存机制。
参考 Reference/Swarm-Evo/core/evolution/embedding_manager.py。
"""

import os
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer

from utils.logger_system import log_msg


class CodeEmbeddingManager:
    """代码嵌入管理器（懒加载 + 缓存）。

    使用 bge-m3 模型进行文本向量化，返回 L2 归一化的 embeddings。

    Attributes:
        _model_name: 模型名称（HuggingFace）
        _model: SentenceTransformer 模型实例（类级别，单例）
        _cache: 文本缓存（text -> embedding）
    """

    _model_name = "BAAI/bge-m3"
    _model: SentenceTransformer = None

    def __init__(self) -> None:
        """初始化嵌入管理器。

        注意：模型是懒加载的，首次调用 embed_texts 时才加载。
        """
        # 实例级缓存（text -> embedding）
        self._cache: Dict[str, np.ndarray] = {}

    @classmethod
    def _ensure_model(cls) -> None:
        """确保模型已加载（懒加载）。

        优先从环境变量 LOCAL_MODEL_PATH 读取本地路径，
        如果不存在则从 HuggingFace 下载。

        Raises:
            RuntimeError: 模型加载失败
        """
        if cls._model is not None:
            return

        # 尝试从环境变量读取本地路径
        local_model_path = os.environ.get(
            "LOCAL_MODEL_PATH", "./embedding-models/bge-m3"
        )

        if os.path.exists(local_model_path):
            log_msg("INFO", f"从本地路径加载 bge-m3 模型: {local_model_path}")
            cls._model = SentenceTransformer(local_model_path)
        else:
            log_msg(
                "INFO",
                f"本地模型不存在（{local_model_path}），从 HuggingFace 下载 {cls._model_name}...",
            )
            cls._model = SentenceTransformer(cls._model_name)

        cls._model.eval()
        log_msg("INFO", "bge-m3 模型加载完成")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """批量文本向量化（L2 归一化）。

        Args:
            texts: 文本列表

        Returns:
            L2 归一化的 embeddings，形状 (len(texts), embedding_dim)

        Raises:
            RuntimeError: 模型加载失败
        """
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        # [1] 检查缓存
        cached_embeddings: List[np.ndarray] = []
        missing_texts: List[str] = []

        for text in texts:
            if text in self._cache:
                cached_embeddings.append(self._cache[text])
            else:
                cached_embeddings.append(None)
                # 添加 "code:" 前缀（与 Reference 保持一致）
                missing_texts.append(f"code:\n{text}")

        # [2] 批量编码缺失的文本
        if missing_texts:
            self._ensure_model()

            if self._model is None:
                raise RuntimeError("bge-m3 模型加载失败")

            embeddings = self._model.encode(
                missing_texts,
                batch_size=8,
                show_progress_bar=False,
                normalize_embeddings=True,  # L2 归一化
            )

            # 确保返回 numpy array
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)

            embeddings = embeddings.astype(np.float32)

            # [3] 更新缓存
            idx = 0
            for i, original_text in enumerate(texts):
                if cached_embeddings[i] is None:
                    vec = embeddings[idx]
                    self._cache[original_text] = vec
                    cached_embeddings[i] = vec
                    idx += 1

        # [4] 拼接结果
        result = np.vstack(cached_embeddings)
        return result
