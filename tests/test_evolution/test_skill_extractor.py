"""SkillExtractor 单元测试。"""

import pytest
from unittest.mock import Mock, patch
import numpy as np

from core.evolution.skill_extractor import SkillExtractor
from core.evolution.experience_pool import ExperiencePool, TaskRecord
from utils.config import Config


@pytest.fixture
def mock_config():
    """创建 Mock 配置。"""
    config = Mock(spec=Config)
    config.evolution = Mock()
    config.evolution.skill = Mock()
    config.evolution.skill.min_cluster_size = 5
    config.llm = Mock()
    config.llm.code = Mock()
    config.llm.code.model = "gpt-4"
    return config


@pytest.fixture
def mock_experience_pool():
    """创建 Mock 经验池。"""
    pool = Mock(spec=ExperiencePool)
    return pool


@pytest.fixture
def sample_records():
    """创建示例记录。"""
    return [
        TaskRecord(
            agent_id="agent_0",
            task_type="explore",
            input_hash="hash1",
            output_quality=0.8,
            strategy_summary="使用 RandomForest 进行分类",
            timestamp=1.0,
        ),
        TaskRecord(
            agent_id="agent_1",
            task_type="explore",
            input_hash="hash2",
            output_quality=0.7,
            strategy_summary="使用 XGBoost 进行分类",
            timestamp=2.0,
        ),
        TaskRecord(
            agent_id="agent_2",
            task_type="explore",
            input_hash="hash3",
            output_quality=0.9,
            strategy_summary="使用 LightGBM 进行分类",
            timestamp=3.0,
        ),
        TaskRecord(
            agent_id="agent_3",
            task_type="explore",
            input_hash="hash4",
            output_quality=0.6,
            strategy_summary="使用 SVM 进行分类",
            timestamp=4.0,
        ),
        TaskRecord(
            agent_id="agent_0",
            task_type="explore",
            input_hash="hash5",
            output_quality=0.85,
            strategy_summary="使用集成方法进行分类",
            timestamp=5.0,
        ),
    ]


class TestSkillExtractor:
    """测试 SkillExtractor 类。"""

    def test_init(self, mock_config, mock_experience_pool):
        """测试初始化。"""
        extractor = SkillExtractor(mock_experience_pool, mock_config)

        assert extractor.experience_pool == mock_experience_pool
        assert extractor.config == mock_config
        assert extractor.min_cluster_size == 5

    def test_extract_skills_insufficient_records(
        self, mock_config, mock_experience_pool
    ):
        """测试记录数不足。"""
        # 配置经验池返回少量记录
        mock_experience_pool.query.return_value = [Mock()] * 3

        extractor = SkillExtractor(mock_experience_pool, mock_config)
        skills = extractor.extract_skills("explore", min_cluster_size=5)

        assert skills == []

    @patch("core.evolution.skill_extractor.query_with_config")
    def test_extract_skills_success(
        self, mock_query, mock_config, mock_experience_pool, sample_records
    ):
        """测试成功提取 Skill。"""
        # 配置经验池返回示例记录
        mock_experience_pool.query.return_value = sample_records

        # Mock LLM 响应
        mock_query.return_value = (
            "## Explore Skill: 分类策略\n\n### 核心策略\n- 使用树模型\n"
        )

        extractor = SkillExtractor(mock_experience_pool, mock_config)

        # Mock 嵌入和聚类
        with (
            patch.object(extractor, "_embed_texts") as mock_embed,
            patch.object(extractor, "_cluster") as mock_cluster,
        ):
            # 模拟嵌入
            mock_embed.return_value = np.random.rand(5, 128).astype(np.float32)

            # 模拟聚类结果（2 个簇）
            mock_cluster.return_value = {
                0: [0, 1, 2],  # 簇 0: 前 3 个记录
                1: [3, 4],  # 簇 1: 后 2 个记录
            }

            skills = extractor.extract_skills("explore", min_cluster_size=3)

        # 验证结果
        assert len(skills) == 2

        # 验证 Skill 结构
        skill = skills[0]
        assert "id" in skill
        assert "task_type" in skill
        assert "content" in skill
        assert "coverage" in skill
        assert "avg_accuracy" in skill
        assert "avg_generation_rate" in skill
        assert "composite_score" in skill
        assert "status" in skill

        assert skill["task_type"] == "explore"
        assert skill["status"] == "candidate"

    def test_embed_texts(self, mock_config, mock_experience_pool):
        """测试文本向量化。"""
        extractor = SkillExtractor(mock_experience_pool, mock_config)

        texts = ["文本1", "文本2", "文本3"]

        with patch.object(extractor.embedding_manager, "embed_texts") as mock_embed:
            mock_embed.return_value = np.random.rand(3, 128).astype(np.float32)

            embeddings = extractor._embed_texts(texts)

            assert embeddings.shape == (3, 128)
            mock_embed.assert_called_once_with(texts)

    def test_cluster(self, mock_config, mock_experience_pool):
        """测试 HDBSCAN 聚类。"""
        extractor = SkillExtractor(mock_experience_pool, mock_config)

        embeddings = np.random.rand(10, 128).astype(np.float32)

        # Mock HDBSCAN
        with patch("core.evolution.skill_extractor.hdbscan.HDBSCAN") as mock_hdbscan:
            mock_clusterer = Mock()
            # 模拟聚类结果: 3 个簇 + 噪声点
            mock_clusterer.fit_predict.return_value = np.array(
                [0, 0, 0, 1, 1, 2, 2, 2, -1, -1]
            )
            mock_hdbscan.return_value = mock_clusterer

            clusters = extractor._cluster(embeddings, min_cluster_size=3)

        # 验证结果
        assert len(clusters) == 3
        assert 0 in clusters
        assert 1 in clusters
        assert 2 in clusters
        assert -1 not in clusters  # 噪声点应被排除

        assert clusters[0] == [0, 1, 2]
        assert clusters[1] == [3, 4]
        assert clusters[2] == [5, 6, 7]

    @patch("core.evolution.skill_extractor.query_with_config")
    def test_summarize_cluster(self, mock_query, mock_config, mock_experience_pool):
        """测试 LLM 总结策略簇。"""
        extractor = SkillExtractor(mock_experience_pool, mock_config)

        strategies = ["策略1", "策略2", "策略3"]
        task_type = "explore"

        # Mock LLM 响应
        mock_query.return_value = "## Explore Skill: 测试\n\n### 核心策略\n- 测试策略\n"

        content = extractor._summarize_cluster(strategies, task_type)

        assert isinstance(content, str)
        assert "Skill" in content
        mock_query.assert_called_once()

    @patch("core.evolution.skill_extractor.query_with_config")
    def test_summarize_cluster_llm_failure(
        self, mock_query, mock_config, mock_experience_pool
    ):
        """测试 LLM 失败时的 Fallback。"""
        extractor = SkillExtractor(mock_experience_pool, mock_config)

        strategies = ["策略1", "策略2", "策略3"]
        task_type = "explore"

        # Mock LLM 抛出异常
        mock_query.side_effect = Exception("LLM 调用失败")

        content = extractor._summarize_cluster(strategies, task_type)

        # 应返回 Fallback 内容
        assert isinstance(content, str)
        assert "Skill" in content
        assert "策略模式" in content

    def test_calc_avg_accuracy(self, mock_config, mock_experience_pool, sample_records):
        """测试计算平均准确率。"""
        extractor = SkillExtractor(mock_experience_pool, mock_config)

        indices = [0, 1, 2]  # 前 3 个记录
        avg_accuracy = extractor._calc_avg_accuracy(indices, sample_records)

        # 手动计算期望值
        expected = (0.8 + 0.7 + 0.9) / 3
        assert abs(avg_accuracy - expected) < 0.01

    def test_calc_generation_rate(
        self, mock_config, mock_experience_pool, sample_records
    ):
        """测试计算生成率。"""
        extractor = SkillExtractor(mock_experience_pool, mock_config)

        # 所有记录都有 output_quality > 0
        indices = [0, 1, 2]
        rate = extractor._calc_generation_rate(indices, sample_records)
        assert rate == 1.0

        # 添加一些 output_quality = 0 的记录
        failed_records = sample_records + [
            TaskRecord(
                agent_id="agent_x",
                task_type="explore",
                input_hash="hashx",
                output_quality=0.0,
                strategy_summary="失败",
                timestamp=6.0,
            )
        ]

        indices = [0, 5]  # 1 个成功，1 个失败
        rate = extractor._calc_generation_rate(indices, failed_records)
        assert rate == 0.5
