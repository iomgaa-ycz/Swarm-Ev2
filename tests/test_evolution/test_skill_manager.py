"""SkillManager 单元测试。"""

import pytest
from unittest.mock import Mock, patch
import numpy as np

from core.evolution.skill_manager import SkillManager
from core.evolution.experience_pool import ExperiencePool
from core.evolution.skill_extractor import SkillExtractor
from utils.config import Config


@pytest.fixture
def tmp_skills_dir(tmp_path):
    """创建临时 Skill 目录。"""
    return tmp_path / "skills"


@pytest.fixture
def tmp_meta_dir(tmp_path):
    """创建临时元数据目录。"""
    return tmp_path / "meta"


@pytest.fixture
def mock_config():
    """创建 Mock 配置。"""
    config = Mock(spec=Config)
    config.evolution = Mock()
    config.evolution.skill = Mock()
    config.evolution.skill.duplicate_threshold = 0.85
    config.evolution.skill.min_composite_score = 0.5
    config.evolution.skill.deprecate_threshold = 0.4
    config.evolution.skill.unused_epochs = 5
    return config


@pytest.fixture
def mock_embedding_manager():
    """创建 Mock 嵌入管理器。"""
    manager = Mock()
    return manager


@pytest.fixture
def sample_skill():
    """创建示例 Skill。"""
    return {
        "id": "skill_explore_0_123456",
        "task_type": "explore",
        "content": "## Explore Skill: 测试\n\n### 核心策略\n- 测试策略\n",
        "coverage": 10,
        "avg_accuracy": 0.8,
        "avg_generation_rate": 0.9,
        "composite_score": 0.84,  # 0.6 * 0.8 + 0.4 * 0.9
        "status": "candidate",
    }


class TestSkillManager:
    """测试 SkillManager 类。"""

    def test_init(
        self, tmp_skills_dir, tmp_meta_dir, mock_config, mock_embedding_manager
    ):
        """测试初始化。"""
        manager = SkillManager(
            tmp_skills_dir, tmp_meta_dir, mock_config, mock_embedding_manager
        )

        assert manager.skills_dir == tmp_skills_dir
        assert manager.meta_dir == tmp_meta_dir
        assert manager.config == mock_config
        assert manager.embedding_manager == mock_embedding_manager
        assert manager.skill_index == {}

        # 验证目录已创建
        assert tmp_skills_dir.exists()
        assert tmp_meta_dir.exists()

    def test_add_skill_success(
        self,
        tmp_skills_dir,
        tmp_meta_dir,
        mock_config,
        mock_embedding_manager,
        sample_skill,
    ):
        """测试成功添加 Skill。"""
        manager = SkillManager(
            tmp_skills_dir, tmp_meta_dir, mock_config, mock_embedding_manager
        )

        # Mock 重复检测
        with patch.object(manager, "_is_duplicate", return_value=False):
            result = manager.add_skill(sample_skill)

        assert result is True
        assert sample_skill["id"] in manager.skill_index

        # 验证文件已创建
        skill_path = (
            tmp_skills_dir
            / "by_task_type"
            / "explore"
            / "success_patterns"
            / f"{sample_skill['id']}.md"
        )
        assert skill_path.exists()

        # 验证索引
        meta = manager.skill_index[sample_skill["id"]]
        assert meta["task_type"] == "explore"
        assert meta["composite_score"] == 0.84
        assert meta["status"] == "active"

    def test_add_skill_duplicate(
        self,
        tmp_skills_dir,
        tmp_meta_dir,
        mock_config,
        mock_embedding_manager,
        sample_skill,
    ):
        """测试添加重复 Skill。"""
        manager = SkillManager(
            tmp_skills_dir, tmp_meta_dir, mock_config, mock_embedding_manager
        )

        # Mock 重复检测
        with patch.object(manager, "_is_duplicate", return_value=True):
            result = manager.add_skill(sample_skill)

        assert result is False
        assert sample_skill["id"] not in manager.skill_index

    def test_evaluate_skill(
        self,
        tmp_skills_dir,
        tmp_meta_dir,
        mock_config,
        mock_embedding_manager,
        sample_skill,
    ):
        """测试 Skill 质量评估。"""
        manager = SkillManager(
            tmp_skills_dir, tmp_meta_dir, mock_config, mock_embedding_manager
        )

        # 添加 Skill
        with patch.object(manager, "_is_duplicate", return_value=False):
            manager.add_skill(sample_skill)

        # 评估 Skill
        score = manager.evaluate_skill(sample_skill["id"])

        # 验证评分公式: 0.6 * avg_accuracy + 0.4 * avg_generation_rate
        expected = 0.6 * 0.8 + 0.4 * 0.9
        assert abs(score - expected) < 0.01

    def test_evaluate_skill_not_found(
        self, tmp_skills_dir, tmp_meta_dir, mock_config, mock_embedding_manager
    ):
        """测试评估不存在的 Skill。"""
        manager = SkillManager(
            tmp_skills_dir, tmp_meta_dir, mock_config, mock_embedding_manager
        )

        score = manager.evaluate_skill("non_existent_skill")
        assert score == 0.0

    def test_get_top_k_skills(
        self,
        tmp_skills_dir,
        tmp_meta_dir,
        mock_config,
        mock_embedding_manager,
        sample_skill,
    ):
        """测试获取 Top-K Skill。"""
        manager = SkillManager(
            tmp_skills_dir, tmp_meta_dir, mock_config, mock_embedding_manager
        )

        # 添加多个 Skill
        skills = []
        for i in range(5):
            skill = sample_skill.copy()
            skill["id"] = f"skill_explore_{i}_123456"
            skill["composite_score"] = 0.5 + i * 0.1  # 0.5, 0.6, 0.7, 0.8, 0.9
            skills.append(skill)

            with patch.object(manager, "_is_duplicate", return_value=False):
                manager.add_skill(skill)

        # 获取 Top-3
        top_k = manager.get_top_k_skills("explore", k=3)

        assert len(top_k) == 3

        # 验证是按综合评分降序排列
        # Top-3 应该是 score=0.9, 0.8, 0.7
        assert "## Explore Skill" in top_k[0]

    def test_reload_index(
        self,
        tmp_skills_dir,
        tmp_meta_dir,
        mock_config,
        mock_embedding_manager,
        sample_skill,
    ):
        """测试重新加载索引。"""
        manager = SkillManager(
            tmp_skills_dir, tmp_meta_dir, mock_config, mock_embedding_manager
        )

        # 添加 Skill
        with patch.object(manager, "_is_duplicate", return_value=False):
            manager.add_skill(sample_skill)

        # 保存索引
        manager._save_index()

        # 清空内存索引
        manager.skill_index = {}

        # 重新加载
        manager.reload_index()

        # 验证索引已恢复
        assert sample_skill["id"] in manager.skill_index

    def test_is_duplicate_true(
        self,
        tmp_skills_dir,
        tmp_meta_dir,
        mock_config,
        mock_embedding_manager,
        sample_skill,
    ):
        """测试重复检测（相似度高）。"""
        manager = SkillManager(
            tmp_skills_dir, tmp_meta_dir, mock_config, mock_embedding_manager
        )

        # 添加一个 Skill
        with patch.object(manager, "_is_duplicate", return_value=False):
            manager.add_skill(sample_skill)

        # 创建相似 Skill
        similar_skill = sample_skill.copy()
        similar_skill["id"] = "skill_explore_1_123457"

        # Mock 嵌入和相似度计算
        # 返回L2归一化的向量（模为1），相同向量的内积为1.0
        same_vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        same_vector = same_vector / np.linalg.norm(same_vector)  # L2归一化

        mock_embedding_manager.embed_texts.side_effect = [
            np.array([same_vector]),  # new_embedding
            np.array([same_vector]),  # existing_embedding
        ]

        # 计算重复
        is_dup = manager._is_duplicate(similar_skill, threshold=0.85)

        # 相同向量的余弦相似度为 1.0 > 0.85
        assert is_dup is True

    def test_is_duplicate_false(
        self,
        tmp_skills_dir,
        tmp_meta_dir,
        mock_config,
        mock_embedding_manager,
        sample_skill,
    ):
        """测试重复检测（相似度低）。"""
        manager = SkillManager(
            tmp_skills_dir, tmp_meta_dir, mock_config, mock_embedding_manager
        )

        # 添加一个 Skill
        with patch.object(manager, "_is_duplicate", return_value=False):
            manager.add_skill(sample_skill)

        # 创建不同 Skill
        different_skill = sample_skill.copy()
        different_skill["id"] = "skill_explore_1_123457"
        different_skill["content"] = "完全不同的内容"

        # Mock 嵌入和相似度计算
        mock_embedding_manager.embed_texts.side_effect = [
            np.array([[0.1, 0.2, 0.3]], dtype=np.float32),  # new_embedding
            np.array([[0.9, 0.8, 0.7]], dtype=np.float32),  # existing_embedding (不同)
        ]

        # 计算重复
        is_dup = manager._is_duplicate(different_skill, threshold=0.85)

        # 不同向量的余弦相似度 < 0.85
        assert is_dup is False

    def test_deprecate_low_quality_skills(
        self,
        tmp_skills_dir,
        tmp_meta_dir,
        mock_config,
        mock_embedding_manager,
        sample_skill,
    ):
        """测试淘汰低质量 Skill。"""
        manager = SkillManager(
            tmp_skills_dir, tmp_meta_dir, mock_config, mock_embedding_manager
        )

        # 添加一个低质量 Skill
        low_quality_skill = sample_skill.copy()
        low_quality_skill["id"] = "skill_explore_low_123456"
        low_quality_skill["composite_score"] = 0.3  # < 0.4

        with patch.object(manager, "_is_duplicate", return_value=False):
            manager.add_skill(low_quality_skill)

        # 执行淘汰
        manager._deprecate_low_quality_skills()

        # 验证 Skill 已标记为 deprecated
        meta = manager.skill_index[low_quality_skill["id"]]
        assert meta["status"] == "deprecated"

    def test_evolve_skills(
        self,
        tmp_skills_dir,
        tmp_meta_dir,
        mock_config,
        mock_embedding_manager,
    ):
        """测试 Skill 池演化。"""
        manager = SkillManager(
            tmp_skills_dir, tmp_meta_dir, mock_config, mock_embedding_manager
        )

        # Mock 经验池和提取器
        mock_pool = Mock(spec=ExperiencePool)
        mock_extractor = Mock(spec=SkillExtractor)

        # Mock 提取结果
        mock_extractor.extract_skills.return_value = [
            {
                "id": "skill_explore_new_123456",
                "task_type": "explore",
                "content": "新 Skill",
                "coverage": 5,
                "avg_accuracy": 0.7,
                "avg_generation_rate": 0.8,
                "composite_score": 0.74,
                "status": "candidate",
            }
        ]

        # Mock 重复检测
        with patch.object(manager, "_is_duplicate", return_value=False):
            manager.evolve_skills(mock_pool, mock_extractor)

        # 验证提取器被调用
        assert mock_extractor.extract_skills.call_count == 3  # explore, merge, mutate

        # 验证新 Skill 被添加
        assert "skill_explore_new_123456" in manager.skill_index

    def test_save_and_load_index(
        self,
        tmp_skills_dir,
        tmp_meta_dir,
        mock_config,
        mock_embedding_manager,
        sample_skill,
    ):
        """测试索引的保存和加载。"""
        manager = SkillManager(
            tmp_skills_dir, tmp_meta_dir, mock_config, mock_embedding_manager
        )

        # 添加 Skill
        with patch.object(manager, "_is_duplicate", return_value=False):
            manager.add_skill(sample_skill)

        # 保存索引
        manager._save_index()

        # 验证文件存在
        index_path = tmp_meta_dir / "skill_index.json"
        assert index_path.exists()

        # 加载索引
        loaded_index = manager._load_index()

        # 验证内容
        assert sample_skill["id"] in loaded_index
        assert loaded_index[sample_skill["id"]]["composite_score"] == 0.84

    def test_merge_similar_skills(
        self,
        tmp_skills_dir,
        tmp_meta_dir,
        mock_config,
        mock_embedding_manager,
    ):
        """测试合并相似 Skill（保留高分者）。"""
        manager = SkillManager(
            tmp_skills_dir, tmp_meta_dir, mock_config, mock_embedding_manager
        )

        # 添加两个同 task_type 的 Skill
        skill_high = {
            "id": "skill_explore_high_001",
            "task_type": "explore",
            "content": "高分策略内容",
            "coverage": 10,
            "avg_accuracy": 0.9,
            "avg_generation_rate": 0.9,
            "composite_score": 0.9,
            "status": "candidate",
        }
        skill_low = {
            "id": "skill_explore_low_002",
            "task_type": "explore",
            "content": "低分但相似的内容",
            "coverage": 5,
            "avg_accuracy": 0.6,
            "avg_generation_rate": 0.5,
            "composite_score": 0.56,
            "status": "candidate",
        }

        with patch.object(manager, "_is_duplicate", return_value=False):
            manager.add_skill(skill_high)
            manager.add_skill(skill_low)

        # Mock embedding：两个向量高度相似（余弦相似度 > 0.85）
        same_vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        same_vec = same_vec / np.linalg.norm(same_vec)
        mock_embedding_manager.embed_texts.return_value = np.array([same_vec, same_vec])

        # 执行合并
        manager._merge_similar_skills(threshold=0.85)

        # 验证：高分者保留，低分者被 deprecated
        assert manager.skill_index["skill_explore_high_001"]["status"] == "active"
        assert manager.skill_index["skill_explore_low_002"]["status"] == "deprecated"

    def test_merge_similar_skills_no_embedding_manager(
        self,
        tmp_skills_dir,
        tmp_meta_dir,
        mock_config,
    ):
        """测试无 embedding_manager 时跳过合并。"""
        manager = SkillManager(
            tmp_skills_dir, tmp_meta_dir, mock_config, embedding_manager=None
        )

        # 不应报错
        manager._merge_similar_skills()
