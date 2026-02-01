"""Skill 池管理器模块。

负责 Skill 的质量评估、演化（新增/合并/淘汰）和元数据维护。
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

from core.evolution.experience_pool import ExperiencePool
from core.evolution.skill_extractor import SkillExtractor
from core.evolution.code_embedding_manager import CodeEmbeddingManager
from utils.config import Config
from utils.logger_system import log_msg, log_json


class SkillManager:
    """Skill 池管理器。

    负责 Skill 的添加、评估、演化和元数据管理。

    核心功能:
        - 添加新 Skill（检测重复）
        - 评估 Skill 质量（综合评分）
        - 演化 Skill 池（新增/合并/淘汰）
        - 获取 Top-K Skill

    Attributes:
        skills_dir: Skill 文件根目录
        meta_dir: 元数据目录
        config: 全局配置
        embedding_manager: 文本嵌入管理器
        skill_index: Skill 索引（skill_id -> 元数据）
    """

    def __init__(
        self,
        skills_dir: Path,
        meta_dir: Path,
        config: Config,
        embedding_manager: Optional[CodeEmbeddingManager] = None,
    ):
        """初始化 SkillManager。

        Args:
            skills_dir: Skill 文件根目录
            meta_dir: 元数据目录
            config: 全局配置
            embedding_manager: 文本嵌入管理器（可选，None 时自动创建）
        """
        self.skills_dir = Path(skills_dir)
        self.meta_dir = Path(meta_dir)
        self.config = config
        self.embedding_manager = embedding_manager or CodeEmbeddingManager()

        # 确保目录存在
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

        # 加载索引
        self.skill_index: Dict[str, Dict[str, Any]] = self._load_index()

        # 演化参数
        self.duplicate_threshold = config.evolution.skill.duplicate_threshold
        self.min_composite_score = config.evolution.skill.min_composite_score
        self.deprecate_threshold = config.evolution.skill.deprecate_threshold
        self.unused_epochs = config.evolution.skill.unused_epochs

        log_msg(
            "INFO",
            f"SkillManager 初始化完成（已加载 {len(self.skill_index)} 个 Skill）",
        )

    def add_skill(self, skill: Dict[str, Any]) -> bool:
        """添加新 Skill。

        Args:
            skill: Skill 数据（来自 SkillExtractor）

        Returns:
            是否成功添加
        """
        skill_id = skill["id"]
        task_type = skill["task_type"]

        # [1] 检测重复
        if self._is_duplicate(skill):
            log_msg("WARNING", f"Skill {skill_id} 重复，跳过")
            return False

        # [2] 写入文件
        skill_path = (
            self.skills_dir
            / "by_task_type"
            / task_type
            / "success_patterns"
            / f"{skill_id}.md"
        )
        skill_path.parent.mkdir(parents=True, exist_ok=True)
        skill_path.write_text(skill["content"], encoding="utf-8")

        # [3] 更新索引
        self.skill_index[skill_id] = {
            "task_type": task_type,
            "coverage": skill["coverage"],
            "avg_accuracy": skill["avg_accuracy"],
            "avg_generation_rate": skill["avg_generation_rate"],
            "composite_score": skill["composite_score"],
            "status": "active",
            "created_at": time.time(),
            "last_used": None,
            "usage_count": 0,
            "file_path": str(skill_path.relative_to(self.skills_dir)),
        }

        self._save_index()

        log_msg(
            "INFO",
            f"Skill {skill_id} 添加成功（综合评分: {skill['composite_score']:.3f}）",
        )
        log_json(
            {
                "event": "skill_added",
                "skill_id": skill_id,
                "task_type": task_type,
                "composite_score": skill["composite_score"],
            }
        )

        return True

    def evaluate_skill(self, skill_id: str) -> float:
        """计算 Skill 综合评分。

        使用 Reference/Swarm-Evo 公式:
            composite_score = 0.6 × avg_accuracy + 0.4 × avg_generation_rate

        Args:
            skill_id: Skill ID

        Returns:
            综合评分
        """
        meta = self.skill_index.get(skill_id)
        if not meta:
            return 0.0

        composite_score = 0.6 * meta["avg_accuracy"] + 0.4 * meta["avg_generation_rate"]

        return composite_score

    def evolve_skills(
        self, experience_pool: ExperiencePool, extractor: SkillExtractor
    ) -> None:
        """Skill 池演化（新增/合并/淘汰）。

        Args:
            experience_pool: 共享经验池
            extractor: Skill 提取器
        """
        log_msg("INFO", "===== 开始 Skill 池演化 =====")

        # [1] 新增：提取新 Skill
        for task_type in ["explore", "merge", "mutate"]:
            new_skills = extractor.extract_skills(task_type)

            for skill in new_skills:
                # 质量过滤
                if skill["composite_score"] >= self.min_composite_score:
                    self.add_skill(skill)

        # [2] 合并：检测相似 Skill
        self._merge_similar_skills()

        # [3] 淘汰：移除低质量 Skill
        self._deprecate_low_quality_skills()

        log_msg(
            "INFO",
            f"===== Skill 池演化完成（当前 {len(self.skill_index)} 个 Skill） =====",
        )

    def get_top_k_skills(self, task_type: str, k: int) -> List[str]:
        """获取 Top-K Skill 内容。

        Args:
            task_type: 任务类型
            k: Top-K 数量

        Returns:
            Skill 内容列表
        """
        # 筛选任务类型和状态
        candidates = [
            (skill_id, meta)
            for skill_id, meta in self.skill_index.items()
            if meta["task_type"] == task_type and meta["status"] == "active"
        ]

        # 按综合评分排序
        candidates.sort(key=lambda x: x[1]["composite_score"], reverse=True)

        # 取 Top-K
        top_k = candidates[:k]

        # 加载 Skill 内容
        skills = []
        for skill_id, meta in top_k:
            content = self._load_skill_content(skill_id)
            if content:
                skills.append(content)

        log_msg(
            "INFO",
            f"获取 Top-{k} Skill（task_type={task_type}），实际返回 {len(skills)} 个",
        )

        return skills

    def reload_index(self) -> None:
        """重新加载 Skill 索引。"""
        self.skill_index = self._load_index()
        log_msg("INFO", f"Skill 索引已重新加载（{len(self.skill_index)} 个 Skill）")

    def _is_duplicate(self, skill: Dict[str, Any], threshold: float = None) -> bool:
        """检测 Skill 是否重复（语义相似度）。

        Args:
            skill: 新 Skill
            threshold: 相似度阈值（None 时使用配置值）

        Returns:
            是否重复
        """
        if threshold is None:
            threshold = self.duplicate_threshold

        task_type = skill["task_type"]
        new_content = skill["content"]

        # 向量化新 Skill
        new_embedding = self.embedding_manager.embed_texts([new_content])[0]

        # 与现有 Skill 比较
        for existing_id, meta in self.skill_index.items():
            if meta["task_type"] != task_type:
                continue

            existing_content = self._load_skill_content(existing_id)
            if not existing_content:
                continue

            existing_embedding = self.embedding_manager.embed_texts([existing_content])[
                0
            ]

            # 余弦相似度（L2 归一化后的内积）
            similarity = float(np.dot(new_embedding, existing_embedding))

            if similarity > threshold:
                log_msg(
                    "INFO",
                    f"检测到重复 Skill（相似度: {similarity:.3f} > {threshold}）",
                )
                return True

        return False

    def _merge_similar_skills(self, threshold: float = None) -> None:
        """合并相似 Skill。

        Args:
            threshold: 相似度阈值（None 时使用配置值）
        """
        if threshold is None:
            threshold = self.duplicate_threshold

        log_msg("INFO", "开始合并相似 Skill...")

        # 简化实现：仅检测并标记，不执行实际合并
        # TODO: 实现 LLM 驱动的 Skill 合并逻辑

        log_msg("INFO", "相似 Skill 合并完成")

    def _deprecate_low_quality_skills(self) -> None:
        """淘汰低质量 Skill。"""
        log_msg("INFO", "开始淘汰低质量 Skill...")

        deprecated_count = 0

        for skill_id, meta in list(self.skill_index.items()):
            # [1] 连续未使用
            if meta["usage_count"] == 0:
                epochs_since_creation = (
                    time.time() - meta["created_at"]
                ) / 3600  # 简化：每小时 = 1 Epoch
                if epochs_since_creation >= self.unused_epochs:
                    self._move_to_deprecated(skill_id)
                    deprecated_count += 1
                    continue

            # [2] 综合评分过低
            if meta["composite_score"] < self.deprecate_threshold:
                self._move_to_deprecated(skill_id)
                deprecated_count += 1

        log_msg("INFO", f"淘汰低质量 Skill 完成（淘汰 {deprecated_count} 个）")

    def _move_to_deprecated(self, skill_id: str) -> None:
        """将 Skill 标记为 deprecated。

        Args:
            skill_id: Skill ID
        """
        if skill_id not in self.skill_index:
            return

        self.skill_index[skill_id]["status"] = "deprecated"
        self._save_index()

        log_msg("INFO", f"Skill {skill_id} 已标记为 deprecated")

    def _load_skill_content(self, skill_id: str) -> Optional[str]:
        """加载 Skill 文件内容。

        Args:
            skill_id: Skill ID

        Returns:
            Skill 内容（Markdown 格式），失败返回 None
        """
        meta = self.skill_index.get(skill_id)
        if not meta:
            return None

        skill_path = self.skills_dir / meta["file_path"]

        if not skill_path.exists():
            log_msg("WARNING", f"Skill 文件不存在: {skill_path}")
            return None

        return skill_path.read_text(encoding="utf-8")

    def _save_index(self) -> None:
        """保存 Skill 索引到 JSON。"""
        index_path = self.meta_dir / "skill_index.json"

        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(self.skill_index, f, ensure_ascii=False, indent=2)

    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """从 JSON 加载 Skill 索引。

        Returns:
            Skill 索引
        """
        index_path = self.meta_dir / "skill_index.json"

        if not index_path.exists():
            return {}

        try:
            with open(index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            log_msg("WARNING", f"Skill 索引加载失败: {e}")
            return {}
