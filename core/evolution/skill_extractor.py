"""Skill 提取器模块。

从经验池中提取成功策略模式，使用 HDBSCAN 聚类 + LLM 总结生成 Skill。
"""

import time
from typing import List, Dict, Any

import numpy as np
import hdbscan

from core.evolution.experience_pool import ExperiencePool, TaskRecord
from core.evolution.code_embedding_manager import CodeEmbeddingManager
from core.backend import query
from utils.config import Config
from utils.logger_system import log_msg, log_json


class SkillExtractor:
    """Skill 提取器。

    从经验池中提取成功策略模式，生成可复用的 Skill。

    工作流程:
        1. 从经验池查询成功记录（output_quality > 0）
        2. 提取 strategy_summary 并使用 bge-m3 向量化
        3. HDBSCAN 聚类（min_cluster_size=5）
        4. 每个簇调用 LLM 总结生成 Skill Markdown
        5. 返回 Skill 列表

    Attributes:
        experience_pool: 共享经验池
        embedding_manager: 文本嵌入管理器
        config: 全局配置
        min_cluster_size: HDBSCAN 最小簇大小
    """

    def __init__(self, experience_pool: ExperiencePool, config: Config):
        """初始化 Skill 提取器。

        Args:
            experience_pool: 共享经验池
            config: 全局配置
        """
        self.experience_pool = experience_pool
        self.embedding_manager = CodeEmbeddingManager()
        self.config = config

        # 聚类参数
        self.min_cluster_size = config.evolution.skill.min_cluster_size

        log_msg(
            "INFO",
            f"SkillExtractor 初始化完成（min_cluster_size={self.min_cluster_size}）",
        )

    def extract_skills(
        self, task_type: str, min_cluster_size: int = None
    ) -> List[Dict[str, Any]]:
        """从经验池提取 Skill。

        Args:
            task_type: 任务类型（explore/merge/mutate）
            min_cluster_size: 最小簇大小（None 时使用配置值）

        Returns:
            Skill 列表，每个 Skill 包含:
                - id: Skill 唯一标识
                - task_type: 任务类型
                - content: Skill 内容（Markdown）
                - coverage: 覆盖的记录数
                - avg_accuracy: 平均准确率
                - avg_generation_rate: 平均生成率
                - composite_score: 综合评分
                - status: 状态（candidate）
        """
        if min_cluster_size is None:
            min_cluster_size = self.min_cluster_size

        log_msg(
            "INFO",
            f"开始提取 {task_type} 类型的 Skill（min_cluster_size={min_cluster_size}）",
        )

        # [1] 获取成功案例
        records = self.experience_pool.query(
            task_type=task_type, k=0, filters={"output_quality": (">", 0)}
        )

        if len(records) < min_cluster_size:
            log_msg(
                "WARNING",
                f"记录数不足（{len(records)} < {min_cluster_size}），跳过 Skill 提取",
            )
            return []

        log_msg("INFO", f"从经验池获取 {len(records)} 条成功记录")

        # [2] 提取 strategy_summary 并向量化
        strategies = [r.strategy_summary for r in records]
        embeddings = self._embed_texts(strategies)

        # [3] HDBSCAN 聚类
        clusters = self._cluster(embeddings, min_cluster_size)

        if not clusters:
            log_msg("WARNING", "聚类结果为空，未发现明显的策略模式")
            return []

        log_msg("INFO", f"聚类完成，发现 {len(clusters)} 个策略簇")

        # [4] LLM 总结每个簇
        skills = []
        for cluster_id, indices in clusters.items():
            cluster_strategies = [strategies[i] for i in indices]

            # LLM 总结
            skill_content = self._summarize_cluster(cluster_strategies, task_type)

            # 计算簇的质量指标
            avg_accuracy = self._calc_avg_accuracy(indices, records)
            avg_generation_rate = self._calc_generation_rate(indices, records)
            composite_score = 0.6 * avg_accuracy + 0.4 * avg_generation_rate

            # 生成 Skill ID
            skill_id = f"skill_{task_type}_{cluster_id}_{int(time.time())}"

            skill = {
                "id": skill_id,
                "task_type": task_type,
                "content": skill_content,
                "coverage": len(indices),
                "avg_accuracy": avg_accuracy,
                "avg_generation_rate": avg_generation_rate,
                "composite_score": composite_score,
                "status": "candidate",
            }

            skills.append(skill)

            log_json(
                {
                    "event": "skill_extracted",
                    "skill_id": skill_id,
                    "task_type": task_type,
                    "coverage": len(indices),
                    "composite_score": composite_score,
                }
            )

        log_msg("INFO", f"提取完成，共生成 {len(skills)} 个 Skill")
        return skills

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """文本向量化（bge-m3）。

        Args:
            texts: 文本列表

        Returns:
            L2 归一化的 embeddings
        """
        log_msg("INFO", f"开始向量化 {len(texts)} 个文本...")
        embeddings = self.embedding_manager.embed_texts(texts)
        log_msg("INFO", f"向量化完成，embeddings 形状: {embeddings.shape}")
        return embeddings

    def _cluster(
        self, embeddings: np.ndarray, min_cluster_size: int
    ) -> Dict[int, List[int]]:
        """HDBSCAN 聚类。

        Args:
            embeddings: 文本向量
            min_cluster_size: 最小簇大小

        Returns:
            聚类结果，{cluster_id: [record_indices]}
        """
        log_msg("INFO", f"开始 HDBSCAN 聚类（min_cluster_size={min_cluster_size}）...")

        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        labels = clusterer.fit_predict(embeddings)

        # 组织结果（排除噪声点 label=-1）
        clusters = {}
        for i, label in enumerate(labels):
            if label == -1:  # 噪声点
                continue
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(i)

        log_msg(
            "INFO",
            f"聚类完成，发现 {len(clusters)} 个簇，噪声点数: {np.sum(labels == -1)}",
        )

        return clusters

    def _summarize_cluster(self, strategies: List[str], task_type: str) -> str:
        """LLM 总结策略簇为 Skill。

        Args:
            strategies: 策略列表
            task_type: 任务类型

        Returns:
            Skill 内容（Markdown 格式）
        """
        log_msg("INFO", f"开始 LLM 总结（{len(strategies)} 个策略）...")

        # 构建 Prompt
        prompt = f"""你是一个 Skill 生成器。以下是多个成功的 {task_type} 策略：

{chr(10).join(f"{i + 1}. {s}" for i, s in enumerate(strategies))}

**任务**: 总结这些策略的共性，生成一个可复用的 Skill。

**输出格式**（Markdown）:
```markdown
## {task_type.capitalize()} Skill: <简短标题>

### 核心策略
- ...

### 示例
- ...

### 注意事项
- ...
```

**要求**:
1. 提取共性策略，而非罗列所有细节
2. 描述清晰、可操作
3. 保持简洁（200-400 字）
"""

        # 调用 LLM
        try:
            response = query(
                model=self.config.llm.code.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
            )

            skill_content = response.strip()
            log_msg("INFO", f"LLM 总结完成，生成 {len(skill_content)} 字符")
            return skill_content

        except Exception as e:
            log_msg("ERROR", f"LLM 总结失败: {e}")
            # Fallback: 返回简单格式
            return self._fallback_summary(strategies, task_type)

    def _fallback_summary(self, strategies: List[str], task_type: str) -> str:
        """备用总结策略（LLM 失败时）。

        Args:
            strategies: 策略列表
            task_type: 任务类型

        Returns:
            简单的 Skill 内容
        """
        content = f"## {task_type.capitalize()} Skill: 策略模式\n\n"
        content += "### 核心策略\n"
        for i, strategy in enumerate(strategies[:3], 1):
            content += f"{i}. {strategy[:100]}...\n"
        content += "\n### 注意事项\n- 需要根据具体情况调整\n"
        return content

    def _calc_avg_accuracy(
        self, indices: List[int], records: List[TaskRecord]
    ) -> float:
        """计算簇平均准确率。

        Args:
            indices: 簇中记录的索引
            records: 所有记录

        Returns:
            平均 output_quality
        """
        qualities = [records[i].output_quality for i in indices]
        return np.mean(qualities) if qualities else 0.0

    def _calc_generation_rate(
        self, indices: List[int], records: List[TaskRecord]
    ) -> float:
        """计算簇平均生成率。

        Args:
            indices: 簇中记录的索引
            records: 所有记录

        Returns:
            有效生成的比例（output_quality > 0）
        """
        valid_count = sum(1 for i in indices if records[i].output_quality > 0)
        return valid_count / len(indices) if indices else 0.0
