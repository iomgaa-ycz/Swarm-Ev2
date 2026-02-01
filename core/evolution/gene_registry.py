"""Infrastructure for tracking gene-level pheromones."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Dict, List, Optional

from core.state import Node

# 基因位点名称
_LOCUS_NAMES = [
    "DATA",
    "MODEL",
    "LOSS",
    "OPTIMIZER",
    "REGULARIZATION",
    "INITIALIZATION",
    "TRAINING_TRICKS",
]


def normalize_gene_text(text: str) -> str:
    """
    归一化基因文本用于稳定哈希。

    Args:
        text: 原始基因文本

    Returns:
        归一化后的文本
    """
    if text is None:
        return ""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    normalized = normalized.strip()
    # 将连续空行合并为单个空行
    normalized = re.sub(r"\n\s*\n+", "\n\n", normalized)
    return normalized


def compute_gene_id(text: str) -> str:
    """
    计算基因 ID（SHA1 hash 前 12 位）。

    Args:
        text: 基因文本

    Returns:
        基因 ID
    """
    normalized = normalize_gene_text(text)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]


class GeneRegistry:
    """
    追踪每个基因位点的信息素统计。

    存储结构：
    _registry[locus][gene_id] = {
        "pheromone": float,      # 平均信息素值
        "acc_sum": float,        # 累计信息素总和
        "count": int,            # 出现次数
        "last_seen_step": int,   # 最后出现步骤
        "content": str,          # 基因内容
        "source_node_id": str,   # 来源节点 ID
    }
    """

    def __init__(self) -> None:
        """初始化基因注册表。"""
        self._registry: Dict[str, Dict[str, Dict[str, Any]]] = {
            locus: {} for locus in _LOCUS_NAMES
        }

    def update_from_reviewed_node(self, node: Node) -> None:
        """
        从已评估节点更新基因信息素。

        Args:
            node: 已评估的节点对象
        """
        # 获取节点信息素
        pheromone_value = getattr(node, "pheromone_node", None)
        if pheromone_value is None and node.metadata:
            pheromone_value = node.metadata.get("pheromone_node")
        if pheromone_value is None:
            return

        # 仅处理有效节点
        if node.metric_value is None or node.is_buggy:
            return

        # 更新每个基因位点
        for locus in _LOCUS_NAMES:
            gene_content = node.genes.get(locus) if node.genes else None

            if not gene_content:
                continue

            normalized = normalize_gene_text(gene_content)
            if not normalized:
                continue

            gene_id = compute_gene_id(normalized)

            # 获取或创建基因条目
            entry = self._registry[locus].setdefault(
                gene_id,
                {
                    "pheromone": 0.1,
                    "acc_sum": 0.0,
                    "count": 0,
                    "last_seen_step": -1,
                    "content": gene_content,
                    "source_node_id": node.id,
                },
            )

            # 更新信息素（累积平均）
            entry["acc_sum"] += float(pheromone_value)
            entry["count"] += 1
            entry["pheromone"] = entry["acc_sum"] / entry["count"]
            entry["last_seen_step"] = node.step
            entry["content"] = gene_content
            entry["source_node_id"] = node.id

    def get_gene_pheromone(
        self,
        locus: str,
        gene_id: str,
        default_init: float = 0.1,
        current_step: Optional[int] = None,
        decay_rate: float = 0.03,
    ) -> float:
        """
        返回时间衰减的基因信息素。

        公式: pheromone_eff = pheromone * exp(-decay_rate * (current_step - last_seen_step))

        Args:
            locus: 基因位点名称
            gene_id: 基因 ID
            default_init: 默认初始值
            current_step: 当前步骤
            decay_rate: 衰减率（默认 0.03）

        Returns:
            时间衰减的信息素值
        """
        locus_entries = self._registry.get(locus)
        if not locus_entries:
            return default_init

        entry = locus_entries.get(gene_id)
        if not entry:
            return default_init

        pheromone = float(entry.get("pheromone", default_init))

        # 如果没有 step 信息，返回原始值
        if current_step is None:
            return pheromone

        last_seen_step = entry.get("last_seen_step", -1)
        if last_seen_step < 0:
            return pheromone

        # 计算时间衰减
        step_diff = max(0, current_step - last_seen_step)
        decay = math.exp(-decay_rate * step_diff)

        return pheromone * decay

    def build_gene_pools(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        构建所有基因位点的基因池。

        Returns:
            Dict[locus, List[gene_info]]: 每个位点的基因列表
        """
        pools: Dict[str, List[Dict[str, Any]]] = {locus: [] for locus in _LOCUS_NAMES}

        for locus, entries in self._registry.items():
            for gene_id, record in entries.items():
                pools[locus].append(
                    {
                        "gene_id": gene_id,
                        "content": record.get("content", ""),
                        "pheromone": record.get("pheromone", 0.1),
                        "source_node_id": record.get("source_node_id"),
                        "last_seen_step": record.get("last_seen_step", -1),
                    }
                )

        return pools
