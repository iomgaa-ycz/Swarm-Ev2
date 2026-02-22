"""
Deterministic per-locus gene selection using
pheromone-based pure quality ranking (exploitation-only).

Quality is computed as a weighted combination of:
- node-level pheromone (reflecting score, success, and recency)
- gene-level pheromone (time-decayed historical utility)

No diversity regularization or similarity-based penalty is applied
during merge-stage gene selection.

Top-1 gene is selected independently for each locus.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List

from core.evolution.gene_registry import (
    GeneRegistry,
    compute_gene_id,
    normalize_gene_text,
)
from core.state import Journal, Node
from utils.logger_system import log_msg

# =========================
# Constants & Hyperparams
# =========================

LOCUS_TO_FIELD = {
    "DATA": "data_source",
    "MODEL": "model_source",
    "LOSS": "loss_source",
    "OPTIMIZER": "optimizer_source",
    "REGULARIZATION": "regularization_source",
    "INITIALIZATION": "initialization_source",
    "TRAINING_TRICKS": "tricks_source",
}

ALL_LOCI = list(LOCUS_TO_FIELD.keys())

DEFAULT_INIT_PHEROMONE = 0.1

# Quality blend (30% gene pheromone + 70% node pheromone)
QUALITY_BLEND = 0.3


# =========================
# Data structure
# =========================


@dataclass
class GeneItem:
    """基因项（用于基因池构建和选择）。"""

    locus: str
    gene_id: str
    content: str  # normalized (for embedding / similarity)
    raw_content: str  # original code (for merge)
    source_node_id: str
    gene_pheromone: float
    node_pheromone: float
    source_score: float
    created_step: int
    quality: float = 0.0


# =========================
# Public API
# =========================


def select_gene_plan(
    journal: Journal,
    gene_registry: GeneRegistry,
    current_step: int,
) -> Dict[str, Any]:
    """
    Select a merge-compatible gene plan using
    Quality top-1 per locus.

    Args:
        journal: Journal 对象（包含所有节点）
        gene_registry: 基因注册表
        current_step: 当前步骤

    Returns:
        基因计划字典，格式：
        {
            "reasoning": str,
            "data_source": {
                "locus": "DATA",
                "source_node_id": "node_xyz",
                "gene_id": "abc123",
                "code": "..."
            },
            "model_source": {...},
            ...
        }
    """
    pools = build_decision_gene_pools(
        journal,
        gene_registry,
        current_step=current_step,
    )

    for locus, items in pools.items():
        log_msg(
            "INFO",
            f"[POOL] {locus}: size={len(items)} genes={[i.gene_id[:6] for i in items]}",
        )

    winners: Dict[str, GeneItem] = {}
    for locus in ALL_LOCI:
        items = pools.get(locus, [])
        if not items:
            raise ValueError(f"Missing locus: {locus}")
        winner = _select_locus_winner(items)
        winners[locus] = winner

    gene_plan: Dict[str, Any] = {
        "reasoning": (
            "Per-locus merge selection using pure quality ranking "
            "(exploitation-only, no diversity regularization)."
        )
    }

    for locus, item in winners.items():
        field_name = LOCUS_TO_FIELD[locus]
        gene_plan[field_name] = {
            "locus": locus,
            "source_node_id": item.source_node_id,
            "gene_id": item.gene_id,
            "code": item.raw_content,
            "source_score": item.source_score,
        }

    return gene_plan


# =========================
# Gene pool construction
# =========================


def build_decision_gene_pools(
    journal: Journal,
    gene_registry: GeneRegistry,
    current_step: int,
) -> Dict[str, List[GeneItem]]:
    """
    构建决策基因池。

    Args:
        journal: Journal 对象
        gene_registry: 基因注册表
        current_step: 当前步骤

    Returns:
        每个基因位点的基因项列表
    """
    pools: Dict[str, Dict[str, GeneItem]] = {locus: {} for locus in ALL_LOCI}

    # 处理 Journal.nodes (list)
    nodes_iter = (
        journal.nodes if isinstance(journal.nodes, list) else journal.nodes.values()
    )
    for node in nodes_iter:
        if not _is_valid_node(node):
            continue

        for locus in ALL_LOCI:
            raw_content = (node.genes or {}).get(locus)
            if not raw_content:
                continue

            normalized = normalize_gene_text(raw_content)
            if not normalized:
                continue

            gene_id = compute_gene_id(normalized)
            gene_pheromone = gene_registry.get_gene_pheromone(
                locus,
                gene_id,
                DEFAULT_INIT_PHEROMONE,
                current_step=current_step,
            )

            node_pheromone = 0.0
            if node.metadata:
                node_pheromone = float(node.metadata.get("pheromone_node", 0.0))

            item = GeneItem(
                locus=locus,
                gene_id=gene_id,
                content=normalized,
                raw_content=raw_content,
                source_node_id=node.id,
                gene_pheromone=float(gene_pheromone),
                node_pheromone=float(node_pheromone),
                source_score=float(
                    node.metric_value if node.metric_value is not None else 0.0
                ),
                created_step=int(node.step),
            )
            item.quality = _compute_quality(item)

            existing = pools[locus].get(gene_id)
            if existing is None or _is_better_item(item, existing):
                pools[locus][gene_id] = item

    return {locus: list(items.values()) for locus, items in pools.items()}


# =========================
# Core selection logic
# =========================


def _select_locus_winner(items: List[GeneItem]) -> GeneItem:
    """
    Merge-only selection:
    pure exploitation, no diversity regularization.

    Args:
        items: 候选基因项列表

    Returns:
        质量最高的基因项
    """
    scored = []
    for item in items:
        scored.append((item.quality, item))

    scored.sort(
        key=lambda x: (
            -x[0],  # quality（主排序）
            -x[1].node_pheromone,  # 来自更强 node
            -x[1].source_score,  # 更高原始得分
            x[1].created_step,  # 更新的 gene 优先
            x[1].source_node_id,  # 稳定 tie-break
            x[1].gene_id,
        )
    )

    winner = scored[0][1]

    log_msg(
        "INFO",
        f"[MERGE-WINNER] {winner.locus}: "
        f"gene={winner.gene_id[:6]} "
        f"quality={winner.quality:.4f} "
        f"node={winner.source_node_id[:6]}",
    )

    return winner


# =========================
# Utilities
# =========================


def _is_valid_node(node: Node) -> bool:
    """
    检查节点是否有效（用于基因池构建）。

    Args:
        node: 节点对象

    Returns:
        是否有效
    """
    if node.metric_value is None or node.is_buggy:
        return False
    if not node.code:
        return False
    if not node.genes:
        return False
    review_success = node.metadata.get("review_success") if node.metadata else None
    if review_success is not None and not review_success:
        return False
    return True


def _is_better_item(candidate: GeneItem, incumbent: GeneItem) -> bool:
    """
    比较两个基因项，判断候选者是否更好。

    Args:
        candidate: 候选基因项
        incumbent: 现有基因项

    Returns:
        候选者是否更好
    """
    if candidate.gene_pheromone != incumbent.gene_pheromone:
        return candidate.gene_pheromone > incumbent.gene_pheromone
    if candidate.node_pheromone != incumbent.node_pheromone:
        return candidate.node_pheromone > incumbent.node_pheromone
    if candidate.source_score != incumbent.source_score:
        return candidate.source_score > incumbent.source_score
    return candidate.source_node_id < incumbent.source_node_id


def _compute_quality(item: GeneItem) -> float:
    """
    计算基因质量（30% gene pheromone + 70% node pheromone）。

    Args:
        item: 基因项

    Returns:
        质量值
    """
    return (
        QUALITY_BLEND * item.gene_pheromone
        + (1.0 - QUALITY_BLEND) * item.node_pheromone
    )


# ══════════════════════════════════════════════════════
# Pheromone + 退化检测（新架构 Phase 2 Merge 使用）
# ══════════════════════════════════════════════════════


def pheromone_with_degenerate_check(
    journal: Journal,
    gene_registry: GeneRegistry,
    current_step: int,
) -> Dict[str, Any]:
    """Pheromone 全局 TOP-1 选择 + 退化检测。

    若 7 个基因全来自同一节点（退化），则将每个位点替换为
    来自其他节点的次优基因（second-best），以避免 merge 退化为复制。

    Args:
        journal: Journal 对象
        gene_registry: 基因注册表
        current_step: 当前步骤

    Returns:
        基因计划字典（与 select_gene_plan 格式兼容）
    """
    # Step 1: 标准 pheromone 全局 TOP-1（不传父代，全局视野）
    gene_plan = select_gene_plan(journal, gene_registry, current_step)

    # Step 2: 退化检测
    source_ids = {
        v["source_node_id"]
        for v in gene_plan.values()
        if isinstance(v, dict) and "source_node_id" in v
    }

    if len(source_ids) != 1:
        return gene_plan  # 来源多样，无退化

    dominant_id = next(iter(source_ids))
    log_msg("WARNING", f"[DEGENERATE] 7 基因全来自节点 {dominant_id[:8]}，启动退化检测替换")

    pools = build_decision_gene_pools(journal, gene_registry, current_step)

    for locus, field_name in LOCUS_TO_FIELD.items():
        others = [
            item for item in pools.get(locus, [])
            if item.source_node_id != dominant_id
        ]
        if others:
            second_best = max(others, key=lambda x: x.quality)
            gene_plan[field_name] = {
                "locus": locus,
                "source_node_id": second_best.source_node_id,
                "gene_id": second_best.gene_id,
                "code": second_best.raw_content,
                "source_score": second_best.source_score,
            }
            log_msg(
                "INFO",
                f"[DEGENERATE FIX] {locus}: 替换为 {second_best.source_node_id[:8]} "
                f"(quality={second_best.quality:.4f})",
            )
        # 若无其他来源候选，保留原 dominant（不强行替换）

    return gene_plan


def get_primary_parent(gene_plan: Dict[str, Any], journal: Journal) -> "Node":
    """从基因计划中推断主父代（贡献基因数最多的节点）。

    用于 merge.j2 的单一参考父代。选择规则：
    1. 统计每个 source_node_id 贡献的基因数量
    2. 取贡献最多的节点
    3. 并列时选 metric_value 最高的节点

    Args:
        gene_plan: pheromone_with_degenerate_check() 返回的基因计划
        journal: Journal 对象（用于查找节点）

    Returns:
        贡献基因最多的 Node 对象

    Raises:
        ValueError: gene_plan 为空或 journal 中找不到对应节点
    """
    # 统计各 source_node_id 出现次数
    source_counts = Counter(
        v["source_node_id"]
        for v in gene_plan.values()
        if isinstance(v, dict) and "source_node_id" in v
    )

    if not source_counts:
        raise ValueError("gene_plan 中无有效的 source_node_id")

    max_count = max(source_counts.values())
    candidates = [nid for nid, cnt in source_counts.items() if cnt == max_count]

    # 快速路径：唯一候选
    if len(candidates) == 1:
        node = journal.get_node_by_id(candidates[0])
        if node is None:
            raise ValueError(f"Journal 中找不到节点: {candidates[0][:8]}")
        return node

    # 并列时选 metric 最高的节点
    best_id = None
    best_metric = float("-inf")
    for nid in candidates:
        node = journal.get_node_by_id(nid)
        if node is None:
            continue
        metric = node.metric_value or 0.0
        if metric > best_metric:
            best_metric = metric
            best_id = nid

    if best_id is None:
        # fallback: 取 gene_plan 中第一个出现的 source_node_id
        for v in gene_plan.values():
            if isinstance(v, dict) and "source_node_id" in v:
                best_id = v["source_node_id"]
                break

    node = journal.get_node_by_id(best_id)
    if node is None:
        raise ValueError(f"Journal 中找不到节点: {best_id[:8]}")
    return node
