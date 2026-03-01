"""
gene_selector lower_is_better 方向测试。

测试目标：
- _is_better_item 根据方向正确比较 source_score
- _select_locus_winner 根据方向正确排序
"""

import pytest

from core.evolution.gene_selector import (
    GeneItem,
    _is_better_item,
    _select_locus_winner,
)


# ── 辅助函数 ──────────────────────────────────────────────────


def _make_item(
    gene_id: str = "g1",
    source_score: float = 0.5,
    lower_is_better: bool = False,
    node_pheromone: float = 0.5,
    gene_pheromone: float = 0.5,
    created_step: int = 0,
    source_node_id: str = "node_a",
) -> GeneItem:
    """构造 GeneItem。"""
    item = GeneItem(
        locus="DATA",
        gene_id=gene_id,
        content="normalized",
        raw_content="raw",
        source_node_id=source_node_id,
        gene_pheromone=gene_pheromone,
        node_pheromone=node_pheromone,
        source_score=source_score,
        lower_is_better=lower_is_better,
        created_step=created_step,
    )
    item.quality = 0.3 * gene_pheromone + 0.7 * node_pheromone
    return item


# ── _is_better_item 方向测试 ──────────────────────────────────


class TestIsBetterItemDirection:

    def test_higher_is_better_larger_score_wins(self):
        """higher_is_better 时 source_score 大者胜出（现有行为）。"""
        candidate = _make_item(
            gene_id="c", source_score=0.9, gene_pheromone=0.5, node_pheromone=0.5
        )
        incumbent = _make_item(
            gene_id="i", source_score=0.3, gene_pheromone=0.5, node_pheromone=0.5
        )
        assert _is_better_item(candidate, incumbent) is True

    def test_higher_is_better_smaller_score_loses(self):
        """higher_is_better 时 source_score 小者不胜。"""
        candidate = _make_item(
            gene_id="c", source_score=0.3, gene_pheromone=0.5, node_pheromone=0.5
        )
        incumbent = _make_item(
            gene_id="i", source_score=0.9, gene_pheromone=0.5, node_pheromone=0.5
        )
        assert _is_better_item(candidate, incumbent) is False

    def test_lower_is_better_smaller_score_wins(self):
        """lower_is_better 时 source_score 小者胜出。"""
        candidate = _make_item(
            gene_id="c",
            source_score=0.05,
            lower_is_better=True,
            gene_pheromone=0.5,
            node_pheromone=0.5,
        )
        incumbent = _make_item(
            gene_id="i",
            source_score=0.30,
            lower_is_better=True,
            gene_pheromone=0.5,
            node_pheromone=0.5,
        )
        assert _is_better_item(candidate, incumbent) is True

    def test_lower_is_better_larger_score_loses(self):
        """lower_is_better 时 source_score 大者不胜。"""
        candidate = _make_item(
            gene_id="c",
            source_score=0.30,
            lower_is_better=True,
            gene_pheromone=0.5,
            node_pheromone=0.5,
        )
        incumbent = _make_item(
            gene_id="i",
            source_score=0.05,
            lower_is_better=True,
            gene_pheromone=0.5,
            node_pheromone=0.5,
        )
        assert _is_better_item(candidate, incumbent) is False


# ── _select_locus_winner 方向测试 ─────────────────────────────


class TestSelectLocusWinnerDirection:

    def test_lower_is_better_small_score_preferred(self):
        """RMSE 场景：quality 相同时 source_score 小者优先。"""
        item_good = _make_item(
            gene_id="good",
            source_score=0.05,
            lower_is_better=True,
            node_pheromone=0.5,
            gene_pheromone=0.5,
            source_node_id="node_good",
        )
        item_bad = _make_item(
            gene_id="bad",
            source_score=0.30,
            lower_is_better=True,
            node_pheromone=0.5,
            gene_pheromone=0.5,
            source_node_id="node_bad",
        )

        winner = _select_locus_winner([item_bad, item_good])
        assert winner.gene_id == "good", (
            f"RMSE 场景应选 source_score 更小的 good，实际选了 {winner.gene_id}"
        )

    def test_higher_is_better_large_score_preferred(self):
        """AUC 场景：quality 相同时 source_score 大者优先（现有行为）。"""
        item_good = _make_item(
            gene_id="good",
            source_score=0.9,
            lower_is_better=False,
            node_pheromone=0.5,
            gene_pheromone=0.5,
            source_node_id="node_good",
        )
        item_bad = _make_item(
            gene_id="bad",
            source_score=0.3,
            lower_is_better=False,
            node_pheromone=0.5,
            gene_pheromone=0.5,
            source_node_id="node_bad",
        )

        winner = _select_locus_winner([item_bad, item_good])
        assert winner.gene_id == "good", (
            f"AUC 场景应选 source_score 更大的 good，实际选了 {winner.gene_id}"
        )
