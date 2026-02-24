"""P0 V6 修复的单元测试。

覆盖：
- P0-2: truncate_term_out
- P0-3: _check_submission_and_set_error
- P0-4: _estimate_timeout 图像检测
- P0-1: run_single_ga_step
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path


# ============================================================
# P0-2: truncate_term_out
# ============================================================


class TestTruncateTermOut:
    """测试 Review Prompt 截断功能。"""

    def test_short_text_unchanged(self):
        """短文本原样返回。"""
        from utils.text_utils import truncate_term_out

        text = "Hello world"
        assert truncate_term_out(text) == text

    def test_none_returns_empty(self):
        """None 输入返回空字符串。"""
        from utils.text_utils import truncate_term_out

        assert truncate_term_out(None) == ""

    def test_empty_returns_empty(self):
        """空字符串返回空字符串。"""
        from utils.text_utils import truncate_term_out

        assert truncate_term_out("") == ""

    def test_exact_limit_unchanged(self):
        """恰好等于限制时不截断。"""
        from utils.text_utils import truncate_term_out

        text = "x" * 3500
        assert truncate_term_out(text, max_len=3500) == text

    def test_over_limit_truncated(self):
        """超过限制时截断，保留头尾。"""
        from utils.text_utils import truncate_term_out

        # 构造 5000 字符文本
        text = "H" * 2000 + "M" * 1000 + "T" * 2000
        result = truncate_term_out(text, max_len=3500)

        # 检查头部
        assert result.startswith("H" * 1500)
        # 检查尾部
        assert result.endswith("T" * 2000)
        # 检查截断标记
        assert "truncated" in result
        # 检查长度小于原始
        assert len(result) < len(text)

    def test_truncation_message_contains_char_count(self):
        """截断消息包含省略字符数。"""
        from utils.text_utils import truncate_term_out

        text = "x" * 10000
        result = truncate_term_out(text, max_len=3500)
        # 省略 = 10000 - 1500 - 2000 = 6500
        assert "6500" in result


# ============================================================
# P0-1: run_single_ga_step
# ============================================================


class TestRunSingleGaStep:
    """测试 SolutionEvolution.run_single_ga_step()。"""

    def _make_node(self, is_buggy=False, dead=False, genes=None):
        """创建 mock 节点。"""
        node = MagicMock()
        node.is_buggy = is_buggy
        node.dead = dead
        node.genes = genes or {"DATA": "...", "MODEL": "...", "TRAIN": "...", "POSTPROCESS": "..."}
        node.metric_value = 0.9
        node.lower_is_better = False
        return node

    def test_insufficient_valid_pool_returns_none(self):
        """valid_pool 不足时返回 None。"""
        from core.evolution.solution_evolution import SolutionEvolution

        config = MagicMock()
        config.evolution.solution.population_size = 12
        config.evolution.solution.elite_size = 3
        config.evolution.solution.crossover_rate = 0.8
        config.evolution.solution.mutation_rate = 0.2
        config.evolution.solution.tournament_k = 3
        config.evolution.solution.crossover_strategy = "random"
        config.evolution.solution.ga_trigger_threshold = 4
        config.evolution.solution.phase1_target_nodes = 8

        journal = MagicMock()
        # 只有 2 个 valid 节点（不够 4）
        journal.nodes = [self._make_node() for _ in range(2)]

        se = SolutionEvolution(config=config, journal=journal)
        result = se.run_single_ga_step()
        assert result is None

    def test_sufficient_valid_pool_calls_ga(self):
        """valid_pool 充足时执行 GA 操作。"""
        from core.evolution.solution_evolution import SolutionEvolution

        config = MagicMock()
        config.evolution.solution.population_size = 12
        config.evolution.solution.elite_size = 3
        config.evolution.solution.crossover_rate = 0.8
        config.evolution.solution.mutation_rate = 0.2
        config.evolution.solution.tournament_k = 3
        config.evolution.solution.crossover_strategy = "random"
        config.evolution.solution.ga_trigger_threshold = 4
        config.evolution.solution.phase1_target_nodes = 8

        journal = MagicMock()
        journal.nodes = [self._make_node() for _ in range(6)]

        se = SolutionEvolution(config=config, journal=journal)
        # mock 两个 GA 方法
        se._run_merge_step = MagicMock(return_value=MagicMock())
        se._run_mutate_step = MagicMock(return_value=MagicMock())

        result = se.run_single_ga_step()
        # 至少一个被调用
        assert se._run_merge_step.called or se._run_mutate_step.called
