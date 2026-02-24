"""P0/P1 V6 修复的单元测试。

覆盖：
- P0-2: truncate_term_out
- P0-3: _check_submission_and_set_error
- P0-4: _estimate_timeout 图像检测
- P0-1: run_single_ga_step
- P1-2: _build_draft_history TimeoutError 警告注入
- P1-3A: _get_cached_data_preview 缓存读取
- P1-3B: preview_special_file pd.read_csv 提示
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


# ============================================================
# P1-2: _build_draft_history TimeoutError 警告注入
# ============================================================


class TestBuildDraftHistoryTimeout:
    """测试 _build_draft_history() 对 TimeoutError 节点的警告注入。"""

    def _make_node(self, dead=False, exc_type=None, approach_tag=None, exec_time=0.0):
        """创建 mock 节点。"""
        node = MagicMock()
        node.dead = dead
        node.exc_type = exc_type
        node.approach_tag = approach_tag
        node.exec_time = exec_time
        return node

    def _make_orchestrator(self, nodes):
        """创建带 mock journal 的 Orchestrator，只暴露 _build_draft_history。"""
        import threading

        orch = MagicMock()
        orch.journal = MagicMock()
        orch.journal.nodes = nodes
        orch.journal_lock = threading.Lock()

        from core.orchestrator import Orchestrator

        # 绑定真实方法
        orch._build_draft_history = Orchestrator._build_draft_history.__get__(orch)
        return orch

    def test_no_timeout_nodes(self):
        """没有超时节点时，只返回正常 approach_tag。"""
        nodes = [
            self._make_node(approach_tag="LightGBM + 5-fold CV"),
            self._make_node(approach_tag="XGBoost + tuning"),
        ]
        orch = self._make_orchestrator(nodes)
        history = orch._build_draft_history()

        assert len(history) == 2
        assert "LightGBM + 5-fold CV" in history
        assert "XGBoost + tuning" in history

    def test_timeout_node_injected(self):
        """超时死节点的警告被注入到 history 中。"""
        nodes = [
            self._make_node(approach_tag="LightGBM + 5-fold CV"),
            self._make_node(
                dead=True,
                exc_type="TimeoutError",
                approach_tag="ResNet50 + ImageNet",
                exec_time=3600.0,
            ),
        ]
        orch = self._make_orchestrator(nodes)
        history = orch._build_draft_history()

        assert len(history) == 2
        assert history[0] == "LightGBM + 5-fold CV"
        assert "[TIMED OUT after 3600s]" in history[1]
        assert "ResNet50 + ImageNet" in history[1]
        assert "reduce model complexity" in history[1]

    def test_timeout_expired_also_captured(self):
        """TimeoutExpired 类型也被捕获。"""
        nodes = [
            self._make_node(
                dead=True,
                exc_type="TimeoutExpired",
                approach_tag="Heavy CNN",
                exec_time=1800.0,
            ),
        ]
        orch = self._make_orchestrator(nodes)
        history = orch._build_draft_history()

        assert len(history) == 1
        assert "[TIMED OUT after 1800s]" in history[0]

    def test_dead_non_timeout_not_injected(self):
        """死节点但非超时错误不会被注入。"""
        nodes = [
            self._make_node(dead=True, exc_type="ValueError", approach_tag="Bad model"),
        ]
        orch = self._make_orchestrator(nodes)
        history = orch._build_draft_history()

        assert len(history) == 0

    def test_no_duplicate_timeout_warnings(self):
        """相同超时警告不会重复。"""
        nodes = [
            self._make_node(
                dead=True, exc_type="TimeoutError",
                approach_tag="ResNet50", exec_time=3600.0,
            ),
            self._make_node(
                dead=True, exc_type="TimeoutError",
                approach_tag="ResNet50", exec_time=3600.0,
            ),
        ]
        orch = self._make_orchestrator(nodes)
        history = orch._build_draft_history()

        assert len(history) == 1


# ============================================================
# P1-3A: _get_cached_data_preview 缓存读取
# ============================================================


class TestGetCachedDataPreview:
    """测试 _get_cached_data_preview() 缓存读取。"""

    def test_file_exists_returns_content(self, tmp_path):
        """缓存文件存在时返回内容。"""
        from core.orchestrator import Orchestrator

        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        preview_file = logs_dir / "data_preview.md"
        preview_file.write_text("# Data Preview\n- train.csv: 100 rows", encoding="utf-8")

        orch = MagicMock()
        orch.config.project.workspace_dir = tmp_path
        orch._get_cached_data_preview = Orchestrator._get_cached_data_preview.__get__(orch)

        result = orch._get_cached_data_preview()
        assert "Data Preview" in result
        assert "train.csv" in result

    def test_cache_miss_regenerates_from_input(self, tmp_path):
        """缓存不存在但 input 目录存在时，重新生成并缓存。"""
        from core.orchestrator import Orchestrator

        # 创建 input 目录和一个 CSV 文件
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        csv_file = input_dir / "train.csv"
        csv_file.write_text("id,name,value\n1,foo,100\n2,bar,200\n", encoding="utf-8")

        orch = MagicMock()
        orch.config.project.workspace_dir = tmp_path
        orch._get_cached_data_preview = Orchestrator._get_cached_data_preview.__get__(orch)

        result = orch._get_cached_data_preview()
        assert result != ""
        assert "train.csv" in result
        # 验证缓存文件已写入
        assert (tmp_path / "logs" / "data_preview.md").exists()

    def test_cache_miss_no_input_dir_returns_empty(self, tmp_path):
        """缓存不存在且 input 目录也不存在时返回空字符串。"""
        from core.orchestrator import Orchestrator

        orch = MagicMock()
        orch.config.project.workspace_dir = tmp_path
        orch._get_cached_data_preview = Orchestrator._get_cached_data_preview.__get__(orch)

        result = orch._get_cached_data_preview()
        assert result == ""

    def test_logs_dir_not_exists_returns_empty(self, tmp_path):
        """logs 目录不存在且无 input 时返回空字符串。"""
        from core.orchestrator import Orchestrator

        orch = MagicMock()
        orch.config.project.workspace_dir = tmp_path / "nonexistent"
        orch._get_cached_data_preview = Orchestrator._get_cached_data_preview.__get__(orch)

        result = orch._get_cached_data_preview()
        assert result == ""


# ============================================================
# P1-3B: preview_special_file pd.read_csv 提示
# ============================================================


class TestPreviewSpecialFileSep:
    """测试 preview_special_file() 输出包含 pd.read_csv(sep=...) 提示。"""

    def test_comma_separated(self, tmp_path):
        """逗号分隔文件包含 pd.read_csv 提示。"""
        from utils.data_preview import preview_special_file

        f = tmp_path / "train.csv"
        f.write_text("id,name,value\n1,foo,100\n2,bar,200\n", encoding="utf-8")

        result = preview_special_file(f)
        assert "comma-separated" in result
        assert "pd.read_csv('train.csv', sep=',')" in result

    def test_tab_separated(self, tmp_path):
        """Tab 分隔文件包含 pd.read_csv(sep='\\t') 提示。"""
        from utils.data_preview import preview_special_file

        f = tmp_path / "data.tsv"
        f.write_text("id\tname\tvalue\n1\tfoo\t100\n", encoding="utf-8")

        result = preview_special_file(f)
        assert "tab-separated" in result
        assert r"pd.read_csv('data.tsv', sep='\t')" in result

    def test_space_separated(self, tmp_path):
        """空格分隔文件包含 pd.read_csv(sep=' ') 提示。"""
        from utils.data_preview import preview_special_file

        f = tmp_path / "data.txt"
        f.write_text("id name value\n1 foo 100\n", encoding="utf-8")

        result = preview_special_file(f)
        assert "space-separated" in result
        assert "pd.read_csv('data.txt', sep=' ')" in result

    def test_unstructured_no_sep_hint(self, tmp_path):
        """非结构化文件不包含 pd.read_csv 提示。"""
        from utils.data_preview import preview_special_file

        f = tmp_path / "readme.txt"
        f.write_text("Thisisasolidblockoftextwithnoseparators\n", encoding="utf-8")

        result = preview_special_file(f)
        assert "pd.read_csv" not in result
