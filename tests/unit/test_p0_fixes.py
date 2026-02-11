"""P0 修复项单元测试。

覆盖:
- P0-A: SolutionEvolution lower_is_better 排序修复
- P0-B: Metric 合理性范围校验 (METRIC_BOUNDS)
- P0-C: stdout Metric 正则提取
- P0-F: Submission 格式验证
"""

import json
from unittest.mock import Mock, patch

import pytest
from omegaconf import OmegaConf

from core.state import Node, Journal
from core.evolution.solution_evolution import SolutionEvolution
from core.orchestrator import Orchestrator, METRIC_BOUNDS


# ============================================================
# Fixtures
# ============================================================


def _make_node(
    metric: float, lower_is_better: bool = False, buggy: bool = False
) -> Node:
    """创建测试用 Node。"""
    node = Node(code="pass")
    node.metric_value = metric if not buggy else None
    node.lower_is_better = lower_is_better
    node.is_buggy = buggy
    return node


@pytest.fixture
def mock_config(tmp_path):
    """Mock 配置对象。"""
    return OmegaConf.create(
        {
            "project": {"workspace_dir": str(tmp_path)},
            "agent": {"max_steps": 10, "time_limit": 3600},
            "search": {"num_drafts": 3, "debug_prob": 0.5, "parallel_num": 1},
            "execution": {"timeout": 60},
            "llm": {
                "feedback": {
                    "model": "test",
                    "provider": "openai",
                    "temperature": 0.5,
                    "api_key": "test",
                    "base_url": "https://test.com/v1",
                }
            },
            "evolution": {
                "solution": {
                    "population_size": 5,
                    "elite_size": 2,
                    "crossover_rate": 0.8,
                    "mutation_rate": 0.2,
                    "tournament_k": 3,
                    "steps_per_epoch": 10,
                    "crossover_strategy": "random",
                }
            },
        }
    )


@pytest.fixture
def journal():
    """创建 Journal 实例。"""
    return Journal()


@pytest.fixture
def orchestrator(mock_config, journal):
    """创建 Orchestrator 实例（仅用于调用内部方法）。"""
    agent = Mock()
    agent.name = "test_agent"
    return Orchestrator(
        agents=[agent],
        config=mock_config,
        journal=journal,
        task_desc="Evaluate using log_loss metric. AUC is also reported.",
    )


# ============================================================
# P0-A: SolutionEvolution lower_is_better 修复
# ============================================================


class TestP0A_SelectElites:
    """精英保留 lower_is_better 修复测试。"""

    def test_higher_is_better_selects_highest(self, mock_config, journal):
        """higher_is_better 场景：选择 metric 最大的个体。"""
        sol_evo = SolutionEvolution(config=mock_config, journal=journal)
        sol_evo.population = [
            _make_node(0.80, lower_is_better=False),
            _make_node(0.95, lower_is_better=False),
            _make_node(0.70, lower_is_better=False),
            _make_node(0.88, lower_is_better=False),
            _make_node(0.92, lower_is_better=False),
        ]
        elites = sol_evo._select_elites()
        assert len(elites) == 2
        assert elites[0].metric_value == 0.95
        assert elites[1].metric_value == 0.92

    def test_lower_is_better_selects_lowest(self, mock_config, journal):
        """lower_is_better 场景：选择 metric 最小的个体。"""
        sol_evo = SolutionEvolution(config=mock_config, journal=journal)
        sol_evo.population = [
            _make_node(0.08, lower_is_better=True),
            _make_node(0.05, lower_is_better=True),
            _make_node(0.12, lower_is_better=True),
            _make_node(0.06, lower_is_better=True),
            _make_node(0.10, lower_is_better=True),
        ]
        elites = sol_evo._select_elites()
        assert len(elites) == 2
        assert elites[0].metric_value == 0.05
        assert elites[1].metric_value == 0.06

    def test_none_metric_pushed_to_end(self, mock_config, journal):
        """metric=None 的个体排在最后，不被选为精英。"""
        sol_evo = SolutionEvolution(config=mock_config, journal=journal)
        sol_evo.population = [
            _make_node(0.80, lower_is_better=False),
            _make_node(0.0, lower_is_better=False, buggy=True),  # metric=None
            _make_node(0.90, lower_is_better=False),
            _make_node(0.0, lower_is_better=False, buggy=True),  # metric=None
            _make_node(0.85, lower_is_better=False),
        ]
        elites = sol_evo._select_elites()
        assert all(e.metric_value is not None for e in elites)
        assert elites[0].metric_value == 0.90


class TestP0A_TournamentSelect:
    """锦标赛选择 lower_is_better 修复测试。"""

    def test_higher_is_better_selects_max(self, mock_config, journal):
        """higher_is_better 场景：从 tournament 的 k 个候选中选最大值。"""
        sol_evo = SolutionEvolution(config=mock_config, journal=journal)
        nodes = [
            _make_node(0.80, lower_is_better=False),
            _make_node(0.95, lower_is_better=False),
            _make_node(0.70, lower_is_better=False),
            _make_node(0.88, lower_is_better=False),
            _make_node(0.92, lower_is_better=False),
        ]
        sol_evo.population = nodes

        # 固定 tournament 候选为 [0.80, 0.70, 0.88]，期望选最大值 0.88
        with patch("random.sample", return_value=[nodes[0], nodes[2], nodes[3]]):
            winner = sol_evo._tournament_select()
        assert winner.metric_value == 0.88

    def test_lower_is_better_selects_min(self, mock_config, journal):
        """lower_is_better 场景：从 tournament 的 k 个候选中选最小值。"""
        sol_evo = SolutionEvolution(config=mock_config, journal=journal)
        nodes = [
            _make_node(0.08, lower_is_better=True),
            _make_node(0.05, lower_is_better=True),
            _make_node(0.12, lower_is_better=True),
            _make_node(0.06, lower_is_better=True),
            _make_node(0.10, lower_is_better=True),
        ]
        sol_evo.population = nodes

        # 固定 tournament 候选为 [0.08, 0.12, 0.10]，期望选最小值 0.08
        with patch("random.sample", return_value=[nodes[0], nodes[2], nodes[4]]):
            winner = sol_evo._tournament_select()
        assert winner.metric_value == 0.08

    def test_lower_is_better_tournament_none_pushed(self, mock_config, journal):
        """lower_is_better tournament 中 None metric 节点不被选中。"""
        sol_evo = SolutionEvolution(config=mock_config, journal=journal)
        good_node = _make_node(0.10, lower_is_better=True)
        none_node1 = _make_node(0.0, lower_is_better=True, buggy=True)  # metric=None
        none_node2 = _make_node(0.0, lower_is_better=True, buggy=True)
        sol_evo.population = [good_node, none_node1, none_node2]

        # tournament_k=3, 选中全部 3 个，None 应该排最后
        with patch("random.sample", return_value=[good_node, none_node1, none_node2]):
            winner = sol_evo._tournament_select()
        assert winner.metric_value == 0.10


class TestP0A_IsLowerBetter:
    """_is_lower_better() 辅助方法测试。"""

    def test_returns_true_for_lower_is_better(self, mock_config, journal):
        """种群中节点 lower_is_better=True 时返回 True。"""
        sol_evo = SolutionEvolution(config=mock_config, journal=journal)
        sol_evo.population = [
            _make_node(0.05, lower_is_better=True),
            _make_node(0.08, lower_is_better=True),
        ]
        assert sol_evo._is_lower_better() is True

    def test_returns_false_for_higher_is_better(self, mock_config, journal):
        """种群中节点 lower_is_better=False 时返回 False。"""
        sol_evo = SolutionEvolution(config=mock_config, journal=journal)
        sol_evo.population = [
            _make_node(0.90, lower_is_better=False),
        ]
        assert sol_evo._is_lower_better() is False

    def test_defaults_false_when_all_none(self, mock_config, journal):
        """所有节点 metric=None 时默认 False。"""
        sol_evo = SolutionEvolution(config=mock_config, journal=journal)
        sol_evo.population = [
            _make_node(0.0, buggy=True),  # metric=None
            _make_node(0.0, buggy=True),
        ]
        assert sol_evo._is_lower_better() is False


# ============================================================
# P0-B: Metric 合理性范围校验
# ============================================================


class TestP0B_MetricBounds:
    """METRIC_BOUNDS 常量测试。"""

    def test_bounds_structure(self):
        """所有 bounds entry 格式为 (min, max)。"""
        for key, (lo, hi) in METRIC_BOUNDS.items():
            if lo is not None and hi is not None:
                assert lo < hi, f"{key}: min={lo} >= max={hi}"


class TestP0B_CheckMetricPlausibility:
    """_check_metric_plausibility() 范围校验测试。"""

    def test_logloss_zero_rejected(self, orchestrator):
        """logloss=0.0 被拒绝（理论不可能）。"""
        orchestrator.best_node = _make_node(0.265)
        # task_desc 包含 "log_loss"
        assert orchestrator._check_metric_plausibility(0.0) is False

    def test_logloss_negative_rejected(self, orchestrator):
        """logloss 负值被拒绝。"""
        orchestrator.best_node = _make_node(0.265)
        assert orchestrator._check_metric_plausibility(-0.1) is False

    def test_auc_overflow_rejected(self, orchestrator):
        """AUC > 1.0 被拒绝。"""
        # task_desc 包含 "AUC"
        orchestrator._task_desc_compressed = "Evaluate using AUC metric"
        orchestrator.best_node = _make_node(0.85)
        assert orchestrator._check_metric_plausibility(1.5) is False

    def test_normal_metric_passes(self, orchestrator):
        """正常范围内的 metric 通过检查。"""
        orchestrator.best_node = _make_node(0.265)
        assert orchestrator._check_metric_plausibility(0.280) is True

    def test_no_best_node_passes(self, orchestrator):
        """无 best_node 时默认通过。"""
        orchestrator.best_node = None
        assert orchestrator._check_metric_plausibility(0.5) is True

    def test_metric_zero_with_high_best_rejected(self, orchestrator):
        """metric=0.0 且 best > 0.01 时被拒绝。"""
        orchestrator._task_desc_compressed = "Some generic task"
        orchestrator.best_node = _make_node(0.5)
        assert orchestrator._check_metric_plausibility(0.0) is False

    def test_ratio_check_rejects_extreme(self, orchestrator):
        """相对比率超过 50 倍被拒绝（Phase 3）。"""
        orchestrator._task_desc_compressed = "Some generic task"
        orchestrator.best_node = _make_node(0.85)
        # ratio = 500 / 0.85 ≈ 588 > 50
        assert orchestrator._check_metric_plausibility(500.0) is False

    def test_ratio_check_passes_within_bounds(self, orchestrator):
        """相对比率在 50 倍内通过（Phase 3）。"""
        orchestrator._task_desc_compressed = "Some generic task"
        orchestrator.best_node = _make_node(0.85)
        # ratio = 1.2 / 0.85 ≈ 1.41 <= 50
        assert orchestrator._check_metric_plausibility(1.2) is True

    def test_negative_rmse_rejected(self, orchestrator):
        """RMSE 负值被 Phase 1 绝对范围拒绝。"""
        orchestrator._task_desc_compressed = "Evaluate using rmse metric"
        orchestrator.best_node = _make_node(0.5)
        assert orchestrator._check_metric_plausibility(-0.1) is False


# ============================================================
# P0-C: stdout Metric 正则提取
# ============================================================


class TestP0C_ParseMetricFromStdout:
    """_parse_metric_from_stdout() 测试。"""

    def test_single_match(self, orchestrator):
        """单行匹配。"""
        stdout = "Training done.\nValidation metric: 0.95432\nSaved submission."
        assert orchestrator._parse_metric_from_stdout(stdout) == pytest.approx(0.95432)

    def test_multi_fold_takes_last(self, orchestrator):
        """多折输出取最后一个（平均值）。"""
        stdout = (
            "Fold 1 - Validation metric: 0.91\n"
            "Fold 2 - Validation metric: 0.93\n"
            "Fold 3 - Validation metric: 0.92\n"
            "Mean - Validation metric: 0.9200\n"
        )
        assert orchestrator._parse_metric_from_stdout(stdout) == pytest.approx(0.92)

    def test_scientific_notation(self, orchestrator):
        """科学计数法。"""
        stdout = "Validation metric: 1.23e-04"
        assert orchestrator._parse_metric_from_stdout(stdout) == pytest.approx(1.23e-4)

    def test_no_match_returns_none(self, orchestrator):
        """无匹配返回 None。"""
        stdout = "Training complete. Loss: 0.5"
        assert orchestrator._parse_metric_from_stdout(stdout) is None

    def test_empty_input(self, orchestrator):
        """空输入返回 None。"""
        assert orchestrator._parse_metric_from_stdout("") is None
        assert orchestrator._parse_metric_from_stdout(None) is None

    def test_negative_value(self, orchestrator):
        """负数值。"""
        stdout = "Validation metric: -0.532"
        assert orchestrator._parse_metric_from_stdout(stdout) == pytest.approx(-0.532)

    def test_integer_value(self, orchestrator):
        """整数值。"""
        stdout = "Validation metric: 100"
        assert orchestrator._parse_metric_from_stdout(stdout) == pytest.approx(100.0)

    def test_no_leading_zero(self, orchestrator):
        """无前导零的小数。"""
        stdout = "Validation metric: .95"
        assert orchestrator._parse_metric_from_stdout(stdout) == pytest.approx(0.95)


# ============================================================
# P0-C (补充): Review 集成测试 — Phase 5.5 三路决策
# ============================================================


class TestP0C_ReviewIntegration:
    """_review_node() Phase 5.5 集成测试：LLM 与 stdout metric 合并逻辑。"""

    def test_stdout_fallback_when_llm_no_metric(self, orchestrator):
        """LLM 未提取 metric 时使用 stdout 值补位。"""
        node = Node(code="print('test')", plan="plan")
        node.term_out = "Training done.\nValidation metric: 0.8500\nSaved."
        node.exec_time = 1.0
        node.exc_type = None

        review_data = {
            "is_bug": False,
            "metric": None,
            "lower_is_better": False,
            "has_csv_submission": True,
            "key_change": "test",
            "insight": "test",
        }

        with (
            patch.object(orchestrator, "_check_submission_exists", return_value=True),
            patch.object(
                orchestrator,
                "_validate_submission_format",
                return_value={"valid": True, "errors": [], "row_count": 100},
            ),
            patch.object(
                orchestrator,
                "_call_review_with_tool_debug",
                return_value=(json.dumps(review_data), review_data),
            ),
        ):
            orchestrator._review_node(node)

        # stdout 值应被用作补位
        assert node.metric_value == pytest.approx(0.85)
        assert node.is_buggy is False

    def test_llm_priority_on_mismatch(self, orchestrator):
        """LLM 与 stdout 不一致时保留 LLM 值，并记录 mismatch。"""
        node = Node(code="print('test')", plan="plan")
        node.term_out = "Validation metric: 0.7000"
        node.exec_time = 1.0
        node.exc_type = None

        review_data = {
            "is_bug": False,
            "metric": 0.90,
            "lower_is_better": False,
            "has_csv_submission": True,
            "key_change": "test",
            "insight": "test",
        }

        with (
            patch.object(orchestrator, "_check_submission_exists", return_value=True),
            patch.object(
                orchestrator,
                "_validate_submission_format",
                return_value={"valid": True, "errors": [], "row_count": 100},
            ),
            patch.object(
                orchestrator,
                "_call_review_with_tool_debug",
                return_value=(json.dumps(review_data), review_data),
            ),
            patch("core.orchestrator.log_json") as mock_log_json,
        ):
            orchestrator._review_node(node)

        # LLM 值优先
        assert node.metric_value == pytest.approx(0.90)
        assert node.is_buggy is False
        # 应记录 mismatch 事件
        mock_log_json.assert_called_once()
        call_args = mock_log_json.call_args[0][0]
        assert call_args["event"] == "metric_mismatch"
        assert call_args["llm_metric"] == 0.90
        assert call_args["stdout_metric"] == pytest.approx(0.70)

    def test_both_agree_no_mismatch_log(self, orchestrator):
        """LLM 与 stdout 一致（<1%差异）时不记录 mismatch。"""
        node = Node(code="print('test')", plan="plan")
        # 0.9005 vs 0.90 → diff=0.0005, threshold=0.009 → 不触发
        node.term_out = "Validation metric: 0.9005"
        node.exec_time = 1.0
        node.exc_type = None

        review_data = {
            "is_bug": False,
            "metric": 0.90,
            "lower_is_better": False,
            "has_csv_submission": True,
            "key_change": "test",
            "insight": "test",
        }

        with (
            patch.object(orchestrator, "_check_submission_exists", return_value=True),
            patch.object(
                orchestrator,
                "_validate_submission_format",
                return_value={"valid": True, "errors": [], "row_count": 100},
            ),
            patch.object(
                orchestrator,
                "_call_review_with_tool_debug",
                return_value=(json.dumps(review_data), review_data),
            ),
            patch("core.orchestrator.log_json") as mock_log_json,
        ):
            orchestrator._review_node(node)

        assert node.metric_value == pytest.approx(0.90)
        # 不应记录 mismatch
        mock_log_json.assert_not_called()

    def test_both_none_marks_buggy(self, orchestrator):
        """LLM 和 stdout 都无 metric 时节点标记为 buggy。"""
        node = Node(code="print('test')", plan="plan")
        node.term_out = "Training complete. No metric output."
        node.exec_time = 1.0
        node.exc_type = None

        review_data = {
            "is_bug": False,
            "metric": None,
            "lower_is_better": False,
            "has_csv_submission": True,
            "key_change": "test",
            "insight": "test",
        }

        with (
            patch.object(orchestrator, "_check_submission_exists", return_value=True),
            patch.object(
                orchestrator,
                "_validate_submission_format",
                return_value={"valid": True, "errors": [], "row_count": 100},
            ),
            patch.object(
                orchestrator,
                "_call_review_with_tool_debug",
                return_value=(json.dumps(review_data), review_data),
            ),
        ):
            orchestrator._review_node(node)

        # metric_value=None → is_buggy 条件 3 触发
        assert node.metric_value is None
        assert node.is_buggy is True


# ============================================================
# P0-E: Review Schema 增强测试
# ============================================================


class TestP0E_ReviewSchema:
    """Review tool schema 和 prompt 增强测试。"""

    def test_metric_name_in_schema_and_required(self, orchestrator):
        """metric_name 字段存在且在 required 列表中。"""
        schema = orchestrator._get_review_tool_schema()
        assert "metric_name" in schema["parameters"]["properties"]
        assert "metric_name" in schema["parameters"]["required"]

    def test_alignment_check_in_prompt(self, orchestrator):
        """Review prompt 包含 Metric Alignment Check 段落。"""
        node = Node(
            code="print('test')",
            plan="plan",
            term_out="output",
            exec_time=1.0,
        )
        messages = orchestrator._build_review_messages(
            node, "(Initial solution, no diff)"
        )
        assert "Metric Alignment Check" in messages
        assert "Focal Loss" in messages  # 提示检查 loss/metric 对齐


# ============================================================
# P0-F: Submission 格式验证
# ============================================================


class TestP0F_ValidateSubmissionFormat:
    """_validate_submission_format() 测试。"""

    def test_missing_file(self, orchestrator, tmp_path):
        """文件不存在。"""
        orchestrator.config.project.workspace_dir = tmp_path
        result = orchestrator._validate_submission_format("nonexistent")
        assert result["valid"] is False
        assert "不存在" in result["errors"][0]

    def test_valid_submission(self, orchestrator, tmp_path):
        """正常 submission 通过。"""
        orchestrator.config.project.workspace_dir = tmp_path
        sub_dir = tmp_path / "submission"
        sub_dir.mkdir()
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        import pandas as pd

        # 创建 sample 和 submission
        sample = pd.DataFrame({"id": [1, 2, 3], "target": [0, 1, 0]})
        sample.to_csv(input_dir / "sample_submission.csv", index=False)
        sub = pd.DataFrame({"id": [1, 2, 3], "target": [0.1, 0.9, 0.2]})
        sub.to_csv(sub_dir / "submission_test123.csv", index=False)

        result = orchestrator._validate_submission_format("test123")
        assert result["valid"] is True
        assert result["row_count"] == 3

    def test_nan_values_rejected(self, orchestrator, tmp_path):
        """包含 NaN 的 submission 被拒绝。"""
        orchestrator.config.project.workspace_dir = tmp_path
        sub_dir = tmp_path / "submission"
        sub_dir.mkdir()

        import pandas as pd
        import numpy as np

        sub = pd.DataFrame({"id": [1, 2, 3], "target": [0.1, np.nan, 0.2]})
        sub.to_csv(sub_dir / "submission_nan_test.csv", index=False)

        result = orchestrator._validate_submission_format("nan_test")
        assert result["valid"] is False
        assert any("NaN" in e for e in result["errors"])

    def test_row_count_mismatch_rejected(self, orchestrator, tmp_path):
        """行数不匹配被拒绝。"""
        orchestrator.config.project.workspace_dir = tmp_path
        sub_dir = tmp_path / "submission"
        sub_dir.mkdir()
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        import pandas as pd

        sample = pd.DataFrame({"id": range(100), "target": range(100)})
        sample.to_csv(input_dir / "sample_submission.csv", index=False)
        sub = pd.DataFrame({"id": range(70), "target": range(70)})
        sub.to_csv(sub_dir / "submission_rowtest.csv", index=False)

        result = orchestrator._validate_submission_format("rowtest")
        assert result["valid"] is False
        assert any("行数不匹配" in e for e in result["errors"])

    def test_column_mismatch_is_warning_not_failure(self, orchestrator, tmp_path):
        """列名不匹配仅记录 error 但不设 valid=False。"""
        orchestrator.config.project.workspace_dir = tmp_path
        sub_dir = tmp_path / "submission"
        sub_dir.mkdir()
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        import pandas as pd

        sample = pd.DataFrame({"id": [1, 2, 3], "target": [0, 1, 0]})
        sample.to_csv(input_dir / "sample_submission.csv", index=False)
        # 列名不同但行数一致
        sub = pd.DataFrame({"id": [1, 2, 3], "prediction": [0.1, 0.9, 0.2]})
        sub.to_csv(sub_dir / "submission_coltest.csv", index=False)

        result = orchestrator._validate_submission_format("coltest")
        assert result["valid"] is True  # 列名不匹配不导致失败
        assert any("列名不匹配" in e for e in result["errors"])

    def test_sample_submission_camelcase_fallback(self, orchestrator, tmp_path):
        """支持 sampleSubmission.csv 驼峰命名变体。"""
        orchestrator.config.project.workspace_dir = tmp_path
        sub_dir = tmp_path / "submission"
        sub_dir.mkdir()
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        import pandas as pd

        # 使用驼峰命名
        sample = pd.DataFrame({"id": [1, 2], "target": [0, 1]})
        sample.to_csv(input_dir / "sampleSubmission.csv", index=False)
        sub = pd.DataFrame({"id": [1, 2], "target": [0.5, 0.8]})
        sub.to_csv(sub_dir / "submission_camel.csv", index=False)

        result = orchestrator._validate_submission_format("camel")
        assert result["valid"] is True
        assert result["row_count"] == 2

    def test_invalid_submission_marks_buggy_in_review(self, orchestrator, tmp_path):
        """submission 格式无效时 _review_node 标记节点为 buggy。"""
        orchestrator.config.project.workspace_dir = tmp_path
        sub_dir = tmp_path / "submission"
        sub_dir.mkdir()

        import pandas as pd
        import numpy as np

        node = Node(code="print('test')", plan="plan")
        node.term_out = "Validation metric: 0.90"
        node.exec_time = 1.0
        node.exc_type = None

        # 创建包含 NaN 的 submission（格式无效）
        sub = pd.DataFrame({"id": [1, 2], "target": [0.5, np.nan]})
        sub.to_csv(sub_dir / f"submission_{node.id}.csv", index=False)

        review_data = {
            "is_bug": False,
            "metric": 0.90,
            "lower_is_better": False,
            "has_csv_submission": True,
            "key_change": "test",
            "insight": "test",
        }

        with patch.object(
            orchestrator,
            "_call_review_with_tool_debug",
            return_value=(json.dumps(review_data), review_data),
        ):
            orchestrator._review_node(node)

        # 格式无效 → has_submission=False → is_buggy 条件 4 触发
        assert node.is_buggy is True
        assert node.metric_value is None
