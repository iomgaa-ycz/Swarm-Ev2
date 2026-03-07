"""Orchestrator 单元测试模块。"""

import json
import threading
import time
import pytest
from unittest.mock import Mock, MagicMock, patch
from omegaconf import OmegaConf

from core.orchestrator import Orchestrator, METRIC_BOUNDS
from core.state import Node, Journal


@pytest.fixture
def mock_config(tmp_path):
    """Mock 配置对象。"""
    cfg = OmegaConf.create(
        {
            "project": {
                "workspace_dir": tmp_path,
            },
            "agent": {
                "max_steps": 10,
                "time_limit": 3600,
            },
            "search": {
                "num_drafts": 3,
                "debug_prob": 0.5,
                "parallel_num": 1,  # 并行执行数
            },
            "execution": {
                "timeout": 60,
                "adaptive_timeout": False,
            },
            "llm": {
                "feedback": {
                    "model": "glm-4.6",
                    "provider": "openai",
                    "temperature": 0.5,
                    "api_key": "test-key",
                    "base_url": "https://test.com/v1",
                }
            },
        }
    )
    return cfg


@pytest.fixture
def mock_agent():
    """Mock Agent 对象。"""
    agent = Mock()
    agent.name = "test_agent"
    return agent


@pytest.fixture
def journal():
    """创建 Journal 实例。"""
    return Journal()


@pytest.fixture
def orchestrator(mock_agent, mock_config, journal):
    """创建 Orchestrator 实例。"""
    return Orchestrator(
        agents=[mock_agent],  # 使用 agents 列表
        config=mock_config,
        journal=journal,
        task_desc="Test task description",
    )


class TestOrchestrator:
    """Orchestrator 测试类。"""

    def test_init(self, orchestrator, mock_agent, mock_config, journal):
        """测试初始化。"""
        assert orchestrator.agents == [mock_agent]
        assert orchestrator.config == mock_config
        assert orchestrator.journal == journal
        assert orchestrator.task_desc == "Test task description"
        assert orchestrator.current_epoch == 0
        assert orchestrator.best_node is None

    def test_review_node_success(self, orchestrator):
        """测试 Review 评估成功（纯文本 JSON 方案）。"""
        node = Node(code="print('test')", plan="Test plan")

        # Mock backend_query 返回纯文本 JSON（无 function calling）
        review_json = json.dumps({
            "is_bug": False,
            "metric": 0.90,
            "key_change": "Added print statement",
            "insight": "Good result",
            "lower_is_better": False,
            "has_csv_submission": True,
            "metric_name": "auc",
            "bottleneck": "Limited features",
            "suggested_direction": "Add more features",
            "approach_tag": "Simple print test",
        })

        with patch("core.orchestrator.backend_query") as mock_query:
            with patch.object(
                orchestrator, "_check_submission_exists", return_value=True
            ):
                with patch.object(
                    orchestrator,
                    "_validate_submission_format",
                    return_value={"valid": True, "errors": [], "row_count": 100},
                ):
                    mock_query.return_value = review_json
                    orchestrator._review_node(node)

        # 验证
        assert node.is_buggy is False
        assert node.metric_value == 0.90
        assert node.analysis == "Added print statement"
        assert node.analysis_detail is not None
        assert node.analysis_detail["key_change"] == "Added print statement"
        assert node.lower_is_better is False

    def test_review_node_parsing_failure(self, orchestrator):
        """测试 Review 解析失败回退。"""
        node = Node(code="raise ValueError()", plan="Buggy plan", exc_type="ValueError")

        # Mock backend.query 抛出异常
        with patch("core.orchestrator.backend_query") as mock_query:
            mock_query.side_effect = Exception("API error")

            orchestrator._review_node(node)

        # 验证：应该标记为 buggy（因为有异常类型）
        assert node.is_buggy is True
        assert node.metric_value is None

    def test_update_best_node_maximize(self, orchestrator):
        """测试 best_node 更新逻辑（越大越好）。"""
        # 第一个节点
        node1 = Node(
            code="code1",
            plan="plan1",
            metric_value=0.85,
            is_buggy=False,
            lower_is_better=False,
        )
        orchestrator._update_best_node(node1)
        assert orchestrator.best_node == node1

        # 第二个节点（更好）
        node2 = Node(
            code="code2",
            plan="plan2",
            metric_value=0.90,
            is_buggy=False,
            lower_is_better=False,
        )
        orchestrator._update_best_node(node2)
        assert orchestrator.best_node == node2

        # 第三个节点（更差，不应该更新）
        node3 = Node(
            code="code3",
            plan="plan3",
            metric_value=0.80,
            is_buggy=False,
            lower_is_better=False,
        )
        orchestrator._update_best_node(node3)
        assert orchestrator.best_node == node2  # 仍然是 node2

    def test_update_best_node_minimize(self, orchestrator):
        """测试 best_node 更新逻辑（越小越好）。"""
        # 第一个节点
        node1 = Node(
            code="code1",
            plan="plan1",
            metric_value=0.5,
            is_buggy=False,
            lower_is_better=True,
        )
        orchestrator._update_best_node(node1)
        assert orchestrator.best_node == node1

        # 第二个节点（更好，metric 更小）
        node2 = Node(
            code="code2",
            plan="plan2",
            metric_value=0.3,
            is_buggy=False,
            lower_is_better=True,
        )
        orchestrator._update_best_node(node2)
        assert orchestrator.best_node == node2

        # 第三个节点（更差，不应该更新）
        node3 = Node(
            code="code3",
            plan="plan3",
            metric_value=0.6,
            is_buggy=False,
            lower_is_better=True,
        )
        orchestrator._update_best_node(node3)
        assert orchestrator.best_node == node2  # 仍然是 node2

    def test_build_review_messages(self, orchestrator):
        """测试 Review messages 构建（包含变更上下文 + JSON 模板）。"""
        node = Node(
            code="print('test')",
            plan="Test plan",
            term_out="test\n",
            exec_time=0.5,
            exc_type=None,
        )

        # 传入变更上下文参数
        change_context = "(Initial solution, no diff)"
        messages = orchestrator._build_review_messages(node, change_context)

        assert "Test task description" in messages
        assert "print('test')" in messages
        assert "test\n" in messages
        assert "0.50s" in messages
        assert "Code Changes" in messages
        assert "Initial solution" in messages
        # 验证 JSON 模板（取代了 "Call submit_review"）
        assert "submit_review" not in messages
        assert '"is_bug"' in messages
        assert '"metric_name"' in messages
        assert '"approach_tag"' in messages
        assert "ALL 10 keys required" in messages


class TestDebugChain:
    """_debug_chain 链式 Debug 逻辑测试。"""

    def test_skips_debug_when_no_error(self, orchestrator, mock_agent):
        """exc_type 为 None 时直接跳过 debug，不调用 agent._debug。"""
        node = Node(code="print('ok')", plan="plan", exc_type=None)

        result = orchestrator._debug_chain(node, mock_agent, context=None, max_attempts=2)

        # 没有异常，立即返回原节点
        assert result is node
        mock_agent._debug.assert_not_called()

    def test_marks_dead_after_max_attempts(self, orchestrator, mock_agent):
        """agent._debug 始终返回仍有异常的节点，耗尽后标记 node.dead=True。"""
        from unittest.mock import MagicMock

        node = Node(code="raise ValueError()", plan="plan", exc_type="ValueError")

        # Mock context（_debug_chain 内部会访问 context 的属性）
        mock_context = MagicMock()
        mock_context.task_desc = "test task"
        mock_context.device_info = "CPU"
        mock_context.conda_packages = "pandas"
        mock_context.conda_env_name = "Swarm-Evo"

        # _debug 每次都返回一个仍然有错的新节点（不同 code，仍有 exc_type）
        def make_buggy_node(*args, **kwargs):
            n = Node(code="raise RuntimeError()", plan="plan", exc_type="RuntimeError")
            return n

        mock_agent._debug.side_effect = make_buggy_node

        # _execute_code 也需要 mock（返回仍有 exc_type 的结果）
        exec_result = MagicMock()
        exec_result.term_out = ["error output"]
        exec_result.exec_time = 0.1
        exec_result.exc_type = "RuntimeError"
        exec_result.exc_info = "RuntimeError: ..."

        with patch.object(orchestrator, "_execute_code", return_value=exec_result):
            result = orchestrator._debug_chain(
                node, mock_agent, context=mock_context, max_attempts=2
            )

        # 耗尽 2 次后，原节点应被标记为 dead
        assert result.dead is True
        assert result.debug_attempts == 2
        assert mock_agent._debug.call_count == 2


# ============================================================
# 以下测试迁移自 test_p0_fixes.py
# ============================================================


def _make_metric_node(
    metric: float, lower_is_better: bool = False, buggy: bool = False
) -> Node:
    """创建带 metric 的测试用 Node。"""
    node = Node(code="pass")
    node.metric_value = metric if not buggy else None
    node.lower_is_better = lower_is_better
    node.is_buggy = buggy
    return node


@pytest.fixture
def orch_metric(mock_config, journal):
    """创建带 metric task_desc 的 Orchestrator（用于 metric 相关测试）。"""
    agent = Mock()
    agent.name = "test_agent"
    return Orchestrator(
        agents=[agent],
        config=mock_config,
        journal=journal,
        task_desc="Evaluate using log_loss metric. AUC is also reported.",
    )


class TestSanitizeMetricValue:
    """_sanitize_metric_value() 负数 metric 修正测试。"""

    def test_sanitize_negative_rmsle(self, orch_metric):
        """metric=-0.065, metric_name='rmsle' → 返回 0.065。"""
        result = orch_metric._sanitize_metric_value(-0.065, "rmsle")
        assert result == pytest.approx(0.065)

    def test_sanitize_negative_rmse(self, orch_metric):
        """metric=-0.5, metric_name='rmse' → 返回 0.5。"""
        result = orch_metric._sanitize_metric_value(-0.5, "rmse")
        assert result == pytest.approx(0.5)

    def test_sanitize_negative_kappa_kept(self, orch_metric):
        """metric=-0.3, metric_name='qwk' → 保留 -0.3（合法负值）。"""
        result = orch_metric._sanitize_metric_value(-0.3, "qwk")
        assert result == pytest.approx(-0.3)

    def test_sanitize_positive_unchanged(self, orch_metric):
        """metric=0.85, metric_name='auc' → 保留 0.85。"""
        result = orch_metric._sanitize_metric_value(0.85, "auc")
        assert result == pytest.approx(0.85)

    def test_sanitize_unknown_metric_kept(self, orch_metric):
        """metric=-0.5, metric_name='custom_score' → 保留 -0.5。"""
        result = orch_metric._sanitize_metric_value(-0.5, "custom_score")
        assert result == pytest.approx(-0.5)


class TestPlausibilityNoBestNode:
    """_check_metric_plausibility() 无 best_node 时仍执行 METRIC_BOUNDS 检查。"""

    def test_plausibility_no_best_node_bounds_still_checked(self, orch_metric):
        """best_node=None 时，METRIC_BOUNDS 绝对范围检查仍生效。"""
        orch_metric.best_node = None
        orch_metric._task_desc_clean = "Evaluate using auc metric"
        assert orch_metric._check_metric_plausibility(1.5) is False

    def test_plausibility_no_best_node_normal_passes(self, orch_metric):
        """best_node=None 时，正常值仍通过。"""
        orch_metric.best_node = None
        orch_metric._task_desc_clean = "Evaluate using auc metric"
        assert orch_metric._check_metric_plausibility(0.85) is True


class TestMetricBounds:
    """METRIC_BOUNDS 常量测试。"""

    def test_bounds_structure(self):
        """所有 bounds entry 格式为 (min, max)。"""
        for key, (lo, hi) in METRIC_BOUNDS.items():
            if lo is not None and hi is not None:
                assert lo < hi, f"{key}: min={lo} >= max={hi}"


class TestCheckMetricPlausibility:
    """_check_metric_plausibility() 范围校验测试。"""

    def test_logloss_zero_rejected(self, orch_metric):
        """logloss=0.0 被拒绝（理论不可能）。"""
        orch_metric.best_node = _make_metric_node(0.265)
        assert orch_metric._check_metric_plausibility(0.0) is False

    def test_logloss_negative_rejected(self, orch_metric):
        """logloss 负值被拒绝。"""
        orch_metric.best_node = _make_metric_node(0.265)
        assert orch_metric._check_metric_plausibility(-0.1) is False

    def test_auc_overflow_rejected(self, orch_metric):
        """AUC > 1.0 被拒绝。"""
        orch_metric._task_desc_clean = "Evaluate using AUC metric"
        orch_metric.best_node = _make_metric_node(0.85)
        assert orch_metric._check_metric_plausibility(1.5) is False

    def test_normal_metric_passes(self, orch_metric):
        """正常范围内的 metric 通过检查。"""
        orch_metric.best_node = _make_metric_node(0.265)
        assert orch_metric._check_metric_plausibility(0.280) is True

    def test_no_best_node_passes(self, orch_metric):
        """无 best_node 时默认通过。"""
        orch_metric.best_node = None
        assert orch_metric._check_metric_plausibility(0.5) is True

    def test_metric_zero_with_high_best_rejected(self, orch_metric):
        """metric=0.0 且 best > 0.01 时被拒绝。"""
        orch_metric._task_desc_clean = "Some generic task"
        orch_metric.best_node = _make_metric_node(0.5)
        assert orch_metric._check_metric_plausibility(0.0) is False

    def test_ratio_check_rejects_extreme(self, orch_metric):
        """相对比率超过 50 倍被拒绝。"""
        orch_metric._task_desc_clean = "Some generic task"
        orch_metric.best_node = _make_metric_node(0.85)
        assert orch_metric._check_metric_plausibility(500.0) is False

    def test_ratio_check_passes_within_bounds(self, orch_metric):
        """相对比率在 50 倍内通过。"""
        orch_metric._task_desc_clean = "Some generic task"
        orch_metric.best_node = _make_metric_node(0.85)
        assert orch_metric._check_metric_plausibility(1.2) is True

    def test_negative_rmse_rejected(self, orch_metric):
        """RMSE 负值被绝对范围拒绝。"""
        orch_metric._task_desc_clean = "Evaluate using rmse metric"
        orch_metric.best_node = _make_metric_node(0.5)
        assert orch_metric._check_metric_plausibility(-0.1) is False


class TestReviewIntegration:
    """_review_node() 集成测试：纯文本 JSON 方案。"""

    def _make_review_json(self, **overrides) -> str:
        """构建完整 10 字段 review JSON 字符串。"""
        data = {
            "is_bug": False,
            "metric": 0.90,
            "lower_is_better": False,
            "has_csv_submission": True,
            "key_change": "test change",
            "insight": "test insight",
            "metric_name": "log_loss",
            "bottleneck": "test bottleneck",
            "suggested_direction": "test direction",
            "approach_tag": "test approach",
        }
        data.update(overrides)
        return json.dumps(data)

    def test_llm_metric_used_directly(self, orch_metric):
        """LLM 返回 metric 直接使用。"""
        node = Node(code="print('test')", plan="plan")
        node.term_out = "Training done.\nSaved."
        node.exec_time = 1.0
        node.exc_type = None

        with (
            patch.object(orch_metric, "_check_submission_exists", return_value=True),
            patch.object(
                orch_metric,
                "_validate_submission_format",
                return_value={"valid": True, "errors": [], "row_count": 100},
            ),
            patch("core.orchestrator.backend_query") as mock_query,
        ):
            mock_query.return_value = self._make_review_json(metric=0.85)
            orch_metric._review_node(node)

        assert node.metric_value == pytest.approx(0.85)
        assert node.is_buggy is False

    def test_none_metric_marks_buggy(self, orch_metric):
        """LLM 返回 metric=None 时节点标记为 buggy。"""
        node = Node(code="print('test')", plan="plan")
        node.term_out = "Training complete. No metric output."
        node.exec_time = 1.0
        node.exc_type = None

        with (
            patch.object(orch_metric, "_check_submission_exists", return_value=True),
            patch.object(
                orch_metric,
                "_validate_submission_format",
                return_value={"valid": True, "errors": [], "row_count": 100},
            ),
            patch("core.orchestrator.backend_query") as mock_query,
        ):
            mock_query.return_value = self._make_review_json(metric=None)
            orch_metric._review_node(node)

        assert node.metric_value is None
        assert node.is_buggy is True

    def test_retry_on_missing_fields(self, orch_metric):
        """首次返回缺失字段时触发重试，第二次成功。"""
        node = Node(code="print('test')", plan="plan")
        node.term_out = "Validation metric: 0.90"
        node.exec_time = 1.0
        node.exc_type = None

        incomplete_json = json.dumps({
            "is_bug": False, "metric": 0.90,
            "lower_is_better": False, "has_csv_submission": True,
            "key_change": "test", "insight": "test",
            "metric_name": "auc",
        })
        complete_json = self._make_review_json(metric=0.90)

        with (
            patch.object(orch_metric, "_check_submission_exists", return_value=True),
            patch.object(
                orch_metric,
                "_validate_submission_format",
                return_value={"valid": True, "errors": [], "row_count": 100},
            ),
            patch("core.orchestrator.backend_query") as mock_query,
        ):
            mock_query.side_effect = [incomplete_json, complete_json]
            orch_metric._review_node(node)

        assert node.metric_value == pytest.approx(0.90)
        assert node.is_buggy is False
        assert mock_query.call_count == 2


class TestReviewPrompt:
    """Review prompt 增强测试。"""

    def test_alignment_check_in_prompt(self, orch_metric):
        """Review prompt 包含 Metric Alignment Check 段落。"""
        node = Node(
            code="print('test')", plan="plan", term_out="output", exec_time=1.0,
        )
        messages = orch_metric._build_review_messages(
            node, "(Initial solution, no diff)"
        )
        assert "Metric Alignment Check" in messages
        assert "Focal Loss" in messages

    def test_json_template_in_prompt(self, orch_metric):
        """Review prompt 包含 JSON 输出模板（10 个字段）。"""
        node = Node(
            code="print('test')", plan="plan", term_out="output", exec_time=1.0,
        )
        messages = orch_metric._build_review_messages(
            node, "(Initial solution, no diff)"
        )
        assert "submit_review" not in messages
        assert '"is_bug"' in messages
        assert '"metric_name"' in messages
        assert '"approach_tag"' in messages
        assert '"bottleneck"' in messages
        assert "ALL 10 keys required" in messages


class TestValidateSubmissionFormat:
    """_validate_submission_format() 测试。"""

    def test_missing_file(self, orch_metric, tmp_path):
        """文件不存在。"""
        orch_metric.config.project.workspace_dir = tmp_path
        result = orch_metric._validate_submission_format("nonexistent")
        assert result["valid"] is False
        assert "不存在" in result["errors"][0]

    def test_valid_submission(self, orch_metric, tmp_path):
        """正常 submission 通过。"""
        orch_metric.config.project.workspace_dir = tmp_path
        sub_dir = tmp_path / "submission"
        sub_dir.mkdir()
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        import pandas as pd

        sample = pd.DataFrame({"id": [1, 2, 3], "target": [0, 1, 0]})
        sample.to_csv(input_dir / "sample_submission.csv", index=False)
        sub = pd.DataFrame({"id": [1, 2, 3], "target": [0.1, 0.9, 0.2]})
        sub.to_csv(sub_dir / "submission_test123.csv", index=False)

        result = orch_metric._validate_submission_format("test123")
        assert result["valid"] is True
        assert result["row_count"] == 3

    def test_nan_values_rejected(self, orch_metric, tmp_path):
        """包含 NaN 的 submission 被拒绝。"""
        orch_metric.config.project.workspace_dir = tmp_path
        sub_dir = tmp_path / "submission"
        sub_dir.mkdir()

        import pandas as pd
        import numpy as np

        sub = pd.DataFrame({"id": [1, 2, 3], "target": [0.1, np.nan, 0.2]})
        sub.to_csv(sub_dir / "submission_nan_test.csv", index=False)

        result = orch_metric._validate_submission_format("nan_test")
        assert result["valid"] is False
        assert any("NaN" in e for e in result["errors"])

    def test_row_count_mismatch_rejected(self, orch_metric, tmp_path):
        """行数不匹配被拒绝。"""
        orch_metric.config.project.workspace_dir = tmp_path
        sub_dir = tmp_path / "submission"
        sub_dir.mkdir()
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        import pandas as pd

        sample = pd.DataFrame({"id": range(100), "target": range(100)})
        sample.to_csv(input_dir / "sample_submission.csv", index=False)
        sub = pd.DataFrame({"id": range(70), "target": range(70)})
        sub.to_csv(sub_dir / "submission_rowtest.csv", index=False)

        result = orch_metric._validate_submission_format("rowtest")
        assert result["valid"] is False
        assert any("行数不匹配" in e for e in result["errors"])

    def test_sample_submission_camelcase_fallback(self, orch_metric, tmp_path):
        """支持 sampleSubmission.csv 驼峰命名变体。"""
        orch_metric.config.project.workspace_dir = tmp_path
        sub_dir = tmp_path / "submission"
        sub_dir.mkdir()
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        import pandas as pd

        sample = pd.DataFrame({"id": [1, 2], "target": [0, 1]})
        sample.to_csv(input_dir / "sampleSubmission.csv", index=False)
        sub = pd.DataFrame({"id": [1, 2], "target": [0.5, 0.8]})
        sub.to_csv(sub_dir / "submission_camel.csv", index=False)

        result = orch_metric._validate_submission_format("camel")
        assert result["valid"] is True
        assert result["row_count"] == 2

    def test_invalid_submission_marks_buggy_in_review(self, orch_metric, tmp_path):
        """submission 格式无效时 _review_node 标记节点为 buggy。"""
        orch_metric.config.project.workspace_dir = tmp_path
        sub_dir = tmp_path / "submission"
        sub_dir.mkdir()

        import pandas as pd
        import numpy as np

        node = Node(code="print('test')", plan="plan")
        node.term_out = "Validation metric: 0.90"
        node.exec_time = 1.0
        node.exc_type = None

        sub = pd.DataFrame({"id": [1, 2], "target": [0.5, np.nan]})
        sub.to_csv(sub_dir / f"submission_{node.id}.csv", index=False)

        review_json = json.dumps({
            "is_bug": False, "metric": 0.90, "lower_is_better": False,
            "has_csv_submission": True, "key_change": "test change",
            "insight": "test insight", "metric_name": "auc",
            "bottleneck": "test bottleneck", "suggested_direction": "test direction",
            "approach_tag": "test approach",
        })

        with patch("core.orchestrator.backend_query", return_value=review_json):
            orch_metric._review_node(node)

        assert node.is_buggy is True
        assert node.metric_value is None


# ============================================================
# 以下测试迁移自 test_p0_v6_fixes.py
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
        orch = MagicMock()
        orch.journal = MagicMock()
        orch.journal.nodes = nodes
        orch.journal_lock = threading.Lock()
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
                dead=True, exc_type="TimeoutError",
                approach_tag="ResNet50 + ImageNet", exec_time=3600.0,
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
                dead=True, exc_type="TimeoutExpired",
                approach_tag="Heavy CNN", exec_time=1800.0,
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


class TestGetCachedDataPreview:
    """测试 _get_cached_data_preview() 缓存读取。"""

    def test_file_exists_returns_content(self, tmp_path):
        """缓存文件存在时返回内容。"""
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
        assert (tmp_path / "logs" / "data_preview.md").exists()

    def test_cache_miss_no_input_dir_returns_empty(self, tmp_path):
        """缓存不存在且 input 目录也不存在时返回空字符串。"""
        orch = MagicMock()
        orch.config.project.workspace_dir = tmp_path
        orch._get_cached_data_preview = Orchestrator._get_cached_data_preview.__get__(orch)

        result = orch._get_cached_data_preview()
        assert result == ""

    def test_logs_dir_not_exists_returns_empty(self, tmp_path):
        """logs 目录不存在且无 input 时返回空字符串。"""
        orch = MagicMock()
        orch.config.project.workspace_dir = tmp_path / "nonexistent"
        orch._get_cached_data_preview = Orchestrator._get_cached_data_preview.__get__(orch)

        result = orch._get_cached_data_preview()
        assert result == ""


# ============================================================
# 额外覆盖率测试
# ============================================================


class TestCheckTimeLimit:
    """_check_time_limit() 测试。"""

    def test_within_limit(self, orchestrator):
        """未达时间限制返回 False。"""
        orchestrator.start_time = time.time()
        assert orchestrator._check_time_limit() is False

    def test_exceeded_limit(self, orchestrator):
        """超过时间限制返回 True。"""
        orchestrator.start_time = time.time() - 99999
        assert orchestrator._check_time_limit() is True


class TestPrepareStep:
    """_prepare_step() 测试。"""

    def test_creates_submission_dir(self, orchestrator, tmp_path):
        """创建 submission 目录。"""
        orchestrator.config.project.workspace_dir = tmp_path
        orchestrator._prepare_step()
        assert (tmp_path / "submission").exists()


class TestGenerateCodeDiff:
    """_generate_code_diff() 测试。"""

    def test_initial_solution(self, orchestrator):
        """首次生成返回 Initial solution。"""
        result = orchestrator._generate_code_diff(None, "print('hello')")
        assert "Initial solution" in result

    def test_no_changes(self, orchestrator):
        """代码无变更返回 No changes detected。"""
        code = "print('hello')"
        result = orchestrator._generate_code_diff(code, code)
        assert "No changes" in result

    def test_with_changes(self, orchestrator):
        """代码有变更返回 diff。"""
        result = orchestrator._generate_code_diff("a = 1\nb = 2", "a = 1\nb = 3")
        assert "-b = 2" in result
        assert "+b = 3" in result

    def test_truncation(self, orchestrator):
        """超长 diff 被截断。"""
        parent = "\n".join(f"line_{i}" for i in range(200))
        current = "\n".join(f"changed_{i}" for i in range(200))
        result = orchestrator._generate_code_diff(parent, current)
        assert "truncated" in result


class TestUpdateBestNodeExtended:
    """_update_best_node() 额外路径。"""

    def test_skip_none_metric(self, orchestrator):
        """metric_value 为 None 时跳过。"""
        node = Node(code="code", plan="plan", metric_value=None, is_buggy=False)
        orchestrator._update_best_node(node)
        assert orchestrator.best_node is None

    def test_skip_buggy_node(self, orchestrator):
        """buggy 节点跳过。"""
        node = Node(code="code", plan="plan", metric_value=0.8, is_buggy=True)
        orchestrator._update_best_node(node)
        assert orchestrator.best_node is None

    def test_replace_invalid_best(self, orchestrator):
        """best_node 自身 is_buggy=True 时被替换。"""
        old_best = Node(code="old", plan="p", metric_value=None, is_buggy=True)
        orchestrator.best_node = old_best

        new_node = Node(code="new", plan="p", metric_value=0.5, is_buggy=False)
        orchestrator._update_best_node(new_node)
        assert orchestrator.best_node == new_node


class TestCheckSubmissionExists:
    """_check_submission_exists() 测试。"""

    def test_exists(self, orchestrator, tmp_path):
        """submission 文件存在。"""
        orchestrator.config.project.workspace_dir = tmp_path
        sub_dir = tmp_path / "submission"
        sub_dir.mkdir()
        (sub_dir / "submission_abc.csv").write_text("id,val\n1,2")

        assert orchestrator._check_submission_exists("abc") is True

    def test_not_exists(self, orchestrator, tmp_path):
        """submission 文件不存在。"""
        orchestrator.config.project.workspace_dir = tmp_path
        sub_dir = tmp_path / "submission"
        sub_dir.mkdir()

        assert orchestrator._check_submission_exists("nonexistent") is False


class TestSaveNodeSolution:
    """_save_node_solution() 测试。"""

    def test_saves_code_and_output(self, orchestrator, tmp_path):
        """保存节点代码和输出到目录。"""
        orchestrator.config.project.workspace_dir = tmp_path
        (tmp_path / "working").mkdir()

        node = Node(code="print('test')", plan="Test plan")
        node.term_out = "test output"
        node.exec_time = 1.5
        node.exc_type = None
        node.is_buggy = False
        node.metric_value = 0.9
        node.metadata = {}

        orchestrator._save_node_solution(node)

        node_dir = tmp_path / "working" / f"solution_{node.id[:8]}"
        assert node_dir.exists()
        assert (node_dir / "solution.py").read_text() == "print('test')"
        assert (node_dir / "output.txt").exists()
        assert (node_dir / "plan.txt").exists()


class TestPrintNodeSummary:
    """_print_node_summary() 测试。"""

    def test_success_summary(self, orchestrator):
        """成功节点打印不抛异常。"""
        node = Node(code="code", plan="plan")
        node.metric_value = 0.85
        node.is_buggy = False
        node.exec_time = 2.0
        node.lower_is_better = False

        # 不应抛异常
        orchestrator._print_node_summary(node)

    def test_buggy_summary(self, orchestrator):
        """Buggy 节点打印不抛异常。"""
        node = Node(code="code", plan="plan")
        node.metric_value = None
        node.is_buggy = True
        node.exec_time = 0.1
        node.lower_is_better = False

        orchestrator._print_node_summary(node)


class TestEstimateTimeout:
    """_estimate_timeout() 测试。"""

    def test_adaptive_disabled(self, orchestrator):
        """自适应超时禁用时返回固定值。"""
        orchestrator.config.execution.adaptive_timeout = False
        orchestrator.config.execution.timeout = 3600

        result = orchestrator._estimate_timeout()
        assert result == 3600

    def test_no_input_dir(self, orchestrator, tmp_path):
        """input 目录不存在时返回基础超时。"""
        orchestrator.config.execution.adaptive_timeout = True
        orchestrator.config.execution.timeout = 3600
        orchestrator.config.execution.timeout_max = 10800
        orchestrator.config.project.workspace_dir = tmp_path

        result = orchestrator._estimate_timeout()
        assert result == 3600

    def test_small_dataset(self, orchestrator, tmp_path):
        """小数据集返回 1.0x。"""
        orchestrator.config.execution.adaptive_timeout = True
        orchestrator.config.execution.timeout = 3600
        orchestrator.config.execution.timeout_max = 10800
        orchestrator.config.project.workspace_dir = tmp_path
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "small.csv").write_text("a,b\n1,2")

        result = orchestrator._estimate_timeout()
        assert result == 3600  # 1.0x

    def test_large_dataset(self, orchestrator, tmp_path):
        """大数据集返回更高超时。"""
        orchestrator.config.execution.adaptive_timeout = True
        orchestrator.config.execution.timeout = 3600
        orchestrator.config.execution.timeout_max = 10800
        orchestrator.config.project.workspace_dir = tmp_path
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        # 创建 >100MB 文件
        (input_dir / "big.bin").write_bytes(b"\x00" * (101 * 1024 * 1024))

        result = orchestrator._estimate_timeout()
        assert result == 5400  # 1.5x


class TestBuildReviewMessagesExtended:
    """_build_review_messages() 额外路径。"""

    def test_with_parent_node(self, orch_metric):
        """带父节点的 Review 消息。"""
        parent = Node(code="old code", plan="old plan")
        parent.metric_value = 0.75
        parent.exc_type = None

        node = Node(code="new code", plan="new plan")
        node.term_out = "output"
        node.exec_time = 1.0

        messages = orch_metric._build_review_messages(
            node, "diff here", parent_node=parent
        )
        assert "0.7500" in messages
        assert "Baseline" in messages

    def test_with_best_node(self, orch_metric):
        """有 best_node 时显示 Current Best。"""
        orch_metric.best_node = _make_metric_node(0.90)

        node = Node(code="code", plan="plan", term_out="out", exec_time=1.0)
        messages = orch_metric._build_review_messages(
            node, "(Initial solution, no diff)"
        )
        assert "0.9000" in messages


class TestSaveBestSolution:
    """_save_best_solution() 测试。"""

    def test_saves_code(self, orchestrator, tmp_path):
        """保存最佳解决方案代码。"""
        orchestrator.config.project.workspace_dir = tmp_path

        node = Node(code="best_code()")
        orchestrator._save_best_solution(node)

        assert (tmp_path / "best_solution" / "solution.py").read_text() == "best_code()"

    def test_copies_submission(self, orchestrator, tmp_path):
        """复制 submission 文件到 best_solution。"""
        orchestrator.config.project.workspace_dir = tmp_path
        sub_dir = tmp_path / "submission"
        sub_dir.mkdir()

        node = Node(code="code()")
        (sub_dir / f"submission_{node.id}.csv").write_text("id,val\n1,2")

        orchestrator._save_best_solution(node)

        assert (tmp_path / "best_solution" / "submission.csv").exists()


# ============================================================
# 新增覆盖率测试：_condense_output, _check_submission_and_set_error,
#                _get_cached_data_preview, _review_node Phase 9
# ============================================================


class TestCondenseOutput:
    """_condense_output() 压缩执行输出测试。"""

    def test_short_output_returned_as_is(self, orchestrator):
        """输出未超过阈值时直接返回原文，不调用 LLM。"""
        short_text = "Training done.\nValidation metric: 0.95"
        with patch("core.orchestrator.backend_query") as mock_query:
            result = orchestrator._condense_output(short_text)
        assert result == short_text
        mock_query.assert_not_called()

    def test_empty_output_returns_empty(self, orchestrator):
        """空字符串输入返回空字符串。"""
        with patch("core.orchestrator.backend_query") as mock_query:
            result = orchestrator._condense_output("")
        assert result == ""
        mock_query.assert_not_called()

    def test_long_output_triggers_llm(self, orchestrator):
        """超过 8000 字符的输出触发 LLM 调用。"""
        long_text = "x" * 9000
        condensed = "=== EXECUTION SUMMARY ===\n[STATUS]\nsuccess"
        with patch("core.orchestrator.backend_query", return_value=condensed) as mock_query:
            result = orchestrator._condense_output(long_text)
        assert result == condensed
        mock_query.assert_called_once()

    def test_exactly_at_threshold_not_condensed(self, orchestrator):
        """恰好等于 8000 字符时不触发 LLM（len <= max_len）。"""
        text_at_limit = "y" * 8000
        with patch("core.orchestrator.backend_query") as mock_query:
            result = orchestrator._condense_output(text_at_limit)
        assert result == text_at_limit
        mock_query.assert_not_called()

    def test_long_output_llm_receives_prompt_with_text(self, orchestrator):
        """LLM 调用时 prompt 包含原始 term_out 内容。"""
        long_text = "important log line\n" + "z" * 9000
        with patch("core.orchestrator.backend_query", return_value="summary") as mock_query:
            orchestrator._condense_output(long_text)
        call_kwargs = mock_query.call_args
        user_msg = call_kwargs[1]["user_message"] if call_kwargs[1] else call_kwargs[0][1]
        assert "important log line" in user_msg


class TestCheckSubmissionAndSetError:
    """_check_submission_and_set_error() 三条提前返回路径 + 校验失败路径测试。"""

    def test_early_return_when_exc_type_present(self, orchestrator):
        """node.exc_type 非 None 时立即返回，不调用文件检查。"""
        node = Node(code="code", plan="plan")
        node.exc_type = "RuntimeError"
        node.term_out = "error output"

        with patch.object(orchestrator, "_check_submission_exists") as mock_check:
            orchestrator._check_submission_and_set_error(node)

        mock_check.assert_not_called()
        assert node.exc_type == "RuntimeError"  # 未被修改

    def test_early_return_when_no_submission_file(self, orchestrator):
        """node.exc_type=None 但 submission 文件不存在时提前返回。"""
        node = Node(code="code", plan="plan")
        node.exc_type = None
        node.term_out = "output"

        with patch.object(orchestrator, "_check_submission_exists", return_value=False):
            with patch.object(orchestrator, "_validate_submission_format") as mock_validate:
                orchestrator._check_submission_and_set_error(node)

        mock_validate.assert_not_called()
        assert node.exc_type is None  # 未被修改

    def test_early_return_when_submission_valid(self, orchestrator):
        """submission 文件存在且格式有效时提前返回，不修改节点。"""
        node = Node(code="code", plan="plan")
        node.exc_type = None
        node.term_out = "output"

        with patch.object(orchestrator, "_check_submission_exists", return_value=True):
            with patch.object(
                orchestrator,
                "_validate_submission_format",
                return_value={"valid": True, "errors": [], "row_count": 100},
            ):
                orchestrator._check_submission_and_set_error(node)

        assert node.exc_type is None  # 未被修改

    def test_validation_failure_sets_exc_type(self, orchestrator):
        """submission 格式校验失败时设置 exc_type='SubmissionValidationError'。"""
        node = Node(code="code", plan="plan")
        node.exc_type = None
        node.term_out = "training output"

        with patch.object(orchestrator, "_check_submission_exists", return_value=True):
            with patch.object(
                orchestrator,
                "_validate_submission_format",
                return_value={
                    "valid": False,
                    "errors": ["包含 NaN 值", "行数不匹配"],
                    "row_count": 0,
                },
            ):
                orchestrator._check_submission_and_set_error(node)

        assert node.exc_type == "SubmissionValidationError"
        assert "SUBMISSION VALIDATION FAILED" in node.term_out
        assert "包含 NaN 值" in node.term_out

    def test_validation_failure_appends_to_existing_term_out(self, orchestrator):
        """校验失败时错误信息追加到已有 term_out，而非覆盖。"""
        node = Node(code="code", plan="plan")
        node.exc_type = None
        node.term_out = "original output line"

        with patch.object(orchestrator, "_check_submission_exists", return_value=True):
            with patch.object(
                orchestrator,
                "_validate_submission_format",
                return_value={"valid": False, "errors": ["NaN detected"], "row_count": 0},
            ):
                orchestrator._check_submission_and_set_error(node)

        assert "original output line" in node.term_out
        assert "SUBMISSION VALIDATION FAILED" in node.term_out

    def test_validation_failure_multiple_errors_joined(self, orchestrator):
        """多个校验错误用分号拼接进入 term_out。"""
        node = Node(code="code", plan="plan")
        node.exc_type = None
        node.term_out = ""

        with patch.object(orchestrator, "_check_submission_exists", return_value=True):
            with patch.object(
                orchestrator,
                "_validate_submission_format",
                return_value={
                    "valid": False,
                    "errors": ["错误A", "错误B", "错误C"],
                    "row_count": 0,
                },
            ):
                orchestrator._check_submission_and_set_error(node)

        assert "错误A; 错误B; 错误C" in node.term_out


class TestGetCachedDataPreviewExtended:
    """_get_cached_data_preview() 额外路径测试（补充已有 TestGetCachedDataPreview）。"""

    def test_cache_read_error_returns_empty(self, tmp_path):
        """缓存文件存在但读取抛异常时返回空字符串。"""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        preview_file = logs_dir / "data_preview.md"
        preview_file.write_text("content", encoding="utf-8")

        orch = MagicMock()
        orch.config.project.workspace_dir = tmp_path
        orch._get_cached_data_preview = Orchestrator._get_cached_data_preview.__get__(orch)

        # 模拟 read_text 抛出 IO 错误
        with patch("pathlib.Path.read_text", side_effect=IOError("disk error")):
            result = orch._get_cached_data_preview()

        assert result == ""

    def test_generate_failure_returns_empty(self, tmp_path):
        """input 目录存在但生成函数抛异常时返回空字符串。"""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        orch = MagicMock()
        orch.config.project.workspace_dir = tmp_path
        orch._get_cached_data_preview = Orchestrator._get_cached_data_preview.__get__(orch)

        with patch("utils.data_preview.generate", side_effect=Exception("gen error")):
            result = orch._get_cached_data_preview()

        assert result == ""

    def test_cache_written_after_generate(self, tmp_path):
        """生成成功后结果被写入缓存文件。"""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "train.csv").write_text("id,target\n1,0\n2,1", encoding="utf-8")

        orch = MagicMock()
        orch.config.project.workspace_dir = tmp_path
        orch._get_cached_data_preview = Orchestrator._get_cached_data_preview.__get__(orch)

        fake_preview = "## Preview\n- train.csv: 2 rows"
        with patch("utils.data_preview.generate", return_value=fake_preview):
            result = orch._get_cached_data_preview()

        assert result == fake_preview
        cache_path = tmp_path / "logs" / "data_preview.md"
        assert cache_path.exists()
        assert cache_path.read_text(encoding="utf-8") == fake_preview


class TestReviewNodePhase9Pheromone:
    """_review_node() Phase 9 信息素计算测试。"""

    def _make_review_json(self, **overrides) -> str:
        """构建完整 10 字段 review JSON 字符串。"""
        data = {
            "is_bug": False,
            "metric": 0.85,
            "lower_is_better": False,
            "has_csv_submission": True,
            "key_change": "added features",
            "insight": "test insight",
            "metric_name": "auc",
            "bottleneck": "data quality",
            "suggested_direction": "more features",
            "approach_tag": "LightGBM baseline",
        }
        data.update(overrides)
        return json.dumps(data)

    def test_pheromone_computed_when_gene_registry_present(self, orchestrator, journal):
        """gene_registry 存在且节点非 buggy 时执行信息素计算。"""
        # 注入 gene_registry mock
        gene_registry = MagicMock()
        orchestrator.gene_registry = gene_registry

        # 预填 journal 中已有一个非 buggy 节点（scores 非空的前提）
        existing = Node(code="pass", plan="p")
        existing.metric_value = 0.75
        existing.is_buggy = False
        journal.append(existing)
        orchestrator.journal = journal

        node = Node(code="import lightgbm as lgb\nprint('done')", plan="plan")
        node.term_out = "Validation metric: 0.85"
        node.exec_time = 1.0
        node.exc_type = None

        with (
            patch.object(orchestrator, "_check_submission_exists", return_value=True),
            patch.object(
                orchestrator,
                "_validate_submission_format",
                return_value={"valid": True, "errors": [], "row_count": 100},
            ),
            patch("core.orchestrator.backend_query", return_value=self._make_review_json()),
            patch(
                "core.evolution.pheromone.compute_node_pheromone",
                return_value=0.72,
            ) as mock_pheromone,
            patch("core.evolution.pheromone.ensure_node_stats"),
        ):
            orchestrator._review_node(node)

        # 信息素结果被写入 metadata
        assert "pheromone_node" in node.metadata
        # gene_registry 被调用更新
        gene_registry.update_from_reviewed_node.assert_called_once_with(node)

    def test_pheromone_skipped_when_no_gene_registry(self, orchestrator, journal):
        """gene_registry=None 时跳过信息素计算。"""
        orchestrator.gene_registry = None
        orchestrator.journal = journal

        node = Node(code="print('test')", plan="plan")
        node.term_out = "Validation metric: 0.90"
        node.exec_time = 1.0
        node.exc_type = None

        with (
            patch.object(orchestrator, "_check_submission_exists", return_value=True),
            patch.object(
                orchestrator,
                "_validate_submission_format",
                return_value={"valid": True, "errors": [], "row_count": 100},
            ),
            patch("core.orchestrator.backend_query", return_value=self._make_review_json()),
        ):
            orchestrator._review_node(node)

        assert "pheromone_node" not in node.metadata

    def test_pheromone_skipped_when_node_buggy(self, orchestrator, journal):
        """节点为 buggy 时跳过 Phase 9 信息素计算。"""
        gene_registry = MagicMock()
        orchestrator.gene_registry = gene_registry
        orchestrator.journal = journal

        node = Node(code="raise ValueError()", plan="plan")
        node.term_out = ""
        node.exec_time = 0.1
        node.exc_type = "ValueError"  # 执行有异常 → is_buggy=True

        with (
            patch.object(orchestrator, "_check_submission_exists", return_value=False),
            patch(
                "core.orchestrator.backend_query",
                return_value=self._make_review_json(is_bug=True, metric=None),
            ),
        ):
            orchestrator._review_node(node)

        assert node.is_buggy is True
        assert "pheromone_node" not in node.metadata
        gene_registry.update_from_reviewed_node.assert_not_called()

    def test_pheromone_skipped_when_empty_scores(self, orchestrator, journal):
        """journal 中无有效节点（scores 为空列表）时不调用信息素函数。"""
        gene_registry = MagicMock()
        orchestrator.gene_registry = gene_registry
        # journal 为空，scores 列表为 []
        orchestrator.journal = journal

        node = Node(code="print('x')", plan="plan")
        node.term_out = "Validation metric: 0.80"
        node.exec_time = 1.0
        node.exc_type = None

        with (
            patch.object(orchestrator, "_check_submission_exists", return_value=True),
            patch.object(
                orchestrator,
                "_validate_submission_format",
                return_value={"valid": True, "errors": [], "row_count": 100},
            ),
            patch("core.orchestrator.backend_query", return_value=self._make_review_json()),
            patch("core.evolution.pheromone.compute_node_pheromone") as mock_pheromone,
        ):
            orchestrator._review_node(node)

        # scores 为空时不应调用 compute_node_pheromone
        mock_pheromone.assert_not_called()


class TestValidateReviewResponse:
    """_validate_review_response() 规则测试（类型检查、一致性修正）。"""

    def _base_response(self, **overrides) -> dict:
        """构建合法的基础 review 响应。"""
        data = {
            "is_bug": False,
            "metric": 0.85,
            "lower_is_better": False,
            "has_csv_submission": True,
            "key_change": "test change",
            "insight": "test insight",
            "metric_name": "auc",
            "bottleneck": "data quality",
            "suggested_direction": "more features",
            "approach_tag": "LightGBM",
        }
        data.update(overrides)
        return data

    def test_valid_response_passes_through(self, orchestrator):
        """合法响应原样通过。"""
        resp = self._base_response()
        result = orchestrator._validate_review_response(resp, MagicMock(), True)
        assert result["metric"] == pytest.approx(0.85)
        assert result["is_bug"] is False

    def test_missing_required_field_raises(self, orchestrator):
        """缺少必填字段时抛出 ValueError。"""
        resp = self._base_response()
        del resp["approach_tag"]
        with pytest.raises(ValueError, match="缺少必填字段"):
            orchestrator._validate_review_response(resp, MagicMock(), True)

    def test_empty_string_field_raises(self, orchestrator):
        """必填字段为空字符串时抛出 ValueError。"""
        resp = self._base_response(key_change="   ")
        with pytest.raises(ValueError, match="为空"):
            orchestrator._validate_review_response(resp, MagicMock(), True)

    def test_invalid_metric_type_set_to_none(self, orchestrator):
        """metric 类型非 float/int 时设为 None。"""
        resp = self._base_response(metric="not_a_number")
        result = orchestrator._validate_review_response(resp, MagicMock(), True)
        assert result["metric"] is None

    def test_is_bug_true_forces_metric_none(self, orchestrator):
        """is_bug=True 时强制将 metric 设为 None。"""
        resp = self._base_response(is_bug=True, metric=0.5)
        result = orchestrator._validate_review_response(resp, MagicMock(), True)
        assert result["metric"] is None

    def test_llm_claims_csv_but_no_file_overridden(self, orchestrator):
        """LLM 声称有 csv 但文件不存在时，has_csv_submission 被覆盖为 False。"""
        resp = self._base_response(has_csv_submission=True)
        result = orchestrator._validate_review_response(resp, MagicMock(), has_submission=False)
        assert result["has_csv_submission"] is False

    def test_metric_int_converted_to_float(self, orchestrator):
        """metric 为 int 时转换为 float。"""
        resp = self._base_response(metric=1)
        result = orchestrator._validate_review_response(resp, MagicMock(), True)
        assert isinstance(result["metric"], float)
        assert result["metric"] == pytest.approx(1.0)


# ============================================================
# 新增覆盖率测试：_save_node_solution 扩展路径
# ============================================================


class TestSaveNodeSolutionExtended:
    """_save_node_solution() 扩展路径测试。"""

    def test_saves_exc_info(self, orchestrator, tmp_path):
        """exc_info 非空时写入 output.txt 的异常信息段落。"""
        orchestrator.config.project.workspace_dir = tmp_path

        node = Node(code="print(1)")
        node.exc_info = "Traceback (most recent call last):\n  ...\nValueError: bad"
        node.term_out = ""
        node.exec_time = 0.1
        node.exc_type = "ValueError"
        node.is_buggy = True
        node.metric_value = None
        node.metadata = {}

        orchestrator._save_node_solution(node)

        node_dir = tmp_path / "working" / f"solution_{node.id[:8]}"
        output_txt = (node_dir / "output.txt").read_text(encoding="utf-8")
        assert "=== 异常信息 ===" in output_txt
        assert "ValueError: bad" in output_txt

    def test_saves_prompt_data(self, orchestrator, tmp_path):
        """prompt_data 非空时创建 prompt.json 文件。"""
        orchestrator.config.project.workspace_dir = tmp_path

        node = Node(code="print(1)")
        node.prompt_data = {"model": "gpt-4", "temperature": 0.5}
        node.term_out = ""
        node.exec_time = 0.2
        node.exc_type = None
        node.is_buggy = False
        node.metric_value = 0.9
        node.metadata = {}

        orchestrator._save_node_solution(node)

        node_dir = tmp_path / "working" / f"solution_{node.id[:8]}"
        assert (node_dir / "prompt.json").exists()
        data = json.loads((node_dir / "prompt.json").read_text(encoding="utf-8"))
        assert data["model"] == "gpt-4"
        assert data["temperature"] == pytest.approx(0.5)

    def test_saves_plan(self, orchestrator, tmp_path):
        """plan 非空时创建 plan.txt 文件。"""
        orchestrator.config.project.workspace_dir = tmp_path

        node = Node(code="print(1)", plan="Use XGBoost with 5-fold CV")
        node.term_out = ""
        node.exec_time = 0.3
        node.exc_type = None
        node.is_buggy = False
        node.metric_value = 0.85
        node.metadata = {}

        orchestrator._save_node_solution(node)

        node_dir = tmp_path / "working" / f"solution_{node.id[:8]}"
        assert (node_dir / "plan.txt").exists()
        assert "XGBoost" in (node_dir / "plan.txt").read_text(encoding="utf-8")

    def test_saves_review_debug(self, orchestrator, tmp_path):
        """metadata['review_debug'] 存在时创建 review.json 文件。"""
        orchestrator.config.project.workspace_dir = tmp_path

        node = Node(code="print(1)")
        node.term_out = ""
        node.exec_time = 0.4
        node.exc_type = None
        node.is_buggy = False
        node.metric_value = 0.88
        node.metadata = {"review_debug": {"response": "test review", "attempts": 1}}

        orchestrator._save_node_solution(node)

        node_dir = tmp_path / "working" / f"solution_{node.id[:8]}"
        assert (node_dir / "review.json").exists()
        data = json.loads((node_dir / "review.json").read_text(encoding="utf-8"))
        assert data["response"] == "test review"
        assert data["attempts"] == 1

    def test_exception_logged_not_raised(self, orchestrator, tmp_path):
        """文件系统操作失败时记录日志但不抛出异常。"""
        orchestrator.config.project.workspace_dir = tmp_path

        node = Node(code="print(1)")
        node.term_out = ""
        node.exec_time = 0.1
        node.exc_type = None
        node.is_buggy = False
        node.metric_value = 0.9
        node.metadata = {}

        # 模拟 mkdir 抛出 PermissionError
        with patch("pathlib.Path.mkdir", side_effect=PermissionError("denied")):
            # 不应抛出异常
            orchestrator._save_node_solution(node)

    def test_no_prompt_data_no_prompt_json(self, orchestrator, tmp_path):
        """prompt_data 为 None 时不创建 prompt.json。"""
        orchestrator.config.project.workspace_dir = tmp_path

        node = Node(code="print(1)")
        node.prompt_data = None
        node.term_out = ""
        node.exec_time = 0.1
        node.exc_type = None
        node.is_buggy = False
        node.metric_value = 0.9
        node.metadata = {}

        orchestrator._save_node_solution(node)

        node_dir = tmp_path / "working" / f"solution_{node.id[:8]}"
        assert not (node_dir / "prompt.json").exists()

    def test_no_review_debug_no_review_json(self, orchestrator, tmp_path):
        """metadata 中无 review_debug 时不创建 review.json。"""
        orchestrator.config.project.workspace_dir = tmp_path

        node = Node(code="print(1)")
        node.term_out = ""
        node.exec_time = 0.1
        node.exc_type = None
        node.is_buggy = False
        node.metric_value = 0.9
        node.metadata = {}

        orchestrator._save_node_solution(node)

        node_dir = tmp_path / "working" / f"solution_{node.id[:8]}"
        assert not (node_dir / "review.json").exists()

    def test_copies_submission_when_exists(self, orchestrator, tmp_path):
        """submission 文件存在时复制到节点目录。"""
        orchestrator.config.project.workspace_dir = tmp_path
        sub_dir = tmp_path / "submission"
        sub_dir.mkdir(parents=True)

        node = Node(code="print(1)")
        node.term_out = ""
        node.exec_time = 0.1
        node.exc_type = None
        node.is_buggy = False
        node.metric_value = 0.9
        node.metadata = {}

        (sub_dir / f"submission_{node.id}.csv").write_text("id,val\n1,2")

        orchestrator._save_node_solution(node)

        node_dir = tmp_path / "working" / f"solution_{node.id[:8]}"
        assert (node_dir / "submission.csv").exists()


# ============================================================
# 新增覆盖率测试：_save_best_solution 扩展路径
# ============================================================


class TestSaveBestSolutionExtended:
    """_save_best_solution() 扩展路径测试。"""

    def test_sync_to_home_submission_fails_gracefully(self, orchestrator, tmp_path):
        """同步到 /home/submission/ 失败时记录警告但不抛出异常。"""
        orchestrator.config.project.workspace_dir = tmp_path
        sub_dir = tmp_path / "submission"
        sub_dir.mkdir(parents=True)

        node = Node(code="best_code()")
        (sub_dir / f"submission_{node.id}.csv").write_text("id,val\n1,2")

        # 模拟 /home/submission/ 的 mkdir 抛出 PermissionError
        original_mkdir = type(sub_dir).mkdir

        def selective_mkdir(self, *args, **kwargs):
            if str(self) == "/home/submission":
                raise PermissionError("只读文件系统")
            return original_mkdir(self, *args, **kwargs)

        with patch("pathlib.Path.mkdir", selective_mkdir):
            # 不应抛出异常
            orchestrator._save_best_solution(node)

        # best_solution 本地目录仍应创建成功（mkdir 被 patch 后跳过，直接验证不抛即可）

    def test_sync_shutil_copy2_fails_gracefully(self, orchestrator, tmp_path):
        """shutil.copy2 同步失败时记录警告但不抛出异常。"""
        import shutil as shutil_mod

        orchestrator.config.project.workspace_dir = tmp_path
        sub_dir = tmp_path / "submission"
        sub_dir.mkdir(parents=True)

        node = Node(code="best_code()")
        (sub_dir / f"submission_{node.id}.csv").write_text("id,val\n1,2")

        call_count = {"n": 0}
        original_copy2 = shutil_mod.copy2

        def selective_copy2(src, dst, *args, **kwargs):
            # 第二次调用（同步到 /home/submission/）才抛出
            call_count["n"] += 1
            if call_count["n"] >= 2:
                raise PermissionError("只读文件系统")
            return original_copy2(src, dst, *args, **kwargs)

        with patch("core.orchestrator.shutil.copy2", selective_copy2):
            # 不应抛出异常
            orchestrator._save_best_solution(node)

        # best_solution/submission.csv 来自 shutil.copy（非 copy2），应仍存在
        assert (tmp_path / "best_solution" / "solution.py").exists()

    def test_save_best_exception_not_raised(self, orchestrator, tmp_path):
        """顶层 except 捕获所有异常，不向调用方抛出。"""
        orchestrator.config.project.workspace_dir = tmp_path

        node = Node(code="best_code()")

        # 模拟 best_dir.mkdir 抛出异常（触发最外层 except）
        with patch("pathlib.Path.mkdir", side_effect=OSError("no space")):
            # 不应抛出异常
            orchestrator._save_best_solution(node)

    def test_no_submission_file_skips_copy(self, orchestrator, tmp_path):
        """submission 文件不存在时跳过 copy，不抛出异常。"""
        orchestrator.config.project.workspace_dir = tmp_path

        node = Node(code="best_code()")
        # 不创建 submission 文件

        orchestrator._save_best_solution(node)

        best_dir = tmp_path / "best_solution"
        assert (best_dir / "solution.py").read_text() == "best_code()"
        assert not (best_dir / "submission.csv").exists()


# ============================================================
# 新增覆盖率测试：_debug_chain 扩展路径
# ============================================================


class TestDebugChainExtended:
    """_debug_chain() 扩展路径测试（补充已有 TestDebugChain）。"""

    def test_skip_timeout_error(self, orchestrator):
        """exc_type='TimeoutError' 时直接跳过 debug，返回原节点。"""
        node = Node(code="print(1)", exc_type="TimeoutError")
        mock_agent = MagicMock()

        result = orchestrator._debug_chain(node, mock_agent, context=None, max_attempts=2)

        assert result is node
        mock_agent._debug.assert_not_called()

    def test_skip_timeout_expired(self, orchestrator):
        """exc_type='TimeoutExpired'（旧版兼容别名）时跳过 debug。"""
        node = Node(code="print(1)", exc_type="TimeoutExpired")
        mock_agent = MagicMock()

        result = orchestrator._debug_chain(node, mock_agent, context=None, max_attempts=2)

        assert result is node
        mock_agent._debug.assert_not_called()

    def test_skip_memory_error(self, orchestrator):
        """exc_type='MemoryError' 时跳过 debug。"""
        node = Node(code="print(1)", exc_type="MemoryError")
        mock_agent = MagicMock()

        result = orchestrator._debug_chain(node, mock_agent, context=None, max_attempts=2)

        assert result is node
        mock_agent._debug.assert_not_called()

    def test_skip_none_exc_type(self, orchestrator):
        """exc_type=None 时跳过 debug（无错误无需修复）。"""
        node = Node(code="print(1)", exc_type=None)
        mock_agent = MagicMock()

        result = orchestrator._debug_chain(node, mock_agent, context=None, max_attempts=2)

        assert result is node
        mock_agent._debug.assert_not_called()

    def test_debug_success_on_first_attempt(self, orchestrator):
        """第一次 debug 成功（fixed_node.exc_type=None）时立即返回修复节点。"""
        node = Node(code="raise ValueError()", exc_type="ValueError")
        fixed = Node(code="print('fixed')", exc_type=None)

        mock_agent = MagicMock()
        mock_agent._debug.return_value = fixed

        mock_context = MagicMock()
        mock_context.task_desc = "test"
        mock_context.device_info = "CPU"
        mock_context.conda_packages = "pandas"
        mock_context.conda_env_name = "Swarm-Evo"

        exec_result = MagicMock()
        exec_result.term_out = ["all good"]
        exec_result.exec_time = 0.5
        exec_result.exc_type = None
        exec_result.exc_info = None

        with patch.object(orchestrator, "_execute_code", return_value=exec_result):
            with patch.object(orchestrator, "_get_cached_data_preview", return_value=""):
                result = orchestrator._debug_chain(
                    node, mock_agent, context=mock_context, max_attempts=2
                )

        assert result is fixed
        assert result.exc_type is None
        mock_agent._debug.assert_called_once()

    def test_debug_no_valid_code_stops_early(self, orchestrator):
        """agent._debug 返回无效代码（None）时停止并标记 dead。"""
        node = Node(code="raise ValueError()", exc_type="ValueError")

        mock_agent = MagicMock()
        mock_agent._debug.return_value = None  # 无效返回

        mock_context = MagicMock()
        mock_context.task_desc = "test"
        mock_context.device_info = "CPU"
        mock_context.conda_packages = "pandas"
        mock_context.conda_env_name = "Swarm-Evo"

        with patch.object(orchestrator, "_get_cached_data_preview", return_value=""):
            result = orchestrator._debug_chain(
                node, mock_agent, context=mock_context, max_attempts=2
            )

        assert result.dead is True
        mock_agent._debug.assert_called_once()  # 停止后不再重试

    def test_debug_same_code_stops_early(self, orchestrator):
        """agent._debug 返回与当前相同的代码时停止并标记 dead。"""
        original_code = "raise ValueError()"
        node = Node(code=original_code, exc_type="ValueError")

        # 返回相同代码的节点
        same_code_node = Node(code=original_code, exc_type="ValueError")
        mock_agent = MagicMock()
        mock_agent._debug.return_value = same_code_node

        mock_context = MagicMock()
        mock_context.task_desc = "test"
        mock_context.device_info = "CPU"
        mock_context.conda_packages = "pandas"
        mock_context.conda_env_name = "Swarm-Evo"

        with patch.object(orchestrator, "_get_cached_data_preview", return_value=""):
            result = orchestrator._debug_chain(
                node, mock_agent, context=mock_context, max_attempts=2
            )

        # 代码未变化，立即停止
        assert result.dead is True
        mock_agent._debug.assert_called_once()

    def test_uses_config_default_max_attempts(self, orchestrator):
        """max_attempts=None 时从 config 读取默认值。"""
        from omegaconf import OmegaConf

        # 注入 config.evolution.solution.debug_max_attempts = 1
        orchestrator.config = OmegaConf.merge(
            orchestrator.config,
            OmegaConf.create({"evolution": {"solution": {"debug_max_attempts": 1}}}),
        )

        node = Node(code="raise ValueError()", exc_type="ValueError")
        mock_agent = MagicMock()
        mock_agent._debug.return_value = None  # 无效返回，触发 break

        mock_context = MagicMock()
        mock_context.task_desc = "test"
        mock_context.device_info = "CPU"
        mock_context.conda_packages = "pandas"
        mock_context.conda_env_name = "Swarm-Evo"

        with patch.object(orchestrator, "_get_cached_data_preview", return_value=""):
            result = orchestrator._debug_chain(node, mock_agent, context=mock_context)

        # max_attempts=1，只调用一次
        mock_agent._debug.assert_called_once()
        assert result.dead is True


class TestExecuteCode:
    """测试 _execute_code 方法。"""

    def test_execute_code_rewrites_and_runs(self, orchestrator):
        """执行代码时重写 submission 路径并调用 interpreter。"""
        orchestrator.workspace = MagicMock()
        orchestrator.workspace.rewrite_submission_path.return_value = "modified_code"
        orchestrator.interpreter = MagicMock()
        mock_result = MagicMock()
        orchestrator.interpreter.run.return_value = mock_result

        result = orchestrator._execute_code("original_code", "node123")

        orchestrator.workspace.rewrite_submission_path.assert_called_once_with(
            "original_code", "node123"
        )
        orchestrator.interpreter.run.assert_called_once_with(
            "modified_code", node_id="node123", reset_session=True
        )
        assert result is mock_result


class TestEstimateTimeoutExtended:
    """测试 _estimate_timeout 的额外路径。"""

    def _enable_adaptive(self, orch):
        """辅助: 启用自适应超时并添加 timeout_max 配置。"""
        orch.config = OmegaConf.merge(
            orch.config,
            OmegaConf.create({"execution": {"adaptive_timeout": True, "timeout_max": 10800}}),
        )

    def test_image_count_multiplier(self, orchestrator):
        """大量图像文件触发 image_multiplier。"""
        self._enable_adaptive(orchestrator)
        input_dir = orchestrator.config.project.workspace_dir / "input"
        input_dir.mkdir(parents=True)

        # 创建 201 个假 .jpg 文件触发 image_count > 200 分支
        for i in range(201):
            (input_dir / f"img_{i:04d}.jpg").write_bytes(b"\xff\xd8" * 10)

        timeout = orchestrator._estimate_timeout()
        # image_multiplier=2.0，base=60 → 120
        assert timeout >= 120

    def test_very_large_dataset_multiplier(self, orchestrator):
        """超大数据集 (>1GB) 触发 3x 倍率。"""
        self._enable_adaptive(orchestrator)
        input_dir = orchestrator.config.project.workspace_dir / "input"
        input_dir.mkdir(parents=True)

        # 创建 >1024MB 的文件 — 用 mock 更高效
        with patch("pathlib.Path.rglob") as mock_rglob:
            mock_file = MagicMock()
            mock_file.is_file.return_value = True
            mock_file.stat.return_value = MagicMock(st_size=1100 * 1024 * 1024)
            mock_file.suffix = ".csv"
            mock_rglob.return_value = [mock_file]

            timeout = orchestrator._estimate_timeout()
            # size_multiplier=3.0, base=60 → 180
            assert timeout >= 180

    def test_large_image_collection_multiplier(self, orchestrator):
        """超过 1000 张图像触发 3x image_multiplier。"""
        self._enable_adaptive(orchestrator)
        input_dir = orchestrator.config.project.workspace_dir / "input"
        input_dir.mkdir(parents=True)

        with patch("pathlib.Path.rglob") as mock_rglob:
            mock_files = []
            for i in range(1001):
                mf = MagicMock()
                mf.is_file.return_value = True
                mf.stat.return_value = MagicMock(st_size=1000)
                mf.suffix = ".jpg"
                mock_files.append(mf)
            mock_rglob.return_value = mock_files

            timeout = orchestrator._estimate_timeout()
            # image_multiplier=3.0, base=60 → 180
            assert timeout >= 180

    def test_rglob_exception_returns_base(self, orchestrator):
        """rglob 异常时回退到基础超时。"""
        self._enable_adaptive(orchestrator)
        input_dir = orchestrator.config.project.workspace_dir / "input"
        input_dir.mkdir(parents=True)

        with patch("pathlib.Path.rglob", side_effect=PermissionError("denied")):
            timeout = orchestrator._estimate_timeout()
            assert timeout == 60  # base timeout


class TestValidateSubmissionFormatColumnReorder:
    """测试 _validate_submission_format 列顺序重排。"""

    def test_column_reorder_to_match_sample(self, orchestrator):
        """submission 列顺序与 sample 不同时自动重排。"""
        import pandas as pd

        ws = orchestrator.config.project.workspace_dir
        input_dir = ws / "input"
        input_dir.mkdir(exist_ok=True)
        submission_dir = ws / "submission"
        submission_dir.mkdir(exist_ok=True)

        # sample 列顺序: id, target
        sample_path = input_dir / "sample_submission.csv"
        pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]}).to_csv(
            sample_path, index=False
        )

        # submission 列顺序: target, id (反序)
        node_id = "reorder_test"
        sub_path = submission_dir / f"submission_{node_id}.csv"
        pd.DataFrame({"target": [0.5, 0.8], "id": [1, 2]}).to_csv(
            sub_path, index=False
        )

        result = orchestrator._validate_submission_format(node_id)
        assert result["valid"] is True

        # 验证文件已被重写为正确列顺序
        rewritten = pd.read_csv(sub_path)
        assert list(rewritten.columns) == ["id", "target"]


class TestReviewNodePhase0:
    """测试 _review_node Phase 0 变更上下文。"""

    def test_gene_plan_as_change_context(self, orchestrator):
        """merge 模式使用 gene_plan 作为变更上下文。"""
        node = Node(code="print(1)")

        with patch.object(orchestrator, "_check_submission_exists", return_value=False), \
             patch("core.orchestrator.backend_query") as mock_query:
            mock_query.return_value = json.dumps({
                "is_bug": True,
                "bug_description": "test",
                "has_csv_submission": False,
                "metric": None,
                "metric_name": None,
            })
            orchestrator._review_node(node, gene_plan="## Merge Plan\nCombine A+B")

        # 验证 gene_plan 被使用（不崩溃即通过）

    def test_parent_node_diff_as_change_context(self, orchestrator):
        """draft/mutate 模式使用父节点 diff 作为变更上下文。"""
        node = Node(code="print(2)")
        parent = Node(code="print(1)")

        with patch.object(orchestrator, "_check_submission_exists", return_value=False), \
             patch("core.orchestrator.backend_query") as mock_query:
            mock_query.return_value = json.dumps({
                "is_bug": True,
                "bug_description": "test",
                "has_csv_submission": False,
                "metric": None,
                "metric_name": None,
            })
            orchestrator._review_node(node, parent_node=parent)
