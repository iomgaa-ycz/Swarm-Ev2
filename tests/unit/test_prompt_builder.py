"""PromptManager 与 Journal 单元测试。

原 PromptBuilder 测试已移除（PromptBuilder 已删除）。
保留 Journal.generate_summary() 相关测试。
"""

from core.state import Node, Journal
from utils.prompt_manager import PromptManager


class TestPromptManager:
    """PromptManager 基础测试。"""

    def test_prompt_manager_init(self):
        """测试 PromptManager 初始化。"""
        from pathlib import Path

        base_dir = Path("benchmark/mle-bench")
        pm = PromptManager(
            template_dir=base_dir / "prompt_templates",
            skills_dir=base_dir / "skills",
            agent_configs_dir=base_dir / "agent_configs",
        )
        assert pm.env is not None

    def test_build_prompt_explore(self):
        """测试构建 explore prompt。"""
        from pathlib import Path

        base_dir = Path("benchmark/mle-bench")
        pm = PromptManager(
            template_dir=base_dir / "prompt_templates",
            skills_dir=base_dir / "skills",
            agent_configs_dir=base_dir / "agent_configs",
        )

        prompt = pm.build_prompt(
            "explore",
            "agent_0",
            {
                "task_desc": "Predict housing prices",
                "parent_node": None,
                "memory": "",
                "data_preview": None,
                "time_remaining": 3600,
                "steps_remaining": 10,
            },
        )

        # 验证包含必要部分
        assert "# Agent Role" in prompt
        assert "Predict housing prices" in prompt
        assert "# Response Format" in prompt or "## Output" in prompt

    def test_format_time(self):
        """测试时间格式化。"""
        from pathlib import Path

        base_dir = Path("benchmark/mle-bench")
        pm = PromptManager(
            template_dir=base_dir / "prompt_templates",
            skills_dir=base_dir / "skills",
            agent_configs_dir=base_dir / "agent_configs",
        )

        assert pm._format_time(0) == "0 seconds"
        assert pm._format_time(30) == "30 seconds"
        assert pm._format_time(60) == "1 minute"
        assert pm._format_time(3600) == "1 hour"


class TestJournalGenerateSummary:
    """Journal.generate_summary() 测试类。"""

    def test_generate_summary_empty(self):
        """测试空 Journal。"""
        journal = Journal()
        summary = journal.generate_summary()
        assert summary == "No previous solutions."

    def test_generate_summary_all_nodes(self):
        """测试包含所有节点（good + buggy）。"""
        journal = Journal()

        node1 = Node(
            code="x = 1",
            plan="Use RandomForest",
            analysis="Good solution",
            analysis_detail={
                "key_change": "Added RandomForest model",
                "insight": "Good solution",
                "bottleneck": None,
                "suggested_direction": "Try XGBoost",
            },
            metric_value=0.85,
            is_buggy=False,
        )

        node2 = Node(
            code="x = 2",
            plan="Try deep NN",
            analysis="NaN loss detected",
            analysis_detail={
                "key_change": "Switched to deep NN",
                "insight": "NaN loss detected due to exploding gradients",
            },
            is_buggy=True,
        )

        journal.append(node1)
        journal.append(node2)

        summary = journal.generate_summary()

        assert "## Current Best" in summary
        assert "0.85" in summary
        assert "## Changelog" in summary
        assert "## Constraints" in summary or "NaN loss" in summary

    def test_generate_summary_with_code(self):
        """测试 include_code=True。"""
        journal = Journal()

        node = Node(
            code="import pandas as pd\nprint('hello')",
            plan="Load data",
            analysis="Success",
            analysis_detail={
                "key_change": "Load data",
                "insight": "Success",
            },
            metric_value=0.9,
        )

        journal.append(node)
        summary = journal.generate_summary(include_code=False)
        assert "## Current Best" in summary
        assert "0.9" in summary

    def test_generate_summary_marks_buggy(self):
        """测试 buggy 节点在 Changelog 中有 BUGGY 标记。"""
        journal = Journal()

        node1 = Node(
            code="x = 1",
            plan="Good approach",
            analysis="Works well",
            analysis_detail={
                "key_change": "Good approach",
                "insight": "Works well",
            },
            metric_value=0.9,
            is_buggy=False,
        )

        node2 = Node(
            code="x = 2",
            plan="Buggy approach",
            analysis="Failed",
            analysis_detail={
                "key_change": "Buggy approach",
                "insight": "Failed due to memory error",
            },
            is_buggy=True,
        )

        journal.append(node1)
        journal.append(node2)

        summary = journal.generate_summary()

        assert "BUGGY" in summary
        assert "Failed" in summary or "memory error" in summary
