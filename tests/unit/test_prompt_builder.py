"""PromptBuilder 单元测试。

测试 Prompt 构建器的各种场景。
"""

from utils.prompt_builder import PromptBuilder
from core.state import Node


class TestPromptBuilder:
    """PromptBuilder 测试类。"""

    def test_prompt_builder_init(self):
        """测试 PromptBuilder 初始化。"""
        # 默认初始化
        builder1 = PromptBuilder()
        assert builder1.obfuscate is False

        # 混淆模式
        builder2 = PromptBuilder(obfuscate=True)
        assert builder2.obfuscate is True

    def test_build_explore_prompt_draft_mode(self):
        """测试初稿模式（parent_node=None）。"""
        builder = PromptBuilder(obfuscate=False)
        prompt = builder.build_explore_prompt(
            task_desc="Predict housing prices",
            parent_node=None,  # 初稿模式
            memory="",
            data_preview=None,
            time_remaining=3600,
            steps_remaining=10,
        )

        # 验证包含必要部分
        assert "# Introduction" in prompt
        assert "Kaggle grandmaster" in prompt  # obfuscate=False
        assert "# Task Description" in prompt
        assert "Predict housing prices" in prompt
        assert "# Guidelines" in prompt
        assert "# Response Format" in prompt

        # 验证不包含 "Previous Attempt"（初稿模式）
        assert "# Previous Attempt" not in prompt
        assert "# Execution Result" not in prompt

    def test_build_explore_prompt_improve_mode(self):
        """测试改进模式（parent_node 正常）。"""
        builder = PromptBuilder(obfuscate=False)

        # 创建父节点（正常执行）
        parent_node = Node(
            code="import pandas as pd\nprint('Accuracy: 0.85')",
            plan="Use RandomForest",
            term_out="Accuracy: 0.85",
            is_buggy=False,
            metric_value=0.85,
        )

        prompt = builder.build_explore_prompt(
            task_desc="Predict housing prices",
            parent_node=parent_node,  # 改进模式
            memory="",
            data_preview=None,
            time_remaining=3600,
            steps_remaining=10,
        )

        # 验证包含父节点信息
        assert "# Previous Attempt" in prompt
        assert "import pandas as pd" in prompt
        assert "# Execution Result" in prompt
        assert "Accuracy: 0.85" in prompt

    def test_build_explore_prompt_bugfix_mode(self):
        """测试修复模式（parent_node.is_buggy=True）。"""
        builder = PromptBuilder(obfuscate=False)

        # 创建父节点（有 bug）
        parent_node = Node(
            code="import pandas as pd\ndf = pd.read_csv('missing.csv')",
            plan="Load data",
            term_out="FileNotFoundError: missing.csv not found",
            exc_type="FileNotFoundError",
            exc_info="missing.csv not found",
            is_buggy=True,
        )

        prompt = builder.build_explore_prompt(
            task_desc="Predict housing prices",
            parent_node=parent_node,  # 修复模式
            memory="",
            data_preview=None,
            time_remaining=3600,
            steps_remaining=10,
        )

        # 验证包含错误信息
        assert "# Previous Attempt" in prompt
        assert "# Execution Result" in prompt
        assert "FileNotFoundError" in prompt
        assert "missing.csv not found" in prompt

    def test_build_explore_prompt_with_memory(self):
        """测试 Memory 机制。"""
        builder = PromptBuilder(obfuscate=False)

        memory_text = """Design: Use RandomForest
Results: Validation accuracy 0.85
Validation Metric: 0.85"""

        prompt = builder.build_explore_prompt(
            task_desc="Predict housing prices",
            parent_node=None,
            memory=memory_text,  # 包含 Memory
            data_preview=None,
            time_remaining=3600,
            steps_remaining=10,
        )

        # 验证包含 Memory 部分
        assert "# Memory" in prompt
        assert "Use RandomForest" in prompt
        assert "Validation accuracy 0.85" in prompt

    def test_build_explore_prompt_with_data_preview(self):
        """测试数据预览。"""
        builder = PromptBuilder(obfuscate=False)

        data_preview_text = """train.csv: 1000 rows, 10 columns
test.csv: 500 rows, 9 columns"""

        prompt = builder.build_explore_prompt(
            task_desc="Predict housing prices",
            parent_node=None,
            memory="",
            data_preview=data_preview_text,  # 包含数据预览
            time_remaining=3600,
            steps_remaining=10,
        )

        # 验证包含 Data Overview
        assert "# Data Overview" in prompt
        assert "train.csv: 1000 rows" in prompt
        assert "test.csv: 500 rows" in prompt

    def test_format_time(self):
        """测试时间格式化。"""
        builder = PromptBuilder()

        # 测试各种时间格式
        assert builder._format_time(0) == "0 seconds"
        assert builder._format_time(30) == "30 seconds"
        assert builder._format_time(60) == "1 minute"
        assert builder._format_time(120) == "2 minutes"
        assert builder._format_time(3600) == "1 hour"
        assert builder._format_time(3661) == "1 hour 1 minute"
        assert builder._format_time(7200) == "2 hours"
        assert builder._format_time(7320) == "2 hours 2 minutes"

    def test_obfuscate_mode(self):
        """测试混淆模式。"""
        builder = PromptBuilder(obfuscate=True)
        prompt = builder.build_explore_prompt(
            task_desc="Predict housing prices",
            parent_node=None,
            memory="",
            data_preview=None,
            time_remaining=3600,
            steps_remaining=10,
        )

        # 验证混淆模式使用通用角色介绍
        assert "expert machine learning engineer" in prompt
        assert "Kaggle grandmaster" not in prompt


class TestJournalGenerateSummary:
    """Journal.generate_summary() 测试类。

    注意：generate_summary() 现已重构为 Evolution Log 格式，
    包含 Current Best、Changelog、Constraints、Unexplored 等部分。
    """

    def test_generate_summary_empty(self):
        """测试空 Journal。"""
        from core.state import Journal

        journal = Journal()
        summary = journal.generate_summary()

        assert summary == "No previous solutions."

    def test_generate_summary_all_nodes(self):
        """测试包含所有节点（good + buggy）。"""
        from core.state import Journal

        journal = Journal()

        # 正常节点（使用新的 analysis_detail 格式）
        node1 = Node(
            code="x = 1",
            plan="Use RandomForest",
            analysis="Good solution",
            analysis_detail={
                "key_change": "Added RandomForest model",
                "metric_delta": "baseline",
                "insight": "Good solution",
                "bottleneck": None,
                "suggested_direction": "Try XGBoost",
            },
            metric_value=0.85,
            is_buggy=False,
        )

        # Buggy 节点
        node2 = Node(
            code="x = 2",
            plan="Try deep NN",
            analysis="NaN loss detected",
            analysis_detail={
                "key_change": "Switched to deep NN",
                "metric_delta": "N/A",
                "insight": "NaN loss detected due to exploding gradients",
            },
            is_buggy=True,
        )

        journal.append(node1)
        journal.append(node2)

        summary = journal.generate_summary()

        # 验证包含 Current Best 部分
        assert "## Current Best" in summary
        assert "0.85" in summary

        # 验证包含 Changelog 部分
        assert "## Changelog" in summary

        # 验证包含 buggy 节点的约束（Constraints 部分）
        assert "## Constraints" in summary or "NaN loss" in summary

    def test_generate_summary_with_code(self):
        """测试 include_code=True（当前版本不再包含代码）。"""
        from core.state import Journal

        journal = Journal()

        node = Node(
            code="import pandas as pd\nprint('hello')",
            plan="Load data",
            analysis="Success",
            analysis_detail={
                "key_change": "Load data",
                "metric_delta": "baseline",
                "insight": "Success",
            },
            metric_value=0.9,
        )

        journal.append(node)

        # 新版本的 generate_summary 不再支持 include_code
        # 但保持接口兼容
        summary = journal.generate_summary(include_code=False)
        assert "## Current Best" in summary
        assert "0.9" in summary

    def test_generate_summary_marks_buggy(self):
        """测试 buggy 节点在 Changelog 中有 BUGGY 标记。"""
        from core.state import Journal

        journal = Journal()

        node1 = Node(
            code="x = 1",
            plan="Good approach",
            analysis="Works well",
            analysis_detail={
                "key_change": "Good approach",
                "metric_delta": "baseline",
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
                "metric_delta": "N/A",
                "insight": "Failed due to memory error",
            },
            is_buggy=True,
        )

        journal.append(node1)
        journal.append(node2)

        summary = journal.generate_summary()

        # 验证 Changelog 中包含 BUGGY 标记
        assert "BUGGY" in summary
        # 验证 Constraints 部分包含失败原因
        assert "Failed" in summary or "memory error" in summary
