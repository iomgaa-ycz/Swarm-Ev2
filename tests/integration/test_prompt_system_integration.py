"""Phase 3.2 Prompt 体系集成测试。

测试整个 Prompt 系统的安装、配置和功能完整性。

Usage:
    # 使用 pytest 运行（推荐）
    pytest tests/integration/test_prompt_system_integration.py -v

    # 或者直接运行脚本（独立验证）
    python tests/integration/test_prompt_system_integration.py
"""

import pytest
from pathlib import Path
from utils.prompt_manager import PromptManager


class TestPromptSystemIntegration:
    """Prompt 体系集成测试类。"""

    @pytest.fixture(scope="class")
    def base_dir(self):
        """返回 Prompt 系统根目录。"""
        return Path("benchmark/mle-bench")

    @pytest.fixture(scope="class")
    def prompt_manager(self, base_dir):
        """初始化 PromptManager 实例。"""
        return PromptManager(
            template_dir=base_dir / "prompt_templates",
            skills_dir=base_dir / "skills",
            agent_configs_dir=base_dir / "agent_configs",
        )

    def test_file_structure_templates(self, base_dir):
        """测试 Jinja2 模板文件存在。"""
        templates = [
            "prompt_templates/explore.j2",
            "prompt_templates/merge.j2",
            "prompt_templates/mutate.j2",
        ]
        for template in templates:
            assert (base_dir / template).exists(), f"模板文件缺失: {template}"

    def test_file_structure_static_skills(self, base_dir):
        """测试静态 Skill 文件存在。"""
        skills = [
            "skills/static/output_format.md",
            "skills/static/workspace_rules.md",
            "skills/static/ml_best_practices.md",
            "skills/static/code_style.md",
        ]
        for skill in skills:
            assert (base_dir / skill).exists(), f"静态 Skill 缺失: {skill}"

    def test_file_structure_task_skills(self, base_dir):
        """测试任务特定 Skill 文件存在。"""
        skills = [
            "skills/by_task_type/merge/crossover_strategies.md",
            "skills/by_task_type/merge/conflict_resolution.md",
            "skills/by_task_type/mutate/mutation_strategies.md",
            "skills/by_task_type/mutate/local_optimization.md",
        ]
        for skill in skills:
            assert (base_dir / skill).exists(), f"任务 Skill 缺失: {skill}"

    def test_file_structure_agent_configs(self, base_dir):
        """测试 Agent 配置文件存在。"""
        # 检查所有 4 个 Agent 的配置
        for agent_id in ["agent_0", "agent_1", "agent_2", "agent_3"]:
            configs = ["role.md", "strategy_explore.md", "strategy_merge.md", "strategy_mutate.md"]
            for config in configs:
                path = base_dir / "agent_configs" / agent_id / config
                assert path.exists(), f"Agent 配置缺失: {agent_id}/{config}"

    def test_file_structure_metadata(self, base_dir):
        """测试 Skill 元数据文件存在。"""
        metadata = [
            "skills/meta/skill_index.json",
            "skills/meta/skill_lineage.json",
            "skills/meta/update_history.json",
        ]
        for meta in metadata:
            assert (base_dir / meta).exists(), f"元数据文件缺失: {meta}"

    def test_load_static_skill(self, prompt_manager):
        """测试加载静态 Skill。"""
        skill = prompt_manager.load_skill("static/output_format")
        assert "Response Format" in skill, "静态 Skill 内容不正确"

    def test_load_agent_config(self, prompt_manager):
        """测试加载 Agent 配置。"""
        role = prompt_manager.load_agent_config("agent_0", "role")
        assert "Agent Role" in role, "Agent 配置内容不正确"

    def test_build_explore_prompt_basic(self, prompt_manager):
        """测试构建基本 Explore Prompt。"""
        context = {
            "task_desc": "Predict customer churn using historical data",
            "parent_node": None,
            "memory": "",
            "data_preview": "train.csv: 10000 rows, 25 features",
            "time_remaining": 3600,
            "steps_remaining": 20,
        }

        prompt = prompt_manager.build_prompt("explore", "agent_0", context)

        assert "Predict customer churn" in prompt, "任务描述缺失"
        assert "1 hour" in prompt, "时间格式化错误"
        assert "20" in prompt, "步数缺失"

    def test_build_explore_prompt_with_parent(self, prompt_manager):
        """测试构建带父节点的 Explore Prompt。"""
        # 创建 Mock Node
        class MockNode:
            def __init__(self):
                self.code = "import pandas as pd\nmodel = XGBoost()"
                self.term_out = "Validation metric: 0.85"
                self.is_buggy = False
                self.metric_value = 0.85

        parent_node = MockNode()
        context = {
            "task_desc": "Improve the XGBoost model",
            "parent_node": parent_node,
            "memory": "Previous best: 0.82",
            "data_preview": None,
            "time_remaining": 7200,
            "steps_remaining": 15,
        }

        prompt = prompt_manager.build_prompt("explore", "agent_1", context)

        assert "XGBoost" in prompt, "父节点代码缺失"
        assert "0.85" in prompt or "Validation metric" in prompt, "执行结果缺失"
        assert "2 hours" in prompt, "时间格式化错误"

    def test_all_agents_have_configs(self, base_dir):
        """测试所有 4 个 Agent 都有完整配置。"""
        for agent_id in ["agent_0", "agent_1", "agent_2", "agent_3"]:
            for section in ["role", "strategy_explore", "strategy_merge", "strategy_mutate"]:
                path = base_dir / "agent_configs" / agent_id / f"{section}.md"
                assert path.exists(), f"{agent_id} 缺少 {section} 配置"

                # 验证文件不为空
                content = path.read_text()
                assert len(content) > 0, f"{agent_id}/{section} 文件为空"


def main():
    """主函数（用于独立运行）。"""
    print("\n" + "=" * 60)
    print("Phase 3.2 Prompt 体系集成测试")
    print("=" * 60 + "\n")

    # 使用 pytest 运行测试
    exit_code = pytest.main([__file__, "-v", "--tb=short"])

    if exit_code == 0:
        print("\n" + "=" * 60)
        print("✅ 所有集成测试通过！")
        print("=" * 60)
        print("\nPhase 3.2 Prompt 体系验证完成，可以进入下一阶段开发。")
    else:
        print("\n" + "=" * 60)
        print("❌ 某些集成测试失败！")
        print("=" * 60)

    return exit_code


if __name__ == "__main__":
    import sys

    sys.exit(main())
