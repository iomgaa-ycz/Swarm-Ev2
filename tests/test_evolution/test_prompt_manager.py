"""PromptManager 单元测试。

测试 Prompt 管理器的核心功能：Skill 加载、Agent 配置加载、Prompt 构建、Top-K 注入。
"""

import pytest
from unittest.mock import MagicMock

from utils.prompt_manager import PromptManager
from core.evolution.experience_pool import ExperiencePool, TaskRecord
from core.state import Node


@pytest.fixture
def temp_prompt_system(tmp_path):
    """创建临时的 Prompt 系统目录结构。"""
    # 创建目录
    template_dir = tmp_path / "prompt_templates"
    skills_dir = tmp_path / "skills"
    agent_configs_dir = tmp_path / "agent_configs"

    template_dir.mkdir()
    skills_dir.mkdir()
    agent_configs_dir.mkdir()

    # 创建测试模板（draft.j2）
    (template_dir / "draft.j2").write_text(
        """# Role
{{ load_agent_config(agent_id, 'role') }}

# Task
{{ task_desc }}

{% if parent_node %}
# Previous Code
```python
{{ parent_node.code }}
```

# Execution Output
```
{{ parent_node.term_out }}
```
{% endif %}

# Time: {{ time_str }}
# Steps: {{ steps_remaining }}
"""
    )

    # 创建测试 Skill
    static_dir = skills_dir / "static"
    static_dir.mkdir()
    (static_dir / "output_format.md").write_text(
        "# Output Format\nUse markdown code blocks."
    )
    (static_dir / "workspace_rules.md").write_text(
        "# Workspace\nUse ./input/ for data."
    )

    # 创建测试 Agent 配置
    agent_dir = agent_configs_dir / "agent_0"
    agent_dir.mkdir()
    (agent_dir / "role.md").write_text("# Test Agent\nYou are a test agent.")
    (agent_dir / "strategy_explore.md").write_text(
        "# Explore Strategy\nStart with baselines."
    )

    return {
        "template_dir": template_dir,
        "skills_dir": skills_dir,
        "agent_configs_dir": agent_configs_dir,
    }


@pytest.fixture
def prompt_manager(temp_prompt_system):
    """创建 PromptManager 实例。"""
    return PromptManager(
        template_dir=temp_prompt_system["template_dir"],
        skills_dir=temp_prompt_system["skills_dir"],
        agent_configs_dir=temp_prompt_system["agent_configs_dir"],
    )


class TestPromptManagerInit:
    """测试 PromptManager 初始化。"""

    def test_init_success(self, prompt_manager):
        """测试正常初始化。"""
        assert prompt_manager is not None
        assert prompt_manager.env is not None
        assert "load_skill" in prompt_manager.env.globals
        assert "load_agent_config" in prompt_manager.env.globals

    def test_init_with_invalid_template_dir(self, tmp_path):
        """测试使用无效的模板目录初始化。"""
        # 注意：Jinja2 不会在初始化时抛出异常，只会在加载模板时抛出
        # 这里只验证初始化不崩溃
        manager = PromptManager(
            template_dir=tmp_path / "nonexistent",
            skills_dir=tmp_path / "skills",
            agent_configs_dir=tmp_path / "agent_configs",
        )
        assert manager is not None


class TestLoadSkill:
    """测试静态 Skill 加载。"""

    def test_load_skill_success(self, prompt_manager):
        """测试成功加载 Skill。"""
        content = prompt_manager.load_skill("static/output_format")
        assert "Output Format" in content
        assert "markdown code blocks" in content

    def test_load_skill_with_md_suffix(self, prompt_manager):
        """测试加载 Skill（带 .md 后缀）。"""
        content = prompt_manager.load_skill("static/output_format.md")
        assert "Output Format" in content

    def test_load_skill_not_found(self, prompt_manager):
        """测试加载不存在的 Skill。"""
        with pytest.raises(FileNotFoundError):
            prompt_manager.load_skill("static/nonexistent")


class TestLoadAgentConfig:
    """测试 Agent 配置加载。"""

    def test_load_agent_config_success(self, prompt_manager):
        """测试成功加载 Agent 配置。"""
        content = prompt_manager.load_agent_config("agent_0", "role")
        assert "Test Agent" in content
        assert "test agent" in content

    def test_load_agent_config_with_md_suffix(self, prompt_manager):
        """测试加载 Agent 配置（带 .md 后缀）。"""
        content = prompt_manager.load_agent_config("agent_0", "role.md")
        assert "Test Agent" in content

    def test_load_agent_config_not_found(self, prompt_manager):
        """测试加载不存在的 Agent 配置。"""
        with pytest.raises(FileNotFoundError):
            prompt_manager.load_agent_config("agent_999", "role")

    def test_load_agent_config_invalid_section(self, prompt_manager):
        """测试加载无效的配置节。"""
        with pytest.raises(FileNotFoundError):
            prompt_manager.load_agent_config("agent_0", "nonexistent_section")


class TestInjectTopKSkills:
    """测试 Top-K Skill 注入。"""

    def test_inject_top_k_skills_with_pool(self, prompt_manager):
        """测试从经验池注入 Top-K Skill。"""
        # 创建 Mock ExperiencePool
        pool = MagicMock(spec=ExperiencePool)
        pool.query.return_value = [
            TaskRecord(
                agent_id="agent_0",
                task_type="explore",
                input_hash="hash1",
                output_quality=0.9,
                strategy_summary="Used gradient boosting with cross-validation",
                timestamp=1234567890.0,
            ),
            TaskRecord(
                agent_id="agent_1",
                task_type="explore",
                input_hash="hash2",
                output_quality=0.85,
                strategy_summary="Applied feature engineering and ensembling",
                timestamp=1234567891.0,
            ),
        ]

        result = prompt_manager.inject_top_k_skills(
            task_type="explore",
            k=5,
            experience_pool=pool,
        )

        # 验证调用
        pool.query.assert_called_once_with(
            task_type="explore",
            k=5,
            output_quality=(">", 0.5),
        )

        # 验证输出
        assert "Top-K Success Examples" in result
        assert "gradient boosting" in result
        assert "feature engineering" in result
        assert "0.90" in result  # 检查质量值

    def test_inject_top_k_skills_without_pool(self, prompt_manager):
        """测试无经验池时返回空字符串。"""
        result = prompt_manager.inject_top_k_skills(
            task_type="explore",
            k=5,
            experience_pool=None,
        )
        assert result == ""

    def test_inject_top_k_skills_empty_pool(self, prompt_manager):
        """测试经验池为空时返回空字符串。"""
        pool = MagicMock(spec=ExperiencePool)
        pool.query.return_value = []

        result = prompt_manager.inject_top_k_skills(
            task_type="explore",
            k=5,
            experience_pool=pool,
        )
        assert result == ""


class TestBuildPrompt:
    """测试完整 Prompt 构建。"""

    def test_build_prompt_explore_basic(self, prompt_manager):
        """测试基本 Explore Prompt 构建。"""
        context = {
            "task_desc": "Predict house prices",
            "parent_node": None,
            "memory": "",
            "data_preview": None,
            "time_remaining": 3600,
            "steps_remaining": 10,
        }

        prompt = prompt_manager.build_prompt(
            task_type="explore",
            agent_id="agent_0",
            context=context,
        )

        # 验证 Prompt 包含关键内容
        assert "Test Agent" in prompt  # Agent role
        assert "Predict house prices" in prompt  # Task description
        assert "1 hour" in prompt  # Time remaining
        assert "10" in prompt  # Steps remaining

    def test_build_prompt_explore_with_parent_node(self, prompt_manager):
        """测试包含父节点的 Explore Prompt 构建。"""
        # 创建 Mock Node
        parent_node = MagicMock(spec=Node)
        parent_node.code = "import pandas as pd\nprint('test')"
        parent_node.term_out = "test output"
        parent_node.is_buggy = False
        parent_node.metric_value = 0.85

        context = {
            "task_desc": "Improve the model",
            "parent_node": parent_node,
            "memory": "Previous attempts used XGBoost",
            "data_preview": "train.csv: 1000 rows",
            "time_remaining": 7200,
            "steps_remaining": 20,
        }

        prompt = prompt_manager.build_prompt(
            task_type="explore",
            agent_id="agent_0",
            context=context,
        )

        # 验证 Prompt 包含父节点信息
        assert "import pandas as pd" in prompt
        assert "test output" in prompt
        assert "2 hours" in prompt

    def test_build_prompt_invalid_task_type(self, prompt_manager):
        """测试无效的任务类型。"""
        context = {
            "task_desc": "Test task",
            "parent_node": None,
            "memory": "",
            "data_preview": None,
            "time_remaining": 3600,
            "steps_remaining": 10,
        }

        with pytest.raises(Exception):  # Jinja2 TemplateNotFound
            prompt_manager.build_prompt(
                task_type="invalid_task",
                agent_id="agent_0",
                context=context,
            )


class TestFormatTime:
    """测试时间格式化功能。"""

    def test_format_time_zero(self, prompt_manager):
        """测试格式化零秒。"""
        assert prompt_manager._format_time(0) == "0 seconds"

    def test_format_time_seconds_only(self, prompt_manager):
        """测试仅秒数。"""
        assert prompt_manager._format_time(45) == "45 seconds"

    def test_format_time_minutes_only(self, prompt_manager):
        """测试仅分钟。"""
        assert prompt_manager._format_time(300) == "5 minutes"

    def test_format_time_hours_and_minutes(self, prompt_manager):
        """测试小时和分钟。"""
        result = prompt_manager._format_time(3660)  # 1 hour 1 minute
        assert "1 hour" in result
        assert "1 minute" in result

    def test_format_time_hours_only(self, prompt_manager):
        """测试仅小时。"""
        result = prompt_manager._format_time(7200)  # 2 hours
        assert "2 hours" in result
        assert "minute" not in result

    def test_format_time_complex(self, prompt_manager):
        """测试复杂时间。"""
        result = prompt_manager._format_time(3665)  # 1 hour 1 minute 5 seconds
        assert "1 hour" in result
        assert "1 minute" in result
        # Seconds should not be shown when hours/minutes exist


class TestIntegration:
    """集成测试。"""

    def test_full_workflow(self, prompt_manager):
        """测试完整工作流。"""
        # 创建经验池
        pool = MagicMock(spec=ExperiencePool)
        pool.query.return_value = [
            TaskRecord(
                agent_id="agent_0",
                task_type="explore",
                input_hash="hash1",
                output_quality=0.92,
                strategy_summary="Used ensemble methods",
                timestamp=1234567890.0,
            ),
        ]

        # 创建父节点
        parent_node = MagicMock(spec=Node)
        parent_node.code = "# Previous solution\nmodel = XGBoost()"
        parent_node.term_out = "Validation metric: 0.85"
        parent_node.is_buggy = False
        parent_node.metric_value = 0.85

        # 构建上下文
        context = {
            "task_desc": "Classify customer churn",
            "parent_node": parent_node,
            "memory": "Tried logistic regression, got 0.75",
            "data_preview": "train.csv: 5000 rows, 20 features",
            "time_remaining": 10800,  # 3 hours
            "steps_remaining": 15,
            "experience_pool": pool,
            "top_k": 3,
        }

        # 构建 Prompt
        prompt = prompt_manager.build_prompt(
            task_type="explore",
            agent_id="agent_0",
            context=context,
        )

        # 验证所有组件都存在
        assert "Test Agent" in prompt  # Role
        assert "Classify customer churn" in prompt  # Task
        assert "XGBoost" in prompt  # Parent code
        assert "Validation metric: 0.85" in prompt  # Execution result
        assert "3 hours" in prompt  # Time
        assert "15" in prompt  # Steps


if __name__ == "__main__":
    pytest.main(
        [__file__, "-v", "--cov=utils.prompt_manager", "--cov-report=term-missing"]
    )
