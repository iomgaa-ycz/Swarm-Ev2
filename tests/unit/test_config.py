"""配置系统单元测试。"""

import pytest
from pathlib import Path
from omegaconf import OmegaConf
import re


from utils.config import (
    load_config,
    validate_config,
    generate_exp_name,
    setup_workspace,
)


class TestLoadConfig:
    """配置加载测试类。"""

    def test_load_config_default(self, tmp_path: Path) -> None:
        """测试加载默认配置文件。"""
        # 创建测试数据目录
        test_data_dir = tmp_path / "test_data"
        test_data_dir.mkdir()
        (test_data_dir / "sample.csv").write_text("id,value\n1,100\n")

        # 创建测试配置文件
        config_content = f"""
project:
  name: "Swarm-Ev2"
  version: "0.1.0"
  workspace_dir: "{tmp_path / "workspace"}"
  log_dir: "{tmp_path / "logs"}"
  exp_name: null

data:
  data_dir: "{test_data_dir}"
  desc_file: null
  goal: "Test goal"
  eval: null
  preprocess_data: true
  copy_data: false

llm:
  code:
    provider: "openai"
    model: "gpt-4-turbo"
    temperature: 0.5
    api_key: "test-key-123"
    base_url: "https://api.openai.com/v1"
    max_tokens: null
  feedback:
    provider: "openai"
    model: "gpt-4-turbo"
    temperature: 0.5
    api_key: "test-key-456"
    base_url: "https://api.openai.com/v1"
    max_tokens: null

execution:
  timeout: 3600
  agent_file_name: "runfile.py"
  format_tb_ipython: false

agent:
  max_steps: 50
  time_limit: 86400
  k_fold_validation: 5
  expose_prediction: false
  data_preview: true
  convert_system_to_user: false

search:
  strategy: "mcts"
  max_debug_depth: 3
  debug_prob: 0.5
  num_drafts: 5
  parallel_num: 3

logging:
  level: "INFO"
  console_output: true
  file_output: true
"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(config_content)

        # 加载配置
        cfg = load_config(config_path=config_path, use_cli=False)

        # 验证加载成功
        assert cfg.project.name == "Swarm-Ev2"
        assert cfg.llm.code.model == "gpt-4-turbo"
        assert cfg.project.exp_name is not None  # 自动生成
        assert cfg.data.data_dir == test_data_dir

    def test_load_config_with_cli(self, tmp_path: Path) -> None:
        """测试 CLI 参数覆盖配置。"""
        # 创建测试数据目录
        test_data_dir = tmp_path / "test_data"
        test_data_dir.mkdir()

        # 创建基础配置
        config_content = f"""
project:
  name: "Original"
  version: "0.1.0"
  workspace_dir: "{tmp_path / "workspace"}"
  log_dir: "{tmp_path / "logs"}"
  exp_name: null

data:
  data_dir: "{test_data_dir}"
  desc_file: null
  goal: "Test"
  eval: null
  preprocess_data: true
  copy_data: false

llm:
  code:
    provider: "openai"
    model: "gpt-3.5-turbo"
    temperature: 0.5
    api_key: "original-key"
    base_url: "https://api.openai.com/v1"
    max_tokens: null
  feedback:
    provider: "openai"
    model: "gpt-3.5-turbo"
    temperature: 0.5
    api_key: "original-key"
    base_url: "https://api.openai.com/v1"
    max_tokens: null

execution:
  timeout: 3600
  agent_file_name: "runfile.py"
  format_tb_ipython: false

agent:
  max_steps: 50
  time_limit: 86400
  k_fold_validation: 5
  expose_prediction: false
  data_preview: true
  convert_system_to_user: false

search:
  strategy: "mcts"
  max_debug_depth: 3
  debug_prob: 0.5
  num_drafts: 5
  parallel_num: 3

logging:
  level: "INFO"
  console_output: true
  file_output: true
"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(config_content)

        # 模拟 CLI 参数
        cli_cfg = OmegaConf.from_dotlist(
            ["project.name=TestProject", "llm.code.model=gpt-4-turbo"]
        )

        # 加载并合并配置
        base_cfg = OmegaConf.load(config_path)
        merged_cfg = OmegaConf.merge(base_cfg, cli_cfg)
        cfg = validate_config(merged_cfg)

        # 验证 CLI 参数覆盖成功
        assert cfg.project.name == "TestProject"
        assert cfg.llm.code.model == "gpt-4-turbo"


class TestValidateConfig:
    """配置验证测试类。"""

    def test_validate_config_missing_data_dir(self, tmp_path: Path) -> None:
        """测试缺少必填字段时抛出异常。"""
        config_dict = {
            "project": {
                "name": "Test",
                "version": "0.1.0",
                "workspace_dir": str(tmp_path / "workspace"),
                "log_dir": str(tmp_path / "logs"),
                "exp_name": None,
            },
            "data": {
                "data_dir": None,  # 缺少必填字段
                "desc_file": None,
                "goal": "Test",
                "eval": None,
                "preprocess_data": True,
                "copy_data": False,
            },
            "llm": {
                "code": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "temperature": 0.5,
                    "api_key": "test",
                    "base_url": "https://api.openai.com/v1",
                    "max_tokens": None,
                },
                "feedback": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "temperature": 0.5,
                    "api_key": "test",
                    "base_url": "https://api.openai.com/v1",
                    "max_tokens": None,
                },
            },
            "execution": {
                "timeout": 3600,
                "agent_file_name": "run.py",
                "format_tb_ipython": False,
            },
            "agent": {
                "max_steps": 50,
                "time_limit": 86400,
                "k_fold_validation": 5,
                "expose_prediction": False,
                "data_preview": True,
                "convert_system_to_user": False,
            },
            "search": {
                "strategy": "mcts",
                "max_debug_depth": 3,
                "debug_prob": 0.5,
                "num_drafts": 5,
                "parallel_num": 3,
            },
            "logging": {"level": "INFO", "console_output": True, "file_output": True},
        }

        cfg = OmegaConf.create(config_dict)

        # 验证抛出 ValueError
        with pytest.raises(ValueError, match="`data.data_dir` 必须提供"):
            validate_config(cfg)

    def test_validate_config_missing_goal_and_desc(self, tmp_path: Path) -> None:
        """测试缺少 goal 和 desc_file 时仅警告不抛异常。"""
        test_data_dir = tmp_path / "test_data"
        test_data_dir.mkdir()

        config_dict = {
            "project": {
                "name": "Test",
                "version": "0.1.0",
                "workspace_dir": str(tmp_path / "workspace"),
                "log_dir": str(tmp_path / "logs"),
                "exp_name": None,
            },
            "data": {
                "data_dir": str(test_data_dir),
                "desc_file": None,
                "goal": None,  # 缺少 goal 和 desc_file
                "eval": None,
                "preprocess_data": True,
                "copy_data": False,
            },
            "llm": {
                "code": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "temperature": 0.5,
                    "api_key": "test",
                    "base_url": "https://api.openai.com/v1",
                    "max_tokens": None,
                },
                "feedback": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "temperature": 0.5,
                    "api_key": "test",
                    "base_url": "https://api.openai.com/v1",
                    "max_tokens": None,
                },
            },
            "execution": {
                "timeout": 3600,
                "agent_file_name": "run.py",
                "format_tb_ipython": False,
            },
            "agent": {
                "max_steps": 50,
                "time_limit": 86400,
                "k_fold_validation": 5,
                "expose_prediction": False,
                "data_preview": True,
                "convert_system_to_user": False,
            },
            "search": {
                "strategy": "mcts",
                "max_debug_depth": 3,
                "debug_prob": 0.5,
                "num_drafts": 5,
                "parallel_num": 3,
            },
            "logging": {"level": "INFO", "console_output": True, "file_output": True},
        }

        cfg = OmegaConf.create(config_dict)

        # 代码实际仅 log_msg("WARNING")，不抛异常，验证正常返回
        result = validate_config(cfg)
        assert result is not None
        assert result.data.goal is None
        assert result.data.desc_file is None


class TestGenerateExpName:
    """实验名称生成测试类。"""

    def test_generate_exp_name_format(self) -> None:
        """测试实验名称生成格式。"""
        exp_name = generate_exp_name()

        # 验证格式: YYYYMMDD_HHMMSS_xxxx
        pattern = r"^\d{8}_\d{6}_[a-z]{4}$"
        assert re.match(pattern, exp_name), f"实验名称格式不正确: {exp_name}"

    def test_generate_exp_name_uniqueness(self) -> None:
        """测试多次调用生成不同名称。"""
        names = [generate_exp_name() for _ in range(10)]

        # 验证所有名称唯一（由于时间戳和随机后缀，应该不重复）
        assert len(set(names)) == len(names), "生成的实验名称存在重复"


class TestSetupWorkspace:
    """工作空间初始化测试类。"""

    def test_setup_workspace_creates_directories(self, tmp_path: Path) -> None:
        """测试工作空间目录创建。"""
        # 创建测试数据
        test_data_dir = tmp_path / "test_data"
        test_data_dir.mkdir()
        (test_data_dir / "sample.csv").write_text("id,value\n1,100\n")

        # 创建配置
        config_dict = {
            "project": {
                "name": "Test",
                "version": "0.1.0",
                "workspace_dir": str(tmp_path / "workspace"),
                "log_dir": str(tmp_path / "logs"),
                "exp_name": "test_exp",
            },
            "data": {
                "data_dir": str(test_data_dir),
                "desc_file": None,
                "goal": "Test",
                "eval": None,
                "preprocess_data": True,
                "copy_data": False,
            },
            "llm": {
                "code": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "temperature": 0.5,
                    "api_key": "test",
                    "base_url": "https://api.openai.com/v1",
                    "max_tokens": None,
                },
                "feedback": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "temperature": 0.5,
                    "api_key": "test",
                    "base_url": "https://api.openai.com/v1",
                    "max_tokens": None,
                },
            },
            "execution": {
                "timeout": 3600,
                "agent_file_name": "run.py",
                "format_tb_ipython": False,
            },
            "agent": {
                "max_steps": 50,
                "time_limit": 86400,
                "k_fold_validation": 5,
                "expose_prediction": False,
                "data_preview": True,
                "convert_system_to_user": False,
            },
            "search": {
                "strategy": "mcts",
                "max_debug_depth": 3,
                "debug_prob": 0.5,
                "num_drafts": 5,
                "parallel_num": 3,
            },
            "logging": {"level": "INFO", "console_output": True, "file_output": True},
        }

        cfg_omega = OmegaConf.create(config_dict)
        cfg = validate_config(cfg_omega)

        # 初始化工作空间
        setup_workspace(cfg)

        # 验证目录创建成功
        workspace_dir = cfg.project.workspace_dir
        assert (workspace_dir / "input").exists()
        assert (workspace_dir / "working").exists()
        assert (workspace_dir / "submission").exists()


class TestConfigHashable:
    """Config 可哈希性测试类。"""

    def test_config_as_dict_key(self, tmp_path: Path) -> None:
        """测试 Config 对象可用作 dict key。"""
        test_data_dir = tmp_path / "test_data"
        test_data_dir.mkdir()

        config_dict = {
            "project": {
                "name": "Test",
                "version": "0.1.0",
                "workspace_dir": str(tmp_path / "workspace"),
                "log_dir": str(tmp_path / "logs"),
                "exp_name": "test_exp_1",
            },
            "data": {
                "data_dir": str(test_data_dir),
                "desc_file": None,
                "goal": "Test",
                "eval": None,
                "preprocess_data": True,
                "copy_data": False,
            },
            "llm": {
                "code": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "temperature": 0.5,
                    "api_key": "test",
                    "base_url": "https://api.openai.com/v1",
                    "max_tokens": None,
                },
                "feedback": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "temperature": 0.5,
                    "api_key": "test",
                    "base_url": "https://api.openai.com/v1",
                    "max_tokens": None,
                },
            },
            "execution": {
                "timeout": 3600,
                "agent_file_name": "run.py",
                "format_tb_ipython": False,
            },
            "agent": {
                "max_steps": 50,
                "time_limit": 86400,
                "k_fold_validation": 5,
                "expose_prediction": False,
                "data_preview": True,
                "convert_system_to_user": False,
            },
            "search": {
                "strategy": "mcts",
                "max_debug_depth": 3,
                "debug_prob": 0.5,
                "num_drafts": 5,
                "parallel_num": 3,
            },
            "logging": {"level": "INFO", "console_output": True, "file_output": True},
        }

        cfg = validate_config(OmegaConf.create(config_dict))

        # 测试可用作 dict key
        test_dict = {cfg: "test_value"}
        assert test_dict[cfg] == "test_value"

    def test_config_in_set(self, tmp_path: Path) -> None:
        """测试 Config 对象可用作 set 成员。"""
        test_data_dir = tmp_path / "test_data"
        test_data_dir.mkdir()

        config_dict = {
            "project": {
                "name": "Test",
                "version": "0.1.0",
                "workspace_dir": str(tmp_path / "workspace"),
                "log_dir": str(tmp_path / "logs"),
                "exp_name": "test_exp_2",
            },
            "data": {
                "data_dir": str(test_data_dir),
                "desc_file": None,
                "goal": "Test",
                "eval": None,
                "preprocess_data": True,
                "copy_data": False,
            },
            "llm": {
                "code": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "temperature": 0.5,
                    "api_key": "test",
                    "base_url": "https://api.openai.com/v1",
                    "max_tokens": None,
                },
                "feedback": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "temperature": 0.5,
                    "api_key": "test",
                    "base_url": "https://api.openai.com/v1",
                    "max_tokens": None,
                },
            },
            "execution": {
                "timeout": 3600,
                "agent_file_name": "run.py",
                "format_tb_ipython": False,
            },
            "agent": {
                "max_steps": 50,
                "time_limit": 86400,
                "k_fold_validation": 5,
                "expose_prediction": False,
                "data_preview": True,
                "convert_system_to_user": False,
            },
            "search": {
                "strategy": "mcts",
                "max_debug_depth": 3,
                "debug_prob": 0.5,
                "num_drafts": 5,
                "parallel_num": 3,
            },
            "logging": {"level": "INFO", "console_output": True, "file_output": True},
        }

        cfg = validate_config(OmegaConf.create(config_dict))

        # 测试可用作 set 成员
        test_set = {cfg}
        assert cfg in test_set


class TestLLMProviderValidation:
    """LLM Provider 配置验证测试类。"""

    def test_llm_config_with_provider_fields(self, tmp_path: Path) -> None:
        """测试 LLMStageConfig 包含 provider、base_url、max_tokens 字段。"""
        test_data_dir = tmp_path / "test_data"
        test_data_dir.mkdir()

        config_dict = {
            "project": {
                "name": "Test",
                "version": "0.1.0",
                "workspace_dir": str(tmp_path / "workspace"),
                "log_dir": str(tmp_path / "logs"),
                "exp_name": "test_provider",
            },
            "data": {
                "data_dir": str(test_data_dir),
                "desc_file": None,
                "goal": "Test provider fields",
                "eval": None,
                "preprocess_data": True,
                "copy_data": False,
            },
            "llm": {
                "code": {
                    "provider": "openai",
                    "model": "gpt-4-turbo",
                    "temperature": 0.5,
                    "api_key": "test",
                    "base_url": "https://api.openai.com/v1",
                    "max_tokens": 4096,
                },
                "feedback": {
                    "provider": "anthropic",
                    "model": "claude-3-opus",
                    "temperature": 0.7,
                    "api_key": "test",
                    "base_url": "https://api.anthropic.com/v1",
                    "max_tokens": 2048,
                },
            },
            "execution": {
                "timeout": 3600,
                "agent_file_name": "run.py",
                "format_tb_ipython": False,
            },
            "agent": {
                "max_steps": 50,
                "time_limit": 86400,
                "k_fold_validation": 5,
                "expose_prediction": False,
                "data_preview": True,
                "convert_system_to_user": False,
            },
            "search": {
                "strategy": "mcts",
                "max_debug_depth": 3,
                "debug_prob": 0.5,
                "num_drafts": 5,
                "parallel_num": 3,
            },
            "logging": {"level": "INFO", "console_output": True, "file_output": True},
        }

        cfg = validate_config(OmegaConf.create(config_dict))

        # 验证 provider 字段
        assert cfg.llm.code.provider == "openai"
        assert cfg.llm.feedback.provider == "anthropic"

        # 验证 base_url 字段
        assert cfg.llm.code.base_url == "https://api.openai.com/v1"
        assert cfg.llm.feedback.base_url == "https://api.anthropic.com/v1"

        # 验证 max_tokens 字段
        assert cfg.llm.code.max_tokens == 4096
        assert cfg.llm.feedback.max_tokens == 2048

    def test_validate_invalid_provider(self, tmp_path: Path) -> None:
        """测试无效的 provider 值（应抛出 ValueError）。"""
        test_data_dir = tmp_path / "test_data"
        test_data_dir.mkdir()

        config_dict = {
            "project": {
                "name": "Test",
                "version": "0.1.0",
                "workspace_dir": str(tmp_path / "workspace"),
                "log_dir": str(tmp_path / "logs"),
                "exp_name": "test_invalid_provider",
            },
            "data": {
                "data_dir": str(test_data_dir),
                "desc_file": None,
                "goal": "Test invalid provider",
                "eval": None,
                "preprocess_data": True,
                "copy_data": False,
            },
            "llm": {
                "code": {
                    "provider": "invalid_provider",  # 无效的 provider
                    "model": "gpt-4",
                    "temperature": 0.5,
                    "api_key": "test",
                    "base_url": "https://api.openai.com/v1",
                    "max_tokens": None,
                },
                "feedback": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "temperature": 0.5,
                    "api_key": "test",
                    "base_url": "https://api.openai.com/v1",
                    "max_tokens": None,
                },
            },
            "execution": {
                "timeout": 3600,
                "agent_file_name": "run.py",
                "format_tb_ipython": False,
            },
            "agent": {
                "max_steps": 50,
                "time_limit": 86400,
                "k_fold_validation": 5,
                "expose_prediction": False,
                "data_preview": True,
                "convert_system_to_user": False,
            },
            "search": {
                "strategy": "mcts",
                "max_debug_depth": 3,
                "debug_prob": 0.5,
                "num_drafts": 5,
                "parallel_num": 3,
            },
            "logging": {"level": "INFO", "console_output": True, "file_output": True},
        }

        cfg = OmegaConf.create(config_dict)

        # 验证抛出 ValueError
        with pytest.raises(ValueError, match="无效的 provider"):
            validate_config(cfg)

    def test_default_base_url(self, tmp_path: Path) -> None:
        """测试 base_url 默认值。"""
        test_data_dir = tmp_path / "test_data"
        test_data_dir.mkdir()

        config_dict = {
            "project": {
                "name": "Test",
                "version": "0.1.0",
                "workspace_dir": str(tmp_path / "workspace"),
                "log_dir": str(tmp_path / "logs"),
                "exp_name": "test_default_base_url",
            },
            "data": {
                "data_dir": str(test_data_dir),
                "desc_file": None,
                "goal": "Test default base_url",
                "eval": None,
                "preprocess_data": True,
                "copy_data": False,
            },
            "llm": {
                "code": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "temperature": 0.5,
                    "api_key": "test",
                    # 不提供 base_url，应使用默认值
                },
                "feedback": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "temperature": 0.5,
                    "api_key": "test",
                },
            },
            "execution": {
                "timeout": 3600,
                "agent_file_name": "run.py",
                "format_tb_ipython": False,
            },
            "agent": {
                "max_steps": 50,
                "time_limit": 86400,
                "k_fold_validation": 5,
                "expose_prediction": False,
                "data_preview": True,
                "convert_system_to_user": False,
            },
            "search": {
                "strategy": "mcts",
                "max_debug_depth": 3,
                "debug_prob": 0.5,
                "num_drafts": 5,
                "parallel_num": 3,
            },
            "logging": {"level": "INFO", "console_output": True, "file_output": True},
        }

        cfg = validate_config(OmegaConf.create(config_dict))

        # 验证默认 base_url
        assert cfg.llm.code.base_url == "https://api.openai.com/v1"
        assert cfg.llm.feedback.base_url == "https://api.openai.com/v1"


def _make_config_dict(tmp_path: Path, **overrides) -> dict:
    """创建最小有效配置字典的辅助函数。"""
    data_dir = overrides.pop("data_dir", str(tmp_path / "test_data"))
    desc_file = overrides.pop("desc_file", None)
    goal = overrides.pop("goal", "Test goal")
    code_provider = overrides.pop("code_provider", "openai")
    feedback_provider = overrides.pop("feedback_provider", "openai")
    api_key = overrides.pop("api_key", "test-key")

    return {
        "project": {
            "name": "Test",
            "version": "0.1.0",
            "workspace_dir": str(tmp_path / "workspace"),
            "log_dir": str(tmp_path / "logs"),
            "exp_name": None,
        },
        "data": {
            "data_dir": data_dir,
            "desc_file": desc_file,
            "goal": goal,
            "eval": None,
            "preprocess_data": True,
            "copy_data": False,
        },
        "llm": {
            "code": {
                "provider": code_provider,
                "model": "gpt-4",
                "temperature": 0.5,
                "api_key": api_key,
                "base_url": "https://api.openai.com/v1",
                "max_tokens": None,
            },
            "feedback": {
                "provider": feedback_provider,
                "model": "gpt-4",
                "temperature": 0.5,
                "api_key": api_key,
                "base_url": "https://api.openai.com/v1",
                "max_tokens": None,
            },
        },
        "execution": {
            "timeout": 3600,
            "agent_file_name": "run.py",
            "format_tb_ipython": False,
        },
        "agent": {
            "max_steps": 50,
            "time_limit": 86400,
            "k_fold_validation": 5,
            "expose_prediction": False,
            "data_preview": True,
            "convert_system_to_user": False,
        },
        "search": {
            "strategy": "mcts",
            "max_debug_depth": 3,
            "debug_prob": 0.5,
            "num_drafts": 5,
            "parallel_num": 3,
        },
        "logging": {"level": "INFO", "console_output": True, "file_output": True},
    }


class TestValidateConfigExtended:
    """validate_config 额外路径测试。"""

    def test_data_dir_not_exists(self, tmp_path):
        """数据目录不存在时抛出 ValueError。"""
        cfg_dict = _make_config_dict(tmp_path, data_dir=str(tmp_path / "nonexistent"))
        cfg = OmegaConf.create(cfg_dict)

        with pytest.raises(ValueError, match="数据目录不存在"):
            validate_config(cfg)

    def test_desc_file_not_exists(self, tmp_path):
        """描述文件不存在时抛出 ValueError。"""
        data_dir = tmp_path / "test_data"
        data_dir.mkdir()

        cfg_dict = _make_config_dict(
            tmp_path,
            data_dir=str(data_dir),
            desc_file=str(tmp_path / "nonexistent.md"),
        )
        cfg = OmegaConf.create(cfg_dict)

        with pytest.raises(ValueError, match="描述文件不存在"):
            validate_config(cfg)

    def test_feedback_provider_invalid(self, tmp_path):
        """feedback provider 无效时抛出 ValueError。"""
        data_dir = tmp_path / "test_data"
        data_dir.mkdir()

        cfg_dict = _make_config_dict(
            tmp_path,
            data_dir=str(data_dir),
            feedback_provider="invalid_fb",
        )
        cfg = OmegaConf.create(cfg_dict)

        with pytest.raises(ValueError, match="无效的 provider"):
            validate_config(cfg)

    def test_auto_detect_desc_file(self, tmp_path):
        """自动检测 data_dir/description.md。"""
        data_dir = tmp_path / "test_data"
        data_dir.mkdir()
        desc = data_dir / "description.md"
        desc.write_text("# Test Description", encoding="utf-8")

        cfg_dict = _make_config_dict(
            tmp_path, data_dir=str(data_dir), desc_file=None, goal=None
        )
        cfg = OmegaConf.create(cfg_dict)

        result = validate_config(cfg)
        assert result.data.desc_file == desc.resolve()

    def test_api_key_unresolved_warning(self, tmp_path):
        """API Key 未解析时不抛异常（仅警告）。"""
        data_dir = tmp_path / "test_data"
        data_dir.mkdir()

        cfg_dict = _make_config_dict(
            tmp_path, data_dir=str(data_dir), api_key="${env:MISSING_KEY}"
        )
        cfg = OmegaConf.create(cfg_dict)

        # 不应抛异常
        result = validate_config(cfg)
        assert result is not None


class TestLoadConfigExtended:
    """load_config 额外路径测试。"""

    def test_config_file_not_found(self, tmp_path):
        """配置文件不存在时抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError, match="配置文件不存在"):
            load_config(config_path=tmp_path / "nonexistent.yaml", use_cli=False)

    def test_env_file_missing(self, tmp_path):
        """不存在 .env 文件时正常运行。"""
        data_dir = tmp_path / "test_data"
        data_dir.mkdir()

        cfg_dict = _make_config_dict(tmp_path, data_dir=str(data_dir))
        config_path = tmp_path / "config.yaml"

        from omegaconf import OmegaConf as OC
        config_path.write_text(OC.to_yaml(OC.create(cfg_dict)), encoding="utf-8")

        # env_file 指向不存在的路径
        cfg = load_config(
            config_path=config_path,
            env_file=tmp_path / ".env.nonexistent",
            use_cli=False,
        )
        assert cfg is not None
