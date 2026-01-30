"""配置系统单元测试。"""

import pytest
from pathlib import Path
from omegaconf import OmegaConf
import tempfile
import os
import re


from utils.config import (
    load_config,
    validate_config,
    generate_exp_name,
    setup_workspace,
    Config,
    ProjectConfig,
    DataConfig,
    LLMConfig,
    LLMStageConfig,
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
  workspace_dir: "{tmp_path / 'workspace'}"
  log_dir: "{tmp_path / 'logs'}"
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
    model: "gpt-4-turbo"
    temperature: 0.5
    api_key: "test-key-123"
  feedback:
    model: "gpt-4-turbo"
    temperature: 0.5
    api_key: "test-key-456"

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
  workspace_dir: "{tmp_path / 'workspace'}"
  log_dir: "{tmp_path / 'logs'}"
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
    model: "gpt-3.5-turbo"
    temperature: 0.5
    api_key: "original-key"
  feedback:
    model: "gpt-3.5-turbo"
    temperature: 0.5
    api_key: "original-key"

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
        cli_cfg = OmegaConf.from_dotlist([
            "project.name=TestProject",
            "llm.code.model=gpt-4-turbo"
        ])

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
                "exp_name": None
            },
            "data": {
                "data_dir": None,  # 缺少必填字段
                "desc_file": None,
                "goal": "Test",
                "eval": None,
                "preprocess_data": True,
                "copy_data": False
            },
            "llm": {
                "code": {"model": "gpt-4", "temperature": 0.5, "api_key": "test"},
                "feedback": {"model": "gpt-4", "temperature": 0.5, "api_key": "test"}
            },
            "execution": {"timeout": 3600, "agent_file_name": "run.py", "format_tb_ipython": False},
            "agent": {
                "max_steps": 50,
                "time_limit": 86400,
                "k_fold_validation": 5,
                "expose_prediction": False,
                "data_preview": True,
                "convert_system_to_user": False
            },
            "search": {
                "strategy": "mcts",
                "max_debug_depth": 3,
                "debug_prob": 0.5,
                "num_drafts": 5,
                "parallel_num": 3
            },
            "logging": {"level": "INFO", "console_output": True, "file_output": True}
        }

        cfg = OmegaConf.create(config_dict)

        # 验证抛出 ValueError
        with pytest.raises(ValueError, match="`data.data_dir` 必须提供"):
            validate_config(cfg)

    def test_validate_config_missing_goal_and_desc(self, tmp_path: Path) -> None:
        """测试缺少 goal 和 desc_file 时抛出异常。"""
        test_data_dir = tmp_path / "test_data"
        test_data_dir.mkdir()

        config_dict = {
            "project": {
                "name": "Test",
                "version": "0.1.0",
                "workspace_dir": str(tmp_path / "workspace"),
                "log_dir": str(tmp_path / "logs"),
                "exp_name": None
            },
            "data": {
                "data_dir": str(test_data_dir),
                "desc_file": None,
                "goal": None,  # 缺少 goal 和 desc_file
                "eval": None,
                "preprocess_data": True,
                "copy_data": False
            },
            "llm": {
                "code": {"model": "gpt-4", "temperature": 0.5, "api_key": "test"},
                "feedback": {"model": "gpt-4", "temperature": 0.5, "api_key": "test"}
            },
            "execution": {"timeout": 3600, "agent_file_name": "run.py", "format_tb_ipython": False},
            "agent": {
                "max_steps": 50,
                "time_limit": 86400,
                "k_fold_validation": 5,
                "expose_prediction": False,
                "data_preview": True,
                "convert_system_to_user": False
            },
            "search": {
                "strategy": "mcts",
                "max_debug_depth": 3,
                "debug_prob": 0.5,
                "num_drafts": 5,
                "parallel_num": 3
            },
            "logging": {"level": "INFO", "console_output": True, "file_output": True}
        }

        cfg = OmegaConf.create(config_dict)

        # 验证抛出 ValueError
        with pytest.raises(ValueError, match="必须提供 `data.desc_file` 或 `data.goal` 之一"):
            validate_config(cfg)


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
                "exp_name": "test_exp"
            },
            "data": {
                "data_dir": str(test_data_dir),
                "desc_file": None,
                "goal": "Test",
                "eval": None,
                "preprocess_data": True,
                "copy_data": False
            },
            "llm": {
                "code": {"model": "gpt-4", "temperature": 0.5, "api_key": "test"},
                "feedback": {"model": "gpt-4", "temperature": 0.5, "api_key": "test"}
            },
            "execution": {"timeout": 3600, "agent_file_name": "run.py", "format_tb_ipython": False},
            "agent": {
                "max_steps": 50,
                "time_limit": 86400,
                "k_fold_validation": 5,
                "expose_prediction": False,
                "data_preview": True,
                "convert_system_to_user": False
            },
            "search": {
                "strategy": "mcts",
                "max_debug_depth": 3,
                "debug_prob": 0.5,
                "num_drafts": 5,
                "parallel_num": 3
            },
            "logging": {"level": "INFO", "console_output": True, "file_output": True}
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
                "exp_name": "test_exp_1"
            },
            "data": {
                "data_dir": str(test_data_dir),
                "desc_file": None,
                "goal": "Test",
                "eval": None,
                "preprocess_data": True,
                "copy_data": False
            },
            "llm": {
                "code": {"model": "gpt-4", "temperature": 0.5, "api_key": "test"},
                "feedback": {"model": "gpt-4", "temperature": 0.5, "api_key": "test"}
            },
            "execution": {"timeout": 3600, "agent_file_name": "run.py", "format_tb_ipython": False},
            "agent": {
                "max_steps": 50,
                "time_limit": 86400,
                "k_fold_validation": 5,
                "expose_prediction": False,
                "data_preview": True,
                "convert_system_to_user": False
            },
            "search": {
                "strategy": "mcts",
                "max_debug_depth": 3,
                "debug_prob": 0.5,
                "num_drafts": 5,
                "parallel_num": 3
            },
            "logging": {"level": "INFO", "console_output": True, "file_output": True}
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
                "exp_name": "test_exp_2"
            },
            "data": {
                "data_dir": str(test_data_dir),
                "desc_file": None,
                "goal": "Test",
                "eval": None,
                "preprocess_data": True,
                "copy_data": False
            },
            "llm": {
                "code": {"model": "gpt-4", "temperature": 0.5, "api_key": "test"},
                "feedback": {"model": "gpt-4", "temperature": 0.5, "api_key": "test"}
            },
            "execution": {"timeout": 3600, "agent_file_name": "run.py", "format_tb_ipython": False},
            "agent": {
                "max_steps": 50,
                "time_limit": 86400,
                "k_fold_validation": 5,
                "expose_prediction": False,
                "data_preview": True,
                "convert_system_to_user": False
            },
            "search": {
                "strategy": "mcts",
                "max_debug_depth": 3,
                "debug_prob": 0.5,
                "num_drafts": 5,
                "parallel_num": 3
            },
            "logging": {"level": "INFO", "console_output": True, "file_output": True}
        }

        cfg = validate_config(OmegaConf.create(config_dict))

        # 测试可用作 set 成员
        test_set = {cfg}
        assert cfg in test_set
