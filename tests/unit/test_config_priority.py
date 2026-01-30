"""配置优先级测试。

测试配置优先级: CLI 参数 > 环境变量 > YAML 配置
"""

import pytest
from pathlib import Path
import os

from utils.config import load_config


class TestConfigPriority:
    """配置优先级测试类。"""

    def test_load_config_with_env_file(self, tmp_path: Path) -> None:
        """测试从 .env 文件加载环境变量。"""
        # 创建测试数据目录
        test_data_dir = tmp_path / "test_data"
        test_data_dir.mkdir()

        # 创建 .env 文件
        env_content = """OPENAI_API_KEY=sk-env-file-key-123
ANTHROPIC_API_KEY=sk-ant-env-key-456
"""
        env_file = tmp_path / ".env"
        env_file.write_text(env_content)

        # 创建配置文件（使用环境变量插值）
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
  goal: "Test"
  eval: null
  preprocess_data: true
  copy_data: false

llm:
  code:
    provider: "openai"
    model: "gpt-4-turbo"
    temperature: 0.5
    api_key: ${{env:OPENAI_API_KEY}}
    base_url: "https://api.openai.com/v1"
    max_tokens: null
  feedback:
    provider: "openai"
    model: "gpt-4-turbo"
    temperature: 0.5
    api_key: ${{env:ANTHROPIC_API_KEY}}
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

        # 加载配置（指定 .env 文件）
        cfg = load_config(config_path=config_path, use_cli=False, env_file=env_file)

        # 验证环境变量被正确加载
        assert cfg.llm.code.api_key == "sk-env-file-key-123"
        assert cfg.llm.feedback.api_key == "sk-ant-env-key-456"

    def test_config_priority_system_env_over_dotenv(self, tmp_path: Path) -> None:
        """测试系统环境变量优先级高于 .env 文件。"""
        # 创建测试数据目录
        test_data_dir = tmp_path / "test_data"
        test_data_dir.mkdir()

        # 设置系统环境变量（优先级高）
        os.environ["TEST_PRIORITY_KEY"] = "sk-from-system-env"

        # 创建 .env 文件（优先级低）
        env_content = """TEST_PRIORITY_KEY=sk-from-dotenv-file
"""
        env_file = tmp_path / ".env"
        env_file.write_text(env_content)

        # 创建配置文件
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
  goal: "Test"
  eval: null
  preprocess_data: true
  copy_data: false

llm:
  code:
    provider: "openai"
    model: "gpt-4-turbo"
    temperature: 0.5
    api_key: ${{env:TEST_PRIORITY_KEY}}
    base_url: "https://api.openai.com/v1"
    max_tokens: null
  feedback:
    provider: "openai"
    model: "gpt-4-turbo"
    temperature: 0.5
    api_key: "fallback-key"

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
        cfg = load_config(config_path=config_path, use_cli=False, env_file=env_file)

        # 验证系统环境变量优先级更高
        assert cfg.llm.code.api_key == "sk-from-system-env"

        # 清理环境变量
        del os.environ["TEST_PRIORITY_KEY"]

    def test_config_priority_env_over_yaml(self, tmp_path: Path) -> None:
        """测试环境变量优先级高于 YAML 配置。"""
        # 创建测试数据目录
        test_data_dir = tmp_path / "test_data"
        test_data_dir.mkdir()

        # 设置环境变量
        os.environ["TEST_ENV_YAML_KEY"] = "sk-from-env-var"

        # 创建配置文件（同时有默认值和环境变量插值）
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
  goal: "Test"
  eval: null
  preprocess_data: true
  copy_data: false

llm:
  code:
    provider: "openai"
    model: "gpt-4-turbo"
    temperature: 0.5
    api_key: ${{env:TEST_ENV_YAML_KEY}}
    base_url: "https://api.openai.com/v1"
    max_tokens: null
  feedback:
    provider: "openai"
    model: "gpt-4-turbo"
    temperature: 0.5
    api_key: "yaml-default-key"

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
        cfg = load_config(config_path=config_path, use_cli=False, env_file=None)

        # 验证环境变量优先级高于 YAML 默认值
        assert cfg.llm.code.api_key == "sk-from-env-var"
        assert cfg.llm.feedback.api_key == "yaml-default-key"

        # 清理环境变量
        del os.environ["TEST_ENV_YAML_KEY"]

    def test_config_priority_full_chain(self, tmp_path: Path) -> None:
        """测试完整配置优先级链: CLI > 环境变量 > YAML。"""
        # 创建测试数据目录
        test_data_dir = tmp_path / "test_data"
        test_data_dir.mkdir()

        # 设置环境变量（中等优先级）
        os.environ["TEST_FULL_CHAIN_MODEL"] = "gpt-3.5-turbo"

        # 创建配置文件（最低优先级）
        config_content = f"""
project:
  name: "yaml-name"
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
    provider: "openai"
    model: ${{env:TEST_FULL_CHAIN_MODEL}}
    temperature: 0.5
    api_key: "test-key"
    base_url: "https://api.openai.com/v1"
    max_tokens: null
  feedback:
    provider: "openai"
    model: "yaml-model"
    temperature: 0.5
    api_key: "test-key"

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

        # 使用 OmegaConf 模拟 CLI 参数（最高优先级）
        from omegaconf import OmegaConf

        base_cfg = OmegaConf.load(config_path)
        cli_cfg = OmegaConf.from_dotlist(["llm.code.model=gpt-4-turbo"])
        merged_cfg = OmegaConf.merge(base_cfg, cli_cfg)

        from utils.config import validate_config

        cfg = validate_config(merged_cfg)

        # 验证优先级链
        # llm.code.model: CLI 覆盖了环境变量
        assert cfg.llm.code.model == "gpt-4-turbo"

        # llm.feedback.model: 使用 YAML 默认值（无 CLI 和环境变量）
        assert cfg.llm.feedback.model == "yaml-model"

        # project.name: 使用 YAML 默认值
        assert cfg.project.name == "yaml-name"

        # 清理环境变量
        del os.environ["TEST_FULL_CHAIN_MODEL"]
