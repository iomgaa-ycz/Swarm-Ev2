"""时间限制集成测试。

验证时间限制在实际场景中正确生效。
"""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.orchestrator import Orchestrator
from core.state import Journal
from utils.config import load_config


@pytest.mark.integration
def test_time_limit_integration(tmp_path):
    """集成测试：验证时间限制在实际场景中生效。

    场景：
    1. 设置 time_limit = 5 秒
    2. 运行 10 个 Epoch
    3. 验证程序在 5-10 秒内停止（允许单次任务延迟）

    预期行为：
    - 检测到时间限制后立即停止（不等待所有 Epoch 完成）
    - 实际运行时间接近 time_limit（允许合理误差）
    """
    # 加载配置
    config = load_config()
    config.agent.time_limit = 5  # 5 秒限制
    config.project.workspace_dir = tmp_path
    config.search.parallel_num = 1  # 串行执行，简化测试

    # 初始化
    journal = Journal()
    mock_agent = MagicMock()
    mock_agent.name = "test_agent"

    orchestrator = Orchestrator(
        agents=[mock_agent],
        config=config,
        journal=journal,
        task_desc="Integration test",
    )

    # Mock _step_task 以控制执行时间
    original_step_task = orchestrator._step_task

    def mock_step_task(parent_node):
        """模拟每个任务耗时 0.5 秒。"""
        time.sleep(0.5)
        return None  # 简化：不创建实际节点

    orchestrator._step_task = mock_step_task

    # 执行
    start_time = time.time()
    orchestrator.run(num_epochs=10, steps_per_epoch=5)
    elapsed = time.time() - start_time

    # 验证
    assert elapsed < 10, f"程序应在 10 秒内停止，实际耗时 {elapsed:.2f}s"
    assert elapsed >= 5, f"程序应至少运行 5 秒（time_limit），实际 {elapsed:.2f}s"

    print(f"✅ 集成测试通过：time_limit=5s, 实际运行={elapsed:.2f}s")


@pytest.mark.integration
def test_time_limit_epoch_boundary(tmp_path):
    """测试时间限制在 Epoch 边界正确触发。

    验证：
    - 第 N 个 Epoch 开始时检测到超时
    - 不会启动第 N+1 个 Epoch
    """
    config = load_config()
    config.agent.time_limit = 3  # 3 秒限制
    config.project.workspace_dir = tmp_path
    config.search.parallel_num = 1

    journal = Journal()
    mock_agent = MagicMock()
    mock_agent.name = "test_agent"

    orchestrator = Orchestrator(
        agents=[mock_agent],
        config=config,
        journal=journal,
        task_desc="Epoch boundary test",
    )

    # Mock：每个 Epoch 耗时 2 秒
    epoch_count = [0]

    def mock_run_epoch(steps):
        epoch_count[0] += 1
        time.sleep(2)  # 每个 Epoch 2 秒
        return True  # 正常完成（时间限制由 _check_time_limit 检测）

    orchestrator._run_single_epoch = mock_run_epoch

    # 执行（应该运行 1-2 个 Epoch）
    start_time = time.time()
    orchestrator.run(num_epochs=10, steps_per_epoch=5)
    elapsed = time.time() - start_time

    # 验证
    assert epoch_count[0] <= 2, f"应该运行 1-2 个 Epoch，实际 {epoch_count[0]} 个"
    assert elapsed < 6, f"应在 6 秒内停止，实际 {elapsed:.2f}s"

    print(
        f"✅ Epoch 边界测试通过：运行了 {epoch_count[0]} 个 Epoch，耗时 {elapsed:.2f}s"
    )
