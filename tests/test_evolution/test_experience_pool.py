"""共享经验池单元测试。"""

import threading
import time
from dataclasses import dataclass

import pytest

from core.evolution.experience_pool import ExperiencePool, TaskRecord


@dataclass
class MockExperiencePoolConfig:
    """模拟经验池配置。"""

    max_records: int = 100
    top_k: int = 5
    save_path: str = ""


@dataclass
class MockEvolutionConfig:
    """模拟进化配置。"""

    experience_pool: MockExperiencePoolConfig = None

    def __post_init__(self):
        if self.experience_pool is None:
            self.experience_pool = MockExperiencePoolConfig()


@dataclass
class MockConfig:
    """模拟配置对象。"""

    evolution: MockEvolutionConfig = None

    def __post_init__(self):
        if self.evolution is None:
            self.evolution = MockEvolutionConfig()


@pytest.fixture
def temp_config(tmp_path):
    """创建临时配置（使用临时目录）。"""
    temp_file = tmp_path / "experience_pool.json"

    config = MockConfig(
        evolution=MockEvolutionConfig(
            experience_pool=MockExperiencePoolConfig(
                max_records=100,
                top_k=5,
                save_path=str(temp_file),
            )
        )
    )

    return config


class TestTaskRecord:
    """测试 TaskRecord 数据类。"""

    def test_create_task_record(self):
        """测试创建 TaskRecord。"""
        record = TaskRecord(
            agent_id="agent_0",
            task_type="explore",
            input_hash="abc123",
            output_quality=0.85,
            strategy_summary="Use RandomForest",
            timestamp=time.time(),
        )

        assert record.agent_id == "agent_0"
        assert record.task_type == "explore"
        assert record.output_quality == 0.85


class TestExperiencePool:
    """测试 ExperiencePool 类。"""

    def test_init_empty(self, temp_config):
        """测试初始化空经验池。"""
        pool = ExperiencePool(temp_config)

        assert len(pool.records) == 0
        assert pool.save_path.parent.exists()

    def test_add_record(self, temp_config):
        """测试添加记录。"""
        pool = ExperiencePool(temp_config)

        record = TaskRecord(
            agent_id="agent_0",
            task_type="explore",
            input_hash="abc123",
            output_quality=0.85,
            strategy_summary="Use RandomForest",
            timestamp=time.time(),
        )

        pool.add(record)

        assert len(pool.records) == 1
        assert pool.records[0].agent_id == "agent_0"

    def test_query_by_task_type(self, temp_config):
        """测试按任务类型查询。"""
        pool = ExperiencePool(temp_config)

        # 添加不同任务类型的记录
        for i in range(3):
            pool.add(
                TaskRecord(
                    agent_id=f"agent_{i}",
                    task_type="explore" if i < 2 else "merge",
                    input_hash=f"hash_{i}",
                    output_quality=0.7 + i * 0.1,
                    strategy_summary=f"Strategy {i}",
                    timestamp=time.time(),
                )
            )

        # 查询 explore 任务
        results = pool.query("explore", k=10)

        assert len(results) == 2
        assert all(r.task_type == "explore" for r in results)

        # 查询 merge 任务
        results = pool.query("merge", k=10)

        assert len(results) == 1
        assert results[0].task_type == "merge"

    def test_query_with_filters(self, temp_config):
        """测试带过滤条件的查询。"""
        pool = ExperiencePool(temp_config)

        # 添加多条记录
        for i in range(5):
            pool.add(
                TaskRecord(
                    agent_id="agent_0" if i < 3 else "agent_1",
                    task_type="explore",
                    input_hash=f"hash_{i}",
                    output_quality=0.5 + i * 0.1,
                    strategy_summary=f"Strategy {i}",
                    timestamp=time.time(),
                )
            )

        # 过滤 output_quality > 0.7（应返回 0.8, 0.9 = 2 个）
        results = pool.query("explore", k=10, output_quality=(">", 0.7))

        assert len(results) == 2
        assert all(r.output_quality > 0.7 for r in results)

        # 过滤 agent_id == "agent_0"
        results = pool.query("explore", k=10, agent_id="agent_0")

        assert len(results) == 3
        assert all(r.agent_id == "agent_0" for r in results)

    def test_query_top_k(self, temp_config):
        """测试 Top-K 查询（按 output_quality 降序）。"""
        pool = ExperiencePool(temp_config)

        # 添加 5 条记录
        qualities = [0.8, 0.6, 0.9, 0.5, 0.7]
        for i, quality in enumerate(qualities):
            pool.add(
                TaskRecord(
                    agent_id=f"agent_{i}",
                    task_type="explore",
                    input_hash=f"hash_{i}",
                    output_quality=quality,
                    strategy_summary=f"Strategy {i}",
                    timestamp=time.time(),
                )
            )

        # 查询 Top-3
        results = pool.query("explore", k=3)

        assert len(results) == 3
        # 验证降序排列：0.9, 0.8, 0.7
        assert results[0].output_quality == 0.9
        assert results[1].output_quality == 0.8
        assert results[2].output_quality == 0.7

    def test_get_agent_stats(self, temp_config):
        """测试 Agent 统计。"""
        pool = ExperiencePool(temp_config)

        # 添加 agent_0 的记录
        for i in range(5):
            pool.add(
                TaskRecord(
                    agent_id="agent_0",
                    task_type="explore",
                    input_hash=f"hash_{i}",
                    output_quality=0.5 + i * 0.1 if i < 4 else -0.1,  # 最后一个失败
                    strategy_summary=f"Strategy {i}",
                    timestamp=time.time(),
                )
            )

        stats = pool.get_agent_stats("agent_0")

        assert stats["total_count"] == 5
        assert stats["success_count"] == 4  # output_quality > 0
        assert (
            abs(stats["avg_quality"] - 0.5) < 0.01
        )  # (0.5+0.6+0.7+0.8-0.1)/5 = 2.5/5 = 0.5
        assert abs(stats["success_rate"] - 0.8) < 0.01  # 4/5

    def test_get_agent_stats_no_records(self, temp_config):
        """测试不存在的 Agent 统计。"""
        pool = ExperiencePool(temp_config)

        stats = pool.get_agent_stats("non_existent")

        assert stats["total_count"] == 0
        assert stats["success_count"] == 0
        assert stats["avg_quality"] == 0.0
        assert stats["success_rate"] == 0.0

    def test_max_records_limit(self, temp_config):
        """测试记录数上限。"""
        # 设置较小的上限
        temp_config.evolution.experience_pool.max_records = 3

        pool = ExperiencePool(temp_config)

        # 添加 5 条记录
        for i in range(5):
            pool.add(
                TaskRecord(
                    agent_id=f"agent_{i}",
                    task_type="explore",
                    input_hash=f"hash_{i}",
                    output_quality=0.5 + i * 0.1,
                    strategy_summary=f"Strategy {i}",
                    timestamp=time.time() + i,  # 递增时间
                )
            )
            time.sleep(0.01)  # 确保时间戳不同

        # 应该只保留最新的 3 条
        assert len(pool.records) == 3

        # 验证是最新的 3 条（agent_2, agent_3, agent_4）
        agent_ids = {r.agent_id for r in pool.records}
        assert (
            "agent_2" in agent_ids or "agent_3" in agent_ids or "agent_4" in agent_ids
        )

    def test_thread_safety(self, temp_config):
        """测试多线程并发写入。"""
        pool = ExperiencePool(temp_config)

        def add_records(agent_id: str, count: int):
            """线程函数：添加多条记录。"""
            for i in range(count):
                pool.add(
                    TaskRecord(
                        agent_id=agent_id,
                        task_type="explore",
                        input_hash=f"{agent_id}_hash_{i}",
                        output_quality=0.5 + i * 0.1,
                        strategy_summary=f"Strategy {i}",
                        timestamp=time.time(),
                    )
                )

        # 创建 4 个线程，每个添加 10 条记录
        threads = []
        for i in range(4):
            thread = threading.Thread(
                target=add_records,
                args=(f"agent_{i}", 10),
            )
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证总记录数
        assert len(pool.records) == 40

    def test_save_and_load(self, temp_config):
        """测试 JSON 持久化。"""
        pool1 = ExperiencePool(temp_config)

        # 添加记录
        for i in range(3):
            pool1.add(
                TaskRecord(
                    agent_id=f"agent_{i}",
                    task_type="explore",
                    input_hash=f"hash_{i}",
                    output_quality=0.5 + i * 0.1,
                    strategy_summary=f"Strategy {i}",
                    timestamp=time.time(),
                )
            )

        # 保存
        pool1.save()
        assert pool1.save_path.exists()

        # 创建新的经验池并加载
        pool2 = ExperiencePool(temp_config)

        assert len(pool2.records) == 3
        assert pool2.records[0].agent_id == "agent_0"
        assert pool2.records[1].agent_id == "agent_1"
        assert pool2.records[2].agent_id == "agent_2"

    def test_load_corrupted_file(self, temp_config):
        """测试加载损坏的 JSON 文件。"""
        # 写入损坏的 JSON
        with open(temp_config.evolution.experience_pool.save_path, "w") as f:
            f.write("{invalid json")

        # 加载应该失败但不抛出异常
        pool = ExperiencePool(temp_config)

        assert len(pool.records) == 0  # 从空白开始

    def test_load_nonexistent_file(self, temp_config):
        """测试加载不存在的文件。"""
        pool = ExperiencePool(temp_config)

        assert len(pool.records) == 0
        assert not pool.save_path.exists()
