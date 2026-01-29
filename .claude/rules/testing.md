# Testing Requirements (Python)

## 测试框架

使用 **pytest** + **pytest-asyncio** 进行测试。

## 最低测试覆盖率: 80%

## 测试类型

1. **单元测试** - 独立函数、工具类
2. **集成测试** - Agent 交互、工作流

> **注意**: 本项目为纯后端系统，无需前端 E2E 测试。

## 测试驱动开发 (TDD)

推荐工作流：
1. 先写测试 (RED)
2. 运行测试 - 应该失败
3. 写最小实现 (GREEN)
4. 运行测试 - 应该通过
5. 重构 (IMPROVE)
6. 验证覆盖率 (80%+)

## pytest 示例

```python
import pytest
from unittest.mock import AsyncMock, patch

class TestAgent:
    """Agent 测试类。"""

    @pytest.mark.asyncio
    async def test_agent_execute_success(self):
        """测试 Agent 正常执行。"""
        agent = MyAgent()
        result = await agent.execute("test input")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_agent_with_mock(self):
        """使用 Mock 测试 Agent。"""
        with patch("module.external_api") as mock_api:
            mock_api.return_value = AsyncMock(return_value={"data": "test"})
            result = await process_with_api()
            assert result == {"data": "test"}
```

## 测试文件组织

```
tests/
├── conftest.py          # pytest fixtures
├── test_agents/         # Agent 测试
├── test_tools/          # 工具函数测试
└── test_integration/    # 集成测试
```

## 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行并显示覆盖率
pytest tests/ --cov=src --cov-report=term-missing

# 运行特定测试
pytest tests/test_agents/test_my_agent.py -v
```

## 故障排查

1. 检查测试隔离性
2. 验证 Mock 是否正确
3. 修复实现，而非测试（除非测试本身有误）
