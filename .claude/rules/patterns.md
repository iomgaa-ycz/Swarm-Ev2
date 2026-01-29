# Common Patterns (Python)

## 数据模型 (Pydantic)

```python
from pydantic import BaseModel
from typing import Optional, List, Generic, TypeVar

T = TypeVar("T")

class ApiResponse(BaseModel, Generic[T]):
    """统一 API 响应格式。"""
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    meta: Optional[dict] = None
```

## 异步上下文管理器

```python
from contextlib import asynccontextmanager
from typing import AsyncGenerator

@asynccontextmanager
async def managed_resource() -> AsyncGenerator[Resource, None]:
    """管理资源的生命周期。"""
    resource = await acquire_resource()
    try:
        yield resource
    finally:
        await release_resource(resource)
```

## Repository 模式

```python
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional, List

T = TypeVar("T")

class Repository(ABC, Generic[T]):
    """数据访问抽象层。"""

    @abstractmethod
    async def find_all(self, filters: Optional[dict] = None) -> List[T]:
        ...

    @abstractmethod
    async def find_by_id(self, id: str) -> Optional[T]:
        ...

    @abstractmethod
    async def create(self, data: dict) -> T:
        ...

    @abstractmethod
    async def update(self, id: str, data: dict) -> T:
        ...

    @abstractmethod
    async def delete(self, id: str) -> None:
        ...
```

## LangGraph Agent 模式

```python
from langgraph.graph import StateGraph
from typing import TypedDict, Annotated
from operator import add

class AgentState(TypedDict):
    """Agent 状态定义。"""
    messages: Annotated[list, add]
    current_step: str
    result: Optional[str]

def create_agent_graph() -> StateGraph:
    """创建 Agent 工作流图。"""
    graph = StateGraph(AgentState)
    graph.add_node("think", think_node)
    graph.add_node("act", act_node)
    graph.add_edge("think", "act")
    return graph.compile()
```

## 配置管理

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """应用配置，从环境变量加载。"""
    openai_api_key: str
    debug: bool = False
    max_workers: int = 4

    class Config:
        env_file = ".env"
```
