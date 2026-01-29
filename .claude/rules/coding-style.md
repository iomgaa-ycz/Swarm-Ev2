# Coding Style (Python)

## 不可变性原则

尽量避免原地修改对象，优先返回新对象：

```python
# 错误: 原地修改
def update_user(user: dict, name: str) -> dict:
    user["name"] = name  # 修改了原对象!
    return user

# 正确: 返回新对象
def update_user(user: dict, name: str) -> dict:
    return {**user, "name": name}
```

## 文件组织

小文件优于大文件：
- 高内聚，低耦合
- 典型 200-400 行，最大 800 行
- 从大模块中提取工具函数
- 按功能/领域组织，而非按类型

## 错误处理

使用项目统一的日志系统：

```python
from utils.logger_system import log_msg, log_json

try:
    result = await risky_operation()
    return result
except Exception as e:
    log_msg("ERROR", f"操作失败: {e}")  # 自动抛出异常
```

## 输入验证

使用 Pydantic 验证用户输入：

```python
from pydantic import BaseModel, EmailStr, Field

class UserInput(BaseModel):
    email: EmailStr
    age: int = Field(ge=0, le=150)

validated = UserInput(**input_data)
```

## 类型注解

强制所有函数签名包含完整类型注解：

```python
from typing import Optional, Union, Dict, List

def process_data(
    items: List[Dict[str, any]],
    config: Optional[Dict] = None
) -> Union[List[str], None]:
    """处理数据并返回结果。"""
    ...
```

## 代码质量检查清单

完成工作前检查：
- [ ] 代码可读，命名规范
- [ ] 函数短小 (<50 行)
- [ ] 文件专注 (<800 行)
- [ ] 无深层嵌套 (>4 层)
- [ ] 正确的错误处理
- [ ] 无 print() 语句（使用 log_msg）
- [ ] 无硬编码值
- [ ] 包含中文 Docstring
