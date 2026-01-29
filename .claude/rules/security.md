# Security Guidelines (Python)

## 强制安全检查

每次提交前检查：
- [ ] 无硬编码密钥（API keys, passwords, tokens）
- [ ] 所有用户输入已验证
- [ ] SQL 注入防护（参数化查询）
- [ ] 命令注入防护（避免 shell=True）
- [ ] 认证/授权已验证
- [ ] 错误消息不泄露敏感数据
- [ ] 敏感数据已脱敏处理

## 密钥管理

```python
import os
from utils.logger_system import log_msg

# 错误: 硬编码密钥
api_key = "sk-proj-xxxxx"

# 正确: 环境变量
api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    log_msg("ERROR", "OPENAI_API_KEY 未配置")
```

## 使用 Pydantic 配置

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """安全地从环境变量加载配置。"""
    openai_api_key: str
    database_url: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
```

## 命令执行安全

```python
import subprocess

# 错误: shell=True 有命令注入风险
subprocess.run(f"ls {user_input}", shell=True)

# 正确: 使用列表参数
subprocess.run(["ls", user_input], shell=False)
```

## 安全响应协议

如果发现安全问题：
1. 立即停止
2. 使用 **security-reviewer** agent
3. 修复 CRITICAL 问题后再继续
4. 轮换任何暴露的密钥
5. 审查整个代码库中的类似问题
