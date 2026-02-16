# P1 实施计划：代理加速 + 自适应超时

## 1.1 摘要

通过配置局域网 Clash 代理（环境变量方式）加速容器内网络下载，并实现自适应超时策略，双管齐下解决 P1 超时问题。需兼容 `main.py`（本地模式）和 `run_mle_adapter.py`（MLE-Bench Docker 模式）两套运行路径。

## 1.2 审查点 (User Review Required) [已确认]

1. **Clash 代理地址**: `http://192.168.31.250:7890` [已填入 .env]
2. **代理认证问题**: 实测发现代理返回 `407 Proxy Authentication Required`，需要用户名/密码。**当前实现先跳过代理**，直连可用（`curl https://pypi.org` 成功）。如需启用代理，请提供认证信息。
3. **容器网络验证**: MLE-Bench 容器默认 `bridge` 网络模式，**可访问互联网**（实测直连 PyPI 成功）。预训练权重下载问题应该可以解决。
4. **NO_PROXY 排除列表**: 已设置 `localhost,127.0.0.1`（本地 grading server）
5. **自适应超时上限**: 7200s（2h）已配置

## 1.3 拟议变更

### Part A: 代理配置（环境变量方式）

#### A1. `utils/proxy.py` [NEW]
统一代理配置工具，供两套运行模式共用。

- `setup_proxy_env()`: 读取代理环境变量，确保大小写变体同时设置（`HTTP_PROXY`/`http_proxy`，`HTTPS_PROXY`/`https_proxy`，`NO_PROXY`/`no_proxy`），因为不同的 Python 库检查不同大小写：
  - `urllib.request` (PyTorch Hub 下载): 检查小写 `http_proxy`
  - `requests` (pip, HuggingFace): 检查大写 `HTTP_PROXY`
  - `httpx` (OpenAI SDK): 同时检查两种
- `log_proxy_status()`: 启动时日志输出代理配置状态（脱敏显示）

```python
# utils/proxy.py 伪代码
def setup_proxy_env() -> None:
    """确保代理环境变量大小写同步。"""
    proxy_vars = ["HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"]
    for var in proxy_vars:
        upper_val = os.environ.get(var, "")
        lower_val = os.environ.get(var.lower(), "")
        # 以先找到的为准，同步到另一个
        val = upper_val or lower_val
        if val:
            os.environ[var] = val
            os.environ[var.lower()] = val
```

#### A2. `.env.example` [MODIFY]
添加代理环境变量模板。

```diff
+# ============================================================
+# 网络代理配置（可选）
+# ============================================================
+# 适用场景：国内环境加速 PyTorch Hub / pip / HuggingFace 下载
+# 示例：局域网 Clash 代理
+# HTTP_PROXY=http://192.168.1.1:7890
+# HTTPS_PROXY=http://192.168.1.1:7890
+# NO_PROXY=localhost,127.0.0.1
```

#### A3. `run_mle_adapter.py` [MODIFY]
在 `map_env_vars()` 之后调用 `setup_proxy_env()`。

- 函数 `map_env_vars()` 后增加 `from utils.proxy import setup_proxy_env; setup_proxy_env()`

#### A4. `main.py` [MODIFY]
在配置加载后调用 `setup_proxy_env()`（`load_config()` 内部已调用 `load_dotenv()`，所以此时 `.env` 中的代理变量已在 `os.environ` 中）。

- `main()` 函数 Phase 1 末尾添加 `setup_proxy_env()` 调用

#### A5. `Reference/mle-bench/agents/swarm-evo/config.yaml` [MODIFY]
添加代理环境变量传递到 Docker 容器。

```diff
 swarm-evo:
   <<: *defaults
   env_vars:
     <<: *env_vars
     API_KEY: ${{ secrets.API_KEY }}
     API_BASE: ${{ secrets.API_BASE }}
     MODEL_NAME: ${{ secrets.MODEL_NAME }}
+    HTTP_PROXY: ${{ secrets.HTTP_PROXY }}
+    HTTPS_PROXY: ${{ secrets.HTTPS_PROXY }}
+    NO_PROXY: ${{ secrets.NO_PROXY }}
```

#### A6. `start.sh` [MODIFY]
启动时打印代理状态（帮助调试）。

```diff
 #!/bin/bash
 cd /home/agent
 export PYTHONPATH=$PYTHONPATH:/home/agent
 eval "$(conda shell.bash hook)"
 conda activate agent
 mkdir -p /home/code /home/logs
+# 代理状态日志
+if [ -n "$HTTP_PROXY" ]; then
+    echo "[Proxy] HTTP_PROXY=$HTTP_PROXY" >> /home/logs/entrypoint.log
+fi
 python /home/agent/run_mle_adapter.py
```

### Part B: 自适应超时

#### B1. `config/default.yaml` [MODIFY]
添加自适应超时配置。

```diff
 execution:
-  timeout: 3600  # 单次执行超时时间（秒）
+  timeout: 3600  # 基础超时时间（秒）
+  timeout_max: 7200  # 最大超时时间（秒）
+  adaptive_timeout: true  # 是否启用自适应超时
   agent_file_name: "runfile.py"
   format_tb_ipython: false
```

#### B2. `config/mle_bench.yaml` [MODIFY]
同上。

#### B3. `utils/config.py` [MODIFY]
`ExecutionConfig` 添加新字段。

- `ExecutionConfig` [MODIFY]: 添加 `timeout_max: int = 7200`, `adaptive_timeout: bool = True`
- `validate_config()` [MODIFY]: 添加新字段的 schema 默认值

#### B4. `core/orchestrator.py` [MODIFY]
新增 `_estimate_timeout()` 方法，在执行代码前动态计算超时。

```python
# 伪代码
def _estimate_timeout(self) -> int:
    """根据数据集特征估算合理超时时间。"""
    if not self.config.execution.adaptive_timeout:
        return self.config.execution.timeout

    base = self.config.execution.timeout  # 3600
    max_timeout = self.config.execution.timeout_max  # 7200

    # 数据集大小检测
    input_dir = self.config.project.workspace_dir / "input"
    total_size_mb = sum(f.stat().st_size for f in input_dir.rglob("*") if f.is_file()) / (1024 * 1024)

    multiplier = 1.0
    if total_size_mb > 500:   # 大数据集 (>500MB)
        multiplier = 2.0
    elif total_size_mb > 100:  # 中等数据集 (>100MB)
        multiplier = 1.5

    return min(int(base * multiplier), max_timeout)
```

- `__init__()` [MODIFY]: 调用 `_estimate_timeout()` 设置 Interpreter 超时
- 或在创建 Interpreter 时使用计算后的超时值

#### B5. `core/executor/interpreter.py` [MODIFY]
支持运行时更新超时值（可选，如果需要按执行动态调整）。

- `set_timeout(timeout: int)` [NEW]: 允许外部更新超时值

## 1.4 验证计划

### 代理配置验证

1. **本地模式测试**:
```bash
# .env 中设置代理后
conda run -n Swarm-Evo python -c "
import os
from utils.proxy import setup_proxy_env, log_proxy_status
setup_proxy_env()
log_proxy_status()
print(f'http_proxy={os.environ.get(\"http_proxy\", \"未设置\")}')
print(f'HTTP_PROXY={os.environ.get(\"HTTP_PROXY\", \"未设置\")}')
"
```

2. **代理连通性测试**:
```bash
conda run -n Swarm-Evo python -c "
import urllib.request, os
os.environ['http_proxy'] = 'http://YOUR_PROXY:7890'
os.environ['https_proxy'] = 'http://YOUR_PROXY:7890'
# 测试 PyTorch Hub 地址
urllib.request.urlopen('https://download.pytorch.org/models/resnet18-f37072fd.pth', timeout=10)
print('连接成功')
"
```

3. **Docker 容器网络验证**（关键，确认容器可联网）:
```bash
# 在 MLE-Bench 容器内执行
curl -x http://YOUR_PROXY:7890 https://download.pytorch.org/models/resnet18-f37072fd.pth -o /dev/null -w "%{http_code}" --connect-timeout 10
```

### 自适应超时验证

1. **单元测试**: 测试 `_estimate_timeout()` 对不同数据集大小的返回值
```bash
conda run -n Swarm-Evo pytest tests/unit/test_adaptive_timeout.py -v
```

2. **集成验证**: 检查日志输出是否包含动态超时值
```
预期日志: "INFO: 自适应超时: base=3600, dataset=650MB, timeout=7200s"
```

## 1.5 环境变量传递流程总结

### main.py 模式
```
.env → load_dotenv() → os.environ → setup_proxy_env() → Interpreter subprocess 继承
```

### run_mle_adapter.py (Docker) 模式
```
宿主机环境变量 → config.yaml secrets → Docker environment → os.environ → setup_proxy_env() → Interpreter subprocess 继承
```

关键：`interpreter.py:381` 的 `env={**os.environ, "PYTHONUNBUFFERED": "1"}` 确保所有环境变量（包括代理）自动传递到 Agent 生成的代码子进程中。

## 1.6 预估影响

| 改进项 | 影响范围 | 预期效果 |
|--------|---------|---------|
| 代理加速下载 | siim-melanoma, ranzcr-clip | 下载耗时从数分钟→数秒，腾出更多训练时间 |
| 代理 + 预训练权重 | ranzcr-clip, siim-melanoma | **可能解决 `weights=None` 问题！** +1~2 奖牌 |
| 自适应超时 | 所有 DL 竞赛 | 大数据集/DL 任务可用 7200s，更多 epoch |
| 合计 | P1 竞赛 | **+1~2 奖牌**（如果预训练权重下载成功则可能 +2~3） |

## 1.7 风险评估

| 风险 | 可能性 | 缓解措施 |
|------|--------|---------|
| 容器实际禁网（报告说法正确） | 中 | 先验证容器网络，若禁网则代理仅对 main.py 模式有效 |
| 代理地址泄露 | 低 | 通过 .env 管理，.gitignore 已排除 |
| 超时过长导致单节点占用过多资源 | 低 | timeout_max 上限控制 |
| 代理不稳定 | 低 | 代理断开时降级为直连，不影响功能 |
