# 开发计划：环境信息注入到 Agent Prompt

## 1. 摘要 (Summary)

将设备信息（CPU/RAM/GPU）和 Conda 环境信息动态注入到 Agent 的 Prompt 中，使 Agent 能够根据硬件条件优化代码（如选择 GPU/CPU、调整 batch size）和使用正确的包版本。

---

## 2. 审查点 (User Review Required)

| 审查项 | 问题 | 建议 |
|--------|------|------|
| **Conda 环境名称** | 配置默认值应该是 `Swarm-Evo` 还是其他？ | 建议使用 `Swarm-Evo`，与 CLAUDE.md 保持一致 |
| **GPU 获取失败处理** | 如果 `nvidia-smi` 不可用，是报错还是返回 "无 GPU"？ | MVP 建议：返回 "无 GPU" 而非报错 |
| **信息获取时机** | 在哪个阶段获取？启动时一次 vs 每个 Step？ | 建议：Orchestrator 启动时一次性获取并缓存 |
| **模板位置** | 环境信息放在 `explore.j2` 的哪个位置？ | 建议：放在 GUIDELINES 部分的 "Time and Resource Constraints" 之后 |

---

## 3. 调用链分析

```
Orchestrator.__init__()
    └─→ 获取环境信息（一次性）
        └─→ system_info.get_hardware_description()
        └─→ system_info.get_conda_packages()

Orchestrator._step_task()
    └─→ AgentContext(... device_info, conda_packages)
        └─→ agent.generate(context)
            └─→ PromptBuilder.build_explore_prompt()
                └─→ PromptManager.build_prompt()
                    └─→ explore.j2 模板渲染
                        └─→ {{ device_info }}
                        └─→ {{ conda_packages }}
```

---

## 4. 拟议变更 (Proposed Changes)

### 4.1 配置层

| 文件 | 修改内容 | 标识 |
|------|----------|------|
| `config/default.yaml` | [NEW] 添加 `environment.conda_env_name: "Swarm-Evo"` 配置节 | `[NEW]` |

**变更详情**:
```yaml
# ============================================================
# 环境配置（设备信息 & Conda 环境）
# ============================================================
environment:
  conda_env_name: "Swarm-Evo"  # Conda 环境名称
```

---

### 4.2 信息获取层

| 文件 | 修改内容 | 标识 |
|------|----------|------|
| `utils/system_info.py` | [NEW] 创建系统信息获取模块 | `[NEW]` |

**变更详情**:

从 `Reference/Swarm-Evo/utils/system_info.py` 简化移植，保留核心功能：

| 函数 | 功能 | MVP 简化 |
|------|------|----------|
| `get_cpu_count()` | 获取 CPU 核心数 | 失败返回 1 |
| `get_memory_info()` | 获取内存信息 (GB) | 失败返回 `{"total": 8, "available": 8}` |
| `get_gpu_info()` | 获取 GPU 信息 | 失败返回 `None` |
| `get_hardware_description()` | 组合硬件描述字符串 | 返回如 `"CPU: 8 cores, RAM: 32GB, GPU: RTX 3090"` |
| `get_conda_packages()` | 获取 Conda 环境包信息 | 失败返回默认描述 |

**关键设计决策**:
- MVP 简化：任何获取失败都返回合理默认值，而非抛出异常
- 日志记录：获取失败时使用 `log_msg("WARNING", ...)` 记录

---

### 4.3 上下文传递层

#### 4.3.1 AgentContext 修改

| 文件 | 修改内容 | 标识 |
|------|----------|------|
| `agents/base_agent.py` | [MODIFY] `AgentContext` 添加 `device_info` 和 `conda_packages` 字段 | `[MODIFY]` |

**变更详情**:
```python
@dataclass
class AgentContext(DataClassJsonMixin):
    # ... 现有字段 ...

    # [NEW] 环境信息字段
    device_info: str = ""           # 硬件描述字符串
    conda_packages: str = ""        # Conda 包信息
    conda_env_name: str = ""        # Conda 环境名称
```

#### 4.3.2 Orchestrator 修改

| 文件 | 修改内容 | 标识 |
|------|----------|------|
| `core/orchestrator.py` | [MODIFY] `__init__()` 添加环境信息获取和缓存 | `[MODIFY]` |
| `core/orchestrator.py` | [MODIFY] `_step_task()` 传递环境信息到 AgentContext | `[MODIFY]` |

**变更详情**:

`__init__()` 添加:
```python
# 获取并缓存环境信息（一次性）
from utils.system_info import get_hardware_description, get_conda_packages

self.device_info = get_hardware_description()
self.conda_packages = get_conda_packages(
    config.environment.conda_env_name if hasattr(config, 'environment') else None
)
self.conda_env_name = getattr(config.environment, 'conda_env_name', 'Swarm-Evo')

log_msg("INFO", f"环境信息: {self.device_info}")
```

`_step_task()` 修改:
```python
context = AgentContext(
    task_type="explore",
    parent_node=parent_node,
    # ... 现有字段 ...
    device_info=self.device_info,           # [NEW]
    conda_packages=self.conda_packages,     # [NEW]
    conda_env_name=self.conda_env_name,     # [NEW]
)
```

#### 4.3.3 PromptBuilder 修改

| 文件 | 修改内容 | 标识 |
|------|----------|------|
| `utils/prompt_builder.py` | [MODIFY] `build_explore_prompt()` 传递环境信息 | `[MODIFY]` |

**变更详情**:
```python
def build_explore_prompt(
    self,
    # ... 现有参数 ...
    device_info: str = "",        # [NEW]
    conda_packages: str = "",     # [NEW]
    conda_env_name: str = "",     # [NEW]
) -> str:
    context = {
        # ... 现有字段 ...
        "device_info": device_info,           # [NEW]
        "conda_packages": conda_packages,     # [NEW]
        "conda_env_name": conda_env_name,     # [NEW]
    }
```

#### 4.3.4 PromptManager 修改

| 文件 | 修改内容 | 标识 |
|------|----------|------|
| `utils/prompt_manager.py` | [MODIFY] `build_prompt()` 注入环境信息到模板上下文 | `[MODIFY]` |

**变更详情**:

无需修改，因为 `context` 字典会被 `**context` 展开传递给模板。只需确保 `explore.j2` 使用这些变量。

---

### 4.4 模板层

| 文件 | 修改内容 | 标识 |
|------|----------|------|
| `benchmark/mle-bench/prompt_templates/explore.j2` | [MODIFY] 添加环境信息展示 | `[MODIFY]` |

**变更详情**:

在 `GUIDELINES` 部分添加（第 82-86 行之后）:

```jinja2
{# SECTION: GUIDELINES #}
{{ load_skill("static/workspace_rules") }}

{{ load_skill("static/code_style") }}

## Time and Resource Constraints

- **Total Time Remaining**: {{ time_str }}
- **Total Steps Remaining**: {{ steps_remaining }}

## System Environment

- **Device**: {{ device_info | default("CPU only") }}
- **Conda Environment**: `{{ conda_env_name | default("python") }}`
{% if conda_packages %}
- **Installed Packages**: {{ conda_packages }}
{% endif %}

**Note**: Use time and steps efficiently. If GPU is available, prioritize GPU-accelerated solutions.
{# END SECTION: GUIDELINES #}
```

---

### 4.5 其他模板（可选）

| 文件 | 修改内容 | 标识 |
|------|----------|------|
| `benchmark/mle-bench/prompt_templates/merge.j2` | [OPTIONAL] 添加环境信息（如需要） | `[OPTIONAL]` |
| `benchmark/mle-bench/prompt_templates/mutate.j2` | [OPTIONAL] 添加环境信息（如需要） | `[OPTIONAL]` |

MVP 阶段建议：只修改 `explore.j2`，其他模板后续按需添加。

---

## 5. 文件变更总结

| 文件 | 操作 | 变更描述 |
|------|------|----------|
| `config/default.yaml` | `[NEW]` | 添加 `environment.conda_env_name` |
| `utils/system_info.py` | `[NEW]` | 创建系统信息获取模块（~150 行） |
| `agents/base_agent.py` | `[MODIFY]` | AgentContext 添加 3 个字段 |
| `core/orchestrator.py` | `[MODIFY]` | `__init__` 获取环境信息，`_step_task` 传递 |
| `utils/prompt_builder.py` | `[MODIFY]` | `build_explore_prompt` 添加参数 |
| `benchmark/mle-bench/prompt_templates/explore.j2` | `[MODIFY]` | 添加 System Environment 部分 |

**预估新增代码**: ~200 行

---

## 6. 验证计划 (Verification Plan)

### 6.1 单元验证

```bash
# 1. 验证 system_info 模块
conda run -n Swarm-Evo python -c "
from utils.system_info import get_hardware_description, get_conda_packages
print('Device:', get_hardware_description())
print('Packages:', get_conda_packages('Swarm-Evo')[:200] + '...')
"

# 预期输出:
# Device: CPU: 8 cores, RAM: 16GB, GPU: Apple M1
# Packages: Current Conda environment 'Swarm-Evo' contains 157 packages...
```

### 6.2 集成验证

```bash
# 2. 运行主程序，检查日志
conda run -n Swarm-Evo python main.py

# 3. 验证日志包含环境信息
grep "环境信息" logs/system.log
grep "Device:" logs/system.log

# 预期: 日志显示设备信息
```

### 6.3 Prompt 输出验证

```bash
# 4. 检查生成的 Prompt 是否包含环境信息（手动查看或日志）
# 在 working/solution_xxx/output.txt 或调试日志中检查

# 预期: Prompt 包含类似以下内容:
# ## System Environment
# - **Device**: CPU: 8 cores, RAM: 16GB
# - **Conda Environment**: `Swarm-Evo`
```

---

## 7. 依赖与风险

| 风险项 | 级别 | 缓解措施 |
|--------|------|----------|
| `nvidia-smi` 不可用 | 低 | 返回 `None`，不影响非 GPU 环境 |
| Conda 环境不存在 | 低 | 返回默认描述，日志警告 |
| `psutil` 未安装 | 低 | 优先使用 `/proc/meminfo`，fallback 到默认值 |
| 配置文件格式变更 | 低 | 使用 `getattr()` 兼容旧配置 |

---

## 8. 与参考实现的差异

| 方面 | Swarm-Evo 参考实现 | 本项目 MVP 实现 |
|------|-------------------|-----------------|
| 错误处理 | 抛出异常 | 返回默认值 + 日志警告 |
| 包信息格式 | 详细自然语言描述 | 简化版描述 |
| 获取时机 | 每次构建 Prompt | Orchestrator 启动时缓存 |
| 模板变量 | `device_info`, `conda_packages`, `conda_env_name` | 相同 |

---

## 9. 时间线

| 阶段 | 任务 |
|------|------|
| Phase 1 | 创建 `utils/system_info.py` |
| Phase 2 | 修改 `config/default.yaml` |
| Phase 3 | 修改上下文传递链 (AgentContext → Orchestrator → PromptBuilder) |
| Phase 4 | 修改 `explore.j2` 模板 |
| Phase 5 | 验证 & 测试 |

---

**请审核此计划。批准后我将开始实施。**
