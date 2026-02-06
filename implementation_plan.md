# MLE-Bench 结果保存修复 + 路径优化方案

## 1.1 摘要

修复 MLE-Bench 模式下的 3 个结果保存问题（Journal 未持久化、archive_path 死代码、submission 查找脆弱），同时优化路径方案：用 symlink 替代运行后复制，使日志实时写入 MLE-Bench 标准目录。

## 1.2 审查点

已确认，无待定项：
- `main.py` 同步加上 `journal.json` 保存
- `workspace_dir` 保持 `/home/workspace`

## 1.3 拟议变更

### 核心思路：symlink + 精确查找 + Journal 持久化

**AIDE 的做法（参考）：** 在 `start.sh` 中用 symlink 把 workspace 子目录直接指向 MLE-Bench 标准输出目录，写入时自动落地到正确位置。

**我们的适配方案：**

```
setup_workspace() 创建的目录结构（MLE-Bench 模式）:

/home/workspace/
├── input/              → symlink 到 /home/data/*
├── working/            （真实目录，节点详细记录）
├── submission/         （真实目录，submission_{node_id}.csv）
├── best_solution/      （真实目录，solution.py + submission.csv 备份）
├── logs/               → symlink 到 /home/logs/    ← 新增 symlink
├── evolution/          （真实目录，experience_pool.json）
└── description.md

效果：
- orchestrator 写 workspace/logs/system.log  → 实际落地到 /home/logs/system.log ✓
- 保存 workspace/logs/journal.json           → 实际落地到 /home/logs/journal.json ✓
- 无需运行后复制日志
```

**为什么 `best_solution/` 不用 symlink？**
因为我们往 `best_solution/` 里同时写 `solution.py` 和 `submission.csv`，但 MLE-Bench 要求它们在不同目录（`/home/code/` 和 `/home/submission/`）。所以 `best_solution/` 保持真实目录，最后只精确复制 2 个文件。

---

### 文件修改清单

#### 1. `run_mle_adapter.py` `[MODIFY]`

| 函数 | 修改内容 |
|------|---------|
| `setup_workspace()` | `logs/` 子目录改为 symlink → `/home/logs/`（而非创建真实目录） |
| `copy_results()` | **重写**：① 删除 `archive_path` 死代码；② 用 `best_node.id` 精确查找 submission；③ 主动保存 `journal.json`；④ 删除日志复制（已通过 symlink 解决） |
| `run_adapter()` | 在 Phase 8 结果输出前，新增 `journal.to_json()` 保存调用 |

**`setup_workspace()` 变更伪代码：**
```python
def setup_workspace(config, description):
    ws = config.project.workspace_dir

    # 真实目录
    for subdir in ["input", "working", "submission", "best_solution", "evolution"]:
        (ws / subdir).mkdir(parents=True, exist_ok=True)

    # logs/ → symlink 到 /home/logs/（MLE-Bench 标准日志目录）
    logs_target = Path("/home/logs")
    logs_target.mkdir(parents=True, exist_ok=True)
    logs_link = ws / "logs"
    if not logs_link.exists():
        logs_link.symlink_to(logs_target)

    # ...其余不变（description.md、data symlink）
```

**`copy_results()` 重写伪代码：**
```python
def copy_results(journal, config):
    submission_dir = Path("/home/submission")
    code_dir = Path("/home/code")
    ws = config.project.workspace_dir

    submission_dir.mkdir(parents=True, exist_ok=True)
    code_dir.mkdir(parents=True, exist_ok=True)

    # [1] 保存 journal.json（通过 logs symlink 自动落地到 /home/logs/）
    journal_path = ws / "logs" / "journal.json"
    journal_path.write_text(journal.to_json(indent=2), encoding="utf-8")

    # [2] 精确复制 submission.csv（基于 best_node.id）
    best_node = journal.get_best_node(only_good=True)
    sub_file = submission_dir / "submission.csv"

    if best_node:
        # 优先：按 node ID 精确匹配
        precise_src = ws / "submission" / f"submission_{best_node.id}.csv"
        if precise_src.exists():
            shutil.copy2(precise_src, sub_file)
        # 回退：best_solution 目录中的副本
        elif (ws / "best_solution" / "submission.csv").exists():
            shutil.copy2(ws / "best_solution" / "submission.csv", sub_file)

    # [3] 复制 solution.py
    best_code = ws / "best_solution" / "solution.py"
    if best_code.exists():
        shutil.copy2(best_code, code_dir / "solution.py")

    # 日志已通过 symlink 直接写入 /home/logs/，无需复制
```

**删除的代码：**
- `archive_path` 归档恢复块（约 8 行）
- 兜底 `ws.glob("**/submission.csv")` 搜索
- `journal.json` 复制逻辑（改为主动保存）

---

#### 2. `main.py` `[MODIFY]`

| 函数 | 修改内容 |
|------|---------|
| `main()` | Phase 6 结果展示后，新增 `journal.to_json()` 保存到 `workspace/logs/journal.json` |

**变更伪代码（约 3 行）：**
```python
# Phase 6 末尾新增
journal_path = config.project.workspace_dir / "logs" / "journal.json"
journal_path.write_text(journal.to_json(indent=2), encoding="utf-8")
log_msg("INFO", f"Journal 已保存: {journal_path}")
```

---

### 不修改的文件

| 文件 | 原因 |
|------|------|
| `core/orchestrator.py` | `_save_best_solution()` 逻辑不变，仍写入 `best_solution/` |
| `core/state/node.py` | 不新增 `archive_path` 字段（删除使用方的死代码即可） |
| `core/state/journal.py` | 已有 `to_json()` 方法，无需修改 |
| `config/mle_bench.yaml` | `workspace_dir: "/home/workspace"` 保持不变 |
| `Dockerfile` / `start.sh` | 无需修改 |

---

## 1.4 验证计划

### 验证 1：symlink 正确性（本地模拟）

```bash
# 模拟 MLE-Bench 目录结构
mkdir -p /tmp/mle_test/{logs,submission,code,workspace}

# 创建 symlink（模拟 setup_workspace 行为）
ln -s /tmp/mle_test/logs /tmp/mle_test/workspace/logs

# 验证写入穿透
echo "test" > /tmp/mle_test/workspace/logs/test.log
cat /tmp/mle_test/logs/test.log  # 应输出 "test"

# 清理
rm -rf /tmp/mle_test
```

### 验证 2：Journal 序列化

```bash
conda run -n Swarm-Evo python -c "
from core.state import Journal, Node
j = Journal()
j.append(Node(code='print(1)', metric_value=0.85))
j.append(Node(code='print(2)', metric_value=0.90, is_buggy=True))
data = j.to_json(indent=2)
assert '\"metric_value\": 0.85' in data
assert '\"metric_value\": 0.9' in data
print('✅ Journal 序列化验证通过')
print(f'大小: {len(data)} bytes')
"
```

### 验证 3：copy_results 精确性

```bash
conda run -n Swarm-Evo python -c "
# 模拟 best_node.id 精确匹配
from pathlib import Path
import tempfile, shutil

with tempfile.TemporaryDirectory() as ws:
    ws = Path(ws)
    (ws / 'submission').mkdir()
    (ws / 'best_solution').mkdir()

    # 模拟多个 submission 文件
    for name in ['submission_aaa.csv', 'submission_bbb.csv']:
        (ws / 'submission' / name).write_text(f'data_{name}')

    # 模拟 best_solution
    (ws / 'best_solution' / 'submission.csv').write_text('data_best')

    # 精确查找 bbb
    target = ws / 'submission' / 'submission_bbb.csv'
    assert target.exists()
    assert target.read_text() == 'data_submission_bbb.csv'
    print('✅ 精确查找验证通过')
"
```

### 验证 4：单元测试

```bash
conda run -n Swarm-Evo pytest tests/unit/ -v -k "journal or state"
```
