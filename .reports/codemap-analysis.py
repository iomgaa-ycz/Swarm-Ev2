#!/usr/bin/env python3
"""
Codemap 差异分析脚本

扫描代码库并计算与旧版 codemap 的差异百分比。
"""

import json
from datetime import datetime
from pathlib import Path


def analyze_codebase() -> dict:
    """扫描代码库结构，返回模块统计。"""
    base_dir = Path(__file__).parent.parent

    # 扫描所有 Python 文件
    py_files = {
        "core": list((base_dir / "core").rglob("*.py")),
        "utils": list((base_dir / "utils").rglob("*.py")),
        "tests": list((base_dir / "tests").rglob("*.py")),
    }

    stats = {}
    for category, files in py_files.items():
        stats[category] = {
            "files": [str(f.relative_to(base_dir)) for f in files if f.name != "__init__.py"],
            "count": len([f for f in files if f.name != "__init__.py"]),
            "total_lines": sum(
                len(f.read_text(encoding="utf-8").splitlines())
                for f in files
                if f.name != "__init__.py"
            ),
        }

    return stats


def calculate_diff() -> dict:
    """计算代码变更差异。"""
    old_modules = {
        # Phase 1 基础设施
        "utils/config.py": 457,
        "utils/logger_system.py": 181,
        "utils/file_utils.py": 114,
        # Phase 1 数据结构
        "core/state/node.py": 119,
        "core/state/journal.py": 229,
        "core/state/task.py": 63,
        # Phase 1 后端抽象
        "core/backend/__init__.py": 147,
        "core/backend/backend_openai.py": 133,
        "core/backend/backend_anthropic.py": 143,
        "core/backend/utils.py": 81,
    }

    new_modules = {
        # Phase 2 新增
        "core/executor/interpreter.py": 177,
        "core/executor/workspace.py": 182,
        "utils/data_preview.py": 270,
        "utils/metric.py": 118,
        "utils/response.py": 90,
    }

    total_old = sum(old_modules.values())
    total_new = sum(new_modules.values())
    change_pct = (total_new / total_old * 100) if total_old > 0 else 0

    return {
        "old_total_lines": total_old,
        "new_total_lines": total_new,
        "change_percentage": round(change_pct, 2),
        "new_modules": list(new_modules.keys()),
        "new_module_count": len(new_modules),
    }


def generate_report():
    """生成差异报告。"""
    stats = analyze_codebase()
    diff = calculate_diff()

    report = f"""
# Codemap 差异分析报告
生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 1. 代码库统计

### 核心模块 (core/)
- 文件数: {stats['core']['count']}
- 总行数: {stats['core']['total_lines']}

### 工具模块 (utils/)
- 文件数: {stats['utils']['count']}
- 总行数: {stats['utils']['total_lines']}

### 测试模块 (tests/)
- 文件数: {stats['tests']['count']}
- 总行数: {stats['tests']['total_lines']}

## 2. 变更分析

### Phase 1 已完成模块 (基准)
- 总行数: {diff['old_total_lines']}
- 模块数: 10

### Phase 2 新增模块
- 总行数: {diff['new_total_lines']}
- 模块数: {diff['new_module_count']}
- 变更百分比: {diff['change_percentage']}%

新增文件:
"""

    for module in diff["new_modules"]:
        report += f"  - {module}\n"

    report += f"""
## 3. 结论

变更百分比: {diff['change_percentage']}%

{'✅ 变更 < 30%，可自动更新 codemap' if diff['change_percentage'] < 30 else '⚠️ 变更 >= 30%，需要用户审核'}

---
详细文件列表:

core/:
"""
    for f in stats["core"]["files"]:
        report += f"  - {f}\n"

    report += "\nutils/:\n"
    for f in stats["utils"]["files"]:
        report += f"  - {f}\n"

    return report


if __name__ == "__main__":
    report = generate_report()
    output_file = Path(__file__).parent / "codemap-diff.txt"
    output_file.write_text(report, encoding="utf-8")
    print(f"报告已生成: {output_file}")
    print(report)
