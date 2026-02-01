"""代码库架构分析器。

生成 codemap 差异报告和更新后的架构文档。
"""

import ast
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict


class CodebaseAnalyzer:
    """代码库结构分析器。"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.modules = {}
        self.dependencies = defaultdict(set)
        self.line_counts = {}
        self.exclude_dirs = {"Reference", ".venv", "venv", "__pycache__", ".git"}

    def analyze(self) -> Dict:
        """分析整个代码库。"""
        python_files = self._find_python_files()

        for file_path in python_files:
            self._analyze_file(file_path)

        return {
            "total_files": len(self.modules),
            "total_lines": sum(self.line_counts.values()),
            "modules": self.modules,
            "dependencies": {k: list(v) for k, v in self.dependencies.items()},
            "line_counts": self.line_counts,
            "stats_by_dir": self._get_stats_by_directory(),
        }

    def _find_python_files(self) -> List[Path]:
        """查找所有 Python 文件。"""
        files = []
        for py_file in self.project_root.rglob("*.py"):
            # 跳过排除目录
            if any(exclude in py_file.parts for exclude in self.exclude_dirs):
                continue
            # 跳过 workspace/working 中的临时文件
            if "workspace/working" in str(py_file):
                continue
            files.append(py_file)
        return files

    def _analyze_file(self, file_path: Path):
        """分析单个文件。"""
        try:
            content = file_path.read_text(encoding="utf-8")
            lines = len(content.split("\n"))

            # 提取相对模块路径
            rel_path = file_path.relative_to(self.project_root)
            module_name = str(rel_path).replace("/", ".").replace(".py", "")

            # 解析 AST
            tree = ast.parse(content, filename=str(file_path))
            imports = self._extract_imports(tree)

            self.modules[module_name] = {
                "file": str(rel_path),
                "lines": lines,
                "imports": imports,
            }
            self.line_counts[module_name] = lines

            # 记录依赖关系
            for imp in imports:
                if not imp.startswith("utils.") and not imp.startswith("core."):
                    continue  # 仅记录项目内部依赖
                self.dependencies[module_name].add(imp)

        except Exception as e:
            print(f"分析文件失败 {file_path}: {e}")

    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """提取导入模块列表。"""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        return list(set(imports))

    def _get_stats_by_directory(self) -> Dict[str, Dict]:
        """按目录统计。"""
        stats = defaultdict(lambda: {"files": 0, "lines": 0})

        for module, data in self.modules.items():
            # 提取顶层目录
            parts = module.split(".")
            if len(parts) > 1:
                top_dir = parts[0]
            else:
                top_dir = "root"

            stats[top_dir]["files"] += 1
            stats[top_dir]["lines"] += data["lines"]

        return dict(stats)

    def compare_with_previous(self, old_stats: Dict) -> Dict:
        """与旧版本对比。"""
        current_lines = sum(self.line_counts.values())
        old_lines = old_stats.get("total_lines", 0)

        diff_lines = current_lines - old_lines
        diff_pct = (diff_lines / old_lines * 100) if old_lines > 0 else 0

        return {
            "total_lines_old": old_lines,
            "total_lines_new": current_lines,
            "diff_lines": diff_lines,
            "diff_percentage": round(diff_pct, 2),
            "needs_approval": abs(diff_pct) > 30,
        }


def main():
    """主函数。"""
    project_root = Path(__file__).parent.parent
    analyzer = CodebaseAnalyzer(project_root)

    print("分析代码库结构...")
    results = analyzer.analyze()

    # 保存完整分析结果
    output_file = project_root / ".reports" / "codemap-analysis.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 分析完成: {output_file}")
    print(f"\n统计摘要:")
    print(f"  - 总文件数: {results['total_files']}")
    print(f"  - 总代码行数: {results['total_lines']}")
    print(f"\n目录统计:")
    for dir_name, stats in sorted(results["stats_by_dir"].items()):
        print(f"  - {dir_name:20s}: {stats['files']:3d} 文件, {stats['lines']:5d} 行")

    # 加载旧版本数据（从现有 codemap 估算）
    old_stats = {
        "total_lines": 5311,  # 从 architecture.md 中记录的数据
        "total_files": 26,
    }

    comparison = analyzer.compare_with_previous(old_stats)
    print(f"\n变更对比:")
    print(f"  - 旧版本: {comparison['total_lines_old']} 行")
    print(f"  - 新版本: {comparison['total_lines_new']} 行")
    print(f"  - 差异: {comparison['diff_lines']:+d} 行 ({comparison['diff_percentage']:+.1f}%)")
    print(f"  - 需要用户审批: {'是' if comparison['needs_approval'] else '否'}")

    # 保存差异报告
    diff_file = project_root / ".reports" / "codemap-diff.txt"
    with open(diff_file, "w", encoding="utf-8") as f:
        f.write(f"# Codemap 差异报告\n")
        f.write(f"生成时间: {Path(__file__).stat().st_mtime}\n\n")
        f.write(f"## 总体变更\n")
        f.write(f"- 旧版本: {comparison['total_lines_old']} 行 ({old_stats['total_files']} 文件)\n")
        f.write(f"- 新版本: {comparison['total_lines_new']} 行 ({results['total_files']} 文件)\n")
        f.write(f"- 差异: {comparison['diff_lines']:+d} 行 ({comparison['diff_percentage']:+.1f}%)\n\n")
        f.write(f"## 目录统计\n")
        for dir_name, stats in sorted(results["stats_by_dir"].items()):
            f.write(f"- {dir_name}: {stats['files']} 文件, {stats['lines']} 行\n")

    print(f"\n✓ 差异报告已保存: {diff_file}")


if __name__ == "__main__":
    main()
