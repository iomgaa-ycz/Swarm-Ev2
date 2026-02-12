"""基因兼容性检查模块。

在 merge 操作发送给 LLM 前，检测基因块间的明显不兼容。
当检测到框架冲突时，注入警告到 gene_plan 中让 LLM 自行处理。
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


# 框架互斥组：同一组内的包不应混用
FRAMEWORK_GROUPS: Dict[str, Set[str]] = {
    "torch": {"torch", "torchvision", "timm"},
    "sklearn": {"sklearn", "xgboost", "lightgbm", "catboost"},
    "tensorflow": {"tensorflow", "keras", "tf"},
}


@dataclass
class CompatibilityResult:
    """兼容性检查结果。

    Attributes:
        compatible: 是否兼容（有冲突时仍可能为 True，依赖 LLM 处理）
        conflicts: 冲突描述列表
        action: 处理策略 ("proceed" | "inject_warning")
    """

    compatible: bool
    conflicts: List[str] = field(default_factory=list)
    action: str = "proceed"


def extract_imports(code: str) -> Set[str]:
    """从代码片段中提取 import 的包名。

    Args:
        code: Python 代码字符串

    Returns:
        包名集合（如 {"torch", "sklearn", "numpy"}）
    """
    packages: Set[str] = set()
    for line in code.split("\n"):
        line = line.strip()
        # import torch / import torch.nn as nn
        m = re.match(r"^import\s+([\w.]+)", line)
        if m:
            packages.add(m.group(1).split(".")[0])
        # from torch import nn / from sklearn.model_selection import ...
        m = re.match(r"^from\s+([\w.]+)\s+import", line)
        if m:
            packages.add(m.group(1).split(".")[0])
    return packages


def detect_framework(code: str) -> Optional[str]:
    """检测代码片段使用的主框架。

    Args:
        code: Python 代码字符串

    Returns:
        "torch" | "sklearn" | "tensorflow" | None
    """
    imports = extract_imports(code)
    for framework, keywords in FRAMEWORK_GROUPS.items():
        if imports & keywords:
            return framework
    return None


def check_gene_compatibility(
    parent_a_code: str,
    parent_b_code: str,
    genes_a: Dict[str, str],
    genes_b: Dict[str, str],
    gene_plan_choices: Dict[str, str],
) -> CompatibilityResult:
    """检查基因交叉计划的兼容性。

    检查项:
    1. 空基因块检测
    2. DATA-MODEL 框架一致性（选中的 DATA 和 MODEL 是否使用相同框架）
    3. 全局框架冲突（两个父代是否使用不同框架）

    Args:
        parent_a_code: 父代 A 完整代码
        parent_b_code: 父代 B 完整代码
        genes_a: 父代 A 的基因块字典
        genes_b: 父代 B 的基因块字典
        gene_plan_choices: 基因选择方案 {"DATA": "A", "MODEL": "B", ...}

    Returns:
        CompatibilityResult
    """
    conflicts: List[str] = []

    # [检查 1] 空基因块
    for gene, choice in gene_plan_choices.items():
        source = genes_a if choice == "A" else genes_b
        code_block = source.get(gene, "")
        if not code_block or code_block.strip() in ("", "# (no code)"):
            conflicts.append(f"{gene} from parent_{choice} is empty")

    # [检查 2] DATA-MODEL 框架一致性
    data_choice = gene_plan_choices.get("DATA", "A")
    model_choice = gene_plan_choices.get("MODEL", "A")

    data_code = (genes_a if data_choice == "A" else genes_b).get("DATA", "")
    model_code = (genes_a if model_choice == "A" else genes_b).get("MODEL", "")

    data_framework = detect_framework(data_code)
    model_framework = detect_framework(model_code)

    if data_framework and model_framework and data_framework != model_framework:
        conflicts.append(
            f"Framework mismatch: DATA uses {data_framework} (parent_{data_choice}), "
            f"MODEL uses {model_framework} (parent_{model_choice})"
        )

    # [检查 3] 全局框架一致性
    framework_a = detect_framework(parent_a_code)
    framework_b = detect_framework(parent_b_code)

    if framework_a and framework_b and framework_a != framework_b:
        conflicts.append(
            f"Parents use different frameworks: A={framework_a}, B={framework_b}"
        )

    # 决策逻辑
    if not conflicts:
        return CompatibilityResult(compatible=True)

    # 有冲突时注入警告让 LLM 自行选择与修改
    return CompatibilityResult(
        compatible=True,
        conflicts=conflicts,
        action="inject_warning",
    )
