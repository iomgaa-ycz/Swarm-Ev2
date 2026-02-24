"""基因解析器模块。

提供 Solution 代码的基因块提取、验证和合并功能。
基因方案 V6：4 基因（DATA, MODEL, TRAIN, POSTPROCESS）。
"""

import random
import re
from typing import TYPE_CHECKING, Dict, Tuple

if TYPE_CHECKING:
    from core.state.node import Node

# 4 个必需基因块（V6 标准）
REQUIRED_GENES = [
    "DATA",
    "MODEL",
    "TRAIN",
    "POSTPROCESS",
]

# 每个基因的可变异子方面（用于精细化变异控制）
GENE_SUB_ASPECTS: Dict[str, list] = {
    "DATA": ["feature_engineering", "data_cleaning", "augmentation", "encoding"],
    "MODEL": ["architecture", "loss_function", "optimizer", "regularization"],
    "TRAIN": ["cv_strategy", "early_stopping", "lr_schedule", "epochs"],
    "POSTPROCESS": ["ensemble", "threshold", "tta"],
}


def parse_solution_genes(code: str) -> Dict[str, str]:
    """解析解决方案代码中的基因标记。

    解析 `# [SECTION: NAME]` 标记，将代码分割为不同的基因组件。

    Args:
        code: Python 代码字符串

    Returns:
        基因字典，格式为 {"SECTION_NAME": "code_block"}

    时间复杂度: O(n)，其中 n 为代码长度

    示例:
        >>> code = '''
        ... # [SECTION: DATA]
        ... import pandas as pd
        ... # [SECTION: MODEL]
        ... model = RandomForest()
        ... '''
        >>> genes = parse_solution_genes(code)
        >>> genes
        {'DATA': 'import pandas as pd', 'MODEL': 'model = RandomForest()'}

    注意:
        - 使用简单的单层格式（Swarm-Evo 标准）
        - 代码块内可包含 `# [FIXED]` 和 `# [EVOLVABLE]` 注释标记
        - LLM 通过 Prompt 约束识别并遵守修改规则
    """
    genes: Dict[str, str] = {}

    # 匹配 # [SECTION: NAME] 格式
    pattern = re.compile(r"^#\s*\[SECTION:\s*(\w+)\]", re.MULTILINE)
    matches = list(pattern.finditer(code))

    for i, match in enumerate(matches):
        section_name = match.group(1)
        start_idx = match.end()

        # 确定结束位置（下一个 SECTION 或代码末尾）
        if i + 1 < len(matches):
            end_idx = matches[i + 1].start()
        else:
            end_idx = len(code)

        content = code[start_idx:end_idx].strip()
        genes[section_name] = content

    return genes


def validate_genes(genes: Dict[str, str]) -> bool:
    """验证基因字典是否包含所有必需的基因块。

    Args:
        genes: 基因字典（由 parse_solution_genes 返回）

    Returns:
        True 表示所有必需基因块都存在，False 表示有缺失

    时间复杂度: O(1)（常数级别，4 个基因块）

    示例:
        >>> genes = {"DATA": "...", "MODEL": "..."}
        >>> validate_genes(genes)
        False  # 缺失 TRAIN, POSTPROCESS

        >>> complete_genes = {g: "..." for g in REQUIRED_GENES}
        >>> validate_genes(complete_genes)
        True
    """
    for required_gene in REQUIRED_GENES:
        if required_gene not in genes:
            return False
    return True


def merge_genes(
    genes_a: Dict[str, str],
    genes_b: Dict[str, str],
    gene_plan: Dict[str, str],
) -> str:
    """按基因交叉计划合并两个基因方案。

    根据 gene_plan 指定每个基因块来自哪个父代（A 或 B）。

    Args:
        genes_a: 父代 A 的基因字典
        genes_b: 父代 B 的基因字典
        gene_plan: 交叉计划，格式为 {"GENE_NAME": "A" | "B"}

    Returns:
        合并后的完整代码字符串

    Raises:
        ValueError: 如果 gene_plan 中指定的基因块在对应父代中不存在

    时间复杂度: O(k)，其中 k 为基因块数量（通常为 4）

    示例:
        >>> genes_a = {"DATA": "import pandas", "MODEL": "RandomForest"}
        >>> genes_b = {"DATA": "import numpy", "MODEL": "XGBoost"}
        >>> plan = {"DATA": "A", "MODEL": "B"}
        >>> code = merge_genes(genes_a, genes_b, plan)
        >>> "import pandas" in code
        True
        >>> "XGBoost" in code
        True

    注意:
        - 生成的代码保留 `# [SECTION: NAME]` 标记
        - 基因块按 REQUIRED_GENES 顺序排列
    """
    merged_sections = []

    for gene_name in REQUIRED_GENES:
        # 检查 gene_plan 中是否指定了此基因块
        if gene_name not in gene_plan:
            raise ValueError(f"gene_plan 缺少基因块: {gene_name}")

        source = gene_plan[gene_name]

        # 根据 source 选择父代
        if source == "A":
            if gene_name not in genes_a:
                raise ValueError(f"父代 A 缺少基因块: {gene_name}")
            gene_content = genes_a[gene_name]
        elif source == "B":
            if gene_name not in genes_b:
                raise ValueError(f"父代 B 缺少基因块: {gene_name}")
            gene_content = genes_b[gene_name]
        else:
            raise ValueError(
                f"gene_plan[{gene_name}] 必须为 'A' 或 'B'，实际为: {source}"
            )

        # 构建基因块（保留标记）
        section_str = f"# [SECTION: {gene_name}]\n{gene_content}\n"
        merged_sections.append(section_str)

    # 合并所有基因块
    return "\n".join(merged_sections)


def select_mutation_target(
    node: "Node", stub_threshold: int = 20
) -> Tuple[str, str]:
    """从节点中随机选择一个非 stub 的基因位点和子方面（用于 mutate 目标选择）。

    内容 strip 后长度 < stub_threshold 视为 stub（如 "# Not applicable..."）。
    选择基因后，再从该基因的 GENE_SUB_ASPECTS 中随机选择一个子方面。

    Args:
        node: 待变异的节点
        stub_threshold: 内容长度阈值，低于此值视为 stub（默认 20）

    Returns:
        (gene, aspect) 元组：随机选中的非 stub 基因位点名称和子方面；
        若所有基因均为 stub，则 fallback 到随机选择
    """
    candidates = [
        gene for gene in REQUIRED_GENES
        if len((node.genes.get(gene) or "").strip()) >= stub_threshold
    ]
    if candidates:
        gene = random.choice(candidates)
    else:
        # fallback: 所有基因均为 stub（极端情况），随机选一个
        gene = random.choice(REQUIRED_GENES)

    aspect = random.choice(GENE_SUB_ASPECTS[gene])
    return gene, aspect
