"""代码静态预验证模块。

在 subprocess 执行前，用轻量 Python 检查拦截明显的代码缺陷。
参考 AIDE utils/response.py 的 is_valid_python_script(compile(...)) 检查。
"""

import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class ValidationResult:
    """验证结果。

    Attributes:
        valid: 是否通过验证（errors 为空则 valid=True）
        errors: 错误列表（必须修复）
        warnings: 警告列表（建议修复）
    """

    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def validate_code(code: str) -> ValidationResult:
    """静态预验证（不执行代码）。

    检查项:
    1. 语法检查: compile()
    2. Submission 输出检查: 代码是否包含 to_csv(...submission...)
    3. Metric 输出检查: 代码是否包含 print(...Validation metric...)
    4. 常见陷阱检测: ToTensor + /255, 等

    Args:
        code: Python 代码字符串

    Returns:
        ValidationResult 对象
    """
    errors: List[str] = []
    warnings: List[str] = []

    # 1. 语法检查（参考 AIDE is_valid_python_script）
    try:
        compile(code, "solution.py", "exec")
    except SyntaxError as e:
        errors.append(f"SyntaxError at line {e.lineno}: {e.msg}")
        # 语法错误直接返回，后续检查无意义
        return ValidationResult(valid=False, errors=errors)

    # 2. Submission 输出检查
    if "submission" not in code.lower() or "to_csv" not in code:
        errors.append(
            "Missing submission output: code must save predictions to "
            "submission.csv using to_csv()"
        )

    # 3. Metric 输出检查
    metric_patterns = [
        r"[Vv]alidation metric",
        r"print\(.*metric",
        r"print\(.*score",
    ]
    if not any(re.search(p, code) for p in metric_patterns):
        warnings.append(
            "Missing metric print: code should print 'Validation metric: {value}'"
        )

    # 4. 常见陷阱检测
    if "ToTensor()" in code and "/ 255" in code:
        warnings.append(
            "Possible double normalization: ToTensor() already normalizes "
            "to [0,1]. Remove '/ 255' if using ToTensor()."
        )

    is_valid = len(errors) == 0
    return ValidationResult(valid=is_valid, errors=errors, warnings=warnings)
