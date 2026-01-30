"""
LLM 响应解析工具模块。

提供从 LLM 响应中提取代码、文本等内容的工具函数。
"""

import re


def extract_code(response: str) -> str:
    """从 LLM 响应中提取第一个 Python 代码块。

    支持多种代码块标记格式：
    - ```python ... ```
    - ```py ... ```
    - ``` ... ``` (无语言标记)

    Args:
        response: LLM 响应文本

    Returns:
        提取的代码字符串（去除 markdown 标记），如果未找到则返回空字符串

    Examples:
        >>> extract_code("Here is code:\\n```python\\nprint('hello')\\n```")
        "print('hello')"
    """
    # 匹配 ```python ... ```, ```py ... ```, 或 ``` ... ```
    patterns = [
        r"```python\s*\n(.*?)```",
        r"```py\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            code = match.group(1).strip()
            return code

    # 未找到代码块
    return ""


def extract_text_up_to_code(response: str) -> str:
    """提取 LLM 响应中代码块之前的文本（通常是 plan/说明）。

    Args:
        response: LLM 响应文本

    Returns:
        代码块之前的文本，如果没有代码块则返回完整响应

    Examples:
        >>> extract_text_up_to_code("Plan: use XGBoost\\n```python\\ncode\\n```")
        "Plan: use XGBoost"
    """
    # 查找第一个代码块的位置
    code_block_pattern = r"```(?:python|py)?\s*\n"
    match = re.search(code_block_pattern, response)

    if match:
        # 返回代码块之前的文本
        return response[: match.start()].strip()
    else:
        # 没有代码块，返回完整响应
        return response.strip()


def trim_long_string(text: str, max_length: int = 1000) -> str:
    """截断长字符串并添加省略号。

    Args:
        text: 输入文本
        max_length: 最大长度（默认 1000）

    Returns:
        截断后的文本（如果超长则添加 "... (truncated)"）

    Examples:
        >>> trim_long_string("a" * 2000, max_length=100)
        "aaa...aaa... (truncated)"
    """
    if len(text) <= max_length:
        return text

    # 截断并添加标记
    truncated = text[:max_length]
    return f"{truncated}... (truncated)"
