"""
utils/response.py 的单元测试。
"""

import pytest
from utils.response import (
    extract_code,
    extract_text_up_to_code,
    trim_long_string,
    extract_review,
)


class TestExtractCode:
    """测试 extract_code 函数。"""

    def test_extract_code_with_python_marker(self):
        """测试提取带 python 标记的代码块。"""
        response = "Here is the code:\n```python\nprint('hello')\n```"
        code = extract_code(response)
        assert code == "print('hello')"

    def test_extract_code_with_py_marker(self):
        """测试提取带 py 标记的代码块。"""
        response = "```py\nx = 1\ny = 2\n```"
        code = extract_code(response)
        assert code == "x = 1\ny = 2"

    def test_extract_code_without_language_marker(self):
        """测试提取无语言标记的代码块。"""
        response = "```\ndef foo():\n    pass\n```"
        code = extract_code(response)
        assert code == "def foo():\n    pass"

    def test_extract_code_multiple_blocks(self):
        """测试多个代码块时取第一个。"""
        response = "```python\ncode1\n```\nSome text\n```python\ncode2\n```"
        code = extract_code(response)
        assert code == "code1"

    def test_extract_code_no_code_block(self):
        """测试无代码块时返回空字符串。"""
        response = "This is just text without code"
        code = extract_code(response)
        assert code == ""


class TestExtractTextUpToCode:
    """测试 extract_text_up_to_code 函数。"""

    def test_extract_text_before_code(self):
        """测试提取代码块之前的文本。"""
        response = "Plan: use XGBoost\n```python\nimport xgboost\n```"
        text = extract_text_up_to_code(response)
        assert text == "Plan: use XGBoost"

    def test_extract_text_no_code_block(self):
        """测试无代码块时返回完整响应。"""
        response = "This is a full response without code"
        text = extract_text_up_to_code(response)
        assert text == response

    def test_extract_text_empty_before_code(self):
        """测试代码块之前为空。"""
        response = "```python\ncode\n```"
        text = extract_text_up_to_code(response)
        assert text == ""


class TestTrimLongString:
    """测试 trim_long_string 函数。"""

    def test_trim_short_string(self):
        """测试短字符串不被截断。"""
        text = "short"
        result = trim_long_string(text, max_length=100)
        assert result == text

    def test_trim_long_string(self):
        """测试长字符串被截断。"""
        text = "a" * 2000
        result = trim_long_string(text, max_length=100)
        assert len(result) < len(text)
        assert "... (truncated)" in result

    def test_trim_exact_length(self):
        """测试刚好等于最大长度。"""
        text = "a" * 100
        result = trim_long_string(text, max_length=100)
        assert result == text


class TestExtractReview:
    """测试 extract_review 函数。"""

    def test_extract_review_json_block(self):
        """测试从 ```json ``` 代码块提取。"""
        text = """分析结果如下：
```json
{"is_bug": false, "metric": 0.85, "summary": "模型训练成功"}
```
"""
        result = extract_review(text)
        assert result["is_bug"] is False
        assert result["metric"] == 0.85
        assert result["summary"] == "模型训练成功"

    def test_extract_review_generic_block(self):
        """测试从无语言标记的代码块提取。"""
        text = """结果：
```
{"is_bug": true, "metric": null, "summary": "执行失败"}
```
"""
        result = extract_review(text)
        assert result["is_bug"] is True
        assert result["metric"] is None
        assert result["summary"] == "执行失败"

    def test_extract_review_bare_json(self):
        """测试提取裸 JSON 对象。"""
        text = '结果: {"is_bug": false, "metric": 0.92, "lower_is_better": false}'
        result = extract_review(text)
        assert result["is_bug"] is False
        assert result["metric"] == 0.92
        assert result["lower_is_better"] is False

    def test_extract_review_multiple_candidates(self):
        """测试多个候选时选择第一个有效的。"""
        text = """
```json
invalid json here
```

```json
{"is_bug": false, "metric": 0.75}
```
"""
        result = extract_review(text)
        assert result["metric"] == 0.75

    def test_extract_review_no_json_raises(self):
        """测试无 JSON 时抛出 ValueError。"""
        text = "这是一段没有 JSON 的文本"
        with pytest.raises(ValueError, match="无法从文本中提取 JSON"):
            extract_review(text)

    def test_extract_review_invalid_json_raises(self):
        """测试 JSON 解析失败时抛出 ValueError。"""
        text = """
```json
{invalid: json, syntax}
```
"""
        with pytest.raises(ValueError, match="JSON 解析失败"):
            extract_review(text)

    def test_extract_review_with_nested_braces(self):
        """测试包含嵌套大括号的 JSON。"""
        text = """
```json
{"is_bug": false, "details": {"accuracy": 0.95}, "metric": 0.95}
```
"""
        result = extract_review(text)
        assert result["is_bug"] is False
        assert result["details"]["accuracy"] == 0.95
        assert result["metric"] == 0.95
