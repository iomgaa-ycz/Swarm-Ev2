"""
utils/response.py 的单元测试。
"""

from utils.response import extract_code, extract_text_up_to_code, trim_long_string


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
