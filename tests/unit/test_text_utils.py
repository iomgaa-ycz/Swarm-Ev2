"""text_utils 模块单元测试。"""

from utils.text_utils import compress_task_desc, _extract_section


class TestCompressTaskDesc:
    """compress_task_desc 函数测试。"""

    def test_compress_standard_desc(self):
        """测试标准 Kaggle 格式的压缩。"""
        full_desc = """## Description

This is a competition about predicting house prices. You need to build a model that predicts sale prices based on various features.

Additional details here.

## Evaluation

RMSE (Root Mean Squared Error) is used to evaluate submissions.

Lower is better.

## Dataset Description

Some data info.

### Submission File

Your submission should be a CSV file with the following format:

```csv
Id,SalePrice
1,200000
2,150000
```

## Additional Info

Some other info.
"""
        result = compress_task_desc(full_desc)

        # 验证包含三个部分
        assert "**Task**:" in result
        assert "**Metric**:" in result
        assert "**Format**:" in result

        # 验证内容正确
        assert "house prices" in result
        assert "RMSE" in result
        assert "```csv" in result

        # 验证长度合理（远小于原文）
        assert len(result) < len(full_desc)
        assert len(result) < 1000  # 预期 ~500B

    def test_compress_missing_sections(self):
        """测试缺少某些标题时的处理。"""
        # 只有 Description
        desc_only = """## Description

A simple task description.

## Other Section

Other content.
"""
        result = compress_task_desc(desc_only)
        assert "**Task**:" in result
        assert "**Metric**:" not in result
        assert "**Format**:" not in result

    def test_compress_fallback(self):
        """测试完全无法解析时的回退。"""
        weird_format = "This is just plain text without any markdown headers."

        result = compress_task_desc(weird_format)

        # 应该回退到截取前 500 字符
        assert result.endswith("...")
        assert len(result) <= 504  # 500 + "..."

    def test_compress_empty_input(self):
        """测试空输入。"""
        result = compress_task_desc("")
        assert result == "..."


class TestExtractSection:
    """_extract_section 函数测试。"""

    def test_extract_existing_section(self):
        """测试提取存在的标题。"""
        text = """## Header One

Content one.

## Header Two

Content two.
"""
        result = _extract_section(text, "## Header One")
        assert result == "Content one."

    def test_extract_nested_section(self):
        """测试提取嵌套标题。"""
        text = """## Parent

Parent content.

### Child

Child content.

## Next Parent

Next content.
"""
        result = _extract_section(text, "### Child")
        assert result == "Child content."

    def test_extract_nonexistent_section(self):
        """测试提取不存在的标题。"""
        text = """## Header

Content.
"""
        result = _extract_section(text, "## Nonexistent")
        assert result is None

    def test_extract_last_section(self):
        """测试提取最后一个标题（无后续标题）。"""
        text = """## First

First content.

## Last

Last content here.
"""
        result = _extract_section(text, "## Last")
        assert result == "Last content here."
