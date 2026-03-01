"""text_utils 模块单元测试。

覆盖：
- condense_term_out: 短输出原样返回、LLM 提取、异常传播、metric 精确保留
- compress_task_desc: 标准压缩、缺失标题、回退、空输入
- _extract_section: 提取标题内容
"""

import pytest
from unittest.mock import MagicMock

from utils.text_utils import compress_task_desc, _extract_section, condense_term_out


# ============================================================
# condense_term_out
# ============================================================


class TestCondenseTermOut:
    """测试 condense_term_out() 前置 LLM 提取功能。"""

    def test_short_output_unchanged(self):
        """短输出（<8000 chars）原样返回，不调用 LLM。"""
        text = "Train shape: (1000, 10)\nValidation metric: 0.8500\n"
        mock_llm = MagicMock()

        result = condense_term_out(text, max_len=8000, llm_fn=mock_llm)

        assert result == text
        mock_llm.assert_not_called()

    def test_none_returns_empty(self):
        """None 输入返回空字符串。"""
        result = condense_term_out(None, llm_fn=MagicMock())
        assert result == ""

    def test_empty_returns_empty(self):
        """空字符串返回空字符串。"""
        result = condense_term_out("", llm_fn=MagicMock())
        assert result == ""

    def test_exact_limit_unchanged(self):
        """恰好等于限制时不调用 LLM。"""
        text = "x" * 8000
        mock_llm = MagicMock()

        result = condense_term_out(text, max_len=8000, llm_fn=mock_llm)

        assert result == text
        mock_llm.assert_not_called()

    def test_long_output_calls_llm(self):
        """超过限制时调用 LLM 提取结构化摘要。"""
        text = "x" * 10000
        llm_response = """=== EXECUTION SUMMARY ===

[STATUS]
success

[METRIC]
Validation metric: 0.129663
Metric name: rmse

[TRAINING]
- Data: train=1000x10, test=500x10
- Model: LightGBM
- CV: 5-fold
- Training: N/A
- Convergence: N/A

[WARNINGS]
None

[ERROR_TRACE]
None

[OUTPUT_FILES]
submission.csv: created"""

        mock_llm = MagicMock(return_value=llm_response)

        result = condense_term_out(text, max_len=8000, llm_fn=mock_llm)

        assert result == llm_response
        mock_llm.assert_called_once()
        # 验证 prompt 包含原始文本
        call_args = mock_llm.call_args[0][0]
        assert "x" * 100 in call_args

    def test_llm_failure_propagates(self):
        """LLM 异常直接向上传播，不做降级处理。"""
        text = "x" * 10000
        mock_llm = MagicMock(side_effect=RuntimeError("API call failed"))

        with pytest.raises(RuntimeError, match="API call failed"):
            condense_term_out(text, max_len=8000, llm_fn=mock_llm)

    def test_preserves_exact_metric(self):
        """LLM 返回的 metric 值与原文完全一致（无四舍五入）。"""
        metric_line = "Validation metric: 0.129663"
        text = "progress bar " * 2000 + metric_line + "\n" + "more output " * 500

        llm_response = f"""=== EXECUTION SUMMARY ===

[STATUS]
success

[METRIC]
{metric_line}
Metric name: rmse

[TRAINING]
N/A

[WARNINGS]
None

[ERROR_TRACE]
None

[OUTPUT_FILES]
submission.csv: created"""

        mock_llm = MagicMock(return_value=llm_response)
        result = condense_term_out(text, max_len=8000, llm_fn=mock_llm)

        assert "0.129663" in result
        assert metric_line in result

    def test_prompt_contains_template_sections(self):
        """LLM 收到的 prompt 包含固定格式模板的所有段。"""
        text = "x" * 10000
        mock_llm = MagicMock(return_value="summary")

        condense_term_out(text, max_len=8000, llm_fn=mock_llm)

        prompt = mock_llm.call_args[0][0]
        assert "[STATUS]" in prompt
        assert "[METRIC]" in prompt
        assert "[TRAINING]" in prompt
        assert "[ERROR_TRACE]" in prompt
        assert "[OUTPUT_FILES]" in prompt
        assert "EXECUTION SUMMARY" in prompt


# ============================================================
# compress_task_desc
# ============================================================


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


# ============================================================
# _extract_section
# ============================================================


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
