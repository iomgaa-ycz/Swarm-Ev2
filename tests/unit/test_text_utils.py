"""text_utils 模块单元测试。

覆盖：
- condense_term_out: 短输出原样返回、LLM 提取、异常传播、metric 精确保留
- clean_task_desc: 图片移除、section 移除、代码块清理、长度截断
- _extract_section: 提取标题内容
"""

import pytest
from unittest.mock import MagicMock

from utils.text_utils import clean_task_desc, _extract_section, condense_term_out


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
# clean_task_desc
# ============================================================


class TestCleanTaskDesc:
    """clean_task_desc 函数测试。"""

    def test_removes_image_tags(self):
        """测试移除 Markdown 图片标签。"""
        desc = "## Description\n\nSome text ![image](http://example.com/img.png) more text.\n"
        result = clean_task_desc(desc)
        assert "![image]" not in result
        assert "Some text" in result
        assert "more text" in result

    def test_removes_html_img_tags(self):
        """测试移除 HTML img 标签。"""
        desc = '## Description\n\nText <img src="foo.png" alt="bar"> end.\n'
        result = clean_task_desc(desc)
        assert "<img" not in result
        assert "Text" in result

    def test_removes_prizes_section(self):
        """测试移除 Prizes section。"""
        desc = """## Description

Task details here.

## Prizes

First place: $10000
Second place: $5000

## Evaluation

RMSE is the metric.
"""
        result = clean_task_desc(desc)
        assert "$10000" not in result
        assert "Task details" in result
        assert "RMSE" in result

    def test_removes_citation_subsection(self):
        """测试移除 Citation subsection。"""
        desc = """## Description

Task details.

### Citation

@article{foo, title={bar}}

## Evaluation

AUC metric.
"""
        result = clean_task_desc(desc)
        assert "@article" not in result
        assert "Task details" in result
        assert "AUC" in result

    def test_strips_eval_code_blocks(self):
        """测试移除 Evaluation 中的代码块但保留文字说明。"""
        desc = """## Description

Predict prices.

## Evaluation

RMSE is used for evaluation.

```python
import numpy as np
rmse = np.sqrt(np.mean((y_true - y_pred)**2))
```

Lower is better.
"""
        result = clean_task_desc(desc)
        assert "RMSE is used" in result
        assert "import numpy" not in result
        assert "Lower is better" in result

    def test_preserves_description_and_dataset(self):
        """测试保留 Description 和 Dataset Description 完整内容。"""
        desc = """## Description

This is a competition about whale identification.

Additional paragraph with important details about the task.

## Dataset Description

Train contains 9850 images. Test contains 1000 images.
"""
        result = clean_task_desc(desc)
        assert "whale identification" in result
        assert "Additional paragraph" in result
        assert "9850 images" in result

    def test_truncation_at_6000_chars(self):
        """测试超长描述截断到 6000 chars。"""
        desc = "## Description\n\n" + "x" * 7000
        result = clean_task_desc(desc)
        assert len(result) <= 6000 + len("\n... (truncated)")
        assert result.endswith("... (truncated)")

    def test_short_desc_no_truncation(self):
        """测试短描述不截断。"""
        desc = "## Description\n\nShort task.\n"
        result = clean_task_desc(desc)
        assert "truncated" not in result
        assert "Short task" in result

    def test_empty_input(self):
        """测试空输入。"""
        result = clean_task_desc("")
        assert result == ""

    def test_plain_text_passthrough(self):
        """测试无 Markdown 标题的纯文本直接返回。"""
        text = "Just plain text without headers."
        result = clean_task_desc(text)
        assert result == text


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
