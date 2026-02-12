"""静态预验证模块单元测试。"""

import pytest
from utils.code_validator import validate_code, ValidationResult


class TestValidateCode:
    """validate_code() 测试。"""

    def test_valid_code(self):
        """完整的合法代码应通过验证。"""
        code = """
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("input/train.csv")
X_train, X_val, y_train, y_val = train_test_split(df.drop("target", axis=1), df["target"])

# 模型训练省略...
score = 0.85
print(f"Validation metric: {score}")

submission = pd.DataFrame({"id": range(100), "target": [0]*100})
submission.to_csv("submission/submission.csv", index=False)
"""
        result = validate_code(code)
        assert result.valid is True
        assert len(result.errors) == 0

    def test_syntax_error(self):
        """语法错误应被检测到。"""
        code = "def foo(\n    print('hello')"
        result = validate_code(code)
        assert result.valid is False
        assert any("SyntaxError" in e for e in result.errors)

    def test_missing_submission(self):
        """缺少 submission 输出应报错。"""
        code = """
import pandas as pd
df = pd.read_csv("input/train.csv")
score = 0.85
print(f"Validation metric: {score}")
"""
        result = validate_code(code)
        assert result.valid is False
        assert any("submission" in e.lower() for e in result.errors)

    def test_missing_metric_print(self):
        """缺少 metric 打印应产生警告。"""
        code = """
import pandas as pd
df = pd.read_csv("input/train.csv")
submission = pd.DataFrame({"id": range(100)})
submission.to_csv("submission/submission.csv", index=False)
"""
        result = validate_code(code)
        assert result.valid is True  # 仅警告，不阻止
        assert any("metric" in w.lower() for w in result.warnings)

    def test_double_normalization_warning(self):
        """ToTensor + /255 应产生警告。"""
        code = """
import torch
from torchvision import transforms

transform = transforms.Compose([transforms.ToTensor()])
data = data / 255.0

submission = pd.DataFrame()
submission.to_csv("submission/submission.csv")
print("Validation metric: 0.5")
"""
        result = validate_code(code)
        assert any("double normalization" in w.lower() for w in result.warnings)

    def test_empty_code(self):
        """空代码应被标记为无效。"""
        result = validate_code("")
        # 空代码 compile 会通过，但缺少 submission 和 metric
        assert result.valid is False
        assert any("submission" in e.lower() for e in result.errors)
