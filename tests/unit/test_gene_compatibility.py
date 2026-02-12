"""基因兼容性检查模块单元测试。"""

import importlib.util
import sys
from pathlib import Path

import pytest

# 直接从文件加载模块，避免触发 core.evolution.__init__.py 的 sentence_transformers 依赖
_spec = importlib.util.spec_from_file_location(
    "gene_compatibility",
    Path(__file__).resolve().parents[2]
    / "core"
    / "evolution"
    / "gene_compatibility.py",
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["gene_compatibility"] = _mod
_spec.loader.exec_module(_mod)

extract_imports = _mod.extract_imports
detect_framework = _mod.detect_framework
check_gene_compatibility = _mod.check_gene_compatibility
CompatibilityResult = _mod.CompatibilityResult


class TestExtractImports:
    """extract_imports() 测试。"""

    def test_import_statement(self):
        """标准 import 语句。"""
        code = "import torch\nimport numpy as np"
        result = extract_imports(code)
        assert "torch" in result
        assert "numpy" in result

    def test_from_import(self):
        """from ... import 语句。"""
        code = "from sklearn.model_selection import train_test_split"
        result = extract_imports(code)
        assert "sklearn" in result

    def test_nested_import(self):
        """嵌套 import。"""
        code = "import torch.nn as nn\nfrom torch.optim import Adam"
        result = extract_imports(code)
        assert "torch" in result

    def test_empty_code(self):
        """空代码。"""
        assert extract_imports("") == set()

    def test_non_import_lines(self):
        """非 import 行应被忽略。"""
        code = "x = 1\nprint('hello')\n# import fake"
        assert extract_imports(code) == set()


class TestDetectFramework:
    """detect_framework() 测试。"""

    def test_torch(self):
        """检测 PyTorch 框架。"""
        code = "import torch\nfrom torchvision import transforms"
        assert detect_framework(code) == "torch"

    def test_sklearn(self):
        """检测 sklearn 框架。"""
        code = "from sklearn.ensemble import RandomForestClassifier"
        assert detect_framework(code) == "sklearn"

    def test_xgboost(self):
        """xgboost 应归类为 sklearn 组。"""
        code = "import xgboost as xgb"
        assert detect_framework(code) == "sklearn"

    def test_tensorflow(self):
        """检测 TensorFlow。"""
        code = "import tensorflow as tf"
        assert detect_framework(code) == "tensorflow"

    def test_no_framework(self):
        """纯 numpy/pandas 无框架。"""
        code = "import numpy as np\nimport pandas as pd"
        assert detect_framework(code) is None


class TestCheckGeneCompatibility:
    """check_gene_compatibility() 测试。"""

    def test_compatible_same_framework(self):
        """相同框架应兼容。"""
        parent_a = "import torch\nfrom torch import nn"
        parent_b = "import torch\nimport torchvision"
        result = check_gene_compatibility(
            parent_a_code=parent_a,
            parent_b_code=parent_b,
            genes_a={"DATA": "import torch", "MODEL": "import torch.nn"},
            genes_b={"DATA": "import torch", "MODEL": "import torchvision"},
            gene_plan_choices={"DATA": "A", "MODEL": "B"},
        )
        assert result.compatible is True
        assert len(result.conflicts) == 0

    def test_framework_conflict(self):
        """不同框架应检测冲突并注入警告。"""
        parent_a = "import torch\nfrom torch import nn"
        parent_b = "from sklearn.ensemble import RandomForestClassifier"
        result = check_gene_compatibility(
            parent_a_code=parent_a,
            parent_b_code=parent_b,
            genes_a={"DATA": "import torch", "MODEL": "import torch.nn"},
            genes_b={
                "DATA": "from sklearn import preprocessing",
                "MODEL": "from sklearn.ensemble import RF",
            },
            gene_plan_choices={"DATA": "A", "MODEL": "B"},
        )
        assert result.compatible is True  # 仍然尝试，让 LLM 处理
        assert result.action == "inject_warning"
        assert any("Framework" in c or "framework" in c for c in result.conflicts)

    def test_empty_gene_block(self):
        """空基因块应被检测。"""
        result = check_gene_compatibility(
            parent_a_code="import torch",
            parent_b_code="import torch",
            genes_a={"DATA": "import torch", "MODEL": ""},
            genes_b={"DATA": "import torch", "MODEL": "# (no code)"},
            gene_plan_choices={"DATA": "A", "MODEL": "A"},
        )
        assert any("empty" in c for c in result.conflicts)
        assert result.action == "inject_warning"

    def test_no_conflicts(self):
        """完全兼容的情况。"""
        result = check_gene_compatibility(
            parent_a_code="import numpy",
            parent_b_code="import numpy",
            genes_a={"DATA": "import pandas", "MODEL": "model = None"},
            genes_b={"DATA": "import pandas", "MODEL": "model = None"},
            gene_plan_choices={"DATA": "A", "MODEL": "B"},
        )
        assert result.compatible is True
        assert result.action == "proceed"
        assert len(result.conflicts) == 0
