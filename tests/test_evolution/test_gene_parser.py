"""基因解析器单元测试。"""

import pytest

from core.evolution.gene_parser import (
    parse_solution_genes,
    validate_genes,
    merge_genes,
    REQUIRED_GENES,
)


class TestParseSolutionGenes:
    """测试 parse_solution_genes 函数。"""

    def test_parse_complete_genes(self):
        """测试提取完整 7 个基因块。"""
        code = """
# [SECTION: DATA]
import pandas as pd
train = pd.read_csv("train.csv")

# [SECTION: MODEL]
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

# [SECTION: LOSS]
# 使用默认分类损失

# [SECTION: OPTIMIZER]
# 不适用（树模型）

# [SECTION: REGULARIZATION]
model.max_depth = 10

# [SECTION: INITIALIZATION]
model.random_state = 42

# [SECTION: TRAINING_TRICKS]
model.fit(X_train, y_train)
        """

        genes = parse_solution_genes(code)

        assert len(genes) == 7
        assert "DATA" in genes
        assert "MODEL" in genes
        assert "LOSS" in genes
        assert "OPTIMIZER" in genes
        assert "REGULARIZATION" in genes
        assert "INITIALIZATION" in genes
        assert "TRAINING_TRICKS" in genes

        assert "import pandas" in genes["DATA"]
        assert "RandomForestClassifier" in genes["MODEL"]

    def test_parse_partial_genes(self):
        """测试提取部分基因块。"""
        code = """
# [SECTION: DATA]
import pandas as pd

# [SECTION: MODEL]
model = XGBoost()
        """

        genes = parse_solution_genes(code)

        assert len(genes) == 2
        assert "DATA" in genes
        assert "MODEL" in genes
        assert "LOSS" not in genes

    def test_parse_empty_code(self):
        """测试无基因块的代码。"""
        code = """
import pandas as pd
model = RandomForest()
        """

        genes = parse_solution_genes(code)

        assert len(genes) == 0

    def test_parse_with_whitespace(self):
        """测试标记前后有空格的情况。"""
        code = """
#  [ SECTION:  DATA  ]
import pandas as pd

#[SECTION:MODEL]
model = XGBoost()
        """

        genes = parse_solution_genes(code)

        # 注意：标记格式严格，不匹配空格变体
        # 只有 "#[SECTION:MODEL]" 格式能匹配
        assert len(genes) == 1
        assert "MODEL" in genes


class TestValidateGenes:
    """测试 validate_genes 函数。"""

    def test_validate_complete_genes(self):
        """测试验证完整基因块。"""
        genes = {gene: f"content_{gene}" for gene in REQUIRED_GENES}

        assert validate_genes(genes) is True

    def test_validate_missing_genes(self):
        """测试缺失基因块。"""
        genes = {
            "DATA": "...",
            "MODEL": "...",
            "LOSS": "...",
        }

        assert validate_genes(genes) is False

    def test_validate_empty_genes(self):
        """测试空基因字典。"""
        genes = {}

        assert validate_genes(genes) is False

    def test_validate_extra_genes(self):
        """测试包含额外基因块（应该通过）。"""
        genes = {gene: f"content_{gene}" for gene in REQUIRED_GENES}
        genes["EXTRA"] = "extra content"

        # 只要包含所有必需基因块，额外的不影响
        assert validate_genes(genes) is True


class TestMergeGenes:
    """测试 merge_genes 函数。"""

    def test_merge_simple(self):
        """测试简单合并（2 个基因块）。"""
        genes_a = {
            "DATA": "import pandas as pd",
            "MODEL": "RandomForest",
        }

        genes_b = {
            "DATA": "import numpy as np",
            "MODEL": "XGBoost",
        }

        # 完整 gene_plan（必须包含所有 7 个基因块）
        gene_plan = {gene: "A" for gene in REQUIRED_GENES}
        gene_plan["DATA"] = "A"
        gene_plan["MODEL"] = "B"

        # 补充缺失的基因块
        for gene in REQUIRED_GENES:
            if gene not in genes_a:
                genes_a[gene] = f"# {gene} from A"
            if gene not in genes_b:
                genes_b[gene] = f"# {gene} from B"

        code = merge_genes(genes_a, genes_b, gene_plan)

        # 验证合并结果
        assert "# [SECTION: DATA]" in code
        assert "# [SECTION: MODEL]" in code
        assert "import pandas as pd" in code  # 来自 A
        assert "XGBoost" in code  # 来自 B

    def test_merge_all_from_a(self):
        """测试所有基因块都来自 A。"""
        genes_a = {gene: f"content_A_{gene}" for gene in REQUIRED_GENES}
        genes_b = {gene: f"content_B_{gene}" for gene in REQUIRED_GENES}

        gene_plan = {gene: "A" for gene in REQUIRED_GENES}

        code = merge_genes(genes_a, genes_b, gene_plan)

        # 所有内容都应该来自 A
        for gene in REQUIRED_GENES:
            assert f"content_A_{gene}" in code
            assert f"content_B_{gene}" not in code

    def test_merge_all_from_b(self):
        """测试所有基因块都来自 B。"""
        genes_a = {gene: f"content_A_{gene}" for gene in REQUIRED_GENES}
        genes_b = {gene: f"content_B_{gene}" for gene in REQUIRED_GENES}

        gene_plan = {gene: "B" for gene in REQUIRED_GENES}

        code = merge_genes(genes_a, genes_b, gene_plan)

        # 所有内容都应该来自 B
        for gene in REQUIRED_GENES:
            assert f"content_B_{gene}" in code
            assert f"content_A_{gene}" not in code

    def test_merge_missing_gene_in_plan(self):
        """测试 gene_plan 缺少基因块。"""
        genes_a = {gene: f"content_A_{gene}" for gene in REQUIRED_GENES}
        genes_b = {gene: f"content_B_{gene}" for gene in REQUIRED_GENES}

        # 缺少 DATA 基因块
        gene_plan = {gene: "A" for gene in REQUIRED_GENES if gene != "DATA"}

        with pytest.raises(ValueError, match="gene_plan 缺少基因块: DATA"):
            merge_genes(genes_a, genes_b, gene_plan)

    def test_merge_missing_gene_in_parent_a(self):
        """测试父代 A 缺少基因块。"""
        genes_a = {
            gene: f"content_A_{gene}" for gene in REQUIRED_GENES if gene != "MODEL"
        }
        genes_b = {gene: f"content_B_{gene}" for gene in REQUIRED_GENES}

        gene_plan = {gene: "A" for gene in REQUIRED_GENES}

        with pytest.raises(ValueError, match="父代 A 缺少基因块: MODEL"):
            merge_genes(genes_a, genes_b, gene_plan)

    def test_merge_missing_gene_in_parent_b(self):
        """测试父代 B 缺少基因块。"""
        genes_a = {gene: f"content_A_{gene}" for gene in REQUIRED_GENES}
        genes_b = {
            gene: f"content_B_{gene}" for gene in REQUIRED_GENES if gene != "LOSS"
        }

        gene_plan = {gene: "A" for gene in REQUIRED_GENES}
        gene_plan["LOSS"] = "B"

        with pytest.raises(ValueError, match="父代 B 缺少基因块: LOSS"):
            merge_genes(genes_a, genes_b, gene_plan)

    def test_merge_invalid_source(self):
        """测试无效的 source 值。"""
        genes_a = {gene: f"content_A_{gene}" for gene in REQUIRED_GENES}
        genes_b = {gene: f"content_B_{gene}" for gene in REQUIRED_GENES}

        gene_plan = {gene: "A" for gene in REQUIRED_GENES}
        gene_plan["DATA"] = "C"  # 无效值

        with pytest.raises(ValueError, match="gene_plan\\[DATA\\] 必须为 'A' 或 'B'"):
            merge_genes(genes_a, genes_b, gene_plan)
