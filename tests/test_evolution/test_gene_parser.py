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
        """测试提取完整 4 个基因块。"""
        code = """
# [SECTION: DATA]
import pandas as pd
train = pd.read_csv("train.csv")

# [SECTION: MODEL]
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, max_depth=10)

# [SECTION: TRAIN]
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_train, y_train, cv=5)

# [SECTION: POSTPROCESS]
predictions = model.predict(X_test)
submission.to_csv('./submission/submission.csv', index=False)
        """

        genes = parse_solution_genes(code)

        assert len(genes) == 4
        assert "DATA" in genes
        assert "MODEL" in genes
        assert "TRAIN" in genes
        assert "POSTPROCESS" in genes

        assert "import pandas" in genes["DATA"]
        assert "RandomForestClassifier" in genes["MODEL"]
        assert "cross_val_score" in genes["TRAIN"]
        assert "predict" in genes["POSTPROCESS"]

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
        assert "TRAIN" not in genes

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

    def test_parse_sklearn_solution(self):
        """测试 sklearn 方案的基因解析。"""
        code = """
# [SECTION: DATA]
import pandas as pd
train = pd.read_csv("train.csv")
X = train.drop("target", axis=1)
y = train["target"]

# [SECTION: MODEL]
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1)

# [SECTION: TRAIN]
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)
model.fit(X, y)

# [SECTION: POSTPROCESS]
test = pd.read_csv("test.csv")
predictions = model.predict(test)
pd.DataFrame({"id": test["id"], "target": predictions}).to_csv("submission.csv", index=False)
        """
        genes = parse_solution_genes(code)
        assert len(genes) == 4
        assert "GradientBoostingClassifier" in genes["MODEL"]

    def test_parse_dl_solution(self):
        """测试 DL 方案的基因解析（loss/optimizer 在 MODEL 内）。"""
        code = """
# [SECTION: DATA]
from torch.utils.data import Dataset, DataLoader
train_dataset = CustomDataset(train_df)
train_loader = DataLoader(train_dataset, batch_size=32)

# [SECTION: MODEL]
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 10)
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# [SECTION: TRAIN]
for epoch in range(10):
    for batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(batch), labels)
        loss.backward()
        optimizer.step()

# [SECTION: POSTPROCESS]
model.eval()
predictions = model(test_data).argmax(dim=1)
        """
        genes = parse_solution_genes(code)
        assert len(genes) == 4
        assert "CrossEntropyLoss" in genes["MODEL"]
        assert "Adam" in genes["MODEL"]


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
        """测试简单合并。"""
        genes_a = {gene: f"content_A_{gene}" for gene in REQUIRED_GENES}
        genes_b = {gene: f"content_B_{gene}" for gene in REQUIRED_GENES}

        gene_plan = {gene: "A" for gene in REQUIRED_GENES}
        gene_plan["DATA"] = "A"
        gene_plan["MODEL"] = "B"

        code = merge_genes(genes_a, genes_b, gene_plan)

        # 验证合并结果
        assert "# [SECTION: DATA]" in code
        assert "# [SECTION: MODEL]" in code
        assert "content_A_DATA" in code  # 来自 A
        assert "content_B_MODEL" in code  # 来自 B

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
            gene: f"content_B_{gene}" for gene in REQUIRED_GENES if gene != "TRAIN"
        }

        gene_plan = {gene: "A" for gene in REQUIRED_GENES}
        gene_plan["TRAIN"] = "B"

        with pytest.raises(ValueError, match="父代 B 缺少基因块: TRAIN"):
            merge_genes(genes_a, genes_b, gene_plan)

    def test_merge_invalid_source(self):
        """测试无效的 source 值。"""
        genes_a = {gene: f"content_A_{gene}" for gene in REQUIRED_GENES}
        genes_b = {gene: f"content_B_{gene}" for gene in REQUIRED_GENES}

        gene_plan = {gene: "A" for gene in REQUIRED_GENES}
        gene_plan["DATA"] = "C"  # 无效值

        with pytest.raises(ValueError, match="gene_plan\\[DATA\\] 必须为 'A' 或 'B'"):
            merge_genes(genes_a, genes_b, gene_plan)
