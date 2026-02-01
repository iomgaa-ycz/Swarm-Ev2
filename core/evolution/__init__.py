"""进化机制子系统。

提供基因解析、经验池等进化算法核心功能。
"""

from .gene_parser import (
    parse_solution_genes,
    validate_genes,
    merge_genes,
    REQUIRED_GENES,
)
from .experience_pool import ExperiencePool, TaskRecord
from .task_dispatcher import TaskDispatcher
from .agent_evolution import AgentEvolution
from .gene_registry import GeneRegistry
from .gene_selector import select_gene_plan
from .pheromone import compute_node_pheromone, ensure_node_stats
from .solution_evolution import SolutionEvolution

__all__ = [
    "parse_solution_genes",
    "validate_genes",
    "merge_genes",
    "REQUIRED_GENES",
    "ExperiencePool",
    "TaskRecord",
    "TaskDispatcher",
    "AgentEvolution",
    "GeneRegistry",
    "select_gene_plan",
    "compute_node_pheromone",
    "ensure_node_stats",
    "SolutionEvolution",
]
