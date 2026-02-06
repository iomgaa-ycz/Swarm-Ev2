"""MLE-Bench 适配器。

将 MLE-Bench 容器环境桥接到 Swarm-Ev2 执行管线：
1. 环境变量映射（API_KEY → OPENAI_API_KEY 等）
2. 加载容器专用配置（config/mle_bench.yaml）
3. 构建工作空间并运行双层进化主循环
4. 输出结果到 MLE-Bench 标准目录
"""

import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional


# ============================================================
# Phase 1: 环境变量映射（必须在任何项目模块 import 之前）
# ============================================================


def map_env_vars() -> None:
    """将 MLE-Bench 环境变量映射为 Swarm-Ev2 格式。"""
    mapping = {
        "API_KEY": "OPENAI_API_KEY",
        "API_BASE": "OPENAI_BASE_URL",
        "MODEL_NAME": "LLM_MODEL",
    }
    for src, dst in mapping.items():
        val = os.environ.get(src, "")
        if val:
            os.environ[dst] = val


map_env_vars()


# ============================================================
# Phase 2: 导入项目模块（环境变量已就绪）
# ============================================================

from utils.config import load_config, Config  # noqa: E402
from utils.logger_system import init_logger, log_msg  # noqa: E402
from utils.prompt_manager import PromptManager  # noqa: E402
from agents.coder_agent import CoderAgent  # noqa: E402
from core.executor.workspace import WorkspaceManager  # noqa: E402
from core.state import Journal, Node  # noqa: E402
from core.orchestrator import Orchestrator  # noqa: E402
from core.evolution import (  # noqa: E402
    ExperiencePool,
    TaskDispatcher,
    GeneRegistry,
    AgentEvolution,
    SolutionEvolution,
    CodeEmbeddingManager,
    SkillManager,
)


# ============================================================
# 工具函数
# ============================================================


def find_description(data_dir: Path) -> str:
    """查找并读取竞赛描述文件。

    Args:
        data_dir: 数据目录

    Returns:
        描述内容字符串
    """
    candidates = [
        Path("/home/description.md"),
        data_dir / "description.md",
    ]
    for path in candidates:
        if path.exists():
            return path.read_text(encoding="utf-8")

    competition_id = os.environ.get("COMPETITION_ID", "unknown")
    return f"# Kaggle Competition: {competition_id}\n\nData: {data_dir}\n"


def setup_workspace(config: Config, description: str) -> None:
    """构建 MLE-Bench 兼容的工作空间。

    Args:
        config: 配置对象
        description: 竞赛描述内容
    """
    ws = config.project.workspace_dir
    data_dir = config.data.data_dir

    for subdir in ["input", "working", "submission", "best_solution", "logs"]:
        (ws / subdir).mkdir(parents=True, exist_ok=True)

    # 写入 description.md
    (ws / "description.md").write_text(description, encoding="utf-8")

    # 链接数据文件到 input/
    input_dir = ws / "input"
    if data_dir.exists():
        for item in data_dir.iterdir():
            if item.name == "description.md":
                continue
            target = input_dir / item.name
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(item.resolve())


def copy_results(journal: Journal, config: Config) -> None:
    """复制结果到 MLE-Bench 标准目录。

    Args:
        journal: Journal 实例
        config: 配置对象
    """
    submission_dir = Path("/home/submission")
    code_dir = Path("/home/code")
    logs_dir = Path("/home/logs")
    ws = config.project.workspace_dir

    submission_dir.mkdir(parents=True, exist_ok=True)
    code_dir.mkdir(parents=True, exist_ok=True)

    # 从最佳方案恢复归档
    best_node = journal.get_best_node(only_good=True)
    if best_node and getattr(best_node, "archive_path", None):
        archive = Path(best_node.archive_path)
        if archive.exists():
            import zipfile

            with zipfile.ZipFile(archive, "r") as zf:
                zf.extractall(ws)
            log_msg("INFO", f"已从归档恢复最佳方案: {archive}")

    # 复制 submission.csv
    sub_file = submission_dir / "submission.csv"
    if not sub_file.exists():
        for candidate in [
            ws / "submission" / "submission.csv",
            ws / "best_solution" / "submission.csv",
        ]:
            if candidate.exists():
                shutil.copy2(candidate, sub_file)
                break
        else:
            # 兜底搜索
            for found in ws.glob("**/submission.csv"):
                shutil.copy2(found, sub_file)
                break

    if sub_file.exists():
        log_msg("INFO", f"提交文件就绪: {sub_file}")
    else:
        log_msg("WARNING", "未找到 submission.csv")

    # 复制 solution.py
    for src in [ws / "best_solution" / "solution.py", ws / "solution.py"]:
        if src.exists():
            shutil.copy2(src, code_dir / "solution.py")
            break

    # 保存 journal
    journal_path = ws / "journal.json"
    if journal_path.exists():
        shutil.copy2(journal_path, logs_dir / "journal.json")


# ============================================================
# 主执行函数
# ============================================================


def run_adapter() -> None:
    """MLE-Bench 适配器主入口。"""
    start_time = time.time()
    competition_id = os.environ.get("COMPETITION_ID", "unknown")

    # Phase 3: 日志
    logs_dir = Path("/home/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    init_logger(str(logs_dir))
    log_msg("INFO", f"Swarm-Ev2 MLE-Bench Adapter 启动 | competition={competition_id}")

    # Phase 4: 加载配置
    config_path = Path(__file__).parent / "config" / "mle_bench.yaml"
    try:
        config = load_config(config_path=config_path, use_cli=False)
    except Exception as e:
        log_msg("WARNING", f"配置加载失败: {e}，尝试使用默认配置")
        config = load_config(use_cli=False)

    # Phase 5: 工作空间
    description = find_description(config.data.data_dir)
    setup_workspace(config, description)
    log_msg("INFO", f"工作空间就绪: {config.project.workspace_dir}")

    workspace = WorkspaceManager(config)
    if config.data.preprocess_data:
        try:
            workspace.preprocess_input()
        except Exception as e:
            log_msg("WARNING", f"数据预处理跳过: {e}")

    # Phase 6: 组件初始化
    base_dir = Path(__file__).parent / "benchmark" / "mle-bench"

    embedding_manager = CodeEmbeddingManager(
        model_path=getattr(config.evolution.skill, "embedding_model_path", None),
    )
    skill_manager = SkillManager(
        skills_dir=base_dir / "skills",
        meta_dir=base_dir / "skills" / "meta",
        config=config,
        embedding_manager=embedding_manager,
    )

    prompt_manager = PromptManager(
        template_dir=base_dir / "prompt_templates",
        skills_dir=base_dir / "skills",
        agent_configs_dir=base_dir / "agent_configs",
        skill_manager=skill_manager,
    )

    num_agents = config.evolution.agent.num_agents
    agents = [
        CoderAgent(name=f"agent_{i}", config=config, prompt_manager=prompt_manager)
        for i in range(num_agents)
    ]
    log_msg("INFO", f"Agent 种群初始化: {num_agents} 个")

    experience_pool = ExperiencePool(config)
    task_dispatcher = TaskDispatcher(
        agents=agents,
        epsilon=config.evolution.agent.epsilon,
        learning_rate=config.evolution.agent.learning_rate,
    )
    gene_registry = GeneRegistry()

    agent_evolution: Optional[AgentEvolution] = None
    configs_dir = Path(config.evolution.agent.configs_dir)
    if configs_dir.exists():
        agent_evolution = AgentEvolution(
            agents=agents,
            experience_pool=experience_pool,
            config=config,
            skill_manager=skill_manager,
        )

    journal = Journal()

    orchestrator = Orchestrator(
        agents=agents,
        config=config,
        journal=journal,
        task_desc=description,
        agent_evolution=agent_evolution,
        task_dispatcher=task_dispatcher,
        experience_pool=experience_pool,
        gene_registry=gene_registry,
    )

    solution_evolution = SolutionEvolution(
        config=config,
        journal=journal,
        orchestrator=orchestrator,
        gene_registry=gene_registry,
    )
    log_msg("INFO", "所有组件初始化完成")

    # Phase 7: 双层进化主循环
    num_epochs = max(
        1, config.agent.max_steps // config.evolution.solution.steps_per_epoch
    )
    steps_per_epoch = config.evolution.solution.steps_per_epoch
    log_msg("INFO", f"开始进化: {num_epochs} Epochs x {steps_per_epoch} Steps")

    best_node: Optional[Node] = None
    for epoch in range(num_epochs):
        log_msg("INFO", f"===== Epoch {epoch + 1}/{num_epochs} =====")

        orchestrator._run_single_epoch(steps_per_epoch)

        epoch_best = solution_evolution.run_epoch(steps_per_epoch)
        if epoch_best and (
            not best_node or epoch_best.metric_value > (best_node.metric_value or 0)
        ):
            best_node = epoch_best

        if agent_evolution and (epoch + 1) % 3 == 0:
            agent_evolution.evolve(epoch)

        current_best = journal.get_best_node()
        log_msg(
            "INFO",
            f"Epoch {epoch + 1} 完成 | best={current_best.metric_value if current_best else 'N/A'}",
        )

    best_node = journal.get_best_node(only_good=True)

    # Phase 8: 结果输出
    copy_results(journal, config)

    if best_node:
        log_msg(
            "INFO", f"最佳方案: ID={best_node.id[:12]}, metric={best_node.metric_value}"
        )
    else:
        log_msg("WARNING", "未找到有效方案")

    experience_pool.save()
    elapsed = time.time() - start_time
    log_msg("INFO", f"Swarm-Ev2 MLE-Bench Adapter 结束 | 耗时 {elapsed:.1f}s")


if __name__ == "__main__":
    try:
        run_adapter()
    except Exception as e:
        # 兜底：容器内必须尽量不崩溃
        print(f"Adapter 异常: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
