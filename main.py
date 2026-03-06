"""Swarm-Ev2 主程序入口。

实现双层群体智能架构的端到端测试：
    - Agent 层：4 个 Agent + 经验池 + 任务分发 + Skill 池
    - Solution 层：遗传算法（精英保留 + 锦标赛 + 交叉 + 变异）
"""

import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from utils.config import load_config, Config
from utils.logger_system import init_logger, log_msg, log_exception, log_json
from utils.workspace_builder import build_workspace, validate_dataset
from utils.prompt_manager import PromptManager
from agents.coder_agent import CoderAgent
from agents.base_agent import BaseAgent

# Interpreter 由 Orchestrator 内部创建和管理
from core.executor.workspace import WorkspaceManager
from core.state import Journal, Node
from core.orchestrator import Orchestrator
from core.evolution import (
    ExperiencePool,
    TaskDispatcher,
    GeneRegistry,
    AgentEvolution,
    SolutionEvolution,
    CodeEmbeddingManager,
    SkillManager,
    validate_genes,
)


def initialize_agents(config: Config, prompt_manager: PromptManager) -> List[BaseAgent]:
    """初始化 Agent 种群。

    Args:
        config: 全局配置
        prompt_manager: PromptManager 实例

    Returns:
        Agent 列表
    """
    num_agents = config.evolution.agent.num_agents
    agents = []

    for i in range(num_agents):
        agent = CoderAgent(
            name=f"agent_{i}",
            config=config,
            prompt_manager=prompt_manager,
        )
        agents.append(agent)

    log_msg("INFO", f"Agent 种群初始化完成: {num_agents} 个 Agent")
    return agents


def initialize_evolution_components(
    agents: List[BaseAgent],
    config: Config,
    workspace: WorkspaceManager,
    prompt_manager: PromptManager,
    skill_manager: Optional[SkillManager] = None,
) -> Tuple[ExperiencePool, TaskDispatcher, GeneRegistry, Optional[AgentEvolution]]:
    """初始化进化算法组件。

    Args:
        agents: Agent 列表
        config: 全局配置
        workspace: 工作空间管理器
        prompt_manager: PromptManager 实例
        skill_manager: SkillManager 实例（可选）

    Returns:
        (experience_pool, task_dispatcher, gene_registry, agent_evolution)
    """
    # 经验池
    experience_pool = ExperiencePool(config)
    log_msg("INFO", f"经验池初始化完成（已加载 {len(experience_pool.records)} 条记录）")

    # 任务分发器
    task_dispatcher = TaskDispatcher(
        agents=agents,
        epsilon=config.evolution.agent.epsilon,
        learning_rate=config.evolution.agent.learning_rate,
    )
    log_msg("INFO", "任务分发器初始化完成")

    # 基因注册表
    gene_registry = GeneRegistry()
    log_msg("INFO", "基因注册表初始化完成")

    # Agent 层进化器（可选，需要 default/ 配置目录存在）
    agent_evolution = None
    configs_dir = Path(config.evolution.agent.configs_dir)
    if (configs_dir / "default").exists():
        agent_evolution = AgentEvolution(
            agents=agents,
            experience_pool=experience_pool,
            config=config,
            prompt_manager=prompt_manager,
            skill_manager=skill_manager,
        )
        log_msg("INFO", "Agent 进化器初始化完成（Skill 进化已启用）")
    else:
        log_msg("WARNING", f"Agent 配置目录不存在，跳过 Agent 进化: {configs_dir}/default")

    return experience_pool, task_dispatcher, gene_registry, agent_evolution


def generate_markdown_report(
    journal: Journal,
    experience_pool: ExperiencePool,
    task_dispatcher: TaskDispatcher,
    config: Config,
    start_time: float,
    best_node: Optional[Node],
) -> Path:
    """生成 Markdown 测试报告。

    Args:
        journal: 历史节点记录
        experience_pool: 经验池
        task_dispatcher: 任务分发器
        config: 全局配置
        start_time: 开始时间
        best_node: 最佳节点

    Returns:
        报告文件路径
    """
    # 创建输出目录
    output_dir = Path("tests/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"main_execution_{timestamp}.md"

    # 计算统计数据
    total_nodes = len(journal.nodes)
    success_nodes = len([n for n in journal.nodes if not n.is_buggy])
    buggy_nodes = total_nodes - success_nodes
    elapsed_time = time.time() - start_time

    # 生成报告内容
    content = f"""# Swarm-Ev2 端到端测试报告

**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**运行时长**: {elapsed_time:.2f} 秒

---

## 1. 配置摘要

| 配置项 | 值 |
|--------|-----|
| Agent 数量 | {config.evolution.agent.num_agents} |
| 种群大小 | {config.evolution.solution.population_size} |
| 精英保留 | {config.evolution.solution.elite_size} |
| 交叉率 | {config.evolution.solution.crossover_rate} |
| 变异率 | {config.evolution.solution.mutation_rate} |
| 每 Epoch 步数 | {config.evolution.solution.steps_per_epoch} |
| 探索率 (epsilon) | {config.evolution.agent.epsilon} |

---

## 2. 执行统计

| 指标 | 值 |
|------|-----|
| 总节点数 | {total_nodes} |
| 成功节点 | {success_nodes} |
| 失败节点 | {buggy_nodes} |
| 成功率 | {success_nodes / total_nodes * 100 if total_nodes > 0 else 0:.1f}% |
| 经验池记录 | {len(experience_pool.records)} |

---

## 3. 最佳方案

"""

    if best_node:
        content += f"""| 属性 | 值 |
|------|-----|
| 节点 ID | `{best_node.id[:12]}` |
| 评估指标 | {best_node.metric_value} |
| 越小越好 | {best_node.lower_is_better} |
| 执行时间 | {best_node.exec_time:.2f}s |
| 是否有 Bug | {best_node.is_buggy} |

### 代码摘要

```python
{best_node.code[:500] if best_node.code else "无代码"}...
```

"""
    else:
        content += "**未找到有效方案**\n\n"

    # Agent 擅长度得分
    content += """---

## 4. Agent 擅长度得分

| Agent | draft | merge | mutate |
|-------|-------|-------|--------|
"""

    scores = task_dispatcher.get_specialization_matrix()
    for agent_id, task_scores in scores.items():
        content += (
            f"| {agent_id} "
            f"| {task_scores.get('draft', task_scores.get('explore', 0)):.3f} "
            f"| {task_scores.get('merge', 0):.3f} "
            f"| {task_scores.get('mutate', 0):.3f} |\n"
        )

    # 节点历史
    content += """
---

## 5. 节点执行历史

| 序号 | 节点 ID | 状态 | 指标 | 执行时间 |
|------|---------|------|------|----------|
"""

    for i, node in enumerate(journal.nodes[:20]):  # 限制前 20 个
        status = "✅ 成功" if not node.is_buggy else "❌ 失败"
        metric = f"{node.metric_value:.4f}" if node.metric_value else "N/A"
        content += f"| {i + 1} | `{node.id[:8]}` | {status} | {metric} | {node.exec_time:.2f}s |\n"

    if len(journal.nodes) > 20:
        content += f"\n*...共 {len(journal.nodes)} 个节点，仅显示前 20 个*\n"

    content += """
---

## 6. 经验池样本

最近 5 条记录：

| Agent | 任务类型 | 质量 | 策略摘要 |
|-------|---------|------|----------|
"""

    recent_records = experience_pool.records[-5:] if experience_pool.records else []
    for record in recent_records:
        summary = (
            record.strategy_summary[:50] + "..."
            if len(record.strategy_summary) > 50
            else record.strategy_summary
        )
        content += f"| {record.agent_id} | {record.task_type} | {record.output_quality:.3f} | {summary} |\n"

    content += """
---

*报告由 Swarm-Ev2 自动生成*
"""

    # 写入文件
    report_path.write_text(content, encoding="utf-8")
    log_msg("INFO", f"Markdown 测试报告已生成: {report_path}")

    return report_path


def log_evolution_statistics(
    journal: Journal,
    experience_pool: ExperiencePool,
    task_dispatcher: TaskDispatcher,
    best_node: Optional[Node],
) -> None:
    """记录进化统计信息。

    Args:
        journal: 历史节点记录
        experience_pool: 经验池
        task_dispatcher: 任务分发器
        best_node: 最佳节点
    """
    total_nodes = len(journal.nodes)
    success_nodes = len([n for n in journal.nodes if not n.is_buggy])
    success_rate = success_nodes / total_nodes * 100 if total_nodes > 0 else 0

    log_msg("INFO", "=" * 60)
    log_msg("INFO", "进化统计")
    log_msg(
        "INFO",
        f"[节点] 总={total_nodes}, 成功={success_nodes}, 失败={total_nodes - success_nodes}, 成功率={success_rate:.1f}%",
    )
    log_msg("INFO", f"[经验池] 记录数={len(experience_pool.records)}")

    scores = task_dispatcher.get_specialization_matrix()
    for agent_id, task_scores in scores.items():
        log_msg(
            "INFO",
            f"[Agent] {agent_id}: "
            f"draft={task_scores.get('draft', task_scores.get('explore', 0)):.3f}, "
            f"merge={task_scores.get('merge', 0):.3f}, "
            f"mutate={task_scores.get('mutate', 0):.3f}",
        )

    if best_node:
        log_msg(
            "INFO",
            f"[最佳] ID={best_node.id[:12]}, metric={best_node.metric_value}, lower_is_better={best_node.lower_is_better}, exec_time={best_node.exec_time:.2f}s",
        )
    else:
        log_msg("INFO", "[最佳] 未找到有效方案")
    log_msg("INFO", "=" * 60)

    log_json({
        "event": "evolution_statistics",
        "total_nodes": total_nodes,
        "success_rate": round(success_rate, 1),
        "experience_pool_size": len(experience_pool.records),
        "specialization_matrix": scores,
        "best_metric": best_node.metric_value if best_node else None,
    })


def main() -> None:
    """主执行函数（双层进化架构）。

    执行流程:
        1. 环境准备：加载配置、验证数据集
        2. 工作空间构建：创建目录、复制/链接数据
        3. 组件初始化：Agent 种群、进化组件、Orchestrator
        4. 运行双层进化主循环
        5. 生成 Markdown 测试报告
        6. 结果展示
    """
    log_msg("INFO", "启动 Swarm-Ev2 双层群体智能系统")
    start_time = time.time()

    try:
        # ============================================================
        # Phase 1: 环境准备
        # ============================================================
        log_msg("INFO", "[1/6] 环境准备...")

        # 加载配置
        config = load_config()

        # 验证数据集
        is_valid, error_msg = validate_dataset(config.data.data_dir)
        if not is_valid:
            log_msg("ERROR", f"数据集验证失败: {error_msg}")
            return

        log_msg("INFO", f"数据集验证通过: {config.data.data_dir}")

        # 初始化代理（测试连通性 + 配置环境变量）
        from utils.proxy import init_proxy

        init_proxy()

        # ============================================================
        # Phase 2: 工作空间构建
        # ============================================================
        log_msg("INFO", "[2/6] 工作空间构建...")

        # 清理旧的 workspace 目录
        if config.project.workspace_dir.exists():
            shutil.rmtree(config.project.workspace_dir)
            log_msg("INFO", f"清理旧的工作空间: {config.project.workspace_dir}")

        # 构建新的 workspace
        task_description = build_workspace(
            data_dir=config.data.data_dir,
            workspace_dir=config.project.workspace_dir,
            copy_data=config.data.copy_data,
        )
        log_msg("INFO", f"工作空间构建成功: {config.project.workspace_dir}")

        # 初始化工作空间管理器并执行数据预处理（解压压缩包 + 清理垃圾文件）
        workspace = WorkspaceManager(config)
        if getattr(config.data, "preprocess_data", True):
            log_msg("INFO", "执行数据预处理（解压 + 清理）...")
            workspace.preprocess_input()
            log_msg("INFO", "数据预处理完成")

        # P0-3 修复：保护输入文件
        workspace.protect_input_files()
        log_msg("INFO", "输入文件已保护 (chmod 444)")

        # ============================================================
        # Phase 3: 组件初始化
        # ============================================================
        log_msg("INFO", "[3/6] 组件初始化...")

        # 初始化日志系统
        log_dir = config.project.workspace_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        init_logger(str(log_dir))
        log_msg("INFO", "日志系统初始化完成")

        # 工作空间管理器已在 Phase 2 初始化
        # Interpreter 由 Orchestrator 内部创建和管理
        log_msg("INFO", "工作空间管理器已就绪")

        # 初始化 Skill 进化组件
        base_dir = Path("benchmark/mle-bench")
        embedding_manager = CodeEmbeddingManager(
            model_path=getattr(config.evolution.skill, "embedding_model_path", None),
        )
        skill_manager = SkillManager(
            skills_dir=base_dir / "skills",
            meta_dir=base_dir / "skills" / "meta",
            config=config,
            embedding_manager=embedding_manager,
        )
        log_msg(
            "INFO",
            f"SkillManager 初始化完成（已加载 {len(skill_manager.skill_index)} 个 Skill）",
        )

        # 初始化 PromptManager
        prompt_manager = PromptManager(
            template_dir=base_dir / "prompt_templates",
            skills_dir=base_dir / "skills",
            agent_configs_dir=base_dir / "agent_configs",
            skill_manager=skill_manager,
            num_agents=config.evolution.agent.num_agents,
        )

        # 初始化 Agent 种群
        agents = initialize_agents(config, prompt_manager)

        # 初始化进化组件
        experience_pool, task_dispatcher, gene_registry, agent_evolution = (
            initialize_evolution_components(
                agents=agents,
                config=config,
                workspace=workspace,
                prompt_manager=prompt_manager,
                skill_manager=skill_manager,
            )
        )

        # 初始化 Journal
        journal = Journal()
        log_msg("INFO", "Journal 初始化完成")

        # 初始化 Orchestrator（使用双层进化模式 + 多 Agent 并行）
        orchestrator = Orchestrator(
            agents=agents,  # 传递所有 Agent，支持并行执行
            config=config,
            journal=journal,
            task_desc=task_description,
            agent_evolution=agent_evolution,
            task_dispatcher=task_dispatcher,  # Phase 3 集成
            experience_pool=experience_pool,  # Phase 3 集成
            gene_registry=gene_registry,  # 信息素驱动交叉
        )
        log_msg("INFO", "Orchestrator 初始化完成（双层进化模式 + 并行执行）")

        # 初始化 SolutionEvolution（Phase 3）
        solution_evolution = SolutionEvolution(
            config=config,
            journal=journal,
            orchestrator=orchestrator,
            gene_registry=gene_registry,  # 信息素驱动交叉
        )
        log_msg("INFO", "SolutionEvolution 初始化完成（MVP 简化版）")

        log_msg("INFO", "所有组件初始化完成")

        # 配置摘要
        log_msg(
            "INFO",
            f"配置摘要: agents={config.evolution.agent.num_agents}, "
            f"steps_per_epoch={config.evolution.solution.steps_per_epoch}, "
            f"epsilon={config.evolution.agent.epsilon}, "
            f"ga_trigger={config.evolution.solution.ga_trigger_threshold}, "
            f"debug_max={config.evolution.solution.debug_max_attempts}",
        )

        # ============================================================
        # Phase 4: 混合进化主循环（Draft + GA 交替）
        # ============================================================
        log_msg("INFO", "[4/6] 运行混合进化主循环...")

        total_budget = config.agent.max_steps
        steps_per_epoch = config.evolution.solution.steps_per_epoch
        ga_trigger = config.evolution.solution.ga_trigger_threshold
        global_epoch = 0
        hybrid_active = False

        log_msg(
            "INFO",
            f"开始进化: total_budget={total_budget}, GA触发阈值={ga_trigger}, steps_per_epoch={steps_per_epoch}",
        )

        while (
            len(journal.nodes) < total_budget and not orchestrator._check_time_limit()
        ):
            remaining = total_budget - len(journal.nodes)
            epoch_steps = min(steps_per_epoch, remaining)

            # 检查 valid_pool 是否达到 GA 触发条件
            valid_pool = [
                n
                for n in journal.nodes
                if not n.is_buggy and not n.dead and validate_genes(n.genes)
            ]

            if len(valid_pool) >= ga_trigger:
                # 混合模式：30% Draft + 70% GA
                if not hybrid_active:
                    log_msg(
                        "INFO",
                        f"===== 切换到混合模式: valid_pool={len(valid_pool)}>={ga_trigger} =====",
                    )
                    hybrid_active = True
                orchestrator.run_epoch_hybrid(epoch_steps, solution_evolution)
            else:
                # 纯 Draft 模式（积累种群）
                orchestrator.run_epoch_draft(epoch_steps)

            global_epoch += 1

            # Agent 进化
            if agent_evolution and global_epoch % 3 == 0:
                log_msg(
                    "INFO", f"触发 Agent 层进化（global_epoch={global_epoch}）"
                )
                agent_evolution.evolve(global_epoch)

            # epoch 日志
            valid_pool = [
                n
                for n in journal.nodes
                if not n.is_buggy and not n.dead and validate_genes(n.genes)
            ]
            current_best = journal.get_best_node(
                lower_is_better=orchestrator._global_lower_is_better
            )
            log_msg(
                "INFO",
                f"Epoch {global_epoch} | nodes={len(journal.nodes)}/{total_budget}, "
                f"valid={len(valid_pool)}, hybrid={hybrid_active}, "
                f"best={current_best.metric_value if current_best else 'N/A'}",
            )

        best_node = journal.get_best_node(
            only_good=True, lower_is_better=orchestrator._global_lower_is_better
        )
        log_msg(
            "INFO",
            f"混合进化完成: best_node={'存在' if best_node else '不存在'}",
        )

        # ============================================================
        # Phase 5: 生成 Markdown 测试报告
        # ============================================================
        log_msg("INFO", "[5/6] 生成测试报告...")

        report_path = generate_markdown_report(
            journal=journal,
            experience_pool=experience_pool,
            task_dispatcher=task_dispatcher,
            config=config,
            start_time=start_time,
            best_node=best_node,
        )

        log_msg("INFO", f"测试报告: {report_path}")

        # ============================================================
        # Phase 6: 结果展示
        # ============================================================
        log_msg("INFO", "[6/6] 结果展示...")

        if best_node is None:
            log_msg("WARNING", "未找到有效方案")
        else:
            log_msg(
                "INFO",
                f"最佳方案: ID={best_node.id[:12]}, metric={best_node.metric_value}, "
                f"lower_is_better={best_node.lower_is_better}, exec_time={best_node.exec_time:.2f}s, "
                f"path={config.project.workspace_dir / 'best_solution' / 'solution.py'}",
            )

        # 记录进化统计
        log_evolution_statistics(journal, experience_pool, task_dispatcher, best_node)

        # 保存 Journal
        journal_path = config.project.workspace_dir / "logs" / "journal.json"
        journal_path.write_text(journal.to_json(indent=2), encoding="utf-8")
        log_msg("INFO", f"Journal 已保存: {journal_path}")

        # 保存经验池
        experience_pool.save()
        log_msg("INFO", f"经验池已保存: {experience_pool.save_path}")

        # 导出 Agent 最终配置
        agent_configs_final_dir = config.project.workspace_dir / "logs" / "agent_configs_final"
        prompt_manager.export_agent_configs(agent_configs_final_dir)
        log_msg("INFO", f"Agent 最终配置已导出: {agent_configs_final_dir}")

        # 导出 Skill 池
        skills_final_dir = config.project.workspace_dir / "logs" / "skills_final"
        shutil.copytree(base_dir / "skills", skills_final_dir, dirs_exist_ok=True)
        log_msg("INFO", f"Skill 池已导出: {skills_final_dir}")

        elapsed_time = time.time() - start_time
        log_msg("INFO", f"执行完成！总耗时: {elapsed_time:.2f}s")

        # 记录最终日志
        log_json(
            {
                "event": "main_completed",
                "elapsed_time": elapsed_time,
                "total_nodes": len(journal.nodes),
                "success_nodes": len([n for n in journal.nodes if not n.is_buggy]),
                "experience_pool_size": len(experience_pool.records),
                "best_metric": best_node.metric_value if best_node else None,
            }
        )

    except KeyboardInterrupt:
        log_msg("WARNING", "用户中断执行")
    except Exception as e:
        log_exception(e, "主程序执行失败")
        raise


if __name__ == "__main__":
    main()
