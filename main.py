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
from utils.prompt_builder import PromptBuilder
from agents.coder_agent import CoderAgent
from agents.base_agent import BaseAgent
from core.executor.interpreter import Interpreter
from core.executor.workspace import WorkspaceManager
from core.state import Journal, Node
from core.orchestrator import Orchestrator
from core.evolution import (
    ExperiencePool,
    TaskDispatcher,
    GeneRegistry,
    AgentEvolution,
)


def initialize_agents(
    config: Config, prompt_builder: PromptBuilder, interpreter: Interpreter
) -> List[BaseAgent]:
    """初始化 Agent 种群。

    Args:
        config: 全局配置
        prompt_builder: Prompt 构建器
        interpreter: 代码执行器

    Returns:
        Agent 列表（4 个 Agent）
    """
    num_agents = config.evolution.agent.num_agents
    agents = []

    for i in range(num_agents):
        agent = CoderAgent(
            name=f"agent_{i}",
            config=config,
            prompt_builder=prompt_builder,
            interpreter=interpreter,
        )
        agents.append(agent)

    log_msg("INFO", f"Agent 种群初始化完成: {num_agents} 个 Agent")
    return agents


def initialize_evolution_components(
    agents: List[BaseAgent],
    config: Config,
    workspace: WorkspaceManager,
    interpreter: Interpreter,
) -> Tuple[ExperiencePool, TaskDispatcher, GeneRegistry, Optional[AgentEvolution]]:
    """初始化进化算法组件。

    Args:
        agents: Agent 列表
        config: 全局配置
        workspace: 工作空间管理器
        interpreter: 代码执行器

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

    # Agent 层进化器（可选，需要配置目录存在）
    agent_evolution = None
    configs_dir = Path(config.evolution.agent.configs_dir)
    if configs_dir.exists():
        agent_evolution = AgentEvolution(
            agents=agents,
            experience_pool=experience_pool,
            config=config,
            skill_manager=None,  # 简化版本暂不使用 Skill 管理器
        )
        log_msg("INFO", "Agent 进化器初始化完成")
    else:
        log_msg("WARNING", f"Agent 配置目录不存在，跳过 Agent 进化: {configs_dir}")

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

| Agent | explore | merge | mutate |
|-------|---------|-------|--------|
"""

    scores = task_dispatcher.get_specialization_matrix()
    for agent_id, task_scores in scores.items():
        content += f"| {agent_id} | {task_scores['explore']:.3f} | {task_scores['merge']:.3f} | {task_scores['mutate']:.3f} |\n"

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


def print_evolution_statistics(
    journal: Journal,
    experience_pool: ExperiencePool,
    task_dispatcher: TaskDispatcher,
    best_node: Optional[Node],
) -> None:
    """打印进化统计信息。

    Args:
        journal: 历史节点记录
        experience_pool: 经验池
        task_dispatcher: 任务分发器
        best_node: 最佳节点
    """
    print("\n" + "=" * 60)
    print("📊 进化统计")
    print("=" * 60)

    # 节点统计
    total_nodes = len(journal.nodes)
    success_nodes = len([n for n in journal.nodes if not n.is_buggy])
    print("\n[节点统计]")
    print(f"  总节点数: {total_nodes}")
    print(f"  成功节点: {success_nodes}")
    print(f"  失败节点: {total_nodes - success_nodes}")
    print(
        f"  成功率: {success_nodes / total_nodes * 100 if total_nodes > 0 else 0:.1f}%"
    )

    # 经验池统计
    print("\n[经验池]")
    print(f"  记录数: {len(experience_pool.records)}")

    # Agent 擅长度得分
    print("\n[Agent 擅长度得分]")
    scores = task_dispatcher.get_specialization_matrix()
    for agent_id, task_scores in scores.items():
        print(
            f"  {agent_id}: explore={task_scores['explore']:.3f}, merge={task_scores['merge']:.3f}, mutate={task_scores['mutate']:.3f}"
        )

    # 最佳方案
    print("\n[最佳方案]")
    if best_node:
        print(f"  节点 ID: {best_node.id[:12]}")
        print(f"  评估指标: {best_node.metric_value}")
        print(f"  越小越好: {best_node.lower_is_better}")
        print(f"  执行时间: {best_node.exec_time:.2f}s")
    else:
        print("  未找到有效方案")

    print("\n" + "=" * 60)


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
    print("\n🚀 启动 Swarm-Ev2 双层群体智能系统\n")
    start_time = time.time()

    try:
        # ============================================================
        # Phase 1: 环境准备
        # ============================================================
        print("[1/6] 环境准备...")

        # 加载配置
        config = load_config()

        # 验证数据集
        is_valid, error_msg = validate_dataset(config.data.data_dir)
        if not is_valid:
            print(f"❌ 数据集验证失败: {error_msg}")
            return

        print(f"✅ 数据集验证通过: {config.data.data_dir}")

        # ============================================================
        # Phase 2: 工作空间构建
        # ============================================================
        print("\n[2/6] 工作空间构建...")

        # 清理旧的 workspace 目录
        if config.project.workspace_dir.exists():
            shutil.rmtree(config.project.workspace_dir)
            print(f"  清理旧的工作空间: {config.project.workspace_dir}")

        # 构建新的 workspace
        task_description = build_workspace(
            data_dir=config.data.data_dir,
            workspace_dir=config.project.workspace_dir,
            copy_data=config.data.copy_data,
        )
        print(f"✅ 工作空间构建成功: {config.project.workspace_dir}")

        # ============================================================
        # Phase 3: 组件初始化
        # ============================================================
        print("\n[3/6] 组件初始化...")

        # 初始化日志系统
        log_dir = config.project.workspace_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        init_logger(str(log_dir))
        log_msg("INFO", "日志系统初始化完成")

        # 初始化 Interpreter
        interpreter = Interpreter(
            working_dir=str(config.project.workspace_dir / "working"),
            timeout=config.execution.timeout,
        )
        log_msg("INFO", "代码执行器初始化完成")

        # 初始化工作空间管理器
        workspace = WorkspaceManager(config)
        log_msg("INFO", "工作空间管理器初始化完成")

        # 初始化 PromptBuilder
        prompt_builder = PromptBuilder(obfuscate=False)
        log_msg("INFO", "Prompt 构建器初始化完成")

        # 初始化 Agent 种群
        agents = initialize_agents(config, prompt_builder, interpreter)

        # 初始化进化组件
        experience_pool, task_dispatcher, gene_registry, agent_evolution = (
            initialize_evolution_components(
                agents=agents,
                config=config,
                workspace=workspace,
                interpreter=interpreter,
            )
        )

        # 初始化 Journal
        journal = Journal()
        log_msg("INFO", "Journal 初始化完成")

        # 选择主 Agent（用于 Orchestrator）
        main_agent = agents[0]

        # 初始化 Orchestrator（使用双层进化模式）
        orchestrator = Orchestrator(
            agent=main_agent,
            config=config,
            journal=journal,
            task_desc=task_description,
            agent_evolution=agent_evolution,
        )
        log_msg("INFO", "Orchestrator 初始化完成（双层进化模式）")

        print("✅ 所有组件初始化完成")

        # 打印配置摘要
        print("\n📋 配置摘要:")
        print(f"  Agent 数量: {config.evolution.agent.num_agents}")
        print(f"  每 Epoch 步数: {config.evolution.solution.steps_per_epoch}")
        print(f"  探索率: {config.evolution.agent.epsilon}")

        # ============================================================
        # Phase 4: 运行双层进化主循环
        # ============================================================
        num_epochs = max(
            1, config.agent.max_steps // config.evolution.solution.steps_per_epoch
        )
        steps_per_epoch = config.evolution.solution.steps_per_epoch

        print("\n[4/6] 运行双层进化主循环...")
        print(f"  Epochs: {num_epochs}")
        print(f"  Steps/Epoch: {steps_per_epoch}")
        print(f"  总步数: {num_epochs * steps_per_epoch}")
        print("")

        best_node = orchestrator.run(
            num_epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
        )

        # ============================================================
        # Phase 5: 生成 Markdown 测试报告
        # ============================================================
        print("\n[5/6] 生成测试报告...")

        report_path = generate_markdown_report(
            journal=journal,
            experience_pool=experience_pool,
            task_dispatcher=task_dispatcher,
            config=config,
            start_time=start_time,
            best_node=best_node,
        )

        print(f"✅ 测试报告: {report_path}")

        # ============================================================
        # Phase 6: 结果展示
        # ============================================================
        print("\n[6/6] 结果展示...")

        if best_node is None:
            print("❌ 未找到有效方案")
            log_msg("WARNING", "未找到有效方案")
        else:
            print("\n🎉 最佳方案已生成:")
            print(f"  节点 ID: {best_node.id[:12]}")
            print(f"  评估指标: {best_node.metric_value}")
            print(f"  越小越好: {best_node.lower_is_better}")
            print(f"  执行时间: {best_node.exec_time:.2f}s")
            print(
                f"  代码路径: {config.project.workspace_dir / 'best_solution' / 'solution.py'}"
            )

            log_msg(
                "INFO",
                f"最佳方案: ID={best_node.id[:12]}, metric={best_node.metric_value}",
            )

        # 打印进化统计
        print_evolution_statistics(journal, experience_pool, task_dispatcher, best_node)

        # 保存经验池
        experience_pool.save()
        log_msg("INFO", f"经验池已保存: {experience_pool.save_path}")

        elapsed_time = time.time() - start_time
        print(f"\n✅ 执行完成！总耗时: {elapsed_time:.2f}s\n")

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
        print("\n\n⚠️  用户中断执行")
        log_msg("WARNING", "用户中断执行")
    except Exception as e:
        print(f"\n\n❌ 执行失败: {e}")
        log_exception(e, "主程序执行失败")
        raise


if __name__ == "__main__":
    main()
