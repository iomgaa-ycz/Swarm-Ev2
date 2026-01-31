"""Swarm-Ev2 主程序入口。

自动化运行 Kaggle 竞赛的主执行流程。
"""

import shutil

from utils.config import load_config
from utils.logger_system import init_logger, log_msg, log_exception
from utils.workspace_builder import build_workspace, validate_dataset
from utils.prompt_builder import PromptBuilder
from agents.coder_agent import CoderAgent
from core.executor.interpreter import Interpreter
from core.state import Journal
from core.orchestrator import Orchestrator


def main() -> None:
    """主执行函数。

    执行流程:
        1. 环境准备：加载配置、验证数据集
        2. 工作空间构建：创建目录、复制/链接数据
        3. 组件初始化：日志、Agent、Orchestrator
        4. 运行主循环：自动生成、执行、评估代码
        5. 结果展示：最佳方案、指标、保存路径
    """
    print("\n🚀 启动 Swarm-Ev2 自动化竞赛系统\n")

    try:
        # ============================================================
        # Phase 1: 环境准备
        # ============================================================
        print("[1/5] 环境准备...")

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
        print("\n[2/5] 工作空间构建...")

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
        print("\n[3/5] 组件初始化...")

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

        # 初始化 PromptBuilder
        prompt_builder = PromptBuilder(obfuscate=False)
        log_msg("INFO", "Prompt 构建器初始化完成")

        # 初始化 CoderAgent
        agent = CoderAgent(
            name="CoderAgent",
            config=config,
            prompt_builder=prompt_builder,
            interpreter=interpreter,
        )
        log_msg("INFO", "CoderAgent 初始化完成")

        # 初始化 Journal
        journal = Journal()
        log_msg("INFO", "Journal 初始化完成")

        # 初始化 Orchestrator
        orchestrator = Orchestrator(
            agent=agent,
            config=config,
            journal=journal,
            task_desc=task_description,
        )
        log_msg("INFO", "Orchestrator 初始化完成")
        print("✅ 所有组件初始化完成")

        # ============================================================
        # Phase 4: 运行主循环
        # ============================================================
        print(f"\n[4/5] 运行主循环（最大步数: {config.agent.max_steps}）...\n")

        best_node = orchestrator.run()

        # ============================================================
        # Phase 5: 结果展示
        # ============================================================
        print("\n[5/5] 结果展示...")

        if best_node is None:
            print("❌ 未找到有效方案")
            log_msg("WARNING", "未找到有效方案")
        else:
            print("\n🎉 最佳方案已生成:")
            print(f"  节点 ID: {best_node.id}")
            print(f"  评估指标: {best_node.metric_value}")
            print(f"  越小越好: {best_node.lower_is_better}")
            print(f"  执行时间: {best_node.exec_time:.2f}s")
            print(
                f"  代码路径: {config.project.workspace_dir / 'best_solution' / 'solution.py'}"
            )
            print(
                f"  提交文件: {config.project.workspace_dir / 'best_solution' / 'submission.csv'}"
            )

            log_msg(
                "INFO", f"最佳方案: ID={best_node.id}, metric={best_node.metric_value}"
            )

        # 总结统计
        print("\n📊 执行统计:")
        print(f"  总节点数: {len(journal.nodes)}")
        print(f"  成功节点: {len([n for n in journal.nodes if not n.is_buggy])}")
        print(f"  失败节点: {len([n for n in journal.nodes if n.is_buggy])}")
        print(f"  日志目录: {log_dir}")

        print("\n✅ 执行完成！\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断执行")
        log_msg("WARNING", "用户中断执行")
    except Exception as e:
        print(f"\n\n❌ 执行失败: {e}")
        log_exception(e, "主程序执行失败")


if __name__ == "__main__":
    main()
