# 开发计划：更新 main.py 实现完善的端到端测试

**日期**: 2026-02-01
**目标**: 更新 main.py 以整合 Phase 3 双层群体智能的所有功能，实现完善的端到端测试

---

## 1.1 摘要

更新 main.py，整合经验池、任务分发器、并行评估器、基因注册表、Skill 管理器等 Phase 3 模块，实现双层进化（Solution 层 + Agent 层）的完整端到端测试流程，并生成符合 CLAUDE.md 要求的 Markdown 测试输出记录。

---

## 1.2 审查点

无。

---

## 1.3 拟议变更

| 文件 | 操作 | 修改内容 |
|------|------|----------|
| `main.py` | [MODIFY] | 重构 `main()` 函数，整合双层进化架构 |
| `main.py` | [MODIFY] | 新增 `initialize_agents()` - 创建 Agent 种群 |
| `main.py` | [MODIFY] | 新增 `initialize_evolution_components()` - 初始化进化组件 |
| `main.py` | [MODIFY] | 新增 `run_dual_layer_evolution()` - 双层进化主循环 |
| `main.py` | [MODIFY] | 新增 `generate_markdown_report()` - 生成 Markdown 测试报告 |
| `main.py` | [MODIFY] | 新增 `print_evolution_statistics()` - 输出进化统计 |
| `utils/config.py` | [MODIFY] | 修复 `validate_config()` 函数缺少 `evolution` 配置 |

### 详细变更说明

#### main.py 重构

```python
# 伪代码结构
def main():
    # Phase 1: 环境准备（保持现有逻辑）
    config = load_config()
    validate_dataset(config.data.data_dir)

    # Phase 2: 工作空间构建（保持现有逻辑）
    task_description = build_workspace(...)

    # Phase 3: 初始化组件（新增双层进化架构）
    agents = initialize_agents(config)               # 4 个 Agent
    experience_pool = ExperiencePool(config)        # 共享经验池
    task_dispatcher = TaskDispatcher(agents, ...)   # 任务分发器
    gene_registry = GeneRegistry()                  # 基因注册表
    skill_manager = SkillManager(...)               # Skill 池
    evaluator = ParallelEvaluator(...)              # 并行评估器
    solution_evolution = SolutionEvolution(...)     # Solution 层进化
    agent_evolution = AgentEvolution(...)           # Agent 层进化
    orchestrator = Orchestrator(..., agent_evolution)  # 编排器

    # Phase 4: 运行双层进化主循环
    best_node = orchestrator.run(
        num_epochs=config.evolution.solution.steps_per_epoch // 3 or 3,
        steps_per_epoch=config.evolution.solution.steps_per_epoch
    )

    # Phase 5: 生成 Markdown 测试报告
    generate_markdown_report(journal, experience_pool, ...)

    # Phase 6: 结果展示
    print_evolution_statistics(journal, experience_pool, task_dispatcher)
```

---

## 1.4 验证计划

### 验证命令

```bash
# 运行端到端测试（最小配置）
conda run -n Swarm-Evo python main.py \
  --evolution.solution.population_size=4 \
  --evolution.solution.steps_per_epoch=3 \
  --agent.max_steps=6
```

### 预期结果

- [ ] 程序正常启动，无导入错误
- [ ] 4 个 Agent 初始化成功
- [ ] 经验池初始化并可以写入记录
- [ ] Orchestrator 使用双层进化模式运行
- [ ] 日志输出显示 Step/Epoch 进度
- [ ] 生成 Markdown 测试报告到 `tests/outputs/`
- [ ] 最终输出进化统计（节点数、成功率、Agent 得分等）

### 输出文件验证

```bash
# 检查 Markdown 报告是否生成
ls -la tests/outputs/main_execution_*.md

# 检查经验池 JSON 是否保存
ls -la workspace/evolution/experience_pool.json
```

---

## 变更文件清单

1. `main.py` - 主程序入口重构
2. `utils/config.py` - 配置验证修复（添加 evolution 配置）
