# Hooks System

## Hook 类型

- **PreToolUse**: 工具执行前（验证、参数修改）
- **PostToolUse**: 工具执行后（自动格式化、检查）
- **Stop**: 会话结束时（最终验证）

## 推荐 Hooks (Python 项目)

### PreToolUse
- **tmux reminder**: 长时间命令建议使用 tmux
- **git push review**: push 前打开编辑器审查
- **doc blocker**: 阻止创建不必要的 .md/.txt 文件

### PostToolUse
- **PR creation**: 记录 PR URL 和 GitHub Actions 状态
- **Ruff format**: 编辑 .py 文件后自动格式化
- **Ruff check**: 编辑 .py 文件后运行 linter
- **print() warning**: 警告编辑文件中的 print() 语句

### Stop
- **print() audit**: 会话结束前检查所有修改文件中的 print()

## Python 格式化工具配置

```bash
# 使用 Ruff 格式化和检查
ruff format src/
ruff check src/ --fix
```

## Auto-Accept 权限

谨慎使用：
- 对可信的、定义明确的计划启用
- 探索性工作时禁用
- 不要使用 dangerously-skip-permissions 标志
- 在 `~/.claude.json` 中配置 `allowedTools`

## TodoWrite 最佳实践

使用 TodoWrite 工具：
- 跟踪多步骤任务进度
- 验证对指令的理解
- 启用实时调整
- 显示细粒度的实现步骤

待办列表揭示：
- 步骤顺序错误
- 缺失项目
- 多余项目
- 粒度错误
- 误解的需求
