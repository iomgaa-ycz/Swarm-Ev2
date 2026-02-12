"""CoderAgent 具体实现模块。

提供基于 LLM 的代码生成、执行、评估功能。
支持即时 Debug 循环和静态预验证 + LLM 自动修复。
"""

import time
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

from core.backend import query_with_config
from core.state import Node
from utils.logger_system import log_msg, log_exception
from utils.response import extract_code, extract_text_up_to_code
from utils.code_validator import validate_code

from .base_agent import BaseAgent, AgentContext, AgentResult


class CoderAgent(BaseAgent):
    """代码生成 Agent。

    负责调用 LLM 生成代码并创建 Node 对象。
    代码执行由 Orchestrator 负责，以支持并行执行。

    Attributes:
        name: Agent 名称
        config: 全局配置
        prompt_manager: PromptManager 实例
    """

    def __init__(self, name: str, config, prompt_manager):
        """初始化 CoderAgent。

        Args:
            name: Agent 名称
            config: 全局配置对象
            prompt_manager: PromptManager 实例
        """
        super().__init__(name, config, prompt_manager)
        # interpreter 不再在 Agent 内部使用，由 Orchestrator 统一管理

    def generate(self, context: AgentContext) -> AgentResult:
        """主入口：根据 task_type 分发到具体实现。

        Args:
            context: Agent 执行上下文

        Returns:
            AgentResult 对象
        """
        try:
            if context.task_type == "explore":
                node = self._explore(context)
                # 静态预验证 + LLM 自修复
                node = self._validate_and_fix(node, context)
                return AgentResult(node=node, success=True)
            elif context.task_type == "merge":
                node = self._merge(context)
                node = self._validate_and_fix(node, context)
                return AgentResult(node=node, success=True)
            elif context.task_type == "mutate":
                node = self._mutate(context)
                node = self._validate_and_fix(node, context)
                return AgentResult(node=node, success=True)
            else:
                raise ValueError(f"未知的 task_type: {context.task_type}")

        except Exception as e:
            log_exception(e, f"{self.name} generate() 失败")
            return AgentResult(node=None, success=False, error=str(e))

    def _explore(self, context: AgentContext) -> Node:
        """探索新方案（统一方法）。

        完整流程：
        1. 生成数据预览（可选）
        2. 生成 Memory（历史经验）
        3. 计算剩余时间和步数
        4. 构建 Prompt
        5. 调用 LLM（带重试）
        6. 解析响应（带重试）
        7. 执行代码
        8. 创建 Node 对象

        Args:
            context: Agent 执行上下文

        Returns:
            生成的 Node 对象

        Raises:
            Exception: 如果 LLM 调用或解析失败
        """
        log_msg(
            "INFO",
            f"{self.name} 开始 explore (parent_id={context.parent_node.id if context.parent_node else 'None'})",
        )

        # Phase 1: 准备上下文
        data_preview = self._generate_data_preview()
        memory = context.journal.generate_summary(include_code=False)
        time_remaining, steps_remaining = self._calculate_remaining(context)

        # Phase 2: 构建 Prompt
        prompt = self.prompt_manager.build_prompt(
            "explore",
            self.name,
            {
                "task_desc": context.task_desc,
                "parent_node": context.parent_node,
                "memory": memory,
                "data_preview": data_preview,
                "time_remaining": time_remaining,
                "steps_remaining": steps_remaining,
                "device_info": context.device_info,
                "conda_packages": context.conda_packages,
                "conda_env_name": context.conda_env_name,
                "experience_pool": getattr(context, "experience_pool", None),
            },
        )

        # 记录 prompt 数据用于调试
        prompt_data = self._build_prompt_data(prompt, context.task_type)

        # Phase 3: 调用 LLM（带重试）
        response = self._call_llm_with_retry(prompt, max_retries=5)

        # Phase 4: 解析响应（带重试）
        plan, code = self._parse_response_with_retry(response, max_retries=5)

        # Phase 5: 创建 Node 对象（代码执行由 Orchestrator 负责）
        node = Node(
            code=code,
            plan=plan,
            parent_id=context.parent_node.id if context.parent_node else None,
            task_type=context.task_type,
            prompt_data=prompt_data,
        )

        log_msg(
            "INFO",
            f"{self.name} 代码生成完成: plan={len(plan)} chars, code={len(code)} chars",
        )

        return node

    def _call_llm_with_retry(self, prompt: str, max_retries: int = 5) -> str:
        """调用 LLM 并实现重试机制。

        Args:
            prompt: Prompt 字符串
            max_retries: 最大重试次数（默认 5 次）

        Returns:
            LLM 响应字符串

        Raises:
            Exception: 如果所有重试都失败
        """
        retry_delays = [10, 20, 40, 80]  # 指数退避（秒）

        for attempt in range(max_retries):
            try:
                log_msg(
                    "INFO",
                    f"{self.name} 调用 LLM (attempt {attempt + 1}/{max_retries})",
                )

                response = query_with_config(
                    self.config.llm.code,
                    user_message=prompt,
                    max_tokens=self.config.llm.code.max_tokens,
                )

                log_msg("INFO", f"{self.name} LLM 调用成功")
                return response

            except Exception as e:
                log_msg(
                    "WARNING", f"{self.name} LLM 调用失败 (attempt {attempt + 1}): {e}"
                )

                # 如果还有重试机会，等待后重试
                if attempt < max_retries - 1:
                    delay = retry_delays[min(attempt, len(retry_delays) - 1)]
                    log_msg("INFO", f"{self.name} 等待 {delay}s 后重试...")
                    time.sleep(delay)
                else:
                    # 重试次数耗尽
                    log_msg("ERROR", f"{self.name} LLM 调用失败，已达最大重试次数")
                    raise

    def _parse_response_with_retry(
        self, response: str, max_retries: int = 5
    ) -> Tuple[str, str]:
        """解析 LLM 响应并实现重试机制（针对软格式失败）。

        Args:
            response: LLM 响应字符串
            max_retries: 最大重试次数（默认 5 次，但硬格式失败不重试）

        Returns:
            (plan, code) 元组

        Raises:
            ValueError: 如果解析失败（硬格式失败）
        """
        # 尝试提取代码块
        code = extract_code(response)

        # 硬格式失败：完全没有代码块
        if not code:
            error_msg = "响应中未找到代码块（硬格式失败，不重试）"
            log_msg("ERROR", f"{self.name} {error_msg}")
            raise ValueError(error_msg)

        # 提取 plan（代码块之前的文本）
        plan = extract_text_up_to_code(response)

        log_msg(
            "INFO",
            f"{self.name} 响应解析成功: plan={len(plan)} chars, code={len(code)} chars",
        )
        return plan, code

    def _generate_data_preview(self) -> Optional[str]:
        """生成数据预览（与 AIDE 一致的实现）。

        自动探测 workspace/input/ 目录下的文件结构，
        生成目录树 + CSV 列摘要，供 LLM 理解可用数据。

        Returns:
            数据预览字符串，如果失败则返回 None

        Side Effects:
            保存预览到 workspace/logs/data_preview.md 便于调试检查
        """
        try:
            from utils.data_preview import generate

            input_dir = self.config.project.workspace_dir / "input"
            if not input_dir.exists():
                log_msg("WARNING", f"输入目录不存在: {input_dir}")
                return None

            preview = generate(input_dir)

            # 保存到 logs 目录，便于调试检查 LLM 收到的实际内容
            logs_dir = self.config.project.workspace_dir / "logs"
            logs_dir.mkdir(exist_ok=True)
            preview_file = logs_dir / "data_preview.md"
            preview_file.write_text(preview, encoding="utf-8")
            log_msg("INFO", f"数据预览已保存: {preview_file}")

            log_msg("INFO", f"{self.name} 数据预览生成完成 ({len(preview)} chars)")
            return preview

        except Exception as e:
            log_msg("WARNING", f"{self.name} 数据预览生成失败: {e}")
            return None

    def _merge(self, context: AgentContext) -> Node:
        """基因交叉（merge 任务）。

        Args:
            context: Agent 执行上下文

        Returns:
            生成的 Node 对象
        """
        log_msg(
            "INFO",
            f"{self.name} 开始 merge (parent_a={context.parent_a.id[:8] if context.parent_a else 'None'}, "
            f"parent_b={context.parent_b.id[:8] if context.parent_b else 'None'})",
        )

        # 验证必需字段
        if not context.parent_a or not context.parent_b or not context.gene_plan:
            raise ValueError("merge 任务需要 parent_a, parent_b, gene_plan 字段")

        # 计算剩余时间和步数
        time_remaining, steps_remaining = self._calculate_remaining(context)

        # 构建 merge Prompt
        prompt = self.prompt_manager.build_prompt(
            "merge",
            self.name,
            {
                "task_desc": context.task_desc,
                "parent_a": context.parent_a,
                "parent_b": context.parent_b,
                "gene_plan": context.gene_plan,
                "time_remaining": time_remaining,
                "steps_remaining": steps_remaining,
                "device_info": context.device_info,
                "conda_packages": context.conda_packages,
                "conda_env_name": context.conda_env_name,
                "experience_pool": getattr(context, "experience_pool", None),
            },
        )

        # 记录 prompt 数据用于调试
        prompt_data = self._build_prompt_data(prompt, context.task_type)

        # 调用 LLM
        response = self._call_llm_with_retry(prompt, max_retries=5)

        # 解析响应
        plan, code = self._parse_response_with_retry(response, max_retries=5)

        # 创建 Node
        node = Node(
            code=code,
            plan=plan,
            parent_id=context.parent_a.id,  # 主父代
            task_type=context.task_type,
            prompt_data=prompt_data,
        )

        log_msg("INFO", f"{self.name} merge 完成")
        return node

    def _mutate(self, context: AgentContext) -> Node:
        """基因变异（mutate 任务）。

        Args:
            context: Agent 执行上下文

        Returns:
            生成的 Node 对象
        """
        log_msg(
            "INFO",
            f"{self.name} 开始 mutate (parent={context.parent_node.id[:8] if context.parent_node else 'None'}, "
            f"target_gene={context.target_gene})",
        )

        # 验证必需字段
        if not context.parent_node or not context.target_gene:
            raise ValueError("mutate 任务需要 parent_node, target_gene 字段")

        # 计算剩余时间和步数
        time_remaining, steps_remaining = self._calculate_remaining(context)

        # 构建 mutate Prompt
        prompt = self.prompt_manager.build_prompt(
            "mutate",
            self.name,
            {
                "task_desc": context.task_desc,
                "parent_node": context.parent_node,
                "target_gene": context.target_gene,
                "time_remaining": time_remaining,
                "steps_remaining": steps_remaining,
                "device_info": context.device_info,
                "conda_packages": context.conda_packages,
                "conda_env_name": context.conda_env_name,
                "experience_pool": getattr(context, "experience_pool", None),
            },
        )

        # 记录 prompt 数据用于调试
        prompt_data = self._build_prompt_data(prompt, context.task_type)

        # 调用 LLM
        response = self._call_llm_with_retry(prompt, max_retries=5)

        # 解析响应
        plan, code = self._parse_response_with_retry(response, max_retries=5)

        # 创建 Node
        node = Node(
            code=code,
            plan=plan,
            parent_id=context.parent_node.id,
            task_type=context.task_type,
            prompt_data=prompt_data,
        )

        log_msg("INFO", f"{self.name} mutate 完成")
        return node

    def _debug(self, context: Dict[str, Any]) -> Optional[Node]:
        """修复 buggy 代码（即时 Debug 循环）。

        基于 AIDE 的 debug 机制设计。将完整的错误信息反馈给 LLM 请求修复。

        Args:
            context: 包含以下字段
                - buggy_code: str, 原始代码
                - exc_type: str, 异常类型（可选）
                - term_out: str, 完整执行输出（含 traceback）
                - task_desc: str, 任务描述
                - data_preview: str, 数据预览（可选）
                - validation_errors: str, 预验证错误（可选）
                - device_info: str, 设备信息（可选）
                - conda_packages: str, conda 包信息（可选）
                - conda_env_name: str, conda 环境名（可选）

        Returns:
            修复后的 Node（修复失败返回 None）
        """
        log_msg("INFO", f"{self.name} 开始 Debug")

        try:
            prompt = self.prompt_manager.build_prompt(
                "debug",
                self.name,
                context,
            )

            prompt_data = self._build_prompt_data(prompt, "debug")

            response = self._call_llm_with_retry(prompt, max_retries=3)
            code = extract_code(response)

            if not code:
                log_msg("WARNING", f"{self.name} Debug 未提取到代码")
                return None

            if code == context.get("buggy_code", ""):
                log_msg("WARNING", f"{self.name} Debug 代码未变化")
                return None

            plan = extract_text_up_to_code(response)
            node = Node(
                code=code,
                plan=plan,
                parent_id=None,
                task_type="debug",
                prompt_data=prompt_data,
            )

            log_msg("INFO", f"{self.name} Debug 代码生成完成: {len(code)} chars")
            return node

        except Exception as e:
            log_msg("WARNING", f"{self.name} Debug 失败: {e}")
            return None

    def _validate_and_fix(self, node: Node, context: AgentContext) -> Node:
        """静态预验证 + LLM 自动修复（最多 1 次）。

        在 subprocess 执行前拦截语法错误和明显缺陷。
        验证失败时，将错误信息注入 Prompt 让 LLM 修复 1 次。

        Args:
            node: 原始生成的 Node
            context: Agent 执行上下文

        Returns:
            验证通过或修复后的 Node
        """
        if not node or not node.code:
            return node

        validation = validate_code(node.code)

        if validation.valid:
            # 将 warnings 附加到 plan 中（供后续参考）
            if validation.warnings:
                node.plan += f"\n[Warnings: {'; '.join(validation.warnings)}]"
            return node

        # 预验证失败 → 让 LLM 修复 1 次
        log_msg(
            "INFO",
            f"{self.name} 预验证失败: {validation.errors}，尝试自动修复",
        )

        debug_context = {
            "buggy_code": node.code,
            "term_out": "",
            "validation_errors": "\n".join(validation.errors + validation.warnings),
            "task_desc": context.task_desc,
            "data_preview": "",
            "device_info": context.device_info,
            "conda_packages": context.conda_packages,
            "conda_env_name": context.conda_env_name,
        }

        fixed_node = self._debug(debug_context)
        if fixed_node and fixed_node.code:
            revalidation = validate_code(fixed_node.code)
            if revalidation.valid:
                log_msg("INFO", f"{self.name} 预验证修复成功")
                # 保留原始 parent_id 和 task_type
                fixed_node.parent_id = node.parent_id
                fixed_node.task_type = node.task_type
                return fixed_node

        # 修复失败，返回原节点（标记 warning）
        log_msg(
            "WARNING",
            f"{self.name} 预验证失败且修复未成功: {validation.errors}",
        )
        return node

    def _build_prompt_data(self, prompt: str, task_type: str) -> Dict:
        """构建 prompt 数据字典用于调试记录。

        Args:
            prompt: 完整的 prompt 字符串
            task_type: 任务类型

        Returns:
            包含 prompt 元数据的字典
        """
        return {
            "user_message": prompt,
            "system_message": None,  # 当前实现不使用 system_message
            "model": self.config.llm.code.model,
            "provider": self.config.llm.code.provider,
            "task_type": task_type,
            "agent_name": self.name,
            "timestamp": datetime.now().isoformat(),
        }

    def _calculate_remaining(self, context: AgentContext) -> Tuple[int, int]:
        """计算剩余时间和步数。

        Args:
            context: Agent 执行上下文

        Returns:
            (time_remaining, steps_remaining) 元组（单位：秒，步数）
        """
        # 计算剩余时间
        elapsed_time = time.time() - context.start_time
        total_time_limit = getattr(self.config.agent, "time_limit", 3600)
        time_remaining = max(0, int(total_time_limit - elapsed_time))

        # 计算剩余步数
        max_steps = getattr(self.config.agent, "max_steps", 50)
        steps_remaining = max(0, max_steps - context.current_step)

        return time_remaining, steps_remaining
