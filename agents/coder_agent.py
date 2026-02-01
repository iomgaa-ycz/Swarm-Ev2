"""CoderAgent 具体实现模块。

提供基于 LLM 的代码生成、执行、评估功能。
"""

import time
from typing import Optional, Tuple

from core.backend import query as backend_query
from core.executor.interpreter import Interpreter
from core.state import Node
from utils.logger_system import log_msg, log_exception
from utils.response import extract_code, extract_text_up_to_code

from .base_agent import BaseAgent, AgentContext, AgentResult


class CoderAgent(BaseAgent):
    """代码生成 Agent。

    负责调用 LLM 生成代码、执行代码并创建 Node 对象。

    Attributes:
        name: Agent 名称
        config: 全局配置
        prompt_builder: Prompt 构建器
        interpreter: 代码执行器
    """

    def __init__(
        self,
        name: str,
        config,
        prompt_builder,
        interpreter: Interpreter,
    ):
        """初始化 CoderAgent。

        Args:
            name: Agent 名称
            config: 全局配置对象
            prompt_builder: Prompt 构建器实例
            interpreter: 代码执行器实例
        """
        super().__init__(name, config, prompt_builder)
        self.interpreter = interpreter

    def generate(self, context: AgentContext) -> AgentResult:
        """主入口：根据 task_type 分发到具体实现。

        Args:
            context: Agent 执行上下文

        Returns:
            AgentResult 对象

        Raises:
            NotImplementedError: 如果 task_type 不是 "explore"
        """
        try:
            if context.task_type == "explore":
                node = self._explore(context)
                return AgentResult(node=node, success=True)
            elif context.task_type == "merge":
                raise NotImplementedError("Phase 2 暂不支持 merge 任务类型")
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
        prompt = self.prompt_builder.build_explore_prompt(
            task_desc=context.task_desc,
            parent_node=context.parent_node,
            memory=memory,
            data_preview=data_preview,
            time_remaining=time_remaining,
            steps_remaining=steps_remaining,
        )

        # Phase 3: 调用 LLM（带重试）
        response = self._call_llm_with_retry(prompt, max_retries=5)

        # Phase 4: 解析响应（带重试）
        plan, code = self._parse_response_with_retry(response, max_retries=5)

        # Phase 5: 执行代码
        exec_result = self.interpreter.run(code, reset_session=True)

        # Phase 6: 创建 Node 对象
        node = Node(
            code=code,
            plan=plan,
            parent_id=context.parent_node.id if context.parent_node else None,
            task_type=context.task_type,
            term_out="\n".join(exec_result.term_out),
            exec_time=exec_result.exec_time,
            exc_type=exec_result.exc_type,
            exc_info=str(exec_result.exc_info) if exec_result.exc_info else None,
            is_buggy=not exec_result.success,
        )

        log_msg(
            "INFO",
            f"{self.name} 完成 explore: is_buggy={node.is_buggy}, exec_time={node.exec_time:.2f}s",
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

                response = backend_query(
                    system_message=None,
                    user_message=prompt,
                    model=self.config.llm.code.model,
                    provider=self.config.llm.code.provider,
                    temperature=self.config.llm.code.temperature,
                    max_tokens=self.config.llm.code.max_tokens,
                    api_key=self.config.llm.code.api_key,
                    base_url=getattr(self.config.llm.code, "base_url", None),
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

        # 软格式验证：检查代码是否看起来合理
        # Phase 2 简化实现：只要有代码块就认为成功
        # Phase 3 可添加更严格的验证（如 JSON 格式检查）

        log_msg(
            "INFO",
            f"{self.name} 响应解析成功: plan={len(plan)} chars, code={len(code)} chars",
        )
        return plan, code

    def _generate_data_preview(self) -> Optional[str]:
        """生成数据预览。

        Returns:
            数据预览字符串，如果失败或未启用则返回 None
        """
        # Phase 2 简化实现：暂不生成数据预览
        # 原因：data_preview.generate_preview() 需要知道数据路径
        # Phase 2.4 Orchestrator 实现时会提供完整的数据路径
        if not getattr(self.config.agent, "data_preview", False):
            return None

        try:
            # TODO: Phase 2.4 实现
            # from utils.data_preview import generate_preview
            # return generate_preview(self.config.project.data_dir)
            log_msg("INFO", f"{self.name} 数据预览生成已启用，但 Phase 2.3 暂不实现")
            return None
        except Exception as e:
            log_msg("WARNING", f"{self.name} 数据预览生成失败: {e}")
            return None

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
