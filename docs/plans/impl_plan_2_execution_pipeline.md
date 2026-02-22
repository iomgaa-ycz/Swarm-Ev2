# 实施计划 P2：执行管道层

**范围**: Agent 上下文、CoderAgent 分发、PromptManager、Orchestrator 核心方法、Prompt 模板。
**依赖**: P1 完成后才能执行（依赖 Node.dead/debug_attempts/approach_tag 字段）
**估计改动量**: 7 个文件，约 230 行新增/修改

---

## 2.1 `agents/base_agent.py` [MODIFY]

### 修改位置 1：顶部 import（第 8 行）

**旧：**
```python
from typing import TYPE_CHECKING, Any, Optional, Literal
```
**新：**
```python
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Literal
```

### 修改位置 2：AgentContext Docstring Attributes（第 26–39 行）

在 `gene_plan: merge 任务专用 - 基因交叉计划` 之后追加：
```
        primary_parent: merge 任务专用 - 贡献基因最多的父代（取代 parent_a 作为语义主父代）
        gene_sources: merge 任务专用 - {locus: source_node_id} 字典，记录每个位点的来源
        draft_history: draft 任务专用 - 已用方法标签列表，用于多样性引导
```

也把 `task_type` 条目改为：
```
        task_type: 任务类型（"draft"、"explore"、"merge" 或 "mutate"）
```

### 修改位置 3：Literal（第 42 行）

**旧：**
```python
task_type: Literal["explore", "merge", "mutate"]
```
**新：**
```python
task_type: Literal["draft", "explore", "merge", "mutate"]
```

### 修改位置 4：merge/mutate 专用字段（第 52–57 行）

**旧：**
```python
    # merge 任务专用字段
    parent_a: Optional[Node] = None
    parent_b: Optional[Node] = None
    gene_plan: Optional[dict] = None
    # mutate 任务专用字段
    target_gene: Optional[str] = None
```
**新：**
```python
    # merge 任务专用字段
    parent_a: Optional[Node] = None       # 保留：兼容旧 execute_merge_task 过渡期
    parent_b: Optional[Node] = None       # 保留：兼容旧 execute_merge_task 过渡期
    primary_parent: Optional[Node] = None # 新增：贡献基因最多的父代
    gene_plan: Optional[dict] = None
    gene_sources: Optional[Dict[str, str]] = None  # 新增：{locus: source_node_id}
    # mutate 任务专用字段
    target_gene: Optional[str] = None
    # draft 任务专用字段
    draft_history: Optional[List[str]] = None  # 新增：已用方法标签列表
```

---

## 2.2 `agents/coder_agent.py` [MODIFY]

### 修改位置 1：generate() 分发（第 52–67 行）

**旧：**
```python
        try:
            if context.task_type == "explore":
                node = self._explore(context)
                # 静态预验证 + LLM 自修复
                node = self._validate_and_fix(node, context)
                return AgentResult(node=node, success=True)
            elif context.task_type == "merge":
```
**新：**
```python
        try:
            if context.task_type in ("draft", "explore"):
                node = self._explore(context)
                # 静态预验证 + LLM 自修复
                node = self._validate_and_fix(node, context)
                return AgentResult(node=node, success=True)
            elif context.task_type == "merge":
```

### 修改位置 2：_explore() 日志（第 95–98 行）

**旧：**
```python
        log_msg(
            "INFO",
            f"{self.name} 开始 explore (parent_id={context.parent_node.id if context.parent_node else 'None'})",
        )
```
**新：**
```python
        log_msg(
            "INFO",
            f"{self.name} 开始 {context.task_type} (parent_id={context.parent_node.id if context.parent_node else 'None'})",
        )
```

### 修改位置 3：_explore() build_prompt 调用（第 106–121 行）

**旧（两处改动）：**
```python
        prompt = self.prompt_manager.build_prompt(
            "explore",        # ← 硬编码 "explore"
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
```
**新：**
```python
        prompt = self.prompt_manager.build_prompt(
            context.task_type,   # ← 使用实际 task_type（"draft" → draft.j2, "explore" → explore.j2）
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
                "draft_history": getattr(context, "draft_history", None),  # 新增
            },
        )
```

### 修改位置 4：_merge() 日志（第 272–276 行）

**旧：**
```python
        log_msg(
            "INFO",
            f"{self.name} 开始 merge (parent_a={context.parent_a.id[:8] if context.parent_a else 'None'}, "
            f"parent_b={context.parent_b.id[:8] if context.parent_b else 'None'})",
        )
```
**新：**
```python
        log_msg(
            "INFO",
            f"{self.name} 开始 merge (primary_parent={context.primary_parent.id[:8] if context.primary_parent else 'None'})",
        )
```

### 修改位置 5：_merge() 验证（第 279–280 行）

**旧：**
```python
        if not context.parent_a or not context.parent_b or not context.gene_plan:
            raise ValueError("merge 任务需要 parent_a, parent_b, gene_plan 字段")
```
**新：**
```python
        if not context.primary_parent or not context.gene_plan:
            raise ValueError("merge 任务需要 primary_parent, gene_plan 字段")
```

### 修改位置 6：_merge() build_prompt（第 286–301 行）

**旧：**
```python
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
```
**新：**
```python
        prompt = self.prompt_manager.build_prompt(
            "merge",
            self.name,
            {
                "task_desc": context.task_desc,
                "primary_parent": context.primary_parent,   # 改：取代 parent_a/parent_b
                "gene_plan": context.gene_plan,
                "time_remaining": time_remaining,
                "steps_remaining": steps_remaining,
                "device_info": context.device_info,
                "conda_packages": context.conda_packages,
                "conda_env_name": context.conda_env_name,
                "experience_pool": getattr(context, "experience_pool", None),
            },
        )
```

### 修改位置 7：_merge() Node 创建（第 313–319 行）

**旧：**
```python
        node = Node(
            code=code,
            plan=plan,
            parent_id=context.parent_a.id,  # 主父代
            task_type=context.task_type,
            prompt_data=prompt_data,
        )
```
**新：**
```python
        node = Node(
            code=code,
            plan=plan,
            parent_id=context.primary_parent.id,  # 改：使用 primary_parent
            task_type=context.task_type,
            prompt_data=prompt_data,
        )
```

---

## 2.3 `utils/prompt_manager.py` [MODIFY]

### 修改位置 1：inject_top_k_skills() Docstring（第 126–128 行）

将 `task_type: 任务类型（"explore" / "merge" / "mutate"）` 改为：
```
            task_type: 任务类型（"draft" / "explore" / "merge" / "mutate"）
```

### 修改位置 2：inject_top_k_skills() 查询逻辑（第 147–151 行）

**旧：**
```python
        records = experience_pool.query(
            task_type=task_type,
            k=k,
            output_quality=(">", 0.5),
        )
```
**新：**
```python
        # "draft" 与 "explore" 共用经验池
        query_type = "explore" if task_type == "draft" else task_type
        records = experience_pool.query(
            task_type=query_type,
            k=k,
            output_quality=(">", 0.5),
        )
```

---

## 2.4 `core/orchestrator.py` [MODIFY]

### 修改位置 1：删除 `_try_immediate_debug()`，替换为 `_debug_chain()`

**删除**整个 `_try_immediate_debug()` 函数（第 1638–1701 行，含 Docstring）。

**在原位置插入** `_debug_chain()`：

```python
    def _debug_chain(
        self,
        node: Node,
        agent: BaseAgent,
        context: AgentContext,
        max_attempts: Optional[int] = None,
    ) -> Node:
        """链式 Debug：最多 max_attempts 次迭代修复，耗尽后标记 node.dead=True。

        与废弃的 _try_immediate_debug() 的区别：
        - 支持多次迭代（每次将上一次输出作为下一次输入）
        - 全部失败后设置 node.dead = True
        - debug_attempts 计入 node.debug_attempts

        Args:
            node: 执行后的节点
            agent: 执行该节点的 Agent
            context: Agent 执行上下文
            max_attempts: 最大 debug 次数（默认使用 config.evolution.solution.debug_max_attempts）

        Returns:
            修复成功的节点；或 dead=True 的原节点（所有尝试失败）
        """
        if max_attempts is None:
            max_attempts = getattr(
                self.config.evolution.solution, "debug_max_attempts", 2
            )

        # 不可恢复的错误类型跳过 debug（TimeoutExpired 为旧版兼容别名）
        skip_exc_types = {None, "TimeoutError", "TimeoutExpired", "MemoryError"}
        if node.exc_type in skip_exc_types:
            return node

        current = node
        for attempt in range(1, max_attempts + 1):
            log_msg(
                "INFO",
                f"Debug chain attempt {attempt}/{max_attempts}: "
                f"node={node.id[:8]}, exc_type={current.exc_type}",
            )

            debug_context = {
                "buggy_code": current.code,
                "exc_type": current.exc_type or "",
                "term_out": current.term_out if isinstance(current.term_out, str) else "",
                "task_desc": context.task_desc,
                "data_preview": "",
                "device_info": context.device_info,
                "conda_packages": context.conda_packages,
                "conda_env_name": context.conda_env_name,
            }

            fixed_node = agent._debug(debug_context)

            if fixed_node and fixed_node.code and fixed_node.code != current.code:
                fix_exec_result = self._execute_code(fixed_node.code, fixed_node.id)
                fixed_node.term_out = "\n".join(fix_exec_result.term_out)
                fixed_node.exec_time = fix_exec_result.exec_time
                fixed_node.exc_type = fix_exec_result.exc_type
                fixed_node.exc_info = (
                    str(fix_exec_result.exc_info) if fix_exec_result.exc_info else None
                )
                # 保留原始 parent_id 和 task_type
                fixed_node.parent_id = node.parent_id
                fixed_node.task_type = node.task_type
                fixed_node.debug_attempts = attempt

                if fixed_node.exc_type is None:
                    log_msg("INFO", f"Debug chain 成功（第 {attempt} 次）")
                    return fixed_node

                # 仍有错误，继续下一轮（用新输出作为下一轮输入）
                log_msg(
                    "INFO",
                    f"Debug attempt {attempt} 仍有错误 ({fixed_node.exc_type})，继续",
                )
                current = fixed_node
            else:
                log_msg("INFO", f"Debug attempt {attempt} 未产生有效代码，停止")
                break

        # 所有尝试耗尽，标记为 dead
        node.dead = True
        node.debug_attempts = max_attempts
        log_msg(
            "WARNING",
            f"Debug chain 耗尽（{max_attempts} 次），节点 {node.id[:8]} 标记为 dead",
        )
        return node
```

### 修改位置 2：`_step_task()` 调用更新（第 447 行）

**旧：**
```python
            node = self._try_immediate_debug(node, agent, context)
```
**新：**
```python
            node = self._debug_chain(node, agent, context)
```

### 修改位置 3：在 `_step_task()` 之后（或类方法末尾）追加三个新方法

**追加 `_build_draft_history()`：**

```python
    def _build_draft_history(self) -> List[str]:
        """收集已有方案的 approach_tag 列表（用于 Phase 1 多样性引导）。

        从 Journal 中提取所有非死节点的 approach_tag，去重后按出现顺序返回。

        Returns:
            approach_tag 列表（已去重）
        """
        seen: set = set()
        history: List[str] = []
        with self.journal_lock:
            for node in self.journal.nodes:
                tag = node.approach_tag
                if tag and not node.dead and tag not in seen:
                    seen.add(tag)
                    history.append(tag)
        return history
```

**追加 `_draft_step()`：**

```python
    def _draft_step(self, draft_history: Optional[List[str]] = None) -> Optional[Node]:
        """执行单个 draft 步骤（Phase 1 专用）。

        与 _step_task() 的区别：
        - 始终使用 task_type="draft"（对应 draft.j2 模板，无父代）
        - 注入 draft_history（已用方法列表，用于多样性引导）
        - 使用 _debug_chain() 替代单次 debug

        Args:
            draft_history: 已用方法标签列表（None 表示首次）

        Returns:
            生成的节点（失败时返回 None）
        """
        try:
            # draft 使用 explore 类型 Agent（功能相同）
            if self.task_dispatcher:
                agent = self.task_dispatcher.select_agent("explore")
            else:
                agent = random.choice(self.agents)

            self._prepare_step()

            with self.journal_lock:
                current_step = len(self.journal.nodes)

            context = AgentContext(
                task_type="draft",
                parent_node=None,
                journal=self.journal,
                config=self.config,
                start_time=self.start_time,
                current_step=current_step,
                task_desc=self.task_desc,
                device_info=self.device_info,
                conda_packages=self.conda_packages,
                conda_env_name=self.conda_env_name,
                experience_pool=self.experience_pool,
                draft_history=draft_history,
            )

            result = agent.generate(context)

            if not result.success or result.node is None:
                log_msg("WARNING", f"{agent.name} draft 生成失败: {result.error}")
                return None

            node = result.node

            exec_result = self._execute_code(node.code, node.id)
            node.term_out = "\n".join(exec_result.term_out)
            node.exec_time = exec_result.exec_time
            node.exc_type = exec_result.exc_type
            node.exc_info = str(exec_result.exc_info) if exec_result.exc_info else None

            node = self._debug_chain(node, agent, context)

            self._review_node(node, parent_node=None)

            with self.save_lock:
                self._save_node_solution(node)

            with self.journal_lock:
                self.journal.append(node)
                self._update_best_node(node)

            if self.experience_pool:
                self._write_experience_pool(agent.name, "draft", node)

            self._print_node_summary(node)

            log_msg(
                "INFO",
                f"{agent.name} draft 完成: is_buggy={node.is_buggy}, dead={node.dead}",
            )
            return node

        except Exception as e:
            log_exception(e, "Orchestrator _draft_step() 执行失败")
            return None
```

**追加 `run_epoch_draft()`：**

```python
    def run_epoch_draft(self, total_budget: int) -> List[Node]:
        """执行 Phase 1 Draft Epoch（纯 draft 模式，直到 valid_pool 达标或预算耗尽）。

        终止条件（任一满足）：
        1. valid_pool（非 buggy，非 dead）数量 >= phase1_target_nodes
        2. 已执行步数 >= total_budget
        3. 时间限制到达

        Args:
            total_budget: 最大步数（dead + alive 共同计入）

        Returns:
            本 epoch 生成的所有 Node 列表（含 dead 节点）
        """
        phase1_target = getattr(
            self.config.evolution.solution, "phase1_target_nodes", 8
        )
        generated: List[Node] = []
        step = 0

        log_msg(
            "INFO",
            f"===== Phase 1 Draft Epoch 开始 (target={phase1_target}, budget={total_budget}) =====",
        )

        while step < total_budget:
            if self._check_time_limit():
                break

            # 检查终止条件
            with self.journal_lock:
                valid_pool = [
                    n for n in self.journal.nodes if not n.is_buggy and not n.dead
                ]

            if len(valid_pool) >= phase1_target:
                log_msg(
                    "INFO",
                    f"Phase 1 达标：valid_pool={len(valid_pool)}/{phase1_target}，进入 Phase 2",
                )
                break

            draft_history = self._build_draft_history() or None

            node = self._draft_step(draft_history)
            step += 1

            if node:
                generated.append(node)

            with self.journal_lock:
                current_valid = len([n for n in self.journal.nodes if not n.is_buggy and not n.dead])
            log_msg(
                "INFO",
                f"Phase 1 进度: step={step}/{total_budget}, "
                f"valid={current_valid}/{phase1_target}",
            )

        log_msg(
            "INFO",
            f"===== Phase 1 Draft Epoch 完成: 共 {len(generated)} 个节点 =====",
        )
        return generated
```

注意：`List` 类型需要确认 orchestrator.py 顶部已有 `from typing import List` import（搜索现有 import 确认或追加）。

### 修改位置 4：`execute_merge_task()` 签名与函数体（第 1742–1832 行）

**旧签名（第 1742–1744 行）：**
```python
    def execute_merge_task(
        self, parent_a: Node, parent_b: Node, gene_plan: Dict
    ) -> Optional[Node]:
```
**新签名：**
```python
    def execute_merge_task(
        self,
        primary_parent: Node,
        gene_plan: Dict,
        gene_sources: Optional[Dict[str, str]] = None,
    ) -> Optional[Node]:
```

**旧 Docstring 参数（第 1748–1751 行）：**
```python
        Args:
            parent_a: 父代 A
            parent_b: 父代 B
            gene_plan: 基因交叉计划
```
**新：**
```python
        Args:
            primary_parent: 贡献基因最多的父代（用于 merge.j2 参考框架）
            gene_plan: 基因交叉计划（pheromone_with_degenerate_check 的输出）
            gene_sources: {locus: source_node_id} 字典（可选，用于 node.metadata 记录）
```

**旧日志（第 1763–1765 行）：**
```python
            log_msg(
                "INFO",
                f"{agent.name} 开始 merge (parent_a={parent_a.id[:8]}, parent_b={parent_b.id[:8]})",
            )
```
**新：**
```python
            log_msg(
                "INFO",
                f"{agent.name} 开始 merge (primary_parent={primary_parent.id[:8]})",
            )
```

**旧 AgentContext 构造（第 1771–1786 行）：**
```python
            context = AgentContext(
                task_type="merge",
                parent_node=None,  # merge 不使用 parent_node
                journal=self.journal,
                config=self.config,
                start_time=self.start_time,
                current_step=current_step,
                task_desc=self.task_desc,
                device_info=self.device_info,
                conda_packages=self.conda_packages,
                conda_env_name=self.conda_env_name,
                parent_a=parent_a,
                parent_b=parent_b,
                gene_plan=gene_plan,
                experience_pool=self.experience_pool,
            )
```
**新：**
```python
            context = AgentContext(
                task_type="merge",
                parent_node=None,
                journal=self.journal,
                config=self.config,
                start_time=self.start_time,
                current_step=current_step,
                task_desc=self.task_desc,
                device_info=self.device_info,
                conda_packages=self.conda_packages,
                conda_env_name=self.conda_env_name,
                primary_parent=primary_parent,       # 改
                gene_plan=gene_plan,
                gene_sources=gene_sources,            # 新增
                experience_pool=self.experience_pool,
            )
```

在 `node = result.node` 之后（原第 1795 行），追加 gene_sources 存储：
```python
            node = result.node
            # 记录基因来源（用于分析）
            if gene_sources:
                node.metadata["gene_sources"] = gene_sources
```

**旧即时 Debug（第 1805 行）：**
```python
            node = self._try_immediate_debug(node, agent, context)
```
**新：**
```python
            node = self._debug_chain(node, agent, context)
```

### 修改位置 5：`execute_mutate_task()` Debug 调用（第 1892 行）

**旧：**
```python
            node = self._try_immediate_debug(node, agent, context)
```
**新：**
```python
            node = self._debug_chain(node, agent, context)
```

### 修改位置 6：`_review_node()` 添加 approach_tag 提取（第 769–775 行之后）

当前代码（约第 769–775 行）：
```python
        node.analysis = review_data.get("key_change", "")  # 兼容旧字段
        node.analysis_detail = {
            "key_change": review_data.get("key_change", ""),
            "insight": review_data.get("insight", ""),
            "bottleneck": review_data.get("bottleneck"),
            "suggested_direction": review_data.get("suggested_direction"),
        }

        # Phase 8: 存储 Review 调试数据
        node.metadata["review_debug"] = review_debug
```
**在 `node.analysis_detail` 赋值后、Phase 8 注释之前插入：**
```python
        # Phase 7.5: 提取 approach_tag（仅非 buggy 节点）
        if not node.is_buggy:
            approach_tag = review_data.get("approach_tag")
            if approach_tag:
                node.approach_tag = approach_tag
                log_msg("DEBUG", f"节点 {node.id[:8]} approach_tag: {approach_tag}")
```

### 修改位置 7：`_get_review_tool_schema()` 添加 approach_tag（第 1389–1393 行）

**在 `"suggested_direction"` 字段之后，`required` 列表之前插入：**

```python
                    "approach_tag": {
                        "type": "string",
                        "description": (
                            "本方案的核心方法摘要（1 句话，如 'LightGBM + 5-fold CV + log1p feature'），"
                            "供后续 Draft 多样性引导使用。非 buggy 节点必填。"
                        ),
                        "nullable": True,
                    },
```

注意：`approach_tag` **不加入** `required` 列表（允许 LLM 在 buggy 节点时省略）。

---

## 2.5 `benchmark/mle-bench/prompt_templates/draft.j2` [NEW]

从 `explore.j2` 派生，关键区别：
1. 删除第 21–85 行（`{% if parent_node %}...{% endif %}` 整个块）
2. 在 CONTEXT 开头添加 `draft_history` 多样性约束块

完整内容：

```jinja2
{# ===================================================================== #}
{# DRAFT TASK PROMPT TEMPLATE                                          #}
{# 用于 Phase 1 纯 draft 模式：不参考父代，全力探索多样化方案              #}
{# ===================================================================== #}

{# SECTION: ROLE [EVOLVABLE] #}
{{ load_agent_config(agent_id, "role") }}
{# END SECTION: ROLE #}

{# SECTION: FORMAT [STATIC_SKILL] #}
{{ load_skill("static/output_format") }}
{# END SECTION: FORMAT #}

{# SECTION: TASK #}
# Task Description

{{ task_desc }}
{# END SECTION: TASK #}

{# SECTION: CONTEXT #}
{% if draft_history %}
# Approaches Already Tried

The following approaches have already been attempted in this session.
**You MUST propose a distinctly different method.**

{% for tag in draft_history %}
- {{ tag }}
{% endfor %}

> **Diversity Rule**: Do NOT repeat the same model family or feature engineering strategy listed above.
> Pick a fundamentally different approach (e.g., if LightGBM was tried, try Neural Network, Ridge Regression, or an ensemble of different families).

{% endif %}
{% if memory %}
# Evolution Log

> **CRITICAL**: You MUST read and use this Evolution Log to inform your approach.

{{ memory }}

## Instructions for Using Evolution Log

You MUST:
1. **Read the Changelog** - Understand what changes improved/degraded performance
2. **Respect Constraints** - NEVER repeat errors listed in Constraints section
3. **Build on Insights** - Your approach should address identified bottlenecks
4. **Explore New Directions** - Pick from Unexplored Directions or propose new ones

Your **Thinking** section MUST reference:
- Which insight from Changelog informed your approach
- What specific bottleneck you're targeting
- Why your proposed change should help

**DO NOT** copy previous Thinking verbatim. Your analysis must be specific to this attempt.

{% endif %}

{% if data_preview %}
# Data Overview

{{ data_preview }}
{% endif %}
{# END SECTION: CONTEXT #}

{# SECTION: STRATEGY #}
{{ load_agent_config(agent_id, "strategy_explore") }}
{# END SECTION: STRATEGY #}

{# SECTION: EXAMPLES [DYNAMIC_SKILL] #}
{% if dynamic_skills %}
{{ dynamic_skills }}
{% endif %}
{# END SECTION: EXAMPLES #}

{# SECTION: GUIDELINES #}
{{ load_skill("static/workspace_rules") }}

{{ load_skill("static/code_style") }}

## Time and Resource Constraints

- **Total Time Remaining**: {{ time_str }}
- **Total Steps Remaining**: {{ steps_remaining }}

## System Environment

- **Device**: {{ device_info | default("CPU only") }}
- **Conda Environment**: `{{ conda_env_name | default("python") }}`
{% if conda_packages %}
- **Installed Packages**: {{ conda_packages }}
{% endif %}

**Note**: Use time and steps efficiently. If GPU is available, prioritize GPU-accelerated solutions. (GPU > Multi-core CPU > Single-core CPU)
{# END SECTION: GUIDELINES #}
```

---

## 2.6 `benchmark/mle-bench/prompt_templates/debug.j2` [MODIFY]

### 修改位置：Debug Instructions（第 50–56 行）

**旧：**
```
1. **Analyze the error carefully** and identify the root cause.
2. **Fix the bug with minimal changes** — do NOT refactor or add new features.
3. Ensure the fixed code:
   - Prints the validation metric: `print(f"Validation metric: {value}")`
   - Saves predictions to `./submission/submission.csv`
   - Uses only available packages
4. Your response: a brief description of the fix (2-3 sentences), followed by a **single complete code block**.
```
**新：**
```
1. **Analyze the error carefully** and identify the root cause.
2. **Fix the bug with minimal changes** — do NOT refactor or add new features.
3. **Preserve structure**: Your fix MUST keep all 7 `# [SECTION: X]` markers intact (`DATA`, `MODEL`, `LOSS`, `OPTIMIZER`, `REGULARIZATION`, `INITIALIZATION`, `TRAINING_TRICKS`).
4. **Minimal change**: Change ONLY what is necessary to fix the error. Do not add, remove, or rename any section marker.
5. Ensure the fixed code:
   - Prints the validation metric: `print(f"Validation metric: {value}")`
   - Saves predictions to `./submission/submission.csv`
   - Uses only available packages
6. Your response: a brief description of the fix (2-3 sentences), followed by a **single complete code block**.
```

---

## 2.7 `benchmark/mle-bench/prompt_templates/merge.j2` [MODIFY]

### 修改位置：scaffold 逻辑（第 20–34 行）

**旧：**
```jinja2
{% set scaffold = parent_a if (parent_a.metric_value or 0) >= (parent_b.metric_value or 0) else parent_b %}
## Reference Solution (fitness={{ scaffold.metric_value if scaffold.metric_value else "N/A" }})

A complete working solution for structural reference (imports, helpers, main logic):

```python
{{ scaffold.code }}
```

{% if scaffold.term_out %}
### Execution Result
```
{{ scaffold.term_out[:1000] }}{% if scaffold.term_out|length > 1000 %}... (truncated){% endif %}
```
{% endif %}
```

**新（直接使用 `primary_parent` 变量，由 Orchestrator 传入）：**
```jinja2
## Reference Solution (fitness={{ primary_parent.metric_value if primary_parent.metric_value else "N/A" }})

A complete working solution for structural reference (imports, helpers, main logic):

```python
{{ primary_parent.code }}
```

{% if primary_parent.term_out %}
### Execution Result
```
{{ primary_parent.term_out[:1000] }}{% if primary_parent.term_out|length > 1000 %}... (truncated){% endif %}
```
{% endif %}
```

注意：删除了 `{% set scaffold = ... %}` 这一行（共删除 1 行），并将所有 `scaffold.` 引用替换为 `primary_parent.`。

---

## 2.8 验证步骤

```bash
# 1. AgentContext 字段验证
conda run -n Swarm-Evo python -c "
from agents.base_agent import AgentContext
import inspect
fields = [f.name for f in AgentContext.__dataclass_fields__.values()]
assert 'primary_parent' in fields, 'primary_parent 缺失'
assert 'draft_history' in fields, 'draft_history 缺失'
assert 'gene_sources' in fields, 'gene_sources 缺失'
assert 'draft' in AgentContext.__dataclass_fields__['task_type'].type.__args__, 'draft 未在 Literal 中'
print('AgentContext OK:', fields)
"

# 2. PromptManager draft 处理验证
conda run -n Swarm-Evo python -c "
from utils.prompt_manager import PromptManager
pm = PromptManager.__new__(PromptManager)
import inspect
src = inspect.getsource(pm.inject_top_k_skills)
assert 'query_type' in src, 'query_type 逻辑缺失'
print('PromptManager draft 处理 OK')
"

# 3. _debug_chain 存在性验证
conda run -n Swarm-Evo python -c "
from core.orchestrator import Orchestrator
assert hasattr(Orchestrator, '_debug_chain'), '_debug_chain 方法缺失'
assert not hasattr(Orchestrator, '_try_immediate_debug'), '_try_immediate_debug 应已删除'
assert hasattr(Orchestrator, '_draft_step'), '_draft_step 方法缺失'
assert hasattr(Orchestrator, 'run_epoch_draft'), 'run_epoch_draft 方法缺失'
print('Orchestrator 新方法 OK')
"

# 4. draft.j2 模板存在验证
conda run -n Swarm-Evo python -c "
from pathlib import Path
draft_tmpl = Path('benchmark/mle-bench/prompt_templates/draft.j2')
assert draft_tmpl.exists(), 'draft.j2 不存在'
content = draft_tmpl.read_text()
assert 'draft_history' in content, 'draft_history 变量缺失'
assert 'parent_node' not in content, 'parent_node 不应在 draft.j2 中'
print('draft.j2 OK')
"

# 5. merge.j2 模板验证
conda run -n Swarm-Evo python -c "
from pathlib import Path
content = Path('benchmark/mle-bench/prompt_templates/merge.j2').read_text()
assert 'primary_parent' in content, 'primary_parent 缺失'
assert '{% set scaffold' not in content, 'scaffold 逻辑应已删除'
print('merge.j2 OK')
"

# 6. 完整导入验证
conda run -n Swarm-Evo python -c "
from agents.base_agent import AgentContext, AgentResult
from agents.coder_agent import CoderAgent
from utils.prompt_manager import PromptManager
from core.orchestrator import Orchestrator
print('所有模块导入 OK')
"
```

---

## 2.9 注意事项

### P3 依赖提示

- `solution_evolution.py` 中 `execute_merge_task(parent_a, parent_b, gene_plan_md)` 调用签名在 P2 执行后会**临时不兼容**。P3 将完整重写 `run_epoch()`，届时更新此调用。
- 如需临时运行，可在 P2 执行后为 `execute_merge_task()` 保留一个兼容重载，P3 完成后删除。

### `List` import 确认

`orchestrator.py` 顶部需要有 `from typing import List`。若无，搜索 `from typing import` 行并追加 `List`。

### `_step_task()` 生命周期

`_step_task()` 在两阶段架构中将被 `_draft_step()` + Phase 2 的 merge/mutate 调用完全取代，但**暂不删除**（P3 重写 main.py 后再删除），以防过渡期回滚需要。
