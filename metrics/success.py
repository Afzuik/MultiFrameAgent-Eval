"""metrics.success —— 任务成败判定层（§4.4 / §7.1 的 SR 口径）。

编排器、汇总层与 dashboard 统一从这里取"一次运行是否成功"：
- load_task()        在 tasks/v1/*_tasks.json 中按 task_id 定位任务定义；
- calls_from_trace() 把归一化 Trace 转成 verifier 需要的调用列表；
- verify_run()       委托 tasks.verifiers.verify_task 执行四类检查原语，
                     返回 (passed, findings)，是全项目唯一权威判定入口。

判定逻辑在 tasks.verifiers.py（按需求不得修改）；
本模块只做转发与口径转换。
"""
from __future__ import annotations

import json

from tasks.verifiers import TASKS_DIR, verify_task

__all__ = ["calls_from_trace", "load_task", "verify_run"]


def load_task(task_id: str) -> dict:
    """在 tasks/v1/*_tasks.json 中按 task_id 查找任务并返回其定义。

    找不到时抛出 KeyError：任务库应始终完备，静默返回空字典会掩盖
    task_id 写错等编排层问题。
    """
    for path in sorted(TASKS_DIR.glob("*_tasks.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for task in data.get("tasks", []):
            if task.get("task_id") == task_id:
                return task
    raise KeyError(f"任务集中不存在 task_id={task_id!r}")


def calls_from_trace(trace) -> list[dict]:
    """从归一化 Trace 提取工具调用列表：[{"tool": name, "args": args}]。

    与 protocol.Trace.tool_calls() 等价的 verifier 输入口径；若传入的
    trace 没有 tool_calls() 方法，则按 steps 字段自行提取（鸭子类型兜底）。
    """
    extractor = getattr(trace, "tool_calls", None)
    if callable(extractor):
        return extractor()
    return [
        {"tool": s.tool_name, "args": s.tool_args or {}}
        for s in trace.steps
        if getattr(s, "type", None) == "tool_call" and s.tool_name
    ]


def verify_run(
    task: dict,
    calls: list[dict],
    final_state: dict,
    answer: str,
) -> tuple[bool, list[str]]:
    """执行一次运行的成功判定，委托 tasks.verifiers.verify_task。

    Args:
        task: 任务定义（tasks/v1/*_tasks.json 的单个元素）。
        calls: 工具调用列表（口径见 calls_from_trace）。
        final_state: mock 工具服务的终态（与 verifier 读同一份状态）。
        answer: Agent 的最终回复文本。

    Returns:
        (passed, findings)：passed 为是否通过全部检查；findings 为
        未满足检查的描述列表（空列表表示全部满足、判定通过）。
    """
    return verify_task(task, calls, final_state, answer)
