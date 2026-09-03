"""metrics.cost_latency —— 成本与延迟指标（§7.3 口径）。

所有函数只读 Trace 字段、无副作用，供编排器、aggregate 与 dashboard 复用：
- total_cost()       模型成本：∑ 步骤 cost_usd，步骤全为 0 时兜底 total_cost_usd；
- wall_time()        端到端耗时：优先 trace.wall_time_s，为 0 时用
                     末-首步骤 ts 差兜底；
- n_tool_calls()     工具调用次数（type == "tool_call" 的步骤数）；
- n_model_turns()    模型回合数（assistant 消息步数，预算口径 §6.4）；
- tool_roundtrip_s() 工具往返耗时：相邻 tool_call→observation 的 ts 差之和。
"""
from __future__ import annotations

from itertools import pairwise

__all__ = [
    "n_model_turns",
    "n_tool_calls",
    "tool_roundtrip_s",
    "total_cost",
    "wall_time",
]


def total_cost(trace) -> float:
    """模型总成本（USD）：∑ 各步骤 cost_usd。

    当所有步骤 cost_usd 全为 0（成本只在汇总行记录）时，
    兜底返回 trace.total_cost_usd。
    """
    step_cost = sum(s.cost_usd for s in trace.steps)
    if step_cost > 0:
        return float(step_cost)
    return float(getattr(trace, "total_cost_usd", 0.0) or 0.0)


def wall_time(trace) -> float:
    """端到端耗时（秒）：优先取 trace.wall_time_s。

    wall_time_s 为 0（未记录）时，用末步骤 ts 减首步骤 ts 兜底；
    任一端点 ts 缺失（为 0）或没有任何步骤时返回 0。
    """
    if getattr(trace, "wall_time_s", 0.0):
        return float(trace.wall_time_s)
    steps = trace.steps
    if not steps:
        return 0.0
    first_ts = steps[0].ts
    last_ts = steps[-1].ts
    if not first_ts or not last_ts:
        return 0.0
    return float(last_ts - first_ts)


def n_tool_calls(trace) -> int:
    """工具调用次数：type == "tool_call" 的步骤数。"""
    return sum(1 for s in trace.steps if s.type == "tool_call")


def n_model_turns(trace) -> int:
    """模型回合数：type == "message" 且 role == "assistant" 的步骤数。"""
    return sum(
        1 for s in trace.steps if s.type == "message" and s.role == "assistant"
    )


def tool_roundtrip_s(trace) -> float:
    """工具往返总耗时（秒）：相邻 tool_call→observation 对的 ts 差求和。

    只统计紧邻的 tool_call→observation 两步；对中任一步 ts 缺失（为 0）
    时该对记 0（缺 ts 记 0），不纳入求和。
    """
    steps = trace.steps
    total = 0.0
    for call, obs in pairwise(steps):
        if call.type != "tool_call" or obs.type != "observation":
            continue
        if call.ts and obs.ts:
            total += obs.ts - call.ts
    return float(total)
