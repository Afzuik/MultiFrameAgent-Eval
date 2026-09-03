"""judge.prompts —— LLM-as-Judge 的 prompt 构造（§8.1/§8.2）。

build_judge_messages(task, compressed_trace, final_answer) -> list[dict]
- system：中文评分指令，含 rubric 全文（来自 rubric.RUBRIC，代码与提示
  所见一致）与打分输出协议（只输出 JSON：{"score": 1~5, "reason": ...}）；
- user：任务目标（仅 task.user_goal）+ 压缩轨迹 + 最终答案。

匿名化（§8.3①）：消息文本不携带框架名/模型名——任务只取 user_goal
字段，不序列化整个 task dict；轨迹已在 compress 阶段剔除标识名。
"""
from __future__ import annotations

from judge.rubric import RUBRIC

__all__ = ["build_judge_messages"]


def _rubric_lines() -> str:
    """把 rubric 渲染成"5 分：…\n4 分：…"的全文（高分在前）。"""
    return "\n".join(
        f"{score} 分：{RUBRIC[score]}" for score in sorted(RUBRIC, reverse=True)
    )


def _build_system() -> str:
    """中文 system 提示：评分对象 + rubric 全文 + 输出协议。"""
    lines = [
        "你是一名严格的 Agent 运行轨迹质量评分器。",
        "你会收到一次工具调用任务的【用户目标】与 Agent 的完整【运行轨迹】",
        "（含模型消息、工具调用、工具返回与最终答案），",
        "请对这次运行的整体质量给出 1~5 分的轨迹分。",
        "判定要点：先结合最终答案与关键工具调用判断目标是否达成；",
        "若达成再看路径效率（无效调用/错误尝试/绕路按档扣分）；",
        "若未达成则判断方向是否正确、是否违反约束，落在对应档位。",
        "只依据下方给出的轨迹文本评分，不要臆测轨迹之外的信息。",
        "",
        "评分规则（rubric，1~5 分）：",
        _rubric_lines(),
        "",
        "输出协议：只输出一个 JSON 对象，不要输出任何其他内容",
        "（不要代码围栏、不要 markdown、不要多余解释）：",
        '{"score": <1~5 整数>, "reason": "<一句话中文理由>"}',
    ]
    return "\n".join(lines)


def _build_user(task: dict, compressed_trace: str, final_answer: str) -> str:
    """user 消息：任务目标 + 压缩轨迹 + 最终答案（§8.1 输入构造）。"""
    goal = str(task.get("user_goal", "") or "")
    return "\n".join([
        "【任务目标】",
        goal,
        "",
        "【运行轨迹】",
        compressed_trace,
        "",
        "【最终答案】",
        str(final_answer or ""),
        "",
        "请依据 rubric 输出评分 JSON。",
    ])


def build_judge_messages(task: dict, compressed_trace: str,
                         final_answer: str) -> list[dict]:
    """构造 judge 的 OpenAI 风格消息列表。

    Args:
        task: 任务定义；只取 user_goal 字段（不序列化 task 本身，
            避免任务元数据或设计提示泄露进 judge 输入）。
        compressed_trace: compress.compress_trace 的产物（已匿名化）。
        final_answer: Agent 的最终答复文本。

    Returns:
        [{"role": "system", "content": ...}, {"role": "user", "content": ...}]。
    """
    return [
        {"role": "system", "content": _build_system()},
        {"role": "user", "content": _build_user(task, compressed_trace, final_answer)},
    ]
