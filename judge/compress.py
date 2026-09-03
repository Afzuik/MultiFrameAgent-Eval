"""judge.compress —— 轨迹压缩为 judge 输入文本（§8.1 输入构造）。

压缩规则：
- 步骤按序编号：[步骤 N] 逐条列出（消息 / 工具调用 / 工具返回）；
- observation 内容超过 max_obs_chars 时截断为前 N 字符，并追加
  "…(截断 N 字)" 摘要标注；其余步骤内容原样保留；
- 匿名化（§8.3①）：轨迹元数据（framework / model 等）不进文本，
  且从各步内容中剔除 trace.framework / trace.model 及已知框架名的
  字面出现，保证 judge 输入看不到"是哪家框架/哪个模型跑的"。

输出为纯文本，供 judge/prompts.build_judge_messages 拼入 user 消息。
"""
from __future__ import annotations

import json

from harness.protocol import Trace

# 已知框架标识（值均为小写）。除 trace.framework 外也一并剔除，
# 防轨迹文本里顺带提到其它框架名造成泄漏。
_KNOWN_FRAMEWORKS = ("react", "smolagents", "openhands")

__all__ = ["compress_trace"]


def _scrub(text: str, names: tuple[str, ...]) -> str:
    """剔除文本中出现的全部标识名（匿名化，§8.3①）。"""
    for name in names:
        if name:
            text = text.replace(name, "")
    return text


def _render_content(
    text: str | None, max_obs_chars: int, scrub_names: tuple[str, ...]
) -> str:
    """取步骤内容正文：observation 超长截断 + 标识名剔除。"""
    content = str(text or "")
    if len(content) > max_obs_chars:
        content = content[:max_obs_chars] + f"…(截断 {max_obs_chars} 字)"
    return _scrub(content, scrub_names)


def _tool_call_block(idx: int, step, scrub_names: tuple[str, ...]) -> list[str]:
    """工具调用步骤的文本块（工具名 + 参数，参数为空则省略）。"""
    head = f"[步骤 {idx}] 工具调用：{_scrub(str(step.tool_name or ''), scrub_names)}"
    lines = [head]
    if step.tool_args:
        args_text = _scrub(
            json.dumps(step.tool_args, ensure_ascii=False), scrub_names
        )
        lines.append(f"参数：{args_text}")
    return lines


def compress_trace(trace: Trace, max_obs_chars: int = 500) -> str:
    """把归一化轨迹压缩为纯文本（judge 输入）。

    Args:
        trace: harness.protocol.Trace。仅读取 steps；framework/model
            元数据只用于匿名化剔除，不写入输出。
        max_obs_chars: observation 内容保留的最大字符数（默认 500）。

    Returns:
        纯文本：步骤按 [步骤 N] 编号逐条呈现。
    """
    if max_obs_chars < 1:
        raise ValueError(f"max_obs_chars 必须为正整数，收到 {max_obs_chars!r}")
    framework = str(getattr(trace, "framework", "") or "")
    model = str(getattr(trace, "model", "") or "")
    scrub_names: tuple[str, ...] = tuple(
        dict.fromkeys((framework, model) + _KNOWN_FRAMEWORKS)
    )

    blocks: list[str] = []
    for idx, step in enumerate(trace.steps, start=1):
        if step.type == "tool_call":
            blocks.extend(_tool_call_block(idx, step, scrub_names))
            continue
        if step.type == "observation":
            body = _render_content(step.content, max_obs_chars, scrub_names)
            blocks.append(f"[步骤 {idx}] 工具返回（observation）\n{body}")
            continue
        # 其余类型（message / final_answer 等）内容原样保留，仅剔除标识名
        if step.type == "message":
            role = _scrub(str(getattr(step, "role", "") or ""), scrub_names)
            head = f"[步骤 {idx}] 消息（{role}）"
        else:
            head = f"[步骤 {idx}] {step.type}"
        body = _scrub(str(step.content or ""), scrub_names)
        blocks.append(f"{head}\n{body}" if body else head)
    return "\n".join(blocks)
