"""harness.adapters.react —— 自研 ReAct 适配器（《项目方案.md》§6.5①，零框架依赖）。

CLI 契约（§6.3）：
    python -m harness.adapters.react --spec run_spec.json --out trace.jsonl

职责：读取 run_spec（§6.3 只读规格）→ 纯 JSON 协议 ReAct 循环
（模型输出 / JSON 解析 / 调 mock 工具 / 回填 observation / 直至 final_answer
或预算耗尽）→ 写归一化 trace.jsonl 与 trace.raw.log → 按退出码约定退出
（0=正常 / 1=运行错误 / 2=预算超时）。

设计要点：
- litellm 仅在模块顶部 import；实际调用收敛到模块级 `_call_llm`，
  便于测试 monkeypatch（react._call_llm = 假函数）。
- 工具调用收敛到模块级 `_invoke_tool`（httpx POST，10s 超时）。
- spec["model_params"]["fake_llm"] == true 时不调用模型：按 spec["gt_plan"]
  顺序回放 GT 工具路径，用于无 API key 的全链路验收（--dry-run）。
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
import litellm

from harness import protocol
from harness.protocol import (
    EXIT_BUDGET,
    EXIT_ERROR,
    EXIT_OK,
    STATUS_BUDGET_EXCEEDED,
    STATUS_COMPLETED,
    STATUS_ERROR,
    STATUS_TIMEOUT,
    RunBudget,
    Step,
    Trace,
)

# 单次工具调用 HTTP 超时（秒）
TOOL_TIMEOUT_S = 10.0

# 解析失败后回填给模型的提示（与《项目方案.md》§6.5① 一致）
RETRY_PROMPT = (
    '你上一条输出无法解析为 JSON，请严格只输出 {"tool":...,"args":{...}} '
    '或 {"final_answer":...}'
)


# --------------------------------------------------------------------------
# 模块级可注入函数（测试直接 monkeypatch react.<name>）
# --------------------------------------------------------------------------
def _usage(resp: Any, attr: str) -> int:
    """从响应对象安全读取 usage 字段（兼容对象与 dict 两种形态）。"""
    usage = getattr(resp, "usage", None)
    if usage is None:
        return 0
    if isinstance(usage, dict):
        return int(usage.get(attr, 0) or 0)
    return int(getattr(usage, attr, 0) or 0)


def _completion_cost(resp: Any) -> float:
    """换算一次响应的美元成本；无法换算时返回 0.0。"""
    try:
        return float(litellm.completion_cost(completion_response=resp))
    except Exception:
        return 0.0


def _call_llm(spec: dict, messages: list[dict]) -> tuple[str, Any]:
    """调用 LiteLLM（openai 兼容端点），返回 (原始输出文本, 响应对象)。

    api_key 取 spec["api_key_env"] 指向的环境变量；未设置则传 None
    （交由端点/代理自行处理）。
    """
    params = spec.get("model_params") or {}
    env_name = spec.get("api_key_env") or ""
    api_key = os.environ.get(env_name) if env_name else None
    resp = litellm.completion(
        model=spec["litellm_model"],
        api_base=spec.get("api_base"),
        api_key=api_key or None,
        messages=messages,
        temperature=params.get("temperature", 0.2),
        max_tokens=params.get("max_tokens", 4096),
        timeout=float(params.get("api_timeout_s", 120)),
    )
    content = resp.choices[0].message.content
    if isinstance(content, str):
        return content, resp
    # 内容非字符串（罕见）时序列化为 JSON 文本，保证下游统一按字符串处理
    return json.dumps(content, ensure_ascii=False), resp


def _invoke_tool(spec: dict, tool_name: str, tool_args: dict) -> str:
    """POST 调 mock 工具服务，返回用于回填的文本。

    成功返回完整响应 JSON 文本；网络/协议异常返回 "工具调用异常: ..."。
    """
    base_url = spec["tool_server"]["base_url"]
    instance_id = spec["tool_server"]["instance_id"]
    url = f"{base_url}/instances/{instance_id}/tools/{tool_name}"
    try:
        resp = httpx.post(url, json=tool_args or {}, timeout=TOOL_TIMEOUT_S)
        resp.raise_for_status()
        return json.dumps(resp.json(), ensure_ascii=False)
    except Exception as exc:
        return f"工具调用异常: {exc}"


def parse_model_output(text: str) -> dict | None:
    """把模型原始输出解析为 JSON 对象（§6.5① 输出协议）。

    兼容 ```json 围栏与前/后缀自由文本：剥离围栏后截取
    第一个 '{' 到最后一个 '}' 之间的片段再做 json.loads；
    解析失败或结果不是 dict 时返回 None。
    """
    if not text:
        return None
    t = text.strip()
    # 剥掉首行/末行的代码围栏（```json / ```）
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].strip().startswith("```"):
            t = "\n".join(lines[1:]).strip()
        if t.endswith("```"):
            t = t[:-3].rstrip()
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end <= start:
        return None
    fragment = t[start:end + 1]
    try:
        obj = json.loads(fragment)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


# --------------------------------------------------------------------------
# 轨迹记录
# --------------------------------------------------------------------------
def _record(
    trace: Trace,
    type_: str,
    role: str,
    content: str = "",
    tool_name: str | None = None,
    tool_args: dict | None = None,
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost_usd: float = 0.0,
) -> None:
    """向轨迹追加一步（ts 取当前时间）。"""
    trace.steps.append(Step(
        type=type_, role=role, content=content, tool_name=tool_name,
        tool_args=tool_args, ts=time.time(),
        tokens_in=tokens_in, tokens_out=tokens_out, cost_usd=cost_usd,
    ))


def build_system_prompt(spec: dict) -> str:
    """构造中文 system prompt：工具列表（JSON 呈现）+ user_id + 输出协议。"""
    task = spec.get("task") or {}
    tools_text = json.dumps(spec.get("tool_specs", []), ensure_ascii=False, indent=2)
    lines = [
        "你是工具调用 Agent，任务是通过调用可用工具完成用户的请求。",
        "可用工具列表（JSON，含 name/description/parameters）：",
        tools_text,
    ]
    user_id = (task.get("initial_state") or {}).get("user_id")
    if user_id:
        lines.append(f"当前用户 user_id = {user_id}（多数工具需要携带该参数）。")
    lines.append(
        '输出协议：每次只输出一个 JSON 对象——要么 {"tool": "<工具名>", "args": {...}} '
        '表示调用某个工具；要么 {"final_answer": "<给用户的最终答复>"} 表示任务完成。'
    )
    lines.append("除此之外不要输出任何其他内容（不要代码围栏、不要多余解释）。")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# ReAct 主循环
# --------------------------------------------------------------------------
def _run_loop(trace: Trace, spec: dict, budget: RunBudget,
              raw_lines: list[str]) -> int:
    """执行 ReAct 循环，返回进程退出码（退出前状态写入 trace.status）。"""
    t0 = time.time()
    params = spec.get("model_params") or {}
    fake = bool(params.get("fake_llm"))
    plan = spec.get("gt_plan") or []

    # 首轮消息：system + user（并记录为轨迹步骤，保证轨迹完整）
    sys_prompt = build_system_prompt(spec)
    goal = str(spec["task"].get("user_goal", ""))
    messages: list[dict] = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": goal},
    ]
    _record(trace, "message", "system", content=sys_prompt)
    _record(trace, "message", "user", content=goal)

    turns = 0  # 模型被调用次数（fake 模式 = 已回放的计划步数）
    while True:
        # fake_llm 回放完毕 → 正常结束
        if fake and turns >= len(plan):
            trace.final_answer = str(spec.get("gt_answer", "") or "")
            trace.status = STATUS_COMPLETED
            return EXIT_OK
        # ---- 预算检查（每次调用模型前）----
        if time.time() - t0 > budget.timeout_s:
            trace.status = STATUS_TIMEOUT
            return EXIT_BUDGET
        if trace.total_cost_usd > budget.max_cost_usd:
            trace.status = STATUS_BUDGET_EXCEEDED
            return EXIT_BUDGET
        if turns >= budget.max_steps:
            trace.status = STATUS_BUDGET_EXCEEDED
            return EXIT_BUDGET

        # ---- 取得本轮模型输出（真实调用 或 fake 回放）----
        if fake:
            entry = plan[turns]
            tool_name, tool_args = entry.get("tool"), entry.get("args") or {}
            raw = json.dumps({"tool": tool_name, "args": tool_args}, ensure_ascii=False)
            parsed: dict | None = {"tool": tool_name, "args": tool_args}
            tokens_in = tokens_out = 0
            cost_usd = 0.0
            turns += 1
        else:
            raw, resp = _call_llm(spec, messages)
            tokens_in = _usage(resp, "prompt_tokens")
            tokens_out = _usage(resp, "completion_tokens")
            cost_usd = _completion_cost(resp)
            turns += 1
            raw_lines.append(raw)
            trace.total_cost_usd += cost_usd
            parsed = parse_model_output(raw)

        # 每次模型输出都记一条 assistant 消息步（含解析失败的输出，
        # 保证 trace.model_turns 与预算口径 max_steps 一致）
        _record(trace, "message", "assistant", content=raw,
                tokens_in=tokens_in, tokens_out=tokens_out, cost_usd=cost_usd)

        # 调用后立即复核成本上限（本轮刚产生成本）
        if trace.total_cost_usd > budget.max_cost_usd:
            trace.status = STATUS_BUDGET_EXCEEDED
            return EXIT_BUDGET

        # ---- 解析结果分派 ----
        if parsed is None:
            _record(trace, "observation", "tool",
                    content="解析失败: 模型输出不是合法的 JSON 对象")
            messages.append({"role": "user", "content": RETRY_PROMPT})
            _record(trace, "message", "user", content=RETRY_PROMPT)
            continue
        if "final_answer" in parsed:
            trace.final_answer = str(parsed["final_answer"])
            trace.status = STATUS_COMPLETED
            return EXIT_OK
        tool_name = parsed.get("tool")
        tool_args = parsed.get("args")
        if not isinstance(tool_name, str) or not isinstance(tool_args, dict):
            _record(trace, "observation", "tool",
                    content="解析失败: 输出缺少 tool/final_answer 键或字段类型非法")
            if not fake:
                messages.append({"role": "user", "content": RETRY_PROMPT})
                _record(trace, "message", "user", content=RETRY_PROMPT)
            continue

        # ---- 工具分支：真实调 mock 服务并回填 observation ----
        _record(trace, "tool_call", "assistant", tool_name=tool_name,
                tool_args=tool_args)
        result_text = _invoke_tool(spec, tool_name, tool_args)
        _record(trace, "observation", "tool", content=result_text)
        if not fake:
            # 回填策略（对 DeepSeek 推理模型兼容）：assistant 原样输出 +
            # user 消息携带工具结果。不合成 tool_calls/tool 消息——
            # api.deepseek.com 思考模式要求 tool_calls 消息回传 reasoning_content，
            # 合成消息无法满足该要求（2026-09 真实实验冒烟结论）。
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": f"工具返回: {result_text}",
            })


def run_from_spec(spec: dict, out_path: Path) -> int:
    """按 run_spec 执行一次 ReAct 评测运行，写出 trace 与 raw 日志。

    任何分支（正常/预算/异常）都会写 trace.jsonl；raw 日志为纯文本附产物。
    返回进程退出码（0/1/2，§6.3）。
    """
    task = spec["task"]
    trace = Trace(run_id=spec["run_id"], task_id=task["task_id"],
                  framework="react", model=spec["model"])
    budget = RunBudget(**spec["budget"])
    raw_lines: list[str] = []
    t0 = time.time()
    exit_code: int = EXIT_OK
    try:
        exit_code = _run_loop(trace, spec, budget, raw_lines)
    except Exception as exc:
        trace.status = STATUS_ERROR
        _record(trace, "observation", "tool", content=f"适配器异常: {exc!r}")
        exit_code = EXIT_ERROR
    finally:
        trace.wall_time_s = time.time() - t0
        trace.total_cost_usd = sum(s.cost_usd for s in trace.steps)
        protocol.write_trace(trace, out_path)
        # 原始模型输出另存为同目录 {trace文件名}.raw.log（每任务独立，避免多任务互相覆盖）
        raw_path = out_path.with_name(out_path.stem + ".raw.log")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        if raw_lines:
            raw_path.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
    return exit_code


def main(argv: list[str] | None = None) -> int:
    """CLI 入口：python -m harness.adapters.react --spec ... --out ..."""
    parser = argparse.ArgumentParser(
        prog="python -m harness.adapters.react",
        description="AgentEval 自研 ReAct 适配器（§6.5①）：读 run_spec，写 trace",
    )
    parser.add_argument("--spec", required=True, help="run_spec.json 路径（只读）")
    parser.add_argument("--out", required=True, help="trace.jsonl 输出路径")
    args = parser.parse_args(argv)
    spec_path = Path(args.spec)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    return run_from_spec(spec, Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
