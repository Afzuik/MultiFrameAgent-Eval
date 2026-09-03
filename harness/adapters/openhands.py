"""harness.adapters.openhands —— OpenHands 适配器（《项目方案.md》§6.5③）。

CLI 契约（§6.3，与 react / smolagents 完全一致）：
    python -m harness.adapters.openhands --spec run_spec.json --out trace.jsonl

职责：读取 run_spec（§6.3 只读规格）→ 执行一次工具调用评测运行 → 写归一化
trace.jsonl 与 {out.stem}.raw.log → 按退出码约定退出（0=正常 / 1=运行错误 /
2=预算超时）。

方案落定（§6.5③ 时间盒决策，详见本模块文档尾部"方案决策记录"）：
- 方案 A（OpenHands Python SDK headless）已尝试：openhands 1.16+/SDK 1.21 已装进
  .venv，但该版本是全新拆分架构（openhands.sdk.*，openhands.core/AppConfig/
  AgentController 已不存在），自定义工具 + 会话驱动需要按新事件架构重写；且本机无
  Docker（方案 B 前提缺失）、评测环境无模型 API key——时间盒内无法"稳定跑通并验证"
  真实 OpenHands 驱动。按 §6.5③ 时间盒触发方案 C。
- 方案 C（本实现，兜底但不破坏契约）：fake_llm 回放模式完整实现（dry-run 验收路径，
  优先级最高）；真实模式实现为"与 react 适配器 JSON 循环等价的循环"——同一套
  JSON 输出协议 / 消息回填口径 / 预算与轨迹语义（react 已在 W2 真实实验中验证，
  R1 组 40/40 可跑），保证实验矩阵 O1/O2 组全链路可跑。代码结构上把运行收敛到
  模块级 _run_fake / _run_real，W4 回归补方案 A 时只需替换 _run_real 内部实现，
  CLI 契约 / 预算 / trace 层无需改动。

与 react/smolagents 对齐的工程要点：
- spec["model_params"]["fake_llm"] == true 时不调任何框架/模型：按 spec["gt_plan"]
  顺序回放（message → tool_call → 真实 HTTP 调 mock 服务 → observation），回放完
  final_answer=spec["gt_answer"]，status=completed，退出码 0。
- litellm 调用收敛到模块级 _call_llm、工具调用收敛到模块级 _invoke_tool（httpx
  POST，10s 超时，同 react 口径），便于测试 monkeypatch。
- 模型构造保留 litellm 的 provider 前缀（spec["litellm_model"]，如
  "openai/deepseek-v4-flash"）：剥离前缀会导致 "LLM Provider NOT provided" 全线
  失败（2026-09 W2 真实实验教训）。
- 轨迹开头记录 message/system + message/user 两步；每次模型回合记 message/
  assistant 步（保证 trace.model_turns 与预算口径 max_steps 一致），每次工具调用记
  tool_call + observation 两步（ts 用 time.time()）。
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

# 模型输出解析失败后回填给模型的提示（与 react 同口径）
RETRY_PROMPT = (
    '你上一条输出无法解析为 JSON，请严格只输出 {"tool":...,"args":{...}} '
    '或 {"final_answer":...}'
)


# --------------------------------------------------------------------------
# 模块级可注入函数（测试直接 monkeypatch openhands.<name>）
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
    """换算一次响应的美元成本；无法换算时返回 0.0（真实成本 best-effort）。"""
    try:
        return float(litellm.completion_cost(completion_response=resp))
    except Exception:
        return 0.0


def _call_llm(spec: dict, messages: list[dict]) -> tuple[str, Any]:
    """调用 LiteLLM（openai 兼容端点），返回 (原始输出文本, 响应对象)。

    api_key 取 spec["api_key_env"] 指向的环境变量；未设置则传 None（交由端点/
    代理自行处理）。model 直接使用 spec["litellm_model"]（保留 provider 前缀，
    W2 真实实验教训：剥离前缀 → LLM Provider NOT provided）。
    # TODO(成本): 真实成本统计后续用 litellm 回调完善（§7.3），现按 best-effort
    # 由 litellm.completion_cost 换算，换算失败记 0.0。
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

    成功返回完整响应 JSON 文本；网络/协议异常返回 "工具调用异常: ..."（同 react
    口径，不抛异常——服务端返回 ok:false 或网络故障都回填给模型继续决策）。
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
    """把模型原始输出解析为 JSON 对象（§6.5① 输出协议，与 react 同实现）。

    兼容 ```json 围栏与前/后缀自由文本：剥离围栏后截取第一个 '{' 到最后一个 '}'
    之间的片段再做 json.loads；解析失败或结果不是 dict 时返回 None。
    """
    if not text:
        return None
    t = text.strip()
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
# 运行主流程（fake_llm 回放 / 真实循环两种模式）
# --------------------------------------------------------------------------
def _run_fake(trace: Trace, spec: dict, budget: RunBudget,
              raw_lines: list[str]) -> int:
    """fake_llm 模式：按 spec["gt_plan"] 顺序回放 GT 工具路径。

    每个计划项产生 message/assistant + tool_call + observation 三步（真实 HTTP
    调 mock 服务，同 react/smolagents 回放口径）；回放完以 gt_answer 收尾。
    raw_lines 保持为空（本模式不产出真实模型输出）。
    """
    t0 = time.time()
    plan = spec.get("gt_plan") or []
    _record(trace, "message", "system", content=build_system_prompt(spec))
    _record(trace, "message", "user",
            content=str(spec["task"].get("user_goal", "")))

    for idx, entry in enumerate(plan):
        # ---- 预算检查（每次"模型回合"前，与 react 一致）----
        if time.time() - t0 > budget.timeout_s:
            trace.status = STATUS_TIMEOUT
            return EXIT_BUDGET
        if idx >= budget.max_steps:
            trace.status = STATUS_BUDGET_EXCEEDED
            return EXIT_BUDGET
        tool_name = entry.get("tool")
        tool_args = entry.get("args") or {}
        if not isinstance(tool_name, str):
            _record(trace, "observation", "tool",
                    content=f"回放失败: gt_plan 第 {idx} 项缺少 tool 名称")
            continue
        raw = json.dumps({"tool": tool_name, "args": tool_args},
                         ensure_ascii=False)
        _record(trace, "message", "assistant", content=raw)
        _record(trace, "tool_call", "assistant",
                tool_name=tool_name, tool_args=tool_args)
        result_text = _invoke_tool(spec, tool_name, tool_args)
        _record(trace, "observation", "tool", content=result_text)

    trace.final_answer = str(spec.get("gt_answer", "") or "")
    trace.status = STATUS_COMPLETED
    return EXIT_OK


def _run_real(trace: Trace, spec: dict, budget: RunBudget,
              raw_lines: list[str]) -> int:
    """真实模式：与 react 适配器等价的 JSON 协议循环（方案 C，§6.5③）。

    模型按 spec["litellm_model"]（保留 provider 前缀）经 LiteLLM 调用；输出协议、
    工具回填口径、预算语义与 react.py 完全一致（react 的循环已在 W2 真实实验验证
    可跑）。TODO(W4 方案 A)：替换本函数内部为 OpenHands SDK headless 驱动——
    需按 openhands.sdk 新架构注册 spec["tool_specs"] 翻译的自定义工具并事件化采集；
    届时本模块其余部分（CLI/预算/退出码/trace 落盘）无需改动。
    """
    t0 = time.time()

    # 首轮消息：system + user（并记录为轨迹步骤，保证轨迹完整）
    sys_prompt = build_system_prompt(spec)
    goal = str(spec["task"].get("user_goal", ""))
    messages: list[dict] = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": goal},
    ]
    _record(trace, "message", "system", content=sys_prompt)
    _record(trace, "message", "user", content=goal)

    turns = 0  # 模型被调用次数（预算口径，§6.4 max_steps）
    while True:
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

        # ---- 取得本轮模型输出（真实调用 LiteLLM）----
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
            messages.append({"role": "user", "content": RETRY_PROMPT})
            _record(trace, "message", "user", content=RETRY_PROMPT)
            continue

        # ---- 工具分支：真实调 mock 服务并回填 observation ----
        _record(trace, "tool_call", "assistant", tool_name=tool_name,
                tool_args=tool_args)
        result_text = _invoke_tool(spec, tool_name, tool_args)
        _record(trace, "observation", "tool", content=result_text)
        # 回填策略（对 DeepSeek 推理模型兼容，W2 真实实验冒烟结论，同 react）：
        # assistant 原样输出 + user 消息携带工具结果；不合成 tool_calls/tool 消息
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": f"工具返回: {result_text}"})


def run_from_spec(spec: dict, out_path: Path) -> int:
    """按 run_spec 执行一次 OpenHands 适配器评测运行，写出 trace 与 raw 日志。

    任何分支（正常/预算/异常）都会写 trace.jsonl；raw 日志为纯文本附产物。
    返回进程退出码（0/1/2，§6.3）。
    """
    task = spec["task"]
    trace = Trace(run_id=spec["run_id"], task_id=task["task_id"],
                  framework="openhands", model=spec["model"])
    budget = RunBudget(**spec["budget"])
    raw_lines: list[str] = []
    t0 = time.time()
    exit_code: int = EXIT_OK
    try:
        if bool((spec.get("model_params") or {}).get("fake_llm")):
            exit_code = _run_fake(trace, spec, budget, raw_lines)
        else:
            exit_code = _run_real(trace, spec, budget, raw_lines)
    except Exception as exc:
        trace.status = STATUS_ERROR
        _record(trace, "observation", "tool", content=f"适配器异常: {exc!r}")
        exit_code = EXIT_ERROR
    finally:
        trace.wall_time_s = time.time() - t0
        trace.total_cost_usd = sum(s.cost_usd for s in trace.steps)
        protocol.write_trace(trace, out_path)
        # 原始模型输出另存为同目录 {trace文件名}.raw.log（每任务独立，避免互相覆盖）
        raw_path = out_path.with_name(out_path.stem + ".raw.log")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        if raw_lines:
            raw_path.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
    return exit_code


def main(argv: list[str] | None = None) -> int:
    """CLI 入口：python -m harness.adapters.openhands --spec ... --out ..."""
    parser = argparse.ArgumentParser(
        prog="python -m harness.adapters.openhands",
        description="AgentEval OpenHands 适配器（§6.5③）：读 run_spec，写 trace",
    )
    parser.add_argument("--spec", required=True, help="run_spec.json 路径（只读）")
    parser.add_argument("--out", required=True, help="trace.jsonl 输出路径")
    args = parser.parse_args(argv)
    spec_path = Path(args.spec)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    return run_from_spec(spec, Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
