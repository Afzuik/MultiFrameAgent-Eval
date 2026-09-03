"""harness.adapters.smolagents —— smolagents（CodeAgent）适配器（《项目方案.md》§6.5②）。

CLI 契约（§6.3，与 react 完全一致）：
    python -m harness.adapters.smolagents --spec run_spec.json --out trace.jsonl

职责：读取 run_spec（§6.3 只读规格）→ 用 smolagents CodeAgent 驱动一次
工具调用评测运行 → 写归一化 trace.jsonl 与 {out.stem}.raw.log →
按退出码约定退出（0=正常 / 1=运行错误 / 2=预算超时）。

设计要点（对齐 react.py 骨架）：
- LiteLLMModel 仅真实模式使用；模型构造收敛到模块级 `_make_model`，
  Agent 构造收敛到模块级 `_make_agent`，便于测试 monkeypatch 注入假模型。
- 工具按 spec["tool_specs"] 动态构造 smolagents Tool（宽松 input schema，
  参数校验交给 mock 服务端，返回 ok:false 亦回填给模型）；每个工具函数内部
  追加 tool_call / observation 两步轨迹，并真实 HTTP 调 mock 服务。
- 轨迹开头记录 message/system + message/user 两步（与 react 一致）；通过
  CodeAgent 的 step_callbacks 在每个 agent step 结束时追加 message/assistant 步，
  保证 trace.model_turns 与预算口径 max_steps 一致。
- spec["model_params"]["fake_llm"] == true 时不调用模型：按 spec["gt_plan"]
  顺序回放 GT 工具路径（message → tool_call → 真实 HTTP → observation），
  回放完以 spec["gt_answer"] 作为最终答复 —— W2 dry-run 验收路径。
"""
from __future__ import annotations

import argparse
import json
import keyword
import os
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import httpx
from smolagents.agents import AgentMaxStepsError, CodeAgent
from smolagents.memory import ActionStep
from smolagents.models import LiteLLMModel, Model
from smolagents.monitoring import LogLevel
from smolagents.tools import Tool

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


# --------------------------------------------------------------------------
# 模块级可注入函数（测试直接 monkeypatch smolagents_adapter.<name>）
# --------------------------------------------------------------------------
def build_system_prompt(spec: dict) -> str:
    """构造中文 instructions（CodeAgent instructions 槽 + 轨迹 system 步）。"""
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
        "用 Python 代码调用工具（不要用 print 代替）；工具内部会把参数转发给服务端校验，"
        "服务端返回 JSON——成功形如 {\"ok\": true, ...}，"
        "失败形如 {\"ok\": false, \"error\": \"...\"}。"
    )
    lines.append("任务完成时必须调用 final_answer(答复) 输出给用户的最终答复。")
    return "\n".join(lines)


def _py_arg_name(raw: str, taken: set[str]) -> str:
    """JSON 参数键 → Python 形参名（保留关键字如 from/class 的可用性）。"""
    name = raw if raw.isidentifier() and not keyword.iskeyword(raw) else raw + "_"
    while name in taken:  # 极少见冲突（如同时出现 class 与 class_）时加下划线兜底
        name += "_"
    return name


def _invoke_tool(spec: dict, tool_name: str, tool_args: dict) -> str:
    """POST 调 mock 工具服务，返回用于回填的文本。

    成功返回完整响应 JSON 文本；网络/协议异常返回 "工具调用异常: ..."（与 react 同口径）。
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


def _make_tool(spec: dict, tool_entry: dict, trace: Trace) -> Tool:
    """按一条 tool_spec 动态构造 smolagents Tool（宽松 input schema）。

    - input_types：string → {"type": "string", "nullable": True}；
      其余（integer/number/object/boolean...）→ {"type": "any", "nullable": True}，
      尽量宽松——参数合法性由 mock 服务端校验并返回 ok:false。
    - 工具函数体：先追加 Step(tool_call) → 真实 HTTP 调 mock 服务 →
      追加 Step(observation, 完整响应 JSON 文本) → 返回响应（dict/list/文本）
      或 "工具调用异常: ..." 错误文本给 Agent（不抛异常，与 react 同口径）。
    - Python 形参名对保留关键字（from/class 等）做映射，避免生成代码语法错误；
      转发 HTTP 前还原为 JSON 键（trace/服务端契约保持一致）。
    """
    name = str(tool_entry["name"])
    description = str(tool_entry.get("description") or name)
    parameters = tool_entry.get("parameters") or {}
    properties = parameters.get("properties") or {}

    inputs: dict[str, dict[str, Any]] = {}
    json_key: dict[str, str] = {}   # Python 形参名 → JSON 键
    taken: set[str] = set()
    for raw_key, meta in properties.items():
        meta = meta if isinstance(meta, dict) else {}
        py_name = _py_arg_name(str(raw_key), taken)
        taken.add(py_name)
        type_ = "string" if meta.get("type") == "string" else "any"
        inputs[py_name] = {
            "type": type_,
            "description": str(meta.get("description") or raw_key),
            "nullable": True,
        }
        json_key[py_name] = str(raw_key)

    class _RemoteTool(Tool):
        # 每个工具的参数集不同，forward 统一收 **kwargs，
        # 跳过 smolagents 对形参名与 inputs 键一致的签名校验。
        skip_forward_signature_validation = True
        output_type = "any"

        def forward(self, **kwargs: Any) -> Any:
            # 还原 JSON 键、丢弃显式 None（可选项缺省），记录并调用 mock 服务
            args = {json_key[k]: v for k, v in kwargs.items()
                    if v is not None and k in json_key}
            _record(trace, "tool_call", "assistant",
                    tool_name=name, tool_args=args)
            result_text = _invoke_tool(spec, name, args)
            _record(trace, "observation", "tool", content=result_text)
            try:
                return json.loads(result_text)
            except Exception:
                return result_text

    _RemoteTool.name = name
    _RemoteTool.description = description
    _RemoteTool.inputs = inputs
    return _RemoteTool()


def _make_model(spec: dict) -> Model:
    """按 spec 构造 smolagents LiteLLMModel（模块级可注入，便于测试替换假模型）。

    模型 id 去掉 "openai/" 前缀（CodeAgent 生成的代码无需 provider 前缀）；
    api_key 取 spec["api_key_env"] 指向的环境变量，未设置则 None
    （交由端点/代理自行处理）；temperature / max_completion_tokens 取自 model_params
    （max_tokens 兼容旧配置）。
    """
    params = spec.get("model_params") or {}
    model_id = spec["litellm_model"].removeprefix("openai/")
    env_name = spec.get("api_key_env") or ""
    api_key = os.environ.get(env_name) if env_name else None
    kwargs: dict[str, Any] = {
        "model_id": model_id,
        "api_base": spec.get("api_base") or None,
        "api_key": api_key or None,
        "temperature": params.get("temperature", 0.2),
    }
    max_tokens = params.get("max_completion_tokens") or params.get("max_tokens")
    if max_tokens is not None:
        kwargs["max_completion_tokens"] = int(max_tokens)
    return LiteLLMModel(**kwargs)


class _MaxStepsCodeAgent(CodeAgent):
    """max_steps 耗尽且未产出 final_answer 时直接抛 AgentMaxStepsError。

    smolagents 默认在耗尽后会再调一次模型"补最终答案"（多一次计费），
    与 §6.4 预算语义不符，故覆写为抛错，由适配器置 status=budget_exceeded。
    """

    def provide_final_answer(self, task: str) -> Any:
        raise AgentMaxStepsError("Agent reached max steps.", self.logger)


def _make_agent(model: Model, tools: list[Tool], instructions: str,
                budget: RunBudget) -> CodeAgent:
    """构造 CodeAgent：注入模型/工具/中文 instructions，并施加 max_steps 预算。"""
    return _MaxStepsCodeAgent(
        tools=tools,
        model=model,
        instructions=instructions,
        additional_authorized_imports=["json"],
        max_steps=max(int(budget.max_steps), 1),
        verbosity_level=LogLevel.OFF,
    )


def _step_tokens(step: Any) -> tuple[int, int]:
    """从 ActionStep.token_usage 取本步 tokens（best-effort，取不到记 0）。"""
    usage = getattr(step, "token_usage", None)
    if usage is None:
        return 0, 0
    return (
        int(getattr(usage, "input_tokens", 0) or 0),
        int(getattr(usage, "output_tokens", 0) or 0),
    )


def _step_output_text(step: Any) -> str:
    """取 ActionStep 的模型输出文本（非字符串形态序列化，取不到为空串）。"""
    output = getattr(step, "model_output", None)
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    return json.dumps(output, ensure_ascii=False)


# --------------------------------------------------------------------------
# 运行主流程（fake_llm 回放 / 真实 CodeAgent 两种模式）
# --------------------------------------------------------------------------
def _run_fake(trace: Trace, spec: dict, budget: RunBudget,
              raw_lines: list[str]) -> int:
    """fake_llm 模式：按 gt_plan 回放 GT 工具路径（真实 HTTP 调 mock 服务）。"""
    t0 = time.time()
    plan = spec.get("gt_plan") or []
    _record(trace, "message", "system", content=build_system_prompt(spec))
    _record(trace, "message", "user",
            content=str(spec["task"].get("user_goal", "")))

    for idx, entry in enumerate(plan):
        # ---- 预算检查（与 react 一致：每次"模型回合"前）----
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
    """真实模式：CodeAgent + LiteLLMModel 驱动；超时用后台线程 + join 实现。"""
    t0 = time.time()
    _record(trace, "message", "system", content=build_system_prompt(spec))
    goal = str(spec["task"].get("user_goal", ""))
    _record(trace, "message", "user", content=goal)

    def on_step(memory_step: Any, **kwargs: Any) -> None:
        """每个 agent step 结束时追加一条 assistant 消息步（预算口径 max_steps）。"""
        tokens_in, tokens_out = _step_tokens(memory_step)
        output_text = _step_output_text(memory_step)
        if output_text:
            raw_lines.append(output_text)
        # TODO: 真实成本统计留待后续用 litellm 回调完善（§7.3），暂记 0.0
        _record(trace, "message", "assistant", content=output_text,
                tokens_in=tokens_in, tokens_out=tokens_out, cost_usd=0.0)

    model = _make_model(spec)
    tools = [_make_tool(spec, entry, trace)
             for entry in spec.get("tool_specs") or []]
    instructions = build_system_prompt(spec)
    agent = _make_agent(model, tools, instructions, budget)
    # CodeAgent 的 step_callbacks 在构造时按 ActionStep 注册；
    # 此处对实例补注册本次运行的轨迹回调（注册器支持运行时追加）
    agent.step_callbacks.register(ActionStep, on_step)

    box: dict[str, Any] = {}

    def worker() -> None:
        try:
            box["output"] = agent.run(goal)
        except AgentMaxStepsError:
            box["budget"] = True
        except Exception as exc:  # 交给 run_from_spec 统一兜底（error + 退出码 1）
            box["exc"] = exc

    thread = threading.Thread(target=worker, daemon=True,
                              name="smolagents-agent-run")
    thread.start()
    remaining = max(budget.timeout_s - (time.time() - t0), 0.0)
    thread.join(timeout=remaining)

    if thread.is_alive():
        # 超时：请求中断当前 agent 循环（线程无法强杀，进程退出后即回收）
        trace.status = STATUS_TIMEOUT
        with suppress(Exception):  # interrupt 失败不影响超时结论
            agent.interrupt()
        return EXIT_BUDGET
    if "exc" in box:
        raise box["exc"] from None
    if box.get("budget"):
        trace.status = STATUS_BUDGET_EXCEEDED
        return EXIT_BUDGET

    output = box.get("output")
    if not isinstance(output, str):
        output = json.dumps(output, ensure_ascii=False)
    trace.final_answer = output or ""
    trace.status = STATUS_COMPLETED
    return EXIT_OK


def run_from_spec(spec: dict, out_path: Path) -> int:
    """按 run_spec 执行一次 smolagents 评测运行，写出 trace 与 raw 日志。

    任何分支（正常/预算/异常）都会写 trace.jsonl；raw 日志为纯文本附产物。
    返回进程退出码（0/1/2，§6.3）。
    """
    task = spec["task"]
    trace = Trace(run_id=spec["run_id"], task_id=task["task_id"],
                  framework="smolagents", model=spec["model"])
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
    """CLI 入口：python -m harness.adapters.smolagents --spec ... --out ..."""
    parser = argparse.ArgumentParser(
        prog="python -m harness.adapters.smolagents",
        description="AgentEval smolagents(CodeAgent) 适配器（§6.5②）：读 run_spec，写 trace",
    )
    parser.add_argument("--spec", required=True, help="run_spec.json 路径（只读）")
    parser.add_argument("--out", required=True, help="trace.jsonl 输出路径")
    args = parser.parse_args(argv)
    spec_path = Path(args.spec)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    return run_from_spec(spec, Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
