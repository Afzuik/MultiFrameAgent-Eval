"""harness.adapters.openhands —— OpenHands 适配器（《项目方案.md》§6.5③）。

CLI 契约（§6.3，与 react / smolagents 完全一致）：
    python -m harness.adapters.openhands --spec run_spec.json --out trace.jsonl

职责：读取 run_spec（§6.3 只读规格）→ 执行一次工具调用评测运行 → 写归一化
trace.jsonl 与 {out.stem}.raw.log → 按退出码约定退出（0=正常 / 1=运行错误 /
2=预算超时）。

方案落定（§6.5③ 演进，详见各版本"方案决策记录"）：
- W3 时间盒（历史）：openhands 1.16+/SDK 1.21 为全新拆分架构（openhands.sdk.*），
  当时本机无 Docker、评测环境无模型 API key，无法稳定跑通真实 SDK 驱动，时间盒触发
  方案 C（fake_llm 回放完整实现 + 真实模式暂作 react 等价 JSON 循环）。
- W4 方案 A 回归（本实现）：真实模式已替换为 OpenHands SDK 1.21 headless 驱动——
  模块级可注入的 _make_llm / _make_agent / _make_tools / _make_conversation /
  _make_event_handler / _run_real。OpenHands 走 function calling（模型直接产出工具
  调用 + finish 收尾），不再是 JSON 文本协议循环；CLI 契约 / 预算 / trace 层未动。
  fake_llm 回放路径（_run_fake，dry-run 验收）原样保留、优先级最高。

工程要点（fake 回放 / SDK 真实两条路径的公共约定）：
- spec["model_params"]["fake_llm"] == true 时不调任何框架/模型：按 spec["gt_plan"]
  顺序回放（message → tool_call → 真实 HTTP 调 mock 服务 → observation），回放完
  final_answer=spec["gt_answer"]，status=completed，退出码 0。
- 工具调用收敛到模块级 _invoke_tool（httpx POST，10s 超时，同 react 口径），
  便于测试 monkeypatch；SDK 工具的 executor 内部复用它完成真实 HTTP。
- 模型构造保留 litellm 的 provider 前缀（spec["litellm_model"]，如
  "openai/deepseek-v4-flash"）：剥离前缀会导致 "LLM Provider NOT provided" 全线
  失败（2026-09 W2 真实实验教训）。
- 轨迹开头记录 message/system + message/user 两步；真实模式把 SDK 事件流映射为
  message/assistant（每模型回合一步，保证 trace.model_turns 与预算口径一致）+
  tool_call + observation 两步（ts 用 time.time()）。
"""
from __future__ import annotations

import argparse
import json
import keyword
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
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


def _call_llm_direct(spec: dict, messages: list[dict]) -> tuple[str, Any]:
    """直接调 LiteLLM（openai 兼容端点），返回 (原始输出文本, 响应对象)。

    api_key 取 spec["api_key_env"] 指向的环境变量；未设置则传 None（交由端点/
    代理自行处理）。model 直接使用 spec["litellm_model"]（保留 provider 前缀，
    W2 真实实验教训：剥离前缀 → LLM Provider NOT provided）。
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


def _call_llm(spec: dict, messages: list[dict]) -> tuple[str, Any]:
    """带墙钟兜底的模型调用（与 react 同款修复，见 react._call_llm docstring）。

    2026-09 真实实验：某端点"接受连接但永不返回数据"，litellm timeout 不生效，
    单次调用挂死整个任务（O2 tr_006 挂起 31 分钟）。daemon 线程 + join 兜底，
    超时抛 TimeoutError，由调用方转为 status=timeout。
    """
    timeout_s = float((spec.get("model_params") or {}).get("api_timeout_s", 120))
    box: dict = {}

    def worker() -> None:
        try:
            box["result"] = _call_llm_direct(spec, messages)
        except Exception as exc:
            box["exc"] = exc

    thread = threading.Thread(target=worker, daemon=True, name="openhands-llm-call")
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        raise TimeoutError(f"模型调用超过 {timeout_s:.0f}s 无响应")
    if "exc" in box:
        raise box["exc"]
    return box["result"]


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
# 方案 A：OpenHands SDK headless 驱动（真实模式专用，惰性加载 SDK）
# --------------------------------------------------------------------------
# 说明：openhands.sdk 1.21 是全新拆分架构，本段代码只在真实模式被调用时才导入
# SDK（fake_llm 回放路径与无 SDK 环境不受影响）。SDK 的判别式联合序列化不
# 支持函数内用 class 语句动态建类（qualname 带 <locals> 会在事件落盘时触发
# RecursionError / "Local classes not supported"），因此工具/观测/执行器等
# 运行期具体类一律用 type() 动态创建、并缓存关键依赖到 _OH_RT。
_OH_RT: dict[str, Any] | None = None


def _oh_rt() -> dict[str, Any]:
    """惰性导入 openhands.sdk 并缓存真实模式依赖（抑制启动横幅）。"""
    global _OH_RT
    if _OH_RT is None:
        os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")
        from openhands import sdk
        from openhands.sdk.tool import (  # type: ignore[import-not-found]
            Action,
            Observation,
            ToolAnnotations,
            ToolDefinition,
            ToolExecutor,
            register_tool,
        )
        _OH_RT = {
            "sdk": sdk,
            "Agent": sdk.Agent,
            "LLM": sdk.LLM,
            "Tool": sdk.Tool,
            "LocalConversation": sdk.LocalConversation,
            "Action": Action,
            "Observation": Observation,
            "ToolExecutor": ToolExecutor,
            "ToolDefinition": ToolDefinition,
            "ToolAnnotations": ToolAnnotations,
            "register_tool": register_tool,
        }
    return _OH_RT


def _tag_token(raw: str, limit: int = 24) -> str:
    """把任意 run_id 等字符串压成只含 [A-Za-z0-9_] 的类名安全片段。"""
    token = re.sub(r"[^A-Za-z0-9_]", "_", raw)
    return token[:limit].strip("_") or "run"


def build_oh_system_prompt(spec: dict) -> str:
    """构造真实模式的 system prompt（OpenHands 函数调用语义，中文）。

    与 build_system_prompt（fake/react JSON 文本协议）不同：OpenHands 把工具以
    function calling schema 注入模型，模型直接产出工具调用与 finish 收尾，因此
    这里只给任务定位 + 工具清单 + user_id + finish 用法，绝不要求输出
    {"tool": ...} 之类的 JSON 文本协议。
    """
    task = spec.get("task") or {}
    lines = [
        "你是工具调用 Agent，任务是通过调用可用工具完成用户的请求。",
        "可用工具列表（JSON，含 name/description/parameters）：",
        json.dumps(spec.get("tool_specs", []), ensure_ascii=False, indent=2),
    ]
    user_id = (task.get("initial_state") or {}).get("user_id")
    if user_id:
        lines.append(f"当前用户 user_id = {user_id}（多数工具需要携带该参数）。")
    lines.append("按需调用工具，并根据工具返回继续决策，直到拿到完成任务所需的信息。")
    lines.append(
        "任务完成后必须调用 finish 工具，并把给用户的最终答复放在 "
        "finish 的 message 参数中；不要输出多余的中间过程说明。"
    )
    return "\n".join(lines)


def _json_type_to_py(meta: dict[str, Any] | None) -> Any:
    """JSON schema 类型 → Python 类型（宽松映射，无法确定一律 Any）。"""
    if not isinstance(meta, dict):
        return Any
    t = meta.get("type")
    if isinstance(t, (list, tuple)):
        non_null = [x for x in t if x != "null"]
        t = non_null[0] if len(non_null) == 1 else None
    return {"string": str, "integer": int, "number": float,
            "boolean": bool, "array": list, "object": dict}.get(t, Any)


def _safe_field_name(key: str) -> tuple[str, str | None]:
    """JSON 参数键 → pydantic 字段名（保留 class/from 等关键字可用性）。"""
    fname = key if key.isidentifier() and not keyword.iskeyword(key) else key + "_"
    alias: str | None = key if fname != key else None
    return fname, alias


def _build_action_type(rt: dict[str, Any], tag: str, tool_name: str,
                       parameters: dict[str, Any] | None) -> Any:
    """按工具 JSON schema 生成其 Action 子类（type() 建类 + pydantic create_model）。

    - 必填参数给无默认值 Field（让 function calling schema 呈现 required），
      缺参时 SDK 抛校验错误回填给模型自纠（同 react 服务端 ok:false 自纠口径）；
    - 非法标识符键（class/from...）用下划线字段 + alias 还原 JSON 键；
    - populate_by_name=True：SDK 事件落盘用字段名、回读按别名校验，两者兼容
      （W4 冒烟实证：无该配置时 ActionEvent 持久化回读抛 "action.from Field
      required"）。
    """
    from pydantic import ConfigDict, Field, create_model
    parameters = parameters if isinstance(parameters, dict) else {}
    props = parameters.get("properties") or {}
    required = set(parameters.get("required") or [])
    fields: dict[str, Any] = {}
    for raw_key, meta in props.items():
        meta = meta if isinstance(meta, dict) else {}
        key = str(raw_key)
        fname, alias = _safe_field_name(key)
        fkwargs: dict[str, Any] = {}
        if alias:
            fkwargs["alias"] = alias
        desc = meta.get("description")
        if desc:
            fkwargs["description"] = str(desc)
        ftype = _json_type_to_py(meta)
        if key in required:
            fields[fname] = (ftype, Field(..., **fkwargs))
        else:
            fields[fname] = (ftype | None, Field(default=None, **fkwargs))
    cls_name = f"AeOhAction_{_tag_token(tool_name)}_{tag}"
    return create_model(
        cls_name, __base__=rt["Action"],
        __config__=ConfigDict(extra="ignore", frozen=True, populate_by_name=True),
        **fields,
    )


def _new_executor(rt: dict[str, Any], obs_cls: Any, spec: dict,
                  tool_name: str) -> Any:
    """构造绑定 spec 的远程工具执行器（内部复用 _invoke_tool 真实 HTTP 调 mock）。"""

    def _call(self: Any, action: Any, conversation: Any = None) -> Any:
        args: dict[str, Any] = {}
        dump = getattr(action, "model_dump", None)
        if callable(dump):
            try:
                args = {
                    k: v for k, v in dump(by_alias=True, exclude_none=True).items()
                    if k not in ("summary", "security_risk")
                }
            except Exception:
                args = {}
        # 失败/异常由 _invoke_tool 转成文本回填（ok:false / 网络故障同 react 口径，
        # 不向 SDK 抛异常——SDK 对 ValueError 会转 AgentErrorEvent 打断循环语义）
        result_text = _invoke_tool(spec, tool_name, args)
        return obs_cls.from_text(text=result_text)

    exec_cls = type(f"AeOhExecutor_{tool_name}", (rt["ToolExecutor"],),
                    {"__call__": _call})
    return exec_cls()


def _register_one_tool(rt: dict[str, Any], spec: dict, tag: str, obs_cls: Any,
                       entry: dict[str, Any]) -> Any:
    """注册单个工具（ToolDefinition 动态子类）并返回 Agent.tools 规格。

    工具名/描述/参数 schema 经参数传入，嵌套 _create 只闭包本函数参数，
    规避"闭包直取循环变量"的经典串味问题（B023）。
    """
    tool_name = str(entry["name"])
    description = str(entry.get("description") or tool_name)
    action_cls = _build_action_type(rt, tag, tool_name, entry.get("parameters"))
    executor = _new_executor(rt, obs_cls, spec, tool_name)
    reg_key = f"AeOhTool_{_tag_token(tool_name)}_{tag}"

    def _create(cls: Any, conv_state: Any = None, **params: Any) -> list[Any]:
        if params:
            raise ValueError(f"工具 {tool_name} 不接受初始化参数")
        return [
            cls(
                name=tool_name,
                description=description,
                action_type=action_cls,
                observation_type=None,
                executor=executor,
                annotations=rt["ToolAnnotations"](
                    readOnlyHint=True, destructiveHint=False,
                    idempotentHint=True, openWorldHint=False,
                ),
            )
        ]

    tool_cls = type(
        reg_key, (rt["ToolDefinition"],),
        {"__module__": __name__, "name": tool_name,
         "create": classmethod(_create)},
    )
    rt["register_tool"](reg_key, tool_cls)
    return rt["Tool"](name=reg_key)


def _make_tools(spec: dict) -> list[Any]:
    """按 spec["tool_specs"] 注册自定义工具并返回 Agent.tools 规格列表。

    每个工具动态建一个 ToolDefinition 子类（注册键=类名，展示名用原工具名，
    保证 LLM 侧工具名仍为 search_flights 等）。注册键每次运行带随机后缀，
    进程内多次运行互不覆盖（避免 register_tool 重复注册告警与旧实例串味）。
    """
    rt = _oh_rt()
    tag = f"{_tag_token(str(spec.get('run_id', 'run')))}" + uuid.uuid4().hex[:6]
    obs_cls = type(f"AeOhObservation_{tag}", (rt["Observation"],), {})
    return [_register_one_tool(rt, spec, tag, obs_cls, entry)
            for entry in spec.get("tool_specs") or []]


def _make_llm(spec: dict) -> Any:
    """按 spec 构造 OpenHands SDK LLM（模块级可注入，测试替换假 LLM）。

    model 保留 litellm provider 前缀（W2 真实实验教训）；api_key 取
    spec["api_key_env"] 指向的环境变量；temperature / max_output_tokens /
    timeout 取自 model_params（默认 0.2 / 4096 / 120s）。
    """
    rt = _oh_rt()
    params = spec.get("model_params") or {}
    env_name = spec.get("api_key_env") or ""
    api_key = os.environ.get(env_name) if env_name else None
    return rt["LLM"](
        model=spec["litellm_model"],
        base_url=spec.get("api_base") or None,
        api_key=api_key or None,
        temperature=params.get("temperature", 0.2),
        max_output_tokens=int(params.get("max_tokens", 4096)),
        timeout=int(float(params.get("api_timeout_s", 120))),
        num_retries=1,
    )


def _make_agent(spec: dict, llm: Any, tools: list[Any]) -> Any:
    """构造 OpenHands Agent：注入 LLM/自定义工具/中文系统提示 + finish 收尾。

    include_default_tools=["FinishTool"]：保留 SDK 官方"任务完成"工具（finish），
    同时关闭其余默认工具（本 SDK 版本内置仅 FinishTool/ThinkTool，无 bash/文件
    等危险默认工具，[] 亦可——这里显式保留 finish 以给模型确定的收尾信号）。
    """
    rt = _oh_rt()
    return rt["Agent"](
        llm=llm,
        tools=tools,
        include_default_tools=["FinishTool"],
        system_prompt=build_oh_system_prompt(spec),
    )


def _make_conversation(agent: Any, workspace: str, callbacks: list[Any],
                       budget: RunBudget) -> Any:
    """构造 LocalConversation（模块级可注入，测试替换假会话）。

    - max_iteration_per_run=budget.max_steps：SDK 循环内每次 agent.step 恰为一次
      模型调用，达到上限且未 finish 时 run() 以 MaxIterationsReached 终止（事件
      回调置 budget_exceeded）——模型回合预算口径与 react max_steps 一致；
    - stuck_detection=False：异常重复模式守卫与本适配器预算语义重叠，关闭以免
      提前误停；visualizer=None 关闭控制台可视化输出。
    """
    rt = _oh_rt()
    return rt["LocalConversation"](
        agent=agent,
        workspace=workspace,
        callbacks=callbacks,
        max_iteration_per_run=max(int(budget.max_steps), 1),
        stuck_detection=False,
        visualizer=None,
    )


# --------------------------------------------------------------------------
# 事件流 → Trace 映射（SDK 1.21 openhands/sdk/event 对象）
# --------------------------------------------------------------------------
def _event_message_text(ev: Any) -> str:
    """取 MessageEvent 的可见文本（TextContent 拼接；取不到为空串）。"""
    msg = getattr(ev, "llm_message", None)
    if msg is None:
        return ""
    parts: list[str] = []
    for c in getattr(msg, "content", None) or []:
        text = getattr(c, "text", None)
        if text is not None:
            parts.append(str(text))
    return "".join(parts)


def _event_tool_args(ev: Any) -> dict[str, Any]:
    """取 ActionEvent 的工具实参（剔除 SDK 注入的 summary/security_risk 元字段）。"""
    tc = getattr(ev, "tool_call", None)
    if tc is None:
        return {}
    arguments = getattr(tc, "arguments", None)
    obj: Any = {}
    if isinstance(arguments, str):
        try:
            obj = json.loads(arguments)
        except Exception:
            obj = {}
    elif isinstance(arguments, dict):
        obj = arguments
    if not isinstance(obj, dict):
        return {}
    return {k: v for k, v in obj.items()
            if k not in ("summary", "security_risk")}


def _event_observation_text(ev: Any) -> str:
    """取 ObservationEvent 观测文本（Observation.text；取不到为空串）。"""
    obs = getattr(ev, "observation", None)
    if obs is None:
        return ""
    text = getattr(obs, "text", None)
    return str(text) if text is not None else ""


class _OHMapper:
    """把一次 OpenHands 运行的事件流翻译为 Trace 步骤（事件类名按 SDK 1.21 对齐）。

    - ActionEvent（非 finish、校验通过）→ message/assistant + tool_call 两步；
    - MessageEvent(source=agent 且有文本) → message/assistant 步并记为最终答复；
    - ObservationEvent（非 finish）→ observation 步（content=服务端 JSON 文本）；
    - AgentErrorEvent → observation 步（"工具调用异常: ..."，同 react 回填口径）；
    - ConversationErrorEvent(code=MaxIterationsReached) → budget_exceeded=True；
    - finish 工具 → 记 assistant 步并从 message 参数提取最终答复，不再记
      tool_call/observation（最终答复语义与 react 的 final_answer 对齐）。
    token/cost 采用 best-effort：从 LLM metrics 增量取值，取不到记 0.0。
    """

    FINISH_TOOL = "finish"

    def __init__(self, trace: Trace, spec: dict, budget: RunBudget,
                 raw_lines: list[str], usage_llm: Any = None) -> None:
        self.trace = trace
        self.spec = spec
        self.budget = budget
        self.raw_lines = raw_lines
        self.usage_llm = usage_llm
        self.final_answer = ""
        self.budget_exceeded = False
        self._usage_idx = 0
        self._cost_seen = 0.0

    def _poll_usage(self) -> tuple[int, int, float]:
        """从 usage_llm.metrics 取"刚完成那次调用"的 token/成本增量（best-effort）。

        LocalConversation 每次 agent.step 恰好完成一次模型调用并即时把 usage 记入
        llm.metrics（token_usages 每调用追加一条）；在首个属于该调用的 agent 事件
        到达时取增量即可对齐。取不到（fake/异常路径）一律返回 0。TODO(§7.3)：
        成本在 provider 前缀模型下通常为 0，后续可用 litellm 回调按
        input/output_cost_per_token 精算。
        """
        llm = self.usage_llm
        if llm is None:
            return 0, 0, 0.0
        metrics = getattr(llm, "metrics", None)
        if metrics is None:
            return 0, 0, 0.0
        usages = list(getattr(metrics, "token_usages", None) or [])
        tokens_in = tokens_out = 0
        idx = len(usages)
        if idx > self._usage_idx and usages:
            latest = usages[idx - 1]
            tokens_in = int(getattr(latest, "prompt_tokens", 0) or 0)
            tokens_out = int(getattr(latest, "completion_tokens", 0) or 0)
        self._usage_idx = idx
        cost = float(getattr(metrics, "accumulated_cost", 0.0) or 0.0)
        cost_delta = max(cost - self._cost_seen, 0.0)
        self._cost_seen = cost
        return tokens_in, tokens_out, cost_delta

    def _record_assistant(self, content: str, tokens_in: int,
                          tokens_out: int, cost_usd: float) -> None:
        """追加 message/assistant 步并同步原始模型输出日志。"""
        self.raw_lines.append(content)
        _record(self.trace, "message", "assistant", content=content,
                tokens_in=tokens_in, tokens_out=tokens_out, cost_usd=cost_usd)

    def __call__(self, event: Any) -> None:
        handler = getattr(self, f"_on_{type(event).__name__}", None)
        if handler is not None:
            handler(event)

    def _on_ActionEvent(self, ev: Any) -> None:
        name = str(getattr(ev, "tool_name", "") or "")
        args = _event_tool_args(ev)
        action = getattr(ev, "action", None)
        if name == self.FINISH_TOOL:
            # finish：模型回合 + 最终答复（message 参数），不再记工具调用步骤
            tokens_in, tokens_out, cost_usd = self._poll_usage()
            raw = json.dumps({"tool": name, "args": args}, ensure_ascii=False)
            self._record_assistant(raw, tokens_in, tokens_out, cost_usd)
            message = args.get("message")
            if isinstance(message, str) and message.strip():
                self.final_answer = message
            return
        if action is None:
            # 参数校验失败的动作：不记 tool_call（未到达服务端），
            # 紧随其后的 AgentErrorEvent 会产出 observation 步供模型自纠
            self._poll_usage()
            return
        tokens_in, tokens_out, cost_usd = self._poll_usage()
        raw = json.dumps({"tool": name, "args": args}, ensure_ascii=False)
        self._record_assistant(raw, tokens_in, tokens_out, cost_usd)
        _record(self.trace, "tool_call", "assistant",
                tool_name=name, tool_args=args)

    def _on_MessageEvent(self, ev: Any) -> None:
        if getattr(ev, "source", None) != "agent":
            return  # 用户消息 / 纠错提示不进入轨迹
        tokens_in, tokens_out, cost_usd = self._poll_usage()
        text = _event_message_text(ev)
        if not text.strip():
            return  # 推理-only / 空回复：SDK 自行纠正，无可见内容可记
        self._record_assistant(text, tokens_in, tokens_out, cost_usd)
        self.final_answer = text

    def _on_ObservationEvent(self, ev: Any) -> None:
        if str(getattr(ev, "tool_name", "") or "") == self.FINISH_TOOL:
            return  # finish 的答复已从 ActionEvent.message 提取
        _record(self.trace, "observation", "tool",
                content=_event_observation_text(ev))

    def _on_AgentErrorEvent(self, ev: Any) -> None:
        err = str(getattr(ev, "error", "") or "")
        content = f"工具调用异常: {err}" if err else "工具调用异常"
        _record(self.trace, "observation", "tool", content=content)

    def _on_ConversationErrorEvent(self, ev: Any) -> None:
        if str(getattr(ev, "code", "") or "") == "MaxIterationsReached":
            self.budget_exceeded = True


def _make_event_handler(trace: Trace, spec: dict, budget: RunBudget,
                        raw_lines: list[str], usage_llm: Any = None) -> _OHMapper:
    """构造本次运行的事件→Trace 映射回调（模块级可注入，测试直接替换）。"""
    return _OHMapper(trace, spec, budget, raw_lines, usage_llm=usage_llm)


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
    """真实模式：OpenHands SDK 1.21 headless 驱动（方案 A，§6.5③ W4 回归）。

    流程：_make_llm/_make_tools/_make_agent/_make_conversation 组装
    LocalConversation（自定义 18 工具 + finish，中文 system prompt）→
    send_message(goal) → run()（事件经 callbacks 由 _make_event_handler 映射为
    Trace 步骤）。

    预算语义：
    - 墙钟超时用 daemon 线程 + join(timeout)（同 smolagents）；超时置
      status=timeout 直接返回，**不调用任何 interrupt**（W2 死锁教训：worker 卡
      在模型调用时 interrupt 会阻塞等待其锁；worker 是 daemon 随进程回收）。
    - 模型回合数（max_steps）交给 LocalConversation(max_iteration_per_run=
      budget.max_steps) 兜底：run() 内部达到上限且未 finish 时以
      MaxIterationsReached 终止并返回（非抛异常），事件回调置
      budget_exceeded → status=budget_exceeded + 退出码 2。
    - cost 上限：usage 记 0.0（best-effort，见 _OHMapper._poll_usage TODO），
      与 smolagents 同口径暂不触发；异常兜底沿用 run_from_spec（勿动）。
    """
    t0 = time.time()
    sys_prompt = build_oh_system_prompt(spec)
    goal = str(spec["task"].get("user_goal", ""))
    _record(trace, "message", "system", content=sys_prompt)
    _record(trace, "message", "user", content=goal)

    box: dict[str, Any] = {}

    def worker() -> None:
        try:
            llm = _make_llm(spec)
            tools = _make_tools(spec)
            agent = _make_agent(spec, llm, tools)
            handler = _make_event_handler(trace, spec, budget, raw_lines,
                                          usage_llm=llm)
            ws_dir = tempfile.mkdtemp(prefix="openhands-ae-")
            box["ws_dir"] = ws_dir
            conv = _make_conversation(agent, ws_dir, [handler], budget)
            box["conv"] = conv
            box["handler"] = handler
            conv.send_message(goal)
            conv.run()
            box["ok"] = True
        except Exception as exc:  # 交给 run_from_spec 统一兜底（error + 退出码 1）
            box["exc"] = exc

    thread = threading.Thread(target=worker, daemon=True,
                              name="openhands-sdk-run")
    thread.start()
    remaining = max(budget.timeout_s - (time.time() - t0), 0.0)
    thread.join(timeout=remaining)

    if thread.is_alive():
        # 墙钟超时：worker 仍在跑（daemon，进程退出即回收），不 interrupt
        trace.status = STATUS_TIMEOUT
        return EXIT_BUDGET

    # 会话结束：先清理（close + 删临时工作目录），再判定结果
    conv = box.get("conv")
    if conv is not None:
        try:
            conv.close()
        except Exception as exc:  # 清理失败不影响结果判定
            raw_lines.append(f"[清理会话失败] {exc!r}")
    ws_dir = box.get("ws_dir")
    if isinstance(ws_dir, str) and ws_dir:
        shutil.rmtree(ws_dir, ignore_errors=True)

    if "exc" in box:
        raise box["exc"] from None
    handler = box.get("handler")
    if handler is not None and handler.budget_exceeded:
        trace.status = STATUS_BUDGET_EXCEEDED
        return EXIT_BUDGET
    if handler is not None and handler.final_answer:
        trace.final_answer = handler.final_answer
    trace.status = STATUS_COMPLETED
    return EXIT_OK


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
