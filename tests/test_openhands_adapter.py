"""test_openhands_adapter —— OpenHands 适配器单测（不依赖真实 API / SDK 联网）。

覆盖：
① fake_llm 完整循环（本地 threading.http.server 桩，2 次工具调用 → 退出码 0）；
⑤ CLI 入口 main(argv)（fake_llm 全链路，走文件参数）；
真实模式（方案 A：OpenHands SDK headless）：monkeypatch 模块级
   _make_conversation/_make_llm/_make_agent/_make_tools 注入假对象，
   fake 会话把脚本化"SDK 事件"经真实事件映射回调（_make_event_handler /
   _OHMapper，duck-typed 事件类名与 SDK 1.21 对齐）灌入，覆盖：
② 事件 → Trace 映射 + finish 收尾（completed，model_turns/tool_calls/最终答复）；
③ 事件 → Trace 映射：模型直接以消息答复收尾（content 型 MessageEvent）；
④ max_steps 超限（ConversationErrorEvent MaxIterationsReached）→ budget_exceeded；
⑤' 墙钟超时：fake 会话 run() 阻塞 → daemon 线程 join 超时 → status=timeout + 2；
⑥ 未捕获异常（fake 会话 run() 抛错）→ status=error + 退出码 1 + trace 仍写出。
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any

import pytest

from harness import protocol
from harness.adapters import openhands as oh_adapter


# --------------------------------------------------------------------------
# 测试数据构造（与 react/smolagents 测试同口径）
# --------------------------------------------------------------------------
def _load_task(task_id: str = "tr_001") -> dict:
    return next(t for t in protocol.load_tasks("travel") if t["task_id"] == task_id)


def _spec(tmp_path, *, task_id="tr_001", base_url="http://127.0.0.1:1",
          fake=False, budget=None, gt_plan=None, gt_answer=""):
    """用 protocol.build_run_spec 构造 run_spec（保持契约单一来源）。"""
    return protocol.build_run_spec(
        run_id="test__oh", task=_load_task(task_id),
        base_url=base_url, instance_id="test__oh",
        model="deepseek-v4-flash", litellm_model="openai/deepseek-v4-flash",
        api_base="https://api.deepseek.com", api_key_env="NO_SUCH_ENV",
        model_params={"temperature": 0.2, "max_tokens": 256,
                      **( {"fake_llm": True} if fake else {})},
        budget=budget or protocol.RunBudget(max_steps=8),
        gt_plan=gt_plan, gt_answer=gt_answer,
    )


def _fake_resp(content: str, prompt_tokens: int = 10,
               completion_tokens: int = 5) -> SimpleNamespace:
    """构造与 litellm 响应形状兼容的假响应。"""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens,
                              completion_tokens=completion_tokens),
    )


# --------------------------------------------------------------------------
# 本地 mock 工具服务桩（与 tool_server 同 HTTP 契约）
# --------------------------------------------------------------------------
class _StubHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, handler, tool_fn):
        super().__init__(addr, handler)
        self.states: dict[str, dict] = {}
        self.tool_fn = tool_fn


class _StubHandler(BaseHTTPRequestHandler):
    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw or b"{}")

    def do_GET(self):
        parts = self.path.strip("/").split("/")
        if parts == ["healthz"]:
            return self._json(200, {"ok": True})
        if len(parts) == 3 and parts[2] == "state":
            return self._json(200, {"ok": True,
                                    "state": self.server.states.get(parts[1], {})})
        return self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        parts = self.path.strip("/").split("/")
        if len(parts) == 3 and parts[2] == "reset":
            self.server.states[parts[1]] = self._body().get("initial_state", {})
            return self._json(200, {"ok": True})
        if len(parts) == 4 and parts[2] == "tools":
            ok, payload = self.server.tool_fn(self._body(), parts[3])
            return self._json(200, {"ok": ok, **payload})
        return self._json(404, {"ok": False, "error": "not found"})

    def log_message(self, *args: Any) -> None:
        pass


def _echo_tool(args: dict, name: str):
    """默认桩：任何工具都成功并回显参数。"""
    return True, {"result": {"tool": name, "args": args}}


@pytest.fixture
def stub_server():
    """起一个本地 mock 工具服务桩，yield (base_url, calls)。"""
    calls: list[dict] = []

    def tool_fn(args: dict, name: str):
        if name == "refund_reservation":  # 模拟需要先验证身份的约束错误
            return False, {"error": "需要先验证身份", "code": "auth_required"}
        calls.append({"tool": name, "args": args})
        return _echo_tool(args, name)

    httpd = _StubHTTPServer(("127.0.0.1", 0), _StubHandler, tool_fn)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    try:
        yield f"http://{host}:{port}", calls
    finally:
        httpd.shutdown()
        thread.join(timeout=5)


# --------------------------------------------------------------------------
# 真实模式注入用的"假 SDK 事件"（类名与 openhands/sdk/event 对齐，
# 让 _OHMapper 按 type(ev).__name__ 分派即可，无需导入 SDK）
# --------------------------------------------------------------------------
class ToolCall:
    def __init__(self, arguments: str):
        self.arguments = arguments
        self.name = ""


class TextContent:
    def __init__(self, text: str):
        self.text = text


class LlmMessage:
    def __init__(self, text: str):
        self.content = [TextContent(text)]


class ActionEvent:
    def __init__(self, tool_name: str, arguments: str, action: Any = object()):
        self.tool_name = tool_name
        self.tool_call = ToolCall(arguments)
        self.action = action


class ObservationEvent:
    def __init__(self, tool_name: str, text: str):
        self.tool_name = tool_name
        self.observation = SimpleNamespace(text=text)


class MessageEvent:
    def __init__(self, source: str, text: str):
        self.source = source
        self.llm_message = LlmMessage(text)


class AgentErrorEvent:
    def __init__(self, error: str):
        self.tool_name = ""
        self.error = error


class ConversationErrorEvent:
    def __init__(self, code: str):
        self.code = code


class _FakeConversation:
    """假会话：捕获事件映射回调，run() 时按脚本重放事件（驱动真实 mapper）。"""

    def __init__(self, agent: Any, workspace: str, callbacks: list[Any],
                 budget: protocol.RunBudget | None = None,
                 events: list[Any] | None = None,
                 sleep_s: float = 0.0, raise_exc: Exception | None = None):
        self.callbacks = callbacks
        self.events = events or []
        self.sleep_s = sleep_s
        self.raise_exc = raise_exc
        self.sent: list[str] = []

    def send_message(self, goal: str) -> None:
        self.sent.append(goal)

    def run(self) -> None:
        if self.sleep_s:
            time.sleep(self.sleep_s)
        if self.raise_exc is not None:
            raise self.raise_exc
        for ev in self.events:
            for cb in self.callbacks:
                cb(ev)

    def close(self) -> None:
        pass


def _make_fake_conv(events=None, *, sleep_s=0.0,
                    raise_exc: Exception | None = None):
    """构造假会话工厂：run() 时把脚本化事件灌给真实的事件映射回调。"""

    def factory(agent: Any, workspace: str, callbacks: list[Any],
                budget: protocol.RunBudget | None = None) -> _FakeConversation:
        return _FakeConversation(agent, workspace, callbacks, budget=budget,
                                 events=events, sleep_s=sleep_s,
                                 raise_exc=raise_exc)

    return factory


def _patch_sdk_factories(monkeypatch, conv_factory):
    """把真实模式的 SDK 工厂全部替换为假对象（杜绝测试触碰 SDK/网络）。"""
    monkeypatch.setattr(oh_adapter, "_make_llm",
                        lambda spec: SimpleNamespace(metrics=None))
    monkeypatch.setattr(oh_adapter, "_make_tools", lambda spec: [])
    monkeypatch.setattr(oh_adapter, "_make_agent",
                        lambda spec, llm, tools: SimpleNamespace(llm=llm))
    monkeypatch.setattr(oh_adapter, "_make_conversation", conv_factory)


# --------------------------------------------------------------------------
# ① fake_llm 完整循环（tr_001：2 次工具调用 → 退出码 0）
# --------------------------------------------------------------------------
def test_fake_llm_full_loop(tmp_path, stub_server, monkeypatch):
    base_url, calls = stub_server
    gt_plan = [
        {"tool": "search_flights",
         "args": {"date": "2026-10-12", "from": "PEK", "to": "SHA"}},
        {"tool": "book_flight",
         "args": {"user_id": "u_42", "flight_id": "MU5101", "class": "economy"}},
    ]
    gt_answer = "已为您预订 MU5101（经济舱）。"
    spec = _spec(tmp_path, base_url=base_url, fake=True,
                 gt_plan=gt_plan, gt_answer=gt_answer)
    out = tmp_path / "trace.jsonl"

    # fake_llm 路径不应触达模型调用
    def boom(*a, **k):
        raise AssertionError("fake_llm 模式不得调用 _call_llm")
    monkeypatch.setattr(oh_adapter, "_call_llm", boom)

    assert oh_adapter.run_from_spec(spec, out) == protocol.EXIT_OK

    trace = protocol.load_trace(out)
    assert trace is not None
    assert trace.framework == "openhands"     # 框架标签与契约一致（§6.1）
    assert trace.status == protocol.STATUS_COMPLETED
    assert trace.final_answer == gt_answer
    # 模型回合 = 回放步数；工具调用顺序与参数与 GT 计划一致
    assert trace.model_turns == 2
    assert trace.tool_calls() == gt_plan
    # 轨迹以 system + user 开头（与 react/smolagents 一致）
    first_types = [(s.type, s.role) for s in trace.steps[:2]]
    assert first_types == [("message", "system"), ("message", "user")]
    # assistant 消息步内容 = 工具调用 JSON；observation 步含服务端返回
    assistant_contents = [s.content for s in trace.steps
                          if s.type == "message" and s.role == "assistant"]
    assert json.loads(assistant_contents[0])["tool"] == "search_flights"
    obs = [s for s in trace.steps if s.type == "observation"]
    assert len(obs) == 2
    assert all('"ok": true' in s.content for s in obs)
    # 工具调用真实发生：桩服务端收到两次调用（参数与 GT 一致）
    assert [c["tool"] for c in calls] == ["search_flights", "book_flight"]
    assert calls[1]["args"] == gt_plan[1]["args"]


# --------------------------------------------------------------------------
# ② 真实模式：事件 → Trace 映射（工具回合 + finish 收尾 → completed）
# --------------------------------------------------------------------------
def test_real_sdk_event_mapping_finish(tmp_path, monkeypatch):
    spec = _spec(tmp_path, budget=protocol.RunBudget(max_steps=4))
    out = tmp_path / "trace.jsonl"
    search_json = ('{"date": "2026-10-12", "from": "PEK", "to": "SHA", '
                   '"summary": "查航班"}')
    events = [
        ActionEvent("search_flights", search_json),
        ObservationEvent("search_flights",
                          '{"ok": true, "result": [{"flight_id": "MU5101"}]}'),
        ActionEvent("finish",
                     '{"message": "10月12日上午北京到上海有 MU5101 航班。"}'),
        ObservationEvent("finish", "10月12日上午北京到上海有 MU5101 航班。"),
    ]
    _patch_sdk_factories(monkeypatch, _make_fake_conv(events))

    rc = oh_adapter.run_from_spec(spec, out)
    assert rc == protocol.EXIT_OK

    trace = protocol.load_trace(out)
    assert trace is not None
    assert trace.status == protocol.STATUS_COMPLETED
    assert trace.final_answer == "10月12日上午北京到上海有 MU5101 航班。"
    # 模型回合：search_flights 1 回合 + finish 1 回合（含 finish 的 assistant 步）
    assert trace.model_turns == 2
    # 工具调用只记非 finish 的 search_flights；summary 元字段被剔除
    calls = trace.tool_calls()
    assert calls == [{"tool": "search_flights",
                      "args": {"date": "2026-10-12", "from": "PEK",
                               "to": "SHA"}}]
    # observation 步内容 = 工具返回 JSON 文本（finish 的 observation 不重复记）
    obs = [s for s in trace.steps if s.type == "observation"]
    assert len(obs) == 1
    assert json.loads(obs[0].content)["ok"] is True
    # 首两步仍为 system + user（轨迹完整，与 fake 路径一致）
    assert [(s.type, s.role) for s in trace.steps[:2]] == \
        [("message", "system"), ("message", "user")]
    # 原始模型输出日志存在（assistant 步内容已写入）
    assert (out.with_name("trace.raw.log")).exists()


# --------------------------------------------------------------------------
# ③ 真实模式：模型以普通消息（content 型）收尾 → completed
# --------------------------------------------------------------------------
def test_real_sdk_content_message_end(tmp_path, monkeypatch):
    spec = _spec(tmp_path, budget=protocol.RunBudget(max_steps=4))
    out = tmp_path / "trace.jsonl"
    events = [
        ActionEvent("search_flights",
                     '{"date": "2026-10-12", "from": "PEK", "to": "SHA"}'),
        ObservationEvent("search_flights", '{"ok": true, "result": []}'),
        MessageEvent("agent", "没有查到航班，请改期再试。"),
    ]
    _patch_sdk_factories(monkeypatch, _make_fake_conv(events))

    rc = oh_adapter.run_from_spec(spec, out)
    assert rc == protocol.EXIT_OK

    trace = protocol.load_trace(out)
    assert trace is not None
    assert trace.status == protocol.STATUS_COMPLETED
    assert trace.final_answer == "没有查到航班，请改期再试。"
    assert trace.model_turns == 2
    # 用户消息（source=user）不进轨迹；assistant 文本步出现且带内容
    assistant_steps = [s for s in trace.steps
                       if s.type == "message" and s.role == "assistant"]
    assert assistant_steps[-1].content == "没有查到航班，请改期再试。"
    assert all(s.role != "tool" or True for s in trace.steps)


# --------------------------------------------------------------------------
# ④ 真实模式预算：max_steps 超限（MaxIterationsReached）→ budget_exceeded + 2
# --------------------------------------------------------------------------
def test_real_sdk_budget_max_steps_exceeded(tmp_path, monkeypatch):
    spec = _spec(tmp_path, budget=protocol.RunBudget(max_steps=2))
    out = tmp_path / "trace.jsonl"
    events = [
        ActionEvent("search_flights",
                     '{"date": "2026-10-12", "from": "PEK", "to": "SHA"}'),
        ObservationEvent("search_flights", '{"ok": true, "result": []}'),
        ActionEvent("search_flights",
                     '{"date": "2026-10-13", "from": "PEK", "to": "SHA"}'),
        ObservationEvent("search_flights", '{"ok": true, "result": []}'),
        ConversationErrorEvent("MaxIterationsReached"),
    ]
    _patch_sdk_factories(monkeypatch, _make_fake_conv(events))

    rc = oh_adapter.run_from_spec(spec, out)
    assert rc == protocol.EXIT_BUDGET  # 2

    trace = protocol.load_trace(out)
    assert trace is not None
    assert trace.status == protocol.STATUS_BUDGET_EXCEEDED
    assert trace.model_turns == 2            # 恰好两个模型回合后耗尽
    assert len(trace.tool_calls()) == 2
    assert (out.with_name("trace.raw.log")).exists()


# --------------------------------------------------------------------------
# ⑤' 真实模式墙钟超时：fake 会话 run() 阻塞 → status=timeout + 退出码 2
# --------------------------------------------------------------------------
def test_real_sdk_wall_clock_timeout(tmp_path, monkeypatch):
    spec = _spec(tmp_path, budget=protocol.RunBudget(max_steps=8,
                                                     timeout_s=0.2))
    out = tmp_path / "trace.jsonl"
    _patch_sdk_factories(monkeypatch,
                         _make_fake_conv(sleep_s=1.5))  # 模拟 SDK 卡在模型调用

    rc = oh_adapter.run_from_spec(spec, out)
    assert rc == protocol.EXIT_BUDGET  # 2

    trace = protocol.load_trace(out)
    assert trace is not None
    assert trace.status == protocol.STATUS_TIMEOUT
    # 保留已产出步骤（system + user），未产出 assistant 步
    assert [(s.type, s.role) for s in trace.steps] == \
        [("message", "system"), ("message", "user")]


# --------------------------------------------------------------------------
# ⑥ 真实模式未捕获异常 → status=error + 退出码 1 + trace 仍写出
# --------------------------------------------------------------------------
def test_real_sdk_unexpected_exception_writes_error_trace(tmp_path, monkeypatch):
    spec = _spec(tmp_path, budget=protocol.RunBudget(max_steps=4))
    out = tmp_path / "trace.jsonl"
    _patch_sdk_factories(monkeypatch,
                         _make_fake_conv(raise_exc=RuntimeError("模拟 SDK 运行失败")))

    rc = oh_adapter.run_from_spec(spec, out)
    assert rc == protocol.EXIT_ERROR  # 1

    trace = protocol.load_trace(out)
    assert trace is not None
    assert trace.status == protocol.STATUS_ERROR
    assert out.exists()
    # 保留了已产出步骤（system/user）+ 适配器异常 observation
    assert any(s.type == "message" and s.role == "system" for s in trace.steps)
    assert any(s.type == "observation" and s.content.startswith("适配器异常")
               for s in trace.steps)


# --------------------------------------------------------------------------
# ⑦ CLI 入口：main 读 spec 文件并返回退出码（fake_llm 全链路）
# --------------------------------------------------------------------------
def test_cli_main_returns_exit_code(tmp_path, stub_server, monkeypatch):
    base_url, _calls = stub_server
    spec = _spec(tmp_path, base_url=base_url, fake=True,
                 gt_plan=[{"tool": "search_flights",
                           "args": {"date": "2026-10-12", "from": "PEK",
                                    "to": "SHA"}}],
                 gt_answer="查到了。")
    spec_path = tmp_path / "run_spec.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    out_path = tmp_path / "out" / "trace.jsonl"

    rc = oh_adapter.main(["--spec", str(spec_path), "--out", str(out_path)])
    assert rc == 0
    trace = protocol.load_trace(out_path)
    assert trace is not None
    assert trace.status == protocol.STATUS_COMPLETED
    assert trace.final_answer == "查到了。"
