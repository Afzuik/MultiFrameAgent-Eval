"""test_smolagents_adapter —— smolagents 适配器单测（不依赖真实 API/LLM）。

覆盖（与 test_react_adapter.py 同一套本地 threading.http.server 桩思路）：
① fake_llm 完整循环：tr_004 spec 手工构造 gt_plan（2 次工具调用）→
   真实 HTTP 调 mock 桩 → trace.tool_calls 顺序/参数正确、final_answer=gt_answer、退出码 0；
② 真实模式（monkeypatch 假模型）：工具返回 ok:false 时 observation 正确回填；
③ 预算：max_steps 极小 + 真实模式假模型反复调用工具 → budget_exceeded + 退出码 2；
④ 运行异常（模型抛错）→ status=error + 退出码 1 + trace 仍写出；
⑤ CLI 入口 main(argv)（fake_llm 全链路，走文件参数）；
⑥（附加）wall-clock 超时 → status=timeout + 退出码 2。
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
from smolagents.models import ChatMessage, MessageRole, Model, TokenUsage

from harness import protocol
from harness.adapters import smolagents as sm_adapter


# --------------------------------------------------------------------------
# 测试数据构造
# --------------------------------------------------------------------------
def _load_task(task_id: str = "tr_004") -> dict:
    return next(t for t in protocol.load_tasks("travel") if t["task_id"] == task_id)


def _spec(tmp_path, *, task_id="tr_004", base_url="http://127.0.0.1:1",
          fake=False, budget=None, gt_plan=None, gt_answer=""):
    """用 protocol.build_run_spec 构造 run_spec（保持契约单一来源）。"""
    return protocol.build_run_spec(
        run_id="test__smol", task=_load_task(task_id),
        base_url=base_url, instance_id="test__smol",
        model="deepseek-v4-flash", litellm_model="openai/deepseek-v4-flash",
        api_base="https://api.deepseek.com", api_key_env="NO_SUCH_ENV",
        model_params={"temperature": 0.2, "max_tokens": 256,
                      **( {"fake_llm": True} if fake else {})},
        budget=budget or protocol.RunBudget(max_steps=8),
        gt_plan=gt_plan, gt_answer=gt_answer,
    )


class _FakeModel(Model):
    """按脚本序列返回 ChatMessage 的假模型（供真实模式 monkeypatch _make_model）。

    - outputs：每次 generate 依次返回的模型输出文本；
    - loop=True 时输出耗尽后循环使用最后一条（模拟模型反复调用工具不收敛）；
    - delay>0 时每次 generate 先 sleep（模拟慢模型，用于超时测试）；
    - raise_error=True 时 generate 直接抛 RuntimeError（模拟 API 故障）。
    """

    def __init__(self, outputs: list[str], *, loop: bool = False,
                 delay: float = 0.0, raise_error: bool = False):
        super().__init__(model_id="fake-smolagents-model")
        self.outputs = list(outputs)
        self.loop = loop
        self.delay = delay
        self.raise_error = raise_error
        self.calls = 0

    def generate(self, messages, stop_sequences=None, response_format=None,
                 tools_to_call_from=None, **kwargs) -> ChatMessage:
        if self.delay:
            time.sleep(self.delay)
        if self.raise_error:
            raise RuntimeError("API 挂了")
        if not self.outputs:
            raise AssertionError("假模型没有可用的脚本输出")
        idx = self.calls if self.calls < len(self.outputs) else len(self.outputs) - 1
        if self.loop:
            idx = self.calls % len(self.outputs)
        content = self.outputs[idx]
        self.calls += 1
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=content,
            token_usage=TokenUsage(input_tokens=10, output_tokens=4),
        )


# --------------------------------------------------------------------------
# 本地 mock 工具服务桩（与 tool_server 同 HTTP 契约，思路同 react 测试）
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
    """起一个本地 mock 工具服务桩，yield (base_url, calls)。

    calls 记录服务端收到的每次工具调用，用于断言"真实 HTTP 调工具"。
    """
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
# ① fake_llm 完整循环（tr_004：2 次工具调用 → 退出码 0）
# --------------------------------------------------------------------------
def test_fake_llm_full_loop(tmp_path, stub_server, monkeypatch):
    base_url, calls = stub_server
    gt_plan = [
        {"tool": "search_flights",
         "args": {"date": "2026-10-12", "from": "PEK", "to": "SHA"}},
        {"tool": "book_flight",
         "args": {"user_id": "u_42", "flight_id": "MU5101", "class": "economy"}},
    ]
    gt_answer = "已为您预订上午的 MU5101（经济舱），价格 980 元。"
    spec = _spec(tmp_path, base_url=base_url, fake=True,
                 gt_plan=gt_plan, gt_answer=gt_answer)
    out = tmp_path / "trace.jsonl"

    # fake_llm 路径不应触达模型构造
    def boom(*a, **k):
        raise AssertionError("fake_llm 模式不得调用 _make_model")
    monkeypatch.setattr(sm_adapter, "_make_model", boom)

    assert sm_adapter.run_from_spec(spec, out) == protocol.EXIT_OK

    trace = protocol.load_trace(out)
    assert trace is not None
    assert trace.status == protocol.STATUS_COMPLETED
    assert trace.final_answer == gt_answer
    # 模型回合 = 回放步数；工具调用顺序与参数与 GT 计划一致
    assert trace.model_turns == 2
    assert trace.tool_calls() == gt_plan
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
# ② 真实模式：工具返回 ok:false → observation 正确回填
# --------------------------------------------------------------------------
def test_tool_ok_false_observation(tmp_path, stub_server, monkeypatch):
    base_url, calls = stub_server
    spec = _spec(tmp_path, base_url=base_url,
                 budget=protocol.RunBudget(max_steps=4))
    out = tmp_path / "trace.jsonl"
    monkeypatch.setattr(sm_adapter, "_make_model", lambda spec_: _FakeModel([
        ("Thought: 用户申请退款，先调用退款工具。\n<code>\n"
         "r = refund_reservation(user_id='u_42', reservation_id='R_091')\n"
         "print(r)\n</code>"),
        ("Thought: 服务端要求先验证身份，向用户说明。\n<code>\n"
         "final_answer('抱歉，退款需要先完成身份验证。')\n</code>"),
    ]))

    rc = sm_adapter.run_from_spec(spec, out)
    assert rc == protocol.EXIT_OK

    trace = protocol.load_trace(out)
    assert trace is not None
    assert trace.status == protocol.STATUS_COMPLETED
    assert trace.final_answer == "抱歉，退款需要先完成身份验证。"
    assert trace.model_turns == 2
    # observation 步内容 = 完整返回 JSON（含 ok:false 与错误信息）
    obs = [s for s in trace.steps if s.type == "observation"]
    assert len(obs) == 1
    payload = json.loads(obs[0].content)
    assert payload["ok"] is False
    assert payload["error"] == "需要先验证身份"
    # 工具调用真实发生（refund_reservation 是 stub 桩的 ok:false 分支，不入 calls 列表）
    tool_calls = [s for s in trace.steps if s.type == "tool_call"]
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_name == "refund_reservation"
    assert calls == []  # ok:false 分支不记录 calls（桩内刻意区分）
    # raw 日志记录了两次模型输出
    assert (out.with_name("trace.raw.log")).exists()


# --------------------------------------------------------------------------
# ③ 预算：max_steps 极小 + 真实模式假模型反复调用工具 → budget_exceeded + 2
# --------------------------------------------------------------------------
def test_max_steps_budget_exceeded(tmp_path, monkeypatch):
    spec = _spec(tmp_path, budget=protocol.RunBudget(max_steps=2))
    out = tmp_path / "trace.jsonl"

    tool_code = ("Thought: 反复查询航班不收敛。\n<code>\n"
                 "r = search_flights(date='2026-10-12', from_='PEK', to='SHA')\n"
                 "print(r)\n</code>")
    monkeypatch.setattr(sm_adapter, "_make_model",
                        lambda spec_: _FakeModel([tool_code], loop=True))
    # 不依赖真实 HTTP：工具调用统一回填固定响应
    monkeypatch.setattr(sm_adapter, "_invoke_tool",
                        lambda spec_, name, args: '{"ok": true, "result": {}}')

    rc = sm_adapter.run_from_spec(spec, out)
    assert rc == protocol.EXIT_BUDGET  # 2

    trace = protocol.load_trace(out)
    assert trace is not None
    assert trace.status == protocol.STATUS_BUDGET_EXCEEDED
    assert trace.model_turns == 2            # 恰好两个 agent step 后耗尽
    calls = trace.tool_calls()
    assert len(calls) == 2
    # from_（Python 关键字 from 的映射名）被还原为 JSON 键 from
    assert all(c["args"] == {"date": "2026-10-12", "from": "PEK", "to": "SHA"}
               for c in calls)
    assert (out.with_name("trace.raw.log")).exists()


# --------------------------------------------------------------------------
# ④ 未捕获异常 → status=error + 退出码 1 + trace 仍写出
# --------------------------------------------------------------------------
def test_unexpected_exception_writes_error_trace(tmp_path, monkeypatch):
    spec = _spec(tmp_path, budget=protocol.RunBudget(max_steps=4))
    out = tmp_path / "trace.jsonl"
    monkeypatch.setattr(sm_adapter, "_make_model",
                        lambda spec_: _FakeModel([""], raise_error=True))

    rc = sm_adapter.run_from_spec(spec, out)
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
# ⑤ CLI 入口：main 读 spec 文件并返回退出码（fake_llm 全链路）
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

    rc = sm_adapter.main(["--spec", str(spec_path), "--out", str(out_path)])
    assert rc == 0
    trace = protocol.load_trace(out_path)
    assert trace is not None
    assert trace.status == protocol.STATUS_COMPLETED
    assert trace.final_answer == "查到了。"


# --------------------------------------------------------------------------
# ⑥（附加）wall-clock 超时 → status=timeout + 退出码 2
# --------------------------------------------------------------------------
def test_wall_clock_timeout(tmp_path, monkeypatch):
    spec = _spec(tmp_path, budget=protocol.RunBudget(max_steps=8,
                                                     timeout_s=0.2))
    out = tmp_path / "trace.jsonl"

    tool_code = ("Thought: 执行慢模型调用。\n<code>\n"
                 "r = search_flights(date='2026-10-12')\nprint(r)\n</code>")
    monkeypatch.setattr(sm_adapter, "_make_model",
                        lambda spec_: _FakeModel([tool_code], loop=True,
                                                 delay=1.0))
    monkeypatch.setattr(sm_adapter, "_invoke_tool",
                        lambda spec_, name, args: '{"ok": true, "result": {}}')

    rc = sm_adapter.run_from_spec(spec, out)
    assert rc == protocol.EXIT_BUDGET  # 2
    trace = protocol.load_trace(out)
    assert trace is not None
    assert trace.status == protocol.STATUS_TIMEOUT
    assert out.exists()


# --------------------------------------------------------------------------
# Python 关键字参数名映射（from/class → from_/class_，trace 仍还原 JSON 键）
# --------------------------------------------------------------------------
def test_py_arg_name_mapping():
    assert sm_adapter._py_arg_name("from", set()) == "from_"
    assert sm_adapter._py_arg_name("class", set()) == "class_"
    assert sm_adapter._py_arg_name("user_id", set()) == "user_id"
    assert sm_adapter._py_arg_name("class", {"class_"}) == "class__"
