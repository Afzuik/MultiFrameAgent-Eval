"""test_react_adapter —— 自研 ReAct 适配器单测（不依赖真实 API）。

覆盖：
① 模型输出 JSON 解析（代码围栏 / 前后缀文本 / 非法输入）；
② fake_llm 完整循环跑通（本地 threading http.server 桩 + 真实 httpx POST）；
③ max_steps 超限 → status=budget_exceeded + 退出码 2；
④ 工具返回 ok:false 时 observation 正确回填（标准 tool 消息格式）。
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any

import pytest

from harness import protocol
from harness.adapters import react


# --------------------------------------------------------------------------
# 测试数据构造
# --------------------------------------------------------------------------
def _load_task(task_id: str = "tr_001") -> dict:
    return next(t for t in protocol.load_tasks("travel") if t["task_id"] == task_id)


def _spec(tmp_path, *, task_id="tr_001", base_url="http://127.0.0.1:1",
          fake=False, budget=None, gt_plan=None, gt_answer=""):
    """用 protocol.build_run_spec 构造 run_spec（保持契约单一来源）。"""
    return protocol.build_run_spec(
        run_id="test__run", task=_load_task(task_id),
        base_url=base_url, instance_id="test__run",
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
# ① JSON 解析
# --------------------------------------------------------------------------
class TestParseModelOutput:
    def test_fenced_json(self):
        text = '```json\n{"tool": "search_flights", "args": {"a": 1}}\n```'
        assert react.parse_model_output(text) == {
            "tool": "search_flights", "args": {"a": 1}}

    def test_prefix_and_suffix_text(self):
        text = '好的，我先查询一下：\n{"tool": "x", "args": {}}\n以上为计划。'
        assert react.parse_model_output(text) == {"tool": "x", "args": {}}

    def test_invalid_inputs_return_none(self):
        for text in ["", "这不是 JSON", "{" * 3, "```json\n{broken\n```",
                     "[1, 2, 3]", '"just a string"', None]:
            assert react.parse_model_output(text) is None

    def test_plain_json(self):
        assert react.parse_model_output('{"final_answer": "好了"}') == {
            "final_answer": "好了"}


# --------------------------------------------------------------------------
# 本地 mock 工具服务桩（与并行开发的 tool_server 同 HTTP 契约）
# --------------------------------------------------------------------------
class _StubHTTPServer(ThreadingHTTPServer):
    """带共享状态与工具处理器的 ThreadingHTTPServer。"""
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
# ② fake_llm 完整循环
# --------------------------------------------------------------------------
def test_fake_llm_full_loop(tmp_path, stub_server, monkeypatch):
    base_url, calls = stub_server
    gt_plan = [
        {"tool": "search_flights",
         "args": {"date": "2026-10-12", "from": "PEK", "to": "SHA"}},
        {"tool": "get_user_profile", "args": {"user_id": "u_42"}},
    ]
    spec = _spec(tmp_path, base_url=base_url, fake=True,
                 gt_plan=gt_plan, gt_answer="查询完成。")
    out = tmp_path / "trace.jsonl"

    # fake_llm 路径不应触达模型
    def boom(*a, **k):
        raise AssertionError("fake_llm 模式不得调用 _call_llm")
    monkeypatch.setattr(react, "_call_llm", boom)

    assert react.run_from_spec(spec, out) == protocol.EXIT_OK

    trace = protocol.load_trace(out)
    assert trace is not None
    assert trace.status == protocol.STATUS_COMPLETED
    assert trace.final_answer == "查询完成。"
    # 模型回合 = 回放步数；工具调用与 GT 计划一致
    assert trace.model_turns == 2
    assert trace.tool_calls() == [
        {"tool": "search_flights",
         "args": {"date": "2026-10-12", "from": "PEK", "to": "SHA"}},
        {"tool": "get_user_profile", "args": {"user_id": "u_42"}},
    ]
    # assistant 消息步内容 = 工具调用 JSON；observation 步含服务端返回
    assistant_contents = [s.content for s in trace.steps
                          if s.type == "message" and s.role == "assistant"]
    assert json.loads(assistant_contents[0])["tool"] == "search_flights"
    obs = [s for s in trace.steps if s.type == "observation"]
    assert len(obs) == 2
    assert all('"ok": true' in s.content for s in obs)
    # 工具调用真实发生：桩服务端收到两次调用
    assert [c["tool"] for c in calls] == ["search_flights", "get_user_profile"]


# --------------------------------------------------------------------------
# ③ max_steps 超限 → budget_exceeded + 退出码 2
# --------------------------------------------------------------------------
def test_max_steps_budget_exceeded(tmp_path, monkeypatch):
    spec = _spec(tmp_path, budget=protocol.RunBudget(max_steps=2))
    out = tmp_path / "trace.jsonl"

    scripted = [
        '{"tool": "search_flights", "args": {"date": "2026-10-12", "from": "PEK", "to": "SHA"}}'
    ]

    def fake_llm(spec_, messages):
        return scripted[0], _fake_resp(scripted[0])

    monkeypatch.setattr(react, "_call_llm", fake_llm)
    monkeypatch.setattr(react, "_invoke_tool",
                        lambda *a: '{"ok": true, "result": {}}')

    rc = react.run_from_spec(spec, out)
    assert rc == protocol.EXIT_BUDGET  # 2

    trace = protocol.load_trace(out)
    assert trace is not None
    assert trace.status == protocol.STATUS_BUDGET_EXCEEDED
    assert trace.model_turns == 2            # 恰好两次模型调用后耗尽
    assert len(trace.tool_calls()) == 2
    # raw 日志记录了原始输出
    assert (out.with_name("trace.raw.log")).exists()


# --------------------------------------------------------------------------
# ④ 工具 ok:false → observation 正确回填为 tool 消息
# --------------------------------------------------------------------------
def test_tool_error_backfilled_as_tool_message(tmp_path, monkeypatch):
    spec = _spec(tmp_path, budget=protocol.RunBudget(max_steps=4))
    out = tmp_path / "trace.jsonl"
    seen_messages: list[list[dict]] = []

    error_body = '{"ok": false, "error": "需要先验证身份", "code": "auth_required"}'
    scripted = [
        '{"tool": "refund_reservation", "args": {"user_id": "u_42", "reservation_id": "R_091"}}',
        '{"final_answer": "抱歉，我需要先验证身份。"}',
    ]

    def fake_llm(spec_, messages):
        seen_messages.append(list(messages))
        return scripted[len(seen_messages) - 1], _fake_resp(scripted[-1])

    monkeypatch.setattr(react, "_call_llm", fake_llm)
    monkeypatch.setattr(react, "_invoke_tool", lambda *a: error_body)

    rc = react.run_from_spec(spec, out)
    assert rc == protocol.EXIT_OK

    trace = protocol.load_trace(out)
    assert trace.status == protocol.STATUS_COMPLETED
    # observation 步内容 = 完整返回 JSON
    obs = [s for s in trace.steps if s.type == "observation"]
    assert obs and obs[0].content == error_body

    # 第二次调用时模型收到 assistant 原样输出 + user 消息携带工具结果
    # （DeepSeek 推理模型兼容回填口径：不合成 tool_calls/tool 消息）
    second = seen_messages[1]
    assert not any(m.get("role") == "tool" for m in second)
    user_msgs = [m for m in second if m.get("role") == "user"]
    assert any(f"工具返回: {error_body}" in str(m.get("content")) for m in user_msgs)
    assistant_msgs = [m for m in second if m.get("role") == "assistant"]
    assert any(m.get("content") == scripted[0] for m in assistant_msgs)


# --------------------------------------------------------------------------
# 解析失败重试：坏输出 → 记 observation + user 重试消息 → 恢复
# --------------------------------------------------------------------------
def test_parse_failure_retry_then_final_answer(tmp_path, monkeypatch):
    spec = _spec(tmp_path, budget=protocol.RunBudget(max_steps=4))
    out = tmp_path / "trace.jsonl"
    seen_messages: list[list[dict]] = []

    scripted = [
        "抱歉，我来分析一下：我认为应该先查询航班。",        # 无法解析
        '{"final_answer": "已完成。"}',
    ]

    def fake_llm(spec_, messages):
        seen_messages.append(list(messages))
        idx = len(seen_messages) - 1
        return scripted[idx], _fake_resp(scripted[idx])

    monkeypatch.setattr(react, "_call_llm", fake_llm)

    rc = react.run_from_spec(spec, out)
    assert rc == protocol.EXIT_OK
    trace = protocol.load_trace(out)
    assert trace.status == protocol.STATUS_COMPLETED
    # 解析失败的 assistant 输出也记为 message 步（预算口径一致）
    assert trace.model_turns == 2
    assert any(s.content.startswith("解析失败") for s in trace.steps
               if s.type == "observation")
    # 第二次调用携带了重试提示（role=user）
    assert seen_messages[1][-1]["role"] == "user"
    assert "无法解析为 JSON" in seen_messages[1][-1]["content"]


# --------------------------------------------------------------------------
# 未捕获异常 → status=error + 退出码 1 + trace 仍写出
# --------------------------------------------------------------------------
def test_unexpected_exception_writes_error_trace(tmp_path, monkeypatch):
    spec = _spec(tmp_path, budget=protocol.RunBudget(max_steps=4))
    out = tmp_path / "trace.jsonl"

    def boom(spec_, messages):
        raise RuntimeError("API 挂了")

    monkeypatch.setattr(react, "_call_llm", boom)

    rc = react.run_from_spec(spec, out)
    assert rc == protocol.EXIT_ERROR  # 1
    trace = protocol.load_trace(out)
    assert trace is not None
    assert trace.status == protocol.STATUS_ERROR
    assert out.exists()


# --------------------------------------------------------------------------
# CLI 入口：main 读 spec 文件并返回退出码
# --------------------------------------------------------------------------
def test_cli_main_returns_exit_code(tmp_path, monkeypatch):
    spec = _spec(tmp_path, fake=True,
                 gt_plan=[{"tool": "search_flights",
                           "args": {"date": "2026-10-12", "from": "PEK",
                                    "to": "SHA"}}],
                 gt_answer="查到了。")
    spec_path = tmp_path / "run_spec.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    out_path = tmp_path / "out" / "trace.jsonl"

    rc = react.main(["--spec", str(spec_path), "--out", str(out_path)])
    assert rc == 0
    assert protocol.load_trace(out_path) is not None
