"""test_orchestrator —— 薄编排器测试（可选，纯函数优先 + 一条端到端）。

覆盖：
1. 纯函数：group 过滤、results.csv 断点续跑读取、CSV 追加（含表头）；
2. 端到端 dry-run：本地 fixture 版 travel mock 服务桩 + fake_llm，
   验证 orchestrator.main → 适配器子进程 → verify_run → results.csv 全链路。

端到端把 runs 产物重定向到 tmp_path，不污染仓库 runs/。
"""
from __future__ import annotations

import csv
import json
import socket
import threading
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from harness import orchestrator, protocol

# 与并行开发的 tool_server 同 HTTP 契约的 travel 域本地桩 -----------------
_FIXTURES = json.loads(
    (protocol.REPO_ROOT / "tool_server" / "tool_registry.json").read_text(
        encoding="utf-8")
)["fixtures"]


def _flight_price(flight_id: str) -> int:
    for route in _FIXTURES["flights"].values():
        for flight in route:
            if flight["flight_id"] == flight_id:
                return int(flight["price_cents"])
    return 0


def _travel_tool(state: dict, args: dict, name: str):
    """travel 域确定性工具逻辑（仅覆盖任务集用到的行为与状态流转）。"""
    uid = str(args.get("user_id", ""))
    if name == "search_flights":          # 只读查询，无副作用
        return True, {"result": {"flights": []}}
    if name == "get_user_profile":
        users = state.setdefault("users", {})
        profile = users.get(uid) or _FIXTURES["travel_users"].get(uid, {})
        return True, {"result": {"user_id": uid, **profile}}
    if name == "list_reservations":
        res = state.setdefault("reservations", {}).setdefault(uid, {})
        return True, {"result": {"reservations": res}}
    if name == "verify_identity":
        if args.get("code") != "1234":
            return False, {"error": "验证码错误", "code": "bad_code"}
        users = state.setdefault("users", {})
        profile = users.setdefault(uid, dict(_FIXTURES["travel_users"].get(uid, {})))
        profile["verified"] = True
        return True, {"result": {"verified": True}}
    if name == "book_flight":
        flight_id = args.get("flight_id")
        cls = args.get("class", "economy")
        rid = f"R_{flight_id}_{uid}"
        res = state.setdefault("reservations", {}).setdefault(uid, {})
        if rid in res:
            return False, {"error": "该预订已存在", "code": "duplicate"}
        res[rid] = {"user_id": uid, "flight_id": flight_id, "class": cls,
                    "price_cents": _flight_price(flight_id), "status": "booked"}
        return True, {"result": {"reservation_id": rid, "status": "booked"}}
    if name == "cancel_reservation":
        rid = str(args.get("reservation_id", ""))
        res = state.setdefault("reservations", {}).setdefault(uid, {})
        if rid not in res:
            return False, {"error": "预订不存在", "code": "not_found"}
        res[rid]["status"] = "cancelled"
        return True, {"result": {"reservation_id": rid, "status": "cancelled"}}
    if name == "refund_reservation":
        rid = str(args.get("reservation_id", ""))
        users = state.setdefault("users", {})
        profile = users.get(uid) or _FIXTURES["travel_users"].get(uid, {})
        if not profile.get("verified"):
            return False, {"error": "身份未验证", "code": "auth_required"}
        res = state.setdefault("reservations", {}).setdefault(uid, {})
        if rid not in res:
            return False, {"error": "预订不存在", "code": "not_found"}
        res[rid]["status"] = "refunded"
        return True, {"result": {"reservation_id": rid, "status": "refunded"}}
    return False, {"error": f"未知工具: {name}", "code": "unknown_tool"}


class _TravelHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, handler):
        super().__init__(addr, handler)
        self.states: dict[str, dict] = {}


class _TravelHandler(BaseHTTPRequestHandler):
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
            ok, payload = _travel_tool(
                self.server.states.setdefault(parts[1], {}), self._body(), parts[3])
            return self._json(200, {"ok": ok, **payload})
        return self._json(404, {"ok": False, "error": "not found"})

    def log_message(self, *args) -> None:  # 静默访问日志
        pass


class _TravelServerStub:
    """线程内运行的 travel mock 服务桩（暴露 terminate() 兼容编排器收尾）。"""

    def __init__(self):
        self.httpd: _TravelHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self, port: int) -> _TravelServerStub:
        self.httpd = _TravelHTTPServer(("127.0.0.1", port), _TravelHandler)
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def terminate(self) -> None:
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


# --------------------------------------------------------------------------
# 1) 纯函数
# --------------------------------------------------------------------------
def test_filter_groups():
    matrix = {"runs": [{"group": "R1"}, {"group": "R2"}, {"group": "S1"}]}
    assert [g["group"] for g in orchestrator.filter_groups(matrix, None)] == \
        ["R1", "R2", "S1"]
    assert [g["group"] for g in orchestrator.filter_groups(matrix, "R2")] == ["R2"]
    assert orchestrator.filter_groups(matrix, "ZZZ") == []


def test_results_csv_append_and_resume(tmp_path):
    results_csv = tmp_path / "results.csv"
    orchestrator._append_result(results_csv, {
        "task_id": "tr_001", "group": "R1", "framework": "react",
        "model": "deepseek-v4-flash", "difficulty": "L1",
        "status": "completed", "passed": "True", "findings": "",
        "cost_usd": 0.001, "wall_time_s": 3.2, "n_tool_calls": 1, "n_steps": 1,
    })
    assert results_csv.read_text(encoding="utf-8").splitlines()[0].startswith(
        "task_id,group,framework,model,difficulty,status,passed,findings")
    assert orchestrator._load_existing_task_ids(results_csv) == {"tr_001"}
    # 断点续跑：同 task_id 再追加被视作已存在
    orchestrator._append_result(results_csv, {
        "task_id": "tr_002", "group": "R1", "framework": "react",
        "model": "deepseek-v4-flash", "difficulty": "L1",
        "status": "completed", "passed": "True", "findings": "",
        "cost_usd": 0.001, "wall_time_s": 2.0, "n_tool_calls": 1, "n_steps": 1,
    })
    assert orchestrator._load_existing_task_ids(results_csv) == {"tr_001", "tr_002"}
    with results_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 and all(len(r) == len(orchestrator.RESULT_COLUMNS)
                                  for r in rows)


# --------------------------------------------------------------------------
# 2) 端到端 dry-run（fake_llm 回放 GT，含真实工具状态流转）
# --------------------------------------------------------------------------
def test_dry_run_e2e_r1(tmp_path, monkeypatch):
    """编排器 → 适配器子进程 → mock 桩 → verify_run → results.csv 全链路。"""
    port = _free_port()
    stub = _TravelServerStub().start(port)
    # 产物重定向到 tmp_path，避免污染仓库 runs/
    monkeypatch.setattr(orchestrator, "_runs_root", lambda: tmp_path)
    monkeypatch.setattr(orchestrator, "_start_tool_server", lambda p: stub)
    monkeypatch.setattr(orchestrator, "_stop_tool_server", lambda s: s.terminate())

    # 自含矩阵：e2e 只跑 R1(travel)，与仓库实验矩阵的演进解耦
    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_text(
        "runs:\n"
        "  - group: R1\n"
        "    framework: react\n"
        "    model: deepseek-v4-flash\n"
        "    domains: [travel]\n",
        encoding="utf-8",
    )
    rc = orchestrator.main(["--config", str(matrix_path), "--group", "R1",
                            "--dry-run", "--port", str(port)])
    assert rc == 0

    run_dir = tmp_path / f"{date.today().isoformat()}_R1"
    results_csv = run_dir / "results.csv"
    assert results_csv.is_file()
    with results_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # R1 组 = travel 域全部 14 个任务
    task_ids = {r["task_id"] for r in rows}
    assert len(task_ids) == 14
    assert all(r["passed"] == "True" for r in rows), \
        [r for r in rows if r["passed"] != "True"]
    assert all(r["status"] == "completed" for r in rows)
    assert all(r["group"] == "R1" and r["framework"] == "react" for r in rows)
    # 断点续跑：重跑同一目录应 SKIP（行数不变）
    rc2 = orchestrator.main(["--config", str(matrix_path), "--group", "R1",
                             "--dry-run", "--port", str(port)])
    assert rc2 == 0
    with results_csv.open(newline="", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 14
