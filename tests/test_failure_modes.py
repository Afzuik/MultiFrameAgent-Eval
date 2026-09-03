"""test_failure_modes —— analysis.failure_modes（§10.1 失败模式分类）单元测试。

覆盖三块：
1. classify_run 六类各自命中 + 优先级：
   - passed=True → None（成功运行不归类）；
   - findings 同时含 forbidden_call 与 missing_call → permission_violation；
   - observation 报 ok:false + code ∈ {forbidden, auth_required} →
     permission_violation（且优先于 parse / budget）；
   - observation 以"解析失败"开头 → parse_failure（优先于 budget）；
   - status ∈ {timeout, budget_exceeded} → budget_exhausted；
   - missing_call:{tool} 且该 tool 在真实调用中 / observation 参数类错误码
     → arg_error；
   - compute_f1 precision<1 → tool_selection_error；
   - 其余失败运行 → planning_failure（兜底，含零调用轨迹）；
2. build_distribution：tmp_path 造 run_dir（results.csv 3 行失败 + 1 行
   通过 + 对应 trace）断言 total/failed/by_category/by_category_pct/
   by_framework/details；
3. CLI（python -m analysis.failure_modes）：打印 JSON 并写
   failure_modes.json。
"""

import csv
import json

import pytest

from analysis.failure_modes import (
    CATEGORIES,
    CATEGORY_LABELS,
    build_distribution,
    classify_run,
)
from analysis.failure_modes import (
    main as failure_modes_main,
)
from harness.protocol import Step, Trace, write_trace
from metrics.success import load_task

CSV_COLUMNS = [
    "task_id",
    "group",
    "framework",
    "model",
    "difficulty",
    "status",
    "passed",
    "findings",
    "cost_usd",
    "wall_time_s",
    "n_tool_calls",
    "n_steps",
]


# ---------- Trace / Step 构造工厂 ----------


def _msg(content: str, role: str = "assistant") -> Step:
    return Step(type="message", role=role, content=content)


def _tool_call(tool: str, args: dict) -> Step:
    return Step(type="tool_call", role="assistant", tool_name=tool, tool_args=args)


def _obs(content: str) -> Step:
    return Step(type="observation", role="tool", content=content)


def _error_obs(code: str, error: str = "工具调用被拒绝") -> Step:
    """工具服务错误响应的 observation（{ok:false, code, error} 序列化形态）。"""
    return _obs(json.dumps({"ok": False, "error": error, "code": code}, ensure_ascii=False))


def _make_trace(task_id: str, steps: list[Step]) -> Trace:
    return Trace(
        run_id=f"r_{task_id}",
        task_id=task_id,
        framework="react",
        model="deepseek-v4-flash",
        steps=steps,
    )


def _empty_trace(task_id: str) -> Trace:
    return _make_trace(task_id, [])


# ---------- ① passed=True → None ----------


def test_classify_passed_run_returns_none():
    """成功运行即使 findings 含 forbidden_call 也不归类。"""
    task = load_task("tr_004")
    trace = _empty_trace("tr_004")
    assert classify_run(task, trace, {}, True, ["forbidden_call:book_flight"], "completed") is None


# ---------- ② permission_violation（findings / observation 双通道 + 优先级） ----------


def test_permission_wins_when_findings_mix_forbidden_and_missing():
    """findings 同时含 forbidden_call 与 missing_call → permission_violation
    （规则 1 优先于规则 4 arg_error）。"""
    task = load_task("tr_004")
    trace = _empty_trace("tr_004")
    cat = classify_run(
        task,
        trace,
        {},
        False,
        ["forbidden_call:book_flight", "missing_call:search_flights"],
        "completed",
    )
    assert cat == "permission_violation"


@pytest.mark.parametrize("code", ["forbidden", "auth_required"])
def test_permission_from_observation_error_code(code):
    """observation 报 ok:false + code ∈ {forbidden, auth_required} →
    permission_violation（无需 findings）。"""
    task = load_task("tr_004")
    trace = _make_trace("tr_004", [_msg("尝试退款"), _error_obs(code)])
    assert classify_run(task, trace, {}, False, [], "completed") == "permission_violation"


def test_permission_priority_over_parse_and_budget():
    """同一轨迹里既有解析失败 observation 又报权限错误、且 status 超时 →
    规则 1 permission_violation 最先命中。"""
    task = load_task("tr_004")
    trace = _make_trace(
        "tr_004",
        [
            _obs("解析失败: 模型输出不是合法的 JSON 对象"),
            _error_obs("forbidden"),
        ],
    )
    assert classify_run(task, trace, {}, False, [], "timeout") == "permission_violation"


# ---------- ③ parse_failure（含对 budget 的优先级） ----------


def test_parse_failure_from_observation_prefix():
    """observation 以"解析失败"开头（react 适配器口径）→ parse_failure。"""
    task = load_task("tr_001")
    trace = _make_trace(
        "tr_001",
        [
            _msg("这不是合法 JSON", role="assistant"),
            _obs("解析失败: 模型输出不是合法的 JSON 对象"),
        ],
    )
    assert classify_run(task, trace, {}, False, [], "completed") == "parse_failure"


def test_parse_failure_priority_over_budget():
    """status=timeout 但轨迹含解析失败 → 规则 2 parse_failure 先于
    规则 3 budget_exhausted 命中。"""
    task = load_task("tr_001")
    trace = _make_trace(
        "tr_001",
        [
            _obs("解析失败: 输出缺少 tool/final_answer 键或字段类型非法"),
        ],
    )
    assert classify_run(task, trace, {}, False, [], "timeout") == "parse_failure"


# ---------- ④ budget_exhausted ----------


@pytest.mark.parametrize("status", ["timeout", "budget_exceeded"])
def test_budget_from_status(status):
    """status ∈ {timeout, budget_exceeded} → budget_exhausted。"""
    task = load_task("tr_001")
    trace = _make_trace("tr_001", [_msg("最后一句输出后超预算")])
    assert classify_run(task, trace, {}, False, [], status) == "budget_exhausted"


# ---------- ⑤ arg_error（missing_call + 真实调用 / observation 错误码） ----------


def test_arg_error_when_missing_call_tool_appeared_in_calls():
    """tr_005：search 正确、book 订错航班（MU5105 而非 MU5107）→ findings
    含 missing_call:book_flight 且 book_flight 出现在真实调用 → arg_error。"""
    task = load_task("tr_005")
    trace = _make_trace(
        "tr_005",
        [
            _msg("先查航班再订票"),
            _tool_call("search_flights", {"date": "2026-10-12", "from": "PEK", "to": "SHA"}),
            _obs('{"ok": true, "result": {"flights": []}}'),
            _tool_call(
                "book_flight", {"user_id": "u_42", "flight_id": "MU5105", "class": "economy"}
            ),
            _obs('{"ok": true, "result": {"status": "booked"}}'),
        ],
    )
    assert (
        classify_run(task, trace, {}, False, ["missing_call:book_flight"], "completed")
        == "arg_error"
    )


def test_arg_error_from_observation_code():
    """observation 报参数类错误码（invalid_date）→ arg_error（无需 findings）。"""
    task = load_task("tr_001")
    trace = _make_trace(
        "tr_001",
        [
            _tool_call("search_flights", {"date": "2026/10/12", "from": "PEK", "to": "SHA"}),
            _error_obs("invalid_date", "日期必须为 ISO 格式 YYYY-MM-DD"),
        ],
    )
    assert classify_run(task, trace, {}, False, [], "completed") == "arg_error"


# ---------- ⑥ tool_selection_error（compute_f1 precision<1） ----------


def test_tool_selection_error_when_out_of_domain_call():
    """真实调用含域外工具 search_hotels → compute_f1 precision<1 →
    tool_selection_error（无 forbidden/parse/budget/arg 信号时）。"""
    task = load_task("tr_004")
    trace = _make_trace(
        "tr_004",
        [
            _tool_call("search_flights", {"date": "2026-10-12", "from": "PEK", "to": "SHA"}),
            _tool_call(
                "book_flight", {"user_id": "u_42", "flight_id": "MU5101", "class": "economy"}
            ),
            _tool_call("search_hotels", {"city": "上海"}),
        ],
    )
    assert classify_run(task, trace, {}, False, [], "completed") == "tool_selection_error"


# ---------- ⑦ planning_failure（兜底） ----------


def test_planning_failure_fallback():
    """关键调用都在且都合法，但运行仍失败（如终态/答案不符）→ 无更
    具体信号可命中 → planning_failure 兜底。"""
    task = load_task("tr_001")
    trace = _make_trace(
        "tr_001",
        [
            _tool_call("search_flights", {"date": "2026-10-12", "from": "PEK", "to": "SHA"}),
            _obs('{"ok": true, "result": {"flights": []}}'),
            _msg("我已经查过了。"),
        ],
    )
    assert (
        classify_run(task, trace, {}, False, ["answer_missing:MU5101"], "completed")
        == "planning_failure"
    )


def test_planning_failure_for_zero_call_trace():
    """一次工具调用都没发生 → 非 permission/parse/budget/arg；空调用
    列表下 compute_f1 precision=1.0（见 tool_f1 边界约定）→ 不误判为
    tool_selection_error，落入 planning_failure（缺必要步骤）。"""
    task = load_task("tr_001")
    trace = _make_trace("tr_001", [_msg("我直接回答，不调工具了。")])
    assert (
        classify_run(task, trace, {}, False, ["missing_call:search_flights"], "completed")
        == "planning_failure"
    )


# ---------- 常量完整性 ----------


def test_category_labels_cover_categories():
    """CATEGORY_LABELS 与 CATEGORIES 一一对应且齐全。"""
    assert set(CATEGORIES) == set(CATEGORY_LABELS)
    assert CATEGORY_LABELS["permission_violation"] == "权限/规则违反"
    assert CATEGORY_LABELS["planning_failure"] == "规划失败"


# ---------- build_distribution ----------


def _write_results_csv(run_dir) -> None:
    """写 4 行 results.csv：tr_004/tr_005/tr_001 失败，tr_002 通过。"""
    rows = [
        # 失败 1：findings 触发 permission_violation（react）
        {
            "task_id": "tr_004",
            "group": "R1",
            "framework": "react",
            "model": "deepseek-v4-flash",
            "difficulty": "L2",
            "status": "completed",
            "passed": "False",
            "findings": "forbidden_call:book_flight",
            "cost_usd": "0.001",
            "wall_time_s": "3.0",
            "n_tool_calls": "1",
            "n_steps": "2",
        },
        # 失败 2：status=timeout → budget_exhausted（smolagents）
        {
            "task_id": "tr_005",
            "group": "S1",
            "framework": "smolagents",
            "model": "deepseek-v4-flash",
            "difficulty": "L2",
            "status": "timeout",
            "passed": "False",
            "findings": "",
            "cost_usd": "0.01",
            "wall_time_s": "30.0",
            "n_tool_calls": "1",
            "n_steps": "5",
        },
        # 失败 3：passed="0"，轨迹含解析失败 observation → parse_failure
        {
            "task_id": "tr_001",
            "group": "R1",
            "framework": "react",
            "model": "deepseek-v4-flash",
            "difficulty": "L1",
            "status": "completed",
            "passed": "0",
            "findings": "missing_call:search_flights",
            "cost_usd": "0.0",
            "wall_time_s": "2.0",
            "n_tool_calls": "0",
            "n_steps": "2",
        },
        # 通过行：不入分布（也不用写 trace）
        {
            "task_id": "tr_002",
            "group": "R1",
            "framework": "react",
            "model": "deepseek-v4-flash",
            "difficulty": "L1",
            "status": "completed",
            "passed": "True",
            "findings": "",
            "cost_usd": "0.0",
            "wall_time_s": "0.5",
            "n_tool_calls": "1",
            "n_steps": "1",
        },
    ]
    with (run_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_traces(run_dir) -> None:
    """写 3 条失败轨迹（tr_002 通过行不分类，无需轨迹）。"""
    traces = [
        # tr_004：真调了被禁航班 MU5105 → 与 forbidden_call finding 一致
        _make_trace(
            "tr_004",
            [
                _msg("订便宜的那班"),
                _tool_call(
                    "book_flight", {"user_id": "u_42", "flight_id": "MU5105", "class": "economy"}
                ),
                _obs(
                    '{"ok": true, "result": {"reservation_id": "R_MU5105_u_42", '
                    '"status": "booked"}}'
                ),
            ],
        ),
        # tr_005：超时前最后一次模型输出
        _make_trace(
            "tr_005",
            [
                _msg("再想一下怎么订"),
                _tool_call("search_flights", {"date": "2026-10-12", "from": "PEK", "to": "SHA"}),
                _obs('{"ok": true, "result": {"flights": []}}'),
            ],
        ),
        # tr_001：模型输出无法解析 → 解析失败 observation
        _make_trace(
            "tr_001",
            [
                _msg("我要查航班……（输出不是 JSON）"),
                _obs("解析失败: 模型输出不是合法的 JSON 对象"),
            ],
        ),
    ]
    for trace in traces:
        write_trace(trace, run_dir / "traces" / f"{trace.task_id}.jsonl")


def _make_run_dir(tmp_path):
    run_dir = tmp_path / "run_dir"
    _write_traces(run_dir)
    _write_results_csv(run_dir)
    return run_dir


def test_build_distribution(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    dist = build_distribution(run_dir)

    assert dist["total"] == 4
    assert dist["failed"] == 3
    assert dist["by_category"] == {
        "permission_violation": 1,
        "parse_failure": 1,
        "budget_exhausted": 1,
        "tool_selection_error": 0,
        "arg_error": 0,
        "planning_failure": 0,
    }
    for category in ("permission_violation", "parse_failure", "budget_exhausted"):
        assert dist["by_category_pct"][category] == pytest.approx(33.33)
    assert dist["by_category_pct"]["tool_selection_error"] == 0.0

    # 分框架：react 2 个失败（permission + parse），smolagents 1 个（budget）
    assert dist["by_framework"]["react"]["failed"] == 2
    assert dist["by_framework"]["react"]["by_category"]["permission_violation"] == 1
    assert dist["by_framework"]["react"]["by_category"]["parse_failure"] == 1
    assert dist["by_framework"]["react"]["by_category"]["budget_exhausted"] == 0
    assert dist["by_framework"]["smolagents"]["failed"] == 1
    assert dist["by_framework"]["smolagents"]["by_category"]["budget_exhausted"] == 1

    # 明细：每行含 task_id/framework/difficulty/category/status/findings
    by_id = {d["task_id"]: d for d in dist["details"]}
    assert set(by_id) == {"tr_004", "tr_005", "tr_001"}
    assert by_id["tr_004"]["category"] == "permission_violation"
    assert by_id["tr_004"]["framework"] == "react"
    assert by_id["tr_004"]["difficulty"] == "L2"
    assert by_id["tr_004"]["status"] == "completed"
    assert by_id["tr_004"]["findings"] == ["forbidden_call:book_flight"]
    assert by_id["tr_005"]["category"] == "budget_exhausted"
    assert by_id["tr_005"]["framework"] == "smolagents"
    assert by_id["tr_001"]["category"] == "parse_failure"
    assert by_id["tr_001"]["findings"] == ["missing_call:search_flights"]


def test_failure_modes_cli_writes_json(tmp_path, capsys):
    run_dir = _make_run_dir(tmp_path)
    assert failure_modes_main([str(run_dir)]) == 0

    captured = capsys.readouterr()
    assert '"failed": 3' in captured.out
    assert '"by_category"' in captured.out

    out_path = run_dir / "failure_modes.json"
    assert out_path.is_file()
    assert json.loads(out_path.read_text(encoding="utf-8")) == build_distribution(run_dir)
