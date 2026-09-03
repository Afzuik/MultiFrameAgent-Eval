"""test_tool_f1 —— metrics.tool_f1（§7.2 Tool-Call F1 口径）单元测试。

用真实任务（metrics.success.load_task）构造 Trace，覆盖六个场景：
① GT 路径全对 → recall/precision/f1 全 1.0；
② GT 路径上多调一次同域工具（参数不全）→ 只压 precision、recall 不动；
   另补一例"参数完整的同域冗余调用"→ 不扣分（等价工具集公平性）；
③ 调用域外工具（伪造 search_hotels）→ precision 扣分且记入 invalid_calls；
④ 缺 required 参数（book_flight 不给 user_id/class）→ 该次调用非法；
⑤ 空调用列表 → recall=0、f1=0（precision 取 1.0 的边界约定）；
⑥ P+R=0（全非法调用且 GT 全未命中）→ f1=0.0（防除零边界）。
另附 load_allowed_tools 的注册表读取测试。
"""

import pytest

from harness.protocol import Step, Trace
from metrics.success import load_task
from metrics.tool_f1 import compute_f1, load_allowed_tools

# ---------- Trace 构造工厂 ----------


def _tool_call_steps(tool_calls: list[tuple[str, dict]]) -> list[Step]:
    """把 [(tool, args), ...] 转成连续 tool_call 步骤。"""
    return [
        Step(
            type="tool_call",
            role="assistant",
            tool_name=tool,
            tool_args=dict(args),
            ts=float(i),
        )
        for i, (tool, args) in enumerate(tool_calls, start=1)
    ]


def _trace(task_id: str, tool_calls: list[tuple[str, dict]]) -> Trace:
    """按 task_id 造一条只含工具调用步骤的轨迹。"""
    return Trace(
        run_id=f"r_{task_id}",
        task_id=task_id,
        framework="react",
        model="deepseek-v4-flash",
        steps=_tool_call_steps(tool_calls),
    )


def _gt_tool_calls(task_id: str) -> list[tuple[str, dict]]:
    """取真实任务的 ground_truth_plan 作为 (tool, args) 列表。"""
    task = load_task(task_id)
    return [(s["tool"], s.get("args") or {}) for s in task["ground_truth_plan"]]


# ---------- load_allowed_tools（Precision 允许工具集的静态来源） ----------


def test_load_allowed_tools_per_domain():
    """三域的域内工具名集合应与注册表一致（travel 7/shop 6/analytics 5）。"""
    assert load_allowed_tools("travel") == {
        "search_flights",
        "get_user_profile",
        "list_reservations",
        "verify_identity",
        "book_flight",
        "cancel_reservation",
        "refund_reservation",
    }
    assert load_allowed_tools("shop") == {
        "get_order_status",
        "list_orders",
        "update_address",
        "verify_identity",
        "apply_refund",
        "create_invoice",
    }
    assert load_allowed_tools("analytics") == {
        "get_schema",
        "run_query",
        "export_report",
        "get_report",
        "list_reports",
    }
    assert load_allowed_tools("no_such_domain") == set()


# ---------- ① GT 路径：recall / precision / f1 全 1.0 ----------


def test_f1_gt_path_full_score():
    """tr_004 按 GT 计划重放 → recall=1, precision=1, f1=1。"""
    task = load_task("tr_004")
    trace = _trace("tr_004", _gt_tool_calls("tr_004"))
    res = compute_f1(task, trace)

    assert res["recall"] == pytest.approx(1.0)
    assert res["precision"] == pytest.approx(1.0)
    assert res["f1"] == pytest.approx(1.0)
    assert res["gt_calls"] == 2  # search_flights + book_flight
    assert res["matched_gt"] == 2
    assert res["n_real_calls"] == 2
    assert res["n_valid_calls"] == 2
    assert res["invalid_calls"] == []


# ---------- ② 多调一次同域工具：只压 precision、不动 recall ----------


def test_f1_extra_same_domain_call_reduces_precision_only():
    """GT 路径上多调一次同域 get_user_profile（漏带必填 user_id）→
    precision=2/3，recall 保持 1.0，该冗余调用记入 invalid_calls。"""
    task = load_task("tr_004")
    calls = _gt_tool_calls("tr_004") + [("get_user_profile", {})]
    res = compute_f1(task, _trace("tr_004", calls))

    assert res["recall"] == pytest.approx(1.0)
    assert res["precision"] == pytest.approx(2 / 3)
    assert res["f1"] == pytest.approx(0.8)
    assert res["n_real_calls"] == 3
    assert res["n_valid_calls"] == 2
    assert res["invalid_calls"] == ["get_user_profile"]


def test_f1_extra_valid_same_domain_call_not_penalized():
    """多调一次同域工具且参数完整 → 不扣分（§7.2 说明：Precision 允许
    等价工具集，冗余但合法的调用不冤枉）。"""
    task = load_task("tr_004")
    calls = _gt_tool_calls("tr_004") + [("get_user_profile", {"user_id": "u_42"})]
    res = compute_f1(task, _trace("tr_004", calls))

    assert res["recall"] == pytest.approx(1.0)
    assert res["precision"] == pytest.approx(1.0)
    assert res["f1"] == pytest.approx(1.0)
    assert res["invalid_calls"] == []


# ---------- ③ 调用域外工具：precision 扣分 + invalid_calls 记录 ----------


def test_f1_out_of_domain_tool_invalid():
    """伪造域外工具 search_hotels → precision=2/3 且记入 invalid_calls。"""
    task = load_task("tr_004")
    calls = _gt_tool_calls("tr_004") + [("search_hotels", {"city": "上海"})]
    res = compute_f1(task, _trace("tr_004", calls))

    assert res["recall"] == pytest.approx(1.0)
    assert res["precision"] == pytest.approx(2 / 3)
    assert res["n_valid_calls"] == 2
    assert res["invalid_calls"] == ["search_hotels"]


# ---------- ④ 缺 required 参数：book_flight 只给 flight_id ----------


def test_f1_missing_required_param_invalid():
    """book_flight 只给 flight_id、不给 user_id/class → 该调用非法；
    GT 的 book_flight（参数包含匹配）也未命中 → recall/precision 双降。"""
    task = load_task("tr_004")
    search, _ = _gt_tool_calls("tr_004")[0]
    calls = [
        (search, {"date": "2026-10-12", "from": "PEK", "to": "SHA"}),
        ("book_flight", {"flight_id": "MU5101"}),
    ]
    res = compute_f1(task, _trace("tr_004", calls))

    assert res["recall"] == pytest.approx(0.5)  # 只命中 search_flights
    assert res["precision"] == pytest.approx(0.5)  # search 合法、book 非法
    assert res["f1"] == pytest.approx(0.5)
    assert res["matched_gt"] == 1
    assert res["n_valid_calls"] == 1
    assert res["invalid_calls"] == ["book_flight"]


# ---------- ⑤ 空调用列表：recall=0，f1=0 ----------


def test_f1_empty_calls_recall_zero():
    """一次真实调用都没有 → recall=0、f1=0；precision 按边界约定取 1.0
    （没有真实调用就没有非法调用可扣）。"""
    task = load_task("tr_004")
    res = compute_f1(task, _trace("tr_004", []))

    assert res["recall"] == pytest.approx(0.0)
    assert res["precision"] == pytest.approx(1.0)  # 见 tool_f1 模块 docstring
    assert res["f1"] == pytest.approx(0.0)
    assert res["gt_calls"] == 2
    assert res["matched_gt"] == 0
    assert res["n_real_calls"] == 0
    assert res["n_valid_calls"] == 0
    assert res["invalid_calls"] == []


# ---------- ⑥ P+R=0 边界：全非法调用且 GT 全未命中 ----------


def test_f1_zero_denominator_edge():
    """全部真实调用都非法、GT 关键调用一个未命中 → P=R=0，
    F1 公式分母为 0，按口径返回 0.0（防除零）。"""
    task = load_task("tr_004")
    calls = [("search_hotels", {}), ("book_flight", {"flight_id": "MU5101"})]  # 缺 user_id/class
    res = compute_f1(task, _trace("tr_004", calls))

    assert res["recall"] == pytest.approx(0.0)
    assert res["precision"] == pytest.approx(0.0)
    assert res["f1"] == pytest.approx(0.0)
    assert res["n_valid_calls"] == 0
    assert len(res["invalid_calls"]) == 2


# ---------- 补充：以 GT 关键调用为单位计数，重复调用不重复命中 ----------


def test_f1_gt_step_counted_once_even_if_called_multiple_times():
    """同一个 GT 关键调用重复出现多次 → 只算命中一次（matched_gt 按
    GT 步骤计，不按真实调用计）。"""
    task = load_task("tr_004")
    search, book = _gt_tool_calls("tr_004")
    calls = [search, search, book]  # search_flights 重复调了两次
    res = compute_f1(task, _trace("tr_004", calls))

    assert res["matched_gt"] == 2
    assert res["recall"] == pytest.approx(1.0)
    assert res["n_real_calls"] == 3
    assert res["precision"] == pytest.approx(1.0)  # 全部调用都合法
