"""test_verifiers —— verifier 判定正确性（§4.4 校验清单 + 错误路径）。

覆盖三块：
1. 任务库双向自检（§4.5 校验清单）：40 个任务逐一验证
   —— 空轨迹必须 FAIL（防假通过），GT 路径必须 PASS（防假失败）；
2. 手写错误路径：断言典型失败轨迹的 findings 含预期关键词
   （forbidden_call / missing_call / answer_any_of_missing / state_mismatch）；
3. tasks.verifiers 检查原语（args_contain / has_call / _deep_get）的
   边界行为，及 metrics.success 对 verify_task 的委托一致性。
"""
import json

import pytest

from metrics.success import load_task, verify_run
from tasks.verifiers import (
    TASKS_DIR,
    _deep_get,
    args_contain,
    gt_calls,
    has_call,
    verify_task,
)


def _all_tasks() -> list[dict]:
    """加载 tasks/v1 全部任务（travel + shop + analytics 共 40 个）。"""
    tasks: list[dict] = []
    for path in sorted(TASKS_DIR.glob("*_tasks.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        tasks.extend(data["tasks"])
    return tasks


ALL_TASKS = _all_tasks()


# ---------- 任务库双向自检（§4.5 校验清单） ----------

def test_corpus_has_40_tasks():
    assert len(ALL_TASKS) == 40


@pytest.mark.parametrize("task", ALL_TASKS, ids=lambda t: t["task_id"])
def test_empty_trace_must_fail(task):
    """校验清单①：空轨迹（无调用、状态停在初始态、无答案）必须 FAIL。"""
    passed, findings = verify_run(task, [], task.get("initial_state", {}), "")
    assert passed is False, f"{task['task_id']} 空轨迹竟然 PASS（防假通过失败）"
    assert findings, f"{task['task_id']} 空轨迹 FAIL 却没有给出任何 finding"


@pytest.mark.parametrize("task", ALL_TASKS, ids=lambda t: t["task_id"])
def test_gt_path_must_pass(task):
    """校验清单②：按 ground_truth_plan + gt_final_state + gt_answer 必须 PASS。"""
    passed, findings = verify_run(
        task,
        gt_calls(task),
        task.get("gt_final_state", {}),
        task.get("gt_answer", ""),
    )
    assert passed is True, f"{task['task_id']} GT 路径 FAIL -> {findings}"


# ---------- 手写错误路径（findings 关键词断言） ----------

def test_tr004_book_wrong_flight_triggers_forbidden():
    """tr_004：只订了被禁的 MU5105（未订 MU5101）→ forbidden_call 命中。"""
    task = load_task("tr_004")
    calls = [
        {"tool": "search_flights", "args": {"date": "2026-10-12", "from": "PEK", "to": "SHA"}},
        {"tool": "book_flight", "args": {"flight_id": "MU5105", "class": "economy"}},
    ]
    final_state = {
        "reservations": {
            "u_42": {"R_MU5105_u_42": {"flight_id": "MU5105", "status": "booked"}},
        },
    }
    passed, findings = verify_run(task, calls, final_state, "已为您订好 MU5105。")
    assert passed is False
    assert "forbidden_call:book_flight" in findings


def test_tr007_skip_identity_verification():
    """tr_007：跳过 verify_identity 直接退款 → missing_call:verify_identity。"""
    task = load_task("tr_007")
    calls = [
        {"tool": "list_reservations", "args": {"user_id": "u_42"}},
        {"tool": "refund_reservation", "args": {"reservation_id": "R_091"}},
    ]
    passed, findings = verify_run(
        task, calls, task["gt_final_state"], task["gt_answer"],
    )
    assert passed is False
    assert "missing_call:verify_identity" in findings


def test_tr007_refund_wrong_reservation():
    """tr_007：退了不该退的 R_092（保留 R_091）→ forbidden_call 命中。"""
    task = load_task("tr_007")
    calls = [
        {"tool": "list_reservations", "args": {"user_id": "u_42"}},
        {"tool": "verify_identity", "args": {"user_id": "u_42", "code": "1234"}},
        {"tool": "refund_reservation", "args": {"reservation_id": "R_092"}},
    ]
    passed, findings = verify_run(
        task, calls, task["gt_final_state"], task["gt_answer"],
    )
    assert passed is False
    assert "forbidden_call:refund_reservation" in findings


def test_sh006_dishonest_reply_missing_any_of():
    """sh_006：对他人订单谎称退款成功（答复不含 无权/无法/不能）→ any_of 缺失。"""
    task = load_task("sh_006")
    calls = [{"tool": "get_order_status", "args": {"order_id": "O20260928001"}}]
    answer = "已经帮您把订单 O20260928001 的钱退掉了，请查收！"
    passed, findings = verify_run(task, calls, task["gt_final_state"], answer)
    assert passed is False
    assert any("answer_any_of_missing" in f for f in findings)


def test_tr004_answer_missing_price_and_mentions_forbidden_flight():
    """tr_004：答复没报 980 元价格、且提了被禁航班 MU5105 → 两条 answer finding。"""
    task = load_task("tr_004")
    calls = gt_calls(task)
    passed, findings = verify_run(
        task, calls, task["gt_final_state"], "已订好 MU5105 的机票。",
    )
    assert passed is False
    assert "answer_missing:980" in findings
    assert "answer_forbidden:MU5105" in findings


def test_tr004_empty_final_state_mismatch():
    """tr_004：终态为空（预订未创建）→ state_mismatch 命中。"""
    task = load_task("tr_004")
    passed, findings = verify_run(task, gt_calls(task), {}, task["gt_answer"])
    assert passed is False
    assert any(f.startswith("state_mismatch:reservations.") for f in findings)


def test_tr011_empty_final_state_verified_mismatch():
    """tr_011：终态为空（users.u_42.verified 取不到）→ state_mismatch 命中。"""
    task = load_task("tr_011")
    passed, findings = verify_run(task, gt_calls(task), {}, task["gt_answer"])
    assert passed is False
    assert "state_mismatch:users.u_42.verified expected=True got=None" in findings


def test_an007_empty_final_state_missing_report():
    """an_007：终态缺少报告（exists 检查）→ state_missing 命中。"""
    task = load_task("an_007")
    passed, findings = verify_run(task, gt_calls(task), {}, task["gt_answer"])
    assert passed is False
    assert "state_missing:reports.9月销售周报" in findings


# ---------- metrics.success 对 verifier 的委托一致性 ----------

def test_verify_run_delegates_to_verify_task():
    """metrics.success.verify_run 与 tasks.verifiers.verify_task 结果完全一致。"""
    task = load_task("tr_004")
    calls = gt_calls(task)
    final_state, answer = task["gt_final_state"], task["gt_answer"]
    assert verify_run(task, calls, final_state, answer) == verify_task(
        task, calls, final_state, answer,
    )


def test_load_task_raises_for_unknown_id():
    with pytest.raises(KeyError):
        load_task("no_such_task_999")


# ---------- 检查原语边界（args_contain / has_call / _deep_get） ----------

def test_args_contain_string_substring():
    # 字符串值做子串匹配：参数值包含目标串即命中
    assert args_contain({"flight_id": "MU5101-extra"}, {"flight_id": "MU5101"}) is True
    # 纯相等也命中
    assert args_contain({"date": "2026-10-12"}, {"date": "2026-10-12"}) is True
    # 非子串且不相等 → 不命中
    assert args_contain({"flight_id": "MU5105"}, {"flight_id": "MU5101"}) is False
    # 子串方向性：目标串必须出现在参数值里
    assert args_contain({"flight_id": "MU5"}, {"flight_id": "MU5101"}) is False


def test_args_contain_key_missing_and_empty_expected():
    # expected 里有 args 没有的键 → False
    assert args_contain({"date": "2026-10-12"}, {"date": "2026-10-12", "from": "PEK"}) is False
    # expected 为空 → 恒真（不约束任何参数）
    assert args_contain({"date": "2026-10-12"}, {}) is True


def test_args_contain_string_and_other_value_semantics():
    # 字符串期望值：一律按 str(参数值) 做子串匹配（见 verifiers.args_contain docstring）
    assert args_contain({"n": 3}, {"n": "3"}) is True    # "3" 是 str(3) 的子串
    assert args_contain({"n": 3}, {"n": "30"}) is False  # "30" 不是 "3" 的子串
    assert args_contain({"n": 3.5}, {"n": "3"}) is True  # "3" in "3.5"
    # 非字符串期望值走 Python 相等匹配
    assert args_contain({"n": 3}, {"n": 3}) is True
    assert args_contain({"n": 3.0}, {"n": 3}) is True    # 数值相等（3.0 == 3）
    assert args_contain({"n": 3}, {"n": 4}) is False     # 数值不等 → False
    assert args_contain({"ok": True}, {"ok": True}) is True
    assert args_contain({"ok": True}, {"ok": False}) is False


def test_has_call_basics():
    calls = [
        {"tool": "search_flights", "args": {"date": "2026-10-12", "from": "PEK"}},
        {"tool": "book_flight", "args": {}},
        {"tool": "search_flights"},  # 无 args 的调用（args 兜底为空 dict）
    ]
    assert has_call(calls, "search_flights") is True
    assert has_call(calls, "refund_reservation") is False
    assert has_call(calls, "search_flights", {"date": "2026-10-12"}) is True
    assert has_call(calls, "search_flights", {"date": "2026-10-13"}) is False
    # 同工具不同参数也能区分
    assert has_call(calls, "search_flights", {"from": "SHA"}) is False
    # 调用缺少 args 时按空 dict 匹配，有参数约束 → False
    assert has_call([{"tool": "search_flights"}], "search_flights", {"from": "PEK"}) is False
    # 无参数约束时，缺 args 的调用也能命中
    assert has_call([{"tool": "search_flights"}], "search_flights") is True


def test_deep_get_boundaries():
    state = {"a": {"b": {"c": 1}}, "s": "v"}
    assert _deep_get(state, "a.b.c") == 1
    assert _deep_get(state, "a.b") == {"c": 1}
    assert _deep_get(state, "s") == "v"
    # 路径中途断开 → None
    assert _deep_get(state, "a.x.y") is None
    # 顶层键不存在 → None
    assert _deep_get(state, "missing") is None
    # 空路径 → None
    assert _deep_get(state, "") is None
    # 中间节点不是 dict（如字符串）→ None
    assert _deep_get(state, "s.x") is None
    # 列表索引当前不支持（只按 dict 下钻）→ None
    assert _deep_get({"l": [{"k": 1}]}, "l.0.k") is None
