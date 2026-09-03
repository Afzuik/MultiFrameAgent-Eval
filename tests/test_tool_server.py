"""tool_server mock 工具服务的接口与业务规则测试。

覆盖：5 个 HTTP 端点的契约（200/404/422）、18 个工具的成功与拒绝路径、
实例隔离、确定性（同调用两次结果一致）、日志记录、reset 语义。
"""
from __future__ import annotations

import uuid
import warnings
from typing import Any

import pytest
from fastapi.testclient import TestClient

# 屏蔽 starlette testclient 的第三方告警噪音，保持输出可读
warnings.filterwarnings("ignore", category=DeprecationWarning, module="starlette")

from tool_server.app import app

client = TestClient(app)


# ---------------------------------------------------------------- 测试辅助
def reset(domain: str, initial_state: dict[str, Any] | None = None) -> str:
    """重置一个全新实例并返回其 instance_id。"""
    iid = uuid.uuid4().hex
    payload = {"domain": domain}
    if initial_state:
        payload["initial_state"] = initial_state
    resp = client.post(f"/instances/{iid}/reset", json=payload)
    assert resp.status_code == 200, resp.text
    return iid


def tool(
    iid: str, name: str, args: dict[str, Any], caller: str | None = None
) -> dict[str, Any]:
    """调用一次工具，断言 HTTP 200 并返回业务 JSON。"""
    headers = {"X-Caller": caller} if caller else None
    resp = client.post(f"/instances/{iid}/tools/{name}", json=args, headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()


def tool_ok(iid: str, name: str, args: dict[str, Any]) -> Any:
    """调用工具并断言业务成功，返回 result。"""
    body = tool(iid, name, args)
    assert body["ok"] is True, body
    return body["result"]


def tool_rejected(iid: str, name: str, args: dict[str, Any], code: str) -> dict:
    """调用工具并断言业务拒绝（ok=False），返回响应体。"""
    body = tool(iid, name, args)
    assert body["ok"] is False, body
    assert body["code"] == code, body
    assert body["error"], body
    return body


def state_of(iid: str) -> dict[str, Any]:
    """GET /state 并返回状态 dict。"""
    resp = client.get(f"/instances/{iid}/state")
    assert resp.status_code == 200, resp.text
    return resp.json()


def log_of(iid: str) -> list[dict[str, Any]]:
    """GET /log 并返回日志数组。"""
    resp = client.get(f"/instances/{iid}/log")
    assert resp.status_code == 200, resp.text
    return resp.json()


# ================================================================ 健康检查
def test_healthz() -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


# ================================================================ travel 域
def test_search_flights_returns_route_flights_and_filters() -> None:
    iid = reset("travel")
    result = tool_ok(
        iid, "search_flights",
        {"date": "2026-10-12", "from": "PEK", "to": "SHA"},
    )
    assert isinstance(result, list) and len(result) == 4
    ids = [f["flight_id"] for f in result]
    assert ids == ["MU5101", "MU5103", "MU5105", "MU5107"]
    first = result[0]
    assert first["dep_time"] == "08:30" and first["arr_time"] == "10:10"
    assert first["class"] == "economy" and first["price_cents"] == 98000
    assert first["seats_left"] == 5

    economy = tool_ok(
        iid, "search_flights",
        {"date": "2026-10-12", "from": "PEK", "to": "SHA", "class": "economy"},
    )
    assert [f["flight_id"] for f in economy] == ["MU5101", "MU5105", "MU5107"]
    business = tool_ok(
        iid, "search_flights",
        {"date": "2026-10-12", "from": "PEK", "to": "SHA", "class": "business"},
    )
    assert [f["flight_id"] for f in business] == ["MU5103"]


@pytest.mark.parametrize(
    "bad_date",
    ["2026/10/12", "2026-13-01", "2026-1-2", "12-10-2026", "abc", "", 123],
)
def test_search_flights_invalid_date(bad_date: Any) -> None:
    iid = reset("travel")
    tool_rejected(
        iid, "search_flights",
        {"date": bad_date, "from": "PEK", "to": "SHA"},
        "invalid_date",
    )


def test_search_flights_route_not_found() -> None:
    iid = reset("travel")
    # 小写机场代码（非大写三字码）
    tool_rejected(
        iid, "search_flights",
        {"date": "2026-10-12", "from": "pek", "to": "SHA"},
        "route_not_found",
    )
    # 航线不存在
    tool_rejected(
        iid, "search_flights",
        {"date": "2026-10-12", "from": "PEK", "to": "XXX"},
        "route_not_found",
    )


def test_get_user_profile_success_and_missing() -> None:
    iid = reset("travel")
    profile = tool_ok(iid, "get_user_profile", {"user_id": "u_42"})
    assert profile["name"] == "李雷"
    assert profile["vip"] is False and profile["verified"] is False
    tool_rejected(iid, "get_user_profile", {"user_id": "u_999"}, "user_not_found")


def test_list_reservations_default_user() -> None:
    iid = reset("travel")
    items = tool_ok(iid, "list_reservations", {"user_id": "u_42"})
    assert [item["reservation_id"] for item in items] == ["R_091", "R_092"]
    assert all(item["user_id"] == "u_42" for item in items)
    assert all(item["status"] == "booked" for item in items)


def test_verify_identity_travel_success_and_wrong_code() -> None:
    iid = reset("travel")
    result = tool_ok(iid, "verify_identity", {"user_id": "u_42", "code": "1234"})
    assert result == {"verified": True}
    assert state_of(iid)["users"]["u_42"]["verified"] is True

    iid2 = reset("travel")
    tool_rejected(iid2, "verify_identity", {"user_id": "u_42", "code": "9999"},
                  "invalid_code")
    # 错误验证码不得改变状态
    assert state_of(iid2)["users"]["u_42"]["verified"] is False


def test_verify_identity_unknown_user() -> None:
    iid = reset("travel")
    tool_rejected(
        iid, "verify_identity", {"user_id": "u_999", "code": "1234"},
        "user_not_found",
    )


def test_book_flight_success_decrements_and_idempotent() -> None:
    iid = reset("travel")
    record = tool_ok(
        iid, "book_flight",
        {"user_id": "u_42", "flight_id": "MU5101", "class": "economy"},
    )
    assert record["reservation_id"] == "R_MU5101_u_42"
    assert record["status"] == "booked" and record["price_cents"] == 98000
    st = state_of(iid)
    res = st["reservations"]["u_42"]["R_MU5101_u_42"]
    assert res["status"] == "booked" and res["flight_id"] == "MU5101"
    # 座位从 5 减为 4
    seats = {
        f["flight_id"]: f["seats_left"]
        for f in tool_ok(
            iid, "search_flights",
            {"date": "2026-10-12", "from": "PEK", "to": "SHA"},
        )
    }
    assert seats["MU5101"] == 4
    # 幂等：重复预订返回同一记录且不再扣座位
    again = tool_ok(
        iid, "book_flight",
        {"user_id": "u_42", "flight_id": "MU5101", "class": "economy"},
    )
    assert again == record
    seats2 = {
        f["flight_id"]: f["seats_left"]
        for f in tool_ok(
            iid, "search_flights",
            {"date": "2026-10-12", "from": "PEK", "to": "SHA"},
        )
    }
    assert seats2["MU5101"] == 4


def test_book_flight_rejections() -> None:
    iid = reset("travel")
    # 航班不存在
    tool_rejected(
        iid, "book_flight",
        {"user_id": "u_42", "flight_id": "ZZ9999", "class": "economy"},
        "flight_not_found",
    )
    # 舱位与航班不一致（MU5101 为 economy）
    tool_rejected(
        iid, "book_flight",
        {"user_id": "u_42", "flight_id": "MU5101", "class": "business"},
        "class_mismatch",
    )
    # 无剩余座位：MU5101 economy 仅 5 个座位，订满后第 6 次拒绝
    iid2 = reset("travel")
    for user in [f"f_{n}" for n in range(1, 6)]:
        tool_ok(
            iid2, "book_flight",
            {"user_id": user, "flight_id": "MU5101", "class": "economy"},
        )
    tool_rejected(
        iid2, "book_flight",
        {"user_id": "f_6", "flight_id": "MU5101", "class": "economy"},
        "no_seats",
    )


def test_cancel_reservation_success_and_invalid_state() -> None:
    iid = reset("travel")
    record = tool_ok(
        iid, "cancel_reservation",
        {"user_id": "u_42", "reservation_id": "R_091"},
    )
    assert record["status"] == "cancelled"
    assert state_of(iid)["reservations"]["u_42"]["R_091"]["status"] == "cancelled"
    # 已取消的预订不能再取消
    tool_rejected(
        iid, "cancel_reservation",
        {"user_id": "u_42", "reservation_id": "R_091"},
        "invalid_state",
    )


def test_cancel_reservation_not_found_and_forbidden() -> None:
    iid = reset("travel")
    tool_rejected(
        iid, "cancel_reservation",
        {"user_id": "u_42", "reservation_id": "R_XXX"},
        "not_found",
    )
    # u_42 的预订不能由 u_77 取消
    tool_rejected(
        iid, "cancel_reservation",
        {"user_id": "u_77", "reservation_id": "R_091"},
        "forbidden",
    )


def test_refund_reservation_auth_required_and_success() -> None:
    iid = reset("travel")
    # 未验证身份直接退款被拒（u_42 默认未验证）
    tool_rejected(
        iid, "refund_reservation",
        {"user_id": "u_42", "reservation_id": "R_091"},
        "auth_required",
    )
    tool_ok(iid, "verify_identity", {"user_id": "u_42", "code": "1234"})
    result = tool_ok(
        iid, "refund_reservation",
        {"user_id": "u_42", "reservation_id": "R_091"},
    )
    assert result == {"refund_cents": 95000}
    assert state_of(iid)["reservations"]["u_42"]["R_091"]["status"] == "refunded"


def test_refund_reservation_invalid_state_and_forbidden() -> None:
    iid = reset("travel")
    tool_ok(iid, "verify_identity", {"user_id": "u_42", "code": "1234"})
    # 先取消 R_092，使其状态为 cancelled，退款应被拒
    tool_ok(iid, "cancel_reservation",
            {"user_id": "u_42", "reservation_id": "R_092"})
    tool_rejected(
        iid, "refund_reservation",
        {"user_id": "u_42", "reservation_id": "R_092"},
        "invalid_state",
    )
    # 非本人退款（R_091 属于 u_42）被拒
    tool_rejected(
        iid, "refund_reservation",
        {"user_id": "u_77", "reservation_id": "R_091"},
        "forbidden",
    )


# ================================================================ shop 域
def test_shop_get_order_status_success_and_not_found() -> None:
    iid = reset("shop")
    order = tool_ok(iid, "get_order_status", {"order_id": "O20261012001"})
    assert order["order_id"] == "O20261012001"
    assert order["item"] == "显示器" and order["amount_cents"] == 129900
    assert order["status"] == "paid" and order["address"] == "北京市海淀区中关村大街1号"
    assert "tracking" in order and "invoice" in order
    tool_rejected(iid, "get_order_status", {"order_id": "O20269999999"},
                  "not_found")


def test_shop_list_orders_by_user() -> None:
    iid = reset("shop")
    items = tool_ok(iid, "list_orders", {"user_id": "u_42"})
    assert [item["order_id"] for item in items] == [
        "O20261005001", "O20261008003", "O20261012001", "O20261012002",
    ]
    assert all(item["user_id"] == "u_42" for item in items)
    others = tool_ok(iid, "list_orders", {"user_id": "u_77"})
    assert [item["order_id"] for item in others] == ["O20260928001"]


def test_shop_update_address_success() -> None:
    iid = reset("shop", {"user_id": "u_42"})
    order = tool_ok(
        iid, "update_address",
        {"order_id": "O20261012001", "new_address": "上海市浦东新区张江路 88 号"},
    )
    assert order["address"] == "上海市浦东新区张江路 88 号"
    assert state_of(iid)["orders"]["O20261012001"]["address"] == \
        "上海市浦东新区张江路 88 号"


def test_shop_update_address_rejections() -> None:
    iid = reset("shop", {"user_id": "u_42"})
    # 地址过短
    tool_rejected(
        iid, "update_address",
        {"order_id": "O20261012001", "new_address": "太短"},
        "address_too_short",
    )
    # shipped 订单（O20261012002）不可改
    tool_rejected(
        iid, "update_address",
        {"order_id": "O20261012002", "new_address": "上海市浦东新区张江路 88 号"},
        "invalid_state",
    )
    # delivered 订单（O20261005001）不可改
    tool_rejected(
        iid, "update_address",
        {"order_id": "O20261005001", "new_address": "上海市浦东新区张江路 88 号"},
        "invalid_state",
    )
    # 他人订单（u_77 的 O20260928001）越权
    tool_rejected(
        iid, "update_address",
        {"order_id": "O20260928001", "new_address": "上海市浦东新区张江路 88 号"},
        "forbidden",
    )
    # 订单不存在
    tool_rejected(
        iid, "update_address",
        {"order_id": "O20260000000", "new_address": "上海市浦东新区张江路 88 号"},
        "not_found",
    )
    # 所有拒绝路径都不得改变状态
    st = state_of(iid)
    assert st["orders"]["O20261012001"]["address"] == "北京市海淀区中关村大街1号"
    assert st["orders"]["O20261012002"]["status"] == "shipped"


def test_shop_verify_identity_success_and_wrong_code() -> None:
    iid = reset("shop")
    assert tool_ok(iid, "verify_identity",
                   {"user_id": "u_42", "code": "1234"}) == {"verified": True}
    assert state_of(iid)["users"]["u_42"]["verified"] is True

    iid2 = reset("shop")
    tool_rejected(iid2, "verify_identity",
                  {"user_id": "u_42", "code": "0000"}, "invalid_code")
    assert state_of(iid2)["users"]["u_42"]["verified"] is False
    tool_rejected(iid2, "verify_identity",
                  {"user_id": "u_999", "code": "1234"}, "user_not_found")


def test_shop_apply_refund_auth_required_then_success() -> None:
    iid = reset("shop", {"user_id": "u_42"})
    tool_rejected(iid, "apply_refund", {"order_id": "O20261012001"},
                  "auth_required")
    # 状态保持 paid 不变
    assert state_of(iid)["orders"]["O20261012001"]["status"] == "paid"
    tool_ok(iid, "verify_identity", {"user_id": "u_42", "code": "1234"})
    result = tool_ok(
        iid, "apply_refund",
        {"order_id": "O20261012001", "reason": "不想要了"},
    )
    assert result == {"refund_cents": 129900}
    assert state_of(iid)["orders"]["O20261012001"]["status"] == "refunding"


def test_shop_apply_refund_invalid_state() -> None:
    iid = reset("shop", {"user_id": "u_42"})
    tool_ok(iid, "verify_identity", {"user_id": "u_42", "code": "1234"})
    tool_ok(iid, "apply_refund", {"order_id": "O20261012001"})
    # refunding 状态不可再次退款
    tool_rejected(iid, "apply_refund", {"order_id": "O20261012001"},
                  "invalid_state")


def test_shop_apply_refund_forbidden_and_delivered_allowed() -> None:
    iid = reset("shop", {"user_id": "u_42"})
    # 他人订单（u_77 的耳机单）越权，即使本人已验证也不行
    tool_ok(iid, "verify_identity", {"user_id": "u_42", "code": "1234"})
    tool_rejected(iid, "apply_refund", {"order_id": "O20260928001"},
                  "forbidden")
    assert state_of(iid)["orders"]["O20260928001"]["status"] == "paid"
    # delivered 订单允许退款（鼠标单 89 元 = 8900 分）
    result = tool_ok(iid, "apply_refund", {"order_id": "O20261005001"})
    assert result == {"refund_cents": 8900}
    # 订单不存在
    tool_rejected(iid, "apply_refund", {"order_id": "O20260000000"},
                  "not_found")


def test_shop_create_invoice_rules() -> None:
    iid = reset("shop", {"user_id": "u_42"})
    # paid 订单可开
    order = tool_ok(iid, "create_invoice",
                    {"order_id": "O20261012001", "title": "李雷"})
    assert order["invoice"] == {"title": "李雷", "amount_cents": 129900}
    # delivered 订单可开
    order2 = tool_ok(iid, "create_invoice",
                     {"order_id": "O20261005001", "title": "公司"})
    assert order2["invoice"]["amount_cents"] == 8900
    # shipped 订单不可开
    tool_rejected(
        iid, "create_invoice",
        {"order_id": "O20261012002", "title": "公司"},
        "invalid_state",
    )
    # 订单不存在
    tool_rejected(
        iid, "create_invoice",
        {"order_id": "O20260000000", "title": "公司"},
        "not_found",
    )
    st = state_of(iid)
    assert st["orders"]["O20261012002"]["invoice"] is None


# ================================================================ analytics 域
def test_analytics_initial_state() -> None:
    iid = reset("analytics")
    st = state_of(iid)
    assert st["reports"] == {}
    assert st["session"]["user_id"] == "u_42"


def test_get_schema_all_single_and_invalid() -> None:
    iid = reset("analytics")
    schema = tool_ok(iid, "get_schema", {})["schema"]
    assert [item["table"] for item in schema] == ["sales", "products", "regions"]
    by_table = {item["table"]: item["columns"] for item in schema}
    assert by_table["sales"] == [
        "id", "date", "product", "region", "channel", "amount_cents",
    ]
    assert by_table["products"] == ["id", "name", "category"]
    assert by_table["regions"] == ["name", "manager"]
    single = tool_ok(iid, "get_schema", {"table": "sales"})["schema"]
    assert [item["table"] for item in single] == ["sales"]
    tool_rejected(iid, "get_schema", {"table": "nope"}, "invalid_table")


def test_run_query_september_total() -> None:
    iid = reset("analytics")
    result = tool_ok(
        iid, "run_query",
        {"sql": "SELECT SUM(amount_cents) AS total FROM sales"
                " WHERE date >= '2026-09-01' AND date < '2026-10-01'"},
    )
    assert result["columns"] == ["total"]
    assert result["rows"] == [[2168000]]  # 21680 元


def test_run_query_group_by_region() -> None:
    iid = reset("analytics")
    result = tool_ok(
        iid, "run_query",
        {"sql": "SELECT region, SUM(amount_cents) AS total FROM sales"
                " WHERE date >= '2026-09-01' AND date < '2026-10-01'"
                " GROUP BY region"},
    )
    assert result["columns"] == ["region", "total"]
    assert set(map(tuple, result["rows"])) == {
        ("华北", 778600), ("华东", 809700), ("华南", 579700),
    }


def test_run_query_join_products_category() -> None:
    iid = reset("analytics")
    result = tool_ok(
        iid, "run_query",
        {"sql": "SELECT SUM(s.amount_cents) FROM sales s"
                " JOIN products p ON s.product = p.id"
                " WHERE p.category = '数码' AND s.date >= '2026-09-01'"
                " AND s.date < '2026-10-01'"},
    )
    assert result["rows"] == [[2149200]]


def test_run_query_lowercase_and_trailing_semicolon_ok() -> None:
    iid = reset("analytics")
    low = tool_ok(
        iid, "run_query",
        {"sql": "select count(*) from sales where channel = '线上'"},
    )
    assert low["rows"] == [[10]]
    semi = tool_ok(iid, "run_query", {"sql": "SELECT 1;"})
    assert semi["rows"] == [[1]]


@pytest.mark.parametrize(
    "bad_sql",
    [
        ("INSERT INTO sales (id, date, product, region, channel, amount_cents)"
         " VALUES (99, '2026-10-13', 'P1', '华北', '线上', 100)"),
        "UPDATE sales SET amount_cents = 0",
        "DELETE FROM sales",
        "DROP TABLE sales",
        "CREATE TABLE x (id INT)",
        "SELECT 1; SELECT 2",
        "PRAGMA table_info(sales)",
        "ATTACH DATABASE 'x' AS other",
        "SELECT * FROM sqlite_master",
        "SELECT * FROM users",  # 非白名单表
    ],
)
def test_run_query_security_rejections(bad_sql: str) -> None:
    iid = reset("analytics")
    tool_rejected(iid, "run_query", {"sql": bad_sql}, "invalid_sql")


def test_run_query_syntax_error_is_query_error() -> None:
    iid = reset("analytics")
    tool_rejected(iid, "run_query", {"sql": "SELECT FROM sales"}, "query_error")


def test_run_query_row_limit_500() -> None:
    iid = reset("analytics")
    # sales × sales × sales = 15^3 = 3375 行，应截断到 500 行
    result = tool_ok(
        iid, "run_query",
        {"sql": "SELECT a.id FROM sales a, sales b, sales c"},
    )
    assert len(result["rows"]) == 500
    assert len(result["columns"]) == 1


def test_export_get_list_report() -> None:
    iid = reset("analytics")
    result = tool_ok(
        iid, "export_report",
        {"title": "9月销售周报", "content": "总销售额 21680 元"},
    )
    assert result == {"report_id": "9月销售周报"}
    st = state_of(iid)
    assert "9月销售周报" in st["reports"]
    report = st["reports"]["9月销售周报"]
    assert report["title"] == "9月销售周报"
    assert report["content"] == "总销售额 21680 元"
    assert isinstance(report["created_ts"], float)
    got = tool_ok(iid, "get_report", {"report_id": "9月销售周报"})
    assert got == report
    listing = tool_ok(iid, "list_reports", {})
    assert [item["report_id"] for item in listing] == ["9月销售周报"]


def test_get_report_not_found_and_list_empty() -> None:
    iid = reset("analytics")
    tool_rejected(iid, "get_report", {"report_id": "不存在"}, "not_found")
    assert tool_ok(iid, "list_reports", {}) == []


def test_export_report_overwrite_same_title() -> None:
    iid = reset("analytics")
    tool_ok(iid, "export_report", {"title": "周报", "content": "第一版"})
    tool_ok(iid, "export_report", {"title": "周报", "content": "第二版"})
    assert tool_ok(iid, "get_report", {"report_id": "周报"})["content"] == "第二版"


# ================================================================ 隔离/确定性
def test_reset_isolation_between_instances() -> None:
    a, b = reset("travel"), reset("travel")
    # 预订前两边航班一致
    seats_a = {f["flight_id"]: f["seats_left"]
               for f in tool_ok(a, "search_flights",
                                 {"date": "2026-10-12", "from": "PEK",
                                  "to": "SHA"})}
    seats_b = {f["flight_id"]: f["seats_left"]
               for f in tool_ok(b, "search_flights",
                                 {"date": "2026-10-12", "from": "PEK",
                                  "to": "SHA"})}
    assert seats_a == seats_b and seats_a["MU5101"] == 5
    # 实例 A 预订后：A 状态变化、座位减少，B 完全不受影响
    tool_ok(a, "book_flight",
            {"user_id": "u_42", "flight_id": "MU5101", "class": "economy"})
    assert "R_MU5101_u_42" in state_of(a)["reservations"]["u_42"]
    assert "R_MU5101_u_42" not in state_of(b)["reservations"]["u_42"]
    seats_a2 = {f["flight_id"]: f["seats_left"]
                for f in tool_ok(a, "search_flights",
                                  {"date": "2026-10-12", "from": "PEK",
                                   "to": "SHA"})}
    seats_b2 = {f["flight_id"]: f["seats_left"]
                for f in tool_ok(b, "search_flights",
                                  {"date": "2026-10-12", "from": "PEK",
                                   "to": "SHA"})}
    assert seats_a2["MU5101"] == 4
    assert seats_b2["MU5101"] == 5


def test_initial_state_merge_isolation() -> None:
    # A 的 initial_state 把 u_42 置为已验证，B 默认未验证
    a = reset("travel", {"user_id": "u_42",
                         "users": {"u_42": {"name": "李雷", "vip": False,
                                            "verified": True}}})
    b = reset("travel", {"user_id": "u_42"})
    assert state_of(a)["users"]["u_42"]["verified"] is True
    assert state_of(b)["users"]["u_42"]["verified"] is False
    # u_77（种子用户）在 A 中仍保留，未被 initial_state 覆盖删除
    assert state_of(a)["users"]["u_77"]["name"] == "韩梅梅"


def test_reset_same_instance_clears_state_and_log() -> None:
    iid = reset("travel")
    tool_ok(iid, "verify_identity", {"user_id": "u_42", "code": "1234"})
    assert state_of(iid)["users"]["u_42"]["verified"] is True
    assert len(log_of(iid)) == 1
    resp = client.post(
        f"/instances/{iid}/reset",
        json={"domain": "travel", "initial_state": {"user_id": "u_42"}},
    )
    assert resp.status_code == 200
    assert state_of(iid)["users"]["u_42"]["verified"] is False
    assert log_of(iid) == []


def test_repeated_reads_are_deterministic() -> None:
    travel_iid = reset("travel")
    first = tool(iid=travel_iid, name="search_flights",
                 args={"date": "2026-10-12", "from": "PEK", "to": "SHA"})
    second = tool(iid=travel_iid, name="search_flights",
                  args={"date": "2026-10-12", "from": "PEK", "to": "SHA"})
    assert first == second

    analytics_iid = reset("analytics")
    sql = "SELECT SUM(amount_cents) FROM sales"
    q1 = tool(iid=analytics_iid, name="run_query", args={"sql": sql})
    q2 = tool(iid=analytics_iid, name="run_query", args={"sql": sql})
    assert q1 == q2

    # 相同初始状态的不同实例之间也一致
    other_iid = reset("travel")
    assert tool(iid=other_iid, name="search_flights",
                args={"date": "2026-10-12", "from": "PEK", "to": "SHA"}) == first


# ================================================================ 日志
def test_log_records_success_and_rejection() -> None:
    iid = reset("travel")
    tool_ok(iid, "search_flights",
            {"date": "2026-10-12", "from": "PEK", "to": "SHA"})
    tool_rejected(iid, "book_flight",
                  {"user_id": "u_42", "flight_id": "MU5101",
                   "class": "business"}, "class_mismatch")
    log = log_of(iid)
    assert len(log) == 2
    ok_entry = log[0]
    assert ok_entry["tool"] == "search_flights"
    assert ok_entry["args"] == {"date": "2026-10-12", "from": "PEK", "to": "SHA"}
    assert ok_entry["ok"] is True
    assert isinstance(ok_entry["result"], list)
    assert isinstance(ok_entry["ts"], float)
    assert ok_entry["caller"] == "agent"
    bad_entry = log[1]
    assert bad_entry["ok"] is False
    assert bad_entry["code"] == "class_mismatch"
    assert "error" in bad_entry and "result" not in bad_entry
    # 返回的日志是副本，修改它不影响服务端
    log.clear()
    assert len(log_of(iid)) == 2


def test_log_caller_header_recorded() -> None:
    iid = reset("travel")
    tool(iid, "search_flights",
         {"date": "2026-10-12", "from": "PEK", "to": "SHA"},
         caller="react-run-001")
    assert log_of(iid)[0]["caller"] == "react-run-001"


# ================================================================ HTTP 契约
def test_unknown_instance_returns_404() -> None:
    missing = "no-such-instance"
    assert client.get(f"/instances/{missing}/state").status_code == 404
    assert client.get(f"/instances/{missing}/log").status_code == 404
    resp = client.post(
        f"/instances/{missing}/tools/search_flights",
        json={"date": "2026-10-12", "from": "PEK", "to": "SHA"},
    )
    assert resp.status_code == 404


def test_unknown_tool_in_domain_returns_404() -> None:
    iid = reset("travel")
    # travel 实例中不存在 shop 域工具
    resp = client.post(
        f"/instances/{iid}/tools/get_order_status", json={"order_id": "O1"},
    )
    assert resp.status_code == 404
    resp = client.post(
        f"/instances/{iid}/tools/not_a_tool", json={},
    )
    assert resp.status_code == 404


def test_reset_invalid_domain_returns_422() -> None:
    resp = client.post("/instances/x/reset",
                       json={"domain": "space", "initial_state": {}})
    assert resp.status_code == 422
    resp = client.post("/instances/x/reset", json={"initial_state": {}})
    assert resp.status_code == 422


def test_tool_body_not_object_returns_422() -> None:
    iid = reset("travel")
    resp = client.post(
        f"/instances/{iid}/tools/search_flights", json=[1, 2, 3],
    )
    assert resp.status_code == 422


def test_tool_missing_required_param_returns_422() -> None:
    iid = reset("travel")
    # 缺必填参数 date
    resp = client.post(
        f"/instances/{iid}/tools/search_flights",
        json={"from": "PEK", "to": "SHA"},
    )
    assert resp.status_code == 422
    # 缺必填参数 flight_id
    resp = client.post(
        f"/instances/{iid}/tools/book_flight",
        json={"user_id": "u_42", "class": "economy"},
    )
    assert resp.status_code == 422


def test_shop_business_flow_with_refunding_state() -> None:
    """跨工具组合冒烟：验证 → 退款 → delivered 订单开发票等状态相互影响。"""
    iid = reset("shop", {"user_id": "u_42"})
    tool_ok(iid, "verify_identity", {"user_id": "u_42", "code": "1234"})
    tool_ok(iid, "apply_refund", {"order_id": "O20261012002"})  # shipped -> refunding
    st = state_of(iid)
    assert st["orders"]["O20261012002"]["status"] == "refunding"
    assert st["orders"]["O20261012001"]["status"] == "paid"
    assert st["users"]["u_42"]["verified"] is True
