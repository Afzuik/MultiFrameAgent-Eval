"""travel 域 7 个工具的业务实现。

规则严格对照 tool_server/tool_registry.json 中 travel 组各工具的 description
实现；业务拒绝一律抛 ToolError(code, 中文原因)，由 app 层统一转成
HTTP 200 + {"ok": false, "error", "code"}。
"""
from __future__ import annotations

import copy
import datetime
import re
from typing import Any

from ..state import ToolError

# 三个域共享同一批工具入口（verify_identity 行为一致），分域分发表见 __init__.py
TOOLS: dict[str, Any] = {}


def _register(func: Any) -> Any:
    """装饰器：把工具函数注册进本模块的 TOOLS 分发表。"""
    TOOLS[func.__name__] = func
    return func


_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_AIRPORT_CODE = re.compile(r"^[A-Z]{3}$")


def _check_iso_date(raw: Any) -> str:
    """校验日期为合法 ISO YYYY-MM-DD，否则抛 invalid_date。"""
    if not isinstance(raw, str) or not _ISO_DATE.fullmatch(raw):
        raise ToolError("invalid_date", "日期必须为 ISO 格式 YYYY-MM-DD，例如 2026-10-12")
    try:
        datetime.date.fromisoformat(raw)
    except ValueError:
        raise ToolError("invalid_date", f"日期 {raw} 不是合法的公历日期") from None
    return raw


def _resolve_reservation(
    ctx: Any, user_id: str, reservation_id: str
) -> dict[str, Any]:
    """定位某人名下的预订记录。

    存在但属于他人 → forbidden；彻底不存在 → not_found。
    """
    reservations = ctx.state["reservations"]
    mine = reservations.get(user_id, {})
    if reservation_id in mine:
        return mine[reservation_id]
    for records in reservations.values():
        if reservation_id in records:
            raise ToolError("forbidden", "只能操作本人的预订记录")
    raise ToolError("not_found", f"预订 {reservation_id} 不存在")


@_register
def search_flights(ctx: Any, args: dict[str, Any]) -> list[dict[str, Any]]:
    """按日期与起降地查询航班；class 可选过滤。date 校验 ISO，航线需存在。"""
    _check_iso_date(args.get("date"))
    from_code, to_code = args.get("from"), args.get("to")
    if not (isinstance(from_code, str) and _AIRPORT_CODE.fullmatch(from_code)) or not (
        isinstance(to_code, str) and _AIRPORT_CODE.fullmatch(to_code)
    ):
        raise ToolError("route_not_found", "机场代码必须为大写三字码，如 PEK")
    route = f"{from_code}-{to_code}"
    if route not in ctx._flights:
        raise ToolError("route_not_found", f"未找到航线 {route} 的航班")
    flights = ctx._flights[route]
    cabin = args.get("class")
    matched = [fl for fl in flights if cabin is None or fl["class"] == cabin]
    # 返回时深拷贝，避免调用方通过返回对象间接修改实例内部航班
    return copy.deepcopy(matched)


@_register
def get_user_profile(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    """查询用户资料（姓名、VIP、身份验证状态）。"""
    user_id = args.get("user_id")
    users = ctx.state.get("users", {})
    if user_id not in users:
        raise ToolError("user_not_found", f"用户 {user_id} 不存在")
    return {"user_id": user_id, **copy.deepcopy(users[user_id])}


@_register
def list_reservations(ctx: Any, args: dict[str, Any]) -> list[dict[str, Any]]:
    """列出某用户的全部预订（dict 转 list，含 reservation_id 键，按号排序）。"""
    user_id = args.get("user_id")
    records = ctx.state.get("reservations", {}).get(user_id, {})
    items = [
        {"reservation_id": rid, **copy.deepcopy(record)}
        for rid, record in records.items()
    ]
    items.sort(key=lambda item: item["reservation_id"])
    return items


@_register
def verify_identity(ctx: Any, args: dict[str, Any]) -> dict[str, bool]:
    """验证用户身份。验证码固定为 1234（演示环境）。"""
    user_id = args.get("user_id")
    users = ctx.state.get("users", {})
    if user_id not in users:
        raise ToolError("user_not_found", f"用户 {user_id} 不存在")
    code = args.get("code")
    if str(code) != "1234":
        raise ToolError("invalid_code", "验证码错误，请重新输入")
    users[user_id]["verified"] = True
    return {"verified": True}


@_register
def book_flight(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    """预订航班并创建预订记录；需航班存在、有座位、舱位与该航班一致。"""
    user_id, flight_id = args.get("user_id"), args.get("flight_id")
    cabin = args.get("class")
    flight = ctx._flight_map.get(flight_id)
    if flight is None:
        raise ToolError("flight_not_found", f"航班 {flight_id} 不存在")
    if flight["seats_left"] <= 0:
        raise ToolError("no_seats", f"航班 {flight_id} 已无剩余座位")
    if cabin != flight["class"]:
        raise ToolError(
            "class_mismatch",
            f"航班 {flight_id} 为 {flight['class']} 舱，无法以 {cabin} 舱预订",
        )
    reservations = ctx.state["reservations"]
    bucket = reservations.setdefault(user_id, {})
    reservation_id = f"R_{flight_id}_{user_id}"
    # 幂等：同一 (user, flight) 已预订时原样返回，不重复扣减座位
    if reservation_id in bucket:
        return copy.deepcopy(bucket[reservation_id])
    record = {
        "reservation_id": reservation_id,
        "user_id": user_id,
        "flight_id": flight_id,
        "class": cabin,
        "price_cents": flight["price_cents"],
        "status": "booked",
    }
    bucket[reservation_id] = record
    flight["seats_left"] -= 1
    return copy.deepcopy(record)


@_register
def cancel_reservation(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    """取消本人已预订（booked）的预订记录，状态置 cancelled。"""
    user_id, reservation_id = args.get("user_id"), args.get("reservation_id")
    record = _resolve_reservation(ctx, user_id, reservation_id)
    if record["status"] != "booked":
        raise ToolError(
            "invalid_state",
            f"预订 {reservation_id} 当前状态为 {record['status']}，无法取消",
        )
    record["status"] = "cancelled"
    return copy.deepcopy(record)


@_register
def refund_reservation(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    """对本人已预订且已完成身份验证的预订退款，状态置 refunded。"""
    user_id, reservation_id = args.get("user_id"), args.get("reservation_id")
    record = _resolve_reservation(ctx, user_id, reservation_id)
    if not ctx.users.get(user_id, {}).get("verified", False):
        raise ToolError("auth_required", "退款前必须先完成身份验证")
    if record["status"] != "booked":
        raise ToolError(
            "invalid_state",
            f"预订 {reservation_id} 当前状态为 {record['status']}，无法退款",
        )
    record["status"] = "refunded"
    return {"refund_cents": record["price_cents"]}
