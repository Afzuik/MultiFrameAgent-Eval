"""shop 域 6 个工具的业务实现。

规则严格对照 tool_server/tool_registry.json 中 shop 组各工具的 description
实现。update_address / apply_refund 以 session 用户为操作主体（订单归属判定），
退款类工具 requires_auth：订单所属用户需先通过 verify_identity。
"""
from __future__ import annotations

import copy
from typing import Any

from ..state import ToolError

TOOLS: dict[str, Any] = {}

_REFUNDABLE_STATUS = {"paid", "shipped", "delivered"}
_INVOICE_STATUS = {"paid", "delivered"}


def _register(func: Any) -> Any:
    """装饰器：把工具函数注册进本模块的 TOOLS 分发表。"""
    TOOLS[func.__name__] = func
    return func


def _get_order(ctx: Any, order_id: str) -> dict[str, Any]:
    """按订单号取订单，不存在抛 not_found。"""
    orders = ctx.state.get("orders", {})
    if order_id not in orders:
        raise ToolError("not_found", f"订单 {order_id} 不存在")
    return orders[order_id]


def _require_owner(ctx: Any, order: dict[str, Any]) -> None:
    """订单必须属于 session 用户，否则抛 forbidden。"""
    if order.get("user_id") != ctx.session_user_id:
        raise ToolError("forbidden", "该订单不属于当前用户，无权操作")


@_register
def get_order_status(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    """查询订单状态；订单不存在返回 not_found。"""
    order_id = args.get("order_id")
    order = _get_order(ctx, order_id)
    return {"order_id": order_id, **copy.deepcopy(order)}


@_register
def list_orders(ctx: Any, args: dict[str, Any]) -> list[dict[str, Any]]:
    """列出某用户的全部订单（dict 转 list，含 order_id 键，按单号排序）。"""
    user_id = args.get("user_id")
    orders = ctx.state.get("orders", {})
    items = [
        {"order_id": oid, **copy.deepcopy(order)}
        for oid, order in orders.items()
        if order.get("user_id") == user_id
    ]
    items.sort(key=lambda item: item["order_id"])
    return items


@_register
def update_address(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    """修改收货地址：仅本人 + paid 状态 + 新地址长度 ≥ 10 字符。"""
    order_id, new_address = args.get("order_id"), args.get("new_address")
    order = _get_order(ctx, order_id)
    _require_owner(ctx, order)
    if order["status"] != "paid":
        raise ToolError(
            "invalid_state",
            f"订单 {order_id} 状态为 {order['status']}，仅 paid 订单可修改收货地址",
        )
    if not isinstance(new_address, str) or len(new_address) < 10:
        raise ToolError("address_too_short", "新地址至少需要 10 个字符")
    order["address"] = new_address
    return {"order_id": order_id, **copy.deepcopy(order)}


@_register
def verify_identity(ctx: Any, args: dict[str, Any]) -> dict[str, bool]:
    """验证用户身份。验证码固定为 1234（演示环境），退款前必须先验证。"""
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
def apply_refund(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    """申请退款：本人 + 订单所属用户已验证 + 状态为 paid/shipped/delivered。"""
    order_id = args.get("order_id")
    order = _get_order(ctx, order_id)
    _require_owner(ctx, order)
    owner = order.get("user_id")
    if not ctx.users.get(owner, {}).get("verified", False):
        raise ToolError("auth_required", "申请退款前必须先完成身份验证")
    if order["status"] not in _REFUNDABLE_STATUS:
        raise ToolError(
            "invalid_state",
            f"订单 {order_id} 状态为 {order['status']}，当前不可申请退款",
        )
    order["status"] = "refunding"
    return {"refund_cents": order["amount_cents"]}


@_register
def create_invoice(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    """为订单补开发票：仅 paid/delivered 状态订单可开。"""
    order_id, title = args.get("order_id"), args.get("title")
    order = _get_order(ctx, order_id)
    if order["status"] not in _INVOICE_STATUS:
        raise ToolError(
            "invalid_state",
            f"订单 {order_id} 状态为 {order['status']}，仅 paid/delivered 可开发票",
        )
    order["invoice"] = {"title": title, "amount_cents": order["amount_cents"]}
    return {"order_id": order_id, **copy.deepcopy(order)}
