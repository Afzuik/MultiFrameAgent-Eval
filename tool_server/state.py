"""每实例状态管理模块。

评测时编排器为每个任务创建一个实例（instance_id = run_id），实例拥有完全
独立的可变状态与日志：

- travel 域：state = {users, reservations(按用户分组), session}
- shop 域：  state = {users, orders, session}
- analytics 域：state = {reports, session}，查询数据放实例私有的内存 SQLite
  （sales/products/regions 三张表，由 fixtures 确定性建表）

flights/products/regions/sales 等 fixture 一律按实例深拷贝，保证：
1) 实例间完全隔离；2) 同一 (instance, tool, args) 在等价状态下结果确定。
reset 语义：种子(深拷贝) → initial_state 深合并覆盖 → 写回 session。
"""
from __future__ import annotations

import copy
import sqlite3
import time
from typing import Any

from .registry import FIXTURES

DEFAULT_USER_ID = "u_42"


class ToolError(Exception):
    """业务拒绝信号：app 层捕获后转为 HTTP 200 + {"ok": false, ...}。

    字段 code 为错误码（见各工具实现），message 为中文原因。
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归深合并：两边都是 dict 则递归合并，否则用 override 值直接替换。

    在 base 上原地修改后返回，便于 reset 用 initial_state 覆盖种子。
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


class InstanceState:
    """单个评测实例的全部运行时状态，同时作为工具执行上下文。"""

    def __init__(self) -> None:
        self.domain: str = ""
        self.state: dict[str, Any] = {}
        self.log: list[dict[str, Any]] = []
        # travel：实例私有航班副本（航线 → 航班列表），预订扣减的座位记在这里
        self._flights: dict[str, list[dict[str, Any]]] = {}
        # flight_id → 航班记录引用（指向 _flights 中的同一对象）
        self._flight_map: dict[str, dict[str, Any]] = {}
        # analytics：内存 SQLite 连接（实例私有）
        self._db: sqlite3.Connection | None = None

    # ------------------------------------------------------------------ reset
    def reset(self, domain: str, initial_state: dict[str, Any] | None = None) -> None:
        """按域重建种子状态，用 initial_state 深合并覆盖，并重置日志。

        domain 合法性由 app 层（pydantic Literal）先行校验。
        """
        initial_state = initial_state or {}
        self.domain = domain
        self._db = None
        self._flights = {}
        self._flight_map = {}

        if domain == "travel":
            self.state = self._build_travel_seed()
            self._load_flights()
        elif domain == "shop":
            self.state = self._build_shop_seed()
        elif domain == "analytics":
            self.state = {"reports": {}}
            self._db = self._build_analytics_db()
        else:  # 防御：正常不会走到
            raise ValueError(f"未知 domain: {domain}")

        # initial_state 深合并覆盖（dict 递归、非 dict 替换）
        if initial_state:
            deep_merge(self.state, initial_state)

        # session 记录当前用户（默认 u_42）
        uid = initial_state.get("user_id") or DEFAULT_USER_ID
        self.state["session"] = {"user_id": uid}

        # 开启新一轮评测运行：日志一并清空
        self.log = []

    # ------------------------------------------------------------ seed 构造
    def _build_travel_seed(self) -> dict[str, Any]:
        """travel 种子：users 用 travel_users；reservations 由 seed 按用户分组。

        fixture 中 travel_reservation_seed 以 reservation_id 为顶层键，而状态
        结构约定为 reservations.{user_id}.{reservation_id}（verifier 读取路径
        如 reservations.u_42.R_091.status），故按记录的 user_id 重新分组。
        """
        users = copy.deepcopy(FIXTURES["travel_users"])
        reservations: dict[str, dict[str, dict[str, Any]]] = {}
        for rid, record in FIXTURES["travel_reservation_seed"].items():
            uid = record["user_id"]
            reservations.setdefault(uid, {})[rid] = copy.deepcopy(record)
        return {"users": users, "reservations": reservations}

    def _load_flights(self) -> None:
        """把航班 fixture 深拷贝为实例私有数据，并建立 flight_id 索引。"""
        self._flights = copy.deepcopy(FIXTURES["flights"])
        for flights in self._flights.values():
            for flight in flights:
                self._flight_map[flight["flight_id"]] = flight

    def _build_shop_seed(self) -> dict[str, Any]:
        """shop 种子：users 用 shop_users；orders 用 shop_orders（平铺 dict）。"""
        return {
            "users": copy.deepcopy(FIXTURES["shop_users"]),
            "orders": copy.deepcopy(FIXTURES["shop_orders"]),
        }

    def _build_analytics_db(self) -> sqlite3.Connection:
        """由 fixtures 建内存库三张表：sales(15 行)/products(4 行)/regions(3 行)。"""
        db = sqlite3.connect(":memory:", check_same_thread=False)
        db.execute(
            "CREATE TABLE sales (id INTEGER PRIMARY KEY, date TEXT NOT NULL,"
            " product TEXT NOT NULL, region TEXT NOT NULL, channel TEXT NOT NULL,"
            " amount_cents INTEGER NOT NULL)"
        )
        db.execute(
            "CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT NOT NULL,"
            " category TEXT NOT NULL)"
        )
        db.execute(
            "CREATE TABLE regions (name TEXT PRIMARY KEY, manager TEXT NOT NULL)"
        )
        # products：{P1: {name, category}, ...}
        db.executemany(
            "INSERT INTO products (id, name, category) VALUES (?, ?, ?)",
            [
                (pid, prod["name"], prod["category"])
                for pid, prod in FIXTURES["products"].items()
            ],
        )
        # regions：{华北: {manager}, ...}
        db.executemany(
            "INSERT INTO regions (name, manager) VALUES (?, ?)",
            [(name, reg["manager"]) for name, reg in FIXTURES["regions"].items()],
        )
        # sales：[{id, date, product, region, channel, amount_cents}, ...]
        db.executemany(
            "INSERT INTO sales (id, date, product, region, channel, amount_cents)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    row["id"],
                    row["date"],
                    row["product"],
                    row["region"],
                    row["channel"],
                    row["amount_cents"],
                )
                for row in FIXTURES["sales"]
            ],
        )
        db.commit()
        return db

    # ---------------------------------------------------------------- 日志
    def record_log(
        self,
        tool: str,
        args: dict[str, Any],
        ok: bool,
        *,
        result: Any = None,
        error: str | None = None,
        code: str | None = None,
        caller: str = "agent",
    ) -> None:
        """记录一次工具调用日志条目。ok=True 记 result，否则记 error(+code)。"""
        entry: dict[str, Any] = {
            "tool": tool,
            "args": copy.deepcopy(args),
            "ok": ok,
            "ts": time.time(),
            "caller": caller,
        }
        if ok:
            entry["result"] = result
        else:
            entry["error"] = error
            if code is not None:
                entry["code"] = code
        self.log.append(entry)

    # ------------------------------------------------------------ 便捷访问
    @property
    def users(self) -> dict[str, Any]:
        """当前实例的用户字典（travel/shop 域存在）。"""
        return self.state.get("users", {})

    @property
    def session_user_id(self) -> str:
        """当前会话用户（shop 域权限判定使用）。"""
        session = self.state.get("session", {})
        return session.get("user_id", DEFAULT_USER_ID)

    @property
    def db(self) -> sqlite3.Connection:
        """当前实例的内存 SQLite 连接（仅 analytics 域可用）。"""
        if self._db is None:
            raise ToolError("query_error", "当前实例没有可查询的数据库")
        return self._db
