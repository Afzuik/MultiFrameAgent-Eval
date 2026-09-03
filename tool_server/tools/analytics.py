"""analytics 域 5 个工具的业务实现。

- get_schema / run_query：针对实例私有内存 SQLite（sales/products/regions）
- export_report / get_report / list_reports：操作 state["reports"]

run_query 安全规则（严格只读）：去空白后小写必须以 select 开头、至多一个
分号、不含写关键字、仅允许查询三张业务表；执行异常返回 query_error。
"""
from __future__ import annotations

import re
import sqlite3
import time
from typing import Any

from ..registry import ANALYTICS_TABLES
from ..state import ToolError

TOOLS: dict[str, Any] = {}

# 各表结构（列名以注册表 fixtures 与任务文档为准）
TABLE_SCHEMAS: dict[str, list[str]] = {
    "sales": ["id", "date", "product", "region", "channel", "amount_cents"],
    "products": ["id", "name", "category"],
    "regions": ["name", "manager"],
}

# 写/危险关键字黑名单（在去空白小写文本上按词边界匹配）
_FORBIDDEN_KEYWORDS = (
    "attach",
    "pragma",
    "insert",
    "update",
    "delete",
    "drop",
    "create",
    "alter",
    "replace",
    "vacuum",
    "reindex",
)

_MAX_ROWS = 500


def _register(func: Any) -> Any:
    """装饰器：把工具函数注册进本模块的 TOOLS 分发表。"""
    TOOLS[func.__name__] = func
    return func


def _validate_readonly_sql(sql: str) -> None:
    """校验 SQL 满足只读约束，违规抛 invalid_sql。"""
    if not isinstance(sql, str) or not sql.strip():
        raise ToolError("invalid_sql", "sql 参数必须是非空字符串")
    lowered = sql.lower()
    compact = re.sub(r"\s+", "", lowered)
    if not compact.startswith("select"):
        raise ToolError("invalid_sql", "仅支持以 SELECT 开头的只读查询")
    # 分号只允许出现在语句末尾（至多一个，且其后不能有其它 SQL）
    stripped = sql.strip()
    body = stripped.removesuffix(";")
    if ";" in body:
        raise ToolError("invalid_sql", "分号只能出现在语句末尾，一次只能执行一条语句")
    for keyword in _FORBIDDEN_KEYWORDS:
        if re.search(rf"(?<![a-z0-9_]){keyword}(?![a-z0-9_])", compact):
            raise ToolError(
                "invalid_sql", f"SQL 包含被禁止的关键字 {keyword.upper()}，仅允许只读查询"
            )
    # 仅允许查询 sales/products/regions 三张表（白名单）
    # 表名解析在保留空白的小写原文上进行，避免别名与关键字粘连
    has_from_join = bool(re.search(r"\b(?:from|join)\b", lowered))
    referenced = set(
        re.findall(r"\b(?:from|join)\s+([a-z_][a-z0-9_]*)", lowered)
    )
    if has_from_join:
        if not referenced:
            raise ToolError("invalid_sql", "无法从 SQL 中解析出查询的数据表")
        bad_tables = sorted(referenced - set(ANALYTICS_TABLES))
        if bad_tables:
            raise ToolError(
                "invalid_sql",
                f"仅允许查询 {', '.join(ANALYTICS_TABLES)}，"
                f"出现非法表: {', '.join(bad_tables)}",
            )
    # 无 FROM/JOIN 的纯表达式查询不引用任何表，属安全只读，予以放行
    if "sqlite_" in compact or "sqlite" in referenced:
        raise ToolError("invalid_sql", "不允许访问 SQLite 系统表")


@_register
def get_schema(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    """查看数据表结构；table 缺省时返回全部表。"""
    table = args.get("table")
    if table is None:
        schema = [
            {"table": name, "columns": columns}
            for name, columns in TABLE_SCHEMAS.items()
        ]
        return {"schema": schema}
    if table not in TABLE_SCHEMAS:
        raise ToolError(
            "invalid_table",
            f"未知数据表 {table}，可用表: {', '.join(TABLE_SCHEMAS)}",
        )
    return {"schema": [{"table": table, "columns": TABLE_SCHEMAS[table]}]}


@_register
def run_query(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    """执行只读 SQL，返回 {columns, rows}，行数上限 500。"""
    sql = args.get("sql")
    _validate_readonly_sql(sql)
    try:
        cursor = ctx.db.execute(sql)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = [list(row) for row in cursor.fetchmany(_MAX_ROWS + 1)]
        if len(rows) > _MAX_ROWS:  # 超出上限截断
            rows = rows[:_MAX_ROWS]
    except sqlite3.Error as exc:  # 语法/执行异常
        raise ToolError("query_error", f"SQL 执行失败：{exc}") from exc
    return {"columns": columns, "rows": rows}


@_register
def export_report(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    """按标题导出报告（重复标题覆盖写），返回 report_id。"""
    title, content = args.get("title"), args.get("content")
    reports = ctx.state["reports"]
    reports[title] = {
        "title": title,
        "content": content,
        "created_ts": time.time(),
    }
    return {"report_id": title}


@_register
def get_report(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    """按标题读取已导出报告；不存在抛 not_found。"""
    report_id = args.get("report_id")
    reports = ctx.state.get("reports", {})
    if report_id not in reports:
        raise ToolError("not_found", f"报告 {report_id} 不存在")
    return copy_report(reports[report_id])


@_register
def list_reports(ctx: Any, args: dict[str, Any]) -> list[dict[str, Any]]:
    """列出全部已导出报告的摘要（按标题排序，不含正文）。"""
    reports = ctx.state.get("reports", {})
    items = [
        {
            "report_id": title,
            "title": record["title"],
            "created_ts": record["created_ts"],
        }
        for title, record in reports.items()
    ]
    items.sort(key=lambda item: item["report_id"])
    return items


def copy_report(record: dict[str, Any]) -> dict[str, Any]:
    """深拷贝报告记录（避免调用方经返回值修改实例状态）。"""
    return {
        "title": record["title"],
        "content": record["content"],
        "created_ts": record["created_ts"],
    }
