"""工具注册表加载与查询模块。

唯一数据来源是 tool_server/tool_registry.json：工具名 / 参数 schema / 业务
规则 / fixture 确定性数据（价格单位为分 cent）一律以该文件为准。本模块在
导入时一次性加载并做浅层结构自检，业务模块通过本模块读取工具定义与
fixture 数据，禁止绕过本模块直接读取 JSON 文件。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REGISTRY_PATH = Path(__file__).resolve().parent / "tool_registry.json"
REGISTRY: dict[str, Any] = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))

# 顶层结构：tools（按域分组）、fixtures（确定性种子数据）
FIXTURES: dict[str, Any] = REGISTRY["fixtures"]
TOOLS: dict[str, list[dict[str, Any]]] = REGISTRY["tools"]
VALID_DOMAINS: tuple[str, ...] = tuple(TOOLS.keys())

# analytics 只读 SQL 允许查询的表（对应 fixtures.sales/products/regions）
ANALYTICS_TABLES: tuple[str, ...] = ("sales", "products", "regions")

# 域 → {工具名 → 工具定义}，方便按名查询
DOMAIN_TOOL_DEFS: dict[str, dict[str, dict[str, Any]]] = {
    domain: {tool["name"]: tool for tool in tools}
    for domain, tools in TOOLS.items()
}


def _check_registry() -> None:
    """浅层自检：结构完整、每域工具名唯一，防止注册表被误改后静默出错。"""
    if "version" not in REGISTRY:
        raise RuntimeError("tool_registry.json 缺少 version 字段")
    if not TOOLS:
        raise RuntimeError("tool_registry.json 缺少 tools 字段")
    for domain, tools in TOOLS.items():
        names = [tool.get("name") for tool in tools]
        if len(names) != len(set(names)):
            raise RuntimeError(f"域 {domain} 存在重名工具: {names}")
        for tool in tools:
            for key in ("name", "description", "parameters"):
                if key not in tool:
                    raise RuntimeError(f"域 {domain} 工具 {tool.get('name')} 缺少 {key}")


_check_registry()


def tool_def(domain: str, name: str) -> dict[str, Any] | None:
    """返回某域下某个工具的定义，工具不存在时返回 None。"""
    return DOMAIN_TOOL_DEFS.get(domain, {}).get(name)


def required_params(domain: str, name: str) -> tuple[str, ...]:
    """返回某工具的必填参数名列表（来自注册表 parameters.required）。"""
    definition = tool_def(domain, name)
    if definition is None:
        return ()
    return tuple(definition.get("parameters", {}).get("required", []))


def list_tool_defs(domain: str) -> list[dict[str, Any]]:
    """返回某域全部工具定义列表（浅拷贝，调用方可安全遍历）。"""
    return [dict(tool) for tool in TOOLS.get(domain, [])]
