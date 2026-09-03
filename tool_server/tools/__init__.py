"""工具域实现与分发表。

每个域一个模块（travel.py / shop.py / analytics.py），模块内通过装饰器把
工具函数注册进各自 TOOLS dict。app 层按实例域选择 DOMAIN_TOOLS 中对应
的工具执行；分发表只暴露给 app 使用。
"""
from __future__ import annotations

from typing import Any

from . import analytics, shop, travel

# 域 → {工具名 → 处理函数(ctx, args)}；ctx 为 InstanceState 实例
DOMAIN_TOOLS: dict[str, dict[str, Any]] = {
    "travel": travel.TOOLS,
    "shop": shop.TOOLS,
    "analytics": analytics.TOOLS,
}

__all__ = ["DOMAIN_TOOLS"]
