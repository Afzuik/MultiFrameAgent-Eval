"""tool_registry.json fixture 数据一致性测试。

防止改动 fixture 后任务 verifier 期望值（gt_answer / final_state_checks）
与数据源漂移：任何一行数据改动都会被这里的断言抓住。
数据关系依据：tool_registry.json 内 key_aggregates_2026_09（人工核算基准）
与 tasks/v1 下各任务的 gt_answer / gt_final_state。
"""
import json

from harness.protocol import TOOL_REGISTRY_PATH


def _load() -> dict:
    return json.loads(TOOL_REGISTRY_PATH.read_text(encoding="utf-8"))


def _sep_rows(fx: dict) -> list[dict]:
    return [r for r in fx["sales"] if r["date"].startswith("2026-09")]


def test_september_aggregates_match_manual_baseline():
    """9 月销售聚合必须与 key_aggregates_2026_09 人工核算基准一致。"""
    fx = _load()["fixtures"]
    rows = _sep_rows(fx)
    total = sum(r["amount_cents"] for r in rows)
    online = sum(r["amount_cents"] for r in rows if r["channel"] == "线上")
    offline = total - online
    by_region: dict[str, int] = {}
    for r in rows:
        by_region[r["region"]] = by_region.get(r["region"], 0) + r["amount_cents"]
    p1_rows = [r for r in rows if r["product"] == "P1"]
    digital = sum(
        r["amount_cents"] for r in rows
        if fx["products"][r["product"]]["category"] == "数码"
    )
    south_p2_online = sum(
        r["amount_cents"] for r in rows
        if r["region"] == "华南" and r["channel"] == "线上" and r["product"] == "P2"
    )
    agg = _load()["key_aggregates_2026_09"]
    assert total == agg["total_cents"]            # an_001/an_009: 21680 元
    assert online == agg["online_cents"]          # an_003/an_005: 11894 元
    assert offline == agg["offline_cents"]        # an_005/an_010: 9786 元
    assert by_region == agg["by_region_cents"]    # an_002: 7786/8097/5797
    assert len(p1_rows) == agg["product_p1_count"]            # an_006: 3 台
    assert sum(r["amount_cents"] for r in p1_rows) == agg["product_p1_cents"]  # 12497 元
    assert digital == agg["digital_category_cents"]           # an_004: 21492 元
    assert south_p2_online == agg["south_p2_online_cents"]    # an_008: 1399 元


def test_travel_flight_prices_match_task_answers():
    """航班价格/舱位必须与 travel 任务 gt_answer 引用的数值一致。"""
    fx = _load()["fixtures"]
    flights = {f["flight_id"]: f for route in fx["flights"].values() for f in route}
    assert flights["MU5101"]["price_cents"] == 98000 and flights["MU5101"]["class"] == "economy"
    assert flights["MU5103"]["price_cents"] == 145000 and flights["MU5103"]["class"] == "business"
    assert flights["MU5105"]["price_cents"] == 76000 and flights["MU5105"]["class"] == "economy"
    assert flights["MU5107"]["price_cents"] == 69000  # tr_002/tr_005/tr_012/tr_014: 最便宜
    assert flights["MU5201"]["price_cents"] == 95000  # tr_006/tr_007/tr_013: 退款 950
    assert flights["CZ8801"]["price_cents"] == 82000 and flights["CZ8801"]["class"] == "economy"
    assert flights["ZH9101"]["price_cents"] == 31000  # tr_006: CAN-SZX 干扰票


def test_shop_orders_match_task_answers():
    """电商订单金额/状态/归属必须与 shop 任务 gt_answer 一致。"""
    fx = _load()["fixtures"]
    o = fx["shop_orders"]
    assert o["O20261012001"]["amount_cents"] == 129900 and o["O20261012001"]["status"] == "paid"
    assert o["O20261012002"]["amount_cents"] == 29900 and o["O20261012002"]["status"] == "shipped"
    assert o["O20261012002"]["tracking"] == "SF1234567890"          # sh_002 物流单号
    assert o["O20261005001"]["amount_cents"] == 8900 and o["O20261005001"]["status"] == "delivered"
    assert o["O20261008003"]["amount_cents"] == 5900                # sh_012: 59 元
    assert o["O20260928001"]["user_id"] == "u_77"                   # sh_006: 他人订单
    u42_total = sum(v["amount_cents"] for v in o.values() if v["user_id"] == "u_42")
    assert u42_total == 174600                                      # sh_008: 1746 元
