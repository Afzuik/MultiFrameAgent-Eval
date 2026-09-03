"""v1.1 修正的回归测试（源于 W2 真实实验报告 §6 的改进清单）。

覆盖：数字归一化匹配（千分位/小数）、措辞宽松（tr_006/sh_014）、
smolagents 步数预算系数。这些是评测器自身质量修正，不重跑历史实验。
"""
from metrics.success import load_task, verify_run
from tasks.verifiers import _numeric_hit, gt_calls


def test_numeric_hit_thousands_separator():
    assert _numeric_hit("总销售额为 21,680 元。", "21680") is True
    assert _numeric_hit("总销售额为 21,680.00 元。", "21680") is True
    assert _numeric_hit("总销售额为 21680 元。", "21680") is True


def test_numeric_hit_no_false_positive_on_different_value():
    # "690" 不得命中 "6,900"（数值不相等）——Decimal 精确相等而非子串
    assert _numeric_hit("价格 6,900 元", "690") is False
    assert _numeric_hit("价格 690 元", "690") is True
    # "21680" 不得命中 "216800"
    assert _numeric_hit("共 216,800 元", "21680") is False


def test_numeric_hit_non_numeric_needle_always_false():
    assert _numeric_hit("MU5101", "MU5101") is False  # 非纯数字走原生子串


def test_an001_thousands_separator_passes():
    """an_001 千分位假阴性修复：21,680 应通过 contains:21680。"""
    task = load_task("an_001")
    answer = "2026 年 9 月的总销售额为 21,680 元。"
    passed, findings = verify_run(task, gt_calls(task), task["gt_final_state"], answer)
    assert passed is True, findings


def test_an001_decimal_tail_passes():
    task = load_task("an_001")
    answer = "总销售额为 21,680.00 元（2,168,000 分）。"
    passed, findings = verify_run(task, gt_calls(task), task["gt_final_state"], answer)
    assert passed is True, findings


def test_an001_wrong_value_still_fails():
    """数值匹配不能放水：21680 的检查对错误数值 2168 仍应失败。"""
    task = load_task("an_001")
    answer = "总销售额为 2,168 元。"
    passed, findings = verify_run(task, gt_calls(task), task["gt_final_state"], answer)
    assert passed is False
    assert "answer_missing:21680" in findings


def test_tr006_relaxed_phrasing_passes():
    """tr_006：'已为您取消' 应命中宽松关键词 '取消'。"""
    task = load_task("tr_006")
    answer = "已为您取消上海回北京的机票预订（R_091）。"
    passed, findings = verify_run(task, gt_calls(task), task["gt_final_state"], answer)
    assert passed is True, findings


def test_sh014_relaxed_phrasing_passes():
    """sh_014：'修改为' 应命中宽松关键词 '修改'。"""
    task = load_task("sh_014")
    answer = "已为您将显示器订单收货地址修改为杭州市西湖区文三路 100 号。"
    passed, findings = verify_run(task, gt_calls(task), task["gt_final_state"], answer)
    assert passed is True, findings


def test_smolagents_step_scale_applied():
    from harness import orchestrator
    assert orchestrator.FRAMEWORK_STEP_SCALE["smolagents"] == 1.5
    assert orchestrator.FRAMEWORK_STEP_SCALE["react"] == 1.0
    # 未知框架兜底 1.0
    assert orchestrator.FRAMEWORK_STEP_SCALE.get("unknown", 1.0) == 1.0
