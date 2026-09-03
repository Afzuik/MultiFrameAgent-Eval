"""v1.2 修正的回归测试（空白归一化 + 端点抖动检测阈值）。"""
from metrics.success import load_task, verify_run
from tasks.verifiers import _ws_normalize, gt_calls


def test_ws_normalize_removes_all_whitespace():
    assert _ws_normalize("上海市浦东新区张江路 88 号") == "上海市浦东新区张江路88号"
    assert _ws_normalize("88号") == _ws_normalize("88 号")
    assert _ws_normalize("MU5101") == "MU5101"
    assert _ws_normalize(123) == 123          # 非字符串原样返回
    assert _ws_normalize(True) is True


def test_sh003_address_space_variance_passes():
    """v1.2：终态地址 '88号'（无空格）与期望 '88 号' 应判等。"""
    task = load_task("sh_003")
    final_state = {
        "orders": {"O20261012001": {"address": "上海市浦东新区张江路 88号"}},
    }
    # GT 路径的调用 + 变异后的终态 + GT 答案
    passed, findings = verify_run(task, gt_calls(task), final_state, task["gt_answer"])
    assert passed is True, findings


def test_sh003_wrong_address_still_fails():
    """空白归一化不能放水：地址内容真的不同仍必须失败。"""
    task = load_task("sh_003")
    final_state = {
        "orders": {"O20261012001": {"address": "上海市浦东新区张江路 99 号"}},
    }
    passed, findings = verify_run(task, gt_calls(task), final_state, task["gt_answer"])
    assert passed is False
    assert any("state_mismatch" in f for f in findings)


def test_non_string_equals_unaffected():
    """数值/布尔类 equals 比较不受归一化影响（tr_011 verified==true）。"""
    task = load_task("tr_011")
    final_state = {"users": {"u_42": {"verified": True}}}
    passed, _ = verify_run(task, gt_calls(task), final_state, task["gt_answer"])
    assert passed is True


def test_endpoint_flaky_threshold_constant():
    from harness import orchestrator
    assert orchestrator.ENDPOINT_FLAKY_THRESHOLD == 3
