"""test_metrics —— 指标函数与 run 目录汇总的正确性。

覆盖三块：
1. metrics.cost_latency：total_cost / wall_time / n_tool_calls /
   n_model_turns / tool_roundtrip_s 对构造 Trace 的计算（含兜底分支）；
2. metrics.aggregate.aggregate_run_dir：主路径（results.csv + traces 补缺列）、
   容错路径（仅 traces）与 CLI，断言 summary 内容与 summary.json 落盘；
3. metrics.success.calls_from_trace 与 Trace.tool_calls() 口径一致。
"""
import csv
import json

import pytest

from harness.protocol import Step, Trace, load_trace, write_trace
from metrics.aggregate import aggregate_run_dir
from metrics.aggregate import main as aggregate_main
from metrics.cost_latency import (
    n_model_turns,
    n_tool_calls,
    tool_roundtrip_s,
    total_cost,
    wall_time,
)
from metrics.success import calls_from_trace

CSV_FIELDS = [
    "task_id", "group", "framework", "model", "difficulty", "status",
    "passed", "findings", "cost_usd", "wall_time_s", "n_tool_calls", "n_steps",
]


# ---------- 造数据用的 Trace 工厂 ----------

def _make_passed_trace() -> Trace:
    """tr_001 的"通过"轨迹：6 步、2 次工具调用、成本 0.006、wall_time_s=12.5。"""
    steps = [
        Step(type="message", role="assistant", content="先查上午航班", ts=0.0, cost_usd=0.001),
        Step(
            type="tool_call", role="assistant", tool_name="search_flights",
            tool_args={"date": "2026-10-12", "from": "PEK", "to": "SHA"},
            ts=1.0, cost_usd=0.001,
        ),
        Step(type="observation", role="tool", content="[...]", ts=2.0, cost_usd=0.0),
        Step(
            type="message", role="assistant", content="价格在预算内，订票",
            ts=3.0, cost_usd=0.002,
        ),
        Step(
            type="tool_call", role="assistant", tool_name="book_flight",
            tool_args={"flight_id": "MU5101", "class": "economy"},
            ts=4.0, cost_usd=0.002,
        ),
        Step(type="observation", role="tool", content="[...]", ts=5.0, cost_usd=0.0),
    ]
    return Trace(
        run_id="r_agg", task_id="tr_001", framework="react", model="deepseek-v4-flash",
        steps=steps, final_answer="已订好 MU5101，价格 980 元。", status="completed",
        wall_time_s=12.5, total_cost_usd=0.006,
    )


def _make_failed_trace() -> Trace:
    """tr_007 的"失败"轨迹：4 步、1 次工具调用、成本 0.002、wall_time_s=0（用 ts 差兜底）。"""
    steps = [
        Step(type="message", role="assistant", content="列出我的预订", ts=0.5, cost_usd=0.001),
        Step(
            type="tool_call", role="assistant", tool_name="list_reservations",
            tool_args={"user_id": "u_42"}, ts=1.5, cost_usd=0.001,
        ),
        Step(type="observation", role="tool", content="[...]", ts=2.5, cost_usd=0.0),
        Step(type="message", role="assistant", content="超时前最后一句", ts=3.5, cost_usd=0.0),
    ]
    return Trace(
        run_id="r_agg", task_id="tr_007", framework="react", model="deepseek-v4-flash",
        steps=steps, final_answer="", status="error",
        wall_time_s=0.0, total_cost_usd=0.002,
    )


def _write_traces(run_dir) -> None:
    """按 protocol.write_trace 写 traces/{task_id}.jsonl。"""
    for trace in (_make_passed_trace(), _make_failed_trace()):
        write_trace(trace, run_dir / "traces" / f"{trace.task_id}.jsonl")


def _write_results_csv(run_dir) -> None:
    """写完整 results.csv：部分单元格留空以验证"traces 补缺列"路径。"""
    rows = [
        # passed / L1：wall_time_s、n_tool_calls 留空 → 由轨迹补
        {
            "task_id": "tr_001", "group": "R1", "framework": "react",
            "model": "deepseek-v4-flash", "difficulty": "L1",
            "status": "completed", "passed": "1", "findings": "",
            "cost_usd": "0.006", "wall_time_s": "", "n_tool_calls": "", "n_steps": "6",
        },
        # failed / L3：status、n_steps 留空 → 由轨迹补（status=error、4 步）
        {
            "task_id": "tr_007", "group": "R1", "framework": "react",
            "model": "deepseek-v4-flash", "difficulty": "L3",
            "status": "", "passed": "0",
            "findings": "missing_call:verify_identity,"
                        "state_mismatch:reservations.u_42.R_091.status",
            "cost_usd": "0.002", "wall_time_s": "3.0", "n_tool_calls": "1", "n_steps": "",
        },
    ]
    with (run_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _make_run_dir(tmp_path, with_csv: bool = True):
    """造一个 run 目录：traces 必有，results.csv 可选（容错路径用）。"""
    run_dir = tmp_path / "run_dir"
    _write_traces(run_dir)
    if with_csv:
        _write_results_csv(run_dir)
    return run_dir


# ---------- cost_latency 指标函数 ----------

def test_total_cost_sums_steps_and_ignores_summary_when_nonzero():
    trace = _make_passed_trace()
    # 步骤成本 0.001*2 + 0.002*2 = 0.006；即使 total_cost_usd 不同也以步骤为准
    assert total_cost(trace) == pytest.approx(0.006)


def test_total_cost_falls_back_to_trace_summary_when_steps_all_zero():
    steps = [
        Step(type="message", role="assistant", ts=1.0, cost_usd=0.0),
        Step(type="tool_call", role="assistant", tool_name="x", ts=2.0, cost_usd=0.0),
    ]
    trace = Trace(
        run_id="r", task_id="t", framework="react", model="m",
        steps=steps, status="completed", wall_time_s=1.0, total_cost_usd=0.005,
    )
    assert total_cost(trace) == pytest.approx(0.005)
    # 兜底值本身为 0 → 返回 0
    trace.total_cost_usd = 0.0
    assert total_cost(trace) == 0.0


def test_wall_time_prefers_trace_field():
    trace = _make_passed_trace()  # wall_time_s=12.5，首末 ts 差只有 5.0
    assert wall_time(trace) == pytest.approx(12.5)


def test_wall_time_falls_back_to_first_last_ts_diff():
    trace = _make_failed_trace()  # wall_time_s=0 → 用 3.5 - 0.5
    assert wall_time(trace) == pytest.approx(3.0)


def test_wall_time_zero_without_usable_ts():
    steps = [
        Step(type="message", role="assistant", ts=0.0, cost_usd=0.001),
        Step(type="tool_call", role="assistant", tool_name="x", ts=0.0, cost_usd=0.001),
    ]
    trace = Trace(run_id="r", task_id="t", framework="f", model="m", steps=steps)
    assert wall_time(trace) == 0.0  # 端点 ts 缺失 → 0
    trace.steps = []
    assert wall_time(trace) == 0.0  # 无步骤 → 0


def test_n_tool_calls_and_n_model_turns():
    steps = [
        Step(type="message", role="user", content="hi"),
        Step(type="message", role="assistant", content="好的"),
        Step(type="tool_call", role="assistant", tool_name="search_flights"),
        Step(type="observation", role="tool"),
        Step(type="message", role="assistant", content="查好了"),
        Step(type="tool_call", role="assistant", tool_name="book_flight"),
        Step(type="observation", role="tool"),
    ]
    trace = Trace(run_id="r", task_id="t", framework="f", model="m", steps=steps)
    assert n_tool_calls(trace) == 2
    assert n_model_turns(trace) == 2  # 只数 assistant 的 message，不数 user/tool


def test_tool_roundtrip_sums_adjacent_pairs_only():
    steps = [
        Step(type="tool_call", role="assistant", tool_name="a", ts=10.0),
        Step(type="observation", role="tool", ts=12.0),    # 往返 2.0
        Step(type="message", role="assistant", ts=13.0),
        Step(type="tool_call", role="assistant", tool_name="b", ts=14.0),
        Step(type="observation", role="tool", ts=0.0),     # 缺 ts → 该对记 0
        Step(type="tool_call", role="assistant", tool_name="c", ts=20.0),
        Step(type="message", role="assistant", ts=21.0),   # tool_call→message 不计
        Step(type="observation", role="tool", ts=30.0),    # 前一步非 tool_call 不计
    ]
    trace = Trace(run_id="r", task_id="t", framework="f", model="m", steps=steps)
    assert tool_roundtrip_s(trace) == pytest.approx(2.0)


# ---------- aggregate_run_dir：主路径（完整 results.csv + traces 补缺） ----------

def test_aggregate_run_dir_main_path(tmp_path):
    run_dir = _make_run_dir(tmp_path, with_csv=True)
    summary = aggregate_run_dir(run_dir)

    assert summary["n_tasks"] == 2
    assert summary["n_completed"] == 1      # 只有 tr_001 completed，tr_007 为 error
    assert summary["n_passed"] == 1
    assert summary["sr"] == pytest.approx(0.5)
    # 元数据取首个非空行
    assert summary["group"] == "R1"
    assert summary["framework"] == "react"
    assert summary["model"] == "deepseek-v4-flash"
    # 成本/延迟：wall_time_s 与 n_tool_calls 由 traces 补缺列后参与均值
    assert summary["mean_cost_usd"] == pytest.approx(0.004)
    assert summary["total_cost_usd"] == pytest.approx(0.008)
    assert summary["mean_wall_time_s"] == pytest.approx(7.75)
    assert summary["mean_tool_calls"] == pytest.approx(1.5)
    # 分难度：L1 通过、L3 失败、L2 无任务
    assert summary["by_difficulty"] == {
        "L1": {"n": 1, "sr": 1.0},
        "L2": {"n": 0, "sr": 0.0},
        "L3": {"n": 1, "sr": 0.0},
    }
    # 失败明细：只含未通过的一行
    assert summary["failures"] == [
        {
            "task_id": "tr_007", "difficulty": "L3", "status": "error",
            "findings": "missing_call:verify_identity,"
                        "state_mismatch:reservations.u_42.R_091.status",
        },
    ]
    # summary.json 已写盘，且内容与返回值一致
    assert (run_dir / "summary.json").is_file()
    assert json.loads((run_dir / "summary.json").read_text(encoding="utf-8")) == summary


# ---------- aggregate_run_dir：容错路径（results.csv 缺失，仅 traces） ----------

def test_aggregate_run_dir_without_results_csv(tmp_path):
    run_dir = _make_run_dir(tmp_path, with_csv=False)
    summary = aggregate_run_dir(run_dir)

    assert summary["n_tasks"] == 2
    assert summary["n_completed"] == 1      # tr_001 completed，tr_007 error
    assert summary["n_passed"] == 0         # passed 留空 → 全部按未通过计
    assert summary["sr"] == 0.0
    # 成本/延迟列由 traces 计算（含 wall_time_s 的 ts 差兜底）
    assert summary["mean_cost_usd"] == pytest.approx(0.004)
    assert summary["total_cost_usd"] == pytest.approx(0.008)
    assert summary["mean_wall_time_s"] == pytest.approx(7.75)
    # 难度经 load_task 按真实 task_id 补上（tr_001=L1、tr_007=L3）
    assert summary["by_difficulty"]["L1"] == {"n": 1, "sr": 0.0}
    assert summary["by_difficulty"]["L3"] == {"n": 1, "sr": 0.0}
    # 两条都被记入失败（findings 留空）
    assert len(summary["failures"]) == 2
    assert {f["task_id"] for f in summary["failures"]} == {"tr_001", "tr_007"}
    assert all(f["findings"] == "" for f in summary["failures"])
    # group 无来源 → 空串
    assert summary["group"] == ""
    assert (run_dir / "summary.json").is_file()


# ---------- CLI：python -m metrics.aggregate <run_dir> ----------

def test_aggregate_cli_prints_then_writes(tmp_path, capsys):
    run_dir = _make_run_dir(tmp_path, with_csv=True)
    assert aggregate_main([str(run_dir)]) == 0
    captured = capsys.readouterr()
    assert '"n_tasks": 2' in captured.out
    assert '"sr": 0.5' in captured.out
    assert (run_dir / "summary.json").is_file()


# ---------- load_trace 往返 & calls_from_trace 口径 ----------

def test_trace_jsonl_roundtrip_preserves_metrics(tmp_path):
    trace = _make_passed_trace()
    path = tmp_path / "trace.jsonl"
    write_trace(trace, path)
    loaded = load_trace(path)
    assert loaded is not None
    assert loaded.wall_time_s == pytest.approx(12.5)
    assert loaded.total_cost_usd == pytest.approx(0.006)
    assert total_cost(loaded) == pytest.approx(0.006)
    assert wall_time(loaded) == pytest.approx(12.5)
    assert n_tool_calls(loaded) == 2


def test_calls_from_trace_matches_tool_calls():
    trace = _make_passed_trace()
    assert calls_from_trace(trace) == trace.tool_calls()
    assert [c["tool"] for c in calls_from_trace(trace)] == [
        "search_flights", "book_flight",
    ]
