"""tests/test_dashboard.py —— dashboard.data 纯函数单测（不启动 streamlit）。

覆盖《项目方案.md》§12 dashboard 数据层：
- list_run_dirs 过滤（真实 run 目录保留；dryrun/failed 命名与
  config.yaml dry_run: true 的目录排除；无 results.csv 的排除）；
- load_group_data / load_all_groups 数值正确；
- load_group_data 缺 summary.json / failure_modes.json 时自动生成；
- load_task_rows / load_trace_steps / load_trace_meta 字段与截断。
"""
import json
from pathlib import Path

import pytest

from dashboard.data import (
    list_run_dirs,
    load_all_groups,
    load_group_data,
    load_task_rows,
    load_trace_meta,
    load_trace_steps,
)

# 六类失败模式英文键（与 analysis.failure_modes.CATEGORIES 顺序一致）
_CATS = [
    "permission_violation", "parse_failure", "budget_exhausted",
    "tool_selection_error", "arg_error", "planning_failure",
]


def _write_json(path: Path, obj: object) -> None:
    """写 JSON 文件（ensure_ascii=False，中文可读）。"""
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def _csv_row(
    task_id: str,
    group: str,
    framework: str,
    model: str,
    difficulty: str,
    passed: str,
    findings: str = "",
    status: str = "completed",
    cost: str = "0.001",
    wall: str = "1.0",
    calls: str = "1",
    steps: str = "2",
    f1: str = "1.0",
) -> list[str]:
    """拼一行 results.csv（带 F1 三列，15 列口径）。"""
    return [
        task_id, group, framework, model, difficulty, status, passed, findings,
        cost, wall, calls, steps, f1, f1, f1,
    ]


def _write_results(results_csv: Path, rows: list[list[str]]) -> None:
    """写 results.csv（15 列表头 + rows；results_csv 为文件完整路径）。"""
    header = (
        "task_id,group,framework,model,difficulty,status,passed,findings,"
        "cost_usd,wall_time_s,n_tool_calls,n_steps,f1_recall,f1_precision,f1"
    )
    lines = [header] + [",".join(row) for row in rows]
    results_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _empty_dist(total: int, failed: int, cat: str | None = None) -> dict:
    """手写 failure_modes.json 分布 dict（可选把 1 个失败归入 cat）。"""
    by_cat = {c: 0 for c in _CATS}
    if cat is not None:
        by_cat[cat] = 1
    pct = {c: (100.0 if cat == c else 0.0) for c in _CATS} if failed else {c: 0.0 for c in _CATS}
    return {
        "total": total,
        "failed": failed,
        "by_category": by_cat,
        "by_category_pct": pct,
        "by_framework": {},
        "details": [],
    }


def _build_real_runs(root: Path) -> None:
    """造一个带 2 个真实 run + 各类噪声目录的 runs 根目录。"""
    # A1：react，4 任务 3 通过（sr 0.75）
    a1 = root / "2026-09-03_A1"
    a1.mkdir(parents=True)
    rows = [
        _csv_row("tr_001", "A1", "react", "deepseek-v4-flash", "L1", "True"),
        _csv_row("tr_002", "A1", "react", "deepseek-v4-flash", "L1", "True"),
        _csv_row("tr_003", "A1", "react", "deepseek-v4-flash", "L2", "True"),
        _csv_row(
            "tr_004", "A1", "react", "deepseek-v4-flash", "L3", "False",
            findings="missing_call:book_flight", f1="0.5",
        ),
    ]
    _write_results(a1 / "results.csv", rows)
    _write_json(a1 / "summary.json", {
        "group": "A1", "framework": "react", "model": "deepseek-v4-flash",
        "n_tasks": 4, "n_completed": 4, "n_passed": 3, "sr": 0.75,
        "mean_cost_usd": 0.0025, "total_cost_usd": 0.01,
        "mean_wall_time_s": 2.5, "mean_tool_calls": 1.75,
        "mean_f1": 0.8, "mean_f1_recall": 0.75, "mean_f1_precision": 1.0,
        "by_difficulty": {
            "L1": {"n": 2, "sr": 1.0},
            "L2": {"n": 1, "sr": 1.0},
            "L3": {"n": 1, "sr": 0.0},
        },
        "failures": [{
            "task_id": "tr_004", "difficulty": "L3", "status": "completed",
            "findings": "missing_call:book_flight",
        }],
    })
    dist = _empty_dist(total=4, failed=1, cat="planning_failure")
    dist["details"] = [{
        "task_id": "tr_004", "framework": "react", "difficulty": "L3",
        "category": "planning_failure", "status": "completed",
        "findings": ["missing_call:book_flight"],
    }]
    dist["by_framework"] = {
        "react": {"failed": 1, "by_category": dist["by_category"]},
    }
    _write_json(a1 / "failure_modes.json", dist)

    # A2：smolagents，2 任务全过（sr 1.0）
    a2 = root / "2026-09-03_A2"
    a2.mkdir()
    _write_results(a2 / "results.csv", [
        _csv_row("tr_a", "A2", "smolagents", "qwen-72b", "L1", "True"),
        _csv_row("tr_b", "A2", "smolagents", "qwen-72b", "L2", "True"),
    ])
    _write_json(a2 / "summary.json", {
        "group": "A2", "framework": "smolagents", "model": "qwen-72b",
        "n_tasks": 2, "n_completed": 2, "n_passed": 2, "sr": 1.0,
        "mean_cost_usd": 0.001, "total_cost_usd": 0.002,
        "mean_wall_time_s": 1.0, "mean_tool_calls": 1.0,
        "mean_f1": 1.0, "mean_f1_recall": 1.0, "mean_f1_precision": 1.0,
        "by_difficulty": {
            "L1": {"n": 1, "sr": 1.0},
            "L2": {"n": 1, "sr": 1.0},
            "L3": {"n": 0, "sr": 0.0},
        },
        "failures": [],
    })
    _write_json(a2 / "failure_modes.json", _empty_dist(total=2, failed=0))

    # 噪声目录：dryrun / failed 命名、无 results.csv、config 声明 dry_run
    noise = root / "2026-09-03_X1_dryrun"
    noise.mkdir()
    _write_results(noise / "results.csv", [_csv_row("t1", "X1", "react", "m", "L1", "True")])
    (root / "2026-09-03_X2_failed").mkdir()
    _write_results(root / "2026-09-03_X2_failed" / "results.csv",
        [_csv_row("t1", "X2", "react", "m", "L1", "True")],
    )
    (root / "2026-09-03_X3_nodata").mkdir()
    (root / "2026-09-03_X3_nodata" / "config.yaml").write_text("x: 1\n", encoding="utf-8")
    fake = root / "2026-09-03_O9"
    fake.mkdir()
    _write_results(fake / "results.csv", [_csv_row("t1", "O9", "openhands", "m", "L1", "True")])
    (fake / "config.yaml").write_text("dry_run: true\n", encoding="utf-8")
    (root / "notes.txt").write_text("hello\n", encoding="utf-8")


def test_list_run_dirs_filters(tmp_path: Path) -> None:
    """list_run_dirs：保留真实 run，排除 dryrun/failed 命名与 dry_run 目录。"""
    root = tmp_path / "runs"
    root.mkdir()
    _build_real_runs(root)
    result = list_run_dirs(root)
    assert [p.name for p in result] == ["2026-09-03_A1", "2026-09-03_A2"]
    # 全部结果目录确实存在 results.csv
    assert all((p / "results.csv").is_file() for p in result)
    # 不存在的根目录返回空列表
    assert list_run_dirs(root / "missing") == []


def test_load_all_groups_values(tmp_path: Path) -> None:
    """load_all_groups：按 group 排序、数值正确（含分难度与失败模式）。"""
    root = tmp_path / "runs"
    root.mkdir()
    _build_real_runs(root)
    groups = load_all_groups(root)
    assert [g["group"] for g in groups] == ["A1", "A2"]

    a1, a2 = groups
    assert (a1["framework"], a1["model"]) == ("react", "deepseek-v4-flash")
    assert a1["n_tasks"] == 4 and a1["n_passed"] == 3
    assert a1["sr"] == pytest.approx(0.75)
    assert a1["f1"] == pytest.approx(0.8)
    assert a1["cost_usd"] == pytest.approx(0.0025)
    assert a1["latency_s"] == pytest.approx(2.5)
    assert a1["tool_calls"] == pytest.approx(1.75)
    assert a1["by_difficulty"]["L3"]["sr"] == pytest.approx(0.0)
    assert a1["by_difficulty"]["L1"]["sr"] == pytest.approx(1.0)
    assert len(a1["failures"]) == 1
    assert a1["failures"][0]["task_id"] == "tr_004"
    fm = a1["failure_modes"]
    assert fm["failed"] == 1 and fm["total"] == 4
    assert fm["by_category"]["planning_failure"] == 1
    assert fm["details"][0]["category"] == "planning_failure"

    assert a2["sr"] == pytest.approx(1.0)
    assert a2["failure_modes"]["failed"] == 0

    # 空 runs 根目录返回空列表
    assert load_all_groups(root / "empty") == []


def test_load_group_data_generates_missing(tmp_path: Path) -> None:
    """缺 summary.json / failure_modes.json 时自动生成并落盘。"""
    run_dir = tmp_path / "runs" / "2026-09-03_G1"
    run_dir.mkdir(parents=True)
    # 12 列老口径（无 F1 列）：验证自动汇总从 results.csv 恢复指标
    header = (
        "task_id,group,framework,model,difficulty,status,passed,findings,"
        "cost_usd,wall_time_s,n_tool_calls,n_steps"
    )
    lines = [header]
    lines.append("g_01,G1,react,m1,L1,completed,True,,0.001,1.0,1,2")
    lines.append("g_02,G1,react,m1,L2,completed,True,,0.001,1.0,1,2")
    lines.append(
        "g_03,G1,react,m1,L2,timeout,False,"
        "forbidden_call:refund_reservation,0.0,180.0,3,5"
    )
    (run_dir / "results.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert not (run_dir / "summary.json").exists()
    assert not (run_dir / "failure_modes.json").exists()

    group = load_group_data(run_dir)

    # 生成文件落盘
    assert (run_dir / "summary.json").is_file()
    assert (run_dir / "failure_modes.json").is_file()
    # 聚合数值（2/3 通过，四舍五入到 4 位 = 0.6667）
    assert group["group"] == "G1"
    assert (group["framework"], group["model"]) == ("react", "m1")
    assert group["n_tasks"] == 3
    assert group["sr"] == pytest.approx(2 / 3, abs=1e-3)
    assert group["n_passed"] == 2
    assert len(group["failures"]) == 1
    # failure_modes 自动生成：fake task_id 不在任务库 → 无法分类但结构完整
    assert set(group["failure_modes"]) >= {"total", "failed", "by_category", "details"}
    assert set(group["failure_modes"]["by_category"]) == set(_CATS)
    # 无 F1 列的行：f1 为 None（老 CSV 口径）
    rows = load_task_rows(run_dir)
    assert len(rows) == 3
    assert rows[0]["f1"] is None
    assert rows[0]["passed"] is True
    assert rows[2]["passed"] is False
    assert "forbidden_call" in rows[2]["findings"]

    # 幂等：再次加载走读文件路径，结果一致
    again = load_group_data(run_dir)
    assert again["sr"] == pytest.approx(group["sr"])


def test_load_trace_steps_truncation_and_fields(tmp_path: Path) -> None:
    """load_trace_steps：字段齐全、content 截断 200、耗时差正确。"""
    run_dir = tmp_path / "runs" / "2026-09-03_T1"
    traces = run_dir / "traces"
    traces.mkdir(parents=True)

    long_content = "查询" + "x" * 300  # 超过 200 字符
    lines = [
        json.dumps({
            "run_id": "T1__tr_007", "task_id": "tr_007",
            "framework": "react", "model": "m1",
            "step": 1, "type": "message", "role": "user",
            "content": long_content, "ts": 1.0, "cost_usd": 0.0,
        }, ensure_ascii=False),
        json.dumps({
            "run_id": "T1__tr_007", "task_id": "tr_007",
            "framework": "react", "model": "m1",
            "step": 2, "type": "tool_call", "role": "assistant",
            "content": "", "tool_name": "search_flights",
            "tool_args": {"date": "2026-10-12", "from": "PEK", "to": "SHA"},
            "ts": 3.0, "cost_usd": 0.001,
        }, ensure_ascii=False),
        json.dumps({
            "run_id": "T1__tr_007", "task_id": "tr_007",
            "framework": "react", "model": "m1",
            "step": 3, "type": "final_answer", "role": "assistant",
            "content": "已为您查好航班", "status": "completed",
            "wall_time_s": 5.5, "total_cost_usd": 0.001,
        }, ensure_ascii=False),
    ]
    (traces / "tr_007.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    steps = load_trace_steps(run_dir, "tr_007")
    # final_answer 汇总行不计入步骤：只有 message + tool_call 两步
    assert len(steps) == 2
    assert [s["step"] for s in steps] == [1, 2]

    first, second = steps
    assert first["type"] == "message" and first["role"] == "user"
    # 截断：200 字符 + 省略号，前缀与原内容一致
    assert len(first["content"]) == 201
    assert first["content"].endswith("…")
    assert first["content"][:200] == long_content[:200]
    assert first["tool_name"] is None and first["tool_args"] is None
    assert first["duration_s"] == 0.0

    assert second["type"] == "tool_call"
    assert second["tool_name"] == "search_flights"
    assert second["tool_args"] == {"date": "2026-10-12", "from": "PEK", "to": "SHA"}
    assert second["cost_usd"] == pytest.approx(0.001)
    assert second["ts"] == pytest.approx(3.0)
    assert second["duration_s"] == pytest.approx(2.0)  # 距上一步 2s

    # meta：汇总行信息
    meta = load_trace_meta(run_dir, "tr_007")
    assert meta is not None
    assert meta["status"] == "completed"
    assert meta["final_answer"] == "已为您查好航班"
    assert meta["wall_time_s"] == pytest.approx(5.5)
    assert meta["n_steps"] == 2
    assert meta["n_tool_calls"] == 1

    # 轨迹缺失：空列表 / None，不抛异常
    assert load_trace_steps(run_dir, "no_such_task") == []
    assert load_trace_meta(run_dir, "no_such_task") is None
