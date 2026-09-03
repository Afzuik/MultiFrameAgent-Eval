"""metrics.aggregate —— 单次运行结果汇总（run 目录 → summary.json）。

输入（《项目方案.md》§11.1 目录约定）：
- results.csv：逐任务结果行，列：
  task_id,group,framework,model,difficulty,status,passed,findings,
  cost_usd,wall_time_s,n_tool_calls,n_steps；
- traces/{task_id}.jsonl：归一化轨迹，用于给 results.csv 补缺列
  （某列未记录时从轨迹重算，如成本/延迟/步数等）。

主路径：results.csv 齐全 → 逐行聚合，sr 分母为 n_tasks，
失败/未完成的任务一律计入 fail（§7.1）。
容错路径：results.csv 缺失时仅凭 traces 计算成本/延迟等列、
passed 留空（全部按未通过计，仅作兜底查看用）。

产出 summary（schema 见《项目方案.md》§11 与验收标准 §2.2），
aggregate_run_dir() 会把 summary 写入 run_dir/summary.json 后返回。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from harness.protocol import load_trace
from metrics.cost_latency import n_tool_calls, total_cost, wall_time
from metrics.success import load_task

# W2 的 F1 模块由分析 agent 并行开发；未就绪时 F1 列留空
try:
    from metrics.tool_f1 import compute_f1 as _compute_f1
except ImportError:  # pragma: no cover
    _compute_f1 = None

DIFFICULTIES = ("L1", "L2", "L3")

CSV_COLUMNS = (
    "task_id", "group", "framework", "model", "difficulty", "status",
    "passed", "findings", "cost_usd", "wall_time_s", "n_tool_calls", "n_steps",
    "f1_recall", "f1_precision", "f1",
)

_TRUTHY = {"1", "true", "yes", "pass", "passed", "y"}
_FALSY = {"0", "false", "no", "fail", "failed", "n"}


def _parse_bool(value) -> bool | None:
    """把 CSV 单元格解析为 bool；空/无法识别返回 None。"""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in _TRUTHY:
        return True
    if token in _FALSY:
        return False
    return None


def _to_float(value, default: float = 0.0) -> float:
    """把 CSV 单元格解析为 float；空/非法返回 default。"""
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_empty(value) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _read_results(run_dir: Path) -> list[dict]:
    """读 results.csv，按 CSV_COLUMNS 白名单清洗为行字典。"""
    path = run_dir / "results.csv"
    if not path.is_file():
        return []
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            row = {k: raw.get(k) for k in CSV_COLUMNS if raw.get(k) is not None}
            if not row.get("task_id"):
                continue  # 无 task_id 的行无意义，跳过
            rows.append(row)
    return rows


def _rows_from_traces(run_dir: Path) -> list[dict]:
    """results.csv 缺失时的兜底：仅凭 traces 计算 cost/latency 等列。"""
    traces_dir = run_dir / "traces"
    if not traces_dir.is_dir():
        return []
    rows: list[dict] = []
    for path in sorted(traces_dir.glob("*.jsonl")):
        trace = load_trace(path)
        if trace is None:
            continue
        rows.append({
            "task_id": trace.task_id,
            "framework": trace.framework,
            "model": trace.model,
            "status": trace.status,
            "cost_usd": total_cost(trace),
            "wall_time_s": wall_time(trace),
            "n_tool_calls": n_tool_calls(trace),
            "n_steps": len(trace.steps),
        })
    return rows


def _fill_from_trace(row: dict, trace) -> None:
    """用轨迹补 CSV 中的缺列（只在单元格为空时覆盖）。"""
    if _is_empty(row.get("framework")):
        row["framework"] = trace.framework
    if _is_empty(row.get("model")):
        row["model"] = trace.model
    if _is_empty(row.get("status")):
        row["status"] = trace.status
    if _is_empty(row.get("cost_usd")):
        row["cost_usd"] = total_cost(trace)
    if _is_empty(row.get("wall_time_s")):
        row["wall_time_s"] = wall_time(trace)
    if _is_empty(row.get("n_tool_calls")):
        row["n_tool_calls"] = n_tool_calls(trace)
    if _is_empty(row.get("n_steps")):
        row["n_steps"] = len(trace.steps)
    # F1 三列任一为空时按轨迹重算（§7.2 口径；任务/模块缺失则保持留空）
    if _compute_f1 is not None and any(
        _is_empty(row.get(k)) for k in ("f1_recall", "f1_precision", "f1")
    ):
        try:
            task = load_task(trace.task_id)
        except KeyError:
            return  # 任务库缺失：F1 列保持留空
        try:
            f1 = _compute_f1(task, trace)
            row["f1_recall"] = round(f1["recall"], 4)
            row["f1_precision"] = round(f1["precision"], 4)
            row["f1"] = round(f1["f1"], 4)
        except Exception as exc:
            # F1 兜底计算异常只影响 F1 列（保持留空），不阻断汇总
            print(f"[aggregate] F1 兜底计算失败 {trace.task_id}: {exc}")


def _fill_difficulty(row: dict) -> None:
    """difficulty 缺失时尝试按 task_id 从任务库补（找不到则留空）。"""
    if not _is_empty(row.get("difficulty")):
        return
    task_id = row.get("task_id")
    if not task_id:
        return
    try:
        task = load_task(task_id)
    except KeyError:
        return
    row["difficulty"] = task.get("difficulty", "")


def _first_nonempty(rows: list[dict], key: str) -> str:
    """取某列首个非空值；整列都空时返回空串。"""
    for row in rows:
        value = row.get(key)
        if not _is_empty(value):
            return str(value)
    return ""


def _aggregate(run_dir: Path) -> dict:
    """构建 summary（不写盘），供 aggregate_run_dir 与 CLI 复用。"""
    rows = _read_results(run_dir)
    if not rows:
        rows = _rows_from_traces(run_dir)  # results.csv 缺失/为空 → 仅 traces

    traces_dir = run_dir / "traces"
    for row in rows:
        trace = load_trace(traces_dir / f"{row.get('task_id', '')}.jsonl")
        if trace is not None:
            _fill_from_trace(row, trace)
        _fill_difficulty(row)

    n_tasks = len(rows)
    n_completed = 0
    n_passed = 0
    n_by_diff = {d: 0 for d in DIFFICULTIES}
    passed_by_diff = {d: 0 for d in DIFFICULTIES}
    failures: list[dict] = []

    for row in rows:
        difficulty = str(row.get("difficulty") or "")
        if difficulty in DIFFICULTIES:
            n_by_diff[difficulty] += 1
        status = str(row.get("status") or "").strip()
        if status.lower() == "completed":
            n_completed += 1
        if _parse_bool(row.get("passed")) is True:
            n_passed += 1
            if difficulty in DIFFICULTIES:
                passed_by_diff[difficulty] += 1
        else:
            # 未通过：判定失败（passed=false）或未完成（passed 缺失）
            failures.append({
                "task_id": str(row.get("task_id") or ""),
                "difficulty": difficulty,
                "status": status,
                "findings": str(row.get("findings") or ""),
            })

    costs = [_to_float(row.get("cost_usd")) for row in rows]
    walls = [_to_float(row.get("wall_time_s")) for row in rows]
    calls = [_to_float(row.get("n_tool_calls")) for row in rows]
    # F1：只对非空单元格求均值（老 results.csv 行无 F1 列时由 trace 补算或跳过）
    f1_values = [_to_float(row.get("f1")) for row in rows
                 if not _is_empty(row.get("f1"))]
    recall_values = [_to_float(row.get("f1_recall")) for row in rows
                     if not _is_empty(row.get("f1_recall"))]
    precision_values = [_to_float(row.get("f1_precision")) for row in rows
                        if not _is_empty(row.get("f1_precision"))]

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    summary = {
        "group": _first_nonempty(rows, "group"),
        "framework": _first_nonempty(rows, "framework"),
        "model": _first_nonempty(rows, "model"),
        "n_tasks": n_tasks,
        "n_completed": n_completed,
        "n_passed": n_passed,
        "sr": round(n_passed / n_tasks, 4) if n_tasks else 0.0,
        "mean_cost_usd": round(mean(costs), 6),
        "total_cost_usd": round(sum(costs), 6),
        "mean_wall_time_s": round(mean(walls), 3),
        "mean_tool_calls": round(mean(calls), 3),
        "mean_f1": round(mean(f1_values), 4),
        "mean_f1_recall": round(mean(recall_values), 4),
        "mean_f1_precision": round(mean(precision_values), 4),
        "by_difficulty": {
            d: {
                "n": n_by_diff[d],
                "sr": round(passed_by_diff[d] / n_by_diff[d], 4) if n_by_diff[d] else 0.0,
            }
            for d in DIFFICULTIES
        },
        "failures": failures,
    }
    return summary


def _write_summary(run_dir: Path, summary: dict) -> Path:
    """把 summary 落盘到 run_dir/summary.json。"""
    path = run_dir / "summary.json"
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def aggregate_run_dir(run_dir: Path) -> dict:
    """汇总一个 run 目录：构建 summary、写 summary.json 并返回。"""
    summary = _aggregate(Path(run_dir))
    _write_summary(Path(run_dir), summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI 入口：python -m metrics.aggregate <run_dir>。

    先打印 summary，再写入 run_dir/summary.json。
    """
    parser = argparse.ArgumentParser(
        description="汇总一次 AgentEval 运行（run 目录）并产出 summary.json。",
    )
    parser.add_argument("run_dir", type=Path, help="run 目录（含 results.csv 与 traces/）")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    summary = _aggregate(run_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    _write_summary(run_dir, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
