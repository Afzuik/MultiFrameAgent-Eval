"""analysis.failure_modes —— 失败模式分类 v0（《项目方案.md》§10.1）。

把一次失败运行归入六类先验假设的失败模式（实验后依据分布修正）：

- permission_violation（权限/规则违反）：findings 含 forbidden_call:，
  或 observation 报 ok:false 且 code ∈ {forbidden, auth_required}；
- parse_failure（解析失败）：observation 以"解析失败"开头（react 口径）；
- budget_exhausted（预算耗尽）：status ∈ {timeout, budget_exceeded}；
- arg_error（参数错误）：missing_call:{tool} 且该 tool 在真实调用中，
  或 observation 报参数类错误码；
- tool_selection_error（工具选择错误）：compute_f1 的 precision < 1.0；
- planning_failure（规划失败）：以上都未命中的失败运行（兜底）。

分类优先级固定：permission → parse → budget → arg → tool_selection →
planning，先命中先返回；passed 为 True 的运行一律不归类（返回 None）。

v0 说明：final_state 在真实评测管道中由 mock 服务提供，本 v0 规则未
引用终态（规则里只依赖 findings / trace / status）；调用方不可得终态时
传 {} 即可，个别依赖终态信号的类别可能因此降级为 planning_failure，
属可接受退化（实验后修正）。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from harness.protocol import Trace, load_trace
from metrics.success import load_task
from metrics.tool_f1 import compute_f1

__all__ = [
    "CATEGORIES",
    "CATEGORY_LABELS",
    "build_distribution",
    "classify_run",
    "main",
]

# 六类失败模式（§10.1 先验体系；顺序为展示顺序，非判定优先级）
CATEGORIES = (
    "permission_violation",
    "parse_failure",
    "budget_exhausted",
    "tool_selection_error",
    "arg_error",
    "planning_failure",
)

# 类别 → 中文标签（报告 / dashboard 展示用）
CATEGORY_LABELS = {
    "permission_violation": "权限/规则违反",
    "parse_failure": "解析失败",
    "budget_exhausted": "预算耗尽",
    "tool_selection_error": "工具选择错误",
    "arg_error": "参数错误",
    "planning_failure": "规划失败",
}

# observation 错误响应 {"ok": false, "error": "...", "code": "..."} 中的权限类错误码
_PERMISSION_CODES = frozenset({"forbidden", "auth_required"})
# 参数取值/格式类错误码（工具服务侧显式佐证"参数错误"）
_ARG_ERROR_CODES = frozenset(
    {
        "invalid_date",
        "invalid_code",
        "class_mismatch",
        "address_too_short",
        "invalid_sql",
        "invalid_table",
        "route_not_found",
    }
)
# 预算耗尽对应的 status 取值（§6.4）
_BUDGET_STATUSES = frozenset({"timeout", "budget_exceeded"})
# react 适配器解析失败回填口径（harness.adapters.react，observation 内容前缀）
_PARSE_FAILURE_PREFIX = "解析失败"

# results.csv 中 passed 单元格的常见写法
_TRUTHY = {"1", "true", "yes", "pass", "passed", "y"}
_FALSY = {"0", "false", "no", "fail", "failed", "n"}


def _parse_bool(value: Any) -> bool | None:
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


def _iter_observation_json(trace: Any):
    """逐个产出可解析为 dict 的 observation 步骤内容（解析失败跳过）。"""
    for step in getattr(trace, "steps", []) or []:
        if getattr(step, "type", None) != "observation":
            continue
        content = str(getattr(step, "content", "") or "")
        try:
            payload = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict):
            yield payload


def classify_run(
    task: dict,
    trace: Any,
    final_state: dict,
    passed: bool,
    findings: list[str],
    status: str,
) -> str | None:
    """把一次失败运行归入六类失败模式之一（§10.1，先命中先返回）。

    判定优先级：permission_violation → parse_failure → budget_exhausted →
    arg_error → tool_selection_error → planning_failure（兜底）。
    passed 为 True 的运行不归类（返回 None）——成功运行不是失败样本。

    Args:
        task: 任务定义（含 ground_truth_plan / domain）。
        trace: 归一化轨迹（用于扫描 observation 与真实调用）。
        final_state: mock 服务终态；v0 规则未引用，调用方不可得时传 {}。
        findings: verifier findings（如 ["missing_call:book_flight"]）。
        status: 运行状态（completed / timeout / budget_exceeded / ...）。

    Returns:
        类别常量之一；passed 为 True 时返回 None。
    """
    if passed:
        return None
    findings = [str(f) for f in (findings or [])]

    # 1) 权限/规则违反：forbidden_calls 命中，或工具服务显式
    #    报 ok:false + code ∈ {forbidden, auth_required}
    for finding in findings:
        if finding.startswith("forbidden_call:"):
            return "permission_violation"
    for payload in _iter_observation_json(trace):
        if payload.get("ok") is False and payload.get("code") in _PERMISSION_CODES:
            return "permission_violation"

    # 2) 解析失败：observation 内容以"解析失败"开头（react 适配器口径）
    for step in getattr(trace, "steps", []) or []:
        content = str(getattr(step, "content", "") or "")
        if getattr(step, "type", None) == "observation" and content.startswith(
            _PARSE_FAILURE_PREFIX
        ):
            return "parse_failure"

    # 3) 预算耗尽：超时 / 步数或成本超限（status 口径）
    if str(status or "").strip().lower() in _BUDGET_STATUSES:
        return "budget_exhausted"

    # 4) 参数错误：某 required_calls 工具出现过但参数不匹配
    #    （findings 含 missing_call:{tool} 且该 tool 在真实调用中），
    #    或 observation 显式报参数类错误码
    call_tools = {c.get("tool") for c in trace.tool_calls()}
    for finding in findings:
        if finding.startswith("missing_call:") and finding.split(":", 1)[1] in call_tools:
            return "arg_error"
    for payload in _iter_observation_json(trace):
        if payload.get("code") in _ARG_ERROR_CODES:
            return "arg_error"

    # 5) 工具选择错误：compute_f1 的 precision < 1.0（域外工具/参数不完整）
    f1 = compute_f1(task, trace)
    if f1["precision"] < 1.0:
        return "tool_selection_error"

    # 6) 规划失败：兜底——以上规则都未命中的失败运行
    return "planning_failure"


def _read_rows(results_csv: Path) -> list[dict]:
    """读 results.csv（DictReader 保列名）；文件缺失/为空返回 []。"""
    if not results_csv.is_file():
        return []
    rows: list[dict] = []
    with results_csv.open(newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            if raw.get("task_id") is not None and str(raw["task_id"]).strip():
                rows.append(raw)
    return rows


def build_distribution(run_dir: Path) -> dict:
    """对 run 目录的全部失败运行做失败模式分类（§10.2 分布统计）。

    输入（§11.1 目录约定）：
    - results.csv：逐任务结果行（passed 可为 "True"/"False"/"1"/"0"；
      findings 与编排器一致用 ";" 分隔多条）；
    - traces/{task_id}.jsonl：归一化轨迹（protocol.load_trace）；
    - 任务定义：metrics.success.load_task 按 task_id 定位。

    每个 failed 运行调 classify_run；final_state 不可得时传 {}（分类优先
    用 trace/findings/status 信号，个别类别可能降级为 planning_failure，
    可接受）。task_id 不在任务库的行无法分类，跳过（数据错误）。

    Returns:
        {"total", "failed", "by_category", "by_category_pct",
         "by_framework", "details"}：
        - by_category：{类别: 运行数}，全部六类键齐全；
        - by_category_pct：{类别: 占比百分数}（failed==0 时全 0.0）；
        - by_framework：{framework: {"failed": n, "by_category": {...}}}；
        - details：[{task_id, framework, difficulty, category, status,
                    findings}]。
    """
    rows = _read_rows(run_dir / "results.csv")
    traces_dir = run_dir / "traces"

    by_category: dict[str, int] = {c: 0 for c in CATEGORIES}
    by_framework: dict[str, dict[str, Any]] = {}
    details: list[dict[str, Any]] = []
    failed = 0

    for row in rows:
        if _parse_bool(row.get("passed")) is True:
            continue  # 成功运行不入失败分布
        task_id = str(row.get("task_id") or "").strip()
        try:
            task = load_task(task_id)
        except KeyError:
            continue  # task_id 不在任务库 → 无法分类，跳过该数据错误行
        failed += 1

        # 轨迹缺失兜底：仅凭行级 findings/status 分类
        trace = load_trace(traces_dir / f"{task_id}.jsonl")
        if trace is None:
            trace = Trace(
                run_id="",
                task_id=task_id,
                framework=str(row.get("framework") or ""),
                model=str(row.get("model") or ""),
            )
        framework = str(row.get("framework") or "").strip() or (trace.framework or "")
        difficulty = str(row.get("difficulty") or "").strip() or str(task.get("difficulty") or "")
        status = str(row.get("status") or "").strip() or (trace.status or "")
        findings = [f for f in str(row.get("findings") or "").split(";") if f]

        category = classify_run(task, trace, {}, False, findings, status)
        by_category[category] += 1
        fb = by_framework.setdefault(
            framework,
            {
                "failed": 0,
                "by_category": {c: 0 for c in CATEGORIES},
            },
        )
        fb["failed"] += 1
        fb["by_category"][category] += 1
        details.append(
            {
                "task_id": task_id,
                "framework": framework,
                "difficulty": difficulty,
                "category": category,
                "status": status,
                "findings": findings,
            }
        )

    by_category_pct = {
        c: round(100.0 * by_category[c] / failed, 2) if failed else 0.0 for c in CATEGORIES
    }
    return {
        "total": len(rows),
        "failed": failed,
        "by_category": by_category,
        "by_category_pct": by_category_pct,
        "by_framework": by_framework,
        "details": details,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI 入口：python -m analysis.failure_modes <run_dir>。

    打印失败模式分布 JSON，并写 run_dir/failure_modes.json。
    """
    parser = argparse.ArgumentParser(
        prog="python -m analysis.failure_modes",
        description="统计一次 AgentEval 运行（run 目录）的失败模式分布。",
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="run 目录（含 results.csv 与 traces/）",
    )
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    dist = build_distribution(run_dir)
    print(json.dumps(dist, ensure_ascii=False, indent=2))
    out_path = run_dir / "failure_modes.json"
    out_path.write_text(
        json.dumps(dist, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
