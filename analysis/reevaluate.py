"""analysis.reevaluate —— 历史 run 目录的 v1.1 离线复评（不调用模型）。

背景：v1.1 修正了 verifier 的 answer_checks（数字归一化匹配、措辞宽松），
历史实验（如 R1/S1）是按旧 verifier 判定的。公平对比新旧 backbone 需要把
历史结果用同一把（新）尺子重新判一遍——本模块只做离线复评：

- 只复评 findings 全部为 answer_* 的失败行（调用/终态检查与 v1.1 无关，
  已通过的行不受影响）；
- 用当前 verifier 对 trace.final_answer 重新执行 answer_checks；
- 新判定通过的行记入 corrected_pass；其余失败保持原判定。

口径保证：绝不触碰 required_calls/forbidden_calls/final_state_checks——
那些检查在旧判定中已通过（findings 里没有它们），无需重判。

CLI：python -m analysis.reevaluate <run_dir>
产出：run_dir/reevaluation_v11.json + 打印复评摘要（原 SR → 复评 SR）。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from harness.protocol import load_trace
from metrics.success import load_task
from tasks.verifiers import verify_answer_checks

__all__ = ["main", "reevaluate_run_dir"]


def _parse_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "pass", "passed", "y"}:
        return True
    if token in {"0", "false", "no", "fail", "failed", "n"}:
        return False
    return None


def _answer_only(findings: list[str]) -> bool:
    """findings 全部是 answer_* 类（可离线复评）才返回 True。"""
    return bool(findings) and all(f.startswith("answer_") for f in findings)


def reevaluate_run_dir(run_dir: Path) -> dict:
    """离线复评一个 run 目录，返回复评摘要并写 reevaluation_v11.json。"""
    run_dir = Path(run_dir)
    results_csv = run_dir / "results.csv"
    traces_dir = run_dir / "traces"
    rows = list(csv.DictReader(results_csv.open(newline="", encoding="utf-8")))

    n_total = len(rows)
    n_passed = 0
    corrected: list[dict] = []
    remaining_failures: list[dict] = []

    for row in rows:
        passed = _parse_bool(row.get("passed"))
        if passed is True:
            n_passed += 1
            continue
        task_id = str(row.get("task_id") or "").strip()
        findings = [f for f in str(row.get("findings") or "").split(";") if f]
        trace = load_trace(traces_dir / f"{task_id}.jsonl")
        try:
            task = load_task(task_id)
        except KeyError:
            remaining_failures.append({"task_id": task_id, "reason": "task_missing"})
            continue
        if trace is None:
            remaining_failures.append({"task_id": task_id, "reason": "trace_missing"})
            continue

        # 只有 findings 全为 answer_* 才离线复评：用当前 verifier 单独重判答案维度
        # （绝不重判调用/终态——它们在旧判定中已通过，且终态未持久化无法重算）
        if _answer_only(findings):
            new_findings = verify_answer_checks(task, trace.final_answer)
            if not new_findings:
                n_passed += 1
                corrected.append({"task_id": task_id, "old_findings": findings})
                continue
        remaining_failures.append({"task_id": task_id, "findings": findings})

    summary = {
        "run_dir": str(run_dir),
        "n_tasks": n_total,
        "original_passed": sum(
            1 for r in rows if _parse_bool(r.get("passed")) is True
        ),
        "reevaluated_passed": n_passed,
        "original_sr": round(
            sum(1 for r in rows if _parse_bool(r.get("passed")) is True) / n_total, 4
        ) if n_total else 0.0,
        "reevaluated_sr": round(n_passed / n_total, 4) if n_total else 0.0,
        "corrected_pass": corrected,
        "remaining_failures": remaining_failures,
    }
    out = run_dir / "reevaluation_v11.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m analysis.reevaluate",
        description="历史 run 目录 v1.1 离线复评（不调用模型，只重判 answer 维度）",
    )
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    summary = reevaluate_run_dir(args.run_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
