"""AgentEval 声明式 verifier 引擎。

从任务 JSON 的四类检查原语（required_calls / forbidden_calls /
final_state_checks / answer_checks）直接判定 pass/fail，
无需为每个任务手写 Python 判定函数。

运行时用法（harness 调用）：
    passed, findings = verify_task(task, calls, final_state, answer)

自检用法（设计期，验证任务 JSON 自身一致性）：
    python -m tasks.validate
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TASKS_DIR = Path(__file__).resolve().parent / "v1"


def _deep_get(state: dict, path: str) -> Any:
    """按点分路径取值；路径不存在返回 None。"""
    cur: Any = state
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def args_contain(args: dict, expected: dict) -> bool:
    """参数包含匹配：expected 中每个键都存在，且字符串值做子串匹配、其他值做相等匹配。"""
    for key, value in expected.items():
        if key not in args:
            return False
        if isinstance(value, str):
            if value not in str(args[key]):
                return False
        elif args[key] != value:
            return False
    return True


def has_call(calls: list[dict], tool: str, expected_args: dict | None = None) -> bool:
    for call in calls:
        if call.get("tool") != tool:
            continue
        if expected_args is None or args_contain(call.get("args", {}), expected_args):
            return True
    return False


def verify_task(
    task: dict,
    calls: list[dict],
    final_state: dict,
    answer: str,
) -> tuple[bool, list[str]]:
    """执行四类检查原语，返回 (passed, findings)。"""
    findings: list[str] = []

    for rc in task.get("required_calls", []):
        if not has_call(calls, rc["tool"], rc.get("args_contain")):
            findings.append(f"missing_call:{rc['tool']}")

    for fc in task.get("forbidden_calls", []):
        if has_call(calls, fc["tool"], fc.get("args_contain")):
            findings.append(f"forbidden_call:{fc['tool']}")

    for sc in task.get("final_state_checks", []):
        value = _deep_get(final_state, sc["path"])
        if "equals" in sc and value != sc["equals"]:
            findings.append(
                f"state_mismatch:{sc['path']} expected={sc['equals']!r} got={value!r}"
            )
        elif sc.get("exists") and value is None:
            findings.append(f"state_missing:{sc['path']}")

    for ac in task.get("answer_checks", []):
        if "contains" in ac and ac["contains"] not in answer:
            findings.append(f"answer_missing:{ac['contains']}")
        if "not_contains" in ac and ac["not_contains"] in answer:
            findings.append(f"answer_forbidden:{ac['not_contains']}")
        if "any_of" in ac and not any(item in answer for item in ac["any_of"]):
            findings.append(f"answer_any_of_missing:{ac['any_of']}")

    return (not findings, findings)


def gt_calls(task: dict) -> list[dict]:
    """把 ground_truth_plan 转成 calls 结构。"""
    return [{"tool": step["tool"], "args": step.get("args", {})} for step in task["ground_truth_plan"]]


def self_check_file(path: Path) -> tuple[int, int, list[str]]:
    """对单个任务文件执行双向自检：空轨迹必须 fail，GT 必须 pass。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    passed, failed, problems = 0, 0, []
    for task in data["tasks"]:
        tid = task["task_id"]

        empty_passed, _ = verify_task(task, [], task.get("initial_state", {}), "")
        if empty_passed:
            problems.append(f"{tid}: 空轨迹竟然 PASS（verifier 防假通过失败）")
            failed += 1
        else:
            passed += 1

        gt_passed, gt_findings = verify_task(
            task, gt_calls(task), task.get("gt_final_state", {}), task.get("gt_answer", "")
        )
        if not gt_passed:
            problems.append(f"{tid}: GT 路径 FAIL -> {gt_findings}")
            failed += 1
        else:
            passed += 1
    return passed, failed, problems


def main() -> int:
    total_passed, total_failed = 0, 0
    all_problems: list[str] = []
    for path in sorted(TASKS_DIR.glob("*_tasks.json")):
        passed, failed, problems = self_check_file(path)
        total_passed += passed
        total_failed += failed
        all_problems.extend(problems)
        print(f"{path.name}: 双向自检 {passed} 项通过, {failed} 项失败")
    for problem in all_problems:
        print("  [FAIL]", problem)
    print(f"\n总计: {total_passed} 项通过 / {total_failed} 项失败")
    return 1 if total_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
