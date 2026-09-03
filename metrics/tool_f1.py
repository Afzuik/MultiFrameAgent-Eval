"""metrics.tool_f1 —— Tool-Call F1（《项目方案.md》§7.2 口径）。

以 ground_truth_plan 的"关键调用"为对比单位，衡量一次运行里真实工具
调用的查全率与查准率：

- Recall：GT 中每个关键调用，在真实调用中是否存在"同工具 +
  参数包含匹配"（tasks.verifiers.args_contain：expected 每键存在，
  字符串值子串匹配、其余值相等）→ 命中数 / GT 总数；
- Precision：真实轨迹的每次调用是否"合法"——工具名属于
  GT 工具集 ∪ 任务所在域的允许工具集（protocol.tool_specs_for_domain），
  且该工具在 tool_registry.json 中 parameters.required 的全部键都出现在
  args 中 → 合法数 / 真实调用总数；
- F1 = 2PR/(P+R)；P+R == 0 时取 0.0。

口径说明（§7.2，对多解任务公平）：
- Recall 只看关键依赖：出现过等价的关键调用即算命中，不要求顺序一致、
  不要求 GT 与真实调用一一对应；
- Precision 允许等价工具集：同域内"换一种合法工具达成目标"不扣分，
  只惩罚域外工具与参数不完整的调用（冗余但参数完整的不扣分）。

边界约定（本任务集 GT 均非空，以下仅为函数完备性兜底）：
- 空调用列表：Recall 取 0.0；Precision 取 1.0（没有真实调用就没有非法
  调用可扣）→ F1 恒为 0.0；
- GT 为空：Recall 取 1.0（空集上没有可漏的关键调用）。
"""

from __future__ import annotations

from typing import Any

from harness import protocol
from tasks.verifiers import args_contain

__all__ = ["compute_f1", "load_allowed_tools"]


def load_allowed_tools(domain: str) -> set[str]:
    """从工具注册表（tool_registry.json）取某业务域的域内工具名集合。

    即 Precision 定义中"允许工具集"的静态来源（与 protocol 读同一文件）。
    """
    registry = protocol.load_tool_registry()
    return {t["name"] for t in registry.get("tools", {}).get(domain, [])}


def compute_f1(task: dict, trace: Any) -> dict:
    """计算一次运行的 Tool-Call F1（§7.2）。

    Args:
        task: 任务定义（tasks/v1/*_tasks.json 的单个元素，须含 domain 与
            ground_truth_plan）。
        trace: 归一化轨迹（harness.protocol.Trace，经 tool_calls() 取调用）。

    Returns:
        recall/precision/f1：0~1 浮点（原始比值，不做显示级四舍五入——
        展示层如 metrics.aggregate / harness.orchestrator 自行 round）；
        f1 在 P+R==0 时取 0.0；
        gt_calls/matched_gt/n_real_calls/n_valid_calls：计数；
        invalid_calls：不合法调用的工具名列表（保持真实调用顺序）。
    """
    # ---- GT 关键调用（口径以 ground_truth_plan 为准，而非 required_calls）----
    gt_calls: list[dict] = [
        {"tool": step["tool"], "args": step.get("args") or {}}
        for step in task.get("ground_truth_plan") or []
        if isinstance(step, dict) and step.get("tool")
    ]
    calls: list[dict] = trace.tool_calls()

    # ---- Recall：GT 每个关键调用是否出现"同工具 + 参数包含匹配" ----
    matched_gt = 0
    for gt in gt_calls:
        if any(
            call.get("tool") == gt["tool"] and args_contain(call.get("args") or {}, gt["args"])
            for call in calls
        ):
            matched_gt += 1
    if gt_calls:
        recall = matched_gt / len(gt_calls)
    else:
        recall = 1.0  # GT 为空：没有可漏的关键调用（本任务集不会出现）

    # ---- Precision：真实调用是否合法（GT ∪ 域内允许工具 + 参数完整）----
    domain = task.get("domain", "")
    allowed: set[str] = set()
    required: dict[str, set[str]] = {}
    for spec in protocol.tool_specs_for_domain(domain):
        allowed.add(spec["name"])
        params = spec.get("parameters") or {}
        required[spec["name"]] = set(params.get("required") or [])
    allowed |= {gt["tool"] for gt in gt_calls}

    n_real = len(calls)
    n_valid = 0
    invalid_calls: list[str] = []
    for call in calls:
        tool = call.get("tool") or ""
        args = call.get("args") or {}
        if tool not in allowed or not (required.get(tool) or set()).issubset(args):
            invalid_calls.append(tool)
            continue
        n_valid += 1
    if n_real:
        precision = n_valid / n_real
    else:
        precision = 1.0  # 空调用列表：没有真实调用就没有非法调用可扣

    # ---- F1 = 2PR/(P+R)；P+R==0 时取 0.0 ----
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "gt_calls": len(gt_calls),
        "matched_gt": matched_gt,
        "n_real_calls": n_real,
        "n_valid_calls": n_valid,
        "invalid_calls": invalid_calls,
    }
