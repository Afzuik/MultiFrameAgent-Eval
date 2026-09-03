"""analysis.report —— 跨 run 汇总的 markdown 实验报告生成器（W3）。

对应《项目方案.md》§9.3 的报告章节结构（主表 / 分难度表 / 失败模式分布 /
成本效率散点 / 案例研究 / judge 质量）与 §10.2 失败模式产出；模块路径遵循
§12 目录约定（analysis/report.py）。

数据源（全部只读，除下方两个"缺失时现场生成"分支）：
- run_dir/summary.json：metrics.aggregate 产出（schema 见 metrics/aggregate.py，
  本模块不得修改，仅读取其产出）；
- run_dir/failure_modes.json：analysis.failure_modes 产出（复用 CATEGORIES 与
  CATEGORY_LABELS 中文标签）；
- run_dir/results.csv：案例研究挑成功样本（passed=True 且 wall_time_s 最大）
  与失败样本耗时；
- run_dir/judge_scores.json（可选）：judge 质量占位（agreement_exact /
  agreement_pm1 / mean_score，W3/W4 由 judge 模块落地）。

缺失容错：
- summary.json / failure_modes.json 缺失时先调用 metrics.aggregate.aggregate_run_dir
  与 analysis.failure_modes.build_distribution 现场生成并写盘；
- summary 缺某 key 时对应表格单元格显示 "-"。数值显示用 round(value, n) 与
  aggregate 存储精度对齐：sr/f1 4 位、成本 6 位、耗时/工具调用 3 位
  （aggregate 对 sr/mean_f1*/mean_cost_usd/mean_wall_time_s/mean_tool_calls
  分别 round 到 4/6/3 位）；案例耗时与 judge 数值为展示用途，round 2 位。

build_report(run_dirs, out_path=None) -> str：返回整份 markdown 文本；
out_path 非空时写盘并返回路径文本。
CLI：python -m analysis.report <run_dir...> [-o/--out FILE]，默认打印到 stdout。
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from analysis.failure_modes import CATEGORIES, CATEGORY_LABELS, build_distribution
from metrics.aggregate import aggregate_run_dir

__all__ = ["build_report", "main"]

# results.csv passed 单元格的常见写法（兼容 "True"/"False" 及 bool/数字等）
_TRUTHY = {"1", "true", "yes", "pass", "passed", "y"}
_FALSY = {"0", "false", "no", "fail", "failed", "n"}

# judge_scores.json 中要读取的键（顶层优先、嵌套兜底）
_JUDGE_KEYS = ("agreement_exact", "agreement_pm1", "mean_score")


# ---------- 基础取值 / 格式化 ----------


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


def _num(value: Any) -> float | None:
    """float(value)；None / 空串 / 非法文本返回 None。"""
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _txt(value: Any) -> str:
    """字符串单元格：空 / None → "-"。"""
    if value is None:
        return "-"
    text = str(value).strip()
    return text if text else "-"


def _fmt(value: Any, ndigits: int) -> str:
    """数值单元格：round(value, ndigits) 后的文本；缺失 / 非法 → "-"。"""
    number = _num(value)
    if number is None:
        return "-"
    return str(round(number, ndigits))


def _int(value: Any) -> int:
    """数值单元格转 int；缺失 / 非法 → 0。"""
    number = _num(value)
    return int(number) if number is not None else 0


def _row(cells: list[str]) -> str:
    """把单元格拼成 markdown 表格行。"""
    return "| " + " | ".join(cells) + " |"


# ---------- run 数据加载 ----------


def _load_summary(run_dir: Path) -> dict:
    """读 run_dir/summary.json；缺失时调用 aggregate 现场生成（并写盘）。"""
    path = run_dir / "summary.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return aggregate_run_dir(run_dir)


def _load_failure_modes(run_dir: Path) -> dict:
    """读 run_dir/failure_modes.json；缺失时调用 build_distribution 生成并落盘。"""
    path = run_dir / "failure_modes.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    dist = build_distribution(run_dir)
    path.write_text(json.dumps(dist, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return dist


def _read_results_csv(run_dir: Path) -> list[dict]:
    """读 run_dir/results.csv；文件缺失返回 []（案例研究用）。"""
    path = run_dir / "results.csv"
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f)]


def _dig(data: Any, key: str) -> Any:
    """在 judge JSON 中按 key 取值：顶层优先，dict/list 嵌套递归兜底。"""
    if isinstance(data, dict):
        if key in data and data[key] is not None:
            return data[key]
        for value in data.values():
            found = _dig(value, key)
            if found is not None:
                return found
        return None
    if isinstance(data, list):
        for value in data:
            found = _dig(value, key)
            if found is not None:
                return found
        return None
    return None


# ---------- 各章节渲染 ----------


def _f1_cell(summary: dict) -> str:
    """mean F1(recall/precision) 单元格；mean_f1 缺失时整格 "-"。"""
    f1 = _fmt(summary.get("mean_f1"), 4)
    if f1 == "-":
        return "-"
    recall = _fmt(summary.get("mean_f1_recall"), 4)
    precision = _fmt(summary.get("mean_f1_precision"), 4)
    return f"{f1} ({recall}/{precision})"


def _n_failed(summary: dict) -> str:
    """失败运行数：优先取 summary.failures 条数，退化用 n_tasks - n_passed。"""
    failures = summary.get("failures")
    if isinstance(failures, list):
        return str(len(failures))
    return str(max(_int(summary.get("n_tasks")) - _int(summary.get("n_passed")), 0))


def _diff_cell(summary: dict, difficulty: str) -> str:
    """分难度 SR 单元格；by_difficulty / 该难度条目缺失时显示 "-"。"""
    table = summary.get("by_difficulty")
    if not isinstance(table, dict):
        return "-"
    entry = table.get(difficulty)
    if not isinstance(entry, dict):
        return "-"
    return _fmt(entry.get("sr"), 4)


def _render_main_table(runs: list[Path], summaries: list[dict]) -> list[str]:
    """§9.3-1 主表：每个 run 一行，SR / F1 / 成本 / 延迟 / 工具调用 / 失败数。"""
    header = [
        "run",
        "group",
        "framework",
        "model",
        "SR",
        "mean F1 (recall/precision)",
        "mean cost",
        "mean latency",
        "mean tool calls",
        "失败数",
    ]
    lines = ["## 1. 主表", "", _row(header), _row(["---"] * len(header))]
    for run, summary in zip(runs, summaries):
        lines.append(
            _row(
                [
                    run.name,
                    _txt(summary.get("group")),
                    _txt(summary.get("framework")),
                    _txt(summary.get("model")),
                    _fmt(summary.get("sr"), 4),
                    _f1_cell(summary),
                    _fmt(summary.get("mean_cost_usd"), 6),
                    _fmt(summary.get("mean_wall_time_s"), 3),
                    _fmt(summary.get("mean_tool_calls"), 3),
                    _n_failed(summary),
                ]
            )
        )
    lines.append("")
    return lines


def _render_difficulty_table(runs: list[Path], summaries: list[dict]) -> list[str]:
    """§9.3-2 分难度表：每个 run 的 L1/L2/L3 SR。"""
    lines = ["## 2. 分难度 SR", "", _row(["run", "L1", "L2", "L3"]), _row(["---"] * 4)]
    for run, summary in zip(runs, summaries):
        lines.append(
            _row(
                [
                    run.name,
                    _diff_cell(summary, "L1"),
                    _diff_cell(summary, "L2"),
                    _diff_cell(summary, "L3"),
                ]
            )
        )
    lines.append("")
    return lines


def _framework_counts(runs: list[Path], summaries: list[dict]):
    """合并各 run 的 by_category → ({framework: {category: n}}, {framework: [run 名]})。

    优先按 failure_modes.by_framework 分框架归并（一个 run 目录理论上只含一个
    框架；若含多框架也不串数）；by_framework 缺失时退回把 by_category 整体记到
    summary.framework 名下。
    """
    merged: dict[str, dict[str, int]] = {}
    fw_runs: dict[str, list[str]] = {}
    for run, summary in zip(runs, summaries):
        dist = _load_failure_modes(run)
        per_fw = dist.get("by_framework")
        if isinstance(per_fw, dict) and per_fw:
            items = list(per_fw.items())
        else:
            framework = _txt(summary.get("framework"))
            items = [
                (
                    framework if framework != "-" else "?",
                    {"by_category": dist.get("by_category", {})},
                )
            ]
        for framework, payload in items:
            counts = payload.get("by_category", {}) if isinstance(payload, dict) else {}
            if not isinstance(counts, dict):
                counts = {}
            bucket = merged.setdefault(framework, {c: 0 for c in CATEGORIES})
            for category in CATEGORIES:
                bucket[category] += _int(counts.get(category))
            fw_runs.setdefault(framework, []).append(run.name)
    return merged, fw_runs


def _render_failure_matrix(runs: list[Path], summaries: list[dict]) -> list[str]:
    """§10.2 失败模式分布：合并 by_category，框架 × 类别矩阵（中文标签）。"""
    lines = ["## 3. 失败模式分布（by_category 合并 · 按框架分组）", ""]
    merged, fw_runs = _framework_counts(runs, summaries)
    total_failed = sum(sum(bucket.values()) for bucket in merged.values())
    if total_failed == 0:
        lines.append("> 所有 run 均无失败样本（failed 合计 0），无分布可展示。")
        lines.append("")
        return lines

    frameworks = list(merged)
    header = ["失败模式"]
    header += [f"{fw}（{'/'.join(fw_runs[fw])}）" for fw in frameworks]
    header.append("合计")
    lines.append(_row(header))
    lines.append(_row(["---"] * len(header)))
    for category in CATEGORIES:
        label = CATEGORY_LABELS.get(category, category)
        counts = [merged[fw][category] for fw in frameworks]
        lines.append(_row([label] + [str(n) for n in counts] + [str(sum(counts))]))
    total_counts = [sum(merged[fw].values()) for fw in frameworks]
    lines.append(_row(["合计"] + [str(n) for n in total_counts] + [str(sum(total_counts))]))
    lines.append("")
    return lines


def _render_scatter(runs: list[Path], summaries: list[dict]) -> list[str]:
    """§9.3-4 成本效率散点（文本近似；dashboard 交互版在 W4）。"""
    lines = [
        "## 4. 成本效率散点（文本近似版）",
        "",
        (
            "> 交互式散点（SR × mean cost × mean latency，tooltip 含 F1）在 W4 的 "
            "dashboard 实现；此处先以文本表格近似。"
        ),
        "",
    ]
    header = ["run", "SR", "mean cost", "mean latency"]
    lines.append(_row(header))
    lines.append(_row(["---"] * len(header)))
    for run, summary in zip(runs, summaries):
        lines.append(
            _row(
                [
                    run.name,
                    _fmt(summary.get("sr"), 4),
                    _fmt(summary.get("mean_cost_usd"), 6),
                    _fmt(summary.get("mean_wall_time_s"), 3),
                ]
            )
        )
    lines.append("")
    return lines


def _wall_for(rows: list[dict], task_id: str) -> str:
    """在 results.csv 行里按 task_id 找耗时（展示用，round 2 位）；找不到 "-"。"""
    for row in rows:
        if str(row.get("task_id") or "") == str(task_id):
            wall = _num(row.get("wall_time_s"))
            return "-" if wall is None else f"{round(wall, 2)} s"
    return "-"


def _findings_text(value: Any) -> str:
    """findings 转单行文本：list 用 ";" 连接，空/缺失显示 "-"。"""
    if isinstance(value, list):
        joined = "; ".join(str(item) for item in value)
        return joined if joined else "-"
    return _txt(value)


def _pick_success(rows: list[dict]) -> tuple[float, dict] | None:
    """挑 results.csv 中 passed=True 且 wall_time_s 最大的行；(耗时, 行)。"""
    best: tuple[float, dict] | None = None
    for row in rows:
        if _parse_bool(row.get("passed")) is not True:
            continue
        wall = _num(row.get("wall_time_s"))
        if wall is None:
            continue
        if best is None or wall > best[0]:
            best = (wall, row)
    return best


def _render_case_studies(runs: list[Path], summaries: list[dict]) -> list[str]:
    """§9.3-5 案例研究占位：每 run 挑 1 失败（details 第一条）+ 1 成功样本摘要。

    只列 task_id / findings / 耗时；轨迹逐步分析（逐步归因与根因）留 W4
    人工 / LLM 深挖。
    """
    lines = ["## 5. 案例研究（占位：每 run 各挑 1 失败 + 1 成功样本）", ""]
    lines.append(
        "> 失败样本 = failure_modes.details 第一条；成功样本 = results.csv "
        "中 passed=True 且 wall_time_s 最大。"
    )
    lines.append("> 此处只列摘要；轨迹逐步分析留 W4 人工 / LLM 深挖。")
    lines.append("")
    for run, summary in zip(runs, summaries):
        lines.append(
            f"### {run.name}（{_txt(summary.get('group'))} / {_txt(summary.get('framework'))}）"
        )
        dist = _load_failure_modes(run)
        details = dist.get("details")
        if not isinstance(details, list) or not details:
            lines.append("- 失败案例：无（failed=0 或无 details）")
        else:
            detail = details[0] if isinstance(details[0], dict) else {}
            task_id = _txt(detail.get("task_id"))
            category = str(detail.get("category") or "")
            label = CATEGORY_LABELS.get(category, category or "-")
            status = _txt(detail.get("status"))
            rows = _read_results_csv(run)
            lines.append(
                f"- 失败案例：`{task_id}` ｜ 类别：{label}（{category or '-'}）"
                f"｜ status：{status} ｜ 耗时：{_wall_for(rows, task_id)}"
            )
            lines.append(f"  - findings：{_findings_text(detail.get('findings'))}")
        rows = _read_results_csv(run)
        best = _pick_success(rows)
        if best is None:
            lines.append("- 成功案例：无（results.csv 无 passed=True 行或文件缺失）")
        else:
            wall, row = best
            lines.append(
                f"- 成功案例：`{_txt(row.get('task_id'))}` ｜ 耗时："
                f"{round(wall, 2)} s（passed=True 且 wall_time_s 最大）"
            )
            lines.append("  - findings：-")
        lines.append("")
    return lines


def _render_judge(runs: list[Path], summaries: list[dict]) -> list[str]:
    """§9.3-6 judge 质量占位：有 judge_scores.json 则列一致性，否则标注未运行。"""
    lines = ["## 6. judge 质量（占位：双 judge 一致性 / 均值，§8.3）", ""]
    for run, summary in zip(runs, summaries):
        tag = f"{run.name}（{_txt(summary.get('framework'))}）"
        path = run / "judge_scores.json"
        if not path.is_file():
            lines.append(f"- {tag}：judge：未运行 judge（judge_scores.json 缺失）")
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - 坏文件容错
            lines.append(f"- {tag}：judge：judge_scores.json 解析失败（{exc}）")
            continue
        values = {key: _fmt(_dig(data, key), 2) for key in _JUDGE_KEYS}
        detail = "、".join(f"{key}={values[key]}" for key in _JUDGE_KEYS)
        lines.append(f"- {tag}：judge：{detail}")
    lines.append("")
    return lines


# ---------- 入口 ----------


def build_report(
    run_dirs: list[Path],
    out_path: Path | None = None,
) -> str:
    """汇总多个 run 目录为 markdown 实验报告并返回文本。

    Args:
        run_dirs: run 目录列表（每个含 summary.json 与 failure_modes.json；
            缺失时现场调用 aggregate / failure_modes 生成）。
        out_path: 非空时把报告写到此文件并返回路径文本；缺省只返回报告全文。

    Returns:
        out_path 为空时返回整份 markdown 文本；非空时写盘并返回文件路径文本。
    """
    runs = [Path(path) for path in run_dirs]
    if not runs:
        raise ValueError("build_report 需要至少一个 run 目录")

    # 生成时间取本地时区（datetime.now(UTC).astimezone() → 本地带时区时间）
    now = datetime.now(UTC).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    summaries = [_load_summary(run) for run in runs]
    lines = ["# AgentEval 实验报告", ""]
    lines.append(f"> 生成时间：{now}")
    lines.append(f"> run 目录：{'、'.join(run.name for run in runs)}")
    lines.append("")

    lines += _render_main_table(runs, summaries)
    lines += _render_difficulty_table(runs, summaries)
    lines += _render_failure_matrix(runs, summaries)
    lines += _render_scatter(runs, summaries)
    lines += _render_case_studies(runs, summaries)
    lines += _render_judge(runs, summaries)

    text = "\n".join(lines) + "\n"
    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return str(path)
    return text


def main(argv: list[str] | None = None) -> int:
    """CLI 入口：python -m analysis.report <run_dir...> [-o/--out FILE]。"""
    parser = argparse.ArgumentParser(
        prog="python -m analysis.report",
        description="汇总多个 AgentEval run 目录为 markdown 实验报告。",
    )
    parser.add_argument(
        "run_dir",
        nargs="+",
        type=Path,
        help="一个或多个 run 目录（含 summary.json 与 failure_modes.json）",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="输出 markdown 文件路径；缺省打印到 stdout",
    )
    args = parser.parse_args(argv)
    print(build_report(args.run_dir, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
