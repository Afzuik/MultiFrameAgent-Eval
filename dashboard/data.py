"""dashboard.data —— 数据加载与纯函数层（总览 / 轨迹回放 / 失败分析共用）。

本模块只负责“读取 run 产物 / 缺文件时兜底生成 / 结构化为纯数据”，
不 import streamlit，因此可被 tests/test_dashboard.py 直接单测。

run 目录约定见《项目方案.md》§11.1（runs/<日期>_<组>/ 下含 results.csv、
summary.json、failure_modes.json、traces/{task_id}.jsonl）；
报告结构对应字段见 §9.3（主表 / 分难度表 / 失败模式分布）。

对缺少派生产物（summary.json / failure_modes.json）的 run 目录，
按仓库 CLI 语义调用 metrics.aggregate.aggregate_run_dir 与
analysis.failure_modes.build_distribution 自动生成后返回，支撑
“dashboard 可交互查看任意一次运行”的验收项（§2.2-5）。
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

# ---- 路径自举：Streamlit 脚本沙箱会收窄 sys.path，先把仓库根目录挂上 ----
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.failure_modes import build_distribution
from harness.protocol import load_trace
from metrics.aggregate import aggregate_run_dir

# 仓库根目录 / 默认 runs 根目录（app.py 侧边栏默认值与外部引用）
REPO_ROOT = _REPO_ROOT
DEFAULT_RUNS_ROOT = REPO_ROOT / "runs"

# results.csv passed 单元格的常见真值写法（与 metrics.aggregate 口径一致）
_TRUTHY = frozenset({"1", "true", "yes", "pass", "passed", "y"})

# 框架名 → 中文展示标签（dashboard 全站统一口径）
FRAMEWORK_LABELS: dict[str, str] = {
    "react": "ReAct（自研）",
    "smolagents": "smolagents",
    "openhands": "OpenHands",
}

# load_trace_steps 对单步 content 的默认截断长度（字符）
CONTENT_MAX_LEN = 200


def _config_is_dry_run(run_dir: Path) -> bool:
    """按 config.yaml 的 dry_run 字段判断是否假跑/自测目录。

    缺 config.yaml 或解析失败均视为非 dry_run（保守纳入）；只有显式
    声明 ``dry_run: true`` 才排除（如历史 2026-09-03_O1 目录）。
    """
    path = run_dir / "config.yaml"
    if not path.is_file():
        return False
    try:
        import yaml  # 局部导入：仅此函数需要
    except ImportError:  # pragma: no cover
        return False
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(cfg and cfg.get("dry_run", False))


def list_run_dirs(runs_root: Path) -> list[Path]:
    """列出 runs_root 下的合法 run 目录，按目录名升序返回。

    合法性判定（§11.1 目录约定 + 仓库历史目录实践）：
    1. 必须是目录且内含 results.csv（结果行齐全，视为可评测运行）；
    2. 目录名不含 "dryrun" / "failed"——排除 _dryrun、_failed* 等旧
       调试目录（如 2026-09-03_R1_dryrun、S1_dryrun、S1_failed_prefix）；
    3. config.yaml 未声明 ``dry_run: true``——排除假跑/自测目录
       （如历史 2026-09-03_O1：dry_run: true、全部 0.01s 假通过）。

    Args:
        runs_root: runs 根目录（《项目方案.md》§11.1 数据源目录）。

    Returns:
        合法 run 目录列表（Path）；runs_root 不存在时返回空列表。
    """
    root = Path(runs_root)
    if not root.is_dir():
        return []
    result: list[Path] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        name = path.name.lower()
        if "dryrun" in name or "failed" in name:
            continue
        if not (path / "results.csv").is_file():
            continue
        if _config_is_dry_run(path):
            continue
        result.append(path)
    return result


def load_group_data(run_dir: Path) -> dict[str, Any]:
    """加载单个 run 目录的结构化聚合数据（§9.3 主表 / 分难度表字段）。

    读取优先级：results.csv（逐任务行，用于展示与兜底生成）、
    summary.json（聚合指标）；缺失时调用 metrics.aggregate.aggregate_run_dir
    兜底汇总（与仓库 CLI 语义一致：写盘并返回）。failure_modes.json
    缺失时调用 analysis.failure_modes.build_distribution 计算并落盘。

    Returns:
        run_dir: run 目录绝对路径字符串；
        group / framework / model: 组、框架、backbone 模型名；
        n_tasks / n_completed / n_passed: 任务数 / 完成数 / 通过数；
        sr: 成功率（0~1，§7.1 口径，未通过一律计失败）；
        f1 / f1_recall / f1_precision: 平均工具调用 F1 及其分项；
        cost_usd: 平均单任务成本（$）；
        latency_s: 平均单任务端到端延迟（s）；
        tool_calls: 平均单任务工具调用次数；
        by_difficulty: {"L1": {"n": …, "sr": …}, "L2": …, "L3": …}；
        failures: summary.json 的失败任务列表
                  （task_id / difficulty / status / findings）；
        failure_modes: failure_modes.json 的失败模式分布 dict
                       （by_category / details / by_framework / …）。
    """
    run_dir = Path(run_dir)

    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = aggregate_run_dir(run_dir)

    fm_path = run_dir / "failure_modes.json"
    if fm_path.is_file():
        failure_modes = json.loads(fm_path.read_text(encoding="utf-8"))
    else:
        failure_modes = build_distribution(run_dir)
        fm_path.write_text(
            json.dumps(failure_modes, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return {
        "run_dir": str(run_dir),
        "group": str(summary.get("group") or ""),
        "framework": str(summary.get("framework") or ""),
        "model": str(summary.get("model") or ""),
        "n_tasks": int(summary.get("n_tasks") or 0),
        "n_completed": int(summary.get("n_completed") or 0),
        "n_passed": int(summary.get("n_passed") or 0),
        "sr": float(summary.get("sr") or 0.0),
        "f1": float(summary.get("mean_f1") or 0.0),
        "f1_recall": float(summary.get("mean_f1_recall") or 0.0),
        "f1_precision": float(summary.get("mean_f1_precision") or 0.0),
        "cost_usd": float(summary.get("mean_cost_usd") or 0.0),
        "latency_s": float(summary.get("mean_wall_time_s") or 0.0),
        "tool_calls": float(summary.get("mean_tool_calls") or 0.0),
        "by_difficulty": dict(summary.get("by_difficulty") or {}),
        "failures": list(summary.get("failures") or []),
        "failure_modes": failure_modes,
    }


def load_all_groups(runs_root: Path | None = None) -> list[dict[str, Any]]:
    """加载 runs_root 下全部合法组的结构化数据，按 group 名升序返回。

    Args:
        runs_root: runs 根目录；缺省使用仓库默认 runs/（DEFAULT_RUNS_ROOT）。

    Returns:
        每组一个 load_group_data 的结果 dict；无合法 run 目录时返回空列表。
    """
    root = Path(runs_root) if runs_root is not None else DEFAULT_RUNS_ROOT
    groups = [load_group_data(d) for d in list_run_dirs(root)]
    groups.sort(key=lambda g: str(g["group"]))
    return groups


def _parse_bool(value: Any) -> bool:
    """把 results.csv 的 passed 单元格解析为 bool；无法识别按失败计。"""
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in _TRUTHY


def _to_float(value: Any, default: float = 0.0) -> float:
    """把单元格解析为 float；空 / 非法返回 default。"""
    if value is None or str(value).strip() == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_opt_float(value: Any) -> float | None:
    """把单元格解析为 float；空 / 非法返回 None（老版 CSV 无 F1 列）。"""
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int:
    """把单元格解析为 int（四舍五入）；空 / 非法返回 0。"""
    return round(_to_float(value))


def load_task_rows(run_dir: Path) -> list[dict[str, Any]]:
    """读 run 目录 results.csv，返回逐任务行（数值 / 布尔已清洗，展示用）。

    - passed 解析为 bool（“未通过 / 无法识别一律 False”，与 aggregate 的
      “失败计 fail”口径一致）；
    - cost_usd / wall_time_s / n_tool_calls / n_steps 数值化（空按 0）；
    - f1 / f1_recall / f1_precision 空单元格保留 None（老版 CSV 可无此列）；
    - findings 原样保留为字符串。

    Args:
        run_dir: 单个 run 目录（须含 results.csv）。

    Returns:
        逐任务行 dict 列表；results.csv 缺失时返回空列表。
    """
    path = Path(run_dir) / "results.csv"
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            task_id = (raw.get("task_id") or "").strip()
            if not task_id:
                continue
            rows.append({
                "task_id": task_id,
                "group": raw.get("group") or "",
                "framework": raw.get("framework") or "",
                "model": raw.get("model") or "",
                "difficulty": raw.get("difficulty") or "",
                "status": raw.get("status") or "",
                "passed": _parse_bool(raw.get("passed")),
                "findings": raw.get("findings") or "",
                "cost_usd": _to_float(raw.get("cost_usd")),
                "wall_time_s": _to_float(raw.get("wall_time_s")),
                "n_tool_calls": _to_float(raw.get("n_tool_calls")),
                "n_steps": _to_int(raw.get("n_steps")),
                "f1": _to_opt_float(raw.get("f1")),
                "f1_recall": _to_opt_float(raw.get("f1_recall")),
                "f1_precision": _to_opt_float(raw.get("f1_precision")),
            })
    return rows


def _truncate(text: Any, max_len: int) -> str:
    """把任意内容转为字符串并按 max_len 截断（超出补省略号“…”）。"""
    value = str(text or "")
    if len(value) <= max_len:
        return value
    return value[:max_len] + "…"


def load_trace_steps(
    run_dir: Path,
    task_id: str,
    max_content: int = CONTENT_MAX_LEN,
) -> list[dict[str, Any]]:
    """把单任务归一化轨迹（§11.2 traces/{task_id}.jsonl）转为展示步骤列表。

    每个步骤 dict 字段：
        step: 步骤序号（从 1 起）；
        type / role: 步骤类型与角色原文（"message" / "tool_call" /
                     "observation"…）；
        content: 内容文本，截断至 max_content 字符（超长补“…”）；
        tool_name / tool_args: 工具调用信息（非调用步骤为 None）；
        ts: 时间戳（秒）；duration_s: 距上一步的耗时（秒，首步为 0）；
        cost_usd: 本步模型成本（$）。

    Args:
        run_dir: run 目录；task_id: 任务标识；
        max_content: content 截断长度（默认 200 字符，显示用）。

    Returns:
        步骤 dict 列表；轨迹文件缺失 / 无法解析时返回空列表。
    """
    trace = load_trace(Path(run_dir) / "traces" / f"{task_id}.jsonl")
    if trace is None:
        return []
    steps: list[dict[str, Any]] = []
    prev_ts: float | None = None
    for n, step in enumerate(trace.steps, start=1):
        ts = _to_float(step.ts)
        duration = round(max(ts - prev_ts, 0.0), 3) if prev_ts is not None else 0.0
        prev_ts = ts
        steps.append({
            "step": n,
            "type": str(step.type),
            "role": str(step.role),
            "content": _truncate(step.content, max_content),
            "tool_name": step.tool_name,
            "tool_args": step.tool_args,
            "ts": ts,
            "duration_s": duration,
            "cost_usd": _to_float(step.cost_usd),
        })
    return steps


def load_trace_meta(run_dir: Path, task_id: str) -> dict[str, Any] | None:
    """读取单任务轨迹的汇总信息（JSONL 末尾 final_answer 行，§11.2）。

    Returns:
        run_id / task_id / framework / model / status / final_answer /
        wall_time_s / total_cost_usd / n_steps / n_tool_calls；
        轨迹缺失时返回 None。
    """
    trace = load_trace(Path(run_dir) / "traces" / f"{task_id}.jsonl")
    if trace is None:
        return None
    return {
        "run_id": trace.run_id,
        "task_id": trace.task_id,
        "framework": trace.framework,
        "model": trace.model,
        "status": trace.status,
        "final_answer": trace.final_answer,
        "wall_time_s": round(_to_float(trace.wall_time_s), 3),
        "total_cost_usd": round(_to_float(trace.total_cost_usd), 6),
        "n_steps": len(trace.steps),
        "n_tool_calls": len(trace.tool_calls()),
    }
