"""harness.orchestrator —— 薄编排器 v0（《项目方案.md》§6.2 架构 C）。

职责（只调度、不理解任何框架细节）：
1. 读 experiment_matrix.yaml（runs，--group 过滤）+ models.yaml
   （default_budget / tool_server_port）；
2. 启动 mock 工具服务子进程（python -m tool_server.app），轮询 /healthz；
3. 对每个 group 的每个域逐个任务：POST reset → 生成 run_spec.json →
   以子进程驱动适配器（python -m harness.adapters.react）→ GET 终态 →
   用 metrics.success.verify_run 判定 → 追加 results.csv；
4. 预算、超时强杀、断点续跑（按 task_id 跳过已有行）全部在此层完成；
5. --dry-run（或环境变量 EVAL_DRY_RUN=1）时注入 fake_llm，全程不调模型。

CLI：python -m harness.orchestrator --config configs/experiment_matrix.yaml
      [--group R1] [--dry-run] [--port 8200]
退出码恒为 0（结果以 results.csv 为准），仅参数/环境错误时非 0。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import yaml

from harness import protocol
from harness.protocol import (
    EXIT_OK,
    RunBudget,
    Trace,
)
from metrics.success import verify_run  # 成败判定唯一权威入口（另一 agent 并行开发）

# W2 的 F1 模块由分析 agent 并行开发；未就绪时降级为空值，保证编排器独立可用
try:
    from metrics.tool_f1 import compute_f1 as _compute_f1
except ImportError:  # pragma: no cover —— 模块落地前/独立运行时
    _compute_f1 = None

# results.csv 列序（与 metrics.aggregate.CSV_COLUMNS 保持一致；新增列只能追加在尾部）
RESULT_COLUMNS = [
    "task_id", "group", "framework", "model", "difficulty", "status",
    "passed", "findings", "cost_usd", "wall_time_s", "n_tool_calls", "n_steps",
    "f1_recall", "f1_precision", "f1",
]
# 框架 → 适配器模块（§6.5：同一 CLI 契约，被评框架可插拔）
ADAPTER_MODULES = {
    "react": "harness.adapters.react",
    "smolagents": "harness.adapters.smolagents",
    "openhands": "harness.adapters.openhands",
}
# 框架级步数预算系数（v1.1：代码生成/重框架循环比 JSON 协议需要更多步数）
FRAMEWORK_STEP_SCALE = {
    "react": 1.0,
    "smolagents": 1.5,
    "openhands": 1.5,
}
SERVER_READY_TIMEOUT_S = 30.0      # mock 服务就绪轮询上限
SERVER_GRACE_S = 30.0              # 单任务 subprocess 超时裕量


# --------------------------------------------------------------------------
# 配置读取
# --------------------------------------------------------------------------
def load_yaml(path: Path) -> dict:
    """读取 YAML 配置文件（utf-8）。"""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def filter_groups(matrix: dict, group: str | None) -> list[dict]:
    """按 --group 过滤实验矩阵 runs；未指定则全部返回。"""
    runs = matrix.get("runs", [])
    if not group:
        return list(runs)
    return [r for r in runs if r.get("group") == group]


# --------------------------------------------------------------------------
# mock 工具服务进程生命周期
# --------------------------------------------------------------------------
def _start_tool_server(port: int) -> subprocess.Popen:
    """启动 mock 工具服务子进程，轮询 /healthz 直至就绪（≤30s）。"""
    proc = subprocess.Popen(
        [sys.executable, "-m", "tool_server.app", "--port", str(port)],
        cwd=str(protocol.REPO_ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + SERVER_READY_TIMEOUT_S
    last_err = ""
    while time.time() < deadline:
        if proc.poll() is not None:  # 进程提前退出（如端口被占）
            tail = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(
                f"mock 工具服务启动失败(exit={proc.returncode}):\n{tail[-2000:]}"
            )
        try:
            resp = httpx.get(f"{base_url}/healthz", timeout=1.0)
            if resp.status_code == 200 and resp.json().get("ok") is True:
                return proc
        except Exception as exc:
            last_err = str(exc)
        time.sleep(0.25)
    raise RuntimeError(f"mock 工具服务 {SERVER_READY_TIMEOUT_S:.0f}s 内未就绪: {last_err}")


def _stop_tool_server(server: Any) -> None:
    """停止 mock 工具服务（支持 Popen 与测试桩两种形态）。"""
    if server is None:
        return
    stop = getattr(server, "terminate", None)
    if callable(stop):  # Popen / 测试桩都暴露 terminate()
        try:
            stop()
            wait = getattr(server, "wait", None)
            if callable(wait):
                wait(timeout=5)
        except Exception:
            kill = getattr(server, "kill", None)
            if callable(kill):
                kill()


# --------------------------------------------------------------------------
# run 目录与 results.csv
# --------------------------------------------------------------------------
def _runs_root() -> Path:
    """实验产物根目录（runs/），测试可 monkeypatch 重定向。"""
    return protocol.REPO_ROOT / "runs"


def _group_run_dir(group: str) -> Path:
    """单次运行的目录：runs/<日期>_<group>/（§11.1）。"""
    return _runs_root() / f"{date.today().isoformat()}_{group}"


def _load_existing_task_ids(results_csv: Path) -> set[str]:
    """读取已有 results.csv 的 task_id 集合（断点续跑跳过依据）。"""
    if not results_csv.is_file():
        return set()
    with results_csv.open(newline="", encoding="utf-8") as f:
        return {row["task_id"] for row in csv.DictReader(f) if row.get("task_id")}


def _append_result(results_csv: Path, row: dict) -> None:
    """向 results.csv 追加一行（文件不存在时先写表头）。"""
    is_new = not results_csv.is_file()
    with results_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def _write_config_snapshot(run_dir: Path, snapshot: dict) -> None:
    """写 config.yaml 配置快照（本次运行可复现依据）。"""
    path = run_dir / "config.yaml"
    path.write_text(
        yaml.dump(snapshot, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------
# 单任务编排
# --------------------------------------------------------------------------
def _run_one_task(
    group: dict,
    task: dict,
    model_cfg: dict,
    default_budget: dict,
    base_url: str,
    run_dir: Path,
    dry_run: bool,
    adapter_module: str,
) -> dict:
    """跑单个任务：reset → 生成 run_spec → 子进程驱动适配器 → 判定。返回 CSV 行。"""
    group_name = group["group"]
    task_id = task["task_id"]
    run_id = f"{group_name}__{task_id}"
    framework = group["framework"]
    model = group["model"]

    # 1) 初始化工具服务实例状态
    reset_url = f"{base_url}/instances/{run_id}/reset"
    try:
        resp = httpx.post(reset_url, json={
            "domain": task.get("domain", ""),
            "initial_state": task.get("initial_state", {}),
        }, timeout=10.0)
        resp.raise_for_status()
    except Exception as exc:
        # reset 失败：记一行 error，继续后续任务
        return {
            "task_id": task_id, "group": group_name, "framework": framework,
            "model": model, "difficulty": task.get("difficulty", ""),
            "status": protocol.STATUS_ERROR, "passed": "False",
            "findings": f"reset_failed:{exc}", "cost_usd": 0.0,
            "wall_time_s": 0.0, "n_tool_calls": 0, "n_steps": 0,
            "f1_recall": "", "f1_precision": "", "f1": "",
        }

    # 2) 预算（§6.4：任务 max_steps 覆盖步数，其余取配置默认）
    # v1.1：smolagents(CodeAgent) 步数预算 ×1.5 —— 真实实验 an_004 显示
    # 10 步对 CodeAgent 的代码生成循环偏紧（跨表 JOIN 预算耗尽），
    # 框架级放宽容忍其"多试几步"的恢复行为，效率差异由 F1/调用数体现。
    step_scale = FRAMEWORK_STEP_SCALE.get(framework, 1.0)
    base_steps = int(task.get("max_steps") or default_budget.get("max_steps", 12))
    budget = RunBudget(
        max_steps=max(1, int(base_steps * step_scale)),
        timeout_s=float(default_budget.get("timeout_s", 180)),
        max_cost_usd=float(default_budget.get("max_cost_usd", 0.05)),
    )
    model_params = {
        "temperature": float(model_cfg.get("temperature", 0.2)),
        "max_tokens": int(model_cfg.get("max_tokens", 4096)),
    }
    if dry_run:
        model_params["fake_llm"] = True  # fake_llm：按 gt_plan 回放，不调模型
    spec = protocol.build_run_spec(
        run_id=run_id, task=task, base_url=base_url, instance_id=run_id,
        model=model, litellm_model=model_cfg["litellm_model"],
        api_base=model_cfg.get("api_base", ""),
        api_key_env=model_cfg.get("api_key_env"),
        model_params=model_params, budget=budget,
        gt_plan=task.get("ground_truth_plan") if dry_run else None,
        gt_answer=task.get("gt_answer") if dry_run else None,
    )
    spec_path = run_dir / f"run_spec_{task_id}.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    trace_path = run_dir / "traces" / f"{task_id}.jsonl"

    # 3) 子进程驱动适配器（超时强杀 → 按 timeout 计）
    proc = subprocess.Popen(
        [sys.executable, "-m", adapter_module, "--spec", str(spec_path),
         "--out", str(trace_path)],
        cwd=str(protocol.REPO_ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    timeout_hit = False
    try:
        _, _ = proc.communicate(timeout=budget.timeout_s + SERVER_GRACE_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        timeout_hit = True

    # 4) 读轨迹（缺失 → 构造空 Trace 记 error）
    trace = protocol.load_trace(trace_path)
    if trace is None:
        trace = Trace(run_id=run_id, task_id=task_id, framework=framework,
                      model=model, status=protocol.STATUS_ERROR)
    if timeout_hit:
        trace.status = protocol.STATUS_TIMEOUT

    # 5) 读工具服务终态（best-effort）
    final_state: dict = {}
    try:
        resp = httpx.get(f"{base_url}/instances/{run_id}/state", timeout=10.0)
        if resp.status_code == 200:
            data = resp.json()
            final_state = data.get("state", data) if isinstance(data, dict) else {}
    except Exception:
        final_state = {}

    # 6) 成败判定（metrics.success，全项目唯一权威）
    passed, findings = verify_run(task, trace.tool_calls(), final_state,
                                  trace.final_answer)
    row = {
        "task_id": task_id, "group": group_name, "framework": framework,
        "model": model, "difficulty": task.get("difficulty", ""),
        "status": trace.status, "passed": str(passed),
        "findings": ";".join(findings),
        "cost_usd": round(trace.total_cost_usd, 6),
        "wall_time_s": round(trace.wall_time_s, 2),
        "n_tool_calls": len(trace.tool_calls()),
        "n_steps": trace.model_turns,
    }
    # 7) Tool-Call F1（§7.2；模块未就绪时留空，由 aggregate 兜底补算）
    if _compute_f1 is not None:
        try:
            f1 = _compute_f1(task, trace)
            row["f1_recall"] = round(f1["recall"], 4)
            row["f1_precision"] = round(f1["precision"], 4)
            row["f1"] = round(f1["f1"], 4)
        except Exception as exc:  # F1 计算异常不得影响主流程
            row["f1_recall"] = row["f1_precision"] = row["f1"] = ""
            print(f"[{group_name}] {task_id} F1 计算异常: {exc}")
    return row


def run_group(group: dict, models_cfg: dict, default_budget: dict,
              base_url: str, dry_run: bool) -> int:
    """跑一个 group：遍历 domains 下全部任务，返回本组通过数。"""
    group_name = group["group"]
    model_cfg = models_cfg["models"][group["model"]]
    framework = group["framework"]
    if framework not in ADAPTER_MODULES:
        raise KeyError(
            f"group={group_name} 的 framework={framework!r} 没有对应适配器；"
            f"可用: {sorted(ADAPTER_MODULES)}"
        )
    adapter_module = ADAPTER_MODULES[framework]
    run_dir = _group_run_dir(group_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    # config.yaml 快照
    _write_config_snapshot(run_dir, {
        "run_group": group_name,
        "matrix": group,
        "model_cfg": model_cfg,
        "default_budget": default_budget,
        "dry_run": dry_run,
        "adapter": adapter_module,
    })

    results_csv = run_dir / "results.csv"
    done_ids = _load_existing_task_ids(results_csv)
    n_passed = 0
    n_total = 0
    for domain in group.get("domains", []):
        for task in protocol.load_tasks(domain):
            task_id = task["task_id"]
            if task_id in done_ids:  # 断点续跑：已完成任务跳过
                print(f"[{group_name}] {task_id} SKIP (已存在于 results.csv)")
                continue
            row = _run_one_task(group, task, model_cfg, default_budget,
                                base_url, run_dir, dry_run, adapter_module)
            _append_result(results_csv, row)
            done_ids.add(task_id)
            n_total += 1
            mark = "PASS" if row["passed"] == "True" else "FAIL"
            n_passed += 1 if row["passed"] == "True" else 0
            print(f"[{group_name}] {task_id} {mark} "
                  f"findings={row['findings']} "
                  f"f1={row.get('f1', '')} "
                  f"cost={row['cost_usd']}$ time={row['wall_time_s']}s")
    if n_total:
        print(f"[{group_name}] 汇总: SR={n_passed}/{n_total} "
              f"(results.csv: {results_csv})")
    return n_passed


def main(argv: list[str] | None = None) -> int:
    """编排器 CLI 入口。退出码恒 0（结果以 results.csv 为准）。"""
    parser = argparse.ArgumentParser(
        prog="python -m harness.orchestrator",
        description="AgentEval 薄编排器 v0（§6.2 架构 C）",
    )
    parser.add_argument("--config", default=None,
                        help="experiment_matrix.yaml 路径（默认 configs/ 下）")
    parser.add_argument("--group", default=None, help="只跑指定 group，如 R1")
    parser.add_argument("--dry-run", action="store_true",
                        help="fake_llm 回放 GT 路径，无需 API key")
    parser.add_argument("--port", type=int, default=None,
                        help="mock 工具服务端口（默认取 models.yaml）")
    args = parser.parse_args(argv)

    dry_run = args.dry_run or os.environ.get("EVAL_DRY_RUN") == "1"
    cfg_path = Path(args.config) if args.config else \
        protocol.REPO_ROOT / "configs" / "experiment_matrix.yaml"
    if not cfg_path.is_absolute():
        cfg_path = protocol.REPO_ROOT / cfg_path
    matrix = load_yaml(cfg_path)
    models_cfg = load_yaml(protocol.REPO_ROOT / "configs" / "models.yaml")
    default_budget = models_cfg.get("default_budget", {})
    port = args.port if args.port is not None else \
        int(models_cfg.get("tool_server_port", 8200))

    groups = filter_groups(matrix, args.group)
    if not groups:
        print(f"实验矩阵中没有匹配的 group={args.group!r}")
        return EXIT_OK

    server = _start_tool_server(port)
    try:
        base_url = f"http://127.0.0.1:{port}"
        for group in groups:
            run_group(group, models_cfg, default_budget, base_url, dry_run)
    finally:
        _stop_tool_server(server)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
