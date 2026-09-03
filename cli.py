"""统一 CLI 入口（《项目方案.md》§12）：agent-eval run / validate / aggregate。

薄壳设计：复用各模块自带 `python -m` 入口（子进程调用），
与适配器/编排器的 CLI 契约保持单一来源，避免入口漂移。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer

app = typer.Typer(help="AgentEval 多框架 Agent 评测体系统一入口", no_args_is_help=True)
REPO_ROOT = Path(__file__).resolve().parent


def _run_module(module: str, args: list[str]) -> int:
    return subprocess.run(
        [sys.executable, "-m", module, *args], cwd=REPO_ROOT, check=False
    ).returncode


@app.command()
def run(
    config: str = typer.Option("configs/experiment_matrix.yaml", help="实验矩阵配置路径"),
    group: str = typer.Option(None, help="只跑指定 group（如 R1）"),
    dry_run: bool = typer.Option(False, "--dry-run", help="fake_llm 回放 GT 路径（无需 API key）"),
    port: int = typer.Option(8200, help="mock 工具服务端口"),
) -> None:
    """跑一组评测实验，产出 runs/<日期>_<group>/ 下的轨迹与结果。"""
    args = ["--config", config, "--port", str(port)]
    if group:
        args += ["--group", group]
    if dry_run:
        args += ["--dry-run"]
    raise SystemExit(_run_module("harness.orchestrator", args))


@app.command()
def validate() -> None:
    """任务集双向自检（空轨迹必 fail、GT 路径必 pass）。"""
    raise SystemExit(_run_module("tasks.validate", []))


@app.command()
def aggregate(run_dir: Path = typer.Argument(..., help="run 目录，如 runs/2026-09-05_R1")) -> None:
    """对 run 目录汇总指标，产出 summary.json。"""
    raise SystemExit(_run_module("metrics.aggregate", [str(run_dir)]))


if __name__ == "__main__":
    app()
