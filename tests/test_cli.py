"""统一 CLI 入口测试：命令路由与参数传递（mock 掉子进程模块调用）。"""
from typer.testing import CliRunner

import cli

runner = CliRunner()


def test_validate_command(monkeypatch):
    calls: list[tuple[str, list[str]]] = []

    def fake_run(module: str, args: list[str]) -> int:
        calls.append((module, args))
        return 0

    monkeypatch.setattr(cli, "_run_module", fake_run)
    result = runner.invoke(cli.app, ["validate"])
    assert result.exit_code == 0
    assert calls == [("tasks.validate", [])]


def test_run_command_defaults(monkeypatch):
    calls: list[tuple[str, list[str]]] = []

    def fake_run(module: str, args: list[str]) -> int:
        calls.append((module, args))
        return 0

    monkeypatch.setattr(cli, "_run_module", fake_run)
    result = runner.invoke(cli.app, ["run"])
    assert result.exit_code == 0
    assert calls == [(
        "harness.orchestrator",
        ["--config", "configs/experiment_matrix.yaml", "--port", "8200"],
    )]


def test_run_command_with_group_and_dry_run(monkeypatch):
    calls: list[tuple[str, list[str]]] = []

    def fake_run(module: str, args: list[str]) -> int:
        calls.append((module, args))
        return 0

    monkeypatch.setattr(cli, "_run_module", fake_run)
    result = runner.invoke(cli.app, ["run", "--group", "R1", "--dry-run", "--port", "9999"])
    assert result.exit_code == 0
    assert calls == [(
        "harness.orchestrator",
        ["--config", "configs/experiment_matrix.yaml", "--port", "9999", "--group", "R1", "--dry-run"],
    )]


def test_aggregate_command(monkeypatch):
    calls: list[tuple[str, list[str]]] = []

    def fake_run(module: str, args: list[str]) -> int:
        calls.append((module, args))
        return 0

    monkeypatch.setattr(cli, "_run_module", fake_run)
    result = runner.invoke(cli.app, ["aggregate", "runs/2026-09-05_R1"])
    assert result.exit_code == 0
    assert calls == [("metrics.aggregate", ["runs/2026-09-05_R1"])]
