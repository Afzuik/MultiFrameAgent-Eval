"""protocol 契约的单元测试：run_spec 生成、trace.jsonl 往返、工具规格。"""
import json
from pathlib import Path

from harness.protocol import (
    REPO_ROOT,
    RunBudget,
    Step,
    Trace,
    build_run_spec,
    load_tasks,
    load_trace,
    tool_specs_for_domain,
    write_trace,
)


def test_tool_specs_for_domain_contains_expected_tools():
    specs = tool_specs_for_domain("travel")
    names = {s["name"] for s in specs}
    assert {"search_flights", "book_flight", "verify_identity", "refund_reservation"} <= names
    assert all(set(s) >= {"name", "description", "parameters"} for s in specs)


def test_build_run_spec_contract_fields():
    task = load_tasks("travel")[0]
    spec = build_run_spec(
        run_id="r_001", task=task, base_url="http://127.0.0.1:8200", instance_id="r_001",
        model="deepseek-v4-flash", litellm_model="openai/deepseek-v4-flash",
        api_base="https://api.deepseek.com", api_key_env="DEEPSEEK_API_KEY",
        model_params={"temperature": 0.2}, budget=RunBudget(max_steps=6),
    )
    assert spec["run_id"] == "r_001"
    assert spec["task"]["task_id"] == "tr_001"
    assert spec["task"]["user_goal"]
    assert spec["tool_server"]["base_url"] == "http://127.0.0.1:8200"
    assert spec["budget"]["max_steps"] == 6
    assert len(spec["tool_specs"]) == len(tool_specs_for_domain("travel"))
    assert "gt_plan" not in spec  # 真实运行不得注入 GT


def test_build_run_spec_dry_run_injects_gt():
    task = load_tasks("travel")[0]
    spec = build_run_spec(
        run_id="r", task=task, base_url="http://x", instance_id="r",
        model="m", litellm_model="openai/m", api_base="https://x",
        gt_plan=task["ground_truth_plan"], gt_answer=task["gt_answer"],
    )
    assert spec["gt_plan"] == task["ground_truth_plan"]
    assert spec["gt_answer"] == task["gt_answer"]


def test_trace_jsonl_roundtrip(tmp_path: Path):
    trace = Trace(
        run_id="r_001", task_id="tr_001", framework="react", model="deepseek-v4-flash",
        steps=[
            Step(type="message", role="assistant", content='{"tool":"search_flights","args":{"date":"2026-10-12"}}', ts=1.0, tokens_in=10, tokens_out=5, cost_usd=0.001),
            Step(type="tool_call", role="assistant", content="", tool_name="search_flights",
                 tool_args={"date": "2026-10-12", "from": "PEK", "to": "SHA"}, ts=1.1),
            Step(type="observation", role="tool", content='{"ok":true,"result":[]}', ts=1.2),
        ],
        final_answer="上午航班：MU5101。", status="completed", wall_time_s=1.5, total_cost_usd=0.002,
    )
    path = tmp_path / "trace.jsonl"
    write_trace(trace, path)
    loaded = load_trace(path)
    assert loaded is not None
    assert loaded.run_id == "r_001" and loaded.task_id == "tr_001"
    assert loaded.final_answer == "上午航班：MU5101。"
    assert loaded.status == "completed"
    assert loaded.wall_time_s == 1.5
    assert loaded.total_cost_usd == 0.002
    assert loaded.model_turns == 1
    calls = loaded.tool_calls()
    assert len(calls) == 1
    assert calls[0]["tool"] == "search_flights"
    assert calls[0]["args"]["to"] == "SHA"
    # JSONL 一行一步 + 末行汇总
    lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert lines[-1]["type"] == "final_answer"
    assert len(lines) == 4


def test_load_trace_missing_file(tmp_path: Path):
    assert load_trace(tmp_path / "none.jsonl") is None


def test_repo_layout_sanity():
    assert (REPO_ROOT / "tasks" / "v1").is_dir()
    assert (REPO_ROOT / "tool_server" / "tool_registry.json").is_file()
