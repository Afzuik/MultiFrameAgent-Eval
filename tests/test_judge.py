"""test_judge —— judge 包（§8 LLM-as-Judge）单元测试（不依赖真实 API）。

覆盖：
1. compress：长 observation 截断 + 摘要标注 / 短内容不截断 /
   框架名·模型名从压缩文本剔除（匿名化）；
2. prompts：rubric 全文与打分输出协议都出现在消息里，且不含框架/模型名；
3. judge_one：monkeypatch _call_llm —— 合法 JSON → score/reason；
   非法文本 ×2 → verifier 兜底（passed→5 / failed→1）；score 越界 → 兜底；
4. judge_run：tmp_path 造 run_dir（3 行 results.csv + 对应 traces），
   断言 per_task / agreement / mean_score / n_fallback 与落盘
   judge_scores.json；dry_run 全程不调 API。
"""
from __future__ import annotations

import json

import pytest

from harness.protocol import Step, Trace, write_trace
from judge import judge as judge_mod
from judge import prompts
from judge.compress import compress_trace
from judge.rubric import RUBRIC

# 匿名化测试的禁用词表（框架名 + 模型名片段，§8.3①）
_BANNED = ["react", "smolagents", "openhands", "deepseek", "qwen", "gpt"]

_RESULTS_HEADER = [
    "task_id", "group", "framework", "model", "difficulty", "status",
    "passed", "findings", "cost_usd", "wall_time_s", "n_tool_calls",
    "n_steps", "f1_recall", "f1_precision", "f1",
]


# --------------------------------------------------------------------------
# Trace / run_dir 构造工厂
# --------------------------------------------------------------------------
def _trace(task_id: str = "tr_001", steps: list | None = None) -> Trace:
    """构造一条归一化轨迹（框架/模型固定为 react/deepseek-v4-flash）。"""
    return Trace(
        run_id=f"r_{task_id}",
        task_id=task_id,
        framework="react",
        model="deepseek-v4-flash",
        steps=list(steps or []),
        final_answer="已完成您的请求。",
        status="completed",
    )


def _msg(content: str, role: str = "assistant") -> Step:
    return Step(type="message", role=role, content=content)


def _obs(content: str) -> Step:
    return Step(type="observation", role="tool", content=content)


def _tool_call(name: str, args: dict | None = None) -> Step:
    return Step(type="tool_call", role="assistant", tool_name=name,
                tool_args=args)


def _write_run_dir(tmp_path: pytest.TempPathFactory) -> object:
    """造一个含 3 行 results.csv + 对应 traces 的 run 目录。

    行：tr_001 passed=True / tr_002 passed=False / tr_003 passed=True。
    """
    run_dir = tmp_path / "run"
    (run_dir / "traces").mkdir(parents=True)
    rows = [
        ("tr_001", "True", "search_flights"),
        ("tr_002", "False", "book_flight"),
        ("tr_003", "True", "list_reservations"),
    ]
    for task_id, passed, tool in rows:
        trace = _trace(
            task_id,
            steps=[
                _msg(f"任务：{task_id}", role="user"),
                _tool_call(tool, {"user_id": "u_42"}),
                _obs(f'{{"ok": true, "task": "{task_id}"}}'),
            ],
        )
        write_trace(trace, run_dir / "traces" / f"{task_id}.jsonl")
    with (run_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=_RESULTS_HEADER)
        writer.writeheader()
        for task_id, passed, _ in rows:
            writer.writerow({
                "task_id": task_id, "group": "R1", "framework": "react",
                "model": "deepseek-v4-flash", "difficulty": "L1",
                "status": "completed", "passed": passed, "findings": "",
                "cost_usd": 0.0, "wall_time_s": 1.0, "n_tool_calls": 1,
                "n_steps": 2, "f1_recall": "", "f1_precision": "", "f1": "",
            })
    return run_dir


def _judge_models_cfg() -> dict:
    """两个 judge 模型的假配置（litellm_model 带 provider 前缀）。"""
    return {
        "judge_a": {"litellm_model": "openai/judge-a", "api_key_env": ""},
        "judge_b": {"litellm_model": "openai/judge-b", "api_key_env": ""},
    }


# --------------------------------------------------------------------------
# compress：长 observation 截断 + 摘要标注
# --------------------------------------------------------------------------
def test_compress_truncates_long_observation():
    """observation 超过 500 字符：保留前 500 字符并追加"…(截断 500 字)"。"""
    long_content = "客" * 600
    trace = _trace(steps=[_msg("查航班", role="user"), _obs(long_content)])
    out = compress_trace(trace)
    assert "[步骤 1]" in out and "[步骤 2]" in out
    assert "客" * 500 in out
    assert "…(截断 500 字)" in out
    assert long_content not in out  # 尾部长内容已截掉


def test_compress_short_content_kept():
    """observation 短于阈值：内容完整保留且无截断标注。"""
    content = "查询成功，共 3 个航班。"
    trace = _trace(steps=[_tool_call("search_flights"), _obs(content)])
    out = compress_trace(trace)
    assert content in out
    assert "截断" not in out


def test_compress_anonymized_no_framework_model_names():
    """压缩文本不得出现框架名/模型名（§8.3①，元数据不进文本 + 内容剔除）。"""
    trace = _trace(
        steps=[
            _msg("本轨迹由 deepseek-v4-flash 在 react 框架下运行", role="user"),
            _obs("结果正常"),
        ],
    )
    out = compress_trace(trace)
    assert "deepseek-v4-flash" not in out
    assert "react" not in out


# --------------------------------------------------------------------------
# prompts：rubric 全文 + 输出协议 + 匿名化
# --------------------------------------------------------------------------
def _messages_text(messages: list[dict]) -> str:
    return "\n".join(str(m.get("content", "")) for m in messages)


def test_prompts_contain_full_rubric_and_protocol():
    """system 含 rubric 全部五档定义与打分输出协议（§8.2/§8.3）。"""
    task = {"task_id": "tr_001", "user_goal": "查询航班"}
    messages = prompts.build_judge_messages(task, "轨迹文本", "最终答案")
    text = _messages_text(messages)
    for score, definition in RUBRIC.items():
        assert f"{score} 分：{definition}" in text
    assert '"score"' in text
    assert '"reason"' in text
    assert "1~5" in text


def test_prompts_no_framework_model_names():
    """judge 消息（system+user）不含任何框架名/模型名。"""
    task = {"task_id": "tr_001", "user_goal": "查询 10 月 12 日航班"}
    messages = prompts.build_judge_messages(task, "某条运行轨迹", "已查询")
    text = _messages_text(messages)
    for banned in _BANNED:
        assert banned not in text, f"prompt 泄露敏感词: {banned}"


# --------------------------------------------------------------------------
# judge_one：_call_llm 可注入
# --------------------------------------------------------------------------
def _task() -> dict:
    return {"task_id": "tr_001", "user_goal": "查询航班"}


def test_judge_one_valid_json(monkeypatch):
    """_call_llm 返回合法 JSON → 解析出 score/reason，且只调一次。"""
    calls: list[list] = []

    def fake_llm(messages, judge_model, model_cfg):
        calls.append(messages)
        return '{"score": 4, "reason": "目标达成，有一次冗余查询。"}'

    monkeypatch.setattr(judge_mod, "_call_llm", fake_llm)
    res = judge_mod.judge_one(_task(), _trace(), "最终答案", "judge_a", {})
    assert res["score"] == 4
    assert "冗余查询" in res["reason"]
    assert res["fallback"] is False
    assert len(calls) == 1
    # 消息确实含 rubric（验证走的是真实 prompt 构造）
    assert "评分规则" in str(calls[0][0]["content"])


def test_judge_one_invalid_twice_fallback_passed(monkeypatch):
    """两次都输出非法文本 → 兜底：passed=True → score=5。"""
    calls: list[str] = []

    def fake_llm(messages, judge_model, model_cfg):
        calls.append(judge_model)
        return "我不是 JSON，随便说说"

    monkeypatch.setattr(judge_mod, "_call_llm", fake_llm)
    res = judge_mod.judge_one(_task(), _trace(), "答案", "judge_a", {},
                              verifier_passed=True)
    assert res["score"] == 5
    assert res["fallback"] is True
    assert len(calls) == 2  # 首评 + 重试 1 次


def test_judge_one_invalid_twice_fallback_failed(monkeypatch):
    """两次都输出非法文本 → 兜底：passed=False → score=1。"""
    monkeypatch.setattr(
        judge_mod, "_call_llm",
        lambda messages, judge_model, model_cfg: "垃圾文本",
    )
    res = judge_mod.judge_one(_task(), _trace(), "答案", "judge_a", {},
                              verifier_passed=False)
    assert res["score"] == 1
    assert res["fallback"] is True


def test_judge_one_score_out_of_range_fallback(monkeypatch):
    """{"score": 9} 越界 → 判为无效 → 重试一次后仍无效 → 兜底。"""
    calls: list[str] = []

    def fake_llm(messages, judge_model, model_cfg):
        calls.append(judge_model)
        return '{"score": 9, "reason": "明显越界"}'

    monkeypatch.setattr(judge_mod, "_call_llm", fake_llm)
    res = judge_mod.judge_one(_task(), _trace(), "答案", "judge_a", {},
                              verifier_passed=True)
    assert res["fallback"] is True
    assert res["score"] == 5
    assert len(calls) == 2


def test_judge_one_dry_run_no_api(monkeypatch):
    """dry_run：不调 API，直接按 verifier 兜底（passed→5）。"""
    def forbidden(messages, judge_model, model_cfg):
        raise AssertionError("dry_run 不应调用 _call_llm")

    monkeypatch.setattr(judge_mod, "_call_llm", forbidden)
    res = judge_mod.judge_one(_task(), _trace(), "答案", "judge_a", {},
                              verifier_passed=True, dry_run=True)
    assert res["score"] == 5
    assert res["fallback"] is True


# --------------------------------------------------------------------------
# judge_run：per_task / agreement / mean_score / n_fallback / 落盘
# --------------------------------------------------------------------------
def test_judge_run_agreement_and_outputs(monkeypatch, tmp_path):
    """双 judge 分数不同（4 vs 5）→ 完全一致 0、±1 内一致 1、均值正确。"""
    run_dir = _write_run_dir(tmp_path)

    def fake_llm(messages, judge_model, model_cfg):
        if judge_model == "judge_a":
            return '{"score": 4, "reason": "judge_a 判定"}'
        return '{"score": 5, "reason": "judge_b 判定"}'

    monkeypatch.setattr(judge_mod, "_call_llm", fake_llm)
    result = judge_mod.judge_run(
        run_dir, ["judge_a", "judge_b"], _judge_models_cfg(),
    )

    assert [t["task_id"] for t in result["per_task"]] == \
        ["tr_001", "tr_002", "tr_003"]
    for entry in result["per_task"]:
        assert entry["score_a"] == 4
        assert entry["score_b"] == 5
        assert entry["score_mean"] == 4.5
        assert entry["framework"] == "react"
        assert entry["agreement"] is False
    assert result["agreement_exact"] == 0.0
    assert result["agreement_pm1"] == 1.0
    assert result["mean_score"] == {"react": 4.5}
    assert result["n_fallback"] == 0
    # 落盘 judge_scores.json 与返回值一致
    written = json.loads((run_dir / "judge_scores.json").read_text("utf-8"))
    assert written == result


def test_judge_run_same_scores_exact_agreement(monkeypatch, tmp_path):
    """双 judge 分数一致 → agreement_exact=1，passed 布尔正确解析。"""
    run_dir = _write_run_dir(tmp_path)

    def fake_llm(messages, judge_model, model_cfg):
        return '{"score": 3, "reason": "一致判定"}'

    monkeypatch.setattr(judge_mod, "_call_llm", fake_llm)
    result = judge_mod.judge_run(
        run_dir, ["judge_a", "judge_b"], _judge_models_cfg(),
    )
    assert result["agreement_exact"] == 1.0
    assert result["agreement_pm1"] == 1.0
    assert result["mean_score"] == {"react": 3.0}
    by_task = {t["task_id"]: t for t in result["per_task"]}
    assert by_task["tr_001"]["passed"] is True
    assert by_task["tr_002"]["passed"] is False  # "False" 字符串解析为 False
    assert by_task["tr_003"]["passed"] is True


def test_judge_run_dry_run(monkeypatch, tmp_path):
    """dry_run：全程不调 API，passed→5 / failed→1，n_fallback=任务数×2。"""
    run_dir = _write_run_dir(tmp_path)

    def forbidden(messages, judge_model, model_cfg):
        raise AssertionError("dry_run 不应调用 _call_llm")

    monkeypatch.setattr(judge_mod, "_call_llm", forbidden)
    result = judge_mod.judge_run(
        run_dir, ["judge_a", "judge_b"], _judge_models_cfg(),
        dry_run=True,
    )
    by_task = {t["task_id"]: t for t in result["per_task"]}
    assert by_task["tr_001"]["score_a"] == 5  # passed
    assert by_task["tr_002"]["score_a"] == 1  # failed
    assert result["agreement_exact"] == 1.0  # 双 judge 均按 verifier 兜底
    assert result["agreement_pm1"] == 1.0
    assert result["n_fallback"] == 6  # 3 任务 × 2 judge
    assert (run_dir / "judge_scores.json").is_file()
