"""judge.judge —— 双 judge 评分与一致性报告（《项目方案.md》§8）。

职责：
- judge_one()：单模型单任务评分。调 litellm.completion（与 react 适配器同
  口径：model 保留 provider 前缀、api_base/api_key_env 取自 model_cfg、
  temperature=0.0、max_tokens 小、timeout=120），解析 JSON（剥围栏、深度
  扫描外层 {…}，兼容 reason 里的嵌套引号/花括号）；解析失败或无有效分数
  重试 1 次；仍失败按 §8.3⑤ 回退为 verifier 硬指标（passed=5 / failed=1）。
- judge_run()：读 results.csv + traces/{task_id}.jsonl + 任务定义，对每个
  任务用两个 judge 模型独立评分，输出 per_task / agreement_exact /
  agreement_pm1 / mean_score（按框架分组）/ n_fallback，写 judge_scores.json。
- CLI：python -m judge.judge <run_dir> [--judge-models a,b] [--dry-run]

可注入点：模块级 _call_llm（测试 monkeypatch）；dry_run 不调 API。
人工锚定（§8.3④）由 W4 完成：本模块只保留 per_task 的 score_a/score_b/
score_mean 字段供与锚定分对齐，不在此实现锚定流程。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import litellm
import yaml

from harness import protocol
from judge.compress import compress_trace
from judge.prompts import build_judge_messages
from judge.rubric import MAX_SCORE, MIN_SCORE
from metrics.success import load_task

# 单次评分的 token 上限。评分输出本身很小，但 deepseek-v4-flash 是推理模型：
# 思考 token 计入 max_tokens，256 会被思考吃光导致 content 为空
# （2026-09 真实 judge 运行 27/80 回退的根因），故上调到 1024；
# 模型配置可用 judge_max_tokens 键覆盖。
JUDGE_MAX_TOKENS = 1024
# 单次 completion 超时（秒）；LiteLLM 默认无超时，必须显式给出（§13 风险7）
JUDGE_TIMEOUT_S = 120.0

__all__ = ["judge_one", "judge_run", "main"]


# --------------------------------------------------------------------------
# JSON 解析与分数校验
# --------------------------------------------------------------------------
def _strip_fences(text: str) -> str:
    """剥掉首行/末行的 ```json / ``` 代码围栏。"""
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].strip().startswith("```"):
            t = "\n".join(lines[1:]).strip()
        if t.endswith("```"):
            t = t[:-3].rstrip()
    return t


def _try_loads(snippet: str) -> dict | None:
    try:
        obj = json.loads(snippet)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def parse_judge_json(text: str) -> dict | None:
    """从模型输出中提取评分 JSON 对象（剥围栏 + 深度扫描外层 {…}）。

    兼容：输出带前后缀自由文本、reason 中含嵌套引号/花括号（按字符串内
    转义规则跳过，只认最外层平衡的 JSON 对象）。提取不到返回 None。
    """
    if not text:
        return None
    t = _strip_fences(text)
    whole = _try_loads(t)
    if whole is not None:
        return whole
    start = t.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return _try_loads(t[start:i + 1])
    return None


def _normalize_score(value: Any) -> int | None:
    """把评分输出归一为 1~5 整数；非法（越界/非整数/缺失）返回 None。

    拒绝布尔值（bool 是 int 子类）与 4.7 这类非整数值；
    {"score": 9} 之类越界输出视为无效，触发重试与兜底。
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        number = value
    elif isinstance(value, float):
        if not value.is_integer():
            return None
        number = int(value)
    else:  # str："4" / "4.0" 均可，其余交给 float 校验
        try:
            f = float(str(value).strip())
        except ValueError:
            return None
        if not f.is_integer():
            return None
        number = int(f)
    if MIN_SCORE <= number <= MAX_SCORE:
        return number
    return None


# --------------------------------------------------------------------------
# 模型调用（模块级可注入；测试 monkeypatch judge._call_llm）
# --------------------------------------------------------------------------
def _call_llm(messages: list[dict], judge_model: str, model_cfg: dict) -> str:
    """调用 LiteLLM 完成一次 judge 评分，返回原始输出文本。

    与 react 适配器同口径：model 用 model_cfg["litellm_model"]（保留
    provider 前缀如 openai/deepseek-v4-flash），api_base 与 api_key_env
    同样取自 model_cfg；temperature=0.0 保证可复现，max_tokens 取小值。
    """
    env_name = model_cfg.get("api_key_env") or ""
    api_key = os.environ.get(env_name) if env_name else None
    resp = litellm.completion(
        model=model_cfg["litellm_model"],
        api_base=model_cfg.get("api_base"),
        api_key=api_key or None,
        messages=messages,
        temperature=0.0,
        max_tokens=int(model_cfg.get("judge_max_tokens", JUDGE_MAX_TOKENS)),
        timeout=JUDGE_TIMEOUT_S,
    )
    content = resp.choices[0].message.content
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _verifier_fallback(verifier_passed: bool, judge_model: str,
                       reason: str) -> dict:
    """§8.3⑤ 失败兜底：judge 不可用时回退为 verifier 硬指标。"""
    return {
        "score": MAX_SCORE if verifier_passed else MIN_SCORE,
        "reason": reason,
        "fallback": True,
        "judge_model": judge_model,
    }


# --------------------------------------------------------------------------
# 单任务单模型评分
# --------------------------------------------------------------------------
def judge_one(
    task: dict,
    trace,
    final_answer: str,
    judge_model: str,
    model_cfg: dict,
    *,
    verifier_passed: bool = True,
    dry_run: bool = False,
) -> dict:
    """对一条轨迹用一个 judge 模型评分，返回 {"score", "reason", ...}。

    Args:
        task: 任务定义（仅 user_goal 等字段进入 prompt）。
        trace: 归一化轨迹（harness.protocol.Trace）。
        final_answer: Agent 最终答复（轨迹尾行记录，调用方传入）。
        judge_model: judge 模型名（models_cfg 的键，仅用于记账/兜底）。
        model_cfg: 该模型的配置（litellm_model / api_base / api_key_env）。
        verifier_passed: verifier 硬指标结果，仅用于失败兜底（§8.3⑤）。
        dry_run: True 时不调 API，直接按 verifier 兜底打分。

    Returns:
        成功：{"score": 1~5, "reason": str, "fallback": False,
               "judge_model": name}；失败兜底：fallback=True。
    """
    if dry_run:
        return _verifier_fallback(
            verifier_passed, judge_model,
            "dry_run：未调用模型，按 verifier 硬指标兜底（§8.3⑤）",
        )
    compressed = compress_trace(trace)
    messages = build_judge_messages(task, compressed, final_answer)

    for _ in range(2):  # 首次调用 + 解析失败/无有效分数时重试 1 次
        try:
            raw = _call_llm(messages, judge_model, model_cfg)
        except Exception as exc:  # API 异常按一次失败处理，重试后兜底
            raw = None
            print(f"[judge] {judge_model} 调用异常（将重试）: {exc!r}")
        parsed = parse_judge_json(raw) if raw else None
        score = _normalize_score(parsed.get("score")) if parsed else None
        if score is not None:
            reason = parsed.get("reason")
            return {
                "score": score,
                "reason": str(reason or "").strip(),
                "fallback": False,
                "judge_model": judge_model,
            }
    return _verifier_fallback(
        verifier_passed, judge_model,
        "judge 两次输出均无法解析为合法评分，按 verifier 硬指标兜底（§8.3⑤）",
    )


# --------------------------------------------------------------------------
# run 目录级双 judge 评分
# --------------------------------------------------------------------------
def _parse_passed(value: Any) -> bool:
    """兼容 results.csv 中 passed 列的 "True"/"False" 字符串与布尔值。"""
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in ("true", "1", "yes", "pass", "passed")


def _read_results(results_csv: Path) -> list[dict]:
    """读 results.csv（列与编排器一致）；文件缺失/为空返回空列表。"""
    if not results_csv.is_file():
        return []
    with results_csv.open(newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f) if row.get("task_id")]


def _task_def(task_id: str) -> dict:
    """按 task_id 从任务库取任务定义；缺失时退化为最小 dict。"""
    try:
        return load_task(task_id)
    except KeyError:
        return {"task_id": task_id, "user_goal": ""}


def judge_run(run_dir: Path, judge_models: list[str], models_cfg: dict,
              dry_run: bool = False) -> dict:
    """对一个 run 目录做双 judge 评分并产出 judge_scores.json。

    Args:
        run_dir: run 目录（含 results.csv 与 traces/{task_id}.jsonl，§11.1）。
        judge_models: 两个 judge 模型名（models_cfg 的键，至少 2 个）。
        models_cfg: {模型名: 模型配置}（configs/models.yaml 的 models 段）。
        dry_run: True 时不调任何 API，全部按 verifier 兜底打分。

    Returns:
        {"per_task": [...], "agreement_exact", "agreement_pm1",
         "mean_score", "n_fallback"}，并写入 run_dir/judge_scores.json。
    """
    run_dir = Path(run_dir)
    judge_models = list(judge_models)
    if len(judge_models) < 2:
        raise ValueError("judge_models 至少需要两个模型（§8.3② 双 judge 评分）")
    judge_a, judge_b = judge_models[0], judge_models[1]
    cfg_a = models_cfg[judge_a]
    cfg_b = models_cfg[judge_b]

    per_task: list[dict] = []
    n_fallback = 0
    for row in _read_results(run_dir / "results.csv"):
        task_id = row["task_id"]
        passed = _parse_passed(row.get("passed"))
        framework = row.get("framework") or row.get("group") or "unknown"
        agent_model = row.get("model") or ""
        trace = protocol.load_trace(run_dir / "traces" / f"{task_id}.jsonl")

        if trace is None:  # 轨迹缺失（编排异常）→ 双 judge 直接按 verifier 兜底
            res_a = _verifier_fallback(passed, judge_a, "traces 缺失，按 verifier 兜底")
            res_b = _verifier_fallback(passed, judge_b, "traces 缺失，按 verifier 兜底")
        else:
            task = _task_def(task_id)
            res_a = judge_one(task, trace, trace.final_answer, judge_a,
                              cfg_a, verifier_passed=passed, dry_run=dry_run)
            res_b = judge_one(task, trace, trace.final_answer, judge_b,
                              cfg_b, verifier_passed=passed, dry_run=dry_run)

        score_a = int(res_a["score"])
        score_b = int(res_b["score"])
        n_fallback += int(bool(res_a.get("fallback"))) \
            + int(bool(res_b.get("fallback")))
        per_task.append({
            "task_id": task_id,
            "framework": framework,
            "model": agent_model,
            "passed": passed,
            "score_a": score_a,
            "score_b": score_b,
            "score_mean": round((score_a + score_b) / 2, 2),
            "agreement": score_a == score_b,
        })

    n_tasks = len(per_task)
    agreement_exact = 0.0
    agreement_pm1 = 0.0
    if n_tasks:
        agreement_exact = round(
            sum(t["agreement"] for t in per_task) / n_tasks, 4
        )
        agreement_pm1 = round(
            sum(abs(t["score_a"] - t["score_b"]) <= 1 for t in per_task)
            / n_tasks, 4,
        )

    mean_score: dict[str, float] = {}
    by_framework: dict[str, list[float]] = {}
    for t in per_task:
        by_framework.setdefault(t["framework"], []).append(t["score_mean"])
    for fw, scores in by_framework.items():
        mean_score[fw] = round(sum(scores) / len(scores), 2)

    result = {
        "per_task": per_task,
        "agreement_exact": agreement_exact,
        "agreement_pm1": agreement_pm1,
        "mean_score": mean_score,
        "n_fallback": n_fallback,
    }
    out_path = run_dir / "judge_scores.json"
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _load_models_cfg() -> dict:
    """读 configs/models.yaml 的 models 段（judge 模型配置来源）。"""
    path = protocol.REPO_ROOT / "configs" / "models.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("models", {})


def main(argv: list[str] | None = None) -> int:
    """CLI：python -m judge.judge <run_dir> [--judge-models a,b] [--dry-run]"""
    parser = argparse.ArgumentParser(
        prog="python -m judge.judge",
        description="AgentEval LLM-as-Judge：双 judge 评分与一致性报告（§8）",
    )
    parser.add_argument("run_dir", type=Path, help="run 目录（含 results.csv 与 traces/）")
    parser.add_argument(
        "--judge-models", default="",
        help="逗号分隔的 judge 模型名（models.yaml 的键）；"
             "默认取 models.yaml 全部模型",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="不调用任何模型 API，全部按 verifier 硬指标兜底",
    )
    args = parser.parse_args(argv)

    models_cfg = _load_models_cfg()
    judge_models = [m.strip() for m in args.judge_models.split(",") if m.strip()]
    if not judge_models:
        judge_models = list(models_cfg.keys())
    missing = [m for m in judge_models if m not in models_cfg]
    if missing:
        raise SystemExit(f"models.yaml 中没有 judge 模型: {missing}")

    result = judge_run(args.run_dir, judge_models, models_cfg,
                       dry_run=args.dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
