"""harness.protocol — AgentEval 评测数据契约（唯一权威定义）。

术语澄清（见《项目方案.md》§6）：
- Agent harness（被评对象）：运行 Agent 的框架（自研 ReAct / smolagents / OpenHands）。
- Evaluation harness（评测装置）：本仓库 harness/ 目录的薄编排器。

本模块定义两者之间唯一的进程间数据契约：
1. run_spec.json —— 编排器生成、适配器只读的运行规格（§6.3）；
2. trace.jsonl  —— 适配器产出的归一化轨迹（一行一步 + 一行汇总，§11.2）。

所有适配器必须：
- 通过 CLI 读取 run_spec（--spec）并写出 trace.jsonl（--out）；
- 退出码：0=正常结束，1=运行错误，2=预算/超时（§6.3）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------- 退出码约定（§6.3） ----------
EXIT_OK = 0        # 正常结束（含"完成了但答案错误"）
EXIT_ERROR = 1     # 运行错误（异常/崩溃）
EXIT_BUDGET = 2    # 预算耗尽 / 超时

# Trace 状态（§6.1）
STATUS_COMPLETED = "completed"
STATUS_TIMEOUT = "timeout"
STATUS_ERROR = "error"
STATUS_BUDGET_EXCEEDED = "budget_exceeded"
VALID_STATUSES = (STATUS_COMPLETED, STATUS_TIMEOUT, STATUS_ERROR, STATUS_BUDGET_EXCEEDED)

# 仓库根目录 = harness/ 的上级
REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_REGISTRY_PATH = REPO_ROOT / "tool_server" / "tool_registry.json"
TASKS_V1_DIR = REPO_ROOT / "tasks" / "v1"


# ---------- 核心数据结构（§6.1） ----------
@dataclass
class Step:
    type: str                  # "message" | "tool_call" | "observation" | "final_answer"
    role: str                  # "system" | "user" | "assistant" | "tool"
    content: str = ""
    tool_name: str | None = None
    tool_args: dict | None = None
    ts: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0


@dataclass
class RunBudget:
    max_steps: int = 12
    timeout_s: float = 180.0
    max_cost_usd: float = 0.05


@dataclass
class Trace:
    run_id: str
    task_id: str
    framework: str
    model: str
    steps: list[Step] = field(default_factory=list)
    final_answer: str = ""
    status: str = STATUS_COMPLETED
    wall_time_s: float = 0.0
    total_cost_usd: float = 0.0

    def tool_calls(self) -> list[dict]:
        """工具调用列表（verifier / F1 的输入口径）。"""
        return [
            {"tool": s.tool_name, "args": s.tool_args or {}}
            for s in self.steps
            if s.type == "tool_call" and s.tool_name
        ]

    @property
    def model_turns(self) -> int:
        """模型被调用的次数（预算口径，§6.4 max_steps）。"""
        return sum(1 for s in self.steps if s.type == "message" and s.role == "assistant")


# ---------- 工具注册表 ----------
def load_tool_registry() -> dict:
    with TOOL_REGISTRY_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def tool_specs_for_domain(domain: str) -> list[dict]:
    """取某业务域的工具规格（适配器据此翻译成各框架的工具格式）。"""
    registry = load_tool_registry()
    return [
        {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}
        for t in registry["tools"].get(domain, [])
    ]


def load_tasks(domain: str) -> list[dict]:
    """加载某业务域的全部任务（按文件内顺序）。"""
    path = TASKS_V1_DIR / f"{domain}_tasks.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["tasks"]


# ---------- run_spec（§6.3） ----------
def build_run_spec(
    *,
    run_id: str,
    task: dict,
    base_url: str,
    instance_id: str,
    model: str,
    litellm_model: str,
    api_base: str,
    api_key_env: str | None = None,
    model_params: dict | None = None,
    budget: RunBudget | None = None,
    gt_plan: list[dict] | None = None,   # 仅 dry-run（fake_llm）注入；真实运行必须为 None
    gt_answer: str | None = None,        # 仅 dry-run 注入
) -> dict:
    """生成适配器只读的运行规格（§6.3 run_spec.json）。"""
    budget = budget or RunBudget()
    spec: dict[str, Any] = {
        "run_id": run_id,
        "task": {
            "task_id": task["task_id"],
            "user_goal": task["user_goal"],
            "difficulty": task["difficulty"],
            "initial_state": task.get("initial_state", {}),
        },
        "tool_specs": tool_specs_for_domain(task["domain"]),
        "tool_server": {"base_url": base_url, "instance_id": instance_id},
        "model": model,
        "litellm_model": litellm_model,
        "api_base": api_base,
        "model_params": model_params or {},
        "budget": {
            "max_steps": budget.max_steps,
            "timeout_s": budget.timeout_s,
            "max_cost_usd": budget.max_cost_usd,
        },
    }
    if api_key_env:
        spec["api_key_env"] = api_key_env
    if gt_plan is not None:
        spec["gt_plan"] = gt_plan
    if gt_answer is not None:
        spec["gt_answer"] = gt_answer
    return spec


# ---------- trace.jsonl 序列化（§11.2：一行一步） ----------
def _trace_meta(trace: Trace) -> dict:
    return {
        "run_id": trace.run_id,
        "task_id": trace.task_id,
        "framework": trace.framework,
        "model": trace.model,
    }


def write_trace(trace: Trace, path: Path) -> None:
    """写 trace.jsonl：每步一行 + 末行 final_answer 汇总。"""
    lines: list[str] = []
    meta = _trace_meta(trace)
    for i, s in enumerate(trace.steps, start=1):
        line: dict[str, Any] = dict(
            meta, step=i, type=s.type, role=s.role, content=s.content,
            ts=s.ts, tokens_in=s.tokens_in, tokens_out=s.tokens_out, cost_usd=s.cost_usd,
        )
        if s.tool_name is not None:
            line["tool_name"] = s.tool_name
        if s.tool_args is not None:
            line["tool_args"] = s.tool_args
        lines.append(json.dumps(line, ensure_ascii=False))
    summary = dict(
        meta, step=len(trace.steps) + 1, type="final_answer", role="assistant",
        content=trace.final_answer, status=trace.status,
        wall_time_s=round(trace.wall_time_s, 4),
        total_cost_usd=round(trace.total_cost_usd, 8),
    )
    lines.append(json.dumps(summary, ensure_ascii=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_trace(path: Path) -> Trace | None:
    """读 trace.jsonl；文件不存在返回 None。"""
    if not path.exists():
        return None
    trace: Trace | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        d = json.loads(raw)
        if trace is None:
            trace = Trace(
                run_id=d["run_id"], task_id=d["task_id"],
                framework=d["framework"], model=d["model"],
            )
        if d.get("type") == "final_answer":
            trace.final_answer = d.get("content", "")
            trace.status = d.get("status", STATUS_COMPLETED)
            trace.wall_time_s = d.get("wall_time_s", 0.0)
            trace.total_cost_usd = d.get("total_cost_usd", 0.0)
        else:
            trace.steps.append(Step(
                type=d["type"], role=d["role"], content=d.get("content", ""),
                tool_name=d.get("tool_name"), tool_args=d.get("tool_args"),
                ts=d.get("ts", 0.0), tokens_in=d.get("tokens_in", 0),
                tokens_out=d.get("tokens_out", 0), cost_usd=d.get("cost_usd", 0.0),
            ))
    return trace
