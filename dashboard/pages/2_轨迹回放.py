"""轨迹回放页 —— 选择实验组与任务，回放逐步轨迹时间线。

对应《项目方案.md》§2.2-5（可交互查看任意一次运行的完整轨迹）与
§9.3.5（案例研究逐步分析）。逐任务信息来自 results.csv 行
（status/passed/findings/F1），轨迹来自 traces/{task_id}.jsonl
（dashboard.data.load_trace_steps，content 截断为显示用）。
"""
import json
import sys
from pathlib import Path

import streamlit as st

# ---- 路径自举：保证从任意工作目录 / Streamlit 沙箱均可导入 dashboard.* ----
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dashboard import data as ddata

# 状态 / 步骤类型 / 角色的中文展示标签
_STATUS_LABELS = {
    "completed": "完成",
    "timeout": "超时",
    "budget_exceeded": "预算超限",
    "error": "运行错误",
}
_TYPE_LABELS = {
    "message": "💬 模型消息",
    "tool_call": "🔧 工具调用",
    "observation": "📥 工具返回",
    "final_answer": "✅ 最终回答",
}
_ROLE_LABELS = {
    "system": "系统",
    "user": "用户",
    "assistant": "助手",
    "tool": "工具",
}


def _fw_label(framework: str) -> str:
    return ddata.FRAMEWORK_LABELS.get(framework, framework or "—")


def _status_cn(status: str) -> str:
    return _STATUS_LABELS.get(status, status or "—")


def _f1_text(value: float | None) -> str:
    """F1 单元格显示文本（空值显示 —）。"""
    return "—" if value is None else f"{value:.2f}"


def _group_label(g: dict) -> str:
    """下拉框里实验组的展示文本。"""
    return f"{g['group']} ｜ {_fw_label(g['framework'])} × {g['model']} ｜ SR {g['sr']:.0%}"


def _task_label(r: dict) -> str:
    """下拉框里任务行的展示文本。"""
    mark = "✅ 通过" if r["passed"] else "❌ 失败"
    return f"{r['task_id']} ｜ {r['difficulty'] or '—'} ｜ {mark} ｜ F1 {_f1_text(r['f1'])}"


def _render_task_info(row: dict) -> None:
    """展示 results.csv 行对应的任务基本信息（status/passed/findings/f1…）。"""
    passed = bool(row["passed"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("任务", row["task_id"])
    c2.metric("难度", row["difficulty"] or "—")
    c3.metric("判定", "✅ 通过" if passed else "❌ 失败")
    c4.metric("状态", _status_cn(row["status"]))
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("工具 F1", "—" if row["f1"] is None else f"{row['f1']:.3f}")
    c6.metric("成本 ($)", f"{row['cost_usd']:.4f}")
    c7.metric("延迟 (s)", f"{row['wall_time_s']:.2f}")
    c8.metric("工具调用", f"{row['n_tool_calls']:.0f}")
    if row["findings"]:
        with st.expander("verifier findings（失败原因）"):
            st.code(row["findings"], language=None)


def _render_timeline(steps: list[dict], meta: dict) -> None:
    """逐步轨迹时间线：类型标签 + 内容 + 耗时（st.expander 列表）。"""
    st.subheader("逐步轨迹时间线")
    if meta:
        st.caption(
            f"共 {meta['n_steps']} 步 · {meta['n_tool_calls']} 次工具调用 · "
            f"端到端 {meta['wall_time_s']}s · 总成本 ${meta['total_cost_usd']:.6f}"
        )
    if not steps:
        st.warning("该任务没有轨迹步骤（轨迹文件缺失或为空）。")
        return
    for s in steps:
        type_label = _TYPE_LABELS.get(s["type"], s["type"] or "步骤")
        role_label = _ROLE_LABELS.get(s["role"], s["role"] or "")
        heading = f"{s['step']:>2}. {type_label}"
        if s["tool_name"]:
            heading += f"：{s['tool_name']}"
        else:
            heading += f"（{role_label}）"
        with st.expander(heading):
            st.caption(
                f"{role_label} · 耗时 {s['duration_s']}s · 成本 ${s['cost_usd']:.6f}"
            )
            if s["content"]:
                st.code(s["content"], language=None)
            if s["tool_args"]:
                st.caption("工具参数：")
                st.code(
                    json.dumps(s["tool_args"], ensure_ascii=False, indent=1),
                    language=None,
                )


def _render_final_answer(meta: dict) -> None:
    """轨迹末尾的最终回答（final_answer 汇总行，§11.2）。"""
    st.subheader("最终回答")
    if meta and meta.get("final_answer", "").strip():
        st.success(meta["final_answer"])
    else:
        st.caption("（该轨迹未记录最终回答——任务可能未完成或运行异常。）")


def main() -> None:
    """轨迹回放页主体。"""
    st.title("轨迹回放")
    st.caption("逐组逐任务查看一次运行的完整轨迹（§2.2-5 / §9.3.5 案例研究）")

    runs_root = st.session_state.get("runs_root", str(ddata.DEFAULT_RUNS_ROOT))

    @st.cache_data(show_spinner="正在加载评测数据…")
    def _load(root: str) -> list[dict]:
        return ddata.load_all_groups(Path(root))

    groups = _load(runs_root)
    if not groups:
        st.info("当前 runs 根目录下暂无合法 run 目录，请先运行实验或修改左侧目录。")
        st.stop()

    group = st.selectbox(
        "实验组（run 目录）",
        groups,
        format_func=_group_label,
        help="每组对应 runs/ 下的一个 run 目录",
    )

    run_dir = Path(group["run_dir"])
    rows = sorted(ddata.load_task_rows(run_dir), key=lambda r: r["task_id"])
    if not rows:
        st.warning(f"run 目录 {run_dir.name} 的 results.csv 为空或无任务行。")
        st.stop()

    task = st.selectbox(
        "任务（results.csv 行）",
        rows,
        format_func=_task_label,
        help="下拉后展示该任务基本信息与逐步轨迹",
    )

    st.divider()
    _render_task_info(task)

    meta = ddata.load_trace_meta(run_dir, task["task_id"])
    steps = ddata.load_trace_steps(run_dir, task["task_id"])
    _render_timeline(steps, meta)
    _render_final_answer(meta)


main()
