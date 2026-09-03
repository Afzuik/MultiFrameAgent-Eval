"""失败分析页 —— 全部组的失败模式合并矩阵 / 案例明细 / 堆叠柱状图。

对应《项目方案.md》§10.2（失败任务 × 类别分布、框架 × 失败模式交叉矩阵）
与 §9.3.3（失败模式分布）。类别展示用 analysis.failure_modes 的
CATEGORY_LABELS 中文标签（六类，§10.1）。
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---- 路径自举：保证从任意工作目录 / Streamlit 沙箱均可导入 dashboard.* ----
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.failure_modes import CATEGORY_LABELS
from dashboard import data as ddata

# 六类失败模式（键序即展示顺序，§10.1）
_CATEGORIES = list(CATEGORY_LABELS)


def _fw_label(framework: str) -> str:
    return ddata.FRAMEWORK_LABELS.get(framework, framework or "—")


def _matrix_frame(groups: list[dict]) -> pd.DataFrame:
    """合并矩阵：行 = 实验组，列 = 六类失败模式计数（中文标签）。"""
    rows = []
    for g in groups:
        by_cat = (g.get("failure_modes") or {}).get("by_category") or {}
        row = {
            "组": f"{g['group']} · {_fw_label(g['framework'])}",
            "失败任务": int((g.get("failure_modes") or {}).get("failed") or 0),
        }
        for cat in _CATEGORIES:
            row[CATEGORY_LABELS[cat]] = int(by_cat.get(cat) or 0)
        rows.append(row)
    columns = ["组", "失败任务"] + [CATEGORY_LABELS[c] for c in _CATEGORIES]
    return pd.DataFrame(rows, columns=columns)


def _detail_rows(groups: list[dict]) -> list[dict]:
    """拍平全部组的失败案例：task_id/组/框架/难度/类别/状态/findings。"""
    details: list[dict] = []
    for g in groups:
        for item in (g.get("failure_modes") or {}).get("details") or []:
            details.append({
                "组": g["group"],
                "任务ID": item.get("task_id", ""),
                "框架": _fw_label(str(item.get("framework") or "")),
                "难度": item.get("difficulty", ""),
                "类别": CATEGORY_LABELS.get(
                    str(item.get("category") or ""), item.get("category", "")
                ),
                "状态": item.get("status", ""),
                "失败原因": "; ".join(item.get("findings") or []),
            })
    return details


def _stacked_bar(groups: list[dict]) -> "go.Figure":
    """堆叠柱状图：按框架汇总六类失败模式计数（§10.2 框架特征性弱点）。"""
    per_fw: dict[str, dict[str, int]] = {}
    for g in groups:
        bucket = per_fw.setdefault(g["framework"], {c: 0 for c in _CATEGORIES})
        by_cat = (g.get("failure_modes") or {}).get("by_category") or {}
        for cat in _CATEGORIES:
            bucket[cat] += int(by_cat.get(cat) or 0)
    fig = go.Figure()
    for cat in _CATEGORIES:
        fig.add_trace(go.Bar(
            name=CATEGORY_LABELS[cat],
            x=[_fw_label(k) for k in per_fw],
            y=[per_fw[k][cat] for k in per_fw],
        ))
    fig.update_layout(
        barmode="stack",
        title="每框架各失败类别计数（跨组汇总，可点击图例筛选）",
        xaxis_title="框架",
        yaxis_title="失败任务数",
        legend_title_text="失败类别",
        height=460,
    )
    return fig


def main() -> None:
    """失败分析页主体。"""
    st.title("失败分析")
    st.caption("全部组失败模式合并矩阵 + 每案例明细 + 框架级分布（§9.3.3 / §10.2）")

    runs_root = st.session_state.get("runs_root", str(ddata.DEFAULT_RUNS_ROOT))

    @st.cache_data(show_spinner="正在加载评测数据…")
    def _load(root: str) -> list[dict]:
        return ddata.load_all_groups(Path(root))

    groups = _load(runs_root)
    if not groups:
        st.info("当前 runs 根目录下暂无合法 run 目录，请先运行实验或修改左侧目录。")
        st.stop()

    st.subheader("失败模式合并矩阵（组 × 六类失败模式）")
    st.dataframe(_matrix_frame(groups), hide_index=True)
    st.caption(
        "类别判定口径见 analysis.failure_modes（§10.1）：权限/规则违反、解析失败、"
        "预算耗尽、工具选择错误、参数错误、规划失败（兜底）。"
    )

    st.plotly_chart(_stacked_bar(groups))

    st.subheader("失败案例明细")
    details = _detail_rows(groups)
    if not details:
        st.caption("当前无失败案例（全部通过或任务不在任务库而无法归类）。")
    else:
        labels = sorted({d["类别"] for d in details})
        selected = st.multiselect(
            "按类别筛选案例（默认全部）",
            options=labels,
            default=labels,
            format_func=lambda x: x,
        )
        frame = pd.DataFrame([d for d in details if d["类别"] in selected])
        st.dataframe(frame, hide_index=True)
        st.caption("点击行可在左侧「轨迹回放」页定位该任务（组 + 任务ID）细看轨迹。")


main()
