"""总览页 —— 全部实验组横向对比主表 / SR 柱状图 / 分难度成功率。

对应《项目方案.md》§9.3.1（主表：6 组 × SR/F1/成本/延迟/工具调用/失败数）
与 §9.3.2（分难度表：L1/L2/L3 分别的 SR）。数据为空时给出引导提示。
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ---- 路径自举：保证从任意工作目录 / Streamlit 沙箱均可导入 dashboard.* ----
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dashboard import data as ddata

# 主表列（数值已按展示精度四舍五入）
_MAIN_COLUMNS = [
    "组", "框架", "模型", "任务数", "通过/总数", "SR", "工具 F1",
    "成本 $/任务", "延迟 s/任务", "工具调用/任务", "失败数",
]
# 分难度表列
_DIFF_COLUMNS = [
    "组", "任务数", "L1 任务数", "L1 SR", "L2 任务数", "L2 SR",
    "L3 任务数", "L3 SR",
]


def _fw_label(framework: str) -> str:
    """框架英文名 → 中文展示标签（未知值原样返回）。"""
    return ddata.FRAMEWORK_LABELS.get(framework, framework or "—")


def _main_frame(groups: list[dict]) -> pd.DataFrame:
    """§9.3.1 主表：每实验组一行的核心指标。"""
    rows = []
    for g in groups:
        n_failed = int(g["n_tasks"]) - int(g["n_passed"])
        rows.append({
            "组": g["group"],
            "框架": _fw_label(g["framework"]),
            "模型": g["model"],
            "任务数": g["n_tasks"],
            "通过/总数": f"{g['n_passed']}/{g['n_tasks']}",
            "SR": round(g["sr"], 4),
            "工具 F1": round(g["f1"], 4),
            "成本 $/任务": round(g["cost_usd"], 4),
            "延迟 s/任务": round(g["latency_s"], 2),
            "工具调用/任务": round(g["tool_calls"], 2),
            "失败数": n_failed,
        })
    return pd.DataFrame(rows, columns=_MAIN_COLUMNS)


def _difficulty_frame(groups: list[dict]) -> pd.DataFrame:
    """§9.3.2 分难度 SR 表：每组的 L1/L2/L3 各自任务数与成功率。"""
    rows = []
    for g in groups:
        row = {"组": g["group"], "任务数": g["n_tasks"]}
        for level in ("L1", "L2", "L3"):
            info = g["by_difficulty"].get(level, {})
            row[f"{level} 任务数"] = int(info.get("n") or 0)
            row[f"{level} SR"] = round(float(info.get("sr") or 0.0), 4)
        rows.append(row)
    return pd.DataFrame(rows, columns=_DIFF_COLUMNS)


def _sr_bar(groups: list[dict]) -> "px.Figure":
    """SR 柱状图：各组成功率，按框架分组着色（§9.3.1 直观对比）。"""
    frame = pd.DataFrame([
        {
            "组": g["group"],
            "框架": _fw_label(g["framework"]),
            "模型": g["model"],
            "SR": round(g["sr"], 4),
        }
        for g in groups
    ])
    fig = px.bar(
        frame, x="组", y="SR", color="框架", barmode="group",
        color_discrete_sequence=px.colors.qualitative.Set2,
        title="各组任务成功率（按框架着色，0~1 表示比例）",
        labels={"SR": "成功率", "组": "实验组", "框架": "框架"},
        hover_data={"模型": True},
    )
    fig.update_traces(texttemplate="%{y:.0%}", textposition="outside")
    fig.update_yaxes(range=[0, 1.05], tickformat=".0%")
    fig.update_layout(legend_title_text="框架", height=420)
    return fig


def main() -> None:
    """总览页主体。"""
    st.title("总览")
    st.caption("6 组（3 框架 × 2 backbone 模型）控制变量对照 · 数据源见左侧 runs 根目录")

    runs_root = st.session_state.get("runs_root", str(ddata.DEFAULT_RUNS_ROOT))

    @st.cache_data(show_spinner="正在加载评测数据…")
    def _load(root: str) -> list[dict]:
        return ddata.load_all_groups(Path(root))

    groups = _load(runs_root)
    if not groups:
        st.info(
            "当前 runs 根目录下暂无合法 run 目录（需含 results.csv，且目录名不含 "
            "dryrun/failed）。\n\n请先运行实验（`agent-eval run` 或仓库 CLI）产出 "
            "runs/ 数据，或在左侧修改 runs 根目录。"
        )
        st.stop()

    st.subheader("主表：六组对比")
    st.dataframe(_main_frame(groups), hide_index=True)

    st.plotly_chart(_sr_bar(groups))

    st.subheader("分难度成功率（SR，0~1 表示比例，如 0.75 = 75%）")
    st.dataframe(_difficulty_frame(groups), hide_index=True)
    st.caption(
        "L1 单工具单轮 / L2 多轮状态依赖 / L3 多工具组合 + 约束推理（§4.3）；"
        "分档 SR 有助于定位框架差异集中在哪档难度（§9.3.2）。"
    )


main()
