"""AgentEval Dashboard —— Streamlit 多页入口（《项目方案.md》§12 / §9.3）。

启动方式（仓库根目录下）：
    python -m streamlit run dashboard/app.py

职责：
- st.set_page_config（宽屏、中文标题）；
- 侧边栏选择 runs 根目录（默认仓库 runs/，三页共享 session_state）；
- st.navigation 三页导航：总览 / 轨迹回放 / 失败分析。
"""
import sys
from pathlib import Path

import streamlit as st

# ---- 路径自举：保证从任意工作目录 / Streamlit 沙箱均可导入 dashboard.* ----
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dashboard.data import DEFAULT_RUNS_ROOT


def _render_sidebar() -> None:
    """侧边栏：runs 根目录选择，写入 session_state["runs_root"] 供三页共享。"""
    with st.sidebar:
        st.title("🧪 AgentEval")
        st.caption("多框架 Agent 工具调用评测看板")
        st.divider()
        st.session_state.setdefault("runs_root", str(DEFAULT_RUNS_ROOT))
        st.text_input(
            "runs 根目录",
            key="runs_root",
            help="存放 run 目录（含 results.csv 等）的数据源根目录，见《项目方案.md》§11.1",
        )
        if not Path(st.session_state["runs_root"]).is_dir():
            st.error("该目录不存在，请检查路径或先运行实验生成 runs/ 数据。")
        else:
            st.caption("当前目录有效：按 run 目录名识别实验组（跳过 dryrun/failed）。")
        st.divider()
        st.caption("数据来源：仓库 runs/（.gitignore 忽略，本地实验产物）")


def main() -> None:
    """多页入口：总览 → 轨迹回放 → 失败分析（对应 §9.3 报告章节）。"""
    st.set_page_config(page_title="AgentEval 评测看板", page_icon="🧪", layout="wide")
    _render_sidebar()
    nav = st.navigation(
        [
            st.Page("pages/1_总览.py", title="总览", icon="📊", default=True),
            st.Page("pages/2_轨迹回放.py", title="轨迹回放", icon="🎬"),
            st.Page("pages/3_失败分析.py", title="失败分析", icon="🔬"),
        ],
        position="sidebar",
    )
    nav.run()


if __name__ == "__main__":
    main()
