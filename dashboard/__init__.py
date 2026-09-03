"""AgentEval Dashboard 包（《项目方案.md》§12 目录约定 / §9.3 报告结构）。

- app.py：Streamlit 多页入口（st.navigation：总览 / 轨迹回放 / 失败分析，
  侧边栏选择 runs 根目录）；
- data.py：数据加载与纯函数层（不依赖 streamlit，可单测）；
- pages/：三页 UI，各自为独立的 st.Page 目标脚本。

启动方式（仓库根目录下）：
    python -m streamlit run dashboard/app.py
"""
