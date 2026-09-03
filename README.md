# AgentEval

轻量、可复现、**多框架横向对比**的 LLM Agent 评测体系，聚焦**多轮工具调用**任务。

> 让"同一个任务、同一个模型，不同 Agent 框架"的控制变量对比成为可能。

## 特性

- **任务集**：40 个多轮工具调用任务（差旅 travel / 电商 shop / 数据查询 analytics × L1/L2/L3 三档难度），全部带**可执行 verifier**（四原语：required_calls / forbidden_calls / final_state_checks / answer_checks）
- **有状态 mock 工具服务**：FastAPI + 确定性数据，18 个工具、权限与业务规则、实例隔离
- **统一评测装置**：薄编排器 + 子进程适配器 CLI 契约（崩溃隔离 / 超时强杀 / 断点续跑）
- **指标**：任务成功率 SR、工具调用 F1、轨迹分（LLM-as-judge）、成本、延迟
- **轨迹全量落盘**：JSONL 一行一步，双源采集（服务端工具日志 + 适配器消息日志）

## 快速开始

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e ".[dev]"

.venv/bin/python -m pytest          # 全部单元测试
.venv/bin/python -m tasks.validate  # 任务集双向自检（空轨迹 fail / GT pass）

# 无 API key 的 dry-run（fake_llm 回放 GT 路径，验证全链路）
EVAL_DRY_RUN=1 .venv/bin/python -m harness.orchestrator \
    --config configs/experiment_matrix.yaml --group R1

# 真实运行（需先设置 configs/models.yaml 中对应模型的 api_key_env 环境变量，
# 如 DEEPSEEK_API_KEY / MIMO_API_KEY / DASHSCOPE_API_KEY）
.venv/bin/python -m harness.orchestrator \
    --config configs/experiment_matrix.yaml --group R1
```

## 复现指南（30 分钟一组完整实验）

1. **装依赖**：`uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -e ".[dev]"`
2. **配置模型 key**（任选其一即可复现一组）：`export DEEPSEEK_API_KEY=...` 或 `export MIMO_API_KEY=...`（见 `configs/models.yaml` 各条目 `api_key_env`）
3. **跑一组实验**：`.venv/bin/python -m harness.orchestrator --config configs/experiment_matrix.yaml --group R1`（约 15~40 分钟，产出 `runs/<日期>_R1/`：results.csv + 40 份轨迹 + run_spec + config.yaml 快照）
4. **看结果**：
   - 汇总指标：`.venv/bin/python -m metrics.aggregate runs/<日期>_R1`
   - 失败分布：`.venv/bin/python -m analysis.failure_modes runs/<日期>_R1`
   - 自动报告：`.venv/bin/python -m analysis.report runs/<日期>_R1 -o 报告.md`
   - 历史结果 v1.1 离线复评：`.venv/bin/python -m analysis.reevaluate runs/<日期>_R1`
   - LLM-as-judge 评分：`.venv/bin/python -m judge.judge runs/<日期>_R1 --judge-models deepseek-v4-flash,deepseek-v4-flash`
   - Dashboard：`.venv/bin/python -m streamlit run dashboard/app.py`

产出目录 `runs/<日期>_<group>/`：`config.yaml`（配置快照）、`traces/{task_id}.jsonl`（轨迹）、`results.csv`（结果行）、`summary.json`（汇总指标）、`failure_modes.json`（失败分布）、`judge_scores.json`（judge 评分）。

断点续跑：重跑同一 group 会自动跳过已完成任务（按 task_id）；中断后直接重跑即可。

## 项目状态（W4 ✅）

- [x] W1~W3 全部（任务集 / mock 服务 / 三框架适配器 / 编排器 / F1 / 失败分类 / judge / 报告生成 / 6 组实验）
- [x] v1.1/v1.2 评测器修正（数字归一化、措辞宽松、空白归一化、端点抖动检测）
- [x] Streamlit dashboard：总览 / 轨迹回放 / 失败分析三页（`streamlit run dashboard/app.py`）
- [x] 最终实验报告 `docs/REPORT_FINAL.md` + 6 例轨迹案例研究 `docs/CASESTUDY.md`
- [x] judge 全 5 组评分（R1 4.39 / S1 4.34 / R2 4.72 / S2 4.34 / O2 4.45）
- [x] MIT LICENSE + 30 分钟复现指南；全仓 268 测试全绿
- [ ] 后续可选：OpenHands 方案 A 回归、judge 人工锚定、GitHub push、样例数据

## 文档

- `项目方案.md` —— 完整项目方案（任务集 / mock 服务 / harness / 指标 / judge / 实验设计 / 里程碑）
- `任务设计指南.md` —— 任务样本设计方法论（难度维度、陷阱库、自检清单）
- `tool_server/tool_registry.json` —— 工具定义 + 确定性 fixture 数据（唯一数据来源）

## 目录结构

见《项目方案.md》§12。

## 致谢（设计灵感）

τ-bench（verifier 判定模式）、Berkeley BFCL（参数校验）、SWE-bench（报告结构）、tool-eval-bench、langchain-ai/agentevals、hodoscope。
