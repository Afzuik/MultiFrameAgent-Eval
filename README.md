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

# 真实运行（需先设置 configs/models.yaml 中对应模型的 api_key_env 环境变量）
.venv/bin/python -m harness.orchestrator \
    --config configs/experiment_matrix.yaml --group R1
```

产出目录 `runs/<日期>_<group>/`：`config.yaml`（配置快照）、`traces/{task_id}.jsonl`（轨迹）、`results.csv`（结果行）、`summary.json`（汇总指标）。

## 项目状态（W2 ✅）

- [x] W1 全部（见 git log：任务集 / mock 服务 / ReAct 适配器 / 编排器 v0 / metrics）
- [x] smolagents 适配器：CodeAgent + LiteLLMModel，同 §6.3 CLI 契约，fake_llm dry-run 一致
- [x] Tool-Call F1（§7.2 口径）：metrics/tool_f1.py，results.csv 新增 f1_recall/f1_precision/f1 三列
- [x] 失败模式分类 v0（§10.1 六类）：analysis/failure_modes.py + `failure_modes.json`
- [x] 实验矩阵：R1（react）/ S1（smolagents）× deepseek-v4-flash × 全量 40 任务
- [x] 验收：dry-run 两组 80/80 PASS，SR/F1/成本/延迟四指标齐（`runs/2026-09-03_R1|S1/`），全仓 229 测试全绿
- [ ] 真实模型实验：设置 `DEEPSEEK_API_KEY` 后去掉 `EVAL_DRY_RUN=1` 重跑即可
- [ ] W3：OpenHands 适配器、LLM-as-judge、全量 6 组实验

## 文档

- `项目方案.md` —— 完整项目方案（任务集 / mock 服务 / harness / 指标 / judge / 实验设计 / 里程碑）
- `任务设计指南.md` —— 任务样本设计方法论（难度维度、陷阱库、自检清单）
- `tool_server/tool_registry.json` —— 工具定义 + 确定性 fixture 数据（唯一数据来源）

## 目录结构

见《项目方案.md》§12。

## 致谢（设计灵感）

τ-bench（verifier 判定模式）、Berkeley BFCL（参数校验）、SWE-bench（报告结构）、tool-eval-bench、langchain-ai/agentevals、hodoscope。
