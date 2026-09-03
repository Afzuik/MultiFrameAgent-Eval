"""judge —— LLM-as-Judge 轨迹评分模块（《项目方案.md》§8）。

包内组成：
- rubric.py：1~5 分评分标准常量（§8.2）；
- compress.py：轨迹压缩为 judge 输入文本（长 observation 截断 + 匿名化）；
- prompts.py：judge prompt 构造（system 含 rubric 全文与输出协议）；
- judge.py：单任务评分 judge_one + 双 judge 运行与一致性报告 judge_run
  （CLI：python -m judge.judge <run_dir>）。

质量约定（§8.3）：匿名化 / 双 judge / 一致性报告 / 失败兜底在本包实现；
人工锚定（§8.3④）留给 W4 接入，judge_run 的 per_task 已保留
score_a/score_b/score_mean 可直接与人工分对齐。
"""
