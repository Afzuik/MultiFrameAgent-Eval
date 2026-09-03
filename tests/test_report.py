"""test_report —— analysis.report（W3 markdown 实验报告生成器）单元测试。

覆盖：
1. build_report 汇总两个 run_dir（手写 summary.json + failure_modes.json +
   results.csv 小样例，不依赖 aggregate / failure_modes 运行）：断言主表含两行
   与正确 SR/F1 数字、分难度表、失败模式分布的中文类别标签、失败/成功案例摘要、
   judge 占位"未运行 judge"、成本效率散点注明 W4；
2. 含 judge_scores.json 的 run_dir：断言 agreement_exact / agreement_pm1 /
   mean_score 出现；
3. out_path 写盘：返回路径文本且文件内容含报告片段；
4. CLI：main([run_dir...]) 打印到 stdout；main([..., "-o", path]) 写盘；
5. 容错：summary 缺 mean_f1 时主表 F1 单元格显示 "-"。
"""

import json

from analysis.failure_modes import CATEGORIES
from analysis.report import build_report, main

# 主表 F1 单元格数字（mean_f1、recall、precision 各 round 4 位后拼格）
REACT_ROW = (
    "| r1 | R1 | react | deepseek-v4-flash | 0.6667 | 0.8 (0.75/1.0) | 0.0012 | 10.5 | 2.333 | 1 |"
)
SMOLAGENTS_ROW = (
    "| s1 | S1 | smolagents | deepseek-v4-flash | 0.5 | 0.6 (0.6/1.0) | 0.002 | 20.5 | 5.25 | 1 |"
)


# ---------- run 目录构造工厂（全部手写 JSON/CSV，不依赖指标管道） ----------


def _write_json(path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _six_categories(filled: dict) -> dict:
    """构造六类齐全的 by_category（其余为 0）。"""
    return {c: filled.get(c, 0) for c in CATEGORIES}


def _write_react_run(run_dir) -> None:
    """react（R1）：3 任务 2 通过；失败 sh_002 属 planning_failure。"""
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "group": "R1",
        "framework": "react",
        "model": "deepseek-v4-flash",
        "n_tasks": 3,
        "n_completed": 3,
        "n_passed": 2,
        "sr": 0.6667,
        "mean_cost_usd": 0.0012,
        "total_cost_usd": 0.0036,
        "mean_wall_time_s": 10.5,
        "mean_tool_calls": 2.333,
        "mean_f1": 0.8,
        "mean_f1_recall": 0.75,
        "mean_f1_precision": 1.0,
        "by_difficulty": {
            "L1": {"n": 1, "sr": 1.0},
            "L2": {"n": 1, "sr": 0.0},
            "L3": {"n": 1, "sr": 1.0},
        },
        "failures": [
            {
                "task_id": "sh_002",
                "difficulty": "L2",
                "status": "completed",
                "findings": "missing_call:get_order_status",
            },
        ],
    }
    _write_json(run_dir / "summary.json", summary)

    dist = {
        "total": 3,
        "failed": 1,
        "by_category": _six_categories({"planning_failure": 1}),
        "by_category_pct": _six_categories({"planning_failure": 100.0}),
        "by_framework": {
            "react": {
                "failed": 1,
                "by_category": _six_categories({"planning_failure": 1}),
            },
        },
        "details": [
            {
                "task_id": "sh_002",
                "framework": "react",
                "difficulty": "L2",
                "category": "planning_failure",
                "status": "completed",
                "findings": ["missing_call:get_order_status"],
            },
        ],
    }
    _write_json(run_dir / "failure_modes.json", dist)

    # results.csv：tr_001/tr_007 通过、sh_002 失败（成功样本应为 wall 最大的 tr_007）
    rows = [
        (
            "task_id,group,framework,model,difficulty,status,passed,findings,cost_usd,"
            "wall_time_s,n_tool_calls,n_steps"
        ),
        "tr_001,R1,react,deepseek-v4-flash,L1,completed,True,,0.0,5.0,1,2",
        (
            "sh_002,R1,react,deepseek-v4-flash,L2,completed,False,"
            "missing_call:get_order_status,0.0,12.0,1,2"
        ),
        "tr_007,R1,react,deepseek-v4-flash,L3,completed,True,,0.0,15.0,2,3",
    ]
    (run_dir / "results.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_smolagents_run(run_dir) -> None:
    """smolagents（S1）：2 任务 1 通过；失败 tr_012 属 parse_failure。"""
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "group": "S1",
        "framework": "smolagents",
        "model": "deepseek-v4-flash",
        "n_tasks": 2,
        "n_completed": 2,
        "n_passed": 1,
        "sr": 0.5,
        "mean_cost_usd": 0.002,
        "total_cost_usd": 0.004,
        "mean_wall_time_s": 20.5,
        "mean_tool_calls": 5.25,
        "mean_f1": 0.6,
        "mean_f1_recall": 0.6,
        "mean_f1_precision": 1.0,
        # L3 条目缺失 → 分难度表该列应显示 "-"
        "by_difficulty": {
            "L1": {"n": 1, "sr": 1.0},
            "L2": {"n": 1, "sr": 0.0},
        },
        "failures": [
            {
                "task_id": "tr_012",
                "difficulty": "L2",
                "status": "completed",
                "findings": "output_not_parseable",
            },
        ],
    }
    _write_json(run_dir / "summary.json", summary)

    dist = {
        "total": 2,
        "failed": 1,
        "by_category": _six_categories({"parse_failure": 1}),
        "by_category_pct": _six_categories({"parse_failure": 100.0}),
        "by_framework": {
            "smolagents": {
                "failed": 1,
                "by_category": _six_categories({"parse_failure": 1}),
            },
        },
        "details": [
            {
                "task_id": "tr_012",
                "framework": "smolagents",
                "difficulty": "L2",
                "category": "parse_failure",
                "status": "completed",
                "findings": ["output_not_parseable"],
            },
        ],
    }
    _write_json(run_dir / "failure_modes.json", dist)

    rows = [
        (
            "task_id,group,framework,model,difficulty,status,passed,findings,cost_usd,"
            "wall_time_s,n_tool_calls,n_steps"
        ),
        "tr_011,S1,smolagents,deepseek-v4-flash,L1,completed,True,,0.0,8.0,2,3",
        (
            "tr_012,S1,smolagents,deepseek-v4-flash,L2,completed,False,"
            "output_not_parseable,0.0,20.0,1,2"
        ),
    ]
    (run_dir / "results.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _make_two_runs(tmp_path):
    """造 r1（react）+ s1（smolagents）两个 run_dir，返回 [r1, s1]。"""
    r1 = tmp_path / "r1"
    s1 = tmp_path / "s1"
    _write_react_run(r1)
    _write_smolagents_run(s1)
    return [r1, s1]


# ---------- build_report：主表 / 分难度 / 失败分布 / 案例 / 散点 / judge ----------


def test_build_report_two_runs(tmp_path):
    runs = _make_two_runs(tmp_path)
    text = build_report(runs)

    # 标题 + 章节骨架
    assert text.startswith("# AgentEval 实验报告")
    assert "生成时间：" in text
    for section in (
        "## 1. 主表",
        "## 2. 分难度 SR",
        "## 3. 失败模式分布",
        "## 4. 成本效率散点",
        "## 5. 案例研究",
        "## 6. judge 质量",
    ):
        assert section in text

    # 主表：两行 + 正确 SR / F1 数字
    assert REACT_ROW in text
    assert SMOLAGENTS_ROW in text

    # 分难度表：react 三档齐全，smolagents 缺 L3 显示 "-"
    assert "| r1 | 1.0 | 0.0 | 1.0 |" in text
    assert "| s1 | 1.0 | 0.0 | - |" in text

    # 失败模式分布：中文类别标签 + 按框架分组（react 在前、smolagents 在后）
    assert "| 权限/规则违反 | 0 | 0 | 0 |" in text
    assert "| 解析失败 | 0 | 1 | 1 |" in text
    assert "| 规划失败 | 1 | 0 | 1 |" in text
    assert "| 合计 | 1 | 1 | 2 |" in text

    # 成本效率散点：注明交互版在 W4
    assert "| r1 | 0.6667 | 0.0012 | 10.5 |" in text
    assert "dashboard 交互版在 W4" in text or "dashboard 实现" in text

    # 案例研究：失败 = details 第一条（sh_002 / tr_012），成功 = wall_time_s 最大
    assert "失败案例：`sh_002`" in text
    assert "类别：规划失败（planning_failure）" in text
    assert "耗时：12.0 s" in text
    assert "成功案例：`tr_007`" in text
    assert "耗时：15.0 s" in text
    assert "失败案例：`tr_012`" in text
    assert "成功案例：`tr_011`" in text

    # judge 占位：两个 run 都没有 judge_scores.json → 各标一次未运行
    assert text.count("未运行 judge") == 2


def test_build_report_judge_scores_present(tmp_path):
    """judge_scores.json 存在 → agreement_exact/pm1 与 mean_score 出现。"""
    runs = _make_two_runs(tmp_path)
    _write_json(
        runs[1] / "judge_scores.json",
        {
            "n_samples": 10,
            "agreement_exact": 0.8,
            "agreement_pm1": 1.0,
            "mean_score": 3.6,
        },
    )
    text = build_report(runs)

    assert "agreement_exact=0.8、agreement_pm1=1.0、mean_score=3.6" in text
    # r1 仍无 judge_scores.json → 该 run 保持占位
    assert text.count("未运行 judge") == 1


def test_build_report_out_path_writes(tmp_path):
    """out_path 非空 → 写盘并返回路径文本。"""
    runs = _make_two_runs(tmp_path)
    out = tmp_path / "reports" / "exp_r1_s1.md"
    assert build_report(runs, out) == str(out)
    assert out.is_file()

    content = out.read_text(encoding="utf-8")
    assert content.startswith("# AgentEval 实验报告")
    assert REACT_ROW in content
    assert SMOLAGENTS_ROW in content


def test_build_report_missing_mean_f1_uses_dash(tmp_path):
    """容错：summary 缺 mean_f1 → 主表 F1 单元格显示 "-"。"""
    run_dir = tmp_path / "rx"
    _write_react_run(run_dir)

    # 去掉 mean_f1 / recall / precision 三键（模拟无 F1 列的旧 summary）
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    for key in ("mean_f1", "mean_f1_recall", "mean_f1_precision"):
        summary.pop(key, None)
    _write_json(run_dir / "summary.json", summary)

    text = build_report([run_dir])
    assert "| rx | R1 | react | deepseek-v4-flash | 0.6667 | - |" in text


# ---------- CLI ----------


def test_cli_prints_to_stdout(tmp_path, capsys):
    runs = _make_two_runs(tmp_path)
    assert main([str(runs[0]), str(runs[1])]) == 0

    out = capsys.readouterr().out
    assert out.startswith("# AgentEval 实验报告")
    assert REACT_ROW in out
    assert SMOLAGENTS_ROW in out


def test_cli_out_path_writes(tmp_path, capsys):
    runs = _make_two_runs(tmp_path)
    out = tmp_path / "cli_report.md"
    assert main([str(runs[0]), "-o", str(out)]) == 0
    assert out.is_file()

    content = out.read_text(encoding="utf-8")
    assert "| r1 | R1 | react |" in content
    # stdout 打印的是写盘后的路径文本
    assert str(out) in capsys.readouterr().out
