# 实验第三阶段材料：验证与冻结规则

- [分析报告](reports/VALIDATION_REPORT.md)
- [口径与分母](tables/scope_counts.csv)
- [候选复用计数](tables/generation_reuse.csv)
- [1071个逻辑候选视图与源行](tables/logical_candidate_views.csv)
- [全局分数及区间](tables/validation_policy_scores.csv)
- [主要配对差及区间](tables/validation_policy_paired.csv)
- [九格决定、配对区间及分母](tables/validation_cell_decisions.csv)
- [逐输入原分数引用](tables/input_policy_scores.csv)
- [保存规则分数核对](tables/saved_rule_checks.csv)
- [完整性敏感性规则](tables/rule_sensitivity.csv)
- [新探索候选缺少的验证条件](tables/new_exploration_candidate_coverage.csv)
- [Bootstrap诊断与簇数](tables/bootstrap_diagnostics.csv)
- [冻结时间线](tables/freeze_timeline.csv)

原候选与规则副本位于sources/，敏感性规则另存sensitivity_rule_replays.json。图表在figures/，含PNG和SVG。环境、版本、随机种子及统计方法见environment.json。

复现（仓库根目录）：

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONNOUSERSITE="1"
& "C:/Users/Osheepne/anaconda3/python.exe" -s -B pc-analysis-output/scripts/analyze_validation.py
```

依赖均为已有NumPy/Matplotlib。本次直接复用validation保存score，评分重算0次；全部新文件位于本阶段目录。未使用正式测试分数，未运行GPU或修改冻结规则。
