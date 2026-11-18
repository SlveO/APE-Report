# 实验第二阶段材料：九格参数探索

- [分析报告](reports/PARAMETER_EXPLORATION.md)
- [各口径分母](tables/scope_counts.csv)
- [16组全局排名](tables/global_parameter_rankings.csv)
- [九格×16参数排名及热图数据](tables/cell_parameter_rankings.csv)
- [各格候选与默认差值](tables/cell_candidates.csv)
- [完整性敏感性候选](tables/candidate_sensitivity.csv)
- [完整/缺失配对与样本范围](tables/parameter_pairing.csv)
- [逐题已保存分数及证据标签](tables/exploration_records.csv)
- [共同完整ID](tables/common_complete_ids.csv)
- [T/S及L/N分层走势](tables/marginal_effects.csv)
- [全局参数对照图](figures/global_parameter_comparison.png)

全部图另存SVG。完整性定义和源字段见报告，原候选副本位于sources/candidates_original.json。

复现（仓库根目录）：

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONNOUSERSITE="1"
& "C:/Users/Osheepne/anaconda3/python.exe" -s -B pc-analysis-output/scripts/analyze_exploration.py
```

依赖为已有NumPy/Matplotlib，环境见environment.json；本阶段无随机抽样，未执行Bootstrap。输入包含explore保存分数及前置审核标签，评分重算次数0。当前阶段完成后停止，验证阶段由用户安排。
