# 实验第四阶段：长度检查材料

- [分析报告](reports/LENGTH_CHECK_REPORT.md)
- [样本范围与分母](tables/scope_counts.csv)
- [60组配对原分数、答案、参数与计时](tables/paired_records.csv)
- [映射及相同输出检查](tables/mapping_and_output_checks.csv)
- [宏平均、六格质量与95%区间](tables/quality_paired.csv)
- [数据集分层质量](tables/dataset_cell_quality.csv)
- [效率均值、中位数、P90及区间](tables/efficiency_descriptive.csv)
- [配对耗时与加速比](tables/efficiency_paired.csv)
- [证据缺失的六组配对](tables/evidence_missing_pairs.csv)
- [原答案案例](tables/selected_cases.csv)
- [重采样单位与簇数](tables/bootstrap_units.csv)
- [生成上限及EOS计数](tables/generation_limits.csv)
- [命中生成上限的源记录](tables/generation_cap_records.csv)
- [模型加载记录](tables/model_load_workers.csv)

图表均保存PNG/SVG；原冻结规则副本位于sources/。environment.json记录环境、随机种子和统计权重，STAGE_STATUS.json记录停止点。

复现：

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONNOUSERSITE="1"
& "C:/Users/Osheepne/anaconda3/python.exe" -s -B pc-analysis-output/scripts/analyze_length_check.py
```

使用已有NumPy/Matplotlib。所有新文件限于pc-analysis-output；输入为length_check保存score/timing、前置审核标签、冻结规则和固定验证ID顺序。评分重算0次。
