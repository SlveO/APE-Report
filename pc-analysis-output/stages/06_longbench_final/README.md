# 实验第六阶段：LongBench正式五方法比较

本阶段完成，研究状态SUBMITTED；执行停止于本阶段。

- [分析报告](reports/LONGBENCH_FINAL_REPORT.md)
- [审核记录](reports/AUDIT.md)
- [五方法质量主表](tables/main_quality.csv)
- [全部方法对质量差](tables/main_quality_paired.csv)
- [动态与固定主要比较](tables/primary_dynamic_fixed.csv)
- [510题五方法成员与分层标签](tables/membership.csv)
- [2550条逐请求原分数与计时](tables/per_request_records.csv)
- [动态固定逐题原答案](tables/dynamic_fixed_per_sample.csv)
- [规则分支解释](tables/mapping_interpretation.csv)
- [全部分支/N/重合层统计](tables/quality_paired.csv)
- [效率均值、中位数、P90及区间](tables/main_efficiency.csv)
- [配对时间差和两种加速比](tables/efficiency_paired.csv)
- [输出长度与时延关系](tables/paired_output_length_groups.csv)
- [上限与EOS](tables/generation_limits.csv)
- [原答案案例](tables/selected_cases.csv)
- [来源组及连通组件](tables/bootstrap_membership.csv)
- [原汇总对照](tables/original_summary_crosscheck.csv)
- [关键数字来源与公式](tables/provenance.csv)
- [字段及权重解释](tables/field_dictionary.csv)
- [环境和种子](environment.json)

figures/含8张静态图，各保存PNG和SVG；sources/保留原冻结规则。质量图的数据来自main_quality、main_quality_paired、mapping_interpretation；效率图来自main_efficiency、dynamic_fixed_per_sample；输入图来自membership。主F1按三数据集等权，分支内展示按实际题目混合汇总；每项均保留分母、单位与区间。

从仓库根目录复现：

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONNOUSERSITE="1"
$env:PYTHONDONTWRITEBYTECODE="1"
& "C:/Users/Osheepne/anaconda3/python.exe" -s -B pc-analysis-output/scripts/analyze_longbench_final.py
& "C:/Users/Osheepne/anaconda3/python.exe" -s -B pc-analysis-output/scripts/plot_longbench_final.py
& "C:/Users/Osheepne/anaconda3/python.exe" -s -B pc-analysis-output/scripts/summarize_longbench_final.py
```

使用现有Python3.12.7、NumPy1.26.4、Matplotlib3.9.2，无新增依赖。seed20261116，5000次配对聚类Bootstrap；分组、子序列、软件版本及分析提交见environment.json。各脚本输出和缓存位于pc-analysis-output，本阶段只读取已恢复证据。F1重新评分0次。
