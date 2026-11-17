# 实验第一阶段：LongBench完整600题三方法基线

- [结果与官方对比分析](reports/BASELINE_REPORT.md)
- [质量基线CSV](tables/baseline_quality.csv)
- [方法间配对差异及95%CI](tables/baseline_quality_paired.csv)
- [官方表值与本次差值](tables/official_comparison.csv)
- [官方全部参考行及设置](tables/official_reference.csv)
- [设置差异](tables/official_setting_comparison.csv)
- [效率均值/中位/P90及CI](tables/baseline_efficiency.csv)
- [配对耗时与加速比](tables/baseline_efficiency_paired.csv)
- [逐题可追溯结果](tables/baseline_per_sample.csv)
- [原答案案例](tables/baseline_cases.csv)
- [质量图](figures/baseline_quality.png)
- [冷请求时延图](figures/baseline_latency.png)
- [官方对照图](figures/official_comparison.png)

图表均另存SVG，官方代码原文及获取URL保存在sources/；env、种子和范围见environment.json。

复现（在仓库根目录）：

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONNOUSERSITE="1"
& "C:/Users/Osheepne/anaconda3/python.exe" -s -B pc-analysis-output/scripts/analyze_baseline.py
```

使用已有NumPy1.26.4和Matplotlib3.9.2；原始证据只读，所有新增文件均在pc-analysis-output/。前置审核另存01_audit，本目录按实验顺序编号为第一阶段；后续实验未在本脚本执行。研究状态SUBMITTED。
