# 实验第八阶段：Doc2Dial

本阶段完成,研究状态SUBMITTED；执行停止于本阶段。

- [阶段报告](reports/DOC2DIAL_REPORT.md)
- [审核记录](reports/AUDIT.md)
- [语料BLEU主表](tables/main_quality.csv)
- [资料聚类质量敏感性](tables/main_quality_document_sensitivity.csv)
- [方法/冷热质量配对](tables/main_quality_paired.csv)
- [质量配对的资料敏感性](tables/main_quality_paired_document_sensitivity.csv)
- [500目标成员与簇](tables/membership.csv)
- [完整历史及输入处理](tables/target_inputs.csv)
- [3000原答案/参考/计时](tables/per_request_records.csv)
- [每目标ngram充分统计](tables/bleu_sufficient_statistics.csv)
- [1500冷热对](tables/cold_hot_pairs.csv)
- [热重新生成及事件证据](tables/hot_generation_evidence.csv)
- [效率均值/中位数/P90/CI](tables/main_efficiency.csv)
- [缓存收益](tables/cache_effects.csv)
- [全部配对加速比](tables/efficiency_paired.csv)
- [实际L/N与映射](tables/input_distribution.csv)
- [历史角色/参考重复数量](tables/history_condition_counts.csv)
- [生成上限与EOS](tables/generation_limits.csv)
- [完整原答案案例](tables/selected_cases.csv)
- [语言问题原始证据](sources/reference_language_evidence.json)
- [官方评分与模板来源](sources/official_scoring_and_template.md)
- [关键数字来源及公式](tables/provenance.csv)
- [字段含义](tables/field_dictionary.csv)
- [重采样簇与子序列](tables/bootstrap_diagnostics.csv)
- [环境与种子](environment.json)

figures/含8组PNG/SVG,源数据为quality/quality_paired、efficiency_descriptive/efficiency_paired、generation_limits、membership及dynamic_fixed_per_target各CSV。指标BLEU为0–100原值；13a分词数和模型token分开记录。主分析按对话,资料作敏感性。

复现依赖已交付的前置审核CSV及scripts/audit_stage1_scoring.py纯函数。评分helper通过runpy加载,main未执行。CPU不需要安装SacreBLEU；对照设置为服务器2.4.3签名,六语料统计和分数核对保存在original_summary_crosscheck.csv。

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONNOUSERSITE="1"
$env:PYTHONDONTWRITEBYTECODE="1"
& "C:/Users/Osheepne/anaconda3/python.exe" -s -B pc-analysis-output/scripts/analyze_doc2dial_stage8.py
& "C:/Users/Osheepne/anaconda3/python.exe" -s -B pc-analysis-output/scripts/plot_doc2dial_stage8.py
& "C:/Users/Osheepne/anaconda3/python.exe" -s -B pc-analysis-output/scripts/summarize_doc2dial_stage8.py
```

使用既有Python、NumPy、Matplotlib,无新依赖。seed20261116,5000次配对簇Bootstrap,每次重新累计语料计数。六条件共同保留；中位数/P90基于各条件500真实请求。缓存、临时文件和产物均位于本阶段目录。

[交付检查结果](logs/validation.json)记录16项表格分母、六组服务器评分一致性、1500冷热配对、单资料子组空CI及图表/引用检查。[检查脚本](../../scripts/validate_doc2dial_stage8.py)可在以上分析和报告生成后运行：

```powershell
& "C:/Users/Osheepne/anaconda3/python.exe" -s -B pc-analysis-output/scripts/validate_doc2dial_stage8.py
```
