# 实验第七阶段：Lost in the Middle

本阶段完成；研究状态SUBMITTED，执行停止于本阶段。

- [阶段报告](reports/LITM_REPORT.md)
- [审核记录](reports/AUDIT.md)
- [官方首行质量主表](tables/main_quality.csv)
- [方法配对差和胜平负](tables/main_quality_paired.csv)
- [位置差](tables/main_position_contrasts.csv)
- [位置效应的方法差](tables/main_paired_position_contrasts.csv)
- [500题九条件分数](tables/per_question_scores.csv)
- [1500输入及文档顺序](tables/membership.csv)
- [4500原答案和逐请求计时](tables/per_request_records.csv)
- [501条首行修正及原答案](tables/scoring_changed_records.csv)
- [评分口径对配对结论的影响](tables/method_pair_scoring_scope_shift.csv)
- [九个官方入口实例](tables/official_entrypoint_fixtures.csv)
- [效率均值/中位数/P90及CI](tables/efficiency_descriptive.csv)
- [配对加速比与时间差](tables/efficiency_paired.csv)
- [上限与EOS](tables/generation_limits.csv)
- [同输出长度与时延分组](tables/paired_output_length_groups.csv)
- [案例全部位置及原答案](tables/selected_case_positions.csv)
- [500重采样单位](tables/bootstrap_questions.csv)
- [原汇总核对](tables/original_summary_crosscheck.csv)
- [来源与公式](tables/provenance.csv)
- [字段含义](tables/field_dictionary.csv)
- [官方评分来源查阅](sources/official_scoring_notes.md)
- [环境与种子](environment.json)

figures/含8组PNG/SVG图。质量曲线来自main_quality.csv，评分口径图来自quality.csv，位置差来自main_position_contrasts.csv，上限图来自generation_limits/scoring_scope_impact，案例图来自selected_case_positions，长度时延图来自per_request_records.csv。主指标为官方首行EM，全文分数另列。

本阶段依赖已交付前置审核的scoring_litm_input_checks.csv、scoring_litm_first_line_changed_IDs.csv和九实例表；复用修正标签，保持原score。CPU命令从仓库根目录运行：

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONNOUSERSITE="1"
$env:PYTHONDONTWRITEBYTECODE="1"
& "C:/Users/Osheepne/anaconda3/python.exe" -s -B pc-analysis-output/scripts/analyze_litm_stage7.py
& "C:/Users/Osheepne/anaconda3/python.exe" -s -B pc-analysis-output/scripts/plot_litm_stage7.py
& "C:/Users/Osheepne/anaconda3/python.exe" -s -B pc-analysis-output/scripts/summarize_litm_stage7.py
```

使用已有依赖，版本见environment.json。seed20261116，5000次问题配对Bootstrap，三位置/三方法共同保留。总体每方法1500请求、500问题；质量先按题平均三位置，效率分位数在实际请求上计算。新增文件、缓存和临时目录均位于pc-analysis-output；未执行Doc2Dial阶段。
