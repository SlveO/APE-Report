# 第1阶段材料索引

当前阶段：数据、配置、划分及评分审核。交付状态：完成并登记发现。研究状态：SUBMITTED。

## 1 阅读顺序

- [综合审核与下一步需求](reports/AUDIT.md)
- [数据字典与任务连接](reports/STAGE1_CORE.md)
- [划分及开发构造](reports/STAGE1_SPLITS.md)
- [评分及参考连接](reports/STAGE1_SCORING.md)
- [配置、事件与生成停止](reports/STAGE1_RUNTIME.md)

## 2 表格索引

|类别|主要文件|用途|
|---|---|---|
|总发现|[audit_findings.csv](tables/audit_findings.csv) / [follow_up_requirements.csv](tables/follow_up_requirements.csv)|现象、影响、处理方式和证据|
|任务和字段|[record_stage_counts.csv](tables/record_stage_counts.csv) / [task_record_join.csv](tables/task_record_join.csv) / [data_dictionary.csv](tables/data_dictionary.csv)|数量、源行和冻结输入连接|
|快照|[snapshot_lineage.csv](tables/snapshot_lineage.csv)|36项来源的manifest别名关系|
|真实证据删除|[construction_missing_evidence.csv](tables/construction_missing_evidence.csv) / [construction_missing_evidence_generations.csv](tables/construction_missing_evidence_generations.csv)|80份构造与1605条实际生成|
|位置与构造|[construction_checks.csv](tables/construction_checks.csv) / [construction_cross_block.csv](tables/construction_cross_block.csv)|实际源字符/token证据与保存元数据的差异|
|划分共享|[splits_overlap_documents.csv](tables/splits_overlap_documents.csv) / [splits_overlap_affected_ids.csv](tables/splits_overlap_affected_ids.csv)|共享篇目、支撑资料及受影响ID|
|补充背景|[construction_supplement_sources.csv](tables/construction_supplement_sources.csv)|供体区间、批准依据及分割|
|LITM入口|[scoring_litm_first_line_changed_IDs.csv](tables/scoring_litm_first_line_changed_IDs.csv) / [scoring_litm_first_line_minimal_evidence.csv](tables/scoring_litm_first_line_minimal_evidence.csv)|501条评分差异及最小原文|
|Doc2Dial条件|[scoring_doc2dial_history_condition_evidence.csv](tables/scoring_doc2dial_history_condition_evidence.csv)|连续agent和历史参考重复条件|
|输入范围|[runtime_unique_inputs.csv](tables/runtime_unique_inputs.csv) / [runtime_input_scope_summary.csv](tables/runtime_input_scope_summary.csv)|以唯一输入为分母的L/N与原篇数|
|生成与事件|[runtime_generation_limits.csv](tables/runtime_generation_limits.csv) / [runtime_workers.csv](tables/runtime_workers.csv) / [runtime_hot_pairs.csv](tables/runtime_hot_pairs.csv)|上限、退出和热请求重生成|
|评分一致性|[scoring_definition_consistency.csv](tables/scoring_definition_consistency.csv) / [scoring_doc2dial_corpus_consistency.csv](tables/scoring_doc2dial_corpus_consistency.csv)|逐条及语料充分统计核对|

CSV均为UTF-8，包含样本ID、来源、计数分母或单位；原文件行号为一基，frozen_index_zero_based为零基。不同异常类别可能覆盖同一构造，请分别使用各表分母。

## 3 复现

输入为仓库restored及manifest，输出仅此阶段目录。核心、评分和运行审核仅使用Python标准库；构造审核使用已有tokenizers0.20.3。本阶段确定性执行，无Bootstrap和随机重采样；生成实验seed=42，后续统计规划seed=20261116、5000次。

```powershell
Set-Location 'D:\_Greatest_programmer\_Projects\APEReport'
$env:PYTHONUTF8='1'
$env:PYTHONNOUSERSITE='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:TEMP=(Join-Path (Get-Location) 'pc-analysis-output\tmp')
$env:TMP=$env:TEMP
foreach ($part in @('core','splits','scoring','runtime')) {
    & 'C:\Users\Osheepne\anaconda3\python.exe' -s -B ('pc-analysis-output/scripts/audit_stage1_' + $part + '.py')
    if ($LASTEXITCODE -ne 0) { throw ('audit script failed: ' + $part) }
}
& 'C:\Users\Osheepne\anaconda3\python.exe' -s -B pc-analysis-output/scripts/summarize_stage1.py
```

构造审核保留已发现的问题，具体failed检查见splits_checks.csv；脚本执行成功与研究验收状态分别登记。首次定向排查记录保存在reports/，运行环境见environment.json。本次无需新增依赖。后续阶段由用户安排。
