# 实验第五阶段：注意力诊断材料

研究状态SUBMITTED；本阶段完成后停止。

- [分析报告](reports/ATTENTION_REPORT.md)
- [审核记录](reports/AUDIT.md)
- [六组答案与参数配对](tables/answer_pairs.csv)
- [完整证据区域与原保存标记](tables/evidence_regions.csv)
- [231条捕获审核](tables/capture_audit.csv)
- [36项步骤覆盖](tables/step_coverage.csv)
- [924条逐头区域权重](tables/head_steps.csv)
- [同输入请求末token比较](tables/query_region_pairs.csv)
- [解码历史对齐](tables/decode_alignment_summary.csv)
- [原答案案例](tables/selected_cases.csv)
- [诊断效率](tables/diagnostic_efficiency.csv)
- [生成停止](tables/generation_stops.csv)
- [模型加载](tables/model_load_workers.csv)
- [数据表字典](tables/table_dictionary.csv)
- [字段、公式与单位](tables/field_dictionary.csv)
- [环境和提交](environment.json)

figures/中包含6组逐头query图、6组实际decode图、1张证据比例总图和1张文档分支质量总图，均有PNG及SVG。图表数据来自query_head_bins.csv、physical_cache_bins.csv和query_region_pairs.csv；区域以逐token掩码精确求和，图像每箱64 token。sources/保留证据映射最小记录及实际参数。

从仓库根目录复现：

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONNOUSERSITE="1"
$env:PYTHONDONTWRITEBYTECODE="1"
& "C:/Users/Osheepne/anaconda3/python.exe" -s -B pc-analysis-output/scripts/analyze_attention_stage5.py
& "C:/Users/Osheepne/anaconda3/python.exe" -s -B pc-analysis-output/scripts/summarize_attention_stage5.py
```

本阶段无随机重采样，未重新评分、生成答案或安装依赖。Python/NumPy/Matplotlib/tokenizers均复用现有环境；缓存、临时文件与新增产物位于本阶段目录。各时间只对应观察器开启的诊断条件。
